"""Conditional ScanRefer joint fine-tuning after three completed Stage A epochs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


MODULE_PREFIXES = (
    "decoder.5.local_visual.", "pvground_support_corrector.", "pvground_span_mixer.",
)


def build_optimizer(args, model):
    import torch

    groups = {name: [] for name in ("native", "backbone", "pvground")}
    for name, parameter in model.named_parameters():
        name = name.removeprefix("module.")
        parameter.requires_grad_(not name.startswith("text_encoder."))
        if not parameter.requires_grad:
            continue
        group = (
            "pvground" if name.startswith(MODULE_PREFIXES)
            else "backbone" if name.startswith("backbone_net.") else "native"
        )
        groups[group].append((name, parameter))
    rates = {"native": args.lr, "backbone": args.lr_backbone,
             "pvground": args.candidate_local_visual_lr}
    assert all(groups.values())
    return torch.optim.AdamW([
        {"name": name, "parameter_names": [key for key, _ in values],
         "params": [parameter for _, parameter in values], "lr": rates[name]}
        for name, values in groups.items()
    ], weight_decay=args.weight_decay)


class ValidationHistory:
    """Count decreases between consecutive complete Stage B validations."""

    def __init__(self):
        self.rows = []
        self.decrease_streak = 0

    def observe(self, row):
        assert row["sample_count"] == 9508
        assert row["epoch"] == len(self.rows) + 1
        assert 0 <= row["hits050"] <= row["hits025"] <= 9508
        if self.rows and row["hits050"] < self.rows[-1]["hits050"]:
            self.decrease_streak += 1
        else:
            self.decrease_streak = 0
        self.rows.append(row)
        return self.decrease_streak >= 2


def semantic_row(evaluator, epoch):
    hits = {}
    for threshold, suffix in ((0.25, "025"), (0.5, "050")):
        key = ("last_", threshold, 1, "bbf")
        assert evaluator.gts[key] == 9508
        hits["hits" + suffix] = int(evaluator.dets[key])
    return dict(hits, epoch=epoch, sample_count=9508,
                metric_branch="last_semantic_alignment_top1")


def training_arguments(stage, decision):
    return [
        "--local_rank", "0", "--num_decoder_layers", "6", "--use_color",
        "--weight_decay", "0.0005", "--data_root", "/root/autodl-tmp/PV-Ground/data/",
        "--batch_size", "8", "--gradient_accumulation_steps", "6",
        "--lr", "2e-6", "--lr_backbone", "4e-6", "--text_encoder_lr", "0",
        "--candidate_local_visual_lr", "2e-5", "--dataset", "scanrefer",
        "--test_dataset", "scanrefer", "--detect_intermediate", "--joint_det",
        "--use_soft_token_loss", "--use_contrastive_align", "--butd",
        "--self_attend", "--augment_det", "--model", "MCLN", "--rng_seed", "0",
        "--checkpoint_path", decision["initialization_checkpoint"],
        "--model_only_initialization", "--checkpoint_start_epoch", "1",
        "--use_pvground_innovations", "--expected_eval_sample_count", "9508",
        "--max_epoch", "10", "--lr_decay_epochs", "100",
        "--val_freq", "1", "--save_freq", "1", "--print_freq", "100",
        "--checkpoint_metric_retention",
        "--checkpoint_retention_metrics", "rec_acc025", "rec_acc050",
        "--log_dir", str(stage / "stage_b"), "--exp", "scanrefer_stage_b_20261102",
    ]


def require_matched_validation(stage, decision):
    audit = stage / "protocol_audit"
    assert int((audit / "pipeline_exit_code").read_text()) == 0
    receipt = json.loads((audit / "matched_validation.json").read_text())
    assert receipt["selected_epoch"] == decision["selected"]["epoch"]
    selected = receipt["selected"]
    assert selected["samples"] == 9508
    assert selected["checkpoint_sha256"] == decision["initialization_sha256"]
    assert selected["initialization"]["loaded_tensors"] == 1168
    assert selected["initialization"]["checkpoint_epoch"] == receipt["selected_epoch"]
    assert selected["backend"] == {
        "cudnn_enabled": True, "cudnn_benchmark": True, "cudnn_deterministic": True,
    }
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--local-rank", type=int, default=0)
    options = parser.parse_args()
    assert options.local_rank == 0
    stage = options.stage
    decision = json.loads((stage / "stage_a_complete_epochs.json").read_text())[
        "stage_b_decision"
    ]
    assert decision["status"] == "ready"
    matched = require_matched_validation(stage, decision)
    checkpoint = Path(decision["initialization_checkpoint"])
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == decision["initialization_sha256"]

    import datetime
    import torch
    import train_dist_mod as trainer
    import main_utils

    assert hasattr(main_utils.BaseTrainTester, "should_stop_after_epoch")
    sys.argv = [sys.argv[0]] + training_arguments(stage, decision)
    args = trainer.prepare_source_moe_gate_checkpoint_config(trainer.parse_option())
    assert not args.pvground_train_only and not args.eval
    history = ValidationHistory()

    class StageBTrainer(trainer.TrainTester):
        get_optimizer = staticmethod(build_optimizer)

        def _build_grounding_evaluator(self, settings, prefixes):
            self.semantic_evaluator = super()._build_grounding_evaluator(settings, prefixes)
            return self.semantic_evaluator

        def should_stop_after_epoch(self, epoch, metrics, settings):
            row = semantic_row(self.semantic_evaluator, epoch)
            run_dir = Path(settings.log_dir)
            source = run_dir / "ckpt_epoch_{}.pth".format(epoch)
            saved = torch.load(source, map_location="cpu", weights_only=False)
            assert saved["epoch"] == epoch and len(saved["model"]) == 1168
            assert saved["config"].model_only_initialization
            assert not saved["config"].pvground_train_only
            groups = saved["optimizer"]["param_groups"]
            assert [(group["name"], group["lr"]) for group in groups] == [
                ("native", 2e-6), ("backbone", 4e-6), ("pvground", 2e-5),
            ]
            preserved = run_dir / "preserved_checkpoints"
            preserved.mkdir(exist_ok=True)
            target = preserved / source.name
            target.hardlink_to(source)
            row.update(checkpoint=str(target), checkpoint_epoch=epoch, model_tensors=1168)
            stop = history.observe(row)
            best = max(history.rows, key=lambda item: (
                item["hits050"], item["hits025"], -item["epoch"],
            ))
            main_utils._atomic_checkpoint_hardlink(
                best["checkpoint"], str(run_dir / "semantic_best.pth"),
            )
            main_utils._atomic_json_save(best, str(run_dir / "semantic_best.json"))
            main_utils._atomic_json_save({
                "rows": history.rows, "decrease_streak": history.decrease_streak,
                "comparison_baseline": "previous_complete_stage_b_epoch",
                "stop_requested": stop,
                "final_target_met": any(
                    item["hits025"] >= 5705 and item["hits050"] >= 4945
                    for item in history.rows
                ),
            }, str(run_dir / "stage_b_semantic_history.json"))
            return stop

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
        timeout=datetime.timedelta(seconds=5400),
    )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    main_utils._atomic_json_save({
        "phase": "stage_b", "parent_decision": decision,
        "matched_validation": matched,
        "optimizer_restored": False, "scheduler_restored": False,
        "comparison_baseline": "previous_complete_stage_b_epoch",
    }, str(stage / "stage_b_start.json"))
    runner = StageBTrainer(args)
    runner.main(args)
    main_utils._atomic_json_save({
        "phase": "stage_b_complete",
        "reason": "two_consecutive_acc050_decreases" if history.decrease_streak >= 2
                  else "max_epochs",
        "complete_epochs": len(history.rows),
        "run_dir": args.log_dir,
    }, str(stage / "stage_b_complete.json"))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
