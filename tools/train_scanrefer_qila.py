"""Train the ScanRefer QILA-style IoU-aware ranking follow-up."""

import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


B2_SHA256 = "46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_checkpoint(path, epoch, expected_model_only_initialization=None):
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["epoch"] == epoch
    assert len(saved["model"]) == 1168
    assert saved["optimizer"]["state"] and saved["scheduler"]
    if expected_model_only_initialization is not None:
        assert (
            bool(saved["config"].model_only_initialization)
            is expected_model_only_initialization
        )
    groups = saved["optimizer"]["param_groups"]
    assert [(group["name"], group["lr"]) for group in groups] == [
        ("native", 2e-6), ("backbone", 4e-6), ("pvground", 2e-5),
    ]
    result = {
        "epoch": epoch,
        "checkpoint": str(path),
        "sha256": sha256(path),
        "model_tensors": len(saved["model"]),
        "optimizer_nonempty": bool(saved["optimizer"]["state"]),
        "scheduler_nonempty": bool(saved["scheduler"]),
        "optimizer_groups": [
            {"name": group["name"], "lr": group["lr"]} for group in groups
        ],
    }
    del saved
    return result


def build_optimizer(args, model):
    import torch

    groups = {"native": [], "backbone": [], "pvground": []}
    prefixes = (
        "decoder.5.local_visual.",
        "pvground_support_corrector.",
        "pvground_span_mixer.",
    )
    for name, parameter in model.named_parameters():
        name = name.removeprefix("module.")
        parameter.requires_grad_(not name.startswith("text_encoder."))
        if not parameter.requires_grad:
            continue
        group = (
            "pvground" if name.startswith(prefixes)
            else "backbone" if name.startswith("backbone_net.")
            else "native"
        )
        groups[group].append((name, parameter))
    rates = {
        "native": args.lr,
        "backbone": args.lr_backbone,
        "pvground": args.candidate_local_visual_lr,
    }
    assert all(groups.values())
    return torch.optim.AdamW([
        {
            "name": name,
            "parameter_names": [key for key, _ in values],
            "params": [parameter for _, parameter in values],
            "lr": rates[name],
        }
        for name, values in groups.items()
    ], weight_decay=args.weight_decay)


def semantic_row(evaluator, epoch):
    result = {}
    for threshold, suffix in ((0.25, "025"), (0.5, "050")):
        key = ("last_", threshold, 1, "bbf")
        assert evaluator.gts[key] == 9508
        result["hits" + suffix] = int(evaluator.dets[key])
    return dict(
        result,
        epoch=epoch,
        sample_count=9508,
        metric_branch="last_semantic_alignment_top1",
        hit_count_source="evaluator_integer_dets",
    )


def recover_semantic_rows(run_root, through_epoch):
    text = (run_root.parent / "train.log").read_text(errors="replace")
    matches = re.findall(
        r"last_ semantic alignment Acc(0\.25|0\.50): Top-1: ([0-9.]+)",
        text,
    )
    assert len(matches) >= through_epoch * 2
    rows = []
    for epoch in range(1, through_epoch + 1):
        pair = matches[(epoch - 1) * 2:epoch * 2]
        assert [threshold for threshold, _ in pair] == ["0.25", "0.50"]
        checkpoint_matches = list(
            run_root.rglob("ckpt_epoch_{}.pth".format(epoch))
        )
        assert len(checkpoint_matches) == 1
        checkpoint = checkpoint_matches[0]
        row = {
            "epoch": epoch,
            "sample_count": 9508,
            "metric_branch": "last_semantic_alignment_top1",
            "hit_count_source":
                "uniquely_reconstructed_from_rounded_full9508_log",
            "checkpoint": str(checkpoint),
        }
        for (threshold, rate), suffix in zip(pair, ("025", "050")):
            displayed = Decimal(rate)
            candidates = [
                hits for hits in range(9509)
                if abs(Decimal(hits) / Decimal(9508) - displayed)
                < Decimal("0.000005")
            ]
            assert len(candidates) == 1
            row["acc" + suffix] = str(displayed)
            row["hits" + suffix] = candidates[0]
        row.update(audit_checkpoint(checkpoint, epoch, epoch == 1))
        row["checkpoint"] = str(checkpoint)
        rows.append(row)
    return rows


def training_arguments(run_dir, checkpoint, resume):
    arguments = [
        "--local_rank", "0", "--num_decoder_layers", "6", "--use_color",
        "--weight_decay", "0.0005",
        "--data_root", "/root/autodl-tmp/PV-Ground/data/",
        "--batch_size", "8", "--gradient_accumulation_steps", "6",
        "--lr", "2e-6", "--lr_backbone", "4e-6", "--text_encoder_lr", "0",
        "--candidate_local_visual_lr", "2e-5",
        "--dataset", "scanrefer", "--test_dataset", "scanrefer",
        "--detect_intermediate", "--joint_det", "--use_soft_token_loss",
        "--use_contrastive_align", "--butd", "--self_attend", "--augment_det",
        "--model", "MCLN", "--rng_seed", "0",
        "--checkpoint_path", str(checkpoint), "--use_pvground_innovations",
        "--expected_eval_sample_count", "9508",
        "--bbf_quality_rank_loss_weight", "0.3",
        "--max_epoch", "8", "--lr_decay_epochs", "100",
        "--val_freq", "1", "--save_freq", "1", "--print_freq", "100",
        "--checkpoint_metric_retention",
        "--checkpoint_retention_metrics", "rec_acc025", "rec_acc050",
        "--log_dir", str(run_dir), "--exp", "scanrefer_qila_20261105",
    ]
    if not resume:
        arguments.extend(
            ["--model_only_initialization", "--checkpoint_start_epoch", "1"]
        )
    return arguments


def main():
    import torch
    import train_dist_mod as trainer
    import main_utils

    run_dir = Path(sys.argv[1])
    initialization = Path(sys.argv[2])
    resume = len(sys.argv) == 4 and sys.argv[3] == "resume"
    assert len(sys.argv) in (3, 4)
    assert run_dir.is_dir()
    history_path = run_dir / "qila_semantic_history.json"
    if resume:
        assert (run_dir / "started.json").is_file()
        assert not (run_dir / "complete.json").exists()
        resume_epoch = int(initialization.stem.removeprefix("ckpt_epoch_"))
        parent = audit_checkpoint(initialization, resume_epoch, False)
        recovered = recover_semantic_rows(run_dir, resume_epoch)
        history = recovered
        for epoch, row in enumerate(
            json.loads(history_path.read_text())["rows"]
            if history_path.exists() else [],
            start=1,
        ):
            assert row["epoch"] == epoch
            assert row["hits025"] == recovered[epoch - 1]["hits025"]
            assert row["hits050"] == recovered[epoch - 1]["hits050"]
        assert len(history) == resume_epoch
        main_utils._atomic_json_save(
            {
                "rows": history,
                "selection": "Acc050_then_Acc025_then_earlier_epoch",
                "early_stopping": False,
                "stop_requested": False,
                "final_target_met": False,
            },
            str(history_path),
        )
        main_utils._atomic_checkpoint_hardlink(
            history[0]["checkpoint"], str(run_dir / "semantic_best.pth")
        )
        main_utils._atomic_checkpoint_hardlink(
            history[0]["checkpoint"], str(run_dir / "semantic_latest.pth")
        )
        main_utils._atomic_json_save(
            history[0], str(run_dir / "semantic_best.json")
        )
    else:
        assert not (run_dir / "started.json").exists()
        assert sha256(initialization) == B2_SHA256
        parent = audit_checkpoint(initialization, 2, True)
        history = []

    sys.argv = [
        sys.argv[0], *training_arguments(run_dir, initialization, resume)
    ]
    args = trainer.prepare_source_moe_gate_checkpoint_config(
        trainer.parse_option()
    )
    assert args.max_epoch == 8
    assert args.model_only_initialization is (not resume)
    assert not args.pvground_train_only
    assert args.bbf_quality_rank_loss_weight == 0.3
    assert args.checkpoint_path == str(initialization)
    if resume:
        assert args.checkpoint_start_epoch is None
    else:
        assert args.checkpoint_start_epoch == 1
    assert args.pp_checkpoint is None
    assert args.expected_eval_sample_count == 9508

    class QILATrainer(trainer.TrainTester):
        get_optimizer = staticmethod(build_optimizer)

        def _build_grounding_evaluator(self, settings, prefixes):
            self.semantic_evaluator = super()._build_grounding_evaluator(
                settings, prefixes
            )
            return self.semantic_evaluator

        def should_stop_after_epoch(self, epoch, metrics, settings):
            run = Path(settings.log_dir)
            (run / "updates").mkdir(exist_ok=True)
            for source_name, output_name in (
                ("pvground_first_update.json", "first"),
                ("pvground_second_update.json", "second"),
            ):
                witness_path = run / source_name
                if not witness_path.exists() and output_name == "second":
                    continue
                witness = json.loads(witness_path.read_text(encoding="utf-8"))
                assert witness["epoch"] == epoch
                assert witness["finite_loss"]
                assert witness["optimizer_step_complete"]
                shutil.copyfile(
                    run / source_name,
                    run / "updates" / "epoch_{:02d}_{}.json".format(
                        epoch, output_name
                    ),
                )

            row = semantic_row(self.semantic_evaluator, epoch)
            source = run / "ckpt_epoch_{}.pth".format(epoch)
            audit = audit_checkpoint(source, epoch, False)
            preserved = run / "preserved_checkpoints"
            preserved.mkdir(exist_ok=True)
            target = preserved / source.name
            target.hardlink_to(source)
            row.update(audit)
            row["checkpoint"] = str(target)
            history.append(row)
            best = max(
                history,
                key=lambda item: (item["hits050"], item["hits025"], -item["epoch"]),
            )
            main_utils._atomic_checkpoint_hardlink(
                best["checkpoint"], str(run / "semantic_best.pth")
            )
            main_utils._atomic_json_save(
                best, str(run / "semantic_best.json")
            )
            main_utils._atomic_checkpoint_hardlink(
                best["checkpoint"], str(run_dir / "semantic_best.pth")
            )
            main_utils._atomic_json_save(
                best, str(run_dir / "semantic_best.json")
            )
            main_utils._atomic_checkpoint_hardlink(
                row["checkpoint"], str(run_dir / "semantic_latest.pth")
            )
            main_utils._atomic_json_save(
                {
                    "rows": history,
                    "selection": "Acc050_then_Acc025_then_earlier_epoch",
                    "early_stopping": False,
                    "stop_requested": False,
                    "final_target_met": any(
                        item["hits025"] >= 5705 and item["hits050"] >= 4945
                        for item in history
                    ),
                },
                str(history_path),
            )
            return False

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(seconds=5400),
    )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if resume:
        main_utils._atomic_json_save(
            {
                "resumed_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "checkpoint": parent,
                "optimizer_restored": True,
                "scheduler_restored": True,
            },
            str(run_dir / "resume_epoch{}.json".format(resume_epoch)),
        )
    else:
        main_utils._atomic_json_save(
            {
                "started_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "initialization": parent,
                "optimizer_restored": False,
                "scheduler_restored": False,
                "budget_epochs": 8,
                "early_stopping": False,
                "recipe": {
                    "seed": 0,
                    "batch_size": 8,
                    "gradient_accumulation_steps": 6,
                    "native_lr": 2e-6,
                    "backbone_lr": 4e-6,
                    "pvground_lr": 2e-5,
                    "qila_rank_loss_weight": 0.3,
                    "selection": "IoU-aware BBF candidate ranking",
                },
            },
            str(run_dir / "started.json"),
        )
    QILATrainer(args).main(args)
    main_utils._atomic_json_save(
        {
            "completed_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "complete_epochs": len(history),
            "actual_run_dir": args.log_dir,
            "budget_epochs": 8,
            "reason": "full_budget_completed",
            "final_target_met": any(
                item["hits025"] >= 5705 and item["hits050"] >= 4945
                for item in history
            ),
        },
        str(run_dir / "complete.json"),
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
