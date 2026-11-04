"""Fixed one-full-epoch pilot: common B2, weight 0 versus weight 0.1."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--condition", choices=("preflight", "control", "span"), required=True)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--span-only", action="store_true")
    options = parser.parse_args()
    assert options.local_rank == 0
    trial = options.trial
    condition = options.condition
    preparation = json.loads((trial / "preparation.json").read_text())
    checkpoint = Path(preparation["checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == preparation["checkpoint_sha256"]
    assert not (trial / f"{condition}_start.json").exists()

    import torch
    import main_utils
    import train_dist_mod as trainer
    from tools.train_scanrefer_stage_b import build_optimizer, semantic_row, training_arguments

    arguments = training_arguments(trial, {"initialization_checkpoint": str(checkpoint)})
    for key, value in (("--max_epoch", "1"), ("--log_dir", str(trial / condition)),
                       ("--exp", f"scanrefer_span_gate_{condition}_20261103")):
        arguments[arguments.index(key) + 1] = value
    weight = 0.0 if condition == "control" else 0.1
    arguments += ["--pvground_span_gate_loss_weight", str(weight)]
    if options.span_only:
        arguments += ["--pvground_span_train_only"]
    if condition == "preflight":
        arguments += ["--max_train_batches", "12" if options.span_only else "6"]
    sys.argv = [sys.argv[0]] + arguments
    args = trainer.prepare_source_moe_gate_checkpoint_config(trainer.parse_option())
    assert args.max_epoch == 1 and args.checkpoint_start_epoch == 1
    assert args.model_only_initialization and not args.pvground_train_only
    assert args.pp_checkpoint is None and args.expected_eval_sample_count == 9508

    class ComparisonTrainer(trainer.TrainTester):
        get_optimizer = staticmethod(
            main_utils.BaseTrainTester.get_optimizer if options.span_only else build_optimizer
        )

        def _build_grounding_evaluator(self, settings, prefixes):
            self.semantic_evaluator = super()._build_grounding_evaluator(settings, prefixes)
            return self.semantic_evaluator

        @staticmethod
        def _compute_loss(end_points, criterion, set_criterion, settings):
            loss, end_points = main_utils.BaseTrainTester._compute_loss(
                end_points, criterion, set_criterion, settings,
            )
            if condition == "preflight" and not (trial / "real_batch_gate_gradient.json").exists():
                raw = end_points["pvground_span_raw_axis_gate"]
                auxiliary = end_points["pvground_span_gate_loss"]
                gradient, = torch.autograd.grad(auxiliary, raw, retain_graph=True)
                assert torch.isfinite(loss) and torch.isfinite(auxiliary)
                assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                assert end_points["pvground_span_gate_valid_axes"] > 0
                main_utils._atomic_json_save({
                    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "loss": loss.item(), "auxiliary_loss": auxiliary.item(),
                    "weight": weight, "raw_gate_shape": list(raw.shape),
                    "valid_matched_axes": end_points["pvground_span_gate_valid_axes"].item(),
                    "raw_gate_gradient_l1": gradient.abs().sum().item(),
                    "outside_clamp_with_auxiliary_gradient": int(
                        (((raw < 0) | (raw > 1)) & (gradient != 0)).sum().item()),
                    "formal_validation_accessed": False,
                }, str(trial / "real_batch_gate_gradient.json"))
            return loss, end_points

        def should_stop_after_epoch(self, epoch, metrics, settings):
            row = semantic_row(self.semantic_evaluator, epoch)
            run = Path(settings.log_dir)
            saved_path = run / f"ckpt_epoch_{epoch}.pth"
            saved = torch.load(saved_path, map_location="cpu", weights_only=False)
            assert saved["epoch"] == 1 and len(saved["model"]) == 1168
            assert saved["optimizer"]["state"] and saved["scheduler"]
            assert saved["config"].pvground_span_gate_loss_weight == weight
            if options.span_only:
                initial = torch.load(checkpoint, map_location="cpu", weights_only=False)
                frozen = [key for key in initial["model"]
                          if not key.removeprefix("module.").startswith("pvground_span_mixer.")]
                assert all(torch.equal(initial["model"][key], saved["model"][key])
                           for key in frozen)
                span_names = [key for key in initial["model"]
                              if key.removeprefix("module.").startswith("pvground_span_mixer.")]
                assert len(span_names) == 14
                assert any(not torch.equal(initial["model"][key], saved["model"][key])
                           for key in span_names)
                main_utils._atomic_json_save({
                    "frozen_model_tensors": len(frozen),
                    "frozen_tensors_equal_initialization": True,
                    "span_tensors": len(span_names),
                    "span_changed": True,
                }, str(trial / f"{condition}_frozen_state.json"))
            digest = hashlib.sha256(saved_path.read_bytes()).hexdigest()
            preserved = run / "preserved_checkpoints"
            preserved.mkdir()
            target = preserved / saved_path.name
            target.hardlink_to(saved_path)
            row.update(
                condition=condition, gate_loss_weight=weight,
                checkpoint=str(target), checkpoint_sha256=digest,
                own_cpu_epoch=1, model_tensors=1168,
                nonempty_optimizer=True, has_scheduler=True,
                comparison_budget="one full epoch; exploratory single-seed pilot",
                span_only=options.span_only,
            )
            main_utils._atomic_checkpoint_hardlink(str(target), str(run / "semantic_best.pth"))
            main_utils._atomic_json_save(row, str(run / "semantic_best.json"))
            main_utils._atomic_json_save(row, str(trial / f"{condition}_result.json"))
            return False

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
        timeout=datetime.timedelta(seconds=5400),
    )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    main_utils._atomic_json_save({
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "condition": condition, "checkpoint_sha256": preparation["checkpoint_sha256"],
        "optimizer_restored": False, "scheduler_restored": False,
        "gate_loss_weight": weight, "seed": 0, "max_epoch": 1,
        "span_only": options.span_only,
        "train_budget": ("12 batches diagnostic" if options.span_only else "6 batches diagnostic")
        if condition == "preflight" else "full48655 augmented entries",
    }, str(trial / f"{condition}_start.json"))
    ComparisonTrainer(args).main(args)
    main_utils._atomic_json_save({
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "condition": condition, "run_dir": args.log_dir,
        "full_validation_expected": condition != "preflight",
    }, str(trial / f"{condition}_complete.json"))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
