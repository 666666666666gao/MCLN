"""One-epoch B2-initialized Reader competitor-context experiment."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--condition", choices=("preflight", "reader"), required=True)
    parser.add_argument("--local-rank", type=int, default=0)
    options = parser.parse_args()
    trial = options.trial
    checkpoint = Path(json.loads((trial / "preparation.json").read_text())["checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        "46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8"
    )
    import torch
    import main_utils
    import train_dist_mod as trainer
    from tools.train_scanrefer_stage_b import build_optimizer, semantic_row, training_arguments

    arguments = training_arguments(trial, {"initialization_checkpoint": str(checkpoint)})
    for key, value in (("--max_epoch", "1"), ("--log_dir", str(trial / options.condition)),
                       ("--exp", "scanrefer_reader_competitor_" + options.condition)):
        arguments[arguments.index(key) + 1] = value
    if options.condition == "preflight":
        arguments += ["--max_train_batches", "6"]
    sys.argv = [sys.argv[0]] + arguments
    args = trainer.prepare_source_moe_gate_checkpoint_config(trainer.parse_option())
    assert args.model_only_initialization and args.expected_eval_sample_count == 9508

    class ReaderTrainer(trainer.TrainTester):
        get_optimizer = staticmethod(build_optimizer)

        def _build_grounding_evaluator(self, settings, prefixes):
            self.semantic_evaluator = super()._build_grounding_evaluator(settings, prefixes)
            return self.semantic_evaluator

        def should_stop_after_epoch(self, epoch, metrics, settings):
            row = semantic_row(self.semantic_evaluator, epoch)
            run = Path(settings.log_dir)
            saved_path = run / f"ckpt_epoch_{epoch}.pth"
            saved = torch.load(saved_path, map_location="cpu", weights_only=False)
            assert saved["epoch"] == 1 and len(saved["model"]) == 1169
            assert saved["optimizer"]["state"] and saved["scheduler"]
            scale = saved["model"]["module.decoder.5.local_visual.competitor_scale"].item()
            preserved = run / "preserved_checkpoints"
            preserved.mkdir()
            target = preserved / saved_path.name
            target.hardlink_to(saved_path)
            row.update(checkpoint=str(target), checkpoint_sha256=hashlib.sha256(
                saved_path.read_bytes()).hexdigest(), competitor_scale=scale,
                own_cpu_epoch=1, model_tensors=1169, nonempty_optimizer=True,
                has_scheduler=True, condition=options.condition)
            main_utils._atomic_checkpoint_hardlink(str(target), str(run / "semantic_best.pth"))
            main_utils._atomic_json_save(row, str(run / "semantic_best.json"))
            main_utils._atomic_json_save(row, str(trial / "reader_result.json"))
            return False

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=5400))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    main_utils._atomic_json_save({
        "condition": options.condition, "checkpoint": str(checkpoint),
        "optimizer_restored": False, "scheduler_restored": False,
        "budget": "six batches" if options.condition == "preflight" else "one full epoch",
    }, str(trial / (options.condition + "_start.json")))
    ReaderTrainer(args).main(args)
    main_utils._atomic_json_save({"run_dir": args.log_dir}, str(
        trial / (options.condition + "_complete.json")))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
