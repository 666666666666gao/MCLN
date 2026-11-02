"""Matched official-checkpoint evaluation before ScanRefer Stage A."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--e0-receipt", required=True)
    options, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    import train_dist_mod as trainer
    import main_utils

    args = trainer.prepare_source_moe_gate_checkpoint_config(trainer.parse_option())
    assert args.eval and args.model_only_initialization
    initialization = {}

    def initialize_for_evaluation(settings, model, optimizer, scheduler, **kwargs):
        saved = torch.load(settings.checkpoint_path, map_location="cpu")
        assert saved["epoch"] == 54
        native = {name.removeprefix("module."): value
                  for name, value in saved["model"].items()
                  if not name.endswith("embeddings.position_ids")}
        unwrapped = model.module
        current = unwrapped.state_dict()
        new_prefixes = ("decoder.5.local_visual.",
                        "pvground_support_corrector.", "pvground_span_mixer.")
        missing = set(current) - set(native)
        assert not (set(native) - set(current))
        assert all(name.startswith(new_prefixes) for name in missing)
        assert len(native) == 1134
        for name, value in native.items():
            assert current[name].shape == value.shape and current[name].dtype == value.dtype
        current.update(native)
        unwrapped.load_state_dict(current, strict=True)
        initialization.update(native_tensors=len(native), new_tensors=len(missing),
                              checkpoint_epoch=54, optimizer_restored=False)
        settings.start_epoch = 0

    main_utils.load_checkpoint = initialize_for_evaluation
    trainer.TrainTester.get_optimizer = staticmethod(lambda settings, model: None)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
        timeout=datetime.timedelta(seconds=5400),
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    original = trainer.TrainTester._build_grounding_evaluator
    evaluators = []
    points = hashlib.sha256()
    identities = hashlib.sha256()
    samples = 0

    def build(instance, settings, prefixes):
        evaluator = original(instance, settings, prefixes)
        evaluate = evaluator.evaluate_bbox_by_sem_align

        def semantic(end_points, prefix):
            nonlocal samples
            result = evaluate(end_points, prefix)
            if prefix == "last_":
                batch = end_points["point_clouds"].detach().cpu().contiguous()
                points.update(batch.numpy().tobytes())
                for index, scene in enumerate(end_points["scan_ids"]):
                    identities.update(json.dumps([
                        scene, int(end_points["target_id"][index]),
                        end_points["utterances"][index],
                    ], ensure_ascii=True).encode())
                samples += len(batch)
            return result

        evaluator.evaluate_bbox_by_sem_align = semantic
        evaluators.append(evaluator)
        return evaluator

    trainer.TrainTester._build_grounding_evaluator = build
    trainer.TrainTester(args).main(args)
    assert len(evaluators) == 1 and samples == 9508
    evaluator = evaluators[0]
    hits = {}
    for threshold in (0.25, 0.5):
        key = ("last_", threshold, 1, "bbf")
        assert evaluator.gts[key] == 9508
        hits[str(threshold)] = int(evaluator.dets[key])
    receipt = {
        "samples": samples, "hits": hits,
        "point_sha256": points.hexdigest(),
        "identity_sha256": identities.hexdigest(),
        "innovations": args.use_pvground_innovations,
        "seed": args.rng_seed,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint_path).read_bytes()).hexdigest(),
        "argv": rest,
        "initialization": initialization,
    }
    Path(options.e0_receipt).write_text(json.dumps(receipt, indent=2) + "\n")
    print("E0_COMPLETE", json.dumps(receipt))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
