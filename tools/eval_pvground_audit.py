"""Single-GPU evaluation entry point; accepts the normal MCLN eval flags."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools.pvground_forward_audit import PVGroundForwardAudit
from tools.pvground_semantic_audit import PVGroundSemanticAudit


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument(
        "--audit-bypass", choices=("none", "reader", "support", "span"),
        default="none",
    )
    audit_args, model_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + model_args
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import train_dist_mod as trainer

    args = trainer.prepare_source_moe_gate_checkpoint_config(
        trainer.parse_option()
    )
    if not args.eval or not args.use_pvground_innovations:
        raise ValueError("audit requires --eval --use_pvground_innovations")
    if args.test_dataset != "scanrefer" or args.butd_cls:
        raise ValueError("semantic audit requires current ScanRefer without detector filtering")
    if args.model_only_initialization:
        raise ValueError("audit must load the complete trained checkpoint")
    if int(os.environ["WORLD_SIZE"]) != 1:
        raise ValueError("audit requires a single evaluation rank")
    checkpoint = Path(args.checkpoint_path)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
        timeout=datetime.timedelta(seconds=5400),
    )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    original_get_model = trainer.TrainTester.get_model
    original_build_evaluator = trainer.TrainTester._build_grounding_evaluator
    audits = []
    semantic_audits = []
    evaluators = []
    with Path(audit_args.audit_jsonl).open("w") as stream:
        stream.write(json.dumps({
            "kind": "metadata", "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest.hexdigest(),
            "bypass": audit_args.audit_bypass,
            "argv": model_args,
        }) + "\n")

        def get_model(options):
            model = original_get_model(options)
            audits.append(PVGroundForwardAudit(
                model, stream, audit_args.audit_bypass
            ))
            return model

        def build_evaluator(instance, options, prefixes):
            from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz

            evaluator = original_build_evaluator(instance, options, prefixes)
            audit = PVGroundSemanticAudit(
                stream, _iou3d_par, box_cxcyczwhd_to_xyzxyz
            )
            original_semantic = evaluator.evaluate_bbox_by_sem_align

            def evaluate_semantic(end_points, prefix):
                result = original_semantic(end_points, prefix)
                if prefix == "last_":
                    audit.update(end_points, audits[0].last_gates)
                return result

            evaluator.evaluate_bbox_by_sem_align = evaluate_semantic
            semantic_audits.append(audit)
            evaluators.append(evaluator)
            return evaluator

        trainer.TrainTester.get_model = staticmethod(get_model)
        trainer.TrainTester._build_grounding_evaluator = build_evaluator
        trainer.TrainTester(args).main(args)
        if len(audits) != 1 or len(semantic_audits) != 1:
            raise ValueError("expected one audited model and evaluator")
        semantic = semantic_audits[0]
        semantic.verify(evaluators[0])
        if semantic.samples != 9508 or audits[0].samples != 9508:
            raise ValueError("ScanRefer audit requires all 9508 validation examples")
        stream.write(json.dumps({
            "kind": "forward_complete", "samples": audits[0].samples,
            "semantic_samples": semantic.samples,
            "semantic_hits025": semantic.hits[0.25],
            "semantic_hits050": semantic.hits[0.5],
            "verified_against_evaluator": True,
        }) + "\n")
        audits[0].close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
