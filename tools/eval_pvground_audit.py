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
    audits = []
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

        trainer.TrainTester.get_model = staticmethod(get_model)
        trainer.TrainTester(args).main(args)
        if len(audits) != 1:
            raise ValueError("expected exactly one audited model")
        stream.write(json.dumps({
            "kind": "forward_complete", "samples": audits[0].samples,
            "limitation": "Pair with full semantic evaluator counts and metrics.",
        }) + "\n")
        audits[0].close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
