#!/usr/bin/env python3
"""Validate a complete evaluation receipt and its final checkpoint."""

import argparse
import json
from pathlib import Path

import torch

try:
    from scripts.audit_source_moe_candidate_oracle import metrics_from_receipt
except ModuleNotFoundError:
    from audit_source_moe_candidate_oracle import metrics_from_receipt


AUDIT_SCHEMA = "mcln-training-completion-audit-v1"


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_training_completion(
        metrics_receipt, checkpoint_path, expected_epoch,
        expected_sample_count=9508, require_position_subgroups=False):
    if (not isinstance(expected_epoch, int) or isinstance(expected_epoch, bool)
            or expected_epoch <= 0):
        raise ValueError("expected epoch must be a positive integer")
    if (not isinstance(expected_sample_count, int)
            or isinstance(expected_sample_count, bool)
            or expected_sample_count <= 0):
        raise ValueError("expected sample count must be a positive integer")

    metrics = metrics_from_receipt(
        metrics_receipt,
        expected_sample_count=expected_sample_count,
        require_position_subgroups=require_position_subgroups,
    )
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError("final checkpoint does not exist")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("final checkpoint must be a dictionary")
    checkpoint_epoch = checkpoint.get("epoch")
    if (not isinstance(checkpoint_epoch, int)
            or isinstance(checkpoint_epoch, bool)
            or checkpoint_epoch != expected_epoch):
        raise ValueError(
            "final checkpoint epoch {} does not match expected {}".format(
                checkpoint_epoch, expected_epoch
            )
        )
    return {
        "schema": AUDIT_SCHEMA,
        "passed": True,
        "expected_epoch": expected_epoch,
        "checkpoint_epoch": checkpoint_epoch,
        "metrics": metrics,
    }


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-sample-count", type=int, default=9508)
    parser.add_argument("--require-position-subgroups", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    result = audit_training_completion(
        _load_json(args.metrics),
        args.checkpoint,
        expected_epoch=args.expected_epoch,
        expected_sample_count=args.expected_sample_count,
        require_position_subgroups=args.require_position_subgroups,
    )
    if args.output:
        _write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
