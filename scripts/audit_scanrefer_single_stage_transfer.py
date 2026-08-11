#!/usr/bin/env python
"""Audit the exact two-stage-to-single-stage ScanRefer transfer boundary."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cache_scanrefer_rec_candidates import (
    _prepare_model_config,
    checkpoint_sha256,
    strip_module_prefix,
)
from train_dist_mod import TrainTester


DETECTED_STREAM_PATTERNS = (
    re.compile(r"^butd_class_embeddings\."),
    re.compile(r"^class_embeddings\."),
    re.compile(r"^box_embeddings\."),
    re.compile(r"^cross_encoder\.layers\.\d+\.cross_layer\.(cross_d|norm_d)\."),
    re.compile(r"^decoder\.\d+\.(cross_d|norm_d)\."),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify a protected checkpoint can initialize MCLN single-stage."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    return parser.parse_args(argv)


def is_detected_stream_tensor(name):
    return any(pattern.match(name) for pattern in DETECTED_STREAM_PATTERNS)


def audit_transfer(checkpoint_path, data_root):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    source_config = checkpoint.get("config")
    if source_config is None:
        raise ValueError("checkpoint has no config")
    if not bool(getattr(source_config, "butd", False)):
        raise ValueError("transfer source must use the detected-box stream")
    if bool(getattr(source_config, "butd_gt", False)) or bool(
            getattr(source_config, "butd_cls", False)):
        raise ValueError("GT/class detected-box transfer sources are forbidden")

    config = _prepare_model_config(checkpoint, str(data_root))
    config.butd = False
    config.butd_gt = False
    config.butd_cls = False
    model = TrainTester.get_model(config)
    target_state = model.state_dict()
    source_state = strip_module_prefix(checkpoint.get("model", {}))
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    mismatched = sorted(
        name for name in set(target_state).intersection(source_state)
        if target_state[name].shape != source_state[name].shape
    )
    invalid_unexpected = [
        name for name in unexpected if not is_detected_stream_tensor(name)
    ]
    if missing or mismatched or invalid_unexpected or not unexpected:
        raise ValueError(
            "unsafe single-stage transfer: missing={}, mismatched={}, "
            "invalid_unexpected={}, dropped={}".format(
                missing, mismatched, invalid_unexpected, len(unexpected)
            )
        )
    model.load_state_dict(
        {name: source_state[name] for name in target_state}, strict=True
    )
    return {
        "schema": "scanrefer-single-stage-transfer-audit-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(str(checkpoint_path)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "shared_tensor_count": len(target_state),
        "dropped_detected_stream_tensor_count": len(unexpected),
        "runtime_model_inputs": {
            "use_color": bool(config.use_color),
            "use_height": bool(config.use_height),
            "use_multiview": bool(config.use_multiview),
            "butd": False,
            "butd_gt": False,
            "butd_cls": False,
        },
    }


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(
        audit_transfer(args.checkpoint, args.data_root),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
