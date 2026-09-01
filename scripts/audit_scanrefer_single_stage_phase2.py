#!/usr/bin/env python
"""Audit the immutable checkpoint and smoke gate for single-stage phase 2."""

from __future__ import print_function

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cache_scanrefer_rec_candidates import (
    _prepare_model_config,
    strip_module_prefix,
)
from scripts.v133_receipt_utils import atomic_write_new_json
from train_dist_mod import TrainTester


SOURCE_SCHEMA = "scanrefer-single-stage-phase2-source-audit-v1"
SMOKE_SCHEMA = "scanrefer-single-stage-phase2-smoke-gate-v1"
SOURCE_FILES = (
    "main_utils.py",
    "models/mcln.py",
    "models/mcln_training_groups.py",
    "scripts/audit_scanrefer_single_stage_phase2.py",
    "scripts/run_scanrefer_single_stage_phase2.sh",
    "train_dist_mod.py",
    "utils/lr_scheduler.py",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path):
    path = Path(path).expanduser().absolute()
    entry = os.lstat(str(path))
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("audit input must be a regular non-symlink file")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256(path),
        "size": int(entry.st_size),
        "mode": format(stat.S_IMODE(entry.st_mode), "04o"),
        "inode": int(entry.st_ino),
    }


def _config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _validate_source_config(config):
    dataset = _config_value(config, "dataset")
    if dataset not in (["scanrefer"], ("scanrefer",), "scanrefer"):
        raise ValueError("source checkpoint is not ScanRefer")
    expected = {
        "test_dataset": "scanrefer",
        "joint_det": True,
        "butd": False,
        "butd_gt": False,
        "butd_cls": False,
        "use_source_choice_selector": True,
        "source_choice_selector_sources": (
            "default,default_rank_blend_contrastive010"
        ),
        "source_choice_selector_default_source": "default",
        "rng_seed": 0,
    }
    mismatches = {
        name: {"expected": value, "actual": _config_value(config, name)}
        for name, value in expected.items()
        if _config_value(config, name) != value
    }
    if mismatches:
        raise ValueError("single-stage source config differs: {}".format(mismatches))


def _source_manifest(repo_root):
    return {
        relative: _snapshot(Path(repo_root) / relative)
        for relative in SOURCE_FILES
    }


def build_source_audit(args):
    checkpoint_snapshot = _snapshot(args.checkpoint)
    if checkpoint_snapshot["sha256"] != args.expected_sha256:
        raise ValueError("single-stage source checkpoint SHA-256 changed")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("epoch") != args.expected_epoch:
        raise ValueError("single-stage source checkpoint epoch changed")
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("single-stage source checkpoint has no config")
    _validate_source_config(config)

    runtime_config = _prepare_model_config(checkpoint, args.data_root)
    model = TrainTester.get_model(runtime_config)
    current = model.state_dict()
    saved = strip_module_prefix(checkpoint.get("model", {}))
    missing = sorted(set(current) - set(saved))
    unexpected = sorted(set(saved) - set(current))
    mismatched = sorted(
        name for name in set(current).intersection(saved)
        if current[name].shape != saved[name].shape
        or current[name].dtype != saved[name].dtype
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "single-stage source state is not exact: missing={}, "
            "unexpected={}, mismatched={}".format(
                missing, unexpected, mismatched
            )
        )
    model.load_state_dict(saved, strict=True)

    optimizer = checkpoint.get("optimizer")
    scheduler = checkpoint.get("scheduler")
    if not isinstance(optimizer, dict) or not isinstance(scheduler, dict):
        raise ValueError("single-stage source lacks optimizer/scheduler evidence")
    return {
        "schema": SOURCE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "checkpoint": checkpoint_snapshot,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_tensor_count": len(current),
        "strict_model_load": True,
        "source_world_size": 4,
        "source_batch_per_gpu": 18,
        "source_global_batch": 72,
        "phase2_world_size": 1,
        "phase2_batch_per_gpu": 18,
        "phase2_global_batch": 18,
        "optimizer_policy": (
            "fresh_optimizer; do_not_resume_four_gpu_optimizer_or_scheduler"
        ),
        "linear_lr_scale": 0.25,
        "source_optimizer_state_count": len(optimizer.get("state", {})),
        "source_optimizer_group_count": len(optimizer.get("param_groups", [])),
        "source_manifest": _source_manifest(Path(args.repo_root).resolve()),
    }


def verify_source_audit(path, args):
    snapshot = _snapshot(path)
    if snapshot["mode"] != "0444":
        raise ValueError("source audit receipt must have mode 0444")
    with Path(path).open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source audit schema changed")
    if receipt["checkpoint"]["sha256"] != args.expected_sha256:
        raise ValueError("source audit checkpoint binding changed")
    if _snapshot(args.checkpoint)["sha256"] != args.expected_sha256:
        raise ValueError("live source checkpoint changed")
    live_manifest = _source_manifest(Path(args.repo_root).resolve())
    if receipt.get("source_manifest") != live_manifest:
        raise ValueError("single-stage phase2 source changed after audit")
    return receipt


def _validate_smoke_metrics(metrics):
    if (
            metrics.get("schema") != "mcln-retrain-metrics-v1"
            or metrics.get("sample_count") != 128):
        raise ValueError("single-stage smoke metrics are incomplete")
    for family, fields in (
            (metrics["position"]["learned_selector"], ("hits025", "hits050")),
            (metrics["mask"], ("hits025", "hits050"))):
        for field in fields:
            value = family[field]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 128:
                raise ValueError("single-stage smoke hit count is invalid")
    miou = metrics["mask"]["miou"]
    if not isinstance(miou, (int, float)) or not math.isfinite(miou) or not 0 <= miou <= 1:
        raise ValueError("single-stage smoke mIoU is invalid")


def build_smoke_gate(args):
    source = verify_source_audit(args.source_audit, args)
    metrics_snapshot = _snapshot(args.smoke_metrics)
    with Path(args.smoke_metrics).open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    _validate_smoke_metrics(metrics)
    config_snapshot = _snapshot(args.smoke_config)
    with Path(args.smoke_config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "batch_size": 18,
        "debug": True,
        "debug_train_holdout": True,
        "joint_det": False,
        "butd": False,
        "max_epoch": 1,
        "checkpoint_metric_retention": True,
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items() if config.get(key) != value
    }
    if mismatches:
        raise ValueError("single-stage smoke config changed: {}".format(mismatches))
    return {
        "schema": SMOKE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "passed": True,
        "source_audit_sha256": _sha256(args.source_audit),
        "source_checkpoint_sha256": source["checkpoint"]["sha256"],
        "metrics": metrics_snapshot,
        "config": config_snapshot,
        "rec_hits025": metrics["position"]["learned_selector"]["hits025"],
        "rec_hits050": metrics["position"]["learned_selector"]["hits050"],
        "mask_hits025": metrics["mask"]["hits025"],
        "mask_hits050": metrics["mask"]["hits050"],
        "mask_miou": metrics["mask"]["miou"],
    }


def verify_smoke_gate(path, args):
    verify_source_audit(args.source_audit, args)
    snapshot = _snapshot(path)
    if snapshot["mode"] != "0444":
        raise ValueError("smoke gate must have mode 0444")
    with Path(path).open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("schema") != SMOKE_SCHEMA or receipt.get("passed") is not True:
        raise ValueError("single-stage smoke gate did not pass")
    if receipt.get("source_audit_sha256") != _sha256(args.source_audit):
        raise ValueError("smoke gate source audit binding changed")
    for name in ("metrics", "config"):
        record = receipt[name]
        if _snapshot(record["path"])["sha256"] != record["sha256"]:
            raise ValueError("smoke gate {} evidence changed".format(name))
    return receipt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True,
        choices=("source-build", "source-verify", "smoke-build", "smoke-verify"),
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-epoch", type=int, default=7)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-audit")
    parser.add_argument("--smoke-metrics")
    parser.add_argument("--smoke-config")
    parser.add_argument("--gate")
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "source-build":
        if not args.output:
            raise ValueError("source-build requires --output")
        payload = build_source_audit(args)
        output = atomic_write_new_json(payload, args.output)
    elif args.mode == "source-verify":
        if not args.source_audit:
            raise ValueError("source-verify requires --source-audit")
        payload = verify_source_audit(args.source_audit, args)
        output = Path(args.source_audit)
    elif args.mode == "smoke-build":
        required = (args.source_audit, args.smoke_metrics, args.smoke_config, args.output)
        if any(not value for value in required):
            raise ValueError("smoke-build requires source, metrics, config, output")
        payload = build_smoke_gate(args)
        output = atomic_write_new_json(payload, args.output)
    else:
        if not args.source_audit or not args.gate:
            raise ValueError("smoke-verify requires --source-audit and --gate")
        payload = verify_smoke_gate(args.gate, args)
        output = Path(args.gate)
    print(json.dumps({
        "mode": args.mode,
        "output": str(output),
        "sha256": _sha256(output),
        "schema": payload["schema"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
