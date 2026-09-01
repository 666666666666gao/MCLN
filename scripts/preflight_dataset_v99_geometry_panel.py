#!/usr/bin/env python
"""Fail early unless a dataset cache can supply the fixed V99 audit panel."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.audit_scanrefer_mask_geometry import (
    PANEL_BUCKETS,
    load_train_cache_panel_records,
    select_baseline_stratified_panel,
)
from scripts.cache_scanrefer_rec_candidates import checkpoint_sha256


SCHEMA = "mcln-dataset-v99-geometry-panel-preflight-v1"


def _canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _eligible_scene_count(records, expressions_per_scene):
    grouped = {}
    for record in records:
        grouped.setdefault(str(record["scan_id"]), []).append(record)
    return sum(
        1 for rows in grouped.values()
        if len(rows) >= expressions_per_scene
        and set(PANEL_BUCKETS).issubset({row["bucket"] for row in rows})
    )


def _publish_or_validate(path, payload):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing geometry-panel preflight changed")
        return
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True
    ).encode("ascii") + b"\n"
    descriptor = os.open(
        str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    try:
        offset = 0
        while offset < len(encoded):
            count = os.write(descriptor, encoded[offset:])
            if count <= 0:
                raise OSError("preflight receipt write made no progress")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("scanrefer", "nr3d", "sr3d"), required=True
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--scene-count", type=int, default=64)
    parser.add_argument("--expressions-per-scene", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.scene_count <= 0 or args.expressions_per_scene < len(PANEL_BUCKETS):
        parser.error("invalid fixed-panel dimensions")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    train_cache = Path(args.train_cache).expanduser().resolve()
    fingerprint = checkpoint_sha256(checkpoint)
    manifest, records = load_train_cache_panel_records(
        train_cache, expected_checkpoint_sha256=fingerprint
    )
    if str(manifest.get("dataset", "scanrefer")) != args.dataset:
        raise ValueError("train cache dataset does not match preflight dataset")
    eligible = _eligible_scene_count(records, args.expressions_per_scene)
    panel = select_baseline_stratified_panel(
        records,
        scene_count=args.scene_count,
        expressions_per_scene=args.expressions_per_scene,
        seed=args.selection_seed,
    )
    payload = {
        "schema": SCHEMA,
        "passed": True,
        "dataset": args.dataset,
        "dataset_only": True,
        "checkpoint_sha256": fingerprint,
        "train_cache_manifest_sha256": _canonical_sha256(manifest),
        "required_scene_count": args.scene_count,
        "eligible_scene_count": eligible,
        "expressions_per_scene": args.expressions_per_scene,
        "selected_sample_count": len(panel),
        "selection_seed": args.selection_seed,
        "selected_identity_sha256": _canonical_sha256([
            [row["dataset_index"], row["scan_id"], row["bucket"]]
            for row in panel
        ]),
    }
    _publish_or_validate(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
