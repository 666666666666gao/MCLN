#!/usr/bin/env python
"""Fail-closed A/B audit for the V108 mesh-superpoint train cache."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


ALLOWED_FEATURES = (
    "mask_confidence",
    "mask_foreground_ratio",
    "mask_text_query_dice",
)
EXPECTED_COUNTS = {
    "sample_count": 36665,
    "default_hits025": 34892,
    "default_hits050": 31870,
    "oracle_hits025": 36405,
    "oracle_hits050": 35409,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_manifest(cache):
    path = Path(cache).resolve() / "manifest.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("candidate manifest must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def update_counts(counts, row):
    valid = row["valid_mask"].bool()
    ious = row["candidate_ious"].float()
    queries = row["query_indices"].long()
    default_query = int(row["default_top1_query_index"])
    default_valid = valid & queries.eq(default_query)
    if not bool(default_valid.any().item()):
        raise RuntimeError("row has no valid default candidate")
    default_iou = float(ious.masked_fill(~default_valid, -1.0).max().item())
    oracle_iou = float(ious.masked_fill(~valid, -1.0).max().item())
    counts["sample_count"] += 1
    counts["default_hits025"] += int(default_iou >= 0.25)
    counts["default_hits050"] += int(default_iou >= 0.50)
    counts["oracle_hits025"] += int(oracle_iou >= 0.25)
    counts["oracle_hits050"] += int(oracle_iou >= 0.50)


def audit(old_cache, new_cache, missing_scenes_path):
    old_cache = Path(old_cache).resolve()
    new_cache = Path(new_cache).resolve()
    old_manifest, old_manifest_sha = load_manifest(old_cache)
    new_manifest, new_manifest_sha = load_manifest(new_cache)
    expected_manifest_fields = (
        "cache_schema_version", "feature_schema_version", "feature_dim",
        "feature_names", "split", "dataset_size", "source_dataset_size",
        "sample_count", "candidate_rule", "checkpoint_epoch",
        "checkpoint_sha256", "target_iou_policy", "model_inputs",
        "backbone_config", "deterministic",
    )
    for key in expected_manifest_fields:
        if old_manifest.get(key) != new_manifest.get(key):
            raise RuntimeError(f"candidate manifest field changed: {key}")
    if old_manifest.get("sample_count") != 36665:
        raise RuntimeError("candidate sample count is not 36665")
    old_shards = old_manifest.get("shards")
    new_shards = new_manifest.get("shards")
    if old_shards != new_shards or len(old_shards) != 144:
        raise RuntimeError("candidate shard sequence changed")
    feature_names = tuple(old_manifest.get("feature_names", ()))
    allowed_indices = tuple(feature_names.index(name) for name in ALLOWED_FEATURES)
    allowed_set = set(allowed_indices)
    invariant_indices = tuple(
        index for index in range(len(feature_names)) if index not in allowed_set
    )
    missing_path = Path(missing_scenes_path).resolve()
    missing_raw = missing_path.read_bytes()
    missing_scenes = {
        line.strip() for line in missing_raw.decode("utf-8").splitlines()
        if line.strip()
    }
    if len(missing_scenes) != 789:
        raise RuntimeError("fallback scene manifest is not exactly 789 scenes")
    if hashlib.sha256(missing_raw).hexdigest() != (
            "caf63109bdf9f19cd8132b3c70eb1f2467d70fc605d174c6ec801b34c1c31079"):
        raise RuntimeError("fallback scene manifest identity changed")

    invariant_keys = (
        "dataset_index", "scan_id", "target_id",
        "default_top1_query_index", "boxes", "query_indices", "valid_mask",
        "default_scores", "contrastive_scores", "candidate_ious",
    )
    counts = {key: 0 for key in EXPECTED_COUNTS}
    fallback_rows = 0
    regular_rows = 0
    changed_rows = 0
    changed_scenes = set()
    feature_changed_rows = {name: 0 for name in ALLOWED_FEATURES}
    feature_max_abs = {name: 0.0 for name in ALLOWED_FEATURES}
    seen_scenes = set()
    expected_index = 0
    for shard_name in old_shards:
        old_payload = torch.load(old_cache / shard_name, map_location="cpu")
        new_payload = torch.load(new_cache / shard_name, map_location="cpu")
        old_rows = old_payload.get("rows") if isinstance(old_payload, dict) else None
        new_rows = new_payload.get("rows") if isinstance(new_payload, dict) else None
        if not isinstance(old_rows, list) or not isinstance(new_rows, list):
            raise RuntimeError(f"candidate rows missing in {shard_name}")
        if len(old_rows) != len(new_rows):
            raise RuntimeError(f"candidate row count changed in {shard_name}")
        for old_row, new_row in zip(old_rows, new_rows):
            if set(old_row) != set(new_row):
                raise RuntimeError("candidate row keys changed")
            if old_row.get("dataset_index") != expected_index:
                raise RuntimeError("old candidate row order changed")
            if new_row.get("dataset_index") != expected_index:
                raise RuntimeError("new candidate row order changed")
            expected_index += 1
            for key in invariant_keys:
                old_value = old_row[key]
                new_value = new_row[key]
                if isinstance(old_value, torch.Tensor):
                    if not isinstance(new_value, torch.Tensor) or not torch.equal(
                            old_value, new_value):
                        raise RuntimeError(
                            f"candidate invariant tensor changed: {key} at {expected_index - 1}"
                        )
                elif old_value != new_value:
                    raise RuntimeError(
                        f"candidate invariant value changed: {key} at {expected_index - 1}"
                    )
            old_features = old_row["features"]
            new_features = new_row["features"]
            if not torch.equal(
                    old_features[:, invariant_indices],
                    new_features[:, invariant_indices]):
                raise RuntimeError(
                    f"non-mask candidate feature changed at {expected_index - 1}"
                )
            scan_id = old_row["scan_id"]
            seen_scenes.add(scan_id)
            is_fallback = scan_id in missing_scenes
            if is_fallback:
                fallback_rows += 1
            else:
                regular_rows += 1
                if not torch.equal(old_features, new_features):
                    raise RuntimeError(
                        f"regular-scene candidate features changed: {scan_id}"
                    )
            row_changed = False
            for name, index in zip(ALLOWED_FEATURES, allowed_indices):
                delta = (new_features[:, index] - old_features[:, index]).abs()
                if bool(delta.ne(0).any().item()):
                    feature_changed_rows[name] += 1
                    feature_max_abs[name] = max(
                        feature_max_abs[name], float(delta.max().item())
                    )
                    row_changed = True
            if row_changed:
                if not is_fallback:
                    raise RuntimeError("feature change escaped fallback scene set")
                changed_rows += 1
                changed_scenes.add(scan_id)
            update_counts(counts, new_row)
    if expected_index != 36665:
        raise RuntimeError("candidate A/B did not consume 36665 rows")
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"candidate metric counts changed: {counts}")
    if not changed_rows or not changed_scenes:
        raise RuntimeError("mesh-derived superpoints changed no mask features")
    cache_fallback_scenes = seen_scenes & missing_scenes
    cache_regular_scenes = seen_scenes - missing_scenes
    if len(cache_fallback_scenes) != 361 or len(cache_regular_scenes) != 201:
        raise RuntimeError("ScanRefer train fallback/regular scene partition changed")
    return {
        "schema": "mcln-v108-meshsp-candidate-ab-audit-v1",
        "version": 1,
        "validation_data_accessed": False,
        "old_cache": str(old_cache),
        "new_cache": str(new_cache),
        "old_manifest_sha256": old_manifest_sha,
        "new_manifest_sha256": new_manifest_sha,
        "fallback_manifest_sha256": hashlib.sha256(missing_raw).hexdigest(),
        "sample_count": expected_index,
        "scene_count": len(seen_scenes),
        "fallback_scene_count_total_scannet": len(missing_scenes),
        "fallback_scene_count_in_scanrefer": len(cache_fallback_scenes),
        "regular_scene_count_in_scanrefer": len(cache_regular_scenes),
        "fallback_row_count": fallback_rows,
        "regular_row_count": regular_rows,
        "changed_row_count": changed_rows,
        "changed_scene_count": len(changed_scenes),
        "changed_scenes_subset_of_fallback": changed_scenes.issubset(missing_scenes),
        "allowed_feature_names": list(ALLOWED_FEATURES),
        "feature_changed_row_counts": feature_changed_rows,
        "feature_max_abs_deltas": feature_max_abs,
        "exact_candidate_metric_counts": counts,
        "candidate_identity_boxes_ious_exact": True,
        "regular_scene_features_exact": True,
        "non_mask_feature_columns_exact": True,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-cache", required=True)
    parser.add_argument("--new-cache", required=True)
    parser.add_argument("--fallback-scenes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(str(output))
    report = audit(args.old_cache, args.new_cache, args.fallback_scenes)
    raw = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    fd = os.open(str(output), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
