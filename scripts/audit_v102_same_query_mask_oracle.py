#!/usr/bin/env python
"""Audit mask-only headroom at frozen V101 OOF-selected parent queries."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import torch


EXPECTED_SIDECAR_SCHEMA = "rec-v101-oof-row-decisions-v1"
EXPECTED_CACHE_SCHEMA = "rec-joint-box-mask-cache-v1"
EXPECTED_ROW_COUNT = 36665
EXPECTED_SCENE_COUNT = 562
SOURCE_NAMES = ("text", "query", "fused")
THRESHOLDS = (-1.0, -0.5, 0.0, 0.5, 1.0)
POLICY_COUNT = len(SOURCE_NAMES) * len(THRESHOLDS)


def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_policy_metadata(manifest):
    if tuple(manifest.get("source_names", ())) != SOURCE_NAMES:
        raise ValueError("mask source order changed")
    if tuple(float(x) for x in manifest.get("thresholds", ())) != THRESHOLDS:
        raise ValueError("mask threshold order changed")
    return SOURCE_NAMES.index("fused") * len(THRESHOLDS) + THRESHOLDS.index(0.0)


def validate_alignment(dataset_indices, rows):
    actual = torch.tensor([int(row["dataset_index"]) for row in rows], dtype=torch.long)
    expected = torch.as_tensor(dataset_indices, dtype=torch.long).cpu()
    if not torch.equal(actual, expected):
        raise ValueError("joint cache dataset index alignment changed")


def selected_query_policy_values(policy_ious, selected_queries, legacy_index):
    policy_ious = torch.as_tensor(policy_ious, dtype=torch.float32)
    selected_queries = torch.as_tensor(selected_queries, dtype=torch.long)
    if policy_ious.dim() != 3 or policy_ious.shape[-1] != POLICY_COUNT:
        raise ValueError("policy IoUs must have shape [N,K,15]")
    if selected_queries.shape != (policy_ious.shape[0],):
        raise ValueError("selected queries must have shape [N]")
    if bool(((selected_queries < 0) | (selected_queries >= policy_ious.shape[1])).any()):
        raise ValueError("selected query index out of range")
    rows = torch.arange(policy_ious.shape[0])
    selected = policy_ious[rows, selected_queries]
    oracle, oracle_policy = selected.max(dim=-1)
    return {
        "baseline": selected[:, int(legacy_index)],
        "oracle": oracle,
        "oracle_policy": oracle_policy,
        "selected_policy_ious": selected,
    }


def metric_summary(values):
    values = torch.as_tensor(values, dtype=torch.float64).cpu()
    if values.dim() != 1 or values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("metric values must be a finite non-empty vector")
    count = int(values.numel())
    return {
        "count": count,
        "hits025": int(values.gt(0.25).sum()),
        "hits050": int(values.gt(0.50).sum()),
        "acc025": float(values.gt(0.25).double().mean()),
        "acc050": float(values.gt(0.50).double().mean()),
        "miou": float(values.mean()),
    }


def delta_summary(baseline, selected):
    baseline = torch.as_tensor(baseline, dtype=torch.float64)
    selected = torch.as_tensor(selected, dtype=torch.float64)
    if baseline.shape != selected.shape:
        raise ValueError("delta vectors must share shape")
    before = metric_summary(baseline)
    after = metric_summary(selected)
    return {
        "before": before,
        "after": after,
        "delta_hits025": after["hits025"] - before["hits025"],
        "delta_hits050": after["hits050"] - before["hits050"],
        "delta_acc025": after["acc025"] - before["acc025"],
        "delta_acc050": after["acc050"] - before["acc050"],
        "delta_miou": after["miou"] - before["miou"],
        "improved_iou": int(selected.gt(baseline).sum()),
        "equal_iou": int(selected.eq(baseline).sum()),
        "degraded_iou": int(selected.lt(baseline).sum()),
    }


def load_cache_rows(cache_dir, manifest):
    rows = []
    for name in manifest.get("shards", ()):
        payload = torch.load(Path(cache_dir) / name, map_location="cpu")
        if isinstance(payload, dict) and "rows" in payload:
            payload = payload["rows"]
        if not isinstance(payload, (list, tuple)):
            raise ValueError("joint cache shard must contain rows")
        rows.extend(payload)
    return rows


def write_exclusive_json(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--joint-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(str(output))

    sidecar_path = Path(args.sidecar).expanduser().absolute()
    sidecar_sha = sha256_file(sidecar_path)
    sidecar = torch.load(sidecar_path, map_location="cpu")
    if sidecar.get("schema") != EXPECTED_SIDECAR_SCHEMA:
        raise ValueError("V101 sidecar schema changed")
    if sidecar.get("validation_data_accessed") is not False:
        raise ValueError("V101 sidecar is not train-only")
    if int(sidecar.get("row_count", -1)) != EXPECTED_ROW_COUNT:
        raise ValueError("V101 sidecar row count changed")
    if int(sidecar.get("scene_count", -1)) != EXPECTED_SCENE_COUNT:
        raise ValueError("V101 sidecar scene count changed")

    cache_dir = Path(args.joint_cache).expanduser().absolute()
    manifest_path = cache_dir / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("schema") != EXPECTED_CACHE_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("joint cache is incomplete or has wrong schema")
    if manifest.get("split") != "train" or manifest.get("validation_data_accessed") is not False:
        raise ValueError("joint cache is not train-only")
    if int(manifest.get("sample_count", -1)) != EXPECTED_ROW_COUNT:
        raise ValueError("joint cache sample count changed")
    legacy_index = validate_policy_metadata(manifest)
    rows = load_cache_rows(cache_dir, manifest)
    if len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError("joint cache row coverage changed")
    validate_alignment(sidecar["dataset_indices"], rows)

    policy_ious = torch.stack([
        torch.as_tensor(row["mask_ious"], dtype=torch.float32).reshape(16, POLICY_COUNT)
        for row in rows
    ])
    selected_query = torch.as_tensor(sidecar["selected_parent_positions"], dtype=torch.long)
    baseline_query = torch.div(
        torch.as_tensor(sidecar["baseline_indices"], dtype=torch.long),
        7, rounding_mode="floor",
    )
    selected = selected_query_policy_values(policy_ious, selected_query, legacy_index)
    parent = selected_query_policy_values(policy_ious, baseline_query, legacy_index)
    folds = torch.as_tensor(sidecar["fold_ids"], dtype=torch.long)
    accepted = torch.as_tensor(sidecar["accepted"], dtype=torch.bool)

    global_policy = []
    for policy in range(POLICY_COUNT):
        values = selected["selected_policy_ious"][:, policy]
        global_policy.append({
            "policy_index": policy,
            "source": SOURCE_NAMES[policy // len(THRESHOLDS)],
            "threshold": THRESHOLDS[policy % len(THRESHOLDS)],
            **metric_summary(values),
        })
    policy_counts = torch.bincount(selected["oracle_policy"], minlength=POLICY_COUNT)
    result = {
        "schema": "rec-v102-same-query-mask-oracle-audit-v1",
        "version": 1,
        "validation_data_accessed": False,
        "row_count": EXPECTED_ROW_COUNT,
        "scene_count": EXPECTED_SCENE_COUNT,
        "sidecar": {"path": str(sidecar_path), "sha256": sidecar_sha},
        "joint_cache": {"path": str(cache_dir), "manifest_sha256": manifest_sha},
        "policy_contract": {
            "source_names": list(SOURCE_NAMES), "thresholds": list(THRESHOLDS),
            "legacy_policy_index": legacy_index,
        },
        "v101_query_binding_delta": delta_summary(parent["baseline"], selected["baseline"]),
        "same_query_oracle_delta": delta_summary(selected["baseline"], selected["oracle"]),
        "accepted_partition": {
            "accepted": delta_summary(selected["baseline"][accepted], selected["oracle"][accepted]),
            "unchanged": delta_summary(selected["baseline"][~accepted], selected["oracle"][~accepted]),
        },
        "folds": [
            {"held_fold": fold, **delta_summary(
                selected["baseline"][folds.eq(fold)], selected["oracle"][folds.eq(fold)]
            )}
            for fold in range(5)
        ],
        "global_fixed_policies": global_policy,
        "oracle_policy_counts": [int(x) for x in policy_counts],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output_sha = write_exclusive_json(output, result)
    print(json.dumps({
        "output": str(output), "sha256": output_sha,
        "same_query_oracle_delta": result["same_query_oracle_delta"],
        "v101_query_binding_delta": result["v101_query_binding_delta"],
        "mode": output.stat().st_mode & 0o777,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
