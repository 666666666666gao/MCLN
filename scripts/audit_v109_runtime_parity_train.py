#!/usr/bin/env python
"""Exact train-cache parity audit for V109 materialization versus runtime."""

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

import torch

from models.rec_geometry_reranker import blend_rec_geometry_scores
from models.rec_pareto_contextual_hierarchy import (
    apply_pareto_contextual_policy,
)
from scripts.run_v108_meshsp_pareto_oof import (
    load_v108_meshsp_training_inputs,
    validate_v108_materialization_artifact,
)
from scripts.train_rec_reranker import normalize_features
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    HIERARCHICAL_MATERIALIZATION_BATCH_SIZE,
    _disabled_parent_autocast,
    _normalized_hierarchical_model_batch,
    capture_immutable_artifact_identities,
    materialize_hierarchical_rows,
)
from scripts.train_rec_geometry_reranker import build_geometry_training_batch
from train_dist_mod import (
    _build_rec_hierarchical_runtime_batch,
    load_rec_hierarchical_runtime_artifact,
)


V109_ARTIFACT_SHA256 = (
    "20db69ddc27680a035384277bc48cd44109215e3d7d1158cdc4a4f21ff7c785b"
)
FIELDS = (
    "query_features", "variant_features", "query_aux_continuous",
    "query_aux_binary", "variant_aux_continuous", "variant_aux_binary",
    "query_valid", "variant_valid",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly_identity(path):
    path = Path(path).expanduser().absolute()
    entry = os.lstat(str(path))
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError("V109 artifact must be a regular file")
    mode = stat.S_IMODE(entry.st_mode)
    if mode != 0o444:
        raise ValueError("V109 artifact must have mode 0444")
    return {
        "path": str(path.resolve(strict=True)), "device": int(entry.st_dev),
        "inode": int(entry.st_ino), "mode": mode, "size": int(entry.st_size),
        "mtime_ns": int(entry.st_mtime_ns), "ctime_ns": int(entry.st_ctime_ns),
        "sha256": _sha256(path),
    }


def _write_readonly_json(path, value):
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444
    )
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("parity output write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _comparison(left, right):
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return {"equal": False, "shape_or_dtype_mismatch": True}
    equal = torch.equal(left, right)
    result = {"equal": bool(equal), "shape_or_dtype_mismatch": False}
    if left.dtype.is_floating_point:
        unequal = left.ne(right)
        finite_mismatch = unequal & torch.isfinite(left) & torch.isfinite(right)
        nonfinite_mismatch = unequal & ~(
            torch.isfinite(left) & torch.isfinite(right)
        )
        result["different_elements"] = int(unequal.sum().item())
        result["nonfinite_mismatches"] = int(
            nonfinite_mismatch.sum().item()
        )
        result["max_abs"] = (
            float((left[finite_mismatch] - right[finite_mismatch]).abs().max().item())
            if bool(finite_mismatch.any().item()) else 0.0
        )
    else:
        result["different_elements"] = int(left.ne(right).sum().item())
    return result


def _accumulate(summary, name, comparison, row_start):
    state = summary.setdefault(name, {
        "equal": True, "max_abs": 0.0, "different_elements": 0,
        "first_different_row": None,
    })
    state["equal"] = state["equal"] and comparison["equal"]
    state["different_elements"] += comparison.get("different_elements", 0)
    state["max_abs"] = max(state["max_abs"], comparison.get("max_abs", 0.0))
    if not comparison["equal"] and state["first_different_row"] is None:
        state["first_different_row"] = int(row_start)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--v109-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(str(output))
    if _sha256(args.v109_artifact) != V109_ARTIFACT_SHA256:
        raise ValueError("V109 artifact SHA changed")

    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_artifact).expanduser().resolve(),
        "geometry": Path(args.geometry_artifact).expanduser().resolve(),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    protected_before["v109"] = _readonly_identity(args.v109_artifact)
    loaded = load_v108_meshsp_training_inputs(
        Path(args.base_cache), Path(args.geometry_cache),
        Path(args.parent_artifact), Path(args.geometry_artifact),
        device=args.device,
    )
    rows = loaded["joined_rows"]
    records = materialize_hierarchical_rows(
        rows, loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=True,
        artifact_validator=validate_v108_materialization_artifact,
    )
    model, artifact = load_rec_hierarchical_runtime_artifact(
        args.v109_artifact, args.device,
        loaded["parent"][0], loaded["geometry_model"],
        loaded["geometry_artifact"],
    )
    if getattr(model, "_artifact_sha256", None) != V109_ARTIFACT_SHA256:
        raise ValueError("V109 runtime loader did not bind the frozen artifact")
    device = torch.device(args.device)
    geometry_model = loaded["geometry_model"].to(device).eval()
    field_summary = {}
    decision_summary = {}
    compared = 0
    with torch.no_grad(), _disabled_parent_autocast(device):
        for start in range(0, len(rows), HIERARCHICAL_MATERIALIZATION_BATCH_SIZE):
            row_batch = rows[start:start + HIERARCHICAL_MATERIALIZATION_BATCH_SIZE]
            record_batch = records[
                start:start + HIERARCHICAL_MATERIALIZATION_BATCH_SIZE
            ]
            offline = _normalized_hierarchical_model_batch(
                record_batch, artifact["normalization"], device
            )
            batch = build_geometry_training_batch(row_batch, loaded["parent"])
            raw_features = batch["features"].to(device, dtype=torch.float32)
            flat_valid = batch["valid_mask"].to(device)
            normalized_geometry = normalize_features(
                raw_features, flat_valid,
                loaded["geometry_artifact"]["feature_mean"],
                loaded["geometry_artifact"]["feature_std"],
            )
            geometry_outputs = geometry_model(normalized_geometry, flat_valid)
            learned_logits = geometry_outputs["ranking_logits"].float()
            parent_state = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch["parent_state"].items()
            }
            geometry_valid = flat_valid.reshape(len(row_batch), 16, 7)
            blended = blend_rec_geometry_scores(
                parent_state, learned_logits, geometry_valid,
                float(loaded["geometry_artifact"]["geometry_weight"]),
                loaded["geometry_artifact"]["regressed_variant_index"],
            )
            flat_scores = blended["flat_scores"]
            candidate_batch = {
                "default_scores": torch.stack([
                    row["base"]["default_scores"] for row in row_batch
                ]).to(device, dtype=torch.float32),
                "default_top1_query_index": torch.tensor([
                    int(row["base"]["default_top1_query_index"])
                    for row in row_batch
                ], dtype=torch.long, device=device),
            }
            model_inputs = {
                "features": raw_features, "valid_mask": flat_valid,
                "feature_names": batch["feature_names"],
                "query_positions": batch["query_positions"].to(device),
                "variant_indices": batch["variant_indices"].to(device),
            }
            runtime = _build_rec_hierarchical_runtime_batch(
                candidate_batch, parent_state, model_inputs, geometry_valid,
                learned_logits, flat_scores, loaded["geometry_artifact"],
                artifact,
            )
            for field in FIELDS:
                _accumulate(
                    field_summary, field,
                    _comparison(offline[field], runtime[field]), start,
                )

            offline_baselines = torch.tensor([
                record["baseline_index"] for record in record_batch
            ], dtype=torch.long, device=device)
            runtime_baselines = flat_scores.argmax(dim=1)
            _accumulate(
                decision_summary, "baseline_indices",
                _comparison(offline_baselines, runtime_baselines), start,
            )
            stored_scores = torch.stack([
                record["baseline_scores"] for record in record_batch
            ]).to(device)
            _accumulate(
                decision_summary, "baseline_scores",
                _comparison(stored_scores, flat_scores), start,
            )
            offline_outputs = model(**offline)
            runtime_outputs = model(**runtime)
            for field in ("query_logits", "variant_logits"):
                _accumulate(
                    decision_summary, field,
                    _comparison(
                        offline_outputs[field], runtime_outputs[field]
                    ), start,
                )
            offline_policy = apply_pareto_contextual_policy(
                stored_scores, offline_outputs["query_logits"],
                offline_outputs["variant_logits"], offline["query_valid"],
                offline["variant_valid"],
                float(artifact["policy"]["aggregate_margin"]),
                min_head_gain025=artifact["policy"]["min_head_gain025"],
                min_head_gain050=artifact["policy"]["min_head_gain050"],
            )
            runtime_policy = apply_pareto_contextual_policy(
                flat_scores, runtime_outputs["query_logits"],
                runtime_outputs["variant_logits"], runtime["query_valid"],
                runtime["variant_valid"],
                float(artifact["policy"]["aggregate_margin"]),
                min_head_gain025=artifact["policy"]["min_head_gain025"],
                min_head_gain050=artifact["policy"]["min_head_gain050"],
            )
            for field in (
                    "baseline_indices", "proposal_indices", "selected_indices",
                    "switch_mask", "pareto_pass", "head_gain",
                    "aggregate_gain"):
                _accumulate(
                    decision_summary, "policy_" + field,
                    _comparison(
                        offline_policy[field], runtime_policy[field]
                    ), start,
                )
            compared += len(row_batch)

    protected_after = capture_immutable_artifact_identities(protected_paths)
    protected_after["v109"] = _readonly_identity(args.v109_artifact)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during parity audit")
    all_equal = all(
        item["equal"]
        for group in (field_summary, decision_summary)
        for item in group.values()
    )
    report = {
        "schema": "rec-v109-train-runtime-parity-audit-v1", "version": 1,
        "validation_data_accessed": False,
        "contaminated_calibration_accessed": False,
        "weights_modified": False, "row_count": compared,
        "all_equal": all_equal, "field_comparisons": field_summary,
        "decision_comparisons": decision_summary,
        "input_sha256": loaded["input_sha256"],
        "artifact_sha256": V109_ARTIFACT_SHA256,
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    output_sha = _write_readonly_json(output, report)
    print(json.dumps({
        "output": str(output), "sha256": output_sha,
        "row_count": compared, "all_equal": all_equal,
        "field_comparisons": field_summary,
        "decision_comparisons": decision_summary,
    }, sort_keys=True), flush=True)
    if not all_equal:
        raise RuntimeError("V109 offline/runtime train parity failed")


if __name__ == "__main__":
    main()
