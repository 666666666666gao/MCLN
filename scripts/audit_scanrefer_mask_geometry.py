#!/usr/bin/env python
"""Audit mask-derived REC geometry on a deterministic train-only panel."""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch


PANEL_SCHEMA_VERSION = "rec-mask-geometry-audit-panel-v1"
PANEL_BUCKETS = ("fail025", "mid", "pass050")
ROOT = Path(__file__).resolve().parents[1]
PARITY_ATOL = 2e-3
PARITY_RTOL = 2e-3
IOU_PARITY_ATOL = 1e-2
IOU_PARITY_RTOL = 1e-2


def _load_manifest(cache_dir):
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("cache manifest does not exist: {}".format(
            manifest_path
        ))
    with manifest_path.open("r") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("cache manifest must contain a JSON object")
    return manifest


def _manifest_sha256(manifest):
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _iter_manifest_rows(cache_dir, manifest):
    cache_dir = Path(cache_dir)
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("cache manifest shards must be a list")
    for shard_index, shard_name in enumerate(shards):
        expected_name = "shard_{:06d}.pt".format(shard_index)
        if shard_name != expected_name:
            raise ValueError("cache shard names must be contiguous")
        shard_path = cache_dir / shard_name
        if not shard_path.is_file():
            raise ValueError("cache shard does not exist: {}".format(
                shard_path
            ))
        payload = torch.load(shard_path, map_location="cpu")
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("cache shard must contain a row list")
        for row in rows:
            yield row


def _validate_train_manifest(manifest, expected_checkpoint_sha256):
    if manifest.get("cache_schema_version") != 1:
        raise ValueError("unsupported candidate cache schema")
    if manifest.get("feature_schema_version") != "rec-query-v1":
        raise ValueError("unsupported candidate feature schema")
    if manifest.get("split") != "train":
        raise ValueError("mask geometry audit requires a train split cache")
    if manifest.get("checkpoint_sha256") != str(
            expected_checkpoint_sha256):
        raise ValueError("train cache checkpoint fingerprint does not match")
    if not isinstance(manifest.get("checkpoint_epoch"), int):
        raise ValueError("train cache checkpoint epoch is missing")
    if manifest.get("target_iou_policy") != "root_only":
        raise ValueError("train cache must use root_only target IoUs")
    if manifest.get("deterministic") is not True:
        raise ValueError("train cache must declare deterministic extraction")
    counts = tuple(manifest.get(key) for key in (
        "sample_count", "dataset_size", "source_dataset_size"
    ))
    if any(not isinstance(value, int) or value <= 0 for value in counts):
        raise ValueError("train cache sizes must be positive integers")
    if len(set(counts)) != 1:
        raise ValueError("train cache must be a complete source dataset cache")
    feature_dim = manifest.get("feature_dim")
    feature_names = manifest.get("feature_names")
    if (not isinstance(feature_dim, int) or feature_dim <= 0
            or not isinstance(feature_names, list)
            or len(feature_names) != feature_dim):
        raise ValueError("train cache feature schema is incomplete")
    if not isinstance(manifest.get("model_inputs"), dict) \
            or not manifest["model_inputs"]:
        raise ValueError("train cache model input provenance is missing")
    if not isinstance(manifest.get("backbone_config"), dict) \
            or not manifest["backbone_config"]:
        raise ValueError("train cache backbone provenance is missing")
    candidate_rule = manifest.get("candidate_rule")
    if not isinstance(candidate_rule, dict):
        raise ValueError("train cache candidate rule is missing")
    for key in ("topk_per_source", "max_candidates"):
        value = candidate_rule.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError("train cache candidate rule is invalid")


def load_train_cache_panel_records(cache_dir, expected_checkpoint_sha256):
    """Load lightweight panel records from a provenance-checked train cache."""
    manifest = _load_manifest(cache_dir)
    _validate_train_manifest(manifest, expected_checkpoint_sha256)
    sample_count = manifest["sample_count"]

    records = []
    for expected_index, row in enumerate(
            _iter_manifest_rows(cache_dir, manifest)):
        if not isinstance(row, dict):
            raise ValueError("cache rows must be dictionaries")
        if row.get("dataset_index") != expected_index:
            raise ValueError("cache dataset indices must be contiguous")
        records.append(cache_row_to_panel_record(row))
    if len(records) != sample_count:
        raise ValueError("cache row count does not match manifest")
    return manifest, records


def load_selected_cache_rows(cache_dir, manifest, dataset_indices):
    """Recover full cached tensors for an exact set of dataset indices."""
    requested = {int(index) for index in dataset_indices}
    selected = {}
    for row in _iter_manifest_rows(cache_dir, manifest):
        dataset_index = int(row["dataset_index"])
        if dataset_index in requested:
            if dataset_index in selected:
                raise ValueError("cache dataset indices must be unique")
            selected[dataset_index] = row
    missing = requested.difference(selected)
    if missing:
        raise ValueError("selected dataset indices are absent from cache")
    return selected


def build_cache_replay_groups(
        selected_indices, source_dataset_size, extraction_batch_size,
        replay_boundaries=(0,)):
    """Restore contiguous batches used by the original cache extraction."""
    if (not isinstance(source_dataset_size, int)
            or isinstance(source_dataset_size, bool)
            or source_dataset_size <= 0):
        raise ValueError("source_dataset_size must be a positive integer")
    if (not isinstance(extraction_batch_size, int)
            or isinstance(extraction_batch_size, bool)
            or extraction_batch_size <= 0):
        raise ValueError("extraction_batch_size must be a positive integer")
    selected = [int(index) for index in selected_indices]
    if len(set(selected)) != len(selected):
        raise ValueError("selected dataset indices must be unique")
    if any(index < 0 or index >= source_dataset_size for index in selected):
        raise ValueError("selected dataset index is outside the source dataset")
    replay_boundaries = tuple(int(value) for value in replay_boundaries)
    if (not replay_boundaries or replay_boundaries[0] != 0
            or tuple(sorted(set(replay_boundaries))) != replay_boundaries
            or any(value < 0 or value >= source_dataset_size
                   for value in replay_boundaries)):
        raise ValueError(
            "replay_boundaries must be sorted unique starts beginning at zero"
        )
    grouped = {}
    for index in selected:
        boundary = max(value for value in replay_boundaries if value <= index)
        start = boundary + (
            (index - boundary) // extraction_batch_size
        ) * extraction_batch_size
        grouped.setdefault((start, boundary), []).append(index)
    output = []
    for start, boundary in sorted(grouped):
        stop = min(start + extraction_batch_size, source_dataset_size)
        selected_in_batch = tuple(sorted(grouped[(start, boundary)]))
        output.append({
            "batch_indices": tuple(range(start, stop)),
            "selected_indices": selected_in_batch,
            "selected_positions": tuple(
                index - start for index in selected_in_batch
            ),
            "replay_boundary": boundary,
        })
    return output


def _batch_identity(value, index):
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.numel() == 1 else item.detach().cpu()
    return value[index]


def assert_candidate_cache_parity(
        candidate_batch, cached_rows, dataset_indices, scan_ids, target_ids,
        atol=PARITY_ATOL, rtol=PARITY_RTOL,
        iou_atol=IOU_PARITY_ATOL,
        iou_rtol=IOU_PARITY_RTOL, identity_only=False):
    """Check fresh candidates against a cache.

    The default contract is strict byte-level candidate parity.  ``identity_only``
    is an explicit provenance escape hatch for a downstream audit whose old
    cache is used only to select immutable dataset rows.  It still requires
    scene/target identity and finite, well-formed values, but reports candidate
    query/score/box drift instead of treating historical tensors as runtime
    truth.
    """
    if type(identity_only) is not bool:
        raise TypeError("identity_only must be boolean")
    dataset_indices = [int(index) for index in dataset_indices]
    batch_size = candidate_batch["query_indices"].shape[0]
    if len(dataset_indices) != batch_size:
        raise ValueError("dataset indices must match fresh candidate batch")
    fields = (
        "query_indices", "valid_mask", "boxes", "candidate_ious",
        "features", "default_scores", "contrastive_scores",
    )
    max_differences = {
        field: 0.0 for field in fields
        if field not in ("query_indices", "valid_mask")
    }
    if identity_only:
        max_differences.update({
            "query_identity_drift_count": 0,
            "validity_identity_drift_count": 0,
            "default_query_identity_drift_count": 0,
            "default_bucket_drift_count": 0,
            "shape_mismatch_count": 0,
        })
    for batch_idx, dataset_index in enumerate(dataset_indices):
        if dataset_index not in cached_rows:
            raise ValueError("selected dataset index is absent from cache rows")
        cached = cached_rows[dataset_index]
        if str(_batch_identity(scan_ids, batch_idx)) != str(cached["scan_id"]):
            raise ValueError("fresh scan identity does not match cache")
        if int(_batch_identity(target_ids, batch_idx)) != int(
                cached["target_id"]):
            raise ValueError("fresh target identity does not match cache")
        for field in fields:
            fresh = candidate_batch[field][batch_idx].detach().cpu()
            cached_value = torch.as_tensor(cached[field]).detach().cpu()
            if fresh.shape != cached_value.shape:
                if identity_only:
                    max_differences["shape_mismatch_count"] += 1
                    if field == "query_indices":
                        max_differences["query_identity_drift_count"] += 1
                    elif field == "valid_mask":
                        max_differences["validity_identity_drift_count"] += 1
                    continue
                raise ValueError("fresh {} shape does not match cache".format(
                    field
                ))
            if field == "features":
                if (not torch.isfinite(fresh).all()
                        or not torch.isfinite(cached_value).all()):
                    raise ValueError(
                        "fresh and cached features must be finite"
                    )
                equal = True
            elif field in ("default_scores", "contrastive_scores"):
                if (not torch.isfinite(fresh).all()
                        or not torch.isfinite(cached_value).all()):
                    raise ValueError(
                        "fresh and cached candidate scores must be finite"
                    )
                equal = True
            elif field == "candidate_ious":
                equal = torch.allclose(
                    fresh.float(), cached_value.float(),
                    atol=iou_atol, rtol=iou_rtol,
                )
            elif field == "boxes":
                equal = torch.allclose(
                    fresh.float(), cached_value.float(), atol=atol, rtol=rtol
                )
            else:
                equal = torch.equal(fresh, cached_value.to(fresh.dtype))
            if torch.is_floating_point(fresh):
                max_difference = float(
                    (fresh.float() - cached_value.float()).abs().max().item()
                )
                max_differences[field] = max(
                    max_differences[field], max_difference
                )
            if identity_only:
                if field == "query_indices" and not equal:
                    max_differences["query_identity_drift_count"] += 1
                elif field == "valid_mask" and not equal:
                    max_differences["validity_identity_drift_count"] += 1
                # All other differences are retained in the numeric maxima
                # above and are intentionally diagnostic in this mode.
                continue
            if not equal:
                if torch.is_floating_point(fresh):
                    raise ValueError(
                        "fresh {} do not match cache (max diff {:.6g})".format(
                            field, max_difference
                        )
                    )
                raise ValueError("fresh {} do not match cache".format(field))
        fresh_default = int(
            candidate_batch["default_top1_query_index"][batch_idx].item()
        )
        cached_default = int(cached["default_top1_query_index"])
        if fresh_default != cached_default:
            if identity_only:
                max_differences["default_query_identity_drift_count"] += 1
            else:
                raise ValueError("fresh default Top-1 query does not match cache")
        cached_record = cache_row_to_panel_record(cached)
        fresh_matches = (
            candidate_batch["query_indices"][batch_idx].detach().cpu().long()
            == fresh_default
        ) & candidate_batch["valid_mask"][batch_idx].detach().cpu().bool()
        if int(fresh_matches.sum().item()) != 1:
            if identity_only:
                max_differences["default_query_identity_drift_count"] += 1
                continue
            raise ValueError("fresh default Top-1 query does not match cache")
        fresh_position = int(fresh_matches.nonzero(as_tuple=False)[0, 0].item())
        fresh_iou = float(
            candidate_batch["candidate_ious"][batch_idx, fresh_position].item()
        )
        if fresh_iou <= 0.25:
            fresh_bucket = "fail025"
        elif fresh_iou <= 0.50:
            fresh_bucket = "mid"
        else:
            fresh_bucket = "pass050"
        if fresh_bucket != cached_record["bucket"]:
            if identity_only:
                max_differences["default_bucket_drift_count"] += 1
                continue
            raise ValueError("fresh default Top-1 IoU bucket does not match cache")
    return max_differences


def extract_default_variant_diagnostics(geometry, default_positions):
    """Gather rejection, occupancy, and final-volume data for Top-1 queries."""
    boxes = geometry["boxes"]
    valid = geometry["valid_mask"].bool()
    if boxes.dim() != 4 or boxes.shape[-1] != 6:
        raise ValueError("geometry boxes must have shape [B,K,G,6]")
    if valid.shape != boxes.shape[:3]:
        raise ValueError("geometry valid_mask must match boxes")
    batch_size, num_candidates, num_variants = valid.shape
    default_positions = torch.as_tensor(
        default_positions, device=boxes.device, dtype=torch.long
    )
    if default_positions.shape != (batch_size,):
        raise ValueError("default_positions must have shape [B]")
    if bool(((default_positions < 0)
             | (default_positions >= num_candidates)).any().item()):
        raise ValueError("default position is outside candidate axis")
    variant_configs = tuple(geometry["variant_configs"])
    if len(variant_configs) != num_variants:
        raise ValueError("variant configs must match geometry axis")
    regressed_indices = [
        idx for idx, config in enumerate(variant_configs)
        if config["source"] == "regressed"
    ]
    if len(regressed_indices) != 1:
        raise ValueError("geometry needs exactly one regressed variant")
    regressed_idx = regressed_indices[0]
    row_indices = torch.arange(batch_size, device=boxes.device)
    default_boxes = boxes[row_indices, default_positions]
    default_valid = valid[row_indices, default_positions]
    regressed_volume = default_boxes[:, regressed_idx, 3:].clamp(
        min=0.0
    ).prod(dim=-1)
    variant_volume = default_boxes[..., 3:].clamp(min=0.0).prod(dim=-1)
    volume_ratios = torch.where(
        default_valid & (regressed_volume.unsqueeze(1) > 0.0),
        variant_volume / regressed_volume.unsqueeze(1).clamp(min=1e-12),
        torch.zeros_like(variant_volume),
    )
    rejection_codes = torch.zeros(
        batch_size, num_variants, dtype=torch.long, device=boxes.device
    )
    point_fractions = boxes.new_zeros(batch_size, num_variants)
    mask_diagnostics = geometry["mask_diagnostics"]
    if len(mask_diagnostics) != batch_size:
        raise ValueError("mask diagnostics must match batch size")

    for batch_idx in range(batch_size):
        candidate_idx = int(default_positions[batch_idx].item())
        for variant_idx, config in enumerate(variant_configs):
            source = config["source"]
            if source == "regressed":
                continue
            group_name = "{}_t{:g}".format(
                source, float(config["logit_threshold"])
            )
            diagnostics = mask_diagnostics[batch_idx].get(group_name)
            if diagnostics is None:
                raise ValueError("mask diagnostics are missing {}".format(
                    group_name
                ))
            quantiles = diagnostics["quantiles"].to(boxes.device).float()
            target_quantile = quantiles.new_tensor(float(config["quantile"]))
            quantile_matches = torch.isclose(
                quantiles, target_quantile, atol=1e-8, rtol=0.0
            ).nonzero(as_tuple=False).reshape(-1)
            if quantile_matches.numel() != 1:
                raise ValueError("mask diagnostic quantile is ambiguous")
            quantile_idx = int(quantile_matches[0].item())
            rejection_codes[batch_idx, variant_idx] = diagnostics[
                "rejection_codes"
            ][candidate_idx, quantile_idx].to(boxes.device)
            point_fractions[batch_idx, variant_idx] = diagnostics[
                "selected_point_fractions"
            ][candidate_idx].to(boxes.device)
    return {
        "rejection_codes": rejection_codes.detach().cpu(),
        "selected_point_fractions": point_fractions.detach().cpu(),
        "final_volume_ratios": volume_ratios.detach().cpu(),
        "default_variant_valid": default_valid.detach().cpu(),
    }


def _as_cpu_vector(value, name):
    value = torch.as_tensor(value).detach().cpu()
    if value.dim() != 1:
        raise ValueError("{} must be one-dimensional".format(name))
    return value


def cache_row_to_panel_record(row):
    """Extract deployable identity and default-IoU strata from one cache row."""
    if not isinstance(row, dict):
        raise TypeError("cache row must be a dictionary")
    query_indices = _as_cpu_vector(row.get("query_indices"), "query_indices")
    valid_mask = _as_cpu_vector(row.get("valid_mask"), "valid_mask").bool()
    candidate_ious = _as_cpu_vector(
        row.get("candidate_ious"), "candidate_ious"
    ).float()
    if not (
            query_indices.shape == valid_mask.shape == candidate_ious.shape):
        raise ValueError("candidate cache vectors must have equal shape")
    default_query = int(row["default_top1_query_index"])
    matches = (query_indices.long() == default_query) & valid_mask
    if int(matches.sum().item()) != 1:
        raise ValueError(
            "default Top-1 query must occur exactly once among valid candidates"
        )
    position = int(matches.nonzero(as_tuple=False)[0, 0].item())
    default_iou = float(candidate_ious[position].item())
    if not math.isfinite(default_iou):
        raise ValueError("default Top-1 IoU must be finite")
    if default_iou <= 0.25:
        bucket = "fail025"
    elif default_iou <= 0.50:
        bucket = "mid"
    else:
        bucket = "pass050"
    return {
        "dataset_index": int(row["dataset_index"]),
        "scan_id": str(row["scan_id"]),
        "target_id": int(row["target_id"]),
        "default_position": position,
        "default_iou": default_iou,
        "bucket": bucket,
    }


def _digest_order(namespace, seed, value):
    payload = "{}:{}:{}".format(namespace, int(seed), value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_baseline_stratified_panel(
        records, scene_count=64, expressions_per_scene=4, seed=0):
    """Select deterministic scenes with default failures, mids, and passes."""
    if not isinstance(scene_count, int) or isinstance(scene_count, bool) \
            or scene_count <= 0:
        raise ValueError("scene_count must be a positive integer")
    if (not isinstance(expressions_per_scene, int)
            or isinstance(expressions_per_scene, bool)
            or expressions_per_scene < len(PANEL_BUCKETS)):
        raise ValueError(
            "expressions_per_scene must cover all diagnostic buckets"
        )
    grouped = {}
    dataset_indices = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("panel records must be dictionaries")
        bucket = record.get("bucket")
        if bucket not in PANEL_BUCKETS:
            raise ValueError("panel record has an unknown bucket")
        dataset_index = int(record["dataset_index"])
        if dataset_index in dataset_indices:
            raise ValueError("panel record dataset indices must be unique")
        dataset_indices.add(dataset_index)
        scan_id = str(record["scan_id"])
        grouped.setdefault(scan_id, []).append(dict(record))

    eligible = []
    for scan_id, scene_rows in grouped.items():
        present = {row["bucket"] for row in scene_rows}
        if (set(PANEL_BUCKETS).issubset(present)
                and len(scene_rows) >= expressions_per_scene):
            eligible.append(scan_id)
    eligible.sort(key=lambda scan_id: (
        _digest_order(PANEL_SCHEMA_VERSION, seed, scan_id), scan_id
    ))
    if len(eligible) < scene_count:
        raise ValueError(
            "requested {} scenes but only {} eligible scenes exist".format(
                scene_count, len(eligible)
            )
        )

    selected = []
    for scan_id in eligible[:scene_count]:
        rows = sorted(grouped[scan_id], key=lambda row: (
            _digest_order(
                PANEL_SCHEMA_VERSION + "-row",
                seed,
                "{}:{}".format(scan_id, int(row["dataset_index"])),
            ),
            int(row["dataset_index"]),
        ))
        chosen = []
        chosen_indices = set()
        for bucket in PANEL_BUCKETS:
            row = next(row for row in rows if row["bucket"] == bucket)
            chosen.append(row)
            chosen_indices.add(int(row["dataset_index"]))
        for row in rows:
            if len(chosen) == expressions_per_scene:
                break
            dataset_index = int(row["dataset_index"])
            if dataset_index not in chosen_indices:
                chosen.append(row)
                chosen_indices.add(dataset_index)
        if len(chosen) != expressions_per_scene:
            raise RuntimeError("eligible scene did not provide enough rows")
        selected.extend(chosen)
    return selected


def _validate_metric_inputs(
        regressed_ious, regressed_valid, geometry_ious, geometry_valid,
        default_positions, variant_names):
    regressed_ious = torch.as_tensor(regressed_ious).detach().cpu().float()
    regressed_valid = torch.as_tensor(
        regressed_valid
    ).detach().cpu().bool()
    geometry_ious = torch.as_tensor(geometry_ious).detach().cpu().float()
    geometry_valid = torch.as_tensor(
        geometry_valid
    ).detach().cpu().bool()
    default_positions = torch.as_tensor(
        default_positions
    ).detach().cpu().long()
    if regressed_ious.dim() != 2 or regressed_ious.shape[0] == 0:
        raise ValueError("regressed_ious must have nonempty shape [N,K]")
    if regressed_valid.shape != regressed_ious.shape:
        raise ValueError("regressed_valid must match regressed_ious")
    if (geometry_ious.dim() != 3
            or geometry_ious.shape[:2] != regressed_ious.shape):
        raise ValueError("geometry_ious must have shape [N,K,G]")
    if geometry_valid.shape != geometry_ious.shape:
        raise ValueError("geometry_valid must match geometry_ious")
    if default_positions.shape != regressed_ious.shape[:1]:
        raise ValueError("default_positions must have shape [N]")
    if bool(((default_positions < 0)
             | (default_positions >= regressed_ious.shape[1])).any().item()):
        raise ValueError("default position is outside the candidate axis")
    variant_names = tuple(str(name) for name in variant_names)
    if len(variant_names) != geometry_ious.shape[2]:
        raise ValueError("variant_names must match the geometry axis")
    if len(set(variant_names)) != len(variant_names):
        raise ValueError("variant names must be unique")
    if not regressed_valid.any(dim=1).all():
        raise ValueError("every row needs a valid regressed candidate")
    row_indices = torch.arange(regressed_ious.shape[0])
    if not regressed_valid[row_indices, default_positions].all():
        raise ValueError("default candidate must be valid")
    if not torch.isfinite(regressed_ious[regressed_valid]).all():
        raise ValueError("valid regressed IoUs must be finite")
    if not torch.isfinite(geometry_ious[geometry_valid]).all():
        raise ValueError("valid geometry IoUs must be finite")
    return (
        regressed_ious,
        regressed_valid,
        geometry_ious,
        geometry_valid,
        default_positions,
        variant_names,
    )


def _threshold_metrics(values):
    result = {}
    for suffix, threshold in (("025", 0.25), ("050", 0.50)):
        hits = int((values > threshold).sum().item())
        result["hits{}".format(suffix)] = hits
        result["acc{}".format(suffix)] = hits / float(values.numel())
    return result


def summarize_geometry_metrics(
        regressed_ious, regressed_valid, geometry_ious, geometry_valid,
        default_positions, variant_names):
    """Summarize strict default, fallback, and augmented-pool metrics."""
    (
        regressed_ious,
        regressed_valid,
        geometry_ious,
        geometry_valid,
        default_positions,
        variant_names,
    ) = _validate_metric_inputs(
        regressed_ious,
        regressed_valid,
        geometry_ious,
        geometry_valid,
        default_positions,
        variant_names,
    )
    num_rows = regressed_ious.shape[0]
    row_indices = torch.arange(num_rows)
    default_ious = regressed_ious[row_indices, default_positions]
    baseline_oracle = regressed_ious.masked_fill(
        ~regressed_valid, -float("inf")
    ).max(dim=1).values
    summary = {
        "sample_count": int(num_rows),
        "baseline_default": _threshold_metrics(default_ious),
        "baseline_oracle": _threshold_metrics(baseline_oracle),
        "variants": {},
    }

    for variant_idx, variant_name in enumerate(variant_names):
        variant_default_ious = geometry_ious[
            row_indices, default_positions, variant_idx
        ]
        variant_default_valid = geometry_valid[
            row_indices, default_positions, variant_idx
        ]
        fallback_ious = torch.where(
            variant_default_valid, variant_default_ious, default_ious
        )
        raw_ious = variant_default_ious.masked_fill(
            ~variant_default_valid, -float("inf")
        )
        variant_pool_best = geometry_ious[:, :, variant_idx].masked_fill(
            ~geometry_valid[:, :, variant_idx], -float("inf")
        ).max(dim=1).values
        augmented_oracle = torch.maximum(baseline_oracle, variant_pool_best)
        metrics = {
            "invalid_count": int((~variant_default_valid).sum().item()),
            "invalid_rate": float((~variant_default_valid).float().mean().item()),
        }
        for key, value in _threshold_metrics(fallback_ious).items():
            metrics["fallback_{}".format(key)] = value
        for key, value in _threshold_metrics(raw_ious).items():
            metrics["raw_{}".format(key)] = value
        for suffix, threshold in (("025", 0.25), ("050", 0.50)):
            baseline_hit = default_ious > threshold
            fallback_hit = fallback_ious > threshold
            metrics["fixes{}".format(suffix)] = int(
                ((~baseline_hit) & fallback_hit).sum().item()
            )
            metrics["breaks{}".format(suffix)] = int(
                (baseline_hit & (~fallback_hit)).sum().item()
            )
            oracle_hit = baseline_oracle > threshold
            augmented_hit = augmented_oracle > threshold
            metrics["augmented_oracle_acc{}".format(suffix)] = float(
                augmented_hit.float().mean().item()
            )
            metrics["oracle_fixes{}".format(suffix)] = int(
                ((~oracle_hit) & augmented_hit).sum().item()
            )
        summary["variants"][variant_name] = metrics

    combined_geometry = geometry_ious.masked_fill(
        ~geometry_valid, -float("inf")
    ).flatten(1).max(dim=1).values
    combined_oracle = torch.maximum(baseline_oracle, combined_geometry)
    combined_metrics = _threshold_metrics(combined_oracle)
    for suffix, threshold in (("025", 0.25), ("050", 0.50)):
        baseline_hit = baseline_oracle > threshold
        combined_hit = combined_oracle > threshold
        combined_metrics["fixes{}".format(suffix)] = int(
            ((~baseline_hit) & combined_hit).sum().item()
        )
        combined_metrics["breaks{}".format(suffix)] = int(
            (baseline_hit & (~combined_hit)).sum().item()
        )
    summary["combined_oracle"] = combined_metrics
    return summary


_REJECTION_BITS = (
    ("empty", 1),
    ("too_few", 2),
    ("full", 4),
    ("over_fraction", 8),
    ("nonfinite_logits", 16),
    ("nonfinite_coordinates", 32),
    ("nonfinite_box", 64),
    ("degenerate", 128),
)


def _distribution_stats(values, mask=None):
    values = torch.as_tensor(values).detach().cpu().float().reshape(-1)
    finite = torch.isfinite(values)
    if mask is not None:
        mask = torch.as_tensor(mask).detach().cpu().bool().reshape(-1)
        if mask.shape != values.shape:
            raise ValueError("distribution mask must match values")
        finite &= mask
    values = values[finite]
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    quantiles = torch.quantile(
        values, values.new_tensor([0.05, 0.50, 0.95])
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(quantiles[1].item()),
        "p05": float(quantiles[0].item()),
        "p95": float(quantiles[2].item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def summarize_variant_diagnostics(
        rejection_codes, default_variant_valid, selected_point_fractions,
        final_volume_ratios, variant_names):
    """Decode top-one mask rejection reasons and geometry distributions."""
    rejection_codes = torch.as_tensor(
        rejection_codes
    ).detach().cpu().long()
    default_variant_valid = torch.as_tensor(
        default_variant_valid
    ).detach().cpu().bool()
    selected_point_fractions = torch.as_tensor(
        selected_point_fractions
    ).detach().cpu().float()
    final_volume_ratios = torch.as_tensor(
        final_volume_ratios
    ).detach().cpu().float()
    if rejection_codes.dim() != 2 or rejection_codes.shape[0] == 0:
        raise ValueError("rejection_codes must have nonempty shape [N,G]")
    for value, name in (
            (default_variant_valid, "default_variant_valid"),
            (selected_point_fractions, "selected_point_fractions"),
            (final_volume_ratios, "final_volume_ratios")):
        if value.shape != rejection_codes.shape:
            raise ValueError("{} must match rejection_codes".format(name))
    variant_names = tuple(str(name) for name in variant_names)
    if len(variant_names) != rejection_codes.shape[1]:
        raise ValueError("variant_names must match diagnostics")

    num_rows = rejection_codes.shape[0]
    output = {}
    for variant_idx, variant_name in enumerate(variant_names):
        codes = rejection_codes[:, variant_idx]
        rejections = {
            name: int(((codes & bit) != 0).sum().item())
            for name, bit in _REJECTION_BITS
        }
        output[variant_name] = {
            "rejections": rejections,
            "rejection_rates": {
                name: count / float(num_rows)
                for name, count in rejections.items()
            },
            "foreground_fraction": _distribution_stats(
                selected_point_fractions[:, variant_idx]
            ),
            "final_volume_ratio": _distribution_stats(
                final_volume_ratios[:, variant_idx],
                default_variant_valid[:, variant_idx],
            ),
        }
    return output


def parse_args(argv=None):
    """Parse the train-only mask geometry audit command line."""
    parser = argparse.ArgumentParser(
        description="Audit ScanRefer mask-derived REC geometry on train data."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--dataset",
        choices=("scanrefer", "nr3d", "sr3d"),
        default="scanrefer",
        help="dataset-only annotation source used by the candidate cache",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-count", type=int, default=64)
    parser.add_argument("--expressions-per-scene", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument(
        "--selection-mode",
        choices=("baseline-stratified",),
        default="baseline-stratified",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-extraction-batch-size", type=int, default=None)
    parser.add_argument(
        "--cache-replay-boundaries", type=int, nargs="+", default=[0]
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--max-point-fraction", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.cache_extraction_batch_size is None:
        args.cache_extraction_batch_size = args.batch_size
    elif args.cache_extraction_batch_size != args.batch_size:
        parser.error(
            "--batch-size and --cache-extraction-batch-size must match"
        )
    for name in (
            "scene_count", "expressions_per_scene", "batch_size",
            "cache_extraction_batch_size", "min_points"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.expressions_per_scene < len(PANEL_BUCKETS):
        parser.error("--expressions-per-scene must cover all three buckets")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if (not args.cache_replay_boundaries
            or args.cache_replay_boundaries[0] != 0
            or sorted(set(args.cache_replay_boundaries))
            != args.cache_replay_boundaries):
        parser.error(
            "--cache-replay-boundaries must be sorted unique and start at 0"
        )
    if not 0.0 < args.max_point_fraction <= 1.0:
        parser.error("--max-point-fraction must lie in (0, 1]")
    return args


def _ensure_project_imports():
    for path in (str(ROOT), str(ROOT / "pointnet2")):
        if path not in sys.path:
            sys.path.insert(0, path)


def _atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_output_staging(output_dir, overwrite):
    """Create a same-filesystem staging directory without touching old output."""
    final_path = Path(output_dir).expanduser().resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        if not final_path.is_dir():
            raise ValueError("audit output path exists and is not a directory")
        if not overwrite:
            raise ValueError(
                "audit output already exists; pass --overwrite to replace it"
            )
    staging = Path(tempfile.mkdtemp(
        prefix=final_path.name + ".staging.",
        dir=str(final_path.parent),
    ))
    return final_path, staging


def _publish_output_bundle(final_path, staging):
    """Publish a complete three-file audit bundle after all work succeeds."""
    final_path = Path(final_path)
    staging = Path(staging)
    required = ("selection.json", "summary.json", "rows.pt")
    missing = [name for name in required if not (staging / name).is_file()]
    if missing:
        raise ValueError("staged audit bundle is incomplete: {}".format(
            ", ".join(missing)
        ))
    backup = None
    if final_path.exists():
        backup = final_path.with_name(
            final_path.name + ".backup." + uuid.uuid4().hex
        )
        os.replace(str(final_path), str(backup))
    try:
        os.replace(str(staging), str(final_path))
    except Exception:
        if backup is not None and backup.exists() and not final_path.exists():
            os.replace(str(backup), str(final_path))
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(str(backup))


def _set_deterministic(seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _build_replay_loader(dataset, replay_groups, args, device):
    from torch.utils.data import DataLoader
    from scripts.cache_scanrefer_rec_candidates import _seed_worker

    generator = torch.Generator()
    generator.manual_seed(args.selection_seed)
    return DataLoader(
        dataset,
        batch_sampler=[list(group["batch_indices"]) for group in replay_groups],
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _select_batch_value(value, positions, batch_size):
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and value.shape[0] == batch_size:
            index = torch.as_tensor(
                positions, dtype=torch.long, device=value.device
            )
            return value.index_select(0, index)
        return value
    if isinstance(value, list) and len(value) == batch_size:
        return [value[position] for position in positions]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[position] for position in positions)
    if isinstance(value, dict):
        return {
            key: _select_batch_value(item, positions, batch_size)
            for key, item in value.items()
        }
    return value


def _select_batch_mapping(mapping, positions, batch_size):
    return {
        key: _select_batch_value(value, positions, batch_size)
        for key, value in mapping.items()
    }


def _prune_dataset_scenes(dataset, selected_scan_ids):
    selected = set(selected_scan_ids)
    if hasattr(dataset, "scans"):
        dataset.scans = {
            key: value for key, value in dataset.scans.items()
            if key in selected
        }
    if hasattr(dataset, "superpoints"):
        dataset.superpoints = {
            key: value for key, value in dataset.superpoints.items()
            if key in selected
        }


def _fresh_default_positions(candidate_batch):
    matches = (
        candidate_batch["query_indices"]
        == candidate_batch["default_top1_query_index"].unsqueeze(1)
    ) & candidate_batch["valid_mask"].bool()
    if not (matches.sum(dim=1) == 1).all():
        raise ValueError("fresh default Top-1 must occur exactly once")
    return matches.long().argmax(dim=1)


def _compute_geometry_ious(geometry, batch_data):
    from models.rec_reranker import compute_query_ious

    batch_size, num_candidates, num_variants, _ = geometry["boxes"].shape
    gt_boxes = torch.cat([
        batch_data["center_label"][:, :1, :3].float(),
        batch_data["size_gts"][:, :1].float(),
    ], dim=-1)
    gt_mask = batch_data["box_label_mask"][:, :1]
    ious = compute_query_ious(
        geometry["boxes"].reshape(batch_size, -1, 6),
        gt_boxes,
        gt_mask,
    ).reshape(batch_size, num_candidates, num_variants)
    return ious.masked_fill(~geometry["valid_mask"].bool(), 0.0)


def _append_compact_rows(
        output_rows, dataset_indices, panel_by_index, batch_data,
        candidate_batch, geometry, geometry_ious, default_positions,
        diagnostics):
    tensor_fields = {
        "query_indices": candidate_batch["query_indices"],
        "candidate_valid": candidate_batch["valid_mask"],
        "candidate_features": candidate_batch["features"],
        "regressed_boxes": candidate_batch["boxes"],
        "regressed_ious": candidate_batch["candidate_ious"],
        "geometry_boxes": geometry["boxes"],
        "geometry_valid": geometry["valid_mask"],
        "geometry_features": geometry["geometry_features"],
        "geometry_ious": geometry_ious,
        "rejection_codes": diagnostics["rejection_codes"],
        "selected_point_fractions": diagnostics[
            "selected_point_fractions"
        ],
        "final_volume_ratios": diagnostics["final_volume_ratios"],
        "default_variant_valid": diagnostics["default_variant_valid"],
    }
    for batch_idx, dataset_index in enumerate(dataset_indices):
        panel = panel_by_index[int(dataset_index)]
        row = {
            "dataset_index": int(dataset_index),
            "scan_id": str(_batch_identity(
                batch_data["scan_ids"], batch_idx
            )),
            "target_id": int(_batch_identity(
                batch_data["target_id"], batch_idx
            )),
            "bucket": panel["bucket"],
            "cached_default_iou": float(panel["default_iou"]),
            "default_position": int(default_positions[batch_idx].item()),
            "default_top1_query_index": int(
                candidate_batch["default_top1_query_index"][batch_idx].item()
            ),
        }
        for key, value in tensor_fields.items():
            row[key] = value[batch_idx].detach().cpu().clone()
        output_rows.append(row)


def _cat_row_tensors(rows, key):
    return torch.stack([row[key] for row in rows], dim=0)


def _panel_bucket_counts(panel):
    counts = {bucket: 0 for bucket in PANEL_BUCKETS}
    for row in panel:
        counts[row["bucket"]] += 1
    return counts


def _print_summary(summary):
    default = summary["baseline_default"]
    oracle = summary["baseline_oracle"]
    combined = summary["combined_oracle"]
    print(
        "Diagnostic default Top-1: Acc@0.25={:.5f} Acc@0.50={:.5f}".format(
            default["acc025"], default["acc050"]
        )
    )
    print(
        "Regressed Top-16 oracle: Acc@0.25={:.5f} Acc@0.50={:.5f}".format(
            oracle["acc025"], oracle["acc050"]
        )
    )
    for variant_name, metrics in summary["variants"].items():
        print(
            "{}: fallback={:.5f}/{:.5f} fixes={}/{} breaks={}/{} "
            "invalid={:.3f} augmented_oracle={:.5f}/{:.5f}".format(
                variant_name,
                metrics["fallback_acc025"],
                metrics["fallback_acc050"],
                metrics["fixes025"], metrics["fixes050"],
                metrics["breaks025"], metrics["breaks050"],
                metrics["invalid_rate"],
                metrics["augmented_oracle_acc025"],
                metrics["augmented_oracle_acc050"],
            )
        )
    print(
        "Combined geometry oracle: Acc@0.25={:.5f} Acc@0.50={:.5f}".format(
            combined["acc025"], combined["acc050"]
        )
    )


def _run_audit_to_staging(args, output_dir, started_at):
    """Generate a complete frozen-checkpoint audit inside a staging path."""
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    train_cache = Path(args.train_cache).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError("checkpoint does not exist: {}".format(
            checkpoint_path
        ))
    if not train_cache.is_dir():
        raise ValueError("train cache does not exist: {}".format(train_cache))
    if not data_root.is_dir():
        raise ValueError("data root does not exist: {}".format(data_root))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    _ensure_project_imports()
    os.chdir(str(ROOT))
    from scripts.cache_scanrefer_rec_candidates import (
        _build_dataset,
        _load_frozen_model,
        _move_batch_to_device,
        _normalized_data_root,
        _prepare_model_config,
        checkpoint_sha256,
    )
    from models.rec_candidate_adapter import (
        attach_candidate_targets,
        build_rec_candidate_batch,
    )
    from models.rec_mask_geometry import (
        MASK_GEOMETRY_SCHEMA_VERSION,
        build_rec_mask_geometry_candidates,
    )
    from train_dist_mod import TrainTester

    fingerprint = checkpoint_sha256(checkpoint_path)
    manifest, panel_records = load_train_cache_panel_records(
        train_cache, expected_checkpoint_sha256=fingerprint
    )
    manifest_dataset = str(manifest.get("dataset", "scanrefer"))
    if manifest_dataset != args.dataset:
        raise ValueError(
            "train cache dataset {} does not match requested {}".format(
                manifest_dataset, args.dataset
            )
        )
    panel = select_baseline_stratified_panel(
        panel_records,
        scene_count=args.scene_count,
        expressions_per_scene=args.expressions_per_scene,
        seed=args.selection_seed,
    )
    dataset_indices = [row["dataset_index"] for row in panel]
    replay_groups = build_cache_replay_groups(
        dataset_indices,
        source_dataset_size=int(manifest["source_dataset_size"]),
        extraction_batch_size=args.cache_extraction_batch_size,
        replay_boundaries=args.cache_replay_boundaries,
    )
    panel_by_index = {row["dataset_index"]: row for row in panel}
    cached_rows = load_selected_cache_rows(
        train_cache, manifest, dataset_indices
    )
    common_provenance = {
        "panel_schema_version": PANEL_SCHEMA_VERSION,
        "dataset": args.dataset,
        "split": "train",
        "population_estimate": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": fingerprint,
        "checkpoint_epoch": int(manifest["checkpoint_epoch"]),
        "data_root": str(data_root),
        "train_cache": str(train_cache),
        "train_cache_manifest_sha256": _manifest_sha256(manifest),
        "cache_schema_version": manifest["cache_schema_version"],
        "feature_schema_version": manifest["feature_schema_version"],
        "candidate_rule": dict(manifest["candidate_rule"]),
        "cache_extraction_batch_size": args.cache_extraction_batch_size,
        "cache_replay_boundaries": list(args.cache_replay_boundaries),
        "cache_batch_provenance": (
            "recovered from Codex session command and shard resume mtimes; "
            "verified by candidate parity"
        ),
        "selection_mode": args.selection_mode,
        "selection_seed": args.selection_seed,
        "scene_count": args.scene_count,
        "expressions_per_scene": args.expressions_per_scene,
        "min_points": int(args.min_points),
        "max_point_fraction": float(args.max_point_fraction),
        "candidate_parity_tolerances": {
            "atol": PARITY_ATOL,
            "rtol": PARITY_RTOL,
            "feature_comparison": "finite diagnostic drift only",
            "score_comparison": "finite diagnostic drift only",
            "iou_atol": IOU_PARITY_ATOL,
            "iou_rtol": IOU_PARITY_RTOL,
        },
    }
    selection_payload = {
        "panel_schema_version": PANEL_SCHEMA_VERSION,
        "selection_mode": args.selection_mode,
        "selection_seed": args.selection_seed,
        "scene_count": args.scene_count,
        "expressions_per_scene": args.expressions_per_scene,
        "sample_count": len(panel),
        "replay_batch_count": len(replay_groups),
        "cache_extraction_batch_size": args.cache_extraction_batch_size,
        "cache_replay_boundaries": list(args.cache_replay_boundaries),
        "checkpoint_sha256": fingerprint,
        "train_cache": str(train_cache),
        "population_estimate": False,
        "bucket_counts": _panel_bucket_counts(panel),
        "rows": panel,
        "provenance": common_provenance,
    }
    print(
        "Selected {} train expressions from {} scenes: {}".format(
            len(panel), args.scene_count, selection_payload["bucket_counts"]
        ),
        flush=True,
    )

    _set_deterministic(args.selection_seed, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    config = _prepare_model_config(
        checkpoint, _normalized_data_root(data_root)
    )
    dataset = _build_dataset(config, "train", args.dataset)
    if max(dataset_indices) >= len(dataset):
        raise ValueError("selected dataset index exceeds current train dataset")
    replay_indices = {
        index for group in replay_groups for index in group["batch_indices"]
    }
    replay_scan_ids = {
        str(dataset.annos[index]["scan_id"]) for index in replay_indices
    }
    _prune_dataset_scenes(dataset, replay_scan_ids)
    model = _load_frozen_model(checkpoint, config, device)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch != int(manifest["checkpoint_epoch"]):
        raise ValueError("checkpoint epoch does not match train cache")
    del checkpoint
    loader = _build_replay_loader(dataset, replay_groups, args, device)

    output_rows = []
    variant_names = None
    variant_configs = None
    geometry_feature_names = None
    parity_max_differences = {}
    with torch.inference_mode():
        for batch_number, (replay_group, batch_data) in enumerate(
                zip(replay_groups, loader), start=1):
            batch_size = len(batch_data["scan_ids"])
            if batch_size != len(replay_group["batch_indices"]):
                raise RuntimeError("cache replay batch size changed")
            batch_data = _move_batch_to_device(batch_data, device)
            inputs = TrainTester._get_inputs(batch_data)
            inputs["train"] = False
            end_points = model(inputs)
            candidate_batch = build_rec_candidate_batch(
                end_points,
                inputs,
                topk_per_source=int(
                    manifest["candidate_rule"]["topk_per_source"]
                ),
                max_candidates=int(
                    manifest["candidate_rule"]["max_candidates"]
                ),
            )
            candidate_batch = attach_candidate_targets(
                candidate_batch, batch_data, root_only=True
            )
            positions = replay_group["selected_positions"]
            selected_indices = replay_group["selected_indices"]
            selected_batch_data = _select_batch_mapping(
                batch_data, positions, batch_size
            )
            selected_inputs = _select_batch_mapping(
                inputs, positions, batch_size
            )
            selected_end_points = _select_batch_mapping(
                end_points, positions, batch_size
            )
            selected_candidates = _select_batch_mapping(
                candidate_batch, positions, batch_size
            )
            batch_parity_differences = assert_candidate_cache_parity(
                selected_candidates,
                cached_rows,
                selected_indices,
                selected_batch_data["scan_ids"],
                selected_batch_data["target_id"],
            )
            for field, difference in batch_parity_differences.items():
                parity_max_differences[field] = max(
                    parity_max_differences.get(field, 0.0), difference
                )
            geometry = build_rec_mask_geometry_candidates(
                selected_end_points,
                selected_inputs,
                selected_candidates,
                variant_config={
                    "min_points": args.min_points,
                    "max_point_fraction": args.max_point_fraction,
                },
            )
            if geometry["schema_version"] != MASK_GEOMETRY_SCHEMA_VERSION:
                raise ValueError("mask geometry schema changed during audit")
            current_names = tuple(geometry["variant_names"])
            current_configs = tuple(geometry["variant_configs"])
            current_feature_names = tuple(geometry["geometry_feature_names"])
            if variant_names is None:
                variant_names = current_names
                variant_configs = current_configs
                geometry_feature_names = current_feature_names
            elif (variant_names != current_names
                  or variant_configs != current_configs
                  or geometry_feature_names != current_feature_names):
                raise ValueError("mask geometry schema changed between batches")
            geometry_ious = _compute_geometry_ious(
                geometry, selected_batch_data
            )
            default_positions = _fresh_default_positions(selected_candidates)
            diagnostics = extract_default_variant_diagnostics(
                geometry, default_positions
            )
            _append_compact_rows(
                output_rows,
                selected_indices,
                panel_by_index,
                selected_batch_data,
                selected_candidates,
                geometry,
                geometry_ious,
                default_positions,
                diagnostics,
            )
            print(
                "Audited {}/{} expressions (batch {})".format(
                    len(output_rows), len(panel), batch_number
                ),
                flush=True,
            )
            del (
                end_points, candidate_batch, selected_end_points,
                selected_inputs, selected_candidates, selected_batch_data,
                geometry, geometry_ious
            )

    if len(output_rows) != len(panel):
        raise RuntimeError("audit inference ended before the panel was complete")
    regressed_ious = _cat_row_tensors(output_rows, "regressed_ious")
    regressed_valid = _cat_row_tensors(output_rows, "candidate_valid")
    geometry_ious = _cat_row_tensors(output_rows, "geometry_ious")
    geometry_valid = _cat_row_tensors(output_rows, "geometry_valid")
    default_positions = torch.tensor([
        row["default_position"] for row in output_rows
    ], dtype=torch.long)
    summary = summarize_geometry_metrics(
        regressed_ious,
        regressed_valid,
        geometry_ious,
        geometry_valid,
        default_positions,
        variant_names,
    )
    variant_diagnostics = summarize_variant_diagnostics(
        _cat_row_tensors(output_rows, "rejection_codes"),
        _cat_row_tensors(output_rows, "default_variant_valid"),
        _cat_row_tensors(output_rows, "selected_point_fractions"),
        _cat_row_tensors(output_rows, "final_volume_ratios"),
        variant_names,
    )
    for name in variant_names:
        summary["variants"][name]["diagnostics"] = variant_diagnostics[name]
    final_provenance = dict(common_provenance)
    final_provenance.update({
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "variant_names": list(variant_names),
        "variant_configs": list(variant_configs),
        "geometry_feature_names": list(geometry_feature_names),
        "candidate_parity_max_abs_difference": parity_max_differences,
    })
    selection_payload["provenance"] = final_provenance
    summary.update({
        "panel_schema_version": PANEL_SCHEMA_VERSION,
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "population_estimate": False,
        "split": "train",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": fingerprint,
        "checkpoint_epoch": checkpoint_epoch,
        "train_cache": str(train_cache),
        "selection_mode": args.selection_mode,
        "selection_seed": args.selection_seed,
        "scene_count": args.scene_count,
        "expressions_per_scene": args.expressions_per_scene,
        "cache_extraction_batch_size": args.cache_extraction_batch_size,
        "replay_batch_count": len(replay_groups),
        "bucket_counts": _panel_bucket_counts(panel),
        "variant_names": list(variant_names),
        "variant_configs": list(variant_configs),
        "geometry_feature_names": list(geometry_feature_names),
        "min_points": int(args.min_points),
        "max_point_fraction": float(args.max_point_fraction),
        "elapsed_seconds": float(time.time() - started_at),
        "provenance": final_provenance,
    })
    rows_payload = {
        "panel_schema_version": PANEL_SCHEMA_VERSION,
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "checkpoint_sha256": fingerprint,
        "variant_names": variant_names,
        "variant_configs": variant_configs,
        "geometry_feature_names": geometry_feature_names,
        "provenance": final_provenance,
        "rows": output_rows,
    }
    _atomic_write_json(output_dir / "selection.json", selection_payload)
    _atomic_torch_save(output_dir / "rows.pt", rows_payload)
    _atomic_write_json(output_dir / "summary.json", summary)
    _print_summary(summary)
    return summary


def run_audit(args):
    """Run the audit and publish its artifacts only after complete success."""
    final_path, staging = _prepare_output_staging(
        args.output_dir, args.overwrite
    )
    try:
        summary = _run_audit_to_staging(args, staging, time.time())
        _publish_output_bundle(final_path, staging)
    finally:
        if staging.exists():
            shutil.rmtree(str(staging))
    print("Audit artifacts: {}".format(final_path), flush=True)
    return summary


def main(argv=None):
    run_audit(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
