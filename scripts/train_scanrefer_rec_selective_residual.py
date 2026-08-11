#!/usr/bin/env python
"""Train-only materialization and fitting for selective REC residuals."""

import argparse
import copy
import hashlib
import io
import itertools
import json
import math
import os
import random
import stat
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_


from models.rec_geometry_reranker import build_flat_parent_prior
from models.rec_selective_residual import (
    RESIDUAL_BREAK_COSTS,
    RESIDUAL_HIDDEN_DIMS,
    RESIDUAL_MARGIN_PERCENTILES,
    RESIDUAL_WEIGHT_DECAYS,
    SelectiveResidualModel,
    apply_selective_policy,
    build_residual_scene_folds,
    build_selective_pair_features,
    build_selective_pair_targets,
    canonical_scene_fold_sha256,
    choose_selective_configuration,
    compute_selective_residual_loss,
    expected_selective_gain,
    scene_clustered_hit_delta_bootstrap,
)
from scripts.rec_geometry_cache import canonical_json_sha256
from scripts.train_rec_geometry_reranker import (
    AUTHORITATIVE_SPLIT_SEED0,
    FLAT_GEOMETRY_CANDIDATE_COUNT,
    GEOMETRY_CANDIDATE_COUNT,
    GEOMETRY_INPUT_DIM,
    GEOMETRY_VARIANT_COUNT,
    PARENT_INFERENCE_LOCAL_BATCH_SIZE,
    _cached_parent_compact_scores,
    _disabled_parent_autocast,
    _stable_flat_top1_indices,
    _stable_rank_normalize_once,
    build_geometry_training_batch,
    load_geometry_reranker_artifact,
    load_geometry_training_data,
)
from scripts.train_rec_reranker import normalize_features


AUTHORITATIVE_BACKBONE_SHA256 = (
    "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
)
AUTHORITATIVE_PARENT_ARTIFACT_SHA256 = (
    "f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b"
)
AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256 = (
    "835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f"
)
AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256 = (
    "411ec7d5d80a7be9596de20b348667d529e6a8f568b8ab0c0e0922b8719f9045"
)
AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256 = (
    "2f099adb04823c8a4bdfb32040431c8b9150b6da39a617640c9b871f52ba3750"
)
AUTHORITATIVE_GEOMETRY_METADATA_SHA256 = (
    "6965b4a21daf52a25b7793e1df4fbff3ca26ed9e5db011cc847f2b601eb8c062"
)
FORBIDDEN_TRAIN_PATH_COMPONENTS = (
    "val", "validation", "official", "claim", "receipt",
)
RESIDUAL_MATERIALIZATION_BATCH_SIZE = 256
RESIDUAL_EPOCHS = 10
RESIDUAL_BATCH_SIZE = 256
RESIDUAL_LEARNING_RATE = 3e-4
RESIDUAL_GRAD_CLIP_NORM = 1.0
RESIDUAL_MODEL_SEED = 0
RESIDUAL_GAIN_QUANTILES = (
    0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0,
)
AUTHORITATIVE_BACKBONE_PATH = Path(
    "/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/"
    "mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth"
)
RESIDUAL_RECORD_FIELDS = (
    "dataset_index",
    "scan_id",
    "target_id",
    "pair_features",
    "pair_valid",
    "candidate_ious",
    "baseline_index",
    "baseline_scores",
    "query_positions",
    "variant_indices",
)


def _file_identity(stat_result):
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _stable_file_sha256(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("artifact does not exist: {}".format(path))
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()))
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = _file_identity(os.fstat(handle.fileno()))
        live = _file_identity(path.stat())
    except OSError as error:
        raise ValueError(
            "could not hash immutable artifact {}: {}".format(path, error)
        )
    if before != after or after != live:
        raise ValueError("immutable artifact changed while being hashed")
    return digest.hexdigest()


def _validate_train_only_path(path, name):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("{} path must be path-like".format(name))
    resolved = Path(path).expanduser().resolve()
    for component in resolved.parts:
        lowered = component.lower()
        if any(token in lowered for token in FORBIDDEN_TRAIN_PATH_COMPONENTS):
            raise ValueError(
                "{} path contains forbidden component: {}".format(
                    name, component
                )
            )
    return resolved


def _parent_model(parent):
    if (not isinstance(parent, (tuple, list)) or len(parent) != 2
            or not isinstance(parent[0], torch.nn.Module)
            or not isinstance(parent[1], dict)):
        raise ValueError("parent must contain a model and artifact")
    return parent[0]


def _require_authoritative_inputs(
        joined_rows, base_manifest, geometry_manifest, parent,
        geometry_model, geometry_artifact, parent_sha, geometry_sha):
    if not isinstance(joined_rows, list) or not joined_rows:
        raise ValueError("joined training rows must be a nonempty list")
    if not isinstance(base_manifest, dict):
        raise ValueError("base manifest must be an object")
    if not isinstance(geometry_manifest, dict):
        raise ValueError("geometry manifest must be an object")
    if (base_manifest.get("split") != "train"
            or geometry_manifest.get("split") != "train"):
        raise ValueError("residual fitting requires train manifests")
    if (base_manifest.get("sample_count") != len(joined_rows)
            or geometry_manifest.get("sample_count") != len(joined_rows)):
        raise ValueError("train manifest sample counts differ from joined rows")
    expected_manifest_values = {
        "backbone": base_manifest.get("checkpoint_sha256"),
        "base_cache_content": geometry_manifest.get(
            "base_cache_binding", {}
        ).get("content_sha256"),
        "geometry_cache_content": geometry_manifest.get(
            "cache_content_digest"
        ),
        "geometry_metadata": geometry_manifest.get(
            "immutable_metadata_digest"
        ),
    }
    expected = {
        "backbone": AUTHORITATIVE_BACKBONE_SHA256,
        "base_cache_content": AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
        "geometry_cache_content": AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256,
        "geometry_metadata": AUTHORITATIVE_GEOMETRY_METADATA_SHA256,
    }
    if expected_manifest_values != expected:
        raise ValueError("train cache provenance differs from authoritative input")
    parent_model = _parent_model(parent)
    if (parent_sha != AUTHORITATIVE_PARENT_ARTIFACT_SHA256
            or getattr(parent_model, "_artifact_sha256", None) != parent_sha):
        raise ValueError("parent artifact SHA-256 mismatch")
    if (geometry_sha != AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256
            or getattr(geometry_model, "_artifact_sha256", None)
            != geometry_sha):
        raise ValueError("geometry artifact SHA-256 mismatch")
    if (not isinstance(geometry_artifact, dict)
            or geometry_artifact.get("checkpoint_sha256")
            != AUTHORITATIVE_BACKBONE_SHA256
            or geometry_artifact.get("parent_artifact_sha256")
            != AUTHORITATIVE_PARENT_ARTIFACT_SHA256
            or float(geometry_artifact.get("geometry_weight", -1.0)) != 1.0):
        raise ValueError("geometry artifact provenance mismatch")


def load_residual_training_inputs(
        base_cache, geometry_cache, parent_artifact_path,
        geometry_artifact_path, device="cuda:0"):
    """Stable-load exactly the four approved train-only inputs."""
    base_path = _validate_train_only_path(base_cache, "base cache")
    geometry_path = _validate_train_only_path(
        geometry_cache, "geometry cache"
    )
    parent_path = _validate_train_only_path(
        parent_artifact_path, "parent artifact"
    )
    geometry_artifact_path = _validate_train_only_path(
        geometry_artifact_path, "geometry artifact"
    )
    parent_sha = _stable_file_sha256(parent_path)
    geometry_sha = _stable_file_sha256(geometry_artifact_path)
    if parent_sha != AUTHORITATIVE_PARENT_ARTIFACT_SHA256:
        raise ValueError("parent artifact file is not the approved snapshot")
    if geometry_sha != AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256:
        raise ValueError("geometry artifact file is not the approved snapshot")

    joined, base_manifest, geometry_manifest, parent = (
        load_geometry_training_data(base_path, geometry_path, parent_path)
    )
    geometry_model, geometry_artifact = load_geometry_reranker_artifact(
        geometry_artifact_path,
        device=device,
        parent_artifact_path=parent_path,
        base_manifest=base_manifest,
        geometry_manifest=geometry_manifest,
    )
    _require_authoritative_inputs(
        joined,
        base_manifest,
        geometry_manifest,
        parent,
        geometry_model,
        geometry_artifact,
        parent_sha,
        geometry_sha,
    )
    return {
        "joined_rows": joined,
        "base_manifest": base_manifest,
        "geometry_manifest": geometry_manifest,
        "parent": parent,
        "geometry_model": geometry_model,
        "geometry_artifact": geometry_artifact,
        "input_sha256": {
            "backbone": AUTHORITATIVE_BACKBONE_SHA256,
            "parent": parent_sha,
            "geometry": geometry_sha,
            "base_cache_content": AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
            "geometry_cache_content": (
                AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256
            ),
            "geometry_metadata": AUTHORITATIVE_GEOMETRY_METADATA_SHA256,
        },
        "validation_data_accessed": False,
    }


def _validate_joined_row_order(rows, require_contiguous=True):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("materialization rows must be a nonempty sequence")
    if not isinstance(require_contiguous, bool):
        raise TypeError("require_contiguous must be boolean")
    previous_index = -1
    for expected_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"base", "geometry"}:
            raise ValueError("joined row must contain base and geometry objects")
        base = row["base"]
        geometry = row["geometry"]
        if not isinstance(base, dict) or not isinstance(geometry, dict):
            raise ValueError("joined row members must be objects")
        for key in ("dataset_index", "target_id"):
            base_value = base.get(key)
            geometry_value = geometry.get(key)
            if (not isinstance(base_value, int) or isinstance(base_value, bool)
                    or base_value < 0 or base_value != geometry_value):
                raise ValueError("joined {} identity mismatch".format(key))
        dataset_index = base["dataset_index"]
        if dataset_index <= previous_index:
            raise ValueError(
                "materialization dataset indices must be strictly ordered"
            )
        if require_contiguous and dataset_index != expected_index:
            raise ValueError(
                "materialization dataset indices must be contiguous and ordered"
            )
        previous_index = dataset_index
        scan_id = base.get("scan_id")
        if (not isinstance(scan_id, str) or not scan_id
                or geometry.get("scan_id") != scan_id):
            raise ValueError("joined scan identity mismatch")


def _validate_materialization_artifact(artifact):
    if not isinstance(artifact, dict):
        raise ValueError("geometry artifact must be an object")
    mean = artifact.get("feature_mean")
    std = artifact.get("feature_std")
    names = artifact.get("feature_names")
    if (not isinstance(mean, torch.Tensor)
            or not isinstance(std, torch.Tensor)
            or mean.dtype != torch.float32
            or std.dtype != torch.float32
            or mean.device.type != "cpu"
            or std.device.type != "cpu"
            or mean.shape != (GEOMETRY_INPUT_DIM,)
            or std.shape != (GEOMETRY_INPUT_DIM,)
            or not bool(torch.isfinite(mean).all().item())
            or not bool(torch.isfinite(std).all().item())
            or bool((std <= 0.0).any().item())):
        raise ValueError("geometry normalization contract is invalid")
    if (not isinstance(names, list)
            or len(names) != GEOMETRY_INPUT_DIM
            or len(set(names)) != GEOMETRY_INPUT_DIM
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("geometry feature schema is invalid")
    if (artifact.get("input_dim") != GEOMETRY_INPUT_DIM
            or artifact.get("regressed_variant_index") != 0
            or artifact.get("checkpoint_sha256")
            != AUTHORITATIVE_BACKBONE_SHA256
            or artifact.get("parent_artifact_sha256")
            != AUTHORITATIVE_PARENT_ARTIFACT_SHA256):
        raise ValueError("geometry artifact contract is invalid")
    weight = artifact.get("geometry_weight")
    if (not isinstance(weight, (float, int)) or isinstance(weight, bool)
            or float(weight) != 1.0):
        raise ValueError("approved geometry weight must be exactly 1.0")


def _validate_geometry_outputs(outputs, batch_size, device, valid):
    if not isinstance(outputs, dict) or set(outputs) != {
            "ranking_logits", "threshold_logits", "iou_estimate"}:
        raise ValueError("geometry model output schema changed")
    expected_shapes = {
        "ranking_logits": (batch_size, FLAT_GEOMETRY_CANDIDATE_COUNT),
        "threshold_logits": (
            batch_size, FLAT_GEOMETRY_CANDIDATE_COUNT, 2
        ),
        "iou_estimate": (batch_size, FLAT_GEOMETRY_CANDIDATE_COUNT),
    }
    for name, value in outputs.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError("geometry {} must be a tensor".format(name))
        if (value.dtype != torch.float32 or value.device != device
                or tuple(value.shape) != expected_shapes[name]):
            raise ValueError("geometry {} layout changed".format(name))
        if not bool(torch.isfinite(value[valid]).all().item()):
            raise ValueError("valid geometry {} must be finite".format(name))


def _require_frozen_model(model, name):
    if model.training:
        raise RuntimeError("{} entered training mode".format(name))
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("{} acquired trainable parameters".format(name))
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("{} acquired gradients".format(name))


def materialize_residual_rows(
        rows, parent, geometry_model, geometry_artifact,
        batch_size=RESIDUAL_MATERIALIZATION_BATCH_SIZE, device="cuda:0",
        require_contiguous=True):
    """Materialize frozen deployable features plus detached train labels."""
    _validate_joined_row_order(
        rows, require_contiguous=require_contiguous
    )
    _validate_materialization_artifact(geometry_artifact)
    if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
            or batch_size <= 0):
        raise ValueError("materialization batch_size must be positive")
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("materialization supports only CPU and CUDA")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA materialization requested but unavailable")
    if not isinstance(geometry_model, torch.nn.Module):
        raise TypeError("geometry_model must be a module")

    parent_model = _parent_model(parent)
    parent_model.to(device=resolved_device, dtype=torch.float32)
    parent_model.eval().requires_grad_(False)
    _require_frozen_model(parent_model, "parent model")
    parent_artifact = parent[1]
    for start in range(0, len(rows), PARENT_INFERENCE_LOCAL_BATCH_SIZE):
        parent_rows = rows[start:start + PARENT_INFERENCE_LOCAL_BATCH_SIZE]
        _cached_parent_compact_scores(
            parent_model,
            parent_artifact,
            [row["base"] for row in parent_rows],
        )
    _require_frozen_model(parent_model, "parent model")
    geometry_model.to(device=resolved_device, dtype=torch.float32)
    geometry_model.eval().requires_grad_(False)
    _require_frozen_model(geometry_model, "geometry model")

    records = []
    with torch.no_grad(), _disabled_parent_autocast(resolved_device):
        for start in range(0, len(rows), batch_size):
            row_batch = rows[start:start + batch_size]
            batch = build_geometry_training_batch(row_batch, parent)
            if list(batch["feature_names"]) != geometry_artifact[
                    "feature_names"]:
                raise ValueError("live geometry feature schema changed")
            valid = batch["valid_mask"].to(resolved_device)
            raw_features = batch["features"].to(
                resolved_device, dtype=torch.float32
            )
            normalized = normalize_features(
                raw_features,
                valid,
                geometry_artifact["feature_mean"],
                geometry_artifact["feature_std"],
            )
            if not bool(torch.isfinite(normalized).all().item()):
                raise ValueError("normalized geometry features are non-finite")
            outputs = geometry_model(normalized, valid)
            _validate_geometry_outputs(
                outputs, len(row_batch), resolved_device, valid
            )
            _require_frozen_model(geometry_model, "geometry model")
            _require_frozen_model(parent_model, "parent model")

            parent_state = {
                key: value.to(resolved_device)
                if isinstance(value, torch.Tensor) else value
                for key, value in batch["parent_state"].items()
            }
            geometry_valid = valid.reshape(
                len(row_batch),
                GEOMETRY_CANDIDATE_COUNT,
                GEOMETRY_VARIANT_COUNT,
            )
            parent_prior = build_flat_parent_prior(
                parent_state,
                geometry_valid,
                geometry_artifact["regressed_variant_index"],
            )
            parent_rank = _stable_rank_normalize_once(parent_prior, valid)
            geometry_rank = _stable_rank_normalize_once(
                outputs["ranking_logits"], valid
            )
            weight = float(geometry_artifact["geometry_weight"])
            baseline_scores = (
                (1.0 - weight) * parent_rank + weight * geometry_rank
            ).masked_fill(~valid, -float("inf"))
            baseline_indices = _stable_flat_top1_indices(
                baseline_scores, valid
            )
            pair = build_selective_pair_features(
                normalized_features=normalized,
                valid_mask=valid,
                baseline_indices=baseline_indices,
                parent_rank=parent_rank,
                geometry_rank=geometry_rank,
                threshold_logits=outputs["threshold_logits"],
                iou_estimate=outputs["iou_estimate"],
                query_positions=batch["query_positions"].to(resolved_device),
            )
            candidate_ious = batch["candidate_ious"].to(
                dtype=torch.float32
            )
            for local_index, joined_row in enumerate(row_batch):
                base = joined_row["base"]
                records.append({
                    "dataset_index": int(base["dataset_index"]),
                    "scan_id": base["scan_id"],
                    "target_id": int(base["target_id"]),
                    "pair_features": pair["features"][
                        local_index
                    ].detach().cpu().clone(),
                    "pair_valid": pair["valid_mask"][
                        local_index
                    ].detach().cpu().clone(),
                    "candidate_ious": candidate_ious[
                        local_index
                    ].detach().cpu().clone(),
                    "baseline_index": int(
                        baseline_indices[local_index].item()
                    ),
                    "baseline_scores": baseline_scores[
                        local_index
                    ].detach().cpu().clone(),
                    "query_positions": batch["query_positions"][
                        local_index
                    ].detach().cpu().clone(),
                    "variant_indices": batch["variant_indices"][
                        local_index
                    ].detach().cpu().clone(),
                })
    _validate_residual_records(
        records, require_contiguous=require_contiguous
    )
    return records


def _require_cpu_tensor(value, name, shape, dtype):
    if (not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != dtype
            or tuple(value.shape) != shape):
        raise ValueError("residual record {} layout is invalid".format(name))


def _validate_residual_records(records, require_contiguous=True):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("residual records must be a nonempty sequence")
    previous_index = -1
    for expected_index, record in enumerate(records):
        if (not isinstance(record, dict)
                or set(record) != set(RESIDUAL_RECORD_FIELDS)):
            raise ValueError("residual record fields do not match schema")
        dataset_index = record["dataset_index"]
        if (not isinstance(dataset_index, int)
                or isinstance(dataset_index, bool)
                or dataset_index <= previous_index
                or (require_contiguous and dataset_index != expected_index)):
            raise ValueError(
                "residual record indices must be canonical and ordered"
            )
        previous_index = dataset_index
        if (not isinstance(record["scan_id"], str)
                or not record["scan_id"]
                or not isinstance(record["target_id"], int)
                or isinstance(record["target_id"], bool)
                or record["target_id"] < 0):
            raise ValueError("residual record identity is invalid")
        _require_cpu_tensor(
            record["pair_features"], "pair_features", (112, 185),
            torch.float32
        )
        _require_cpu_tensor(
            record["pair_valid"], "pair_valid", (112,), torch.bool
        )
        _require_cpu_tensor(
            record["candidate_ious"], "candidate_ious", (112,),
            torch.float32
        )
        _require_cpu_tensor(
            record["baseline_scores"], "baseline_scores", (112,),
            torch.float32
        )
        _require_cpu_tensor(
            record["query_positions"], "query_positions", (112,), torch.long
        )
        _require_cpu_tensor(
            record["variant_indices"], "variant_indices", (112,), torch.long
        )
        baseline = record["baseline_index"]
        if (not isinstance(baseline, int) or isinstance(baseline, bool)
                or not 0 <= baseline < 112
                or bool(record["pair_valid"][baseline].item())):
            raise ValueError("residual baseline index is invalid")
        candidate_valid = record["pair_valid"].clone()
        candidate_valid[baseline] = True
        if not bool(torch.isfinite(record["pair_features"]).all().item()):
            raise ValueError("residual pair features must be finite")
        if (not bool(torch.isfinite(record["candidate_ious"]).all().item())
                or bool((record["candidate_ious"] < 0.0).any().item())
                or bool((record["candidate_ious"] > 1.0).any().item())):
            raise ValueError("residual candidate IoUs must lie in [0,1]")
        scores = record["baseline_scores"]
        if (not bool(torch.isfinite(scores[candidate_valid]).all().item())
                or not bool(torch.isneginf(scores[~candidate_valid]).all().item())):
            raise ValueError("residual baseline score mask is invalid")
        if _stable_flat_top1_indices(
                scores.unsqueeze(0), candidate_valid.unsqueeze(0)
        ).item() != baseline:
            raise ValueError("residual baseline selection is inconsistent")
        if not torch.equal(
                record["pair_features"][~record["pair_valid"]],
                torch.zeros_like(
                    record["pair_features"][~record["pair_valid"]]
                )):
            raise ValueError("invalid residual pair features must be zero")


def _digest_int(digest, value):
    digest.update(int(value).to_bytes(8, "little", signed=False))


def _digest_string(digest, value):
    encoded = value.encode("utf-8")
    _digest_int(digest, len(encoded))
    digest.update(encoded)


def _digest_tensor(digest, value):
    digest.update(value.contiguous().numpy().tobytes(order="C"))


def canonical_residual_rows_sha256(records):
    """Hash every canonical identity, deployable feature, and train label."""
    _validate_residual_records(records, require_contiguous=False)
    digest = hashlib.sha256()
    for record in records:
        _digest_int(digest, record["dataset_index"])
        _digest_string(digest, record["scan_id"])
        _digest_int(digest, record["target_id"])
        _digest_int(digest, record["baseline_index"])
        for field in (
                "pair_features", "pair_valid", "candidate_ious",
                "baseline_scores", "query_positions", "variant_indices"):
            _digest_tensor(digest, record[field])
    return digest.hexdigest()


def canonical_selected_iou_sha256(records):
    """Hash ordered baseline-selected IoUs as raw float32 values."""
    _validate_residual_records(records, require_contiguous=False)
    values = torch.stack([
        record["candidate_ious"][record["baseline_index"]]
        for record in records
    ]).to(dtype=torch.float32)
    return hashlib.sha256(
        values.contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def _set_residual_seed(device):
    random.seed(RESIDUAL_MODEL_SEED)
    torch.manual_seed(RESIDUAL_MODEL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RESIDUAL_MODEL_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _candidate_valid(record):
    valid = record["pair_valid"].clone()
    valid[record["baseline_index"]] = True
    return valid


def summarize_residual_training_labels(records):
    """Count fixed break/neutral/fix targets by query relationship."""
    _validate_residual_records(records, require_contiguous=False)
    pair_valid = torch.stack([record["pair_valid"] for record in records])
    candidate_valid = torch.stack([
        _candidate_valid(record) for record in records
    ])
    candidate_ious = torch.stack([
        record["candidate_ious"] for record in records
    ])
    baseline_indices = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    targets = build_selective_pair_targets(
        candidate_ious,
        candidate_valid,
        baseline_indices,
    )
    query_positions = torch.stack([
        record["query_positions"] for record in records
    ])
    rows = torch.arange(len(records))
    baseline_queries = query_positions[rows, baseline_indices]
    same_query = pair_valid & query_positions.eq(
        baseline_queries.unsqueeze(1)
    )
    groups = {
        "all": pair_valid,
        "same_query": same_query,
        "different_query": pair_valid & ~same_query,
    }
    summary = {}
    for group_name, mask in groups.items():
        threshold_summary = {}
        for threshold_index, threshold in enumerate(("0.25", "0.50")):
            labels = targets[:, :, threshold_index][mask]
            counts = {
                "break": int(labels.eq(0).sum().item()),
                "neutral": int(labels.eq(1).sum().item()),
                "fix": int(labels.eq(2).sum().item()),
                "total": int(labels.numel()),
            }
            if sum(counts[name] for name in (
                    "break", "neutral", "fix")) != counts["total"]:
                raise RuntimeError("residual label counts do not reconcile")
            threshold_summary[threshold] = counts
        summary[group_name] = threshold_summary
    return summary


def _gain_distribution_statistics(values):
    if not int(values.numel()):
        return None
    ordered = values.sort().values
    quantiles = []
    for quantile in RESIDUAL_GAIN_QUANTILES:
        rank = int(math.ceil(quantile * len(ordered)))
        rank = min(max(rank, 1), len(ordered))
        quantiles.append({
            "quantile": float(quantile),
            "value": float(ordered[rank - 1].item()),
        })
    statistics = {
        "minimum": float(ordered[0].item()),
        "maximum": float(ordered[-1].item()),
        "mean": float(values.mean().item()),
        "population_standard_deviation": float(
            values.std(unbiased=False).item()
        ),
        "nearest_rank_quantiles": quantiles,
    }
    scalar_values = (
        statistics["minimum"],
        statistics["maximum"],
        statistics["mean"],
        statistics["population_standard_deviation"],
    )
    if (not all(math.isfinite(value) for value in scalar_values)
            or not all(math.isfinite(record["value"])
                       for record in quantiles)):
        raise ValueError("OOF gain statistics must be finite")
    return statistics


def summarize_oof_pair_gain(pair_gain, pair_valid):
    """Summarize valid and positive OOF gains on a fixed quantile grid."""
    if (not isinstance(pair_gain, torch.Tensor)
            or pair_gain.device.type != "cpu"
            or pair_gain.dtype != torch.float32
            or pair_gain.dim() != 2
            or pair_gain.shape[0] <= 0
            or pair_gain.shape[1] != 112):
        raise ValueError("OOF pair gain layout is invalid")
    if (not isinstance(pair_valid, torch.Tensor)
            or pair_valid.device.type != "cpu"
            or pair_valid.dtype != torch.bool
            or tuple(pair_valid.shape) != tuple(pair_gain.shape)):
        raise ValueError("OOF pair validity layout is invalid")
    valid_values = pair_gain[pair_valid]
    if not bool(torch.isfinite(valid_values).all().item()):
        raise ValueError("valid OOF pair gains must be finite")
    positive_values = valid_values[valid_values.gt(0.0)]
    return {
        "valid": {
            "count": int(valid_values.numel()),
            "statistics": _gain_distribution_statistics(valid_values),
        },
        "positive": {
            "count": int(positive_values.numel()),
            "statistics": _gain_distribution_statistics(positive_values),
        },
    }


def _packed_training_batch(records, device):
    if not records:
        raise ValueError("residual training batch cannot be empty")
    pair_valid = torch.stack([record["pair_valid"] for record in records])
    candidate_valid = torch.stack([
        _candidate_valid(record) for record in records
    ])
    baseline_indices = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    candidate_ious = torch.stack([
        record["candidate_ious"] for record in records
    ])
    full_targets = build_selective_pair_targets(
        candidate_ious,
        candidate_valid,
        baseline_indices,
    )
    alternative_counts = pair_valid.sum(dim=1)
    packed_count = max(int(alternative_counts.max().item()), 1)
    features = torch.zeros(
        len(records), packed_count, 185, dtype=torch.float32
    )
    valid = torch.zeros(len(records), packed_count, dtype=torch.bool)
    targets = torch.ones(
        len(records), packed_count, 2, dtype=torch.long
    )
    for row_index, record in enumerate(records):
        indices = record["pair_valid"].nonzero(
            as_tuple=False
        ).reshape(-1)
        count = int(indices.numel())
        if count:
            features[row_index, :count] = record["pair_features"][indices]
            valid[row_index, :count] = True
            targets[row_index, :count] = full_targets[row_index, indices]
    return (
        features.to(device),
        valid.to(device),
        targets.to(device),
    )


def _fit_residual_model(
        records, hidden_dim, weight_decay, break_cost, device,
        batch_observer=None, observer_context=None):
    if not records:
        raise ValueError("residual fit records cannot be empty")
    if batch_observer is not None and not callable(batch_observer):
        raise TypeError("batch_observer must be callable")
    observer_context = dict(observer_context or {})
    _set_residual_seed(device)
    model = SelectiveResidualModel(
        input_dim=185,
        hidden_dim=int(hidden_dim),
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=RESIDUAL_LEARNING_RATE,
        weight_decay=float(weight_decay),
    )
    shuffle_state = random.Random(RESIDUAL_MODEL_SEED)
    for epoch in range(RESIDUAL_EPOCHS):
        model.train()
        order = list(range(len(records)))
        shuffle_state.shuffle(order)
        for start in range(0, len(order), RESIDUAL_BATCH_SIZE):
            indices = order[start:start + RESIDUAL_BATCH_SIZE]
            row_batch = [records[index] for index in indices]
            if batch_observer is not None:
                event = dict(observer_context)
                event.update({
                    "epoch": epoch,
                    "scan_ids": tuple(
                        record["scan_id"] for record in row_batch
                    ),
                })
                batch_observer(event)
            pair_features, pair_valid, targets = _packed_training_batch(
                row_batch, device
            )
            logits = model(pair_features, pair_valid)
            loss, _stats = compute_selective_residual_loss(
                logits,
                targets,
                pair_valid,
                break_cost=float(break_cost),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), RESIDUAL_GRAD_CLIP_NORM)
            optimizer.step()
    model.eval().requires_grad_(False)
    return model


def _predict_pair_gain(model, records, device):
    if not records:
        raise ValueError("residual prediction records cannot be empty")
    model.to(device).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(records), RESIDUAL_BATCH_SIZE):
            row_batch = records[start:start + RESIDUAL_BATCH_SIZE]
            features = torch.stack([
                record["pair_features"] for record in row_batch
            ]).to(device)
            valid = torch.stack([
                record["pair_valid"] for record in row_batch
            ]).to(device)
            logits = model(features, valid)
            gain = expected_selective_gain(logits).masked_fill(~valid, 0.0)
            output.append(gain.detach().to(dtype=torch.float32).cpu())
    return torch.cat(output, dim=0)


def _oof_pair_gain_sha256(records, pair_gain):
    if (not isinstance(pair_gain, torch.Tensor)
            or pair_gain.dtype != torch.float32
            or pair_gain.device.type != "cpu"
            or tuple(pair_gain.shape) != (len(records), 112)):
        raise ValueError("OOF pair gain layout is invalid")
    digest = hashlib.sha256()
    for record, values in zip(records, pair_gain):
        _digest_int(digest, record["dataset_index"])
        _digest_string(digest, record["scan_id"])
        _digest_tensor(digest, values)
    return digest.hexdigest()


def _nearest_rank_positive_margin(pair_gain, pair_valid, percentile):
    positive = pair_gain[pair_valid & pair_gain.gt(0.0)]
    if not int(positive.numel()):
        return None
    ordered = positive.sort().values
    rank = int(math.ceil(float(percentile) / 100.0 * len(ordered)))
    rank = min(max(rank, 1), len(ordered))
    margin = float(ordered[rank - 1].item())
    if not math.isfinite(margin) or margin <= 0.0:
        raise RuntimeError("positive OOF margin is invalid")
    return margin


def _policy_candidate(records, pair_gain, config, percentile, margin):
    base_scores = torch.stack([
        record["baseline_scores"] for record in records
    ])
    pair_valid = torch.stack([record["pair_valid"] for record in records])
    if percentile is None:
        selected_indices = torch.tensor([
            record["baseline_index"] for record in records
        ], dtype=torch.long)
        switch_mask = torch.zeros(len(records), dtype=torch.bool)
    else:
        policy = apply_selective_policy(
            base_scores,
            pair_gain,
            pair_valid,
            margin,
        )
        selected_indices = policy["selected_indices"]
        switch_mask = policy["switch_mask"]
    baseline_indices = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    candidate_ious = torch.stack([
        record["candidate_ious"] for record in records
    ])
    rows = torch.arange(len(records))
    baseline_ious = candidate_ious[rows, baseline_indices]
    proposed_ious = candidate_ious[rows, selected_indices]
    return {
        "hidden_dim": config["hidden_dim"],
        "weight_decay": config["weight_decay"],
        "break_cost": config["break_cost"],
        "margin_percentile": percentile,
        "margin": margin,
        "scan_ids": [record["scan_id"] for record in records],
        "baseline_hits025": baseline_ious.gt(0.25).long().tolist(),
        "proposed_hits025": proposed_ious.gt(0.25).long().tolist(),
        "baseline_hits050": baseline_ious.gt(0.50).long().tolist(),
        "proposed_hits050": proposed_ious.gt(0.50).long().tolist(),
        "switch_bits": switch_mask.long().tolist(),
    }


def cross_fit_selective_residual(
        records, device="cpu", batch_observer=None,
        selector=choose_selective_configuration):
    """Produce scene-disjoint OOF gains for the fixed 12-model grid."""
    _validate_residual_records(records, require_contiguous=False)
    if not callable(selector):
        raise TypeError("selector must be callable")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA cross-fit requested but unavailable")
    scene_folds = build_residual_scene_folds([
        record["scan_id"] for record in records
    ])
    pair_valid = torch.stack([record["pair_valid"] for record in records])
    fold_training_labels = {}
    fold_pair_counts = {}
    for held_out_fold in range(5):
        fold_fit_records = [
            record for record in records
            if scene_folds[record["scan_id"]] != held_out_fold
        ]
        if not fold_fit_records:
            raise ValueError("every residual fold needs fit rows")
        labels = summarize_residual_training_labels(fold_fit_records)
        pair_count = sum(
            int(record["pair_valid"].sum().item())
            for record in fold_fit_records
        )
        if any(
                labels["all"][threshold]["total"] != pair_count
                for threshold in ("0.25", "0.50")):
            raise RuntimeError("fold training label count changed")
        fold_training_labels[held_out_fold] = labels
        fold_pair_counts[held_out_fold] = pair_count
    configurations = []
    policy_candidates = []
    grid = tuple(itertools.product(
        RESIDUAL_HIDDEN_DIMS,
        RESIDUAL_WEIGHT_DECAYS,
        RESIDUAL_BREAK_COSTS,
    ))
    for config_index, (hidden_dim, weight_decay, break_cost) in enumerate(grid):
        config = {
            "hidden_dim": int(hidden_dim),
            "weight_decay": float(weight_decay),
            "break_cost": float(break_cost),
        }
        oof_pair_gain = torch.zeros(len(records), 112, dtype=torch.float32)
        predicted = torch.zeros(len(records), dtype=torch.bool)
        fold_records = []
        for held_out_fold in range(5):
            fit_indices = [
                index for index, record in enumerate(records)
                if scene_folds[record["scan_id"]] != held_out_fold
            ]
            held_indices = [
                index for index, record in enumerate(records)
                if scene_folds[record["scan_id"]] == held_out_fold
            ]
            if not fit_indices or not held_indices:
                raise ValueError("every residual fold needs fit and held rows")
            fit_records = [records[index] for index in fit_indices]
            held_records = [records[index] for index in held_indices]
            model = _fit_residual_model(
                fit_records,
                hidden_dim=hidden_dim,
                weight_decay=weight_decay,
                break_cost=break_cost,
                device=resolved_device,
                batch_observer=batch_observer,
                observer_context={
                    "phase": "cross_fit",
                    "config_index": config_index,
                    "held_out_fold": held_out_fold,
                },
            )
            held_gain = _predict_pair_gain(
                model, held_records, resolved_device
            )
            oof_pair_gain[held_indices] = held_gain
            predicted[held_indices] = True
            fold_records.append({
                "fold": held_out_fold,
                "fit_scene_count": len({
                    record["scan_id"] for record in fit_records
                }),
                "fit_row_count": len(fit_records),
                "fit_pair_count": fold_pair_counts[held_out_fold],
                "held_scene_count": len({
                    record["scan_id"] for record in held_records
                }),
                "held_row_count": len(held_records),
                "training_labels": copy.deepcopy(
                    fold_training_labels[held_out_fold]
                ),
            })
            del model
        if not bool(predicted.all().item()):
            raise RuntimeError("OOF predictions do not cover every fit row")
        config_record = dict(config)
        config_record.update({
            "configuration_index": config_index,
            "folds": fold_records,
            "gain_summary": summarize_oof_pair_gain(
                oof_pair_gain, pair_valid
            ),
            "oof_pair_gain_sha256": _oof_pair_gain_sha256(
                records, oof_pair_gain
            ),
            "prediction_count": int(predicted.sum().item()),
        })
        configurations.append(config_record)
        policy_candidates.append(_policy_candidate(
            records, oof_pair_gain, config, None, float("inf")
        ))
        for percentile in RESIDUAL_MARGIN_PERCENTILES:
            margin = _nearest_rank_positive_margin(
                oof_pair_gain, pair_valid, percentile
            )
            if margin is not None:
                policy_candidates.append(_policy_candidate(
                    records, oof_pair_gain, config, percentile, margin
                ))
    choice = selector(policy_candidates)
    if not isinstance(choice, dict) or "eligible" not in choice:
        raise ValueError("selector returned an invalid choice")
    return {
        "scene_folds": scene_folds,
        "scene_fold_sha256": canonical_scene_fold_sha256(scene_folds),
        "configurations": configurations,
        "policy_candidate_count": len(policy_candidates),
        "choice": choice,
    }


@dataclass(frozen=True)
class CalibrationGateResult:
    passed: bool
    failures: tuple
    required_hits025: int
    required_hits050: int
    observed_hits025: int
    observed_hits050: int


def calibration_gate(metrics, baseline):
    """Apply the fixed offline cache gate without reinterpretation."""
    if not isinstance(metrics, dict) or not isinstance(baseline, dict):
        raise TypeError("calibration metrics and baseline must be mappings")
    metric_fields = {
        "sample_count",
        "hits025",
        "hits050",
        "baseline_hits025",
        "baseline_hits050",
        "oracle_hits025",
        "oracle_hits050",
        "candidate_iou_sha256",
        "row_materialization_sha256",
    }
    baseline_fields = {
        "sample_count",
        "hits025",
        "hits050",
        "oracle_hits025",
        "oracle_hits050",
        "candidate_iou_sha256",
        "row_materialization_sha256",
    }
    if not metric_fields.issubset(metrics) or not baseline_fields.issubset(
            baseline):
        raise ValueError("calibration gate fields are incomplete")
    failures = []
    if metrics["hits025"] < 3524:
        failures.append("hits025")
    if metrics["hits050"] < 3315:
        failures.append("hits050")
    comparisons = (
        ("sample_count", "sample_count"),
        ("baseline_hits025", "hits025"),
        ("baseline_hits050", "hits050"),
        ("oracle_hits025", "oracle_hits025"),
        ("oracle_hits050", "oracle_hits050"),
        ("candidate_iou_sha256", "candidate_iou_sha256"),
        ("row_materialization_sha256", "row_materialization_sha256"),
    )
    for metric_name, baseline_name in comparisons:
        if metrics[metric_name] != baseline[baseline_name]:
            failures.append(metric_name)
    if baseline["sample_count"] == 3625:
        authoritative = {
            "hits025": 3461,
            "hits050": 3315,
            "oracle_hits025": 3606,
            "oracle_hits050": 3588,
        }
        for name, expected in authoritative.items():
            if baseline[name] != expected:
                failures.append("authoritative_" + name)
    return CalibrationGateResult(
        passed=not failures,
        failures=tuple(failures),
        required_hits025=3524,
        required_hits050=3315,
        observed_hits025=int(metrics["hits025"]),
        observed_hits050=int(metrics["hits050"]),
    )


def refit_selective_residual(
        records, choice, device="cpu", batch_observer=None):
    """Refit the OOF-selected configuration once on every fit scene."""
    _validate_residual_records(records, require_contiguous=False)
    if (not isinstance(choice, dict) or choice.get("eligible") is not True):
        raise ValueError("refit requires one eligible OOF choice")
    for name, allowed in (
            ("hidden_dim", RESIDUAL_HIDDEN_DIMS),
            ("weight_decay", RESIDUAL_WEIGHT_DECAYS),
            ("break_cost", RESIDUAL_BREAK_COSTS)):
        if choice.get(name) not in allowed:
            raise ValueError("refit choice {} is outside the grid".format(name))
    margin = choice.get("margin")
    percentile = choice.get("margin_percentile")
    if (not isinstance(margin, (float, int)) or isinstance(margin, bool)
            or not math.isfinite(float(margin)) or float(margin) <= 0.0
            or percentile not in RESIDUAL_MARGIN_PERCENTILES):
        raise ValueError("refit choice has an invalid intervention gate")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA refit requested but unavailable")
    return _fit_residual_model(
        records,
        hidden_dim=choice["hidden_dim"],
        weight_decay=choice["weight_decay"],
        break_cost=choice["break_cost"],
        device=resolved_device,
        batch_observer=batch_observer,
        observer_context={"phase": "refit"},
    )


def _canonical_candidate_iou_sha256(records):
    _validate_residual_records(records, require_contiguous=False)
    digest = hashlib.sha256()
    for record in records:
        _digest_int(digest, record["dataset_index"])
        _digest_tensor(digest, record["candidate_ious"])
        _digest_tensor(digest, record["pair_valid"])
        _digest_int(digest, record["baseline_index"])
    return digest.hexdigest()


def _selected_iou_sha256(candidate_ious, selected_indices):
    rows = torch.arange(candidate_ious.shape[0])
    values = candidate_ious[rows, selected_indices].to(torch.float32)
    return hashlib.sha256(
        values.contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def build_cache_calibration_baseline(records):
    """Build frozen cache counts and digests before any residual switch."""
    _validate_residual_records(records, require_contiguous=False)
    candidate_ious = torch.stack([
        record["candidate_ious"] for record in records
    ])
    baseline_indices = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    valid = torch.stack([_candidate_valid(record) for record in records])
    rows = torch.arange(len(records))
    baseline_ious = candidate_ious[rows, baseline_indices]
    oracle_ious = candidate_ious.masked_fill(~valid, -float("inf")).max(
        dim=1
    ).values
    return {
        "sample_count": len(records),
        "hits025": int(baseline_ious.gt(0.25).sum().item()),
        "hits050": int(baseline_ious.gt(0.50).sum().item()),
        "oracle_hits025": int(oracle_ious.gt(0.25).sum().item()),
        "oracle_hits050": int(oracle_ious.gt(0.50).sum().item()),
        "candidate_iou_sha256": _canonical_candidate_iou_sha256(records),
        "row_materialization_sha256": canonical_residual_rows_sha256(records),
        "baseline_selected_iou_sha256": canonical_selected_iou_sha256(
            records
        ),
    }


def evaluate_selective_residual_policy(
        model, records, margin, device="cpu"):
    """Evaluate one fixed residual once and emit cache-only diagnostics."""
    _validate_residual_records(records, require_contiguous=False)
    if (not isinstance(margin, (float, int)) or isinstance(margin, bool)
            or not math.isfinite(float(margin)) or float(margin) <= 0.0):
        raise ValueError("evaluation margin must be positive and finite")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")
    pair_gain = _predict_pair_gain(model, records, resolved_device)
    base_scores = torch.stack([
        record["baseline_scores"] for record in records
    ])
    pair_valid = torch.stack([record["pair_valid"] for record in records])
    policy = apply_selective_policy(
        base_scores, pair_gain, pair_valid, float(margin)
    )
    baseline_indices = policy["baseline_indices"]
    selected_indices = policy["selected_indices"]
    switch_mask = policy["switch_mask"]
    candidate_ious = torch.stack([
        record["candidate_ious"] for record in records
    ])
    candidate_valid = torch.stack([
        _candidate_valid(record) for record in records
    ])
    rows = torch.arange(len(records))
    baseline_ious = candidate_ious[rows, baseline_indices]
    selected_ious = candidate_ious[rows, selected_indices]
    oracle_ious = candidate_ious.masked_fill(
        ~candidate_valid, -float("inf")
    ).max(dim=1).values
    baseline025 = baseline_ious.gt(0.25)
    baseline050 = baseline_ious.gt(0.50)
    selected025 = selected_ious.gt(0.25)
    selected050 = selected_ious.gt(0.50)
    scan_ids = [record["scan_id"] for record in records]
    bootstrap025 = scene_clustered_hit_delta_bootstrap(
        scan_ids, baseline025.long().tolist(), selected025.long().tolist()
    )
    bootstrap050 = scene_clustered_hit_delta_bootstrap(
        scan_ids, baseline050.long().tolist(), selected050.long().tolist()
    )
    per_scene = {
        scan_id: {"hits025": 0, "hits050": 0}
        for scan_id in sorted(set(scan_ids))
    }
    for row_index, scan_id in enumerate(scan_ids):
        per_scene[scan_id]["hits025"] += int(
            selected025[row_index].item() - baseline025[row_index].item()
        )
        per_scene[scan_id]["hits050"] += int(
            selected050[row_index].item() - baseline050[row_index].item()
        )
    query_positions = torch.stack([
        record["query_positions"] for record in records
    ])
    variant_indices = torch.stack([
        record["variant_indices"] for record in records
    ])
    baseline_query = query_positions[rows, baseline_indices]
    selected_query = query_positions[rows, selected_indices]
    baseline_variant = variant_indices[rows, baseline_indices]
    selected_variant = variant_indices[rows, selected_indices]
    fixes025 = selected025 & ~baseline025
    fixes050 = selected050 & ~baseline050
    recoverable025 = ~baseline025 & oracle_ious.gt(0.25)
    recovered025 = fixes025 & recoverable025
    recoverable_count = int(recoverable025.sum().item())
    recovered_count = int(recovered025.sum().item())
    baseline = build_cache_calibration_baseline(records)
    return {
        "sample_count": len(records),
        "hits025": int(selected025.sum().item()),
        "hits050": int(selected050.sum().item()),
        "baseline_hits025": int(baseline025.sum().item()),
        "baseline_hits050": int(baseline050.sum().item()),
        "oracle_hits025": int(oracle_ious.gt(0.25).sum().item()),
        "oracle_hits050": int(oracle_ious.gt(0.50).sum().item()),
        "fixes025": int(fixes025.sum().item()),
        "breaks025": int((~selected025 & baseline025).sum().item()),
        "fixes050": int(fixes050.sum().item()),
        "breaks050": int((~selected050 & baseline050).sum().item()),
        "switches": int(switch_mask.sum().item()),
        "abstentions": int((~switch_mask).sum().item()),
        "switch_rate": float(switch_mask.float().mean().item()),
        "wrong_query_recoveries025": int(
            (fixes025 & selected_query.ne(baseline_query)).sum().item()
        ),
        "wrong_variant_recoveries025": int((
            fixes025
            & selected_query.eq(baseline_query)
            & selected_variant.ne(baseline_variant)
        ).sum().item()),
        "wrong_query_recoveries050": int(
            (fixes050 & selected_query.ne(baseline_query)).sum().item()
        ),
        "wrong_variant_recoveries050": int((
            fixes050
            & selected_query.eq(baseline_query)
            & selected_variant.ne(baseline_variant)
        ).sum().item()),
        "recoverable_misses025": recoverable_count,
        "recovered_misses025": recovered_count,
        "recoverable_miss_coverage025": (
            recovered_count / float(recoverable_count)
            if recoverable_count else 0.0
        ),
        "bootstrap025": bootstrap025,
        "bootstrap050": bootstrap050,
        "per_scene_deltas": per_scene,
        "candidate_iou_sha256": baseline["candidate_iou_sha256"],
        "row_materialization_sha256": baseline[
            "row_materialization_sha256"
        ],
        "baseline_selected_iou_sha256": baseline[
            "baseline_selected_iou_sha256"
        ],
        "selected_iou_sha256": _selected_iou_sha256(
            candidate_ious, selected_indices
        ),
        "pair_gain_sha256": _oof_pair_gain_sha256(records, pair_gain),
    }


def canonical_residual_joined_identity_sha256(joined_rows):
    """Hash ordered dataset/scene identities without reading train labels."""
    if not isinstance(joined_rows, (list, tuple)) or not joined_rows:
        raise ValueError("joined residual rows must be a nonempty sequence")
    digest = hashlib.sha256()
    previous_index = -1
    for row in joined_rows:
        if not isinstance(row, dict) or set(row) != {"base", "geometry"}:
            raise ValueError("joined row must contain base and geometry objects")
        base = row["base"]
        geometry = row["geometry"]
        if not isinstance(base, dict) or not isinstance(geometry, dict):
            raise ValueError("joined row members must be objects")
        dataset_index = base.get("dataset_index")
        if (not isinstance(dataset_index, int)
                or isinstance(dataset_index, bool)
                or dataset_index <= previous_index
                or geometry.get("dataset_index") != dataset_index):
            raise ValueError("joined residual dataset identity changed")
        scan_id = base.get("scan_id")
        if (not isinstance(scan_id, str) or not scan_id
                or geometry.get("scan_id") != scan_id):
            raise ValueError("joined residual scan identity changed")
        _digest_int(digest, dataset_index)
        _digest_string(digest, scan_id)
        previous_index = dataset_index
    return digest.hexdigest()


def split_residual_joined_rows(joined_rows):
    """Split authoritative joined identities before feature materialization."""
    if not isinstance(joined_rows, (list, tuple)) or not joined_rows:
        raise ValueError("joined residual rows must be a nonempty sequence")
    expected_sample_count = AUTHORITATIVE_SPLIT_SEED0["sample_count"]
    if len(joined_rows) != expected_sample_count:
        raise ValueError("authoritative joined residual row count changed")
    scene_ids = []
    for expected_index, row in enumerate(joined_rows):
        if not isinstance(row, dict) or set(row) != {"base", "geometry"}:
            raise ValueError("joined row must contain base and geometry objects")
        base = row["base"]
        geometry = row["geometry"]
        if not isinstance(base, dict) or not isinstance(geometry, dict):
            raise ValueError("joined row members must be objects")
        base_index = base.get("dataset_index")
        geometry_index = geometry.get("dataset_index")
        if (not isinstance(base_index, int) or isinstance(base_index, bool)
                or base_index != expected_index
                or geometry_index != base_index):
            raise ValueError("joined residual dataset identity changed")
        scan_id = base.get("scan_id")
        if (not isinstance(scan_id, str) or not scan_id
                or geometry.get("scan_id") != scan_id):
            raise ValueError("joined residual scan identity changed")
        scene_ids.append(scan_id)

    scenes = sorted(set(scene_ids))
    if len(scenes) != AUTHORITATIVE_SPLIT_SEED0["scene_count"]:
        raise ValueError("authoritative joined residual scene count changed")
    shuffled = list(scenes)
    random.Random(0).shuffle(shuffled)
    calibration_count = int(round(len(scenes) * 0.10))
    calibration_scenes = set(shuffled[:calibration_count])
    fit_rows = [
        row for row, scan_id in zip(joined_rows, scene_ids)
        if scan_id not in calibration_scenes
    ]
    calibration_rows = [
        row for row, scan_id in zip(joined_rows, scene_ids)
        if scan_id in calibration_scenes
    ]
    fit_scenes = sorted(set(scenes).difference(calibration_scenes))
    calibration_scene_list = sorted(calibration_scenes)
    metadata = {
        "split_seed": 0,
        "calibration_fraction": 0.10,
        "scene_count": len(scenes),
        "fit_scene_count": len(fit_scenes),
        "calibration_scene_count": len(calibration_scene_list),
        "sample_count": len(joined_rows),
        "fit_sample_count": len(fit_rows),
        "calibration_sample_count": len(calibration_rows),
        "fit_scene_sha256": canonical_json_sha256(fit_scenes),
        "calibration_scene_sha256": canonical_json_sha256(
            calibration_scene_list
        ),
        "mapping_sha256": canonical_json_sha256({
            "fit": fit_scenes,
            "calibration": calibration_scene_list,
        }),
    }
    if metadata != AUTHORITATIVE_SPLIT_SEED0:
        raise ValueError("authoritative residual joined-row split changed")
    return {
        "fit_rows": fit_rows,
        "calibration_rows": calibration_rows,
        "metadata": metadata,
    }


def split_residual_records(records):
    """Apply the immutable seed-0 90/10 scene split to canonical records."""
    _validate_residual_records(records)
    scene_ids = [record["scan_id"] for record in records]
    scenes = sorted(set(scene_ids))
    if len(scenes) < 2:
        raise ValueError("residual split requires at least two scenes")
    shuffled = list(scenes)
    random.Random(0).shuffle(shuffled)
    calibration_count = int(round(len(scenes) * 0.10))
    calibration_count = max(1, min(calibration_count, len(scenes) - 1))
    calibration_scenes = set(shuffled[:calibration_count])
    fit_records = [
        record for record in records
        if record["scan_id"] not in calibration_scenes
    ]
    calibration_records = [
        record for record in records
        if record["scan_id"] in calibration_scenes
    ]
    fit_scenes = sorted({record["scan_id"] for record in fit_records})
    calibration_scene_list = sorted(calibration_scenes)
    metadata = {
        "split_seed": 0,
        "calibration_fraction": 0.10,
        "scene_count": len(scenes),
        "fit_scene_count": len(fit_scenes),
        "calibration_scene_count": len(calibration_scene_list),
        "sample_count": len(records),
        "fit_sample_count": len(fit_records),
        "calibration_sample_count": len(calibration_records),
        "fit_scene_sha256": canonical_json_sha256(fit_scenes),
        "calibration_scene_sha256": canonical_json_sha256(
            calibration_scene_list
        ),
        "mapping_sha256": canonical_json_sha256({
            "fit": fit_scenes,
            "calibration": calibration_scene_list,
        }),
    }
    if len(records) == 36665 and metadata != AUTHORITATIVE_SPLIT_SEED0:
        raise ValueError("authoritative residual scene split changed")
    return {
        "fit_records": fit_records,
        "calibration_records": calibration_records,
        "metadata": metadata,
    }


def build_selective_pair_feature_names(normalized_feature_names):
    """Name the fixed 185 alternative-minus-baseline feature coordinates."""
    if (not isinstance(normalized_feature_names, (list, tuple))
            or len(normalized_feature_names) != 179
            or len(set(normalized_feature_names)) != 179
            or any(not isinstance(name, str) or not name
                   for name in normalized_feature_names)):
        raise ValueError("normalized feature names must contain 179 names")
    names = ["delta:" + name for name in normalized_feature_names]
    names.extend([
        "delta:parent_rank",
        "delta:geometry_rank",
        "delta:threshold_probability_025",
        "delta:threshold_probability_050",
        "delta:iou_estimate",
        "same_query",
    ])
    if len(names) != 185 or len(set(names)) != 185:
        raise RuntimeError("selective pair feature names are not unique")
    return names


_RESIDUAL_ARTIFACT_FIELDS = {
    "schema",
    "version",
    "deployable",
    "validation_data_accessed",
    "input_sha256",
    "input_dim",
    "feature_names",
    "thresholds",
    "head_weights",
    "class_names",
    "model_config",
    "model_state_dict",
    "training_contract",
    "selection",
    "scene_folds",
    "scene_fold_sha256",
    "row_materialization_sha256",
    "oof_record",
    "calibration_record",
    "calibration_baseline",
}
_ARTIFACT_SELECTION_FIELDS = {
    "eligible",
    "selected",
    "hidden_dim",
    "weight_decay",
    "break_cost",
    "margin_percentile",
    "margin",
    "switches",
    "delta_hits025",
    "delta_hits050",
}
_INPUT_SHA_FIELDS = {
    "backbone",
    "parent",
    "geometry",
    "base_cache_content",
    "geometry_cache_content",
    "geometry_metadata",
}


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _artifact_selection(selection):
    if not isinstance(selection, dict):
        raise TypeError("selection must be a mapping")
    missing = _ARTIFACT_SELECTION_FIELDS.difference(selection)
    if missing:
        raise ValueError("selection is missing artifact fields")
    return {
        key: copy.deepcopy(selection[key])
        for key in sorted(_ARTIFACT_SELECTION_FIELDS)
    }


def build_selective_residual_artifact(
        model, selection, scene_folds, feature_names, input_sha256,
        row_materialization_sha256, oof_record, calibration_record,
        calibration_baseline):
    """Build one cache-calibrated but explicitly nondeployable artifact."""
    if not isinstance(model, SelectiveResidualModel):
        raise TypeError("artifact model must be SelectiveResidualModel")
    selection_record = _artifact_selection(selection)
    if (selection_record["eligible"] is not True
            or selection_record["selected"] != "residual"):
        raise ValueError("artifact requires an eligible residual selection")
    if model.hidden_dim != selection_record["hidden_dim"]:
        raise ValueError("artifact model differs from selected hidden_dim")
    if not isinstance(input_sha256, dict):
        raise TypeError("input_sha256 must be a mapping")
    gate = calibration_gate(calibration_record, calibration_baseline)
    if not gate.passed:
        raise ValueError("cache calibration gate did not pass")
    artifact = {
        "schema": "rec-selective-residual-v1",
        "version": 1,
        "deployable": False,
        "validation_data_accessed": False,
        "input_sha256": copy.deepcopy(input_sha256),
        "input_dim": 185,
        "feature_names": list(feature_names),
        "thresholds": [0.25, 0.50],
        "head_weights": [2.0, 1.0],
        "class_names": ["break", "neutral", "fix"],
        "model_config": {
            "input_dim": 185,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout,
        },
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "training_contract": {
            "seed": RESIDUAL_MODEL_SEED,
            "epochs": RESIDUAL_EPOCHS,
            "batch_size": RESIDUAL_BATCH_SIZE,
            "learning_rate": RESIDUAL_LEARNING_RATE,
            "gradient_clip_norm": RESIDUAL_GRAD_CLIP_NORM,
            "weight_decay": float(selection_record["weight_decay"]),
            "break_cost": float(selection_record["break_cost"]),
            "selected_margin": float(selection_record["margin"]),
            "selected_margin_percentile": float(
                selection_record["margin_percentile"]
            ),
            "hidden_dim_grid": list(RESIDUAL_HIDDEN_DIMS),
            "weight_decay_grid": list(RESIDUAL_WEIGHT_DECAYS),
            "break_cost_grid": list(RESIDUAL_BREAK_COSTS),
            "margin_percentile_grid": list(RESIDUAL_MARGIN_PERCENTILES),
        },
        "selection": selection_record,
        "scene_folds": copy.deepcopy(scene_folds),
        "scene_fold_sha256": canonical_scene_fold_sha256(scene_folds),
        "row_materialization_sha256": row_materialization_sha256,
        "oof_record": copy.deepcopy(oof_record),
        "calibration_record": copy.deepcopy(calibration_record),
        "calibration_baseline": copy.deepcopy(calibration_baseline),
    }
    validate_selective_residual_artifact(
        artifact,
        AUTHORITATIVE_BACKBONE_SHA256,
        AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
        feature_names,
    )
    return artifact


def validate_selective_residual_artifact(
        artifact, expected_backbone_sha256, expected_parent_sha256,
        expected_geometry_sha256, expected_feature_names):
    """Validate the exact cache-calibrated residual artifact contract."""
    if not isinstance(artifact, dict) or set(artifact) != _RESIDUAL_ARTIFACT_FIELDS:
        raise ValueError("selective residual artifact fields differ from schema")
    if (artifact.get("schema") != "rec-selective-residual-v1"
            or type(artifact.get("version")) is not int
            or artifact["version"] != 1
            or artifact.get("deployable") is not False
            or artifact.get("validation_data_accessed") is not False):
        raise ValueError("selective residual artifact top-level policy is invalid")
    for name, value in (
            ("backbone", expected_backbone_sha256),
            ("parent", expected_parent_sha256),
            ("geometry", expected_geometry_sha256)):
        if not _is_sha256(value):
            raise ValueError("expected {} SHA-256 is invalid".format(name))
    inputs = artifact.get("input_sha256")
    if (not isinstance(inputs, dict) or set(inputs) != _INPUT_SHA_FIELDS
            or any(not _is_sha256(value) for value in inputs.values())):
        raise ValueError("artifact input SHA-256 schema is invalid")
    expected_inputs = {
        "backbone": expected_backbone_sha256,
        "parent": expected_parent_sha256,
        "geometry": expected_geometry_sha256,
        "base_cache_content": AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
        "geometry_cache_content": AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256,
        "geometry_metadata": AUTHORITATIVE_GEOMETRY_METADATA_SHA256,
    }
    for name, expected in expected_inputs.items():
        if inputs.get(name) != expected:
            raise ValueError("artifact {} provenance mismatch".format(name))
    if (not isinstance(expected_feature_names, (list, tuple))
            or len(expected_feature_names) != 185
            or artifact.get("feature_names") != list(expected_feature_names)
            or artifact.get("input_dim") != 185
            or artifact.get("thresholds") != [0.25, 0.50]
            or artifact.get("head_weights") != [2.0, 1.0]
            or artifact.get("class_names") != ["break", "neutral", "fix"]):
        raise ValueError("artifact feature or head schema is invalid")
    config = artifact.get("model_config")
    if (not isinstance(config, dict)
            or set(config) != {"input_dim", "hidden_dim", "dropout"}
            or config.get("input_dim") != 185
            or config.get("hidden_dim") not in RESIDUAL_HIDDEN_DIMS
            or config.get("dropout") != 0.1):
        raise ValueError("artifact model config is invalid")
    selection = artifact.get("selection")
    if (not isinstance(selection, dict)
            or set(selection) != _ARTIFACT_SELECTION_FIELDS
            or selection.get("eligible") is not True
            or selection.get("selected") != "residual"
            or selection.get("hidden_dim") != config["hidden_dim"]
            or selection.get("weight_decay") not in RESIDUAL_WEIGHT_DECAYS
            or selection.get("break_cost") not in RESIDUAL_BREAK_COSTS
            or selection.get("margin_percentile")
            not in RESIDUAL_MARGIN_PERCENTILES
            or not isinstance(selection.get("margin"), (float, int))
            or isinstance(selection.get("margin"), bool)
            or not math.isfinite(float(selection["margin"]))
            or float(selection["margin"]) <= 0.0
            or type(selection.get("switches")) is not int
            or selection["switches"] < 0
            or type(selection.get("delta_hits025")) is not int
            or type(selection.get("delta_hits050")) is not int):
        raise ValueError("artifact selection contract is invalid")
    training = artifact.get("training_contract")
    expected_training = {
        "seed": RESIDUAL_MODEL_SEED,
        "epochs": RESIDUAL_EPOCHS,
        "batch_size": RESIDUAL_BATCH_SIZE,
        "learning_rate": RESIDUAL_LEARNING_RATE,
        "gradient_clip_norm": RESIDUAL_GRAD_CLIP_NORM,
        "weight_decay": float(selection["weight_decay"]),
        "break_cost": float(selection["break_cost"]),
        "selected_margin": float(selection["margin"]),
        "selected_margin_percentile": float(selection["margin_percentile"]),
        "hidden_dim_grid": list(RESIDUAL_HIDDEN_DIMS),
        "weight_decay_grid": list(RESIDUAL_WEIGHT_DECAYS),
        "break_cost_grid": list(RESIDUAL_BREAK_COSTS),
        "margin_percentile_grid": list(RESIDUAL_MARGIN_PERCENTILES),
    }
    if training != expected_training:
        raise ValueError("artifact training contract is invalid")
    scene_folds = artifact.get("scene_folds")
    if (not isinstance(scene_folds, dict)
            or canonical_scene_fold_sha256(scene_folds)
            != artifact.get("scene_fold_sha256")):
        raise ValueError("artifact scene fold binding is invalid")
    if not _is_sha256(artifact.get("row_materialization_sha256")):
        raise ValueError("artifact row materialization SHA-256 is invalid")
    oof = artifact.get("oof_record")
    if (not isinstance(oof, dict)
            or set(oof) != {
                "prediction_count", "pair_gain_sha256",
                "delta_hits025", "delta_hits050"
            }
            or type(oof.get("prediction_count")) is not int
            or oof["prediction_count"] <= 0
            or not _is_sha256(oof.get("pair_gain_sha256"))
            or oof.get("delta_hits025") != selection["delta_hits025"]
            or oof.get("delta_hits050") != selection["delta_hits050"]):
        raise ValueError("artifact OOF record is invalid")
    gate = calibration_gate(
        artifact.get("calibration_record"),
        artifact.get("calibration_baseline"),
    )
    if not gate.passed:
        raise ValueError("artifact calibration gate is not satisfied")
    state = artifact.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("artifact model state must be a mapping")
    model = SelectiveResidualModel(**config)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "artifact model state is incompatible: {}".format(error)
        )
    if any(value.device.type != "cpu" for value in state.values()):
        raise ValueError("artifact model state must be stored on CPU")
    return copy.deepcopy(config)


def _payloads_equal(first, second):
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (isinstance(first, torch.Tensor)
                and isinstance(second, torch.Tensor)
                and first.dtype == second.dtype
                and tuple(first.shape) == tuple(second.shape)
                and torch.equal(first, second))
    if isinstance(first, dict) or isinstance(second, dict):
        return (isinstance(first, dict) and isinstance(second, dict)
                and set(first) == set(second)
                and all(_payloads_equal(first[key], second[key])
                        for key in first))
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (type(first) is type(second) and len(first) == len(second)
                and all(_payloads_equal(a, b) for a, b in zip(first, second)))
    return first == second


def _fsync_directory(path):
    descriptor = os.open(
        str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_directory_path_without_symlinks(path):
    path = Path(path).expanduser().absolute()
    if not path.is_absolute():
        raise ValueError("directory path must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(str(current))
        except OSError as error:
            raise ValueError(
                "directory component does not exist: {}".format(current)
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(
                "directory path contains symlink component: {}".format(
                    current
                )
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                "directory path component is not a directory: {}".format(
                    current
                )
            )
    return path


def _directory_reservation(path):
    path = Path(path).expanduser().absolute()
    metadata = os.stat(str(path), follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("reserved output must be a directory")
    return {
        "path": str(path),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
    }


def reserve_selective_residual_output(output_dir):
    """Exclusively reserve the final output directory before computation."""
    if not isinstance(output_dir, (str, os.PathLike)):
        raise TypeError("output directory must be path-like")
    output = Path(output_dir).expanduser().absolute()
    _require_directory_path_without_symlinks(output.parent)
    os.mkdir(str(output), 0o700)
    reservation = _directory_reservation(output)
    if stat.S_IMODE(os.stat(
            str(output), follow_symlinks=False
    ).st_mode) != 0o700:
        raise RuntimeError("reserved output directory mode is not 0700")
    _fsync_directory(output.parent)
    return reservation


def _open_reserved_directory(output_dir, reservation):
    output = Path(output_dir).expanduser().absolute()
    if (not isinstance(reservation, dict)
            or set(reservation) != {"path", "device", "inode"}
            or reservation.get("path") != str(output)
            or type(reservation.get("device")) is not int
            or type(reservation.get("inode")) is not int):
        raise ValueError("output reservation is invalid")
    live = _directory_reservation(output)
    if live != reservation:
        raise RuntimeError("reserved output directory identity changed")
    descriptor = os.open(
        str(output),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if (not stat.S_ISDIR(opened.st_mode)
            or int(opened.st_dev) != reservation["device"]
            or int(opened.st_ino) != reservation["inode"]):
        os.close(descriptor)
        raise RuntimeError("opened output directory differs from reservation")
    return descriptor


def _verify_reserved_directory(directory_fd, reservation):
    opened = os.fstat(directory_fd)
    if (not stat.S_ISDIR(opened.st_mode)
            or int(opened.st_dev) != reservation["device"]
            or int(opened.st_ino) != reservation["inode"]
            or _directory_reservation(reservation["path"]) != reservation):
        raise RuntimeError("reserved output directory identity changed")


def _exclusive_write_bytes(
        directory_fd, reservation, name, payload, mode=0o444):
    """Write one fresh file through a held directory descriptor."""
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or "/" in name or os.path.sep in name):
        raise ValueError("exclusive output name is invalid")
    if not isinstance(payload, bytes):
        raise TypeError("exclusive output payload must be bytes")
    if mode != 0o444:
        raise ValueError("final output mode must be 0444")
    _verify_reserved_directory(directory_fd, reservation)
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("exclusive output write made no progress")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (not stat.S_ISREG(opened.st_mode)
                or int(opened.st_dev) != int(linked.st_dev)
                or int(opened.st_ino) != int(linked.st_ino)
                or int(opened.st_size) != len(payload)
                or stat.S_IMODE(opened.st_mode) != mode):
            raise RuntimeError("exclusive output pathname identity changed")
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)
    _verify_reserved_directory(directory_fd, reservation)
    return {
        "device": int(opened.st_dev),
        "inode": int(opened.st_ino),
        "size": int(opened.st_size),
        "mode": stat.S_IMODE(opened.st_mode),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _serialize_selective_residual_artifact(artifact):
    validate_selective_residual_artifact(
        artifact,
        AUTHORITATIVE_BACKBONE_SHA256,
        AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
        artifact.get("feature_names") if isinstance(artifact, dict) else (),
    )
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    return buffer.getvalue()


def _read_stable_bytes(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("residual artifact must be a regular non-symlink file")
    try:
        with path.open("rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()))
            snapshot = handle.read()
            after = _file_identity(os.fstat(handle.fileno()))
        live = _file_identity(path.stat())
    except OSError as error:
        raise ValueError("could not read residual artifact: {}".format(error))
    if before != after or after != live:
        raise ValueError("residual artifact changed during stable load")
    return path, snapshot, hashlib.sha256(snapshot).hexdigest()


def load_selective_residual_artifact(
        path, device="cpu", parent_sha256=None, geometry_sha256=None):
    """Stable-load and freeze one cache-calibrated residual artifact."""
    resolved, snapshot, snapshot_sha256 = _read_stable_bytes(path)
    try:
        artifact = torch.load(io.BytesIO(snapshot), map_location="cpu")
    except Exception as error:
        raise ValueError("could not deserialize residual artifact: {}".format(
            error
        ))
    parent_sha256 = (
        AUTHORITATIVE_PARENT_ARTIFACT_SHA256
        if parent_sha256 is None else parent_sha256
    )
    geometry_sha256 = (
        AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256
        if geometry_sha256 is None else geometry_sha256
    )
    config = validate_selective_residual_artifact(
        artifact,
        AUTHORITATIVE_BACKBONE_SHA256,
        parent_sha256,
        geometry_sha256,
        artifact.get("feature_names") if isinstance(artifact, dict) else (),
    )
    model = SelectiveResidualModel(**config)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA residual load requested but unavailable")
    model.to(resolved_device).eval().requires_grad_(False)
    model._artifact_path = str(resolved)
    model._artifact_sha256 = snapshot_sha256
    return model, artifact


def save_selective_residual_artifact(path, artifact):
    """Publish one new artifact without an overwrite-capable rename."""
    output = Path(path).expanduser().absolute()
    parent = _require_directory_path_without_symlinks(output.parent)
    reservation = _directory_reservation(parent)
    directory_fd = _open_reserved_directory(parent, reservation)
    try:
        payload = _serialize_selective_residual_artifact(artifact)
        _exclusive_write_bytes(
            directory_fd, reservation, output.name, payload, mode=0o444
        )
    finally:
        os.close(directory_fd)
    _model, reloaded = load_selective_residual_artifact(
        output, device="cpu"
    )
    if not _payloads_equal(artifact, reloaded):
        raise RuntimeError("strict residual artifact reload changed payload")
    return artifact


def capture_immutable_artifact_identities(paths):
    """Capture SHA/mode/inode evidence for the three protected best files."""
    if (not isinstance(paths, dict)
            or set(paths) != {"backbone", "parent", "geometry"}):
        raise ValueError("protected paths must name backbone, parent, geometry")
    identities = {}
    for name, value in paths.items():
        path = Path(value).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError("protected {} must be a regular file".format(name))
        metadata = path.stat()
        mode = stat.S_IMODE(metadata.st_mode)
        if mode != 0o444:
            raise ValueError("protected {} must have mode 0444".format(name))
        identities[name] = {
            "path": str(path),
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "mode": mode,
            "size": int(metadata.st_size),
            "mtime_ns": int(metadata.st_mtime_ns),
            "ctime_ns": int(metadata.st_ctime_ns),
            "sha256": _stable_file_sha256(path),
        }
    return identities


_RESULT_CONTEXT_FIELDS = {
    "input_sha256",
    "split",
    "fit_joined_identity_sha256",
    "fit_materialization_sha256",
    "oof",
    "calibration",
}
_RESULT_OOF_CONTEXT_FIELDS = {
    "baseline",
    "scene_folds",
    "scene_fold_sha256",
    "configuration_count",
    "configurations",
    "policy_candidate_count",
    "choice",
}
_RESULT_RECEIPT_FIELDS = _RESULT_CONTEXT_FIELDS | {
    "schema",
    "version",
    "selected",
    "deployable",
    "report_only",
    "eligible_for_model_selection",
    "validation_data_accessed",
    "protocol",
    "fit_binding_sha256",
    "artifact",
    "protected_before",
    "protected_after",
}
_RESULT_OOF_FIELDS = _RESULT_OOF_CONTEXT_FIELDS | {
    "configuration_summaries_sha256",
    "choice_sha256",
}
_PROTECTED_IDENTITY_FIELDS = {
    "path", "device", "inode", "mode", "size", "mtime_ns", "ctime_ns",
    "sha256",
}
_OOF_BASELINE_FIELDS = {
    "sample_count",
    "hits025",
    "hits050",
    "oracle_hits025",
    "oracle_hits050",
    "candidate_iou_sha256",
    "row_materialization_sha256",
    "baseline_selected_iou_sha256",
}
_CONFIGURATION_FIELDS = {
    "hidden_dim",
    "weight_decay",
    "break_cost",
    "configuration_index",
    "folds",
    "gain_summary",
    "oof_pair_gain_sha256",
    "prediction_count",
}
_DIAGNOSTIC_FIELDS = {
    "hidden_dim",
    "weight_decay",
    "break_cost",
    "margin_percentile",
    "margin",
    "no_switch",
    "sample_count",
    "switches",
    "abstentions",
    "switch_rate",
    "baseline",
    "proposed",
    "effects",
    "delta_hits025",
    "delta_hits050",
    "fold_deltas",
    "bootstrap025",
    "bootstrap050",
    "eligibility_predicates",
    "failed_predicates",
    "eligible",
    "selected",
}
_ELIGIBILITY_PREDICATE_FIELDS = {
    "not_no_switch",
    "all_folds_nonnegative025",
    "all_folds_nonnegative050",
    "pooled_delta025_positive",
    "bootstrap025_lower_bound_positive",
    "bootstrap050_lower_bound_nonnegative",
}


def _result_protocol():
    return {
        "scope": "train-only-scene-cross-fit",
        "selection_uses_validation": False,
        "seed": RESIDUAL_MODEL_SEED,
        "fold_count": 5,
        "epochs": RESIDUAL_EPOCHS,
        "batch_size": RESIDUAL_BATCH_SIZE,
        "materialization_batch_size": RESIDUAL_MATERIALIZATION_BATCH_SIZE,
        "learning_rate": RESIDUAL_LEARNING_RATE,
        "gradient_clip_norm": RESIDUAL_GRAD_CLIP_NORM,
        "thresholds": [0.25, 0.50],
        "head_weights": [2.0, 1.0],
        "hidden_dim_grid": list(RESIDUAL_HIDDEN_DIMS),
        "weight_decay_grid": list(RESIDUAL_WEIGHT_DECAYS),
        "break_cost_grid": list(RESIDUAL_BREAK_COSTS),
        "margin_percentile_grid": list(RESIDUAL_MARGIN_PERCENTILES),
        "margin_rule": "nearest-rank-positive-oof-gain",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 0,
    }


def _calibration_gate_receipt(gate):
    if not isinstance(gate, CalibrationGateResult):
        raise TypeError("calibration gate must be CalibrationGateResult")
    return {
        "passed": gate.passed,
        "failures": list(gate.failures),
        "required_hits025": gate.required_hits025,
        "required_hits050": gate.required_hits050,
        "observed_hits025": gate.observed_hits025,
        "observed_hits050": gate.observed_hits050,
    }


def _validate_protected_receipt_snapshot(snapshot, name):
    if (not isinstance(snapshot, dict)
            or set(snapshot) != {"backbone", "parent", "geometry"}):
        raise ValueError("{} protected snapshot is invalid".format(name))
    for artifact_name, identity in snapshot.items():
        if (not isinstance(identity, dict)
                or set(identity) != _PROTECTED_IDENTITY_FIELDS
                or not isinstance(identity["path"], str)
                or not identity["path"]
                or type(identity["device"]) is not int
                or identity["device"] < 0
                or type(identity["inode"]) is not int
                or identity["inode"] <= 0
                or identity["mode"] != 0o444
                or type(identity["size"]) is not int
                or identity["size"] <= 0
                or type(identity["mtime_ns"]) is not int
                or type(identity["ctime_ns"]) is not int
                or not _is_sha256(identity["sha256"])):
            raise ValueError(
                "{} protected {} identity is invalid".format(
                    name, artifact_name
                )
            )


def _validate_label_summary(summary, expected_pair_count):
    if (not isinstance(summary, dict)
            or set(summary) != {"all", "same_query", "different_query"}):
        raise ValueError("training label summary groups are invalid")
    for threshold in ("0.25", "0.50"):
        group_counts = {}
        for group_name in ("all", "same_query", "different_query"):
            group = summary[group_name]
            if not isinstance(group, dict) or set(group) != {"0.25", "0.50"}:
                raise ValueError("training label thresholds are invalid")
            counts = group[threshold]
            if (not isinstance(counts, dict)
                    or set(counts) != {"break", "neutral", "fix", "total"}
                    or any(type(counts[name]) is not int or counts[name] < 0
                           for name in counts)
                    or sum(counts[name] for name in (
                        "break", "neutral", "fix"
                    )) != counts["total"]):
                raise ValueError("training label counts do not reconcile")
            group_counts[group_name] = counts
        if group_counts["all"]["total"] != expected_pair_count:
            raise ValueError("training label total differs from fit pairs")
        for label in ("break", "neutral", "fix", "total"):
            if (group_counts["same_query"][label]
                    + group_counts["different_query"][label]
                    != group_counts["all"][label]):
                raise ValueError("query-partitioned label counts differ")


def _validate_gain_summary(summary):
    if not isinstance(summary, dict) or set(summary) != {"valid", "positive"}:
        raise ValueError("gain summary groups are invalid")
    counts = {}
    for name in ("valid", "positive"):
        record = summary[name]
        if (not isinstance(record, dict)
                or set(record) != {"count", "statistics"}
                or type(record["count"]) is not int
                or record["count"] < 0):
            raise ValueError("gain summary count is invalid")
        counts[name] = record["count"]
        statistics = record["statistics"]
        if record["count"] == 0:
            if statistics is not None:
                raise ValueError("empty gain summary fabricated statistics")
            continue
        if (not isinstance(statistics, dict)
                or set(statistics) != {
                    "minimum", "maximum", "mean",
                    "population_standard_deviation",
                    "nearest_rank_quantiles",
                }):
            raise ValueError("gain statistics fields are invalid")
        scalars = [
            statistics["minimum"],
            statistics["maximum"],
            statistics["mean"],
            statistics["population_standard_deviation"],
        ]
        if (any(isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) for value in scalars)
                or statistics["minimum"] > statistics["maximum"]
                or statistics["population_standard_deviation"] < 0.0):
            raise ValueError("gain statistics scalars are invalid")
        quantiles = statistics["nearest_rank_quantiles"]
        if (not isinstance(quantiles, list)
                or [record.get("quantile") for record in quantiles]
                != list(RESIDUAL_GAIN_QUANTILES)
                or any(set(record) != {"quantile", "value"}
                       or not isinstance(record["value"], (int, float))
                       or isinstance(record["value"], bool)
                       or not math.isfinite(float(record["value"]))
                       for record in quantiles)):
            raise ValueError("gain nearest-rank quantiles are invalid")
    if counts["positive"] > counts["valid"]:
        raise ValueError("positive gain count exceeds valid count")


def _validate_candidate_diagnostic(record, baseline, scene_count):
    if not isinstance(record, dict) or set(record) != _DIAGNOSTIC_FIELDS:
        raise ValueError("OOF candidate diagnostic fields are invalid")
    if (record["hidden_dim"] not in RESIDUAL_HIDDEN_DIMS
            or record["weight_decay"] not in RESIDUAL_WEIGHT_DECAYS
            or record["break_cost"] not in RESIDUAL_BREAK_COSTS
            or type(record["no_switch"]) is not bool
            or record["sample_count"] != baseline["sample_count"]
            or type(record["switches"]) is not int
            or type(record["abstentions"]) is not int
            or not 0 <= record["switches"] <= record["sample_count"]
            or record["switches"] + record["abstentions"]
            != record["sample_count"]
            or not isinstance(record["switch_rate"], (int, float))
            or isinstance(record["switch_rate"], bool)
            or not math.isclose(
                float(record["switch_rate"]),
                record["switches"] / float(record["sample_count"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )):
        raise ValueError("OOF candidate identity or switch counts are invalid")
    if record["no_switch"]:
        if (record["margin_percentile"] is not None
                or record["margin"] is not None):
            raise ValueError("no-switch diagnostic margin must be null")
    elif (record["margin_percentile"] not in RESIDUAL_MARGIN_PERCENTILES
          or not isinstance(record["margin"], (int, float))
          or isinstance(record["margin"], bool)
          or not math.isfinite(float(record["margin"]))
          or float(record["margin"]) <= 0.0):
        raise ValueError("OOF candidate margin is invalid")
    for container_name in ("baseline", "proposed"):
        container = record[container_name]
        if (not isinstance(container, dict)
                or set(container) != {"0.25", "0.50"}):
            raise ValueError("candidate hit thresholds are invalid")
        for threshold in ("0.25", "0.50"):
            hit_record = container[threshold]
            if (not isinstance(hit_record, dict)
                    or set(hit_record) != {"hits"}
                    or type(hit_record["hits"]) is not int
                    or not 0 <= hit_record["hits"] <= record["sample_count"]):
                raise ValueError("candidate hit count is invalid")
    if (record["baseline"]["0.25"]["hits"] != baseline["hits025"]
            or record["baseline"]["0.50"]["hits"] != baseline["hits050"]):
        raise ValueError("candidate baseline hits differ from OOF baseline")
    effects = record["effects"]
    if not isinstance(effects, dict) or set(effects) != {"0.25", "0.50"}:
        raise ValueError("candidate effects thresholds are invalid")
    for threshold, delta_name in (("0.25", "delta_hits025"),
                                  ("0.50", "delta_hits050")):
        threshold_effects = effects[threshold]
        if (not isinstance(threshold_effects, dict)
                or set(threshold_effects) != {
                    "fixes", "breaks", "neutral_switches",
                    "kept_correct", "kept_wrong",
                }
                or any(type(value) is not int or value < 0
                       for value in threshold_effects.values())
                or sum(threshold_effects.values()) != record["sample_count"]
                or (threshold_effects["fixes"]
                    + threshold_effects["breaks"]
                    + threshold_effects["neutral_switches"])
                != record["switches"]):
            raise ValueError("candidate effect counts do not reconcile")
        observed_delta = (
            record["proposed"][threshold]["hits"]
            - record["baseline"][threshold]["hits"]
        )
        if (observed_delta != threshold_effects["fixes"]
                - threshold_effects["breaks"]
                or observed_delta != record[delta_name]):
            raise ValueError("candidate hit delta does not reconcile")
    fold_deltas = record["fold_deltas"]
    if (not isinstance(fold_deltas, dict)
            or set(fold_deltas) != {str(fold) for fold in range(5)}):
        raise ValueError("candidate fold deltas are invalid")
    for fold_record in fold_deltas.values():
        if (not isinstance(fold_record, dict)
                or set(fold_record) != {"hits025", "hits050"}
                or any(type(value) is not int
                       for value in fold_record.values())):
            raise ValueError("candidate fold delta record is invalid")
    if (sum(value["hits025"] for value in fold_deltas.values())
            != record["delta_hits025"]
            or sum(value["hits050"] for value in fold_deltas.values())
            != record["delta_hits050"]):
        raise ValueError("candidate fold deltas do not sum to pooled delta")
    bootstraps = {}
    for threshold, name in (("0.25", "bootstrap025"),
                            ("0.50", "bootstrap050")):
        bootstrap = record[name]
        if (not isinstance(bootstrap, dict)
                or set(bootstrap) != {
                    "confidence", "delta_hits", "lower_bound_95",
                    "replicates", "scene_count", "seed",
                }
                or bootstrap["confidence"] != 0.95
                or bootstrap["replicates"] != 10000
                or bootstrap["scene_count"] != scene_count
                or bootstrap["seed"] != 0
                or type(bootstrap["delta_hits"]) is not int
                or type(bootstrap["lower_bound_95"]) is not int
                or bootstrap["delta_hits"] != record[
                    "delta_hits025" if threshold == "0.25"
                    else "delta_hits050"
                ]):
            raise ValueError("candidate bootstrap record is invalid")
        bootstraps[threshold] = bootstrap
    predicates = record["eligibility_predicates"]
    expected_predicates = {
        "not_no_switch": not record["no_switch"],
        "all_folds_nonnegative025": all(
            value["hits025"] >= 0 for value in fold_deltas.values()
        ),
        "all_folds_nonnegative050": all(
            value["hits050"] >= 0 for value in fold_deltas.values()
        ),
        "pooled_delta025_positive": record["delta_hits025"] > 0,
        "bootstrap025_lower_bound_positive": (
            bootstraps["0.25"]["lower_bound_95"] > 0
        ),
        "bootstrap050_lower_bound_nonnegative": (
            bootstraps["0.50"]["lower_bound_95"] >= 0
        ),
    }
    if (not isinstance(predicates, dict)
            or set(predicates) != _ELIGIBILITY_PREDICATE_FIELDS
            or predicates != expected_predicates
            or record["failed_predicates"] != sorted(
                name for name, passed in predicates.items() if not passed
            )
            or type(record["eligible"]) is not bool
            or record["eligible"] != all(predicates.values())
            or record["selected"]
            != ("residual" if record["eligible"] else "baseline")):
        raise ValueError("candidate eligibility predicates do not reconcile")


def validate_selective_residual_result_receipt(receipt):
    """Strictly validate a complete train-only residual result receipt."""
    if not isinstance(receipt, dict) or set(receipt) != _RESULT_RECEIPT_FIELDS:
        raise ValueError("residual result receipt fields differ from v2 schema")
    if (receipt.get("schema")
            != "rec-selective-residual-result-receipt-v2"
            or receipt.get("version") != 2
            or receipt.get("deployable") is not False
            or receipt.get("report_only") is not False
            or receipt.get("eligible_for_model_selection") is not True
            or receipt.get("validation_data_accessed") is not False
            or receipt.get("protocol") != _result_protocol()):
        raise ValueError("residual result receipt policy is invalid")
    inputs = receipt["input_sha256"]
    if (not isinstance(inputs, dict) or set(inputs) != _INPUT_SHA_FIELDS
            or any(not _is_sha256(value) for value in inputs.values())):
        raise ValueError("receipt input SHA-256 fields are invalid")
    if receipt["split"] != AUTHORITATIVE_SPLIT_SEED0:
        raise ValueError("receipt train scene split is not authoritative")
    for name in ("fit_joined_identity_sha256",
                 "fit_materialization_sha256", "fit_binding_sha256"):
        if not _is_sha256(receipt[name]):
            raise ValueError("receipt {} is invalid".format(name))
    expected_fit_binding = canonical_json_sha256({
        "fit_joined_identity_sha256": receipt[
            "fit_joined_identity_sha256"
        ],
        "fit_materialization_sha256": receipt[
            "fit_materialization_sha256"
        ],
        "split_mapping_sha256": receipt["split"]["mapping_sha256"],
    })
    if receipt["fit_binding_sha256"] != expected_fit_binding:
        raise ValueError("receipt fit evidence binding changed")
    before = receipt["protected_before"]
    after = receipt["protected_after"]
    _validate_protected_receipt_snapshot(before, "before")
    _validate_protected_receipt_snapshot(after, "after")
    if before != after:
        raise ValueError("protected artifacts changed across receipt")
    for name in ("backbone", "parent", "geometry"):
        if before[name]["sha256"] != inputs[name]:
            raise ValueError("receipt protected input SHA-256 mismatch")

    oof = receipt["oof"]
    if not isinstance(oof, dict) or set(oof) != _RESULT_OOF_FIELDS:
        raise ValueError("receipt OOF fields are invalid")
    baseline = oof["baseline"]
    if (not isinstance(baseline, dict)
            or set(baseline) != _OOF_BASELINE_FIELDS
            or baseline["sample_count"]
            != receipt["split"]["fit_sample_count"]
            or baseline["row_materialization_sha256"]
            != receipt["fit_materialization_sha256"]
            or any(type(baseline[name]) is not int for name in (
                "sample_count", "hits025", "hits050",
                "oracle_hits025", "oracle_hits050",
            ))
            or not 0 < baseline["hits050"] <= baseline["hits025"]
            <= baseline["sample_count"]
            or not baseline["hits025"] <= baseline["oracle_hits025"]
            <= baseline["sample_count"]
            or not baseline["hits050"] <= baseline["oracle_hits050"]
            <= baseline["sample_count"]
            or any(not _is_sha256(baseline[name]) for name in (
                "candidate_iou_sha256", "row_materialization_sha256",
                "baseline_selected_iou_sha256",
            ))):
        raise ValueError("receipt OOF baseline is invalid")
    scene_folds = oof["scene_folds"]
    if (not isinstance(scene_folds, dict)
            or len(scene_folds) != receipt["split"]["fit_scene_count"]
            or any(not isinstance(scene_id, str) or not scene_id
                   or type(fold) is not int or fold not in range(5)
                   for scene_id, fold in scene_folds.items())
            or set(scene_folds.values()) != set(range(5))
            or canonical_scene_fold_sha256(scene_folds)
            != oof["scene_fold_sha256"]):
        raise ValueError("receipt OOF scene-fold binding is invalid")
    configurations = oof["configurations"]
    expected_grid = list(itertools.product(
        RESIDUAL_HIDDEN_DIMS,
        RESIDUAL_WEIGHT_DECAYS,
        RESIDUAL_BREAK_COSTS,
    ))
    if (not isinstance(configurations, list)
            or oof["configuration_count"] != len(expected_grid)
            or len(configurations) != len(expected_grid)
            or oof["configuration_summaries_sha256"]
            != canonical_json_sha256(configurations)):
        raise ValueError("receipt OOF configuration table is invalid")
    for configuration_index, (configuration, expected_config) in enumerate(
            zip(configurations, expected_grid)):
        if (not isinstance(configuration, dict)
                or set(configuration) != _CONFIGURATION_FIELDS
                or configuration["configuration_index"] != configuration_index
                or (configuration["hidden_dim"],
                    configuration["weight_decay"],
                    configuration["break_cost"]) != expected_config
                or configuration["prediction_count"]
                != baseline["sample_count"]
                or not _is_sha256(configuration["oof_pair_gain_sha256"])):
            raise ValueError("receipt OOF configuration record is invalid")
        _validate_gain_summary(configuration["gain_summary"])
        folds = configuration["folds"]
        if (not isinstance(folds, list) or len(folds) != 5
                or {fold.get("fold") for fold in folds} != set(range(5))):
            raise ValueError("receipt configuration fold count is invalid")
        held_row_total = 0
        held_scene_total = 0
        fit_pair_total = 0
        for fold in folds:
            if (not isinstance(fold, dict)
                    or set(fold) != {
                        "fold", "fit_scene_count", "fit_row_count",
                        "fit_pair_count",
                        "held_scene_count", "held_row_count",
                        "training_labels",
                    }
                    or any(type(fold[name]) is not int or fold[name] <= 0
                           for name in (
                               "fit_scene_count", "fit_row_count",
                               "fit_pair_count",
                               "held_scene_count", "held_row_count",
                           ))
                    or fold["fit_scene_count"] + fold["held_scene_count"]
                    != receipt["split"]["fit_scene_count"]
                    or fold["fit_row_count"] + fold["held_row_count"]
                    != baseline["sample_count"]):
                raise ValueError("receipt configuration fold is invalid")
            _validate_label_summary(
                fold["training_labels"], fold["fit_pair_count"]
            )
            held_row_total += fold["held_row_count"]
            held_scene_total += fold["held_scene_count"]
            fit_pair_total += fold["fit_pair_count"]
        if (held_row_total != baseline["sample_count"]
                or held_scene_total != receipt["split"]["fit_scene_count"]):
            raise ValueError("receipt held-out folds do not partition fit data")
        if fit_pair_total != 4 * configuration[
                "gain_summary"]["valid"]["count"]:
            raise ValueError("receipt fit pair counts do not reconcile")

    choice = oof["choice"]
    if (not isinstance(choice, dict)
            or oof["choice_sha256"] != canonical_json_sha256(choice)):
        raise ValueError("receipt OOF choice binding is invalid")
    diagnostics = choice.get("candidate_diagnostics")
    if (not isinstance(diagnostics, list) or not diagnostics
            or choice.get("candidate_count") != len(diagnostics)
            or oof["policy_candidate_count"] != len(diagnostics)):
        raise ValueError("receipt OOF policy candidate count is invalid")
    for diagnostic in diagnostics:
        _validate_candidate_diagnostic(
            diagnostic, baseline, receipt["split"]["fit_scene_count"]
        )
    eligible_records = [
        diagnostic for diagnostic in diagnostics if diagnostic["eligible"]
    ]
    if choice.get("eligible_candidate_count") != len(eligible_records):
        raise ValueError("receipt eligible candidate count is invalid")
    if choice.get("eligible") is False:
        if (set(choice) != {
                "candidate_count", "eligible_candidate_count",
                "candidate_diagnostics", "eligible", "reason", "selected",
                }
                or choice.get("reason") != "no-eligible-configuration"
                or choice.get("selected") != "baseline"
                or eligible_records):
            raise ValueError("receipt rejected OOF choice is invalid")
    elif choice.get("eligible") is True:
        if (set(choice) != _DIAGNOSTIC_FIELDS | {
                "candidate_count", "eligible_candidate_count",
                "candidate_diagnostics",
                }
                or choice.get("selected") != "residual"
                or not any(all(choice[name] == record[name]
                               for name in _DIAGNOSTIC_FIELDS)
                           for record in eligible_records)):
            raise ValueError("receipt eligible OOF choice is invalid")
    else:
        raise ValueError("receipt OOF choice eligibility is invalid")

    calibration = receipt["calibration"]
    artifact = receipt["artifact"]
    if choice["eligible"] is False:
        if (calibration != {
                "status": "not_run",
                "reason": "oof_selection_rejected",
                }
                or artifact is not None
                or receipt["selected"] != "baseline"):
            raise ValueError("rejected OOF receipt ran calibration")
    else:
        if (not isinstance(calibration, dict)
                or set(calibration) != {
                    "status", "baseline", "record", "gate",
                }
                or calibration["status"] != "run"):
            raise ValueError("eligible OOF receipt lacks calibration")
        gate = calibration_gate(
            calibration["record"], calibration["baseline"]
        )
        if calibration["gate"] != _calibration_gate_receipt(gate):
            raise ValueError("receipt calibration gate does not reconcile")
        if gate.passed:
            if (not isinstance(artifact, dict)
                    or set(artifact) != {"name", "sha256"}
                    or artifact["name"] != "selected_residual.pth"
                    or not _is_sha256(artifact["sha256"])
                    or receipt["selected"] != "staged_residual"):
                raise ValueError("passing receipt artifact binding is invalid")
        elif artifact is not None or receipt["selected"] != "baseline":
            raise ValueError("failed calibration receipt selected residual")
    try:
        canonical_json_sha256(receipt)
    except (TypeError, ValueError) as error:
        raise ValueError("receipt is not strict canonical JSON: {}".format(
            error
        ))
    return copy.deepcopy(receipt)


def build_selective_residual_result_receipt(
        result_context, artifact_binding, protected_before, protected_after):
    """Build and self-validate one truthful v2 experiment receipt."""
    if (not isinstance(result_context, dict)
            or set(result_context) != _RESULT_CONTEXT_FIELDS):
        raise ValueError("residual result context fields are invalid")
    oof_context = result_context["oof"]
    if (not isinstance(oof_context, dict)
            or set(oof_context) != _RESULT_OOF_CONTEXT_FIELDS):
        raise ValueError("residual result OOF context fields are invalid")
    oof = copy.deepcopy(oof_context)
    oof["configuration_summaries_sha256"] = canonical_json_sha256(
        oof["configurations"]
    )
    oof["choice_sha256"] = canonical_json_sha256(oof["choice"])
    fit_binding_sha256 = canonical_json_sha256({
        "fit_joined_identity_sha256": result_context[
            "fit_joined_identity_sha256"
        ],
        "fit_materialization_sha256": result_context[
            "fit_materialization_sha256"
        ],
        "split_mapping_sha256": result_context["split"]["mapping_sha256"],
    })
    receipt = {
        "schema": "rec-selective-residual-result-receipt-v2",
        "version": 2,
        "selected": (
            "staged_residual" if artifact_binding is not None else "baseline"
        ),
        "deployable": False,
        "report_only": False,
        "eligible_for_model_selection": True,
        "validation_data_accessed": False,
        "protocol": _result_protocol(),
        "input_sha256": copy.deepcopy(result_context["input_sha256"]),
        "split": copy.deepcopy(result_context["split"]),
        "fit_joined_identity_sha256": result_context[
            "fit_joined_identity_sha256"
        ],
        "fit_materialization_sha256": result_context[
            "fit_materialization_sha256"
        ],
        "fit_binding_sha256": fit_binding_sha256,
        "oof": oof,
        "calibration": copy.deepcopy(result_context["calibration"]),
        "artifact": copy.deepcopy(artifact_binding),
        "protected_before": copy.deepcopy(protected_before),
        "protected_after": copy.deepcopy(protected_after),
    }
    return validate_selective_residual_result_receipt(receipt)


def publish_selective_residual_experiment(
        output_dir, artifact, result_context, protected_paths,
        protected_before=None, reservation=None):
    """Publish fresh files into a reserved directory, receipt last."""
    output = Path(output_dir).expanduser().absolute()
    if reservation is None:
        reservation = reserve_selective_residual_output(output)
    directory_fd = _open_reserved_directory(output, reservation)
    try:
        live_before = capture_immutable_artifact_identities(protected_paths)
        if protected_before is None:
            protected_before = live_before
        if protected_before != live_before:
            raise RuntimeError("protected artifacts changed before publication")
        calibration = (
            result_context.get("calibration")
            if isinstance(result_context, dict) else None
        )
        calibration_passed = False
        if (isinstance(calibration, dict)
                and calibration.get("status") == "run"):
            calibration_passed = calibration_gate(
                calibration.get("record"), calibration.get("baseline")
            ).passed
        if calibration_passed != (artifact is not None):
            raise ValueError(
                "artifact presence must match the measured calibration gate"
            )
        preflight_binding = (
            None if artifact is None else {
                "name": "selected_residual.pth",
                "sha256": "0" * 64,
            }
        )
        build_selective_residual_result_receipt(
            result_context,
            artifact_binding=preflight_binding,
            protected_before=protected_before,
            protected_after=protected_before,
        )
        artifact_path = None
        artifact_sha256 = None
        if artifact is not None:
            artifact_path = output / "selected_residual.pth"
            artifact_payload = _serialize_selective_residual_artifact(artifact)
            artifact_identity = _exclusive_write_bytes(
                directory_fd,
                reservation,
                artifact_path.name,
                artifact_payload,
                mode=0o444,
            )
            artifact_sha256 = artifact_identity["sha256"]
            _model, reloaded_artifact = load_selective_residual_artifact(
                artifact_path, device="cpu"
            )
            if not _payloads_equal(artifact, reloaded_artifact):
                raise RuntimeError(
                    "strict residual artifact reload changed payload"
                )
        protected_after = capture_immutable_artifact_identities(protected_paths)
        if protected_after != protected_before:
            raise RuntimeError("protected artifacts changed during experiment")
        artifact_binding = (
            None if artifact_path is None else {
                "name": artifact_path.name,
                "sha256": artifact_sha256,
            }
        )
        receipt = build_selective_residual_result_receipt(
            result_context,
            artifact_binding=artifact_binding,
            protected_before=protected_before,
            protected_after=protected_after,
        )
        receipt_payload = json.dumps(
            receipt,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        pending_name = ".result-receipt.json.pending"
        pending_identity = _exclusive_write_bytes(
            directory_fd,
            reservation,
            pending_name,
            receipt_payload,
            mode=0o444,
        )
        reloaded_receipt = json.loads(receipt_payload.decode("ascii"))
        validate_selective_residual_result_receipt(reloaded_receipt)
        if reloaded_receipt != receipt:
            raise RuntimeError("strict receipt reload changed payload")
        os.link(
            pending_name,
            "result-receipt.json",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
        final_identity = os.stat(
            "result-receipt.json",
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (int(final_identity.st_dev) != pending_identity["device"]
                or int(final_identity.st_ino) != pending_identity["inode"]
                or int(final_identity.st_size) != pending_identity["size"]
                or stat.S_IMODE(final_identity.st_mode) != 0o444):
            raise RuntimeError("completion receipt identity changed")
        os.unlink(pending_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        _verify_reserved_directory(directory_fd, reservation)
    finally:
        os.close(directory_fd)
    final_receipt = output / "result-receipt.json"
    if stat.S_IMODE(final_receipt.stat().st_mode) != 0o444:
        raise RuntimeError("published residual receipt is not read-only")
    if capture_immutable_artifact_identities(protected_paths) != protected_before:
        raise RuntimeError("protected artifacts changed after publication")
    return {
        "output_dir": output,
        "receipt_path": final_receipt,
        "artifact_path": (
            None if artifact_path is None else artifact_path
        ),
        "artifact_sha256": artifact_sha256,
        "receipt": receipt,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit the fixed train-only ScanRefer selective residual"
    )
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True, choices=("cuda:0",))
    return parser.parse_args(argv)


def run_selective_residual_training(
        base_cache, geometry_cache, parent_artifact, geometry_artifact,
        output_dir, device="cuda:0"):
    """Run the fixed OOF/refit/cache-gate protocol and publish once."""
    base_path = _validate_train_only_path(base_cache, "base cache")
    geometry_cache_path = _validate_train_only_path(
        geometry_cache, "geometry cache"
    )
    parent_path = _validate_train_only_path(
        parent_artifact, "parent artifact"
    )
    geometry_path = _validate_train_only_path(
        geometry_artifact, "geometry artifact"
    )
    _validate_train_only_path(output_dir, "output")
    output_path = Path(output_dir).expanduser().absolute()
    if str(device) != "cuda:0":
        raise ValueError("authoritative residual training requires cuda:0")
    reservation = reserve_selective_residual_output(output_path)
    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": parent_path,
        "geometry": geometry_path,
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    loaded = load_residual_training_inputs(
        base_path,
        geometry_cache_path,
        parent_path,
        geometry_path,
        device=device,
    )
    joined_split = split_residual_joined_rows(loaded["joined_rows"])
    fit_records = materialize_residual_rows(
        joined_split["fit_rows"],
        loaded["parent"],
        loaded["geometry_model"],
        loaded["geometry_artifact"],
        batch_size=RESIDUAL_MATERIALIZATION_BATCH_SIZE,
        device=device,
        require_contiguous=False,
    )
    fit_joined_identity_sha256 = canonical_residual_joined_identity_sha256(
        joined_split["fit_rows"]
    )
    fit_baseline = build_cache_calibration_baseline(fit_records)
    fit_materialization_sha256 = fit_baseline[
        "row_materialization_sha256"
    ]
    cross_fit = cross_fit_selective_residual(fit_records, device=device)
    choice = cross_fit["choice"]
    result_context = {
        "input_sha256": copy.deepcopy(loaded["input_sha256"]),
        "split": copy.deepcopy(joined_split["metadata"]),
        "fit_joined_identity_sha256": fit_joined_identity_sha256,
        "fit_materialization_sha256": fit_materialization_sha256,
        "oof": {
            "baseline": fit_baseline,
            "scene_folds": cross_fit["scene_folds"],
            "scene_fold_sha256": cross_fit["scene_fold_sha256"],
            "configuration_count": len(cross_fit["configurations"]),
            "configurations": cross_fit["configurations"],
            "policy_candidate_count": cross_fit["policy_candidate_count"],
            "choice": choice,
        },
        "calibration": {
            "status": "not_run",
            "reason": "oof_selection_rejected",
        },
    }
    if choice.get("eligible") is not True:
        return publish_selective_residual_experiment(
            output_path,
            artifact=None,
            result_context=result_context,
            protected_paths=protected_paths,
            protected_before=protected_before,
            reservation=reservation,
        )

    refit_model = refit_selective_residual(
        fit_records, choice, device
    )
    calibration_records = materialize_residual_rows(
        joined_split["calibration_rows"],
        loaded["parent"],
        loaded["geometry_model"],
        loaded["geometry_artifact"],
        batch_size=RESIDUAL_MATERIALIZATION_BATCH_SIZE,
        device=device,
        require_contiguous=False,
    )
    calibration_baseline = build_cache_calibration_baseline(
        calibration_records
    )
    calibration_record = evaluate_selective_residual_policy(
        refit_model,
        calibration_records,
        margin=float(choice["margin"]),
        device=device,
    )
    gate = calibration_gate(calibration_record, calibration_baseline)
    result_context["calibration"] = {
        "status": "run",
        "baseline": calibration_baseline,
        "record": calibration_record,
        "gate": _calibration_gate_receipt(gate),
    }
    artifact = None
    if gate.passed:
        selected_configuration = None
        for configuration in cross_fit["configurations"]:
            if all(configuration[name] == choice[name] for name in (
                    "hidden_dim", "weight_decay", "break_cost")):
                selected_configuration = configuration
                break
        if selected_configuration is None:
            raise RuntimeError("selected OOF configuration record is missing")
        feature_names = build_selective_pair_feature_names(
            loaded["geometry_artifact"]["feature_names"]
        )
        artifact = build_selective_residual_artifact(
            model=refit_model,
            selection=choice,
            scene_folds=cross_fit["scene_folds"],
            feature_names=feature_names,
            input_sha256=loaded["input_sha256"],
            row_materialization_sha256=fit_materialization_sha256,
            oof_record={
                "prediction_count": len(fit_records),
                "pair_gain_sha256": selected_configuration[
                    "oof_pair_gain_sha256"
                ],
                "delta_hits025": choice["delta_hits025"],
                "delta_hits050": choice["delta_hits050"],
            },
            calibration_record=calibration_record,
            calibration_baseline=calibration_baseline,
        )
    return publish_selective_residual_experiment(
        output_path,
        artifact=artifact,
        result_context=result_context,
        protected_paths=protected_paths,
        protected_before=protected_before,
        reservation=reservation,
    )


def main(argv=None):
    args = parse_args(argv)
    publication = run_selective_residual_training(
        args.base_cache,
        args.geometry_cache,
        args.parent_artifact,
        args.geometry_artifact,
        args.output_dir,
        device=args.device,
    )
    summary = {
        "output_dir": str(publication["output_dir"]),
        "receipt_path": str(publication["receipt_path"]),
        "artifact_path": (
            None if publication["artifact_path"] is None
            else str(publication["artifact_path"])
        ),
        "artifact_sha256": publication["artifact_sha256"],
        "selected": publication["receipt"]["selected"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


__all__ = [
    "AUTHORITATIVE_BACKBONE_SHA256",
    "AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256",
    "AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256",
    "AUTHORITATIVE_GEOMETRY_METADATA_SHA256",
    "AUTHORITATIVE_PARENT_ARTIFACT_SHA256",
    "AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256",
    "RESIDUAL_BATCH_SIZE",
    "RESIDUAL_EPOCHS",
    "RESIDUAL_GAIN_QUANTILES",
    "RESIDUAL_LEARNING_RATE",
    "CalibrationGateResult",
    "AUTHORITATIVE_BACKBONE_PATH",
    "build_cache_calibration_baseline",
    "build_selective_pair_feature_names",
    "build_selective_residual_artifact",
    "build_selective_residual_result_receipt",
    "calibration_gate",
    "capture_immutable_artifact_identities",
    "canonical_residual_rows_sha256",
    "canonical_residual_joined_identity_sha256",
    "canonical_selected_iou_sha256",
    "cross_fit_selective_residual",
    "evaluate_selective_residual_policy",
    "load_selective_residual_artifact",
    "load_residual_training_inputs",
    "main",
    "materialize_residual_rows",
    "parse_args",
    "publish_selective_residual_experiment",
    "refit_selective_residual",
    "reserve_selective_residual_output",
    "run_selective_residual_training",
    "save_selective_residual_artifact",
    "split_residual_joined_rows",
    "split_residual_records",
    "summarize_oof_pair_gain",
    "summarize_residual_training_labels",
    "validate_selective_residual_artifact",
    "validate_selective_residual_result_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
