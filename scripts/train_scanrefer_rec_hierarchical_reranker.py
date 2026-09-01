#!/usr/bin/env python
"""Train-only hierarchical query-variant REC reranking pipeline."""

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
import struct
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from models.rec_geometry_reranker import (
    build_deployed_parent_state,
    build_flat_parent_prior,
)
from models.rec_hierarchical_reranker import (
    HIERARCHICAL_FALSE_POSITIVE_COSTS,
    QUERY_AUX_BINARY_DIM,
    QUERY_AUX_CONTINUOUS_DIM,
    QUERY_COUNT,
    QUERY_FEATURE_DIM,
    VARIANT_AUX_BINARY_DIM,
    VARIANT_AUX_CONTINUOUS_DIM,
    VARIANT_COUNT,
    VARIANT_FEATURE_DIM,
    HIERARCHICAL_HIDDEN_DIMS,
    HIERARCHICAL_MARGIN_PERCENTILES,
    HIERARCHICAL_WEIGHT_DECAYS,
    HierarchicalQueryVariantReranker,
    apply_hierarchical_policy,
    build_hierarchical_scene_folds,
    build_hierarchical_targets,
    canonical_hierarchical_scene_fold_sha256,
    choose_hierarchical_configuration,
    compute_hierarchical_loss,
    hierarchical_scene_clustered_hit_delta_bootstrap,
    select_hierarchical_proposal,
)
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
)
from scripts.train_rec_reranker import normalize_features
from scripts.rec_geometry_cache import canonical_json_sha256
from scripts.train_scanrefer_rec_selective_residual import (
    AUTHORITATIVE_BACKBONE_PATH,
    AUTHORITATIVE_BACKBONE_SHA256,
    AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
    AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256,
    AUTHORITATIVE_GEOMETRY_METADATA_SHA256,
    AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
    AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
    CalibrationGateResult,
    _parent_model,
    _directory_reservation,
    _exclusive_write_bytes,
    _fsync_directory,
    _open_reserved_directory,
    _payloads_equal,
    _require_directory_path_without_symlinks,
    _require_frozen_model,
    _verify_reserved_directory,
    _validate_geometry_outputs,
    _validate_joined_row_order,
    _validate_materialization_artifact,
    _validate_train_only_path,
    calibration_gate as _residual_calibration_gate,
    canonical_residual_joined_identity_sha256,
    capture_immutable_artifact_identities,
    load_residual_training_inputs,
    split_residual_joined_rows,
)


HIERARCHICAL_MATERIALIZATION_BATCH_SIZE = 256
HIERARCHICAL_EPOCHS = 12
HIERARCHICAL_BATCH_SIZE = 256
HIERARCHICAL_LEARNING_RATE = 3e-4
HIERARCHICAL_GRAD_CLIP_NORM = 1.0
HIERARCHICAL_DROPOUT = 0.1
HIERARCHICAL_MODEL_SEED = 0
HIERARCHICAL_GAIN_QUANTILES = (
    0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0,
)
HIERARCHICAL_MIN_STD = 1e-6
HIERARCHICAL_NORMALIZATION_SCHEMA = "rec-hierarchical-normalization-v1"
HIERARCHICAL_MODEL_BATCH_FIELDS = (
    "query_features",
    "variant_features",
    "query_aux_continuous",
    "query_aux_binary",
    "variant_aux_continuous",
    "variant_aux_binary",
    "query_valid",
    "variant_valid",
)
HIERARCHICAL_NORMALIZATION_GROUPS = (
    "query_features",
    "variant_features",
    "query_aux_continuous",
    "variant_aux_continuous",
)
_HIERARCHICAL_NORMALIZATION_MASKS = {
    "query_features": "query_valid",
    "variant_features": "variant_valid",
    "query_aux_continuous": "query_valid",
    "variant_aux_continuous": "variant_valid",
}
_HIERARCHICAL_NORMALIZATION_FEATURE_NAMES = {
    "query_features": tuple(
        "query_feature_{:03d}".format(index)
        for index in range(QUERY_FEATURE_DIM)
    ),
    "variant_features": tuple(
        "variant_feature_{:03d}".format(index)
        for index in range(VARIANT_FEATURE_DIM)
    ),
    "query_aux_continuous": (
        "default_score",
        "default_rank",
        "parent_score",
        "parent_rank",
    ),
    "variant_aux_continuous": (
        "geometry_score",
        "geometry_rank",
    ),
}
HIERARCHICAL_RECORD_FIELDS = (
    "dataset_index",
    "scan_id",
    "target_id",
    "query_features",
    "variant_features",
    "query_aux_continuous",
    "query_aux_binary",
    "variant_aux_continuous",
    "variant_aux_binary",
    "query_valid",
    "variant_valid",
    "candidate_ious",
    "baseline_index",
    "baseline_scores",
)
HIERARCHICAL_ARTIFACT_SCHEMA = "rec-hierarchical-query-variant-v1"
HIERARCHICAL_ARTIFACT_VERSION = 1
HIERARCHICAL_ARTIFACT_NAME = "selected_hierarchical.pth"
_HIERARCHICAL_INPUT_SHA_FIELDS = {
    "backbone",
    "parent",
    "geometry",
    "base_cache_content",
    "geometry_cache_content",
    "geometry_metadata",
}
_HIERARCHICAL_ARTIFACT_SELECTION_FIELDS = {
    "eligible",
    "selected",
    "hidden_dim",
    "weight_decay",
    "false_positive_cost",
    "margin_percentile",
    "margin",
    "switches",
    "delta_hits025",
    "delta_hits050",
}
_HIERARCHICAL_ARTIFACT_FIELDS = {
    "schema",
    "version",
    "deployable",
    "validation_data_accessed",
    "input_sha256",
    "feature_names",
    "model_config",
    "model_state_dict",
    "normalization",
    "normalization_sha256",
    "training_contract",
    "selection",
    "scene_folds",
    "scene_fold_sha256",
    "row_materialization_sha256",
    "candidate_iou_sha256",
    "oof_record",
    "oof_record_sha256",
    "calibration_record",
    "calibration_baseline",
}
_HIERARCHICAL_OOF_RECORD_FIELDS = {
    "prediction_count",
    "proposal_sha256",
    "gain_sha256",
    "delta_hits025",
    "delta_hits050",
}
HIERARCHICAL_RESULT_RECEIPT_SCHEMA = (
    "rec-hierarchical-result-receipt-v1"
)
HIERARCHICAL_RESULT_RECEIPT_VERSION = 1
_HIERARCHICAL_RESULT_CONTEXT_FIELDS = {
    "input_sha256",
    "split",
    "fit_joined_identity_sha256",
    "fit_materialization_sha256",
    "fit_deployable_sha256",
    "fit_candidate_iou_sha256",
    "fit_normalization_sha256",
    "oof",
    "calibration",
}
_HIERARCHICAL_RESULT_OOF_CONTEXT_FIELDS = {
    "baseline",
    "scene_folds",
    "scene_fold_sha256",
    "configuration_count",
    "configurations",
    "policy_candidate_count",
    "choice",
}
_HIERARCHICAL_RESULT_OOF_FIELDS = (
    _HIERARCHICAL_RESULT_OOF_CONTEXT_FIELDS
    | {"configuration_summaries_sha256", "choice_sha256"}
)
_HIERARCHICAL_RESULT_RECEIPT_FIELDS = (
    _HIERARCHICAL_RESULT_CONTEXT_FIELDS | {
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
)
_HIERARCHICAL_PROTECTED_IDENTITY_FIELDS = {
    "path", "device", "inode", "mode", "size", "mtime_ns", "ctime_ns",
    "sha256",
}
_HIERARCHICAL_OOF_BASELINE_FIELDS = {
    "sample_count",
    "hits025",
    "hits050",
    "oracle_hits025",
    "oracle_hits050",
    "candidate_iou_sha256",
    "row_materialization_sha256",
    "baseline_selected_iou_sha256",
}
_HIERARCHICAL_CONFIGURATION_FIELDS = {
    "hidden_dim",
    "weight_decay",
    "false_positive_cost",
    "configuration_index",
    "folds",
    "gain_summary",
    "oof_proposal_sha256",
    "oof_gain_sha256",
    "prediction_count",
}
_HIERARCHICAL_FOLD_FIELDS = {
    "fold",
    "fit_scene_count",
    "fit_row_count",
    "fit_query_count",
    "fit_variant_count",
    "held_scene_count",
    "held_row_count",
    "normalization_sha256",
    "normalization_counts",
    "training_labels",
}
_HIERARCHICAL_DIAGNOSTIC_FIELDS = {
    "hidden_dim",
    "weight_decay",
    "false_positive_cost",
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
    "transition_diagnostics",
}
_HIERARCHICAL_ELIGIBILITY_PREDICATE_FIELDS = {
    "not_no_switch",
    "all_folds_nonnegative025",
    "all_folds_nonnegative050",
    "pooled_delta025_positive",
    "bootstrap025_lower_bound_nonnegative",
    "bootstrap050_lower_bound_nonnegative",
}
_HIERARCHICAL_TRANSITION_FIELDS = {
    "selected_query_changes",
    "same_query_variant_changes",
    "wrong_query_recoveries025",
    "wrong_query_recoveries050",
    "wrong_variant_recoveries025",
    "wrong_variant_recoveries050",
}


def _require_cpu_tensor(value, name, shape, dtype):
    if (not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != dtype
            or tuple(value.shape) != tuple(shape)):
        raise ValueError(
            "hierarchical record {} layout is invalid".format(name)
        )


def _validate_hierarchical_records(records, require_contiguous=True):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("hierarchical records must be a nonempty sequence")
    if not isinstance(require_contiguous, bool):
        raise TypeError("require_contiguous must be boolean")
    previous_index = -1
    tensor_contract = {
        "query_features": ((QUERY_COUNT, QUERY_FEATURE_DIM), torch.float32),
        "variant_features": (
            (QUERY_COUNT, VARIANT_COUNT, VARIANT_FEATURE_DIM), torch.float32
        ),
        "query_aux_continuous": (
            (QUERY_COUNT, QUERY_AUX_CONTINUOUS_DIM), torch.float32
        ),
        "query_aux_binary": (
            (QUERY_COUNT, QUERY_AUX_BINARY_DIM), torch.bool
        ),
        "variant_aux_continuous": (
            (QUERY_COUNT, VARIANT_COUNT, VARIANT_AUX_CONTINUOUS_DIM),
            torch.float32,
        ),
        "variant_aux_binary": (
            (QUERY_COUNT, VARIANT_COUNT, VARIANT_AUX_BINARY_DIM), torch.bool
        ),
        "query_valid": ((QUERY_COUNT,), torch.bool),
        "variant_valid": ((QUERY_COUNT, VARIANT_COUNT), torch.bool),
        "candidate_ious": (
            (QUERY_COUNT, VARIANT_COUNT), torch.float32
        ),
        "baseline_scores": (
            (QUERY_COUNT * VARIANT_COUNT,), torch.float32
        ),
    }
    for expected_offset, record in enumerate(records):
        if (not isinstance(record, dict)
                or tuple(record.keys()) != HIERARCHICAL_RECORD_FIELDS):
            raise ValueError("hierarchical record fields do not match schema")
        dataset_index = record["dataset_index"]
        if (not isinstance(dataset_index, int)
                or isinstance(dataset_index, bool)
                or dataset_index <= previous_index
                or (require_contiguous
                    and dataset_index != expected_offset)):
            raise ValueError(
                "hierarchical record indices must be canonical and ordered"
            )
        previous_index = dataset_index
        if (not isinstance(record["scan_id"], str)
                or not record["scan_id"]
                or not isinstance(record["target_id"], int)
                or isinstance(record["target_id"], bool)
                or record["target_id"] < 0):
            raise ValueError("hierarchical record identity is invalid")
        for name, (shape, dtype) in tensor_contract.items():
            _require_cpu_tensor(record[name], name, shape, dtype)

        query_valid = record["query_valid"]
        variant_valid = record["variant_valid"]
        if (not torch.equal(query_valid, variant_valid.any(dim=1))
                or not bool(query_valid.any().item())):
            raise ValueError("hierarchical validity masks are inconsistent")
        for name, valid in (
                ("query_features", query_valid),
                ("query_aux_continuous", query_valid),
                ("variant_features", variant_valid),
                ("variant_aux_continuous", variant_valid)):
            value = record[name]
            if not bool(torch.isfinite(value[valid]).all().item()):
                raise ValueError(
                    "hierarchical {} has non-finite valid values".format(name)
                )
            if not torch.equal(
                    value[~valid], torch.zeros_like(value[~valid])):
                raise ValueError(
                    "hierarchical {} padding must be zero".format(name)
                )
        if (not torch.equal(
                record["query_aux_binary"][~query_valid],
                torch.zeros_like(
                    record["query_aux_binary"][~query_valid]
                ))
                or not torch.equal(
                    record["variant_aux_binary"][~variant_valid],
                    torch.zeros_like(
                        record["variant_aux_binary"][~variant_valid]
                    ))):
            raise ValueError("hierarchical binary padding must be false")
        candidate_ious = record["candidate_ious"]
        if (not bool(torch.isfinite(candidate_ious).all().item())
                or bool((candidate_ious < 0.0).any().item())
                or bool((candidate_ious > 1.0).any().item())):
            raise ValueError("hierarchical candidate IoUs must lie in [0,1]")
        flat_valid = variant_valid.reshape(-1)
        scores = record["baseline_scores"]
        if (not bool(torch.isfinite(scores[flat_valid]).all().item())
                or not bool(torch.isneginf(scores[~flat_valid]).all().item())):
            raise ValueError("hierarchical baseline score mask is invalid")
        baseline_index = record["baseline_index"]
        if (not isinstance(baseline_index, int)
                or isinstance(baseline_index, bool)
                or not 0 <= baseline_index < QUERY_COUNT * VARIANT_COUNT
                or not bool(flat_valid[baseline_index].item())):
            raise ValueError("hierarchical baseline index is invalid")
        selected = _stable_flat_top1_indices(
            scores.unsqueeze(0), flat_valid.unsqueeze(0)
        ).item()
        if selected != baseline_index:
            raise ValueError("hierarchical baseline selection is inconsistent")
        query_binary = record["query_aux_binary"]
        variant_binary = record["variant_aux_binary"]
        if (query_binary[:, 0].sum().item() != 1
                or query_binary[:, 1].sum().item() != 1
                or variant_binary[..., 0].sum().item() != 1
                or variant_binary[..., 1].sum().item() != 1
                or not bool(variant_binary[..., 1].reshape(-1)[
                    baseline_index
                ].item())):
            raise ValueError("hierarchical Top-1 indicators are inconsistent")


def _canonical_axes(batch_size, device):
    query_positions = torch.arange(
        QUERY_COUNT, dtype=torch.long, device=device
    ).view(1, QUERY_COUNT, 1).expand(
        batch_size, QUERY_COUNT, VARIANT_COUNT
    ).reshape(batch_size, -1)
    variant_indices = torch.arange(
        VARIANT_COUNT, dtype=torch.long, device=device
    ).view(1, 1, VARIANT_COUNT).expand(
        batch_size, QUERY_COUNT, VARIANT_COUNT
    ).reshape(batch_size, -1)
    return query_positions, variant_indices


def _one_hot_flat(indices, batch_size, device):
    positions = torch.arange(
        QUERY_COUNT * VARIANT_COUNT, dtype=torch.long, device=device
    ).unsqueeze(0)
    return positions.eq(indices.reshape(batch_size, 1))


def materialize_hierarchical_rows(
        rows, parent, geometry_model, geometry_artifact,
        batch_size=HIERARCHICAL_MATERIALIZATION_BATCH_SIZE,
        device="cuda:0", require_contiguous=True,
        artifact_validator=None):
    """Materialize deployable hierarchy inputs plus detached train labels."""
    _validate_joined_row_order(
        rows, require_contiguous=require_contiguous
    )
    if artifact_validator is None:
        artifact_validator = _validate_materialization_artifact
    if not callable(artifact_validator):
        raise TypeError("artifact_validator must be callable")
    artifact_validator(geometry_artifact)
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
    if (GEOMETRY_CANDIDATE_COUNT != QUERY_COUNT
            or GEOMETRY_VARIANT_COUNT != VARIANT_COUNT
            or FLAT_GEOMETRY_CANDIDATE_COUNT
            != QUERY_COUNT * VARIANT_COUNT
            or GEOMETRY_INPUT_DIM
            != QUERY_FEATURE_DIM + VARIANT_FEATURE_DIM + 2):
        raise RuntimeError("upstream geometry tensor contract changed")

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
            current_size = len(row_batch)
            batch = build_geometry_training_batch(row_batch, parent)
            if list(batch["feature_names"]) != geometry_artifact[
                    "feature_names"]:
                raise ValueError("live geometry feature schema changed")
            expected_query_positions, expected_variant_indices = \
                _canonical_axes(current_size, batch["features"].device)
            if (not torch.equal(
                    batch["query_positions"], expected_query_positions)
                    or not torch.equal(
                        batch["variant_indices"], expected_variant_indices
                    )):
                raise ValueError("live geometry candidate axes changed")

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
                outputs, current_size, resolved_device, valid
            )
            _require_frozen_model(geometry_model, "geometry model")
            _require_frozen_model(parent_model, "parent model")

            parent_state = {
                key: value.to(resolved_device)
                if isinstance(value, torch.Tensor) else value
                for key, value in batch["parent_state"].items()
            }
            variant_valid = valid.reshape(
                current_size, QUERY_COUNT, VARIANT_COUNT
            )
            query_valid = variant_valid.any(dim=2)
            regressed_variant_index = geometry_artifact[
                "regressed_variant_index"
            ]
            if not torch.equal(
                    query_valid, parent_state["candidate_valid"]):
                raise ValueError(
                    "hierarchical query validity differs from parent state"
                )

            parent_prior = build_flat_parent_prior(
                parent_state,
                variant_valid,
                regressed_variant_index,
            )
            flat_parent_rank = _stable_rank_normalize_once(
                parent_prior, valid
            )
            geometry_rank = _stable_rank_normalize_once(
                outputs["ranking_logits"], valid
            )
            weight = float(geometry_artifact["geometry_weight"])
            baseline_scores = (
                (1.0 - weight) * flat_parent_rank
                + weight * geometry_rank
            ).masked_fill(~valid, -float("inf"))
            baseline_indices = _stable_flat_top1_indices(
                baseline_scores, valid
            )
            geometry_indices = _stable_flat_top1_indices(
                outputs["ranking_logits"].masked_fill(
                    ~valid, -float("inf")
                ),
                valid,
            )

            default_scores = torch.stack([
                row["base"]["default_scores"] for row in row_batch
            ]).to(resolved_device, dtype=torch.float32)
            if default_scores.shape != (current_size, QUERY_COUNT):
                raise ValueError("default score layout changed")
            if not bool(torch.isfinite(
                    default_scores[query_valid]
            ).all().item()):
                raise ValueError("valid default scores must be finite")
            default_rank = _stable_rank_normalize_once(
                default_scores, query_valid
            )
            parent_scores = parent_state["compact_scores"]
            parent_rank = _stable_rank_normalize_once(
                parent_scores, query_valid
            )
            query_indices = parent_state["query_indices"]
            default_state = build_deployed_parent_state(
                default_scores,
                query_indices,
                query_valid,
                parent_state["query_scores"].shape[1],
            )
            default_top1 = default_state["parent_top1_mask"]
            parent_top1 = parent_state["parent_top1_mask"] & query_valid
            if (not bool(default_top1.sum(dim=1).eq(1).all().item())
                    or not bool(parent_top1.sum(dim=1).eq(1).all().item())):
                raise ValueError("query Top-1 identity changed")

            structured_raw = raw_features.reshape(
                current_size, QUERY_COUNT, VARIANT_COUNT, GEOMETRY_INPUT_DIM
            )
            query_features = torch.stack([
                row["base"]["features"] for row in row_batch
            ]).to(resolved_device, dtype=torch.float32)
            if query_features.shape != (
                    current_size, QUERY_COUNT, QUERY_FEATURE_DIM):
                raise ValueError(
                    "base query features changed hierarchical shape"
                )
            expected_query_features = query_features.unsqueeze(2).expand(
                -1, -1, VARIANT_COUNT, -1
            )
            if not torch.equal(
                    structured_raw[..., :QUERY_FEATURE_DIM][variant_valid],
                    expected_query_features[variant_valid]):
                raise ValueError("query features differ across valid variants")
            variant_features = structured_raw[
                ...,
                QUERY_FEATURE_DIM:
                QUERY_FEATURE_DIM + VARIANT_FEATURE_DIM,
            ]
            query_mask = query_valid.unsqueeze(-1)
            variant_mask = variant_valid.unsqueeze(-1)
            query_features = torch.where(
                query_mask,
                query_features,
                torch.zeros_like(query_features),
            )
            variant_features = torch.where(
                variant_mask,
                variant_features,
                torch.zeros_like(variant_features),
            )
            query_aux_continuous = torch.stack((
                default_scores,
                default_rank,
                parent_scores,
                parent_rank,
            ), dim=-1)
            query_aux_continuous = torch.where(
                query_mask,
                query_aux_continuous,
                torch.zeros_like(query_aux_continuous),
            )
            query_aux_binary = torch.stack(
                (default_top1, parent_top1), dim=-1
            ) & query_mask
            structured_geometry_scores = outputs[
                "ranking_logits"
            ].reshape(current_size, QUERY_COUNT, VARIANT_COUNT)
            structured_geometry_rank = geometry_rank.reshape(
                current_size, QUERY_COUNT, VARIANT_COUNT
            )
            variant_aux_continuous = torch.stack((
                structured_geometry_scores,
                structured_geometry_rank,
            ), dim=-1)
            variant_aux_continuous = torch.where(
                variant_mask,
                variant_aux_continuous,
                torch.zeros_like(variant_aux_continuous),
            )
            geometry_top1 = _one_hot_flat(
                geometry_indices, current_size, resolved_device
            ).reshape(current_size, QUERY_COUNT, VARIANT_COUNT)
            baseline_top1 = _one_hot_flat(
                baseline_indices, current_size, resolved_device
            ).reshape(current_size, QUERY_COUNT, VARIANT_COUNT)
            variant_aux_binary = torch.stack(
                (geometry_top1, baseline_top1), dim=-1
            ) & variant_mask
            candidate_ious = batch["candidate_ious"].reshape(
                current_size, QUERY_COUNT, VARIANT_COUNT
            ).to(dtype=torch.float32)

            for local_index, joined_row in enumerate(row_batch):
                base = joined_row["base"]
                records.append({
                    "dataset_index": int(base["dataset_index"]),
                    "scan_id": base["scan_id"],
                    "target_id": int(base["target_id"]),
                    "query_features": query_features[
                        local_index
                    ].detach().cpu().clone(),
                    "variant_features": variant_features[
                        local_index
                    ].detach().cpu().clone(),
                    "query_aux_continuous": query_aux_continuous[
                        local_index
                    ].detach().cpu().clone(),
                    "query_aux_binary": query_aux_binary[
                        local_index
                    ].detach().cpu().clone(),
                    "variant_aux_continuous": variant_aux_continuous[
                        local_index
                    ].detach().cpu().clone(),
                    "variant_aux_binary": variant_aux_binary[
                        local_index
                    ].detach().cpu().clone(),
                    "query_valid": query_valid[
                        local_index
                    ].detach().cpu().clone(),
                    "variant_valid": variant_valid[
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
                })
    _validate_hierarchical_records(
        records, require_contiguous=require_contiguous
    )
    return records


def _digest_int(digest, value):
    digest.update(int(value).to_bytes(8, "little", signed=False))


def _digest_string(digest, value):
    encoded = value.encode("utf-8")
    _digest_int(digest, len(encoded))
    digest.update(encoded)


def _digest_tensor(digest, value):
    digest.update(value.contiguous().numpy().tobytes(order="C"))


def _canonical_hierarchical_digest(records, fields):
    _validate_hierarchical_records(records, require_contiguous=False)
    digest = hashlib.sha256()
    for record in records:
        _digest_int(digest, record["dataset_index"])
        _digest_string(digest, record["scan_id"])
        _digest_int(digest, record["target_id"])
        for field in fields:
            value = record[field]
            if isinstance(value, torch.Tensor):
                _digest_tensor(digest, value)
            else:
                _digest_int(digest, value)
    return digest.hexdigest()


def canonical_hierarchical_rows_sha256(records):
    """Hash identities, deployable fields, and training IoU labels."""
    return _canonical_hierarchical_digest(
        records, HIERARCHICAL_RECORD_FIELDS[3:]
    )


def canonical_hierarchical_deployable_sha256(records):
    """Hash identities and every inference-time hierarchy input."""
    fields = tuple(
        field for field in HIERARCHICAL_RECORD_FIELDS[3:]
        if field != "candidate_ious"
    )
    return _canonical_hierarchical_digest(records, fields)


def canonical_hierarchical_candidate_iou_sha256(records):
    """Hash ordered candidate IoU labels independently of features."""
    return _canonical_hierarchical_digest(records, ("candidate_ious",))


def _update_streaming_population(count, mean, m2, values):
    values = values.to(dtype=torch.float64, device="cpu")
    batch_count = int(values.shape[0])
    if batch_count == 0:
        return count, mean, m2
    batch_mean = values.mean(dim=0)
    centered = values - batch_mean
    batch_m2 = (centered * centered).sum(dim=0)
    if count == 0:
        return batch_count, batch_mean, batch_m2
    total = count + batch_count
    delta = batch_mean - mean
    combined_mean = mean + delta * (batch_count / float(total))
    combined_m2 = (
        m2
        + batch_m2
        + delta * delta * (count * batch_count / float(total))
    )
    return total, combined_mean, combined_m2


def _normalization_sha256(statistics):
    digest = hashlib.sha256()
    _digest_string(digest, statistics["schema"])
    digest.update(struct.pack("<d", float(statistics["minimum_std"])))
    for group_name in HIERARCHICAL_NORMALIZATION_GROUPS:
        group = statistics["groups"][group_name]
        _digest_string(digest, group_name)
        _digest_int(digest, group["count"])
        _digest_int(digest, len(group["feature_names"]))
        for feature_name in group["feature_names"]:
            _digest_string(digest, feature_name)
        _digest_tensor(digest, group["mean"])
        _digest_tensor(digest, group["std"])
    return digest.hexdigest()


def _validate_hierarchical_normalization(statistics):
    if not isinstance(statistics, dict) or set(statistics) != {
            "schema", "minimum_std", "groups", "sha256"}:
        raise ValueError("hierarchical normalization fields changed")
    if statistics["schema"] != HIERARCHICAL_NORMALIZATION_SCHEMA:
        raise ValueError("hierarchical normalization schema changed")
    minimum_std = statistics["minimum_std"]
    if (isinstance(minimum_std, bool)
            or not isinstance(minimum_std, (float, int))
            or float(minimum_std) != HIERARCHICAL_MIN_STD):
        raise ValueError("hierarchical minimum std changed")
    groups = statistics["groups"]
    if (not isinstance(groups, dict)
            or tuple(groups.keys()) != HIERARCHICAL_NORMALIZATION_GROUPS):
        raise ValueError("hierarchical normalization groups changed")
    for group_name in HIERARCHICAL_NORMALIZATION_GROUPS:
        group = groups[group_name]
        if not isinstance(group, dict) or set(group) != {
                "count", "feature_names", "mean", "std"}:
            raise ValueError(
                "hierarchical {} statistics fields changed".format(
                    group_name
                )
            )
        count = group["count"]
        if (not isinstance(count, int) or isinstance(count, bool)
                or count <= 0):
            raise ValueError(
                "hierarchical {} statistics count is invalid".format(
                    group_name
                )
            )
        expected_names = list(
            _HIERARCHICAL_NORMALIZATION_FEATURE_NAMES[group_name]
        )
        if group["feature_names"] != expected_names:
            raise ValueError(
                "hierarchical {} feature schema changed".format(group_name)
            )
        dimension = len(expected_names)
        for field in ("mean", "std"):
            value = group[field]
            if (not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.dtype != torch.float32
                    or value.shape != (dimension,)
                    or not bool(torch.isfinite(value).all().item())):
                raise ValueError(
                    "hierarchical {} {} is invalid".format(
                        group_name, field
                    )
                )
        if bool((group["std"] < HIERARCHICAL_MIN_STD).any().item()):
            raise ValueError(
                "hierarchical {} std is below its floor".format(group_name)
            )
    sha256 = statistics["sha256"]
    if (not isinstance(sha256, str) or len(sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in sha256)
            or sha256 != _normalization_sha256(statistics)):
        raise ValueError("hierarchical normalization SHA-256 mismatch")


def fit_hierarchical_normalization(records):
    """Fit fold-local float64 population statistics on valid entries."""
    _validate_hierarchical_records(records, require_contiguous=False)
    groups = {}
    for group_name in HIERARCHICAL_NORMALIZATION_GROUPS:
        dimension = len(
            _HIERARCHICAL_NORMALIZATION_FEATURE_NAMES[group_name]
        )
        count = 0
        mean = torch.zeros(dimension, dtype=torch.float64)
        m2 = torch.zeros(dimension, dtype=torch.float64)
        mask_name = _HIERARCHICAL_NORMALIZATION_MASKS[group_name]
        for record in records:
            count, mean, m2 = _update_streaming_population(
                count,
                mean,
                m2,
                record[group_name][record[mask_name]],
            )
        if count <= 0:
            raise ValueError(
                "hierarchical {} has no valid fit values".format(group_name)
            )
        variance = (m2 / float(count)).clamp_min(0.0)
        std = variance.sqrt().clamp_min(HIERARCHICAL_MIN_STD)
        groups[group_name] = {
            "count": count,
            "feature_names": list(
                _HIERARCHICAL_NORMALIZATION_FEATURE_NAMES[group_name]
            ),
            "mean": mean.to(dtype=torch.float32),
            "std": std.to(dtype=torch.float32),
        }
    statistics = {
        "schema": HIERARCHICAL_NORMALIZATION_SCHEMA,
        "minimum_std": HIERARCHICAL_MIN_STD,
        "groups": groups,
    }
    statistics["sha256"] = _normalization_sha256(statistics)
    _validate_hierarchical_normalization(statistics)
    return statistics


def _validate_hierarchical_model_batch(batch):
    if (not isinstance(batch, dict)
            or tuple(batch.keys()) != HIERARCHICAL_MODEL_BATCH_FIELDS):
        raise ValueError(
            "hierarchical model batch fields do not match the contract"
        )
    for field in HIERARCHICAL_NORMALIZATION_GROUPS:
        value = batch[field]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise TypeError("{} must have float32 dtype".format(field))
    for field in (
            "query_aux_binary", "variant_aux_binary",
            "query_valid", "variant_valid"):
        value = batch[field]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
            raise TypeError("{} must have bool dtype".format(field))
    query_features = batch["query_features"]
    if query_features.dim() != 3 or query_features.shape[0] <= 0:
        raise ValueError("query_features must have nonempty shape [B,16,152]")
    batch_size = query_features.shape[0]
    shapes = {
        "query_features": (batch_size, QUERY_COUNT, QUERY_FEATURE_DIM),
        "variant_features": (
            batch_size, QUERY_COUNT, VARIANT_COUNT, VARIANT_FEATURE_DIM
        ),
        "query_aux_continuous": (
            batch_size, QUERY_COUNT, QUERY_AUX_CONTINUOUS_DIM
        ),
        "query_aux_binary": (
            batch_size, QUERY_COUNT, QUERY_AUX_BINARY_DIM
        ),
        "variant_aux_continuous": (
            batch_size, QUERY_COUNT, VARIANT_COUNT,
            VARIANT_AUX_CONTINUOUS_DIM,
        ),
        "variant_aux_binary": (
            batch_size, QUERY_COUNT, VARIANT_COUNT,
            VARIANT_AUX_BINARY_DIM,
        ),
        "query_valid": (batch_size, QUERY_COUNT),
        "variant_valid": (batch_size, QUERY_COUNT, VARIANT_COUNT),
    }
    for field, expected in shapes.items():
        if tuple(batch[field].shape) != expected:
            raise ValueError(
                "{} must have shape {}".format(field, expected)
            )
    devices = {value.device for value in batch.values()}
    if len(devices) != 1:
        raise ValueError("hierarchical model batch tensors must share a device")
    query_valid = batch["query_valid"]
    variant_valid = batch["variant_valid"]
    if (not torch.equal(query_valid, variant_valid.any(dim=2))
            or not bool(query_valid.any(dim=1).all().item())):
        raise ValueError("hierarchical model batch masks are inconsistent")
    for field, mask_name in _HIERARCHICAL_NORMALIZATION_MASKS.items():
        if not bool(torch.isfinite(
                batch[field][batch[mask_name]]
        ).all().item()):
            raise ValueError("{} must be finite at valid entries".format(field))


def normalize_hierarchical_batch(batch, statistics):
    """Apply bound fold-local statistics and zero all invalid padding."""
    _validate_hierarchical_normalization(statistics)
    _validate_hierarchical_model_batch(batch)
    normalized = {}
    for field in HIERARCHICAL_MODEL_BATCH_FIELDS:
        if field not in HIERARCHICAL_NORMALIZATION_GROUPS:
            normalized[field] = batch[field]
            continue
        value = batch[field]
        mask = batch[_HIERARCHICAL_NORMALIZATION_MASKS[field]]
        group = statistics["groups"][field]
        mean = group["mean"].to(device=value.device, dtype=value.dtype)
        std = group["std"].to(device=value.device, dtype=value.dtype)
        safe_value = torch.where(
            mask.unsqueeze(-1), value, torch.zeros_like(value)
        )
        transformed = (safe_value - mean) / std
        normalized[field] = torch.where(
            mask.unsqueeze(-1),
            transformed,
            torch.zeros_like(transformed),
        )
    return normalized


def summarize_hierarchical_training_labels(records):
    """Count strict query/variant targets at both fixed thresholds."""
    _validate_hierarchical_records(records, require_contiguous=False)
    candidate_ious = torch.stack([
        record["candidate_ious"] for record in records
    ])
    variant_valid = torch.stack([
        record["variant_valid"] for record in records
    ])
    targets = build_hierarchical_targets(candidate_ious, variant_valid)
    summary = {}
    for level, target_name, valid in (
            ("query", "query_targets", targets["query_valid"]),
            ("variant", "variant_targets", variant_valid)):
        threshold_summary = {}
        values = targets[target_name]
        for threshold_index, threshold in enumerate(("0.25", "0.50")):
            labels = values[..., threshold_index][valid]
            positive = int(labels.sum().item())
            total = int(labels.numel())
            threshold_summary[threshold] = {
                "positive": positive,
                "negative": total - positive,
                "total": total,
            }
        summary[level] = threshold_summary
    return summary


def _set_hierarchical_seed(device):
    random.seed(HIERARCHICAL_MODEL_SEED)
    torch.manual_seed(HIERARCHICAL_MODEL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(HIERARCHICAL_MODEL_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _stack_hierarchical_model_batch(records):
    if not records:
        raise ValueError("hierarchical model batch cannot be empty")
    return {
        field: torch.stack([record[field] for record in records])
        for field in HIERARCHICAL_MODEL_BATCH_FIELDS
    }


def _normalized_hierarchical_model_batch(records, statistics, device):
    batch = _stack_hierarchical_model_batch(records)
    normalized = normalize_hierarchical_batch(batch, statistics)
    return {
        field: value.to(device) for field, value in normalized.items()
    }


def _fit_hierarchical_model(
        records, statistics, hidden_dim, weight_decay,
        false_positive_cost, device, batch_observer=None,
        observer_context=None):
    if not records:
        raise ValueError("hierarchical fit records cannot be empty")
    _validate_hierarchical_records(records, require_contiguous=False)
    _validate_hierarchical_normalization(statistics)
    if batch_observer is not None and not callable(batch_observer):
        raise TypeError("batch_observer must be callable")
    observer_context = dict(observer_context or {})
    _set_hierarchical_seed(device)
    model = HierarchicalQueryVariantReranker(
        hidden_dim=int(hidden_dim), dropout=HIERARCHICAL_DROPOUT
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HIERARCHICAL_LEARNING_RATE,
        weight_decay=float(weight_decay),
    )
    shuffle_state = random.Random(HIERARCHICAL_MODEL_SEED)
    for epoch in range(HIERARCHICAL_EPOCHS):
        model.train()
        order = list(range(len(records)))
        shuffle_state.shuffle(order)
        for start in range(0, len(order), HIERARCHICAL_BATCH_SIZE):
            indices = order[start:start + HIERARCHICAL_BATCH_SIZE]
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
            model_batch = _normalized_hierarchical_model_batch(
                row_batch, statistics, device
            )
            candidate_ious = torch.stack([
                record["candidate_ious"] for record in row_batch
            ]).to(device)
            targets = build_hierarchical_targets(
                candidate_ious, model_batch["variant_valid"]
            )
            outputs = model(**model_batch)
            loss, _stats = compute_hierarchical_loss(
                query_logits=outputs["query_logits"],
                variant_logits=outputs["variant_logits"],
                query_targets=targets["query_targets"],
                variant_targets=targets["variant_targets"],
                query_valid=model_batch["query_valid"],
                variant_valid=model_batch["variant_valid"],
                false_positive_cost=float(false_positive_cost),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(
                model.parameters(), HIERARCHICAL_GRAD_CLIP_NORM
            )
            optimizer.step()
    model.eval().requires_grad_(False)
    return model


def _predict_hierarchical_proposals(
        model, records, statistics, device):
    if not records:
        raise ValueError("hierarchical prediction records cannot be empty")
    _validate_hierarchical_records(records, require_contiguous=False)
    _validate_hierarchical_normalization(statistics)
    model.to(device).eval()
    proposals = []
    gains = []
    with torch.no_grad():
        for start in range(0, len(records), HIERARCHICAL_BATCH_SIZE):
            row_batch = records[start:start + HIERARCHICAL_BATCH_SIZE]
            model_batch = _normalized_hierarchical_model_batch(
                row_batch, statistics, device
            )
            outputs = model(**model_batch)
            selected = select_hierarchical_proposal(
                outputs["query_logits"],
                outputs["variant_logits"],
                model_batch["query_valid"],
                model_batch["variant_valid"],
            )
            flat_utility = selected["variant_utility"].reshape(
                len(row_batch), QUERY_COUNT * VARIANT_COUNT
            )
            baseline_indices = torch.tensor([
                record["baseline_index"] for record in row_batch
            ], dtype=torch.long, device=device)
            rows = torch.arange(len(row_batch), device=device)
            proposal_indices = selected["flat_indices"]
            proposal_gain = (
                flat_utility[rows, proposal_indices]
                - flat_utility[rows, baseline_indices]
            )
            flat_valid = model_batch["variant_valid"].reshape(
                len(row_batch), -1
            )
            if not bool(flat_valid[rows, proposal_indices].all().item()):
                raise RuntimeError("hierarchical model proposed invalid variants")
            if not bool(torch.isfinite(proposal_gain).all().item()):
                raise RuntimeError("hierarchical proposal gain is non-finite")
            proposals.append(proposal_indices.detach().cpu().long())
            gains.append(proposal_gain.detach().cpu().float())
    return torch.cat(proposals), torch.cat(gains)


def _gain_distribution_statistics(values):
    if not int(values.numel()):
        return None
    ordered = values.sort().values
    quantiles = []
    for quantile in HIERARCHICAL_GAIN_QUANTILES:
        rank = int(math.ceil(quantile * len(ordered)))
        rank = min(max(rank, 1), len(ordered))
        quantiles.append({
            "quantile": float(quantile),
            "value": float(ordered[rank - 1].item()),
        })
    result = {
        "minimum": float(ordered[0].item()),
        "maximum": float(ordered[-1].item()),
        "mean": float(values.mean().item()),
        "population_standard_deviation": float(
            values.std(unbiased=False).item()
        ),
        "nearest_rank_quantiles": quantiles,
    }
    scalar_values = (
        result["minimum"], result["maximum"], result["mean"],
        result["population_standard_deviation"],
    )
    if (not all(math.isfinite(value) for value in scalar_values)
            or not all(math.isfinite(item["value"])
                       for item in quantiles)):
        raise ValueError("hierarchical gain statistics must be finite")
    return result


def summarize_hierarchical_oof_gain(gain):
    if (not isinstance(gain, torch.Tensor)
            or gain.device.type != "cpu"
            or gain.dtype != torch.float32
            or gain.dim() != 1
            or gain.shape[0] <= 0
            or not bool(torch.isfinite(gain).all().item())):
        raise ValueError("hierarchical OOF gain layout is invalid")
    positive = gain[gain.gt(0.0)]
    return {
        "all": {
            "count": int(gain.numel()),
            "statistics": _gain_distribution_statistics(gain),
        },
        "positive": {
            "count": int(positive.numel()),
            "statistics": _gain_distribution_statistics(positive),
        },
    }


def nearest_rank_hierarchical_margin(gain, percentile):
    """Return the nearest-rank percentile among positive row gains."""
    if (not isinstance(gain, torch.Tensor)
            or gain.dtype != torch.float32
            or gain.dim() != 1
            or gain.shape[0] <= 0
            or not bool(torch.isfinite(gain).all().item())):
        raise ValueError("hierarchical gain must be a finite float32 vector")
    if (isinstance(percentile, bool)
            or not isinstance(percentile, (int, float))
            or not math.isfinite(float(percentile))
            or not 0.0 <= float(percentile) <= 100.0):
        raise ValueError("percentile must lie in [0,100]")
    positive = gain[gain.gt(0.0)]
    if not int(positive.numel()):
        return None
    ordered = positive.sort().values
    rank = int(math.ceil(float(percentile) / 100.0 * len(ordered)))
    rank = min(max(rank, 1), len(ordered))
    margin = float(ordered[rank - 1].item())
    if not math.isfinite(margin) or margin <= 0.0:
        raise RuntimeError("positive hierarchical margin is invalid")
    return margin


def _validate_oof_prediction_vectors(records, proposals, gain):
    if (not isinstance(proposals, torch.Tensor)
            or proposals.device.type != "cpu"
            or proposals.dtype != torch.long
            or tuple(proposals.shape) != (len(records),)):
        raise ValueError("OOF proposal layout is invalid")
    if (not isinstance(gain, torch.Tensor)
            or gain.device.type != "cpu"
            or gain.dtype != torch.float32
            or tuple(gain.shape) != (len(records),)
            or not bool(torch.isfinite(gain).all().item())):
        raise ValueError("OOF gain layout is invalid")
    if (bool((proposals < 0).any().item())
            or bool((proposals >= QUERY_COUNT * VARIANT_COUNT).any().item())):
        raise ValueError("OOF proposals are out of range")
    valid = torch.stack([
        record["variant_valid"].reshape(-1) for record in records
    ])
    rows = torch.arange(len(records))
    if not bool(valid[rows, proposals].all().item()):
        raise ValueError("OOF proposals must identify valid variants")


def _canonical_oof_vector_sha256(records, values):
    digest = hashlib.sha256()
    for record, value in zip(records, values):
        _digest_int(digest, record["dataset_index"])
        _digest_string(digest, record["scan_id"])
        _digest_tensor(digest, value.reshape(1))
    return digest.hexdigest()


def build_hierarchical_policy_candidate(
        records, proposals, gain, config, percentile, margin):
    """Build one fixed-margin OOF policy record and transition diagnostics."""
    _validate_hierarchical_records(records, require_contiguous=False)
    _validate_oof_prediction_vectors(records, proposals, gain)
    if not isinstance(config, dict) or set(config) != {
            "hidden_dim", "weight_decay", "false_positive_cost"}:
        raise ValueError("hierarchical configuration fields changed")
    base_scores = torch.stack([
        record["baseline_scores"] for record in records
    ])
    variant_valid = torch.stack([
        record["variant_valid"] for record in records
    ])
    baseline_indices = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    if percentile is None:
        if not math.isinf(float(margin)) or float(margin) < 0.0:
            raise ValueError("no-switch margin must be +inf")
        selected_indices = baseline_indices
        switch_mask = torch.zeros(len(records), dtype=torch.bool)
    else:
        policy = apply_hierarchical_policy(
            base_scores,
            proposals,
            gain,
            variant_valid,
            margin,
        )
        selected_indices = policy["selected_indices"]
        switch_mask = policy["switch_mask"]
    candidate_ious = torch.stack([
        record["candidate_ious"].reshape(-1) for record in records
    ])
    rows = torch.arange(len(records))
    baseline_ious = candidate_ious[rows, baseline_indices]
    proposed_ious = candidate_ious[rows, selected_indices]
    baseline_queries = baseline_indices.div(
        VARIANT_COUNT, rounding_mode="floor"
    )
    selected_queries = selected_indices.div(
        VARIANT_COUNT, rounding_mode="floor"
    )
    query_changes = switch_mask & selected_queries.ne(baseline_queries)
    variant_changes = switch_mask & selected_queries.eq(baseline_queries)
    transition_diagnostics = {
        "selected_query_changes": int(query_changes.sum().item()),
        "same_query_variant_changes": int(variant_changes.sum().item()),
    }
    for suffix, threshold in (("025", 0.25), ("050", 0.50)):
        fixes = baseline_ious.le(threshold) & proposed_ious.gt(threshold)
        transition_diagnostics[
            "wrong_query_recoveries{}".format(suffix)
        ] = int((fixes & query_changes).sum().item())
        transition_diagnostics[
            "wrong_variant_recoveries{}".format(suffix)
        ] = int((fixes & variant_changes).sum().item())
    return {
        "hidden_dim": config["hidden_dim"],
        "weight_decay": config["weight_decay"],
        "false_positive_cost": config["false_positive_cost"],
        "margin_percentile": percentile,
        "margin": margin,
        "scan_ids": [record["scan_id"] for record in records],
        "baseline_hits025": baseline_ious.gt(0.25).long().tolist(),
        "proposed_hits025": proposed_ious.gt(0.25).long().tolist(),
        "baseline_hits050": baseline_ious.gt(0.50).long().tolist(),
        "proposed_hits050": proposed_ious.gt(0.50).long().tolist(),
        "switch_bits": switch_mask.long().tolist(),
        "transition_diagnostics": transition_diagnostics,
    }


def cross_fit_hierarchical_reranker(
        records, device="cpu", batch_observer=None,
        normalization_observer=None,
        selector=choose_hierarchical_configuration):
    """Produce scene-disjoint OOF proposals for the fixed eight-model grid."""
    _validate_hierarchical_records(records, require_contiguous=False)
    if batch_observer is not None and not callable(batch_observer):
        raise TypeError("batch_observer must be callable")
    if (normalization_observer is not None
            and not callable(normalization_observer)):
        raise TypeError("normalization_observer must be callable")
    if not callable(selector):
        raise TypeError("selector must be callable")
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("cross-fit supports only CPU and CUDA")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA cross-fit requested but unavailable")
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    configurations = []
    policy_candidates = []
    grid = tuple(itertools.product(
        HIERARCHICAL_HIDDEN_DIMS,
        HIERARCHICAL_WEIGHT_DECAYS,
        HIERARCHICAL_FALSE_POSITIVE_COSTS,
    ))
    if len(grid) != 8:
        raise RuntimeError("hierarchical fixed grid changed")
    for config_index, (
            hidden_dim, weight_decay, false_positive_cost) in enumerate(grid):
        config = {
            "hidden_dim": int(hidden_dim),
            "weight_decay": float(weight_decay),
            "false_positive_cost": float(false_positive_cost),
        }
        oof_proposals = torch.full(
            (len(records),), -1, dtype=torch.long
        )
        oof_gain = torch.zeros(len(records), dtype=torch.float32)
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
                raise ValueError(
                    "every hierarchical fold needs fit and held rows"
                )
            fit_records = [records[index] for index in fit_indices]
            held_records = [records[index] for index in held_indices]
            statistics = fit_hierarchical_normalization(fit_records)
            observer_context = {
                "phase": "cross_fit",
                "config_index": config_index,
                "held_out_fold": held_out_fold,
            }
            if normalization_observer is not None:
                event = dict(observer_context)
                event.update({
                    "scan_ids": tuple(
                        record["scan_id"] for record in fit_records
                    ),
                    "normalization_sha256": statistics["sha256"],
                })
                normalization_observer(event)
            labels = summarize_hierarchical_training_labels(fit_records)
            fit_query_count = sum(
                int(record["query_valid"].sum().item())
                for record in fit_records
            )
            fit_variant_count = sum(
                int(record["variant_valid"].sum().item())
                for record in fit_records
            )
            if any(
                    labels["query"][threshold]["total"]
                    != fit_query_count
                    or labels["variant"][threshold]["total"]
                    != fit_variant_count
                    for threshold in ("0.25", "0.50")):
                raise RuntimeError("fold training label counts changed")
            model = _fit_hierarchical_model(
                fit_records,
                statistics,
                hidden_dim=hidden_dim,
                weight_decay=weight_decay,
                false_positive_cost=false_positive_cost,
                device=resolved_device,
                batch_observer=batch_observer,
                observer_context=observer_context,
            )
            held_proposals, held_gain = _predict_hierarchical_proposals(
                model, held_records, statistics, resolved_device
            )
            oof_proposals[held_indices] = held_proposals
            oof_gain[held_indices] = held_gain
            predicted[held_indices] = True
            fold_records.append({
                "fold": held_out_fold,
                "fit_scene_count": len({
                    record["scan_id"] for record in fit_records
                }),
                "fit_row_count": len(fit_records),
                "fit_query_count": fit_query_count,
                "fit_variant_count": fit_variant_count,
                "held_scene_count": len({
                    record["scan_id"] for record in held_records
                }),
                "held_row_count": len(held_records),
                "normalization_sha256": statistics["sha256"],
                "normalization_counts": {
                    group_name: statistics["groups"][group_name]["count"]
                    for group_name in HIERARCHICAL_NORMALIZATION_GROUPS
                },
                "training_labels": copy.deepcopy(labels),
            })
            del model
        if not bool(predicted.all().item()):
            raise RuntimeError(
                "OOF predictions do not cover every hierarchical fit row"
            )
        _validate_oof_prediction_vectors(records, oof_proposals, oof_gain)
        config_record = dict(config)
        config_record.update({
            "configuration_index": config_index,
            "folds": fold_records,
            "gain_summary": summarize_hierarchical_oof_gain(oof_gain),
            "oof_proposal_sha256": _canonical_oof_vector_sha256(
                records, oof_proposals
            ),
            "oof_gain_sha256": _canonical_oof_vector_sha256(
                records, oof_gain
            ),
            "prediction_count": int(predicted.sum().item()),
        })
        configurations.append(config_record)
        policy_candidates.append(build_hierarchical_policy_candidate(
            records,
            oof_proposals,
            oof_gain,
            config,
            None,
            float("inf"),
        ))
        for percentile in HIERARCHICAL_MARGIN_PERCENTILES:
            margin = nearest_rank_hierarchical_margin(
                oof_gain, percentile
            )
            if margin is not None:
                policy_candidates.append(build_hierarchical_policy_candidate(
                    records,
                    oof_proposals,
                    oof_gain,
                    config,
                    percentile,
                    margin,
                ))
    choice = selector(policy_candidates)
    if not isinstance(choice, dict) or "eligible" not in choice:
        raise ValueError("selector returned an invalid hierarchical choice")
    return {
        "scene_folds": scene_folds,
        "scene_fold_sha256": canonical_hierarchical_scene_fold_sha256(
            scene_folds
        ),
        "configurations": configurations,
        "policy_candidate_count": len(policy_candidates),
        "choice": choice,
    }


def _validate_hierarchical_refit_choice(choice):
    if (not isinstance(choice, dict)
            or choice.get("eligible") is not True
            or choice.get("selected") != "hierarchical"):
        raise ValueError("refit requires one frozen eligible OOF choice")
    for name, allowed in (
            ("hidden_dim", HIERARCHICAL_HIDDEN_DIMS),
            ("weight_decay", HIERARCHICAL_WEIGHT_DECAYS),
            ("false_positive_cost", HIERARCHICAL_FALSE_POSITIVE_COSTS)):
        value = choice.get(name)
        if isinstance(value, bool) or value not in allowed:
            raise ValueError(
                "refit choice {} is outside the fixed grid".format(name)
            )
    percentile = choice.get("margin_percentile")
    margin = choice.get("margin")
    if (isinstance(percentile, bool)
            or percentile not in HIERARCHICAL_MARGIN_PERCENTILES
            or isinstance(margin, bool)
            or not isinstance(margin, (float, int))
            or not math.isfinite(float(margin))
            or float(margin) <= 0.0):
        raise ValueError("refit choice has an invalid intervention gate")


def refit_hierarchical_reranker(
        records, choice, device="cpu", batch_observer=None,
        normalization_observer=None):
    """Refit the OOF-selected model once using all fit-scene statistics."""
    _validate_hierarchical_records(records, require_contiguous=False)
    _validate_hierarchical_refit_choice(choice)
    if batch_observer is not None and not callable(batch_observer):
        raise TypeError("batch_observer must be callable")
    if (normalization_observer is not None
            and not callable(normalization_observer)):
        raise TypeError("normalization_observer must be callable")
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("hierarchical refit supports only CPU and CUDA")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA refit requested but unavailable")

    statistics = fit_hierarchical_normalization(records)
    if normalization_observer is not None:
        normalization_observer({
            "phase": "refit",
            "scan_ids": tuple(record["scan_id"] for record in records),
            "normalization_sha256": statistics["sha256"],
        })
    model = _fit_hierarchical_model(
        records,
        statistics,
        hidden_dim=choice["hidden_dim"],
        weight_decay=choice["weight_decay"],
        false_positive_cost=choice["false_positive_cost"],
        device=resolved_device,
        batch_observer=batch_observer,
        observer_context={"phase": "refit"},
    )
    return model, statistics


def _canonical_selected_hierarchical_iou_sha256(
        records, selected_indices):
    _validate_hierarchical_records(records, require_contiguous=False)
    if (not isinstance(selected_indices, torch.Tensor)
            or selected_indices.device.type != "cpu"
            or selected_indices.dtype != torch.long
            or selected_indices.shape != (len(records),)):
        raise ValueError("selected hierarchical index layout is invalid")
    digest = hashlib.sha256()
    for record, selected_index in zip(records, selected_indices.tolist()):
        flat_valid = record["variant_valid"].reshape(-1)
        if (not 0 <= selected_index < QUERY_COUNT * VARIANT_COUNT
                or not bool(flat_valid[selected_index].item())):
            raise ValueError("selected hierarchical index is invalid")
        _digest_int(digest, record["dataset_index"])
        _digest_int(digest, selected_index)
        value = record["candidate_ious"].reshape(-1)[
            selected_index:selected_index + 1
        ]
        _digest_tensor(digest, value)
    return digest.hexdigest()


def build_hierarchical_cache_calibration_baseline(records):
    """Build immutable hierarchy cache counts before applying a proposal."""
    _validate_hierarchical_records(records, require_contiguous=False)
    candidate_ious = torch.stack([
        record["candidate_ious"].reshape(-1) for record in records
    ])
    variant_valid = torch.stack([
        record["variant_valid"].reshape(-1) for record in records
    ])
    baseline_indices = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    rows = torch.arange(len(records))
    baseline_ious = candidate_ious[rows, baseline_indices]
    oracle_ious = candidate_ious.masked_fill(
        ~variant_valid, -float("inf")
    ).max(dim=1).values
    return {
        "sample_count": len(records),
        "hits025": int(baseline_ious.gt(0.25).sum().item()),
        "hits050": int(baseline_ious.gt(0.50).sum().item()),
        "oracle_hits025": int(oracle_ious.gt(0.25).sum().item()),
        "oracle_hits050": int(oracle_ious.gt(0.50).sum().item()),
        "candidate_iou_sha256": (
            canonical_hierarchical_candidate_iou_sha256(records)
        ),
        "row_materialization_sha256": canonical_hierarchical_rows_sha256(
            records
        ),
        "baseline_selected_iou_sha256": (
            _canonical_selected_hierarchical_iou_sha256(
                records, baseline_indices
            )
        ),
    }


def evaluate_hierarchical_cache_policy(
        model, records, statistics, margin, device="cpu"):
    """Evaluate one frozen hierarchy on the cache without model selection."""
    _validate_hierarchical_records(records, require_contiguous=False)
    _validate_hierarchical_normalization(statistics)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("hierarchical evaluation model must be a module")
    if (isinstance(margin, bool)
            or not isinstance(margin, (float, int))
            or not math.isfinite(float(margin))
            or float(margin) <= 0.0):
        raise ValueError("hierarchical evaluation margin must be positive")
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("hierarchical evaluation supports only CPU and CUDA")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")

    proposals, gain = _predict_hierarchical_proposals(
        model, records, statistics, resolved_device
    )
    base_scores = torch.stack([
        record["baseline_scores"] for record in records
    ])
    variant_valid = torch.stack([
        record["variant_valid"] for record in records
    ])
    policy = apply_hierarchical_policy(
        base_scores,
        proposals,
        gain,
        variant_valid,
        float(margin),
    )
    baseline_indices = policy["baseline_indices"]
    selected_indices = policy["selected_indices"]
    switch_mask = policy["switch_mask"]
    candidate_ious = torch.stack([
        record["candidate_ious"].reshape(-1) for record in records
    ])
    flat_valid = variant_valid.reshape(len(records), -1)
    rows = torch.arange(len(records))
    baseline_ious = candidate_ious[rows, baseline_indices]
    selected_ious = candidate_ious[rows, selected_indices]
    oracle_ious = candidate_ious.masked_fill(
        ~flat_valid, -float("inf")
    ).max(dim=1).values
    baseline025 = baseline_ious.gt(0.25)
    baseline050 = baseline_ious.gt(0.50)
    selected025 = selected_ious.gt(0.25)
    selected050 = selected_ious.gt(0.50)
    fixes025 = selected025 & ~baseline025
    fixes050 = selected050 & ~baseline050
    scan_ids = [record["scan_id"] for record in records]
    bootstrap025 = hierarchical_scene_clustered_hit_delta_bootstrap(
        scan_ids, baseline025.long().tolist(), selected025.long().tolist()
    )
    bootstrap050 = hierarchical_scene_clustered_hit_delta_bootstrap(
        scan_ids, baseline050.long().tolist(), selected050.long().tolist()
    )
    baseline_queries = baseline_indices.div(
        VARIANT_COUNT, rounding_mode="floor"
    )
    selected_queries = selected_indices.div(
        VARIANT_COUNT, rounding_mode="floor"
    )
    query_changes = switch_mask & selected_queries.ne(baseline_queries)
    variant_changes = switch_mask & selected_queries.eq(baseline_queries)
    recoverable025 = ~baseline025 & oracle_ious.gt(0.25)
    recovered025 = fixes025 & recoverable025
    recoverable_count = int(recoverable025.sum().item())
    recovered_count = int(recovered025.sum().item())
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
    baseline = build_hierarchical_cache_calibration_baseline(records)
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
        "selected_query_changes": int(query_changes.sum().item()),
        "same_query_variant_changes": int(variant_changes.sum().item()),
        "wrong_query_recoveries025": int(
            (fixes025 & query_changes).sum().item()
        ),
        "wrong_variant_recoveries025": int(
            (fixes025 & variant_changes).sum().item()
        ),
        "wrong_query_recoveries050": int(
            (fixes050 & query_changes).sum().item()
        ),
        "wrong_variant_recoveries050": int(
            (fixes050 & variant_changes).sum().item()
        ),
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
        "selected_iou_sha256": _canonical_selected_hierarchical_iou_sha256(
            records, selected_indices
        ),
        "proposal_sha256": _canonical_oof_vector_sha256(records, proposals),
        "gain_sha256": _canonical_oof_vector_sha256(records, gain),
        "normalization_sha256": statistics["sha256"],
    }


def hierarchical_calibration_gate(metrics, baseline):
    """Apply the fixed cache gate shared with the sealed residual protocol."""
    return _residual_calibration_gate(metrics, baseline)


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def build_hierarchical_feature_names(geometry_feature_names):
    """Bind hierarchy coordinates to the exact frozen geometry schema."""
    if (not isinstance(geometry_feature_names, (list, tuple))
            or len(geometry_feature_names) != GEOMETRY_INPUT_DIM
            or len(set(geometry_feature_names)) != GEOMETRY_INPUT_DIM
            or any(not isinstance(name, str) or not name
                   for name in geometry_feature_names)):
        raise ValueError(
            "geometry feature names must contain 179 unique names"
        )
    geometry_names = list(geometry_feature_names)
    names = {
        "geometry_input": geometry_names,
        "query_features": geometry_names[:QUERY_FEATURE_DIM],
        "variant_features": geometry_names[
            QUERY_FEATURE_DIM:QUERY_FEATURE_DIM + VARIANT_FEATURE_DIM
        ],
        "query_aux_continuous": [
            "default_score", "default_rank", "parent_score", "parent_rank",
        ],
        "query_aux_binary": [
            "default_is_top1", "parent_is_top1",
        ],
        "variant_aux_continuous": [
            "geometry_score", "geometry_rank",
        ],
        "variant_aux_binary": [
            "geometry_is_top1", "frozen_baseline_is_top1",
        ],
    }
    expected_lengths = {
        "geometry_input": GEOMETRY_INPUT_DIM,
        "query_features": QUERY_FEATURE_DIM,
        "variant_features": VARIANT_FEATURE_DIM,
        "query_aux_continuous": QUERY_AUX_CONTINUOUS_DIM,
        "query_aux_binary": QUERY_AUX_BINARY_DIM,
        "variant_aux_continuous": VARIANT_AUX_CONTINUOUS_DIM,
        "variant_aux_binary": VARIANT_AUX_BINARY_DIM,
    }
    if (set(names) != set(expected_lengths)
            or any(len(names[name]) != length
                   for name, length in expected_lengths.items())):
        raise RuntimeError("hierarchical feature-name contract changed")
    return names


def _hierarchical_artifact_selection(selection):
    if not isinstance(selection, dict):
        raise TypeError("hierarchical selection must be a mapping")
    if not _HIERARCHICAL_ARTIFACT_SELECTION_FIELDS.issubset(selection):
        raise ValueError("hierarchical selection lacks artifact fields")
    return {
        key: copy.deepcopy(selection[key])
        for key in sorted(_HIERARCHICAL_ARTIFACT_SELECTION_FIELDS)
    }


def _expected_hierarchical_training_contract(selection):
    return {
        "seed": HIERARCHICAL_MODEL_SEED,
        "epochs": HIERARCHICAL_EPOCHS,
        "batch_size": HIERARCHICAL_BATCH_SIZE,
        "learning_rate": HIERARCHICAL_LEARNING_RATE,
        "gradient_clip_norm": HIERARCHICAL_GRAD_CLIP_NORM,
        "dropout": HIERARCHICAL_DROPOUT,
        "weight_decay": float(selection["weight_decay"]),
        "false_positive_cost": float(selection["false_positive_cost"]),
        "selected_margin": float(selection["margin"]),
        "selected_margin_percentile": float(
            selection["margin_percentile"]
        ),
        "hidden_dim_grid": list(HIERARCHICAL_HIDDEN_DIMS),
        "weight_decay_grid": list(HIERARCHICAL_WEIGHT_DECAYS),
        "false_positive_cost_grid": list(
            HIERARCHICAL_FALSE_POSITIVE_COSTS
        ),
        "margin_percentile_grid": list(HIERARCHICAL_MARGIN_PERCENTILES),
    }


def _validate_hierarchical_artifact_selection(selection, model_config):
    if (not isinstance(selection, dict)
            or set(selection) != _HIERARCHICAL_ARTIFACT_SELECTION_FIELDS
            or selection.get("eligible") is not True
            or selection.get("selected") != "hierarchical"
            or selection.get("hidden_dim") != model_config["hidden_dim"]
            or isinstance(selection.get("hidden_dim"), bool)
            or selection.get("hidden_dim") not in HIERARCHICAL_HIDDEN_DIMS
            or isinstance(selection.get("weight_decay"), bool)
            or selection.get("weight_decay") not in HIERARCHICAL_WEIGHT_DECAYS
            or isinstance(selection.get("false_positive_cost"), bool)
            or selection.get("false_positive_cost")
            not in HIERARCHICAL_FALSE_POSITIVE_COSTS
            or isinstance(selection.get("margin_percentile"), bool)
            or selection.get("margin_percentile")
            not in HIERARCHICAL_MARGIN_PERCENTILES
            or isinstance(selection.get("margin"), bool)
            or not isinstance(selection.get("margin"), (float, int))
            or not math.isfinite(float(selection["margin"]))
            or float(selection["margin"]) <= 0.0
            or type(selection.get("switches")) is not int
            or selection["switches"] < 0
            or type(selection.get("delta_hits025")) is not int
            or type(selection.get("delta_hits050")) is not int):
        raise ValueError("hierarchical artifact selection is invalid")


def build_hierarchical_artifact(
        model, selection, normalization, scene_folds,
        geometry_feature_names, input_sha256,
        row_materialization_sha256, candidate_iou_sha256, oof_record,
        calibration_record, calibration_baseline):
    """Build a cache-calibrated artifact that is not deployable yet."""
    if not isinstance(model, HierarchicalQueryVariantReranker):
        raise TypeError(
            "artifact model must be HierarchicalQueryVariantReranker"
        )
    if model.training or any(
            parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("artifact model must be frozen in evaluation mode")
    selection_record = _hierarchical_artifact_selection(selection)
    model_config = {
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout,
    }
    _validate_hierarchical_artifact_selection(
        selection_record, model_config
    )
    _validate_hierarchical_normalization(normalization)
    if normalization["sha256"] != _normalization_sha256(normalization):
        raise ValueError("artifact normalization binding changed")
    feature_names = build_hierarchical_feature_names(
        geometry_feature_names
    )
    if (not isinstance(input_sha256, dict)
            or set(input_sha256) != _HIERARCHICAL_INPUT_SHA_FIELDS):
        raise ValueError("hierarchical input SHA schema is invalid")
    if not isinstance(scene_folds, dict):
        raise TypeError("hierarchical scene_folds must be a mapping")
    if (not _is_sha256(row_materialization_sha256)
            or not _is_sha256(candidate_iou_sha256)):
        raise ValueError("hierarchical materialization SHA is invalid")
    if (not isinstance(oof_record, dict)
            or set(oof_record) != _HIERARCHICAL_OOF_RECORD_FIELDS):
        raise ValueError("hierarchical OOF record fields are invalid")
    gate = hierarchical_calibration_gate(
        calibration_record, calibration_baseline
    )
    if not gate.passed:
        raise ValueError("hierarchical cache calibration gate did not pass")
    artifact = {
        "schema": HIERARCHICAL_ARTIFACT_SCHEMA,
        "version": HIERARCHICAL_ARTIFACT_VERSION,
        "deployable": False,
        "validation_data_accessed": False,
        "input_sha256": copy.deepcopy(input_sha256),
        "feature_names": feature_names,
        "model_config": model_config,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "normalization": copy.deepcopy(normalization),
        "normalization_sha256": normalization["sha256"],
        "training_contract": _expected_hierarchical_training_contract(
            selection_record
        ),
        "selection": selection_record,
        "scene_folds": copy.deepcopy(scene_folds),
        "scene_fold_sha256": canonical_hierarchical_scene_fold_sha256(
            scene_folds
        ),
        "row_materialization_sha256": row_materialization_sha256,
        "candidate_iou_sha256": candidate_iou_sha256,
        "oof_record": copy.deepcopy(oof_record),
        "oof_record_sha256": canonical_json_sha256(oof_record),
        "calibration_record": copy.deepcopy(calibration_record),
        "calibration_baseline": copy.deepcopy(calibration_baseline),
    }
    validate_hierarchical_artifact(
        artifact,
        expected_geometry_feature_names=geometry_feature_names,
    )
    return artifact


def validate_hierarchical_artifact(
        artifact, expected_geometry_feature_names=None,
        expected_backbone_sha256=AUTHORITATIVE_BACKBONE_SHA256,
        expected_parent_sha256=AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        expected_geometry_sha256=AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
        expected_deployable=False):
    """Validate the staged hierarchical artifact and all internal bindings."""
    if (not isinstance(artifact, dict)
            or set(artifact) != _HIERARCHICAL_ARTIFACT_FIELDS):
        raise ValueError("hierarchical artifact fields differ from schema")
    if (type(expected_deployable) is not bool
            or artifact.get("schema") != HIERARCHICAL_ARTIFACT_SCHEMA
            or type(artifact.get("version")) is not int
            or artifact["version"] != HIERARCHICAL_ARTIFACT_VERSION
            or artifact.get("deployable") is not expected_deployable
            or artifact.get("validation_data_accessed") is not False):
        raise ValueError("hierarchical artifact top-level policy is invalid")
    expected_inputs = {
        "backbone": expected_backbone_sha256,
        "parent": expected_parent_sha256,
        "geometry": expected_geometry_sha256,
        "base_cache_content": AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
        "geometry_cache_content": AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256,
        "geometry_metadata": AUTHORITATIVE_GEOMETRY_METADATA_SHA256,
    }
    if (set(expected_inputs) != _HIERARCHICAL_INPUT_SHA_FIELDS
            or any(not _is_sha256(value)
                   for value in expected_inputs.values())
            or artifact.get("input_sha256") != expected_inputs):
        raise ValueError("hierarchical artifact provenance mismatch")
    feature_names = artifact.get("feature_names")
    if not isinstance(feature_names, dict):
        raise ValueError("hierarchical artifact feature names are invalid")
    live_geometry_names = (
        feature_names.get("geometry_input")
        if expected_geometry_feature_names is None
        else expected_geometry_feature_names
    )
    if feature_names != build_hierarchical_feature_names(
            live_geometry_names):
        raise ValueError("hierarchical artifact feature binding changed")
    model_config = artifact.get("model_config")
    if (not isinstance(model_config, dict)
            or set(model_config) != {"hidden_dim", "dropout"}
            or isinstance(model_config.get("hidden_dim"), bool)
            or model_config.get("hidden_dim") not in HIERARCHICAL_HIDDEN_DIMS
            or model_config.get("dropout") != HIERARCHICAL_DROPOUT):
        raise ValueError("hierarchical artifact model config is invalid")
    selection = artifact.get("selection")
    _validate_hierarchical_artifact_selection(selection, model_config)
    if artifact.get("training_contract") != \
            _expected_hierarchical_training_contract(selection):
        raise ValueError("hierarchical artifact training contract changed")
    normalization = artifact.get("normalization")
    _validate_hierarchical_normalization(normalization)
    if (artifact.get("normalization_sha256") != normalization["sha256"]
            or normalization["sha256"]
            != _normalization_sha256(normalization)):
        raise ValueError("hierarchical artifact normalization SHA changed")
    scene_folds = artifact.get("scene_folds")
    if (not isinstance(scene_folds, dict) or not scene_folds
            or any(not isinstance(scene, str) or not scene
                   or type(fold) is not int or not 0 <= fold < 5
                   for scene, fold in scene_folds.items())
            or set(scene_folds.values()) != set(range(5))
            or artifact.get("scene_fold_sha256")
            != canonical_hierarchical_scene_fold_sha256(scene_folds)):
        raise ValueError("hierarchical artifact scene-fold binding changed")
    if (not _is_sha256(artifact.get("row_materialization_sha256"))
            or not _is_sha256(artifact.get("candidate_iou_sha256"))):
        raise ValueError("hierarchical artifact materialization SHA is invalid")
    oof_record = artifact.get("oof_record")
    if (not isinstance(oof_record, dict)
            or set(oof_record) != _HIERARCHICAL_OOF_RECORD_FIELDS
            or type(oof_record.get("prediction_count")) is not int
            or oof_record["prediction_count"] <= 0
            or not _is_sha256(oof_record.get("proposal_sha256"))
            or not _is_sha256(oof_record.get("gain_sha256"))
            or type(oof_record.get("delta_hits025")) is not int
            or type(oof_record.get("delta_hits050")) is not int
            or oof_record["delta_hits025"] != selection["delta_hits025"]
            or oof_record["delta_hits050"] != selection["delta_hits050"]
            or artifact.get("oof_record_sha256")
            != canonical_json_sha256(oof_record)):
        raise ValueError("hierarchical artifact OOF record is invalid")
    gate = hierarchical_calibration_gate(
        artifact.get("calibration_record"),
        artifact.get("calibration_baseline"),
    )
    if not gate.passed:
        raise ValueError("hierarchical artifact calibration gate is not met")
    state = artifact.get("model_state_dict")
    if (not isinstance(state, dict)
            or any(not isinstance(value, torch.Tensor)
                   or value.device.type != "cpu"
                   for value in state.values())):
        raise ValueError("hierarchical artifact model state is invalid")
    model = HierarchicalQueryVariantReranker(**model_config)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "hierarchical artifact model state is incompatible: {}".format(
                error
            )
        )
    return copy.deepcopy(model_config)


def _serialize_hierarchical_artifact(
        artifact, expected_deployable=False):
    geometry_names = (
        artifact.get("feature_names", {}).get("geometry_input", ())
        if isinstance(artifact, dict) else ()
    )
    validate_hierarchical_artifact(
        artifact,
        expected_geometry_feature_names=geometry_names,
        expected_deployable=expected_deployable,
    )
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    return buffer.getvalue()


def _read_stable_hierarchical_bytes(path):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("hierarchical artifact path must be path-like")
    path = Path(path).expanduser().absolute()
    _require_directory_path_without_symlinks(path.parent)

    def identity(metadata):
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )

    try:
        entry = os.lstat(str(path))
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
            raise ValueError(
                "hierarchical artifact must be a regular non-symlink file"
            )
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            before = identity(os.fstat(descriptor))
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                snapshot = handle.read()
            after = identity(os.fstat(descriptor))
        finally:
            os.close(descriptor)
        live = identity(os.stat(str(path), follow_symlinks=False))
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(
            "could not read hierarchical artifact: {}".format(error)
        )
    if before != after or after != live:
        raise ValueError("hierarchical artifact changed during stable load")
    return path, snapshot, hashlib.sha256(snapshot).hexdigest()


def load_hierarchical_artifact(
        path, device="cpu", expected_geometry_feature_names=None,
        expected_artifact_sha256=None,
        parent_sha256=AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        geometry_sha256=AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
        expected_deployable=False):
    """Stable-load and freeze one cache-calibrated hierarchy artifact."""
    resolved, snapshot, snapshot_sha256 = _read_stable_hierarchical_bytes(
        path
    )
    if (expected_artifact_sha256 is not None
            and (not _is_sha256(expected_artifact_sha256)
                 or snapshot_sha256 != expected_artifact_sha256)):
        raise ValueError("hierarchical artifact SHA-256 mismatch")
    try:
        artifact = torch.load(io.BytesIO(snapshot), map_location="cpu")
    except Exception as error:
        raise ValueError(
            "could not deserialize hierarchical artifact: {}".format(error)
        )
    config = validate_hierarchical_artifact(
        artifact,
        expected_geometry_feature_names=expected_geometry_feature_names,
        expected_parent_sha256=parent_sha256,
        expected_geometry_sha256=geometry_sha256,
        expected_deployable=expected_deployable,
    )
    model = HierarchicalQueryVariantReranker(**config)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("hierarchical artifact supports only CPU and CUDA")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA hierarchical artifact load requested")
    model.to(resolved_device).eval().requires_grad_(False)
    model._artifact_path = str(resolved)
    model._artifact_sha256 = snapshot_sha256
    return model, artifact


def save_hierarchical_artifact(path, artifact):
    """Publish one fresh staged artifact without overwriting any byte."""
    output = Path(path).expanduser().absolute()
    parent = _require_directory_path_without_symlinks(output.parent)
    reservation = _directory_reservation(parent)
    directory_fd = _open_reserved_directory(parent, reservation)
    try:
        payload = _serialize_hierarchical_artifact(artifact)
        _exclusive_write_bytes(
            directory_fd, reservation, output.name, payload, mode=0o444
        )
    finally:
        os.close(directory_fd)
    _model, reloaded = load_hierarchical_artifact(
        output,
        device="cpu",
        expected_geometry_feature_names=artifact["feature_names"][
            "geometry_input"
        ],
    )
    if not _payloads_equal(artifact, reloaded):
        raise RuntimeError("strict hierarchical artifact reload changed payload")
    return artifact


def reserve_hierarchical_output(output_dir):
    """Exclusively reserve a fresh hierarchical experiment directory."""
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


def _read_reserved_output_bytes(
        directory_fd, reservation, name, expected_identity):
    _verify_reserved_directory(directory_fd, reservation)
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    linked = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    live_identity = {
        "device": int(linked.st_dev),
        "inode": int(linked.st_ino),
        "size": int(linked.st_size),
        "mode": stat.S_IMODE(linked.st_mode),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if (not stat.S_ISREG(before.st_mode)
            or (int(before.st_dev), int(before.st_ino), int(before.st_size),
                int(before.st_mtime_ns), int(before.st_ctime_ns))
            != (int(after.st_dev), int(after.st_ino), int(after.st_size),
                int(after.st_mtime_ns), int(after.st_ctime_ns))
            or live_identity != expected_identity):
        raise RuntimeError("reserved output changed during strict reload")
    return payload


def publish_hierarchical_experiment(
        output_dir, artifact, result_context, protected_paths,
        protected_before=None, reservation=None):
    """Publish fresh hierarchy evidence with the completion receipt last."""
    output = Path(output_dir).expanduser().absolute()
    if reservation is None:
        reservation = reserve_hierarchical_output(output)
    directory_fd = _open_reserved_directory(output, reservation)
    artifact_path = None
    artifact_sha256 = None
    receipt = None
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
            calibration_passed = hierarchical_calibration_gate(
                calibration.get("record"), calibration.get("baseline")
            ).passed
        if calibration_passed != (artifact is not None):
            raise ValueError(
                "artifact presence must match the hierarchical cache gate"
            )
        preflight_binding = (
            None if artifact is None else {
                "name": HIERARCHICAL_ARTIFACT_NAME,
                "sha256": "0" * 64,
            }
        )
        build_hierarchical_result_receipt(
            result_context,
            artifact_binding=preflight_binding,
            protected_before=protected_before,
            protected_after=protected_before,
        )

        if artifact is not None:
            artifact_path = output / HIERARCHICAL_ARTIFACT_NAME
            artifact_payload = _serialize_hierarchical_artifact(artifact)
            artifact_identity = _exclusive_write_bytes(
                directory_fd,
                reservation,
                artifact_path.name,
                artifact_payload,
                mode=0o444,
            )
            artifact_sha256 = artifact_identity["sha256"]
            _model, reloaded_artifact = load_hierarchical_artifact(
                artifact_path,
                device="cpu",
                expected_geometry_feature_names=artifact[
                    "feature_names"
                ]["geometry_input"],
                expected_artifact_sha256=artifact_sha256,
            )
            if not _payloads_equal(artifact, reloaded_artifact):
                raise RuntimeError(
                    "strict hierarchical artifact reload changed payload"
                )

        protected_after = capture_immutable_artifact_identities(
            protected_paths
        )
        if protected_after != protected_before:
            raise RuntimeError("protected artifacts changed during experiment")
        artifact_binding = (
            None if artifact_path is None else {
                "name": artifact_path.name,
                "sha256": artifact_sha256,
            }
        )
        receipt = build_hierarchical_result_receipt(
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
        reloaded_payload = _read_reserved_output_bytes(
            directory_fd, reservation, pending_name, pending_identity
        )
        reloaded_receipt = json.loads(reloaded_payload.decode("ascii"))
        validate_hierarchical_result_receipt(reloaded_receipt)
        if reloaded_receipt != receipt:
            raise RuntimeError("strict receipt reload changed payload")

        final_name = "result-receipt.json"
        linked_final = False
        try:
            os.link(
                pending_name,
                final_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            linked_final = True
            final_identity = os.stat(
                final_name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (int(final_identity.st_dev) != pending_identity["device"]
                    or int(final_identity.st_ino)
                    != pending_identity["inode"]
                    or int(final_identity.st_size)
                    != pending_identity["size"]
                    or stat.S_IMODE(final_identity.st_mode) != 0o444):
                raise RuntimeError("completion receipt identity changed")
            os.fsync(directory_fd)
        except Exception:
            if linked_final:
                try:
                    os.unlink(final_name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                except OSError:
                    pass
            raise
        try:
            os.unlink(pending_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except OSError:
            pass
        _verify_reserved_directory(directory_fd, reservation)
    finally:
        os.close(directory_fd)

    final_receipt = output / "result-receipt.json"
    if stat.S_IMODE(final_receipt.stat().st_mode) != 0o444:
        raise RuntimeError("published hierarchical receipt is not read-only")
    if capture_immutable_artifact_identities(
            protected_paths) != protected_before:
        raise RuntimeError("protected artifacts changed after publication")
    return {
        "output_dir": output,
        "receipt_path": final_receipt,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "receipt": receipt,
    }


def _hierarchical_result_protocol():
    return {
        "scope": "train-only-hierarchical-query-variant-cross-fit",
        "selection_uses_validation": False,
        "inference_uses_ground_truth": False,
        "seed": HIERARCHICAL_MODEL_SEED,
        "fold_count": 5,
        "epochs": HIERARCHICAL_EPOCHS,
        "batch_size": HIERARCHICAL_BATCH_SIZE,
        "materialization_batch_size": HIERARCHICAL_MATERIALIZATION_BATCH_SIZE,
        "learning_rate": HIERARCHICAL_LEARNING_RATE,
        "gradient_clip_norm": HIERARCHICAL_GRAD_CLIP_NORM,
        "thresholds": [0.25, 0.50],
        "head_weights": [2.0, 1.0],
        "hidden_dim_grid": list(HIERARCHICAL_HIDDEN_DIMS),
        "weight_decay_grid": list(HIERARCHICAL_WEIGHT_DECAYS),
        "false_positive_cost_grid": list(
            HIERARCHICAL_FALSE_POSITIVE_COSTS
        ),
        "margin_percentile_grid": list(HIERARCHICAL_MARGIN_PERCENTILES),
        "margin_rule": "nearest-rank-positive-oof-gain",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 0,
    }


def hierarchical_calibration_gate_receipt(gate):
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


def _validate_hierarchical_protected_snapshot(snapshot, name):
    if (not isinstance(snapshot, dict)
            or set(snapshot) != {"backbone", "parent", "geometry"}):
        raise ValueError("{} protected snapshot is invalid".format(name))
    for artifact_name, identity in snapshot.items():
        if (not isinstance(identity, dict)
                or set(identity) != _HIERARCHICAL_PROTECTED_IDENTITY_FIELDS
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


def _validate_hierarchical_label_summary(
        summary, expected_query_count, expected_variant_count):
    if not isinstance(summary, dict) or set(summary) != {"query", "variant"}:
        raise ValueError("hierarchical training label levels are invalid")
    for level, expected_count in (
            ("query", expected_query_count),
            ("variant", expected_variant_count)):
        thresholds = summary[level]
        if (not isinstance(thresholds, dict)
                or set(thresholds) != {"0.25", "0.50"}):
            raise ValueError("hierarchical training label thresholds changed")
        positives = {}
        for threshold in ("0.25", "0.50"):
            counts = thresholds[threshold]
            if (not isinstance(counts, dict)
                    or set(counts) != {"positive", "negative", "total"}
                    or any(type(value) is not int or value < 0
                           for value in counts.values())
                    or counts["positive"] + counts["negative"]
                    != counts["total"]
                    or counts["total"] != expected_count):
                raise ValueError(
                    "hierarchical training label counts do not reconcile"
                )
            positives[threshold] = counts["positive"]
        if positives["0.50"] > positives["0.25"]:
            raise ValueError("0.50 positives must be a subset count of 0.25")


def _validate_hierarchical_gain_summary(summary, expected_count):
    if not isinstance(summary, dict) or set(summary) != {"all", "positive"}:
        raise ValueError("hierarchical gain summary groups are invalid")
    counts = {}
    for name in ("all", "positive"):
        record = summary[name]
        if (not isinstance(record, dict)
                or set(record) != {"count", "statistics"}
                or type(record["count"]) is not int
                or record["count"] < 0):
            raise ValueError("hierarchical gain summary count is invalid")
        counts[name] = record["count"]
        statistics = record["statistics"]
        if record["count"] == 0:
            if statistics is not None:
                raise ValueError("empty gain summary has statistics")
            continue
        if (not isinstance(statistics, dict)
                or set(statistics) != {
                    "minimum", "maximum", "mean",
                    "population_standard_deviation",
                    "nearest_rank_quantiles",
                }):
            raise ValueError("hierarchical gain statistics fields changed")
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
            raise ValueError("hierarchical gain statistics are invalid")
        quantiles = statistics["nearest_rank_quantiles"]
        if (not isinstance(quantiles, list)
                or [item.get("quantile") for item in quantiles]
                != list(HIERARCHICAL_GAIN_QUANTILES)
                or any(not isinstance(item, dict)
                       or set(item) != {"quantile", "value"}
                       or isinstance(item["value"], bool)
                       or not isinstance(item["value"], (int, float))
                       or not math.isfinite(float(item["value"]))
                       for item in quantiles)
                or any(first["value"] > second["value"]
                       for first, second in zip(quantiles, quantiles[1:]))):
            raise ValueError("hierarchical gain quantiles are invalid")
        if name == "positive" and statistics["minimum"] <= 0.0:
            raise ValueError("positive gain summary contains nonpositive values")
    if counts["all"] != expected_count or counts["positive"] > counts["all"]:
        raise ValueError("hierarchical gain counts do not reconcile")


def _validate_hierarchical_candidate_diagnostic(
        record, baseline, scene_count):
    if (not isinstance(record, dict)
            or set(record) != _HIERARCHICAL_DIAGNOSTIC_FIELDS):
        raise ValueError("hierarchical OOF diagnostic fields changed")
    if (isinstance(record["hidden_dim"], bool)
            or record["hidden_dim"] not in HIERARCHICAL_HIDDEN_DIMS
            or isinstance(record["weight_decay"], bool)
            or record["weight_decay"] not in HIERARCHICAL_WEIGHT_DECAYS
            or isinstance(record["false_positive_cost"], bool)
            or record["false_positive_cost"]
            not in HIERARCHICAL_FALSE_POSITIVE_COSTS
            or type(record["no_switch"]) is not bool
            or record["sample_count"] != baseline["sample_count"]
            or type(record["switches"]) is not int
            or type(record["abstentions"]) is not int
            or not 0 <= record["switches"] <= record["sample_count"]
            or record["switches"] + record["abstentions"]
            != record["sample_count"]
            or isinstance(record["switch_rate"], bool)
            or not isinstance(record["switch_rate"], (int, float))
            or not math.isclose(
                float(record["switch_rate"]),
                record["switches"] / float(record["sample_count"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )):
        raise ValueError("hierarchical diagnostic identity is invalid")
    if record["no_switch"]:
        if (record["margin_percentile"] is not None
                or record["margin"] is not None):
            raise ValueError("no-switch margin must be null")
    elif (isinstance(record["margin_percentile"], bool)
          or record["margin_percentile"]
          not in HIERARCHICAL_MARGIN_PERCENTILES
          or isinstance(record["margin"], bool)
          or not isinstance(record["margin"], (int, float))
          or not math.isfinite(float(record["margin"]))
          or float(record["margin"]) <= 0.0):
        raise ValueError("hierarchical diagnostic margin is invalid")

    for container_name in ("baseline", "proposed"):
        container = record[container_name]
        if (not isinstance(container, dict)
                or set(container) != {"0.25", "0.50"}):
            raise ValueError("hierarchical diagnostic thresholds changed")
        for threshold in ("0.25", "0.50"):
            hits = container[threshold]
            if (not isinstance(hits, dict) or set(hits) != {"hits"}
                    or type(hits["hits"]) is not int
                    or not 0 <= hits["hits"] <= record["sample_count"]):
                raise ValueError("hierarchical diagnostic hits are invalid")
        if container["0.50"]["hits"] > container["0.25"]["hits"]:
            raise ValueError("hierarchical 0.50 hits exceed 0.25 hits")
    if (record["baseline"]["0.25"]["hits"] != baseline["hits025"]
            or record["baseline"]["0.50"]["hits"]
            != baseline["hits050"]):
        raise ValueError("hierarchical diagnostic baseline changed")

    effects = record["effects"]
    if not isinstance(effects, dict) or set(effects) != {"0.25", "0.50"}:
        raise ValueError("hierarchical diagnostic effects changed")
    fixes = {}
    for threshold, delta_name in (
            ("0.25", "delta_hits025"), ("0.50", "delta_hits050")):
        threshold_effects = effects[threshold]
        if (not isinstance(threshold_effects, dict)
                or set(threshold_effects) != {
                    "fixes", "breaks", "neutral_switches",
                    "kept_correct", "kept_wrong",
                }
                or any(type(value) is not int or value < 0
                       for value in threshold_effects.values())
                or sum(threshold_effects.values()) != record["sample_count"]
                or sum(threshold_effects[name] for name in (
                    "fixes", "breaks", "neutral_switches"
                )) != record["switches"]):
            raise ValueError("hierarchical diagnostic effects do not reconcile")
        observed_delta = (
            record["proposed"][threshold]["hits"]
            - record["baseline"][threshold]["hits"]
        )
        if (observed_delta != threshold_effects["fixes"]
                - threshold_effects["breaks"]
                or observed_delta != record[delta_name]):
            raise ValueError("hierarchical diagnostic delta changed")
        fixes[threshold] = threshold_effects["fixes"]

    fold_deltas = record["fold_deltas"]
    if (not isinstance(fold_deltas, dict)
            or set(fold_deltas) != {str(fold) for fold in range(5)}
            or any(not isinstance(value, dict)
                   or set(value) != {"hits025", "hits050"}
                   or any(type(delta) is not int for delta in value.values())
                   for value in fold_deltas.values())
            or sum(value["hits025"] for value in fold_deltas.values())
            != record["delta_hits025"]
            or sum(value["hits050"] for value in fold_deltas.values())
            != record["delta_hits050"]):
        raise ValueError("hierarchical diagnostic fold deltas changed")
    bootstraps = {}
    for threshold, field, delta_name in (
            ("0.25", "bootstrap025", "delta_hits025"),
            ("0.50", "bootstrap050", "delta_hits050")):
        bootstrap = record[field]
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
                or bootstrap["delta_hits"] != record[delta_name]):
            raise ValueError("hierarchical diagnostic bootstrap changed")
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
        "bootstrap025_lower_bound_nonnegative": (
            bootstraps["0.25"]["lower_bound_95"] >= 0
        ),
        "bootstrap050_lower_bound_nonnegative": (
            bootstraps["0.50"]["lower_bound_95"] >= 0
        ),
    }
    if (not isinstance(predicates, dict)
            or set(predicates) != _HIERARCHICAL_ELIGIBILITY_PREDICATE_FIELDS
            or predicates != expected_predicates
            or record["failed_predicates"] != sorted(
                name for name, passed in predicates.items() if not passed
            )
            or type(record["eligible"]) is not bool
            or record["eligible"] != all(predicates.values())
            or record["selected"]
            != ("hierarchical" if record["eligible"] else "baseline")):
        raise ValueError("hierarchical eligibility predicates changed")
    transition = record["transition_diagnostics"]
    if (not isinstance(transition, dict)
            or set(transition) != _HIERARCHICAL_TRANSITION_FIELDS
            or any(type(value) is not int or value < 0
                   for value in transition.values())
            or transition["selected_query_changes"]
            + transition["same_query_variant_changes"]
            != record["switches"]
            or transition["wrong_query_recoveries025"]
            + transition["wrong_variant_recoveries025"] != fixes["0.25"]
            or transition["wrong_query_recoveries050"]
            + transition["wrong_variant_recoveries050"] != fixes["0.50"]):
        raise ValueError("hierarchical transition diagnostics changed")


def _validate_hierarchical_configuration(
        configuration, expected_config, configuration_index,
        baseline, scene_count):
    if (not isinstance(configuration, dict)
            or set(configuration) != _HIERARCHICAL_CONFIGURATION_FIELDS
            or configuration["configuration_index"] != configuration_index
            or (configuration["hidden_dim"], configuration["weight_decay"],
                configuration["false_positive_cost"]) != expected_config
            or configuration["prediction_count"] != baseline["sample_count"]
            or not _is_sha256(configuration["oof_proposal_sha256"])
            or not _is_sha256(configuration["oof_gain_sha256"])):
        raise ValueError("hierarchical OOF configuration is invalid")
    _validate_hierarchical_gain_summary(
        configuration["gain_summary"], baseline["sample_count"]
    )
    folds = configuration["folds"]
    if (not isinstance(folds, list) or len(folds) != 5
            or {fold.get("fold") for fold in folds} != set(range(5))):
        raise ValueError("hierarchical configuration folds changed")
    held_row_total = 0
    held_scene_total = 0
    fit_row_total = 0
    for fold in folds:
        if (not isinstance(fold, dict)
                or set(fold) != _HIERARCHICAL_FOLD_FIELDS
                or any(type(fold[name]) is not int or fold[name] <= 0
                       for name in (
                           "fit_scene_count", "fit_row_count",
                           "fit_query_count", "fit_variant_count",
                           "held_scene_count", "held_row_count",
                       ))
                or fold["fit_scene_count"] + fold["held_scene_count"]
                != scene_count
                or fold["fit_row_count"] + fold["held_row_count"]
                != baseline["sample_count"]
                or fold["fit_query_count"] < fold["fit_row_count"]
                or fold["fit_query_count"]
                > fold["fit_row_count"] * QUERY_COUNT
                or fold["fit_variant_count"] < fold["fit_query_count"]
                or fold["fit_variant_count"]
                > fold["fit_row_count"] * QUERY_COUNT * VARIANT_COUNT
                or not _is_sha256(fold["normalization_sha256"])):
            raise ValueError("hierarchical configuration fold is invalid")
        expected_counts = {
            "query_features": fold["fit_query_count"],
            "variant_features": fold["fit_variant_count"],
            "query_aux_continuous": fold["fit_query_count"],
            "variant_aux_continuous": fold["fit_variant_count"],
        }
        if fold["normalization_counts"] != expected_counts:
            raise ValueError("hierarchical fold normalization counts changed")
        _validate_hierarchical_label_summary(
            fold["training_labels"],
            fold["fit_query_count"],
            fold["fit_variant_count"],
        )
        held_row_total += fold["held_row_count"]
        held_scene_total += fold["held_scene_count"]
        fit_row_total += fold["fit_row_count"]
    if (held_row_total != baseline["sample_count"]
            or held_scene_total != scene_count
            or fit_row_total != 4 * baseline["sample_count"]):
        raise ValueError("hierarchical held folds do not partition fit data")
    return tuple(
        (fold["fold"], fold["fit_scene_count"], fold["fit_row_count"],
         fold["held_scene_count"], fold["held_row_count"])
        for fold in sorted(folds, key=lambda value: value["fold"])
    )


def validate_hierarchical_result_receipt(receipt):
    """Strictly validate one complete train-only hierarchical receipt."""
    if (not isinstance(receipt, dict)
            or set(receipt) != _HIERARCHICAL_RESULT_RECEIPT_FIELDS):
        raise ValueError("hierarchical result receipt fields changed")
    if (receipt.get("schema") != HIERARCHICAL_RESULT_RECEIPT_SCHEMA
            or type(receipt.get("version")) is not int
            or receipt["version"] != HIERARCHICAL_RESULT_RECEIPT_VERSION
            or receipt.get("deployable") is not False
            or receipt.get("report_only") is not False
            or receipt.get("eligible_for_model_selection") is not True
            or receipt.get("validation_data_accessed") is not False
            or receipt.get("protocol") != _hierarchical_result_protocol()):
        raise ValueError("hierarchical result receipt policy is invalid")
    inputs = receipt["input_sha256"]
    expected_inputs = {
        "backbone": AUTHORITATIVE_BACKBONE_SHA256,
        "parent": AUTHORITATIVE_PARENT_ARTIFACT_SHA256,
        "geometry": AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256,
        "base_cache_content": AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256,
        "geometry_cache_content": AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256,
        "geometry_metadata": AUTHORITATIVE_GEOMETRY_METADATA_SHA256,
    }
    if inputs != expected_inputs:
        raise ValueError("hierarchical receipt input provenance changed")
    if receipt["split"] != AUTHORITATIVE_SPLIT_SEED0:
        raise ValueError("hierarchical receipt split is not authoritative")
    for name in (
            "fit_joined_identity_sha256", "fit_materialization_sha256",
            "fit_deployable_sha256", "fit_candidate_iou_sha256",
            "fit_binding_sha256"):
        if not _is_sha256(receipt[name]):
            raise ValueError("hierarchical receipt {} is invalid".format(name))
    fit_normalization = receipt["fit_normalization_sha256"]
    if fit_normalization is not None and not _is_sha256(fit_normalization):
        raise ValueError("hierarchical fit normalization SHA is invalid")
    expected_fit_binding = canonical_json_sha256({
        "fit_joined_identity_sha256": receipt[
            "fit_joined_identity_sha256"
        ],
        "fit_materialization_sha256": receipt[
            "fit_materialization_sha256"
        ],
        "fit_deployable_sha256": receipt["fit_deployable_sha256"],
        "fit_candidate_iou_sha256": receipt["fit_candidate_iou_sha256"],
        "fit_normalization_sha256": fit_normalization,
        "split_mapping_sha256": receipt["split"]["mapping_sha256"],
    })
    if receipt["fit_binding_sha256"] != expected_fit_binding:
        raise ValueError("hierarchical fit evidence binding changed")
    before = receipt["protected_before"]
    after = receipt["protected_after"]
    _validate_hierarchical_protected_snapshot(before, "before")
    _validate_hierarchical_protected_snapshot(after, "after")
    if before != after:
        raise ValueError("protected artifacts changed across receipt")
    for name in ("backbone", "parent", "geometry"):
        if before[name]["sha256"] != inputs[name]:
            raise ValueError("hierarchical protected provenance changed")

    oof = receipt["oof"]
    if not isinstance(oof, dict) or set(oof) != _HIERARCHICAL_RESULT_OOF_FIELDS:
        raise ValueError("hierarchical receipt OOF fields changed")
    baseline = oof["baseline"]
    if (not isinstance(baseline, dict)
            or set(baseline) != _HIERARCHICAL_OOF_BASELINE_FIELDS
            or baseline["sample_count"] != receipt["split"]["fit_sample_count"]
            or baseline["row_materialization_sha256"]
            != receipt["fit_materialization_sha256"]
            or baseline["candidate_iou_sha256"]
            != receipt["fit_candidate_iou_sha256"]
            or any(type(baseline[name]) is not int for name in (
                "sample_count", "hits025", "hits050",
                "oracle_hits025", "oracle_hits050",
            ))
            or not 0 <= baseline["hits050"] <= baseline["hits025"]
            <= baseline["sample_count"]
            or not baseline["hits025"] <= baseline["oracle_hits025"]
            <= baseline["sample_count"]
            or not baseline["hits050"] <= baseline["oracle_hits050"]
            <= baseline["sample_count"]
            or any(not _is_sha256(baseline[name]) for name in (
                "candidate_iou_sha256", "row_materialization_sha256",
                "baseline_selected_iou_sha256",
            ))):
        raise ValueError("hierarchical OOF baseline is invalid")
    scene_folds = oof["scene_folds"]
    if (not isinstance(scene_folds, dict)
            or len(scene_folds) != receipt["split"]["fit_scene_count"]
            or any(not isinstance(scene, str) or not scene
                   or type(fold) is not int or fold not in range(5)
                   for scene, fold in scene_folds.items())
            or set(scene_folds.values()) != set(range(5))
            or oof["scene_fold_sha256"]
            != canonical_hierarchical_scene_fold_sha256(scene_folds)):
        raise ValueError("hierarchical OOF scene folds changed")
    expected_grid = list(itertools.product(
        HIERARCHICAL_HIDDEN_DIMS,
        HIERARCHICAL_WEIGHT_DECAYS,
        HIERARCHICAL_FALSE_POSITIVE_COSTS,
    ))
    configurations = oof["configurations"]
    if (not isinstance(configurations, list)
            or oof["configuration_count"] != len(expected_grid)
            or len(configurations) != len(expected_grid)
            or oof["configuration_summaries_sha256"]
            != canonical_json_sha256(configurations)):
        raise ValueError("hierarchical OOF configuration table changed")
    fold_signature = None
    for index, (configuration, expected) in enumerate(
            zip(configurations, expected_grid)):
        live_signature = _validate_hierarchical_configuration(
            configuration,
            expected,
            index,
            baseline,
            receipt["split"]["fit_scene_count"],
        )
        if fold_signature is None:
            fold_signature = live_signature
        elif live_signature != fold_signature:
            raise ValueError("hierarchical fold partitions differ by config")

    choice = oof["choice"]
    if (not isinstance(choice, dict)
            or oof["choice_sha256"] != canonical_json_sha256(choice)):
        raise ValueError("hierarchical OOF choice binding changed")
    diagnostics = choice.get("candidate_diagnostics")
    if (not isinstance(diagnostics, list) or not diagnostics
            or choice.get("candidate_count") != len(diagnostics)
            or oof["policy_candidate_count"] != len(diagnostics)
            or not len(expected_grid) <= len(diagnostics)
            <= len(expected_grid) * (1 + len(HIERARCHICAL_MARGIN_PERCENTILES))):
        raise ValueError("hierarchical policy candidate count changed")
    identities = []
    sentinels = set()
    for diagnostic in diagnostics:
        _validate_hierarchical_candidate_diagnostic(
            diagnostic, baseline, receipt["split"]["fit_scene_count"]
        )
        identity = (
            diagnostic["hidden_dim"], diagnostic["weight_decay"],
            diagnostic["false_positive_cost"],
            diagnostic["margin_percentile"],
        )
        identities.append(identity)
        if diagnostic["no_switch"]:
            sentinels.add(identity[:3])
    if (len(set(identities)) != len(identities)
            or sentinels != set(expected_grid)):
        raise ValueError("hierarchical policy candidate identities changed")
    eligible_records = [
        diagnostic for diagnostic in diagnostics if diagnostic["eligible"]
    ]
    if choice.get("eligible_candidate_count") != len(eligible_records):
        raise ValueError("hierarchical eligible candidate count changed")
    if choice.get("eligible") is False:
        if (set(choice) != {
                "candidate_count", "eligible_candidate_count",
                "candidate_diagnostics", "eligible", "reason", "selected",
                }
                or choice.get("reason") != "no-eligible-configuration"
                or choice.get("selected") != "baseline"
                or eligible_records):
            raise ValueError("hierarchical rejected OOF choice is invalid")
    elif choice.get("eligible") is True:
        if not eligible_records:
            raise ValueError("hierarchical eligible choice has no candidate")
        winner = max(
            eligible_records,
            key=lambda record: (
                2 * record["delta_hits025"] + record["delta_hits050"],
                record["margin"],
                -record["switches"],
                -record["hidden_dim"],
                record["weight_decay"],
                record["false_positive_cost"],
            ),
        )
        if (set(choice) != _HIERARCHICAL_DIAGNOSTIC_FIELDS | {
                "candidate_count", "eligible_candidate_count",
                "candidate_diagnostics",
                }
                or any(choice[name] != winner[name]
                       for name in _HIERARCHICAL_DIAGNOSTIC_FIELDS)):
            raise ValueError("hierarchical eligible OOF winner changed")
    else:
        raise ValueError("hierarchical choice eligibility is invalid")

    calibration = receipt["calibration"]
    artifact = receipt["artifact"]
    if choice["eligible"] is False:
        if (fit_normalization is not None
                or calibration != {
                    "status": "not_run",
                    "reason": "oof_selection_rejected",
                }
                or artifact is not None
                or receipt["selected"] != "baseline"):
            raise ValueError("rejected hierarchical OOF advanced protocol")
    else:
        if (fit_normalization is None
                or not isinstance(calibration, dict)
                or set(calibration) != {
                    "status", "baseline", "record", "gate",
                }
                or calibration["status"] != "run"):
            raise ValueError("eligible hierarchical OOF lacks calibration")
        gate = hierarchical_calibration_gate(
            calibration["record"], calibration["baseline"]
        )
        if calibration["gate"] != hierarchical_calibration_gate_receipt(gate):
            raise ValueError("hierarchical calibration gate changed")
        if gate.passed:
            if (not isinstance(artifact, dict)
                    or set(artifact) != {"name", "sha256"}
                    or artifact["name"] != HIERARCHICAL_ARTIFACT_NAME
                    or not _is_sha256(artifact["sha256"])
                    or receipt["selected"] != "staged_hierarchical"):
                raise ValueError("hierarchical staged artifact binding changed")
        elif artifact is not None or receipt["selected"] != "baseline":
            raise ValueError("failed cache gate selected hierarchical model")
    try:
        canonical_json_sha256(receipt)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "hierarchical receipt is not canonical JSON: {}".format(error)
        )
    return copy.deepcopy(receipt)


def build_hierarchical_result_receipt(
        result_context, artifact_binding, protected_before, protected_after):
    """Build and self-validate one truthful hierarchical result receipt."""
    if (not isinstance(result_context, dict)
            or set(result_context) != _HIERARCHICAL_RESULT_CONTEXT_FIELDS):
        raise ValueError("hierarchical result context fields changed")
    oof_context = result_context["oof"]
    if (not isinstance(oof_context, dict)
            or set(oof_context)
            != _HIERARCHICAL_RESULT_OOF_CONTEXT_FIELDS):
        raise ValueError("hierarchical result OOF context fields changed")
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
        "fit_deployable_sha256": result_context["fit_deployable_sha256"],
        "fit_candidate_iou_sha256": result_context[
            "fit_candidate_iou_sha256"
        ],
        "fit_normalization_sha256": result_context[
            "fit_normalization_sha256"
        ],
        "split_mapping_sha256": result_context["split"]["mapping_sha256"],
    })
    receipt = {
        "schema": HIERARCHICAL_RESULT_RECEIPT_SCHEMA,
        "version": HIERARCHICAL_RESULT_RECEIPT_VERSION,
        "selected": (
            "staged_hierarchical"
            if artifact_binding is not None else "baseline"
        ),
        "deployable": False,
        "report_only": False,
        "eligible_for_model_selection": True,
        "validation_data_accessed": False,
        "protocol": _hierarchical_result_protocol(),
        "input_sha256": copy.deepcopy(result_context["input_sha256"]),
        "split": copy.deepcopy(result_context["split"]),
        "fit_joined_identity_sha256": result_context[
            "fit_joined_identity_sha256"
        ],
        "fit_materialization_sha256": result_context[
            "fit_materialization_sha256"
        ],
        "fit_deployable_sha256": result_context["fit_deployable_sha256"],
        "fit_candidate_iou_sha256": result_context[
            "fit_candidate_iou_sha256"
        ],
        "fit_normalization_sha256": result_context[
            "fit_normalization_sha256"
        ],
        "fit_binding_sha256": fit_binding_sha256,
        "oof": oof,
        "calibration": copy.deepcopy(result_context["calibration"]),
        "artifact": copy.deepcopy(artifact_binding),
        "protected_before": copy.deepcopy(protected_before),
        "protected_after": copy.deepcopy(protected_after),
    }
    return validate_hierarchical_result_receipt(receipt)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fit the fixed train-only ScanRefer hierarchical reranker"
        )
    )
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True, choices=("cuda:0",))
    return parser.parse_args(argv)


def _selected_hierarchical_configuration(configurations, choice):
    selected = [
        configuration for configuration in configurations
        if all(configuration.get(name) == choice.get(name) for name in (
            "hidden_dim", "weight_decay", "false_positive_cost"
        ))
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "OOF choice does not identify one hierarchical configuration"
        )
    return selected[0]


def run_hierarchical_training(
        base_cache, geometry_cache, parent_artifact, geometry_artifact,
        output_dir, device="cuda:0"):
    """Run the fixed OOF/refit/cache-gate protocol and publish once."""
    if str(device) != "cuda:0":
        raise ValueError("authoritative hierarchical training requires cuda:0")
    if not isinstance(output_dir, (str, os.PathLike)):
        raise TypeError("output directory must be path-like")
    output_path = Path(output_dir).expanduser().absolute()
    reservation = reserve_hierarchical_output(output_path)
    validated_output = _validate_train_only_path(output_path, "output")
    if validated_output != output_path:
        raise RuntimeError("reserved hierarchical output path changed")

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
    fit_records = materialize_hierarchical_rows(
        joined_split["fit_rows"],
        loaded["parent"],
        loaded["geometry_model"],
        loaded["geometry_artifact"],
        batch_size=HIERARCHICAL_MATERIALIZATION_BATCH_SIZE,
        device=device,
        require_contiguous=False,
    )
    fit_joined_identity_sha256 = \
        canonical_residual_joined_identity_sha256(
            joined_split["fit_rows"]
        )
    fit_baseline = build_hierarchical_cache_calibration_baseline(
        fit_records
    )
    fit_materialization_sha256 = fit_baseline[
        "row_materialization_sha256"
    ]
    fit_candidate_iou_sha256 = fit_baseline["candidate_iou_sha256"]
    fit_deployable_sha256 = canonical_hierarchical_deployable_sha256(
        fit_records
    )
    cross_fit = cross_fit_hierarchical_reranker(
        fit_records, device=device
    )
    choice = copy.deepcopy(cross_fit["choice"])
    result_context = {
        "input_sha256": copy.deepcopy(loaded["input_sha256"]),
        "split": copy.deepcopy(joined_split["metadata"]),
        "fit_joined_identity_sha256": fit_joined_identity_sha256,
        "fit_materialization_sha256": fit_materialization_sha256,
        "fit_deployable_sha256": fit_deployable_sha256,
        "fit_candidate_iou_sha256": fit_candidate_iou_sha256,
        "fit_normalization_sha256": None,
        "oof": {
            "baseline": copy.deepcopy(fit_baseline),
            "scene_folds": copy.deepcopy(cross_fit["scene_folds"]),
            "scene_fold_sha256": cross_fit["scene_fold_sha256"],
            "configuration_count": len(cross_fit["configurations"]),
            "configurations": copy.deepcopy(cross_fit["configurations"]),
            "policy_candidate_count": cross_fit[
                "policy_candidate_count"
            ],
            "choice": copy.deepcopy(choice),
        },
        "calibration": {
            "status": "not_run",
            "reason": "oof_selection_rejected",
        },
    }
    if choice.get("eligible") is not True:
        return publish_hierarchical_experiment(
            output_path,
            artifact=None,
            result_context=result_context,
            protected_paths=protected_paths,
            protected_before=protected_before,
            reservation=reservation,
        )

    refit_model, normalization = refit_hierarchical_reranker(
        fit_records, choice, device=device
    )
    result_context["fit_normalization_sha256"] = normalization["sha256"]
    calibration_records = materialize_hierarchical_rows(
        joined_split["calibration_rows"],
        loaded["parent"],
        loaded["geometry_model"],
        loaded["geometry_artifact"],
        batch_size=HIERARCHICAL_MATERIALIZATION_BATCH_SIZE,
        device=device,
        require_contiguous=False,
    )
    calibration_baseline = build_hierarchical_cache_calibration_baseline(
        calibration_records
    )
    calibration_record = evaluate_hierarchical_cache_policy(
        refit_model,
        calibration_records,
        normalization,
        margin=float(choice["margin"]),
        device=device,
    )
    gate = hierarchical_calibration_gate(
        calibration_record, calibration_baseline
    )
    result_context["calibration"] = {
        "status": "run",
        "baseline": copy.deepcopy(calibration_baseline),
        "record": copy.deepcopy(calibration_record),
        "gate": hierarchical_calibration_gate_receipt(gate),
    }

    artifact = None
    if gate.passed:
        selected_configuration = _selected_hierarchical_configuration(
            cross_fit["configurations"], choice
        )
        oof_record = {
            "prediction_count": selected_configuration[
                "prediction_count"
            ],
            "proposal_sha256": selected_configuration[
                "oof_proposal_sha256"
            ],
            "gain_sha256": selected_configuration["oof_gain_sha256"],
            "delta_hits025": choice["delta_hits025"],
            "delta_hits050": choice["delta_hits050"],
        }
        artifact = build_hierarchical_artifact(
            model=refit_model,
            selection=choice,
            normalization=normalization,
            scene_folds=cross_fit["scene_folds"],
            geometry_feature_names=loaded["geometry_artifact"][
                "feature_names"
            ],
            input_sha256=loaded["input_sha256"],
            row_materialization_sha256=calibration_baseline[
                "row_materialization_sha256"
            ],
            candidate_iou_sha256=calibration_baseline[
                "candidate_iou_sha256"
            ],
            oof_record=oof_record,
            calibration_record=calibration_record,
            calibration_baseline=calibration_baseline,
        )
    return publish_hierarchical_experiment(
        output_path,
        artifact=artifact,
        result_context=result_context,
        protected_paths=protected_paths,
        protected_before=protected_before,
        reservation=reservation,
    )


def main(argv=None):
    args = parse_args(argv)
    return run_hierarchical_training(
        args.base_cache,
        args.geometry_cache,
        args.parent_artifact,
        args.geometry_artifact,
        args.output_dir,
        device=args.device,
    )


__all__ = [
    "AUTHORITATIVE_BACKBONE_PATH",
    "AUTHORITATIVE_BACKBONE_SHA256",
    "AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256",
    "AUTHORITATIVE_GEOMETRY_CACHE_CONTENT_SHA256",
    "AUTHORITATIVE_GEOMETRY_METADATA_SHA256",
    "AUTHORITATIVE_PARENT_ARTIFACT_SHA256",
    "AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256",
    "HIERARCHICAL_MATERIALIZATION_BATCH_SIZE",
    "HIERARCHICAL_ARTIFACT_NAME",
    "HIERARCHICAL_ARTIFACT_SCHEMA",
    "HIERARCHICAL_ARTIFACT_VERSION",
    "HIERARCHICAL_RESULT_RECEIPT_SCHEMA",
    "HIERARCHICAL_RESULT_RECEIPT_VERSION",
    "HIERARCHICAL_BATCH_SIZE",
    "HIERARCHICAL_DROPOUT",
    "HIERARCHICAL_EPOCHS",
    "HIERARCHICAL_GAIN_QUANTILES",
    "HIERARCHICAL_GRAD_CLIP_NORM",
    "HIERARCHICAL_LEARNING_RATE",
    "HIERARCHICAL_MODEL_SEED",
    "HIERARCHICAL_MIN_STD",
    "HIERARCHICAL_MODEL_BATCH_FIELDS",
    "HIERARCHICAL_NORMALIZATION_GROUPS",
    "HIERARCHICAL_NORMALIZATION_SCHEMA",
    "HIERARCHICAL_RECORD_FIELDS",
    "canonical_hierarchical_candidate_iou_sha256",
    "canonical_hierarchical_deployable_sha256",
    "canonical_hierarchical_rows_sha256",
    "build_hierarchical_cache_calibration_baseline",
    "build_hierarchical_artifact",
    "build_hierarchical_feature_names",
    "build_hierarchical_result_receipt",
    "build_hierarchical_policy_candidate",
    "capture_immutable_artifact_identities",
    "cross_fit_hierarchical_reranker",
    "fit_hierarchical_normalization",
    "evaluate_hierarchical_cache_policy",
    "hierarchical_calibration_gate",
    "hierarchical_calibration_gate_receipt",
    "load_hierarchical_artifact",
    "load_residual_training_inputs",
    "materialize_hierarchical_rows",
    "nearest_rank_hierarchical_margin",
    "parse_args",
    "publish_hierarchical_experiment",
    "normalize_hierarchical_batch",
    "refit_hierarchical_reranker",
    "reserve_hierarchical_output",
    "run_hierarchical_training",
    "save_hierarchical_artifact",
    "summarize_hierarchical_training_labels",
    "summarize_hierarchical_oof_gain",
    "split_residual_joined_rows",
    "validate_hierarchical_artifact",
    "validate_hierarchical_result_receipt",
]


if __name__ == "__main__":
    main()
