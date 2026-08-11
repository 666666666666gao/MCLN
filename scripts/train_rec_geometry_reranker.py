#!/usr/bin/env python
"""Train a scene-disjoint flat REC mask-geometry reranker."""

import argparse
import copy
import contextlib
import hashlib
import io
import math
import os
import random
from pathlib import Path
import stat
import sys
import tempfile

import torch
from torch.nn.utils import clip_grad_norm_


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rec_geometry_reranker import (
    FLAT_PARENT_PRIOR_VERSION,
    REC_GEOMETRY_MODEL_SCHEMA_VERSION,
    build_deployed_parent_state,
    build_flat_parent_prior,
    build_rec_geometry_model_inputs,
    stable_flat_descending_indices,
)
from models.rec_mask_geometry import (
    DEFAULT_REC_MASK_GEOMETRY_VARIANTS,
    MASK_GEOMETRY_SCHEMA_VERSION,
    REC_MASK_GEOMETRY_FEATURE_NAMES,
)
from models.rec_candidate_adapter import FEATURE_SCHEMA_VERSION
from models.rec_reranker import (
    QueryReranker,
    blend_candidate_scores,
    compute_rec_reranker_loss,
)
from scripts.rec_geometry_cache import (
    GEOMETRY_CACHE_SCHEMA_VERSION,
    canonical_json_sha256,
    join_base_and_geometry_rows,
    load_bound_candidate_cache,
    load_geometry_cache,
)
from scripts.train_rec_reranker import (
    ARTIFACT_VERSION as PARENT_ARTIFACT_VERSION,
    BACKBONE_CONFIG_KEYS,
    CACHE_SCHEMA_VERSION,
    LEGACY_BACKBONE_CONFIG_KEYS,
    MODEL_INPUT_KEYS,
    _validate_artifact as _validate_parent_artifact,
    normalize_backbone_config,
    normalize_features,
)


BASE_FEATURE_DIM = 152
GEOMETRY_FEATURE_DIM = 25
GEOMETRY_INPUT_DIM = BASE_FEATURE_DIM + GEOMETRY_FEATURE_DIM + 2
GEOMETRY_VARIANT_COUNT = 7
GEOMETRY_CANDIDATE_COUNT = 16


def _has_supported_backbone_config(backbone):
    if not isinstance(backbone, dict):
        return False
    supported_fields = (
        set(LEGACY_BACKBONE_CONFIG_KEYS), set(BACKBONE_CONFIG_KEYS)
    )
    if set(backbone) not in supported_fields:
        return False
    try:
        normalize_backbone_config(backbone)
    except ValueError:
        return False
    return True
FLAT_GEOMETRY_CANDIDATE_COUNT = (
    GEOMETRY_CANDIDATE_COUNT * GEOMETRY_VARIANT_COUNT
)
DEPLOYED_QUERY_COUNT = 256
DEFAULT_CALIBRATION_FRACTION = 0.10
MIN_FEATURE_STD = 1e-6
DEFAULT_GEOMETRY_WEIGHTS = (
    0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0
)
GEOMETRY_ARTIFACT_VERSION = 2
GEOMETRY_SCORE_MODE = "parent-flat-rank-blend-v1"
GEOMETRY_TIE_POLICY = "score-desc-flat-index-asc-v1"
EVALUATOR_FILTER_POLICY = "evaluator-valid-no-gt-filter-v1"
TARGET_IOU_POLICY = "root_only"
GRAD_CLIP_NORM = 1.0
PARENT_INFERENCE_LOCAL_BATCH_SIZE = 12
PARENT_INFERENCE_CONTRACT_FIELDS = (
    "schema",
    "version",
    "device_type",
    "device_index",
    "local_batch_size",
    "world_size",
    "row_order",
    "remainder_policy",
    "feature_source",
    "dtype",
    "autocast",
    "allow_tf32",
    "eval",
    "no_grad",
    "score_builder",
    "score_builder_version",
    "canonical_query_tie_policy",
    "content_digest_version",
    "row_count",
    "score_content_sha256",
)
PARENT_INFERENCE_SCHEMA = "rec-parent-inference-contract"
PARENT_INFERENCE_VERSION = 1
PARENT_SCORE_BUILDER = "normalized-query-reranker-rank-blend"
PARENT_SCORE_BUILDER_VERSION = 1
PARENT_QUERY_TIE_POLICY = "score-desc-query-index-asc-v1"
PARENT_SCORE_CONTENT_DIGEST_VERSION = \
    "ordered-identity-raw-float32-sha256-v1"

AUTHORITATIVE_PARENT_ARTIFACT_SHA256 = (
    "f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b"
)
AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256 = (
    "411ec7d5d80a7be9596de20b348667d529e6a8f568b8ab0c0e0922b8719f9045"
)
AUTHORITATIVE_SPLIT_SEED0 = {
    "split_seed": 0,
    "calibration_fraction": 0.10,
    "scene_count": 562,
    "fit_scene_count": 506,
    "calibration_scene_count": 56,
    "sample_count": 36665,
    "fit_sample_count": 33040,
    "calibration_sample_count": 3625,
    "fit_scene_sha256": (
        "790264c59d4e4f5937b49b0440c020d485c0929a843176a3a434f2ce8d797a17"
    ),
    "calibration_scene_sha256": (
        "f58524379488c4bd061849167f537ba3a10671317b30c89dd580ba147e8e5cdc"
    ),
    "mapping_sha256": (
        "72685aa01285dbe72b9e0331acd5f10457f773e9e158ae4f884b9c4176cf95bd"
    ),
}

GEOMETRY_ARTIFACT_FIELDS = (
    "artifact_version",
    "model_state_dict",
    "model_config",
    "model_schema_version",
    "geometry_cache_schema_version",
    "geometry_schema_version",
    "base_cache_schema_version",
    "base_feature_schema_version",
    "input_dim",
    "feature_names",
    "feature_mean",
    "feature_std",
    "variant_names",
    "variant_configs",
    "regressed_variant_index",
    "min_points",
    "max_point_fraction",
    "candidate_rule",
    "checkpoint_sha256",
    "checkpoint_epoch",
    "model_inputs",
    "backbone_config",
    "parent_artifact_sha256",
    "parent_provenance",
    "parent_inference_contract",
    "num_queries",
    "flat_parent_prior_version",
    "tie_policy",
    "score_mode",
    "geometry_weight",
    "target_iou_policy",
    "evaluator_filter_policy",
    "filter_non_gt_boxes",
    "train_parent_score_content_sha256",
    "train_base_cache_content_digest",
    "train_base_cache_manifest_digest",
    "train_geometry_cache_content_digest",
    "train_geometry_immutable_metadata_digest",
    "scene_split",
    "epoch",
    "calibration_metrics",
    "training_args",
)
_PARENT_PROVENANCE_FIELDS = frozenset((
    "artifact_version",
    "adapter_schema_version",
    "input_dim",
    "feature_names",
    "model_config",
    "candidate_rule",
    "checkpoint_sha256",
    "target_iou_policy",
    "model_inputs",
    "backbone_config",
    "score_mode",
    "reranker_weight",
    "epoch",
    "feature_mean_sha256",
    "feature_std_sha256",
))
_SCENE_SPLIT_FIELDS = frozenset(AUTHORITATIVE_SPLIT_SEED0)
CALIBRATION_METRIC_FIELDS = (
    "sample_count",
    "hits025",
    "hits050",
    "parent_hits025",
    "parent_hits050",
    "fixes025",
    "breaks025",
    "fixes050",
    "breaks050",
    "geometry_oracle_hits025",
    "geometry_oracle_hits050",
    "acc025",
    "acc050",
    "parent_acc025",
    "parent_acc050",
    "geometry_oracle_acc025",
    "geometry_oracle_acc050",
    "score",
)
_CALIBRATION_METRIC_FIELDS = frozenset(CALIBRATION_METRIC_FIELDS)
_TRAINING_ARGUMENT_FIELDS = frozenset((
    "split_seed",
    "model_seed",
    "calibration_fraction",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "patience",
    "device",
    "parent_allow_tf32",
    "grad_clip_norm",
    "geometry_weight_grid",
))


def _strict_int(value, name, minimum=0):
    if (not isinstance(value, int) or isinstance(value, bool)
            or value < minimum):
        raise ValueError("{} is invalid".format(name))
    return value


def _is_exact_int(value, expected):
    return (isinstance(value, int) and not isinstance(value, bool)
            and value == expected)


def _joined_identity(row):
    if not isinstance(row, dict) or set(row) != {"base", "geometry"}:
        raise ValueError("joined rows must contain nested base and geometry rows")
    base = row["base"]
    geometry = row["geometry"]
    if not isinstance(base, dict) or not isinstance(geometry, dict):
        raise ValueError("joined base and geometry rows must be objects")
    identity_keys = ("dataset_index", "scan_id", "target_id")
    if any(base.get(key) != geometry.get(key) for key in identity_keys):
        raise ValueError("joined base and geometry row identities differ")
    _strict_int(base.get("dataset_index"), "joined dataset index")
    _strict_int(base.get("target_id"), "joined target id")
    scan_id = base.get("scan_id")
    if not isinstance(scan_id, str) or not scan_id:
        raise ValueError("joined scan identity is invalid")
    return base, geometry


def deterministic_scene_split(rows, seed,
                              calibration_fraction=DEFAULT_CALIBRATION_FRACTION):
    """Split nested ``{base, geometry}`` pairs without crossing scenes."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("cannot split empty joined geometry rows")
    fraction = float(calibration_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")
    scene_by_row = []
    for row in rows:
        base, _ = _joined_identity(row)
        scene_by_row.append(base["scan_id"])
    scenes = sorted(set(scene_by_row))
    if len(scenes) == 1:
        return list(rows), []
    shuffled = list(scenes)
    random.Random(int(seed)).shuffle(shuffled)
    calibration_count = int(round(len(scenes) * fraction))
    calibration_count = max(1, min(calibration_count, len(scenes) - 1))
    calibration_scenes = set(shuffled[:calibration_count])
    fit_rows = [
        row for row, scene in zip(rows, scene_by_row)
        if scene not in calibration_scenes
    ]
    calibration_rows = [
        row for row, scene in zip(rows, scene_by_row)
        if scene in calibration_scenes
    ]
    return fit_rows, calibration_rows


def build_scene_split_metadata(fit_rows, calibration_rows, split_seed,
                               calibration_fraction=DEFAULT_CALIBRATION_FRACTION):
    """Build canonical scene-membership digests for artifact provenance."""
    if not isinstance(fit_rows, (list, tuple)) or not fit_rows:
        raise ValueError("fit rows cannot be empty")
    if not isinstance(calibration_rows, (list, tuple)):
        raise ValueError("calibration rows must be a sequence")
    fraction = float(calibration_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")
    fit_scenes = sorted({_joined_identity(row)[0]["scan_id"] for row in fit_rows})
    calibration_scenes = sorted({
        _joined_identity(row)[0]["scan_id"] for row in calibration_rows
    })
    if set(fit_scenes) & set(calibration_scenes):
        raise ValueError("fit and calibration scene sets overlap")
    mapping = {"fit": fit_scenes, "calibration": calibration_scenes}
    return {
        "split_seed": int(split_seed),
        "calibration_fraction": fraction,
        "scene_count": len(fit_scenes) + len(calibration_scenes),
        "fit_scene_count": len(fit_scenes),
        "calibration_scene_count": len(calibration_scenes),
        "sample_count": len(fit_rows) + len(calibration_rows),
        "fit_sample_count": len(fit_rows),
        "calibration_sample_count": len(calibration_rows),
        "fit_scene_sha256": canonical_json_sha256(fit_scenes),
        "calibration_scene_sha256": canonical_json_sha256(
            calibration_scenes
        ),
        "mapping_sha256": canonical_json_sha256(mapping),
    }


def _require_tensor(value, shape, dtype, name):
    if (not isinstance(value, torch.Tensor)
            or tuple(value.shape) != tuple(shape)
            or value.dtype != dtype):
        raise ValueError("{} must have dtype {} and shape {}".format(
            name, dtype, tuple(shape)
        ))
    if value.device.type != "cpu":
        raise ValueError("{} must use CPU storage".format(name))
    return value


def _parent_parts(parent):
    if (not isinstance(parent, (list, tuple)) or len(parent) != 2
            or not isinstance(parent[0], torch.nn.Module)
            or not isinstance(parent[1], dict)):
        raise ValueError("parent must be a (model, artifact) pair")
    model, artifact = parent
    mean = artifact.get("feature_mean")
    std = artifact.get("feature_std")
    names = artifact.get("feature_names")
    if (not isinstance(mean, torch.Tensor)
            or not isinstance(std, torch.Tensor)
            or mean.shape != (BASE_FEATURE_DIM,)
            or std.shape != (BASE_FEATURE_DIM,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or bool((std < MIN_FEATURE_STD).any().item())):
        raise ValueError("parent artifact normalization is invalid")
    if (artifact.get("input_dim") != BASE_FEATURE_DIM
            or not isinstance(names, list)
            or len(names) != BASE_FEATURE_DIM
            or len(set(names)) != BASE_FEATURE_DIM
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("parent artifact feature schema is invalid")
    if artifact.get("score_mode") != "rank_blend":
        raise ValueError("parent artifact score mode is unsupported")
    weight = artifact.get("reranker_weight")
    if (not isinstance(weight, (float, int)) or isinstance(weight, bool)
            or not 0.0 <= float(weight) <= 1.0):
        raise ValueError("parent artifact reranker weight is invalid")
    backbone = artifact.get("backbone_config")
    if (not isinstance(backbone, dict)
            or backbone.get("num_target") != DEPLOYED_QUERY_COUNT):
        raise ValueError("parent artifact must use the deployed 256-query axis")
    model.eval()
    model.requires_grad_(False)
    return model, artifact


def _validate_parent_artifact_metadata(model, artifact):
    if not _is_sha256(getattr(model, "_artifact_sha256", None)):
        return
    expected_provenance = getattr(
        model, "_artifact_parent_provenance", None
    )
    if (not isinstance(expected_provenance, dict)
            or _parent_provenance(artifact) != expected_provenance):
        raise ValueError(
            "in-memory parent artifact metadata differs from hashed artifact"
        )


def _validate_live_parent_state(model, artifact):
    if not _is_sha256(getattr(model, "_artifact_sha256", None)):
        return
    _validate_parent_artifact_metadata(model, artifact)
    expected = getattr(model, "_artifact_expected_state", None)
    artifact_state = artifact.get("model_state_dict")
    actual = model.state_dict()
    if (not isinstance(expected, dict) or set(expected) != set(actual)
            or not _payloads_equal(artifact_state, expected)
            or any(not isinstance(expected[name], torch.Tensor)
                   or expected[name].dtype != actual[name].dtype
                   or tuple(expected[name].shape) != tuple(actual[name].shape)
                   or not torch.equal(
                       expected[name].detach().cpu(),
                       actual[name].detach().cpu(),
                   ) for name in actual)):
        raise ValueError("live parent model state differs from hashed artifact")


def _module_device(model):
    for value in list(model.parameters()) + list(model.buffers()):
        return value.device
    return torch.device("cpu")


def _parent_matmul_allow_tf32(device):
    if device.type != "cuda":
        return False
    return bool(torch.backends.cuda.matmul.allow_tf32)


def _disabled_parent_autocast(device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    if device.type != "cpu":
        raise ValueError("parent inference supports only CPU or CUDA devices")
    cpu_amp = getattr(getattr(torch, "cpu", None), "amp", None)
    cpu_autocast = getattr(cpu_amp, "autocast", None)
    if cpu_autocast is not None:
        return cpu_autocast(enabled=False)
    generic_autocast = getattr(torch, "autocast", None)
    if generic_autocast is not None:
        return generic_autocast(device_type="cpu", enabled=False)
    if (hasattr(torch, "is_autocast_cpu_enabled")
            and torch.is_autocast_cpu_enabled()):
        raise RuntimeError("this PyTorch build cannot disable CPU autocast")
    return contextlib.nullcontext()


def _parent_score_cache_signature(model, artifact, device=None):
    device = _module_device(model) if device is None else torch.device(device)
    return {
        "artifact_object_id": id(artifact),
        "parent_artifact_sha256": getattr(model, "_artifact_sha256", None),
        "reranker_weight": float(artifact["reranker_weight"]),
        "device_type": device.type,
        "device_index": device.index,
        "allow_tf32": _parent_matmul_allow_tf32(device),
    }


def _validate_batch_row(row):
    base, geometry = _joined_identity(row)
    _require_tensor(
        base.get("features"),
        (GEOMETRY_CANDIDATE_COUNT, BASE_FEATURE_DIM),
        torch.float32,
        "base features",
    )
    base_valid = _require_tensor(
        base.get("valid_mask"),
        (GEOMETRY_CANDIDATE_COUNT,),
        torch.bool,
        "base validity",
    )
    query_indices = _require_tensor(
        base.get("query_indices"),
        (GEOMETRY_CANDIDATE_COUNT,),
        torch.int64,
        "base query indices",
    )
    _require_tensor(
        base.get("default_scores"),
        (GEOMETRY_CANDIDATE_COUNT,),
        torch.float32,
        "base default scores",
    )
    if not torch.equal(geometry.get("candidate_valid"), base_valid):
        raise ValueError("geometry candidate validity differs from base validity")
    if not torch.equal(geometry.get("query_indices"), query_indices):
        raise ValueError("geometry query indices differ from base query indices")
    shapes = {
        "geometry_boxes": (
            (GEOMETRY_CANDIDATE_COUNT, GEOMETRY_VARIANT_COUNT, 6),
            torch.float32,
        ),
        "geometry_features": (
            (GEOMETRY_CANDIDATE_COUNT, GEOMETRY_VARIANT_COUNT,
             GEOMETRY_FEATURE_DIM),
            torch.float32,
        ),
        "geometry_ious": (
            (GEOMETRY_CANDIDATE_COUNT, GEOMETRY_VARIANT_COUNT),
            torch.float32,
        ),
        "geometry_valid": (
            (GEOMETRY_CANDIDATE_COUNT, GEOMETRY_VARIANT_COUNT),
            torch.bool,
        ),
        "evaluator_valid": (
            (GEOMETRY_CANDIDATE_COUNT, GEOMETRY_VARIANT_COUNT),
            torch.bool,
        ),
    }
    for key, (shape, dtype) in shapes.items():
        _require_tensor(geometry.get(key), shape, dtype, key)
    evaluator_valid = geometry["evaluator_valid"]
    if bool((evaluator_valid & ~geometry["geometry_valid"]).any().item()):
        raise ValueError("evaluator validity exceeds geometry validity")
    if not bool(evaluator_valid.any().item()):
        raise ValueError("joined row has no evaluator-valid geometry")
    return base, geometry


def _cached_parent_compact_scores(model, artifact, base_rows):
    _validate_parent_artifact_metadata(model, artifact)
    signature = _parent_score_cache_signature(model, artifact)
    cache_state = getattr(model, "_geometry_parent_score_cache", None)
    if not isinstance(cache_state, dict):
        cache_state = {"signature": signature, "rows": {}}
        model._geometry_parent_score_cache = cache_state
    elif cache_state.get("signature") != signature:
        if cache_state.get("sealed") is True:
            raise ValueError("sealed parent score cache signature changed")
        cache_state = {"signature": signature, "rows": {}}
        model._geometry_parent_score_cache = cache_state
    cache = cache_state["rows"]
    compact_by_position = [None] * len(base_rows)
    missing_positions = []
    missing_rows = []
    for position, base_row in enumerate(base_rows):
        cached = cache.get(id(base_row))
        if cached is not None and cached[0] is base_row:
            compact_by_position[position] = cached[1]
        else:
            missing_positions.append(position)
            missing_rows.append(base_row)
    if missing_rows:
        if cache_state.get("sealed") is True:
            raise ValueError("sealed parent score cache is missing a requested row")
        _validate_live_parent_state(model, artifact)
        base_features = torch.stack([row["features"] for row in missing_rows])
        base_valid = torch.stack([row["valid_mask"] for row in missing_rows])
        default_scores = torch.stack([
            row["default_scores"] for row in missing_rows
        ])
        parent_device = _module_device(model)
        with torch.no_grad(), _disabled_parent_autocast(parent_device):
            local_valid = base_valid.to(parent_device)
            normalized = normalize_features(
                base_features.to(parent_device, dtype=torch.float32),
                local_valid,
                artifact["feature_mean"],
                artifact["feature_std"],
            )
            if not bool(torch.isfinite(normalized).all().item()):
                raise ValueError("normalized parent features are non-finite")
            outputs = model(normalized, base_valid.to(parent_device))
            missing_scores = blend_candidate_scores(
                default_scores.to(parent_device, dtype=torch.float32),
                outputs["ranking_logits"],
                local_valid,
                reranker_weight=float(artifact["reranker_weight"]),
            ).detach().to(dtype=torch.float32).cpu()
        for local_index, position in enumerate(missing_positions):
            score = missing_scores[local_index].clone()
            base_row = base_rows[position]
            cache[id(base_row)] = (base_row, score)
            compact_by_position[position] = score
    return torch.stack(compact_by_position)


def _parent_score_content_sha256(ordered_base_rows, compact_scores):
    if (not isinstance(compact_scores, torch.Tensor)
            or compact_scores.dtype != torch.float32
            or compact_scores.device.type != "cpu"
            or tuple(compact_scores.shape)
            != (len(ordered_base_rows), GEOMETRY_CANDIDATE_COUNT)):
        raise ValueError("materialized parent scores have an invalid layout")
    digest = hashlib.sha256()
    for base, score in zip(ordered_base_rows, compact_scores):
        dataset_index = _strict_int(
            base.get("dataset_index"), "parent digest dataset index"
        )
        target_id = _strict_int(
            base.get("target_id"), "parent digest target id"
        )
        scan_id = base.get("scan_id")
        if not isinstance(scan_id, str) or not scan_id:
            raise ValueError("parent digest scan identity is invalid")
        scan_bytes = scan_id.encode("utf-8")
        digest.update(dataset_index.to_bytes(8, "little"))
        digest.update(len(scan_bytes).to_bytes(8, "little"))
        digest.update(scan_bytes)
        digest.update(target_id.to_bytes(8, "little"))
        digest.update(score.contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parent_inference_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = int(torch.distributed.get_world_size())
    else:
        world_size = 1
    if world_size != 1:
        raise ValueError("parent score materialization requires world_size=1")
    return world_size


def _build_parent_inference_contract(model, row_count, content_sha256):
    device = _module_device(model)
    return {
        "schema": PARENT_INFERENCE_SCHEMA,
        "version": PARENT_INFERENCE_VERSION,
        "device_type": device.type,
        "device_index": device.index,
        "local_batch_size": PARENT_INFERENCE_LOCAL_BATCH_SIZE,
        "world_size": _parent_inference_world_size(),
        "row_order": "dataset-index-contiguous",
        "remainder_policy": "natural-remainder",
        "feature_source": "bound-base-cache-features",
        "dtype": "float32",
        "autocast": False,
        "allow_tf32": _parent_matmul_allow_tf32(device),
        "eval": True,
        "no_grad": True,
        "score_builder": PARENT_SCORE_BUILDER,
        "score_builder_version": PARENT_SCORE_BUILDER_VERSION,
        "canonical_query_tie_policy": PARENT_QUERY_TIE_POLICY,
        "content_digest_version": PARENT_SCORE_CONTENT_DIGEST_VERSION,
        "row_count": int(row_count),
        "score_content_sha256": content_sha256,
    }


def _sealed_parent_materialization_metadata(parent):
    model, artifact = _parent_parts(parent)
    cache_state = getattr(model, "_geometry_parent_score_cache", None)
    if (not isinstance(cache_state, dict)
            or cache_state.get("sealed") is not True):
        raise ValueError("sealed parent score materialization is required")
    if cache_state.get("signature") != _parent_score_cache_signature(
            model, artifact):
        raise ValueError("sealed parent score materialization signature changed")
    contract = cache_state.get("parent_inference_contract")
    content_sha256 = cache_state.get("train_parent_score_content_sha256")
    if (not isinstance(contract, dict)
            or set(contract) != set(PARENT_INFERENCE_CONTRACT_FIELDS)
            or not _is_sha256(content_sha256)
            or contract.get("score_content_sha256") != content_sha256):
        raise ValueError("sealed parent score materialization metadata is invalid")
    rows = cache_state.get("rows")
    ordered_ids = cache_state.get("ordered_row_object_ids")
    if (not isinstance(rows, dict)
            or not isinstance(ordered_ids, tuple)
            or not ordered_ids
            or len(set(ordered_ids)) != len(ordered_ids)
            or set(rows) != set(ordered_ids)):
        raise ValueError("sealed parent score cache row set is invalid")
    ordered_rows = []
    ordered_scores = []
    for row_object_id in ordered_ids:
        if (not isinstance(row_object_id, int)
                or isinstance(row_object_id, bool)
                or row_object_id <= 0):
            raise ValueError("sealed parent score cache row id is invalid")
        entry = rows.get(row_object_id)
        if (not isinstance(entry, tuple) or len(entry) != 2):
            raise ValueError("sealed parent score cache entry is invalid")
        base_row, score = entry
        if not isinstance(base_row, dict) or id(base_row) != row_object_id:
            raise ValueError("sealed parent score cache object identity is invalid")
        if (not isinstance(score, torch.Tensor)
                or score.device.type != "cpu"
                or score.dtype != torch.float32
                or tuple(score.shape) != (GEOMETRY_CANDIDATE_COUNT,)
                or not bool(torch.isfinite(score).all().item())):
            raise ValueError("sealed parent score cache tensor is invalid")
        ordered_rows.append(base_row)
        ordered_scores.append(score)
    rebuilt_scores = torch.stack(ordered_scores)
    rebuilt_digest = _parent_score_content_sha256(
        ordered_rows, rebuilt_scores
    )
    rebuilt_contract = _build_parent_inference_contract(
        model, len(ordered_rows), rebuilt_digest
    )
    if rebuilt_digest != content_sha256 or rebuilt_contract != contract:
        raise ValueError("sealed parent score cache digest is invalid")
    return copy.deepcopy(contract), content_sha256


def materialize_parent_scores(
        rows, parent, device="cpu",
        local_batch_size=PARENT_INFERENCE_LOCAL_BATCH_SIZE):
    """Materialize frozen parent scores in the deployed local batch shape."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("parent score materialization rows cannot be empty")
    if not _is_exact_int(
            local_batch_size, PARENT_INFERENCE_LOCAL_BATCH_SIZE):
        raise ValueError("parent inference local batch size must be 12")
    _parent_inference_world_size()
    model, artifact = _parent_parts(parent)
    resolved_device = _resolve_device(device)
    cache_state = getattr(model, "_geometry_parent_score_cache", None)
    if (isinstance(cache_state, dict)
            and cache_state.get("sealed") is True
            and cache_state.get("signature")
            != _parent_score_cache_signature(model, artifact)):
        raise ValueError("sealed parent score cache signature changed")
    signature = _parent_score_cache_signature(
        model, artifact, device=resolved_device
    )
    if (isinstance(cache_state, dict)
            and cache_state.get("sealed") is True
            and cache_state.get("signature") != signature):
        raise ValueError("sealed parent score cache signature changed")
    model.to(device=resolved_device, dtype=torch.float32).eval()
    if _parent_score_cache_signature(model, artifact) != signature:
        raise RuntimeError("resolved parent score cache signature changed")
    sealed_reentry = (
        isinstance(cache_state, dict)
        and cache_state.get("signature") == signature
        and cache_state.get("sealed") is True
    )
    if (not isinstance(cache_state, dict)
            or cache_state.get("signature") != signature
            or cache_state.get("sealed") is not True):
        model._geometry_parent_score_cache = {
            "signature": signature,
            "rows": {},
        }
    ordered = sorted(
        (_validate_batch_row(row) for row in rows),
        key=lambda pair: pair[0]["dataset_index"],
    )
    indices = [pair[0]["dataset_index"] for pair in ordered]
    if indices != list(range(len(ordered))):
        raise ValueError("parent materialization dataset indices are not contiguous")
    ordered_row_object_ids = tuple(id(pair[0]) for pair in ordered)
    if (sealed_reentry
            and cache_state.get("ordered_row_object_ids")
            != ordered_row_object_ids):
        raise ValueError(
            "sealed parent materialization requires the exact train row set"
        )
    if sealed_reentry:
        _sealed_parent_materialization_metadata(parent)
        return torch.stack([
            cache_state["rows"][row_object_id][1]
            for row_object_id in ordered_row_object_ids
        ])
    scores = []
    for start in range(0, len(ordered), PARENT_INFERENCE_LOCAL_BATCH_SIZE):
        base_rows = [
            pair[0]
            for pair in ordered[start:start + PARENT_INFERENCE_LOCAL_BATCH_SIZE]
        ]
        scores.append(_cached_parent_compact_scores(model, artifact, base_rows))
    compact_scores = torch.cat(scores, dim=0).float()
    ordered_base_rows = [pair[0] for pair in ordered]
    content_sha256 = _parent_score_content_sha256(
        ordered_base_rows, compact_scores
    )
    contract = _build_parent_inference_contract(
        model, len(ordered_base_rows), content_sha256
    )
    cache_state = model._geometry_parent_score_cache
    cache_state["parent_inference_contract"] = contract
    cache_state["train_parent_score_content_sha256"] = content_sha256
    cache_state["ordered_row_object_ids"] = ordered_row_object_ids
    cache_state["sealed"] = True
    return compact_scores


def build_geometry_training_batch(rows, parent):
    """Build one target-bearing flat batch without feeding targets to models."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("geometry batch rows cannot be empty")
    model, artifact = _parent_parts(parent)
    validated = [_validate_batch_row(row) for row in rows]
    base_rows = [pair[0] for pair in validated]
    geometry_rows = [pair[1] for pair in validated]
    base_features = torch.stack([row["features"] for row in base_rows])
    base_valid = torch.stack([row["valid_mask"] for row in base_rows])
    query_indices = torch.stack([row["query_indices"] for row in base_rows])
    compact_scores = _cached_parent_compact_scores(
        model, artifact, base_rows
    )
    parent_state = build_deployed_parent_state(
        compact_scores,
        query_indices,
        base_valid,
        DEPLOYED_QUERY_COUNT,
    )

    geometry_features = torch.stack([
        row["geometry_features"] for row in geometry_rows
    ])
    evaluator_valid = torch.stack([
        row["evaluator_valid"] for row in geometry_rows
    ])
    flat = build_rec_geometry_model_inputs(
        base_features,
        geometry_features,
        compact_scores,
        parent_state["parent_top1_mask"],
        evaluator_valid,
        artifact["feature_names"],
        REC_MASK_GEOMETRY_FEATURE_NAMES,
    )
    if (flat["features"].shape[-1] != GEOMETRY_INPUT_DIM
            or flat["features"].shape[1]
            != FLAT_GEOMETRY_CANDIDATE_COUNT):
        raise RuntimeError("flat geometry feature contract changed")
    geometry_boxes = torch.stack([
        row["geometry_boxes"] for row in geometry_rows
    ]).reshape(len(rows), FLAT_GEOMETRY_CANDIDATE_COUNT, 6)
    geometry_ious = torch.stack([
        row["geometry_ious"] for row in geometry_rows
    ]).reshape(len(rows), FLAT_GEOMETRY_CANDIDATE_COUNT)
    return {
        "features": flat["features"],
        "boxes": geometry_boxes,
        "valid_mask": flat["valid_mask"],
        "candidate_ious": geometry_ious,
        "feature_names": flat["feature_names"],
        "query_positions": flat["query_positions"],
        "variant_indices": flat["variant_indices"],
        "parent_state": parent_state,
    }


def compute_geometry_feature_stats(rows, parent, min_std=MIN_FEATURE_STD,
                                   batch_size=256):
    """Stream float64 population statistics over fit evaluator candidates."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("fit geometry rows cannot be empty")
    if (not isinstance(min_std, (float, int)) or isinstance(min_std, bool)
            or not math.isfinite(float(min_std))
            or float(min_std) < MIN_FEATURE_STD):
        raise ValueError("min_std must be at least the artifact floor")
    _strict_int(batch_size, "feature statistics batch size", minimum=1)
    count = 0
    mean = torch.zeros(GEOMETRY_INPUT_DIM, dtype=torch.float64)
    squared_deviation = torch.zeros(
        GEOMETRY_INPUT_DIM, dtype=torch.float64
    )
    for start in range(0, len(rows), int(batch_size)):
        batch = build_geometry_training_batch(
            rows[start:start + int(batch_size)], parent
        )
        values = batch["features"][batch["valid_mask"]].to(torch.float64)
        if values.numel() == 0:
            continue
        batch_count = int(values.shape[0])
        batch_mean = values.mean(dim=0)
        batch_squared_deviation = (
            (values - batch_mean).pow(2).sum(dim=0)
        )
        if count == 0:
            mean = batch_mean
            squared_deviation = batch_squared_deviation
            count = batch_count
            continue
        total = count + batch_count
        delta = batch_mean - mean
        squared_deviation = (
            squared_deviation
            + batch_squared_deviation
            + delta.pow(2) * (count * batch_count / float(total))
        )
        mean = mean + delta * (batch_count / float(total))
        count = total
    if count == 0:
        raise ValueError("fit rows contain no evaluator-valid geometry")
    std = torch.sqrt(squared_deviation / float(count)).clamp(
        min=float(min_std)
    )
    mean = mean.float()
    std = std.float()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("geometry feature statistics are non-finite")
    return mean, std


def calibration_score(acc025, acc050):
    """Return the acceptance-target-aware calibration objective."""
    acc025 = float(acc025)
    acc050 = float(acc050)
    return min(acc025 / 0.60, acc050 / 0.47) + 0.1 * (
        acc025 + acc050
    )


def _validate_normalization(mean, std):
    if (not isinstance(mean, torch.Tensor)
            or not isinstance(std, torch.Tensor)
            or mean.shape != (GEOMETRY_INPUT_DIM,)
            or std.shape != (GEOMETRY_INPUT_DIM,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or bool((std < MIN_FEATURE_STD).any().item())):
        raise ValueError("geometry feature normalization is invalid")


def _parent_flat_indices(parent_state, regressed_variant_index):
    query_indices = parent_state["query_indices"]
    candidate_valid = parent_state["candidate_valid"]
    top1 = parent_state["top1_query_index"]
    matches = (
        query_indices == top1.unsqueeze(1)
    ) & candidate_valid
    if not bool(matches.any(dim=1).all().item()):
        raise ValueError("deployed parent Top-1 is absent from compact candidates")
    compact_positions = matches.to(torch.int64).argmax(dim=1)
    return (
        compact_positions * GEOMETRY_VARIANT_COUNT
        + int(regressed_variant_index)
    )


def _stable_rank_normalize_once(scores, valid):
    orders = stable_flat_descending_indices(scores, valid)
    normalized = torch.full(
        scores.shape,
        -float("inf"),
        dtype=torch.float32,
        device=scores.device,
    )
    for batch_index, order in enumerate(orders):
        indices = torch.tensor(order, dtype=torch.long, device=scores.device)
        denominator = float(max(len(order) - 1, 1))
        ranks = torch.arange(
            len(order), dtype=torch.float32, device=scores.device
        )
        normalized[batch_index, indices] = 1.0 - ranks / denominator
    return normalized


def _stable_flat_top1_indices(scores, valid):
    if scores.shape != valid.shape or scores.dim() != 2:
        raise ValueError("flat score and validity shapes differ")
    masked = scores.masked_fill(~valid, -float("inf"))
    best = masked.max(dim=1, keepdim=True).values
    best_mask = valid & (masked == best)
    if not bool(best_mask.any(dim=1).all().item()):
        raise ValueError("every flat score row needs a valid maximum")
    indices = torch.arange(
        scores.shape[1], dtype=torch.long, device=scores.device
    ).unsqueeze(0).expand_as(scores)
    return indices.masked_fill(~best_mask, scores.shape[1]).min(dim=1).values


def evaluate_geometry_blends(
        model, rows, feature_mean, feature_std, parent,
        geometry_weights=DEFAULT_GEOMETRY_WEIGHTS, batch_size=256,
        device="cpu", regressed_variant_index=0):
    """Evaluate the exact parent path and every declared flat score blend."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("geometry evaluation rows cannot be empty")
    _strict_int(batch_size, "geometry evaluation batch size", minimum=1)
    weights = tuple(float(weight) for weight in geometry_weights)
    if weights != DEFAULT_GEOMETRY_WEIGHTS:
        raise ValueError("geometry evaluation requires the exact weight grid")
    _validate_normalization(feature_mean, feature_std)
    _strict_int(
        regressed_variant_index,
        "regressed variant index",
        minimum=0,
    )
    if regressed_variant_index >= GEOMETRY_VARIANT_COUNT:
        raise ValueError("regressed variant index is out of range")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    model.to(resolved_device)
    was_training = model.training
    model.eval()
    counts = {
        weight: {
            "sample_count": 0,
            "hits025": 0,
            "hits050": 0,
            "parent_hits025": 0,
            "parent_hits050": 0,
            "fixes025": 0,
            "breaks025": 0,
            "fixes050": 0,
            "breaks050": 0,
            "geometry_oracle_hits025": 0,
            "geometry_oracle_hits050": 0,
            "selected_ious": [],
        }
        for weight in weights
    }
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            batch = build_geometry_training_batch(
                rows[start:start + int(batch_size)], parent
            )
            features = normalize_features(
                batch["features"].to(resolved_device),
                batch["valid_mask"].to(resolved_device),
                feature_mean,
                feature_std,
            )
            if not bool(torch.isfinite(features).all().item()):
                raise ValueError("normalized geometry features are non-finite")
            valid = batch["valid_mask"].to(resolved_device)
            ious = batch["candidate_ious"].to(resolved_device)
            outputs = model(features, valid)
            parent_state = {
                key: value.to(resolved_device)
                if isinstance(value, torch.Tensor) else value
                for key, value in batch["parent_state"].items()
            }
            parent_indices = _parent_flat_indices(
                parent_state, regressed_variant_index
            )
            parent_ious = torch.gather(
                ious, 1, parent_indices.unsqueeze(1)
            ).squeeze(1)
            parent_hit025 = parent_ious > 0.25
            parent_hit050 = parent_ious > 0.50
            oracle_ious = ious.masked_fill(~valid, -float("inf")).max(
                dim=1
            ).values
            geometry_valid = valid.reshape(
                valid.shape[0],
                GEOMETRY_CANDIDATE_COUNT,
                GEOMETRY_VARIANT_COUNT,
            )
            parent_prior = build_flat_parent_prior(
                parent_state, geometry_valid, regressed_variant_index
            )
            parent_rank = _stable_rank_normalize_once(
                parent_prior, valid
            )
            learned_rank = _stable_rank_normalize_once(
                outputs["ranking_logits"], valid
            )
            for weight in weights:
                if weight == 0.0:
                    selected_indices = parent_indices
                else:
                    flat_scores = (
                        (1.0 - weight) * parent_rank
                        + weight * learned_rank
                    ).masked_fill(~valid, -float("inf"))
                    selected_indices = _stable_flat_top1_indices(
                        flat_scores, valid
                    )
                selected_ious = torch.gather(
                    ious, 1, selected_indices.unsqueeze(1)
                ).squeeze(1)
                selected_hit025 = selected_ious > 0.25
                selected_hit050 = selected_ious > 0.50
                row_counts = counts[weight]
                row_counts["sample_count"] += int(ious.shape[0])
                row_counts["hits025"] += int(selected_hit025.sum().item())
                row_counts["hits050"] += int(selected_hit050.sum().item())
                row_counts["parent_hits025"] += int(
                    parent_hit025.sum().item()
                )
                row_counts["parent_hits050"] += int(
                    parent_hit050.sum().item()
                )
                row_counts["fixes025"] += int(
                    (selected_hit025 & ~parent_hit025).sum().item()
                )
                row_counts["breaks025"] += int(
                    (~selected_hit025 & parent_hit025).sum().item()
                )
                row_counts["fixes050"] += int(
                    (selected_hit050 & ~parent_hit050).sum().item()
                )
                row_counts["breaks050"] += int(
                    (~selected_hit050 & parent_hit050).sum().item()
                )
                row_counts["geometry_oracle_hits025"] += int(
                    (oracle_ious > 0.25).sum().item()
                )
                row_counts["geometry_oracle_hits050"] += int(
                    (oracle_ious > 0.50).sum().item()
                )
                row_counts["selected_ious"].extend(
                    float(value) for value in selected_ious.detach().cpu()
                )
    model.train(was_training)
    metrics = {}
    for weight in weights:
        row_counts = counts[weight]
        denominator = float(row_counts["sample_count"])
        acc025 = row_counts["hits025"] / denominator
        acc050 = row_counts["hits050"] / denominator
        metrics[weight] = dict(row_counts)
        metrics[weight]["selected_ious"] = tuple(
            row_counts["selected_ious"]
        )
        metrics[weight].update({
            "acc025": acc025,
            "acc050": acc050,
            "parent_acc025": row_counts["parent_hits025"] / denominator,
            "parent_acc050": row_counts["parent_hits050"] / denominator,
            "geometry_oracle_acc025": (
                row_counts["geometry_oracle_hits025"] / denominator
            ),
            "geometry_oracle_acc050": (
                row_counts["geometry_oracle_hits050"] / denominator
            ),
            "score": calibration_score(acc025, acc050),
        })
    return metrics


def choose_best_geometry_blend(metrics_by_weight):
    """Choose a non-regressing blend; score ties keep the lower weight."""
    if not isinstance(metrics_by_weight, dict) or tuple(
            metrics_by_weight) != DEFAULT_GEOMETRY_WEIGHTS:
        raise ValueError("geometry metrics do not match the exact weight grid")
    baseline = metrics_by_weight[0.0]
    baseline_acc025 = float(baseline["acc025"])
    baseline_acc050 = float(baseline["acc050"])
    best_weight = 0.0
    best_metrics = baseline
    best_score = calibration_score(baseline_acc025, baseline_acc050)
    for weight in DEFAULT_GEOMETRY_WEIGHTS[1:]:
        metrics = metrics_by_weight[weight]
        acc025 = float(metrics["acc025"])
        acc050 = float(metrics["acc050"])
        if (acc025 < baseline_acc025 or acc050 < baseline_acc050):
            continue
        score = calibration_score(acc025, acc050)
        if score > best_score:
            best_weight = weight
            best_metrics = metrics
            best_score = score
    return best_weight, best_metrics


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _file_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_snapshot(path, label):
    resolved = Path(path).expanduser().resolve()
    try:
        initial = resolved.stat()
    except OSError as error:
        raise ValueError("{} does not exist: {}".format(label, error))
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("{} must be a regular file".format(label))
    try:
        with resolved.open("rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()))
            snapshot = handle.read()
            after = _file_identity(os.fstat(handle.fileno()))
        current = _file_identity(resolved.stat())
    except OSError as error:
        raise ValueError("could not read {}: {}".format(label, error))
    if before != after or after != current:
        raise ValueError("{} changed during stable snapshot load".format(label))
    return resolved, snapshot, hashlib.sha256(snapshot).hexdigest()


def _resolve_device(device):
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _same_file_or_path(first, second):
    first = Path(first).expanduser().resolve()
    second = Path(second).expanduser().resolve()
    if first == second:
        return True
    try:
        return first.exists() and second.exists() and os.path.samefile(
            str(first), str(second)
        )
    except OSError:
        return False


def _path_is_within(path, directory):
    resolved_path = str(Path(path).expanduser().resolve())
    resolved_directory = str(Path(directory).expanduser().resolve())
    try:
        return os.path.commonpath([
            resolved_path, resolved_directory
        ]) == resolved_directory
    except ValueError:
        return False


def _validate_training_output_location(output, parent_artifact_path,
                                       protected_directories=()):
    if _same_file_or_path(output, parent_artifact_path):
        raise ValueError("training output must not alias the parent artifact")
    for directory in protected_directories:
        if _path_is_within(output, directory):
            raise ValueError("training output must be outside cache directories")


def load_parent_reranker_snapshot(path, device="cpu"):
    """Hash and deserialize the frozen parent from the same stable bytes."""
    resolved, snapshot, snapshot_sha256 = _read_stable_snapshot(
        path, "parent reranker artifact"
    )
    try:
        artifact = torch.load(io.BytesIO(snapshot), map_location="cpu")
    except Exception as error:
        raise ValueError("could not deserialize parent artifact: {}".format(error))
    model_config = _validate_parent_artifact(artifact)
    model = QueryReranker(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        dropout=float(model_config["dropout"]),
    )
    try:
        model.load_state_dict(artifact["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("parent artifact model state is incompatible: {}".format(
            error
        ))
    model.to(_resolve_device(device)).eval().requires_grad_(False)
    model._artifact_path = str(resolved)
    model._artifact_sha256 = snapshot_sha256
    model._artifact_parent_provenance = _parent_provenance(artifact)
    model._artifact_expected_state = {
        name: value.detach().cpu().clone()
        for name, value in artifact["model_state_dict"].items()
    }
    return model, artifact


def _tensor_sha256(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError("tensor digest input must be a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parent_provenance(parent_artifact):
    return {
        "artifact_version": parent_artifact["artifact_version"],
        "adapter_schema_version": parent_artifact["adapter_schema_version"],
        "input_dim": parent_artifact["input_dim"],
        "feature_names": copy.deepcopy(parent_artifact["feature_names"]),
        "model_config": copy.deepcopy(parent_artifact["model_config"]),
        "candidate_rule": copy.deepcopy(parent_artifact["candidate_rule"]),
        "checkpoint_sha256": parent_artifact["checkpoint_sha256"],
        "target_iou_policy": parent_artifact["target_iou_policy"],
        "model_inputs": copy.deepcopy(parent_artifact["model_inputs"]),
        "backbone_config": copy.deepcopy(parent_artifact["backbone_config"]),
        "score_mode": parent_artifact["score_mode"],
        "reranker_weight": float(parent_artifact["reranker_weight"]),
        "epoch": parent_artifact["epoch"],
        "feature_mean_sha256": _tensor_sha256(
            parent_artifact["feature_mean"]
        ),
        "feature_std_sha256": _tensor_sha256(
            parent_artifact["feature_std"]
        ),
    }


def _model_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            return float(module.p)
    return 0.0


def _cpu_state_dict(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _validate_manifest_provenance(base_manifest, geometry_manifest):
    if not isinstance(base_manifest, dict) or not isinstance(
            geometry_manifest, dict):
        raise ValueError("base and geometry manifests must be objects")
    if base_manifest.get("split") != "train" or geometry_manifest.get(
            "split") != "train":
        raise ValueError("geometry training artifacts require train manifests")
    base_sample_count = base_manifest.get("sample_count")
    geometry_sample_count = geometry_manifest.get("sample_count")
    if (not isinstance(base_sample_count, int)
            or isinstance(base_sample_count, bool)
            or base_sample_count <= 0
            or base_manifest.get("dataset_size") != base_sample_count
            or base_manifest.get("source_dataset_size") != base_sample_count
            or geometry_sample_count != base_sample_count
            or geometry_manifest.get("dataset_size") != base_sample_count
            or geometry_manifest.get("source_dataset_size")
            != base_sample_count):
        raise ValueError("train manifest sample counts are inconsistent")
    if (not _is_exact_int(
            base_manifest.get("cache_schema_version"), CACHE_SCHEMA_VERSION)
            or base_manifest.get("feature_schema_version")
            != FEATURE_SCHEMA_VERSION
            or base_manifest.get("feature_dim") != BASE_FEATURE_DIM
            or base_manifest.get("feature_names") is None
            or len(base_manifest["feature_names"]) != BASE_FEATURE_DIM):
        raise ValueError("base manifest feature schema is invalid")
    if (not _is_exact_int(
            geometry_manifest.get("geometry_cache_schema_version"),
            GEOMETRY_CACHE_SCHEMA_VERSION)
            or geometry_manifest.get("geometry_schema_version")
            != MASK_GEOMETRY_SCHEMA_VERSION
            or geometry_manifest.get("geometry_feature_names")
            != list(REC_MASK_GEOMETRY_FEATURE_NAMES)):
        raise ValueError("geometry manifest feature schema is invalid")
    expected_configs = [
        dict(config) for config in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
    ]
    if (geometry_manifest.get("variant_names")
            != [config["name"] for config in expected_configs]
            or geometry_manifest.get("variant_configs") != expected_configs
            or not _is_exact_int(
                geometry_manifest.get("regressed_variant_index"), 0)):
        raise ValueError("geometry manifest variants are invalid")
    if geometry_manifest.get("filter_non_gt_boxes") is not False:
        raise ValueError("geometry training requires filter_non_gt_boxes=False")
    min_points = geometry_manifest.get("min_points")
    max_fraction = geometry_manifest.get("max_point_fraction")
    if (not isinstance(min_points, int) or isinstance(min_points, bool)
            or min_points <= 0
            or not isinstance(max_fraction, (int, float))
            or isinstance(max_fraction, bool)
            or not math.isfinite(float(max_fraction))
            or not 0.0 < float(max_fraction) <= 1.0):
        raise ValueError("geometry manifest filters are invalid")
    matching_fields = (
        "candidate_rule",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "model_inputs",
        "backbone_config",
        "target_iou_policy",
    )
    if any(base_manifest.get(name) != geometry_manifest.get(name)
           for name in matching_fields):
        raise ValueError("base and geometry manifest provenance differs")
    candidate_rule = base_manifest.get("candidate_rule")
    if candidate_rule != {
            "topk_per_source": 8, "max_candidates": 16}:
        raise ValueError("training candidate rule is not authoritative")
    if (base_manifest.get("target_iou_policy") != TARGET_IOU_POLICY
            or base_manifest.get("checkpoint_epoch") is None
            or not _is_sha256(base_manifest.get("checkpoint_sha256"))):
        raise ValueError("training checkpoint or target policy is invalid")
    model_inputs = base_manifest.get("model_inputs")
    backbone = base_manifest.get("backbone_config")
    if (not isinstance(model_inputs, dict)
            or set(model_inputs) != set(MODEL_INPUT_KEYS)
            or any(not isinstance(model_inputs[key], bool)
                   for key in MODEL_INPUT_KEYS)
            or model_inputs["butd_gt"] or model_inputs["butd_cls"]
            or not _has_supported_backbone_config(backbone)
            or backbone.get("num_target") != DEPLOYED_QUERY_COUNT):
        raise ValueError("training model provenance is invalid")
    binding = geometry_manifest.get("base_cache_binding")
    if not isinstance(binding, dict):
        raise ValueError("geometry manifest base binding is missing")
    if binding.get("sample_count") != base_sample_count:
        raise ValueError("geometry base binding sample count is inconsistent")
    for key in ("content_sha256", "manifest_sha256"):
        if not _is_sha256(binding.get(key)):
            raise ValueError("geometry manifest base binding digest is invalid")
    for key in ("cache_content_digest", "immutable_metadata_digest"):
        if not _is_sha256(geometry_manifest.get(key)):
            raise ValueError("geometry manifest digest is invalid")
    return binding


def _validate_parent_against_manifests(parent, base_manifest):
    model, parent_artifact = _parent_parts(parent)
    _validate_live_parent_state(model, parent_artifact)
    parent_sha = getattr(model, "_artifact_sha256", None)
    if not _is_sha256(parent_sha):
        raise ValueError("parent model lacks a stable artifact SHA-256")
    expected = {
        "feature_names": base_manifest["feature_names"],
        "candidate_rule": base_manifest["candidate_rule"],
        "checkpoint_sha256": base_manifest["checkpoint_sha256"],
        "target_iou_policy": base_manifest["target_iou_policy"],
        "model_inputs": base_manifest["model_inputs"],
        "backbone_config": base_manifest["backbone_config"],
    }
    if any(parent_artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("parent artifact provenance differs from train cache")
    return model, parent_artifact, parent_sha


def _validate_scene_split(value):
    if not isinstance(value, dict) or set(value) != _SCENE_SPLIT_FIELDS:
        raise ValueError("scene split metadata fields do not match schema")
    for key in (
            "split_seed", "scene_count", "fit_scene_count",
            "calibration_scene_count", "sample_count", "fit_sample_count",
            "calibration_sample_count"):
        if (not isinstance(value[key], int) or isinstance(value[key], bool)
                or value[key] < 0):
            raise ValueError("scene split integer metadata is invalid")
    fraction = value["calibration_fraction"]
    if (not isinstance(fraction, float)
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) < 1.0):
        raise ValueError("scene split calibration fraction is invalid")
    if (value["fit_scene_count"] + value["calibration_scene_count"]
            != value["scene_count"]
            or value["fit_sample_count"] + value["calibration_sample_count"]
            != value["sample_count"]):
        raise ValueError("scene split counts are inconsistent")
    for key in (
            "fit_scene_sha256", "calibration_scene_sha256",
            "mapping_sha256"):
        if not _is_sha256(value[key]):
            raise ValueError("scene split digest is invalid")


def _validate_calibration_metrics(value):
    if not isinstance(value, dict) or set(value) != _CALIBRATION_METRIC_FIELDS:
        raise ValueError("calibration metric fields do not match schema")
    count = value["sample_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("calibration sample count is invalid")
    integer_fields = _CALIBRATION_METRIC_FIELDS - {
        "acc025", "acc050", "parent_acc025", "parent_acc050",
        "geometry_oracle_acc025", "geometry_oracle_acc050", "score",
    }
    for key in integer_fields:
        if (not isinstance(value[key], int) or isinstance(value[key], bool)
                or not 0 <= value[key] <= count):
            raise ValueError("calibration count {} is invalid".format(key))
    accuracy_pairs = (
        ("acc025", "hits025"),
        ("acc050", "hits050"),
        ("parent_acc025", "parent_hits025"),
        ("parent_acc050", "parent_hits050"),
        ("geometry_oracle_acc025", "geometry_oracle_hits025"),
        ("geometry_oracle_acc050", "geometry_oracle_hits050"),
    )
    for accuracy_key, count_key in accuracy_pairs:
        accuracy = value[accuracy_key]
        if (not isinstance(accuracy, (float, int))
                or isinstance(accuracy, bool)
                or not math.isfinite(float(accuracy))
                or float(accuracy) != value[count_key] / float(count)):
            raise ValueError("calibration accuracy {} is invalid".format(
                accuracy_key
            ))
    for suffix in ("025", "050"):
        if (value["hits" + suffix] - value["parent_hits" + suffix]
                != value["fixes" + suffix] - value["breaks" + suffix]):
            raise ValueError("calibration fix/break counts are inconsistent")
        both_hits = value["hits" + suffix] - value["fixes" + suffix]
        neither_hits = (
            count
            - both_hits
            - value["fixes" + suffix]
            - value["breaks" + suffix]
        )
        if both_hits < 0 or neither_hits < 0:
            raise ValueError("calibration fix/break contingency is impossible")
        if (value["geometry_oracle_hits" + suffix]
                < max(value["hits" + suffix],
                      value["parent_hits" + suffix])):
            raise ValueError("calibration geometry oracle is inconsistent")
        if value["hits" + suffix] < value["parent_hits" + suffix]:
            raise ValueError("selected calibration metrics regress parent")
    for prefix in ("hits", "parent_hits", "geometry_oracle_hits"):
        if value[prefix + "050"] > value[prefix + "025"]:
            raise ValueError("calibration thresholds are not monotonic")
    expected_score = calibration_score(value["acc025"], value["acc050"])
    if (not isinstance(value["score"], (float, int))
            or isinstance(value["score"], bool)
            or float(value["score"]) != expected_score):
        raise ValueError("calibration score is inconsistent")


def _validate_training_args(value, scene_split, model_config):
    if not isinstance(value, dict) or set(value) != _TRAINING_ARGUMENT_FIELDS:
        raise ValueError("training argument fields do not match schema")
    positive_ints = (
        "hidden_dim", "batch_size", "max_epochs", "patience"
    )
    for key in positive_ints:
        if (not isinstance(value[key], int) or isinstance(value[key], bool)
                or value[key] <= 0):
            raise ValueError("training argument {} is invalid".format(key))
    for key in ("split_seed", "model_seed"):
        if not isinstance(value[key], int) or isinstance(value[key], bool):
            raise ValueError("training seed is invalid")
    dropout = value["dropout"]
    if (not isinstance(dropout, float)
            or not math.isfinite(float(dropout))
            or not 0.0 <= float(dropout) < 1.0):
        raise ValueError("training dropout is invalid")
    if (not isinstance(value["lr"], (float, int))
            or isinstance(value["lr"], bool)
            or not math.isfinite(float(value["lr"]))
            or float(value["lr"]) <= 0.0
            or not isinstance(value["weight_decay"], (float, int))
            or isinstance(value["weight_decay"], bool)
            or not math.isfinite(float(value["weight_decay"]))
            or float(value["weight_decay"]) < 0.0):
        raise ValueError("training optimizer arguments are invalid")
    grid = value["geometry_weight_grid"]
    if (not isinstance(value["device"], str) or not value["device"]
            or not isinstance(value["parent_allow_tf32"], bool)
            or not isinstance(value["grad_clip_norm"], (float, int))
            or isinstance(value["grad_clip_norm"], bool)
            or not math.isfinite(float(value["grad_clip_norm"]))
            or float(value["grad_clip_norm"]) != GRAD_CLIP_NORM
            or not isinstance(grid, list)
            or any(not isinstance(weight, float)
                   or not math.isfinite(float(weight)) for weight in grid)
            or tuple(grid) != DEFAULT_GEOMETRY_WEIGHTS):
        raise ValueError("training runtime arguments are invalid")
    if (not isinstance(value["calibration_fraction"], float)
            or int(value["split_seed"]) != scene_split["split_seed"]
            or value["calibration_fraction"]
            != float(scene_split["calibration_fraction"])
            or value["hidden_dim"] != model_config["hidden_dim"]
            or float(value["dropout"]) != float(model_config["dropout"])):
        raise ValueError("training arguments differ from artifact metadata")


def _validate_parent_inference_contract(
        value, content_sha256, scene_split, training_args):
    if (not isinstance(value, dict)
            or set(value) != set(PARENT_INFERENCE_CONTRACT_FIELDS)):
        raise ValueError("parent inference contract fields do not match schema")
    expected = {
        "schema": PARENT_INFERENCE_SCHEMA,
        "row_order": "dataset-index-contiguous",
        "remainder_policy": "natural-remainder",
        "feature_source": "bound-base-cache-features",
        "dtype": "float32",
        "autocast": False,
        "eval": True,
        "no_grad": True,
        "score_builder": PARENT_SCORE_BUILDER,
        "canonical_query_tie_policy": PARENT_QUERY_TIE_POLICY,
        "content_digest_version": PARENT_SCORE_CONTENT_DIGEST_VERSION,
    }
    if any(value.get(key) != expected_value
           for key, expected_value in expected.items()):
        raise ValueError("parent inference contract policy is invalid")
    boolean_contract = {
        "autocast": False,
        "eval": True,
        "no_grad": True,
    }
    if any(not isinstance(value.get(key), bool)
           or value[key] is not expected_value
           for key, expected_value in boolean_contract.items()):
        raise ValueError("parent inference contract execution mode is invalid")
    integer_contract = {
        "version": PARENT_INFERENCE_VERSION,
        "local_batch_size": PARENT_INFERENCE_LOCAL_BATCH_SIZE,
        "world_size": 1,
        "score_builder_version": PARENT_SCORE_BUILDER_VERSION,
    }
    if any(not _is_exact_int(value.get(key), expected_value)
           for key, expected_value in integer_contract.items()):
        raise ValueError("parent inference contract version is invalid")
    if (not isinstance(value.get("allow_tf32"), bool)
            or not _is_sha256(content_sha256)
            or value.get("score_content_sha256") != content_sha256):
        raise ValueError("parent inference contract score binding is invalid")
    device_type = value.get("device_type")
    device_index = value.get("device_index")
    if (device_type not in {"cpu", "cuda"}
            or (device_type == "cpu" and device_index is not None)
            or (device_type == "cpu" and value["allow_tf32"] is not False)
            or (device_type == "cuda"
                and (not isinstance(device_index, int)
                     or isinstance(device_index, bool)
                     or device_index < 0))):
        raise ValueError("parent inference contract device is invalid")
    row_count = value.get("row_count")
    if (not isinstance(row_count, int) or isinstance(row_count, bool)
            or row_count <= 0 or row_count != scene_split["sample_count"]):
        raise ValueError("parent inference contract sample count is invalid")
    try:
        training_device = torch.device(training_args["device"])
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError("parent inference training device is invalid: {}".format(
            error
        ))
    if (training_device.type != device_type
            or training_device.index != device_index):
        raise ValueError("parent inference contract differs from training device")
    if training_args["parent_allow_tf32"] != value["allow_tf32"]:
        raise ValueError("parent inference TF32 differs from training arguments")


def _validate_authoritative_parent_inference(
        base_content_sha256, device_type, device_index, allow_tf32):
    if base_content_sha256 != AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256:
        return
    if (device_type != "cuda" or device_index != 0
            or allow_tf32 is not True):
        raise ValueError(
            "authoritative parent inference requires cuda:0 with TF32 enabled"
        )


def _artifact_metrics(metrics):
    if not isinstance(metrics, dict):
        raise ValueError("calibration metrics must be an object")
    try:
        result = {
            key: copy.deepcopy(metrics[key])
            for key in CALIBRATION_METRIC_FIELDS
        }
    except KeyError as error:
        raise ValueError("calibration metrics are missing {}".format(error))
    _validate_calibration_metrics(result)
    return result


def build_geometry_artifact(
        model, feature_mean, feature_std, parent, base_manifest,
        geometry_manifest, scene_split, epoch, calibration_metrics,
        training_args, geometry_weight):
    """Build the exact train-only geometry artifact payload."""
    if not isinstance(model, QueryReranker) or model.input_dim != GEOMETRY_INPUT_DIM:
        raise ValueError("geometry model must be a 179D QueryReranker")
    _validate_normalization(feature_mean, feature_std)
    binding = _validate_manifest_provenance(base_manifest, geometry_manifest)
    _, parent_artifact, parent_sha = _validate_parent_against_manifests(
        parent, base_manifest
    )
    parent_inference_contract, parent_score_content_sha256 = (
        _sealed_parent_materialization_metadata(parent)
    )
    _validate_scene_split(scene_split)
    if parent_inference_contract["row_count"] != scene_split["sample_count"]:
        raise ValueError("parent materialization row count differs from train split")
    _strict_int(epoch, "geometry artifact epoch", minimum=1)
    if (not isinstance(geometry_weight, (float, int))
            or isinstance(geometry_weight, bool)
            or not math.isfinite(float(geometry_weight))):
        raise ValueError("geometry weight is invalid")
    weight = float(geometry_weight)
    if weight not in DEFAULT_GEOMETRY_WEIGHTS:
        raise ValueError("geometry weight is outside the declared grid")
    metrics = _artifact_metrics(calibration_metrics)
    model_config = {
        "input_dim": GEOMETRY_INPUT_DIM,
        "hidden_dim": int(model.hidden_dim),
        "dropout": _model_dropout(model),
    }
    training_args = copy.deepcopy(dict(training_args))
    _validate_training_args(training_args, scene_split, model_config)
    feature_names = (
        list(base_manifest["feature_names"])
        + list(geometry_manifest["geometry_feature_names"])
        + ["parent_score", "parent_is_deployed_top1"]
    )
    artifact = {
        "artifact_version": GEOMETRY_ARTIFACT_VERSION,
        "model_state_dict": _cpu_state_dict(model),
        "model_config": model_config,
        "model_schema_version": REC_GEOMETRY_MODEL_SCHEMA_VERSION,
        "geometry_cache_schema_version": GEOMETRY_CACHE_SCHEMA_VERSION,
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "base_cache_schema_version": CACHE_SCHEMA_VERSION,
        "base_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_dim": GEOMETRY_INPUT_DIM,
        "feature_names": feature_names,
        "feature_mean": feature_mean.detach().cpu().float().clone(),
        "feature_std": feature_std.detach().cpu().float().clone(),
        "variant_names": copy.deepcopy(geometry_manifest["variant_names"]),
        "variant_configs": copy.deepcopy(geometry_manifest["variant_configs"]),
        "regressed_variant_index": geometry_manifest[
            "regressed_variant_index"
        ],
        "min_points": geometry_manifest["min_points"],
        "max_point_fraction": float(
            geometry_manifest["max_point_fraction"]
        ),
        "candidate_rule": copy.deepcopy(base_manifest["candidate_rule"]),
        "checkpoint_sha256": base_manifest["checkpoint_sha256"],
        "checkpoint_epoch": base_manifest["checkpoint_epoch"],
        "model_inputs": copy.deepcopy(base_manifest["model_inputs"]),
        "backbone_config": copy.deepcopy(base_manifest["backbone_config"]),
        "parent_artifact_sha256": parent_sha,
        "parent_provenance": _parent_provenance(parent_artifact),
        "parent_inference_contract": parent_inference_contract,
        "num_queries": DEPLOYED_QUERY_COUNT,
        "flat_parent_prior_version": FLAT_PARENT_PRIOR_VERSION,
        "tie_policy": GEOMETRY_TIE_POLICY,
        "score_mode": GEOMETRY_SCORE_MODE,
        "geometry_weight": weight,
        "target_iou_policy": TARGET_IOU_POLICY,
        "evaluator_filter_policy": EVALUATOR_FILTER_POLICY,
        "filter_non_gt_boxes": False,
        "train_parent_score_content_sha256": parent_score_content_sha256,
        "train_base_cache_content_digest": binding["content_sha256"],
        "train_base_cache_manifest_digest": binding["manifest_sha256"],
        "train_geometry_cache_content_digest": geometry_manifest[
            "cache_content_digest"
        ],
        "train_geometry_immutable_metadata_digest": geometry_manifest[
            "immutable_metadata_digest"
        ],
        "scene_split": copy.deepcopy(scene_split),
        "epoch": int(epoch),
        "calibration_metrics": metrics,
        "training_args": training_args,
    }
    validate_geometry_artifact(
        artifact,
        parent=parent,
        base_manifest=base_manifest,
        geometry_manifest=geometry_manifest,
        scene_split=scene_split,
    )
    return artifact


def validate_geometry_artifact(artifact, parent=None, base_manifest=None,
                               geometry_manifest=None, scene_split=None):
    """Validate exact artifact structure and optional live provenance."""
    if not isinstance(artifact, dict):
        raise ValueError("geometry artifact must contain an object")
    if set(artifact) != set(GEOMETRY_ARTIFACT_FIELDS):
        raise ValueError("geometry artifact fields do not match exact schema")
    expected_versions = {
        "artifact_version": GEOMETRY_ARTIFACT_VERSION,
        "geometry_cache_schema_version": GEOMETRY_CACHE_SCHEMA_VERSION,
        "base_cache_schema_version": CACHE_SCHEMA_VERSION,
    }
    if (any(not _is_exact_int(artifact.get(key), value)
            for key, value in expected_versions.items())
            or artifact.get("model_schema_version")
            != REC_GEOMETRY_MODEL_SCHEMA_VERSION
            or artifact.get("geometry_schema_version")
            != MASK_GEOMETRY_SCHEMA_VERSION
            or artifact.get("base_feature_schema_version")
            != FEATURE_SCHEMA_VERSION):
        raise ValueError("geometry artifact schema version mismatch")
    model_config = artifact.get("model_config")
    if (not isinstance(model_config, dict)
            or set(model_config) != {"input_dim", "hidden_dim", "dropout"}
            or model_config.get("input_dim") != GEOMETRY_INPUT_DIM
            or not isinstance(model_config.get("hidden_dim"), int)
            or isinstance(model_config.get("hidden_dim"), bool)
            or model_config["hidden_dim"] <= 0
            or not isinstance(model_config.get("dropout"), (float, int))
            or isinstance(model_config.get("dropout"), bool)
            or not 0.0 <= float(model_config["dropout"]) < 1.0
            or artifact.get("input_dim") != GEOMETRY_INPUT_DIM):
        raise ValueError("geometry artifact model config is invalid")
    provenance = artifact.get("parent_provenance")
    parent_model_config = (
        provenance.get("model_config")
        if isinstance(provenance, dict) else None
    )
    if (not isinstance(provenance, dict)
            or set(provenance) != _PARENT_PROVENANCE_FIELDS
            or not _is_exact_int(
                provenance.get("artifact_version"), PARENT_ARTIFACT_VERSION)
            or provenance.get("adapter_schema_version")
            != FEATURE_SCHEMA_VERSION
            or provenance.get("input_dim") != BASE_FEATURE_DIM
            or not isinstance(provenance.get("feature_names"), list)
            or len(provenance["feature_names"]) != BASE_FEATURE_DIM
            or len(set(provenance["feature_names"])) != BASE_FEATURE_DIM
            or any(not isinstance(name, str) or not name
                   for name in provenance["feature_names"])
            or not isinstance(parent_model_config, dict)
            or set(parent_model_config)
            != {"input_dim", "hidden_dim", "dropout"}
            or parent_model_config.get("input_dim") != BASE_FEATURE_DIM
            or not isinstance(parent_model_config.get("hidden_dim"), int)
            or isinstance(parent_model_config.get("hidden_dim"), bool)
            or parent_model_config["hidden_dim"] <= 0
            or not isinstance(parent_model_config.get("dropout"), (float, int))
            or isinstance(parent_model_config.get("dropout"), bool)
            or not 0.0 <= float(parent_model_config["dropout"]) < 1.0
            or provenance.get("candidate_rule")
            != {"topk_per_source": 8, "max_candidates": 16}
            or provenance.get("target_iou_policy") != TARGET_IOU_POLICY
            or provenance.get("score_mode") != "rank_blend"
            or not isinstance(provenance.get("reranker_weight"), (float, int))
            or isinstance(provenance.get("reranker_weight"), bool)
            or not 0.0 <= float(provenance["reranker_weight"]) <= 1.0
            or not isinstance(provenance.get("epoch"), int)
            or isinstance(provenance.get("epoch"), bool)
            or provenance["epoch"] < 0
            or any(not _is_sha256(provenance.get(key)) for key in (
                "checkpoint_sha256", "feature_mean_sha256",
                "feature_std_sha256"))):
        raise ValueError("geometry artifact parent provenance is invalid")
    expected_names = (
        provenance["feature_names"]
        + list(REC_MASK_GEOMETRY_FEATURE_NAMES)
        + ["parent_score", "parent_is_deployed_top1"]
    )
    if (artifact.get("feature_names") != expected_names
            or len(expected_names) != GEOMETRY_INPUT_DIM
            or len(set(expected_names)) != GEOMETRY_INPUT_DIM):
        raise ValueError("geometry artifact feature names are invalid")
    _validate_normalization(
        artifact.get("feature_mean"), artifact.get("feature_std")
    )
    if (artifact["feature_mean"].dtype != torch.float32
            or artifact["feature_std"].dtype != torch.float32
            or artifact["feature_mean"].device.type != "cpu"
            or artifact["feature_std"].device.type != "cpu"):
        raise ValueError("geometry artifact normalization storage is invalid")
    configs = [dict(config) for config in DEFAULT_REC_MASK_GEOMETRY_VARIANTS]
    if (artifact.get("variant_names")
            != [config["name"] for config in configs]
            or artifact.get("variant_configs") != configs
            or not _is_exact_int(
                artifact.get("regressed_variant_index"), 0)):
        raise ValueError("geometry artifact variant schema is invalid")
    min_points = artifact.get("min_points")
    max_fraction = artifact.get("max_point_fraction")
    if (not isinstance(min_points, int) or isinstance(min_points, bool)
            or min_points <= 0
            or not isinstance(max_fraction, (float, int))
            or isinstance(max_fraction, bool)
            or not 0.0 < float(max_fraction) <= 1.0):
        raise ValueError("geometry artifact filters are invalid")
    if artifact.get("candidate_rule") != {
            "topk_per_source": 8, "max_candidates": 16}:
        raise ValueError("geometry artifact candidate rule is invalid")
    if (not _is_sha256(artifact.get("checkpoint_sha256"))
            or not isinstance(artifact.get("checkpoint_epoch"), int)
            or isinstance(artifact.get("checkpoint_epoch"), bool)
            or artifact["checkpoint_epoch"] < 0
            or artifact.get("num_queries") != DEPLOYED_QUERY_COUNT):
        raise ValueError("geometry artifact checkpoint provenance is invalid")
    if (artifact.get("model_inputs") != provenance.get("model_inputs")
            or artifact.get("backbone_config")
            != provenance.get("backbone_config")
            or artifact.get("candidate_rule")
            != provenance.get("candidate_rule")
            or artifact.get("checkpoint_sha256")
            != provenance.get("checkpoint_sha256")
            or artifact.get("target_iou_policy")
            != provenance.get("target_iou_policy")):
        raise ValueError("geometry and parent artifact provenance differs")
    if (not isinstance(artifact["model_inputs"], dict)
            or set(artifact["model_inputs"]) != set(MODEL_INPUT_KEYS)
            or artifact["model_inputs"]["butd_gt"]
            or artifact["model_inputs"]["butd_cls"]
            or not _has_supported_backbone_config(
                artifact["backbone_config"]
            )
            or artifact["backbone_config"]["num_target"]
            != DEPLOYED_QUERY_COUNT):
        raise ValueError("geometry artifact model provenance is invalid")
    geometry_weight = artifact.get("geometry_weight")
    if (not _is_sha256(artifact.get("parent_artifact_sha256"))
            or artifact.get("flat_parent_prior_version")
            != FLAT_PARENT_PRIOR_VERSION
            or artifact.get("tie_policy") != GEOMETRY_TIE_POLICY
            or artifact.get("score_mode") != GEOMETRY_SCORE_MODE
            or not isinstance(geometry_weight, float)
            or not math.isfinite(geometry_weight)
            or geometry_weight not in DEFAULT_GEOMETRY_WEIGHTS
            or artifact.get("target_iou_policy") != TARGET_IOU_POLICY
            or artifact.get("evaluator_filter_policy")
            != EVALUATOR_FILTER_POLICY
            or artifact.get("filter_non_gt_boxes") is not False):
        raise ValueError("geometry artifact score or evaluator policy is invalid")
    digest_fields = (
        "train_base_cache_content_digest",
        "train_base_cache_manifest_digest",
        "train_geometry_cache_content_digest",
        "train_geometry_immutable_metadata_digest",
    )
    if any(not _is_sha256(artifact.get(key)) for key in digest_fields):
        raise ValueError("geometry artifact train cache digest is invalid")
    _validate_scene_split(artifact.get("scene_split"))
    _strict_int(
        artifact.get("epoch"), "geometry artifact epoch", minimum=1
    )
    _validate_calibration_metrics(artifact.get("calibration_metrics"))
    if (artifact["calibration_metrics"]["sample_count"]
            != artifact["scene_split"]["calibration_sample_count"]):
        raise ValueError(
            "calibration metrics do not match scene split sample count"
        )
    if geometry_weight == 0.0:
        metrics = artifact["calibration_metrics"]
        if (metrics["hits025"] != metrics["parent_hits025"]
                or metrics["hits050"] != metrics["parent_hits050"]
                or any(metrics[key] != 0 for key in (
                    "fixes025", "breaks025", "fixes050", "breaks050"
                ))):
            raise ValueError("zero-weight calibration is not exact parent path")
    _validate_training_args(
        artifact.get("training_args"), artifact["scene_split"], model_config
    )
    _validate_parent_inference_contract(
        artifact.get("parent_inference_contract"),
        artifact.get("train_parent_score_content_sha256"),
        artifact["scene_split"],
        artifact["training_args"],
    )
    parent_inference_contract = artifact["parent_inference_contract"]
    _validate_authoritative_parent_inference(
        artifact["train_base_cache_content_digest"],
        parent_inference_contract["device_type"],
        parent_inference_contract["device_index"],
        parent_inference_contract["allow_tf32"],
    )
    model_state = artifact.get("model_state_dict")
    probe_model = QueryReranker(**model_config)
    expected_state = probe_model.state_dict()
    if (not isinstance(model_state, dict) or not model_state
            or set(model_state) != set(expected_state)
            or any(not isinstance(name, str) or not name
                   or not isinstance(value, torch.Tensor)
                   or value.device.type != "cpu"
                   or value.dtype != expected_state[name].dtype
                   or tuple(value.shape) != tuple(expected_state[name].shape)
                   or not bool(torch.isfinite(value).all().item())
                   for name, value in model_state.items())):
        raise ValueError("geometry artifact model state is invalid")
    try:
        probe_model.load_state_dict(model_state, strict=True)
    except (RuntimeError, TypeError) as error:
            raise ValueError("geometry artifact model state is incompatible: {}".format(
            error
        ))
    if any(not bool(torch.isfinite(value).all().item())
           for value in probe_model.state_dict().values()):
        raise ValueError("geometry artifact loaded model state is non-finite")
    if (artifact["train_base_cache_content_digest"]
            == AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256):
        if (artifact["parent_artifact_sha256"]
                != AUTHORITATIVE_PARENT_ARTIFACT_SHA256
                or artifact["scene_split"] != AUTHORITATIVE_SPLIT_SEED0):
            raise ValueError("authoritative train artifact binding mismatch")

    if parent is not None:
        parent_model, parent_artifact = _parent_parts(parent)
        if (getattr(parent_model, "_artifact_sha256", None)
                != artifact["parent_artifact_sha256"]):
            raise ValueError("geometry artifact parent artifact SHA mismatch")
        if _parent_provenance(parent_artifact) != provenance:
            raise ValueError("geometry artifact parent structural mismatch")
        cache_state = getattr(
            parent_model, "_geometry_parent_score_cache", None
        )
        if cache_state is not None:
            live_contract, live_digest = _sealed_parent_materialization_metadata(
                parent
            )
            if (live_contract != artifact["parent_inference_contract"]
                    or live_digest
                    != artifact["train_parent_score_content_sha256"]):
                raise ValueError(
                    "geometry artifact parent materialization mismatch"
                )
    if base_manifest is not None or geometry_manifest is not None:
        if base_manifest is None or geometry_manifest is None:
            raise ValueError("both train manifests are required for validation")
        binding = _validate_manifest_provenance(
            base_manifest, geometry_manifest
        )
        if artifact["scene_split"]["sample_count"] != base_manifest[
                "sample_count"]:
            raise ValueError(
                "geometry artifact scene sample count differs from train cache"
            )
        expected_context = {
            "train_base_cache_content_digest": binding["content_sha256"],
            "train_base_cache_manifest_digest": binding["manifest_sha256"],
            "train_geometry_cache_content_digest": geometry_manifest[
                "cache_content_digest"
            ],
            "train_geometry_immutable_metadata_digest": geometry_manifest[
                "immutable_metadata_digest"
            ],
            "feature_names": (
                list(base_manifest["feature_names"])
                + list(geometry_manifest["geometry_feature_names"])
                + ["parent_score", "parent_is_deployed_top1"]
            ),
            "variant_names": geometry_manifest["variant_names"],
            "variant_configs": geometry_manifest["variant_configs"],
            "regressed_variant_index": geometry_manifest[
                "regressed_variant_index"
            ],
            "min_points": geometry_manifest["min_points"],
            "max_point_fraction": float(
                geometry_manifest["max_point_fraction"]
            ),
            "candidate_rule": base_manifest["candidate_rule"],
            "checkpoint_sha256": base_manifest["checkpoint_sha256"],
            "checkpoint_epoch": base_manifest["checkpoint_epoch"],
            "model_inputs": base_manifest["model_inputs"],
            "backbone_config": base_manifest["backbone_config"],
        }
        if any(artifact[key] != value for key, value in expected_context.items()):
            raise ValueError("geometry artifact train manifest mismatch")
    if scene_split is not None:
        _validate_scene_split(scene_split)
        if artifact["scene_split"] != scene_split:
            raise ValueError("geometry artifact scene split mismatch")
    return model_config


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
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _output_path_without_following_final(path):
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    return absolute.parent.resolve() / absolute.name


def save_geometry_reranker_artifact(path, artifact):
    """Validate, atomically fsync, and strict-reload one geometry artifact."""
    validate_geometry_artifact(artifact)
    output = _output_path_without_following_final(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError("geometry artifact output must not be a symlink")
    if output.exists() and not output.is_file():
        raise ValueError("geometry artifact output must be a regular file")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".tmp.", dir=str(output.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(artifact, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _, reloaded = load_geometry_reranker_artifact(
            temporary, device="cpu"
        )
        if not _payloads_equal(artifact, reloaded):
            raise RuntimeError(
                "strict geometry artifact reload changed its payload"
            )
        os.replace(str(temporary), str(output))
        _fsync_directory(output.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return artifact


def load_geometry_reranker_artifact(
        path, device="cpu", parent_artifact_path=None,
        base_manifest=None, geometry_manifest=None):
    """Stable-load a validated frozen geometry scorer artifact."""
    resolved, snapshot, snapshot_sha256 = _read_stable_snapshot(
        path, "geometry reranker artifact"
    )
    try:
        artifact = torch.load(io.BytesIO(snapshot), map_location="cpu")
    except Exception as error:
        raise ValueError("could not deserialize geometry artifact: {}".format(
            error
        ))
    parent = None
    if parent_artifact_path is not None:
        parent = load_parent_reranker_snapshot(parent_artifact_path, device="cpu")
    model_config = validate_geometry_artifact(
        artifact,
        parent=parent,
        base_manifest=base_manifest,
        geometry_manifest=geometry_manifest,
    )
    model = QueryReranker(**model_config)
    try:
        model.load_state_dict(artifact["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("geometry artifact model state is incompatible: {}".format(
            error
        ))
    model.to(_resolve_device(device)).eval().requires_grad_(False)
    model._artifact_path = str(resolved)
    model._artifact_sha256 = snapshot_sha256
    return model, artifact


def load_geometry_training_data(base_cache, geometry_cache,
                                parent_artifact_path):
    """Strict-load and join the three immutable train-only inputs."""
    base_path = Path(base_cache).expanduser().resolve()
    base_rows, base_manifest, base_binding = load_bound_candidate_cache(
        base_path, "train"
    )
    geometry_rows, geometry_manifest = load_geometry_cache(
        geometry_cache, "train"
    )
    geometry_binding = geometry_manifest.get("base_cache_binding")
    if not isinstance(geometry_binding, dict):
        raise ValueError("geometry cache lacks its bound base cache")
    if geometry_binding.get("path") != str(base_path):
        raise ValueError("geometry cache is bound to a different base path")
    if geometry_binding != base_binding:
        raise ValueError("geometry cache base binding differs from live cache")
    _validate_manifest_provenance(base_manifest, geometry_manifest)
    joined = join_base_and_geometry_rows(
        base_rows, geometry_rows, base_manifest, geometry_manifest
    )
    parent = load_parent_reranker_snapshot(parent_artifact_path, device="cpu")
    _, _, parent_sha = _validate_parent_against_manifests(
        parent, base_manifest
    )
    if (base_binding["content_sha256"]
            == AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256
            and parent_sha != AUTHORITATIVE_PARENT_ARTIFACT_SHA256):
        raise ValueError("authoritative training requires the frozen parent SHA")
    return joined, base_manifest, geometry_manifest, parent


def _set_deterministic_seed(seed, device):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _training_batch(rows, parent, feature_mean, feature_std, device):
    batch = build_geometry_training_batch(rows, parent)
    valid = batch["valid_mask"]
    normalized = normalize_features(
        batch["features"], valid, feature_mean, feature_std
    )
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("normalized geometry training features are non-finite")
    return {
        "features": normalized.to(
            device, non_blocking=(device.type == "cuda")
        ),
        "valid_mask": valid.to(
            device, non_blocking=(device.type == "cuda")
        ),
        "candidate_ious": batch["candidate_ious"].to(
            device, non_blocking=(device.type == "cuda")
        ),
    }


def _chosen_artifact_metrics(metrics_by_weight, weight):
    return _artifact_metrics(metrics_by_weight[weight])


def fit_and_save_geometry_model(
        model, fit_rows, calibration_rows, feature_mean, feature_std,
        parent, base_manifest, geometry_manifest, output, split_seed=0,
        model_seed=0, calibration_fraction=DEFAULT_CALIBRATION_FRACTION,
        lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=100,
        patience=10, device="cuda:0", verbose=False,
        protected_output_directories=()):
    """Fit, select, atomically save, and strict-reproduce one scorer."""
    parent_path = (
        getattr(parent[0], "_artifact_path", None)
        if isinstance(parent, (list, tuple)) and parent else None
    )
    protected = tuple(protected_output_directories)
    if isinstance(parent_path, str) and parent_path:
        binding = (
            geometry_manifest.get("base_cache_binding")
            if isinstance(geometry_manifest, dict) else None
        )
        if (isinstance(binding, dict)
                and isinstance(binding.get("path"), str)):
            protected = protected + (binding["path"],)
        _validate_training_output_location(
            output, parent_path, protected_directories=protected
        )
    if not isinstance(fit_rows, (list, tuple)) or not fit_rows:
        raise ValueError("geometry fit rows cannot be empty")
    if not isinstance(calibration_rows, (list, tuple)) or not calibration_rows:
        raise ValueError("geometry calibration rows cannot be empty")
    if not isinstance(model, QueryReranker) or model.input_dim != GEOMETRY_INPUT_DIM:
        raise ValueError("geometry model must be a 179D QueryReranker")
    _validate_normalization(feature_mean, feature_std)
    for name, value in (
            ("batch_size", batch_size),
            ("max_epochs", max_epochs),
            ("patience", patience)):
        _strict_int(value, name, minimum=1)
    if (not isinstance(lr, (float, int)) or isinstance(lr, bool)
            or not math.isfinite(float(lr))
            or float(lr) <= 0.0
            or not isinstance(weight_decay, (float, int))
            or isinstance(weight_decay, bool)
            or not math.isfinite(float(weight_decay))
            or float(weight_decay) < 0.0):
        raise ValueError("geometry optimizer arguments are invalid")
    resolved_device = _resolve_device(device)
    _set_deterministic_seed(model_seed, resolved_device)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    best_score = -float("inf")
    best_epoch = None
    best_state = None
    best_metrics = None
    best_weight = None
    stale_epochs = 0

    for epoch_index in range(int(max_epochs)):
        model.train()
        indices = list(range(len(fit_rows)))
        random.Random(int(model_seed) + epoch_index).shuffle(indices)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start:start + int(batch_size)]
            batch_rows = [fit_rows[index] for index in batch_indices]
            batch = _training_batch(
                batch_rows,
                parent,
                feature_mean,
                feature_std,
                resolved_device,
            )
            optimizer.zero_grad()
            outputs = model(batch["features"], batch["valid_mask"])
            loss, _ = compute_rec_reranker_loss(
                outputs,
                batch["candidate_ious"],
                batch["valid_mask"],
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            current_count = int(batch["features"].shape[0])
            loss_sum += float(loss.detach().item()) * current_count
            seen += current_count

        metrics_by_weight = evaluate_geometry_blends(
            model,
            calibration_rows,
            feature_mean,
            feature_std,
            parent,
            geometry_weights=DEFAULT_GEOMETRY_WEIGHTS,
            batch_size=batch_size,
            device=resolved_device,
            regressed_variant_index=geometry_manifest[
                "regressed_variant_index"
            ],
        )
        selected_weight, selected_metrics = choose_best_geometry_blend(
            metrics_by_weight
        )
        score = calibration_score(
            selected_metrics["acc025"], selected_metrics["acc050"]
        )
        epoch = epoch_index + 1
        if verbose:
            print(
                "Epoch {:03d} loss={:.6f} calibration "
                "Acc@0.25={:.5f} Acc@0.50={:.5f} weight={:.2f} "
                "score={:.6f}".format(
                    epoch,
                    loss_sum / float(max(seen, 1)),
                    selected_metrics["acc025"],
                    selected_metrics["acc050"],
                    selected_weight,
                    score,
                ),
                flush=True,
            )
        # Strict improvement preserves both the lower-weight tie decision
        # within an epoch and the earliest epoch across equal scores.
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            best_metrics = _chosen_artifact_metrics(
                metrics_by_weight, selected_weight
            )
            best_weight = selected_weight
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    if (best_state is None or best_metrics is None or best_weight is None
            or best_epoch is None):
        raise RuntimeError("geometry training did not produce a best state")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    restored_by_weight = evaluate_geometry_blends(
        model,
        calibration_rows,
        feature_mean,
        feature_std,
        parent,
        geometry_weights=DEFAULT_GEOMETRY_WEIGHTS,
        batch_size=batch_size,
        device=resolved_device,
        regressed_variant_index=geometry_manifest["regressed_variant_index"],
    )
    restored_metrics = _chosen_artifact_metrics(
        restored_by_weight, best_weight
    )
    if restored_metrics != best_metrics:
        raise RuntimeError("restored best geometry state changed calibration")

    scene_split = build_scene_split_metadata(
        fit_rows,
        calibration_rows,
        split_seed=split_seed,
        calibration_fraction=calibration_fraction,
    )
    training_args = {
        "split_seed": int(split_seed),
        "model_seed": int(model_seed),
        "calibration_fraction": float(calibration_fraction),
        "hidden_dim": int(model.hidden_dim),
        "dropout": _model_dropout(model),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "device": str(resolved_device),
        "parent_allow_tf32": _sealed_parent_materialization_metadata(
            parent
        )[0]["allow_tf32"],
        "grad_clip_norm": GRAD_CLIP_NORM,
        "geometry_weight_grid": list(DEFAULT_GEOMETRY_WEIGHTS),
    }
    artifact = build_geometry_artifact(
        model,
        feature_mean,
        feature_std,
        parent,
        base_manifest,
        geometry_manifest,
        scene_split,
        epoch=best_epoch,
        calibration_metrics=best_metrics,
        training_args=training_args,
        geometry_weight=best_weight,
    )
    _validate_training_output_location(
        output, parent_path, protected_directories=protected
    )
    save_geometry_reranker_artifact(output, artifact)

    parent_path = getattr(parent[0], "_artifact_path", None)
    if not isinstance(parent_path, str) or not parent_path:
        raise RuntimeError("frozen parent snapshot path was not retained")
    reloaded_model, reloaded_artifact = load_geometry_reranker_artifact(
        output,
        device=resolved_device,
        parent_artifact_path=parent_path,
        base_manifest=base_manifest,
        geometry_manifest=geometry_manifest,
    )
    reloaded_by_weight = evaluate_geometry_blends(
        reloaded_model,
        calibration_rows,
        reloaded_artifact["feature_mean"],
        reloaded_artifact["feature_std"],
        parent,
        geometry_weights=DEFAULT_GEOMETRY_WEIGHTS,
        batch_size=batch_size,
        device=resolved_device,
        regressed_variant_index=reloaded_artifact[
            "regressed_variant_index"
        ],
    )
    reloaded_metrics = _chosen_artifact_metrics(
        reloaded_by_weight, reloaded_artifact["geometry_weight"]
    )
    if reloaded_metrics != reloaded_artifact["calibration_metrics"]:
        raise RuntimeError("strict artifact reload changed calibration metrics")
    return artifact


def train_geometry_reranker(
        base_cache, geometry_cache, parent_artifact_path, output,
        split_seed=0, model_seed=0, hidden_dim=256, dropout=0.1,
        lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=100,
        patience=10, device="cuda:0",
        calibration_fraction=DEFAULT_CALIBRATION_FRACTION,
        verbose=False):
    """Train the production scorer from strict train-only cache snapshots."""
    _validate_training_output_location(
        output,
        parent_artifact_path,
        protected_directories=(base_cache, geometry_cache),
    )
    for name, value in (
            ("hidden_dim", hidden_dim),
            ("batch_size", batch_size),
            ("max_epochs", max_epochs),
            ("patience", patience)):
        _strict_int(value, name, minimum=1)
    if (not isinstance(dropout, (float, int)) or isinstance(dropout, bool)
            or not 0.0 <= float(dropout) < 1.0):
        raise ValueError("dropout must lie in [0, 1)")
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")
    resolved_device = _resolve_device(device)
    _parent_inference_world_size()
    joined, base_manifest, geometry_manifest, parent = (
        load_geometry_training_data(
            base_cache, geometry_cache, parent_artifact_path
        )
    )
    base_digest = geometry_manifest["base_cache_binding"]["content_sha256"]
    _validate_authoritative_parent_inference(
        base_digest,
        resolved_device.type,
        resolved_device.index,
        _parent_matmul_allow_tf32(resolved_device),
    )
    materialize_parent_scores(joined, parent, device=resolved_device)
    fit_rows, calibration_rows = deterministic_scene_split(
        joined,
        seed=split_seed,
        calibration_fraction=calibration_fraction,
    )
    if not calibration_rows:
        raise ValueError("geometry training needs at least two scenes")
    split_metadata = build_scene_split_metadata(
        fit_rows,
        calibration_rows,
        split_seed=split_seed,
        calibration_fraction=calibration_fraction,
    )
    if (base_digest == AUTHORITATIVE_TRAIN_BASE_CONTENT_SHA256
            and split_metadata != AUTHORITATIVE_SPLIT_SEED0):
        raise ValueError("authoritative seed-0 scene split metadata mismatch")
    feature_mean, feature_std = compute_geometry_feature_stats(
        fit_rows, parent, batch_size=batch_size
    )
    _set_deterministic_seed(model_seed, resolved_device)
    model = QueryReranker(
        input_dim=GEOMETRY_INPUT_DIM,
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(resolved_device)
    return fit_and_save_geometry_model(
        model,
        fit_rows,
        calibration_rows,
        feature_mean,
        feature_std,
        parent,
        base_manifest,
        geometry_manifest,
        output,
        split_seed=split_seed,
        model_seed=model_seed,
        calibration_fraction=calibration_fraction,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        device=resolved_device,
        verbose=verbose,
        protected_output_directories=(base_cache, geometry_cache),
    )


def parse_args(argv=None):
    """Parse the immutable train-only geometry scorer command line."""
    parser = argparse.ArgumentParser(
        description="Train the ScanRefer REC mask-geometry reranker"
    )
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def run_training(args):
    """Run training from parsed command-line arguments."""
    return train_geometry_reranker(
        args.base_cache,
        args.geometry_cache,
        args.parent_artifact,
        args.output,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        verbose=args.verbose,
    )


def main(argv=None):
    args = parse_args(argv)
    artifact = run_training(args)
    metrics = artifact["calibration_metrics"]
    print(
        "Saved {} epoch={} weight={:.2f} calibration "
        "Acc@0.25={:.5f} Acc@0.50={:.5f}".format(
            str(Path(args.output).expanduser().resolve()),
            artifact["epoch"],
            artifact["geometry_weight"],
            metrics["acc025"],
            metrics["acc050"],
        ),
        flush=True,
    )
    return artifact


if __name__ == "__main__":
    main()
