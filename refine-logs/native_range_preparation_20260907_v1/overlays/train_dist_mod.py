# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------BeaUTyDETR_dks---------
"""Main script for language modulation."""

import contextlib
import copy
import hashlib
import io
import math
import os
from pathlib import Path
import stat

import numpy as np
import torch
import torch.distributed as dist

from main_utils import (
    build_parent_relative_text_verifier_audit_diagnostics,
    build_source_moe_gate_decision_diagnostics,
    fpr_scene_sample_identity_digest,
    is_counterfactual_parent_bounded_audit,
    parse_option,
    prepare_source_moe_gate_checkpoint_config,
    save_source_choice_diagnostics_receipt,
    BaseTrainTester,
)
from data.model_util_scannet import ScannetDatasetConfig
from models import MCLN
from models.candidate_local_visual import CandidateLocalVisual
from models.candidate_range_visual import CandidateRangeVisual
from src.joint_det_dataset import Joint3DDataset
from src.grounding_evaluator import GroundingEvaluator
from models import APCalculator, parse_predictions, parse_groundtruths
from models.rec_candidate_adapter import (
    FEATURE_SCHEMA_VERSION,
    build_rec_candidate_batch,
)
from models.rec_evaluator_filter import build_detector_overlap_valid
from models.density_aware_target_box_audit import (
    DensityAwareTargetBoxAuditAccumulator,
)
from models.rec_reranker import blend_candidate_scores
from models.rec_geometry_reranker import (
    build_deployed_parent_state,
    build_flat_parent_prior,
    build_rec_geometry_model_inputs,
    blend_rec_geometry_scores,
    _stable_masked_rank_normalize,
)
from models.rec_selective_residual import (
    apply_selective_policy,
    build_selective_pair_features,
    expected_selective_gain,
)
from models.rec_hierarchical_reranker import (
    apply_hierarchical_policy,
    select_hierarchical_proposal,
)
from models.rec_pareto_contextual_hierarchy import (
    V99_ARTIFACT_SCHEMA,
    V113_ARTIFACT_SCHEMA,
    apply_asymmetric_risk_contextual_policy,
    apply_pareto_contextual_policy,
)
from models.rec_mask_geometry import build_rec_mask_geometry_candidates
from scripts.train_rec_reranker import normalize_features

from tqdm import tqdm
import datetime

import ipdb
st = ipdb.set_trace

import numpy as np


V101_ARTIFACT_SCHEMA = "rec-pareto-contextual-full-train-artifact-v1"
V109_ARTIFACT_SCHEMA = (
    "rec-pareto-contextual-meshsp-nested-policy-full-train-artifact-v1"
)
PARETO_CONTEXTUAL_ARTIFACT_SCHEMAS = frozenset({
    V99_ARTIFACT_SCHEMA,
    V101_ARTIFACT_SCHEMA,
    V109_ARTIFACT_SCHEMA,
})

FPR_SCENE_DISJOINT_FOLD_COUNT = 5
FPR_SCENE_DISJOINT_TOTAL_SCENES = 511
FPR_SCENE_DISJOINT_TOTAL_SAMPLES = 32919
FPR_SCENE_DISJOINT_SPLITS = {
    0: {"fit_scenes": 402, "holdout_scenes": 109,
        "fit_samples": 25790, "holdout_samples": 7129},
    1: {"fit_scenes": 400, "holdout_scenes": 111,
        "fit_samples": 25578, "holdout_samples": 7341},
    2: {"fit_scenes": 408, "holdout_scenes": 103,
        "fit_samples": 26590, "holdout_samples": 6329},
    3: {"fit_scenes": 417, "holdout_scenes": 94,
        "fit_samples": 26714, "holdout_samples": 6205},
    4: {"fit_scenes": 417, "holdout_scenes": 94,
        "fit_samples": 27004, "holdout_samples": 5915},
}


class FPRSceneDisjointDatasetView(object):
    """Attach immutable source-row identities to one dataset partition."""

    def __init__(self, dataset, sample_ids):
        if len(dataset) != len(sample_ids):
            raise ValueError("FPR scene view identities must align with data")
        self._dataset = dataset
        self._sample_ids = tuple(sample_ids)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        item = self._dataset[index]
        if not isinstance(item, dict):
            raise ValueError("FPR scene dataset item must be a dict")
        item = dict(item)
        item["fpr_scene_audit_sample_index"] = np.int64(
            self._sample_ids[index]
        )
        return item

    def __getattr__(self, name):
        if name in ("_dataset", "_sample_ids"):
            raise AttributeError(name)
        return getattr(self._dataset, name)


class DensityTargetBoxSceneDatasetView(object):
    """Attach immutable source-row identities for the density audit."""

    def __init__(self, dataset, sample_ids):
        if len(dataset) != len(sample_ids):
            raise ValueError("density scene identities must align with data")
        self._dataset = dataset
        self._sample_ids = tuple(sample_ids)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        item = self._dataset[index]
        if not isinstance(item, dict):
            raise ValueError("density scene dataset item must be a dict")
        item = dict(item)
        item["density_scene_audit_sample_index"] = np.int64(
            self._sample_ids[index]
        )
        return item

    def __getattr__(self, name):
        if name in ("_dataset", "_sample_ids"):
            raise AttributeError(name)
        return getattr(self._dataset, name)


def fpr_scene_disjoint_fold(scan_id):
    """Map one immutable scene id to its preregistered Nr3D fold."""
    if not isinstance(scan_id, str) or not scan_id:
        raise ValueError("FPR scene-disjoint audit requires string scan ids")
    return int(hashlib.sha256(scan_id.encode("utf-8")).hexdigest()[:8], 16) % 5


def build_fpr_scene_disjoint_dataset_views(base_dataset, fold,
                                             expected_counts):
    """Create disjoint train/holdout annotation views of Nr3D train data."""
    if fold not in FPR_SCENE_DISJOINT_SPLITS:
        raise ValueError("FPR scene-disjoint fold must be in [0, 4]")
    if not isinstance(expected_counts, dict):
        raise ValueError("FPR scene-disjoint expected counts are required")
    canonical = FPR_SCENE_DISJOINT_SPLITS[fold]
    if expected_counts != canonical:
        raise ValueError(
            "FPR scene-disjoint expected counts drifted: {} != {}".format(
                expected_counts, canonical
            )
        )
    annotations = getattr(base_dataset, "annos", None)
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("FPR scene-disjoint base annotations are missing")
    if getattr(base_dataset, "split", None) != "train":
        raise ValueError("FPR scene-disjoint base must use the train split")

    all_scenes = set()
    fit_annotations = []
    holdout_annotations = []
    fit_scenes = set()
    holdout_scenes = set()
    fit_sample_ids = []
    holdout_sample_ids = []
    for sample_id, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError("FPR scene-disjoint annotation must be a dict")
        if annotation.get("dataset") != "nr3d":
            raise ValueError("FPR scene-disjoint audit accepts only Nr3D")
        scan_id = annotation.get("scan_id")
        assigned_fold = fpr_scene_disjoint_fold(scan_id)
        all_scenes.add(scan_id)
        if assigned_fold == fold:
            holdout_annotations.append(annotation)
            holdout_sample_ids.append(sample_id)
            holdout_scenes.add(scan_id)
        else:
            fit_annotations.append(annotation)
            fit_sample_ids.append(sample_id)
            fit_scenes.add(scan_id)

    if len(annotations) != FPR_SCENE_DISJOINT_TOTAL_SAMPLES:
        raise ValueError(
            "Nr3D train sample count drifted: {} != {}".format(
                len(annotations), FPR_SCENE_DISJOINT_TOTAL_SAMPLES
            )
        )
    if len(all_scenes) != FPR_SCENE_DISJOINT_TOTAL_SCENES:
        raise ValueError(
            "Nr3D train scene count drifted: {} != {}".format(
                len(all_scenes), FPR_SCENE_DISJOINT_TOTAL_SCENES
            )
        )
    observed = {
        "fit_scenes": len(fit_scenes),
        "holdout_scenes": len(holdout_scenes),
        "fit_samples": len(fit_annotations),
        "holdout_samples": len(holdout_annotations),
    }
    if observed != canonical:
        raise ValueError(
            "Nr3D scene-disjoint split drifted: {} != {}".format(
                observed, canonical
            )
        )
    if fit_scenes.intersection(holdout_scenes):
        raise ValueError("FPR fit and holdout scenes overlap")
    if fit_scenes.union(holdout_scenes) != all_scenes:
        raise ValueError("FPR scene partition is incomplete")

    fit_dataset = copy.copy(base_dataset)
    holdout_dataset = copy.copy(base_dataset)
    fit_dataset.annos = fit_annotations
    holdout_dataset.annos = holdout_annotations
    for dataset in (fit_dataset, holdout_dataset):
        dataset.dataset_dict = {"nr3d": 1}
        dataset.joint_det = False
        dataset.augment_det = False
        dataset.overfit = False
    fit_dataset.augment = True
    holdout_dataset.augment = False
    fit_dataset = FPRSceneDisjointDatasetView(
        fit_dataset, fit_sample_ids
    )
    holdout_dataset = FPRSceneDisjointDatasetView(
        holdout_dataset, holdout_sample_ids
    )
    return fit_dataset, holdout_dataset, {
        "schema": "mcln-fpr-tv-nr3d-scene-fold-v1",
        "fold": fold,
        "fold_count": FPR_SCENE_DISJOINT_FOLD_COUNT,
        "hash": "sha256-prefix32-mod5",
        "total_scenes": len(all_scenes),
        "total_samples": len(annotations),
        "fit_sample_identity_sha256": (
            fpr_scene_sample_identity_digest(fit_sample_ids)
        ),
        "holdout_sample_identity_sha256": (
            fpr_scene_sample_identity_digest(holdout_sample_ids)
        ),
        **observed
    }


def build_density_target_box_scene_dataset_views(
        base_dataset, fold, expected_counts):
    """Reuse the frozen Nr3D split with density-specific row identities."""
    fit_view, holdout_view, metadata = (
        build_fpr_scene_disjoint_dataset_views(
            base_dataset, fold, expected_counts
        )
    )
    fit_dataset = DensityTargetBoxSceneDatasetView(
        fit_view._dataset, fit_view._sample_ids
    )
    holdout_dataset = DensityTargetBoxSceneDatasetView(
        holdout_view._dataset, holdout_view._sample_ids
    )
    metadata = dict(metadata)
    metadata["schema"] = "mcln-density-target-box-nr3d-scene-fold-v1"
    return fit_dataset, holdout_dataset, metadata


_PARENT_RUNTIME_COMPATIBILITY = {
    "schema": "rec-parent-inference-contract",
    "version": 1,
    "device_type": "cuda",
    "device_index": 0,
    "local_batch_size": 12,
    "world_size": 1,
    "remainder_policy": "natural-remainder",
    "dtype": "float32",
    "autocast": False,
    "allow_tf32": True,
    "eval": True,
    "no_grad": True,
    "score_builder": "normalized-query-reranker-rank-blend",
    "score_builder_version": 1,
    "canonical_query_tie_policy": "score-desc-query-index-asc-v1",
}


def _disabled_runtime_autocast(device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    if device.type != "cpu":
        raise ValueError("REC reranker inference supports only CPU or CUDA")
    cpu_amp = getattr(getattr(torch, "cpu", None), "amp", None)
    cpu_autocast = getattr(cpu_amp, "autocast", None)
    if cpu_autocast is not None:
        return cpu_autocast(enabled=False)
    generic_autocast = getattr(torch, "autocast", None)
    if generic_autocast is not None:
        return generic_autocast(device_type="cpu", enabled=False)
    return contextlib.nullcontext()


@contextlib.contextmanager
def _runtime_tf32(device, allow_tf32):
    if device.type != "cuda":
        yield
        return
    previous = bool(torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _module_device(model):
    for value in list(model.parameters()) + list(model.buffers()):
        return value.device
    return torch.device("cpu")


def apply_rec_joint_box_mask_runtime_policy(
        features, valid_mask, baseline_scores, adapter, artifact):
    """Apply the train-gated joint quality policy to flat geometry scores.

    The adapter sees inference-only 179-D geometry features.  It can promote a
    candidate only when both predicted box tiers are protected; otherwise the
    exact baseline score tensor is returned.
    """
    from scripts.train_scanrefer_joint_box_mask import (
        FEATURE_DIM,
        select_quality_policy,
    )
    from models.rec_joint_box_mask import (
        MASK_LOGIT_THRESHOLDS,
        MASK_POLICY_COUNT,
        MASK_SOURCE_NAMES,
    )

    if (not isinstance(features, torch.Tensor) or features.dim() != 3
            or tuple(features.shape[1:]) != (112, FEATURE_DIM)
            or features.dtype != torch.float32):
        raise ValueError("joint adapter features must have shape [B,112,179]")
    if (not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != features.shape[:2]):
        raise ValueError("joint adapter validity does not match features")
    if (not isinstance(baseline_scores, torch.Tensor)
            or baseline_scores.dtype != torch.float32
            or baseline_scores.shape != valid_mask.shape
            or baseline_scores.device != features.device):
        raise ValueError("joint adapter baseline scores are malformed")
    if not bool(valid_mask.any(dim=1).all().item()):
        raise ValueError("joint adapter needs one valid candidate per row")
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("joint adapter features are non-finite")
    if not isinstance(adapter, torch.nn.Module) or adapter.training:
        raise ValueError("joint adapter must be a frozen eval module")
    if any(parameter.requires_grad for parameter in adapter.parameters()):
        raise ValueError("joint adapter parameters must be frozen")
    if not isinstance(artifact, dict):
        raise ValueError("joint adapter artifact is invalid")
    mean = artifact.get("feature_mean")
    std = artifact.get("feature_std")
    if (not isinstance(mean, torch.Tensor)
            or not isinstance(std, torch.Tensor)
            or tuple(mean.shape) != (FEATURE_DIM,)
            or tuple(std.shape) != (FEATURE_DIM,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or bool((std <= 0).any().item())):
        raise ValueError("joint adapter normalization is invalid")
    mean = mean.to(features.device, dtype=features.dtype)
    std = std.to(features.device, dtype=features.dtype)
    normalized = (features - mean) / std
    normalized = torch.where(
        valid_mask.unsqueeze(-1), normalized, torch.zeros_like(normalized)
    )
    with torch.no_grad(), _disabled_runtime_autocast(features.device):
        outputs = adapter(
            normalized.reshape(-1, 16, 7, FEATURE_DIM),
            valid_mask.reshape(-1, 16, 7),
        )
    if not isinstance(outputs, dict):
        raise ValueError("joint adapter outputs are malformed")
    mask_pred = outputs.get("mask_iou")
    box_logits = outputs.get("box_logits")
    mask_policy_logits = outputs.get("mask_policy_logits")
    expected_mask_shape = (features.shape[0], 16, 7)
    if (not isinstance(mask_pred, torch.Tensor)
            or tuple(mask_pred.shape) != expected_mask_shape
            or not isinstance(box_logits, torch.Tensor)
            or tuple(box_logits.shape) != expected_mask_shape + (2,)
            or mask_pred.device != features.device
            or box_logits.device != features.device
            or not bool(torch.isfinite(mask_pred).all().item())
            or not bool(torch.isfinite(box_logits).all().item())):
        raise ValueError("joint adapter prediction tensors are malformed")
    if (not isinstance(mask_policy_logits, torch.Tensor)
            or tuple(mask_policy_logits.shape)
            != (features.shape[0], 16, MASK_POLICY_COUNT)
            or mask_policy_logits.device != features.device
            or not bool(torch.isfinite(mask_policy_logits).all().item())):
        raise ValueError("joint adapter mask policy logits are malformed")
    policy = select_quality_policy(
        mask_pred.reshape(features.shape[0], -1),
        box_logits.reshape(features.shape[0], -1, 2),
        baseline_scores,
        valid_mask,
        switch_margin=float(artifact.get("switch_margin", 0.02)),
        box_margin=float(artifact.get("box_margin", 0.05)),
    )
    selected = policy["selected_flat_index"]
    baseline = policy["baseline_flat_index"]
    switched = policy["switched"]
    scores = baseline_scores.clone()
    rows = torch.arange(scores.shape[0], device=scores.device)
    switched_rows = switched.nonzero(as_tuple=False).reshape(-1)
    if switched_rows.numel():
        # A score above the current finite maximum gives the selected candidate
        # a deterministic Top-1 position while preserving all other ranks.
        selected_rows = rows[switched_rows]
        selected_indices = selected[switched_rows]
        current_max = scores[selected_rows, baseline[switched_rows]]
        scores[selected_rows, selected_indices] = current_max + 1.0
    scores = scores.masked_fill(~valid_mask, -float("inf"))
    selected_parent_positions = torch.div(
        selected, 7, rounding_mode="floor"
    )
    selected_mask_policy = mask_policy_logits[
        rows, selected_parent_positions
    ].argmax(dim=-1)
    threshold_count = len(MASK_LOGIT_THRESHOLDS)
    selected_mask_source = torch.div(
        selected_mask_policy, threshold_count, rounding_mode="floor"
    )
    selected_mask_threshold_index = torch.remainder(
        selected_mask_policy, threshold_count
    )
    threshold_values = torch.tensor(
        MASK_LOGIT_THRESHOLDS,
        dtype=features.dtype,
        device=features.device,
    )
    selected_mask_threshold = threshold_values[
        selected_mask_threshold_index
    ]
    if (bool((selected_mask_source < 0).any().item())
            or bool((selected_mask_source >= len(MASK_SOURCE_NAMES)).any().item())):
        raise RuntimeError("joint adapter selected an invalid mask source")
    return {
        "scores": scores,
        "selected_flat_indices": selected.detach(),
        "baseline_flat_indices": baseline.detach(),
        "switched": switched.detach(),
        "selected_parent_positions": selected_parent_positions.detach(),
        "selected_mask_policy_indices": selected_mask_policy.detach(),
        "selected_mask_source_indices": selected_mask_source.detach(),
        "selected_mask_threshold_indices": (
            selected_mask_threshold_index.detach()
        ),
        "selected_mask_thresholds": selected_mask_threshold.detach(),
    }


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_parent_inference_runtime_compatibility(contract):
    """Validate only the train/runtime execution-compatible projection."""
    if not isinstance(contract, dict):
        raise ValueError("parent inference contract must be an object")
    projection = {
        key: contract.get(key) for key in _PARENT_RUNTIME_COMPATIBILITY
    }
    incompatible = [
        key for key, expected in _PARENT_RUNTIME_COMPATIBILITY.items()
        if (type(projection[key]) is not type(expected)
            or projection[key] != expected)
    ]
    if incompatible:
        raise ValueError(
            "parent inference contract is not runtime-compatible: {}".format(
                ", ".join(incompatible)
            )
        )
    return projection


def validate_rec_geometry_runtime_environment(
        args, actual_batch_size, device, *, batch_idx, num_batches):
    """Require the authoritative local CUDA batch execution contract."""
    configured_batch_size = getattr(args, "batch_size", None)
    if (not isinstance(configured_batch_size, int)
            or isinstance(configured_batch_size, bool)
            or configured_batch_size != 12):
        raise ValueError(
            "geometry reranker runtime requires args.batch_size=12"
        )
    if (not isinstance(actual_batch_size, int)
            or isinstance(actual_batch_size, bool)
            or not 1 <= actual_batch_size <= 12):
        raise ValueError(
            "geometry reranker input batches must contain at most 12 rows"
        )
    if (not isinstance(batch_idx, int) or isinstance(batch_idx, bool)
            or not isinstance(num_batches, int)
            or isinstance(num_batches, bool)
            or num_batches < 1
            or not 0 <= batch_idx < num_batches):
        raise ValueError("geometry reranker runtime batch position is invalid")
    if actual_batch_size != 12 and batch_idx != num_batches - 1:
        raise ValueError(
            "only the final batch may contain fewer than 12 rows"
        )
    device = torch.device(device)
    if device.type != "cuda" or device.index != 0:
        raise ValueError("geometry reranker runtime requires cuda:0")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = int(torch.distributed.get_world_size())
    else:
        world_size = 1
    if world_size != 1:
        raise ValueError("geometry reranker runtime requires world_size=1")


def load_rec_geometry_runtime_artifacts(parent_path, geometry_path,
                                        device):
    """Stable-load each frozen artifact once and bind geometry to that parent."""
    from models.rec_finetune import (
        REC_FINETUNE_GEOMETRY_SCHEMA,
        REC_FINETUNE_PARENT_SCHEMA,
        load_rec_finetune_runtime_artifacts,
        rec_finetune_artifact_schema,
    )

    parent_schema = (
        rec_finetune_artifact_schema(parent_path)
        if Path(parent_path).expanduser().is_file() else None
    )
    geometry_schema = (
        rec_finetune_artifact_schema(geometry_path)
        if Path(geometry_path).expanduser().is_file() else None
    )
    parent_is_new = parent_schema == REC_FINETUNE_PARENT_SCHEMA
    geometry_is_new = geometry_schema == REC_FINETUNE_GEOMETRY_SCHEMA
    new_schemas = {
        REC_FINETUNE_PARENT_SCHEMA, REC_FINETUNE_GEOMETRY_SCHEMA
    }
    if parent_schema in new_schemas or geometry_schema in new_schemas:
        if not parent_is_new or not geometry_is_new:
            raise ValueError("mixed legacy and REC fine-tune artifacts")
        return load_rec_finetune_runtime_artifacts(
            parent_path, geometry_path, device=device
        )

    from scripts.train_rec_geometry_reranker import (
        load_geometry_reranker_artifact,
        load_parent_reranker_snapshot,
        validate_geometry_artifact,
    )

    parent_model, parent_artifact = load_parent_reranker_snapshot(
        parent_path, device=device
    )
    geometry_model, geometry_artifact = load_geometry_reranker_artifact(
        geometry_path, device=device
    )
    validate_geometry_artifact(
        geometry_artifact, parent=(parent_model, parent_artifact)
    )
    return (
        parent_model, parent_artifact, geometry_model, geometry_artifact
    )


def validate_rec_selective_residual_runtime_provenance(
        parent_model, geometry_model, geometry_artifact, residual_model,
        residual_artifact, device):
    """Bind one deployable residual to the live frozen scoring stack."""
    if not isinstance(residual_artifact, dict):
        raise ValueError("residual artifact must be an object")
    parent_sha = getattr(parent_model, "_artifact_sha256", None)
    geometry_sha = getattr(geometry_model, "_artifact_sha256", None)
    residual_sha = getattr(residual_model, "_artifact_sha256", None)
    inputs = residual_artifact.get("input_sha256")
    if (not _is_sha256(parent_sha) or not isinstance(inputs, dict)
            or inputs.get("parent") != parent_sha):
        raise ValueError("residual parent artifact SHA mismatch")
    if (not _is_sha256(geometry_sha)
            or inputs.get("geometry") != geometry_sha):
        raise ValueError("residual geometry artifact SHA mismatch")
    if not _is_sha256(residual_sha):
        raise ValueError("residual model lacks a stable artifact SHA")
    if (inputs.get("backbone")
            != geometry_artifact.get("checkpoint_sha256")
            and geometry_artifact.get("checkpoint_sha256") is not None):
        raise ValueError("residual backbone artifact SHA mismatch")
    from scripts.train_scanrefer_rec_selective_residual import (
        build_selective_pair_feature_names,
    )

    expected_names = build_selective_pair_feature_names(
        geometry_artifact.get("feature_names")
    )
    selection = residual_artifact.get("selection")
    if (residual_artifact.get("schema") != "rec-selective-residual-v1"
            or residual_artifact.get("deployable") is not True
            or residual_artifact.get("validation_data_accessed") is not False
            or residual_artifact.get("input_dim") != 185
            or residual_artifact.get("feature_names") != expected_names
            or not isinstance(selection, dict)
            or not isinstance(selection.get("margin"), (float, int))
            or isinstance(selection.get("margin"), bool)
            or not math.isfinite(float(selection["margin"]))
            or float(selection["margin"]) <= 0.0):
        raise ValueError("residual runtime artifact contract is invalid")
    _validate_frozen_artifact_model(
        residual_model, residual_artifact, "selective residual"
    )
    if _module_device(residual_model) != torch.device(device):
        raise ValueError("residual model device differs from live backbone")
    return {
        "parent_sha256": parent_sha,
        "geometry_sha256": geometry_sha,
        "residual_sha256": residual_sha,
        "margin": float(selection["margin"]),
    }


def load_rec_selective_residual_runtime_artifact(
        path, device, parent_model, geometry_model, geometry_artifact):
    """Stable-load one residual and validate all frozen provenance."""
    parent_sha = getattr(parent_model, "_artifact_sha256", None)
    geometry_sha = getattr(geometry_model, "_artifact_sha256", None)
    if not _is_sha256(parent_sha) or not _is_sha256(geometry_sha):
        raise ValueError("frozen rerankers lack stable artifact SHAs")
    from scripts.train_scanrefer_rec_selective_residual import (
        load_selective_residual_artifact,
    )

    model, artifact = load_selective_residual_artifact(
        path,
        device=device,
        parent_sha256=parent_sha,
        geometry_sha256=geometry_sha,
    )
    validate_rec_selective_residual_runtime_provenance(
        parent_model,
        geometry_model,
        geometry_artifact,
        model,
        artifact,
        device,
    )
    return model, artifact


def validate_rec_hierarchical_runtime_provenance(
        parent_model, geometry_model, geometry_artifact,
        hierarchical_model, hierarchical_artifact, device):
    """Bind one deployed hierarchy to the live frozen scoring stack."""
    if not isinstance(hierarchical_artifact, dict):
        raise ValueError("hierarchical artifact must be an object")
    parent_sha = getattr(parent_model, "_artifact_sha256", None)
    geometry_sha = getattr(geometry_model, "_artifact_sha256", None)
    hierarchical_sha = getattr(
        hierarchical_model, "_artifact_sha256", None
    )
    inputs = hierarchical_artifact.get("input_sha256")
    if (not _is_sha256(parent_sha) or not isinstance(inputs, dict)
            or inputs.get("parent") != parent_sha):
        raise ValueError("hierarchical parent artifact SHA mismatch")
    if (not _is_sha256(geometry_sha)
            or inputs.get("geometry") != geometry_sha):
        raise ValueError("hierarchical geometry artifact SHA mismatch")
    if not _is_sha256(hierarchical_sha):
        raise ValueError("hierarchical model lacks a stable artifact SHA")
    checkpoint_sha = geometry_artifact.get("checkpoint_sha256")
    if (checkpoint_sha is not None
            and inputs.get("backbone") != checkpoint_sha):
        raise ValueError("hierarchical backbone artifact SHA mismatch")
    artifact_schema = hierarchical_artifact.get("schema")
    if artifact_schema == V113_ARTIFACT_SCHEMA:
        from scripts.build_v113_meshsp_asymmetric_risk_artifact import (
            validate_v113_artifact,
        )
        validate_v113_artifact(
            hierarchical_artifact,
            expected_parent_sha256=parent_sha,
            expected_geometry_sha256=geometry_sha,
            expected_feature_names=geometry_artifact.get("feature_names"),
        )
        policy = hierarchical_artifact.get("policy", {})
        margin = policy.get("aggregate_lcb_margin")
        min_head_gain025 = policy.get("min_head_lcb025")
        min_head_gain050 = policy.get("min_head_lcb050")
        scene_fold_sha256 = hierarchical_artifact["fit"][
            "scene_fold_sha256"
        ]
        oof_record_sha256 = hierarchical_artifact["oof_evidence"][
            "sha256"
        ]
    elif artifact_schema in PARETO_CONTEXTUAL_ARTIFACT_SCHEMAS:
        if artifact_schema == V99_ARTIFACT_SCHEMA:
            from scripts.build_v99_pareto_contextual_artifact import (
                validate_v99_artifact,
            )
            validate_pareto_artifact = validate_v99_artifact
        elif artifact_schema == V101_ARTIFACT_SCHEMA:
            from scripts.build_v101_full_train_pareto_artifact import (
                validate_v101_artifact,
            )
            validate_pareto_artifact = validate_v101_artifact
        elif artifact_schema == V109_ARTIFACT_SCHEMA:
            from scripts.build_v109_meshsp_nested_policy_artifact import (
                validate_v109_artifact,
            )
            validate_pareto_artifact = validate_v109_artifact
        else:
            raise ValueError("unsupported Pareto hierarchy artifact schema")
        validate_pareto_artifact(
            hierarchical_artifact,
            expected_parent_sha256=parent_sha,
            expected_geometry_sha256=geometry_sha,
            expected_feature_names=geometry_artifact.get("feature_names"),
        )
        policy = hierarchical_artifact.get("policy", {})
        margin = policy.get("aggregate_margin")
        min_head_gain025 = policy.get("min_head_gain025", 0.0)
        min_head_gain050 = policy.get("min_head_gain050", 0.0)
        scene_fold_sha256 = hierarchical_artifact["fit"][
            "scene_fold_sha256"
        ]
        oof_record_sha256 = hierarchical_artifact["oof_evidence"]["sha256"]
    else:
        from scripts.train_scanrefer_rec_hierarchical_reranker import (
            validate_hierarchical_artifact,
        )
        validate_hierarchical_artifact(
            hierarchical_artifact,
            expected_geometry_feature_names=geometry_artifact.get(
                "feature_names"
            ),
            expected_backbone_sha256=inputs.get("backbone"),
            expected_parent_sha256=parent_sha,
            expected_geometry_sha256=geometry_sha,
            expected_deployable=True,
        )
        selection = hierarchical_artifact.get("selection")
        margin = (
            selection.get("margin") if isinstance(selection, dict) else None
        )
        scene_fold_sha256 = hierarchical_artifact["scene_fold_sha256"]
        oof_record_sha256 = hierarchical_artifact["oof_record_sha256"]
        min_head_gain025 = 0.0
        min_head_gain050 = 0.0
    if (not isinstance(margin, (float, int))
            or isinstance(margin, bool)
            or not math.isfinite(float(margin))
            or float(margin) <= 0.0):
        raise ValueError("hierarchical runtime margin is invalid")
    for name, value in {
            "min_head_gain025": min_head_gain025,
            "min_head_gain050": min_head_gain050}.items():
        if (not isinstance(value, (float, int)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError("hierarchical runtime {} is invalid".format(name))
    _validate_frozen_artifact_model(
        hierarchical_model, hierarchical_artifact, "hierarchical reranker"
    )
    if _module_device(hierarchical_model) != torch.device(device):
        raise ValueError("hierarchical model device differs from live backbone")
    return {
        "parent_sha256": parent_sha,
        "geometry_sha256": geometry_sha,
        "hierarchical_sha256": hierarchical_sha,
        "margin": float(margin),
        "normalization_sha256": hierarchical_artifact[
            "normalization_sha256"
        ],
        "scene_fold_sha256": scene_fold_sha256,
        "oof_record_sha256": oof_record_sha256,
    }


def load_rec_hierarchical_runtime_artifact(
        path, device, parent_model, geometry_model, geometry_artifact):
    """Stable-load one deployed hierarchy and validate frozen provenance."""
    parent_sha = getattr(parent_model, "_artifact_sha256", None)
    geometry_sha = getattr(geometry_model, "_artifact_sha256", None)
    if not _is_sha256(parent_sha) or not _is_sha256(geometry_sha):
        raise ValueError("frozen rerankers lack stable artifact SHAs")
    resolved = Path(path).expanduser().absolute()
    header = None
    try:
        entry = os.lstat(str(resolved))
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
            raise ValueError(
                "hierarchical artifact must be a regular non-symlink file"
            )
        descriptor = os.open(
            str(resolved), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            before = os.fstat(descriptor)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                snapshot = handle.read()
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        live = os.stat(str(resolved), follow_symlinks=False)
    except FileNotFoundError:
        # Preserve the legacy loader seam used by unit tests and downstream
        # wrappers that resolve the path themselves.  Real V99 artifacts are
        # immutable files and always take the stable schema-inspection path.
        snapshot = None
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("could not inspect hierarchical artifact") from error
    identity = lambda value: (
        int(value.st_dev), int(value.st_ino), int(value.st_size),
        int(value.st_mtime_ns), int(value.st_ctime_ns),
    )
    if snapshot is not None:
        if (identity(before) != identity(after)
                or identity(after) != identity(live)):
            raise ValueError(
                "hierarchical artifact changed during schema inspection"
            )
        try:
            header = torch.load(io.BytesIO(snapshot), map_location="cpu")
        except Exception as error:
            raise ValueError(
                "could not inspect hierarchical artifact schema"
            ) from error
    header_schema = header.get("schema") if isinstance(header, dict) else None
    if header_schema == V113_ARTIFACT_SCHEMA:
        from scripts.build_v113_meshsp_asymmetric_risk_artifact import (
            load_v113_artifact,
        )
        model, artifact = load_v113_artifact(
            path,
            device=device,
            parent_sha256=parent_sha,
            geometry_sha256=geometry_sha,
            expected_geometry_feature_names=geometry_artifact.get(
                "feature_names"
            ),
        )
    elif header_schema in PARETO_CONTEXTUAL_ARTIFACT_SCHEMAS:
        if header_schema == V99_ARTIFACT_SCHEMA:
            from scripts.build_v99_pareto_contextual_artifact import (
                load_v99_artifact,
            )
            load_pareto_artifact = load_v99_artifact
        elif header_schema == V101_ARTIFACT_SCHEMA:
            from scripts.build_v101_full_train_pareto_artifact import (
                load_v101_artifact,
            )
            load_pareto_artifact = load_v101_artifact
        elif header_schema == V109_ARTIFACT_SCHEMA:
            from scripts.build_v109_meshsp_nested_policy_artifact import (
                load_v109_artifact,
            )
            load_pareto_artifact = load_v109_artifact
        else:
            raise ValueError("unsupported Pareto hierarchy artifact schema")
        model, artifact = load_pareto_artifact(
            path,
            device=device,
            parent_sha256=parent_sha,
            geometry_sha256=geometry_sha,
            expected_geometry_feature_names=geometry_artifact.get(
                "feature_names"
            ),
        )
    else:
        from scripts.train_scanrefer_rec_hierarchical_reranker import (
            load_hierarchical_artifact,
        )
        model, artifact = load_hierarchical_artifact(
            path,
            device=device,
            expected_geometry_feature_names=geometry_artifact.get(
                "feature_names"
            ),
            expected_deployable=True,
            parent_sha256=parent_sha,
            geometry_sha256=geometry_sha,
        )
    validate_rec_hierarchical_runtime_provenance(
        parent_model,
        geometry_model,
        geometry_artifact,
        model,
        artifact,
        device,
    )
    return model, artifact


def _validate_frozen_artifact_model(model, artifact, label):
    expected = artifact.get("model_state_dict")
    actual = model.state_dict()
    if (not isinstance(expected, dict) or set(expected) != set(actual)
            or any(not isinstance(expected[name], torch.Tensor)
                   or expected[name].dtype != actual[name].dtype
                   or tuple(expected[name].shape) != tuple(actual[name].shape)
                   or not torch.equal(
                       expected[name].detach().cpu(),
                       actual[name].detach().cpu(),
                   ) for name in actual)):
        raise ValueError("live {} state differs from its artifact".format(label))
    if model.training or any(value.requires_grad for value in model.parameters()):
        raise ValueError("{} must be frozen in eval mode".format(label))


def validate_rec_geometry_runtime_provenance(
        args, parent_model, parent_artifact, geometry_model,
        geometry_artifact, device):
    """Validate stable hashes, parent binding, backbone, and live model state."""
    parent_sha = getattr(parent_model, "_artifact_sha256", None)
    geometry_sha = getattr(geometry_model, "_artifact_sha256", None)
    if (not _is_sha256(parent_sha)
            or parent_sha != geometry_artifact.get("parent_artifact_sha256")):
        raise ValueError("geometry artifact parent artifact SHA mismatch")
    if not _is_sha256(geometry_sha):
        raise ValueError("geometry model lacks a stable artifact SHA")
    projection = validate_parent_inference_runtime_compatibility(
        geometry_artifact.get("parent_inference_contract")
    )
    device = torch.device(device)
    if (device.type != projection["device_type"]
            or device.index != projection["device_index"]):
        raise ValueError("live device differs from parent runtime contract")

    validate_rec_reranker_provenance(args, parent_artifact)
    validate_rec_reranker_provenance(args, geometry_artifact)
    from models.rec_finetune import (
        REC_FINETUNE_GEOMETRY_SCHEMA,
        REC_FINETUNE_PARENT_SCHEMA,
    )

    parent_is_new = (
        parent_artifact.get("schema") == REC_FINETUNE_PARENT_SCHEMA
    )
    geometry_is_new = (
        geometry_artifact.get("schema") == REC_FINETUNE_GEOMETRY_SCHEMA
    )
    new_schemas = {
        REC_FINETUNE_PARENT_SCHEMA, REC_FINETUNE_GEOMETRY_SCHEMA
    }
    if (parent_artifact.get("schema") in new_schemas
            or geometry_artifact.get("schema") in new_schemas):
        if not parent_is_new or not geometry_is_new:
            raise ValueError("mixed legacy and REC fine-tune artifacts")
        from models.rec_finetune import validate_rec_finetune_artifact_pair

        validate_rec_finetune_artifact_pair(
            parent_artifact,
            geometry_artifact,
            parent_artifact_sha256=parent_sha,
            parent_model=parent_model,
            geometry_model=geometry_model,
        )
    else:
        from scripts.train_rec_geometry_reranker import (
            _validate_live_parent_state,
            validate_geometry_artifact,
        )

        _validate_live_parent_state(parent_model, parent_artifact)
        validate_geometry_artifact(
            geometry_artifact, parent=(parent_model, parent_artifact)
        )
    _validate_frozen_artifact_model(
        parent_model, parent_artifact, "parent reranker"
    )
    _validate_frozen_artifact_model(
        geometry_model, geometry_artifact, "geometry reranker"
    )
    if (_module_device(parent_model) != device
            or _module_device(geometry_model) != device):
        raise ValueError("reranker model device differs from live backbone")
    return projection


def validate_rec_joint_box_mask_runtime_provenance(
        parent_model, geometry_model, geometry_artifact,
        joint_model, joint_artifact, device):
    """Bind a deployable joint adapter to the exact frozen scoring stack."""
    if not isinstance(joint_artifact, dict):
        raise ValueError("joint adapter artifact must be an object")
    if (joint_artifact.get("schema")
            != "rec-joint-box-mask-adapter-v2"
            or joint_artifact.get("deployable") is not True
            or joint_artifact.get("selection") != "joint_adapter"
            or joint_artifact.get("validation_data_accessed") is not False
            or joint_artifact.get("inference_uses_ground_truth") is not False):
        raise ValueError("joint adapter runtime artifact contract is invalid")
    parent_sha = getattr(parent_model, "_artifact_sha256", None)
    geometry_sha = getattr(geometry_model, "_artifact_sha256", None)
    if (not _is_sha256(parent_sha)
            or joint_artifact.get("parent_artifact_sha256") != parent_sha):
        raise ValueError("joint adapter parent artifact SHA mismatch")
    if (not _is_sha256(geometry_sha)
            or joint_artifact.get("geometry_artifact_sha256") != geometry_sha):
        raise ValueError("joint adapter geometry artifact SHA mismatch")
    backbone_sha = geometry_artifact.get("checkpoint_sha256")
    if (backbone_sha is not None
            and joint_artifact.get("backbone_checkpoint_sha256")
            != backbone_sha):
        raise ValueError("joint adapter backbone artifact SHA mismatch")
    names = joint_artifact.get("feature_names")
    if (not isinstance(names, list) or len(names) != 179
            or names != list(geometry_artifact.get("feature_names", ()))):
        raise ValueError("joint adapter feature schema differs from geometry")
    for key in ("switch_margin", "box_margin"):
        value = joint_artifact.get(key)
        if (not isinstance(value, (float, int)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError("joint adapter {} is invalid".format(key))
    from models.rec_joint_box_mask import (
        LEGACY_MASK_POLICY_INDEX,
        MASK_LOGIT_THRESHOLDS,
        MASK_SOURCE_NAMES,
    )
    if (joint_artifact.get("mask_policy_source_names")
            != list(MASK_SOURCE_NAMES)
            or joint_artifact.get("mask_policy_logit_thresholds")
            != list(MASK_LOGIT_THRESHOLDS)
            or joint_artifact.get("legacy_mask_policy_index")
            != LEGACY_MASK_POLICY_INDEX):
        raise ValueError("joint adapter mask policy schema is invalid")
    _validate_frozen_artifact_model(
        joint_model, joint_artifact, "joint box-mask adapter"
    )
    if _module_device(joint_model) != torch.device(device):
        raise ValueError("joint adapter device differs from live backbone")
    return {
        "parent_sha256": parent_sha,
        "geometry_sha256": geometry_sha,
        "adapter_sha256": getattr(joint_model, "_artifact_sha256", None),
    }


def load_rec_joint_box_mask_runtime_artifact(
        path, device, parent_model, geometry_model, geometry_artifact):
    """Load and provenance-check one frozen joint adapter."""
    from scripts.train_scanrefer_joint_box_mask import (
        load_joint_adapter_artifact,
    )
    model, artifact = load_joint_adapter_artifact(path, device=device)
    validate_rec_joint_box_mask_runtime_provenance(
        parent_model, geometry_model, geometry_artifact,
        model, artifact, device,
    )
    return model, artifact


def validate_rec_reranker_provenance(args, artifact):
    """Ensure an artifact was built for this backbone and input contract."""
    from scripts.cache_scanrefer_rec_candidates import checkpoint_sha256
    from scripts.train_rec_reranker import (
        BACKBONE_CONFIG_KEYS,
        normalize_backbone_config,
    )

    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise ValueError("reranker validation requires a backbone checkpoint")
    if checkpoint_sha256(checkpoint_path) != artifact.get("checkpoint_sha256"):
        raise ValueError("reranker backbone checkpoint fingerprint mismatch")
    expected_inputs = artifact.get("model_inputs")
    keys = (
        "use_color", "use_height", "use_multiview",
        "butd", "butd_gt", "butd_cls",
    )
    if not isinstance(expected_inputs, dict):
        raise ValueError("reranker artifact model inputs are invalid")
    mismatches = [
        key for key in keys
        if expected_inputs.get(key) is not bool(getattr(args, key, False))
    ]
    if mismatches:
        raise ValueError(
            "reranker model input mismatch: {}".format(
                ", ".join(mismatches)
            )
        )
    try:
        expected_config = normalize_backbone_config(
            artifact.get("backbone_config")
        )
    except ValueError:
        raise ValueError("reranker artifact model config is invalid")
    config_keys = BACKBONE_CONFIG_KEYS
    defaults = {
        "use_source_moe": False,
        "source_moe_shared_source": "default",
        "source_moe_top_k": 2,
        "source_moe_balance_loss_weight": 0.01,
        "source_moe_query_layers": 1,
        "source_moe_query_heads": 4,
        "source_moe_query_dropout": 0.1,
        "source_moe_query_max_delta": 0.25,
        "source_moe_use_fallback_gate": False,
        "source_moe_gate_hidden_dim": 128,
        "source_moe_gate_candidate_top_k": 8,
        "source_moe_gate_break_cost": 2.0,
        "source_moe_gate_decision_margin": 0.0,
        "source_moe_gate_mask_utility_weight": 0.25,
        "source_moe_gate_uncertainty_weight": 0.0,
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_action_mode": "decision",
    }
    config_mismatches = []
    for key in config_keys:
        runtime_value = getattr(args, key, defaults.get(key))
        if key == "source_moe_gate_action_mode" and runtime_value is None:
            runtime_value = defaults[key]
        if expected_config.get(key) != runtime_value:
            config_mismatches.append(key)
    if config_mismatches:
        raise ValueError(
            "reranker model config mismatch: {}".format(
                ", ".join(config_mismatches)
            )
        )


def _build_rec_reranker_outputs_float32(end_points, inputs, reranker,
                                        artifact):
    """Build the compact and full-query deployed parent score state."""
    if artifact.get("adapter_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("reranker artifact feature schema does not match adapter")
    candidate_rule = artifact.get("candidate_rule")
    if not isinstance(candidate_rule, dict):
        raise ValueError("reranker artifact candidate rule is invalid")
    topk_per_source = candidate_rule.get("topk_per_source")
    max_candidates = candidate_rule.get("max_candidates")
    if (not isinstance(topk_per_source, int) or topk_per_source <= 0
            or not isinstance(max_candidates, int) or max_candidates <= 0):
        raise ValueError("reranker artifact candidate rule is invalid")

    candidate_batch = build_rec_candidate_batch(
        end_points,
        inputs,
        topk_per_source=topk_per_source,
        max_candidates=max_candidates,
    )
    features = candidate_batch["features"]
    valid_mask = candidate_batch["valid_mask"]
    if list(candidate_batch["feature_names"]) != artifact.get("feature_names"):
        raise ValueError("reranker artifact feature names do not match adapter")
    if artifact.get("input_dim") != features.shape[-1]:
        raise ValueError("reranker artifact feature dimension does not match adapter")
    feature_mean = artifact.get("feature_mean")
    feature_std = artifact.get("feature_std")
    if (not isinstance(feature_mean, torch.Tensor)
            or not isinstance(feature_std, torch.Tensor)
            or feature_mean.shape != (features.shape[-1],)
            or feature_std.shape != (features.shape[-1],)
            or not torch.isfinite(feature_mean).all()
            or not torch.isfinite(feature_std).all()
            or (feature_std <= 0).any()):
        raise ValueError("reranker artifact feature normalization is invalid")
    feature_mean = feature_mean.to(features.device, features.dtype)
    feature_std = feature_std.to(features.device, features.dtype)
    normalized = (features - feature_mean) / feature_std
    normalized = torch.where(
        valid_mask.unsqueeze(-1), normalized, torch.zeros_like(normalized)
    )
    outputs = reranker(normalized.float(), valid_mask)
    ranking_logits = outputs.get("ranking_logits")
    if (not isinstance(ranking_logits, torch.Tensor)
            or ranking_logits.shape != valid_mask.shape):
        raise ValueError("reranker did not return candidate ranking logits")
    if artifact.get("score_mode") != "rank_blend":
        raise ValueError("reranker artifact score mode is unsupported")
    reranker_weight = artifact.get("reranker_weight")
    if (not isinstance(reranker_weight, (float, int))
            or isinstance(reranker_weight, bool)):
        raise ValueError("reranker artifact weight is invalid")
    compact_scores = blend_candidate_scores(
        candidate_batch["default_scores"],
        ranking_logits,
        valid_mask,
        reranker_weight=float(reranker_weight),
    )
    parent_state = build_deployed_parent_state(
        compact_scores,
        candidate_batch["query_indices"],
        valid_mask,
        candidate_batch["num_queries"],
    )
    return {
        "candidate_batch": candidate_batch,
        "compact_scores": compact_scores,
        "query_scores": parent_state["query_scores"],
    }


def build_rec_reranker_outputs(end_points, inputs, reranker, artifact):
    """Run the complete deployed parent score builder in float32 eval mode."""
    center = end_points.get("last_center")
    if not isinstance(center, torch.Tensor):
        raise ValueError("last_center is required for REC parent inference")
    reranker.eval().requires_grad_(False)
    with torch.no_grad(), _disabled_runtime_autocast(center.device):
        return _build_rec_reranker_outputs_float32(
            end_points, inputs, reranker, artifact
        )


def build_rec_reranker_scores(end_points, inputs, reranker, artifact):
    """Compatibility wrapper returning deployed full-query parent scores."""
    return build_rec_reranker_outputs(
        end_points, inputs, reranker, artifact
    )["query_scores"]


def _require_parent_runtime_outputs(parent_outputs):
    expected_keys = {
        "candidate_batch", "compact_scores", "query_scores",
    }
    if not isinstance(parent_outputs, dict) or set(parent_outputs) != expected_keys:
        raise ValueError("parent runtime outputs do not match exact schema")
    candidate_batch = parent_outputs["candidate_batch"]
    if not isinstance(candidate_batch, dict):
        raise ValueError("parent candidate_batch must be an object")
    required_candidate_keys = (
        "features", "feature_names", "query_indices", "valid_mask",
        "num_queries",
    )
    if any(key not in candidate_batch for key in required_candidate_keys):
        raise ValueError("parent candidate_batch is incomplete")
    parent_state = build_deployed_parent_state(
        parent_outputs["compact_scores"],
        candidate_batch["query_indices"],
        candidate_batch["valid_mask"],
        candidate_batch["num_queries"],
    )
    if not torch.equal(
            parent_state["query_scores"], parent_outputs["query_scores"]):
        raise ValueError("parent query scores differ from deployed compact state")
    return candidate_batch, parent_state


def _validate_geometry_artifact_runtime_schema(artifact, candidate_batch,
                                               geometry_batch):
    if not isinstance(artifact, dict):
        raise ValueError("geometry artifact must be an object")
    features = candidate_batch["features"]
    geometry_features = geometry_batch.get("geometry_features")
    geometry_valid = geometry_batch.get("valid_mask")
    geometry_boxes = geometry_batch.get("boxes")
    if (not isinstance(features, torch.Tensor) or features.dim() != 3
            or features.dtype != torch.float32
            or not isinstance(geometry_features, torch.Tensor)
            or geometry_features.dim() != 4
            or geometry_features.dtype != torch.float32
            or not isinstance(geometry_valid, torch.Tensor)
            or geometry_valid.dtype != torch.bool):
        raise ValueError("geometry runtime feature tensors are malformed")
    batch_size, candidates, base_dim = features.shape
    if candidates != 16 or candidate_batch.get("num_queries") != 256:
        raise ValueError("geometry runtime requires 16 candidates and 256 queries")
    if (geometry_features.shape[:2] != (batch_size, candidates)
            or geometry_features.shape[2:] != (7, 25)
            or geometry_valid.shape != (batch_size, candidates, 7)):
        raise ValueError("geometry runtime requires [B,16,7,25] features")
    if (not isinstance(geometry_boxes, torch.Tensor)
            or geometry_boxes.dtype != torch.float32
            or geometry_boxes.shape != (batch_size, candidates, 7, 6)):
        raise ValueError(
            "geometry boxes must have shape [B,16,7,6]"
        )
    if any(value.device != features.device for value in (
            geometry_features, geometry_valid, geometry_boxes)):
        raise ValueError("geometry boxes and features must share a device")
    expected_names = (
        list(candidate_batch["feature_names"])
        + list(geometry_batch.get("geometry_feature_names", ()))
        + ["parent_score", "parent_is_deployed_top1"]
    )
    if (base_dim != 152 or artifact.get("input_dim") != 179
            or artifact.get("feature_names") != expected_names
            or len(expected_names) != 179):
        raise ValueError("geometry artifact feature schema is incompatible")
    configs = artifact.get("variant_configs")
    if (not isinstance(configs, list)
            or geometry_batch.get("variant_names")
            != tuple(artifact.get("variant_names", ()))
            or list(geometry_batch.get("variant_configs", ())) != configs
            or geometry_batch.get("min_points") != artifact.get("min_points")
            or float(geometry_batch.get("max_point_fraction"))
            != float(artifact.get("max_point_fraction"))
            or artifact.get("regressed_variant_index") != 0):
        raise ValueError("geometry artifact variant schema is incompatible")
    return geometry_features, geometry_valid


def validate_rec_geometry_runtime_outputs(outputs):
    """Fail closed on malformed parent-axis or flat-axis attachments."""
    if not isinstance(outputs, dict):
        raise ValueError("geometry runtime outputs must be an object")
    mode = outputs.get("rec_geometry_runtime_mode")
    if mode == "parent_query_axis":
        if set(outputs) != {
                "rec_reranker_scores", "rec_geometry_runtime_mode"}:
            raise ValueError("parent query mode cannot carry geometry tensors")
        scores = outputs.get("rec_reranker_scores")
        if (not isinstance(scores, torch.Tensor) or scores.dim() != 2
                or scores.shape[1] != 256):
            raise ValueError("parent query scores must have shape [B,256]")
        if scores.dtype != torch.float32:
            raise ValueError("parent query scores must use float32")
        if (bool(torch.isnan(scores).any().item())
                or bool(torch.isposinf(scores).any().item())
                or not bool(torch.isfinite(scores).any(dim=1).all().item())):
            raise ValueError("parent query scores are invalid")
        return outputs
    if mode != "flat_geometry_axis":
        raise ValueError("geometry runtime mode is invalid")
    expected_keys = {
        "rec_reranker_scores",
        "rec_geometry_runtime_mode",
        "rec_geometry_boxes",
        "rec_geometry_scores",
        "rec_geometry_valid_mask",
        "rec_geometry_fallback_index",
    }
    joint_keys = {
        "rec_joint_selected_flat_index",
        "rec_joint_selected_parent_position",
        "rec_joint_mask_policy_index",
        "rec_joint_mask_source_index",
        "rec_joint_mask_threshold_index",
        "rec_joint_mask_threshold",
    }
    observed_joint_keys = set(outputs).intersection(joint_keys)
    if observed_joint_keys and observed_joint_keys != joint_keys:
        raise ValueError("joint mask runtime fields are incomplete")
    if set(outputs) not in (expected_keys, expected_keys | joint_keys):
        raise ValueError("flat geometry output fields do not match exact schema")
    boxes = outputs["rec_geometry_boxes"]
    scores = outputs["rec_geometry_scores"]
    valid = outputs["rec_geometry_valid_mask"]
    fallback = outputs["rec_geometry_fallback_index"]
    parent_scores = outputs["rec_reranker_scores"]
    if (not isinstance(boxes, torch.Tensor) or boxes.dim() != 3
            or boxes.shape[-1] != 6
            or not isinstance(scores, torch.Tensor) or scores.dim() != 2
            or tuple(scores.shape) != tuple(boxes.shape[:2])
            or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or tuple(valid.shape) != tuple(scores.shape)
            or not isinstance(fallback, torch.Tensor)
            or fallback.dtype != torch.long
            or tuple(fallback.shape) != (scores.shape[0],)
            or not isinstance(parent_scores, torch.Tensor)
            or parent_scores.dim() != 2):
        raise ValueError("flat geometry tensor shapes are invalid")
    if tuple(parent_scores.shape) != (scores.shape[0], 256):
        raise ValueError("parent query scores must have shape [B,256]")
    if (boxes.dtype != torch.float32 or scores.dtype != torch.float32
            or parent_scores.dtype != torch.float32):
        raise ValueError(
            "geometry boxes and scores must use float32"
        )
    tensors = (boxes, scores, valid, fallback, parent_scores)
    if any(value.device != scores.device for value in tensors):
        raise ValueError("geometry runtime tensors must share the same device")
    if scores.shape[1] != 112:
        raise ValueError("flat geometry axis must contain 112 candidates")
    nonempty_rows = valid.any(dim=1)
    if not bool(torch.isfinite(scores[valid]).all().item()):
        raise ValueError("valid geometry scores must be finite")
    if not bool(torch.isneginf(scores[~valid]).all().item()):
        raise ValueError("invalid geometry scores must be -inf")
    valid_boxes = boxes[valid]
    if not bool(torch.isfinite(valid_boxes).all().item()):
        raise ValueError("valid geometry boxes must be finite")
    if not bool(torch.isfinite(boxes).all().item()):
        raise ValueError("geometry boxes must be finite")
    if not bool((valid_boxes[:, 3:] > 0.0).all().item()):
        raise ValueError("valid geometry boxes must have positive size")
    if (bool((fallback < 0).any().item())
            or bool((fallback >= scores.shape[1]).any().item())):
        raise ValueError("geometry fallback index is out of range")
    fallback_valid = torch.gather(valid, 1, fallback.unsqueeze(1)).squeeze(1)
    if not bool(fallback_valid[nonempty_rows].all().item()):
        raise ValueError("geometry fallback index must identify a valid candidate")
    if (bool(torch.isnan(parent_scores).any().item())
            or bool(torch.isposinf(parent_scores).any().item())
            or not bool(torch.isfinite(parent_scores).any(dim=1).all().item())):
            raise ValueError("parent query scores are invalid")
    if observed_joint_keys:
        selected = outputs["rec_joint_selected_flat_index"]
        parent_position = outputs["rec_joint_selected_parent_position"]
        policy = outputs["rec_joint_mask_policy_index"]
        source = outputs["rec_joint_mask_source_index"]
        threshold_index = outputs["rec_joint_mask_threshold_index"]
        threshold = outputs["rec_joint_mask_threshold"]
        integer_tensors = (
            selected, parent_position, policy, source, threshold_index
        )
        if (any(not isinstance(value, torch.Tensor)
                or value.dtype != torch.long
                or tuple(value.shape) != (scores.shape[0],)
                or value.device != scores.device
                for value in integer_tensors)
                or not isinstance(threshold, torch.Tensor)
                or threshold.dtype != torch.float32
                or tuple(threshold.shape) != (scores.shape[0],)
                or threshold.device != scores.device
                or not bool(torch.isfinite(threshold).all().item())):
            raise ValueError("joint mask runtime tensor schema is invalid")
        from models.rec_joint_box_mask import (
            MASK_LOGIT_THRESHOLDS,
            MASK_POLICY_COUNT,
            MASK_SOURCE_NAMES,
        )
        if (bool((selected < 0).any().item())
                or bool((selected >= scores.shape[1]).any().item())
                or bool((parent_position < 0).any().item())
                or bool((parent_position >= 16).any().item())
                or bool((policy < 0).any().item())
                or bool((policy >= MASK_POLICY_COUNT).any().item())
                or bool((source < 0).any().item())
                or bool((source >= len(MASK_SOURCE_NAMES)).any().item())
                or bool((threshold_index < 0).any().item())
                or bool((threshold_index >= len(
                    MASK_LOGIT_THRESHOLDS
                )).any().item())):
            raise ValueError("joint mask runtime index is out of range")
        if not torch.equal(
                parent_position,
                torch.div(selected, 7, rounding_mode="floor")):
            raise ValueError("joint mask parent position does not match flat index")
        if not torch.equal(
                source,
                torch.div(
                    policy, len(MASK_LOGIT_THRESHOLDS),
                    rounding_mode="floor",
                )) or not torch.equal(
                    threshold_index,
                    torch.remainder(policy, len(MASK_LOGIT_THRESHOLDS))):
            raise ValueError("joint mask policy decomposition is inconsistent")
        expected_thresholds = torch.tensor(
            MASK_LOGIT_THRESHOLDS,
            dtype=threshold.dtype,
            device=threshold.device,
        )[threshold_index]
        if not torch.equal(threshold, expected_thresholds):
            raise ValueError("joint mask threshold payload is inconsistent")
        selected_valid = valid.gather(1, selected.unsqueeze(1)).squeeze(1)
        if not bool(selected_valid[nonempty_rows].all().item()):
            raise ValueError("joint mask selected flat candidate must be valid")
        stable_top = scores[nonempty_rows].argmax(dim=1)
        if not torch.equal(
                selected[nonempty_rows], stable_top):
            raise ValueError("joint mask selected flat index must match final score")
    return outputs


_REC_GEOMETRY_VARIANT_COUNT = 7
_RUNTIME_TARGET_FIELDS = frozenset({
    "candidate_ious",
    "geometry_ious",
    "threshold_labels",
    "gt_boxes",
    "gt_bboxes",
    "gt_masks",
    "center_label",
    "size_gts",
    "box_label_mask",
    "target_iou",
    "target_ious",
    "iou",
    "iou_targets",
})


def _is_runtime_target_field(key):
    if key in _RUNTIME_TARGET_FIELDS:
        return True
    lowered = str(key).lower()
    return (
        "iou" in lowered
        or lowered.startswith("gt_")
        or lowered.startswith("target")
        or lowered.endswith("_targets")
    )


def _reject_runtime_target_fields(value, label="runtime payload"):
    """Reject training-only target data before it reaches evaluation state."""
    if not isinstance(value, dict):
        return
    forbidden = sorted(
        key for key in value if _is_runtime_target_field(key)
    )
    if forbidden:
        raise ValueError(
            "{} cannot carry GT/IoU target fields: {}".format(
                label, ", ".join(forbidden)
            )
        )
    for key, nested in value.items():
        if isinstance(nested, dict):
            _reject_runtime_target_fields(nested, "{}.{}".format(label, key))


def build_rec_geometry_parent_query_indices(parent_outputs, runtime_outputs):
    """Build the flat-to-original-query mapping used by joint mask evaluation.

    The parent candidate adapter is the only source of query identity here. It
    is inference-only state; no ground-truth or IoU fields are accepted.
    """
    _reject_runtime_target_fields(parent_outputs, "parent runtime outputs")
    _reject_runtime_target_fields(runtime_outputs, "geometry runtime outputs")
    validate_rec_geometry_runtime_outputs(runtime_outputs)
    if runtime_outputs.get("rec_geometry_runtime_mode") != "flat_geometry_axis":
        raise ValueError(
            "joint box mask requires flat geometry runtime candidates"
        )

    candidate_batch, _ = _require_parent_runtime_outputs(parent_outputs)
    query_indices = candidate_batch.get("query_indices")
    candidate_valid = candidate_batch.get("valid_mask")
    flat_scores = runtime_outputs.get("rec_geometry_scores")
    if (not isinstance(query_indices, torch.Tensor)
            or query_indices.dtype != torch.long
            or query_indices.dim() != 2):
        raise ValueError(
            "runtime parent query mapping must use int64 query_indices"
        )
    if (not isinstance(candidate_valid, torch.Tensor)
            or candidate_valid.dtype != torch.bool
            or candidate_valid.shape != query_indices.shape):
        raise ValueError("runtime parent candidate validity has invalid shape")
    if (not isinstance(flat_scores, torch.Tensor)
            or flat_scores.dim() != 2
            or flat_scores.shape[0] != query_indices.shape[0]):
        raise ValueError("flat geometry scores and parent mapping have invalid shape")
    if query_indices.shape[1] != 16 or flat_scores.shape[1] != 112:
        raise ValueError(
            "joint box mask requires 16 parent candidates and 112 flat variants"
        )
    if flat_scores.shape[1] != query_indices.shape[1] * _REC_GEOMETRY_VARIANT_COUNT:
        raise ValueError("flat geometry variant axis is incompatible with parent mapping")
    if query_indices.device != flat_scores.device or candidate_valid.device != flat_scores.device:
        raise ValueError("runtime parent mapping tensors must share the geometry device")
    num_queries = candidate_batch.get("num_queries")
    if num_queries != 256:
        raise ValueError("runtime parent mapping requires 256 detector queries")
    if (bool((query_indices < 0).any().item())
            or bool((query_indices >= num_queries).any().item())):
        raise ValueError("runtime parent query mapping contains an out-of-range index")
    if not bool(candidate_valid.any(dim=1).all().item()):
        raise ValueError("runtime parent mapping has no valid candidate")
    flat_valid = runtime_outputs["rec_geometry_valid_mask"]
    expanded_parent_valid = candidate_valid.unsqueeze(-1).expand(
        -1, -1, _REC_GEOMETRY_VARIANT_COUNT
    ).reshape_as(flat_valid)
    if bool((flat_valid & ~expanded_parent_valid).any().item()):
        raise ValueError(
            "flat geometry validity cannot exceed parent candidate validity"
        )

    return query_indices.unsqueeze(-1).expand(
        -1, -1, _REC_GEOMETRY_VARIANT_COUNT
    ).reshape(query_indices.shape[0], -1).contiguous()


def _stable_runtime_top1_mask(scores, valid):
    if (not isinstance(scores, torch.Tensor)
            or not isinstance(valid, torch.Tensor)
            or scores.dtype != torch.float32
            or valid.dtype != torch.bool
            or scores.dim() != 2
            or scores.shape != valid.shape):
        raise ValueError("hierarchical Top-1 tensors are malformed")
    masked = scores.masked_fill(~valid, -float("inf"))
    best = masked.max(dim=1, keepdim=True).values
    best_mask = valid & masked.eq(best)
    if not bool(best_mask.any(dim=1).all().item()):
        raise ValueError("hierarchical Top-1 requires one valid maximum")
    positions = torch.arange(
        scores.shape[1], dtype=torch.long, device=scores.device
    ).unsqueeze(0).expand_as(scores)
    selected = positions.masked_fill(
        ~best_mask, scores.shape[1]
    ).min(dim=1).values
    return positions.eq(selected.unsqueeze(1)), selected


def _build_rec_hierarchical_runtime_batch(
        candidate_batch, parent_state, model_inputs, geometry_valid,
        learned_logits, flat_scores, geometry_artifact,
        hierarchical_artifact):
    from scripts.train_scanrefer_rec_hierarchical_reranker import (
        build_hierarchical_feature_names,
        normalize_hierarchical_batch,
    )

    if not isinstance(hierarchical_artifact, dict):
        raise ValueError("hierarchical artifact must be an object")
    expected_names = build_hierarchical_feature_names(
        model_inputs["feature_names"]
    )
    if hierarchical_artifact.get("feature_names") != expected_names:
        raise ValueError("hierarchical feature schema is incompatible")
    raw_features = model_inputs["features"].float()
    flat_valid = model_inputs["valid_mask"]
    batch_size = raw_features.shape[0]
    if (tuple(raw_features.shape) != (batch_size, 112, 179)
            or tuple(geometry_valid.shape) != (batch_size, 16, 7)
            or tuple(learned_logits.shape) != (batch_size, 112)
            or tuple(flat_scores.shape) != (batch_size, 112)):
        raise ValueError("hierarchical runtime geometry axes changed")
    expected_query_positions = torch.arange(
        16, dtype=torch.long, device=raw_features.device
    ).view(1, 16, 1).expand(batch_size, 16, 7).reshape(
        batch_size, 112
    )
    expected_variant_indices = torch.arange(
        7, dtype=torch.long, device=raw_features.device
    ).view(1, 1, 7).expand(batch_size, 16, 7).reshape(
        batch_size, 112
    )
    if (not torch.equal(
            model_inputs["query_positions"], expected_query_positions)
            or not torch.equal(
                model_inputs["variant_indices"], expected_variant_indices
            )):
        raise ValueError("hierarchical runtime candidate axes changed")

    variant_valid = geometry_valid
    query_valid = variant_valid.any(dim=2)
    regressed_variant = geometry_artifact.get("regressed_variant_index")
    if (regressed_variant != 0
            or not torch.equal(
                query_valid, parent_state["candidate_valid"])):
        raise ValueError("hierarchical query validity source changed")
    structured = raw_features.reshape(batch_size, 16, 7, 179)
    query_features = candidate_batch.get("features")
    if (not isinstance(query_features, torch.Tensor)
            or query_features.dtype != torch.float32
            or tuple(query_features.shape) != (batch_size, 16, 152)
            or query_features.device != raw_features.device):
        raise ValueError("hierarchical base query features are malformed")
    repeated_query_features = query_features.unsqueeze(2).expand(
        -1, -1, 7, -1
    )
    if not torch.equal(
            structured[..., :152][variant_valid],
            repeated_query_features[variant_valid]):
        raise ValueError("hierarchical query features differ by variant")
    variant_features = structured[..., 152:177]
    query_mask = query_valid.unsqueeze(-1)
    variant_mask = variant_valid.unsqueeze(-1)
    query_features = torch.where(
        query_mask, query_features, torch.zeros_like(query_features)
    )
    variant_features = torch.where(
        variant_mask, variant_features, torch.zeros_like(variant_features)
    )

    default_scores = candidate_batch.get("default_scores")
    parent_scores = parent_state["compact_scores"].float()
    query_indices = parent_state["query_indices"]
    if (not isinstance(default_scores, torch.Tensor)
            or default_scores.dtype != torch.float32
            or tuple(default_scores.shape) != (batch_size, 16)
            or default_scores.device != raw_features.device
            or not bool(torch.isfinite(
                default_scores[query_valid]
            ).all().item())):
        raise ValueError("hierarchical default score state is malformed")
    default_rank = _stable_masked_rank_normalize(
        default_scores, query_valid
    )
    parent_rank = _stable_masked_rank_normalize(
        parent_scores, query_valid
    )
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
        raise ValueError("hierarchical query Top-1 identity changed")
    query_aux_continuous = torch.stack((
        default_scores, default_rank, parent_scores, parent_rank
    ), dim=-1)
    query_aux_continuous = torch.where(
        query_mask,
        query_aux_continuous,
        torch.zeros_like(query_aux_continuous),
    )
    query_aux_binary = torch.stack(
        (default_top1, parent_top1), dim=-1
    ) & query_mask

    geometry_rank = _stable_masked_rank_normalize(
        learned_logits.float(), flat_valid
    )
    structured_geometry_scores = learned_logits.reshape(
        batch_size, 16, 7
    )
    structured_geometry_rank = geometry_rank.reshape(
        batch_size, 16, 7
    )
    variant_aux_continuous = torch.stack((
        structured_geometry_scores, structured_geometry_rank
    ), dim=-1)
    variant_aux_continuous = torch.where(
        variant_mask,
        variant_aux_continuous,
        torch.zeros_like(variant_aux_continuous),
    )
    geometry_top1, _geometry_indices = _stable_runtime_top1_mask(
        learned_logits.float(), flat_valid
    )
    baseline_top1, _baseline_indices = _stable_runtime_top1_mask(
        flat_scores, flat_valid
    )
    variant_aux_binary = torch.stack((
        geometry_top1.reshape(batch_size, 16, 7),
        baseline_top1.reshape(batch_size, 16, 7),
    ), dim=-1) & variant_mask
    raw_batch = {
        "query_features": query_features,
        "variant_features": variant_features,
        "query_aux_continuous": query_aux_continuous,
        "query_aux_binary": query_aux_binary,
        "variant_aux_continuous": variant_aux_continuous,
        "variant_aux_binary": variant_aux_binary,
        "query_valid": query_valid,
        "variant_valid": variant_valid,
    }
    normalized = normalize_hierarchical_batch(
        raw_batch, hierarchical_artifact.get("normalization")
    )
    if any(value.device != raw_features.device for value in normalized.values()):
        raise ValueError("hierarchical normalized tensors changed device")
    return normalized


def _build_rec_geometry_runtime_outputs_float32(
        end_points, inputs, parent_outputs, geometry_reranker,
        geometry_artifact, residual_model=None, residual_artifact=None,
        hierarchical_model=None, hierarchical_artifact=None,
        joint_model=None, joint_artifact=None):
    """Build the inference-only parent or flat geometry attachment payload."""
    candidate_batch, parent_state = _require_parent_runtime_outputs(
        parent_outputs
    )
    weight = geometry_artifact.get("geometry_weight")
    if (not isinstance(weight, (float, int)) or isinstance(weight, bool)):
        raise ValueError("geometry artifact weight is invalid")
    weight = float(weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("geometry artifact weight is invalid")
    if weight == 0.0:
        if (residual_model is not None or hierarchical_model is not None
                or joint_model is not None):
            raise ValueError(
                "optional geometry policies require the flat geometry axis"
            )
        return validate_rec_geometry_runtime_outputs({
            "rec_reranker_scores": parent_state["query_scores"],
            "rec_geometry_runtime_mode": "parent_query_axis",
        })

    variant_config = {
        "variants": geometry_artifact.get("variant_configs"),
        "min_points": geometry_artifact.get("min_points"),
        "max_point_fraction": geometry_artifact.get("max_point_fraction"),
    }
    geometry_batch = build_rec_mask_geometry_candidates(
        end_points, inputs, candidate_batch, variant_config=variant_config
    )
    geometry_features, raw_geometry_valid = (
        _validate_geometry_artifact_runtime_schema(
            geometry_artifact, candidate_batch, geometry_batch
        )
    )
    filter_non_gt_boxes = geometry_artifact.get(
        "filter_non_gt_boxes", False
    )
    if not isinstance(filter_non_gt_boxes, bool):
        raise ValueError("geometry evaluator filtering flag must be boolean")
    if filter_non_gt_boxes:
        detected_boxes = inputs.get("det_boxes")
        detected_valid = inputs.get("det_bbox_label_mask")
        if (not isinstance(detected_boxes, torch.Tensor)
                or not isinstance(detected_valid, torch.Tensor)):
            raise ValueError(
                "GT geometry filtering needs deployable detector inputs"
            )
        evaluator_geometry_valid = build_detector_overlap_valid(
            geometry_batch["boxes"],
            raw_geometry_valid,
            detected_boxes,
            detected_valid.bool(),
            iou_threshold=0.25,
        )
    else:
        evaluator_geometry_valid = raw_geometry_valid
    evaluator_nonempty = evaluator_geometry_valid.reshape(
        evaluator_geometry_valid.shape[0], -1
    ).any(dim=1)
    geometry_valid = torch.where(
        evaluator_nonempty.reshape(-1, 1, 1),
        evaluator_geometry_valid,
        raw_geometry_valid,
    )
    evaluator_query_valid = geometry_valid.any(dim=2)
    parent_state = build_deployed_parent_state(
        parent_state["compact_scores"],
        parent_state["query_indices"],
        evaluator_query_valid,
        parent_state["query_scores"].shape[1],
    )
    model_inputs = build_rec_geometry_model_inputs(
        candidate_batch["features"].float(),
        geometry_features.float(),
        parent_state["compact_scores"].float(),
        parent_state["parent_top1_mask"],
        geometry_valid,
        candidate_batch["feature_names"],
        geometry_batch["geometry_feature_names"],
    )
    if (tuple(model_inputs["features"].shape[1:]) != (112, 179)
            or list(model_inputs["feature_names"])
            != geometry_artifact["feature_names"]):
        raise ValueError("geometry model input schema differs from artifact")
    normalized = normalize_features(
        model_inputs["features"].float(),
        model_inputs["valid_mask"],
        geometry_artifact.get("feature_mean"),
        geometry_artifact.get("feature_std"),
    )
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("normalized geometry features must be finite")
    scorer_outputs = geometry_reranker(
        normalized.float(), model_inputs["valid_mask"]
    )
    learned_logits = (
        scorer_outputs.get("ranking_logits")
        if isinstance(scorer_outputs, dict) else None
    )
    if (not isinstance(learned_logits, torch.Tensor)
            or learned_logits.shape != model_inputs["valid_mask"].shape):
        raise ValueError("geometry scorer ranking logits are malformed")
    blended = blend_rec_geometry_scores(
        parent_state,
        learned_logits.float(),
        geometry_valid,
        weight,
        geometry_artifact["regressed_variant_index"],
    )
    if blended.get("use_parent_query_axis") is not False:
        raise RuntimeError("nonzero geometry weight did not build a flat axis")
    flat_valid = blended["flat_valid_mask"]
    flat_scores = blended["flat_scores"]
    if residual_model is not None:
        from scripts.train_scanrefer_rec_selective_residual import (
            build_selective_pair_feature_names,
        )

        expected_pair_names = build_selective_pair_feature_names(
            model_inputs["feature_names"]
        )
        if (residual_artifact.get("input_dim") != 185
                or residual_artifact.get("feature_names")
                != expected_pair_names):
            raise ValueError("residual feature schema is incompatible")
        selection = residual_artifact.get("selection")
        margin = (
            selection.get("margin") if isinstance(selection, dict) else None
        )
        if (not isinstance(margin, (float, int))
                or isinstance(margin, bool)
                or not math.isfinite(float(margin))
                or float(margin) <= 0.0):
            raise ValueError("residual artifact margin is invalid")
        threshold_logits = (
            scorer_outputs.get("threshold_logits")
            if isinstance(scorer_outputs, dict) else None
        )
        iou_estimate = (
            scorer_outputs.get("iou_estimate")
            if isinstance(scorer_outputs, dict) else None
        )
        if (not isinstance(threshold_logits, torch.Tensor)
                or threshold_logits.dtype != torch.float32
                or tuple(threshold_logits.shape)
                != tuple(flat_valid.shape) + (2,)
                or not isinstance(iou_estimate, torch.Tensor)
                or iou_estimate.dtype != torch.float32
                or tuple(iou_estimate.shape) != tuple(flat_valid.shape)
                or threshold_logits.device != flat_valid.device
                or iou_estimate.device != flat_valid.device
                or not bool(torch.isfinite(
                    threshold_logits[flat_valid]
                ).all().item())
                or not bool(torch.isfinite(
                    iou_estimate[flat_valid]
                ).all().item())):
            raise ValueError("geometry residual heads are malformed")
        parent_prior = build_flat_parent_prior(
            parent_state,
            geometry_valid,
            geometry_artifact["regressed_variant_index"],
        )
        parent_rank = _stable_masked_rank_normalize(
            parent_prior, flat_valid
        )
        geometry_rank = _stable_masked_rank_normalize(
            learned_logits.float(), flat_valid
        )
        reconstructed = (
            (1.0 - weight) * parent_rank + weight * geometry_rank
        ).masked_fill(~flat_valid, -float("inf"))
        if not torch.equal(reconstructed, flat_scores):
            raise RuntimeError("residual frozen geometry score reconstruction drifted")
        baseline_indices = flat_scores.argmax(dim=1)
        pair = build_selective_pair_features(
            normalized_features=normalized.float(),
            valid_mask=flat_valid,
            baseline_indices=baseline_indices,
            parent_rank=parent_rank,
            geometry_rank=geometry_rank,
            threshold_logits=threshold_logits.float(),
            iou_estimate=iou_estimate.float(),
            query_positions=model_inputs["query_positions"],
        )
        residual_logits = residual_model(
            pair["features"], pair["valid_mask"]
        )
        if (not isinstance(residual_logits, torch.Tensor)
                or residual_logits.dtype != torch.float32
                or tuple(residual_logits.shape)
                != tuple(flat_valid.shape) + (2, 3)
                or residual_logits.device != flat_valid.device
                or not bool(torch.isfinite(residual_logits).all().item())):
            raise ValueError("residual model logits are malformed")
        pair_gain = expected_selective_gain(residual_logits)
        policy = apply_selective_policy(
            flat_scores,
            pair_gain,
            pair["valid_mask"],
            float(margin),
        )
        flat_scores = policy["scores"]
    elif hierarchical_model is not None:
        hierarchy_batch = _build_rec_hierarchical_runtime_batch(
            candidate_batch,
            parent_state,
            model_inputs,
            geometry_valid,
            learned_logits.float(),
            flat_scores,
            geometry_artifact,
            hierarchical_artifact,
        )
        is_v113_committee = (
            hierarchical_artifact.get("schema") == V113_ARTIFACT_SCHEMA
        )
        is_pareto_contextual = (
            hierarchical_artifact.get("schema")
            in PARETO_CONTEXTUAL_ARTIFACT_SCHEMAS
        )
        if is_v113_committee:
            committee_policy = hierarchical_artifact.get("policy", {})
            margin = committee_policy.get("aggregate_lcb_margin")
        elif is_pareto_contextual:
            pareto_policy = hierarchical_artifact.get("policy", {})
            margin = pareto_policy.get("aggregate_margin")
            min_head_gain025 = pareto_policy.get(
                "min_head_gain025", 0.0
            )
            min_head_gain050 = pareto_policy.get(
                "min_head_gain050", 0.0
            )
        else:
            selection = hierarchical_artifact.get("selection")
            margin = (
                selection.get("margin")
                if isinstance(selection, dict) else None
            )
        if (not isinstance(margin, (float, int))
                or isinstance(margin, bool)
                or not math.isfinite(float(margin))
                or float(margin) <= 0.0):
            raise ValueError("hierarchical artifact margin is invalid")
        hierarchy_outputs = hierarchical_model(**hierarchy_batch)
        if is_v113_committee:
            if (not isinstance(hierarchy_outputs, dict)
                    or set(hierarchy_outputs) != {
                        "member_query_logits", "member_variant_logits",
                    }):
                raise ValueError("V113 committee outputs are malformed")
            policy = apply_asymmetric_risk_contextual_policy(
                flat_scores,
                hierarchy_outputs["member_query_logits"],
                hierarchy_outputs["member_variant_logits"],
                hierarchy_batch["query_valid"],
                hierarchy_batch["variant_valid"],
                float(margin),
                min_head_lcb025=committee_policy.get("min_head_lcb025"),
                min_head_lcb050=committee_policy.get("min_head_lcb050"),
                risk_lambda025=committee_policy.get("risk_lambda025"),
                risk_lambda050=committee_policy.get("risk_lambda050"),
            )
        else:
            if (not isinstance(hierarchy_outputs, dict)
                    or set(hierarchy_outputs) != {
                        "query_logits", "variant_logits", "query_embedding",
                        "variant_embedding",
                    }):
                raise ValueError("hierarchical model outputs are malformed")
        if is_pareto_contextual:
            policy = apply_pareto_contextual_policy(
                flat_scores,
                hierarchy_outputs["query_logits"],
                hierarchy_outputs["variant_logits"],
                hierarchy_batch["query_valid"],
                hierarchy_batch["variant_valid"],
                float(margin),
                min_head_gain025=min_head_gain025,
                min_head_gain050=min_head_gain050,
            )
        elif not is_v113_committee:
            proposal = select_hierarchical_proposal(
                hierarchy_outputs["query_logits"],
                hierarchy_outputs["variant_logits"],
                hierarchy_batch["query_valid"],
                hierarchy_batch["variant_valid"],
            )
            flat_utility = proposal["variant_utility"].reshape(
                flat_scores.shape[0], -1
            )
            rows = torch.arange(
                flat_scores.shape[0], device=flat_scores.device
            )
            baseline_indices = flat_scores.argmax(dim=1)
            proposal_indices = proposal["flat_indices"]
            predicted_gain = (
                flat_utility[rows, proposal_indices]
                - flat_utility[rows, baseline_indices]
            )
            policy = apply_hierarchical_policy(
                flat_scores,
                proposal_indices,
                predicted_gain.float(),
                hierarchy_batch["variant_valid"],
                float(margin),
            )
        flat_scores = policy["scores"]
    joint_policy = None
    if (joint_model is None) != (joint_artifact is None):
        raise ValueError("REC joint adapter has partial runtime state")
    if joint_model is not None:
        joint_policy = apply_rec_joint_box_mask_runtime_policy(
            model_inputs["features"].float(),
            flat_valid,
            flat_scores.float(),
            joint_model,
            joint_artifact,
        )
        flat_scores = joint_policy["scores"]
    fallback_positions = []
    fallback_variants = []
    regressed_variant = int(geometry_artifact["regressed_variant_index"])
    variant_priority = [regressed_variant] + [
        index for index in range(geometry_valid.shape[2])
        if index != regressed_variant
    ]
    for row_mask in parent_state["parent_top1_mask"]:
        positions = row_mask.nonzero(as_tuple=False).reshape(-1)
        if positions.numel() != 1:
            raise ValueError("canonical parent Top-1 needs one compact position")
        fallback_positions.append(int(positions[0].item()))
    for batch_index, compact_position in enumerate(fallback_positions):
        valid_variants = geometry_valid[batch_index, compact_position]
        fallback_variants.append(next(
            index for index in variant_priority
            if bool(valid_variants[index].item())
        ))
    fallback = torch.tensor(
        fallback_positions,
        dtype=torch.long,
        device=flat_valid.device,
    ) * geometry_valid.shape[2] + torch.tensor(
        fallback_variants,
        dtype=torch.long,
        device=flat_valid.device,
    )
    flat_valid = evaluator_geometry_valid.reshape(
        evaluator_geometry_valid.shape[0], -1
    )
    flat_scores = flat_scores.masked_fill(~flat_valid, -float("inf"))
    outputs = {
        "rec_reranker_scores": parent_state["query_scores"],
        "rec_geometry_runtime_mode": "flat_geometry_axis",
        "rec_geometry_boxes": geometry_batch["boxes"].reshape(
            geometry_valid.shape[0], -1, 6
        ),
        "rec_geometry_scores": flat_scores,
        "rec_geometry_valid_mask": flat_valid,
        "rec_geometry_fallback_index": fallback,
    }
    if joint_policy is not None:
        outputs.update({
            "rec_joint_selected_flat_index": joint_policy[
                "selected_flat_indices"
            ],
            "rec_joint_selected_parent_position": joint_policy[
                "selected_parent_positions"
            ],
            "rec_joint_mask_policy_index": joint_policy[
                "selected_mask_policy_indices"
            ],
            "rec_joint_mask_source_index": joint_policy[
                "selected_mask_source_indices"
            ],
            "rec_joint_mask_threshold_index": joint_policy[
                "selected_mask_threshold_indices"
            ],
            "rec_joint_mask_threshold": joint_policy[
                "selected_mask_thresholds"
            ],
        })
    return validate_rec_geometry_runtime_outputs(outputs)


def build_rec_geometry_runtime_outputs(
        end_points, inputs, parent_outputs, geometry_reranker,
        geometry_artifact, residual_model=None, residual_artifact=None,
        hierarchical_model=None, hierarchical_artifact=None,
        joint_model=None, joint_artifact=None):
    """Run the complete flat-geometry builder in float32 inference mode."""
    if (residual_model is None) != (residual_artifact is None):
        raise ValueError("REC geometry runtime has partial residual context")
    if (hierarchical_model is None) != (hierarchical_artifact is None):
        raise ValueError("REC geometry runtime has partial hierarchy context")
    if residual_model is not None and hierarchical_model is not None:
        raise ValueError(
            "selective residual and hierarchy are mutually exclusive"
        )
    if (joint_model is None) != (joint_artifact is None):
        raise ValueError("REC joint adapter has partial runtime state")
    if joint_model is not None and (
            residual_model is not None or hierarchical_model is not None):
        raise ValueError(
            "joint adapter cannot be combined with residual or hierarchy"
        )
    compact_scores = (
        parent_outputs.get("compact_scores")
        if isinstance(parent_outputs, dict) else None
    )
    device = (
        compact_scores.device
        if isinstance(compact_scores, torch.Tensor) else torch.device("cpu")
    )
    geometry_reranker.eval().requires_grad_(False)
    if residual_model is not None:
        residual_model.eval().requires_grad_(False)
    if hierarchical_model is not None:
        hierarchical_model.eval().requires_grad_(False)
    with torch.no_grad(), _disabled_runtime_autocast(device):
        return _build_rec_geometry_runtime_outputs_float32(
            end_points,
            inputs,
            parent_outputs,
            geometry_reranker,
            geometry_artifact,
            residual_model,
            residual_artifact,
            hierarchical_model,
            hierarchical_artifact,
            joint_model,
            joint_artifact,
        )

class TrainTester(BaseTrainTester):
    """Train/test a language grounder."""

    # logger.
    def __init__(self, args):
        """Initialize."""
        super().__init__(args)
        self.rec_reranker = None
        self.rec_reranker_artifact = None
        self.rec_geometry_reranker = None
        self.rec_geometry_reranker_artifact = None
        self._rec_geometry_runtime_projection = None
        self.rec_selective_residual = None
        self.rec_selective_residual_artifact = None
        self.rec_hierarchical_reranker = None
        self.rec_hierarchical_reranker_artifact = None
        self.rec_joint_box_mask = None
        self.rec_joint_box_mask_artifact = None

    def _ensure_rec_geometry_runtime_loaded(self, args, device):
        state = (
            getattr(self, "rec_reranker", None),
            getattr(self, "rec_reranker_artifact", None),
            getattr(self, "rec_geometry_reranker", None),
            getattr(self, "rec_geometry_reranker_artifact", None),
        )
        present = tuple(value is not None for value in state)
        if any(present) and not all(present):
            raise ValueError("REC geometry runtime has partial artifact state")
        if not any(present):
            parent_path = getattr(args, "rec_reranker_checkpoint", None)
            geometry_path = getattr(
                args, "rec_geometry_reranker_checkpoint", None
            )
            if not parent_path or not geometry_path:
                raise ValueError(
                    "--eval_use_rec_geometry_reranker_scores requires both "
                    "--rec_reranker_checkpoint and "
                    "--rec_geometry_reranker_checkpoint"
                )
            loaded = load_rec_geometry_runtime_artifacts(
                parent_path, geometry_path, device=device
            )
            projection = validate_rec_geometry_runtime_provenance(
                args, loaded[0], loaded[1], loaded[2], loaded[3], device
            )
            self.rec_reranker = loaded[0]
            self.rec_reranker_artifact = loaded[1]
            self.rec_geometry_reranker = loaded[2]
            self.rec_geometry_reranker_artifact = loaded[3]
            self._rec_geometry_runtime_projection = projection
        elif getattr(self, "_rec_geometry_runtime_projection", None) is None:
            self._rec_geometry_runtime_projection = (
                validate_rec_geometry_runtime_provenance(
                    args, state[0], state[1], state[2], state[3], device
                )
            )
        return self._rec_geometry_runtime_projection

    def _ensure_rec_joint_box_mask_runtime_loaded(self, args, device):
        state = (
            getattr(self, "rec_joint_box_mask", None),
            getattr(self, "rec_joint_box_mask_artifact", None),
        )
        if (state[0] is None) != (state[1] is None):
            raise ValueError("REC joint adapter has partial artifact state")
        if state[0] is None:
            path = getattr(args, "rec_joint_box_mask_checkpoint", None)
            if not path:
                raise ValueError(
                    "--eval_use_rec_joint_box_mask requires "
                    "--rec_joint_box_mask_checkpoint"
                )
            geometry_state = (
                getattr(self, "rec_reranker", None),
                getattr(self, "rec_geometry_reranker", None),
                getattr(self, "rec_geometry_reranker_artifact", None),
            )
            if any(value is None for value in geometry_state):
                raise ValueError(
                    "joint adapter requires loaded parent and geometry"
                )
            state = load_rec_joint_box_mask_runtime_artifact(
                path,
                device,
                parent_model=geometry_state[0],
                geometry_model=geometry_state[1],
                geometry_artifact=geometry_state[2],
            )
            self.rec_joint_box_mask = state[0]
            self.rec_joint_box_mask_artifact = state[1]
        return state

    def _ensure_rec_selective_residual_runtime_loaded(self, args, device):
        state = (
            getattr(self, "rec_selective_residual", None),
            getattr(self, "rec_selective_residual_artifact", None),
        )
        if (state[0] is None) != (state[1] is None):
            raise ValueError("REC selective residual has partial artifact state")
        if state[0] is None:
            path = getattr(
                args, "rec_selective_residual_checkpoint", None
            )
            if not path:
                raise ValueError(
                    "--eval_use_rec_selective_residual_scores requires "
                    "--rec_selective_residual_checkpoint"
                )
            geometry_state = (
                getattr(self, "rec_reranker", None),
                getattr(self, "rec_geometry_reranker", None),
                getattr(self, "rec_geometry_reranker_artifact", None),
            )
            if any(value is None for value in geometry_state):
                raise ValueError(
                    "selective residual requires loaded parent and geometry"
                )
            state = load_rec_selective_residual_runtime_artifact(
                path,
                device,
                parent_model=geometry_state[0],
                geometry_model=geometry_state[1],
                geometry_artifact=geometry_state[2],
            )
            self.rec_selective_residual = state[0]
            self.rec_selective_residual_artifact = state[1]
        return state

    def _ensure_rec_hierarchical_runtime_loaded(self, args, device):
        state = (
            getattr(self, "rec_hierarchical_reranker", None),
            getattr(self, "rec_hierarchical_reranker_artifact", None),
        )
        if (state[0] is None) != (state[1] is None):
            raise ValueError("REC hierarchy has partial artifact state")
        geometry_state = (
            getattr(self, "rec_reranker", None),
            getattr(self, "rec_geometry_reranker", None),
            getattr(self, "rec_geometry_reranker_artifact", None),
        )
        if any(value is None for value in geometry_state):
            raise ValueError("hierarchical reranker requires parent and geometry")
        if state[0] is None:
            path = getattr(
                args, "rec_hierarchical_reranker_checkpoint", None
            )
            if not path:
                raise ValueError(
                    "--eval_use_rec_hierarchical_reranker_scores requires "
                    "--rec_hierarchical_reranker_checkpoint"
                )
            state = load_rec_hierarchical_runtime_artifact(
                path,
                device,
                parent_model=geometry_state[0],
                geometry_model=geometry_state[1],
                geometry_artifact=geometry_state[2],
            )
            self.rec_hierarchical_reranker = state[0]
            self.rec_hierarchical_reranker_artifact = state[1]
        validate_rec_hierarchical_runtime_provenance(
            geometry_state[0],
            geometry_state[1],
            geometry_state[2],
            state[0],
            state[1],
            device,
        )
        return state

    def _attach_rec_reranker_scores(
            self, end_points, inputs, args, *, batch_idx=None,
            num_batches=None):
        """Load the frozen reranker once and attach full-query scores."""
        use_geometry = bool(getattr(
            args, "eval_use_rec_geometry_reranker_scores", False
        ))
        use_residual = bool(getattr(
            args, "eval_use_rec_selective_residual_scores", False
        ))
        use_hierarchy = bool(getattr(
            args, "eval_use_rec_hierarchical_reranker_scores", False
        ))
        use_joint_mask = bool(getattr(
            args, "eval_use_rec_joint_box_mask", False
        ))
        if use_joint_mask and not use_geometry:
            raise ValueError(
                "--eval_use_rec_joint_box_mask requires flat geometry "
                "reranker scores"
            )
        if use_joint_mask and (use_residual or use_hierarchy):
            raise ValueError(
                "joint box-mask adapter cannot be combined with residual "
                "or hierarchy scores"
            )
        if use_joint_mask and not getattr(
                args, "eval_use_rec_reranker_scores", False):
            raise ValueError(
                "joint box-mask adapter requires parent reranker scores"
            )
        if use_joint_mask and not getattr(
                args, "rec_joint_box_mask_checkpoint", None):
            raise ValueError(
                "joint box-mask adapter requires its checkpoint"
            )
        if use_residual and use_hierarchy:
            raise ValueError(
                "selective residual and hierarchy are mutually exclusive"
            )
        if use_residual:
            if not getattr(args, "eval_use_rec_reranker_scores", False):
                raise ValueError(
                    "selective residual requires parent reranker scores"
                )
            if not use_geometry:
                raise ValueError(
                    "selective residual requires geometry reranker scores"
                )
            for name in (
                    "rec_reranker_checkpoint",
                    "rec_geometry_reranker_checkpoint",
                    "rec_selective_residual_checkpoint"):
                if not getattr(args, name, None):
                    raise ValueError(
                        "selective residual requires all three checkpoints"
                    )
        if use_hierarchy:
            if not getattr(args, "eval_use_rec_reranker_scores", False):
                raise ValueError(
                    "hierarchical reranker requires parent reranker scores"
                )
            if not use_geometry:
                raise ValueError(
                    "hierarchical reranker requires geometry reranker scores"
                )
            for name in (
                    "rec_reranker_checkpoint",
                    "rec_geometry_reranker_checkpoint",
                    "rec_hierarchical_reranker_checkpoint"):
                if not getattr(args, name, None):
                    raise ValueError(
                        "hierarchical reranker requires all three checkpoints"
                    )
        if (not getattr(args, "eval_use_rec_reranker_scores", False)
                and not use_geometry and not use_residual
                and not use_hierarchy):
            return
        device = end_points["last_center"].device
        if use_geometry:
            actual_batch_size = int(end_points["last_center"].shape[0])
            validate_rec_geometry_runtime_environment(
                args,
                actual_batch_size,
                device,
                batch_idx=batch_idx,
                num_batches=num_batches,
            )
            projection = self._ensure_rec_geometry_runtime_loaded(args, device)
            residual_state = (None, None)
            hierarchy_state = (None, None)
            joint_state = (None, None)
            if use_residual:
                residual_state = (
                    self._ensure_rec_selective_residual_runtime_loaded(
                        args, device
                    )
                )
            if use_hierarchy:
                hierarchy_state = (
                    self._ensure_rec_hierarchical_runtime_loaded(
                        args, device
                    )
                )
            if use_joint_mask:
                joint_state = self._ensure_rec_joint_box_mask_runtime_loaded(
                    args, device
                )
            with _runtime_tf32(device, projection["allow_tf32"]):
                parent_outputs = build_rec_reranker_outputs(
                    end_points,
                    inputs,
                    self.rec_reranker,
                    self.rec_reranker_artifact,
                )
                runtime_outputs = build_rec_geometry_runtime_outputs(
                    end_points,
                    inputs,
                    parent_outputs,
                    self.rec_geometry_reranker,
                    self.rec_geometry_reranker_artifact,
                    residual_model=residual_state[0],
                    residual_artifact=residual_state[1],
                    hierarchical_model=hierarchy_state[0],
                    hierarchical_artifact=hierarchy_state[1],
                    joint_model=joint_state[0],
                    joint_artifact=joint_state[1],
                )
            # The runtime builder is validated before this inference-only
            # identity attachment.  This keeps target/IoU data outside the
            # runtime contract and makes parent-axis mode fail closed.
            if use_joint_mask:
                runtime_outputs = dict(runtime_outputs)
                runtime_outputs[
                    "rec_geometry_parent_query_indices"
                ] = build_rec_geometry_parent_query_indices(
                    parent_outputs, runtime_outputs
                )
            end_points.update(runtime_outputs)
            return
        parent_state = (
            getattr(self, "rec_reranker", None),
            getattr(self, "rec_reranker_artifact", None),
        )
        if (parent_state[0] is None) != (parent_state[1] is None):
            raise ValueError("REC parent runtime has partial artifact state")
        if parent_state[0] is None:
            checkpoint = getattr(args, "rec_reranker_checkpoint", None)
            if not checkpoint:
                raise ValueError(
                    "--eval_use_rec_reranker_scores requires "
                    "--rec_reranker_checkpoint"
                )
            from scripts.train_rec_reranker import load_reranker_artifact
            device = end_points["last_center"].device
            self.rec_reranker, self.rec_reranker_artifact = (
                load_reranker_artifact(checkpoint, device=device)
            )
            validate_rec_reranker_provenance(
                args, self.rec_reranker_artifact
            )
        end_points["rec_reranker_scores"] = build_rec_reranker_scores(
            end_points,
            inputs,
            self.rec_reranker,
            self.rec_reranker_artifact,
        )

    def _build_grounding_evaluator(self, args, prefixes):
        return GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1, 5, 10],
            prefixes=prefixes,
            filter_non_gt_boxes=args.butd_cls,
            logger=self.logger,
            model=args.model,
            eval_use_selector_choice_scores=(
                args.eval_use_selector_choice_scores
            ),
            eval_use_rec_reranker_scores=(
                args.eval_use_rec_reranker_scores
            ),
            eval_use_rec_geometry_reranker_scores=(
                args.eval_use_rec_geometry_reranker_scores
                or getattr(
                    args, "eval_use_rec_selective_residual_scores", False
                ) or getattr(
                    args, "eval_use_rec_hierarchical_reranker_scores", False
                )
            ),
            eval_use_rec_joint_box_mask=getattr(
                args, "eval_use_rec_joint_box_mask", False
            ),
        )

    # BRIEF Initialize dataset.
    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""

        dataset_dict = {}  # dict to use multiple datasets
        for dset in args.dataset:
            dataset_dict[dset] = 1
        if args.joint_det:
            dataset_dict['scannet'] = 10
        print('Loading datasets:', sorted(list(dataset_dict.keys())))

        if bool(getattr(
                args,
                'density_aware_target_box_scene_disjoint_audit',
                False,
        )):
            if args.eval or args.debug or args.eval_train:
                raise ValueError(
                    "density scene audit is a train-lifecycle audit mode"
                )
            if list(args.dataset) != ['nr3d'] or args.test_dataset != 'nr3d':
                raise ValueError("density scene audit requires Nr3D only")
            if not args.joint_det or not args.butd_cls:
                raise ValueError(
                    "density scene audit preserves joint_det+butd_cls"
                )
            if args.butd or args.butd_gt:
                raise ValueError("density scene audit rejects butd/butd_gt")
            role = getattr(
                args, 'density_aware_target_box_scene_disjoint_role', None
            )
            if role not in ('parent', 'control', 'method'):
                raise ValueError("density scene audit role is invalid")
            fold = int(getattr(
                args, 'density_aware_target_box_scene_disjoint_fold', -1
            ))
            if fold != 2:
                raise ValueError("density scene audit is preregistered to fold 2")
            expected_counts = {
                "fit_scenes": getattr(
                    args,
                    'density_aware_target_box_scene_disjoint_expected_fit_scenes',
                    -1,
                ),
                "holdout_scenes": getattr(
                    args,
                    'density_aware_target_box_scene_disjoint_expected_holdout_scenes',
                    -1,
                ),
                "fit_samples": getattr(
                    args,
                    'density_aware_target_box_scene_disjoint_expected_fit_samples',
                    -1,
                ),
                "holdout_samples": getattr(
                    args,
                    'density_aware_target_box_scene_disjoint_expected_holdout_samples',
                    -1,
                ),
            }
            base_dataset = Joint3DDataset(
                dataset_dict={'nr3d': 1},
                test_dataset='nr3d',
                split='train',
                use_color=args.use_color,
                use_height=args.use_height,
                overfit=False,
                data_path=args.data_root,
                detect_intermediate=args.detect_intermediate,
                use_multiview=args.use_multiview,
                butd=args.butd,
                butd_gt=args.butd_gt,
                butd_cls=args.butd_cls,
                augment_det=False,
                skip_missing_superpoints=args.skip_missing_superpoints,
                use_sacr_source=False,
                legacy_scene_graph_cache_path='',
                legacy_scene_graph_cache_strict=False,
                legacy_scene_graph_cache_expected_target_selection='',
                legacy_scene_graph_cache_expected_sha256='',
            )
            train_dataset, test_dataset, metadata = (
                build_density_target_box_scene_dataset_views(
                    base_dataset, fold, expected_counts
                )
            )
            args.density_aware_target_box_scene_disjoint_split_metadata = (
                metadata
            )
            holdout_samples = metadata['holdout_samples']
            if (
                    args.expected_eval_sample_count is not None
                    and args.expected_eval_sample_count != holdout_samples):
                raise ValueError(
                    "density held-out expected sample count drifted"
                )
            args.expected_eval_sample_count = holdout_samples
            print(
                "Density Nr3D scene-disjoint fold {} role {}: fit={}/{}; "
                "holdout={}/{}; overlap=0; dataset_scope=nr3d-only".format(
                    fold,
                    role,
                    metadata['fit_samples'], metadata['fit_scenes'],
                    metadata['holdout_samples'], metadata['holdout_scenes'],
                )
            )
            return train_dataset, test_dataset

        if bool(getattr(args, 'fpr_scene_disjoint_audit', False)):
            if args.eval or args.debug or args.eval_train:
                raise ValueError(
                    "FPR scene-disjoint audit is a train-then-holdout mode"
                )
            if list(args.dataset) != ['nr3d'] or args.test_dataset != 'nr3d':
                raise ValueError(
                    "FPR scene-disjoint audit requires Nr3D only"
                )
            if not args.joint_det or not args.butd_cls:
                raise ValueError(
                    "FPR scene-disjoint audit preserves joint_det+butd_cls"
                )
            if args.butd or args.butd_gt:
                raise ValueError(
                    "FPR scene-disjoint audit rejects butd/butd_gt"
                )
            fold = int(getattr(args, 'fpr_scene_disjoint_fold', -1))
            expected_counts = {
                "fit_scenes": getattr(
                    args, 'fpr_scene_disjoint_expected_fit_scenes', -1
                ),
                "holdout_scenes": getattr(
                    args, 'fpr_scene_disjoint_expected_holdout_scenes', -1
                ),
                "fit_samples": getattr(
                    args, 'fpr_scene_disjoint_expected_fit_samples', -1
                ),
                "holdout_samples": getattr(
                    args, 'fpr_scene_disjoint_expected_holdout_samples', -1
                ),
            }
            base_dataset = Joint3DDataset(
                dataset_dict={'nr3d': 1},
                test_dataset='nr3d',
                split='train',
                use_color=args.use_color,
                use_height=args.use_height,
                overfit=False,
                data_path=args.data_root,
                detect_intermediate=args.detect_intermediate,
                use_multiview=args.use_multiview,
                butd=args.butd,
                butd_gt=args.butd_gt,
                butd_cls=args.butd_cls,
                augment_det=False,
                skip_missing_superpoints=args.skip_missing_superpoints,
                use_sacr_source=True,
                legacy_scene_graph_cache_path=getattr(
                    args, 'legacy_scene_graph_cache', ''),
                legacy_scene_graph_cache_strict=getattr(
                    args, 'legacy_scene_graph_cache_strict', False),
                legacy_scene_graph_cache_expected_target_selection=getattr(
                    args,
                    'legacy_scene_graph_cache_expected_target_selection',
                    '',
                ),
                legacy_scene_graph_cache_expected_sha256=getattr(
                    args, 'legacy_scene_graph_cache_expected_sha256', ''),
            )
            train_dataset, test_dataset, metadata = (
                build_fpr_scene_disjoint_dataset_views(
                    base_dataset, fold, expected_counts
                )
            )
            args.fpr_scene_disjoint_split_metadata = metadata
            holdout_samples = metadata['holdout_samples']
            if (
                    args.expected_eval_sample_count is not None
                    and args.expected_eval_sample_count != holdout_samples):
                raise ValueError(
                    "FPR held-out expected sample count drifted"
                )
            args.expected_eval_sample_count = holdout_samples
            print(
                "FPR Nr3D scene-disjoint fold {}: fit={}/{}; "
                "holdout={}/{}; overlap=0".format(
                    fold,
                    metadata['fit_samples'], metadata['fit_scenes'],
                    metadata['holdout_samples'],
                    metadata['holdout_scenes'],
                )
            )
            return train_dataset, test_dataset

        debug_train_holdout = bool(getattr(args, 'debug_train_holdout', False))
        if debug_train_holdout:
            if not args.debug:
                raise ValueError("debug_train_holdout requires --debug")
            debug_dataset_dict = {args.test_dataset: 1}
            if args.joint_det:
                print(
                    "Debug train holdout excludes auxiliary detection data; "
                    "formal joint_det remains enabled"
                )
            shared_kwargs = dict(
                dataset_dict=debug_dataset_dict,
                test_dataset=args.test_dataset,
                split='train',
                use_color=args.use_color,
                use_height=args.use_height,
                overfit=False,
                data_path=args.data_root,
                detect_intermediate=args.detect_intermediate,
                use_multiview=args.use_multiview,
                butd=args.butd,
                butd_gt=args.butd_gt,
                butd_cls=args.butd_cls,
                augment_det=args.augment_det,
                skip_missing_superpoints=args.skip_missing_superpoints,
                use_sacr_source=(
                    getattr(args, 'use_sacr_source', False)
                    or getattr(args, 'use_sacr_score_refiner', False)
                    or getattr(
                        args, 'use_parent_relative_text_verifier', False
                    )
                ),
                legacy_scene_graph_cache_path=getattr(
                    args, 'legacy_scene_graph_cache', ''),
                legacy_scene_graph_cache_strict=getattr(
                    args, 'legacy_scene_graph_cache_strict', False),
                legacy_scene_graph_cache_expected_target_selection=getattr(
                    args,
                    'legacy_scene_graph_cache_expected_target_selection',
                    '',
                ),
                legacy_scene_graph_cache_expected_sha256=getattr(
                    args, 'legacy_scene_graph_cache_expected_sha256', ''),
            )
            train_dataset = Joint3DDataset(
                **dict(
                    shared_kwargs,
                    overfit=True,
                    scanrefer_debug_scene_partition='train',
                )
            )
            test_dataset = Joint3DDataset(
                **dict(
                    shared_kwargs,
                    overfit=True,
                    augment_det=False,
                    scanrefer_debug_scene_partition='holdout',
                )
            )
            test_dataset.augment = False
            if len(train_dataset) != 128 or len(test_dataset) != 128:
                raise ValueError(
                    "debug train holdout could not collect two 128-example sets"
                )
            train_scenes = {
                annotation['scan_id'] for annotation in train_dataset.annos
            }
            holdout_scenes = {
                annotation['scan_id'] for annotation in test_dataset.annos
            }
            if len(train_scenes) != 128 or len(holdout_scenes) != 120:
                raise ValueError(
                    "debug train/holdout scene cardinality changed: "
                    "train={}, holdout={} (expected 128/120)".format(
                        len(train_scenes), len(holdout_scenes)
                    )
                )
            if train_scenes.intersection(holdout_scenes):
                raise ValueError("debug train/holdout scenes are not disjoint")
            if args.eval:
                train_dataset = None
            print(
                "Debug train holdout: train={} examples/{} scenes; "
                "holdout={} examples/{} scenes; overlap=0".format(
                    0 if train_dataset is None else 128,
                    0 if train_dataset is None else len(train_scenes),
                    128, len(holdout_scenes),
                )
            )
            return train_dataset, test_dataset

        if args.eval:
            train_dataset = None
        else:
            train_dataset = Joint3DDataset(
                dataset_dict=dataset_dict,
                test_dataset=args.test_dataset,
                split='train' if not args.debug else 'val',
                use_color=args.use_color, use_height=args.use_height,
                overfit=args.debug,
                data_path=args.data_root,
                detect_intermediate=args.detect_intermediate,
                use_multiview=args.use_multiview,
                butd=args.butd,
                butd_gt=args.butd_gt,
                butd_cls=args.butd_cls,
                augment_det=args.augment_det,
                skip_missing_superpoints=args.skip_missing_superpoints,
                use_sacr_source=(
                    getattr(args, 'use_sacr_source', False)
                    or getattr(args, 'use_sacr_score_refiner', False)
                    or getattr(
                        args, 'use_parent_relative_text_verifier', False
                    )
                ),
                legacy_scene_graph_cache_path=getattr(
                    args, 'legacy_scene_graph_cache', ''),
                legacy_scene_graph_cache_strict=getattr(
                    args, 'legacy_scene_graph_cache_strict', False),
                legacy_scene_graph_cache_expected_target_selection=getattr(
                    args,
                    'legacy_scene_graph_cache_expected_target_selection',
                    '',
                ),
                legacy_scene_graph_cache_expected_sha256=getattr(
                    args, 'legacy_scene_graph_cache_expected_sha256', ''),
            )

        if is_counterfactual_parent_bounded_audit(args):
            if args.eval or args.debug or args.eval_train:
                raise ValueError(
                    "counterfactual Parent bounded audit requires the exact "
                    "Nr3D training split"
                )
            print(
                "Counterfactual Parent bounded audit: validation dataset "
                "construction disabled"
            )
            return train_dataset, None
        
        test_dataset = Joint3DDataset(
            dataset_dict=dataset_dict,
            test_dataset=args.test_dataset,
            split='val' if not args.eval_train else 'train',
            use_color=args.use_color, use_height=args.use_height,
            overfit=args.debug,
            data_path=args.data_root,
            detect_intermediate=args.detect_intermediate,
            use_multiview=args.use_multiview,
            butd=args.butd,
            butd_gt=args.butd_gt,
            butd_cls=args.butd_cls,
            wo_obj_name=args.wo_obj_name,
            skip_missing_superpoints=args.skip_missing_superpoints,
            use_sacr_source=(
                getattr(args, 'use_sacr_source', False)
                or getattr(args, 'use_sacr_score_refiner', False)
                or getattr(
                    args, 'use_parent_relative_text_verifier', False
                )
            ),
            legacy_scene_graph_cache_path=getattr(
                args, 'legacy_scene_graph_cache', ''),
            legacy_scene_graph_cache_strict=getattr(
                args, 'legacy_scene_graph_cache_strict', False),
            legacy_scene_graph_cache_expected_target_selection=getattr(
                args,
                'legacy_scene_graph_cache_expected_target_selection',
                '',
            ),
            legacy_scene_graph_cache_expected_sha256=getattr(
                args, 'legacy_scene_graph_cache_expected_sha256', ''),
        )
        return train_dataset, test_dataset

    # BRIEF Initialize the model.
    @staticmethod
    def get_model(args):
        """Initialize the model."""
        num_input_channel = int(args.use_color) * 3
        if args.use_height:
            num_input_channel += 1
        if args.use_multiview:
            num_input_channel += 128
        if args.use_soft_token_loss:
            num_class = 256
        else:
            num_class = 19
        ModelClass = eval(args.model)
        model = ModelClass(
            num_class=num_class,
            num_obj_class=485,
            input_feature_dim=num_input_channel,
            num_queries=args.num_target,
            num_decoder_layers=args.num_decoder_layers,
            self_position_embedding=args.self_position_embedding,
            contrastive_align_loss=args.use_contrastive_align,
            butd=args.butd or args.butd_gt or args.butd_cls,
            pointnet_ckpt=args.pp_checkpoint,
            pointnet_ckpt_sha256=getattr(
                args, 'pp_checkpoint_sha256', ''
            ),
            data_path = args.data_root,
            self_attend=args.self_attend,
            use_source_choice_selector=args.use_source_choice_selector,
            source_choice_selector_sources=args.source_choice_selector_sources,
            source_choice_selector_hidden_dim=args.source_choice_selector_hidden_dim,
            use_source_moe=getattr(args, 'use_source_moe', False),
            source_moe_shared_source=getattr(
                args, 'source_moe_shared_source', 'default'
            ),
            source_moe_top_k=getattr(args, 'source_moe_top_k', 2),
            source_moe_balance_loss_weight=getattr(
                args, 'source_moe_balance_loss_weight', 0.01
            ),
            source_moe_query_layers=getattr(
                args, 'source_moe_query_layers', 1
            ),
            source_moe_query_heads=getattr(
                args, 'source_moe_query_heads', 4
            ),
            source_moe_query_dropout=getattr(
                args, 'source_moe_query_dropout', 0.1
            ),
            source_moe_query_max_delta=getattr(
                args, 'source_moe_query_max_delta', 0.25
            ),
            source_moe_use_fallback_gate=getattr(
                args, 'source_moe_use_fallback_gate', False
            ),
            source_moe_gate_hidden_dim=getattr(
                args, 'source_moe_gate_hidden_dim', 128
            ),
            source_moe_gate_candidate_top_k=getattr(
                args, 'source_moe_gate_candidate_top_k', 8
            ),
            source_moe_gate_break_cost=getattr(
                args, 'source_moe_gate_break_cost', 2.0
            ),
            source_moe_gate_decision_margin=getattr(
                args, 'source_moe_gate_decision_margin', 0.0
            ),
            source_moe_gate_mask_utility_weight=getattr(
                args, 'source_moe_gate_mask_utility_weight', 0.25
            ),
            source_moe_gate_uncertainty_weight=getattr(
                args, 'source_moe_gate_uncertainty_weight', 0.0
            ),
            source_moe_gate_use_evidence_features=getattr(
                args, 'source_moe_gate_use_evidence_features', False
            ),
            source_moe_gate_context_layers=getattr(
                args, 'source_moe_gate_context_layers', 0
            ),
            source_moe_gate_context_heads=getattr(
                args, 'source_moe_gate_context_heads', 4
            ),
            source_moe_gate_context_dropout=getattr(
                args, 'source_moe_gate_context_dropout', 0.1
            ),
            source_moe_gate_action_mode=(
                getattr(args, 'source_moe_gate_action_mode', None)
                or 'decision'
            ),
            use_query_mask_fusion_calibrator=getattr(
                args, 'use_query_mask_fusion_calibrator', False
            ),
            query_mask_fusion_hidden_dim=getattr(
                args, 'query_mask_fusion_hidden_dim', 128
            ),
            query_mask_fusion_dropout=getattr(
                args, 'query_mask_fusion_dropout', 0.0
            ),
            query_mask_fusion_max_delta=getattr(
                args, 'query_mask_fusion_max_delta', 0.25
            ),
            query_mask_fusion_detach_inputs=True,
            use_egqs_mask_refiner=getattr(
                args, 'use_egqs_mask_refiner', False
            ),
            egqs_mask_refiner_arch=getattr(
                args, 'egqs_mask_refiner_arch', 'egqs'
            ),
            egqs_mask_refiner_hidden_dim=getattr(
                args, 'egqs_mask_refiner_hidden_dim', 32
            ),
            egqs_mask_refiner_max_delta=getattr(
                args, 'egqs_mask_refiner_max_delta', 2.0
            ),
            egqs_mask_refiner_components=getattr(
                args, 'egqs_mask_refiner_components', 'all'
            ),
            egqs_mask_refiner_graph_mode=getattr(
                args, 'egqs_mask_refiner_graph_mode', 'bilateral'
            ),
            egqs_mask_refiner_neighbor_count=getattr(
                args, 'egqs_mask_refiner_neighbor_count', 8
            ),
            egqs_mask_refiner_detach_inputs=True,
            use_joint_query_quality_reranker=getattr(
                args, 'use_joint_query_quality_reranker', False
            ),
            joint_query_quality_hidden_dim=getattr(
                args, 'joint_query_quality_hidden_dim', 128
            ),
            joint_query_quality_heads=getattr(
                args, 'joint_query_quality_heads', 4
            ),
            joint_query_quality_layers=getattr(
                args, 'joint_query_quality_layers', 1
            ),
            joint_query_quality_dropout=getattr(
                args, 'joint_query_quality_dropout', 0.1
            ),
            joint_query_quality_max_delta=getattr(
                args, 'joint_query_quality_max_delta', 1.25
            ),
            joint_query_quality_mask_weight=getattr(
                args, 'joint_query_quality_mask_weight', 0.25
            ),
            joint_query_quality_score_weight=getattr(
                args, 'joint_query_quality_score_weight', 1.0
            ),
            joint_query_quality_direct_residual_scale=getattr(
                args, 'joint_query_quality_direct_residual_scale', 1.0
            ),
            joint_query_quality_use_metric_aligned_utility=getattr(
                args, 'joint_query_quality_use_metric_aligned_utility', False
            ),
            joint_query_quality_preserve_parent_score=getattr(
                args, 'joint_query_quality_preserve_parent_score', False
            ),
            joint_query_quality_candidate_promotion_margin=getattr(
                args,
                'joint_query_quality_candidate_promotion_margin',
                0.0,
            ),
            joint_query_quality_use_parent_transition_advantage=getattr(
                args,
                'joint_query_quality_use_parent_transition_advantage',
                False,
            ),
            joint_query_quality_use_decomposed_transition_advantage=getattr(
                args,
                'joint_query_quality_use_decomposed_transition_advantage',
                False,
            ),
            joint_query_quality_use_setwise_tier_advantage=getattr(
                args,
                'joint_query_quality_use_setwise_tier_advantage',
                False,
            ),
            joint_query_quality_use_decoupled_setwise_heads=getattr(
                args,
                'joint_query_quality_use_decoupled_setwise_heads',
                False,
            ),
            joint_query_quality_use_factorized_setwise_safety=getattr(
                args,
                'joint_query_quality_use_factorized_setwise_safety',
                False,
            ),
            joint_query_quality_use_factorized_setwise_risk_bound=getattr(
                args,
                'joint_query_quality_use_factorized_setwise_risk_bound',
                False,
            ),
            joint_query_quality_use_setwise_safety_veto_gate=getattr(
                args,
                'joint_query_quality_use_setwise_safety_veto_gate',
                False,
            ),
            joint_query_quality_use_cost_calibrated_setwise_risk_bound=getattr(
                args,
                'joint_query_quality_use_cost_calibrated_setwise_risk_bound',
                False,
            ),
            joint_query_quality_use_setwise_safety_slack_quantile_bound=getattr(
                args,
                'joint_query_quality_use_setwise_safety_slack_quantile_bound',
                False,
            ),
            joint_query_quality_use_setwise_safety_slack_pairwise_order=getattr(
                args,
                'joint_query_quality_use_setwise_safety_slack_pairwise_order',
                False,
            ),
            joint_query_quality_use_proposal_conditioned_safety=getattr(
                args,
                'joint_query_quality_use_proposal_conditioned_safety',
                False,
            ),
            joint_query_quality_use_parent_referenced_safety=getattr(
                args,
                'joint_query_quality_use_parent_referenced_safety',
                False,
            ),
            joint_query_quality_use_coupled_safe_repair_witness=getattr(
                args,
                'joint_query_quality_use_coupled_safe_repair_witness',
                False,
            ),
            joint_query_quality_use_bidirectional_coupled_boundary=getattr(
                args,
                'joint_query_quality_use_bidirectional_coupled_boundary',
                False,
            ),
            joint_query_quality_use_centered_coupled_separation=getattr(
                args,
                'joint_query_quality_use_centered_coupled_separation',
                False,
            ),
            joint_query_quality_use_hazard_conditioned_coupled_separation=(
                getattr(
                    args,
                    'joint_query_quality_use_hazard_conditioned_coupled_separation',
                    False,
                )
            ),
            joint_query_quality_use_monotonic_box_safety_folding=getattr(
                args,
                'joint_query_quality_use_monotonic_box_safety_folding',
                False,
            ),
            joint_query_quality_use_same_candidate_branchwise_witness=getattr(
                args,
                'joint_query_quality_use_same_candidate_branchwise_witness',
                False,
            ),
            joint_query_quality_use_parent_non_degradation_certificate=getattr(
                args,
                'joint_query_quality_use_parent_non_degradation_certificate',
                False,
            ),
            joint_query_quality_use_criterion_responsible_hazard_attribution=getattr(
                args,
                'joint_query_quality_use_criterion_responsible_hazard_attribution',
                False,
            ),
            joint_query_quality_use_independent_joint_hazard_certificate=getattr(
                args,
                'joint_query_quality_use_independent_joint_hazard_certificate',
                False,
            ),
            joint_query_quality_use_frozen_raw_joint_hazard_features=getattr(
                args,
                'joint_query_quality_use_frozen_raw_joint_hazard_features',
                False,
            ),
            joint_query_quality_use_factorized_hit_advantage=getattr(
                args,
                'joint_query_quality_use_factorized_hit_advantage',
                False,
            ),
            joint_query_quality_use_factorized_nested_dominance=getattr(
                args,
                'joint_query_quality_use_factorized_nested_dominance',
                False,
            ),
            joint_query_quality_factorized_hit_break_cost=getattr(
                args,
                'joint_query_quality_factorized_hit_break_cost',
                4.0,
            ),
            joint_query_quality_parent_transition_break_cost=getattr(
                args,
                'joint_query_quality_parent_transition_break_cost',
                4.0,
            ),
            joint_query_quality_parent_transition_candidate_top_k=getattr(
                args,
                'joint_query_quality_parent_transition_candidate_top_k',
                0,
            ),
            joint_query_quality_use_mask_calibration=getattr(
                args, 'joint_query_quality_use_mask_calibration', False
            ),
            joint_query_quality_max_mask_alpha_delta=getattr(
                args, 'joint_query_quality_max_mask_alpha_delta', 1.0
            ),
            joint_query_quality_max_mask_logit_bias=getattr(
                args, 'joint_query_quality_max_mask_logit_bias', 2.0
            ),
            joint_query_quality_use_source_mask_evidence=getattr(
                args, 'joint_query_quality_use_source_mask_evidence', False
            ),
            joint_query_quality_use_gate_evidence=getattr(
                args, 'joint_query_quality_use_gate_evidence', False
            ),
            joint_query_quality_use_spatial_mask_refiner=getattr(
                args, 'joint_query_quality_use_spatial_mask_refiner', False
            ),
            joint_query_quality_spatial_mask_hidden_dim=getattr(
                args, 'joint_query_quality_spatial_mask_hidden_dim', 32
            ),
            joint_query_quality_max_spatial_mask_delta=getattr(
                args, 'joint_query_quality_max_spatial_mask_delta', 2.0
            ),
            joint_query_quality_use_adaptive_source_mixing=getattr(
                args,
                'joint_query_quality_use_adaptive_source_mixing',
                False,
            ),
            joint_query_quality_use_source_distribution_reliability=getattr(
                args,
                'joint_query_quality_use_source_distribution_reliability',
                False,
            ),
            joint_query_quality_source_names=getattr(
                args, 'joint_query_quality_source_names', ''
            ),
            joint_query_quality_max_source_mix_delta=getattr(
                args, 'joint_query_quality_max_source_mix_delta', 1.0
            ),
            joint_query_quality_source_mix_temperature=getattr(
                args, 'joint_query_quality_source_mix_temperature', 0.5
            ),
            joint_query_quality_detach_inputs=(
                not getattr(args, 'use_sacr_source', False)
            ),
            use_decoder_query_adapter=getattr(
                args, 'use_decoder_query_adapter', False
            ),
            decoder_query_adapter_hidden_dim=getattr(
                args, 'decoder_query_adapter_hidden_dim', 288
            ),
            decoder_query_adapter_heads=getattr(
                args, 'decoder_query_adapter_heads', 4
            ),
            decoder_query_adapter_dropout=getattr(
                args, 'decoder_query_adapter_dropout', 0.1
            ),
            decoder_query_adapter_max_delta=getattr(
                args, 'decoder_query_adapter_max_delta', 0.25
            ),
            use_sacr_source=getattr(args, 'use_sacr_source', False),
            use_sacr_score_refiner=getattr(
                args, 'use_sacr_score_refiner', False
            ),
            sacr_score_use_parent_relative_abstention=getattr(
                args, 'sacr_score_use_parent_relative_abstention', False
            ),
            sacr_score_use_relation_counterfactual=getattr(
                args, 'sacr_score_use_relation_counterfactual', False
            ),
            sacr_score_parent_gate_hidden_dim=getattr(
                args, 'sacr_score_parent_gate_hidden_dim', 32
            ),
            sacr_score_max_delta=getattr(
                args, 'sacr_score_max_delta', 0.25
            ),
            sacr_score_promotion_margin=getattr(
                args, 'sacr_score_promotion_margin', 0.01
            ),
            sacr_counterfactual_parent_top_k=getattr(
                args, 'sacr_counterfactual_parent_top_k', 16
            ),
            sacr_counterfactual_target_tolerance=getattr(
                args, 'sacr_counterfactual_target_tolerance', 0.05
            ),
            sacr_counterfactual_attribute_tolerance=getattr(
                args, 'sacr_counterfactual_attribute_tolerance', 0.05
            ),
            sacr_counterfactual_relation_scale=getattr(
                args, 'sacr_counterfactual_relation_scale', 4.0
            ),
            sacr_counterfactual_deployment_threshold=getattr(
                args, 'sacr_counterfactual_deployment_threshold', 0.05
            ),
            sacr_hidden_dim=getattr(args, 'sacr_hidden_dim', 288),
            sacr_max_pairs=getattr(args, 'sacr_max_pairs', 3),
            sacr_top_m_targets=getattr(args, 'sacr_top_m_targets', 32),
            sacr_top_k_anchors=getattr(args, 'sacr_top_k_anchors', 16),
            sacr_geo_dim=getattr(args, 'sacr_geo_dim', 16),
            sacr_min_parse_confidence=getattr(
                args, 'sacr_min_parse_confidence', 0.0
            ),
            sacr_score_contract_audit=getattr(
                args, 'sacr_score_contract_audit', False
            ),
            sacr_residual_scale_init=getattr(
                args, 'sacr_residual_scale_init', 0.1
            ),
            use_parent_relative_text_verifier=getattr(
                args, 'use_parent_relative_text_verifier', False
            ),
            parent_relative_text_verifier_top_k=getattr(
                args, 'parent_relative_text_verifier_top_k', 5
            ),
            parent_relative_text_verifier_max_candidates=getattr(
                args, 'parent_relative_text_verifier_max_candidates', 10
            ),
            parent_relative_text_verifier_hidden_dim=getattr(
                args, 'parent_relative_text_verifier_hidden_dim', 256
            ),
            parent_relative_text_verifier_heads=getattr(
                args, 'parent_relative_text_verifier_heads', 4
            ),
            parent_relative_text_verifier_dropout=getattr(
                args, 'parent_relative_text_verifier_dropout', 0.1
            ),
            parent_relative_text_verifier_max_parent_score_gap=getattr(
                args,
                'parent_relative_text_verifier_max_parent_score_gap',
                0.25,
            ),
            parent_relative_text_verifier_promotion_margin=getattr(
                args,
                'parent_relative_text_verifier_promotion_margin',
                1e-4,
            ),
            parent_relative_text_verifier_min_parse_confidence=getattr(
                args,
                'parent_relative_text_verifier_min_parse_confidence',
                0.5,
            ),
            parent_relative_text_verifier_min_anchor_mass=getattr(
                args,
                'parent_relative_text_verifier_min_anchor_mass',
                0.5,
            ),
            parent_relative_text_verifier_promotion_epsilon=getattr(
                args,
                'parent_relative_text_verifier_promotion_epsilon',
                1e-4,
            ),
            parent_relative_text_verifier_detach_inputs=getattr(
                args,
                'parent_relative_text_verifier_detach_inputs',
                True,
            ),
            parent_relative_text_verifier_filter_non_gt_boxes=bool(
                getattr(args, 'butd_cls', False)
            ),
            parent_relative_text_verifier_counterfactual_training=getattr(
                args,
                'parent_relative_text_verifier_counterfactual_training',
                False,
            ),
        )
        # params =  sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        # print(f'Total parameters: {params:.2f}M')
        local_variant = getattr(args, 'candidate_local_visual_variant', 'local')
        if local_variant != 'local' and not getattr(args, 'use_candidate_local_visual', False):
            raise ValueError('range visual reading requires --use_candidate_local_visual')
        if getattr(args, 'use_candidate_local_visual', False):
            if args.num_decoder_layers != 6:
                raise ValueError('candidate-local visual reading requires the fixed six-layer model')
            model.decoder[-1].local_visual = (
                CandidateLocalVisual() if local_variant == 'local'
                else CandidateRangeVisual(local_variant))
        return model

    # BRIEF input data.
    @staticmethod
    def _get_inputs(batch_data):
        inputs = {
            'point_clouds': batch_data['point_clouds'].float(), # ([B, 50000, 6]) xyz + colour
            'text': batch_data['utterances'],                   # list[B]  text
            "det_boxes": batch_data['all_detected_boxes'],      # ([B, 132, 6]) groupfree detection boxes
            "det_bbox_label_mask": batch_data['all_detected_bbox_label_mask'],  # ([B, 132]) mask
            "det_class_ids": batch_data['all_detected_class_ids'],   # ([B, 132])  box id
            "superpoint": batch_data['superpoint'],  # ([B, 50000]) superpoint map
        }
        for key in [
                "positive_map",
                "modify_positive_map",
                "pron_positive_map",
                "other_entity_map",
                "rel_positive_map",
                "target_spans",
                "entity_spans",
                "attr_spans",
                "rel_spans",
                "structured_anchor_ids",
                "coverage_stats",
                "parse_confidence",
                "decomposition_status",
                "decomp_global_only_mask",
                "decomp_weak_generic_mask",
                "structured_annotation_available",
        ]:
            if key in batch_data:
                inputs[key] = batch_data[key]
        return inputs


    # BRIEF only eval one epoch.
    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        # [Option] Object detection evaluation on ScanNet dataset.
        if args.test_dataset == 'scannet':      
            return self.evaluate_one_epoch_det(
                epoch, test_loader, model,
                criterion, set_criterion, args
            )

        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        # 7 layers: proposal, last, 0-4
        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
            prefixes = ['last_']
            prefixes.append('proposal_')
        else:
            prefixes = ['proposal_']  # only proposal
        prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]

        density_scene_audit = bool(getattr(
            args, 'density_aware_target_box_scene_disjoint_audit', False
        ))
        density_accumulator = (
            DensityAwareTargetBoxAuditAccumulator()
            if density_scene_audit else None
        )
        evaluator = (
            None if density_scene_audit
            else self._build_grounding_evaluator(args, prefixes)
        )

        # NOTE Main eval branch
        test_loader = tqdm(test_loader, ascii=True)
        for batch_idx, batch_data in enumerate(test_loader):
            # note forward and compute loss
            stat_dict, end_points = self._main_eval_branch(     
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )
            if density_accumulator is not None:
                density_accumulator.update(end_points)
            if evaluator is not None:
                for prefix in prefixes:
                    # note only consider the last layer
                    if prefix != 'last_':
                        continue

                    # evaluation
                    evaluator.evaluate(end_points, prefix)      

        if density_accumulator is not None:
            metadata = getattr(
                args,
                'density_aware_target_box_scene_disjoint_split_metadata',
                None,
            )
            if not isinstance(metadata, dict):
                raise ValueError("density scene split metadata is missing")
            metrics = density_accumulator.finalize(
                expected_sample_count=metadata['holdout_samples'],
                expected_identity_sha256=metadata[
                    'holdout_sample_identity_sha256'
                ],
            )
            return {"density_aware_target_box_scene_audit": metrics}

        evaluator.synchronize_between_processes()
        if dist.get_rank() == 0:
            if evaluator is not None:
                # tensorboard eval socre
                s_25 = evaluator.dets[("last_", 0.25, 1, "bbs")] / max(evaluator.gts[("last_", 0.25, 1, "bbs")], 1)
                s_50 = evaluator.dets[("last_", 0.5,  1, "bbs")] / max(evaluator.gts[("last_", 0.5,  1, "bbs")], 1)
                c_25 = evaluator.dets[("last_", 0.25, 1, "bbf")] / max(evaluator.gts[("last_", 0.25, 1, "bbf")], 1)
                c_50 = evaluator.dets[("last_", 0.5,  1, "bbf")] / max(evaluator.gts[("last_", 0.5,  1, "bbf")], 1)
                self.tensorboard.item["val_score"]["soft_token_0.25"] = s_25
                self.tensorboard.item["val_score"]["soft_token_0.5"] = s_50
                self.tensorboard.item["val_score"]["contrastive_0.25"] = c_25
                self.tensorboard.item["val_score"]["contrastive_0.5"] = c_50
                self.tensorboard.dump_tensorboard("val_score", epoch)
                # tensorboard eval loss
                for key in self.tensorboard.item["val_loss"]:
                    self.tensorboard.item["val_loss"][key] = stat_dict[key] / len(test_loader)
                self.tensorboard.dump_tensorboard("val_loss", epoch)

                evaluator.print_stats()
                self._log_source_moe_diagnostics(
                    stat_dict, float(max(len(test_loader), 1))
                )
                if (getattr(args, 'use_source_choice_selector', False)
                        or getattr(args, 'use_source_moe', False)):
                    expected_sample_count = getattr(
                        args, 'expected_eval_sample_count', None
                    )
                    diagnostics_exporter = getattr(
                        evaluator, 'export_source_choice_diagnostics', None
                    )
                    if callable(diagnostics_exporter):
                        diagnostics = diagnostics_exporter(
                            expected_sample_count=expected_sample_count
                        )
                        if diagnostics is not None:
                            gate_decision = (
                                build_source_moe_gate_decision_diagnostics(
                                    stat_dict
                                )
                            )
                            if gate_decision is not None:
                                if (
                                        expected_sample_count is not None
                                        and gate_decision["sample_count"]
                                        != expected_sample_count):
                                    raise ValueError(
                                        "gate decision diagnostics contain "
                                        "{} samples, expected {}".format(
                                            gate_decision["sample_count"],
                                            expected_sample_count,
                                        )
                                    )
                                diagnostics["gate_decision"] = gate_decision
                            save_source_choice_diagnostics_receipt(
                                args.log_dir, epoch, diagnostics
                            )
                    metrics = evaluator.export_retrain_metrics(
                        expected_sample_count=getattr(
                            args, 'expected_eval_sample_count', None
                        )
                    )
                    if bool(getattr(
                            args, 'fpr_scene_disjoint_audit', False)):
                        metrics[
                            'parent_relative_text_verifier_scene_audit'
                        ] = (
                            build_parent_relative_text_verifier_audit_diagnostics(
                                stat_dict,
                                getattr(
                                    args, 'expected_eval_sample_count', None
                                ),
                            )
                        )
                    return metrics
        return None
    
    # BRIEF Scannet detection evalution
    @torch.no_grad()
    def evaluate_one_epoch_det(self, epoch, test_loader,
                               model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        dataset_config = ScannetDatasetConfig(18)
        # Used for AP calculation
        CONFIG_DICT = {
            'remove_empty_box': False, 'use_3d_nms': True,
            'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
            'per_class_proposal': True, 'conf_thresh': 0.0,
            'dataset_config': dataset_config,
            'hungarian_loss': True
        }
        stat_dict = {}
        model.eval()  # set model to eval mode (for bn and dp)
        if set_criterion is not None:
            set_criterion.eval()

        if args.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_']
            prefixes += [
                f'{i}head_' for i in range(args.num_decoder_layers - 1)
            ]
        else:
            prefixes = ['proposal_']  # only proposal
        prefixes = ['last_']
        ap_calculator_list = [
            APCalculator(iou_thresh, dataset_config.class2type)
            for iou_thresh in args.ap_iou_thresholds
        ]
        mAPs = [
            [iou_thresh, {k: 0 for k in prefixes}]
            for iou_thresh in args.ap_iou_thresholds
        ]

        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        # Main eval branch
        # NOTE char span and token span.
        wordidx = np.array([
            0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11,
            12, 13, 13, 14, 15, 16, 16, 17, 17, 18, 18
        ])  # 18+1（not mentioned）
        tokenidx = np.array([
            1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 21, 23,
            25, 27, 29, 31, 32, 34, 36, 38, 39, 41, 42, 44, 45
        ])  # 18 token span

        test_loader = tqdm(test_loader, ascii=True)
        for batch_idx, batch_data in enumerate(test_loader):
            # note eval
            stat_dict, end_points = self._main_eval_branch(
                batch_idx, batch_data, test_loader, model, stat_dict,
                criterion, set_criterion, args
            )

            # step score   contrast
            proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
            proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
            sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
            sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
            sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
            sem_scores = sem_scores.to(sem_scores_.device)
            sem_scores[:, :sem_scores_.size(1), :sem_scores_.size(2)] = sem_scores_
            end_points['last_sem_cls_scores'] = sem_scores  # ([B, 256, 256])

            # step
            sem_cls = torch.zeros_like(end_points['last_sem_cls_scores'])[..., :19] # ([B, 256, 19])
            for w, t in zip(wordidx, tokenidx):
                sem_cls[..., w] += end_points['last_sem_cls_scores'][..., t]
            end_points['last_sem_cls_scores'] = sem_cls     # ([B, 256, 19])

            # step Parse predictions
            # for prefix in prefixes:
            prefix = 'last_'
            # pred
            batch_pred_map_cls = parse_predictions(
                end_points, CONFIG_DICT, prefix,
                size_cls_agnostic=True)
            batch_gt_map_cls = parse_groundtruths(
                end_points, CONFIG_DICT,
                size_cls_agnostic=True)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        # for prefix in prefixes:
        prefix = 'last_'
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(
                batch_pred_map_cls_dict[prefix],
                batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            self.logger.info(
                '=====================>'
                f'{prefix} IOU THRESH: {args.ap_iou_thresholds[i]}'
                '<====================='
            )
            for key in metrics_dict:
                self.logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

        for mAP in mAPs:
            self.logger.info(
                f'IoU[{mAP[0]}]:\t'
                + ''.join([
                    f'{key}: {mAP[1][key]:.4f} \t'
                    for key in sorted(mAP[1].keys())
                ])
            )

        return None


if __name__ == '__main__':
    # huggingface
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    opt = parse_option()
    opt = prepare_source_moe_gate_checkpoint_config(opt)
    
    # distributed 
    torch.cuda.set_device(opt.local_rank)
    # https://github.com/open-mmlab/mmcv/issues/1969#issuecomment-1304721237
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))  
    
    # cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(opt)
    ckpt_path = train_tester.main(opt)
