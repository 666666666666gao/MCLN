#!/usr/bin/env python
"""Train a scene-disjoint ScanRefer REC query reranker from cached rows."""

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rec_candidate_adapter import FEATURE_SCHEMA_VERSION
from models.rec_reranker import (
    QueryReranker,
    blend_candidate_scores,
    compute_rec_reranker_loss,
)


CACHE_SCHEMA_VERSION = 1
TARGET_IOU_POLICY = "root_only"
ARTIFACT_VERSION = 1
GRAD_CLIP_NORM = 1.0
DEFAULT_RERANKER_WEIGHTS = (
    0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0
)
MODEL_INPUT_KEYS = (
    "use_color", "use_height", "use_multiview",
    "butd", "butd_gt", "butd_cls",
)
LEGACY_BACKBONE_CONFIG_KEYS = (
    "model", "num_target", "num_decoder_layers",
    "self_position_embedding", "self_attend",
    "use_soft_token_loss", "use_contrastive_align",
    "detect_intermediate", "use_source_choice_selector",
    "source_choice_selector_sources", "source_choice_selector_hidden_dim",
)
SOURCE_MOE_BACKBONE_CONFIG_DEFAULTS = {
    "use_source_moe": False,
    "source_moe_shared_source": "default",
    "source_moe_top_k": 2,
    "source_moe_balance_loss_weight": 0.01,
    "source_moe_query_layers": 1,
    "source_moe_query_heads": 4,
    "source_moe_query_dropout": 0.1,
    "source_moe_query_max_delta": 0.25,
}
SOURCE_MOE_GATE_BACKBONE_CONFIG_DEFAULTS = {
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
BACKBONE_CONFIG_KEYS = (
    LEGACY_BACKBONE_CONFIG_KEYS
    + tuple(SOURCE_MOE_BACKBONE_CONFIG_DEFAULTS.keys())
    + tuple(SOURCE_MOE_GATE_BACKBONE_CONFIG_DEFAULTS.keys())
)


def normalize_backbone_config(backbone_config):
    """Return the full backbone contract while accepting legacy artifacts."""
    if (not isinstance(backbone_config, dict)
            or any(key not in backbone_config
                   for key in LEGACY_BACKBONE_CONFIG_KEYS)):
        raise ValueError("backbone config is invalid")

    moe_keys = tuple(SOURCE_MOE_BACKBONE_CONFIG_DEFAULTS.keys())
    present = [key in backbone_config for key in moe_keys]
    if any(present) and not all(present):
        raise ValueError("source-MoE backbone config is incomplete")
    gate_keys = tuple(
        key for key in SOURCE_MOE_GATE_BACKBONE_CONFIG_DEFAULTS
        if key not in (
            "source_moe_gate_uncertainty_weight",
            "source_moe_gate_use_evidence_features",
            "source_moe_gate_action_mode",
        )
    )
    gate_present = [key in backbone_config for key in gate_keys]
    if any(gate_present) and not all(gate_present):
        raise ValueError("source-MoE fallback-gate config is incomplete")

    normalized = copy.deepcopy(backbone_config)
    for key, value in SOURCE_MOE_BACKBONE_CONFIG_DEFAULTS.items():
        normalized.setdefault(key, value)
    for key, value in SOURCE_MOE_GATE_BACKBONE_CONFIG_DEFAULTS.items():
        normalized.setdefault(key, value)

    positive_ints = (
        "source_moe_top_k", "source_moe_query_heads",
        "source_moe_gate_hidden_dim", "source_moe_gate_candidate_top_k",
    )
    if any(
            not isinstance(normalized[key], int)
            or isinstance(normalized[key], bool)
            or normalized[key] <= 0
            for key in positive_ints):
        raise ValueError("source-MoE integer config is invalid")
    layers = normalized["source_moe_query_layers"]
    if (not isinstance(layers, int) or isinstance(layers, bool)
            or layers < 0):
        raise ValueError("source-MoE query layer count is invalid")
    if (not isinstance(normalized["use_source_moe"], bool)
            or not isinstance(
                normalized["source_moe_use_fallback_gate"], bool
            )
            or not isinstance(
                normalized["source_moe_gate_use_evidence_features"], bool
            )
            or not isinstance(normalized["source_moe_shared_source"], str)
            or not normalized["source_moe_shared_source"]):
        raise ValueError("source-MoE backbone config is invalid")
    if normalized["source_moe_gate_action_mode"] not in (
            "decision", "expected_utility", "direct_utility",
            "hierarchical_utility", "pairwise_verifier",
            "topn_pairwise_verifier", "topn_dual_evidence_verifier",
            "topn_absolute_quality_delta",
            "cascade_absolute_quality_correction",
            "cascade_opportunity_quality_correction",
            "cascade_opportunity_verified_correction",
            "cascade_joint_risk_correction",
            "cascade_v19_fallback_set_correction",
            "cascade_v19_rich_set_correction",
            "cascade_v23_dense_quality_correction",
            "cascade_v24_relative_risk_correction",
            "cascade_v25_pairwise_calibrated_correction",
            "cascade_v26_prior_restored_pairwise_correction",
            "cascade_v28_selected_abstention_correction",
            "cascade_v29_counterfactual_selected_correction",
            "cascade_v37_counterfactual_benefit_hazard_correction",
            "cascade_v38_complementary_logodds_correction",
            "cascade_v39_hazard_residual_correction"):
        raise ValueError("source-MoE gate action mode is invalid")
    if (normalized["source_moe_gate_action_mode"] != "decision"
            and not normalized["source_moe_use_fallback_gate"]):
        raise ValueError(
            "non-default gate action mode requires the fallback gate"
        )

    balance = normalized["source_moe_balance_loss_weight"]
    dropout = normalized["source_moe_query_dropout"]
    max_delta = normalized["source_moe_query_max_delta"]
    gate_break_cost = normalized["source_moe_gate_break_cost"]
    gate_margin = normalized["source_moe_gate_decision_margin"]
    gate_mask_weight = normalized["source_moe_gate_mask_utility_weight"]
    gate_uncertainty_weight = normalized[
        "source_moe_gate_uncertainty_weight"
    ]
    for name, value in (
            ("balance loss weight", balance),
            ("query dropout", dropout),
            ("query max delta", max_delta),
            ("gate break cost", gate_break_cost),
            ("gate decision margin", gate_margin),
            ("gate mask utility weight", gate_mask_weight),
            ("gate uncertainty weight", gate_uncertainty_weight)):
        if (not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(float(value))):
            raise ValueError("source-MoE {} is invalid".format(name))
    if float(balance) < 0.0 or not 0.0 <= float(dropout) < 1.0 \
            or float(max_delta) < 0.0 or float(gate_break_cost) < 1.0 \
            or float(gate_margin) < 0.0 or float(gate_mask_weight) < 0.0 \
            or float(gate_uncertainty_weight) < 0.0:
        raise ValueError("source-MoE numeric config is invalid")
    if (normalized["source_moe_use_fallback_gate"]
            and not normalized["use_source_moe"]):
        raise ValueError("fallback gate requires source-MoE")
    if (normalized["source_moe_gate_use_evidence_features"]
            and not normalized["source_moe_use_fallback_gate"]):
        raise ValueError("gate evidence features require the fallback gate")
    if (normalized["use_source_moe"]
            and normalized.get("use_source_choice_selector") is True):
        raise ValueError("source selector and source-MoE cannot both be enabled")
    source_spec = normalized.get("source_choice_selector_sources")
    if not isinstance(source_spec, str):
        raise ValueError("source list is invalid")
    sources = tuple(
        name.strip()
        for name in source_spec.split(",")
        if name.strip()
    )
    if (normalized["use_source_moe"]
            and normalized["source_moe_shared_source"] not in sources):
        raise ValueError("source-MoE shared source is unavailable")
    return normalized


def _load_manifest(cache_dir):
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("training cache manifest does not exist: {}".format(
            manifest_path
        ))
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError("training cache manifest is malformed: {}".format(
            error
        ))
    if not isinstance(manifest, dict):
        raise ValueError("training cache manifest must contain an object")
    return manifest


def _validate_manifest(manifest, expected_split):
    if manifest.get("split") != expected_split:
        if expected_split == "train":
            raise ValueError("reranker training requires a training split cache")
        raise ValueError("candidate cache does not match the expected split")
    if manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported cache schema version")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported feature schema version")
    if manifest.get("target_iou_policy") != TARGET_IOU_POLICY:
        raise ValueError("unsupported target IoU policy")
    model_inputs = manifest.get("model_inputs")
    if (not isinstance(model_inputs, dict)
            or any(not isinstance(model_inputs.get(key), bool)
                   for key in MODEL_INPUT_KEYS)):
        raise ValueError("cache model inputs are invalid")
    backbone_config = manifest.get("backbone_config")
    try:
        normalize_backbone_config(backbone_config)
    except ValueError:
        raise ValueError("cache backbone config is invalid")

    feature_dim = manifest.get("feature_dim")
    feature_names = manifest.get("feature_names")
    if (not isinstance(feature_dim, int) or isinstance(feature_dim, bool)
            or feature_dim <= 0):
        raise ValueError("cache feature dimension must be a positive integer")
    if (not isinstance(feature_names, list)
            or len(feature_names) != feature_dim
            or not all(isinstance(name, str) for name in feature_names)
            or len(set(feature_names)) != len(feature_names)):
        raise ValueError("cache feature names do not match its feature dimension")

    candidate_rule = manifest.get("candidate_rule")
    if not isinstance(candidate_rule, dict):
        raise ValueError("cache candidate rule must be an object")
    max_candidates = candidate_rule.get("max_candidates")
    topk_per_source = candidate_rule.get("topk_per_source")
    for name, value in (
            ("max_candidates", max_candidates),
            ("topk_per_source", topk_per_source)):
        if (not isinstance(value, int) or isinstance(value, bool)
                or value <= 0):
            raise ValueError(
                "cache candidate rule {} must be positive".format(name)
            )

    fingerprint = manifest.get("checkpoint_sha256")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("cache checkpoint fingerprint is missing")
    sample_count = manifest.get("sample_count")
    if (not isinstance(sample_count, int) or isinstance(sample_count, bool)
            or sample_count <= 0):
        raise ValueError("cache sample count must be positive")
    dataset_size = manifest.get("dataset_size")
    if (not isinstance(dataset_size, int) or isinstance(dataset_size, bool)
            or dataset_size <= 0):
        raise ValueError("cache dataset size must be positive")
    if sample_count != dataset_size:
        raise ValueError(
            "training cache is incomplete: sample count does not match "
            "dataset size"
        )
    source_dataset_size = manifest.get("source_dataset_size")
    if (not isinstance(source_dataset_size, int)
            or isinstance(source_dataset_size, bool)
            or source_dataset_size <= 0):
        raise ValueError("cache source dataset size must be positive")
    if dataset_size != source_dataset_size:
        raise ValueError(
            "training cache is limited: dataset size does not match "
            "source dataset size"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("training cache must contain at least one shard")
    return feature_dim, max_candidates


def _validate_tensor(row, key, shape, floating=None):
    value = row.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError("cache row {} has malformed {}".format(
            row.get("dataset_index", "?"), key
        ))
    if floating is True and not value.is_floating_point():
        raise ValueError("cache row {} {} must be floating point".format(
            row.get("dataset_index", "?"), key
        ))
    if floating is False and value.is_floating_point():
        raise ValueError("cache row {} {} must be integral".format(
            row.get("dataset_index", "?"), key
        ))
    return value


def _validate_row(row, expected_index, feature_dim, max_candidates):
    if not isinstance(row, dict):
        raise ValueError("cache row {} must be an object".format(expected_index))
    dataset_index = row.get("dataset_index")
    if (not isinstance(dataset_index, int) or isinstance(dataset_index, bool)
            or dataset_index != expected_index):
        raise ValueError(
            "cached dataset indices are not contiguous at {}".format(
                expected_index
            )
        )
    if not isinstance(row.get("scan_id"), str) or not row["scan_id"]:
        raise ValueError("cache row {} has an invalid scan_id".format(
            expected_index
        ))
    target_id = row.get("target_id")
    if not isinstance(target_id, int) or isinstance(target_id, bool):
        raise ValueError("cache row {} has an invalid target_id".format(
            expected_index
        ))

    features = _validate_tensor(
        row, "features", (max_candidates, feature_dim), floating=True
    )
    boxes = _validate_tensor(
        row, "boxes", (max_candidates, 6), floating=True
    )
    query_indices = _validate_tensor(
        row, "query_indices", (max_candidates,), floating=False
    )
    valid_mask = _validate_tensor(
        row, "valid_mask", (max_candidates,), floating=False
    )
    default_scores = _validate_tensor(
        row, "default_scores", (max_candidates,), floating=True
    )
    contrastive_scores = _validate_tensor(
        row, "contrastive_scores", (max_candidates,), floating=True
    )
    candidate_ious = _validate_tensor(
        row, "candidate_ious", (max_candidates,), floating=True
    )
    if valid_mask.dtype != torch.bool:
        raise ValueError("cache row {} valid_mask must be boolean".format(
            expected_index
        ))
    if not valid_mask.any():
        raise ValueError("cache row {} has no valid candidates".format(
            expected_index
        ))
    for key, tensor in (
            ("features", features),
            ("boxes", boxes),
            ("default_scores", default_scores),
            ("contrastive_scores", contrastive_scores),
            ("candidate_ious", candidate_ious)):
        if not torch.isfinite(tensor).all():
            raise ValueError("cache row {} has non-finite {}".format(
                expected_index, key
            ))
    valid_ious = candidate_ious[valid_mask]
    if (valid_ious < 0.0).any() or (valid_ious > 1.0).any():
        raise ValueError("cache row {} candidate IoUs are outside [0, 1]".format(
            expected_index
        ))
    if torch.unique(query_indices[valid_mask]).numel() != int(valid_mask.sum()):
        raise ValueError("cache row {} has duplicate valid query indices".format(
            expected_index
        ))
    default_query = row.get("default_top1_query_index")
    if not isinstance(default_query, int) or isinstance(default_query, bool):
        raise ValueError("cache row {} has invalid default Top-1 query".format(
            expected_index
        ))
    default_matches = (query_indices == default_query) & valid_mask
    if int(default_matches.sum()) != 1:
        raise ValueError(
            "cache row {} default Top-1 query is not a valid candidate".format(
                expected_index
            )
        )


def load_candidate_cache(cache_dir, expected_split):
    """Load and validate one completed candidate cache on CPU."""
    if expected_split not in ("train", "val"):
        raise ValueError("expected_split must be either 'train' or 'val'")
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not cache_dir.is_dir():
        raise ValueError("training cache directory does not exist: {}".format(
            cache_dir
        ))
    manifest = _load_manifest(cache_dir)
    feature_dim, max_candidates = _validate_manifest(manifest, expected_split)

    expected_shards = []
    for index, shard_name in enumerate(manifest["shards"]):
        expected_name = "shard_{:06d}.pt".format(index)
        if shard_name != expected_name:
            raise ValueError("training cache has non-contiguous shard names")
        expected_shards.append(expected_name)
    actual_shards = sorted(
        path.name for path in cache_dir.glob("shard_*.pt") if path.is_file()
    )
    if actual_shards != expected_shards:
        raise ValueError("training cache shards do not match the manifest")

    rows = []
    for shard_name in expected_shards:
        shard_path = cache_dir / shard_name
        try:
            payload = torch.load(shard_path, map_location="cpu")
        except Exception as error:
            raise ValueError("could not load cache shard {}: {}".format(
                shard_name, error
            ))
        shard_rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(shard_rows, list) or not shard_rows:
            raise ValueError("cache shard {} does not contain a row list".format(
                shard_name
            ))
        for row in shard_rows:
            _validate_row(row, len(rows), feature_dim, max_candidates)
            rows.append(row)
    if len(rows) != manifest["sample_count"]:
        raise ValueError("cache sample count does not match loaded rows")
    return rows, manifest


def load_training_cache(cache_dir):
    """Load and validate one completed training cache on CPU."""
    return load_candidate_cache(cache_dir, expected_split="train")


def deterministic_scene_split(rows, seed, calibration_fraction=0.10):
    """Split rows by scene while preserving their original row order."""
    if not rows:
        raise ValueError("cannot split an empty training cache")
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")
    scenes = sorted({row["scan_id"] for row in rows})
    if len(scenes) == 1:
        return list(rows), []
    shuffled = list(scenes)
    random.Random(int(seed)).shuffle(shuffled)
    calibration_count = int(round(len(scenes) * float(calibration_fraction)))
    calibration_count = max(1, min(calibration_count, len(scenes) - 1))
    calibration_scenes = set(shuffled[:calibration_count])
    fit_rows = [
        row for row in rows if row["scan_id"] not in calibration_scenes
    ]
    calibration_rows = [
        row for row in rows if row["scan_id"] in calibration_scenes
    ]
    return fit_rows, calibration_rows


def compute_feature_stats(rows, min_std=1e-6):
    """Compute stable population statistics over valid candidates only."""
    if not rows:
        raise ValueError("fit rows cannot be empty")
    if not float(min_std) > 0.0:
        raise ValueError("min_std must be positive")
    feature_dim = int(rows[0]["features"].shape[-1])
    count = 0
    mean = torch.zeros(feature_dim, dtype=torch.float64)
    squared_deviation = torch.zeros(feature_dim, dtype=torch.float64)
    for row in rows:
        features = row["features"]
        valid = row["valid_mask"].bool()
        if features.dim() != 2 or features.shape[-1] != feature_dim:
            raise ValueError("fit row features have inconsistent dimensions")
        values = features[valid].detach().cpu().to(torch.float64)
        if values.numel() == 0:
            continue
        batch_count = values.shape[0]
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
        raise ValueError("fit rows contain no valid candidates")
    std = torch.sqrt(squared_deviation / float(count)).clamp(
        min=float(min_std)
    )
    mean = mean.to(torch.float32)
    std = std.to(torch.float32)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("feature statistics are non-finite")
    return mean, std


def normalize_features(features, valid_mask, mean, std):
    """Normalize valid candidates and canonicalize invalid padding to zero."""
    if features.dim() not in (2, 3):
        raise ValueError("features must have shape [K,D] or [B,K,D]")
    if tuple(valid_mask.shape) != tuple(features.shape[:-1]):
        raise ValueError("valid_mask must match the candidate axes")
    if (not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor)
            or mean.dim() != 1 or std.dim() != 1
            or mean.shape != std.shape
            or mean.shape[0] != features.shape[-1]):
        raise ValueError("feature statistics do not match the input dimension")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("feature statistics must be finite")
    if (std <= 0).any():
        raise ValueError("feature standard deviations must be positive")
    local_mean = mean.to(device=features.device, dtype=features.dtype)
    local_std = std.to(device=features.device, dtype=features.dtype)
    normalized = (features - local_mean) / local_std
    return torch.where(
        valid_mask.bool().unsqueeze(-1),
        normalized,
        torch.zeros_like(normalized),
    )


class CandidateCacheDataset(Dataset):
    """Normalized view over validated candidate cache rows."""

    def __init__(self, rows, feature_mean, feature_std):
        if not rows:
            raise ValueError("candidate cache dataset cannot be empty")
        self.rows = list(rows)
        self.feature_mean = feature_mean.detach().cpu().clone()
        self.feature_std = feature_std.detach().cpu().clone()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        valid_mask = row["valid_mask"].bool()
        return {
            "features": normalize_features(
                row["features"].float(),
                valid_mask,
                self.feature_mean,
                self.feature_std,
            ),
            "valid_mask": valid_mask,
            "candidate_ious": row["candidate_ious"].float(),
            "default_scores": row["default_scores"].float(),
            "query_indices": row["query_indices"].long(),
            "default_top1_query_index": torch.tensor(
                row["default_top1_query_index"], dtype=torch.long
            ),
        }


def _resolve_device(device):
    if device is None or str(device) == "auto":
        resolved = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    return resolved


def _make_loader(dataset, batch_size, shuffle, seed, device):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=generator,
    )


def _move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        for key, value in batch.items()
    }


def calibration_score(acc025, acc050):
    """Return the target-aware score used for early stopping."""
    acc025 = float(acc025)
    acc050 = float(acc050)
    return min(acc025 / 0.60, acc050 / 0.47) + 0.1 * (
        acc025 + acc050
    )


def evaluate_reranker_blends(model, rows, feature_mean, feature_std,
                             reranker_weights, batch_size=64, device="cpu"):
    """Evaluate several default/learned rank blends in one model pass."""
    if not rows:
        raise ValueError("evaluation rows cannot be empty")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    weights = tuple(float(weight) for weight in reranker_weights)
    if not weights or len(set(weights)) != len(weights):
        raise ValueError("reranker weights must be a nonempty unique sequence")
    if any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("reranker weights must lie in [0, 1]")
    device = _resolve_device(device)
    dataset = CandidateCacheDataset(rows, feature_mean, feature_std)
    loader = _make_loader(dataset, batch_size, False, 0, device)
    was_training = model.training
    model.eval()
    counts = {
        weight: {
            "sample_count": 0,
            "hits025": 0,
            "hits050": 0,
            "default_hits025": 0,
            "default_hits050": 0,
        }
        for weight in weights
    }
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            outputs = model(batch["features"], batch["valid_mask"])
            default_matches = (
                batch["query_indices"]
                == batch["default_top1_query_index"].unsqueeze(1)
            ) & batch["valid_mask"]
            if not default_matches.any(dim=1).all():
                raise ValueError("default Top-1 query is absent during evaluation")
            default_ious = batch["candidate_ious"].masked_fill(
                ~default_matches, -1.0
            ).max(dim=1).values
            batch_count = int(batch["candidate_ious"].shape[0])
            for weight in weights:
                blended_scores = blend_candidate_scores(
                    batch["default_scores"],
                    outputs["ranking_logits"],
                    batch["valid_mask"],
                    reranker_weight=weight,
                )
                selected = blended_scores.argmax(dim=1)
                selected_ious = torch.gather(
                    batch["candidate_ious"], 1, selected.unsqueeze(1)
                ).squeeze(1)
                row_counts = counts[weight]
                row_counts["sample_count"] += batch_count
                row_counts["hits025"] += int(
                    (selected_ious > 0.25).sum().item()
                )
                row_counts["hits050"] += int(
                    (selected_ious > 0.50).sum().item()
                )
                row_counts["default_hits025"] += int(
                    (default_ious > 0.25).sum().item()
                )
                row_counts["default_hits050"] += int(
                    (default_ious > 0.50).sum().item()
                )
    model.train(was_training)
    metrics_by_weight = {}
    for weight in weights:
        row_counts = counts[weight]
        denominator = float(row_counts["sample_count"])
        acc025 = row_counts["hits025"] / denominator
        acc050 = row_counts["hits050"] / denominator
        metrics_by_weight[weight] = {
            "sample_count": row_counts["sample_count"],
            "hits025": row_counts["hits025"],
            "hits050": row_counts["hits050"],
            "default_hits025": row_counts["default_hits025"],
            "default_hits050": row_counts["default_hits050"],
            "acc025": acc025,
            "acc050": acc050,
            "default_acc025": (
                row_counts["default_hits025"] / denominator
            ),
            "default_acc050": (
                row_counts["default_hits050"] / denominator
            ),
            "score": calibration_score(acc025, acc050),
        }
    return metrics_by_weight


def evaluate_reranker(model, rows, feature_mean, feature_std,
                      batch_size=64, device="cpu", reranker_weight=1.0):
    """Evaluate one default/learned rank blend with strict REC thresholds."""
    weight = float(reranker_weight)
    return evaluate_reranker_blends(
        model,
        rows,
        feature_mean,
        feature_std,
        reranker_weights=(weight,),
        batch_size=batch_size,
        device=device,
    )[weight]


def choose_best_reranker_blend(metrics_by_weight):
    """Choose a calibration blend without regressing either default metric."""
    if 0.0 not in metrics_by_weight:
        raise ValueError("reranker blend candidates must include weight 0")
    baseline = metrics_by_weight[0.0]
    best_weight = 0.0
    best_metrics = baseline
    for weight in sorted(metrics_by_weight):
        metrics = metrics_by_weight[weight]
        if (metrics["acc025"] + 1e-12 < baseline["acc025"]
                or metrics["acc050"] + 1e-12 < baseline["acc050"]):
            continue
        if metrics["score"] > best_metrics["score"] + 1e-12:
            best_weight = weight
            best_metrics = metrics
    return best_weight, best_metrics


def select_best_reranker_blend(model, rows, feature_mean, feature_std,
                               reranker_weights=DEFAULT_RERANKER_WEIGHTS,
                               batch_size=64, device="cpu"):
    """Select the most conservative calibration-optimal blend."""
    weights = tuple(sorted(float(weight) for weight in reranker_weights))
    metrics_by_weight = evaluate_reranker_blends(
        model,
        rows,
        feature_mean,
        feature_std,
        reranker_weights=weights,
        batch_size=batch_size,
        device=device,
    )
    return choose_best_reranker_blend(metrics_by_weight)


def _model_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            return float(module.p)
    return 0.0


def _cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _atomic_torch_save(path, payload):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def save_reranker_artifact(output, model, feature_mean, feature_std, manifest,
                           epoch, calibration_metrics, training_args,
                           reranker_weight=1.0):
    """Build and atomically save a self-contained reranker artifact."""
    input_dim = int(model.input_dim)
    if manifest.get("feature_dim") != input_dim:
        raise ValueError("model input dimension does not match cache metadata")
    if (feature_mean.shape != (input_dim,)
            or feature_std.shape != (input_dim,)):
        raise ValueError("feature statistics do not match model input dimension")
    if not torch.isfinite(feature_mean).all() or not torch.isfinite(
            feature_std).all():
        raise ValueError("feature statistics must be finite")
    if (feature_std <= 0).any():
        raise ValueError("feature standard deviations must be positive")
    reranker_weight = float(reranker_weight)
    if not 0.0 <= reranker_weight <= 1.0:
        raise ValueError("reranker weight must lie in [0, 1]")
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model_state_dict": _cpu_state_dict(model),
        "model_config": {
            "input_dim": input_dim,
            "hidden_dim": int(model.hidden_dim),
            "dropout": _model_dropout(model),
        },
        "feature_mean": feature_mean.detach().cpu().float().clone(),
        "feature_std": feature_std.detach().cpu().float().clone(),
        "adapter_schema_version": manifest["feature_schema_version"],
        "input_dim": input_dim,
        "feature_names": list(manifest["feature_names"]),
        "candidate_rule": copy.deepcopy(manifest["candidate_rule"]),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "target_iou_policy": manifest["target_iou_policy"],
        "model_inputs": copy.deepcopy(manifest["model_inputs"]),
        "backbone_config": copy.deepcopy(manifest["backbone_config"]),
        "score_mode": "rank_blend",
        "reranker_weight": reranker_weight,
        "epoch": int(epoch),
        "calibration_metrics": copy.deepcopy(calibration_metrics),
        "training_args": copy.deepcopy(dict(training_args)),
    }
    _atomic_torch_save(output, artifact)
    return artifact


def _validate_artifact(artifact):
    if not isinstance(artifact, dict):
        raise ValueError("reranker artifact must contain a dictionary")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported reranker artifact version")
    if artifact.get("adapter_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("reranker artifact feature schema does not match adapter")
    if artifact.get("target_iou_policy") != TARGET_IOU_POLICY:
        raise ValueError("reranker artifact target IoU policy is unsupported")
    model_inputs = artifact.get("model_inputs")
    if (not isinstance(model_inputs, dict)
            or any(not isinstance(model_inputs.get(key), bool)
                   for key in MODEL_INPUT_KEYS)):
        raise ValueError("reranker artifact model inputs are invalid")
    backbone_config = artifact.get("backbone_config")
    try:
        normalize_backbone_config(backbone_config)
    except ValueError:
        raise ValueError("reranker artifact backbone config is invalid")
    if artifact.get("score_mode") != "rank_blend":
        raise ValueError("reranker artifact score mode is unsupported")
    reranker_weight = artifact.get("reranker_weight")
    if (not isinstance(reranker_weight, (float, int))
            or isinstance(reranker_weight, bool)
            or not 0.0 <= float(reranker_weight) <= 1.0):
        raise ValueError("reranker artifact weight is invalid")
    input_dim = artifact.get("input_dim")
    model_config = artifact.get("model_config")
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("reranker artifact input dimension is invalid")
    if not isinstance(model_config, dict) or model_config.get(
            "input_dim") != input_dim:
        raise ValueError("reranker artifact model configuration is invalid")
    hidden_dim = model_config.get("hidden_dim")
    dropout = model_config.get("dropout")
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("reranker artifact hidden dimension is invalid")
    if not isinstance(dropout, (float, int)) or not 0.0 <= dropout < 1.0:
        raise ValueError("reranker artifact dropout is invalid")
    feature_names = artifact.get("feature_names")
    if (not isinstance(feature_names, list)
            or len(feature_names) != input_dim
            or not all(isinstance(name, str) for name in feature_names)
            or len(set(feature_names)) != len(feature_names)):
        raise ValueError("reranker artifact feature names are invalid")
    for key in ("feature_mean", "feature_std"):
        value = artifact.get(key)
        if (not isinstance(value, torch.Tensor)
                or value.shape != (input_dim,)
                or not torch.isfinite(value).all()):
            raise ValueError("reranker artifact {} is invalid".format(key))
    if (artifact["feature_std"] <= 0).any():
        raise ValueError("reranker artifact feature_std must be positive")
    candidate_rule = artifact.get("candidate_rule")
    if (not isinstance(candidate_rule, dict)
            or not isinstance(candidate_rule.get("topk_per_source"), int)
            or isinstance(candidate_rule.get("topk_per_source"), bool)
            or candidate_rule["topk_per_source"] <= 0
            or not isinstance(candidate_rule.get("max_candidates"), int)
            or isinstance(candidate_rule.get("max_candidates"), bool)
            or candidate_rule["max_candidates"] <= 0):
        raise ValueError("reranker artifact candidate rule is invalid")
    if (not isinstance(artifact.get("checkpoint_sha256"), str)
            or not artifact["checkpoint_sha256"]):
        raise ValueError("reranker artifact checkpoint fingerprint is invalid")
    if not isinstance(artifact.get("model_state_dict"), dict):
        raise ValueError("reranker artifact model state is invalid")
    epoch = artifact.get("epoch")
    if (not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0):
        raise ValueError("reranker artifact epoch is invalid")
    calibration_metrics = artifact.get("calibration_metrics")
    if not isinstance(calibration_metrics, dict):
        raise ValueError("reranker artifact calibration_metrics are invalid")
    for name in ("acc025", "acc050"):
        value = calibration_metrics.get(name)
        if (not isinstance(value, (float, int)) or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0):
            raise ValueError(
                "reranker artifact calibration_metrics are invalid"
            )
    training_args = artifact.get("training_args")
    if not isinstance(training_args, dict) or not training_args:
        raise ValueError("reranker artifact training_args are invalid")
    return model_config


def load_reranker_artifact(path, device="cpu"):
    """Load a validated artifact and return its eval-mode model and metadata."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("reranker artifact does not exist: {}".format(path))
    artifact = torch.load(path, map_location="cpu")
    model_config = _validate_artifact(artifact)
    model = QueryReranker(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        dropout=float(model_config["dropout"]),
    )
    try:
        model.load_state_dict(artifact["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("reranker artifact model state is incompatible: {}".format(
            error
        ))
    model.to(_resolve_device(device))
    model.eval()
    return model, artifact


def _validate_training_arguments(hidden_dim, dropout, lr, weight_decay,
                                 batch_size, max_epochs, patience):
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("hidden_dim must be a positive integer")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must lie in [0, 1)")
    if not float(lr) > 0.0:
        raise ValueError("lr must be positive")
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be non-negative")
    for name, value in (
            ("batch_size", batch_size),
            ("max_epochs", max_epochs),
            ("patience", patience)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("{} must be a positive integer".format(name))


def _set_deterministic_seed(seed, device):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_reranker(train_cache, output, seed=0, hidden_dim=256, dropout=0.1,
                   lr=1e-3, weight_decay=1e-4, batch_size=32,
                   max_epochs=100, patience=10, device="auto", verbose=False):
    """Train with fit scenes, select on calibration scenes, and save the best."""
    _validate_training_arguments(
        hidden_dim, dropout, lr, weight_decay,
        batch_size, max_epochs, patience,
    )
    device = _resolve_device(device)
    rows, manifest = load_training_cache(train_cache)
    fit_rows, calibration_rows = deterministic_scene_split(rows, seed)
    if not calibration_rows:
        raise ValueError(
            "reranker training needs at least two scenes for calibration"
        )
    feature_mean, feature_std = compute_feature_stats(fit_rows)
    fit_dataset = CandidateCacheDataset(
        fit_rows, feature_mean, feature_std
    )

    _set_deterministic_seed(seed, device)
    model = QueryReranker(
        input_dim=manifest["feature_dim"],
        hidden_dim=hidden_dim,
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    best_metrics = None
    best_reranker_weight = None
    stale_epochs = 0

    for epoch_index in range(int(max_epochs)):
        model.train()
        loader = _make_loader(
            fit_dataset,
            batch_size,
            True,
            int(seed) + epoch_index,
            device,
        )
        loss_sum = 0.0
        sample_count = 0
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            optimizer.zero_grad()
            outputs = model(batch["features"], batch["valid_mask"])
            loss, _ = compute_rec_reranker_loss(
                outputs, batch["candidate_ious"], batch["valid_mask"]
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            current_batch = int(batch["features"].shape[0])
            loss_sum += float(loss.detach().item()) * current_batch
            sample_count += current_batch

        reranker_weight, metrics = select_best_reranker_blend(
            model,
            calibration_rows,
            feature_mean,
            feature_std,
            reranker_weights=DEFAULT_RERANKER_WEIGHTS,
            batch_size=batch_size,
            device=device,
        )
        score = metrics["score"]
        epoch_number = epoch_index + 1
        if verbose:
            print(
                "Epoch {:03d} loss={:.6f} calibration "
                "Acc@0.25={:.5f} Acc@0.50={:.5f} "
                "weight={:.2f} score={:.6f}".format(
                    epoch_number,
                    loss_sum / float(max(sample_count, 1)),
                    metrics["acc025"],
                    metrics["acc050"],
                    reranker_weight,
                    score,
                ),
                flush=True,
            )
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch_number
            best_state = _cpu_state_dict(model)
            best_metrics = metrics
            best_reranker_weight = reranker_weight
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    if (best_state is None or best_metrics is None
            or best_reranker_weight is None):
        raise RuntimeError("reranker training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    restored_metrics = evaluate_reranker(
        model,
        calibration_rows,
        feature_mean,
        feature_std,
        batch_size=batch_size,
        device=device,
        reranker_weight=best_reranker_weight,
    )
    if restored_metrics != best_metrics:
        raise RuntimeError("restored best reranker changed calibration metrics")

    training_args = {
        "train_cache": str(Path(train_cache).expanduser().resolve()),
        "output": str(Path(output).expanduser().resolve()),
        "seed": int(seed),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "device": str(device),
        "grad_clip_norm": GRAD_CLIP_NORM,
        "reranker_weight_grid": list(DEFAULT_RERANKER_WEIGHTS),
        "fit_sample_count": len(fit_rows),
        "calibration_sample_count": len(calibration_rows),
        "fit_scene_count": len({row["scan_id"] for row in fit_rows}),
        "calibration_scene_count": len({
            row["scan_id"] for row in calibration_rows
        }),
    }
    return save_reranker_artifact(
        output,
        model,
        feature_mean,
        feature_std,
        manifest,
        epoch=best_epoch,
        calibration_metrics=best_metrics,
        training_args=training_args,
        reranker_weight=best_reranker_weight,
    )


def parse_args(argv=None):
    """Parse reranker training command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a ScanRefer REC query reranker."
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    try:
        _validate_training_arguments(
            args.hidden_dim,
            args.dropout,
            args.lr,
            args.weight_decay,
            args.batch_size,
            args.max_epochs,
            args.patience,
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def run_training(args):
    artifact = train_reranker(
        train_cache=args.train_cache,
        output=args.output,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        verbose=True,
    )
    metrics = artifact["calibration_metrics"]
    print(
        "Best epoch {} calibration Acc@0.25={:.5f} Acc@0.50={:.5f}".format(
            artifact["epoch"], metrics["acc025"], metrics["acc050"]
        )
    )
    print("Saved reranker artifact: {}".format(
        Path(args.output).expanduser().resolve()
    ))
    return artifact


def main(argv=None):
    run_training(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
