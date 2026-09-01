# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
"""Shared utilities for all main scripts."""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import numbers
import os
import random
import re
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from models.mcln_training_groups import (
    build_mcln_optimizer_param_groups,
    load_mcln_optimizer_state,
    migrate_mcln_scheduler_state,
)
from models.sacr_relation_counterfactual import (
    RELATION_COUNTERFACTUAL_TRAINABLE_PREFIXES,
)
from utils import get_scheduler, setup_logger

from utils import record_tensorboard

from tqdm import tqdm


TRAIN_LOSS_RECEIPT_SCHEMA = "mcln-train-loss-epoch-v1"
CHECKPOINT_RETENTION_SCHEMA = "mcln-checkpoint-retention-v1"
SOURCE_CHOICE_DIAGNOSTICS_SCHEMA = "mcln-source-choice-diagnostics-v1"
FPR_SCENE_AUDIT_SCHEMA = "mcln-fpr-tv-scene-disjoint-audit-v1"
DENSITY_TARGET_BOX_SCENE_AUDIT_SCHEMA = (
    "mcln-density-target-box-scene-disjoint-role-v1"
)
DENSITY_TARGET_BOX_SCENE_AUDIT_E57_SHA256 = (
    "fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
)
FPR_SCENE_DISJOINT_E57_SHA256 = (
    "76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1"
)
FPR_SCENE_DISJOINT_AV4_E57_SHA256 = (
    "fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
)
FPR_SCENE_DISJOINT_CONFIG_SHA256 = (
    "f193d6ab0bbadba2a2e3331bb73d53d78ba1d5abf49dbe662c62eec0bb701c35"
)
FPR_SCENE_DISJOINT_AV4_CONFIG_SHA256 = (
    "aaf4d8edc59e99e056f294b4c031467d2570fb43a879099261b4048054ce4177"
)
CHECKPOINT_RETENTION_METRICS = (
    "rec_acc025",
    "rec_acc050",
    "mask_acc025",
    "mask_acc050",
    "mask_miou",
)
STRUCTURED_COLLATE_KEYS = {
    "target_spans",
    "entity_spans",
    "attr_spans",
    "rel_spans",
    "coverage_stats",
    "decomposition_status",
}


def _gradient_accumulation_plan(loader_batch_count, max_train_batches,
                                accumulation_steps,
                                drop_incomplete_accumulation_group=False):
    """Return one authoritative micro-batch/optimizer-step plan."""
    requested_batch_count = (
        loader_batch_count
        if max_train_batches <= 0
        else min(loader_batch_count, max_train_batches)
    )
    dropped_batch_count = 0
    if drop_incomplete_accumulation_group:
        dropped_batch_count = requested_batch_count % accumulation_steps
    effective_batch_count = requested_batch_count - dropped_batch_count
    if effective_batch_count <= 0:
        raise ValueError(
            "gradient accumulation plan has no complete training group"
        )
    optimizer_step_count = math.ceil(
        effective_batch_count / accumulation_steps
    )
    return {
        "requested_batch_count": requested_batch_count,
        "effective_batch_count": effective_batch_count,
        "dropped_batch_count": dropped_batch_count,
        "optimizer_step_count": optimizer_step_count,
    }


def joint_det_structured_collate(batch):
    """Preserve variable-length structured annotations within a batch."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    keys = set(batch[0])
    if any(set(sample) != keys for sample in batch):
        raise ValueError("all samples in a batch must expose the same keys")
    collated = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key in STRUCTURED_COLLATE_KEYS:
            collated[key] = values
        else:
            collated[key] = default_collate(values)
    return collated


def _requires_joint_det_structured_collate(args):
    """Return whether this run consumes variable-length structured fields."""
    return any(bool(getattr(args, name, False)) for name in (
        "use_sacr_source",
        "use_sacr_score_refiner",
        "use_parent_relative_text_verifier",
    ))


def is_counterfactual_parent_bounded_audit(args):
    """Return whether the exact bounded A-V4 train-only path is active."""
    return (
        bool(getattr(
            args,
            "parent_relative_text_verifier_counterfactual_training",
            False,
        ))
        and int(getattr(args, "max_train_batches", 0)) > 0
    )


def _optional_test_dataset_size(args, test_loader):
    """Return the test size, or None only for the bounded train-only audit."""
    if test_loader is None:
        if not is_counterfactual_parent_bounded_audit(args):
            raise ValueError("only the bounded train-only audit may omit test")
        return None
    return len(test_loader.dataset)


def prepare_parent_relative_text_verifier_score_gradient_audit(end_points):
    """Retain the two A-V4 score axes before backward for audit evidence."""
    actual_verifier_batch = end_points.get(
        "parent_relative_text_verifier_batch"
    )
    actual_score_axis = (
        actual_verifier_batch.get("default_scores")
        if isinstance(actual_verifier_batch, dict) else None
    )
    if not torch.is_tensor(actual_score_axis):
        raise ValueError("actual Parent score-gradient audit is missing")
    if not actual_score_axis.requires_grad:
        raise ValueError(
            "actual Parent score-gradient audit does not require gradients"
        )
    actual_score_axis.retain_grad()

    counterfactual_verifier_batch = end_points.get(
        "parent_relative_text_verifier_counterfactual_batch"
    )
    if counterfactual_verifier_batch is None:
        return actual_score_axis, None
    counterfactual_score_axis = (
        counterfactual_verifier_batch.get("default_scores")
        if isinstance(counterfactual_verifier_batch, dict) else None
    )
    if not torch.is_tensor(counterfactual_score_axis):
        raise ValueError(
            "counterfactual Parent score-gradient audit is missing"
        )
    if not counterfactual_score_axis.requires_grad:
        raise ValueError(
            "counterfactual Parent score-gradient audit does not require "
            "gradients"
        )
    counterfactual_score_axis.retain_grad()
    return actual_score_axis, counterfactual_score_axis


def save_eval_metrics_receipt(log_dir, epoch, metrics):
    """Atomically persist the exact evaluator counters for one checkpoint."""
    if metrics is None:
        return None
    if not isinstance(metrics, dict):
        raise ValueError("evaluation metrics receipt must be a dictionary")
    output = os.path.join(log_dir, "eval_metrics_epoch_{}.json".format(epoch))
    temporary = output + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


def save_source_choice_diagnostics_receipt(log_dir, epoch, diagnostics):
    """Atomically persist optional source-choice oracle diagnostics."""
    if diagnostics is None:
        return None
    if (
            not isinstance(diagnostics, dict)
            or diagnostics.get("schema") != SOURCE_CHOICE_DIAGNOSTICS_SCHEMA):
        raise ValueError("source-choice diagnostics receipt is invalid")
    output = os.path.join(
        log_dir, "source_choice_diagnostics_epoch_{}.json".format(epoch)
    )
    temporary = output + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


_FPR_AUDIT_COUNT_NAMES = (
    "audit_sample_count",
    "audit_switch_count",
    "audit_fix025_count",
    "audit_break025_count",
    "audit_kept_correct025_count",
    "audit_kept_wrong025_count",
    "audit_fix050_count",
    "audit_break050_count",
    "audit_kept_correct050_count",
    "audit_kept_wrong050_count",
)


def build_parent_relative_text_verifier_audit_diagnostics(
        stat_dict, expected_sample_count):
    """Validate and summarize exact held-out FPR-TV decision counts."""
    if not isinstance(stat_dict, dict):
        raise ValueError("FPR scene-audit statistics must be a dictionary")
    if (
            not isinstance(expected_sample_count, int)
            or isinstance(expected_sample_count, bool)
            or expected_sample_count <= 0):
        raise ValueError("expected FPR audit sample count must be positive")

    counts = {}
    for name in _FPR_AUDIT_COUNT_NAMES:
        key = "parent_relative_text_verifier_{}".format(name)
        if key not in stat_dict:
            raise ValueError("FPR scene audit is missing {}".format(key))
        value = stat_dict[key]
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("{} must be scalar".format(key))
            value = value.detach().reshape(()).cpu().item()
        if (
                not isinstance(value, numbers.Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))):
            raise ValueError("{} must be finite numeric".format(key))
        rounded = int(round(float(value)))
        if abs(float(value) - float(rounded)) > 1e-6 or rounded < 0:
            raise ValueError("{} must be a non-negative exact count".format(key))
        counts[name] = rounded

    sample_count = counts["audit_sample_count"]
    if sample_count != expected_sample_count:
        raise ValueError(
            "FPR scene audit contains {} samples, expected {}".format(
                sample_count, expected_sample_count
            )
        )
    switch_count = counts["audit_switch_count"]
    if switch_count > sample_count:
        raise ValueError("FPR switch count exceeds sample count")

    result = {
        "schema": "mcln-fpr-tv-decision-counts-v1",
        "sample_count": sample_count,
        "switch_count": switch_count,
        "switch_rate": switch_count / float(sample_count),
        "thresholds": {},
    }
    for suffix, threshold in (("025", 0.25), ("050", 0.50)):
        fix = counts["audit_fix{}_count".format(suffix)]
        broken = counts["audit_break{}_count".format(suffix)]
        kept_correct = counts[
            "audit_kept_correct{}_count".format(suffix)
        ]
        kept_wrong = counts["audit_kept_wrong{}_count".format(suffix)]
        if fix + broken + kept_correct + kept_wrong != sample_count:
            raise ValueError(
                "FPR @{} transition counts do not partition the audit set"
                .format(threshold)
            )
        if fix + broken > switch_count:
            raise ValueError(
                "FPR @{} changed decisions exceed switches".format(
                    threshold
                )
            )
        parent_hits = kept_correct + broken
        selected_hits = kept_correct + fix
        result["thresholds"][suffix] = {
            "threshold": threshold,
            "fix_count": fix,
            "break_count": broken,
            "kept_correct_count": kept_correct,
            "kept_wrong_count": kept_wrong,
            "parent_hits": parent_hits,
            "selected_hits": selected_hits,
            "parent_accuracy": parent_hits / float(sample_count),
            "selected_accuracy": selected_hits / float(sample_count),
            "net_hits": fix - broken,
            "fix_per_switch": (
                fix / float(switch_count) if switch_count else 0.0
            ),
            "transition_precision": (
                fix / float(fix + broken) if fix + broken else 0.0
            ),
        }
    return result


_FPR_AUDIT_TRAINABLE_PREFIXES = (
    "structured_slot_builder.",
    "sacr_head.",
    "parent_relative_text_verifier.",
)
_FPR_AUDIT_SENTINEL_KEYS = (
    "last_center",
    "last_pred_size",
    "last_pred_masks",
    "sp_last_pred_masks",
    "adaptive_weights",
    "parent_relative_text_verifier_parent_scores",
)
_FPR_SCENE_DISJOINT_FIXED_CONFIG = (
    ("rng_seed", 0),
    ("model", "MCLN"),
    ("num_target", 256),
    ("sampling", "kps"),
    ("num_encoder_layers", 3),
    ("num_decoder_layers", 6),
    ("self_position_embedding", "loc_learned"),
    ("query_points_obj_topk", 4),
    ("use_color", True),
    ("use_height", False),
    ("use_multiview", False),
    ("batch_size", 16),
    ("gradient_accumulation_steps", 1),
    ("drop_incomplete_accumulation_group", False),
    ("max_train_batches", 0),
    ("start_epoch", 58),
    ("joint_det", True),
    ("butd", False),
    ("butd_gt", False),
    ("butd_cls", True),
    ("augment_det", False),
    ("detect_intermediate", True),
    ("use_soft_token_loss", True),
    ("use_contrastive_align", True),
    ("self_attend", True),
    ("skip_missing_superpoints", True),
    ("hard_example_replay_manifest", ""),
    ("hard_example_replay_manifest_sha256", ""),
    ("optimizer", "adamW"),
    ("weight_decay", 5e-4),
    ("lr", 1e-4),
    ("lr_backbone", 1e-3),
    ("text_encoder_lr", 1e-5),
    ("lr_scheduler", "step"),
    ("lr_decay_epochs", [150]),
    ("lr_decay_rate", 0.1),
    ("clip_norm", 0.1),
    ("warmup_epoch", -1),
    ("migrate_scheduler_for_gradient_accumulation", False),
    ("resume_lr_scale", 1.0),
    ("resume_lr_scale_expected_lineage", None),
    ("restore_e57_lr_to_initial", False),
    ("model_only_initialization", False),
    ("checkpoint_start_epoch", None),
    ("use_source_choice_selector", True),
    ("source_choice_selector_train_only", False),
    ("eval_use_selector_choice_scores", True),
    (
        "source_choice_selector_sources",
        "default,default_rank_blend_contrastive010",
    ),
    ("source_choice_selector_default_source", "default"),
    ("source_choice_selector_hidden_dim", 288),
    ("source_choice_selector_lr", 1.25e-4),
    ("source_choice_selector_loss_weight", 0.5),
    (
        "source_choice_selector_choice_target",
        "precision_gain_default_sourcewise_focal_bce",
    ),
    ("source_choice_selector_min_iou_gap", 0.03),
    ("sacr_hidden_dim", 288),
    ("sacr_max_pairs", 3),
    ("sacr_top_m_targets", 32),
    ("sacr_top_k_anchors", 16),
    ("sacr_geo_dim", 16),
    ("sacr_min_parse_confidence", 0.0),
    ("use_parent_relative_text_verifier", True),
    ("parent_relative_text_verifier_train_only", True),
    ("parent_relative_text_verifier_top_k", 5),
    ("parent_relative_text_verifier_max_candidates", 10),
    ("parent_relative_text_verifier_hidden_dim", 256),
    ("parent_relative_text_verifier_heads", 4),
    ("parent_relative_text_verifier_dropout", 0.1),
    ("parent_relative_text_verifier_max_parent_score_gap", 0.25),
    ("parent_relative_text_verifier_promotion_margin", 1e-4),
    ("parent_relative_text_verifier_min_parse_confidence", 0.5),
    ("parent_relative_text_verifier_min_anchor_mass", 0.5),
    ("parent_relative_text_verifier_promotion_epsilon", 1e-4),
    ("parent_relative_text_verifier_detach_inputs", False),
    ("parent_relative_text_verifier_lr", 3e-4),
    ("parent_relative_text_verifier_loss_weight", 1.0),
    ("parent_relative_text_verifier_positive_margin", 0.25),
    ("parent_relative_text_verifier_neutral_margin", 0.25),
    ("use_source_moe", False),
    ("use_sacr_source", False),
    ("use_sacr_score_refiner", False),
    ("rec_reranker_checkpoint", None),
    ("eval_use_rec_reranker_scores", False),
    ("rec_geometry_reranker_checkpoint", None),
    ("eval_use_rec_geometry_reranker_scores", False),
    ("rec_joint_box_mask_checkpoint", None),
    ("eval_use_rec_joint_box_mask", False),
    ("rec_selective_residual_checkpoint", None),
    ("eval_use_rec_selective_residual_scores", False),
    ("rec_hierarchical_reranker_checkpoint", None),
    ("eval_use_rec_hierarchical_reranker_scores", False),
    ("relation_counterfactual_aux_loss_weight", 0.0),
    ("tier_hard_query_aux_loss_weight", 0.0),
    ("checkpoint_metric_retention", False),
    ("max_epoch", 58),
    ("val_freq", 1),
)

_FPR_SCENE_DISJOINT_DYNAMIC_CONFIG_FIELDS = frozenset({
    "checkpoint_path",
    "exp",
    "fpr_scene_disjoint_expected_fit_samples",
    "fpr_scene_disjoint_expected_fit_scenes",
    "fpr_scene_disjoint_expected_holdout_samples",
    "fpr_scene_disjoint_expected_holdout_scenes",
    "fpr_scene_disjoint_fold",
    "log_dir",
    "expected_eval_sample_count",
    "pp_checkpoint",
    # Added after all v1/v2 fold contracts were preregistered.  The legacy
    # audit explicitly rejects it below, then excludes the false default from
    # the historical canonical byte stream so consumed folds remain exact.
    "parent_relative_text_verifier_counterfactual_training",
    "fpr_scene_disjoint_av4_audit",
})


def fpr_scene_sample_identity_digest(sample_ids):
    """Return an order-independent digest of unique audit sample ids."""
    values = list(sample_ids)
    if any(
            not isinstance(value, int) or isinstance(value, bool)
            or value < 0 for value in values):
        raise ValueError("FPR scene sample ids must be nonnegative integers")
    if len(set(values)) != len(values):
        raise ValueError("FPR scene sample ids must be unique")
    encoded = json.dumps(
        sorted(values), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_fpr_scene_disjoint_config_receipt(args):
    """Build the complete canonical config without the preregistered gate."""
    if list(getattr(args, "dataset", ())) != ["nr3d"]:
        raise ValueError("FPR scene audit config requires dataset=[nr3d]")
    if getattr(args, "test_dataset", None) != "nr3d":
        raise ValueError("FPR scene audit config requires test_dataset=nr3d")
    if str(getattr(args, "legacy_scene_graph_cache", "") or ""):
        raise ValueError("FPR scene audit config requires online raw parsing")
    counterfactual_training = bool(getattr(
        args, "parent_relative_text_verifier_counterfactual_training", False
    ))
    av4_audit = bool(getattr(
        args, "fpr_scene_disjoint_av4_audit", False
    ))
    if counterfactual_training and not av4_audit:
        raise ValueError(
            "counterfactual Parent training requires explicit A-V4 audit; "
            "legacy FPR scene audits forbid counterfactual Parent training"
        )
    if av4_audit and not counterfactual_training:
        raise ValueError(
            "A-V4 scene audit requires counterfactual Parent training"
        )
    fixed_values = {}
    for name, expected in _FPR_SCENE_DISJOINT_FIXED_CONFIG:
        actual = getattr(args, name, None)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "FPR scene audit fixed config {} drifted: {!r} != {!r}"
                .format(name, actual, expected)
            )
        fixed_values[name] = copy.deepcopy(actual)
    values = {}
    for name in sorted(vars(args)):
        if name in _FPR_SCENE_DISJOINT_DYNAMIC_CONFIG_FIELDS:
            continue
        values[name] = copy.deepcopy(getattr(args, name))
    values["dataset"] = ["nr3d"]
    values["test_dataset"] = "nr3d"
    values["legacy_scene_graph_cache"] = ""
    if av4_audit:
        fixed_values[
            "parent_relative_text_verifier_counterfactual_training"
        ] = True
        fixed_values["fpr_scene_disjoint_av4_audit"] = True
        values[
            "parent_relative_text_verifier_counterfactual_training"
        ] = True
        values["fpr_scene_disjoint_av4_audit"] = True
    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": (
            "mcln-fpr-tv-av4-scene-fold-config-v1" if av4_audit
            else "mcln-fpr-tv-five-fold-config-v2"
        ),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "fixed_values": fixed_values,
        "values": values,
    }


def build_fpr_scene_disjoint_config_receipt(args):
    """Validate the one preregistered config shared by all five folds."""
    receipt = _canonical_fpr_scene_disjoint_config_receipt(args)
    expected_sha256 = (
        FPR_SCENE_DISJOINT_AV4_CONFIG_SHA256
        if receipt["schema"] == "mcln-fpr-tv-av4-scene-fold-config-v1"
        else FPR_SCENE_DISJOINT_CONFIG_SHA256
    )
    if receipt["sha256"] != expected_sha256:
        raise ValueError(
            "FPR scene audit configuration SHA-256 drifted: {} != {}"
            .format(
                receipt["sha256"], expected_sha256
            )
        )
    return receipt


def _audit_hash_record(digest, value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        try:
            payload = tensor.numpy().tobytes(order="C")
        except TypeError:
            payload = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value):
            _audit_hash_record(digest, str(key))
            _audit_hash_record(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(
            (b"list\0" if isinstance(value, list) else b"tuple\0")
        )
        digest.update(len(value).to_bytes(8, "little"))
        for item in value:
            _audit_hash_record(digest, item)
        return
    if isinstance(value, str):
        payload = value.encode("utf-8")
        digest.update(b"str\0")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, (bool, int, float)):
        digest.update(
            (type(value).__name__ + "\0" + repr(value)).encode("ascii")
        )
        return
    raise ValueError(
        "unsupported FPR audit value type {}".format(type(value).__name__)
    )


def _audit_value_sha256(value):
    digest = hashlib.sha256()
    _audit_hash_record(digest, value)
    return digest.hexdigest()


def capture_fpr_audit_model_state(model):
    """Hash trainable and frozen model state as disjoint partitions."""
    unwrapped = model.module if hasattr(model, "module") else model
    state = unwrapped.state_dict()
    partitions = {
        "trainable": [],
        "frozen": [],
    }
    for name in sorted(state):
        partition = (
            "trainable"
            if any(name.startswith(prefix)
                   for prefix in _FPR_AUDIT_TRAINABLE_PREFIXES)
            else "frozen"
        )
        partitions[partition].append((name, state[name]))
    if not partitions["trainable"] or not partitions["frozen"]:
        raise ValueError("FPR audit state partition is empty")
    receipt = {}
    for partition, records in partitions.items():
        digest = hashlib.sha256()
        total_numel = 0
        for name, tensor in records:
            _audit_hash_record(digest, name)
            _audit_hash_record(digest, tensor)
            total_numel += int(tensor.numel())
        receipt[partition] = {
            "sha256": digest.hexdigest(),
            "tensor_count": len(records),
            "numel": total_numel,
        }
    return receipt


def capture_fpr_audit_output_state(end_points):
    """Hash exact Box/Mask/parent-score sentinel outputs."""
    if not isinstance(end_points, dict):
        raise ValueError("FPR audit sentinel output must be a dictionary")
    missing = [
        key for key in _FPR_AUDIT_SENTINEL_KEYS if key not in end_points
    ]
    if missing:
        raise ValueError(
            "FPR audit sentinel outputs are missing: {}".format(
                ", ".join(missing)
            )
        )
    per_key = {
        key: _audit_value_sha256(end_points[key])
        for key in _FPR_AUDIT_SENTINEL_KEYS
    }
    return {
        "schema": "mcln-fpr-tv-frozen-output-sentinel-v1",
        "keys": per_key,
        "combined_sha256": _audit_value_sha256(per_key),
    }


_SOURCE_MOE_GATE_DECISION_COUNT_KEYS = {
    "sample_count": "source_moe_gate_supervised_sample_count",
    "oracle_switch_count": "source_moe_gate_oracle_switch_count",
    "predicted_switch_count": "source_moe_gate_predicted_switch_count",
    "beneficial_switch_count": "source_moe_gate_beneficial_switch_count",
    "harmful_switch_count": "source_moe_gate_harmful_switch_count",
    "oracle_query_match_count": "source_moe_gate_oracle_query_match_count",
}
_SOURCE_MOE_GATE_OPTIONAL_DECISION_COUNT_KEYS = {
    "row_target_switch_count": "source_moe_gate_row_target_switch_count",
}


def build_source_moe_gate_decision_diagnostics(stat_dict):
    """Convert accumulated gate counts into exact validation rates."""
    if not isinstance(stat_dict, dict):
        raise ValueError("gate decision statistics must be a dictionary")
    present = {
        name: key in stat_dict
        for name, key in _SOURCE_MOE_GATE_DECISION_COUNT_KEYS.items()
    }
    if not any(present.values()):
        return None
    if not all(present.values()):
        missing = sorted(name for name, exists in present.items() if not exists)
        raise ValueError(
            "gate decision statistics are incomplete: {}".format(
                ", ".join(missing)
            )
        )

    counts = {}
    for name, key in _SOURCE_MOE_GATE_DECISION_COUNT_KEYS.items():
        value = stat_dict[key]
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("{} must be a scalar count".format(key))
            value = value.detach().reshape(()).cpu().item()
        if (not isinstance(value, numbers.Real) or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or not math.isclose(float(value), round(float(value)),
                                    rel_tol=0.0, abs_tol=1e-9)):
            raise ValueError("{} must be a non-negative integer".format(key))
        counts[name] = int(round(float(value)))

    optional_counts = {}
    for name, key in _SOURCE_MOE_GATE_OPTIONAL_DECISION_COUNT_KEYS.items():
        if key not in stat_dict:
            continue
        value = stat_dict[key]
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("{} must be a scalar count".format(key))
            value = value.detach().reshape(()).cpu().item()
        if (not isinstance(value, numbers.Real) or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or not math.isclose(float(value), round(float(value)),
                                    rel_tol=0.0, abs_tol=1e-9)):
            raise ValueError("{} must be a non-negative integer".format(key))
        optional_counts[name] = int(round(float(value)))

    sample_count = counts["sample_count"]
    if sample_count <= 0:
        raise ValueError("gate decision sample_count must be positive")
    oracle_count = counts["oracle_switch_count"]
    predicted_count = counts["predicted_switch_count"]
    beneficial_count = counts["beneficial_switch_count"]
    harmful_count = counts["harmful_switch_count"]
    query_match_count = counts["oracle_query_match_count"]
    if oracle_count > sample_count or predicted_count > sample_count:
        raise ValueError("gate switch counts exceed sample_count")
    if any(value > sample_count for value in optional_counts.values()):
        raise ValueError("gate optional switch counts exceed sample_count")
    if beneficial_count + harmful_count != predicted_count:
        raise ValueError(
            "beneficial and harmful switches must partition predictions"
        )
    if beneficial_count > oracle_count or query_match_count > oracle_count:
        raise ValueError("gate oracle-derived counts exceed oracle switches")

    diagnostics = {
        **counts,
        "oracle_switch_rate": oracle_count / float(sample_count),
        "predicted_switch_rate": predicted_count / float(sample_count),
        "oracle_switch_recall": (
            beneficial_count / float(oracle_count)
            if oracle_count else 0.0
        ),
        "predicted_switch_precision": (
            beneficial_count / float(predicted_count)
            if predicted_count else 0.0
        ),
        "false_switch_rate": (
            harmful_count / float(predicted_count)
            if predicted_count else 0.0
        ),
        "oracle_query_match_rate": (
            query_match_count / float(oracle_count)
            if oracle_count else 0.0
        ),
    }
    if "row_target_switch_count" in optional_counts:
        diagnostics["row_target_switch_count"] = optional_counts[
            "row_target_switch_count"
        ]
        diagnostics["row_target_switch_rate"] = (
            optional_counts["row_target_switch_count"]
            / float(sample_count)
        )
    return diagnostics

def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_target', type=int, default=256,
                        help='Proposal number')
    parser.add_argument('--sampling', default='kps', type=str,
                        help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)    # 6
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')
    parser.add_argument('--model', type=str, default='BeaUTyDETR')

    # Loss
    parser.add_argument('--query_points_obj_topk', default=4, type=int)
    parser.add_argument('--use_contrastive_align', action='store_true')
    parser.add_argument('--use_soft_token_loss', action='store_true')
    parser.add_argument('--detect_intermediate', action='store_true')
    parser.add_argument('--joint_det', action='store_true')
    parser.add_argument('--use_source_choice_selector', action='store_true',
                        default=False)
    parser.add_argument('--source_choice_selector_train_only',
                        action='store_true', default=False)
    parser.add_argument('--source_choice_selector_lr', type=float,
                        default=0.001)
    parser.add_argument('--source_choice_selector_hidden_dim', type=int,
                        default=288)
    parser.add_argument('--source_choice_selector_loss_weight', type=float,
                        default=0.0)
    parser.add_argument('--source_choice_selector_sources', type=str,
                        default='default,mask_text')
    parser.add_argument('--source_choice_selector_default_source', type=str,
                        default='default')
    parser.add_argument(
        '--source_choice_selector_choice_target',
        type=str,
        default='precision_gain_default_sourcewise_focal_bce',
        choices=[
            'precision_gain_default_sourcewise_focal_bce',
            'precision_gain_default_ce',
        ],
    )
    parser.add_argument('--source_choice_selector_min_iou_gap', type=float,
                        default=0.05)
    parser.add_argument('--use_source_moe', action='store_true', default=False)
    parser.add_argument('--source_moe_train_only', action='store_true',
                        default=False)
    parser.add_argument('--source_moe_gate_train_only', action='store_true',
                        default=False)
    parser.add_argument('--source_moe_gate_new_heads_only',
                        action='store_true', default=False)
    parser.add_argument('--source_moe_gate_resume_optimizer',
                        action='store_true', default=False)
    parser.add_argument('--source_moe_shared_source', type=str,
                        default='default')
    parser.add_argument('--source_moe_top_k', type=int, default=2)
    parser.add_argument('--source_moe_balance_loss_weight', type=float,
                        default=0.01)
    parser.add_argument('--source_moe_rank_loss_weight', type=float,
                        default=1.0)
    parser.add_argument('--source_moe_mask_rank_loss_weight', type=float,
                        default=0.25)
    parser.add_argument('--source_moe_rank_temperature', type=float,
                        default=0.1)
    parser.add_argument('--source_moe_anchor_loss_weight', type=float,
                        default=0.0)
    parser.add_argument('--source_moe_anchor_margin', type=float, default=0.05)
    parser.add_argument('--source_moe_query_layers', type=int, default=1)
    parser.add_argument('--source_moe_query_heads', type=int, default=4)
    parser.add_argument('--source_moe_query_dropout', type=float, default=0.1)
    parser.add_argument('--source_moe_query_max_delta', type=float,
                        default=0.25)
    parser.add_argument('--source_moe_lr', type=float, default=0.001)
    parser.add_argument('--source_moe_use_fallback_gate', action='store_true',
                        default=False)
    parser.add_argument('--source_moe_gate_hidden_dim', type=int, default=128)
    parser.add_argument('--source_moe_gate_candidate_top_k', type=int,
                        default=8)
    parser.add_argument('--source_moe_gate_break_cost', type=float,
                        default=2.0)
    parser.add_argument('--source_moe_gate_decision_margin', type=float,
                        default=0.0)
    parser.add_argument('--source_moe_gate_mask_utility_weight', type=float,
                        default=0.25)
    parser.add_argument('--source_moe_gate_uncertainty_weight', type=float,
                        default=None)
    parser.add_argument('--source_moe_gate_use_evidence_features',
                        action='store_true', default=False)
    parser.add_argument('--source_moe_gate_context_layers', type=int,
                        default=0)
    parser.add_argument('--source_moe_gate_context_heads', type=int,
                        default=4)
    parser.add_argument('--source_moe_gate_context_dropout', type=float,
                        default=0.1)
    parser.add_argument(
        '--source_moe_gate_action_mode',
        choices=(
            'decision', 'expected_utility', 'direct_utility',
            'hierarchical_utility', 'pairwise_verifier',
            'topn_pairwise_verifier', 'topn_dual_evidence_verifier',
            'topn_absolute_quality_delta',
            'cascade_absolute_quality_correction',
            'cascade_opportunity_quality_correction',
            'cascade_opportunity_verified_correction',
            'cascade_joint_risk_correction',
            'cascade_v19_fallback_set_correction',
            'cascade_v19_rich_set_correction',
            'cascade_v23_dense_quality_correction',
            'cascade_v24_relative_risk_correction',
            'cascade_v25_pairwise_calibrated_correction',
            'cascade_v26_prior_restored_pairwise_correction',
            'cascade_v28_selected_abstention_correction',
            'cascade_v29_counterfactual_selected_correction',
            'cascade_v37_counterfactual_benefit_hazard_correction',
            'cascade_v38_complementary_logodds_correction',
            'cascade_v39_hazard_residual_correction',
        ),
        default=None,
        help='fallback action score; inherit checkpoint mode when unset',
    )
    parser.add_argument('--source_moe_gate_loss_weight', type=float,
                        default=0.0)
    parser.add_argument('--source_moe_gate_mask_loss_weight', type=float,
                        default=0.25)
    parser.add_argument('--source_moe_gate_focal_gamma', type=float,
                        default=2.0)
    parser.add_argument('--source_moe_gate_false_override_weight', type=float,
                        default=2.0)
    parser.add_argument(
        '--source_moe_gate_objective',
        choices=(
            'balanced_focal', 'calibrated_utility',
            'balanced_calibrated_utility',
            'hierarchical_risk_calibrated',
            'pairwise_risk_calibrated',
            'topn_risk_calibrated',
            'topn_dual_risk_calibrated',
            'topn_absolute_quality_calibrated',
            'cascade_absolute_quality_calibrated',
            'cascade_opportunity_balanced_calibrated',
            'cascade_opportunity_verified_calibrated',
            'cascade_joint_risk_calibrated',
            'cascade_v19_fallback_set_risk_calibrated',
            'cascade_v19_rich_set_empirical_risk',
            'cascade_v23_dense_quality_risk',
            'cascade_v24_relative_risk',
            'cascade_v25_pairwise_calibrated_risk',
            'cascade_v26_prior_restored_pairwise_risk',
            'cascade_v27_uncertainty_quality_risk',
            'cascade_v28_selected_abstention_risk',
            'cascade_v29_counterfactual_selected_risk',
            'cascade_v37_counterfactual_benefit_hazard_risk',
            'cascade_v38_complementary_logodds_risk',
            'cascade_v39_hazard_residual_risk',
        ),
        default=None,
        help='gate training objective; inherit checkpoint objective when unset',
    )
    parser.add_argument('--source_moe_gate_setwise_temperature', type=float,
                        default=0.0)
    parser.add_argument('--source_moe_gate_boundary_loss_weight', type=float,
                        default=0.0)
    parser.add_argument('--source_moe_gate_lr', type=float, default=0.0003)
    parser.add_argument('--mask_head_lr_multiplier', type=float, default=1.0)
    parser.add_argument('--use_decoder_query_adapter', action='store_true',
                        default=False)
    parser.add_argument('--decoder_query_adapter_train_only',
                        action='store_true', default=False)
    parser.add_argument('--decoder_query_adapter_lr', type=float,
                        default=0.0003)
    parser.add_argument('--decoder_query_adapter_hidden_dim', type=int,
                        default=288)
    parser.add_argument('--decoder_query_adapter_heads', type=int, default=4)
    parser.add_argument('--decoder_query_adapter_dropout', type=float,
                        default=0.1)
    parser.add_argument('--decoder_query_adapter_max_delta', type=float,
                        default=0.25)
    parser.add_argument('--use_query_mask_fusion_calibrator',
                        action='store_true', default=False)
    parser.add_argument('--query_mask_fusion_train_only',
                        action='store_true', default=False)
    parser.add_argument('--query_mask_fusion_resume_optimizer',
                        action='store_true', default=False)
    parser.add_argument('--query_mask_fusion_lr', type=float, default=0.001)
    parser.add_argument('--query_mask_fusion_hidden_dim', type=int, default=128)
    parser.add_argument('--query_mask_fusion_dropout', type=float, default=0.0)
    parser.add_argument('--query_mask_fusion_max_delta', type=float, default=0.25)
    parser.add_argument('--use_egqs_mask_refiner', action='store_true',
                        default=False)
    parser.add_argument('--egqs_mask_refiner_train_only', action='store_true',
                        default=False)
    parser.add_argument('--egqs_mask_refiner_lr', type=float, default=0.0003)
    parser.add_argument(
        '--egqs_mask_refiner_arch', type=str, default='egqs',
        choices=('egqs', 'graph'),
    )
    parser.add_argument('--egqs_mask_refiner_hidden_dim', type=int, default=32)
    parser.add_argument('--egqs_mask_refiner_max_delta', type=float, default=2.0)
    parser.add_argument(
        '--egqs_mask_refiner_components', type=str, default='all',
        choices=('content', 'evidence', 'geometry', 'all'),
    )
    parser.add_argument(
        '--egqs_mask_refiner_graph_mode', type=str, default='bilateral',
        choices=('spatial', 'bilateral'),
    )
    parser.add_argument(
        '--egqs_mask_refiner_neighbor_count', type=int, default=8,
    )
    parser.add_argument('--use_joint_query_quality_reranker',
                        action='store_true', default=False)
    parser.add_argument('--joint_query_quality_train_only',
                        action='store_true', default=False)
    parser.add_argument('--joint_query_quality_lr', type=float, default=0.001)
    parser.add_argument('--joint_query_quality_hidden_dim', type=int,
                        default=128)
    parser.add_argument('--joint_query_quality_heads', type=int, default=4)
    parser.add_argument('--joint_query_quality_layers', type=int, default=1)
    parser.add_argument('--joint_query_quality_dropout', type=float,
                        default=0.1)
    parser.add_argument('--joint_query_quality_max_delta', type=float,
                        default=1.25)
    parser.add_argument('--joint_query_quality_mask_weight', type=float,
                        default=0.25)
    parser.add_argument('--joint_query_quality_score_weight', type=float,
                        default=1.0)
    parser.add_argument(
        '--joint_query_quality_direct_residual_scale',
        type=float, default=1.0,
    )
    parser.add_argument(
        '--joint_query_quality_use_metric_aligned_utility',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_preserve_parent_score',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_candidate_promotion_margin',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_use_parent_transition_advantage',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_decomposed_transition_advantage',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_setwise_tier_advantage',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_decoupled_setwise_heads',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_factorized_setwise_safety',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_factorized_setwise_risk_bound',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_setwise_safety_veto_gate',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_cost_calibrated_setwise_risk_bound',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_setwise_safety_slack_quantile_bound',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_setwise_safety_slack_pairwise_order',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_proposal_conditioned_safety',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_parent_referenced_safety',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_coupled_safe_repair_witness',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_bidirectional_coupled_boundary',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_centered_coupled_separation',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_hazard_conditioned_coupled_separation',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_monotonic_box_safety_folding',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_same_candidate_branchwise_witness',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_parent_non_degradation_certificate',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_criterion_responsible_hazard_attribution',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_independent_joint_hazard_certificate',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_frozen_raw_joint_hazard_features',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_factorized_hit_advantage',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_use_factorized_nested_dominance',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_factorized_hit_break_cost',
        type=float, default=4.0,
    )
    parser.add_argument(
        '--joint_query_quality_parent_transition_break_cost',
        type=float, default=4.0,
    )
    parser.add_argument(
        '--joint_query_quality_parent_transition_candidate_top_k',
        type=int, default=0,
    )
    parser.add_argument('--joint_query_quality_use_mask_calibration',
                        action='store_true', default=False)
    parser.add_argument('--joint_query_quality_max_mask_alpha_delta',
                        type=float, default=1.0)
    parser.add_argument('--joint_query_quality_max_mask_logit_bias',
                        type=float, default=2.0)
    parser.add_argument('--joint_query_quality_use_source_mask_evidence',
                        action='store_true', default=False)
    parser.add_argument('--joint_query_quality_use_gate_evidence',
                        action='store_true', default=False)
    parser.add_argument('--joint_query_quality_use_spatial_mask_refiner',
                        action='store_true', default=False)
    parser.add_argument('--joint_query_quality_spatial_mask_hidden_dim',
                        type=int, default=32)
    parser.add_argument('--joint_query_quality_max_spatial_mask_delta',
                        type=float, default=2.0)
    parser.add_argument('--joint_query_quality_use_adaptive_source_mixing',
                        action='store_true', default=False)
    parser.add_argument(
        '--joint_query_quality_use_source_distribution_reliability',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_source_names', type=str, default='',
        help='optional source pool used only by joint adaptive mixing',
    )
    parser.add_argument('--joint_query_quality_max_source_mix_delta',
                        type=float, default=1.0)
    parser.add_argument('--joint_query_quality_source_mix_temperature',
                        type=float, default=0.5)
    parser.add_argument('--joint_query_quality_loss_weight', type=float,
                        default=1.0)
    parser.add_argument('--joint_query_quality_temperature', type=float,
                        default=0.25)
    parser.add_argument('--joint_query_quality_aux_loss_weight', type=float,
                        default=1.0)
    parser.add_argument('--joint_query_quality_anchor_loss_weight', type=float,
                        default=0.5)
    parser.add_argument('--joint_query_quality_anchor_margin', type=float,
                        default=0.05)
    parser.add_argument(
        '--joint_query_quality_bidirectional_anchor',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--joint_query_quality_anchor_margin_050', type=float, default=0.10,
    )
    parser.add_argument(
        '--joint_query_quality_metric_utility_temperature',
        type=float, default=0.05,
    )
    parser.add_argument(
        '--joint_query_quality_pairwise_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_listwise_loss_weight',
        type=float, default=1.0,
    )
    parser.add_argument(
        '--joint_query_quality_transition_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_setwise_repair_boundary_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_setwise_negative_tail_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_setwise_rank_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_setwise_dense_safety_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_setwise_balanced_safety_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_setwise_factorized_safety_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_setwise_factorized_risk_bound_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_factorized_hit_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_factorized_pair_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_transition_break_cost',
        type=float, default=4.0,
    )
    parser.add_argument(
        '--joint_query_quality_transition_neutral_weight',
        type=float, default=0.25,
    )
    parser.add_argument(
        '--joint_query_quality_deploy_candidate_top_k',
        type=int, default=0,
    )
    parser.add_argument(
        '--joint_query_quality_source_candidate_top_k',
        type=int, default=0,
    )
    parser.add_argument(
        '--joint_query_quality_oracle_candidate_top_k',
        type=int, default=0,
    )
    parser.add_argument('--joint_query_quality_source_mix_loss_weight',
                        type=float, default=0.0)
    parser.add_argument('--joint_query_quality_source_mix_alignment_temperature',
                        type=float, default=0.25)
    parser.add_argument(
        '--joint_query_quality_source_mix_query_focus_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_candidate_mask_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_candidate_lovasz_loss_weight',
        type=float, default=0.0,
    )
    parser.add_argument(
        '--joint_query_quality_candidate_mask_top_k', type=int, default=16,
    )
    parser.add_argument('--use_sacr_source', action='store_true', default=False)
    parser.add_argument(
        '--legacy_scene_graph_cache', type=str, default='',
        help=(
            'utterance-only deterministic cache for legacy '
            'Scene_graph_parse outputs'
        ),
    )
    parser.add_argument(
        '--legacy_scene_graph_cache_strict',
        action='store_true', default=False,
        help=(
            'fail if an annotation from a dataset declared by the cache '
            'manifest is absent'
        ),
    )
    parser.add_argument(
        '--legacy_scene_graph_cache_expected_target_selection',
        type=str, default='',
        choices=('', 'first_object', 'conservative_syntax_v1'),
        help='fail if cache target-selection provenance differs',
    )
    parser.add_argument(
        '--legacy_scene_graph_cache_expected_sha256',
        type=str, default='',
        help='required immutable SHA-256 for a raw scene-graph bundle',
    )
    parser.add_argument('--use_sacr_score_refiner', action='store_true',
                        default=False)
    parser.add_argument('--sacr_score_refiner_train_only',
                        action='store_true', default=False)
    parser.add_argument(
        '--use_parent_relative_text_verifier',
        action='store_true', default=False,
        help=(
            'verify a compact Top-K candidate set against the immutable '
            'parent and otherwise abstain'
        ),
    )
    parser.add_argument(
        '--parent_relative_text_verifier_train_only',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--parent_relative_text_verifier_counterfactual_training',
        action='store_true', default=False,
        help=(
            'add fixed training-only text-Top1 and leave-one-out Parent '
            'views; deployment always keeps the actual V99 Parent'
        ),
    )
    parser.add_argument('--parent_relative_text_verifier_top_k', type=int,
                        default=5)
    parser.add_argument(
        '--parent_relative_text_verifier_max_candidates', type=int, default=10
    )
    parser.add_argument(
        '--parent_relative_text_verifier_hidden_dim', type=int, default=256
    )
    parser.add_argument('--parent_relative_text_verifier_heads', type=int,
                        default=4)
    parser.add_argument('--parent_relative_text_verifier_dropout', type=float,
                        default=0.1)
    parser.add_argument(
        '--parent_relative_text_verifier_max_parent_score_gap',
        type=float, default=0.25,
    )
    parser.add_argument(
        '--parent_relative_text_verifier_promotion_margin',
        type=float, default=1e-4,
    )
    parser.add_argument(
        '--parent_relative_text_verifier_min_parse_confidence',
        type=float, default=0.5,
    )
    parser.add_argument(
        '--parent_relative_text_verifier_min_anchor_mass',
        type=float, default=0.5,
    )
    parser.add_argument(
        '--parent_relative_text_verifier_promotion_epsilon',
        type=float, default=1e-4,
    )
    parser.add_argument(
        '--parent_relative_text_verifier_detach_inputs',
        action='store_true', default=False,
    )
    parser.add_argument('--parent_relative_text_verifier_lr', type=float,
                        default=3e-4)
    parser.add_argument(
        '--parent_relative_text_verifier_loss_weight', type=float, default=1.0
    )
    parser.add_argument(
        '--parent_relative_text_verifier_positive_margin',
        type=float, default=0.25,
    )
    parser.add_argument(
        '--parent_relative_text_verifier_neutral_margin',
        type=float, default=0.25,
    )
    parser.add_argument(
        '--fpr_scene_disjoint_audit', action='store_true', default=False,
        help=(
            'train FPR-TV on four deterministic Nr3D train-scene folds and '
            'evaluate only the held-out train-scene fold'
        ),
    )
    parser.add_argument(
        '--fpr_scene_disjoint_av4_audit', action='store_true', default=False,
        help=(
            'use the separately preregistered A-V4 counterfactual-Parent '
            'contract for an FPR scene-disjoint audit'
        ),
    )
    parser.add_argument('--fpr_scene_disjoint_fold', type=int, default=-1)
    parser.add_argument(
        '--fpr_scene_disjoint_expected_fit_scenes', type=int, default=-1
    )
    parser.add_argument(
        '--fpr_scene_disjoint_expected_holdout_scenes', type=int, default=-1
    )
    parser.add_argument(
        '--fpr_scene_disjoint_expected_fit_samples', type=int, default=-1
    )
    parser.add_argument(
        '--fpr_scene_disjoint_expected_holdout_samples', type=int, default=-1
    )
    parser.add_argument(
        '--fpr_scene_disjoint_checkpoint_sha256', type=str, default=''
    )
    parser.add_argument(
        '--sacr_score_use_parent_relative_abstention',
        action='store_true', default=False,
        help='anchor SACR residuals to the parent and abstain per sample',
    )
    parser.add_argument(
        '--sacr_score_use_relation_counterfactual',
        action='store_true', default=False,
        help='mine relation/anchor target swaps instead of generic ranking',
    )
    parser.add_argument('--sacr_score_parent_gate_hidden_dim', type=int,
                        default=32)
    parser.add_argument('--sacr_score_refiner_lr', type=float, default=0.0003)
    parser.add_argument('--sacr_score_refiner_loss_weight', type=float,
                        default=1.0)
    parser.add_argument('--sacr_score_temperature', type=float, default=0.1)
    parser.add_argument('--sacr_score_mask_weight', type=float, default=0.25)
    parser.add_argument('--sacr_score_max_delta', type=float, default=0.25)
    parser.add_argument('--sacr_score_min_box_advantage', type=float,
                        default=0.03)
    parser.add_argument('--sacr_score_promotion_margin', type=float,
                        default=0.01)
    parser.add_argument('--sacr_counterfactual_parent_top_k', type=int,
                        default=16)
    parser.add_argument('--sacr_counterfactual_target_tolerance', type=float,
                        default=0.05)
    parser.add_argument('--sacr_counterfactual_attribute_tolerance', type=float,
                        default=0.05)
    parser.add_argument('--sacr_counterfactual_geometry_threshold', type=float,
                        default=0.08)
    parser.add_argument('--sacr_counterfactual_iou_gap', type=float,
                        default=0.10)
    parser.add_argument('--sacr_counterfactual_correct_iou_threshold',
                        type=float, default=0.25)
    parser.add_argument('--sacr_counterfactual_pair_margin', type=float,
                        default=0.25)
    parser.add_argument('--sacr_counterfactual_max_negatives', type=int,
                        default=4)
    parser.add_argument('--sacr_counterfactual_relation_scale', type=float,
                        default=4.0)
    parser.add_argument('--sacr_counterfactual_deployment_threshold',
                        type=float, default=0.05)
    parser.add_argument('--relation_counterfactual_aux_loss_weight',
                        type=float, default=0.0)
    parser.add_argument('--relation_counterfactual_aux_parent_top_k',
                        type=int, default=32)
    parser.add_argument('--relation_counterfactual_aux_target_tolerance',
                        type=float, default=0.10)
    parser.add_argument('--relation_counterfactual_aux_attribute_tolerance',
                        type=float, default=0.10)
    parser.add_argument('--relation_counterfactual_aux_geometry_threshold',
                        type=float, default=0.08)
    parser.add_argument('--relation_counterfactual_aux_correct_iou_threshold',
                        type=float, default=0.25)
    parser.add_argument('--relation_counterfactual_aux_pair_margin',
                        type=float, default=0.05)
    parser.add_argument('--relation_counterfactual_aux_max_negatives',
                        type=int, default=8)
    parser.add_argument('--relation_counterfactual_aux_target_confidence_floor',
                        type=float, default=0.05)
    parser.add_argument('--relation_counterfactual_aux_attribute_confidence_floor',
                        type=float, default=0.02)
    parser.add_argument('--relation_counterfactual_aux_acc025_pair_weight',
                        type=float, default=2.0)
    parser.add_argument(
        '--relation_counterfactual_aux_conservative_anchor_set',
        action='store_true',
    )
    parser.add_argument('--tier_hard_query_aux_loss_weight',
                        type=float, default=0.0)
    parser.add_argument('--tier_hard_query_aux_candidate_top_k',
                        type=int, default=128)
    parser.add_argument('--tier_hard_query_aux_max_negatives',
                        type=int, default=8)
    parser.add_argument('--tier_hard_query_aux_target_tolerance',
                        type=float, default=0.15)
    parser.add_argument('--tier_hard_query_aux_target_confidence_floor',
                        type=float, default=0.01)
    parser.add_argument('--tier_hard_query_aux_pair_margin',
                        type=float, default=0.05)
    parser.add_argument('--tier_hard_query_aux_preserve_weight',
                        type=float, default=0.25)
    parser.add_argument('--tier_hard_query_aux_acc025_pair_weight',
                        type=float, default=2.0)
    parser.add_argument('--density_aware_target_box_loss_weight',
                        type=float, default=0.0)
    parser.add_argument('--density_aware_target_box_checkpoint_sha256',
                        type=str, default='')
    parser.add_argument(
        '--density_aware_target_box_scene_disjoint_audit',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--density_aware_target_box_scene_disjoint_role',
        choices=('parent', 'control', 'method'), default=None,
    )
    parser.add_argument(
        '--density_aware_target_box_scene_disjoint_fold',
        type=int, default=-1,
    )
    parser.add_argument(
        '--density_aware_target_box_scene_disjoint_expected_fit_scenes',
        type=int, default=-1,
    )
    parser.add_argument(
        '--density_aware_target_box_scene_disjoint_expected_holdout_scenes',
        type=int, default=-1,
    )
    parser.add_argument(
        '--density_aware_target_box_scene_disjoint_expected_fit_samples',
        type=int, default=-1,
    )
    parser.add_argument(
        '--density_aware_target_box_scene_disjoint_expected_holdout_samples',
        type=int, default=-1,
    )
    parser.add_argument('--sacr_score_mask_tolerance', type=float,
                        default=0.02)
    parser.add_argument('--sacr_score_raw_margin', type=float, default=0.1)
    parser.add_argument('--sacr_score_dense_weight', type=float, default=0.25)
    parser.add_argument('--sacr_score_preserve_weight', type=float,
                        default=1.0)
    parser.add_argument('--sacr_score_gate_weight', type=float, default=0.05)
    parser.add_argument('--sacr_score_saturation_weight', type=float,
                        default=0.05)
    parser.add_argument('--sacr_hidden_dim', type=int, default=288)
    parser.add_argument('--sacr_max_pairs', type=int, default=3)
    parser.add_argument('--sacr_top_m_targets', type=int, default=32)
    parser.add_argument('--sacr_top_k_anchors', type=int, default=16)
    parser.add_argument('--sacr_geo_dim', type=int, default=16)
    parser.add_argument('--sacr_min_parse_confidence', type=float, default=0.0)
    parser.add_argument('--sacr_score_contract_audit', action='store_true',
                        default=False)
    parser.add_argument('--sacr_residual_scale_init', type=float, default=0.1)
    parser.add_argument('--mask_loss_scale', type=float, default=1.0)
    parser.add_argument('--consistency_loss_scale', type=float, default=1.0)
    parser.add_argument('--eval_use_selector_choice_scores',
                        action='store_true', default=False)
    parser.add_argument('--expected_eval_sample_count', type=int, default=None)
    parser.add_argument('--rec_reranker_checkpoint', default=None)
    parser.add_argument('--eval_use_rec_reranker_scores',
                        action='store_true', default=False)
    parser.add_argument('--rec_geometry_reranker_checkpoint', default=None)
    parser.add_argument('--eval_use_rec_geometry_reranker_scores',
                        action='store_true', default=False)
    parser.add_argument('--rec_joint_box_mask_checkpoint', default=None)
    parser.add_argument('--eval_use_rec_joint_box_mask',
                        action='store_true', default=False)
    parser.add_argument('--rec_selective_residual_checkpoint', default=None)
    parser.add_argument('--eval_use_rec_selective_residual_scores',
                        action='store_true', default=False)
    parser.add_argument('--rec_hierarchical_reranker_checkpoint', default=None)
    parser.add_argument('--eval_use_rec_hierarchical_reranker_scores',
                        action='store_true', default=False)

    # Data
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size during training')
    parser.add_argument('--dataset', type=str, default=['scanrefer'],
                        nargs='+', help='list of datasets to train on')
    parser.add_argument('--test_dataset', default='scanrefer')
    parser.add_argument('--data_root', default='./')
    parser.add_argument('--use_height', action='store_true',
                        help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true',
                        help='Use RGB color in input.')     # color
    parser.add_argument('--use_multiview', action='store_true')
    parser.add_argument('--wo_obj_name', default='None')    # grounding without object name
    parser.add_argument('--butd', action='store_true')
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataloader_prefetch_factor', type=int, default=2)
    parser.add_argument('--persistent_train_workers', action='store_true',
                        default=False)
    parser.add_argument('--skip_missing_superpoints', action='store_true',
                        default=False)
    parser.add_argument(
        '--hard_example_replay_manifest', type=str, default='',
        help=(
            'Training-only manifest whose fixed hard referring examples are '
            'replayed exactly once per epoch without changing losses.'
        ),
    )
    parser.add_argument(
        '--hard_example_replay_manifest_sha256', type=str, default='',
        help='Exact SHA-256 required with --hard_example_replay_manifest.',
    )

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help=(
            'Number of consecutive micro-batches whose mean gradient is '
            'applied in one optimizer/scheduler step.'
        ),
    )
    parser.add_argument(
        '--drop_incomplete_accumulation_group',
        action='store_true', default=False,
        help=(
            'Drop the final incomplete accumulation group so every optimizer '
            'step has the requested effective batch size.'
        ),
    )
    parser.add_argument(
        '--migrate_scheduler_for_gradient_accumulation',
        action='store_true', default=False,
        help=(
            'On a full-state resume whose checkpoint used a different '
            'accumulation factor, normalize scheduler progress to completed '
            'epochs while preserving the checkpoint learning rates.'
        ),
    )
    parser.add_argument(
        '--resume_lr_scale', type=float, default=1.0,
        help=(
            'Explicit one-time multiplier for every optimizer parameter-group '
            'learning rate after an exact optimizer/scheduler checkpoint '
            'resume. Values must be in (0, 1]; 1 preserves legacy behavior.'
        ),
    )
    parser.add_argument(
        '--resume_lr_scale_expected_lineage', type=float, default=None,
        help=(
            'Fail-closed authorization for an additional resume LR decay. '
            'When the checkpoint already records a cumulative decay, this '
            'value is required and must equal that recorded lineage exactly.'
        ),
    )
    parser.add_argument(
        '--restore_e57_lr_to_initial', action='store_true', default=False,
        help=(
            'One-time, Nr3D-E57-only exact-resume repair that restores every '
            'optimizer current learning rate to its checkpoint initial_lr and '
            'updates only scheduler _last_lr. The model, optimizer moments, '
            'scheduler phase/base/milestones, and epoch remain unchanged.'
        ),
    )
    parser.add_argument(
        '--e57_lr_restore_claim', type=str, default='',
        help=(
            'Immutable one-shot claim created by the reviewed E57 launcher; '
            'required only with --restore_e57_lr_to_initial.'
        ),
    )
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_backbone", default=1e-4, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--syncbn', action='store_true')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--small_lr',  default=False,action='store_true')

    # io
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path')
    parser.add_argument(
        '--checkpoint_start_epoch', type=int, default=None,
        help=(
            'Explicit first training epoch after loading a checkpoint. '
            'Use this for initialization checkpoints whose epoch is not numeric.'
        ),
    )
    parser.add_argument(
        '--model_only_initialization', action='store_true', default=False,
        help=(
            'Load compatible model tensors from an initialization checkpoint '
            'without restoring optimizer/scheduler state. Requires an explicit '
            '--checkpoint_start_epoch and is invalid for evaluation.'
        ),
    )
    parser.add_argument(
        '--checkpoint_metric_retention', action='store_true', default=False,
        help=(
            'Atomically keep the latest checkpoint and the best checkpoints '
            'for REC@0.25, REC@0.50, mask@0.25, mask@0.50, and mask mIoU.'
        ),
    )
    parser.add_argument(
        '--checkpoint_retention_metrics',
        nargs='+',
        choices=CHECKPOINT_RETENTION_METRICS,
        default=list(CHECKPOINT_RETENTION_METRICS),
        help=(
            'Metric aliases retained by --checkpoint_metric_retention. '
            'Defaults to all five metrics; use rec_acc025 for REC-only runs.'
        ),
    )
    parser.add_argument('--log_dir', default='log',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--exp', default='exp',
                        help='exp name to save model checkpoint')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--max_train_batches', type=int, default=0,
                        help='0 runs full epochs; positive values run a bounded train-only audit')
    parser.add_argument('--save_freq', type=int, default=10)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=5)  # epoch-wise

    # others
    parser.add_argument("--local_rank", type=int,default=1,
                        help='local rank for DistributedDataParallel')  # note
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5],
                        nargs='+', help='A list of AP IoU thresholds')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument("--debug", action='store_true',
                        help="try to overfit few samples")
    parser.add_argument(
        "--debug_train_holdout", action='store_true', default=False,
        help=(
            "with --debug, train and evaluate on two deterministic, "
            "scene-disjoint 128-example subsets of the training split"
        ),
    )
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--pp_checkpoint', default=None)    # pointnet checkpoint
    parser.add_argument('--pp_checkpoint_sha256', type=str, default='')
    parser.add_argument('--reduce_lr', action='store_true')

    args, _ = parser.parse_known_args()
    args.source_moe_gate_uncertainty_weight_explicit = (
        args.source_moe_gate_uncertainty_weight is not None
    )
    if args.source_moe_gate_uncertainty_weight is None:
        args.source_moe_gate_uncertainty_weight = 0.0
    args.source_moe_gate_objective_explicit = (
        args.source_moe_gate_objective is not None
    )
    if args.source_moe_gate_objective is None:
        args.source_moe_gate_objective = 'balanced_focal'

    args.eval = args.eval or args.eval_train

    return args


def prepare_source_moe_gate_checkpoint_config(args):
    """Bind source-arbiter continuation/evaluation to checkpoint config.

    Several SourceMoE inference fields (notably ``query_max_delta``) are not
    tensors in the state dict.  They must be inherited before model creation
    when continuing a trained SourceMoE; otherwise a gate/module continuation
    or standalone evaluation can silently use a different candidate ranking.
    A first SourceMoE-only initialization from a plain MCLN checkpoint keeps
    the explicitly requested runtime contract.  Joint-query-only training may
    also extend a trained source-choice selector; in that case its exact source
    schema is inherited without constructing SourceMoE.
    """
    if not hasattr(args, "source_moe_gate_uncertainty_weight"):
        args.source_moe_gate_uncertainty_weight = 0.0
    gate_train_only = getattr(args, "source_moe_gate_train_only", False)
    moe_train_only = getattr(args, "source_moe_train_only", False)
    query_mask_fusion_train_only = getattr(
        args, "query_mask_fusion_train_only", False
    )
    egqs_mask_refiner_train_only = getattr(
        args, "egqs_mask_refiner_train_only", False
    )
    joint_query_quality_train_only = getattr(
        args, "joint_query_quality_train_only", False
    )
    sacr_score_refiner_train_only = getattr(
        args, "sacr_score_refiner_train_only", False
    )
    parent_relative_text_verifier_train_only = getattr(
        args, "parent_relative_text_verifier_train_only", False
    )
    parent_relative_text_verifier_eval = (
        getattr(args, "eval", False)
        and getattr(args, "use_parent_relative_text_verifier", False)
    )
    sacr_score_eval = (
        getattr(args, "eval", False)
        and getattr(args, "use_sacr_score_refiner", False)
    )
    source_moe_eval = (
        getattr(args, "eval", False)
        and getattr(args, "use_source_moe", False)
    )
    requested_action_mode = getattr(
        args, "source_moe_gate_action_mode", None
    )
    requested_gate_objective = getattr(
        args, "source_moe_gate_objective", "balanced_focal"
    )
    objective_explicit = bool(getattr(
        args, "source_moe_gate_objective_explicit", False
    ))
    uncertainty_weight_explicit = bool(getattr(
        args, "source_moe_gate_uncertainty_weight_explicit", False
    ))
    if requested_action_mode not in (
            None, "decision", "expected_utility", "direct_utility",
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
        raise ValueError("invalid SourceMoE gate action mode")
    if requested_gate_objective not in (
            "balanced_focal", "calibrated_utility",
            "balanced_calibrated_utility",
            "hierarchical_risk_calibrated",
            "pairwise_risk_calibrated",
            "topn_risk_calibrated", "topn_dual_risk_calibrated",
            "topn_absolute_quality_calibrated",
            "cascade_absolute_quality_calibrated",
            "cascade_opportunity_balanced_calibrated",
            "cascade_opportunity_verified_calibrated",
            "cascade_joint_risk_calibrated",
            "cascade_v19_fallback_set_risk_calibrated",
            "cascade_v19_rich_set_empirical_risk",
            "cascade_v23_dense_quality_risk",
            "cascade_v24_relative_risk",
            "cascade_v25_pairwise_calibrated_risk",
            "cascade_v26_prior_restored_pairwise_risk",
            "cascade_v27_uncertainty_quality_risk",
            "cascade_v28_selected_abstention_risk",
            "cascade_v29_counterfactual_selected_risk",
            "cascade_v37_counterfactual_benefit_hazard_risk",
            "cascade_v38_complementary_logodds_risk",
            "cascade_v39_hazard_residual_risk"):
        raise ValueError("invalid SourceMoE gate objective")
    if (not gate_train_only and not moe_train_only
            and not query_mask_fusion_train_only
            and not egqs_mask_refiner_train_only
            and not joint_query_quality_train_only
            and not sacr_score_refiner_train_only
            and not parent_relative_text_verifier_train_only
            and not parent_relative_text_verifier_eval
            and not sacr_score_eval
            and not source_moe_eval):
        if requested_action_mode is None:
            args.source_moe_gate_action_mode = "decision"
        return args
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise ValueError(
            "SourceMoE gate training/evaluation requires an existing "
            "checkpoint"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise ValueError("candidate checkpoint has no model config")

    def config_value(name, default=None, required=False):
        if isinstance(checkpoint_config, dict):
            if name in checkpoint_config:
                return checkpoint_config[name]
        elif hasattr(checkpoint_config, name):
            return getattr(checkpoint_config, name)
        if required:
            raise ValueError(
                "candidate checkpoint config is missing {}".format(name)
            )
        return default

    checkpoint_has_source_moe = (
        config_value("use_source_moe", False) is True
    )
    checkpoint_has_source_selector = (
        config_value("use_source_choice_selector", False) is True
    )

    def require_exact_v99_selector_config():
        if getattr(args, "butd_cls", False) is not True:
            raise ValueError(
                "parent-relative verifier requires runtime butd_cls filtering"
            )
        if config_value("butd_cls", required=True) is not True:
            raise ValueError(
                "parent-relative verifier checkpoint did not use butd_cls"
            )
        if not checkpoint_has_source_selector or checkpoint_has_source_moe:
            raise ValueError(
                "parent-relative verifier requires the exact V99 "
                "source-choice parent"
            )
        for name, expected in _E57_V99_SELECTOR_CONFIG:
            actual = config_value(name, required=True)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    "parent-relative verifier V99 {} mismatch: {!r} != {!r}"
                    .format(name, actual, expected)
                )

    sacr_checkpoint_contract_required = (
        sacr_score_refiner_train_only or sacr_score_eval
    )
    checkpoint_state = checkpoint.get("model")
    if not isinstance(checkpoint_state, dict):
        if sacr_checkpoint_contract_required:
            raise ValueError("candidate checkpoint has no model state")
        checkpoint_state = {}

    def canonical_state_name(name):
        return name[7:] if name.startswith("module.") else name

    score_component_presence = {
        "structured_slot_builder": any(
            canonical_state_name(name).startswith(
                "structured_slot_builder."
            ) for name in checkpoint_state
        ),
        "sacr_head": any(
            canonical_state_name(name).startswith("sacr_head.")
            for name in checkpoint_state
        ),
        "sacr_score_gate": any(
            canonical_state_name(name) == "sacr_score_gate"
            for name in checkpoint_state
        ),
        "sacr_parent_relative_gate": any(
            canonical_state_name(name).startswith(
                "sacr_parent_relative_gate."
            )
            for name in checkpoint_state
        ),
    }
    verifier_component_presence = {
        "structured_slot_builder": any(
            canonical_state_name(name).startswith(
                "structured_slot_builder."
            ) for name in checkpoint_state
        ),
        "sacr_head": any(
            canonical_state_name(name).startswith("sacr_head.")
            for name in checkpoint_state
        ),
        "parent_relative_text_verifier": any(
            canonical_state_name(name).startswith(
                "parent_relative_text_verifier."
            ) for name in checkpoint_state
        ),
    }
    checkpoint_has_parent_relative_text_verifier = (
        config_value("use_parent_relative_text_verifier", False) is True
    )
    if parent_relative_text_verifier_eval:
        if not checkpoint_has_parent_relative_text_verifier or not all(
                verifier_component_presence.values()):
            raise ValueError(
                "parent-relative verifier evaluation requires a complete "
                "trained verifier checkpoint: config={}, state={}".format(
                    checkpoint_has_parent_relative_text_verifier,
                    verifier_component_presence,
                )
            )
        require_exact_v99_selector_config()
        if getattr(args, "use_source_moe", False):
            raise ValueError(
                "parent-relative verifier evaluation rejects SourceMoE"
            )
        args.use_source_choice_selector = True
        for key, unused_expected in _E57_V99_SELECTOR_CONFIG:
            setattr(args, key, config_value(key, required=True))
        for key in (
                "sacr_hidden_dim",
                "sacr_max_pairs",
                "sacr_top_m_targets",
                "sacr_top_k_anchors",
                "sacr_geo_dim",
                "sacr_min_parse_confidence",
                "parent_relative_text_verifier_top_k",
                "parent_relative_text_verifier_max_candidates",
                "parent_relative_text_verifier_hidden_dim",
                "parent_relative_text_verifier_heads",
                "parent_relative_text_verifier_dropout",
                "parent_relative_text_verifier_max_parent_score_gap",
                "parent_relative_text_verifier_promotion_margin",
                "parent_relative_text_verifier_min_parse_confidence",
                "parent_relative_text_verifier_min_anchor_mass",
                "parent_relative_text_verifier_promotion_epsilon",
                "parent_relative_text_verifier_detach_inputs"):
            setattr(args, key, config_value(key, required=True))
        if requested_action_mode is None:
            args.source_moe_gate_action_mode = "decision"
        del checkpoint
        return args
    checkpoint_has_legacy_score_gate = score_component_presence[
        "sacr_score_gate"
    ]
    checkpoint_has_parent_relative_gate = score_component_presence[
        "sacr_parent_relative_gate"
    ]
    checkpoint_has_score_refiner = (
        checkpoint_has_legacy_score_gate
        or checkpoint_has_parent_relative_gate
    )
    if (
            sacr_checkpoint_contract_required
            and checkpoint_has_legacy_score_gate
            and checkpoint_has_parent_relative_gate):
        raise ValueError(
            "candidate checkpoint mixes legacy and parent-relative SACR gates"
        )
    if (
            sacr_checkpoint_contract_required
            and checkpoint_has_score_refiner
            and not (
                score_component_presence["structured_slot_builder"]
                and score_component_presence["sacr_head"]
            )):
        raise ValueError(
            "candidate checkpoint has a partial SACR score refiner state: {}"
            .format(score_component_presence)
        )
    checkpoint_config_has_score_refiner = (
        config_value("use_sacr_score_refiner", False) is True
    )
    checkpoint_uses_parent_relative_sacr = (
        config_value(
            "sacr_score_use_parent_relative_abstention", False
        ) is True
    )
    checkpoint_uses_relation_counterfactual_sacr = (
        config_value(
            "sacr_score_use_relation_counterfactual", False
        ) is True
    )
    if (
            sacr_checkpoint_contract_required
            and checkpoint_uses_parent_relative_sacr
            and checkpoint_uses_relation_counterfactual_sacr):
        raise ValueError(
            "candidate checkpoint mixes SACR deployment variants"
        )
    if (
            sacr_checkpoint_contract_required
            and checkpoint_uses_parent_relative_sacr
            != checkpoint_has_parent_relative_gate):
        raise ValueError(
            "candidate checkpoint parent-relative SACR config/state disagree"
        )
    if (
            sacr_checkpoint_contract_required
            and checkpoint_has_legacy_score_gate
            == checkpoint_uses_parent_relative_sacr
            and checkpoint_has_score_refiner):
        raise ValueError(
            "candidate checkpoint SACR gate type disagrees with config"
        )
    requested_parent_relative_sacr = bool(getattr(
        args, "sacr_score_use_parent_relative_abstention", False
    ))
    requested_relation_counterfactual_sacr = bool(getattr(
        args, "sacr_score_use_relation_counterfactual", False
    ))
    if (
            sacr_checkpoint_contract_required
            and checkpoint_config_has_score_refiner
            != checkpoint_has_score_refiner):
        raise ValueError(
            "candidate checkpoint SACR score config/state disagree"
        )

    if sacr_score_refiner_train_only or sacr_score_eval:
        if checkpoint_has_score_refiner:
            if not checkpoint_has_source_selector:
                raise ValueError(
                    "trained V133 checkpoint must retain its selector parent"
                )
            if getattr(args, "use_source_moe", False):
                raise ValueError(
                    "selector-backed V133 continuation cannot enable SourceMoE"
                )
            args.use_source_choice_selector = True
            args.source_choice_selector_sources = config_value(
                "source_choice_selector_sources", required=True
            )
            args.source_choice_selector_hidden_dim = config_value(
                "source_choice_selector_hidden_dim", required=True
            )
            score_runtime_keys = (
                "sacr_hidden_dim",
                "sacr_max_pairs",
                "sacr_top_m_targets",
                "sacr_top_k_anchors",
                "sacr_geo_dim",
                "sacr_min_parse_confidence",
                "sacr_score_max_delta",
            )
            for key in score_runtime_keys:
                setattr(args, key, config_value(key, required=True))
            if checkpoint_uses_parent_relative_sacr:
                args.sacr_score_use_parent_relative_abstention = True
                for key in (
                        "sacr_score_parent_gate_hidden_dim",
                        "sacr_score_min_box_advantage",
                        "sacr_score_promotion_margin",
                        "sacr_score_mask_tolerance",
                        "sacr_score_raw_margin",
                        "sacr_score_dense_weight",
                        "sacr_score_preserve_weight",
                        "sacr_score_gate_weight",
                        "sacr_score_saturation_weight"):
                    setattr(args, key, config_value(key, required=True))
            elif checkpoint_uses_relation_counterfactual_sacr:
                args.sacr_score_use_relation_counterfactual = True
                for key in (
                        "sacr_score_promotion_margin",
                        "sacr_score_mask_tolerance",
                        "sacr_counterfactual_parent_top_k",
                        "sacr_counterfactual_target_tolerance",
                        "sacr_counterfactual_attribute_tolerance",
                        "sacr_counterfactual_geometry_threshold",
                        "sacr_counterfactual_iou_gap",
                        "sacr_counterfactual_correct_iou_threshold",
                        "sacr_counterfactual_pair_margin",
                        "sacr_counterfactual_max_negatives",
                        "sacr_counterfactual_relation_scale",
                        "sacr_counterfactual_deployment_threshold"):
                    setattr(args, key, config_value(key, required=True))
            elif sacr_score_eval and requested_parent_relative_sacr:
                raise ValueError(
                    "parent-relative SACR evaluation requires a checkpoint "
                    "trained with that deployment contract"
                )
            elif sacr_score_eval and requested_relation_counterfactual_sacr:
                raise ValueError(
                    "relation-counterfactual SACR evaluation requires a "
                    "checkpoint trained with that deployment contract"
                )
            if sacr_score_refiner_train_only:
                for key in (
                        "sacr_score_refiner_lr",
                        "sacr_score_refiner_loss_weight",
                        "sacr_score_temperature",
                        "sacr_score_mask_weight"):
                    setattr(args, key, config_value(key, required=True))
            args._sacr_score_checkpoint_has_refiner = True
            if requested_action_mode is None:
                args.source_moe_gate_action_mode = "decision"
            del checkpoint
            return args
        if sacr_score_eval and not sacr_score_refiner_train_only:
            raise ValueError(
                "SACR score evaluation requires a trained refiner checkpoint"
            )
        args._sacr_score_checkpoint_has_refiner = False
    if not checkpoint_has_source_moe and moe_train_only:
        if requested_action_mode is None:
            args.source_moe_gate_action_mode = "decision"
        del checkpoint
        return args
    if not checkpoint_has_source_moe and egqs_mask_refiner_train_only:
        if getattr(args, "use_source_moe", False):
            raise ValueError(
                "plain/selector EGQS initialization cannot add SourceMoE"
            )
        if checkpoint_has_source_selector:
            args.use_source_choice_selector = True
            args.source_choice_selector_sources = config_value(
                "source_choice_selector_sources", required=True
            )
            args.source_choice_selector_hidden_dim = config_value(
                "source_choice_selector_hidden_dim", required=True
            )
        if requested_action_mode is None:
            args.source_moe_gate_action_mode = "decision"
        del checkpoint
        return args
    if (not checkpoint_has_source_moe
            and (
                joint_query_quality_train_only
                or sacr_score_refiner_train_only
                or parent_relative_text_verifier_train_only
            )
            and checkpoint_has_source_selector):
        if getattr(args, "use_source_moe", False):
            raise ValueError(
                "selector-backed score training cannot enable SourceMoE"
            )
        args.use_source_choice_selector = True
        if parent_relative_text_verifier_train_only:
            require_exact_v99_selector_config()
            for key, unused_expected in _E57_V99_SELECTOR_CONFIG:
                setattr(args, key, config_value(key, required=True))
        else:
            args.source_choice_selector_sources = config_value(
                "source_choice_selector_sources", required=True
            )
            args.source_choice_selector_hidden_dim = config_value(
                "source_choice_selector_hidden_dim", required=True
            )
        if (
                joint_query_quality_train_only
                and getattr(args, "joint_query_quality_use_gate_evidence", False)):
            raise ValueError(
                "joint query gate evidence requires a SourceMoE checkpoint"
            )
        if requested_action_mode is None:
            args.source_moe_gate_action_mode = "decision"
        del checkpoint
        return args
    if not checkpoint_has_source_moe:
        raise ValueError(
            "checkpoint must contain a trained SourceMoE or, for joint-query-"
            "only/SACR-score-only training, a trained source-choice selector"
        )
    candidate_keys = (
        "source_choice_selector_sources",
        "source_choice_selector_hidden_dim",
        "source_moe_shared_source",
        "source_moe_top_k",
        "source_moe_query_layers",
        "source_moe_query_heads",
        "source_moe_query_dropout",
        "source_moe_query_max_delta",
    )
    for key in candidate_keys:
        setattr(args, key, config_value(key, required=True))

    runtime_requires_gate = bool(getattr(
        args, "source_moe_use_fallback_gate", False
    ))
    checkpoint_has_gate = config_value(
        "source_moe_use_fallback_gate", False
    )
    if moe_train_only and runtime_requires_gate and not checkpoint_has_gate:
        raise ValueError(
            "SourceMoE continuation requires a checkpoint with a trained "
            "fallback gate"
        )
    if checkpoint_has_gate:
        gate_keys = (
            "source_moe_gate_hidden_dim",
            "source_moe_gate_candidate_top_k",
            "source_moe_gate_break_cost",
            "source_moe_gate_decision_margin",
            "source_moe_gate_mask_utility_weight",
        )
        for key in gate_keys:
            setattr(args, key, config_value(key, required=True))
        if not uncertainty_weight_explicit:
            setattr(
                args,
                "source_moe_gate_uncertainty_weight",
                config_value("source_moe_gate_uncertainty_weight", 0.0),
            )
        evidence_features = config_value(
            "source_moe_gate_use_evidence_features", False
        )
        if not isinstance(evidence_features, bool):
            raise ValueError(
                "candidate checkpoint has an invalid gate evidence flag"
            )
        setattr(
            args,
            "source_moe_gate_use_evidence_features",
            evidence_features,
        )
        gate_objective = config_value(
            "source_moe_gate_objective", "balanced_focal"
        )
        if gate_objective not in (
                "balanced_focal", "calibrated_utility",
                "balanced_calibrated_utility",
                "hierarchical_risk_calibrated",
                "pairwise_risk_calibrated",
                "topn_risk_calibrated", "topn_dual_risk_calibrated",
                "topn_absolute_quality_calibrated",
                "cascade_absolute_quality_calibrated",
                "cascade_opportunity_balanced_calibrated",
                "cascade_opportunity_verified_calibrated",
                "cascade_joint_risk_calibrated",
                "cascade_v19_fallback_set_risk_calibrated",
                "cascade_v19_rich_set_empirical_risk",
                "cascade_v23_dense_quality_risk",
                "cascade_v24_relative_risk",
                "cascade_v25_pairwise_calibrated_risk",
                "cascade_v26_prior_restored_pairwise_risk",
                "cascade_v27_uncertainty_quality_risk",
                "cascade_v28_selected_abstention_risk",
                "cascade_v29_counterfactual_selected_risk",
                "cascade_v37_counterfactual_benefit_hazard_risk",
                "cascade_v38_complementary_logodds_risk",
                "cascade_v39_hazard_residual_risk"):
            raise ValueError(
                "candidate checkpoint has an invalid gate objective"
            )
        if not objective_explicit:
            setattr(args, "source_moe_gate_objective", gate_objective)
        for key, default in (
                ("source_moe_gate_context_layers", 0),
                ("source_moe_gate_context_heads", 4),
                ("source_moe_gate_context_dropout", 0.1),
                ("source_moe_gate_setwise_temperature", 0.0)):
            setattr(args, key, config_value(key, default))
        checkpoint_action_mode = config_value(
            "source_moe_gate_action_mode", "decision"
        )
        if checkpoint_action_mode not in (
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
            raise ValueError(
                "candidate checkpoint has an invalid gate action mode"
            )
        if requested_action_mode is None:
            args.source_moe_gate_action_mode = checkpoint_action_mode
    elif source_moe_eval:
        args.source_moe_gate_use_evidence_features = False
        args.source_moe_gate_context_layers = 0
        if requested_action_mode is None:
            args.source_moe_gate_action_mode = "decision"
    elif requested_action_mode is None:
        args.source_moe_gate_action_mode = "decision"
    args.use_source_moe = True
    args.source_moe_use_fallback_gate = (
        True if gate_train_only else bool(checkpoint_has_gate)
    )
    del checkpoint
    return args

def validate_source_moe_gate_checkpoint_contract(model, checkpoint):
    """Reject silently incompatible trained fallback-gate checkpoints."""
    checkpoint_state = checkpoint.get("model")
    if not isinstance(checkpoint_state, dict):
        raise ValueError("checkpoint model state is invalid")

    def canonical_gate_state(state):
        result = {}
        for key, value in state.items():
            canonical = key[7:] if key.startswith("module.") else key
            if (
                    "source_moe.fallback_gate." in canonical
                    or "source_moe.adaptive_source_mixer." in canonical):
                result[canonical] = value
        return result

    checkpoint_gate = canonical_gate_state(checkpoint_state)
    if not checkpoint_gate:
        return
    current_gate = canonical_gate_state(model.state_dict())
    if not current_gate:
        raise ValueError(
            "checkpoint contains a trained SourceMoE fallback gate but the "
            "current model does not"
        )

    config = checkpoint.get("config")
    if isinstance(config, dict):
        checkpoint_evidence = config.get(
            "source_moe_gate_use_evidence_features", False
        )
    else:
        checkpoint_evidence = getattr(
            config, "source_moe_gate_use_evidence_features", False
        )
    if not isinstance(checkpoint_evidence, bool):
        raise ValueError("checkpoint gate evidence flag is invalid")
    unwrapped = model.module if hasattr(model, "module") else model
    source_moe = getattr(unwrapped, "source_moe", None)
    current_evidence = bool(getattr(
        source_moe, "gate_use_evidence_features", False
    ))
    if checkpoint_evidence is not current_evidence:
        raise ValueError(
            "checkpoint and current SourceMoE gate evidence contracts differ"
        )

    fallback_gate = getattr(source_moe, "fallback_gate", None)
    checkpoint_context = (
        config.get("source_moe_gate_context_layers", 0),
        config.get("source_moe_gate_context_heads", 4),
        config.get("source_moe_gate_context_dropout", 0.1),
    ) if isinstance(config, dict) else (
        getattr(config, "source_moe_gate_context_layers", 0),
        getattr(config, "source_moe_gate_context_heads", 4),
        getattr(config, "source_moe_gate_context_dropout", 0.1),
    )
    current_context = (
        getattr(fallback_gate, "context_layers", 0),
        getattr(fallback_gate, "context_heads", 4),
        getattr(fallback_gate, "context_dropout", 0.1),
    )
    if checkpoint_context != current_context:
        raise ValueError(
            "checkpoint and current SourceMoE gate context contracts differ"
        )

    checkpoint_action_mode = (
        config.get("source_moe_gate_action_mode", "decision")
        if isinstance(config, dict)
        else getattr(config, "source_moe_gate_action_mode", "decision")
    )
    if checkpoint_action_mode not in (
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
        raise ValueError("checkpoint gate action mode is invalid")
    migratable_missing = set()
    if checkpoint_action_mode not in (
            "direct_utility", "hierarchical_utility",
            "pairwise_verifier", "topn_pairwise_verifier",
            "topn_dual_evidence_verifier",
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
        migratable_missing.update({
            key for key in current_gate
            if key.endswith("fallback_gate.utility_head.weight")
            or key.endswith("fallback_gate.utility_head.bias")
        })
    if checkpoint_action_mode != "hierarchical_utility":
        migratable_missing.update({
            key for key in current_gate
            if "fallback_gate.row_switch_head." in key
        })
    if checkpoint_action_mode not in (
            "pairwise_verifier", "topn_pairwise_verifier",
            "topn_dual_evidence_verifier",
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
        migratable_missing.update({
            key for key in current_gate
            if "fallback_gate.pairwise_switch_head." in key
        })
    if checkpoint_action_mode != "topn_dual_evidence_verifier":
        migratable_missing.update({
            key for key in current_gate
            if "fallback_gate.safety_switch_head." in key
        })
    if checkpoint_action_mode not in (
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
        migratable_missing.update({
            key for key in current_gate
            if "fallback_gate.absolute_quality_head." in key
        })
    if checkpoint_action_mode not in (
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
        migratable_missing.update({
            key for key in current_gate
            if "fallback_gate.cascade_quality_adapter." in key
            or "fallback_gate.cascade_correction_head." in key
        })
    if checkpoint_action_mode not in (
            "cascade_opportunity_quality_correction",
            "cascade_opportunity_verified_correction",
            "cascade_joint_risk_correction"):
        migratable_missing.update({
            key for key in current_gate
            if "fallback_gate.cascade_opportunity_head." in key
        })
    if checkpoint_action_mode not in (
            "cascade_opportunity_verified_correction",
            "cascade_joint_risk_correction"):
        migratable_missing.update({
            key for key in current_gate
            if "fallback_gate.cascade_candidate_safety_head." in key
        })
    if checkpoint_action_mode != "cascade_joint_risk_correction":
        migratable_missing.update({
            key for key in current_gate
            if "fallback_gate.cascade_joint_action_head." in key
        })

    current_action_mode = getattr(fallback_gate, "action_mode", "decision")
    if current_action_mode == (
            "cascade_v28_selected_abstention_correction"):
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v28_selected_abstention_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V28 selected-abstention gate requires a complete V19 "
                "initializer or an exact V28 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if (
                    "fallback_gate.cascade_dense_quality_set_head." in key
                    or "fallback_gate.cascade_pairwise_calibrated_set_head."
                    in key
                    or "fallback_gate.cascade_selected_abstention_head."
                    in key
                    or "source_moe.adaptive_source_mixer." in key
                )
            }
        else:
            migratable_missing = set()
    elif current_action_mode == (
            "cascade_v29_counterfactual_selected_correction"):
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v29_counterfactual_selected_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V29 counterfactual risk gate requires a complete V19 "
                "initializer or an exact V29 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if (
                    "fallback_gate.cascade_dense_quality_set_head." in key
                    or "fallback_gate.cascade_pairwise_calibrated_set_head."
                    in key
                    or "fallback_gate.cascade_counterfactual_selected_risk_head."
                    in key
                    or "source_moe.adaptive_source_mixer." in key
                )
            }
        else:
            migratable_missing = set()
    elif current_action_mode == (
            "cascade_v37_counterfactual_benefit_hazard_correction"):
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v37_counterfactual_benefit_hazard_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V37 benefit-hazard gate requires a complete V19 "
                "initializer or an exact V37 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if (
                    "fallback_gate.cascade_dense_quality_set_head." in key
                    or "fallback_gate.cascade_pairwise_calibrated_set_head."
                    in key
                    or "fallback_gate.cascade_counterfactual_benefit_hazard_head."
                    in key
                    or "source_moe.adaptive_source_mixer." in key
                )
            }
        else:
            migratable_missing = set()
    elif current_action_mode == (
            "cascade_v38_complementary_logodds_correction"):
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v38_complementary_logodds_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V38 complementary log-odds gate requires a complete V19 "
                "initializer or an exact V38 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if (
                    "fallback_gate.cascade_dense_quality_set_head." in key
                    or "fallback_gate.cascade_pairwise_calibrated_set_head."
                    in key
                    or "fallback_gate.cascade_counterfactual_logodds_head."
                    in key
                    or "source_moe.adaptive_source_mixer." in key
                )
            }
        else:
            migratable_missing = set()
    elif current_action_mode == (
            "cascade_v39_hazard_residual_correction"):
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v39_hazard_residual_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V39 hazard-residual gate requires a complete V19 "
                "initializer or an exact V39 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if (
                    "fallback_gate.cascade_dense_quality_set_head." in key
                    or "fallback_gate.cascade_pairwise_calibrated_set_head."
                    in key
                    or "fallback_gate.cascade_counterfactual_hazard_residual_head."
                    in key
                    or "source_moe.adaptive_source_mixer." in key
                )
            }
        else:
            migratable_missing = set()
    elif current_action_mode == (
            "cascade_v26_prior_restored_pairwise_correction"):
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v26_prior_restored_pairwise_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V26 prior-restored pairwise gate requires a complete V19 "
                "initializer or an exact V26 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if (
                    "fallback_gate.cascade_dense_quality_set_head." in key
                    or "fallback_gate.cascade_pairwise_calibrated_set_head."
                    in key
                    or "source_moe.adaptive_source_mixer." in key
                )
            }
        else:
            migratable_missing = set()
    elif current_action_mode == "cascade_v23_dense_quality_correction":
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v23_dense_quality_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V23 dense-quality gate requires a complete V19 initializer "
                "or an exact V23 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if (
                    "fallback_gate.cascade_dense_quality_set_head." in key
                    or "source_moe.adaptive_source_mixer." in key
                )
            }
        else:
            migratable_missing = set()
    elif current_action_mode == (
            "cascade_v25_pairwise_calibrated_correction"):
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v25_pairwise_calibrated_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V25 pairwise calibrated gate requires a complete V19 "
                "initializer or an exact V25 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if (
                    "fallback_gate.cascade_dense_quality_set_head." in key
                    or "fallback_gate.cascade_pairwise_calibrated_set_head."
                    in key
                    or "source_moe.adaptive_source_mixer." in key
                )
            }
        else:
            migratable_missing = set()
    elif current_action_mode == "cascade_v24_relative_risk_correction":
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v24_relative_risk_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V24 relative-risk gate requires a complete V19 initializer "
                "or an exact V24 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if (
                    "fallback_gate.cascade_dense_quality_set_head." in key
                    or "fallback_gate.cascade_relative_risk_set_head." in key
                    or "source_moe.adaptive_source_mixer." in key
                )
            }
        else:
            migratable_missing = set()
    elif current_action_mode == "cascade_v19_rich_set_correction":
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v19_rich_set_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V22 rich fallback-set gate requires a complete V19 "
                "initializer or an exact V22 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if "fallback_gate.cascade_rich_fallback_set_action_head."
                in key
            }
        else:
            migratable_missing = set()
    elif current_action_mode == "cascade_v19_fallback_set_correction":
        allowed_initializers = (
            "cascade_opportunity_verified_correction",
            "cascade_v19_fallback_set_correction",
        )
        if checkpoint_action_mode not in allowed_initializers:
            raise ValueError(
                "V21 fallback-set gate requires a complete V19 initializer "
                "or an exact V21 checkpoint"
            )
        if checkpoint_action_mode == "cascade_opportunity_verified_correction":
            migratable_missing = {
                key for key in current_gate
                if "fallback_gate.cascade_fallback_set_action_head." in key
            }
        else:
            migratable_missing = set()
    elif current_action_mode in (
            "cascade_absolute_quality_correction",
            "cascade_opportunity_quality_correction",
            "cascade_opportunity_verified_correction",
            "cascade_joint_risk_correction"):
        # Cascade stage one is the trained V12 pairwise verifier.  These two
        # legacy heads are therefore required even when the source checkpoint
        # predates the cascade-specific modules.
        migratable_missing = {
            key for key in migratable_missing
            if "fallback_gate.utility_head." not in key
            and "fallback_gate.pairwise_switch_head." not in key
        }
    missing = sorted(
        set(current_gate) - set(checkpoint_gate) - migratable_missing
    )
    unexpected = sorted(set(checkpoint_gate) - set(current_gate))
    mismatched = sorted(
        key for key in set(current_gate) & set(checkpoint_gate)
        if getattr(current_gate[key], "shape", None)
        != getattr(checkpoint_gate[key], "shape", None)
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append("missing={}".format(",".join(missing)))
        if unexpected:
            details.append("unexpected={}".format(",".join(unexpected)))
        if mismatched:
            details.append("shape={}".format(",".join(mismatched)))
        raise ValueError(
            "checkpoint fallback-gate state is incompatible: {}".format(
                "; ".join(details)
            )
        )


_SOURCE_MOE_RESUME_CONFIG_DEFAULTS = (
    ("use_source_moe", False),
    ("source_choice_selector_sources", "default,mask_text"),
    ("source_choice_selector_hidden_dim", 288),
    ("source_moe_shared_source", "default"),
    ("source_moe_top_k", 2),
    ("source_moe_balance_loss_weight", 0.01),
    ("source_moe_rank_loss_weight", 1.0),
    ("source_moe_mask_rank_loss_weight", 0.25),
    ("source_moe_rank_temperature", 0.1),
    ("source_moe_anchor_loss_weight", 0.0),
    ("source_moe_anchor_margin", 0.05),
    ("source_moe_query_layers", 1),
    ("source_moe_query_heads", 4),
    ("source_moe_query_dropout", 0.1),
    ("source_moe_query_max_delta", 0.25),
    ("source_moe_use_fallback_gate", False),
    ("source_moe_gate_hidden_dim", 128),
    ("source_moe_gate_candidate_top_k", 8),
    ("source_moe_gate_break_cost", 2.0),
    ("source_moe_gate_decision_margin", 0.0),
    ("source_moe_gate_mask_utility_weight", 0.25),
    ("source_moe_gate_uncertainty_weight", 0.0),
    ("source_moe_gate_use_evidence_features", False),
    ("source_moe_gate_context_layers", 0),
    ("source_moe_gate_context_heads", 4),
    ("source_moe_gate_context_dropout", 0.1),
    ("source_moe_gate_action_mode", "decision"),
    ("source_moe_gate_new_heads_only", False),
    ("source_moe_gate_loss_weight", 0.0),
    ("source_moe_gate_mask_loss_weight", 0.25),
    ("source_moe_gate_focal_gamma", 2.0),
    ("source_moe_gate_false_override_weight", 2.0),
    ("source_moe_gate_objective", "balanced_focal"),
    ("source_moe_gate_setwise_temperature", 0.0),
    ("source_moe_gate_boundary_loss_weight", 0.0),
    ("source_moe_gate_lr", 0.0003),
)


def validate_source_moe_resume_checkpoint_contract(args, checkpoint):
    """Require an exact SourceMoE contract when optimizer state is resumed."""
    gate_optimizer_resume = bool(getattr(
        args, "source_moe_gate_resume_optimizer", False
    ))
    if gate_optimizer_resume and (
            not getattr(args, "source_moe_gate_train_only", False)
            or getattr(args, "eval", False)
            or getattr(args, "reduce_lr", False)):
        raise ValueError(
            "source_moe_gate_resume_optimizer requires non-eval gate-only "
            "training without reduce_lr"
        )
    restores_optimizer = (
        gate_optimizer_resume
        or (
            not getattr(args, "eval", False)
            and not getattr(args, "reduce_lr", False)
            and not getattr(
                args, "source_choice_selector_train_only", False
            )
            and not getattr(args, "source_moe_train_only", False)
            and not getattr(args, "source_moe_gate_train_only", False)
            and not getattr(args, "query_mask_fusion_train_only", False)
            and not getattr(args, "egqs_mask_refiner_train_only", False)
            and not getattr(args, "joint_query_quality_train_only", False)
            and not getattr(
                args, "parent_relative_text_verifier_train_only", False
            )
        )
    )
    if not restores_optimizer or not getattr(args, "use_source_moe", False):
        return

    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise ValueError(
            "SourceMoE optimizer resume requires checkpoint config"
        )

    def checkpoint_value(name, default):
        if isinstance(checkpoint_config, dict):
            return checkpoint_config.get(name, default)
        return getattr(checkpoint_config, name, default)

    def canonical_value(name, value):
        if name == "source_choice_selector_sources":
            if isinstance(value, str):
                value = value.split(",")
            if isinstance(value, (tuple, list)):
                return tuple(
                    str(item).strip() for item in value
                    if str(item).strip()
                )
        return value

    mismatches = []
    for name, default in _SOURCE_MOE_RESUME_CONFIG_DEFAULTS:
        current = canonical_value(name, getattr(args, name, default))
        saved = canonical_value(name, checkpoint_value(name, default))
        if current != saved:
            mismatches.append(
                "{} (checkpoint={!r}, runtime={!r})".format(
                    name, saved, current
                )
            )
    if mismatches:
        raise ValueError(
            "SourceMoE optimizer resume config differs from checkpoint: {}. "
            "A true resume must reuse the checkpoint values; use --reduce_lr "
            "for an intentional fresh optimizer.".format(
                "; ".join(mismatches)
            )
        )


_QUERY_MASK_FUSION_RESUME_CONFIG_DEFAULTS = (
    ("use_query_mask_fusion_calibrator", False),
    ("query_mask_fusion_train_only", False),
    ("query_mask_fusion_lr", 0.001),
    ("query_mask_fusion_hidden_dim", 128),
    ("query_mask_fusion_dropout", 0.0),
    ("query_mask_fusion_max_delta", 0.25),
)


def validate_query_mask_fusion_resume_checkpoint_contract(args, checkpoint):
    """Require an exact query-mask optimizer/scheduler resume contract."""
    resume_optimizer = bool(getattr(
        args, "query_mask_fusion_resume_optimizer", False
    ))
    if not resume_optimizer:
        return
    if (
            not getattr(args, "use_query_mask_fusion_calibrator", False)
            or not getattr(args, "query_mask_fusion_train_only", False)
            or getattr(args, "eval", False)
            or getattr(args, "reduce_lr", False)):
        raise ValueError(
            "query_mask_fusion_resume_optimizer requires non-eval query-mask "
            "fusion-only training without reduce_lr"
        )
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise ValueError(
            "query-mask optimizer resume requires checkpoint config"
        )

    def checkpoint_value(name, default):
        if isinstance(checkpoint_config, dict):
            return checkpoint_config.get(name, default)
        return getattr(checkpoint_config, name, default)

    mismatches = []
    for name, default in _QUERY_MASK_FUSION_RESUME_CONFIG_DEFAULTS:
        current = getattr(args, name, default)
        saved = checkpoint_value(name, default)
        if current != saved:
            mismatches.append(
                "{} (checkpoint={!r}, runtime={!r})".format(
                    name, saved, current
                )
            )
    if mismatches:
        raise ValueError(
            "query-mask optimizer resume config differs from checkpoint: {}"
            .format("; ".join(mismatches))
        )
    for state_name in ("optimizer", "scheduler"):
        if state_name not in checkpoint:
            raise ValueError(
                "query-mask optimizer resume checkpoint is missing {} state"
                .format(state_name)
            )


# BRIEF load checkpoint.
def _validated_resume_lr_scale(args, checkpoint):
    scale = getattr(args, "resume_lr_scale", 1.0)
    if (
            isinstance(scale, bool)
            or not isinstance(scale, numbers.Real)
            or not math.isfinite(float(scale))
            or not 0.0 < float(scale) <= 1.0):
        raise ValueError("resume_lr_scale must be finite and in (0, 1]")
    scale = float(scale)

    checkpoint_config = checkpoint.get("config", {})

    def checkpoint_value(name, default):
        if isinstance(checkpoint_config, dict):
            return checkpoint_config.get(name, default)
        return getattr(checkpoint_config, name, default)

    prior_lineage = checkpoint_value(
        "resume_lr_scale_lineage",
        checkpoint_value("resume_lr_scale", 1.0),
    )
    if (
            isinstance(prior_lineage, bool)
            or not isinstance(prior_lineage, numbers.Real)
            or not math.isfinite(float(prior_lineage))
            or not 0.0 < float(prior_lineage) <= 1.0):
        raise ValueError(
            "checkpoint resume_lr_scale lineage must be finite and in (0, 1]"
        )
    prior_lineage = float(prior_lineage)
    expected_lineage = getattr(
        args, "resume_lr_scale_expected_lineage", None
    )
    if expected_lineage is not None:
        if (
                isinstance(expected_lineage, bool)
                or not isinstance(expected_lineage, numbers.Real)
                or not math.isfinite(float(expected_lineage))
                or not 0.0 < float(expected_lineage) <= 1.0):
            raise ValueError(
                "resume_lr_scale_expected_lineage must be finite and in "
                "(0, 1]"
            )
        expected_lineage = float(expected_lineage)
    # Persist the cumulative lineage even through a scale=1 continuation. The
    # next checkpoint serializes args, so a later resume cannot silently apply
    # the same decay for a second time.
    args.resume_lr_scale_lineage = prior_lineage
    if scale == 1.0:
        if expected_lineage is not None:
            raise ValueError(
                "resume_lr_scale_expected_lineage is only valid when "
                "resume_lr_scale is less than 1"
            )
        return scale

    incompatible_flags = [
        name for name in (
            "eval",
            "reduce_lr",
            "model_only_initialization",
            "frozen",
            "small_lr",
            "source_choice_selector_train_only",
            "source_moe_train_only",
            "source_moe_gate_train_only",
            "query_mask_fusion_train_only",
            "egqs_mask_refiner_train_only",
            "joint_query_quality_train_only",
            "decoder_query_adapter_train_only",
            "sacr_score_refiner_train_only",
            "parent_relative_text_verifier_train_only",
            "migrate_scheduler_for_gradient_accumulation",
        )
        if bool(getattr(args, name, False))
    ]
    if getattr(args, "checkpoint_start_epoch", None) is not None:
        incompatible_flags.append("checkpoint_start_epoch")
    if getattr(args, "lr_scheduler", "step") != "step":
        incompatible_flags.append("lr_scheduler")
    if getattr(args, "warmup_epoch", -1) > 0:
        incompatible_flags.append("warmup_epoch")
    if incompatible_flags:
        raise ValueError(
            "resume_lr_scale requires an otherwise exact plain-step "
            "full-state resume; incompatible settings: {}".format(
                ", ".join(incompatible_flags)
            )
        )
    missing = object()
    checkpoint_scheduler = checkpoint_value("lr_scheduler", missing)
    checkpoint_warmup = checkpoint_value("warmup_epoch", missing)
    if checkpoint_scheduler is missing or checkpoint_warmup is missing:
        raise ValueError(
            "resume_lr_scale requires checkpoint lr_scheduler and "
            "warmup_epoch provenance"
        )
    runtime_scheduler = getattr(args, "lr_scheduler", "step")
    runtime_warmup = getattr(args, "warmup_epoch", -1)
    if (
            checkpoint_scheduler != runtime_scheduler
            or checkpoint_warmup != runtime_warmup):
        raise ValueError(
            "resume_lr_scale checkpoint scheduler config differs from "
            "runtime: lr_scheduler={!r}/{!r}, warmup_epoch={!r}/{!r}"
            .format(
                checkpoint_scheduler, runtime_scheduler,
                checkpoint_warmup, runtime_warmup,
            )
        )
    if expected_lineage is not None and expected_lineage != prior_lineage:
        raise ValueError(
            "resume_lr_scale expected checkpoint lineage={} but found {}"
            .format(expected_lineage, prior_lineage)
        )
    if prior_lineage != 1.0 and expected_lineage is None:
        raise ValueError(
            "resume_lr_scale cannot be applied to a checkpoint whose run "
            "already recorded cumulative resume_lr_scale lineage={} without "
            "an explicit matching resume_lr_scale_expected_lineage"
            .format(prior_lineage)
        )
    return scale


def _scale_resumed_learning_rates(optimizer, scheduler, scale):
    if not isinstance(
            scheduler, torch.optim.lr_scheduler.MultiStepLR):
        raise ValueError(
            "resume_lr_scale requires a plain MultiStepLR scheduler"
        )
    old_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    scheduler_last_lrs = [
        float(value) for value in scheduler.get_last_lr()
    ]
    if len(scheduler_last_lrs) != len(old_lrs) or any(
            not math.isclose(current, recorded, rel_tol=0.0, abs_tol=0.0)
            for current, recorded in zip(old_lrs, scheduler_last_lrs)):
        raise ValueError(
            "optimizer current learning rates differ from scheduler _last_lr"
        )
    new_lrs = [value * scale for value in old_lrs]
    for group, learning_rate in zip(optimizer.param_groups, new_lrs):
        group["lr"] = learning_rate
    # Preserve initial_lr, base_lrs, milestones, and scheduler progress.
    # Chainable MultiStepLR reads the current optimizer LR and multiplies it
    # only when a future milestone is crossed.
    scheduler._last_lr = list(new_lrs)
    return old_lrs, new_lrs


_E57_RESTORE_EPOCH = 57
_E57_RESTORE_EXPERIMENT = (
    "nr3d_mcln_joint_butdcls_v99_e57_restore_initial_once_e58_e62_b16a1"
)
_E57_RESTORE_CHECKPOINT_PATH = (
    "/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/"
    "control/official_rec_monitor/"
    "official_best_rec025_epoch_57_0p56500823.pth"
)
_E57_RESTORE_CHECKPOINT_SHA256 = (
    "fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
)
_E57_RESTORE_CLAIM_PATH = (
    "/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/"
    "control/e57_restore_initial_once_claim/claim.json"
)
_E57_RESTORE_LAUNCHER_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "scripts",
    "run_nr3d_v99_e57_restore_initial_once.sh",
)
_E57_RESTORE_LAUNCHER_SHA256 = (
    "06374e93594a8d1a669eabb4bdf7d23c1991b31cfd0bcbc984c3958cbe951402"
)
_E57_RESTORE_PIPELINE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "scripts",
    "run_dataset_v99_pipeline.sh",
)
_E57_RESTORE_PIPELINE_SHA256 = (
    "264eabacb8c034ad51f4fc30ce33ef990408a19e68400069a74c575f58da31a9"
)
_E57_RESTORE_CURRENT_LRS = (1e-5, 1e-4, 1e-5, 1.25e-5)
_E57_RESTORE_INITIAL_LRS = (1e-4, 1e-3, 1e-4, 1.25e-4)
_E57_RESTORE_SCHEDULER_LAST_EPOCH = 159942
_E57_RESTORE_SCHEDULER_STEP_COUNT = 159943
_E57_RESTORE_SCHEDULER_MILESTONES = {423706: 1}
_E57_RESTORE_RUNTIME_CONTRACT = (
    ("augment_det", False),
    ("batch_size", 16),
    ("butd", False),
    ("butd_cls", True),
    ("butd_gt", False),
    ("checkpoint_start_epoch", None),
    ("checkpoint_metric_retention", True),
    ("checkpoint_retention_metrics", ["rec_acc025"]),
    ("data_root", "/root/autodl-tmp/DATA_ROOT/"),
    ("dataloader_prefetch_factor", 2),
    ("dataset", ["nr3d"]),
    ("detect_intermediate", True),
    ("drop_incomplete_accumulation_group", False),
    ("eval", False),
    ("eval_use_selector_choice_scores", True),
    ("expected_eval_sample_count", 7899),
    ("frozen", False),
    ("gradient_accumulation_steps", 1),
    ("joint_det", True),
    ("lr", 1e-4),
    ("lr_backbone", 1e-3),
    ("lr_decay_epochs", [150]),
    ("lr_decay_rate", 0.1),
    ("lr_scheduler", "step"),
    ("mask_head_lr_multiplier", 1.0),
    ("max_epoch", 62),
    ("max_train_batches", 0),
    ("model", "MCLN"),
    ("model_only_initialization", False),
    ("num_decoder_layers", 6),
    ("num_target", 256),
    ("num_workers", 4),
    ("optimizer", "adamW"),
    ("persistent_train_workers", True),
    ("pp_checkpoint", "/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth"),
    ("print_freq", 20),
    ("reduce_lr", False),
    ("restore_e57_lr_to_initial", True),
    ("resume_lr_scale", 1.0),
    ("save_freq", 1),
    ("self_attend", True),
    ("skip_missing_superpoints", True),
    ("small_lr", False),
    (
        "source_choice_selector_choice_target",
        "precision_gain_default_sourcewise_focal_bce",
    ),
    ("source_choice_selector_default_source", "default"),
    ("source_choice_selector_hidden_dim", 288),
    ("source_choice_selector_loss_weight", 0.5),
    ("source_choice_selector_lr", 1.25e-4),
    ("source_choice_selector_min_iou_gap", 0.03),
    (
        "source_choice_selector_sources",
        "default,default_rank_blend_contrastive010",
    ),
    ("start_epoch", 1),
    ("test_dataset", "nr3d"),
    ("text_encoder_lr", 1e-5),
    ("use_color", True),
    ("use_contrastive_align", True),
    ("use_height", False),
    ("use_multiview", False),
    ("use_soft_token_loss", True),
    ("use_source_choice_selector", True),
    ("val_freq", 1),
    ("warmup_epoch", -1),
    ("weight_decay", 5e-4),
    ("wo_obj_name", "None"),
)
_E57_V99_SELECTOR_CONFIG = (
    ("eval_use_selector_choice_scores", True),
    (
        "source_choice_selector_sources",
        "default,default_rank_blend_contrastive010",
    ),
    ("source_choice_selector_default_source", "default"),
    ("source_choice_selector_hidden_dim", 288),
    ("source_choice_selector_lr", 1.25e-4),
    ("source_choice_selector_loss_weight", 0.5),
    (
        "source_choice_selector_choice_target",
        "precision_gain_default_sourcewise_focal_bce",
    ),
    ("source_choice_selector_min_iou_gap", 0.03),
)
_E57_NON_V99_BOOLEAN_FLAGS = (
    "use_source_moe",
    "source_moe_train_only",
    "source_moe_gate_train_only",
    "source_moe_gate_new_heads_only",
    "source_moe_gate_resume_optimizer",
    "use_decoder_query_adapter",
    "decoder_query_adapter_train_only",
    "use_query_mask_fusion_calibrator",
    "query_mask_fusion_train_only",
    "query_mask_fusion_resume_optimizer",
    "use_egqs_mask_refiner",
    "egqs_mask_refiner_train_only",
    "use_joint_query_quality_reranker",
    "joint_query_quality_train_only",
    "joint_query_quality_use_spatial_mask_refiner",
    "use_sacr_source",
    "use_sacr_score_refiner",
    "sacr_score_refiner_train_only",
    "use_parent_relative_text_verifier",
    "parent_relative_text_verifier_train_only",
    "sacr_score_use_parent_relative_abstention",
    "sacr_score_use_relation_counterfactual",
    "relation_counterfactual_aux_conservative_anchor_set",
    "eval_use_rec_reranker_scores",
    "eval_use_rec_geometry_reranker_scores",
    "eval_use_rec_joint_box_mask",
    "eval_use_rec_selective_residual_scores",
    "eval_use_rec_hierarchical_reranker_scores",
)
_E57_NON_V99_PATH_FIELDS = (
    "legacy_scene_graph_cache",
    "rec_reranker_checkpoint",
    "rec_geometry_reranker_checkpoint",
    "rec_joint_box_mask_checkpoint",
    "rec_selective_residual_checkpoint",
    "rec_hierarchical_reranker_checkpoint",
)


def _sha256_open_checkpoint(handle):
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path):
    with open(path, "rb") as handle:
        return _sha256_open_checkpoint(handle)


def _e57_restore_runtime_contract():
    return {
        name: copy.deepcopy(expected)
        for name, expected in _E57_RESTORE_RUNTIME_CONTRACT
    }


def _load_checkpoint_payload(args):
    fpr_scene_audit = bool(getattr(
        args, "fpr_scene_disjoint_audit", False
    ))
    density_scene_audit = bool(getattr(
        args, "density_aware_target_box_scene_disjoint_audit", False
    ))
    density_aware_target_box = (
        float(getattr(args, "density_aware_target_box_loss_weight", 0.0))
        > 0.0 or density_scene_audit
    )
    if fpr_scene_audit and density_aware_target_box:
        raise ValueError(
            "density-aware target-box audit is incompatible with FPR scene audit"
        )
    if not (
            bool(getattr(args, "restore_e57_lr_to_initial", False))
            or fpr_scene_audit or density_aware_target_box):
        return torch.load(args.checkpoint_path, map_location="cpu")
    if fpr_scene_audit:
        expected_sha256 = str(getattr(
            args, "fpr_scene_disjoint_checkpoint_sha256", ""
        ) or "").lower()
        required_sha256 = (
            FPR_SCENE_DISJOINT_AV4_E57_SHA256
            if bool(getattr(args, "fpr_scene_disjoint_av4_audit", False))
            else FPR_SCENE_DISJOINT_E57_SHA256
        )
        if expected_sha256 != required_sha256:
            raise ValueError(
                "FPR scene audit requires the protected E57 SHA-256"
            )
        with open(args.checkpoint_path, "rb") as handle:
            observed_before = _sha256_open_checkpoint(handle)
            if observed_before != expected_sha256:
                raise ValueError(
                    "FPR scene audit checkpoint SHA-256 mismatch: {} != {}"
                    .format(observed_before, expected_sha256)
                )
            handle.seek(0)
            checkpoint = torch.load(handle, map_location="cpu")
            handle.seek(0)
            observed_after = _sha256_open_checkpoint(handle)
            if observed_after != observed_before:
                raise RuntimeError(
                    "FPR scene audit checkpoint changed while loading"
                )
        checkpoint_epoch = checkpoint.get("epoch")
        if (
                not isinstance(checkpoint_epoch, int)
                or isinstance(checkpoint_epoch, bool)
                or checkpoint_epoch != 57):
            raise ValueError(
                "FPR scene audit requires checkpoint-internal epoch 57"
            )
        args.fpr_scene_disjoint_consumed_checkpoint_sha256 = (
            observed_before
        )
        args.fpr_scene_disjoint_consumed_checkpoint_epoch = checkpoint_epoch
        return checkpoint
    if density_aware_target_box:
        expected_sha256 = str(getattr(
            args, "density_aware_target_box_checkpoint_sha256", ""
        ) or "").lower()
        if (len(expected_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in expected_sha256)):
            raise ValueError(
                "density-aware target-box checkpoint SHA-256 must be 64 hex digits"
            )
        if (
                density_scene_audit
                and expected_sha256
                != DENSITY_TARGET_BOX_SCENE_AUDIT_E57_SHA256):
            raise ValueError(
                "density scene audit requires the protected full-state E57"
            )
        with open(args.checkpoint_path, "rb") as handle:
            observed_before = _sha256_open_checkpoint(handle)
            if observed_before != expected_sha256:
                raise ValueError(
                    "density-aware target-box checkpoint SHA-256 mismatch: "
                    "{} != {}".format(observed_before, expected_sha256)
                )
            handle.seek(0)
            checkpoint = torch.load(handle, map_location="cpu")
            handle.seek(0)
            observed_after = _sha256_open_checkpoint(handle)
        if observed_after != observed_before:
            raise RuntimeError(
                "density-aware target-box checkpoint changed while loading"
            )
        checkpoint_epoch = checkpoint.get("epoch")
        if (not isinstance(checkpoint_epoch, int)
                or isinstance(checkpoint_epoch, bool)):
            raise ValueError(
                "density-aware target-box checkpoint epoch must be an integer"
            )
        if density_scene_audit and checkpoint_epoch != 57:
            raise ValueError(
                "density scene audit requires checkpoint-internal epoch 57"
            )
        args.density_aware_target_box_consumed_checkpoint_sha256 = (
            observed_before
        )
        args.density_aware_target_box_consumed_checkpoint_epoch = (
            checkpoint_epoch
        )
        return checkpoint
    with open(args.checkpoint_path, "rb") as handle:
        observed_before = _sha256_open_checkpoint(handle)
        if observed_before != _E57_RESTORE_CHECKPOINT_SHA256:
            raise ValueError(
                "E57 LR restore checkpoint SHA-256 mismatch: {} != {}"
                .format(observed_before, _E57_RESTORE_CHECKPOINT_SHA256)
            )
        handle.seek(0)
        checkpoint = torch.load(handle, map_location="cpu")
        handle.seek(0)
        observed_after = _sha256_open_checkpoint(handle)
        if observed_after != observed_before:
            raise RuntimeError("E57 LR restore checkpoint changed while loading")
    return checkpoint


def _commit_e57_lr_restore_claim(args):
    claim_path = os.path.realpath(getattr(args, "e57_lr_restore_claim", ""))
    expected_claim_path = os.path.realpath(_E57_RESTORE_CLAIM_PATH)
    if claim_path != expected_claim_path:
        raise ValueError("E57 LR restore one-shot claim path mismatch")
    with open(claim_path, "rb") as handle:
        raw_claim = handle.read()
    try:
        claim = json.loads(raw_claim.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("E57 LR restore one-shot claim is invalid") from error
    claim_sha256 = hashlib.sha256(raw_claim).hexdigest()

    main_utils_path = os.path.realpath(__file__)
    launcher_path = os.path.realpath(_E57_RESTORE_LAUNCHER_PATH)
    pipeline_path = os.path.realpath(_E57_RESTORE_PIPELINE_PATH)
    observed_main_utils_sha256 = _sha256_file(main_utils_path)
    observed_launcher_sha256 = _sha256_file(launcher_path)
    observed_pipeline_sha256 = _sha256_file(pipeline_path)
    if observed_launcher_sha256 != _E57_RESTORE_LAUNCHER_SHA256:
        raise ValueError("E57 LR restore launcher SHA-256 mismatch")
    if observed_pipeline_sha256 != _E57_RESTORE_PIPELINE_SHA256:
        raise ValueError("E57 LR restore pipeline SHA-256 mismatch")

    training_contract = _e57_restore_runtime_contract()
    expected_fields = {
        "schema": "mcln-e57-lr-restore-one-shot-claim-v1",
        "experiment": _E57_RESTORE_EXPERIMENT,
        "launcher": launcher_path,
        "launcher_sha256": _E57_RESTORE_LAUNCHER_SHA256,
        "main_utils_sha256": observed_main_utils_sha256,
        "pipeline_sha256": _E57_RESTORE_PIPELINE_SHA256,
        "checkpoint": os.path.realpath(_E57_RESTORE_CHECKPOINT_PATH),
        "checkpoint_sha256": _E57_RESTORE_CHECKPOINT_SHA256,
        "one_shot_consumed": True,
        "training_contract": training_contract,
    }
    for name, expected in expected_fields.items():
        actual = claim.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "E57 LR restore one-shot claim {} mismatch: {!r} != {!r}"
                .format(name, actual, expected)
            )

    runtime_log_dir = os.path.realpath(getattr(args, "log_dir", ""))
    epoch_leaf = os.path.basename(runtime_log_dir)
    experiment_dir = os.path.dirname(runtime_log_dir)
    dataset_dir = os.path.dirname(experiment_dir)
    claimed_backbone_value = claim.get("backbone_run_dir")
    if not isinstance(claimed_backbone_value, str):
        raise ValueError("E57 LR restore claim backbone path is invalid")
    claimed_backbone_root = os.path.realpath(claimed_backbone_value)
    runtime_datasets = _canonical_single_dataset(
        getattr(args, "dataset", None)
    )
    if (
            not epoch_leaf.isdigit()
            or os.path.basename(experiment_dir) != expected_fields["experiment"]
            or os.path.basename(dataset_dir) != runtime_datasets
            or os.path.dirname(dataset_dir) != claimed_backbone_root):
        raise ValueError(
            "E57 LR restore runtime log directory is not bound to its claim"
        )

    claim_root = os.path.dirname(claim_path)
    commit_path = os.path.join(claim_root, "restore_committed.json")
    payload = {
        "schema": "mcln-e57-lr-restore-commit-v1",
        "claim": claim_path,
        "claim_sha256": claim_sha256,
        "checkpoint_sha256": _E57_RESTORE_CHECKPOINT_SHA256,
        "experiment": expected_fields["experiment"],
        "launcher": launcher_path,
        "launcher_sha256": observed_launcher_sha256,
        "main_utils": main_utils_path,
        "main_utils_sha256": observed_main_utils_sha256,
        "pipeline": pipeline_path,
        "pipeline_sha256": observed_pipeline_sha256,
        "training_contract": training_contract,
        "backbone_run_dir": claimed_backbone_root,
        "runtime_log_dir": runtime_log_dir,
        "pid": os.getpid(),
        "committed_unix_time": time.time(),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(commit_path, flags, 0o444)
    except FileExistsError as error:
        raise ValueError(
            "E57 LR restore one-shot claim was already committed"
        ) from error
    serialized = (
        json.dumps(payload, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        # A partial file intentionally remains fail-closed if writing fails.
        directory_fd = os.open(claim_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return commit_path


def _checkpoint_config_value(checkpoint_config, name, default):
    if isinstance(checkpoint_config, dict):
        return checkpoint_config.get(name, default)
    return getattr(checkpoint_config, name, default)


def _canonical_single_dataset(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return None


def _exact_float_sequence(actual, expected, label):
    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        raise ValueError("{} topology mismatch".format(label))
    values = tuple(float(value) for value in actual)
    if any(
            not math.isclose(value, reference, rel_tol=0.0, abs_tol=0.0)
            for value, reference in zip(values, expected)):
        raise ValueError(
            "{} mismatch: {} != {}".format(label, values, expected)
        )
    return values


def _validate_e57_restore_runtime_contract(args):
    missing = object()
    for name, expected in _E57_RESTORE_RUNTIME_CONTRACT:
        actual = getattr(args, name, missing)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "E57 LR restore runtime contract {} mismatch: {!r} != {!r}"
                .format(name, actual, expected)
            )
    if getattr(args, "exp", None) != _E57_RESTORE_EXPERIMENT:
        raise ValueError("E57 LR restore experiment identity mismatch")
    if (
            os.path.realpath(getattr(args, "checkpoint_path", ""))
            != os.path.realpath(_E57_RESTORE_CHECKPOINT_PATH)):
        raise ValueError("E57 LR restore checkpoint path mismatch")


def _validated_e57_lr_restore(args, checkpoint):
    requested = bool(getattr(args, "restore_e57_lr_to_initial", False))
    checkpoint_config = checkpoint.get("config", {})
    prior_lineage = _checkpoint_config_value(
        checkpoint_config,
        "e57_lr_restore_lineage",
        bool(_checkpoint_config_value(
            checkpoint_config, "restore_e57_lr_to_initial", False
        )),
    )
    if not isinstance(prior_lineage, bool):
        raise ValueError("checkpoint E57 LR-restore lineage must be boolean")
    # Preserve the one-time marker through ordinary scale=1 continuations.
    args.e57_lr_restore_lineage = prior_lineage
    if not requested:
        return False
    if prior_lineage:
        raise ValueError("E57 learning rates were already restored")
    _validate_e57_restore_runtime_contract(args)
    if (
            os.path.realpath(getattr(args, "e57_lr_restore_claim", ""))
            != os.path.realpath(_E57_RESTORE_CLAIM_PATH)):
        raise ValueError("E57 LR restore requires the reviewed one-shot claim")
    if float(getattr(args, "resume_lr_scale", 1.0)) != 1.0:
        raise ValueError(
            "restore_e57_lr_to_initial is mutually exclusive with "
            "resume_lr_scale"
        )
    resume_scale_lineage = _checkpoint_config_value(
        checkpoint_config,
        "resume_lr_scale_lineage",
        _checkpoint_config_value(checkpoint_config, "resume_lr_scale", 1.0),
    )
    if (
            isinstance(resume_scale_lineage, bool)
            or not isinstance(resume_scale_lineage, numbers.Real)
            or not math.isfinite(float(resume_scale_lineage))
            or float(resume_scale_lineage) != 1.0):
        raise ValueError(
            "E57 LR restore requires unmodified resume_lr_scale lineage"
        )

    incompatible_flags = [
        name for name in (
            "eval",
            "reduce_lr",
            "model_only_initialization",
            "frozen",
            "small_lr",
            "source_choice_selector_train_only",
            "joint_query_quality_train_only",
            "migrate_scheduler_for_gradient_accumulation",
        ) + _E57_NON_V99_BOOLEAN_FLAGS
        if bool(getattr(args, name, False))
    ]
    for name in _E57_NON_V99_PATH_FIELDS:
        if getattr(args, name, None) not in (None, ""):
            incompatible_flags.append(name)
    if float(getattr(
            args, "relation_counterfactual_aux_loss_weight", 0.0)) != 0.0:
        incompatible_flags.append("relation_counterfactual_aux_loss_weight")
    if float(getattr(
            args, "tier_hard_query_aux_loss_weight", 0.0)) != 0.0:
        incompatible_flags.append("tier_hard_query_aux_loss_weight")
    if getattr(args, "checkpoint_start_epoch", None) is not None:
        incompatible_flags.append("checkpoint_start_epoch")
    if incompatible_flags:
        raise ValueError(
            "restore_e57_lr_to_initial requires an otherwise exact full-state "
            "training resume; incompatible settings: {}".format(
                ", ".join(incompatible_flags)
            )
        )
    if getattr(args, "lr_scheduler", "step") != "step":
        raise ValueError("E57 LR restore requires the step scheduler")
    if getattr(args, "warmup_epoch", -1) != -1:
        raise ValueError("E57 LR restore requires warmup_epoch=-1")
    if getattr(args, "batch_size", None) != 16:
        raise ValueError("E57 LR restore requires batch_size=16")
    if getattr(args, "gradient_accumulation_steps", 1) != 1:
        raise ValueError("E57 LR restore requires accumulation=1")
    if _canonical_single_dataset(getattr(args, "dataset", None)) != "nr3d":
        raise ValueError("E57 LR restore requires the Nr3D runtime dataset")
    if not bool(getattr(args, "joint_det", False)):
        raise ValueError("E57 LR restore requires joint_det")
    if not bool(getattr(args, "butd_cls", False)):
        raise ValueError("E57 LR restore requires butd_cls")
    if not bool(getattr(args, "use_source_choice_selector", False)):
        raise ValueError("E57 LR restore requires the V99 selector")
    for name, expected in _E57_V99_SELECTOR_CONFIG:
        actual = getattr(args, name, None)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "E57 LR restore runtime V99 {} mismatch: {!r} != {!r}"
                .format(name, actual, expected)
            )

    checkpoint_epoch = checkpoint.get("epoch")
    if (
            not isinstance(checkpoint_epoch, int)
            or isinstance(checkpoint_epoch, bool)
            or checkpoint_epoch != _E57_RESTORE_EPOCH):
        raise ValueError("E57 LR restore requires checkpoint epoch 57")
    missing = object()
    checkpoint_dataset = _checkpoint_config_value(
        checkpoint_config, "dataset", missing
    )
    if _canonical_single_dataset(checkpoint_dataset) != "nr3d":
        raise ValueError("E57 checkpoint dataset provenance mismatch")
    expected_config = (
        ("lr_scheduler", "step"),
        ("warmup_epoch", -1),
        ("batch_size", 16),
        ("gradient_accumulation_steps", 1),
        ("joint_det", True),
        ("butd_cls", True),
        ("use_source_choice_selector", True),
    ) + _E57_V99_SELECTOR_CONFIG
    for name, expected in expected_config:
        actual = _checkpoint_config_value(
            checkpoint_config,
            name,
            1 if name == "gradient_accumulation_steps" else missing,
        )
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "E57 checkpoint {} provenance mismatch: {!r} != {!r}"
                .format(name, actual, expected)
            )
    for name in _E57_NON_V99_BOOLEAN_FLAGS:
        actual = _checkpoint_config_value(checkpoint_config, name, False)
        if type(actual) is not bool or actual:
            raise ValueError(
                "E57 checkpoint contains non-V99 branch {}={!r}"
                .format(name, actual)
            )
    for name in _E57_NON_V99_PATH_FIELDS:
        actual = _checkpoint_config_value(checkpoint_config, name, None)
        if actual not in (None, ""):
            raise ValueError(
                "E57 checkpoint contains non-V99 path {}={!r}"
                .format(name, actual)
            )
    checkpoint_aux_weight = _checkpoint_config_value(
        checkpoint_config, "relation_counterfactual_aux_loss_weight", 0.0
    )
    if (
            isinstance(checkpoint_aux_weight, bool)
            or not isinstance(checkpoint_aux_weight, numbers.Real)
            or not math.isfinite(float(checkpoint_aux_weight))
            or float(checkpoint_aux_weight) != 0.0):
        raise ValueError("E57 checkpoint relation auxiliary loss is nonzero")

    checkpoint_optimizer = checkpoint.get("optimizer")
    checkpoint_scheduler = checkpoint.get("scheduler")
    if not isinstance(checkpoint_optimizer, dict):
        raise ValueError("E57 checkpoint optimizer state is missing")
    if not isinstance(checkpoint_scheduler, dict):
        raise ValueError("E57 checkpoint scheduler state is missing")
    parameter_groups = checkpoint_optimizer.get("param_groups")
    optimizer_state = checkpoint_optimizer.get("state")
    if not isinstance(parameter_groups, list) or len(parameter_groups) != 4:
        raise ValueError("E57 checkpoint must have four optimizer groups")
    if not isinstance(optimizer_state, dict) or len(optimizer_state) != 716:
        raise ValueError("E57 checkpoint must have 716 optimizer states")
    checkpoint_current_lrs = []
    checkpoint_initial_lrs = []
    for index, group in enumerate(parameter_groups):
        if "lr" not in group or "initial_lr" not in group:
            raise ValueError(
                "E57 optimizer group {} lacks LR provenance".format(index)
            )
        checkpoint_current_lrs.append(float(group["lr"]))
        checkpoint_initial_lrs.append(float(group["initial_lr"]))
    _exact_float_sequence(
        checkpoint_current_lrs,
        _E57_RESTORE_CURRENT_LRS,
        "E57 optimizer current LRs",
    )
    _exact_float_sequence(
        checkpoint_initial_lrs,
        _E57_RESTORE_INITIAL_LRS,
        "E57 optimizer initial LRs",
    )
    _exact_float_sequence(
        checkpoint_scheduler.get("base_lrs"),
        _E57_RESTORE_INITIAL_LRS,
        "E57 scheduler base LRs",
    )
    _exact_float_sequence(
        checkpoint_scheduler.get("_last_lr"),
        _E57_RESTORE_CURRENT_LRS,
        "E57 scheduler current LRs",
    )
    scheduler_last_epoch = checkpoint_scheduler.get("last_epoch")
    if (
            not isinstance(scheduler_last_epoch, int)
            or isinstance(scheduler_last_epoch, bool)
            or scheduler_last_epoch != _E57_RESTORE_SCHEDULER_LAST_EPOCH):
        raise ValueError("E57 scheduler last_epoch provenance mismatch")
    scheduler_step_count = checkpoint_scheduler.get("_step_count")
    if (
            not isinstance(scheduler_step_count, int)
            or isinstance(scheduler_step_count, bool)
            or scheduler_step_count != _E57_RESTORE_SCHEDULER_STEP_COUNT):
        raise ValueError("E57 scheduler step-count provenance mismatch")
    if dict(checkpoint_scheduler.get("milestones", {})) != (
            _E57_RESTORE_SCHEDULER_MILESTONES):
        raise ValueError("E57 scheduler milestone provenance mismatch")
    return True


def _e57_learning_rate_restore_plan(optimizer, scheduler):
    if not isinstance(
            scheduler, torch.optim.lr_scheduler.MultiStepLR):
        raise ValueError("E57 LR restore requires a plain MultiStepLR")
    if len(optimizer.param_groups) != 4 or len(optimizer.state) != 716:
        raise ValueError("loaded E57 optimizer topology mismatch")
    if (
            scheduler.last_epoch != _E57_RESTORE_SCHEDULER_LAST_EPOCH
            or scheduler._step_count != _E57_RESTORE_SCHEDULER_STEP_COUNT
            or dict(scheduler.milestones)
            != _E57_RESTORE_SCHEDULER_MILESTONES):
        raise ValueError("loaded E57 scheduler provenance mismatch")
    old_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
    initial_lrs = tuple(
        float(group["initial_lr"]) for group in optimizer.param_groups
    )
    _exact_float_sequence(
        old_lrs, _E57_RESTORE_CURRENT_LRS, "loaded E57 current LRs"
    )
    _exact_float_sequence(
        initial_lrs, _E57_RESTORE_INITIAL_LRS, "loaded E57 initial LRs"
    )
    _exact_float_sequence(
        scheduler.get_last_lr(),
        _E57_RESTORE_CURRENT_LRS,
        "loaded E57 scheduler current LRs",
    )
    _exact_float_sequence(
        scheduler.base_lrs,
        _E57_RESTORE_INITIAL_LRS,
        "loaded E57 scheduler base LRs",
    )
    return list(old_lrs), list(initial_lrs)


def _apply_e57_learning_rate_restore(optimizer, scheduler, new_lrs):
    for group, learning_rate in zip(optimizer.param_groups, new_lrs):
        group["lr"] = learning_rate
    # MultiStepLR is chainable: future milestones multiply the optimizer's
    # current LR. Preserve every scheduler field except its current-LR receipt.
    scheduler._last_lr = list(new_lrs)


def load_checkpoint(args, model, optimizer, scheduler,
                    optimizer_steps_per_epoch=None):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = _load_checkpoint_payload(args)
    resume_lr_scale = _validated_resume_lr_scale(args, checkpoint)
    restore_e57_lr = _validated_e57_lr_restore(args, checkpoint)
    validate_query_mask_fusion_resume_checkpoint_contract(args, checkpoint)
    validate_source_moe_resume_checkpoint_contract(args, checkpoint)
    validate_source_moe_gate_checkpoint_contract(model, checkpoint)
    requested_start_epoch = getattr(args, "start_epoch", 1)
    gate_optimizer_resume = bool(getattr(
        args, "source_moe_gate_resume_optimizer", False
    ))
    query_optimizer_resume = bool(getattr(
        args, "query_mask_fusion_resume_optimizer", False
    ))
    try:
        args.start_epoch = int(checkpoint['epoch']) + 1
    except Exception:
        args.start_epoch = 0
    if (
            (
                getattr(args, "source_choice_selector_train_only", False)
                or getattr(args, "source_moe_train_only", False)
                or getattr(args, "source_moe_gate_train_only", False)
                or getattr(args, "query_mask_fusion_train_only", False)
                or getattr(args, "egqs_mask_refiner_train_only", False)
                or getattr(args, "joint_query_quality_train_only", False)
                or getattr(args, "decoder_query_adapter_train_only", False)
                or getattr(args, "sacr_score_refiner_train_only", False)
                or getattr(
                    args, "parent_relative_text_verifier_train_only", False
                )
            )
            and not args.eval
            and not gate_optimizer_resume
            and not query_optimizer_resume):
        args.start_epoch = requested_start_epoch
    if ((gate_optimizer_resume or query_optimizer_resume)
            and requested_start_epoch != args.start_epoch):
        resume_label = (
            "gate" if gate_optimizer_resume else "query-mask fusion"
        )
        raise ValueError(
            "{} optimizer resume must start at checkpoint epoch + 1 "
            "(expected {}, requested {})".format(
                resume_label, args.start_epoch, requested_start_epoch
            )
        )
    checkpoint_start_epoch = getattr(args, "checkpoint_start_epoch", None)
    if ((gate_optimizer_resume or query_optimizer_resume)
            and checkpoint_start_epoch is not None):
        raise ValueError(
            "optimizer resume cannot override checkpoint_start_epoch"
        )
    if checkpoint_start_epoch is not None:
        if checkpoint_start_epoch < 0:
            raise ValueError("checkpoint_start_epoch must be non-negative")
        args.start_epoch = checkpoint_start_epoch
        print(
            "=> overriding checkpoint epoch; first requested epoch is {}".format(
                args.start_epoch
            )
        )
    model_only_initialization = bool(getattr(
        args, "model_only_initialization", False
    ))
    if model_only_initialization:
        if args.eval:
            raise ValueError("model-only initialization is training-only")
        if checkpoint_start_epoch is None:
            raise ValueError(
                "model-only initialization requires --checkpoint_start_epoch"
            )
        if gate_optimizer_resume or query_optimizer_resume:
            raise ValueError(
                "model-only initialization cannot resume an optimizer"
            )
    current_state = model.state_dict()
    checkpoint_state = checkpoint['model']
    score_state_prefixes = (
        "structured_slot_builder.",
        "sacr_head.",
        "sacr_score_gate",
        "sacr_parent_relative_gate.",
    )

    def canonical_state_name(name):
        return name[7:] if name.startswith("module.") else name

    parent_relative_text_verifier_eval = (
        getattr(args, "eval", False)
        and getattr(args, "use_parent_relative_text_verifier", False)
    )
    if parent_relative_text_verifier_eval:
        current_full_state = {
            canonical_state_name(key): value
            for key, value in current_state.items()
        }
        checkpoint_full_state = {
            canonical_state_name(key): value
            for key, value in checkpoint_state.items()
        }
        if set(current_full_state) != set(checkpoint_full_state):
            raise ValueError(
                "trained parent-relative verifier checkpoint is not an exact "
                "full model: missing={}, unexpected={}".format(
                    sorted(set(current_full_state) - set(checkpoint_full_state)),
                    sorted(set(checkpoint_full_state) - set(current_full_state)),
                )
            )
        incompatible_full_tensors = []
        for name in sorted(current_full_state):
            current_value = current_full_state[name]
            saved_value = checkpoint_full_state[name]
            if (
                    not hasattr(current_value, "shape")
                    or not hasattr(saved_value, "shape")
                    or current_value.shape != saved_value.shape
                    or current_value.dtype != saved_value.dtype):
                incompatible_full_tensors.append(name)
        if incompatible_full_tensors:
            raise ValueError(
                "trained parent-relative verifier checkpoint full model "
                "tensors differ in shape/dtype: "
                + ", ".join(incompatible_full_tensors)
            )

    def is_score_state(name):
        canonical = canonical_state_name(name)
        return any(
            canonical.startswith(prefix) for prefix in score_state_prefixes
        )

    current_score_state = {
        canonical_state_name(key): (key, value)
        for key, value in current_state.items() if is_score_state(key)
    }
    checkpoint_score_state = {
        canonical_state_name(key): (key, value)
        for key, value in checkpoint_state.items() if is_score_state(key)
    }
    checkpoint_has_trained_score_refiner = (
        "sacr_score_gate" in checkpoint_score_state
        or any(
            name.startswith("sacr_parent_relative_gate.")
            for name in checkpoint_score_state
        )
    )
    checkpoint_config = checkpoint.get("config", {})
    if isinstance(checkpoint_config, dict):
        checkpoint_uses_parent_relative_sacr = (
            checkpoint_config.get(
                "sacr_score_use_parent_relative_abstention", False
            ) is True
        )
    else:
        checkpoint_uses_parent_relative_sacr = (
            getattr(
                checkpoint_config,
                "sacr_score_use_parent_relative_abstention",
                False,
            ) is True
        )
    converting_v133_to_parent_relative = (
        getattr(args, "sacr_score_refiner_train_only", False)
        and getattr(
            args, "sacr_score_use_parent_relative_abstention", False
        )
        and checkpoint_has_trained_score_refiner
        and "sacr_score_gate" in checkpoint_score_state
        and not checkpoint_uses_parent_relative_sacr
    )
    discarded_legacy_score_state = (
        {"sacr_score_gate"}
        if converting_v133_to_parent_relative else set()
    )
    expected_new_parent_gate_state = {
        name for name in current_score_state
        if name.startswith("sacr_parent_relative_gate.")
    } if converting_v133_to_parent_relative else set()
    expected_new_parent_gate_load_keys = {
        current_score_state[name][0]
        for name in expected_new_parent_gate_state
    }
    if (
            getattr(args, "use_sacr_score_refiner", False)
            and checkpoint_has_trained_score_refiner):
        expected_score_state = (
            (set(checkpoint_score_state) - discarded_legacy_score_state)
            | expected_new_parent_gate_state
        )
        if set(current_score_state) != expected_score_state:
            raise ValueError(
                "trained SACR score checkpoint state is not exact: "
                "missing={}, unexpected={}".format(
                    sorted(set(current_score_state) - set(checkpoint_score_state)),
                    sorted(set(checkpoint_score_state) - set(current_score_state)),
                )
            )
        incompatible_score_tensors = []
        for name in sorted(
                set(current_score_state) & set(checkpoint_score_state)):
            current_value = current_score_state[name][1]
            saved_value = checkpoint_score_state[name][1]
            if (
                    not hasattr(current_value, "shape")
                    or not hasattr(saved_value, "shape")
                    or current_value.shape != saved_value.shape
                    or current_value.dtype != saved_value.dtype):
                incompatible_score_tensors.append(name)
        if incompatible_score_tensors:
            raise ValueError(
                "trained SACR score checkpoint tensors differ in shape/dtype: "
                + ", ".join(incompatible_score_tensors)
            )
        current_full_state = {
            canonical_state_name(key): (key, value)
            for key, value in current_state.items()
        }
        checkpoint_full_state = {
            canonical_state_name(key): (key, value)
            for key, value in checkpoint_state.items()
        }
        expected_full_state = (
            (set(checkpoint_full_state) - discarded_legacy_score_state)
            | expected_new_parent_gate_state
        )
        if set(current_full_state) != expected_full_state:
            raise ValueError(
                "trained SACR checkpoint full model state is not exact: "
                "missing={}, unexpected={}".format(
                    sorted(set(current_full_state) - set(checkpoint_full_state)),
                    sorted(set(checkpoint_full_state) - set(current_full_state)),
                )
            )
        incompatible_full_tensors = []
        for name in sorted(
                set(current_full_state) & set(checkpoint_full_state)):
            current_value = current_full_state[name][1]
            saved_value = checkpoint_full_state[name][1]
            if (
                    not hasattr(current_value, "shape")
                    or not hasattr(saved_value, "shape")
                    or current_value.shape != saved_value.shape
                    or current_value.dtype != saved_value.dtype):
                incompatible_full_tensors.append(name)
        if incompatible_full_tensors:
            raise ValueError(
                "trained SACR checkpoint full model tensors differ in "
                "shape/dtype: " + ", ".join(incompatible_full_tensors)
            )
    if converting_v133_to_parent_relative:
        checkpoint_state = {
            key: value for key, value in checkpoint_state.items()
            if canonical_state_name(key) != "sacr_score_gate"
        }
    for key, value in list(checkpoint_state.items()):
        if (
            key in current_state
            and hasattr(value, "shape")
            and hasattr(current_state[key], "shape")
            and len(value.shape) > 0
            and len(value.shape) == len(current_state[key].shape)
            and value.shape[1:] == current_state[key].shape[1:]
            and value.shape[0] != current_state[key].shape[0]
        ):
            resized_value = current_state[key].clone()
            rows = min(value.shape[0], current_state[key].shape[0])
            resized_value[:rows] = value[:rows]
            checkpoint_state[key] = resized_value
            print(
                "=> partially loaded checkpoint parameter '{}': {} -> {}".format(
                    key, tuple(value.shape), tuple(current_state[key].shape)
                )
            )
    mismatched_keys = [
        key for key, value in checkpoint_state.items()
        if (
            key in current_state
            and hasattr(value, "shape")
            and hasattr(current_state[key], "shape")
            and value.shape != current_state[key].shape
        )
    ]
    if mismatched_keys:
        checkpoint_state = {
            key: value for key, value in checkpoint_state.items()
            if key not in mismatched_keys
        }
        print(
            "=> skipped checkpoint parameters with mismatched shapes: {}".format(
                ", ".join(mismatched_keys)
            )
        )
    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    if (
            getattr(args, "eval", False)
            and getattr(args, "use_parent_relative_text_verifier", False)
            and (
                incompatible.missing_keys
                or incompatible.unexpected_keys
            )):
        raise ValueError(
            "trained parent-relative verifier checkpoint did not load as an "
            "exact full model: missing={}, unexpected={}".format(
                sorted(incompatible.missing_keys),
                sorted(incompatible.unexpected_keys),
            )
        )
    if model_only_initialization:
        loaded_keys = set(checkpoint_state).intersection(current_state)
        loaded_backbone_keys = {
            key for key in loaded_keys
            if key.startswith("module.backbone_net.")
        }
        if len(loaded_backbone_keys) < 90:
            raise ValueError(
                "model-only initialization loaded only {} backbone tensors; "
                "expected at least 90".format(len(loaded_backbone_keys))
            )
        print(
            "=> model-only initialization loaded {} tensors ({} backbone); "
            "ignored {} source-only tensors; optimizer and scheduler remain "
            "fresh".format(
                len(loaded_keys), len(loaded_backbone_keys),
                len(incompatible.unexpected_keys),
            )
        )
    if (
            getattr(args, "use_sacr_score_refiner", False)
            and checkpoint_has_trained_score_refiner
            and (
                set(incompatible.missing_keys)
                != expected_new_parent_gate_load_keys
                or incompatible.unexpected_keys)):
        raise ValueError(
            "trained SACR checkpoint did not load as an exact full model: "
            "missing={}, unexpected={}".format(
                sorted(incompatible.missing_keys),
                sorted(incompatible.unexpected_keys),
            )
        )
    if getattr(args, "query_mask_fusion_train_only", False):
        expected_missing = {
            key for key in current_state
            if (
                key[7:] if key.startswith("module.") else key
            ).startswith("query_mask_fusion_calibrator.")
            and key not in checkpoint_state
        }
        actual_missing = set(incompatible.missing_keys)
        if (actual_missing != expected_missing
                or incompatible.unexpected_keys):
            raise ValueError(
                "query mask fusion initialization has unexpected checkpoint "
                "differences: missing={}, unexpected={}".format(
                    sorted(actual_missing),
                    sorted(incompatible.unexpected_keys),
                )
            )
        print(
            "=> query mask fusion checkpoint contract verified: {} new "
            "parameters".format(len(expected_missing))
        )
    if getattr(args, "egqs_mask_refiner_train_only", False):
        expected_missing = {
            key for key in current_state
            if (
                key[7:] if key.startswith("module.") else key
            ).startswith("egqs_mask_refiner.")
            and key not in checkpoint_state
        }
        actual_missing = set(incompatible.missing_keys)
        if (actual_missing != expected_missing
                or incompatible.unexpected_keys):
            raise ValueError(
                "EGQS mask refiner initialization has unexpected checkpoint "
                "differences: missing={}, unexpected={}".format(
                    sorted(actual_missing),
                    sorted(incompatible.unexpected_keys),
                )
            )
        print(
            "=> EGQS mask refiner checkpoint contract verified: {} new "
            "parameters".format(len(expected_missing))
        )
    if getattr(args, "joint_query_quality_train_only", False):
        joint_new_prefixes = ["joint_query_quality_reranker."]
        if getattr(args, "use_sacr_source", False):
            joint_new_prefixes.extend([
                "structured_slot_builder.",
                "sacr_head.",
                "sacr_residual_scale",
            ])
        expected_missing = {
            key for key in current_state
            if any(
                (key[7:] if key.startswith("module.") else key).startswith(
                    prefix
                )
                for prefix in joint_new_prefixes
            )
            and key not in checkpoint_state
        }
        actual_missing = set(incompatible.missing_keys)
        if (actual_missing != expected_missing
                or incompatible.unexpected_keys):
            raise ValueError(
                "joint query quality initialization has unexpected "
                "checkpoint differences: missing={}, unexpected={}".format(
                    sorted(actual_missing),
                    sorted(incompatible.unexpected_keys),
                )
            )
        print(
            "=> joint query quality checkpoint contract verified: {} new "
            "parameters".format(len(expected_missing))
        )
    if getattr(args, "sacr_score_refiner_train_only", False):
        score_new_prefixes = (
            "structured_slot_builder.",
            "sacr_head.",
            "sacr_score_gate",
            "sacr_parent_relative_gate.",
        )
        expected_missing = (
            expected_new_parent_gate_load_keys
            if checkpoint_has_trained_score_refiner
            else {
                key for key in current_state
                if any(
                    canonical_state_name(key).startswith(prefix)
                    for prefix in score_new_prefixes
                )
                and key not in checkpoint_state
            }
        )
        actual_missing = set(incompatible.missing_keys)
        if (actual_missing != expected_missing
                or incompatible.unexpected_keys):
            raise ValueError(
                "SACR score initialization has unexpected checkpoint "
                "differences: missing={}, unexpected={}".format(
                    sorted(actual_missing),
                    sorted(incompatible.unexpected_keys),
                )
            )
        print(
            "=> SACR score checkpoint contract verified: {} new "
            "parameters".format(len(expected_missing))
        )
    if getattr(args, "parent_relative_text_verifier_train_only", False):
        verifier_new_prefixes = (
            "structured_slot_builder.",
            "sacr_head.",
            "parent_relative_text_verifier.",
        )
        expected_missing = {
            key for key in current_state
            if any(
                canonical_state_name(key).startswith(prefix)
                for prefix in verifier_new_prefixes
            )
            and key not in checkpoint_state
        }
        actual_missing = set(incompatible.missing_keys)
        if (actual_missing != expected_missing
                or incompatible.unexpected_keys):
            raise ValueError(
                "parent-relative verifier initialization has unexpected "
                "checkpoint differences: missing={}, unexpected={}".format(
                    sorted(actual_missing),
                    sorted(incompatible.unexpected_keys),
                )
            )
        print(
            "=> parent-relative verifier checkpoint contract verified: {} "
            "new parameters".format(len(expected_missing))
        )
    if getattr(args, "decoder_query_adapter_train_only", False):
        expected_missing = {
            key for key in current_state
            if (
                key[7:] if key.startswith("module.") else key
            ).startswith("decoder_query_adapter.")
            and key not in checkpoint_state
        }
        actual_missing = set(incompatible.missing_keys)
        if (actual_missing != expected_missing
                or incompatible.unexpected_keys):
            raise ValueError(
                "decoder query adapter initialization has unexpected "
                "checkpoint differences: missing={}, unexpected={}".format(
                    sorted(actual_missing),
                    sorted(incompatible.unexpected_keys),
                )
            )
        print(
            "=> decoder query adapter checkpoint contract verified: {} new "
            "parameters".format(len(expected_missing))
        )
    load_optimizer_state = (
        gate_optimizer_resume
        or query_optimizer_resume
        or (
            not args.eval
            and not args.reduce_lr
            and not model_only_initialization
            and not getattr(
                args, "source_choice_selector_train_only", False
            )
            and not getattr(args, "source_moe_train_only", False)
            and not getattr(args, "source_moe_gate_train_only", False)
            and not getattr(args, "query_mask_fusion_train_only", False)
            and not getattr(args, "egqs_mask_refiner_train_only", False)
            and not getattr(args, "joint_query_quality_train_only", False)
            and not getattr(args, "decoder_query_adapter_train_only", False)
            and not getattr(args, "sacr_score_refiner_train_only", False)
            and not getattr(
                args, "parent_relative_text_verifier_train_only", False
            )
        )
    )
    e57_lr_restored = False
    e57_lr_restore_commit_path = None
    if load_optimizer_state:
        checkpoint_optimizer_state = checkpoint['optimizer']
        checkpoint_scheduler_state = checkpoint['scheduler']
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        scheduler_before = copy.deepcopy(scheduler.state_dict())
        strict_joint_source_choice = (
            (
                getattr(args, "use_source_choice_selector", False)
                or getattr(args, "use_source_moe", False)
            )
            and not getattr(args, "frozen", False)
            and not getattr(args, "small_lr", False)
            and not gate_optimizer_resume
            and not query_optimizer_resume
        )
        try:
            if strict_joint_source_choice:
                optimizer_migration = load_mcln_optimizer_state(
                    optimizer, checkpoint_optimizer_state, model
                )
            else:
                optimizer.load_state_dict(checkpoint_optimizer_state)
                optimizer_migration = None
            scheduler_state = checkpoint_scheduler_state
            if (
                    (resume_lr_scale != 1.0 or restore_e57_lr)
                    and optimizer_migration is not None):
                raise ValueError(
                    "learning-rate intervention requires exact optimizer "
                    "parameter groups and rejects optimizer-state migration"
                )
            if optimizer_migration is not None:
                scheduler_state = migrate_mcln_scheduler_state(
                    scheduler_state, optimizer_migration
                )
                for group, learning_rate in zip(
                        optimizer.param_groups,
                        optimizer_migration["current_lrs"]):
                    group["lr"] = learning_rate
            checkpoint_config = checkpoint.get("config")
            if isinstance(checkpoint_config, dict):
                checkpoint_accumulation_steps = checkpoint_config.get(
                    "gradient_accumulation_steps", 1
                )
            else:
                checkpoint_accumulation_steps = getattr(
                    checkpoint_config, "gradient_accumulation_steps", 1
                )
            runtime_accumulation_steps = getattr(
                args, "gradient_accumulation_steps", 1
            )
            accumulation_changed = (
                checkpoint_accumulation_steps != runtime_accumulation_steps
            )
            migrate_accumulation_scheduler = bool(getattr(
                args,
                "migrate_scheduler_for_gradient_accumulation",
                False,
            ))
            if accumulation_changed and not migrate_accumulation_scheduler:
                raise ValueError(
                    "full-state resume changes gradient accumulation from {} "
                    "to {}; pass "
                    "--migrate_scheduler_for_gradient_accumulation to "
                    "normalize scheduler progress explicitly".format(
                        checkpoint_accumulation_steps,
                        runtime_accumulation_steps,
                    )
                )
            if migrate_accumulation_scheduler and not accumulation_changed:
                raise ValueError(
                    "scheduler accumulation migration was requested but the "
                    "checkpoint and runtime accumulation factors are equal"
                )
            if accumulation_changed:
                if getattr(args, "lr_scheduler", "step") != "step":
                    raise ValueError(
                        "gradient accumulation scheduler migration currently "
                        "requires the step scheduler"
                    )
                if not isinstance(optimizer_steps_per_epoch, int):
                    raise ValueError(
                        "optimizer_steps_per_epoch is required for scheduler "
                        "accumulation migration"
                    )
                checkpoint_epoch = checkpoint.get("epoch")
                if (
                        not isinstance(checkpoint_epoch, int)
                        or isinstance(checkpoint_epoch, bool)
                        or checkpoint_epoch < 0):
                    raise ValueError(
                        "scheduler accumulation migration requires a valid "
                        "checkpoint epoch"
                    )
                # Keep the newly constructed scheduler's runtime-scale
                # milestones, but place it at the same completed-epoch
                # boundary as the checkpoint. Optimizer state above supplies
                # the checkpoint's (possibly manually decayed) current LRs.
                scheduler_state = copy.deepcopy(scheduler_before)
                normalized_last_epoch = (
                    checkpoint_epoch * optimizer_steps_per_epoch
                )
                scheduler_state["last_epoch"] = normalized_last_epoch
                if "_step_count" in scheduler_state:
                    scheduler_state["_step_count"] = (
                        normalized_last_epoch + 1
                    )
                scheduler_state["_last_lr"] = [
                    group["lr"] for group in optimizer.param_groups
                ]
            scheduler.load_state_dict(scheduler_state)
            if resume_lr_scale != 1.0:
                old_lrs, new_lrs = _scale_resumed_learning_rates(
                    optimizer, scheduler, resume_lr_scale
                )
                args.resume_lr_scale_lineage = (
                    float(args.resume_lr_scale_lineage)
                    * resume_lr_scale
                )
                print(
                    "=> scaled exact-resume learning rates by {}: {} -> {}"
                    .format(resume_lr_scale, old_lrs, new_lrs)
                )
            if restore_e57_lr:
                old_lrs, new_lrs = _e57_learning_rate_restore_plan(
                    optimizer, scheduler
                )
                e57_lr_restore_commit_path = _commit_e57_lr_restore_claim(
                    args
                )
                _apply_e57_learning_rate_restore(
                    optimizer, scheduler, new_lrs
                )
                e57_lr_restored = True
                print(
                    "=> restored exact E57 current learning rates to "
                    "checkpoint initial_lr: {} -> {}".format(
                        old_lrs, new_lrs
                    )
                )
        except BaseException as original_error:
            rollback_errors = []
            try:
                optimizer.load_state_dict(optimizer_before)
            except BaseException as rollback_error:
                rollback_errors.append(("optimizer", rollback_error))
            try:
                scheduler.load_state_dict(copy.deepcopy(scheduler_before))
            except BaseException as rollback_error:
                rollback_errors.append(("scheduler", rollback_error))
            if rollback_errors:
                details = ", ".join(
                    "{}: {}".format(label, error)
                    for label, error in rollback_errors
                )
                raise RuntimeError(
                    "optimizer/scheduler checkpoint load failed and rollback "
                    "also failed ({})".format(details)
                ) from original_error
            raise
        if optimizer_migration is not None:
            print(
                "=> migrated legacy 3-group source-choice optimizer and "
                "scheduler state to strict 4-group training"
            )
        if accumulation_changed:
            print(
                "=> normalized scheduler progress for gradient accumulation "
                "{} -> {} at checkpoint epoch {} ({} optimizer steps/epoch); "
                "preserved checkpoint learning rates {}".format(
                    checkpoint_accumulation_steps,
                    runtime_accumulation_steps,
                    checkpoint_epoch,
                    optimizer_steps_per_epoch,
                    [group["lr"] for group in optimizer.param_groups],
                )
            )
        elif gate_optimizer_resume:
            print(
                "=> resumed exact gate-only optimizer and scheduler state"
            )
        elif query_optimizer_resume:
            print(
                "=> resumed exact query-mask optimizer and scheduler state"
            )
    elif restore_e57_lr:
        raise ValueError(
            "restore_e57_lr_to_initial requires optimizer/scheduler resume"
        )
    if e57_lr_restored:
        args.e57_lr_restore_lineage = True
        args.e57_lr_restore_commit_path = e57_lr_restore_commit_path

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint.get('epoch', 'model-only')
    ))

    del checkpoint
    torch.cuda.empty_cache()


def _fsync_directory(path):
    """Persist directory metadata after an atomic file replacement."""
    directory_fd = os.open(path or ".", os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_torch_save(state, path):
    temporary = "{}.tmp.{}".format(path, os.getpid())
    try:
        with open(temporary, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(os.path.dirname(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json_save(payload, path):
    temporary = "{}.tmp.{}".format(path, os.getpid())
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(os.path.dirname(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_checkpoint_hardlink(source, destination):
    """Point a stable checkpoint name at source without duplicating storage."""
    if os.path.exists(destination) and os.path.samefile(source, destination):
        return
    temporary = "{}.tmp.{}".format(destination, os.getpid())
    try:
        if os.path.exists(temporary):
            os.unlink(temporary)
        os.link(source, temporary)
        os.replace(temporary, destination)
        _fsync_directory(os.path.dirname(destination))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_retention_metric_values(values):
    if not isinstance(values, dict):
        raise ValueError("checkpoint retention metric record must be a dictionary")
    normalized = {}
    for name in CHECKPOINT_RETENTION_METRICS:
        value = values.get(name)
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise ValueError("checkpoint retention metric {} is invalid".format(name))
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("checkpoint retention metric {} is out of range".format(name))
        normalized[name] = value
    return normalized


def extract_checkpoint_retention_metrics(metrics):
    """Extract final-deployed model-selection metrics from an eval receipt."""
    if not isinstance(metrics, dict):
        raise ValueError("checkpoint retention requires an evaluation receipt")
    sample_count = metrics.get("sample_count")
    if (
        not isinstance(sample_count, numbers.Integral)
        or isinstance(sample_count, bool)
        or sample_count <= 0
    ):
        raise ValueError("evaluation receipt sample_count must be a positive integer")
    try:
        position_subgroups = metrics["position_subgroups"]
        mask = metrics["mask"]
        hits = {
            "rec_acc025": 0,
            "rec_acc050": 0,
            "mask_acc025": mask["hits025"],
            "mask_acc050": mask["hits050"],
        }
        mask_miou = mask["miou"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "evaluation receipt is missing checkpoint retention metrics"
        ) from error
    if (
        not isinstance(position_subgroups, dict)
        or set(position_subgroups) != {"unique", "multiple"}
    ):
        raise ValueError(
            "evaluation receipt position subgroups are invalid"
        )
    position_sample_count = 0
    for subgroup_name in ("unique", "multiple"):
        subgroup = position_subgroups[subgroup_name]
        if not isinstance(subgroup, dict):
            raise ValueError(
                "evaluation receipt {} position subgroup is invalid".format(
                    subgroup_name
                )
            )
        subgroup_count = subgroup.get("sample_count")
        if (
            not isinstance(subgroup_count, numbers.Integral)
            or isinstance(subgroup_count, bool)
            or subgroup_count < 0
        ):
            raise ValueError(
                "evaluation receipt {} sample_count is invalid".format(
                    subgroup_name
                )
            )
        subgroup_hits = {}
        for suffix in ("025", "050"):
            hit_count = subgroup.get("hits" + suffix)
            if (
                not isinstance(hit_count, numbers.Integral)
                or isinstance(hit_count, bool)
                or hit_count < 0
                or hit_count > subgroup_count
            ):
                raise ValueError(
                    "evaluation receipt rec_acc{} hits are invalid".format(
                        suffix
                    )
                )
            subgroup_hits[suffix] = int(hit_count)
        if subgroup_hits["050"] > subgroup_hits["025"]:
            raise ValueError(
                "evaluation receipt {} position hits are not nested".format(
                    subgroup_name
                )
            )
        position_sample_count += int(subgroup_count)
        hits["rec_acc025"] += subgroup_hits["025"]
        hits["rec_acc050"] += subgroup_hits["050"]
    if position_sample_count != sample_count:
        raise ValueError(
            "evaluation receipt position subgroups do not partition samples"
        )
    values = {}
    for name, hit_count in hits.items():
        if (
            not isinstance(hit_count, numbers.Integral)
            or isinstance(hit_count, bool)
            or hit_count < 0
            or hit_count > sample_count
        ):
            raise ValueError("evaluation receipt {} hits are invalid".format(name))
        values[name] = float(hit_count) / float(sample_count)
    values["mask_miou"] = mask_miou
    return _validate_retention_metric_values(values)


def _load_checkpoint_retention_manifest(path):
    if not os.path.exists(path):
        return {
            "schema": CHECKPOINT_RETENTION_SCHEMA,
            "latest_epoch": None,
            "records": {},
            "best": {},
        }
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != CHECKPOINT_RETENTION_SCHEMA
        or not isinstance(manifest.get("records"), dict)
    ):
        raise ValueError("checkpoint retention manifest is incompatible")
    normalized_records = {}
    for epoch_text, values in manifest["records"].items():
        if not isinstance(epoch_text, str) or re.fullmatch(r"\d+", epoch_text) is None:
            raise ValueError("checkpoint retention manifest has an invalid epoch")
        normalized_records[epoch_text] = _validate_retention_metric_values(values)
    latest_epoch = manifest.get("latest_epoch")
    if latest_epoch is not None and (
        not isinstance(latest_epoch, numbers.Integral)
        or isinstance(latest_epoch, bool)
        or latest_epoch < 0
    ):
        raise ValueError("checkpoint retention manifest latest_epoch is invalid")
    return {
        "schema": CHECKPOINT_RETENTION_SCHEMA,
        "latest_epoch": latest_epoch,
        "records": normalized_records,
        "best": {},
    }


def _normalize_checkpoint_retention_metrics(retained_metrics):
    if retained_metrics is None:
        return CHECKPOINT_RETENTION_METRICS
    if (
        not isinstance(retained_metrics, (list, tuple))
        or isinstance(retained_metrics, str)
        or not retained_metrics
    ):
        raise ValueError(
            "checkpoint retained metrics must be a non-empty list or tuple"
        )
    normalized = tuple(retained_metrics)
    if len(set(normalized)) != len(normalized):
        raise ValueError("checkpoint retained metrics contain duplicates")
    unsupported = set(normalized) - set(CHECKPOINT_RETENTION_METRICS)
    if unsupported:
        raise ValueError(
            "unsupported checkpoint retained metrics: {}".format(
                ", ".join(sorted(unsupported))
            )
        )
    return normalized


def update_checkpoint_retention(
    log_dir, epoch, metrics=None, retained_metrics=None
):
    """Retain the latest checkpoint and the requested independent bests."""
    if (
        not isinstance(epoch, numbers.Integral)
        or isinstance(epoch, bool)
        or epoch < 0
    ):
        raise ValueError("checkpoint retention epoch must be non-negative")
    epoch = int(epoch)
    retained_metrics = _normalize_checkpoint_retention_metrics(
        retained_metrics
    )
    checkpoint_path = os.path.join(log_dir, "ckpt_epoch_{}.pth".format(epoch))
    if not os.path.isfile(checkpoint_path):
        raise ValueError(
            "checkpoint retention cannot find {}".format(checkpoint_path)
        )
    manifest_path = os.path.join(log_dir, "checkpoint_retention.json")
    manifest = _load_checkpoint_retention_manifest(manifest_path)
    previous_latest = manifest["latest_epoch"]
    if previous_latest is not None and epoch < previous_latest:
        raise ValueError(
            "checkpoint retention cannot move latest epoch backwards"
        )
    if metrics is not None:
        manifest["records"][str(epoch)] = extract_checkpoint_retention_metrics(
            metrics
        )
    manifest["latest_epoch"] = epoch

    best = {}
    for metric_name in retained_metrics:
        candidates = [
            (values[metric_name], int(epoch_text))
            for epoch_text, values in manifest["records"].items()
        ]
        if candidates:
            value, best_epoch = max(
                candidates,
                key=lambda candidate: (candidate[0], -candidate[1]),
            )
            best[metric_name] = {
                "epoch": best_epoch,
                "value": value,
            }
    manifest["best"] = best

    _atomic_checkpoint_hardlink(
        checkpoint_path,
        os.path.join(log_dir, "ckpt_epoch_last.pth"),
    )
    for metric_name, record in best.items():
        source = os.path.join(
            log_dir, "ckpt_epoch_{}.pth".format(record["epoch"])
        )
        if not os.path.isfile(source):
            raise ValueError(
                "best {} checkpoint is missing: {}".format(metric_name, source)
            )
        _atomic_checkpoint_hardlink(
            source,
            os.path.join(log_dir, "ckpt_best_{}.pth".format(metric_name)),
        )
    _atomic_json_save(manifest, manifest_path)
    for metric_name in CHECKPOINT_RETENTION_METRICS:
        if metric_name in retained_metrics:
            continue
        stale_alias = os.path.join(
            log_dir, "ckpt_best_{}.pth".format(metric_name)
        )
        if os.path.exists(stale_alias):
            os.unlink(stale_alias)
    _fsync_directory(log_dir)
    keep_epochs = {epoch}
    keep_epochs.update(record["epoch"] for record in best.values())
    removed = []
    for filename in os.listdir(log_dir):
        match = re.fullmatch(r"ckpt_epoch_(\d+)\.pth", filename)
        if match is None:
            continue
        checkpoint_epoch = int(match.group(1))
        if checkpoint_epoch not in keep_epochs:
            path = os.path.join(log_dir, filename)
            os.unlink(path)
            removed.append(path)
    return {
        "manifest": manifest,
        "removed": sorted(removed),
    }


# BRIEF save model.
def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    """Save checkpoint if requested."""
    if save_cur or epoch % args.save_freq == 0:
        state = {
            'config': args,
            'save_path': '',
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        
        spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        state['save_path'] = spath
        _atomic_torch_save(state, spath)
        print("Saved in {}".format(spath))
        return spath
    else:
        print("not saving checkpoint")
        return None


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    # logger.
    def __init__(self, args):
        """Initialize."""
        name = args.log_dir.split('/')[-1]  # log_dir: './logs/eda', name: eda
        
        # Create log dir
        args.log_dir = os.path.join(
            args.log_dir,
            ','.join(args.dataset),
            args.exp,
            f'{int(time.time())}'
        )
        os.makedirs(args.log_dir, exist_ok=True)

        # Create logger
        self.logger = setup_logger(
            output=args.log_dir, distributed_rank=dist.get_rank(),
            name=name
        )

        # tensorboard
        self.tensorboard = record_tensorboard.TensorBoard(args.log_dir, distributed_rank=dist.get_rank())

        # Save config file and initialize tb writer
        if dist.get_rank() == 0:
            path = os.path.join(args.log_dir, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset


    # BRIEF dataloader.
    def get_loaders(self, args):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            torch.set_num_threads(1)
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        # Datasets
        train_only_audit = is_counterfactual_parent_bounded_audit(args)
        train_dataset, test_dataset = self.get_datasets(args)
        if train_only_audit:
            if args.eval or train_dataset is None or test_dataset is not None:
                raise ValueError(
                    "counterfactual Parent bounded audit must construct only "
                    "the training dataset"
                )
        
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)

        if args.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if args.dataloader_prefetch_factor <= 0:
            raise ValueError("dataloader_prefetch_factor must be positive")
        multiprocessing_loader_args = {}
        if args.num_workers > 0:
            multiprocessing_loader_args["prefetch_factor"] = (
                args.dataloader_prefetch_factor
            )

        if args.eval:
            train_loader = None
        else:
            hard_replay_manifest = str(getattr(
                args, "hard_example_replay_manifest", ""
            ) or "")
            hard_replay_sha256 = str(getattr(
                args, "hard_example_replay_manifest_sha256", ""
            ) or "")
            if bool(hard_replay_manifest) != bool(hard_replay_sha256):
                raise ValueError(
                    "hard-example replay manifest and SHA-256 must be "
                    "provided together"
                )
            if hard_replay_manifest:
                from models.hard_example_replay import (
                    HardExampleReplayDistributedSampler,
                )
                train_sampler = HardExampleReplayDistributedSampler(
                    train_dataset,
                    hard_replay_manifest,
                    hard_replay_sha256,
                    batch_size=args.batch_size,
                    seed=args.rng_seed,
                )
                print(
                    "hard_example_replay_plan={}".format(
                        json.dumps(train_sampler.summary(), sort_keys=True)
                    )
                )
            else:
                train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,      # TODO 
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=not bool(getattr(
                    args, "fpr_scene_disjoint_audit", False
                )),
                generator=g,
                persistent_workers=(
                    args.persistent_train_workers and args.num_workers > 0
                ),
                collate_fn=(
                    joint_det_structured_collate
                    if _requires_joint_det_structured_collate(args)
                    else None
                ),
                **multiprocessing_loader_args
            )
        
        if train_only_audit:
            test_loader = None
        else:
            if test_dataset is None:
                raise ValueError("non-audit runs require a test dataset")
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                sampler=test_sampler,
                drop_last=False,
                generator=g,
                collate_fn=(
                    joint_det_structured_collate
                    if _requires_joint_det_structured_collate(args)
                    else None
                ),
                **multiprocessing_loader_args
            )
        return train_loader, test_loader

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion(args):
        """Get loss criterion for training."""
        losses = ['boxes', 'labels', 'masks']
        if args.use_contrastive_align:
            losses.append('contrastive_align')
        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss)
        set_criterion = SetCriterion(
            matcher=matcher,
            losses=losses, eos_coef=0.1, temperature=0.07
        )
        criterion = compute_hungarian_loss

        return criterion, set_criterion

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
        selector_only = getattr(args, "source_choice_selector_train_only", False)
        moe_only = getattr(args, "source_moe_train_only", False)
        gate_only = getattr(args, "source_moe_gate_train_only", False)
        query_mask_fusion_only = getattr(
            args, "query_mask_fusion_train_only", False
        )
        egqs_mask_refiner_only = getattr(
            args, "egqs_mask_refiner_train_only", False
        )
        joint_query_quality_only = getattr(
            args, "joint_query_quality_train_only", False
        )
        decoder_query_adapter_only = getattr(
            args, "decoder_query_adapter_train_only", False
        )
        sacr_score_refiner_only = getattr(
            args, "sacr_score_refiner_train_only", False
        )
        parent_relative_text_verifier_only = getattr(
            args, "parent_relative_text_verifier_train_only", False
        )
        gate_new_heads_only = getattr(
            args, "source_moe_gate_new_heads_only", False
        )
        gate_new_head_names = ()
        gate_extra_prefixes = ()
        if sum(bool(value) for value in (
                selector_only, moe_only, gate_only,
                query_mask_fusion_only, egqs_mask_refiner_only,
                joint_query_quality_only, decoder_query_adapter_only,
                sacr_score_refiner_only,
                parent_relative_text_verifier_only)) > 1:
            raise ValueError(
                "selector-only, source-MoE-only, gate-only, and query mask "
                "fusion-only/EGQS-only/joint-query-quality-only/SACR-score-"
                "only modes are "
                "mutually "
                "exclusive"
            )
        if moe_only and not getattr(args, "use_source_moe", False):
            raise ValueError("source_moe_train_only requires use_source_moe")
        if gate_only and not (
                getattr(args, "use_source_moe", False)
                and getattr(args, "source_moe_use_fallback_gate", False)):
            raise ValueError(
                "source_moe_gate_train_only requires the fallback gate"
            )
        if query_mask_fusion_only and not getattr(
                args, "use_query_mask_fusion_calibrator", False):
            raise ValueError(
                "query_mask_fusion_train_only requires the calibrator"
            )
        if egqs_mask_refiner_only and not getattr(
                args, "use_egqs_mask_refiner", False):
            raise ValueError(
                "egqs_mask_refiner_train_only requires the EGQS refiner"
            )
        if joint_query_quality_only and not getattr(
                args, "use_joint_query_quality_reranker", False):
            raise ValueError(
                "joint_query_quality_train_only requires the reranker"
            )
        if decoder_query_adapter_only and not getattr(
                args, "use_decoder_query_adapter", False):
            raise ValueError(
                "decoder_query_adapter_train_only requires the adapter"
            )
        if sacr_score_refiner_only and not getattr(
                args, "use_sacr_score_refiner", False):
            raise ValueError(
                "sacr_score_refiner_train_only requires the refiner"
            )
        if parent_relative_text_verifier_only and not getattr(
                args, "use_parent_relative_text_verifier", False):
            raise ValueError(
                "parent_relative_text_verifier_train_only requires the "
                "verifier"
            )
        counterfactual_parent_training = getattr(
            args,
            "parent_relative_text_verifier_counterfactual_training",
            False,
        )
        if counterfactual_parent_training and not (
                parent_relative_text_verifier_only
                and getattr(args, "use_parent_relative_text_verifier", False)):
            raise ValueError(
                "counterfactual Parent supervision requires parent-relative "
                "verifier train-only mode"
            )
        if (
                parent_relative_text_verifier_only
                and getattr(
                    args,
                    "parent_relative_text_verifier_detach_inputs",
                    False,
                )):
            raise ValueError(
                "parent-relative verifier train-only mode requires attached "
                "structured/SACR inputs"
            )
        if (
                getattr(args, "use_parent_relative_text_verifier", False)
                and not getattr(args, "eval", False)
                and not parent_relative_text_verifier_only):
            raise ValueError(
                "training the parent-relative text verifier requires "
                "parent_relative_text_verifier_train_only so Box, Mask, "
                "and Backbone remain frozen"
            )
        parent_relative_sacr = getattr(
            args, "sacr_score_use_parent_relative_abstention", False
        )
        relation_counterfactual_sacr = getattr(
            args, "sacr_score_use_relation_counterfactual", False
        )
        if parent_relative_sacr and relation_counterfactual_sacr:
            raise ValueError(
                "SACR deployment variants are mutually exclusive"
            )
        if parent_relative_sacr and not getattr(
                args, "use_sacr_score_refiner", False):
            raise ValueError(
                "parent-relative SACR requires use_sacr_score_refiner"
            )
        if parent_relative_sacr:
            if (
                    not isinstance(
                        args.sacr_score_parent_gate_hidden_dim, int
                    )
                    or isinstance(
                        args.sacr_score_parent_gate_hidden_dim, bool
                    )
                    or args.sacr_score_parent_gate_hidden_dim < 1):
                raise ValueError(
                    "sacr_score_parent_gate_hidden_dim must be positive"
                )
            sacr_scalars = (
                ("sacr_score_temperature", 0.0, None, False),
                ("sacr_score_max_delta", 0.0, 0.25, False),
                ("sacr_score_min_box_advantage", 0.0, None, True),
                ("sacr_score_promotion_margin", 0.0, None, True),
                ("sacr_score_mask_tolerance", 0.0, None, True),
                ("sacr_score_raw_margin", 0.0, None, True),
                ("sacr_score_dense_weight", 0.0, None, True),
                ("sacr_score_preserve_weight", 0.0, None, True),
                ("sacr_score_gate_weight", 0.0, None, True),
                ("sacr_score_saturation_weight", 0.0, None, True),
            )
            for name, lower, upper, inclusive_lower in sacr_scalars:
                value = getattr(args, name)
                valid = (
                    isinstance(value, (float, int))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and (
                        float(value) >= lower
                        if inclusive_lower else float(value) > lower
                    )
                    and (upper is None or float(value) <= upper)
                )
                if not valid:
                    raise ValueError(
                        "{} is invalid for parent-relative SACR".format(name)
                    )
            if args.sacr_score_promotion_margin >= args.sacr_score_max_delta:
                raise ValueError(
                    "sacr_score_promotion_margin must be below max_delta"
                )
        if relation_counterfactual_sacr:
            if not getattr(args, "use_sacr_score_refiner", False):
                raise ValueError(
                    "relation-counterfactual SACR requires the refiner"
                )
            for name in (
                    "sacr_counterfactual_parent_top_k",
                    "sacr_counterfactual_max_negatives"):
                value = getattr(args, name)
                if (
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 1):
                    raise ValueError("{} must be positive".format(name))
            for name, positive in (
                    ("sacr_counterfactual_target_tolerance", False),
                    ("sacr_counterfactual_attribute_tolerance", False),
                    ("sacr_counterfactual_geometry_threshold", False),
                    ("sacr_counterfactual_iou_gap", False),
                    ("sacr_counterfactual_correct_iou_threshold", False),
                    ("sacr_counterfactual_pair_margin", False),
                    ("sacr_counterfactual_relation_scale", True),
                    ("sacr_counterfactual_deployment_threshold", False)):
                value = getattr(args, name)
                if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        or float(value) < 0.0
                        or (positive and float(value) == 0.0)):
                    raise ValueError("{} is invalid".format(name))
        if gate_new_heads_only:
            if not gate_only:
                raise ValueError(
                    "source_moe_gate_new_heads_only requires gate-only training"
                )
            unwrapped = model.module if hasattr(model, "module") else model
            fallback_gate = getattr(
                getattr(unwrapped, "source_moe", None),
                "fallback_gate", None,
            )
            action_mode = (
                None if fallback_gate is None
                else getattr(fallback_gate, "action_mode", None)
            )
            objective = getattr(args, "source_moe_gate_objective", None)
            if action_mode == "cascade_absolute_quality_correction" and (
                    objective == "cascade_absolute_quality_calibrated"):
                gate_new_head_names = (
                    "absolute_quality_head",
                    "cascade_quality_adapter",
                    "cascade_correction_head",
                )
            elif action_mode == "cascade_opportunity_quality_correction" and (
                    objective == "cascade_opportunity_balanced_calibrated"):
                gate_new_head_names = (
                    "absolute_quality_head",
                    "cascade_quality_adapter",
                    "cascade_correction_head",
                    "cascade_opportunity_head",
                )
            elif action_mode == (
                    "cascade_opportunity_verified_correction") and (
                    objective == "cascade_opportunity_verified_calibrated"):
                gate_new_head_names = (
                    "absolute_quality_head",
                    "cascade_quality_adapter",
                    "cascade_correction_head",
                    "cascade_opportunity_head",
                    "cascade_candidate_safety_head",
                )
            elif action_mode == "cascade_joint_risk_correction" and (
                    objective == "cascade_joint_risk_calibrated"):
                gate_new_head_names = (
                    "absolute_quality_head",
                    "cascade_quality_adapter",
                    "cascade_correction_head",
                    "cascade_opportunity_head",
                    "cascade_candidate_safety_head",
                    "cascade_joint_action_head",
                )
            elif action_mode == "cascade_v19_fallback_set_correction" and (
                    objective
                    == "cascade_v19_fallback_set_risk_calibrated"):
                gate_new_head_names = (
                    "cascade_fallback_set_action_head",
                )
            elif action_mode == "cascade_v19_rich_set_correction" and (
                    objective == "cascade_v19_rich_set_empirical_risk"):
                gate_new_head_names = (
                    "cascade_rich_fallback_set_action_head",
                )
            elif action_mode == "cascade_v23_dense_quality_correction" and (
                    objective in (
                        "cascade_v23_dense_quality_risk",
                        "cascade_v27_uncertainty_quality_risk",
                    )):
                gate_new_head_names = (
                    "cascade_dense_quality_set_head",
                )
                gate_extra_prefixes = (
                    "source_moe.adaptive_source_mixer.",
                )
            elif action_mode == "cascade_v24_relative_risk_correction" and (
                    objective == "cascade_v24_relative_risk"):
                gate_new_head_names = (
                    "cascade_dense_quality_set_head",
                    "cascade_relative_risk_set_head",
                )
                gate_extra_prefixes = (
                    "source_moe.adaptive_source_mixer.",
                )
            elif action_mode == (
                    "cascade_v25_pairwise_calibrated_correction") and (
                    objective == "cascade_v25_pairwise_calibrated_risk"):
                gate_new_head_names = (
                    "cascade_dense_quality_set_head",
                    "cascade_pairwise_calibrated_set_head",
                )
                gate_extra_prefixes = (
                    "source_moe.adaptive_source_mixer.",
                )
            elif action_mode == (
                    "cascade_v26_prior_restored_pairwise_correction") and (
                    objective
                    == "cascade_v26_prior_restored_pairwise_risk"):
                gate_new_head_names = (
                    "cascade_dense_quality_set_head",
                    "cascade_pairwise_calibrated_set_head",
                )
                gate_extra_prefixes = (
                    "source_moe.adaptive_source_mixer.",
                )
            elif action_mode == (
                    "cascade_v28_selected_abstention_correction") and (
                    objective
                    == "cascade_v28_selected_abstention_risk"):
                gate_new_head_names = (
                    "cascade_dense_quality_set_head",
                    "cascade_pairwise_calibrated_set_head",
                    "cascade_selected_abstention_head",
                )
                gate_extra_prefixes = (
                    "source_moe.adaptive_source_mixer.",
                )
            elif action_mode == (
                    "cascade_v29_counterfactual_selected_correction") and (
                    objective
                    == "cascade_v29_counterfactual_selected_risk"):
                gate_new_head_names = (
                    "cascade_dense_quality_set_head",
                    "cascade_pairwise_calibrated_set_head",
                    "cascade_counterfactual_selected_risk_head",
                )
                gate_extra_prefixes = (
                    "source_moe.adaptive_source_mixer.",
                )
            elif action_mode == (
                    "cascade_v37_counterfactual_benefit_hazard_correction") and (
                    objective
                    == "cascade_v37_counterfactual_benefit_hazard_risk"):
                gate_new_head_names = (
                    "cascade_dense_quality_set_head",
                    "cascade_pairwise_calibrated_set_head",
                    "cascade_counterfactual_benefit_hazard_head",
                )
                gate_extra_prefixes = (
                    "source_moe.adaptive_source_mixer.",
                )
            elif action_mode == (
                    "cascade_v38_complementary_logodds_correction") and (
                    objective == "cascade_v38_complementary_logodds_risk"):
                gate_new_head_names = (
                    "cascade_dense_quality_set_head",
                    "cascade_pairwise_calibrated_set_head",
                    "cascade_counterfactual_logodds_head",
                )
                gate_extra_prefixes = (
                    "source_moe.adaptive_source_mixer.",
                )
            elif action_mode == (
                    "cascade_v39_hazard_residual_correction") and (
                    objective == "cascade_v39_hazard_residual_risk"):
                gate_new_head_names = (
                    "cascade_dense_quality_set_head",
                    "cascade_pairwise_calibrated_set_head",
                    "cascade_counterfactual_hazard_residual_head",
                )
                gate_extra_prefixes = (
                    "source_moe.adaptive_source_mixer.",
                )
            else:
                raise ValueError(
                    "source_moe_gate_new_heads_only requires a matching "
                    "cascade action and calibrated objective"
                )
        if (selector_only or moe_only or gate_only
                or query_mask_fusion_only or egqs_mask_refiner_only
                or joint_query_quality_only or decoder_query_adapter_only
                or sacr_score_refiner_only
                or parent_relative_text_verifier_only):
            if gate_only:
                if gate_new_heads_only:
                    trainable_prefixes = tuple(
                        "source_moe.fallback_gate.{}.".format(name)
                        for name in gate_new_head_names
                    ) + gate_extra_prefixes
                    trainable_prefix = ",".join(trainable_prefixes)
                else:
                    trainable_prefixes = ("source_moe.fallback_gate",)
                    trainable_prefix = trainable_prefixes[0]
                learning_rate = args.source_moe_gate_lr
            elif moe_only:
                trainable_prefixes = ("source_moe",)
                trainable_prefix = "source_moe"
                learning_rate = args.source_moe_lr
            elif query_mask_fusion_only:
                trainable_prefixes = ("query_mask_fusion_calibrator",)
                trainable_prefix = "query_mask_fusion_calibrator"
                learning_rate = args.query_mask_fusion_lr
            elif egqs_mask_refiner_only:
                trainable_prefixes = ("egqs_mask_refiner",)
                trainable_prefix = "egqs_mask_refiner"
                learning_rate = args.egqs_mask_refiner_lr
            elif joint_query_quality_only:
                trainable_prefixes = ("joint_query_quality_reranker",)
                if getattr(args, "use_sacr_source", False):
                    trainable_prefixes += (
                        "structured_slot_builder",
                        "sacr_head",
                        "sacr_residual_scale",
                    )
                trainable_prefix = ",".join(trainable_prefixes)
                learning_rate = args.joint_query_quality_lr
            elif decoder_query_adapter_only:
                trainable_prefixes = ("decoder_query_adapter",)
                trainable_prefix = trainable_prefixes[0]
                learning_rate = args.decoder_query_adapter_lr
            elif sacr_score_refiner_only:
                if getattr(
                        args,
                        "sacr_score_use_relation_counterfactual",
                        False):
                    trainable_prefixes = (
                        RELATION_COUNTERFACTUAL_TRAINABLE_PREFIXES
                    )
                elif getattr(
                        args,
                        "sacr_score_use_parent_relative_abstention",
                        False):
                    trainable_prefixes = (
                        "structured_slot_builder",
                        "sacr_head",
                        "sacr_parent_relative_gate",
                    )
                else:
                    trainable_prefixes = (
                        "structured_slot_builder",
                        "sacr_head",
                        "sacr_score_gate",
                    )
                trainable_prefix = ",".join(trainable_prefixes)
                learning_rate = args.sacr_score_refiner_lr
            elif parent_relative_text_verifier_only:
                trainable_prefixes = (
                    "structured_slot_builder",
                    "sacr_head",
                    "parent_relative_text_verifier",
                )
                trainable_prefix = ",".join(trainable_prefixes)
                learning_rate = args.parent_relative_text_verifier_lr
            else:
                trainable_prefixes = ("source_choice_selector",)
                trainable_prefix = "source_choice_selector"
                learning_rate = args.source_choice_selector_lr
            print("-------------------------------{}-only training------------------------------------".format(trainable_prefix))
            trainable = 0
            for n, p in model.named_parameters():
                canonical_name = n[7:] if n.startswith("module.") else n
                p.requires_grad = any(
                    canonical_name.startswith(prefix)
                    for prefix in trainable_prefixes
                )
                if (p.requires_grad
                        and joint_query_quality_only
                        and (getattr(
                            args,
                            "joint_query_quality_use_parent_transition_advantage",
                            False,
                        ) or getattr(
                            args,
                            "joint_query_quality_use_decomposed_transition_advantage",
                            False,
                        ) or getattr(
                            args,
                            "joint_query_quality_use_setwise_tier_advantage",
                            False,
                        ) or getattr(
                            args,
                            "joint_query_quality_use_factorized_hit_advantage",
                            False,
                        ))
                        and canonical_name.startswith((
                            "joint_query_quality_reranker.quality_head.",
                            "joint_query_quality_reranker.residual_head.",
                        ))):
                    p.requires_grad = False
                if p.requires_grad:
                    trainable += p.numel()
            print(
                "{}_train_only: trainable parameters {}".format(
                    trainable_prefix, trainable
                )
            )
            if trainable == 0:
                raise ValueError("train-only mode found no matching parameters")
            param_dicts = [
                {
                    "params": [
                        p for p in model.parameters() if p.requires_grad
                    ],
                    "lr": learning_rate,
                },
                {
                    "params": [],
                    "lr": args.lr_backbone
                },
                {
                    "params": [],
                    "lr": args.text_encoder_lr
                }
            ]
        elif args.frozen:
            print("-------------------------------frozen EDA parameters------------------------------------")
            for n, p in model.named_parameters():
                if "x_mask" not in n and "x_query" not in n and "seed_decoder" not in n:
                    p.requires_grad = False
            param_dicts = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "x_mask" in n or "x_query" in n or "seed_decoder" in n
                    ]
                },
                {
                    "params": [],
                    "lr": args.lr_backbone
                },
                {
                    "params": [],
                    "lr": args.text_encoder_lr
                }
            ]
        elif args.small_lr:
            param_dicts = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "x_mask" in n or "x_query" in n or "seed_decoder" in n
                    ],
                    "lr": args.lr
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" not in n and "text_encoder" not in n 
                        and "x_mask" not in n and "x_query" not in n and "seed_decoder" not in n
                        and p.requires_grad
                    ],
                    "lr": args.lr * 0.01
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" in n and p.requires_grad
                    ],
                    "lr": args.lr_backbone * 0.01
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "text_encoder" in n and p.requires_grad
                    ],
                    "lr": args.text_encoder_lr * 0.01
                }
            ]
        elif args.use_source_choice_selector or getattr(
                args, 'use_source_moe', False):
            param_dicts = build_mcln_optimizer_param_groups(
                model,
                decoder_lr=args.lr,
                backbone_lr=args.lr_backbone,
                selector_lr=(
                    args.source_moe_lr
                    if getattr(args, 'use_source_moe', False)
                    else args.source_choice_selector_lr
                ),
                mask_head_lr_multiplier=args.mask_head_lr_multiplier,
                require_selector=True,
            )
        else:
            param_dicts = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" not in n and "text_encoder" not in n
                        and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" in n and p.requires_grad
                    ],
                    "lr": args.lr_backbone
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "text_encoder" in n and p.requires_grad
                    ],
                    "lr": args.text_encoder_lr
                }
            ]
        optimizer = optim.AdamW(param_dicts,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
        return optimizer


    def _save_density_scene_audit_role_receipt(
            self, args, epoch, metrics, training_receipt):
        role = getattr(
            args, "density_aware_target_box_scene_disjoint_role", None
        )
        if role not in ("parent", "control", "method"):
            raise ValueError("density scene audit role is invalid")
        if not isinstance(metrics, dict):
            raise ValueError("density scene audit metrics are missing")
        audit_metrics = metrics.get(
            "density_aware_target_box_scene_audit"
        )
        if not isinstance(audit_metrics, dict):
            raise ValueError("density scene audit metric payload is missing")
        split_metadata = getattr(
            args,
            "density_aware_target_box_scene_disjoint_split_metadata",
            None,
        )
        if not isinstance(split_metadata, dict):
            raise ValueError("density scene audit split metadata is missing")
        generated_weights = []
        for root, _directories, filenames in os.walk(args.log_dir):
            for filename in filenames:
                if filename.endswith(".pth"):
                    generated_weights.append(os.path.join(root, filename))
        if generated_weights:
            raise ValueError("density scene audit generated a checkpoint")
        receipt = {
            "schema": DENSITY_TARGET_BOX_SCENE_AUDIT_SCHEMA,
            "role": role,
            "epoch": int(epoch),
            "checkpoint_path": args.checkpoint_path,
            "checkpoint_sha256": getattr(
                args,
                "density_aware_target_box_consumed_checkpoint_sha256",
                None,
            ),
            "checkpoint_epoch": getattr(
                args,
                "density_aware_target_box_consumed_checkpoint_epoch",
                None,
            ),
            "density_aware_target_box_loss_weight": float(getattr(
                args, "density_aware_target_box_loss_weight", 0.0
            )),
            "split": split_metadata,
            "training": training_receipt,
            "evaluation": metrics,
            "generated_weights": generated_weights,
            "formal_validation_accessed": False,
            "audit_only": True,
            "long_training_authorized": False,
        }
        if dist.get_rank() == 0:
            path = os.path.join(
                args.log_dir,
                "density_target_box_scene_audit_{}_epoch_{}.json".format(
                    role, int(epoch)
                ),
            )
            _atomic_json_save(receipt, path)
            self.logger.info(
                "Density scene audit role {} completed; receipt={}".format(
                    role, path
                )
            )
        if dist.is_initialized():
            dist.barrier()


    # BRIEF main training/testing
    def main(self, args):
        """Run main training/testing pipeline."""
        max_train_batches = getattr(args, "max_train_batches", 0)
        if (
                not isinstance(max_train_batches, int)
                or isinstance(max_train_batches, bool)
                or not 0 <= max_train_batches <= 10000):
            raise ValueError("max_train_batches must be in [0, 10000]")
        gradient_accumulation_steps = getattr(
            args, "gradient_accumulation_steps", 1
        )
        if (
                not isinstance(gradient_accumulation_steps, int)
                or isinstance(gradient_accumulation_steps, bool)
                or not 1 <= gradient_accumulation_steps <= 64):
            raise ValueError(
                "gradient_accumulation_steps must be in [1, 64]"
            )
        drop_incomplete_accumulation_group = bool(getattr(
            args, "drop_incomplete_accumulation_group", False
        ))
        fpr_scene_audit = bool(getattr(
            args, "fpr_scene_disjoint_audit", False
        ))
        density_scene_audit = bool(getattr(
            args, "density_aware_target_box_scene_disjoint_audit", False
        ))
        counterfactual_parent_audit = bool(getattr(
            args,
            "parent_relative_text_verifier_counterfactual_training",
            False,
        )) and max_train_batches > 0
        if fpr_scene_audit and density_scene_audit:
            raise ValueError("FPR and density scene audits are mutually exclusive")
        if counterfactual_parent_audit:
            if fpr_scene_audit or density_scene_audit:
                raise ValueError(
                    "counterfactual Parent audit is a standalone bounded audit"
                )
            if (
                    args.eval
                    or max_train_batches != 100
                    or args.batch_size != 16
                    or gradient_accumulation_steps != 1
                    or drop_incomplete_accumulation_group
                    or args.start_epoch != 58
                    or args.max_epoch != 58
                    or getattr(args, "checkpoint_metric_retention", False)
                    or list(args.dataset) != ["nr3d"]
                    or args.test_dataset != "nr3d"
                    or not args.joint_det
                    or not args.butd_cls
                    or args.butd
                    or args.butd_gt
                    or not getattr(
                        args, "use_parent_relative_text_verifier", False
                    )
                    or not getattr(
                        args,
                        "parent_relative_text_verifier_train_only",
                        False,
                    )):
                raise ValueError(
                    "counterfactual Parent audit requires Nr3D E58, "
                    "B16 x A1, exact 100 batches, and verifier-only training"
                )
            if dist.is_initialized() and dist.get_world_size() != 1:
                raise ValueError(
                    "counterfactual Parent audit requires one formal rank"
                )
        if fpr_scene_audit:
            if args.eval or max_train_batches != 0:
                raise ValueError(
                    "FPR scene audit requires full train-only epoch mode"
                )
            if (
                    not getattr(
                        args, "use_parent_relative_text_verifier", False
                    )
                    or not getattr(
                        args,
                        "parent_relative_text_verifier_train_only",
                        False,
                    )):
                raise ValueError(
                    "FPR scene audit requires verifier train-only mode"
                )
            if (
                    getattr(args, "use_source_moe", False)
                    or getattr(args, "use_sacr_source", False)
                    or getattr(args, "use_sacr_score_refiner", False)):
                raise ValueError(
                    "FPR scene audit rejects alternative deployment branches"
                )
            if (
                    args.batch_size != 16
                    or gradient_accumulation_steps != 1
                    or drop_incomplete_accumulation_group):
                raise ValueError("FPR scene audit requires exact B16 x A1")
            if (
                    args.max_epoch != 58 or args.val_freq != 1
                    or getattr(args, "checkpoint_metric_retention", False)):
                raise ValueError(
                    "FPR scene audit requires only E58 and no retention"
                )
            if dist.is_initialized() and dist.get_world_size() != 1:
                raise ValueError("FPR scene audit requires one formal rank")
            args.fpr_scene_disjoint_config_receipt = (
                build_fpr_scene_disjoint_config_receipt(args)
            )
        if density_scene_audit:
            role = getattr(
                args, "density_aware_target_box_scene_disjoint_role", None
            )
            if role not in ("parent", "control", "method"):
                raise ValueError("density scene audit role is invalid")
            expected_weight = 1.0 if role == "method" else 0.0
            observed_weight = float(getattr(
                args, "density_aware_target_box_loss_weight", 0.0
            ))
            if observed_weight != expected_weight:
                raise ValueError(
                    "density scene audit role {} requires weight {}".format(
                        role, expected_weight
                    )
                )
            expected_max_batches = 0 if role == "parent" else 100
            if max_train_batches != expected_max_batches:
                raise ValueError(
                    "density scene audit role {} requires max_train_batches={}"
                    .format(role, expected_max_batches)
                )
            expected_counts = {
                "density_aware_target_box_scene_disjoint_fold": 2,
                "density_aware_target_box_scene_disjoint_expected_fit_scenes": 408,
                "density_aware_target_box_scene_disjoint_expected_holdout_scenes": 103,
                "density_aware_target_box_scene_disjoint_expected_fit_samples": 26590,
                "density_aware_target_box_scene_disjoint_expected_holdout_samples": 6329,
            }
            for name, expected in expected_counts.items():
                if getattr(args, name, None) != expected:
                    raise ValueError(
                        "density scene audit {} drifted".format(name)
                    )
            if (
                    args.eval
                    or args.batch_size != 16
                    or gradient_accumulation_steps != 1
                    or drop_incomplete_accumulation_group
                    or args.max_epoch != 58
                    or args.val_freq != 1
                    or getattr(args, "checkpoint_metric_retention", False)
                    or list(args.dataset) != ["nr3d"]
                    or args.test_dataset != "nr3d"
                    or not args.joint_det
                    or not args.butd_cls
                    or args.butd
                    or args.butd_gt):
                raise ValueError("density scene audit core contract drifted")
            if str(getattr(
                    args, "density_aware_target_box_checkpoint_sha256", ""
            ) or "").lower() != DENSITY_TARGET_BOX_SCENE_AUDIT_E57_SHA256:
                raise ValueError("density scene audit checkpoint identity drifted")
            incompatible = (
                "use_source_moe",
                "use_sacr_source",
                "use_sacr_score_refiner",
                "use_parent_relative_text_verifier",
                "relation_counterfactual_aux_conservative_anchor_set",
                "eval_use_rec_reranker_scores",
                "eval_use_rec_geometry_reranker_scores",
                "eval_use_rec_joint_box_mask",
                "eval_use_rec_selective_residual_scores",
                "eval_use_rec_hierarchical_reranker_scores",
            )
            active = [name for name in incompatible if bool(getattr(
                args, name, False
            ))]
            if active:
                raise ValueError(
                    "density scene audit rejects branches: {}".format(
                        ", ".join(active)
                    )
                )
            if dist.is_initialized() and dist.get_world_size() != 1:
                raise ValueError("density scene audit requires one formal rank")
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        if not args.eval:
            n_data = len(train_loader.dataset)
            self.logger.info(f"length of training dataset: {n_data}")
            accumulation_plan = _gradient_accumulation_plan(
                len(train_loader),
                max_train_batches,
                gradient_accumulation_steps,
                drop_incomplete_accumulation_group,
            )
            optimizer_steps_per_epoch = accumulation_plan[
                "optimizer_step_count"
            ]
            self.logger.info(
                "gradient accumulation: {} micro-batches/step; "
                "{} requested, {} effective, {} dropped; "
                "{} optimizer steps per epoch".format(
                    gradient_accumulation_steps,
                    accumulation_plan["requested_batch_count"],
                    accumulation_plan["effective_batch_count"],
                    accumulation_plan["dropped_batch_count"],
                    optimizer_steps_per_epoch,
                )
            )
        n_data = _optional_test_dataset_size(args, test_loader)
        if n_data is None:
            self.logger.info(
                "testing dataset disabled for counterfactual Parent bounded "
                "audit"
            )
        else:
            self.logger.info(f"length of testing dataset: {n_data}")

        # Get model
        model = self.get_model(args)

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler
        if not args.eval:
            scheduler = get_scheduler(
                optimizer, optimizer_steps_per_epoch, args
            )
        else:
            scheduler = None
        
        # Move model to devices
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                # synBN
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
            else:
                model = model.cuda()

        # note Distributed Data-Parallel Training (DDP)
        find_unused_parameters = not (
            getattr(args, "query_mask_fusion_train_only", False)
            or getattr(args, "egqs_mask_refiner_train_only", False)
            or getattr(args, "joint_query_quality_train_only", False)
            or getattr(args, "sacr_score_refiner_train_only", False)
            or getattr(
                args, "parent_relative_text_verifier_train_only", False
            )
        )
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters
        )

        # Check for a checkpoint
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(
                args, model, optimizer, scheduler,
                optimizer_steps_per_epoch=(
                    None if args.eval else optimizer_steps_per_epoch
                ),
            )
        fpr_audit_sentinel_batch = None
        fpr_audit_before_state = None
        fpr_audit_before_outputs = None
        counterfactual_audit_before_state = None
        self._counterfactual_audit_sentinel_batch = None
        self._counterfactual_audit_before_outputs = None
        if counterfactual_parent_audit:
            counterfactual_audit_before_state = (
                capture_fpr_audit_model_state(model)
            )
        if fpr_scene_audit:
            if args.start_epoch != 58:
                raise ValueError(
                    "FPR scene audit must resume E57 into exact E58"
                )
            try:
                fpr_audit_sentinel_batch = next(iter(test_loader))
            except StopIteration:
                raise ValueError("FPR scene audit holdout loader is empty")
            fpr_audit_before_outputs = self._capture_fpr_audit_sentinel(
                model, copy.deepcopy(fpr_audit_sentinel_batch)
            )
            fpr_audit_before_state = capture_fpr_audit_model_state(model)
        if density_scene_audit:
            if args.start_epoch != 58:
                raise ValueError(
                    "density scene audit must resume E57 into exact E58"
                )
            if getattr(
                    args,
                    "density_aware_target_box_consumed_checkpoint_sha256",
                    None,
            ) != DENSITY_TARGET_BOX_SCENE_AUDIT_E57_SHA256:
                raise ValueError("density scene audit consumed checkpoint drifted")
            if getattr(
                    args,
                    "density_aware_target_box_consumed_checkpoint_epoch",
                    None,
            ) != 57:
                raise ValueError("density scene audit consumed epoch drifted")
            if getattr(
                    args,
                    "density_aware_target_box_scene_disjoint_role",
                    None,
            ) == "parent":
                metrics = self.evaluate_one_epoch(
                    57, test_loader, model, criterion, set_criterion, args
                )
                self._save_density_scene_audit_role_receipt(
                    args, 57, metrics, None
                )
                return
        
        # ##############################################
        # NOTE [eval-only] Just eval and end execution #
        # ##############################################
        if args.eval:
            print("Testing evaluation.....................")
            metrics = self.evaluate_one_epoch(
                args.start_epoch, test_loader,
                model, criterion, set_criterion, args
            )
            if dist.get_rank() == 0:
                save_eval_metrics_receipt(
                    args.log_dir, args.start_epoch, metrics
                )
            return

        # ##############################
        # NOTE Training and Validation #
        # ##############################
        last_eval_epoch = None
        metric_retention = getattr(
            args, "checkpoint_metric_retention", False
        )
        retained_metrics = getattr(
            args,
            "checkpoint_retention_metrics",
            CHECKPOINT_RETENTION_METRICS,
        )
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()

            # train *
            train_receipt = self.train_one_epoch(
                epoch, train_loader, model,
                criterion, set_criterion,
                optimizer, scheduler, args
            )

            if counterfactual_parent_audit:
                sentinel_batch = self._counterfactual_audit_sentinel_batch
                before_outputs = self._counterfactual_audit_before_outputs
                if sentinel_batch is None or before_outputs is None:
                    raise ValueError(
                        "counterfactual Parent audit sentinel is missing"
                    )
                after_outputs = self._capture_fpr_audit_sentinel(
                    model, copy.deepcopy(sentinel_batch)
                )
                after_state = capture_fpr_audit_model_state(model)
                state_integrity = {
                    "before": counterfactual_audit_before_state,
                    "after": after_state,
                    "frozen_exact": (
                        counterfactual_audit_before_state["frozen"]["sha256"]
                        == after_state["frozen"]["sha256"]
                    ),
                    "trainable_changed": (
                        counterfactual_audit_before_state["trainable"][
                            "sha256"
                        ] != after_state["trainable"]["sha256"]
                    ),
                }
                output_integrity = {
                    "before": before_outputs,
                    "after": after_outputs,
                    "exact": (
                        before_outputs["combined_sha256"]
                        == after_outputs["combined_sha256"]
                    ),
                }
                if not state_integrity["frozen_exact"]:
                    raise ValueError(
                        "counterfactual Parent audit changed frozen state"
                    )
                if not state_integrity["trainable_changed"]:
                    raise ValueError(
                        "counterfactual Parent audit did not update verifier"
                    )
                if not output_integrity["exact"]:
                    raise ValueError(
                        "counterfactual Parent audit changed frozen outputs"
                    )
                train_receipt["state_integrity"] = state_integrity
                train_receipt["output_integrity"] = output_integrity

            if fpr_scene_audit:
                split_metadata = getattr(
                    args, "fpr_scene_disjoint_split_metadata", None
                )
                if not isinstance(split_metadata, dict):
                    raise ValueError("FPR scene audit split metadata is missing")
                expected_fit_samples = split_metadata.get("fit_samples")
                expected_fit_batches = math.ceil(
                    expected_fit_samples / float(args.batch_size)
                )
                if (
                        train_receipt.get("sample_count")
                        != expected_fit_samples
                        or train_receipt.get("batch_count")
                        != expected_fit_batches
                        or train_receipt.get("sample_identity_count")
                        != expected_fit_samples
                        or train_receipt.get("sample_identity_unique_count")
                        != expected_fit_samples
                        or train_receipt.get("sample_identity_sha256")
                        != split_metadata.get(
                            "fit_sample_identity_sha256"
                        )):
                    raise ValueError(
                        "FPR scene audit did not consume its complete fit "
                        "partition"
                    )
                fpr_audit_after_outputs = self._capture_fpr_audit_sentinel(
                    model, copy.deepcopy(fpr_audit_sentinel_batch)
                )
                fpr_audit_after_state = capture_fpr_audit_model_state(model)
                metrics = self.evaluate_one_epoch(
                    epoch, test_loader, model, criterion, set_criterion, args
                )
                if not isinstance(metrics, dict):
                    raise ValueError(
                        "FPR scene audit did not produce evaluator metrics"
                    )
                decision = metrics.get(
                    "parent_relative_text_verifier_scene_audit"
                )
                if not isinstance(decision, dict):
                    raise ValueError(
                        "FPR scene audit decision diagnostics are missing"
                    )
                state_integrity = {
                    "before": fpr_audit_before_state,
                    "after": fpr_audit_after_state,
                    "frozen_exact": (
                        fpr_audit_before_state["frozen"]["sha256"]
                        == fpr_audit_after_state["frozen"]["sha256"]
                    ),
                    "trainable_changed": (
                        fpr_audit_before_state["trainable"]["sha256"]
                        != fpr_audit_after_state["trainable"]["sha256"]
                    ),
                }
                output_integrity = {
                    "before": fpr_audit_before_outputs,
                    "after": fpr_audit_after_outputs,
                    "exact": (
                        fpr_audit_before_outputs["combined_sha256"]
                        == fpr_audit_after_outputs["combined_sha256"]
                    ),
                }
                failures = []
                if not state_integrity["frozen_exact"]:
                    failures.append("frozen_model_state_drift")
                if not state_integrity["trainable_changed"]:
                    failures.append("trainable_state_unchanged")
                if not output_integrity["exact"]:
                    failures.append("box_mask_parent_output_drift")
                if decision["switch_count"] <= 0:
                    failures.append("no_heldout_switch")
                threshold025 = decision["thresholds"]["025"]
                threshold050 = decision["thresholds"]["050"]
                if threshold025["fix_count"] <= threshold025["break_count"]:
                    failures.append("acc025_fix_not_greater_than_break")
                if threshold050["fix_count"] < threshold050["break_count"]:
                    failures.append("acc050_net_negative")
                generated_weights = []
                for root, _dirs, files in os.walk(args.log_dir):
                    for filename in files:
                        if filename.endswith(".pth"):
                            generated_weights.append(
                                os.path.join(root, filename)
                            )
                if generated_weights:
                    failures.append("unexpected_checkpoint_output")
                av4_scene_audit = bool(getattr(
                    args, "fpr_scene_disjoint_av4_audit", False
                ))
                receipt = {
                    "schema": FPR_SCENE_AUDIT_SCHEMA,
                    "epoch": int(epoch),
                    "checkpoint_path": args.checkpoint_path,
                    "checkpoint_sha256": getattr(
                        args,
                        "fpr_scene_disjoint_consumed_checkpoint_sha256",
                        None,
                    ),
                    "checkpoint_epoch": getattr(
                        args,
                        "fpr_scene_disjoint_consumed_checkpoint_epoch",
                        None,
                    ),
                    "split": split_metadata,
                    "frozen_config": getattr(
                        args, "fpr_scene_disjoint_config_receipt", None
                    ),
                    "training": train_receipt,
                    "evaluation": metrics,
                    "state_integrity": state_integrity,
                    "output_integrity": output_integrity,
                    "generated_weights": generated_weights,
                    "fold_gate_pass": not failures,
                    "gate_failures": failures,
                    "next_stage": (
                        (
                            "independent_review_only"
                            if not failures else "method_sealed"
                        )
                        if av4_scene_audit else (
                            "await_all_five_folds"
                            if not failures else "method_correction_only"
                        )
                    ),
                    "long_training_authorized": False,
                }
                if av4_scene_audit:
                    receipt["audit_only"] = True
                    receipt["formal_validation_accessed"] = False
                if dist.get_rank() == 0:
                    save_eval_metrics_receipt(args.log_dir, epoch, metrics)
                    receipt_path = os.path.join(
                        args.log_dir,
                        "fpr_scene_disjoint_audit_fold_{}_epoch_{}.json"
                        .format(
                            getattr(args, "fpr_scene_disjoint_fold", -1),
                            epoch,
                        ),
                    )
                    _atomic_json_save(receipt, receipt_path)
                    self.logger.info(
                        "FPR scene-disjoint fold gate pass={} failures={}; "
                        "receipt={}".format(
                            not failures, failures, receipt_path
                        )
                    )
                if dist.is_initialized():
                    dist.barrier()
                return

            if density_scene_audit:
                role = getattr(
                    args,
                    "density_aware_target_box_scene_disjoint_role",
                    None,
                )
                if role not in ("control", "method"):
                    raise ValueError(
                        "only control/method roles may enter density training"
                    )
                if (
                        train_receipt.get("batch_count") != 100
                        or train_receipt.get("optimizer_step_count") != 100
                        or train_receipt.get("sample_count") != 1600
                        or train_receipt.get("sample_identity_count") != 1600
                        or train_receipt.get("sample_identity_unique_count")
                        != 1600):
                    raise ValueError(
                        "density scene audit did not consume exact 100xB16"
                    )
                metrics = self.evaluate_one_epoch(
                    epoch, test_loader, model, criterion, set_criterion, args
                )
                self._save_density_scene_audit_role_receipt(
                    args, epoch, metrics, train_receipt
                )
                return

            if max_train_batches > 0:
                train_receipt.update({
                    "epoch": int(epoch),
                    "max_train_batches": int(max_train_batches),
                    "checkpoint_path": args.checkpoint_path,
                })
                if counterfactual_parent_audit:
                    if (
                            train_receipt.get("batch_count") != 100
                            or train_receipt.get("optimizer_step_count") != 100
                            or train_receipt.get("sample_count") != 1600):
                        raise ValueError(
                            "counterfactual Parent audit did not consume "
                            "exact 100 x B16"
                        )
                    train_receipt.update({
                        "audit_only": True,
                        "formal_validation_accessed": False,
                        "long_training_authorized": False,
                    })
                if (
                        float(getattr(
                            args,
                            "density_aware_target_box_loss_weight",
                            0.0,
                        )) > 0.0):
                    train_receipt.update({
                        "checkpoint_sha256": getattr(
                            args,
                            "density_aware_target_box_consumed_checkpoint_sha256",
                            None,
                        ),
                        "checkpoint_epoch": getattr(
                            args,
                            "density_aware_target_box_consumed_checkpoint_epoch",
                            None,
                        ),
                    })
                if dist.get_rank() == 0:
                    receipt_path = os.path.join(
                        args.log_dir,
                        "train_audit_receipt_epoch_{}.json".format(epoch),
                    )
                    _atomic_json_save(train_receipt, receipt_path)
                    self.logger.info(
                        "Bounded train audit completed; receipt saved in {}"
                        .format(receipt_path)
                    )
                if dist.is_initialized():
                    dist.barrier()
                return
            
            # log
            self.logger.info(
                'epoch {}, total time {:.2f}, '
                'lr_base {:.5f}, lr_pointnet {:.5f}'.format(
                    epoch, (time.time() - tic),
                    optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr']
                )
            )

            # Persist every completed epoch before validation so an interrupted
            # evaluation can still resume from the exact optimizer state.
            if dist.get_rank() == 0 and metric_retention:
                save_checkpoint(
                    args, epoch, model, optimizer, scheduler, save_cur=True
                )
                retention = update_checkpoint_retention(
                    args.log_dir,
                    epoch,
                    retained_metrics=retained_metrics,
                )
                if retention["removed"]:
                    self.logger.info(
                        "Checkpoint retention removed: {}".format(
                            ", ".join(retention["removed"])
                        )
                    )

            # save model and validate
            if epoch % args.val_freq == 0:
                if dist.get_rank() == 0 and not metric_retention:
                    save_checkpoint(args, epoch, model, optimizer, scheduler)
                
                # validate *
                print("Test evaluation.......")
                metrics = self.evaluate_one_epoch(
                    epoch, test_loader,
                    model, criterion, set_criterion, args
                )
                last_eval_epoch = epoch
                if dist.get_rank() == 0:
                    receipt_path = save_eval_metrics_receipt(
                        args.log_dir, epoch, metrics
                    )
                    if receipt_path is not None:
                        self.logger.info(
                            "Evaluation receipt saved in {}".format(
                                receipt_path
                            )
                        )
                    if metric_retention:
                        retention = update_checkpoint_retention(
                            args.log_dir,
                            epoch,
                            metrics,
                            retained_metrics=retained_metrics,
                        )
                        best_summary = ", ".join(
                            "{}=epoch{}:{:.6f}".format(
                                name, record["epoch"], record["value"]
                            )
                            for name, record in sorted(
                                retention["manifest"]["best"].items()
                            )
                        )
                        self.logger.info(
                            "Checkpoint retention best: {}".format(
                                best_summary
                            )
                        )

        # Training is over
        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        if not metric_retention:
            if dist.get_rank() == 0:
                save_checkpoint(args, 'last', model, optimizer, scheduler, True)
                self.logger.info("Saved in {}".format(saved_path))
        elif dist.get_rank() == 0:
            if not os.path.isfile(saved_path):
                raise RuntimeError(
                    "metric retention did not produce a latest checkpoint"
                )
            self.logger.info(
                "Latest checkpoint hard link is {}".format(saved_path)
            )
        if last_eval_epoch != args.max_epoch:
            metrics = self.evaluate_one_epoch(
                args.max_epoch, test_loader,
                model, criterion, set_criterion, args
            )
            if dist.get_rank() == 0:
                receipt_path = save_eval_metrics_receipt(
                    args.log_dir, args.max_epoch, metrics
                )
                if metric_retention:
                    update_checkpoint_retention(
                        args.log_dir,
                        args.max_epoch,
                        metrics,
                        retained_metrics=retained_metrics,
                    )
                if receipt_path is not None:
                    self.logger.info(
                        "Evaluation receipt saved in {}".format(receipt_path)
                    )
        return saved_path

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    def _capture_fpr_audit_sentinel(self, model, batch_data):
        """Run one deterministic held-out batch and hash frozen outputs."""
        model.eval()
        with torch.no_grad():
            batch_data = self._to_gpu(batch_data)
            inputs = self._get_inputs(batch_data)
            inputs["train"] = False
            end_points = model(inputs)
        return capture_fpr_audit_output_state(end_points)

    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances']
        }

    @staticmethod
    def _compute_loss(end_points, criterion, set_criterion, args):
        loss, end_points = criterion(
            end_points, args.num_decoder_layers,
            set_criterion,
            query_points_obj_topk=args.query_points_obj_topk,
            source_choice_selector_loss_weight=(
                args.source_choice_selector_loss_weight
                if args.use_source_choice_selector else 0.0
            ),
            source_choice_selector_default_source=(
                args.source_choice_selector_default_source
            ),
            source_choice_selector_choice_target=(
                args.source_choice_selector_choice_target
            ),
            source_choice_selector_min_iou_gap=(
                args.source_choice_selector_min_iou_gap
            ),
            mask_loss_scale=args.mask_loss_scale,
            consistency_loss_scale=args.consistency_loss_scale,
            source_moe_balance_loss_weight=(
                getattr(args, 'source_moe_balance_loss_weight', 0.01)
                if getattr(args, 'use_source_moe', False) else 0.0
            ),
            source_moe_rank_loss_weight=(
                getattr(args, 'source_moe_rank_loss_weight', 1.0)
                if getattr(args, 'use_source_moe', False) else 0.0
            ),
            source_moe_mask_rank_loss_weight=getattr(
                args, 'source_moe_mask_rank_loss_weight', 0.25
            ),
            source_moe_rank_temperature=getattr(
                args, 'source_moe_rank_temperature', 0.1
            ),
            source_moe_anchor_loss_weight=getattr(
                args, 'source_moe_anchor_loss_weight', 0.0
            ),
            source_moe_anchor_margin=getattr(
                args, 'source_moe_anchor_margin', 0.05
            ),
            source_moe_gate_loss_weight=(
                getattr(args, 'source_moe_gate_loss_weight', 0.0)
                if getattr(args, 'source_moe_use_fallback_gate', False)
                else 0.0
            ),
            source_moe_gate_mask_loss_weight=getattr(
                args, 'source_moe_gate_mask_loss_weight', 0.25
            ),
            source_moe_gate_focal_gamma=getattr(
                args, 'source_moe_gate_focal_gamma', 2.0
            ),
            source_moe_gate_false_override_weight=getattr(
                args, 'source_moe_gate_false_override_weight', 2.0
            ),
            source_moe_gate_break_cost=getattr(
                args, 'source_moe_gate_break_cost', 2.0
            ),
            source_moe_gate_mask_utility_weight=getattr(
                args, 'source_moe_gate_mask_utility_weight', 0.25
            ),
            source_moe_gate_objective=getattr(
                args, 'source_moe_gate_objective', 'balanced_focal'
            ),
            source_moe_gate_setwise_temperature=getattr(
                args, 'source_moe_gate_setwise_temperature', 0.0
            ),
            source_moe_gate_boundary_loss_weight=getattr(
                args, 'source_moe_gate_boundary_loss_weight', 0.0
            ),
            joint_query_quality_loss_weight=(
                getattr(args, 'joint_query_quality_loss_weight', 1.0)
                if getattr(
                    args, 'use_joint_query_quality_reranker', False
                ) else 0.0
            ),
            joint_query_quality_mask_weight=getattr(
                args, 'joint_query_quality_mask_weight', 0.25
            ),
            joint_query_quality_temperature=getattr(
                args, 'joint_query_quality_temperature', 0.25
            ),
            joint_query_quality_aux_loss_weight=getattr(
                args, 'joint_query_quality_aux_loss_weight', 1.0
            ),
            joint_query_quality_anchor_loss_weight=getattr(
                args, 'joint_query_quality_anchor_loss_weight', 0.5
            ),
            joint_query_quality_anchor_margin=getattr(
                args, 'joint_query_quality_anchor_margin', 0.05
            ),
            joint_query_quality_use_metric_aligned_utility=getattr(
                args,
                'joint_query_quality_use_metric_aligned_utility',
                False,
            ),
            joint_query_quality_metric_utility_temperature=getattr(
                args,
                'joint_query_quality_metric_utility_temperature',
                0.05,
            ),
            joint_query_quality_bidirectional_anchor=getattr(
                args, 'joint_query_quality_bidirectional_anchor', False
            ),
            joint_query_quality_anchor_margin_050=getattr(
                args, 'joint_query_quality_anchor_margin_050', 0.10
            ),
            joint_query_quality_pairwise_loss_weight=getattr(
                args, 'joint_query_quality_pairwise_loss_weight', 0.0
            ),
            joint_query_quality_listwise_loss_weight=getattr(
                args, 'joint_query_quality_listwise_loss_weight', 1.0
            ),
            joint_query_quality_transition_loss_weight=getattr(
                args, 'joint_query_quality_transition_loss_weight', 0.0
            ),
            joint_query_quality_setwise_repair_boundary_loss_weight=getattr(
                args,
                'joint_query_quality_setwise_repair_boundary_loss_weight', 0.0
            ),
            joint_query_quality_setwise_negative_tail_loss_weight=getattr(
                args,
                'joint_query_quality_setwise_negative_tail_loss_weight', 0.0
            ),
            joint_query_quality_setwise_rank_loss_weight=getattr(
                args, 'joint_query_quality_setwise_rank_loss_weight', 0.0
            ),
            joint_query_quality_setwise_dense_safety_loss_weight=getattr(
                args,
                'joint_query_quality_setwise_dense_safety_loss_weight', 0.0
            ),
            joint_query_quality_setwise_balanced_safety_loss_weight=getattr(
                args,
                'joint_query_quality_setwise_balanced_safety_loss_weight',
                0.0,
            ),
            joint_query_quality_setwise_factorized_safety_loss_weight=getattr(
                args,
                'joint_query_quality_setwise_factorized_safety_loss_weight',
                0.0,
            ),
            joint_query_quality_setwise_factorized_risk_bound_loss_weight=getattr(
                args,
                'joint_query_quality_setwise_factorized_risk_bound_loss_weight',
                0.0,
            ),
            joint_query_quality_factorized_hit_loss_weight=getattr(
                args, 'joint_query_quality_factorized_hit_loss_weight', 0.0
            ),
            joint_query_quality_factorized_pair_loss_weight=getattr(
                args, 'joint_query_quality_factorized_pair_loss_weight', 0.0
            ),
            joint_query_quality_transition_break_cost=getattr(
                args, 'joint_query_quality_transition_break_cost', 4.0
            ),
            joint_query_quality_transition_neutral_weight=getattr(
                args, 'joint_query_quality_transition_neutral_weight', 0.25
            ),
            joint_query_quality_deploy_candidate_top_k=getattr(
                args, 'joint_query_quality_deploy_candidate_top_k', 0
            ),
            joint_query_quality_source_candidate_top_k=getattr(
                args, 'joint_query_quality_source_candidate_top_k', 0
            ),
            joint_query_quality_oracle_candidate_top_k=getattr(
                args, 'joint_query_quality_oracle_candidate_top_k', 0
            ),
            joint_query_quality_source_mix_loss_weight=getattr(
                args, 'joint_query_quality_source_mix_loss_weight', 0.0
            ),
            joint_query_quality_source_mix_alignment_temperature=getattr(
                args,
                'joint_query_quality_source_mix_alignment_temperature',
                0.25,
            ),
            joint_query_quality_source_mix_query_focus_weight=getattr(
                args,
                'joint_query_quality_source_mix_query_focus_weight',
                0.0,
            ),
            joint_query_quality_candidate_mask_loss_weight=getattr(
                args, 'joint_query_quality_candidate_mask_loss_weight', 0.0
            ),
            joint_query_quality_candidate_lovasz_loss_weight=getattr(
                args, 'joint_query_quality_candidate_lovasz_loss_weight', 0.0
            ),
            joint_query_quality_candidate_mask_top_k=getattr(
                args, 'joint_query_quality_candidate_mask_top_k', 16
            ),
            sacr_score_refiner_loss_weight=(
                getattr(args, 'sacr_score_refiner_loss_weight', 1.0)
                if getattr(args, 'use_sacr_score_refiner', False) else 0.0
            ),
            sacr_score_temperature=getattr(
                args, 'sacr_score_temperature', 0.1
            ),
            sacr_score_mask_weight=getattr(
                args, 'sacr_score_mask_weight', 0.25
            ),
            sacr_score_use_parent_relative_abstention=getattr(
                args, 'sacr_score_use_parent_relative_abstention', False
            ),
            sacr_score_use_relation_counterfactual=getattr(
                args, 'sacr_score_use_relation_counterfactual', False
            ),
            sacr_score_max_delta=getattr(
                args, 'sacr_score_max_delta', 0.25
            ),
            sacr_score_min_box_advantage=getattr(
                args, 'sacr_score_min_box_advantage', 0.03
            ),
            sacr_score_promotion_margin=getattr(
                args, 'sacr_score_promotion_margin', 0.01
            ),
            sacr_score_mask_tolerance=getattr(
                args, 'sacr_score_mask_tolerance', 0.02
            ),
            sacr_score_raw_margin=getattr(
                args, 'sacr_score_raw_margin', 0.1
            ),
            sacr_score_dense_weight=getattr(
                args, 'sacr_score_dense_weight', 0.25
            ),
            sacr_score_preserve_weight=getattr(
                args, 'sacr_score_preserve_weight', 1.0
            ),
            sacr_score_gate_weight=getattr(
                args, 'sacr_score_gate_weight', 0.05
            ),
            sacr_score_saturation_weight=getattr(
                args, 'sacr_score_saturation_weight', 0.05
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
            sacr_counterfactual_geometry_threshold=getattr(
                args, 'sacr_counterfactual_geometry_threshold', 0.08
            ),
            sacr_counterfactual_iou_gap=getattr(
                args, 'sacr_counterfactual_iou_gap', 0.10
            ),
            sacr_counterfactual_correct_iou_threshold=getattr(
                args, 'sacr_counterfactual_correct_iou_threshold', 0.25
            ),
            sacr_counterfactual_pair_margin=getattr(
                args, 'sacr_counterfactual_pair_margin', 0.25
            ),
            sacr_counterfactual_max_negatives=getattr(
                args, 'sacr_counterfactual_max_negatives', 4
            ),
            relation_counterfactual_aux_loss_weight=getattr(
                args, 'relation_counterfactual_aux_loss_weight', 0.0
            ),
            relation_counterfactual_aux_parent_top_k=getattr(
                args, 'relation_counterfactual_aux_parent_top_k', 32
            ),
            relation_counterfactual_aux_target_tolerance=getattr(
                args, 'relation_counterfactual_aux_target_tolerance', 0.10
            ),
            relation_counterfactual_aux_attribute_tolerance=getattr(
                args, 'relation_counterfactual_aux_attribute_tolerance', 0.10
            ),
            relation_counterfactual_aux_geometry_threshold=getattr(
                args, 'relation_counterfactual_aux_geometry_threshold', 0.08
            ),
            relation_counterfactual_aux_correct_iou_threshold=getattr(
                args,
                'relation_counterfactual_aux_correct_iou_threshold',
                0.25,
            ),
            relation_counterfactual_aux_pair_margin=getattr(
                args, 'relation_counterfactual_aux_pair_margin', 0.05
            ),
            relation_counterfactual_aux_max_negatives=getattr(
                args, 'relation_counterfactual_aux_max_negatives', 8
            ),
            relation_counterfactual_aux_target_confidence_floor=getattr(
                args,
                'relation_counterfactual_aux_target_confidence_floor',
                0.05,
            ),
            relation_counterfactual_aux_attribute_confidence_floor=getattr(
                args,
                'relation_counterfactual_aux_attribute_confidence_floor',
                0.02,
            ),
            relation_counterfactual_aux_acc025_pair_weight=getattr(
                args, 'relation_counterfactual_aux_acc025_pair_weight', 2.0
            ),
            relation_counterfactual_aux_conservative_anchor_set=getattr(
                args,
                'relation_counterfactual_aux_conservative_anchor_set',
                False,
            ),
            tier_hard_query_aux_loss_weight=getattr(
                args, 'tier_hard_query_aux_loss_weight', 0.0
            ),
            tier_hard_query_aux_candidate_top_k=getattr(
                args, 'tier_hard_query_aux_candidate_top_k', 128
            ),
            tier_hard_query_aux_max_negatives=getattr(
                args, 'tier_hard_query_aux_max_negatives', 8
            ),
            tier_hard_query_aux_target_tolerance=getattr(
                args, 'tier_hard_query_aux_target_tolerance', 0.15
            ),
            tier_hard_query_aux_target_confidence_floor=getattr(
                args,
                'tier_hard_query_aux_target_confidence_floor',
                0.01,
            ),
            tier_hard_query_aux_pair_margin=getattr(
                args, 'tier_hard_query_aux_pair_margin', 0.05
            ),
            tier_hard_query_aux_preserve_weight=getattr(
                args, 'tier_hard_query_aux_preserve_weight', 0.25
            ),
            tier_hard_query_aux_acc025_pair_weight=getattr(
                args, 'tier_hard_query_aux_acc025_pair_weight', 2.0
            ),
            density_aware_target_box_loss_weight=getattr(
                args, 'density_aware_target_box_loss_weight', 0.0
            ),
            density_scene_audit_return_match_indices=bool(getattr(
                args, 'density_aware_target_box_scene_disjoint_audit', False
            )),
            query_mask_fusion_train_only=getattr(
                args, 'query_mask_fusion_train_only', False
            ) or getattr(args, 'egqs_mask_refiner_train_only', False),
            joint_query_quality_train_only=getattr(
                args, 'joint_query_quality_train_only', False
            ),
            sacr_score_refiner_train_only=getattr(
                args, 'sacr_score_refiner_train_only', False
            ),
            parent_relative_text_verifier_loss_weight=getattr(
                args, 'parent_relative_text_verifier_loss_weight', 0.0
            ),
            parent_relative_text_verifier_positive_margin=getattr(
                args, 'parent_relative_text_verifier_positive_margin', 0.25
            ),
            parent_relative_text_verifier_neutral_margin=getattr(
                args, 'parent_relative_text_verifier_neutral_margin', 0.25
            ),
            parent_relative_text_verifier_train_only=getattr(
                args, 'parent_relative_text_verifier_train_only', False
            ),
            parent_relative_text_verifier_counterfactual_training=getattr(
                args,
                'parent_relative_text_verifier_counterfactual_training',
                False,
            ),
        )
        return loss, end_points

    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            moe_scalar = (
                key.startswith('moe_expert_usage_')
                or (
                    key.startswith('source_moe_gate_')
                    and key.endswith('_count')
                )
                or key in (
                    'moe_routed_scale', 'moe_router_entropy',
                    'moe_rerank_abs_mean', 'moe_rerank_abs_max',
                    'moe_gate_switch_ratio',
                    'moe_gate_correction_switch_ratio',
                    'moe_gate_v19_correction_switch_ratio',
                    'moe_gate_positive_candidate_ratio',
                    'moe_gate_max_margin_mean',
                    'moe_gate_quality_uncertainty_mean',
                    'moe_gate_context_scale',
                    'source_moe_gate_oracle_switch_recall_ratio',
                    'source_moe_gate_predicted_switch_precision_ratio',
                    'source_moe_gate_false_switch_ratio',
                    'source_moe_gate_oracle_query_match_ratio',
                )
            )
            query_mask_fusion_scalar = key.startswith(
                'query_mask_fusion_'
            ) and torch.is_tensor(end_points[key]) and end_points[key].numel() == 1
            joint_query_quality_scalar = key.startswith(
                'joint_query_quality_'
            ) and torch.is_tensor(end_points[key]) and end_points[key].numel() == 1
            sacr_scalar = key.startswith(
                'sacr_'
            ) and torch.is_tensor(end_points[key]) and end_points[key].numel() == 1
            egqs_scalar = key.startswith(
                'egqs_mask_refiner_'
            ) and torch.is_tensor(end_points[key]) and end_points[key].numel() == 1
            relation_aux_scalar = key.startswith(
                'relation_counterfactual_aux_'
            ) and torch.is_tensor(end_points[key]) and end_points[key].numel() == 1
            tier_aux_scalar = key.startswith(
                'tier_hard_query_aux_'
            ) and torch.is_tensor(end_points[key]) and end_points[key].numel() == 1
            parent_relative_text_verifier_scalar = key.startswith(
                'parent_relative_text_verifier_'
            ) and torch.is_tensor(end_points[key]) and end_points[key].numel() == 1
            density_aware_target_box_scalar = key.startswith(
                'density_aware_target_box_'
            ) and torch.is_tensor(end_points[key]) and end_points[key].numel() == 1
            if ('loss' in key or 'acc' in key or 'ratio' in key or moe_scalar
                    or query_mask_fusion_scalar
                    or joint_query_quality_scalar or sacr_scalar
                    or egqs_scalar or relation_aux_scalar
                    or tier_aux_scalar
                    or parent_relative_text_verifier_scalar
                    or density_aware_target_box_scalar):
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict

    @staticmethod
    def _finite_scalar_float(value, label):
        if isinstance(value, bool):
            raise ValueError("{} must be a numeric scalar".format(label))
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("{} must be a numeric scalar".format(label))
            value = value.detach().reshape(()).cpu().item()
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise ValueError("{} must be a numeric scalar".format(label))
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("{} must be finite".format(label))
        return result

    @staticmethod
    def _optimizer_reference_device(optimizer, loss=None):
        if torch.is_tensor(loss):
            return loss.device
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                return parameter.device
        return torch.device("cpu")

    @staticmethod
    def _distributed_any(local_failure, device):
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return bool(local_failure)
        flag = torch.tensor(
            1 if local_failure else 0,
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    @classmethod
    def _validated_batch_loss_values(cls, loss, end_points, optimizer):
        total_loss_error = None
        total_loss_value = None
        if not torch.is_tensor(loss) or loss.dim() != 0:
            total_loss_error = "total loss must be a scalar tensor"
        else:
            try:
                total_loss_value = cls._finite_scalar_float(
                    loss, "total loss"
                )
            except ValueError as error:
                total_loss_error = str(error)
        device = cls._optimizer_reference_device(optimizer, loss=loss)
        if cls._distributed_any(total_loss_error is not None, device):
            raise ValueError(
                total_loss_error
                or "total loss must be finite and scalar on every rank"
            )

        values = {}
        loss_value_error = None
        for key in sorted(end_points.keys()):
            if "loss" not in key:
                continue
            try:
                values[key] = cls._finite_scalar_float(
                    end_points[key], "end_points {}".format(key)
                )
            except ValueError as error:
                loss_value_error = str(error)
                break
        if cls._distributed_any(loss_value_error is not None, device):
            raise ValueError(
                loss_value_error
                or "loss-like end_points must be finite numeric scalars "
                "on every rank"
            )
        if "total_loss" not in values:
            values["total_loss"] = total_loss_value
        return values

    @classmethod
    def _reject_nonfinite_optimizer_gradients(
            cls, optimizer, loss, model=None):
        name_by_parameter = {}
        if model is not None:
            unwrapped = model.module if hasattr(model, "module") else model
            name_by_parameter = {
                id(parameter): name
                for name, parameter in unwrapped.named_parameters()
            }
        local_failures = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                gradient_values = (
                    gradient.coalesce().values()
                    if gradient.is_sparse else gradient
                )
                if not bool(torch.isfinite(gradient_values).all().item()):
                    local_failures.append(name_by_parameter.get(
                        id(parameter), "<unnamed-parameter>"
                    ))
        device = cls._optimizer_reference_device(optimizer, loss=loss)
        local_failure = bool(local_failures)
        if cls._distributed_any(local_failure, device):
            detail = (
                ", ".join(local_failures[:16])
                if local_failures else "another distributed rank"
            )
            raise ValueError(
                "optimizer gradient tensors must all be finite: {}".format(
                    detail
                )
            )

    @staticmethod
    def _stat_to_float(value):
        if isinstance(value, (float, int)):
            return float(value)
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    @staticmethod
    def _is_source_choice_diagnostic_key(key):
        return (
            key.startswith("source_choice_")
            and "loss" not in key
            and (
                "acc" in key
                or "ratio" in key
                or "fix" in key
                or "break" in key
                or "headroom" in key
            )
        )

    @classmethod
    def _format_source_choice_diagnostics(cls, stat_dict, denom):
        keys = [
            key for key in sorted(stat_dict.keys())
            if cls._is_source_choice_diagnostic_key(key)
        ]
        return ''.join([
            f'{key} {cls._stat_to_float(stat_dict[key]) / denom:.4f} \t'
            for key in keys
        ])

    def _log_source_choice_diagnostics(self, stat_dict, denom):
        message = self._format_source_choice_diagnostics(stat_dict, denom)
        if message:
            self.logger.info('[source_choice] ' + message)

    def _log_source_moe_diagnostics(self, stat_dict, denom):
        keys = [
            key for key in sorted(stat_dict.keys())
            if key.startswith('source_moe_')
            and ('acc' in key or 'ratio' in key)
        ]
        keys.extend([
            key for key in (
                'moe_routed_scale', 'moe_router_entropy',
                'moe_rerank_abs_mean', 'moe_rerank_abs_max',
                'moe_gate_switch_ratio',
                'moe_gate_correction_switch_ratio',
                'moe_gate_v19_correction_switch_ratio',
                'moe_gate_positive_candidate_ratio',
                'moe_gate_max_margin_mean',
                'moe_gate_quality_uncertainty_mean',
                'moe_gate_context_scale',
            )
            if key in stat_dict
        ])
        keys.extend([
            key for key in sorted(stat_dict.keys())
            if key.startswith('moe_expert_usage_')
        ])
        keys.extend([
            key for key in sorted(stat_dict.keys())
            if key.startswith('joint_query_quality_')
            and 'loss' not in key
        ])
        keys.extend([
            key for key in sorted(stat_dict.keys())
            if key.startswith('sacr_')
        ])
        keys.extend([
            key for key in sorted(stat_dict.keys())
            if key.startswith('relation_counterfactual_aux_')
            and key != 'relation_counterfactual_aux_loss'
        ])
        keys.extend([
            key for key in sorted(stat_dict.keys())
            if key.startswith('tier_hard_query_aux_')
            and key != 'tier_hard_query_aux_loss'
        ])
        keys.extend([
            key for key in sorted(stat_dict.keys())
            if key.startswith('parent_relative_text_verifier_')
            and key != 'parent_relative_text_verifier_loss'
        ])
        if keys:
            self.logger.info('[source_moe] ' + ''.join([
                '{} {:.4f} \t'.format(
                    key, self._stat_to_float(stat_dict[key]) / denom
                )
                for key in keys
            ]))

    @staticmethod
    def _set_source_moe_train_mode(model, args):
        source_moe_only = getattr(args, "source_moe_train_only", False)
        gate_only = getattr(args, "source_moe_gate_train_only", False)
        query_only = getattr(args, "query_mask_fusion_train_only", False)
        egqs_only = getattr(args, "egqs_mask_refiner_train_only", False)
        joint_query_only = getattr(
            args, "joint_query_quality_train_only", False
        )
        decoder_query_adapter_only = getattr(
            args, "decoder_query_adapter_train_only", False
        )
        sacr_score_refiner_only = getattr(
            args, "sacr_score_refiner_train_only", False
        )
        parent_relative_text_verifier_only = getattr(
            args, "parent_relative_text_verifier_train_only", False
        )
        gate_new_heads_only = getattr(
            args, "source_moe_gate_new_heads_only", False
        )
        if not (
                source_moe_only or gate_only or query_only or egqs_only
                or joint_query_only or decoder_query_adapter_only
                or sacr_score_refiner_only
                or parent_relative_text_verifier_only):
            model.train()
            return

        model.eval()
        unwrapped = model.module if hasattr(model, "module") else model
        if parent_relative_text_verifier_only:
            if bool(getattr(
                    args,
                    "parent_relative_text_verifier_counterfactual_training",
                    False,
            )):
                # Only the MCLN root training bit controls construction of the
                # differentiable actual score leaf and the training-only
                # counterfactual views.  Do not call train() here: every frozen
                # child must remain in eval mode for exact V99 Parent behavior.
                unwrapped.training = True
                if model is not unwrapped:
                    model.training = True
            for module_name in (
                    "structured_slot_builder", "sacr_head",
                    "parent_relative_text_verifier"):
                module = getattr(unwrapped, module_name, None)
                if module is None:
                    raise ValueError(
                        "parent-relative-verifier-only mode requires {}"
                        .format(module_name)
                    )
                module.train()
            return
        if sacr_score_refiner_only:
            for module_name in ("structured_slot_builder", "sacr_head"):
                module = getattr(unwrapped, module_name, None)
                if module is None:
                    raise ValueError(
                        "SACR-score-only mode requires {}".format(
                            module_name
                        )
                    )
                module.train()
            return
        if decoder_query_adapter_only:
            adapter = getattr(unwrapped, "decoder_query_adapter", None)
            if adapter is None:
                raise ValueError(
                    "decoder-query-adapter-only mode requires an adapter"
                )
            adapter.train()
            return
        if query_only:
            calibrator = getattr(
                unwrapped, "query_mask_fusion_calibrator", None
            )
            if calibrator is None:
                raise ValueError(
                    "query-mask-fusion-only mode requires a calibrator"
                )
            calibrator.train()
            return
        if egqs_only:
            refiner = getattr(unwrapped, "egqs_mask_refiner", None)
            if refiner is None:
                raise ValueError(
                    "EGQS-only mode requires an EGQS mask refiner"
                )
            refiner.train()
            return
        if joint_query_only:
            reranker = getattr(
                unwrapped, "joint_query_quality_reranker", None
            )
            if reranker is None:
                raise ValueError(
                    "joint-query-quality-only mode requires a reranker"
                )
            reranker.train()
            if getattr(args, "use_sacr_source", False):
                for module_name in ("structured_slot_builder", "sacr_head"):
                    module = getattr(unwrapped, module_name, None)
                    if module is None:
                        raise ValueError(
                            "SACR training requires {}".format(module_name)
                        )
                    module.train()
            return

        source_moe = getattr(unwrapped, "source_moe", None)
        if source_moe is None:
            raise ValueError("source-MoE-only mode requires a source_moe module")
        if not gate_only:
            source_moe.train()
            return

        fallback_gate = getattr(source_moe, "fallback_gate", None)
        if fallback_gate is None:
            raise ValueError(
                "gate-only mode requires a source_moe fallback gate"
            )
        if not gate_new_heads_only:
            fallback_gate.train()
            return
        action_mode = getattr(fallback_gate, "action_mode", None)
        if action_mode not in (
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
            raise ValueError("new-head-only mode requires the cascade gate")

        fallback_gate.eval()
        if action_mode == "cascade_v19_fallback_set_correction":
            module_names = ["cascade_fallback_set_action_head"]
        elif action_mode == "cascade_v19_rich_set_correction":
            module_names = ["cascade_rich_fallback_set_action_head"]
        elif action_mode == "cascade_v23_dense_quality_correction":
            module_names = ["cascade_dense_quality_set_head"]
        elif action_mode == "cascade_v24_relative_risk_correction":
            module_names = [
                "cascade_dense_quality_set_head",
                "cascade_relative_risk_set_head",
            ]
        elif action_mode in (
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            module_names = [
                "cascade_dense_quality_set_head",
                "cascade_pairwise_calibrated_set_head",
            ]
            if action_mode == "cascade_v28_selected_abstention_correction":
                module_names.append("cascade_selected_abstention_head")
            if action_mode == "cascade_v29_counterfactual_selected_correction":
                module_names.append(
                    "cascade_counterfactual_selected_risk_head"
                )
            if action_mode == (
                    "cascade_v37_counterfactual_benefit_hazard_correction"):
                module_names.append(
                    "cascade_counterfactual_benefit_hazard_head"
                )
            if action_mode == "cascade_v38_complementary_logodds_correction":
                module_names.append("cascade_counterfactual_logodds_head")
            if action_mode == "cascade_v39_hazard_residual_correction":
                module_names.append(
                    "cascade_counterfactual_hazard_residual_head"
                )
        else:
            module_names = [
                "absolute_quality_head",
                "cascade_quality_adapter",
                "cascade_correction_head",
            ]
            if action_mode in (
                    "cascade_opportunity_quality_correction",
                    "cascade_opportunity_verified_correction",
                    "cascade_joint_risk_correction"):
                module_names.append("cascade_opportunity_head")
            if action_mode in (
                    "cascade_opportunity_verified_correction",
                    "cascade_joint_risk_correction"):
                module_names.append("cascade_candidate_safety_head")
            if action_mode == "cascade_joint_risk_correction":
                module_names.append("cascade_joint_action_head")
        for module_name in module_names:
            module = getattr(fallback_gate, module_name, None)
            if module is None:
                raise ValueError(
                    "cascade gate is missing {}".format(module_name)
                )
            module.train()
        if action_mode in (
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction"):
            adaptive_source_mixer = getattr(
                source_moe, "adaptive_source_mixer", None
            )
            if adaptive_source_mixer is None:
                raise ValueError(
                    "dense-quality action is missing adaptive_source_mixer"
                )
            adaptive_source_mixer.train()


    # BRIEF Training
    def train_one_epoch(self, epoch, train_loader, model,
                        criterion, set_criterion,
                        optimizer, scheduler, args):
        """
        Run a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        stat_dict = {}  # collect statistics
        loss_sums = {}
        batch_count = 0
        optimizer_step_count = 0
        grad_norm_sum = 0.0
        processed_sample_count = 0
        processed_sample_ids = []
        accumulation_steps = getattr(
            args, "gradient_accumulation_steps", 1
        )
        max_train_batches = getattr(args, "max_train_batches", 0)
        accumulation_plan = _gradient_accumulation_plan(
            len(train_loader),
            max_train_batches,
            accumulation_steps,
            bool(getattr(
                args, "drop_incomplete_accumulation_group", False
            )),
        )
        total_batches = accumulation_plan["effective_batch_count"]
        self._set_source_moe_train_mode(model, args)
        counterfactual_parent_audit = bool(getattr(
            args,
            "parent_relative_text_verifier_counterfactual_training",
            False,
        )) and max_train_batches > 0

        # Loop over batches
        train_loader = tqdm(train_loader, ascii=True)
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= total_batches:
                break
            point_clouds = batch_data.get("point_clouds")
            if (
                    not torch.is_tensor(point_clouds)
                    or point_clouds.dim() < 1
                    or int(point_clouds.shape[0]) <= 0):
                raise ValueError(
                    "training batch lacks a positive point-cloud batch axis"
                )
            if counterfactual_parent_audit and batch_idx == 0:
                self._counterfactual_audit_sentinel_batch = copy.deepcopy(
                    batch_data
                )
                self._counterfactual_audit_before_outputs = (
                    self._capture_fpr_audit_sentinel(
                        model,
                        copy.deepcopy(
                            self._counterfactual_audit_sentinel_batch
                        ),
                    )
                )
                self._set_source_moe_train_mode(model, args)
            processed_sample_count += int(point_clouds.shape[0])
            fpr_identity_audit = bool(getattr(
                args, "fpr_scene_disjoint_audit", False
            ))
            density_identity_audit = bool(getattr(
                args,
                "density_aware_target_box_scene_disjoint_audit",
                False,
            ))
            if fpr_identity_audit or density_identity_audit:
                identity_key = (
                    "fpr_scene_audit_sample_index"
                    if fpr_identity_audit
                    else "density_scene_audit_sample_index"
                )
                sample_ids = batch_data.get(identity_key)
                if (
                        not torch.is_tensor(sample_ids)
                        or sample_ids.dim() != 1
                        or int(sample_ids.shape[0])
                        != int(point_clouds.shape[0])):
                    raise ValueError(
                        "scene audit batch sample identities are missing"
                    )
                processed_sample_ids.extend(
                    int(value) for value in sample_ids.detach().cpu().tolist()
                )
            accumulation_group_start = (
                batch_idx // accumulation_steps
            ) * accumulation_steps
            if batch_idx == accumulation_group_start:
                optimizer.zero_grad()
            accumulation_group_size = min(
                accumulation_steps,
                total_batches - accumulation_group_start,
            )
            should_step = (
                batch_idx + 1
                == accumulation_group_start + accumulation_group_size
            )
            sync_context = (
                contextlib.nullcontext()
                if should_step or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            with sync_context:
                # get the input data: pointcloud and text
                inputs = self._get_inputs(batch_data)

                # note Forward pass
                end_points = model(inputs)

                # note Compute loss and gradients, update parameters.
                for key in batch_data:
                    assert (key not in end_points)
                    end_points[key] = batch_data[key]
                loss, end_points = self._compute_loss(
                    end_points, criterion, set_criterion, args
                )

                batch_loss_values = self._validated_batch_loss_values(
                    loss, end_points, optimizer
                )

                score_gradient_audit_axes = None
                if getattr(
                        args,
                        "parent_relative_text_verifier_counterfactual_training",
                        False):
                    score_gradient_audit_axes = (
                        prepare_parent_relative_text_verifier_score_gradient_audit(
                            end_points
                        )
                    )

                (
                    loss / float(accumulation_group_size)
                ).backward()

                if getattr(
                        args,
                        "parent_relative_text_verifier_counterfactual_training",
                        False):
                    actual_score_axis = score_gradient_audit_axes[0]
                    if (not torch.is_tensor(actual_score_axis)
                            or actual_score_axis.grad is None):
                        raise ValueError(
                            "actual Parent score-gradient audit is missing"
                        )
                    actual_score_gradient_l1 = (
                        actual_score_axis.grad.detach().abs().sum()
                    )
                    if not bool(torch.isfinite(
                            actual_score_gradient_l1).item()):
                        raise ValueError(
                            "actual Parent score gradient must be finite"
                        )
                    counterfactual_score_axis = score_gradient_audit_axes[1]
                    if counterfactual_score_axis is None:
                        counterfactual_score_gradient_l1 = (
                            actual_score_gradient_l1 * 0.0
                        )
                    else:
                        if (not torch.is_tensor(counterfactual_score_axis)
                                or counterfactual_score_axis.grad is None):
                            raise ValueError(
                                "counterfactual Parent score-gradient audit "
                                "is missing"
                            )
                        counterfactual_score_gradient_l1 = (
                            counterfactual_score_axis.grad.detach().abs().sum()
                        )
                        if not bool(torch.isfinite(
                                counterfactual_score_gradient_l1).item()):
                            raise ValueError(
                                "counterfactual Parent score gradient must "
                                "be finite"
                            )
                    end_points[
                        "parent_relative_text_verifier_actual_"
                        "selected_score_gradient_l1"
                    ] = actual_score_gradient_l1
                    end_points[
                        "parent_relative_text_verifier_counterfactual_"
                        "selected_score_gradient_l1"
                    ] = counterfactual_score_gradient_l1

            self._reject_nonfinite_optimizer_gradients(
                optimizer, loss, model=model
            )

            if should_step:
                if args.clip_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_norm
                    )
                    grad_norm_sum += self._finite_scalar_float(
                        grad_total_norm, "gradient norm"
                    )

                optimizer.step()
                scheduler.step()
                optimizer_step_count += 1

            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, end_points)
            batch_count += 1
            for key, value in batch_loss_values.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + value

            # print loss
            if (batch_idx + 1) % args.print_freq == 0:
                # Terminal logs
                self.logger.info(
                    f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  '  # Train: [30][2000/2432]
                )
                self.logger.info(''.join([
                    f'{key} {stat_dict[key] / (batch_idx + 1):.4f} \t'
                    for key in sorted(stat_dict.keys())
                    if 'loss' in key and 'proposal_' not in key
                    and 'last_' not in key and 'head_' not in key
                ])) # loss，loss_bbox，loss_ce，loss_sem_align，loss_giou，query_points_generation_loss
                self._log_source_choice_diagnostics(
                    stat_dict, float(batch_idx + 1)
                )
                self._log_source_moe_diagnostics(
                    stat_dict, float(batch_idx + 1)
                )

                # # reset stat_dict
                # for key in sorted(stat_dict.keys()):
                #     stat_dict[key] = 0
                
                if dist.get_rank() == 0:
                    for key in self.tensorboard.item["train_loss"]:
                        self.tensorboard.item["train_loss"][key] = stat_dict[key] / (batch_idx + 1)
                    self.tensorboard.dump_tensorboard("train_loss", (epoch-1)*len(train_loader)+batch_idx+1)

        if batch_count <= 0:
            raise ValueError("training epoch has no batches")
        if optimizer_step_count <= 0:
            raise ValueError("training epoch has no optimizer steps")
        if args.clip_norm > 0:
            stat_dict['grad_norm'] = (
                grad_norm_sum
                * float(batch_count)
                / float(optimizer_step_count)
            )

        # tensorboard
        if dist.get_rank() == 0:
            # loss
            for key in self.tensorboard.item["train_loss"]:
                self.tensorboard.item["train_loss"][key] = stat_dict[key] / batch_count
            self.tensorboard.dump_tensorboard("train_loss", (epoch-1)*len(train_loader)+batch_idx+1)
            # lr
            self.tensorboard.item["train_lr"]["lr_base"] = optimizer.param_groups[0]['lr']
            self.tensorboard.item["train_lr"]["lr_pointnet"] = optimizer.param_groups[1]['lr']
            self.tensorboard.dump_tensorboard("train_lr", epoch)
            query_diagnostics = (
                'query_mask_fusion_abs_residual_mean',
                'query_mask_fusion_abs_residual_max',
                'query_mask_fusion_weight_std_mean',
            )
            available = [key for key in query_diagnostics if key in stat_dict]
            if available:
                self.logger.info(
                    'Query mask fusion: ' + ', '.join(
                        '{}={:.6f}'.format(
                            key,
                            stat_dict[key] / float(batch_count),
                        )
                        for key in available
                    )
                )
            egqs_diagnostics = tuple(
                key for key in stat_dict
                if key.startswith('egqs_mask_refiner_')
            )
            if egqs_diagnostics:
                self.logger.info(
                    'EGQS mask refiner: ' + ', '.join(
                        '{}={:.6f}'.format(
                            key,
                            stat_dict[key] / float(batch_count),
                        )
                        for key in sorted(egqs_diagnostics)
                    )
                )
        receipt = {
            "schema": TRAIN_LOSS_RECEIPT_SCHEMA,
            "batch_count": batch_count,
            "loss_means": {
                key: loss_sums[key] / float(batch_count)
                for key in sorted(loss_sums.keys())
            },
            "stat_means": {
                key: float(stat_dict[key] / float(batch_count))
                for key in sorted(stat_dict.keys())
            },
        }
        if counterfactual_parent_audit:
            receipt["optimizer_step_count"] = optimizer_step_count
            receipt["sample_count"] = processed_sample_count
        if (
                bool(getattr(args, "fpr_scene_disjoint_audit", False))
                or bool(getattr(
                    args,
                    "density_aware_target_box_scene_disjoint_audit",
                    False,
                ))):
            receipt["sample_count"] = processed_sample_count
            receipt["sample_identity_count"] = len(processed_sample_ids)
            receipt["sample_identity_unique_count"] = len(set(
                processed_sample_ids
            ))
            receipt["sample_identity_sha256"] = (
                fpr_scene_sample_identity_digest(processed_sample_ids)
            )
            if bool(getattr(
                    args, "fpr_scene_disjoint_av4_audit", False)):
                receipt["optimizer_step_count"] = optimizer_step_count
            if bool(getattr(
                    args,
                    "density_aware_target_box_scene_disjoint_audit",
                    False,
            )):
                receipt["optimizer_step_count"] = optimizer_step_count
        return receipt

    # BRIEF eval 
    @torch.no_grad()
    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model,
                          stat_dict,
                          criterion, set_criterion, args):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # STEP Forward pass
        end_points = model(inputs)
        if (getattr(args, "eval_use_rec_reranker_scores", False)
                or getattr(
                    args, "eval_use_rec_geometry_reranker_scores", False
                ) or getattr(
                    args, "eval_use_rec_selective_residual_scores", False
                ) or getattr(
                    args, "eval_use_rec_hierarchical_reranker_scores", False
                ) or getattr(
                    args, "eval_use_rec_joint_box_mask", False
                )):
            self._attach_rec_reranker_scores(
                end_points,
                inputs,
                args,
                batch_idx=batch_idx,
                num_batches=len(test_loader),
            )

        # from thop import profile
        # macs, _ = profile(model, inputs=(inputs, ))
        # print(f"Total FLOPs: {macs} (or {macs / 1e9} GFLOPs)")

        # STEP Compute loss
        for key in batch_data:
            assert (key not in end_points)
            end_points[key] = batch_data[key]
        _, end_points = self._compute_loss(
            end_points, criterion, set_criterion, args
        )
        for key in end_points:
            if 'pred_size' in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)

        # Accumulate statistics and print out
        stat_dict = self._accumulate_stats(stat_dict, end_points)
        if (batch_idx + 1) % args.print_freq == 0:
            self.logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ')
            self.logger.info(''.join([
                f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                for key in sorted(stat_dict.keys())
                if 'loss' in key and 'proposal_' not in key
                and 'last_' not in key and 'head_' not in key
            ]))
            self._log_source_choice_diagnostics(
                stat_dict, float(batch_idx + 1)
            )
        return stat_dict, end_points

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        return None
