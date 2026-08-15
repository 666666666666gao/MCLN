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
import copy
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
from utils import get_scheduler, setup_logger

from utils import record_tensorboard

from tqdm import tqdm


TRAIN_LOSS_RECEIPT_SCHEMA = "mcln-train-loss-epoch-v1"
CHECKPOINT_RETENTION_SCHEMA = "mcln-checkpoint-retention-v1"
SOURCE_CHOICE_DIAGNOSTICS_SCHEMA = "mcln-source-choice-diagnostics-v1"
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
    parser.add_argument('--use_sacr_score_refiner', action='store_true',
                        default=False)
    parser.add_argument('--sacr_score_refiner_train_only',
                        action='store_true', default=False)
    parser.add_argument('--sacr_score_refiner_lr', type=float, default=0.0003)
    parser.add_argument('--sacr_score_refiner_loss_weight', type=float,
                        default=1.0)
    parser.add_argument('--sacr_score_temperature', type=float, default=0.1)
    parser.add_argument('--sacr_score_mask_weight', type=float, default=0.25)
    parser.add_argument('--sacr_score_max_delta', type=float, default=0.25)
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

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
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
        '--checkpoint_metric_retention', action='store_true', default=False,
        help=(
            'Atomically keep the latest checkpoint and the best checkpoints '
            'for REC@0.25, REC@0.50, mask@0.25, mask@0.50, and mask mIoU.'
        ),
    )
    parser.add_argument('--log_dir', default='log',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--exp', default='exp',
                        help='exp name to save model checkpoint')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
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
    checkpoint_state = checkpoint.get("model")
    if not isinstance(checkpoint_state, dict):
        raise ValueError("candidate checkpoint has no model state")

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
    }
    checkpoint_has_score_refiner = score_component_presence[
        "sacr_score_gate"
    ]
    if checkpoint_has_score_refiner and not all(
            score_component_presence.values()):
        raise ValueError(
            "candidate checkpoint has a partial SACR score refiner state: {}"
            .format(score_component_presence)
        )
    checkpoint_config_has_score_refiner = (
        config_value("use_sacr_score_refiner", False) is True
    )
    if (
            checkpoint_config_has_score_refiner
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
            )
            and checkpoint_has_source_selector):
        if getattr(args, "use_source_moe", False):
            raise ValueError(
                "selector-backed score training cannot enable SourceMoE"
            )
        args.use_source_choice_selector = True
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
def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
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
    current_state = model.state_dict()
    checkpoint_state = checkpoint['model']
    score_state_prefixes = (
        "structured_slot_builder.",
        "sacr_head.",
        "sacr_score_gate",
    )

    def canonical_state_name(name):
        return name[7:] if name.startswith("module.") else name

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
    )
    if (
            getattr(args, "use_sacr_score_refiner", False)
            and checkpoint_has_trained_score_refiner):
        if set(current_score_state) != set(checkpoint_score_state):
            raise ValueError(
                "trained SACR score checkpoint state is not exact: "
                "missing={}, unexpected={}".format(
                    sorted(set(current_score_state) - set(checkpoint_score_state)),
                    sorted(set(checkpoint_score_state) - set(current_score_state)),
                )
            )
        incompatible_score_tensors = []
        for name in sorted(current_score_state):
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
        if set(current_full_state) != set(checkpoint_full_state):
            raise ValueError(
                "trained SACR checkpoint full model state is not exact: "
                "missing={}, unexpected={}".format(
                    sorted(set(current_full_state) - set(checkpoint_full_state)),
                    sorted(set(checkpoint_full_state) - set(current_full_state)),
                )
            )
        incompatible_full_tensors = []
        for name in sorted(current_full_state):
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
            getattr(args, "use_sacr_score_refiner", False)
            and checkpoint_has_trained_score_refiner
            and (incompatible.missing_keys or incompatible.unexpected_keys)):
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
        )
        expected_missing = (
            set()
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
        )
    )
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
            if optimizer_migration is not None:
                scheduler_state = migrate_mcln_scheduler_state(
                    scheduler_state, optimizer_migration
                )
                for group, learning_rate in zip(
                        optimizer.param_groups,
                        optimizer_migration["current_lrs"]):
                    group["lr"] = learning_rate
            scheduler.load_state_dict(scheduler_state)
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
        elif gate_optimizer_resume:
            print(
                "=> resumed exact gate-only optimizer and scheduler state"
            )
        elif query_optimizer_resume:
            print(
                "=> resumed exact query-mask optimizer and scheduler state"
            )

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
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
    """Extract the five model-selection metrics from an exact eval receipt."""
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
        learned = metrics["position"]["learned_selector"]
        mask = metrics["mask"]
        hits = {
            "rec_acc025": learned["hits025"],
            "rec_acc050": learned["hits050"],
            "mask_acc025": mask["hits025"],
            "mask_acc050": mask["hits050"],
        }
        mask_miou = mask["miou"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "evaluation receipt is missing checkpoint retention metrics"
        ) from error
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


def update_checkpoint_retention(log_dir, epoch, metrics=None):
    """Retain the latest numeric checkpoint and five independent metric bests."""
    if (
        not isinstance(epoch, numbers.Integral)
        or isinstance(epoch, bool)
        or epoch < 0
    ):
        raise ValueError("checkpoint retention epoch must be non-negative")
    epoch = int(epoch)
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
    for metric_name in CHECKPOINT_RETENTION_METRICS:
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
        train_dataset, test_dataset = self.get_datasets(args)
        
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
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,      # TODO 
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
                generator=g,
                persistent_workers=(
                    args.persistent_train_workers and args.num_workers > 0
                ),
                collate_fn=(
                    joint_det_structured_collate
                    if (
                        getattr(args, "use_sacr_source", False)
                        or getattr(args, "use_sacr_score_refiner", False)
                    )
                    else None
                ),
                **multiprocessing_loader_args
            )
        
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
                if (
                    getattr(args, "use_sacr_source", False)
                    or getattr(args, "use_sacr_score_refiner", False)
                )
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
        gate_new_heads_only = getattr(
            args, "source_moe_gate_new_heads_only", False
        )
        gate_new_head_names = ()
        gate_extra_prefixes = ()
        if sum(bool(value) for value in (
                selector_only, moe_only, gate_only,
                query_mask_fusion_only, egqs_mask_refiner_only,
                joint_query_quality_only, decoder_query_adapter_only,
                sacr_score_refiner_only)) > 1:
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
                or sacr_score_refiner_only):
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
                trainable_prefixes = (
                    "structured_slot_builder",
                    "sacr_head",
                    "sacr_score_gate",
                )
                trainable_prefix = ",".join(trainable_prefixes)
                learning_rate = args.sacr_score_refiner_lr
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


    # BRIEF main training/testing
    def main(self, args):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        if not args.eval:
            n_data = len(train_loader.dataset)
            self.logger.info(f"length of training dataset: {n_data}")
        n_data = len(test_loader.dataset)
        self.logger.info(f"length of testing dataset: {n_data}")

        # Get model
        model = self.get_model(args)

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler
        if not args.eval:
            scheduler = get_scheduler(optimizer, len(train_loader), args)
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
        )
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters
        )

        # Check for a checkpoint
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
        
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
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()

            # train *
            self.train_one_epoch(
                epoch, train_loader, model,
                criterion, set_criterion,
                optimizer, scheduler, args
            )
            
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
                retention = update_checkpoint_retention(args.log_dir, epoch)
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
                            args.log_dir, epoch, metrics
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
                        args.log_dir, args.max_epoch, metrics
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
            query_mask_fusion_train_only=getattr(
                args, 'query_mask_fusion_train_only', False
            ) or getattr(args, 'egqs_mask_refiner_train_only', False),
            joint_query_quality_train_only=getattr(
                args, 'joint_query_quality_train_only', False
            ),
            sacr_score_refiner_train_only=getattr(
                args, 'sacr_score_refiner_train_only', False
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
            if ('loss' in key or 'acc' in key or 'ratio' in key or moe_scalar
                    or query_mask_fusion_scalar
                    or joint_query_quality_scalar or sacr_scalar
                    or egqs_scalar):
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
        gate_new_heads_only = getattr(
            args, "source_moe_gate_new_heads_only", False
        )
        if not (
                source_moe_only or gate_only or query_only or egqs_only
                or joint_query_only or decoder_query_adapter_only
                or sacr_score_refiner_only):
            model.train()
            return

        model.eval()
        unwrapped = model.module if hasattr(model, "module") else model
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
        self._set_source_moe_train_mode(model, args)

        # Loop over batches
        train_loader = tqdm(train_loader, ascii=True)
        for batch_idx, batch_data in enumerate(train_loader):
            # Move to GPU
            batch_data = self._to_gpu(batch_data)
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

            optimizer.zero_grad()
            loss.backward()

            self._reject_nonfinite_optimizer_gradients(
                optimizer, loss, model=model
            )

            if args.clip_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_norm
                )
                stat_dict['grad_norm'] = grad_total_norm
            
            optimizer.step()
            scheduler.step()

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

        # tensorboard
        if dist.get_rank() == 0:
            # loss
            for key in self.tensorboard.item["train_loss"]:
                self.tensorboard.item["train_loss"][key] = stat_dict[key] / len(train_loader)
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
                            stat_dict[key] / float(len(train_loader)),
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
                            stat_dict[key] / float(len(train_loader)),
                        )
                        for key in sorted(egqs_diagnostics)
                    )
                )
        return {
            "schema": TRAIN_LOSS_RECEIPT_SCHEMA,
            "batch_count": batch_count,
            "loss_means": {
                key: loss_sums[key] / float(batch_count)
                for key in sorted(loss_sums.keys())
            },
        }

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
