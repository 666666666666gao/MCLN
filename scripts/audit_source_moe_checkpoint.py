#!/usr/bin/env python3
"""Audit a SourceMoE checkpoint against its protected initializer."""

import argparse
import json
import math
from pathlib import Path

import torch


SCHEMA = "mcln-source-moe-checkpoint-audit-v1"
V20_CHANGED_PREFIXES = (
    "source_moe.fallback_gate.absolute_quality_head.",
    "source_moe.fallback_gate.cascade_quality_adapter.",
    "source_moe.fallback_gate.cascade_correction_head.",
    "source_moe.fallback_gate.cascade_opportunity_head.",
    "source_moe.fallback_gate.cascade_candidate_safety_head.",
    "source_moe.fallback_gate.cascade_joint_action_head.",
)
V20_NEW_PREFIXES = (
    "source_moe.fallback_gate.cascade_joint_action_head.",
)
V21_CHANGED_PREFIXES = (
    "source_moe.fallback_gate.cascade_fallback_set_action_head.",
)
V21_NEW_PREFIXES = V21_CHANGED_PREFIXES
V22_CHANGED_PREFIXES = (
    "source_moe.fallback_gate.cascade_rich_fallback_set_action_head.",
)
V22_NEW_PREFIXES = V22_CHANGED_PREFIXES
V23_CHANGED_PREFIXES = (
    "source_moe.adaptive_source_mixer.",
    "source_moe.fallback_gate.cascade_dense_quality_set_head.",
)
V23_NEW_PREFIXES = V23_CHANGED_PREFIXES
V24_CHANGED_PREFIXES = (
    "source_moe.adaptive_source_mixer.",
    "source_moe.fallback_gate.cascade_dense_quality_set_head.",
    "source_moe.fallback_gate.cascade_relative_risk_set_head.",
)
V24_NEW_PREFIXES = V24_CHANGED_PREFIXES
V25_CHANGED_PREFIXES = (
    "source_moe.adaptive_source_mixer.",
    "source_moe.fallback_gate.cascade_dense_quality_set_head.",
    "source_moe.fallback_gate.cascade_pairwise_calibrated_set_head.",
)
V25_NEW_PREFIXES = V25_CHANGED_PREFIXES
V26_CHANGED_PREFIXES = V25_CHANGED_PREFIXES
V26_NEW_PREFIXES = V26_CHANGED_PREFIXES
V28_CHANGED_PREFIXES = V25_CHANGED_PREFIXES + (
    "source_moe.fallback_gate.cascade_selected_abstention_head.",
)
V28_NEW_PREFIXES = V28_CHANGED_PREFIXES
V29_CHANGED_PREFIXES = V25_CHANGED_PREFIXES + (
    "source_moe.fallback_gate.cascade_counterfactual_selected_risk_head.",
)
V29_NEW_PREFIXES = V29_CHANGED_PREFIXES
V37_CHANGED_PREFIXES = V25_CHANGED_PREFIXES + (
    "source_moe.fallback_gate.cascade_counterfactual_benefit_hazard_head.",
)
V37_NEW_PREFIXES = V37_CHANGED_PREFIXES
V38_CHANGED_PREFIXES = V25_CHANGED_PREFIXES + (
    "source_moe.fallback_gate.cascade_counterfactual_logodds_head.",
)
V38_NEW_PREFIXES = V38_CHANGED_PREFIXES
V39_CHANGED_PREFIXES = V25_CHANGED_PREFIXES + (
    "source_moe.fallback_gate.cascade_counterfactual_hazard_residual_head.",
)
V39_NEW_PREFIXES = V39_CHANGED_PREFIXES
QUERY_MASK_FUSION_PREFIXES = (
    "query_mask_fusion_calibrator.",
)
EGQS_MASK_REFINER_PREFIXES = (
    "egqs_mask_refiner.",
)
JOINT_QUERY_QUALITY_PREFIXES = (
    "joint_query_quality_reranker.",
)
SACR_JOINT_QUERY_PREFIXES = JOINT_QUERY_QUALITY_PREFIXES + (
    "structured_slot_builder.",
    "sacr_head.",
    "sacr_residual_scale",
)
AUDIT_PROFILES = {
    "v20": {
        "changed_prefixes": V20_CHANGED_PREFIXES,
        "new_prefixes": V20_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 24,
        "expected_new": 6,
        "expected_optimizer_states": 30,
        "expected_optimizer_numel": 152202,
        "expected_action": "cascade_joint_risk_correction",
        "expected_objective": "cascade_joint_risk_calibrated",
    },
    "v21": {
        "changed_prefixes": V21_CHANGED_PREFIXES,
        "new_prefixes": V21_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 15,
        "expected_optimizer_states": 15,
        "expected_optimizer_numel": 149504,
        "expected_action": "cascade_v19_fallback_set_correction",
        "expected_objective": (
            "cascade_v19_fallback_set_risk_calibrated"
        ),
    },
    "v22": {
        "changed_prefixes": V22_CHANGED_PREFIXES,
        "new_prefixes": V22_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 17,
        "expected_optimizer_states": 17,
        "expected_optimizer_numel": 169264,
        "expected_action": "cascade_v19_rich_set_correction",
        "expected_objective": "cascade_v19_rich_set_empirical_risk",
    },
    "v23": {
        "changed_prefixes": V23_CHANGED_PREFIXES,
        "new_prefixes": V23_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 39,
        "expected_optimizer_states": 39,
        "expected_optimizer_numel": 588603,
        "expected_action": "cascade_v23_dense_quality_correction",
        "expected_objective": "cascade_v23_dense_quality_risk",
    },
    "v27": {
        "changed_prefixes": V23_CHANGED_PREFIXES,
        "new_prefixes": V23_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 39,
        "expected_optimizer_states": 39,
        "expected_optimizer_numel": 588603,
        "expected_action": "cascade_v23_dense_quality_correction",
        "expected_objective": "cascade_v27_uncertainty_quality_risk",
    },
    "v24": {
        "changed_prefixes": V24_CHANGED_PREFIXES,
        "new_prefixes": V24_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 60,
        "expected_optimizer_states": 60,
        "expected_optimizer_numel": 759167,
        "expected_action": "cascade_v24_relative_risk_correction",
        "expected_objective": "cascade_v24_relative_risk",
    },
    "v25": {
        "changed_prefixes": V25_CHANGED_PREFIXES,
        "new_prefixes": V25_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 69,
        "expected_optimizer_states": 69,
        "expected_optimizer_numel": 825997,
        "expected_action": "cascade_v25_pairwise_calibrated_correction",
        "expected_objective": "cascade_v25_pairwise_calibrated_risk",
    },
    "v26": {
        "changed_prefixes": V26_CHANGED_PREFIXES,
        "new_prefixes": V26_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 69,
        "expected_optimizer_states": 69,
        "expected_optimizer_numel": 825997,
        "expected_action": "cascade_v26_prior_restored_pairwise_correction",
        "expected_objective": "cascade_v26_prior_restored_pairwise_risk",
    },
    "v28": {
        "changed_prefixes": V28_CHANGED_PREFIXES,
        "new_prefixes": V28_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 75,
        "expected_optimizer_states": 75,
        "expected_optimizer_numel": 876174,
        "expected_action": "cascade_v28_selected_abstention_correction",
        "expected_objective": "cascade_v28_selected_abstention_risk",
    },
    "v29": {
        "changed_prefixes": V29_CHANGED_PREFIXES,
        "new_prefixes": V29_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 75,
        "expected_optimizer_states": 75,
        "expected_optimizer_numel": 876174,
        "expected_action": "cascade_v29_counterfactual_selected_correction",
        "expected_objective": "cascade_v29_counterfactual_selected_risk",
    },
    "v37": {
        "changed_prefixes": V37_CHANGED_PREFIXES,
        "new_prefixes": V37_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 75,
        "expected_optimizer_states": 75,
        "expected_optimizer_numel": 876303,
        "expected_action": (
            "cascade_v37_counterfactual_benefit_hazard_correction"
        ),
        "expected_objective": (
            "cascade_v37_counterfactual_benefit_hazard_risk"
        ),
    },
    "v38": {
        "changed_prefixes": V38_CHANGED_PREFIXES,
        "new_prefixes": V38_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 75,
        "expected_optimizer_states": 75,
        "expected_optimizer_numel": 876303,
        "expected_action": "cascade_v38_complementary_logodds_correction",
        "expected_objective": "cascade_v38_complementary_logodds_risk",
    },
    "v39": {
        "changed_prefixes": V39_CHANGED_PREFIXES,
        "new_prefixes": V39_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 75,
        "expected_optimizer_states": 75,
        "expected_optimizer_numel": 876303,
        "expected_action": "cascade_v39_hazard_residual_correction",
        "expected_objective": "cascade_v39_hazard_residual_risk",
    },
    "qmask": {
        "changed_prefixes": QUERY_MASK_FUSION_PREFIXES,
        "new_prefixes": QUERY_MASK_FUSION_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 12,
        "expected_optimizer_states": 12,
        "expected_optimizer_numel": 92429,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "query_mask_fusion",
    },
    "v41": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 20,
        "expected_optimizer_states": 20,
        "expected_optimizer_numel": 153531,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_quality",
    },
    "v42": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 22,
        "expected_optimizer_states": 22,
        "expected_optimizer_numel": 153919,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_mask_calibration",
        "expected_mask_alpha_delta": 1.0,
        "expected_mask_logit_bias": 2.0,
    },
    "v43": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 22,
        "expected_optimizer_states": 22,
        "expected_optimizer_numel": 155219,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_mask_evidence",
        "expected_mask_alpha_delta": 1.0,
        "expected_mask_logit_bias": 2.0,
        "expected_source_mask_evidence": True,
    },
    "v43_selector": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1078,
        "expected_changed": 0,
        "expected_new": 22,
        "expected_optimizer_states": 22,
        "expected_optimizer_numel": 155219,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_mask_evidence",
        "expected_mask_alpha_delta": 1.0,
        "expected_mask_logit_bias": 2.0,
        "expected_source_mask_evidence": True,
    },
    "v46": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 22,
        "expected_optimizer_states": 22,
        "expected_optimizer_numel": 158339,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_gate_evidence",
        "expected_mask_alpha_delta": 1.0,
        "expected_mask_logit_bias": 2.0,
        "expected_source_mask_evidence": True,
        "expected_gate_evidence": True,
    },
    "v48": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 34,
        "expected_optimizer_states": 34,
        "expected_optimizer_numel": 176979,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_spatial_mask_refinement",
        "expected_mask_alpha_delta": 1.0,
        "expected_mask_logit_bias": 2.0,
        "expected_source_mask_evidence": True,
        "expected_gate_evidence": False,
        "expected_spatial_mask_refiner": True,
        "expected_spatial_mask_hidden_dim": 32,
        "expected_max_spatial_mask_delta": 2.0,
    },
    "v105": {
        "changed_prefixes": EGQS_MASK_REFINER_PREFIXES,
        "new_prefixes": EGQS_MASK_REFINER_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 16,
        "expected_optimizer_states": 16,
        "expected_optimizer_numel": 26095,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "egqs_mask_refiner",
    },
    "v106": {
        "changed_prefixes": EGQS_MASK_REFINER_PREFIXES,
        "new_prefixes": EGQS_MASK_REFINER_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 4,
        "expected_optimizer_states": 4,
        "expected_optimizer_numel": 2888,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "graph_mask_refiner",
    },
    "v49": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 45,
        "expected_optimizer_states": 45,
        "expected_optimizer_numel": 229460,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_adaptive_source_mixing",
        "expected_mask_alpha_delta": 1.0,
        "expected_mask_logit_bias": 2.0,
        "expected_source_mask_evidence": True,
        "expected_gate_evidence": False,
        "expected_spatial_mask_refiner": True,
        "expected_spatial_mask_hidden_dim": 32,
        "expected_max_spatial_mask_delta": 2.0,
        "expected_adaptive_source_mixing": True,
        "expected_source_distribution_reliability": False,
        "expected_max_source_mix_delta": 1.0,
        "expected_source_mix_temperature": 0.5,
    },
    "v50_sacr": {
        "changed_prefixes": SACR_JOINT_QUERY_PREFIXES,
        "new_prefixes": SACR_JOINT_QUERY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 66,
        "expected_optimizer_states": 66,
        "expected_optimizer_numel": 1150390,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_sacr_adaptive_source_mixing",
        "expected_mask_alpha_delta": 1.0,
        "expected_mask_logit_bias": 2.0,
        "expected_source_mask_evidence": True,
        "expected_gate_evidence": False,
        "expected_spatial_mask_refiner": True,
        "expected_spatial_mask_hidden_dim": 32,
        "expected_max_spatial_mask_delta": 2.0,
        "expected_adaptive_source_mixing": True,
        "expected_source_distribution_reliability": False,
        "expected_max_source_mix_delta": 1.0,
        "expected_source_mix_temperature": 0.5,
        "expected_sacr": True,
        "expected_parent_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_joint_source_names": (
            "default", "contrastive_text", "mask_text", "sacr_structured",
        ),
    },
    "v51_parent_promotion": {
        "changed_prefixes": SACR_JOINT_QUERY_PREFIXES,
        "new_prefixes": SACR_JOINT_QUERY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 52,
        "expected_optimizer_states": 52,
        "expected_optimizer_numel": 1126942,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_quality",
        "expected_adaptive_source_mixing": True,
        "expected_source_distribution_reliability": False,
        "expected_max_source_mix_delta": 1.0,
        "expected_source_mix_temperature": 0.5,
        "expected_sacr": True,
        "expected_parent_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_joint_source_names": (
            "default", "contrastive_text", "mask_text", "sacr_structured",
        ),
        "expected_preserve_parent_score": True,
        "expected_candidate_promotion_margin": 0.05,
        "expected_max_delta": 0.25,
        "expected_direct_residual_scale": 0.25,
        "expected_metric_aligned_utility": False,
    },
    "v55_nested_dominance": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 22,
        "expected_optimizer_states": 18,
        "expected_optimizer_numel": 152886,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_quality",
        "expected_adaptive_source_mixing": False,
        "expected_sacr": False,
        "expected_parent_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_joint_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_preserve_parent_score": True,
        "expected_candidate_promotion_margin": 0.0,
        "expected_max_delta": 0.25,
        "expected_direct_residual_scale": 0.25,
        "expected_metric_aligned_utility": False,
        "expected_parent_transition_advantage": False,
        "expected_factorized_hit_advantage": True,
        "expected_factorized_nested_dominance": True,
        "expected_factorized_hit_break_cost": 1.0,
        "expected_parent_transition_break_cost": 4.0,
        "expected_parent_transition_candidate_top_k": 32,
    },
    "v53_factorized_hit": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 22,
        "expected_optimizer_states": 18,
        "expected_optimizer_numel": 152886,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_quality",
        "expected_adaptive_source_mixing": False,
        "expected_sacr": False,
        "expected_parent_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_joint_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_preserve_parent_score": True,
        "expected_candidate_promotion_margin": 0.05,
        "expected_max_delta": 0.25,
        "expected_direct_residual_scale": 0.25,
        "expected_metric_aligned_utility": False,
        "expected_parent_transition_advantage": False,
        "expected_factorized_hit_advantage": True,
        "expected_parent_transition_break_cost": 4.0,
        "expected_parent_transition_candidate_top_k": 32,
    },
    "v62_decomposed_transition": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 26,
        "expected_optimizer_states": 22,
        "expected_optimizer_numel": 219320,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_quality",
        "expected_adaptive_source_mixing": False,
        "expected_sacr": False,
        "expected_parent_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_joint_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_preserve_parent_score": True,
        "expected_candidate_promotion_margin": 0.0,
        "expected_max_delta": 0.25,
        "expected_direct_residual_scale": 0.25,
        "expected_metric_aligned_utility": False,
        "expected_parent_transition_advantage": False,
        "expected_decomposed_transition_advantage": True,
        "expected_factorized_hit_advantage": False,
        "expected_parent_transition_break_cost": 4.0,
        "expected_parent_transition_candidate_top_k": 32,
    },
    "v63_setwise_tier": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 25,
        "expected_optimizer_states": 21,
        "expected_optimizer_numel": 219060,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_quality",
        "expected_adaptive_source_mixing": False,
        "expected_sacr": False,
        "expected_parent_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_joint_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_preserve_parent_score": True,
        "expected_candidate_promotion_margin": 0.0,
        "expected_max_delta": 0.25,
        "expected_direct_residual_scale": 0.25,
        "expected_metric_aligned_utility": False,
        "expected_parent_transition_advantage": False,
        "expected_decomposed_transition_advantage": False,
        "expected_setwise_tier_advantage": True,
        "expected_factorized_hit_advantage": False,
        "expected_parent_transition_break_cost": 4.0,
        "expected_parent_transition_candidate_top_k": 32,
    },
    "v52_parent_transition": {
        "changed_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "new_prefixes": JOINT_QUERY_QUALITY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 26,
        "expected_optimizer_states": 22,
        "expected_optimizer_numel": 219578,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": "joint_query_quality",
        "expected_adaptive_source_mixing": False,
        "expected_sacr": False,
        "expected_parent_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_joint_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_preserve_parent_score": True,
        "expected_candidate_promotion_margin": 0.05,
        "expected_max_delta": 0.25,
        "expected_direct_residual_scale": 0.25,
        "expected_metric_aligned_utility": False,
        "expected_parent_transition_advantage": True,
        "expected_parent_transition_break_cost": 4.0,
        "expected_parent_transition_candidate_top_k": 32,
    },
    "v51_rapf_source_reliability": {
        "changed_prefixes": SACR_JOINT_QUERY_PREFIXES,
        "new_prefixes": SACR_JOINT_QUERY_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 66,
        "expected_optimizer_states": 66,
        "expected_optimizer_numel": 1151158,
        "expected_action": None,
        "expected_objective": None,
        "expected_contract": (
            "joint_query_sacr_source_distribution_reliability"
        ),
        "expected_mask_alpha_delta": 1.0,
        "expected_mask_logit_bias": 2.0,
        "expected_source_mask_evidence": True,
        "expected_gate_evidence": False,
        "expected_spatial_mask_refiner": True,
        "expected_spatial_mask_hidden_dim": 32,
        "expected_max_spatial_mask_delta": 2.0,
        "expected_adaptive_source_mixing": True,
        "expected_source_distribution_reliability": True,
        "expected_max_source_mix_delta": 1.0,
        "expected_source_mix_temperature": 0.5,
        "expected_sacr": True,
        "expected_parent_source_names": (
            "default", "contrastive_text", "mask_text",
        ),
        "expected_joint_source_names": (
            "default", "contrastive_text", "mask_text", "sacr_structured",
        ),
    },
}


def _positive_int(value, label, allow_zero=False):
    if (not isinstance(value, int) or isinstance(value, bool)
            or value < (0 if allow_zero else 1)):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError("{} must be a {} integer".format(label, qualifier))
    return value


def _strip_ddp_prefix(name):
    return name[7:] if name.startswith("module.") else name


def _matches_prefix(name, prefixes):
    normalized = _strip_ddp_prefix(name)
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _checkpoint_field(config, name):
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _require_checkpoint(checkpoint, label):
    if not isinstance(checkpoint, dict):
        raise ValueError("{} checkpoint must be a dictionary".format(label))
    model = checkpoint.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError("{} checkpoint model is invalid".format(label))
    for name, tensor in model.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(
                "{} checkpoint model must map names to tensors".format(label)
            )
    return model


def _require_finite_model(model):
    nonfinite = []
    for name, tensor in model.items():
        if (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
            if not bool(torch.isfinite(tensor).all().item()):
                nonfinite.append(name)
    if nonfinite:
        raise ValueError(
            "candidate model contains non-finite tensors: {}".format(
                ", ".join(sorted(nonfinite))
            )
        )


def _scalar_step(value, label):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not bool(torch.isfinite(value).all().item()):
            raise ValueError("{} must be a finite scalar".format(label))
        value = value.detach().cpu().item()
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(float(value), round(float(value)),
                                rel_tol=0.0, abs_tol=1e-9)
            or float(value) < 0.0):
        raise ValueError("{} must be a non-negative integer".format(label))
    return int(round(float(value)))


def _audit_optimizer(optimizer, expected_states, expected_step,
                     expected_numel, allowed_zero_parameter_ids=()):
    if (not isinstance(optimizer, dict)
            or not isinstance(optimizer.get("state"), dict)
            or not isinstance(optimizer.get("param_groups"), list)):
        raise ValueError("candidate optimizer state is invalid")
    state = optimizer["state"]
    allowed_zero_parameter_ids = set(allowed_zero_parameter_ids)
    if not all(isinstance(value, int) and value >= 0
               for value in allowed_zero_parameter_ids):
        raise ValueError("allowed zero optimizer ids must be non-negative ints")
    if len(state) != expected_states:
        raise ValueError(
            "optimizer has {} states, expected {}".format(
                len(state), expected_states
            )
        )

    group_ids = []
    for group in optimizer["param_groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise ValueError("optimizer parameter group is invalid")
        group_ids.extend(group["params"])
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("optimizer parameter ids are duplicated")
    if set(group_ids) != set(state):
        raise ValueError("optimizer states do not match parameter groups")

    steps = set()
    moment_numel = 0
    zero_second_moment_count = 0
    zero_parameter_ids = set()
    for parameter_id, values in state.items():
        if not isinstance(values, dict):
            raise ValueError("optimizer state entry is invalid")
        steps.add(_scalar_step(values.get("step"), "optimizer step"))
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = values.get(moment_name)
            if (not isinstance(moment, torch.Tensor)
                    or not torch.is_floating_point(moment)
                    or moment.numel() <= 0
                    or not bool(torch.isfinite(moment).all().item())):
                raise ValueError(
                    "optimizer {} for {} is invalid".format(
                        moment_name, parameter_id
                    )
                )
            if int(torch.count_nonzero(moment).item()) == 0:
                if parameter_id in allowed_zero_parameter_ids:
                    zero_parameter_ids.add(parameter_id)
                elif moment_name == "exp_avg_sq" and int(
                        torch.count_nonzero(values["exp_avg"]).item()
                ) > 0:
                    zero_second_moment_count += 1
                else:
                    raise ValueError(
                        "optimizer {} for {} is entirely zero".format(
                            moment_name, parameter_id
                        )
                    )
            moment_numel += moment.numel()
    if steps != {expected_step}:
        raise ValueError(
            "optimizer steps are {}, expected {}".format(
                sorted(steps), expected_step
            )
        )
    if (allowed_zero_parameter_ids
            and zero_parameter_ids != allowed_zero_parameter_ids):
        raise ValueError(
            "optimizer zero-moment ids are {}, expected inactive ids {}".format(
                sorted(zero_parameter_ids),
                sorted(allowed_zero_parameter_ids),
            )
        )
    parameter_numel = moment_numel // 2
    if moment_numel % 2 != 0 or parameter_numel != expected_numel:
        raise ValueError(
            "optimizer moment parameter numel is {}, expected {}".format(
                parameter_numel, expected_numel
            )
        )
    return {
        "state_count": len(state),
        "step": expected_step,
        "parameter_numel": parameter_numel,
        "moment_tensor_count": 2 * len(state),
        "moments_finite": True,
        "moments_nonzero": not bool(zero_parameter_ids),
        "active_moments_nonzero": True,
        "zero_second_moment_count": zero_second_moment_count,
        "allowed_zero_parameter_ids": sorted(allowed_zero_parameter_ids),
        "observed_zero_parameter_ids": sorted(zero_parameter_ids),
    }


def audit_checkpoint(
        baseline, candidate, changed_prefixes=V20_CHANGED_PREFIXES,
        new_prefixes=V20_NEW_PREFIXES, expected_common=1228,
        expected_changed=24, expected_new=6, expected_optimizer_states=30,
        expected_optimizer_step=6110, expected_optimizer_numel=152202,
        expected_epoch=2,
        expected_action="cascade_joint_risk_correction",
        expected_objective="cascade_joint_risk_calibrated",
        expected_contract="gate", expected_mask_alpha_delta=None,
        expected_mask_logit_bias=None,
        expected_source_mask_evidence=None, expected_gate_evidence=None,
        expected_spatial_mask_refiner=None,
        expected_spatial_mask_hidden_dim=None,
        expected_max_spatial_mask_delta=None,
        expected_adaptive_source_mixing=None,
        expected_source_distribution_reliability=None,
        expected_max_source_mix_delta=None,
        expected_source_mix_temperature=None,
        expected_source_mix_loss_weight=None,
        expected_source_mix_alignment_temperature=None,
        expected_source_mix_query_focus_weight=None,
        expected_sacr=None, expected_parent_source_names=None,
        expected_joint_source_names=None,
        expected_preserve_parent_score=None,
        expected_candidate_promotion_margin=None,
        expected_max_delta=None, expected_direct_residual_scale=None,
        expected_metric_aligned_utility=None,
        expected_parent_transition_advantage=None,
        expected_decomposed_transition_advantage=None,
        expected_setwise_tier_advantage=None,
        expected_factorized_hit_advantage=None,
        expected_factorized_nested_dominance=None,
        expected_factorized_hit_break_cost=None,
        expected_parent_transition_break_cost=None,
        expected_parent_transition_candidate_top_k=None):
    """Return an exact initializer-to-candidate checkpoint contract audit."""
    for value, label in (
            (expected_common, "expected common tensor count"),
            (expected_changed, "expected changed tensor count"),
            (expected_new, "expected new tensor count"),
            (expected_optimizer_states, "expected optimizer state count"),
            (expected_optimizer_numel, "expected optimizer parameter numel"),
            (expected_epoch, "expected checkpoint epoch")):
        _positive_int(
            value,
            label,
            allow_zero=(
                label.endswith("tensor count")
                or label == "expected checkpoint epoch"
            ),
        )
    _positive_int(
        expected_optimizer_step, "expected optimizer step", allow_zero=True
    )
    if expected_contract not in (
            "gate", "query_mask_fusion", "egqs_mask_refiner",
            "graph_mask_refiner",
            "joint_query_quality",
            "joint_query_mask_calibration", "joint_query_mask_evidence",
            "joint_query_gate_evidence",
            "joint_query_spatial_mask_refinement",
            "joint_query_adaptive_source_mixing",
            "joint_query_sacr_adaptive_source_mixing",
            "joint_query_sacr_source_distribution_reliability"):
        raise ValueError(
            "expected contract must be gate, query_mask_fusion, "
            "egqs_mask_refiner, graph_mask_refiner, or "
            "joint_query_quality/joint_query_mask_calibration/"
            "joint_query_mask_evidence/joint_query_gate_evidence/"
            "joint_query_spatial_mask_refinement/"
            "joint_query_adaptive_source_mixing/"
            "joint_query_sacr_adaptive_source_mixing/"
            "joint_query_sacr_source_distribution_reliability"
        )
    if (not isinstance(changed_prefixes, (tuple, list))
            or not changed_prefixes
            or not all(isinstance(value, str) and value
                       for value in changed_prefixes)):
        raise ValueError("changed prefixes must be non-empty strings")
    if (not isinstance(new_prefixes, (tuple, list))
            or not new_prefixes
            or not all(isinstance(value, str) and value
                       for value in new_prefixes)):
        raise ValueError("new prefixes must be non-empty strings")

    baseline_model = _require_checkpoint(baseline, "baseline")
    candidate_model = _require_checkpoint(candidate, "candidate")
    _require_finite_model(candidate_model)

    baseline_names = set(baseline_model)
    candidate_names = set(candidate_model)
    missing = sorted(baseline_names - candidate_names)
    if missing:
        raise ValueError(
            "candidate checkpoint is missing baseline tensors: {}".format(
                ", ".join(missing)
            )
        )
    common = sorted(baseline_names & candidate_names)
    new = sorted(candidate_names - baseline_names)
    changed = sorted(
        name for name in common
        if not torch.equal(baseline_model[name], candidate_model[name])
    )
    changed_outside = sorted(
        name for name in changed
        if not _matches_prefix(name, changed_prefixes)
    )
    new_outside = sorted(
        name for name in new if not _matches_prefix(name, new_prefixes)
    )
    if changed_outside:
        raise ValueError(
            "model tensors changed outside the allowlist: {}".format(
                ", ".join(changed_outside)
            )
        )
    if new_outside:
        raise ValueError(
            "model tensors were added outside the allowlist: {}".format(
                ", ".join(new_outside)
            )
        )
    observed = (len(common), len(changed), len(new))
    expected = (expected_common, expected_changed, expected_new)
    if observed != expected:
        raise ValueError(
            "model tensor counts are common/changed/new {}, expected {}".format(
                observed, expected
            )
        )

    if candidate.get("epoch") != expected_epoch:
        raise ValueError(
            "candidate epoch is {}, expected {}".format(
                candidate.get("epoch"), expected_epoch
            )
        )
    config = candidate.get("config")
    allowed_zero_optimizer_ids = ()
    if expected_contract == "gate":
        if (_checkpoint_field(config, "source_moe_gate_action_mode")
                != expected_action):
            raise ValueError("candidate action mode is incompatible")
        if (_checkpoint_field(config, "source_moe_gate_objective")
                != expected_objective):
            raise ValueError("candidate gate objective is incompatible")
        if _checkpoint_field(config, "source_moe_gate_train_only") is not True:
            raise ValueError("candidate must use gate-only training")
        if (_checkpoint_field(config, "source_moe_gate_new_heads_only")
                is not True):
            raise ValueError("candidate must use new-head-only training")
        contract = {
            "action": expected_action,
            "objective": expected_objective,
            "gate_only": True,
            "new_heads_only": True,
        }
    elif expected_contract == "query_mask_fusion":
        if (_checkpoint_field(config, "use_query_mask_fusion_calibrator")
                is not True):
            raise ValueError("candidate must enable query mask fusion")
        if (_checkpoint_field(config, "query_mask_fusion_train_only")
                is not True):
            raise ValueError("candidate must use query-mask-only training")
        contract = {
            "query_mask_fusion": True,
            "query_mask_fusion_only": True,
        }
    elif expected_contract == "egqs_mask_refiner":
        if (_checkpoint_field(config, "use_egqs_mask_refiner") is not True):
            raise ValueError("candidate must enable the EGQS mask refiner")
        if (_checkpoint_field(
                config, "egqs_mask_refiner_train_only") is not True):
            raise ValueError("candidate must use EGQS-mask-only training")
        components = _checkpoint_field(
            config, "egqs_mask_refiner_components"
        )
        if components not in ("content", "evidence", "geometry", "all"):
            raise ValueError("candidate has invalid EGQS components")
        if (_checkpoint_field(config, "egqs_mask_refiner_hidden_dim") != 32
                or _checkpoint_field(
                    config, "egqs_mask_refiner_max_delta"
                ) != 2.0):
            raise ValueError("candidate EGQS architecture is incompatible")
        if components == "content":
            allowed_zero_optimizer_ids = tuple(range(12, 16))
        elif components == "evidence":
            allowed_zero_optimizer_ids = tuple(range(2, 12)) + (14, 15)
        elif components == "geometry":
            allowed_zero_optimizer_ids = tuple(range(2, 14))
        contract = {
            "egqs_mask_refiner": True,
            "egqs_mask_refiner_only": True,
            "components": components,
            "hidden_dim": 32,
            "max_delta": 2.0,
        }
    elif expected_contract == "graph_mask_refiner":
        if (_checkpoint_field(config, "use_egqs_mask_refiner") is not True):
            raise ValueError("candidate must enable the graph mask refiner")
        if (_checkpoint_field(
                config, "egqs_mask_refiner_train_only") is not True):
            raise ValueError("candidate must use graph-mask-only training")
        architecture = _checkpoint_field(config, "egqs_mask_refiner_arch")
        graph_mode = _checkpoint_field(
            config, "egqs_mask_refiner_graph_mode"
        )
        if architecture != "graph" or graph_mode not in (
                "spatial", "bilateral"):
            raise ValueError("candidate graph architecture is incompatible")
        if (_checkpoint_field(
                config, "egqs_mask_refiner_neighbor_count") != 8
                or _checkpoint_field(
                    config, "egqs_mask_refiner_max_delta") != 2.0):
            raise ValueError("candidate graph contract is incompatible")
        contract = {
            "graph_mask_refiner": True,
            "graph_mask_refiner_only": True,
            "graph_mode": graph_mode,
            "neighbor_count": 8,
            "max_delta": 2.0,
        }
    else:
        if (_checkpoint_field(config, "use_joint_query_quality_reranker")
                is not True):
            raise ValueError("candidate must enable joint query quality")
        if (_checkpoint_field(config, "joint_query_quality_train_only")
                is not True):
            raise ValueError(
                "candidate must use joint-query-quality-only training"
            )
        mask_calibration = _checkpoint_field(
            config, "joint_query_quality_use_mask_calibration"
        ) is True
        adaptive_contracts = (
            "joint_query_adaptive_source_mixing",
            "joint_query_sacr_adaptive_source_mixing",
            "joint_query_sacr_source_distribution_reliability",
        )
        if (expected_contract in (
                "joint_query_mask_calibration", "joint_query_mask_evidence",
                "joint_query_gate_evidence",
                "joint_query_spatial_mask_refinement",
                *adaptive_contracts)
                and not mask_calibration):
            raise ValueError(
                "candidate must enable joint query mask calibration"
            )
        source_mask_evidence = _checkpoint_field(
            config, "joint_query_quality_use_source_mask_evidence"
        ) is True
        gate_evidence = _checkpoint_field(
            config, "joint_query_quality_use_gate_evidence"
        ) is True
        spatial_mask_refiner = _checkpoint_field(
            config, "joint_query_quality_use_spatial_mask_refiner"
        ) is True
        observed_spatial_hidden_dim = _checkpoint_field(
            config, "joint_query_quality_spatial_mask_hidden_dim"
        )
        observed_spatial_delta = _checkpoint_field(
            config, "joint_query_quality_max_spatial_mask_delta"
        )
        adaptive_source_mixing = _checkpoint_field(
            config, "joint_query_quality_use_adaptive_source_mixing"
        ) is True
        source_distribution_reliability = _checkpoint_field(
            config,
            "joint_query_quality_use_source_distribution_reliability",
        ) is True
        observed_source_mix_delta = _checkpoint_field(
            config, "joint_query_quality_max_source_mix_delta"
        )
        observed_source_mix_temperature = _checkpoint_field(
            config, "joint_query_quality_source_mix_temperature"
        )
        observed_source_mix_loss_weight = _checkpoint_field(
            config, "joint_query_quality_source_mix_loss_weight"
        )
        observed_source_mix_alignment_temperature = _checkpoint_field(
            config,
            "joint_query_quality_source_mix_alignment_temperature",
        )
        observed_source_mix_query_focus_weight = _checkpoint_field(
            config,
            "joint_query_quality_source_mix_query_focus_weight",
        )
        observed_preserve_parent_score = _checkpoint_field(
            config, "joint_query_quality_preserve_parent_score"
        ) is True
        observed_candidate_promotion_margin = _checkpoint_field(
            config, "joint_query_quality_candidate_promotion_margin"
        )
        observed_max_delta = _checkpoint_field(
            config, "joint_query_quality_max_delta"
        )
        observed_direct_residual_scale = _checkpoint_field(
            config, "joint_query_quality_direct_residual_scale"
        )
        observed_metric_aligned_utility = _checkpoint_field(
            config, "joint_query_quality_use_metric_aligned_utility"
        ) is True
        observed_parent_transition_advantage = _checkpoint_field(
            config, "joint_query_quality_use_parent_transition_advantage"
        ) is True
        observed_decomposed_transition_advantage = _checkpoint_field(
            config,
            "joint_query_quality_use_decomposed_transition_advantage",
        ) is True
        observed_setwise_tier_advantage = _checkpoint_field(
            config, "joint_query_quality_use_setwise_tier_advantage"
        ) is True
        observed_factorized_hit_advantage = _checkpoint_field(
            config, "joint_query_quality_use_factorized_hit_advantage"
        ) is True
        observed_factorized_nested_dominance = _checkpoint_field(
            config,
            "joint_query_quality_use_factorized_nested_dominance",
        ) is True
        observed_factorized_hit_break_cost = _checkpoint_field(
            config, "joint_query_quality_factorized_hit_break_cost"
        )
        observed_parent_transition_break_cost = _checkpoint_field(
            config, "joint_query_quality_parent_transition_break_cost"
        )
        observed_parent_transition_candidate_top_k = _checkpoint_field(
            config,
            "joint_query_quality_parent_transition_candidate_top_k",
        )
        if expected_contract in (
                "joint_query_mask_calibration", "joint_query_mask_evidence",
                "joint_query_gate_evidence",
                "joint_query_spatial_mask_refinement",
                *adaptive_contracts):
            observed_alpha_delta = _checkpoint_field(
                config, "joint_query_quality_max_mask_alpha_delta"
            )
            observed_logit_bias = _checkpoint_field(
                config, "joint_query_quality_max_mask_logit_bias"
            )
            if (expected_mask_alpha_delta is not None
                    and observed_alpha_delta != expected_mask_alpha_delta):
                raise ValueError(
                    "candidate mask alpha delta contract is incompatible"
                )
            if (expected_mask_logit_bias is not None
                    and observed_logit_bias != expected_mask_logit_bias):
                raise ValueError(
                    "candidate mask logit bias contract is incompatible"
                )
        if (expected_source_mask_evidence is not None
                and source_mask_evidence is not expected_source_mask_evidence):
            raise ValueError(
                "candidate source mask evidence contract is incompatible"
            )
        if (expected_gate_evidence is not None
                and gate_evidence is not expected_gate_evidence):
            raise ValueError(
                "candidate gate evidence contract is incompatible"
            )
        if (expected_spatial_mask_refiner is not None
                and spatial_mask_refiner is not expected_spatial_mask_refiner):
            raise ValueError(
                "candidate spatial mask refiner contract is incompatible"
            )
        if (expected_contract == "joint_query_mask_calibration"
                and source_mask_evidence):
            raise ValueError(
                "V42 mask calibration must not enable source mask evidence"
            )
        if (expected_contract == "joint_query_mask_evidence"
                and not source_mask_evidence):
            raise ValueError(
                "candidate must enable source-specific mask evidence"
            )
        if expected_contract == "joint_query_gate_evidence" and (
                not source_mask_evidence or not gate_evidence):
            raise ValueError(
                "candidate must enable source-mask and gate evidence"
            )
        if expected_contract == "joint_query_spatial_mask_refinement":
            if not source_mask_evidence or not spatial_mask_refiner:
                raise ValueError(
                    "candidate must enable source-mask evidence and spatial "
                    "mask refinement"
                )
            if (expected_spatial_mask_hidden_dim is not None
                    and observed_spatial_hidden_dim
                    != expected_spatial_mask_hidden_dim):
                raise ValueError(
                    "candidate spatial mask hidden dimension is incompatible"
                )
            if (expected_max_spatial_mask_delta is not None
                    and observed_spatial_delta != expected_max_spatial_mask_delta):
                raise ValueError(
                    "candidate spatial mask delta contract is incompatible"
                )
        if expected_contract in adaptive_contracts:
            if (not source_mask_evidence or not spatial_mask_refiner
                    or not adaptive_source_mixing):
                raise ValueError(
                    "candidate must enable source-mask evidence, spatial "
                    "mask refinement, and adaptive source mixing"
                )
        observed_sacr = _checkpoint_field(config, "use_sacr_source") is True
        if expected_sacr is not None and observed_sacr is not expected_sacr:
            raise ValueError("candidate SACR source contract is incompatible")

        def normalized_source_names(value):
            if isinstance(value, str):
                return tuple(
                    item.strip() for item in value.split(",") if item.strip()
                )
            if isinstance(value, (tuple, list)):
                return tuple(str(item) for item in value)
            return ()

        observed_parent_sources = normalized_source_names(
            _checkpoint_field(config, "source_choice_selector_sources")
        )
        observed_joint_sources = normalized_source_names(
            _checkpoint_field(config, "joint_query_quality_source_names")
        )
        # Match MCLN's runtime source-pool resolution: an empty joint pool
        # inherits the parent selector sources (models/mcln.py).
        if not observed_joint_sources:
            observed_joint_sources = observed_parent_sources
        if (expected_parent_source_names is not None
                and observed_parent_sources
                != tuple(expected_parent_source_names)):
            raise ValueError("candidate parent source pool is incompatible")
        if (expected_joint_source_names is not None
                and observed_joint_sources != tuple(expected_joint_source_names)):
            raise ValueError("candidate joint source pool is incompatible")
        if expected_contract == "joint_query_sacr_adaptive_source_mixing":
            if (not observed_sacr
                    or "sacr_structured" not in observed_joint_sources
                    or "sacr_structured" in observed_parent_sources):
                raise ValueError(
                    "SACR must extend only the joint adaptive source pool"
                )
        if (expected_adaptive_source_mixing is not None
                and adaptive_source_mixing
                is not expected_adaptive_source_mixing):
            raise ValueError(
                "candidate adaptive source mixing contract is incompatible"
            )
        if (expected_max_source_mix_delta is not None
                and observed_source_mix_delta
                != expected_max_source_mix_delta):
            raise ValueError(
                "candidate source mix delta contract is incompatible"
            )
        if (expected_source_mix_temperature is not None
                and observed_source_mix_temperature
                != expected_source_mix_temperature):
            raise ValueError(
                "candidate source mix temperature contract is incompatible"
            )
        if (expected_source_distribution_reliability is not None
                and source_distribution_reliability
                is not expected_source_distribution_reliability):
            raise ValueError(
                "candidate source distribution reliability contract is "
                "incompatible"
            )
        if (expected_source_mix_loss_weight is not None
                and observed_source_mix_loss_weight
                != expected_source_mix_loss_weight):
            raise ValueError(
                "candidate source mix loss weight contract is incompatible"
            )
        if (expected_source_mix_alignment_temperature is not None
                and observed_source_mix_alignment_temperature
                != expected_source_mix_alignment_temperature):
            raise ValueError(
                "candidate source mix alignment temperature contract is "
                "incompatible"
            )
        if (expected_source_mix_query_focus_weight is not None
                and observed_source_mix_query_focus_weight
                != expected_source_mix_query_focus_weight):
            raise ValueError(
                "candidate source mix query focus weight contract is "
                "incompatible"
            )
        for observed_value, expected_value, label in (
                (observed_preserve_parent_score,
                 expected_preserve_parent_score,
                 "parent score preservation"),
                (observed_candidate_promotion_margin,
                 expected_candidate_promotion_margin,
                 "candidate promotion margin"),
                (observed_max_delta, expected_max_delta,
                 "joint query max delta"),
                (observed_direct_residual_scale,
                 expected_direct_residual_scale,
                 "direct residual scale"),
                (observed_metric_aligned_utility,
                 expected_metric_aligned_utility,
                 "metric aligned utility"),
                (observed_parent_transition_advantage,
                 expected_parent_transition_advantage,
                 "parent transition advantage"),
                (observed_decomposed_transition_advantage,
                 expected_decomposed_transition_advantage,
                 "decomposed transition advantage"),
                (observed_setwise_tier_advantage,
                 expected_setwise_tier_advantage,
                 "setwise tier advantage"),
                (observed_factorized_hit_advantage,
                 expected_factorized_hit_advantage,
                 "factorized hit advantage"),
                (observed_factorized_nested_dominance,
                 expected_factorized_nested_dominance,
                 "factorized nested dominance"),
                (observed_factorized_hit_break_cost,
                 expected_factorized_hit_break_cost,
                 "factorized hit break cost"),
                (observed_parent_transition_break_cost,
                 expected_parent_transition_break_cost,
                 "parent transition break cost"),
                (observed_parent_transition_candidate_top_k,
                 expected_parent_transition_candidate_top_k,
                 "parent transition candidate top k")):
            if (expected_value is not None
                    and observed_value != expected_value):
                raise ValueError(
                    "candidate {} contract is incompatible".format(label)
                )
        if (expected_contract == "joint_query_quality" and mask_calibration):
            raise ValueError(
                "plain joint query quality candidate must not enable mask "
                "calibration"
            )
        contract = {
            "joint_query_quality": True,
            "joint_query_quality_only": True,
            "joint_query_mask_calibration": mask_calibration,
        }
        if (expected_preserve_parent_score is not None
                or expected_candidate_promotion_margin is not None
                or expected_max_delta is not None
                or expected_direct_residual_scale is not None
                or expected_metric_aligned_utility is not None
                or observed_preserve_parent_score
                or observed_candidate_promotion_margin not in (None, 0.0)):
            contract.update({
                "joint_query_preserve_parent_score": (
                    observed_preserve_parent_score
                ),
                "joint_query_candidate_promotion_margin": (
                    observed_candidate_promotion_margin
                ),
                "joint_query_max_delta": observed_max_delta,
                "joint_query_direct_residual_scale": (
                    observed_direct_residual_scale
                ),
                "joint_query_metric_aligned_utility": (
                    observed_metric_aligned_utility
                ),
            })
        if (expected_parent_transition_advantage is not None
                or observed_parent_transition_advantage
                or expected_decomposed_transition_advantage is not None
                or observed_decomposed_transition_advantage
                or expected_setwise_tier_advantage is not None
                or observed_setwise_tier_advantage
                or expected_factorized_hit_advantage is not None
                or observed_factorized_hit_advantage
                or expected_factorized_nested_dominance is not None
                or observed_factorized_nested_dominance):
            contract.update({
                "joint_query_parent_transition_advantage": (
                    observed_parent_transition_advantage
                ),
                "joint_query_decomposed_transition_advantage": (
                    observed_decomposed_transition_advantage
                ),
                "joint_query_setwise_tier_advantage": (
                    observed_setwise_tier_advantage
                ),
                "joint_query_factorized_hit_advantage": (
                    observed_factorized_hit_advantage
                ),
                "joint_query_factorized_nested_dominance": (
                    observed_factorized_nested_dominance
                ),
                "joint_query_factorized_hit_break_cost": (
                    observed_factorized_hit_break_cost
                ),
                "joint_query_parent_transition_break_cost": (
                    observed_parent_transition_break_cost
                ),
                "joint_query_parent_transition_candidate_top_k": (
                    observed_parent_transition_candidate_top_k
                ),
            })
        if expected_gate_evidence is not None or gate_evidence:
            contract["joint_query_gate_evidence"] = gate_evidence
        if expected_spatial_mask_refiner is not None or spatial_mask_refiner:
            contract["joint_query_spatial_mask_refiner"] = (
                spatial_mask_refiner
            )
        if mask_calibration:
            contract.update({
                "joint_query_source_mask_evidence": source_mask_evidence,
                "joint_query_mask_alpha_delta": observed_alpha_delta,
                "joint_query_mask_logit_bias": observed_logit_bias,
            })
        if spatial_mask_refiner:
            contract.update({
                "joint_query_spatial_mask_hidden_dim": (
                    observed_spatial_hidden_dim
                ),
                "joint_query_max_spatial_mask_delta": observed_spatial_delta,
            })
        if adaptive_source_mixing:
            contract.update({
                "joint_query_adaptive_source_mixing": True,
                "joint_query_max_source_mix_delta": (
                    observed_source_mix_delta
                ),
                "joint_query_source_distribution_reliability": (
                    source_distribution_reliability
                ),
                "joint_query_source_mix_temperature": (
                    observed_source_mix_temperature
                ),
                "joint_query_source_mix_loss_weight": (
                    observed_source_mix_loss_weight
                ),
                "joint_query_source_mix_alignment_temperature": (
                    observed_source_mix_alignment_temperature
                ),
                "joint_query_source_mix_query_focus_weight": (
                    observed_source_mix_query_focus_weight
                ),
            })
        if observed_sacr:
            contract.update({
                "sacr_source": True,
                "parent_source_names": list(observed_parent_sources),
                "joint_source_names": list(observed_joint_sources),
            })

    optimizer = _audit_optimizer(
        candidate.get("optimizer"),
        expected_states=expected_optimizer_states,
        expected_step=expected_optimizer_step,
        expected_numel=expected_optimizer_numel,
        allowed_zero_parameter_ids=allowed_zero_optimizer_ids,
    )
    return {
        "schema": SCHEMA,
        "checkpoint_epoch": expected_epoch,
        "model": {
            "common_tensor_count": len(common),
            "changed_tensor_count": len(changed),
            "new_tensor_count": len(new),
            "changed_tensors": changed,
            "new_tensors": new,
            "finite": True,
            "allowlist_pass": True,
        },
        "optimizer": optimizer,
        "contract": contract,
        "pass": True,
    }


def _load_checkpoint(path):
    return torch.load(str(path), map_location="cpu")


def _atomic_write_json(path, value):
    output = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--profile", choices=tuple(AUDIT_PROFILES), default="v20"
    )
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-common", type=int)
    parser.add_argument("--expected-changed", type=int)
    parser.add_argument("--expected-new", type=int)
    parser.add_argument("--expected-optimizer-states", type=int)
    parser.add_argument("--expected-optimizer-numel", type=int)
    parser.add_argument("--expected-source-mix-loss-weight", type=float)
    parser.add_argument(
        "--expected-source-mix-alignment-temperature", type=float
    )
    parser.add_argument(
        "--expected-source-mix-query-focus-weight", type=float
    )
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    profile = AUDIT_PROFILES[args.profile]

    def expected(name):
        override = getattr(args, name)
        return profile[name] if override is None else override

    result = audit_checkpoint(
        _load_checkpoint(args.baseline),
        _load_checkpoint(args.checkpoint),
        changed_prefixes=profile["changed_prefixes"],
        new_prefixes=profile["new_prefixes"],
        expected_common=expected("expected_common"),
        expected_changed=expected("expected_changed"),
        expected_new=expected("expected_new"),
        expected_optimizer_states=expected("expected_optimizer_states"),
        expected_optimizer_step=args.expected_step,
        expected_optimizer_numel=expected("expected_optimizer_numel"),
        expected_epoch=args.expected_epoch,
        expected_action=profile["expected_action"],
        expected_objective=profile["expected_objective"],
        expected_contract=profile.get("expected_contract", "gate"),
        expected_mask_alpha_delta=profile.get("expected_mask_alpha_delta"),
        expected_mask_logit_bias=profile.get("expected_mask_logit_bias"),
        expected_source_mask_evidence=profile.get(
            "expected_source_mask_evidence"
        ),
        expected_gate_evidence=profile.get("expected_gate_evidence"),
        expected_spatial_mask_refiner=profile.get(
            "expected_spatial_mask_refiner"
        ),
        expected_spatial_mask_hidden_dim=profile.get(
            "expected_spatial_mask_hidden_dim"
        ),
        expected_max_spatial_mask_delta=profile.get(
            "expected_max_spatial_mask_delta"
        ),
        expected_adaptive_source_mixing=profile.get(
            "expected_adaptive_source_mixing"
        ),
        expected_source_distribution_reliability=profile.get(
            "expected_source_distribution_reliability"
        ),
        expected_max_source_mix_delta=profile.get(
            "expected_max_source_mix_delta"
        ),
        expected_source_mix_temperature=profile.get(
            "expected_source_mix_temperature"
        ),
        expected_source_mix_loss_weight=(
            args.expected_source_mix_loss_weight
        ),
        expected_source_mix_alignment_temperature=(
            args.expected_source_mix_alignment_temperature
        ),
        expected_source_mix_query_focus_weight=(
            args.expected_source_mix_query_focus_weight
        ),
        expected_sacr=profile.get("expected_sacr"),
        expected_parent_source_names=profile.get(
            "expected_parent_source_names"
        ),
        expected_joint_source_names=profile.get(
            "expected_joint_source_names"
        ),
        expected_preserve_parent_score=profile.get(
            "expected_preserve_parent_score"
        ),
        expected_candidate_promotion_margin=profile.get(
            "expected_candidate_promotion_margin"
        ),
        expected_max_delta=profile.get("expected_max_delta"),
        expected_direct_residual_scale=profile.get(
            "expected_direct_residual_scale"
        ),
        expected_metric_aligned_utility=profile.get(
            "expected_metric_aligned_utility"
        ),
        expected_parent_transition_advantage=profile.get(
            "expected_parent_transition_advantage"
        ),
        expected_decomposed_transition_advantage=profile.get(
            "expected_decomposed_transition_advantage"
        ),
        expected_setwise_tier_advantage=profile.get(
            "expected_setwise_tier_advantage"
        ),
        expected_factorized_hit_advantage=profile.get(
            "expected_factorized_hit_advantage"
        ),
        expected_factorized_nested_dominance=profile.get(
            "expected_factorized_nested_dominance"
        ),
        expected_factorized_hit_break_cost=profile.get(
            "expected_factorized_hit_break_cost"
        ),
        expected_parent_transition_break_cost=profile.get(
            "expected_parent_transition_break_cost"
        ),
        expected_parent_transition_candidate_top_k=profile.get(
            "expected_parent_transition_candidate_top_k"
        ),
    )
    result["profile"] = args.profile
    if args.output:
        _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
