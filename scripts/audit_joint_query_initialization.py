#!/usr/bin/env python3
"""Audit joint-query initialization against a protected MCLN checkpoint."""

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
for import_path in (ROOT_DIR, ROOT_DIR / "pointnet2"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from train_dist_mod import TrainTester


PROFILES = {
    "v41": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "expected_missing": 20,
        "expected_parameters": 153531,
    },
    "v42": {
        "mask_calibration": True,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "expected_missing": 22,
        "expected_parameters": 153919,
    },
    "v43": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "expected_missing": 22,
        "expected_parameters": 155219,
    },
    "v46": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": True,
        "expected_missing": 22,
        "expected_parameters": 158339,
    },
    "v48": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "spatial_mask_refiner": True,
        "spatial_mask_hidden_dim": 32,
        "max_spatial_mask_delta": 2.0,
        "expected_missing": 34,
        "expected_parameters": 176979,
    },
    "v49": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "spatial_mask_refiner": True,
        "spatial_mask_hidden_dim": 32,
        "max_spatial_mask_delta": 2.0,
        "adaptive_source_mixing": True,
        "source_count": 3,
        "shared_source_index": 0,
        "max_source_mix_delta": 1.0,
        "source_mix_temperature": 0.5,
        "expected_missing": 45,
        "expected_parameters": 229460,
    },
    "v50_sacr": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "spatial_mask_refiner": True,
        "spatial_mask_hidden_dim": 32,
        "max_spatial_mask_delta": 2.0,
        "adaptive_source_mixing": True,
        "source_count": 4,
        "shared_source_index": 0,
        "max_source_mix_delta": 1.0,
        "source_mix_temperature": 0.5,
        "joint_source_names": (
            "default,contrastive_text,mask_text,sacr_structured"
        ),
        "use_sacr_source": True,
        "expected_missing": 66,
        "expected_parameters": 1150390,
    },
    "v51_bmq_rank": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "spatial_mask_refiner": True,
        "spatial_mask_hidden_dim": 32,
        "max_spatial_mask_delta": 2.0,
        "adaptive_source_mixing": True,
        "source_count": 4,
        "shared_source_index": 0,
        "max_source_mix_delta": 1.0,
        "source_mix_temperature": 0.5,
        "joint_source_names": (
            "default,contrastive_text,mask_text,sacr_structured"
        ),
        "use_sacr_source": True,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": True,
        "expected_missing": 66,
        "expected_parameters": 1150390,
    },
    "v51_parent_promotion": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": True,
        "source_count": 4,
        "shared_source_index": 0,
        "max_source_mix_delta": 1.0,
        "source_mix_temperature": 0.5,
        "joint_source_names": (
            "default,contrastive_text,mask_text,sacr_structured"
        ),
        "use_sacr_source": True,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.05,
        "expected_missing": 52,
        "expected_parameters": 1126942,
    },
    "v56_parent_relative_rank": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "factorized_hit_advantage": True,
        "factorized_nested_dominance": True,
        "factorized_hit_break_cost": 1.0,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 22,
        "expected_parameters": 153789,
    },
    "v55_nested_dominance": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "factorized_hit_advantage": True,
        "factorized_nested_dominance": True,
        "factorized_hit_break_cost": 1.0,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 22,
        "expected_parameters": 153789,
    },
    "v54_factorized_rank": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "factorized_hit_advantage": True,
        "factorized_hit_break_cost": 1.0,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 22,
        "expected_parameters": 153789,
    },
    "v53_factorized_hit": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.05,
        "factorized_hit_advantage": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 22,
        "expected_parameters": 153789,
    },
    "v62_decomposed_transition": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "decomposed_transition_advantage": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 26,
        "expected_parameters": 220223,
    },
    "v63_setwise_tier": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 25,
        "expected_parameters": 219963,
    },
    "v69_decoupled_setwise": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286139,
    },
    "v72_factorized_setwise_safety": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286395,
    },
    "v73_factorized_setwise_risk_bound": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v74_setwise_safety_veto_gate": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v75_cost_calibrated_risk_bound": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "cost_calibrated_setwise_risk_bound": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v76_safety_slack_quantile_bound": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v77_safety_slack_pairwise_order": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v78_proposal_conditioned_safety": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "proposal_conditioned_safety": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v79_parent_referenced_safety": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v80_coupled_safe_repair_witness": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "coupled_safe_repair_witness": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v81_bidirectional_coupled_boundary": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "coupled_safe_repair_witness": True,
        "bidirectional_coupled_boundary": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v82_centered_coupled_separation": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "coupled_safe_repair_witness": True,
        "bidirectional_coupled_boundary": True,
        "centered_coupled_separation": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v83_hazard_conditioned_coupled_separation": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "coupled_safe_repair_witness": True,
        "bidirectional_coupled_boundary": True,
        "centered_coupled_separation": True,
        "hazard_conditioned_coupled_separation": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v84_monotonic_box_safety_folding": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "coupled_safe_repair_witness": True,
        "bidirectional_coupled_boundary": True,
        "centered_coupled_separation": True,
        "hazard_conditioned_coupled_separation": True,
        "monotonic_box_safety_folding": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v85_same_candidate_branchwise_witness": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "coupled_safe_repair_witness": True,
        "bidirectional_coupled_boundary": True,
        "centered_coupled_separation": True,
        "hazard_conditioned_coupled_separation": True,
        "monotonic_box_safety_folding": True,
        "same_candidate_branchwise_witness": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v86_parent_non_degradation_certificate": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "coupled_safe_repair_witness": True,
        "bidirectional_coupled_boundary": True,
        "centered_coupled_separation": True,
        "hazard_conditioned_coupled_separation": True,
        "monotonic_box_safety_folding": True,
        "same_candidate_branchwise_witness": True,
        "parent_non_degradation_certificate": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v87_criterion_responsible_hazard_attribution": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "coupled_safe_repair_witness": True,
        "bidirectional_coupled_boundary": True,
        "centered_coupled_separation": True,
        "hazard_conditioned_coupled_separation": True,
        "monotonic_box_safety_folding": True,
        "same_candidate_branchwise_witness": True,
        "parent_non_degradation_certificate": True,
        "criterion_responsible_hazard_attribution": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 30,
        "expected_parameters": 286779,
    },
    "v88_independent_joint_hazard_certificate": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "coupled_safe_repair_witness": True,
        "bidirectional_coupled_boundary": True,
        "centered_coupled_separation": True,
        "hazard_conditioned_coupled_separation": True,
        "monotonic_box_safety_folding": True,
        "same_candidate_branchwise_witness": True,
        "parent_non_degradation_certificate": True,
        "independent_joint_hazard_certificate": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 35,
        "expected_parameters": 353083,
    },
    "v89_frozen_raw_joint_hazard_features": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.0,
        "setwise_tier_advantage": True,
        "decoupled_setwise_heads": True,
        "factorized_setwise_safety": True,
        "factorized_setwise_risk_bound": True,
        "setwise_safety_veto_gate": True,
        "safety_slack_quantile_bound": True,
        "safety_slack_pairwise_order": True,
        "parent_referenced_safety": True,
        "coupled_safe_repair_witness": True,
        "bidirectional_coupled_boundary": True,
        "centered_coupled_separation": True,
        "hazard_conditioned_coupled_separation": True,
        "monotonic_box_safety_folding": True,
        "same_candidate_branchwise_witness": True,
        "parent_non_degradation_certificate": True,
        "independent_joint_hazard_certificate": True,
        "frozen_raw_joint_hazard_features": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 35,
        "expected_parameters": 365371,
    },
    "v52_parent_transition": {
        "mask_calibration": False,
        "source_mask_evidence": False,
        "gate_evidence": False,
        "adaptive_source_mixing": False,
        "use_sacr_source": False,
        "max_delta": 0.25,
        "direct_residual_scale": 0.25,
        "metric_aligned_utility": False,
        "preserve_parent_score": True,
        "candidate_promotion_margin": 0.05,
        "parent_transition_advantage": True,
        "parent_transition_break_cost": 4.0,
        "parent_transition_candidate_top_k": 32,
        "expected_missing": 26,
        "expected_parameters": 220481,
    },
    "v51_rapf_source_reliability": {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "spatial_mask_refiner": True,
        "spatial_mask_hidden_dim": 32,
        "max_spatial_mask_delta": 2.0,
        "adaptive_source_mixing": True,
        "source_distribution_reliability": True,
        "source_count": 4,
        "shared_source_index": 0,
        "max_source_mix_delta": 1.0,
        "source_mix_temperature": 0.5,
        "joint_source_names": (
            "default,contrastive_text,mask_text,sacr_structured"
        ),
        "use_sacr_source": True,
        "expected_missing": 66,
        "expected_parameters": 1151158,
    },
}


PROFILES["v51_anchor"] = dict(
    PROFILES["v51_bmq_rank"], metric_aligned_utility=False
)


def _normalized_state(state):
    return {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in state.items()
    }


def _atomic_write_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def audit_initialization(checkpoint_path, profile):
    if profile not in PROFILES:
        raise ValueError("unknown joint-query initialization profile")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if (not isinstance(checkpoint, dict)
            or not isinstance(checkpoint.get("model"), dict)
            or checkpoint.get("config") is None):
        raise ValueError("protected checkpoint is incomplete")

    contract = PROFILES[profile]
    args = copy.deepcopy(checkpoint["config"])
    overrides = {
        "use_joint_query_quality_reranker": True,
        "joint_query_quality_hidden_dim": 128,
        "joint_query_quality_heads": 4,
        "joint_query_quality_layers": 1,
        "joint_query_quality_dropout": 0.1,
        "joint_query_quality_max_delta": contract.get(
            "max_delta", 1.25
        ),
        "joint_query_quality_mask_weight": 0.25,
        "joint_query_quality_score_weight": 1.0,
        "joint_query_quality_direct_residual_scale": contract.get(
            "direct_residual_scale", 1.0
        ),
        "joint_query_quality_use_metric_aligned_utility": contract.get(
            "metric_aligned_utility", False
        ),
        "joint_query_quality_preserve_parent_score": contract.get(
            "preserve_parent_score", False
        ),
        "joint_query_quality_candidate_promotion_margin": contract.get(
            "candidate_promotion_margin", 0.0
        ),
        "joint_query_quality_use_parent_transition_advantage": contract.get(
            "parent_transition_advantage", False
        ),
        "joint_query_quality_use_decomposed_transition_advantage": (
            contract.get("decomposed_transition_advantage", False)
        ),
        "joint_query_quality_use_setwise_tier_advantage": contract.get(
            "setwise_tier_advantage", False
        ),
        "joint_query_quality_use_decoupled_setwise_heads": contract.get(
            "decoupled_setwise_heads", False
        ),
        "joint_query_quality_use_factorized_setwise_safety": contract.get(
            "factorized_setwise_safety", False
        ),
        "joint_query_quality_use_factorized_setwise_risk_bound": contract.get(
            "factorized_setwise_risk_bound", False
        ),
        "joint_query_quality_use_setwise_safety_veto_gate": contract.get(
            "setwise_safety_veto_gate", False
        ),
        "joint_query_quality_use_cost_calibrated_setwise_risk_bound": (
            contract.get("cost_calibrated_setwise_risk_bound", False)
        ),
        "joint_query_quality_use_setwise_safety_slack_quantile_bound": (
            contract.get("safety_slack_quantile_bound", False)
        ),
        "joint_query_quality_use_setwise_safety_slack_pairwise_order": (
            contract.get("safety_slack_pairwise_order", False)
        ),
        "joint_query_quality_use_proposal_conditioned_safety": (
            contract.get("proposal_conditioned_safety", False)
        ),
        "joint_query_quality_use_parent_referenced_safety": (
            contract.get("parent_referenced_safety", False)
        ),
        "joint_query_quality_use_coupled_safe_repair_witness": (
            contract.get("coupled_safe_repair_witness", False)
        ),
        "joint_query_quality_use_bidirectional_coupled_boundary": (
            contract.get("bidirectional_coupled_boundary", False)
        ),
        "joint_query_quality_use_centered_coupled_separation": (
            contract.get("centered_coupled_separation", False)
        ),
        "joint_query_quality_use_hazard_conditioned_coupled_separation": (
            contract.get(
                "hazard_conditioned_coupled_separation", False
            )
        ),
        "joint_query_quality_use_monotonic_box_safety_folding": (
            contract.get("monotonic_box_safety_folding", False)
        ),
        "joint_query_quality_use_same_candidate_branchwise_witness": (
            contract.get("same_candidate_branchwise_witness", False)
        ),
        "joint_query_quality_use_parent_non_degradation_certificate": (
            contract.get("parent_non_degradation_certificate", False)
        ),
        "joint_query_quality_use_criterion_responsible_hazard_attribution": (
            contract.get("criterion_responsible_hazard_attribution", False)
        ),
        "joint_query_quality_use_independent_joint_hazard_certificate": (
            contract.get("independent_joint_hazard_certificate", False)
        ),
        "joint_query_quality_use_frozen_raw_joint_hazard_features": (
            contract.get("frozen_raw_joint_hazard_features", False)
        ),
        "joint_query_quality_use_factorized_hit_advantage": contract.get(
            "factorized_hit_advantage", False
        ),
        "joint_query_quality_use_factorized_nested_dominance": contract.get(
            "factorized_nested_dominance", False
        ),
        "joint_query_quality_factorized_hit_break_cost": contract.get(
            "factorized_hit_break_cost", 4.0
        ),
        "joint_query_quality_parent_transition_break_cost": contract.get(
            "parent_transition_break_cost", 4.0
        ),
        "joint_query_quality_parent_transition_candidate_top_k": contract.get(
            "parent_transition_candidate_top_k", 0
        ),
        "joint_query_quality_use_mask_calibration": contract[
            "mask_calibration"
        ],
        "joint_query_quality_max_mask_alpha_delta": 1.0,
        "joint_query_quality_max_mask_logit_bias": 2.0,
        "joint_query_quality_use_source_mask_evidence": contract[
            "source_mask_evidence"
        ],
        "joint_query_quality_use_gate_evidence": contract[
            "gate_evidence"
        ],
        "joint_query_quality_use_spatial_mask_refiner": contract.get(
            "spatial_mask_refiner", False
        ),
        "joint_query_quality_spatial_mask_hidden_dim": contract.get(
            "spatial_mask_hidden_dim", 32
        ),
        "joint_query_quality_max_spatial_mask_delta": contract.get(
            "max_spatial_mask_delta", 2.0
        ),
        "joint_query_quality_use_adaptive_source_mixing": contract.get(
            "adaptive_source_mixing", False
        ),
        "joint_query_quality_use_source_distribution_reliability": (
            contract.get("source_distribution_reliability", False)
        ),
        "joint_query_quality_source_names": contract.get(
            "joint_source_names", ""
        ),
        "joint_query_quality_max_source_mix_delta": contract.get(
            "max_source_mix_delta", 1.0
        ),
        "joint_query_quality_source_mix_temperature": contract.get(
            "source_mix_temperature", 0.5
        ),
        "use_sacr_source": contract.get("use_sacr_source", False),
        "sacr_hidden_dim": 288,
        "sacr_max_pairs": 3,
        "sacr_top_m_targets": 32,
        "sacr_top_k_anchors": 16,
        "sacr_geo_dim": 16,
        "sacr_min_parse_confidence": 0.0,
        "sacr_residual_scale_init": 0.1,
    }
    for name, value in overrides.items():
        setattr(args, name, value)

    model = TrainTester.get_model(args)
    source = _normalized_state(checkpoint["model"])
    target = model.state_dict()
    incompatible = model.load_state_dict(source, strict=False)
    loaded = model.state_dict()

    source_names = set(source)
    target_names = set(target)
    common = sorted(source_names & target_names)
    missing = sorted(target_names - source_names)
    unexpected = sorted(source_names - target_names)
    changed = sorted(
        name for name in common if not torch.equal(source[name], loaded[name])
    )
    shape_mismatches = sorted(
        name for name in common if source[name].shape != target[name].shape
    )
    joint = model.joint_query_quality_reranker
    new_parameter_numel = sum(
        parameter.numel() for parameter in joint.parameters()
    )
    expected_prefixes = ["joint_query_quality_reranker."]
    if contract.get("use_sacr_source", False):
        expected_prefixes.extend([
            "structured_slot_builder.",
            "sacr_head.",
            "sacr_residual_scale",
        ])
        new_parameter_numel += sum(
            parameter.numel()
            for parameter in model.structured_slot_builder.parameters()
        )
        new_parameter_numel += sum(
            parameter.numel() for parameter in model.sacr_head.parameters()
        )
        new_parameter_numel += model.sacr_residual_scale.numel()
    expected_missing = {
        name for name in target
        if any(name.startswith(prefix) for prefix in expected_prefixes)
    }
    zero_initialized_heads = all(
        int(torch.count_nonzero(value).item()) == 0
        for name, value in joint.state_dict().items()
        if name.startswith((
            "quality_head.", "residual_head.", "mask_calibration_head.",
            "parent_transition_head.3.", "setwise_tier_head.3.",
            "setwise_promotion_head.3.", "setwise_safety_head.3.",
            "independent_joint_hazard_head.3.",
            "factorized_hit_head.",
            "spatial_mask_refiner.query_projection.2.",
            "adaptive_source_mixer.source_router.2.",
            "adaptive_source_mixer.strength_head.2.",
        ))
    )
    decomposed_safe_initialization = True
    if joint.decomposed_transition_head is not None:
        final = joint.decomposed_transition_head[-1]
        expected_bias = final.bias.new_tensor([
            0.0, math.log(joint.parent_transition_break_cost),
            0.0, math.log(joint.parent_transition_break_cost),
        ])
        decomposed_safe_initialization = (
            int(torch.count_nonzero(final.weight).item()) == 0
            and torch.equal(final.bias, expected_bias)
        )
    safety_contract_matches = (
        joint.preserve_parent_score
        == contract.get("preserve_parent_score", False)
        and joint.candidate_promotion_margin
        == contract.get("candidate_promotion_margin", 0.0)
        and joint.max_delta == contract.get("max_delta", 1.25)
        and joint.use_parent_transition_advantage
        == contract.get("parent_transition_advantage", False)
        and joint.use_decomposed_transition_advantage
        == contract.get("decomposed_transition_advantage", False)
        and joint.use_setwise_tier_advantage
        == contract.get("setwise_tier_advantage", False)
        and joint.use_decoupled_setwise_heads
        == contract.get("decoupled_setwise_heads", False)
        and joint.use_factorized_setwise_safety
        == contract.get("factorized_setwise_safety", False)
        and joint.use_factorized_setwise_risk_bound
        == contract.get("factorized_setwise_risk_bound", False)
        and joint.use_setwise_safety_veto_gate
        == contract.get("setwise_safety_veto_gate", False)
        and joint.use_cost_calibrated_setwise_risk_bound
        == contract.get("cost_calibrated_setwise_risk_bound", False)
        and joint.use_setwise_safety_slack_quantile_bound
        == contract.get("safety_slack_quantile_bound", False)
        and joint.use_setwise_safety_slack_pairwise_order
        == contract.get("safety_slack_pairwise_order", False)
        and joint.use_proposal_conditioned_safety
        == contract.get("proposal_conditioned_safety", False)
        and joint.use_parent_referenced_safety
        == contract.get("parent_referenced_safety", False)
        and joint.use_coupled_safe_repair_witness
        == contract.get("coupled_safe_repair_witness", False)
        and joint.use_bidirectional_coupled_boundary
        == contract.get("bidirectional_coupled_boundary", False)
        and joint.use_centered_coupled_separation
        == contract.get("centered_coupled_separation", False)
        and joint.use_hazard_conditioned_coupled_separation
        == contract.get("hazard_conditioned_coupled_separation", False)
        and joint.use_monotonic_box_safety_folding
        == contract.get("monotonic_box_safety_folding", False)
        and joint.use_same_candidate_branchwise_witness
        == contract.get("same_candidate_branchwise_witness", False)
        and joint.use_parent_non_degradation_certificate
        == contract.get("parent_non_degradation_certificate", False)
        and joint.use_criterion_responsible_hazard_attribution
        == contract.get("criterion_responsible_hazard_attribution", False)
        and joint.use_independent_joint_hazard_certificate
        == contract.get("independent_joint_hazard_certificate", False)
        and joint.use_frozen_raw_joint_hazard_features
        == contract.get("frozen_raw_joint_hazard_features", False)
        and joint.use_factorized_hit_advantage
        == contract.get("factorized_hit_advantage", False)
        and joint.use_factorized_nested_dominance
        == contract.get("factorized_nested_dominance", False)
        and joint.factorized_hit_break_cost
        == contract.get("factorized_hit_break_cost", 4.0)
        and joint.parent_transition_break_cost
        == contract.get("parent_transition_break_cost", 4.0)
        and joint.parent_transition_candidate_top_k
        == contract.get("parent_transition_candidate_top_k", 0)
    )
    passed = (
        len(missing) == contract["expected_missing"]
        and set(missing) == expected_missing
        and not unexpected
        and not changed
        and not shape_mismatches
        and set(incompatible.missing_keys) == expected_missing
        and not incompatible.unexpected_keys
        and new_parameter_numel == contract["expected_parameters"]
        and zero_initialized_heads
        and decomposed_safe_initialization
        and safety_contract_matches
    )
    result = {
        "schema": "mcln-{}-protected-initialization-audit-v1".format(profile),
        "profile": profile,
        "checkpoint": str(checkpoint_path),
        "source_state_count": len(source),
        "target_state_count": len(target),
        "common_tensor_count": len(common),
        "changed_common_tensor_count": len(changed),
        "changed_common_tensors": changed,
        "missing_tensor_count": len(missing),
        "missing_tensors": missing,
        "unexpected_tensor_count": len(unexpected),
        "unexpected_tensors": unexpected,
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatches": shape_mismatches,
        "joint_query_quality_state_count": len(joint.state_dict()),
        "joint_query_quality_parameter_numel": sum(
            parameter.numel() for parameter in joint.parameters()
        ),
        "new_module_parameter_numel": new_parameter_numel,
        "mask_calibration": contract["mask_calibration"],
        "source_mask_evidence": contract["source_mask_evidence"],
        "gate_evidence": contract["gate_evidence"],
        "spatial_mask_refiner": contract.get("spatial_mask_refiner", False),
        "adaptive_source_mixing": contract.get(
            "adaptive_source_mixing", False
        ),
        "source_distribution_reliability": contract.get(
            "source_distribution_reliability", False
        ),
        "use_sacr_source": contract.get("use_sacr_source", False),
        "direct_residual_scale": contract.get(
            "direct_residual_scale", 1.0
        ),
        "metric_aligned_utility": contract.get(
            "metric_aligned_utility", False
        ),
        "max_delta": joint.max_delta,
        "preserve_parent_score": joint.preserve_parent_score,
        "candidate_promotion_margin": joint.candidate_promotion_margin,
        "parent_transition_advantage": (
            joint.use_parent_transition_advantage
        ),
        "decomposed_transition_advantage": (
            joint.use_decomposed_transition_advantage
        ),
        "setwise_tier_advantage": (
            joint.use_setwise_tier_advantage
        ),
        "decoupled_setwise_heads": (
            joint.use_decoupled_setwise_heads
        ),
        "factorized_setwise_safety": (
            joint.use_factorized_setwise_safety
        ),
        "factorized_setwise_risk_bound": (
            joint.use_factorized_setwise_risk_bound
        ),
        "setwise_safety_veto_gate": (
            joint.use_setwise_safety_veto_gate
        ),
        "cost_calibrated_setwise_risk_bound": (
            joint.use_cost_calibrated_setwise_risk_bound
        ),
        "safety_slack_quantile_bound": (
            joint.use_setwise_safety_slack_quantile_bound
        ),
        "safety_slack_pairwise_order": (
            joint.use_setwise_safety_slack_pairwise_order
        ),
        "proposal_conditioned_safety": (
            joint.use_proposal_conditioned_safety
        ),
        "parent_referenced_safety": (
            joint.use_parent_referenced_safety
        ),
        "coupled_safe_repair_witness": (
            joint.use_coupled_safe_repair_witness
        ),
        "bidirectional_coupled_boundary": (
            joint.use_bidirectional_coupled_boundary
        ),
        "centered_coupled_separation": (
            joint.use_centered_coupled_separation
        ),
        "hazard_conditioned_coupled_separation": (
            joint.use_hazard_conditioned_coupled_separation
        ),
        "monotonic_box_safety_folding": (
            joint.use_monotonic_box_safety_folding
        ),
        "same_candidate_branchwise_witness": (
            joint.use_same_candidate_branchwise_witness
        ),
        "parent_non_degradation_certificate": (
            joint.use_parent_non_degradation_certificate
        ),
        "criterion_responsible_hazard_attribution": (
            joint.use_criterion_responsible_hazard_attribution
        ),
        "independent_joint_hazard_certificate": (
            joint.use_independent_joint_hazard_certificate
        ),
        "frozen_raw_joint_hazard_features": (
            joint.use_frozen_raw_joint_hazard_features
        ),
        "factorized_hit_advantage": (
            joint.use_factorized_hit_advantage
        ),
        "factorized_nested_dominance": (
            joint.use_factorized_nested_dominance
        ),
        "factorized_hit_break_cost": joint.factorized_hit_break_cost,
        "parent_transition_break_cost": (
            joint.parent_transition_break_cost
        ),
        "parent_transition_candidate_top_k": (
            joint.parent_transition_candidate_top_k
        ),
        "safety_contract_matches": safety_contract_matches,
        "zero_initialized_output_heads": zero_initialized_heads,
        "decomposed_safe_initialization": decomposed_safe_initialization,
        "pass": passed,
    }
    if not passed:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = audit_initialization(args.checkpoint, args.profile)
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
