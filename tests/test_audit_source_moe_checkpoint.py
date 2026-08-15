from types import SimpleNamespace

import pytest
import torch

from scripts.audit_source_moe_checkpoint import (
    AUDIT_PROFILES,
    V20_CHANGED_PREFIXES,
    V20_NEW_PREFIXES,
    V21_CHANGED_PREFIXES,
    V21_NEW_PREFIXES,
    V22_CHANGED_PREFIXES,
    V22_NEW_PREFIXES,
    V23_CHANGED_PREFIXES,
    V23_NEW_PREFIXES,
    V25_CHANGED_PREFIXES,
    V25_NEW_PREFIXES,
    V26_CHANGED_PREFIXES,
    V26_NEW_PREFIXES,
    V28_CHANGED_PREFIXES,
    V28_NEW_PREFIXES,
    V29_CHANGED_PREFIXES,
    V29_NEW_PREFIXES,
    V37_CHANGED_PREFIXES,
    V37_NEW_PREFIXES,
    V38_CHANGED_PREFIXES,
    V38_NEW_PREFIXES,
    V39_CHANGED_PREFIXES,
    V39_NEW_PREFIXES,
    JOINT_QUERY_QUALITY_PREFIXES,
    QUERY_MASK_FUSION_PREFIXES,
    SACR_JOINT_QUERY_PREFIXES,
    audit_checkpoint,
)


def _checkpoint(
        model, epoch=2, step=10, moment_numel=(2, 1),
        action="cascade_joint_risk_correction",
        objective="cascade_joint_risk_calibrated"):
    state = {}
    parameter_ids = []
    for parameter_id, numel in enumerate(moment_numel):
        parameter_ids.append(parameter_id)
        state[parameter_id] = {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.ones(numel),
            "exp_avg_sq": torch.full((numel,), 2.0),
        }
    return {
        "model": model,
        "epoch": epoch,
        "config": SimpleNamespace(
            source_moe_gate_action_mode=action,
            source_moe_gate_objective=objective,
            source_moe_gate_train_only=True,
            source_moe_gate_new_heads_only=True,
        ),
        "optimizer": {
            "state": state,
            "param_groups": [{"params": parameter_ids}],
        },
    }


def _models():
    prefix = "source_moe.fallback_gate."
    baseline = {
        "backbone.weight": torch.tensor([1.0]),
        prefix + "absolute_quality_head.weight": torch.tensor([0.0]),
        prefix + "cascade_candidate_safety_head.bias": torch.tensor([0.0]),
    }
    candidate = {
        "backbone.weight": torch.tensor([1.0]),
        prefix + "absolute_quality_head.weight": torch.tensor([1.0]),
        prefix + "cascade_candidate_safety_head.bias": torch.tensor([2.0]),
        prefix + "cascade_joint_action_head.weight": torch.tensor([3.0]),
    }
    return baseline, candidate


def _audit(baseline, candidate, **kwargs):
    expected = {
        "expected_common": 3,
        "expected_changed": 2,
        "expected_new": 1,
        "expected_optimizer_states": 2,
        "expected_optimizer_step": 10,
        "expected_optimizer_numel": 3,
        "expected_epoch": 2,
    }
    expected.update(kwargs)
    return audit_checkpoint(
        _checkpoint(baseline),
        _checkpoint(candidate, epoch=expected["expected_epoch"]),
        **expected
    )


def test_checkpoint_audit_accepts_exact_v20_contract():
    baseline, candidate = _models()

    result = _audit(baseline, candidate)

    assert result["pass"] is True
    assert result["model"]["common_tensor_count"] == 3
    assert result["model"]["changed_tensor_count"] == 2
    assert result["model"]["new_tensor_count"] == 1
    assert result["optimizer"] == {
        "state_count": 2,
        "step": 10,
        "parameter_numel": 3,
        "moment_tensor_count": 4,
        "moments_finite": True,
        "moments_nonzero": True,
        "zero_second_moment_count": 0,
    }


def test_checkpoint_audit_accepts_query_mask_fusion_contract():
    baseline = {
        "backbone.weight": torch.tensor([1.0]),
        "mask_head.weight": torch.tensor([2.0]),
    }
    candidate = dict(baseline)
    candidate.update({
        "query_mask_fusion_calibrator.norm.weight": torch.tensor([3.0]),
        "query_mask_fusion_calibrator.residual_head.bias": torch.tensor([4.0]),
    })
    baseline_checkpoint = _checkpoint(baseline)
    candidate_checkpoint = _checkpoint(candidate)
    candidate_checkpoint["config"] = SimpleNamespace(
        use_query_mask_fusion_calibrator=True,
        query_mask_fusion_train_only=True,
    )

    result = audit_checkpoint(
        baseline_checkpoint,
        candidate_checkpoint,
        changed_prefixes=QUERY_MASK_FUSION_PREFIXES,
        new_prefixes=QUERY_MASK_FUSION_PREFIXES,
        expected_common=2,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=None,
        expected_objective=None,
        expected_contract="query_mask_fusion",
    )

    assert result["pass"] is True
    assert result["contract"] == {
        "query_mask_fusion": True,
        "query_mask_fusion_only": True,
    }
    assert AUDIT_PROFILES["qmask"]["expected_optimizer_numel"] == 92429


def test_checkpoint_audit_accepts_joint_query_quality_contract():
    baseline = {
        "backbone.weight": torch.tensor([1.0]),
        "source_moe.router.weight": torch.tensor([2.0]),
    }
    candidate = dict(baseline)
    candidate.update({
        "joint_query_quality_reranker.input_projection.0.weight": (
            torch.tensor([3.0])
        ),
        "joint_query_quality_reranker.residual_head.bias": torch.tensor([4.0]),
    })
    baseline_checkpoint = _checkpoint(baseline)
    candidate_checkpoint = _checkpoint(candidate)
    candidate_checkpoint["config"] = SimpleNamespace(
        use_joint_query_quality_reranker=True,
        joint_query_quality_train_only=True,
    )

    result = audit_checkpoint(
        baseline_checkpoint,
        candidate_checkpoint,
        changed_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        new_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        expected_common=2,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=None,
        expected_objective=None,
        expected_contract="joint_query_quality",
    )

    assert result["pass"] is True
    assert result["contract"] == {
        "joint_query_quality": True,
        "joint_query_quality_only": True,
        "joint_query_mask_calibration": False,
    }
    assert AUDIT_PROFILES["v41"] == {
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
    }


def test_checkpoint_audit_accepts_v42_mask_calibration_contract():
    baseline = {
        "backbone.weight": torch.tensor([1.0]),
        "source_moe.router.weight": torch.tensor([2.0]),
    }
    candidate = dict(baseline)
    candidate.update({
        "joint_query_quality_reranker.input_projection.0.weight": (
            torch.tensor([3.0])
        ),
        "joint_query_quality_reranker.mask_calibration_head.bias": (
            torch.tensor([4.0])
        ),
    })
    baseline_checkpoint = _checkpoint(baseline)
    candidate_checkpoint = _checkpoint(candidate)
    candidate_checkpoint["config"] = SimpleNamespace(
        use_joint_query_quality_reranker=True,
        joint_query_quality_train_only=True,
        joint_query_quality_use_mask_calibration=True,
        joint_query_quality_use_source_mask_evidence=False,
        joint_query_quality_max_mask_alpha_delta=1.0,
        joint_query_quality_max_mask_logit_bias=2.0,
    )

    result = audit_checkpoint(
        baseline_checkpoint,
        candidate_checkpoint,
        changed_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        new_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        expected_common=2,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=None,
        expected_objective=None,
        expected_contract="joint_query_mask_calibration",
        expected_mask_alpha_delta=1.0,
        expected_mask_logit_bias=2.0,
    )

    assert result["pass"] is True
    assert result["contract"] == {
        "joint_query_quality": True,
        "joint_query_quality_only": True,
        "joint_query_mask_calibration": True,
        "joint_query_source_mask_evidence": False,
        "joint_query_mask_alpha_delta": 1.0,
        "joint_query_mask_logit_bias": 2.0,
    }
    assert AUDIT_PROFILES["v42"] == {
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
    }


def test_checkpoint_audit_accepts_v43_source_mask_evidence_contract():
    baseline = {"backbone.weight": torch.tensor([1.0])}
    candidate = dict(baseline)
    candidate.update({
        "joint_query_quality_reranker.input_projection.0.weight": (
            torch.tensor([3.0, 4.0])
        ),
        "joint_query_quality_reranker.mask_calibration_head.bias": (
            torch.tensor([5.0])
        ),
    })
    candidate_checkpoint = _checkpoint(candidate)
    candidate_checkpoint["config"] = SimpleNamespace(
        use_joint_query_quality_reranker=True,
        joint_query_quality_train_only=True,
        joint_query_quality_use_mask_calibration=True,
        joint_query_quality_use_source_mask_evidence=True,
        joint_query_quality_max_mask_alpha_delta=1.0,
        joint_query_quality_max_mask_logit_bias=2.0,
    )

    result = audit_checkpoint(
        _checkpoint(baseline),
        candidate_checkpoint,
        changed_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        new_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        expected_common=1,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=None,
        expected_objective=None,
        expected_contract="joint_query_mask_evidence",
        expected_mask_alpha_delta=1.0,
        expected_mask_logit_bias=2.0,
        expected_source_mask_evidence=True,
    )

    assert result["pass"] is True
    assert result["contract"]["joint_query_source_mask_evidence"] is True
    assert AUDIT_PROFILES["v43"] == {
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
    }
    selector_profile = AUDIT_PROFILES["v43_selector"]
    assert selector_profile["expected_common"] == 1078
    assert selector_profile["expected_new"] == 22
    assert selector_profile["expected_optimizer_states"] == 22
    assert selector_profile["expected_optimizer_numel"] == 155219
    assert selector_profile["expected_contract"] == (
        "joint_query_mask_evidence"
    )


def test_checkpoint_audit_accepts_v46_gate_evidence_contract():
    baseline = {"backbone.weight": torch.tensor([1.0])}
    candidate = dict(baseline)
    candidate.update({
        "joint_query_quality_reranker.input_projection.0.weight": (
            torch.tensor([3.0, 4.0])
        ),
        "joint_query_quality_reranker.mask_calibration_head.bias": (
            torch.tensor([5.0])
        ),
    })
    candidate_checkpoint = _checkpoint(candidate)
    candidate_checkpoint["config"] = SimpleNamespace(
        use_joint_query_quality_reranker=True,
        joint_query_quality_train_only=True,
        joint_query_quality_use_mask_calibration=True,
        joint_query_quality_use_source_mask_evidence=True,
        joint_query_quality_use_gate_evidence=True,
        joint_query_quality_max_mask_alpha_delta=1.0,
        joint_query_quality_max_mask_logit_bias=2.0,
    )

    result = audit_checkpoint(
        _checkpoint(baseline),
        candidate_checkpoint,
        changed_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        new_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        expected_common=1,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=None,
        expected_objective=None,
        expected_contract="joint_query_gate_evidence",
        expected_mask_alpha_delta=1.0,
        expected_mask_logit_bias=2.0,
        expected_source_mask_evidence=True,
        expected_gate_evidence=True,
    )

    assert result["pass"] is True
    assert result["contract"]["joint_query_gate_evidence"] is True
    assert AUDIT_PROFILES["v46"] == {
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
    }


def test_checkpoint_audit_accepts_v48_spatial_mask_contract():
    baseline = {"backbone.weight": torch.tensor([1.0])}
    candidate = dict(baseline)
    candidate.update({
        "joint_query_quality_reranker.input_projection.0.weight": (
            torch.tensor([3.0, 4.0])
        ),
        "joint_query_quality_reranker.spatial_mask_refiner."
        "query_projection.2.bias": torch.tensor([5.0]),
    })
    candidate_checkpoint = _checkpoint(candidate)
    candidate_checkpoint["config"] = SimpleNamespace(
        use_joint_query_quality_reranker=True,
        joint_query_quality_train_only=True,
        joint_query_quality_use_mask_calibration=True,
        joint_query_quality_use_source_mask_evidence=True,
        joint_query_quality_use_gate_evidence=False,
        joint_query_quality_use_spatial_mask_refiner=True,
        joint_query_quality_max_mask_alpha_delta=1.0,
        joint_query_quality_max_mask_logit_bias=2.0,
        joint_query_quality_spatial_mask_hidden_dim=32,
        joint_query_quality_max_spatial_mask_delta=2.0,
    )

    result = audit_checkpoint(
        _checkpoint(baseline),
        candidate_checkpoint,
        changed_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        new_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        expected_common=1,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=None,
        expected_objective=None,
        expected_contract="joint_query_spatial_mask_refinement",
        expected_mask_alpha_delta=1.0,
        expected_mask_logit_bias=2.0,
        expected_source_mask_evidence=True,
        expected_gate_evidence=False,
        expected_spatial_mask_refiner=True,
        expected_spatial_mask_hidden_dim=32,
        expected_max_spatial_mask_delta=2.0,
    )

    assert result["pass"] is True
    assert result["contract"]["joint_query_spatial_mask_refiner"] is True
    assert result["contract"]["joint_query_spatial_mask_hidden_dim"] == 32
    assert result["contract"]["joint_query_max_spatial_mask_delta"] == 2.0
    assert AUDIT_PROFILES["v48"] == {
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
    }


def test_checkpoint_audit_accepts_v49_adaptive_source_mix_contract():
    baseline = {"backbone.weight": torch.tensor([1.0])}
    candidate = dict(baseline)
    candidate.update({
        "joint_query_quality_reranker.adaptive_source_mixer."
        "source_router.2.weight": torch.tensor([3.0, 4.0]),
        "joint_query_quality_reranker.adaptive_source_mixer."
        "strength_head.2.bias": torch.tensor([5.0]),
    })
    candidate_checkpoint = _checkpoint(candidate)
    candidate_checkpoint["config"] = SimpleNamespace(
        use_joint_query_quality_reranker=True,
        joint_query_quality_train_only=True,
        joint_query_quality_use_mask_calibration=True,
        joint_query_quality_use_source_mask_evidence=True,
        joint_query_quality_use_gate_evidence=False,
        joint_query_quality_use_spatial_mask_refiner=True,
        joint_query_quality_use_adaptive_source_mixing=True,
        joint_query_quality_max_mask_alpha_delta=1.0,
        joint_query_quality_max_mask_logit_bias=2.0,
        joint_query_quality_spatial_mask_hidden_dim=32,
        joint_query_quality_max_spatial_mask_delta=2.0,
        joint_query_quality_max_source_mix_delta=1.0,
        joint_query_quality_source_mix_temperature=0.5,
    )

    result = audit_checkpoint(
        _checkpoint(baseline),
        candidate_checkpoint,
        changed_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        new_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        expected_common=1,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=None,
        expected_objective=None,
        expected_contract="joint_query_adaptive_source_mixing",
        expected_mask_alpha_delta=1.0,
        expected_mask_logit_bias=2.0,
        expected_source_mask_evidence=True,
        expected_gate_evidence=False,
        expected_spatial_mask_refiner=True,
        expected_spatial_mask_hidden_dim=32,
        expected_max_spatial_mask_delta=2.0,
        expected_adaptive_source_mixing=True,
        expected_max_source_mix_delta=1.0,
        expected_source_mix_temperature=0.5,
    )

    assert result["pass"] is True
    assert result["contract"]["joint_query_adaptive_source_mixing"] is True
    assert result["contract"]["joint_query_max_source_mix_delta"] == 1.0
    assert result["contract"]["joint_query_source_mix_temperature"] == 0.5
    assert AUDIT_PROFILES["v49"]["expected_new"] == 45
    assert AUDIT_PROFILES["v49"]["expected_optimizer_states"] == 45
    assert AUDIT_PROFILES["v49"]["expected_optimizer_numel"] == 229460


def test_checkpoint_audit_inherits_empty_joint_source_pool_from_parent():
    baseline = {"backbone.weight": torch.tensor([1.0])}
    candidate = dict(baseline)
    candidate.update({
        "joint_query_quality_reranker.input_projection.0.weight": (
            torch.tensor([3.0, 4.0])
        ),
        "joint_query_quality_reranker.residual_head.bias": torch.tensor([5.0]),
    })
    checkpoint = _checkpoint(candidate)
    checkpoint["config"] = SimpleNamespace(
        use_joint_query_quality_reranker=True,
        joint_query_quality_train_only=True,
        source_choice_selector_sources=(
            "default,contrastive_text,mask_text"
        ),
        joint_query_quality_source_names="",
    )

    result = audit_checkpoint(
        _checkpoint(baseline), checkpoint,
        changed_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        new_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
        expected_common=1,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=None,
        expected_objective=None,
        expected_contract="joint_query_quality",
        expected_parent_source_names=(
            "default", "contrastive_text", "mask_text"
        ),
        expected_joint_source_names=(
            "default", "contrastive_text", "mask_text"
        ),
    )

    assert result["pass"] is True


def test_checkpoint_audit_accepts_v50_sacr_as_joint_only_fourth_source():
    baseline = {"backbone.weight": torch.tensor([1.0])}
    candidate = dict(baseline)
    candidate.update({
        "joint_query_quality_reranker.adaptive_source_mixer."
        "source_router.2.weight": torch.tensor([3.0, 4.0]),
        "structured_slot_builder.target_attn.weight": torch.tensor([5.0]),
        "sacr_residual_scale": torch.tensor([0.2]),
    })
    checkpoint = _checkpoint(candidate, moment_numel=(2, 1, 1))
    checkpoint["config"] = SimpleNamespace(
        use_joint_query_quality_reranker=True,
        joint_query_quality_train_only=True,
        joint_query_quality_use_mask_calibration=True,
        joint_query_quality_use_source_mask_evidence=True,
        joint_query_quality_use_gate_evidence=False,
        joint_query_quality_use_spatial_mask_refiner=True,
        joint_query_quality_use_adaptive_source_mixing=True,
        joint_query_quality_max_mask_alpha_delta=1.0,
        joint_query_quality_max_mask_logit_bias=2.0,
        joint_query_quality_spatial_mask_hidden_dim=32,
        joint_query_quality_max_spatial_mask_delta=2.0,
        joint_query_quality_max_source_mix_delta=1.0,
        joint_query_quality_source_mix_temperature=0.5,
        joint_query_quality_source_mix_query_focus_weight=0.75,
        use_sacr_source=True,
        source_choice_selector_sources=(
            "default", "contrastive_text", "mask_text"
        ),
        joint_query_quality_source_names=(
            "default", "contrastive_text", "mask_text", "sacr_structured"
        ),
    )
    result = audit_checkpoint(
        _checkpoint(baseline), checkpoint,
        changed_prefixes=SACR_JOINT_QUERY_PREFIXES,
        new_prefixes=SACR_JOINT_QUERY_PREFIXES,
        expected_common=1,
        expected_changed=0,
        expected_new=3,
        expected_optimizer_states=3,
        expected_optimizer_step=10,
        expected_optimizer_numel=4,
        expected_epoch=2,
        expected_action=None,
        expected_objective=None,
        expected_contract="joint_query_sacr_adaptive_source_mixing",
        expected_mask_alpha_delta=1.0,
        expected_mask_logit_bias=2.0,
        expected_source_mask_evidence=True,
        expected_gate_evidence=False,
        expected_spatial_mask_refiner=True,
        expected_spatial_mask_hidden_dim=32,
        expected_max_spatial_mask_delta=2.0,
        expected_adaptive_source_mixing=True,
        expected_max_source_mix_delta=1.0,
        expected_source_mix_temperature=0.5,
        expected_source_mix_query_focus_weight=0.75,
        expected_sacr=True,
        expected_parent_source_names=(
            "default", "contrastive_text", "mask_text"
        ),
        expected_joint_source_names=(
            "default", "contrastive_text", "mask_text", "sacr_structured"
        ),
    )
    assert result["pass"] is True
    assert result["contract"]["sacr_source"] is True
    assert result["contract"][
        "joint_query_source_mix_query_focus_weight"
    ] == pytest.approx(0.75)
    assert AUDIT_PROFILES["v50_sacr"]["expected_new"] == 66
    assert AUDIT_PROFILES["v50_sacr"]["expected_optimizer_states"] == 66
    assert AUDIT_PROFILES["v50_sacr"]["expected_optimizer_numel"] == 1150390


def test_v51_checkpoint_profile_requires_distribution_reliability():
    v50 = AUDIT_PROFILES["v50_sacr"]
    v51 = AUDIT_PROFILES["v51_rapf_source_reliability"]

    assert v51["expected_contract"] == (
        "joint_query_sacr_source_distribution_reliability"
    )
    assert v51["expected_source_distribution_reliability"] is True
    assert v51["expected_new"] == v50["expected_new"]
    assert v51["expected_optimizer_states"] == v50[
        "expected_optimizer_states"
    ]
    assert v51["expected_optimizer_numel"] - v50[
        "expected_optimizer_numel"
    ] == 6 * 128




def test_v55_checkpoint_profile_requires_nested_dominance_contract():
    profile = AUDIT_PROFILES["v55_nested_dominance"]

    assert profile["expected_contract"] == "joint_query_quality"
    assert profile["expected_common"] == 1228
    assert profile["expected_new"] == 22
    assert profile["expected_optimizer_states"] == 18
    assert profile["expected_optimizer_numel"] == 152886
    assert profile["expected_preserve_parent_score"] is True
    assert profile["expected_candidate_promotion_margin"] == pytest.approx(0.0)
    assert profile["expected_factorized_hit_advantage"] is True
    assert profile["expected_factorized_nested_dominance"] is True
    assert profile["expected_factorized_hit_break_cost"] == pytest.approx(1.0)
    assert profile["expected_parent_transition_candidate_top_k"] == 32


def test_v53_checkpoint_profile_requires_factorized_hit_contract():
    profile = AUDIT_PROFILES["v53_factorized_hit"]

    assert profile["expected_contract"] == "joint_query_quality"
    assert profile["expected_common"] == 1228
    assert profile["expected_new"] == 22
    assert profile["expected_optimizer_states"] == 18
    assert profile["expected_optimizer_numel"] == 152886
    assert profile["expected_sacr"] is False
    assert profile["expected_adaptive_source_mixing"] is False
    assert profile["expected_preserve_parent_score"] is True
    assert profile["expected_parent_transition_advantage"] is False
    assert profile["expected_factorized_hit_advantage"] is True
    assert profile["expected_parent_transition_break_cost"] == pytest.approx(4.0)
    assert profile["expected_parent_transition_candidate_top_k"] == 32

def test_v62_checkpoint_profile_requires_decomposed_transition_contract():
    profile = AUDIT_PROFILES["v62_decomposed_transition"]

    assert profile["expected_contract"] == "joint_query_quality"
    assert profile["expected_common"] == 1228
    assert profile["expected_new"] == 26
    assert profile["expected_optimizer_states"] == 22
    assert profile["expected_optimizer_numel"] == 219320
    assert profile["expected_preserve_parent_score"] is True
    assert profile["expected_candidate_promotion_margin"] == pytest.approx(0.0)
    assert profile["expected_parent_transition_advantage"] is False
    assert profile["expected_decomposed_transition_advantage"] is True
    assert profile["expected_factorized_hit_advantage"] is False
    assert profile["expected_parent_transition_break_cost"] == pytest.approx(4.0)
    assert profile["expected_parent_transition_candidate_top_k"] == 32


def test_v52_checkpoint_profile_requires_parent_transition_contract():
    profile = AUDIT_PROFILES["v52_parent_transition"]

    assert profile["expected_contract"] == "joint_query_quality"
    assert profile["expected_common"] == 1228
    assert profile["expected_new"] == 26
    assert profile["expected_optimizer_states"] == 22
    assert profile["expected_optimizer_numel"] == 219578
    assert profile["expected_sacr"] is False
    assert profile["expected_adaptive_source_mixing"] is False
    assert profile["expected_preserve_parent_score"] is True
    assert profile["expected_candidate_promotion_margin"] == pytest.approx(0.05)
    assert profile["expected_parent_transition_advantage"] is True
    assert profile["expected_parent_transition_break_cost"] == pytest.approx(4.0)
    assert profile["expected_parent_transition_candidate_top_k"] == 32


def test_checkpoint_audit_rejects_parent_transition_contract_drift():
    baseline = {"backbone.weight": torch.tensor([1.0])}
    candidate = dict(baseline)
    candidate.update({
        "joint_query_quality_reranker.parent_transition_head.0.weight": (
            torch.tensor([3.0])
        ),
        "joint_query_quality_reranker.parent_transition_head.3.bias": (
            torch.tensor([4.0])
        ),
    })
    baseline_checkpoint = _checkpoint(baseline)
    candidate_checkpoint = _checkpoint(candidate)
    candidate_checkpoint["config"] = SimpleNamespace(
        use_joint_query_quality_reranker=True,
        joint_query_quality_train_only=True,
        joint_query_quality_use_mask_calibration=False,
        joint_query_quality_use_source_mask_evidence=False,
        joint_query_quality_use_gate_evidence=False,
        joint_query_quality_use_spatial_mask_refiner=False,
        joint_query_quality_use_adaptive_source_mixing=False,
        joint_query_quality_use_source_distribution_reliability=False,
        joint_query_quality_max_source_mix_delta=1.0,
        joint_query_quality_source_mix_temperature=0.5,
        joint_query_quality_source_mix_loss_weight=0.0,
        joint_query_quality_source_mix_alignment_temperature=0.25,
        joint_query_quality_source_mix_query_focus_weight=0.0,
        joint_query_quality_preserve_parent_score=True,
        joint_query_quality_candidate_promotion_margin=0.05,
        joint_query_quality_max_delta=0.25,
        joint_query_quality_direct_residual_scale=0.25,
        joint_query_quality_use_metric_aligned_utility=False,
        joint_query_quality_use_parent_transition_advantage=True,
        joint_query_quality_parent_transition_break_cost=2.0,
        joint_query_quality_parent_transition_candidate_top_k=32,
        use_sacr_source=False,
        source_choice_selector_sources="default,contrastive_text,mask_text",
        joint_query_quality_source_names="default,contrastive_text,mask_text",
    )

    with pytest.raises(ValueError, match="parent transition break cost"):
        audit_checkpoint(
            baseline_checkpoint,
            candidate_checkpoint,
            changed_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
            new_prefixes=JOINT_QUERY_QUALITY_PREFIXES,
            expected_common=1,
            expected_changed=0,
            expected_new=2,
            expected_optimizer_states=2,
            expected_optimizer_step=10,
            expected_optimizer_numel=3,
            expected_epoch=2,
            expected_action=None,
            expected_objective=None,
            expected_contract="joint_query_quality",
            expected_parent_transition_advantage=True,
            expected_parent_transition_break_cost=4.0,
            expected_parent_transition_candidate_top_k=32,
        )

def test_checkpoint_audit_accepts_v21_frozen_v19_contract():
    prefix = "source_moe.fallback_gate."
    baseline = {
        "backbone.weight": torch.tensor([1.0]),
        prefix + "cascade_opportunity_head.weight": torch.tensor([2.0]),
        prefix + "cascade_candidate_safety_head.bias": torch.tensor([3.0]),
    }
    candidate = dict(baseline)
    candidate.update({
        prefix + "cascade_fallback_set_action_head.score.weight": (
            torch.tensor([4.0, 5.0])
        ),
        prefix + "cascade_fallback_set_action_head.ffn_norm.bias": (
            torch.tensor([6.0])
        ),
    })
    action = "cascade_v19_fallback_set_correction"
    objective = "cascade_v19_fallback_set_risk_calibrated"

    result = audit_checkpoint(
        _checkpoint(baseline),
        _checkpoint(
            candidate,
            action=action,
            objective=objective,
        ),
        changed_prefixes=V21_CHANGED_PREFIXES,
        new_prefixes=V21_NEW_PREFIXES,
        expected_common=3,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=action,
        expected_objective=objective,
    )

    assert result["pass"] is True
    assert result["model"]["changed_tensor_count"] == 0
    assert result["model"]["new_tensor_count"] == 2
    assert AUDIT_PROFILES["v21"] == {
        "changed_prefixes": V21_CHANGED_PREFIXES,
        "new_prefixes": V21_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 15,
        "expected_optimizer_states": 15,
        "expected_optimizer_numel": 149504,
        "expected_action": action,
        "expected_objective": objective,
    }


def test_checkpoint_audit_accepts_v22_frozen_v19_contract():
    prefix = "source_moe.fallback_gate."
    baseline = {
        "backbone.weight": torch.tensor([1.0]),
        prefix + "cascade_opportunity_head.weight": torch.tensor([2.0]),
        prefix + "cascade_candidate_safety_head.bias": torch.tensor([3.0]),
    }
    candidate = dict(baseline)
    candidate.update({
        prefix
        + "cascade_rich_fallback_set_action_head.rich_norm.weight": (
            torch.tensor([4.0, 5.0])
        ),
        prefix
        + "cascade_rich_fallback_set_action_head.set_head.score.weight": (
            torch.tensor([6.0])
        ),
    })
    action = "cascade_v19_rich_set_correction"
    objective = "cascade_v19_rich_set_empirical_risk"

    result = audit_checkpoint(
        _checkpoint(baseline),
        _checkpoint(candidate, action=action, objective=objective),
        changed_prefixes=V22_CHANGED_PREFIXES,
        new_prefixes=V22_NEW_PREFIXES,
        expected_common=3,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=action,
        expected_objective=objective,
    )

    assert result["pass"] is True
    assert result["model"]["changed_tensor_count"] == 0
    assert result["model"]["new_tensor_count"] == 2
    assert AUDIT_PROFILES["v22"] == {
        "changed_prefixes": V22_CHANGED_PREFIXES,
        "new_prefixes": V22_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 17,
        "expected_optimizer_states": 17,
        "expected_optimizer_numel": 169264,
        "expected_action": action,
        "expected_objective": objective,
    }


def test_checkpoint_audit_accepts_v23_frozen_v19_contract():
    prefix = "source_moe."
    baseline = {
        "backbone.weight": torch.tensor([1.0]),
        prefix + "fallback_gate.cascade_opportunity_head.weight": (
            torch.tensor([2.0])
        ),
    }
    candidate = dict(baseline)
    candidate.update({
        prefix + "adaptive_source_mixer.mix_residual.2.weight": (
            torch.tensor([3.0])
        ),
        prefix
        + "fallback_gate.cascade_dense_quality_set_head.quality_head.weight": (
            torch.tensor([4.0, 5.0])
        ),
    })
    action = "cascade_v23_dense_quality_correction"
    objective = "cascade_v23_dense_quality_risk"

    result = audit_checkpoint(
        _checkpoint(baseline),
        _checkpoint(candidate, action=action, objective=objective),
        changed_prefixes=V23_CHANGED_PREFIXES,
        new_prefixes=V23_NEW_PREFIXES,
        expected_common=2,
        expected_changed=0,
        expected_new=2,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=action,
        expected_objective=objective,
    )

    assert result["pass"] is True
    assert result["model"]["changed_tensor_count"] == 0
    assert result["model"]["new_tensor_count"] == 2
    assert AUDIT_PROFILES["v23"] == {
        "changed_prefixes": V23_CHANGED_PREFIXES,
        "new_prefixes": V23_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 39,
        "expected_optimizer_states": 39,
        "expected_optimizer_numel": 588603,
        "expected_action": action,
        "expected_objective": objective,
    }

    assert AUDIT_PROFILES["v27"] == {
        "changed_prefixes": V23_CHANGED_PREFIXES,
        "new_prefixes": V23_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 39,
        "expected_optimizer_states": 39,
        "expected_optimizer_numel": 588603,
        "expected_action": action,
        "expected_objective": "cascade_v27_uncertainty_quality_risk",
    }

    assert AUDIT_PROFILES["v28"] == {
        "changed_prefixes": V28_CHANGED_PREFIXES,
        "new_prefixes": V28_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 75,
        "expected_optimizer_states": 75,
        "expected_optimizer_numel": 876174,
        "expected_action": "cascade_v28_selected_abstention_correction",
        "expected_objective": "cascade_v28_selected_abstention_risk",
    }

    assert AUDIT_PROFILES["v29"] == {
        "changed_prefixes": V29_CHANGED_PREFIXES,
        "new_prefixes": V29_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 75,
        "expected_optimizer_states": 75,
        "expected_optimizer_numel": 876174,
        "expected_action": "cascade_v29_counterfactual_selected_correction",
        "expected_objective": "cascade_v29_counterfactual_selected_risk",
    }

    assert AUDIT_PROFILES["v37"] == {
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
    }

    assert AUDIT_PROFILES["v38"] == {
        "changed_prefixes": V38_CHANGED_PREFIXES,
        "new_prefixes": V38_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 75,
        "expected_optimizer_states": 75,
        "expected_optimizer_numel": 876303,
        "expected_action": "cascade_v38_complementary_logodds_correction",
        "expected_objective": "cascade_v38_complementary_logodds_risk",
    }

    assert AUDIT_PROFILES["v39"] == {
        "changed_prefixes": V39_CHANGED_PREFIXES,
        "new_prefixes": V39_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 75,
        "expected_optimizer_states": 75,
        "expected_optimizer_numel": 876303,
        "expected_action": "cascade_v39_hazard_residual_correction",
        "expected_objective": "cascade_v39_hazard_residual_risk",
    }


def test_checkpoint_audit_accepts_v25_frozen_v19_contract():
    prefix = "source_moe."
    baseline = {
        "backbone.weight": torch.tensor([1.0]),
        prefix + "fallback_gate.cascade_opportunity_head.weight": (
            torch.tensor([2.0])
        ),
    }
    candidate = dict(baseline)
    candidate.update({
        prefix + "adaptive_source_mixer.mix_residual.2.weight": (
            torch.tensor([3.0])
        ),
        prefix
        + "fallback_gate.cascade_dense_quality_set_head.quality_head.weight": (
            torch.tensor([4.0])
        ),
        prefix
        + "fallback_gate.cascade_pairwise_calibrated_set_head.utility_head.bias": (
            torch.tensor([5.0])
        ),
    })
    action = "cascade_v25_pairwise_calibrated_correction"
    objective = "cascade_v25_pairwise_calibrated_risk"

    result = audit_checkpoint(
        _checkpoint(baseline),
        _checkpoint(
            candidate,
            moment_numel=(1, 1, 1),
            action=action,
            objective=objective,
        ),
        changed_prefixes=V25_CHANGED_PREFIXES,
        new_prefixes=V25_NEW_PREFIXES,
        expected_common=2,
        expected_changed=0,
        expected_new=3,
        expected_optimizer_states=3,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=action,
        expected_objective=objective,
    )

    assert result["pass"] is True
    assert AUDIT_PROFILES["v25"] == {
        "changed_prefixes": V25_CHANGED_PREFIXES,
        "new_prefixes": V25_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 69,
        "expected_optimizer_states": 69,
        "expected_optimizer_numel": 825997,
        "expected_action": action,
        "expected_objective": objective,
    }


def test_checkpoint_audit_accepts_v26_frozen_v19_contract():
    prefix = "source_moe."
    baseline = {
        "backbone.weight": torch.tensor([1.0]),
        prefix + "fallback_gate.cascade_opportunity_head.weight": (
            torch.tensor([2.0])
        ),
    }
    candidate = dict(baseline)
    candidate.update({
        prefix + "adaptive_source_mixer.mix_residual.2.weight": (
            torch.tensor([3.0])
        ),
        prefix
        + "fallback_gate.cascade_dense_quality_set_head.quality_head.weight": (
            torch.tensor([4.0])
        ),
        prefix
        + "fallback_gate.cascade_pairwise_calibrated_set_head.benefit_head.bias": (
            torch.tensor([5.0])
        ),
    })
    action = "cascade_v26_prior_restored_pairwise_correction"
    objective = "cascade_v26_prior_restored_pairwise_risk"

    result = audit_checkpoint(
        _checkpoint(baseline),
        _checkpoint(
            candidate,
            moment_numel=(1, 1, 1),
            action=action,
            objective=objective,
        ),
        changed_prefixes=V26_CHANGED_PREFIXES,
        new_prefixes=V26_NEW_PREFIXES,
        expected_common=2,
        expected_changed=0,
        expected_new=3,
        expected_optimizer_states=3,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action=action,
        expected_objective=objective,
    )

    assert result["pass"] is True
    assert AUDIT_PROFILES["v26"] == {
        "changed_prefixes": V26_CHANGED_PREFIXES,
        "new_prefixes": V26_NEW_PREFIXES,
        "expected_common": 1228,
        "expected_changed": 0,
        "expected_new": 69,
        "expected_optimizer_states": 69,
        "expected_optimizer_numel": 825997,
        "expected_action": action,
        "expected_objective": objective,
    }


def test_checkpoint_audit_accepts_debug_epoch_zero():
    baseline, candidate = _models()

    result = _audit(
        baseline,
        candidate,
        expected_epoch=0,
    )

    assert result["checkpoint_epoch"] == 0


def test_checkpoint_audit_rejects_change_outside_allowlist():
    baseline, candidate = _models()
    candidate["backbone.weight"] = torch.tensor([2.0])

    with pytest.raises(ValueError, match="changed outside the allowlist"):
        _audit(baseline, candidate, expected_changed=3)


def test_checkpoint_audit_rejects_new_tensor_outside_allowlist():
    baseline, candidate = _models()
    candidate.pop("source_moe.fallback_gate.cascade_joint_action_head.weight")
    candidate["backbone.new"] = torch.tensor([3.0])

    with pytest.raises(ValueError, match="added outside the allowlist"):
        _audit(baseline, candidate)


def test_checkpoint_audit_rejects_nonfinite_model_tensor():
    baseline, candidate = _models()
    candidate["source_moe.fallback_gate.absolute_quality_head.weight"] = (
        torch.tensor([float("nan")])
    )

    with pytest.raises(ValueError, match="non-finite"):
        _audit(baseline, candidate)


def test_checkpoint_audit_rejects_zero_adam_moment():
    baseline, candidate = _models()
    checkpoint = _checkpoint(candidate)
    checkpoint["optimizer"]["state"][0]["exp_avg"].zero_()

    with pytest.raises(ValueError, match="entirely zero"):
        audit_checkpoint(
            _checkpoint(baseline),
            checkpoint,
            expected_common=3,
            expected_changed=2,
            expected_new=1,
            expected_optimizer_states=2,
            expected_optimizer_step=10,
            expected_optimizer_numel=3,
            expected_epoch=2,
        )


def test_checkpoint_audit_rejects_config_contract_drift():
    baseline, candidate = _models()
    checkpoint = _checkpoint(candidate)
    checkpoint["config"].source_moe_gate_objective = "other"

    with pytest.raises(ValueError, match="objective is incompatible"):
        audit_checkpoint(
            _checkpoint(baseline),
            checkpoint,
            expected_common=3,
            expected_changed=2,
            expected_new=1,
            expected_optimizer_states=2,
            expected_optimizer_step=10,
            expected_optimizer_numel=3,
            expected_epoch=2,
        )


def test_checkpoint_audit_accepts_second_moment_underflow_with_active_first():
    baseline, candidate = _models()
    baseline_checkpoint = _checkpoint(baseline)
    candidate_checkpoint = _checkpoint(candidate)
    candidate_checkpoint["optimizer"]["state"][0]["exp_avg_sq"].zero_()

    result = audit_checkpoint(
        baseline_checkpoint,
        candidate_checkpoint,
        changed_prefixes=V20_CHANGED_PREFIXES,
        new_prefixes=V20_NEW_PREFIXES,
        expected_common=3,
        expected_changed=2,
        expected_new=1,
        expected_optimizer_states=2,
        expected_optimizer_step=10,
        expected_optimizer_numel=3,
        expected_epoch=2,
        expected_action="cascade_joint_risk_correction",
        expected_objective="cascade_joint_risk_calibrated",
    )

    assert result["pass"] is True
    assert result["optimizer"]["moments_nonzero"] is True
    assert result["optimizer"]["zero_second_moment_count"] == 1


def test_v63_checkpoint_profile_requires_setwise_tier_contract():
    profile = AUDIT_PROFILES["v63_setwise_tier"]

    assert profile["expected_contract"] == "joint_query_quality"
    assert profile["expected_common"] == 1228
    assert profile["expected_new"] == 25
    assert profile["expected_optimizer_states"] == 21
    assert profile["expected_optimizer_numel"] == 219060
    assert profile["expected_preserve_parent_score"] is True
    assert profile["expected_candidate_promotion_margin"] == pytest.approx(0.0)
    assert profile["expected_parent_transition_advantage"] is False
    assert profile["expected_decomposed_transition_advantage"] is False
    assert profile["expected_setwise_tier_advantage"] is True
    assert profile["expected_factorized_hit_advantage"] is False
    assert profile["expected_parent_transition_candidate_top_k"] == 32
