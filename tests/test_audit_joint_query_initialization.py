from types import SimpleNamespace

import torch

from models.joint_query_quality import JointQueryQualityReranker
from scripts import audit_joint_query_initialization as initialization_audit


class _InitializationModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = torch.nn.Parameter(torch.tensor([1.0]))
        self.joint_query_quality_reranker = JointQueryQualityReranker(
            152,
            hidden_dim=args.joint_query_quality_hidden_dim,
            num_heads=args.joint_query_quality_heads,
            num_layers=args.joint_query_quality_layers,
            dropout=args.joint_query_quality_dropout,
            max_delta=args.joint_query_quality_max_delta,
            mask_weight=args.joint_query_quality_mask_weight,
            quality_score_weight=args.joint_query_quality_score_weight,
            direct_residual_scale=getattr(
                args, "joint_query_quality_direct_residual_scale", 1.0
            ),
            use_metric_aligned_utility=getattr(
                args,
                "joint_query_quality_use_metric_aligned_utility",
                False,
            ),
            preserve_parent_score=getattr(
                args,
                "joint_query_quality_preserve_parent_score",
                False,
            ),
            candidate_promotion_margin=getattr(
                args,
                "joint_query_quality_candidate_promotion_margin",
                0.0,
            ),
            use_parent_transition_advantage=getattr(
                args,
                "joint_query_quality_use_parent_transition_advantage",
                False,
            ),
            use_decomposed_transition_advantage=getattr(
                args,
                "joint_query_quality_use_decomposed_transition_advantage",
                False,
            ),
            use_setwise_tier_advantage=getattr(
                args,
                "joint_query_quality_use_setwise_tier_advantage",
                False,
            ),
            use_decoupled_setwise_heads=getattr(
                args,
                "joint_query_quality_use_decoupled_setwise_heads",
                False,
            ),
            use_factorized_setwise_safety=getattr(
                args,
                "joint_query_quality_use_factorized_setwise_safety",
                False,
            ),
            use_factorized_setwise_risk_bound=getattr(
                args,
                "joint_query_quality_use_factorized_setwise_risk_bound",
                False,
            ),
            use_setwise_safety_veto_gate=getattr(
                args,
                "joint_query_quality_use_setwise_safety_veto_gate",
                False,
            ),
            use_cost_calibrated_setwise_risk_bound=getattr(
                args,
                "joint_query_quality_use_cost_calibrated_setwise_risk_bound",
                False,
            ),
            use_setwise_safety_slack_quantile_bound=getattr(
                args,
                "joint_query_quality_use_setwise_safety_slack_quantile_bound",
                False,
            ),
            use_setwise_safety_slack_pairwise_order=getattr(
                args,
                "joint_query_quality_use_setwise_safety_slack_pairwise_order",
                False,
            ),
            use_proposal_conditioned_safety=getattr(
                args,
                "joint_query_quality_use_proposal_conditioned_safety",
                False,
            ),
            use_parent_referenced_safety=getattr(
                args,
                "joint_query_quality_use_parent_referenced_safety",
                False,
            ),
            use_coupled_safe_repair_witness=getattr(
                args,
                "joint_query_quality_use_coupled_safe_repair_witness",
                False,
            ),
            use_bidirectional_coupled_boundary=getattr(
                args,
                "joint_query_quality_use_bidirectional_coupled_boundary",
                False,
            ),
            use_centered_coupled_separation=getattr(
                args,
                "joint_query_quality_use_centered_coupled_separation",
                False,
            ),
            use_hazard_conditioned_coupled_separation=getattr(
                args,
                "joint_query_quality_use_hazard_conditioned_coupled_separation",
                False,
            ),
            use_monotonic_box_safety_folding=getattr(
                args,
                "joint_query_quality_use_monotonic_box_safety_folding",
                False,
            ),
            use_same_candidate_branchwise_witness=getattr(
                args,
                "joint_query_quality_use_same_candidate_branchwise_witness",
                False,
            ),
            use_parent_non_degradation_certificate=getattr(
                args,
                "joint_query_quality_use_parent_non_degradation_certificate",
                False,
            ),
            use_criterion_responsible_hazard_attribution=getattr(
                args,
                "joint_query_quality_use_criterion_responsible_hazard_attribution",
                False,
            ),
            use_independent_joint_hazard_certificate=getattr(
                args,
                "joint_query_quality_use_independent_joint_hazard_certificate",
                False,
            ),
            use_frozen_raw_joint_hazard_features=getattr(
                args,
                "joint_query_quality_use_frozen_raw_joint_hazard_features",
                False,
            ),
            use_factorized_hit_advantage=getattr(
                args,
                "joint_query_quality_use_factorized_hit_advantage",
                False,
            ),
            use_factorized_nested_dominance=getattr(
                args,
                "joint_query_quality_use_factorized_nested_dominance",
                False,
            ),
            factorized_hit_break_cost=getattr(
                args,
                "joint_query_quality_factorized_hit_break_cost",
                4.0,
            ),
            parent_transition_break_cost=getattr(
                args,
                "joint_query_quality_parent_transition_break_cost",
                4.0,
            ),
            parent_transition_candidate_top_k=getattr(
                args,
                "joint_query_quality_parent_transition_candidate_top_k",
                0,
            ),
            use_mask_calibration=(
                args.joint_query_quality_use_mask_calibration
            ),
            max_mask_alpha_delta=(
                args.joint_query_quality_max_mask_alpha_delta
            ),
            max_mask_logit_bias=(
                args.joint_query_quality_max_mask_logit_bias
            ),
            use_source_mask_evidence=(
                args.joint_query_quality_use_source_mask_evidence
            ),
            use_gate_evidence=args.joint_query_quality_use_gate_evidence,
            use_spatial_mask_refiner=(
                args.joint_query_quality_use_spatial_mask_refiner
            ),
            spatial_mask_hidden_dim=(
                args.joint_query_quality_spatial_mask_hidden_dim
            ),
            max_spatial_mask_delta=(
                args.joint_query_quality_max_spatial_mask_delta
            ),
            use_adaptive_source_mixing=(
                args.joint_query_quality_use_adaptive_source_mixing
            ),
            use_source_distribution_reliability=getattr(
                args,
                "joint_query_quality_use_source_distribution_reliability",
                False,
            ),
            source_count=(
                initialization_audit.PROFILES["v49"]["source_count"]
                if args.joint_query_quality_use_adaptive_source_mixing
                else None
            ),
            shared_source_index=(
                initialization_audit.PROFILES["v49"][
                    "shared_source_index"
                ]
                if args.joint_query_quality_use_adaptive_source_mixing
                else None
            ),
            max_source_mix_delta=(
                args.joint_query_quality_max_source_mix_delta
            ),
            source_mix_temperature=(
                args.joint_query_quality_source_mix_temperature
            ),
        )


def test_v46_initialization_audit_enforces_gate_evidence_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v46"
    )

    assert initialization_audit.PROFILES["v46"] == {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": True,
        "expected_missing": 22,
        "expected_parameters": 158339,
    }
    assert result["pass"] is True
    assert result["missing_tensor_count"] == 22
    assert result["joint_query_quality_state_count"] == 22
    assert result["joint_query_quality_parameter_numel"] == 158339
    assert result["mask_calibration"] is True
    assert result["source_mask_evidence"] is True
    assert result["gate_evidence"] is True
    assert result["zero_initialized_output_heads"] is True


def test_v48_initialization_audit_enforces_spatial_mask_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v48"
    )

    assert initialization_audit.PROFILES["v48"] == {
        "mask_calibration": True,
        "source_mask_evidence": True,
        "gate_evidence": False,
        "spatial_mask_refiner": True,
        "spatial_mask_hidden_dim": 32,
        "max_spatial_mask_delta": 2.0,
        "expected_missing": 34,
        "expected_parameters": 176979,
    }
    assert result["pass"] is True
    assert result["missing_tensor_count"] == 34
    assert result["joint_query_quality_state_count"] == 34
    assert result["joint_query_quality_parameter_numel"] == 176979
    assert result["spatial_mask_refiner"] is True
    assert result["zero_initialized_output_heads"] is True


def test_v49_initialization_audit_enforces_adaptive_source_mix_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v49"
    )

    assert result["pass"] is True
    assert result["missing_tensor_count"] == 45
    assert result["joint_query_quality_state_count"] == 45
    assert result["joint_query_quality_parameter_numel"] == 229460
    assert result["spatial_mask_refiner"] is True
    assert result["adaptive_source_mixing"] is True
    assert result["zero_initialized_output_heads"] is True


def test_v50_initialization_profile_matches_live_sacr_contract():
    profile = initialization_audit.PROFILES["v50_sacr"]

    assert profile["source_count"] == 4
    assert profile["shared_source_index"] == 0
    assert profile["joint_source_names"] == (
        "default,contrastive_text,mask_text,sacr_structured"
    )
    assert profile["use_sacr_source"] is True
    assert profile["expected_missing"] == 66
    assert profile["expected_parameters"] == 1150390


def test_v51_initialization_profile_adds_only_distribution_reliability():
    v50 = initialization_audit.PROFILES["v50_sacr"]
    v51 = initialization_audit.PROFILES["v51_rapf_source_reliability"]

    assert v51["source_distribution_reliability"] is True
    assert v51["source_count"] == v50["source_count"]
    assert v51["joint_source_names"] == v50["joint_source_names"]
    assert v51["expected_missing"] == v50["expected_missing"]
    assert v51["expected_parameters"] - v50["expected_parameters"] == 6 * 128



def test_v51_bmq_initialization_profile_is_parameter_neutral():
    profile = initialization_audit.PROFILES["v51_bmq_rank"]
    baseline = initialization_audit.PROFILES["v50_sacr"]
    assert profile["expected_missing"] == baseline["expected_missing"]
    assert profile["expected_parameters"] == baseline["expected_parameters"]
    assert profile["direct_residual_scale"] == 0.25
    assert profile["metric_aligned_utility"] is True


def test_v51_anchor_initialization_profile_matches_runtime_contract():
    anchor = initialization_audit.PROFILES["v51_anchor"]
    bmq = initialization_audit.PROFILES["v51_bmq_rank"]

    assert anchor["expected_missing"] == bmq["expected_missing"]
    assert anchor["expected_parameters"] == bmq["expected_parameters"]
    assert anchor["direct_residual_scale"] == 0.25
    assert anchor["metric_aligned_utility"] is False
    assert bmq["metric_aligned_utility"] is True



def test_v51_parent_promotion_profile_is_parameter_checked_and_mask_isolated():
    profile = initialization_audit.PROFILES["v51_parent_promotion"]

    assert profile["preserve_parent_score"] is True
    assert profile["candidate_promotion_margin"] == 0.05
    assert profile["max_delta"] == 0.25
    assert profile["direct_residual_scale"] == 0.25
    assert profile["metric_aligned_utility"] is False
    assert profile["mask_calibration"] is False
    assert profile["source_mask_evidence"] is False
    assert profile.get("spatial_mask_refiner", False) is False
    assert profile["adaptive_source_mixing"] is True
    assert profile["use_sacr_source"] is True
    assert profile["expected_missing"] == 52
    assert profile["expected_parameters"] == 1126942


def test_v51_parent_safe_launcher_uses_training_inference_contract():
    launcher = (
        initialization_audit.ROOT_DIR
        / "scripts" / "run_double_stage_v51_bmq_rank.sh"
    ).read_text()
    profile = launcher.split(
        "  parent_safe)", 1
    )[1].split("  bmq_safe)", 1)[0]

    assert "INITIALIZATION_PROFILE=v51_parent_promotion" in profile
    assert "PRESERVE_PARENT_SCORE=1" in profile
    assert "CANDIDATE_PROMOTION_MARGIN=0.05" in profile
    assert "USE_MASK_CALIBRATION=0" in profile
    assert "USE_SOURCE_MASK_EVIDENCE=0" in profile
    assert "USE_SPATIAL_MASK_REFINER=0" in profile
    assert "CANDIDATE_MASK_LOSS_WEIGHT=0.0" in profile
    assert "CANDIDATE_LOVASZ_LOSS_WEIGHT=0.0" in profile
    assert (
        'JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE="${PRESERVE_PARENT_SCORE}"'
        in launcher
    )
    assert (
        'JOINT_QUERY_QUALITY_CANDIDATE_PROMOTION_MARGIN="${CANDIDATE_PROMOTION_MARGIN}"'
        in launcher
    )


def test_v72_initialization_audit_checks_factorized_safety_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v72_factorized_setwise_safety"
    )

    assert result["pass"] is True
    assert result["missing_tensor_count"] == 30
    assert result["joint_query_quality_state_count"] == 30
    assert result["joint_query_quality_parameter_numel"] == 286395
    assert result["decoupled_setwise_heads"] is True
    assert result["factorized_setwise_safety"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v73_initialization_audit_checks_risk_bound_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v73_factorized_setwise_risk_bound"
    )

    assert result["pass"] is True
    assert result["missing_tensor_count"] == 30
    assert result["joint_query_quality_state_count"] == 30
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["decoupled_setwise_heads"] is True
    assert result["factorized_setwise_safety"] is True
    assert result["factorized_setwise_risk_bound"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v74_initialization_audit_checks_safety_veto_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v74_setwise_safety_veto_gate"
    )

    assert result["pass"] is True
    assert result["missing_tensor_count"] == 30
    assert result["joint_query_quality_state_count"] == 30
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["decoupled_setwise_heads"] is True
    assert result["factorized_setwise_safety"] is True
    assert result["factorized_setwise_risk_bound"] is True
    assert result["setwise_safety_veto_gate"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v75_initialization_audit_checks_cost_calibrated_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v75_cost_calibrated_risk_bound"
    )

    assert result["pass"] is True
    assert result["missing_tensor_count"] == 30
    assert result["joint_query_quality_state_count"] == 30
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["setwise_safety_veto_gate"] is True
    assert result["cost_calibrated_setwise_risk_bound"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v76_initialization_audit_checks_slack_quantile_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v76_safety_slack_quantile_bound"
    )

    assert result["pass"] is True
    assert result["missing_tensor_count"] == 30
    assert result["joint_query_quality_state_count"] == 30
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["setwise_safety_veto_gate"] is True
    assert result["cost_calibrated_setwise_risk_bound"] is False
    assert result["safety_slack_quantile_bound"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v77_initialization_audit_checks_slack_pairwise_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v77_safety_slack_pairwise_order"
    )

    assert result["pass"] is True
    assert result["missing_tensor_count"] == 30
    assert result["joint_query_quality_state_count"] == 30
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["safety_slack_quantile_bound"] is True
    assert result["safety_slack_pairwise_order"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v78_initialization_audit_checks_proposal_conditioned_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v78_proposal_conditioned_safety"
    )

    assert result["pass"] is True
    assert result["missing_tensor_count"] == 30
    assert result["joint_query_quality_state_count"] == 30
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["safety_slack_pairwise_order"] is True
    assert result["proposal_conditioned_safety"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v79_initialization_audit_checks_parent_referenced_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )

    result = initialization_audit.audit_initialization(
        checkpoint_path, "v79_parent_referenced_safety"
    )

    assert result["pass"] is True
    assert result["missing_tensor_count"] == 30
    assert result["joint_query_quality_state_count"] == 30
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["safety_slack_pairwise_order"] is True
    assert result["proposal_conditioned_safety"] is False
    assert result["parent_referenced_safety"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v80_initialization_audit_checks_coupled_witness_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )
    result = initialization_audit.audit_initialization(
        checkpoint_path, "v80_coupled_safe_repair_witness"
    )
    assert result["pass"] is True
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["parent_referenced_safety"] is True
    assert result["coupled_safe_repair_witness"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v81_initialization_audit_checks_bidirectional_boundary_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )
    result = initialization_audit.audit_initialization(
        checkpoint_path, "v81_bidirectional_coupled_boundary"
    )
    assert result["pass"] is True
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["parent_referenced_safety"] is True
    assert result["coupled_safe_repair_witness"] is True
    assert result["bidirectional_coupled_boundary"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v82_initialization_audit_checks_centered_separation_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )
    result = initialization_audit.audit_initialization(
        checkpoint_path, "v82_centered_coupled_separation"
    )
    assert result["pass"] is True
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["bidirectional_coupled_boundary"] is True
    assert result["centered_coupled_separation"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v83_initialization_audit_checks_hazard_conditioned_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )
    result = initialization_audit.audit_initialization(
        checkpoint_path, "v83_hazard_conditioned_coupled_separation"
    )
    assert result["pass"] is True
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["centered_coupled_separation"] is True
    assert result["hazard_conditioned_coupled_separation"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v84_initialization_audit_checks_monotonic_box_folding_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )
    result = initialization_audit.audit_initialization(
        checkpoint_path, "v84_monotonic_box_safety_folding"
    )
    assert result["pass"] is True
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["hazard_conditioned_coupled_separation"] is True
    assert result["monotonic_box_safety_folding"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v85_initialization_audit_checks_branchwise_witness_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )
    result = initialization_audit.audit_initialization(
        checkpoint_path, "v85_same_candidate_branchwise_witness"
    )
    assert result["pass"] is True
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["monotonic_box_safety_folding"] is True
    assert result["same_candidate_branchwise_witness"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v86_initialization_audit_checks_parent_certificate_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )
    result = initialization_audit.audit_initialization(
        checkpoint_path, "v86_parent_non_degradation_certificate"
    )
    assert result["pass"] is True
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["same_candidate_branchwise_witness"] is True
    assert result["parent_non_degradation_certificate"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v87_initialization_audit_checks_responsible_hazard_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )
    result = initialization_audit.audit_initialization(
        checkpoint_path, "v87_criterion_responsible_hazard_attribution"
    )
    assert result["pass"] is True
    assert result["joint_query_quality_parameter_numel"] == 286779
    assert result["parent_non_degradation_certificate"] is True
    assert result["criterion_responsible_hazard_attribution"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v88_initialization_audit_checks_independent_hazard_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )
    result = initialization_audit.audit_initialization(
        checkpoint_path, "v88_independent_joint_hazard_certificate"
    )
    assert result["pass"] is True
    assert result["joint_query_quality_parameter_numel"] == 353083
    assert result["parent_non_degradation_certificate"] is True
    assert result["independent_joint_hazard_certificate"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True


def test_v89_initialization_audit_checks_frozen_raw_hazard_contract(
        tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "protected_v19.pth"
    torch.save({
        "model": {"backbone": torch.tensor([1.0])},
        "config": SimpleNamespace(),
    }, checkpoint_path)
    monkeypatch.setattr(
        initialization_audit.TrainTester,
        "get_model",
        staticmethod(_InitializationModel),
    )
    result = initialization_audit.audit_initialization(
        checkpoint_path, "v89_frozen_raw_joint_hazard_features"
    )
    assert result["pass"] is True
    assert result["joint_query_quality_parameter_numel"] == 365371
    assert result["independent_joint_hazard_certificate"] is True
    assert result["frozen_raw_joint_hazard_features"] is True
    assert result["zero_initialized_output_heads"] is True
    assert result["safety_contract_matches"] is True
