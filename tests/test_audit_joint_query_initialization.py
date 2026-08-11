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
