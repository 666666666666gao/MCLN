"""Integration tests for wiring SourceMoE into MCLN training plumbing."""

import argparse
from pathlib import Path

import pytest
import torch

from models.mcln_training_groups import parameter_group_name
from models.source_moe import (
    SourceMoE,
    compute_source_moe_fallback_gate_loss,
)
from main_utils import (
    parse_option,
    prepare_source_moe_gate_checkpoint_config,
    validate_source_moe_gate_checkpoint_contract,
    validate_source_moe_resume_checkpoint_contract,
)


def _parse(argv):
    import sys

    saved = sys.argv
    try:
        sys.argv = ["prog"] + argv
        return parse_option()
    finally:
        sys.argv = saved


def test_moe_parameters_land_in_selector_optimizer_group():
    assert parameter_group_name("source_moe.router.0.weight") == "selector"
    assert parameter_group_name("module.source_moe.routed_scale") == "selector"
    assert parameter_group_name("source_moe.text_projector.bias") == "selector"


def test_moe_grouping_does_not_disturb_existing_groups():
    assert parameter_group_name("source_choice_selector.choice_mlp.0.weight") == "selector"
    assert parameter_group_name("backbone_net.conv1.weight") == "backbone"
    assert parameter_group_name("x_query.0.weight") == "mask_head"
    assert parameter_group_name("decoder.0.linear1.weight") == "decoder"


def test_moe_flags_default_to_disabled():
    args = _parse([])
    assert args.use_source_moe is False
    assert args.source_moe_shared_source == "default"
    assert args.source_moe_top_k == 2
    assert args.source_moe_balance_loss_weight == pytest.approx(0.01)
    assert args.source_moe_rank_loss_weight == pytest.approx(1.0)
    assert args.source_moe_mask_rank_loss_weight == pytest.approx(0.25)
    assert args.source_moe_rank_temperature == pytest.approx(0.1)
    assert args.source_moe_anchor_loss_weight == pytest.approx(0.0)
    assert args.source_moe_anchor_margin == pytest.approx(0.05)
    assert args.source_moe_query_layers == 1
    assert args.source_moe_query_heads == 4
    assert args.source_moe_train_only is False
    assert args.source_moe_gate_train_only is False
    assert args.source_moe_gate_new_heads_only is False
    assert args.source_moe_gate_resume_optimizer is False
    assert args.source_moe_use_fallback_gate is False
    assert args.source_moe_gate_candidate_top_k == 8
    assert args.source_moe_gate_break_cost == pytest.approx(2.0)
    assert args.source_moe_gate_uncertainty_weight == pytest.approx(0.0)
    assert args.source_moe_gate_uncertainty_weight_explicit is False
    assert args.source_moe_gate_use_evidence_features is False
    assert args.source_moe_gate_context_layers == 0
    assert args.source_moe_gate_context_heads == 4
    assert args.source_moe_gate_context_dropout == pytest.approx(0.1)
    assert args.source_moe_gate_action_mode is None
    assert args.source_moe_gate_objective == "balanced_focal"
    assert args.source_moe_gate_objective_explicit is False
    assert args.source_moe_gate_setwise_temperature == pytest.approx(0.0)
    assert args.source_moe_gate_loss_weight == pytest.approx(0.0)
    assert args.expected_eval_sample_count is None
    assert args.joint_query_quality_use_adaptive_source_mixing is False
    assert (
        args.joint_query_quality_use_source_distribution_reliability is False
    )
    assert args.joint_query_quality_max_source_mix_delta == pytest.approx(1.0)
    assert args.joint_query_quality_source_mix_temperature == pytest.approx(0.5)
    assert args.joint_query_quality_source_mix_query_focus_weight == 0.0


def test_adaptive_source_mix_flags_parse():
    args = _parse([
        "--use_joint_query_quality_reranker",
        "--joint_query_quality_use_adaptive_source_mixing",
        "--joint_query_quality_use_source_distribution_reliability",
        "--joint_query_quality_max_source_mix_delta", "0.75",
        "--joint_query_quality_source_mix_temperature", "0.35",
        "--joint_query_quality_source_mix_query_focus_weight", "0.75",
    ])
    assert args.joint_query_quality_use_adaptive_source_mixing is True
    assert args.joint_query_quality_use_source_distribution_reliability is True
    assert args.joint_query_quality_max_source_mix_delta == pytest.approx(0.75)
    assert args.joint_query_quality_source_mix_temperature == pytest.approx(
        0.35
    )
    assert args.joint_query_quality_source_mix_query_focus_weight == (
        pytest.approx(0.75)
    )


def test_moe_flags_parse():
    args = _parse([
        "--use_source_moe",
        "--source_moe_shared_source", "default",
        "--source_moe_top_k", "1",
        "--source_moe_balance_loss_weight", "0.05",
        "--source_moe_rank_loss_weight", "0.75",
        "--source_moe_mask_rank_loss_weight", "0.4",
        "--source_moe_rank_temperature", "0.2",
        "--source_moe_anchor_loss_weight", "1.5",
        "--source_moe_anchor_margin", "0.1",
        "--source_moe_query_layers", "2",
        "--source_moe_lr", "0.002",
        "--source_moe_use_fallback_gate",
        "--source_moe_gate_candidate_top_k", "6",
        "--source_moe_gate_break_cost", "3.0",
        "--source_moe_gate_use_evidence_features",
        "--source_moe_gate_context_layers", "2",
        "--source_moe_gate_context_heads", "8",
        "--source_moe_gate_context_dropout", "0.2",
        "--source_moe_gate_action_mode", "expected_utility",
        "--source_moe_gate_objective", "balanced_calibrated_utility",
        "--source_moe_gate_setwise_temperature", "0.25",
        "--source_moe_gate_loss_weight", "1.25",
    ])
    assert args.use_source_moe is True
    assert args.source_moe_top_k == 1
    assert args.source_moe_balance_loss_weight == pytest.approx(0.05)
    assert args.source_moe_rank_loss_weight == pytest.approx(0.75)
    assert args.source_moe_mask_rank_loss_weight == pytest.approx(0.4)
    assert args.source_moe_rank_temperature == pytest.approx(0.2)
    assert args.source_moe_anchor_loss_weight == pytest.approx(1.5)
    assert args.source_moe_anchor_margin == pytest.approx(0.1)
    assert args.source_moe_query_layers == 2
    assert args.source_moe_lr == pytest.approx(0.002)
    assert args.source_moe_use_fallback_gate is True
    assert args.source_moe_gate_candidate_top_k == 6
    assert args.source_moe_gate_break_cost == pytest.approx(3.0)
    assert args.source_moe_gate_use_evidence_features is True
    assert args.source_moe_gate_context_layers == 2
    assert args.source_moe_gate_context_heads == 8
    assert args.source_moe_gate_context_dropout == pytest.approx(0.2)
    assert args.source_moe_gate_action_mode == "expected_utility"
    assert args.source_moe_gate_objective == "balanced_calibrated_utility"
    assert args.source_moe_gate_objective_explicit is True
    assert args.source_moe_gate_setwise_temperature == pytest.approx(0.25)
    assert args.source_moe_gate_loss_weight == pytest.approx(1.25)


def test_moe_flags_parse_direct_utility_action():
    args = _parse([
        "--source_moe_gate_action_mode", "direct_utility",
    ])

    assert args.source_moe_gate_action_mode == "direct_utility"


def test_moe_flags_parse_hierarchical_utility_action():
    args = _parse([
        "--source_moe_gate_action_mode", "hierarchical_utility",
        "--source_moe_gate_objective",
        "hierarchical_risk_calibrated",
    ])

    assert args.source_moe_gate_action_mode == "hierarchical_utility"
    assert args.source_moe_gate_objective == (
        "hierarchical_risk_calibrated"
    )


def test_moe_flags_parse_pairwise_verifier_action():
    args = _parse([
        "--source_moe_gate_action_mode", "pairwise_verifier",
        "--source_moe_gate_objective", "pairwise_risk_calibrated",
    ])

    assert args.source_moe_gate_action_mode == "pairwise_verifier"
    assert args.source_moe_gate_objective == "pairwise_risk_calibrated"


def test_moe_flags_parse_topn_pairwise_verifier_action():
    args = _parse([
        "--source_moe_gate_action_mode", "topn_pairwise_verifier",
        "--source_moe_gate_objective", "pairwise_risk_calibrated",
    ])

    assert args.source_moe_gate_action_mode == "topn_pairwise_verifier"
    assert args.source_moe_gate_objective == "pairwise_risk_calibrated"


def test_moe_flags_parse_topn_risk_calibrated_objective():
    args = _parse([
        "--source_moe_gate_action_mode", "topn_pairwise_verifier",
        "--source_moe_gate_objective", "topn_risk_calibrated",
    ])

    assert args.source_moe_gate_action_mode == "topn_pairwise_verifier"
    assert args.source_moe_gate_objective == "topn_risk_calibrated"


def test_moe_flags_parse_dual_evidence_verifier_objective():
    args = _parse([
        "--source_moe_gate_action_mode", "topn_dual_evidence_verifier",
        "--source_moe_gate_objective", "topn_dual_risk_calibrated",
    ])

    assert args.source_moe_gate_action_mode == (
        "topn_dual_evidence_verifier"
    )
    assert args.source_moe_gate_objective == "topn_dual_risk_calibrated"


def test_moe_flags_parse_absolute_quality_delta_objective():
    args = _parse([
        "--source_moe_gate_action_mode", "topn_absolute_quality_delta",
        "--source_moe_gate_objective",
        "topn_absolute_quality_calibrated",
    ])

    assert args.source_moe_gate_action_mode == (
        "topn_absolute_quality_delta"
    )
    assert args.source_moe_gate_objective == (
        "topn_absolute_quality_calibrated"
    )


def test_moe_flags_parse_cascade_absolute_quality_objective():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_absolute_quality_correction",
        "--source_moe_gate_objective",
        "cascade_absolute_quality_calibrated",
    ])

    assert args.source_moe_gate_train_only is True
    assert args.source_moe_gate_new_heads_only is True
    assert args.source_moe_gate_action_mode == (
        "cascade_absolute_quality_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_absolute_quality_calibrated"
    )


def test_moe_flags_parse_opportunity_cascade_objective():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_opportunity_quality_correction",
        "--source_moe_gate_objective",
        "cascade_opportunity_balanced_calibrated",
    ])

    assert args.source_moe_gate_train_only is True
    assert args.source_moe_gate_new_heads_only is True
    assert args.source_moe_gate_action_mode == (
        "cascade_opportunity_quality_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_opportunity_balanced_calibrated"
    )


def test_moe_flags_parse_verified_opportunity_cascade_objective():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_opportunity_verified_correction",
        "--source_moe_gate_objective",
        "cascade_opportunity_verified_calibrated",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_opportunity_verified_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_opportunity_verified_calibrated"
    )


def test_moe_flags_parse_joint_risk_cascade_objective():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_joint_risk_correction",
        "--source_moe_gate_objective",
        "cascade_joint_risk_calibrated",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_joint_risk_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_joint_risk_calibrated"
    )


def test_moe_flags_parse_v19_fallback_set_objective():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v19_fallback_set_correction",
        "--source_moe_gate_objective",
        "cascade_v19_fallback_set_risk_calibrated",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_v19_fallback_set_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_v19_fallback_set_risk_calibrated"
    )


def test_moe_flags_parse_v22_rich_set_objective():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v19_rich_set_correction",
        "--source_moe_gate_objective",
        "cascade_v19_rich_set_empirical_risk",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_v19_rich_set_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_v19_rich_set_empirical_risk"
    )


def test_moe_flags_parse_v23_dense_quality_objective():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v23_dense_quality_correction",
        "--source_moe_gate_objective",
        "cascade_v23_dense_quality_risk",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_v23_dense_quality_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_v23_dense_quality_risk"
    )


def test_moe_flags_parse_v27_uncertainty_quality_contract():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v23_dense_quality_correction",
        "--source_moe_gate_objective",
        "cascade_v27_uncertainty_quality_risk",
        "--source_moe_gate_uncertainty_weight", "0.5",
    ])

    assert args.source_moe_gate_objective == (
        "cascade_v27_uncertainty_quality_risk"
    )
    assert args.source_moe_gate_uncertainty_weight == pytest.approx(0.5)
    assert args.source_moe_gate_uncertainty_weight_explicit is True


def test_moe_flags_parse_v28_selected_abstention_contract():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v28_selected_abstention_correction",
        "--source_moe_gate_objective",
        "cascade_v28_selected_abstention_risk",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_v28_selected_abstention_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_v28_selected_abstention_risk"
    )


def test_moe_flags_parse_v29_counterfactual_selected_contract():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v29_counterfactual_selected_correction",
        "--source_moe_gate_objective",
        "cascade_v29_counterfactual_selected_risk",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_v29_counterfactual_selected_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_v29_counterfactual_selected_risk"
    )


def test_moe_flags_parse_v37_benefit_hazard_contract():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v37_counterfactual_benefit_hazard_correction",
        "--source_moe_gate_objective",
        "cascade_v37_counterfactual_benefit_hazard_risk",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_v37_counterfactual_benefit_hazard_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_v37_counterfactual_benefit_hazard_risk"
    )


def test_moe_flags_parse_v38_complementary_logodds_contract():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v38_complementary_logodds_correction",
        "--source_moe_gate_objective",
        "cascade_v38_complementary_logodds_risk",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_v38_complementary_logodds_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_v38_complementary_logodds_risk"
    )


def test_moe_flags_parse_v39_hazard_residual_contract():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v39_hazard_residual_correction",
        "--source_moe_gate_objective",
        "cascade_v39_hazard_residual_risk",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_v39_hazard_residual_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_v39_hazard_residual_risk"
    )


def test_moe_flags_parse_v25_pairwise_calibrated_objective():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v25_pairwise_calibrated_correction",
        "--source_moe_gate_objective",
        "cascade_v25_pairwise_calibrated_risk",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_v25_pairwise_calibrated_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_v25_pairwise_calibrated_risk"
    )


def test_moe_flags_parse_v26_prior_restored_pairwise_objective():
    args = _parse([
        "--source_moe_gate_train_only",
        "--source_moe_gate_new_heads_only",
        "--source_moe_gate_action_mode",
        "cascade_v26_prior_restored_pairwise_correction",
        "--source_moe_gate_objective",
        "cascade_v26_prior_restored_pairwise_risk",
    ])

    assert args.source_moe_gate_action_mode == (
        "cascade_v26_prior_restored_pairwise_correction"
    )
    assert args.source_moe_gate_objective == (
        "cascade_v26_prior_restored_pairwise_risk"
    )


def test_moe_flags_parse_row_boundary_loss_weight():
    args = _parse([
        "--source_moe_gate_boundary_loss_weight", "1.5",
    ])

    assert args.source_moe_gate_boundary_loss_weight == pytest.approx(1.5)


def test_moe_flags_parse_exact_gate_optimizer_resume():
    args = _parse(["--source_moe_gate_resume_optimizer"])

    assert args.source_moe_gate_resume_optimizer is True


def test_balance_loss_enters_total_loss_only_when_moe_enabled():
    """Guards the loss wiring: weight must be zeroed when MoE is off."""
    from models.losses import compute_hungarian_loss

    signature = compute_hungarian_loss.__code__.co_varnames[
        : compute_hungarian_loss.__code__.co_argcount
    ]
    assert "source_moe_balance_loss_weight" in signature
    assert "source_moe_gate_setwise_temperature" in signature


def test_hungarian_loss_accepts_hierarchical_risk_objective_contract():
    from models.losses import compute_hungarian_loss

    with pytest.raises(KeyError, match="center_label"):
        compute_hungarian_loss(
            end_points={},
            num_decoder_layers=1,
            set_criterion=None,
            source_moe_gate_objective="hierarchical_risk_calibrated",
        )


def test_hungarian_loss_accepts_pairwise_risk_objective_contract():
    from models.losses import compute_hungarian_loss

    with pytest.raises(KeyError, match="center_label"):
        compute_hungarian_loss(
            end_points={},
            num_decoder_layers=1,
            set_criterion=None,
            source_moe_gate_objective="pairwise_risk_calibrated",
        )


def test_hungarian_loss_accepts_topn_risk_objective_contract():
    from models.losses import compute_hungarian_loss

    with pytest.raises(KeyError, match="center_label"):
        compute_hungarian_loss(
            end_points={},
            num_decoder_layers=1,
            set_criterion=None,
            source_moe_gate_objective="topn_risk_calibrated",
        )


def test_hungarian_loss_accepts_dual_risk_objective_contract():
    from models.losses import compute_hungarian_loss

    with pytest.raises(KeyError, match="center_label"):
        compute_hungarian_loss(
            end_points={},
            num_decoder_layers=1,
            set_criterion=None,
            source_moe_gate_objective="topn_dual_risk_calibrated",
        )


def test_hungarian_loss_accepts_absolute_quality_objective_contract():
    from models.losses import compute_hungarian_loss

    with pytest.raises(KeyError, match="center_label"):
        compute_hungarian_loss(
            end_points={},
            num_decoder_layers=1,
            set_criterion=None,
            source_moe_gate_objective=(
                "topn_absolute_quality_calibrated"
            ),
        )


def test_hungarian_loss_accepts_cascade_quality_objective_contract():
    from models.losses import compute_hungarian_loss

    with pytest.raises(KeyError, match="center_label"):
        compute_hungarian_loss(
            end_points={},
            num_decoder_layers=1,
            set_criterion=None,
            source_moe_gate_objective=(
                "cascade_absolute_quality_calibrated"
            ),
        )


def test_hungarian_loss_accepts_opportunity_cascade_objective_contract():
    from models.losses import compute_hungarian_loss

    with pytest.raises(KeyError, match="center_label"):
        compute_hungarian_loss(
            end_points={},
            num_decoder_layers=1,
            set_criterion=None,
            source_moe_gate_objective=(
                "cascade_opportunity_balanced_calibrated"
            ),
        )


def test_hungarian_loss_accepts_verified_opportunity_objective_contract():
    from models.losses import compute_hungarian_loss

    with pytest.raises(KeyError, match="center_label"):
        compute_hungarian_loss(
            end_points={},
            num_decoder_layers=1,
            set_criterion=None,
            source_moe_gate_objective=(
                "cascade_opportunity_verified_calibrated"
            ),
        )


def test_hungarian_loss_accepts_joint_risk_objective_contract():
    from models.losses import compute_hungarian_loss

    with pytest.raises(KeyError, match="center_label"):
        compute_hungarian_loss(
            end_points={},
            num_decoder_layers=1,
            set_criterion=None,
            source_moe_gate_objective="cascade_joint_risk_calibrated",
        )


def test_hungarian_loss_accepts_v19_fallback_set_objective_contract():
    from models.losses import compute_hungarian_loss

    with pytest.raises(KeyError, match="center_label"):
        compute_hungarian_loss(
            end_points={},
            num_decoder_layers=1,
            set_criterion=None,
            source_moe_gate_objective=(
                "cascade_v19_fallback_set_risk_calibrated"
            ),
        )


def test_gate_loss_fallback_resolver_prefers_deployed_v19_query():
    from models.losses import resolve_source_moe_gate_loss_fallback_query

    default = torch.tensor([0])
    anchor = torch.tensor([1])
    deployed_v19 = torch.tensor([2])
    assert torch.equal(
        resolve_source_moe_gate_loss_fallback_query({
            "moe_gate_action_anchor_query": anchor,
            "moe_gate_supervision_fallback_query": deployed_v19,
        }, default),
        deployed_v19,
    )
    assert torch.equal(
        resolve_source_moe_gate_loss_fallback_query({
            "moe_gate_action_anchor_query": anchor,
        }, default),
        anchor,
    )


def test_moe_supervision_uses_annotation_source_not_benchmark_name():
    from models.losses import build_source_moe_grounding_sample_mask

    end_points = {
        "language_dataset": ["scanrefer"] * 4,
        "sample_dataset": ["scanrefer", "scannet", "nr3d", "sr3d"],
    }
    mask = build_source_moe_grounding_sample_mask(
        end_points, batch_size=4, device=torch.device("cpu")
    )

    assert torch.equal(mask, torch.tensor([True, False, True, True]))


def test_moe_supervision_requires_per_sample_annotation_source():
    from models.losses import build_source_moe_grounding_sample_mask

    with pytest.raises(ValueError, match="sample_dataset metadata"):
        build_source_moe_grounding_sample_mask(
            {"language_dataset": ["scanrefer", "scanrefer"]},
            batch_size=2,
            device=torch.device("cpu"),
        )


def test_source_moe_launcher_exposes_lr_decay_epochs():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_scanrefer_source_moe.sh"
    ).read_text(encoding="utf-8")

    assert 'LR_DECAY_EPOCHS_TEXT="${LR_DECAY_EPOCHS}"' in launcher
    assert 'elif [[ "${PHASE}" == "continue" ]]' in launcher
    assert 'LR_DECAY_EPOCHS_TEXT="3"' in launcher
    assert '--lr_decay_epochs "${LR_DECAY_EPOCHS_ARGS[@]}"' in launcher
    assert "LR_DECAY_EPOCHS must contain non-negative integers" in launcher


def test_source_moe_launcher_safe_continue_contract():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_scanrefer_source_moe.sh"
    ).read_text(encoding="utf-8")

    assert "PHASE=continue requires CHECKPOINT_PATH" in launcher
    assert 'START_EPOCH="${START_EPOCH:-72}"' in launcher
    assert 'MAX_EPOCH="${MAX_EPOCH:-80}"' in launcher
    assert "SOURCE_MOE_USE_FALLBACK_GATE=1" in launcher
    assert "SOURCE_MOE_GATE_LOSS_WEIGHT_CONTINUE:-1.0" in launcher
    assert "PHASE_ARGS=(--source_moe_train_only)" in launcher
    assert "RETENTION_ARGS=(--checkpoint_metric_retention)" in launcher


def test_source_moe_launcher_exposes_exact_gate_optimizer_resume():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_scanrefer_source_moe.sh"
    ).read_text(encoding="utf-8")

    assert 'SOURCE_MOE_GATE_RESUME_OPTIMIZER="${' in launcher
    assert "GATE_RESUME_ARGS=(--source_moe_gate_resume_optimizer)" in launcher
    assert "requires PHASE=gate" in launcher


def test_source_moe_launcher_exposes_cascade_new_head_allowlist():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_scanrefer_source_moe.sh"
    ).read_text(encoding="utf-8")

    assert "SOURCE_MOE_GATE_NEW_HEADS_ONLY" in launcher
    assert "GATE_NEW_HEADS_ARGS=(--source_moe_gate_new_heads_only)" in launcher
    assert "cascade_absolute_quality_correction" in launcher
    assert "cascade_absolute_quality_calibrated" in launcher
    assert "cascade_opportunity_quality_correction" in launcher
    assert "cascade_opportunity_balanced_calibrated" in launcher
    assert "cascade_opportunity_verified_correction" in launcher
    assert "cascade_opportunity_verified_calibrated" in launcher
    assert "cascade_joint_risk_correction" in launcher
    assert "cascade_joint_risk_calibrated" in launcher
    assert "cascade_v19_fallback_set_correction" in launcher
    assert "cascade_v19_fallback_set_risk_calibrated" in launcher
    assert "cascade_v19_rich_set_correction" in launcher
    assert "cascade_v19_rich_set_empirical_risk" in launcher
    assert "cascade_v23_dense_quality_correction" in launcher
    assert "cascade_v23_dense_quality_risk" in launcher
    assert "cascade_v25_pairwise_calibrated_correction" in launcher
    assert "cascade_v25_pairwise_calibrated_risk" in launcher
    assert "cascade_v26_prior_restored_pairwise_correction" in launcher
    assert "cascade_v26_prior_restored_pairwise_risk" in launcher


def test_source_moe_launcher_exposes_contextual_gate_controls():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_scanrefer_source_moe.sh"
    ).read_text(encoding="utf-8")

    assert "SOURCE_MOE_GATE_CONTEXT_LAYERS" in launcher
    assert "--source_moe_gate_context_layers" in launcher
    assert "--source_moe_gate_context_heads" in launcher
    assert "--source_moe_gate_context_dropout" in launcher
    assert "--source_moe_gate_setwise_temperature" in launcher


def test_source_moe_eval_launcher_is_checkpoint_bound_and_gpu_safe():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_scanrefer_source_moe_checkpoint.sh"
    ).read_text(encoding="utf-8")

    assert 'CHECKPOINT_PATH=<checkpoint> is required' in launcher
    assert 'EVAL_EPOCH must be a non-negative integer' in launcher
    assert 'EXPECTED_EVAL_SAMPLE_COUNT=9508' in launcher
    assert 'SOURCE_MOE_GATE_LOSS_WEIGHT=1.0' in launcher
    assert 'ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"' in launcher
    assert 'refusing concurrent eval' in launcher
    assert 'train_scanrefer_source_moe.sh" --eval' in launcher
    assert 'RUN_AUDIT="${RUN_AUDIT:-1}"' in launcher
    assert (
        "source_moe_v4_contract_eval/scannet,scanrefer/"
        "ssq_moe_v4_contract_eval/1785534811/eval_metrics_epoch_1.json"
        in launcher
    )
    assert 'source_choice_diagnostics_epoch_${EVAL_EPOCH}.json' in launcher
    assert 'audit_source_moe_candidate_oracle.py' in launcher
    assert 'source_moe_oracle_audit.json' in launcher


def test_moe_and_selector_are_mutually_exclusive():
    from models.mcln import MCLN

    with pytest.raises(ValueError):
        MCLN(
            num_class=1,
            num_obj_class=485,
            input_feature_dim=3,
            num_queries=8,
            num_decoder_layers=1,
            use_source_choice_selector=True,
            use_source_moe=True,
            source_choice_selector_sources="default,mask_text",
        )


def test_joint_query_quality_accepts_selector_arbiter_without_source_moe(
        monkeypatch):
    from models.mcln import MCLN

    def passed_arbiter_validation(*_args, **_kwargs):
        raise RuntimeError("passed source-arbiter validation")

    monkeypatch.setattr(
        "models.mcln.RobertaTokenizerFast.from_pretrained",
        passed_arbiter_validation,
    )
    with pytest.raises(RuntimeError, match="passed source-arbiter validation"):
        MCLN(
            num_class=1,
            num_obj_class=485,
            input_feature_dim=3,
            num_queries=8,
            num_decoder_layers=1,
            use_source_choice_selector=True,
            use_joint_query_quality_reranker=True,
            source_choice_selector_sources=(
                "default,default_rank_blend_contrastive010"
            ),
        )


def test_joint_query_quality_still_requires_a_source_arbiter():
    from models.mcln import MCLN

    with pytest.raises(ValueError, match="SourceMoE or the source-choice"):
        MCLN(
            num_class=1,
            num_obj_class=485,
            input_feature_dim=3,
            num_queries=8,
            num_decoder_layers=1,
            use_joint_query_quality_reranker=True,
        )


def test_v22_mcln_requires_contrastive_projection_features():
    from models.mcln import MCLN

    with pytest.raises(ValueError, match="contrastive_align_loss"):
        MCLN(
            num_class=1,
            num_obj_class=485,
            input_feature_dim=3,
            num_queries=8,
            num_decoder_layers=1,
            contrastive_align_loss=False,
            use_source_moe=True,
            source_moe_use_fallback_gate=True,
            source_moe_gate_action_mode=(
                "cascade_v19_rich_set_correction"
            ),
            source_choice_selector_sources="default,mask_text",
        )


def test_moe_writes_the_key_the_evaluator_consumes():
    """grounding_evaluator reads 'selected_source_scores'; MoE must fill it."""
    module = SourceMoE(
        source_names=("default", "mask_text"), d_model=8, text_dim=8
    )
    out = module(
        candidate_feats=torch.randn(2, 5, 8),
        candidate_boxes=torch.rand(2, 5, 6),
        source_scores={
            "default": torch.randn(2, 5),
            "mask_text": torch.randn(2, 5),
        },
    )
    assert "selected_source_scores" in out
    assert out["selected_source_scores"].shape == (2, 5)


def test_joint_query_only_inherits_selector_checkpoint_contract(tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "selector.pth"
    torch.save({
        "config": SimpleNamespace(
            use_source_moe=False,
            use_source_choice_selector=True,
            source_choice_selector_sources=(
                "default,default_rank_blend_contrastive010"
            ),
            source_choice_selector_hidden_dim=288,
        ),
    }, checkpoint_path)
    args = SimpleNamespace(
        eval=False,
        checkpoint_path=str(checkpoint_path),
        use_source_moe=False,
        use_source_choice_selector=True,
        source_moe_train_only=False,
        source_moe_gate_train_only=False,
        query_mask_fusion_train_only=False,
        joint_query_quality_train_only=True,
        joint_query_quality_use_gate_evidence=False,
        source_choice_selector_sources="default,mask_text",
        source_choice_selector_hidden_dim=128,
    )

    returned = prepare_source_moe_gate_checkpoint_config(args)

    assert returned is args
    assert args.use_source_moe is False
    assert args.use_source_choice_selector is True
    assert args.source_choice_selector_sources == (
        "default,default_rank_blend_contrastive010"
    )
    assert args.source_choice_selector_hidden_dim == 288


def test_selector_joint_query_rejects_source_moe_gate_evidence(tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "selector.pth"
    torch.save({
        "config": {
            "use_source_moe": False,
            "use_source_choice_selector": True,
            "source_choice_selector_sources": "default,mask_text",
            "source_choice_selector_hidden_dim": 288,
        },
    }, checkpoint_path)
    args = SimpleNamespace(
        eval=False,
        checkpoint_path=str(checkpoint_path),
        use_source_moe=False,
        use_source_choice_selector=True,
        source_moe_train_only=False,
        source_moe_gate_train_only=False,
        query_mask_fusion_train_only=False,
        joint_query_quality_train_only=True,
        joint_query_quality_use_gate_evidence=True,
    )

    with pytest.raises(ValueError, match="requires a SourceMoE checkpoint"):
        prepare_source_moe_gate_checkpoint_config(args)


@pytest.mark.parametrize("failure", ["missing", "shape", "nonfinite"])
def test_v22_source_moe_rich_feature_plumbing_fails_closed(failure):
    module = SourceMoE(
        source_names=("default", "mask_text"),
        d_model=8,
        text_dim=8,
        hidden_dim=16,
        query_heads=4,
        use_fallback_gate=True,
        gate_hidden_dim=12,
        gate_candidate_top_k=2,
        gate_action_mode="cascade_v19_rich_set_correction",
    )
    kwargs = {
        "candidate_feats": torch.randn(2, 5, 8),
        "candidate_boxes": torch.rand(2, 5, 6),
        "source_scores": {
            "default": torch.randn(2, 5),
            "mask_text": torch.randn(2, 5),
        },
        "gate_rich_candidate_feats": torch.randn(2, 5, 40),
    }
    if failure == "missing":
        kwargs.pop("gate_rich_candidate_feats")
    elif failure == "shape":
        kwargs["gate_rich_candidate_feats"] = torch.randn(2, 5, 39)
    else:
        kwargs["gate_rich_candidate_feats"][0, 0, 0] = float("inf")

    with pytest.raises(ValueError, match="gate_rich_candidate_feats"):
        module(**kwargs)


def test_v22_source_moe_accepts_exact_rich_feature_schema():
    module = SourceMoE(
        source_names=("default", "mask_text"),
        d_model=8,
        text_dim=8,
        hidden_dim=16,
        query_heads=4,
        use_fallback_gate=True,
        gate_hidden_dim=12,
        gate_candidate_top_k=2,
        gate_action_mode="cascade_v19_rich_set_correction",
    )
    output = module(
        candidate_feats=torch.randn(2, 5, 8),
        candidate_boxes=torch.rand(2, 5, 6),
        source_scores={
            "default": torch.randn(2, 5),
            "mask_text": torch.randn(2, 5),
        },
        gate_rich_candidate_feats=torch.randn(2, 5, 40),
    )

    assert module.gate_rich_feature_dim == 40
    assert output["selected_source_scores"].shape == (2, 5)


def test_moe_only_optimizer_freezes_backbone_and_trains_complete_moe():
    from types import SimpleNamespace
    from main_utils import BaseTrainTester

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
            )

    model = Toy()
    args = SimpleNamespace(
        source_choice_selector_train_only=False,
        source_moe_train_only=True,
        use_source_moe=True,
        source_moe_lr=2e-3,
        lr_backbone=1e-4,
        text_encoder_lr=1e-5,
        lr=1e-4,
        weight_decay=0.0,
    )
    optimizer = BaseTrainTester.get_optimizer(args, model)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.source_moe.parameters())
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-3)


def test_gate_only_optimizer_trains_exact_fallback_gate_allowlist():
    from types import SimpleNamespace
    from main_utils import BaseTrainTester

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_use_evidence_features=True,
                gate_evidence_dim=9,
            )

    model = Toy()
    args = SimpleNamespace(
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=True,
        use_source_moe=True,
        source_moe_use_fallback_gate=True,
        source_moe_gate_lr=7e-4,
        lr_backbone=1e-4,
        text_encoder_lr=1e-5,
        lr=1e-4,
        weight_decay=0.0,
    )
    optimizer = BaseTrainTester.get_optimizer(args, model)
    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith("source_moe.fallback_gate.")
        for name in trainable_names
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("source_moe.fallback_gate.")
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)


def test_cascade_optimizer_trains_only_84743_new_head_parameters():
    from types import SimpleNamespace
    from main_utils import BaseTrainTester

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=128,
                gate_candidate_top_k=2,
                gate_context_layers=1,
                gate_context_heads=4,
                gate_context_dropout=0.2,
                gate_action_mode="cascade_absolute_quality_correction",
            )

    model = Toy()
    args = SimpleNamespace(
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=True,
        source_moe_gate_new_heads_only=True,
        source_moe_gate_objective=(
            "cascade_absolute_quality_calibrated"
        ),
        use_source_moe=True,
        source_moe_use_fallback_gate=True,
        source_moe_gate_lr=3e-4,
        lr_backbone=1e-4,
        text_encoder_lr=1e-5,
        lr=1e-4,
        weight_decay=0.0,
    )

    optimizer = BaseTrainTester.get_optimizer(args, model)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    allowed = (
        "source_moe.fallback_gate.absolute_quality_head.",
        "source_moe.fallback_gate.cascade_quality_adapter.",
        "source_moe.fallback_gate.cascade_correction_head.",
    )
    assert trainable
    assert all(name.startswith(allowed) for name in trainable)
    assert sum(parameter.numel() for parameter in trainable.values()) == 84_743
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == {id(parameter) for parameter in trainable.values()}

    BaseTrainTester._set_source_moe_train_mode(model, args)
    gate = model.source_moe.fallback_gate
    assert model.training is False
    assert model.source_moe.training is False
    assert gate.training is False
    assert gate.context_encoder.training is False
    assert gate.encoder.training is False
    assert gate.absolute_quality_head.training is True
    assert gate.cascade_quality_adapter.training is True
    assert gate.cascade_correction_head.training is True

    outputs = gate(
        query_features=torch.randn(2, 4, gate.query_dim),
        candidate_scores=torch.tensor([[0.0, 3.0, 2.0, 1.0],
                                       [0.0, 3.0, 2.0, 1.0]]),
        shared_scores=torch.tensor([[4.0, 3.0, 2.0, 1.0],
                                    [4.0, 3.0, 2.0, 1.0]]),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=outputs["moe_gate_box_logits"],
        decision_logits=outputs["moe_gate_decision_logits"],
        box_ious=torch.tensor([[0.20, 0.70, 0.30, 0.10],
                               [0.70, 0.20, 0.30, 0.10]]),
        default_indices=outputs["moe_gate_action_anchor_query"],
        candidate_mask=outputs["moe_gate_candidate_mask"],
        action_margin=outputs["moe_gate_action_margin"],
        row_switch_margin=outputs["moe_gate_row_switch_margin"],
        absolute_box_logits=outputs["moe_gate_absolute_box_logits"],
        absolute_box_iou=outputs["moe_gate_absolute_box_iou"],
        setwise_temperature=0.25,
        objective="cascade_absolute_quality_calibrated",
    )
    result["loss"].backward()

    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in trainable
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable.values()
    )


@pytest.mark.parametrize(
    "action_mode,objective,expected_parameters,has_safety_head,has_joint_head",
    [
        (
            "cascade_opportunity_quality_correction",
            "cascade_opportunity_balanced_calibrated",
            118_024,
            False,
            False,
        ),
        (
            "cascade_opportunity_verified_correction",
            "cascade_opportunity_verified_calibrated",
            134_921,
            True,
            False,
        ),
        (
            "cascade_joint_risk_correction",
            "cascade_joint_risk_calibrated",
            152_202,
            True,
            True,
        ),
    ],
)
def test_opportunity_cascade_optimizer_trains_only_new_head_parameters(
        action_mode, objective, expected_parameters, has_safety_head,
        has_joint_head):
    from types import SimpleNamespace
    from main_utils import BaseTrainTester

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=128,
                gate_candidate_top_k=2,
                gate_context_layers=1,
                gate_context_heads=4,
                gate_context_dropout=0.2,
                gate_action_mode=action_mode,
            )

    model = Toy()
    args = SimpleNamespace(
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=True,
        source_moe_gate_new_heads_only=True,
        source_moe_gate_objective=objective,
        use_source_moe=True,
        source_moe_use_fallback_gate=True,
        source_moe_gate_lr=3e-4,
        lr_backbone=1e-4,
        text_encoder_lr=1e-5,
        lr=1e-4,
        weight_decay=0.0,
    )

    optimizer = BaseTrainTester.get_optimizer(args, model)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    allowed = (
        "source_moe.fallback_gate.absolute_quality_head.",
        "source_moe.fallback_gate.cascade_quality_adapter.",
        "source_moe.fallback_gate.cascade_correction_head.",
        "source_moe.fallback_gate.cascade_opportunity_head.",
    ) + ((
        "source_moe.fallback_gate.cascade_candidate_safety_head.",
    ) if has_safety_head else ()) + ((
        "source_moe.fallback_gate.cascade_joint_action_head.",
    ) if has_joint_head else ())
    assert trainable
    assert all(name.startswith(allowed) for name in trainable)
    assert sum(
        parameter.numel() for parameter in trainable.values()
    ) == expected_parameters
    if has_joint_head:
        assert len(trainable) == 30
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == {id(parameter) for parameter in trainable.values()}

    BaseTrainTester._set_source_moe_train_mode(model, args)
    gate = model.source_moe.fallback_gate
    assert model.training is False
    assert model.source_moe.training is False
    assert gate.training is False
    assert gate.absolute_quality_head.training is True
    assert gate.cascade_quality_adapter.training is True
    assert gate.cascade_correction_head.training is True
    assert gate.cascade_opportunity_head.training is True
    if has_safety_head:
        assert gate.cascade_candidate_safety_head.training is True
    if has_joint_head:
        assert gate.cascade_joint_action_head.training is True

    outputs = gate(
        query_features=torch.randn(2, 4, gate.query_dim),
        candidate_scores=torch.tensor([[0.0, 3.0, 2.0, 1.0],
                                       [0.0, 3.0, 2.0, 1.0]]),
        shared_scores=torch.tensor([[4.0, 3.0, 2.0, 1.0],
                                    [4.0, 3.0, 2.0, 1.0]]),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=outputs["moe_gate_box_logits"],
        decision_logits=outputs["moe_gate_decision_logits"],
        box_ious=torch.tensor([[0.20, 0.70, 0.30, 0.10],
                               [0.70, 0.20, 0.30, 0.10]]),
        default_indices=outputs["moe_gate_action_anchor_query"],
        candidate_mask=outputs["moe_gate_candidate_mask"],
        action_margin=outputs["moe_gate_action_margin"],
        row_switch_margin=outputs["moe_gate_row_switch_margin"],
        row_safety_margin=outputs.get("moe_gate_row_safety_margin"),
        joint_action_margin=outputs.get("moe_gate_joint_action_margin"),
        absolute_box_logits=outputs["moe_gate_absolute_box_logits"],
        absolute_box_iou=outputs["moe_gate_absolute_box_iou"],
        setwise_temperature=0.25,
        objective=objective,
    )
    result["loss"].backward()

    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in trainable
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable.values()
    )


def test_v21_optimizer_trains_only_149504_parameter_set_head():
    from types import SimpleNamespace
    from main_utils import BaseTrainTester

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=128,
                gate_candidate_top_k=2,
                gate_context_layers=0,
                gate_context_heads=4,
                gate_context_dropout=0.1,
                gate_action_mode="cascade_v19_fallback_set_correction",
            )

    model = Toy()
    args = SimpleNamespace(
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=True,
        source_moe_gate_new_heads_only=True,
        source_moe_gate_objective=(
            "cascade_v19_fallback_set_risk_calibrated"
        ),
        use_source_moe=True,
        source_moe_use_fallback_gate=True,
        source_moe_gate_lr=3e-4,
        lr_backbone=1e-4,
        text_encoder_lr=1e-5,
        lr=1e-4,
        weight_decay=0.0,
    )

    optimizer = BaseTrainTester.get_optimizer(args, model)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    prefix = (
        "source_moe.fallback_gate.cascade_fallback_set_action_head."
    )
    assert len(trainable) == 15
    assert all(name.startswith(prefix) for name in trainable)
    assert sum(parameter.numel() for parameter in trainable.values()) == 149_504
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == {id(parameter) for parameter in trainable.values()}

    BaseTrainTester._set_source_moe_train_mode(model, args)
    gate = model.source_moe.fallback_gate
    assert gate.training is False
    assert gate.absolute_quality_head.training is False
    assert gate.cascade_quality_adapter.training is False
    assert gate.cascade_correction_head.training is False
    assert gate.cascade_opportunity_head.training is False
    assert gate.cascade_candidate_safety_head.training is False
    assert gate.cascade_fallback_set_action_head.training is True

    outputs = gate(
        query_features=torch.randn(2, 4, gate.query_dim),
        candidate_scores=torch.tensor([
            [0.0, 3.0, 2.0, 1.0],
            [0.0, 3.0, 2.0, 1.0],
        ]),
        shared_scores=torch.tensor([
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
        ]),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=outputs["moe_gate_box_logits"],
        decision_logits=outputs["moe_gate_decision_logits"],
        box_ious=torch.tensor([
            [0.20, 0.70, 0.30, 0.10],
            [0.70, 0.20, 0.30, 0.10],
        ]),
        default_indices=outputs["moe_gate_supervision_fallback_query"],
        candidate_mask=outputs["moe_gate_candidate_mask"],
        action_margin=outputs["moe_gate_action_margin"],
        joint_action_margin=outputs["moe_gate_joint_action_margin"],
        setwise_temperature=0.25,
        objective="cascade_v19_fallback_set_risk_calibrated",
    )
    result["loss"].backward()
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in trainable
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable.values()
    )


def test_v22_optimizer_trains_only_169264_parameter_rich_set_head():
    from types import SimpleNamespace
    from main_utils import BaseTrainTester

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=64,
                text_dim=64,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=128,
                gate_candidate_top_k=2,
                gate_context_layers=0,
                gate_context_heads=4,
                gate_context_dropout=0.1,
                gate_action_mode="cascade_v19_rich_set_correction",
            )

    model = Toy()
    args = SimpleNamespace(
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=True,
        source_moe_gate_new_heads_only=True,
        source_moe_gate_objective="cascade_v19_rich_set_empirical_risk",
        use_source_moe=True,
        source_moe_use_fallback_gate=True,
        source_moe_gate_lr=3e-4,
        lr_backbone=1e-4,
        text_encoder_lr=1e-5,
        lr=1e-4,
        weight_decay=0.0,
    )

    optimizer = BaseTrainTester.get_optimizer(args, model)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    prefix = (
        "source_moe.fallback_gate."
        "cascade_rich_fallback_set_action_head."
    )
    assert len(trainable) == 17
    assert all(name.startswith(prefix) for name in trainable)
    assert sum(parameter.numel() for parameter in trainable.values()) == 169_264
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == {id(parameter) for parameter in trainable.values()}

    BaseTrainTester._set_source_moe_train_mode(model, args)
    gate = model.source_moe.fallback_gate
    assert gate.training is False
    assert gate.absolute_quality_head.training is False
    assert gate.cascade_quality_adapter.training is False
    assert gate.cascade_correction_head.training is False
    assert gate.cascade_opportunity_head.training is False
    assert gate.cascade_candidate_safety_head.training is False
    assert gate.cascade_rich_fallback_set_action_head.training is True

    outputs = gate(
        query_features=torch.randn(2, 4, gate.query_dim),
        candidate_scores=torch.tensor([
            [0.0, 3.0, 2.0, 1.0],
            [0.0, 3.0, 2.0, 1.0],
        ]),
        shared_scores=torch.tensor([
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
        ]),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
        rich_candidate_features=torch.randn(2, 4, 152),
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=outputs["moe_gate_box_logits"],
        decision_logits=outputs["moe_gate_decision_logits"],
        box_ious=torch.tensor([
            [0.20, 0.70, 0.30, 0.10],
            [0.70, 0.20, 0.30, 0.10],
        ]),
        default_indices=outputs["moe_gate_supervision_fallback_query"],
        candidate_mask=outputs["moe_gate_candidate_mask"],
        action_margin=outputs["moe_gate_action_margin"],
        joint_action_margin=outputs["moe_gate_joint_action_margin"],
        setwise_temperature=0.25,
        objective="cascade_v19_rich_set_empirical_risk",
    )
    result["loss"].backward()
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in trainable
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable.values()
    )


def test_v23_optimizer_trains_only_dense_head_and_adaptive_mixer():
    from types import SimpleNamespace
    from main_utils import BaseTrainTester

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.source_moe = SourceMoE(
                source_names=("default", "contrastive_text", "mask_text"),
                shared_source="default",
                d_model=64,
                text_dim=288,
                hidden_dim=288,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=128,
                gate_candidate_top_k=8,
                gate_context_layers=0,
                gate_context_heads=4,
                gate_context_dropout=0.1,
                gate_action_mode="cascade_v23_dense_quality_correction",
            )

    model = Toy()
    args = SimpleNamespace(
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=True,
        source_moe_gate_new_heads_only=True,
        source_moe_gate_objective="cascade_v23_dense_quality_risk",
        use_source_moe=True,
        source_moe_use_fallback_gate=True,
        source_moe_gate_lr=3e-4,
        lr_backbone=1e-4,
        text_encoder_lr=1e-5,
        lr=1e-4,
        weight_decay=0.0,
    )

    optimizer = BaseTrainTester.get_optimizer(args, model)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    prefixes = (
        "source_moe.adaptive_source_mixer.",
        "source_moe.fallback_gate.cascade_dense_quality_set_head.",
    )
    assert len(trainable) == 39
    assert all(name.startswith(prefixes) for name in trainable)
    assert sum(parameter.numel() for parameter in trainable.values()) == 588_603
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == {id(parameter) for parameter in trainable.values()}

    BaseTrainTester._set_source_moe_train_mode(model, args)
    gate = model.source_moe.fallback_gate
    assert gate.training is False
    assert gate.cascade_dense_quality_set_head.training is True
    assert model.source_moe.adaptive_source_mixer.training is True
    assert gate.cascade_opportunity_head.training is False
    assert gate.cascade_candidate_safety_head.training is False


@pytest.mark.parametrize(("action_mode", "objective"), (
    (
        "cascade_v25_pairwise_calibrated_correction",
        "cascade_v25_pairwise_calibrated_risk",
    ),
    (
        "cascade_v26_prior_restored_pairwise_correction",
        "cascade_v26_prior_restored_pairwise_risk",
    ),
))
def test_pairwise_optimizer_trains_only_dense_and_adaptive_heads(
        action_mode, objective):
    from types import SimpleNamespace
    from main_utils import BaseTrainTester

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.source_moe = SourceMoE(
                source_names=("default", "contrastive_text", "mask_text"),
                shared_source="default",
                d_model=64,
                text_dim=288,
                hidden_dim=288,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=128,
                gate_candidate_top_k=8,
                gate_context_layers=0,
                gate_context_heads=4,
                gate_context_dropout=0.1,
                gate_action_mode=action_mode,
            )

    model = Toy()
    args = SimpleNamespace(
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=True,
        source_moe_gate_new_heads_only=True,
        source_moe_gate_objective=objective,
        use_source_moe=True,
        source_moe_use_fallback_gate=True,
        source_moe_gate_lr=3e-4,
        lr_backbone=1e-4,
        text_encoder_lr=1e-5,
        lr=1e-4,
        weight_decay=0.0,
    )

    optimizer = BaseTrainTester.get_optimizer(args, model)
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    prefixes = (
        "source_moe.adaptive_source_mixer.",
        "source_moe.fallback_gate.cascade_dense_quality_set_head.",
        "source_moe.fallback_gate.cascade_pairwise_calibrated_set_head.",
    )
    assert len(trainable) == 69
    assert all(name.startswith(prefixes) for name in trainable)
    assert sum(parameter.numel() for parameter in trainable.values()) == 825_997
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == {id(parameter) for parameter in trainable.values()}

    BaseTrainTester._set_source_moe_train_mode(model, args)
    gate = model.source_moe.fallback_gate
    assert gate.training is False
    assert gate.cascade_dense_quality_set_head.training is True
    assert gate.cascade_pairwise_calibrated_set_head.training is True
    assert model.source_moe.adaptive_source_mixer.training is True
    assert gate.cascade_opportunity_head.training is False
    assert gate.cascade_candidate_safety_head.training is False


@pytest.mark.parametrize(
    "gate_only,action_mode,objective,error",
    [
        (
            False,
            "cascade_absolute_quality_correction",
            "cascade_absolute_quality_calibrated",
            "requires gate-only",
        ),
        (True, "pairwise_verifier", "pairwise_risk_calibrated", "cascade"),
        (
            True,
            "cascade_absolute_quality_correction",
            "pairwise_risk_calibrated",
            "calibrated objective",
        ),
        (
            True,
            "cascade_opportunity_quality_correction",
            "cascade_absolute_quality_calibrated",
            "matching cascade action",
        ),
    ],
)
def test_cascade_new_head_optimizer_fails_closed(
        gate_only, action_mode, objective, error):
    from types import SimpleNamespace
    from main_utils import BaseTrainTester

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode=action_mode,
            )

    args = SimpleNamespace(
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=gate_only,
        source_moe_gate_new_heads_only=True,
        source_moe_gate_objective=objective,
        use_source_moe=True,
        source_moe_use_fallback_gate=True,
        source_moe_gate_lr=3e-4,
        lr_backbone=1e-4,
        text_encoder_lr=1e-5,
        lr=1e-4,
        weight_decay=0.0,
    )
    with pytest.raises(ValueError, match=error):
        BaseTrainTester.get_optimizer(args, Toy())


def test_gate_only_inherits_non_tensor_candidate_config_before_model_build(
        tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "candidate.pth"
    torch.save({
        "config": SimpleNamespace(
            use_source_moe=True,
            source_choice_selector_sources="default,contrastive_text,mask_text",
            source_choice_selector_hidden_dim=288,
            source_moe_shared_source="default",
            source_moe_top_k=1,
            source_moe_query_layers=1,
            source_moe_query_heads=4,
            source_moe_query_dropout=0.1,
            source_moe_query_max_delta=0.10,
            source_moe_use_fallback_gate=False,
        ),
    }, checkpoint_path)
    args = SimpleNamespace(
        source_moe_gate_train_only=True,
        checkpoint_path=str(checkpoint_path),
        source_moe_query_max_delta=0.25,
        source_moe_top_k=2,
        source_moe_gate_candidate_top_k=8,
        source_moe_gate_break_cost=2.0,
        source_moe_gate_use_evidence_features=True,
        source_moe_gate_context_layers=1,
        source_moe_gate_context_heads=4,
        source_moe_gate_context_dropout=0.0,
        source_moe_gate_objective="calibrated_utility",
        source_moe_gate_setwise_temperature=0.25,
    )

    returned = prepare_source_moe_gate_checkpoint_config(args)

    assert returned is args
    assert args.use_source_moe is True
    assert args.source_moe_use_fallback_gate is True
    assert args.source_moe_query_max_delta == pytest.approx(0.10)
    assert args.source_moe_top_k == 1
    assert args.source_moe_gate_candidate_top_k == 8
    assert args.source_moe_gate_use_evidence_features is True
    assert args.source_moe_gate_context_layers == 1
    assert args.source_moe_gate_context_heads == 4
    assert args.source_moe_gate_context_dropout == pytest.approx(0.0)
    assert args.source_moe_gate_objective == "calibrated_utility"
    assert args.source_moe_gate_setwise_temperature == pytest.approx(0.25)
    assert args.source_moe_gate_uncertainty_weight == pytest.approx(0.0)


def test_source_moe_only_continuation_inherits_checkpoint_contract(tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "trained-contextual-moe.pth"
    torch.save({
        "config": {
            "use_source_moe": True,
            "source_choice_selector_sources": "default,mask_text",
            "source_choice_selector_hidden_dim": 192,
            "source_moe_shared_source": "default",
            "source_moe_top_k": 1,
            "source_moe_query_layers": 2,
            "source_moe_query_heads": 4,
            "source_moe_query_dropout": 0.2,
            "source_moe_query_max_delta": 0.1,
            "source_moe_use_fallback_gate": True,
            "source_moe_gate_hidden_dim": 64,
            "source_moe_gate_candidate_top_k": 6,
            "source_moe_gate_break_cost": 3.0,
            "source_moe_gate_decision_margin": 0.05,
            "source_moe_gate_mask_utility_weight": 0.4,
            "source_moe_gate_use_evidence_features": True,
            "source_moe_gate_objective": "balanced_calibrated_utility",
            "source_moe_gate_context_layers": 1,
            "source_moe_gate_context_heads": 4,
            "source_moe_gate_context_dropout": 0.0,
            "source_moe_gate_setwise_temperature": 0.25,
        },
    }, checkpoint_path)
    args = SimpleNamespace(
        eval=False,
        use_source_moe=True,
        source_moe_train_only=True,
        source_moe_gate_train_only=False,
        checkpoint_path=str(checkpoint_path),
        source_moe_query_max_delta=0.5,
        source_moe_use_fallback_gate=False,
    )

    prepare_source_moe_gate_checkpoint_config(args)

    assert args.source_choice_selector_sources == "default,mask_text"
    assert args.source_choice_selector_hidden_dim == 192
    assert args.source_moe_query_layers == 2
    assert args.source_moe_query_max_delta == pytest.approx(0.1)
    assert args.source_moe_use_fallback_gate is True
    assert args.source_moe_gate_hidden_dim == 64
    assert args.source_moe_gate_candidate_top_k == 6
    assert args.source_moe_gate_use_evidence_features is True
    assert args.source_moe_gate_context_layers == 1
    assert args.source_moe_gate_objective == "balanced_calibrated_utility"
    assert args.source_moe_gate_setwise_temperature == pytest.approx(0.25)


def test_source_moe_only_continuation_can_require_trained_gate(tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "router-without-gate.pth"
    torch.save({
        "config": {
            "use_source_moe": True,
            "source_choice_selector_sources": "default,mask_text",
            "source_choice_selector_hidden_dim": 192,
            "source_moe_shared_source": "default",
            "source_moe_top_k": 1,
            "source_moe_query_layers": 1,
            "source_moe_query_heads": 4,
            "source_moe_query_dropout": 0.1,
            "source_moe_query_max_delta": 0.1,
            "source_moe_use_fallback_gate": False,
        },
    }, checkpoint_path)
    args = SimpleNamespace(
        eval=False,
        use_source_moe=True,
        source_moe_train_only=True,
        source_moe_gate_train_only=False,
        checkpoint_path=str(checkpoint_path),
        source_moe_use_fallback_gate=True,
    )

    with pytest.raises(ValueError, match="trained fallback gate"):
        prepare_source_moe_gate_checkpoint_config(args)


def test_source_moe_only_initialization_keeps_requested_contract(tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "plain-mcln.pth"
    torch.save({"config": {"use_source_moe": False}}, checkpoint_path)
    args = SimpleNamespace(
        eval=False,
        use_source_moe=True,
        source_moe_train_only=True,
        source_moe_gate_train_only=False,
        checkpoint_path=str(checkpoint_path),
        source_moe_query_max_delta=0.25,
        source_moe_top_k=2,
        source_moe_use_fallback_gate=False,
    )

    returned = prepare_source_moe_gate_checkpoint_config(args)

    assert returned is args
    assert args.source_moe_query_max_delta == pytest.approx(0.25)
    assert args.source_moe_top_k == 2
    assert args.source_moe_use_fallback_gate is False


def test_source_moe_eval_inherits_checkpoint_inference_contract(tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "contextual-gate.pth"
    torch.save({
        "config": {
            "use_source_moe": True,
            "source_choice_selector_sources": "default,mask_text",
            "source_choice_selector_hidden_dim": 192,
            "source_moe_shared_source": "default",
            "source_moe_top_k": 1,
            "source_moe_query_layers": 2,
            "source_moe_query_heads": 4,
            "source_moe_query_dropout": 0.2,
            "source_moe_query_max_delta": 0.1,
            "source_moe_use_fallback_gate": True,
            "source_moe_gate_hidden_dim": 64,
            "source_moe_gate_candidate_top_k": 6,
            "source_moe_gate_break_cost": 3.0,
            "source_moe_gate_decision_margin": 0.05,
            "source_moe_gate_mask_utility_weight": 0.4,
            "source_moe_gate_use_evidence_features": True,
            "source_moe_gate_objective": "balanced_calibrated_utility",
            "source_moe_gate_context_layers": 1,
            "source_moe_gate_context_heads": 4,
            "source_moe_gate_context_dropout": 0.0,
            "source_moe_gate_setwise_temperature": 0.25,
        },
    }, checkpoint_path)
    args = SimpleNamespace(
        eval=True,
        use_source_moe=True,
        source_moe_gate_train_only=False,
        checkpoint_path=str(checkpoint_path),
        source_moe_query_max_delta=0.5,
        source_moe_use_fallback_gate=False,
        source_moe_gate_context_layers=0,
        source_moe_gate_setwise_temperature=0.0,
    )

    prepare_source_moe_gate_checkpoint_config(args)

    assert args.source_choice_selector_sources == "default,mask_text"
    assert args.source_choice_selector_hidden_dim == 192
    assert args.source_moe_top_k == 1
    assert args.source_moe_query_layers == 2
    assert args.source_moe_query_max_delta == pytest.approx(0.1)
    assert args.source_moe_use_fallback_gate is True
    assert args.source_moe_gate_hidden_dim == 64
    assert args.source_moe_gate_candidate_top_k == 6
    assert args.source_moe_gate_use_evidence_features is True
    assert args.source_moe_gate_context_layers == 1
    assert args.source_moe_gate_objective == "balanced_calibrated_utility"
    assert args.source_moe_gate_setwise_temperature == pytest.approx(0.25)


def test_source_moe_eval_inherits_gate_action_mode_or_honors_ablation(
        tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "expected-utility-gate.pth"
    torch.save({
        "config": {
            "use_source_moe": True,
            "source_choice_selector_sources": "default,mask_text",
            "source_choice_selector_hidden_dim": 128,
            "source_moe_shared_source": "default",
            "source_moe_top_k": 1,
            "source_moe_query_layers": 1,
            "source_moe_query_heads": 4,
            "source_moe_query_dropout": 0.1,
            "source_moe_query_max_delta": 0.1,
            "source_moe_use_fallback_gate": True,
            "source_moe_gate_hidden_dim": 64,
            "source_moe_gate_candidate_top_k": 6,
            "source_moe_gate_break_cost": 2.0,
            "source_moe_gate_decision_margin": 0.0,
            "source_moe_gate_mask_utility_weight": 0.25,
            "source_moe_gate_use_evidence_features": False,
            "source_moe_gate_action_mode": "expected_utility",
        },
    }, checkpoint_path)
    args = SimpleNamespace(
        eval=True,
        use_source_moe=True,
        source_moe_gate_train_only=False,
        source_moe_train_only=False,
        checkpoint_path=str(checkpoint_path),
        source_moe_use_fallback_gate=False,
        source_moe_gate_action_mode=None,
    )

    prepare_source_moe_gate_checkpoint_config(args)
    assert args.source_moe_gate_action_mode == "expected_utility"

    args.source_moe_gate_action_mode = "decision"
    prepare_source_moe_gate_checkpoint_config(args)
    assert args.source_moe_gate_action_mode == "decision"

    args.source_moe_gate_action_mode = "direct_utility"
    prepare_source_moe_gate_checkpoint_config(args)
    assert args.source_moe_gate_action_mode == "direct_utility"

    args.source_moe_gate_action_mode = "hierarchical_utility"
    prepare_source_moe_gate_checkpoint_config(args)
    assert args.source_moe_gate_action_mode == "hierarchical_utility"

    args.source_moe_gate_action_mode = "pairwise_verifier"
    prepare_source_moe_gate_checkpoint_config(args)
    assert args.source_moe_gate_action_mode == "pairwise_verifier"

    args.source_moe_gate_action_mode = "topn_pairwise_verifier"
    prepare_source_moe_gate_checkpoint_config(args)
    assert args.source_moe_gate_action_mode == "topn_pairwise_verifier"

    args.source_moe_gate_action_mode = "topn_dual_evidence_verifier"
    prepare_source_moe_gate_checkpoint_config(args)
    assert args.source_moe_gate_action_mode == (
        "topn_dual_evidence_verifier"
    )

    args.source_moe_gate_action_mode = "topn_absolute_quality_delta"
    prepare_source_moe_gate_checkpoint_config(args)
    assert args.source_moe_gate_action_mode == (
        "topn_absolute_quality_delta"
    )


def test_source_moe_eval_preserves_router_checkpoint_without_gate(tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "router.pth"
    torch.save({
        "config": SimpleNamespace(
            use_source_moe=True,
            source_choice_selector_sources="default,mask_text",
            source_choice_selector_hidden_dim=128,
            source_moe_shared_source="default",
            source_moe_top_k=1,
            source_moe_query_layers=1,
            source_moe_query_heads=4,
            source_moe_query_dropout=0.1,
            source_moe_query_max_delta=0.1,
            source_moe_use_fallback_gate=False,
        ),
    }, checkpoint_path)
    args = SimpleNamespace(
        eval=True,
        use_source_moe=True,
        source_moe_gate_train_only=False,
        checkpoint_path=str(checkpoint_path),
        source_moe_use_fallback_gate=True,
        source_moe_gate_use_evidence_features=True,
        source_moe_gate_context_layers=1,
    )

    prepare_source_moe_gate_checkpoint_config(args)

    assert args.source_moe_use_fallback_gate is False
    assert args.source_moe_gate_use_evidence_features is False
    assert args.source_moe_gate_context_layers == 0


def test_old_trained_gate_checkpoint_defaults_evidence_contract_to_false(
        tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "old-gate.pth"
    torch.save({
        "config": SimpleNamespace(
            use_source_moe=True,
            source_choice_selector_sources="default,contrastive_text,mask_text",
            source_choice_selector_hidden_dim=288,
            source_moe_shared_source="default",
            source_moe_top_k=1,
            source_moe_query_layers=1,
            source_moe_query_heads=4,
            source_moe_query_dropout=0.1,
            source_moe_query_max_delta=0.10,
            source_moe_use_fallback_gate=True,
            source_moe_gate_hidden_dim=128,
            source_moe_gate_candidate_top_k=8,
            source_moe_gate_break_cost=2.0,
            source_moe_gate_decision_margin=0.0,
            source_moe_gate_mask_utility_weight=0.25,
        ),
    }, checkpoint_path)
    args = SimpleNamespace(
        source_moe_gate_train_only=True,
        checkpoint_path=str(checkpoint_path),
        source_moe_gate_use_evidence_features=True,
    )

    prepare_source_moe_gate_checkpoint_config(args)

    assert args.source_moe_gate_use_evidence_features is False
    assert args.source_moe_gate_objective == "balanced_focal"
    assert args.source_moe_gate_context_layers == 0
    assert args.source_moe_gate_context_heads == 4
    assert args.source_moe_gate_context_dropout == pytest.approx(0.1)
    assert args.source_moe_gate_setwise_temperature == pytest.approx(0.0)


def test_trained_gate_checkpoint_inherits_calibrated_objective(tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "calibrated-gate.pth"
    torch.save({
        "config": {
            "use_source_moe": True,
            "source_choice_selector_sources": (
                "default,contrastive_text,mask_text"
            ),
            "source_choice_selector_hidden_dim": 288,
            "source_moe_shared_source": "default",
            "source_moe_top_k": 1,
            "source_moe_query_layers": 1,
            "source_moe_query_heads": 4,
            "source_moe_query_dropout": 0.1,
            "source_moe_query_max_delta": 0.10,
            "source_moe_use_fallback_gate": True,
            "source_moe_gate_hidden_dim": 128,
            "source_moe_gate_candidate_top_k": 8,
            "source_moe_gate_break_cost": 2.0,
            "source_moe_gate_decision_margin": 0.0,
            "source_moe_gate_mask_utility_weight": 0.25,
            "source_moe_gate_use_evidence_features": True,
            "source_moe_gate_objective": "calibrated_utility",
            "source_moe_gate_context_layers": 1,
            "source_moe_gate_context_heads": 4,
            "source_moe_gate_context_dropout": 0.0,
            "source_moe_gate_setwise_temperature": 0.25,
        },
    }, checkpoint_path)
    args = SimpleNamespace(
        source_moe_gate_train_only=True,
        checkpoint_path=str(checkpoint_path),
        source_moe_gate_objective="balanced_focal",
    )

    prepare_source_moe_gate_checkpoint_config(args)

    assert args.source_moe_gate_objective == "calibrated_utility"
    assert args.source_moe_gate_context_layers == 1
    assert args.source_moe_gate_context_heads == 4
    assert args.source_moe_gate_context_dropout == pytest.approx(0.0)
    assert args.source_moe_gate_setwise_temperature == pytest.approx(0.25)


def test_gate_only_explicit_objective_overrides_checkpoint_objective(tmp_path):
    from types import SimpleNamespace

    checkpoint_path = tmp_path / "trained-gate.pth"
    torch.save({
        "config": {
            "use_source_moe": True,
            "source_choice_selector_sources": "default,mask_text",
            "source_choice_selector_hidden_dim": 192,
            "source_moe_shared_source": "default",
            "source_moe_top_k": 1,
            "source_moe_query_layers": 1,
            "source_moe_query_heads": 4,
            "source_moe_query_dropout": 0.1,
            "source_moe_query_max_delta": 0.1,
            "source_moe_use_fallback_gate": True,
            "source_moe_gate_hidden_dim": 64,
            "source_moe_gate_candidate_top_k": 6,
            "source_moe_gate_break_cost": 2.0,
            "source_moe_gate_decision_margin": 0.0,
            "source_moe_gate_mask_utility_weight": 0.25,
            "source_moe_gate_use_evidence_features": False,
            "source_moe_gate_objective": "balanced_calibrated_utility",
            "source_moe_gate_context_layers": 0,
            "source_moe_gate_context_heads": 4,
            "source_moe_gate_context_dropout": 0.1,
            "source_moe_gate_setwise_temperature": 0.25,
        },
    }, checkpoint_path)
    args = SimpleNamespace(
        source_moe_gate_train_only=True,
        checkpoint_path=str(checkpoint_path),
        source_moe_gate_objective="hierarchical_risk_calibrated",
        source_moe_gate_objective_explicit=True,
        source_moe_gate_uncertainty_weight=0.5,
        source_moe_gate_uncertainty_weight_explicit=True,
    )

    prepare_source_moe_gate_checkpoint_config(args)

    assert args.source_moe_gate_objective == (
        "hierarchical_risk_calibrated"
    )
    assert args.source_moe_gate_uncertainty_weight == pytest.approx(0.5)


def test_trained_gate_checkpoint_contract_rejects_evidence_shape_mismatch():
    class Toy(torch.nn.Module):
        def __init__(self, evidence):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_use_evidence_features=evidence,
                gate_evidence_dim=9 if evidence else None,
            )

    old_model = Toy(evidence=False)
    enriched_model = Toy(evidence=True)
    old_checkpoint = {
        "config": {"source_moe_gate_use_evidence_features": False},
        "model": old_model.state_dict(),
    }
    validate_source_moe_gate_checkpoint_contract(old_model, old_checkpoint)
    with pytest.raises(ValueError, match="evidence contracts differ"):
        validate_source_moe_gate_checkpoint_contract(
            enriched_model, old_checkpoint
        )

    lying_checkpoint = dict(old_checkpoint)
    lying_checkpoint["config"] = {
        "source_moe_gate_use_evidence_features": True
    }
    with pytest.raises(ValueError, match="state is incompatible"):
        validate_source_moe_gate_checkpoint_contract(
            enriched_model, lying_checkpoint
        )


def test_trained_gate_checkpoint_contract_rejects_context_mismatch():
    class Toy(torch.nn.Module):
        def __init__(self, context_layers):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=16,
                gate_context_layers=context_layers,
                gate_context_heads=4,
                gate_context_dropout=0.0,
            )

    legacy_model = Toy(context_layers=0)
    contextual_model = Toy(context_layers=1)
    checkpoint = {
        "config": {
            "source_moe_gate_use_evidence_features": False,
            "source_moe_gate_context_layers": 0,
            "source_moe_gate_context_heads": 4,
            "source_moe_gate_context_dropout": 0.0,
        },
        "model": legacy_model.state_dict(),
    }

    validate_source_moe_gate_checkpoint_contract(legacy_model, checkpoint)
    with pytest.raises(ValueError, match="context contracts differ"):
        validate_source_moe_gate_checkpoint_contract(
            contextual_model, checkpoint
        )


def test_gate_checkpoint_contract_migrates_only_legacy_utility_head_state():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode="direct_utility",
            )

    model = Toy()
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if "fallback_gate.utility_head." not in key
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    legacy_checkpoint = {
        "config": dict(common_config, source_moe_gate_action_mode="decision"),
        "model": legacy_state,
    }

    validate_source_moe_gate_checkpoint_contract(model, legacy_checkpoint)

    invalid_direct_checkpoint = {
        "config": dict(
            common_config, source_moe_gate_action_mode="direct_utility"
        ),
        "model": legacy_state,
    }
    with pytest.raises(ValueError, match="utility_head"):
        validate_source_moe_gate_checkpoint_contract(
            model, invalid_direct_checkpoint
        )

    invalid_hierarchical_checkpoint = {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="hierarchical_utility",
        ),
        "model": legacy_state,
    }
    with pytest.raises(ValueError, match="row_switch_head|utility_head"):
        validate_source_moe_gate_checkpoint_contract(
            model, invalid_hierarchical_checkpoint
        )

    missing_pairwise_state = {
        key: value
        for key, value in model.state_dict().items()
        if "fallback_gate.pairwise_switch_head." not in key
    }
    legacy_without_pairwise = {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="hierarchical_utility",
        ),
        "model": missing_pairwise_state,
    }
    validate_source_moe_gate_checkpoint_contract(
        model, legacy_without_pairwise
    )

    invalid_pairwise_checkpoint = {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="pairwise_verifier",
        ),
        "model": missing_pairwise_state,
    }
    with pytest.raises(ValueError, match="pairwise_switch_head"):
        validate_source_moe_gate_checkpoint_contract(
            model, invalid_pairwise_checkpoint
        )

    invalid_topn_pairwise_checkpoint = {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="topn_pairwise_verifier",
        ),
        "model": missing_pairwise_state,
    }
    with pytest.raises(ValueError, match="pairwise_switch_head"):
        validate_source_moe_gate_checkpoint_contract(
            model, invalid_topn_pairwise_checkpoint
        )

    missing_safety_state = {
        key: value
        for key, value in model.state_dict().items()
        if "fallback_gate.safety_switch_head." not in key
    }
    migratable_pairwise_checkpoint = {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="topn_pairwise_verifier",
        ),
        "model": missing_safety_state,
    }
    validate_source_moe_gate_checkpoint_contract(
        model, migratable_pairwise_checkpoint
    )

    invalid_dual_checkpoint = {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="topn_dual_evidence_verifier",
        ),
        "model": missing_safety_state,
    }
    with pytest.raises(ValueError, match="safety_switch_head"):
        validate_source_moe_gate_checkpoint_contract(
            model, invalid_dual_checkpoint
        )


def test_absolute_quality_checkpoint_migrates_old_gate_and_fails_closed():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode="topn_absolute_quality_delta",
            )

    model = Toy()
    state_without_absolute = {
        key: value
        for key, value in model.state_dict().items()
        if "fallback_gate.absolute_quality_head." not in key
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    legacy_checkpoint = {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="topn_pairwise_verifier",
        ),
        "model": state_without_absolute,
    }
    validate_source_moe_gate_checkpoint_contract(
        model, legacy_checkpoint
    )

    invalid_absolute_checkpoint = {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="topn_absolute_quality_delta",
        ),
        "model": state_without_absolute,
    }
    with pytest.raises(ValueError, match="absolute_quality_head"):
        validate_source_moe_gate_checkpoint_contract(
            model, invalid_absolute_checkpoint
        )


def test_cascade_checkpoint_migrates_v12_and_requires_all_declared_heads():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode="cascade_absolute_quality_correction",
            )

    model = Toy()
    full_state = model.state_dict()
    new_module_fragments = (
        "fallback_gate.absolute_quality_head.",
        "fallback_gate.cascade_quality_adapter.",
        "fallback_gate.cascade_correction_head.",
    )
    v12_state = {
        key: value
        for key, value in full_state.items()
        if not any(fragment in key for fragment in new_module_fragments)
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    v12_checkpoint = {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="pairwise_verifier",
        ),
        "model": v12_state,
    }

    validate_source_moe_gate_checkpoint_contract(model, v12_checkpoint)

    for required_fragment in (
            "fallback_gate.utility_head.",
            "fallback_gate.pairwise_switch_head."):
        missing_legacy = {
            key: value
            for key, value in v12_state.items()
            if required_fragment not in key
        }
        with pytest.raises(ValueError, match=required_fragment.split(".")[1]):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": v12_checkpoint["config"],
                "model": missing_legacy,
            })

    for required_fragment in new_module_fragments:
        incomplete_cascade = {
            key: value
            for key, value in full_state.items()
            if required_fragment not in key
        }
        with pytest.raises(ValueError, match=required_fragment.split(".")[1]):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=(
                        "cascade_absolute_quality_correction"
                    ),
                ),
                "model": incomplete_cascade,
            })


def test_verified_opportunity_cascade_migrates_v12_v17_v18_and_fails_closed():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode=(
                    "cascade_opportunity_verified_correction"
                ),
            )

    model = Toy()
    full_state = model.state_dict()
    cascade_fragments = (
        "fallback_gate.absolute_quality_head.",
        "fallback_gate.cascade_quality_adapter.",
        "fallback_gate.cascade_correction_head.",
    )
    opportunity_fragment = "fallback_gate.cascade_opportunity_head."
    safety_fragment = "fallback_gate.cascade_candidate_safety_head."
    all_new_fragments = cascade_fragments + (
        opportunity_fragment, safety_fragment,
    )
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }

    v12_state = {
        key: value
        for key, value in full_state.items()
        if not any(fragment in key for fragment in all_new_fragments)
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="pairwise_verifier",
        ),
        "model": v12_state,
    })

    v17_state = {
        key: value
        for key, value in full_state.items()
        if opportunity_fragment not in key and safety_fragment not in key
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_absolute_quality_correction"
            ),
        ),
        "model": v17_state,
    })

    v18_state = {
        key: value
        for key, value in full_state.items()
        if safety_fragment not in key
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_opportunity_quality_correction"
            ),
        ),
        "model": v18_state,
    })

    for required_fragment in all_new_fragments:
        incomplete = {
            key: value
            for key, value in full_state.items()
            if required_fragment not in key
        }
        with pytest.raises(ValueError, match=required_fragment.split(".")[1]):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=(
                        "cascade_opportunity_verified_correction"
                    ),
                ),
                "model": incomplete,
            })


def test_joint_cascade_migrates_v12_v17_v18_v19_and_fails_closed():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode="cascade_joint_risk_correction",
            )

    model = Toy()
    full_state = model.state_dict()
    cascade_fragments = (
        "fallback_gate.absolute_quality_head.",
        "fallback_gate.cascade_quality_adapter.",
        "fallback_gate.cascade_correction_head.",
    )
    opportunity_fragment = "fallback_gate.cascade_opportunity_head."
    safety_fragment = "fallback_gate.cascade_candidate_safety_head."
    joint_fragment = "fallback_gate.cascade_joint_action_head."
    all_new_fragments = cascade_fragments + (
        opportunity_fragment,
        safety_fragment,
        joint_fragment,
    )
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    migrations = (
        (
            "pairwise_verifier",
            all_new_fragments,
        ),
        (
            "cascade_absolute_quality_correction",
            (opportunity_fragment, safety_fragment, joint_fragment),
        ),
        (
            "cascade_opportunity_quality_correction",
            (safety_fragment, joint_fragment),
        ),
        (
            "cascade_opportunity_verified_correction",
            (joint_fragment,),
        ),
    )
    for action_mode, removed_fragments in migrations:
        migrated_state = {
            key: value
            for key, value in full_state.items()
            if not any(fragment in key for fragment in removed_fragments)
        }
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=action_mode,
            ),
            "model": migrated_state,
        })

    for required_fragment in all_new_fragments:
        incomplete = {
            key: value
            for key, value in full_state.items()
            if required_fragment not in key
        }
        with pytest.raises(ValueError, match=required_fragment.split(".")[1]):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=(
                        "cascade_joint_risk_correction"
                    ),
                ),
                "model": incomplete,
            })


def test_v21_migrates_only_complete_v19_and_fails_closed():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode="cascade_v19_fallback_set_correction",
            )

    model = Toy()
    full_state = model.state_dict()
    set_fragment = "fallback_gate.cascade_fallback_set_action_head."
    v19_state = {
        key: value
        for key, value in full_state.items()
        if set_fragment not in key
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_opportunity_verified_correction"
            ),
        ),
        "model": v19_state,
    })
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_v19_fallback_set_correction"
            ),
        ),
        "model": full_state,
    })

    required_v19_fragments = (
        "fallback_gate.absolute_quality_head.",
        "fallback_gate.cascade_quality_adapter.",
        "fallback_gate.cascade_correction_head.",
        "fallback_gate.cascade_opportunity_head.",
        "fallback_gate.cascade_candidate_safety_head.",
    )
    for required_fragment in required_v19_fragments:
        incomplete = {
            key: value
            for key, value in v19_state.items()
            if required_fragment not in key
        }
        with pytest.raises(
                ValueError, match=required_fragment.split(".")[1]):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=(
                        "cascade_opportunity_verified_correction"
                    ),
                ),
                "model": incomplete,
            })

    for rejected_action in (
            "pairwise_verifier",
            "cascade_absolute_quality_correction",
            "cascade_opportunity_quality_correction",
            "cascade_joint_risk_correction"):
        with pytest.raises(ValueError, match="complete V19 initializer"):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=rejected_action,
                ),
                "model": v19_state,
            })


def test_v22_migrates_only_complete_v19_or_exact_v22_and_fails_closed():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "mask_text"),
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode="cascade_v19_rich_set_correction",
            )

    model = Toy()
    full_state = model.state_dict()
    rich_fragment = (
        "fallback_gate.cascade_rich_fallback_set_action_head."
    )
    v19_state = {
        key: value for key, value in full_state.items()
        if rich_fragment not in key
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_opportunity_verified_correction"
            ),
        ),
        "model": v19_state,
    })
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode="cascade_v19_rich_set_correction",
        ),
        "model": full_state,
    })

    required_v19_fragments = (
        "fallback_gate.absolute_quality_head.",
        "fallback_gate.cascade_quality_adapter.",
        "fallback_gate.cascade_correction_head.",
        "fallback_gate.cascade_opportunity_head.",
        "fallback_gate.cascade_candidate_safety_head.",
    )
    for required_fragment in required_v19_fragments:
        incomplete = {
            key: value for key, value in v19_state.items()
            if required_fragment not in key
        }
        with pytest.raises(
                ValueError, match=required_fragment.split(".")[1]):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=(
                        "cascade_opportunity_verified_correction"
                    ),
                ),
                "model": incomplete,
            })

    incomplete_v22 = {
        key: value for key, value in full_state.items()
        if not key.endswith(
            "cascade_rich_fallback_set_action_head.rich_norm.bias"
        )
    }
    with pytest.raises(ValueError, match="rich_norm.bias"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_v19_rich_set_correction"
                ),
            ),
            "model": incomplete_v22,
        })

    for rejected_action in (
            "cascade_joint_risk_correction",
            "cascade_v19_fallback_set_correction"):
        with pytest.raises(ValueError, match="complete V19 initializer"):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=rejected_action,
                ),
                "model": v19_state,
            })


def test_v23_migrates_only_complete_v19_or_exact_v23_and_fails_closed():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "contrastive_text", "mask_text"),
                shared_source="default",
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode="cascade_v23_dense_quality_correction",
            )

    model = Toy()
    full_state = model.state_dict()
    new_fragments = (
        "source_moe.adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
    )
    v19_state = {
        key: value for key, value in full_state.items()
        if not any(fragment in key for fragment in new_fragments)
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_opportunity_verified_correction"
            ),
        ),
        "model": v19_state,
    })
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_v23_dense_quality_correction"
            ),
        ),
        "model": full_state,
    })

    incomplete_v19 = {
        key: value for key, value in v19_state.items()
        if "fallback_gate.cascade_candidate_safety_head." not in key
    }
    with pytest.raises(ValueError, match="candidate_safety_head"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_opportunity_verified_correction"
                ),
            ),
            "model": incomplete_v19,
        })

    incomplete_v23 = {
        key: value for key, value in full_state.items()
        if not key.endswith("adaptive_source_mixer.mix_residual.2.bias")
    }
    with pytest.raises(ValueError, match="mix_residual.2.bias"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_v23_dense_quality_correction"
                ),
            ),
            "model": incomplete_v23,
        })

    for rejected_action in (
            "cascade_joint_risk_correction",
            "cascade_v19_fallback_set_correction",
            "cascade_v19_rich_set_correction"):
        with pytest.raises(ValueError, match="complete V19 initializer"):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=rejected_action,
                ),
                "model": v19_state,
            })


def test_v25_migrates_only_complete_v19_or_exact_v25_and_fails_closed():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "contrastive_text", "mask_text"),
                shared_source="default",
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode=(
                    "cascade_v25_pairwise_calibrated_correction"
                ),
            )

    model = Toy()
    full_state = model.state_dict()
    new_fragments = (
        "source_moe.adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
    )
    v19_state = {
        key: value for key, value in full_state.items()
        if not any(fragment in key for fragment in new_fragments)
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_opportunity_verified_correction"
            ),
        ),
        "model": v19_state,
    })
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_v25_pairwise_calibrated_correction"
            ),
        ),
        "model": full_state,
    })

    incomplete_v19 = {
        key: value for key, value in v19_state.items()
        if "fallback_gate.cascade_candidate_safety_head." not in key
    }
    with pytest.raises(ValueError, match="candidate_safety_head"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_opportunity_verified_correction"
                ),
            ),
            "model": incomplete_v19,
        })

    incomplete_v25 = {
        key: value for key, value in full_state.items()
        if not key.endswith(
            "cascade_pairwise_calibrated_set_head.utility_head.bias"
        )
    }
    with pytest.raises(ValueError, match="utility_head.bias"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_v25_pairwise_calibrated_correction"
                ),
            ),
            "model": incomplete_v25,
        })

    for rejected_action in (
            "cascade_joint_risk_correction",
            "cascade_v19_fallback_set_correction",
            "cascade_v19_rich_set_correction",
            "cascade_v23_dense_quality_correction",
            "cascade_v24_relative_risk_correction"):
        with pytest.raises(ValueError, match="complete V19 initializer"):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=rejected_action,
                ),
                "model": v19_state,
            })


def test_v26_migrates_only_complete_v19_or_exact_v26_and_rejects_v25():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "contrastive_text", "mask_text"),
                shared_source="default",
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode=(
                    "cascade_v26_prior_restored_pairwise_correction"
                ),
            )

    model = Toy()
    full_state = model.state_dict()
    new_fragments = (
        "source_moe.adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
    )
    v19_state = {
        key: value for key, value in full_state.items()
        if not any(fragment in key for fragment in new_fragments)
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_opportunity_verified_correction"
            ),
        ),
        "model": v19_state,
    })
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_v26_prior_restored_pairwise_correction"
            ),
        ),
        "model": full_state,
    })

    incomplete_v26 = {
        key: value for key, value in full_state.items()
        if not key.endswith(
            "cascade_pairwise_calibrated_set_head.benefit_head.bias"
        )
    }
    with pytest.raises(ValueError, match="benefit_head.bias"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_v26_prior_restored_pairwise_correction"
                ),
            ),
            "model": incomplete_v26,
        })

    for rejected_action in (
            "cascade_v24_relative_risk_correction",
            "cascade_v25_pairwise_calibrated_correction"):
        with pytest.raises(ValueError, match="complete V19 initializer"):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=rejected_action,
                ),
                "model": full_state,
            })


def test_v29_migrates_only_complete_v19_or_exact_v29_and_fails_closed():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "contrastive_text", "mask_text"),
                shared_source="default",
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode=(
                    "cascade_v29_counterfactual_selected_correction"
                ),
            )

    model = Toy()
    full_state = model.state_dict()
    new_fragments = (
        "source_moe.adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
        "fallback_gate.cascade_counterfactual_selected_risk_head.",
    )
    v19_state = {
        key: value for key, value in full_state.items()
        if not any(fragment in key for fragment in new_fragments)
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_opportunity_verified_correction"
            ),
        ),
        "model": v19_state,
    })
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_v29_counterfactual_selected_correction"
            ),
        ),
        "model": full_state,
    })

    incomplete_v29 = {
        key: value for key, value in full_state.items()
        if not key.endswith(
            "cascade_counterfactual_selected_risk_head.risk_head.4.bias"
        )
    }
    with pytest.raises(ValueError, match="risk_head.4.bias"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_v29_counterfactual_selected_correction"
                ),
            ),
            "model": incomplete_v29,
        })

    for rejected_action in (
            "cascade_v23_dense_quality_correction",
            "cascade_v25_pairwise_calibrated_correction",
            "cascade_v28_selected_abstention_correction"):
        with pytest.raises(ValueError, match="complete V19 initializer"):
            validate_source_moe_gate_checkpoint_contract(model, {
                "config": dict(
                    common_config,
                    source_moe_gate_action_mode=rejected_action,
                ),
                "model": v19_state,
            })


def test_v37_migrates_only_complete_v19_or_exact_v37_and_fails_closed():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "contrastive_text", "mask_text"),
                shared_source="default",
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode=(
                    "cascade_v37_counterfactual_benefit_hazard_correction"
                ),
            )

    model = Toy()
    full_state = model.state_dict()
    new_fragments = (
        "source_moe.adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
        "fallback_gate.cascade_counterfactual_benefit_hazard_head.",
    )
    v19_state = {
        key: value for key, value in full_state.items()
        if not any(fragment in key for fragment in new_fragments)
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_opportunity_verified_correction"
            ),
        ),
        "model": v19_state,
    })
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_v37_counterfactual_benefit_hazard_correction"
            ),
        ),
        "model": full_state,
    })

    incomplete_v37 = {
        key: value for key, value in full_state.items()
        if not key.endswith(
            "cascade_counterfactual_benefit_hazard_head.risk_head.4.bias"
        )
    }
    with pytest.raises(ValueError, match="risk_head.4.bias"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_v37_counterfactual_benefit_hazard_correction"
                ),
            ),
            "model": incomplete_v37,
        })


def test_v38_migrates_complete_v19_and_rejects_incomplete_exact_state():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "contrastive_text", "mask_text"),
                shared_source="default",
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode=(
                    "cascade_v38_complementary_logodds_correction"
                ),
            )

    model = Toy()
    full_state = model.state_dict()
    new_fragments = (
        "source_moe.adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
        "fallback_gate.cascade_counterfactual_logodds_head.",
    )
    v19_state = {
        key: value for key, value in full_state.items()
        if not any(fragment in key for fragment in new_fragments)
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_opportunity_verified_correction"
            ),
        ),
        "model": v19_state,
    })
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_v38_complementary_logodds_correction"
            ),
        ),
        "model": full_state,
    })

    incomplete = {
        key: value for key, value in full_state.items()
        if not key.endswith(
            "cascade_counterfactual_logodds_head.risk_head.4.bias"
        )
    }
    with pytest.raises(ValueError, match="risk_head.4.bias"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_v38_complementary_logodds_correction"
                ),
            ),
            "model": incomplete,
        })

    with pytest.raises(ValueError, match="complete V19 initializer"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_v29_counterfactual_selected_correction"
                ),
            ),
            "model": v19_state,
        })


def test_v39_migrates_complete_v19_and_rejects_incomplete_exact_state():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source_moe = SourceMoE(
                source_names=("default", "contrastive_text", "mask_text"),
                shared_source="default",
                d_model=8,
                text_dim=8,
                hidden_dim=16,
                query_heads=4,
                use_fallback_gate=True,
                gate_hidden_dim=12,
                gate_action_mode="cascade_v39_hazard_residual_correction",
            )

    model = Toy()
    full_state = model.state_dict()
    new_fragments = (
        "source_moe.adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
        "fallback_gate.cascade_counterfactual_hazard_residual_head.",
    )
    v19_state = {
        key: value for key, value in full_state.items()
        if not any(fragment in key for fragment in new_fragments)
    }
    common_config = {
        "source_moe_gate_use_evidence_features": False,
        "source_moe_gate_context_layers": 0,
        "source_moe_gate_context_heads": 4,
        "source_moe_gate_context_dropout": 0.1,
    }
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_opportunity_verified_correction"
            ),
        ),
        "model": v19_state,
    })
    validate_source_moe_gate_checkpoint_contract(model, {
        "config": dict(
            common_config,
            source_moe_gate_action_mode=(
                "cascade_v39_hazard_residual_correction"
            ),
        ),
        "model": full_state,
    })
    incomplete = {
        key: value for key, value in full_state.items()
        if not key.endswith(
            "cascade_counterfactual_hazard_residual_head.risk_head.4.bias"
        )
    }
    with pytest.raises(ValueError, match="risk_head.4.bias"):
        validate_source_moe_gate_checkpoint_contract(model, {
            "config": dict(
                common_config,
                source_moe_gate_action_mode=(
                    "cascade_v39_hazard_residual_correction"
                ),
            ),
            "model": incomplete,
        })


def test_joint_resume_rejects_changed_setwise_training_contract():
    from types import SimpleNamespace

    args = SimpleNamespace(
        use_source_moe=True,
        eval=False,
        reduce_lr=False,
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=False,
        source_moe_gate_objective="calibrated_utility",
        source_moe_gate_setwise_temperature=0.0,
    )
    checkpoint = {
        "config": {
            "use_source_moe": True,
            "source_moe_gate_objective": "calibrated_utility",
            "source_moe_gate_setwise_temperature": 0.25,
        }
    }

    with pytest.raises(
            ValueError, match="source_moe_gate_setwise_temperature"):
        validate_source_moe_resume_checkpoint_contract(args, checkpoint)


def test_gate_optimizer_resume_requires_gate_only_and_exact_contract():
    from types import SimpleNamespace

    checkpoint = {
        "config": {
            "use_source_moe": True,
            "source_moe_gate_action_mode": "pairwise_verifier",
            "source_moe_gate_objective": "pairwise_risk_calibrated",
            "source_moe_gate_setwise_temperature": 0.25,
            "source_moe_gate_lr": 3e-4,
        }
    }
    args = SimpleNamespace(
        use_source_moe=True,
        eval=False,
        reduce_lr=False,
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=True,
        source_moe_gate_resume_optimizer=True,
        source_moe_gate_action_mode="pairwise_verifier",
        source_moe_gate_objective="pairwise_risk_calibrated",
        source_moe_gate_setwise_temperature=0.25,
        source_moe_gate_lr=3e-4,
    )

    validate_source_moe_resume_checkpoint_contract(args, checkpoint)

    args.source_moe_gate_setwise_temperature = 0.5
    with pytest.raises(
            ValueError, match="source_moe_gate_setwise_temperature"):
        validate_source_moe_resume_checkpoint_contract(args, checkpoint)

    args.source_moe_gate_setwise_temperature = 0.25
    args.source_moe_gate_train_only = False
    with pytest.raises(ValueError, match="requires non-eval gate-only"):
        validate_source_moe_resume_checkpoint_contract(args, checkpoint)


def test_fresh_optimizer_allows_intentional_setwise_contract_change():
    from types import SimpleNamespace

    args = SimpleNamespace(
        use_source_moe=True,
        eval=False,
        reduce_lr=True,
        source_choice_selector_train_only=False,
        source_moe_train_only=False,
        source_moe_gate_train_only=False,
        source_moe_gate_objective="calibrated_utility",
        source_moe_gate_setwise_temperature=0.25,
    )
    checkpoint = {
        "config": {
            "use_source_moe": True,
            "source_moe_gate_objective": "balanced_focal",
            "source_moe_gate_setwise_temperature": 0.0,
        }
    }

    validate_source_moe_resume_checkpoint_contract(args, checkpoint)
