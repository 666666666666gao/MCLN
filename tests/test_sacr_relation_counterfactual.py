import pytest
import torch

from models.sacr_relation_counterfactual import (
    RELATION_COUNTERFACTUAL_TRAINABLE_PREFIXES,
    apply_relation_counterfactual_refinement,
    compute_relation_counterfactual_loss,
)


def _parse(argv):
    import sys
    from main_utils import parse_option

    saved = sys.argv
    try:
        sys.argv = ["prog"] + argv
        return parse_option()
    finally:
        sys.argv = saved


def test_relation_counterfactual_flag_is_explicit_and_default_off():
    assert _parse([]).sacr_score_use_relation_counterfactual is False
    assert _parse([
        "--sacr_score_use_relation_counterfactual"
    ]).sacr_score_use_relation_counterfactual is True


def test_relation_counterfactual_trains_only_differentiable_relation_paths():
    names = RELATION_COUNTERFACTUAL_TRAINABLE_PREFIXES
    assert "structured_slot_builder.rel_attn." in names
    assert "structured_slot_builder.anchor_attn." in names
    assert "sacr_head.anchor_mlp." in names
    assert "sacr_head.relation_mlp." in names
    assert not any("target_attr_mlp" in name for name in names)
    assert not any("global_mlp" in name for name in names)


def test_relation_counterfactual_deployment_preserves_parent_and_promotes_only_supported_swap():
    parent_scores = torch.tensor([[1.0, 0.90, 0.88, 0.50]])
    relation_scores = torch.tensor([[0.0, 0.4, -0.3, 0.8]])
    geometry = torch.zeros(1, 4, 11)
    output = apply_relation_counterfactual_refinement(
        relation_scores=relation_scores,
        geometry_signatures=geometry,
        relation_candidate_mask=torch.ones((1, 4), dtype=torch.bool),
        target_affinity=torch.tensor([[0.80, 0.78, 0.79, 0.10]]),
        attribute_affinity=torch.tensor([[0.50, 0.48, 0.52, 0.10]]),
        attribute_present=torch.ones(1, dtype=torch.bool),
        parent_scores=parent_scores,
        candidate_valid=torch.ones((1, 4), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        parse_confidence=torch.ones(1),
        anchor_top1_mass=torch.ones(1),
        max_delta=0.25,
        promotion_margin=0.01,
        parent_top_k=4,
        target_tolerance=0.05,
        attribute_tolerance=0.05,
        relation_scale=4.0,
        deployment_threshold=0.05,
    )

    assert output["parent_indices"].tolist() == [0]
    assert output["proposal_mask"].tolist() == [[False, True, True, False]]
    assert output["promotion_mask"].tolist() == [[False, True, False, False]]
    assert output["scores"][0, 0].item() == parent_scores[0, 0].item()
    assert output["scores"][0, 1].item() > parent_scores[0, 0].item()
    assert output["scores"][0, 2].item() == parent_scores[0, 2].item()
    assert output["residual"].abs().max().item() <= 0.25


def test_relation_counterfactual_loss_mines_same_target_geometry_mismatch_only():
    relation_scores = torch.tensor(
        [[0.5, 0.1, -0.2, 0.9]], requires_grad=True
    )
    geometry = torch.zeros(1, 4, 11)
    geometry[0, 0] = 1.0
    geometry[0, 2] = -1.0
    geometry[0, 3] = 1.0
    supervision = compute_relation_counterfactual_loss(
        relation_scores=relation_scores,
        geometry_signatures=geometry,
        relation_candidate_mask=torch.ones((1, 4), dtype=torch.bool),
        target_affinity=torch.tensor([[0.78, 0.80, 0.77, 0.10]]),
        attribute_affinity=torch.tensor([[0.48, 0.50, 0.52, 0.10]]),
        attribute_present=torch.ones(1, dtype=torch.bool),
        parent_scores=torch.tensor([[1.0, 0.90, 0.85, 0.80]]),
        candidate_valid=torch.ones((1, 4), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        box_ious=torch.tensor([[0.10, 0.70, 0.20, 0.05]]),
        parent_top_k=4,
        target_tolerance=0.05,
        attribute_tolerance=0.05,
        geometry_threshold=0.08,
        iou_gap=0.10,
        pair_margin=0.25,
        max_negatives=4,
    )

    assert torch.isfinite(supervision["loss"])
    assert supervision["loss"].item() > 0.0
    assert supervision["active_row_ratio"].item() == pytest.approx(1.0)
    assert supervision["hard_negative_row_ratio"].item() == pytest.approx(1.0)
    assert supervision["hard_negative_count_mean"].item() == pytest.approx(2.0)
    assert supervision["parent_hard_negative_ratio"].item() == pytest.approx(1.0)
    supervision["loss"].backward()
    assert relation_scores.grad is not None
    assert torch.isfinite(relation_scores.grad).all()
    assert relation_scores.grad.abs().sum().item() > 0.0


def test_relation_counterfactual_loss_ignores_geometry_consistent_candidate():
    geometry = torch.zeros(1, 3, 11)
    supervision = compute_relation_counterfactual_loss(
        relation_scores=torch.tensor([[0.5, 0.1, 0.6]], requires_grad=True),
        geometry_signatures=geometry,
        relation_candidate_mask=torch.ones((1, 3), dtype=torch.bool),
        target_affinity=torch.tensor([[0.78, 0.80, 0.79]]),
        attribute_affinity=torch.zeros(1, 3),
        attribute_present=torch.zeros(1, dtype=torch.bool),
        parent_scores=torch.tensor([[1.0, 0.9, 0.8]]),
        candidate_valid=torch.ones((1, 3), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        box_ious=torch.tensor([[0.10, 0.70, 0.20]]),
        parent_top_k=3,
        geometry_threshold=0.08,
    )

    assert supervision["hard_negative_count_mean"].item() == 0.0
    assert supervision["loss"].item() == 0.0
