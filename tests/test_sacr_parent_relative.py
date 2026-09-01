from types import SimpleNamespace

import pytest
import torch

from main_utils import load_checkpoint
from models.losses import compute_sacr_score_parent_relative_loss
from models.mcln_training_groups import parameter_group_name
from models.sacr_parent_relative import (
    SACRParentRelativeGate,
    apply_parent_relative_sacr_refinement,
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


def _write_score_checkpoint(path, parent_relative=False):
    config = {
        "use_sacr_score_refiner": True,
        "use_source_choice_selector": True,
        "source_choice_selector_sources": "default,rank",
        "source_choice_selector_hidden_dim": 288,
        "sacr_hidden_dim": 288,
        "sacr_max_pairs": 3,
        "sacr_top_m_targets": 32,
        "sacr_top_k_anchors": 16,
        "sacr_geo_dim": 16,
        "sacr_min_parse_confidence": 0.0,
        "sacr_score_max_delta": 0.25,
        "sacr_score_refiner_lr": 3e-4,
        "sacr_score_refiner_loss_weight": 1.0,
        "sacr_score_temperature": 0.1,
        "sacr_score_mask_weight": 0.25,
    }
    if parent_relative:
        config.update({
            "sacr_score_use_parent_relative_abstention": True,
            "sacr_score_parent_gate_hidden_dim": 32,
            "sacr_score_min_box_advantage": 0.07,
            "sacr_score_promotion_margin": 0.01,
            "sacr_score_mask_tolerance": 0.02,
            "sacr_score_raw_margin": 0.1,
            "sacr_score_dense_weight": 0.25,
            "sacr_score_preserve_weight": 1.0,
            "sacr_score_gate_weight": 0.05,
            "sacr_score_saturation_weight": 0.05,
        })
    state = {
        "structured_slot_builder.weight": torch.zeros(1),
        "sacr_head.weight": torch.zeros(1),
    }
    if parent_relative:
        state["sacr_parent_relative_gate.network.0.weight"] = torch.zeros(1)
    else:
        state["sacr_score_gate"] = torch.zeros(1)
    torch.save({"config": config, "model": state}, path)


class _LegacyScoreToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.structured_slot_builder = torch.nn.Linear(1, 1, bias=False)
        self.sacr_head = torch.nn.Linear(1, 1, bias=False)
        self.sacr_score_gate = torch.nn.Parameter(torch.zeros(1))


class _ParentRelativeScoreToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.structured_slot_builder = torch.nn.Linear(1, 1, bias=False)
        self.sacr_head = torch.nn.Linear(1, 1, bias=False)
        self.sacr_parent_relative_gate = torch.nn.Linear(1, 1)


def test_parent_relative_flag_is_default_off_and_explicitly_enabled():
    assert _parse([]).sacr_score_use_parent_relative_abstention is False
    assert _parse([
        "--sacr_score_use_parent_relative_abstention"
    ]).sacr_score_use_parent_relative_abstention is True
    assert parameter_group_name(
        "sacr_parent_relative_gate.network.0.weight"
    ) == "selector"


def test_v133_checkpoint_can_initialize_parent_relative_training(tmp_path):
    from main_utils import prepare_source_moe_gate_checkpoint_config

    checkpoint_path = tmp_path / "v133.pth"
    _write_score_checkpoint(checkpoint_path, parent_relative=False)
    args = _parse([
        "--checkpoint_path", str(checkpoint_path),
        "--use_sacr_score_refiner",
        "--sacr_score_refiner_train_only",
        "--sacr_score_use_parent_relative_abstention",
    ])

    prepared = prepare_source_moe_gate_checkpoint_config(args)

    assert prepared.sacr_score_use_parent_relative_abstention is True
    assert prepared._sacr_score_checkpoint_has_refiner is True


def test_parent_relative_checkpoint_restores_its_exact_runtime_contract(
        tmp_path):
    from main_utils import prepare_source_moe_gate_checkpoint_config

    checkpoint_path = tmp_path / "v134.pth"
    _write_score_checkpoint(checkpoint_path, parent_relative=True)
    args = _parse([
        "--checkpoint_path", str(checkpoint_path),
        "--use_sacr_score_refiner",
        "--eval",
    ])

    prepared = prepare_source_moe_gate_checkpoint_config(args)

    assert prepared.sacr_score_use_parent_relative_abstention is True
    assert prepared.sacr_score_min_box_advantage == pytest.approx(0.07)
    saved_state = torch.load(checkpoint_path, map_location="cpu")["model"]
    assert "sacr_score_gate" not in saved_state
    assert any(
        name.startswith("sacr_parent_relative_gate.")
        for name in saved_state
    )


def test_v133_checkpoint_cannot_be_reinterpreted_at_evaluation(tmp_path):
    from main_utils import prepare_source_moe_gate_checkpoint_config

    checkpoint_path = tmp_path / "v133.pth"
    _write_score_checkpoint(checkpoint_path, parent_relative=False)
    args = _parse([
        "--checkpoint_path", str(checkpoint_path),
        "--use_sacr_score_refiner",
        "--sacr_score_use_parent_relative_abstention",
        "--eval",
    ])

    with pytest.raises(ValueError, match="requires a checkpoint trained"):
        prepare_source_moe_gate_checkpoint_config(args)


def test_v133_initialization_discards_the_legacy_global_gate(tmp_path):
    source = _LegacyScoreToy()
    with torch.no_grad():
        source.structured_slot_builder.weight.fill_(2.0)
        source.sacr_head.weight.fill_(3.0)
        source.sacr_score_gate.fill_(4.0)
    checkpoint_path = tmp_path / "v133_full.pth"
    torch.save({
        "model": source.state_dict(),
        "config": {
            "use_sacr_score_refiner": True,
            "sacr_score_use_parent_relative_abstention": False,
        },
        "epoch": 2,
    }, checkpoint_path)
    target = _ParentRelativeScoreToy()
    args = SimpleNamespace(
        checkpoint_path=str(checkpoint_path),
        eval=False,
        reduce_lr=False,
        start_epoch=1,
        sacr_score_refiner_train_only=True,
        use_sacr_score_refiner=True,
        sacr_score_use_parent_relative_abstention=True,
    )

    load_checkpoint(
        args,
        target,
        torch.optim.SGD(target.parameters(), lr=0.1),
        scheduler=None,
    )

    assert "sacr_score_gate" not in target.state_dict()
    assert target.structured_slot_builder.weight.item() == pytest.approx(2.0)
    assert target.sacr_head.weight.item() == pytest.approx(3.0)


def test_parent_relative_deployment_is_exact_identity_for_uniform_raw_scores():
    parent_scores = torch.tensor([[1.0, 0.9, 0.6]])
    output = apply_parent_relative_sacr_refinement(
        raw_scores=torch.full((1, 3), 2.0),
        parent_scores=parent_scores,
        candidate_valid=torch.ones((1, 3), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        sample_gate=torch.ones(1),
        max_delta=0.25,
        promotion_margin=0.01,
    )

    assert torch.equal(output["scores"], parent_scores)
    assert torch.equal(output["residual"], torch.zeros_like(parent_scores))
    assert torch.equal(output["sample_gate"], torch.ones(1))


def test_parent_relative_deployment_changes_only_budget_feasible_candidates():
    parent_scores = torch.tensor([[1.0, 0.9, 0.6]])
    output = apply_parent_relative_sacr_refinement(
        raw_scores=torch.tensor([[0.0, 2.0, 4.0]]),
        parent_scores=parent_scores,
        candidate_valid=torch.ones((1, 3), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        sample_gate=torch.ones(1),
        max_delta=0.25,
        promotion_margin=0.01,
    )

    assert output["parent_indices"].tolist() == [0]
    assert output["feasible_candidate_mask"].tolist() == [
        [False, True, False]
    ]
    assert output["scores"][0, 0].item() == parent_scores[0, 0].item()
    assert output["scores"][0, 1].item() > parent_scores[0, 1].item()
    assert output["scores"][0, 2].item() == parent_scores[0, 2].item()
    assert output["residual"].abs().max().item() <= 0.25


def test_parent_relative_feasibility_reserves_the_promotion_margin():
    output = apply_parent_relative_sacr_refinement(
        raw_scores=torch.tensor([[0.0, 3.0]]),
        parent_scores=torch.tensor([[1.0, 0.755]]),
        candidate_valid=torch.ones((1, 2), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        sample_gate=torch.ones(1),
        max_delta=0.25,
        promotion_margin=0.01,
    )

    assert output["feasible_candidate_mask"].tolist() == [[False, False]]
    assert torch.equal(output["scores"], torch.tensor([[1.0, 0.755]]))


def test_parent_relative_gate_is_per_sample_and_strictly_abstains_without_feasible_candidate():
    gate = SACRParentRelativeGate(hidden_dim=8, top_k_anchors=4)
    output = gate(
        raw_scores=torch.tensor([
            [0.0, 0.3, -0.2],
            [0.0, 0.4, 0.2],
        ]),
        parent_scores=torch.tensor([
            [1.0, 0.9, 0.6],
            [1.0, 0.6, 0.5],
        ]),
        candidate_valid=torch.ones((2, 3), dtype=torch.bool),
        structured_valid_mask=torch.ones(2, dtype=torch.bool),
        parse_confidence=torch.tensor([0.9, 0.8]),
        anchor_top1_mass=torch.tensor([0.7, 0.6]),
        anchor_entropy=torch.tensor([0.2, 0.3]),
        relation_active_ratio=torch.tensor([1.0, 0.5]),
        max_delta=0.25,
        promotion_margin=0.01,
    )

    assert output["features"].shape == (2, 8)
    assert output["sample_gate"][0].item() == pytest.approx(
        torch.sigmoid(torch.tensor(-4.0)).item()
    )
    assert output["sample_gate"][1].item() == 0.0
    assert not any(name == "sacr_score_gate" for name, _ in gate.named_parameters())


def test_parent_relative_loss_trains_repair_and_abstention_rows():
    parent_scores = torch.tensor([
        [1.0, 0.9, 0.6],
        [1.0, 0.9, 0.8],
    ])
    raw_scores = torch.tensor([
        [0.0, 0.2, -0.1],
        [0.0, 0.3, 0.2],
    ], requires_grad=True)
    gate_logit = torch.tensor([1.0], requires_grad=True)
    deployment = apply_parent_relative_sacr_refinement(
        raw_scores=raw_scores,
        parent_scores=parent_scores,
        candidate_valid=torch.ones((2, 3), dtype=torch.bool),
        structured_valid_mask=torch.ones(2, dtype=torch.bool),
        sample_gate=gate_logit.sigmoid().expand(2),
        max_delta=0.25,
        promotion_margin=0.01,
    )
    supervision = compute_sacr_score_parent_relative_loss(
        scores=deployment["scores"],
        parent_scores=parent_scores,
        relative_raw_scores=deployment["relative_raw_scores"],
        sample_gate=deployment["sample_gate"],
        parent_indices=deployment["parent_indices"],
        feasible_candidate_mask=deployment["feasible_candidate_mask"],
        box_ious=torch.tensor([
            [0.40, 0.60, 0.70],
            [0.70, 0.50, 0.60],
        ]),
        mask_ious=torch.tensor([
            [0.50, 0.49, 0.80],
            [0.70, 0.50, 0.60],
        ]),
        valid_mask=deployment["apply_mask"],
        structured_valid_mask=torch.ones(2, dtype=torch.bool),
        sample_mask=torch.ones(2, dtype=torch.bool),
        mask_supervision_mask=torch.ones(2, dtype=torch.bool),
    )

    assert torch.isfinite(supervision["loss"])
    assert supervision["loss"].item() > 0.0
    assert supervision["repairable_row_ratio"].item() == pytest.approx(0.5)
    assert supervision["feasible_rank_loss"].item() > 0.0
    assert supervision["mask_supervised_row_ratio"].item() == pytest.approx(1.0)
    supervision["loss"].backward()
    assert raw_scores.grad is not None
    assert torch.isfinite(raw_scores.grad).all()
    assert raw_scores.grad.abs().sum().item() > 0.0
    assert gate_logit.grad is not None
    assert torch.isfinite(gate_logit.grad).all()
    assert gate_logit.grad.abs().sum().item() > 0.0


def test_parent_relative_loss_bounds_inherited_raw_score_scale():
    parent_scores = torch.tensor([[1.0, 0.9, 0.8]])
    raw_scores = torch.tensor(
        [[0.0, -1e15, 1e15]], requires_grad=True
    )
    deployment = apply_parent_relative_sacr_refinement(
        raw_scores=raw_scores,
        parent_scores=parent_scores,
        candidate_valid=torch.ones((1, 3), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        sample_gate=torch.tensor([0.018], requires_grad=True),
        max_delta=0.25,
        promotion_margin=0.01,
    )
    supervision = compute_sacr_score_parent_relative_loss(
        scores=deployment["scores"],
        parent_scores=parent_scores,
        relative_raw_scores=deployment["relative_raw_scores"],
        sample_gate=deployment["sample_gate"],
        parent_indices=deployment["parent_indices"],
        feasible_candidate_mask=deployment["feasible_candidate_mask"],
        box_ious=torch.tensor([[0.20, 0.60, 0.10]]),
        valid_mask=deployment["apply_mask"],
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        sample_mask=torch.ones(1, dtype=torch.bool),
        mask_supervision_mask=torch.zeros(1, dtype=torch.bool),
    )

    assert torch.isfinite(supervision["loss"])
    assert supervision["loss"].item() < 100.0
    supervision["loss"].backward()
    assert torch.isfinite(raw_scores.grad).all()


def test_parent_relative_loss_rejects_mask_unsafe_repair():
    parent_scores = torch.tensor([[1.0, 0.9]])
    raw_scores = torch.tensor([[0.0, 0.2]], requires_grad=True)
    gate = torch.tensor([0.8], requires_grad=True)
    deployment = apply_parent_relative_sacr_refinement(
        raw_scores=raw_scores,
        parent_scores=parent_scores,
        candidate_valid=torch.ones((1, 2), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        sample_gate=gate,
        max_delta=0.25,
        promotion_margin=0.01,
    )
    supervision = compute_sacr_score_parent_relative_loss(
        scores=deployment["scores"],
        parent_scores=parent_scores,
        relative_raw_scores=deployment["relative_raw_scores"],
        sample_gate=deployment["sample_gate"],
        parent_indices=deployment["parent_indices"],
        feasible_candidate_mask=deployment["feasible_candidate_mask"],
        box_ious=torch.tensor([[0.40, 0.70]]),
        mask_ious=torch.tensor([[0.80, 0.70]]),
        valid_mask=deployment["apply_mask"],
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        mask_supervision_mask=torch.ones(1, dtype=torch.bool),
    )

    assert supervision["repairable_row_ratio"].item() == 0.0
    assert supervision["mask_unsafe_candidate_ratio"].item() == 1.0
    assert supervision["preserve_sample_gate_mean"].item() > 0.0


def test_parent_relative_loss_keeps_small_threshold_crossing_repairs():
    parent_scores = torch.tensor([[1.0, 0.9]])
    raw_scores = torch.tensor([[0.0, 0.2]], requires_grad=True)
    deployment = apply_parent_relative_sacr_refinement(
        raw_scores=raw_scores,
        parent_scores=parent_scores,
        candidate_valid=torch.ones((1, 2), dtype=torch.bool),
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
        sample_gate=torch.tensor([0.8], requires_grad=True),
        max_delta=0.25,
        promotion_margin=0.01,
    )
    supervision = compute_sacr_score_parent_relative_loss(
        scores=deployment["scores"],
        parent_scores=parent_scores,
        relative_raw_scores=deployment["relative_raw_scores"],
        sample_gate=deployment["sample_gate"],
        parent_indices=deployment["parent_indices"],
        feasible_candidate_mask=deployment["feasible_candidate_mask"],
        box_ious=torch.tensor([[0.24, 0.26]]),
        valid_mask=deployment["apply_mask"],
        structured_valid_mask=torch.ones(1, dtype=torch.bool),
    )

    assert supervision["repairable_row_ratio"].item() == 1.0


def test_parent_relative_loss_is_differentiable_zero_without_supervision():
    scores = torch.tensor([[1.0, 0.9]], requires_grad=True)
    result = compute_sacr_score_parent_relative_loss(
        scores=scores,
        parent_scores=scores.detach(),
        relative_raw_scores=torch.zeros_like(scores, requires_grad=True),
        sample_gate=torch.zeros(1, requires_grad=True),
        parent_indices=torch.zeros(1, dtype=torch.long),
        feasible_candidate_mask=torch.tensor([[False, True]]),
        box_ious=torch.zeros_like(scores),
        valid_mask=torch.zeros_like(scores, dtype=torch.bool),
        structured_valid_mask=torch.zeros(1, dtype=torch.bool),
        sample_mask=torch.zeros(1, dtype=torch.bool),
    )

    assert result["loss"].item() == 0.0
    result["loss"].backward()
    assert scores.grad is not None
    assert torch.equal(scores.grad, torch.zeros_like(scores))
