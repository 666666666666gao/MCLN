import pytest
import torch

from models.rec_relative_mask_policy import (
    ALLOWED_MASK_POLICY_INDICES,
    LEGACY_ALLOWED_POLICY_POSITION,
    RELATIVE_MASK_POLICY_COUNT,
    RelativeMaskTransitionPostprocessor,
    compute_relative_mask_policy_loss,
    select_relative_mask_policy_ensemble,
    select_relative_mask_policy_iou_priority_ensemble,
)
from models.rec_joint_box_mask import LEGACY_MASK_POLICY_INDEX


def _inputs(batch=2):
    geometry = torch.randn(batch, 16, 7, 179)
    mask = torch.randn(batch, 16, 52)
    valid = torch.ones(batch, 16, 7, dtype=torch.bool)
    valid[:, -1] = False
    return geometry, mask, valid


def _seed_output(batch=2):
    delta = torch.zeros(batch, 16, RELATIVE_MASK_POLICY_COUNT)
    probabilities = torch.zeros(
        batch, 16, RELATIVE_MASK_POLICY_COUNT, 2, 3
    )
    probabilities[..., 1] = 1.0
    return {
        "delta_iou": delta,
        "transition_probabilities": probabilities,
        "query_valid": torch.ones(batch, 16, dtype=torch.bool),
    }


def test_model_outputs_relative_transitions_and_masks_padding():
    model = RelativeMaskTransitionPostprocessor().eval()
    geometry, mask, valid = _inputs()
    with torch.no_grad():
        outputs = model(geometry, mask, valid)
    assert outputs["delta_iou"].shape == (
        2, 16, RELATIVE_MASK_POLICY_COUNT
    )
    assert outputs["transition_logits"].shape == (
        2, 16, RELATIVE_MASK_POLICY_COUNT, 2, 3
    )
    assert outputs["transition_probabilities"].shape == (
        2, 16, RELATIVE_MASK_POLICY_COUNT, 2, 3
    )
    assert torch.allclose(
        outputs["transition_probabilities"][:, :-1].sum(-1),
        torch.ones(2, 15, RELATIVE_MASK_POLICY_COUNT, 2),
    )
    assert torch.equal(
        outputs["delta_iou"][:, -1],
        torch.zeros(2, RELATIVE_MASK_POLICY_COUNT),
    )
    assert torch.equal(
        outputs["transition_probabilities"][:, -1],
        torch.zeros(2, RELATIVE_MASK_POLICY_COUNT, 2, 3),
    )


def test_loss_is_finite_uses_frozen_components_and_backpropagates():
    model = RelativeMaskTransitionPostprocessor()
    geometry, mask, valid = _inputs()
    outputs = model(geometry, mask, valid)
    policy_ious = torch.rand(2, 16, 15)
    loss, components = compute_relative_mask_policy_loss(
        outputs, policy_ious, valid.any(dim=2)
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "delta_iou", "transition025", "transition050",
        "listwise", "regret",
    }
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_three_seed_selector_requires_all_seed_heads_positive_and_keeps_parent():
    outputs = [_seed_output(), _seed_output(), _seed_output()]
    parents = torch.tensor([3, 5])
    # Candidate local 0 (original policy 4) is a unanimous safe switch on row 0.
    for output in outputs:
        output["delta_iou"][0, 3, 0] = 0.10
        output["transition_probabilities"][0, 3, 0, :, 0] = 0.10
        output["transition_probabilities"][0, 3, 0, :, 1] = 0.10
        output["transition_probabilities"][0, 3, 0, :, 2] = 0.80
        output["delta_iou"][1, 5, 1] = 0.10
        output["transition_probabilities"][1, 5, 1, :, 0] = 0.10
        output["transition_probabilities"][1, 5, 1, :, 1] = 0.10
        output["transition_probabilities"][1, 5, 1, :, 2] = 0.80
    # One seed predicts @.50 break risk on row 1, forcing legacy fallback.
    outputs[2]["transition_probabilities"][1, 5, 1, 1] = torch.tensor(
        [0.70, 0.20, 0.10]
    )
    result = select_relative_mask_policy_ensemble(outputs, parents)
    assert result["selected_parent_positions"].tolist() == [3, 5]
    assert result["selected_policy_indices"].tolist() == [
        ALLOWED_MASK_POLICY_INDICES[0], LEGACY_MASK_POLICY_INDEX,
    ]
    assert result["accepted"].tolist() == [True, False]
    assert result["worst_delta_iou"][0].item() == pytest.approx(0.10)
    assert result["worst_effects"][0].tolist() == pytest.approx([0.70, 0.70])


def test_selector_excludes_anchor_and_forbidden_policies_by_construction():
    outputs = [_seed_output(batch=1) for _ in range(3)]
    parent = torch.tensor([2])
    # Even an extreme learned anchor output must be mechanically neutralized.
    for output in outputs:
        output["delta_iou"][0, 2, LEGACY_ALLOWED_POLICY_POSITION] = 1.0
        output["transition_probabilities"][
            0, 2, LEGACY_ALLOWED_POLICY_POSITION, :, 0
        ] = 0.0
        output["transition_probabilities"][
            0, 2, LEGACY_ALLOWED_POLICY_POSITION, :, 1
        ] = 0.0
        output["transition_probabilities"][
            0, 2, LEGACY_ALLOWED_POLICY_POSITION, :, 2
        ] = 1.0
    result = select_relative_mask_policy_ensemble(outputs, parent)
    assert result["accepted"].item() is False
    assert result["selected_policy_indices"].item() == LEGACY_MASK_POLICY_INDEX
    assert set(ALLOWED_MASK_POLICY_INDICES) == set(range(4, 15))


def test_selector_rejects_non_three_seed_or_invalid_parent_inputs():
    with pytest.raises(ValueError, match="exactly three"):
        select_relative_mask_policy_ensemble(
            [_seed_output(batch=1)], torch.tensor([0])
        )
    outputs = [_seed_output(batch=1) for _ in range(3)]
    with pytest.raises(ValueError, match="parent"):
        select_relative_mask_policy_ensemble(outputs, torch.tensor([16]))


def test_iou_priority_selector_changes_only_eligible_candidate_ordering():
    outputs = [_seed_output(batch=1) for _ in range(3)]
    parent = torch.tensor([3])
    for output in outputs:
        # Original policy 4: lower delta-IoU but larger aggregate utility.
        output["delta_iou"][0, 3, 0] = 0.10
        output["transition_probabilities"][0, 3, 0, :, :] = torch.tensor(
            [[0.05, 0.05, 0.90], [0.05, 0.05, 0.90]]
        )
        # Original policy 5: higher delta-IoU with smaller positive effects.
        output["delta_iou"][0, 3, 1] = 0.20
        output["transition_probabilities"][0, 3, 1, :, :] = torch.tensor(
            [[0.20, 0.30, 0.50], [0.20, 0.30, 0.50]]
        )
    aggregate = select_relative_mask_policy_ensemble(outputs, parent)
    iou_priority = select_relative_mask_policy_iou_priority_ensemble(
        outputs, parent
    )
    assert aggregate["selected_policy_indices"].item() == 4
    assert iou_priority["selected_policy_indices"].item() == 5
    assert iou_priority["accepted"].item() is True
    assert iou_priority["ranking"] == "worst_delta_iou_then_aggregate"
    assert iou_priority["selected_parent_positions"].tolist() == [3]


def test_iou_priority_selector_ties_choose_lowest_original_index():
    outputs = [_seed_output(batch=1) for _ in range(3)]
    parent = torch.tensor([2])
    for output in outputs:
        for local in (0, 1):
            output["delta_iou"][0, 2, local] = 0.20
            output["transition_probabilities"][0, 2, local, :, :] = (
                torch.tensor([[0.10, 0.20, 0.70], [0.10, 0.20, 0.70]])
            )
    result = select_relative_mask_policy_iou_priority_ensemble(
        outputs, parent
    )
    assert result["selected_policy_indices"].item() == 4
