import pytest
import torch

from models.rec_joint_box_mask import (
    LEGACY_MASK_POLICY_INDEX,
    MASK_POLICY_COUNT,
    JointBoxMaskAdapter,
    MASK_SOURCE_NAMES,
    compute_mask_candidate_targets,
    compute_weighted_mask_candidate_targets,
    compress_point_mask_to_superpoints,
    flat_to_parent_query,
    fuse_mask_logits,
    iou_tier,
    compute_joint_box_mask_losses,
    calibrate_mask_logits,
    select_joint_oracle,
)


def test_compute_mask_candidate_targets_uses_strict_thresholds():
    text = torch.tensor([[[2.0, -2.0], [2.0, -2.0]]])
    query = torch.tensor([[[-2.0, 2.0], [2.0, -2.0]]])
    gt = torch.tensor([[True, False]])

    out = compute_mask_candidate_targets(
        text,
        query,
        torch.tensor([0.5]),
        gt,
        torch.tensor([[True, True]]),
        torch.tensor([0.0]),
    )

    assert out["ious"].shape == (1, 2, 3, 1)
    assert out["source_names"] == ("text", "query", "fused")
    assert out["source_names"] == MASK_SOURCE_NAMES
    assert out["ious"][0, 1, 2, 0].item() == 1.0
    assert out["hits050"][0, 1, 2, 0].item() is True


@pytest.mark.parametrize(
    "logits,expected_iou",
    [
        ([2.0, 2.0, -2.0, -2.0], 0.5),
        ([2.0, 2.0, 2.0, 2.0], 0.25),
    ],
)
def test_compute_mask_candidate_targets_does_not_count_boundary_ious(
        logits, expected_iou):
    text = torch.tensor([[logits]])
    query = text.clone()
    gt = torch.tensor([[True, False, False, False]])

    out = compute_mask_candidate_targets(
        text,
        query,
        torch.tensor([0.25]),
        gt,
        torch.tensor([[True]]),
        torch.tensor([0.0]),
    )

    assert out["ious"][0, 0, 0, 0].item() == pytest.approx(expected_iou)
    assert out["hits025"][0, 0, 0, 0].item() is (expected_iou > 0.25)
    assert out["hits050"][0, 0, 0, 0].item() is False


def test_compute_mask_candidate_targets_masks_invalid_candidates():
    logits = torch.tensor([[[2.0, -2.0], [2.0, -2.0]]])
    out = compute_mask_candidate_targets(
        logits,
        logits,
        torch.tensor([0.5]),
        torch.tensor([[True, False]]),
        torch.tensor([[True, False]]),
        torch.tensor([0.0]),
    )

    assert torch.equal(out["ious"][0, 1], torch.zeros(3, 1))
    assert not bool(out["hits025"][0, 1].any().item())
    assert not bool(out["hits050"][0, 1].any().item())
    assert torch.equal(out["selected_superpoint_count"][0, 1],
                       torch.zeros(3, 1, dtype=torch.long))


def test_compute_mask_candidate_targets_rejects_row_without_candidate():
    logits = torch.zeros(1, 2, 3)

    with pytest.raises(ValueError, match="valid candidate"):
        compute_mask_candidate_targets(
            logits,
            logits,
            torch.tensor([0.5]),
            torch.tensor([[True, False, False]]),
            torch.tensor([[False, False]]),
            torch.tensor([0.0]),
        )


def test_compute_mask_candidate_targets_rejects_non_boolean_ground_truth():
    logits = torch.zeros(1, 1, 2)

    with pytest.raises(TypeError, match="bool"):
        compute_mask_candidate_targets(
            logits,
            logits,
            torch.tensor([0.5]),
            torch.tensor([[1, 0]]),
            torch.tensor([[True]]),
            torch.tensor([0.0]),
        )


def test_compute_mask_candidate_targets_rejects_non_finite_logits():
    text = torch.tensor([[[float("nan"), 0.0]]])

    with pytest.raises(ValueError, match="finite"):
        compute_mask_candidate_targets(
            text,
            torch.zeros_like(text),
            torch.tensor([0.5]),
            torch.tensor([[True, False]]),
            torch.tensor([[True]]),
            torch.tensor([0.0]),
        )


def test_compute_mask_candidate_targets_rejects_empty_union():
    logits = torch.full((1, 1, 2), -2.0)

    with pytest.raises(ValueError, match="empty union"):
        compute_mask_candidate_targets(
            logits,
            logits,
            torch.tensor([0.5]),
            torch.tensor([[False, False]]),
            torch.tensor([[True]]),
            torch.tensor([0.0]),
        )


def test_fuse_mask_logits_matches_current_expression_exactly():
    torch.manual_seed(0)
    text = torch.randn(2, 3, 5)
    query = torch.randn(2, 3, 5)
    alpha = torch.tensor([0.25, 0.75])

    actual = fuse_mask_logits(text, query, alpha)
    expected = alpha.reshape(2, 1, 1) * text + (
        1.0 - alpha.reshape(2, 1, 1)
    ) * query

    assert torch.equal(actual, expected)


def test_flat_variant_maps_to_parent_query():
    flat = torch.tensor([0, 6, 7, 111], dtype=torch.long)

    assert torch.equal(
        flat_to_parent_query(flat, 7), torch.tensor([0, 0, 1, 15])
    )


def test_flat_variant_rejects_non_int64_indices():
    with pytest.raises(TypeError, match="int64"):
        flat_to_parent_query(torch.tensor([0, 1], dtype=torch.int32), 7)


def test_flat_variant_rejects_non_positive_variant_count():
    with pytest.raises(ValueError, match="positive"):
        flat_to_parent_query(torch.tensor([0], dtype=torch.long), 0)


@pytest.mark.parametrize("variant_count", [True, False, 2.5, "7"])
def test_flat_variant_rejects_non_integer_variant_count(variant_count):
    with pytest.raises(TypeError, match="integer"):
        flat_to_parent_query(
            torch.tensor([0], dtype=torch.long), variant_count
        )


def test_joint_oracle_improves_mask_without_breaking_box_tiers():
    box = torch.tensor([[[0.60], [0.55], [0.20]]])
    mask = torch.tensor([[0.30, 0.80, 0.95]])

    out = select_joint_oracle(box, mask, torch.tensor([0]))

    assert out["selected_flat_index"].item() == 1
    assert out["selected_parent_query"].item() == 1
    assert out["selected_box_iou"].item() > 0.50
    assert out["selected_mask_iou"].item() == pytest.approx(0.80)
    assert not bool(out["position_break025"].any().item())
    assert not bool(out["position_break050"].any().item())
    assert out["mask_fix050"].item() is True


def test_joint_oracle_breaks_ties_by_box_then_flat_index():
    box = torch.tensor([[[0.60, 0.70], [0.80, 0.80]]])
    mask = torch.tensor([[0.50, 0.50]])

    out = select_joint_oracle(box, mask, torch.tensor([0]))

    assert out["selected_flat_index"].item() == 2
    assert out["selected_parent_query"].item() == 1


def test_joint_oracle_ignores_invalid_best_variant():
    box = torch.tensor([[[0.60], [0.70], [0.80]]])
    mask = torch.tensor([[0.30, 0.60, 0.95]])
    valid = torch.tensor([[[True], [True], [False]]])

    out = select_joint_oracle(
        box, mask, torch.tensor([0]), valid_mask=valid
    )

    assert out["selected_flat_index"].item() == 1


def test_joint_oracle_uses_strict_position_tiers():
    values = torch.tensor([0.25, 0.25001, 0.50, 0.50001])
    assert iou_tier(values).tolist() == [0, 1, 1, 2]

    box = torch.tensor([[[0.50], [0.25]]])
    mask = torch.tensor([[0.10, 0.90]])
    out = select_joint_oracle(box, mask, torch.tensor([0]))
    assert out["selected_flat_index"].item() == 0


def test_joint_oracle_retains_baseline_when_no_alternative_is_eligible():
    box = torch.tensor([[[0.60], [0.20]]])
    mask = torch.tensor([[0.40, 0.95]])

    out = select_joint_oracle(box, mask, torch.tensor([0]))

    assert out["selected_flat_index"].item() == 0
    assert out["selected_mask_iou"].item() == pytest.approx(0.40)


def test_joint_oracle_rejects_invalid_baseline_identity():
    box = torch.tensor([[[0.60], [0.20]]])
    mask = torch.tensor([[0.40, 0.95]])
    valid = torch.tensor([[[False], [True]]])

    with pytest.raises(ValueError, match="baseline.*valid"):
        select_joint_oracle(box, mask, torch.tensor([0]), valid_mask=valid)


def test_joint_adapter_outputs_exact_contract_and_identity_mask_policy():
    model = JointBoxMaskAdapter(input_dim=186, hidden_dim=32, dropout=0.1)
    out = model(
        torch.zeros(2, 16, 7, 186),
        torch.ones(2, 16, 7, dtype=torch.bool),
    )
    assert set(out) == {
        "box_logits", "mask_iou", "mask_logits", "log_scale",
        "mask_policy_logits",
    }
    assert out["box_logits"].shape == (2, 16, 7, 2)
    assert out["mask_iou"].shape == (2, 16, 7)
    assert out["mask_logits"].shape == (2, 16, 7, 2)
    assert out["log_scale"].shape == (2, 16, 7)
    assert out["mask_policy_logits"].shape == (2, 16, MASK_POLICY_COUNT)
    assert torch.equal(
        out["mask_policy_logits"].argmax(dim=-1),
        torch.full((2, 16), LEGACY_MASK_POLICY_INDEX),
    )
    calibrated = calibrate_mask_logits(
        torch.zeros(2, 16, 5), torch.ones(2, 16, 7, 3), torch.ones(2, 16, 7, 3)
    )
    assert calibrated["weight"].min().item() >= 0.0
    assert calibrated["weight"].max().item() <= 1.0
    assert calibrated["temperature_text"].min().item() >= 0.25
    assert calibrated["temperature_text"].max().item() <= 4.0
    assert calibrated["temperature_query"].min().item() >= 0.25
    assert calibrated["temperature_query"].max().item() <= 4.0
    assert calibrated["bias"].abs().max().item() <= 2.0
    assert calibrated["threshold"].abs().max().item() <= 1.0


def test_disabled_calibration_is_bitwise_current_fusion():
    torch.manual_seed(4)
    text = torch.randn(2, 3, 5)
    query = torch.randn(2, 3, 5)
    alpha = torch.tensor([0.2, 0.8])
    raw = torch.zeros(2, 3, 5)
    calibrated = calibrate_mask_logits(
        raw, text, query, disabled=True, legacy_alpha=alpha
    )
    expected = fuse_mask_logits(text, query, alpha)
    assert torch.equal(calibrated["logits"], expected)
    assert torch.equal(calibrated["binary"], expected > 0.0)


@pytest.mark.parametrize(
    "legacy_alpha,match",
    [
        (torch.tensor([float("nan")]), "finite"),
        (torch.tensor([-0.01]), r"\[0,1\]"),
        (torch.tensor([1.01]), r"\[0,1\]"),
    ],
)
def test_disabled_calibration_rejects_invalid_legacy_alpha(
        legacy_alpha, match):
    logits = torch.zeros(1, 2, 3)
    calibration = torch.zeros(1, 2, 5)

    with pytest.raises(ValueError, match=match):
        calibrate_mask_logits(
            calibration,
            logits,
            logits,
            disabled=True,
            legacy_alpha=legacy_alpha,
        )


def test_joint_losses_are_finite_and_ignore_invalid_candidates():
    torch.manual_seed(3)
    features = torch.randn(2, 3, 4, 6, requires_grad=True)
    valid = torch.tensor([[[True, True, False, False], [True, False, False, False],
                           [False, False, False, False]],
                          [[True, True, True, False], [True, True, False, False],
                           [True, False, False, False]]])
    model = JointBoxMaskAdapter(input_dim=6, hidden_dim=16)
    out = model(features, valid)
    targets = {
        "box_tier": torch.zeros(2, 3, 4, 2),
        "mask_iou": torch.rand(2, 3, 4),
        "mask_hits050": torch.zeros(2, 3, 4),
        "mask_policy_ious": torch.rand(2, 3, MASK_POLICY_COUNT),
    }
    losses = compute_joint_box_mask_losses(out, targets, valid)
    assert losses
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    policy_gradient = model.mask_policy_head.weight.grad
    assert policy_gradient is not None
    assert torch.isfinite(policy_gradient).all()
    assert policy_gradient.abs().sum().item() > 0.0


def test_weighted_superpoint_targets_match_expanded_point_iou():
    superpoints = torch.tensor([0, 0, 1, 1, 1, 2])
    point_target = torch.tensor([True, False, True, True, False, False])
    compressed = compress_point_mask_to_superpoints(
        point_target, superpoints, num_superpoints=3
    )
    assert compressed["point_counts"].tolist() == [2, 3, 1]
    assert compressed["target_counts"].tolist() == [1, 2, 0]

    text = torch.tensor([[[2.0, 2.0, -2.0]]])
    query = text.clone()
    result = compute_weighted_mask_candidate_targets(
        text,
        query,
        torch.tensor([0.5]),
        compressed["point_counts"].unsqueeze(0),
        compressed["target_counts"].unsqueeze(0),
        torch.tensor([[True]]),
        torch.tensor([0.0]),
    )
    expanded_prediction = torch.tensor([True, True, True, True, True, False])
    expected = (
        (expanded_prediction & point_target).sum().float()
        / (expanded_prediction | point_target).sum().float()
    )
    assert result["ious"][0, 0, 0, 0].item() == pytest.approx(expected.item())
