import torch

from models.rec_query_mask_policy import (
    QueryMaskPolicyPostprocessor,
    compute_mask_policy_loss,
    monotone_mask_hit_probabilities,
    select_mask_only_policy,
)
from models.rec_joint_box_mask import LEGACY_MASK_POLICY_INDEX


def _inputs(batch=2):
    features = torch.randn(batch, 16, 7, 179)
    mask_features = torch.randn(batch, 16, 52)
    valid = torch.ones(batch, 16, 7, dtype=torch.bool)
    valid[:, -1] = False
    return features, mask_features, valid


def test_model_outputs_dense_monotone_policies_and_masks_padding():
    model = QueryMaskPolicyPostprocessor().eval()
    features, mask_features, valid = _inputs()
    with torch.no_grad():
        outputs = model(features, mask_features, valid)
    assert outputs["iou"].shape == (2, 16, 15)
    assert outputs["hit_logits"].shape == (2, 16, 15, 2)
    assert outputs["hit_probabilities"].shape == (2, 16, 15, 2)
    assert torch.all(outputs["hit_probabilities"][..., 1]
                     <= outputs["hit_probabilities"][..., 0])
    assert torch.equal(outputs["iou"][:, -1], torch.zeros(2, 15))


def test_mask_only_policy_keeps_parent_and_falls_back_unless_all_heads_gain():
    iou = torch.full((2, 16, 15), 0.4)
    hits = torch.full((2, 16, 15, 2), 0.4)
    parents = torch.tensor([3, 5])
    iou[0, 3, LEGACY_MASK_POLICY_INDEX] = 0.5
    hits[0, 3, LEGACY_MASK_POLICY_INDEX] = torch.tensor([0.5, 0.4])
    iou[0, 3, 2] = 0.7
    hits[0, 3, 2] = torch.tensor([0.7, 0.6])
    # Row 1 has utility gain but loses @.50, so it must fall back.
    iou[1, 5, LEGACY_MASK_POLICY_INDEX] = 0.5
    hits[1, 5, LEGACY_MASK_POLICY_INDEX] = torch.tensor([0.5, 0.5])
    iou[1, 5, 4] = 0.9
    hits[1, 5, 4] = torch.tensor([0.9, 0.4])
    result = select_mask_only_policy({
        "iou": iou,
        "hit_probabilities": hits,
        "query_valid": torch.ones(2, 16, dtype=torch.bool),
    }, parents)
    assert result["selected_parent_positions"].tolist() == [3, 5]
    assert result["selected_policy_indices"].tolist() == [2, LEGACY_MASK_POLICY_INDEX]
    assert result["accepted"].tolist() == [True, False]


def test_policy_loss_is_finite_and_backpropagates():
    model = QueryMaskPolicyPostprocessor()
    features, mask_features, valid = _inputs()
    outputs = model(features, mask_features, valid)
    labels = torch.rand(2, 16, 15)
    loss, components = compute_mask_policy_loss(
        outputs, labels, valid.any(dim=2)
    )
    assert torch.isfinite(loss)
    assert set(components) == {"iou", "hit", "listwise", "regret"}
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_monotone_probability_rejects_nonfinite_logits():
    logits = torch.tensor([[0.0, float("nan")]])
    try:
        monotone_mask_hit_probabilities(logits)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite logits must fail closed")
