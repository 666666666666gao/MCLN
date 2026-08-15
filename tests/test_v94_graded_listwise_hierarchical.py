import torch

from scripts.run_v94_graded_listwise_hierarchical import (
    TARGET_TEMPERATURE,
    graded_listwise_loss,
    graded_quality,
    masked_soft_listwise_cross_entropy,
)


def test_graded_quality_preserves_continuous_iou_and_threshold_jumps():
    iou = torch.tensor([[[0.10, 0.25, 0.30, 0.50, 0.90]]])
    quality = graded_quality(iou)
    expected = torch.tensor([[[0.10, 0.25, 1.30, 1.50, 3.90]]])
    torch.testing.assert_close(quality, expected)


def test_masked_listwise_prefers_quality_order_and_has_finite_gradient():
    target = torch.tensor([[3.0, 1.0, 0.0, -7.0]])
    valid = torch.tensor([[True, True, True, False]])
    aligned = torch.tensor([[3.0, 1.0, 0.0, 100.0]], requires_grad=True)
    reversed_scores = torch.tensor([[0.0, 1.0, 3.0, -100.0]])
    aligned_loss = masked_soft_listwise_cross_entropy(
        aligned, target, valid, dim=1
    )
    reversed_loss = masked_soft_listwise_cross_entropy(
        reversed_scores, target, valid, dim=1
    )
    assert TARGET_TEMPERATURE == 0.25
    assert aligned_loss < reversed_loss
    aligned_loss.backward()
    assert torch.isfinite(aligned.grad).all()
    assert aligned.grad[0, 3].item() == 0.0


def test_two_level_loss_ignores_invalid_query_and_is_finite():
    query_logits = torch.zeros(1, 16, 2, requires_grad=True)
    variant_logits = torch.zeros(1, 16, 7, 2, requires_grad=True)
    query_valid = torch.zeros(1, 16, dtype=torch.bool)
    query_valid[:, :2] = True
    variant_valid = torch.zeros(1, 16, 7, dtype=torch.bool)
    variant_valid[:, 0, :3] = True
    variant_valid[:, 1, :2] = True
    candidate_ious = torch.zeros(1, 16, 7)
    candidate_ious[:, 0, :3] = torch.tensor([0.10, 0.60, 0.30])
    candidate_ious[:, 1, :2] = torch.tensor([0.20, 0.70])
    outputs = {
        "query_logits": query_logits,
        "variant_logits": variant_logits,
    }
    loss, stats = graded_listwise_loss(
        outputs, candidate_ious, query_valid, variant_valid
    )
    assert torch.isfinite(loss)
    assert stats["query_loss"] > 0.0
    assert stats["variant_loss"] > 0.0
    loss.backward()
    assert torch.isfinite(query_logits.grad).all()
    assert torch.isfinite(variant_logits.grad).all()
    assert variant_logits.grad[:, 2:].abs().sum().item() == 0.0

