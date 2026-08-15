import torch

from scripts.run_v96_unbounded_listwise_hierarchical import (
    graded_listwise_loss,
    raw_utility,
)


def test_raw_utility_is_unbounded_and_uses_fixed_weights():
    logits = torch.tensor([[-4.0, 3.0], [4.0, -3.0]])
    expected = torch.tensor([-5.0, 5.0])
    torch.testing.assert_close(raw_utility(logits), expected)


def test_unbounded_two_level_loss_prefers_aligned_query_and_variant():
    query_valid = torch.zeros(1, 16, dtype=torch.bool)
    query_valid[:, :2] = True
    variant_valid = torch.zeros(1, 16, 7, dtype=torch.bool)
    variant_valid[:, 0, :2] = True
    variant_valid[:, 1, :2] = True
    candidate_ious = torch.zeros(1, 16, 7)
    candidate_ious[:, 0, :2] = torch.tensor([0.1, 0.2])
    candidate_ious[:, 1, :2] = torch.tensor([0.3, 0.8])
    aligned_query = torch.zeros(1, 16, 2, requires_grad=True)
    aligned_variant = torch.zeros(1, 16, 7, 2, requires_grad=True)
    aligned_query.data[:, 1, 0] = 4.0
    aligned_variant.data[:, 1, 1, 0] = 4.0
    reversed_query = torch.zeros(1, 16, 2)
    reversed_variant = torch.zeros(1, 16, 7, 2)
    reversed_query[:, 0, 0] = 4.0
    reversed_variant[:, 1, 0, 0] = 4.0
    aligned_loss, _ = graded_listwise_loss(
        {"query_logits": aligned_query, "variant_logits": aligned_variant},
        candidate_ious,
        query_valid,
        variant_valid,
    )
    reversed_loss, _ = graded_listwise_loss(
        {"query_logits": reversed_query, "variant_logits": reversed_variant},
        candidate_ious,
        query_valid,
        variant_valid,
    )
    assert aligned_loss < reversed_loss
    aligned_loss.backward()
    assert torch.isfinite(aligned_query.grad).all()
    assert torch.isfinite(aligned_variant.grad).all()
