import torch

from scripts.nr3d_mask_supervision_probe import query_gradient_support, superpoint_neighborhood_counts


def test_query_gradient_support_preserves_unmatched_zero_and_disconnected_state():
    logits = torch.tensor([[.2, -.3], [.1, .4], [-.1, .2]], requires_grad=True)
    target = torch.tensor([[1., 0.]])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[[1]], target)
    gradient, = torch.autograd.grad(loss, [logits])
    assert query_gradient_support(gradient)['nonzero_query_ids'] == [1]
    assert query_gradient_support(gradient)['connected']
    assert not query_gradient_support(None)['connected']


def test_target_superpoint_neighbor_count_is_seed_membership_not_receptive_field():
    superpoints = torch.tensor([0, 0, 1, 1, 1, 3])
    target = torch.tensor([True, True, True, False, False, True])
    seeds = torch.tensor([0, 3, 4])
    neighbors = torch.tensor([[1, 2], [0, 1], [0, 0], [0, 2]])
    result = superpoint_neighborhood_counts(superpoints, target, seeds, neighbors, 4)
    assert result['majority_positive_slots'] == 2
    assert result['majority_slots_without_target_seed_center'] == 1
    assert result['target_touching_slots_without_target_seed_center'] == 1
    assert result['majority_target_neighbor_counts'] == [0, 1]
    assert not result['neighbor_centers_measure_receptive_field_coverage']
