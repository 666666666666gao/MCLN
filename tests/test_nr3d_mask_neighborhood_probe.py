import torch

from scripts.nr3d_mask_neighborhood_probe import NearestTwoGroup, describe_neighborhoods


def test_nearest_features_and_gradients_follow_spatial_order():
    xyz = torch.tensor([[[.15, 0., 0.], [.05, 0., 0.], [.08, 0., 0.]]])
    centers = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    features = torch.tensor([[[10., 20., 30.]]], requires_grad=True)
    grouped, indices = NearestTwoGroup()(xyz, centers, features)
    assert indices.tolist() == [[[1, 2], [0, 2]]]
    assert grouped.tolist() == [[[[20., 30.], [10., 30.]]]]
    grouped.sum().backward()
    assert features.grad.tolist() == [[[1., 1., 2.]]]


def test_empty_foreground_and_absent_slots_are_separate():
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    centers = torch.tensor([[0., 0., 0.], [1.4, 0., 0.], [0., 0., 0.]])
    native = torch.tensor([[0, 0], [0, 0], [0, 0]])
    nearest = torch.tensor([[0, 1], [1, 2], [0, 1]])
    result = describe_neighborhoods(xyz, centers, native, nearest,
                                    torch.tensor([0, 1, 1]), torch.tensor([False, True, True]),
                                    torch.tensor([0, 1, 2]), .2)
    assert result['counts']['present']['slots'] == 2
    majority = result['counts']['majority_positive']
    assert majority['slots'] == majority['empty_radius'] == majority['target_center_restored'] == 1
    assert majority['native_without_target_seed'] == 1
    assert majority['nearest_without_target_seed'] == 0
