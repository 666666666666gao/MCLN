import torch

from scripts.nr3d_point_voxel_mapping import point_voxel_mapping


def test_every_point_keeps_an_inverse_id_and_its_actual_rgb():
    points = torch.tensor([[-1., 2., .1, .1, .2, .3],
                           [-.995, 2.005, .105, .4, .5, .6],
                           [-.995, 2.005, .105, .7, .8, .9],
                           [-.97, 2.015, .13, .2, .3, .4]])
    cells, inverse, local, origin = point_voxel_mapping(points)
    assert len(inverse) == len(points) and len(cells) == 2
    assert inverse[0] == inverse[1] == inverse[2] and inverse[3] != inverse[0]
    assert torch.bincount(inverse).sum() == len(points)
    assert torch.equal(local[:, 3:], points[:, 3:])
    restored = origin + (cells[inverse].float() + local[:, :3] + .5) * .02
    assert torch.allclose(restored, points[:, :3], atol=1e-6, rtol=0)
    assert ((local[:, :3] >= -.5) & (local[:, :3] < .5)).all()


def test_point_order_only_permutes_the_inverse_map_and_continuous_features():
    torch.manual_seed(19)
    points = torch.randn(25, 6)
    order = torch.randperm(len(points))
    cells, inverse, local, origin = point_voxel_mapping(points)
    reordered = point_voxel_mapping(points[order])
    assert torch.equal(cells, reordered[0]) and torch.equal(origin, reordered[3])
    assert torch.equal(inverse[order], reordered[1]) and torch.equal(local[order], reordered[2])


def test_points_sharing_a_cell_retain_distinct_continuous_offsets_and_rgb():
    points = torch.tensor([[0., 0., 0., .1, .2, .3],
                           [.005, .01, .015, .3, .2, .1]], requires_grad=True)
    cells, inverse, local, _ = point_voxel_mapping(points)
    assert len(cells) == 1 and inverse.tolist() == [0, 0]
    assert not torch.equal(local[0], local[1])
    local[:, 3:].square().sum().backward()
    assert torch.equal(points.grad[:, 3:], points.detach()[:, 3:] * 2)
