import torch
from torch import nn

from scripts import nr3d_point_detail_memory as detail
from utils.scatter_util import deterministic_scatter_mean_dim0


def scene():
    torch.manual_seed(19)
    features = torch.randn(9, 128)
    cloud = torch.randn(9, 6)
    # Native superpoint numbering may have missing slots.
    labels = torch.tensor([0, 2, 0, 3, 2, 3, 0, 2, 3])
    centers = deterministic_scatter_mean_dim0(cloud[:, :3], labels)
    return features, cloud, labels, centers


def test_zero_start_and_two_step_gradient_path():
    features, cloud, labels, centers = scene()
    addon = detail.PointDetailSuperpointResidual()
    assert sum(p.numel() for p in addon.parameters()) == 54144
    output = addon(features, cloud, labels, centers)
    assert torch.equal(output, torch.zeros(4, 288))
    target = torch.randn_like(output)
    (output - target).square().mean().backward()
    assert addon.output.weight.grad.norm() > 0
    assert addon.point_encoder[0].weight.grad.count_nonzero() == 0
    with torch.no_grad():
        addon.output.weight -= .1 * addon.output.weight.grad
    addon.zero_grad()
    (addon(features, cloud, labels, centers) - target).square().mean().backward()
    for parameter in addon.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


def test_membership_locality_missing_slot_and_point_order():
    features, cloud, labels, centers = scene()
    addon = detail.PointDetailSuperpointResidual()
    with torch.no_grad():
        addon.output.weight.normal_(0, .02)
    original = addon(features, cloud, labels, centers)
    order = torch.tensor([8, 0, 7, 1, 6, 2, 5, 3, 4])
    reordered = addon(features[order], cloud[order], labels[order], centers)
    assert torch.equal(original, reordered)
    changed_cloud = cloud.clone()
    changed_cloud[labels == 2, 3:] += 5
    changed = addon(features, changed_cloud, labels, centers)
    assert torch.equal(changed[[0, 1, 3]], original[[0, 1, 3]])
    assert not torch.equal(changed[2], original[2])
    assert torch.equal(original[1], torch.zeros(288))


def test_relative_geometry_is_translation_invariant():
    features, cloud, labels, centers = scene()
    addon = detail.PointDetailSuperpointResidual()
    with torch.no_grad():
        addon.output.weight.normal_(0, .02)
    shift = torch.tensor([4., -2., 1.])
    translated = cloud.clone()
    translated[:, :3] += shift
    assert torch.allclose(addon(features, cloud, labels, centers),
                          addon(features, translated, labels, centers + shift), atol=1e-6)


class SyntheticGrouper(nn.Module):
    def forward(self, xyz, centers, features):
        indices = torch.zeros(1, centers.shape[1], 2, dtype=torch.long)
        grouped = features[:, :, :1, None].expand(1, 288, centers.shape[1], 2)
        return grouped, indices


class SyntheticSA1(nn.Module):
    def forward(self, cloud):
        return cloud[:, :, :3], cloud.new_ones(cloud.shape[0], 128, cloud.shape[1]), None


class SyntheticMaskPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_net = nn.Module()
        self.backbone_net.sa1 = SyntheticSA1()
        self.super_grouper = SyntheticGrouper()
        self.base_feature = nn.Parameter(torch.ones(288))

    def forward(self, inputs):
        xyz, _, _ = self.backbone_net.sa1(inputs['point_clouds'])
        masks, indices = [], []
        for i in range(xyz.shape[0]):
            centers = deterministic_scatter_mean_dim0(xyz[i], inputs['superpoint'][i])
            grouped, index = self.super_grouper(
                xyz[i:i+1], centers[None], self.base_feature[None, :, None])
            masks.append(grouped.max(dim=-1)[0])
            indices.append(index)
        return masks, indices


def test_temporary_attachment_handles_batches_and_preserves_parent(monkeypatch):
    _, cloud, labels, _ = scene()
    model = SyntheticMaskPath().eval().requires_grad_(False)
    addon = detail.PointDetailSuperpointResidual()
    calls = []

    def synthetic_interpolation(point_xyz, anchor_xyz, anchor_features):
        # Tests hook routing only, not the native CUDA interpolation kernel.
        calls.append(point_xyz.clone())
        return anchor_features.T

    monkeypatch.setattr(detail, 'interpolate_sa1', synthetic_interpolation)
    batch = {'point_clouds': torch.stack((cloud, cloud + .1)),
             'superpoint': torch.stack((labels, labels))}
    before = {key: value.clone() for key, value in model.state_dict().items()}
    native_masks, native_indices = model(batch)
    attachment = detail.SuperpointDetailIntervention(model, addon)
    masks, indices = model(batch)
    assert len(calls) == 2
    assert all(torch.equal(a, b) for a, b in zip(masks, native_masks))
    assert all(torch.equal(a, b) for a, b in zip(indices, native_indices))
    assert attachment.inputs is None and attachment.anchors is None
    with torch.no_grad():
        addon.output.weight.normal_(0, .02)
    changed, _ = model(batch)
    assert len(calls) == 4
    assert all(not torch.equal(a, b) for a, b in zip(changed, native_masks))
    sum(value.square().sum() for value in changed).backward()
    assert addon.output.weight.grad.norm() > 0
    assert model.base_feature.grad is None
    attachment.remove()
    restored, _ = model(batch)
    assert len(calls) == 4
    assert all(torch.equal(a, b) for a, b in zip(restored, native_masks))
    assert before.keys() == model.state_dict().keys()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
