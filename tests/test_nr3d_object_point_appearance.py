import torch
from torch import nn

from scripts.nr3d_object_point_appearance import (
    LastDecoderObjectAppearanceIntervention, ObjectPointAppearanceResidual)


def example():
    torch.manual_seed(23)
    points = torch.cat((torch.randn(2, 12, 3) * .1, torch.rand(2, 12, 3)), dim=-1)
    boxes = torch.zeros(2, 3, 6)
    boxes[:, :2, 3:] = 1
    boxes[:, 1, 0] = .1
    valid = torch.tensor([[True, True, False], [True, False, False]])
    return points, boxes, valid


def test_zero_output_and_gradients_use_real_crops_and_leave_padding_empty():
    points, boxes, valid = example()
    addon = ObjectPointAppearanceResidual()
    output = addon(points, boxes, valid)
    assert output.shape == (2, 3, 288) and torch.count_nonzero(output) == 0
    target = torch.randn_like(output)
    (output * target).sum().backward()
    assert addon.output.weight.grad.norm() > 0
    for parameter in addon.point_encoder.parameters():
        assert parameter.grad is not None and parameter.grad.norm() == 0
    addon.zero_grad(set_to_none=True)
    with torch.no_grad():
        addon.output.weight.copy_(torch.eye(288, 128) * .001)
    output = addon(points, boxes, valid)
    assert torch.count_nonzero(output[~valid]) == 0
    assert torch.count_nonzero(output[valid]) > 0
    (output * target).sum().backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in addon.parameters())


def test_object_features_depend_on_inside_rgb_and_ignore_outside_rgb():
    points, boxes, valid = example()
    outside = torch.tensor([[[3., 3., 3., .1, .2, .3]]]).repeat(2, 1, 1)
    points = torch.cat((points, outside), dim=1)
    addon = ObjectPointAppearanceResidual()
    with torch.no_grad():
        addon.output.weight.normal_(std=.01)
    original = addon(points, boxes, valid)
    changed = points.clone()
    changed[:, -1, 3:] += 10
    assert torch.equal(original, addon(changed, boxes, valid))
    changed[:, 0, 3:] += 1
    assert not torch.equal(original, addon(changed, boxes, valid))


def test_point_object_and_batch_permutations_and_translation_keep_correspondence():
    points, boxes, valid = example()
    addon = ObjectPointAppearanceResidual()
    with torch.no_grad():
        addon.output.weight.normal_(std=.01)
    original = addon(points, boxes, valid)
    point_order = torch.arange(11, -1, -1)
    assert torch.allclose(original, addon(points[:, point_order], boxes, valid), atol=1e-6)
    order = torch.tensor([2, 0, 1])
    assert torch.allclose(original[:, order], addon(points, boxes[:, order], valid[:, order]), atol=1e-6)
    assert torch.equal(original.flip(0), addon(points.flip(0), boxes.flip(0), valid.flip(0)))
    moved_points, moved_boxes = points.clone(), boxes.clone()
    shift = torch.tensor([2., -1., 3.])
    moved_points[:, :, :3] += shift
    moved_boxes[:, :, :3] += shift
    assert torch.allclose(original, addon(moved_points, moved_boxes, valid), atol=1e-6)


class Decoder(nn.Module):
    def forward(self, query, detected_feats, detected_mask):
        return query + detected_feats.masked_fill(detected_mask.unsqueeze(-1), 0).sum(dim=1)


class NativeObjectPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.ModuleList([Decoder(), Decoder()])

    def forward(self, inputs):
        query = inputs['query']
        first = self.decoder[0](query, detected_feats=inputs['memory'], detected_mask=~inputs['det_bbox_label_mask'])
        last = self.decoder[1](first, detected_feats=inputs['memory'], detected_mask=~inputs['det_bbox_label_mask'])
        return first, last


def test_attachment_changes_only_last_memory_and_restores_original_forward():
    points, boxes, valid = example()
    inputs = {'point_clouds': points, 'det_boxes': boxes, 'det_bbox_label_mask': valid,
              'memory': torch.randn(2, 3, 288), 'query': torch.randn(2, 288)}
    model, addon = NativeObjectPath(), ObjectPointAppearanceResidual()
    original = model.decoder[-1].forward
    baseline = model(inputs)
    old_memory = inputs['memory'].clone()
    attachment = LastDecoderObjectAppearanceIntervention(model, addon)
    assert not model.state_dict()
    assert all(torch.equal(a, b) for a, b in zip(baseline, model(inputs)))
    with torch.no_grad():
        addon.output.weight.copy_(torch.eye(288, 128) * .001)
    changed = model(inputs)
    assert torch.equal(changed[0], baseline[0]) and not torch.equal(changed[1], baseline[1])
    assert torch.equal(old_memory, inputs['memory']) and attachment.inputs is None
    attachment.remove()
    assert model.decoder[-1].forward == original
    assert all(torch.equal(a, b) for a, b in zip(baseline, model(inputs)))
