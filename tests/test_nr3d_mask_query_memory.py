import torch
from torch import nn

from scripts.nr3d_mask_query_memory import MaskQueryMemoryIntervention, MaskQueryMemoryReadout


def example():
    torch.manual_seed(17)
    query = torch.randn(3, 288)
    memory = torch.randn(5, 288)
    xyz = torch.randn(5, 3)
    boxes = torch.cat((torch.randn(3, 3), torch.full((3, 3), 4.)), dim=1)
    return query, memory, xyz, boxes


def test_zero_identity_and_nonzero_gradient_path_after_output_perturbation():
    query, memory, xyz, boxes = example()
    module = MaskQueryMemoryReadout()
    target = torch.randn_like(query)
    output = module(query, memory, xyz, boxes)
    assert torch.equal(output, query)
    (output * target).sum().backward()
    assert module.output.weight.grad.norm() > 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if name != 'output.weight':
            assert parameter.grad.norm() == 0
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.output.weight.copy_(torch.eye(288, 64) * .001)
    (module(query, memory, xyz, boxes) * target).sum().backward()
    assert all(parameter.grad.norm() > 0 and torch.isfinite(parameter.grad).all()
               for parameter in module.parameters())


def test_each_candidate_reads_memory_at_its_own_box():
    module = MaskQueryMemoryReadout()
    with torch.no_grad():
        module.query_projection.weight.zero_()
        module.key_projection.weight.zero_()
        module.value_projection.weight.zero_()
        module.value_projection.weight[0, 0] = 1
        module.output.weight[0, 0] = 1
    query = torch.zeros(2, 288)
    memory = torch.zeros(2, 288)
    memory[:, 0] = torch.tensor([1., -1.])
    xyz = torch.tensor([[-2., 0., 0.], [2., 0., 0.]])
    boxes = torch.cat((xyz, torch.ones(2, 3)), dim=1)
    output = module(query, memory, xyz, boxes)
    assert output[0, 0] > 0 and output[1, 0] < 0
    assert torch.equal(output[:, 1:], torch.zeros(2, 287))


def test_coordinate_translation_and_both_index_permutations_preserve_readout():
    query, memory, xyz, boxes = example()
    module = MaskQueryMemoryReadout()
    with torch.no_grad():
        module.output.weight.normal_(std=.01)
    reference = module(query, memory, xyz, boxes)
    shift = torch.tensor([4., -3., 2.])
    shifted_boxes = boxes.clone()
    shifted_boxes[:, :3] += shift
    assert torch.allclose(reference, module(query, memory, xyz + shift, shifted_boxes), atol=1e-6)
    order = torch.tensor([2, 4, 0, 3, 1])
    assert torch.allclose(reference, module(query, memory[order], xyz[order], boxes), atol=1e-6)
    candidates = torch.tensor([2, 0, 1])
    assert torch.allclose(reference[candidates], module(query[candidates], memory, xyz, boxes[candidates]), atol=1e-6)


class Grouper(nn.Module):
    def forward(self, point_xyz, superpoint_xyz, features):
        return features


class NativeMaskPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.super_grouper = Grouper()

    def _seg_seeds_prediction(self, query, mask_feats, end_points, prefix=''):
        result = torch.einsum('bnd,bdm->bnm', query, mask_feats)
        end_points[prefix + 'pred_mask_seeds'] = result
        return result

    def forward(self, inputs):
        boxes = inputs['boxes']
        outputs = {'last_center': boxes[:, :, :3], 'last_pred_size': boxes[:, :, 3:],
                   'identity_scores': inputs['identity_scores'], 'text_mask': inputs['text_mask'],
                   'alpha': inputs['alpha']}
        features = [self.super_grouper(None, xyz.unsqueeze(0), memory.T.unsqueeze(0))
                    for xyz, memory in zip(inputs['xyz'], inputs['memory'])]
        outputs['query_masks'] = [self._seg_seeds_prediction(query.unsqueeze(0), feature, outputs, 'last_')
                                  for query, feature in zip(inputs['query'], features)]
        return outputs


def test_native_writer_batch_indexing_and_removal_keep_other_paths_exact():
    query, memory, xyz, boxes = example()
    inputs = {'point_clouds': torch.zeros(2, 9, 6), 'boxes': boxes.repeat(2, 1, 1),
              'query': query.repeat(2, 1, 1), 'memory': [memory, memory[:4]], 'xyz': [xyz, xyz[:4]],
              'identity_scores': torch.randn(2, 3), 'text_mask': torch.randn(2, 3), 'alpha': torch.rand(2)}
    model = NativeMaskPath()
    baseline = model(inputs)
    original = model._seg_seeds_prediction
    addon = MaskQueryMemoryReadout()
    attachment = MaskQueryMemoryIntervention(model, addon)
    assert not model.state_dict()
    zero = model(inputs)
    assert all(torch.equal(a, b) for a, b in zip(baseline['query_masks'], zero['query_masks']))
    with torch.no_grad():
        addon.output.weight.copy_(torch.eye(288, 64) * .001)
    changed = model(inputs)
    assert all(not torch.equal(a, b) for a, b in zip(baseline['query_masks'], changed['query_masks']))
    assert torch.equal(changed['last_pred_mask_seeds'], changed['query_masks'][-1])
    for key in ['last_center', 'last_pred_size', 'identity_scores', 'text_mask', 'alpha']:
        assert torch.equal(changed[key], baseline[key])
    assert attachment.superpoint_xyz == []
    attachment.remove()
    assert model._seg_seeds_prediction == original
    restored = model(inputs)
    assert all(torch.equal(a, b) for a, b in zip(baseline['query_masks'], restored['query_masks']))
