"""Exercise the native loss entrypoint with real Hungarian and Mask losses."""
import copy
import sys

import pytest
import torch

from main_utils import BaseTrainTester, parse_option
from models.losses import HungarianMatcher, SetCriterion, compute_hungarian_loss
from scripts import native_mask_geometry_supervision as geometry_module


def fixture(dataset, samples):
    count = len(samples)
    xyz = torch.stack([torch.arange(30.) / 10.] * 3, -1)
    centers = torch.tensor([[.5, .5, .5], [2., 2., 2.]])
    result = {
        'point_clouds': xyz[None].repeat(count, 1, 1),
        'center_label': centers[None].repeat(count, 1, 1),
        'size_gts': torch.full((count, 2, 3), .6),
        'sem_cls_label': torch.zeros(count, 2, dtype=torch.long),
        'box_label_mask': torch.ones(count, 2),
        'auxi_entity_positive_map': torch.zeros(count, 1, 256),
        'auxi_box': torch.zeros(count, 1, 6),
        'language_dataset': [dataset] * count,
        'sample_dataset': samples,
        'superpoints': torch.arange(30)[None].repeat(count, 1),
        'super_xyz_list': [xyz[None]] * count,
        'adaptive_weights': [torch.tensor(.1)] * count,
        'gt_masks': torch.zeros(count, 2, 30),
    }
    result['gt_masks'][:, 0, 3:9] = 1.
    result['gt_masks'][:, 1, 18:24] = 1.
    for key in ['positive_map', 'modify_positive_map', 'pron_positive_map', 'other_entity_map', 'rel_positive_map']:
        result[key] = torch.zeros(count, 2, 256)
    result['positive_map'][:, :, 0] = 1.
    for prefix in ['proposal_', 'last_']:
        # The root is proposal Query 0 but final Query 1.
        ordered = centers if prefix == 'proposal_' else centers.flip(0)
        result[prefix + 'center'] = ordered[None].repeat(count, 1, 1).requires_grad_()
        result[prefix + 'pred_size'] = torch.full((count, 2, 3), .6, requires_grad=True)
        result[prefix + 'sem_cls_scores'] = torch.zeros(count, 2, 256, requires_grad=True)
    result['last_pred_masks'] = [torch.zeros(1, 2, 30, requires_grad=True) for _ in samples]
    masks = torch.full((2, 30), -4.)
    masks[0, 18:24] = 4.
    masks[1, 3:9] = 4.
    result['sp_last_pred_masks'] = [masks.clone().requires_grad_() for _ in samples]
    return result


def criterion():
    return SetCriterion(HungarianMatcher(soft_token=True), ['boxes', 'labels', 'masks'])


@pytest.mark.parametrize('dataset', ['nr3d', 'sr3d'])
def test_native_entrypoint_uses_final_matches_and_excludes_detection_gradients(monkeypatch, dataset):
    monkeypatch.setattr(sys, 'argv', ['train_dist_mod.py', '--dataset', dataset,
                                    '--test_dataset', dataset, '--native_mask_geometry_supervision'])
    args = parse_option()
    assert args.native_mask_geometry_supervision
    args.num_decoder_layers = 1
    current = fixture(dataset, ['scannet', dataset, dataset])
    original = copy.deepcopy(current)
    control_args = copy.copy(args)
    control_args.native_mask_geometry_supervision = False
    before, _ = BaseTrainTester._compute_loss(original, compute_hungarian_loss, criterion(), control_args)
    after, outputs = BaseTrainTester._compute_loss(current, compute_hungarian_loss, criterion(), args)
    assert torch.isfinite(before) and torch.isfinite(after)
    assert outputs['mask_geometry_query_indices'].tolist() == [1, 1]
    assert outputs['mask_geometry_referring_rows'].item() == 2
    torch.testing.assert_allclose(after - before, outputs['mask_geometry_loss'], atol=1e-5, rtol=1e-5)
    before.backward()
    after.backward()
    for index in range(3):
        difference = current['sp_last_pred_masks'][index].grad - original['sp_last_pred_masks'][index].grad
        if index == 0:
            assert torch.count_nonzero(difference) == 0
        else:
            assert torch.count_nonzero(difference[0]) == 0
            assert torch.count_nonzero(difference[1]) > 0 and torch.isfinite(difference).all()


def test_detection_only_batch_keeps_native_loss_and_skips_geometry(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Detection rows have no referring root geometry')
    monkeypatch.setattr(geometry_module, 'native_mask_geometry_loss', forbidden)
    before = fixture('nr3d', ['scannet', 'scannet'])
    after = copy.deepcopy(before)
    native, _ = compute_hungarian_loss(before, 1, criterion())
    result, outputs = compute_hungarian_loss(after, 1, criterion(), native_mask_geometry_supervision=True)
    assert torch.equal(native, result)
    assert outputs['mask_geometry_loss'].item() == outputs['mask_geometry_referring_rows'].item() == 0
    native.backward(); result.backward()
    for left, right in zip(before['sp_last_pred_masks'], after['sp_last_pred_masks']):
        assert torch.equal(left.grad, right.grad)


def test_default_off_does_not_call_geometry_or_add_output_fields(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train_dist_mod.py'])
    assert not parse_option().native_mask_geometry_supervision
    def forbidden(*args, **kwargs):
        raise AssertionError('Default native loss must not call the geometry auxiliary')
    monkeypatch.setattr(geometry_module, 'native_mask_geometry_loss', forbidden)
    before = fixture('scanrefer', ['scanrefer'])
    after = copy.deepcopy(before)
    first, _ = compute_hungarian_loss(before, 1, criterion())
    second, outputs = compute_hungarian_loss(after, 1, criterion(), native_mask_geometry_supervision=False)
    assert torch.equal(first, second)
    assert not any(name.startswith('mask_geometry_') for name in outputs)


def test_explicit_batch_selection_matches_single_row_objective():
    mixed = fixture('sr3d', ['scannet', 'sr3d'])
    single = fixture('sr3d', ['sr3d'])
    matches = [(torch.tensor([0, 1]), torch.tensor([1, 0]))] * 2
    roots = torch.cat([mixed['center_label'][:, 0], mixed['size_gts'][:, 0]], -1)
    selected, _ = geometry_module.native_mask_geometry_loss(mixed, mixed, roots, matches, batch_indices=[1])
    expected, _ = geometry_module.native_mask_geometry_loss(single, single, roots[1:], matches[1:])
    assert torch.equal(selected, expected)


def test_head_only_early_return_cannot_silently_drop_requested_geometry():
    with pytest.raises(ValueError, match='requires full training mode'):
        compute_hungarian_loss(fixture('nr3d', ['nr3d']), 1, criterion(),
                               native_mask_geometry_supervision=True, query_mask_fusion_train_only=True)
