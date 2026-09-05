"""Measure native and nearest-two Mask neighborhoods on16 fixed fit rows."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    opt = parser.parse_args()
    addon = opt.manifest.parent
    manifest = json.loads(opt.manifest.read_text())
    source = Path(manifest['model_source'])

    def verify_inputs():
        source_manifest = source / 'g0_source_manifest.json'
        assert file_sha(source_manifest) == manifest['source_manifest_sha256']
        for name, digest in json.loads(source_manifest.read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(addon / name) == digest, name
        for name, metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'], name
        assert file_sha(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']

    verify_inputs()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(addon / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator
    from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
    from models.rec_evaluator_filter import build_detector_overlap_valid
    from scripts.nr3d_mask_branch_diagnostic import diagnose_root_candidates, superpoint_mask_ious
    from scripts.nr3d_mask_neighborhood_probe import NearestTwoGroup, describe_neighborhoods
    from pointnet2 import pointnet2_utils
    from scripts.run_nr3d_view_pair_role import read_train_rows

    raw_rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))
    row_ids = manifest['fit_row_ids']
    assert len(row_ids) == len({raw_rows[i]['scan_id'] for i in row_ids}) == 16
    chosen, scenes = [], set()
    for i, row in enumerate(raw_rows):
        fold = int(hashlib.sha256((manifest['split_salt'] + '\0' + row['scan_id']).encode()).hexdigest()[:8], 16) % 5
        if fold != 0 and row['scan_id'] not in scenes:
            chosen.append(i)
            scenes.add(row['scan_id'])
            if len(chosen) == 16:
                break
    assert chosen == row_ids
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu')
    assert checkpoint['evaluation_only'] and 'optimizer' not in checkpoint
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(checkpoint['config']))
    assert args.mask_loss_scale == args.consistency_loss_scale == 1
    assert not args.source_choice_selector_train_only and not args.use_source_moe
    model = TrainTester.get_model(args).cuda().eval()
    state = {name[7:]: value for name, value in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    assert model.decoder_query_adapter is None
    del checkpoint

    def verify_state():
        for name, value in model.state_dict().items():
            assert torch.equal(value.detach().cpu(), state[name]), name
        assert all(parameter.grad is None for parameter in model.parameters())

    verify_state()

    class FixedFit(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in row_ids]
            super()._scene_graph_parse(annos)

    dataset = FixedFit(dataset_dict={'nr3d': 1}, test_dataset='nr3d', split='train',
                       data_path='/root/autodl-tmp/DATA_ROOT/', use_color=args.use_color,
                       detect_intermediate=args.detect_intermediate, butd_cls=args.butd_cls,
                       skip_missing_superpoints=args.skip_missing_superpoints)
    dataset.augment = False
    assert len(dataset) == 16
    for anno, row_id in zip(dataset.annos, row_ids):
        assert anno['scan_id'] == raw_rows[row_id]['scan_id']
        assert anno['target_id'] == int(raw_rows[row_id]['target_id'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                                         generator=torch.Generator().manual_seed(0))
    criterion, set_criterion = TrainTester.get_criterion(args)
    evaluator = GroundingEvaluator(only_root=True, prefixes=['last_'], topks=[1],
                                   filter_non_gt_boxes=True, eval_use_selector_choice_scores=True)

    m3 = json.loads((addon / 'm3_receipt.json').read_text())
    m3_rows = {row['fit_row_id']: row for batch in m3['batches'] for row in batch['rows']}
    original_grouper = model.super_grouper
    assert original_grouper.radius == .2 and original_grouper.nsample == 2
    assert not original_grouper.use_xyz and not original_grouper.sample_uniformly
    nearest_grouper = NearestTwoGroup()
    xyz = torch.tensor([[[.15, 0., 0.], [.05, 0., 0.], [.08, 0., 0.]]], device='cuda')
    centers = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], device='cuda')
    features = torch.tensor([[[10., 20., 30.]]], device='cuda')
    with torch.no_grad():
        native_features, native_ids = original_grouper(xyz, centers, features)
        nearest_features, nearest_ids = nearest_grouper(xyz, centers, features)
    assert native_ids.tolist() == [[[0, 1], [0, 0]]]
    assert native_features.tolist() == [[[[10., 20.], [10., 10.]]]]
    assert nearest_ids.tolist() == [[[1, 2], [0, 2]]]
    assert nearest_features.tolist() == [[[[20., 30.], [10., 30.]]]]
    extension = Path(pointnet2_utils._ext.__file__)
    synthetic = {'native_indices': native_ids.tolist(), 'nearest_indices': nearest_ids.tolist(),
                 'extension_path': str(extension), 'extension_sha256': file_sha(extension), 'passed': True}
    print('M4 CUDA SYNTHETIC', json.dumps(synthetic), flush=True)
    captured = {}

    def group_hook(module, inputs, outputs):
        captured['groups'].append((inputs[0], inputs[1], outputs[1]))

    def feature_hook(name):
        def hook(module, inputs, outputs):
            captured[name] = outputs
        return hook

    def criterion_hook(module, inputs, outputs):
        if 'sp_pred_masks' in inputs[0]:
            assert 'indices' not in captured
            captured['indices'] = outputs[1]

    handles = [original_grouper.register_forward_hook(group_hook),
               nearest_grouper.register_forward_hook(group_hook),
               model.x_mask.register_forward_hook(feature_hook('projected_seeds')),
               model.decoder[-1].register_forward_hook(feature_hook('decoder_query')),
               set_criterion.register_forward_hook(criterion_hook)]

    def observe(inputs, batch, ids, arm, original_matches):
        captured.clear()
        captured['groups'] = []
        outputs = model(inputs)
        outputs.update(batch)
        total, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        assert torch.isfinite(total)
        rows = diagnose_root_candidates(outputs, evaluator)
        assert len(captured['groups']) == 4
        unchanged = {key: outputs[key].clone() for key in ['last_center', 'last_pred_size',
                       'selected_source_scores', 'seed_xyz', 'seed_inds']}
        unchanged.update({key: captured[key].clone() for key in ['projected_seeds', 'decoder_query']})
        groups = captured['groups'][:]
        matches = []
        for bid, (row, row_id) in enumerate(zip(rows, ids)):
            matched, target_indices = captured['indices'][bid]
            assert matched.numel() == target_indices.numel() == 1 and int(target_indices[0]) == 0
            index = int(matched[0])
            matches.append(index)
            fixed_index = index if original_matches is None else original_matches[bid]
            raw_ious = superpoint_mask_ious(outputs['sp_last_pred_masks'][bid],
                                           outputs['superpoints'][bid].long(), batch['gt_masks'][bid, 0].bool())
            input_hash = hashlib.sha256(inputs['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest()
            assert input_hash == m3_rows[row_id]['input_point_cloud_sha256']
            indexed_xyz = inputs['point_clouds'][bid, outputs['seed_inds'][bid].long(), :3]
            seed_delta = outputs['seed_xyz'][bid] - indexed_xyz
            row.update(fit_row_id=row_id, scan_id=raw_rows[row_id]['scan_id'], target_id=int(raw_rows[row_id]['target_id']),
                       arm=arm, input_point_cloud_sha256=input_hash, native_matched_query=index,
                       original_matched_query=fixed_index, matched_raw_mask_iou=float(raw_ious[index]),
                       original_matched_raw_mask_iou=float(raw_ious[fixed_index]),
                       all_raw_query_mask_ious=raw_ious.tolist(),
                       seed_index_coordinate_equal=bool(torch.equal(outputs['seed_xyz'][bid], indexed_xyz)),
                       seed_index_coordinate_max_abs_error=float(seed_delta.abs().max()),
                       m3_native_raw_mask_max_abs_delta=float((raw_ious - raw_ious.new_tensor(m3_rows[row_id]['all_raw_query_mask_ious'])).abs().max()))
        return {'rows': rows, 'native_loss': float(total), 'matches': matches}, unchanged, groups

    batches = []
    started = time.time()
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            batch = TrainTester._to_gpu(raw_batch)
            inputs = TrainTester._get_inputs(batch)
            inputs['train'] = False
            ids = row_ids[batch_index * 4:(batch_index + 1) * 4]
            model.super_grouper = original_grouper
            native, native_tensors, native_groups = observe(inputs, batch, ids, 'native_ball', None)
            model.super_grouper = nearest_grouper
            nearest, nearest_tensors, nearest_groups = observe(inputs, batch, ids, 'nearest_two', native['matches'])
            model.super_grouper = original_grouper
            for name in native_tensors:
                assert torch.equal(native_tensors[name], nearest_tensors[name]), name
            neighborhoods = []
            for bid in range(4):
                seed_xyz, sp_xyz, native_indices = native_groups[bid]
                nearest_seed_xyz, nearest_sp_xyz, nearest_indices = nearest_groups[bid]
                assert torch.equal(seed_xyz, nearest_seed_xyz) and torch.equal(sp_xyz, nearest_sp_xyz)
                for selection in ['rec_selection', 'mask_selection']:
                    assert native['rows'][bid][selection]['query'] == nearest['rows'][bid][selection]['query']
                locality = describe_neighborhoods(seed_xyz[0], sp_xyz[0], native_indices[0].long(),
                              nearest_indices[0].long(), batch['superpoint'][bid].long(), batch['gt_masks'][bid, 0].bool(),
                              native_tensors['seed_inds'][bid].long(), original_grouper.radius)
                locality['fit_row_id'] = ids[bid]
                neighborhoods.append(locality)
            result = {'fit_row_ids': ids, 'native': native, 'nearest': nearest,
                      'neighborhoods': neighborhoods, 'grounding_and_shared_features_exactly_equal': True}
            batches.append(result)
            print('M4 BATCH', json.dumps({'batch': batch_index + 1, 'fit_row_ids': ids,
                  'native_loss': native['native_loss'], 'nearest_loss': nearest['native_loss'],
                  'majority_counts': [item['counts']['majority_positive'] for item in neighborhoods],
                  'elapsed_seconds': time.time() - started}), flush=True)
            del native_tensors, nearest_tensors, native_groups, nearest_groups
    for handle in handles:
        handle.remove()
    captured.clear()
    verify_state()
    verify_inputs()
    receipt = {'schema': 'mcln-nr3d-mask-neighborhood-probe-v1', 'status': 'complete',
               'batches': batches, 'synthetic_cuda': synthetic, 'fit_row_ids': row_ids,
               'model_forwards': 8, 'optimizer_steps': 0, 'checkpoint_writes': 0,
               'formal_rows': 0, 'heldout_rows': 0, 'model_mode': 'eval_no_grad',
               'original_grouper_restored': model.super_grouper is original_grouper,
               'source_data_and_protected_state_unchanged': True,
               'native_thresholds_and_query_selection_unchanged': True,
               'input_points_equal_m3': True, 'formal_promotion': False,
               'manifest_sha256': file_sha(opt.manifest), 'elapsed_seconds': time.time() - started,
               'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
    with (addon / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('M4 COMPLETE', json.dumps({key: value for key, value in receipt.items() if key != 'batches'}), flush=True)


if __name__ == '__main__':
    main()
