"""Native sparse-memory contract and gradient probe on sixteen fixed M3 fit rows.

No training, held-out quality evaluation or formal validation.
"""

import argparse
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
    options = parser.parse_args()
    directory = options.manifest.parent
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])

    def verify_inputs():
        source_manifest = source / 'g0_source_manifest.json'
        assert file_sha(source_manifest) == manifest['source_manifest_sha256']
        for name, digest in json.loads(source_manifest.read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(directory / name) == digest, name
        for name, metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'], name
        assert file_sha(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']
        assert file_sha(Path(manifest['m3_receipt'])) == manifest['m3_receipt_sha256']
        for name, digest in manifest['runtime_receipts'].items():
            assert file_sha(Path(name)) == digest, name

    verify_inputs()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import spconv
    import cumm
    assert sys.prefix == manifest['python_prefix']
    assert torch.__version__ == '1.10.2+cu111' and spconv.__version__ == '2.3.6'
    assert cumm.__version__ == '0.4.11'
    import scripts
    scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from scripts.nr3d_sparse_point_memory import (
        SparsePointSuperpointResidual, SparseSuperpointIntervention)
    from scripts.nr3d_point_voxel_mapping import point_voxel_mapping
    from scripts.run_nr3d_view_pair_role import read_train_rows

    raw_rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))
    row_ids = manifest['fit_row_ids']
    chosen, scenes = [], set()
    for index, row in enumerate(raw_rows):
        fold = int(hashlib.sha256((manifest['split_salt'] + '\0' + row['scan_id']).encode()).hexdigest()[:8], 16) % 5
        if fold != 0 and row['scan_id'] not in scenes:
            chosen.append(index)
            scenes.add(row['scan_id'])
            if len(chosen) == 16:
                break
    assert row_ids == chosen
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
    assert args.use_color and not args.use_height and not args.use_multiview
    assert not args.source_choice_selector_train_only and not args.use_source_moe
    assert args.mask_loss_scale == args.consistency_loss_scale == 1
    state = {name[7:]: value for name, value in checkpoint['model'].items()}
    model = TrainTester.get_model(args).cuda().eval()
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    del checkpoint
    addon = SparsePointSuperpointResidual().cuda()
    initial_addon = {name: value.detach().cpu().clone() for name, value in addon.state_dict().items()}
    criterion, set_criterion = TrainTester.get_criterion(args)

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
    for row_id, anno in zip(row_ids, dataset.annos):
        assert anno['scan_id'] == raw_rows[row_id]['scan_id']
        assert anno['target_id'] == int(raw_rows[row_id]['target_id'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                                         generator=torch.Generator().manual_seed(0))
    m3 = json.loads(Path(manifest['m3_receipt']).read_text())
    expected_points = {row['fit_row_id']: row['input_point_cloud_sha256']
                       for batch in m3['batches'] for row in batch['rows']}
    captured = {}

    def capture_query(module, inputs, output):
        captured['query'] = output.detach().clone()

    handle = model.decoder[-1].register_forward_hook(capture_query)
    rec_keys = ['seed_inds', 'query_points_sample_inds', 'last_center', 'last_pred_size',
                'selected_source_scores']

    def snapshot(outputs):
        result = {name: outputs[name].detach().clone() for name in rec_keys}
        result['query'] = captured['query']
        for name in ('sp_last_pred_masks', 'last_pred_masks', 'adaptive_weights'):
            for index, value in enumerate(outputs[name]):
                result[name + '_' + str(index)] = value.detach().clone()
        return result

    def timed_forward(inputs):
        torch.cuda.synchronize()
        begin = time.time()
        outputs = model(inputs)
        torch.cuda.synchronize()
        return outputs, time.time() - begin

    mask_coefficients = {'loss_mask': 10, 'loss_dice': 2, 'sp_loss_mask': 5, 'sp_loss_dice': 1,
                         'adaptive_weight_loss_mask': 10, 'adaptive_weight_loss_dice': 2,
                         'corresponding_loss_mask': 10, 'corresponding_loss_dice': 2}
    results = []
    input_records = []
    parameter_count = sum(parameter.numel() for parameter in addon.parameters())
    parameter_shapes = {name: list(value.shape) for name, value in addon.named_parameters()}
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for index, raw in enumerate(loader):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        ids = row_ids[index * 4:(index + 1) * 4]
        assert inputs['point_clouds'].shape == (4, 50000, 6)
        for scene, row_id in enumerate(ids):
            digest = hashlib.sha256(inputs['point_clouds'][scene].cpu().numpy().tobytes()).hexdigest()
            assert digest == expected_points[row_id], row_id
            cloud = inputs['point_clouds'][scene]
            superpoints = inputs['superpoint'][scene]
            cells, inverse, local, origin = point_voxel_mapping(cloud)
            counts = torch.bincount(inverse, minlength=len(cells))
            restored = origin + (cells[inverse].float() + local[:, :3] + .5) * .02
            error = float((restored - cloud[:, :3]).abs().max())
            assert len(inverse) == 50000 and int(counts.sum()) == 50000
            assert error <= 1e-5 and torch.equal(local[:, 3:], cloud[:, 3:])
            pairs = torch.unique(torch.stack((inverse, superpoints), dim=1), dim=0)
            sp_per_cell = torch.bincount(pairs[:, 0], minlength=len(cells))
            input_records.append({'fit_row_id': row_id, 'scan_id': raw_rows[row_id]['scan_id'],
                'point_cloud_sha256': digest, 'points': 50000, 'occupied_voxels': len(cells),
                'grid_shape_xyz': (cells.max(dim=0).values + 1).tolist(),
                'superpoints': int(superpoints.max()) + 1, 'max_points_in_voxel': int(counts.max()),
                'voxels_spanning_multiple_superpoints': int((sp_per_cell > 1).sum()),
                'points_in_multi_superpoint_voxels': int((sp_per_cell[inverse] > 1).sum()),
                'max_xyz_reconstruction_error_m': error, 'actual_point_rgb_preserved': True})
            del cells, inverse, local, origin, counts, restored, pairs, sp_per_cell
        with torch.no_grad():
            native_outputs, native_seconds = timed_forward(inputs)
            original = snapshot(native_outputs)
            del native_outputs
        attachment = SparseSuperpointIntervention(model, addon)
        outputs, zero_seconds = timed_forward(inputs)
        zero = snapshot(outputs)
        assert all(torch.equal(original[name], zero[name]) for name in original)
        outputs.update(batch)
        total, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        mask_loss = sum(outputs[name] * scale for name, scale in mask_coefficients.items())
        zero_gradients = torch.autograd.grad(mask_loss, tuple(addon.parameters()))
        assert torch.isfinite(total) and all(torch.isfinite(g).all() for g in zero_gradients)
        assert zero_gradients[-1].norm() > 0
        assert all(g.count_nonzero() == 0 for g in zero_gradients[:-1])
        zero_norm = float(zero_gradients[-1].norm())
        del outputs, total, mask_loss, zero_gradients, zero
        with torch.no_grad():
            addon.output.weight.copy_(torch.eye(288, 64, device='cuda') * .001)
        outputs, perturbed_seconds = timed_forward(inputs)
        perturbed = snapshot(outputs)
        assert all(torch.isfinite(value).all() for value in perturbed.values())
        assert all(torch.equal(original[name], perturbed[name]) for name in rec_keys + ['query'])
        mask_delta = max(float((perturbed[name] - original[name]).abs().max())
                         for name in original if name.startswith('sp_last_pred_masks_'))
        assert mask_delta > 0
        outputs.update(batch)
        total, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        mask_loss = sum(outputs[name] * scale for name, scale in mask_coefficients.items())
        gradients = torch.autograd.grad(mask_loss, tuple(addon.parameters()))
        assert torch.isfinite(total)
        assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
        results.append({'fit_row_ids': ids, 'zero_output_gradient_norm': zero_norm,
                        'perturbed_mask_gradient_norms': [float(g.norm()) for g in gradients],
                        'fixed_perturbation_raw_query_mask_max_abs_delta': mask_delta,
                        'native_no_grad_forward_seconds': native_seconds,
                        'zero_addon_autograd_forward_seconds': zero_seconds,
                        'perturbed_addon_autograd_forward_seconds': perturbed_seconds})
        attachment.remove()
        with torch.no_grad():
            addon.output.weight.zero_()
        del outputs, total, mask_loss, gradients, original, perturbed
        print('SPARSE POINT PREFLIGHT', json.dumps(results[-1]), flush=True)

    handle.remove()
    assert set(model.state_dict()) == set(state)
    assert all(torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value.detach().cpu(), initial_addon[name]) for name, value in addon.state_dict().items())
    verify_inputs()
    receipt = {'schema': 'mcln-sparse-point-preflight-v1', 'status': 'complete',
               'manifest_sha256': file_sha(options.manifest), 'fit_rows': 16, 'fit_scenes': 16,
               'native_forwards': 12, 'optimizer_steps': 0, 'checkpoint_writes': 0,
               'formal_rows': 0, 'holdout_rows': 0, 'parameters': parameter_count, 'parameter_shapes': parameter_shapes,
               'voxel_size_m': .02, 'point_mapping_records': input_records,
               'pointnet_sa1_features_used_by_new_branch': False, 'zero_start_identity': True,
               'fixed_perturbation_rec_unchanged': True, 'native_mask_loss_connected': True,
               'frozen_state_inputs_and_addon_restored': True, 'batches': results,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'elapsed_seconds_excluding_dataset_init': time.time() - started,
               'quality_screen_executed': False}
    receipt['runtime'] = {'python': sys.version, 'prefix': sys.prefix,
                          'torch': torch.__version__, 'spconv': spconv.__version__,
                          'cumm': cumm.__version__, 'gpu': torch.cuda.get_device_name(0),
                          'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                          'cudnn_tf32': torch.backends.cudnn.allow_tf32}
    with (directory / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('SPARSE POINT PREFLIGHT COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
