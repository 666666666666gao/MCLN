"""Native CUDA contract/gradient probe on the existing sixteen M3 fit rows.

Run in isolation after the current C1 task and full crop-contract audit. No training or quality screen.
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
        assert file_sha(Path(manifest['crop_audit_receipt'])) == manifest['crop_audit_receipt_sha256']

    verify_inputs()
    crop_audit = json.loads(Path(manifest['crop_audit_receipt']).read_text())
    assert crop_audit['status'] == 'complete' and crop_audit['input_contract_pass']
    assert crop_audit['torch_module_predicate_matches_numpy_explicit_bounds']
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from scripts.nr3d_object_point_appearance import (
        ObjectPointAppearanceResidual, LastDecoderObjectAppearanceIntervention)
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
    addon = ObjectPointAppearanceResidual().cuda()
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

    def query_hook(index):
        def capture(module, inputs, output):
            captured['query_' + str(index)] = output.detach().clone()
        return capture

    handles = [layer.register_forward_hook(query_hook(index)) for index, layer in enumerate(model.decoder)]
    rec_keys = ['seed_inds', 'query_points_sample_inds', 'last_center', 'last_pred_size',
                'selected_source_scores', 'last_sem_cls_scores', 'last_proj_queries']

    def snapshot(outputs):
        result = {name: outputs[name].detach().clone() for name in rec_keys}
        result.update(captured)
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

    def grounding_objective(outputs):
        assert all(name == 'nr3d' for name in outputs['language_dataset'])
        return (outputs['loss_ce'] + 5 * outputs['loss_bbox'] + outputs['loss_giou']
                + outputs['loss_sem_align']) / (args.num_decoder_layers + 1)
    results = []
    gradient_positive_ever = {name: False for name, _ in addon.named_parameters()}
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
        with torch.no_grad():
            native_outputs, native_seconds = timed_forward(inputs)
            original = snapshot(native_outputs)
            del native_outputs
        attachment = LastDecoderObjectAppearanceIntervention(model, addon)
        outputs, zero_seconds = timed_forward(inputs)
        zero = snapshot(outputs)
        assert all(torch.equal(original[name], zero[name]) for name in original)
        outputs.update(batch)
        total, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        grounding_loss = grounding_objective(outputs)
        zero_gradients = torch.autograd.grad(grounding_loss, tuple(addon.parameters()), retain_graph=True)
        zero_total_gradients = torch.autograd.grad(total, tuple(addon.parameters()))
        assert torch.isfinite(total) and all(torch.isfinite(g).all() for g in zero_gradients)
        assert zero_gradients[-1].norm() > 0
        assert all(g.count_nonzero() == 0 for g in zero_gradients[:-1])
        assert all(torch.isfinite(g).all() for g in zero_total_gradients)
        assert zero_total_gradients[-1].norm() > 0
        zero_total_norm = float(zero_total_gradients[-1].norm())
        zero_norm = float(zero_gradients[-1].norm())
        del outputs, total, grounding_loss, zero_gradients, zero_total_gradients, zero
        with torch.no_grad():
            addon.output.weight.copy_(torch.eye(288, 128, device='cuda') * .001)
        outputs, perturbed_seconds = timed_forward(inputs)
        perturbed = snapshot(outputs)
        assert all(torch.isfinite(value).all() for value in perturbed.values())
        untouched = ['seed_inds', 'query_points_sample_inds'] + ['query_' + str(i) for i in range(5)] + [name for name in original
                     if name.startswith(('last_pred_masks_', 'adaptive_weights_'))]
        assert all(torch.equal(original[name], perturbed[name]) for name in untouched)
        mask_delta = max(float((perturbed[name] - original[name]).abs().max())
                         for name in original if name.startswith('sp_last_pred_masks_'))
        assert mask_delta > 0
        identity_deltas = {name: float((perturbed[name] - original[name]).abs().max())
                           for name in ['query_5', 'last_sem_cls_scores', 'last_proj_queries']}
        assert all(value > 0 for value in identity_deltas.values())
        grounding_deltas = {name: float((perturbed[name] - original[name]).abs().max())
                            for name in ['last_center', 'last_pred_size', 'selected_source_scores']}
        outputs.update(batch)
        total, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        grounding_loss = grounding_objective(outputs)
        gradients = torch.autograd.grad(grounding_loss, tuple(addon.parameters()), retain_graph=True)
        total_gradients = torch.autograd.grad(total, tuple(addon.parameters()))
        assert torch.isfinite(total)
        assert all(torch.isfinite(g).all() for g in gradients + total_gradients)
        for (name, _), gradient in zip(addon.named_parameters(), gradients):
            gradient_positive_ever[name] |= bool(gradient.norm() > 0)
        results.append({'fit_row_ids': ids, 'zero_output_grounding_gradient_norm': zero_norm,
                        'zero_output_total_gradient_norm': zero_total_norm,
                        'fixed_perturbation_identity_deltas': identity_deltas,
                        'fixed_perturbation_grounding_deltas': grounding_deltas,
                        'perturbed_total_gradient_norms': [float(g.norm()) for g in total_gradients],
                        'perturbed_grounding_gradient_norms': [float(g.norm()) for g in gradients],
                        'fixed_perturbation_raw_query_mask_max_abs_delta': mask_delta,
                        'native_no_grad_forward_seconds': native_seconds,
                        'zero_addon_autograd_forward_seconds': zero_seconds,
                        'perturbed_addon_autograd_forward_seconds': perturbed_seconds})
        attachment.remove()
        with torch.no_grad():
            addon.output.weight.zero_()
        del outputs, total, grounding_loss, gradients, total_gradients, original, perturbed
        print('OBJECT APPEARANCE PREFLIGHT', json.dumps(results[-1]), flush=True)

    for handle in handles:
        handle.remove()
    assert all(gradient_positive_ever.values())
    assert set(model.state_dict()) == set(state)
    assert all(torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value.detach().cpu(), initial_addon[name]) for name, value in addon.state_dict().items())
    verify_inputs()
    receipt = {'schema': 'mcln-object-appearance-native-preflight-v1', 'status': 'complete',
               'manifest_sha256': file_sha(options.manifest), 'fit_rows': 16, 'fit_scenes': 16,
               'native_forwards': 12, 'optimizer_steps': 0, 'checkpoint_writes': 0,
               'formal_rows': 0, 'holdout_rows': 0, 'parameters': 41472,
               'original_text_mask_and_alpha_unchanged': True, 'zero_start_identity': True,
               'finite_nonzero_parameter_gradient_observed': gradient_positive_ever,
               'early_queries_and_sampling_unchanged': True, 'native_grounding_loss_connected': True,
               'last_query_and_both_semantic_paths_changed': True, 'native_total_loss_connected': True,
               'frozen_state_inputs_and_addon_restored': True, 'batches': results,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'elapsed_seconds_excluding_dataset_init': time.time() - started,
               'quality_screen_executed': False}
    with (directory / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('OBJECT APPEARANCE PREFLIGHT COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
