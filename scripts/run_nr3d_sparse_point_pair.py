"""Matched full-fit Mask learning with native versus raw sparse-point memory."""

import argparse
import copy
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


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    addon = options.manifest.parent
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])
    assert manifest['epochs'] == 1 and manifest['steps_per_arm'] == 6687
    assert manifest['train_batch_size'] == 4 and manifest['eval_batch_size'] == 16
    assert manifest['native_learning_rate'] == 1e-5 and manifest['sparse_learning_rate'] == 1e-4
    assert manifest['no_augmentation']

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
        assert file_sha(Path(manifest['sparse_preflight_receipt'])) == manifest['sparse_preflight_receipt_sha256']
        assert file_sha(Path(manifest['baseline_reference'])) == manifest['baseline_reference_sha256']
        assert file_sha(Path(manifest['warmup_regression_receipt'])) == manifest['warmup_regression_receipt_sha256']
        for name, digest in manifest['runtime_receipts'].items():
            assert file_sha(Path(name)) == digest, name

    verify_inputs()
    sparse_preflight = json.loads(Path(manifest['sparse_preflight_receipt']).read_text())
    assert sparse_preflight['status'] == 'complete' and sparse_preflight['optimizer_steps'] == 0
    assert sparse_preflight['zero_start_identity'] and sparse_preflight['native_mask_loss_connected']
    assert sparse_preflight['fixed_perturbation_rec_unchanged']
    warmup_regression = json.loads(Path(manifest['warmup_regression_receipt']).read_text())
    assert warmup_regression['status'] == 'complete' and warmup_regression['optimizer_steps'] == 0
    assert manifest['native_warmup_forwards'] == 1
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
    scripts.__path__ = [str(addon / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator
    from scripts.nr3d_candidate_contract import diagnose_root_candidates
    from scripts.nr3d_sparse_point_memory import SparsePointSuperpointResidual, SparseSuperpointIntervention
    from scripts.run_nr3d_view_pair_role import read_train_rows

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    raw_rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))
    partitions = {'fit': [], 'holdout': []}
    for i, row in enumerate(raw_rows):
        fold = int(hashlib.sha256((manifest['split_salt'] + '\0' + row['scan_id']).encode()).hexdigest()[:8], 16) % 5
        partitions['holdout' if fold == 0 else 'fit'].append(i)
    assert len(partitions['fit']) == 26747 and len(partitions['holdout']) == 6172
    assert partitions == manifest['row_ids']
    scenes = {name: {raw_rows[i]['scan_id'] for i in ids} for name, ids in partitions.items()}
    assert not scenes['fit'].intersection(scenes['holdout'])
    assert len(scenes['fit']) == 413 and len(scenes['holdout']) == 98
    selected_ids = sorted(partitions['fit'] + partitions['holdout'])

    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu')
    assert checkpoint['evaluation_only'] and 'optimizer' not in checkpoint
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(checkpoint['config']))
    assert not args.source_choice_selector_train_only and not args.use_source_moe
    assert args.mask_loss_scale == args.consistency_loss_scale == 1
    assert args.clip_norm == .1 and args.weight_decay == .0005
    assert args.use_color and not args.use_height and not args.use_multiview
    initial_state = {name[7:]: value for name, value in checkpoint['model'].items()}
    del checkpoint
    models, optimizers, trainable = {}, {}, {}
    prefixes = ('x_query.', 'x_mask.', 'rel_encoder.')
    for arm in ['native', 'sparse']:
        model = TrainTester.get_model(args).cuda().eval()
        model.load_state_dict(initial_state, strict=True)
        assert model.decoder_query_adapter is None
        assert model.super_grouper.radius == .2 and model.super_grouper.nsample == 2
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        assert len(parameters) == 16
        if arm == 'native':
            # Same addon initialization order as the completed 16-row preflight.
            sparse_addon = SparsePointSuperpointResidual().cuda()
            initial_sparse_state = {name: value.detach().cpu().clone()
                                    for name, value in sparse_addon.state_dict().items()}
        else:
            sparse_attachment = SparseSuperpointIntervention(model, sparse_addon)
            parameters.update({'sparse_point.' + name: value
                               for name, value in sparse_addon.named_parameters()})
        models[arm] = model
        trainable[arm] = parameters
        groups = [{'params': [p for name, p in parameters.items() if not name.startswith('sparse_point.')],
                   'lr': manifest['native_learning_rate']}]
        if arm == 'sparse':
            groups.append({'params': list(sparse_addon.parameters()), 'lr': manifest['sparse_learning_rate']})
        optimizers[arm] = torch.optim.AdamW(groups, lr=1e-5, weight_decay=args.weight_decay)
    assert len(trainable['sparse']) == 33
    assert list(trainable['native']) == list(trainable['sparse'])[:16]
    assert sum(p.numel() for p in trainable['native'].values()) == 1348960
    assert sum(p.numel() for p in trainable['sparse'].values()) == 1616896
    criterion, set_criterion = TrainTester.get_criterion(args)
    evaluator = GroundingEvaluator(only_root=True, prefixes=['last_'], topks=[1],
                                   filter_non_gt_boxes=True, eval_use_selector_choice_scores=True)

    def verify_state(before_training=False):
        changed = {}
        for arm, model in models.items():
            changed[arm] = []
            for name, value in model.state_dict().items():
                same = torch.equal(value.detach().cpu(), initial_state[name])
                if before_training or name not in trainable[arm]:
                    assert same, (arm, name)
                elif not same:
                    changed[arm].append(name)
            assert all(parameter.grad is None for name, parameter in model.named_parameters()
                       if name not in trainable[arm])
            if arm == 'sparse':
                for name, value in sparse_addon.state_dict().items():
                    same = torch.equal(value.detach().cpu(), initial_sparse_state[name])
                    if before_training:
                        assert same, name
                    elif not same:
                        changed[arm].append('sparse_point.' + name)
        return changed

    class SelectedDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in selected_ids]
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            item = super().__getitem__(index)
            item['sparse_row_id'] = np.int64(self.sparse_ids[index])
            return item

    base = SelectedDataset(dataset_dict={'nr3d': 1}, test_dataset='nr3d', split='train',
                           data_path='/root/autodl-tmp/DATA_ROOT/', use_color=args.use_color,
                           detect_intermediate=args.detect_intermediate, butd_cls=args.butd_cls,
                           skip_missing_superpoints=args.skip_missing_superpoints)
    assert len(base.annos) == len(selected_ids)
    by_id = {}
    for row_id, anno in zip(selected_ids, base.annos):
        assert anno['scan_id'] == raw_rows[row_id]['scan_id'] and anno['target_id'] == int(raw_rows[row_id]['target_id'])
        by_id[row_id] = anno
    datasets = {}
    for name, ids in partitions.items():
        dataset = copy.copy(base)
        dataset.annos = [by_id[i] for i in ids]
        dataset.sparse_ids = ids
        dataset.augment = False
        datasets[name] = dataset

    def loader(name, seed, batch_size, shuffle):
        return torch.utils.data.DataLoader(datasets[name], batch_size=batch_size, shuffle=shuffle,
                 num_workers=0, generator=torch.Generator().manual_seed(seed))

    def inputs_for(raw):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        return inputs, batch

    def grounding_snapshot(outputs):
        return {key: outputs[key].detach().clone() for key in ['last_center', 'last_pred_size', 'selected_source_scores']}

    def require_same_grounding(first, second):
        for key in first:
            assert torch.equal(first[key], second[key]), key

    started = time.time()
    preflight_inputs, preflight_batch = inputs_for(next(iter(loader('fit', 0, 4, False))))
    # A recorded native first-backward transient requires one zero-update warmup.
    warmup_outputs = models['native'](preflight_inputs)
    warmup_outputs.update(preflight_batch)
    warmup_loss, warmup_outputs = TrainTester._compute_loss(warmup_outputs, criterion, set_criterion, args)
    assert torch.isfinite(warmup_loss)
    warmup_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable['native'].values())
    warmup_value = float(warmup_loss)
    optimizers['native'].zero_grad(set_to_none=True)
    write_json(addon / 'warmup.json', {'native_forwards': 1, 'backwards': 1,
        'optimizer_steps': 0, 'loss': warmup_value,
        'input_point_sha256': [hashlib.sha256(cloud.cpu().numpy().tobytes()).hexdigest()
                               for cloud in preflight_inputs['point_clouds']]})
    del warmup_outputs, warmup_loss
    preflight = {}
    shared_gradients = {}
    first_grounding = None
    for arm, model in models.items():
        outputs = model(preflight_inputs)
        outputs.update(preflight_batch)
        loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        assert torch.isfinite(loss)
        snapshot = grounding_snapshot(outputs)
        if first_grounding is None:
            first_grounding = snapshot
        else:
            require_same_grounding(first_grounding, snapshot)
        loss.backward()
        shared_gradients[arm] = {name: trainable[arm][name].grad.detach().clone()
                                 for name in trainable['native']}
        norms = {}
        for name, parameter in trainable[arm].items():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            norms[name] = float(parameter.grad.norm())
        for name, value in norms.items():
            if name.startswith('sparse_point.') and name != 'sparse_point.output.weight':
                assert value == 0, name  # Required at zero output projection.
            else:
                assert value > 0, name
        optimizers[arm].zero_grad(set_to_none=True)
        preflight[arm] = {'loss': float(loss), 'gradient_norms': norms}
    assert preflight['native']['loss'] == preflight['sparse']['loss']
    assert preflight['native']['loss'] == warmup_value
    assert all(preflight['native']['gradient_norms'][name] == preflight['sparse']['gradient_norms'][name]
               for name in trainable['native'])
    shared_differences = {}
    for name, original in shared_gradients['native'].items():
        current = shared_gradients['sparse'][name]
        assert torch.allclose(original, current, atol=1e-6, rtol=1e-5), name
        shared_differences[name] = float((current - original).abs().max())
    del shared_gradients
    verify_state(before_training=True)
    preflight.update(optimizer_steps=0, trainable_parameters={
                         arm: sum(p.numel() for p in parameters.values())
                         for arm, parameters in trainable.items()},
                     native_learning_rate=1e-5, sparse_learning_rate=1e-4,
                     weight_decay=args.weight_decay, clip_norm=args.clip_norm,
                     fit_rows=26747, fit_scenes=len(scenes['fit']), holdout_rows=6172, holdout_scenes=98,
                     identical_grounding=True, frozen_state_unchanged=True,
                     native_warmup_forwards=1, shared_gpu_gradient_norms_exact=True,
                     shared_gradient_max_abs_differences=shared_differences,
                     shared_gradient_atol=1e-6, shared_gradient_rtol=1e-5)
    write_json(addon / 'preflight.json', preflight)
    print('SPARSE POINT PAIR PREFLIGHT', json.dumps(preflight), flush=True)
    del outputs, loss, preflight_inputs, preflight_batch, first_grounding, snapshot

    def evaluate(stage):
        seed_everything(1000)
        records = []
        with torch.no_grad():
            for index, raw in enumerate(loader('holdout', 1000, 16, False)):
                inputs, batch = inputs_for(raw)
                observations, snapshots = {}, {}
                for arm, model in models.items():
                    outputs = model(inputs)
                    outputs.update(batch)
                    observations[arm] = diagnose_root_candidates(outputs, evaluator)
                    snapshots[arm] = grounding_snapshot(outputs)
                require_same_grounding(snapshots['native'], snapshots['sparse'])
                for bid, row_id in enumerate(batch['sparse_row_id'].tolist()):
                    digest = hashlib.sha256()
                    for tensor in snapshots['native'].values():
                        digest.update(tensor[bid].cpu().numpy().tobytes())
                    record = {'id': row_id, 'scan_id': raw_rows[row_id]['scan_id'], 'grounding_sha256': digest.hexdigest(),
                              'input_point_sha256': hashlib.sha256(inputs['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest()}
                    for arm in models:
                        observed = observations[arm][bid]
                        rec = observed['rec_selection']
                        oracle = observed['box_oracle_after_filter']
                        record[arm] = {'rec_query': None if rec is None else rec['query'],
                                       'rec_box_iou': None if rec is None else rec['box_iou'],
                                       'rec_query_mask_iou': None if rec is None else rec['mask_iou'],
                                       'mask_query': observed['mask_selection']['query'],
                                       'mask_iou': observed['mask_selection']['mask_iou'],
                                       'legal_box_oracle_iou': None if oracle is None else oracle['box_iou'],
                                       'legal_box_oracle_query': None if oracle is None else oracle['query'],
                                       'legal_box_oracle_query_mask_iou': None if oracle is None else oracle['mask_iou']}
                    if stage == 'baseline':
                        assert record['native'] == record['sparse'], row_id
                    records.append(record)
                if (index + 1) % 50 == 0:
                    print('SPARSE POINT PAIR EVAL', json.dumps({'stage': stage, 'batches': index + 1, 'rows': len(records),
                          'elapsed_seconds': time.time() - started}), flush=True)
        assert [row['id'] for row in records] == partitions['holdout']
        write_json(addon / (stage + '_rows.json'), records)
        return records

    baseline = evaluate('baseline')
    reference = json.loads(Path(manifest['baseline_reference']).read_text())
    assert len(reference) == len(baseline)
    for original, current in zip(reference, baseline):
        for key in ['id', 'scan_id', 'grounding_sha256', 'input_point_sha256', 'native']:
            assert original[key] == current[key], (current['id'], key)
    write_json(addon / 'baseline_identity.json', {
        'rows': len(baseline), 'protected_reference_exact': True,
        'reference_sha256': manifest['baseline_reference_sha256'],
        'current_rows_sha256': file_sha(addon / 'baseline_rows.json'), 'optimizer_steps': 0})
    del reference
    seen_ids = []
    fit_point_records = []
    step = 0
    for epoch in range(1):
        seed_everything(epoch)
        for raw in loader('fit', epoch, 4, True):
            inputs, batch = inputs_for(raw)
            fit_point_records.append({'step': step + 1, 'row_ids': batch['sparse_row_id'].tolist(),
                'point_cloud_sha256': [hashlib.sha256(cloud.cpu().numpy().tobytes()).hexdigest()
                                       for cloud in inputs['point_clouds']]})
            snapshots, statistics = {}, {}
            for arm, model in models.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                outputs.update(batch)
                loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
                assert torch.isfinite(loss)
                snapshots[arm] = grounding_snapshot(outputs)
                loss.backward()
                for name, parameter in trainable[arm].items():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (arm, name)
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable[arm].values(), args.clip_norm)
                assert torch.isfinite(gradient_norm)
                optimizer.step()
                statistics[arm] = {'loss': float(loss), 'gradient_norm_before_clip': float(gradient_norm)}
            require_same_grounding(snapshots['native'], snapshots['sparse'])
            step += 1
            seen_ids.extend(batch['sparse_row_id'].tolist())
            if step % 64 == 0:
                print('SPARSE POINT PAIR TRAIN', json.dumps({'step': step, 'epoch': epoch, 'arms': statistics,
                      'sparse_output_weight_norm': float(sparse_addon.output.weight.norm()),
                      'elapsed_seconds': time.time() - started}), flush=True)
    assert step == 6687
    for epoch in range(1):
        assert sorted(seen_ids) == partitions['fit']
    assert len(fit_point_records) == 6687
    assert [row_id for row in fit_point_records for row_id in row['row_ids']] == seen_ids
    assert all(len(row['row_ids']) == len(row['point_cloud_sha256']) == 4 for row in fit_point_records[:-1])
    assert len(fit_point_records[-1]['row_ids']) == len(fit_point_records[-1]['point_cloud_sha256']) == 3
    write_json(addon / 'fit_point_batches.json', fit_point_records)
    changed = verify_state()
    assert len(changed['native']) == 16 and len(changed['sparse']) == 33
    terminal = evaluate('terminal')
    for original, updated in zip(baseline, terminal):
        for key in ['id', 'scan_id', 'grounding_sha256', 'input_point_sha256']:
            assert original[key] == updated[key], (original['id'], key)
        for arm in models:
            for key in ['rec_query', 'rec_box_iou', 'mask_query', 'legal_box_oracle_query', 'legal_box_oracle_iou']:
                assert original[arm][key] == updated[arm][key], (original['id'], arm, key)
    changed = verify_state()
    artifacts = {}
    for arm in models:
        optimizer = optimizers[arm]
        assert len(optimizer.state) == len(trainable[arm])
        assert all(float(value['step']) == 6687 for value in optimizer.state.values())
        assert all(torch.isfinite(value[key]).all() for value in optimizer.state.values() for key in ['exp_avg', 'exp_avg_sq'])
        expected_rates = [1e-5] if arm == 'native' else [1e-5, 1e-4]
        assert [group['lr'] for group in optimizer.param_groups] == expected_rates
        assert all(group['weight_decay'] == .0005 for group in optimizer.param_groups)
        path = addon / (arm + '_mask_state.pt')
        assert not path.exists()
        torch.save({'mask_projection_state': {name: parameter.detach().cpu() for name, parameter in trainable[arm].items()},
                    'optimizer': optimizer.state_dict(), 'steps': step, 'arm': arm,
                    'parent_checkpoint_sha256': manifest['checkpoint_sha256']}, path)
        artifacts[arm] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha(path)}
    sparse_attachment.remove()
    verify_inputs()
    receipt = {'schema': 'mcln-nr3d-sparse-point-pair-v2', 'status': 'complete', 'optimizer_steps_per_arm': step,
               'fit_rows': 26747, 'fit_scenes': len(scenes['fit']), 'epochs': 1, 'holdout_rows': 6172,
               'holdout_scenes': 98, 'heldout_scenes_seen_by_frozen_backbone': True,
               'formal_rows': 0, 'formal_promotion': False,
               'fit_order_sha256': hashlib.sha256(json.dumps(seen_ids).encode()).hexdigest(), 'fit_order_ids': seen_ids,
               'frozen_parameters_and_buffers_unchanged': True, 'source_data_and_parent_checkpoint_unchanged': True,
               'grounding_and_query_selection_exactly_equal_to_start': True, 'changed_parameter_names': changed,
               'baseline_matches_protected_reference': True,
               'native_warmup_forwards': 1, 'warmup_optimizer_steps': 0,
               'fit_point_batches_sha256': file_sha(addon / 'fit_point_batches.json'),
               'learning_rates': {'native_shared': 1e-5, 'sparse_shared': 1e-5, 'sparse_new': 1e-4},
               'baseline_rows_sha256': file_sha(addon / 'baseline_rows.json'), 'terminal_rows_sha256': file_sha(addon / 'terminal_rows.json'),
               'artifacts': artifacts, 'manifest_sha256': file_sha(options.manifest), 'elapsed_seconds': time.time() - started,
               'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
    receipt['runtime'] = {'prefix': sys.prefix, 'torch': torch.__version__,
        'spconv': spconv.__version__, 'cumm': cumm.__version__,
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32}
    write_json(addon / 'receipt.json', receipt)
    print('SPARSE POINT PAIR COMPLETE', json.dumps({key: value for key, value in receipt.items() if key != 'fit_order_ids'}), flush=True)


if __name__ == '__main__':
    main()
