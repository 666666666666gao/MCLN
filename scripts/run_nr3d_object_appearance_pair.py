"""Matched final Decoder object-attention learning with raw point appearance."""

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
    directory = options.manifest.parent
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])
    assert manifest['epochs'] == 2 and manifest['steps_per_arm'] == 1024
    assert manifest['train_batch_size'] == 4 and manifest['eval_batch_size'] == 16
    assert manifest['learning_rate'] == 1e-5 and manifest['no_augmentation']

    def verify_inputs():
        source_manifest = source / 'g0_source_manifest.json'
        assert file_sha(source_manifest) == manifest['source_manifest_sha256']
        for name, digest in json.loads(source_manifest.read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(directory / name) == digest, name
        for name, metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'], name
        for key in ['checkpoint', 'native_preflight_receipt', 'crop_audit_receipt', 'protected_baseline_rows']:
            assert file_sha(Path(manifest[key])) == manifest[key + '_sha256'], key

    verify_inputs()
    preflight_receipt = json.loads(Path(manifest['native_preflight_receipt']).read_text())
    assert preflight_receipt['status'] == 'complete' and preflight_receipt['optimizer_steps'] == 0
    assert preflight_receipt['zero_start_identity'] and preflight_receipt['native_grounding_loss_connected']
    assert preflight_receipt['early_queries_and_sampling_unchanged']
    assert preflight_receipt['original_text_mask_and_alpha_unchanged']
    assert preflight_receipt['frozen_state_inputs_and_addon_restored']
    crop_audit = json.loads(Path(manifest['crop_audit_receipt']).read_text())
    assert crop_audit['input_contract_pass'] and crop_audit['torch_module_predicate_matches_numpy_explicit_bounds']
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator
    from scripts.nr3d_candidate_contract import diagnose_root_candidates
    from scripts.nr3d_object_point_appearance import ObjectPointAppearanceResidual, LastDecoderObjectAppearanceIntervention
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
    partitions['fit'] = partitions['fit'][:2048]
    assert partitions == manifest['row_ids']
    scenes = {name: {raw_rows[i]['scan_id'] for i in ids} for name, ids in partitions.items()}
    assert not scenes['fit'].intersection(scenes['holdout'])
    assert len(scenes['fit']) == 262 and len(scenes['holdout']) == 98
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
    assert args.num_decoder_layers == 6 and args.butd_cls
    initial_state = {name[7:]: value for name, value in checkpoint['model'].items()}
    del checkpoint
    models, optimizers, trainable, captured_queries = {}, {}, {}, {}
    handles = []

    def query_hook(arm, index):
        def capture(module, inputs, output):
            captured_queries[arm]['query_' + str(index)] = output.detach().clone()
        return capture

    prefixes = ('decoder.5.cross_d.', 'decoder.5.norm_d.')
    for arm in ['native', 'appearance']:
        model = TrainTester.get_model(args).cuda().eval()
        model.load_state_dict(initial_state, strict=True)
        assert model.decoder_query_adapter is None and model.query_mask_fusion_calibrator is None
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        assert len(parameters) == 6 and sum(p.numel() for p in parameters.values()) == 333504
        if arm == 'native':
            # Match the module initialization order of the completed native preflight.
            appearance_addon = ObjectPointAppearanceResidual().cuda()
            initial_addon = {name: value.detach().cpu().clone() for name, value in appearance_addon.state_dict().items()}
        else:
            attachment = LastDecoderObjectAppearanceIntervention(model, appearance_addon)
            parameters.update({'appearance.' + name: value for name, value in appearance_addon.named_parameters()})
        captured_queries[arm] = {}
        handles.extend(layer.register_forward_hook(query_hook(arm, index))
                       for index, layer in enumerate(model.decoder))
        models[arm] = model
        trainable[arm] = parameters
        optimizers[arm] = torch.optim.AdamW(parameters.values(), lr=1e-5, weight_decay=args.weight_decay)
    assert len(trainable['appearance']) == 11
    assert sum(p.numel() for p in trainable['appearance'].values()) == 374976
    assert list(trainable['native']) == list(trainable['appearance'])[:6]
    criterion, set_criterion = TrainTester.get_criterion(args)
    evaluator = GroundingEvaluator(only_root=True, prefixes=['last_'], topks=[1],
                                   filter_non_gt_boxes=True, eval_use_selector_choice_scores=True)

    def verify_state(before_training=False):
        changed = {}
        for arm, model in models.items():
            changed[arm] = []
            assert set(model.state_dict()) == set(initial_state)
            for name, value in model.state_dict().items():
                same = torch.equal(value.detach().cpu(), initial_state[name])
                if before_training or name not in trainable[arm]:
                    assert same, (arm, name)
                elif not same:
                    changed[arm].append(name)
            assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in trainable[arm])
            if arm == 'appearance':
                for name, value in appearance_addon.state_dict().items():
                    same = torch.equal(value.detach().cpu(), initial_addon[name])
                    if before_training:
                        assert same, name
                    elif not same:
                        changed[arm].append('appearance.' + name)
        return changed

    class SelectedDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in selected_ids]
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            item = super().__getitem__(index)
            item['appearance_row_id'] = np.int64(self.appearance_ids[index])
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
        dataset.appearance_ids = ids
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

    def tensor_hash(tensors):
        digest = hashlib.sha256()
        for tensor in tensors:
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def frozen_hashes(arm, outputs):
        early = [outputs[key] for key in ['seed_inds', 'query_points_sample_inds']]
        early += [captured_queries[arm]['query_' + str(i)] for i in range(5)]
        return {'early_queries_sha256': [tensor_hash([value[i] for value in early]) for i in range(len(outputs['last_pred_masks']))],
                'frozen_text_mask_alpha_sha256': [tensor_hash([mask, alpha])
                    for mask, alpha in zip(outputs['last_pred_masks'], outputs['adaptive_weights'])]}

    def output_snapshot(outputs):
        result = {name: outputs[name].detach().clone() for name in
                  ['last_center', 'last_pred_size', 'selected_source_scores', 'last_sem_cls_scores', 'last_proj_queries']}
        result.update({'query_mask_' + str(i): value.detach().clone() for i, value in enumerate(outputs['sp_last_pred_masks'])})
        return result

    started = time.time()
    preflight_inputs, preflight_batch = inputs_for(next(iter(loader('fit', 0, 4, False))))
    preflight, first_gradients, snapshots, signatures = {}, {}, {}, {}
    for arm, model in models.items():
        outputs = model(preflight_inputs)
        signatures[arm] = frozen_hashes(arm, outputs)
        snapshots[arm] = output_snapshot(outputs)
        snapshots[arm]['last_query'] = captured_queries[arm]['query_5']
        outputs.update(preflight_batch)
        loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        assert torch.isfinite(loss)
        loss.backward()
        norms = {}
        for name, parameter in trainable[arm].items():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            norms[name] = float(parameter.grad.norm())
            if name.startswith('appearance.') and name != 'appearance.output.weight':
                assert norms[name] == 0, name
            else:
                assert norms[name] > 0, name
            if arm == 'native':
                first_gradients[name] = parameter.grad.detach().clone()
            elif name in first_gradients:
                assert torch.equal(first_gradients[name], parameter.grad), name
        preflight[arm] = {'loss': float(loss), 'gradient_norms': norms}
        optimizers[arm].zero_grad(set_to_none=True)
        del outputs, loss
    assert signatures['native'] == signatures['appearance']
    assert preflight['native']['loss'] == preflight['appearance']['loss']
    assert all(torch.equal(value, snapshots['appearance'][name]) for name, value in snapshots['native'].items())
    verify_state(before_training=True)
    preflight.update(optimizer_steps=0, zero_outputs_and_shared_gradients_identical=True,
                     trainable_parameters={arm: sum(p.numel() for p in parameters.values()) for arm, parameters in trainable.items()},
                     learning_rate=1e-5, weight_decay=args.weight_decay, clip_norm=args.clip_norm)
    write_json(directory / 'preflight.json', preflight)
    print('OBJECT APPEARANCE PAIR PREFLIGHT', json.dumps(preflight), flush=True)
    del preflight_inputs, preflight_batch, first_gradients, snapshots, signatures

    def evaluate(stage):
        seed_everything(1000)
        records = []
        with torch.no_grad():
            for index, raw in enumerate(loader('holdout', 1000, 16, False)):
                inputs, batch = inputs_for(raw)
                observations, hashes, arm_rows = {}, {}, {}
                for arm, model in models.items():
                    outputs = model(inputs)
                    outputs.update(batch)
                    observations[arm] = diagnose_root_candidates(outputs, evaluator)
                    hashes[arm] = frozen_hashes(arm, outputs)
                    arm_rows[arm] = []
                    for bid, observed in enumerate(observations[arm]):
                        rec, oracle = observed['rec_selection'], observed['box_oracle_after_filter']
                        arm_rows[arm].append({
                            'rec_query': None if rec is None else rec['query'],
                            'rec_box_iou': None if rec is None else rec['box_iou'],
                            'rec_query_mask_iou': None if rec is None else rec['mask_iou'],
                            'mask_query': observed['mask_selection']['query'], 'mask_iou': observed['mask_selection']['mask_iou'],
                            'legal_box_oracle_iou': None if oracle is None else oracle['box_iou'],
                            'legal_box_oracle_query': None if oracle is None else oracle['query'],
                            'legal_box_oracle_query_mask_iou': None if oracle is None else oracle['mask_iou'],
                            'candidate_profile': observed['score_profiles']['protected_selector'],
                            'last_query_sha256': tensor_hash([captured_queries[arm]['query_5'][bid]]),
                            'grounding_sha256': tensor_hash([outputs[key][bid] for key in
                                ['last_center', 'last_pred_size', 'selected_source_scores']]),
                            'semantic_sha256': tensor_hash([outputs[key][bid] for key in ['last_sem_cls_scores', 'last_proj_queries']])})
                    del outputs
                assert hashes['native'] == hashes['appearance']
                for bid, row_id in enumerate(batch['appearance_row_id'].tolist()):
                    record = {'id': row_id, 'scan_id': raw_rows[row_id]['scan_id'],
                              'input_point_sha256': tensor_hash([inputs['point_clouds'][bid]])}
                    record.update({key: values[bid] for key, values in hashes['native'].items()})
                    record.update({arm: arm_rows[arm][bid] for arm in models})
                    if stage == 'baseline':
                        assert record['native'] == record['appearance'], row_id
                    records.append(record)
                if (index + 1) % 50 == 0:
                    print('OBJECT APPEARANCE PAIR EVAL', json.dumps({'stage': stage, 'batches': index + 1,
                          'rows': len(records), 'elapsed_seconds': time.time() - started}), flush=True)
        assert [row['id'] for row in records] == partitions['holdout']
        write_json(directory / (stage + '_rows.json'), records)
        return records

    baseline = evaluate('baseline')
    protected = json.loads(Path(manifest['protected_baseline_rows']).read_text())
    assert len(protected) == len(baseline) == 6172
    for old, new in zip(protected, baseline):
        for key in ['id', 'scan_id', 'input_point_sha256', 'frozen_text_mask_alpha_sha256']:
            assert old[key] == new[key], (old['id'], key)
        assert old['grounding_sha256'] == new['native']['grounding_sha256']
        assert all(value == new['native'][key] for key, value in old['native'].items()), old['id']
    write_json(directory / 'baseline_identity.json', {'status': 'pass', 'rows': 6172,
               'protected_baseline_rows_sha256': manifest['protected_baseline_rows_sha256'],
               'baseline_rows_sha256': file_sha(directory / 'baseline_rows.json'), 'both_arms_equal': True,
               'optimizer_steps': 0, 'formal_rows': 0})
    del protected
    seen_ids, fit_point_batches = [], []
    step = 0
    for epoch in range(2):
        seed_everything(epoch)
        for raw in loader('fit', epoch, 4, True):
            inputs, batch = inputs_for(raw)
            hashes, statistics = {}, {}
            record = {'step': step + 1, 'row_ids': batch['appearance_row_id'].tolist(),
                      'point_tensor_sha256': tensor_hash([inputs['point_clouds']])}
            for arm, model in models.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                hashes[arm] = frozen_hashes(arm, outputs)
                outputs.update(batch)
                loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
                assert torch.isfinite(loss)
                loss.backward()
                for name, parameter in trainable[arm].items():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (arm, name)
                norm = torch.nn.utils.clip_grad_norm_(trainable[arm].values(), args.clip_norm)
                assert torch.isfinite(norm)
                optimizer.step()
                statistics[arm] = {'loss': float(loss), 'gradient_norm_before_clip': float(norm)}
                del outputs, loss
            assert hashes['native'] == hashes['appearance']
            record.update(hashes['native'])
            fit_point_batches.append(record)
            step += 1
            seen_ids.extend(batch['appearance_row_id'].tolist())
            if step % 64 == 0:
                print('OBJECT APPEARANCE PAIR TRAIN', json.dumps({'step': step, 'epoch': epoch, 'arms': statistics,
                      'elapsed_seconds': time.time() - started}), flush=True)
    assert step == 1024
    write_json(directory / 'fit_point_batches.json', fit_point_batches)
    for epoch in range(2):
        assert sorted(seen_ids[epoch * 2048:(epoch + 1) * 2048]) == partitions['fit']
    changed = verify_state()
    assert len(changed['native']) == 6 and len(changed['appearance']) == 11
    terminal = evaluate('terminal')
    for old, new in zip(baseline, terminal):
        for key in ['id', 'scan_id', 'input_point_sha256', 'early_queries_sha256', 'frozen_text_mask_alpha_sha256']:
            assert old[key] == new[key], (old['id'], key)
    changed = verify_state()
    artifacts = {}
    for arm in models:
        optimizer = optimizers[arm]
        assert len(optimizer.state) == len(trainable[arm])
        assert all(float(value['step']) == 1024 for value in optimizer.state.values())
        assert all(torch.isfinite(value[key]).all() for value in optimizer.state.values() for key in ['exp_avg', 'exp_avg_sq'])
        assert all(torch.isfinite(value).all() for value in trainable[arm].values())
        path = directory / (arm + '_object_state.pt')
        assert not path.exists()
        torch.save({'object_attention_state': {name: parameter.detach().cpu() for name, parameter in trainable[arm].items()},
                    'optimizer': optimizer.state_dict(), 'steps': step, 'arm': arm,
                    'parent_checkpoint_sha256': manifest['checkpoint_sha256']}, path)
        artifacts[arm] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha(path)}
    attachment.remove()
    for handle in handles:
        handle.remove()
    verify_inputs()
    receipt = {'schema': 'mcln-nr3d-object-appearance-pair-v1', 'status': 'complete', 'optimizer_steps_per_arm': step,
               'fit_rows': 2048, 'fit_scenes': 262, 'epochs': 2, 'holdout_rows': 6172, 'holdout_scenes': 98,
               'heldout_scenes_seen_by_frozen_backbone': True, 'formal_rows': 0, 'formal_promotion': False,
               'fit_order_sha256': hashlib.sha256(json.dumps(seen_ids).encode()).hexdigest(), 'fit_order_ids': seen_ids,
               'fit_point_batches_sha256': file_sha(directory / 'fit_point_batches.json'),
               'text_mask_and_alpha_exactly_equal_to_start': True, 'early_queries_and_sampling_exactly_equal_to_start': True,
               'frozen_parameters_and_buffers_unchanged': True, 'source_data_and_parent_checkpoint_unchanged': True,
               'baseline_matches_protected_6172_rows': True, 'candidate_boxes_and_selections_allowed_to_change': True,
               'changed_parameter_names': changed, 'baseline_rows_sha256': file_sha(directory / 'baseline_rows.json'),
               'terminal_rows_sha256': file_sha(directory / 'terminal_rows.json'), 'artifacts': artifacts,
               'manifest_sha256': file_sha(options.manifest), 'elapsed_seconds': time.time() - started,
               'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
    write_json(directory / 'receipt.json', receipt)
    print('OBJECT APPEARANCE PAIR COMPLETE', json.dumps({key: value for key, value in receipt.items() if key != 'fit_order_ids'}), flush=True)


if __name__ == '__main__':
    main()
