"""Isolated native-Decoder L1 preflight and fixed paired training."""

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
    parser.add_argument('--stage', choices=['preflight', 'train'], required=True)
    options = parser.parse_args()
    addon_dir = options.manifest.parent
    destination = addon_dir / options.stage
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])

    def verify_inputs():
        source_manifest = source / 'g0_source_manifest.json'
        assert file_sha(source_manifest) == manifest['source_manifest_sha256']
        for name, digest in json.loads(source_manifest.read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(addon_dir / name) == digest, name
        for name, metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'], name
        assert file_sha(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']
        assert file_sha(Path(manifest['m3_receipt'])) == manifest['m3_receipt_sha256']

    verify_inputs()
    if options.stage == 'train':
        preflight_path = addon_dir / 'preflight/receipt.json'
        preflight = json.loads(preflight_path.read_text())
        assert preflight['status'] == 'complete' and preflight['optimizer_steps'] == 0
        assert preflight['manifest_sha256'] == file_sha(options.manifest)
        assert preflight['zero_start_identity'] and preflight['native_loss_connected']
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(addon_dir / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator
    from scripts.nr3d_candidate_contract import diagnose_root_candidates
    from scripts.nr3d_text_position_key import TextPositionKey, LastTextAttentionIntervention
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
    for index, row in enumerate(raw_rows):
        fold = int(hashlib.sha256((manifest['split_salt'] + '\0' + row['scan_id']).encode()).hexdigest()[:8], 16) % 5
        partitions['holdout' if fold == 0 else 'fit'].append(index)
    assert partitions == manifest['row_ids']
    assert len(partitions['fit']) == 26747 and len(partitions['holdout']) == 6172
    scenes = {name: {raw_rows[i]['scan_id'] for i in ids} for name, ids in partitions.items()}
    assert not scenes['fit'].intersection(scenes['holdout'])
    assert len(scenes['fit']) == 413 and len(scenes['holdout']) == 98
    selected_ids = manifest['preflight_row_ids'] if options.stage == 'preflight' else sorted(partitions['fit'] + partitions['holdout'])
    assert all(i in partitions['fit'] for i in manifest['preflight_row_ids'])
    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu')
    assert checkpoint['evaluation_only'] and 'optimizer' not in checkpoint
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(checkpoint['config']))
    assert not args.source_choice_selector_train_only and not args.use_source_moe
    assert args.mask_loss_scale == args.consistency_loss_scale == 1
    assert args.weight_decay == .0005 and args.clip_norm == .1
    initial_state = {name[7:]: value for name, value in checkpoint['model'].items()}
    del checkpoint
    model = TrainTester.get_model(args).cuda().eval()
    model.load_state_dict(initial_state, strict=True)
    model.requires_grad_(False)
    assert model.decoder_query_adapter is None
    attention = model.decoder[-1].cross_l
    assert attention.embed_dim == 288 and attention.num_heads == 8
    addons = {mode: TextPositionKey(288, 8, mode).cuda() for mode in ['text', 'position']}
    assert all(sum(p.numel() for p in addon.parameters()) == 82944 for addon in addons.values())
    criterion, set_criterion = TrainTester.get_criterion(args)
    evaluator = GroundingEvaluator(only_root=True, prefixes=['last_'], topks=[1],
                                   filter_non_gt_boxes=True, eval_use_selector_choice_scores=True)

    def verify_state():
        assert set(model.state_dict()) == set(initial_state)
        for name, value in model.state_dict().items():
            assert torch.equal(value.detach().cpu(), initial_state[name]), name
        assert all(parameter.grad is None for parameter in model.parameters())

    class SelectedDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in selected_ids]
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            item = super().__getitem__(index)
            item['l1_row_id'] = np.int64(self.l1_ids[index])
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
    dataset_ids = {'preflight': selected_ids} if options.stage == 'preflight' else partitions
    datasets = {}
    for name, ids in dataset_ids.items():
        dataset = copy.copy(base)
        dataset.annos = [by_id[i] for i in ids]
        dataset.l1_ids = ids
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

    def tensor_hash(tensor):
        return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

    started = time.time()
    captured = {}

    def capture_query(module, inputs, output):
        captured['query'] = output.detach().clone()

    query_hook = model.decoder[-1].register_forward_hook(capture_query)

    def output_snapshot(outputs):
        result = {name: outputs[name].detach().clone()
                  for name in ['last_center', 'last_pred_size', 'selected_source_scores']}
        result['query'] = captured['query']
        for index, values in enumerate(outputs['sp_last_pred_masks']):
            result['mask_' + str(index)] = values.detach().clone()
        return result

    if options.stage == 'preflight':
        m3 = json.loads(Path(manifest['m3_receipt']).read_text())
        expected = {row['fit_row_id']: row['input_point_cloud_sha256']
                    for batch in m3['batches'] for row in batch['rows']}
        batches, forwards = [], 0
        for raw in loader('preflight', 0, 4, False):
            inputs, batch = inputs_for(raw)
            row_ids = batch['l1_row_id'].tolist()
            for index, row_id in enumerate(row_ids):
                assert tensor_hash(inputs['point_clouds'][index]) == expected[row_id], row_id
            with torch.no_grad():
                outputs = model(inputs)
                original = output_snapshot(outputs)
            forwards += 1
            batch_result = {'row_ids': row_ids, 'arms': {}}
            for mode, addon in addons.items():
                attachment = LastTextAttentionIntervention(model, addon)
                outputs = model(inputs)
                current = output_snapshot(outputs)
                assert all(torch.equal(original[name], current[name]) for name in original), mode
                outputs.update(batch)
                total, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
                grounding = (outputs['loss_ce'] + 5 * outputs['loss_bbox'] + outputs['loss_giou']
                             + outputs['loss_sem_align']) / (args.num_decoder_layers + 1)
                components = {'loss_mask': 10, 'loss_dice': 2, 'sp_loss_mask': 5, 'sp_loss_dice': 1,
                              'adaptive_weight_loss_mask': 10, 'adaptive_weight_loss_dice': 2,
                              'corresponding_loss_mask': 10, 'corresponding_loss_dice': 2}
                mask_loss = sum(outputs[name] * scale for name, scale in components.items())
                gradients = {}
                for name, loss in [('total', total), ('grounding', grounding), ('mask', mask_loss)]:
                    assert torch.isfinite(loss)
                    gradient, = torch.autograd.grad(loss, [addon.weight], retain_graph=name != 'mask')
                    assert torch.isfinite(gradient).all() and gradient.norm() > 0, (mode, name)
                    gradients[name] = {'loss': float(loss), 'weight_gradient_norm': float(gradient.norm())}
                del total, grounding, mask_loss, loss, outputs, gradient
                with torch.no_grad():
                    addon.weight.copy_(torch.eye(288, device='cuda') * .001)
                    outputs = model(inputs)
                    perturbed = output_snapshot(outputs)
                    deltas = {name: float((perturbed[name]-original[name]).abs().max()) for name in original}
                    assert all(torch.isfinite(value).all() for value in perturbed.values())
                    assert deltas['query'] > 0 and deltas['last_center'] > 0 and deltas['last_pred_size'] > 0
                    assert max(deltas[name] for name in deltas if name.startswith('mask_')) > 0
                    addon.weight.zero_()
                assert attachment.position_calls == attachment.decoder_calls == attachment.attention_calls == 2
                attachment.remove()
                forwards += 2
                batch_result['arms'][mode] = {'native_gradients': gradients, 'fixed_diagonal_intervention_max_abs_deltas': deltas}
            batches.append(batch_result)
            print('L1 PREFLIGHT BATCH', json.dumps(batch_result), flush=True)
        verify_state()
        verify_inputs()
        assert all(torch.count_nonzero(addon.weight) == 0 for addon in addons.values())
        assert forwards == 20 and len(batches) == 4
        receipt = {'schema': 'mcln-text-position-l1-preflight-v1', 'status': 'complete',
                   'rows': 16, 'fit_scenes': 16, 'forwards': forwards, 'optimizer_steps': 0,
                   'zero_start_identity': True, 'native_loss_connected': True,
                   'frozen_state_unchanged': True, 'inputs_identical_to_m3': True,
                   'checkpoint_key_order_matches_model': list(model.state_dict()) == list(initial_state),
                   'source_data_and_parent_checkpoint_unchanged': True,
                   'formal_rows': 0, 'holdout_rows': 0, 'manifest_sha256': file_sha(options.manifest),
                   'parameters_per_arm': 82944, 'batches': batches,
                   'elapsed_seconds_excluding_dataset_init': time.time()-started}
        query_hook.remove()
        write_json(destination / 'receipt.json', receipt)
        print('L1 PREFLIGHT COMPLETE', json.dumps({k:v for k,v in receipt.items() if k != 'batches'}), flush=True)
        return

    query_hook.remove()
    attachment = LastTextAttentionIntervention(model, addons['text'])
    optimizers = {mode: torch.optim.AdamW(addon.parameters(), lr=1e-5, weight_decay=args.weight_decay)
                  for mode, addon in addons.items()}

    def evaluate(stage):
        seed_everything(1000)
        records = []
        arm_names = ['text'] if stage == 'baseline' else ['text', 'position']
        with torch.no_grad():
            for index, raw in enumerate(loader('holdout', 1000, 16, False)):
                inputs, batch = inputs_for(raw)
                observations = {}
                for mode in arm_names:
                    attachment.addon = addons[mode]
                    outputs = model(inputs)
                    outputs.update(batch)
                    observations[mode] = diagnose_root_candidates(outputs, evaluator)
                for bid, row_id in enumerate(batch['l1_row_id'].tolist()):
                    record = {'id': row_id, 'scan_id': raw_rows[row_id]['scan_id'],
                              'input_point_sha256': tensor_hash(inputs['point_clouds'][bid])}
                    for mode in arm_names:
                        observed = observations[mode][bid]
                        rec = observed['rec_selection']
                        oracle = observed['box_oracle_after_filter']
                        record['protected' if stage == 'baseline' else mode] = {
                            'rec_query': None if rec is None else rec['query'],
                            'rec_box_iou': None if rec is None else rec['box_iou'],
                            'mask_query': observed['mask_selection']['query'],
                            'mask_iou': observed['mask_selection']['mask_iou'],
                            'legal_box_oracle_iou': None if oracle is None else oracle['box_iou'],
                            'candidate_count': observed['score_profiles']['protected_selector']['after_filter']['candidate_count'],
                        }
                    records.append(record)
                if (index+1) % 50 == 0:
                    print('L1 EVAL', json.dumps({'stage': stage, 'rows': len(records), 'batches': index+1,
                                                'elapsed_seconds': time.time()-started}), flush=True)
        assert [row['id'] for row in records] == partitions['holdout']
        write_json(destination / (stage+'_rows.json'), records)
        return records

    baseline = evaluate('baseline')
    seed_everything(0)
    seen_ids = []
    for step, raw in enumerate(loader('fit', 0, 4, True), 1):
        inputs, batch = inputs_for(raw)
        statistics = {}
        for mode, addon in addons.items():
            attachment.addon = addon
            optimizer = optimizers[mode]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            outputs.update(batch)
            loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
            assert torch.isfinite(loss)
            loss.backward()
            assert addon.weight.grad is not None and torch.isfinite(addon.weight.grad).all()
            norm = torch.nn.utils.clip_grad_norm_(addon.parameters(), args.clip_norm)
            assert torch.isfinite(norm)
            optimizer.step()
            statistics[mode] = {'loss': float(loss), 'gradient_norm_before_clip': float(norm),
                                'weight_norm': float(addon.weight.norm())}
        seen_ids.extend(batch['l1_row_id'].tolist())
        if step % 128 == 0 or step == 6687:
            print('L1 TRAIN', json.dumps({'step': step, 'rows_seen': len(seen_ids), 'arms': statistics,
                                         'elapsed_seconds': time.time()-started}), flush=True)
    assert step == 6687 and sorted(seen_ids) == partitions['fit']
    verify_state()
    terminal = evaluate('terminal')
    for original, updated in zip(baseline, terminal):
        for key in ['id', 'scan_id', 'input_point_sha256']:
            assert original[key] == updated[key], (original['id'], key)
    verify_state()
    artifacts = {}
    for mode, addon in addons.items():
        assert torch.isfinite(addon.weight).all() and addon.weight.norm() > 0
        optimizer = optimizers[mode]
        assert len(optimizer.state) == 1
        for value in optimizer.state.values():
            assert float(value['step']) == 6687
            assert all(torch.isfinite(value[key]).all() for key in ['exp_avg', 'exp_avg_sq'])
        path = destination / (mode+'_key_state.pt')
        assert not path.exists()
        torch.save({'addon_state': addon.state_dict(), 'optimizer': optimizer.state_dict(),
                    'mode': mode, 'steps': step, 'parent_checkpoint_sha256': manifest['checkpoint_sha256']}, path)
        artifacts[mode] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha(path)}
    attachment.remove()
    verify_inputs()
    receipt = {'schema': 'mcln-text-position-l1-training-v1', 'status': 'complete',
               'optimizer_steps_per_arm': step, 'fit_rows': 26747, 'fit_scenes': 413,
               'holdout_rows': 6172, 'holdout_scenes': 98, 'formal_rows': 0, 'formal_promotion': False,
               'heldout_scenes_seen_by_frozen_backbone': True, 'epochs': 1,
               'frozen_parameters_and_buffers_unchanged': True, 'source_data_and_parent_checkpoint_unchanged': True,
               'fit_order_sha256': hashlib.sha256(json.dumps(seen_ids).encode()).hexdigest(), 'fit_order_ids': seen_ids,
               'baseline_rows_sha256': file_sha(destination/'baseline_rows.json'),
               'terminal_rows_sha256': file_sha(destination/'terminal_rows.json'),
               'preflight_receipt_sha256': file_sha(addon_dir/'preflight/receipt.json'),
               'artifacts': artifacts, 'manifest_sha256': file_sha(options.manifest),
               'elapsed_seconds_excluding_dataset_init': time.time()-started,
               'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
    write_json(destination/'receipt.json', receipt)
    print('L1 COMPLETE', json.dumps({k:v for k,v in receipt.items() if k != 'fit_order_ids'}), flush=True)


if __name__ == '__main__':
    main()
