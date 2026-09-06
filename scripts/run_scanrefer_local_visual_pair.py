"""Fixed ScanRefer comparison of native cross_v and candidate-local visual reading."""

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


def paired_effect(reference, candidate, field, threshold):
    before = [row[field] > threshold for row in reference]
    after = [row[field] > threshold for row in candidate]
    repair = sum(new and not old for old, new in zip(before, after))
    damage = sum(old and not new for old, new in zip(before, after))
    return {'repair': repair, 'damage': damage, 'net': repair - damage}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    option = parser.parse_args()
    directory = option.manifest.parent
    manifest = json.loads(option.manifest.read_text())
    source = Path(manifest['model_source'])
    assert file_sha(source / 'local_visual_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'local_visual_source_manifest.json').read_text())['files'].items():
        assert file_sha(source / name) == digest, name
    for name, digest in manifest['files'].items():
        assert file_sha(directory / name) == digest, name
    for name, item in manifest['artifacts'].items():
        assert file_sha(item['path']) == item['sha256'], name
    assert manifest['mode'] in ('preflight', 'train')
    if manifest['mode'] == 'train':
        assert file_sha(manifest['native_probe_receipt']) == manifest['native_probe_receipt_sha256']
        native_probe = json.loads(Path(manifest['native_probe_receipt']).read_text())
        assert native_probe['status'] == 'pass' and native_probe['disposable_optimizer_steps'] == 2
    assert file_sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    split = json.loads(Path(manifest['split_protocol']).read_text())
    partitions = dict(split['row_ids'])
    assert manifest['steps_per_arm'] == math.ceil(len(partitions['fit']) / 12)
    assert manifest['epochs'] == 1 and manifest['batch_size'] == 12
    assert manifest['core_learning_rate'] == 1e-6 and manifest['local_learning_rate'] == 1e-4
    assert manifest['weight_decay'] == .0005 and manifest['clip_norm'] == .1
    assert manifest['loss'] == 'native_gt_only' and manifest['readouts_frozen']
    os.chdir(str(source))
    sys.path.insert(0, str(source))

    import copy
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from models.rec_reranker import compute_query_ious
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scripts.scanrefer_joint_readout import JointRecReadout
    from models.candidate_local_visual import CandidateLocalVisual
    from scripts.scanrefer_rec_evaluation import rec_evaluation_view
    from scripts.scanrefer_data_contract import set_scanrefer_data_root, verify_scanrefer_superpoints

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    command = build_authoritative_command(directory / 'unused_official_output')
    command = set_scanrefer_data_root(command, manifest['data_root'])
    for split_name in ['train', 'val']:
        verify_scanrefer_superpoints(manifest['data_root'], split_name, manifest['superpoint_files'][split_name])
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.data_root == manifest['data_root']
    assert args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
    assert args.butd and not args.butd_cls and not args.butd_gt
    assert args.use_color and not args.use_height and not args.use_multiview
    assert args.batch_size == 12 and args.num_decoder_layers == 6
    assert args.checkpoint_path == manifest['artifacts']['backbone']['path']
    payload = torch.load(args.checkpoint_path, map_location='cpu')
    initial = {name[7:]: value for name, value in payload['model'].items()}
    first_model = TrainTester.get_model(args).cuda().eval()
    first_model.load_state_dict(initial, strict=True)
    for name, parameter in first_model.named_parameters():
        parameter.requires_grad_(name.startswith(('decoder.5.', 'prediction_heads.5.')))
    models = {'control': first_model, 'local': copy.deepcopy(first_model)}
    models['local'].decoder[-1].local_visual = CandidateLocalVisual().cuda().eval()
    local_initial = {name: value.detach().cpu().clone()
                     for name, value in models['local'].state_dict().items()
                     if name.startswith('decoder.5.local_visual.')}
    artifacts = {name: torch.load(item['path'], map_location='cpu')
                 for name, item in manifest['artifacts'].items() if name != 'backbone'}
    first_readout = JointRecReadout(artifacts).cuda().eval().requires_grad_(False)
    readouts = {arm: first_readout for arm in models}
    readout_initial = {name: value.detach().cpu().clone() for name, value in first_readout.state_dict().items()}
    core_names = [name for name, value in first_model.named_parameters() if value.requires_grad]
    trainable, optimizers = {}, {}
    for arm, model in models.items():
        core_parameters = [value for name, value in model.named_parameters() if name in core_names]
        groups = [{'params': core_parameters, 'lr': manifest['core_learning_rate']}]
        if arm == 'local':
            groups.append({'params': list(model.decoder[-1].local_visual.parameters()),
                           'lr': manifest['local_learning_rate']})
        trainable[arm] = [value for value in model.parameters() if value.requires_grad]
        optimizers[arm] = torch.optim.AdamW(groups, weight_decay=manifest['weight_decay'])
    criterion, set_criterion = TrainTester.get_criterion(args)

    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            actual = {'fit': [], 'holdout': []}
            for index, row in enumerate(annos):
                row['_local_training_id'] = index
                code = (manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()
                fold = int(hashlib.sha256(code).hexdigest()[:8], 16) % 5
                actual['holdout' if fold == 0 else 'fit'].append(index)
            assert actual == partitions
            # Preserve all scene expressions for native distractor construction before partitioning.
            super()._scene_graph_parse(annos)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result['local_training_id'] = self.annos[index]['_local_training_id']
            return result

    dataset = FitDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=args.data_root, use_color=args.use_color, use_height=args.use_height,
        use_multiview=args.use_multiview, detect_intermediate=args.detect_intermediate,
        butd=args.butd, butd_gt=args.butd_gt, butd_cls=args.butd_cls,
        augment_det=False, skip_missing_superpoints=args.skip_missing_superpoints)
    assert [row['_local_training_id'] for row in dataset.annos] == list(range(36665))
    dataset.augment = False
    spaces = {part: sorted({dataset.annos[i]['scan_id'].split('_')[0] for i in ids})
              for part, ids in partitions.items()}
    assert not set(spaces['fit']).intersection(spaces['holdout'])
    write_json(directory / 'protocol.json', {'row_ids': partitions, 'physical_spaces': spaces,
        'source_protocol_sha256': manifest['split_protocol_sha256'],
        'previous_pretraining_has_seen_development_holdout': True,
        'steps_per_arm': manifest['steps_per_arm'], 'epochs': 1,
        'authoritative_base_command': command, 'core_trainable_tensors': core_names,
        'local_trainable_tensors': list(local_initial),
        'readouts_frozen': True, 'mode': manifest['mode']})

    def loader(part, seed, shuffle):
        return torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, partitions[part]),
            batch_size=12, shuffle=shuffle, num_workers=0,
            generator=torch.Generator().manual_seed(seed))

    tester = object.__new__(TrainTester)
    tester.logger = logging.getLogger('scanrefer-local-visual-pair')

    def evaluate(stage):
        seed_everything(1000)
        records = {arm: [] for arm in models}
        evaluators = {arm: tester._build_grounding_evaluator(args, ['last_']) for arm in models}
        begin = time.time()
        with torch.no_grad():
            for batch_index, raw in enumerate(loader('holdout', 1000, False)):
                batch = TrainTester._to_gpu(raw)
                inputs = TrainTester._get_inputs(batch)
                inputs['train'] = False
                roots = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
                root_valid = batch['box_label_mask'][:, :1].bool()
                point_hashes = [hashlib.sha256(cloud.cpu().numpy().tobytes()).hexdigest()
                                for cloud in inputs['point_clouds']]
                for arm, model in models.items():
                    outputs = model(inputs)
                    readout = readouts[arm](outputs, inputs)
                    outputs.update(batch)
                    outputs.update(readout['runtime'])
                    negative_sizes = (outputs['last_pred_size'] < 0).any(dim=-1).sum(dim=-1).tolist()
                    evaluated = rec_evaluation_view(outputs)
                    evaluator = evaluators[arm]
                    evaluator.evaluate_bbox_by_pos_align(evaluated, 'last_')
                    # Call the native semantic Mask evaluator and capture its actual per-row IoUs.
                    mask_ious = []
                    original_iou = evaluator.calculate_masks_iou

                    def record_iou(prediction, target):
                        value = original_iou(prediction, target)
                        mask_ious.append(float(value))
                        return value

                    evaluator.calculate_masks_iou = record_iou
                    evaluator.evaluate_masks_by_sem_align(evaluated, 'last_')
                    evaluator.calculate_masks_iou = original_iou
                    assert len(mask_ious) == len(roots)
                    runtime = readout['runtime']
                    selected = runtime['rec_geometry_scores'].masked_fill(
                        ~runtime['rec_geometry_valid_mask'], -float('inf')).argmax(dim=1)
                    all_ious = compute_query_ious(runtime['rec_geometry_boxes'], roots, root_valid)
                    chosen_ious = all_ious.gather(1, selected[:, None])[:, 0]
                    for index, row_id in enumerate(raw['local_training_id'].tolist()):
                        records[arm].append({'row_id': row_id, 'scan_id': raw['scan_ids'][index],
                            'physical_space': raw['scan_ids'][index].split('_')[0],
                            'point_sha256': point_hashes[index], 'rec_iou': float(chosen_ious[index]),
                            'mask_iou': mask_ious[index], 'selected_variant_position': int(selected[index]),
                            'raw_negative_size_queries': negative_sizes[index]})
                    del outputs, readout, runtime, evaluated
                if batch_index == 0 or (batch_index + 1) % 128 == 0:
                    print('SCANREFER LOCAL VISUAL EVAL', json.dumps({'stage': stage,
                        'rows': len(records['local']), 'total': len(partitions['holdout']),
                        'elapsed_seconds': time.time() - begin}), flush=True)
        metrics = {}
        for arm, rows in records.items():
            evaluator = evaluators[arm]
            count = len(rows)
            assert [row['row_id'] for row in rows] == partitions['holdout']
            metric = {'rows': count, 'mask_miou': sum(row['mask_iou'] for row in rows) / count * 100.}
            for threshold, suffix in [(.25, '025'), (.5, '050')]:
                rec_hits = sum(row['rec_iou'] > threshold for row in rows)
                mask_hits = sum(row['mask_iou'] > threshold for row in rows)
                assert evaluator.gts[('last_', threshold, 1, 'bbs')] == count
                assert evaluator.dets[('last_', threshold, 1, 'bbs')] == rec_hits
                key = 'overall_mask' if threshold == .25 else 'overall50_mask'
                assert evaluator.dets[key] == mask_hits
                metric['rec_hits' + suffix] = rec_hits
                metric['mask_hits' + suffix] = mask_hits
            assert abs(evaluator.dets['mask_sem'] - sum(row['mask_iou'] for row in rows)) < 1e-8
            metrics[arm] = metric
        write_json(directory / (stage + '_rows.json'), records)
        write_json(directory / (stage + '_metrics.json'), metrics)
        print('SCANREFER LOCAL VISUAL EVAL COMPLETE', json.dumps({'stage': stage, 'metrics': metrics,
              'elapsed_seconds': time.time() - begin}), flush=True)
        return records, metrics

    if manifest['mode'] == 'preflight':
        partitions['probe'] = split['selected_ids']
        assert len(partitions['probe']) == 16
        observations = []
        start = time.time()
        for index, raw in enumerate(loader('probe', 0, False)):
            batch = TrainTester._to_gpu(raw)
            inputs = TrainTester._get_inputs(batch)
            inputs['train'] = False
            with torch.no_grad():
                reference = models['control'](inputs)
                reference_runtime = first_readout(reference, inputs)['runtime']
            model = models['local']
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            begin = time.time()
            candidate = model(inputs)
            torch.cuda.synchronize()
            seconds = time.time() - begin
            keys = ['last_center', 'last_pred_size', 'last_sem_cls_scores',
                    'last_proj_queries', 'sp_last_pred_masks', 'last_pred_masks', 'adaptive_weights']
            for key in keys:
                if torch.is_tensor(reference[key]):
                    assert torch.equal(reference[key], candidate[key]), key
                else:
                    assert len(reference[key]) == len(candidate[key])
                    assert all(torch.equal(a, b) for a, b in zip(reference[key], candidate[key])), key
            with torch.no_grad():
                candidate_runtime = first_readout(candidate, inputs)['runtime']
                for key, value in reference_runtime.items():
                    assert torch.equal(value, candidate_runtime[key]) if torch.is_tensor(value) else value == candidate_runtime[key], key
            candidate.update(batch)
            loss, candidate = TrainTester._compute_loss(candidate, criterion, set_criterion, args)
            assert torch.isfinite(loss)
            loss.backward()
            local = model.decoder[-1].local_visual
            output_grad = float(local.output_projection.weight.grad.norm())
            assert output_grad > 0 and math.isfinite(output_grad)
            observations.append({'batch': index, 'rows': len(raw['scan_ids']),
                'forward_seconds': seconds, 'zero_initialization_output_parity': True,
                'v99_runtime_parity': True, 'native_loss': float(loss),
                'local_output_weight_gradient': output_grad,
                'point_sha256': [hashlib.sha256(cloud.cpu().numpy().tobytes()).hexdigest() for cloud in inputs['point_clouds']]})
            print('SCANREFER LOCAL VISUAL PREFLIGHT', json.dumps(observations[-1]), flush=True)
            del candidate, reference, candidate_runtime, reference_runtime, loss
        # Two disposable updates test whether gradients enter the initially zero-gated reader.
        model = models['local']
        optimizer = optimizers['local']
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            outputs.update(batch)
            loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable['local'], manifest['clip_norm'])
            assert all(torch.isfinite(p.grad).all() for p in trainable['local'] if p.grad is not None)
            optimizer.step()
            del outputs, loss
        gradient_norms = {name: float(p.grad.norm()) for name, p in model.decoder[-1].local_visual.named_parameters()}
        for name in ['point_encoder.0.weight', 'query_projection.weight', 'key_projection.weight', 'value_projection.weight']:
            assert gradient_norms[name] > 0, (name, gradient_norms[name])
        for name, value in model.state_dict().items():
            if name not in core_names and name not in local_initial:
                assert torch.equal(value.cpu(), initial[name]), name
        assert all(torch.equal(value.cpu(), readout_initial[name]) for name, value in first_readout.state_dict().items())
        receipt = {'schema': 'mcln-scanrefer-local-visual-preflight-v1', 'status': 'pass',
            'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'real_train_rows': 16, 'formal_rows': 0, 'disposable_optimizer_steps': 2,
            'checkpoint_writes': 0, 'observations': observations, 'second_step_local_gradients': gradient_norms,
            'frozen_parameters_buffers_and_readouts_unchanged': True,
            'core_trainable_tensors': core_names, 'local_trainable_tensors': list(local_initial),
            'local_parameters': sum(p.numel() for p in model.decoder[-1].local_visual.parameters()),
            'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2,
            'elapsed_seconds': time.time() - start, 'manifest_sha256': file_sha(option.manifest)}
        write_json(directory / 'receipt.json', receipt)
        print('SCANREFER LOCAL VISUAL PREFLIGHT COMPLETE', json.dumps(receipt), flush=True)
        return

    baseline, baseline_metrics = evaluate('baseline')
    assert baseline['control'] == baseline['local']
    assert baseline_metrics['control'] == baseline_metrics['local']
    for model in models.values():
        for name, value in model.state_dict().items():
            expected = local_initial[name] if name in local_initial else initial[name]
            assert torch.equal(value.detach().cpu(), expected), name
    seed_everything(0)
    batches = []
    begin_training = time.time()
    previous_report = begin_training
    step = 0
    for raw in loader('fit', 0, True):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        roots = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
        root_valid = batch['box_label_mask'][:, :1].bool()
        statistics = {}
        for arm, model in models.items():
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            outputs.update(batch)
            native, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
            loss = native
            assert torch.isfinite(loss), (arm, step)
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in trainable[arm]
                       if parameter.grad is not None), (arm, step)
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable[arm], manifest['clip_norm'])
            assert torch.isfinite(gradient_norm), (arm, step)
            optimizer.step()
            statistics[arm] = {'loss': float(loss), 'native_loss': float(native),
                'gradient_norm_before_clip': float(gradient_norm)}
            del outputs, native, loss
        step += 1
        batches.append({'step': step, 'row_ids': raw['local_training_id'].tolist(),
            'point_sha256': [hashlib.sha256(cloud.cpu().numpy().tobytes()).hexdigest()
                            for cloud in inputs['point_clouds']]})
        if step == 1 or step % 64 == 0 or step == manifest['steps_per_arm']:
            elapsed = time.time() - begin_training
            print('SCANREFER LOCAL VISUAL TRAIN', json.dumps({'step': step, 'total': manifest['steps_per_arm'],
                'arms': statistics, 'elapsed_seconds': elapsed,
                'since_previous_report_seconds': time.time() - previous_report,
                'estimated_training_remaining_seconds': (manifest['steps_per_arm'] - step) * elapsed / step}), flush=True)
            previous_report = time.time()
    assert step == manifest['steps_per_arm']
    assert sorted(row_id for item in batches for row_id in item['row_ids']) == partitions['fit']
    assert all(len(item['row_ids']) == 12 for item in batches[:-1])
    write_json(directory / 'fit_point_batches.json', batches)
    changed, checkpoints = {}, {}
    for arm, model in models.items():
        changed[arm] = []
        for name, value in model.state_dict().items():
            expected = local_initial[name] if name in local_initial else initial[name]
            same = torch.equal(value.detach().cpu(), expected)
            if name not in core_names and name not in local_initial:
                assert same, (arm, name)
            elif not same:
                changed[arm].append(name)
        assert changed[arm]
        optimizer = optimizers[arm]
        assert all(torch.isfinite(value[key]).all() for value in optimizer.state.values()
                   for key in ['exp_avg', 'exp_avg_sq'])
        assert all(float(value['step']) <= step for value in optimizer.state.values())
        checkpoint = directory / (arm + '_local_visual_state.pt')
        assert not checkpoint.exists()
        # Preserve the completed fit endpoint before the potentially long terminal evaluation.
        torch.save({'schema': 'mcln-scanrefer-local-visual-trained-state-v1', 'arm': arm, 'steps': step,
            'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
            'readout': readouts[arm].export_artifacts(), 'optimizer': optimizer.state_dict(),
            'core_trainable_tensors': core_names, 'local_trainable_tensors': list(local_initial) if arm == 'local' else [],
            'model_config': {'candidate_local_visual': arm == 'local'},
            'manifest_sha256': file_sha(option.manifest),
            'pretrained_artifacts': manifest['artifacts'],
            'decision_rule': 'existing V99 geometry and fixed Pareto; updated weights are a new system'}, checkpoint)
        checkpoints[arm] = {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                            'sha256': file_sha(checkpoint)}
    write_json(directory / 'fit_complete.json', {'steps_per_arm': step, 'checkpoints': checkpoints,
        'changed_core_tensors': changed, 'training_seconds': time.time() - begin_training})
    terminal, terminal_metrics = evaluate('terminal')
    for arm in models:
        for before, after in zip(baseline[arm], terminal[arm]):
            for key in ['row_id', 'scan_id', 'physical_space', 'point_sha256']:
                assert before[key] == after[key], (arm, key, before['row_id'])
    effects = {reference: {str(t): paired_effect(
        baseline['local'] if reference == 'baseline' else terminal['control'],
        terminal['local'], 'rec_iou', t) for t in (.25, .5)} for reference in ('baseline', 'control')}
    development_nonregression = all(effect['net'] >= 0 for comparison in effects.values() for effect in comparison.values())
    assert all(torch.equal(value.cpu(), readout_initial[name]) for name, value in first_readout.state_dict().items())
    receipt = {'schema': 'mcln-scanrefer-local-visual-pair-v1', 'status': 'complete',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'steps_per_arm': step, 'fit_rows': len(partitions['fit']), 'holdout_rows': len(partitions['holdout']),
        'previous_pretraining_has_seen_development_holdout': True, 'formal_rows': 0,
        'baseline_metrics': baseline_metrics, 'terminal_metrics': terminal_metrics,
        'local_rec_effects': effects, 'development_dual_rec_nonregression': development_nonregression,
        'fixed_endpoint_ready_for_official_evaluation': True, 'readouts_unchanged': True,
        'changed_core_tensors': changed, 'frozen_core_parameters_and_buffers_unchanged_after_fit': True,
        'checkpoints': checkpoints, 'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2,
        'manifest_sha256': file_sha(option.manifest),
        'evaluation_extent_policy': 'existing rec_candidate_adapter floor at 1e-6;raw loss inputs unchanged',
        'fit_batches_sha256': file_sha(directory / 'fit_point_batches.json'),
        'baseline_rows_sha256': file_sha(directory / 'baseline_rows.json'),
        'terminal_rows_sha256': file_sha(directory / 'terminal_rows.json')}
    write_json(directory / 'receipt.json', receipt)
    print('SCANREFER LOCAL VISUAL PAIR COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
