"""One fixed9508-row evaluation of protected V99, GT-only control and teacher-box candidate."""

import argparse
import datetime
import hashlib
import json
import logging
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


def row_metrics(rows):
    result = {'rows': len(rows), 'mask_miou': sum(row['mask_iou'] for row in rows) / len(rows) * 100.}
    for threshold, suffix in [(.25, '025'), (.5, '050')]:
        for field, name in [('rec_iou', 'rec'), ('mask_iou', 'mask')]:
            result[name + '_hits' + suffix] = sum(row[field] > threshold for row in rows)
    return result


def promotion_check(protected, candidate):
    assert protected['rows'] == candidate['rows'] == 9508
    checks = {
        'rec025_historical_v99': candidate['rec_hits025'] >= 5572,
        'rec050_historical_v99': candidate['rec_hits050'] >= 4797,
        'rec025_paired_protected': candidate['rec_hits025'] >= protected['rec_hits025'],
        'rec050_paired_protected': candidate['rec_hits050'] >= protected['rec_hits050'],
        'scan_mask025_paper': candidate['mask_hits025'] * 100. / 9508 >= 58.70,
        'scan_mask050_paper': candidate['mask_hits050'] * 100. / 9508 >= 50.70,
        'scan_mask_miou_paper': candidate['mask_miou'] >= 44.72,
    }
    return {'checks': checks, 'advance_to_nr3d_sr3d_rec': all(checks.values()),
            'stretch_goals_are_not_promotion_gates': True,
            'nr3d_sr3d_mask_not_a_promotion_gate': True}


def verify_training_endpoint(directory):
    receipt = json.loads((directory / 'receipt.json').read_text())
    manifest = json.loads((directory / 'input_manifest.json').read_text())
    assert receipt['status'] == 'complete' and receipt['formal_rows'] == 0
    assert receipt['steps_per_arm'] == manifest['steps_per_arm'] == 2482
    assert manifest['epochs'] == 1 and receipt['holdout_rows'] == 6887
    assert receipt['manifest_sha256'] == file_sha(directory / 'input_manifest.json')
    assert receipt['baseline_rows_sha256'] == file_sha(directory / 'baseline_rows.json')
    assert receipt['terminal_rows_sha256'] == file_sha(directory / 'terminal_rows.json')
    baseline = json.loads((directory / 'baseline_rows.json').read_text())
    terminal = json.loads((directory / 'terminal_rows.json').read_text())
    assert baseline['gt_teacher_box'] == baseline['gt_only']
    for arm in ['gt_only', 'gt_teacher_box']:
        assert len(baseline[arm]) == len(terminal[arm]) == 6887
        assert row_metrics(baseline[arm]) == receipt['baseline_metrics'][arm]
        assert row_metrics(terminal[arm]) == receipt['terminal_metrics'][arm]
        for before, after in zip(baseline[arm], terminal[arm]):
            for key in ['row_id', 'scan_id', 'physical_space', 'point_sha256']:
                assert before[key] == after[key]
    assert manifest['readouts_frozen'] and receipt['readout_parameters_and_metadata_unchanged']
    assert receipt['eligible_for_fixed_terminal_formal_evaluation']
    for reference in [receipt['baseline_metrics']['gt_teacher_box'], receipt['terminal_metrics']['gt_only']]:
        assert all(receipt['terminal_metrics']['gt_teacher_box']['rec_hits' + suffix] >= reference['rec_hits' + suffix]
                   for suffix in ['025', '050'])
    assert receipt['schema'] == 'mcln-scanrefer-native-box-transfer-pair-v1'
    for checkpoint in receipt['checkpoints'].values():
        assert file_sha(checkpoint['path']) == checkpoint['sha256']
    return receipt, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    option = parser.parse_args()
    directory = option.manifest.parent
    manifest = json.loads(option.manifest.read_text())
    assert manifest['schema'] == 'mcln-scanrefer-native-box-transfer-official-input-v1'
    training = Path(manifest['training_directory'])
    assert file_sha(training / 'receipt.json') == manifest['training_receipt_sha256']
    receipt, train_manifest = verify_training_endpoint(training)
    assert file_sha(training / 'independent_audit.json') == manifest['training_audit_sha256']
    audit = json.loads((training / 'independent_audit.json').read_text())
    assert audit['status'] == 'pass' and audit['eligible_for_fixed_terminal_formal_evaluation']
    assert audit['receipt_sha256'] == manifest['training_receipt_sha256']
    assert manifest['candidate_predeclared'] == 'gt_teacher_box_v99'
    assert manifest['arms'] == ['protected_v99', 'gt_only_v99', 'gt_teacher_box_v99']
    for name, digest in manifest['files'].items():
        assert file_sha(directory / name) == digest, name
    source = Path(train_manifest['model_source'])
    assert file_sha(source / 'local_visual_source_manifest.json') == train_manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'local_visual_source_manifest.json').read_text())['files'].items():
        assert file_sha(source / name) == digest, name
    for item in train_manifest['artifacts'].values():
        assert file_sha(item['path']) == item['sha256']
    result_directory = directory / 'result'
    result_directory.mkdir()
    os.chdir(str(source))
    sys.path.insert(0, str(source))

    import copy
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts'), str(source / 'scripts')]
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from models.rec_reranker import compute_query_ious
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scripts.scanrefer_joint_readout import JointRecReadout
    from scripts.audit_scanrefer_joint_readout_pair import assert_metadata_equal
    from scripts.scanrefer_rec_evaluation import rec_evaluation_view
    from scripts.scanrefer_data_contract import set_scanrefer_data_root, verify_scanrefer_superpoints

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.distributed.init_process_group(backend='nccl',
        init_method=(result_directory / 'distributed_init').as_uri(), world_size=1, rank=0)
    command = build_authoritative_command(result_directory / 'original_cli_reference')
    command = set_scanrefer_data_root(command, manifest['data_root'])
    data_inputs = verify_scanrefer_superpoints(manifest['data_root'], 'val', manifest['val_superpoint_files'])
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.data_root == manifest['data_root']
    assert args.eval and not args.eval_train and not args.debug
    assert args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
    assert args.batch_size == 12 and args.num_workers == 2
    assert args.butd and not args.butd_cls and not args.butd_gt
    assert not args.eval_use_selector_choice_scores
    tester = object.__new__(TrainTester)
    tester.logger = logging.getLogger('scanrefer-frozen-readout-official')
    train_loader, test_loader = tester.get_loaders(args)
    assert train_loader is None and len(test_loader.dataset) == 9508
    annotations = test_loader.dataset.annos
    identity = [(row['scan_id'], row['target_id'], row['utterance']) for row in annotations]
    write_json(result_directory / 'protocol.json', {'authoritative_base_command': command,
        'identities': identity, 'rows': 9508, 'workers': 2, 'batch_size': 12,
        'native_loader_and_worker_seeding': True, 'formal_checkpoint_arm': 'gt_teacher_box', 'formal_control_arm': 'gt_only', 'candidate_predeclared': 'gt_teacher_box',
        'data_root': args.data_root, 'superpoint_inputs': data_inputs,
        'historical_protected_rec_hits': [5572, 4797], 'scan_mask_paper_percent': [58.70, 50.70, 44.72]})
    protected_payload = torch.load(train_manifest['artifacts']['backbone']['path'], map_location='cpu')
    protected_state = {name[7:]: value for name, value in protected_payload['model'].items()}
    protected = TrainTester.get_model(args).cuda().eval()
    protected.load_state_dict(protected_state, strict=True)
    models = {'protected_v99': protected}
    old_artifacts = {name: torch.load(item['path'], map_location='cpu')
                     for name, item in train_manifest['artifacts'].items() if name != 'backbone'}
    artifacts = {'protected_v99': old_artifacts}
    states = {'protected_v99': protected_state}
    assert protected.decoder[-1].local_visual is None
    for arm, label in [('gt_only', 'gt_only_v99'), ('gt_teacher_box', 'gt_teacher_box_v99')]:
        trained = torch.load(receipt['checkpoints'][arm]['path'], map_location='cpu')
        assert trained['schema'] == 'mcln-scanrefer-native-box-head-state-v1'
        assert trained['arm'] == arm and trained['steps'] == 2482
        assert trained['manifest_sha256'] == receipt['manifest_sha256']
        protocol = json.loads((training / 'protocol.json').read_text())
        names = protocol['core_trainable_tensors']
        assert len(names) == 16 and sorted(trained['head_parameters']) == sorted(names)
        assert trained['pretrained_artifacts'] == train_manifest['artifacts']
        restored = dict(protected_state)
        for name, value in trained['head_parameters'].items():
            assert name.startswith(('prediction_heads.5.center_residual_head.', 'prediction_heads.5.size_pred_head.'))
            assert value.shape == protected_state[name].shape and value.dtype == protected_state[name].dtype
            assert torch.isfinite(value).all()
            restored[name] = value
        model = copy.deepcopy(protected)
        model.load_state_dict(restored, strict=True)
        assert all(torch.equal(value.cpu(), restored[name]) for name, value in model.state_dict().items())
        models[label], artifacts[label], states[label] = model, old_artifacts, restored
        del trained
    readouts = {arm: JointRecReadout(value).cuda().eval() for arm, value in artifacts.items()}
    evaluators = {arm: tester._build_grounding_evaluator(args, ['last_']) for arm in models}
    records = {arm: [] for arm in models}
    native_records = {arm: [] for arm in models}
    native_evaluators = {arm: copy.deepcopy(evaluator) for arm, evaluator in evaluators.items()}
    for evaluator in native_evaluators.values():
        evaluator.eval_use_selector_choice_scores = False
        evaluator.eval_use_rec_reranker_scores = False
        evaluator.eval_use_rec_geometry_reranker_scores = False
        evaluator.eval_use_rec_joint_box_mask = False
        assert not evaluator.filter_non_gt_boxes
    begin = time.time()
    with torch.no_grad():
        for batch_index, raw in enumerate(test_loader):
            batch = TrainTester._to_gpu(raw)
            inputs = TrainTester._get_inputs(batch)
            inputs['train'] = False
            roots = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
            root_valid = batch['box_label_mask'][:, :1].bool()
            points = [hashlib.sha256(cloud.cpu().numpy().tobytes()).hexdigest()
                      for cloud in inputs['point_clouds']]
            for arm, model in models.items():
                outputs = model(inputs)
                native_outputs = rec_evaluation_view(dict(outputs, **batch))
                native_evaluator = native_evaluators[arm]
                native_selected = []
                original_top = native_evaluator._position_top_indices

                def record_top(scores, valid, axis_mode, max_topk):
                    top = original_top(scores, valid, axis_mode, max_topk)
                    native_selected.append(int(top[0, 0]))
                    return top

                native_evaluator._position_top_indices = record_top
                native_evaluator.evaluate_bbox_by_pos_align(native_outputs, 'last_')
                native_evaluator._position_top_indices = original_top
                assert len(native_selected) == len(roots)
                native_boxes = torch.cat([native_outputs['last_center'], native_outputs['last_pred_size']], dim=-1)
                native_ious = compute_query_ious(native_boxes, roots, root_valid)
                for offset, query_index in enumerate(native_selected):
                    native_records[arm].append({'row_id': len(native_records[arm]),
                        'scan_id': raw['scan_ids'][offset], 'point_sha256': points[offset],
                        'query_index': query_index, 'rec_iou': float(native_ious[offset, query_index])})
                del native_outputs, native_boxes, native_ious
                readout = readouts[arm](outputs, inputs)
                runtime = readout['runtime']
                outputs.update(batch)
                outputs.update(runtime)
                evaluated = rec_evaluation_view(outputs)
                evaluator = evaluators[arm]
                evaluator.evaluate_bbox_by_pos_align(evaluated, 'last_')
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
                selected = runtime['rec_geometry_scores'].masked_fill(
                    ~runtime['rec_geometry_valid_mask'], -float('inf')).argmax(dim=1)
                ious = compute_query_ious(runtime['rec_geometry_boxes'], roots, root_valid)
                chosen = ious.gather(1, selected[:, None])[:, 0]
                for offset in range(len(roots)):
                    row_id = len(records[arm])
                    assert raw['scan_ids'][offset] == identity[row_id][0]
                    records[arm].append({'row_id': row_id, 'scan_id': raw['scan_ids'][offset],
                        'physical_space': raw['scan_ids'][offset].split('_')[0],
                        'point_sha256': points[offset], 'rec_iou': float(chosen[offset]),
                        'mask_iou': mask_ious[offset], 'selected_variant_position': int(selected[offset])})
                del outputs, readout, runtime, evaluated
            if batch_index == 0 or (batch_index + 1) % 128 == 0:
                elapsed = time.time() - begin
                count = len(records['gt_teacher_box_v99'])
                print('SCANREFER NATIVE BOX TRANSFER OFFICIAL', json.dumps({'rows': count, 'total': 9508,
                    'elapsed_seconds': elapsed, 'estimated_remaining_seconds': (9508 - count) * elapsed / count}), flush=True)
    metrics = {arm: row_metrics(rows) for arm, rows in records.items()}
    for arm, rows in records.items():
        assert len(rows) == 9508
        evaluator = evaluators[arm]
        for threshold, suffix in [(.25, '025'), (.5, '050')]:
            assert evaluator.gts[('last_', threshold, 1, 'bbs')] == 9508
            assert evaluator.dets[('last_', threshold, 1, 'bbs')] == metrics[arm]['rec_hits' + suffix]
            mask_key = 'overall_mask' if threshold == .25 else 'overall50_mask'
            assert evaluator.dets[mask_key] == metrics[arm]['mask_hits' + suffix]
        assert abs(evaluator.dets['mask_sem'] * 100. / 9508 - metrics[arm]['mask_miou']) < 1e-8
        assert all(torch.equal(value.detach().cpu(), states[arm][name])
                   for name, value in models[arm].state_dict().items())
        assert all(torch.equal(value.detach().cpu(), artifacts[arm][name]['model_state_dict'][key])
                   for name, scorer in readouts[arm].scorers.items() for key, value in scorer.state_dict().items())
    for candidate_arm in ['gt_only_v99', 'gt_teacher_box_v99']:
        for before, after in zip(records['protected_v99'], records[candidate_arm]):
            for key in ['row_id', 'scan_id', 'physical_space', 'point_sha256']:
                assert before[key] == after[key]
    write_json(result_directory / 'rows.json', records)
    native_metrics = {}
    for arm, rows in native_records.items():
        assert len(rows) == 9508
        native_metrics[arm] = {'rows': len(rows)}
        for threshold, suffix in [(.25, '025'), (.5, '050')]:
            hits = sum(row['rec_iou'] > threshold for row in rows)
            assert native_evaluators[arm].dets[('last_', threshold, 1, 'bbs')] == hits
            assert native_evaluators[arm].gts[('last_', threshold, 1, 'bbs')] == len(rows)
            native_metrics[arm]['rec_hits' + suffix] = hits
        for native_row, system_row in zip(rows, records[arm]):
            for key in ['row_id', 'scan_id', 'point_sha256']:
                assert native_row[key] == system_row[key]
    write_json(result_directory / 'native_rows.json', native_records)
    result = {'schema': 'mcln-scanrefer-native-box-transfer-official-v1', 'status': 'complete',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'formal_rows': 9508, 'optimizer_steps': 0, 'checkpoint_writes': 0,
        'metrics': metrics, 'native_rec_metrics': native_metrics,
        'native_rows_sha256': file_sha(result_directory / 'native_rows.json'),
        'promotion': promotion_check(metrics['protected_v99'], metrics['gt_teacher_box_v99']),
        'manifest_sha256': file_sha(option.manifest), 'trained_checkpoints': receipt['checkpoints'],
        'candidate_predeclared': 'gt_teacher_box_v99', 'gt_only_is_mechanism_control_not_validation_selected_candidate': True,
        'data_root': args.data_root, 'superpoint_inputs': data_inputs,
        'rows_sha256': file_sha(result_directory / 'rows.json'), 'all_model_states_unchanged': True,
        'native_evaluators_match_row_metrics': True,
        'evaluation_extent_policy': 'existing rec_candidate_adapter floor at 1e-6',
        'elapsed_seconds': time.time() - begin, 'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2}
    write_json(result_directory / 'receipt.json', result)
    print('SCANREFER NATIVE BOX TRANSFER OFFICIAL COMPLETE', json.dumps(result), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
