"""Isolated stage diagnostic; never replaces the fixed formal evaluation.

Derived from the existing formal evaluator with the same forward and selection paths.
A completed, independently audited formal run must be supplied as its reference.
"""

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


from scripts.evaluate_scanrefer_local_visual_official import (
    file_sha, write_json, row_metrics, verify_training_endpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    option = parser.parse_args()
    directory = option.manifest.parent
    manifest = json.loads(option.manifest.read_text())
    assert manifest['schema'] == 'mcln-scanrefer-stage-diagnostic-input-v1'
    reference_directory = Path(manifest['reference_formal_directory'])
    reference_names = {'input_manifest.json', 'controller.exit', 'result/receipt.json',
                       'result/rows.json', 'result/native_rows.json',
                       'result/protocol.json', 'result/independent_audit.json'}
    assert set(manifest['reference_files']) == reference_names
    for name, digest in manifest['reference_files'].items():
        assert file_sha(reference_directory / name) == digest, name
    assert (reference_directory / 'controller.exit').read_text().strip() == '0'
    reference_receipt = json.loads((reference_directory / 'result/receipt.json').read_text())
    reference_audit = json.loads((reference_directory / 'result/independent_audit.json').read_text())
    reference_input = json.loads((reference_directory / 'input_manifest.json').read_text())
    assert reference_receipt['status'] == 'complete' and reference_receipt['formal_rows'] == 9508
    assert reference_audit['integrity_pass']
    assert reference_audit['receipt_sha256'] == manifest['reference_files']['result/receipt.json']
    for name in ('training_directory', 'training_receipt_sha256', 'data_root', 'val_superpoint_files'):
        assert manifest[name] == reference_input[name], name
    reference_native = json.loads((reference_directory / 'result/native_rows.json').read_text())
    reference_final = json.loads((reference_directory / 'result/rows.json').read_text())
    reference_protocol = json.loads((reference_directory / 'result/protocol.json').read_text())
    training = Path(manifest['training_directory'])
    assert file_sha(training / 'receipt.json') == manifest['training_receipt_sha256']
    receipt, train_manifest = verify_training_endpoint(training)
    for name, digest in manifest['files'].items():
        assert file_sha(directory / name) == digest, name
    source = Path(train_manifest['model_source'])
    assert file_sha(source / 'local_visual_source_manifest.json') == train_manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'local_visual_source_manifest.json').read_text())['files'].items():
        assert file_sha(source / name) == digest, name
    for item in train_manifest['artifacts'].values():
        assert file_sha(item['path']) == item['sha256']
    result_directory = directory / 'diagnostic_result'
    result_directory.mkdir()
    os.chdir(str(source))
    sys.path.insert(0, str(source))

    import copy
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(directory / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from models.rec_reranker import compute_query_ious
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scripts.scanrefer_joint_readout import JointRecReadout
    from models.candidate_local_visual import CandidateLocalVisual
    from scripts.scanrefer_rec_evaluation import rec_evaluation_view
    from scripts.trace_scanrefer_readout_stages import STAGES, trace_readout_stages
    from scripts.scanrefer_stage_diagnostics import FeatureMoments, summarize_stages
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
    tester.logger = logging.getLogger('scanrefer-stage-diagnostic')
    train_loader, test_loader = tester.get_loaders(args)
    assert train_loader is None and len(test_loader.dataset) == 9508
    annotations = test_loader.dataset.annos
    identity = [(row['scan_id'], row['target_id'], row['utterance']) for row in annotations]
    assert [list(value) for value in identity] == reference_protocol['identities']
    write_json(result_directory / 'protocol.json', {'authoritative_base_command': command,
        'identities': identity, 'rows': 9508, 'workers': 2, 'batch_size': 12,
        'native_loader_and_worker_seeding': True, 'checkpoint_arm': 'local', 'formal_rows': 0, 'purpose': 'stage_diagnostic',
        'data_root': args.data_root, 'superpoint_inputs': data_inputs,
        'historical_protected_rec_hits': [5572, 4797], 'scan_mask_paper_percent': [58.70, 50.70, 44.72]})
    protected_payload = torch.load(train_manifest['artifacts']['backbone']['path'], map_location='cpu')
    protected_state = {name[7:]: value for name, value in protected_payload['model'].items()}
    protected = TrainTester.get_model(args).cuda().eval()
    protected.load_state_dict(protected_state, strict=True)
    models = {'protected_v99': protected, 'local_v99': copy.deepcopy(protected)}
    trained = torch.load(receipt['checkpoints']['local']['path'], map_location='cpu')
    assert trained['schema'] == 'mcln-scanrefer-local-visual-trained-state-v1'
    assert trained['arm'] == 'local' and trained['steps'] == 2482
    assert trained['manifest_sha256'] == receipt['manifest_sha256']
    assert trained['model_config'] == {'candidate_local_visual': True}
    models['local_v99'].decoder[-1].local_visual = CandidateLocalVisual().cuda().eval()
    models['local_v99'].load_state_dict(trained['model'], strict=True)
    old_artifacts = {name: torch.load(item['path'], map_location='cpu')
                     for name, item in train_manifest['artifacts'].items() if name != 'backbone'}
    artifacts = {'protected_v99': old_artifacts, 'local_v99': trained['readout']}
    readouts = {arm: JointRecReadout(value).cuda().eval() for arm, value in artifacts.items()}
    states = {'protected_v99': protected_state, 'local_v99': trained['model']}
    trace_records = {arm: [] for arm in models}
    feature_moments = {arm: {'parent': FeatureMoments(152), 'geometry': FeatureMoments(179)}
                       for arm in models}
    feature_hooks = [readouts[arm].scorers[name].register_forward_pre_hook(moment)
                     for arm, moments in feature_moments.items() for name, moment in moments.items()]
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
                del native_outputs, native_ious
                readout = readouts[arm](outputs, inputs)
                trace = trace_readout_stages(native_boxes,
                    torch.tensor(native_selected, dtype=torch.long, device=native_boxes.device),
                    readout, readouts[arm].metadata)
                # Root GT is used only after the GT-free stage selections are fixed.
                stage_ious = compute_query_ious(trace['boxes'], roots, root_valid)
                candidate = readout['parent']['candidate_batch']
                top16_ious = compute_query_ious(candidate['boxes'], roots, root_valid)
                top16_oracle = top16_ious.masked_fill(~candidate['valid_mask'], -1.).max(dim=1).values
                for offset in range(len(roots)):
                    row_id = len(trace_records[arm])
                    stages = {name: {'query_index': int(trace['query_indices'][offset, stage]),
                                    'variant_index': int(trace['variant_indices'][offset, stage]),
                                    'box': trace['boxes'][offset, stage].cpu().tolist(),
                                    'rec_iou': float(stage_ious[offset, stage])}
                              for stage, name in enumerate(STAGES)}
                    trace_records[arm].append({
                        'row_id': row_id, 'scan_id': raw['scan_ids'][offset],
                        'target_id': identity[row_id][1], 'utterance': identity[row_id][2],
                        'point_sha256': points[offset], 'root_box': roots[offset, 0].cpu().tolist(),
                        'stages': stages, 'top16_oracle_iou': float(top16_oracle[offset]),
                        'top16_query_indices': trace['top16_query_indices'][offset].cpu().tolist(),
                        'top16_valid': trace['top16_valid'][offset].cpu().tolist(),
                        'top16_boxes': candidate['boxes'][offset].detach().cpu().tolist(),
                        'effective_variant_valid': trace['effective_variant_valid'][offset].cpu().tolist(),
                        'deployed_variant_valid': trace['deployed_variant_valid'][offset].cpu().tolist(),
                        'geometry_flat_index': int(trace['geometry_flat_indices'][offset]),
                        'proposal_flat_index': int(trace['proposal_flat_indices'][offset]),
                        'final_flat_index': int(trace['final_flat_indices'][offset]),
                        'pareto_pass': bool(trace['pareto_pass'][offset]),
                        'predicted_head_gain': trace['predicted_head_gain'][offset].cpu().tolist(),
                        'predicted_aggregate_gain': float(trace['predicted_aggregate_gain'][offset])})
                del native_boxes, trace, stage_ious, top16_ious, top16_oracle
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
                count = len(records['local_v99'])
                print('SCANREFER STAGE DIAGNOSTIC', json.dumps({'rows': count, 'total': 9508,
                    'elapsed_seconds': elapsed, 'estimated_remaining_seconds': (9508 - count) * elapsed / count}), flush=True)
    for handle in feature_hooks:
        handle.remove()
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
    for before, after in zip(records['protected_v99'], records['local_v99']):
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
    stage_summary = summarize_stages(trace_records, reference_native, reference_final)
    for arm in models:
        for suffix in ('025', '050'):
            assert stage_summary['arms'][arm]['metrics']['native']['hits' + suffix] == native_metrics[arm]['rec_hits' + suffix]
            assert stage_summary['arms'][arm]['metrics']['v99_final']['hits' + suffix] == metrics[arm]['rec_hits' + suffix]
    write_json(result_directory / 'stage_rows.json', trace_records)
    write_json(result_directory / 'stage_summary.json', stage_summary)
    write_json(result_directory / 'normalized_features.json', {
        arm: {name: moment.export() for name, moment in moments.items()}
        for arm, moments in feature_moments.items()})
    result = {'schema': 'mcln-scanrefer-stage-diagnostic-result-v1', 'status': 'complete',
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'formal_rows': 0, 'diagnostic_rows': 9508, 'optimizer_steps': 0, 'checkpoint_writes': 0,
        'metrics': metrics, 'native_rec_metrics': native_metrics,
        'native_rows_sha256': file_sha(result_directory / 'native_rows.json'),
        'used_for_promotion': False, 'selection_uses_gt': False,
        'all_forward_final_scores_verified_equal': True,
        'reference_formal_directory': str(reference_directory),
        'reference_files': manifest['reference_files'],
        'stage_rows_sha256': file_sha(result_directory / 'stage_rows.json'),
        'stage_summary_sha256': file_sha(result_directory / 'stage_summary.json'),
        'normalized_features_sha256': file_sha(result_directory / 'normalized_features.json'),
        'manifest_sha256': file_sha(option.manifest), 'trained_checkpoint': receipt['checkpoints']['local'],
        'data_root': args.data_root, 'superpoint_inputs': data_inputs,
        'rows_sha256': file_sha(result_directory / 'rows.json'), 'all_model_states_unchanged': True,
        'native_evaluators_match_row_metrics': True,
        'evaluation_extent_policy': 'existing rec_candidate_adapter floor at 1e-6',
        'elapsed_seconds': time.time() - begin, 'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2}
    write_json(result_directory / 'receipt.json', result)
    print('SCANREFER STAGE DIAGNOSTIC COMPLETE', json.dumps(result), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
