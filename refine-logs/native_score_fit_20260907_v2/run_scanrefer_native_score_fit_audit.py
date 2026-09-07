"""Read-only comparison of actual evaluator and adapter scores on fixed fit rows."""
import argparse
import datetime
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    option = parser.parse_args()
    directory = option.manifest.resolve().parent
    manifest = json.loads(option.manifest.read_text())
    assert manifest['schema'] == 'mcln-native-score-fit-audit-v1'
    source_manifest = Path(manifest['source_manifest'])
    assert sha(source_manifest) == manifest['source_manifest_sha256']
    source = Path(manifest['model_source'])
    files = json.loads(source_manifest.read_text())['files']
    for name, digest in files.items():
        assert sha(source / name) == digest, name
    for name, digest in manifest['files'].items():
        assert sha(directory / name) == digest, name
    for name, item in manifest['artifacts'].items():
        assert sha(item['path']) == item['sha256'], name
    assert sha(manifest['historical_rows']) == manifest['historical_rows_sha256']
    historical = json.loads(Path(manifest['historical_rows']).read_text())
    assert len(historical) == manifest['rows'] == 512
    assert [r['row_id'] for r in historical] == manifest['selected_row_ids']
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import copy
    import numpy as np
    import torch
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from models.rec_candidate_adapter import build_full_rec_query_state, compact_rec_query_state
    from models.source_choice_adapter import compute_default_source_scores
    from models.rec_reranker import select_candidate_indices, compute_query_ious
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scanrefer_data_contract import set_scanrefer_data_root, verify_scanrefer_superpoints
    from scanrefer_rec_evaluation import rec_evaluation_view

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    command = set_scanrefer_data_root(build_authoritative_command(directory / 'unused_output'), manifest['data_root'])
    data = verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['train_superpoint_files'])
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.dataset == ['scanrefer'] and args.butd and not args.butd_cls and not args.butd_gt
    assert not args.eval_use_selector_choice_scores and not args.native_mask_geometry_supervision
    assert args.checkpoint_path == manifest['artifacts']['backbone']['path']
    initial = {k[7:]: v for k, v in torch.load(args.checkpoint_path, map_location='cpu')['model'].items()}
    model = TrainTester.get_model(args).cuda().eval().requires_grad_(False)
    model.load_state_dict(initial, strict=True)
    assert len(initial) == 1144 and model.decoder[-1].local_visual is None
    parent = torch.load(manifest['artifacts']['parent']['path'], map_location='cpu')
    topk = parent['candidate_rule']['topk_per_source']
    maximum = parent['candidate_rule']['max_candidates']
    assert topk == 8 and maximum == 16
    protocol = {}

    class AuditDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            fit = []
            for index, row in enumerate(annos):
                row['_score_audit_id'] = index
                code = (manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()
                if int(hashlib.sha256(code).hexdigest()[:8], 16) % 5 != 0:
                    fit.append(index)
            selected = [fit[i * len(fit) // 512] for i in range(512)]
            assert selected == manifest['selected_row_ids']
            scenes = {annos[i]['scan_id'] for i in selected}
            protocol.update(rows=512, fit_rows=len(fit), selected_row_ids=selected,
                            selected_scans=sorted(scenes), selection='historical equally spaced fit positions')
            annos[:] = [r for r in annos if r['scan_id'] in scenes]
            super()._scene_graph_parse(annos)

    dataset = AuditDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=args.data_root, use_color=args.use_color, use_height=args.use_height,
        use_multiview=args.use_multiview, detect_intermediate=args.detect_intermediate,
        butd=args.butd, butd_gt=args.butd_gt, butd_cls=args.butd_cls,
        augment_det=False, skip_missing_superpoints=args.skip_missing_superpoints)
    by_id = {r['_score_audit_id']: r for r in dataset.annos}
    dataset.annos = [by_id[i] for i in protocol['selected_row_ids']]
    dataset.augment = False
    write_json(directory / 'protocol.json', protocol)
    loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False, num_workers=0,
                                         generator=torch.Generator().manual_seed(0))
    tester = object.__new__(TrainTester)
    tester.logger = logging.getLogger('native-score-fit-audit')
    evaluator = tester._build_grounding_evaluator(args, ['last_'])
    evaluator.eval_use_selector_choice_scores = False
    evaluator.eval_use_rec_geometry_reranker_scores = False
    evaluator.eval_use_rec_reranker_scores = False
    evaluator.eval_use_rec_joint_box_mask = False
    assert evaluator.only_root and not evaluator.filter_non_gt_boxes
    records = []
    start = time.time()
    with torch.no_grad():
        for raw in loader:
            batch = TrainTester._to_gpu(raw)
            inputs = TrainTester._get_inputs(batch)
            inputs['train'] = False
            outputs = model(inputs)
            evaluated = rec_evaluation_view(dict(outputs, **batch))
            captured = []
            original = evaluator._position_top_indices

            def capture(scores, valid, axis_mode, max_topk):
                result = original(scores, valid, axis_mode, max_topk)
                captured.append((scores.detach().clone(), valid.detach().clone(), int(result[0, 0]), axis_mode))
                return result

            evaluator._position_top_indices = capture
            evaluator.evaluate_bbox_by_pos_align(evaluated, 'last_')
            evaluator._position_top_indices = original
            assert len(captured) == len(inputs['text'])
            native_scores = torch.cat([item[0] for item in captured])
            native_valid = torch.stack([item[1].reshape(-1) for item in captured])
            assert native_scores.shape[1] == 256 and native_valid.all()
            source_scores = compute_default_source_scores(outputs, inputs)
            max_error = (native_scores - source_scores).abs().max()
            assert max_error < 2e-6
            full = build_full_rec_query_state(outputs, inputs)
            compact = compact_rec_query_state(full, topk_per_source=topk, max_candidates=maximum)
            counter_indices, counter_valid = select_candidate_indices(
                native_scores, full['contrastive_scores'], topk_per_source=topk, max_candidates=maximum)
            # All selections above are computed without root coordinates or IoU.
            roots = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
            ious = compute_query_ious(full['boxes'], roots, batch['box_label_mask'][:, :1].bool())
            features = full['features']
            names = full['feature_names']
            probabilities = outputs['last_sem_cls_scores'].float().softmax(-1)
            main_map = inputs['positive_map'][:, 0, :probabilities.shape[-1]]
            binary_main_scores = (probabilities * (main_map > 0).float()[:, None]).sum(-1)
            for i in range(len(roots)):
                old = historical[len(records)]
                point_hash = hashlib.sha256(inputs['point_clouds'][i].cpu().numpy().tobytes()).hexdigest()
                assert raw['scan_ids'][i] == old['scan_id']
                nonzero = (main_map[i] != 0).nonzero().reshape(-1)
                component = {n: features[i, :, names.index('score_' + n)].cpu().tolist()
                             for n in ['main', 'modifier', 'pronoun', 'relation', 'other']}
                old_boxes = full['boxes'].new_tensor(old['native_boxes'])
                row = {'row_id': old['row_id'], 'scan_id': old['scan_id'], 'point_sha256': point_hash,
                       'point_sha_matches_historical': point_hash == old['point_sha256'],
                       'main_map_indices': nonzero.cpu().tolist(), 'main_map_values': main_map[i, nonzero].cpu().tolist(),
                       'native_scores': native_scores[i].cpu().tolist(), 'source_choice_scores': source_scores[i].cpu().tolist(),
                       'adapter_scores': full['default_scores'][i].cpu().tolist(),
                       'contrastive_scores': full['contrastive_scores'][i].cpu().tolist(),
                       'binary_main_scores': binary_main_scores[i].cpu().tolist(), 'components': component,
                       'native_query': captured[i][2], 'native_axis_mode': captured[i][3],
                       'adapter_query': int(full['default_scores'][i].argsort(descending=True)[0]),
                       'original_candidates': compact['query_indices'][i, compact['valid_mask'][i]].cpu().tolist(),
                       'native_score_counterfactual_candidates': counter_indices[i, counter_valid[i]].cpu().tolist(),
                       'boxes': full['boxes'][i].cpu().tolist(), 'root_box': roots[i, 0].cpu().tolist(),
                       'query_ious': ious[i].cpu().tolist(),
                       'historical_box_max_abs_difference': float((full['boxes'][i] - old_boxes).abs().max()),
                       'historical_adapter_score_max_abs_difference': float((full['default_scores'][i] - full['default_scores'].new_tensor(old['native_scores'])).abs().max())}
                records.append(row)
            print('NATIVE SCORE FIT AUDIT', json.dumps({'rows': len(records), 'total': 512,
                  'elapsed_seconds': time.time() - start, 'native_source_max_error': float(max_error)}), flush=True)
            del outputs, evaluated, full, compact, inputs, batch, probabilities, captured, features
    assert len(records) == 512 and [r['row_id'] for r in records] == manifest['selected_row_ids']
    assert all(torch.equal(value.cpu(), initial[name]) for name, value in model.state_dict().items())
    for name, item in manifest['artifacts'].items():
        assert sha(item['path']) == item['sha256'], name
    for name, digest in files.items():
        assert sha(source / name) == digest, name
    verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['train_superpoint_files'])
    for threshold in [.25, .5]:
        key = ('last_', threshold, 1, 'bbs')
        assert evaluator.gts[key] == 512
        assert evaluator.dets[key] == sum(r['query_ious'][r['native_query']] > threshold for r in records)
    shards = []
    for start_row in range(0, 512, 128):
        filename = 'rows_{:03d}_{:03d}.json.gz'.format(start_row, start_row + 127)
        raw = json.dumps(records[start_row:start_row+128], sort_keys=True, allow_nan=False).encode()
        with gzip.GzipFile(filename=str(directory / filename), mode='wb', mtime=0) as stream:
            stream.write(raw)
        shards.append({'file': filename, 'rows': 128, 'sha256': sha(directory / filename),
                       'uncompressed_sha256': hashlib.sha256(raw).hexdigest()})
    receipt = {'status': 'complete', 'schema': 'mcln-native-score-fit-result-v1',
               'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               'rows': 512, 'scope': 'Previously seen fit rows;diagnostic counterfactual not deployed result',
               'manifest_sha256': sha(option.manifest), 'shards': shards, 'superpoint_inputs': data,
               'model_and_source_unchanged': True, 'model_tensors': len(initial),
               'optimizer_steps': 0, 'checkpoint_writes': 0, 'formal_rows': 0,
               'max_gpu_mib': torch.cuda.max_memory_allocated() / 1024**2,
               'elapsed_seconds': time.time() - start,
               'map_count_histogram': {str(n): sum(len(r['main_map_values']) == n for r in records)
                                       for n in sorted({len(r['main_map_values']) for r in records})},
               'top1_disagreements': sum(r['native_query'] != r['adapter_query'] for r in records),
               'native_top1_excluded_original': sum(r['native_query'] not in r['original_candidates'] for r in records),
               'native_score_counterfactual_membership_changes': sum(set(r['original_candidates']) != set(r['native_score_counterfactual_candidates']) for r in records),
               'historical_point_mismatches': sum(not r['point_sha_matches_historical'] for r in records)}
    write_json(directory / 'receipt.json', receipt)
    print('NATIVE SCORE FIT COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
