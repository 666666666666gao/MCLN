"""Read-only native REC for the fixed initial/control/appearance module-holdout endpoints."""
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
    parser.add_argument('--output', type=Path, required=True)
    option = parser.parse_args()
    option.manifest = option.manifest.resolve()
    training_root = option.manifest.parent
    directory = training_root
    manifest = json.loads(option.manifest.read_text())
    superpoint_files_sha256 = hashlib.sha256(
        json.dumps(manifest['superpoint_files'], sort_keys=True).encode()).hexdigest()
    source = Path(manifest['model_source'])
    assert file_sha(source / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'appearance_source_manifest.json').read_text())['files'].items():
        assert file_sha(source / name) == digest, name
    for name, digest in manifest['files'].items():
        assert file_sha(directory / name) == digest, name
    for name, item in manifest['artifacts'].items():
        assert file_sha(item['path']) == item['sha256'], name
    assert manifest['mode'] == 'train'
    assert file_sha(manifest['native_probe_receipt']) == manifest['native_probe_receipt_sha256']
    native_probe = json.loads(Path(manifest['native_probe_receipt']).read_text())
    assert native_probe['status'] == 'pass' and len(native_probe['disposable_steps']) == 2
    assert native_probe['original_state_tensors_preserved'] == 1144
    assert native_probe['source_manifest_sha256'] == manifest['source_manifest_sha256']
    cache_root = Path(manifest['cache_root'])
    assert file_sha(cache_root / 'receipt.json') == manifest['cache_receipt_sha256']
    cache_receipt = json.loads((cache_root / 'receipt.json').read_text())
    assert cache_receipt['status'] == 'complete' and cache_receipt['scene_count'] == 562
    assert file_sha(cache_root / 'scenes.jsonl') == cache_receipt['scenes_sha256']
    assert file_sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    split = json.loads(Path(manifest['split_protocol']).read_text())
    partitions = dict(split['row_ids'])
    assert manifest['steps_per_arm'] == math.ceil(len(partitions['fit']) / 12)
    assert manifest['epochs'] == 1 and manifest['batch_size'] == 12
    assert manifest['core_learning_rate'] == 1e-6 and manifest['appearance_learning_rate'] == 1e-4
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
    from models.pretrained_object_appearance import PretrainedObjectAppearance
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
    assert (training_root / 'audit_queue.exit').read_text().strip() == '0'
    audit = json.loads((training_root / 'audit.json').read_text())
    terminal = json.loads((training_root / 'receipt.json').read_text())
    assert terminal['status'] == 'complete' and terminal['steps_per_arm'] == 2482
    assert terminal['manifest_sha256'] == file_sha(option.manifest)
    baseline_rows = json.loads((training_root / 'baseline_rows.json').read_text())['control']
    assert file_sha(training_root / 'baseline_rows.json') == terminal['baseline_rows_sha256']
    directory = option.output.resolve()
    directory.mkdir(exist_ok=False)
    models, expected_states = {}, {}
    for arm in ['initial', 'control', 'appearance']:
        model = TrainTester.get_model(args).cuda().eval().requires_grad_(False)
        if arm == 'initial':
            state = initial
        else:
            checkpoint = terminal['checkpoints'][arm]
            assert file_sha(checkpoint['path']) == checkpoint['sha256']
            payload = torch.load(checkpoint['path'], map_location='cpu')
            assert payload['arm'] == arm and payload['steps'] == 2482
            assert payload['manifest_sha256'] == file_sha(option.manifest)
            state = payload['model']
        if arm == 'appearance':
            model.object_appearance = PretrainedObjectAppearance().cuda().eval().requires_grad_(False)
        model.load_state_dict(state, strict=True)
        assert model.decoder[-1].local_visual is None
        models[arm], expected_states[arm] = model, state
    assert not args.eval_use_selector_choice_scores
    cache_records = {v['scene_id']: v for v in map(json.loads, (cache_root / 'scenes.jsonl').read_text().splitlines())}
    feature_cache = {}
    for scene_id, record in cache_records.items():
        path = cache_root / record['file']
        assert file_sha(path) == record['file_sha256']
        with np.load(str(path)) as stored:
            feature_cache[scene_id] = {k: stored[k] for k in ['features', 'available', 'boxes']}

    def make_inputs(batch):
        inputs = TrainTester._get_inputs(batch)
        inputs['det_visual_features'] = batch['object_visual_features']
        inputs['det_visual_available'] = batch['object_visual_available']
        return inputs

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
            scene_id = self.annos[index]['scan_id']
            stored = feature_cache[scene_id]
            count = len(stored['boxes'])
            assert np.array_equal(result['all_detected_boxes'][:count], stored['boxes'])
            assert int(result['all_detected_bbox_label_mask'].sum()) == count
            assert hashlib.sha256(result['point_clouds'].tobytes()).hexdigest() == cache_records[scene_id]['native_point_sha256']
            size = len(result['all_detected_bbox_label_mask'])
            result['object_visual_features'] = np.zeros((size, 1280), dtype=np.float32)
            result['object_visual_available'] = np.zeros(size, dtype=np.bool_)
            result['object_visual_features'][:count] = stored['features']
            result['object_visual_available'][:count] = stored['available']
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
    def loader(part, seed, shuffle):
        return torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, partitions[part]),
            batch_size=12, shuffle=shuffle, num_workers=0,
            generator=torch.Generator().manual_seed(seed))

    tester = object.__new__(TrainTester)
    tester.logger = logging.getLogger('scanrefer-object-appearance-pair')

    evaluators = {}
    records = {arm: [] for arm in models}
    for arm in models:
        evaluator = tester._build_grounding_evaluator(args, ['last_'])
        evaluator.eval_use_selector_choice_scores = False
        evaluator.eval_use_rec_geometry_reranker_scores = False
        evaluator.eval_use_rec_reranker_scores = False
        evaluator.eval_use_rec_joint_box_mask = False
        assert evaluator.only_root and not evaluator.filter_non_gt_boxes
        evaluators[arm] = evaluator
    seed_everything(1000)
    begin = time.time()
    with torch.no_grad():
        for batch_index, raw in enumerate(loader('holdout', 1000, False)):
            batch = TrainTester._to_gpu(raw)
            inputs = make_inputs(batch)
            inputs['train'] = False
            roots = torch.cat([batch['center_label'][:, :1], batch['size_gts'][:, :1]], dim=-1)
            point_hashes = [hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest() for p in inputs['point_clouds']]
            for arm, model in models.items():
                outputs = model(inputs)
                evaluated = rec_evaluation_view(dict(outputs, **batch))
                evaluator = evaluators[arm]
                captured = []
                original = evaluator._position_top_indices

                def capture(scores, valid, axis_mode, max_topk):
                    result = original(scores, valid, axis_mode, max_topk)
                    assert axis_mode == 'default_query_axis' and bool(valid.all())
                    captured.append(int(result[0, 0]))
                    return result

                evaluator._position_top_indices = capture
                evaluator.evaluate_bbox_by_pos_align(evaluated, 'last_')
                evaluator._position_top_indices = original
                assert len(captured) == len(roots)
                boxes = torch.cat([evaluated['last_center'], evaluated['last_pred_size']], dim=-1)
                all_ious = compute_query_ious(boxes, roots, batch['box_label_mask'][:, :1].bool())
                for index, row_id in enumerate(raw['local_training_id'].tolist()):
                    old = baseline_rows[len(records[arm])]
                    assert (row_id, raw['scan_ids'][index], point_hashes[index]) == (old['row_id'], old['scan_id'], old['point_sha256'])
                    selected = captured[index]
                    records[arm].append(dict(row_id=row_id, scan_id=old['scan_id'], physical_space=old['physical_space'],
                        point_sha256=point_hashes[index], selected_query=selected,
                        selected_box=boxes[index, selected].cpu().tolist(), root_box=roots[index, 0].cpu().tolist(),
                        rec_iou=float(all_ious[index, selected])))
                del outputs, evaluated, boxes, all_ious
            if batch_index == 0 or (batch_index + 1) % 128 == 0:
                print('NATIVE APPEARANCE EVAL', json.dumps(dict(rows=len(records['appearance']), total=6887,
                      elapsed_seconds=time.time()-begin)), flush=True)
    metrics = {}
    for arm, rows in records.items():
        assert [r['row_id'] for r in rows] == partitions['holdout']
        metrics[arm] = {}
        for threshold, suffix in [(.25, '025'), (.5, '050')]:
            hits = sum(r['rec_iou'] > threshold for r in rows)
            key = ('last_', threshold, 1, 'bbs')
            assert evaluators[arm].gts[key] == 6887 and evaluators[arm].dets[key] == hits
            metrics[arm]['hits'+suffix] = hits
        assert all(torch.equal(v.cpu(), expected_states[arm][k]) for k, v in models[arm].state_dict().items())
    effects = {ref: {str(t): paired_effect(records[ref], records['appearance'], 'rec_iou', t)
                    for t in [.25, .5]} for ref in ['initial', 'control']}
    for item in manifest['artifacts'].values():
        assert file_sha(item['path']) == item['sha256']
    for item in terminal['checkpoints'].values():
        assert file_sha(item['path']) == item['sha256']
    for name, digest in json.loads((source / 'appearance_source_manifest.json').read_text())['files'].items():
        assert file_sha(source / name) == digest, name
    verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['superpoint_files']['train'])
    write_json(directory / 'rows.json', records)
    receipt = dict(schema='mcln-object-appearance-native-endpoint-rec-v1', status='complete',
        time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        rows_per_arm=6887, formal_rows=0, optimizer_steps=0, checkpoint_writes=0,
        previous_pretraining_has_seen_development_holdout=True, metrics=metrics, appearance_effects=effects,
        rows_sha256=file_sha(directory / 'rows.json'), training_manifest_sha256=file_sha(option.manifest),
        training_receipt_sha256=file_sha(training_root / 'receipt.json'), audit_sha256=file_sha(training_root / 'audit.json'),
        script_sha256=file_sha(__file__), checkpoints=terminal['checkpoints'],
        state_and_files_unchanged=True, native_selection='actual evaluator default query axis; no Parent/Geometry/V99',
        extent_policy='existing evaluation floor at 1e-6', elapsed_seconds=time.time()-begin,
        max_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)
    write_json(directory / 'receipt.json', receipt)
    print('NATIVE APPEARANCE COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
