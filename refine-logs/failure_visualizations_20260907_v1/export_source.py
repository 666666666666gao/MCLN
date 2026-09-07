import ast
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

SOURCE = Path('/root/autodl-tmp/mcln_scanrefer_local_visual_preflight_20260906_v2/model_source')
sys.path.insert(0, str(SOURCE))
from src.visual_data_handlers import Scan

OUT = Path('/root/autodl-tmp/mcln_failure_visualizations_20260907_v1')
OUT.mkdir(exist_ok=True)
DATA = Path('/root/autodl-tmp/DATA_ROOT')
STAGE = Path('/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_20260907_v1')
CACHE = DATA / 'output/network_v99_baseline_gt/nr3d/analysis/76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1/candidate_cache/val'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def iou(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    lo = np.maximum(a[..., :3] - a[..., 3:] / 2, b[:3] - b[3:] / 2)
    hi = np.minimum(a[..., :3] + a[..., 3:] / 2, b[:3] + b[3:] / 2)
    inter = np.maximum(hi - lo, 0).prod(-1)
    return inter / (a[..., 3:].prod(-1) + b[3:].prod(-1) - inter)

def root_box(scene, target_id):
    bounds = scene.get_object_bbox(target_id).reshape(-1)
    return np.concatenate([(bounds[:3] + bounds[3:]) * .5, bounds[3:] - bounds[:3]])

with (DATA / 'val_v3scans.pkl').open('rb') as f:
    assert pickle.load(f) == 1
    scans = pickle.load(f)
print('Loaded actual validation scenes:', len(scans), flush=True)
scan_records = json.loads((STAGE / 'diagnostic_result/stage_rows.json').read_text())['protected_v99']
scan_annotations = json.loads((DATA / 'scanrefer/ScanRefer_filtered_val.json').read_text())
lookup = {}
for a in scan_annotations:
    lookup[(a['scene_id'], int(a['object_id']), ' '.join(a['token']))] = a
nr_scenes = set(ast.literal_eval((SOURCE / 'data/meta_data/nr3d_test_scans.txt').read_text()))
with (DATA / 'refer_it_3d/nr3d.csv').open() as f:
    nr_annotations = [r for r in csv.DictReader(f) if r['scan_id'] in nr_scenes and r['correct_guess'].lower() == 'true']
assert len(nr_annotations) == 7899
cache_manifest = json.loads((CACHE / 'manifest.json').read_text())
assert cache_manifest['checkpoint_sha256'] == '76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1'
nr_records = []
for shard in cache_manifest['shards']:
    for r in torch.load(CACHE / shard, map_location='cpu')['rows']:
        a = nr_annotations[r['dataset_index']]
        assert r['scan_id'] == a['scan_id'] and r['target_id'] == int(a['target_id'])
        index = int(torch.nonzero(r['query_indices'] == r['default_top1_query_index']).view(-1).item())
        assert bool(r['valid_mask'][index])
        nr_records.append({'row_id': r['dataset_index'], 'scan_id': r['scan_id'], 'target_id': r['target_id'],
            'utterance': a['utterance'], 'original_annotation': a, 'query_index': r['default_top1_query_index'],
            'pred_box': r['boxes'][index].tolist(), 'rec_iou': float(r['candidate_ious'][index]),
            'top16_oracle_iou': float(r['candidate_ious'].masked_fill(~r['valid_mask'], -1).max()),
            'cached_boxes': r['boxes'].numpy(), 'cached_ious': r['candidate_ious'].numpy(),
            'valid_mask': r['valid_mask'].numpy(), 'shard': shard})
assert len(nr_records) == len(nr_annotations)

records = {'ScanRefer': [], 'Nr3D': []}
for r in scan_records:
    s = r['stages']['v99_final']
    records['ScanRefer'].append(dict(r, pred_box=s['box'], rec_iou=s['rec_iou'], query_index=s['query_index']))
records['Nr3D'] = nr_records
selected = []
for dataset, rows in records.items():
    used = set()
    for group in ['selection', 'strict_overlap', 'top16_coverage']:
        pool = []
        for r in rows:
            if r['scan_id'] in used:
                continue
            scene = scans[r['scan_id']]
            gt = root_box(scene, r['target_id'])
            label = scene.get_object_instance_label(r['target_id'])
            count = len(scene.three_d_objects[r['target_id']]['points'])
            if label in ['wall', 'floor', 'ceiling'] or min(r['pred_box'][3:]) <= .025:
                continue
            separation = np.linalg.norm(np.asarray(r['pred_box'][:3]) - gt[:3])
            if group == 'selection':
                good = r['rec_iou'] < .01 and r['top16_oracle_iou'] > .65 and count >= 200 and .4 < separation < 3.5 and .1 < np.prod(gt[3:]) < 5
            elif group == 'strict_overlap':
                good = .27 < r['rec_iou'] < .46 and count >= 200 and .1 < np.prod(gt[3:]) < 5
            else:
                good = r['rec_iou'] < .25 and r['top16_oracle_iou'] < .25 and 40 <= count <= 227 and np.prod(gt[3:]) < .4
            if good:
                pool.append(r)
        assert pool, (dataset, group)
        r = sorted(pool, key=lambda v: v['row_id'])[0]
        used.add(r['scan_id'])
        scene = scans[r['scan_id']]
        gt = root_box(scene, r['target_id']).astype(np.float32)
        pred = np.asarray(r['pred_box'], dtype=np.float32)
        actual_iou = float(iou(pred, gt))
        assert abs(actual_iou - r['rec_iou']) < 1e-5, (dataset, r['row_id'], actual_iou, r['rec_iou'])
        cloud_input = np.concatenate([scene.pc, scene.color - np.array([109.8, 97.2, 83.8]) / 256], axis=1).astype(np.float32)
        cloud_sha = hashlib.sha256(cloud_input.tobytes()).hexdigest()
        if dataset == 'ScanRefer':
            assert cloud_sha == r['point_sha256'], (r['row_id'], cloud_sha, r['point_sha256'])
            assert np.allclose(gt, r['root_box'], atol=1e-6, rtol=0), (r['row_id'], gt.tolist(), r['root_box'], (gt - np.array(r['root_box'])).tolist())
            a = lookup[(r['scan_id'], r['target_id'], r['utterance'])]
            description = a['description']
            prediction_source = str(STAGE / 'diagnostic_result/stage_rows.json')
            prediction_version = 'Protected E71 + Parent + Geometry + V99; stage diagnostic 2026-09-07'
        else:
            valid = r['valid_mask']
            assert np.allclose(iou(r['cached_boxes'][valid], gt), r['cached_ious'][valid], atol=1e-5, rtol=0)
            a = r['original_annotation']
            description = a['utterance']
            prediction_source = str(CACHE / r['shard'])
            prediction_version = 'Protected averaged E57; archived root_only Default candidate-cache diagnostic'
        case_id = dataset.lower() + '_' + group + '_' + r['scan_id'] + '_' + str(r['row_id'])
        target_mask = np.zeros(len(scene.pc), dtype=bool)
        target_mask[scene.three_d_objects[r['target_id']]['points']] = True
        npz = OUT / (case_id + '.npz')
        assert not npz.exists()
        np.savez_compressed(npz, xyz=scene.pc.astype(np.float32), rgb=scene.color.astype(np.float32),
                            target_mask=target_mask, gt_box=gt, pred_box=pred)
        case = {'case_id': case_id, 'dataset': dataset, 'category': group, 'scene_id': r['scan_id'],
                'row_id': r['row_id'], 'target_id': r['target_id'], 'target_name': scene.get_object_instance_label(r['target_id']),
                'description': description, 'cached_input_utterance': r['utterance'], 'original_annotation': a,
                'query_index': r['query_index'], 'pred_box': pred.tolist(), 'gt_box': gt.tolist(),
                'recorded_iou': r['rec_iou'], 'recomputed_iou': actual_iou, 'top16_oracle_iou': r['top16_oracle_iou'],
                'target_points': int(target_mask.sum()), 'scene_points': len(scene.pc), 'input_point_sha256': cloud_sha,
                'input_hash_matches_cached_forward': dataset == 'ScanRefer',
                'geometry_alignment_checks': 'GT from actual scene instance; cached prediction IoU recomputed; all valid cached Nr candidates also checked',
                'prediction_source': prediction_source, 'prediction_source_sha256': sha(prediction_source),
                'prediction_version': prediction_version, 'npz_file': npz.name, 'npz_sha256': sha(npz),
                'benchmark_score_claim': False, 'synthetic_predictions': False}
        selected.append(case)
        print(json.dumps({k: case[k] for k in ['case_id','target_name','description','recorded_iou','top16_oracle_iou','target_points']}), flush=True)
manifest = {'schema': 'mcln-real-grounding-failure-render-input-v1', 'cases': selected,
            'validation_scene_pickle': str(DATA / 'val_v3scans.pkl'), 'validation_scene_pickle_sha256': sha(DATA / 'val_v3scans.pkl'),
            'nr_cache_manifest_sha256': sha(CACHE / 'manifest.json'),
            'scan_diagnostic_receipt_sha256': sha(STAGE / 'diagnostic_result/receipt.json'),
            'selection_note': 'Three distinct scenes per available dataset, selected for display from real failed outputs; not a representative accuracy estimate.',
            'sr3d_status': 'Historical checkpoint and predicted boxes not located; awaiting backup, no substitute predictions generated.',
            'gpu_forwards': 0, 'optimizer_steps': 0}
(OUT / 'cases.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
print('EXPORTED_CASES', len(selected), flush=True)
