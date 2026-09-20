"""Export real audited EG predictions and their original point/mesh provenance."""
import argparse
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

BASE = Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
NR = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1')
SR = Path('/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1')
OUT = Path('/root/autodl-tmp/mcln_eg3dvg_failure_visuals_20260920_v1')
sys.path.insert(0, str(BASE / 'referit_input_source'))
from src.visual_data_handlers import Scan  # Resolves author cache class.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()


def ious(boxes, gt):
    size, gs = np.maximum(boxes[..., 3:], 1e-6), np.maximum(gt[3:], 1e-6)
    lo = np.maximum(boxes[..., :3] - size / 2, gt[:3] - gs / 2)
    hi = np.minimum(boxes[..., :3] + size / 2, gt[:3] + gs / 2)
    inter = np.maximum(hi - lo, 0).prod(-1)
    return inter / (size.prod(-1) + gs.prod() - inter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=['ScanRefer', 'Nr3D', 'Sr3D'],
                        default=['ScanRefer', 'Nr3D'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    output = args.out
    output.mkdir()
    (output / 'data').mkdir()
    spec = json.loads((BASE / 'spec.json').read_text())
    cache = Path(spec['data_root']) / 'val_v3scans.pkl'
    with cache.open('rb') as f:
        assert pickle.load(f) == 1
        scenes = pickle.load(f)
    selected, sources = [], []
    inputs = {'ScanRefer': (BASE, 9508), 'Nr3D': (NR, 7899), 'Sr3D': (SR, 17726)}
    for dataset in args.datasets:
        root, n = inputs[dataset]
        receipt = json.loads((root / 'formal/receipt.json').read_text())
        audit = json.loads((root / 'formal/audit.json').read_text())
        assert audit['integrity_pass'] and receipt['rows'] == n
        assert audit['receipt_sha256'] == sha(root / 'formal/receipt.json')
        assert receipt['candidates_sha256'] == sha(root / 'formal/candidates.npy')
        assert receipt['rows_sha256'] == sha(root / 'formal/rows.jsonl.gz')
        with gzip.open(str(root / 'formal/rows.jsonl.gz'), 'rt') as f:
            rows = [json.loads(line) for line in f]
        candidates = np.load(str(root / 'formal/candidates.npy'), mmap_mode='r')
        used = set()
        for category in ['selection', 'strict_overlap', 'full256_coverage']:
            found = None
            for row in rows:
                if row['scan_id'] in used:
                    continue
                scene = scenes[row['scan_id']]
                gt = np.asarray(row['gt_box'], np.float64)
                pred = np.asarray(row['bbs']['box'], np.float64)
                label = scene.get_object_instance_label(row['target_id'])
                count = len(scene.three_d_objects[row['target_id']]['points'])
                if label in ['wall', 'floor', 'ceiling'] or min(gt[3:]) <= .06 or min(pred[3:]) <= .025:
                    continue
                if not (.03 < np.prod(gt[3:]) < 5 and count >= 100):
                    continue
                overlaps = ious(candidates[row['row_id'], :, 6:12].astype(np.float64), gt)
                oracle = float(overlaps.max())
                value = float(ious(pred, gt))
                assert abs(value - row['bbs']['iou']) < 1e-4
                distance = np.linalg.norm(pred[:3] - gt[:3])
                eligible = {
                    'selection': value < .05 and oracle > .65 and .4 < distance < 3.5,
                    'strict_overlap': .27 < value < .46,
                    'full256_coverage': value < .25 and oracle < .5 and distance < 3.5}
                if eligible[category]:
                    found = (row, scene, gt, pred, value, oracle, label, count)
                    break
            assert found is not None, (dataset, category)
            row, scene, gt, pred, value, oracle, label, count = found
            used.add(row['scan_id'])
            bounds = scene.get_object_bbox(row['target_id']).reshape(-1)
            actual_gt = np.r_[(bounds[:3] + bounds[3:]) / 2, bounds[3:] - bounds[:3]]
            assert np.allclose(actual_gt, gt, atol=1e-6, rtol=0)
            cloud = np.concatenate([scene.pc, scene.color - np.array([109.8, 97.2, 83.8]) / 256], 1).astype(np.float32)
            assert hashlib.sha256(cloud.tobytes()).hexdigest() == row['point_sha256']
            case_id = dataset.lower() + '_' + category + '_' + row['scan_id'] + '_' + str(row['row_id'])
            path = output / 'data' / (case_id + '.npz')
            mask = np.zeros(len(scene.pc), dtype=bool)
            mask[scene.three_d_objects[row['target_id']]['points']] = True
            np.savez_compressed(path, xyz=scene.pc.astype(np.float32), rgb=scene.color.astype(np.float32),
                                target_mask=mask, gt_box=gt, pred_box=pred)
            selected.append({'case_id': case_id, 'dataset': dataset, 'category': category,
                'scene_id': row['scan_id'], 'row_id': row['row_id'], 'target_id': row['target_id'],
                'target_name': label, 'description': row['utterance'], 'query_index': row['bbs']['query'],
                'gt_box': gt.tolist(), 'pred_box': pred.tolist(), 'recomputed_iou': value,
                'full256_native_oracle_iou': oracle, 'target_points': count,
                'input_point_sha256': row['point_sha256'], 'input_hash_matches_cached_forward': True,
                'npz_file': path.name, 'npz_sha256': sha(path),
                'checkpoint_sha256': receipt['checkpoint_sha256'], 'prediction_source': str(root / 'formal'),
                'prediction_rows_sha256': receipt['rows_sha256'], 'primary_mode': 'bbs',
                'prediction_version': 'Author ScanRefer epoch69 EG-3DVG; zero-update ' + dataset + ' evaluation',
                'benchmark_score_claim': False, 'synthetic_predictions': False})
            print('EXPORTED ' + case_id, flush=True)
    mesh_root = Path('/root/autodl-tmp/DATA_ROOT/scannet/scans')
    paths = [mesh_root / name / (name + '_vh_clean_2.ply') for name in sorted({r['scene_id'] for r in selected})]
    paths.append(BASE / 'referit_input_source/data/meta_data/scans_axis_alignment_matrices.json')
    for path in paths:
        sources.append({'remote_path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path),
                        'local_file': ('meshes/' if path.suffix == '.ply' else '') + path.name})
    manifest = {'cases': selected, 'mesh_sources': sources, 'scene_cache_sha256': sha(cache),
                'gpu_forwards': 0, 'optimizer_steps': 0, 'sr3d_complete': 'Sr3D' in args.datasets,
                'selection_note': 'First qualifying row per display category, distinct scenes within each dataset. Selected failures are not an accuracy sample.',
                'coverage_definition': 'No native averaged box among all256 exceeds IoU0.5; not a Top16 claim.',
                'provenance_note': 'New EG baseline visualizations; do not relabel as protected V99/E57/Sr3D results.'}
    (output / 'cases.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('EG_FAILURE_EXPORT_COMPLETE ' + str(output), flush=True)


if __name__ == '__main__':
    main()
