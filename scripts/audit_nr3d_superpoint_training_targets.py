"""Measure the point-space quality of existing majority-superpoint GT labels."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from scripts.nr3d_superpoint_mask_oracle import optimal_superpoint_mask_iou


def describe_target(labels, target_mask):
    _, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    positives = np.bincount(inverse, weights=target_mask)
    total = int(target_mask.sum())
    assert total > 0
    foreground = 2 * positives > counts
    intersection = positives[foreground].sum()
    background = (counts[foreground] - positives[foreground]).sum()
    return {'target_points': total, 'occupied_superpoints': len(counts),
            'native_superpoint_slots': int(labels.max()) + 1,
            'empty_native_slots': int(labels.max()) + 1 - len(counts),
            'target_touching_superpoints': int((positives > 0).sum()),
            'majority_positive_superpoints': int(foreground.sum()),
            'majority_mask_iou': float(intersection / (total + background)),
            'majority_target_point_recall': float(intersection / total),
            'optimal_superpoint_mask_iou': optimal_superpoint_mask_iou(labels, target_mask)}


def aggregate(rows):
    return {'rows': len(rows),
            'zero_majority_positive_rows': sum(row['majority_positive_superpoints'] == 0 for row in rows),
            'majority_mask_hits025': sum(row['majority_mask_iou'] > .25 for row in rows),
            'majority_mask_hits050': sum(row['majority_mask_iou'] > .5 for row in rows),
            'majority_mask_iou_sum': sum(row['majority_mask_iou'] for row in rows),
            'optimal_mask_hits025': sum(row['optimal_superpoint_mask_iou'] > .25 for row in rows),
            'optimal_mask_hits050': sum(row['optimal_superpoint_mask_iou'] > .5 for row in rows),
            'optimal_mask_iou_sum': sum(row['optimal_superpoint_mask_iou'] for row in rows),
            'majority_le050_but_optimal_gt050': sum(row['majority_mask_iou'] <= .5 < row['optimal_superpoint_mask_iou'] for row in rows),
            'majority_target_point_recall_sum': sum(row['majority_target_point_recall'] for row in rows)}


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data-provenance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.results / 'receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['native_row_parity']
    assert file_sha(args.results / 'rows.jsonl') == receipt['rows_sha256']
    raw_rows = [json.loads(line) for line in (args.results / 'rows.jsonl').read_text().splitlines()]
    oracle = json.loads((args.results / 'superpoint_oracle.json').read_text())
    assert oracle['rows_sha256'] == receipt['rows_sha256']
    provenance = json.loads(args.data_provenance.read_text())['files']
    os.chdir(str(args.source))
    sys.path.insert(0, str(args.source))
    from src.joint_det_dataset import unpickle_data
    import torch
    data = Path('/root/autodl-tmp/DATA_ROOT')
    scenes = sorted({row['scan_id'] for row in raw_rows})
    paths = [data / 'val_v3scans.pkl'] + [data / 'superpoints/val' / (scene + '_superpoint.pth') for scene in scenes]
    checked_files = {}
    for path in paths:
        expected = provenance[str(path)]
        assert path.stat().st_size == expected['bytes'] and file_sha(path) == expected['sha256'], str(path)
        checked_files[str(path)] = expected
    scans = list(unpickle_data(str(paths[0])))[0]
    labels = {scene: np.asarray(torch.load(str(path))) for scene, path in zip(scenes, paths[1:])}
    objects = {}
    rows = []
    groups = defaultdict(list)
    categories = defaultdict(list)
    for row, old_oracle in zip(raw_rows, oracle['rows']):
        key = row['scan_id'], row['target_id']
        if key not in objects:
            scan = scans[key[0]]
            target = np.zeros(len(scan.orig_pc), dtype=bool)
            target[scan.three_d_objects[key[1]]['points']] = True
            assert labels[key[0]].shape == target.shape
            objects[key] = describe_target(labels[key[0]], target)
        result = dict(objects[key], id=row['id'], scan_id=key[0], target_id=key[1], target_name=row['target_name'])
        assert result['target_points'] == row['root_target_input_points']
        assert result['optimal_superpoint_mask_iou'] == old_oracle['superpoint_mask_oracle_iou']
        assert result['majority_mask_iou'] <= result['optimal_superpoint_mask_iou'] + 1e-12
        rows.append(result)
        groups['overall'].append(result)
        if result['target_points'] <= 227:
            groups['sparse_at_most_227_points'].append(result)
        if result['target_points'] <= 32:
            groups['very_sparse_at_most_32_points'].append(result)
        if (row['box_oracle_after_filter'] is not None
                and row['box_oracle_after_filter']['box_iou'] > .5
                and row['mask_oracle_all_queries']['mask_iou'] <= .5):
            groups['good_box_but_all_query_masks_le050'].append(result)
            if result['optimal_superpoint_mask_iou'] > .5:
                groups['good_box_bad_masks_despite_superpoint_bound_gt050'].append(result)
        categories[result['target_name']].append(result)
    assert len(rows) == 7899 and len(objects) == 1213
    assert len(groups['good_box_but_all_query_masks_le050']) == 1105
    assert len(groups['good_box_bad_masks_despite_superpoint_bound_gt050']) == 951
    result = {'schema': 'mcln-nr3d-superpoint-training-target-audit-v1', 'rows': rows,
              'groups': {name: aggregate(members) for name, members in groups.items()},
              'categories': {name: aggregate(members) for name, members in categories.items()},
              'unique_objects': aggregate(list(objects.values())),
              'data_files': checked_files, 'rows_sha256': receipt['rows_sha256'],
              'superpoint_oracle_sha256': file_sha(args.results / 'superpoint_oracle.json'),
              'model_forwards': 0, 'optimizer_steps': 0, 'formal_promotion': False,
              'training_rule_changed': False,
              'interpretation': 'GT-label representation diagnostic on sealed validation inputs; majority masks are training targets, not model predictions or new official scores.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'groups': result['groups'], 'unique_objects': result['unique_objects']}, indent=2))


if __name__ == '__main__':
    main()
