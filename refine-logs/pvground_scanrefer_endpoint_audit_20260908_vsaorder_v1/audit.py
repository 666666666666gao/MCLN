"""Recount the fixed PV-Ground ScanRefer fit endpoint from saved native outputs."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metrics(rows, mode):
    values = [row[mode] for row in rows]
    mask_sum = sum(row['mask_iou'] for row in values)
    return {'rec_hits25': sum(row['iou'] > .25 for row in values),
            'rec_hits50': sum(row['iou'] > .5 for row in values),
            'mask_hits25': sum(row['mask_iou'] > .25 for row in values),
            'mask_hits50': sum(row['mask_iou'] > .5 for row in values),
            'mask_iou_sum': mask_sum, 'mask_miou': mask_sum / len(rows) * 100.}


def audit_stage(root, stage, expected_ids):
    directory = root / stage
    receipt = json.loads((directory / 'receipt.json').read_bytes())
    assert receipt['status'] == 'pass' and receipt['stage'] == stage
    assert receipt['rows'] == len(expected_ids) == 6887 and receipt['formal_rows'] == 0
    for name in ['rows', 'boxes', 'scores']:
        suffix = '.jsonl' if name == 'rows' else '.npy'
        assert sha(directory / (name + suffix)) == receipt[name + '_sha256'], name
    rows = [json.loads(line) for line in (directory / 'rows.jsonl').read_text().splitlines()]
    assert [row['row_id'] for row in rows] == expected_ids
    boxes = np.load(str(directory / 'boxes.npy'), mmap_mode='r')
    scores = np.load(str(directory / 'scores.npy'), mmap_mode='r')
    assert boxes.shape == (6887, 256, 6) and scores.shape == (6887, 2, 256)
    assert boxes.dtype == scores.dtype == np.dtype('float32')
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    largest_iou_error = 0.
    for index, row in enumerate(rows):
        gt = np.asarray(row['root_box'], dtype=np.float64)
        assert gt.shape == (6,) and np.isfinite(gt).all() and (gt[3:] > 0).all()
        candidate = np.asarray(boxes[index], dtype=np.float64).copy()
        candidate[:, 3:] = np.maximum(candidate[:, 3:], 1e-6)
        lo = np.maximum(candidate[:, :3] - candidate[:, 3:] / 2, gt[:3] - gt[3:] / 2)
        hi = np.minimum(candidate[:, :3] + candidate[:, 3:] / 2, gt[:3] + gt[3:] / 2)
        intersection = np.maximum(hi - lo, 0).prod(axis=-1)
        ious = intersection / (candidate[:, 3:].prod(axis=-1) + gt[3:].prod() - intersection)
        assert np.isfinite(ious).all()
        for mode_index, mode in enumerate(['bbs', 'bbf']):
            selected = row[mode]
            query = selected['query']
            assert isinstance(query, int) and 0 <= query < 256
            assert scores[index, mode_index, query] == scores[index, mode_index].max()
            actual_box = np.asarray(selected['box'], dtype=np.float64)
            # Exported boxes use float32 clamp(1e-6); the CPU arithmetic is float64.
            assert np.allclose(actual_box, candidate[query], rtol=0, atol=1e-12)
            error = abs(float(ious[query]) - selected['iou'])
            largest_iou_error = max(largest_iou_error, error)
            assert error < 1e-5, (stage, row['row_id'], mode, error)
            assert math.isfinite(selected['mask_iou']) and 0 <= selected['mask_iou'] <= 1
            assert math.isfinite(selected['iou']) and 0 <= selected['iou'] <= 1
    actual = {mode: metrics(rows, mode) for mode in ['bbs', 'bbf']}
    assert actual == receipt['metrics']
    return rows, actual, {'rows': len(rows), 'candidate_boxes_checked': int(np.prod(boxes.shape[:2])),
                          'selection_checks': len(rows) * 2, 'max_selected_iou_error': largest_iou_error}


def audit(root):
    assert (root / 'controller.exit').read_text().strip() == '0'
    receipt = json.loads((root / 'receipt.json').read_bytes())
    spec = json.loads((root / 'spec.json').read_bytes())
    manifest = json.loads(Path(spec['input_manifest']).read_bytes())
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    assert not set(partitions['fit']).intersection(partitions['holdout'])
    assert receipt['status'] == 'complete' and receipt['formal_rows'] == 0
    assert receipt['training_steps'] == 3723 and receipt['fit_rows'] == 29778
    assert receipt['holdout_rows'] == 6887 and receipt['primary_mode'] == spec['primary_mode'] == 'bbs'
    assert receipt['checkpoint_sha256'] == spec['checkpoint_sha256']
    assert sha(root / 'spec.json') == receipt['spec_sha256']
    assert sha(root / 'train.py') == receipt['script_sha256'] == spec['files']['train.py']
    assert sha(root / 'plan.md') == spec['files']['plan.md']
    assert sha(root / 'terminal.pth') == receipt['terminal_sha256']
    assert sha(root / 'train.jsonl') == receipt['train_log_sha256']
    assert receipt['frozen_parameters_unchanged'] and receipt['fit_seen_exactly_once']
    training = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    assert [row['step'] for row in training] == list(range(1, 3724))
    seen = [index for step in training for index in step['rows']]
    assert Counter(seen) == Counter(partitions['fit'])
    assert not set(seen).intersection(partitions['holdout'])
    assert [len(step['rows']) for step in training] == [8] * 3722 + [2]
    for row in training:
        assert row['total_steps'] == 3723
        for key in ['loss', 'grad_norm', 'seconds', 'loss_bbox', 'loss_giou', 'loss_ce', 'loss_sem_align']:
            assert math.isfinite(row[key]), (row['step'], key)
    before, baseline, baseline_audit = audit_stage(root, 'initial', partitions['holdout'])
    after, terminal, terminal_audit = audit_stage(root, 'terminal', partitions['holdout'])
    assert baseline == receipt['initial'] and terminal == receipt['terminal']
    transitions = {}
    bands = {}
    for mode in ['bbs', 'bbf']:
        transitions[mode] = {}
        bands[mode] = [[0] * 3 for _ in range(3)]
        for old, new in zip(before, after):
            for key in ['row_id', 'scan_id', 'target_id', 'point_sha256', 'root_box']:
                assert old[key] == new[key], key
            old_band = int(old[mode]['iou'] > .25) + int(old[mode]['iou'] > .5)
            new_band = int(new[mode]['iou'] > .25) + int(new[mode]['iou'] > .5)
            bands[mode][old_band][new_band] += 1
        for threshold in [.25, .5]:
            fixes = sum(old[mode]['iou'] <= threshold < new[mode]['iou'] for old, new in zip(before, after))
            breaks = sum(new[mode]['iou'] <= threshold < old[mode]['iou'] for old, new in zip(before, after))
            transitions[mode][str(threshold)] = {'fixes': fixes, 'breaks': breaks, 'net': fixes - breaks}
    assert transitions == receipt['transitions']
    nonregression = all(transitions['bbs'][str(t)]['net'] >= 0 for t in [.25, .5])
    assert nonregression == receipt['primary_rec_nonregression']
    return {'integrity_pass': True, 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'formal_rows': 0, 'fit_rows': len(seen), 'training_steps': len(training), 'holdout_rows': len(after),
            'primary_mode': 'bbs', 'initial': baseline, 'terminal': terminal, 'transitions': transitions,
            'transition_bands': ['[0,0.25]', '(0.25,0.50]', '(0.50,1]'], 'transition_counts': bands,
            'stages': {'initial': baseline_audit, 'terminal': terminal_audit},
            'primary_rec_nonregression': nonregression, 'candidate_for_fixed_formal_evaluation': nonregression,
            'mask_audit_scope': 'recount exported per-row mask IoU; original binary masks are not reloaded',
            'new_scene_generalization_claim': False, 'nr3d_sr3d_promotion': False,
            'receipt_sha256': sha(root / 'receipt.json'), 'auditor_sha256': sha(__file__)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('PVG_FINETUNE_AUDIT_COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
