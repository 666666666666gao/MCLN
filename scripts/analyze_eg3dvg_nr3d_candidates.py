"""Offline Nr3D candidate coverage and ranking from audited native predictions."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def overlaps(boxes, gt):
    size, target_size = np.maximum(boxes[..., 3:], 1e-6), np.maximum(gt[..., 3:], 1e-6)
    low = np.maximum(boxes[..., :3] - size / 2, gt[..., :3] - target_size / 2)
    high = np.minimum(boxes[..., :3] + size / 2, gt[..., :3] + target_size / 2)
    intersection = np.maximum(high - low, 0).prod(-1)
    return intersection / (size.prod(-1) + target_size.prod(-1) - intersection)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    folder = args.root / 'formal'
    receipt = json.loads((folder / 'receipt.json').read_text())
    audit = json.loads((folder / 'audit.json').read_text())
    assert receipt['status'] == 'complete' and receipt['rows'] == 7899
    assert audit['integrity_pass'] and audit['receipt_sha256'] == sha(folder / 'receipt.json')
    assert sha(folder / 'candidates.npy') == receipt['candidates_sha256']
    assert sha(folder / 'rows.jsonl.gz') == receipt['rows_sha256']
    candidates = np.load(str(folder / 'candidates.npy'), mmap_mode='r')
    assert candidates.shape == (7899, 256, 17) and candidates.dtype == np.float32
    with gzip.open(str(folder / 'rows.jsonl.gz'), 'rt') as f:
        rows = [json.loads(line) for line in f]
    assert [r['row_id'] for r in rows] == list(range(7899))
    gt = np.asarray([r['gt_box'] for r in rows], dtype=np.float64)[:, None, :]
    raw_iou = overlaps(candidates[:, :, :6].astype(np.float64), gt)
    native_iou = overlaps(candidates[:, :, 6:12].astype(np.float64), gt)
    supported = candidates[:, :, 16].astype(bool)
    ix = np.arange(len(rows))
    modes = [('bbs', 12), ('bbf', 13), ('bbs_unfiltered', 14), ('bbf_unfiltered', 15)]
    result = {'status': 'complete', 'rows': len(rows), 'primary_mode': 'bbs',
              'model_forwards': 0, 'optimizer_steps': 0, 'inference_rules_changed': False,
              'checkpoint_sha256': receipt['checkpoint_sha256'],
              'receipt_sha256': sha(folder / 'receipt.json'),
              'candidates_sha256': receipt['candidates_sha256'],
              'rows_sha256': receipt['rows_sha256'], 'thresholds': {},
              'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    for threshold, hit_key in [(.25, 'rec_hits25'), (.5, 'rec_hits50')]:
        raw, native = raw_iou > threshold, native_iou > threshold
        oracle, supported_oracle = native.any(1), (native & supported).any(1)
        values = {'raw_256_unfiltered_oracle_hits': int(raw.any(1).sum()),
                  'native_256_unfiltered_oracle_hits': int(oracle.sum()),
                  'native_object_supported_oracle_hits': int(supported_oracle.sum()), 'modes': {}}
        successes = {}
        for mode, column in modes:
            query = np.asarray([r[mode]['query'] for r in rows])
            score = candidates[:, :, column]
            assert np.array_equal(score[ix, query], score.max(1))
            selected, selected_raw = native[ix, query], raw[ix, query]
            assert int(selected.sum()) == receipt['metrics'][mode][hit_key]
            has_good_but_failed = ~selected & oracle
            best_good_score = np.where(native, score, -np.inf).max(1)
            # Rank interval among all 256 scores. Do not invent a tie-breaking order.
            first_rank = 1 + (score > best_good_score[:, None]).sum(1)
            last_rank = (score >= best_good_score[:, None]).sum(1)
            rank_counts = {}
            for name, rank in [('optimistic', first_rank), ('pessimistic', last_rank)]:
                rank_counts[name] = {label: int((has_good_but_failed & (rank >= low) & (rank <= high)).sum())
                    for label, low, high in [('1', 1, 1), ('2', 2, 2), ('3-16', 3, 16), ('17-64', 17, 64), ('65-256', 65, 256)]}
                assert sum(rank_counts[name].values()) == int(has_good_but_failed.sum())
            values['modes'][mode] = {
                'selected_native_hits': int(selected.sum()),
                'selected_raw_same_query_hits': int(selected_raw.sum()),
                'native_box_repairs_same_query': int((~selected_raw & selected).sum()),
                'native_box_breaks_same_query': int((selected_raw & ~selected).sum()),
                'failures_without_native_qualifying_candidate': int((~selected & ~oracle).sum()),
                'failures_with_native_qualifying_candidate': int(has_good_but_failed.sum()),
                'failures_with_supported_qualifying_candidate': int((~selected & supported_oracle).sum()),
                'failed_good_candidate_rank_interval_counts': rank_counts,
                'failed_good_candidate_score_ties': int((has_good_but_failed & (first_rank != last_rank)).sum()),
                'selected_object_unsupported_rows': int((~supported[ix, query]).sum())}
            successes[mode] = selected
        values['object_filter_same_forward_bbs'] = {
            'repairs': int((~successes['bbs_unfiltered'] & successes['bbs']).sum()),
            'breaks': int((successes['bbs_unfiltered'] & ~successes['bbs']).sum())}
        result['thresholds'][str(threshold)] = values
    result['interpretation_boundary'] = (
        'GT oracle and ranks are offline diagnostics, not deployable recall or accuracy. '
        'Object-supported oracle uses the existing author object-support flag; score filtering is multiplication by zero, not candidate removal. '
        'All modes use the same GT scene object inputs. Unfiltered modes are not single-stage. '
        'IoU qualification does not prove semantic instance identity. Counts do not identify causal training failures. '
        'The current fixed training and primary bbs protocol are not changed by this analysis.')
    with (folder / 'candidate_analysis.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
