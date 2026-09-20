"""Read audited ScanRefer exports to separate coverage, ranking and box changes."""
import argparse
import datetime
import gzip
import json
from pathlib import Path

import numpy as np

from analyze_eg3dvg_candidates import overlaps, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    folder = args.root / 'formal'
    receipt = json.loads((folder / 'receipt.json').read_text())
    audit = json.loads((folder / 'audit.json').read_text())
    assert receipt['status'] == 'complete' and receipt['rows'] == 9508
    assert receipt['primary_mode'] == 'bbs' and audit['integrity_pass']
    assert audit['receipt_sha256'] == sha(folder / 'receipt.json')
    for name, key in [('candidates.npy', 'candidates_sha256'), ('rows.jsonl.gz', 'rows_sha256')]:
        assert sha(folder / name) == receipt[key]
    candidates = np.load(str(folder / 'candidates.npy'), mmap_mode='r')
    assert candidates.shape == (9508, 256, 14) and candidates.dtype == np.float32
    with gzip.open(str(folder / 'rows.jsonl.gz'), 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    assert [r['row_id'] for r in rows] == list(range(9508))
    gt = np.asarray([r['gt_box'] for r in rows], dtype=np.float64)[:, None, :]
    raw_iou = overlaps(candidates[:, :, :6].astype(np.float64), gt)
    averaged_iou = overlaps(candidates[:, :, 6:12].astype(np.float64), gt)
    indices = np.arange(len(rows))
    result = dict(status='complete', rows=9508, primary_mode='bbs',
                  checkpoint_sha256=receipt['checkpoint_sha256'],
                  receipt_sha256=sha(folder / 'receipt.json'),
                  candidates_sha256=receipt['candidates_sha256'],
                  model_forwards=0, optimizer_steps=0, inference_rules_changed=False,
                  thresholds={}, time_cst=datetime.datetime.now(
                      datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    for threshold, key in [(.25, 'rec_hits25'), (.5, 'rec_hits50')]:
        raw, averaged = raw_iou > threshold, averaged_iou > threshold
        oracle = averaged.any(1)
        values = dict(raw_256_unfiltered_oracle_hits=int(raw.any(1).sum()),
                      averaged_256_unfiltered_oracle_hits=int(oracle.sum()), modes={})
        for mode, column in [('bbs', 12), ('bbf', 13)]:
            selected = np.asarray([r[mode]['query'] for r in rows])
            score = candidates[:, :, column]
            assert np.array_equal(score[indices, selected], score.max(1))
            success, raw_success = averaged[indices, selected], raw[indices, selected]
            assert int(success.sum()) == receipt['metrics'][mode][key]
            missed = ~success & oracle
            best_good = np.where(averaged, score, -np.inf).max(1)
            first_rank = 1 + (score > best_good[:, None]).sum(1)
            # Worst first-qualifying rank within a score tie: all tied bad boxes first.
            last_rank = first_rank + ((score == best_good[:, None]) & ~averaged).sum(1)
            ranks = {}
            for label, rank in [('optimistic', first_rank), ('pessimistic', last_rank)]:
                ranks[label] = {name: int((missed & (rank >= low) & (rank <= high)).sum())
                               for name, low, high in [('1', 1, 1), ('2', 2, 2),
                                                      ('3-16', 3, 16), ('17-64', 17, 64),
                                                      ('65-256', 65, 256)]}
                assert sum(ranks[label].values()) == int(missed.sum())
            absent = int((~success & ~oracle).sum())
            assert int(success.sum()) + absent + int(missed.sum()) == 9508
            values['modes'][mode] = dict(
                selected_averaged_hits=int(success.sum()), selected_raw_same_query_hits=int(raw_success.sum()),
                averaged_repairs_same_query=int((~raw_success & success).sum()),
                averaged_breaks_same_query=int((raw_success & ~success).sum()),
                failures_without_qualifying_candidate=absent,
                failures_with_qualifying_candidate=int(missed.sum()),
                failed_first_qualifying_rank_counts=ranks,
                failed_first_qualifying_rank_ambiguous=int((missed & (first_rank != last_rank)).sum()))
        result['thresholds'][str(threshold)] = values
    result['interpretation_boundary'] = (
        'Unfiltered 256-box GT oracle and ranks are offline diagnostics, not deployable accuracy. '
        'IoU qualification does not establish semantic instance identity. '
        'Counts are not a causal decomposition of training effects. '
        'bbs remains primary and bbf diagnostic; no score, checkpoint or inference rule is selected here.')
    with (folder / 'candidate_rank_analysis.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
