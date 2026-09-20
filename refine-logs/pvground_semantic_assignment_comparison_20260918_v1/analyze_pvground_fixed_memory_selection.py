"""CPU analysis of frozen exports; no new scoring rule, training, or evaluation."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load(root, stage, expected_rows=6887):
    directory = root / stage
    receipt = json.loads((directory / 'receipt.json').read_bytes())
    for name, key in [('rows.jsonl', 'rows_sha256'), ('boxes.npy', 'boxes_sha256'),
                      ('scores.npy', 'scores_sha256')]:
        assert sha(directory / name) == receipt[key], name
    rows = [json.loads(line) for line in (directory / 'rows.jsonl').read_text().splitlines()]
    boxes = np.load(str(directory / 'boxes.npy'), mmap_mode='r')
    scores = np.load(str(directory / 'scores.npy'), mmap_mode='r')
    assert len(rows) == receipt['rows'] == expected_rows
    assert boxes.shape == (expected_rows, 256, 6)
    assert scores.shape == (expected_rows, 2, 256) and np.isfinite(scores).all()
    ious = np.empty((expected_rows, 256), dtype=np.float64)
    for index, (raw, row) in enumerate(zip(boxes, rows)):
        box = raw.astype(np.float64)
        box[:, 3:] = np.maximum(box[:, 3:], 1e-6)
        gt = np.asarray(row['root_box'], dtype=np.float64)
        lo = np.maximum(box[:, :3] - box[:, 3:] / 2, gt[:3] - gt[3:] / 2)
        hi = np.minimum(box[:, :3] + box[:, 3:] / 2, gt[:3] + gt[3:] / 2)
        inter = np.maximum(hi - lo, 0).prod(-1)
        ious[index] = inter / (box[:, 3:].prod(-1) + gt[3:].prod() - inter)
    assert np.isfinite(ious).all()
    selected = np.asarray([row['bbs']['query'] for row in rows])
    picked = ious[np.arange(len(rows)), selected]
    assert np.max(np.abs(picked - np.asarray([r['bbs']['iou'] for r in rows]))) < 1e-5
    primary = np.asarray(scores[:, 0])
    # Use the saved selected query for actual decisions. Rank intervals below
    # explicitly expose ties instead of inventing a cross-platform argsort order.
    assert np.all(primary[np.arange(len(rows)), selected] == primary.max(-1))
    result = dict(receipt_sha256=sha(directory / 'receipt.json'), thresholds={})
    arrays = {}
    for threshold, key in [(.25, 'rec_hits25'), (.5, 'rec_hits50')]:
        good = ious > threshold
        available = good.any(-1)
        hit = picked > threshold
        assert int(hit.sum()) == receipt['metrics']['bbs'][key]
        best_good_score = np.where(good, primary, -np.inf).max(-1)
        rank_min = 1 + (primary > best_good_score[:, None]).sum(-1)
        rank_max = rank_min + ((primary == best_good_score[:, None]) & ~good).sum(-1)
        rank_min[~available] = 257
        rank_max[~available] = 257
        topk = {str(k): dict(optimistic=int((rank_min <= k).sum()),
                            conservative=int((rank_max <= k).sum()))
                for k in [1, 2, 4, 8, 16, 32, 64, 256]}
        missed = available & ~hit
        entry = dict(selected_hits=int(hit.sum()), raw256_oracle=int(available.sum()),
                     selection_gap=int(missed.sum()), no_candidate=int((~available).sum()),
                     topk_good_box_presence_bounds=topk,
                     missed_good_score_ties=int((missed & (rank_min != rank_max)).sum()),
                     missed_best_good_rank_median=float(np.median(rank_min[missed])))
        result['thresholds'][str(threshold)] = entry
        arrays[str(threshold)] = dict(hit=hit, available=available, rank=rank_min,
                                      rank_max=rank_max, picked=picked)
    return rows, result, arrays


def paired(before, after):
    output = {}
    for threshold in ['0.25', '0.5']:
        old, new = before[threshold], after[threshold]
        breaks = old['hit'] & ~new['hit']
        fixes = ~old['hit'] & new['hit']
        gap_old = old['available'] & ~old['hit']
        gap_new = new['available'] & ~new['hit']
        net = int(new['hit'].sum() - old['hit'].sum())
        oracle_delta = int(new['available'].sum() - old['available'].sum())
        gap_delta = int(gap_new.sum() - gap_old.sum())
        assert net == oracle_delta - gap_delta
        buckets = dict(no_qualifying_box=int((breaks & ~new['available']).sum()))
        for lo, hi in [(1, 2), (3, 16), (17, 64), (65, 256)]:
            buckets['good_rank_%d_%d' % (lo, hi)] = int(
                (breaks & new['available'] & (new['rank'] >= lo) & (new['rank'] <= hi)).sum())
        assert sum(buckets.values()) == int(breaks.sum())
        output[threshold] = dict(fixes=int(fixes.sum()), breaks=int(breaks.sum()), net=net,
            oracle_delta=oracle_delta, selection_gap_delta=gap_delta,
            arithmetic_identity='selected_delta = oracle_delta - selection_gap_delta',
            broken_samples_after=buckets,
            broken_samples_rank_ties=int((breaks & new['available'] &
                                         (new['rank'] != new['rank_max'])).sum()),
            fixed_samples_previously_no_candidate=int((fixes & ~old['available']).sum()))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d-root', type=Path, required=True)
    parser.add_argument('--e-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        status='complete', primary_mode='bbs', rows=6887, formal_rows=0, model_forwards=0,
        optimizer_steps=0, script_sha256=sha(Path(__file__)), stages={}, pairs={},
        scope='Existing backbone-seen holdout. All 256 raw boxes with native size clamp; no added legal filtering. GT-only capacity analysis, not deployed accuracy or identity attribution. No permanent query-index correspondence. Rank ties are reported as bounds. Arithmetic decomposition is descriptive, not causal.')
    sources = [('E_initial', args.e_root, 'initial'), ('D_terminal', args.d_root, 'terminal'),
               ('E_terminal', args.e_root, 'terminal')]
    data = {}
    identities = None
    for label, root, stage in sources:
        assert (root / 'controller.exit').read_text().strip() == '0'
        rows, summary, arrays = load(root, stage)
        current = [[r[k] for k in ['row_id', 'scan_id', 'target_id', 'point_sha256', 'root_box']] for r in rows]
        if identities is None:
            identities = current
        else:
            assert current == identities
        result['stages'][label] = summary
        data[label] = arrays
    result['pairs']['E_initial_to_terminal'] = paired(data['E_initial'], data['E_terminal'])
    result['pairs']['D_to_E_terminal'] = paired(data['D_terminal'], data['E_terminal'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
