"""Analyze a completed EG candidate export without inference or score changes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def overlaps(boxes, gt):
    sizes = np.maximum(boxes[..., 3:], 1e-6)
    gs = np.maximum(gt[..., 3:], 1e-6)
    lo = np.maximum(boxes[..., :3] - sizes / 2, gt[..., :3] - gs / 2)
    hi = np.minimum(boxes[..., :3] + sizes / 2, gt[..., :3] + gs / 2)
    inter = np.maximum(hi - lo, 0).prod(-1)
    return inter / (sizes.prod(-1) + gs.prod(-1) - inter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    receipt = json.loads((root / 'formal/receipt.json').read_text())
    audit = json.loads((root / 'formal/audit.json').read_text())
    assert receipt['status'] == 'complete' and audit['integrity_pass']
    assert receipt['rows'] == 9508 and receipt['primary_mode'] == 'bbs'
    assert audit['receipt_sha256'] == sha(root / 'formal/receipt.json')
    assert sha(root / 'formal/candidates.npy') == receipt['candidates_sha256']
    assert sha(root / 'formal/rows.jsonl.gz') == receipt['rows_sha256']
    candidates = np.load(str(root / 'formal/candidates.npy'), mmap_mode='r')
    with gzip.open(str(root / 'formal/rows.jsonl.gz'), 'rt') as f:
        rows = [json.loads(line) for line in f]
    gt = np.array([r['gt_box'] for r in rows], dtype=np.float64)[:, None, :]
    raw_iou = overlaps(candidates[:, :, :6].astype(np.float64), gt)
    native_iou = overlaps(candidates[:, :, 6:12].astype(np.float64), gt)
    selected = {mode: np.array([r[mode]['query'] for r in rows]) for mode in ['bbs', 'bbf']}
    ix = np.arange(len(rows))
    result = {'status': 'complete', 'rows': len(rows), 'model_forwards': 0,
              'optimizer_steps': 0, 'inference_rules_changed': False,
              'receipt_sha256': sha(root / 'formal/receipt.json'),
              'candidates_sha256': receipt['candidates_sha256'],
              'primary_mode': 'bbs', 'thresholds': {}}
    for threshold, key in [(.25, 'rec_hits25'), (.5, 'rec_hits50')]:
        raw = raw_iou > threshold
        native = native_iou > threshold
        ro, no = raw.any(1), native.any(1)
        modes = {}
        successes = {}
        for mode, q in selected.items():
            sr, sn = raw[ix, q], native[ix, q]
            successes[mode] = sn
            assert int(sn.sum()) == receipt['metrics'][mode][key]
            modes[mode] = {'selected_native_hits': int(sn.sum()),
                           'selected_raw_same_query_hits': int(sr.sum()),
                           'native_box_repairs_same_query': int((~sr & sn).sum()),
                           'native_box_breaks_same_query': int((sr & ~sn).sum()),
                           'failures_without_native_qualifying_candidate': int((~sn & ~no).sum()),
                           'failures_with_native_qualifying_candidate': int((~sn & no).sum())}
        result['thresholds'][str(threshold)] = {
            'raw_256_unfiltered_oracle_hits': int(ro.sum()),
            'native_256_unfiltered_oracle_hits': int(no.sum()),
            'modes': modes,
            'bbf_vs_bbs_diagnostic_only': {
                'repairs': int((~successes['bbs'] & successes['bbf']).sum()),
                'breaks': int((successes['bbs'] & ~successes['bbf']).sum())}}
    result['interpretation_boundary'] = (
        'Oracle uses GT solely for offline diagnosis and does not represent deployable accuracy. '
        'Same-query raw/native comparison isolates output choice only after that query has been selected; '
        'it does not identify semantic-instance correctness or a causal training bottleneck. '
        'bbf remains diagnostic; no output-head selection is performed.')
    with (root / 'formal/candidate_analysis.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
