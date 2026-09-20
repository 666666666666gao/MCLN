"""CPU recount of selected EG-3DVG boxes; does not rerun or alter the model."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def iou(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    sa, sb = np.maximum(a[3:], 1e-6), np.maximum(b[3:], 1e-6)
    low = np.maximum(a[:3] - sa / 2, b[:3] - sb / 2)
    high = np.minimum(a[:3] + sa / 2, b[:3] + sb / 2)
    inter = np.maximum(high - low, 0).prod()
    return float(inter / (sa.prod() + sb.prod() - inter))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    receipt = json.loads((root / 'formal/receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['rows'] == 9508
    assert receipt['all_model_states_unchanged'] and receipt['training_steps'] == 0
    rows_path = root / 'formal/rows.jsonl.gz'
    assert sha(rows_path) == receipt['rows_sha256']
    with gzip.open(str(rows_path), 'rt') as f:
        rows = [json.loads(line) for line in f]
    assert [row['row_id'] for row in rows] == list(range(9508))
    manifest = json.loads((root / 'annotation_manifest.json').read_text())
    assert len(manifest) == len(rows)
    counts = {m: {'rec_hits25': 0, 'rec_hits50': 0} for m in ['bbs', 'bbf']}
    maximum_error = 0.
    for row, annotation in zip(rows, manifest):
        assert row['scan_id'] == annotation['scan_id'] and row['target_id'] == annotation['target_id']
        expected_text = ' '.join(annotation['utterance'].replace(',', ' ,').split()) + ' . not mentioned'
        assert row['utterance'] == expected_text
        for mode in counts:
            v = row[mode]
            assert 0 <= v['query'] < 256
            assert np.allclose(v['box'], (np.asarray(v['raw_box']) + np.asarray(v['mask_box'])) / 2, atol=1e-6, rtol=1e-6)
            recomputed = iou(row['gt_box'], v['box'])
            assert np.isfinite(recomputed)
            maximum_error = max(maximum_error, abs(recomputed - v['iou']))
            assert abs(recomputed - v['iou']) < 1e-4
            for threshold, name in [(.25, 'rec_hits25'), (.5, 'rec_hits50')]:
                assert (recomputed > threshold) == (v['iou'] > threshold)
                counts[mode][name] += int(recomputed > threshold)
    assert counts == receipt['metrics']
    audit = {'integrity_pass': True, 'rows': len(rows), 'metrics': counts,
             'maximum_iou_error': maximum_error, 'receipt_sha256': sha(root / 'formal/receipt.json'),
             'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
             'model_forwards': 0, 'optimizer_steps': 0, 'primary_mode': 'bbs',
             'rec25_v99_floor': counts['bbs']['rec_hits25'] >= 5572,
             'rec50_v99_floor': counts['bbs']['rec_hits50'] >= 4797,
             'mask_metric_gate': False}
    with (root / 'formal/audit.json').open('x') as f:
        json.dump(audit, f, indent=2)
    print('EG_AUDIT_COMPLETE ' + json.dumps(audit))


if __name__ == '__main__':
    main()
