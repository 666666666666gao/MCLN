"""Paired REC accounting for the fixed EG Sr3D adaptation and its actual transfer start."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def read_rows(root):
    receipt = json.loads((root / 'formal/receipt.json').read_text())
    audit = json.loads((root / 'formal/audit.json').read_text())
    assert receipt['status'] == 'complete' and receipt['rows'] == 17726
    assert audit['integrity_pass'] and receipt['metrics'] == audit['metrics']
    path = root / 'formal/rows.jsonl.gz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['rows_sha256']
    with gzip.open(str(path), 'rt') as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 17726
    return receipt, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=Path, required=True)
    parser.add_argument('--adaptation', type=Path, required=True)
    args = parser.parse_args()
    first, before = read_rows(args.start)
    last, after = read_rows(args.adaptation / 'evaluation')
    fit = json.loads((args.adaptation / 'fit/receipt.json').read_text())
    assert fit['optimizer_steps'] == 9730 and fit['rows'] == 77836
    assert fit['parent_checkpoint_sha256'] == first['checkpoint_sha256']
    assert fit['checkpoint_sha256'] == last['checkpoint_sha256']
    for a, b in zip(before, after):
        for key in ['row_id', 'scan_id', 'target_id', 'utterance', 'point_sha256', 'gt_box', 'object_boxes']:
            assert a[key] == b[key], (a['row_id'], key)
    result = {'status': 'complete', 'rows': 17726, 'identical_inputs': True,
              'start_checkpoint_sha256': first['checkpoint_sha256'],
              'end_checkpoint_sha256': last['checkpoint_sha256'],
              'fit_optimizer_steps': 9730, 'analysis_model_forwards': 0, 'primary_mode': 'bbs', 'modes': {}}
    for mode in ['bbs', 'bbf', 'bbs_unfiltered', 'bbf_unfiltered']:
        result['modes'][mode] = {}
        for threshold, key in [(.25, 'rec_hits25'), (.5, 'rec_hits50')]:
            repaired = [a['row_id'] for a, b in zip(before, after) if a[mode]['iou'] <= threshold < b[mode]['iou']]
            broken = [a['row_id'] for a, b in zip(before, after) if b[mode]['iou'] <= threshold < a[mode]['iou']]
            start_hits = first['metrics'][mode][key]
            end_hits = last['metrics'][mode][key]
            assert end_hits - start_hits == len(repaired) - len(broken)
            result['modes'][mode][str(threshold)] = {'start_hits': start_hits, 'end_hits': end_hits,
                'repairs': len(repaired), 'breaks': len(broken), 'net': end_hits - start_hits,
                'repair_row_ids': repaired, 'break_row_ids': broken}
    result['claim_boundary'] = 'Native one-pass adaptation from author ScanRefer weights; no new module or augmentation change. Only bbs filtered is primary.'
    with (args.adaptation / 'paired_rec.json').open('x') as f:
        json.dump(result, f, indent=2)
    print('EG_SR_PAIRED_REC_COMPLETE ' + json.dumps({m: {t: {k: v for k, v in values.items() if not k.endswith('_ids')}
          for t, values in by_threshold.items()} for m, by_threshold in result['modes'].items()}), flush=True)


if __name__ == '__main__':
    main()
