"""Read-only counterfactual: native Mask selection versus the REC Query Mask."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def metrics(values):
    # Match the native evaluator's sequential per-row accumulation exactly.
    total = 0.0
    for value in values:
        total += value
    return {'rows': len(values), 'hits025': sum(x > .25 for x in values),
            'hits050': sum(x > .5 for x in values),
            'iou_sum': total, 'mean_iou': total / len(values)}


def summarize(directory):
    receipt_path = directory / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    compressed = (directory / 'rows.jsonl.gz').read_bytes()
    raw = gzip.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == receipt['native_metrics']['sample_count'] == 7899
    native = metrics([r['mask_selection']['mask_iou'] for r in rows])
    for key in ('hits025', 'hits050', 'iou_sum'):
        assert native[key] == receipt['summary']['mask_' + key]
    paired = [r for r in rows if r['rec_selection'] is not None]
    different = [r for r in paired if r['rec_selection']['query'] != r['mask_selection']['query']]
    def before_query(row):
        return row['score_profiles']['protected_selector']['before_filter']['top_query']
    def after_query(row):
        return row['score_profiles']['protected_selector']['after_filter']['top_query']
    selection_paths = {
        'mask_equals_before_filter_top_rows': sum(r['mask_selection']['query'] == before_query(r) for r in rows),
        'filter_changes_top_rows_including_no_legal_query': sum(before_query(r) != after_query(r) for r in rows),
        'different_query_rows_with_changed_filter_top': sum(before_query(r) != after_query(r) for r in different),
        'different_query_indices_do_not_prove_different_physical_instances': True,
    }
    old = [r['mask_selection']['mask_iou'] for r in paired]
    new = [r['rec_selection']['mask_iou'] for r in paired]
    changes = {}
    for threshold in (.25, .5):
        fixes = sum(a > threshold and b <= threshold for a, b in zip(new, old))
        breaks = sum(a <= threshold and b > threshold for a, b in zip(new, old))
        changes[str(threshold)] = {'fixes': fixes, 'breaks': breaks, 'net': fixes - breaks}
    return {
        'schema': 'mcln-nr3d-query-identity-diagnostic-v1',
        'source_receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        'source_rows_gzip_sha256': hashlib.sha256(compressed).hexdigest(),
        'source_rows_sha256': receipt['rows_sha256'],
        'source_native_metric_parity': True,
        'new_forwards': 0, 'optimizer_steps': 0, 'deployment_changed': False,
        'native_all_rows': native,
        'rows_without_legal_rec_query': [r['id'] for r in rows if r['rec_selection'] is None],
        'different_query_rows': len(different),
        'selection_path_diagnostic': selection_paths,
        'paired_native_mask': metrics(old), 'paired_rec_query_mask': metrics(new),
        'paired_mask_changes': changes,
        'paired_miou_delta_pp': 100 * (metrics(new)['mean_iou'] - metrics(old)['mean_iou']),
        'interpretation': 'Existing validation-output diagnostic, not a trained method result. '
                          'The row without a legal REC Query has no replacement Mask defined. '
                          'All differing Query selections coincide with a changed REC filter top; '
                          'this does not establish a missing shared-identity representation. '
                          'Do not convert this into a deployed fallback or a validation-tuned selector.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
