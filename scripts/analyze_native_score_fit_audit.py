"""Recompute fixed-fit score and candidate evidence from saved numeric rows."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def overlap(a, b):
    # Independent scalar reconstruction, without importing model/evaluator code.
    intervals = [[(box[k] - max(box[k + 3], 1e-6) / 2,
                   box[k] + max(box[k + 3], 1e-6) / 2) for k in range(3)]
                 for box in (a, b)]
    volume = [1.0, 1.0]
    intersection = 1.0
    for k in range(3):
        for j in range(2):
            volume[j] *= intervals[j][k][1] - intervals[j][k][0]
        intersection *= max(0.0, min(intervals[0][k][1], intervals[1][k][1])
                            - max(intervals[0][k][0], intervals[1][k][0]))
    return intersection / max(volume[0] + volume[1] - intersection, 1e-6)


def candidates(default, contrastive):
    first = sorted(range(256), key=lambda i: (-default[i], i))
    second = sorted(range(256), key=lambda i: (-contrastive[i], i))
    selected = []
    for q in first[:8] + second[:8] + first:
        if q not in selected:
            selected.append(q)
        if len(selected) == 16:
            return selected
    raise AssertionError('Incomplete candidate set')


def hits(rows, key):
    return [sum(r[key] > t for r in rows) for t in (.25, .5)]


def changes(rows, before, after):
    return {str(t): {'repairs': sum(r[before] <= t < r[after] for r in rows),
                     'breaks': sum(r[after] <= t < r[before] for r in rows),
                     'net': sum((r[after] > t) - (r[before] > t) for r in rows)}
            for t in (.25, .5)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    receipt = json.loads((args.directory / 'receipt.json').read_text())
    manifest_bytes = (args.directory / 'input_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert digest(manifest_bytes) == receipt['manifest_sha256']
    assert receipt['status'] == 'complete' and receipt['model_and_source_unchanged']
    rows = []
    for shard in receipt['shards']:
        encoded = (args.directory / shard['file']).read_bytes()
        assert digest(encoded) == shard['sha256']
        decoded = gzip.decompress(encoded)
        assert digest(decoded) == shard['uncompressed_sha256']
        part = json.loads(decoded)
        assert len(part) == shard['rows'] == 128
        rows.extend(part)
    assert len(rows) == 512 and [r['row_id'] for r in rows] == manifest['selected_row_ids']
    summaries = []
    max_iou_error = max_native_error = max_adapter_error = max_source_error = 0.0
    for row in rows:
        parts = row['components']
        for q in range(256):
            common = sum(parts[n][q] for n in ('modifier', 'pronoun', 'relation')) - parts['other'][q]
            max_native_error = max(max_native_error, abs(row['native_scores'][q] - row['binary_main_scores'][q] - common))
            max_adapter_error = max(max_adapter_error, abs(row['adapter_scores'][q] - parts['main'][q] - common))
            max_source_error = max(max_source_error, abs(row['native_scores'][q] - row['source_choice_scores'][q]))
        assert max_native_error < 2e-6 and max_adapter_error < 2e-6 and max_source_error < 2e-6
        assert row['native_axis_mode'] == 'default_query_axis'
        for score_key, query_key in [('native_scores', 'native_query'), ('adapter_scores', 'adapter_query')]:
            assert row[score_key][row[query_key]] == max(row[score_key])
        assert candidates(row['adapter_scores'], row['contrastive_scores']) == row['original_candidates']
        assert candidates(row['native_scores'], row['contrastive_scores']) == row['native_score_counterfactual_candidates']
        ious = [overlap(box, row['root_box']) for box in row['boxes']]
        max_iou_error = max(max_iou_error, max(abs(a - b) for a, b in zip(ious, row['query_ious'])))
        assert max_iou_error < 2e-5
        assert all((a > t) == (b > t) for a, b in zip(ious, row['query_ious']) for t in (.25, .5))
        summaries.append({'row_id': row['row_id'], 'scan_id': row['scan_id'],
                          'main_tokens': len(row['main_map_values']),
                          'all_main_values_one': all(v == 1 for v in row['main_map_values']),
                          'main_sum': sum(row['main_map_values']),
                          'native_iou': ious[row['native_query']], 'adapter_iou': ious[row['adapter_query']],
                          'original_oracle': max(ious[q] for q in row['original_candidates']),
                          'counterfactual_oracle': max(ious[q] for q in row['native_score_counterfactual_candidates']),
                          'full256_oracle': max(ious), 'native_query': row['native_query'],
                          'adapter_query': row['adapter_query'],
                          'native_top1_retained': row['native_query'] in row['original_candidates'],
                          'membership_changed': set(row['original_candidates']) != set(row['native_score_counterfactual_candidates'])})
    disagreed = [r for r in summaries if r['native_query'] != r['adapter_query']]
    assert len(disagreed) == receipt['top1_disagreements']
    assert sum(not r['native_top1_retained'] for r in summaries) == receipt['native_top1_excluded_original']
    assert sum(r['membership_changed'] for r in summaries) == receipt['native_score_counterfactual_membership_changes']
    groups = {}
    for count in sorted(set(r['main_tokens'] for r in summaries)):
        selected = [r for r in summaries if r['main_tokens'] == count]
        groups[str(count)] = {'rows': len(selected), 'main_all_one_rows': sum(r['all_main_values_one'] for r in selected),
                              'native_hits': hits(selected, 'native_iou'), 'adapter_hits': hits(selected, 'adapter_iou'),
                              'top1_disagreements': sum(r['native_query'] != r['adapter_query'] for r in selected),
                              'membership_changes': sum(r['membership_changed'] for r in selected)}
    report = {'scope': 'fixed 512 previously seen ScanRefer fit rows; no formal result or deployment change',
              'rows': 512, 'score_reconstruction_max_errors': {'native': max_native_error, 'adapter': max_adapter_error, 'source_choice': max_source_error},
              'all_candidate_iou_max_error': max_iou_error,
              'all_candidate_threshold_decisions_match': True,
              'actual_main_value_counts': dict(Counter(str(v) for r in rows for v in r['main_map_values'])),
              'groups_by_main_token_count': groups,
              'hits025_050': {key: hits(summaries, key) for key in ['native_iou', 'adapter_iou', 'original_oracle', 'counterfactual_oracle', 'full256_oracle']},
              'native_vs_adapter': changes(summaries, 'adapter_iou', 'native_iou'),
              'counterfactual_vs_original_oracle': changes(summaries, 'original_oracle', 'counterfactual_oracle'),
              'top1_disagreements': len(disagreed), 'native_top1_excluded_original': receipt['native_top1_excluded_original'],
              'counterfactual_membership_changes': receipt['native_score_counterfactual_membership_changes'],
              'historical_point_mismatches': receipt['historical_point_mismatches'],
              'historical_max_box_difference': max(r['historical_box_max_abs_difference'] for r in rows),
              'historical_max_adapter_score_difference': max(r['historical_adapter_score_max_abs_difference'] for r in rows),
              'per_row': summaries, 'optimizer_steps': 0, 'formal_rows': 0,
              'runner_receipt_sha256': digest((args.directory / 'receipt.json').read_bytes())}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_row'}, sort_keys=True))


if __name__ == '__main__':
    main()
