"""Summarize missing Mask evidence on cohorts fixed from the sealed P1 audit."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


BRANCHES = ['text', 'query', 'fused']


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mask_metrics(values):
    return {'rows': len(values),
            'hits025': sum(value > .25 for value in values),
            'hits050': sum(value > .5 for value in values),
            'iou_sum': sum(values),
            'mean_iou': sum(values) / len(values) if values else None}


def summarize_group(rows):
    branches = [row['mask_branches'] for row in rows]
    result = {'rows': len(rows), 'oracles_use_gt': True, 'oracles': {}, 'selected': {}}
    for axis in ['all_query_oracles', 'good_legal_box_mask_oracles']:
        masks = {branch: [item[axis][branch]['mask_iou_by_branch'][branch]
                          for item in branches if item[axis][branch] is not None]
                 for branch in BRANCHES}
        result['oracles'][axis] = {branch: mask_metrics(values) for branch, values in masks.items()}
        states = Counter()
        for item in branches:
            if item[axis]['query'] is None:
                states['no_good_legal_box_query'] += 1
            else:
                flags = [item[axis][branch]['mask_iou_by_branch'][branch] > .5 for branch in BRANCHES]
                state = '_'.join('{}{}'.format(branch, int(flag)) for branch, flag in zip(BRANCHES, flags))
                states[state] += 1
        assert sum(states.values()) == len(rows)
        result['oracles'][axis]['joint_pass050_states'] = dict(sorted(states.items()))
    for selection in ['native_rec', 'native_mask', 'best_legal_box']:
        selected = [item['selected_queries'][selection] for item in branches
                    if item['selected_queries'][selection] is not None]
        result['selected'][selection] = {
            'rows': len(selected),
            'box_hits050': sum(item['box_iou'] > .5 for item in selected),
            'masks': {branch: mask_metrics([item['mask_iou_by_branch'][branch] for item in selected])
                      for branch in BRANCHES},
        }
    alpha = [item['alpha'] for item in branches]
    result['alpha'] = {'mean': float(np.mean(alpha)),
                       'percentiles_0_25_50_75_100': np.percentile(alpha, [0, 25, 50, 75, 100]).tolist()}
    result['raw_query_oracle_pass_but_same_query_fusion_fails050'] = sum(
        item['all_query_oracles']['query']['mask_iou_by_branch']['query'] > .5
        and item['all_query_oracles']['query']['mask_iou_by_branch']['fused'] <= .5
        for item in branches)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--addon', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.addon / 'input_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        assert file_sha(args.addon / relative) == expected, relative
    for name, item in manifest['comparison_files'].items():
        assert file_sha(Path(item['path'])) == item['sha256'], name
    assert (args.addon / 'controller.exit').read_text().strip() == '0'
    run = args.addon / 'results'
    receipt = json.loads((run / 'receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['native_row_parity']
    assert receipt['protected_state_unchanged'] and receipt['data_hashes_before_after_match']
    assert receipt['optimizer_steps'] == receipt['checkpoint_writes'] == 0
    assert not receipt['formal_promotion']
    assert file_sha(run / 'rows.jsonl') == receipt['rows_sha256']
    assert file_sha(args.addon / 'input_manifest.json') == receipt['manifest_sha256']
    rows = [json.loads(line) for line in (run / 'rows.jsonl').read_text().splitlines()]
    old_path = Path(manifest['comparison_files']['p1_rows']['path'])
    old = [json.loads(line) for line in old_path.read_text().splitlines()]
    labels = json.loads(Path(manifest['comparison_files']['majority_targets']['path']).read_text())['rows']
    assert len(rows) == len(old) == len(labels) == 7899
    assert [row['id'] for row in rows] == list(range(7899))
    assert len({row['scan_id'] for row in rows}) == 130
    changed_fields = Counter()
    changed_rows = []
    cohorts = {'overall': rows, 'original_good_box_bad_fused_1105': [],
               'original_sp_bound_limited_154': [], 'original_majority_limited_123': [],
               'original_majority_pass_prediction_fail_828': []}
    for current, previous, label in zip(rows, old, labels):
        for key in ['id', 'scan_id', 'target_id']:
            assert current[key] == previous[key] == label[key]
        assert current['root_target_input_points'] == previous['root_target_input_points'] == label['target_points']
        old_fields = dict(previous)
        old_fields['normalized_token_count'] = old_fields.pop('raw_token_count')
        differing = [key for key, value in old_fields.items() if current[key] != value]
        if differing:
            changed_rows.append({'id': current['id'], 'fields': differing})
            changed_fields.update(differing)
        good_box = previous['box_oracle_after_filter']
        if good_box is not None and good_box['box_iou'] > .5 and previous['mask_oracle_all_queries']['mask_iou'] <= .5:
            cohorts['original_good_box_bad_fused_1105'].append(current)
            if label['optimal_superpoint_mask_iou'] <= .5:
                name = 'original_sp_bound_limited_154'
            elif label['majority_mask_iou'] <= .5:
                name = 'original_majority_limited_123'
            else:
                name = 'original_majority_pass_prediction_fail_828'
            cohorts[name].append(current)
    assert [len(group) for group in cohorts.values()] == [7899, 1105, 154, 123, 828]
    groups = {name: summarize_group(members) for name, members in cohorts.items()}
    native = groups['overall']['selected']['native_mask']['masks']['fused']
    assert native['hits025'] == receipt['summary']['mask_hits025']
    assert native['hits050'] == receipt['summary']['mask_hits050']
    assert native['iou_sum'] == receipt['summary']['mask_iou_sum']
    result = {'schema': 'mcln-nr3d-mask-branch-analysis-v1', 'status': 'complete',
              'groups': groups, 'p1_prediction_comparison': {
                  'all_existing_fields_equal': not changed_rows,
                  'changed_row_count': len(changed_rows), 'changed_fields': dict(changed_fields),
                  'changed_rows': changed_rows},
              'original_cohort_membership_preserved': True,
              'historical_metrics_reproduced': receipt['historical_metrics_reproduced'],
              'receipt_sha256': file_sha(run / 'receipt.json'),
              'rows_sha256': receipt['rows_sha256'], 'formal_promotion': False,
              'analysis_gpu_forwards': 0, 'optimizer_steps': 0}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'groups': groups, 'p1_changed_rows': len(changed_rows)}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
