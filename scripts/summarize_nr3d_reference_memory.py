"""Report all fixed R1 arms after the complete holdout receipt is sealed."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


MODES = ['protected', 'default', 'query_global', 'query_pair', 'object_global', 'object_pair']
EFFECTS = {
    'memory_with_global': {'object_global': 1, 'query_global': -1},
    'memory_with_pair': {'object_pair': 1, 'query_pair': -1},
    'readout_with_query': {'query_pair': 1, 'query_global': -1},
    'readout_with_object': {'object_pair': 1, 'object_global': -1},
    'interaction': {'object_pair': 1, 'object_global': -1, 'query_pair': -1, 'query_global': 1},
    'primary_minus_protected': {'object_pair': 1, 'protected': -1},
}


def mask_iou(row, mode):
    return row['protected_mask_iou'] if mode == 'protected' else row['scores'][mode]['mask_iou']


def scene_cluster_intervals(rows, resamples=2000):
    """Keep all expressions of each sampled scene and expression weighting."""
    scenes = sorted({row['scan_id'] for row in rows})
    scene_index = {scene: index for index, scene in enumerate(scenes)}
    counts = np.zeros(len(scenes), dtype=np.float64)
    differences = np.zeros((len(scenes), 3 * len(EFFECTS)), dtype=np.float64)
    for row in rows:
        index = scene_index[row['scan_id']]
        counts[index] += 1
        for effect_index, coefficients in enumerate(EFFECTS.values()):
            offset = 3 * effect_index
            for mode, coefficient in coefficients.items():
                for column, threshold in enumerate([.25, .5]):
                    differences[index, offset + column] += coefficient * int(row['scores'][mode]['box_iou'] > threshold)
                differences[index, offset + 2] += coefficient * mask_iou(row, mode)
    rng = np.random.RandomState(20260905)
    multiplicities = rng.multinomial(len(scenes), np.full(len(scenes), 1.0 / len(scenes)), size=resamples)
    bootstrap = 100 * (multiplicities @ differences) / (multiplicities @ counts)[:, None]
    limits = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    estimates = 100 * differences.sum(axis=0) / counts.sum()
    result = {'unit': 'percentage_points', 'method': 'paired_whole_scene_percentile_bootstrap',
              'scenes': len(scenes), 'rows': len(rows), 'resamples': resamples, 'seed': 20260905,
              'screening_gates_changed': False, 'effects': {}}
    for index, name in enumerate(EFFECTS):
        result['effects'][name] = {
            metric: {'estimate': float(estimates[3 * index + column]),
                     'percentile_95_interval': limits[:, 3 * index + column].tolist()}
            for column, metric in enumerate(['rec025', 'rec050', 'mask_mean_iou'])}
    return result


def metrics(rows):
    result = {'rows': len(rows), 'scores': {}}
    for mode in MODES:
        masks = [mask_iou(row, mode) for row in rows]
        result['scores'][mode] = {
            'rec_hits025': sum(row['scores'][mode]['box_iou'] > .25 for row in rows),
            'rec_hits050': sum(row['scores'][mode]['box_iou'] > .5 for row in rows),
            'mask_hits025': sum(value > .25 for value in masks),
            'mask_hits050': sum(value > .5 for value in masks),
            'mask_iou_sum': sum(masks),
        }
    return result


def paired_changes(rows, old_mode, new_mode):
    result = {}
    for metric, field in [('rec', 'box_iou'), ('mask', 'mask_iou')]:
        for suffix, threshold in [('025', .25), ('050', .5)]:
            old = [(mask_iou(row, old_mode) if field == 'mask_iou' else row['scores'][old_mode][field]) > threshold for row in rows]
            new = [(mask_iou(row, new_mode) if field == 'mask_iou' else row['scores'][new_mode][field]) > threshold for row in rows]
            fixes = sum(not a and b for a, b in zip(old, new))
            breaks = sum(a and not b for a, b in zip(old, new))
            result[metric + suffix] = {'reference_hits': sum(old), 'new_hits': sum(new),
                                       'fixes': fixes, 'breaks': breaks, 'delta_hits': fixes - breaks}
    return result


def summarize(rows):
    groups = {'overall': rows,
              'raw_tokens_2_to_6': [row for row in rows if 2 <= row['raw_token_count'] <= 6],
              'raw_tokens_7_to_8': [row for row in rows if 7 <= row['raw_token_count'] <= 8],
              'raw_tokens_9_to_12': [row for row in rows if 9 <= row['raw_token_count'] <= 12],
              'raw_tokens_13plus': [row for row in rows if row['raw_token_count'] >= 13],
              'hard_2plus_distractors': [row for row in rows if row['distractor_count'] >= 2],
              'sparse_at_most_227_sampled_points': [row for row in rows if row['target_points'] <= 227]}
    scenes = defaultdict(list)
    for row in rows:
        scenes[row['scan_id']].append(row)
    comparisons = {
        name: {group: paired_changes(members,
                                     next(mode for mode, sign in coefficients.items() if sign == -1),
                                     next(mode for mode, sign in coefficients.items() if sign == 1))
               for group, members in groups.items()}
        for name, coefficients in EFFECTS.items() if name != 'interaction'}
    availability = {name: sum(row['object_availability_proxy'][name] for row in rows)
                    for name in ['detector_objects', 'full_256', 'full_legal', 'target_top32',
                                 'object_input_slots', 'object_predicted_class_correct']}
    availability['is_text_anchor_ground_truth'] = False
    coverage = {
        stage: {str(k): {'hits025': sum(row['oracle'][stage][str(k)] > .25 for row in rows),
                         'hits050': sum(row['oracle'][stage][str(k)] > .5 for row in rows)}
                for k in [16, 32, 64, 256]}
        for stage in ['before_filter', 'after_filter']}
    return {'groups': {name: metrics(members) for name, members in groups.items()},
            'scenes': {name: metrics(members) for name, members in scenes.items()},
            'paired_changes': comparisons, 'coverage': coverage,
            'object_availability_proxy': availability,
            'zero_legal_rows': sum(row['legal_queries'] == 0 for row in rows),
            'scene_cluster_uncertainty': scene_cluster_intervals(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.run / 'receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['protected_evaluator_row_parity']
    assert receipt['protected_state_unchanged'] and receipt['backbone_gradients_absent']
    raw = (args.run / 'holdout_rows.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['holdout_rows_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == len({row['id'] for row in rows}) == receipt['census']['holdout']['rows']
    result = summarize(rows)
    for mode, values in result['groups']['overall']['scores'].items():
        for metric in ['rec_hits025', 'rec_hits050', 'mask_hits025', 'mask_hits050']:
            assert values[metric] == receipt['summary'][mode][metric]
        assert values['mask_iou_sum'] / len(rows) == receipt['summary'][mode]['mask_mean_iou']
    result.update(schema='mcln-r1-completed-diagnostics-v1', formal_validation=False,
                  holdout_rows_sha256=receipt['holdout_rows_sha256'], decision=receipt['decision'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'overall': result['groups']['overall'],
                      'scene_cluster_uncertainty': result['scene_cluster_uncertainty'],
                      'decision': result['decision']}, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
