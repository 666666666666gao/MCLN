"""Fixed sparse-point screen and Query-conditioned Mask diagnostics."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(rows, arm, mask_field='mask_iou'):
    values = [row[arm][mask_field] for row in rows
              if row[arm][mask_field] is not None]
    assert values
    return {'rows': len(rows), 'evaluated_rows': len(values),
            'mask_mean_iou': sum(values) / len(values),
            'mask_hits_025': sum(value > .25 for value in values),
            'mask_hits_050': sum(value > .5 for value in values)}


def compare(reference, candidate, mask_field='mask_iou'):
    assert len(reference) == len(candidate) and reference
    scenes, changes = {}, []
    for a, b in zip(reference, candidate):
        assert a['id'] == b['id'] and a['scan_id'] == b['scan_id']
        old, new = a['native'][mask_field], b['sparse'][mask_field]
        assert (old is None) == (new is None)
        if old is None:
            continue
        delta = [new - old, int(new > .25) - int(old > .25),
                 int(new > .5) - int(old > .5)]
        scene = scenes.setdefault(a['scan_id'], [0, 0., 0, 0])
        scene[0] += 1
        for i, value in enumerate(delta, 1):
            scene[i] += value
        changes.append((old, new))
    assert changes
    thresholds = {}
    for value, suffix in [(.25, '025'), (.5, '050')]:
        fixed = sum(a <= value < b for a, b in changes)
        broken = sum(b <= value < a for a, b in changes)
        thresholds[suffix] = {'fixes': fixed, 'breaks': broken, 'net': fixed - broken}
    cluster_values = np.asarray(list(scenes.values()))
    rng = np.random.RandomState(0)
    draws = cluster_values[rng.randint(0, len(scenes), (2000, len(scenes)))].sum(axis=1)
    bounds = np.percentile(draws[:, 1:] / draws[:, :1], [2.5, 97.5], axis=0)
    mean = sum(b - a for a, b in changes) / len(changes)
    return {'rows': len(reference), 'evaluated_rows': len(changes),
            'mask_field': mask_field, 'delta_mask_mean_iou': mean,
            'thresholds': thresholds, 'scene_count': len(scenes),
            'per_scene': {name: dict(zip(['rows', 'mask_iou_delta_sum', 'net025', 'net050'], values))
                          for name, values in scenes.items()},
            'paired_scene_bootstrap_95_ci': dict(zip(
                ['mask_mean_iou', 'mask_acc025', 'mask_acc050'], bounds.T.tolist()))}


def passes_fixed_screen(comparisons):
    return all(value['delta_mask_mean_iou'] >= .002 and
               all(item['net'] >= 0 for item in value['thresholds'].values())
               for value in comparisons.values())


def good_box_cohort(baseline, terminal):
    assert len(baseline) == len(terminal)
    selected = [i for i, row in enumerate(baseline)
                if row['native']['legal_box_oracle_iou'] is not None
                and row['native']['legal_box_oracle_iou'] > .5]
    return [baseline[i] for i in selected], [terminal[i] for i in selected]


def require_identity(baseline, terminal):
    assert len(baseline) == len(terminal)
    for a, b in zip(baseline, terminal):
        for key in ['id', 'scan_id', 'grounding_sha256', 'input_point_sha256']:
            assert a[key] == b[key], (a['id'], key)
        assert a['native'] == a['sparse'], a['id']
        for arm in ['native', 'sparse']:
            for key in ['rec_query', 'rec_box_iou', 'mask_query',
                        'legal_box_oracle_query', 'legal_box_oracle_iou']:
                assert a['native'][key] == b[arm][key], (a['id'], arm, key)


def verify_terminal_run(directory):
    receipt_raw = (directory / 'receipt.json').read_bytes()
    receipt = json.loads(receipt_raw)
    manifest_raw = (directory / 'input_manifest.json').read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == receipt['manifest_sha256']
    manifest = json.loads(manifest_raw)
    assert receipt['schema'] == 'mcln-nr3d-sparse-point-pair-v3'
    assert receipt['status'] == 'complete' and receipt['optimizer_steps_per_arm'] == 6687
    assert receipt['frozen_parameters_and_buffers_unchanged']
    assert receipt['source_data_and_parent_checkpoint_unchanged']
    assert receipt['grounding_and_query_selection_exactly_equal_to_start']
    assert receipt['formal_rows'] == 0 and not receipt['formal_promotion']
    fit_order = receipt['fit_order_ids']
    assert hashlib.sha256(json.dumps(fit_order).encode()).hexdigest() == receipt['fit_order_sha256']
    assert len(fit_order) == 26747
    assert Counter(fit_order) == Counter(manifest['row_ids']['fit'])
    assert receipt['baseline_matches_protected_reference']
    assert receipt['native_warmup_forwards'] == 0 and receipt['shared_gradient_relative_l2_limit'] == 1e-4
    gradient_raw = (directory / 'first_batch_gradient_comparison.json').read_bytes()
    assert hashlib.sha256(gradient_raw).hexdigest() == receipt['first_batch_gradient_comparison_sha256']
    assert all(row['passed'] and row['relative_l2'] <= 1e-4
               for row in json.loads(gradient_raw)['shared_gradient_comparisons'].values())
    assert receipt['fit_rows'] == 26747 and receipt['fit_scenes'] == 413 and receipt['epochs'] == 1
    assert receipt['learning_rates'] == {'native_shared': 1e-5, 'sparse_shared': 1e-5, 'sparse_new': 1e-4}
    point_raw = (directory / 'fit_point_batches.json').read_bytes()
    assert hashlib.sha256(point_raw).hexdigest() == receipt['fit_point_batches_sha256']
    point_records = json.loads(point_raw)
    assert len(point_records) == 6687
    assert [r['step'] for r in point_records] == list(range(1, 6688))
    assert [row_id for r in point_records for row_id in r['row_ids']] == fit_order
    assert all(len(r['row_ids']) == len(r['point_cloud_sha256']) == 4 for r in point_records[:-1])
    assert len(point_records[-1]['row_ids']) == len(point_records[-1]['point_cloud_sha256']) == 3
    baseline_raw = (directory / 'baseline_rows.json').read_bytes()
    terminal_raw = (directory / 'terminal_rows.json').read_bytes()
    assert hashlib.sha256(baseline_raw).hexdigest() == receipt['baseline_rows_sha256']
    assert hashlib.sha256(terminal_raw).hexdigest() == receipt['terminal_rows_sha256']
    baseline, terminal = json.loads(baseline_raw), json.loads(terminal_raw)
    assert len(baseline) == len(terminal) == 6172
    assert [row['id'] for row in baseline] == manifest['row_ids']['holdout']
    require_identity(baseline, terminal)
    comparisons = {'sparse_minus_terminal_native': compare(terminal, terminal),
                   'sparse_minus_protected_start': compare(baseline, terminal)}
    cohort_start, cohort_end = good_box_cohort(baseline, terminal)
    diagnostics = {}
    for name, start, end, field in [
            ('selected_rec_query', baseline, terminal, 'rec_query_mask_iou'),
            ('legal_best_box_query_on_start_good_box_rows', cohort_start, cohort_end,
             'legal_box_oracle_query_mask_iou')]:
        diagnostics[name] = {
            'is_quality_gate': False,
            'metrics': {stage: {arm: metrics(rows, arm, field) for arm in ['native', 'sparse']}
                        for stage, rows in [('baseline', start), ('terminal', end)]},
            'comparisons': {'sparse_minus_terminal_native': compare(end, end, field),
                            'sparse_minus_protected_start': compare(start, end, field)}}
    rec = [row['native']['rec_box_iou'] for row in baseline]
    return {'receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(),
            'manifest_sha256': receipt['manifest_sha256'], 'integrity_pass': True,
            'metrics': {stage: {arm: metrics(rows, arm) for arm in ['native', 'sparse']}
                        for stage, rows in [('baseline', baseline), ('terminal', terminal)]},
            'fixed_rec_hits': {suffix: sum(value is not None and value > threshold for value in rec)
                               for threshold, suffix in [(.25, '025'), (.5, '050')]},
            'missing_legal_rec_rows': sum(value is None for value in rec),
            'comparisons': comparisons, 'fixed_screen_pass': passes_fixed_screen(comparisons),
            'conditional_mask_diagnostics': diagnostics,
            'formal_rows': 0, 'formal_promotion': False,
            'heldout_backbone_has_seen_scenes': True,
            'equal_parameter_capacity_control': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    result = verify_terminal_run(options.directory)
    with options.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'fixed_screen_pass': result['fixed_screen_pass'], 'metrics': result['metrics']}))
