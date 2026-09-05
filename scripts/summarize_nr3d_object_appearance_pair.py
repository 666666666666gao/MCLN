"""Registered object-appearance screen, repairs/breaks and scene uncertainty."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(rows, arm):
    values = [row[arm] for row in rows]
    result = {'rows': len(values), 'mask_mean_iou': sum(x['mask_iou'] for x in values) / len(values),
              'missing_legal_rec_rows': sum(x['rec_box_iou'] is None for x in values)}
    for threshold, suffix in [(.25, '025'), (.5, '050')]:
        for field, prefix in [('rec_box_iou', 'rec'), ('mask_iou', 'mask'),
                              ('legal_box_oracle_iou', 'legal_full256_box_oracle')]:
            result[prefix + '_hits_' + suffix] = sum(x[field] is not None and x[field] > threshold for x in values)
    return result


def compare(reference, candidate):
    assert len(reference) == len(candidate) and reference
    names = ['rec025', 'rec050', 'mask025', 'mask050']
    changes = {name: {'fixes': 0, 'breaks': 0} for name in names}
    scenes = {}
    same_query = moved_same_query = 0
    for a, b in zip(reference, candidate):
        assert a['id'] == b['id'] and a['scan_id'] == b['scan_id']
        old, new = a['native'], b['appearance']
        equal_query = old['rec_query'] is not None and old['rec_query'] == new['rec_query']
        same_query += equal_query
        moved_same_query += equal_query and old['rec_box_iou'] != new['rec_box_iou']
        delta = [1]
        for field, prefix in [('rec_box_iou', 'rec'), ('mask_iou', 'mask')]:
            for threshold, suffix in [(.25, '025'), (.5, '050')]:
                first = old[field] is not None and old[field] > threshold
                second = new[field] is not None and new[field] > threshold
                changes[prefix + suffix]['fixes'] += not first and second
                changes[prefix + suffix]['breaks'] += first and not second
                delta.append(int(second) - int(first))
        delta.append(new['mask_iou'] - old['mask_iou'])
        scenes.setdefault(a['scan_id'], np.zeros(6))
        scenes[a['scan_id']] += np.asarray(delta)
    for value in changes.values():
        value['net'] = value['fixes'] - value['breaks']
    values = np.asarray(list(scenes.values()))
    rng = np.random.RandomState(0)
    draws = values[rng.randint(0, len(values), (2000, len(values)))].sum(axis=1)
    bounds = np.percentile(draws[:, 1:] / draws[:, :1], [2.5, 97.5], axis=0)
    delta_mask = float(values[:, 5].sum() / len(reference))
    return {'thresholds': changes, 'delta_mask_mean_iou': delta_mask,
            'same_rec_query_rows': int(same_query), 'same_query_changed_box_iou_rows': int(moved_same_query),
            'per_scene': {name: dict(zip(['rows'] + names + ['mask_iou_delta_sum'], vector.tolist()))
                          for name, vector in scenes.items()},
            'paired_scene_bootstrap_95_ci': {name: bounds[:, i].tolist() for i, name in enumerate(names + ['mask_mean_iou'])},
            'fixed_screen_pass': changes['rec025']['net'] >= 10 and
                                 all(changes[name]['net'] >= 0 for name in names[1:]) and delta_mask >= 0}


def verify_terminal_run(directory):
    receipt_raw = (directory / 'receipt.json').read_bytes()
    receipt = json.loads(receipt_raw)
    manifest_raw = (directory / 'input_manifest.json').read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == receipt['manifest_sha256']
    manifest = json.loads(manifest_raw)
    assert receipt['schema'] == 'mcln-nr3d-object-appearance-pair-v1'
    assert receipt['status'] == 'complete' and receipt['optimizer_steps_per_arm'] == 1024
    for key in ['frozen_parameters_and_buffers_unchanged', 'source_data_and_parent_checkpoint_unchanged',
                'early_queries_and_sampling_exactly_equal_to_start', 'text_mask_and_alpha_exactly_equal_to_start',
                'baseline_matches_protected_6172_rows']:
        assert receipt[key], key
    assert receipt['formal_rows'] == 0 and not receipt['formal_promotion']
    contents = {}
    for name in ['baseline_rows', 'terminal_rows', 'fit_point_batches']:
        raw = (directory / (name + '.json')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt[name + '_sha256']
        contents[name] = json.loads(raw)
    fit_order = receipt['fit_order_ids']
    assert len(fit_order) == 4096
    assert hashlib.sha256(json.dumps(fit_order).encode()).hexdigest() == receipt['fit_order_sha256']
    for epoch in range(2):
        assert sorted(fit_order[epoch * 2048:(epoch + 1) * 2048]) == manifest['row_ids']['fit']
    assert len(contents['fit_point_batches']) == 1024
    for index, batch in enumerate(contents['fit_point_batches']):
        assert batch['step'] == index + 1 and batch['row_ids'] == fit_order[index * 4:(index + 1) * 4]
        assert len(batch['point_tensor_sha256']) == 64
        int(batch['point_tensor_sha256'], 16)
        for key in ['early_queries_sha256', 'frozen_text_mask_alpha_sha256']:
            assert len(batch[key]) == 4 and all(len(value) == 64 for value in batch[key])
    baseline, terminal = contents['baseline_rows'], contents['terminal_rows']
    assert len(baseline) == len(terminal) == 6172
    assert [row['id'] for row in baseline] == [row['id'] for row in terminal] == manifest['row_ids']['holdout']
    for a, b in zip(baseline, terminal):
        assert a['native'] == a['appearance'], a['id']
        for key in ['id', 'scan_id', 'input_point_sha256', 'early_queries_sha256', 'frozen_text_mask_alpha_sha256']:
            assert a[key] == b[key], (a['id'], key)
    comparisons = {'appearance_minus_terminal_native': compare(terminal, terminal),
                   'appearance_minus_protected_start': compare(baseline, terminal)}
    return {'receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(), 'manifest_sha256': receipt['manifest_sha256'],
            'integrity_pass': True, 'metrics': {'protected_start': metrics(baseline, 'native'),
                  'terminal_native': metrics(terminal, 'native'), 'terminal_appearance': metrics(terminal, 'appearance')},
            'comparisons': comparisons, 'fixed_screen_pass': all(v['fixed_screen_pass'] for v in comparisons.values()),
            'formal_rows': 0, 'formal_promotion': False, 'heldout_backbone_has_seen_scenes': True,
            'equal_parameter_capacity_control': False, 'candidate_boxes_and_selections_allowed_to_change': True}


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
