"""Evaluate the fixed M5 endpoint and paired scene uncertainty."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(rows, arm):
    values = [row[arm] for row in rows]
    result = {'rows': len(rows), 'mask_mean_iou': sum(row['mask_iou'] for row in values) / len(rows),
              'missing_legal_rec_rows': sum(row['rec_box_iou'] is None for row in values)}
    for threshold, suffix in [(.25, '025'), (.5, '050')]:
        result['mask_hits_' + suffix] = sum(row['mask_iou'] > threshold for row in values)
        result['rec_hits_' + suffix] = sum(row['rec_box_iou'] is not None and row['rec_box_iou'] > threshold for row in values)
    return result


def compare(reference, candidate, reference_arm):
    scenes = {}
    changes = []
    for a, b in zip(reference, candidate):
        assert a['id'] == b['id'] and a['scan_id'] == b['scan_id']
        old, new = a[reference_arm]['mask_iou'], b['nearest']['mask_iou']
        delta = [new - old, int(new > .25) - int(old > .25), int(new > .5) - int(old > .5)]
        scene = scenes.setdefault(a['scan_id'], {'rows':0, 'delta_mask_iou_sum':0., 'net025':0, 'net050':0})
        scene['rows'] += 1
        scene['delta_mask_iou_sum'] += delta[0]
        scene['net025'] += delta[1]
        scene['net050'] += delta[2]
        changes.append((old, new))
    fixes = {}
    for threshold, suffix in [(.25, '025'), (.5, '050')]:
        fixed = sum(a <= threshold < b for a, b in changes)
        broken = sum(b <= threshold < a for a, b in changes)
        fixes[suffix] = {'fixes':fixed, 'breaks':broken, 'net':fixed-broken}
    cluster_values = np.asarray([[v['rows'], v['delta_mask_iou_sum'], v['net025'], v['net050']] for v in scenes.values()])
    rng = np.random.RandomState(0)
    draws = cluster_values[rng.randint(0, len(scenes), (2000, len(scenes)))].sum(axis=1)
    bounds = np.percentile(draws[:, 1:] / draws[:, :1], [2.5, 97.5], axis=0)
    mean = sum(b-a for a,b in changes) / len(changes)
    return {'delta_mask_mean_iou':mean, 'thresholds':fixes, 'scene_count':len(scenes), 'per_scene':scenes,
            'paired_scene_bootstrap_95_ci':{'delta_mask_mean_iou':bounds[:,0].tolist(),
                                            'delta_mask_acc025':bounds[:,1].tolist(), 'delta_mask_acc050':bounds[:,2].tolist()},
            'fixed_screen_pass':mean >= .002 and all(value['net'] >= 0 for value in fixes.values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    raw = (options.directory / 'receipt.json').read_bytes()
    receipt = json.loads(raw)
    assert receipt['status'] == 'complete' and receipt['optimizer_steps_per_arm'] == 1024
    assert receipt['frozen_parameters_and_buffers_unchanged'] and receipt['source_data_and_parent_checkpoint_unchanged']
    assert receipt['grounding_and_query_selection_exactly_equal_to_start']
    baseline_raw = (options.directory / 'baseline_rows.json').read_bytes()
    terminal_raw = (options.directory / 'terminal_rows.json').read_bytes()
    assert hashlib.sha256(baseline_raw).hexdigest() == receipt['baseline_rows_sha256']
    assert hashlib.sha256(terminal_raw).hexdigest() == receipt['terminal_rows_sha256']
    baseline, terminal = json.loads(baseline_raw), json.loads(terminal_raw)
    assert len(baseline) == len(terminal) == 6172
    for a,b in zip(baseline,terminal):
        for name in ['id','scan_id','grounding_sha256','input_point_sha256']:
            assert a[name] == b[name], (a['id'],name)
        for arm in ['native','nearest']:
            for name in ['rec_query','rec_box_iou','mask_query']:
                assert a[arm][name] == b[arm][name], (a['id'],arm,name)
    comparisons = {'terminal_nearest_minus_terminal_native':compare(terminal, terminal, 'native'),
                   'terminal_nearest_minus_protected_start':compare(baseline, terminal, 'native')}
    result = {'receipt_sha256':hashlib.sha256(raw).hexdigest(), 'metrics':{
                stage:{arm:metrics(rows,arm) for arm in ['native','nearest']}
                for stage,rows in [('baseline',baseline),('terminal',terminal)]},
              'comparisons':comparisons, 'fixed_screen_pass':all(x['fixed_screen_pass'] for x in comparisons.values()),
              'integrity_pass':True, 'formal_rows':0, 'formal_promotion':False,
              'heldout_backbone_has_seen_scenes':True}
    with options.output.open('x', encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'fixed_screen_pass':result['fixed_screen_pass'], 'metrics':result['metrics']}))
