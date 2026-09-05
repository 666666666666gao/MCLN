"""Fixed L1 decision, paired threshold changes and scene uncertainty."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(rows, arm):
    values = [row[arm] for row in rows]
    result = {'rows': len(values), 'mask_mean_iou': sum(x['mask_iou'] for x in values)/len(values),
              'missing_legal_rec_rows': sum(x['rec_box_iou'] is None for x in values)}
    for threshold, suffix in [(.25, '025'), (.5, '050')]:
        result['rec_hits_'+suffix] = sum(x['rec_box_iou'] is not None and x['rec_box_iou'] > threshold for x in values)
        result['mask_hits_'+suffix] = sum(x['mask_iou'] > threshold for x in values)
        result['legal_full256_box_oracle_hits_'+suffix] = sum(
            x['legal_box_oracle_iou'] is not None and x['legal_box_oracle_iou'] > threshold for x in values)
    return result


def compare(reference, candidate, reference_arm):
    assert len(reference) == len(candidate)
    names = ['rec025', 'rec050', 'mask025', 'mask050']
    changes = {name: {'fixes':0, 'breaks':0} for name in names}
    scene_vectors = {}
    same_rec_query = changed_query_box_iou = 0
    for a, b in zip(reference, candidate):
        assert a['id'] == b['id'] and a['scan_id'] == b['scan_id']
        old, new = a[reference_arm], b['position']
        same_query = old['rec_query'] is not None and old['rec_query'] == new['rec_query']
        same_rec_query += same_query
        changed_query_box_iou += same_query and old['rec_box_iou'] != new['rec_box_iou']
        delta = [1]
        for field, prefix in [('rec_box_iou', 'rec'), ('mask_iou', 'mask')]:
            for threshold, suffix in [(.25, '025'), (.5, '050')]:
                first = old[field] is not None and old[field] > threshold
                second = new[field] is not None and new[field] > threshold
                changes[prefix+suffix]['fixes'] += not first and second
                changes[prefix+suffix]['breaks'] += first and not second
                delta.append(int(second)-int(first))
        delta.append(new['mask_iou']-old['mask_iou'])
        scene_vectors.setdefault(a['scan_id'], np.zeros(6))
        scene_vectors[a['scan_id']] += np.asarray(delta)
    for value in changes.values():
        value['net'] = value['fixes']-value['breaks']
    values = np.asarray(list(scene_vectors.values()))
    rng = np.random.RandomState(0)
    draws = values[rng.randint(0, len(values), (2000, len(values)))].sum(axis=1)
    bounds = np.percentile(draws[:, 1:]/draws[:, :1], [2.5, 97.5], axis=0)
    delta_mask = float(values[:, 5].sum()/len(reference))
    return {'thresholds':changes, 'delta_mask_mean_iou':delta_mask,
            'same_rec_query_rows':int(same_rec_query), 'same_query_changed_box_iou_rows':int(changed_query_box_iou),
            'per_scene':{name:dict(zip(['rows']+names+['mask_iou_delta_sum'], vector.tolist()))
                         for name, vector in scene_vectors.items()},
            'paired_scene_bootstrap_95_ci':{name:bounds[:, i].tolist() for i, name in enumerate(names+['mask_mean_iou'])},
            'fixed_screen_pass':changes['rec025']['net'] >= 10 and
                                all(changes[name]['net'] >= 0 for name in names[1:]) and delta_mask >= 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_directory', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    receipt_raw = (options.train_directory/'receipt.json').read_bytes()
    receipt = json.loads(receipt_raw)
    manifest_raw = (options.train_directory.parent/'input_manifest.json').read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == receipt['manifest_sha256']
    manifest = json.loads(manifest_raw)
    assert receipt['status'] == 'complete' and receipt['optimizer_steps_per_arm'] == 6687
    assert receipt['frozen_parameters_and_buffers_unchanged'] and receipt['source_data_and_parent_checkpoint_unchanged']
    baseline_raw = (options.train_directory/'baseline_rows.json').read_bytes()
    terminal_raw = (options.train_directory/'terminal_rows.json').read_bytes()
    assert hashlib.sha256(baseline_raw).hexdigest() == receipt['baseline_rows_sha256']
    assert hashlib.sha256(terminal_raw).hexdigest() == receipt['terminal_rows_sha256']
    baseline, terminal = json.loads(baseline_raw), json.loads(terminal_raw)
    assert len(baseline) == len(terminal) == 6172
    assert [x['id'] for x in baseline] == [x['id'] for x in terminal] == manifest['row_ids']['holdout']
    for a,b in zip(baseline, terminal):
        for key in ['id', 'scan_id', 'input_point_sha256']:
            assert a[key] == b[key], (a['id'], key)
    comparisons = {'position_minus_terminal_text':compare(terminal,terminal,'text'),
                   'position_minus_protected_start':compare(baseline,terminal,'protected')}
    result = {'receipt_sha256':hashlib.sha256(receipt_raw).hexdigest(), 'integrity_pass':True,
              'metrics':{'protected_start':metrics(baseline,'protected'),
                         'terminal_text':metrics(terminal,'text'), 'terminal_position':metrics(terminal,'position')},
              'comparisons':comparisons, 'fixed_screen_pass':all(value['fixed_screen_pass'] for value in comparisons.values()),
              'formal_rows':0, 'formal_promotion':False, 'heldout_backbone_has_seen_scenes':True}
    with options.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'fixed_screen_pass':result['fixed_screen_pass'],'metrics':result['metrics']}))
