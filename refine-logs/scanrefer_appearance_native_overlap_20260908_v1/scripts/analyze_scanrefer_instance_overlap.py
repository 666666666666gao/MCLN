"""Saved-output geometry attribution; nearest GT box is not semantic identity."""

import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from scripts.audit_scanrefer_stage_diagnostic import ARMS, STAGES, box_iou


TIE_ATOL = 1e-6
DERIVED = ('geometry_query_native', 'final_query_native')
CATEGORIES = ('root_unique_max', 'other_same_label_unique_max',
              'other_different_label_unique_max', 'tied_max', 'no_overlap')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def attribute_overlap(ious, object_ids, labels, target_id):
    """Classify geometry only, without using the REC acceptance thresholds."""
    ious = np.asarray(ious)
    assert np.isfinite(ious).all() and len(ious) == len(object_ids) == len(labels)
    root = object_ids.index(target_id)
    best = float(ious.max())
    ties = np.flatnonzero(np.abs(ious - best) <= TIE_ATOL).tolist()
    winner = int(ious.argmax())
    if best == 0:
        category = 'no_overlap'
    elif len(ties) != 1:
        category = 'tied_max'
    elif winner == root:
        category = 'root_unique_max'
    elif labels[winner] == labels[root]:
        category = 'other_same_label_unique_max'
    else:
        category = 'other_different_label_unique_max'
    others = [float(value) for index, value in enumerate(ious) if index != root]
    return {'category': category, 'root_iou': float(ious[root]), 'max_iou': best,
            'best_object_id': object_ids[winner] if category not in ('no_overlap', 'tied_max') else None,
            'best_other_iou': max(others),
            'tied_object_ids': [object_ids[index] for index in ties] if category == 'tied_max' else []}


def stages_with_original_boxes(row):
    result = dict(row['stages'])
    for name, parent in zip(DERIVED, ('geometry', 'v99_final')):
        query = row['stages'][parent]['query_index']
        assert row['top16_query_indices'].count(query) == 1
        slot = row['top16_query_indices'].index(query)
        assert row['top16_valid'][slot]
        result[name] = {'box': row['top16_boxes'][slot], 'query_index': query}
    return result


def counts(records):
    result = {'rows': len(records), 'categories': dict(Counter(r['category'] for r in records))}
    for name, threshold in (('025', .25), ('050', .5)):
        failed = [r for r in records if r['root_iou'] <= threshold]
        result[name] = {'hits': len(records) - len(failed), 'failures': len(failed),
                        'failure_categories': {c: sum(r['category'] == c for r in failed) for c in CATEGORIES}}
    return result


def transitions(before, after):
    assert len(before) == len(after)
    result = {'category_transitions': dict(Counter(a['category'] + '->' + b['category'] for a, b in zip(before, after)))}
    for name, threshold in (('025', .25), ('050', .5)):
        repaired = [(a, b) for a, b in zip(before, after) if a['root_iou'] <= threshold < b['root_iou']]
        broken = [(a, b) for a, b in zip(before, after) if b['root_iou'] <= threshold < a['root_iou']]
        result[name] = {'repairs': len(repaired), 'breaks': len(broken), 'net': len(repaired) - len(broken),
                        'repair_categories': dict(Counter(a['category'] + '->' + b['category'] for a, b in repaired)),
                        'break_categories': dict(Counter(a['category'] + '->' + b['category'] for a, b in broken))}
    return result


def analyze(directory):
    started = time.time()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    manifest = json.loads((directory / 'input_manifest.json').read_bytes())
    for path, digest in manifest['input_files'].items():
        assert sha(path) == digest, path
    stage_path = Path(manifest['stage_rows'])
    source = Path(manifest['scan_source'])
    sys.path.append(str(source))
    with Path(manifest['scene_pickle']).open('rb') as stream:
        assert pickle.load(stream) == 1
        scenes = pickle.load(stream)
    source_rows = json.loads(stage_path.read_bytes())
    assert set(source_rows) == set(ARMS)
    scene_ids = sorted({r['scan_id'] for r in source_rows[ARMS[0]]})
    geometry = {}
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        boxes, labels, point_counts = [], [], []
        for oid, obj in enumerate(scene.three_d_objects):
            assert len(obj['points']) > 0, (scene_id, oid)
            bounds = scene.get_object_bbox(oid).reshape(-1)
            boxes.append(np.concatenate(((bounds[:3] + bounds[3:]) * .5, bounds[3:] - bounds[:3])).tolist())
            labels.append(scene.get_object_instance_label(oid))
            point_counts.append(len(obj['points']))
        cloud = np.concatenate((scene.pc, scene.color - np.array([109.8, 97.2, 83.8]) / 256), axis=1).astype(np.float32)
        geometry[scene_id] = {'object_ids': list(range(len(boxes))), 'boxes': boxes, 'labels': labels,
                              'point_counts': point_counts, 'point_sha256': hashlib.sha256(cloud.tobytes()).hexdigest()}
    analyzed = {arm: [] for arm in ARMS}
    max_difference = 0.
    for arm in ARMS:
        assert len(source_rows[arm]) == 9508
        for index, row in enumerate(source_rows[arm]):
            old = source_rows[ARMS[0]][index]
            for key in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'):
                assert row[key] == old[key]
            scene = geometry[row['scan_id']]
            assert row['point_sha256'] == scene['point_sha256']
            assert np.allclose(scene['boxes'][row['target_id']], row['root_box'], atol=1e-6, rtol=0)
            expanded = stages_with_original_boxes(row)
            gt_boxes = np.asarray(scene['boxes']).copy()
            # Preserve the exact native GT tensor after checking its reconstruction.
            gt_boxes[row['target_id']] = row['root_box']
            values = box_iou(np.asarray([expanded[s]['box'] for s in STAGES + DERIVED])[:, None, :],
                             gt_boxes[None, :, :])
            entry = {key: row[key] for key in ('row_id', 'scan_id', 'target_id')}
            entry['stages'] = {}
            for si, stage in enumerate(STAGES + DERIVED):
                result = attribute_overlap(values[si], scene['object_ids'], scene['labels'], row['target_id'])
                result['query_index'] = expanded[stage]['query_index']
                if stage in STAGES:
                    recorded = row['stages'][stage]['rec_iou']
                    max_difference = max(max_difference, abs(recorded - result['root_iou']))
                    assert abs(recorded - result['root_iou']) < 1e-5, (arm, index, stage)
                    for threshold in (.25, .5):
                        assert (recorded > threshold) == (result['root_iou'] > threshold)
                entry['stages'][stage] = result
            analyzed[arm].append(entry)
        print('OVERLAP ARM COMPLETE ' + arm, flush=True)
    summary = {'schema': 'mcln-scanrefer-instance-overlap-diagnostic-v1', 'arms': {}, 'paired_stages': {},
               'rows_per_arm': 9508, 'scenes': len(geometry), 'semantic_instance_identity_proven': False,
               'tie_atol': TIE_ATOL, 'gpu_forwards': 0, 'optimizer_steps': 0, 'new_formal_rows': 0,
               'used_for_promotion_or_tuning': False, 'source_root_iou_max_absolute_difference': max_difference,
               'limitations': 'Unique maximum overlap is a geometric proxy, not proof of semantic identity. All observed annotated objects including structural classes are compared. Reuses historical diagnostic outputs, not current training results.'}
    for arm in ARMS:
        view = {stage: [r['stages'][stage] for r in analyzed[arm]] for stage in STAGES + DERIVED}
        summary['arms'][arm] = {'stages': {s: counts(v) for s, v in view.items()},
                                 'same_query_geometry_effect': transitions(view['geometry_query_native'], view['geometry']),
                                 'same_query_final_effect': transitions(view['final_query_native'], view['v99_final'])}
    for stage in STAGES:
        summary['paired_stages'][stage] = transitions([r['stages'][stage] for r in analyzed[ARMS[0]]],
                                                      [r['stages'][stage] for r in analyzed[ARMS[1]]])
    with gzip.open(str(directory / 'overlap_rows.json.gz'), 'xb') as stream:
        stream.write(json.dumps({'scene_geometry': geometry, 'arms': analyzed}, separators=(',', ':')).encode())
    for path, digest in manifest['input_files'].items():
        assert sha(path) == digest, path
    summary.update({'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    'elapsed_seconds': time.time() - started, 'input_manifest_sha256': sha(directory / 'input_manifest.json'),
                    'rows_sha256': sha(directory / 'overlap_rows.json.gz'), 'analysis_sha256': sha(Path(__file__)),
                    'input_hashes_unchanged': True, 'all_scene_point_hashes_and_root_boxes_verified': True})
    with (directory / 'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print('OVERLAP SUMMARY ' + json.dumps({'elapsed_seconds': summary['elapsed_seconds'], 'scenes': summary['scenes'],
                                        'final': {a: summary['arms'][a]['stages']['v99_final'] for a in ARMS}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    analyze(parser.parse_args().directory)
