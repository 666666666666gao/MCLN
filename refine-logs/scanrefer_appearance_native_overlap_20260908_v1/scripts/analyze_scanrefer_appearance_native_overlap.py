"""Attribute saved native endpoint boxes to annotated geometry, without rerunning a model."""

import argparse
import csv
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

from scripts.analyze_scanrefer_instance_overlap import (
    CATEGORIES, attribute_overlap, counts, sha, transitions,
)
from scripts.audit_scanrefer_stage_diagnostic import box_iou


ARMS = ('initial', 'control', 'appearance')


def annotate_row(row, annotation, geometry):
    """Recover the target from annotation order, then verify its actual geometry."""
    assert annotation['scene_id'] == row['scan_id']
    assert row['point_sha256'] == geometry['point_sha256']
    target_id = int(annotation['object_id'])
    slot = geometry['object_ids'].index(target_id)
    boxes = np.asarray(geometry['boxes'], dtype=np.float64).copy()
    assert np.allclose(boxes[slot], row['root_box'], atol=1e-6, rtol=0)
    boxes[slot] = row['root_box']
    ious = box_iou(np.asarray(row['selected_box'])[None, :], boxes)
    result = attribute_overlap(ious, geometry['object_ids'], geometry['labels'], target_id)
    assert abs(result['root_iou'] - row['rec_iou']) < 1e-5
    for threshold in (.25, .5):
        assert (result['root_iou'] > threshold) == (row['rec_iou'] > threshold)
    result.update({
        'row_id': row['row_id'], 'scan_id': row['scan_id'], 'target_id': target_id,
        'target_label': geometry['labels'][slot], 'ann_id': annotation['ann_id'],
        'utterance': ' '.join(annotation['token']), 'selected_box': row['selected_box'],
        'root_box': row['root_box'], 'selected_query': row['selected_query'],
    })
    return result


def analyze(directory):
    started = time.time()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    manifest = json.loads((directory/'input_manifest.json').read_bytes())
    for name, digest in manifest['input_files'].items():
        assert sha(name) == digest, name
    sys.path.append(manifest['model_source'])
    with Path(manifest['scene_pickle']).open('rb') as stream:
        assert pickle.load(stream) == 1
        scenes = pickle.load(stream)
    print('NATIVE OVERLAP DATA LOADED', flush=True)
    scene_ids = set(Path(manifest['annotation_scene_list']).read_text().splitlines())
    raw_annotations = json.loads(Path(manifest['annotations']).read_bytes())
    annotations = [a for a in raw_annotations if a['scene_id'] in scene_ids]
    assert len(annotations) == 36665
    native = json.loads(Path(manifest['native_rows']).read_bytes())
    receipt = json.loads(Path(manifest['native_receipt']).read_bytes())
    split = json.loads(Path(manifest['split_protocol']).read_bytes())
    assert receipt['status'] == 'complete' and receipt['rows_per_arm'] == 6887
    assert sha(manifest['native_rows']) == receipt['rows_sha256']
    assert set(native) == set(ARMS)
    assert [r['row_id'] for r in native['initial']] == split['row_ids']['holdout']
    needed_scenes = sorted({r['scan_id'] for r in native['initial']})
    geometry = {}
    for scene_id in needed_scenes:
        scene = scenes[scene_id]
        boxes, labels, point_counts = [], [], []
        for object_id, obj in enumerate(scene.three_d_objects):
            assert len(obj['points']) > 0
            bounds = scene.get_object_bbox(object_id).reshape(-1)
            boxes.append(np.concatenate(((bounds[:3] + bounds[3:])*.5, bounds[3:] - bounds[:3])).tolist())
            labels.append(scene.get_object_instance_label(object_id))
            point_counts.append(len(obj['points']))
        cloud = np.concatenate((scene.pc, scene.color - np.array([109.8, 97.2, 83.8])/256), axis=1).astype(np.float32)
        geometry[scene_id] = {
            'object_ids': list(range(len(boxes))), 'boxes': boxes, 'labels': labels,
            'point_counts': point_counts,
            'point_sha256': hashlib.sha256(cloud.tobytes()).hexdigest(),
        }
    del scenes
    analyzed = {}
    for arm in ARMS:
        assert len(native[arm]) == 6887
        analyzed[arm] = []
        for index, row in enumerate(native[arm]):
            for key in ('row_id', 'scan_id', 'physical_space', 'point_sha256', 'root_box'):
                assert row[key] == native['initial'][index][key]
            analyzed[arm].append(annotate_row(row, annotations[row['row_id']], geometry[row['scan_id']]))
        print('NATIVE OVERLAP ARM COMPLETE ' + arm, flush=True)
    summary = {
        'status': 'complete', 'rows_per_arm': 6887, 'scenes': len(geometry),
        'annotation_rows': len(annotations), 'formal_rows': 0, 'gpu_forwards': 0,
        'optimizer_steps': 0, 'previous_backbone_saw_module_holdout': True,
        'semantic_instance_identity_proven': False, 'used_for_tuning_or_promotion': False,
        'arms': {arm: counts(analyzed[arm]) for arm in ARMS},
        'native_comparisons': {
            arm: transitions(analyzed[arm], analyzed['appearance']) for arm in ('initial', 'control')
        },
        'limitations': 'Maximum annotated box overlap is a geometric proxy, not semantic identity. Raw instance labels are used; structural and nested objects remain in the comparison. Full-system rows lack selected boxes, so system-stage geometric attribution is unavailable.',
    }
    for arm in ARMS:
        for suffix in ('025', '050'):
            assert summary['arms'][arm][suffix]['hits'] == receipt['metrics'][arm]['hits'+suffix]
    with Path(manifest['paired_rows']).open() as stream:
        paired = list(csv.DictReader(stream))
    assert len(paired) == 6887
    for pair, row in zip(paired, native['initial']):
        assert int(pair['row_id']) == row['row_id'] and pair['point_sha256'] == row['point_sha256']
    summary['full_system_damage_native_geometry'] = {}
    for arm in ('initial', 'control'):
        summary['full_system_damage_native_geometry'][arm] = {}
        for threshold in (.25, .5):
            selected = [analyzed['appearance'][i] for i, pair in enumerate(paired)
                        if float(pair[arm+'_system_iou']) > threshold >= float(pair['appearance_system_iou'])]
            summary['full_system_damage_native_geometry'][arm][str(threshold)] = counts(selected)
    for name, digest in manifest['input_files'].items():
        assert sha(name) == digest, name
    rows_raw = json.dumps({'scene_geometry': geometry, 'arms': analyzed}, separators=(',', ':'), allow_nan=False).encode()
    with (directory/'overlap_rows.json.gz').open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            compressed.write(rows_raw)
    summary.update({
        'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'elapsed_seconds': time.time()-started,
        'input_manifest_sha256': sha(directory/'input_manifest.json'),
        'rows_sha256': sha(directory/'overlap_rows.json.gz'),
        'all_input_hashes_unchanged': True, 'all_point_hashes_and_root_boxes_verified': True,
    })
    with (directory/'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('NATIVE OVERLAP COMPLETE ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    analyze(parser.parse_args().directory.resolve())
