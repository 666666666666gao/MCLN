"""Attribute saved fixed-training-scene boxes to annotated scene geometry on CPU."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    base = Path('/root/autodl-tmp')
    reference = base / 'mcln_pvground_normalization_intervention_20260917_v1'
    previous = base / 'mcln_scanrefer_appearance_native_overlap_20260908_v1'
    manifest = json.loads((previous / 'input_manifest.json').read_bytes())
    diagnostic = json.loads((reference / 'diagnostic.json').read_bytes())
    assert (reference / 'controller.exit').read_text().strip() == '0'
    assert diagnostic['status'] == 'complete'
    assert diagnostic['terminal_sha256'] == 'ce03188965491a82bcb1c5a6d26f590d3a243a01985457f220d5503c75b2fcf5'
    inputs = {}
    for name, key in [('rows.json', 'rows_sha256'), ('candidate_values.npz', 'arrays_sha256'),
                      ('input_selection.json', 'input_selection_sha256')]:
        path = reference / name
        assert sha(path) == diagnostic[key]
        inputs[str(path)] = diagnostic[key]
    for name in ['scene_pickle', 'annotations', 'split_protocol']:
        path = Path(manifest[name])
        assert sha(path) == manifest['input_files'][str(path)]
        inputs[str(path)] = sha(path)
    for name in ['analyze_scanrefer_instance_overlap.py', 'audit_scanrefer_stage_diagnostic.py']:
        path = previous / 'scripts' / name
        assert sha(path) == manifest['input_files'][str(path)]
    sys.path.insert(0, str(previous))
    sys.path.append(manifest['model_source'])
    from scripts.analyze_scanrefer_instance_overlap import attribute_overlap, counts
    from scripts.audit_scanrefer_stage_diagnostic import box_iou
    rows = json.loads((reference / 'rows.json').read_bytes())
    selected = json.loads((reference / 'input_selection.json').read_bytes())
    split = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert [r['row_id'] for r in rows] == [r['row_id'] for r in selected['rows']]
    assert len(rows) == len({r['scan_id'].split('_')[0] for r in rows}) == 128
    assert all(r['row_id'] in set(split['fit']) for r in rows)
    annos = json.loads(Path(manifest['annotations']).read_bytes())
    with Path(manifest['scene_pickle']).open('rb') as stream:
        assert pickle.load(stream) == 1
        scenes = pickle.load(stream)
    values = np.load(str(reference / 'candidate_values.npz'))
    records = []
    max_error = 0.
    for row in rows:
        row_id = row['row_id']
        anno = annos[row_id]
        assert anno['scene_id'] == row['scan_id']
        target = int(anno['object_id'])
        scene = scenes[row['scan_id']]
        cloud = np.concatenate((scene.pc, scene.color - np.array([109.8, 97.2, 83.8]) / 256), axis=1).astype(np.float32)
        assert hashlib.sha256(cloud.tobytes()).hexdigest() == row['point_sha256']
        scene_boxes, labels = [], []
        for oid in range(len(scene.three_d_objects)):
            bounds = scene.get_object_bbox(oid).reshape(-1)
            scene_boxes.append(np.concatenate(((bounds[:3] + bounds[3:]) / 2, bounds[3:] - bounds[:3])))
            labels.append(scene.get_object_instance_label(oid))
        scene_boxes = np.asarray(scene_boxes)
        assert np.allclose(scene_boxes[target], row['root_box'], atol=1e-6, rtol=0)
        scene_boxes[target] = row['root_box']
        candidates = values[str(row_id) + '_normal']
        assert candidates.shape == (256, 8) and np.isfinite(candidates).all()
        ious = box_iou(candidates[:, None, :6], scene_boxes[None])
        error = float(np.abs(ious[:, target] - candidates[:, 7]).max())
        max_error = max(max_error, error)
        assert error < 1e-5
        choice = row['normal']['selected']
        assert candidates[choice, 6] == candidates[:, 6].max()
        object_ids = list(range(len(scene_boxes)))
        entry = dict(row_id=row_id, scan_id=row['scan_id'], target_id=target, root_label=labels[target],
                     selected_query=choice, selected=attribute_overlap(ious[choice], object_ids, labels, target), alternatives={})
        for threshold in [.25, .5]:
            good = np.flatnonzero(ious[:, target] > threshold)
            key = str(threshold)
            if not len(good):
                entry['alternatives'][key] = dict(exists=False)
                continue
            best = int(good[np.argmax(candidates[good, 6])])
            tied = int(np.sum(candidates[good, 6] == candidates[best, 6]))
            entry['alternatives'][key] = dict(exists=True, query=best,
                good_score_ties=tied, rank_min=int(1 + (candidates[:, 6] > candidates[best, 6]).sum()),
                score_margin=float(candidates[best, 6] - candidates[choice, 6]),
                overlap=attribute_overlap(ious[best], object_ids, labels, target),
                selected_to_good_iou=float(box_iou(candidates[choice, :6], candidates[best, :6])))
        records.append(entry)
    result = dict(status='complete', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        rows=128, model_forwards=0, optimizer_steps=0, formal_rows=0,
        terminal_sha256=diagnostic['terminal_sha256'], input_files=inputs,
        selected_counts=counts([r['selected'] for r in records]), errors={}, root_iou_max_error=max_error,
        source_arm='normal only; no parent BN substitution',
        scope='Fixed preselected 128 fit scenes, no holdout. Maximum overlap is a geometry proxy, not semantic identity. No cross-query identity or causal regression claim. No inference rule changes.')
    for threshold in [.25, .5]:
        key = str(threshold)
        failed = [r for r in records if r['selected']['root_iou'] <= threshold]
        covered = [r for r in failed if r['alternatives'][key]['exists']]
        result['errors'][key] = dict(failures=len(failed), covered=len(covered),
            selected_categories=dict(Counter(r['selected']['category'] for r in failed)),
            covered_selected_categories=dict(Counter(r['selected']['category'] for r in covered)),
            good_categories=dict(Counter(r['alternatives'][key]['overlap']['category'] for r in covered)),
            both_root_unique_max=sum(r['selected']['category'] == r['alternatives'][key]['overlap']['category'] == 'root_unique_max' for r in covered),
            good_rank2=sum(r['alternatives'][key]['rank_min'] == 2 for r in covered))
    raw = (json.dumps(records, indent=2, allow_nan=False) + '\n').encode()
    with (output / 'rows.json').open('xb') as stream:
        stream.write(raw)
    result['rows_sha256'] = hashlib.sha256(raw).hexdigest()
    result['script_sha256'] = sha(Path(__file__))
    with (output / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
