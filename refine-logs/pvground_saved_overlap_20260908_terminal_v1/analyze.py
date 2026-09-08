"""Reuse verified scene geometry for CPU-only PV endpoint attribution."""
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(output):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    base = Path('/root/autodl-tmp')
    previous = base/'mcln_scanrefer_appearance_native_overlap_20260908_v1'
    previous_summary = json.loads((previous/'summary.json').read_bytes())
    previous_manifest = json.loads((previous/'input_manifest.json').read_bytes())
    assert previous_summary['all_point_hashes_and_root_boxes_verified']
    assert sha(previous/'input_manifest.json') == previous_summary['input_manifest_sha256']
    assert sha(previous/'overlap_rows.json.gz') == previous_summary['rows_sha256']
    inputs = {str(previous/name): sha(previous/name) for name in
              ['summary.json', 'input_manifest.json', 'overlap_rows.json.gz']}
    for name in ['analyze_scanrefer_instance_overlap.py', 'audit_scanrefer_stage_diagnostic.py']:
        path = previous/'scripts'/name
        assert sha(path) == previous_manifest['input_files'][str(path)]
        inputs[str(path)] = sha(path)
    sys.path.insert(0, str(previous))
    from scripts.analyze_scanrefer_instance_overlap import attribute_overlap, counts, transitions
    from scripts.audit_scanrefer_stage_diagnostic import box_iou
    geometry = json.loads(gzip.decompress((previous/'overlap_rows.json.gz').read_bytes()))['scene_geometry']
    locations = {
        'native_initial': ('vsaorder', 'initial'),
        'native_terminal': ('vsaorder', 'terminal'),
        'empty_pool_initial': ('emptypool', 'initial'),
        'empty_pool_terminal': ('emptypool', 'terminal'),
    }
    analyzed = {}; source_rows = {}; metrics = {}; maximum_error = 0.
    for name, (trial, stage) in locations.items():
        root = base/('mcln_pvground_scanrefer_finetune_20260908_'+trial+'_v1')/stage
        receipt = json.loads((root/'receipt.json').read_bytes())
        assert receipt['status'] == 'pass' and receipt['rows'] == 6887 and receipt['formal_rows'] == 0
        assert sha(root/'rows.jsonl') == receipt['rows_sha256']
        for filename in ['receipt.json', 'rows.jsonl']:
            inputs[str(root/filename)] = sha(root/filename)
        rows = [json.loads(line) for line in (root/'rows.jsonl').read_text().splitlines()]
        assert len(rows) == 6887
        source_rows[name] = rows; analyzed[name] = {mode: [] for mode in ['bbs', 'bbf']}
        for row in rows:
            scene = geometry[row['scan_id']]
            assert row['point_sha256'] == scene['point_sha256']
            slot = scene['object_ids'].index(row['target_id'])
            boxes = np.asarray(scene['boxes']).copy()
            assert np.allclose(boxes[slot], row['root_box'], atol=1e-6, rtol=0)
            boxes[slot] = row['root_box']
            for mode in ['bbs', 'bbf']:
                values = box_iou(np.asarray(row[mode]['box'])[None, :], boxes)
                result = attribute_overlap(values, scene['object_ids'], scene['labels'], row['target_id'])
                error = abs(result['root_iou']-row[mode]['iou'])
                maximum_error = max(maximum_error, error)
                assert error < 1e-5
                for threshold in [.25, .5]:
                    assert (result['root_iou'] > threshold) == (row[mode]['iou'] > threshold)
                result.update({key: row[key] for key in ['row_id', 'scan_id', 'target_id']})
                result['selected_query'] = row[mode]['query']
                analyzed[name][mode].append(result)
        metrics[name] = {mode: counts(records) for mode, records in analyzed[name].items()}
        for mode in ['bbs', 'bbf']:
            for suffix, key in [('025', 'rec_hits25'), ('050', 'rec_hits50')]:
                assert metrics[name][mode][suffix]['hits'] == receipt['metrics'][mode][key]
    comparisons = {}
    for before, after in [('native_initial', 'native_terminal'),
                          ('native_initial', 'empty_pool_initial'),
                          ('empty_pool_initial', 'empty_pool_terminal'),
                          ('native_terminal', 'empty_pool_terminal')]:
        for a, b in zip(source_rows[before], source_rows[after]):
            assert all(a[k] == b[k] for k in ['row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'])
        pair = before + '->' + after
        comparisons[pair] = {}
        for mode in ['bbs', 'bbf']:
            old, new = analyzed[before][mode], analyzed[after][mode]
            same_root = [(a, b) for a, b in zip(old, new)
                         if a['category'] == b['category'] == 'root_unique_max']
            comparisons[pair][mode] = transitions(old, new)
            comparisons[pair][mode]['both_root_unique_max'] = dict(
                rows=len(same_root), **transitions([a for a, b in same_root], [b for a, b in same_root]))
    assert all(sha(Path(path)) == digest for path, digest in inputs.items())
    result = dict(status='complete', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        rows_per_snapshot=6887, scenes=len({r['scan_id'] for r in source_rows['native_initial']}),
        model_forwards=0, optimizer_steps=0, new_formal_rows=0, training_changes=False,
        used_for_promotion_or_tuning=False, semantic_instance_identity_proven=False,
        all_point_hashes_and_root_boxes_verified=True, input_files=inputs,
        script_sha256=sha(Path(__file__)), root_iou_max_absolute_difference=maximum_error,
        snapshots=metrics, comparisons=comparisons,
        limitations='Maximum annotated box overlap is a geometric proxy, not semantic identity. Includes structural and nested objects; same-label means exact raw annotation label. Both-root comparisons do not separate reranking from regression changes. Initial snapshots and fixed training endpoints are reported separately. Backbone-seen module holdout, not unseen-scene generalization.')
    with (output/'overlap_rows.json.gz').open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            compressed.write(json.dumps(analyzed, separators=(',', ':'), allow_nan=False).encode())
    result['rows_sha256'] = sha(output/'overlap_rows.json.gz')
    with (output/'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], snapshots=metrics,
                         comparisons=comparisons, model_forwards=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    analyze(parser.parse_args().output)
