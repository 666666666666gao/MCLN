"""CPU-only verification of the two saved preflight forward/step observations."""
import ast
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

started = time.perf_counter()
out = Path(__file__).resolve().parent
probe = out.parent
source = Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit')
hashes = json.loads((out / 'INPUT_HASHES.json').read_bytes())
new_hashes = {}


def read(path):
    raw = path.read_bytes()
    new_hashes[str(path)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return raw


def box_iou(box, target):
    box = np.asarray(box, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    box[3:] = np.maximum(box[3:], 1e-6)
    low = np.maximum(box[:3] - box[3:] / 2, target[:3] - target[3:] / 2)
    high = np.minimum(box[:3] + box[3:] / 2, target[:3] + target[3:] / 2)
    intersection = np.maximum(high - low, 0).prod()
    return float(intersection / (box[3:].prod() + target[3:].prod() - intersection))


reported = json.loads(read(probe / 'SAVED_TRAINING_WINNER_RESPONSIBILITY.json'))
manifest = json.loads(read(source / 'MANIFEST_STDOUT.json'))
manifest_map = {a['name']: a for a in manifest['files']}
preflight = json.loads(read(source / 'preflight.json'))
all_records = []
independent_records = []
binding_records = []
archive_records = []
points_per_step = {}
target_per_step = {}
max_ties = 0
minimum_abs_logit = {'own': float('inf'), 'fused': float('inf')}
for step in (1, 2):
    directory = source / ('preflight_support_step_' + str(step))
    receipt = json.loads(read(directory / 'receipt.json'))
    assert receipt['step'] == step and len(receipt['rows']) == 8
    assert receipt['status'] == 'ACTUAL_SUPPORT_INPUTS_CAPTURED'
    global_witness = preflight['witnesses'][step - 1]
    assert global_witness['step'] == step
    assert global_witness['support_input_witness'] == receipt['rows']
    assert len(list(directory.glob('*.npz'))) == 8
    points_per_step[step], target_per_step[step] = {}, {}
    for bid, witness in enumerate(receipt['rows']):
        path = directory / Path(witness['path']).name
        assert path.name == 'row_%02d.npz' % bid
        raw = read(path)
        h = new_hashes[str(path)]
        assert h['bytes'] == witness['bytes'] and h['sha256'] == witness['sha256']
        binding_records.append({'path': str(path), **h})
        with np.load(io.BytesIO(raw), allow_pickle=False) as a:
            q = witness['selected_query']
            assert a['scores'].shape == (256,) and np.isfinite(a['scores']).all()
            assert int(np.argmax(a['scores'])) == q
            max_ties += int(np.count_nonzero(a['scores'] == a['scores'][q]) > 1)
            assert a['actual_gt_boxes'].shape == (1, 6) and a['actual_gt_masks'].shape == (1, 50000)
            assert np.isin(a['actual_gt_masks'], [0, 1]).all()
            assert a['matched_queries'].shape == a['matched_gt_ids'].shape == (1,)
            assert a['matched_gt_ids'].tolist() == [0]
            assert global_witness['original_parent_assignment'][bid] == {
                'queries': a['matched_queries'].tolist(), 'actual_gt_ids': a['matched_gt_ids'].tolist()}
            assert a['observed_queries'].tolist() == sorted(set(a['matched_queries'].tolist() + [q]))
            assert witness['observed_queries'] == a['observed_queries'].tolist()
            assert int(witness['row_id']) == global_witness['rows'][bid]
            observed_index = int(np.where(a['observed_queries'] == q)[0][0])
            mapping = a['superpoints']
            own = a['observed_query_logits'][observed_index]
            text = a['observed_text_logits'][observed_index]
            alpha = a['alpha']
            assert own.dtype == text.dtype == alpha.dtype == np.float32
            assert mapping.shape == (50000,) and np.issubdtype(mapping.dtype, np.integer)
            assert mapping.min() >= 0 and mapping.max() < len(own)
            assert np.isfinite(own).all() and np.isfinite(text).all()
            fused = alpha * text + (1 - alpha) * own
            # The producer saved its actual sigmoid()>0.5 fused foreground.
            assert np.array_equal(fused > 0, a['foreground_sp'][q])
            truth = a['actual_gt_masks'][0].astype(bool)
            counts = np.bincount(mapping, minlength=len(own))
            gt_counts = np.bincount(mapping, weights=truth, minlength=len(own)).astype(np.int64)
            metrics = {}
            raw_counts = {}
            for name, logit in (('own', own), ('fused', fused)):
                foreground_sp = logit > 0
                intersection = int(gt_counts[foreground_sp].sum())
                union = int(counts[foreground_sp].sum()) + int(truth.sum()) - intersection
                metrics[name] = intersection / union
                raw_counts[name] = {'intersection_points': intersection, 'union_points': union}
                minimum_abs_logit[name] = min(minimum_abs_logit[name], float(np.abs(logit).min()))
                actual_float64_sigmoid = 1. / (1. + np.exp(-logit.astype(np.float64)))
                assert np.array_equal(actual_float64_sigmoid > .5, foreground_sp)
            is_matched = bool(np.any(a['matched_queries'] == q))
            record = dict(step=step, training_row_id=witness['row_id'], scan_id=witness['scene'],
                selected_query=q, selected_query_is_matched=is_matched, matched_queries=a['matched_queries'].tolist(),
                selected_own_mask_iou=metrics['own'], selected_fused_mask_iou=metrics['fused'],
                own_and_fused_masks_qualified=metrics['own'] > .5 and metrics['fused'] > .5, GT_count=1)
            all_records.append(record)
            point_hash = hashlib.sha256(a['raw_points'].tobytes()).hexdigest()
            assert point_hash == witness['raw_points_sha256']
            assert a['raw_points'].shape == (50000, 6)
            points_per_step[step][int(witness['row_id'])] = point_hash
            target_per_step[step][int(witness['row_id'])] = {
                'boxes': hashlib.sha256(a['actual_gt_boxes'].tobytes()).hexdigest(),
                'masks': hashlib.sha256(a['actual_gt_masks'].tobytes()).hexdigest(),
                'superpoints': hashlib.sha256(a['superpoints'].tobytes()).hexdigest()}
            native = np.concatenate([a['native_center'][q], a['native_size'][q]])
            mask = np.concatenate([a['mask_center'][q], a['mask_size'][q]])
            independent_records.append({**record, 'matched_gt_ids': a['matched_gt_ids'].tolist(),
                'point_mask_counts': raw_counts, 'raw_points_sha256': point_hash,
                'selected_native_endpoint_iou64': box_iou(native, a['actual_gt_boxes'][0]),
                'selected_mask_reference_endpoint_iou64': box_iou(mask, a['actual_gt_boxes'][0]),
                'final_B_box_saved_in_this_witness': False,
                'selected_direct_B_output_gradient_zero_by_original_assignment': not is_matched})

summary = dict(rows=16, batches=2,
    selected_query_unmatched=sum(not a['selected_query_is_matched'] for a in all_records),
    own_and_fused_qualified=sum(a['own_and_fused_masks_qualified'] for a in all_records),
    selected_query_unmatched_own_and_fused_qualified=sum(
        not a['selected_query_is_matched'] and a['own_and_fused_masks_qualified'] for a in all_records))
assert summary == reported['summary'] and all_records == reported['records']
assert binding_records == reported['input_bindings']
assert points_per_step[1] == points_per_step[2] and target_per_step[1] == target_per_step[2]
assert len(points_per_step[1]) == 8
script_path = probe / 'inspect_saved_training_winners.py'
tree = ast.parse(read(script_path).decode('utf-8-sig'), filename=str(script_path))
removed = [ast.unparse(node) for node in tree.body[-2:]]
assert '.write_text(' in removed[0] and removed[1].startswith('print(')
tree.body = tree.body[:-2]
scope = {'__file__': str(script_path), '__name__': 'audited_training_inspector'}
exec(compile(tree, str(script_path), 'exec'), scope)
assert scope['result'] == reported
prediction_path = probe / 'PREDICTION_AND_REPEATED_BATCH_DIAGNOSTIC.json'
prediction = json.loads(read(prediction_path))
prediction_script_path = probe / 'inspect_prediction_and_repeated_batch.py'
prediction_tree = ast.parse(read(prediction_script_path).decode('utf-8-sig'), filename=str(prediction_script_path))
prediction_removed = [ast.unparse(node) for node in prediction_tree.body[-2:]]
assert '.write_text(' in prediction_removed[0] and prediction_removed[1].startswith('print(')
prediction_tree.body = prediction_tree.body[:-2]
prediction_scope = {'__file__': str(prediction_script_path), '__name__': 'audited_prediction_inspector'}
exec(compile(prediction_tree, str(prediction_script_path), 'exec'), prediction_scope)
assert prediction_scope['result'] == prediction
assert prediction['training_observations'] == len(all_records) == 16
assert prediction['unique_training_rows'] == len(points_per_step[1]) == 8
assert prediction['original_observation_summary'] == summary
assert prediction['original_responsibility_sha256'] == new_hashes[str(probe / 'SAVED_TRAINING_WINNER_RESPONSIBILITY.json')]['sha256']
formal_rows = [json.loads(line) for line in read(source / 'formal/rows.jsonl').splitlines()]
gate = np.asarray([r['arms']['extremal_support']['axis_gate'] for r in formal_rows])
ordered_gates = np.sort(gate, axis=0)
quantiles = []
for q in prediction['quantile_positions']:
    coordinate = q * (len(gate) - 1)
    left, right = int(np.floor(coordinate)), int(np.ceil(coordinate))
    fraction = coordinate - left
    quantiles.append((ordered_gates[left] * (1 - fraction) + ordered_gates[right] * fraction).tolist())
assert np.max(np.abs(np.asarray(quantiles) - prediction['actual_axis_gate_quantiles'])) < 1e-15
assert np.max(np.abs(np.sqrt(np.mean(gate * gate, axis=0) - np.mean(gate, axis=0) ** 2) - prediction['actual_axis_gate_std'])) < 1e-12
assert int(np.sum(gate == 0)) == prediction['gate_axes_at_zero']
assert int(np.sum(gate == 1)) == prediction['gate_axes_at_one']
groups = []
for t in (.25, .5):
    states = [(r['controls']['native']['iou'] > t, r['controls']['mask']['iou'] > t,
               r['arms']['extremal_support']['iou'] > t) for r in formal_rows]
    groups.append(dict(threshold=t,
        native_qualified_reference_unqualified=sum(n and not m for n, m, f in states),
        native_qualified_reference_unqualified_final_unqualified=sum(n and not m and not f for n, m, f in states),
        reference_qualified_native_unqualified=sum(m and not n for n, m, f in states),
        reference_qualified_native_unqualified_final_unqualified=sum(m and not n and not f for n, m, f in states),
        final_repairs_reference=sum(not m and f for n, m, f in states),
        final_breaks_reference=sum(m and not f for n, m, f in states)))
assert groups == prediction['threshold_groups']
imports = json.loads(read(source / 'imports.json'))
bound_sources = {
    'models.losses': Path('C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/models/losses.py'),
    'src.joint_det_dataset': Path('C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/runtime_binding/dataset_source/src/joint_det_dataset.py')}
for module, path in bound_sources.items():
    read(path)
    assert new_hashes[str(path)]['sha256'] == imports['sha256'][module]
score_source = Path('C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/native_root_bbs.py')
read(score_source)
spec = json.loads(read(source / 'pair_spec.json'))
assert new_hashes[str(score_source)]['sha256'] == spec['runner_files']['native_root_bbs.py']
for path_text, entry in new_hashes.items():
    path = Path(path_text)
    if path.is_relative_to(source):
        relative = path.relative_to(source).as_posix()
        if relative in manifest_map:
            assert entry == {k: manifest_map[relative][k] for k in ('bytes', 'sha256')}
            archive_records.append(relative)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == v['sha256'] for p, v in new_hashes.items())
hashes.update(new_hashes)
(out / 'INPUT_HASHES.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
qualifying_unmatched = [r for r in independent_records if not r['selected_query_is_matched'] and r['own_and_fused_masks_qualified']]
result = dict(status='PASS', review_independence='deterministic', acceptance_status='accepted',
    summary=summary, observations=16, distinct_training_rows=8, distinct_scans=len({r['scan_id'] for r in all_records}),
    same_loaded_augmented_batch_reused_across_two_steps=True, raw_points_and_GT_identical_between_steps=True,
    rows_with_tied_max_scores=max_ties, minimum_abs_observed_selected_logit=minimum_abs_logit,
    records=independent_records, input_bindings=binding_records,
    matched_original_collection_manifest_entries=archive_records, probe_replay_exact=True,
    probe_removed_expressions=removed,
    unmatched_own_and_fused_qualified_mask_reference_endpoint_not_above_050=sum(
        r['selected_mask_reference_endpoint_iou64'] <= .5 for r in qualifying_unmatched),
    unmatched_own_and_fused_qualified_native_endpoint_not_above_050=sum(
        r['selected_native_endpoint_iou64'] <= .5 for r in qualifying_unmatched),
    no_full_dataset_prevalence_claim=True, no_later_MCLN_assignment_claim=True,
    no_final_B_error_classification_from_this_witness=True,
    repeated_batch_and_prediction_addendum=dict(status='PASS', exact_replay=True, independently_verified=True,
        sample_independence_wording_corrected=True, input_superpoints_also_equal=True,
        actual_gate_quantiles=quantiles, gate_axes_at_zero=prediction['gate_axes_at_zero'],
        gate_axes_at_one=prediction['gate_axes_at_one'], threshold_groups=groups,
        gate_range_is_not_parametric_range=True, endpoint_GT_groups_are_not_inference_rule=True),
    source_files_match_original_imports={module: str(path) for module, path in bound_sources.items()},
    native_root_bbs_source_matches_original_spec=str(score_source),
    elapsed_seconds=time.perf_counter() - started)
(out / 'TRAINING_WINNER_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ('input_bindings', 'probe_removed_expressions', 'matched_original_collection_manifest_entries')}, indent=2))
