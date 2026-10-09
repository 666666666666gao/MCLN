"""Independent, offline NumPy audit; writes only inside this audit directory."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).absolute().parent
ROOT = AUDIT.parent
TMP = ROOT.parent
OLD = TMP / 'pvground_mask_reference_20261006'
SOURCE = OLD / 'complete_fit/fused_mask_reference/initial_formal'
CURRENT = TMP / 'pvground_compressed_geometry_support_20261008'
N = 9508
NAMES = ('native_regression', 'mask_reference', 'fixed_half_blend')
THRESHOLDS = (.25, .5)


def read(path):
    return json.loads(path.read_bytes())


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (AUDIT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def overlap(predicted, target, precision):
    p = np.asarray(predicted, dtype=precision)
    g = np.asarray(target, dtype=precision)
    assert np.isfinite(p).all() and np.isfinite(g).all()
    assert (p[:, 3:] > 0).all() and (g[:, 3:] > 0).all()
    pmin, pmax = p[:, :3] - p[:, 3:] / 2, p[:, :3] + p[:, 3:] / 2
    gmin, gmax = g[:, :3] - g[:, 3:] / 2, g[:, :3] + g[:, 3:] / 2
    lengths = np.clip(np.minimum(pmax, gmax) - np.maximum(pmin, gmin), 0, None)
    intersection = np.prod(lengths, axis=1)
    denominator = np.prod(p[:, 3:], axis=1) + np.prod(g[:, 3:], axis=1) - intersection
    assert (denominator > 0).all()
    result = intersection / denominator
    assert np.isfinite(result).all() and (result >= 0).all()
    return result


def transitions(first, second):
    result = {}
    for t in THRESHOLDS:
        a, b = first > t, second > t
        result[str(t)] = dict(before_hits=int(a.sum()), after_hits=int(b.sum()),
                             repairs=int((~a & b).sum()), damages=int((a & ~b).sum()),
                             net=int(b.sum()) - int(a.sum()))
    return result


print(json.dumps(dict(event='start', python=sys.executable, version=sys.version,
                      numpy=np.__version__, offline=True, source_models_imported=False)), flush=True)
summary = read(ROOT / 'SUMMARY.json')
source_rows = read_lines(SOURCE / 'rows.jsonl')
selected = read_lines(ROOT / 'SELECTED_BOX_ROWS.jsonl')
receipt = read(SOURCE / 'receipt.json')
manifest = read(ROOT / 'ARRAY_MANIFEST.json')
intake = read(OLD / 'complete_fit/INTAKE.json')
intake_files = {item['name']: item for item in intake['files']}
bindings = {
    'source_rows_sha256': SOURCE / 'rows.jsonl',
    'source_receipt_sha256': SOURCE / 'receipt.json',
    'source_script_sha256': ROOT / 'analyze_fixed_query_boxes.py',
    'array_manifest_sha256': ROOT / 'ARRAY_MANIFEST.json',
    'output_rows_sha256': ROOT / 'SELECTED_BOX_ROWS.jsonl',
}
for key, path in bindings.items():
    assert sha(path) == summary[key], (key, str(path))
assert receipt['rows_sha256'] == sha(SOURCE / 'rows.jsonl')
assert receipt['status'] == 'pass' and receipt['stage'] == 'initial_formal'
assert receipt['rows'] == receipt['formal_rows'] == len(source_rows) == len(selected) == N
assert [r['row_id'] for r in source_rows] == [r['row_id'] for r in selected] == list(range(N))
assert len(manifest) == 1189 and len({r['path'] for r in manifest}) == 1189
assert sorted(str(p) for p in SOURCE.glob('batch_*.npz')) == sorted(r['path'] for r in manifest)
for row, output in zip(source_rows, selected):
    assert row['same_forward_geometry_exact'] is True
    assert row['native_head_calls'] == 1 and row['diagnostic_native_head_replay_calls'] == 0
    for key in ('row_id', 'scan_id', 'target_id', 'point_sha256', 'root_box'):
        assert row[key] == output[key], (row['row_id'], key)
    assert row['bbs']['query'] == output['query']
    assert row['bbs']['reference_valid'] == output['reference_valid']
    assert row['bbs']['reference_box'] == row['bbs']['box']

all_boxes = {name: [] for name in NAMES}
truths = []
verified_arrays = []
invalid_candidates = 0
invalid_selected = 0
score_ties = 0
offset = 0
for item in manifest:
    path = Path(item['path'])
    assert path.parent == SOURCE and path.name == f'batch_{offset:05d}.npz'
    digest = sha(path)
    assert path.stat().st_size == item['bytes'] and digest == item['sha256']
    archived = intake_files[path.relative_to(OLD / 'complete_fit').as_posix()]
    assert archived['sha256'] == digest and archived['bytes'] == path.stat().st_size
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'row_ids', 'original_prior', 'reference', 'final', 'scores', 'reference_valid', 'root_gt'}
        count = min(8, N-offset)
        np.testing.assert_array_equal(z['row_ids'], np.arange(offset, offset+count))
        truth = z['root_gt']
        assert truth.shape == (count, 6) and truth.dtype == np.float32
        np.testing.assert_array_equal(truth, np.asarray([r['root_box'] for r in source_rows[offset:offset+count]], dtype=np.float32))
        q = np.asarray([r['bbs']['query'] for r in source_rows[offset:offset+count]], dtype=np.int64)
        assert ((q >= 0) & (q < 256)).all()
        scores = z['scores']
        assert scores.shape == (count, 256) and np.isfinite(scores).all()
        winner_score = scores.max(axis=1)
        np.testing.assert_array_equal(scores[np.arange(count), q], winner_score)
        score_ties += int(((scores == winner_score[:, None]).sum(axis=1) > 1).sum())
        for key in ('original_prior', 'reference', 'final'):
            assert z[key].shape == (count, 256, 6) and z[key].dtype == np.float32
            assert np.isfinite(z[key]).all() and (z[key][..., 3:] > 0).all()
        np.testing.assert_array_equal(z['final'], z['reference'])
        valid = z['reference_valid']
        assert valid.shape == (count, 256) and valid.dtype == np.bool_
        invalid_candidates += int((~valid).sum())
        np.testing.assert_array_equal(z['reference'][~valid], z['original_prior'][~valid])
        selected_valid = valid[np.arange(count), q]
        np.testing.assert_array_equal(selected_valid, [r['reference_valid'] for r in selected[offset:offset+count]])
        invalid_selected += int((~selected_valid).sum())
        native = z['original_prior'][np.arange(count), q]
        mask = z['reference'][np.arange(count), q]
        blend = (native + mask) * np.float32(.5)
        for name, values in zip(NAMES, (native, mask, blend)):
            np.testing.assert_array_equal(values, np.asarray([r['boxes'][name] for r in selected[offset:offset+count]], dtype=np.float32))
            all_boxes[name].append(values.copy())
        np.testing.assert_array_equal(native, np.asarray([r['bbs']['coarse_box'] for r in source_rows[offset:offset+count]], dtype=np.float32))
        np.testing.assert_array_equal(mask, np.asarray([r['bbs']['reference_box'] for r in source_rows[offset:offset+count]], dtype=np.float32))
        truths.append(truth.copy())
    verified_arrays.append(dict(path=str(path), bytes=item['bytes'], sha256=digest,
                                source_intake_hash_matches=True))
    offset += count
assert offset == N
all_boxes = {name: np.concatenate(parts) for name, parts in all_boxes.items()}
truth = np.concatenate(truths)
values32 = {name: overlap(box, truth, np.float32) for name, box in all_boxes.items()}
values64 = {name: overlap(box, truth, np.float64) for name, box in all_boxes.items()}
table = []
precision_details = {}
gpu_details = {}
for name in NAMES:
    a, b = values32[name], values64[name]
    np.testing.assert_array_equal(a.astype(np.float64), [r['cpu_float32_iou'][name] for r in selected])
    np.testing.assert_array_equal(b, [r['cpu_float64_iou'][name] for r in selected])
    hits = [int((a > t).sum()) for t in THRESHOLDS]
    table.append(dict(condition=name, rows=N, hits25=hits[0], hits50=hits[1],
                      accuracy25=hits[0]/N*100, accuracy50=hits[1]/N*100,
                      float64_hits=[int((b > t).sum()) for t in THRESHOLDS],
                      float32_float64_threshold_flips={str(t): int(((a > t) != (b > t)).sum()) for t in THRESHOLDS},
                      meets_numerical_dual_gate=hits[0] >= 5658 and hits[1] >= 4850))
    precision_details[name] = dict(max_absolute_iou_difference=float(np.abs(a-b).max()),
        float32_iou_above_one=int((a > 1).sum()), float32_maximum_iou=float(a.max()),
        float64_iou_above_one=int((b > 1).sum()), float64_maximum_iou=float(b.max()),
        exact_threshold_equalities={str(t): int((a == t).sum()) for t in THRESHOLDS},
        minimum_float64_distance_to_threshold={str(t): float(np.abs(b-t).min()) for t in THRESHOLDS})
assert table == summary['table']
comparisons = dict(native_to_mask=transitions(values32['native_regression'], values32['mask_reference']),
                   native_to_half_blend=transitions(values32['native_regression'], values32['fixed_half_blend']),
                   mask_to_half_blend=transitions(values32['mask_reference'], values32['fixed_half_blend']))
assert comparisons == summary['comparisons']
for name, key, out_key in (('native_regression', 'coarse_iou', 'saved_gpu_native_iou'),
                          ('mask_reference', 'reference_iou', 'saved_gpu_mask_iou')):
    saved = np.asarray([r['bbs'][key] for r in source_rows])
    np.testing.assert_array_equal(saved, [r[out_key] for r in selected])
    flips = {str(t): int(((saved > t) != (values32[name] > t)).sum()) for t in THRESHOLDS}
    assert flips == summary['saved_gpu_cpu_float32_threshold_flips'][name]
    gpu_details[name] = dict(threshold_flips=flips, max_absolute_iou_difference=float(np.abs(saved-values32[name]).max()))
assert invalid_selected == summary['invalid_selected_references'] == 39
assert summary['independent_selected_iou_values_per_precision'] == N*3
assert [int((values32['mask_reference'] > t).sum()) for t in THRESHOLDS] == [receipt['metrics']['bbs']['rec_hits25'], receipt['metrics']['bbs']['rec_hits50']]
assert all(summary[k] is False for k in ('coefficient_search', 'retained_best_changed', 'live_training_changed', 'three_effective_contributions', 'full_goal_complete'))

old_spec = read(OLD / 'fused_mask_reference_spec.json')
current_spec = read(CURRENT / 'pair_spec.json')
imports_path = OLD / 'complete_fit/fused_mask_reference/imports.json'
imports = read(imports_path)
loader_path = TMP / 'pvground_g_p2_20261002/complete/source/joint_det_dataset.py'
evaluator_path = TMP / 'pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py'
helper_root = TMP / 'pvground_geometry_readback_20261004/revision2'
assert sha(loader_path) == imports['sha256']['src.joint_det_dataset']
assert sha(evaluator_path) == imports['sha256']['native_evaluator'] == old_spec['native_evaluator_sha256']
for name in ('native_root_bbs.py', 'readback_preflight_checks.py'):
    assert sha(helper_root / name) == old_spec['runner_files'][name]
assert sha(OLD / 'mask_reference.py') == old_spec['mask_reference_sha256']
assert sha(OLD / 'fused_mask_reference_spec.json') == read(OLD / 'complete_fit/fused_mask_reference/load.json')['spec_sha256']
for name in ('run_geometry_fit.py', 'mask_reference.py', 'fused_mask_reference_spec.json'):
    assert sha(OLD / name) == sha(OLD / 'complete_fit' / name) == intake_files[name]['sha256']

cross = read(ROOT / 'RETAINED_CROSS_FORWARD_HYPOTHESIS.json')
current_path = CURRENT / 'complete_fit/formal/rows.jsonl'
current_receipt_path = current_path.parent / 'receipt.json'
current_rows = read_lines(current_path)
current_receipt = read(current_receipt_path)
for key, path in (('source_current_rows_sha256', current_path), ('source_current_receipt_sha256', current_receipt_path),
                  ('source_archived_control_rows_sha256', ROOT / 'SELECTED_BOX_ROWS.jsonl'),
                  ('source_script_sha256', ROOT / 'compare_retained_with_archived_prior.py')):
    assert sha(path) == cross[key]
assert current_receipt['rows_sha256'] == sha(current_path)
assert current_receipt['status'] == 'pass' and current_receipt['stage'] == 'formal'
assert current_receipt['rows'] == current_receipt['formal_rows'] == len(current_rows) == N
for output, current in zip(selected, current_rows):
    for key in ('row_id', 'scan_id', 'target_id', 'point_sha256', 'root_box', 'query'):
        assert output[key] == current[key], (output['row_id'], key)
    assert output['boxes']['mask_reference'] == current['parent_box']
common_dependencies = {}
for key in ('base_terminal_sha256', 'checkpoint_sha256', 'env_spec_sha256', 'input_manifest',
            'native_evaluator_sha256', 'seed', 'batch_size', 'reference_mode', 'retained_hidden_prior_updates'):
    assert old_spec[key] == current_spec[key]
    common_dependencies[key] = old_spec[key]
assert old_spec['runner_files'] == current_spec['runner_files']
assert old_spec['source_port_sha256'] == current_spec['old_source_port_sha256']
assert old_spec['mask_reference_sha256'] == current_spec['new_runner_files']['mask_reference.py']
common_dependencies['runner_files'] = old_spec['runner_files']
common_dependencies['source_port_sha256_to_old_source_port_sha256'] = old_spec['source_port_sha256']
common_dependencies['mask_reference_sha256'] = old_spec['mask_reference_sha256']
corrected = np.asarray([r['arms']['content']['box'] for r in current_rows], dtype=np.float32)
cross_boxes = dict(retained_pure_mask=corrected,
                   archived_prior_half_blend_hypothesis=(all_boxes['native_regression'] + corrected)*np.float32(.5))
cross32 = {name: overlap(box, truth, np.float32) for name, box in cross_boxes.items()}
cross64 = {name: overlap(box, truth, np.float64) for name, box in cross_boxes.items()}
cross_conditions = {}
for name in cross_boxes:
    a, b = cross32[name], cross64[name]
    hits = [int((a > t).sum()) for t in THRESHOLDS]
    cross_conditions[name] = dict(hits=hits, percentages=[h/N*100 for h in hits],
        float64_hits=[int((b > t).sum()) for t in THRESHOLDS],
        float32_float64_threshold_flips=[int(((a>t)!=(b>t)).sum()) for t in THRESHOLDS],
        numerical_gate_only=hits[0] >= 5658 and hits[1] >= 4850)
assert cross_conditions == cross['conditions']
cross_comparison = transitions(cross32['retained_pure_mask'], cross32['archived_prior_half_blend_hypothesis'])
assert cross_comparison == cross['comparison']
assert cross['same_forward_comparison'] is False and cross['source_only_hypothesis_not_deployment_metric'] is True

reviewed_paths = list(bindings.values()) + [ROOT/'SUMMARY.json', ROOT/'AUDIT_REQUEST.txt', ROOT/'FINDINGS.md',
    OLD/'run_geometry_fit.py', OLD/'mask_reference.py', OLD/'fused_mask_reference_spec.json',
    OLD/'complete_fit/run_geometry_fit.py', OLD/'complete_fit/mask_reference.py', OLD/'complete_fit/fused_mask_reference_spec.json',
    OLD/'complete_fit/INTAKE.json', OLD/'complete_fit/fused_mask_reference/imports.json', OLD/'complete_fit/fused_mask_reference/load.json',
    OLD/'complete_fit/fused_mask_reference/initial_formal.log', OLD/'complete_fit/fit_status.json', OLD/'complete_fit/fit_controller.log',
    OLD/'complete_fit/fit_controller.exit', OLD/'complete_fit/controller.py',
    OLD/'complete_fit/fused_mask_reference/initial_formal.exit', loader_path, evaluator_path,
    TMP/'pvground_g_p2_20261002/complete/source/input_manifest.json',
    helper_root/'native_root_bbs.py', helper_root/'readback_preflight_checks.py',
    ROOT/'compare_retained_with_archived_prior.py', ROOT/'RETAINED_CROSS_FORWARD_HYPOTHESIS.json',
    CURRENT/'pair_spec.json', current_path, current_receipt_path,
    TMP/'pvground_novelty_20261009/EG-3DVG.txt', TMP/'pvground_novelty_20261009/NOVELTY_REVIEW.md']
file_hashes = {str(path): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in reviewed_paths}
write('REVIEWED_FILE_HASHES.json', file_hashes)
write('VERIFIED_ARRAY_HASHES.json', verified_arrays)
result = dict(status='PASS', independent_implementation=True, original_analyzer_imported_or_run=False,
    completed_cst=datetime.datetime.now().astimezone().isoformat(), python=sys.executable, numpy=np.__version__,
    rows=N, source_shards=len(manifest), source_array_bytes=sum(item['bytes'] for item in manifest),
    scenes=len({r['scan_id'] for r in source_rows}), scene_target_pairs=len({(r['scan_id'], r['target_id']) for r in source_rows}),
    candidates_per_row=256, all_candidate_final_equals_reference=True,
    source_intake_all_array_hashes_match=True, source_selected_rows_receipt_bound=True,
    score_tied_winner_rows=score_ties, invalid_candidates=invalid_candidates, invalid_selected=invalid_selected,
    source_invalid_prior_behavior_exact=True, table=table, comparisons=comparisons,
    precision_details=precision_details, saved_gpu_cpu_comparison=gpu_details,
    deterministic_threshold_comparisons=THRESHOLDS, metric_operator='>',
    float64_geometry_half_blend_hits=[int((overlap((all_boxes['native_regression'].astype(np.float64)+all_boxes['mask_reference'].astype(np.float64))*.5, truth, np.float64)>t).sum()) for t in THRESHOLDS],
    original_dataset_loader_hash_bound_to_import_receipt=True, native_evaluator_hash_bound_to_import_receipt=True,
    scope_new_forwards=0, scope_new_optimizer_steps=0, cross_forward=dict(status='PASS_FOR_ARITHMETIC_ONLY',
        row_input_gt_query_and_uncorrected_mask_alignment=N, common_core_spec_dependencies=common_dependencies,
        conditions=cross_conditions, comparison=cross_comparison, current_native_prior_unsaved=True,
        same_forward_current_model_validated=False, checkpoint_identity_not_freshly_loaded_or_hashed=True,
        weight_promotion_eligible=False), reviewed_file_hashes='REVIEWED_FILE_HASHES.json',
    array_hashes='VERIFIED_ARRAY_HASHES.json')
write('DETERMINISTIC_VERIFY.json', result)
print(json.dumps(result, ensure_ascii=False), flush=True)
