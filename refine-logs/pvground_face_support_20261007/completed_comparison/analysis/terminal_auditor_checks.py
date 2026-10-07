"""Independent terminal audit: stored artifacts only, no producer imports."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np

root = Path(__file__).resolve().parents[1]
complete = root / 'complete_fit'
parent_root = root.parent / 'pvground_mask_reference_20261006'
data_root = root.parent / 'pvground_g_p2_20261002'
arms = ('face_center', 'face_region')
box_names = ('original_prior', 'reference', 'final_face_center', 'final_face_region')
thresholds = (0.25, 0.5)
started = time.perf_counter()


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def tally(values):
    a = np.asarray(values)
    return {str(t): int(np.count_nonzero(a > t)) for t in thresholds}


def transitions(before, after):
    a, b = np.asarray(before), np.asarray(after)
    return {str(t): dict(before_hits=int((a > t).sum()), after_hits=int((b > t).sum()),
                        repairs=int(((a <= t) & (b > t)).sum()), damages=int(((a > t) & (b <= t)).sum()),
                        net=int((b > t).sum() - (a > t).sum())) for t in thresholds}


def independently_compute_overlap(boxes, ground_truth):
    # Explicit corner construction; no imports from either producer or analyzer.
    b = boxes.astype(np.float64)
    g = ground_truth.astype(np.float64)[:, None, :]
    b_min, b_max = b[:, :, :3] - b[:, :, 3:] * 0.5, b[:, :, :3] + b[:, :, 3:] * 0.5
    g_min, g_max = g[:, :, :3] - g[:, :, 3:] * 0.5, g[:, :, :3] + g[:, :, 3:] * 0.5
    widths = np.clip(np.minimum(b_max, g_max) - np.maximum(b_min, g_min), 0, None)
    intersection = widths[:, :, 0] * widths[:, :, 1] * widths[:, :, 2]
    b_volume = b[:, :, 3] * b[:, :, 4] * b[:, :, 5]
    g_volume = g[:, :, 3] * g[:, :, 4] * g[:, :, 5]
    result = intersection / (b_volume + g_volume - intersection)
    assert np.isfinite(result).all() and (result >= 0).all() and (result <= 1.000000001).all()
    return result


summary = read(root / 'analysis/SUMMARY.json')
spec = read(root / 'pair_spec.json')
intake = read(complete / 'INTAKE.json')
assert intake['status'] == 'CLOSED_FIT_ARTIFACTS_COLLECTED' and intake['weights_copied'] == 0
assert len(intake['files']) == intake['files_copied']
assert len({v['name'] for v in intake['files']}) == len(intake['files'])
for item in intake['files']:
    path = complete / item['name']
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], str(path)
assert sum(v['bytes'] for v in intake['files']) == intake['total_bytes']
for name in ('controller.py', 'run_face_support_pair.py', 'paired_geometry_loop.py', 'pair_spec.json',
             'face_region_box_refiner.py', 'face_support_model_factory.py', 'selected_mask_reference_factory.py',
             'mask_reference.py', 'query_supported_geometry.py'):
    assert sha(root / name) == sha(complete / name), name
for name in ('fit_controller.exit', 'initial_formal.exit', 'train.exit', 'formal.exit'):
    assert (complete / name).read_text().strip() == '0'
wait = read(root / 'fit_wait.json')
status = read(complete / 'fit_status.json')
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
assert wait['terminal']['status'] == status
assert status['status'] == 'complete' and status['completed_modes'] == ['initial_formal', 'train', 'formal']
assert status['exit_code'] == 0 and status['optimizer_steps_per_arm'] == 3723

source_locations = {
    'src.joint_det_dataset': data_root / 'complete/source/joint_det_dataset.py',
    'models.pv_ground': root.parent / 'pvground_geometry_readback_20261004/revision2/source_preview/PV-Ground/models/pv_ground.py',
    'models.losses': root.parent / 'pvground_runtime_bundle_20260908_v1/PV-Ground/models/losses.py',
    'main_utils': root.parent / 'pvground_runtime_bundle_20260908_v1/PV-Ground/main_utils.py',
    'prepare_data': root.parent / 'pvground_runtime_bundle_20260908_v1/PV-Ground/prepare_data.py',
    'native_evaluator': root.parent / 'pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py',
}
imports = read(complete / 'imports.json')
assert set(source_locations) == set(imports['sha256'])
for name, path in source_locations.items():
    assert sha(path) == imports['sha256'][name], name
for name, digest in spec['runner_files'].items():
    assert sha(root.parent / 'pvground_final_quality_20261005/runtime_bundle' / name) == digest, name
for key, name in (('sampler_sha256', 'face_region_box_refiner.py'), ('face_factory_sha256', 'face_support_model_factory.py'),
                  ('paired_loop_sha256', 'paired_geometry_loop.py'), ('selected_factory_sha256', 'selected_mask_reference_factory.py'),
                  ('mask_reference_sha256', 'mask_reference.py')):
    assert sha(root / name) == spec[key], key
assert sha(root / 'selected_mask_reference_factory.py') == sha(parent_root / 'postrun/selected_mask_reference_factory.py')
manifest = read(data_root / 'input_manifest.json')
assert sha(data_root / 'split_protocol.json') == manifest['split_protocol_sha256']
assert sha(data_root / 'complete/source/appearance_source_manifest.json') == manifest['source_manifest_sha256']
appearance = read(data_root / 'complete/source/appearance_source_manifest.json')
assert appearance['files']['src/joint_det_dataset.py'] == sha(source_locations['src.joint_det_dataset'])

fit = read(complete / 'receipt.json')
training = lines(complete / 'train.jsonl')
split = read(data_root / 'split_protocol.json')['row_ids']
ids = [v for record in training for v in record['rows']]
assert len(training) == 3723 and [v['step'] for v in training] == list(range(1, 3724))
assert len(ids) == len(set(ids)) == 29778 and Counter(ids) == Counter(split['fit'])
assert len(split['holdout']) == len(set(split['holdout'])) == 6887
assert not set(ids).intersection(split['holdout'])
assert [len(v['rows']) for v in training] == [8] * 3722 + [2]
assert sha(complete / 'train.jsonl') == fit['train_log_sha256']
assert fit['spec_sha256'] == sha(root / 'pair_spec.json')
for row in training:
    assert row['frozen_parent_forwards_per_batch'] == row['native_final_semantic_head_calls'] == 1
    assert row['zero_R_semantic_exact'] and row['shared_native_score_and_masks_exact']
    for arm in arms:
        r = row['arms'][arm]
        assert r['cross_head_gradients_absent'] and r['sampling_mode'] == arm
        assert all(math.isfinite(r[k]) for k in ('loss', 'native_loss', 'G_correction', 'matched_boundary_loss', 'extra_geometry_loss', 'gradient_norm'))
        assert r['gradient_norm'] > 0 and r['reference_keep_weight'] == 0
for arm in arms:
    restored = read(complete / arm / 'formal_restore.json')
    assert restored['status'] == 'pass' and restored['sampling_mode'] == arm
    assert restored['terminal_sha256'] == fit['terminal_sha256'][arm]
    assert restored['optimizer']['all_keys_moments_steps_and_groups_exact']
    assert restored['optimizer']['moment_and_step_states'] == 10
extra = {a: dict(candidate_occurrences=sum(r['arms'][a]['extra_counts']['extra_candidates'] for r in training),
                 outside_face_occurrences=sum(r['arms'][a]['extra_counts']['extra_boundary_outside'] for r in training)) for a in arms}
assert extra == summary['training_extra_scope']
parent = lines(parent_root / 'complete_fit/fused_mask_reference/initial_formal/rows.jsonl')
assert len(parent) == 9508 and tally([v['bbs']['iou'] for v in parent]) == {'0.25': 5598, '0.5': 4848}
parent_restore = read(parent_root / 'selected_candidate_CPU_restore.json')
assert parent_restore['receipt']['checkpoint_sha256'] == spec['selected_terminal_sha256']
assert parent_restore['receipt']['factory_sha256'] == sha(root / 'selected_mask_reference_factory.py')
assert parent_restore['receipt']['all_full_model_states_equal_to_original_construction']

stage_rows = {s: lines(complete / s / 'rows.jsonl') for s in ('initial_formal', 'formal')}
stage_checks = {}
for stage, rows in stage_rows.items():
    receipt = read(complete / stage / 'receipt.json')
    assert len(rows) == 9508 and [v['row_id'] for v in rows] == list(range(9508))
    assert sha(complete / stage / 'rows.jsonl') == receipt['rows_sha256']
    for old, row in zip(parent, rows):
        assert all(row[k] == old[k] for k in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'))
        assert row['frozen_parent_forwards'] == row['native_head_calls'] == 1
    stage_checks[stage] = dict(npz_count=0, rows=0, final_batch_rows=None, unique_scans=len({r['scan_id'] for r in rows}),
        unique_physical_scenes=len({r['scan_id'].split('_')[0] for r in rows}),
        top_score_tie_rows=0, selected_query_argmax_mismatches=0,
        coverage={k: {str(t): 0 for t in thresholds} for k in box_names},
        selected_cpu_gpu_label_flips={k: {str(t): 0 for t in thresholds} for k in box_names},
        selected_cpu_gpu_iou_max_error={k: 0.0 for k in box_names},
        oracle_label_mismatches={a: {str(t): 0 for t in thresholds} for a in arms},
        available_unselected={a: {str(t): 0 for t in thresholds} for a in arms},
        no_qualified={a: {str(t): 0 for t in thresholds} for a in arms},
        invalid_selected=0, invalid_all=0,
        arm_different_box_elements=0, arm_different_box_expressions=0, arm_box_max_absolute_difference=0.0,
        arm_full256_hit_label_mismatches={str(t): 0 for t in thresholds})

drift = {k: dict(different_elements=0, expressions_changed=0, maximum_absolute_difference=0.0,
                candidate_slots_changed=0, selected_slots_changed=0) for k in ('scores', 'original_prior', 'reference', 'reference_valid')}
schema = {'row_ids', 'root_gt', 'original_prior', 'reference', 'reference_valid', 'scores', 'final_face_center', 'final_face_region'}
paths = {s: sorted((complete / s).glob('batch_*.npz')) for s in stage_rows}
assert len(paths['initial_formal']) == len(paths['formal']) == 1189
for initial_path, formal_path in zip(paths['initial_formal'], paths['formal']):
    assert initial_path.name == formal_path.name
    batch_pair = {}
    for stage, path in (('initial_formal', initial_path), ('formal', formal_path)):
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == schema
            batch_pair[stage] = {k: z[k] for k in z.files}
        batch = batch_pair[stage]
        check = stage_checks[stage]
        offset = check['rows']
        n = len(batch['row_ids'])
        assert path.name == f'batch_{offset:05d}.npz' and n == min(8, 9508 - offset)
        np.testing.assert_array_equal(batch['row_ids'], np.arange(offset, offset + n))
        rows = stage_rows[stage][offset:offset + n]
        q = np.array([r['query'] for r in rows])
        idx = np.arange(n)
        np.testing.assert_array_equal(batch['root_gt'], np.array([r['root_box'] for r in rows]))
        assert batch['root_gt'].shape == (n, 6) and np.isfinite(batch['root_gt']).all()
        assert (batch['root_gt'][:, 3:] > 0).all()
        assert batch['scores'].shape == batch['reference_valid'].shape == (n, 256)
        assert batch['reference_valid'].dtype == np.bool_ and np.isfinite(batch['scores']).all()
        assert ((q >= 0) & (q < 256)).all()
        np.testing.assert_array_equal(batch['scores'][idx, q], batch['scores'].max(axis=1))
        check['top_score_tie_rows'] += int(((batch['scores'] == batch['scores'].max(axis=1)[:, None]).sum(axis=1) > 1).sum())
        check['selected_query_argmax_mismatches'] += int((q != np.argmax(batch['scores'], axis=1)).sum())
        per_box_iou = {}
        for name in box_names:
            boxes = batch[name]
            assert boxes.shape == (n, 256, 6) and np.isfinite(boxes).all() and (boxes[..., 3:] > 0).all()
            if name in ('original_prior', 'reference'):
                prefix = 'coarse' if name == 'original_prior' else 'reference'
                selected_boxes = [r[prefix + '_box'] for r in rows]
                gpu_iou = np.array([r[prefix + '_iou'] for r in rows])
            else:
                arm = name.removeprefix('final_')
                selected_boxes = [r['arms'][arm]['box'] for r in rows]
                gpu_iou = np.array([r['arms'][arm]['iou'] for r in rows])
            np.testing.assert_array_equal(boxes[idx, q], selected_boxes)
            overlap = independently_compute_overlap(boxes, batch['root_gt'])
            per_box_iou[name] = overlap
            check['selected_cpu_gpu_iou_max_error'][name] = max(check['selected_cpu_gpu_iou_max_error'][name], float(np.abs(overlap[idx, q] - gpu_iou).max()))
            for t in thresholds:
                key = str(t)
                covered = (overlap > t).any(axis=1)
                selected_hit = overlap[idx, q] > t
                check['coverage'][name][key] += int(covered.sum())
                check['selected_cpu_gpu_label_flips'][name][key] += int((selected_hit != (gpu_iou > t)).sum())
                if name.startswith('final_'):
                    label = 'oracle25' if t == 0.25 else 'oracle50'
                    check['oracle_label_mismatches'][arm][key] += int((covered != np.array([r['arms'][arm][label][-1] for r in rows], bool)).sum())
                    check['available_unselected'][arm][key] += int((covered & ~selected_hit).sum())
                    check['no_qualified'][arm][key] += int((~covered).sum())
            if stage == 'initial_formal' and name.startswith('final_'):
                np.testing.assert_array_equal(boxes, batch['reference'])
        check['invalid_selected'] += int((~batch['reference_valid'][idx, q]).sum())
        check['invalid_all'] += int((~batch['reference_valid']).sum())
        np.testing.assert_array_equal(batch['reference_valid'][idx, q], [r['reference_valid'] for r in rows])
        difference = batch['final_face_center'] != batch['final_face_region']
        check['arm_different_box_elements'] += int(difference.sum())
        check['arm_different_box_expressions'] += int(difference.reshape(n, -1).any(axis=1).sum())
        check['arm_box_max_absolute_difference'] = max(check['arm_box_max_absolute_difference'], float(np.abs(batch['final_face_center'].astype(np.float64) - batch['final_face_region'].astype(np.float64)).max()))
        for t in thresholds:
            check['arm_full256_hit_label_mismatches'][str(t)] += int(((per_box_iou['final_face_center'] > t).any(axis=1) != (per_box_iou['final_face_region'] > t).any(axis=1)).sum())
        check['rows'] += n
        check['npz_count'] += 1
        check['final_batch_rows'] = n
    a, b = batch_pair['initial_formal'], batch_pair['formal']
    np.testing.assert_array_equal(a['row_ids'], b['row_ids'])
    np.testing.assert_array_equal(a['root_gt'], b['root_gt'])
    for k, check in drift.items():
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
        changed = a[k] != b[k]
        check['different_elements'] += int(changed.sum())
        check['expressions_changed'] += int(changed.reshape(n, -1).any(axis=1).sum())
        check['candidate_slots_changed'] += int(changed.reshape(n, 256, -1).any(axis=-1).sum())
        check['selected_slots_changed'] += int(changed[idx, q].reshape(n, -1).any(axis=1).sum())
        check['maximum_absolute_difference'] = max(check['maximum_absolute_difference'], float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()))

for stage, check in stage_checks.items():
    expected = summary['cpu_stage_recounts'][stage]
    assert check['rows'] == 9508 and check['npz_count'] == 1189 and check['final_batch_rows'] == 4
    for own, other in (('coverage', 'cpu_full256_coverage'), ('selected_cpu_gpu_label_flips', 'cpu_selected_threshold_flips_by_box_type'),
                       ('oracle_label_mismatches', 'cpu_full256_oracle_label_mismatches'), ('available_unselected', 'qualified_candidate_available_but_unselected'),
                       ('no_qualified', 'no_qualified_candidate'), ('invalid_selected', 'invalid_reference_selected'), ('invalid_all', 'invalid_reference_all256')):
        assert check[own] == expected[other], (stage, own)
    rows = stage_rows[stage]
    check['arm_selected_comparison'] = transitions([r['arms'][arms[0]]['iou'] for r in rows], [r['arms'][arms[1]]['iou'] for r in rows])
    assert check['arm_selected_comparison'] == summary['arms_compared'][stage]
    check['mask_stored_recount'] = dict(hits=tally([r['mask_iou'] for r in rows]), mean_iou_percent=sum(r['mask_iou'] for r in rows) / 9508 * 100)
    check['parent_query_changes'] = sum(r['query'] != p['bbs']['query'] for r, p in zip(rows, parent))
    check['arms'] = {}
    for arm in arms:
        entry = next(e for e in summary['table'] if e['stage'] == stage and e['arm'] == arm)
        values = [r['arms'][arm]['iou'] for r in rows]
        own = dict(hits=tally(values), reference_to_final=transitions([r['reference_iou'] for r in rows], values),
                   prior_to_final=transitions([r['coarse_iou'] for r in rows], values), parent_to_final=transitions([p['bbs']['iou'] for p in parent], values),
                   selected_mask_good_box_bad=sum(r['mask_iou'] > 0.5 >= r['arms'][arm]['iou'] for r in rows))
        assert list(own['hits'].values()) == [entry['rec_hits25'], entry['rec_hits50']]
        assert own['reference_to_final'] == entry['internal_reference_to_final']
        assert own['prior_to_final'] == entry['internal_prior_to_final'] and own['parent_to_final'] == entry['parent_delta']
        assert own['selected_mask_good_box_bad'] == entry['selected_mask_good_box_bad']
        assert check['parent_query_changes'] == entry['parent_selected_query_changes']
        check['arms'][arm] = own
for k, values in summary['cross_stage_all256_shared_array_drift'].items():
    assert all(drift[k][field] == value for field, value in values.items())
selected_drift = {k: sum(a[k] != b[k] for a, b in zip(stage_rows['initial_formal'], stage_rows['formal']))
                  for k in ('query', 'mask_iou', 'coarse_box', 'reference_box', 'reference_valid')}
assert selected_drift == summary['cross_stage_selected_common_drift']
adaptation = {arm: transitions([r['arms'][arm]['iou'] for r in stage_rows['initial_formal']], [r['arms'][arm]['iou'] for r in stage_rows['formal']]) for arm in arms}
assert adaptation == summary['initial_to_trained']
holdout = {}
for stage in ('initial', 'terminal'):
    r = lines(complete / stage / 'rows.jsonl')
    assert [v['row_id'] for v in r] == split['holdout']
    rec = read(complete / stage / 'receipt.json')
    assert sha(complete / stage / 'rows.jsonl') == rec['rows_sha256']
    holdout[stage] = dict(rows=len(r), unique_scans=len({v['scan_id'] for v in r}), metrics={a: tally([v['arms'][a]['iou'] for v in r]) for a in arms})
    for a in arms:
        assert list(holdout[stage]['metrics'][a].values()) == [fit[stage][a]['rec_hits25'], fit[stage][a]['rec_hits50']]

result = dict(status='PASS_DETERMINISTIC_LOCAL_RECOUNT', fresh_reviewer=True, producer_analyzer_executions=0,
    interpreter=sys.executable, numpy_version=np.__version__, intake_files_verified=len(intake['files']),
    intake_bytes_verified=intake['total_bytes'], exact_current_collected_source_files=9, imported_source_hashes_verified=6,
    helper_hashes_verified=len(spec['runner_files']), training_steps=3723, fit_rows_unique=29778,
    fit_equals_manifest_partition=True, physical_batch=8, last_fit_batch=2,
    one_parent_and_one_native_head_call_per_logged_fit_batch=True, all_logged_gradients_finite_and_positive=True,
    logged_independent_head_gradients=True, extra_training_scope=extra, stages=stage_checks,
    independently_computed_float64_iou_values=9508 * 256 * 4 * 2, cross_stage_shared_drift=drift,
    selected_common_drift=selected_drift, adaptation=adaptation, holdout=holdout,
    protected_parent_hits=tally([v['bbs']['iou'] for v in parent]),
    no_raw_mask_recompute=True, no_nn_or_ssh_or_weight_load=True,
    elapsed_seconds=time.perf_counter() - started)
out = root / 'analysis/TERMINAL_AUDITOR_CHECKS.json'
out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
