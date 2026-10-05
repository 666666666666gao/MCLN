"""Independent local NumPy audit of immutable probe artifacts; no model imports."""
import ast
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
COMPLETE = BASE / 'complete'
TMP = BASE.parent
HELPERS = TMP / 'pvground_final_quality_20261005/runtime_bundle'
DATA = TMP / 'pvground_g_p2_20261002/complete/source'
NATIVE = TMP / 'pvground_runtime_bundle_20260908_v1/PV-Ground'
PORT = TMP / 'pvground_geometry_readback_20261004/revision2'
PRIOR_LOG = TMP / 'pvground_final_quality_20261005/complete/quality/train.jsonl'
checks = []


def verify(name, condition):
    assert bool(condition), name
    checks.append(name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def faces(box):
    return np.concatenate((box[..., :3] - box[..., 3:] / 2,
                           box[..., :3] + box[..., 3:] / 2), axis=-1)


def iou(box, truth):
    pf, tf = faces(box), faces(truth)
    extent = np.maximum(0, np.minimum(pf[..., 3:], tf[..., 3:])
                           - np.maximum(pf[..., :3], tf[..., :3]))
    intersection = np.prod(extent, axis=-1)
    return intersection / (np.prod(box[..., 3:], axis=-1)
                           + np.prod(truth[..., 3:], axis=-1) - intersection)


def native_box_gradient(box, truth, count):
    """Piecewise analytic derivative of native matched (10*L1+2*GIoU)/7."""
    p, t = faces(box), faces(truth)
    width = np.maximum(np.minimum(p[3:], t[3:]) - np.maximum(p[:3], t[:3]), 0)
    hull = np.maximum(p[3:], t[3:]) - np.minimum(p[:3], t[:3])
    intersection, enclosing = np.prod(width), np.prod(hull)
    union = np.prod(box[3:]) + np.prod(truth[3:]) - intersection
    result = []
    for coordinate in range(6):
        low_grad = np.zeros(3)
        high_grad = np.zeros(3)
        axis = coordinate % 3
        low_grad[axis] = 1 if coordinate < 3 else -.5
        high_grad[axis] = 1 if coordinate < 3 else .5
        # Equality uses the equal-argument half derivative of torch min/max.
        min_high = (p[3:] < t[3:]).astype(float) + .5 * (p[3:] == t[3:])
        max_low = (p[:3] > t[:3]).astype(float) + .5 * (p[:3] == t[:3])
        extent_grad = (high_grad * min_high - low_grad * max_low) * (width > 0)
        intersection_grad = sum(extent_grad[k] * np.prod(np.delete(width, k)) for k in range(3))
        hull_grad = (high_grad * ((p[3:] > t[3:]) + .5 * (p[3:] == t[3:]))
                     - low_grad * ((p[:3] < t[:3]) + .5 * (p[:3] == t[:3])))
        enclosing_grad = sum(hull_grad[k] * np.prod(np.delete(hull, k)) for k in range(3))
        volume_grad = 0 if coordinate < 3 else np.prod(np.delete(box[3:], axis))
        union_grad = volume_grad - intersection_grad
        giou_grad = ((intersection_grad * union - intersection * union_grad) / union**2
                     + (union_grad * enclosing - union * enclosing_grad) / enclosing**2)
        l1_grad = (10 if coordinate < 3 else 5) * np.sign(box[coordinate] - truth[coordinate])
        result.append((l1_grad - 2 * giou_grad) / (7 * count))
    return np.asarray(result)


intake = read_json(COMPLETE / 'INTAKE.json')
receipt = read_json(COMPLETE / 'receipt.json')
summary = read_json(BASE / 'analysis/SUMMARY.json')
spec = read_json(BASE / 'spec.json')
rows = [json.loads(line) for line in (COMPLETE / 'rows.jsonl').read_text(encoding='utf-8').splitlines()]
source_review = read_json(BASE / 'SOURCE_REVIEW.json')
for entry in intake['files']:
    path = COMPLETE / entry['name']
    verify('intake_bytes_and_sha:' + entry['name'], path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'])
verify('intake_total_bytes', sum(entry['bytes'] for entry in intake['files']) == intake['total_bytes'])
for name in ('EXPERIMENT_PLAN.md', 'SOURCE_REVIEW.md', 'SOURCE_REVIEW.json', 'controller.py',
             'run_mask_geometry_probe.py', 'spec.json', 'launch.json'):
    verify('deployed_local_identity:' + name, (BASE / name).read_bytes() == (COMPLETE / name).read_bytes())
for entry in source_review['reviewed_files']:
    path = Path(entry['path'])
    verify('prior_review_identity:' + path.name, path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'])
for name, expected in spec['runner_files'].items():
    verify('helper_identity:' + name, sha(HELPERS / name) == expected)
imports = read_json(COMPLETE / 'imports.json')
import_paths = {'src.joint_det_dataset': DATA / 'joint_det_dataset.py',
                'models.losses': NATIVE / 'models/losses.py',
                'main_utils': NATIVE / 'main_utils.py',
                'prepare_data': NATIVE / 'prepare_data.py',
                'models.pv_ground': PORT / 'source_preview/PV-Ground/models/pv_ground.py'}
for name, path in import_paths.items():
    verify('actual_import_identity:' + name, sha(path) == imports['sha256'][name])
manifest = read_json(DATA / 'input_manifest.json')
appearance = read_json(DATA / 'appearance_source_manifest.json')
partition = read_json(DATA / 'split_protocol.json')
verify('manifest_source_identity', sha(DATA / 'appearance_source_manifest.json') == manifest['source_manifest_sha256'])
verify('manifest_loader_identity', appearance['files']['src/joint_det_dataset.py'] == sha(DATA / 'joint_det_dataset.py'))
verify('split_identity', sha(DATA / 'split_protocol.json') == manifest['split_protocol_sha256'])
fit, holdout = set(partition['row_ids']['fit']), set(partition['row_ids']['holdout'])
verify('partition_size_disjoint', len(fit) == 29778 and len(holdout) == 6887 and not fit.intersection(holdout))
verify('source_port_identity', sha(PORT / 'complete_preflight/source_port.json') == spec['source_port_sha256'])
verify('receipt_runner_spec_rows_identity', receipt['runner_sha256'] == sha(COMPLETE / 'run_mask_geometry_probe.py')
       and receipt['spec_sha256'] == sha(COMPLETE / 'spec.json') and receipt['rows_sha256'] == sha(COMPLETE / 'rows.jsonl'))
verify('summary_receipt_identity', summary['receipt_sha256'] == sha(COMPLETE / 'receipt.json'))
verify('generation_identity', read_json(BASE / 'GENERATION.json')['generated_sha256'] == sha(BASE / 'run_mask_geometry_probe.py'))
with PRIOR_LOG.open(encoding='utf-8') as stream:
    prior_first8 = [json.loads(next(stream)) for _ in range(8)]
row_order = [row['row_id'] for row in rows]
verify('first64_order_and_unique', row_order == [row_id for batch in prior_first8 for row_id in batch['rows']]
       and len(rows) == len(set(row_order)) == 64)
verify('all_probe_rows_fit', all(row_id in fit and row_id not in holdout for row_id in row_order))
verify('stored_scans_in_fit_physical_fold', all(int(hashlib.sha256((manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()).hexdigest()[:8], 16) % 5 != 0 for row in rows))
verify('all_scanrefer', all(row['sample_dataset'] == row['language_dataset'] == 'scanrefer' for row in rows))
verify('all_root_only', all(row['valid_native_GT_slots'] == [0] and row['native_matched_slots'] == [0] for row in rows))
verify('all_point_hashes_well_formed', all(len(row['point_sha256']) == 64 and int(row['point_sha256'], 16) >= 0 for row in rows))

totals = Counter()
max_errors = Counter()
errors, changes = [], []
grad_differences = []
outside_candidates = outside_faces = coarse_hits = 0
outside_with_final_good = outside_all = 0
example_outside_final_good = None
all_matched = all_box_nonzero = all_boundary_nonzero = 0
union_range = [50000, 0]
top_ties = 0
array_schema = None
step = 3.0 ** (2.0 / 30)
knots = np.asarray([-4.] + [1-step**i for i in range(15, 0, -1)] + [0.]
                   + [step**i-1 for i in range(1, 16)] + [4.])
verify('boundary_knot_math', len(knots) == 33 and np.all(np.diff(knots) > 0)
       and knots[0] == -4 and knots[-1] == 4 and np.allclose(knots, -knots[::-1], atol=0))
for batch_index in range(8):
    with np.load(COMPLETE / ('batch_%02d.npz' % batch_index), allow_pickle=False) as archive:
        a = {key: archive[key] for key in archive.files}
    schema = {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in a.items()}
    if array_schema is None:
        array_schema = schema
    verify('array_schema:' + str(batch_index), schema == array_schema and a['boxes'].shape == (8, 256, 6))
    verify('finite_arrays:' + str(batch_index), all(np.isfinite(value).all() for value in a.values()))
    verify('candidate_domains:' + str(batch_index), (a['boxes'][..., 3:] > 0).all()
           and (a['root_box'][..., 3:] > 0).all() and (a['coarse_boxes'][..., 3:] >= 1e-6).all()
           and np.isin(a['matched_slot'], [-1, 0]).all()
           and (a['box_gradient_max'] >= 0).all() and (a['boundary_gradient_max'] >= 0).all())
    verify('point_count_domains:' + str(batch_index), np.issubdtype(a['mask_intersection'].dtype, np.integer)
           and np.issubdtype(a['mask_union'].dtype, np.integer)
           and (a['mask_intersection'] >= 0).all() and (a['mask_union'] > 0).all()
           and (a['mask_intersection'] <= a['mask_union']).all() and (a['mask_union'] <= 50000).all())
    union_range = [min(union_range[0], int(a['mask_union'].min())), max(union_range[1], int(a['mask_union'].max()))]
    box = a['boxes'].astype(np.float64)
    root = a['root_box'].astype(np.float64)[:, None, :]
    coarse = a['coarse_boxes'].astype(np.float64)
    bi = iou(box, root)
    mi = a['mask_intersection'] / a['mask_union']
    verify('iou_values:' + str(batch_index), np.allclose(bi, a['box_iou'], rtol=2e-5, atol=2e-6)
           and np.allclose(mi, a['mask_iou'], rtol=1e-7, atol=1e-7))
    verify('all_thresholds:' + str(batch_index), np.array_equal(bi > .5, a['box_iou'] > .5)
           and np.array_equal(mi > .5, a['mask_iou'] > .5))
    max_errors['box_iou_abs'] = max(max_errors['box_iou_abs'], float(np.abs(bi-a['box_iou']).max()))
    max_errors['mask_iou_abs'] = max(max_errors['mask_iou_abs'], float(np.abs(mi-a['mask_iou']).max()))
    bf, cf, tf = faces(box), faces(coarse), faces(root)
    face_error = np.max(np.abs(bf - tf), axis=-1)
    face_change = np.max(np.abs(bf - cf), axis=-1)
    verify('face_values:' + str(batch_index), np.allclose(face_error, a['max_face_error'], atol=1e-6)
           and np.allclose(face_change, a['max_face_change'], atol=1e-6))
    max_errors['face_error_abs'] = max(max_errors['face_error_abs'], float(np.abs(face_error-a['max_face_error']).max()))
    max_errors['face_change_abs'] = max(max_errors['face_change_abs'], float(np.abs(face_change-a['max_face_change']).max()))
    reference = np.maximum(coarse[..., 3:], 1e-6)
    offsets = (tf - cf) * np.concatenate((-4 / reference, 4 / reference), axis=-1)
    targets = np.concatenate(((coarse[..., :3] - tf[..., :3]) * 4 / reference - 2,
                              (tf[..., 3:] - coarse[..., :3]) * 4 / reference - 2), axis=-1)
    verify('face_parameterization_identity:' + str(batch_index), np.allclose(offsets, targets, atol=1e-9))
    decoded_center = coarse[..., :3] + (offsets[..., 3:] - offsets[..., :3]) * reference / 8
    decoded_size = reference * (1 + (offsets[..., :3] + offsets[..., 3:]) / 4)
    verify('face_decode_exact_gt_identity:' + str(batch_index), np.allclose(decoded_center, root[..., :3], atol=1e-9)
           and np.allclose(decoded_size, root[..., 3:], atol=1e-9))
    outside = (offsets < -4) | (offsets > 4)
    matched_outside_faces = int(outside[a['matched_slot'] >= 0].sum())
    verify('matched_boundary_range_receipt:' + str(batch_index), matched_outside_faces == receipt['batches_metadata'][batch_index]['boundary_counts']['boundary_target_outside'])
    ci = iou(coarse, root)
    for position in range(8):
        global_index = batch_index * 8 + position
        row = rows[global_index]
        prefix = str(global_index) + ':'
        verify(prefix+'ids_and_gt', int(a['row_id'][position]) == row['row_id'] and row['batch_index'] == batch_index
               and row['candidates'] == 256 and np.array_equal(a['root_box'][position], row['root_box']))
        role = a['matched_slot'][position]
        expected = np.full(256, -1, dtype=np.int16)
        expected[row['native_matched_queries']] = row['native_matched_slots']
        verify(prefix+'assignment_mapping', np.array_equal(role, expected) and np.count_nonzero(role >= 0) == 1)
        selected = int(np.argmax(a['bbs'][position]))
        top_ties += np.count_nonzero(a['bbs'][position] == a['bbs'][position, selected]) > 1
        verify(prefix+'selected', selected == row['selected_query']
               and float(a['box_iou'][position, selected]) == row['selected_box_iou']
               and float(a['mask_iou'][position, selected]) == row['selected_mask_iou']
               and int(role[selected]) == row['selected_matched_slot'])
        bg, dg = a['box_gradient_max'][position], a['boundary_gradient_max'][position]
        verify(prefix+'unmatched_gradients', (bg[role < 0] == 0).all() and (dg[role < 0] == 0).all())
        matched_query = row['native_matched_queries'][0]
        analytic = native_box_gradient(box[position, matched_query], root[position, 0], 8)
        difference = abs(float(np.abs(analytic).max()) - float(bg[matched_query]))
        grad_differences.append(difference)
        verify(prefix+'native_box_gradient_analytic', difference < 2e-6)
        all_matched += int((role >= 0).sum())
        all_box_nonzero += int((bg > 0).sum())
        all_boundary_nonzero += int((dg > 0).sum())
        mask_only = (mi[position] > .5) & (bi[position] <= .5)
        both = (mi[position] > .5) & (bi[position] > .5)
        pool = mask_only & (role < 0)
        count = {'mask_only': int(mask_only.sum()), 'both_good': int(both.sum()),
                 'mask_only_unmatched': int(pool.sum()),
                 'mask_only_matched_root': int((mask_only & (role == 0)).sum()),
                 'mask_only_matched_other': int((mask_only & (role > 0)).sum()),
                 'mask_only_box_gradient_nonzero': int((mask_only & (bg > 0)).sum()),
                 'mask_only_boundary_gradient_nonzero': int((mask_only & (dg > 0)).sum()),
                 'unmatched_good_box': int(((bi[position] > .5) & (role < 0)).sum())}
        verify(prefix+'role_counts', count == row['counts'])
        totals.update(count)
        totals['rows_with_mask_only_unmatched'] += bool(pool.any())
        totals['rows_with_both_good'] += bool(both.any())
        totals['selected_mask_only'] += bool(mask_only[selected])
        totals['selected_mask_only_unmatched'] += bool(pool[selected])
        totals['valid_native_GT_slots_total'] += len(row['valid_native_GT_slots'])
        errors.extend(face_error[position, pool].tolist())
        changes.extend(face_change[position, pool].tolist())
        outside_candidates += int(outside[position, pool].any(-1).sum())
        outside_faces += int(outside[position, pool].sum())
        coarse_hits += int((pool & (ci[position] > .5)).sum())
        outside_any = outside[position].any(-1)
        outside_all += int(outside_any.sum())
        counterexamples = outside_any & (bi[position] > .5)
        outside_with_final_good += int(counterexamples.sum())
        if example_outside_final_good is None and counterexamples.any():
            query = int(np.flatnonzero(counterexamples)[0])
            example_outside_final_good = {'row_id': row['row_id'], 'query': query,
                'box_iou': float(bi[position, query]), 'face_targets': offsets[position, query].tolist()}
    verify('batch_receipt:' + str(batch_index), receipt['batches_metadata'][batch_index]['rows'] == a['row_id'].tolist()
           and receipt['batches_metadata'][batch_index]['matching_calls'] == 7
           and receipt['batches_metadata'][batch_index]['native_box_count'] == 8
           and receipt['batches_metadata'][batch_index]['boundary_counts']['boundary_matched_boxes'] == 8)
verify('all_counts_vs_receipt_and_summary', dict(totals) == receipt['totals'] == summary['counts'])
verify('range_counts_vs_summary', outside_candidates == summary['mask_only_unmatched_boundary_target_outside_candidates']
       and outside_faces == summary['mask_only_unmatched_boundary_target_outside_faces']
       and coarse_hits == summary['mask_only_unmatched_coarse_box_hits50'])
verify('medians_vs_summary', abs(np.median(errors)-summary['mask_only_unmatched_median_max_face_error_m']) < 1e-6
       and abs(np.median(changes)-summary['mask_only_unmatched_median_max_face_change_m']) < 1e-6)
status = read_json(COMPLETE / 'status.json')
wait = read_json(BASE / 'wait.json')
verify('closure_receipts', status == wait['status'] and status['status'] == 'complete' and status['exit_code'] == 0
       and wait['observer_closed'] and wait['exitcode'] == 0 and not wait['controller_alive']
       and (COMPLETE / 'probe.exit').read_text().strip() == (COMPLETE / 'controller.exit').read_text().strip() == '0')
verify('controller_log_identity', json.loads((COMPLETE / 'controller.log').read_text()) == status)
probe_lines = (COMPLETE / 'probe.log').read_text(encoding='utf-8').splitlines()
logged_batches = [json.loads(line.split(' ', 1)[1]) for line in probe_lines if line.startswith('FIT_ROLE_PROBE_BATCH ')]
logged_receipts = [json.loads(line.split(' ', 1)[1]) for line in probe_lines if line.startswith('FIT_ROLE_PROBE_COMPLETE ')]
verify('probe_logs_receipt', len(logged_batches) == 8 and logged_receipts == [receipt]
       and logged_batches[-1]['totals'] == dict(totals))
verify('no_update_receipts', receipt['optimizer_steps'] == receipt['weight_files_created'] == 0
       and not receipt['optimizer_constructed'] and receipt['model_state_unchanged'] and receipt['model_gradients_absent']
       and status['protected_parent_hashes_exact'] and not any(COMPLETE.glob('*.pth')) and not any(COMPLETE.glob('*.pt')))
tree = ast.parse((BASE / 'run_mask_geometry_probe.py').read_text(encoding='utf-8'))
call_names = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
verify('no_model_backward_or_weight_save_calls', not any(name.endswith('.backward') or name.startswith('torch.optim.')
       or name in ('torch.save', 'optimizer.step') for name in call_names))
verify('two_leaf_autograd_calls', call_names.count('torch.autograd.grad') == 2)
analysis_tree = ast.parse((BASE / 'analyze_probe.py').read_text(encoding='utf-8'))
assert_sites = sorted(node.lineno for node in ast.walk(analysis_tree) if isinstance(node, ast.Assert))
expanded_assert_count = len(intake['files']) + 1 + 8 * (5 + 8 * 6) + 3
bookkeeping = 8 * 4 + 64 * 8 + 4
verify('summary_bookkeeping_formula', summary['cpu_checks'] == bookkeeping)

inputs = sorted(set([path for path in BASE.glob('*') if path.is_file() and path.name not in ('active_continuation_state.json',)]
                   + list(COMPLETE.glob('*')) + [BASE / 'analysis/SUMMARY.json']
                   + [path for path in HELPERS.glob('*.py')]
                   + list(import_paths.values())
                   + [DATA / name for name in ('input_manifest.json', 'appearance_source_manifest.json', 'split_protocol.json')]
                   + [PRIOR_LOG, NATIVE / 'src/grounding_evaluator.py', PORT / 'complete_preflight/source_port.json']
                   + [Path(entry['path']) for entry in source_review['reviewed_files']]), key=str)
identities = [{'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size} for path in inputs]
result = {
    'status': 'PASS', 'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'execution': 'LOCAL_CPU_NUMPY_ONLY', 'model_import_or_replay': False, 'remote_actions': False,
    'rows': 64, 'candidates': 16384, 'batches': 8, 'seed': 2027,
    'unique_scan_ids': len({row['scan_id'] for row in rows}),
    'unique_physical_scene_ids': len({row['scan_id'].split('_')[0] for row in rows}),
    'first64_order_matches': True, 'point_tensor_hashes_recomputed': False,
    'all_root_only': True, 'counts': dict(totals),
    'all_matched': all_matched, 'all_unmatched': 16384-all_matched,
    'all_nonzero_box_gradients': all_box_nonzero, 'all_nonzero_boundary_gradients': all_boundary_nonzero,
    'unmatched_nonzero_box_gradients': 0, 'unmatched_nonzero_boundary_gradients': 0,
    'box_gradient_analytic_cases': len(grad_differences),
    'box_gradient_analytic_max_abs_difference': max(grad_differences),
    'box_threshold_differences': 0, 'mask_count_threshold_differences': 0,
    'numerical_max_abs_differences': dict(max_errors), 'bbs_top_tie_rows': int(top_ties),
    'mask_union_point_count_range': union_range,
    'face_error_median_float64': float(np.median(errors)),
    'face_change_median_float64': float(np.median(changes)),
    'outside_pool_candidates': outside_candidates, 'outside_pool_faces': outside_faces,
    'coarse_hits50_in_pool': coarse_hits, 'outside_all_candidates': outside_all,
    'outside_exact_gt_targets_but_final_box_iou_gt_half': outside_with_final_good,
    'outside_counterexample': example_outside_final_good,
    'raw_mask_reconstructed': False, 'hungarian_costs_reconstructed': False,
    'boundary_logit_gradients_reconstructed': False,
    'runtime_state_and_parent_weights_independently_rehashed': False,
    'runtime_receipts_consistent': True,
    'original_analyzer_cpu_checks': {'saved': summary['cpu_checks'], 'bookkeeping_formula': '8*4+64*8+4',
        'assert_statement_sites': assert_sites, 'distinct_assert_sites': len(assert_sites),
        'expanded_assert_evaluations_from_source_and_actual_loop_sizes': expanded_assert_count,
        'interpretation': '548 is a manually accumulated label; it is neither distinct sites nor actual assertion evaluations. Original SUMMARY.json preserved.'},
    'array_schema': array_schema, 'actual_check_count': len(checks), 'actual_checks': checks,
    'reviewed_files': identities,
    'original_summary_sha256': sha(BASE / 'analysis/SUMMARY.json')}
path = BASE / 'analysis/AUDIT_CPU.json'
path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key not in ('reviewed_files', 'actual_checks', 'array_schema')}, indent=2))
