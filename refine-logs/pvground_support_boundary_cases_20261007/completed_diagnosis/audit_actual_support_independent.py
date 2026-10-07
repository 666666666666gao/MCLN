"""Independent bounded CPU checks of immutable archived diagnostic evidence.

No project helper, torch, checkpoint, network or neural forward is imported.
During preliminary intake, an in-flight file is recorded and not interpreted.
"""
import collections
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PHASE = sys.argv[1]
assert PHASE in ('preliminary', 'final')
HASHES = {}


def read_bytes(path):
    raw = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def read_json(path):
    return json.loads(read_bytes(path))


def read_rows(path):
    return [json.loads(line) for line in read_bytes(path).decode('utf-8').splitlines()]


def require(condition, message):
    assert condition, message


def box_bounds(box):
    center = np.asarray(box[:3], dtype=np.float64)
    size = np.asarray(box[3:], dtype=np.float64)
    return np.r_[center - size / 2, center + size / 2]


def box_iou(box, gt):
    b = np.asarray(box, dtype=np.float64).copy()
    b[3:] = np.maximum(b[3:], 1e-6)
    a, t = box_bounds(b), box_bounds(gt)
    intersection = np.prod(np.maximum(0, np.minimum(a[3:], t[3:]) - np.maximum(a[:3], t[:3])))
    return float(intersection / (np.prod(b[3:]) + np.prod(np.asarray(gt[3:], dtype=np.float64)) - intersection))


def new_stat():
    return dict(rows=0, invalid_ranges=0, any_pure_background_extreme_rows=0,
                any_mixed_extreme_rows=0, any_background_extremal_point_rows=0,
                excessive_faces=0, missing_faces=0, pure_background_extreme_faces=0,
                mixed_extreme_faces=0, excessive_background_extreme_faces=0,
                coarse_hits25=0, reference_hits25=0, coarse_hits50=0,
                reference_hits50=0, query_changed_rows=0)


packet = read_json(ROOT / 'case_manifest.json')
spec = read_json(ROOT / 'diagnostic_spec.json')
receipt = read_json(ROOT / 'complete/receipt.json')
imports = read_json(ROOT / 'complete/imports.json')
load = read_json(ROOT / 'complete/load.json')
wait = read_json(ROOT / 'wait.json')
rows = read_rows(ROOT / 'complete/rows.jsonl')
require(HASHES[str(ROOT / 'complete/rows.jsonl')] == receipt['rows_sha256'], 'rows receipt digest')
require(wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode'] == 0, 'terminal closure')
require(receipt['cases'] == 191 and receipt['optimizer_updates'] == receipt['weights_created'] == 0, 'receipt scope')
require(len(rows) == len({r['row_id'] for r in rows}) == 191, 'unique actual rows')
require(spec['seed'] == 2027 and spec['optimizer_steps'] == spec['weights_created'] == 0, 'spec scope')
cases = {item['cached']['row_id']: item for item in packet['diagnostic_rows']}
require(set(cases) == {r['row_id'] for r in rows}, 'manifest actual row set')
blocks = sorted({r['row_id'] // 8 for r in rows})
forward_ids = [i for b in blocks for i in range(8*b, min(8*b+8, 9508))]
require(blocks == packet['batch_indices'] and forward_ids == packet['forward_row_ids'], 'original B8 contexts')
require(len(blocks) == spec['forward_batches'] == receipt['forward_batches'] == 161, '161 batches')
require(len(forward_ids) == spec['forward_rows'] == receipt['original_batch_context_rows'] == 1288, '1288 context rows')

history_path = ROOT.parent / 'pvground_mask_reference_20261006/complete_fit/fused_mask_reference/initial_formal/rows.jsonl'
cohort_path = ROOT.parent / 'pvground_reference_keep_20261006/cached_reference_errors/0.25_damaged.jsonl'
history = {r['row_id']: r for r in read_rows(history_path)}
cohort = {r['row_id']: r for r in read_rows(cohort_path)}
require(HASHES[str(history_path)] == packet['historical_source_sha256'], 'primary historical rows hash')
require(HASHES[str(cohort_path)] == packet['cohort_source_sha256'], 'primary cohort hash')
require(len(history) == 9508 and set(cohort) == set(cases), 'historical scope')

source_paths = {
    'src.joint_det_dataset': ROOT.parent / 'pvground_g_p2_20261002/complete/source/joint_det_dataset.py',
    'models.pv_ground': ROOT.parent / 'pvground_geometry_readback_20261004/revision2/source_preview/PV-Ground/models/pv_ground.py',
    'selected_mask_reference_factory': ROOT / 'selected_mask_reference_factory.py',
    'mask_reference': ROOT / 'mask_reference.py',
    'native_root_bbs': ROOT.parent / 'pvground_final_quality_20261005/runtime_bundle/native_root_bbs.py',
}
for name, path in source_paths.items():
    read_bytes(path)
    require(HASHES[str(path)] == imports['sha256'][name], 'actual imported source hash: ' + name)
for name in ('collect_support_cases.py', 'support_evidence.py', 'case_manifest.json', 'diagnostic_spec.json'):
    require(read_bytes(ROOT / name) == read_bytes(ROOT / 'complete' / name), 'local vs executed source: ' + name)
for name in ('analyze_closed.py', 'EXPERIMENT_PLAN.md', 'prepare_inputs.py'):
    read_bytes(ROOT / name)

role_counts = collections.Counter()
changed_by_group = collections.Counter()
groups = collections.Counter()
for r in rows:
    c = cases[r['row_id']]
    old = history[r['row_id']]
    require(c['cached'] == cohort[r['row_id']], 'cohort content')
    for key in ('scan_id', 'target_id'):
        require(r[key] == c['cached'][key] == old[key], 'identity ' + key)
    require(r['point_sha256'] == c['point_sha256'] == old['point_sha256'], 'reported input identity')
    require(r['historical_query'] == c['cached']['query'] == old['bbs']['query'], 'historical numerical query')
    require(c['cached']['root_gt'] == old['root_box'], 'historical GT')
    require(c['cached']['prior_iou'] > .25 and c['cached']['reference_iou'] <= .25, 'historical damage definition')
    expected_group = ('overextended' if c['cached']['gt_volume_covered'] >= .95 and c['cached']['reference_to_gt_volume_ratio'] > 4 else 'missing_gt_extent')
    require(r['group'] == c['group'] == expected_group, 'historical group')
    require(expected_group != 'missing_gt_extent' or c['cached']['gt_volume_covered'] < .95, 'missing extent definition')
    require(r['selection_changed'] == (r['deployed_query'] != r['historical_query']), 'change flag')
    require([x['role'] for x in r['roles']] == ['deployed', 'historical'], 'role order')
    require([x['query'] for x in r['roles']] == [r['deployed_query'], r['historical_query']], 'role query')
    require(r['native_semantic_head_calls'] == 1, 'semantic call count')
    groups[r['group']] += 1
    changed_by_group[r['group']] += int(r['selection_changed'])
    for role in r['roles']:
        role_counts[role['role'] + ('/valid' if role['valid_mask_range'] else '/invalid')] += 1
require(dict(groups) == packet['groups'] == receipt['groups'], 'source group counts')

available_paths = sorted((ROOT / 'complete/arrays').glob('*.npz'))
row_by_archive = {r['arrays']: r for r in rows}
require(all(p.name in row_by_archive for p in available_paths), 'unexpected archive')
verified = []
in_flight = []
aggregate = {}
faces_checked = 0
point_labels_checked = 0
maximum_reference_bound_error = 0.0
maximum_target_member_to_gt_bound_error = 0.0
maximum_fusion_error = 0.0
array_member_count = 0
expected_keys = {'xyz', 'superpoint', 'target', 'queries', 'root_gt', 'text_logits', 'query_logits', 'fused_logits', 'fused_active', 'alpha', 'reference_boxes', 'reference_valid', 'coarse_boxes', 'all256_scores'}

for path in available_paths:
    r = row_by_archive[path.name]
    raw = read_bytes(path)
    if len(raw) != r['bytes'] or HASHES[str(path)] != r['sha256']:
        in_flight.append(dict(file=path.name, observed_bytes=len(raw), expected_bytes=r['bytes'], observed_sha256=HASHES[str(path)], expected_sha256=r['sha256']))
        continue
    with np.load(io.BytesIO(raw), allow_pickle=False) as zipped:
        a = {key: zipped[key] for key in zipped.files}
    require(set(a) == expected_keys, 'archive fields')
    array_member_count += len(a)
    n = len(a['text_logits'])
    expected_shapes = dict(xyz=(50000,3), superpoint=(50000,), target=(50000,), queries=(2,), root_gt=(6,), text_logits=(n,), query_logits=(2,n), fused_logits=(2,n), fused_active=(2,n), alpha=(), reference_boxes=(2,6), reference_valid=(2,), coarse_boxes=(2,6), all256_scores=(256,))
    for key, shape in expected_shapes.items():
        require(a[key].shape == shape and np.isfinite(a[key]).all(), 'shape/finite: ' + key)
    require(a['target'].dtype == np.bool_ and a['fused_active'].dtype == np.bool_ and a['reference_valid'].dtype == np.bool_, 'boolean labels')
    require(np.issubdtype(a['superpoint'].dtype, np.integer) and a['superpoint'].min() >= 0 and a['superpoint'].max() < n, 'superpoint ids')
    require(a['queries'].tolist() == [r['deployed_query'], r['historical_query']], 'archive role queries')
    c = cases[r['row_id']]['cached']
    require(a['root_gt'].tolist() == c['root_gt'], 'actual archived GT')
    require(0 <= a['alpha'].item() <= 1, 'fusion alpha')
    fusion = a['alpha'] * a['text_logits'][None,:] + (1-a['alpha']) * a['query_logits']
    fusion_error = float(np.max(np.abs(fusion.astype(np.float64) - a['fused_logits'])))
    maximum_fusion_error = max(maximum_fusion_error, fusion_error)
    require(fusion_error <= 2e-6, 'fusion reconstruction')
    require(np.array_equal(a['fused_active'], a['fused_logits'] > 0), 'active positive logit mask')
    require(a['all256_scores'][r['deployed_query']] == a['all256_scores'].max(), 'native score selected max')
    xyz = a['xyz'].astype(np.float64)
    ids = a['superpoint']
    target = a['target']
    require(target.any(), 'real GT members nonempty')
    point_labels_checked += len(target)
    target_bounds = np.r_[xyz[target].min(axis=0), xyz[target].max(axis=0)]
    require(target_bounds.tolist() == r['target_member_bounds'], 'target member extents')
    gt_bounds = box_bounds(a['root_gt'])
    maximum_target_member_to_gt_bound_error = max(maximum_target_member_to_gt_bound_error, float(np.max(np.abs(target_bounds-gt_bounds))))
    comparisons = []
    for slot, role in enumerate(r['roles']):
        active_points = a['fused_active'][slot, ids]
        member_indices = np.flatnonzero(active_points)
        require(role['foreground_points'] == int(active_points.sum()), 'foreground count')
        require(role['false_positive_points'] == int(np.count_nonzero(active_points & ~target)), 'false positives')
        require(role['false_negative_points'] == int(np.count_nonzero(~active_points & target)), 'false negatives')
        require(role['target_points'] == int(target.sum()), 'target count')
        valid = len(member_indices) > 0
        if valid:
            selected = xyz[member_indices]
            bound = np.r_[selected.min(axis=0), selected.max(axis=0)]
            valid = bool(np.all(bound[3:] > bound[:3]))
        require(bool(a['reference_valid'][slot]) == role['valid_mask_range'] == valid, 'independent mask validity')
        stat = aggregate.setdefault(r['group'] + '/' + role['role'], new_stat())
        stat['rows'] += 1
        stat['invalid_ranges'] += int(not valid)
        stat['query_changed_rows'] += int(r['selection_changed'])
        row_pure_background = row_mixed = row_background = False
        if not valid:
            require(role['faces'] == [], 'no invented invalid faces')
            require(np.allclose(a['reference_boxes'][slot], a['coarse_boxes'][slot], rtol=0, atol=1e-6), 'native invalid range prior')
        else:
            bound_error = float(np.max(np.abs(bound - box_bounds(a['reference_boxes'][slot]))))
            maximum_reference_bound_error = max(maximum_reference_bound_error, bound_error)
            require(bound_error < 2e-6 and len(role['faces']) == 6, 'actual predicted range')
            for face_index, face in enumerate(role['faces']):
                axis = face_index % 3
                extrema = [int(i) for i in member_indices if xyz[i, axis] == bound[face_index]]
                require(face['face'] == ('x-','y-','z-','x+','y+','z+')[face_index], 'face order')
                require(face['extremal_point_indices'] == extrema, 'raw extremal member indices')
                require(face['extent'] == float(bound[face_index]), 'face extent')
                require(face['gt_box_extent'] == float(gt_bounds[face_index]), 'GT box extent')
                require(face['gt_member_extent'] == float(target_bounds[face_index]), 'GT member extent')
                signed = float((gt_bounds[face_index] - bound[face_index]) * (1 if face_index < 3 else -1))
                require(face['signed_outward_error_m'] == signed, 'signed extent error')
                require(face['extremal_points'] == len(extrema) and face['extremal_target_points'] == int(target[extrema].sum()), 'extreme GT counts')
                pure_target, pure_background, mixed = [], [], []
                for group_id in sorted(set(int(ids[i]) for i in extrema)):
                    members = target[ids == group_id]
                    ones = int(members.sum())
                    if ones == 0:
                        pure_background.append(group_id)
                    elif ones == len(members):
                        pure_target.append(group_id)
                    else:
                        mixed.append(group_id)
                    key = str(group_id)
                    require(face['superpoint_target_fraction'][key] == ones/len(members), 'GT member purity')
                    for field, value in (('text_logits', a['text_logits'][group_id]), ('query_logits', a['query_logits'][slot, group_id]), ('fused_logits', a['fused_logits'][slot, group_id])):
                        require(face[field][key] == float(value), 'raw extreme evidence logit')
                require(face['pure_target_superpoints'] == pure_target and face['pure_background_superpoints'] == pure_background and face['mixed_superpoints'] == mixed, 'member-derived superpoint classes')
                background = int(target[extrema].sum()) < len(extrema)
                excessive = signed > 1e-6
                stat['excessive_faces'] += int(excessive)
                stat['missing_faces'] += int(signed < -1e-6)
                stat['pure_background_extreme_faces'] += int(bool(pure_background))
                stat['mixed_extreme_faces'] += int(bool(mixed))
                stat['excessive_background_extreme_faces'] += int(excessive and background)
                row_pure_background |= bool(pure_background)
                row_mixed |= bool(mixed)
                row_background |= background
                faces_checked += 1
        stat['any_pure_background_extreme_rows'] += int(row_pure_background)
        stat['any_mixed_extreme_rows'] += int(row_mixed)
        stat['any_background_extremal_point_rows'] += int(row_background)
        coarse_iou = box_iou(a['coarse_boxes'][slot], a['root_gt'])
        reference_iou = box_iou(a['reference_boxes'][slot], a['root_gt'])
        for threshold, suffix in ((.25, '25'), (.5, '50')):
            stat['coarse_hits' + suffix] += int(coarse_iou > threshold)
            stat['reference_hits' + suffix] += int(reference_iou > threshold)
        comparisons.append(dict(role=role['role'], query=role['query'], coarse_iou=coarse_iou, reference_iou=reference_iou))
    historical_coarse_diff = float(np.max(np.abs(a['coarse_boxes'][1].astype(np.float64) - np.asarray(c['prior_box']))))
    historical_reference_diff = float(np.max(np.abs(a['reference_boxes'][1].astype(np.float64) - np.asarray(c['reference_box']))))
    deployed_coarse_diff = float(np.max(np.abs(a['coarse_boxes'][0].astype(np.float64) - np.asarray(c['prior_box']))))
    deployed_reference_diff = float(np.max(np.abs(a['reference_boxes'][0].astype(np.float64) - np.asarray(c['reference_box']))))
    verified.append(dict(row_id=r['row_id'], group=r['group'], selection_changed=r['selection_changed'],
                         historical_coarse_box_max_difference=historical_coarse_diff,
                         historical_reference_box_max_difference=historical_reference_diff,
                         deployed_coarse_box_max_difference=deployed_coarse_diff,
                         deployed_reference_box_max_difference=deployed_reference_diff,
                         max_score_ties=int(np.count_nonzero(a['all256_scores'] == a['all256_scores'].max())),
                         queries=a['queries'].tolist(), comparisons=comparisons, array_sha256=HASHES[str(path)]))

summary_comparison = 'NOT_RUN_PENDING_COMPLETE_INTAKE_AND_ANALYSIS'
if PHASE == 'final':
    require(len(verified) == 191 and not in_flight, 'all complete archived arrays required')
    intake = read_json(ROOT / 'complete/INTAKE.json')
    summary = read_json(ROOT / 'analysis/SUMMARY.json')
    analyzed = read_rows(ROOT / 'analysis/rows.jsonl')
    require(intake['status'] == 'CLOSED_ARTIFACTS_COLLECTED' and not intake['neural_replay'], 'complete intake')
    require(summary['cases'] == 191 and summary['source_groups'] == packet['groups'], 'summary scope')
    require(summary['current_vs_historical_selected_query_changes'] == sum(changed_by_group.values()), 'summary actual changes')
    require(summary['groups'] == aggregate, 'independent all-row aggregate vs summary')
    require(not summary['new_overall_accuracy_claim'] and not summary['effective_module_claim'], 'claim ceiling')
    analysis_by_id = {r['row_id']: r for r in analyzed}
    require(len(analyzed) == len(analysis_by_id) == 191 and set(analysis_by_id) == set(cases), 'analysis rows')
    for v in verified:
        ar = analysis_by_id[v['row_id']]
        original = next(r for r in rows if r['row_id'] == v['row_id'])
        require(ar['roles'] == original['roles'] and ar['target_member_bounds'] == original['target_member_bounds'], 'analysis member evidence')
        require(ar['comparisons'] == v['comparisons'], 'independent IoUs vs CPU analysis')
    summary_comparison = 'ALL191_INDEPENDENT_CPU_MATCH'

result = dict(
    status='INDEPENDENT_CPU_SNAPSHOT_CHECKS_PASSED', phase=PHASE,
    generated_at=datetime.datetime.now().astimezone().isoformat(),
    reviewer_task='/root/pvg_support_boundary_actual_audit',
    execution_scope='ACTUAL_ARTIFACTS', execution_kind='CPU_NUMPY_ONLY',
    project_helper_imports=0, torch_imports=0, neural_forwards_by_reviewer=0,
    checkpoint_reads=0, network_calls=0,
    actual_rows_read=len(rows), historical_rows_read=len(history), cohort_rows_read=len(cohort),
    scenes=len({r['scan_id'] for r in rows}), scene_target_pairs=len({(r['scan_id'], r['target_id']) for r in rows}),
    recorded_query_changes=sum(changed_by_group.values()), query_changes_by_group=dict(changed_by_group),
    recorded_role_validity=dict(role_counts),
    archived_files_present_at_snapshot=len(available_paths), archived_files_verified=len(verified),
    in_flight_files=in_flight, not_present_count=191-len(available_paths),
    sample_point_labels_executed=point_labels_checked, archived_array_members_loaded=array_member_count,
    role_slices_executed=2*len(verified), faces_executed=faces_checked,
    archived_historical_coarse_box_changed_rows=sum(v['historical_coarse_box_max_difference'] > 2e-6 for v in verified),
    archived_historical_reference_box_changed_rows=sum(v['historical_reference_box_max_difference'] > 2e-6 for v in verified),
    archived_unchanged_selected_query_rows=sum(not v['selection_changed'] for v in verified),
    archived_unchanged_selected_query_but_coarse_changed=sum(not v['selection_changed'] and v['historical_coarse_box_max_difference'] > 2e-6 for v in verified),
    maximum_reference_bound_error_m=maximum_reference_bound_error,
    maximum_target_member_to_gt_bound_error_m=maximum_target_member_to_gt_bound_error,
    maximum_fusion_logit_error=maximum_fusion_error,
    summary_comparison=summary_comparison, verified_subset_groups=aggregate,
    archived_rows=verified, audited_input_hashes=HASHES,
    limitations=[
        'XYZ, membership and target labels are archived; RGB is absent. Full XYZRGB input SHA cannot be independently regenerated from these archives.',
        'All191 JSON records were read, but only archived_files_verified completed NPZ files were executed in this snapshot.',
        'Historical role indexes the historical numerical Query slot in the fresh forward; it does not recover the original stochastic Query/support.',
        'Incoming preliminary archives are a row-ordered subset and cannot estimate cohort-wide support rates.',
        'No underlying raw dataset payload or model weights were opened; GT provenance is established by imported-source hash, lineage records and actual archived GT members.'
    ])
out = ROOT / ('PRELIMINARY_CPU_CHECK.json' if PHASE == 'preliminary' else 'analysis/INDEPENDENT_CPU_AUDIT.json')
out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: result[key] for key in ('phase','actual_rows_read','scenes','scene_target_pairs','recorded_query_changes','query_changes_by_group','archived_files_present_at_snapshot','archived_files_verified','in_flight_files','not_present_count','sample_point_labels_executed','archived_array_members_loaded','role_slices_executed','faces_executed','archived_historical_coarse_box_changed_rows','archived_historical_reference_box_changed_rows','archived_unchanged_selected_query_rows','archived_unchanged_selected_query_but_coarse_changed','maximum_reference_bound_error_m','maximum_target_member_to_gt_bound_error_m','maximum_fusion_logit_error','summary_comparison','verified_subset_groups')}, indent=2))
