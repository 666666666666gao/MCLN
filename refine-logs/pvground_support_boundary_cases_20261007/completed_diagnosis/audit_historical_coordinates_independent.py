"""Independent all191 cached-coordinate audit, using four raw archive fields only."""
import collections
import datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
HASHES = {}
TOL = 2e-6


def read(path):
    raw = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def json_read(path):
    return json.loads(read(path))


def jsonl_read(path):
    return [json.loads(line) for line in read(path).decode('utf-8').splitlines()]


def bounds(value):
    value = np.asarray(value, dtype=np.float64)
    return np.concatenate((value[:3] - value[3:] / 2, value[:3] + value[3:] / 2))


intake = json_read(ROOT / 'complete/INTAKE.json')
assert intake['status'] == 'CLOSED_ARTIFACTS_COLLECTED' and not intake['neural_replay']
assert intake['optimizer_updates'] == intake['weights_copied'] == 0
inventory = {item['name']: item for item in intake['files']}
assert len(inventory) == len(intake['files']) == 205
actual_names = {path.relative_to(ROOT / 'complete').as_posix() for path in (ROOT / 'complete').rglob('*') if path.is_file()}
assert actual_names == set(inventory) | {'INTAKE.json'}
total_bytes = 0
for name, item in inventory.items():
    path = ROOT / 'complete' / name
    raw = read(path)
    assert len(raw) == item['bytes'] and HASHES[str(path)] == item['sha256'], name
    total_bytes += len(raw)
assert total_bytes == intake['bytes']
assert read(ROOT / 'complete/collector.exit').strip() == b'0'
assert read(ROOT / 'complete/controller.exit').strip() == b'0'

manifest = json_read(ROOT / 'case_manifest.json')
recorded_rows = jsonl_read(ROOT / 'HISTORICAL_FACE_ROWS.jsonl')
recorded = {r['row_id']: r for r in recorded_rows}
summary = json_read(ROOT / 'HISTORICAL_FACE_SUMMARY.json')
fresh = json_read(ROOT / 'analysis/INDEPENDENT_CPU_AUDIT.json')
original_records = {r['row_id']: r for r in jsonl_read(ROOT / 'complete/rows.jsonl')}
assert len(recorded) == len(recorded_rows) == len(manifest['diagnostic_rows']) == 191
assert summary['coordinate_tolerance_m'] == TOL
assert not summary['current_predicted_masks_used'] and not summary['old_query_or_foreground_recovered']
assert summary['optimizer_updates'] == summary['neural_forwards'] == 0
assert not summary['new_overall_accuracy_claim'] and not summary['effective_module_claim']
for name in ('historical_face_provenance.py', 'analyze_historical_faces.py'):
    read(ROOT / name)

counts = collections.Counter()
groups = {}
audited_rows = []
all_candidate_pairs = 0
all_candidate_target_pairs = 0
all_candidate_superpoint_pairs = 0
maximum_nearest_member_error = 0.0
arrays_loaded = 0
red_case_delta = None

for case in manifest['diagnostic_rows']:
    cached = case['cached']
    row_id = cached['row_id']
    assert cached['reference_valid']
    result = recorded[row_id]
    assert original_records[row_id]['point_sha256'] == case['point_sha256']
    assert result['historical_query_index'] == cached['query']
    assert result['historical_reference_box'] == cached['reference_box']
    assert result['historical_expression_group'] == case['group']
    assert result['current_masks_used'] is False and result['historical_foreground_reconstructed'] is False
    archive = ROOT / 'complete/arrays' / ('row_%05d.npz' % row_id)
    raw = read(archive)
    with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
        # Deliberately do not load a fresh predicted mask, score, Query feature or box.
        xyz = loaded['xyz'].astype(np.float64)
        gt_labels = loaded['target']
        membership = loaded['superpoint'].astype(np.int64)
        root_gt = loaded['root_gt']
    arrays_loaded += 4
    assert xyz.shape == (50000,3) and gt_labels.shape == membership.shape == (50000,)
    assert gt_labels.dtype == np.bool_ and gt_labels.any()
    assert root_gt.tolist() == cached['root_gt']
    old_faces = bounds(cached['reference_box'])
    gt_faces = bounds(root_gt)
    assert len(result['faces']) == 6
    stat = groups.setdefault(case['group'], dict(rows=0, only_background_extreme_rows=0,
        all_possible_sources_pure_background_rows=0, any_possible_mixed_source_rows=0,
        only_background_extreme_faces=0, all_possible_sources_pure_background_faces=0,
        all_possible_sources_mixed_faces=0, ambiguous_member_faces=0,
        gt_missing_faces=0, gt_missing_faces_with_observed_target_beyond=0,
        gt_missing_faces_without_observed_target_beyond=0))
    per_row = dict(row_id=row_id, group=case['group'], faces=[])
    has_bg = has_pure_bg = has_possible_mixed = False
    for direction in range(6):
        face = result['faces'][direction]
        axis = direction % 3
        position = old_faces[direction]
        distance = np.abs(xyz[:,axis] - position)
        # Enumerate the complete coordinate tolerance slab, with no fresh mask restriction.
        candidates = np.nonzero(distance <= TOL)[0]
        assert len(candidates) > 0
        maximum_nearest_member_error = max(maximum_nearest_member_error, float(distance.min()))
        num_target = int(np.count_nonzero(gt_labels[candidates]))
        num_bg = len(candidates) - num_target
        candidate_sps = sorted(set(membership[candidates].tolist()))
        kinds = collections.Counter()
        for sp_id in candidate_sps:
            raw_member_labels = gt_labels[membership == sp_id]
            positive = int(np.count_nonzero(raw_member_labels))
            kinds['background' if positive == 0 else 'target' if positive == len(raw_member_labels) else 'mixed'] += 1
        certainty = ('only_background_member_possible' if num_target == 0 else
                     'only_target_member_possible' if num_bg == 0 else
                     'target_and_background_members_possible')
        sign = -1.0 if direction < 3 else 1.0
        signed_error = float(sign * (position - gt_faces[direction]))
        outside_count = int(np.count_nonzero(sign * (xyz[gt_labels,axis] - position) > TOL))
        expected = dict(face=('x-','y-','z-','x+','y+','z+')[direction],
            historical_reference_coordinate=float(position), coordinate_candidate_points=len(candidates),
            possible_target_points=num_target, possible_background_points=num_bg,
            member_certainty=certainty, possible_superpoints=candidate_sps,
            possible_pure_background_superpoints=kinds['background'],
            possible_pure_target_superpoints=kinds['target'], possible_mixed_superpoints=kinds['mixed'],
            all_possible_superpoints_pure_background=kinds['background'] == len(candidate_sps),
            all_possible_superpoints_mixed=kinds['mixed'] == len(candidate_sps),
            signed_gt_outward_error_m=signed_error, observed_target_members_beyond_face=outside_count)
        assert face == expected, (row_id, direction)
        missing = signed_error < -TOL
        only_bg = num_target == 0
        pure_bg = kinds['background'] == len(candidate_sps)
        all_mixed = kinds['mixed'] == len(candidate_sps)
        ambiguous = num_target > 0 and num_bg > 0
        stat['only_background_extreme_faces'] += int(only_bg)
        stat['all_possible_sources_pure_background_faces'] += int(pure_bg)
        stat['all_possible_sources_mixed_faces'] += int(all_mixed)
        stat['ambiguous_member_faces'] += int(ambiguous)
        stat['gt_missing_faces'] += int(missing)
        stat['gt_missing_faces_with_observed_target_beyond'] += int(missing and outside_count > 0)
        stat['gt_missing_faces_without_observed_target_beyond'] += int(missing and outside_count == 0)
        has_bg |= only_bg
        has_pure_bg |= pure_bg
        has_possible_mixed |= kinds['mixed'] > 0
        counts[certainty] += 1
        counts['faces'] += 1
        counts['possible_mixed_superpoint_faces'] += int(kinds['mixed'] > 0)
        counts['all_possible_superpoints_pure_target_faces'] += int(kinds['target'] == len(candidate_sps))
        counts['faces_with_multiple_coordinate_candidates'] += int(len(candidates) > 1)
        counts['faces_with_observed_target_beyond'] += int(outside_count > 0)
        counts['candidate_superpoint_background_occurrences'] += kinds['background']
        counts['candidate_superpoint_target_occurrences'] += kinds['target']
        counts['candidate_superpoint_mixed_occurrences'] += kinds['mixed']
        all_candidate_pairs += len(candidates)
        all_candidate_target_pairs += num_target
        all_candidate_superpoint_pairs += len(candidate_sps)
        per_row['faces'].append(dict(face=expected['face'], coordinate_candidate_points=len(candidates),
            candidate_indices_sha256=hashlib.sha256(candidates.astype('<i8').tobytes()).hexdigest(),
            member_certainty=certainty, possible_mixed_superpoints=kinds['mixed'],
            gt_missing=bool(missing), observed_target_members_beyond_face=outside_count,
            nearest_member_coordinate_error_m=float(distance.min())))
    stat['rows'] += 1
    stat['only_background_extreme_rows'] += int(has_bg)
    stat['all_possible_sources_pure_background_rows'] += int(has_pure_bg)
    stat['any_possible_mixed_source_rows'] += int(has_possible_mixed)
    audited_rows.append(per_row)

assert len(audited_rows) == 191 and counts['faces'] == 1146
assert summary['groups'] == groups, 'independently enumerated historical group totals'
assert summary['cases'] == 191
redcheck = json_read(ROOT / 'HISTORICAL_RECOVERY_RED_CHECK.json')
fresh_by_id = {row['row_id']: row for row in fresh['archived_rows']}
red_actual = fresh_by_id[redcheck['row_id']]
assert red_actual['queries'][1] == redcheck['numeric_query']
assert red_actual['historical_coarse_box_max_difference'] == redcheck['max_absolute_difference_m']
assert red_actual['historical_coarse_box_max_difference'] > redcheck['tolerance_m']
local_check = json_read(ROOT / 'HISTORICAL_FACE_LOCAL_CHECK.json')
assert local_check['formal_validation_cases'] == 0 and local_check['neural_forwards'] == 0
source_review = json_read(ROOT / 'HISTORICAL_FACE_SOURCE_REVIEW.json')

result = dict(status='ALL191_HISTORICAL_COORDINATE_PROVENANCE_INDEPENDENT_MATCH',
    generated_at=datetime.datetime.now().astimezone().isoformat(), execution_scope='ACTUAL_ARTIFACTS',
    reviewer_task='/root/pvg_support_boundary_actual_audit', cases_executed=191, faces_executed=1146,
    intake_files_size_and_sha256_verified=len(inventory), intake_total_bytes_verified=total_bytes,
    archived_npz_files_executed=191, array_fields_loaded=arrays_loaded,
    only_archive_fields_used=['xyz','target','superpoint','root_gt'], raw_point_rows_executed=191*50000,
    point_to_face_coordinate_comparisons=191*50000*6,
    coordinate_candidate_face_point_pairs=all_candidate_pairs,
    coordinate_candidate_target_face_point_pairs=all_candidate_target_pairs,
    coordinate_candidate_background_face_point_pairs=all_candidate_pairs-all_candidate_target_pairs,
    coordinate_candidate_face_superpoint_pairs=all_candidate_superpoint_pairs,
    maximum_nearest_member_coordinate_error_m=maximum_nearest_member_error,
    tolerance_m=TOL, independent_face_counts=dict(counts), groups=groups,
    historical_query_or_foreground_recovered=False, fresh_predicted_mask_fields_loaded=0,
    redcheck_numeric_difference_independently_confirmed=True,
    redcheck_repeated_failing_executions_rerun_by_reviewer=0,
    synthetic_local_check_formal_cases_counted=0,
    helper_imports=0, torch_imports=0, neural_forwards=0, checkpoint_reads=0,
    network_calls=0, optimizer_updates=0,
    limits=[
        'Coordinate candidates are all raw members within the declared axial tolerance, a conservative superset of possible old support. No fresh mask or feature narrows them.',
        'Mixed-source possibility does not prove the old foreground selected that superpoint, and ambiguous target/background candidates remain ambiguous.',
        'A coordinate with only background candidates certifies background-only coordinate provenance within this tolerance and recorded input identity; it does not reconstruct the historical foreground.',
        'Observed target points beyond a missing cached face show omission in the old extent, not a trainable solution or universal recoverability.',
        'Full XYZRGB hashes are collector-recorded because RGB is not archived; the raw dataset payload was not reopened.',
        'The historical-recovery redcheck remains rejected. Its actual numeric discrepancy is independently verified; its two earlier exit1 executions were not replayed.',
        'Source review and synthetic local check were read but are not counted as actual191-case evidence.'
    ], archived_rows=audited_rows, audited_input_hashes=HASHES)
destination = ROOT / 'analysis/INDEPENDENT_HISTORICAL_FACE_AUDIT.json'
destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: result[key] for key in ('status','cases_executed','faces_executed','intake_files_size_and_sha256_verified','intake_total_bytes_verified','archived_npz_files_executed','array_fields_loaded','raw_point_rows_executed','point_to_face_coordinate_comparisons','coordinate_candidate_face_point_pairs','coordinate_candidate_target_face_point_pairs','coordinate_candidate_background_face_point_pairs','coordinate_candidate_face_superpoint_pairs','maximum_nearest_member_coordinate_error_m','independent_face_counts','groups','redcheck_numeric_difference_independently_confirmed')}, indent=2))
