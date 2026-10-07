"""Once-only CPU provenance of old cached faces, after full immutable intake."""
import hashlib
import json
from pathlib import Path

import numpy as np
from historical_face_provenance import historical_faces, TOLERANCE_M

root = Path(__file__).resolve().parent
assert not (root / 'HISTORICAL_FACE_SUMMARY.json').exists()
assert json.loads((root / 'analysis/SUMMARY.json').read_bytes())['cases'] == 191
intake = json.loads((root / 'complete/INTAKE.json').read_bytes())
assert intake['status'] == 'CLOSED_ARTIFACTS_COLLECTED' and not intake['neural_replay']
manifest = json.loads((root / 'case_manifest.json').read_bytes())
actual = {row['row_id']: row for row in map(json.loads, (root / 'complete/rows.jsonl').read_text().splitlines())}
archived = {item['name']: item for item in intake['files']}
groups, rows = {}, []
for case in manifest['diagnostic_rows']:
    cached = case['cached']
    row = actual[cached['row_id']]
    assert row['point_sha256'] == case['point_sha256']
    name = 'arrays/row_%05d.npz' % cached['row_id']
    path = root / 'complete' / name
    item = archived[name]
    assert path.stat().st_size == item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    result = historical_faces(arrays, cached)
    result['historical_expression_group'] = case['group']
    rows.append(result)
    stat = groups.setdefault(case['group'], dict(rows=0, only_background_extreme_rows=0,
        all_possible_sources_pure_background_rows=0, any_possible_mixed_source_rows=0,
        only_background_extreme_faces=0, all_possible_sources_pure_background_faces=0,
        all_possible_sources_mixed_faces=0, ambiguous_member_faces=0,
        gt_missing_faces=0, gt_missing_faces_with_observed_target_beyond=0,
        gt_missing_faces_without_observed_target_beyond=0))
    faces = result['faces']
    stat['rows'] += 1
    stat['only_background_extreme_rows'] += int(any(face['member_certainty'] == 'only_background_member_possible' for face in faces))
    stat['all_possible_sources_pure_background_rows'] += int(any(face['all_possible_superpoints_pure_background'] for face in faces))
    stat['any_possible_mixed_source_rows'] += int(any(face['possible_mixed_superpoints'] for face in faces))
    for face in faces:
        stat['only_background_extreme_faces'] += int(face['member_certainty'] == 'only_background_member_possible')
        stat['all_possible_sources_pure_background_faces'] += int(face['all_possible_superpoints_pure_background'])
        stat['all_possible_sources_mixed_faces'] += int(face['all_possible_superpoints_mixed'])
        stat['ambiguous_member_faces'] += int(face['member_certainty'] == 'target_and_background_members_possible')
        missing = face['signed_gt_outward_error_m'] < -TOLERANCE_M
        stat['gt_missing_faces'] += int(missing)
        stat['gt_missing_faces_with_observed_target_beyond'] += int(missing and face['observed_target_members_beyond_face'] > 0)
        stat['gt_missing_faces_without_observed_target_beyond'] += int(missing and face['observed_target_members_beyond_face'] == 0)
assert len(rows) == 191
summary = dict(status='ACTUAL_CACHED_FACE_COORDINATE_PROVENANCE', cases=191, groups=groups,
    coordinate_tolerance_m=TOLERANCE_M, current_predicted_masks_used=False,
    old_query_or_foreground_recovered=False, optimizer_updates=0, neural_forwards=0,
    new_overall_accuracy_claim=False, effective_module_claim=False,
    scope='Possible extremal members at old cached reference coordinates in identical input points; no old mask recovery',
    limitations=['A background-only face is certain only when all coordinate candidates are background',
        'Possible mixed source does not prove the old mask selected it; row categories overlap',
        'GT qualification remains offline and never changes inference',
        'No observed target beyond a missing GT face is an input-support diagnostic, not a universal impossibility claim'])
(root / 'HISTORICAL_FACE_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
with (root / 'HISTORICAL_FACE_ROWS.jsonl').open('x', encoding='utf-8') as stream:
    for row in rows:
        stream.write(json.dumps(row) + '\n')
print(json.dumps(summary), flush=True)
