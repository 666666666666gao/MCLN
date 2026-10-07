"""Recount only the collected191 cases, with no model or training invocation."""
import hashlib
import json
from pathlib import Path

import numpy as np
from support_evidence import analyze_case

root = Path(__file__).resolve().parent
destination = root / 'analysis'
assert not destination.exists()
wait = json.loads((root / 'wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode'] == 0
complete = root / 'complete'
intake = json.loads((complete / 'INTAKE.json').read_bytes())
receipt = json.loads((complete / 'receipt.json').read_bytes())
assert intake['status'] == 'CLOSED_ARTIFACTS_COLLECTED' and not intake['neural_replay']
assert receipt['cases'] == 191 and receipt['optimizer_updates'] == receipt['weights_created'] == 0
rows = [json.loads(line) for line in (complete / 'rows.jsonl').read_text().splitlines()]
packet = json.loads((root / 'case_manifest.json').read_bytes())
cases = {item['cached']['row_id']: item for item in packet['diagnostic_rows']}
assert len(rows) == len(cases) == 191 and {row['row_id'] for row in rows} == set(cases)
destination.mkdir()

def iou(box, truth):
    size = np.maximum(box[3:].astype(np.float64), 1e-6)
    gt_size = truth[3:].astype(np.float64)
    low = np.maximum(box[:3] - size / 2, truth[:3] - gt_size / 2)
    high = np.minimum(box[:3] + size / 2, truth[:3] + gt_size / 2)
    intersection = np.maximum(high - low, 0).prod()
    return float(intersection / (size.prod() + gt_size.prod() - intersection))

groups = {}
actual_rows = []
changed = 0
for row in rows:
    case = cases[row['row_id']]
    assert row['point_sha256'] == case['point_sha256'] and row['group'] == case['group']
    path = complete / 'arrays' / row['arrays']
    assert path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    evidence = analyze_case(arrays)
    assert evidence['roles'] == row['roles'] and evidence['target_member_bounds'] == row['target_member_bounds']
    assert arrays['queries'].tolist() == [row['deployed_query'], row['historical_query']]
    assert arrays['root_gt'].tolist() == case['cached']['root_gt']
    changed += int(row['selection_changed'])
    comparisons = []
    for slot, value in enumerate(evidence['roles']):
        key = row['group'] + '/' + value['role']
        stat = groups.setdefault(key, dict(rows=0, invalid_ranges=0, any_pure_background_extreme_rows=0,
            any_mixed_extreme_rows=0, any_background_extremal_point_rows=0, excessive_faces=0,
            missing_faces=0, pure_background_extreme_faces=0, mixed_extreme_faces=0,
            excessive_background_extreme_faces=0, coarse_hits25=0, reference_hits25=0,
            coarse_hits50=0, reference_hits50=0, query_changed_rows=0))
        stat['rows'] += 1
        stat['query_changed_rows'] += int(row['selection_changed'])
        stat['invalid_ranges'] += int(not value['valid_mask_range'])
        faces = value['faces']
        stat['any_pure_background_extreme_rows'] += int(any(face['pure_background_superpoints'] for face in faces))
        stat['any_mixed_extreme_rows'] += int(any(face['mixed_superpoints'] for face in faces))
        stat['any_background_extremal_point_rows'] += int(any(face['extremal_target_points'] < face['extremal_points'] for face in faces))
        for face in faces:
            excessive = face['signed_outward_error_m'] > 1e-6
            missing = face['signed_outward_error_m'] < -1e-6
            background = face['extremal_target_points'] < face['extremal_points']
            stat['excessive_faces'] += int(excessive)
            stat['missing_faces'] += int(missing)
            stat['pure_background_extreme_faces'] += int(bool(face['pure_background_superpoints']))
            stat['mixed_extreme_faces'] += int(bool(face['mixed_superpoints']))
            stat['excessive_background_extreme_faces'] += int(excessive and background)
        prior_iou = iou(arrays['coarse_boxes'][slot], arrays['root_gt'])
        reference_iou = iou(arrays['reference_boxes'][slot], arrays['root_gt'])
        for threshold, suffix in ((.25, '25'), (.5, '50')):
            stat['coarse_hits' + suffix] += int(prior_iou > threshold)
            stat['reference_hits' + suffix] += int(reference_iou > threshold)
        comparisons.append(dict(role=value['role'], query=value['query'], coarse_iou=prior_iou, reference_iou=reference_iou))
    actual_rows.append(dict(row_id=row['row_id'], group=row['group'], selection_changed=row['selection_changed'],
        comparisons=comparisons, **evidence))
summary = dict(status='ACTUAL_TARGETED_SUPPORT_DIAGNOSIS_RECOUNTED', cases=191, source_groups=packet['groups'],
    groups=groups, current_vs_historical_selected_query_changes=changed, exact_point_and_gt_identities=True,
    actual_nn_forwards=receipt['forward_batches'], optimizer_updates=0, weights_created=0,
    new_overall_accuracy_claim=False, effective_module_claim=False, retained_best_hits=[5598,4848],
    eval_gumbel_sampling=True, historical_full_sequence_rng_reproduced=False,
    historical_role='Fresh-forward numeric Query slot of the historical selected index; not historical prediction recovery',
    scope='Selected191 historical development errors, not full validation incidence or new training targets',
    excessive_or_missing_face_tolerance_m=1e-6,
    limitations=['Descriptive GT grouping only; no GT rule or new scoring enters deployment',
        'A mixed superpoint is a real member-label mixture, not proof every member is wrong',
        'Row-level background and mixed-extreme groups overlap; their counts must not be summed',
        'Current and historical Query roles are explicit; changed predictions are not erased',
        'Native eval still samples Gumbel weights; subset execution does not reproduce the old full-sequence RNG',
        'Historical grouping identifies expressions only; numeric-slot evidence cannot establish historical paired prediction causality'])
(destination / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
with (destination / 'rows.jsonl').open('x', encoding='utf-8') as stream:
    for row in actual_rows:
        stream.write(json.dumps(row) + '\n')
print(json.dumps(summary), flush=True)
