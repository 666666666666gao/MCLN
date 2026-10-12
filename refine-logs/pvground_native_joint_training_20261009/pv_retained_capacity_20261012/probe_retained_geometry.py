"""Saved PV retained-source geometry capacity; GT choices never enter inference."""
import datetime
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
source = Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/formal')
receipt_path = source / 'receipt.json'
receipt = json.loads(receipt_path.read_bytes())
rows_path = source / 'rows.jsonl'
raw = rows_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
assert receipt['formal_rows'] == receipt['rows'] == len(rows) == 9508
assert receipt['metrics']['extremal_support']['rec_hits25'] == 5677
assert receipt['metrics']['extremal_support']['rec_hits50'] == 4920
assert [row['row_id'] for row in rows] == list(range(9508))
native32 = np.asarray([row['controls']['native']['box'] for row in rows], dtype=np.float32)
reference32 = np.asarray([row['controls']['mask']['box'] for row in rows], dtype=np.float32)
final32 = np.asarray([row['arms']['extremal_support']['box'] for row in rows], dtype=np.float32)
gt32 = np.asarray([row['root_box'] for row in rows], dtype=np.float32)
saved = {arm: np.asarray([row['controls'][arm]['iou'] for row in rows])
         for arm in ('native', 'mask')}
saved['final'] = np.asarray([row['arms']['extremal_support']['iou'] for row in rows])
gates = np.asarray([row['arms']['extremal_support']['axis_gate'] for row in rows], dtype=np.float64)
assert gates.shape == (9508, 3) and ((gates >= 0) & (gates <= 1)).all()


def bounds(box):
    size = np.maximum(box[..., 3:], 1e-6)
    return box[..., :3] - size * .5, box[..., :3] + size * .5


def iou(low, high, gt_low, gt_high):
    intersection = np.maximum(np.minimum(high, gt_high) - np.maximum(low, gt_low), 0).prod(-1)
    volume = (high - low).prod(-1)
    gt_volume = (gt_high - gt_low).prod(-1)
    return intersection / (volume + gt_volume - intersection)


precision = {}
for arm, box in (('native', native32), ('mask', reference32), ('final', final32)):
    low, high = bounds(box)
    gl, gh = bounds(gt32)
    values = iou(low, high, gl, gh)
    precision[arm] = dict(float32_max_error_against_saved=float(np.abs(values - saved[arm]).max()),
                         threshold_disagreements={str(t): int(((values > t) != (saved[arm] > t)).sum())
                                                  for t in (.25, .5)})
    assert not any(precision[arm]['threshold_disagreements'].values())
nlow, nhigh = bounds(native32.astype(np.float64))
rlow, rhigh = bounds(reference32.astype(np.float64))
glow, ghigh = bounds(gt32.astype(np.float64))
assert (nhigh > nlow).all() and (rhigh > rlow).all()
native64 = iou(nlow, nhigh, glow, ghigh)
reference64 = iou(rlow, rhigh, glow, ghigh)
fl, fh = bounds(final32.astype(np.float64))
final64 = iou(fl, fh, glow, ghigh)
expected_low = (1 - gates) * rlow + gates * nlow
expected_high = (1 - gates) * rhigh + gates * nhigh
reconstruction_error = max(float(np.abs(expected_low - fl).max()), float(np.abs(expected_high - fh).max()))
assert reconstruction_error < 1e-5

# Fixed-other-axis IoU is fractional-linear between face/GT crossing knots.
# The Cartesian knot set contains a continuous tied-axis maximum in real
# arithmetic. This diagnostic numerically evaluates it with float64.
knots = [np.zeros_like(nlow), np.ones_like(nlow)]
for gt_face in (glow, ghigh):
    for first, last in ((rlow, nlow), (rhigh, nhigh)):
        delta = last - first
        value = np.divide(gt_face - first, delta, out=np.zeros_like(delta), where=delta != 0)
        knots.append(np.clip(value, 0, 1))
knots = np.stack(knots, 1)
choices = np.asarray(list(itertools.product(range(6), repeat=3)), dtype=np.int64)
axis_values, axis_gates = [], []
for start in range(0, len(rows), 128):
    current = knots[start:start + 128]
    candidate = np.stack([current[:, choices[:, a], a] for a in range(3)], -1)
    low = (1 - candidate) * rlow[start:start + 128, None] + candidate * nlow[start:start + 128, None]
    high = (1 - candidate) * rhigh[start:start + 128, None] + candidate * nhigh[start:start + 128, None]
    values = iou(low, high, glow[start:start + 128, None], ghigh[start:start + 128, None])
    selected = values.argmax(-1)
    axis_values.extend(values[np.arange(len(current)), selected].tolist())
    axis_gates.extend(candidate[np.arange(len(current)), selected].tolist())
axis_values = np.asarray(axis_values)
axis_gates = np.asarray(axis_gates)
face_low = np.clip(glow, np.minimum(nlow, rlow), np.maximum(nlow, rlow))
face_high = np.clip(ghigh, np.minimum(nhigh, rhigh), np.maximum(nhigh, rhigh))
assert (face_high > face_low).all()
face_values = iou(face_low, face_high, glow, ghigh)
assert (axis_values + 1e-12 >= np.maximum(native64, reference64)).all()
assert (face_values + 1e-12 >= axis_values).all()
mask_ious = np.asarray([row['mask_iou'] for row in rows])
volume = gt32[:, 3:].prod(-1)
cuts = np.quantile(volume, [.25, .5, .75])
group = np.digitize(volume, cuts)
table = []
for threshold in (.25, .5):
    final_good = saved['final'] > threshold
    axis_good = axis_values > threshold
    face_good = face_values > threshold
    table.append(dict(threshold=threshold, actual_native_hits=int((saved['native'] > threshold).sum()),
                      actual_mask_reference_hits=int((saved['mask'] > threshold).sum()),
                      actual_final_hits=int(final_good.sum()),
                      GT_whole_box_choice_hits=int(((native64 > threshold) | (reference64 > threshold)).sum()),
                      GT_continuous_tied_axis_hits=int(axis_good.sum()),
                      GT_independent_face_projection_hits=int(face_good.sum()),
                      axis_hits_outside_actual_final=int((axis_good & ~final_good).sum()),
                      actual_final_hits_outside_axis=int((final_good & ~axis_good).sum()),
                      face_hits_outside_axis=int((face_good & ~axis_good).sum()),
                      actual_final_errors_unreachable_by_axis=int((~final_good & ~axis_good).sum()),
                      actual_final_errors_reachable_only_with_face_release=int((~final_good & ~axis_good & face_good).sum()),
                      smallest_half_axis_reachable_actual_errors=int((~final_good & axis_good & (group < 2)).sum()),
                      smallest_half_face_extra_errors=int((~final_good & ~axis_good & face_good & (group < 2)).sum())))
all256 = np.zeros(9508, dtype=np.float64)
array_bindings = []
batch_paths = sorted(source.glob('batch_*.npz'))
assert len(batch_paths) == 1189
for path in batch_paths:
    packed = path.read_bytes()
    array_bindings.append(dict(name=path.name, bytes=len(packed), sha256=hashlib.sha256(packed).hexdigest()))
    with np.load(str(path), allow_pickle=False) as arrays:
        ids = arrays['row_ids']
        assert np.array_equal(arrays['root_gt'], gt32[ids])
        q = np.asarray([rows[int(index)]['query'] for index in ids], dtype=np.int64)
        b = np.arange(len(ids))
        assert np.array_equal(arrays['extremal_support'][b, q], final32[ids])
        assert np.array_equal(arrays['native'][b, q], native32[ids])
        assert np.array_equal(arrays['mask'][b, q], reference32[ids])
        assert np.array_equal(arrays['scores'][b, q], arrays['scores'].max(1))
        low, high = bounds(arrays['extremal_support'].astype(np.float64))
        gt_low, gt_high = bounds(arrays['root_gt'].astype(np.float64))
        all256[ids] = iou(low, high, gt_low[:, None], gt_high[:, None]).max(1)
coverage = {}
for threshold in (.25, .5):
    wrong = saved['final'] <= threshold
    coverage[str(threshold)] = dict(actual_errors=int(wrong.sum()),
        other_actual_candidate_qualified=int((wrong & (all256 > threshold)).sum()),
        no_actual_candidate_qualified=int((wrong & (all256 <= threshold)).sum()),
        selected_current_axis_family_qualified=int((wrong & (axis_values > threshold)).sum()),
        no_axis_repair_but_other_actual_candidate_qualified=int((wrong & (axis_values <= threshold) & (all256 > threshold)).sum()),
        neither_axis_repair_nor_other_actual_candidate_qualified=int((wrong & (axis_values <= threshold) & (all256 <= threshold)).sum()))
examples = []
extra = np.flatnonzero((face_values > .5) & (axis_values <= .5) & (saved['final'] <= .5))
extra = extra[np.argsort(-(face_values[extra] - axis_values[extra]))][:12]
for index in extra:
    row = rows[int(index)]
    examples.append(dict(row_id=int(index), scan_id=row['scan_id'], target_id=row['target_id'], query=row['query'],
        actual_native_iou=float(saved['native'][index]), actual_mask_iou=float(saved['mask'][index]),
        actual_final_iou=float(saved['final'][index]), GT_axis_iou=float(axis_values[index]),
        GT_face_iou=float(face_values[index]), actual_axis_gate=gates[index].tolist(),
        GT_axis_gate=axis_gates[index].tolist(), native_box=native32[index].tolist(),
        reference_box=reference32[index].tolist(), final_box=final32[index].tolist(), root_box=gt32[index].tolist()))
result = dict(status='CPU_GT_GEOMETRIC_CAPACITY_OF_RETAINED_PV_SOURCE_CACHE',
    recorded_cst=datetime.datetime.now().astimezone().isoformat(), rows=9508,
    source_snapshot='20261009_frozen_parent_A_B_formal_cache_with_retained_5677_4920_counts',
    source_rows_sha256=hashlib.sha256(raw).hexdigest(), source_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    precision=precision, gate_box_reconstruction_max_error=reconstruction_error,
    maximum_actual_final64_excess_over_axis64=float(np.maximum(final64 - axis_values, 0).max()),
    table=table, all256_failure_partition=coverage,
    selected_mask_box_matrix050=dict(both_good=int(((mask_ious > .5) & (saved['final'] > .5)).sum()),
        box_only=int(((mask_ious <= .5) & (saved['final'] > .5)).sum()),
        mask_only=int(((mask_ious > .5) & (saved['final'] <= .5)).sum()),
        neither=int(((mask_ious <= .5) & (saved['final'] <= .5)).sum())),
    arbitrary_independent_face_crossing_possible_rows=int((np.maximum(nlow, rlow) >= np.minimum(nhigh, rhigh)).any(1).sum()),
    examples=examples, GT_chooses_all_diagnostic_geometry=True, method_accuracy_claim=False,
    new_neural_forwards=0, optimizer_updates=0, GPU_calls=0, remote_queries=0,
    active_training_changes=0, retained_best_changed=False, full_goal_complete=False,
    limitations=['GT-conditioned endpoint-family values are not model accuracy, deployable choices or promised gains.',
        'This exact cache produced the retained-source count; equal counts alone do not prove bitwise correspondence to the later native E0 replay.',
        'All256 overlap is offline GT geometry; it does not prove learnable selection or physical-instance identity.',
        'Continuous and projected geometries are numerical float64 evaluations of the real-arithmetic families.',
        'No new six-face architecture, new matching responsibility, learning-rate recipe or cross-dataset performance was tested.'])
(root / 'ARRAY_BINDINGS.json').write_text(json.dumps(array_bindings, indent=2) + '\n', encoding='utf-8')
(root / 'RETAINED_PV_GEOMETRIC_CAPACITY.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: result[key] for key in ('status', 'table', 'all256_failure_partition', 'selected_mask_box_matrix050')}))
