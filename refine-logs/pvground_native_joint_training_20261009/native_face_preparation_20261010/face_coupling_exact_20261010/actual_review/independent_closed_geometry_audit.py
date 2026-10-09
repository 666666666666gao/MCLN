"""Independent NumPy-only verification of already closed selected-query rows."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path('C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2')
SOURCE = Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1')
OUT = ROOT / 'face_coupling_exact_20261010/actual_review'
TRACE = ROOT / '.aris/traces/experiment-audit/2026-10-10_face_coupling'
assert 'torch' not in sys.modules
before = json.loads((TRACE / 'input_sha256_before.json').read_bytes())
for filename, expected in before.items():
    blob = Path(filename).read_bytes()
    assert len(blob) == expected['bytes']
    assert hashlib.sha256(blob).hexdigest() == expected['sha256']

raw = (SOURCE / 'complete_fit/formal/rows.jsonl').read_bytes()
receipt_raw = (SOURCE / 'complete_fit/formal/receipt.json').read_bytes()
recount_raw = (SOURCE / 'postrun_results/CPU_RECOUNT.json').read_bytes()
receipt, recount = json.loads(receipt_raw), json.loads(recount_raw)
claimed = json.loads((ROOT / 'face_coupling_exact_20261010/FACE_COUPLING_EXACT_DIAGNOSTIC.json').read_bytes())
rows_hash = hashlib.sha256(raw).hexdigest()
assert rows_hash == receipt['rows_sha256'] == recount['formal_rows_sha256'] == claimed['source_formal_rows_sha256']
assert hashlib.sha256(receipt_raw).hexdigest() == recount['formal_receipt_sha256']
assert hashlib.sha256(recount_raw).hexdigest() == claimed['source_CPU_recount_sha256']
assert receipt['status'] == 'pass' and recount['status'] == 'CLOSED_SPAN_CPU_RECOUNT_COMPLETE'
records = [json.loads(line) for line in raw.splitlines()]
assert len(records) == 9508
assert [r['row_id'] for r in records] == list(range(9508))
assert all(0 <= r['query'] < 256 for r in records)
assert all(r['parent_forwards'] == r['final_semantic_head_calls'] == 1 for r in records)

NAMES = ('native', 'mask', 'fixed_half', 'whole_support', 'extremal_support')
stored = {name: [r['arms' if name in NAMES[3:] else 'controls'][name] for r in records] for name in NAMES}
boxes32 = {name: np.array([r['box'] for r in stored[name]], dtype=np.float32) for name in NAMES}
truth32 = np.array([r['root_box'] for r in records], dtype=np.float32)
assert all(b.shape == (9508, 6) and np.isfinite(b).all() and (b[:, 3:] > 0).all()
           for b in [truth32, *boxes32.values()])


def faces(b):
    radius = b[:, 3:] * .5
    return b[:, :3] - radius, b[:, :3] + radius


def overlap_iou(lo, hi, tlo, thi):
    shared = np.maximum(np.minimum(hi, thi) - np.maximum(lo, tlo), 0).prod(axis=-1)
    volume = (hi - lo).prod(axis=-1)
    target_volume = (thi - tlo).prod(axis=-1)
    union = volume + target_volume - shared
    assert (union > 0).all()
    return shared / union


# Match deployed center/size float32 volume arithmetic without importing the source.
def deployed_iou(b, t):
    lo, hi = faces(b)
    tlo, thi = faces(t)
    shared = np.maximum(np.minimum(hi, thi) - np.maximum(lo, tlo), 0).prod(axis=1)
    return shared / (b[:, 3:].prod(axis=1) + t[:, 3:].prod(axis=1) - shared)


actual32 = {name: deployed_iou(b, truth32) for name, b in boxes32.items()}
actual64 = {name: deployed_iou(b.astype(np.float64), truth32.astype(np.float64)) for name, b in boxes32.items()}
counts = {name: [int(np.count_nonzero(v > t)) for t in (.25, .5)] for name, v in actual32.items()}
for name in NAMES:
    assert counts[name] == recount['table'][name]['hits'] == claimed['counts'][name]
    expected = receipt['same_forward_controls'][name] if name in NAMES[:3] else [
        receipt['metrics'][name]['rec_hits25'], receipt['metrics'][name]['rec_hits50']]
    assert counts[name] == expected
    saved_iou = np.array([r['iou'] for r in stored[name]], dtype=np.float64)
    assert np.max(np.abs(saved_iou - actual32[name])) < 1e-5
    assert all(np.array_equal(saved_iou > t, actual32[name] > t) for t in (.25, .5))
    assert all(np.array_equal(actual64[name] > t, actual32[name] > t) for t in (.25, .5))

nl, nh = faces(boxes32['native'].astype(np.float64))
ml, mh = faces(boxes32['mask'].astype(np.float64))
gl, gh = faces(truth32.astype(np.float64))
dl, dh = nl - ml, nh - mh
assert (nh > nl).all() and (mh > ml).all() and (gh > gl).all()

# Build each actual breakpoint set independently: retain only roots in (0,1),
# then pad the sorted unique set with its final endpoint. This is not the
# diagnostic's clipping construction and requires no division by zero.
points = np.empty((9508, 3, 6), dtype=np.float64)
unique_breakpoint_counts = np.empty((9508, 3), dtype=np.int64)
for r in range(9508):
    for axis in range(3):
        knots = {0., 1.}
        for base, slope in ((ml[r, axis], dl[r, axis]), (mh[r, axis], dh[r, axis])):
            if slope != 0.:
                for target in (gl[r, axis], gh[r, axis]):
                    root = (target - base) / slope
                    if 0. < root < 1.:
                        knots.add(root)
        knots = sorted(knots)
        unique_breakpoint_counts[r, axis] = len(knots)
        points[r, axis] = knots + [1.] * (6 - len(knots))

candidate_low = ml[:, :, None] + dl[:, :, None] * points
candidate_high = mh[:, :, None] + dh[:, :, None] * points
length = candidate_high - candidate_low
intersection = np.maximum(np.minimum(candidate_high, gh[:, :, None]) -
                          np.maximum(candidate_low, gl[:, :, None]), 0.)
assert np.isfinite(length).all() and (length > 0).all()
target_length = gh - gl
axis_best = np.max(intersection / (length + target_length[:, :, None] - intersection), axis=2)
projected_bound = np.min(axis_best, axis=1)

# All 6x6x6 products are evaluated with corner volumes directly, avoiding the
# diagnostic's center/size reconstruction and its combination loop.
shared = (intersection[:, 0, :, None, None] * intersection[:, 1, None, :, None] *
          intersection[:, 2, None, None, :])
volume = length[:, 0, :, None, None] * length[:, 1, None, :, None] * length[:, 2, None, None, :]
union = volume + target_length.prod(axis=1)[:, None, None, None] - shared
assert (union > 0).all()
cube = shared / union
exact = cube.max(axis=(1, 2, 3))
argmax = np.array(np.unravel_index(cube.reshape(9508, -1).argmax(axis=1), (6, 6, 6))).T
best_gate = np.take_along_axis(points, argmax[:, :, None], axis=2)[:, :, 0]

ind_lo = np.maximum(np.minimum(gl, np.maximum(ml, nl)), np.minimum(ml, nl))
ind_hi = np.maximum(np.minimum(gh, np.maximum(mh, nh)), np.minimum(mh, nh))
assert (ind_hi > ind_lo).all()
assert ((ind_lo >= np.minimum(ml, nl)) & (ind_lo <= np.maximum(ml, nl))).all()
assert ((ind_hi >= np.minimum(mh, nh)) & (ind_hi <= np.maximum(mh, nh))).all()
independent = overlap_iou(ind_lo, ind_hi, gl, gh)
for projected, base, slope in ((ind_lo, ml, dl), (ind_hi, mh, dh)):
    weights = np.divide(projected - base, slope, out=np.zeros_like(slope), where=slope != 0)
    assert ((weights >= 0) & (weights <= 1)).all()
    assert np.max(np.abs(base + weights * slope - projected)) < 1e-12

assert np.max(exact - projected_bound) < 1e-12
assert np.max(exact - independent) < 1e-12
observed_exact_excess = {name: float((v - exact).max()) for name, v in actual64.items()}
observed_projected_excess = {name: float((v - projected_bound).max()) for name, v in actual64.items()}
assert max(observed_exact_excess.values()) < 1e-6
assert max(observed_projected_excess.values()) < 1e-6

gate_reconstruction = {}
for name in NAMES[3:]:
    gate = np.array([r['axis_gate'] for r in stored[name]], dtype=np.float64)
    assert gate.shape == (9508, 3) and np.isfinite(gate).all() and ((gate >= 0) & (gate <= 1)).all()
    expanded = np.concatenate((gate, gate), axis=1)
    reconstructed = (1. - expanded) * boxes32['mask'].astype(np.float64) + expanded * boxes32['native'].astype(np.float64)
    residual = np.abs(reconstructed - boxes32[name].astype(np.float64))
    assert residual.max() < 2e-6
    gate_reconstruction[name] = {'maximum_absolute_center_or_size_residual': float(residual.max()),
                                 'minimum_gate': float(gate.min()), 'maximum_gate': float(gate.max())}

margin = 1e-6
wrong = actual32['extremal_support'] <= .5
fused = np.array([r['mask_iou'] for r in records], dtype=np.float64)
fused_good = fused > .5
lo_delta = np.abs(ml - gl) - np.abs(nl - gl)
hi_delta = np.abs(mh - gh) - np.abs(nh - gh)
opposed = lo_delta * hi_delta < 0
both_margin = opposed & (np.minimum(np.abs(lo_delta), np.abs(hi_delta)) > .02)
exact_bad = exact <= .5 - margin
ind_good = independent > .5 + margin
evidence = exact_bad & ind_good
assert np.all(~evidence | wrong)
table = {}
for name, select in (('all', np.ones(9508, bool)), ('selected_strict_errors', wrong),
                     ('selected_fused_mask_qualified_strict_errors', wrong & fused_good)):
    table[name] = {
        'rows': int(select.sum()),
        'opposite_faces_prefer_different_endpoints': int((select & opposed.any(axis=1)).sum()),
        'opposite_preferences_both_margins_over_2cm': int((select & both_margin.any(axis=1)).sum()),
        'coupled_projection_upper_bound_below_half': int((select & (projected_bound <= .5 - margin)).sum()),
        'exact_continuous_coupled_oracle_below_half': int((select & exact_bad).sum()),
        'exact_continuous_coupled_oracle_above_half': int((select & (exact > .5 + margin)).sum()),
        'independent_GT_face_projection_above_half': int((select & ind_good).sum()),
        'independent_projection_qualifies_but_all_coupled_axis_gates_excluded_by_bound': int((select & evidence).sum())}
assert table == claimed['table']
oracle_hits = [int((exact > t + margin).sum()) for t in (.25, .5)]
ind_hits = [int((independent > t + margin).sum()) for t in (.25, .5)]
assert oracle_hits == claimed['exact_coupled_oracle_hits']
assert ind_hits == claimed['independent_face_projection_hits']
assert int((dl == 0).sum()) == claimed['constant_lower_slopes']
assert int((dh == 0).sum()) == claimed['constant_upper_slopes']

with (ROOT / 'face_coupling_exact_20261010/rows.csv').open(newline='', encoding='utf-8') as stream:
    csv_rows = list(csv.DictReader(stream))
assert len(csv_rows) == 9508
float_columns = {'native_iou': actual32['native'], 'mask_iou': actual32['mask'],
                 'fixed_half_iou': actual32['fixed_half'], 'extremal_iou': actual32['extremal_support'],
                 'coupled_3d_upper_bound': projected_bound,
                 'GT_independent_face_projection_iou': independent, 'exact_coupled_iou': exact,
                 'fused_mask_iou': fused}
csv_residuals = {}
for key, values in float_columns.items():
    actual = np.array([float(r[key]) for r in csv_rows])
    residual = float(np.max(np.abs(actual - values)))
    tolerance = 6e-8 if key in ('native_iou', 'mask_iou', 'fixed_half_iou', 'extremal_iou') else 1e-12
    assert residual < tolerance
    csv_residuals[key] = residual
for idx, (source_row, csv_row) in enumerate(zip(records, csv_rows)):
    for key in ('row_id', 'target_id', 'query'):
        assert int(csv_row[key]) == source_row[key]
    assert csv_row['scan_id'] == source_row['scan_id']
    assert int(csv_row['opposed_axes']) == opposed[idx].sum()
    assert int(csv_row['bound_below_half_and_independent_above_half']) == int(evidence[idx])

disjoint = (nl > mh) | (ml > nh)
loose_only_exclusions = (projected_bound <= .5 - margin) & ind_good
summary = {
    'status': 'PASS_CLOSED_OFFLINE_GEOMETRY_RECOMPUTATION',
    'execution_scope': 'CLOSED_OFFLINE_GEOMETRY_DIAGNOSTIC',
    'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'python': sys.executable, 'numpy': np.__version__,
    'source_rows_sha256': rows_hash,
    'rows': len(records), 'unique_scenes': len({r['scan_id'] for r in records}),
    'unique_scene_targets': len({(r['scan_id'], r['target_id']) for r in records}),
    'deployed_counts': counts,
    'exact_coupled_oracle_hits_margin_1e_6': oracle_hits,
    'independent_projection_hits_margin_1e_6': ind_hits,
    'table': table,
    'constant_lower_slopes': int((dl == 0).sum()),
    'constant_upper_slopes': int((dh == 0).sum()),
    'constant_both_faces_same_axis': int(((dl == 0) & (dh == 0)).sum()),
    'minimum_native_mask_gt_size': float(min(b[:, 3:].min() for b in [truth32, boxes32['native'], boxes32['mask']])),
    'minimum_coupled_candidate_size': float(length.min()),
    'minimum_independent_projection_size': float((ind_hi - ind_lo).min()),
    'unique_breakpoints_min_max': [int(unique_breakpoint_counts.min()), int(unique_breakpoint_counts.max())],
    'maximum_exact_minus_projected_bound': float((exact - projected_bound).max()),
    'maximum_exact_minus_independent_projection': float((exact - independent).max()),
    'observed_boxes_maximum_excess_over_exact': observed_exact_excess,
    'observed_boxes_maximum_excess_over_projected_bound': observed_projected_excess,
    'gate_reconstruction': gate_reconstruction,
    'threshold_neighborhood_counts': {
        str(t): {'exact': int((np.abs(exact - t) <= margin).sum()),
                 'independent': int((np.abs(independent - t) <= margin).sum())}
        for t in (.25, .5)},
    'witnesses': {'count': int(evidence.sum()), 'fused_mask_qualified_count': int((evidence & fused_good).sum()),
                  'minimum_half_minus_exact': float((.5 - exact[evidence]).min()),
                  'minimum_independent_minus_half': float((independent[evidence] - .5).min()),
                  'minimum_independent_minus_exact': float((independent[evidence] - exact[evidence]).min()),
                  'loose_projection_bound_alone_exclusion_count': int(loose_only_exclusions.sum()),
                  'exact_but_not_loose_exclusion_count': int((evidence & ~loose_only_exclusions).sum())},
    'native_mask_disjoint_intervals': {'axis_order': ['x', 'y', 'z'],
                                     'per_axis_count': disjoint.sum(axis=0).tolist(),
                                     'total_row_axis_count': int(disjoint.sum()),
                                     'expressions_with_any_axis': int(disjoint.any(axis=1).sum()),
                                     'strict_definition': 'native_low > mask_high or mask_low > native_high'},
    'csv_maximum_absolute_residuals': csv_residuals,
    'torch_imported': 'torch' in sys.modules,
    'ssh_calls': 0, 'neural_calls': 0, 'gpu_calls': 0, 'new_formal_evaluations': 0,
    'limits': ['Independent face projection is a GT-informed feasible witness, not a learned policy.',
               'Only the same stored selected query and native/Mask endpoint boxes are considered.',
               'The legacy CSV bound_below_half label encodes the exact-oracle condition in this version.',
               'FUSED mask qualification does not establish own-query mask qualification.']}
assert not summary['torch_imported']
with (OUT / 'independent_witnesses.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['row_id', 'scan_id', 'target_id', 'query', 'fused_mask_iou', 'extremal_iou',
                     'exact_coupled_iou', 'independent_projection_iou', 'loose_1d_bound',
                     'best_gate_x', 'best_gate_y', 'best_gate_z', 'fused_mask_qualified'])
    for idx in np.flatnonzero(evidence):
        row = records[idx]
        writer.writerow([row['row_id'], row['scan_id'], row['target_id'], row['query'], fused[idx],
                         actual32['extremal_support'][idx], exact[idx], independent[idx], projected_bound[idx],
                         *best_gate[idx], int(fused_good[idx])])
(OUT / 'INDEPENDENT_RECOMPUTATION.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
after = {filename: {'sha256': hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
                    'bytes': Path(filename).stat().st_size} for filename in before}
assert after == before
(TRACE / 'input_sha256_after.json').write_text(json.dumps(after, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
