"""CPU-only audit; original experiment artifacts are only read."""
import ast
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

started = time.monotonic()
root = Path(__file__).resolve().parent.parent
outdir = Path(__file__).resolve().parent
hashes = json.loads((outdir / 'INPUT_HASHES.json').read_bytes())
for name in ['GEOMETRIC_PARAMETERIZATION_DIAGNOSIS.md', 'source/src/visual_data_handlers.py']:
    raw = (root / name).read_bytes()
    hashes[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
(outdir / 'INPUT_HASHES.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
raw = (root / 'full_paired_rows.jsonl').read_bytes()
binding = json.loads((root / 'full_ROWS_TRANSFER.json').read_bytes())
assert len(raw) == binding['bytes']
assert hashlib.sha256(raw).hexdigest() == binding['sha256']
rows = [json.loads(line) for line in raw.splitlines()]
assert [r['index'] for r in rows] == list(range(9508))
arms = ['native', 'mask_reference', 'full_transfer']
assert all(r['native']['root_gt_box'] == r['mask_reference']['root_gt_box'] == r['full_transfer']['root_gt_box'] for r in rows)
assert all(r['native']['selected_query'] == r['mask_reference']['selected_query'] == r['full_transfer']['selected_query'] for r in rows)
assert all(0 <= r['native']['selected_query'] < 256 for r in rows)
boxes = {a: np.array([r[a]['selected_box'] for r in rows], dtype=np.float64) for a in arms}
gt = np.array([r['native']['root_gt_box'] for r in rows], dtype=np.float64)
saved = {a: np.array([r[a]['selected_iou'] for r in rows], dtype=np.float64) for a in arms}
assert all(np.isfinite(b).all() and (b[:, 3:] > 0).all() for b in [*boxes.values(), gt])

def endpoints(b):
    size = np.maximum(b[..., 3:], np.array(1e-6, dtype=b.dtype))
    return b[..., :3] - size / np.array(2, dtype=b.dtype), b[..., :3] + size / np.array(2, dtype=b.dtype)

def measure(low, high, truth_low, truth_high):
    overlap = np.maximum(0, np.minimum(high, truth_high) - np.maximum(low, truth_low))
    intersection = overlap.prod(axis=-1)
    return intersection / ((high - low).prod(axis=-1) + (truth_high - truth_low).prod(axis=-1) - intersection)

gl, gh = endpoints(gt)
rl, rh = endpoints(boxes['mask_reference'])
nl, nh = endpoints(boxes['native'])
metric32 = {}
metric64 = {}
precision = {}
for arm in arms:
    metric32[arm] = measure(*endpoints(boxes[arm].astype(np.float32)), *endpoints(gt.astype(np.float32)))
    metric64[arm] = measure(*endpoints(boxes[arm]), gl, gh)
    assert np.array_equal(metric32[arm].astype(np.float64), saved[arm])
    precision[arm] = {
        'float32_exact_match_saved': True,
        'float64_max_absolute_difference_from_saved': float(np.max(np.abs(metric64[arm] - saved[arm]))),
        'threshold_disagreements': {str(t): int(np.count_nonzero((metric64[arm] > t) != (saved[arm] > t))) for t in [.25, .5]},
    }

# Re-execute the reviewed probe code with only its final write/print expressions
# omitted from the AST. This leaves all input/output experiment files unchanged.
replayed = {}
replay_ns = {}
for script, result_name in [
    ('probe_axis_mixture_capacity.py', 'AXIS_MIXTURE_CAPACITY_PROBE.json'),
    ('probe_continuous_axis_face_capacity.py', 'CONTINUOUS_AXIS_FACE_CAPACITY_PROBE.json'),
]:
    parsed = ast.parse((root / script).read_text(encoding='utf-8'), filename=script)
    assert isinstance(parsed.body[-2], ast.Expr) and isinstance(parsed.body[-1], ast.Expr)
    assert isinstance(parsed.body[-2].value.func, ast.Attribute) and parsed.body[-2].value.func.attr == 'write_text'
    assert isinstance(parsed.body[-1].value.func, ast.Name) and parsed.body[-1].value.func.id == 'print'
    parsed.body = parsed.body[:-2]
    namespace = {'__file__': str(root / script), '__name__': '__audit_cpu_replay__'}
    exec(compile(parsed, str(root / script), 'exec'), namespace)
    expected = json.loads((root / result_name).read_bytes())
    assert namespace['result'] == expected
    replayed[script] = {'full_json_value_match': True, 'examples_verified': len(expected['examples'])}
    replay_ns[script] = namespace

# Independent knot enumeration: keep only distinct, in-domain crossings, using
# affine interpolation r + t*(n-r), rather than the implementation's fixed 216
# clipped/padded combinations and (1-t)*r+t*n evaluation order.
optimal = []
candidate_counts = []
for i in range(len(rows)):
    per_axis = []
    for axis in range(3):
        choices = {0., 1.}
        for lower, upper in [(rl[i, axis], nl[i, axis]), (rh[i, axis], nh[i, axis])]:
            if upper != lower:
                for target in [gl[i, axis], gh[i, axis]]:
                    crossing = (target - lower) / (upper - lower)
                    if 0 <= crossing <= 1:
                        choices.add(float(crossing))
        per_axis.append(sorted(choices))
    gates = np.array(list(itertools.product(*per_axis)), dtype=np.float64)
    values = measure(rl[i] + gates * (nl[i] - rl[i]), rh[i] + gates * (nh[i] - rh[i]), gl[i], gh[i])
    optimal.append(float(values.max()))
    candidate_counts.append(len(gates))
optimal = np.array(optimal)
axis = replay_ns['probe_continuous_axis_face_capacity.py']['axis_iou']
face = replay_ns['probe_continuous_axis_face_capacity.py']['face_iou']
grid = replay_ns['probe_axis_mixture_capacity.py']['best_iou']
assert np.max(np.abs(optimal - axis)) < 1e-12
assert all(np.array_equal(optimal > t, axis > t) for t in [.25, .5])

# Same 729 grid points in float64 separate finite-grid error from precision.
nodes = np.linspace(0, 1, 9, dtype=np.float64)
gates = np.array(list(itertools.product(nodes, repeat=3)))
grid64 = []
for start in range(0, len(rows), 128):
    low = rl[start:start+128, None] + gates[None] * (nl[start:start+128, None] - rl[start:start+128, None])
    high = rh[start:start+128, None] + gates[None] * (nh[start:start+128, None] - rh[start:start+128, None])
    grid64.extend(measure(low, high, gl[start:start+128, None], gh[start:start+128, None]).max(axis=1).tolist())
grid64 = np.array(grid64)
assert np.all(axis + 1e-12 >= grid64)

low_min, low_max = np.minimum(rl, nl), np.maximum(rl, nl)
high_min, high_max = np.minimum(rh, nh), np.maximum(rh, nh)
project_low = np.minimum(np.maximum(gl, low_min), low_max)
project_high = np.minimum(np.maximum(gh, high_min), high_max)
project_iou = measure(project_low, project_high, gl, gh)
assert np.array_equal(face, project_iou)
assert (project_low >= low_min).all() and (project_low <= low_max).all()
assert (project_high >= high_min).all() and (project_high <= high_max).all()
assert (project_high > project_low).all()
assert np.all(face + 1e-12 >= axis)
invalid_possible = (low_max >= high_min).any(axis=1)

tables = []
for threshold in [.25, .5]:
    endpoint = (saved['native'] > threshold) | (saved['mask_reference'] > threshold)
    tables.append({
        'threshold': threshold,
        **{a + '_hits': int((saved[a] > threshold).sum()) for a in arms},
        'whole_box_GT_choice_hits': int(endpoint.sum()),
        'grid_float32_hits': int((grid > threshold).sum()),
        'same_grid_float64_hits': int((grid64 > threshold).sum()),
        'continuous_axis_float64_hits': int((axis > threshold).sum()),
        'independent_face_float64_hits': int((face > threshold).sum()),
        'continuous_extra_over_grid': int(((axis > threshold) & (grid64 <= threshold)).sum()),
        'faces_extra_over_axis': int(((face > threshold) & (axis <= threshold)).sum()),
        'full_transfer_hits_outside_axis_hits': int(((saved['full_transfer'] > threshold) & (axis <= threshold)).sum()),
        'axis_hits_outside_full_transfer_hits': int(((axis > threshold) & (saved['full_transfer'] <= threshold)).sum()),
        'grid32_grid64_threshold_disagreements': int(((grid > threshold) != (grid64 > threshold)).sum()),
    })

new_continuous = np.flatnonzero((axis > .5) & (grid64 <= .5))
report = {
    'status': 'PASS_DETERMINISTIC_CPU_RECOMPUTATION',
    'python': sys.version, 'python_executable': sys.executable,
    'numpy': np.__version__, 'platform': platform.platform(),
    'rows': len(rows), 'unique_scans': len({r['scan_id'] for r in rows}),
    'unique_scan_target_pairs': len({(r['scan_id'], r['target_id']) for r in rows}),
    'unique_scan_target_utterance_triples': len({(r['scan_id'], r['target_id'], r['utterance']) for r in rows}),
    'unique_point_sha256': len({r['point_sha256'] for r in rows}),
    'contiguous_unique_indices': True, 'all_branch_GT_boxes_identical': True,
    'all_selected_queries_identical': True, 'all_box_values_finite_and_sizes_positive': True,
    'selected_query_min': min(r['native']['selected_query'] for r in rows),
    'selected_query_max': max(r['native']['selected_query'] for r in rows),
    'minimum_size_by_arm': {a: float(boxes[a][:, 3:].min()) for a in arms},
    'input_binding_match': True, 'row_bytes': len(raw), 'row_sha256': binding['sha256'],
    'original_probe_replay': replayed, 'saved_metrics_precision': precision,
    'independent_axis_enumeration_max_abs_difference': float(np.max(np.abs(optimal - axis))),
    'independent_axis_candidate_count_min': min(candidate_counts),
    'independent_axis_candidate_count_max': max(candidate_counts),
    'grid32_grid64_max_absolute_difference': float(np.max(np.abs(grid.astype(np.float64) - grid64))),
    'projection_valid_boxes': int((project_high > project_low).all(axis=1).sum()),
    'projection_minimum_extent': float((project_high - project_low).min()),
    'projection_interval_violations': 0,
    'face_minus_axis_minimum': float((face - axis).min()),
    'independent_six_face_unconstrained_gate_invalid_possible_rows': int(invalid_possible.sum()),
    'independent_six_face_unconstrained_gate_invalid_example_index': int(np.flatnonzero(invalid_possible)[0]),
    'full_transfer_iou_exceeds_axis_float64_rows': int((metric64['full_transfer'] > axis + 1e-12).sum()),
    'table': tables,
    'continuous_extra_over_grid_at_05': [{'index': int(i), 'scan_id': rows[i]['scan_id'], 'grid_float32_iou': float(grid[i]), 'grid_float64_iou': float(grid64[i]), 'axis_float64_iou': float(axis[i])} for i in new_continuous],
    'minimum_distance_to_threshold': {
        name: {str(t): float(np.abs(values - t).min()) for t in [.25, .5]}
        for name, values in [('grid32', grid), ('grid64', grid64), ('axis64', axis), ('face64', face)]
    },
    'new_neural_forwards': 0, 'new_optimizer_updates': 0, 'new_GPU_queries': 0,
    'experiment_inputs_unchanged': all(hashlib.sha256((root / n).read_bytes()).hexdigest() == h['sha256'] for n, h in hashes.items()),
    'elapsed_seconds': time.monotonic() - started,
}
assert report['experiment_inputs_unchanged']
(outdir / 'CPU_VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
np.savez(outdir / 'CPU_VERIFIED_GEOMETRY.npz', axis_iou=axis, independent_axis_iou=optimal, face_iou=face, grid_float32_iou=grid, grid_float64_iou=grid64, projected_lower=project_low, projected_upper=project_high)
print(json.dumps(report, indent=2))
