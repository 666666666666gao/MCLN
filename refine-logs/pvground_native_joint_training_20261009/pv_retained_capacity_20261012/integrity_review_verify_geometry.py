"""Independent, CPU-only audit of the explicitly supplied retained PV cache.

Reads the archived producer/probe inputs; writes only beside this script.
The neural producer is never imported. The inspected NumPy probe is replayed
after removing its three terminal write/print expressions from its AST.
"""
import ast
from collections import Counter
import datetime
import hashlib
import io
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
PROBE_DIR = OUT.parent
TMP = Path('C:/Users/gb/.codex/tmp')
PRODUCER = TMP / 'pvground_extremal_span_evidence_20261009/runner_v1/complete_fit'
SOURCE = PRODUCER / 'formal'
EARLIER = TMP / 'pvground_selected_mask_training_20261009/complete_fit/matched_mask_objective.py'
INPUTS = {}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    INPUTS[str(path)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return raw


def load_json(path):
    return json.loads(read(path))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def bounds(boxes):
    extent = np.maximum(boxes[..., 3:], 1e-6)
    return boxes[..., :3] - extent / 2, boxes[..., :3] + extent / 2


def overlap(low, high, target_low, target_high):
    lengths = np.clip(np.minimum(high, target_high) - np.maximum(low, target_low), 0, None)
    intersection = lengths[..., 0] * lengths[..., 1] * lengths[..., 2]
    volume = np.prod(high - low, axis=-1)
    target_volume = np.prod(target_high - target_low, axis=-1)
    return intersection / (volume + target_volume - intersection)


def native_iou(boxes, target):
    low = np.maximum(boxes[..., :3] - boxes[..., 3:] / 2, target[..., :3] - target[..., 3:] / 2)
    high = np.minimum(boxes[..., :3] + boxes[..., 3:] / 2, target[..., :3] + target[..., 3:] / 2)
    intersection = np.prod(np.maximum(high - low, 0), axis=-1)
    return intersection / (np.prod(boxes[..., 3:], axis=-1) + np.prod(target[..., 3:], axis=-1) - intersection)


receipt = load_json(SOURCE / 'receipt.json')
raw_rows = read(SOURCE / 'rows.jsonl')
rows = [json.loads(line) for line in raw_rows.splitlines()]
reported = load_json(PROBE_DIR / 'RETAINED_PV_GEOMETRIC_CAPACITY.json')
bindings = load_json(PROBE_DIR / 'ARRAY_BINDINGS.json')
manifest = load_json(PRODUCER / 'MANIFEST_STDOUT.json')
intake = load_json(PRODUCER / 'INTAKE.json')
spec = load_json(PRODUCER / 'pair_spec.json')
final_spec = load_json(PRODUCER / 'FINAL_SPEC_RECEIPT.json')
training_receipt = load_json(PRODUCER / 'receipt.json')
preflight = load_json(PRODUCER / 'preflight.json')
for name in ('formal.log', 'formal.exit', 'imports.json', 'NATIVE_COLLECT_STARTED.json',
             'NATIVE_COLLECT_EXIT.json', 'COLLECT_EXIT.json', 'MANIFEST_EXIT.json'):
    read(PRODUCER / name)
for name in spec['new_runner_files']:
    read(PRODUCER / name)
earlier_bytes = read(EARLIER)
math_reference = TMP / 'mcln_portable_best_transfer_20261012/geometric_capacity_audit/EXPERIMENT_AUDIT.md'
read(math_reference)
for name in ('experiment-audit/SKILL.md', 'shared-references/local-codex-policy.md',
             'shared-references/reviewer-independence.md', 'shared-references/experiment-integrity.md',
             'shared-references/review-tracing.md'):
    read(Path('C:/Users/gb/.codex/skills') / name)

assert len(rows) == receipt['rows'] == receipt['formal_rows'] == reported['rows'] == 9508
N = len(rows)
assert [r['row_id'] for r in rows] == list(range(N))
assert INPUTS[str(SOURCE / 'rows.jsonl')]['sha256'] == receipt['rows_sha256'] == reported['source_rows_sha256']
assert INPUTS[str(SOURCE / 'receipt.json')]['sha256'] == reported['source_receipt_sha256']
assert manifest['files'] == intake['files'] and intake['files_copied'] == len(manifest['files']) == 1246
manifest_map = {entry['name']: entry for entry in manifest['files']}
assert len(manifest_map) == len(manifest['files'])
assert len(bindings) == len({entry['name'] for entry in bindings}) == 1189
binding_map = {entry['name']: entry for entry in bindings}
assert final_spec['pair_spec_sha256'] == INPUTS[str(PRODUCER / 'pair_spec.json')]['sha256']
assert training_receipt['spec_sha256'] == final_spec['pair_spec_sha256']
assert training_receipt['status'] == 'complete'
assert earlier_bytes == read(PRODUCER / 'matched_mask_objective.py')
assert all(INPUTS[str(PRODUCER / name)]['sha256'] == digest for name, digest in spec['new_runner_files'].items())

boxes32 = {
    'native': np.asarray([r['controls']['native']['box'] for r in rows], dtype=np.float32),
    'mask': np.asarray([r['controls']['mask']['box'] for r in rows], dtype=np.float32),
    'final': np.asarray([r['arms']['extremal_support']['box'] for r in rows], dtype=np.float32),
}
target32 = np.asarray([r['root_box'] for r in rows], dtype=np.float32)
saved = {name: np.asarray([r['controls'][name]['iou'] for r in rows]) for name in ('native', 'mask')}
saved['final'] = np.asarray([r['arms']['extremal_support']['iou'] for r in rows])
mask_iou = np.asarray([r['mask_iou'] for r in rows])
gates = np.asarray([r['arms']['extremal_support']['axis_gate'] for r in rows], dtype=np.float64)
queries = np.asarray([r['query'] for r in rows], dtype=np.int64)
assert gates.shape == (N, 3) and np.isfinite(gates).all() and ((gates >= 0) & (gates <= 1)).all()
assert all(np.isfinite(v).all() for v in [*boxes32.values(), target32, mask_iou, *saved.values()])
assert (target32[:, 3:] > 0).all()
assert ((mask_iou >= 0) & (mask_iou <= 1)).all()

precision = {}
endpoint_values = {}
gl, gh = bounds(target32.astype(np.float64))
for name, box in boxes32.items():
    l32, h32 = bounds(box)
    g32l, g32h = bounds(target32)
    value32 = overlap(l32, h32, g32l, g32h)
    producer32 = native_iou(box, target32)
    lo, hi = bounds(box.astype(np.float64))
    value64 = overlap(lo, hi, gl, gh)
    endpoint_values[name] = value64
    precision[name] = {
        'boundary_float32_max_abs_error_from_saved': float(np.max(np.abs(value32 - saved[name]))),
        'producer_formula_float32_max_abs_error_from_saved': float(np.max(np.abs(producer32 - saved[name]))),
        'producer_formula_float32_bitwise_equal_to_saved': bool(np.array_equal(producer32, saved[name])),
        'boundary_float64_max_abs_error_from_saved': float(np.max(np.abs(value64 - saved[name]))),
        'threshold_disagreements': {str(t): {
            'boundary_float32': int(np.count_nonzero((value32 > t) != (saved[name] > t))),
            'producer_float32': int(np.count_nonzero((producer32 > t) != (saved[name] > t))),
            'boundary_float64': int(np.count_nonzero((value64 > t) != (saved[name] > t)))
        } for t in (.25, .5)}
    }
    assert all(v == 0 for d in precision[name]['threshold_disagreements'].values() for v in d.values())

nl, nh = bounds(boxes32['native'].astype(np.float64))
rl, rh = bounds(boxes32['mask'].astype(np.float64))
assert (nh > nl).all() and (rh > rl).all()
axis = np.empty(N, dtype=np.float64)
axis_gates = np.empty((N, 3), dtype=np.float64)
combination_counts = np.empty(N, dtype=np.int64)
for i in range(N):
    knots = []
    for a in range(3):
        current = {0., 1.}
        for first, last in ((rl[i, a], nl[i, a]), (rh[i, a], nh[i, a])):
            if first != last:
                for gt_face in (gl[i, a], gh[i, a]):
                    t = (gt_face - first) / (last - first)
                    if 0 < t < 1:
                        current.add(float(t))
        knots.append(sorted(current))
    choices = np.asarray(list(itertools.product(*knots)), dtype=np.float64)
    lows = rl[i] + choices * (nl[i] - rl[i])
    highs = rh[i] + choices * (nh[i] - rh[i])
    values = overlap(lows, highs, gl[i], gh[i])
    arg = int(values.argmax())
    axis[i], axis_gates[i], combination_counts[i] = values[arg], choices[arg], len(values)

# Independent median-of-three form of interval projection.
face_low = np.median(np.stack([gl, np.minimum(nl, rl), np.maximum(nl, rl)]), axis=0)
face_high = np.median(np.stack([gh, np.minimum(nh, rh), np.maximum(nh, rh)]), axis=0)
assert (face_high > face_low).all()
assert ((face_low >= np.minimum(nl, rl)) & (face_low <= np.maximum(nl, rl))).all()
assert ((face_high >= np.minimum(nh, rh)) & (face_high <= np.maximum(nh, rh))).all()
face = overlap(face_low, face_high, gl, gh)
assert np.min(axis - np.maximum(endpoint_values['native'], endpoint_values['mask'])) >= -1e-12
assert np.min(face - axis) >= -1e-12
assert np.max(endpoint_values['final'] - axis) < 1e-12
f_low, f_high = bounds(boxes32['final'].astype(np.float64))
recon_low = (1 - gates) * rl + gates * nl
recon_high = (1 - gates) * rh + gates * nh
gate32 = gates.astype(np.float32)
mixed32 = (1 - np.concatenate([gate32, gate32], axis=1)) * boxes32['mask'] + np.concatenate([gate32, gate32], axis=1) * boxes32['native']
reconstruction = {
    'float64_bound_max_error': max(float(np.max(np.abs(recon_low - f_low))), float(np.max(np.abs(recon_high - f_high)))),
    'producer_float32_center_size_bitwise_exact': bool(np.array_equal(mixed32, boxes32['final'])),
    'producer_float32_center_size_max_error': float(np.max(np.abs(mixed32 - boxes32['final']))),
    'maximum_final64_excess_over_axis64': float(np.maximum(endpoint_values['final'] - axis, 0).max()),
}
assert reconstruction['float64_bound_max_error'] < 1e-5

coverage = np.zeros(N, dtype=np.int64)
all256 = np.empty(N, dtype=np.float64)
all256_argmax = np.empty(N, dtype=np.int64)
score_max_ties = 0
oracle_disagreements = {str(t): 0 for t in (.25, .5)}
arrays_schema = Counter()
actual_bindings = []
for path in sorted(SOURCE.glob('batch_*.npz')):
    packed = read(path)
    current_binding = {'name': path.name, **INPUTS[str(path)]}
    actual_bindings.append(current_binding)
    assert current_binding == binding_map[path.name]
    with np.load(io.BytesIO(packed), allow_pickle=False) as bundle:
        assert set(bundle.files) == {'row_ids', 'root_gt', 'scores', 'native', 'mask', 'fixed_half', 'whole_support', 'extremal_support'}
        ids = bundle['row_ids']
        assert np.issubdtype(ids.dtype, np.integer)
        assert ids.ndim == 1 and len(ids) in (4, 8) and (ids >= 0).all() and (ids < N).all()
        start = int(path.stem.split('_')[1])
        assert np.array_equal(ids, np.arange(start, min(start + 8, N)))
        coverage[ids] += 1
        b = np.arange(len(ids))
        q = queries[ids]
        assert np.array_equal(bundle['root_gt'], target32[ids])
        assert bundle['scores'].shape == (len(ids), 256)
        assert np.isfinite(bundle['scores']).all()
        assert np.array_equal(bundle['scores'][b, q], bundle['scores'].max(axis=1))
        score_max_ties += int(np.count_nonzero((bundle['scores'] == bundle['scores'].max(axis=1, keepdims=True)).sum(axis=1) > 1))
        for name in ('native', 'mask', 'extremal_support', 'whole_support', 'fixed_half'):
            data = bundle[name]
            assert data.shape == (len(ids), 256, 6) and data.dtype == np.float32
            assert np.isfinite(data).all() and (data[..., 3:] > 0).all()
            arrays_schema[(name, str(data.dtype), 256, 6)] += 1
        for source_name, row_name in (('native', 'native'), ('mask', 'mask'), ('extremal_support', 'final')):
            assert np.array_equal(bundle[source_name][b, q], boxes32[row_name][ids])
        low, high = bounds(bundle['extremal_support'].astype(np.float64))
        values = overlap(low, high, gl[ids, None], gh[ids, None])
        all256[ids] = values.max(axis=1)
        all256_argmax[ids] = values.argmax(axis=1)
        for t in (.25, .5):
            historical = np.asarray([rows[int(index)]['arms']['extremal_support']['oracle' + ('25' if t == .25 else '50')][-1] for index in ids], dtype=bool)
            oracle_disagreements[str(t)] += int(np.count_nonzero((all256[ids] > t) != historical))
assert actual_bindings == bindings and (coverage == 1).all()
assert all(value == 0 for value in oracle_disagreements.values())

# Execute only the inspected NumPy probe. No source file is edited or written.
probe_path = PROBE_DIR / 'probe_retained_geometry.py'
probe_source = read(probe_path).decode('utf-8-sig')
tree = ast.parse(probe_source, filename=str(probe_path))
removed = [ast.unparse(node) for node in tree.body[-3:]]
assert all(isinstance(node, ast.Expr) for node in tree.body[-3:])
assert '.write_text(' in removed[0] and '.write_text(' in removed[1] and removed[2].startswith('print(')
tree.body = tree.body[:-3]
environment = {'__file__': str(probe_path), '__name__': 'audited_numpy_probe'}
exec(compile(tree, str(probe_path), 'exec'), environment)
replayed = environment['result']
assert {k: v for k, v in replayed.items() if k != 'recorded_cst'} == {k: v for k, v in reported.items() if k != 'recorded_cst'}
max_axis_difference = float(np.max(np.abs(axis - environment['axis_values'])))
assert max_axis_difference < 1e-12
assert np.array_equal(face, environment['face_values'])
assert np.array_equal(all256, environment['all256'])
for t in (.25, .5):
    assert np.array_equal(axis > t, environment['axis_values'] > t)

quartiles = np.quantile(np.prod(target32[:, 3:], axis=1), [.25, .5, .75])
size_group = np.digitize(np.prod(target32[:, 3:], axis=1), quartiles)
tables, partitions = [], {}
for t in (.25, .5):
    learned = saved['final'] > t
    axis_good, face_good, any_good = axis > t, face > t, all256 > t
    miss = ~learned
    tables.append({
        'threshold': t, 'actual_native_hits': int(np.sum(saved['native'] > t)),
        'actual_mask_reference_hits': int(np.sum(saved['mask'] > t)), 'actual_final_hits': int(np.sum(learned)),
        'GT_whole_box_choice_hits': int(np.sum((endpoint_values['native'] > t) | (endpoint_values['mask'] > t))),
        'GT_continuous_tied_axis_hits': int(np.sum(axis_good)), 'GT_independent_face_projection_hits': int(np.sum(face_good)),
        'axis_hits_outside_actual_final': int(np.sum(axis_good & miss)), 'actual_final_hits_outside_axis': int(np.sum(learned & ~axis_good)),
        'face_hits_outside_axis': int(np.sum(face_good & ~axis_good)),
        'actual_final_errors_unreachable_by_axis': int(np.sum(miss & ~axis_good)),
        'actual_final_errors_reachable_only_with_face_release': int(np.sum(miss & ~axis_good & face_good)),
        'smallest_half_axis_reachable_actual_errors': int(np.sum(miss & axis_good & (size_group < 2))),
        'smallest_half_face_extra_errors': int(np.sum(miss & ~axis_good & face_good & (size_group < 2)))})
    partition = {
        'actual_errors': int(np.sum(miss)), 'other_actual_candidate_qualified': int(np.sum(miss & any_good)),
        'no_actual_candidate_qualified': int(np.sum(miss & ~any_good)),
        'selected_current_axis_family_qualified': int(np.sum(miss & axis_good)),
        'no_axis_repair_but_other_actual_candidate_qualified': int(np.sum(miss & ~axis_good & any_good)),
        'neither_axis_repair_nor_other_actual_candidate_qualified': int(np.sum(miss & ~axis_good & ~any_good))}
    assert partition == reported['all256_failure_partition'][str(t)]
    partition['disjoint_2x2_among_actual_errors'] = {
        'axis_yes_candidate_yes': int(np.sum(miss & axis_good & any_good)),
        'axis_yes_candidate_no': int(np.sum(miss & axis_good & ~any_good)),
        'axis_no_candidate_yes': int(np.sum(miss & ~axis_good & any_good)),
        'axis_no_candidate_no': int(np.sum(miss & ~axis_good & ~any_good))}
    partition['all_rows_any_actual_candidate_hits'] = int(np.sum(any_good))
    partitions[str(t)] = partition
assert tables == reported['table']
matrix = {
    'both_good': int(np.sum((mask_iou > .5) & (saved['final'] > .5))),
    'box_only': int(np.sum((mask_iou <= .5) & (saved['final'] > .5))),
    'mask_only': int(np.sum((mask_iou > .5) & (saved['final'] <= .5))),
    'neither': int(np.sum((mask_iou <= .5) & (saved['final'] <= .5)))}
assert matrix == reported['selected_mask_box_matrix050']
assert [int(np.sum(mask_iou > t)) for t in (.25, .5)] == [receipt['mask_hits25'], receipt['mask_hits50']]
assert abs(float(mask_iou.mean() * 100) - receipt['mask_miou']) < 1e-12
crossing_rows = np.flatnonzero((np.maximum(nl, rl) >= np.minimum(nh, rh)).any(axis=1))
assert len(crossing_rows) == reported['arbitrary_independent_face_crossing_possible_rows']

provenance_checks = []
for path_text, observed in INPUTS.items():
    path = Path(path_text)
    if path.is_relative_to(PRODUCER):
        relative = path.relative_to(PRODUCER).as_posix()
        if relative in manifest_map:
            archived = manifest_map[relative]
            assert archived['sha256'] == observed['sha256'] and archived['bytes'] == observed['bytes'], relative
            provenance_checks.append(relative)
# Rehash every audited input after computation; no claims about remote re-attestation.
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value['sha256'] for path, value in INPUTS.items())
provenance = {
    'status': 'PASS_LOCAL_BYTES_MATCH_ORIGINAL_ARCHIVED_COLLECTION_MANIFEST',
    'original_collection_recorded_time': intake['time_cst'], 'original_manifest_entries': len(manifest['files']),
    'manifest_and_intake_entries_identical': True, 'audited_manifest_matched_files': len(provenance_checks),
    'matched_batch_arrays': len(actual_bindings), 'matched_names': provenance_checks,
    'current_binding_manifest_exact': True, 'new_runner_spec_hashes_match': True,
    'final_spec_pair_spec_hash_matches': True, 'earlier_matching_source_byte_identical': True,
    'all_inputs_unchanged_after_verification': True,
    'scope': 'Comparison against pre-existing locally archived original collection manifests, not new independent remote attestation; no remote raw dataset/checkpoint acquisition.'}
write('PROVENANCE_VERIFICATION.json', provenance)
write('INPUT_HASHES.json', INPUTS)
np.savez_compressed(OUT / 'CPU_VERIFIED_GEOMETRY.npz', row_ids=np.arange(N), queries=queries,
    native_iou64=endpoint_values['native'], mask_iou64=endpoint_values['mask'], final_iou64=endpoint_values['final'],
    saved_final_iou=saved['final'], axis_iou=axis, axis_gates=axis_gates, face_iou=face,
    face_low=face_low, face_high=face_high, all256_max_iou=all256, all256_argmax=all256_argmax,
    combination_counts=combination_counts, mask_iou=mask_iou, crossing_rows=crossing_rows)
evidence = {
    'status': 'PASS', 'review_independence': 'deterministic', 'acceptance_status': 'accepted',
    'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'python': platform.python_version(), 'python_executable': sys.executable, 'numpy': np.__version__,
    'elapsed_seconds': time.perf_counter() - START,
    'scope': {'rows': N, 'batch_files': len(actual_bindings), 'candidate_count_per_row': 256,
        'distinct_scans': len({r['scan_id'] for r in rows}),
        'distinct_scan_target_pairs': len({(r['scan_id'], r['target_id']) for r in rows}),
        'distinct_point_hashes': len({r['point_sha256'] for r in rows}),
        'query_range': [int(queries.min()), int(queries.max())], 'unique_and_complete_row_coverage': True,
        'all_selected_endpoints_bitwise_match_npz': True, 'all_selected_scores_equal_max': True,
        'rows_with_tied_max_scores': score_max_ties, 'seed': spec['seed'], 'split': 'ScanRefer val',
        'later_native_E0_rowwise_identity_verified': False},
    'precision': precision, 'reconstruction': reconstruction,
    'probe_replay': {'non_timestamp_result_exact': True, 'examples_verified': len(reported['examples']),
        'terminal_expressions_removed': removed, 'independent_max_axis_iou_difference': max_axis_difference,
        'independent_face_and_all256_arrays_exact': True,
        'independent_distinct_knot_combination_min_max': [int(combination_counts.min()), int(combination_counts.max())]},
    'geometry': {'minimum_endpoint_width': float(min(np.min(nh - nl), np.min(rh - rl))),
        'minimum_projected_width': float(np.min(face_high - face_low)),
        'minimum_axis_minus_endpoint_max': float(np.min(axis - np.maximum(endpoint_values['native'], endpoint_values['mask']))),
        'minimum_face_minus_axis': float(np.min(face - axis)),
        'arbitrary_independent_face_crossing_possible_rows': len(crossing_rows),
        'arbitrary_crossing_row_ids': crossing_rows.tolist(),
        'closest_axis_to_threshold': {str(t): float(np.min(np.abs(axis - t))) for t in (.25, .5)},
        'closest_face_to_threshold': {str(t): float(np.min(np.abs(face - t))) for t in (.25, .5)}},
    'table': tables, 'all256_failure_partition': partitions,
    'all256_historical_float32_oracle_flag_disagreements': oracle_disagreements,
    'selected_mask_box_matrix050': matrix,
    'size_quartile_cutpoints_float32_product': quartiles.tolist(),
    'size_quartile_group_counts': [int(np.sum(size_group == g)) for g in range(4)],
    'preflight_gradient_evidence': [{'step': w['step'], 'arms': {name: a['direct_output_gradient'] for name, a in w['arms'].items()}} for w in preflight['witnesses']],
    'new_neural_forwards': 0, 'new_optimizer_updates': 0, 'new_remote_calls': 0,
    'active_source_edits': 0, 'inputs_unchanged': True,
}
write('CPU_VERIFICATION.json', evidence)
print(json.dumps({k: evidence[k] for k in ('status', 'elapsed_seconds', 'scope', 'precision', 'reconstruction', 'probe_replay', 'geometry', 'table', 'all256_failure_partition', 'selected_mask_box_matrix050')}, indent=2))
