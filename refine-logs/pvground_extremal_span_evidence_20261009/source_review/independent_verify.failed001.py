"""Fresh CPU/NumPy evidence verification; no Torch import or neural execution."""
import ast
import contextlib
import datetime
import hashlib
import inspect
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent
TMP = ROOT.parent
CONTROL = TMP / 'pvground_mask_box_controls_20261009'
SUPPORT = TMP / 'pvground_compressed_geometry_support_20261008'
PARENT = TMP / 'pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/models'
SELECTED = TMP / 'pvground_selected_mask_training_20261009'
BUNDLE = TMP / 'pvground_final_quality_20261005/runtime_bundle'
SKILLS = Path(r'C:\Users\gb\.codex\skills')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (REVIEW / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


paths = [ROOT / name for name in (
    'SOURCE_AUDIT_REQUEST.txt', 'analyze_axis_span_diagnostic.py', 'AXIS_SPAN_DIAGNOSTIC.json',
    'extremal_span_mixer.py', 'span_refinement_model.py', 'matched_span_objective.py', 'METHOD_PROPOSAL.md')]
paths += [CONTROL / name for name in ('SUMMARY.json', 'SELECTED_BOX_ROWS.jsonl',
                                    'audit/EXPERIMENT_AUDIT.json', 'analyze_fixed_query_boxes.py')]
paths += [SUPPORT / name for name in ('mask_support_corrector.py', 'mask_reference.py')]
paths += [PARENT / name for name in ('pv_ground.py', 'losses.py')]
paths += [SELECTED / name for name in ('matched_mask_objective.py', 'pair_spec.json',
                                     'support_pair_forward.py', 'paired_support_loop.py')]
paths += [BUNDLE / name for name in ('whole_mask_range.py', 'pvground_boundary_box_refiner.py')]
paths += [TMP / 'pvground_mask_reference_20261006/run_geometry_fit.py',
          TMP / 'pvground_geometry_readback_20261004/revision2/readback_preflight_checks.py',
          TMP / 'pvground_runtime_bundle_20260908_v1/env_spec.json',
          TMP / 'pvground_cs_restart_20261002/env_spec.json',
          REVIEW / 'torch_v1_10_2_FunctionsManual.cpp',
          SKILLS / 'experiment-audit/SKILL.md']
paths += [SKILLS / 'shared-references' / name for name in (
    'local-codex-policy.md', 'reviewer-independence.md', 'experiment-integrity.md',
    'review-tracing.md', 'integration-contract.md')]
hashes = {str(path): dict(sha256=digest(path), bytes=path.stat().st_size) for path in paths}
write('INPUT_HASHES.json', hashes)
asts = {str(path): ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for path in paths if path.suffix == '.py'}
summary = json.loads((CONTROL / 'SUMMARY.json').read_bytes())
prior_audit = json.loads((CONTROL / 'audit/EXPERIMENT_AUDIT.json').read_bytes())
saved = json.loads((ROOT / 'AXIS_SPAN_DIAGNOSTIC.json').read_bytes())
rows = [json.loads(line) for line in (CONTROL / 'SELECTED_BOX_ROWS.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(rows) == 9508
assert [r['row_id'] for r in rows] == list(range(9508))
assert all(0 <= r['query'] < 256 for r in rows)
assert digest(CONTROL / 'SELECTED_BOX_ROWS.jsonl') == summary['output_rows_sha256'] == saved['source_rows_sha256']
assert digest(CONTROL / 'SUMMARY.json') == saved['source_controls_summary_sha256']
assert digest(CONTROL / 'audit/EXPERIMENT_AUDIT.json') == saved['source_controls_audit_sha256']
assert digest(ROOT / 'analyze_axis_span_diagnostic.py') == saved['source_script_sha256']
assert prior_audit['verdict'] in ('PASS', 'WARN') and prior_audit['blocking_findings'] == []
gt = np.array([r['root_box'] for r in rows], dtype=np.float64)
mask = np.array([r['boxes']['mask_reference'] for r in rows], dtype=np.float64)
native = np.array([r['boxes']['native_regression'] for r in rows], dtype=np.float64)
for arr in (gt, mask, native):
    assert arr.shape == (9508, 6) and np.isfinite(arr).all() and (arr[:, 3:] > 0).all()

# Independently solve each two-coordinate convex line segment, per axis.
# The scalar loop and the endpoint/KKT checks do not call the submitted solver.
weights = np.zeros((9508, 3), dtype=np.float64)
zero_axes = 0
for i in range(9508):
    for axis in range(3):
        start = mask[i, [axis, axis + 3]]
        direction = native[i, [axis, axis + 3]] - start
        target = gt[i, [axis, axis + 3]]
        squared_length = np.dot(direction, direction)
        if squared_length == 0:
            zero_axes += 1
            continue
        coordinate = np.dot(direction, target - start) / squared_length
        value = min(1.0, max(0.0, float(coordinate)))
        weights[i, axis] = value
        residual = start + value * direction - target
        derivative = 2 * np.dot(direction, residual)
        if value == 0:
            assert derivative >= -1e-12
        elif value == 1:
            assert derivative <= 1e-12
        else:
            assert abs(derivative) <= 1e-10
        assert np.dot(residual, residual) <= np.dot(start - target, start - target) + 1e-12
        assert np.dot(residual, residual) <= np.dot(start + direction - target, start + direction - target) + 1e-12

diagnostic = mask.copy()
diagnostic[:, :3] += weights * (native[:, :3] - mask[:, :3])
diagnostic[:, 3:] += weights * (native[:, 3:] - mask[:, 3:])
assert np.isfinite(diagnostic).all() and (diagnostic[:, 3:] > 0).all()
assert ((diagnostic - gt) ** 2).sum(1).max() >= 0
assert (((diagnostic - gt) ** 2).sum(1) <= ((mask - gt) ** 2).sum(1) + 1e-12).all()
assert (((diagnostic - gt) ** 2).sum(1) <= ((native - gt) ** 2).sum(1) + 1e-12).all()


def independent_iou(box):
    edges = np.stack((box[:, :3] - box[:, 3:] * .5, box[:, :3] + box[:, 3:] * .5), 1)
    truth = np.stack((gt[:, :3] - gt[:, 3:] * .5, gt[:, :3] + gt[:, 3:] * .5), 1)
    overlap = np.minimum(edges[:, 1], truth[:, 1]) - np.maximum(edges[:, 0], truth[:, 0])
    intersection = np.prod(np.clip(overlap, 0, None), axis=1)
    return intersection / (np.prod(box[:, 3:], axis=1) + np.prod(gt[:, 3:], axis=1) - intersection)


baseline_iou = independent_iou(mask)
result_iou = independent_iou(diagnostic)
calc = dict(rows=9508, zero_length_axis_spans=zero_axes,
            both_boxes_identical_rows=int((native == mask).all(1).sum()),
            coefficient_zero_axes=int((weights == 0).sum()),
            coefficient_one_axes=int((weights == 1).sum()),
            interior_coefficient_axes=int(((weights > 0) & (weights < 1)).sum()),
            mask_baseline_hits=[int((baseline_iou > t).sum()) for t in (.25, .5)],
            diagnostic_gt_conditioned_hits=[int((result_iou > t).sum()) for t in (.25, .5)],
            comparisons={str(t): dict(repairs=int(((baseline_iou <= t) & (result_iou > t)).sum()),
                                     damages=int(((baseline_iou > t) & (result_iou <= t)).sum()),
                                     net=int((result_iou > t).sum() - (baseline_iou > t).sum())) for t in (.25, .5)})
assert all(saved[key] == value for key, value in calc.items())
invalid = np.array([not r['reference_valid'] for r in rows])
assert invalid.sum() == 39 and np.array_equal(invalid, (native == mask).all(1))
assert np.array_equal(weights[invalid], np.zeros((39, 3)))
for entry in summary['table']:
    box = np.array([r['boxes'][entry['condition']] for r in rows], dtype=np.float64)
    measured = independent_iou(box)
    assert [int((measured > t).sum()) for t in (.25, .5)] == entry['float64_hits']
    np.testing.assert_allclose(measured, [r['cpu_float64_iou'][entry['condition']] for r in rows], rtol=0, atol=1e-14)

# Run the exact submitted script with only its output directory rebound inside
# this review folder. It retains the original __file__ and source-script hash.
replay_dir = REVIEW / 'submitted_replay'
replay_dir.mkdir(exist_ok=True)
scope = runpy.run_path(str(ROOT / 'analyze_axis_span_diagnostic.py'))
scope['main'].__globals__['ROOT'] = replay_dir
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    scope['main']()
(REVIEW / 'submitted_replay.stdout.txt').write_text(stdout.getvalue(), encoding='utf-8')
replay = json.loads((replay_dir / 'AXIS_SPAN_DIAGNOSTIC.json').read_bytes())
assert saved.keys() == replay.keys()
assert {k: v for k, v in saved.items() if k != 'time_cst'} == {k: v for k, v in replay.items() if k != 'time_cst'}

# STATIC_ONLY AST checks: no Torch import or model instantiation.
mixer_ast = asts[str(ROOT / 'extremal_span_mixer.py')]
linear_shapes = [tuple(ast.literal_eval(a) for a in n.args) for n in ast.walk(mixer_ast)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'nn' and n.func.attr == 'Linear']
parameters = sum((a + 1) * b for a, b in linear_shapes)
assert parameters == 29793 and len(linear_shapes) * 2 == 14
mixer_cls = next(n for n in mixer_ast.body if isinstance(n, ast.ClassDef))
method_names = [n.name for n in mixer_cls.body if isinstance(n, ast.FunctionDef)]
assert 'get_extra_state' not in method_names and 'set_extra_state' not in method_names
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == 'register_buffer' for n in ast.walk(mixer_ast))
wrapper_ast = asts[str(ROOT / 'span_refinement_model.py')]
wrapper_cls = next(n for n in wrapper_ast.body if isinstance(n, ast.ClassDef))
wrapper_forward = next(n for n in wrapper_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
parent_forward = next(n for cls in asts[str(PARENT / 'pv_ground.py')].body if isinstance(cls, ast.ClassDef) and cls.name == 'PVGround'
                      for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
assert [n.arg for n in wrapper_forward.args.args] == ['self'] and wrapper_forward.args.kwarg.arg == 'inputs'
assert [n.arg for n in parent_forward.args.args] == ['self', 'inputs']
signature = inspect.Signature([inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD),
                               inspect.Parameter('inputs', inspect.Parameter.VAR_KEYWORD)])
binding_error = None
try:
    signature.bind(object(), {'points': 'symbolic input only'})
except TypeError as error:
    binding_error = str(error)
assert binding_error == 'too many positional arguments'
neural_sources = [ROOT / n for n in ('extremal_span_mixer.py', 'span_refinement_model.py', 'matched_span_objective.py')]
assert not any('AXIS_SPAN_DIAGNOSTIC' in p.read_text(encoding='utf-8') or 'SELECTED_BOX_ROWS' in p.read_text(encoding='utf-8') for p in neural_sources)
assert (BUNDLE / 'whole_mask_range.py').is_file()
pair_spec = json.loads((SELECTED / 'pair_spec.json').read_bytes())
assert digest(BUNDLE / 'whole_mask_range.py') == pair_spec['runner_files']['whole_mask_range.py']
assert digest(BUNDLE / 'pvground_boundary_box_refiner.py') == pair_spec['runner_files']['pvground_boundary_box_refiner.py']

# Pure NumPy source/member fixture checks index mapping and ties only. This is
# not real-model evidence and is not used for GT diagnostic counts or metrics.
provider = asts[str(BUNDLE / 'whole_mask_range.py')]
definition = next(n for n in provider.body if isinstance(n, ast.FunctionDef) and n.name == 'member_statistics')
provider_scope = {'np': np}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(BUNDLE / 'whole_mask_range.py'), 'exec'), provider_scope)
points = np.array([[0, 0, 0], [2, 1, 1], [0, 2, 1], [1, 3, 2], [3, 1, 0], [4, 2, 3], [6, 5, 5], [6, 5, 5]], dtype=np.float64)
ids = np.array([2, 2, 5, 5, 9, 9, 12, 12])
g = provider_scope['member_statistics'](points, ids)
assert g['native_ids'].tolist() == [2, 5, 9, 12]
assert g['count'].tolist() == [2, 2, 2, 2]
lower = (g['origin'] + g['lower'] * g['span']).astype(np.float32)
upper = (g['origin'] + g['upper'] * g['span']).astype(np.float32)
active_slots = np.array([True, True, True, False])
faces = []
for axis in range(3):
    for positive in (False, True):
        values = upper[:, axis] if positive else lower[:, axis]
        extreme = values[active_slots].max() if positive else values[active_slots].min()
        selected = active_slots & (values == extreme)
        raw_active = points[np.isin(ids, g['native_ids'][active_slots])]
        raw_extreme = raw_active[:, axis].max() if positive else raw_active[:, axis].min()
        expected_ids = np.unique(ids[(points[:, axis] == raw_extreme) & np.isin(ids, g['native_ids'][active_slots])])
        assert np.array_equal(g['native_ids'][selected], expected_ids)
        faces.append(dict(axis=axis, positive=positive, selected_native_ids=expected_ids.tolist()))
assert faces[0]['selected_native_ids'] == [2, 5]
assert faces[4]['selected_native_ids'] == [2, 9]
assert (lower[3] == upper[3]).all()  # nonempty but degenerate differs from absent

torch_lines = (REVIEW / 'torch_v1_10_2_FunctionsManual.cpp').read_text(encoding='utf-8').splitlines()
assert 'Tensor clamp_backward' in torch_lines[562]
assert 'self >= *min' in torch_lines[566] and 'self <= *max' in torch_lines[566]
assert 'torch' not in sys.modules
assert all(digest(Path(path)) == info['sha256'] for path, info in hashes.items())
receipt = dict(status='PASS_FOR_CACHED_DIAGNOSTIC_AND_STATIC_CHECKS_ONLY',
    generated_at=datetime.datetime.now().astimezone().isoformat(), python=sys.executable,
    python_version=sys.version, numpy=np.__version__, input_hashes='INPUT_HASHES.json',
    rows_read=len(rows), scenes=len({r['scan_id'] for r in rows}),
    scene_target_pairs=len({(r['scan_id'], r['target_id']) for r in rows}),
    diagnostic=calc, invalid_selected_references=int(invalid.sum()),
    prior_audit_json_read_completely=True, prior_audit_verdict=prior_audit['verdict'],
    prior_audit_used_as_arithmetic_authority=False,
    independent_kkt_axes_checked=9508 * 3, sse_no_worse_than_both_endpoints=True,
    submitted_script_replay_exact_excluding_timestamp=True,
    original_files_unchanged=True, gt_conditioned=True, deployment_valid=False,
    iou_upper_bound=False, model_metric=False, training_label_dataset=False,
    minimum_diagnostic_size=float(diagnostic[:, 3:].min()),
    iou_strict_decreases=int((result_iou < baseline_iou).sum()),
    ast_files_parsed=len(asts), linear_shapes=linear_shapes,
    trainable_parameters=parameters, trainable_state_tensors=len(linear_shapes) * 2,
    source_mode_state_restoration='MISSING: plain Python attribute, no buffer/extra state or checkpoint serializer',
    wrapper_positional_input_binding='FAIL', wrapper_positional_binding_error=binding_error,
    native_parent_positional_input_supported=True,
    source_provider_fixture=dict(classification='synthetic_static_mapping_fixture_only', faces=faces,
                                 tied_sources_preserved=True, absent_and_degenerate_distinct=True),
    clamp_boundary_source=dict(version='v1.10.2', lines=[563, 577], derivative_at_zero=1,
                               runtime_gradient_witness=False),
    neural_imports=0, neural_forwards=0, gpu_forwards=0, optimizer_updates=0,
    ssh_queries=0, packages_installed=0, actual_M0='NOT_AVAILABLE', launch_approved=False)
write('DIAGNOSTIC_VERIFICATION.json', receipt)
print(json.dumps(receipt, ensure_ascii=False))
