"""Verify the actual two-step receipts; report engineering evidence only."""
import datetime
import hashlib
import json
import math
from pathlib import Path


local = Path(__file__).parent
output = local / 'analysis'
assert not output.exists()
intake = json.loads((local / 'complete/INTAKE.json').read_bytes())
assert intake['status']['status'] == 'complete' and intake['controller_exit'] == 0
assert not intake['controller_alive'] and intake['weight_files_created'] == 0
records = {}
for arm in ('local_range', 'whole_range'):
    path = local / 'complete' / arm / 'preflight.json'
    report = json.loads(path.read_bytes())
    assert report['status'] == 'pass' and report['batch_size'] == 8 and report['optimizer_steps'] == 2
    assert report['head_parameters'] == 400614 and report['formal_rows'] == 0
    assert report['weight_files_created'] == 0 and report['g_strict_restore']
    assert report['member_mask_mapping_exact'] and report['tail_after_native_masks']
    assert report['native_call_order_verified'] and len(report['native_call_witnesses']) == 4
    assert all(item['order_matches_native'] for item in report['native_call_witnesses'])
    assert report['same_cached_inputs_zero_head_pair_exact']
    assert all(value == 0 for value in report['zero_output_differences'].values())
    assert report['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
    assert report['optimizer_exact_check']['moment_and_step_states'] == 806
    assert report['optimizer_exact_check']['param_groups'] == 3
    assert len(report['steps']) == 2 and report['steps'][0]['rows'] == report['steps'][1]['rows']
    for step in report['steps']:
        assert math.isfinite(step['loss']) and math.isfinite(step['grad_norm'])
        assert step['direct_geometry_output_gradient'] > 0 and step['direct_semantic_to_p3_gradients_zero']
        assert len(step['native_mask_loss_to_refiner_gradients']) == 8
        assert all(value == 0 for value in step['native_mask_loss_to_refiner_gradients'].values())
    first, second = report['steps']
    assert first['p3_gradients']['output.weight'] > 0
    assert all(value == 0 for name, value in first['p3_gradients'].items() if not name.startswith('output.'))
    assert all(value > 0 for value in second['p3_gradients'].values())
    route = second['whole_range_loss_route']
    assert route['evidence_shape'] == [8, 256, 109]
    if arm == 'whole_range':
        assert report['use_whole_range'] and route['range_gradient'] > 0
        assert all(value > 0 for value in route['global_only_mask_gradients'].values())
    else:
        assert not report['use_whole_range'] and route['range_gradient'] == 0
        assert all(value == 0 for value in route['global_only_mask_gradients'].values())
    records[arm] = dict(receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        batch_size=8, optimizer_steps=2, head_parameters=400614,
        allocated_bytes=report['peak_allocated_bytes'], reserved_bytes=report['peak_reserved_bytes'],
        runner_wall_seconds_through_restore=report['runner_wall_seconds_through_restore'],
        rows=second['rows'], direct_geometry_output_gradient=second['direct_geometry_output_gradient'],
        step2_head_gradients=second['p3_gradients'], step2_whole_range_loss_route=route,
        optimizer_exact_check=report['optimizer_exact_check'])
assert records['local_range']['rows'] == records['whole_range']['rows']
output.mkdir()
limits = ['Two repeated updates on one real augmented batch per arm, not full training or validation.',
    'Zero-head information on/off replay shares one captured upstream input within each process; cross-process bitwise equality is not proved.',
    'BBox+GIoU VJP is an isolated diagnostic; actual optimizer updates use the full native weighted loss plus G.',
    'Allocator peak is for this batch/setup through in-memory restore, not a worst-case-all-scenes capacity or nvidia-smi total.',
    'All-state and AdamW moment/group/step restoration is in BytesIO; no disk checkpoint or resumed-update/RNG equivalence.',
    'No localization/segmentation accuracy, teacher knowledge, quality supervision, or Nr3D/Sr3D result.',
    'The 109-dimensional range summary is not an exact boundary; sizes still use the existing additive output.']
summary = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), engineering_status='PASS',
    finished_cst=intake['status']['finished_cst'], arms=records, native_imports_and_sources=intake['files'],
    real_optimizer_updates_total=4, formal_fit_updates=0, formal_validation_rows=0,
    weight_files_created=0, model_promoted=False, retained_metric_best='original_g_5615_4495',
    interpretation_limits=limits)
(output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
lines = ['# Actual whole-Mask range engineering preflights', '',
    'Controller completed at ' + summary['finished_cst'] + '. Both arms passed, native exit 0.', '',
    '| Arm | Batch / steps | Head parameters | Allocated GB | Reserved GB | Runner wall s |',
    '|---|---:|---:|---:|---:|---:|']
for arm, value in records.items():
    lines.append('| {} | 8 / 2 | 400614 | {:.3f} | {:.3f} | {:.2f} |'.format(arm,
        value['allocated_bytes'] / 1e9, value['reserved_bytes'] / 1e9, value['runner_wall_seconds_through_restore']))
route = records['whole_range']['step2_whole_range_loss_route']
lines += ['', 'Whole-range step-2 BBox+GIoU evidence gradient: ' + str(route['range_gradient']) + '.',
    'Isolated range-path gradients to actual Text / Query Mask / alpha: ' + json.dumps(route['global_only_mask_gradients']) + '.',
    'Control isolated range-path gradients are zero. All eight native Mask losses and final semantic/G loss have no direct gradient to the refiner.',
    'First-step internal head gradients are zero with the zero output layer; all ten head parameter tensors receive nonzero gradients on step 2.',
    'Each process has exact zero-output/cache replay and native call order; strict in-memory model restore and 806 AdamW states / 3 groups verified.', '',
    'Interpretation limits:', ''] + ['- ' + value for value in limits]
(output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(dict(engineering_status='PASS', arms=2, real_preflight_updates=4,
    new_formal_accuracy_available=False, weights_created=0)), flush=True)
