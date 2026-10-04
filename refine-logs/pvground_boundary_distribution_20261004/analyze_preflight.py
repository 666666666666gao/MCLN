"""Summarize actual closed engineering probes; no model or optimizer replay."""
import datetime
import hashlib
import json
import math
from pathlib import Path

local = Path(__file__).resolve().parent
intake = json.loads((local/'complete_preflight/INTAKE.json').read_bytes())
assert intake['status']['status'] == 'complete' and intake['controller_exit'] == 0
assert not intake['controller_alive'] and intake['weights_downloaded'] == 0
for name, item in intake['files'].items():
    raw = (local/'complete_preflight'/name).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
arms = {}
for arm in ('residual','distribution'):
    directory = local/'complete_preflight'/arm
    proof = json.loads((directory/'preflight.json').read_bytes())
    spec = json.loads((directory/'spec.json').read_bytes())
    assert proof['status'] == 'pass' and proof['boundary_mode'] == spec['boundary_mode'] == arm
    assert proof['head_parameters'] == spec['head_parameters'] == (400614 if arm == 'residual' else 456102)
    assert proof['boundary_loss_weight'] == spec['boundary_loss_weight'] == (0.0 if arm == 'residual' else 1/7)
    assert proof['head_only'] and proof['original_g_state_unchanged'] and proof['upstream_running_state_eval']
    assert proof['optimizer_steps'] == 2 and proof['batch_size'] == 8
    assert proof['same_cached_inputs_zero_head_common_floor_exact']
    assert all(value == 0 for value in proof['zero_output_differences'].values())
    assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
    assert proof['weight_files_created'] == 0 and proof['formal_rows'] == 0
    assert len(proof['native_call_witnesses']) == 4
    assert len(proof['steps']) == 2
    second = proof['steps'][1]
    assert len(second['p3_gradients']) == 10 and all(value > 0 for value in second['p3_gradients'].values())
    for step in proof['steps']:
        assert step['direct_geometry_output_gradient'] > 0
        assert not step['semantic_loss_requires_grad'] and not step['mask_outputs_require_grad']
        assert not step['range_observation_requires_grad']
        assert math.isclose(step['boundary_weighted_loss'],step['boundary_loss']*spec['boundary_loss_weight'],rel_tol=1e-6,abs_tol=1e-7)
        if arm == 'distribution':
            assert step['edge_alone_output_gradient'] > 0
            assert step['boundary_targets_finite']
            assert step['boundary_matched_boxes'] == step['matched_queries']
            assert step['boundary_faces'] == step['matched_queries']*6
    arms[arm] = dict(head_parameters=proof['head_parameters'],
        initial_raw_size_below_floor=proof['support_distance_quantiles']['raw_coarse_size_elements_below_floor'],
        native_output_gradient=[step['direct_geometry_output_gradient'] for step in proof['steps']],
        native_bbox_loss=[step['loss_bbox'] for step in proof['steps']],
        native_giou_loss=[step['loss_giou'] for step in proof['steps']],
        boundary_loss=[step['boundary_loss'] for step in proof['steps']],
        final_size_floor_axes=[step['size_floor_axis_count'] for step in proof['steps']],
        peak_allocated_bytes=proof['peak_allocated_bytes'],peak_reserved_bytes=proof['peak_reserved_bytes'],
        serialization_bytes=proof['serialization_bytes'],
        native_full_forwards=4,updates=2,second_step_positive_gradient_tensors=len(second['p3_gradients']))
    if arm == 'distribution':
        arms[arm].update(edge_alone_output_gradient=[step['edge_alone_output_gradient'] for step in proof['steps']],
            matched_faces=[step['boundary_faces'] for step in proof['steps']],
            targets_outside=[step['boundary_target_outside'] for step in proof['steps']])
assert not (local/'PREFLIGHT_ANALYSIS.json').exists()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_ENGINEERING_PASS',
    arms=arms,real_model_updates=4,native_model_forwards=8,formal_evaluation_rows=0,
    raw_dataset_or_model_replayed_by_this_analyzer=False,weights_created=0,weights_downloaded=0,
    accuracy_evidence=False,goal_achieved=False)
(local/'PREFLIGHT_ANALYSIS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
