"""Read the actual closed frozen-G probe receipts; do not replay the model."""
import datetime
import hashlib
import json
from pathlib import Path

local=Path(__file__).parent;packet=local/'complete_preflight'
assert not (local/'preflight_analysis.json').exists()
intake=json.loads((packet/'INTAKE.json').read_bytes())
spec=json.loads((local/'preflight_spec.json').read_bytes())
proof=json.loads((packet/'whole_range/preflight.json').read_bytes())
assert intake['status']['status']=='complete' and intake['controller_exit']==0 and not intake['controller_alive']
assert intake['original_g_sha256']==spec['base_terminal_sha256']
for relative,entry in intake['files'].items():
    raw=(packet/relative).read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
assert proof['status']=='pass' and proof['head_only'] and proof['batch_size']==8 and proof['optimizer_steps']==2
assert proof['original_g_state_unchanged'] and proof['upstream_parameters_frozen'] and proof['upstream_running_state_eval']
assert proof['g_strict_restore'] and proof['optimizer_restore']
assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
assert proof['optimizer_exact_check']['moment_and_step_states']==10
assert proof['same_cached_inputs_zero_head_pair_exact'] and proof['member_mask_mapping_exact']
assert all(value==0 for value in proof['zero_output_differences'].values())
assert proof['native_call_order_verified'] and len(proof['native_call_witnesses'])==4
assert proof['weight_files_created']==proof['formal_rows']==0
assert len(proof['steps'])==2
for step in proof['steps']:
    assert step['direct_geometry_output_gradient']>0
    assert not step['semantic_loss_requires_grad'] and not step['mask_outputs_require_grad']
    assert not step['range_observation_requires_grad']
assert len(proof['steps'][1]['p3_gradients'])==10 and all(value>0 for value in proof['steps'][1]['p3_gradients'].values())
summary=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),engineering_status='PASS',
    actual_two_step_frozen_G_check=True,original_g_state_unchanged=True,head_parameter_tensors=10,head_parameters=400614,
    batch_size=8,optimizer_steps=2,native_full_forwards=4,second_step_positive_head_gradient_tensors=10,
    output_geometry_gradient_per_step=[step['direct_geometry_output_gradient'] for step in proof['steps']],
    native_bbox_loss_per_step=[step['loss_bbox'] for step in proof['steps']],
    native_giou_loss_per_step=[step['loss_giou'] for step in proof['steps']],
    peak_allocated_bytes=proof['peak_allocated_bytes'],peak_reserved_bytes=proof['peak_reserved_bytes'],
    memory_peak_scope=proof['memory_peak_scope'],serialization_bytes=proof['serialization_bytes'],
    runner_wall_seconds=proof['runner_wall_seconds_through_restore'],controller_seconds=intake['status']['seconds'],
    zero_head_exact=True,head_and_AdamW_restore_exact=True,
    geometry_to_Mask_route='intentionally absent in this frozen protocol',
    weights_created=0,weights_downloaded=0,formal_training_started=False,formal_accuracy_result_available=False,
    original_g_retained=True,goal_achieved=False,
    source_identity_checked_files=len(intake['files']),raw_inputs_or_model_replayed_by_analyzer=False)
(local/'preflight_analysis.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary),flush=True)
