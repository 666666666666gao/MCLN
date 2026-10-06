"""Keep completed M0 evidence separate from live full-fit status."""
import datetime
import json
from pathlib import Path

root=Path(__file__).resolve().parent
wait=json.loads((root/'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['controller_alive'] and wait['exitcode']==0
proofs={arm:json.loads((root/'preflight_complete'/arm/'preflight.json').read_bytes())
    for arm in ('native_reference','fused_mask_reference')}
for arm,proof in proofs.items():
    assert proof['status']=='pass' and proof['optimizer_steps']==2 and proof['weight_files_created']==0
    assert proof['head_parameters']==456102 and proof['head_state_tensors']==10
    assert proof['all_parent_and_R_states_exact'] and proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
    assert proof['invalid_reference_fixture']['actual_empty_support_rows_verified']==39
    assert proof['witnesses'][0]['neutral_initial_decode_equals_reference']
    assert all(value>0 for name,value in proof['witnesses'][1]['raw_parameter_gradient_norms'].items())
    assert all(value['actual_all256_raw_member_extent_verified'] and value['reference_bound_max_error']==0
               for value in proof['witnesses'])
launch=json.loads((root/'fit_launch.json').read_bytes())
assert launch['status']=='TWO_ARM_FIT_LAUNCHED_NOT_COMPLETED' and launch['process'].startswith('711378 ')
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_TWO_ARM_M0_ACCEPTED_FULL_FIT_LAUNCHED',
    preflight_controller_closed=True,preflight_observer_native_session_id=44835,preflight_observer_closed=True,
    preflight_actual_finished_cst=wait['status']['finished_cst'],preflight_optimizer_steps_per_arm=2,
    preflight_serialization_bytes=proofs['native_reference']['serialization_bytes'],
    preflight_peak_allocated_bytes_by_arm={arm:value['peak_allocated_bytes'] for arm,value in proofs.items()},
    actual_fit_launch=launch,fit_controller_pid=711378,sole_fit_observer_native_session_id=27853,
    observer_closed=False,first_fit_observation_cst='2026-10-06T17:14:14+08:00',
    projected_fit_completion_cst='2026-10-06T17:20:54+08:00',poll_seconds=240,
    trained_best_hits=[5616,4511],new_accuracy_result=False,
    actual_witness_limit='8 fixed fit inputs repeated2 times per arm; cached-head-update scores/Masks exact, not cross-forward bitwise identity.',
    preflight_reference_warning='Mask arm matchedDFL outside2/48 faces, versus native0/48; extra faces outside196/768 vs120/858 on first batch. One batch, not formal accuracy or general failure cause.')
(root/'M0_ACCEPTANCE_AND_FIT_LAUNCH.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],status='MASK_REFERENCE_FULL_PAIR_ACTIVE',owned_gpu_job_active=True,
    active_reviewer=None,reference_fit_launch=str(root/'fit_launch.json'),
    reference_fit_observer_native_session_id=27853,reference_preflight_observer_closed=True,
    reference_m0_acceptance=str(root/'M0_ACCEPTANCE_AND_FIT_LAUNCH.json'),
    first_fit_observation_cst=record['first_fit_observation_cst'],
    next_action='Do not restart controller711378 or soleobserver27853. First check17:14:14,240s thereafter; closure autocollects no weights. Fresh terminal audit and actual selected-state restore before best-only cleanup.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': MaskRef M0 both actual2steps PASS, parent/R unchanged,all256 raw extent0error,39 empty prior rule,step2 all10grads,CPU optimizer exact. Closed11:08:20/44835closed0,weightfiles0. Fullpair launched11:14:14 controller711378,soleobserver27853 first17:14:14,poll240,totalestimate22000s. Geometricbest4511 unchanged, no newaccuracy. Source and runtime limits preserved; do not restart jobs/observer.\n')
print(json.dumps(dict(status=record['status'],first_check=record['first_fit_observation_cst'],
    projected_finish=record['projected_fit_completion_cst'],new_accuracy_result=False)))
