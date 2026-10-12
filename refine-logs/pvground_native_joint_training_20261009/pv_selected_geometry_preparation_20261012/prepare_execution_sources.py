"""Prepare future normal protocols and actual two-step checks, without SSH/GPU."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
base = root.parent
remote = '/root/autodl-tmp/pvground_deployment_geometry_training_20261012'
previous = json.loads((base / 'face_native_gpu_v2_20261011/normal_training/source_conditioned_PROTOCOL.json').read_bytes())
arguments = list(previous['common_arguments'])
assert arguments[arguments.index('--max_epoch') + 1] == '3'
arguments[arguments.index('--max_epoch') + 1] = '8'
assert '--joint_det' not in arguments and '--checkpoint_path' not in arguments
for arm in ('native_control', 'selected_geometry'):
    protocol = dict(status='PREPARED_NOT_DEPLOYED', arm=arm,
        model_source=remote + '/PV-Ground', native_init_spec=remote + '/' + arm + '_initialization.json',
        common_arguments=arguments, max_epochs=8, patience=2, target_hits=[5705, 4945],
        primary_score='last/bbs', formal_rows=9508, seed=2027, batch_size=8,
        effective_batch_size=8, accumulation=1, expected_train_rows=36665,
        expected_updates_per_epoch=4583, expected_rows_per_epoch=36664,
        maximum_updates=36664, maximum_rows_presented=293312,
        sampler_seed=2027, E0_included_in_declines=True,
        equality_or_increase_resets=True, best_order='both_goal_then_hits050_then_hits025',
        old_selected_mask_supervision=False, selected_geometry_supervision=arm == 'selected_geometry',
        normal_training_started=False, actual_optimizer_updates=0,
        original_core_frozen=False, original_roberta_frozen=True,
        cold_resume_supported=False, preflight_states_used=False,
        actual_MCLN_terminal_verified=False, GPU_admitted=False)
    (root / (arm + '_protocol.json')).write_bytes((json.dumps(protocol, indent=2) + '\n').encode('utf-8'))

text = (base / 'native_joint_preflight.py').read_bytes().decode('utf-8').replace('\r\n', '\n')
def replace_once(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)

replace_once("choices=['extremal_support']", "choices=['native_control', 'selected_geometry']")
replace_once("str(source / 'init_manifests' / (options.mode + '.json'))", "protocol['native_init_spec']")
face_preflight = (base / 'face_native_gpu_v2_20261011/native_face_preflight.py').read_bytes().decode('utf-8').replace('\r\n', '\n')
start = face_preflight.index('def input_fingerprint(value):')
end = face_preflight.index('criterion, set_criterion = tester.get_criterion(args)', start)
replace_once('criterion, set_criterion = tester.get_criterion(args)',
             face_preflight[start:end] + 'criterion, set_criterion = tester.get_criterion(args)')
replace_once("assert len(model.state_dict()) == 1295", "assert len(model.state_dict()) == 1295\nassert model.use_g_supervision and not model.use_selected_mask_supervision\nassert model.use_selected_geometry_supervision == (options.mode == 'selected_geometry')")
before = "        selected_query_mask_extra_rows=int(end_points['selected_query_mask_extra_rows']),\n        selected_query_mask_extra_loss=float(end_points['selected_query_mask_extra_loss'].detach())))"
after = "        selected_mask_extra_disabled=True))\n    assert 'selected_query_mask_extra_rows' not in end_points\n    if options.mode == 'selected_geometry':\n        record = end_points['selected_geometry_extra_record']\n        extra = end_points['selected_geometry_extra_loss']\n        direct = torch.autograd.grad(extra, [end_points['last_center'], end_points['last_pred_size']], retain_graph=True)\n        direct_locations = torch.stack([value.abs().sum(-1) for value in direct]).sum(0)\n        expected_locations = torch.zeros((8, 256), dtype=torch.bool, device=direct_locations.device)\n        for bid, row in enumerate(record['rows']):\n            if row['extra_geometry']:\n                expected_locations[bid, row['query']] = True\n        assert not (direct_locations[~expected_locations] != 0).any()\n        loss_records[-1].update(extra_geometry=record, extra_direct_gradient_nonzero_queries=int((direct_locations > 0).sum()), extra_direct_gradient_l1=float(direct_locations.sum()))\n    else:\n        assert 'selected_geometry_extra_record' not in end_points"
replace_once(before, after)
replace_once("args._retained_native_metrics = None  # Engineering-only checkpoint, no fabricated formal score.", "args._retained_native_metrics = None  # Engineering-only checkpoint, no fabricated formal score.\nargs._epoch_metric_history = []")
replace_once("status='ACTUAL_NATIVE_NORMAL_TRAINER_TWO_STEP_AND_FULL_RECOVERY_COMPLETE'", "status='ACTUAL_SELECTED_GEOMETRY_NORMAL_TRAINER_TWO_STEP_AND_SAME_PROCESS_RECOVERY_COMPLETE'")
replace_once("source_mode=options.mode, time_cst=datetime.datetime.now().astimezone().isoformat(),", "source_mode=options.mode, time_cst=datetime.datetime.now().astimezone().isoformat(),\n    selected_geometry_supervision=options.mode == 'selected_geometry', selected_mask_supervision_disabled=True,")
replace_once('real_training_batches=2, native_train_one_epoch_called=True,',
             'real_training_batches=2, native_train_one_epoch_called=True, input_batch_sha256=input_batch_sha256,')
ast.parse(text)
(root / 'native_selected_geometry_preflight.py').write_bytes(text.encode('utf-8'))
controller = root / 'normal_run_controller.py'
ast.parse(controller.read_bytes().decode('utf-8'))
result = dict(status='NORMAL_PROTOCOLS_AND_ACTUAL_PREFLIGHT_SOURCES_PREPARED_NOT_EXECUTED',
    source_files=[controller.name, 'native_selected_geometry_preflight.py'],
    common_normal_entry='PV-Ground/train_dist_mod.py', base_source_manifest=str(base / 'NATIVE_SOURCE_PORT.json'),
    additional_local_overlay='source_overlay_v2', GPU_calls=0, remote_calls=0, neural_calls=0,
    maximum_epochs=8, maximum_updates_per_arm=36664, patience=2,
    actual_model_constructed=False, actual_loss_hook_executed=False,
    normal_training_launched=False, launcher_prepared=False,
    actual_preflight_completed=False, same_process_recovery_verified=False,
    cold_resume_verified=False, formal_accuracy=None,
    additional_review_required=True,
    files={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
           for name in ('normal_run_controller.py', 'native_selected_geometry_preflight.py',
                        'native_control_protocol.json', 'selected_geometry_protocol.json')})
(root / 'EXECUTION_SOURCE_PREPARATION.json').write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
print(json.dumps({key: result[key] for key in ('status', 'GPU_calls', 'neural_calls', 'normal_training_launched')}))
