"""Bind complete actual E0 and the sole near-E1 observer; no remote calls."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path('\\\\?\\' + str(Path(__file__).resolve().parent))
name = 'OBS_lower_span_lr_e0_completion'
intake_path = root / (name + '_INTAKE.json')
intake = json.loads(intake_path.read_bytes())
assert intake['controller_alive'] and intake['child_alive']
assert intake['remote_status']['child_pid'] == 432871
assert intake['remote_status']['status'] == 'running'
assert intake['observed_cst'] >= intake['first_due_cst']
files_root = root / (name + '_files')
bindings = {}
for entry in intake['files']:
    raw = (files_root / entry['name']).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert len(raw) == entry['bytes'] and digest == entry['sha256']
    bindings[entry['name']] = {'bytes': len(raw), 'sha256': digest}
assert len(bindings) == 5
metrics_path, = files_root.rglob('native_metrics.jsonl')
metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
assert metrics == [{'epoch': 0, 'rows': 9508, 'hits025': 5677, 'hits050': 4920,
    'primary_score': 'last/bbs', 'same_complete_model': True}]
diagnostic_path, = files_root.rglob('native_diagnostic_epoch000.json')
diagnostic = json.loads(diagnostic_path.read_bytes())
assert diagnostic['rows'] == 9508 and diagnostic['epoch'] == 0
assert diagnostic['025']['hits'] == 5677 and diagnostic['050']['hits'] == 4920
assert diagnostic['025']['full256_covered'] == 8970
assert diagnostic['050']['full256_covered'] == 8234
gate = diagnostic['span_raw_gate_observation']
assert gate['all_query_axes'] == 9508 * 256 * 3
assert gate['raw_below_zero_by_axis'] == [198203, 211220, 167258]
assert gate['raw_above_one_by_axis'] == [0, 0, 0]
assert gate['selected_raw_outside_range_axes'] == 6
tail_path = files_root / 'train.log.tail.txt'
tail = tail_path.read_bytes()
assert b'35/4583 [02:31<' in tail
owner_path = root / 'OBS_lower_span_lr_e1_near_complete_OWNER.json'
owner = json.loads(owner_path.read_bytes())
assert owner['local_pid'] == 26676
assert owner['due_cst'] == '2026-10-14T08:15:00+08:00'
startup_path = root / 'LOWER_SPAN_LR_NORMAL_LIVE_RECEIPT_20261014.json'
current = json.loads(startup_path.read_bytes())
current.update(status='LOWER_B_LR_NORMAL_E0_VERIFIED_FIRST_TRAINING_EPOCH_RUNNING',
    recorded_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_process_witness_cst=intake['observed_cst'],
    current_arm_E0_complete_metrics=metrics[0],
    current_arm_E0_accuracy_percent=[100 * 5677 / 9508, 100 * 4920 / 9508],
    complete_formal_hits_match_historical_E0=True,
    initial_output_bitwise_identity_established=False,
    current_arm_trained_formal_metrics=None,
    current_training_epoch=1, observed_training_batches=35,
    observed_training_elapsed_seconds=151,
    batch_counter_is_not_cold_optimizer_state_audit=True,
    complete_e0_diagnostic=diagnostic,
    selected_raw_outside_range_fraction=6 / (9508 * 3),
    raw_gate_counts_do_not_prove_clamp_caused_training_failure=True,
    current_observation_intake_sha256=hashlib.sha256(intake_path.read_bytes()).hexdigest(),
    current_observation_files=bindings,
    current_observation_tail_sha256=hashlib.sha256(tail).hexdigest(),
    consumed_E0_observer_native_session=12052, consumed_E0_observer_exit_code=0,
    consumed_E0_observer_functions_cell='451',
    observer_pid=26676, observer_native_session_id=11018,
    next_observation_cst=owner['due_cst'],
    next_observation_estimate='E1fit+9508formal expected08:00-08:30; one read08:15',
    next_observation_basis='Closed same-normal warm E2 ~3.53sec/update plus~34min formal; new first35 batches include warmup',
    observer_owner_sha256=hashlib.sha256(owner_path.read_bytes()).hexdigest(),
    startup_receipt_sha256=hashlib.sha256(startup_path.read_bytes()).hexdigest(),
    best_promoted=False, current_arm_independent_gain_established=False,
    current_arm_neural_restarts=0, full_goal_complete=False)
receipt = root / 'LOWER_SPAN_LR_E0_COMPLETE_RECEIPT_20261014.json'
assert not receipt.exists()
receipt.write_bytes((json.dumps(current, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
states = [root / 'CURRENT_PROGRESS.json', root.parent / 'NORMAL_CONTINUATION_STATE.json',
    root.parent / 'face_native_gpu_v2_20261011' / 'CURRENT_PROGRESS.json',
    Path('\\\\?\\C:\\Users\\gb\\.codex\\tmp\\mcln_portable_best_transfer_20261012\\CURRENT_PROGRESS.json'),
    Path('\\\\?\\C:\\Users\\gb\\.codex\\tmp\\pvground_referit_mask_reference_20261006\\current_research_goals.json'),
    Path('\\\\?\\C:\\Users\\gb\\.codex\\tmp\\pvground_query_supported_geometry_20261005\\active_continuation_state.json')]
for path in states:
    state = json.loads(path.read_bytes())
    state.update(current_normal_arm='lower_span_lr', current_summary_status=current['status'],
        current_normal_controller_pid=432868, current_normal_child_pid=432871,
        current_normal_training_started=True, current_normal_E0_complete_metrics=metrics[0],
        current_normal_trained_complete_metrics=None,
        current_low_span_lr_normal_record=current,
        current_summary_arm='lower_span_lr', current_summary_observed_cst=intake['observed_cst'],
        current_low_span_lr_observation_receipt=str(receipt),
        current_normal_observer_pid=26676, current_normal_observer_native_session=11018,
        next_observation_cst=owner['due_cst'], full_goal_complete=False)
    path.write_bytes((json.dumps(state, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps({'status': current['status'], 'captured_files_verified': len(bindings),
    'E0': metrics[0], 'gate': gate, 'next_observation_cst': owner['due_cst'],
    'continuation_files_updated': len(states), 'new_neural_jobs': 0}))
