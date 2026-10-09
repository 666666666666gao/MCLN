"""Adopt the one proved original controller, never relaunch the screen."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
assert not (root / 'NATIVE_PREFLIGHT_LAUNCH.json').exists()
actual = json.loads((root / 'INTERRUPTED_LAUNCH_READBACK.json').read_bytes())
assert actual['new_launches'] == 0 and len(actual['controller_pids']) == 1
assert 'controller.exit' not in actual['files']
status = json.loads(actual['files']['preflight_status.json'])
assert status['status'] == 'running' and status['completed_modes'] == []
assert actual['root'] == '/root/autodl-tmp/pvground_native_joint_training_20261009/preflight_attempt2'
started = datetime.datetime.fromisoformat(actual['admission']['time_cst'])
record = dict(status='NATIVE_ENGINEERING_PREFLIGHT_STARTED_NOT_COMPLETED',
    time_cst=started.isoformat(), root=actual['root'], admission=actual['admission'],
    controller_pid=actual['controller_pids'][0], controller_argv=actual['controller_argv'],
    screen='pvg_native_joint_preflight_attempt2_20261010', attempt=2,
    estimated_seconds=1800, first_check_seconds=1500, later_poll_seconds=240,
    first_observation_cst=(started + datetime.timedelta(seconds=1500)).isoformat(),
    original_launch_script_exit_code=1, original_immediate_pgrep_failed=True,
    original_already_running_controller_adopted=True, new_launches_on_receipt_recovery=0,
    launch_readback_sha256=hashlib.sha256((root / 'INTERRUPTED_LAUNCH_READBACK.json').read_bytes()).hexdigest(),
    previous_failed_run_preserved=True, computational_model_changes=False,
    formal_accuracy=None, normal_epoch_training_started=False)
(root / 'NATIVE_PREFLIGHT_LAUNCH.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
