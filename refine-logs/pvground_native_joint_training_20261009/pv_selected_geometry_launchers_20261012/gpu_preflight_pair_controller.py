"""Under the existing GPU lock, run two fresh native engineering arms serially."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path(__file__).resolve().parent
base = root.parent
admission = json.loads((root / 'admission.json').read_bytes())
assert admission['status'] == 'SELECTED_GEOMETRY_PREFLIGHT_ADMITTED'
environment = json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == admission['env_spec_sha256']
mcln_root = Path('/root/autodl-tmp/mcln_portable_best_transfer_20261012/normal_training')
terminal = json.loads((mcln_root / 'status.json').read_bytes())
assert terminal == admission['MCLN_terminal_status']
assert terminal['status'] == 'complete' and terminal['exit_code'] == 0
assert not Path('/proc/271902').exists() and not Path('/proc/271903').exists()
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(gpu) == 1
index, name, used, total = [part.strip() for part in gpu[0].split(',')]
assert index == '0' and 'A100' in name and int(used) < 500 and 40000 <= int(total) <= 45000
assert shutil.disk_usage(base).free > admission['reserve_bytes']
for relative, digest in admission['helper_sha256'].items():
    assert hashlib.sha256((base / relative).read_bytes()).hexdigest() == digest
stage = json.loads((base / 'source_stage_manifest.json').read_bytes())
assert stage == admission['source_stage']
for row in stage['source_files'].values():
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
variables = dict(os.environ)
variables.update(environment['env'])
variables['PYTHONPATH'] = str(base / 'PV-Ground') + ':' + variables['PYTHONPATH']
variables.update(CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                 MASTER_ADDR='127.0.0.1', MASTER_PORT='29680')
status = dict(status='running', stage='paired_native_engineering',
    started_cst=datetime.datetime.now().astimezone().isoformat(),
    formal_accuracy=None, normal_training_started=False, completed_arms=[])
assert not (root / 'status.json').exists()
(root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
receipts = {}
started = time.monotonic()
for arm in ('native_control', 'selected_geometry'):
    arm_root = root / arm
    assert not arm_root.exists()
    arm_root.mkdir()
    protocol_path = base / (arm + '_protocol.json')
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol['arm'] == arm and not protocol['GPU_admitted']
    command = [admission['runtime'], '-B', '-u', str(base / 'native_selected_geometry_preflight.py'),
               '--protocol', str(protocol_path), '--mode', arm, '--output', str(arm_root / 'witness')]
    with (arm_root / 'preflight.log').open('wb') as stream:
        child = subprocess.Popen(command, cwd=str(base / 'PV-Ground'), env=variables,
                                 stdout=stream, stderr=subprocess.STDOUT)
        status.update(current_arm=arm, child_pid=child.pid, child_argv=command)
        (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        exit_code = child.wait()
    (arm_root / 'preflight.exit').write_text(str(exit_code) + '\n')
    if exit_code:
        status.update(status='failed', exit_code=exit_code,
                      child_closed_cst=datetime.datetime.now().astimezone().isoformat())
        (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    assert exit_code == 0
    receipt = json.loads((arm_root / 'witness/NATIVE_M0_RECEIPT.json').read_bytes())
    assert receipt['source_mode'] == arm and receipt['actual_updates'] == 2
    assert receipt['actual_input_rows'] == 16 and receipt['real_training_batches'] == 2
    assert receipt['real_train_dataset_rows'] == 36665 and receipt['real_formal_dataset_rows'] == 9508
    assert receipt['normal_epoch_updates'] == 4583 and receipt['full_state_tensors'] == 1295
    assert receipt['full_recovery_exact'] and receipt['saved_retained_metrics'] is None
    assert receipt['selected_mask_supervision_disabled']
    assert receipt['selected_geometry_supervision'] == (arm == 'selected_geometry')
    assert receipt['formal_accuracy'] is None and not receipt['preflight_weights_used_for_formal_training']
    if arm == 'selected_geometry':
        assert sum(row['extra_geometry']['extra_rows'] for row in receipt['losses']) > 0
        assert sum(row['extra_direct_gradient_l1'] for row in receipt['losses']) > 0
    receipts[arm] = receipt
    status['completed_arms'].append(arm)
    (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
assert receipts['native_control']['input_batch_sha256'] == receipts['selected_geometry']['input_batch_sha256']
status.update(status='complete', exit_code=0,
    completed_cst=datetime.datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started,
    actual_total_native_updates=4, same_process_recovery_verified=True,
    same_augmented_inputs_verified=True, formal_training_admitted=False)
(root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status), flush=True)
