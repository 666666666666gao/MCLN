"""Serial actual native engineering checks after explicit single-GPU admission."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
admission = json.loads((root / 'admission.json').read_bytes())
assert admission['status'] == 'NATIVE_GPU_PREFLIGHT_ADMITTED_AFTER_ORIGINAL_SPAN_CLOSED'
assert admission['prior_span_controller_exit_code'] == 0
assert admission['original_span_terminal_verified']
protocol = json.loads((root / 'NORMAL_NATIVE_RUN_PROTOCOL.json').read_bytes())
assert protocol['normal_joint_training_started'] is False
environment = json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
environment_sha = hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert environment_sha == admission['env_spec_sha256']
for name, digest in admission['helper_files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
variables = dict(environment['env'])
variables['PYTHONPATH'] = protocol['model_source'] + ':' + variables['PYTHONPATH']
variables.update(CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0', MASTER_ADDR='127.0.0.1')
assert not (root / 'preflight_status.json').exists()
record = dict(status='running', phase='native_engineering_preflight', completed_modes=[],
    started_cst=datetime.datetime.now().astimezone().isoformat(),
    formal_accuracy=None, normal_epoch_training_started=False)
started = time.monotonic()
for index, mode in enumerate(('whole_support', 'extremal_support')):
    variables['MASTER_PORT'] = str(29687 + index)
    record['mode'] = mode
    (root / 'preflight_status.json').write_text(json.dumps(record, indent=2) + '\n')
    command = ['env'] + [key + '=' + value for key, value in variables.items()] + [
        '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python', '-B', '-u',
        str(root / 'native_joint_preflight.py'), '--protocol', str(root / 'NORMAL_NATIVE_RUN_PROTOCOL.json'),
        '--mode', mode, '--output', str(root / mode)]
    with (root / (mode + '.log')).open('wb') as stream:
        process = subprocess.run(command, cwd=protocol['model_source'], stdout=stream, stderr=subprocess.STDOUT)
    (root / (mode + '.exit')).write_text(str(process.returncode) + '\n')
    assert process.returncode == 0
    receipt = json.loads((root / mode / 'NATIVE_M0_RECEIPT.json').read_bytes())
    assert receipt['actual_updates'] == 2 and receipt['full_recovery_exact']
    assert receipt['original_trainable_core_jointly_updated'] and receipt['formal_accuracy'] is None
    record['completed_modes'].append(mode)
record.update(status='complete', mode='complete', elapsed_seconds=time.monotonic()-started,
    completed_cst=datetime.datetime.now().astimezone().isoformat())
(root / 'preflight_status.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
