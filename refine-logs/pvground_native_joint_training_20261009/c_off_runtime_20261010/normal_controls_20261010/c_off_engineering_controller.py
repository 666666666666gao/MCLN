"""Two native training updates for C-off; discard this fitted state afterward."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
admission = json.loads((root / 'admission.json').read_bytes())
assert admission['status'] == 'C_OFF_NATIVE_ENGINEERING_ADMITTED'
protocol_path = root.parent / 'NORMAL_NATIVE_RUN_PROTOCOL.json'
protocol = json.loads(protocol_path.read_bytes())
environment = json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == admission['env_spec_sha256']
for name, digest in admission['helpers'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
variables = dict(os.environ)
variables.update(environment['env'])
variables['PYTHONPATH'] = protocol['model_source'] + ':' + variables['PYTHONPATH']
variables.update(CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                 MASTER_ADDR='127.0.0.1', MASTER_PORT='29719')
started = time.monotonic()
record = dict(status='running', started_cst=datetime.datetime.now().astimezone().isoformat(),
              stage='selected_mask_off_native_preflight', formal_accuracy=None,
              normal_training_started=False)
(root / 'engineering_status.json').write_text(json.dumps(record, indent=2) + '\n')
argv = [admission['runtime'], '-B', '-u', str(root / 'native_c_off_preflight.py'),
        '--protocol', str(protocol_path), '--mode', 'extremal_support',
        '--initial_checkpoint', admission['normal_E0_checkpoint'], '--output', str(root / 'witness')]
with (root / 'preflight.log').open('wb') as stream:
    process = subprocess.run(argv, cwd=protocol['model_source'], env=variables,
                             stdout=stream, stderr=subprocess.STDOUT)
(root / 'preflight.exit').write_text(str(process.returncode) + '\n')
assert process.returncode == 0
receipt = json.loads((root / 'witness' / 'NATIVE_M0_RECEIPT.json').read_bytes())
assert receipt['full_recovery_exact'] and receipt['actual_updates'] == 2
assert receipt['initial_model_state_exactly_matches_normal_E0']
assert receipt['selected_mask_supervision_disabled']
record.update(status='complete', completed_cst=datetime.datetime.now().astimezone().isoformat(),
              elapsed_seconds=time.monotonic() - started, full_recovery_exact=True,
              initial_model_state_exactly_matches_normal_E0=True)
(root / 'engineering_status.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
