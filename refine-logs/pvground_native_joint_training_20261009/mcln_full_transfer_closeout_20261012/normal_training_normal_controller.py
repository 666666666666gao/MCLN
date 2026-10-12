"""Run one reviewed full native adaptation; preserve actual terminal state."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
admission = json.loads((root / 'admission.json').read_bytes())
environment = json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
variables = dict(os.environ)
variables.update(environment['env'])
variables['PYTHONPATH'] = admission['source'] + ':' + admission['source'] + '/pointnet2:' + variables['PYTHONPATH']
variables.update(CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                 MASTER_ADDR='127.0.0.1', MASTER_PORT='29678')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
command = [runtime, '-B', '-u', str(root / 'train_native_mcln_portable.py'),
           '--source', admission['source'], '--spec', admission['spec'],
           '--output', str(root / 'training'), '--max-epochs', '8']
started = time.monotonic()
status = dict(status='running', started_cst=datetime.datetime.now().astimezone().isoformat(),
              child_argv=command, source=admission['source'], scope='FULL_NATIVE_MCLN_TRAINING_MAX8_DECLINE2')
with (root / 'training.log').open('wb') as stream:
    child = subprocess.Popen(command, cwd=admission['source'], env=variables,
                             stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    status['child_pid'] = child.pid
    (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    code = child.wait()
(root / 'child.exit').write_text(str(code) + '\n')
status.update(status='complete' if code == 0 else 'failed', exit_code=code,
              completed_cst=datetime.datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started)
if code == 0:
    status['result'] = json.loads((root / 'training' / 'RESULT.json').read_bytes())
(root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
raise SystemExit(code)
