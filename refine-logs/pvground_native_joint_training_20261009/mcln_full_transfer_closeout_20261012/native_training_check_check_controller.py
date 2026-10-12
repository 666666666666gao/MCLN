"""One real MCLN training check; preserve its actual child exit."""
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
variables.update(CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
command = [runtime, '-B', '-u', str(root / 'check_native_mcln_portable_training.py'),
           '--source', admission['source'], '--spec', admission['spec'], '--output', str(root / 'check')]
started = time.monotonic()
status = dict(status='running', started_cst=datetime.datetime.now().astimezone().isoformat(),
              child_argv=command, source=admission['source'], scope='TWO_NATIVE_TRAINING_UPDATES_ONLY')
with (root / 'check.log').open('wb') as stream:
    process = subprocess.Popen(command, cwd=admission['source'], env=variables, stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT)
    status['child_pid'] = process.pid
    (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    code = process.wait()
status.update(status='complete' if code == 0 else 'failed', exit_code=code,
              completed_cst=datetime.datetime.now().astimezone().isoformat(),
              elapsed_seconds=time.monotonic()-started)
if code == 0:
    status['result'] = json.loads((root / 'check' / 'RESULT.json').read_bytes())
(root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
raise SystemExit(code)
