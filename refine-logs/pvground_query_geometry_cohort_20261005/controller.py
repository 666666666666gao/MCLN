"""One read-only GPU probe; no optimizer, checkpoint or install."""
import datetime
import hashlib
import json
from pathlib import Path
import os
import subprocess
import time

root = Path(__file__).parent
spec = json.loads((root/'spec.json').read_bytes())
assert spec['root'] == str(root) and spec['diagnostic_only']
assert not (root/'status.json').exists()
env = json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())
variables = dict(env['env'])
variables['PYTHONPATH'] = str(root)+':'+spec['helper_root']+':'+variables['PYTHONPATH']
parents = {Path(spec[key]): spec[key+'_sha256'] for key in ('base_terminal', 'geometry_terminal')}
official = env['weight_dirs']['scanrefer']
parents[Path(official['path'])] = official['sha256']
assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in parents.items())
record = dict(status='running', started_cst=datetime.datetime.now().astimezone().isoformat(),
              protected_best_hits=[5616,4506], accuracy_result=False, optimizer_steps=0)
start = time.monotonic()
command = ['env']+[key+'='+value for key,value in variables.items()]+[
    spec['runtime']+'/venv/bin/python','-B','-u',str(root/'run_cohort_probe.py'),
    '--spec',str(root/'spec.json')]
with (root/'probe.log').open('x') as stream:
    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    record['process_pid'] = child.pid
    (root/'status.json').write_text(json.dumps(record, indent=2)+'\n')
    code = child.wait()
(root/'probe.exit').write_text(str(code)+'\n')
record.update(status='failed' if code else 'complete', exit_code=code,
              finished_cst=datetime.datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic()-start)
if code == 0:
    receipt = json.loads((root/'receipt.json').read_bytes())
    assert receipt['rows'] == 64 and receipt['batches'] == 8
    assert receipt['optimizer_steps'] == receipt['weight_files_created'] == 0
    assert receipt['model_state_unchanged'] and receipt['model_gradients_absent']
    assert not list(root.glob('*.pth'))
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in parents.items())
    record['protected_parent_hashes_exact'] = True
(root/'status.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
raise SystemExit(code)
