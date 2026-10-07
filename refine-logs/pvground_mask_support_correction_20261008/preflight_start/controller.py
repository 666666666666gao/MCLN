"""One locked parent per phase; one frozen forward and two independent Mask heads."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time


root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['preflight', 'fit'], required=True)
args = parser.parse_args()
phase = args.phase
assert not (root / (phase + '_status.json')).exists()
spec = json.loads((root / 'pair_spec.json').read_bytes())
environment = json.loads((Path(spec['runtime']) / 'env_spec.json').read_bytes())
variables = dict(environment['env'])
variables['PYTHONPATH'] = str(root) + ':' + spec['helper_root'] + ':' + variables['PYTHONPATH']
parents = {Path(spec[key]): spec[key + '_sha256'] for key in ('base_terminal', 'selected_terminal')}
official = environment['weight_dirs']['scanrefer']
parents[Path(official['path'])] = official['sha256']
assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in parents.items())
if phase == 'fit':
    assert json.loads((root / 'preflight_status.json').read_bytes())['status'] == 'complete'
    preflight = json.loads((root / 'preflight.json').read_bytes())
    assert preflight['status'] == 'pass' and preflight['optimizer_steps_per_arm'] == 2
    assert preflight['weight_files_created'] == 0 and preflight['separate_optimizers_and_gradients']
    assert all(check['actual_native_gpu_integration'] for check in preflight['restore'].values())
started = time.monotonic()
record = dict(status='running', phase=phase, completed_modes=[],
    started_cst=datetime.datetime.now().astimezone().isoformat(),
    protected_best_hits=[5598, 4848], parent_and_box_head_frozen=True)


def write_status():
    (root / (phase + '_status.json')).write_text(json.dumps(record, indent=2) + '\n')


write_status()
for mode in (('preflight',) if phase == 'preflight' else ('initial_formal', 'train', 'formal')):
    command = ['env'] + [key + '=' + value for key, value in variables.items()] + [
        spec['runtime'] + '/venv/bin/python', '-B', '-u', str(root / 'run_mask_support_pair.py'),
        '--spec', str(root / 'pair_spec.json'), '--mode', mode]
    with (root / (mode + '.log')).open('x') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        record.update(mode=mode, child_pid=child.pid)
        write_status()
        code = child.wait()
    (root / (mode + '.exit')).write_text(str(code) + '\n')
    if code:
        record.update(status='failed', exit_code=code, finished_cst=datetime.datetime.now().astimezone().isoformat())
        write_status()
        raise SystemExit(code)
    record['completed_modes'].append(mode)
    write_status()
assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in parents.items())
record.update(status='complete', exit_code=0, finished_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.monotonic() - started, protected_parents_exact=True,
    optimizer_steps_per_arm=2 if phase == 'preflight' else 3723, accuracy_result=phase == 'fit')
write_status()
print(json.dumps(record), flush=True)
