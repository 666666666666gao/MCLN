"""Run the two reviewed engineering preflights once, with no disk weights."""
import datetime
import json
from pathlib import Path
import subprocess
import time


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


root = Path(__file__).parent
assert not (root / 'status.json').exists()
spec = json.loads((root / 'whole_range/spec.json').read_bytes())
environment = json.loads((Path(spec['runtime']) / 'env_spec.json').read_bytes())
python = spec['runtime'] + '/venv/bin/python'
completed = []
for arm in ('local_range', 'whole_range'):
    directory = root / arm
    (root / 'status.json').write_text(json.dumps(dict(status='running', phase=arm,
        started_cst=now(), completed=completed), indent=2) + '\n')
    variables = dict(environment['env'])
    variables['PYTHONPATH'] = str(directory) + ':' + variables['PYTHONPATH']
    command = ['env'] + [key + '=' + value for key, value in variables.items()] + [
        python, '-B', '-u', str(root / 'run_whole_mask_preflight.py'),
        '--spec', str(directory / 'spec.json'), '--mode', 'preflight']
    begin = time.monotonic()
    with (directory / 'preflight.log').open('x') as output:
        code = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
    (directory / 'preflight.exit').write_text(str(code) + '\n')
    if code != 0:
        (root / 'status.json').write_text(json.dumps(dict(status='failed', phase=arm,
            exit_code=code, finished_cst=now(), completed=completed), indent=2) + '\n')
        raise SystemExit(code)
    receipt = json.loads((directory / 'preflight.json').read_bytes())
    assert receipt['status'] == 'pass' and receipt['batch_size'] == 8 and receipt['optimizer_steps'] == 2
    completed.append(dict(arm=arm, exit_code=code, seconds=time.monotonic() - begin,
        finished_cst=now()))
(root / 'status.json').write_text(json.dumps(dict(status='complete', completed=completed,
    finished_cst=now(), formal_training_started=False, weights_created=0), indent=2) + '\n')
print(json.dumps(dict(status='complete', completed=completed, weights_created=0)), flush=True)
