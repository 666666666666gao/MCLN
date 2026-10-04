"""One real-model face-conditioned probe; no saved checkpoint or formal fit."""
import datetime
import json
from pathlib import Path
import subprocess
import time


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


root = Path(__file__).parent
assert not (root / 'status.json').exists()
record = dict(status='running', started_cst=now(), completed=[])
for arm in ('face_conditioned',):
    directory = root / arm
    spec = json.loads((directory / 'spec.json').read_bytes())
    assert spec['boundary_mode'] == 'distribution' and spec['head_architecture'] == arm and spec['head_only'] and spec['use_whole_range']
    environment = json.loads((Path(spec['runtime']) / 'env_spec.json').read_bytes())
    variables = dict(environment['env'])
    variables['PYTHONPATH'] = str(directory) + ':' + variables['PYTHONPATH']
    command = ['env'] + [key + '=' + value for key, value in variables.items()] + [
        spec['runtime'] + '/venv/bin/python', '-B', '-u', str(root / 'run_face_fit.py'),
        '--spec', str(directory / 'spec.json'), '--mode', 'preflight']
    record.update(phase=arm, phase_started_cst=now())
    (root / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
    begin = time.monotonic()
    with (directory / 'preflight.log').open('x') as output:
        code = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
    (directory / 'preflight.exit').write_text(str(code) + '\n')
    if code:
        record.update(status='failed', exit_code=code, finished_cst=now())
        (root / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
        raise SystemExit(code)
    receipt = json.loads((directory / 'preflight.json').read_bytes())
    assert receipt['status'] == 'pass' and receipt['batch_size'] == 8 and receipt['optimizer_steps'] == 2
    assert receipt['boundary_mode'] == 'distribution' and receipt['head_architecture'] == arm and receipt['head_parameters'] == spec['head_parameters']
    assert receipt['head_only'] and receipt['original_g_state_unchanged']
    assert receipt['same_cached_inputs_zero_head_common_floor_exact'] and receipt['weight_files_created'] == 0
    if arm == 'face_conditioned':
        assert all(step['edge_alone_output_gradient'] > 0 for step in receipt['steps'])
    record['completed'].append(dict(arm=arm, seconds=time.monotonic()-begin, finished_cst=now()))
record.update(status='complete', finished_cst=now(), exit_code=0,
              real_model_updates=2, formal_training_started=False, weights_created=0)
(root / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
