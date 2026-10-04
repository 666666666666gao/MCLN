"""Sequential CPU construction then two updates per arm under one GPU lock."""
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
for arm in ('evidence_hidden', 'evidence_visible'):
    directory = root / arm
    spec = json.loads((directory / 'spec.json').read_bytes())
    environment = json.loads((Path(spec['runtime']) / 'env_spec.json').read_bytes())
    variables = dict(environment['env'])
    variables['PYTHONPATH'] = str(directory) + ':' + variables['PYTHONPATH']
    for mode in ('cpu', 'preflight'):
        record.update(phase=arm + '/' + mode, phase_started_cst=now())
        (root / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
        command = ['env'] + [key + '=' + value for key, value in variables.items()] + [
            spec['runtime'] + '/venv/bin/python', '-B', '-u', str(directory / 'run_readback_preflight.py'),
            '--spec', str(directory / 'spec.json'), '--mode', mode]
        begin = time.monotonic()
        with (directory / (mode + '.log')).open('x') as output:
            code = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
        (directory / (mode + '.exit')).write_text(str(code) + '\n')
        if code:
            record.update(status='failed', exit_code=code, finished_cst=now())
            (root / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
            raise SystemExit(code)
        receipt = json.loads((directory / ('load.json' if mode == 'cpu' else 'preflight.json')).read_bytes())
        assert receipt['status'] == 'pass' and receipt['readback_parameters'] == 96672
        assert receipt['use_geometry_evidence'] == (arm == 'evidence_visible')
        if mode == 'preflight':
            assert receipt['optimizer_steps'] == 2 and receipt['weight_files_created'] == 0
            assert receipt['geometry_provider_and_g_states_exact'] and receipt['formal_rows'] == 0
        record['completed'].append(dict(arm=arm, mode=mode, seconds=time.monotonic() - begin))
record.update(status='complete', finished_cst=now(), exit_code=0, real_optimizer_updates=4,
              formal_training_started=False, weight_files_created=0, accuracy_result=False)
(root / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
