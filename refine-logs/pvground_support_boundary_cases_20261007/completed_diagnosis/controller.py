"""One read-only support collector; actual child closure is persisted."""
import datetime
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
assert not (root / 'status.json').exists()
spec = json.loads((root / 'diagnostic_spec.json').read_bytes())
environment = json.loads((Path(spec['runtime']) / 'env_spec.json').read_bytes())
variables = dict(environment['env'])
variables['PYTHONPATH'] = str(root) + ':' + spec['helper_root'] + ':' + variables['PYTHONPATH']
started = time.perf_counter()
record = dict(status='running', started_cst=datetime.datetime.now().astimezone().isoformat(),
    diagnostic_rows=191, optimizer_updates=0, weights_created=0)
(root / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
command = ['env'] + [key + '=' + value for key, value in variables.items()] + [
    spec['runtime'] + '/venv/bin/python', '-B', '-u', str(root / 'collect_support_cases.py')]
with (root / 'collector.log').open('x') as stream:
    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    record['child_pid'] = child.pid
    (root / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
    code = child.wait()
(root / 'collector.exit').write_text(str(code) + '\n')
record.update(status='complete' if code == 0 else 'failed', exitcode=code,
    finished_cst=datetime.datetime.now().astimezone().isoformat(), elapsed_seconds=time.perf_counter()-started)
(root / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
raise SystemExit(code)
