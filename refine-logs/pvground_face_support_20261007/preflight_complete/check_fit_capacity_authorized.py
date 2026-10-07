"""Read current closed-stage storage only; no launch, cleanup or GPU query."""
import datetime
import json
import os
from pathlib import Path
import shlex

import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'closed_fit_capacity.json').exists()
wait = json.loads((local / 'preflight_wait.json').read_bytes())
proof = json.loads((local / 'preflight_complete/preflight.json').read_bytes())
assert wait['observer_closed'] and wait['exitcode'] == 0
assert proof['status'] == 'pass' and proof['optimizer_steps_per_arm'] == 2
spec = json.loads((local / 'pair_spec.json').read_bytes())
reserve = 3 * max(value['serialization_bytes'] for value in proof['cpu_restore'].values()) + 900 * 1024**2
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
code = '''import datetime,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);reserve=int(sys.argv[2])
assert (root/'preflight_controller.exit').read_text().strip()=='0'
assert json.loads((root/'preflight_status.json').read_bytes())['status']=='complete'
free=shutil.disk_usage(root).free
print(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    root=str(root),data_free_bytes=free,system_free_bytes=shutil.disk_usage('/').free,
    required_reserve_bytes=reserve,save_reserve_pass=free>=reserve,
    preflight_closed=True,formal_fit_started=(root/'fit_status.json').exists(),
    optimizer_or_inference_executed=False,files_deleted=0)))
'''
_, stdout, stderr = client.exec_command(shlex.join([spec['runtime'] + '/venv/bin/python',
    '-B', '-c', code, spec['root'], str(reserve)]), timeout=30)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
assert not record['formal_fit_started']
(local / 'closed_fit_capacity.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
client.close()
print(json.dumps(record), flush=True)
