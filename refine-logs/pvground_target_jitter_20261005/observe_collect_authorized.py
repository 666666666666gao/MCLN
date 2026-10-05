"""Bounded completion observation and small output collection; no model replay."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import paramiko

local = Path(__file__).resolve().parent
launch = json.loads((local / 'launch.json').read_bytes())
root = launch['root']
spec = json.loads((local / 'spec.json').read_bytes())
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
probe = '''
import json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);exitfile=root/'run.exit'
print(json.dumps(dict(closed=exitfile.exists(),exitcode=int(exitfile.read_text()) if exitfile.exists() else None,
    log=(root/'run.log').read_text()[-5000:])))
'''
deadline = time.monotonic() + 900
while True:
    command = shlex.join([spec['runtime'] + '/venv/bin/python', '-c', probe, root])
    _, stdout, stderr = client.exec_command(command, timeout=30)
    raw = stdout.read()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    observed = json.loads(raw)
    if observed['closed']:
        break
    assert time.monotonic() < deadline, observed
    print(json.dumps(dict(status='RUNNING', next_check_seconds=240, log=observed['log'][-700:])), flush=True)
    time.sleep(240)
(local / 'wait.json').write_text(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    observer_closed=True, **observed), indent=2) + '\n', encoding='utf-8')
destination = local / 'complete'
destination.mkdir()
sftp = client.open_sftp()
files = []
for name in sorted(sftp.listdir(root)):
    assert not name.endswith(('.pth', '.pt')) and '/' not in name
    with sftp.open(root + '/' + name, 'rb') as stream:
        raw = stream.read()
    (destination / name).write_bytes(raw)
    files.append(dict(name=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
sftp.close()
client.close()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), files=files,
    total_bytes=sum(entry['bytes'] for entry in files), weights_copied=0,
    exitcode=observed['exitcode'], observer_closed=True)
(destination / 'INTAKE.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
assert observed['exitcode'] == 0, observed['log']
