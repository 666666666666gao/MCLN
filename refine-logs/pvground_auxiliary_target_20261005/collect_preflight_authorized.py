"""Collect the actually closed sanity artifacts; no model or optimizer replay."""
import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

local = Path(__file__).resolve().parent
wait = json.loads((local / 'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['controller_alive']
assert wait['exitcode'] == 0 and wait['status']['status'] == 'complete'
destination = local / 'preflight_complete'
destination.mkdir()
root = json.loads((local / 'preflight_launch.json').read_bytes())['root']
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
files = []
names = ['preflight_status.json', 'preflight_controller.log', 'preflight_controller.exit',
    'run_geometry_fit.py', 'query_supported_geometry.py', 'controller.py', 'control_spec.json',
    'member_target_spec.json', 'SOURCE_REVIEW.json', 'SOURCE_REVIEW.md', 'EXPERIMENT_PLAN.md']
for arm in ('control', 'member_target'):
    names.extend(arm + '/' + name for name in ('imports.json', 'load.json', 'preflight.json', 'preflight.log', 'preflight.exit'))
    assert not any(name.endswith(('.pth', '.pt', '.tmp')) for name in sftp.listdir(root + '/' + arm))
for name in names:
    with sftp.open(root + '/' + name, 'rb') as stream:
        raw = stream.read()
    path = destination / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    files.append(dict(name=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
sftp.close()
client.close()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), root=root, files=files,
    total_bytes=sum(entry['bytes'] for entry in files), weights_copied=0, model_replayed=False)
(destination / 'INTAKE.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
