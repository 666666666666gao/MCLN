"""Collect closed preflight logs and receipts only; no model weights."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import paramiko

local = Path(__file__).resolve().parent
wait = json.loads((local / 'readback_preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
launch = json.loads((local / 'readback_preflight_launch.json').read_bytes())
root = launch['root']
target = local / 'complete_preflight'
assert not target.exists()
target.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
files = {}
for directory in ('', 'evidence_hidden', 'evidence_visible'):
    remote_dir = root + ('/' + directory if directory else '')
    for name in sftp.listdir(remote_dir):
        if not name.endswith(('.json', '.log', '.exit')):
            continue
        relative = directory + '/' + name if directory else name
        with sftp.open(root + '/' + relative, 'rb') as stream:
            raw = stream.read()
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files[relative] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
sftp.close()
client.close()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), files=files,
    remote_terminal=wait['terminal'], downloaded_weight_files=0, created_archived_weights=0,
    native_accuracy_rows=0, formal_training_started=False)
(target / 'INTAKE.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(files=len(files), weights=0, terminal_status=wait['terminal']['status']['status'])), flush=True)
