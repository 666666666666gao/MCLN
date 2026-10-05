"""Collect completed diagnostic arrays/logs, no weight or model replay."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import paramiko

local = Path(__file__).resolve().parent
closed = json.loads((local/'wait.json').read_bytes())
assert closed['observer_closed'] and closed['exitcode'] == 0 and closed['status']['status'] == 'complete'
destination = local/'complete'
destination.mkdir()
root = json.loads((local/'spec.json').read_bytes())['root']
client = paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
names = sftp.listdir(root)
assert not any(name.endswith(('.pth','.pt')) for name in names)
files = []
for name in sorted(names):
    assert '/' not in name and '\\' not in name
    with sftp.open(root+'/'+name,'rb') as stream:
        raw = stream.read()
    (destination/name).write_bytes(raw)
    files.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
sftp.close();client.close()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),files=files,
    total_bytes=sum(entry['bytes'] for entry in files),model_replayed=False,weights_copied=0,root=root)
(destination/'INTAKE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
