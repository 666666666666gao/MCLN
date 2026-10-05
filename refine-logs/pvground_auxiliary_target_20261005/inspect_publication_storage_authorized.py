"""Inspect evidence storage only; no GPU/process query or file mutation."""
import json
import os
import shlex

import paramiko


client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
python = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
probe = '''
import datetime,json,os,shutil,subprocess
from pathlib import Path
project=Path('/home/gb/new butd/butd_detr-main/MCLN-main')
evidence=project/'refine-logs'
data=Path('/root/autodl-tmp')
usage=[]
for child in sorted(evidence.iterdir()):
    raw=subprocess.check_output(['du','-sk','--',str(child)],text=True)
    usage.append(dict(name=child.name,bytes_kib=int(raw.split()[0])*1024))
print(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    project=str(project),evidence=str(evidence),evidence_resolved=str(evidence.resolve()),
    evidence_is_symlink=evidence.is_symlink(),evidence_device=evidence.stat().st_dev,
    root_device=Path('/').stat().st_dev,data_device=data.stat().st_dev,
    system_free_bytes=shutil.disk_usage('/').free,data_free_bytes=shutil.disk_usage(data).free,
    evidence_total_bytes_kib=sum(x['bytes_kib'] for x in usage),
    largest_children=sorted(usage,key=lambda x:x['bytes_kib'],reverse=True)[:15])))
'''
_, stdout, stderr = client.exec_command(shlex.join([python, '-B', '-c', probe]), timeout=120)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
from pathlib import Path
(Path(__file__).resolve().parent / 'PUBLICATION_STORAGE_INSPECTION.json').write_bytes(raw)
client.close()
print(json.dumps(record), flush=True)
