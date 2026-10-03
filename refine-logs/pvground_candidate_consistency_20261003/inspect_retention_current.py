"""Inspect owned checkpoint files and their actual filesystems, without changes."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).parent
spec = json.loads((local / 'g_control_spec.json').read_bytes())
remote_code = r'''
import json, os, shutil, subprocess
from pathlib import Path
root=Path('/root/autodl-tmp')
files=[]
for entry in root.iterdir():
    if not entry.is_dir() or not entry.name.startswith(('mcln_', 'cs_', 'pvground_')):
        continue
    for directory, dirs, names in os.walk(str(entry)):
        dirs[:]=[d for d in dirs if d not in ('venv','source','source_port','PV-Ground','.git','__pycache__','site-packages')]
        for name in names:
            if not name.endswith(('.pth','.ckpt')):
                continue
            p=Path(directory)/name
            s=p.stat()
            if s.st_size<1000000:
                continue
            files.append(dict(path=str(p),resolved=str(p.resolve()),bytes=s.st_size,
                device=s.st_dev,parent_device=p.parent.stat().st_dev,
                parent_free_bytes=shutil.disk_usage(str(p.parent)).free,
                file_free_bytes=shutil.disk_usage(str(p)).free,mtime=s.st_mtime))
paths=['/root/autodl-tmp','/root/autodl-tmp/pvground_candidate_consistency_20261003',
       '/root/autodl-tmp/pvground_p3_20261002/p3',
       '/root/autodl-tmp/pvground_tail_support_20261002/tail_raw',
       '/root/autodl-tmp/pvground_tail_support_fused_retry_20261002/tail_fused',
       '/home/gb/new butd/butd_detr-main/MCLN-main']
directories=[dict(path=p,resolved=str(Path(p).resolve()),device=Path(p).stat().st_dev,
                  free_bytes=shutil.disk_usage(p).free) for p in paths]
mounts=[line for line in Path('/proc/self/mountinfo').read_text().splitlines()
        if '/autodl-tmp' in line or '/home/gb' in line]
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).splitlines()
print(json.dumps(dict(directories=directories,checkpoint_files=files,mounts=mounts,compute_processes=gpu)))
'''
client=paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
               password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
python=spec['runtime']+'/venv/bin/python'
_,stdout,stderr=client.exec_command(shlex.join([python,'-c',remote_code]),timeout=90)
raw=stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record=json.loads(raw)
record.update(time_cst=datetime.datetime.now().astimezone().isoformat(),read_only=True)
(local/'retention_inventory_current.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
client.close()
print(json.dumps(dict(time_cst=record['time_cst'],directories=record['directories'],
                     checkpoint_file_count=len(record['checkpoint_files']),
                     checkpoint_bytes=sum(x['bytes'] for x in record['checkpoint_files']),
                     compute_processes=record['compute_processes'],
                     largest=sorted(record['checkpoint_files'],key=lambda x:-x['bytes'])[:12])))
