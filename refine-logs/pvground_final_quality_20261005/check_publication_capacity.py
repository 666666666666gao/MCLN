"""Inspect the actual destination mount before publishing closed evidence."""
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).resolve().parent
probe="import json,os,shutil;from pathlib import Path;p=Path('/home/gb/new butd/butd_detr-main/MCLN-main/refine-logs/pvground_final_quality_20261005');print(json.dumps(dict(destination=str(p.resolve()),destination_device=os.stat(p).st_dev,data_device=os.stat('/root/autodl-tmp').st_dev,system_device=os.stat('/').st_dev,destination_free_bytes=shutil.disk_usage(str(p)).free,complete_exists=(p/'complete').exists())))"
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
_,stdout,stderr=client.exec_command(shlex.join(['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-c',probe]),timeout=30)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record=json.loads(raw);destination=local/'PUBLICATION_CAPACITY.json';assert not destination.exists()
destination.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');client.close()
print(json.dumps(record),flush=True)
