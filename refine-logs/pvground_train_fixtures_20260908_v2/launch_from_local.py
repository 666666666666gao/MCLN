import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/pvground_train_fixtures_20260908_v2'
archive.mkdir()
root='/root/autodl-tmp/mcln_pvground_train_fixtures_20260908_v2'
manifest='/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/input_manifest.json'
source=(repo/'scripts/export_pvground_train_fixtures.py').read_bytes()
spec={'root':root,'manifest':manifest,'fixture_rows':4,'selection':'first four distinct physical fit scenes','seed':2027,'source_sha256':hashlib.sha256(source).hexdigest(),'model_forwards':0,'optimizer_steps':0,'formal_rows':0}
raw=(json.dumps(spec,indent=2)+'\n').encode()
(archive/'spec.json').write_bytes(raw)
(archive/'export_fixtures.py').write_bytes(source)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp();sftp.mkdir(root)
for name,data in [('spec.json',raw),('export_fixtures.py',source)]:
    with sftp.open(root+'/'+name,'wx') as stream:stream.write(data)
command=['/root/miniconda3/envs/bdetr/bin/python','-u',root+'/export_fixtures.py','--manifest',manifest,'--output',root+'/fixtures']
controller='#!/bin/bash\nexport CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1\n'+ ' '.join(shlex.quote(x) for x in command)+'\nresult=$?\nprintf "%s\\n" "$result" > '+shlex.quote(root+'/controller.exit')+'\nexit "$result"\n'
(archive/'controller.sh').write_bytes(controller.encode())
with sftp.open(root+'/controller.sh','wx') as stream:stream.write(controller.encode())
inner='exec bash '+shlex.quote(root+'/controller.sh')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_,stdout,stderr=client.exec_command('screen -dmS mcln_pvg_fixtures_v2 bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/export_fixtures.py'),timeout=30)
process=stdout.read().decode();assert stdout.channel.recv_exit_status()==0 and process.strip(),stderr.read().decode()
receipt={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'process':process.strip(),'screen':'mcln_pvg_fixtures_v2','estimated_seconds':[90,240],'gpu_used':False}
raw=(json.dumps(receipt,indent=2)+'\n').encode()
(archive/'launch.json').write_bytes(raw)
with sftp.open(root+'/launch.json','wx') as stream:stream.write(raw)
(archive/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
sftp.close();client.close();print('PVG_FIXTURES_LAUNCHED '+json.dumps(receipt))
