import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import time
import urllib.request

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/pvground_scan_checkpoint_inspection_20260908_v1'
remote='/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1'
plan=json.loads((archive/'plan.json').read_bytes())
cache=Path('C:/Users/gb/.codex/tmp/mcln_pv_scan_transfer_20260908')
cache.mkdir()
local=cache/plan['filename']
free=shutil.disk_usage(str(cache)).free
assert free>plan['bytes']+2*1024**3
start=time.time()
hash_value=hashlib.sha256()
size=0
with urllib.request.urlopen(plan['url'],timeout=60) as response, local.open('xb') as target:
    assert response.status==200 and int(response.headers['Content-Length'])==plan['bytes']
    for block in iter(lambda:response.read(8*1024*1024),b''):
        target.write(block);hash_value.update(block);size+=len(block)
assert size==plan['bytes'] and hash_value.hexdigest()==plan['sha256']
download={'status':'complete','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'route':'local HTTPS, then SSH transfer; remote direct download failed before any checkpoint bytes',
    'bytes':size,'sha256':hash_value.hexdigest(),'download_seconds':time.time()-start,'local_free_before':free,
    'url':plan['url'],'local_temporary_file':str(local)}
(archive/'local_download_receipt.json').write_bytes((json.dumps(download,indent=2)+'\n').encode())
print('LOCAL PV DOWNLOAD COMPLETE '+json.dumps(download),flush=True)
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(remote+'/controller.exit','rb') as stream:assert stream.read().strip()==b'1'
check="from pathlib import Path; import shutil; r=Path("+repr(remote)+"); assert shutil.disk_usage(str(r)).free>"+str(plan['bytes']+2*1024**3)+"; assert not (r/"+repr(plan['filename'])+").exists(); assert not (r/"+repr(plan['filename']+'.part')+").exists()"
_,stdout,stderr=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
upload_start=time.time()
s.put(str(local),remote+'/'+plan['filename']+'.part')
verify="from pathlib import Path; import hashlib,json; r=Path("+repr(remote)+"); p=r/"+repr(plan['filename']+'.part')+"; h=hashlib.sha256(); f=p.open('rb'); [h.update(b) for b in iter(lambda:f.read(8*1024*1024),b'')]; f.close(); assert p.stat().st_size=="+str(plan['bytes'])+" and h.hexdigest()=="+repr(plan['sha256'])+"; p.rename(r/"+repr(plan['filename'])+"); print(json.dumps({'bytes':"+str(plan['bytes'])+",'sha256':h.hexdigest()}))"
_,stdout,stderr=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(verify),timeout=60)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
verified=json.loads(raw)
inventory_source=(archive/'inspect_checkpoint.py').read_text().split('import torch\n',1)[1]
inventory_source="from pathlib import Path\nimport collections,datetime,hashlib,json,os,torch\nroot=Path(__file__).resolve().parent\nplan=json.loads((root/'plan.json').read_bytes())\nassert os.environ['CUDA_VISIBLE_DEVICES']==''\ndestination=root/plan['filename']\n"+inventory_source
controller='''#!/usr/bin/env bash
set -euo pipefail
cd ROOT
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
trap 'printf "%s\\n" "$?" > cpu_inventory.exit' EXIT
/root/miniconda3/envs/bdetr/bin/python -u inspect_transferred_checkpoint.py
'''.replace('ROOT',shlex.quote(remote))
transfer={'status':'complete','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'remote_file':remote+'/'+plan['filename'],'bytes':verified['bytes'],'sha256':verified['sha256'],
    'upload_seconds':time.time()-upload_start,'original_download_exit_preserved':1,'gpu_forwards':0,'formal_rows':0}
files={'local_download_receipt.json':(archive/'local_download_receipt.json').read_bytes(),
       'transfer_receipt.json':(json.dumps(transfer,indent=2)+'\n').encode(),
       'inspect_transferred_checkpoint.py':inventory_source.encode(),'cpu_inventory.sh':controller.encode(),
       'transfer_from_local.py':Path(__file__).read_bytes()}
for name,blob in files.items():
    (archive/name).write_bytes(blob)
    with s.open(remote+'/'+name,'wx') as stream:stream.write(blob)
    with s.open(remote+'/'+name,'rb') as stream:assert stream.read()==blob
inner='exec bash '+shlex.quote(remote+'/cpu_inventory.sh')+' > '+shlex.quote(remote+'/cpu_inventory.log')+' 2>&1'
_,stdout,stderr=c.exec_command('screen -dmS mcln_pv_scan_cpu_inventory_v1 bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
_,stdout,stderr=c.exec_command('screen -ls',timeout=30)
text=stdout.read().decode();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
matches=[line.strip() for line in text.splitlines() if '.mcln_pv_scan_cpu_inventory_v1' in line];assert len(matches)==1
launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'screen':matches[0],
    'estimated_duration_seconds':60,'actual_model_execution':False,'cpu_inventory_complete':False}
blob=(json.dumps(launch,indent=2)+'\n').encode();(archive/'cpu_inventory_launch.json').write_bytes(blob)
with s.open(remote+'/cpu_inventory_launch.json','wx') as stream:stream.write(blob)
s.close();c.close()
print('PV SCAN TRANSFER VERIFIED '+json.dumps(transfer),flush=True)
print('PV SCAN CPU INVENTORY LAUNCHED '+json.dumps(launch),flush=True)
