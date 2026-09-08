"""Fetch the pinned official Nr3D parent and inspect it on CPU; never train."""
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

repo = Path(__file__).resolve().parents[1]
archive = repo / 'refine-logs/pvground_nr_checkpoint_inspection_20260908_v1'
remote = '/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1'
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
filename = 'PV-Ground_NR3D.pth'
plan = dict(dataset='nr3d', filename=filename, bytes=829830168,
    sha256='d2d9afaf9c293c54977f3555a46c7bb2a72d9f80163a3f602dd8426032d7fa5d',
    revision='cf4a8b1eed045f1309e6a691ea948d1d6c54448e',
    source='https://huggingface.co/AaNnWwTt/PV-Ground',
    purpose='CPU checkpoint preparation; no model forward, optimizer, or formal evaluation',
    scan_qualification_still_required_for_training=True)
plan['url'] = plan['source']+'/resolve/'+plan['revision']+'/'+filename
cache = Path('C:/Users/gb/.codex/tmp/mcln_pv_nr_transfer_20260908')
assert not archive.exists() and not cache.exists()
client = paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
probe = "import json,shutil;from pathlib import Path;assert not Path("+repr(remote)+").exists();print(json.dumps({'free':shutil.disk_usage('/root/autodl-tmp').free,'train_live':Path('/proc/14249').exists()}))"
_, stdout, stderr = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(probe), timeout=30)
state = json.loads(stdout.read());assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert state['train_live'] and state['free']>plan['bytes']+2*1024**3
assert shutil.disk_usage(str(cache.parent)).free>plan['bytes']+2*1024**3
archive.mkdir();cache.mkdir();sftp.mkdir(remote)
plan['remote_disk_before']=state['free']
raw=(json.dumps(plan,indent=2)+'\n').encode()
(archive/'plan.json').write_bytes(raw)
with sftp.open(remote+'/plan.json','wb') as stream:stream.write(raw)
local=cache/filename;started=time.time();digest=hashlib.sha256();size=0
with urllib.request.urlopen(plan['url'],timeout=60) as response, local.open('xb') as stream:
    assert response.status==200
    for block in iter(lambda:response.read(8*1024*1024),b''):
        stream.write(block);digest.update(block);size+=len(block)
assert size==plan['bytes'] and digest.hexdigest()==plan['sha256']
download=dict(bytes=size,sha256=digest.hexdigest(),seconds=time.time()-started,
    route='local HTTPS then SSH; remote direct HTTPS failed in prior Scan parent retrieval',
    temporary_file=str(local))
(archive/'download.json').write_text(json.dumps(download,indent=2)+'\n',encoding='utf-8')
print('NR_PARENT_LOCAL_VERIFIED '+json.dumps(download),flush=True)
started=time.time();sftp.put(str(local),remote+'/'+filename+'.part')
verify="from pathlib import Path;import hashlib,json,shutil;p=Path("+repr(remote+'/'+filename+'.part')+");h=hashlib.sha256();f=p.open('rb');[h.update(b) for b in iter(lambda:f.read(8*1024*1024),b'')];f.close();assert p.stat().st_size=="+str(size)+" and h.hexdigest()=="+repr(plan['sha256'])+";p.rename(p.with_suffix(''));print(json.dumps({'sha256':h.hexdigest(),'free':shutil.disk_usage(str(p.parent)).free}))"
_,stdout,stderr=client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(verify),timeout=60)
verified=json.loads(stdout.read());assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
transfer=dict(remote_file=remote+'/'+filename,bytes=size,sha256=verified['sha256'],
    remote_free_after=verified['free'],seconds=time.time()-started)
inventory=(repo/'scripts/inspect_pvground_nr_checkpoint.py').read_bytes()
compile(inventory,'inventory.py','exec')
with sftp.open(runtime+'/env_spec.json','rb') as stream:environment=json.loads(stream.read())
env_sha=hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
argv=['env']+[k+'='+v for k,v in dict(environment['env'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1').items()]
argv+=[runtime+'/venv/bin/python','-u',remote+'/inventory.py']
controller="#!/usr/bin/env bash\nset -euo pipefail\ncd "+shlex.quote(remote)+"\ntrap 'printf \"%s\\n\" \"$?\" > inventory.exit' EXIT\n"+' '.join(shlex.quote(x) for x in argv)+'\n'
files={'inventory.py':inventory,'controller.sh':controller.encode(),
       'transfer.json':(json.dumps(transfer,indent=2)+'\n').encode(),
       'download.json':(archive/'download.json').read_bytes(),
       'prepare_from_local.py':Path(__file__).read_bytes()}
for name,raw in files.items():
    (archive/name).write_bytes(raw)
    with sftp.open(remote+'/'+name,'wb') as stream:stream.write(raw)
    with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==raw
inner='exec bash '+shlex.quote(remote+'/controller.sh')+' > '+shlex.quote(remote+'/inventory.log')+' 2>&1'
_,stdout,stderr=client.exec_command('screen -dmS mcln_pvg_nr_cpu_inventory_v1 bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
_,stdout,stderr=client.exec_command('screen -ls',timeout=30)
screens=stdout.read().decode();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
matches=[line.strip() for line in screens.splitlines() if '.mcln_pvg_nr_cpu_inventory_v1' in line]
assert len(matches)==1
launch=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    screen=matches[0],env_spec_sha256=env_sha,argv=argv,model_forwards=0,training_launched=False)
raw=(json.dumps(launch,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with sftp.open(remote+'/launch.json','wb') as stream:stream.write(raw)
sftp.close();client.close()
print('NR_PARENT_TRANSFER_VERIFIED '+json.dumps(transfer),flush=True)
print('NR_CPU_INVENTORY_LAUNCHED '+json.dumps(launch),flush=True)
