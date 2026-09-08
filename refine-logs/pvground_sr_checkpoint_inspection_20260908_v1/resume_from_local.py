"""Resume the observed truncated Sr download once, then verify and transfer; no retry loop."""
import datetime,hashlib,json,os,shlex,time,urllib.request
from pathlib import Path
import paramiko
repo=Path(__file__).resolve().parents[1]
archive=repo/'refine-logs/pvground_sr_checkpoint_inspection_20260908_v1'
remote='/root/autodl-tmp/mcln_pvground_sr_checkpoint_inspection_20260908_v1'
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
plan=json.loads((archive/'plan.json').read_bytes());filename=plan['filename']
local=Path('C:/Users/gb/.codex/tmp/mcln_pv_sr_transfer_20260908')/filename
offset=local.stat().st_size
assert offset==397533346
digest=hashlib.sha256()
with local.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
assert digest.hexdigest()=='cec75c80cc877314ea7752d53829e18c5690819010292578e54ed37693c939c0'
failure=dict(status='initial_download_truncated',received_bytes=offset,expected_bytes=plan['bytes'],partial_sha256=digest.hexdigest(),uploaded=False)
with (archive/'initial_download_failure.json').open('x') as stream:json.dump(failure,stream,indent=2);stream.write('\n')
assert not (archive/'download.json').exists()
started=time.time()
request=urllib.request.Request(plan['url'],headers={'Range':'bytes=%d-'%offset})
with urllib.request.urlopen(request,timeout=60) as response:
    expected='bytes %d-%d/%d'%(offset,plan['bytes']-1,plan['bytes'])
    header=dict(status=response.status,content_range=response.headers.get('Content-Range'),content_length=response.headers.get('Content-Length'))
    (archive/'range_response.json').write_text(json.dumps(header,indent=2)+'\n',encoding='utf-8')
    assert response.status==206 and response.headers['Content-Range']==expected,header
    with local.open('ab') as stream:
        for block in iter(lambda:response.read(8*1024*1024),b''):stream.write(block)
size=local.stat().st_size;digest=hashlib.sha256()
with local.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
assert size==plan['bytes'] and digest.hexdigest()==plan['sha256']
download=dict(bytes=size,sha256=digest.hexdigest(),seconds=time.time()-started,resume_offset=offset,
              route='one validated HTTP Range continuation then SSH',temporary_file=str(local))
(archive/'download.json').write_text(json.dumps(download,indent=2)+'\n',encoding='utf-8')
print('SR_PARENT_RESUME_VERIFIED '+json.dumps(download),flush=True)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
assert filename not in sftp.listdir(remote) and filename+'.part' not in sftp.listdir(remote)
probe="import shutil,json;print(json.dumps({'free':shutil.disk_usage('/root/autodl-tmp').free}))"
_,stdout,stderr=client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(probe),timeout=30)
disk=json.loads(stdout.read());assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert disk['free']>size+2*1024**3
started=time.time();sftp.put(str(local),remote+'/'+filename+'.part')
verify="from pathlib import Path;import hashlib,json,shutil;p=Path("+repr(remote+'/'+filename+'.part')+");h=hashlib.sha256();f=p.open('rb');[h.update(b) for b in iter(lambda:f.read(8*1024*1024),b'')];f.close();assert p.stat().st_size=="+str(size)+" and h.hexdigest()=="+repr(plan['sha256'])+";p.rename(p.with_suffix(''));print(json.dumps({'sha256':h.hexdigest(),'free':shutil.disk_usage(str(p.parent)).free}))"
_,stdout,stderr=client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(verify),timeout=60)
verified=json.loads(stdout.read());assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
transfer=dict(remote_file=remote+'/'+filename,bytes=size,sha256=verified['sha256'],
    remote_free_after=verified['free'],seconds=time.time()-started)
inventory=(repo/'scripts/inspect_pvground_sr_checkpoint.py').read_bytes()
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
       'resume_from_local.py':Path(__file__).read_bytes()}
for name,raw in files.items():
    (archive/name).write_bytes(raw)
    with sftp.open(remote+'/'+name,'wb') as stream:stream.write(raw)
    with sftp.open(remote+'/'+name,'rb') as stream:assert stream.read()==raw
inner='exec bash '+shlex.quote(remote+'/controller.sh')+' > '+shlex.quote(remote+'/inventory.log')+' 2>&1'
_,stdout,stderr=client.exec_command('screen -dmS mcln_pvg_sr_cpu_inventory_v1 bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
_,stdout,stderr=client.exec_command('screen -ls',timeout=30)
screens=stdout.read().decode();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
matches=[line.strip() for line in screens.splitlines() if '.mcln_pvg_sr_cpu_inventory_v1' in line]
assert len(matches)==1
launch=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    screen=matches[0],env_spec_sha256=env_sha,argv=argv,model_forwards=0,training_launched=False)
raw=(json.dumps(launch,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with sftp.open(remote+'/launch.json','wb') as stream:stream.write(raw)
sftp.close();client.close()
print('SR_PARENT_TRANSFER_VERIFIED '+json.dumps(transfer),flush=True)
print('SR_CPU_INVENTORY_LAUNCHED '+json.dumps(launch),flush=True)
