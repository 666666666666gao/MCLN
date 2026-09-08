"""Queue one CPU inspection near the first periodic snapshot; leave training untouched."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_observation_checkpoint_audit_20260909_v1'
archive=repo/'refine-logs/pvground_observation_checkpoint_audit_20260909_v1'
training='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260909_observation_v1'
train_spec=json.loads((repo/'refine-logs/pvground_scanrefer_finetune_20260909_observation_v1/spec.json').read_bytes())
files={'check.py':(repo/'scripts/check_pvground_observation_latest.py').read_bytes()}
files['checkpoint_queue.py']=('''import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(ROOT)
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
training=Path(spec['training_root'])
time.sleep(max(0,datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp()-time.time()))
while not (training/'latest.pth').is_file():
    original=Path('/proc')/str(spec['training_pid'])/'cmdline'
    assert original.is_file() and (str(training)+'/controller.py').encode() in original.read_bytes()
    time.sleep(300)
runtime=Path(spec['runtime'])
environment=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
command=[str(runtime/'venv/bin/python'),'-u',str(root/'check.py')]
with (root/'check.log').open('x') as stream:
    result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
        env=dict(os.environ,**dict(environment['env'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')))
(root/'check.exit').write_text(str(result.returncode)+'\\n')
assert result.returncode==0
lines=(root/'check.log').read_text().splitlines()
record=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('PVG_OBSERVATION_LATEST_CPU_PASS ')]
assert len(record)==1 and record[0]['status']=='pass'
(root/'receipt.json').write_text(json.dumps(record[0],indent=2)+'\\n')
print(json.dumps(record[0]),flush=True)
'''.replace('ROOT',repr(root))).encode()
controller=(repo/'refine-logs/pvground_scanrefer_formal_20260909_observation_v1/controller.py').read_text()
controller=controller.replace('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260909_observation_v1',root).replace("root/'formal_queue.py'","root/'checkpoint_queue.py'")
files['controller.py']=controller.encode()
files['plan.md']=b'One read-only CPU restore of first observed periodic checkpoint, after 02:38 CST; no training changes, new weights, scene inference or formal score. Uses the existing formal restore helpers. Original training PID29647; poll300s if snapshot not ready. Record actual observed step, not assumed step512.\n'
spec=dict(training_root=training,training_pid=29647,runtime=train_spec['runtime'],
    env_spec_sha256=train_spec['env_spec_sha256'],first_check_cst='2026-09-09T02:38:00+08:00',
    files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
_,out,err=c.exec_command('ps -p 29647 -o pid=,args=',timeout=30)
original=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and training+'/controller.py' in original
s=c.open_sftp();s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wx') as stream:stream.write(raw)
inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_,out,err=c.exec_command('screen -dmS mcln_pvg_observation_checkpoint_audit_v1 bash -c '+shlex.quote(inner),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,
    original_training_process=original,first_check_cst=spec['first_check_cst'],poll_seconds=300,
    formal_rows=0,model_forwards=0,optimizer_steps=0,status='queued_not_yet_checked')
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wx') as stream:stream.write(raw)
print(json.dumps(record),flush=True)
s.close();c.close()
