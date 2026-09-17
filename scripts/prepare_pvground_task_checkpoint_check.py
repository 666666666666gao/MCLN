"""Launch one read-only CPU check after the first D checkpoint exists."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_task_checkpoint_check_20260917_v1'
training='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_task_observation_v1'
archive=repo/'refine-logs/pvground_task_checkpoint_check_20260917_v1'
train_spec=json.loads((repo/'refine-logs/pvground_scanrefer_finetune_20260917_task_observation_v1/spec.json').read_bytes())
files={'check.py':(repo/'scripts/check_pvground_task_observation_latest.py').read_bytes()}
controller='''import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
runtime=Path(spec['runtime'])
environment=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
with (root/'check.log').open('x') as stream:
    result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'check.py')],stdout=stream,stderr=subprocess.STDOUT,
        env=dict(os.environ,**dict(environment['env'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')))
(root/'check.exit').write_text(str(result.returncode)+'\\n')
if result.returncode==0:
    rows=[json.loads(line.split(' ',1)[1]) for line in (root/'check.log').read_text().splitlines() if line.startswith('PVG_TASK_OBSERVATION_LATEST_CPU_PASS ')]
    assert len(rows)==1 and rows[0]['status']=='pass'
    (root/'receipt.json').write_text(json.dumps(rows[0],indent=2)+'\\n')
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''
files['controller.py']=controller.encode()
files['plan.md']=b'One actual D periodic checkpoint CPU strict restore. Pin one open inode during atomic writer replacements; bind parent/spec/source and all 37 reader states, verify both task matrices updated. No GPU forward, optimizer update, new weights, quality inference or training changes. Record observed step rather than assume512.\n'
spec=dict(runtime=train_spec['runtime'],env_spec_sha256=train_spec['env_spec_sha256'],
          files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(training+'/launch.json','rb') as stream:launch=json.loads(stream.read())
pid=int(launch['process'].split()[0])
with s.open('/proc/'+str(pid)+'/cmdline','rb') as stream:assert (training+'/controller.py').encode() in stream.read()
assert 'latest.pth' in s.listdir(training)
assert Path(root).name not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wx') as stream:stream.write(raw)
inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_,out,err=c.exec_command('screen -dmS mcln_pvg_task_checkpoint_check_v1 bash -c '+shlex.quote(inner),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process,err.read().decode()
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            process=process,training_pid=pid,status='launched_not_yet_verified',gpu_forwards=0,optimizer_steps=0)
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wx') as stream:stream.write(raw)
s.close();c.close();print(json.dumps(record),flush=True)
