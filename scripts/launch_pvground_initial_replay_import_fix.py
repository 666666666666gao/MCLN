"""Resume the frozen diagnostic in a clean directory without stdlib queue shadowing."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
old='/root/autodl-tmp/mcln_pvground_initial_replay_20260908_v1'
root='/root/autodl-tmp/mcln_pvground_initial_replay_20260908_v2'
archive=repo/'refine-logs/pvground_initial_replay_20260908_v2'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(old+'/controller.exit','rb') as f:assert f.read().strip()==b'1'
with s.open(old+'/process_a.log','rb') as f:failure=f.read()
assert b"AttributeError: module 'queue' has no attribute 'Queue'" in failure
assert 'process_a' not in s.listdir(old) and 'process_b.log' not in s.listdir(old)
_,out,err=c.exec_command('ps -eo pid,args',timeout=30)
assert not [line for line in out.read().decode().splitlines() if old in line]
assert out.channel.recv_exit_status()==0 and not err.read()
_,out,err=c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status()==0 and not err.read()
with s.open(old+'/spec.json','rb') as f:spec=json.loads(f.read())
with s.open(spec['runtime']+'/env_spec.json','rb') as f:env=json.loads(f.read())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
with s.open(old+'/replay.py','rb') as f:assert f.read()==(repo/'scripts/replay_pvground_initial_batch.py').read_bytes()
with s.open(old+'/queue.py','rb') as f:assert f.read()==(repo/'scripts/run_pvground_initial_replay_queue.py').read_bytes()
files={'replay.py':(repo/'scripts/replay_pvground_initial_batch.py').read_bytes(),
       'replay_queue.py':(repo/'scripts/run_pvground_initial_replay_queue.py').read_bytes()}
with s.open(old+'/controller.py','rb') as f:controller=f.read()
assert controller.count(b'root/"queue.py"')==1
files['controller.py']=controller.replace(b'root/"queue.py"',b'root/"replay_queue.py"')
for name,raw in files.items():compile(raw,name,'exec')
spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
spec['import_fix_from']=old
spec['import_fix']='deployment filename queue.py -> replay_queue.py; model and diagnostic unchanged'
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
assert root.rsplit('/',1)[1] not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as f:f.write(raw)
    with s.open(root+'/'+name,'rb') as f:assert f.read()==raw
s.symlink(old+'/inputs',root+'/inputs')
with s.open(old+'/input_controller.exit','rb') as f:raw=f.read()
assert raw.strip()==b'0'
with s.open(root+'/input_controller.exit','wb') as f:f.write(raw)
argv=['screen','-dmS','mcln_pvg_initial_replay_v2','/root/miniconda3/bin/python',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,argv)),timeout=30)
assert out.channel.recv_exit_status()==0 and not err.read()
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),root=root,
    previous_attempt=old,previous_failure='stdlib queue shadowed during dependency import',
    model_script_unchanged=True,queue_source_unchanged=True,environment_unchanged=True,
    frozen_inputs_shared=True,argv=argv,optimizer_steps=0,formal_rows=0)
(archive/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));s.close();c.close()
