"""Stage a bounded read-only capture after the current ScanRefer pipeline."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_support_capture_20260908_v1'
archive=repo/'refine-logs/pvground_support_capture_20260908_v1'
previous='/root/autodl-tmp/mcln_pvground_vsa_order_replay_20260908_v1'
dependency='/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260908_vsaorder_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(previous+'/spec.json','rb') as f:spec=json.loads(f.read())
formal_launch=json.loads((repo/'refine-logs/pvground_scanrefer_formal_20260908_vsaorder_v1/launch.json').read_bytes())
pid=int(formal_launch['process'].split()[0])
with s.open('/proc/'+str(pid)+'/cmdline','rb') as f:assert dependency.encode() in f.read()
with s.open(spec['runtime']+'/env_spec.json','rb') as f:env=json.loads(f.read())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
with s.open(previous+'/inputs/receipt.json','rb') as f:inputs=json.loads(f.read())
assert inputs['status']=='pass' and inputs['batch_size']==8
spec.update(dependency_root=dependency,dependency_pid=pid,first_check_cst='2026-09-08T17:25:00+08:00',
            poll_seconds=300,scope='one fixed batch: first observed and two unobserved replay forwards; no updates or formal rows')
files={
    'capture.py':(repo/'scripts/capture_pvground_support.py').read_bytes(),
    'pvground_support_observation.py':(repo/'scripts/pvground_support_observation.py').read_bytes(),
    'replay.py':(repo/'scripts/replay_pvground_vsa_order.py').read_bytes(),
    'replay_queue.py':(repo/'scripts/queue_pvground_support_capture.py').read_bytes(),
}
with s.open(previous+'/controller.py','rb') as f:files['controller.py']=f.read()
spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
probe="import json,shutil,subprocess; print(json.dumps({'disk_free':shutil.disk_usage('/root/autodl-tmp').free,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader']).decode(),'processes':subprocess.check_output(['ps','-p','14249,14358,14359','-o','pid,ppid,etimes,args']).decode()}))"
_,out,err=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(probe),timeout=30)
state=json.loads(out.read());assert out.channel.recv_exit_status()==0,err.read().decode()
assert state['disk_free']>1500000000
assert root.rsplit('/',1)[1] not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as f:f.write(raw)
    with s.open(root+'/'+name,'rb') as f:assert f.read()==raw
s.symlink(previous+'/inputs',root+'/inputs')
argv=['screen','-dmS','mcln_pvg_support_capture_v1','/root/miniconda3/envs/bdetr/bin/python',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,argv)),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
record=dict(root=root,time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            argv=argv,dependency_root=dependency,dependency_pid=pid,first_check_cst=spec['first_check_cst'],
            state_before_launch=state,input_sha256=inputs['input_file_sha256'],
            env_spec_sha256=spec['env_spec_sha256'],optimizer_steps=0,formal_rows=0,
            capture_status='queued only; actual forward not yet executed')
(archive/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));s.close();c.close()
