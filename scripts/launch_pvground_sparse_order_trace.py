"""Launch two observational backbone forwards after the completed full replay."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_sparse_order_20260908_v1'
previous='/root/autodl-tmp/mcln_pvground_initial_replay_20260908_v2'
archive=repo/'refine-logs/pvground_sparse_order_20260908_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(previous+'/controller.exit','rb') as f:assert f.read().strip()==b'0'
with s.open(previous+'/comparison.json','rb') as f:prior=json.loads(f.read())
assert prior['first_observed_difference']=='backbone.point_features'
with s.open(previous+'/spec.json','rb') as f:spec=json.loads(f.read())
_,out,err=c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status()==0 and not err.read()
files={'trace.py':(repo/'scripts/trace_pvground_sparse_order.py').read_bytes()}
files['controller.py']=b'''import fcntl,json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\\n')
spec=json.loads((root/'spec.json').read_bytes());runtime=Path(spec['runtime'])
env=os.environ.copy();env.update(json.loads((runtime/'env_spec.json').read_bytes())['env'])
env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with open(spec['gpu_lock'],'a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (root/'run.log').open('xb') as log:
        result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'trace.py'),'--spec',str(root/'spec.json'),'--output',str(root/'result')],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''
for name,raw in files.items():compile(raw,name,'exec')
spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
spec['scope']='two seeded backbone forwards; sparse coordinate ordering and canonical feature comparison; no mutation'
spec['prerequisite']=previous
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
assert root.rsplit('/',1)[1] not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as f:f.write(raw)
    with s.open(root+'/'+name,'rb') as f:assert f.read()==raw
s.symlink('/root/autodl-tmp/mcln_pvground_initial_replay_20260908_v1/inputs',root+'/inputs')
argv=['screen','-dmS','mcln_pvg_sparse_order_v1','/root/miniconda3/bin/python',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,argv)),timeout=30)
assert out.channel.recv_exit_status()==0 and not err.read()
record=dict(root=root,time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    argv=argv,scope=spec['scope'],optimizer_steps=0,formal_rows=0)
(archive/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));s.close();c.close()
