"""Launch one CPU-only real Nr3D batch preparation, without loading weights."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_nr_native_batch_20260908_v1'
archive=repo/'refine-logs/pvground_nr_native_batch_20260908_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
spec=json.loads((repo/'refine-logs/pvground_referit3d_voxel_cpu_20260908_v1/spec.json').read_bytes())
with s.open(spec['runtime']+'/env_spec.json','rb') as f:env=json.loads(f.read())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
spec.update(env_spec_sha256=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            batch_size=8,rows='existing fixed selection indices0,1,2,3,12,13,14,15 for Nr3D',augmentation=True,
            scope='CPU persist actual native input and labels; no model, optimizer, checkpoints, or formal rows')
files={'prepare.py':(repo/'scripts/prepare_pvground_nr_native_batch.py').read_bytes()}
spec['script_sha256']=hashlib.sha256(files['prepare.py']).hexdigest()
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
files['controller.py']=b'''import json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\\n')
spec=json.loads((root/'spec.json').read_bytes());runtime=Path(spec['runtime'])
env=os.environ.copy();env.update(json.loads((runtime/'env_spec.json').read_bytes())['env'])
env['CUDA_VISIBLE_DEVICES']=''
with (root/'run.log').open('xb') as log:
    result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'prepare.py'),'--spec',str(root/'spec.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''
probe="import json,shutil;print(json.dumps({'free_bytes':shutil.disk_usage('/root/autodl-tmp').free}))"
_,out,err=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(probe),timeout=30)
state=json.loads(out.read());assert out.channel.recv_exit_status()==0,err.read().decode()
assert state['free_bytes']>1500000000
assert root.rsplit('/',1)[1] not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as f:f.write(raw)
    with s.open(root+'/'+name,'rb') as f:assert f.read()==raw
argv=['screen','-dmS','mcln_pvg_nr_native_batch_v1','/root/miniconda3/envs/bdetr/bin/python',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,argv)),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
record=dict(root=root,time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            argv=argv,state_before_launch=state,scope=spec['scope'],model_forwards=0,optimizer_steps=0,formal_rows=0)
(archive/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
s.close();c.close()
