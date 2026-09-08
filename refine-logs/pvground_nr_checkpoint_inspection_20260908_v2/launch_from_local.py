"""Inspect the existing parent using the author's flag semantics and actual buffer schema."""
import hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path(__file__).resolve().parents[1]
old=repo/'refine-logs/pvground_nr_checkpoint_inspection_20260908_v1'
archive=repo/'refine-logs/pvground_nr_checkpoint_inspection_20260908_v2'
root='/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v2'
parent='/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1'
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
assert (old/'inventory.exit').read_text().strip()=='1'
plan=json.loads((old/'plan.json').read_bytes())
plan.update(filename=parent+'/PV-Ground_NR3D.pth',
    correction='Author combines butd OR butd_gt OR butd_cls; Nr stores position_ids; original weights unchanged',
    previous_inventory_exit=1)
archive.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.mkdir(root)
with s.open(runtime+'/env_spec.json','rb') as stream:env=json.loads(stream.read())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
prefix=['env']+[k+'='+v for k,v in dict(env['env'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1').items()]
prefix+=[runtime+'/venv/bin/python','-u']
controller='''import json,subprocess
from pathlib import Path
root=Path(ROOT)
prefix=PREFIX
for stage in ['inventory','strict_load']:
    argv=prefix+[str(root/(stage+'.py'))]
    with (root/(stage+'.log')).open('w') as out,(root/(stage+'.stderr')).open('w') as err:
        result=subprocess.run(argv,stdout=out,stderr=err)
    (root/(stage+'.exit')).write_text(str(result.returncode)+'\\n')
    (root/(stage+'_execution.json')).write_text(json.dumps({'argv':argv,'exit':result.returncode},indent=2)+'\\n')
    print(stage+' '+str(result.returncode),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
'''.replace('ROOT',repr(root)).replace('PREFIX',repr(prefix))
shell="#!/usr/bin/env bash\nset -euo pipefail\ncd "+shlex.quote(root)+"\ntrap 'printf \"%s\\n\" \"$?\" > controller.exit' EXIT\n/root/miniconda3/envs/bdetr/bin/python -u controller.py\n"
files={'plan.json':(json.dumps(plan,indent=2)+'\n').encode(),
       'inventory.py':(repo/'scripts/inspect_pvground_nr_checkpoint.py').read_bytes(),
       'strict_load.py':(repo/'scripts/strict_load_pvground_nr_cpu.py').read_bytes(),
       'controller.py':controller.encode(),'controller.sh':shell.encode(),
       'launch_from_local.py':Path(__file__).read_bytes()}
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as stream:stream.write(raw)
    with s.open(root+'/'+name,'rb') as stream:assert stream.read()==raw
inner='exec bash '+shlex.quote(root+'/controller.sh')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_,out,err=c.exec_command('screen -dmS mcln_pvg_nr_cpu_inspect_v2 bash -c '+shlex.quote(inner),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('ps -eo pid,ppid,args',timeout=30)
processes=out.read().decode();assert out.channel.recv_exit_status()==0,err.read().decode()
matches=[line.strip() for line in processes.splitlines() if line.rstrip().endswith('python -u controller.py')]
assert matches
record=dict(root=root,controller_processes=matches,estimated_seconds=60,model_forwards=0,training_launched=False,
    source_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wb') as stream:stream.write(raw)
s.close();c.close();print(json.dumps(record),flush=True)
