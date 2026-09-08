"""Queue CPU strict loading after the existing parent transfer/inventory."""
import hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path(__file__).resolve().parents[1]
archive=repo/'refine-logs/pvground_nr_checkpoint_inspection_20260908_v1'
root='/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1'
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();assert 'strict_load.py' not in s.listdir(root)
with s.open(runtime+'/env_spec.json','rb') as stream:env=json.loads(stream.read())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
argv=['env']+[k+'='+v for k,v in dict(env['env'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1').items()]
argv+=[runtime+'/venv/bin/python','-u',root+'/strict_load.py']
queue='''import datetime,json,subprocess,time
from pathlib import Path
root=Path(ROOT)
argv=ARGV
print('NR_STRICT_WAITING_FOR_INVENTORY',flush=True)
while not (root/'inventory.exit').exists():time.sleep(300)
dependency=int((root/'inventory.exit').read_text().strip())
if dependency!=0:
    (root/'strict_dependency_failure.json').write_text(json.dumps({'inventory_exit':dependency})+'\\n')
    raise SystemExit(1)
with (root/'strict_load.log').open('w') as out,(root/'strict_load.stderr').open('w') as err:
    result=subprocess.run(argv,stdout=out,stderr=err)
(root/'strict_load.exit').write_text(str(result.returncode)+'\\n')
(root/'strict_execution.json').write_text(json.dumps({'argv':argv,'exit':result.returncode,'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()},indent=2)+'\\n')
print('NR_STRICT_COMPLETE '+str(result.returncode),flush=True)
raise SystemExit(result.returncode)
'''.replace('ROOT',repr(root)).replace('ARGV',repr(argv))
controller="#!/usr/bin/env bash\nset -euo pipefail\ncd "+shlex.quote(root)+"\ntrap 'printf \"%s\\n\" \"$?\" > strict_controller.exit' EXIT\nexec_python=/root/miniconda3/envs/bdetr/bin/python\n\"$exec_python\" -u nr_strict_queue.py\n"
files={'strict_load.py':(repo/'scripts/strict_load_pvground_nr_cpu.py').read_bytes(),
       'nr_strict_queue.py':queue.encode(),'strict_controller.sh':controller.encode(),
       'strict_queue_from_local.py':Path(__file__).read_bytes()}
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as stream:stream.write(raw)
    with s.open(root+'/'+name,'rb') as stream:assert stream.read()==raw
inner='exec bash '+shlex.quote(root+'/strict_controller.sh')+' > '+shlex.quote(root+'/strict_queue.log')+' 2>&1'
_,out,err=c.exec_command('screen -dmS mcln_pvg_nr_strict_queue_v1 bash -c '+shlex.quote(inner),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('ps -eo pid,ppid,args',timeout=30)
processes=out.read().decode();assert out.channel.recv_exit_status()==0,err.read().decode()
matches=[line.strip() for line in processes.splitlines() if line.rstrip().endswith('python -u nr_strict_queue.py')]
assert len(matches)==1
record=dict(process=matches[0],root=root,wait_seconds=300,model_forwards=0,training_launched=False,
    source_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'strict_queue_launch.json').write_bytes(raw)
with s.open(root+'/strict_queue_launch.json','wb') as stream:stream.write(raw)
s.close();c.close();print(json.dumps(record),flush=True)
