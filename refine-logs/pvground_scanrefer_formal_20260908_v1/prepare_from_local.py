import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
root='/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260908_v1'
training='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_v1'
audit_root='/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_v1'
archive=repo/'refine-logs/pvground_scanrefer_formal_20260908_v1'
train_spec_bytes=(repo/'refine-logs/pvground_scanrefer_finetune_20260908_v1/spec.json').read_bytes()
train_spec=json.loads(train_spec_bytes)
sources={'evaluate.py':repo/'scripts/evaluate_pvground_scanrefer_official.py',
    'audit.py':repo/'scripts/audit_pvground_scanrefer_official.py',
    'plan.md':repo/'docs/PVG_SCANREFER_FORMAL_PLAN_2026-09-08.md',
    'formal_input_contract.json':repo/'refine-logs/pvground_scanrefer_endpoint_audit_20260908_v1/formal_input_contract.json'}
queue='''import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(ROOT)
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert (root/'cpu_probe.exit').read_text().strip()=='0'
assert json.loads((root/'preparation_receipt.json').read_bytes())['status']=='pass'
deadline=datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp()
time.sleep(max(0,deadline-time.time()))
training=Path(spec['training_root']);audit_root=Path(spec['training_audit_root'])
while not (audit_root/'controller.exit').is_file():
    process=Path('/proc')/str(spec['training_audit_controller_pid'])/'cmdline'
    assert process.is_file() and (str(audit_root)+'/controller.py').encode() in process.read_bytes(),'endpoint audit controller disappeared without exit receipt'
    time.sleep(spec['poll_seconds'])
assert (training/'controller.exit').is_file()
codes={'training':int((training/'controller.exit').read_text()),'audit':int((audit_root/'controller.exit').read_text())}
if any(codes.values()):
    result={'status':'dependency_failed','exit_codes':codes,'formal_rows':0}
    (root/'decision.json').write_text(json.dumps(result)+'\\n')
    print('PVG_FORMAL_DEPENDENCY_FAILED '+json.dumps(result),flush=True)
    raise SystemExit(1)
audited=json.loads((audit_root/'audit.json').read_bytes())
assert audited['integrity_pass']
assert audited['receipt_sha256']==hashlib.sha256((training/'receipt.json').read_bytes()).hexdigest()
if not audited['primary_rec_nonregression']:
    result={'status':'skipped_primary_rec_regression','formal_rows':0,'primary_mode':'bbs','transitions':audited['transitions']['bbs']}
    (root/'decision.json').write_text(json.dumps(result)+'\\n')
    print('PVG_FORMAL_SKIPPED '+json.dumps(result),flush=True)
    raise SystemExit(0)
runtime=Path(spec['runtime'])
environment=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
command=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',str(runtime/'venv/bin/python'),'-u',str(root/'evaluate.py'),'--spec',str(root/'spec.json')]
decision={'status':'launching_fixed_formal','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'training_receipt_sha256':hashlib.sha256((training/'receipt.json').read_bytes()).hexdigest(),
          'training_audit_sha256':hashlib.sha256((audit_root/'audit.json').read_bytes()).hexdigest(),'formal_rows_expected':9508}
(root/'decision.json').write_text(json.dumps(decision)+'\\n')
print('PVG_FORMAL_LAUNCH '+json.dumps(decision),flush=True)
result=subprocess.run(command,cwd=str(runtime/'PV-Ground'),env=dict(os.environ,**environment['env']))
(root/'evaluation.exit').write_text(str(result.returncode)+'\\n')
if result.returncode!=0:raise SystemExit(result.returncode)
result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'audit.py'),'--root',str(root),'--out',str(root/'audit.json')])
(root/'audit.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''.replace('ROOT',repr(root))
controller='''import subprocess
from pathlib import Path
root=Path(ROOT)
result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'queue.py')])
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''.replace('ROOT',repr(root))
probe='''import datetime,hashlib,importlib.util,json,sys,time
from pathlib import Path
root=Path(ROOT)
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
begin=time.time()
modules={}
for name in ['audit','evaluate']:
    module_spec=importlib.util.spec_from_file_location('formal_'+name,str(root/(name+'.py')))
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    modules[name]=module
directory=Path(spec['training_root'])/'initial'
rows,metrics,checks=modules['audit'].recount_native_rows(directory)
assert len(rows)==6887
assert metrics=={mode:modules['evaluate'].row_metrics(rows,mode) for mode in ['bbs','bbf']}
assert 'torch' not in sys.modules
result={'status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'scope':'new formal recount code checked on completed initial holdout outputs',
        'actual_recount_rows':6887,'formal_rows':0,'gpu_forwards':0,'optimizer_steps':0,
        'torch_imported':False,'elapsed_seconds':time.time()-begin,'checks':checks,
        'initial_receipt_sha256':hashlib.sha256((directory/'receipt.json').read_bytes()).hexdigest(),
        'evaluator_sha256':spec['files']['evaluate.py'],'auditor_sha256':spec['files']['audit.py']}
with (root/'preparation_receipt.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\\n')
print('PVG_FORMAL_CPU_PREPARATION_PASS '+json.dumps(result),flush=True)
'''.replace('ROOT',repr(root))
for name,text in [('queue.py',queue),('controller.py',controller),('cpu_probe.py',probe)]:
    compile(text,name,'exec')
data={name:path.read_bytes() for name,path in sources.items()}
data.update({'queue.py':queue.encode(),'controller.py':controller.encode(),'cpu_probe.py':probe.encode()})
spec={'schema':'pvg-scanrefer-official-fixed-v1','training_root':training,'training_audit_root':audit_root,
    'training_spec_sha256':hashlib.sha256(train_spec_bytes).hexdigest(),
    'training_audit_controller_pid':6398,'runtime':train_spec['runtime'],'env_spec_sha256':train_spec['env_spec_sha256'],
    'first_check_cst':'2026-09-08T10:30:00+08:00','poll_seconds':300,'primary_mode':'bbs','seed':2027,'batch_size':8,
    'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in data.items()}}
data['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(training+'/spec.json','rb') as stream:
    assert stream.read()==train_spec_bytes
_,stdout,stderr=client.exec_command('ps -p 5874,6398 -o pid,etimes,args',timeout=30)
processes=stdout.read().decode()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert training+'/controller.py' in processes and audit_root+'/controller.py' in processes
archive.mkdir();sftp.mkdir(root)
for name,raw in data.items():
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw
    (archive/name).write_bytes(raw)
    with sftp.open(root+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==raw,name
_,stdout,stderr=client.exec_command('/root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/cpu_probe.py'),timeout=60)
raw=stdout.read();error=stderr.read();code=stdout.channel.recv_exit_status()
(archive/'cpu_probe.log').write_bytes(raw+error)
(archive/'cpu_probe.exit').write_text(str(code)+'\n')
with sftp.open(root+'/cpu_probe.log','wx') as stream:stream.write(raw+error)
with sftp.open(root+'/cpu_probe.exit','wx') as stream:stream.write(str(code)+'\n')
print(raw.decode(),end='',flush=True);print(error.decode(),end='',flush=True)
assert code==0,'CPU preparation failed; original training unchanged'
sftp.get(root+'/preparation_receipt.json',str(archive/'preparation_receipt.json'))
screen='mcln_pvg_formal_queue_v1'
inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_,stdout,stderr=client.exec_command('screen -dmS '+screen+' bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=stdout.read().decode().strip()
assert stdout.channel.recv_exit_status()==0 and process,stderr.read().decode()
record={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'process':process,'screen':screen,'first_check_cst':spec['first_check_cst'],'poll_seconds':300,
    'training_and_endpoint_audit_processes':processes,'cpu_preparation':'pass','formal_rows':0,'gpu_forwards':0,
    'conditional_on_fixed_endpoint_rec_nonregression':True,'training_modified':False,
    'spec_sha256':hashlib.sha256(data['spec.json']).hexdigest()}
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
(archive/'prepare_from_local.py').write_bytes(Path(__file__).read_bytes())
with sftp.open(root+'/launch.json','wx') as stream:stream.write(raw)
sftp.close();client.close()
print('PVG_FORMAL_QUEUE_PREPARED '+json.dumps(record),flush=True)
