"""Queue CPU-only cross-control analysis after the existing independent audits."""
import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_empty_pool_comparison_20260908_v1'
native='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1'
candidate='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_emptypool_v1'
audit='/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_emptypool_v1'
archive=repo/'refine-logs/pvground_empty_pool_comparison_20260908_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(native+'/controller.exit') as f:assert f.read().strip()==b'0'
with s.open('/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_vsaorder_v1/audit.json') as f:
    assert json.loads(f.read())['integrity_pass']
audit_launch=json.loads((repo/'refine-logs/pvground_scanrefer_endpoint_audit_20260908_emptypool_v1/launch.json').read_bytes())
audit_pid=int(audit_launch['process'].split()[0])
with s.open('/proc/'+str(audit_pid)+'/cmdline') as f:assert (audit+'/controller.py').encode() in f.read()
digests=[]
for path in [native,candidate]:
    with s.open(path+'/spec.json','rb') as f:digests.append(hashlib.sha256(f.read()).hexdigest())
spec=dict(native_root=native,candidate_root=candidate,candidate_audit=audit,audit_controller_pid=audit_pid,
          training_spec_sha256=digests,poll_seconds=300,formal_rows=0,model_forwards=0,optimizer_steps=0,
          first_check_cst='2026-09-08T18:56:00+08:00',training_changes=False,decision_gate_changes=False)
queue='''import datetime,hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).parent
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
audit=Path(spec['candidate_audit']);candidate=Path(spec['candidate_root'])
time.sleep(max(0,datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp()-time.time()))
for stage,filename in [('initial','initial_audit.json'),('terminal','audit.json')]:
    while not (audit/filename).is_file():
        if (audit/'controller.exit').is_file():
            raise RuntimeError('Audit terminated without required '+filename+'; exit='+ (audit/'controller.exit').read_text().strip())
        process=Path('/proc')/str(spec['audit_controller_pid'])/'cmdline'
        assert process.is_file() and (str(audit)+'/controller.py').encode() in process.read_bytes(),'Original audit controller missing; do not restart training'
        time.sleep(spec['poll_seconds'])
    receipt=json.loads((audit/filename).read_bytes())
    assert receipt['integrity_pass'] and receipt['formal_rows']==0
    bound=candidate/('receipt.json' if stage=='terminal' else 'initial/receipt.json')
    assert hashlib.sha256(bound.read_bytes()).hexdigest()==receipt['receipt_sha256']
    command=['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'compare.py'),'--spec',str(root/'spec.json'),'--stage',stage,'--output',str(root/(stage+'.json'))]
    with (root/(stage+'.log')).open('xb') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    (root/(stage+'.exit')).write_text(str(result.returncode)+'\\n')
    assert result.returncode==0,stage
    print('PVG_CROSS_CONTROL_COMPLETE '+stage,flush=True)
'''
controller='''import os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\\n')
with (root/'run.log').open('xb') as stream:
    result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'queue.py')],stdout=stream,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''
files={'compare.py':(repo/'scripts/compare_pvground_empty_pool_control.py').read_bytes(),'queue.py':queue.encode(),'controller.py':controller.encode()}
spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
assert Path(root).name not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as f:f.write(raw)
    with s.open(root+'/'+name,'rb') as f:assert f.read()==raw
command=['screen','-dmS','mcln_pvg_empty_compare_v1','/root/miniconda3/envs/bdetr/bin/python','-u',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,command)),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process,err.read().decode()
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,
            first_check_cst=spec['first_check_cst'],poll_seconds=300,model_forwards=0,optimizer_steps=0,formal_rows=0)
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wb') as f:f.write(raw)
s.close();c.close();print(json.dumps(record),flush=True)
