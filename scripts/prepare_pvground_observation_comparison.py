"""Prepare actual initial comparisons and audit-gated fixed terminal comparisons on CPU."""
import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_observation_comparison_20260909_v1'
archive=repo/'refine-logs/pvground_observation_comparison_20260909_v1'
candidate='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260909_observation_v1'
audit='/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260909_observation_v1'
controls={'A':'/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1',
          'B':'/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_sourcequery_v1'}
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
def read(path):
    with s.open(path,'rb') as f:return f.read()
assert (audit+'/controller.py').encode() in read('/proc/29653/cmdline')
assert json.loads(read(audit+'/initial_audit.json'))['integrity_pass']
files={'compare.py':(repo/'scripts/compare_pvground_observation_control.py').read_bytes()}
for label,control in controls.items():
    assert read(control+'/controller.exit').strip()==b'0'
    configs=[read(p+'/spec.json') for p in [control,candidate]]
    pair=dict(native_root=control,candidate_root=candidate,control_source_query=label=='B',
        training_spec_sha256=[hashlib.sha256(raw).hexdigest() for raw in configs],
        source_port_sha256=[hashlib.sha256(read(json.loads(raw)['source_port'])).hexdigest() for raw in configs])
    files[label+'_C.json']=(json.dumps(pair,indent=2)+'\n').encode()
queue='''import datetime,hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).parent
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
audit=Path(spec['candidate_audit']);candidate=Path(spec['candidate_root'])
for stage,filename in [('initial','initial_audit.json'),('terminal','audit.json')]:
    if stage=='terminal':time.sleep(max(0,datetime.datetime.fromisoformat(spec['terminal_first_check_cst']).timestamp()-time.time()))
    while not (audit/filename).is_file():
        if (audit/'controller.exit').is_file():raise RuntimeError('Original audit terminated without '+filename)
        process=Path('/proc')/str(spec['audit_controller_pid'])/'cmdline'
        assert process.is_file() and (str(audit)+'/controller.py').encode() in process.read_bytes()
        time.sleep(300)
    receipt=json.loads((audit/filename).read_bytes())
    assert receipt['integrity_pass'] and receipt['formal_rows']==0
    bound=candidate/('receipt.json' if stage=='terminal' else 'initial/receipt.json')
    assert hashlib.sha256(bound.read_bytes()).hexdigest()==receipt['receipt_sha256']
    for label in ['A','B']:
        prefix=stage+'_'+label+'_C'
        cmd=['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'compare.py'),'--spec',str(root/(label+'_C.json')),'--stage',stage,'--output',str(root/(prefix+'.json'))]
        with (root/(prefix+'.log')).open('xb') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        (root/(prefix+'.exit')).write_text(str(result.returncode)+'\\n')
        assert result.returncode==0,prefix
        print('PVG_OBSERVATION_COMPARISON_COMPLETE '+prefix,flush=True)
'''
controller='''import os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\\n')
with (root/'run.log').open('xb') as stream:result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'queue.py')],stdout=stream,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''
files.update({'queue.py':queue.encode(),'controller.py':controller.encode(),
    'plan.md':b'CPU-only actual initial comparisons A/C and B/C now; terminal comparisons after original audit29653, firstcheck04:45 then300s. Verify actual exports, same inputs, fixed fit order, own and cross-control changes. Full initial export differences are measured, not required to vanish. Raw256 oracle uses GT only for analysis. No model execution, training/decision edits, or checkpoint writes.\n'})
spec=dict(candidate_root=candidate,candidate_audit=audit,audit_controller_pid=29653,
    terminal_first_check_cst='2026-09-09T04:45:00+08:00',poll_seconds=300,model_forwards=0,formal_rows=0,training_changes=False,
    files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
assert Path(root).name not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wx') as f:f.write(raw)
    assert read(root+'/'+name)==raw
cmd=['screen','-dmS','mcln_pvg_observation_comparison_v1','/root/miniconda3/envs/bdetr/bin/python','-u',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,cmd)),timeout=30);assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process,err.read().decode()
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,
    first_terminal_check_cst=spec['terminal_first_check_cst'],poll_seconds=300,formal_rows=0,model_forwards=0,training_changes=False)
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wx') as f:f.write(raw)
s.close();c.close();print(json.dumps(record),flush=True)
