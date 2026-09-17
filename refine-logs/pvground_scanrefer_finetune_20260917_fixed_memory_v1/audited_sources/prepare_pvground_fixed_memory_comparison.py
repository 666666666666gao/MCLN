"""Prepare actual initial comparisons and audit-gated fixed terminal comparisons on CPU."""
import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_fixed_memory_comparison_20260917_v1'
archive=repo/'refine-logs/pvground_fixed_memory_comparison_20260917_v1'
candidate='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_fixed_memory_v1'
audit='/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260917_fixed_memory_v1'
controls={'D':'/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_task_observation_v1'}
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
def read(path):
    with s.open(path,'rb') as f:return f.read()
audit_pid=int(json.loads(read(audit+'/launch.json'))['process'].split()[0])
assert (audit+'/controller.py').encode() in read('/proc/'+str(audit_pid)+'/cmdline')
files={'compare.py':(repo/'scripts/compare_pvground_fixed_memory_control.py').read_bytes()}
for label,control in controls.items():
    assert read(control+'/controller.exit').strip()==b'0'
    configs=[read(p+'/spec.json') for p in [control,candidate]]
    pair=dict(native_root=control,candidate_root=candidate,control_source_query=True,control_observation_state=True,
        training_spec_sha256=[hashlib.sha256(raw).hexdigest() for raw in configs],
        source_port_sha256=[hashlib.sha256(read(json.loads(raw)['source_port'])).hexdigest() for raw in configs])
    files[label+'_E.json']=(json.dumps(pair,indent=2)+'\n').encode()
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
    for label in ['D']:
        prefix=stage+'_'+label+'_E'
        cmd=['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'compare.py'),'--spec',str(root/(label+'_E.json')),'--stage',stage,'--output',str(root/(prefix+'.json'))]
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
    'plan.md':b'CPU-only D/E comparison after the original independent audit. Wait300s; terminal check near measured fit ETA. Bind actual exports, identical inputs and fixed fit order. Report initial and terminal differences; do not assume byte equality. Raw256 oracle is GT-only analysis. No model execution, gate edits or checkpoint writes.\n'})
spec=dict(candidate_root=candidate,candidate_audit=audit,audit_controller_pid=audit_pid,
    terminal_first_check_cst=(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))+datetime.timedelta(minutes=180)).isoformat(),poll_seconds=300,model_forwards=0,formal_rows=0,training_changes=False,
    files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
assert Path(root).name not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wx') as f:f.write(raw)
    assert read(root+'/'+name)==raw
cmd=['screen','-dmS','mcln_pvg_fixed_memory_comparison_v1','/root/miniconda3/envs/bdetr/bin/python','-u',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,cmd)),timeout=30);assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process,err.read().decode()
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,
    first_terminal_check_cst=spec['terminal_first_check_cst'],poll_seconds=300,formal_rows=0,model_forwards=0,training_changes=False)
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wx') as f:f.write(raw)
s.close();c.close();print(json.dumps(record),flush=True)
