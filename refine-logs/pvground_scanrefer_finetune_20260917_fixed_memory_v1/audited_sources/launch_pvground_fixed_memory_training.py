"""Launch one fixed source-reading control only after its actual interface receipt passes."""
import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
train='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_fixed_memory_v1'
audit='/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260917_fixed_memory_v1'
interface='/root/autodl-tmp/mcln_pvground_task_observation_interface_20260917_v1'
oldtrain=repo/'refine-logs/pvground_scanrefer_finetune_20260917_task_observation_v1'
oldaudit=repo/'refine-logs/pvground_scanrefer_endpoint_audit_20260917_task_observation_v1'
archives={p:repo/'refine-logs'/Path(p).name.replace('mcln_','',1) for p in [train,audit]}
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
for dependency in ['finetune','endpoint_audit','formal']:
    with s.open('/root/autodl-tmp/mcln_pvground_scanrefer_'+dependency+'_20260917_task_observation_v1/controller.exit','rb') as f:assert f.read().strip()==b'0'
with s.open(interface+'/controller.exit','rb') as f:assert f.read().strip()==b'0'
with s.open(interface+'/results/receipt.json','rb') as f:checked=json.loads(f.read())
module=(repo/'models/pvground_observation_query.py').read_bytes();module_sha=hashlib.sha256(module).hexdigest()
assert checked['observation_state'] and checked['task_read'] and checked['new_parameters']==923616
task_module=(repo/'models/pvground_task_observation_query.py').read_bytes()
task_sha=hashlib.sha256(task_module).hexdigest()
assert checked['status']=='pass' and checked['optimizer_steps']==2 and checked['source_query_read']
assert checked['module_sha256']==task_sha and checked['strict_cpu_restore']
assert checked['direct_routing_verified'] and checked['new_checkpoint_files']==0
model_source='/root/autodl-tmp/mcln_pvground_task_observation_source_20260917_v1/PV-Ground';source_port=model_source+'/../source_port.json'
with s.open(source_port,'rb') as f:port_raw=f.read();port=json.loads(port_raw)
assert hashlib.sha256(port_raw).hexdigest()==checked['source_port_sha256']
_,out,err=c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('df -B1 --output=avail /root/autodl-tmp',timeout=30)
free=int(out.read().decode().splitlines()[-1]);assert out.channel.recv_exit_status()==0,err.read().decode()
# Bound each delta by actual D terminal; two files, 256MiB exports/logs and 768MiB reserve.
delta_estimate=342299567
artifacts=2*delta_estimate+256*1024**2
assert free>artifacts+768*1024**2,(free,artifacts)
spec=json.loads((oldtrain/'spec.json').read_bytes())
spec.update(fixed_visual_memory=True,root=train,model_source=model_source,source_port=source_port,
    training_interface_receipt=interface+'/results/receipt.json',source_query_read=True,
    observation_state=True,observation_module_sha256=module_sha,task_read=True,task_module_sha256=task_sha,
    source_query_module_sha256=hashlib.sha256((repo/'models/pvground_source_query.py').read_bytes()).hexdigest(),source_port_sha256=checked['source_port_sha256'],
    native_control_root=json.loads((oldtrain/'spec.json').read_bytes())['native_control_root'],
    comparison_control_root=json.loads((oldtrain/'spec.json').read_bytes())['root'],
    disk_budget=dict(delta_estimate_bytes=delta_estimate,new_artifact_budget=artifacts,reserve_bytes=768*1024**2,free_before=free),
    comparison='own initial and same-budget D; fixed complete pretrained backbone parameters and running state; bbs primary')
files={'train.py':(repo/'scripts/run_pvground_scanrefer_fixed_memory.py').read_bytes(),
       'pvground_task_observation_query.py':task_module,'pvground_observation_query.py':module,'pvground_source_query.py':(repo/'models/pvground_source_query.py').read_bytes(),'plan.md':(repo/'docs/PVG_FIXED_VISUAL_MEMORY_CONTROL_2026-09-17.md').read_bytes()}
spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
files['controller.py']=(oldtrain/'controller.py').read_text().replace(json.loads((oldtrain/'spec.json').read_bytes())['root'],train).encode()

def install(root,files):
    assert Path(root).name not in s.listdir('/root/autodl-tmp');s.mkdir(root);archives[root].mkdir()
    for name,raw in files.items():
        if name.endswith('.py'):compile(raw,name,'exec')
        (archives[root]/name).write_bytes(raw)
        with s.open(root+'/'+name,'wb') as f:f.write(raw)
        with s.open(root+'/'+name,'rb') as f:assert f.read()==raw

def launch(root,screen):
    inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
    _,out,err=c.exec_command('screen -dmS '+screen+' bash -c '+shlex.quote(inner),timeout=30)
    assert out.channel.recv_exit_status()==0,err.read().decode()
    _,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
    process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process,err.read().decode()
    record=dict(process=process,screen=screen,time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    raw=(json.dumps(record,indent=2)+'\n').encode();(archives[root]/'launch.json').write_bytes(raw)
    with s.open(root+'/launch.json','wb') as f:f.write(raw)
    return record

install(train,files)
for name in ['input_manifest.json','source_manifest.json']:(archives[train]/name).write_bytes((oldtrain/name).read_bytes())
(archives[train]/'source_port.json').write_bytes(port_raw)
training_launch=launch(train,'mcln_pvg_scan_fixed_memory_v1')
audit_spec=json.loads((oldaudit/'spec.json').read_bytes())
audit_spec.update(training_root=train,training_controller_pid=int(training_launch['process'].split()[0]),
    training_spec_sha256=hashlib.sha256(files['spec.json']).hexdigest(),
    first_check_cst=(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))+datetime.timedelta(minutes=20)).isoformat())
audit_files={'audit.py':(repo/'scripts/audit_pvground_fixed_memory.py').read_bytes(),'plan.md':(repo/'docs/PVG_FIXED_VISUAL_MEMORY_CONTROL_2026-09-17.md').read_bytes()}
oldroot='/root/autodl-tmp/mcln_'+oldaudit.name
audit_files['audit_queue.py']=(oldaudit/'audit_queue.py').read_text().replace(oldroot,audit).encode()
audit_files['controller.py']=(oldaudit/'controller.py').read_text().replace(oldroot,audit).encode()
queue=audit_files['audit_queue.py'].decode().replace('import datetime,hashlib,importlib.util,json,subprocess,time','import datetime,hashlib,importlib.util,json,os,subprocess,time')
queue=queue.replace("command=['/root/miniconda3/envs/bdetr/bin/python'", "command=[str(Path(input_spec['runtime'])/'venv/bin/python')")
queue=queue.replace('result=subprocess.run(command)', "environment=json.loads((Path(input_spec['runtime'])/'env_spec.json').read_bytes())['env']\n        result=subprocess.run(command,env=dict(os.environ,**dict(environment,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')))")
audit_files['audit_queue.py']=queue.encode()
anchor="        raise SystemExit(result.returncode)"
assert queue.count(anchor)==1
cleanup="""        if result.returncode==0:
            verified=json.loads((root/'audit.json').read_bytes())
            assert verified['integrity_pass'] and verified['fixed_visual_memory_verified']
            receipt=json.loads((training/'receipt.json').read_bytes())
            assert verified['receipt_sha256']==module.sha(training/'receipt.json')
            assert receipt['terminal_sha256']==module.sha(training/'terminal.pth')
            latest=training/'latest.pth'
            assert latest.resolve().parent==training.resolve() and latest.is_file()
            record={'path':str(latest),'sha256':module.sha(latest),'bytes':latest.stat().st_size,
                    'retained_terminal_sha256':receipt['terminal_sha256'],
                    'reason':'superseded snapshot after independent fixed endpoint audit'}
            (root/'cleanup_planned.json').write_text(json.dumps(record)+'\\n')
            latest.unlink()
            record['deleted']=not latest.exists()
            (root/'cleanup_receipt.json').write_text(json.dumps(record)+'\\n')
"""
queue=queue.replace(anchor,cleanup+anchor)
audit_files['audit_queue.py']=queue.encode()
audit_spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in audit_files.items() if name!='controller.py'}
audit_files['spec.json']=(json.dumps(audit_spec,indent=2)+'\n').encode()
install(audit,audit_files);audit_launch=launch(audit,'mcln_pvg_fixed_memory_audit_v1')
s.close();c.close();print(json.dumps(dict(training=training_launch,audit=audit_launch,steps=3723,disk_free=free,
    artifact_budget=artifacts,new_parameters=checked['new_parameters'],formal_rows=0)),flush=True)
