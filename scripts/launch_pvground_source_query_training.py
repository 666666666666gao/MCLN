"""Launch one fixed source-reading control only after its actual interface receipt passes."""
import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
train='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_sourcequery_v1'
audit='/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_sourcequery_v1'
interface='/root/autodl-tmp/mcln_pvground_source_query_interface_20260908_v3'
oldtrain=repo/'refine-logs/pvground_scanrefer_finetune_20260908_vsaorder_v1'
oldaudit=repo/'refine-logs/pvground_scanrefer_endpoint_audit_20260908_vsaorder_v1'
archives={p:repo/'refine-logs'/Path(p).name.replace('mcln_','',1) for p in [train,audit]}
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(interface+'/controller.exit','rb') as f:assert f.read().strip()==b'0'
with s.open(interface+'/results/receipt.json','rb') as f:checked=json.loads(f.read())
module=(repo/'models/pvground_source_query.py').read_bytes();module_sha=hashlib.sha256(module).hexdigest()
assert checked['status']=='pass' and checked['optimizer_steps']==2 and checked['source_query_read']
assert checked['module_sha256']==module_sha and checked['train_layer_replay']['output_exact']
assert checked['train_layer_replay']['rng_end_equal'] and checked['new_checkpoint_files']==0
model_source=checked['model_source'];source_port=model_source+'/../source_port.json'
with s.open(source_port,'rb') as f:port_raw=f.read();port=json.loads(port_raw)
assert hashlib.sha256(port_raw).hexdigest()==checked['source_port_sha256']
_,out,err=c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('df -B1 --output=avail /root/autodl-tmp',timeout=30)
free=int(out.read().decode().splitlines()[-1]);assert out.channel.recv_exit_status()==0,err.read().decode()
# Native delta ~331.2MB; this module adds 3 float32 optimizer/parameter copies.
# Two simultaneous deltas + 300MiB exports/logs + 1GiB reserve fit without deleting parents.
delta_estimate=331181662+12*checked['new_parameters']
artifacts=2*delta_estimate+300*1024**2
assert free>artifacts+1024**3,(free,artifacts)
spec=json.loads((oldtrain/'spec.json').read_bytes())
spec.update(root=train,model_source=model_source,source_port=source_port,
    training_interface_receipt=interface+'/results/receipt.json',source_query_read=True,
    source_query_module_sha256=module_sha,source_port_sha256=checked['source_port_sha256'],
    native_control_root=json.loads((oldtrain/'spec.json').read_bytes())['root'],
    disk_budget=dict(delta_estimate_bytes=delta_estimate,new_artifact_budget=artifacts,reserve_bytes=1024**3,free_before=free),
    comparison='own initial and completed same-budget native; primary bbs; added compute/parameters reported; no coverage claim')
files={'train.py':(repo/'scripts/run_pvground_scanrefer_source_query.py').read_bytes(),
       'pvground_source_query.py':module,'plan.md':(repo/'docs/PVG_SOURCE_QUERY_CONTROL_2026-09-08.md').read_bytes()}
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
training_launch=launch(train,'mcln_pvg_scan_sourcequery_v1')
audit_spec=json.loads((oldaudit/'spec.json').read_bytes())
audit_spec.update(training_root=train,training_controller_pid=int(training_launch['process'].split()[0]),
    training_spec_sha256=hashlib.sha256(files['spec.json']).hexdigest(),
    first_check_cst=(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))+datetime.timedelta(minutes=20)).isoformat())
audit_files={name:(oldaudit/name).read_bytes() for name in ['audit.py','plan.md']}
oldroot='/root/autodl-tmp/mcln_'+oldaudit.name
audit_files['audit_queue.py']=(oldaudit/'audit_queue.py').read_text().replace(oldroot,audit).encode()
audit_files['controller.py']=(oldaudit/'controller.py').read_text().replace(oldroot,audit).encode()
audit_files['spec.json']=(json.dumps(audit_spec,indent=2)+'\n').encode()
install(audit,audit_files);audit_launch=launch(audit,'mcln_pvg_sourcequery_audit_v1')
s.close();c.close();print(json.dumps(dict(training=training_launch,audit=audit_launch,steps=3723,disk_free=free,
    artifact_budget=artifacts,new_parameters=checked['new_parameters'],formal_rows=0)),flush=True)
