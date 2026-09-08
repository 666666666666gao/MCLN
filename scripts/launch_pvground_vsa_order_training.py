"""Start the fixed ScanRefer interface-correction trial and its independent CPU audit."""
import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path(__file__).resolve().parents[1]
train='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1'
audit='/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_vsaorder_v1'
oldtrain=repo/'refine-logs/pvground_scanrefer_finetune_20260908_detalign_v1'
oldaudit=repo/'refine-logs/pvground_scanrefer_endpoint_audit_20260908_detalign_v1'
archives={p:repo/'refine-logs'/Path(p).name.replace('mcln_','',1) for p in [train,audit]}
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
replay='/root/autodl-tmp/mcln_pvground_vsa_order_replay_20260908_v1'
with s.open(replay+'/controller.exit','rb') as f:assert f.read().strip()==b'0'
with s.open(replay+'/comparison.json','rb') as f:comparison=json.loads(f.read())
assert comparison['within_process_seed_repeat_exact']==[True,True]
assert comparison['cross_process_first_forward']['output.raw_bbs_query']['exact']
with s.open('/root/autodl-tmp/mcln_pvground_vsa_order_source_20260908_v1/preparation.json','rb') as f:prepared=json.loads(f.read())
assert prepared['changed_files']==['models/pv_utils.py'] and prepared['original_runtime_unchanged']
_,out,err=c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status()==0 and not err.read()
_,out,err=c.exec_command("df -B1 --output=avail /root/autodl-tmp",timeout=30)
free=int(out.read().decode().splitlines()[-1]);assert free>3*1024**3 and out.channel.recv_exit_status()==0 and not err.read()
spec=json.loads((oldtrain/'spec.json').read_bytes())
spec.update(root=train,model_source=prepared['model_source'],source_port=prepared['source_port'])
files={'train.py':(repo/'scripts/run_pvground_scanrefer_vsa_order.py').read_bytes(),
       'plan.md':(repo/'docs/PVG_VSA_BATCH_ORDER_2026-09-08.md').read_bytes()}
spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
files['controller.py']=(oldtrain/'controller.py').read_text().replace(json.loads((oldtrain/'spec.json').read_bytes())['root'],train).encode()

def install(root,files):
    assert Path(root).name not in s.listdir('/root/autodl-tmp')
    s.mkdir(root);archives[root].mkdir()
    for name,raw in files.items():
        if name.endswith('.py'):compile(raw,name,'exec')
        (archives[root]/name).write_bytes(raw)
        with s.open(root+'/'+name,'wb') as f:f.write(raw)
        with s.open(root+'/'+name,'rb') as f:assert f.read()==raw

def launch(root,screen):
    inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
    _,out,err=c.exec_command('screen -dmS '+screen+' bash -c '+shlex.quote(inner),timeout=30)
    assert out.channel.recv_exit_status()==0 and not err.read()
    _,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
    process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process and not err.read()
    record=dict(process=process,screen=screen,time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    raw=(json.dumps(record,indent=2)+'\n').encode();(archives[root]/'launch.json').write_bytes(raw)
    with s.open(root+'/launch.json','wb') as f:f.write(raw)
    return record

install(train,files)
for name in ['input_manifest.json','source_manifest.json']:
    (archives[train]/name).write_bytes((oldtrain/name).read_bytes())
(archives[train]/'model_source_preparation.json').write_text(json.dumps(prepared,indent=2)+'\n')
training_launch=launch(train,'mcln_pvg_scan_vsaorder_v1')
audit_spec=json.loads((oldaudit/'spec.json').read_bytes())
audit_spec.update(training_root=train,training_controller_pid=int(training_launch['process'].split()[0]),
    training_spec_sha256=hashlib.sha256(files['spec.json']).hexdigest(),
    first_check_cst=(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))+datetime.timedelta(minutes=15)).isoformat())
audit_files={name:(oldaudit/name).read_bytes() for name in ['audit.py','plan.md']}
oldroot='/root/autodl-tmp/'+ 'mcln_'+oldaudit.name
audit_files['audit_queue.py']=(oldaudit/'queue.py').read_text().replace(oldroot,audit).encode()
audit_files['controller.py']=(oldaudit/'controller.py').read_text().replace(oldroot,audit).replace("root/'queue.py'","root/'audit_queue.py'").encode()
assert b"root/'queue.py'" not in audit_files['controller.py']
audit_files['spec.json']=(json.dumps(audit_spec,indent=2)+'\n').encode()
install(audit,audit_files);audit_launch=launch(audit,'mcln_pvg_vsaorder_audit_v1')
print(json.dumps(dict(training=training_launch,audit=audit_launch,steps=3723,disk_free=free,formal_rows=0,
    expected_total_hours=3,remaining_cross_process_numeric_difference=True)))
s.close();c.close()
