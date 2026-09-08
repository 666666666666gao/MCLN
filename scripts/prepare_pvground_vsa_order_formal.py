import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
old=repo/'refine-logs/pvground_scanrefer_formal_20260908_detalign_v1'
root='/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260908_vsaorder_v1'
training='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1'
audit='/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_vsaorder_v1'
archive=repo/'refine-logs/pvground_scanrefer_formal_20260908_vsaorder_v1'
manifest=json.loads((repo/'refine-logs/pvground_scanrefer_finetune_20260908_vsaorder_v1/input_manifest.json').read_bytes())
spec=json.loads((old/'spec.json').read_bytes())
spec.update(training_root=training,training_audit_root=audit,training_audit_controller_pid=int(json.loads((repo/'refine-logs/pvground_scanrefer_endpoint_audit_20260908_vsaorder_v1/launch.json').read_bytes())['process'].split()[0]),first_check_cst=(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))+datetime.timedelta(minutes=165)).isoformat(),training_spec_sha256=hashlib.sha256((repo/'refine-logs/pvground_scanrefer_finetune_20260908_vsaorder_v1/spec.json').read_bytes()).hexdigest())
files={name:(old/name).read_bytes() for name in ['audit.py','plan.md']}
files['evaluate.py']=(repo/'scripts/evaluate_pvground_scanrefer_vsa_order.py').read_bytes()
contract=json.loads((old/'formal_input_contract.json').read_bytes())
contract.update(dataset_source=manifest['model_source'],source_manifest_sha256=manifest['source_manifest_sha256'],source_adaptation='retains aligned detection augmentation; model VSA ordering corrected via training spec model_source/source_port',prior_data_contract_sha256=hashlib.sha256((old/'formal_input_contract.json').read_bytes()).hexdigest())
files['formal_input_contract.json']=(json.dumps(contract,indent=2)+'\n').encode()
for name in ['queue.py','controller.py','cpu_probe.py']:
    text=(old/name).read_text().replace('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260908_detalign_v1',root)
    if name=='cpu_probe.py':
        text=text.replace("directory=Path(spec['training_root'])/'initial'", "directory=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_v1/initial')")
    if name=='controller.py':text=text.replace("root/'queue.py'","root/'formal_queue.py'")
    files['formal_queue.py' if name=='queue.py' else name]=text.encode()
spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();archive.mkdir();s.mkdir(root)
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wx') as f:f.write(raw)
command='/root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/cpu_probe.py')+' > '+shlex.quote(root+'/cpu_probe.log')+' 2>&1'
_,out,err=c.exec_command(command,timeout=60)
exit_code=out.channel.recv_exit_status()
with s.open(root+'/cpu_probe.exit','wx') as f:f.write(str(exit_code)+'\n')
for name in ['cpu_probe.log','cpu_probe.exit','preparation_receipt.json']:
    s.get(root+'/'+name,str(archive/name))
assert exit_code==0
inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_,out,err=c.exec_command('screen -dmS mcln_pvg_vsaorder_formal_v1 bash -c '+shlex.quote(inner),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,screen='mcln_pvg_vsaorder_formal_v1',first_check_cst=spec['first_check_cst'],poll_seconds=300,formal_rows=0,cpu_probe_source='previous completed6887 initial; unchanged evaluator/recount code',training_changes=False)
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wx') as f:f.write(raw)
(archive/'prepare_from_local.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(record));s.close();c.close()
