import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
train='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_detalign_v1'
audit='/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_detalign_v1'
source_root='/root/autodl-tmp/mcln_scanrefer_detection_aligned_source_20260908_v1'
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
archives={r:repo/('refine-logs/'+Path(r).name.replace('mcln_','',1)) for r in [train,audit]}
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
prepare='''import hashlib,json,shutil,subprocess
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=Path(SOURCE)
check=Path('/root/autodl-tmp/mcln_scanrefer_detection_augmentation_20260908_v1')
r=json.loads((check/'receipt.json').read_bytes())
assert r['status']=='pass' and r['rows']==32 and r['fixed_max_box_error']<3e-5
assert (check/'controller.exit').read_text().strip()=='0'
base=Path('/root/autodl-tmp/mcln_scanrefer_object_appearance_native_20260908_v1/model_source')
old=json.loads((base/'appearance_source_manifest.json').read_bytes())
assert len(old['files'])==625
dst=root/'model_source'
dst.mkdir(parents=True)
for name,digest in old['files'].items():
    assert sha(base/name)==digest,name
    target=dst/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(base/name,target)
shutil.copyfile(check/'fixed_dataset.py',dst/'src/joint_det_dataset.py')
new=dict(old)
new['parent_source']=str(base)
new['parent_manifest_sha256']=sha(base/'appearance_source_manifest.json')
new['files']={name:sha(dst/name) for name in old['files']}
changed=[name for name in old['files'] if new['files'][name]!=old['files'][name]]
assert changed==['src/joint_det_dataset.py']
assert new['files'][changed[0]]==r['fixed_dataset_sha256']
new['single_change']='detected corners flip before rotation to match points'
(dst/'appearance_source_manifest.json').write_text(json.dumps(new,indent=2)+'\\n')
m=json.loads(Path('/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/input_manifest.json').read_bytes())
m['model_source']=str(dst)
m['source_manifest_sha256']=sha(dst/'appearance_source_manifest.json')
(root/'input_manifest.json').write_text(json.dumps(m,indent=2)+'\\n')
env=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip())
assert gpu<500
previous=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_v1')
prev=json.loads((previous/'receipt.json').read_bytes())
assert (previous/'controller.exit').read_text().strip()=='0'
assert sha(previous/'terminal.pth')==prev['terminal_sha256']
latest=previous/'latest.pth'
assert latest.resolve().parent==previous.resolve() and latest.is_file()
cleanup=dict(path=str(latest),bytes=latest.stat().st_size,sha256=sha(latest),reason='completed failed run; fixed terminal and parent retained')
latest.unlink()
assert sha(previous/'terminal.pth')==prev['terminal_sha256']
free=shutil.disk_usage('/root/autodl-tmp').free
assert free>3*1024**3
result=dict(status='pass',source_files=625,changed_files=changed,source_manifest_sha256=m['source_manifest_sha256'],dataset_sha256=r['fixed_dataset_sha256'],disk_free=free,gpu_mib=gpu,cleanup=cleanup)
(root/'preparation_receipt.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
'''.replace('SOURCE',repr(source_root))
_,out,err=c.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(prepare),timeout=60)
raw=out.read();error=err.read();assert out.channel.recv_exit_status()==0,error.decode()
prepared=json.loads(raw)
for directory,archive in archives.items():archive.mkdir();s.mkdir(directory)
training_archive=archives[train]
for name in ['input_manifest.json','preparation_receipt.json']:
    s.get(source_root+'/'+name,str(training_archive/name))
s.get(source_root+'/model_source/appearance_source_manifest.json',str(training_archive/'source_manifest.json'))
(training_archive/'prepare_source.py').write_bytes(prepare.encode())
old_dir=repo/'refine-logs/pvground_scanrefer_finetune_20260908_v1'
spec=json.loads((old_dir/'spec.json').read_bytes())
spec['root']=train;spec['input_manifest']=source_root+'/input_manifest.json'
files={'train.py':(old_dir/'train.py').read_bytes(),'plan.md':(repo/'docs/SCANREFER_DETECTION_AUGMENTATION_ORDER_2026-09-08.md').read_bytes()}
spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
files['controller.py']=(old_dir/'controller.py').read_text().replace('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_v1',train).encode()

def install(directory,archive,files):
    for name,raw in files.items():
        assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw
        if name.endswith('.py'):compile(raw,name,'exec')
        (archive/name).write_bytes(raw)
        with s.open(directory+'/'+name,'wx') as f:f.write(raw)
        with s.open(directory+'/'+name,'rb') as f:assert f.read()==raw


def launch(directory,screen):
    inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(directory+'/controller.py')+' > '+shlex.quote(directory+'/run.log')+' 2>&1'
    _,out,err=c.exec_command('screen -dmS '+screen+' bash -c '+shlex.quote(inner),timeout=30)
    assert out.channel.recv_exit_status()==0,err.read().decode()
    _,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+directory+'/controller.py$'),timeout=30)
    process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process,err.read().decode()
    record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,screen=screen)
    raw=(json.dumps(record,indent=2)+'\n').encode()
    (archives[directory]/'launch.json').write_bytes(raw)
    with s.open(directory+'/launch.json','wx') as f:f.write(raw)
    return record

install(train,training_archive,files)
training_launch=launch(train,'mcln_pvg_scan_detalign_v1')
old_audit=repo/'refine-logs/pvground_scanrefer_endpoint_audit_20260908_v1'
audit_spec=json.loads((old_audit/'spec.json').read_bytes())
audit_spec.update(training_root=train,training_controller_pid=int(training_launch['process'].split()[0]),training_spec_sha256=hashlib.sha256(files['spec.json']).hexdigest(),first_check_cst=(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))+datetime.timedelta(minutes=15)).isoformat())
audit_files={name:(old_audit/name).read_bytes() for name in ['audit.py','plan.md']}
for name in ['queue.py','controller.py']:
    audit_files[name]=(old_audit/name).read_text().replace('/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_v1',audit).encode()
audit_files['spec.json']=(json.dumps(audit_spec,indent=2)+'\n').encode()
install(audit,archives[audit],audit_files)
audit_launch=launch(audit,'mcln_pvg_detalign_audit_v1')
(training_archive/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(preparation=prepared,training=training_launch,audit=audit_launch,expected_training_steps=3723,estimated_total_hours=2.8,formal_queue='not yet prepared; mandatory gate unchanged')))
s.close();c.close()
