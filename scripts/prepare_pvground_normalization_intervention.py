"""Launch the fixed 128-training-scene native score diagnostic."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_normalization_intervention_20260917_v1'
archive = repo/'refine-logs/pvground_normalization_intervention_20260917_v1'
training = '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_task_observation_v1'
fixtures = '/root/autodl-tmp/mcln_pvground_training_interface_20260908_v1/fixtures'
c = paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s = c.open_sftp()

def read(path):
    with s.open(path,'rb') as stream:
        return stream.read()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()


for dependency in ['finetune','endpoint_audit','formal']:
    assert read('/root/autodl-tmp/mcln_pvground_scanrefer_'+dependency+'_20260917_task_observation_v1/controller.exit').strip() == b'0'
assert read('/root/autodl-tmp/mcln_pvground_task_observation_comparison_20260917_v1/controller.exit').strip() == b'0'
reference='/root/autodl-tmp/mcln_pvground_native_score_diagnostic_20260917_v2'
assert read(reference+'/controller.exit').strip()==b'0'
train_spec_raw = read(training+'/spec.json')
train_spec = json.loads(train_spec_raw)
receipt = json.loads(read(training+'/receipt.json'))
assert receipt['status'] == 'complete' and receipt['training_steps'] == 3723
assert receipt['terminal_sha256'] == 'ce03188965491a82bcb1c5a6d26f590d3a243a01985457f220d5503c75b2fcf5'
assert sha(train_spec_raw) == receipt['spec_sha256']
runtime = train_spec['runtime']
env = json.loads(read(runtime+'/env_spec.json'))
assert sha(json.dumps(env,sort_keys=True,separators=(',',':')).encode()) == train_spec['env_spec_sha256']
_,out,err = c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status() == 0,err.read().decode()
_,out,err = c.exec_command('df -B1 --output=avail /root/autodl-tmp',timeout=30)
free = int(out.read().decode().splitlines()[-1])
assert out.channel.recv_exit_status() == 0 and free > 1024**3+32*1024**2,err.read().decode()
files = {'diagnose.py':(repo/'scripts/diagnose_pvground_normalization_intervention.py').read_bytes(),
         'evaluate.py':(repo/'scripts/evaluate_pvground_scanrefer_task_observation.py').read_bytes(),
         'fixture_receipt.json':read(fixtures+'/receipt.json'),
         'plan.md':(repo/'docs/PVG_NORMALIZATION_INTERVENTION_PLAN_2026-09-17.md').read_bytes()}
for new_name,old_name in [('reference_diagnostic.json','diagnostic.json'),('reference_rows.json','rows.json'),('reference_selection.json','input_selection.json')]:
    files[new_name]=read(reference+'/'+old_name)
for name,key in [('pvground_source_query.py','source_query_module_sha256'),
                 ('pvground_observation_query.py','observation_module_sha256'),
                 ('pvground_task_observation_query.py','task_module_sha256')]:
    files[name] = (repo/'models'/name).read_bytes()
    assert sha(files[name]) == train_spec[key]
spec = dict(training_root=training,training_spec_sha256=sha(train_spec_raw),terminal_sha256=receipt['terminal_sha256'],
    fixtures=fixtures,fixture_receipt_sha256=sha(files['fixture_receipt.json']),seed=2027,
    selection='first expression of first 128 distinct fit physical scenes',
    training_rows=128,batch_size=8,model_forwards=48,optimizer_steps=0,formal_rows=0,
    scope='fixed D terminal BatchNorm running-state intervention; three eval arms; no optimizer or formal evaluation',
    disk_free_before=free,max_new_artifact_budget=32*1024**2,
    files={name:sha(raw) for name,raw in files.items()})
files['census.json'] = (repo/'refine-logs/pvground_normalization_census_20260917_v1/census.json').read_bytes()
spec['files']['census.json'] = sha(files['census.json'])
files['spec.json'] = (json.dumps(spec,indent=2)+'\n').encode()
argv = ['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',runtime+'/venv/bin/python','-u',root+'/diagnose.py','--spec',root+'/spec.json']
controller = ('import os,subprocess\nfrom pathlib import Path\nroot=Path(__file__).parent\n'
    '(root/"controller.pid").write_text(str(os.getpid())+"\\n")\n'
    'env=dict(os.environ,**'+repr(dict(env['env'],CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1'))+')\n'
    'with (root/"run.log").open("xb") as log:\n'
    '    result=subprocess.run('+repr(argv)+',env=env,stdout=log,stderr=subprocess.STDOUT)\n'
    '(root/"controller.exit").write_text(str(result.returncode)+"\\n")\n'
    'raise SystemExit(result.returncode)\n')
files['controller.py'] = controller.encode()
assert Path(root).name not in s.listdir('/root/autodl-tmp') and not archive.exists()
s.mkdir(root); archive.mkdir()
for name,raw in files.items():
    if name.endswith('.py'):
        compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as stream:
        stream.write(raw)
    assert read(root+'/'+name) == raw
command = 'screen -dmS mcln_pvg_normalization_intervention_v1 '+shlex.quote('/root/miniconda3/envs/bdetr/bin/python')+' -u '+shlex.quote(root+'/controller.py')
_,out,err = c.exec_command(command,timeout=30)
assert out.channel.recv_exit_status() == 0,err.read().decode()
_,out,err = c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process = out.read().decode().strip()
assert out.channel.recv_exit_status() == 0 and process,err.read().decode()
record = dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    root=root,process=process,expected_seconds=300,first_check_after_seconds=300,
    training_steps=0,formal_rows=0,disk_free_before=free)
raw = (json.dumps(record,indent=2)+'\n').encode()
(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wb') as stream:
    stream.write(raw)
s.close(); c.close()
print(json.dumps(record),flush=True)
