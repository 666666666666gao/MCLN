"""Launch the source-reviewed no-update probe once in the unchanged warm runtime."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).resolve().parent
assert not (local/'launch.json').exists()
spec=json.loads((local/'pair_spec.json').read_bytes())
review=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and not review['blocking_findings'] and review['verdict'] in ('PASS','WARN')
for item in review['reviewed_files']:assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as f:env=json.loads(f.read())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
files=list(spec['new_runner_files'])+['pair_spec.json','panel.json','EXPERIMENT_PLAN.md','SOURCE_REVIEW.json','SOURCE_REVIEW.md']
code=r'''import hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root']);old=Path(b['original_completed_experiment'])
assert root==Path('/root/autodl-tmp/pvground_support_geometry_scale_20261008') and not root.exists()
assert root.parent.resolve()==Path('/root/autodl-tmp')
assert (old/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((old/'fit_status.json').read_bytes())['status']=='complete'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
capacity=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(capacity)==1
index,name,used,total=[value.strip() for value in capacity[0].split(',')]
assert int(index)==0 and 'A100' in name and 40000<=int(total)<=45000 and int(used)<500
assert shutil.disk_usage(root.parent).free>b['reserve']
assert hashlib.sha256(Path(b['selected_terminal']).read_bytes()).hexdigest()==b['selected_terminal_sha256']
root.mkdir()
print(json.dumps(dict(warm_runtime_reused=True,environment_rebuilt=False,gpu_idle=True,
    gpu_capacity=dict(index=int(index),name=name,used_mib=int(used),total_mib=int(total)),
    data_free_bytes=shutil.disk_usage(root).free,reserve=b['reserve'])))
'''
stdin,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code]),timeout=120)
stdin.write(json.dumps(dict(spec,reserve=64*1024**2+sum((local/name).stat().st_size for name in files))));stdin.channel.shutdown_write()
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
resources=json.loads(raw)
for name in files:
    data=(local/name).read_bytes()
    with sftp.open(spec['root']+'/'+name,'wx') as f:f.write(data)
    with sftp.open(spec['root']+'/'+name,'rb') as f:assert f.read()==data
variables=dict(env['env']);variables['PYTHONPATH']=spec['root']+':'+spec['helper_root']+':'+variables['PYTHONPATH']
command=shlex.join(['flock','-n',env['resource_limits']['gpu_lock'],'env',*[key+'='+value for key,value in variables.items()],
    spec['runtime']+'/venv/bin/python','-B','-u',spec['root']+'/run_mask_support_pair.py','--spec',spec['root']+'/pair_spec.json','--mode','preflight'])
shell=command+' > '+shlex.quote(spec['root']+'/diagnostic.log')+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(spec['root']+'/diagnostic.exit')+'; exit "$code"'
_,stdout,stderr=client.exec_command(shlex.join(['screen','-dmS','pvg_geometry_scale_20261008','bash','-c',shell]),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
pattern='^'+spec['runtime']+'/venv/bin/python -B -u '+spec['root']+'/run_mask_support_pair.py --spec '+spec['root']+'/pair_spec.json --mode preflight$'
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
process=stdout.read().decode().strip();assert stdout.channel.recv_exit_status()==0 and process,stderr.read().decode()
started=datetime.datetime.now().astimezone()
record=dict(status='SOURCE_REVIEWED_READONLY_NEURAL_DIAGNOSTIC_STARTED',time_cst=started.isoformat(),
    process=process,resources=resources,optimizer_updates_expected=0,formal_accuracy_result=False,rows=58,
    source_spec_sha256=hashlib.sha256((local/'pair_spec.json').read_bytes()).hexdigest(),
    first_observation_cst=(started+datetime.timedelta(seconds=420)).isoformat(),later_observation_interval_seconds=240,
    other_jobs_preempted=False,new_weights_expected=0)
(local/'launch.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
sftp.close();client.close();print(json.dumps(record),flush=True)
