"""Deploy the reviewed one-batch stable-G probe into the existing warm environment."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).parent
assert not (local/'preflight_launch.json').exists()
review=json.loads((local/'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
assert review['execution_scope']=='SOURCE_ONLY'
for item in review['reviewed_files']:
    assert __import__('hashlib').sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
prior_local=local.parent/'pvground_boundary_distribution_20261004'
intake=json.loads((prior_local/'complete/INTAKE.json').read_bytes())
audit=json.loads((prior_local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert intake['status']['status']=='complete' and intake['controller_exit']==0 and not intake['controller_alive']
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_issues']
spec=json.loads((local/'face_preflight_spec.json').read_bytes())
root='/root/autodl-tmp/pvground_face_preflight_20261004'
assert spec['root']==root+'/face_conditioned' and spec['head_only'] and spec['use_whole_range']
prior=json.loads((prior_local/'launch.json').read_bytes())
controller_pid=prior['process'].split()[0]
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    environment=json.loads(stream.read())
python=spec['runtime']+'/venv/bin/python'
probe='''
import json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);prior=Path(sys.argv[2]);pid=int(sys.argv[3])
best=Path(sys.argv[4]);best_sha=sys.argv[5]
import hashlib
assert hashlib.sha256(best.read_bytes()).hexdigest()==best_sha
assert not root.exists()
assert not Path('/proc/%d'%pid).exists(),'previous source-pair controller still live'
status=json.loads((prior/'status.json').read_bytes())
assert status['status']=='complete' and (prior/'controller.exit').read_text().strip()=='0'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu,'GPU already has a compute process'
free=shutil.disk_usage(root.parent).free
assert free>=64*1024**2,'insufficient preflight log space'
print(json.dumps(dict(original_pair_terminal=True,GPU_idle=True,directory_free_bytes=free,system_free_bytes=shutil.disk_usage("/").free,current_best_sha256=best_sha,
    log_reserve_bytes=64*1024**2,weight_reserve_bytes=0)))
'''
_,stdout,stderr=client.exec_command(shlex.join([python,'-c',probe,root,prior['root'],controller_pid,spec['current_best']['path'],spec['current_best']['sha256']]),timeout=60)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
resources=json.loads(raw)
(local/'preflight_resource_check.json').write_bytes(raw)
sftp.mkdir(root)
for name in ('run_face_fit.py','preflight_controller.py','EXPERIMENT_PLAN.md','EXPERIMENT_CODE_REVIEW.json'):
    content=(local/name).read_bytes()
    with sftp.open(root+'/'+name,'wx') as stream:stream.write(content)
    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==content
modules=['whole_model_preflight_checks.py', 'pvground_whole_mask_box_refiner.py', 'whole_mask_range.py', 'pvground_candidate_box_refiner.py', 'pvground_tail_support_box_refiner.py', 'pvground_tail_preflight.py', 'pvground_semantic_assignment.py', 'pvground_source_query.py', 'pvground_observation_query.py', 'pvground_task_observation_query.py', 'initial_range_comparison.py', 'pvground_boundary_box_refiner.py', 'pvground_face_conditioned_box_refiner.py']
for arm in ('face_conditioned',):
    directory=root+'/'+arm;sftp.mkdir(directory)
    for name in modules:
        content=(local/name).read_bytes()
        with sftp.open(directory+'/'+name,'wx') as stream:stream.write(content)
        with sftp.open(directory+'/'+name,'rb') as stream:assert stream.read()==content
    content=(local/('face_preflight_spec.json')).read_bytes()
    with sftp.open(directory+'/spec.json','wx') as stream:stream.write(content)
    with sftp.open(directory+'/spec.json','rb') as stream:assert stream.read()==content
command=shlex.join(['flock','-n',environment['resource_limits']['gpu_lock'],python,'-B','-u',root+'/preflight_controller.py'])
inner=command+' > '+shlex.quote(root+'/controller.log')+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(root+'/controller.exit')+'; exit "$code"'
screen='pvg_face_preflight_20261004'
_,stdout,stderr=client.exec_command('screen -dmS '+screen+' bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
pattern='^'+python+' -B -u '+root+'/preflight_controller.py$'
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
process=stdout.read().decode().strip()
assert stdout.channel.recv_exit_status()==0 and len(process.splitlines())==1,stderr.read().decode()
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,process=process,screen=screen,
    resources=resources,environment_reused=True,order=['face_conditioned'],batch_size=8,optimizer_steps_per_arm=2,
    head_only=True,formal_training_started=False,weights_created=0,weights_deleted=0,
    execution_status='LAUNCHED_NOT_COMPLETED',estimate_seconds=540,first_check_seconds=420,later_poll_seconds=240,
    estimate_basis='one probe; completed flat distribution about450seconds; face-attention overhead unmeasured')
raw=(json.dumps(record,indent=2)+'\n').encode()
(local/'preflight_launch.json').write_bytes(raw)
with sftp.open(root+'/launch.json','wx') as stream:stream.write(raw)
sftp.close();client.close()
print(json.dumps(record),flush=True)
