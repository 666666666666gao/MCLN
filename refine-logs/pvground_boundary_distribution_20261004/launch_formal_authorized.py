"""Launch the stable-G pair only after its actual frozen-protocol sanity passes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).parent
assert not (local/'launch.json').exists()
review=json.loads((local/'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
assert review['execution_scope']=='SOURCE_ONLY'
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
engineering=json.loads((local/'complete_preflight/INTAKE.json').read_bytes())
proof=json.loads((local/'complete_preflight/distribution/preflight.json').read_bytes())
assert engineering['status']['status']=='complete' and engineering['controller_exit']==0
assert not engineering['controller_alive']
assert proof['status']=='pass' and proof['head_only'] and proof['original_g_state_unchanged']
assert proof['same_cached_inputs_zero_head_common_floor_exact'] and proof['optimizer_steps']==2
for arm in ('residual','distribution'):
    arm_proof=json.loads((local/'complete_preflight'/arm/'preflight.json').read_bytes())
    assert arm_proof['status']=='pass' and arm_proof['boundary_mode']==arm
    assert arm_proof['original_g_state_unchanged'] and arm_proof['optimizer_steps']==2
    assert arm_proof['same_cached_inputs_zero_head_common_floor_exact']
spec=json.loads((local/'distribution_spec.json').read_bytes())
root='/root/autodl-tmp/pvground_boundary_fit_20261004'
python=spec['runtime']+'/venv/bin/python'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:environment=json.loads(stream.read())
probe='''
import json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);preflight=Path('/root/autodl-tmp/pvground_boundary_preflight_20261004')
assert not root.exists()
assert json.loads((preflight/'status.json').read_bytes())['status']=='complete'
assert (preflight/'controller.exit').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
proof=json.loads((preflight/'distribution/preflight.json').read_bytes())
assert proof['status']=='pass' and proof['optimizer_steps']==2 and proof['head_only']
assert proof['original_g_state_unchanged'] and proof['same_cached_inputs_zero_head_common_floor_exact']
required=2*proof['serialization_bytes']+256*1024**2
free=shutil.disk_usage(root.parent).free
assert free>=required,(free,required)
print(json.dumps(dict(GPU_idle=True,actual_frozen_protocol_preflight_complete=True,free_bytes=free,required_bytes=required)))
'''
_,stdout,stderr=client.exec_command(shlex.join([python,'-c',probe,root]),timeout=60)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
resources=json.loads(raw);sftp.mkdir(root)
for name in ('run_boundary_fit.py','controller.py','EXPERIMENT_PLAN.md','EXPERIMENT_CODE_REVIEW.json'):
    content=(local/name).read_bytes()
    with sftp.open(root+'/'+name,'wx') as stream:stream.write(content)
    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==content
modules=['whole_model_preflight_checks.py', 'pvground_whole_mask_box_refiner.py', 'whole_mask_range.py', 'pvground_candidate_box_refiner.py', 'pvground_tail_support_box_refiner.py', 'pvground_tail_preflight.py', 'pvground_semantic_assignment.py', 'pvground_source_query.py', 'pvground_observation_query.py', 'pvground_task_observation_query.py', 'initial_range_comparison.py', 'pvground_boundary_box_refiner.py']
for arm in ('residual','distribution'):
    directory=root+'/'+arm;sftp.mkdir(directory)
    for name in modules:
        content=(local/name).read_bytes()
        with sftp.open(directory+'/'+name,'wx') as stream:stream.write(content)
        with sftp.open(directory+'/'+name,'rb') as stream:assert stream.read()==content
    content=(local/(arm+'_spec.json')).read_bytes()
    with sftp.open(directory+'/spec.json','wx') as stream:stream.write(content)
    with sftp.open(directory+'/spec.json','rb') as stream:assert stream.read()==content
command=shlex.join(['flock','-n',environment['resource_limits']['gpu_lock'],python,'-B','-u',root+'/controller.py'])
inner=command+' > '+shlex.quote(root+'/controller.log')+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(root+'/controller.exit')+'; exit "$code"'
screen='pvg_boundary_fit_20261004'
_,stdout,stderr=client.exec_command('screen -dmS '+screen+' bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
pattern='^'+python+' -B -u '+root+'/controller.py$'
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
process=stdout.read().decode().strip()
assert stdout.channel.recv_exit_status()==0 and len(process.splitlines())==1,stderr.read().decode()
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,process=process,screen=screen,
    resources=resources,order=['residual','distribution'],fresh_optimizer=True,base='original_g_5615_4495',
    seed=2027,batch_size=8,effective_batch_size=8,steps_per_arm=3723,fit_rows_per_arm=29778,formal_rows_per_arm=9508,
    head_only=True,original_g_parameters_and_running_state_fixed=True,formal_training_started=True,
    formal_result_available=False,estimate_hours_pair=[4.5,5.5],
    estimate_basis='completed frozen-G pair4h24m46s; distribution overhead unmeasured',
    poll_seconds=240,retention='metric-best plus protected parents and one small head-only active recovery; verified nonbest retired',
    teacher=False,quality_loss=False,boundary_distribution_and_DFL_in_second_arm=True,contrastive_expansion=False,p2=False,goal_achieved=False)
raw=(json.dumps(record,indent=2)+'\n').encode()
(local/'launch.json').write_bytes(raw)
with sftp.open(root+'/launch.json','wx') as stream:stream.write(raw)
sftp.close();client.close();print(json.dumps(record),flush=True)
