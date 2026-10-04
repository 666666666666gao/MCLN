"""Start one full arm only after its actual source and GPU preflight gates."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parent
assert not (local/'fit_launch.json').exists()
review = json.loads((local/'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
wait = json.loads((local/'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and wait['terminal']['exitcode'] == 0
assert not wait['terminal']['controller_alive'] and wait['terminal']['status']['status'] == 'complete'
proof = json.loads((local/'preflight_complete/preflight.json').read_bytes())
assert proof['status'] == 'pass' and proof['optimizer_steps'] == 2 and proof['batch_size'] == 8
assert proof['isolated_quality_route_verified'] and proof['geometry_provider_and_g_states_exact']
assert proof['initial_zero_output_native_exact'] and proof['quality_weight'] == 1.0
assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
spec = json.loads((local/'quality_fit_spec.json').read_bytes())
root = '/root/autodl-tmp/pvground_final_quality_20261005'
assert spec['root'] == root+'/quality'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
    password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
with sftp.open(root+'/preflight/preflight.json','rb') as stream:
    assert stream.read() == (local/'preflight_complete/preflight.json').read_bytes()
with sftp.open(root+'/quality/spec.json','rb') as stream:
    assert stream.read() == (local/'quality_fit_spec.json').read_bytes()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    environment = json.loads(stream.read())
python = spec['runtime']+'/venv/bin/python'
reserve = 2*proof['serialization_bytes'] + 256*1024**2
probe = '''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);reserve=int(sys.argv[2]);spec=json.loads(sys.argv[3])
assert json.loads((root/'preflight_status.json').read_bytes())['status']=='complete'
assert (root/'preflight_controller.exit').read_text().strip()=='0'
assert not (root/'fit_status.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for name,digest in spec['runner_files'].items():
    assert hashlib.sha256((root/'quality'/name).read_bytes()).hexdigest()==digest
assert shutil.disk_usage(root).free >= reserve
print(json.dumps(dict(GPU_idle=True,preflight_controller_closed=True,required_reserve_bytes=reserve,
    data_free_bytes=shutil.disk_usage(root).free,system_free_bytes=shutil.disk_usage('/').free)))
'''
_,stdout,stderr = client.exec_command(shlex.join([python,'-c',probe,root,str(reserve),json.dumps(spec)]),timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
(local/'fit_resource_check.json').write_bytes(raw)
command = shlex.join(['flock','-n',environment['resource_limits']['gpu_lock'],python,'-B','-u',root+'/controller.py','--stage','fit'])
inner = command+' > '+shlex.quote(root+'/fit_controller.log')+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(root+'/fit_controller.exit')+'; exit "$code"'
screen = 'pvg_final_quality_fit_20261005'
_,stdout,stderr = client.exec_command(shlex.join(['screen','-dmS',screen,'bash','-c',inner]),timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^'+python+' -B -u '+root+'/controller.py --stage fit$'
_,stdout,stderr = client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,process=process,screen=screen,
    resources=resources,status='LAUNCHED_NOT_COMPLETED',formal_training_started=True,accuracy_result=False,
    fit_rows=29778,batch_size=8,effective_batch=8,accumulation_steps=1,tail_batch=2,updates=3723,
    primary_mode='bbs',primary_threshold=.5,quality_weight=1.0,control_reused=spec['control_root'],
    first_check_seconds=6200,later_poll_seconds=240,estimated_seconds=8200,
    estimate_basis='actual visible control fit+6887 holdout6529.66s and formal1491.62s; new loss only, no larger model')
(local/'fit_launch.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with sftp.open(root+'/fit_launch.json','wx') as stream:
    stream.write((local/'fit_launch.json').read_bytes())
sftp.close()
client.close()
print(json.dumps(record),flush=True)
