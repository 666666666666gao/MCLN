"""Launch the reviewed warm M0 once, without rebuilding the environment."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'preflight_launch.json').exists()
spec = json.loads((local / 'pair_spec.json').read_bytes())
review = json.loads((local / 'SOURCE_REVIEW.json').read_bytes())
assert review['review_type'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
assert hashlib.sha256((local/'pair_spec.json').read_bytes()).hexdigest() == review['source_seal']['pair_spec_sha256']
for name,digest in review['source_seal']['python_files'].items():
    assert hashlib.sha256((local/name).read_bytes()).hexdigest() == digest
launch_review=json.loads((local/'LAUNCH_SOURCE_REVIEW.json').read_bytes())
assert launch_review['verdict'] in ('PASS','WARN') and not launch_review['blocking_findings']
for item in launch_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    env = json.loads(stream.read())
assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
files = list(spec['new_runner_files']) + ['pair_spec.json', 'controller.py',
    'EXPERIMENT_PLAN.md', 'SOURCE_REVIEW.json', 'SOURCE_REVIEW.md', 'LAUNCH_SOURCE_REVIEW.json', 'LAUNCH_SOURCE_REVIEW.md']
code = r'''import datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root']);old=Path(b['previous_completed_experiment'])
assert root==Path('/root/autodl-tmp/pvground_selected_mask_training_20261009') and not root.exists()
assert root.parent.resolve()==Path('/root/autodl-tmp')
assert (old/'campaign.exit').read_text().strip()=='0'
assert json.loads((old/'campaign/receipt.json').read_bytes())['status']=='complete'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
capacity=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(capacity)==1
index,name,used,total=[value.strip() for value in capacity[0].split(',')]
assert int(index)==0 and 'A100' in name and 40000<=int(total)<=45000 and int(used)<500
assert shutil.disk_usage(root.parent).free>b['reserve']
for key in ('selected_terminal','base_terminal','warm_support_terminal'):
    assert hashlib.sha256(Path(b[key]).read_bytes()).hexdigest()==b[key+'_sha256']
root.mkdir()
for arm in b['support_modes']:(root/arm).mkdir()
print(json.dumps(dict(warm_runtime_reused=True,environment_rebuilt=False,gpu_idle=True,
    gpu_capacity=dict(index=int(index),name=name,used_mib=int(used),total_mib=int(total)),
    data_free_bytes=shutil.disk_usage(root).free,reserve=b['reserve'],
    previous_campaign_closed=True)))
'''
stdin, stdout, stderr = client.exec_command(shlex.join([spec['runtime'] + '/venv/bin/python', '-B', '-c', code]), timeout=120)
stdin.write(json.dumps(dict(spec, reserve=256*1024**2 + sum((local/name).stat().st_size for name in files))))
stdin.channel.shutdown_write()
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
(local / 'resource_check.json').write_bytes(raw)
for name in files:
    data = (local / name).read_bytes()
    with sftp.open(spec['root'] + '/' + name, 'wx') as stream:
        stream.write(data)
    with sftp.open(spec['root'] + '/' + name, 'rb') as stream:
        assert stream.read() == data
command = shlex.join(['flock', '-n', env['resource_limits']['gpu_lock'],
    spec['runtime'] + '/venv/bin/python', '-B', '-u', spec['root'] + '/controller.py', '--phase', 'preflight'])
shell = command + ' > ' + shlex.quote(spec['root'] + '/preflight_controller.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(spec['root'] + '/preflight_controller.exit') + '; exit "$code"'
screen = 'pvg_selected_mask_preflight_20261009'
_, stdout, stderr = client.exec_command(shlex.join(['screen', '-dmS', screen, 'bash', '-c', shell]), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^' + spec['runtime'] + '/venv/bin/python -B -u ' + spec['root'] + '/controller.py --phase preflight$'
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote(pattern), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
started = datetime.datetime.now().astimezone()
record = dict(status='WARM_COMPARISON_M0_STARTED_NOT_PASSED', time_cst=started.isoformat(),
    root=spec['root'], controller_pid=int(process.split()[0]), process=process, screen=screen,
    resources=resources, optimizer_steps_planned_per_arm=2, new_weight_files_expected=0,
    accuracy_result=False, first_check_seconds=300, estimated_seconds=480, later_poll_seconds=240,
    first_observation_cst=(started + datetime.timedelta(seconds=300)).isoformat(),
    estimate_basis='Actual prior warm pair M0 elapsed479.433s; estimate480s, first check3min before endpoint; later240s.')
(local / 'preflight_launch.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
with sftp.open(spec['root'] + '/preflight_launch.json', 'wx') as stream:
    stream.write((local / 'preflight_launch.json').read_bytes())
sftp.close()
client.close()
print(json.dumps(record), flush=True)
