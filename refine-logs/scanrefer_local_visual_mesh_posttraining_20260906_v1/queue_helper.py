import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1'
pair = repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_posttraining_20260906_v1'
training = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1'
baseline_raw = (pair / 'baseline_verification.json').read_bytes()
baseline = json.loads(baseline_raw)
assert baseline['status'] == 'pass' and baseline['observed_steps_per_arm'] >= 64
launch = json.loads((pair / 'launch.json').read_bytes())
screen_pid = int(launch['screen_session'][0].split('.')[0])
waiter = (local / 'wait_for_training.py').read_bytes()
plan = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'training_directory': training, 'training_screen_pid': screen_pid,
    'first_check_cst': baseline['next_check_cst'], 'interval_seconds': 240,
    'baseline_verification_sha256': hashlib.sha256(baseline_raw).hexdigest(),
    'post_training_sha256': hashlib.sha256((local / 'post_training.py').read_bytes()).hexdigest(),
    'waiter_sha256': hashlib.sha256(waiter).hexdigest(),
    'scope': 'Wait for training screen termination and successful controller;then independent CPU endpoint audit and one fixed9508-row formal launch. No Nr/Sr launch.'}
controller = '''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd ''' + shlex.quote(remote) + '''
/root/miniconda3/envs/bdetr/bin/python -u wait_for_training.py
status=$?
printf '%s\\n' "$status" > queue_controller.exit
exit "$status"
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(training + '/baseline_verification.json', 'rb') as stream:
    assert stream.read() == baseline_raw
for name in ['post_training.py', 'preparation.json', 'controller_check.json']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        assert stream.read() == (local / name).read_bytes()
_, out, err = client.exec_command('ps -p ' + str(screen_pid) + ' -o pid=,args=', timeout=30)
process = out.read().decode()
assert out.channel.recv_exit_status() == 0 and training in process, err.read().decode()
uploads = {'wait_for_training.py': waiter, 'queue_plan.json': (json.dumps(plan, indent=2, sort_keys=True) + '\n').encode(),
    'wait_controller.sh': controller.encode(), 'queue_helper.py': Path(__file__).read_bytes()}
for name, raw in uploads.items():
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    (local / name).write_bytes(raw)
_, out, err = client.exec_command('bash -n ' + shlex.quote(remote + '/wait_controller.sh'), timeout=30)
out.read()
assert out.channel.recv_exit_status() == 0, err.read().decode()
screen = 'mcln_scanrefer_mesh_posttraining_v1'
command = 'screen -dmS ' + screen + ' bash -lc ' + shlex.quote('cd ' + shlex.quote(remote) + ' && bash wait_controller.sh > queue.log 2>&1')
_, out, err = client.exec_command(command, timeout=30)
out.read()
assert out.channel.recv_exit_status() == 0, err.read().decode()
_, out, err = client.exec_command('screen -ls', timeout=30)
sessions = out.read().decode()
assert out.channel.recv_exit_status() == 0, err.read().decode()
matching = [line.strip() for line in sessions.splitlines() if screen in line]
assert len(matching) == 1
_, out, err = client.exec_command('ps -eo pid,ppid,comm,args', timeout=30)
rows = [line.split(None, 3) for line in out.read().decode().splitlines()[1:]]
assert out.channel.recv_exit_status() == 0, err.read().decode()
workers = [row for row in rows if len(row) == 4 and row[2] == 'python' and row[3].endswith(' -u wait_for_training.py')]
assert len(workers) == 1
with sftp.open(remote + '/wait_started.json', 'rb') as stream:
    started_raw = stream.read()
(local / 'wait_started.json').write_bytes(started_raw)
started = json.loads(started_raw)
assert started['pid'] == int(workers[0][0])
receipt = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'screen_session': matching, 'worker': workers[0], 'command': command,
    'queue_plan_sha256': hashlib.sha256(uploads['queue_plan.json']).hexdigest(),
    'wait_controller_sha256': hashlib.sha256(controller.encode()).hexdigest(),
    'first_check_cst': plan['first_check_cst'], 'interval_seconds': 240,
    'status': 'verified live waiting worker;actual endpoint audit and formal launch still pending',
    'gpu_forwards': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0}
raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
with sftp.open(remote + '/queue_launch.json', 'wx') as stream:
    stream.write(raw)
(local / 'queue_launch.json').write_bytes(raw)
sftp.close()
client.close()
print(raw.decode())
