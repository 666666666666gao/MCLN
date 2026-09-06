import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import paramiko

zone = datetime.timezone(datetime.timedelta(hours=8))
first_check = datetime.datetime(2026, 9, 6, 22, 54, tzinfo=zone)
local = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1')
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
_, out, err = client.exec_command('ps -eo pid,ppid,comm,args', timeout=30)
lines = out.read().decode().splitlines()[1:]
assert out.channel.recv_exit_status() == 0, err.read().decode()
processes = {int(parts[0]): parts for parts in (line.split(None, 3) for line in lines) if len(parts) == 4}
matches = [p for p in processes.values() if p[2] == 'python' and remote + '/input_manifest.json' in p[3]
           and processes[int(p[1])][2] == 'flock']
assert len(matches) == 1, matches
pid = int(matches[0][0])
sftp = client.open_sftp()
with sftp.open(remote + '/run.log', 'rb') as stream:
    stream.seek(max(0, stream.stat().st_size - 16000))
    tail = stream.read().decode()
_, out, err = client.exec_command('nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits', timeout=30)
gpu = out.read().decode().strip()
assert out.channel.recv_exit_status() == 0, err.read().decode()
startup = {'time_cst': datetime.datetime.now(zone).isoformat(), 'process_live': True,
    'remote_training_pid': pid, 'process': matches[0], 'gpu_memory_mib_and_utilization_percent': gpu,
    'log_tail': '\n'.join(tail.splitlines()[-6:]), 'optimizer_updates_not_yet_observed': True}
schedule = {'first_check_cst': first_check.isoformat(), 'interval_seconds_after_first_check': 240,
    'stop_after': 'first report at >=64 updates per arm, or authoritative process termination',
    'remote_training_pid': pid, 'local_observer_pid': os.getpid(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'previous_baseline_seconds': 1790.9076552391052, 'current_preflight_input_startup_minutes': 7,
    'eta_scope': 'input parsing plus6887 baseline rows; terminal ETA will use measured training speed'}
for name, raw in {'startup.json': (json.dumps(startup, indent=2) + '\n').encode(),
                  'first_update_observation_schedule.json': (json.dumps(schedule, indent=2) + '\n').encode(),
                  'observe_first_updates.py': Path(__file__).read_bytes()}.items():
    (local / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
sftp.close()
client.close()
print(json.dumps({'startup': startup, 'schedule': schedule}), flush=True)
delay = (first_check - datetime.datetime.now(zone)).total_seconds()
if delay > 0:
    time.sleep(delay)
while True:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    sftp = client.open_sftp()
    names = sftp.listdir(remote)
    with sftp.open(remote + '/run.log', 'rb') as stream:
        stream.seek(max(0, stream.stat().st_size - 64000))
        tail = stream.read().decode()
    _, out, err = client.exec_command('ps -p ' + str(pid) + ' -o pid,stat,etime,args', timeout=30)
    process = out.read().decode()
    code = out.channel.recv_exit_status()
    assert code in (0, 1), err.read().decode()
    live = code == 0 and remote + '/input_manifest.json' in process
    progress = {}
    for line in tail.splitlines():
        for label in ['SCANREFER LOCAL VISUAL TRAIN ', 'SCANREFER LOCAL VISUAL EVAL ', 'SCANREFER LOCAL VISUAL EVAL COMPLETE ']:
            if line.startswith(label):
                progress[label.strip()] = json.loads(line[len(label):])
    downloaded = {}
    for name in ['controller.exit', 'receipt.json', 'fit_complete.json', 'baseline_metrics.json']:
        if name in names:
            with sftp.open(remote + '/' + name, 'rb') as stream:
                raw = stream.read()
            (local / name).write_bytes(raw)
            downloaded[name] = hashlib.sha256(raw).hexdigest()
    now = datetime.datetime.now(zone)
    result = {'time_cst': now.isoformat(), 'process_live': live, 'process': process,
        'progress': progress, 'downloaded_sha256': downloaded}
    stem = 'progress_' + now.strftime('%Y%m%d_%H%M%S')
    (local / (stem + '.json')).write_text(json.dumps(result, indent=2))
    (local / (stem + '.log')).write_text(tail)
    sftp.close()
    client.close()
    print(json.dumps(result), flush=True)
    if progress.get('SCANREFER LOCAL VISUAL TRAIN', {}).get('step', 0) >= 64 or not live:
        break
    time.sleep(240)
