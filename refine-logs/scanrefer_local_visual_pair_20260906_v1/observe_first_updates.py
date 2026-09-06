import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import paramiko

zone = datetime.timezone(datetime.timedelta(hours=8))
first_check = datetime.datetime(2026, 9, 6, 17, 20, tzinfo=zone)
local = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_local_visual_pair_20260906_v1')
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_pair_20260906_v1'
schedule = {'first_check_cst': first_check.isoformat(), 'interval_seconds_after_first_check': 240,
    'stop_after': 'first report at >=64 updates per arm, or authoritative process termination',
    'remote_training_pid': 36968, 'local_observer_pid': os.getpid(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'eta_scope': 'initial data parsing plus6887 baseline rows; estimate terminal after measured training speed'}
with (local / 'first_update_observation_schedule.json').open('x') as stream:
    json.dump(schedule, stream, indent=2)
print(json.dumps(schedule), flush=True)
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
        tail = stream.read().decode('utf-8')
    _, stdout, stderr = client.exec_command('ps -p 36968 -o pid,stat,etime,args', timeout=30)
    process = stdout.read().decode()
    process_status = stdout.channel.recv_exit_status()
    assert process_status in (0, 1), stderr.read().decode()
    live = process_status == 0 and remote + '/input_manifest.json' in process
    progress = {}
    for line in tail.splitlines():
        for label in ['SCANREFER LOCAL VISUAL TRAIN ', 'SCANREFER LOCAL VISUAL EVAL ',
                      'SCANREFER LOCAL VISUAL EVAL COMPLETE ']:
            if line.startswith(label):
                progress[label.strip()] = json.loads(line[len(label):])
    terminal_files = {}
    for name in ['controller.exit', 'receipt.json', 'fit_complete.json', 'baseline_metrics.json']:
        if name in names:
            with sftp.open(remote + '/' + name, 'rb') as stream:
                raw = stream.read()
            (local / name).write_bytes(raw)
            terminal_files[name] = hashlib.sha256(raw).hexdigest()
    sftp.close()
    client.close()
    now = datetime.datetime.now(zone)
    result = {'time_cst': now.isoformat(), 'process_live': live, 'process': process,
              'progress': progress, 'terminal_files': terminal_files}
    name = 'progress_' + now.strftime('%Y%m%d_%H%M%S')
    (local / (name + '.json')).write_text(json.dumps(result, indent=2))
    (local / (name + '.log')).write_text(tail)
    print(json.dumps(result), flush=True)
    step = progress.get('SCANREFER LOCAL VISUAL TRAIN', {}).get('step', 0)
    if step >= 64 or not live:
        break
    time.sleep(240)
