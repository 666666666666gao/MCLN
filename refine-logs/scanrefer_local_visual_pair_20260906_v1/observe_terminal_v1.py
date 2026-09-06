import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import paramiko


parser = argparse.ArgumentParser()
parser.add_argument('--first-check-cst', required=True)
option = parser.parse_args()
zone = datetime.timezone(datetime.timedelta(hours=8))
first_check = datetime.datetime.fromisoformat(option.first_check_cst)
assert first_check.utcoffset() == datetime.timedelta(hours=8)
local = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_local_visual_pair_20260906_v1')
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_pair_20260906_v1'
schedule = {'first_check_cst': first_check.isoformat(), 'interval_seconds_after_first_check': 240,
    'stop_after': 'completed receipt and controller exit, or authoritative process termination',
    'remote_training_pid': 36968, 'local_observer_pid': os.getpid(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'eta_scope': 'terminal evaluation endpoint; first check selected from measured training speed'}
with (local / 'terminal_observation_schedule_v1.json').open('x') as stream:
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
    _, stdout, stderr = client.exec_command('df -B1 /root/autodl-tmp', timeout=30)
    disk = stdout.read().decode()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    progress = {}
    for line in tail.splitlines():
        for label in ['SCANREFER LOCAL VISUAL TRAIN ', 'SCANREFER LOCAL VISUAL EVAL ',
                      'SCANREFER LOCAL VISUAL EVAL COMPLETE ']:
            if line.startswith(label + '{'):
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
              'progress': progress, 'terminal_files': terminal_files, 'disk_bytes_report': disk}
    name = 'terminal_progress_' + now.strftime('%Y%m%d_%H%M%S')
    (local / (name + '.json')).write_text(json.dumps(result, indent=2))
    (local / (name + '.log')).write_text(tail)
    print(json.dumps(result), flush=True)
    if not live or ('receipt.json' in terminal_files and 'controller.exit' in terminal_files):
        break
    time.sleep(240)
