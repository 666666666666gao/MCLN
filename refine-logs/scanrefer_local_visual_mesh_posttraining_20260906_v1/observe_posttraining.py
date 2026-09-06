import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1'
pair = repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_posttraining_20260906_v1'
training = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1'
formal = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_official_20260906_v1'
queue = json.loads((local / 'queue_launch.json').read_bytes())
pid = int(queue['worker'][0])
zone = datetime.timezone(datetime.timedelta(hours=8))
first = datetime.datetime.fromisoformat(queue['first_check_cst']) + datetime.timedelta(seconds=60)
schedule = {'time_cst': datetime.datetime.now(zone).isoformat(), 'local_observer_pid': os.getpid(),
    'remote_queue_pid': pid, 'first_check_cst': first.isoformat(), 'interval_seconds': 240,
    'stop_after': 'actual formal launch receipt or authoritative queue termination',
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
raw = (json.dumps(schedule, indent=2, sort_keys=True) + '\n').encode()
with (local / 'observation_schedule.json').open('xb') as stream:
    stream.write(raw)
print(raw.decode(), flush=True)
delay = (first - datetime.datetime.now(zone)).total_seconds()
if delay > 0:
    time.sleep(delay)
while True:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    sftp = client.open_sftp()
    names = sftp.listdir(remote)
    _, out, err = client.exec_command('ps -p ' + str(pid) + ' -o pid,stat,etime,args', timeout=30)
    process = out.read().decode()
    status = out.channel.recv_exit_status()
    assert status in (0, 1), err.read().decode()
    live = status == 0 and 'wait_for_training.py' in process
    with sftp.open(remote + '/queue.log', 'rb') as stream:
        stream.seek(max(0, stream.stat().st_size - 24000))
        tail = stream.read().decode()
    with sftp.open(training + '/run.log', 'rb') as stream:
        stream.seek(max(0, stream.stat().st_size - 64000))
        train_tail = stream.read().decode()
    progress = {}
    for line in train_tail.splitlines():
        for label in ['SCANREFER LOCAL VISUAL TRAIN ', 'SCANREFER LOCAL VISUAL EVAL ',
                      'SCANREFER LOCAL VISUAL EVAL COMPLETE ', 'SCANREFER LOCAL VISUAL PAIR COMPLETE ']:
            if line.startswith(label + '{'):
                progress[label.strip()] = json.loads(line[len(label):])
    downloaded = {}
    train_names = sftp.listdir(training)
    for name in ['controller.exit', 'receipt.json', 'fit_complete.json', 'terminal_metrics.json', 'independent_audit.json']:
        if name in train_names:
            with sftp.open(training + '/' + name, 'rb') as stream:
                raw = stream.read()
            (pair / name).write_bytes(raw)
            downloaded['training/' + name] = hashlib.sha256(raw).hexdigest()
    for name in ['executed.json', 'queue_controller.exit', 'wait_progress.jsonl']:
        if name in names:
            with sftp.open(remote + '/' + name, 'rb') as stream:
                raw = stream.read()
            (local / name).write_bytes(raw)
            downloaded[name] = hashlib.sha256(raw).hexdigest()
    if 'executed.json' in names:
        local_formal = repo / 'refine-logs/scanrefer_local_visual_mesh_official_20260906_v1'
        local_formal.mkdir()
        for name in ['launch.json', 'input_manifest.json', 'controller.sh', 'data_inputs.json']:
            with sftp.open(formal + '/' + name, 'rb') as stream:
                raw = stream.read()
            (local_formal / name).write_bytes(raw)
            downloaded['formal/' + name] = hashlib.sha256(raw).hexdigest()
    now = datetime.datetime.now(zone)
    result = {'time_cst': now.isoformat(), 'queue_process_live': live, 'queue_process': process,
        'training_progress': progress, 'formal_launch_receipt_present': 'executed.json' in names,
        'downloaded_sha256': downloaded, 'queue_log_tail': '\n'.join(tail.splitlines()[-12:])}
    stem = 'progress_' + now.strftime('%Y%m%d_%H%M%S')
    (local / (stem + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    (local / (stem + '_queue.txt')).write_text(tail, encoding='utf-8')
    (local / (stem + '_training.txt')).write_text(train_tail, encoding='utf-8')
    sftp.close()
    client.close()
    print(json.dumps({key: result[key] for key in ['time_cst', 'queue_process_live', 'formal_launch_receipt_present', 'downloaded_sha256']}), flush=True)
    if 'executed.json' in names or not live:
        break
    time.sleep(240)
