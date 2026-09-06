import datetime
import json
import os
from pathlib import Path
import re
import shlex
import time

import paramiko

directory = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_joint_readout_pair_20260906_v1')
remote = '/root/autodl-tmp/mcln_scanrefer_joint_readout_pair_20260906_v1'
launch = json.loads((directory / 'launch.json').read_bytes())
zone = datetime.timezone(datetime.timedelta(hours=8))
first = datetime.datetime(2026, 9, 6, 11, 42, tzinfo=zone)
print(json.dumps({'first_observation_cst': first.isoformat(), 'subsequent_interval_seconds': 240,
                  'stop_after': 'first logged optimizer step >=64 or controller exit'}), flush=True)
time.sleep(max(0, first.timestamp() - time.time()))
while True:
    c = paramiko.SSHClient()
    c.load_system_host_keys()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    s = c.open_sftp()
    now = datetime.datetime.now(zone)
    names = s.listdir(remote)
    with s.open(remote + '/run.log') as stream:
        log = stream.read().decode()
    lines = log.splitlines()
    training = [json.loads(line.split('SCANREFER JOINT TRAIN ', 1)[1])
                for line in lines if line.startswith('SCANREFER JOINT TRAIN ')]
    evaluation = [json.loads(line.split('SCANREFER JOINT EVAL ', 1)[1])
                  for line in lines if line.startswith('SCANREFER JOINT EVAL ')]
    _, out, err = c.exec_command('pgrep -af ' + shlex.quote('^' + re.escape(launch['native_command']) + '$'), timeout=30)
    process = out.read().decode().strip()
    process_exit = out.channel.recv_exit_status()
    err.read()
    result = {'time_cst': now.isoformat(), 'process': process, 'process_check_exit': process_exit,
              'latest_training': training[-1] if training else None,
              'latest_evaluation': evaluation[-1] if evaluation else None,
              'log_tail': lines[-5:], 'files': names}
    for name in ['protocol.json', 'baseline_metrics.json', 'baseline_rows.json', 'fit_complete.json',
                 'receipt.json', 'terminal_metrics.json', 'terminal_rows.json', 'controller.exit']:
        if name in names and not (directory / name).exists():
            s.get(remote + '/' + name, str(directory / name))
    finished = 'controller.exit' in names
    if finished:
        result['controller_exit'] = int((directory / 'controller.exit').read_text().strip())
    s.close()
    c.close()
    (directory / ('progress_' + now.strftime('%Y%m%d_%H%M%S') + '.json')).write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (directory / ('run_' + now.strftime('%Y%m%d_%H%M%S') + '.log')).write_text(log, encoding='utf-8')
    print(json.dumps(result), flush=True)
    if finished or (training and training[-1]['step'] >= 64):
        break
    assert process_exit == 0 and process
    time.sleep(240)
