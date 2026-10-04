"""One read-only observer, scheduled near measured phase ends, then at least240s."""
import datetime
import json
import os
from pathlib import Path
import shlex
import time
import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'wait.json').exists()
launch = json.loads((local / 'launch.json').read_bytes())
spec = json.loads((local / 'distribution_spec.json').read_bytes())
root = launch['root']
pid = int(launch['process'].split()[0])
observations = []
delay = 6300
code = '''
import datetime,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);pid=int(sys.argv[2])
status=json.loads((root/'status.json').read_bytes())
stage=status['stage'];log=root/(stage+'.log')
with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-30000));lines=stream.read().decode().splitlines()
progress=[]
for line in lines:
    for prefix in ('PVG_TRAIN_PROGRESS ','PVG_EVAL_PROGRESS ','PVG_EVAL_COMPLETE '):
        if line.startswith(prefix):progress.append(dict(kind=prefix.strip(),value=json.loads(line[len(prefix):])))
print(json.dumps(dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    status=status,controller_alive=Path('/proc/%d'%pid).exists(),
    controller_exit=(root/'controller.exit').read_text().strip() if (root/'controller.exit').exists() else None,
    directory_free_bytes=shutil.disk_usage(root).free,progress=progress[-4:],
    log_tail=chr(10).join(lines[-5:]),
    owned_weights=[dict(path=str(p),bytes=p.stat().st_size) for arm in ('residual','distribution')
        for p in (root/arm).glob('*.pth')])))
'''
while True:
    planned = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=delay)
    record = dict(status='waiting', observations=observations, next_scheduled_cst=planned.isoformat(),
        next_delay_seconds=delay, original_estimate_hours=launch['estimate_hours_pair'],
        first_check_basis='actual prior frozen original-G train phase6495s; first check6300s')
    (local / 'wait.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(waiting_until=planned.isoformat(), seconds=delay)), flush=True)
    time.sleep(delay)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
        password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    _, stdout, stderr = client.exec_command(shlex.join([
        spec['runtime'] + '/venv/bin/python', '-c', code, root, str(pid)]), timeout=60)
    raw = stdout.read()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    observation = json.loads(raw)
    client.close()
    name = 'observation_%02d.json' % (len(observations) + 1)
    assert not (local / name).exists()
    (local / name).write_bytes(raw)
    observations.append(dict(file=name, time_cst=observation['time_cst'], stage=observation['status']['stage']))
    latest = observation['progress'][-1] if observation['progress'] else None
    print(json.dumps(dict(file=name, time_cst=observation['time_cst'], status=observation['status'],
        controller_alive=observation['controller_alive'], controller_exit=observation['controller_exit'],
        free_bytes=observation['directory_free_bytes'], latest_progress=latest,
        owned_weights=observation['owned_weights'])), flush=True)
    status = observation['status']['status']
    if status in ('complete', 'failed'):
        if observation['controller_alive'] or observation['controller_exit'] is None:
            delay = 240
            continue
        (local / 'wait.json').write_text(json.dumps(dict(status=status, observations=observations,
            finished_observation_cst=observation['time_cst'], controller_exit=observation['controller_exit']),
            indent=2) + '\n', encoding='utf-8')
        raise SystemExit(int(observation['controller_exit']))
    assert observation['controller_alive'], 'controller absent without terminal status'
    delay = 240
    if latest is not None and latest['kind'] == 'PVG_TRAIN_PROGRESS' and latest['value']['step'] >= 64:
        value = latest['value']
        remaining = (value['total_steps'] - value['step']) * value['cumulative_seconds'] / value['step']
        delay = max(240, int(remaining - 180))
    elif latest is not None and latest['kind'] == 'PVG_EVAL_PROGRESS' and latest['value']['rows'] >= 512:
        value = latest['value']
        remaining = (value['total'] - value['rows']) * value['seconds'] / value['rows']
        delay = max(240, int(remaining - 120))
