"""Resume read-only observation after actual SSH banner EOF; no training restart."""
"""One read-only observer; estimate completion from actual full-loop progress."""
import datetime
import json
import os
from pathlib import Path
import shlex
import time
import paramiko

local = Path(__file__).parent
assert (local/'wait.json').exists()
launch = json.loads((local/'launch.json').read_bytes())
root = launch['root']
python = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
pid = int(launch['process'].split()[0])
first_delay = 240
observations = json.loads((local/'wait.json').read_bytes())['observations']
delay = first_delay
while True:
    planned = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=delay)
    record = dict(status='waiting', observations=observations, next_scheduled_cst=planned.isoformat(),
        next_delay_seconds=delay, original_estimate_hours=launch['estimate_hours_pair'])
    (local/'wait.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(waiting_until=planned.isoformat(), seconds=delay)), flush=True)
    time.sleep(delay)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    code = '''
import datetime,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);pid=int(sys.argv[2]);status=json.loads((root/'status.json').read_bytes())
stage=status.get('stage','');log=root/(stage.replace('/','/')+'.log')
progress=[]
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-24000));lines=stream.read().decode().splitlines()
    for line in lines:
        for prefix in ('PVG_TRAIN_PROGRESS ','PVG_EVAL_PROGRESS ','PVG_EVAL_COMPLETE '):
            if line.startswith(prefix):progress.append(dict(kind=prefix.strip(),value=json.loads(line[len(prefix):])))
print(json.dumps(dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    status=status,controller_alive=Path('/proc/%d'%pid).exists(),
    controller_exit=(root/'controller.exit').read_text().strip() if (root/'controller.exit').exists() else None,
    directory_free_bytes=shutil.disk_usage(root).free,progress=progress[-4:],
    owned_weights=[dict(path=str(p),bytes=p.stat().st_size) for arm in ('local_range','whole_range')
        for p in (root/arm).glob('*.pth')])) )
'''
    _, stdout, stderr = client.exec_command(shlex.join([python, '-c', code, root, str(pid)]), timeout=60)
    raw = stdout.read()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    observation = json.loads(raw)
    client.close()
    index = len(observations)+1
    name = 'observation_%02d.json' % index
    (local/name).write_bytes(raw)
    observations.append(dict(file=name, time_cst=observation['time_cst'], stage=observation['status'].get('stage')))
    print(json.dumps(observation), flush=True)
    status = observation['status']['status']
    if status in ('complete', 'failed'):
        if observation['controller_alive'] or observation['controller_exit'] is None:
            delay = 240
            continue
        (local/'wait.json').write_text(json.dumps(dict(status=status, observations=observations,
            finished_observation_cst=observation['time_cst'], controller_exit=observation['controller_exit']), indent=2)+'\n', encoding='utf-8')
        break
    assert observation['controller_alive'], 'controller absent without terminal status'
    delay = 240
    for progress in reversed(observation['progress']):
        value = progress['value']
        if progress['kind'] == 'PVG_TRAIN_PROGRESS' and value['step'] >= 64:
            remaining = (value['total_steps']-value['step'])*value['cumulative_seconds']/value['step']
            delay = max(240, int(remaining-180))
            break
        if progress['kind'] == 'PVG_EVAL_PROGRESS' and value['rows'] >= 512:
            remaining = (value['total']-value['rows'])*value['seconds']/value['rows']
            delay = max(240, int(remaining-120))
            break
