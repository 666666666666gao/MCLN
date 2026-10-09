"""Observe one original phase at its estimated endpoint, then every240 seconds."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import time

import paramiko


runner = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=('preflight', 'fit'), required=True)
phase = parser.parse_args().phase
assert not (runner / (phase + '_observer_started.json')).exists()
assert not (runner / (phase + '_wait.json')).exists()
launch = json.loads((runner / (phase + '_launch.json')).read_bytes())
spec = json.loads((runner / 'pair_spec.json').read_bytes())
first = datetime.datetime.fromisoformat(launch['first_observation_cst'])
wait = dict(observer_local_pid=os.getpid(), controller_pid=launch['controller_pid'],
    phase=phase, first_observation_cst=first.isoformat(), remote_queries_performed=0,
    estimated_seconds=launch['estimated_seconds'], later_poll_seconds=240)
(runner / (phase + '_observer_started.json')).write_text(json.dumps(wait, indent=2) + '\n')
time.sleep(max(0, first.timestamp() - time.time()))
code = '''import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]);pid=sys.argv[2];phase=sys.argv[3];proc=Path('/proc')/pid;alive=proc.exists()
command=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode() if alive else ''
assert not alive or str(root)+'/span_controller.py --phase '+phase in command
status=json.loads((root/(phase+'_status.json')).read_bytes())
print(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),controller_alive=alive,
    status=status,exitcode=int((root/(phase+'_controller.exit')).read_text())
    if (root/(phase+'_controller.exit')).exists() else None)))
'''
directory = runner / (phase + '_observations')
directory.mkdir()
index = 0
while True:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
                   password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    _, stdout, stderr = client.exec_command(shlex.join([spec['runtime'] + '/venv/bin/python',
        '-B', '-c', code, launch['root'], str(launch['controller_pid']), phase]), timeout=30)
    raw, error = stdout.read(), stderr.read()
    exit_code = stdout.channel.recv_exit_status()
    client.close()
    index += 1
    (directory / ('observation_%03d.stdout.json' % index)).write_bytes(raw)
    (directory / ('observation_%03d.stderr.txt' % index)).write_bytes(error)
    (directory / ('observation_%03d.exit' % index)).write_text(str(exit_code) + '\n')
    assert exit_code == 0, error.decode()
    record = json.loads(raw)
    print('SPAN_PHASE_OBSERVATION ' + json.dumps(record), flush=True)
    if record['exitcode'] is not None:
        assert not record['controller_alive']
        receipt = dict(time_cst=record['time_cst'], observer_closed=True, terminal=record,
            phase=phase, observation_count=index, new_nn_steps=0, collector_not_started=True)
        (runner / (phase + '_wait.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        print('SPAN_PHASE_CLOSED ' + json.dumps(receipt), flush=True)
        break
    assert record['controller_alive']
    wait.update(remote_queries_performed=index,
        next_observation_cst=(datetime.datetime.now().astimezone() + datetime.timedelta(seconds=240)).isoformat(),
        observed_mode=record['status']['mode'],
        elapsed_seconds=time.time() - datetime.datetime.fromisoformat(launch['time_cst']).timestamp())
    (runner / (phase + '_observer_wait.json')).write_text(json.dumps(wait, indent=2) + '\n')
    time.sleep(240)
