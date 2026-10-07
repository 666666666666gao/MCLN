"""One whole-job estimated-endpoint observer; closed-task receipt only."""
import datetime
import json
import os
from pathlib import Path
import shlex
import time

import paramiko


local = Path(__file__).resolve().parent
assert not (local / 'fit_observer_started.json').exists()
assert not (local / 'fit_wait.json').exists()
launch = json.loads((local / 'fit_launch.json').read_bytes())
spec = json.loads((local / 'pair_spec.json').read_bytes())
first = datetime.datetime.fromisoformat(launch['time_cst']) + datetime.timedelta(seconds=launch['first_check_seconds'])
wait = dict(observer_local_pid=os.getpid(), controller_pid=launch['controller_pid'],
    first_observation_cst=first.isoformat(), remote_queries_performed=0,
    estimated_seconds=launch['estimated_seconds'], later_poll_seconds=240)
(local / 'fit_observer_started.json').write_text(json.dumps(wait, indent=2) + '\n', encoding='utf-8')
time.sleep(max(0, first.timestamp() - time.time()))
code = '''import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]);proc=Path('/proc')/sys.argv[2];alive=proc.exists()
command=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode() if alive else ''
assert not alive or str(root)+'/controller.py' in command
status=json.loads((root/'fit_status.json').read_bytes())
print(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),controller_alive=alive,status=status,
    exitcode=int((root/'fit_controller.exit').read_text()) if (root/'fit_controller.exit').exists() else None)))
'''
directory = local / 'fit_observations'
directory.mkdir()
index = 0
while True:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro',port=33476,username='root',
                   password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
    _, stdout, stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python',
        '-B','-c',code,launch['root'],str(launch['controller_pid'])]),timeout=30)
    raw = stdout.read()
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    record = json.loads(raw)
    client.close()
    index += 1
    (directory / ('observation_%03d.json' % index)).write_bytes(raw)
    print('PAIRED_SUPPORT_FIT_OBSERVATION ' + json.dumps(record), flush=True)
    if record['exitcode'] is not None:
        assert not record['controller_alive']
        receipt = dict(time_cst=record['time_cst'],observer_closed=True,terminal=record,
            observation_count=index,new_nn_steps=0,collector_not_started=True)
        (local/'fit_wait.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        print('PAIRED_SUPPORT_FIT_CLOSED ' + json.dumps(receipt), flush=True)
        break
    assert record['controller_alive']
    elapsed = time.time() - datetime.datetime.fromisoformat(launch['time_cst']).timestamp()
    wait.update(remote_queries_performed=index,
        next_observation_cst=(datetime.datetime.now().astimezone()+datetime.timedelta(seconds=240)).isoformat(),
        observed_mode=record['status']['mode'], elapsed_seconds=elapsed,
        estimate_exceeded_twice=elapsed>2*launch['estimated_seconds'])
    (local/'fit_observer_wait.json').write_text(json.dumps(wait,indent=2)+'\n',encoding='utf-8')
    if wait['estimate_exceeded_twice']:
        raise RuntimeError('Actual job still live beyond twice estimate; retain original process and observation, no relaunch')
    time.sleep(240)
