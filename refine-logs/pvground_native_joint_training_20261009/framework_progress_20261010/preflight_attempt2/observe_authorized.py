"""One owner: first check at the admitted estimate, then240-second intervals."""
import datetime
import json
import os
from pathlib import Path
import shlex
import time

import paramiko

root = Path(__file__).resolve().parent
launch = json.loads((root / 'NATIVE_PREFLIGHT_LAUNCH.json').read_bytes())
assert launch['status'] == 'NATIVE_ENGINEERING_PREFLIGHT_STARTED_NOT_COMPLETED'
assert not (root / 'NATIVE_PREFLIGHT_OBSERVER_STARTED.json').exists()
assert not (root / 'NATIVE_PREFLIGHT_WAIT.json').exists()
first = datetime.datetime.fromisoformat(launch['first_observation_cst'])
assert launch['later_poll_seconds'] == 240
(root / 'NATIVE_PREFLIGHT_OBSERVER_STARTED.json').write_text(json.dumps(dict(
    observer_local_pid=os.getpid(), controller_pid=launch['controller_pid'],
    first_observation_cst=first.isoformat(), later_poll_seconds=240,
    remote_queries_performed=0), indent=2) + '\n')
time.sleep(max(0, first.timestamp() - time.time()))
code = r'''import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]);proc=Path('/proc')/sys.argv[2];expected=json.loads(sys.argv[3])
alive=proc.exists()
argv=[item.decode() for item in (proc/'cmdline').read_bytes().split(bytes([0]))[:-1]] if alive else []
assert not alive or argv==expected
status=json.loads((root/'preflight_status.json').read_bytes()) if (root/'preflight_status.json').exists() else None
print(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
 controller_alive=alive,status=status,exitcode=int((root/'controller.exit').read_text())
 if (root/'controller.exit').exists() else None)))
'''
directory = root / 'native_preflight_observations'
assert not directory.exists()
directory.mkdir()
index = 0
while True:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
        password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    _, stdout, stderr = client.exec_command(shlex.join([
        '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python', '-B', '-c', code,
        launch['root'], str(launch['controller_pid']), json.dumps(launch['controller_argv'])]), timeout=30)
    raw, error = stdout.read(), stderr.read()
    exit_code = stdout.channel.recv_exit_status()
    client.close()
    index += 1
    (directory / ('observation_%03d.stdout.json' % index)).write_bytes(raw)
    (directory / ('observation_%03d.stderr.txt' % index)).write_bytes(error)
    (directory / ('observation_%03d.exit.json' % index)).write_text(json.dumps(dict(exit_code=exit_code)) + '\n')
    assert exit_code == 0, error.decode()
    record = json.loads(raw)
    print('NATIVE_M0_OBSERVATION ' + json.dumps(record), flush=True)
    if record['exitcode'] is not None:
        assert not record['controller_alive']
        receipt = dict(time_cst=record['time_cst'], observer_closed=True, terminal=record,
            observation_count=index, new_nn_steps=0, normal_epoch_training_started=False)
        (root / 'NATIVE_PREFLIGHT_WAIT.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('NATIVE_M0_CLOSED ' + json.dumps(receipt), flush=True)
        break
    assert record['controller_alive']
    (root / 'NATIVE_PREFLIGHT_OBSERVER_WAIT.json').write_text(json.dumps(dict(
        remote_queries_performed=index, observed_status=record['status'],
        next_observation_cst=(datetime.datetime.now().astimezone()+datetime.timedelta(seconds=240)).isoformat()), indent=2) + '\n')
    time.sleep(240)
