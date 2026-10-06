"""Read only after the five-minute diagnostic estimate, then at 240-second intervals."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'observer_wait.json').exists()
launch = json.loads((local / 'launch.json').read_bytes())
spec = json.loads((local / 'spec.json').read_bytes())
first = datetime.datetime.fromisoformat(launch['time_cst']) + datetime.timedelta(minutes=5)
wait = dict(observer_local_pid=os.getpid(), controller_pid=launch['controller_pid'],
            first_observation_cst=first.isoformat(), repeat_interval_seconds=240,
            remote_queries_performed=0)
(local / 'observer_wait.json').write_text(json.dumps(wait, indent=2) + '\n', encoding='utf-8')
time.sleep(max(0, first.timestamp() - time.time()))
code = '''import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]);pid=int(sys.argv[2]);proc=Path('/proc')/str(pid)
alive=proc.exists();cmd=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode() if alive else ''
assert not alive or str(root) in cmd
exit_path=root/'controller.exit'
print(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
 controller_alive=alive,controller_exit=int(exit_path.read_text()) if exit_path.exists() else None)))
'''
count = 0
while True:
    client = paramiko.SSHClient(); client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
                   password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    _, stdout, stderr = client.exec_command(shlex.join([
        spec['runtime'] + '/venv/bin/python', '-B', '-c', code,
        launch['root'], str(launch['controller_pid'])]), timeout=30)
    raw = stdout.read(); assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    record = json.loads(raw); count += 1
    (local / 'observation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    if record['controller_exit'] is not None:
        assert not record['controller_alive']
        sftp = client.open_sftp(); dest = local / 'complete'; dest.mkdir()
        intake = []
        for folder in ('', 'nr3d_native/'):
            for name in sftp.listdir(launch['root'] + '/' + folder):
                if name.endswith(('.json', '.log', '.exit')):
                    relative = folder + name
                    with sftp.open(launch['root'] + '/' + relative, 'rb') as stream: data = stream.read()
                    path = dest / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
                    assert path.read_bytes() == data
                    intake.append(dict(name=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        sftp.close(); client.close()
        result = dict(observer_closed=True, terminal=record, observation_count=count,
                      files_copied=len(intake), intake=intake, model_or_optimizer_replayed=False)
        (local / 'wait.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(status='ZERO_UPDATE_DIAGNOSTIC_CLOSED', exitcode=record['controller_exit'],
                              files_copied=len(intake), observation_count=count)), flush=True)
        break
    assert record['controller_alive']; client.close()
    wait.update(remote_queries_performed=count,
                next_observation_cst=(datetime.datetime.now().astimezone() + datetime.timedelta(seconds=240)).isoformat())
    (local / 'observer_wait.json').write_text(json.dumps(wait, indent=2) + '\n', encoding='utf-8')
    time.sleep(240)
