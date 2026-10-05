"""One observer per stage; use an estimated first check and 240-second polls."""
import datetime
import json
import math
import os
from pathlib import Path
import shlex
import time

import paramiko


local = Path(__file__).resolve().parent
stage = 'fit'
assert stage in ('preflight','fit')
wait_file = local/(stage+'_wait.json')
started = local/(stage+'_observer_started.json')
assert not wait_file.exists() and not started.exists()
launch = json.loads((local/(stage+'_launch.json')).read_bytes())
spec = json.loads((local/'control_spec.json').read_bytes())
root = launch['root']
directory = local/(stage+'_observations')
directory.mkdir()
started.write_text(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),stage=stage,
    launch=str(local/(stage+'_launch.json')),first_check_seconds=launch['first_check_seconds'],poll_seconds=240),indent=2)+'\n',encoding='utf-8')
elapsed = datetime.datetime.now().astimezone() - datetime.datetime.fromisoformat(launch['time_cst'])
time.sleep(max(0,launch['first_check_seconds']-elapsed.total_seconds()))
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
pattern = '^'+spec['runtime']+'/venv/bin/python -B -u '+root+'/controller.py --phase '+stage+'$'
max_observations = 1 + math.ceil((2*launch['estimated_seconds']-launch['first_check_seconds'])/240)
for index in range(1,max_observations+1):
    _,stdout,stderr = client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
    process = stdout.read().decode().strip()
    code = stdout.channel.recv_exit_status()
    assert code in (0,1),stderr.read().decode()
    with sftp.open(root+'/'+stage+'_status.json','rb') as stream:
        status = json.loads(stream.read())
    files = sftp.listdir(root)
    exitcode = None
    if stage+'_controller.exit' in files:
        with sftp.open(root+'/'+stage+'_controller.exit','rb') as stream:
            exitcode = int(stream.read().decode().strip())
    observation = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),stage=stage,
        controller_alive=bool(process),process=process,status=status,exitcode=exitcode)
    (directory/('observation_'+str(index)+'.json')).write_text(json.dumps(observation,indent=2)+'\n',encoding='utf-8')
    print('QUERY_GEOMETRY_OBSERVATION '+json.dumps(observation),flush=True)
    if not process and exitcode is not None:
        result = dict(time_cst=observation['time_cst'],observer_closed=True,stage=stage,terminal=observation,
            observation_count=index,inference_or_optimizer_replayed=False)
        wait_file.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print('QUERY_GEOMETRY_OBSERVER_CLOSED '+json.dumps(result),flush=True)
        sftp.close()
        client.close()
        raise SystemExit(0)
    time.sleep(240)
raise RuntimeError('Exceeded planned observer window; inspect recorded controller before any further action')
