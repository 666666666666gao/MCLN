"""One estimated wait, then240-second SSH status checks until closure."""
import datetime
import json
import os
from pathlib import Path
import shlex
import time
import paramiko

local = Path(__file__).resolve().parent
launch = json.loads((local/'preflight_launch.json').read_bytes())
spec = json.loads((local/'control_spec.json').read_bytes())
assert not (local/'preflight_wait.json').exists()
directory = local/'preflight_observations'
directory.mkdir()
elapsed = (datetime.datetime.now().astimezone()-datetime.datetime.fromisoformat(launch['time_cst'])).total_seconds()
time.sleep(max(0, launch['first_check_seconds']-elapsed))
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
root = launch['root']
pattern = '^'+spec['runtime']+'/venv/bin/python -B -u '+root+'/controller.py --phase preflight$'
for index in range(1,8):
    _,stdout,stderr = client.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
    process = stdout.read().decode().strip()
    code = stdout.channel.recv_exit_status()
    assert code in (0,1),stderr.read().decode()
    with sftp.open(root+'/preflight_status.json','rb') as stream:
        status = json.loads(stream.read())
    exitcode = None
    if 'preflight_controller.exit' in sftp.listdir(root):
        with sftp.open(root+'/preflight_controller.exit','rb') as stream:
            exitcode = int(stream.read().decode().strip())
    record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status=status,
                  controller_alive=bool(process),exitcode=exitcode)
    (directory/('observation_'+str(index)+'.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('QUERY_GEOMETRY_PREFLIGHT_OBSERVATION '+json.dumps(record),flush=True)
    if not process and exitcode is not None:
        record.update(observer_closed=True,observation_count=index)
        (local/'preflight_wait.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        sftp.close();client.close()
        raise SystemExit(0)
    time.sleep(240)
raise RuntimeError('Probe exceeded its estimated window; inspect captured primary logs before any further action.')
