"""Single estimated-time observer, then one CPU check and M0 evidence intake."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import tarfile
import time
import paramiko

local = Path(__file__).resolve().parent
assert not (local/'preflight_wait.json').exists() and not (local/'complete').exists()
launch = json.loads((local/'preflight_launch.json').read_bytes())
spec = json.loads((local/'spec.json').read_bytes())
deadline = datetime.datetime.fromisoformat(launch['time_cst']) + datetime.timedelta(seconds=120)
time.sleep(max(0, (deadline-datetime.datetime.now().astimezone()).total_seconds()))
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
while True:
    with sftp.open(spec['root']+'/preflight_status.json','rb') as stream:
        status = json.loads(stream.read())
    if status['completed']:
        break
    print('ACTUAL_PREFLIGHT_CONTROLLER_WAIT '+json.dumps(status),flush=True)
    time.sleep(180)
assert status['exit_code'] == 0,status
command = shlex.join([spec['runtime']+'/venv/bin/python','-B',spec['root']+'/analyze_extent.py',
                      '--directory',spec['root']+'/preflight'])
_,stdout,stderr = client.exec_command(command,timeout=300)
cpu = json.loads(stdout.read()); assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert cpu['rows']==cpu['preflight_actual_raw_member_rows_replayed']==8
assert not any(value for item in cpu['CPU_stored_threshold_flips'].values() for value in item.values())
code = '''import sys,tarfile
from pathlib import Path
root=Path(sys.argv[1])
paths=sorted((root/'preflight').rglob('*'))+[root/'preflight_status.json',root/'preflight.exit',root/'preflight.log']
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for path in paths:
        if path.is_file() and path.suffix in ('.json','.jsonl','.npz','.log','.exit'):
            archive.add(str(path),arcname=str(path.relative_to(root)),recursive=False)
'''
_,stdout,stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code,spec['root']]),timeout=300)
wire = stdout.read(); assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
complete = local/'complete'; complete.mkdir()
files = {}
with tarfile.open(fileobj=io.BytesIO(wire),mode='r:gz') as archive:
    for item in archive:
        assert item.isfile()
        path = complete/item.name
        assert complete.resolve() in path.resolve().parents
        path.parent.mkdir(parents=True,exist_ok=True)
        data = archive.extractfile(item).read()
        path.write_bytes(data)
        files[item.name] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
assert json.loads((complete/'preflight/CPU_SUMMARY.json').read_bytes()) == cpu
record = dict(status='ACTUAL_PREFLIGHT_CLOSED_AND_CPU_RAW_REPLAYED',controller=status,
              observer_closed=True,first_check_cst=deadline.isoformat(),remote_poll_seconds=180,
              time_cst=datetime.datetime.now().astimezone().isoformat(),cpu=cpu,
              files=files,wire_bytes=len(wire),weights_downloaded=0)
(local/'preflight_wait.json').write_text(json.dumps(record,indent=2)+'\n')
sftp.close(); client.close(); print(json.dumps(record),flush=True)
