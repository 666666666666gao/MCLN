"""One formal observer, delayed near estimated finish; collect read-only evidence."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import tarfile
import time
import paramiko

local = Path(__file__).resolve().parent
assert not (local/'formal_wait.json').exists() and not (local/'formal_observer_live.json').exists()
review = json.loads((local/'FORMAL_LAUNCH_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
launch = json.loads((local/'formal_launch.json').read_bytes())
spec = json.loads((local/'spec.json').read_bytes())
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['root']+'/formal_status.json','rb') as stream:
    status = json.loads(stream.read())
assert status['mode']=='formal' and not status['completed']
commands = {}
for role in ('controller','child'):
    with sftp.open('/proc/'+str(status[role+'_pid'])+'/cmdline','rb') as stream:
        commands[role] = stream.read().replace(b'\0',b' ').decode().strip()
assert spec['root']+'/controller.py --mode formal' in commands['controller']
assert spec['root']+'/run_extent_diagnostic.py' in commands['child'] and '--mode formal' in commands['child']
deadline = datetime.datetime.fromisoformat(status['started_cst']) + datetime.timedelta(
    seconds=launch['estimate_total_seconds']-launch['first_poll_before_estimated_end_seconds'])
live = dict(status='ACTUAL_FORMAL_CONTROLLER_AND_CHILD_LIVE',controller=status,commands=commands,
            time_cst=datetime.datetime.now().astimezone().isoformat(),first_check_cst=deadline.isoformat(),
            remote_poll_seconds=launch['remote_poll_seconds'],observer_closed=False,optimizer_updates=0,new_weights=0)
(local/'formal_observer_live.json').write_text(json.dumps(live,indent=2)+'\n')
print(json.dumps(live),flush=True)
time.sleep(max(0,(deadline-datetime.datetime.now().astimezone()).total_seconds()))
while True:
    with sftp.open(spec['root']+'/formal_status.json','rb') as stream:
        status = json.loads(stream.read())
    if status['completed']:
        break
    print('ACTUAL_FORMAL_CONTROLLER_WAIT '+json.dumps(status),flush=True)
    time.sleep(launch['remote_poll_seconds'])
assert status['exit_code']==0,status
command = shlex.join([spec['runtime']+'/venv/bin/python','-B',spec['root']+'/analyze_extent.py',
                      '--directory',spec['root']+'/formal'])
_,stdout,stderr = client.exec_command(command,timeout=300)
cpu = json.loads(stdout.read()); assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert cpu['rows']==9508 and cpu['preflight_actual_raw_member_rows_replayed']==0
assert not any(value for item in cpu['CPU_stored_threshold_flips'].values() for value in item.values())
code = '''import sys,tarfile
from pathlib import Path
root=Path(sys.argv[1])
paths=sorted((root/'formal').rglob('*'))+[root/'formal_status.json',root/'formal.exit',root/'formal.log']
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for path in paths:
        if path.is_file() and path.suffix in ('.json','.jsonl','.npz','.log','.exit'):
            archive.add(str(path),arcname=str(path.relative_to(root)),recursive=False)
'''
_,stdout,stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code,spec['root']]),timeout=300)
complete = local/'complete'; assert complete.is_dir() and not (complete/'formal').exists()
files = {}
with tarfile.open(fileobj=stdout,mode='r|gz') as archive:
    for item in archive:
        assert item.isfile()
        path = complete/item.name
        assert complete.resolve() in path.resolve().parents and not path.exists()
        path.parent.mkdir(parents=True,exist_ok=True)
        data = archive.extractfile(item).read()
        path.write_bytes(data)
        files[item.name] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert json.loads((complete/'formal/CPU_SUMMARY.json').read_bytes())==cpu
receipt = json.loads((complete/'formal/receipt.json').read_bytes())
assert receipt['model_states_unchanged'] and not receipt['optimizer_created'] and receipt['optimizer_updates']==0
record = dict(status='ACTUAL_FORMAL_CLOSED_AND_CPU_MEMBER_REPLAYED',controller=status,
              observer_closed=True,first_check_cst=deadline.isoformat(),remote_poll_seconds=launch['remote_poll_seconds'],
              time_cst=datetime.datetime.now().astimezone().isoformat(),cpu=cpu,files=files,
              collected_bytes=sum(item['bytes'] for item in files.values()),weights_downloaded=0,
              raw_point_replay_scope='Only actual8rowM0; formal9508 verifies storedSPextrema and quantile neighbor interpolation')
(local/'formal_wait.json').write_text(json.dumps(record,indent=2)+'\n')
sftp.close(); client.close()
print(json.dumps(dict(status=record['status'],controller=status,cpu=cpu,
                     file_count=len(files),collected_bytes=record['collected_bytes'],weights_downloaded=0)),flush=True)
