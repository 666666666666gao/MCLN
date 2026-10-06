"""One timed read-only M0 observer; estimates use actual completed-arm duration."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import paramiko

local=Path(__file__).resolve().parent
assert not (local/'observer_wait.json').exists() and not (local/'preflight_wait.json').exists()
launch=json.loads((local/'preflight_launch.json').read_bytes())
spec=json.loads((local/'control_spec.json').read_bytes())
first=datetime.datetime.fromisoformat(launch['time_cst'])+datetime.timedelta(seconds=launch['first_check_seconds'])
wait=dict(observer_local_pid=os.getpid(),controller_pid=launch['controller_pid'],first_observation_cst=first.isoformat(),remote_queries_performed=0)
(local/'observer_wait.json').write_text(json.dumps(wait,indent=2)+'\n',encoding='utf-8')
time.sleep(max(0,first.timestamp()-time.time()))
code='''import datetime,json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);proc=Path('/proc')/sys.argv[2];alive=proc.exists()
command=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode() if alive else ''
assert not alive or str(root)+'/controller.py' in command
status=json.loads((root/'preflight_status.json').read_bytes())
result=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),controller_alive=alive,status=status,
    controller_exit=int((root/'preflight_controller.exit').read_text()) if (root/'preflight_controller.exit').exists() else None,receipts={})
for arm in ('control','keep'):
    path=root/arm/'preflight.json'
    if path.exists():result['receipts'][arm]=json.loads(path.read_bytes())
child=Path('/proc')/str(status.get('child_pid',0))
if child.exists():
    stat=(child/'stat').read_text();parts=stat[stat.rfind(')')+2:].split()
    result['child_elapsed_seconds']=time.monotonic()-int(parts[19])/os.sysconf('SC_CLK_TCK')
print(json.dumps(result))
'''
observations=[]
while True:
    client=paramiko.SSHClient();client.load_system_host_keys()
    client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
    _,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code,launch['root'],str(launch['controller_pid'])]),timeout=30)
    raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
    record=json.loads(raw);observations.append(record)
    (local/'observation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    if record['controller_exit'] is not None:
        assert not record['controller_alive']
        sftp=client.open_sftp();dest=local/'preflight_complete';dest.mkdir();intake=[]
        for folder in ('','control/','keep/'):
            for name in sftp.listdir(launch['root']+'/'+folder):
                if name.endswith(('.json','.log','.exit','.py','.md')):
                    relative=folder+name
                    with sftp.open(launch['root']+'/'+relative,'rb') as stream:data=stream.read()
                    path=dest/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                    assert path.read_bytes()==data
                    intake.append(dict(name=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        sftp.close();client.close()
        result=dict(observer_closed=True,controller_alive=False,exitcode=record['controller_exit'],
            status=record['status'],terminal=record,files_copied=len(intake),intake=intake,observations=len(observations))
        (local/'preflight_wait.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(status='REFERENCE_KEEP_M0_CLOSED',exitcode=record['controller_exit'],completed=list(record['receipts']))),flush=True)
        break
    assert record['controller_alive']
    timings=[r['elapsed_seconds'] for r in record['receipts'].values()]
    delay=max(240,max(timings)-record.get('child_elapsed_seconds',0)-180) if timings else 240
    client.close();wait.update(remote_queries_performed=len(observations),
        next_observation_cst=(datetime.datetime.now().astimezone()+datetime.timedelta(seconds=delay)).isoformat())
    (local/'observer_wait.json').write_text(json.dumps(wait,indent=2)+'\n',encoding='utf-8');time.sleep(delay)
