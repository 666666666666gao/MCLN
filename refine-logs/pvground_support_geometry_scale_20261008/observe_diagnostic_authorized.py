"""Observe the original diagnostic at its scheduled node; no launch or model action."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import paramiko

local=Path(__file__).resolve().parent
assert not (local/'observer_terminal.json').exists()
launch=json.loads((local/'launch.json').read_bytes())
spec=json.loads((local/'pair_spec.json').read_bytes())
pid=int(launch['process'].split()[0])
target=datetime.datetime.fromisoformat(launch['first_observation_cst'])
print('WAIT_UNTIL '+target.isoformat(),flush=True)
code=r'''import datetime,json,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root']);pid=b['pid'];proc=Path('/proc')/str(pid)
alive=proc.exists()
if alive:
    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert str(root/'run_mask_support_pair.py') in cmd and '--mode preflight' in cmd
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),controller_alive=alive,pid=pid)
for name,key in [('diagnostic.exit','exitcode'),('progress.json','progress'),('diagnostic.json','diagnostic')]:
    path=root/name
    if path.exists():record[key]=int(path.read_text()) if key=='exitcode' else json.loads(path.read_bytes())
log=root/'diagnostic.log'
if log.exists():record['tail']=log.read_text(errors='replace').splitlines()[-35:]
print(json.dumps(record))
'''
index=0
while True:
    wait=max(0,(target-datetime.datetime.now().astimezone()).total_seconds())
    if wait:time.sleep(wait)
    client=paramiko.SSHClient();client.load_system_host_keys()
    client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
    stdin,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code]),timeout=120)
    stdin.write(json.dumps(dict(root=spec['root'],pid=pid)));stdin.channel.shutdown_write()
    raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
    record=json.loads(raw);index+=1
    (local/('observation_%03d.json'%index)).write_bytes(raw)
    print(json.dumps(dict(observation=index,time_cst=record['time_cst'],controller_alive=record['controller_alive'],exitcode=record.get('exitcode'),progress=record.get('progress'))),flush=True)
    if 'exitcode' in record:
        assert not record['controller_alive']
        receipt=dict(observer_closed=True,terminal=record,observations=index,nn_or_optimizer_restarted=False)
        (local/'observer_terminal.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        if record['exitcode']==0:
            assert record['diagnostic']['status']=='complete' and record['diagnostic']['optimizer_updates']==0
            sftp=client.open_sftp()
            archive=local/'actual';archive.mkdir()
            for name in ('diagnostic.json','progress.json','diagnostic.exit','diagnostic.log','imports.json','load.json'):
                with sftp.open(spec['root']+'/'+name,'rb') as f:data=f.read()
                path=archive/name
                with path.open('xb') as f:f.write(data)
                assert path.read_bytes()==data
            sftp.close()
        client.close();break
    assert record['controller_alive'],'Original diagnostic disappeared without an exit receipt; do not restart it'
    client.close();target=datetime.datetime.now().astimezone()+datetime.timedelta(seconds=240)
print('ORIGINAL_DIAGNOSTIC_OBSERVER_CLOSED',flush=True)
