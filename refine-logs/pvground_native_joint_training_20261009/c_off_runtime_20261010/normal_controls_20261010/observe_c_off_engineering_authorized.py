"""Wait to the planned node; one native SSH result query each 240 seconds."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time

root=Path(__file__).resolve().parent
project=root.parent.parent
launch=json.loads((root/'ENGINEERING_LAUNCH.json').read_bytes())
assert not (root/'ENGINEERING_INTAKE.json').exists()
due=datetime.datetime.fromisoformat(launch['first_observation_cst'])
remaining=(due-datetime.datetime.now().astimezone()).total_seconds()
if remaining>0:
    time.sleep(remaining)
witness=json.loads((project/'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment=dict(os.environ)
environment.update(SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
code=r'''import base64,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/pvground_native_joint_controls_20261010/selected_mask_off/engineering')
alive=Path('/proc/'+str(b['pid'])).exists()
status_path=root/'engineering_status.json'
status=json.loads(status_path.read_bytes()) if status_path.exists() else None
files=[];weights=[]
if not alive:
 for path in sorted(root.rglob('*')):
  if not path.is_file():continue
  row=dict(name=str(path.relative_to(root)),bytes=path.stat().st_size)
  if path.suffix in ('.json','.log','.txt','.py','.exit'):
   raw=path.read_bytes();row.update(sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode());files.append(row)
  elif path.suffix=='.pth':
   h=hashlib.sha256()
   with path.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
   row.update(path=str(path),sha256=h.hexdigest());weights.append(row)
 port=b['port']
 for name,row in port['files'].items():assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
else:port=None
print(json.dumps(dict(controller_alive=alive,status=status,files=files,weights=weights,
 port=port,new_neural_calls=0,formal_accuracy=None,normal_training_started=False)))
'''
argv=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none','-o','StrictHostKeyChecking=yes',
    '-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
observations=[]
while True:
    when=datetime.datetime.now().astimezone().isoformat()
    response=subprocess.run(argv,env=environment,
        input=json.dumps(dict(root=launch['root'],pid=launch['controller_pid'],port=json.loads((root.parent/'NATIVE_SOURCE_PORT.json').read_bytes()))).encode(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
    (root/'ENGINEERING_READ_STDERR.txt').write_bytes(response.stderr)
    (root/'ENGINEERING_READ_STDOUT.json').write_bytes(response.stdout)
    (root/'ENGINEERING_READ_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
    assert response.returncode==0,'Read preserved ENGINEERING_READ_STDERR.txt; never restart for observer failure'
    value=json.loads(response.stdout)
    observations.append(dict(time_cst=when,controller_alive=value['controller_alive'],status=value['status']))
    (root/'ENGINEERING_OBSERVATIONS.json').write_text(json.dumps(dict(
        first_due_cst=launch['first_observation_cst'],poll_seconds=240,observations=observations),indent=2)+'\n')
    if not value['controller_alive']:
        break
    time.sleep(240)
destination=root/'actual_engineering'
assert not destination.exists()
destination.mkdir()
for row in value['files']:
    name=PurePosixPath(row['name'])
    path=destination.joinpath(*name.parts)
    assert destination.resolve() in path.resolve().parents
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(row.pop('base64'))
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    path.write_bytes(raw)
record=dict(status='ACTUAL_ENGINEERING_CLOSED_INTAKE',remote_root=launch['root'],
    controller_pid=launch['controller_pid'],controller_alive=False,observations=observations,
    remote_status=value['status'],files=value['files'],weights=value['weights'],
    port=value['port'],new_neural_calls=0,formal_accuracy=None,normal_training_started=False)
(root/'ENGINEERING_INTAKE.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],remote_status=record['remote_status'],
    observations=len(observations),files=len(value['files']),weights=len(value['weights']))),flush=True)
assert value['status']['status']=='complete'
