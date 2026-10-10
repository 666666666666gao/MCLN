"""One C-off E2 boundary observation using the actual E2 rate and completed E1 validation."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
project=root.parent.parent
launch=json.loads((root/'NORMAL_LAUNCH.json').read_bytes())
start=json.loads((root/'NORMAL_START_WITNESS.json').read_bytes())
assert start['status']=='ORDINARY_NATIVE_ENTRY_ALIVE' and start['controller_pid']==launch['controller_pid']
assert not (root/'NORMAL_EPOCH2_OBSERVATION.json').exists() and not (root/'NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').exists()
plan=json.loads((root/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())
assert plan['original_first_observer_closed']
due=datetime.datetime.fromisoformat(plan['due_cst'])
owner=dict(status='SINGLE_PLANNED_EPOCH2_OBSERVER_WAITING',local_pid=os.getpid(),
    runtime_executable=sys.executable,created_cst=datetime.datetime.now().astimezone().isoformat(),
    first_due_cst=due.isoformat(),later_poll_seconds=240,remote_controller_pid=launch['controller_pid'],
    remote_child_pid=start['child_pid'],planned_observations=1,
    estimate_basis=plan['basis'],
    later_node_requires_actual_progress_estimate=True)
(root/'NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
print(json.dumps(owner),flush=True)
remaining=(due-datetime.datetime.now().astimezone()).total_seconds()
while remaining>0:
 time.sleep(min(300,remaining))
 remaining=(due-datetime.datetime.now().astimezone()).total_seconds()
witness=json.loads((project/'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment=dict(os.environ)
environment.update(SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
code=r'''import base64,datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root']);assert root==Path('/root/autodl-tmp/pvground_native_joint_controls_20261010/selected_mask_off')
status=json.loads((root/'normal_status.json').read_bytes())
controller=Path('/proc/'+str(b['controller_pid']));child=Path('/proc/'+str(b['child_pid']))
alive=controller.exists();child_alive=child.exists()
if alive:assert controller.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')==b['controller_argv'][4:]
if child_alive:assert child.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')==b['child_argv']
metric_paths=list((root/'logs').rglob('native_metrics.jsonl'));assert len(metric_paths)<=1
metrics=[];files=[]
for path in metric_paths:
 raw=path.read_bytes();metrics=[json.loads(line) for line in raw.decode().splitlines()]
 assert all(row['rows']==9508 and row['primary_score']=='last/bbs' for row in metrics)
 files.append(dict(name=str(path.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode()))
log=root/'train.log';size=log.stat().st_size
with log.open('rb') as stream:stream.seek(max(0,size-40000));tail=stream.read()
for name in ('normal_status.json','train.exit','controller.log','admission.json'):
 path=root/name
 if path.is_file():
  raw=path.read_bytes();files.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode()))
print(json.dumps(dict(status='SINGLE_SCHEDULED_NORMAL_OBSERVATION',observed_cst=datetime.datetime.now().astimezone().isoformat(),
 controller_alive=alive,child_alive=child_alive,remote_status=status,metrics=metrics,files=files,
 train_log_bytes=size,train_tail_base64=base64.b64encode(tail).decode(),new_neural_calls=0,weight_promoted=False)))
'''
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
body=dict(root=launch['root'],controller_pid=launch['controller_pid'],controller_argv=launch['controller_argv'],
    child_pid=start['child_pid'],child_argv=start['child_argv'])
response=subprocess.run(argv,env=environment,input=json.dumps(body).encode(),stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'NORMAL_EPOCH2_OBSERVATION_STDOUT.json').write_bytes(response.stdout)
(root/'NORMAL_EPOCH2_OBSERVATION_STDERR.txt').write_bytes(response.stderr)
(root/'NORMAL_EPOCH2_OBSERVATION_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
assert response.returncode==0,'Inspect private preserved observer stderr; do not restart training'
value=json.loads(response.stdout);destination=root/'normal_epoch2_observation';assert not destination.exists();destination.mkdir()
for row in value['files']:
 path=destination/row['name'];assert destination.resolve() in path.resolve().parents
 raw=base64.b64decode(row.pop('base64'));assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
(destination/'train_tail.txt').write_bytes(base64.b64decode(value.pop('train_tail_base64')))
value.update(first_due_cst=owner['first_due_cst'],observation_count=1,later_poll_seconds=240,
    later_node_requires_actual_progress_estimate=True,formal_results_unaudited=True,full_goal_complete=False)
(root/'NORMAL_EPOCH2_OBSERVATION.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(status=value['status'],controller_alive=value['controller_alive'],child_alive=value['child_alive'],
    metrics=value['metrics'],time_cst=value['observed_cst'])),flush=True)

