"""One read at the estimated completion boundary; never starts neural work."""
import argparse
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

parser=argparse.ArgumentParser()
parser.add_argument('--stage',choices=('cpu','sanity','full'),required=True)
parser.add_argument('--node',required=True)
parser.add_argument('--due-cst',required=True)
args=parser.parse_args()
assert args.node.isidentifier()
root=Path(__file__).resolve().parent
launch=json.loads((root/(args.stage+'_LAUNCH.json')).read_bytes())
prefix=args.stage+'_'+args.node
assert not (root/(prefix+'_OWNER.json')).exists()
due=datetime.datetime.fromisoformat(args.due_cst)
owner=dict(local_pid=os.getpid(),created_cst=datetime.datetime.now().astimezone().isoformat(),
           due_cst=due.isoformat(),stage=args.stage,controller_pid=launch['controller_pid'],
           planned_observations=1,later_poll_seconds=240)
(root/(prefix+'_OWNER.json')).write_text(json.dumps(owner,indent=2)+'\n')
print(json.dumps(owner),flush=True)
remaining=(due-datetime.datetime.now().astimezone()).total_seconds()
while remaining>0:
    time.sleep(min(300,remaining))
    remaining=(due-datetime.datetime.now().astimezone()).total_seconds()
code=r'''import base64,datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/mcln_portable_best_transfer_20261012')/b['stage']
controller=Path('/proc/'+str(b['controller_pid']))
if controller.exists():
 expected=b['controller_argv'] if b['stage']=='cpu' else b['controller_argv'][4:]
 assert controller.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')==expected
status=json.loads((root/'status.json').read_bytes())
child=Path('/proc/'+str(status['child_pid']))
if child.exists():assert child.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')==status['child_argv']
files=[]
for name in ('status.json','admission.json','child.exit','CPU_RESULT.json','replay/RESULT.json'):
 path=root/name
 if path.is_file():
  raw=path.read_bytes();files.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode()))
tails=[]
for name in ('stage.log','controller.log'):
 path=root/name
 if path.is_file():
  raw=path.read_bytes();tails.append(dict(name=name,total_bytes=len(raw),base64=base64.b64encode(raw[-16000:]).decode()))
print(json.dumps(dict(observed_cst=datetime.datetime.now().astimezone().isoformat(),stage=b['stage'],
 controller_alive=controller.exists(),child_alive=child.exists(),remote_status=status,files=files,tails=tails,
 new_neural_forwards=0,new_GPU_queries=0,new_optimizer_updates=0)))
'''
witness=json.loads(Path('C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/SCP_TRANSPORT_WITNESS.json').read_bytes())
environment=dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
command=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none','-o','StrictHostKeyChecking=yes',
 '-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
 '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
 'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
response=subprocess.run(command,input=json.dumps(launch).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=environment,timeout=300)
(root/(prefix+'_STDOUT.json')).write_bytes(response.stdout)
(root/(prefix+'_STDERR.txt')).write_bytes(response.stderr)
(root/(prefix+'_EXIT.json')).write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
assert response.returncode==0,response.stderr.decode(errors='replace')
observed=json.loads(response.stdout)
destination=root/(prefix+'_observation');destination.mkdir()
for row in observed['files']:
    raw=base64.b64decode(row.pop('base64'))
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    path=destination/row['name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for row in observed['tails']:
    (destination/(row['name']+'.tail.txt')).write_bytes(base64.b64decode(row.pop('base64')))
observed.update(first_due_cst=due.isoformat(),best_promoted=False,full_goal_complete=False)
(root/(prefix+'_OBSERVATION.json')).write_text(json.dumps(observed,indent=2)+'\n')
print(json.dumps({name:observed[name] for name in ('observed_cst','stage','controller_alive','child_alive')}),flush=True)
