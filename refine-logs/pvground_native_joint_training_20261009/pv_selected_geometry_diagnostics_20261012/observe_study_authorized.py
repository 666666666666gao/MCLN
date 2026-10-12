"""One scheduled read of the already launched pair or native formal arm."""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--stage', required=True, choices=['preflight', 'native_control', 'selected_geometry'])
parser.add_argument('--node', required=True)
parser.add_argument('--due-cst', required=True)
parser.add_argument('--collect-rows', action='store_true')
args = parser.parse_args()
assert args.node.isidentifier()
root = Path(__file__).resolve().parent
launch_name = 'PREFLIGHT_PAIR_LAUNCH.json' if args.stage == 'preflight' else 'NORMAL_' + args.stage + '_LAUNCH.json'
launch = json.loads((root / launch_name).read_bytes())
expected_root = '/root/autodl-tmp/pvground_deployment_geometry_training_20261012/'
expected_root += 'preflight' if args.stage == 'preflight' else 'normal/' + args.stage
assert launch['root'] == expected_root
prefix = 'OBS_' + args.stage + '_' + args.node
owner_path = root / (prefix + '_OWNER.json')
assert not owner_path.exists()
due = datetime.datetime.fromisoformat(args.due_cst)
owner = dict(local_pid=os.getpid(), created_cst=datetime.datetime.now().astimezone().isoformat(),
    due_cst=due.isoformat(), stage=args.stage, controller_pid=launch['controller_pid'],
    launch_file=launch_name, scheduled_reads=1, subsequent_poll_seconds=240)
owner_path.write_bytes((json.dumps(owner, indent=2) + '\n').encode())
print(json.dumps(owner), flush=True)
remaining = (due - datetime.datetime.now().astimezone()).total_seconds()
while remaining > 0:
    time.sleep(min(300, remaining))
    remaining = (due - datetime.datetime.now().astimezone()).total_seconds()
code = r'''import base64,datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['expected_root']);launch=b['launch'];stage=b['stage']
assert str(root)==launch['root']
controller=Path('/proc/'+str(launch['controller_pid']))
if controller.exists():
 assert controller.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')==launch['controller_argv'][4:]
status_name='status.json' if stage=='preflight' else 'normal_status.json'
status=json.loads((root/status_name).read_bytes())
child=Path('/proc/'+str(status['child_pid']))
if child.exists():
 assert child.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')==status['child_argv']
files=[]
def capture(path):
 raw=path.read_bytes();files.append(dict(name=path.relative_to(root).as_posix(),bytes=len(raw),
  sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode()))
capture(root/status_name);capture(root/'admission.json')
tails=[]
def tail(path):
 raw=path.read_bytes();tails.append(dict(name=path.relative_to(root).as_posix(),total_bytes=len(raw),
  base64=base64.b64encode(raw[-20000:]).decode()))
tail(root/'controller.log')
receipts={};weights=[]
if stage=='preflight':
 for arm in ('native_control','selected_geometry'):
  arm_root=root/arm
  for name in ('preflight.exit','witness/NATIVE_M0_TRAINING_DIAGNOSTIC.json','witness/NATIVE_M0_RECEIPT.json'):
   path=arm_root/name
   if path.is_file():capture(path)
  path=arm_root/'witness/NATIVE_M0_RECEIPT.json'
  if path.is_file():receipts[arm]=json.loads(path.read_bytes())
  path=arm_root/'preflight.log'
  if path.is_file():tail(path)
  if status['status']=='complete' and not controller.exists() and not child.exists():
   checkpoints=list(arm_root.rglob('*.pth'))
   assert len(checkpoints)==1 and checkpoints[0].name=='best.pth'
   checkpoint=checkpoints[0];receipt=receipts[arm]
   raw=checkpoint.read_bytes();digest=hashlib.sha256(raw).hexdigest()
   assert len(raw)==receipt['checkpoint_bytes'] and digest==receipt['checkpoint_sha256']
   weights.append(dict(arm=arm,path=str(checkpoint),bytes=len(raw),sha256=digest,unscored_engineering_only=True))
   del raw
else:
 for path in sorted(root.joinpath('logs').rglob('*')):
  if path.is_file() and (path.name in ('native_metrics.jsonl','native_stop.json','selected_geometry_responsibility.jsonl')
   or path.name.startswith('native_diagnostic_epoch')):capture(path)
 if b['collect_rows']:
  assert status['status']=='complete' and not controller.exists() and not child.exists()
  for path in sorted(root.joinpath('logs').rglob('native_rows_epoch*.jsonl')):capture(path)
 for name in ('protocol.json','train.exit'):
  path=root/name
  if path.is_file():capture(path)
 path=root/'train.log'
 if path.is_file():tail(path)
 if status['status']=='complete' and not controller.exists() and not child.exists():
  for path in sorted(root.joinpath('logs').rglob('*.pth')):
   assert path.name in ('best.pth','latest.pth')
   weights.append(dict(path=str(path),bytes=path.stat().st_size))
print(json.dumps(dict(observed_cst=datetime.datetime.now().astimezone().isoformat(),stage=stage,
 root=str(root),controller_pid=launch['controller_pid'],controller_alive=controller.exists(),
 child_alive=child.exists(),remote_status=status,files=files,tails=tails,receipts=receipts,weights=weights,
 new_neural_forwards=0,new_GPU_queries=0,new_optimizer_updates=0)))
'''
project = root.parent.parent
witness = json.loads((project / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,
    SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o',
    'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', ' '.join(shlex.quote(part) for part in [runtime, '-B', '-u', '-c', code])]
assert not args.collect_rows or args.stage != 'preflight'
body = dict(stage=args.stage, expected_root=expected_root, launch=launch, collect_rows=args.collect_rows)
response = subprocess.run(argv, env=environment, input=json.dumps(body).encode(), stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW, timeout=300)
(root / (prefix + '_STDOUT.json')).write_bytes(response.stdout)
(root / (prefix + '_STDERR_PRIVATE.txt')).write_bytes(response.stderr)
(root / (prefix + '_EXIT.json')).write_bytes((json.dumps(dict(exit_code=response.returncode)) + '\n').encode())
assert response.returncode == 0
observed = json.loads(response.stdout)
destination = root / (prefix + '_files')
destination.mkdir()
for row in observed['files']:
    raw = base64.b64decode(row.pop('base64'))
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    path = destination / row['name']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
for row in observed['tails']:
    path = destination / (row['name'] + '.tail.txt')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(row.pop('base64')))
observed.update(first_due_cst=due.isoformat(), actual_weights_loaded=False,
                best_promoted=False, full_goal_complete=False)
(root / (prefix + '_INTAKE.json')).write_bytes((json.dumps(observed, indent=2) + '\n').encode())
print(json.dumps({name: observed[name] for name in ('observed_cst', 'stage', 'controller_alive', 'child_alive')}), flush=True)
