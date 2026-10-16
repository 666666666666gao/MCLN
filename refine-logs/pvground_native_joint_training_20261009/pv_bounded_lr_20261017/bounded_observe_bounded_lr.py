"""One timed read of one already launched finite LR run; no NN/GPU actions."""
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
parser.add_argument('--run-name', required=True, choices=['lr_scale_0p1', 'lr_scale_0p3'])
parser.add_argument('--node', required=True)
parser.add_argument('--due-cst', required=True)
args = parser.parse_args()
assert args.node.isidentifier()
root = Path('\\\\?\\' + str(Path(__file__).resolve().parent))
launch = json.loads((root / ('NORMAL_' + args.run_name + '_LAUNCH.json')).read_bytes())
expected = '/root/autodl-tmp/pvground_bounded_lr_20261017/' + args.run_name
assert launch['root'] == expected
prefix = 'OBS_' + args.run_name + '_' + args.node
owner = root / (prefix + '_OWNER.json')
assert not owner.exists()
due = datetime.datetime.fromisoformat(args.due_cst)
owner.write_text(json.dumps(dict(local_pid=os.getpid(), due_cst=due.isoformat(),
    created_cst=datetime.datetime.now().astimezone().isoformat(), scheduled_reads=1,
    controller_pid=launch['controller_pid'], followup_seconds=240), indent=2) + '\n')
remaining = (due - datetime.datetime.now().astimezone()).total_seconds()
while remaining > 0:
    time.sleep(min(300, remaining))
    remaining = (due - datetime.datetime.now().astimezone()).total_seconds()
code = r'''import base64,datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['expected']);launch=b['launch']
controller=Path('/proc/'+str(launch['controller_pid']))
if controller.exists():assert controller.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')==launch['controller_argv'][4:]
status=json.loads((root/'normal_status.json').read_bytes());child=Path('/proc/'+str(status['child_pid']))
if child.exists():assert child.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')==status['child_argv']
files=[];tails=[]
def capture(p):
 raw=p.read_bytes();files.append(dict(name=p.relative_to(root).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode()))
for name in ('normal_status.json','admission.json','protocol.json'):capture(root/name)
for p in sorted((root/'logs').rglob('*')):
 if p.is_file() and (p.name in ('native_metrics.jsonl','native_stop.json','selected_geometry_responsibility.jsonl') or p.name.startswith('native_diagnostic_epoch')):capture(p)
if (root/'train.exit').is_file():capture(root/'train.exit')
for name in ('controller.log','train.log'):
 raw=(root/name).read_bytes();tails.append(dict(name=name,total_bytes=len(raw),base64=base64.b64encode(raw[-20000:]).decode()))
print(json.dumps(dict(observed_cst=datetime.datetime.now().astimezone().isoformat(),root=str(root),controller_pid=launch['controller_pid'],controller_alive=controller.exists(),child_alive=child.exists(),remote_status=status,files=files,tails=tails,new_neural_forwards=0,new_optimizer_updates=0,new_GPU_queries=0)))
'''
witness = json.loads((root.parent.parent.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,
    SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
ssh = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
       '-o', 'StrictHostKeyChecking=yes', '-o',
       'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
       '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
       'root@region-9.autodl.pro', ' '.join(shlex.quote(p) for p in [runtime, '-B', '-u', '-c', code])]
reply = subprocess.run(ssh, env=environment, input=json.dumps(dict(expected=expected, launch=launch)).encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
                       creationflags=subprocess.CREATE_NO_WINDOW)
(root / (prefix + '_STDOUT.json')).write_bytes(reply.stdout)
(root / (prefix + '_STDERR_PRIVATE.txt')).write_bytes(reply.stderr)
(root / (prefix + '_EXIT.json')).write_text(json.dumps(dict(exit_code=reply.returncode)) + '\n')
assert reply.returncode == 0
observed = json.loads(reply.stdout)
destination = root / (prefix + '_files')
destination.mkdir()
for item in observed['files']:
    raw = base64.b64decode(item.pop('base64'))
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    p = destination / item['name']
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
for item in observed['tails']:
    (destination / (item['name'] + '.tail.txt')).write_bytes(base64.b64decode(item.pop('base64')))
observed.update(first_due_cst=due.isoformat(), best_promoted=False, full_goal_complete=False)
(root / (prefix + '_INTAKE.json')).write_text(json.dumps(observed, indent=2) + '\n')
print(json.dumps({key: observed[key] for key in ('observed_cst', 'controller_alive', 'child_alive')}), flush=True)
