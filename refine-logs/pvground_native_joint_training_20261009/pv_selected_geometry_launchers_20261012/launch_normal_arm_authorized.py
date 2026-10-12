"""Admit one fresh formal arm only after the actual paired preflight audit."""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--arm', required=True, choices=['native_control', 'selected_geometry'])
parser.add_argument('--pair-intake', required=True)
parser.add_argument('--mcln-terminal-observation', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
prefix = 'NORMAL_' + args.arm
assert not (root / (prefix + '_LAUNCH.json')).exists()
pair_path = Path(args.pair_intake)
pair = json.loads(pair_path.read_bytes())
assert not pair['controller_alive'] and not pair['child_alive']
assert pair['remote_status']['status'] == 'complete' and pair['remote_status']['exit_code'] == 0
assert pair['remote_status']['same_augmented_inputs_verified']
assert pair['remote_status']['actual_total_native_updates'] == 4
terminal_path = Path(args.mcln_terminal_observation)
terminal = json.loads(terminal_path.read_bytes())
assert not terminal['controller_alive'] and not terminal['child_alive']
assert terminal['remote_status']['status'] == 'complete' and terminal['remote_status']['exit_code'] == 0
review_path = root / 'source_review/SOURCE_REVIEW_LAUNCHERS.json'
review = json.loads(review_path.read_bytes())
assert review['scope'] == 'SOURCE_ONLY' and review['source_preparation_pass'] and not review['scoped_blockers']
audit_path = root / 'actual_pair_review/EXPERIMENT_AUDIT.json'
audit = json.loads(audit_path.read_bytes())
assert audit['execution_scope'] == 'ACTUAL_SELECTED_GEOMETRY_NATIVE_PREFLIGHT_PAIR'
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
assert audit['audited_input_hashes'][str(pair_path.resolve())] == hashlib.sha256(pair_path.read_bytes()).hexdigest()
stage = json.loads((root / 'SOURCE_STAGE_R3_RECEIPT.json').read_bytes())
protocol = json.loads((root / (args.arm + '_protocol.json')).read_bytes())
assert protocol['arm'] == args.arm and protocol['max_epochs'] == 8 and protocol['patience'] == 2
assert not protocol['normal_training_started'] and not protocol['GPU_admitted']
files = {'normal_run_controller.py': (root / 'normal_run_controller_v3.py').read_bytes(),
         'protocol.json': (json.dumps(protocol, indent=2) + '\n').encode()}
code = r'''import base64,datetime,hashlib,json,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);base=Path('/root/autodl-tmp/pvground_deployment_geometry_training_20261012');root=base/'normal'/b['arm']
assert not root.exists()
terminal=json.loads(Path('/root/autodl-tmp/mcln_portable_best_transfer_20261012/normal_training/status.json').read_bytes())
assert terminal==b['terminal'] and terminal['status']=='complete' and terminal['exit_code']==0
assert not Path('/proc/271902').exists() and not Path('/proc/271903').exists()
pair=json.loads((base/'preflight/status.json').read_bytes());assert pair==b['pair']['remote_status']
assert pair['status']=='complete' and pair['exit_code']==0
assert not Path('/proc/'+str(b['pair']['controller_pid'])).exists()
for arm,receipt in b['pair']['receipts'].items():
 assert json.loads((base/'preflight'/arm/'witness/NATIVE_M0_RECEIPT.json').read_bytes())==receipt
 assert receipt['full_recovery_exact'] and receipt['actual_updates']==2 and receipt['full_state_tensors']==1295
 assert receipt['selected_mask_supervision_disabled']
 assert receipt['selected_geometry_supervision']==(arm=='selected_geometry')
assert b['pair']['receipts']['native_control']['input_batch_sha256']==b['pair']['receipts']['selected_geometry']['input_batch_sha256']
# The direct strategy comparison follows the completed native control on the sole card.
if b['arm']=='selected_geometry':
 control=json.loads((base/'normal/native_control/normal_status.json').read_bytes())
 assert control['status']=='complete' and control['exit_code']==0
 assert not Path('/proc/'+str(control['child_pid'])).exists()
stage=json.loads((base/'source_stage_manifest.json').read_bytes());assert stage==b['stage']
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
env_sha=hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha==stage['env_spec_sha256']
root.mkdir(parents=True)
for name,row in b['files'].items():
 raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];(root/name).write_bytes(raw)
protocol=json.loads((root/'protocol.json').read_bytes())
assert protocol['arm']==b['arm'] and protocol['target_hits']==[5705,4945]
assert '--checkpoint_path' not in protocol['common_arguments'] and '--frozen' not in protocol['common_arguments']
spec=json.loads(Path(protocol['native_init_spec']).read_bytes())
assert spec['use_g_supervision'] and not spec['use_selected_mask_supervision']
assert spec['use_selected_geometry_supervision']==(b['arm']=='selected_geometry')
admission=dict(status='NORMAL_PENDING_LOCKED_DEVICE_ADMISSION',
 actual_pair_preflight_complete=True,fresh_initialization_without_preflight_states=True,
 MCLN_terminal_observed=True,MCLN_terminal_and_device_released=False,MCLN_terminal_status=terminal,
 actual_pair_status=pair,actual_pair_controller_pid=b['pair']['controller_pid'],
 env_spec_sha256=env_sha,source_stage=stage,reserve_bytes=2605151860,
 source_review_sha256=b['review_sha256'],actual_pair_audit_sha256=b['audit_sha256'],
 GPU_device_checked=False,time_cst=datetime.datetime.now().astimezone().isoformat())
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock'],b['runtime'],
 '-B','-u',str(root/'normal_run_controller.py')]
with (root/'controller.log').open('wb') as stream:
 child=subprocess.Popen(command,cwd=str(base/'PV-Ground'),stdin=subprocess.DEVNULL,
  stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
assert child.poll() is None
print(json.dumps(dict(status='NORMAL_CONTROLLER_LAUNCHED_DEVICE_AND_CHILD_WITNESS_PENDING',
 arm=b['arm'],controller_pid=child.pid,controller_argv=command,root=str(root),
 admission=admission,initial_observation_after_seconds=1500,subsequent_poll_seconds=240,
 ordinary_training_child_witness_pending=True,formal_accuracy=None,full_goal_complete=False)))
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
body = dict(arm=args.arm, terminal=terminal['remote_status'], pair=pair, stage=stage, runtime=runtime,
    review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
    audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    files={name: dict(base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
           for name, raw in files.items()})
response = subprocess.run(argv, env=environment, input=json.dumps(body).encode(), stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / (prefix + '_STDOUT.json')).write_bytes(response.stdout)
(root / (prefix + '_STDERR_PRIVATE.txt')).write_bytes(response.stderr)
(root / (prefix + '_TRANSPORT_EXIT.json')).write_bytes((json.dumps(dict(exit_code=response.returncode)) + '\n').encode())
assert response.returncode == 0
record = json.loads(response.stdout)
record['launched_cst'] = datetime.datetime.now().astimezone().isoformat()
(root / (prefix + '_LAUNCH.json')).write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps({key: record[key] for key in ('status', 'arm', 'controller_pid')}))
