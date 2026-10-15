"""Launch one fresh normal arm only after the actual serial native sanity pair."""
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
parser.add_argument('--arm', required=True, choices=['final_duplicate', 'coarse_endpoint'])
parser.add_argument('--pair-intake', required=True)
parser.add_argument('--strategy-terminal-observation', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
prefix = 'NORMAL_' + args.arm
assert not (root / (prefix + '_LAUNCH.json')).exists()
pair_path = Path(args.pair_intake)
pair = json.loads(pair_path.read_bytes())
assert not pair['controller_alive'] and not pair['child_alive']
assert pair['remote_status']['status'] == 'complete' and pair['remote_status']['exit_code'] == 0
assert pair['remote_status']['actual_total_native_updates'] == 4
assert pair['remote_status']['same_augmented_inputs_verified'] and pair['remote_status']['same_initial_model_state_verified']
assert set(pair['receipts']) == {'final_duplicate', 'coarse_endpoint'}
for arm, receipt in pair['receipts'].items():
    assert receipt['source_mode'] == arm and receipt['actual_updates'] == 2
    assert receipt['full_recovery_exact'] and receipt['full_state_tensors'] == 1295
    assert not receipt['preflight_weights_used_for_formal_training'] and receipt['formal_accuracy'] is None
    assert receipt['selected_geometry_supervision'] and receipt['selected_mask_supervision_disabled']
assert pair['receipts']['final_duplicate']['initial_model_state_sha256'] == pair['receipts']['coarse_endpoint']['initial_model_state_sha256']
assert pair['receipts']['final_duplicate']['input_batch_sha256'] == pair['receipts']['coarse_endpoint']['input_batch_sha256']
audit_path = root / 'actual_pair_review/EXPERIMENT_AUDIT.json'
audit = json.loads(audit_path.read_bytes())
assert audit['execution_scope'] == 'ACTUAL_MATCHED_ENDPOINT_NATIVE_PREFLIGHT_PAIR'
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
assert audit['audited_input_hashes'][str(pair_path.resolve())] == hashlib.sha256(pair_path.read_bytes()).hexdigest()
terminal = json.loads(Path(args.strategy_terminal_observation).read_bytes())
assert terminal['controller_pid'] == 432868 and terminal['remote_status']['child_pid'] == 432871
assert not terminal['controller_alive'] and not terminal['child_alive']
assert terminal['remote_status']['status'] == 'complete' and terminal['remote_status']['exit_code'] == 0
stage = json.loads((root / 'isolated_source_stage_20261014/SOURCE_STAGE_RECEIPT.json').read_bytes())
review_path = root / 'review_execution/EXECUTION_REVIEW_COMPLETE.json'
review = json.loads(review_path.read_bytes())
assert review['status'] == 'PASS_SOURCE_ONLY' and review['unresolved_blocker_count'] == 0
assert review['verified_source_sha256'][Path(__file__).name] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
protocol_source = root / 'runtime_preparation' / (args.arm + '_protocol.json')
protocol = json.loads(protocol_source.read_bytes())
assert protocol['arm'] == args.arm and protocol['target_hits'] == [5705, 4945]
protocol.update(status='SOURCE_STAGED_NOT_GPU_ADMITTED_OR_QUEUED',
    source_local_preparation_only=False, source_stage_only=True)
protocol_bytes = (json.dumps(protocol, indent=2) + '\n').encode('utf-8')
assert hashlib.sha256(protocol_bytes).hexdigest() == stage['assets'][args.arm + '_protocol.json']['sha256']
files = {'normal_run_controller.py': (root / 'launch_preparation/normal_matched_endpoint_controller.py').read_bytes(),
    'protocol.json': protocol_bytes}
code = r'''import base64,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);base=Path('/root/autodl-tmp/pvground_matched_endpoint_20261014');root=base/'normal'/b['arm']
assert not root.exists()
terminal=json.loads(Path('/root/autodl-tmp/pvground_span_lr_20261013/normal/lower_span_lr/normal_status.json').read_bytes())
assert terminal==b['terminal'] and terminal['status']=='complete' and terminal['exit_code']==0
assert not Path('/proc/432868').exists() and not Path('/proc/432871').exists()
pair=json.loads((base/'preflight/status.json').read_bytes());assert pair==b['pair']['remote_status']
assert pair['status']=='complete' and pair['exit_code']==0
assert not Path('/proc/'+str(b['pair']['controller_pid'])).exists()
for arm,receipt in b['pair']['receipts'].items():
 assert json.loads((base/'preflight'/arm/'witness/NATIVE_M0_RECEIPT.json').read_bytes())==receipt
 assert receipt['matched_endpoint']=={'final_duplicate':'final','coarse_endpoint':'coarse'}[arm]
stage=json.loads((base/'source_stage_manifest.json').read_bytes());assert stage==b['stage']
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
env_sha=hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha==stage['env_spec_sha256']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
reserve=2605151860
assert shutil.disk_usage(base).free>reserve
root.mkdir(parents=True)
for name,row in b['files'].items():
 raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];(root/name).write_bytes(raw)
protocol=json.loads((root/'protocol.json').read_bytes())
assert protocol['model_source']==stage['model_sources'][b['arm']]
assert (root/'protocol.json').read_bytes()==(base/(b['arm']+'_protocol.json')).read_bytes()
assert '--checkpoint_path' not in protocol['common_arguments'] and '--frozen' not in protocol['common_arguments']
assert hashlib.sha256((base/'normal_matched_endpoint_controller.py').read_bytes()).hexdigest()==b['files']['normal_run_controller.py']['sha256']
admission=dict(status='NORMAL_PENDING_LOCKED_DEVICE_ADMISSION',actual_pair_preflight_complete=True,
 fresh_initialization_without_preflight_states=True,prior_strategy_terminal_observed=True,
 prior_strategy_terminal_status=terminal,prior_strategy_controller_pid=432868,
 actual_pair_status=pair,actual_pair_controller_pid=b['pair']['controller_pid'],
 env_spec_sha256=env_sha,source_stage=stage,reserve_bytes=reserve,
 source_review_sha256=b['review_sha256'],actual_pair_audit_sha256=b['audit_sha256'],GPU_device_checked=False,
 time_cst=datetime.datetime.now().astimezone().isoformat())
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock'],b['runtime'],
 '-B','-u',str(root/'normal_run_controller.py')]
with (root/'controller.log').open('wb') as stream:
 child=subprocess.Popen(command,cwd=protocol['model_source'],stdin=subprocess.DEVNULL,
  stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
assert child.poll() is None
print(json.dumps(dict(status='NORMAL_CONTROLLER_LAUNCHED_DEVICE_AND_CHILD_WITNESS_PENDING',
 arm=b['arm'],controller_pid=child.pid,controller_argv=command,root=str(root),admission=admission,
 initial_observation_after_seconds=1500,subsequent_poll_seconds=240,
 ordinary_training_child_witness_pending=True,formal_accuracy=None,full_goal_complete=False)))
'''
witness = json.loads((root.parent.parent.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
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
    files={name: dict(base64=base64.b64encode(raw).decode('ascii'), sha256=hashlib.sha256(raw).hexdigest())
        for name, raw in files.items()})
response = subprocess.run(argv, env=environment, input=json.dumps(body).encode('utf-8'),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / (prefix + '_STDOUT.json')).write_bytes(response.stdout)
(root / (prefix + '_STDERR_PRIVATE.txt')).write_bytes(response.stderr)
(root / (prefix + '_TRANSPORT_EXIT.json')).write_bytes((json.dumps(dict(exit_code=response.returncode)) + '\n').encode('utf-8'))
assert response.returncode == 0
record = json.loads(response.stdout)
record['launched_cst'] = datetime.datetime.now().astimezone().isoformat()
(root / (prefix + '_LAUNCH.json')).write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))
print(json.dumps({key: record[key] for key in ('status', 'arm', 'controller_pid')}), flush=True)
