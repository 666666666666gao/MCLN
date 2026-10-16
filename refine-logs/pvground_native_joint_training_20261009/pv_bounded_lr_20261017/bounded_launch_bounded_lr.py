"""Launch one reviewed finite LR setting against unchanged neural source."""
import argparse
import base64
import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--run-name', required=True, choices=['lr_scale_0p1', 'lr_scale_0p3'])
args = parser.parse_args()
root = Path(__file__).resolve().parent
prior = root.parent / 'coarse_endpoint_training_20261014'
prefix = 'NORMAL_' + args.run_name
assert not (root / (prefix + '_LAUNCH.json')).exists()
review_path = root / 'source_review/SOURCE_REVIEW_COMPLETE.json'
review = json.loads(review_path.read_bytes())
assert review['status'] == 'PASS_SOURCE_ONLY' and review['unresolved_blocker_count'] == 0
for name in ('launch_bounded_lr.py', 'normal_bounded_lr_controller.py', 'observe_bounded_lr.py',
             'stage_bounded_lr_source.py', 'source_overlay/main_utils.py', args.run_name + '_protocol.json'):
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == review['verified_source_sha256'][name]
audit_path = prior / 'normal_pair_review_20261017/EXPERIMENT_AUDIT.json'
audit = json.loads(audit_path.read_bytes())
assert audit['execution_scope'] == 'COMPLETE_MATCHED_ENDPOINT_NORMAL_PAIR'
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
observations = {}
for arm, name in [('final_duplicate', 'OBS_final_duplicate_e2_validation_expected_terminal_INTAKE.json'),
                  ('coarse_endpoint', 'OBS_coarse_endpoint_e2_validation_final_INTAKE.json')]:
    path = prior / name
    assert audit['audited_input_hashes'][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    observed = json.loads(path.read_bytes())
    assert not observed['controller_alive'] and not observed['child_alive']
    assert observed['remote_status']['status'] == 'complete' and observed['remote_status']['exit_code'] == 0
    observations[arm] = observed['remote_status']
protocol = json.loads((root / (args.run_name + '_protocol.json')).read_bytes())
assert protocol['run_name'] == args.run_name and protocol['target_hits'] == [5724, 4945]
scale = {'lr_scale_0p1': .1, 'lr_scale_0p3': .3}[args.run_name]
assert protocol['learning_rate_scale'] == scale
original = json.loads((prior / 'runtime_preparation/coarse_endpoint_protocol.json').read_bytes())
expected_args = list(original['common_arguments'])
for flag in ('--lr', '--lr_backbone', '--lr_modules', '--lr_span'):
    index = expected_args.index(flag) + 1
    assert Decimal(protocol['common_arguments'][index]) == Decimal(expected_args[index]) * Decimal(str(scale))
    expected_args[index] = protocol['common_arguments'][index]
assert protocol['common_arguments'] == expected_args
assert protocol['native_init_spec'] == original['native_init_spec']
stage = json.loads((root / 'SOURCE_STAGE_RECEIPT.json').read_bytes())
assert protocol['model_source'] == stage['model_sources']['coarse_endpoint']
pair = json.loads((prior / 'OBS_preflight_actual_pair_terminal_expected_INTAKE.json').read_bytes())
lower = json.loads((root.parent / 'OBS_lower_span_lr_e8_complete_INTAKE.json').read_bytes())
files = {'normal_run_controller.py': (root / 'normal_bounded_lr_controller.py').read_bytes(),
         'protocol.json': (root / (args.run_name + '_protocol.json')).read_bytes()}
code = r'''import base64,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path('/root/autodl-tmp/pvground_bounded_lr_20261017')/b['run_name']
assert not root.exists()
prior_base=Path('/root/autodl-tmp/pvground_matched_endpoint_20261014')
source_base=Path('/root/autodl-tmp/pvground_bounded_lr_20261017')
for arm,controller,child in [('final_duplicate',741474,741477),('coarse_endpoint',792357,792360)]:
 status=json.loads((prior_base/'normal'/arm/'normal_status.json').read_bytes())
 assert status==b['completed'][arm] and status['status']=='complete' and status['exit_code']==0
 assert not Path('/proc/'+str(controller)).exists() and not Path('/proc/'+str(child)).exists()
stage=json.loads((source_base/'source_stage_manifest.json').read_bytes());assert stage==b['stage']
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
env_sha=hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha==stage['env_spec_sha256']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
reserve=2605151860
assert shutil.disk_usage(source_base).free>reserve
root.mkdir(parents=True)
for name,item in b['files'].items():
 raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 (root/name).write_bytes(raw)
protocol=json.loads((root/'protocol.json').read_bytes())
assert protocol['model_source']==stage['model_sources']['coarse_endpoint']
assert '--checkpoint_path' not in protocol['common_arguments'] and '--frozen' not in protocol['common_arguments']
admission=dict(status='NORMAL_PENDING_LOCKED_DEVICE_ADMISSION',actual_pair_preflight_complete=True,
 fresh_initialization_without_preflight_states=True,prior_strategy_terminal_observed=True,
 prior_strategy_terminal_status=b['lower']['remote_status'],prior_strategy_controller_pid=432868,
 actual_pair_status=b['pair']['remote_status'],actual_pair_controller_pid=b['pair']['controller_pid'],
 completed_normal_pair_statuses=b['completed'],env_spec_sha256=env_sha,source_stage=stage,reserve_bytes=reserve,
 source_review_sha256=b['review_sha256'],actual_pair_audit_sha256=b['audit_sha256'],GPU_device_checked=False,
 time_cst=datetime.datetime.now().astimezone().isoformat())
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock'],b['runtime'],'-B','-u',str(root/'normal_run_controller.py')]
with (root/'controller.log').open('wb') as out:
 child=subprocess.Popen(command,cwd=protocol['model_source'],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
assert child.poll() is None
print(json.dumps(dict(status='NORMAL_CONTROLLER_LAUNCHED_DEVICE_AND_CHILD_WITNESS_PENDING',
 run_name=b['run_name'],controller_pid=child.pid,controller_argv=command,root=str(root),
 admission=admission,initial_observation_after_seconds=3300,subsequent_poll_seconds=240,
 ordinary_training_child_witness_pending=True,formal_accuracy=None,full_goal_complete=False)))
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
body = dict(run_name=args.run_name, completed=observations, stage=stage, pair=pair, lower=lower, runtime=runtime,
            review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
            audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            files={name: dict(base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
                   for name, raw in files.items()})
reply = subprocess.run(ssh, env=environment, input=json.dumps(body).encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / (prefix + '_STDOUT.json')).write_bytes(reply.stdout)
(root / (prefix + '_STDERR_PRIVATE.txt')).write_bytes(reply.stderr)
(root / (prefix + '_TRANSPORT_EXIT.json')).write_text(json.dumps(dict(exit_code=reply.returncode)) + '\n')
assert reply.returncode == 0
record = json.loads(reply.stdout)
record['launched_cst'] = datetime.datetime.now().astimezone().isoformat()
(root / (prefix + '_LAUNCH.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: record[key] for key in ('status', 'run_name', 'controller_pid')}), flush=True)
