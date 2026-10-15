"""Launch the serial pair only after an actual lower-span terminal observation."""
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
parser.add_argument('--strategy-terminal-observation', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
observation = json.loads(Path(args.strategy_terminal_observation).read_bytes())
assert observation['stage'] == 'lower_span_lr' and observation['controller_pid'] == 432868
assert not observation['controller_alive'] and not observation['child_alive']
assert observation['remote_status']['status'] == 'complete' and observation['remote_status']['exit_code'] == 0
assert observation['remote_status']['child_pid'] == 432871
stage = json.loads((root / 'isolated_source_stage_20261014/SOURCE_STAGE_RECEIPT.json').read_bytes())
assert stage['source_file_count'] == 240 and not stage['active_source_changed']
review_path = root / 'review_execution/EXECUTION_REVIEW_COMPLETE.json'
review = json.loads(review_path.read_bytes())
assert review['status'] == 'PASS_SOURCE_ONLY' and review['unresolved_blocker_count'] == 0
assert review['verified_source_sha256'][Path(__file__).name] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert not (root / 'PREFLIGHT_LAUNCH.json').exists()
code = r'''import datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);base=Path('/root/autodl-tmp/pvground_matched_endpoint_20261014');root=base/'preflight'
assert not root.exists()
terminal=json.loads(Path('/root/autodl-tmp/pvground_span_lr_20261013/normal/lower_span_lr/normal_status.json').read_bytes())
assert terminal==b['terminal'] and terminal['status']=='complete' and terminal['exit_code']==0
assert terminal['child_pid']==432871
assert not Path('/proc/432868').exists() and not Path('/proc/432871').exists()
stage=json.loads((base/'source_stage_manifest.json').read_bytes());assert stage==b['stage']
assert stage['source_file_count']==240
for row in stage['source_files'].values():
 raw=Path(row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
for name,row in stage['assets'].items():
 raw=(base/name).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
env_sha=hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha==stage['env_spec_sha256']
reserve=2*846014332+2*67108864
assert shutil.disk_usage(base).free>reserve
root.mkdir()
admission=dict(status='MATCHED_ENDPOINT_PREFLIGHT_ADMITTED',runtime=b['runtime'],
 env_spec_sha256=env_sha,source_stage=stage,prior_strategy_terminal_status=terminal,
 prior_strategy_controller_pid=432868,reserve_bytes=reserve,
 helper_sha256={name:row['sha256'] for name,row in stage['assets'].items()},
 source_review_sha256=b['review_sha256'],formal_training_admitted=False,
 time_cst=datetime.datetime.now().astimezone().isoformat())
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
(root/'matched_endpoint_preflight_pair_controller.py').write_bytes((base/'matched_endpoint_preflight_pair_controller.py').read_bytes())
command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock'],b['runtime'],
 '-B','-u',str(root/'matched_endpoint_preflight_pair_controller.py')]
with (root/'controller.log').open('wb') as stream:
 child=subprocess.Popen(command,cwd=stage['model_sources']['final_duplicate'],stdin=subprocess.DEVNULL,
  stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
assert child.poll() is None
print(json.dumps(dict(status='MATCHED_ENDPOINT_PREFLIGHT_CONTROLLER_LAUNCHED_CHILD_WITNESS_PENDING',
 controller_pid=child.pid,controller_argv=command,root=str(root),admission=admission,
 estimated_seconds=1300,initial_observation_after_seconds=1120,subsequent_poll_seconds=240,
 actual_preflight_complete=False,formal_accuracy=None,normal_training_started=False)))
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
body = dict(terminal=observation['remote_status'], stage=stage, runtime=runtime,
    review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest())
response = subprocess.run(argv, env=environment, input=json.dumps(body).encode('utf-8'),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'PREFLIGHT_STDOUT.json').write_bytes(response.stdout)
(root / 'PREFLIGHT_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(root / 'PREFLIGHT_TRANSPORT_EXIT.json').write_bytes((json.dumps(dict(exit_code=response.returncode)) + '\n').encode('utf-8'))
assert response.returncode == 0
record = json.loads(response.stdout)
record['launched_cst'] = datetime.datetime.now().astimezone().isoformat()
(root / 'PREFLIGHT_LAUNCH.json').write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))
print(json.dumps({key: record[key] for key in ('status', 'controller_pid', 'actual_preflight_complete', 'normal_training_started')}), flush=True)
