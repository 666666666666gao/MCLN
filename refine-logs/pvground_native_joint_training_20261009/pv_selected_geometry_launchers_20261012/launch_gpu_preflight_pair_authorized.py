"""After the original MCLN task actually closes, launch the serial native pair."""
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
parser.add_argument('--mcln-terminal-observation', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
assert not (root / 'PREFLIGHT_PAIR_LAUNCH.json').exists()
observation = json.loads(Path(args.mcln_terminal_observation).read_bytes())
assert observation['stage'] == 'normal_training'
assert not observation['controller_alive'] and not observation['child_alive']
assert observation['remote_status']['status'] == 'complete' and observation['remote_status']['exit_code'] == 0
review_path = root / 'source_review/SOURCE_REVIEW_LAUNCHERS.json'
review = json.loads(review_path.read_bytes())
assert review['scope'] == 'SOURCE_ONLY' and review['source_preparation_pass']
assert not review['scoped_blockers']
stage = json.loads((root / 'SOURCE_STAGE_R3_RECEIPT.json').read_bytes())
assert stage['status'] == 'SOURCE_AND_LAUNCHERS_STAGED_GPU_PENDING'
helpers = {}
for name in ('gpu_preflight_pair_controller.py', 'native_selected_geometry_preflight.py',
             'native_control_protocol.json', 'selected_geometry_protocol.json',
             'native_control_initialization.json', 'selected_geometry_initialization.json'):
    raw = (root / name).read_bytes()
    helpers[name] = dict(base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
code = r'''import base64,datetime,hashlib,json,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);base=Path('/root/autodl-tmp/pvground_deployment_geometry_training_20261012');root=base/'preflight'
assert not root.exists()
terminal=json.loads(Path('/root/autodl-tmp/mcln_portable_best_transfer_20261012/normal_training/status.json').read_bytes())
assert terminal==b['terminal'] and terminal['status']=='complete' and terminal['exit_code']==0
assert not Path('/proc/271902').exists() and not Path('/proc/271903').exists()
stage=json.loads((base/'source_stage_manifest.json').read_bytes());assert stage==b['stage']
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==stage['env_spec_sha256']
for name,row in b['helpers'].items():
 raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 if name=='gpu_preflight_pair_controller.py':(base/name).write_bytes(raw)
 else:assert (base/name).read_bytes()==raw
root.mkdir()
# Two unscored checkpoints coexist; estimate includes container/log reserve.
admission=dict(status='SELECTED_GEOMETRY_PREFLIGHT_ADMITTED',runtime=b['runtime'],
 env_spec_sha256=stage['env_spec_sha256'],source_stage=stage,MCLN_terminal_status=terminal,
 helper_sha256={name:row['sha256'] for name,row in b['helpers'].items()},
 reserve_bytes=2*846014332+2*67108864,source_review_sha256=b['review_sha256'],
 formal_training_admitted=False,time_cst=datetime.datetime.now().astimezone().isoformat())
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
(root/'gpu_preflight_pair_controller.py').write_bytes((base/'gpu_preflight_pair_controller.py').read_bytes())
command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock'],b['runtime'],
 '-B','-u',str(root/'gpu_preflight_pair_controller.py')]
with (root/'controller.log').open('wb') as stream:
 child=subprocess.Popen(command,cwd=str(base/'PV-Ground'),stdin=subprocess.DEVNULL,
  stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
assert child.poll() is None
print(json.dumps(dict(status='PAIRED_NATIVE_GPU_PREFLIGHT_LAUNCHED_NOT_COMPLETED',
 controller_pid=child.pid,controller_argv=command,root=str(root),admission=admission,
 estimated_seconds=1300,initial_observation_after_seconds=1120,subsequent_poll_seconds=240,
 formal_accuracy=None,normal_training_started=False)))
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
body = dict(terminal=observation['remote_status'], stage=stage, helpers=helpers, runtime=runtime,
            review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest())
response = subprocess.run(argv, env=environment, input=json.dumps(body).encode(), stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'PREFLIGHT_PAIR_STDOUT.json').write_bytes(response.stdout)
(root / 'PREFLIGHT_PAIR_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(root / 'PREFLIGHT_PAIR_TRANSPORT_EXIT.json').write_bytes((json.dumps(dict(exit_code=response.returncode)) + '\n').encode())
assert response.returncode == 0
record = json.loads(response.stdout)
record['launched_cst'] = datetime.datetime.now().astimezone().isoformat()
(root / 'PREFLIGHT_PAIR_LAUNCH.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps({key: record[key] for key in ('status', 'controller_pid', 'normal_training_started')}))
