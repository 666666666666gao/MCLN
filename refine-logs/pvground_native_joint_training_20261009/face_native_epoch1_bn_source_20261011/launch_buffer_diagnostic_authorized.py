"""Explicit one-stage SSH admission; never run before the scheduled closeout."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=('sanity', 'full'), required=True)
options = parser.parse_args()
root = Path(__file__).resolve().parent
project = root.parents[2]
review = json.loads((root / 'source_review/SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
names = ['buffer_diagnostic_controller.py', 'recover_face_normal_terminal.py',
         'paired_native_buffer_replay.py', 'batchnorm_buffer_intervention.py']
for path in [root / name for name in names] + [Path(__file__), root / 'EXPERIMENT_PLAN.md']:
    assert review['audited_input_hashes'][str(path.resolve())] == hashlib.sha256(path.read_bytes()).hexdigest()
assert not (root / (options.stage + '_LAUNCH.json')).exists()
files = {name: dict(base64=base64.b64encode((root / name).read_bytes()).decode(),
                    sha256=hashlib.sha256((root / name).read_bytes()).hexdigest()) for name in names}
code = r'''import base64,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
body=json.load(sys.stdin)
base=Path('/root/autodl-tmp/pvground_face_native_gpu_v2_20261011')
normal=base/'normal/source_conditioned'
terminal=json.loads((normal/'normal_status.json').read_bytes())
assert terminal['status']=='complete' and terminal['exit_code']==0
assert [row['epoch'] for row in terminal['metrics']]==[0,1,2,3]
assert not Path('/proc/'+str(terminal['child_pid'])).exists()
protocol_path=normal/'NORMAL_NATIVE_RUN_PROTOCOL.json'
protocol=json.loads(protocol_path.read_bytes())
assert protocol['face_residual_mode']=='source_conditioned'
assert protocol['intended_updates']==13749 and protocol['intended_real_rows']==109992
latest=[row for row in terminal['weights'] if Path(row['path']).name=='latest.pth']
assert len(latest)==1
identity=latest[0];path=Path(identity['path']);digest=hashlib.sha256()
with path.open('rb') as stream:
 for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
assert path.stat().st_size==identity['bytes'] and digest.hexdigest()==identity['sha256']
source=Path(protocol['model_source'])
manifest_path=base/'source_stage_manifest.json'
manifest=json.loads(manifest_path.read_bytes())
for row in manifest['source_files'].values():
 assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
env_sha=hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
stage=body['stage'];parent=Path('/root/autodl-tmp/pvground_face_bn_diagnostic_20261011');root=parent/stage
assert stage in ('sanity','full') and not root.exists()
if stage=='full':
 prior=json.loads((parent/'sanity/diagnostic_status.json').read_bytes())
 assert prior['status']=='complete' and prior['exit_code']==0
 result=prior['result']
 assert result['scope']=='REAL_FIRST_BATCH_8_SANITY' and result['checkpoint']==identity
 assert result['full_trained_state_restored_exactly'] and result['all_non_bn_states_unchanged']
 assert result['paired_input_read_once'] and result['branch_rng_restored']
 recovery=json.loads((parent/'sanity/cold_recovery/RECOVERY_RECEIPT.json').read_bytes())
 assert recovery['status']=='FACE_NORMAL_BEST_AND_FIXED_E3_FULL_COLD_RECOVERY_COMPLETE'
 assert all(row['full_recovery_exact'] for row in recovery['checkpoints'])
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
root.mkdir(parents=True)
for name,row in body['files'].items():
 raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 (root/name).write_bytes(raw)
admission=dict(stage=stage,normal_exit_code=0,saved_epoch=3,checkpoint=identity,
 protocol=str(protocol_path),terminal=str(normal/'normal_status.json'),source=str(source),
 source_manifest=str(manifest_path),env_spec_sha256=env_sha,
 source_review_sha256=body['source_review_sha256'])
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock'],
 '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-B','-u',str(root/'buffer_diagnostic_controller.py')]
with (root/'controller.log').open('wb') as stream:
 process=subprocess.Popen(command,cwd=str(source),stdin=subprocess.DEVNULL,
  stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
assert process.poll() is None
print(json.dumps(dict(stage=stage,root=str(root),controller_pid=process.pid,
 controller_argv=command,admission=admission,launched_cst=datetime.datetime.now().astimezone().isoformat())))
'''
witness = json.loads((project / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
body = dict(stage=options.stage, files=files,
    source_review_sha256=hashlib.sha256((root / 'source_review/SOURCE_REVIEW.json').read_bytes()).hexdigest())
process = subprocess.run(argv, input=json.dumps(body).encode(), stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, env=environment, timeout=300)
(root / (options.stage + '_LAUNCH_STDERR.txt')).write_bytes(process.stderr)
assert process.returncode == 0, process.stderr.decode(errors='replace')
result = json.loads(process.stdout)
(root / (options.stage + '_LAUNCH.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
