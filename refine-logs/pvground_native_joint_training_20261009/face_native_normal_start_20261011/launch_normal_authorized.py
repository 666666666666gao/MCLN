"""Admit one normal three-epoch face arm after both actual sanity checks."""
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
parser.add_argument('--mode', choices=['source_conditioned', 'without_additional_source'], required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
face = root.parent
task = face.parent
project = task.parent
assert not (root / ('NORMAL_' + args.mode + '_LAUNCH.json')).exists()
audit_path = face / 'actual_pair_review/EXPERIMENT_AUDIT.json'
audit = json.loads(audit_path.read_bytes())
assert audit['execution_scope'] == 'ACTUAL_NATIVE_FACE_PREFLIGHT_PAIR'
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
intakes, receipts = {}, {}
for mode in ('source_conditioned', 'without_additional_source'):
    intake_path = face / ('ENGINEERING_' + mode + '_INTAKE.json')
    receipt_path = face / ('actual_engineering_' + mode) / 'witness/NATIVE_M0_RECEIPT.json'
    for path in (intake_path, receipt_path):
        assert audit['audited_input_hashes'][str(path.resolve())] == hashlib.sha256(path.read_bytes()).hexdigest()
    intake = json.loads(intake_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    assert not intake['controller_alive'] and intake['remote_status']['status'] == 'complete'
    assert intake['remote_status']['exit_code'] == 0
    assert receipt['face_residual_mode'] == mode and receipt['face_architecture_saved_and_reconstructed']
    assert receipt['actual_updates'] == 2 and receipt['actual_input_rows'] == 16
    assert receipt['full_recovery_exact'] and receipt['selected_mask_supervision_disabled']
    assert receipt['retained_1295_states_exactly_match_normal_E0']
    assert receipt['full_state_tensors'] == 1301 and receipt['restored_optimizer_states'] == 826
    assert receipt['saved_retained_metrics'] is None and receipt['formal_accuracy'] is None
    assert receipt['gradient_norms'][1]['face_internal'] > 0
    assert sum(step['face_output'] for step in receipt['gradient_norms']) > 0
    intakes[mode], receipts[mode] = intake, receipt
assert receipts['source_conditioned']['input_batch_sha256'] == receipts['without_additional_source']['input_batch_sha256']
review_path = root / 'source_review/SOURCE_REVIEW.json'
review = json.loads(review_path.read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY' and review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
names = ['normal_joint_controller.py', args.mode + '_PROTOCOL.json']
for path in [root / name for name in names] + [Path(__file__), face / (args.mode + '_init.json')]:
    assert review['audited_input_hashes'][str(path.resolve())] == hashlib.sha256(path.read_bytes()).hexdigest()
protocol = json.loads((root / (args.mode + '_PROTOCOL.json')).read_bytes())
assert protocol['intended_updates'] == 13749 and protocol['intended_real_rows'] == 109992
assert protocol['face_residual_mode'] == args.mode and not protocol['full_training_admitted']
assert not protocol['actual_normal_epoch_training_started']
remote_root = '/root/autodl-tmp/pvground_face_native_gpu_v2_20261011/normal/' + args.mode
protocol.update(status='NATIVE_FACE_NORMAL_FULL_TRAINING_ADMITTED', full_training_admitted=True,
    native_init_spec=remote_root + '/init.json')
files = dict(normal_joint_controller=(root / 'normal_joint_controller.py').read_bytes())
files = {'normal_joint_controller.py': files['normal_joint_controller'],
    'NORMAL_NATIVE_RUN_PROTOCOL.json': (json.dumps(protocol, indent=2) + '\n').encode(),
    'init.json': (face / (args.mode + '_init.json')).read_bytes()}
staging = json.loads((face / 'SOURCE_STAGE_RECEIPT.json').read_bytes())
code = r'''import base64,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);base=Path('/root/autodl-tmp/pvground_face_native_gpu_v2_20261011')
root=base/'normal'/b['mode'];source=base/'PV-Ground';assert not root.exists()
terminal=json.loads(Path('/root/autodl-tmp/pvground_native_joint_controls_20261010/selected_mask_off/normal_status.json').read_bytes())
assert terminal['status']=='complete' and terminal['exit_code']==0
for mode,intake in b['intakes'].items():
 assert not Path('/proc/'+str(intake['controller_pid'])).exists()
 engineering=base/'engineering'/mode
 assert json.loads((engineering/'engineering_status.json').read_bytes())==intake['remote_status']
 assert json.loads((engineering/'witness/NATIVE_M0_RECEIPT.json').read_bytes())==b['receipts'][mode]
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
env_sha=hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
assert json.loads((base/'source_stage_manifest.json').read_bytes())==b['staging']
for row in b['staging']['source_files'].values():
 assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
protocol=json.loads(base64.b64decode(b['files']['NORMAL_NATIVE_RUN_PROTOCOL.json']['base64']))
assert protocol['full_training_admitted'] and protocol['face_residual_mode']==b['mode']
assert protocol['native_init_spec']==str(root/'init.json') and protocol['selection_target_hits']==[5705,4945]
assert protocol['intended_updates']==13749 and protocol['intended_real_rows']==109992
spec=json.loads(base64.b64decode(b['files']['init.json']['base64']))
assert spec['face_residual_mode']==b['mode'] and spec['use_g_supervision'] and not spec['use_selected_mask_supervision']
assert spec['span_source_mode']=='extremal_support' and not spec['new_span_output_zero_initialized']
for key in ('official_checkpoint','g_checkpoint','support_checkpoint','span_checkpoint'):
 h=hashlib.sha256()
 with Path(spec[key]).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==spec[key+'_sha256']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(gpu)==1
index,name,used,total=[value.strip() for value in gpu[0].split(',')]
assert index=='0' and 'A100' in name and int(used)<500 and 40000<=int(total)<=45000
# Three complete ~842MB checkpoint copies can coexist during atomic save.
# Face states add ~0.3MB over the measured parent; keep64MiB extra space.
reserve=2610000000;free=shutil.disk_usage(base).free;assert free>reserve
root.mkdir(parents=True)
for name,row in b['files'].items():
 raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 (root/name).write_bytes(raw)
admission=dict(status='NATIVE_NORMAL_JOINT_TRAINING_ADMITTED',time_cst=datetime.datetime.now().astimezone().isoformat(),
 source_mode='extremal_support',face_residual_mode=b['mode'],env_spec_sha256=env_sha,
 helper_files={name:row['sha256'] for name,row in b['files'].items()},
 actual_M0_complete_and_audited=True,preflight_states_not_used=True,
 actual_audit_sha256=b['actual_audit_sha256'],source_review_sha256=b['source_review_sha256'],
 retained1295_tensors_equal_normal_E0=True,new_face_states=6,total_model_states=1301,
 selected_mask_supervision_disabled=True,normal_reserve_bytes=reserve,data_free_bytes=free,
 expected_updates=13749,expected_actual_rows=109992,selection_target_hits=[5705,4945],
 no_bitwise_cross_process_pair_claim=True)
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock'],b['runtime'],
 '-B','-u',str(root/'normal_joint_controller.py')]
with (root/'controller.log').open('wb') as stream:
 process=subprocess.Popen(command,cwd=str(source),stdin=subprocess.DEVNULL,stdout=stream,
  stderr=subprocess.STDOUT,start_new_session=True)
assert process.poll() is None
print(json.dumps(dict(admission=admission,controller_pid=process.pid,controller_argv=command,
 root=str(root),model_source=str(source))))
'''
witness = json.loads((project / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
body = dict(mode=args.mode, staging=staging, intakes=intakes, receipts=receipts, runtime=runtime,
    actual_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    source_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
    files={name: dict(base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest()) for name,raw in files.items()})
response = subprocess.run(argv, env=environment, input=json.dumps(body).encode(), stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
prefix = 'NORMAL_' + args.mode
(root / (prefix + '_STDOUT.json')).write_bytes(response.stdout)
(root / (prefix + '_STDERR_PRIVATE.txt')).write_bytes(response.stderr)
(root / (prefix + '_EXIT.json')).write_text(json.dumps(dict(exit_code=response.returncode)) + '\n')
assert response.returncode == 0, 'Read saved stderr; never restart merely for an observation timeout.'
record = json.loads(response.stdout)
record.update(status='FACE_ORDINARY_JOINT_TRAINING_LAUNCHED_NOT_COMPLETED',
    time_cst=datetime.datetime.now().astimezone().isoformat(), formal_accuracy=None,
    first_observation_seconds=17100, later_poll_seconds=240, planned_epochs=3,
    ordinary_training_child_witness_pending=True, full_goal_complete=False)
(root / (prefix + '_LAUNCH.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(status=record['status'], controller_pid=record['controller_pid'],
    face_residual_mode=args.mode, formal_accuracy=None)))
