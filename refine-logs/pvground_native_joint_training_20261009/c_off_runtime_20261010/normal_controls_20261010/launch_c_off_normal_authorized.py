"""Launch only the planned three-epoch C-off arm after its actual native M0."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
previous = root.parent
project = previous.parent
assert not (root / 'NORMAL_LAUNCH.json').exists()
intake_path = root / 'ENGINEERING_INTAKE.json'
intake = json.loads(intake_path.read_bytes())
assert not intake['controller_alive'] and intake['remote_status']['status'] == 'complete'
receipt_path = root / 'actual_engineering' / 'witness' / 'NATIVE_M0_RECEIPT.json'
receipt = json.loads(receipt_path.read_bytes())
assert receipt['actual_updates'] == 2 and receipt['full_recovery_exact']
assert receipt['initial_model_state_exactly_matches_normal_E0']
assert receipt['selected_mask_supervision_disabled'] and receipt['formal_accuracy'] is None
actual_path = root / 'actual_review' / 'EXPERIMENT_AUDIT.json'
actual = json.loads(actual_path.read_bytes())
assert actual['execution_scope'] == 'ACTUAL_NATIVE_C_OFF_PREFLIGHT'
assert actual['verdict'] in ('PASS', 'WARN') and not actual['blocking_findings']
for path in (intake_path, receipt_path):
    assert actual['audited_input_hashes'][str(path.resolve())] == hashlib.sha256(path.read_bytes()).hexdigest()
review_path = root / 'normal_source_review' / 'SOURCE_REVIEW.json'
review = json.loads(review_path.read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
names = ['normal_joint_controller.py', 'NORMAL_NATIVE_RUN_PROTOCOL.json', 'init.json']
files = {name: (root / name).read_bytes() for name in names}
for path in [root / name for name in names] + [Path(__file__)]:
    assert review['audited_input_hashes'][str(path.resolve())] == hashlib.sha256(path.read_bytes()).hexdigest()
assert len(intake['weights']) == 1
weight = intake['weights'][0]
assert weight['bytes'] == receipt['checkpoint_bytes'] and weight['sha256'] == receipt['checkpoint_sha256']
port = json.loads((previous / 'NATIVE_SOURCE_PORT.json').read_bytes())
launch = json.loads((root / 'ENGINEERING_LAUNCH.json').read_bytes())
code = r'''import base64,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin)
root=Path('/root/autodl-tmp/pvground_native_joint_controls_20261010/selected_mask_off')
source=Path(b['port']['model_source']); engineering=root/'engineering'
assert root.is_dir() and not (root/'normal_status.json').exists() and not (root/'logs').exists()
assert not Path('/proc/'+str(b['engineering_pid'])).exists()
assert json.loads((engineering/'engineering_status.json').read_bytes())['status']=='complete'
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==b['port']['env_spec_sha256']
for name,row in b['port']['files'].items():
 assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
spec=json.loads((root/'init.json').read_bytes())
assert spec['use_g_supervision'] and not spec['use_selected_mask_supervision']
assert spec['span_source_mode']=='extremal_support' and not spec['new_span_output_zero_initialized']
for key in ('official_checkpoint','g_checkpoint','support_checkpoint','span_checkpoint'):
 h=hashlib.sha256()
 with Path(spec[key]).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==spec[key+'_sha256']
for name,row in b['files'].items():
 raw=base64.b64decode(row['base64'])
 assert hashlib.sha256(raw).hexdigest()==row['sha256'] and (root/name).read_bytes()==raw
free=shutil.disk_usage(root).free
assert free+b['discard']['bytes']>2605151860
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
capacity=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(capacity)==1
index,name,used,total=[value.strip() for value in capacity[0].split(',')]
assert int(index)==0 and 'A100' in name and int(used)<500 and 40000<=int(total)<=45000
discard=Path(b['discard']['path'])
assert engineering.resolve() in discard.resolve().parents and discard.name=='best.pth' and not discard.is_symlink()
assert discard.stat().st_size==b['discard']['bytes']
h=hashlib.sha256()
with discard.open('rb') as stream:
 for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==b['discard']['sha256']
import torch
payload=torch.load(str(discard),map_location='cpu')
assert payload['retained_metrics'] is None and len(payload['model'])==1295
assert not payload['architecture']['use_selected_mask_supervision']
del payload
discard.unlink()
retirement=dict(status='ONLY_UNSCORED_FULLY_RECOVERED_C_OFF_M0_CHECKPOINT_RETIRED',
 removed=b['discard'],actual_audit_sha256=b['actual_audit_sha256'],
 best_and_all_parent_dependencies_untouched=True,local_recovery_and_text_evidence_retained=True)
(engineering/'retirement.json').write_text(json.dumps(retirement,indent=2)+'\n')
admission=dict(status='NATIVE_NORMAL_JOINT_TRAINING_ADMITTED',
 time_cst=datetime.datetime.now().astimezone().isoformat(),source_mode='extremal_support',
 env_spec_sha256=b['port']['env_spec_sha256'],helpers={k:v['sha256'] for k,v in b['files'].items()},
 helper_files={k:v['sha256'] for k,v in b['files'].items()},actual_M0_complete_and_audited=True,
 actual_audit_sha256=b['actual_audit_sha256'],source_review_sha256=b['source_review_sha256'],
 engineering_intake_sha256=b['engineering_intake_sha256'],actual_receipt_sha256=b['receipt_sha256'],
 preflight_states_not_used=True,retained_span_init_sha256=spec['span_checkpoint_sha256'],
 initial1295_tensors_equal_normal_E0=True,selected_mask_supervision_disabled=True,
 parent_pretraining_updates_disclosed=True,data_free_bytes=shutil.disk_usage(root).free,
 normal_reserve_bytes=2605151860,prior_metrics_not_assumed_to_reproduce_in_native_pipeline=True)
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock'],b['runtime'],'-B','-u',str(root/'normal_joint_controller.py')]
with (root/'controller.log').open('wb') as stream:
 process=subprocess.Popen(command,cwd=str(source),stdin=subprocess.DEVNULL,
  stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
assert process.poll() is None
print(json.dumps(dict(admission=admission,controller_pid=process.pid,controller_argv=command,
 root=str(root),model_source=str(source),retirement=retirement)))
'''
witness = json.loads((project / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
variables = dict(os.environ)
variables.update(SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
                 SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
        '-o', 'StrictHostKeyChecking=yes', '-o',
        'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
        '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
        'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
data = dict(port=port, engineering_pid=launch['controller_pid'], runtime=runtime, discard=weight,
            actual_audit_sha256=hashlib.sha256(actual_path.read_bytes()).hexdigest(),
            source_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
            engineering_intake_sha256=hashlib.sha256(intake_path.read_bytes()).hexdigest(),
            receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            files={name:dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
                   for name,raw in files.items()})
response = subprocess.run(argv, env=variables, input=json.dumps(data).encode(), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'NORMAL_LAUNCH_STDOUT.json').write_bytes(response.stdout)
(root / 'NORMAL_LAUNCH_STDERR.txt').write_bytes(response.stderr)
(root / 'NORMAL_LAUNCH_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode)) + '\n')
assert response.returncode == 0, 'Inspect preserved NORMAL_LAUNCH_STDERR.txt and exact remote state before repair'
record = json.loads(response.stdout)
record.update(status='C_OFF_ORDINARY_JOINT_TRAINING_CONTROLLER_LAUNCHED_NOT_COMPLETED',
              time_cst=datetime.datetime.now().astimezone().isoformat(),formal_accuracy=None,
              first_observation_seconds=18000,later_poll_seconds=240,planned_epochs=3,
              ordinary_training_child_witness_pending=True,full_goal_complete=False)
(root / 'NORMAL_LAUNCH.json').write_text(json.dumps(record,indent=2) + '\n')
print(json.dumps(record),flush=True)
