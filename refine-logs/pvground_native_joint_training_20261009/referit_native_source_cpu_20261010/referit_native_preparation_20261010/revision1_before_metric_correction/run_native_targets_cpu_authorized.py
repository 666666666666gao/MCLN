"""One separate CPU target check; requires closed source review, no GPU query."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
parent = root.parent
audit_path = root / 'source_review/EXPERIMENT_CODE_REVIEW.json'
audit = json.loads(audit_path.read_bytes())
assert audit['verdict'].upper() in ('PASS', 'WARN') and audit['blocking_issue_count'] == 0
for name, expected in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected.removeprefix('sha256:')
attempt = root / 'cpu_execution'
assert not attempt.exists()
attempt.mkdir()
prepared = json.loads((root / 'SOURCE_ADAPTER_PREPARATION.json').read_bytes())
for name, expected in prepared['original_source_sha256'].items():
    assert hashlib.sha256((parent / 'source' / name).read_bytes()).hexdigest() == expected
files = {path.relative_to(root).as_posix():path.read_bytes()
         for leaf in (root / 'source', root / 'original') for path in leaf.rglob('*.py')}
files['CPU_INPUTS.json'] = (root / 'CPU_INPUTS.json').read_bytes()
files['check_native_targets_cpu.py'] = (root / 'check_native_targets_cpu.py').read_bytes()
file_hashes = {name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
payload = dict(root='/root/autodl-tmp/pvground_referit_native_targets_cpu_20261010',
    original_source=json.loads((root / 'CPU_INPUTS.json').read_bytes())['native_model_source'],
    original_sha256=prepared['original_source_sha256'],
    files={name:base64.b64encode(raw).decode() for name,raw in files.items()}, file_sha256=file_hashes)
code = r'''import base64,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/pvground_referit_native_targets_cpu_20261010') and not root.exists()
for name,digest in b['original_sha256'].items():
 assert hashlib.sha256((Path(b['original_source'])/name).read_bytes()).hexdigest()==digest
root.mkdir()
for name,encoded in b['files'].items():
 path=root/name;assert root in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(encoded)
 assert hashlib.sha256(raw).hexdigest()==b['file_sha256'][name]
 path.write_bytes(raw)
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
argv=['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-B','-u',str(root/'check_native_targets_cpu.py')]
started=datetime.datetime.now().astimezone().isoformat()
response=subprocess.run(argv,cwd=str(root),env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(root/'CPU_STDOUT.json').write_bytes(response.stdout);(root/'CPU_STDERR.txt').write_bytes(response.stderr)
out=dict(started_cst=started,finished_cst=datetime.datetime.now().astimezone().isoformat(),exit_code=response.returncode,
 argv=argv,cuda_visible_devices='',current_training_status_reads=0,GPU_calls=0,
 stdout_base64=base64.b64encode(response.stdout).decode(),stderr_base64=base64.b64encode(response.stderr).decode())
(root/'CPU_EXIT.json').write_text(json.dumps({k:v for k,v in out.items() if not k.endswith('_base64')},indent=2)+'\n')
print(json.dumps(out));sys.exit(response.returncode)
'''
witness = json.loads((parent.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
response = subprocess.run(argv, input=json.dumps(payload).encode(), env=environment,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(attempt / 'CPU_REMOTE_RAW_STDOUT.json').write_bytes(response.stdout)
(attempt / 'CPU_REMOTE_RAW_STDERR.txt').write_bytes(response.stderr)
(attempt / 'CPU_TRANSPORT_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode)) + '\n', encoding='utf-8')
assert response.stdout, 'No CPU receipt; preserve transport failure and verify remote state before another attempt'
receipt = json.loads(response.stdout)
(attempt / 'CPU_STDOUT.json').write_bytes(base64.b64decode(receipt.pop('stdout_base64')))
(attempt / 'CPU_STDERR.txt').write_bytes(base64.b64decode(receipt.pop('stderr_base64')))
receipt.update(source_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    purpose='synthetic_CE_and_mask_targets_only', full_PV_or_author_state_checked=False,
    normal_training_queried=False, Nr_Sr_training_launched=False)
(attempt / 'CPU_EXECUTION.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
assert response.returncode == receipt['exit_code'] == 0
result = json.loads((attempt / 'CPU_STDOUT.json').read_bytes())
(attempt / 'CPU_PARSED_RESULT.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=result['status'], exit_code=receipt['exit_code'],
    real_dataset_rows=result['real_dataset_rows'], normal_training_status_reads=0, GPU_calls=0)), flush=True)
