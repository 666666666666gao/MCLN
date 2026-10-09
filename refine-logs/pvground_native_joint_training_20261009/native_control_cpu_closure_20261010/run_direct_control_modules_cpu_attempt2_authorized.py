"""One isolated CPU module check, gated by source audit; no active-run query."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
controls = root / 'native_direct_controls_20261010'
attempt_dir = controls / 'cpu_transport_attempt2'
proof = json.loads((root / 'read_only_static_diagnostic_20261010_attempt1/REMOTE_STATIC_WITNESS.json').read_bytes())
assert proof['CPU_root_exists'] is False and not proof['CPU_receipt_files']
audit_path = controls / 'actual_source_review/EXPERIMENT_AUDIT.json'
audit = json.loads(audit_path.read_bytes())
assert audit['verdict'].upper() in ('PASS', 'WARN') and audit['blocking_issue_count'] == 0
assert audit['execution_scope'] == 'ISOLATED_NATIVE_CONTROL_SOURCE_NOT_LAUNCHED'
for name, expected in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected.removeprefix('sha256:')
assert not (attempt_dir / 'CPU_MODULE_EXECUTION.json').exists()
prepared = json.loads((controls / 'DIRECT_CONTROL_PREPARATION.json').read_bytes())
port = json.loads((root / 'NATIVE_SOURCE_PORT.json').read_bytes())
original_sha = dict(prepared['original_source_sha256'])
original_sha['whole_mask_range.py'] = port['files']['whole_mask_range.py']['sha256']
files = {}
for name, expected in prepared['prepared_sha256'].items():
    raw = (controls / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    files[name] = raw
files['check_direct_control_modules_cpu.py'] = (controls / 'check_direct_control_modules_cpu.py').read_bytes()
file_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
remote = '/root/autodl-tmp/pvground_native_direct_controls_cpu_20261010'
bundle = dict(original_source=port['model_source'], original_sha256=original_sha,
              prepared_sha256=file_hashes)
assert bundle == json.loads((controls / 'CPU_BUNDLE.json').read_bytes())
payload = dict(root=remote, files={name: base64.b64encode(raw).decode() for name, raw in files.items()},
               bundle=bundle)
code = r'''import base64,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/pvground_native_direct_controls_cpu_20261010') and not root.exists()
root.mkdir()
for name,encoded in b['files'].items():
 path=root/name;assert root in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(encoded)
 assert hashlib.sha256(raw).hexdigest()==b['bundle']['prepared_sha256'][name]
 path.write_bytes(raw)
bundle=root/'CPU_BUNDLE.json';bundle.write_text(json.dumps(b['bundle'],indent=2)+'\n')
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
argv=['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-B','-u',str(root/'check_direct_control_modules_cpu.py'),str(bundle)]
started=datetime.datetime.now().astimezone().isoformat()
response=subprocess.run(argv,cwd=str(root),env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(root/'CPU_STDOUT.json').write_bytes(response.stdout);(root/'CPU_STDERR.txt').write_bytes(response.stderr)
out=dict(started_cst=started,finished_cst=datetime.datetime.now().astimezone().isoformat(),exit_code=response.returncode,
 argv=argv,cuda_visible_devices='',current_training_status_reads=0,GPU_calls=0,
 stdout_base64=base64.b64encode(response.stdout).decode(),stderr_base64=base64.b64encode(response.stderr).decode())
(root/'CPU_EXIT.json').write_text(json.dumps({k:v for k,v in out.items() if not k.endswith('_base64')},indent=2)+'\n')
print(json.dumps(out));sys.exit(response.returncode)
'''
witness = json.loads((root.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
response = subprocess.run(argv, input=json.dumps(payload).encode(), env=environment,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(attempt_dir / 'CPU_REMOTE_RAW_STDOUT.json').write_bytes(response.stdout)
(attempt_dir / 'CPU_REMOTE_RAW_STDERR.txt').write_bytes(response.stderr)
(attempt_dir / 'CPU_TRANSPORT_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode)) + '\n', encoding='utf-8')
assert response.stdout, 'No CPU remote receipt; preserve transport failure without restarting'
remote_receipt = json.loads(response.stdout)
(attempt_dir / 'CPU_MODULE_STDOUT.json').write_bytes(base64.b64decode(remote_receipt.pop('stdout_base64')))
(attempt_dir / 'CPU_MODULE_STDERR.txt').write_bytes(base64.b64decode(remote_receipt.pop('stderr_base64')))
remote_receipt.update(source_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    original_source_unchanged=True, current_training_queried=False,
    same_source_full_PV_GPU_preflight_pending=True, control_training_launched=False)
(attempt_dir / 'CPU_MODULE_EXECUTION.json').write_text(json.dumps(remote_receipt, indent=2) + '\n', encoding='utf-8')
assert response.returncode == remote_receipt['exit_code'] == 0
result = json.loads((attempt_dir / 'CPU_MODULE_STDOUT.json').read_bytes())
(attempt_dir / 'CPU_MODULE_WITNESS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=result['status'], exit_code=remote_receipt['exit_code'],
    formal_accuracy=result['formal_accuracy'], current_training_status_reads=0, GPU_calls=0)))
