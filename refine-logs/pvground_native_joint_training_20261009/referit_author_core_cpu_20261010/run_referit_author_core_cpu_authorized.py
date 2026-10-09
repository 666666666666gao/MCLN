"""One bounded CPU factory/load check; no query or mutation of active training."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
prepared = root/'referit_author_core_20261010'
audit_path = prepared/'source_review/EXPERIMENT_CODE_REVIEW_R3.json'
audit = json.loads(audit_path.read_bytes())
assert audit['verdict'] in ('PASS','WARN') and audit['blocking_issue_count'] == 0
for name,digest in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest.removeprefix('sha256:')
attempt = prepared/'cpu_execution'
assert not attempt.exists()
attempt.mkdir()
port = json.loads((root/'NATIVE_SOURCE_PORT.json').read_bytes())
files = {path.relative_to(prepared).as_posix():path.read_bytes()
    for leaf in (prepared/'source',prepared/'nr3d',prepared/'sr3d') for path in leaf.rglob('*') if path.is_file()}
files['check_author_core_cpu.py'] = (root/'referit_author_core_20261010_check_cpu.py').read_bytes()
files['NORMAL_NATIVE_RUN_PROTOCOL.json'] = (root/'NORMAL_NATIVE_RUN_PROTOCOL.json').read_bytes()
files['NATIVE_SOURCE_PORT.json'] = (root/'NATIVE_SOURCE_PORT.json').read_bytes()
payload = dict(root='/root/autodl-tmp/pvground_referit_author_core_cpu_20261010',
    env_spec_sha256=port['env_spec_sha256'],
    files={name:base64.b64encode(raw).decode() for name,raw in files.items()})
code = r'''import base64,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/pvground_referit_author_core_cpu_20261010') and not root.exists()
runtime=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1')
spec=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()==b['env_spec_sha256']
root.mkdir()
for name,encoded in b['files'].items():
 path=root/name;assert root in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(encoded))
port=json.loads((root/'NATIVE_SOURCE_PORT.json').read_bytes());source=root/'PV-Ground'
for name,metadata in port['files'].items():
 raw=Path(metadata['path']).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==metadata['sha256']
 path=source/name;assert source in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for path in (root/'source').rglob('*.py'):
 target=source/path.relative_to(root/'source');target.write_bytes(path.read_bytes())
environment=dict(os.environ,**spec['env'])
environment.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
environment['PYTHONPATH']=str(source)+':'+environment['PYTHONPATH']
argv=[str(runtime/'venv/bin/python'),'-B','-u',str(root/'check_author_core_cpu.py')]
started=datetime.datetime.now().astimezone().isoformat()
response=subprocess.run(argv,cwd=str(source),env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(root/'CPU_STDOUT.txt').write_bytes(response.stdout);(root/'CPU_STDERR.txt').write_bytes(response.stderr)
out=dict(started_cst=started,finished_cst=datetime.datetime.now().astimezone().isoformat(),exit_code=response.returncode,
 source=str(source),env_spec_sha256=b['env_spec_sha256'],GPU_calls=0,normal_training_queries=0,optimizer_steps=0,
 stdout_base64=base64.b64encode(response.stdout).decode(),stderr_base64=base64.b64encode(response.stderr).decode(),
 result_base64=base64.b64encode((root/'AUTHOR_CORE_CPU_RESULT.json').read_bytes()).decode() if (root/'AUTHOR_CORE_CPU_RESULT.json').is_file() else None)
(root/'CPU_EXECUTION.json').write_text(json.dumps({k:v for k,v in out.items() if not k.endswith('_base64')},indent=2)+'\n')
print(json.dumps(out));sys.exit(response.returncode)
'''
witness = json.loads((root.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
response = subprocess.run(argv,env=environment,input=json.dumps(payload).encode(),
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(attempt/'RAW_STDOUT.json').write_bytes(response.stdout)
(attempt/'RAW_STDERR.txt').write_bytes(response.stderr)
(attempt/'TRANSPORT_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
assert response.stdout, 'No CPU receipt; preserve transport failure; do not restart blindly.'
receipt = json.loads(response.stdout)
(attempt/'CPU_STDOUT.txt').write_bytes(base64.b64decode(receipt.pop('stdout_base64')))
(attempt/'CPU_STDERR.txt').write_bytes(base64.b64decode(receipt.pop('stderr_base64')))
result = receipt.pop('result_base64')
receipt.update(source_review_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
(attempt/'CPU_EXECUTION.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert response.returncode == receipt['exit_code'] == 0
assert result is not None
raw = base64.b64decode(result)
(attempt/'AUTHOR_CORE_CPU_RESULT.json').write_bytes(raw)
print(json.dumps(dict(status=json.loads(raw)['status'],exit_code=receipt['exit_code'],
    GPU_calls=0,normal_training_queries=0,real_loader_rows=0)),flush=True)
