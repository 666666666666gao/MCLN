"""One reviewed native evaluator CPU check; no training or dataset computation."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
audit_path = root / 'source_review/EXPERIMENT_CODE_REVIEW.json'
audit = json.loads(audit_path.read_bytes())
assert audit['execution_scope'] == 'SOURCE_ONLY' and audit['verdict'] in ('PASS', 'WARN')
assert not audit['blocking_findings']
for name, digest in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest.removeprefix('sha256:')
attempt = root / 'cpu_execution'
assert not attempt.exists()
attempt.mkdir()
files = {name: base64.b64encode((root / name).read_bytes()).decode() for name in (
    'check_native_same_query_cpu.py', 'CHECK_SPEC.json',
    'BASE_WARM_SOURCE_HASHES.json', 'WARM_SOURCE_HASHES.json')}
files.update({'source/' + path.relative_to(root / 'source').as_posix():
    base64.b64encode(path.read_bytes()).decode() for path in (root / 'source').rglob('*.py')})
remote_code = r'''import base64,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path('/root/autodl-tmp/pvground_referit_same_query_cpu_20261010')
assert not root.exists()
spec=json.loads(base64.b64decode(b['files']['CHECK_SPEC.json']))
base=json.loads(base64.b64decode(b['files']['BASE_WARM_SOURCE_HASHES.json']))
expected=json.loads(base64.b64decode(b['files']['WARM_SOURCE_HASHES.json']))
warm=Path(spec['warm_source'])
for name,digest in base.items():
 assert hashlib.sha256((warm/name).read_bytes()).hexdigest()==digest
runtime=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1')
env=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
root.mkdir();source=root/'PV-Ground';source.mkdir()
for name in base:
 path=source/name;assert source in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((warm/name).read_bytes())
for name,encoded in b['files'].items():
 if name.startswith('source/'):
  path=source/name[7:];assert source in path.resolve().parents
 else:
  path=root/name;assert root in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(encoded))
for name,digest in expected.items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
environment=dict(os.environ,**env['env'])
environment.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
environment['PYTHONPATH']=str(source)+':'+environment['PYTHONPATH']
started=datetime.datetime.now().astimezone().isoformat()
response=subprocess.run([str(runtime/'venv/bin/python'),'-B','-u',str(root/'check_native_same_query_cpu.py')],
 cwd=str(source),env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(root/'CPU_STDOUT.txt').write_bytes(response.stdout);(root/'CPU_STDERR_PRIVATE.txt').write_bytes(response.stderr)
out=dict(started_cst=started,finished_cst=datetime.datetime.now().astimezone().isoformat(),exit_code=response.returncode,
 warm_source=str(warm),isolated_source=str(source),env_spec_sha256=spec['env_spec_sha256'],current_training_queries=0,
 stdout_base64=base64.b64encode(response.stdout).decode(),stderr_base64=base64.b64encode(response.stderr).decode(),
 result_base64=base64.b64encode((root/'SAME_QUERY_CPU_RESULT.json').read_bytes()).decode() if (root/'SAME_QUERY_CPU_RESULT.json').is_file() else None)
print(json.dumps(out));sys.exit(response.returncode)
'''
witness = json.loads((root.parent.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,
    SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o',
    'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', remote_code])]
response = subprocess.run(argv, env=environment, input=json.dumps(dict(files=files)).encode(),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(attempt / 'RAW_STDOUT.json').write_bytes(response.stdout)
(attempt / 'RAW_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(attempt / 'TRANSPORT_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
receipt = json.loads(response.stdout)
(attempt / 'CPU_STDOUT.txt').write_bytes(base64.b64decode(receipt.pop('stdout_base64')))
(attempt / 'CPU_STDERR_PRIVATE.txt').write_bytes(base64.b64decode(receipt.pop('stderr_base64')))
result = receipt.pop('result_base64')
receipt['source_review_sha256'] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
(attempt / 'CPU_EXECUTION.json').write_text(json.dumps(receipt, indent=2)+'\n')
assert response.returncode == receipt['exit_code'] == 0 and result is not None
raw = base64.b64decode(result)
(attempt / 'SAME_QUERY_CPU_RESULT.json').write_bytes(raw)
print(json.dumps(dict(status=json.loads(raw)['status'], exit_code=0,
    formal_accuracy=None, GPU_training_admission=False)), flush=True)
