"""One reviewed limited CPU data-interface check; no active-training queries."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
audit_path=root/'source_review/EXPERIMENT_CODE_REVIEW.json'
audit=json.loads(audit_path.read_bytes())
assert audit['execution_scope']=='SOURCE_ONLY' and audit['verdict'] in ('PASS','WARN')
assert not audit['blocking_findings']
for name,digest in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest.removeprefix('sha256:')
intake=json.loads((root/'RESOURCE_INTAKE.json').read_bytes())
assert intake['admitted_by_executor'] is True
attempt=root/'cpu_execution'
assert not attempt.exists()
attempt.mkdir()
files={name:base64.b64encode((root/name).read_bytes()).decode() for name in (
    'check_native_referit_data_cpu.py','CHECK_SPEC.json','WARM_SOURCE_HASHES.json')}
remote_code=r'''import base64,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path('/root/autodl-tmp/pvground_referit_data_interface_cpu_20261010')
assert not root.exists()
spec=json.loads(base64.b64decode(b['files']['CHECK_SPEC.json']))
expected=json.loads(base64.b64decode(b['files']['WARM_SOURCE_HASHES.json']))
source=Path(spec['warm_source'])
for name,digest in expected.items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
runtime=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1')
env=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
data=Path(spec['data_root'])
assert all((data/(split+'_v3scans.pkl')).is_file() for split in ('train','val'))
root.mkdir()
for name,encoded in b['files'].items():
 path=root/name;assert root in path.resolve().parents
 path.write_bytes(base64.b64decode(encoded))
environment=dict(os.environ,**env['env'])
environment.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
environment['PYTHONPATH']=str(source)+':'+environment['PYTHONPATH']
started=datetime.datetime.now().astimezone().isoformat()
response=subprocess.run([str(runtime/'venv/bin/python'),'-B','-u',str(root/'check_native_referit_data_cpu.py')],
 cwd=str(source),env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(root/'CPU_STDOUT.txt').write_bytes(response.stdout);(root/'CPU_STDERR_PRIVATE.txt').write_bytes(response.stderr)
out=dict(started_cst=started,finished_cst=datetime.datetime.now().astimezone().isoformat(),exit_code=response.returncode,
 warm_source=str(source),env_spec_sha256=spec['env_spec_sha256'],current_training_queries=0,
 stdout_base64=base64.b64encode(response.stdout).decode(),stderr_base64=base64.b64encode(response.stderr).decode(),
 result_base64=base64.b64encode((root/'REFERIT_DATA_CPU_RESULT.json').read_bytes()).decode() if (root/'REFERIT_DATA_CPU_RESULT.json').is_file() else None)
print(json.dumps(out));sys.exit(response.returncode)
'''
witness=json.loads((root.parent.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment=dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',remote_code])]
response=subprocess.run(argv,env=environment,input=json.dumps(dict(files=files)).encode(),
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(attempt/'RAW_STDOUT.json').write_bytes(response.stdout)
(attempt/'RAW_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(attempt/'TRANSPORT_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
receipt=json.loads(response.stdout)
(attempt/'CPU_STDOUT.txt').write_bytes(base64.b64decode(receipt.pop('stdout_base64')))
(attempt/'CPU_STDERR_PRIVATE.txt').write_bytes(base64.b64decode(receipt.pop('stderr_base64')))
result=receipt.pop('result_base64')
receipt['source_review_sha256']=hashlib.sha256(audit_path.read_bytes()).hexdigest()
(attempt/'CPU_EXECUTION.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert response.returncode==receipt['exit_code']==0 and result is not None
raw=base64.b64decode(result)
(attempt/'REFERIT_DATA_CPU_RESULT.json').write_bytes(raw)
print(json.dumps(dict(status=json.loads(raw)['status'],exit_code=0,formal_accuracy=None,GPU_training_admission=False)),flush=True)
