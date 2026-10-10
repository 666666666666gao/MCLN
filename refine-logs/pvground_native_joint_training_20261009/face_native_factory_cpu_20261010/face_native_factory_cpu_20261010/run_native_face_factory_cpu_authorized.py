"""One reviewed CPU construction check; no mutation/query of active training."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
previous=root.parent
review_path=root/'source_review/EXPERIMENT_CODE_REVIEW.json'
review=json.loads(review_path.read_bytes())
assert review['execution_scope']=='SOURCE_ONLY'
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for name,digest in review['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
attempt=root/'cpu_execution'
assert not attempt.exists()
attempt.mkdir()
files={str(path.relative_to(root)).replace('\\','/'):path.read_bytes()
       for path in (root/'source').glob('*.py')}
for name in ('check_native_face_factory_cpu.py','source_conditioned.json','without_additional_source.json',
             'NATIVE_SOURCE_PORT.json','NORMAL_NATIVE_RUN_PROTOCOL.json','NORMAL_E0_IDENTITY.json'):
    files[name]=(root/name).read_bytes()
port=json.loads((root/'NATIVE_SOURCE_PORT.json').read_bytes())
data=dict(root='/root/autodl-tmp/pvground_face_native_factory_cpu_20261010',
    env_spec_sha256=port['env_spec_sha256'],
    files={name:base64.b64encode(raw).decode() for name,raw in files.items()})
code=r'''import base64,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/pvground_face_native_factory_cpu_20261010') and not root.exists()
runtime=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1')
environment=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==b['env_spec_sha256']
port=json.loads(base64.b64decode(b['files']['NATIVE_SOURCE_PORT.json']))
for name,row in port['files'].items():assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
identity=json.loads(base64.b64decode(b['files']['NORMAL_E0_IDENTITY.json']))
checkpoint=Path(identity['path']);assert checkpoint.stat().st_size==identity['bytes']
h=hashlib.sha256()
with checkpoint.open('rb') as stream:
 for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==identity['sha256']
root.mkdir()
for name,encoded in b['files'].items():
 path=root/name;assert root in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(encoded))
variables=dict(os.environ,**environment['env'])
variables.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
variables['PYTHONPATH']=str(root/'source')+':'+port['model_source']+':'+variables['PYTHONPATH']
started=datetime.datetime.now().astimezone().isoformat()
response=subprocess.run([str(runtime/'venv/bin/python'),'-B','-u',str(root/'check_native_face_factory_cpu.py')],
 cwd=port['model_source'],env=variables,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(root/'CPU_STDOUT.txt').write_bytes(response.stdout);(root/'CPU_STDERR.txt').write_bytes(response.stderr)
record=dict(started_cst=started,finished_cst=datetime.datetime.now().astimezone().isoformat(),exit_code=response.returncode,
 root=str(root),warm_source=port['model_source'],env_spec_sha256=b['env_spec_sha256'],
 GPU_calls=0,current_training_queries=0,optimizer_steps=0,saved_weight_files=0)
(root/'CPU_EXECUTION.json').write_text(json.dumps(record,indent=2)+'\n')
result=root/'NATIVE_FACE_FACTORY_CPU_RESULT.json'
record.update(stdout_base64=base64.b64encode(response.stdout).decode(),stderr_base64=base64.b64encode(response.stderr).decode(),
 result_base64=base64.b64encode(result.read_bytes()).decode() if result.is_file() else None)
print(json.dumps(record));sys.exit(response.returncode)
'''
witness=json.loads((previous.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
variables=dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
command=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
response=subprocess.run(command,env=variables,input=json.dumps(data).encode(),stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(attempt/'RAW_STDOUT.json').write_bytes(response.stdout)
(attempt/'RAW_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(attempt/'TRANSPORT_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
assert response.stdout,'No CPU receipt; preserve transport failure and inspect original state, do not restart blindly'
record=json.loads(response.stdout)
for key,name in [('stdout_base64','CPU_STDOUT.txt'),('stderr_base64','CPU_STDERR_PRIVATE.txt')]:
    (attempt/name).write_bytes(base64.b64decode(record.pop(key)))
result=record.pop('result_base64')
record['source_review_sha256']=hashlib.sha256(review_path.read_bytes()).hexdigest()
(attempt/'CPU_EXECUTION.json').write_text(json.dumps(record,indent=2)+'\n')
assert response.returncode==record['exit_code']==0,'Inspect private CPU error and original evidence before any repair'
assert result is not None
raw=base64.b64decode(result)
(attempt/'NATIVE_FACE_FACTORY_CPU_RESULT.json').write_bytes(raw)
print(json.dumps(dict(status=json.loads(raw)['status'],exit_code=record['exit_code'],GPU_calls=0,
    current_training_queries=0,real_loader_rows=0)),flush=True)
