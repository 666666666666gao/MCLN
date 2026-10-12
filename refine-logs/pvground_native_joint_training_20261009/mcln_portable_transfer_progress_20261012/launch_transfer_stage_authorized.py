"""Stage reviewed isolated source and admit one MCLN transfer stage; no retries."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('--stage',choices=('cpu','sanity','full'),required=True)
options = parser.parse_args()
root = Path(__file__).resolve().parent
assert not (root/(options.stage+'_LAUNCH.json')).exists()
review = json.loads((root/'source_review/SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for name,digest in review['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
manifest = json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
for name,row in manifest.items():
    path = root/'source'/name
    assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
archive = io.BytesIO()
with tarfile.open(fileobj=archive,mode='w:gz') as stream:
    for name in manifest: stream.add(str(root/'source'/name),arcname=name)
body = dict(stage=options.stage,archive=base64.b64encode(archive.getvalue()).decode(),manifest=manifest,
            remote_assets=json.loads((root/'REMOTE_ASSETS.json').read_bytes()),
            spec=base64.b64encode((root/'TRANSFER_SPEC.json').read_bytes()).decode(),
            controller=base64.b64encode((root/'transfer_controller.py').read_bytes()).decode(),
            source_review_sha256=hashlib.sha256((root/'source_review/SOURCE_REVIEW.json').read_bytes()).hexdigest())
code = r'''import base64,datetime,hashlib,io,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
body=json.load(sys.stdin)
parent=Path('/root/autodl-tmp/mcln_portable_best_transfer_20261012')
source=parent/'MCLN';stage=body['stage'];root=parent/stage
assert not root.exists()
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
env_sha=hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
if stage=='cpu':
 assert not source.exists()
 source.mkdir(parents=True)
 with tarfile.open(fileobj=io.BytesIO(base64.b64decode(body['archive'])),mode='r:gz') as stream:
  assert all(not Path(row.name).is_absolute() and '..' not in Path(row.name).parts for row in stream.getmembers())
  stream.extractall(str(source))
 for name,row in body['remote_assets'].items():
  original=Path(row['path'])
  assert original.stat().st_size==row['bytes'] and hashlib.sha256(original.read_bytes()).hexdigest()==row['sha256']
  target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(str(original),str(target))
 (parent/'TRANSFER_SPEC.json').write_bytes(base64.b64decode(body['spec']))
 (parent/'source_manifest.json').write_text(json.dumps(body['manifest'],indent=2)+'\n')
else:
 prior=json.loads((parent/('cpu' if stage=='sanity' else 'sanity')/'status.json').read_bytes())
 assert prior['status']=='complete' and prior['exit_code']==0
 if stage=='sanity': assert prior['result']['status']=='ACTUAL_MCLN_FACTORY_CORE_AND_PORTABLE_STATES_CPU_PASS'
 if stage=='full':
  assert prior['result']['scope']=='REAL8ROW_NATIVE_MCLN_TRANSFER_SANITY'
  assert prior['result']['all_1159_states_unchanged'] and prior['result']['original_semantic_scores_equal']
 bnroot=Path('/root/autodl-tmp/pvground_face_bn_diagnostic_20261011/full')
 bn=json.loads((bnroot/'diagnostic_status.json').read_bytes())
 assert bn['status']=='complete' and bn['exit_code']==0 and not Path('/proc/'+str(bn['child_pid'])).exists()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
for name,row in body['manifest'].items():
 path=source/name;assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
for name,row in body['remote_assets'].items():
 path=source/name;assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
assert (parent/'TRANSFER_SPEC.json').read_bytes()==base64.b64decode(body['spec'])
root.mkdir()
(root/'transfer_controller.py').write_bytes(base64.b64decode(body['controller']))
admission=dict(stage=stage,source=str(source),spec=str(parent/'TRANSFER_SPEC.json'),
 source_review_sha256=body['source_review_sha256'],env_spec_sha256=env_sha)
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
command=[runtime,'-B','-u',str(root/'transfer_controller.py')]
if stage!='cpu':command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock']]+command
with (root/'controller.log').open('wb') as stream:
 process=subprocess.Popen(command,cwd=str(source),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
assert process.poll() is None
print(json.dumps(dict(stage=stage,root=str(root),controller_pid=process.pid,controller_argv=command,
 launched_cst=datetime.datetime.now().astimezone().isoformat(),admission=admission,source_files=len(body['manifest']))))
'''
witness_path = Path('C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/SCP_TRANSPORT_WITNESS.json')
witness = json.loads(witness_path.read_bytes())
environment = dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
                   SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
command = ['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
           '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
           '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
           'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
process = subprocess.run(command,input=json.dumps(body).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=environment,timeout=300)
(root/(options.stage+'_LAUNCH_STDOUT.txt')).write_bytes(process.stdout)
(root/(options.stage+'_LAUNCH_STDERR.txt')).write_bytes(process.stderr)
assert process.returncode==0,process.stderr.decode(errors='replace')
result=json.loads(process.stdout)
(root/(options.stage+'_LAUNCH.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
