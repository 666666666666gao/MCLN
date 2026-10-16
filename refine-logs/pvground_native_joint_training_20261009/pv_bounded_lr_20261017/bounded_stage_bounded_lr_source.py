"""Clone reviewed coarse source; update only the new target selection threshold."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
prior = root.parent / 'coarse_endpoint_training_20261014'
assert not (root / 'SOURCE_STAGE_RECEIPT.json').exists()
review = json.loads((root / 'source_review/SOURCE_REVIEW_COMPLETE.json').read_bytes())
assert review['status'] == 'PASS_SOURCE_ONLY' and review['unresolved_blocker_count'] == 0
for name in ('stage_bounded_lr_source.py', 'source_overlay/main_utils.py'):
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == review['verified_source_sha256'][name]
parent = json.loads((prior / 'isolated_source_stage_20261014/SOURCE_STAGE_RECEIPT.json').read_bytes())
changed = (root / 'source_overlay/main_utils.py').read_bytes()
code = r'''import ast,base64,datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);parent=b['parent'];old=Path('/root/autodl-tmp/pvground_matched_endpoint_20261014')
assert json.loads((old/'source_stage_manifest.json').read_bytes())==parent
root=Path('/root/autodl-tmp/pvground_bounded_lr_20261017');assert not root.exists()
assert root.parent.resolve()==Path('/root/autodl-tmp').resolve()
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==parent['env_spec_sha256']
replacement=base64.b64decode(b['changed']);assert hashlib.sha256(replacement).hexdigest()==b['changed_sha256']
source=Path(parent['model_sources']['coarse_endpoint']);prepared={}
for key,row in parent['source_files'].items():
 if key.startswith('coarse_endpoint/'):
  name=key[len('coarse_endpoint/'):];path=source/name
  assert source.resolve() in path.resolve().parents
  raw=path.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  prepared[name]=raw
assert len(prepared)==120
assert prepared['main_utils.py'].count(b"metrics['hits025'] >= 5705")==1
assert replacement==prepared['main_utils.py'].replace(b"metrics['hits025'] >= 5705",b"metrics['hits025'] >= 5724")
prepared['main_utils.py']=replacement
destination=root/'PV-Ground';destination.mkdir(parents=True)
files={}
for name,raw in prepared.items():
 path=destination/name;assert destination.resolve() in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 if path.suffix=='.py':ast.parse(raw)
 files['coarse_endpoint/'+name]=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
record=dict(status='BOUNDED_LR_SOURCE_STAGED_NO_NN',staged_cst=datetime.datetime.now().astimezone().isoformat(),
 model_sources=dict(coarse_endpoint=str(destination)),source_file_count=120,source_files=files,
 env_spec_sha256=parent['env_spec_sha256'],parent_stage_sha256=b['parent_sha256'],
 modified_existing_source_files=['main_utils.py'],neural_or_loss_source_changed=False,
 only_source_change='native_metric_order target hits025 5705 to5724 for NEW runs',
 source_review_sha256=b['review_sha256'],new_GPU_calls=0,new_neural_forwards=0,new_optimizer_updates=0,queued=False)
(root/'source_stage_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''
witness = json.loads((root.parent.parent.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,
    SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
ssh = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
       '-o', 'StrictHostKeyChecking=yes', '-o',
       'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
       '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
       'root@region-9.autodl.pro', ' '.join(shlex.quote(p) for p in [runtime, '-B', '-u', '-c', code])]
parent_path = prior / 'isolated_source_stage_20261014/SOURCE_STAGE_RECEIPT.json'
body = dict(parent=parent, parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            changed=base64.b64encode(changed).decode(), changed_sha256=hashlib.sha256(changed).hexdigest(),
            review_sha256=hashlib.sha256((root / 'source_review/SOURCE_REVIEW_COMPLETE.json').read_bytes()).hexdigest())
reply = subprocess.run(ssh, env=environment, input=json.dumps(body).encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'SOURCE_STAGE_STDOUT.json').write_bytes(reply.stdout)
(root / 'SOURCE_STAGE_STDERR_PRIVATE.txt').write_bytes(reply.stderr)
(root / 'SOURCE_STAGE_EXIT.json').write_text(json.dumps(dict(exit_code=reply.returncode)) + '\n')
assert reply.returncode == 0
record = json.loads(reply.stdout)
(root / 'SOURCE_STAGE_RECEIPT.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(status=record['status'], source_file_count=record['source_file_count'],
                     new_neural_forwards=0, new_training_launches=0)))
