"""Stage reviewed source into a new directory; never run a neural/GPU task."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
project = root.parent.parent
assert not (root / 'SOURCE_STAGE_RECEIPT.json').exists()
review_path = root / 'source_review/SOURCE_REVIEW_R2.json'
review = json.loads(review_path.read_bytes())
assert review['scope'] == 'SOURCE_ONLY' and review['source_preparation_pass']
assert not review['scoped_blockers']
prior = json.loads((root.parent / 'NATIVE_SOURCE_PORT.json').read_bytes())
assert len(prior['files']) == 116
overlay = {}
for path in sorted((root / 'source_overlay_v2').rglob('*.py')):
    name = path.relative_to(root / 'source_overlay_v2').as_posix()
    raw = path.read_bytes()
    overlay[name] = dict(base64=base64.b64encode(raw).decode(),
                         sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
assert len(overlay) == 5
names = ('native_selected_geometry_preflight.py', 'normal_run_controller.py',
         'native_control_protocol.json', 'selected_geometry_protocol.json',
         'native_control_initialization.json', 'selected_geometry_initialization.json')
assets = {}
for name in names:
    raw = (root / name).read_bytes()
    assets[name] = dict(base64=base64.b64encode(raw).decode(),
                        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
code = r'''import ast,base64,datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin)
root=Path('/root/autodl-tmp/pvground_deployment_geometry_training_20261012')
assert not root.exists()
prior=b['prior'];assert prior['model_source']=='/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground'
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==prior['env_spec_sha256']
source=root/'PV-Ground';source.mkdir(parents=True)
records={}
for name,row in prior['files'].items():
 raw=Path(row['path']).read_bytes()
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=source/name;assert source.resolve() in target.resolve().parents
 target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 records[name]=dict(path=str(target),bytes=len(raw),sha256=row['sha256'],role=row['role'])
for name,row in b['overlay'].items():
 raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=source/name;assert source.resolve() in target.resolve().parents
 target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 records[name]=dict(path=str(target),bytes=len(raw),sha256=row['sha256'],role='selected_geometry_native_training')
assert len(records)==117
for name,row in records.items():
 if name.endswith('.py'):ast.parse(Path(row['path']).read_bytes().decode('utf-8'),filename=name)
for name,row in b['assets'].items():
 raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 (root/name).write_bytes(raw)
 if name.endswith('.py'):ast.parse(raw.decode('utf-8'),filename=name)
for arm in ('native_control','selected_geometry'):
 protocol=json.loads((root/(arm+'_protocol.json')).read_bytes())
 spec=json.loads((root/(arm+'_initialization.json')).read_bytes())
 assert protocol['model_source']==str(source) and protocol['native_init_spec']==str(root/(arm+'_initialization.json'))
 assert protocol['max_epochs']==8 and protocol['patience']==2 and not protocol['GPU_admitted']
 assert spec['use_selected_geometry_supervision']==(arm=='selected_geometry')
 assert spec['use_g_supervision'] and not spec['use_selected_mask_supervision']
record=dict(status='SELECTED_GEOMETRY_SOURCE_STAGED_NO_GPU_EXECUTION',
 staged_cst=datetime.datetime.now().astimezone().isoformat(),model_source=str(source),
 source_files=records,source_file_count=len(records),source_review_sha256=b['review_sha256'],
 env_spec_sha256=prior['env_spec_sha256'],GPU_calls=0,neural_calls=0,training_progress_queries=0,
 actual_optimizer_updates=0,normal_training_started=False)
(root/'source_stage_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
witness = json.loads((project / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,
    SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o',
    'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'],
    '-o', 'NumberOfPasswordPrompts=1', 'root@region-9.autodl.pro',
    ' '.join(shlex.quote(part) for part in [runtime, '-B', '-u', '-c', code])]
body = dict(prior=prior, overlay=overlay, assets=assets,
            review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest())
response = subprocess.run(argv, env=environment, input=json.dumps(body).encode(),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'SOURCE_STAGE_STDOUT.json').write_bytes(response.stdout)
(root / 'SOURCE_STAGE_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(root / 'SOURCE_STAGE_EXIT.json').write_bytes((json.dumps(dict(exit_code=response.returncode)) + '\n').encode())
assert response.returncode == 0
record = json.loads(response.stdout)
(root / 'SOURCE_STAGE_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps({key: record[key] for key in ('status', 'source_file_count', 'GPU_calls', 'training_progress_queries')}))
