"""Stage two source-only copies; never read training progress or launch a job."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


root = Path(__file__).resolve().parent
base = root.parent
output = root / 'isolated_source_stage_20261014'
assert not output.exists()
output.mkdir()
prior_path = base / 'span_lr_runtime_staging_20261013_v2/SOURCE_STAGE_RECEIPT.json'
prior = json.loads(prior_path.read_bytes())
assert prior['source_file_count'] == len(prior['source_files']) == 118
closure = json.loads((root / 'LOCAL_PREPARATION_CLOSURE_20261014.json').read_bytes())
assert closure['source_review_status'] == closure['runtime_review_status'] == 'PASS_SOURCE_ONLY'
review_path = root / 'review_launch/LAUNCHERS_REVIEW_COMPLETE.json'
review = json.loads(review_path.read_bytes())
assert review['status'] == 'PASS_SOURCE_ONLY' and review['unresolved_blocker_count'] == 0
assert review['source_launchers_pass'] and review['source_stager_pass']
replacements = {'models/losses.py': (root / 'source/models/losses.py').read_bytes()}
additions = {'matched_endpoint_geometry_objective.py': (root / 'source/matched_endpoint_geometry_objective.py').read_bytes()}
assets = {
    'initialization_reference.json': (root / 'runtime_preparation/initialization_reference.json').read_bytes(),
    'native_matched_endpoint_preflight.py': (root / 'runtime_preparation/native_matched_endpoint_preflight.py').read_bytes(),
    'matched_endpoint_preflight_pair_controller.py': (root / 'launch_preparation/matched_endpoint_preflight_pair_controller.py').read_bytes(),
    'normal_matched_endpoint_controller.py': (root / 'launch_preparation/normal_matched_endpoint_controller.py').read_bytes(),
}
configurations = {}
for arm in ('final_duplicate', 'coarse_endpoint'):
    protocol_path = root / 'runtime_preparation' / (arm + '_protocol.json')
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol['arm'] == arm and protocol['source_local_preparation_only']
    protocol.update(status='SOURCE_STAGED_NOT_GPU_ADMITTED_OR_QUEUED',
        source_local_preparation_only=False, source_stage_only=True)
    assets[arm + '_protocol.json'] = (json.dumps(protocol, indent=2) + '\n').encode('utf-8')
    configurations[arm] = (root / 'runtime_preparation' / (arm + '_matched_endpoint_config.json')).read_bytes()
reviewed_paths = ['stage_isolated_sources.py', 'source/models/losses.py',
    'source/matched_endpoint_geometry_objective.py',
    'runtime_preparation/native_matched_endpoint_preflight.py',
    'launch_preparation/matched_endpoint_preflight_pair_controller.py',
    'launch_preparation/normal_matched_endpoint_controller.py',
    'runtime_preparation/initialization_reference.json']
for arm in configurations:
    reviewed_paths += ['runtime_preparation/' + arm + '_protocol.json',
        'runtime_preparation/' + arm + '_matched_endpoint_config.json']
for name in reviewed_paths:
    raw = (root / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == review['verified_source_sha256'][name]


def packed(files):
    return {name: dict(base64=base64.b64encode(raw).decode('ascii'), bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest()) for name, raw in files.items()}


code = r'''import ast,base64,datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);prior=b['prior'];src=Path(prior['model_source']).resolve()
assert str(src)=='/root/autodl-tmp/pvground_span_lr_20261013/PV-Ground'
root=Path('/root/autodl-tmp/pvground_matched_endpoint_20261014');assert not root.exists()
assert root.parent.resolve()==Path('/root/autodl-tmp').resolve()
env=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==prior['env_spec_sha256']
source_bytes={}
for name,row in prior['source_files'].items():
 p=src/name;assert src in p.resolve().parents
 raw=p.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],name
 source_bytes[name]=raw
assert len(source_bytes)==118
for name,row in b['replacements'].items():
 assert name in source_bytes
 raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 source_bytes[name]=raw
for name,row in b['additions'].items():
 assert name not in source_bytes
 raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 source_bytes[name]=raw
records={};model_sources={}
for arm,row in b['configurations'].items():
 sources=dict(source_bytes)
 raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 sources['matched_endpoint_config.json']=raw
 destination=root/arm/'PV-Ground';destination.mkdir(parents=True)
 assert root.resolve() in destination.resolve().parents
 model_sources[arm]=str(destination)
 for name,data in sources.items():
  if name.endswith('.py'):ast.parse(data.decode('utf-8'),filename=name)
  p=destination/name;assert destination.resolve() in p.resolve().parents
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert p.read_bytes()==data
  records[arm+'/'+name]=dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
assert len(records)==240
for name,row in b['assets'].items():
 raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 (root/name).write_bytes(raw)
for arm,source in model_sources.items():
 protocol=json.loads((root/(arm+'_protocol.json')).read_bytes())
 assert protocol['model_source']==source and protocol['native_init_spec']==str(root/'initialization_reference.json')
 config=json.loads((Path(source)/'matched_endpoint_config.json').read_bytes())
 assert config['endpoint']==protocol['matched_endpoint']
record=dict(status='TWO_ISOLATED_SOURCE_COPIES_STAGED_NOT_GPU_ADMITTED_OR_QUEUED',
 staged_cst=datetime.datetime.now().astimezone().isoformat(),remote_root=str(root),
 model_sources=model_sources,source_file_count=len(records),source_files=records,
 parent_source_receipt_sha256=b['parent_receipt_sha256'],env_spec_sha256=prior['env_spec_sha256'],
 reviewed_local_source_marker_sha256=b['review_marker_sha256'],
 modified_existing_source_files=['models/losses.py'],
 new_source_files=['matched_endpoint_geometry_objective.py','matched_endpoint_config.json'],
 assets={n:dict(bytes=r['bytes'],sha256=r['sha256']) for n,r in b['assets'].items()},
 active_source_changed=False,training_progress_queries=0,GPU_calls=0,model_forwards=0,
 optimizer_updates=0,queued=False,normal_training_started=False,actual_GPU_preflight_completed=False,
 new_formal_accuracy=None)
(root/'source_stage_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
witness = json.loads((base.parent.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,
    SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o',
    'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', ' '.join(shlex.quote(part) for part in [runtime, '-B', '-u', '-c', code])]
body = dict(prior=prior, replacements=packed(replacements), additions=packed(additions),
    configurations=packed(configurations), assets=packed(assets),
    parent_receipt_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
    review_marker_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest())
response = subprocess.run(argv, env=environment, input=json.dumps(body).encode('utf-8'),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(output / 'SOURCE_STAGE_STDOUT.json').write_bytes(response.stdout)
(output / 'SOURCE_STAGE_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(output / 'SOURCE_STAGE_EXIT.json').write_bytes((json.dumps(dict(exit_code=response.returncode)) + '\n').encode('utf-8'))
assert response.returncode == 0
record = json.loads(response.stdout)
assert record['source_file_count'] == 240 and not record['active_source_changed']
(output / 'SOURCE_STAGE_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))
print(json.dumps({key: record[key] for key in ('status', 'source_file_count',
    'modified_existing_source_files', 'new_source_files', 'GPU_calls', 'training_progress_queries',
    'normal_training_started', 'active_source_changed')}), flush=True)
