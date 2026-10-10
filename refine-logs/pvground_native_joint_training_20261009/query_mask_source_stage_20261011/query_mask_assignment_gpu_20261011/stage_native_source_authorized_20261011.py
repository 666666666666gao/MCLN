"""Stage the bound native source bundle and reviewed matcher variant; never run ML."""
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
review = json.loads((root / 'source_stage_review/SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY' and review['verdict'] in ('PASS', 'WARN')
assert not review['blocking_findings']
for path, digest in review['audited_input_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
prior = json.loads((root.parent / 'NATIVE_SOURCE_PORT.json').read_bytes())
assert len(prior['files']) == 116 and sum(row['bytes'] for row in prior['files'].values()) == 24660224
hashes = json.loads((root / 'SOURCE_HASHES.json').read_bytes())
assert len(hashes) == 14 and set(hashes) <= set(prior['files'])
overlay = {}
for name, digest in hashes.items():
    raw = (root / 'source' / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    overlay[name] = dict(base64=base64.b64encode(raw).decode(), sha256=digest, bytes=len(raw))
assets = {}
for name in ('native_query_mask_matcher_preflight.py', 'init.json', 'text.json', 'query.json'):
    raw = (root / name).read_bytes()
    assets[name] = dict(base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
for mode in ('text', 'query'):
    config = json.loads((root / (mode + '.json')).read_bytes())
    assert config['matcher_mask_source'] == mode
    assert not config['serial_gpu_preflight_admitted'] and not config['full_training_admitted']
    assert config['model_source'] == '/root/autodl-tmp/pvground_query_mask_assignment_20261011/PV-Ground'
    assert config['native_init_spec'] == '/root/autodl-tmp/pvground_query_mask_assignment_20261011/init.json'
code = r'''import ast,base64,datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path('/root/autodl-tmp/pvground_query_mask_assignment_20261011')
assert not root.exists()
prior=b['prior'];assert prior['model_source']=='/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground'
spec=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json')
env=json.loads(spec.read_bytes())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==prior['env_spec_sha256']
source=root/'PV-Ground';source.mkdir(parents=True)
records={}
for name,row in prior['files'].items():
    parent=Path(row['path']);raw=parent.read_bytes()
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    target=source/name;assert source.resolve() in target.resolve().parents
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    records[name]=dict(path=str(target),source_path=str(parent),bytes=len(raw),sha256=row['sha256'],role=row['role'])
for name,row in b['overlay'].items():
    raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    target=source/name;assert source.resolve() in target.resolve().parents
    target.write_bytes(raw)
    records[name].update(bytes=len(raw),sha256=row['sha256'],role='reviewed_query_mask_matcher_variant')
for name,row in records.items():
    target=source/name;assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
    if name.endswith('.py'):ast.parse(target.read_text(encoding='utf-8'),filename=name)
asset_records={}
for name,row in b['assets'].items():
    raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    target=root/name;assert root.resolve() in target.resolve().parents
    target.write_bytes(raw);assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
    if name.endswith('.py'):ast.parse(raw.decode('utf-8'),filename=name)
    asset_records[name]=dict(path=str(target),bytes=len(raw),sha256=row['sha256'])
for mode in ('text','query'):
    config=json.loads((root/(mode+'.json')).read_bytes())
    assert not config['serial_gpu_preflight_admitted'] and not config['full_training_admitted']
    assert config['model_source']==str(source) and config['native_init_spec']==str(root/'init.json')
changes=[name for name,row in records.items() if row['sha256']!=prior['files'][name]['sha256']]
assert sorted(changes)==['main_utils.py','models/losses.py']
record=dict(status='BOUND_NATIVE_QUERY_MASK_MATCHER_SOURCE_STAGED_NOT_EXECUTED',
    staged_cst=datetime.datetime.now().astimezone().isoformat(),root=str(root),model_source=str(source),
    source_files=records,source_file_count=len(records),source_bytes=sum(row['bytes'] for row in records.values()),
    changed_parent_source_files=sorted(changes),assets=asset_records,env_spec_sha256=prior['env_spec_sha256'],
    shared_existing_runtime_required=True,actual_import_origins_verified=False,
    actual_model_or_data_constructor_calls=0,neural_calls=0,optimizer_updates=0,GPU_queries=0,
    training_status_reads=0,active_training_source_mutations=0,
    serial_gpu_preflight_admitted=False,full_training_admitted=False,
    GPU_preflight_executed=False,normal_training_started=False,launcher_or_controller_created=False,
    source_stage_review_sha256=b['source_stage_review_sha256'],full_goal_complete=False)
(root/'source_stage_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
witness = json.loads((project / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
                   SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
        '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
        'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
body = dict(prior=prior, overlay=overlay, assets=assets,
            source_stage_review_sha256=hashlib.sha256((root / 'source_stage_review/SOURCE_REVIEW.json').read_bytes()).hexdigest())
result = subprocess.run(argv, env=environment, input=json.dumps(body).encode(),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'SOURCE_STAGE_STDOUT.json').write_bytes(result.stdout)
(root / 'SOURCE_STAGE_STDERR_PRIVATE.txt').write_bytes(result.stderr)
(root / 'SOURCE_STAGE_EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode)) + '\n', encoding='utf-8')
assert result.returncode == 0, 'Inspect the preserved source-stage stderr and exact partial directory before any retry.'
receipt = json.loads(result.stdout)
assert receipt['source_file_count'] == 116 and not receipt['GPU_preflight_executed']
assert receipt['changed_parent_source_files'] == ['main_utils.py', 'models/losses.py']
(root / 'SOURCE_STAGE_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('status','staged_cst','source_file_count','source_bytes',
    'changed_parent_source_files','GPU_preflight_executed','normal_training_started','full_goal_complete')}))
