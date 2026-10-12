"""Stage reviewed read-only diagnostics before this study has any GPU job."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
assert not (root / 'SOURCE_STAGE_R4_RECEIPT.json').exists()
review_path = root / 'source_review/SOURCE_REVIEW_DIAGNOSTICS.json'
review = json.loads(review_path.read_bytes())
assert review['scope'] == 'SOURCE_ONLY' and review['source_preparation_pass']
assert not review['scoped_blockers']
prior = json.loads((root / 'SOURCE_STAGE_R3_RECEIPT.json').read_bytes())
files = {}
for name in ('train_dist_mod.py', 'selected_geometry_eval_diagnostics.py'):
    raw = (root / 'source_overlay_v4' / name).read_bytes()
    files[name] = dict(base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
code = r'''import ast,base64,datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);base=Path('/root/autodl-tmp/pvground_deployment_geometry_training_20261012')
prior=json.loads((base/'source_stage_manifest.json').read_bytes());assert prior==b['prior']
assert not (base/'preflight').exists() and not (base/'normal').exists()
history=base/'source_history'
assert history.is_dir() and not (history/'source_stage_manifest_r3.json').exists()
(history/'source_stage_manifest_r3.json').write_bytes((base/'source_stage_manifest.json').read_bytes())
(history/'train_dist_mod_r3.py').write_bytes((base/'PV-Ground/train_dist_mod.py').read_bytes())
assert not (base/'PV-Ground/selected_geometry_eval_diagnostics.py').exists()
record=dict(prior);record['source_files']={name:dict(row) for name,row in prior['source_files'].items()}
assert len(record['source_files'])==117
for name,row in b['files'].items():
 raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];ast.parse(raw.decode('utf-8'))
 path=base/'PV-Ground'/name;path.write_bytes(raw)
 record['source_files'][name]=dict(path=str(path),sha256=row['sha256'],bytes=len(raw),
  role='read_only_compact_native_evaluation_diagnostics')
assert len(record['source_files'])==118
for row in record['source_files'].values():assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
record.update(status='SOURCE_AND_LAUNCHERS_STAGED_GPU_PENDING',staged_cst=datetime.datetime.now().astimezone().isoformat(),
 source_file_count=118,source_review_sha256=b['review_sha256'],diagnostics_source_review_sha256=b['review_sha256'],
 R3_manifest_and_native_entry_preserved=True,neural_calls=0,GPU_calls=0,training_progress_queries=0,
 normal_training_started=False,actual_GPU_preflight_completed=False)
(base/'source_stage_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
project = root.parent.parent
witness = json.loads((project / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ,
    SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o',
    'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', ' '.join(shlex.quote(part) for part in [runtime, '-B', '-u', '-c', code])]
body = dict(prior=prior, files=files, review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest())
response = subprocess.run(argv, env=environment, input=json.dumps(body).encode(), stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'SOURCE_STAGE_R4_STDOUT.json').write_bytes(response.stdout)
(root / 'SOURCE_STAGE_R4_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(root / 'SOURCE_STAGE_R4_TRANSPORT_EXIT.json').write_bytes((json.dumps(dict(exit_code=response.returncode)) + '\n').encode())
assert response.returncode == 0
record = json.loads(response.stdout)
(root / 'SOURCE_STAGE_R4_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps({key: record[key] for key in ('status', 'source_file_count', 'GPU_calls', 'training_progress_queries')}))
