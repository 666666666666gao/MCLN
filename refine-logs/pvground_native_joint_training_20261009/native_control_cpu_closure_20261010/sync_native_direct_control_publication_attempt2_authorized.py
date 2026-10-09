"""Complete only the pending static SSH copy; no training-state query or launch."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
publication_path = root / 'native_direct_controls_publication.json'
publication = json.loads(publication_path.read_bytes())
assert publication['section'] == '20.376.130' and publication['github_main_verified'] is True
assert publication['remote_handoff_sync_complete'] is False
proof = json.loads((root / 'read_only_static_diagnostic_20261010_attempt1/REMOTE_STATIC_WITNESS.json').read_bytes())
assert proof['document_matches_prior_confirmed'] is True and proof['copied_evidence_sha256'] == {}
assert not (root / 'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_RECEIPT.json').exists()
repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
document = (repo / doc).read_bytes()
assert hashlib.sha256(document).hexdigest() == publication['doc_sha256']
prefix = publication['prefix']
leaf = repo / prefix
files = {path.relative_to(repo).as_posix(): path.read_bytes() for path in sorted(leaf.rglob('*')) if path.is_file()}
assert len(files) == publication['new_files']
code = r'''import base64,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);project=Path('/home/gb/new butd/butd_detr-main/MCLN-main');doc=project/b['doc']
assert hashlib.sha256(doc.read_bytes()).hexdigest()==b['old_sha']
leaf=(project/b['prefix']).resolve()
assert leaf==Path('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_native_joint_training_20261009/native_direct_preparation_20261010') and not leaf.exists()
for name,data in b['files'].items():
 path=project/name;assert leaf in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(data);path.write_bytes(raw);assert path.read_bytes()==raw
raw=base64.b64decode(b['document']);temporary=doc.with_name(doc.name+'.tmp_native_direct_controls')
assert not temporary.exists();temporary.write_bytes(raw);assert hashlib.sha256(temporary.read_bytes()).hexdigest()==b['sha'];temporary.replace(doc)
print(json.dumps(dict(doc_sha256=b['sha'],files=len(b['files']),neural_calls=0,training_status_reads=0)))
'''
witness = json.loads((root.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
payload = dict(doc=doc, old_sha=publication['remote_last_confirmed_sha256'], prefix=prefix,
    sha=publication['doc_sha256'], document=base64.b64encode(document).decode(),
    files={name:base64.b64encode(raw).decode() for name,raw in files.items()})
response = subprocess.run(argv, env=environment, input=json.dumps(payload).encode(),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_STDOUT.json').write_bytes(response.stdout)
(root / 'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_STDERR.txt').write_bytes(response.stderr)
(root / 'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode)) + '\n', encoding='utf-8')
assert response.returncode == 0, response.stderr.decode()
receipt = json.loads(response.stdout)
assert receipt['doc_sha256'] == publication['doc_sha256'] and receipt['files'] == publication['new_files']
guard = root.parent.parent / 'sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert publication['remote_last_confirmed_sha256'].encode() in raw
guard.write_bytes(raw.replace(publication['remote_last_confirmed_sha256'].encode(), publication['doc_sha256'].encode()))
receipt.update(time_cst=datetime.datetime.now().astimezone().isoformat(), remote_sync_complete=True,
    active_training_source_unchanged=True, current_normal_training_queries=0)
(root / 'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
publication.update(status='NATIVE_DIRECT_CONTROLS_ALL_COPIES_SYNCHRONIZED',
    remote_handoff_sync_complete=True, remote_evidence_sync_complete=True,
    remote_current_sha256=publication['doc_sha256'], remote_sync_time_cst=receipt['time_cst'])
publication_path.write_text(json.dumps(publication, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt), flush=True)
