"""Verify the native-source publication, without touching active jobs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

root = Path(__file__).resolve().parent
record = json.loads((root / 'publication.json').read_bytes())
home = Path('C:/Users/gb')
repos = [home / '.codex_mcln_g0_20260905', home / '.codex_pvground_cs_20261002',
         home / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
raw = (repos[0] / doc).read_bytes()
assert hashlib.sha256(raw).hexdigest() == record['doc_sha256']
assert all((repo / doc).read_bytes() == raw for repo in repos)
assert (home / 'Desktop/document' / Path(doc).name).read_bytes() == raw
for index, repo in enumerate(repos):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == record['heads'][index]
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip()
names = subprocess.check_output(['git', '-C', str(repos[0]), 'ls-files', '--', record['prefix']]).decode().splitlines()
assert len(names) == record['new_files']
assert all((repos[0] / name).read_bytes() == (repos[1] / name).read_bytes() for name in names)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(project + doc, 'rb') as stream:
    assert stream.read() == raw
for name in names:
    with sftp.open(project + name, 'rb') as stream:
        assert stream.read() == (repos[0] / name).read_bytes()
sftp.close()
client.close()
proof = dict(status='NATIVE_SOURCE_PUBLICATION_VERIFIED',
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    heads=record['heads'], doc_sha256=record['doc_sha256'], files_verified=len(names),
    neural_calls=0, training_status_reads=0, native_joint_training_started=False,
    full_goal_complete=False)
(root / 'PUBLICATION_VERIFIED.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof))
