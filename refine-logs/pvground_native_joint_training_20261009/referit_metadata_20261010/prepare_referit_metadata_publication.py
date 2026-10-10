"""Reuse the proven static copy route for the completed metadata preparation."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
old = (root / 'publish_normal_epoch1_authorized.py').read_text(encoding='utf-8')
tail = old[old.index("code = r'''import base64,hashlib,json,sys") :]
for before, after in (
    ('normal_epoch1_observation_20261010', 'referit_metadata_20261010'),
    ('tmp_normal_epoch1_observation', 'tmp_referit_metadata'),
    ('NORMAL_EPOCH1_PUBLICATION_REMOTE', 'REFERIT_METADATA_PUBLICATION_REMOTE'),
    ('normal_epoch1_local_commit', 'referit_metadata_local_commit'),
    ('normal_epoch1_publication', 'referit_metadata_publication'),
    ('NORMAL_EPOCH1_ALL_', 'REFERIT_METADATA_ALL_'),
    ('20.376.135', '20.376.136'),
    ('Record first normal joint epoch and measured next observation time',
     'Verify ReferIt dataset metadata and prediction input availability'),
):
    assert before in tail
    tail = tail.replace(before, after)
begin = tail.index('    interim_results_unaudited=True,')
end = tail.index("(root / 'referit_metadata_local_commit.json')", begin)
tail = tail[:begin] + """    metadata_check_audited=True, metadata_counts=closure['metadata_counts'],
    bounded_check_verdict=closure['bounded_check_verdict'], audit_verdict=closure['audit_verdict'],
    actual_loader_verified=False, REC_accuracy=None,
    Nr3D_or_Sr3D_training_launched=False, active_training_source_changed=False,
    normal_training_status_queries=0, next_observation_cst=plan['due_cst'], full_goal_complete=False)
""" + tail[end:]
begin = tail.index('print(json.dumps({k:local[k]')
tail = tail[:begin] + """print(json.dumps({k:local[k] for k in ('status','section','heads','doc_sha256','new_files',
    'metadata_counts','bounded_check_verdict','next_observation_cst','full_goal_complete')}),flush=True)
"""
head = '''"""Publish bounded data readiness without another current training query."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
prior=json.loads((root/'normal_epoch1_publication.json').read_bytes())
assert prior['section']=='20.376.135' and prior['remote_handoff_sync_complete'] and prior['github_main_verified']
prior_remote=json.loads(Path(prior['remote_sync_receipt']).read_bytes())
assert prior_remote['remote_sync_complete'] and prior_remote['doc_sha256']==prior['doc_sha256']
assert not (root/'referit_metadata_publication.json').exists()
closure=json.loads((root/'ACTUAL_REFERIT_METADATA_CLOSURE.json').read_bytes())
assert closure['status']=='BOUNDED_REAL_REFERIT_METADATA_CHECK_CLOSED'
assert closure['audit_verdict']=='WARN' and closure['bounded_check_verdict']=='PASS' and closure['blocking_issue_count']==0
assert closure['GPU_or_training_admission'] is False
data=root/'referit_metadata_20261010'
seal=json.loads((data/'actual_review/ACTUAL_REVIEW_SEAL.json').read_bytes())
for name,digest in seal['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest.removeprefix('sha256:')
for row in seal['output_artifacts']:
    raw=Path(row['path']).read_bytes()
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
plan=json.loads((root/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())
repos=[Path('C:/Users/gb')/name for name in (
    '.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.136' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for index,repo in enumerate(repos):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][index]
    assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes=(root/'HANDOFF_REFERIT_METADATA_20261010.md').read_text(encoding='utf-8')
document=old+('\\n\\n'+notes.replace('\\r\\n','\\n')).replace('\\n','\\r\\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/referit_metadata_20261010/'
names=['HANDOFF_REFERIT_METADATA_20261010.md','ACTUAL_REFERIT_METADATA_CLOSURE.json',
    'record_referit_metadata_closure.py','prepare_referit_metadata_publication.py',
    'publish_referit_metadata_authorized.py','REFERIT_OBJECT_PROTOCOL_SOURCE_NOTE_20261010.md']
names.extend(path.relative_to(root).as_posix() for path in data.rglob('*')
    if path.is_file() and path.name!='RAW_STDERR_PRIVATE.txt')
files={prefix+name:(root/name).read_bytes() for name in sorted(set(names))}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\\n'
assert not any('STDERR' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert all(not (repo/prefix).exists() for repo in repos[:2])
'''
source = head + tail
ast.parse(source)
target = root / 'publish_referit_metadata_authorized.py'
assert not target.exists()
target.write_text(source, encoding='utf-8')
print('Prepared static metadata publication; no current training query or publication executed.')
