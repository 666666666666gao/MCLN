"""Prepare a static publication of the scheduled interim observation."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
old = (root / 'publish_face_residual_cpu_authorized_attempt2.py').read_text(encoding='utf-8')
tail = old[old.index("code = r'''import base64,hashlib,json,sys") :]
tail = tail.replace('face_residual_cpu_20261010', 'normal_epoch1_observation_20261010')
tail = tail.replace('tmp_face_residual_cpu', 'tmp_normal_epoch1_observation')
tail = tail.replace('FACE_RESIDUAL_CPU_REMOTE_ATTEMPT2', 'NORMAL_EPOCH1_PUBLICATION_REMOTE')
tail = tail.replace('face_residual_cpu_local_commit', 'normal_epoch1_local_commit')
tail = tail.replace('face_residual_cpu_publication', 'normal_epoch1_publication')
tail = tail.replace('FACE_RESIDUAL_CPU_ALL_', 'NORMAL_EPOCH1_ALL_')
tail = tail.replace('20.376.134', '20.376.135')
tail = tail.replace('Check prepared face residual CPU modules without modifying active training',
    'Record first normal joint epoch and measured next observation time')
tail = tail.replace("assert response.returncode == 0, response.stderr.decode()",
    "assert response.returncode == 0, 'Publication transport failed; inspect exact static state before any repair'")
begin = tail.index('    CPU_outcome_review_completed=True,')
end = tail.index("(root / 'normal_epoch1_local_commit.json')", begin)
tail = tail[:begin] + """    interim_results_unaudited=True, terminal_result=None,
    control_training_launched=False, Nr3D_or_Sr3D_training_launched=False,
    active_training_source_changed=False, normal_training_status_queries=0,
    observed_normal_metrics=summary['metrics'],
    next_observation_cst=plan['due_cst'], full_goal_complete=False)
""" + tail[end:]
begin = tail.index('print(json.dumps({k: local[k]')
tail = tail[:begin] + """print(json.dumps({k:local[k] for k in ('status','section','heads','doc_sha256','new_files',
    'observed_normal_metrics','next_observation_cst','full_goal_complete')}),flush=True)
"""
head = '''"""Publish the existing scheduled observation without reading the active job again."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
prior = json.loads((root/'face_residual_cpu_publication.json').read_bytes())
assert prior['section']=='20.376.134' and prior['remote_handoff_sync_complete']
assert prior['github_main_verified']
prior_remote = json.loads(Path(prior['remote_sync_receipt']).read_bytes())
assert prior_remote['remote_sync_complete'] and prior_remote['doc_sha256']==prior['doc_sha256']
assert not (root/'normal_epoch1_publication.json').exists()
summary = json.loads((root/'NORMAL_FIRST_OBSERVATION_SUMMARY.json').read_bytes())
plan = json.loads((root/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())
owner = json.loads((root/'NORMAL_EPOCH2_OBSERVER_OWNER.json').read_bytes())
assert owner['first_due_cst']==plan['due_cst'] and owner['local_pid']==51724
assert summary['epoch1_hits_delta']==[-25,-324] and summary['formal_results_unaudited']
assert json.loads((root/'NORMAL_FIRST_OBSERVATION_EXIT.json').read_bytes())['exit_code']==0
raw_observation = json.loads((root/'NORMAL_FIRST_OBSERVATION_STDOUT.json').read_bytes())
for row in raw_observation['files']:
    raw=base64.b64decode(row['base64'])
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    assert (root/'normal_first_observation'/row['name']).read_bytes()==raw
assert base64.b64decode(raw_observation['train_tail_base64'])==(root/'normal_first_observation/train_tail.txt').read_bytes()
repos=[Path('C:/Users/gb')/name for name in (
    '.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.135' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for index,repo in enumerate(repos):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][index]
    assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes=(root/'HANDOFF_NORMAL_EPOCH1_20261010.md').read_text(encoding='utf-8')
document=old+('\\n\\n'+notes.replace('\\r\\n','\\n')).replace('\\n','\\r\\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/normal_epoch1_observation_20261010/'
names=['HANDOFF_NORMAL_EPOCH1_20261010.md','NORMAL_FIRST_OBSERVATION.json',
    'NORMAL_FIRST_OBSERVATION_STDOUT.json','NORMAL_FIRST_OBSERVATION_EXIT.json',
    'NORMAL_FIRST_OBSERVATION_SUMMARY.json','NORMAL_NEXT_OBSERVATION_PLAN.json',
    'NORMAL_EPOCH2_OBSERVER_OWNER.json','prepare_normal_epoch2_observation.py',
    'observe_normal_epoch2_planned_authorized.py','prepare_normal_epoch1_publication.py',
    'publish_normal_epoch1_authorized.py']
names.extend(path.relative_to(root).as_posix() for path in (root/'normal_first_observation').rglob('*') if path.is_file())
files={prefix+name:(root/name).read_bytes() for name in sorted(set(names))}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\\n'
assert not any('STDERR' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert all(not (repo/prefix).exists() for repo in repos[:2])
'''
source = head + tail
ast.parse(source)
destination = root / 'publish_normal_epoch1_authorized.py'
assert not destination.exists()
destination.write_text(source, encoding='utf-8')
print(json.dumps(dict(status='STATIC_NORMAL_EPOCH1_PUBLISHER_PREPARED',
    next_observation_cst=json.loads((root/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())['due_cst'],
    current_training_queries=0, publication_executed=False)))
