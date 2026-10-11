"""Reuse the existing document publisher for the user's new complete objective."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'research_target_update_20261011'
targets = json.loads((data / 'CURRENT_RESEARCH_TARGETS.json').read_bytes())
assert targets['scanrefer']['minimum_hits025'] == 5705
assert targets['scanrefer']['minimum_hits050'] == 4945
assert not targets['full_goal_complete']
head = '''"""Publish the updated research objective; no training or GPU operation."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
prior=json.loads((root/'query_mask_source_stage_publication_20261011.json').read_bytes())
assert prior['section']=='20.376.152' and prior['github_main_verified'] and prior['remote_handoff_sync_complete']
assert not (root/'research_target_update_publication_20261011.json').exists()
data=root/'research_target_update_20261011'
summary=json.loads((data/'CURRENT_RESEARCH_TARGETS.json').read_bytes())
assert summary['scanrefer']['minimum_hits025']==5705 and summary['scanrefer']['minimum_hits050']==4945
assert not summary['full_goal_complete'] and not summary['active_training_source_changed']
plan=json.loads((root/'normal_controls_20261010/NORMAL_EPOCH3_OBSERVATION_PLAN.json').read_bytes())
owner=json.loads((root/'normal_controls_20261010/NORMAL_EPOCH3_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid']==28804 and owner['first_due_cst']==plan['due_cst']
repos=[Path('C:/Users/gb')/name for name in ('.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.153' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for i,repo in enumerate(repos):
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][i]
 assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes='## §20.376.153｜用户更新完整目标：ScanRefer60.0／52.0；达标后先验证Nr／Sr作者权重再训练（2026-10-11）\\n\\n'+(data/'CURRENT_RESEARCH_TARGETS.md').read_text(encoding='utf-8')
document=old+('\\n\\n'+notes.replace('\\r\\n','\\n')).replace('\\n','\\r\\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/research_target_update_20261011/'
files={prefix+name:(data/name).read_bytes() for name in ('CURRENT_RESEARCH_TARGETS.json','CURRENT_RESEARCH_TARGETS.md')}
for name in ('update_research_target_20261011.py','prepare_goal_update_publication_20261011.py','publish_goal_update_authorized_20261011.py'):
 files[prefix+name]=(root/name).read_bytes()
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\\n'
contract_name='idea-stage/docs/research_contract.md'
old_contract=(repos[0]/contract_name).read_text(encoding='utf-8-sig')
contract=(data/'CURRENT_RESEARCH_TARGETS.md').read_text(encoding='utf-8')+'\\n\\n## Historical contract before the2026-10-11 user update (superseded)\\n\\n'+old_contract
files[contract_name]=contract.replace('\\r\\n','\\n').replace('\\n','\\r\\n').encode('utf-8')
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert not any('STDERR' in name or '.aris' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
assert all(not (repo/prefix).exists() for repo in repos[:2])
'''
template = (root / 'publish_query_mask_source_stage_authorized_20261011.py').read_text(encoding='utf-8')
tail = template[template.index("remote_code = r'''import base64,hashlib,json,sys"):]
for old, new in (
    ('query_mask_source_stage_20261011','research_target_update_20261011'),
    ('tmp_query_mask_source_stage','tmp_research_target_update'),
    ('QUERY_MASK_SOURCE_STAGE_PUBLICATION','RESEARCH_TARGET_UPDATE_PUBLICATION'),
    ('query_mask_source_stage_local_commit_20261011','research_target_update_local_commit_20261011'),
    ('query_mask_source_stage_publication_20261011','research_target_update_publication_20261011'),
    ('QUERY_MASK_SOURCE_STAGE_ALL_','RESEARCH_TARGET_UPDATE_ALL_'),
    ('20.376.152','20.376.153'),
    ('Record isolated native Query Mask matcher source staging','Update PV-Ground research targets to ScanRefer60 and52')):
    assert old in tail, old
    tail=tail.replace(old,new)
old_assert='path=project/name;assert leaf in path.resolve().parents'
assert old_assert in tail
tail=tail.replace(old_assert, "path=project/name;assert (leaf in path.resolve().parents or (name=='idea-stage/docs/research_contract.md' and project.resolve() in path.resolve().parents))")
content=head+tail
ast.parse(content)
path=root/'publish_goal_update_authorized_20261011.py'
assert not path.exists()
path.write_text(content,encoding='utf-8')
print(json.dumps(dict(status='USER_GOAL_PUBLICATION_PREPARED', section='20.376.153',
    new_neural_calls=0, training_restarted=False)))
