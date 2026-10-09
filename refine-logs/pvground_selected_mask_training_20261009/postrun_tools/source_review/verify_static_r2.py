"""Retention R2 source-only metadata diff and preservation verification."""
import ast
import copy
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import sys

out=Path(__file__).resolve().parent
toolroot=out.parent
local=toolroot.parent
assert sys.version_info[:2]==(3,7),sys.version
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def assign(module,name):
    return next(node for node in module.body if isinstance(node,ast.Assign) and
        any(isinstance(target,ast.Name) and target.id==name for target in node.targets))
manifest=read_json(out/'INPUT_MANIFEST_R2.json')
preserved=read_json(out/'R2_PRIOR_PRESERVATION_BASELINE.json')
for item in preserved:
    assert sha(Path(item['preserved_path']))==item['sha256'],item['preserved_path']
for item in manifest:
    assert sha(Path(item['path']))==item['sha256']==sha(out/item['snapshot']),item['path']
for original,saved in (
    ('RAW_RESPONSE.md','RAW_RESPONSE_R1.md'),('INPUT_MANIFEST.json','INPUT_MANIFEST_R1.json')):
    assert (out/original).read_bytes()==(out/saved).read_bytes()
assert sha(out/'RETENTION_SOURCE_REVIEW_R1.json')=='4b86135d04d07012aa7e6ad65d2bf85506e9558c2d42e42e71be1a32ddfa68b3'
assert sha(out/'RETENTION_SOURCE_REVIEW_R1.md')=='09cf1d4243b314e1a30e1ce31f3f0365bdb0fdf3e108998e21d8072b8e305206'
assert (out/'R2_EXACT_REQUEST.txt').read_bytes()==(toolroot/'SOURCE_R2_REQUEST.txt').read_bytes()
old_manifest=read_json(out/'INPUT_MANIFEST_R1.json')
old=(out/'snapshots/001_prepare_retention_plan.py').read_text(encoding='utf-8')
new=(toolroot/'prepare_retention_plan.py').read_text(encoding='utf-8')
before='new_weights_created=False, ssh_queries=0, deletions=0)'
after='new_trained_weights_created=False, archived_terminal_copies=2, ssh_queries=0, deletions=0)'
assert old.count(before)==1 and new.count(after)==1
assert old.replace(before,after)==new
assert sha(toolroot/'prepare_retention_plan.py')=='a27cca6332abfd0dd438eb2d369f9f993b245b51334fb354316b1d9672ae6bfc'
assert sha(toolroot/'retire_closed_pair_authorized.py')=='9b4a6e40730fc66ebf92e7f40a140d25e88a95987ca701312f0aefc3cfc3c206'
unchanged=0
for item in old_manifest:
    path=Path(item['path'])
    if path.parent.resolve()==toolroot and path.name in ('prepare_retention_plan.py','TOOL_PREPARATION.json'):
        continue
    assert sha(path)==item['sha256'],item['path']
    unchanged+=1
oldprep=read_json(out/'snapshots/003_TOOL_PREPARATION.json')
prep=read_json(toolroot/'TOOL_PREPARATION.json')
assert prep['source_sha256']=={
    'prepare_retention_plan.py':sha(toolroot/'prepare_retention_plan.py'),
    'retire_closed_pair_authorized.py':sha(toolroot/'retire_closed_pair_authorized.py')}
normalized=copy.deepcopy(prep)
normalized['source_sha256']['prepare_retention_plan.py']=oldprep['source_sha256']['prepare_retention_plan.py']
normalized['status']=oldprep['status']
assert normalized==oldprep and prep['archived_terminal_copies']==0
spec=read_json(local/'pair_spec.json')
assert len(spec['new_runner_files'])==9
active=[]
for name,digest in spec['new_runner_files'].items():
    assert sha(local/name)==digest,name
    active.append(dict(path=str(local/name),sha256=digest))
python_inputs=[item for item in manifest if Path(item['path']).suffix=='.py']
for item in python_inputs:
    ast.parse((out/item['snapshot']).read_bytes(),filename=item['path'])
module=ast.parse(new)
plan=assign(module,'plan').value
fields={kw.arg:kw.value for kw in plan.keywords}
assert 'new_weights_created' not in fields
assert ast.literal_eval(fields['new_trained_weights_created']) is False
assert ast.literal_eval(fields['archived_terminal_copies'])==2
assert ast.literal_eval(fields['archived_terminals'])==2
assert ast.literal_eval(fields['deletions'])==0
# The retirement consumer compares the full emitted archive manifest and proposal.
# It does not read the replaced metadata key or the two new keys.
retire=ast.parse((toolroot/'retire_closed_pair_authorized.py').read_bytes())
consumer_strings={node.s for node in ast.walk(retire) if isinstance(node,ast.Str)}
assert not {'new_weights_created','new_trained_weights_created','archived_terminal_copies'}.intersection(consumer_strings)
range_node=next(node for node in ast.walk(retire) if isinstance(node,ast.Call) and
    isinstance(node.func,ast.Name) and node.func.id=='range')
args=[ast.literal_eval(arg) for arg in range_node.args]
assert args==[0,9508,8]
offsets=list(range(*args))
assert len(offsets)==1189 and offsets[-1]==9504
diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
    fromfile='R1/prepare_retention_plan.py',tofile='R2/prepare_retention_plan.py'))
(out/'R2_SOURCE.diff').write_text(diff,encoding='utf-8')
result=dict(scope='STATIC_ONLY',execution_scope='SOURCE_ONLY',
    generated_at=datetime.datetime.utcnow().isoformat()+'Z',checks_passed=True,
    python_executable=sys.executable,python_version=sys.version,
    prior_artifact_contents_preserved=len(preserved),prior_original_paths_unchanged=len(preserved)-2,
    prior_review_aliases_versioned_as_R1=2,R1_raw_and_manifest_exact_copies=True,
    inputs_snapshotted=len(manifest),python_inputs_parsed=len(python_inputs),
    exact_one_line_source_change=True,other_R1_inputs_unchanged=unchanged,
    canonical_active_files_unchanged=active,current_tool_hashes=prep['source_sha256'],
    future_successful_plan_metadata=dict(new_trained_weights_created=False,archived_terminal_copies=2),
    current_preparation_actual_archived_terminal_copies=0,
    consumer_schema_and_full_archive_manifest_equality_unchanged=True,
    actual_plan_or_archive_created=False,actual_decision_created=False,
    actual_deletions=0,actual_model_runtime_witness=False,actual_results_read=False,
    filename_count=len(offsets),last_offset=offsets[-1],
    source_gates_ranking_allowlist_protection_and_byte_checks_unchanged=True,
    torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
    paramiko_imported='paramiko' in sys.modules,project_imports=0,actual_SSH=False,actual_GPU=False)
assert not result['torch_imported'] and not result['numpy_imported'] and not result['paramiko_imported']
(out/'R2_STATIC_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,sort_keys=True))
print(diff)
