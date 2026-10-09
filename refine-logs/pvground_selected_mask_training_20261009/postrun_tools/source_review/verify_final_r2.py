"""Final retention R2 seal; source predicates only, no retention execution."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import sys
out=Path(__file__).resolve().parent
local=out.parent.parent
assert sys.version_info[:2]==(3,7),sys.version
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
baseline=read_json(out/'R2_PRIOR_PRESERVATION_BASELINE.json')
manifest=read_json(out/'INPUT_MANIFEST_R2.json')
for row in baseline:
    assert sha(Path(row['preserved_path']))==row['sha256'],row['preserved_path']
for row in manifest:
    assert sha(Path(row['path']))==row['sha256']==sha(out/row['snapshot']),row['path']
assert sum(Path(row['path'])!=Path(row['preserved_path']) for row in baseline)==2
for ext in ('md','json'):
    assert (out/('RETENTION_SOURCE_REVIEW_R2.'+ext)).read_bytes()==(out/('RETENTION_SOURCE_REVIEW.'+ext)).read_bytes()
assert (out/'RAW_RESPONSE_R2.md').read_bytes()==(out/'RETENTION_SOURCE_REVIEW_R2.md').read_bytes()
assert (out/'RAW_RESPONSE_R1.md').read_bytes()==(out/'RAW_RESPONSE.md').read_bytes()
assert (out/'INPUT_MANIFEST_R1.json').read_bytes()==(out/'INPUT_MANIFEST.json').read_bytes()
assert sha(out/'RETENTION_SOURCE_REVIEW_R1.json')=='4b86135d04d07012aa7e6ad65d2bf85506e9558c2d42e42e71be1a32ddfa68b3'
assert sha(out/'RETENTION_SOURCE_REVIEW_R1.md')=='09cf1d4243b314e1a30e1ce31f3f0365bdb0fdf3e108998e21d8072b8e305206'
assert (out/'R2_EXACT_REQUEST.txt').read_bytes()==(out.parent/'SOURCE_R2_REQUEST.txt').read_bytes()
review=read_json(out/'RETENTION_SOURCE_REVIEW_R2.json')
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict']=='PASS' and not review['blocking_findings']
assert review['reviewed_files']==[dict(path=row['path'],sha256=row['sha256']) for row in manifest]
assert not any(any(word in Path(row['path']).name.lower() for word in
    ('publish','handoff','observer','fit_wait')) for row in review['reviewed_files'])
assert review['actual_deletions']==0 and not review['actual_results_read'] and not review['actual_plan']
assert read_json(out/'R2_NATIVE_VERIFICATION_ATTEMPT_1.json')['result']['exit_code']==0
tree=ast.parse((out.parent/'retire_closed_pair_authorized.py').read_bytes())
required={(local/'postrun_tools'/name).resolve() for name in
    ('prepare_retention_plan.py','retire_closed_pair_authorized.py')}
gates={}
for line in (17,18,21):
    node=next(node for node in tree.body if isinstance(node,ast.Assert) and node.lineno==line)
    gates[str(line)]=bool(eval(compile(ast.Expression(node.test),'<source-review-predicate-only>','eval'),
        {'review':review,'required':required,'Path':Path}))
assert all(gates.values())
files=['RETENTION_SOURCE_REVIEW_R2.md','RETENTION_SOURCE_REVIEW_R2.json','RAW_RESPONSE_R2.md',
    'RETENTION_SOURCE_REVIEW.md','RETENTION_SOURCE_REVIEW.json','RETENTION_SOURCE_REVIEW_R1.md',
    'RETENTION_SOURCE_REVIEW_R1.json','RAW_RESPONSE_R1.md','INPUT_MANIFEST_R1.json',
    'INPUT_MANIFEST_R2.json','R2_EXACT_REQUEST.txt','run_R2.meta.json','R2_NATIVE_TOOL_RECEIPTS.json',
    'R2_SOURCE_READ_RECEIPTS.json','R2_NATIVE_VERIFICATION_ATTEMPT_1.json','R2_STATIC_VERIFICATION.json',
    'R2_PRIOR_PRESERVATION_BASELINE.json','R2_SOURCE.diff','verify_static_r2.py','verify_final_r2.py']
result=dict(scope='STATIC_ONLY',execution_scope='SOURCE_ONLY',
    generated_at=datetime.datetime.utcnow().isoformat()+'Z',python_executable=sys.executable,
    python_version=sys.version,prior_artifact_contents_preserved=len(baseline),
    prior_original_paths_unchanged=len(baseline)-2,review_aliases_versioned_as_R1=2,
    inputs_and_snapshots_exact=len(manifest),reviewed_files_exact=len(review['reviewed_files']),
    R2_aliases_exact=True,R2_raw_report_exact=True,R1_raw_manifest_snapshots_preserved=True,
    publisher_handoff_and_observer_proof_not_bound=True,
    exact_request_preserved=True,source_review_predicates_accept=gates,
    artifacts=[dict(path=str(out/name),sha256=sha(out/name),bytes=(out/name).stat().st_size) for name in files],
    actual_results_read=False,actual_plan_created=False,actual_decision_created=False,
    actual_archive_copies=0,actual_deletions=0,actual_SSH=False,actual_GPU=False,
    torch_imported='torch' in sys.modules,paramiko_imported='paramiko' in sys.modules,
    launch_approved=False,checks_passed=True)
assert not result['torch_imported'] and not result['paramiko_imported']
(out/'R2_FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,sort_keys=True))
