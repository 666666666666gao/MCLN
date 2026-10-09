"""Final source-only seal; no actual retention tool is executed."""
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
baseline=read_json(out/'PRIOR_AUDIT_PRESERVATION_BASELINE.json')
manifest=read_json(out/'INPUT_MANIFEST.json')
for row in baseline:
    assert sha(Path(row['path']))==row['sha256'],row['path']
for row in manifest:
    assert sha(Path(row['path']))==row['sha256']==sha(out/row['snapshot']),row['path']
assert (out/'RAW_RESPONSE.md').read_bytes()==(out/'RETENTION_SOURCE_REVIEW.md').read_bytes()
assert (out/'EXACT_REQUEST.txt').read_bytes()==(out.parent/'SOURCE_AUDIT_REQUEST.txt').read_bytes()
review=read_json(out/'RETENTION_SOURCE_REVIEW.json')
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict']=='PASS' and not review['blocking_findings']
assert review['reviewed_files']==[dict(path=row['path'],sha256=row['sha256']) for row in manifest]
assert review['actual_deletions']==0 and not review['actual_results_read'] and not review['actual_plan']
assert read_json(out/'NATIVE_VERIFICATION_ATTEMPT_1.json')['result']['exit_code']==0
tree=ast.parse((out.parent/'retire_closed_pair_authorized.py').read_bytes())
required={(local/'postrun_tools'/name).resolve() for name in
    ('prepare_retention_plan.py','retire_closed_pair_authorized.py')}
gates={}
for line in (17,18,21):
    node=next(node for node in tree.body if isinstance(node,ast.Assert) and node.lineno==line)
    gates[str(line)]=bool(eval(compile(ast.Expression(node.test),'<source-review-predicate-only>','eval'),
        {'review':review,'required':required,'Path':Path}))
assert all(gates.values())
files=['RETENTION_SOURCE_REVIEW.md','RETENTION_SOURCE_REVIEW.json','RAW_RESPONSE.md',
    'INPUT_MANIFEST.json','EXACT_REQUEST.txt','run.meta.json','NATIVE_TOOL_RECEIPTS.json',
    'SOURCE_READ_RECEIPTS.json','NATIVE_VERIFICATION_ATTEMPT_1.json','STATIC_VERIFICATION.json',
    'PRIOR_AUDIT_PRESERVATION_BASELINE.json','verify_static.py','verify_final.py']
result=dict(scope='STATIC_ONLY',execution_scope='SOURCE_ONLY',
    generated_at=datetime.datetime.utcnow().isoformat()+'Z',python_executable=sys.executable,
    python_version=sys.version,prior_audit_artifacts_preserved=len(baseline),
    inputs_and_snapshots_exact=len(manifest),reviewed_files_exact=len(review['reviewed_files']),
    raw_report_identical=True,exact_request_preserved=True,source_review_predicates_accept=gates,
    artifacts=[dict(path=str(out/name),sha256=sha(out/name),bytes=(out/name).stat().st_size) for name in files],
    actual_results_read=False,actual_plan_created=False,actual_decision_created=False,
    actual_archive_copies=0,actual_deletions=0,actual_SSH=False,actual_GPU=False,
    torch_imported='torch' in sys.modules,paramiko_imported='paramiko' in sys.modules,
    launch_approved=False,checks_passed=True)
assert not result['torch_imported'] and not result['paramiko_imported']
(out/'FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,sort_keys=True))
