"""Final R4 preservation and machine gate verification; stdlib only."""
import ast
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import sys
out=Path(__file__).resolve().parent
assert sys.version_info[:2]==(3,7),sys.version
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
prior=read_json(out/'R4_PRIOR_PRESERVATION_BASELINE.json')
manifest=read_json(out/'INPUT_MANIFEST_R4.json')
for item in prior:
    assert sha(Path(item['path']))==item['sha256'],item['path']
for item in manifest:
    assert sha(Path(item['path']))==item['sha256']==sha(out/item['snapshot']),item['path']
assert (out/'LAUNCH_SOURCE_REVIEW.md').read_bytes()==(out/'RAW_RESPONSE_R4.md').read_bytes()
assert (out/'R4_EXACT_REQUEST.txt').read_bytes()==(out.parent/'LAUNCH_SOURCE_AUDIT_REQUEST.txt').read_bytes()
review=read_json(out/'LAUNCH_SOURCE_REVIEW.json')
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict']=='FAIL'
assert len(review['blocking_findings'])==1 and review['blocking_findings'][0]['id']=='R4-B1'
assert review['reviewed_files']==[dict(path=row['path'],sha256=row['sha256']) for row in manifest]
launcher=ast.parse((out.parent/'launch_span_authorized.py').read_bytes())
gate=next(node for node in launcher.body if isinstance(node,ast.Assert) and node.lineno==28)
assert eval(compile(ast.Expression(gate.test),'<isolated-launch-source-gate>','eval'),{'review':review}) is False
assert not review['actual_deployment'] and not review['actual_M0'] and not review['launch_approved']
assert read_json(out/'R4_NATIVE_VERIFICATION_ATTEMPT_1.json')['result']['exit_code']==1
assert read_json(out/'R4_NATIVE_VERIFICATION_ATTEMPT_2.json')['result']['exit_code']==0
original=(out/'R4_VERIFY_ATTEMPT_1_ORIGINAL.py').read_text(encoding='utf-8')
current=(out/'verify_static_r4.py').read_text(encoding='utf-8')
diff=''.join(difflib.unified_diff(original.splitlines(True),current.splitlines(True),
    fromfile='R4_VERIFY_ATTEMPT_1_ORIGINAL.py',tofile='verify_static_r4.py'))
(out/'R4_VERIFIER_CORRECTION.diff').write_text(diff,encoding='utf-8')
files=['LAUNCH_SOURCE_REVIEW.md','LAUNCH_SOURCE_REVIEW.json','RAW_RESPONSE_R4.md',
    'INPUT_MANIFEST_R4.json','R4_EXACT_REQUEST.txt','run_R4.meta.json',
    'R4_NATIVE_TOOL_RECEIPTS.json','R4_SOURCE_READ_RECEIPTS.json',
    'R4_NATIVE_VERIFICATION_ATTEMPT_1.json','R4_NATIVE_VERIFICATION_ATTEMPT_2.json',
    'R4_VERIFY_ATTEMPT_1_ORIGINAL.py','verify_static_r4.py','R4_VERIFIER_CORRECTION.diff',
    'R4_STATIC_VERIFICATION.json','R4_PRIOR_PRESERVATION_BASELINE.json','verify_final_r4.py']
result=dict(scope='STATIC_ONLY',execution_scope='SOURCE_ONLY',
    generated_at=datetime.datetime.utcnow().isoformat()+'Z',python_executable=sys.executable,
    python_version=sys.version,prior_artifacts_preserved=len(prior),inputs_and_snapshots_exact=len(manifest),
    reviewed_file_hashes_exact=len(review['reviewed_files']),raw_report_identical=True,
    exact_request_preserved=True,launch_source_review_gate_accepts=False,
    failed_invocation_and_original_verifier_preserved=True,
    artifacts=[dict(path=str(out/name),bytes=(out/name).stat().st_size,sha256=sha(out/name)) for name in files],
    torch_imported='torch' in sys.modules,paramiko_imported='paramiko' in sys.modules,
    actual_deployment=False,actual_M0=False,launch_approved=False,checks_passed=True)
assert not result['torch_imported'] and not result['paramiko_imported']
(out/'R4_FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,sort_keys=True))
