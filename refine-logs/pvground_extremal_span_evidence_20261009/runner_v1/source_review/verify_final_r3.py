"""Seal R3 review artifacts and check preservation without experiment imports."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import sys
out=Path(__file__).resolve().parent
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert sys.version_info[:2]==(3,7),sys.version
prior=read_json(out/'R3_PRIOR_PRESERVATION_BASELINE.json')
manifest=read_json(out/'INPUT_MANIFEST_R3.json')
for item in prior:
    assert sha(Path(item['path']))==item['sha256'],item['path']
for item in manifest:
    assert sha(Path(item['path']))==item['sha256']==sha(out/item['snapshot']),item['path']
assert (out/'RAW_RESPONSE_R3.md').read_bytes()==(out/'SOURCE_REVIEW_R3.md').read_bytes()
assert (out/'R3_EXACT_REQUEST.txt').read_bytes()==(out.parent/'SOURCE_R3_REQUEST.txt').read_bytes()
original=(out/'R3_VERIFY_ATTEMPT_1_ORIGINAL.py').read_text(encoding='utf-8')
current=(out/'verify_static_r3.py').read_text(encoding='utf-8')
assert original.replace("Path(n['path']).parent==runner and","Path(n['path']).parent.resolve()==runner and")==current
assert read_json(out/'R3_NATIVE_VERIFICATION_ATTEMPT_1.json')['result']['exit_code']==1
assert read_json(out/'R3_NATIVE_VERIFICATION_ATTEMPT_2.json')['result']['exit_code']==0
verification=read_json(out/'R3_STATIC_VERIFICATION.json')
assert verification['checks_passed'] and len(verification['publication_files'])==13
report=read_json(out/'SOURCE_REVIEW_R3.json')
assert report['source_verdict']=='PASS' and report['overall_verdict']=='WARN'
assert not report['actual_M0'] and not report['launch_approved']
files=[
'SOURCE_REVIEW_R3.md','RAW_RESPONSE_R3.md','SOURCE_REVIEW_R3.json',
'INPUT_MANIFEST_R3.json','R3_EXACT_REQUEST.txt','run_R3.meta.json',
'R3_NATIVE_TOOL_RECEIPTS.json','R3_NATIVE_VERIFICATION_ATTEMPT_1.json',
'R3_NATIVE_VERIFICATION_ATTEMPT_2.json','R3_STATIC_VERIFICATION.json',
'R3_VERIFY_ATTEMPT_1_ORIGINAL.py','verify_static_r3.py',
'R3_LOOP.diff','R3_PRIOR_PRESERVATION_BASELINE.json','verify_final_r3.py']
result=dict(
    scope='STATIC_ONLY',generated_at=datetime.datetime.utcnow().isoformat()+'Z',
    python_executable=sys.executable,python_version=sys.version,
    prior_R1_R2_artifacts_preserved=len(prior),R3_inputs_and_snapshots_exact=len(manifest),
    publication_contract_files=13,report_and_raw_identical=True,exact_request_preserved=True,
    failed_R3_verifier_original_and_receipt_preserved=True,
    reviewer_only_verifier_correction='resolve both parent paths across existing junction',
    artifacts=[dict(path=str(out/name),sha256=sha(out/name),bytes=(out/name).stat().st_size) for name in files],
    torch_imported='torch' in sys.modules,numpy_imported='numpy' in sys.modules,
    actual_M0=False,launch_approved=False,checks_passed=True)
assert not result['torch_imported'] and not result['numpy_imported']
(out/'R3_FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,sort_keys=True))
