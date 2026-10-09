"""Seal R5 artifacts, preserve historical aliases, evaluate source-only gate."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import sys
out=Path(__file__).resolve().parent
runner=out.parent
assert sys.version_info[:2]==(3,7),sys.version
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
preserved=read_json(out/'R5_PRIOR_PRESERVATION_BASELINE.json')
manifest=read_json(out/'INPUT_MANIFEST_R5.json')
for item in preserved:
    assert sha(Path(item['preserved_path']))==item['sha256'],item['preserved_path']
for item in manifest:
    assert sha(Path(item['path']))==item['sha256']==sha(out/item['snapshot']),item['path']
alias_mappings=[item for item in preserved if Path(item['path'])!=Path(item['preserved_path'])]
assert len(alias_mappings)==2
for extension in ('md','json'):
    assert (out/('LAUNCH_SOURCE_REVIEW_R5.'+extension)).read_bytes()==(out/('LAUNCH_SOURCE_REVIEW.'+extension)).read_bytes()
assert (out/'RAW_RESPONSE_R5.md').read_bytes()==(out/'LAUNCH_SOURCE_REVIEW_R5.md').read_bytes()
assert sha(out/'LAUNCH_SOURCE_REVIEW_R4.md')=='726f0390ad06cab10beb95c354420c03324d6dba507ed583f69957d589f774a0'
assert sha(out/'LAUNCH_SOURCE_REVIEW_R4.json')=='711ff42a4d7c1e77d5636dc9adbaf101f23cc2529bd888bea402cb33a1589cdb'
assert (out/'R5_EXACT_REQUEST.txt').read_bytes()==(runner/'LAUNCH_R5_REQUEST.txt').read_bytes()
review=read_json(out/'LAUNCH_SOURCE_REVIEW_R5.json')
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict']=='PASS' and not review['blocking_findings']
assert review['reviewed_files']==[dict(path=row['path'],sha256=row['sha256']) for row in manifest]
launch=ast.parse((runner/'launch_span_authorized.py').read_bytes())
required={ (runner/name).resolve() for name in ('finalize_parent.py','launch_span_authorized.py','span_controller.py','pair_spec_template.json') }
gates={}
for line in (28,29,32):
    node=next(node for node in launch.body if isinstance(node,ast.Assert) and node.lineno==line)
    gates[str(line)]=bool(eval(compile(ast.Expression(node.test),'<source-review-predicate-only>','eval'),
        {'review':review,'required':required,'Path':Path}))
assert all(gates.values())
assert not (runner/'pair_spec.json').exists() and not (runner/'preflight_complete/preflight.json').exists()
assert not review['actual_deployment'] and not review['actual_M0'] and not review['launch_approved']
assert read_json(out/'R5_NATIVE_VERIFICATION_ATTEMPT_1.json')['result']['exit_code']==0
files=['LAUNCH_SOURCE_REVIEW_R5.md','LAUNCH_SOURCE_REVIEW_R5.json','RAW_RESPONSE_R5.md',
    'LAUNCH_SOURCE_REVIEW.md','LAUNCH_SOURCE_REVIEW.json','LAUNCH_SOURCE_REVIEW_R4.md',
    'LAUNCH_SOURCE_REVIEW_R4.json','INPUT_MANIFEST_R5.json','R5_EXACT_REQUEST.txt',
    'run_R5.meta.json','R5_NATIVE_TOOL_RECEIPTS.json','R5_SOURCE_READ_RECEIPTS.json',
    'R5_NATIVE_VERIFICATION_ATTEMPT_1.json','verify_static_r5.py','R5_SOURCE.diff',
    'R5_STATIC_VERIFICATION.json','R5_PRIOR_PRESERVATION_BASELINE.json','verify_final_r5.py']
result=dict(scope='STATIC_ONLY',execution_scope='SOURCE_ONLY',
    generated_at=datetime.datetime.utcnow().isoformat()+'Z',python_executable=sys.executable,
    python_version=sys.version,prior_artifact_contents_preserved=len(preserved),
    prior_original_paths_unchanged=len(preserved)-2,old_aliases_preserved_as_R4_files=2,
    inputs_and_snapshots_exact=len(manifest),reviewed_files_exact=len(review['reviewed_files']),
    R5_aliases_exact=True,R5_raw_report_exact=True,R4_FAIL_exactly_preserved=True,
    source_review_predicates_accept=gates,final_spec_absent=True,actual_M0_proof_absent=True,
    exact_request_preserved=True,
    artifacts=[dict(path=str(out/name),sha256=sha(out/name),bytes=(out/name).stat().st_size) for name in files],
    torch_imported='torch' in sys.modules,paramiko_imported='paramiko' in sys.modules,
    actual_deployment=False,actual_M0=False,launch_approved=False,checks_passed=True)
assert not result['torch_imported'] and not result['paramiko_imported']
(out/'R5_FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,sort_keys=True))
