"""Preserve only the actual completed source-review response and reports."""
import datetime
import hashlib
import json
from pathlib import Path

local=Path(__file__).resolve().parent
review=json.loads((local/'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY'
assert review['review_independence']=='same-family' and review['acceptance_status']=='provisional'
assert review['verdict']=='PASS' and not review['blocking_findings']
for item in review['reviewed_files']:
    raw=Path(item['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==item['sha256']
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_task='/root/pvg_boundary_distribution_source_review',
    response=(local/'ACTUAL_CODE_REVIEWER_RESPONSE.txt').read_text(encoding='utf-8'),
    verdict=review['verdict'],blocking_findings=review['blocking_findings'],
    execution_scope='SOURCE_ONLY',review_independence='same-family',acceptance_status='provisional',
    backend_sku_independently_verified=False,
    reviewer_reports={name:dict(bytes=(local/name).stat().st_size,sha256=hashlib.sha256((local/name).read_bytes()).hexdigest())
        for name in ('EXPERIMENT_CODE_REVIEW.md','EXPERIMENT_CODE_REVIEW.json')})
trace=local/'.aris/traces/experiment-bridge/2026-10-04_boundary_source_run01'
with (trace/'004-code-review.response.json').open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(record,indent=2)+'\n')
state_path=local/'active_continuation_state.json';state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],status='SOURCE_REVIEW_PASS_ACTUAL_GPU_PROBES_PENDING',
    fresh_source_verdict='PASS',fresh_source_review_complete=True,
    fresh_source_blockers=0,review_scope='SOURCE_ONLY',native_preflight_started=False)
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+' actual boundary source review PASS0 blockers;48 input files,31 AST/10JSON checks; actual response/reports traced samefamily/provisional. Required finite continuous target witness repaired before probe. Still no native GPU/update/accuracy result; real two-arm sanity next.\n')
print(json.dumps(dict(actual_review='PASS',blocking=0,reviewed_files=len(review['reviewed_files']),source_only=True)))
