"""Preserve the actual completed reviewer response and the auditor's own verdict."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
audit = json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['review_independence'] == 'same-family' and audit['acceptance_status'] == 'provisional'
response_path = local/'ACTUAL_TERMINAL_REVIEWER_RESPONSE.txt'
response = response_path.read_text(encoding='utf-8')
assert response.strip()
trace = local/'.aris/traces/experiment-audit/2026-10-04_boundary_terminal_run01'
request = json.loads((trace/'001-terminal-integrity.request.json').read_bytes())
assert request['actual_task'] == '/root/pvg_boundary_terminal_integrity'
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_task=request['actual_task'], response=response,
    response_sha256=hashlib.sha256(response_path.read_bytes()).hexdigest(),
    verdict=audit['verdict'], blocking_issues=audit['blocking_issues'],
    reviewer_reports={name:dict(bytes=(local/'analysis'/name).stat().st_size,
        sha256=hashlib.sha256((local/'analysis'/name).read_bytes()).hexdigest())
        for name in ('EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json')},
    requested_model='gpt-6-astra', requested_reasoning='max', fork_turns='none',
    review_independence='same-family', acceptance_status='provisional',
    backend_sku_independently_verified=False)
with (trace/'002-terminal-integrity.response.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(record, indent=2)+'\n')
state_path = local/'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'], status='FULL_PAIR_COMPLETE_FRESH_AUDIT_RECORDED',
    fresh_terminal_audit_complete=True, fresh_terminal_audit_verdict=audit['verdict'],
    fresh_terminal_audit_blockers=len(audit['blocking_issues']), goal_status='ACTIVE_UNMET')
state_path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(actual_response_preserved=True, verdict=audit['verdict'],
    blocking_issues=len(audit['blocking_issues']), goal_achieved=False)))
