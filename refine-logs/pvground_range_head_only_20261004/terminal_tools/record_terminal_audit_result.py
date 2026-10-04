"""Preserve the actual completed fresh reviewer response, never a forecast."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
audit = json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['review_independence'] == 'same-family' and audit['acceptance_status'] == 'provisional'
response = (local/'ACTUAL_TERMINAL_REVIEWER_RESPONSE.txt').read_text(encoding='utf-8')
assert response.strip()
trace = local/'.aris/traces/experiment-audit/2026-10-04_head_only_terminal_run01'
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_task='/root/pvg_head_only_terminal_integrity',response=response,
    verdict=audit['verdict'],blocking_issues=audit['blocking_issues'],
    reviewer_reports={name:dict(bytes=(local/'analysis'/name).stat().st_size,
        sha256=hashlib.sha256((local/'analysis'/name).read_bytes()).hexdigest())
        for name in ('EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json')},
    requested_model='gpt-6-astra',requested_reasoning='max',fork_turns='none',
    review_independence='same-family',acceptance_status='provisional',
    backend_sku_independently_verified=False)
with (trace/'002-terminal-integrity.response.json').open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(record,indent=2)+'\n')
state_path = local/'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],status='FULL_PAIR_COMPLETE_FRESH_AUDIT_RECORDED',
    goal_turn_result='PAIR_COMPLETE_ACTUAL_FRESH_AUDIT_RECORDED',
    fresh_terminal_audit_verdict=audit['verdict'],fresh_terminal_audit_blockers=len(audit['blocking_issues']),
    fresh_terminal_audit_complete=True,active_controller=None,latest_observer_status='74041_CLOSED_EXIT0',
    next_scheduled_remote_check_cst=None,goal_status='ACTIVE_UNMET',new_goal_achieved=False,
    next_steps=['Publish actual terminal §20.376.42 from actual audit','Finish new boundary source review and real probes'])
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+' actual fresh H terminal review '+audit['verdict']+'/'+str(len(audit['blocking_issues']))+' blocking, exact response/reports preserved. Review samefamily/provisional; no model replay or backend independence inferred. GoalACTIVE_UNMET.\n')
print(json.dumps(dict(verdict=audit['verdict'],blocking=len(audit['blocking_issues']),actual_response_preserved=True)))
