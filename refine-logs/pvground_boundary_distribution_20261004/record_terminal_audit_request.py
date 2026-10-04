"""Record a real fresh-review call after collection and analysis, never a forecast."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
wait = json.loads((local/'wait.json').read_bytes())
intake = json.loads((local/'complete/INTAKE.json').read_bytes())
summary = json.loads((local/'analysis/SUMMARY.json').read_bytes())
receipt = json.loads((local/'ACTUAL_TERMINAL_REVIEWER_RECEIPT.json').read_bytes())
assert wait['status'] == 'complete' and wait['controller_exit'] == '0'
assert intake['controller_exit'] == 0 and not intake['controller_alive']
assert receipt['actual_task'] == '/root/pvg_boundary_terminal_integrity'
assert receipt['requested_model'] == 'gpt-6-astra' and receipt['requested_reasoning'] == 'max'
assert receipt['fork_turns'] == 'none'
message = (local/'ACTUAL_TERMINAL_REVIEWER_MESSAGE.txt').read_text(encoding='utf-8')
request_path = local/'TERMINAL_AUDIT_REQUEST.txt'
assert str(request_path) in message and message.strip()
trace = local/'.aris/traces/experiment-audit/2026-10-04_boundary_terminal_run01'
trace.mkdir(parents=True)
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    tool='collaboration.spawn_agent', actual_task=receipt['actual_task'],
    arguments=dict(task_name='pvg_boundary_terminal_integrity', fork_turns='none',
        model='gpt-6-astra', reasoning_effort='max', message=message),
    actual_spawn_receipt=receipt,
    referenced_request=request_path.read_text(encoding='utf-8'),
    referenced_request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
    review_independence='same-family', acceptance_status='provisional',
    reviewer_backend_sku_independently_verified=False)
with (trace/'001-terminal-integrity.request.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(record, indent=2)+'\n')
state_path = local/'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'], status='FULL_PAIR_COMPLETE_FRESH_AUDIT_RUNNING',
    fresh_terminal_audit_task=receipt['actual_task'], fresh_terminal_audit_complete=False,
    fresh_terminal_audit_verdict=None, formal_metrics_available=True,
    active_controller=None, next_scheduled_remote_check_cst=None,
    goal_status='ACTIVE_UNMET')
state_path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(actual_request_preserved=True, actual_task=receipt['actual_task'],
    formal_result_available=True, retained_best=summary['retained_best']['name'],
    review_verdict_available=False, goal_achieved=False)))
