"""Preserve the actual fresh native audit call, not a forecast or verdict."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
trace = local / '.aris/traces/experiment-audit/2026-10-04_face_terminal_run01'
trace.mkdir(parents=True)
request = local / 'TERMINAL_AUDIT_REQUEST.txt'
message = '''Read D:/Program Files/UserCache/gb/codex/tmp/pvground_face_conditioned_20261004/TERMINAL_AUDIT_REQUEST.txt and all70 exact existing primary artifacts it lists. Perform the requested fresh experiment-audit A-F with direct source reading and deterministic local CPU recounts. Do not use conversation context, SSH, CUDA, model imports/execution, weights, private memory, raw goals, credentials/auth wrappers, other agents, or change primary artifacts. Write analysis/EXPERIMENT_AUDIT.md and analysis/EXPERIMENT_AUDIT.json plus audit-owned CPU reports. Distinguish actual dataset GT, native last/bbs and object-input protocol, development/pretrained-seen scope, within-Query versus control effects, architecture/capacity changes, scalar oracle/Mask recount versus raw replay, controller closure and metric-best-only retention. Same-family/provisional, requested Astra/max backend not independently attested. Report actual blockers and honest scope limits; no speculative fallback/compatibility machinery. Provide a final response.'''
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), tool='collaboration.spawn_agent',
    actual_task='/root/pvg_face_terminal_integrity', status='ACTUALLY_CALLED_RUNNING',
    arguments=dict(task_name='pvg_face_terminal_integrity', fork_turns='none', model='gpt-6-astra',
        reasoning_effort='max', message=message), referenced_request=request.read_text(encoding='utf-8'),
    request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),
    review_independence='same-family', acceptance_status='provisional', backend_sku_independently_attested=False)
(local / 'ACTUAL_TERMINAL_REVIEWER_CALL.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
(trace / '001-terminal-integrity.request.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
(trace / 'run.meta.json').write_text(json.dumps(dict(skill='experiment-audit',
    run_id='2026-10-04_face_terminal_run01', started_at=record['time_cst'], executor_family='openai',
    review_independence='same-family', acceptance_status='provisional', status='ACTUALLY_CALLED_RUNNING'), indent=2) + '\n', encoding='utf-8')
state_path = local / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(status='FACE_FULL_TERMINAL_CPU_RECOUNT_DONE_FRESH_AUDIT_RUNNING',
    native_observer_closed=True, sole_live_exec_cell_id=None, sole_live_native_session_id=None,
    controller_alive=False, actual_finished_cst='2026-10-04T20:08:33.480829+08:00',
    native_collection_session_closed=29290, formal_result_available=True, new_accuracy_result=True,
    formal_hits25=5615, formal_hits50=4496, goal_status='ACTIVE_UNMET',
    active_terminal_reviewer='/root/pvg_face_terminal_integrity', fresh_terminal_audit_complete=False,
    next_native_observer_check=None)
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (local / 'NEXT_CONTINUATION.md').open('a', encoding='utf-8') as stream:
    stream.write('\nObserver exec806/native84531 CLOSED normally. Actual face terminal20:08:33, formal5615/4496; CPU recount0 changes; owned904318-byte nonbest weight deleted by verified controller, prior4506 best retained. Collection29290 closed with38 text/row files and0 weights. Fresh /root/pvg_face_terminal_integrity audit ACTUALLY RUNNING. Never resume806 or restart the closed fit. Latest publication still geometry_readback/review_publication.json/main75eaf6b/doc50. Revised readback source AST-only; full factory/runner gate pending.\n')
with (Path('C:/Users/gb') / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround actual face fit closed20:08:33 CST, formal native bbs5615/4496 vs currentbest5616/4506. Remote controller verified and removed904318-byte owned nonbest head terminal; priorbest+G parents protected. Collected38 text/row files0 weights, CPU threshold recount0 changes; fresh Astra/max same-family/provisional face terminal audit actually called. Exec806/native84531 and collector29290 closed; no active fit. GoalACTIVE_UNMET.\n')
print(json.dumps(dict(status=record['status'], actual_task=record['actual_task'],
    observer_closed=True, formal_bbs=[5615,4496], goal_achieved=False)), flush=True)
