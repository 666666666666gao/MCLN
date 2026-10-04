"""Preserve the actual fresh-audit call after completed collection/analysis."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

local = Path(__file__).parent
wait = json.loads((local / 'wait.json').read_bytes())
intake = json.loads((local / 'complete/INTAKE.json').read_bytes())
summary = json.loads((local / 'analysis/SUMMARY.json').read_bytes())
assert wait['status'] == 'complete' and wait['controller_exit'] == '0'
assert intake['controller_exit'] == 0 and not intake['controller_alive']
trace = local / '.aris/traces/experiment-audit/2026-10-04_head_only_terminal_run01'
trace.mkdir(parents=True)
message = r"Perform the installed experiment-audit fresh-context integrity review. Read and follow the actual request at C:\Users\gb\.codex\tmp\pvground_range_head_only_20261004\TERMINAL_AUDIT_REQUEST.txt and every primary path it lists. Write the requested audit-owned reports under that directory's analysis folder, then report the actual verdict and blocking issues. Use only local read-only source/record checks; no model execution, SSH, CUDA, weights, credential/auth-wrapper reads, job changes, or sub-agents. This request is the skill-authorized gpt-6-astra/max/fork-none same-family/provisional review route."
stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
request = dict(time_cst=stamp, tool='collaboration.spawn_agent',
    arguments=dict(task_name='pvg_head_only_terminal_integrity', fork_turns='none',
        model='gpt-6-astra', reasoning_effort='max', message=message),
    actual_spawn_receipt={'task_name': '/root/pvg_head_only_terminal_integrity'},
    actual_task='/root/pvg_head_only_terminal_integrity',
    referenced_request=(local / 'TERMINAL_AUDIT_REQUEST.txt').read_text(encoding='utf-8'),
    reviewer_backend_sku_independently_verified=False,
    review_independence='same-family', acceptance_status='provisional')
with (trace / '001-terminal-integrity.request.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(request, indent=2) + '\n')
state_path = local / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=stamp, goal_turn_result='PAIR_COMPLETE_COLLECTED_ANALYZED_FRESH_AUDIT_RUNNING',
    status='FULL_PAIR_COMPLETE_AUDIT_RUNNING', sole_native_wait_cell=None,
    pending_poll_evidence=None, observer_session_status='74041_CLOSED_EXIT0',
    terminal_observer_native_cell='555_CLOSED_EXIT0', next_scheduled_observation_cst=None,
    next_scheduled_delay_seconds=None, whole_formal_accuracy_result_available=True,
    terminal_collection_execution={'status': 'ACTUAL_ONCE_CLOSED_EXIT0', 'session_id': 2950,
        'files': len(intake['files']), 'weights_downloaded': 0},
    terminal_analysis_execution={'status': 'ACTUAL_ONCE_CLOSED_EXIT0', 'cell_id': '559'},
    fresh_terminal_audit_task=request['actual_task'], fresh_terminal_audit_verdict=None,
    fresh_terminal_audit_complete=False, local_formal_bbs_hits=[5588, 4447],
    whole_formal_bbs_hits=[5589, 4456], retained_metric_best='original_g_5615_4495',
    goal_status='ACTIVE_UNMET', new_goal_achieved=False,
    next_steps=['Receive actual fresh terminal audit and resolve concrete blockers',
        'Record actual reviewer response and publish completed pair plus actual next decision',
        'Develop the next boundary experiment from preserved original G; no promotion or new Nr/Sr results'])
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write('\n- H fixed-G local/whole controller completed10:32:25.061015+08:00, all4phases exit0; observer74041 naturally closed0 (555/2694ce). '
        'Collector2950 closed0 once,63 text/source/row artifacts,0 weight download; analyzer559 closed0. '
        'Formalbbs local5588/4447,whole5589/4456 (+1/+9,strict26repairs/17damages), both belowG5615/4495. '
        'SameQuery whole4495->4456 (84/123), median face16.843475mm; coarseFull2567884->7974 but actual-39. '
        'Both4921029B deltas retired,total9842058B,ownedweights0,protectedG SHA unchanged. Collection snapshot system287125504B,data2505240576B,GPU0%/1MiB/no compute process. '
        'Actual fresh auditor /root/pvg_head_only_terminal_integrity spawned Astra/max/forknone,84 existing primary paths; same-family/provisional, verdict pending. '
        'Actual NEXT_EXPERIMENT_DECISION.md written: preserveG, stop unchanged statistical-head continuation, next explicit boundary/control; quality callback and teacher separately conditional. Goal ACTIVE_UNMET.\n')
print(json.dumps(dict(actual_audit_request_preserved=True, actual_task=request['actual_task'],
    collection_complete=True, analysis_complete=True, verdict_available=False,
    retained_best=summary['retained_best']['name'], goal_achieved=False)))
