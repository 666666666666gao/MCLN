"""Record the actual native fresh-audit call and completed collection."""
import datetime
import json
from pathlib import Path

local = Path(__file__).parent
trace = local/'.aris/traces/experiment-audit/2026-10-04_terminal_run01'
trace.mkdir(parents=True)
message = r'Perform the installed experiment-audit fresh-context integrity review. Read and follow the actual request at C:\Users\gb\.codex\tmp\pvground_whole_mask_fit_20261003\TERMINAL_AUDIT_REQUEST.txt and every primary path it lists. Write the requested audit-owned reports under that directory\'s analysis folder, then report the actual verdict and blocking issues. Use only local read-only source/record checks; no model execution, SSH, CUDA, weights, credential/auth-wrapper reads, job changes, or sub-agents. This request is the skill-authorized gpt-6-astra/max/fork-none same-family/provisional review route.'
message = message.replace(chr(92)+chr(39), chr(39))
request = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    tool='collaboration.spawn_agent', arguments=dict(task_name='pvg_whole_range_terminal_integrity',
        fork_turns='none', model='gpt-6-astra', reasoning_effort='max', message=message),
    actual_task='/root/pvg_whole_range_terminal_integrity',
    referenced_request=(local/'TERMINAL_AUDIT_REQUEST.txt').read_text(encoding='utf-8'),
    reviewer_backend_sku_independently_verified=False,
    review_independence='same-family', acceptance_status='provisional')
with (trace/'001-terminal-integrity.request.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(request, indent=2)+'\n')
state_path = local/'active_formal_continuation_state.json'
state = json.loads(state_path.read_bytes())
wait = json.loads((local/'wait.json').read_bytes())
assert wait['status']=='complete' and wait['controller_exit']=='0'
latest = json.loads((local/wait['observations'][-1]['file']).read_bytes())
intake = json.loads((local/'complete/INTAKE.json').read_bytes())
state.update(time_cst=datetime.datetime.now().astimezone().isoformat(),
    goal_turn_result='PAIR_COMPLETE_ANALYZED_FRESH_AUDIT_RUNNING',
    latest_actual_observation=latest, next_scheduled_observation_cst=None,
    next_scheduled_delay_seconds=None, waiter_status='CLOSED_EXIT0 / 366 / f1c254',
    pair_formal_result_available=True, terminal_collection_executed=True,
    terminal_analysis_executed=True, fresh_terminal_audit_task=request['actual_task'],
    next_steps=['Receive the actual fresh terminal audit and address concrete blockers',
        'Publish complete pair evidence, current best and actual retention',
        'Prepare a bounded stable-G/new-head control from the actual results'],
    collector_native_session='13529 CLOSED_EXIT0 / fa4ca2',
    analyzer_native_execution='f37af5 CLOSED_EXIT0',
    owned_terminal_weights_retained=len(intake['weights_retained']),
    collection_weights_downloaded=0, goal_achieved=False)
state_path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write('\n- PV local/whole pair actually completed05:07:39+08:00, controller exit0; original observer5205 naturally closed05:10:32 (366/f1c254). '
        'Read-only collector13529 closed0/fa4ca2,63 actual text/source/row artifacts,0 weight download; analyzerf37af5 closed0. '
        'Formal9508 bbs local5603/4428,whole5594/4461: -9/+33 paired, whole strict383 repairs/350 damages; both belowG5615/4495 (-12/-67,-21/-34). '
        'SameQuery local4428->4428 (14/14),whole4471->4461 (13/23), median face moves2.705/2.828mm. '
        'Controller verified and retired each347116945B endpoint,total694233890B; protectedG SHA unchanged,no failed archives,0 owned weights. '
        'Fresh skill-required terminal auditor /root/pvg_whole_range_terminal_integrity actually spawned gpt-6-astra/max/forknone,81 primary paths; same-family/provisional, backendSKUunverified, verdict pending. '
        'Goal unmet; no model promotion/newNr/Sr. Next conditional step checks new head on stableG before extra quality readback; no active old source changes.\n')
print(json.dumps(dict(recorded_actual_audit_request=True, waiter_closed=True,
    collection_complete=True, analysis_complete=True, audit_verdict_available=False)))
