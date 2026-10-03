"""Preserve the actual fresh reviewer result, without relabeling its verdict."""
import datetime
import hashlib
import json
from pathlib import Path

local=Path(__file__).parent
audit=json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['review_independence']=='same-family' and audit['acceptance_status']=='provisional'
trace=local/'.aris/traces/experiment-audit/2026-10-04_terminal_run01'
response='''Audit complete: **WARN**, **no blocking issues**. Same-family review; acceptance remains provisional.

- All 81 primary files unchanged; 343 CPU checks passed.
- Formal bbs: local **5603/4428**, whole **5594/4461**. Whole gains 33 strict hits over local but remains below recorded original G; target unmet.
- Qualifications: one seed, development/pretrained-seen scope, non-bitwise starts, and no raw-Mask/full-candidate reconstruction.
- Two owned endpoint deletion receipts reconcile; protected G remains retained. No weights or jobs touched.

Reports: [Audit](/C:/Users/gb/.codex/tmp/pvground_whole_mask_fit_20261003/analysis/EXPERIMENT_AUDIT.md) · [JSON](/C:/Users/gb/.codex/tmp/pvground_whole_mask_fit_20261003/analysis/EXPERIMENT_AUDIT.json). Three supporting audit-owned CPU reports are in the same folder.'''
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_task='/root/pvg_whole_range_terminal_integrity',response=response,
    verdict=audit['verdict'],blocking_issues=audit['blocking_issues'],
    reviewer_reports={name:dict(bytes=(local/'analysis'/name).stat().st_size,
        sha256=hashlib.sha256((local/'analysis'/name).read_bytes()).hexdigest())
        for name in ('EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json')},
    requested_model='gpt-6-astra',requested_reasoning='max',fork_turns='none',
    review_independence='same-family',acceptance_status='provisional',
    backend_sku_independently_verified=False)
with (trace/'002-terminal-integrity.response.json').open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(record,indent=2)+'\n')
state_path=local/'active_formal_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],goal_turn_result='PAIR_COMPLETE_ANALYZED_FRESH_AUDIT_WARN0',
    fresh_terminal_audit_verdict=audit['verdict'],fresh_terminal_audit_blockers=len(audit['blocking_issues']),
    next_steps=['Publish actual complete source pair and best-weight retention',
        'Fresh-review stable-G/refiner-only sanity source before any deployment'],
    goal_achieved=False)
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\n- Actual fresh terminal auditor completed WARN/0 blocking,343 deterministic checks,81 inputs/63.48MB,46564 evaluated rows and7446fit records. '
        'Exact actual native response/reports preserved under F/.aris/traces/experiment-audit/2026-10-04_terminal_run01. Same-family/provisional, no backendSKU independence. '
        'Saved selected final+coarse boxes186256CPU reconstructions,threshold disagreements0; rawMask/fullcandidate/ranking/optimizer not replayed. '
        'Both new endpoints remain belowG; not promoted. Next stable-G only-existing-refiner sanity source prepared locally in pvground_range_head_only_20261004; no deployment/updates yet.\n')
print(json.dumps(dict(verdict=audit['verdict'],blocking=len(audit['blocking_issues']),actual_response_preserved=True)))
