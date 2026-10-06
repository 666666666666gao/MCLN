"""Record the actually returned fresh audit; keep reviewed launch snapshots intact."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
report=root/'analysis/EXPERIMENT_AUDIT.json'
audit=json.loads(report.read_bytes())
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
assert audit['execution_scope']=='TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS'
assert audit['fresh_context']
call_path=root/'analysis/TERMINAL_REVIEW_CALL.json'
call=json.loads(call_path.read_bytes())
assert call['result_received'] is False
call.update(result_received=True,result_received_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_verdict=audit['verdict'],actual_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
    final_native_message='WARN; zero blocking.1197 formal SHA/bytes matched;9508+all NPZ independently recomputed; exact713/376,q005663/393;39 invalidIoU0;originalQuery/strictlabels unchanged,505 tinyIoU differences;raw replay limited8 M0;trainedbest4511 unmet50%.')
call_path.write_text(json.dumps(call,indent=2)+'\n',encoding='utf-8')
summary=json.loads((root/'analysis/SUMMARY.json').read_bytes())
record=dict(status='ACTUAL_CLOSED_DIAGNOSTIC_ACCEPTED_WITH_SCOPE_WARN',time_cst=call['result_received_cst'],
    review_call=str(call_path),audit_report=str(report),verdict=audit['verdict'],blocking_findings=[],
    trained_best_hits=[5616,4511],offline_exact_hits=[5598,4848],offline_quantile_hits=[5590,4781],
    offline_not_new_trained_result=True,goal_status='ACTIVE_UNMET',next_step='Reviewed native vs predicted-Mask spatial reference, not new ranking.')
(root/'analysis/TERMINAL_ACCEPTANCE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],status='MASK_EXTENT_CLOSED_ACCEPTED_PREPARING_REFERENCE_PAIR',
    owned_gpu_job_active=False,active_reviewer='/root/pvg_mask_reference_source_review',
    mask_extent_terminal_acceptance=str(root/'analysis/TERMINAL_ACCEPTANCE.json'),
    next_action='Complete fresh source review; actual2-step sanity before native/fused-reference pair.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (Path('C:/Users/gb/memory/2026-10-06.md')).open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': full9508 Mask extent diagnostic CLOSED, fresh Astra/max SOURCE+ACTUAL audit WARN0blocks. Learned5616/4511; offline exact5598/4848 (713repairs376damages), q5590/4781.39 actualemptyIoU0. Independent allmemberCPU, rawsort replay8 only. Not a trainedweight or goalachievement. Preparing native/fused spatial-reference pair with commonoutputreset; keepall256/nativebbs.\n')
print(json.dumps(record))
