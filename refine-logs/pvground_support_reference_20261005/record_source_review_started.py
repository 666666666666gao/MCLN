"""Record the actual native SOURCE-review creation and continuation state."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

root=Path(__file__).resolve().parent
trace=root/'.aris/traces/experiment-bridge/2026-10-05_run01'
request=trace/'001-source.request.json'
call_data=json.loads(request.read_bytes())
now=datetime.now(timezone(timedelta(hours=8))).isoformat()
call={'time_cst':now,'tool':'collaboration.spawn_agent','agent_id':call_data['returned_task_name'],
    'reviewer_model_requested':call_data['model'],'reviewer_reasoning_requested':call_data['reasoning_effort'],
    'fork_turns':'none','fresh_context':True,'backend_model_identity_attested':False,
    'review_independence':'same-family','acceptance_status':'provisional','result_received':False,
    'status':'ACTUAL_SOURCE_REVIEW_RUNNING','request_path':str(request),
    'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest()}
(root/'SOURCE_REVIEW_CALL.json').write_text(json.dumps(call,indent=2)+'\n',encoding='utf-8')
(trace/'001-source.meta.json').write_text(json.dumps(call,indent=2)+'\n',encoding='utf-8')
(trace/'run.meta.json').write_text(json.dumps({'skill':'experiment-bridge','run_id':'2026-10-05_run01',
    'started_at':now,'executor_model_identity_attested':False,'review_independence':'same-family',
    'acceptance_status':'provisional','project_dir':str(root)},indent=2)+'\n',encoding='utf-8')
(root/'.aris/meta').mkdir(parents=True,exist_ok=True)
with (root/'.aris/meta/events.jsonl').open('a',encoding='utf-8') as stream:
    stream.write(json.dumps({'event':'review_trace','skill':'experiment-bridge','purpose':'source-correctness',
        'agent_id':call['agent_id'],'status':'running','time_cst':now,'trace_path':str(trace)})+'\n')
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(time_cst=now,status='SUPPORT_REFERENCE_SOURCE_REVIEW_ACTIVE',active_reviewer=call['agent_id'],
    owned_gpu_job_active=False,support_reference_root=str(root),support_reference_source_review_pending=True,
    support_reference_deployed=False,protected_best_hits=[5616,4511],
    next_action='Complete actual fresh SOURCE review of minimal own/fused support reference; fix concrete blocking defects, then real two-update GPU sanity before any formal training.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'actual_agent':call['agent_id'],'status':call['status'],'GPU_launched':False}))
