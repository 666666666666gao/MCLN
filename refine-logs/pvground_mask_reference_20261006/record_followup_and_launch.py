"""Record the returned minimal-fix followup and actual launched M0 process."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
review=json.loads((root/'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
launch=json.loads((root/'preflight_launch.json').read_bytes())
assert launch['status']=='PREFLIGHT_LAUNCHED_NOT_COMPLETED' and launch['process']
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    task_name='/root/pvg_mask_reference_source_review',native_followup_result_received=True,
    original_fresh_context_review=True,followup_same_context=True,execution_scope='SOURCE_ONLY',
    acceptance_status='provisional',review_independence='same-family',backend_identity='not_attested',
    minimal_failure_and_fix=str(root/'DEPLOY_PROBE_FAILURE.json'),
    actual_current_report_sha256=hashlib.sha256((root/'SOURCE_REVIEW.json').read_bytes()).hexdigest(),
    verdict=review['verdict'],unresolved_blocking_findings=[],preflight_actual_launch=launch,
    accuracy_result=False,optimizer_updates_planned_per_arm=2)
(root/'SOURCE_REVIEW_FOLLOWUP_CALL.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],status='MASK_REFERENCE_TWO_STEP_PREFLIGHT_LAUNCHED',
    owned_gpu_job_active=True,active_reviewer=None,reference_preflight_launch=str(root/'preflight_launch.json'),
    next_action='One sole estimated preflight observer;collect actual2-step closure before fit launch. Trainedbest4511 unchanged.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (Path('C:/Users/gb/memory/2026-10-06.md')).open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': Mask spatial-reference native/fused pair freshsourcePASS; readonlyprior-status probe firstfailed beforemkdir/GPU, actualcompleted/exit_code fix reviewedPASS. ActualM0 controller '+launch['process']+' launched,2steps/arm planned; no accuracyyet. Common2outputreset, retained8hidden,456102params; no newranking/teacher, all256. Firstcheck720s,240safter; fullfitnotstarted.\n')
print(json.dumps(dict(status=state['status'],actual_launch=launch,source_verdict=review['verdict'])))
