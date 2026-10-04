"""Save the actual fresh full-preflight reviewer call and private request trace."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
request = local / 'FULL_SOURCE_REVIEW_REQUEST.txt'
trace = local / '.aris/traces/experiment-bridge/2026-10-04_readback_full_preflight_run01'
trace.mkdir(parents=True)
message = '''Read D:/Program Files/UserCache/gb/codex/tmp/pvground_geometry_readback_20261004/FULL_SOURCE_REVIEW_REQUEST.txt and ALL32 exact existing source/configuration/interface files listed there. Perform its fresh experiment-bridge source gate on the actual frozen4506 factory, CPU/two-update real-batch caller, native semantic defer port, readback/helpers and deployment controller. Do not use prior conversation or old partial PASS. No SSH, network, CUDA, model/weight execution/imports, credentials, raw goals, memory/auth wrappers, other agents or primary file edits. Stdlib inspection/AST/arithmetic is allowed. Write only READBACK_FULL_SOURCE_REVIEW.md and READBACK_FULL_SOURCE_REVIEW.json in that directory, with SOURCE_ONLY, exact reviewed_files identities, blocking_findings and nonblocking_findings. Scope is CPU/two-update preflight deployment only, no formal training or runtime/accuracy approval. Same-family/provisional; requested Astra/max backend/effort is not independently attested. Report actual correctness defects, no speculative fallback/compatibility frameworks or new hash schemes. Provide a final response.'''
stamp = datetime.datetime.now().astimezone().isoformat()
record = dict(time_cst=stamp, tool='collaboration.spawn_agent', status='ACTUALLY_CALLED_RUNNING',
    actual_task='/root/pvg_readback_full_preflight_review',
    arguments=dict(task_name='pvg_readback_full_preflight_review', fork_turns='none', model='gpt-6-astra',
        reasoning_effort='max', message=message), referenced_request=request.read_text(encoding='utf-8'),
    request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(), review_independence='same-family',
    acceptance_status='provisional', backend_sku_independently_attested=False)
(local / 'FULL_SOURCE_REVIEW_CALL.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
(trace / '001-full-source.request.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
(trace / 'run.meta.json').write_text(json.dumps(dict(skill='experiment-bridge',
    run_id='2026-10-04_readback_full_preflight_run01', started_at=stamp, status=record['status'],
    review_independence='same-family', acceptance_status='provisional'), indent=2) + '\n', encoding='utf-8')
state_path = local.parent / 'pvground_face_conditioned_20261004/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(active_readback_full_source_reviewer=record['actual_task'],
    readback_current_revision_review='FULL_PREFLIGHT_SOURCE_GATE_ACTUALLY_RUNNING',
    readback_full_caller=str(local / 'run_readback_preflight.py'), readback_runtime_checked=False)
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=record['status'], actual_task=record['actual_task'], cpu_gpu_run=False)), flush=True)
