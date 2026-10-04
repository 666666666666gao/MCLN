"""Record the actual native reviewer request, without claiming its result."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
trace = local / '.aris/traces/experiment-bridge/2026-10-04_face_source_run01'
trace.mkdir(parents=True)
message = ('Perform the installed experiment-bridge fresh-context source review. Read and follow the actual request at '
    'D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_face_conditioned_20261004\\CODE_REVIEW_REQUEST.txt and every existing primary path listed. '
    'Write only the requested review-owned Markdown/JSON reports, then send the actual verdict and concrete blocking issues. '
    'No SSH, model imports/execution, CUDA, checkpoint/credential/auth-wrapper/private-goal reads, job changes or sub-agents. '
    'Use the skill-authorized Astra/max/fork-none same-family/provisional route; this review does not replace actual real-model two-step probes.')
request = local / 'CODE_REVIEW_REQUEST.txt'
assert str(request) in message
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_tool='collaboration.spawn_agent', actual_arguments=dict(task_name='pvg_face_conditioned_source_review',
        fork_turns='none', model='gpt-6-astra', reasoning_effort='max', message=message),
    actual_tool_receipt=dict(task_name='/root/pvg_face_conditioned_source_review'),
    request_path=str(request), request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),
    review_independence='same-family', acceptance_status='provisional', backend_sku_independently_verified=False)
(trace / '001-code-review.request.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(actual_task=record['actual_tool_receipt']['task_name'], result_available=False)))
