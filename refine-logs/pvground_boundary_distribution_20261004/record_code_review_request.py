"""Record only the actual native fresh-agent request and task receipt."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
trace = local/'.aris/traces/experiment-bridge/2026-10-04_boundary_source_run01'
trace.mkdir(parents=True)
request = local/'CODE_REVIEW_REQUEST.txt'
message = ('Perform the installed experiment-bridge fresh-context source review. Read and follow the actual request at '
    'C:\\Users\\gb\\.codex\\tmp\\pvground_boundary_distribution_20261004\\CODE_REVIEW_REQUEST.txt and every existing primary path listed. '
    'Write only the requested review-owned Markdown/JSON reports, then send the actual verdict and concrete blocking issues. '
    'No SSH, model imports/execution, CUDA, checkpoint/credential/auth-wrapper/private-goal reads, job changes or sub-agents. '
    'Use the skill-authorized Astra/max/fork-none same-family/provisional route; this review does not replace actual real-model two-step probes.')
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_tool='collaboration.spawn_agent', actual_arguments=dict(task_name='pvg_boundary_distribution_source_review',
        fork_turns='none',model='gpt-6-astra',reasoning_effort='max',message=message),
    actual_tool_receipt={'task_name':'/root/pvg_boundary_distribution_source_review'},
    actual_task='/root/pvg_boundary_distribution_source_review', request_path=str(request),
    request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(), review_independence='same-family',
    acceptance_status='provisional',backend_sku_independently_verified=False)
with (trace/'001-code-review.request.json').open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(record,indent=2)+'\n')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+' actual frozen-G boundary source prepared under pvground_boundary_distribution_20261004; residual400614 vs distribution456102, unchanged1302 whole/local input, independent DFL/7 over native last matched boxes×6. Common size floor explicit; no native model/GPU/update yet. Actual fresh source-review task /root/pvg_boundary_distribution_source_review requested Astra/max/forknone, samefamily/provisional. OriginalG remains best; no new goal.\n')
print(json.dumps(dict(actual_task=record['actual_task'],native_forwards=0,remote_jobs=0,review_result_available=False)))
