"""Preserve the actual fresh native review request; no experiment execution."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).parent
target=root/'.aris/traces/experiment-bridge/2026-10-04_head_only_run01'
target.mkdir(parents=True,exist_ok=True)
path=target/'001-source-review.request.json'
assert not path.exists()
message='Perform the fresh source-only experiment-bridge phase2.5 review requested in C:/Users/gb/.codex/tmp/pvground_range_head_only_20261004/REVIEW_REQUEST.md. Read that request and all specified primary artifacts directly. New protocol freezes all original PV-Ground+G state and trains only existing400614-parameter range head, first one realB8 two-step probe; no experiment run is authorized for reviewer. Check actual source rather than assuming executor is correct. Write only EXPERIMENT_CODE_REVIEW.md and EXPERIMENT_CODE_REVIEW.json in the requested directory, with actual verdict, blocking_findings, nonblocking findings, SOURCE_ONLY scope, same-family/provisional and configured model/effort attribution. No SSH/GPU/model imports/weights/deletion/job changes/source edits/subagents. Report concrete bugs and minimal fixes without hypothetical machinery. Return concise actual result and report paths.'
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_native_task='/root/pvg_range_head_only_source_review',
    model='gpt-6-astra',reasoning_effort='max',fork_turns='none',message=message,
    requested_by_skill='experiment-bridge phase2.5',review_independence='same-family',
    acceptance_status='provisional',backend_sku_independently_verified=False,
    request_path=str(root/'REVIEW_REQUEST.md'),request_sha256=hashlib.sha256((root/'REVIEW_REQUEST.md').read_bytes()).hexdigest(),
    request_contents=(root/'REVIEW_REQUEST.md').read_text(encoding='utf-8'),
    formal_training_started=False,model_forwards=0,optimizer_updates=0)
path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+' nextstableG/head-only protocol localprepared; freshnative /root/pvg_range_head_only_source_review Astra/max/forknone actualrunning per experiment-bridge. H has0forwards/updates/weights, old Fcomplete4495best unchanged; no newstructure/loss/budgetchange. Requesttrace saved, samefamily/provisional.\n')
print(json.dumps({k:record[k] for k in ('time_cst','actual_native_task','model','reasoning_effort','model_forwards','optimizer_updates')}))
