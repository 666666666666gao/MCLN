"""Preserve the actual full-preflight source response, not a runtime verdict."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
review = json.loads((local / 'READBACK_FULL_SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] == 'PASS' and review['execution_scope'] == 'SOURCE_ONLY'
assert not review['blocking_findings'] and len(review['reviewed_files']) == 32
for item in review['reviewed_files']:
    raw = Path(item['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
response = '''PASS — SOURCE_ONLY, same-family/provisional. All 32 files reviewed directly; no blocking or nonblocking correctness defects found.

Saved [Markdown report](<D:/Program Files/UserCache/gb/codex/tmp/pvground_geometry_readback_20261004/READBACK_FULL_SOURCE_REVIEW.md>) and [JSON report](<D:/Program Files/UserCache/gb/codex/tmp/pvground_geometry_readback_20261004/READBACK_FULL_SOURCE_REVIEW.json>).

All identities and arithmetic checks passed. Optional AST execution could not start because the local Python launcher failed. Astra/max attribution remains unattested.

Approval covers only the CPU/two-update preflight source gate—not runtime, accuracy, or formal training.
'''
trace = local / '.aris/traces/experiment-bridge/2026-10-04_readback_full_preflight_run01'
assert not (local / 'ACTUAL_FULL_SOURCE_REVIEW_RESPONSE.txt').exists()
for path in (local / 'ACTUAL_FULL_SOURCE_REVIEW_RESPONSE.txt', trace / '001-full-source.response.md'):
    path.write_text(response, encoding='utf-8')
stamp = datetime.datetime.now().astimezone().isoformat()
call_path = local / 'FULL_SOURCE_REVIEW_CALL.json'
call = json.loads(call_path.read_bytes())
call.update(status='ACTUALLY_COMPLETED_SOURCE_ONLY', verdict='PASS', recorded_at=stamp,
    blocking_findings=0, runtime_pass=False, formal_training_approved=False, accuracy_pass=False)
call_path.write_text(json.dumps(call, indent=2) + '\n', encoding='utf-8')
meta_path = trace / 'run.meta.json'
meta = json.loads(meta_path.read_bytes())
meta.update(status='ACTUALLY_COMPLETED_SOURCE_ONLY', verdict='PASS', recorded_at=stamp)
meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
with (trace / 'events.jsonl').open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(time_cst=stamp, event='actual_final_response_recorded',
        actual_task=call['actual_task'], verdict='PASS_SOURCE_ONLY',
        response_sha256=hashlib.sha256(response.encode()).hexdigest())) + '\n')
state_path = local.parent / 'pvground_face_conditioned_20261004/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(active_readback_full_source_reviewer=None, readback_current_revision_review='PASS_FULL_PREFLIGHT_SOURCE_ONLY',
    readback_runtime_checked=False, readback_formal_training_started=False, native_publisher_session_closed=83528)
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(verdict='PASS_SOURCE_ONLY', files=32, runtime_pass=False, formal_training=False)), flush=True)
