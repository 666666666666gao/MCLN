"""Record the actual completed source-only reviewer response; no runtime claims."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
trace = local / '.aris/traces/experiment-bridge/2026-10-04_readback_source_run01'
review = json.loads((local / 'READBACK_SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] == 'PASS' and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
assert not review['scope_limits']['runtime_pass'] and not review['scope_limits']['accuracy_pass']
bindings = json.loads((local / 'SOURCE_REVIEW_REQUEST_BINDINGS.json').read_bytes())
assert len(bindings) == 18
for item in bindings:
    raw = Path(item['path']).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
response = '''**PASS — SOURCE_ONLY, same-family/provisional.** No source correctness defects or required patches found. All 18 hashes matched; 11 Python files parsed.

Reports: [Markdown](<D:/Program Files/UserCache/gb/codex/tmp/pvground_geometry_readback_20261004/READBACK_SOURCE_REVIEW.md>) · [JSON](<D:/Program Files/UserCache/gb/codex/tmp/pvground_geometry_readback_20261004/READBACK_SOURCE_REVIEW.json>).

`bbs` uses correct algebra but different reduction ordering; real-batch equality remains pending. Future factory/runner, runtime and accuracy are not approved. Requested Astra/max backend is not independently attested.
'''
assert not (local / 'ACTUAL_SOURCE_REVIEW_RESPONSE.txt').exists()
for target in (local / 'ACTUAL_SOURCE_REVIEW_RESPONSE.txt', trace / '001-readback-source-review.response.md'):
    target.write_text(response, encoding='utf-8')
now = datetime.datetime.now().astimezone().isoformat()
call_path = local / 'SOURCE_REVIEW_CALL.json'
call = json.loads(call_path.read_bytes())
call.update(status='ACTUALLY_COMPLETED', reviewer_time_cst=review['time_cst'],
    recorded_at=now, verdict='PASS', blocking_findings=0, runtime_pass=False, accuracy_pass=False)
call_path.write_text(json.dumps(call, indent=2) + '\n', encoding='utf-8')
meta_path = trace / 'run.meta.json'
meta = json.loads(meta_path.read_bytes())
meta.update(status='ACTUALLY_COMPLETED_SOURCE_ONLY', completed_at=review['time_cst'],
    recorded_at=now, verdict='PASS', blocking_findings=0, runtime_pass=False, accuracy_pass=False)
meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
with (trace / 'events.jsonl').open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(time_cst=now, event='actual_reviewer_response_recorded',
        tool='collaboration.spawn_agent', agent=call['agent'], verdict='PASS',
        response_sha256=hashlib.sha256(response.encode()).hexdigest(),
        execution_scope='SOURCE_ONLY', runtime_pass=False, accuracy_pass=False)) + '\n')
state_path = local.parent / 'pvground_face_conditioned_20261004/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(active_readback_source_reviewer=None, readback_source_reviewer_closed=True,
    readback_partial_source_review='PASS_SOURCE_ONLY', readback_runtime_checked=False)
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status='ACTUAL_REVIEW_RECORDED', verdict='PASS_SOURCE_ONLY',
    checked_bindings=len(bindings), required_source_patches=len(review['required_source_patches']),
    runtime_pass=False, accuracy_pass=False, time_cst=now)), flush=True)
