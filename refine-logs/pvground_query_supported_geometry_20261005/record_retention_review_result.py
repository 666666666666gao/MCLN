"""Record the actual received source-review result without executing cleanup."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
review = json.loads((local / 'RETENTION_REVIEW.json').read_bytes())
assert review['verdict'] == 'PASS' and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
assert hashlib.sha256((local / 'RETENTION_REVIEW.md').read_bytes()).hexdigest() == '7bd25748a1ece1ef3a7423c458f9e00622c85abbd61adc734db37aff27c1e489'
assert hashlib.sha256((local / 'RETENTION_REVIEW.json').read_bytes()).hexdigest() == 'a15e065cff354372b7363caf471d7713b6461bf37c3dddc4c1a6627e80479536'
stamp = datetime.datetime.now().astimezone().isoformat()
record = json.loads((local / 'RETENTION_REVIEW_CALL.json').read_bytes())
assert not record['result_received']
record.update(result_received=True, received_recorded_at_cst=stamp,
              verdict=review['verdict'], blocking_findings=[],
              reviewed_at_cst=review['reviewed_at'],
              review_sha256=hashlib.sha256((local / 'RETENTION_REVIEW.json').read_bytes()).hexdigest(),
              deletions_executed=0)
raw = (json.dumps(record, indent=2) + '\n').encode('utf-8')
(local / 'RETENTION_REVIEW_CALL.json').write_bytes(raw)
trace = Path('C:/Users/gb/.aris/traces/experiment-bridge/2026-10-05_query_geometry_retention_run01')
(trace / 'call.json').write_bytes(raw)
(trace / 'response.json').write_bytes((local / 'RETENTION_REVIEW.json').read_bytes())
state = json.loads((local / 'active_continuation_state.json').read_bytes())
state.update(active_reviewer=None, retention_review_pending=False,
             retention_source_review_complete=True, retention_executed=False,
             time_cst=stamp)
(local / 'active_continuation_state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'retention_source_review': 'PASS', 'actual_result_received': True,
                  'deletions_executed': 0, 'time_cst': stamp}))
