"""Preserve the actual completed source report and literal reviewer response."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
trace = local / '.aris/traces/experiment-bridge/2026-10-04_face_source_run01'
review = json.loads((local / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    raw = Path(item['path']).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_task='/root/pvg_face_conditioned_source_review', verdict=review['verdict'],
    blocking_findings=review['blocking_findings'], execution_scope='SOURCE_ONLY',
    response=(local / 'ACTUAL_CODE_REVIEWER_RESPONSE.txt').read_text(encoding='utf-8'),
    review_independence='same-family', acceptance_status='provisional', backend_sku_independently_verified=False,
    reports={name:dict(bytes=(local/name).stat().st_size, sha256=hashlib.sha256((local/name).read_bytes()).hexdigest())
        for name in ('EXPERIMENT_CODE_REVIEW.md', 'EXPERIMENT_CODE_REVIEW.json')})
with (trace / '004-code-review.response.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(record, indent=2)+'\n')
print(json.dumps(dict(actual_review=review['verdict'], blocking=0, reviewed_files=len(review['reviewed_files']), source_only=True)))
