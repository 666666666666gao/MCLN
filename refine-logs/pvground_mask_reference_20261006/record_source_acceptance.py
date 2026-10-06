"""Close the actual native source-review call after receiving its final response."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
report=root/'SOURCE_REVIEW.json'
review=json.loads(report.read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN')
assert not review['blocking_findings'] and review['fresh_context']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
path=root/'SOURCE_REVIEW_CALL.json'
call=json.loads(path.read_bytes())
assert call['result_received'] is False
call.update(result_received=True,result_received_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),actual_verdict=review['verdict'],
    scope_note='Source-only acceptance; no GPU/two-step/initial or terminal accuracy claim. Step0 best requires its own actual M2 strict restore.')
path.write_text(json.dumps(call,indent=2)+'\n',encoding='utf-8')
print(json.dumps(call))
