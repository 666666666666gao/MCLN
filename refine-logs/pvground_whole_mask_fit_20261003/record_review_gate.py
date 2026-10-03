"""Seal the actual source review before the authorized formal launch."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).parent
review = json.loads((root/'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
trace = Path(r'C:\Users\gb\.codex_mcln_g0_20260905\.aris\traces\experiment-bridge\2026-10-03_formal_whole_range_run01')
trace.mkdir(parents=True, exist_ok=True)
for name in ('REVIEW_REQUEST.md', 'EXPERIMENT_CODE_REVIEW.md', 'EXPERIMENT_CODE_REVIEW.json'):
    assert not (trace/name).exists()
    shutil.copyfile(root/name, trace/name)
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_native_task='/root/pvg_whole_range_formal_review', configured_model='gpt-6-astra',
    configured_reasoning_effort='max', fork_turns='none', requested_by_skill='experiment-bridge phase2.5',
    review_independence='same-family', acceptance_status='provisional', backend_sku_independently_verified=False,
    execution_scope='SOURCE_ONLY', verdict=review['verdict'], blocking_findings=review['blocking_findings'],
    actual_preflight_root=str(root.parent/'pvground_whole_mask_integration_20261003/complete'),
    formal_training_started=False, model_or_GPU_execution_by_review_gate=0)
(trace/'native_dispatch_and_gate.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
(root/'review_gate.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record))
