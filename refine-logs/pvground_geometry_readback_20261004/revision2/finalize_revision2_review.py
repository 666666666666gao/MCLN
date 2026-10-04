"""Preserve the actual rescue and revised-source review response."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
review = json.loads((local / 'READBACK_REVISION2_SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] == 'PASS' and review['execution_scope'] == 'SOURCE_ONLY'
assert not review['blocking_findings'] and len(review['reviewed_files']) == 34
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
response = '''已完成：

- [失败审查报告](<D:/Program Files/UserCache/gb/codex/tmp/pvground_geometry_readback_20261004/READBACK_PREFLIGHT_FAILURE_REVIEW.md>)：真实失败、0 次更新，差异原因未证实。
- [V2 源码审查](<D:/Program Files/UserCache/gb/codex/tmp/pvground_geometry_readback_20261004/revision2/READBACK_REVISION2_SOURCE_REVIEW.md>)：**PASS / SOURCE_ONLY**，无阻断项；对应 JSON 已保存。

34 文件身份检查和 30 个 Python 3.7 语法检查通过。未授予运行成功、准确率或正式训练批准；无远程操作。
'''
assert not (local / 'ACTUAL_REVISION2_REVIEW_RESPONSE.txt').exists()
trace = local / '.aris/traces/experiment-bridge/2026-10-04_readback_failure_revision2_run01'
trace.mkdir(parents=True)
(local / 'ACTUAL_REVISION2_REVIEW_RESPONSE.txt').write_text(response, encoding='utf-8')
(trace / '001-rescue-source.response.md').write_text(response, encoding='utf-8')
record = dict(status='ACTUALLY_COMPLETED_SOURCE_ONLY', actual_task='/root/pvg_readback_preflight_failure_review',
    requested_model='gpt-6-astra', requested_reasoning_effort='max', model_effort_independently_attested=False,
    fork_turns='none', review_independence='same-family', acceptance_status='provisional',
    verdict='PASS', blocking_findings=0, reviewed_files=34, AST37_files=30,
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    response_sha256=hashlib.sha256(response.encode()).hexdigest(), runtime_pass=False,
    formal_fit_approved=False, accuracy_pass=False,
    failure_report=str(local.parent / 'READBACK_PREFLIGHT_FAILURE_REVIEW.json'))
(local / 'REVISION2_SOURCE_REVIEW_CALL.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
(trace / 'run.meta.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
(trace / 'events.jsonl').write_text(json.dumps(dict(time_cst=record['time_cst'],
    event='actual_rescue_and_source_final_response_recorded', verdict='PASS_SOURCE_ONLY')) + '\n', encoding='utf-8')
(trace / '001-rescue-source.request.md').write_text(
    'Fresh failure review of original complete_preflight logs (CPU PASS, GPU cross-forward last_center failed,0 updates). '
    'Read unchanged failed attempt; review corrected revision2 actual factory/caller/helpers/port/controller/deploy/observer/collector '
    'and both templates. Require exact same-frame geometry/mask invariance, zero cached-native-head equivalence, '
    'disabled-repeat diagnosis without tolerance PASS, single actual final semantic subhead, native+G two-update preflight only. '
    'No remote actions or formal training approval. Exact paths are in the generated reviewed_files identity list.\n', encoding='utf-8')
print(json.dumps(dict(verdict='PASS_SOURCE_ONLY', files=34, runtime_pass=False)))
