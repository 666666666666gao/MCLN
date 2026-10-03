"""Seal the actual terminal reviewer response before updating living status."""
import datetime
import hashlib
import json
from pathlib import Path


local = Path(__file__).parent
trace = local / '.aris/traces/experiment-audit/2026-10-03_terminal_run01'
response = '''审查完成：**WARN，`blocking_issues=[]`**。

- 65 份文件已核验；69,846 条评估记录、11,169 条训练记录与回执及汇总一致。
- normalized 正式 bbs 为 **5575／4426**，较 expanded-N **−21／−31**，未达目标。
- 警告包括过时状态、实际对应集合随训练变化，以及原始 Mask／全候选／优化器未独立重放。
- 单文件退役脚本 **PASS_STATIC_ONLY**；未执行删除。
- 已记录原生 Astra/max/fresh 请求，保持 **same-family / provisional**。

报告：[MD](C:/Users/gb/.codex/tmp/pvground_candidate_normalization_20261003/analysis/EXPERIMENT_AUDIT.md)、[JSON](C:/Users/gb/.codex/tmp/pvground_candidate_normalization_20261003/analysis/EXPERIMENT_AUDIT.json)、[CPU 重算](C:/Users/gb/.codex/tmp/pvground_candidate_normalization_20261003/analysis/AUDIT_CPU_RECOUNT.json)。'''
supplement = '本次原生 spawn 请求元数据：task_name=pvg_normalization_terminal_integrity，model=gpt-6-astra，reasoning_effort=max，fork_turns=none。实际请求已封存于 C:\\Users\\gb\\.codex\\tmp\\pvground_candidate_normalization_20261003\\.aris\\traces\\experiment-audit\\2026-10-03_terminal_run01\\001-terminal-integrity.request.json。请保持 fresh context / same-family / provisional 与 backend SKU 未独立证明。请完成最终 MD/JSON 报告，包含 verdict 与 blocking_issues；根代理将封存报告后更新过时状态并按用户授权执行单文件退役。'
audit = json.loads((local / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_issues']
now = datetime.datetime.now().astimezone().isoformat()
with (trace / '001-terminal-integrity.response.md').open('x', encoding='utf-8') as stream:
    stream.write(response)
with (trace / '002-attribution.message.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(tool='collaboration.send_message', target='/root/pvg_normalization_terminal_integrity', message=supplement), ensure_ascii=False, indent=2) + '\n')
reports = {}
for name in ('EXPERIMENT_AUDIT.md', 'EXPERIMENT_AUDIT.json', 'AUDIT_CPU_RECOUNT.json'):
    raw = (local / 'analysis' / name).read_bytes()
    with (trace / name).open('xb') as stream:
        stream.write(raw)
    reports[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
request = json.loads((trace / '001-terminal-integrity.request.json').read_bytes())
meta = dict(call_number=1, purpose='terminal-integrity', timestamp=now,
    agent_id=request['actual_task'], requested_model='gpt-6-astra', reasoning_effort='max',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional',
    backend_SKU_independently_verified=False, status='complete', verdict=audit['verdict'], reports=reports)
with (trace / '001-terminal-integrity.meta.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
with (trace / 'run.meta.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(meta, skill='experiment-audit', started_at=request['time_cst'], finished_at=now), ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(audit_verdict=audit['verdict'], blocking_issues=audit['blocking_issues'], actual_response_and_reports_sealed=True)))
