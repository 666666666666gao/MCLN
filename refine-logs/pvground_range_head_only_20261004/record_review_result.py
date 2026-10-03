"""Record the actual fresh review completion and its verified read scope."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).parent
review=json.loads((root/'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict']=='PASS' and not review['blocking_findings']
assert review['execution_scope']=='SOURCE_ONLY'
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
trace=root/'.aris/traces/experiment-bridge/2026-10-04_head_only_run01'
target=trace/'002-source-review.response.json'
assert not target.exists()
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_native_task='/root/pvg_range_head_only_source_review',model='gpt-6-astra',reasoning_effort='max',
    fork_turns='none',review_independence='same-family',acceptance_status='provisional',backend_sku_independently_verified=False,
    verdict='PASS',blocking_findings=[],execution_scope='SOURCE_ONLY',verified_reviewed_files=len(review['reviewed_files']),
    actual_final_response='PASS：0 个阻断、0 个非阻断缺陷。已覆盖新增正式双分支启动门槛和权重保留逻辑，共核对 55 个文件；SOURCE_ONLY、same-family/provisional。真实预检仍待执行。\n\n- [Markdown 报告](C:/Users/gb/.codex/tmp/pvground_range_head_only_20261004/EXPERIMENT_CODE_REVIEW.md)\n- [JSON 报告](C:/Users/gb/.codex/tmp/pvground_range_head_only_20261004/EXPERIMENT_CODE_REVIEW.json)',
    reports={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('EXPERIMENT_CODE_REVIEW.json','EXPERIMENT_CODE_REVIEW.md')},
    model_forwards=0,optimizer_updates=0,weights_created=0)
target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+' stableG/head-only source review ACTUALPASS/0blocking/0nonblocking,55read-file SHAs checked; freshAstra/max samefamily/provisional. Two-step probe and full pair still0GPU/0updates; no false sanity/accuracy claim. Formal deployment gate/retention included, no need rereview unchangedcode.\n')
print(json.dumps({key:record[key] for key in ('time_cst','verdict','verified_reviewed_files','model_forwards','optimizer_updates')}))
