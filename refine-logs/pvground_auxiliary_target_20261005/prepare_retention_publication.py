"""Prepare publication of the reviewed strict-primary promotion correction."""
import ast
from pathlib import Path


root = Path(__file__).resolve().parent
source = (root / 'publish_storage_tools.py').read_text(encoding='utf-8')
start = source.index("section = f'''")
end = source.index("new = old + section.encode('utf-8')", start)
section = """review = json.loads((root / 'RETENTION_REVIEW.json').read_bytes())
assert review['verdict']=='PASS' and not review['blocking_findings'] and review['execution_scope']=='SOURCE_ONLY'
assert all(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'] for item in review['reviewed_files'])
assert json.loads((root / 'RETENTION_REVIEW_CALL.json').read_bytes())['result_received']
section = f'''

## 20.376.72 终态工具源码审查通过，主指标持平时保留4509（{stamp}）

终态收集、CPU重算和限定路径清理工具完成SOURCE_ONLY审查，PASS、阻塞0，24份当前文件SHA核对；同一既有审查上下文的follow-up、same-family/provisional、backend unattested，不称fresh或外部独立。审查核对了4509父rows的实际9508重算、control／member_target目标模式、两组额外权重1及完整头恢复条件。本轮结果尚未闭合，这不是本轮精度或终态审查。

审查发现原重算继承的(hits50,hits25,tie-parent)排序与本轮计划“严格Acc@0.50提高才替换4509”不一致。已改为(hits50,tie-parent,hits25)，对应生成器同步；精确3路径清理脚本在任何unlink前也核验赢家是原父模型，或其严格命中大于4509。训练、匹配、损失和正式评价代码没有变化，未执行任何本轮权重删除；修正发生在本轮结果出现之前。

收集器、原CPU度量及授权入口保留上轮实现；新两生成器与产物一致。清理仍需实际两组闭合、9508回执、CPU重算及另一次真实终态审查，不以源码PASS直接开始删除。完整10状态替代头、原G／官方PV／V99保护和不归档负权重规则继续执行。当前最佳仍5614／4509，尚无新增REC结果；既有观察器38531按18:54:13首查和240秒后续间隔继续等待。源码、审查和修正绑定见refine-logs/pvground_auxiliary_target_20261005/。
'''
"""
source = source[:start] + section + source[end:]
changes = {
    "previous = json.loads((root / 'fit_launch_publication.json').read_bytes())":
        "previous = json.loads((root / 'storage_tools_publication.json').read_bytes())",
    "assert previous['section'] == '20.376.70'": "assert previous['section'] == '20.376.71'",
    "assert not (root / 'storage_tools_publication.json').exists()":
        "assert not (root / 'retention_tools_publication.json').exists()",
    "assert b'## 20.376.71 ' not in old": "assert b'## 20.376.72 ' not in old",
    "Published evidence moved to data disk with exact contents and logical paths; closed-result and retention tools prepared, not executed.":
        "Closed-result/retention source review PASS24; strict primary improvement required for replacing4509; no current result or weight cleanup.",
    "Preserve published evidence on data disk and prepare closed-result tools":
        "Require strict primary gain before replacing retained geometry best",
    "record.update(time_cst=stamp, section='20.376.71'":
        "record.update(time_cst=stamp, section='20.376.72'",
    "status='STORAGE_RELOCATION_AND_CLOSED_TOOLS_PREPARATION_PUBLISHED'":
        "status='REVIEWED_STRICT_PROMOTION_AND_CLOSED_TOOLS_PUBLISHED'",
    "(root / 'storage_tools_publication.json').write_text":
        "(root / 'retention_tools_publication.json').write_text",
    "latest_publication=str(root / 'storage_tools_publication.json')":
        "latest_publication=str(root / 'retention_tools_publication.json')",
    "handoff_section='20.376.71'": "handoff_section='20.376.72'",
    "current_goal_turn_classification='PROGRESS_STORAGE_RELOCATION_AND_CLOSED_TOOLS'":
        "current_goal_turn_classification='PROGRESS_REVIEWED_STRICT_PROMOTION_CORRECTION'",
    "('SOURCE_REVIEW.json', 'SOURCE_REVIEW.md', 'LAUNCH_REVIEW.json', 'LAUNCH_REVIEW.md')":
        "('SOURCE_REVIEW.json', 'SOURCE_REVIEW.md', 'LAUNCH_REVIEW.json', 'LAUNCH_REVIEW.md', 'RETENTION_REVIEW.json', 'RETENTION_REVIEW.md')",
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
memory_start = source.index("    stream.write('\\nPVGround ' + stamp + ': Published evidence")
memory_end = source.index('\nprint(json.dumps(', memory_start)
source = source[:memory_start] + "    stream.write('\\nPVGround ' + stamp + ': Closed-result and retention tool source review PASS24, same-context/same-family provisional. Fixed real plan mismatch in CPU rank to hits50,tie-parent,hits25 and strict-gain guard before anyunlink; no current result or cleanup. Doc72 Main ' + heads[0] + ', ACTIVE_UNMET, observer38531 first18:54.\\n')" + source[memory_end:]
assert not (root / 'publish_retention_tools.py').exists()
ast.parse(source)
(root / 'publish_retention_tools.py').write_text(source, encoding='utf-8')
print('Prepared strict-promotion publication; not executed.')
