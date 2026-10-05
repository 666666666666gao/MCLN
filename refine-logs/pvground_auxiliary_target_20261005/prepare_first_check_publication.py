"""Prepare publication of the actual first check and revised stage estimate."""
import ast
from pathlib import Path


root = Path(__file__).resolve().parent
source = (root / 'publish_retention_tools.py').read_text(encoding='utf-8')
start = source.index("section = f'''")
end = source.index("new = old + section.encode('utf-8')", start)
section = """progress = json.loads((root / 'SCHEDULED_PROGRESS_185943.json').read_bytes())
estimate = json.loads((root / 'FIRST_CHECK_ESTIMATE.json').read_bytes())
assert progress['controller_alive'] and progress['protected_best_sha256_exact']
assert progress['progress']['last_logged_step']==3723 and not estimate['formal_complete']
assert hashlib.sha256((root / 'SCHEDULED_PROGRESS_185943.json').read_bytes()).hexdigest()==estimate['basis_progress_sha256']
section = f'''

## 20.376.73 首查确认control更新完成，终点留出仍评估中；结束时间按实测修订（{stamp}）

唯一观察器于18:54:14实际首查，controller630343及control子进程630346存活，状态为control/train，尚无completed_run。随后{progress['time_cst']}做一次与该窗口对应的只读进度／资源核对，没有模型或优化器重放，也没有启动第二观察器。

这次实际日志已完整写出3723条更新记录，最后step3723、尾batch2；训练累计4453.7007秒，最后512步平均1.1847秒。control的terminal.pth已生成5585925字节，原4509头5585861字节的SHA仍与规格一致。这里仅确认更新和保存发生，不将未评估新权重替换4509。

已完成的initial模块留出为6887条、6172／5609 REC命中，耗时996.1125秒；它是预训练见过场景的模块留出，不是9508正式验证。检查时terminal模块留出仅写出4286／6887条，没有terminal完整回执，也没有本轮formal完整成绩。当前正式最好仍5614／4509，不使用局部评估数字更新成绩。

该次资源快照为A100显存6979／40960MiB、利用率12%，系统盘余量467582976字节、数据盘1774354432字节。只描述检查时刻，不由该快照推断全训练峰值或服务器之后状态。

新估计使用实际4453.7007秒训练、996.1125秒initial评估、约450.13秒重建／初始准备，以及上一轮完整9508实测评估成本并按本轮速度缩放。control formal预计{estimate['control_formal_end_estimate_cst']}附近完成，两组预计{estimate['pair_end_estimate_cst']}附近闭合；下次人工结果核对安排{estimate['next_manual_outcome_check_cst']}附近。均为阶段估计，不是已完成结果；相较原21:44估计，按这次实测调整到约22:03。

既有观察器38531继续240秒间隔，不修改活动实验源码、预算或原启动记录。待control完整formal回执后再报告9508数字；两组都闭合、CPU重算及终态审查完成后才依§20.376.72严格主指标晋级／清理规则处理权重。仍保留全部256候选、唯一last/bbs和同Query框／Mask，目标ACTIVE_UNMET。原始首查、进度和时间估计证据见refine-logs/pvground_auxiliary_target_20261005/。
'''
"""
source = source[:start] + section + source[end:]
changes = {
    "previous = json.loads((root / 'storage_tools_publication.json').read_bytes())":
        "previous = json.loads((root / 'retention_tools_publication.json').read_bytes())",
    "assert previous['section'] == '20.376.71'": "assert previous['section'] == '20.376.72'",
    "assert not (root / 'retention_tools_publication.json').exists()":
        "assert not (root / 'first_check_publication.json').exists()",
    "assert b'## 20.376.72 ' not in old": "assert b'## 20.376.73 ' not in old",
    "Closed-result/retention source review PASS24; strict primary improvement required for replacing4509; no current result or weight cleanup.":
        "Actual scheduled check confirmed control3723updates complete, terminal holdout4286rows; no formal result. Revised pair estimate22:03.",
    "Require strict primary gain before replacing retained geometry best":
        "Record actual first progress check and revise auxiliary fit estimate",
    "record.update(time_cst=stamp, section='20.376.72'":
        "record.update(time_cst=stamp, section='20.376.73'",
    "status='REVIEWED_STRICT_PROMOTION_AND_CLOSED_TOOLS_PUBLISHED'":
        "status='ACTUAL_SCHEDULED_CONTROL_PROGRESS_AND_REVISED_ESTIMATE_PUBLISHED'",
    "(root / 'retention_tools_publication.json').write_text":
        "(root / 'first_check_publication.json').write_text",
    "latest_publication=str(root / 'retention_tools_publication.json')":
        "latest_publication=str(root / 'first_check_publication.json')",
    "handoff_section='20.376.72'": "handoff_section='20.376.73'",
    "current_goal_turn_classification='PROGRESS_REVIEWED_STRICT_PROMOTION_CORRECTION'":
        "current_goal_turn_classification='PROGRESS_ACTUAL_SCHEDULED_CHECK_AND_TIME_ESTIMATE'",
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
marker = "(root / 'first_check_publication.json').write_text"
source = source.replace(marker, "record.update(actual_scheduled_progress=progress,revised_stage_estimate=estimate)\n" + marker)
memory_start = source.index("    stream.write('\\nPVGround ' + stamp + ': Closed-result")
memory_end = source.index('\nprint(json.dumps(', memory_start)
source = source[:memory_start] + "    stream.write('\\nPVGround ' + stamp + ': Actual18:54 observer/18:59 read-only check: control updates3723 complete and terminal saved, terminal holdout4286/6887, no formal9508 result. Best4509SHA exact; system467.6MB/data1.774GB. Revised control formal19:36:54, pair22:02:51, manual next19:34:54, same observer38531/240s. Doc73 Main ' + heads[0] + ', ACTIVE_UNMET.\\n')" + source[memory_end:]
assert not (root / 'publish_first_check.py').exists()
ast.parse(source)
(root / 'publish_first_check.py').write_text(source, encoding='utf-8')
print('Prepared actual first-check publication; not executed.')
