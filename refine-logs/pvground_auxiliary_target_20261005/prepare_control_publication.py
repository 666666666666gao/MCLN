"""Prepare publication of the closed native-target control, not a strategy claim."""
import ast
from pathlib import Path


root=Path(__file__).resolve().parent
source=(root/'publish_first_check.py').read_text(encoding='utf-8')
start=source.index("section = f'''")
end=source.index("new = old + section.encode('utf-8')",start)
section="""intake=json.loads((root/'CONTROL_CLOSED_INTAKE.json').read_bytes())
recount=json.loads((root/'CONTROL_CPU_RECOUNT.json').read_bytes())
control_estimate=json.loads((root/'CONTROL_COMPLETE_ESTIMATE.json').read_bytes())
assert intake['formal_rows']==recount['formal_rows']==9508
assert recount['control']['cpu_box_threshold_changes']==0
assert [recount['control'][k] for k in ('rec_hits25','rec_hits50')]==[5616,4511]
assert not recount['full_pair_complete'] and not recount['best_weight_promoted']
section=f'''

## 20.376.74 native_gt控制组正式结束，CPU完整重算5616／4511；member_gt仍待结果（{stamp}）

control正式9508条评估实际于{intake['formal_finished_cst']}结束，train／formal退出码均0；19:34:27既有观察器已见control/train、control/formal均闭合，member_target/train子进程636595启动。收集8份已结束控制组文件，其中完整formal rows为7623290字节、SHA256=ec73b6acb9691d6c591136c5e5a34e80a8d7d716bedef714ca960bbbd324716c；不下载权重或重放模型。

该组从4509头出发再训练29778条各一次／3723更新，仅训练456102参数既有几何头；父模型及零R状态保持、完整模型与优化器恢复回执通过。初始6887模块留出为6172／5609，终点6177／5613；两者为预训练见过场景的模块留出，不与9508正式数字混用。

| 同一检查点原生last/bbs，9508条 | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| 保留的4509父模型 | 59.0450%〔5614〕 | 47.4232%〔4509〕 |
| 本轮native_gt继续训练控制 | 59.0660%〔5616〕 | 47.4443%〔4511〕 |

完整9508行经原CPU函数核对Box／GT阈值，GPU与CPU命中翻转0。对4509父模型，@0.25修复4／破坏2、净+2；@0.50修复18／破坏16、净+2；最终选中Query改变0。当前控制组同Query内部粗框4495到最终框4511，修复42／破坏26、净+16；这是本模型内部精修作用，不能将+16当成相对4509或独立baseline增益。

该检查点Mask完整三项为5812／9508、5133／9508及47.107628% mIoU，原始回执与逐行标量均保留；CPU独立重算的是框阈值，Mask和候选覆盖仍按已有标量核对，不称原始点级Mask重新计算。本轮暂只有native_gt控制结果，member_gt策略尚无终态，不能用这+2声称扰动前成员目标更有效或稳定显著涨点。

当前4509保护权重继续保留，没有晋级或删除；待策略组完整9508、两组CPU核对及fresh实际终态审查，再依严格Acc@0.50改进规则确定保留头。现有8文件恢复／退出见证通过，仍不等于整组terminal integrity审查完成。新正式模型候选4511距50%线4754尚差243，距V99同一行4797差286；这只是候选指标差，不重新定义总目标。

依据实际控制组从启动到formal结束8385.1874秒，策略组完整结束新估计{control_estimate['member_formal_end_estimate_cst']}，下一次人工结果核对{control_estimate['next_manual_outcome_check_cst']}附近；均为估计，GT资格数量可能改变吞吐。既有观察器38531持续240秒周期，无新观察器、模型、教师或推理排名；活动配置保持。完整行、CPU核对及估计证据在refine-logs/pvground_auxiliary_target_20261005/，目标ACTIVE_UNMET。
'''
"""
source=source[:start]+section+source[end:]
changes={
    "file.suffix in ('.py', '.json', '.md', '.log', '.exit')":
        "file.suffix in ('.py', '.json', '.md', '.log', '.exit', '.jsonl')",
    "previous = json.loads((root / 'retention_tools_publication.json').read_bytes())":
        "previous = json.loads((root / 'first_check_publication.json').read_bytes())",
    "assert previous['section'] == '20.376.72'": "assert previous['section'] == '20.376.73'",
    "assert not (root / 'first_check_publication.json').exists()":
        "assert not (root / 'control_closed_publication.json').exists()",
    "assert b'## 20.376.73 ' not in old": "assert b'## 20.376.74 ' not in old",
    "Actual scheduled check confirmed control3723updates complete, terminal holdout4286rows; no formal result. Revised pair estimate22:03.":
        "Native-target control full9508 closed and CPU-recounted5616/4511, parent delta+2/+2; member results pending, no promotion/cleanup.",
    "Record actual first progress check and revise auxiliary fit estimate":
        "Record closed native-target control and independent full-row recount",
    "record.update(time_cst=stamp, section='20.376.73'":
        "record.update(time_cst=stamp, section='20.376.74'",
    "status='ACTUAL_SCHEDULED_CONTROL_PROGRESS_AND_REVISED_ESTIMATE_PUBLISHED'":
        "status='CLOSED_NATIVE_CONTROL_FULL_ROWS_PUBLISHED_MEMBER_PENDING'",
    "(root / 'first_check_publication.json').write_text":
        "(root / 'control_closed_publication.json').write_text",
    "latest_publication=str(root / 'first_check_publication.json')":
        "latest_publication=str(root / 'control_closed_publication.json')",
    "handoff_section='20.376.73'": "handoff_section='20.376.74'",
    "current_goal_turn_classification='PROGRESS_ACTUAL_SCHEDULED_CHECK_AND_TIME_ESTIMATE'":
        "current_goal_turn_classification='PROGRESS_CLOSED_CONTROL_FULL_ROWS_COLLECTED_AND_RECOUNTED'",
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
marker="(root / 'control_closed_publication.json').write_text"
source=source.replace(marker,"record.update(control_formal_hits=[5616,4511],control_formal_finished_cst=intake['formal_finished_cst'],control_CPU_recount_pass=True,member_result_unobserved=True,full_pair_complete=False,revised_member_estimate=control_estimate)\n"+marker)
memory_start=source.index("    stream.write('\\nPVGround ' + stamp + ': Actual18:54")
memory_end=source.index('\nprint(json.dumps(',memory_start)
source=source[:memory_start]+"    stream.write('\\nPVGround ' + stamp + ': Native_gt control formal actually closed19:30:38, full9508 CPUrecount5616/4511 vs4509parent+2/+2, strict18repair16damage, selectedQchanges0. 8closed files including7.623MBrows collected, restore/exitpass; no parent promotion/delete, memberstrategy pending. Memberfinish estimate21:50:24, nextmanual21:48:24; same observer38531. Doc74 Main ' + heads[0] + ', ACTIVE_UNMET.\\n')"+source[memory_end:]
assert not (root/'publish_closed_control.py').exists()
ast.parse(source)
(root/'publish_closed_control.py').write_text(source,encoding='utf-8')
print('Prepared closed-control publication; not executed.')
