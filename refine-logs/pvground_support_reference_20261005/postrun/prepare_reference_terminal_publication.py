"""Reuse the previous closed-result publisher; do not execute its entry point."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
template=root.parent/'pvground_auxiliary_target_20261005/publish_terminal.py'
destination=root/'postrun/publish_reference_terminal.py'
assert not destination.exists()
source=template.read_text(encoding='utf-8')


def change(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)


source=change(source,'local = Path(__file__).resolve().parent','local = Path(__file__).resolve().parents[1]')
source=change(source,"summary['status'] == 'ACTUAL_CLOSED_ROWS_ANALYZED' and summary['training_order_exact']",
    "summary['status'] == 'ACTUAL_CLOSED_REFERENCE_ROWS_ANALYZED' and summary['fit_order_exact']")
source=change(source,"assert previous['section'] == '20.376.74'","assert previous['section'] == '20.376.78'")
source=source.replace('20.376.75','20.376.79')
source=source.replace('/root/autodl-tmp/pvground_auxiliary_target_20261005','/root/autodl-tmp/pvground_support_reference_20261005')
begin=source.index("effect = summary['strategy_vs_control']")
end=source.index("new = old + section.encode('utf-8')",begin)
source=source[:begin]+'''verdict=audit['verdict']
finished=wait['terminal']['status']['finished_cst']
effect=summary['support_vs_control']['0.5']
table='\\n'.join(f"| {row['system']} | {100*row['rec_hits25']/9508:.4f}%（{row['rec_hits25']}） | {100*row['rec_hits50']/9508:.4f}%（{row['rec_hits50']}） |" for row in summary['table'])
details=[]
for arm,values in summary['systems'].items():
    ref=values['internal_reference']['0.5'];final=values['internal_final']['0.5'];delta=values['parent_delta']['0.5']
    details.append(f"- {arm}：相对4511起点严格修复{delta['repairs']}、破坏{delta['damages']}，净{delta['net']:+d}；同一已选Query粗框→参考框{ref['before_hits']}→{ref['after_hits']}，修复{ref['repairs']}、破坏{ref['damages']}；参考框→最终框{final['before_hits']}→{final['after_hits']}，修复{final['repairs']}、破坏{final['damages']}。内部阶段变化不算相对独立baseline的增量。额外候选学习职责累计{values['extra_roles']}次，额外DFL目标超节点范围累计{values['extra_outside_faces']}面；额外定位loss前／后100步均值{values['extra_loss_first100']:.8f}／{values['extra_loss_last100']:.8f}，参考定位loss对应{values['reference_loss_first100']:.8f}／{values['reference_loss_last100']:.8f}。已选Mask合格但框差{values['selected_mask_good_box_bad']}条；严格Full256几何上界{values['full256_strict_scalar_oracle']}，这些Mask与上界仅为已存标量核算。")
prefix='refine-logs/pvground_support_reference_20261005/'
intake=json.loads((local/'complete/INTAKE.json').read_bytes())
section=f"""

## 20.376.79 固定参考与自身／融合支撑参考正式终态、最佳保留（{stamp}）

承接§77—78，本轮实际于{finished}闭合，四个train／formal子任务与原控制器均退出0。原本地观察者40310在后续会话中句柄缺失且本地进程不存在，没有生成终态回执；{wait['time_cst']}的一次只读服务器核查确认原控制器已闭合、退出码与结果文件，并生成实际闭合回执。未重新启动训练，原观察者的退出码仍未知。

| 完整9508条ScanRefer开发验证，原生last/bbs | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
{table}

两组从同一4511几何头与官方PV／原G重建，fresh optimizer，每组29778条fit各一次、3723更新、B8／有效8、累积1、seed2027、LR1e-5、WD5e-4、clip0.1；父模型、Mask、语义与全零R冻结。旧10项几何状态累计11169→14892更新；支撑参考新增3078参数只学习本轮3723更新。控制456102参数／10状态，支撑参考459180参数／12状态。本轮同时改变支撑参考、参数量及参考L1／GIoU监督，不是纯来源或单loss消融。保留全部256候选、唯一原生bbs及同Query框／Mask；没有教师、双源排名或推理GT资格。

支撑参考相对同预算控制严格修复{effect['repairs']}、破坏{effect['damages']}，净{effect['net']:+d}；已选Query变化{summary['support_vs_control']['selected_query_changes']}。额外定位资格仍由候选自身Query与融合Mask的训练GT支撑确认，排除所有原匹配Query，按表达内及实际batch平均，几何不足判断与目标采用原native GT。

{chr(10).join(details)}

两组训练顺序及29778唯一输入核对一致；每组初始／终点6887是预训练见过场景的模块留出，与9508正式开发验证分开。闭合收集{len(intake['files'])}份原始文本／源码、{sum(item['bytes'] for item in intake['files'].values())}字节，没有下载权重或重跑模型。CPU独立重算已选粗框／参考框／最终框对真实GT的严格阈值；Mask和Full256仅核对存储标量，未重放原始点级Mask或全部候选框。fresh experiment-audit结论{verdict}，无阻断项，实际调用和报告见{prefix}analysis/；同系列、暂定接受，后端身份未证实。单seed结果不证明统计稳定、三模块有效或Nr／Sr泛化。

按预定Acc@0.50规则保留{best['system']}：{best['hits'][0]}／{best['hits'][1]}，路径{best['path']}，SHA256 {best['sha256']}；与父4511严格指标持平时保留父权重。实际删除两份闭合非最佳几何头，释放{retention['released_bytes']}字节，不归档负结果权重；保留完整10或12项几何状态及优化器，重建需要官方PV、原G及相应源码／配置。实际CPU严格头部重载已通过，非全模型推理重放。原PV/G和V99历史链保留。清理后数据盘余量{resources['data_free_bytes']}字节、系统盘{resources['system_free_bytes']}字节，GPU计算进程为空；远端发布complete复用数据盘闭合原文件。

当前保留模型距离4754严格命中尚差{max(0,4754-best['hits'][1])}条，距离V99的4797尚差{max(0,4797-best['hits'][1])}条。开发双阈值通过标志为{summary['scanrefer_target_pass']}，完整三数据集目标仍ACTIVE_UNMET，没有新Nr3D／Sr3D结果。后续依据实际阶段修复／破坏及超范围目标决定是否开展渐进精修；不把本轮结果解释成仅需扩大参考头、普通attention或再次质量回读。
"""
assert chr(65533) not in section
''' + source[end:]
begin=source.index("names = ['fit_wait.json'")
end=source.index("for directory in ('complete', 'analysis'):",begin)
source=source[:begin]+'''names=['fit_wait.json','weight_retention.json','CLOSED_RESOURCES.json',
    'inspect_existing_fit_authorized.py','COLLECTION_PREFETCH_CHANGE.json','COLLECTION_STREAM_CHANGE.json','COLLECTION_ROW_FILE_SIZES.json',
    'postrun/collect_formal_prefetch_authorized.py','postrun/prepare_prefetch_collector.py',
    'postrun/collect_formal_stream_authorized.py',
    'postrun/check_closed_resources.py','postrun/prepare_reference_terminal_publication.py',
    'postrun/publish_reference_terminal.py','TERMINAL_PUBLICATION_PREPARATION.json']
''' + source[end:]
source=source.replace('Actual closed native/member auxiliary target comparison published; metric-best ','Actual closed support-reference comparison published; metric-best ')
source=source.replace('Record closed native/member auxiliary target comparison and best retention','Record closed support-reference comparison and best retention')
begin=source.index("state.update(time_cst=record['time_cst']")
end=source.index('state_path.write_text',begin)
source=source[:begin]+'''state.update(time_cst=record['time_cst'],status='SUPPORT_REFERENCE_TERMINAL_PUBLISHED',
    latest_publication=str(local/'terminal_publication.json'),owned_gpu_job_active=False,
    active_reviewer=None,protected_best_hits=best['hits'],strict_target_gap=max(0,4754-best['hits'][1]),
    published_heads=heads,handoff_section='20.376.79',handoff_sha256=digest,
    support_reference_fit_observer_closed=True,support_reference_fit_observer_session_id=None,
    support_reference_full_pair_complete=True,support_reference_actual_fit_finished_cst=finished,
    support_reference_terminal_integrity_review_pending=False,support_reference_terminal_integrity_verdict=verdict,
    support_reference_postrun_tools_executed=True,
    support_reference_formal_hits_by_arm={row['system']:[row['rec_hits25'],row['rec_hits50']] for row in summary['table']},
    current_goal_turn_classification='PROGRESS_ACTUAL_TERMINAL_AUDIT_RETENTION_PUBLISHED')
''' + source[end:]
begin=source.index("with (workspace / 'memory/2026-10-05.md')")
end=source.index('print(json.dumps(record)',begin)
source=source[:begin]+'''with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\\nPVGround '+record['time_cst']+': Support-reference actual closed pair '+str(summary['table'])+'. Fresh terminal audit '+verdict+' same-family/provisional; retained '+str(best['hits'])+', two closed nonbest heads removed without archive. Doc79 fourlocal+remote exact, main '+heads[0]+'. Original observer40310 missing; actual one-shot remote closure verified, no training restart. Geometry old states total14892, new reference3078params thissegment3723updates. GoalACTIVE_UNMET; no new Nr/Sr results.\\n')

''' + source[end:]
ast.parse(source)
destination.write_text(source,encoding='utf-8')
resource_source=root.parent/'pvground_auxiliary_target_20261005/check_closed_resources.py'
resource=resource_source.read_text(encoding='utf-8')
resource=change(resource,'local = Path(__file__).resolve().parent','local = Path(__file__).resolve().parents[1]')
resource_destination=root/'postrun/check_closed_resources.py'
assert not resource_destination.exists()
ast.parse(resource)
resource_destination.write_text(resource,encoding='utf-8')
record=dict(status='REFERENCE_TERMINAL_PUBLICATION_TOOLS_PREPARED_NOT_EXECUTED',
    source_template=str(template),source_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
    publisher=str(destination),publisher_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
    resource_checker=str(resource_destination),resource_checker_sha256=hashlib.sha256(resource_destination.read_bytes()).hexdigest(),
    ast_parse_passed=True,training_modified=False,published=False,weight_deletion_executed=False,
    requires=['complete rows CPU analysis','actual fresh terminal audit','reviewed best retention','actual closed resources'])
(root/'TERMINAL_PUBLICATION_PREPARATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
