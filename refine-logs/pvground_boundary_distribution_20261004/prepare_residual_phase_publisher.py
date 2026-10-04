"""Prepare publication of completed residual receipts, keeping distribution pending."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
basis = (local.parent/'pvground_range_head_only_20261004/publish_local_phase.py').read_text(encoding='utf-8')
output = local/'publish_residual_phase.py'
assert not output.exists()
start = "section = f'''"
sync_start = "assert all(not (repo/relative).exists()"
record_start = 'record = dict(time_cst='
assert all(basis.count(marker) == 1 for marker in (start,sync_start,record_start))
header = basis[:basis.index(start)]
header = header.replace('local_phase_publication.json','residual_phase_publication.json')
header = header.replace('launch_publication.json','volume_analysis_publication.json')
header = header.replace('local_phase_complete','residual_phase_complete')
header = header.replace('whole_range/train','distribution/train')
header = header.replace('completed local-arm','completed residual-arm').replace('whole arm remains','distribution arm remains')
sync = basis[basis.index(sync_start):basis.index(record_start)]
sync = sync.replace('pvground_range_head_only_20261004','pvground_boundary_distribution_20261004')
sync = sync.replace('head-only local complete9508','frozen-G residual complete9508')
sync = sync.replace('whole still active','distribution still active')
sync = sync.replace('Record completed local range head control while whole arm runs',
    'Record completed residual boundary control while distribution arm runs')
sync = sync.replace("'push','origin','HEAD:main'", "'-c','http.version=HTTP/1.1','push','origin','HEAD:main'")
section = """section = f'''

## 20.376.44 固定原G的边界对照：普通残差完成，六面分布仍活动（{stamp}）

这是本轮第一组完成后的及时记录，不是普通残差／六面分布配对终态。residual训练含初始、终态6887留出评估于{intake['residual_train_finished_cst']}结束；9508条正式开发验证于{intake['residual_formal_finished_cst']}结束。实际恢复10项head delta及原G保护父模型后，原生last/bbs命中{bbs['rec_hits25']}／{bbs['rec_hits50']}，即{100*bbs['rec_hits25']/9508:.4f}%／{100*bbs['rec_hits50']/9508:.4f}%；独立bbf模式为{bbf['rec_hits25']}／{bbf['rec_hits50']}，不拼接两模式单项最好值。

相对原G5615／4495，bbs变化{bbs['rec_hits25']-5615:+d}／{bbs['rec_hits50']-4495:+d}，没有超过原G。bbs Mask命中{bbs['mask_hits25']}／{bbs['mask_hits50']}，mIoU{bbs['mask_miou']:.8f}%。初始6887 bbs为{fit['initial']['bbs']['rec_hits25']}／{fit['initial']['bbs']['rec_hits50']}，终态为{fit['terminal']['bbs']['rec_hits25']}／{fit['terminal']['bbs']['rec_hits50']}。6887来自作者预训练见过场景的模块留出；9508为正式开发验证，不混算两者。

本组实际29778条fit各一次、3723更新、有效batch8（尾batch2）、seed2027、freshAdamW、LR1e-5、WD5e-4、clip0.1；只训练10项400614参数头，原G全部参数及持久buffer保持，原模块eval。输入为同一1302维完整范围／局部支撑；完整109维统计可见。普通中心／尺寸残差使用共同1e-6参考、最终尺寸下限，不把零头的一致性说成对原始负尺寸的逐位保持。原生最终BBox／GIoU监督部署框；语义和Mask路径冻结，最终bbs评分不变，全部256候选保留。没有六面分布监督、教师、P2、新对比目标或质量回写。

实际收取10份已完成配置、加载、初始／终态／正式、退出及清理元数据；正式rows SHA256为{intake['formal_rows_sha256']}。收取0次模型前向、0次优化更新、0份权重下载。controller的CPU所选框两阈值重算与正式bbs／bbf计数均0差异；此时尚未收取全量rows，也未重新计算原始Mask或Full256框，新上下文全配对核对仍待两组闭合。原G历史计数不冒充本轮独立重新评估。

依据用户既有清理授权，自有非最佳residual终点已核对恢复、完整9508评估、CPU计数及SHA后删除{deleted_bytes}字节；不创建失败权重的本地归档。原G父模型SHA仍为{retention['parent_sha256_after']}，此快照的指标最佳为{intake['retained_best']['name']}。收取时系统盘可用{intake['system_disk_free_bytes']}字节、数据盘{intake['data_disk_free_bytes']}字节、实验目录权重{len(intake['owned_weights'])}份；这些是收取快照。

controller继续distribution/train。保持同一完整范围／局部输入、原G固定协议及样本／更新预算，仅在另一组使用456102参数六面分布头和独立DFL/7。表示、监督及输出参数量共同改变，不宣称纯DFL效应或已经实现方向条件化face token解码。该组尚无正式9508结果；完整配对、同Query修复／破坏、起点跨进程差异、分布及目标体积分组均在实际闭合后核对。终态收集、真实审阅回执及发布工具已准备，未把准备称为执行或精度证据。

证据：refine-logs/pvground_boundary_distribution_20261004/residual_phase_complete及本次收取／发布源码。当前同检查点5615／4754和Nr3D／Sr3D目标仍未完成，原G及必要V99权重链继续保护。
'''
new = old + section.encode('utf-8')
prefix = 'refine-logs/pvground_boundary_distribution_20261004/'
payloads = {}
for path in sorted((local/'residual_phase_complete').rglob('*')):
    if path.is_file():
        payloads[prefix+path.relative_to(local).as_posix()] = path.read_bytes()
names = ('collect_residual_phase_authorized.py','residual_phase_collector_preparation.json',
    'prepare_residual_phase_publisher.py','publish_residual_phase.py',
    'residual_phase_publisher_preparation.json','observation_04.json','observation_05.json',
    'record_terminal_audit_request.py','record_terminal_audit_result.py',
    'terminal_audit_recorders_preparation.json','prepare_terminal_publisher.py',
    'publish_terminal.py','terminal_publisher_preparation.json','terminal_publisher_trace_update.json',
    'terminal_publisher_phase_update.json')
for name in names:
    payloads[prefix+name] = (local/name).read_bytes()
"""
footer = '''record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.44',
    heads=heads,github_main=heads[0],handoff_sha256=digest,handoff_bytes=len(new),
    four_local_and_remote_equal=True,exact_committed_payloads=True,payload_count=len(payloads),
    residual_bbs_hits=[bbs['rec_hits25'],bbs['rec_hits50']],distribution_formal_complete=False,
    retained_metric_best=intake['retained_best']['name'],owned_deleted_bytes=deleted_bytes,
    weights_downloaded=0,fresh_pair_audit_complete=False,goal_achieved=False)
(local/'residual_phase_publication.json').write_text(json.dumps(record,indent=2)+'\\n',encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write(f'\\nPVGround {record["time_cst"]} section20.376.44 published {heads[0]} fourlocal+remote SHA{digest}; residual9508 bbs{bbs["rec_hits25"]}/{bbs["rec_hits50"]}, distribution pending; controllerCPUproof only, {deleted_bytes}B verified owned retired/no archives. GoalACTIVE_UNMET.\\n')
print(json.dumps(record),flush=True)
'''
text = header + section + sync + footer
ast.parse(text)
output.write_text(text,encoding='utf-8')
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='PREPARED_NOT_EXECUTED',
    publisher_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    completed_residual_required=True, distribution_result_claimed=False,
    remote_queries=0,model_forwards=0,weights_deleted=0,active_training_changed=False)
(local/'residual_phase_publisher_preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
