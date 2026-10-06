"""Publish only actual accepted M0 and submitted fixed-budget M1, never scores."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parent
text=(root/'publish_preflight_launch.py').read_text(encoding='utf-8')
text=text.replace('preflight_launch_publication.json','fit_launch_publication.json')
text=text.replace('Publish reviewed reference sources and actual two-update sanity launch.',
    'Publish accepted actual two-update runtime and submitted full-budget pair.')
text=text.replace("preflight_controller=launch['process']","fit_controller=launch['process']")
text=text.replace('No full fit/new saved weights, protected5616/4511 unchanged.',
    'Full fit launched; no retrieved new accuracy, protected5616/4511 unchanged.')
text=text.replace('PUBLISH_PREFLIGHT_REVIEW','PUBLISH_FIT_REVIEW')
text=text.replace("launch=json.loads((local/'preflight_launch.json').read_bytes())", "launch=json.loads((local/'fit_launch.json').read_bytes())")
text=text.replace("live=json.loads((local/'PREFLIGHT_EXECUTION_SESSIONS.json').read_bytes())", "live=json.loads((local/'M0_ACCEPTANCE_AND_FIT_LAUNCH.json').read_bytes())")
text=text.replace("assert launch['status']=='PREFLIGHT_LAUNCHED_NOT_COMPLETED' and launch['process'].startswith('703128 ')",
    "assert launch['status']=='TWO_ARM_FIT_LAUNCHED_NOT_COMPLETED' and launch['process'].startswith('711378 ')")
text=text.replace("assert live['controller_pid']==703128 and live['sole_preflight_observer_native_session_id']==44835\nassert live['deployment_session_closed'] and live['deployment_exit_code']==0 and not live['observer_closed']",
    "assert live['fit_controller_pid']==711378 and live['sole_fit_observer_native_session_id']==27853\nassert live['preflight_controller_closed'] and live['preflight_observer_closed'] and not live['observer_closed']")
text=text.replace("=='20.376.81'","=='20.376.82'")
text=text.replace("prefix='refine-logs/pvground_mask_reference_20261006/'","prefix='refine-logs/pvground_mask_reference_20261006/accepted_m0_fit_launch/'")
start=text.index("section=f'''")
end=text.index("assert chr(65533) not in section",start)
text=text[:start]+"""section=f'''

## 20.376.83 Mask空间参考两组真实预检闭合，固定预算完整对照已提交（{stamp}）

承接§82的启动快照。M0实际于2026-10-06 11:08:20.348532闭合，controller703128、最后child707951、exit0；唯一observer44835于11:08:29第二次240秒检查记录闭合，native会话也已exit0，不重启。每组同8条真实fit输入重复2次更新，父模型/R全部状态不变、每组456102参数/10状态，原语义头每普通forward一次；更新后缓存上游评分/Mask重放精确。此不宣称两次fresh forward或跨进程输出逐位一致。

中性33节点分布初始解码精确等于其所选参考中心及既有SIZE_FLOOR尺寸；实际raw点独立核验全部256参考、误差0；39条既存空支撑在CUDA执行确认保留提供的原粗框先验、无GT字段。第二步所有10参数梯度非零、冻结参数无梯度；CPU保存/严格重载模型delta与Adam组/步数/所有moment精确。两组序列化5481547B，0权重文件、0精度结果，全部24实际预检文件154131B已收集/校验。native/fused各461.21/470.05秒，合计controller946.20秒；各峰值allocated3911791616B、reserved5672796160B，仅这次冻结骨干预检，不据此机械扩batch。

小批次也显示范围限制未消失：第一步native匹配DFL目标48面中outside0，Mask参考outside2；额外native143候选/858面中outside120，Mask参考128候选/768面中outside196。第二步相应126/200面。参考不可形成的候选计数1033→1045（每步全部8×256），它们保留原粗参考而没有删除；不把这些小批次计数外推为全验证发生率或失败唯一原因。不修改本轮节点/损失/采样规则。

M1实际于{launch['time_cst']}提交、pgrep核对controller711378，screen pvg_mask_reference_fit_20261006，唯一本地observer27853已启动。各组3723更新、29778输入各一次、B8/effective8/accum1、LR1e−5/WD5e−4/clip0.1；相同官方PV/G/4511、共同重置output2/保留hidden8，从全新Adam开始、不承接M0两步状态。每组初始9508、新fit的6887初始与终点、终点9508；初始架构收益与训练收益分开。只训练已有几何头，原生last/bbs、同QueryBox/Mask、全256保留，不加入新质量排名、teacher或V99双源。

实际启动前GPU空闲，数据盘可用1206886400B、所需reserve960163041B，系统盘463011840B，warm环境/算子缓存复用、无安装。估计总22000秒约17:20:54；sole observer首17:14:14、之后240秒，到闭合后流式收集实际证据、排除权重，不在远端另建大tar。此发布尚未取得M1闭合或新准确率；SourceOnly审查及M0通过不算涨点。结束后fresh terminal审计、所选初始/终点实际strict恢复，并仅保留真实指标最佳已闭合权重，原PV/G与V99必需依赖保护。当前trainedbest仍5616/4511、47.4443%、距4754差243；offline4848仍是§81诊断，目标ACTIVE_UNMET。

已审主源保持原版本；acceptedM0、actualfitlaunch/observer、源码与收集清单见{prefix}。原始point/NPZ与权重不上传Git；三仓、桌面与远端文档同步仍为append-only原字节前缀。旧§82是当时启动快照，当前状态以本节和M0_ACCEPTANCE_AND_FIT_LAUNCH为准。
'''
"""+text[end:]
text=text.replace("new.count(b'## 20.376.82 ')==1","new.count(b'## 20.376.83 ')==1")
start=text.index("names=[path.name")
end=text.index("assert len(names)==len(set(names))",start)
text=text[:start]+"""names=['prepare_fit_publication.py','publish_fit_launch.py','record_m0_and_fit.py',
    'M0_ACCEPTANCE_AND_FIT_LAUNCH.json','preflight_wait.json','fit_launch.json','fit_resource_check.json',
    'fit_observer_started.json','PUBLISH_FIT_REVIEW.json','PUBLISH_FIT_REVIEW.md']
names += [str(path.relative_to(local)).replace('\\\\','/') for path in sorted((local/'preflight_complete').rglob('*')) if path.is_file()]
"""+text[end:]
text=text.replace("'/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006'",
    "'/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006/accepted_m0_fit_launch'")
text=text.replace(' Protected4511 native/fused Mask spatial-reference pair: SOURCE reviewed,actual2-step sanity launched; no accuracyyet.',
    ' Protected4511 Mask spatial-reference pair: actualM0 accepted,full3723fitperarm submitted; no newaccuracyyet.')
text=text.replace('Record reviewed Mask spatial-reference source and actual two-step sanity launch',
    'Record accepted Mask-reference two-step runtime and actual fixed-budget pair launch')
text=text.replace("section='20.376.82'","section='20.376.83'")
text=text.replace('observer_native_session_id=44835','observer_native_session_id=27853')
text=text.replace("status='MASK_REFERENCE_PREFLIGHT_ACTUAL_LAUNCH_PUBLISHED'","status='MASK_REFERENCE_FULL_PAIR_ACTIVE_PUBLISHED'")
text=text.replace("reference_preflight_launch=str(local/'preflight_launch.json')",
    "reference_fit_launch=str(local/'fit_launch.json')")
text=text.replace("Wait sole44835; collect actual M0 closure before fresh full fit. No new accuracy or best yet.",
    "Do not restart711378/27853;sole firstcheck17:14:14,240safter;fullclosure auto collects. No newaccuracy or best yet.")
text=text.replace('Doc82 actualMask-reference M0 launched;controller703128/soleobserver44835;fullfitnotstarted.',
    'Doc83 actualMask-reference M0 accepted and fullpair launched;controller711378/soleobserver27853;no newaccuracyyet.')
ast.parse(text)
(root/'publish_fit_launch.py').write_text(text,encoding='utf-8')
print('FIT_PUBLICATION_PREPARED_NOT_EXECUTED')
