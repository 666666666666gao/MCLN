"""Prepare a static complete-run and native matched-endpoint publication batch."""
import ast
import json
from pathlib import Path


root=Path('\\\\?\\'+str(Path(__file__).resolve().parent))
base=root.parent
candidate=root/'coarse_endpoint_training_20261014'
target=base/'pv_matched_endpoint_normal_20261015'
assert not target.exists()
terminal=json.loads((root/'OBS_lower_span_lr_e8_complete_INTAKE.json').read_bytes())
assert terminal['remote_status']['status']=='complete' and not terminal['controller_alive'] and not terminal['child_alive']
pair=json.loads((candidate/'OBS_preflight_actual_pair_terminal_expected_INTAKE.json').read_bytes())
assert pair['remote_status']['actual_total_native_updates']==4 and not pair['controller_alive'] and not pair['child_alive']
audit=json.loads((candidate/'actual_pair_review/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['verdict']=='WARN' and not audit['blocking_findings']
normal=json.loads((candidate/'NORMAL_FINAL_DUPLICATE_START_RECORD_20261015.json').read_bytes())
assert normal['controller_pid']==741474 and normal['actual_native_child_witness_pending']
target.mkdir()
summary=dict(retained_best_hits=[5677,4920],goal_hits=[5705,4945],lower_B_lr_epochs=8,
    lower_B_lr_terminal_hits=[5635,4706],lower_B_lr_completed_cst=terminal['remote_status']['completed_cst'],
    actual_pair_native_updates=4,actual_pair_audit='WARN_ZERO_BLOCKERS',
    normal_arm='final_duplicate',normal_controller_pid=741474,native_child_witness_pending=True,
    new_normal_full_validation_metrics=None,next_arm='coarse_endpoint',next_arm_not_launched=True,
    next_observation_cst=normal['next_observation_cst'],observer_native_session=52865,observer_pid=37700,
    MCLN_formal_training_launched=True,MCLN_transfer_is_historical_complete=True,full_goal_complete=False)
(target/'CLOSEOUT.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())
notes='''目标不变：同一完整ScanRefer9508模型至少5705/4945命中（60%/52%），三个有直接消融证据的有效机制，并在达标后验证作者Nr3D/Sr3D权重、训练同一最终模型超过对应协议论文指标。当前最好仍5677/4920，即59.7076%/51.7459%，差28/25；本次没有达标或三个有效贡献声明。

### 降低B学习率的正常联合训练已经完整结束

| 完整9508条原生last/bbs | @0.25命中 | @0.5命中 |
|---|---:|---:|
| E0／保留最好 | 5677 | 4920 |
| E1 | 5652 | 4732 |
| E2 | 5628 | 4741 |
| E3 | 5624 | 4725 |
| E4 | 5628 | 4742 |
| E5 | 5627 | 4712 |
| E6 | 5662 | 4740 |
| E7 | 5632 | 4721 |
| E8 | 5635 | 4706 |

这是原正常训练控制器432868／子进程432871的真实终态。两者2026-10-15 21:00:08正常退出，21:05观察确认均不存在。E6→E7→E8严格命中4740→4721→4706，触发两次相邻下降，也已达到最大8轮。持平或上升仍重置下降计数；没有按“连续低于起点”提前停止。共8×4583=36664次训练循环，冷进程优化器step完整恢复未审计；原生每轮drop-last后使用36664条，训练集总36665条。

该轮core/G/backbone仍1e-6，A1e-5，B1e-6，seed2027、batch8、accumulation1、原RoBERTa冻结，其他原生可训练部分与A/B正常联合更新。E8为59.26588136%/49.49516197%，相对E0−42/−214命中。训练后严格最好E4为5628/4742，仍低于E0；没有晋级。不能把八轮完结继续解释为尚未训练完，也不无界延长原配方。

E8缓存诊断给出Full256严格覆盖7341，E0为8234；同一E8所选Query的原粗框／融合Mask端点／最终框严格命中3900/4818/4706。与E0逐条输入配对严格修复345、破坏559；9126次Query槽变化不是已确认的物理实例变化。这些同时包含上游、候选及选择变化，不能单独证明是哪一模块造成下降。15份小工件及2份日志尾部已核对字节与SHA；九轮完整raw-row归档传输在900秒超时，原始远端文件保留，没有因此重新运行模型。本节不声称已独立重算这些raw坐标、Mask或原始数据集GT。

### 接下来只比较定位监督作用在哪个输出

当前原生criterion读取被几何模块覆盖后的last_center/last_pred_size；仍被用于融合的最后层原粗框没有独立的该输出定位项，其他五个辅助层也不是这一具体输出。经融合系数等路径仍可能有间接梯度，因此这不是“整个原粗框头完全无梯度”、已确认bug或下降唯一原因。

新对照复用同一最后层Hungarian索引、GT和原生center L1+0.5 size L1/GIoU公式，额外权重均为(10L_bbox+2L_giou)/7：final_duplicate监督最终框，coarse_endpoint监督保留的原粗框。这样控制额外定位权重，只改变被监督的端点；不重新匹配、不扩展语义正例、不改CE/NCE/Mask目标、不新增参数或第二套推理排名。原来基于最终框的匹配仍保留，不能称为完整恢复原baseline训练协议。该项首先是训练职责修正假设，本身不自动成为第三项创新。

两组均从共同官方PV／G／既有A／既有B权重重建，使用fresh optimizer。A、B已有适配历史应计入总成本，不能称为仅从官方模型训练本轮；不加载刚结束的负终点，也不加载预检权重。固定seed2027、batch8、同一结构、数据与学习率、全部256候选、同Query框与Mask及原生唯一last/bbs；单A100串行，最大8轮、相邻严格下降2次停止。实际预算可能因停止规则不同而不同，比较共同已完成epoch并披露总更新，不将不同预算的终点称为固定预算控制。

### 真实接入预检及证据边界

两组真实工厂与原生train_one_epoch预检于22:27:17至22:47:43完成，每组两次真实更新、16条输入，合计4次更新；相同初始1295状态、相同两批增强输入及五组LR已核对。原core/G为合并梯度和更新见证，A、B有各自的正梯度与更新见证；并未独立量化每个G-reader参数或只来自原loss的梯度。两组保存并在同进程重载全部模型、Adam/scheduler/RNG状态；不声称独立冷进程恢复。额外端点直接梯度覆盖各步8个原匹配Query，未匹配输出该额外项直接梯度为0，不代表共享参数不会影响它们。

fresh独立同模型家族审查11项确定性检查、70个输入，整体WARN、阻断0。实际面板全部root槽0，没有非空多GT保护的实证；审查没有重读原始数据集或checkpoint二进制，不提供9508条精度。请求Astra/max，实际路由UNATTESTED，same-family/provisional。预检工件是工程接入证据，不能当方法涨点。

### 当前正常训练与观察安排

正式控制组final_duplicate控制器741474已从全新共同起点启动，真实训练子进程身份和完整E0结果尚待观察；coarse_endpoint未启动或排队，待控制真实结束后串行。新正常训练尚无可报告精度，不能把历史5677/4920当成本轮已经重验通过。

唯一观察器37700／native52865定于北京时间2026-10-16 00:18:46读取，依据数据加载及完整E0验证估时选择启动后55分钟。此前不查当前日志、进程、GPU、epoch、句柄或另开观察器；如必要，后续间隔240秒。发布本节只同步静态源与记录，不查询当前训练或改在训源码。

### 清理和后续任务

已清理旧lower_B轮较差latest841677344字节；该轮E0模型通过与本地canonical逐张量相同的1295模型及原非model payload完整内容重建后，删除远端重复best615023944字节。重建不是原ZIP字节副本，原metadata保留。另核验原face模型完整本地615120706字节归档后，仅删除其远端逐字节重复best；该模型不是当前官方/G/A/B初始化依赖，本地1301状态仍保留。两份无正式分数的M0权重审查后删除1683352256字节，不作为正式初始化。最后清理后数据盘空闲3044020224字节，高于未降低的2605151860字节保存门槛；这是当时容量记录，不能当服务器实时容量。

前述清理总计3755174250字节；最佳模型、必要官方PV/G/A/B/V99/E71、数据集、日志及唯一候选raw都保留。一次退休连接超时后间隔240秒重试成功；旧C-row重复删除因原文件已不在而在任何unlink前终止，删除0。没有因传输／观察失败重启NN。

当前全目标仍未完成。先验证这个端点职责修改能否超过相同控制和5677/4920强起点，不把相对差控制优势当作新增能力。Nr3D/Sr3D尚没有本轮作者完整baseline或新模型结果；达标后使用相应作者权重，按既有各自原生GT/文本/损失系数接口验证并独立训练，同一最终结构，不按数据集切换推理排名，不做多seed。
'''
(target/'CLOSEOUT.md').write_bytes(notes.encode())
copies={
    'LOWER_B_LR_REPORT.md':root/'lower_span_lr_completed_20261015/REPORT.md',
    'LOWER_B_LR_METRICS.csv':root/'lower_span_lr_completed_20261015/metrics.csv',
    'LOWER_B_LR_COMPLETE_RECEIPT.json':root/'LOWER_SPAN_LR_TERMINAL_RECEIPT_20261015.json',
    'EXPERIMENT_PLAN.md':candidate/'EXPERIMENT_PLAN.md',
    'EXPERIMENT_TRACKER.md':candidate/'EXPERIMENT_TRACKER.md',
    'AT_SEAL_EXPERIMENT_PLAN.md':candidate/'actual_pair_review/AT_SEAL_EXPERIMENT_PLAN.md',
    'AT_SEAL_EXPERIMENT_TRACKER.md':candidate/'actual_pair_review/AT_SEAL_EXPERIMENT_TRACKER.md',
    'models_losses.py':candidate/'source/models/losses.py',
    'matched_endpoint_geometry_objective.py':candidate/'source/matched_endpoint_geometry_objective.py',
    'matched_endpoint_config.json':candidate/'source/matched_endpoint_config.json',
    'stage_isolated_sources.py':candidate/'stage_isolated_sources.py',
    'native_matched_endpoint_preflight.py':candidate/'runtime_preparation/native_matched_endpoint_preflight.py',
    'initialization_reference.json':candidate/'runtime_preparation/initialization_reference.json',
    'final_duplicate_protocol.json':candidate/'runtime_preparation/final_duplicate_protocol.json',
    'coarse_endpoint_protocol.json':candidate/'runtime_preparation/coarse_endpoint_protocol.json',
    'matched_endpoint_preflight_pair_controller.py':candidate/'launch_preparation/matched_endpoint_preflight_pair_controller.py',
    'normal_matched_endpoint_controller.py':candidate/'launch_preparation/normal_matched_endpoint_controller.py',
    'launch_preflight_after_terminal.py':candidate/'launch_preflight_after_terminal.py',
    'launch_normal_after_preflight.py':candidate/'launch_normal_after_preflight.py',
    'MODEL_SOURCE_REVIEW_COMPLETE.json':candidate/'review/REVIEW_COMPLETE.json',
    'RUNTIME_REVIEW_COMPLETE.json':candidate/'review_runtime/RUNTIME_REVIEW_COMPLETE.json',
    'LAUNCHERS_REVIEW_COMPLETE.json':candidate/'review_launch/LAUNCHERS_REVIEW_COMPLETE.json',
    'EXECUTION_REVIEW_COMPLETE.json':candidate/'review_execution/EXECUTION_REVIEW_COMPLETE.json',
    'OBSERVER_REVIEW_COMPLETE.json':candidate/'review_observer/OBSERVER_REVIEW_COMPLETE.json',
    'SOURCE_STAGE_RECEIPT.json':candidate/'isolated_source_stage_20261014/SOURCE_STAGE_RECEIPT.json',
    'EXPERIMENT_AUDIT.md':candidate/'actual_pair_review/EXPERIMENT_AUDIT.md',
    'EXPERIMENT_AUDIT.json':candidate/'actual_pair_review/EXPERIMENT_AUDIT.json',
    'EXPERIMENT_AUDIT_COMPLETE.json':candidate/'actual_pair_review/EXPERIMENT_AUDIT_COMPLETE.json',
    'DETERMINISTIC_CHECKS.json':candidate/'actual_pair_review/DETERMINISTIC_CHECKS.json',
    'audit_cached_evidence.py':candidate/'actual_pair_review/audit_cached_evidence.py',
    'ACTUAL_PAIR_AUDIT_CLOSURE.json':candidate/'MATCHED_ENDPOINT_ACTUAL_PAIR_AUDIT_CLOSURE_20261015.json',
    'ACTUAL_PAIR_TERMINAL_INTAKE_RECORD.json':candidate/'ACTUAL_PREFLIGHT_TERMINAL_INTAKE_RECORD_20261015.json',
    'NORMAL_START_RECORD.json':candidate/'NORMAL_FINAL_DUPLICATE_START_RECORD_20261015.json',
    'NEXT_E0_OBSERVER_OWNER.json':candidate/'OBS_final_duplicate_e0_expected_complete_OWNER.json',
    'M0_WEIGHT_RETIREMENT.json':candidate/'MATCHED_ENDPOINT_UNSCORED_PREFLIGHT_WEIGHT_RETIREMENT_RECEIPT_20261015.json',
    'LOWER_B_LR_LATEST_RETIREMENT.json':root/'LOWER_SPAN_LR_LOSING_LATEST_RETIREMENT_RECEIPT_20261015.json',
    'LOWER_B_LR_ARCHIVED_REMOTE_BEST_RETIREMENT.json':root/'LOWER_SPAN_ARCHIVED_BEST_DUPLICATE_RETIREMENT_RECEIPT_20261015.json',
    'OLD_FACE_REMOTE_DUPLICATE_RETIREMENT.json':root/'VERIFIED_FACE_BEST_REMOTE_DUPLICATE_RETIREMENT_20261015_RECEIPT.json',
    'final_duplicate_NATIVE_M0_RECEIPT.json':candidate/'OBS_preflight_actual_pair_terminal_expected_files/final_duplicate/witness/NATIVE_M0_RECEIPT.json',
    'final_duplicate_NATIVE_M0_TRAINING_DIAGNOSTIC.json':candidate/'OBS_preflight_actual_pair_terminal_expected_files/final_duplicate/witness/NATIVE_M0_TRAINING_DIAGNOSTIC.json',
    'coarse_endpoint_NATIVE_M0_RECEIPT.json':candidate/'OBS_preflight_actual_pair_terminal_expected_files/coarse_endpoint/witness/NATIVE_M0_RECEIPT.json',
    'coarse_endpoint_NATIVE_M0_TRAINING_DIAGNOSTIC.json':candidate/'OBS_preflight_actual_pair_terminal_expected_files/coarse_endpoint/witness/NATIVE_M0_TRAINING_DIAGNOSTIC.json',
    'PAIR_NATIVE_STATUS.json':candidate/'OBS_preflight_actual_pair_terminal_expected_files/status.json',
}
for name,path in copies.items():
    assert path.is_file(),str(path)
    (target/name).write_bytes(path.read_bytes())
template=(base/'publish_pv_task_family_capacity_authorized_20261014.py').read_text(encoding='utf-8')
template=template.replace('pv_task_family_capacity','pv_matched_endpoint_normal').replace('PV_TASK_FAMILY_CAPACITY','PV_MATCHED_ENDPOINT_NORMAL')
template=template.replace('pv_matched_endpoint_normal_publication_20261014','pv_matched_endpoint_normal_publication_20261015')
template=template.replace('pv_matched_endpoint_normal_local_commit_20261014','pv_matched_endpoint_normal_local_commit_20261015')
template=template.replace('pv_matched_endpoint_normal_20261014','pv_matched_endpoint_normal_20261015')
template=template.replace('pv_initial_and_dedup_publication_20261014.json','pv_task_family_capacity_publication_20261014.json')
template=template.replace("prior['section']=='20.376.176'","prior['section']=='20.376.177'")
template=template.replace("b'20.376.177' not in old","b'20.376.178' not in old")
template=template.replace("section='20.376.177'","section='20.376.178'")
template=template.replace('## §20.376.177｜已选候选几何范围与训练职责复核（2026-10-14）','## §20.376.178｜低B学习率八轮终态与端点定位监督正常对照（2026-10-15）')
template=template.replace('Record selected box family capacity and native training targets','Record complete eight-epoch PV run and native matched-endpoint training')
template=template.replace('Historical_E0_retained_5677_4920_local_GT_only_analysis_no_new_NN_metric','Retained_E0_5677_4920_lowerB8_complete_actual_pair4updates_normal_child_pending')
template=template.replace('normal_training_alive_at_recorded_observation=True','normal_training_alive_at_recorded_observation=None')
template=template.replace('lower_span_lr controller and child verified03:20:49; real first epoch35batches; no new trained complete metric','final_duplicate controller741474 launch observed; native child and fullE0 await scheduled00:18:46 observation; no live training query in publication')
template=template.replace('datetime.datetime.now().astimezone().isoformat()','datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()')
ast.parse(template)
publisher=base/'publish_pv_matched_endpoint_normal_authorized_20261015.py'
assert not publisher.exists()
publisher.write_bytes(template.encode())
print(json.dumps(dict(status='STATIC_MATCHED_ENDPOINT_NORMAL_PUBLICATION_PREPARED',files=len(copies)+2,
    next_observation_cst=summary['next_observation_cst'],current_training_queries=0,neural_forwards=0)))
