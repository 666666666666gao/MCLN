"""Write the incremental handoff from completed evaluator receipts and reviews."""
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'referit_same_query_20261010'
result = json.loads((data / 'cpu_execution/SAME_QUERY_CPU_RESULT.json').read_bytes())
receipt = json.loads((data / 'cpu_execution/CPU_EXECUTION.json').read_bytes())
source = json.loads((data / 'source_review/EXPERIMENT_CODE_REVIEW.json').read_bytes())
actual = json.loads((data / 'actual_review/EXPERIMENT_AUDIT.json').read_bytes())
assert result['status'] == 'SYNTHETIC_NATIVE_EVALUATOR_SAME_QUERY_CPU_PASS'
assert receipt['exit_code'] == 0 and result['formal_accuracy'] is None
assert source['verdict'] in ('PASS', 'WARN') and not source['blocking_findings']
assert actual['execution_scope'] == 'SYNTHETIC_NATIVE_EVALUATOR_SAME_QUERY_CPU'
assert actual['verdict'] in ('PASS', 'WARN') and not actual['blocking_findings']
rows = '\n'.join('| {name} | {selected_query} | {synthetic_selected_mask_IoU:g} | 相同 |'.format(**case)
    for case in result['cases'])
notes = """## §20.376.143 — Nr/Sr作者入口与同Query评估：有限原生CPU检查完成

记录时间：{now}。承接§20.376.142，本节仅记录新增入口准备与评估接口证据，不重列历史架构、成绩或清理记录。

### 作者输入入口和真实源码来源

独立的作者入口R1只将train_dist_mod.py末尾原ScanRefer条件断言改为：butd_cls/joint_det/detect_intermediate=true，butd/butd_gt/augment_det=false；其余13份Python和对应Nr/Sr初始化配置不改。输入仍是场景实例框加预测类别，联合检测职责保留。保存的Sr作者脚本有跨数据集权重字面路径，本项目使用此前实际核验的对应Sr作者权重，不能照抄错配路径。R1是SOURCE_ONLY准备，尚未执行正常Nr/Sr入口、模型训练或完整作者配方。

特别澄清：此CPU warm源绑定的评估器是runtime_binding/model_source/src/grounding_evaluator.py，SHA256为39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677，与保存的作者upstream一致。R1说明中提及的dataset_source新版b77376d…是另一份项目源码；它的候选有效性排除、全部不合格记miss行为不能外推为本warm的原生行为。原R1审查与输入保留；本次隔离修订以真实warm为准。

### 同一Query的框与Mask

新隔离评估器保留作者bbs token分数、预测框对任意输入场景对象的IoU>0.25条件、分数乘0和原排序，不增加目标GT、GT Anchor、另一排名或回退。只记录bbs第一名Query，随后两条Mask计数都读取这一Query的预测融合Mask。bbf框统计继续作为诊断。

原mask_pos/mask_sem名称保留，但两者现在都是bbs所选Query的Mask；本地baseline与方法须采用同一明确标注的协议。作者原来的独立Mask排名不能直接替代此结果。这是同实例交付的报告修正，不是REC涨点或三个创新之一。作者零分过滤对负分及全部不合格候选的既有排序行为也原样保留。

### 实际执行及边界

实际授权CPU执行{started}至{finished}，退出码0。环境仍为已绑定的Py3.7/Torch1.10.2；116项基础warm源码逐项核验后复制到新独立目录，只覆盖准备的15份Python，再核验116项最终清单。旧源与活动ScanRefer源码未修改。

检查输入为5个显式构造的工程样例，每个B1/Q256/T256/P4；原始和修订原生GroundingEvaluator分别执行，合计10个实例、10次evaluate调用。人工指定框和Mask GT仅用于接口断言，不是数据集成绩，也不是模型预测生成的伪GT。

| 工程样例 | 实际bbs第一名Query | 对应合成Mask IoU | bbs/bbf框计数及分母对原始版本 |
|---|---:|---:|---|
{rows}

正分对象过滤样例中，原Box选Query1，原Mask独立选择Query0/2；Query1的Mask与GT不重叠，修订后的Mask命中下降是预期的正确结果，不能选择更好的另一Query来报告。负分和全框在场景对象外时的Query176只是本次真实CPU排序所得，不应当作固定跨设备tie规则。

所有框命中和分母与原始版本相同，两条新Mask的IoU、阈值命中及分母均对应实际bbs第一名Query。双GT样例只覆盖only_root评估，不证明多GT训练损失保护；检查不覆盖B>1、非零融合权重、非恒等超点映射或真实Nr/Sr完整表达。CUDA未初始化，数据构造、PV前向、criterion、优化器更新、新权重与当前训练查询均为0；正式精度为null，不据此准入GPU训练。

源码审查{source_verdict}；实际完整性审查{actual_verdict}，阻塞项0。请求gpt-6-astra/max，实际模型与effort仍UNATTESTED；fresh-context但same-family/provisional，不声称跨模型接受。

### 下一步与研究目标

ScanRefer当前唯一C-off正常训练与原观察器21631/PID48772不改，首次查阅仍为10月11日01:03:05北京时间；本节没有提前查询新的轮次或成绩。保留最好5677/4920，但尚未证明正常训练净增益和三项有效贡献。

原C损失按未过滤token分数选择训练Query，与Nr/Sr作者对象过滤后的部署Query仍未证明等价。最终C开关需由ScanRefer对照结果确定；若保留C，应在隔离源码单独解决这一职责差异。当前初始化JSON是接口准备，不能称为已确定的论文最终模型。

最终方法固定后，使用各自作者预训练权重、真实数据和同一完整结构，依次检查GPU前向、原生损失、模块更新和全状态恢复，再在唯一A100上串行启动Nr/Sr同起点训练。目标仍为ScanRefer同一模型Acc@0.25>59.5%、Acc@0.5>51%，三项有效机制及Nr/Sr完整独立验证；总目标ACTIVE_UNMET。

新增证据：refine-logs/pvground_native_joint_training_20261009/referit_same_query_20261010/。私有stderr、raw传输载荷和审查快照不公开；对应源码、工程结果及审查公开同步。
""".format(now=datetime.datetime.now().astimezone().isoformat(),
    started=receipt['started_cst'], finished=receipt['finished_cst'], rows=rows,
    source_verdict=source['verdict'], actual_verdict=actual['verdict'])
(data / 'HANDOFF_SAME_QUERY_CPU_20261010.md').write_text(notes, encoding='utf-8')
print(json.dumps(dict(status='SAME_QUERY_CPU_SECTION143_HANDOFF_PREPARED',
    formal_accuracy=None, full_goal_complete=False)))
