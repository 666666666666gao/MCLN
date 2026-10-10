## §20.376.144 — Nr/Sr原生GPU预检源码准备，尚未准入或执行

记录时间：2026-10-10T22:45:35.992779+08:00。上一节同Query CPU结果已闭合，本节只记录下一步必要工程代码，不新增实验成绩、模型晋级或GPU任务。

新目录复用上一节15份隔离源码原字节，使用对应Nr/Sr作者预训练核心、新G/A/B状态fresh；仅两份新初始化清单关闭C。不加载ScanRefer终点。该C-off版本尚待ScanRefer对照决定，不是已经固定的论文最终结构。若最终保留C或改变A/B，须按实际完整结构修订这项检查。

准备的native_referit_preflight.py沿用已执行过的原生两步训练与完整恢复检查，只适配Nr/Sr真实输入。正常TrainTester.get_loaders加载完整train/val；禁止debug/eval/eval_train，因为原debug会把训练split改成val。原生训练增强、场景实例框加预测类别、联合检测混合不变；实际全量行数和每轮更新数由未来执行记录，不能用有限128条检查代填。

检查将从原生train数据取得8条指代表达输入与8条检测输入，交错形成两批B8；Sr首条表达含辅助实体和anchor，检测行也有多个有效GT。这是刻意选取的工程输入，不是原sampler的完整epoch、独立精度评估或完整作者配方。保持原生网络、criterion、AdamW和scheduler；记录七个prefix的匹配，以last_原匹配为准，验证全部有效GT有对应、G额外集合不改变原匹配Query和检测行的直接监督。

预期检查分组梯度与实际参数更新，再实际保存并重建恢复1295模型状态、Adam、scheduler以及Python/NumPy/Torch/CUDA RNG。工程权重不附正式指标、不用于正式续训。第一步零输出模块的内部梯度可能为0，span内部按两步合计检查；这只是待执行的断言，尚未证明运行通过。额外G集合可能为空，此时零梯度检查不能证明非空替换已发生；Sr检查两个有效GT槽及非空anchor ID，不额外宣称anchor点掩码非空。共享参数更新也可能改变未入选候选预测。

准备阶段发现脚本摘要误用了Windows写入前的LF文本，而实际文件为CRLF，原摘要与原始字节不符；已改为对写入后的read_bytes()计算SHA256，并修正历史“pretrained_span”记录键为“fresh_span”。原问题与快照保留，修正没有重跑GPU或改变15份模型源码。

源码审查WARN、阻塞项0，执行范围SOURCE_ONLY；模型/effort实际身份UNATTESTED，same-family/provisional。当前协议final_method_admitted=false，脚本在Torch/CUDA导入前要求准入成立。此布尔值不是独立准入证据；目前没有部署、启动器、自动排队或GPU执行。

活动ScanRefer C-off训练、原观察器21631/PID48772和10月11日01:03:05首查时间不改，本轮没有提前查询。待其真实终态和最终结构确定、资源及单卡空闲实际核验后，才在独立目录、原单卡锁下串行执行Nr/Sr实际检查及完整训练。保留最好、三项有效贡献和同一完整模型跨基准要求不变；目标仍ACTIVE_UNMET。

新增源码证据：refine-logs/pvground_native_joint_training_20261009/referit_native_gpu_preparation_20261010/。实际前向、loss、更新、恢复、GPU、正式成绩均为未完成，不能将源码审查写成上述事项已通过。
