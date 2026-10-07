**PV-Ground face-support：结果到论文论断**

时间：2026-10-07T13:00:37.494347+00:00。**claim_supported: no**；对本轮具体负结果的置信度high。完整性WARN、blocking findings=0；确定性实件核验PASS。fresh context；same-family / provisional；实际backend/effort身份未认证。

本轮真实结果支持的结论是：在固定seed2027、共同冻结父模型和每臂3723步训练预算下，face-region读取没有比face-center控制带来正式阈值增益，不能计作一个有效论文模块。这不证明该思想在所有配方和数据集上均无效。

| 论断 | 判定 | 依据 |
|---|---|---|
| 配对实验真实完成 | 支持 | 29778唯一fit rows、每臂3723步、实际闭合、2421文件SHA及2378 NPZ一致 |
| face-region优于同预算face-center | 不支持 | 正式9508/141 scenes均5593/4832；两个阈值逐表达标签完全相同 |
| 本轮训练改善已有reference | 不支持 | 同次reference5598/4848；repair4/13、damage9/29，net−5/−16 |
| 保护父模型仍是最佳实际候选 | 支持 | 5598/4848胜过两个terminal；真实checkpoint SHA与1304-state fullCPU witness未变 |
| 采样／盒输出实际发生变化 | 支持，限执行层面 | M0成员变化；终止两臂在9508表达上共12088503坐标元素不同；不等于准确率增益 |
| Full256有合格候选可用于诊断 | 需限定 | 终止coverage8915/8130，3322/3298表达合格候选未被native选中；GT oracle，非部署效果 |
| 同一模型已达用户联合门槛 | 不支持 | 最佳5598/4848，相对5620/4764仍缺22/0 |
| 已有三个有效模块或新的Sr3D/Nr3D结果 | 不支持 | 本轮新增有效模块证据为0，Sr/Nr未在本轮执行 |

内部6887/106 scenes holdout：控制6147/5665，方法6146/5666，仅−1/+1，不能替代正式结果或声称一致增益。当前Mask5812/5133、mIoU47.10762848393134%仅stored IoU重计，不是当前raw-mask CPU独立重算。

跨阶段reference有444坐标元素/36表达漂移、validity有12位/8表达漂移；所有保存的selected共同字段无漂移。历史父模型query另有3个表达改变。因此方法论断应以同次前向的两臂及reference比较为主；不能把所有all256跨阶段差值归于训练。8个hidden tensors继承11169步并达到14892，2个重置output tensors本轮3723步，不能写成零历史初始化。

可直接使用的论文表述：

> 在固定seed2027、冻结PV/G父模型并共享同一次parent前向的等预算实验中，将face-center KNN替换为bounded-face成员读取未改善ScanRefer的正式Acc@0.25/0.5：两臂均5593/4832个命中，且相对同次reference减少5/16个命中。我们保留5598/4848的原模型，并将该采样改动记录为未获支持的消融。

后续先记录这个负消融，再依据已观察的错误提出可检验的新机制，用同预算控制验证贡献；遵守固定seed2027、单native评分、同query Box/Mask约束。只有同一完整模型达到5620/4764且证明三个有效贡献后，才进入冻结方法、对应作者权重独立训练Sr3D/Nr3D的阶段。本报告不触发训练或清理，不把完整性核验通过解释为正面模块论断通过。

完整实件和来源证据见EXPERIMENT_AUDIT.json/.md及TERMINAL_AUDITOR_CHECKS.json。两份判断JSON绑定当前SUMMARY与closed_weight_inspection实际摘要，R2C还绑定本次audit JSON。
