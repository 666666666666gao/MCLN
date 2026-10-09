复核完成：**WARN，0 个阻塞项**。独立 CPU 数学复算通过，原输入哈希未变。

- 现存 native/Mask/half/whole/extremal 计数全部复现：5615/4495、5606/4881、5675/4857、5676/4921、5677/4920。
- 216 个端点/GT 交点组合包含连续三轴门控最大值的证明成立；oracle 为 **5839/5379**，独立面投影为 **5877/5472**。
- **93** 个严格错误是固定 Query 和源框下的排除/可行性证据，其中 **44/499** 属于 FUSED-Mask 合格错误；不代表 own Query Mask 合格。
- 旧 `bound_below_half` 字段实际表示精确 oracle 排除：93 例中仅 9 例由较松的一维上界排除，另 84 例需要三维精确最大值。
- 实际存在 **31 个不相交轴区间，涉及 30 个表达式**（x/y/z = 12/8/11），说明自由独立面门控需要保证上下界顺序。本次投影全部有效。

结果仅支持 GT 离线表示能力诊断，不能声称可学习收益、新正式成绩、身份改善或三模块有效性。未操作正在运行的训练。

报告、机器结果和 seal 已写入 [actual_review](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/face_coupling_exact_20261010/actual_review/EXPERIMENT_AUDIT.md)。请求与响应、元数据和 SHA 快照已存档。请求 gpt-6-astra/max；实际模型/effort/backend **UNATTESTED**，same-family/provisional，无跨家族接受。
