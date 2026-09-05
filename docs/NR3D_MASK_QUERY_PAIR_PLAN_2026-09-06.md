# C1：候选 Mask Query 读取原生记忆的配对短训

本计划在读取C1任何训练后留出结果前固定。B已封存FAIL；C1原模块的4个CPU测试及
16行、12次真实CUDA forward预检PASS。此实验从原受保护Nr3D parent独立启动，
不加载B、M5、P2/R1或L1的训练artifact。

## 两臂与保持固定的路径

| 项目 | native | memory |
|---|---|---|
| 起点 | 原受保护权重 | 相同权重，新增输出矩阵为零 |
| 原生可训练层 | 仅x_query，6张量、664,992参数 | 相同 |
| 新增任务读取 | 无 | 每候选Mask Query以内容及框位置偏置读取原生SP记忆，再调用原点积writer |
| 新增参数 | 0 | 74,880参数、8张量 |
| 总可训练参数 | 664,992 | 739,872 |

此处仅训练候选Mask任务读取，不同时改变共享身份/Box分支、视觉记忆、x_mask、
rel_encoder、Text Mask或alpha。原有候选、合法性过滤、REC及Mask选择规则保持。
两臂的其余全部参数和模型buffer冻结，model.eval；不采用任务token、新损失、
新体素特征或硬裁剪。原query_mask_fusion_calibrator必须为None，避免x_query更新
同时改变alpha。新增模块与完成真实预检的实现文件逐字节相同。

这是相同旧投影更新条件下的结构增量对照，不是等参数量对照；memory的可训练参数
多约11.26%。通过也不能单独证明所有收益来自“任务解耦”，或已经减少梯度冲突。

## 输入与预算

- 沿用既定scene划分：fit取CSV顺序前2048条、262场景；holdout6172条、98场景。
  后者未参与新增参数训练，但主干以前见过，属于模块留出而非正式或新场景泛化。
- 612源码、724数据文件、parent、模块、runner、summary、测试与计划写入SHA清单。
- 每臂2epoch、B4、每epoch512步，共1024更新；无增强、workers=0，每epoch种子0/1，
  两臂共用同一个实际batch。记录每个fit batch的实际模型输入点Tensor SHA和行ID。
- fresh AdamW lr1e-5、weight_decay.0005、clip_norm.1，原生损失及系数。
  首个fit batch只做梯度检查，不调用optimizer。
- 起点、终点各评估完整holdout一次，B16、seed1000、workers=0；不按中途质量续训。
  起点两臂逐行输出相同；完整起点还需与既有B起点逐行核验。

## 完整性与固定质量门槛

逐batch核对REC框与得分以及Text Mask/alpha，两臂必须一致；起终逐行输入、REC及
Mask Query选择、合法最佳框索引及Text Mask/alpha hash也必须相同。
保存6/14张量artifact和optimizer，检查冻结状态、实际参数改变、有限权重/矩、
1024步和fit顺序。新的fit点hash补充跨运行比较证据，不能追溯补齐B/M5历史缺失项。

memory终点须**同时**相对native终点、原保护起点满足：

1. 原生选中融合Mask的平均IoU增加至少.002，即0.2个百分点；
2. Mask@.25命中数不下降；
3. Mask@.50命中数不下降。

沿用evaluator严格IoU>阈值，报告修复/破坏/净变化，以及2000次scene聚类bootstrap。
固定REC Query的融合Mask、5931条起点合法好框对应Query的融合Mask为诊断项，
不替代质量门槛；GT只用于损失、评估和建立诊断集合，不进入记忆特征或部署选择。
这里保留原选择路径以隔离任务读取，尚未将正式REC/Mask选择合并成单一实例决策。

失败封存该版本，不自动加步数/调学习率/调门槛；native控制也不自动升级。
通过后才准备正式验证及后续共享实例选择研究。本实验冻结REC，不能自身完成
Nr3D>60%或Sr3D目标；ScanRefer保护系统及Sr3D缺失权重状态不受本实验改变。

入口 `scripts/run_nr3d_mask_query_pair.py`；独立终态复算
`scripts/summarize_nr3d_mask_query_pair.py`。先在原Py3.7/Torch1.10环境检查后，
使用现有GPU锁启动；按实际速度估计阶段完成时间，接近预计时刻再以240秒观察。
