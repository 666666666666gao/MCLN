# 原生REC与候选特征的评分口径：CPU源码核对

本检查确认同名default在两个接口中并非相同数值定义，但没有证明哪一项应修改，未改变模型、候选、评估规则或指标。SourceChoice早期修复仍正确存在，不能再把它当成新发现的未修复问题。

## 实际路径

| 路径 | 目标词main map | 其他组件及职责 |
|---|---|---|
| 原生evaluator位置评分 | `_parse_gt`把main正值置1 | main+modifier+pronoun+relation−other；后续可以被传入的候选分数/selector输出覆盖 |
| SourceChoice default | `compute_default_source_scores`同样把main正值置1 | 与原生默认表达式一致；历史V4修复已经存在 |
| Parent候选特征适配器 | `_first_map_row`保留输入的原权重 | `build_full_rec_query_state.default_scores`使用同一加减表达式，但main可能是按token数量归一化的值 |
| 原生soft-token Hungarian分类成本 | 读取targets的main map | 同时另有框/几何等成本；不能描述成完全不看几何 |
| 原生position alignment loss | 匹配Query的main/属性/代词/关系使用既有不同权重 | Scan/Nr为0.6/0.2/0.2/0.1，Sr为0.625/0.125/0.125/0.125；并非部署分数的直接监督 |

`src/joint_det_dataset.py:get_positive_map`有按每行token数归一化的构造；`TrainTester._get_inputs`透传map。不同文本构造路径和真实map分布仍需实际fit输入核对，不能由合成例子推断受影响样本数。

## 隔离执行现有表达式

在原Python3.7/Torch1.10环境执行`scripts/audit_native_rec_score_semantics.py`。AST只抽取实际`_parse_gt`、evaluator组件加减、候选适配器组件加减、SourceChoice default函数；不加载模型、dataset或完整evaluator过滤器。三个合成例子保持两候选的main总概率及其他组件概率不变，只将main分配到1/2/4个token。

| main token数 | 原生分数A/B | SourceChoice A/B | 候选适配器A/B | 原生/适配器Top1 |
|---|---:|---:|---:|---|
| 1 | 0.400/0.350 | 0.400/0.350 | 0.400/0.350 | A/A |
| 2 | 0.400/0.350 | 0.400/0.350 | 0.200/0.300 | A/B |
| 4 | 0.400/0.350 | 0.400/0.350 | 0.100/0.275 | A/B |

所有输入map保持原值，SourceChoice与原生在1e−7绝对容差内一致。该反例证明现有定义能够改变排名，不是实际失败比例、模型性能测试或改法有效性证据。未执行Top16构造、Parent/Geometry/V99学习读出，也未比较最终部署结果。

## 与已有实验的关系

不能直接把候选适配器改为二值map并声称修复：受保护Parent/Geometry/V99在当前特征分布和候选规则下训练，改变main会同时改变候选成员、特征和旧归一化的意义。现阶段记录为接口口径差异，保护系统不动。

源码也确认既有SourceMoE已有相对于GT的quality/listwise损失，joint/frozen_readout已有合法Query及Variant的GT排序。另加一个普通质量loss或回退到多源Gate不构成新机制。

下一步只在固定fit输入核验实际main map分布、两种分数排名、原生Top1在原Top16中的保留率；必须分开报告“数值不同”“候选截断不同”和“真实框IoU不同”，不能把候选改变直接记成提升。需要实际输出才能决定这是否是值得继续的接口问题；不使用正式验证场景调map/阈值，也不将合成反例作为新长训授权。REC目标仍按Scan先过保护线、随后Nr/Sr的既定顺序推进。

## 来源与运行

绑定622文件原生源码快照中6个相关文件。首次仅在源SHA检查处停止：本地losses和dataset为CRLF，快照为LF。逐文件对比本地、Git、远端确认LF标准化后内容完全相同，再使用远端实际字节SHA绑定运行；没有修改模型源来绕过检查。初次日志/退出码与最终记录均保留。

归档`refine-logs/native_score_contract_cpu_20260907_v1`；最终receipt SHA `8fb6e1bcc7d5a44c5545faba0ca7691f4af79584a5cd21c886c517ca9c23aeef`。GPU前向0、数据集行0、模型权重加载0、优化器更新0、正式评估0。
