# CEGD 草稿接口审计（2026-09-05）

审计对象为本地 `.codex_mcln_cegd_20260902` 中尚未提交的草稿；此次只读，未改该工作区。
执行方向以master §20.31及用户最新“只升级关系读出”的要求为准，旧§20.18的大组合首版不再直接实现。

| 实际代码 | 与当前实验要求的差异 | P2必须落实的约束 |
|---|---|---|
| `candidate_edge_direct_scorer.py:121` 直接按完整 `base_scores` Top-K；forward无合法性输入 | evaluator随后可能继续过滤，但训练loss把所有Top-K位置标为valid | 在Top-K前传入由现有重叠规则计算的 `valid_query_mask[B,Q]`；训练、评分、evaluator共用其Query映射 |
| 同文件150–193行从Top-K节点中产生Anchor Top-M | 目标候选裁剪也裁掉参照物；`anchor_head(node)`在pair形成前计算，主要是单节点选择 | 目标Top-32与完整有效Query记忆分开；首轮不先按目标分数裁剪Anchor |
| 同文件219–225行所有行都对实体Anchor softmax | 无独立“没有参照关系”的状态 | 两个对照组共用显式无关系状态；不声称单次软聚合实现AND/NOT或多参照逻辑 |
| 同文件295–322行对全部行计算soft-listwise loss | 当全行IoU均未过.25时，仍给坏候选分配排名目标；BCE的零标签不能取消这条梯度 | 无合格候选的行不作为“存在正确答案”的排名正例；若保留连续质量监督，单独标记覆盖状态 |
| 草稿额外包含node token attention、entity/presence证据、9维几何及quality组合 | 与原关系机制相比同时改变多个因素 | P2只比较全句条件与候选对条件的文本读出；共用目标/Anchor集、几何、输出头和训练合同 |

草稿已有两处正确约束应保留：loss通过 `attach_candidate_targets(..., root_only=True)` 对齐第0个目标；
compact分数通过 `query_indices` scatter回原Query轴。这两点不是本次新发现的故障。

首版输入合同因此需要在原5个参数外增加现成的合法候选mask；这是对已存在评估过滤的对齐，
不增加GT目标或GT Anchor输入。seed物理ID属于诊断元数据，无须作为新网络特征输入。

最小对照还必须固定相同的上游权重、候选身份、有效文本mask、几何输入、共同输出头、数据顺序和更新步数。
若仅新分支排除padding、控制分支仍让padding参与全句池化，就无法把结果只归因于候选对读取方式。
P1输出口径闭合前，不据此选择新的split、LR或评分utility；G0/P1完成后再冻结可执行G1合同。

## REC与Mask路径的新增事实

`GroundingEvaluator.evaluate_bbox_by_pos_align()` 对REC调用 `build_detector_overlap_valid(...,.25)`；
`_resolve_learned_mask_queries()` 只应用可选的 `moe_valid_mask`。SourceChoiceSelector输出本身不发布这个mask。
服务器冻结源码的真实evaluator已在三个CPU合成对照中复现：没有重叠过滤时两边选0；只有REC重叠过滤时
REC选1、Mask选0；显式共享valid mask时两边选1。完整回执为
`REC_MASK_SELECTION_COUNTEREXAMPLE_20260905.json`，benchmark rows=0，不是Nr3D性能结果。

不能直接把这个反例当成“应给Mask也加框过滤”的依据：Mask质量与Box质量不是同一量，仍须看真实行的净变化。
已有padding identity v2中的selected fused Mask指的是REC所选Query条件下的Mask，不等于正式Mask evaluator
实际选出的Query。这一口径已明确，排队中的v2输入文件不改。
