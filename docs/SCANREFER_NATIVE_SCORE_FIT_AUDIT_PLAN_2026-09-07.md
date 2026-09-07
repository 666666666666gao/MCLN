# 固定fit输入的原生评分与候选截断核验

目的：判断20.141的合成评分差异是否实际出现在ScanRefer输入中，是否改变原生Top1或Top16覆盖。此为只读诊断，不训练模型、改变部署规则或开展正式验证。

已有correct-mesh教师缓存固定512条fit表达，保留256框和名为native_scores的分数。源码实际将`build_full_rec_query_state.default_scores`写入该字段，只核对原生evaluator聚合命中数。缓存无法还原真实evaluator的逐Query分数/map，不能把其512/512 Top1保留当成真实原生Top1保留证据。

继续使用该缓存预先固定的512条等间距fit身份（分区盐和选择不变），不按本次分歧、错误或GT重新选例。使用正确mesh、保护E71、原Python3.7/Torch1.10/A100环境、B12、无增强、seed0；加载明确622文件快照，所有辅助和local/extent默认关闭。新增脚本及两个已有数据/评估小工具置独立运行目录，不改快照或包搜索路径。

每批只执行一次模型前向：

1. 通过实际evaluator的`_position_top_indices`调用记录其原始分数、合法轴和Top1，禁用额外selector/reranker覆盖；保留原有正尺寸评估表示。
2. 在同一outputs上计算SourceChoice default和候选适配器原default/contrastive分数，保存main map及分项，检查原生与SourceChoice的一致性。
3. 按受保护Parent artifact的既定候选规则生成原Top16，记录真实原生Top1是否保留。原候选构造、模型与旧artifact不变。
4. 仅作诊断，另计算将default输入替换为原生分数、保留同一contrastive和Top-K预算的候选集合；不运行或改写其Parent/V99分数，不将其当新系统成绩。
5. 所有选择先于root GT IoU计算；随后记录各Top1和固定候选集合的框质量。集合oracle明确使用GT，不可部署。

保存真实点SHA、框、分数、稀疏main map、候选索引及合法性，以CPU独立重算分数/排名/框IoU/覆盖。与旧512缓存逐行对照身份、点SHA和框，明确其native_scores字段究竟对应哪一路。完整模型state前后逐值一致，保护权重及源码/mesh清单核验。

禁止新checkpoint、优化器、正式/holdout评估，禁止凭此诊断修改map、阈值、候选规则或宣布精度提升。只有真实输入证据能决定下一步是否值得研究此接口；相同权重重命名或普通质量loss均不是新结构。既定Scan正式不退化及Nr/Sr REC总目标不变。
