# 任务查询的接入位置与实验归因补充

本轮只读核对现有原生路径，在 B 独立配对运行期间明确后续 C 的实现边界。
没有修改运行中的 B、冻结主干、候选规则、损失或正式 evaluator。

## Box 预测头还承担语义评分

`models/modules.py:111` 的 `ClsAgnosticPredictHead` 在同一输入上分别调用
`center_residual_head`、`size_pred_head` 和 `sem_cls_scores_head`。
最后一个输出是 soft-token 分类分数的来源；它不是只有几何回归功能的头。

`models/mcln.py:2057` 附近，contrastive Query 投影先读取 Decoder Query，随后
整个 prediction head 再读取该 Query。因此，若仅在最后 prediction head 的入口
注入“Box 任务 Query”，会改变 soft-token 分数，而先前计算的 contrastive 投影
仍读取原 Query。不能把这种干预宣称为只改变几何，或统一了两个语义证据源。

后续 C 若声明“身份评分保持、仅分开几何与 Mask 读取”，应将任务更新接入几何
子头，并直接核验两种语义分数及源选择张量；若也更新身份推理，则应明确作为另一
项实验变化，训练和评估同时使用相同表示。这里没有实际接入 C，也没有新增 Gate。

每层 Box 中心使用 `cluster_xyz + center_residual`；下一层注意力位置才使用前层
预测中心/尺寸的 detach 结果。新增局部回归时不能无说明地把原 Box MLP 的坐标
基准换成上层预测框，否则零残差也可能改变原输出。范围变化还可能影响合法性过滤，
所以“分数相同”不自动保证最终 REC Query 相同，仍需逐行核验。

## B 改变的是共享 superpoint 记忆

`models/mcln.py:1973` 起先生成 superpoint 特征，之后这些特征分别进入：

1. `x_query` 投影后的各候选 Query 与 superpoint 的 Mask 读出；
2. Text Mask 的三层 SWA/FFN 读取及预测；
3. Text 分支产生的融合权重。

当前 B 残差加入共享 superpoint 特征，因此可以通过上述三条原生路径影响最终
Mask。它没有增加融合 Gate，但也不能称为“只干预原始 Query Mask”。B 的质量
通过条件仍是已冻结的原生融合 Mask 指标，不能在训练后更换主要验收对象。

`scripts/nr3d_candidate_contract.py` 中的 `rec_selection.mask_iou` 和
`box_oracle_after_filter.mask_iou` 来自 evaluator 的点级 **融合 Mask**。
本次“固定 REC Query”及“已有合格最佳框 Query”的条件诊断因此能排除重新选择
Query 的影响，但不能单独区分 Query Mask、Text Mask 和 alpha 各自的贡献。
若结果支持继续开发，再以原始分支输出的独立诊断解释机制；不把融合变化自动写成
原始 Query Mask 的质量改善，也不从注意力图推断已经实现任务解耦。

以上边界补充实验解释，不改变 B 的两个对照、更新预算、输入或固定质量门槛。
两臂的 native 起点/终点仍可与同协议 M5 native 控制作独立复核；该复核是重现性
诊断，不是事后新增的质量门槛。

## 本次起点提供的新选择路径证据

B 的全部 6,172 条起点输出已经完成，逐行与 M5 native 的输入、框、选择和 IoU
完全一致；两臂零初始化输出也完全一致。REC 命中为 6005/5306，原生融合 Mask
命中为 5767/5057，mIoU 68.881519715%。

其中 133 条表达的 REC/Mask Query 不同。仅将已有 REC Query 用于读取同一份
融合 Mask，得到以下反事实变化：

| 阈值 | 修复 | 破坏 | 净变化 |
|---|---:|---:|---:|
| Mask@.25 | 3 | 16 | -13 |
| Mask@.50 | 3 | 12 | -9 |

mIoU 变化为 -0.153471988pp。这与此前正式输入配对 7,898 行上的正变化方向不同；
两者样本集合不同，不能拼接或视为同一结果的重复测量。这个新证据说明，统一输出
Query 的规则本身具有质量取舍，不能直接等同于“新增共享身份表征带来收益”。
最终协议的统一仍需明确对照，也不能由某一集合上的改善自动推断其他集合受益。

已有合法最佳框 IoU>.5 的起点集合为 5,931 条，相应最佳框 Query 的融合 Mask
mIoU 为 71.188117542%。这只是一个 GT 条件诊断，分母不同于全体 6,172 行，
不能据此直接与全体原生 Mask 均值作提升比较。

证据：`refine-logs/point_detail_pair_20260906_v1/baseline_identity.json` 和
`baseline_query_identity.json`；起点逐行文件 SHA
`884d72879a7a9485309ec9dadc588357a8ea69dbe847854e4af6934a4eef84ef`。
未读取本实验训练后质量、未新增 forward、未改变 evaluator。
