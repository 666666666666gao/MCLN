# 任务查询相对原生结构的实际增量

2026-09-05，基于 main 19c3714 与服务器冻结源码；只读检查，没有新训练。
本报告补充局部点细节方案，不改变已封存的 B 预检清单或当前 L1 设置。

## 已有的共享与分工

原生 `models/mcln.py` 的最后一层 Decoder Query 已同时服务于 Box 和 Query Mask。
Box 的中心、尺寸和 soft-token 分数分别由 `ClsAgnosticPredictHead` 的独立 MLP 输出；
Query Mask 则先经过独立的 `x_query` 三层投影，与 superpoint 特征逐候选点积。
因此，不能把“共享最后一层表示”“保留两个任务头”或“候选各有一个 Query Mask”
本身写成后续方案首次提供的能力。广播给 256 个候选的是另一个 Text Mask 分支。

`SetCriterion.forward()` 计算一次 Hungarian 索引，然后将该索引同时交给 Box 和
Mask loss；`loss_masks()` 用对应 Query 索引读取两种 Mask。本身不存在两个任务
分别做一次不相干 Hungarian 匹配的问题。M3 已实际检查匹配索引和梯度支持。

当前 matcher 的 Mask 成本读取 `pred_masks`，而不读取逐候选的 `sp_pred_masks`。
在受保护配置下，前者是广播的 Text Mask；同一样本的各 Query 收到相同的这项成本。
它没有显式比较各候选 Query Mask 的质量。这是代码层面的功能界限，不能单独证明
应该修改 matcher，或证明当前错误主要由这项成本造成；本轮没有改损失或匹配规则。

## 533 条索引差异的准确解释

进一步检查已封存 M2 全部 7,899 行：

- 7,899 行的 Mask Query 都等于 `protected_selector.before_filter.top_query`。
- 534 行的 REC 过滤前后第一名不同，其中 1 行过滤后无合法 Query。
- 其余 533 行，恰好就是上一轮记录的 REC/Mask Query 不一致的全部表达。

原生 Mask 选择用 `selected_source_scores.argmax()`；该路径没有 REC 的 detector
重叠过滤。REC 使用同一分数，但应用自己的候选合法性过滤。记录中的全部差异均
伴随这一选择路径中的第一名变化，而不是发现了另一个独立 Mask 身份评分器。
保存字段不足以逐条证明“Mask Query 必然被过滤”，也不证明不同 Query 对应不同
物理实例；不同 Query 可以覆盖同一目标，排序平局细节也未逐条另行记录。

上一轮固定 REC Query Mask 的配对变化仍成立：7,898 行 Mask@.25 净+47、@.50
净+27、mIoU +0.2884171 个百分点，分别有17/9个命中被破坏。
但这首先是**候选选择路径的反事实诊断**，不能把它当作“新增共享身份表示有效”的证据。
本轮没有改正式 evaluator，也没有将过滤规则变化计入网络收益。

## 后续 C 应当检验什么

固定任务常量本身不是新的视觉信息。对带 bias 的第一层，
`W(z + e_task) + b = Wz + (W e_task + b)`；只在现有 `x_query` 前加一个固定
任务嵌入，可以被第一层 bias 重参数化。这个等式限定于这种插入方式，不能外推为
所有任务 token 或任务解码器都无效。

C 的实质候选是：在同一个候选身份索引下，Mask Query 在输出前实际读取细尺度
点/superpoint 记忆并获得局部更新，而 Box/身份路径继续读取相应的空间与场景信息。
需要比较访问的信息和更新路径，不能只比较增加了多少投影或常量。
原生已经共享训练匹配，实验应保持这一点；最终输出规则的统一另作明确协议对照。
需要实测参数更新、Mask 质量和 REC 修复/破坏，不能预先宣称消除了任务梯度冲突。

先完成 L1，再执行 B 的真实预检与独立学习对照；C 此轮只明确必要增量，尚未实现。
局部 Mask 分支暂不回写 REC 表示，因此不能单靠 B 的 Mask 改善完成 Nr3D REC 目标。
后续局部证据进入对象/身份推理是否有益仍需单独证明。

## 来源对应

本轮已核对下列本地文件 SHA 与冻结源码 manifest 完全一致：

- `models/mcln.py`：1245cf56c7111611cdb8b842ace583371a4e1c21f0b6a226e6562ebb7e6ab73a。
- `models/modules.py`：96861185adf921ad53f9e649887528f17d9d9a94288f969b88f6ee95cf71df14。
- `models/losses.py`：cf720a782db7a34790bef4f86d79810f2fc33eeee25c14a48faecad19a714d35。

选择路径：`src/grounding_evaluator.py` 的 `_resolve_learned_mask_queries` 和
REC 候选过滤路径；逐行字段来自 `scripts/nr3d_candidate_contract.py`。
可重算结果：`scripts/summarize_nr3d_query_identity.py` 与
`refine-logs/mask_branch_diagnostic_20260905_v1/query_identity_analysis.json`。
