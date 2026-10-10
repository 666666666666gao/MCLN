# 原生匹配中 Text Mask 与 Query Mask 的职责

本记录针对已启动的正常 ScanRefer C-off 的原生源码。这里只读源码，不修改训练、优化器、数据、分数或检查安排，也不重新查询远端训练存活状态。既定下一次观察为2026-10-11 05:00:40北京时间。

## 源码证据

- `source/models/pv_ground.py:549`：选取单个文本Mask，然后 `expand(1,256,P)`。因此本条输入的全部候选在 `last_pred_masks` 中获得同一份Text Mask。
- `source/models/pv_ground.py:559`：候选自身Query Mask单独保存在 `sp_last_pred_masks`，并可能由候选支撑修正器更新。
- `source/models/losses.py:910`：末层criterion同时收到 `pred_masks`（Text）和 `sp_pred_masks`（Query）以及融合权重。
- `source/models/losses.py:328`：HungarianMatcher的直接Mask成本只读取 `pred_masks`，做硬二值判断、原始点映射及L1距离；没有直接读取 `sp_pred_masks` 或融合权重。
- `source/main_utils.py:282`：真实训练构建参数是分类1、框L1为0、GIoU为2；Mask成本系数在matcher内为0.0002。
- `source/models/losses.py:817`：匹配结果用于原生定位、语义及Mask损失；`loss_masks`以匹配的Query索引读取其自身Mask。
- 当前数据接口 `MAX_NUM_OBJ=132`，候选数为256。原生一对一匹配中，非空有效GT数量不大于候选数量。

## 可以推导什么

固定同一条输入的候选框与token分数时，公共Text Mask对每一个GT的距离相同于全部Query。它在成本矩阵每一GT列中添加相同常量：

`C[i,g] = C_semantic_geometry[i,g] + k[g]`。

当全部有效GT各匹配一次时，任何完整分配的Mask成本和均为 `sum_g k[g]`。因此这一直接Mask项没有提供区分候选自己的支撑的依据。单目标情况下，结论就是给全部候选加同一个常量不会改变argmin。真实浮点加法可有舍入；受控CPU检查在内存捕获实际成本矩阵，回执保留成本差摘要与分配，不把数学推导冒充逐位运行证明。

## 不能推导什么

Query Mask仍通过 `native_mask_geometry` 与 `candidate_span_mixer` 改变末层框；末层框再参与GIoU匹配。因此**候选自身Mask可以经由几何间接影响匹配**。固定框的CPU检查只隔离直接Mask成本，不能说整个正常模型没有使用Query Mask。

本记录不证明这是正常续训下降的主因，不证明训练分配真的发生了有害变化，也不提供新的ScanRefer精度。公共Text Mask参与原生分割损失，本记录不主张删除该分支。

过去的额外几何目标、选中候选Mask监督及G标签替换，与原生Hungarian的一对一分配不是同一项机制。已经完成的负实验仍保留，不因这一源码发现重新解释为未做过。

## 下一步边界

先完成活动C-off三轮规定预算。若后续需要调整训练责任，应直接比较“保留原生分配”和“候选自身支撑参与末层分配”，保持同一前向输出、真实有效GT对应、语义与定位成本以及同起点同预算控制。不能把所有Query都强行指向root，不能破坏其他实例或检测行的职责。

Mask参与匹配已有明确前作；这里首先是本项目的信息路径诊断，不能据此声称新颖性或第三个有效论文模块已经成立。未实施新的匹配策略，未改活动训练。
