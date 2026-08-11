# MCLN 自适应多源 MoE 与训练内 Query 重排：设计及交接

更新日期：2026-08-04

状态：核心模块、训练损失、同-query 评估、两阶段训练脚本、V24 relative-risk
set head 和 sidecar 配置契约均已实现；真实 GPU smoke 与两轮完整 ScanRefer formal
run 已完成。V24 未刷新受保护 best，本文末尾记录其失败归因、审计和清理结果。

## 1. 目标与口径

| 指标 | 当前保护结果 | 目标 | 还差 |
| --- | ---: | ---: | ---: |
| REC Acc@0.25 | 58.2878%（5542/9508） | >= 59.00%（5610/9508） | 68 hits |
| REC Acc@0.50 | 48.6012%（4621/9508） | >= 49.00%（4659/9508） | 38 hits |
| Mask Acc@0.25 | 59.6971% | 不低于并争取提升 | - |
| Mask Acc@0.50 | 49.0324% | 提升 | - |
| Mask semantic mIoU | 41.7676% | 提升；长期参考 44.72% | - |

上表 REC 保护结果包含旧 parent + geometry 后处理。epoch-71 backbone
本身的 plain 指标是 57.993% / 46.350%。因此必须分开报告：

- 模型级：新 MoE learned query 对比同一 backbone 的 plain `default` query；
- 系统级：完成 learned-score-aware sidecar v2 后，对比 58.2878% / 48.6012%；
- 旧 sidecar 绑定 epoch-71 的 SHA-256 和模型结构，不能直接套在新权重上。

当前 `rec-query-v1` 只使用 default/contrastive 与几何特征，不消费
`selected_source_scores`；geometry sidecar 还会覆盖 MoE 的 query 选择。因此仅为
新 SHA 重跑旧 v1 sidecar 不会叠加 MoE 收益。若模型级结果通过门禁，后处理阶段
必须先新增包含 learned score/rank 的 adapter v2，再训练新的 parent/geometry
artifact。

## 2. 旧实现审计结论

### 2.1 原 selector 不是可微融合

`SourceChoiceSelector` 为每个样本硬选一个源。`argmax` 后索引
`selected_source_scores`，主排名损失无法沿该选择回传；训练目标只是“猜哪个
固定源 top-1 更好”的离散分类。它还只观察每个源的 top-1 query，丢掉其余
255 个候选的信息。

### 2.2 最初接入的 SourceMoE 实际没有学习融合

审计早期 MoE 代码发现：

- `routed_scale` 初始化为 0；
- 总损失只有 balance loss，而 balance loss 不依赖 `routed_scale`；
- 所以 `routed_scale.grad is None`，输出永久退化为 `default`；
- `argsort/scatter` rank normalization 阻断源分数梯度；
- 测试只对融合输出人工求和，没有覆盖真实 `compute_hungarian_loss`。

这不是“MoE 收益不明显”，而是主任务梯度根本没有训练到融合路径。

### 2.3 Box 与 mask 的 query 身份曾经分裂

REC 可使用 learned/geometry query，但两个 mask evaluator 原来仍分别按 position
和 semantic 分数选 query。同一条样本可能用 query A 的框和 query B 的 mask，
旧 reranker 又只受 box IoU 监督，因此 mask@0.5 和 mIoU 没有受到最终选择器的
直接约束。

### 2.4 仍未解决的底层 mask 限制

MCLN text-mask 分支每个样本只产生一个 mask，随后广播到 256 个 query；
`adaptive_weight` 也是 sample 级标量。当前工作先解决“选哪个 query”，尚未把
text mask 和融合权重改成 query-specific。这是后续模块，不应在本轮结果里声称
已解决。

## 3. 当前实际架构

实现位于 `models/source_moe.py`，由 `models/mcln.py` 在最后一层预测完成后接入。
默认使用三个原始源：

| 源 | 含义 | 角色 |
| --- | --- | --- |
| `default` | soft-token position 组合分数 | 共享锚点/保护基线 |
| `contrastive_text` | query-token 对比相似度 | 路由专家 |
| `mask_text` | text mask 与 query mask 的 Dice/置信度 | mask 兼容性专家 |

不把 `default_rank_blend_contrastive005/010` 当专家。它们包含 ScanRefer 上人工
调出的 0.05/0.10 常数，继续使用会把数据集特调重新包装成“MoE”。

历史 9,508-row 四源评估中，default 为约 57.98/46.36，固定源 top-1 oracle
也只有约 58.50/47.01，仍低于 59/49。说明“更准确地选一个单源”存在明确上限；
本实现必须依靠 query 级融合和 256-query context reranker 在各源 top-1 之外找回
候选，才有可能达到目标。

### 3.1 可导 rank 与 query 级稀疏路由

每个源先按 query 做精确 rank normalization。前向仍是尺度不敏感的离散 rank，
反向通过 z-score + sigmoid 的 straight-through proxy 回传。Router 输入为：

```text
[contrastive query feature, pooled text, normalized box, all source ranks]
```

Router 对每个 `[B,Q]` query 单独产生专家概率，top-k 前向稀疏、反向使用
softmax straight-through 梯度。三个源时有两个路由专家，当前取 `top_k=1`；
若取 2 就等于两个专家全选，没有稀疏意义。

最终源融合的实际公式是：

```text
r_source = straight_through_rank(source_score)
r_route  = sum(topk_gate * routed_source_ranks)
fused    = r_default + tanh(routed_scale) * (r_route - r_default)
```

`routed_scale=0`，所以初始化输出严格保持 `default` 排名。Gate 最后一层用
`std=1e-3` 的小随机权重打破 top-k tie；由于 residual scale 为 0，这不会改变
初始预测。负载均衡损失排除无效 query，并归一化到均匀路由时理论最小值 1。

### 3.2 训练图内的 QueryContextReranker

源融合后不是立刻 argmax，而是把同一批 256 个 query 送入一层
`MultiheadAttention + FFN`，输出每个 query 的有界偏置：

```text
selected_source_scores = fused + max_delta * tanh(score_head(context))
```

默认 hidden=288、4 heads、1 layer、dropout=0.1、`max_delta=0.25`。
score head 全零初始化，因此模块加入时仍不改变 baseline。它属于当前 SourceMoE，
不是仍待实现的离线 reranker。

### 3.3 阈值感知 box-primary、mask-aware listwise 监督

真实总损失已加入 `source_moe_rank_loss`，不是只训练 balance loss。Box target：

```text
q_box(iou) = iou
           + 0.25 * sigmoid((iou - 0.25) / 0.05)
           + 0.50 * sigmoid((iou - 0.50) / 0.05)
L_box = KL(softmax(q_box / T) || softmax(score / T))
```

这会重点优化最终报告的 0.25/0.50 阈值，而不是只回归平均 IoU。

Mask 辅目标不能简单取 `max(box_iou, mask_iou)`，否则高 mask/坏 box query 会破坏
REC。当前采用字典序 tier：先保持 box 属于 `<=0.25`、`(0.25,0.5]`、`>0.5`
中的哪一级，再在同一级里按 mask IoU 排序。总排名损失默认：

```text
L_rank = L_box + 0.25 * L_mask_tier
L_total += 1.0 * L_rank + lambda_anchor * L_anchor + 0.01 * L_balance
```

为直接约束 validation 中的 fix/break，代码还提供 shared-query 阈值锚点损失。
对 `t in {0.25, 0.50}`：若 shared query 已正确，则要求它以 margin 压过所有
错误 query；若 shared query 错但候选集中存在正确 query，则要求最高 IoU 的正确
query 以 margin 压过所有错误 query。后一项比较的是分数最高的**全部错误候选**，
而不只是 shared query，因此最小化到 0 才真正保证该阈值下完成 fix。通用 CLI
默认 `lambda_anchor=0` 以兼容旧实验；正式脚本的新实验配置为
`--source_moe_anchor_loss_weight 1.0 --source_moe_anchor_margin 0.05`。

监督只使用 slot-0 root GT，只作用于 grounding 样本，混合训练中的 ScanNet
detection 行会被排除。GT 仅用于训练 target，推理不需要 GT。

### 3.4 同一 query 同时评估 box 与 mask

启用 `--eval_use_selector_choice_scores` 后，REC position 和两个 mask evaluator
都使用 `selected_source_scores.argmax()` 的同一个 parent query。若启用 joint
geometry sidecar，它的显式 parent mapping 优先，仍保证 box-mask identity 一致。

### 3.5 Top-k SelectiveFallbackGate（2026-07-27 新增）

完整 validation 证明软 anchor 只能减少错误覆盖，不能从结构上阻止它。因此
`SourceMoE` 现在显式保留 pre-gate `moe_candidate_scores`，并可启用网络内
`QueryFallbackGate`。它不是 validation 后调阈值的脚本，而是有独立参数和训练
loss 的模型层：

```text
default query = moe_shared_query from the shared-score argsort ranking
candidate pool = top-8(moe_candidate_scores)
pair feature(q) = [feature(q), feature(default), difference, score relations]
box_head_t(q), mask_head_t(q) -> {break, neutral, fix}
true utility(q) = weighted fix - 2 * weighted break + 0.25 * mask utility
decision_head(q) -> {combined break, neutral, combined fix}
margin(q) = logit_fix - max(logit_neutral, logit_break)
selected query = argmax positive margin, otherwise exact default fallback
```

REC 两个阈值权重固定为 `2:1`；另有同结构的 mask 辅助 head，以 0.25 utility
权重形成训练期联合真实收益。三个 head 全零初始化，初始 decision margin 为 0，
由于推理只接受严格正 margin，新增模块在第 0 步逐元素等于 shared baseline。
训练标签完全由 query 相对 default 是否跨过 0.25/0.50 得到，不使用数据集特定
IoU gap；box/mask/decision 三组分类均采用按当前数据自动逆频率平衡的 focal
loss，并对 `break` 再施加 2 倍 false-override 代价。另有与部署动作完全一致的
行级损失：类别为一个显式 fallback 加 top-k query；联合真实收益存在正值时监督
其中收益最大的 query，否则监督 fallback。最终 decision loss 为候选级联合分类
与行级动作 loss 的等权平均。

最初版本曾直接用四组重加权分类概率计算
`P(fix)-2*P(break)`。逆频率 focal 的输出不是校准概率，正式训练后虽然分类 loss
下降，全部 candidate utility 仍为负，故该推理公式已被上述直接监督的 decision
margin 取代。这里修正的是训练目标与部署动作的结构失配，不是修改 validation
decision threshold；正式 threshold 从始至终固定为 0。

这里必须用 top-k query，而不能只 gate 原 MoE top-1：第二组 formal 中，若有
完美 top-1 gate，理论上最多得到 `5512+104=5616` hits@0.25 和
`4408+127=4535` hits@0.50，后者只有 47.70%，数学上不可能达到 49%。同一轮
MoE 排名的 Top-5 已为 61.18%/52.74%，所以可学习 top-k 安全选择既保留 fallback，
也保留了达到两个目标的候选上界。

### 3.6 Enriched evidence 与 calibrated utility（V3/V4）

V3 不改部署动作，只增强 gate 的通用输入证据。每个 query 除原 64 维
contrastive projection、rank、box 和 pooled text 外，再接入冻结 decoder 的
288 维完整 query feature，以及按场景、按源跨 query 标准化的 raw source
confidence。该改动把 gate-only 可训练参数从 72,207 增至 183,951，但仍只有
13 个 gate tensor，主干与候选 MoE 的 state dict 形状均不变。

V3 正式结果证明更多证据本身不能解决选择校准：validation switch 达 14.71%，
但 REC 两阈值均为 `fix < break`。V4 因此不再增加 feature，而做两项语义修正：

1. shared `default` 源严格复刻 `GroundingEvaluator._parse_gt()`，先把主目标
   `positive_map` 的所有正值二值化为 1，再与 modify/pron/relation/other map
   组合。`moe_shared_query` 仍由 argsort rank 派生，不能退回 raw `argmax`，否则
   raw-score tie 时会再次产生训练、gate 和 evaluator 的 default 身份分裂。
2. 新增 `source_moe_gate_objective=calibrated_utility`。部署 margin 仍为
   `m=logit_fix-max(logit_neutral,logit_break)`，切换规则仍固定为 `m>0`；训练直接
   回归连续真实 utility：

```text
L_utility = mean(w * SmoothL1(m, true_utility))
w = false_override_weight, if m > true_utility, else 1
L_decision = (fixed-cost CE + fallback/top-k row CE + L_utility) / 3
```

decision CE 使用固定 break cost，不再按 batch 类别逆频率重加权；行级 CE 也只
保留固定 false-override cost，不再注入经验类别先验。box/mask threshold head
仍保留原 class-balanced focal 作为辅助监督，不参与部署 margin 的概率解释。
CLI 默认仍为 `balanced_focal`，因此 V1-V3 数值路径不变；续训已有 gate 时从
checkpoint config 继承 objective，旧 checkpoint 缺字段时按 legacy 解释。

## 4. 训练策略

入口：`scripts/train_scanrefer_source_moe.sh`，支持 `router`、`gate`、`joint`
三个显式阶段。`gate` 阶段必须提供已训练 MoE checkpoint，只允许
`source_moe.fallback_gate.*` 更新，主干和候选 MoE 均保持 eval/frozen。

脚本把 `SOURCE_MOE_LR`、`SOURCE_MOE_TOP_K`、`SOURCE_MOE_TEMPERATURE`、
`SOURCE_MOE_ANCHOR_WEIGHT`、`SOURCE_MOE_ANCHOR_MARGIN`、
`SOURCE_MOE_MASK_RANK_WEIGHT`、`SOURCE_MOE_QUERY_MAX_DELTA` 和 `VAL_FREQ`
以及 `SOURCE_MOE_GATE_USE_EVIDENCE_FEATURES`、`SOURCE_MOE_GATE_OBJECTIVE`
暴露为环境变量，实验无需改源码；命令末尾仍允许用显式 CLI 参数覆盖。
训练循环会保留整数 epoch checkpoint；若最后一轮已按 `VAL_FREQ` 完成验证，结束
阶段不会再对同一权重重复跑一次全量 validation。

第一阶段只训练 `source_moe.*`：主干参数全部冻结，整个主干保持 `eval()`，仅
MoE 为 `train()`，避免 BN/dropout 造成隐式漂移。默认正式配置：

```bash
PHASE=router MAX_EPOCH=1 \
  EXP=ssq_moe_router_lr3e4_e1 \
  LOG_DIR=/root/autodl-tmp/DATA_ROOT/output/source_moe_formal \
  bash scripts/train_scanrefer_source_moe.sh
```

Router 默认 LR 已从 smoke 的 `1e-3` 降为 `3e-4`。完整 train 每轮约三千 step，
不能根据 32-step debug 直接沿用 `1e-3`。当前正在运行的第一组 formal 是加入
锚点损失前启动的 control，因此其结果不含 `L_anchor`；若 control 的 fix 小于
break，下一组 router-only 才启用上述锚点项，不能混淆两组配置。

只有 router-only 满足以下门禁才进入联合训练：

- validation `fix025 >= break025` 且 `fix050 >= break050`；
- REC 两阈值至少不低于初始化的 plain default；
- mask@0.25 不下降，mask@0.5/mIoU 至少一项有明确提升；
- 路由专家均有使用，entropy、rerank delta、routed scale 均 finite。

联合训练必须显式传第一阶段 checkpoint，脚本不再默默回退到 epoch-71：

```bash
PHASE=joint CHECKPOINT_PATH=/abs/path/to/router/ckpt_epoch_1.pth \
  bash scripts/train_scanrefer_source_moe.sh
```

`--reduce_lr` 会加载模型权重但建立 fresh optimizer。默认 decoder/mask LR 为
`2e-5`、PointNet backbone 为 `2e-4`、MoE 为 `3e-4`；联合阶段同时加入 ScanNet
detection 数据，但 MoE rank loss明确排除其样本。

## 5. 正式实验与验收规则

每个 checkpoint 必须保存以下两组结果，不能只报最好的一个数字：

| 组别 | 必报内容 |
| --- | --- |
| Learned vs shared | learned/default 的 Acc@0.25、Acc@0.5、fix、break |
| Box-mask | mask@0.25、mask@0.5、semantic mIoU，同-query 身份 |
| 路由 | routed scale、entropy、每专家 usage、rerank abs mean/max |
| Provenance | checkpoint SHA-256、config、源码测试结果、完整样本数 |

最终系统若继续使用 parent/geometry 后处理，必须先把 learned score/rank 纳入
新版本 candidate schema，再从新 checkpoint 重新缓存、训练并绑定 artifact；
不能直接复用当前 `rec-query-v1`。配置契约现已记录所有 MoE 结构字段；旧 artifact
缺字段时只按 `use_source_moe=False` 解释，既向后兼容又不会误配新模型。

正式 acceptance：

- 样本数必须为 9,508；
- REC `>= 5610` hits@0.25 且 `>= 4659` hits@0.50；
- mask@0.25 不低于 59.6971%，mask@0.5 与 mIoU 均不能继续恶化；
- 不以 128-row debug、train split 或 oracle 指标代替正式 validation。

## 6. 已完成验证

真实 ScanRefer debug GPU smoke：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_smoke/scanrefer/ssq_moe_debug/1785085628/`

- 32 train batches，前后各一次 128-row eval，全部 loss/gradient finite；
- `routed_scale` 从 0 更新到 0.0152313；
- query score head norm 更新到 0.0693894，router bias 发生变化；
- 与 epoch-71 比较，1,135 个共同非-MoE tensor 逐元素相同；
- 只变化 23 个 MoE tensor，证明 MoE-only 冻结边界正确；
- learned 对 default 在该小切片上 fix=break=1.5625%；
- position/semantic mask 使用相同 query 后 mIoU 完全相同。

该 smoke 不能证明精度收益。它只证明真实 loss、梯度、checkpoint 和 evaluator
链路已打通。锚点相关专项测试为 `37 passed`；加入锚点修复及实验 receipt 后的
最新完整 CPU suite 为 `2806 passed, 3 warnings in 147.78s`，三条均为既有
PyTorch/PointNet warning。

首组 formal control（无 `L_anchor`，`T=0.1`，`max_delta=0.25`）：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_formal/scanrefer/ssq_moe_router_lr3e4_e1/1785087235/`

| 指标 | shared default | learned | fix / break | 结论 |
| --- | ---: | ---: | ---: | --- |
| REC @0.25 | 5512/9508 (57.9722%) | 5406/9508 (56.8574%) | 126 / 232 | -106 hits |
| REC @0.50 | 4407/9508 (46.3504%) | 4281/9508 (45.0252%) | 210 / 336 | -126 hits |
| Mask @0.25 | - | 5647/9508 (59.3921%) | - | 低于保护结果 |
| Mask @0.50 | - | 4626/9508 (48.6538%) | - | 低于保护结果 |
| Mask mIoU | - | 41.5661% | - | 低于保护结果 |

两次完整 validation 的所有结果完全一致。训练累计诊断显示专家 usage 约
47.4%/52.6%、entropy 0.6687，路由未塌缩；但 rerank abs max 达 0.2456，接近
0.25 上限。正式 checkpoint 的 23 个 MoE tensors 均 finite，1,135 个共同非 MoE
tensors 与保护 backbone 逐元素相同。`ckpt_epoch_1.pth` size 607,936,805 bytes，
SHA-256：`86bab6dcb96ce0a95cd2a8cf71ae707788e9aef6032abb6489242c12529cd6f2`。
该 control 未通过任何进入 joint 的门禁。

锚点保护组（`anchor_weight=2.0`、`T=0.2`、`max_delta=0.10`）：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_formal/scanrefer/ssq_moe_router_anchor_w2_t02_d010_e1/1785092585/`

| 指标 | shared default | learned | fix / break | 结论 |
| --- | ---: | ---: | ---: | --- |
| REC @0.25 | 5512/9508 (57.9722%) | 5464/9508 (57.4674%) | 104 / 152 | -48 hits |
| REC @0.50 | 4408/9508 (46.3609%) | 4320/9508 (45.4354%) | 127 / 215 | -88 hits |
| Mask @0.25 | - | 5651/9508 (59.4342%) | - | 低于保护结果 |
| Mask @0.50 | - | 4638/9508 (48.7799%) | - | 低于保护结果 |
| Mask mIoU | - | 41.5278% | - | 低于保护结果 |

完整样本数为 9,508，进程退出码 0。anchor 把 control 的净损失从 -106/-126
缩小到 -48/-88 hits，但 fix 仍小于 break，故再次拒绝 joint。rerank abs mean/max
为 0.0970/0.1000，仍饱和在新上限；expert usage 为 48.47%/51.53%，不是塌缩。
checkpoint `ckpt_epoch_1.pth` size 607,936,933 bytes，SHA-256：
`c73c79529537e106f3d85096fafcf8a8ae2996222bcc49210ed8ae788a372788`。
23 个 MoE tensor 全 finite；1,135 个共有非 MoE tensor 与保护 backbone 逐元素
相同；验证后 `epoch_last` 的模型权重也与 epoch-1 逐元素相同。

结论：不再做 anchor/temperature/max-delta validation sweep。代码已转向 3.5 的
top-k hard fallback gate；其独立正式结果必须继续报告 default、pre-gate candidate
和 gated 三组 query，不能只报告 gated 数字。

首版概率 utility gate 在完整 validation 上零切换，REC 保持
`5512/4406` hits；其 11 个 gate tensor 与冻结边界审计均正常，失败原因是把
class-balanced focal 的非校准概率重新解释成风险收益。改为 3.5 的联合
decision-margin 与行级动作监督后，正式 v2 能切换 6.01%，但选择仍为净负：

| 指标 | shared default | gated | fix / break | 结论 |
| --- | ---: | ---: | ---: | --- |
| REC @0.25 | 5512/9508 (57.9722%) | 5462/9508 (57.4464%) | 50 / 100 | -50 hits |
| REC @0.50 | 4407/9508 (46.3504%) | 4381/9508 (46.0770%) | 60 / 86 | -26 hits |
| Mask @0.25 | - | 5650/9508 (59.4247%) | - | 未保护 |
| Mask @0.50 | - | 4631/9508 (48.7064%) | - | 未保护 |
| Mask mIoU | - | 41.5125% | - | 未保护 |

v2 checkpoint 13 个 gate tensor 全 finite；1,158 个共有候选/backbone tensor 与
输入 anchor checkpoint 逐元素一致；epoch-1 与 epoch-last model state 也逐元素
一致。该结果拒绝 joint 和相同结构续训。它把下一瓶颈定位为输入可分性：当前
gate 只看到 64 维 contrastive 投影和 rank，下一版加入 288 维 decoder query 与
跨 query 标准化的 raw source confidence，但继续冻结候选 MoE 和决策阈值。

V3 enriched-evidence 正式实验：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_formal_v3/scanrefer/ssq_moe_top8_enriched_evidence_e1/1785118054/`

| 指标 | fixed default | V3 gated | fix / break | 结论 |
| --- | ---: | ---: | ---: | --- |
| REC @0.25 | 5512/9508 (57.9722%) | 5419/9508 (56.9941%) | 113 / 206 | -93 hits |
| REC @0.50 | 4408/9508 (46.3609%) | 4305/9508 (45.2787%) | 148 / 251 | -103 hits |
| Mask @0.25 | - | 5634/9508 (59.2554%) | - | 未保护 |
| Mask @0.50 | - | 4618/9508 (48.5696%) | - | 未保护 |
| Mask mIoU | - | 41.3727% | - | 未保护 |

validation switch ratio 为 14.71%，说明 enriched evidence 造成更多覆盖，但没有
提升选择精度。checkpoint size `604018734`，SHA-256
`adea0787e997ab54691dc65df2872241d975533ffad2dcdae0aacbfd3ba01302`；
13 个 gate tensor 全 finite，1,158 个 anchor/common tensor 全部逐元素不变，
box/mask/decision 三个 head 均有非零更新。该版本正式拒绝。

V4 calibrated-utility 128-row smoke：

`output/source_moe_gate_smoke_v4/scanrefer/ssq_moe_top8_calibrated_utility_debug/1785123480/`

- fixed default 为 `63/57` hits，gated 为 `66/60` hits，两个阈值均
  `fix=3, break=0`；这是 debug 切片增益，不替代正式 validation；
- switch ratio 2.27%，margin 仍固定为 0；所有 loss/gradient finite；
- 183,951 个 gate-only 参数，13 个 gate tensor 全 finite，三个 head 与 encoder
  均更新；与 router anchor 共有的 1,158 个 tensor 全部逐元素不变；
- checkpoint SHA-256：
  `4ad15e242d3b0d01a80efb08e6b7ecb018be99620171676206f7dc7c94bce2d0`；
- 完整 CPU suite：`2828 passed, 3 warnings in 134.80s`。

唯一 V4 formal 已从 router anchor 完成，不做 margin sweep：

`output/source_moe_gate_formal_v4/scanrefer/ssq_moe_top8_calibrated_utility_e1/1785123878/`

完整 9,508-row receipt：

| 指标 | fixed default | V4 gated | fix / break | delta |
| --- | ---: | ---: | ---: | ---: |
| REC @0.25 | 5514/9508 (57.9933%) | 5516/9508 (58.0143%) | 4 / 2 | +2 hits |
| REC @0.50 | 4408/9508 (46.3610%) | 4411/9508 (46.3925%) | 4 / 1 | +3 hits |
| Mask @0.25 | - | 5677/9508 (59.7076%) | - | 保护 |
| Mask @0.50 | - | 4662/9508 (49.0324%) | - | 保护 |
| Mask mIoU | - | 41.7754% | - | 保护 |

validation switch ratio 仅 `0.16%`。V4 修复了 V3 的过度覆盖，并在两个 REC
阈值取得严格正增益，但增益只有 2/3 hits，独立 gate-only 仍不足以达到
`0.59/0.49`。正式 checkpoint size `604018862`，SHA-256：
`aaeb4b9ca091e5393f55f29c91b618207e4dda6d231a706576fc73fed9eb5022`。
13 个 gate tensor 与 optimizer state 全 finite；box/mask/decision head 和 encoder
均有更新；与 router anchor 共有的 1,158 个 tensor 全部逐元素不变。

按用户要求，下一阶段不调 validation margin，而是从该 V4 权重进行模型级联合
训练到真实总轮次 80。2026-07-27 曾生成只把 checkpoint epoch 元数据重标为 71
的临时启动副本，并把联合训练输出放到：

`/dev/shm/source_moe_joint_full_v4/scanrefer/ssq_moe_v4_joint_e72_e80/1785128173/`

该进程在 dataset loading 阶段被环境中断，**没有完成 epoch 72**，因此没有联合
checkpoint 或 validation receipt；随后 `/dev/shm` 被清空。此目录和旧“已启动”
记录均不能作为完成实验的证据。

2026-08-01 已改为直接从原始 V4 formal checkpoint 启动，不再依赖修改 metadata
的副本。`--checkpoint_start_epoch 72` 显式覆盖非数字 `epoch="last"`；首次 joint
使用 fresh optimizer，后续 `JOINT_RESUME=1` 则从数值 epoch checkpoint 精确恢复
model、optimizer 和 scheduler，并自动进入下一轮。新训练固定写入持久盘：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_full_v4_persistent/`

首次实际启动的 run：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_full_v4_persistent/scannet,scanrefer/ssq_moe_v4_joint_e72_e80/1785523835/`

启动时 config 已确认 `start_epoch=72`、`max_epoch=80`、`joint_det=true` 和完整
五指标 retention。数据集为 48,655 train / 9,508 validation；前 400 个训练 batch
的所有 loss 与 gradient finite，batch-100 total loss 为 `9.8007`。

但 2026-08-01 的训练中审计发现该 run 的监督契约无效，已在 epoch 72 的
`497/4054` 主动停止，未保存 checkpoint、未执行 validation：

- `Joint3DDataset` 原先把所有 joint batch 样本的 `language_dataset` 固定成
  `test_dataset=scanrefer`，没有暴露 annotation 自身的 `dataset`；
- MoE loss 用 `language_dataset != scannet` 构造 sample mask，因此约四分之一的
  ScanNet detection prompt 也被当成单目标 referring 样本；
- ScanNet prompt 对应多个 GT，而 MoE ranking 固定只监督 slot-0，等价于强迫
  router 学习任意第一个物体，会污染 query reranking 和 fallback gate。

修复后保留 `language_dataset` 供旧 ScanRefer/Sr3D loss 配方使用，新增逐样本
`sample_dataset=anno['dataset']`。SourceMoE rank、gate 和 balance loss 现在严格只
消费非 ScanNet grounding 样本；缺少该 metadata 时 fail-closed，并记录
`source_moe_supervised_sample_ratio` 供真实训练验收。实现 SHA-256：

- `src/joint_det_dataset.py`：`924e609be2e893c2b39e79865af61e02da049bc52c84b5c1857634c97eea53d2`
- `models/losses.py`：`5db5a5dc65235b20f6441db8e1922204ce61ce13c4ab4efb6c0884d5a6057ce6`
- 聚焦测试：`a2112ff303afeb31b8e1a362b3c3ba247006344460f8c8c829d2ac6a6a76e39b`

修复后的定向回归为 `115 passed`，完整回归为
`2835 passed, 3 warnings in 136.55s`。下一次 epoch 72-80 必须从原始 V4 formal
checkpoint 全新启动，不能从上述无效目录恢复；首批验收要求 supervised ratio
明显低于 1 且接近真实 grounding 样本占比。

修复后的新 run 已从同一 V4 formal checkpoint 启动：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_full_v4_samplemask/scannet,scanrefer/ssq_moe_v4_joint_e72_e80_samplemask/1785525545/`

tmux 会话为 `mcln_v4_joint_72_80_samplemask`。真实数据构成为 ScanRefer 36,665
条、ScanNet `(1,201-2)*10=11,990` 条，所以全 epoch 理论 supervised ratio 为
`36665/48655=0.753571`。batch 25/50/100/125 的累计实测分别为
`0.7100/0.7167/0.7367/0.7453`，已向理论值收敛；所有 loss/gradient finite。
batch-125 MoE rank loss `1.2179`、gate loss `1.2474`，相较无效 run 的早期
rank loss 约 `1.56` 明显下降。

随后对 scheduler 做了第二次启动审计：joint 使用 `--reduce_lr` 建立 fresh optimizer
和 fresh `MultiStepLR`，而 scheduler 每 batch 从内部 step 0 开始。原命令仍传全局
milestone `[50,75]`，所以在仅 9 个 epoch 的续训中永远不会衰减；与“总轮次 76
开始第二次 0.1x LR”不一致。该诊断 run 在 epoch 72 约 `353/4054` 停止，同样
没有 checkpoint 或 validation，不可恢复。

launcher 现支持空格分隔的 `LR_DECAY_EPOCHS` 并校验每项为非负整数。当前 PyTorch
同配置模拟确认 `LR_DECAY_EPOCHS=3` 会在完成 epoch 75 最后一个 batch 后衰减，
因此 epoch 76 首 batch 起使用 decoder/backbone/text/MoE 的 0.1x LR。实现：

- `scripts/train_scanrefer_source_moe.sh`：`3aae9ca9db549bae71e79372638cda8c5588204728a45c621d24300a062d19d1`
- `tests/test_source_moe_integration.py`：`6c35bbe60e72e747a807bee10625d064f26c34653573f2d8d4ab0f7ce5fbdefe`
- 聚焦验证：`52 passed, 2 warnings`，非法 milestone 退出码 2；
- 完整回归：`2836 passed, 3 warnings in 147.20s`。

最终有效 run 必须同时满足 sample-mask 修复和 `LR_DECAY_EPOCHS=3`，并从原始 V4
formal checkpoint 全新启动；上述两个中止目录只保留诊断日志。

最终 run 已实际启动：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_full_v4_valid/scannet,scanrefer/ssq_moe_v4_joint_e72_e80_samplemask_sched3/1785527065/`

tmux 为 `mcln_v4_joint_72_80_valid`。实际 `config.json` 已确认
`start_epoch=72`、`max_epoch=80`、`lr_decay_epochs=[3]`、joint ScanNet+ScanRefer、
V4 calibrated-utility gate 和五指标 retention。数据初始化于 `03:52:43` 完成，
训练/验证样本数严格为 `48,655/9,508`；V4 formal checkpoint 成功加载并明确打印
`first requested epoch is 72`。epoch 72 的首个 `25/4054` 验收点中，累计
`source_moe_supervised_sample_ratio=0.7100`，随后 batch 50/75/100 为
`0.7167/0.7467/0.7367`，与理论 `0.753571` 的早期采样波动一致。batch 100 的
total/rank/gate loss 为 `9.4031/1.2057/1.2595`；全部 loss 与 optimizer gradient
均通过逐 batch finite 检查。A100 实际显存约 `26.34 GiB`。

epoch 72 的 batch 550 中期观察（累计训练统计，不等同 validation）：supervised ratio
为 `0.7483`；pre-gate candidate 的 fix-break 净值为
`(0.0344-0.0195)=+0.0149`@0.25、`(0.0483-0.0344)=+0.0139`@0.50，说明
当前 query/reranker 在训练分布上仍有可用候选收益。但 learned gate 的 fix/break 均约
`0.0009`，switch ratio `0.0033`，而 oracle switch ratio `0.2482`、decision target fix
ratio `0.1097`，表现为明显的早期欠切换。当前仅完成本 epoch 约 13.6%，不据此中止或
修改超参；以 epoch-72 完整 validation 的 learned/default、candidate-set oracle 和 mask
五指标作为是否启用 contextual gate 的依据。

启用 `--checkpoint_metric_retention` 后，每个已完成 epoch 原子写盘；始终保留
latest，以及 learned REC@0.25、REC@0.50、mask@0.25、mask@0.50、mask mIoU
各自最佳 epoch。`last` 与五个 `best` 名称都是同盘硬链接，不复制 checkpoint；
不再是 latest 或任一最佳的普通 epoch 权重会自动删除。实现 SHA-256：

- `main_utils.py`：`8bacefee31e3b375225b0976add837bf380efe5845bd9b80c5eb0f6a93f49973`
- `scripts/train_scanrefer_source_moe.sh`：`25890bb18e41f2c562b96923d0b390572cd84ff1ddda87991d32648c75864d67`
- 聚焦测试：`f810cea8eb9a69ba99b199b2dfcd846380aa0e8a84796f21975aa6d5e862af34`

完整回归为 `2833 passed, 3 warnings in 140.59s`。

同日按明确拒绝清单删除 15 个 smoke/V1-V3/无锚点 router 重复或低指标
checkpoint，共 `9,065,199,334` bytes（约 8.44 GiB）；日志、配置和 receipt 全部
保留。历史 `0.58288/0.48601` 的 backbone + parent reranker + geometry reranker
三件套、router anchor 和 V4 formal checkpoint 均未删除。

## 7. 局限与下一步

1. 从原始 V4 formal checkpoint 完成修复后的真实总轮次 72-80 联合训练；每个 validation 点同时记录 learned/fixed
   REC、mask@0.25、mask@0.50 和 mIoU，不用单个 REC 数字挑 checkpoint。
2. 若联合训练中 gated query 改善 REC 但 mask 不升，先降低 mask utility/loss 权重或分析同 box-tier
   的 mask oracle，而不是放宽 box-tier 保护。
3. 若 router-only 无 headroom，下一项结构改动应是 query-specific mask fusion：
   从 query/text/mask statistics 预测 `[B,Q]` alpha，替代 sample 标量。
4. ScanRefer 配置冻结后，再在 Nr3D/Sr3D 训练各自 router；不重新枚举 0.05/0.10
   源组合。跨数据集结果才是“自适应多源具有泛化性”的证据。
5. 单阶段 ScanRefer 复用同一 SourceMoE 和 listwise target，但需单独验证 source
   adapter 的 key/shape，不能直接拿双阶段 checkpoint 宣称兼容。
6. 若模型级门禁通过且仍需后处理，新增 `rec-query-v2`：候选并集加入 learned
   top-k，特征加入 learned raw score/rank/top-margin；随后重训 parent/geometry。

### 候选集上下文 Gate（下一项模型级实验，等待本轮完整结果）

对当前 `QueryFallbackGate` 的代码审计表明，它把每个候选独立编码为
`[candidate, default, candidate-default, six score features]`，再以逐 query MLP 输出
override margin。`QueryContextReranker` 虽然已做全 query attention，但 gate 本身看不到
top-k 候选之间的相对竞争、候选间的几何/语义分散度，也看不到“多个同样安全的 fix”这一
集合信息。更重要的是，现有 row-level selection target 对所有正 utility candidate 使用
单一 `argmax`；阈值式 utility 常出现并列，因此该硬标签会任意偏向一个 query。

若 epoch 72-80 的模型级 learned REC 仍未达到 `0.59/0.49`，下一项不做 margin、源组合
或验证阈值 sweep，而实现 `CandidateSetContextualFallbackGate`：

1. 从 MoE score 中取 default 加最多 K 个非 default 候选，保持 action space 为
   `{fallback, alternatives}`，并显式记录实际 alternative count；
2. 仅在这至多 K+1 个 token 上运行轻量 self-attention，token 特征复用 decoder evidence、
   三源 raw/rank score、相对 default 的 box 几何和 score gap；因此成本与 Q=256 无关；
3. context residual scale 与最终 action head 均零初始化，初始化时严格退化为 shared/default，
   保留当前安全锚点；
4. 用当前 calibrated box+mask utility 构造 fallback+候选的 **soft setwise target**：
   fallback utility 固定 0，多个正 utility candidate 按同一温度分配概率，invalid candidate
   概率严格为 0。以 KL/soft CE 替代任意单一 argmax 的 row label，同时保留现有 break cost、
   per-candidate quality heads 和 over-estimation penalty；
5. 新测试必须覆盖：零初始化 exact identity、候选排列等变性、并列 fix 的等概率 target、
   default 不进入 alternatives、ScanNet sample-mask fail-closed 和 finite gradient。

这是 query reranking/MoE 的集合决策扩展，source score 与 IoU 阈值定义均不依赖某个数据集；
冻结 ScanRefer 后按同一实现迁移到单阶段 ScanRefer、Nr3D 和 Sr3D。

为避免把“三个源各自 top-1 的 oracle”误当成 gate top-k action space 的上限，
`GroundingEvaluator` 已新增 `gate_candidate_oracle` Acc/mIoU 和
`gate_oracle_headroom`。它在每个样本上对 `{default query} union
moe_gate_candidate_mask` 的真实 box IoU 取最大值，因此可直接区分：

- gate candidate-set 已有足够 oracle headroom，但 learned decision 未利用；
- gate candidate-set 自身不足，必须改善 query proposal/reranker 或扩大有效候选。

该补丁只增加验证计数，不改变 forward、loss、checkpoint 或 retention receipt：

- `src/grounding_evaluator.py` SHA-256：
  `9d0006af6cd30a90d660f60906ea4d30cdd1182e6aa354b30299ce2292cd0109`
- `tests/test_grounding_evaluator_source_choice.py` SHA-256：
  `222fa0810aa51fa544520a84cc95f7e6ffb8595cbd7e03bd19f85c36b4d7e825`
- 聚焦回归：`96 passed`；完整回归：
  `2838 passed, 3 warnings in 158.88s`。

当前正式训练进程在该补丁前已加载 evaluator，所以其内置 epoch-72 validation 不会输出
新字段；checkpoint 产生后需用当前代码做一次独立 eval，训练数值和五指标 receipt 不受影响。

### 候选集上下文 Gate 实现与续训合同（2026-08-01）

上述候选集方案已作为默认关闭的实现落地。`context_layers=0` 时参数集合和旧 action
space 完全不变；开启后 K 明确定义为 K 个非 default alternative，并只在
`default + alternatives` 上做轻量 self-attention。context residual scale 为零初始化，
因此新 gate 从 router anchor 初始化时严格退回 default。soft setwise target 只给正 utility
候选分配概率，invalid、negative 和 zero utility candidate 概率为零；temperature 为 0 时
保留原 hard-label 路径。

新增边界验收覆盖单 query（实际 alternative count 为 0）、非法 context heads/layers/dropout、
context 必须依赖 fallback gate、候选排列等变、集合外 query 隔离、attention finite gradient、
tie-aware target 和 checkpoint context mismatch。另在真正恢复 joint optimizer/scheduler 前，
`main_utils.py` 会逐项核对 SourceMoE 候选行为和 rank/anchor/gate/setwise loss 配置；因此
`JOINT_RESUME=1` 不再可能把 `calibrated_utility + soft setwise` 静默续成默认 hard target。
有意改变这些配置时必须走 fresh optimizer（`--reduce_lr` / `JOINT_RESUME=0`）。

独立 eval 也会在模型构建前从 checkpoint 自动恢复 source 列表、top-k、query reranker 和
fallback gate 的所有非 tensor inference 配置；这已用真实 V4 checkpoint 做 CPU 审计。
新 evaluator 将 gate candidate-set oracle 与 headroom 原子写入独立的
`source_choice_diagnostics_epoch_<N>.json`，不向五指标 `mcln-retrain-metrics-v1` 增加字段，
因此 retention 和既有调参工具不受影响。

`audit_source_moe_candidate_oracle.py` 对两个 receipt 做完整计数/均值一致性检查，并把
REC `0.59/0.49` 与 V4 的三项 mask 基线组成五项联合门禁。结论只有四种：REC 与 mask
共同通过为 `learned_target_reached`；REC 达标但 mask 退化为 `repair_mask_tradeoff`；
learned 未达标但 candidate oracle 达标为 `train_contextual_gate`；oracle 也不足则为
`improve_candidate_generation`。

实现 SHA-256：

- `models/source_moe.py`：`bdc2162ae439bfefba3bdfa01428c8d828c30a6d699578ad2d269fa264944d95`
- `models/losses.py`：`95ddbda88a6781a9e5fedd8f048d20531c551d18377032defc10d4a291f8db64`
- `models/mcln.py`：`f6327771e43d5b8c87c84d56090633fe130ff48a4d617dd87cd09eb686d0a48e`
- `main_utils.py`：`de86ff9b6ca0902ce355a6d3bac5ce5c792bc0a19a8806e495988496a3921b34`
- `train_dist_mod.py`：`cfe8d7af5d23d3122c738ac1f869a8d34b5fea51d1fc3aa6750850feb34ad6c6`
- `src/grounding_evaluator.py`：`440d7b32f129b25536a84d0ee651dc56187c3ba8c58adf1d2cba717264ce05ae`
- launcher：`99573449e0e030ab1c5747e24f614d232fff09338d6b6a3f8545ee1dfb90f9cf`
- eval launcher：`c37aea1a52f3d0e41e1c06c6d37e90ecd18c56c019b7eecad3aa394e9593f2c3`
- oracle audit：`593c2d4c95f25706470e3e30a86d7ae6a370e8d27fab3ea567e77e0a30321fd9`
- unit/integration tests：`b492a0d338f2173531ff09113f4d2edbbf6876c0351b52c98902b323e9a66358` / `520480cbd8d37c93abbb9b832a4f6f67a0a6578df0d7f5b5673dbeb6b1108eba`
- evaluator/receipt tests：`88cb1a355148305e185195936fe618e81450af9b055acf7ce31840e1f57c8fc7` / `1bc40da937b8079558a8ebcc70620013f86711b832ba5da0d16b2daac1f3bdd2`
- oracle audit tests：`42cb0804e0e804b2d29849296c9eb8a0a2d0680c6161e359d9ff5c74e73717b4`

验证结果：相关聚焦测试全部通过；完整回归
`2870 passed, 3 warnings in 161.72s`。当前 epoch 72-80 进程启动早于这些修改，
不会加载 contextual gate，也无需重启。

epoch 72 checkpoint 生成后、且正式训练释放 GPU 后，独立复核命令为：

```bash
CHECKPOINT_PATH=/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_full_v4_valid/scannet,scanrefer/ssq_moe_v4_joint_e72_e80_samplemask_sched3/1785527065/ckpt_epoch_72.pth \
EVAL_EPOCH=72 \
LOG_DIR=/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_v4_oracle_eval \
bash scripts/eval_scanrefer_source_moe_checkpoint.sh
```

launcher 默认检查 `nvidia-smi`；存在任何 compute process 时退出码为 3，不会与正式训练
并发。成功后自动定位本次时间戳目录、校验两个 receipt，并生成
`source_moe_oracle_audit.json`。`eval_metrics_epoch_72.json` 与
`source_choice_diagnostics_epoch_72.json` 的 `sample_count` 都必须为 `9508`。

只有当独立评估证明 gate candidate-set oracle 有足够 headroom、而 learned gate 未利用时，
才从无 fallback gate 的 router anchor 启动 V5，避免不兼容地继承 V4 gate head：

```bash
PHASE=gate \
CHECKPOINT_PATH=/root/autodl-tmp/DATA_ROOT/output/source_moe_formal/scanrefer/ssq_moe_router_anchor_w2_t02_d010_e1/1785092585/ckpt_epoch_1.pth \
LOG_DIR=/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_contextual_v5 \
EXP=ssq_moe_context_top8_setwise_t025_e2 \
START_EPOCH=1 MAX_EPOCH=2 \
SOURCE_MOE_USE_FALLBACK_GATE=1 \
SOURCE_MOE_GATE_USE_EVIDENCE_FEATURES=1 \
SOURCE_MOE_GATE_CONTEXT_LAYERS=1 \
SOURCE_MOE_GATE_CONTEXT_HEADS=4 \
SOURCE_MOE_GATE_CONTEXT_DROPOUT=0.1 \
SOURCE_MOE_GATE_OBJECTIVE=calibrated_utility \
SOURCE_MOE_GATE_SETWISE_TEMPERATURE=0.25 \
SOURCE_MOE_GATE_LOSS_WEIGHT_GATE=1.0 \
bash scripts/train_scanrefer_source_moe.sh
```

V5 gate-only 完整验证优于 V4 default/learned 且 mask 不退化后，才用同一 context/setwise
配置进入 `JOINT_RESUME=0` 的真实 epoch 72-80 联合训练。当前 GPU 被正式 V4 run 占用，
禁止并发启动该命令。

## 8. 保护资产

- Backbone（0444）：`/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth`
- SHA-256：`3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`
- Parent reranker：`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/artifacts/reranker_h256_d010_lr1e3_seed0_final_contract.pth`
- SHA-256：`f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b`
- Geometry reranker：`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_artifacts/selected_geometry_reranker.pth`
- SHA-256：`835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f`
- 三份 `0.582878/0.486012` 基线资产权限均为 `0444`；不会进入当前 run 的
  checkpoint retention 或低指标清理范围。
- V4 formal 原记录路径 `output/source_moe_gate_formal_v4/scanrefer/ssq_moe_top8_calibrated_utility_e1/1785123878/ckpt_epoch_last.pth`
  已经当前磁盘审计确认不存在；对应独立评测收据仍在
  `output/source_moe_v4_contract_eval/`，该 checkpoint 不再视为可恢复保护资产。
- Router anchor（0444）：`/root/autodl-tmp/DATA_ROOT/output/source_moe_formal/scanrefer/ssq_moe_router_anchor_w2_t02_d010_e1/1785092585/ckpt_epoch_1.pth`，SHA-256
  `c73c79529537e106f3d85096fafcf8a8ae2996222bcc49210ed8ae788a372788`。
- 四个保护资产（主 backbone、两个 reranker、router anchor）在
  `/root/autodl-tmp/DATA_ROOT/protected_mcln_artifacts/` 另有 `0444` 硬链接，目录权限
  为 `0555`；训练目录清理不得扫描该目录。

## 9. 公开方法来源

- shared/routed expert separation：DeepSeekMoE；
- top-k routing 与 auxiliary balance：Switch Transformer / Mixtral；
- listwise distribution matching：ListNet/RankDistil 类方法；
- 本仓库实现未引入新第三方依赖，按现有 MCLN tensor contract 重写。

## 10. V4 oracle 结论与 V5 启动依据（2026-08-01）

当前代码对只读 V4 checkpoint 的独立 9,508-row 评测已完成，恢复的 inference contract
与原 checkpoint 一致。learned REC 为 `0.580143/0.464030`，mask 为
`0.596761/0.490324/0.417768`；default + top-8 action space 的真实 oracle 达到
`0.630206/0.549642`，mIoU `0.451155`。自动 auditor 返回
`train_contextual_gate`，因此 V5 是实证选择，而不是继续扩大 source 或做数据集阈值 sweep。

此前 epoch 72-80 joint run 在首轮验证得到 `0.000526/0.000105` REC，fixed default 同样
崩塌；V4 对照正常排除了 evaluator 故障。该 run 已停止，坏 checkpoint 硬链接已删除，
日志和收据保留。V5 必须使用 gate-only optimizer，使 trainable allowlist 严格等于
`source_moe.fallback_gate.*`，先保护 backbone、decoder 和 query reranker 的原有能力。

V5 固定配置为 `context_layers=1`、`heads=4`、`top_k=8`、evidence features、
`calibrated_utility` 和 `setwise_temperature=0.25`，从无 fallback gate 的只读 router anchor
初始化。先完成两轮 ScanRefer gate-only 训练和每轮 9,508-row 验证；只有 learned REC、
三项 mask guard 和 gate 决策诊断通过后，才允许用 fresh optimizer 进入 epoch 72-80。

正式 V5 run 位于
`/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_contextual_v5/scanrefer/ssq_moe_context_top8_setwise_t025_e2/1785535922/`，
tmux 为 `mcln_v5_gate_contextual_e1_e2`。启动审计确认只有
`source_moe.fallback_gate.*` 的 `316,432` 个参数可训练；基础模型保持 eval mode，因而
gate-only checkpoint 不会携带新的 backbone BN 统计漂移。

V5 前 2,000 batch 的 oracle switch 约为 `17%`，但 deployed switch 仍为零，context
scale 已从零增长到 `0.0479`。为区分“上下文容量不足”和“calibrated loss 类别先验过于
保守”，实现了默认关闭的第三种 objective：`balanced_calibrated_utility`。它组合：

- `balanced_focal` 的 batch-adaptive inverse-frequency decision class weighting；
- row-level fallback/switch inverse-frequency weighting，并对 fallback 保留 break cost；
- `calibrated_utility` 的 setwise soft target 与 asymmetric utility regression。

原两种 objective 的分支和默认行为保持不变；checkpoint config/resume contract 将新字符串
视为独立训练合同。聚焦测试 `83 passed`，包含稀有 switch row 相对 fixed-cost calibrated
objective 获得更高 selection loss 权重的回归测试。V5 完整结果正常时不用该目标；只有
两轮验证仍 under-switch 时才作为 V6 使用。

V5 epoch 1 完整验证得到 learned REC `0.580248/0.463820`、mask
`0.596550/0.490114/0.417634`，switch 仅 `0.09%`；同一 action space oracle 为
`0.633361/0.552798/0.454365`。因此 contextual encoder 没有破坏默认能力，但 fixed-cost
calibrated objective 仍未把候选上限转化为决策。epoch 2 继续作为 additional-training
control；若 switch 和 REC 仍无实质增长，V6 使用上面的 balanced calibrated objective，
不再改变 top-k、source 集合或数据集阈值。

## 11. V5 最终结论与 V6 balanced gate（2026-08-01）

V5 epoch 2 的 9,508-row learned REC 为 `0.580143/0.463715`，fixed default 为
`0.579933/0.463504`，mask 为 `0.596550/0.490219/0.417589`。candidate oracle 保持
`0.633361/0.552798/0.454365`，但部署 switch 仍只有约 `0.13%`；两轮结果证明候选生成
不是瓶颈，固定代价 calibrated gate 的稀有正 switch 学习不足。三个 V5 checkpoint 已移到
`quarantine_low_quality/source_moe_v5_contextual_e2_1785535922/`，日志和全部收据留在原目录。

V6 使用 `balanced_calibrated_utility` 从无 fallback gate 的只读 router anchor 重新初始化，
而不是继承 V5 的保守 decision head。正式目录为：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_balanced_v6/scanrefer/ssq_moe_context_top8_setwise_t025_balanced_calibrated_e2/1785544218/`

启动验收确认 ScanRefer train/val 为 `36,665/9,508`，只有
`source_moe.fallback_gate.*` 的 `316,432` 个参数可训练；主干保持 eval，anchor 中的
`query_max_delta=0.10` 正确覆盖启动器默认值。V6 启用了五指标 retention，因此每轮先保存
可恢复 checkpoint，再按 REC@0.25、REC@0.50、mask@0.25、mask@0.50 和 mask mIoU 独立
保留最佳权重。

batch 500/1000 的 gate loss 为 `1.2557/1.2057`，context residual scale 为
`0.0258/0.0409`，oracle switch 为 `17.22%/16.83%`，部署 switch 为
`0.07%/0.03%`，mean max margin 为 `-0.5667/-0.5749`。balanced objective 相对 V5
已经显著减小负 margin，但截至 batch 1000 尚未形成真实有益 switch recall。必须以 epoch 1
的 9,508-row 收据和 candidate-oracle auditor 为准；中途 train-batch 指标不能用于宣称改进。

后续长训禁止恢复旧的 full joint optimizer。只有 V6 完整验证优于 V4/V5 且 mask 守门不
退化时，才从 V6 retained checkpoint 以 fresh optimizer 进入总轮次 72-80 的
`source_moe_train_only`：主干继续 eval，只训练 `source_moe.*`，并使用相对 scheduler
milestone，避免全局 `[50,75]` 在 fresh scheduler 中永不触发。

独立评测启动器默认 baseline 已绑定现存的 V4 合同评测收据
`source_moe_v4_contract_eval/scannet,scanrefer/ssq_moe_v4_contract_eval/1785534811/eval_metrics_epoch_1.json`，
不再引用已经删除的 V4 formal 目录。对应 shell 语法、真实 receipt audit 和集成测试均通过。

## 12. V6 epoch 1 结论与清理收据（2026-08-01）

V6 epoch 1 的 9,508-row learned REC 为 `0.578881/0.462663`，fixed default 为
`0.579933/0.463610`，mask 为 `0.595499/0.488851/0.416660`。对应 candidate-set oracle
为 `0.633572/0.552798/0.454344`；oracle action space 足够跨过目标，但 learned gate
未转化该上界。部署 switch 约 `1.24%`，验证统计中的 oracle switch 约 `11.92%`，且
selector fix/break 在 Acc@0.25 上为 `0.200pp/0.305pp`、Acc@0.50 上为
`0.168pp/0.263pp`，净效应为负。

相对 V4 合同基线，五项变化均为负，auditor 返回 `train_contextual_gate`；因此 epoch 2
只作为 balanced objective 的完整训练时长对照，不把 epoch 1 checkpoint 直接放行到
72-80 长训。checkpoint tensor 审计同时证明 1,158 个非 gate tensor 与 router anchor
完全一致，退化不是未加载预训练或主干漂移。

V6 epoch 1 收据确认后，隔离区内三个 V5 低质量 checkpoint 已永久删除，约释放
`1.69 GiB`。V5 日志和结构化收据保留；历史 `0.582878/0.486012` 三件套与 router
anchor 在专用保护目录中的额外硬链接仍为 `0444`、link count `2`，不会进入 retention
或清理逻辑。

用户明确要求把 70-80 轮完整训练作为时长对照，因此 V6 epoch 2 正式收据落盘后仍进入
V7 epoch 72-80，但只放宽性能前置条件，不放宽参数安全条件。V7 从 V6 两轮
`REC@0.25` 最佳硬链接初始化，使用 fresh optimizer，只训练 `source_moe.*` 并保持整个
MCLN 主干 eval；采用 router 阶段已验证的 rank/anchor/temperature
`1.0/2.0/0.2`，gate loss 继续为 `1.0`。相对 LR milestone `3` 在 epoch 76 起使用
`0.1x` LR；五指标 retention 限制磁盘只保留 latest 和各指标最佳。该实验用于区分
“gate-only 两轮不足”与“完整 SourceMoE 协同训练能否利用 oracle headroom”，不能把中途
train batch 指标当作 ScanRefer 正式结果。

V6 epoch 2 验证收据为 learned REC `0.580143/0.463715`、fixed
`0.579933/0.463610`、mask `0.596761/0.490219/0.417715`；candidate oracle
`0.633572/0.552798/0.454344`。五项 retention 最佳均为 epoch 2，V7 从该
`ckpt_best_rec_acc025.pth` 启动。V7 目录为
`output/source_moe_continue_v7/scanrefer/ssq_moe_context_balanced_source_only_e72_e80/1785552345/`，
启动日志已确认 checkpoint 非张量合同（尤其 `query_max_delta=0.1`、context/evidence 和
balanced objective）恢复成功，`start_epoch=72`、`max_epoch=80`、`lr_decay_epochs=[3]`。

V7 初始化完成后的运行时验收：ScanRefer train/val 为 `36,665/9,508`，V6 retained
checkpoint 显式报告加载 epoch 2；`source_moe_train_only` allowlist 共 `1,082,004` 个
参数，覆盖 router、query reranker 和 fallback gate，MCLN 主干不在 optimizer。epoch 72
首批已运行，GPU 占用约 `8.2 GB`。

Epoch 72 batch 500 的 total/rank/anchor/gate loss 为
`12.0598/0.9383/0.0307/1.1093`，均 finite；routed/context scale 已到
`0.5255/0.0991`。部署 switch `0.52%`，oracle switch `16.83%`，表明 query reranker、
router 和 contextual gate 均参与训练，但是否形成净 REC 增益仍只按 9,508-row validation
判断。

## 13. V7 epoch 72 正式审计（2026-08-01）

V7 的 source-MoE-only 续训第 72 轮已完成完整 ScanRefer 验证（`sample_count=9508`）：

- learned REC：`0.5806689/0.4643458`；fixed default：`0.5799327/0.4637148`；
- Mask：`0.5970761/0.4907446/0.4178385`；
- candidate oracle：`0.6349390/0.5553218`，mIoU `0.4557727`。

本轮相对 V6 epoch 2 的 REC 增益只有 `+0.0005259/+0.0003155`，但 Mask 三项均小幅
提升，故 `mask_guard_pass=true` 而 `learned_target_pass=false`。候选 oracle 已超过
`0.59/0.49`，learned gate 仍未部署这部分 headroom；审计结论为
`train_contextual_gate`。该结果支持继续完成 73-80 轮，也提示后续重点应放在 gate 的
正 utility 校准和 query reranker 的部署选择，而不是继续盲目增加候选源。

epoch 72 的 gate candidate oracle 相对 learned headroom 为
`+0.055006/+0.091607`，learned gate 的实际 switch 约 `0.12%`，decision target 中
neutral 占多数。这与 utility regression 对大量 neutral 候选平均、并对 overestimate
加权的保守倾向一致，但需等待 epoch 76 后低学习率阶段的正式收据再决定是否修改损失。
当前不改变运行中的实验合同。

## 14. Expected-utility action ablation（2026-08-01）

epoch 72 诊断显示 fallback gate 的 `box_head`/`mask_head` 已按两个 REC 阈值获得监督，
但 legacy 推理只使用独立 `decision_head` 的 override margin。因此增加了可选
`source_moe_gate_action_mode=expected_utility`：将 box/mask 三分类概率转换为
`break=-break_cost, neutral=0, fix=+1` 的阈值加权期望收益，再用同一 candidate set
执行 fallback/override。该路径不增加 checkpoint tensor，默认仍为 `decision`，旧权重和旧
评估合同完全兼容；显式设置后可对同一 retained checkpoint 做网络内 action ablation。

实现位置为 `models/source_moe.py` 的 `transition_logits_expected_utility` 和
`QueryFallbackGate`，训练损失会在 action mode 下用该 action margin 计算 setwise 选择
信号，避免 box/mask 监督头只训练不用。CLI/续训会从 checkpoint 继承 action mode，显式
参数可用于新消融；`2881 passed, 3 warnings` 全套测试通过。当前运行中的 V7 仍使用
legacy `decision`，待 72-80 轮完成后再在 retained epoch 上比较，不把它混入现有训练曲线。

V7 epoch 73 全量验证得到 learned REC `0.5798275/0.4648717`，Mask
`0.5972865/0.4908498/0.4181844`；candidate oracle 为
`0.6296803/0.5500631/0.4517465`。相对 epoch 72 呈现 REC@0.25/REC@0.50
`-0.0008414/+0.0005259` 的阈值权衡，Mask 三项均小幅提升。验证 gate 的 predicted switch
为 `142/9508=1.493%`，其中有益 `43`、有害 `99`，precision `30.28%`、oracle switch
recall `3.90%`；比 epoch 72 的 `31/84` 有进步，但净部署仍不足以接近目标。retention
保留 epoch 72 REC@0.25 和 epoch 73 其余四项，继续完成 epoch 74-80。

V7 epoch 74 全量验证得到 learned REC `0.5801430/0.4639251`、Mask
`0.5964451/0.4902188/0.4174829`，candidate oracle
`0.6314682/0.5522718/0.4532405`。gate 预测切换 `94` 次，其中有益 `24`、有害 `70`；
虽然 oracle query match 升到 `44.67%`，部署净收益仍接近零。该轮五项均未刷新 retained
best，Mask guard 失败，继续 epoch 75 完成最后一轮高学习率对照。

V7 epoch 75 全量验证得到 learned REC `0.5799327/0.4637148`、Mask
`0.5963399/0.4896929/0.4172467`，candidate oracle
`0.6339924/0.5546908/0.4550207`。gate 预测切换 `120` 次，其中有益 `36`、有害
`84`，precision `30.00%`、oracle switch recall `3.16%`；仍是低 recall 且错误切换多于
正确切换。五项 retained best 均未变化。epoch 72-75 的 `3e-4` 阶段没有把 oracle
headroom 转成部署增益，故 epoch 76-80 只作为既定 `3e-5` 低学习率收敛对照；完整训练
结束后的首要网络内消融仍是同一 checkpoint 上的 `expected_utility` action mode。

V7 epoch 76 的低学习率验证为 learned REC `0.5801430/0.4635044`、Mask
`0.5960244/0.4896929/0.4170740`；candidate oracle 为
`0.6255785/0.5466975/0.4492696`。gate 预测切换 `102` 次，有益 `31`、有害 `71`，
precision `30.39%`、oracle switch recall `2.93%`。该轮仍未刷新 retained best，说明
仅将 LR 降到 `3e-5` 尚未改变部署选择瓶颈；继续收集 epoch 77-80 的完整收敛证据。

V7 epoch 77 的低学习率验证为 learned REC `0.5799327/0.4635044`、Mask
`0.5961296/0.4899032/0.4171509`；candidate oracle 为
`0.6240008/0.5456458/0.4482024`。gate 预测切换 `116` 次，有益 `33`、有害 `83`，
precision 降至 `28.45%`，oracle switch recall 仅 `3.18%`。该轮仍未刷新任何 retained
best，进一步支持将后续优化放在 action calibration，而不是单纯延长相同 objective 的训练。

V7 epoch 78 的低学习率验证为 learned REC `0.5795120/0.4631889`、Mask
`0.5956037/0.4893774/0.4167333`；candidate oracle 为
`0.6242112/0.5451199/0.4481257`。gate 预测切换 `145` 次，有益 `37`、有害 `108`，
precision `25.52%`、oracle switch recall `3.54%`。部署比例增加并未转成收益，反而使
错误切换增多；epoch 79-80 仍按既定合同完成，随后在 retained checkpoint 上进行
`expected_utility` 与 legacy action 的网络内对比。

V7 epoch 79 的低学习率验证回升至 learned REC `0.5803534/0.4640303`、Mask
`0.5966554/0.4903239/0.4174338`，但没有刷新 retained best。candidate oracle 为
`0.6227387/0.5451199/0.4473105`；gate 预测切换 `106` 次，有益 `32`、有害 `74`，
precision `30.19%`、oracle switch recall `3.09%`。完成 epoch 80 后用 epoch 72/73
retained checkpoint 做 action-mode 对照，不将 epoch 79 的短期回升误判为收敛突破。

V7 epoch 80 最终验证为 learned REC `0.5803534/0.4642406`、Mask
`0.5965503/0.4903239/0.4175141`；candidate oracle 为
`0.6224232/0.5442785/0.4470129`。gate 预测切换 `101` 次，有益 `30`、有害 `71`，
precision `29.70%`、oracle switch recall `2.93%`。epoch 72-80 的完整曲线证明当前
balanced calibrated utility objective 的瓶颈不是训练时长，而是 action calibration：
候选 oracle 每轮均通过目标，learned action 每轮都只召回约 2-4% 的 oracle switch，且
错误切换持续多于正确切换。最终 retained best 仍为 epoch 72 REC@0.25、epoch 73 其余
四项；下一实验固定权重，仅将部署 action 切换到 `expected_utility`。

## 15. Retained expected-utility 结果与 action calibration 修正（2026-08-01）

固定 V7 权重的全量消融显示，epoch 72 expected-utility 为 REC
`0.5805637/0.4643458`，切换 `18` 次、precision `61.11%`、oracle recall `0.95%`；
epoch 73 为 REC `0.5807741/0.4645562`，切换 `22` 次、precision `59.09%`、recall
`1.18%`。相比 legacy decision 的约 `25-30%` precision，quality heads 明显更可靠，
但其平均 action margin 约 `-0.376`，把绝大多数候选挡在 fallback 之后。

根因是训练/部署错位：`QueryFallbackGate` 在 expected-utility 模式使用 box/mask heads
生成 `selection_margin`，而 calibrated utility regression 仍回归 `decision_head` 的
`override_margin`。修正后 regression 和 overestimate 统计都绑定实际部署
`selection_margin`；decision 模式的 selection/override margin 相同，旧路径保持精确兼容。
梯度测试覆盖了 regression 到 box/mask heads 的反传，focused tests 为 `92 passed`。

V8 从 epoch 73 retained checkpoint 进行两轮 gate-only 训练，冻结 MCLN、router 和 query
reranker，只更新 fallback gate。配置保持 context 1 层/4 头、top-8、evidence features、
expected utility、setwise temperature `0.25`、gate LR `3e-4`。break cost `2.0` 已直接
编码错误切换风险，因此将额外 `false_override_weight` 设为 `1.0`，避免 class、row 和
regression 三重保守惩罚；该改动基于 action utility 合同，不依赖 ScanRefer 样本阈值。

V8 epoch 1 全量验证反例说明风险权重需要连续校准：`false_override=1.0` 后 predicted
switch 增至 `108`，precision 仅 `25.93%`，Mask 三项明显退化至
`0.5907657/0.4844342/0.4137051`，而 candidate oracle 仍通过目标。epoch 2 继续完成
作为时长对照；后续优先测试中间 `false_override`（保留 utility target 的 break cost）
以及直接 setwise utility head，而不是继续扩大候选源。

V8 epoch 2 仍为过度切换：REC `0.5800379/0.4638199`、Mask
`0.5909760/0.4847497/0.4137946`，predicted switch `118`、precision `25.42%`。
V8 两轮 checkpoint 已删除，结构化收据保留；V7 非 retained epoch 80/latest 也已删除，
历史正式最佳和 V7 epoch 72/73 retained best 未受影响。

## 16. Direct setwise utility head（2026-08-02）

`direct_utility` 在 contextual fallback gate 的共享 hidden representation 上增加一个
`Linear(hidden_dim, 1)` scalar head，零初始化保证迁移旧权重时严格 fallback。它直接输出
候选相对 default 的部署 margin，并由同一 threshold-aware `decision_utility` 同时监督：
setwise loss 学习 fallback 与 top-8 alternatives 的行级选择，calibrated regression 学习
收益幅度。这样不再把 class-balanced focal logits 当作校准概率，也不需要数据集专用的
后处理阈值。

checkpoint migration 只豁免旧 `decision`/`expected_utility` checkpoint 缺少
`utility_head.weight/bias`；direct checkpoint 缺少这两个张量仍视为损坏。CLI、shell、
reranker/cache contracts 均支持新 action mode。focused/full tests 分别为
`95 passed` 和 `2885 passed, 3 warnings`。

V9 从 V7 epoch 73 retained checkpoint 训练 direct utility 两轮，只更新 fallback gate。
运行时可训练参数由 `316,432` 精确增加到 `316,561`，证明只新增 129 个 head 参数；旧 gate
其余 tensor 完整加载。风险合同恢复 `false_override_weight=2.0`，其余保持 top-8、context
1 层/4 头、evidence features、setwise temperature `0.25`、gate LR `3e-4`。

V9 epoch 2 完整验证后仍未突破 V7：learned REC `0.5799327/0.4640303`，Mask
`0.5967606/0.4901136/0.4174779`。候选 oracle 保持 `0.6296803/0.5500631` 与 mIoU
`0.4517495`，但 deployed switch 只有 `143/9508`，其中 `33` 有益、`110` 有害，precision
`23.08%`、oracle recall `2.99%`。direct scalar head 没有解决 candidate max 把 noisy
候选 utility 直接变成 switch 的问题。V9 checkpoint 链接全部删除，日志和收据保留；V7
retained 权重及历史 `0.582878/0.486012` 只读权重未动。

## 17. Hierarchical utility gate（2026-08-02）

V10 把 fallback gate 拆成两级部署动作：

- candidate utility head：仍对每个 top-8 alternative 输出相对 default 的 utility，只用于在
  “已决定切换”的行内选择 query；
- row switch head：对 default hidden、candidate-set mean hidden 和二者差值做集合级判断，
  输出唯一的 fallback-vs-switch margin。

训练目标相应拆分。候选 selection loss 只在 oracle switch 行上优化 top-8 内 query 分布；
row switch 使用 oracle-switch BCE 和 row best utility regression 监督是否离开 fallback。
部署时 `moe_gate_action_margin` 仍保留 candidate utility 供排序，`moe_gate_row_switch_margin`
单独控制切换，因此一个高噪声 candidate margin 不能绕过 row-level veto。

迁移合同保持保守：旧 `decision`/`expected_utility` checkpoint 可缺少 direct `utility_head` 和
新 `row_switch_head`，二者零初始化后严格 fallback；`direct_utility` checkpoint 可缺少
`row_switch_head` 迁移到 hierarchical；但声明为 `hierarchical_utility` 的 checkpoint 若缺少
任一 head，则视为损坏并拒绝。CLI、训练 shell、rec-reranker/cache backbone contract 均接受
`hierarchical_utility` 字符串。

新增测试覆盖：零初始化 fallback、row veto 阻断高 candidate utility、candidate rank 与
row switch 双梯度、CLI 解析、eval override、checkpoint legacy migration 和 hierarchical
缺 tensor 拒绝。验证结果：SourceMoE focused `99 passed`，sidecar/cache focused `74 passed`，
全套 `2889 passed, 3 warnings`。

V10 训练建议从 V7 epoch 73 retained checkpoint 启动，使用：

```bash
EXP=ssq_moe_e73_hierarchical_utility_e2 \
LOG_DIR=/root/autodl-tmp/DATA_ROOT/output/source_moe_hierarchical_utility_train_v10 \
PHASE=gate \
CHECKPOINT_PATH=/root/autodl-tmp/DATA_ROOT/output/source_moe_continue_v7/scanrefer/ssq_moe_context_balanced_source_only_e72_e80/1785552345/ckpt_best_rec_acc050.pth \
SOURCE_MOE_USE_FALLBACK_GATE=1 \
SOURCE_MOE_GATE_ACTION_MODE=hierarchical_utility \
SOURCE_MOE_GATE_OBJECTIVE=balanced_calibrated_utility \
SOURCE_MOE_GATE_SETWISE_TEMPERATURE=0.25 \
SOURCE_MOE_GATE_FALSE_OVERRIDE_WEIGHT=2.0 \
MAX_EPOCH=2 \
START_EPOCH=1 \
scripts/train_scanrefer_source_moe.sh
```

启动合同补充：`source_moe_gate_objective` 的 CLI 默认改为内部 `None`，解析后仍对普通训练
呈现 `balanced_focal`，同时用 `source_moe_gate_objective_explicit` 区分“默认/继承”和
“用户显式新试验”。fresh gate-only 显式 objective 不再被 checkpoint 覆盖；真正 optimizer
resume 仍由原 exact-config contract 管理。`models/losses.py` 的上层 objective 白名单也已同步。
这两处均有回归测试，最终全套为 `2893 passed, 3 warnings`。

正式 V11 使用目录 `1785620910`。此前 `1785618301`（objective 被错误继承）和
`1785620042`（上层白名单 fail closed）均在 checkpoint 前退出，仅保留 config/log 作为
启动审计，不纳入任何指标比较。

监控间隔按实测 epoch 时长估算，而不是固定几分钟轮询：用第 1 轮的训练耗时和验证写收据耗时
估算第 2 轮检查窗口，只有正式 `eval_metrics_epoch_*.json` 落盘后才判断结果和清理权重。

正式 V10 已从 V7 epoch 73 启动，目录为
`output/source_moe_hierarchical_utility_train_v10/scanrefer/ssq_moe_e73_hierarchical_utility_fow2_e2/1785609497/`。
运行时只训练 `source_moe.fallback_gate.*`，共 `366,226` 参数；checkpoint 明确报告加载
epoch 73。首轮按当前速度估算完整训练加验证约 `62-64` 分钟，监控只在预计收据窗口执行。

V10 两轮完整评测显示 row/candidate 分离本身工作正常，但 row class weighting 错配。
epoch 1/2 learned REC 为 `0.5522718/0.4276399` 和 `0.5522718/0.4260618`，Mask 为
`0.5931847/0.4861170/0.4141952` 与 `0.5922381/0.4858014/0.4138547`。相同候选空间的
oracle 仍为 `0.6296803/0.5499580`，而 row gate switch 从 `25.80%` 升到 `30.74%`，
precision 从 `11.21%` 降到 `10.95%`。这排除了候选空间与训练时间，直接定位到 row
oracle 的 inverse-frequency BCE：positive row 虽只约 `11.6%`，却被平衡权重放大到近似
均匀先验，违背了 fallback 风险控制。

V10 的所有 checkpoint 已按两个私有 inode 删除，收据保留。下一版定义为
`hierarchical_risk_calibrated`，保持三项结构不变：candidate utility 做行内选择、row head
做 switch veto、row best-utility regression 做概率校准。唯一改变是目标权重的职责分离：
candidate quality / auxiliary decision 继续 batch-balanced，row switch BCE 改为固定风险成本
（使用 `false_override_weight`）而不使用 inverse-frequency；`break_cost` 仍只定义 utility
target。该目标明确命名为新合同，
避免重新解释 V10 的 `balanced_calibrated_utility` 结果。

实现验证：新 objective 若没有 `row_switch_margin` 则拒绝执行；梯度回归证明相对 V10 的
balanced row BCE，稀有正 switch 的梯度不再被 inverse-frequency 放大，而 fallback 负类仍有
`false_override_weight` 风险约束。测试为 SourceMoE focused `101 passed`、full suite
`2891 passed, 3 warnings`。

V11 复现命令：

```bash
MASTER_PORT=4476 \
EXP=ssq_moe_e73_hierarchical_risk_calibrated_e2 \
LOG_DIR=/root/autodl-tmp/DATA_ROOT/output/source_moe_hierarchical_risk_train_v11 \
PHASE=gate \
CHECKPOINT_PATH=/root/autodl-tmp/DATA_ROOT/output/source_moe_continue_v7/scanrefer/ssq_moe_context_balanced_source_only_e72_e80/1785552345/ckpt_best_rec_acc050.pth \
SOURCE_MOE_GATE_ACTION_MODE=hierarchical_utility \
SOURCE_MOE_GATE_OBJECTIVE=hierarchical_risk_calibrated \
SOURCE_MOE_GATE_SETWISE_TEMPERATURE=0.25 \
SOURCE_MOE_GATE_FALSE_OVERRIDE_WEIGHT=2.0 \
MAX_EPOCH=2 START_EPOCH=1 \
scripts/train_scanrefer_source_moe.sh
```

V11 完整结果表明，仅修正 row-class prior 仍不足以得到可部署的自适应 gate。epoch 1/2
learned REC 分别为 `0.5648927/0.4481489`、`0.5623685/0.4469920`，Mask 分别为
`0.5961296/0.4884308/0.4163539`、`0.5949727/0.4881153/0.4158356`；均低于 V7
retained。两轮 candidate oracle 完全稳定在 `0.6296803/0.5500631/0.4517543`，说明
top-8 候选覆盖足够，瓶颈是 row-level beneficial-switch classification。

结构化诊断给出 epoch 1 的 predicted switch `1021`（beneficial `139`、harmful `882`、
precision `13.61%`、oracle recall `12.61%`），epoch 2 为 `1380`（beneficial `177`、
harmful `1203`、precision `12.83%`、recall `16.06%`）。相较 V10，risk BCE 把 switch
比例从 `25.80%-30.74%` 降到 `10.74%-14.51%`，但 recall 上升仍由更多 false override
换取，第二轮所有正式指标同步下降。后续设计不应再只调全局 class/risk weight；需要给 row
head 更直接的 counterfactual evidence，或采用显式 fallback-vs-candidate pairwise ranking，
并保持训练/部署使用同一个 margin。

本轮 timing 合同也已验证：首轮训练约 `49:28`、验证约 `13:23`，据此只在 epoch 2
预计完成窗口 `08:03-08:04 CST` 检查，收据实际于 `08:03:25 CST` 写入。V11 两个私有
checkpoint inode 的 8 个链接均因低于 V7 retained 而删除；所有 JSON 收据和日志保留。
历史 `0.582878/0.486012` 只读三件套、router anchor 与 V7 retained 权重未进入清理范围。

## 18. Candidate-conditioned pairwise verifier（V12）

V10/V11 的 hierarchical row gate 仍有一个关键因果错位：row representation 聚合所有候选，
row label 却来自 oracle-best candidate，而部署最终操作的是 candidate utility head 当前选中的
单个 query。因此“候选集合里存在好 query”和“模型将要切换到的 query 是好的”被当成了同一
事件。V12 将动作显式拆为 propose-then-verify：

- proposer：`utility_head` 在 top-8 内排序 query；所有有候选的行都参与 listwise utility
  监督，fallback 行也学习最小伤害候选；
- verifier：`pairwise_switch_head` 读取 fallback hidden、实际 proposed hidden、差值、逐维积
  与 proposer margin，只判断这一对 query 是否应切换；
- aligned target：row BCE 和 calibrated regression 使用 proposed candidate 相对 fallback 的
  threshold-aware utility，不再使用不可部署的 oracle-best utility；
- deployment：同一个 verifier margin 与 `decision_margin` 比较，正值才执行 proposer 的 query。

pairwise head 为 `Linear(4H+1,H) -> LayerNorm -> GELU -> Linear(H,1)`，最后一层零初始化，
所以旧 checkpoint 迁移时保持 exact fallback。该模块只依赖候选视觉/几何、文本上下文、来源
分数与相对 utility 标签，不编码 ScanRefer 专用规则，可用于单阶段 ScanRefer、Nr3D 和 Sr3D。
checkpoint 合同允许旧模式缺失该 head，但 `pairwise_verifier` checkpoint 缺少 pairwise 或
utility tensor 时拒绝加载。

复现配置：

```bash
MASTER_PORT=4477 \
EXP=ssq_moe_e73_pairwise_verifier_e2 \
LOG_DIR=/root/autodl-tmp/DATA_ROOT/output/source_moe_pairwise_verifier_train_v12 \
PHASE=gate \
CHECKPOINT_PATH=/root/autodl-tmp/DATA_ROOT/output/source_moe_continue_v7/scanrefer/ssq_moe_context_balanced_source_only_e72_e80/1785552345/ckpt_best_rec_acc050.pth \
SOURCE_MOE_GATE_ACTION_MODE=pairwise_verifier \
SOURCE_MOE_GATE_OBJECTIVE=pairwise_risk_calibrated \
SOURCE_MOE_GATE_USE_EVIDENCE_FEATURES=1 \
SOURCE_MOE_GATE_CONTEXT_LAYERS=1 \
SOURCE_MOE_GATE_CONTEXT_HEADS=4 \
SOURCE_MOE_GATE_SETWISE_TEMPERATURE=0.25 \
SOURCE_MOE_GATE_FALSE_OVERRIDE_WEIGHT=2.0 \
MAX_EPOCH=2 START_EPOCH=1 \
scripts/train_scanrefer_source_moe.sh
```

测试覆盖 zero-init fallback、实际 candidate-conditioned row target、fallback 行 candidate ranking
梯度、proposer/verifier 双梯度、CLI、Hungarian loss、旧 checkpoint 迁移与新 checkpoint
fail-closed。聚焦测试 `163 passed`，全套 `2901 passed, 3 warnings`。

正式 V12 使用 run `1785630296`，tmux `mcln_v12_pairwise_verifier_e1_e2`。真实 V7 epoch 73
checkpoint 迁移后 gate-only allowlist 为 `432,403`，与旧 gate `316,432` + utility
`129` + hierarchical row head `49,665` + pairwise verifier `66,177` 完全一致。首轮按
`1.10 it/s` 及历史验证时长估算收据窗口为 `09:34-09:35 CST`。

V12 epoch 1 learned REC 为 `0.5805637/0.4637148`，Mask 为
`0.5973917/0.4903239/0.4177016`；Mask@0.25 以 1 个 hit 暂时刷新 V7 retained，REC 与
其余 Mask 指标未刷新。结构化诊断显示 oracle-positive 行 `1103`，proposer-selected-positive
行 `408`，实际 switch `109`（beneficial `31`、harmful `78`，precision `28.44%`）。
这证明 candidate-conditioned target 修复了 V10/V11 的过切换崩塌，但所有行 listwise 训练
第一轮仍只在约 `34.0%` oracle 行选到最佳 utility tier，且 verifier 部署偏保守。完整 epoch 2
继续用于判断 proposer recall 是否随训练增加而转化为净 REC 收益。

epoch 2 将 learned REC 提升到 `0.5810896/0.4648717`，Mask 提升到
`0.5979175/0.4911653/0.4183786`；前者刷新 network-only REC@0.25，后者三项全部刷新，
REC@0.50 与 V7 retained 精确持平。switch `37` 次中 beneficial/harmful 为 `18/19`，precision
`48.65%`；proposer-positive row 为 `422`，oracle best-tier match `391/1103=35.45%`。
epoch 1 到 2 五项同向上升，证明 pairwise verifier 尚未收敛。

训练基础设施新增 fail-closed 的 `--source_moe_gate_resume_optimizer`。普通 gate-only 仍按原合同
启动 fresh optimizer；只有显式请求、gate-only、完整配置相同、连续 epoch 且无 reduce-LR/
epoch override 时，才恢复 checkpoint 中的 Adam moments 与 scheduler。V12 epoch 2 收据包含
33 个有状态参数和 scheduler step `6110`。恢复逻辑聚焦测试 `56 passed`，全套
`2906 passed, 3 warnings`；epoch 3-4 将以该状态连续训练，避免把 optimizer reset 当成模型
改进或退化。

精确续训 run `1785639639` 启动审计已通过：明确加载 V12 epoch 2，并打印
`resumed exact gate-only optimizer and scheduler state`；allowlist 仍为 `432,403`。epoch 3
按实测训练/验证时长只在 `12:12-12:13 CST` 检查正式收据。

epoch 3-4 精确续训完成，确认 optimizer reset 不是 V12 后续指标的主要限制。epoch 3 收据在
`12:12:28 CST` 写入，learned REC 为 `0.5808793/0.4646613`，Mask 为
`0.5977072/0.4911653/0.4184125`；epoch 4 收据在 `13:15:09 CST` 写入，learned REC 为
`0.5807741/0.4645562`，Mask 为 `0.5973917/0.4909550/0.4180954`。两轮均低于 V12
epoch 2 的 network-only REC `0.5810896/0.4648717`，但 epoch 3 刷新了 Mask mIoU 到
`0.4184125`，因此 epoch 3 retained 权重保留。

pairwise verifier 的诊断显示方向是正确但瓶颈已经转移。epoch 3 proposer-selected-positive
row 为 `413`，实际 switch `55`，beneficial/harmful `21/34`，precision `38.18%`，
oracle recall `1.90%`；epoch 4 row target 增至 `427`，实际 switch 降到 `30`，
beneficial/harmful `16/14`，precision `53.33%`，oracle recall `1.45%`。也就是说 verifier
正在变保守且更准，但 proposer/top-1 recall 仍太低，且更高 precision 没有转化成更多净 hits。

本次低质量清理只删除 epoch 4/latest 的私有 inode `2170239165`（`ckpt_epoch_4.pth` 与
`ckpt_epoch_last.pth`），保留 epoch 3 的 retained inode `2170239162`、V12 epoch 2 network-only
最佳 inode、历史 `0.582878/0.486012` 后处理三件套、router anchor 与 V7 retained 权重。JSON
receipts、config 和日志均保留，便于后续复现。

V13 不应继续只调 `decision_margin` 或全局 false-override 权重。更有泛化性的结构方向是把
pairwise verifier 从“验证单个 proposer top-1”扩展为“对 top-n candidates 逐一验证后再选择”：
proposer 负责 recall，verifier 负责 precision，最终部署选择 `argmax verifier_margin` 的正收益
candidate，否则 fallback。训练上应把当前 proposed utility、oracle utility 和 hard negatives 同时放入
listwise/pairwise loss，使 top-8 中真正可切换的 query 更稳定进入可部署候选，而不是依赖
ScanRefer 后处理阈值。

## 19. Top-n pairwise verifier（V13）

V13 将 V12 的 propose-then-verify 从单 proposal 扩展到候选集合。top-8 candidate pool
仍由冻结的 MoE candidate score 产生；共享 `pairwise_switch_head` 对每个候选构造
`[fallback hidden, candidate hidden, difference, product, utility margin]`，输出
`[B,Q]` verifier margins。无效/default query 被 mask，部署选择最大 margin 且严格大于
`decision_margin` 的候选，否则返回 shared fallback。

该设计复用 V12 head，不增加参数或数据集专用特征。V12 checkpoint 可直接迁移；声明为
`topn_pairwise_verifier` 的 checkpoint 若缺 `pairwise_switch_head` 仍 fail closed。CLI、
训练 shell、reranker backbone config 合同均新增 action mode 字符串。

训练端有两个互补 loss：

- proposer listwise loss：`utility_head` 对所有 active candidate 学习真实 utility 排序，
  包括无正候选行的最小伤害排序；
- deployed verifier loss：把 fallback 固定 logit 0 与 top-n verifier margins 放在同一
  setwise action space；正行的 soft target 只分配给正 utility candidates 与 fallback，负行
  target 精确为 fallback；逐候选 SmoothL1 regression 校准 margin 幅度，过估计按
  `false_override_weight` 加罚。

相较独立 candidate BCE，row-wise setwise action 不会因每行多个负候选而进一步放大
under-switch，并且 loss 与最终 `argmax positive margin` 动作严格一致。新增测试覆盖：
top-n 可绕过 proposer top-1 选择另一个候选、`[B,Q]` verifier target/gradient、旧 V12 路径
兼容、CLI 和 checkpoint fail-closed。相关测试 `164 passed, 2 warnings`，全套
`2910 passed, 3 warnings`。

A100 128-row smoke 已完成真实 checkpoint load、32 train batches 和 32 eval batches，
trainable 参数仍为 `432,403`；调试权重已删除。正式 run `1785650097` 从 V12 epoch 2
权重启动两轮 fresh-optimizer gate-only 训练，首批 `1.06 it/s`，epoch 1 收据按实测只在
`15:06 CST` 检查。

正式 V13 两轮未通过保留门槛。epoch 1 learned REC `0.5806689/0.4642406`、Mask
`0.5974968/0.4906395/0.4180548`；epoch 2 learned REC `0.5800379/0.4638199`、Mask
`0.5969710/0.4903239/0.4176694`。candidate oracle 仍为
`0.6297854/0.5498528/0.4516363`，说明 top-8 空间没有退化。V13 的 top-n verifier 把
oracle-query match 提高到 `37.91%/39.98%`，但 deployed precision 只有
`32.56%/28.89%`，REC 和 Mask 都低于 V12 epoch 2。

诊断结论：top-n 结构解决了“只验证 proposer top-1”的覆盖问题，但 objective 仍不够适配
`margin > 0` 部署边界。soft setwise target 在正行保留 fallback 概率，使真正正候选过 0
动力不足；per-candidate regression 又把 neutral utility 拉到 0，容易制造边界附近的 false
switch。下一版应保持 V13 部署结构不变，只分离训练目标：proposer 学 recall，verifier 用
hard fallback-vs-positive setwise CE + margin ranking 把正候选推到 `>0`、neutral/break 推到
`<0`，regression 只用于非中性候选或增加安全 margin，不再让 neutral candidate 贴近 0。

V13 两个私有 checkpoint inode 已删除；JSON receipts、config 与 log 保留。历史最佳、V12
epoch 2 与 V12 epoch 3 retained 权重未进入清理范围。

## 20. Risk-separated top-n verifier objective（V14）

V13 的 top-n verifier 已把 oracle-best-tier query match 从 V12 的约 `35.45%` 提高到
`37.91%-39.98%`，但 positive row 的 soft setwise target 同时给 fallback 非零概率，且
neutral candidate 的 regression target 为 0。二者都与部署边界 `margin > 0` 不完全一致：
前者削弱正候选越过 fallback 的梯度，后者让中性候选贴在切换边界上。

V14 保持 V13 的模型参数和部署动作完全不变，只新增 `topn_risk_calibrated` 训练目标：

- 有至少一个正 utility candidate 的行，fallback target 精确为 0，概率只按温度 softmax
  分配给正候选；
- 没有正 utility candidate 的行，target 精确为 fallback，不给 neutral/break candidate
  任何概率；
- positive 与 negative verifier regression target 保持原 threshold-aware utility；
- neutral verifier regression target 从 0 改为 `-setwise_temperature`，在部署边界下方保留
  显式安全间隔。

proposer 的全候选 listwise target、top-8 candidate pool、共享 pairwise head 和最终
`argmax positive margin` 部署均保持不变。该 objective 只使用 box/mask 阈值转移产生的相对
收益和统一温度，不查询 ScanRefer validation、不搜索数据集专用阈值，能够复用到单阶段
ScanRefer、Nr3D 与 Sr3D。

实现同时修正了 `row_switch_margin` 形状校验的缩进，使 `[B]`/`[B,Q]` 合同真正 fail closed。
测试覆盖 positive row 无 fallback mass、negative row exact fallback、neutral safety gap、
positive/negative utility 保真、V13 soft target 数值兼容、CLI、Hungarian loss 与非法 margin
形状。SourceMoE 聚焦测试 `124 passed`，checkpoint/诊断聚焦测试 `30 passed, 2 warnings`，
全套 `2917 passed, 3 warnings`。

A100 128-row smoke run `1785663192` 从 V12 epoch 2 network-only 最佳权重迁移，完成
32 train + 32 eval batches；allowlist 仍精确为 `432,403`。panel 上 predicted switch 为 1，
beneficial/harmful 为 `1/0`，candidate oracle 为 `0.58594/0.53906`，learned REC 为
`0.50000/0.44531`。smoke 的 7 个 checkpoint 硬链接（私有 inode `48606427`）已全部删除，
config、log 和 JSON receipts 保留。V12 epoch 2、V12 epoch 3 与历史 `0.582878` artifacts
再次核验未动。

正式 run `1785663614` 使用相同结构从 V12 epoch 2 启动两轮 fresh gate optimizer。epoch 1
REC 为 `0.5803534/0.4642406`，Mask 为 `0.5971813/0.4905343/0.4178422`；epoch 2
REC 降到 `0.5798275/0.4638199`，Mask 降到
`0.5967606/0.4902188/0.4175319`。candidate oracle 两轮保持
`0.6297854/0.5500631/0.4517465`，所以失败仍发生在 action verifier，而不是候选空间。

V14 的 oracle-query match 从 epoch 1 的 `433/1102=39.29%` 升到 epoch 2 的
`450/1102=40.83%`，但 switch precision 从 `26/80=32.50%` 降到
`23/85=27.06%`。这说明 risk-separated target 确实没有压低候选覆盖，却未能把 harmful
candidate 与 positive candidate 分开；neutral safety gap 也不能处理 utility 明确为负、但
margin 仍被过估计的 hard negatives。继续调 temperature 或 false-override 全局权重不太可能
解决可分性，应改为显式 fallback-vs-candidate hard-negative ranking、风险/收益双头或带不确定性
的 veto head，并保持跨数据集共享输入。

两轮五项均低于 V12 retained。epoch 1 inode `10814120961` 的 6 个链接与 epoch 2 inode
`10814120962` 的 2 个链接已全部删除；正式 run 的 config、log、两轮 metrics/diagnostics 和
retention receipt 保留。当前 retained 仍为 V12 epoch 2 network-only REC/Mask 最佳与 V12
epoch 3 Mask mIoU 最佳，历史后处理 `0.582878/0.486012` artifacts 未删除。

## 21. Dual evidence verifier（V15）

V14 的 query match 达到 `40.83%`，但 harmful switch 仍占 `72.94%`。代码审计发现现有
pairwise verifier 只读取 candidate/default hidden 与 proposer utility；同一 gate 中已经接受
threshold-transition 监督的 `box_head`、`mask_head`、`decision_head` 预测没有显式进入最终
verifier。这使 hard-negative 风险只能间接穿过共享 hidden。

V15 新增 `topn_dual_evidence_verifier` action。原 `pairwise_switch_head` 保持 benefit head；
新的 `safety_switch_head` 读取 candidate hidden、box break/neutral/fix 概率、mask
break/neutral/fix 概率、decision 概率、expected utility 与 direct utility，共
`H + 2*(3T) + 5` 维。部署 margin 为
`min(benefit_margin, safety_margin)`，所以两个 head 都严格大于 0 才切换；任何一个 head
否决都回到 shared fallback。

新的 `topn_dual_risk_calibrated` objective 保留 risk-separated setwise loss 训练 benefit
head，另用 candidate-wise class-balanced binary focal loss 训练 safety head。positive utility
是 safety 正类，neutral/break 是 veto 类，false positive 继续乘统一风险代价。candidate-wise
监督避免 row softmax 只压低相对排序、却允许某个 harmful margin 仍高于 0。新 head 最后一层
零初始化，因此从 V12 迁移时部署精确 fallback；V15 checkpoint 若缺 safety tensor 则
fail closed。

safety head 增加 `19,073` 个参数，gate-only allowlist 从 `432,403` 变为 `451,476`。
SourceMoE/集成测试 `130 passed`，checkpoint/config 聚焦测试 `54 passed, 2 warnings`，全套
`2923 passed, 3 warnings`。A100 smoke run `1785672783` 完成 32 train + 32 eval batches，
验证 V12 checkpoint 迁移、allowlist 和零切换保守起点；私有 smoke inode `8606111000` 的
7 个链接已删除，config/log/JSON receipts 保留。

正式 run `1785673198` 从 V12 epoch 2 network-only 最佳启动两轮 fresh-optimizer gate-only
训练。epoch 1 REC 为 `0.5805637/0.4642406`，Mask 为
`0.5976020/0.4905343/0.4178641`；epoch 2 REC 降到 `0.5803534/0.4642406`，Mask 降到
`0.5974968/0.4901136/0.4176414`。candidate oracle 两轮保持
`0.6297854/0.5500631/0.4517248`，仍显著高于部署结果。

epoch 1 的 oracle-query match 达到 `509/1105=46.06%`，但实际 switch `98` 次，仅
`30` 次 beneficial、`68` 次 harmful，precision `30.61%`。epoch 2 match 回落到
`463/1105=41.90%`，switch 扩大为 `140` 次，其中 `35/105` 为 beneficial/harmful，precision
进一步降至 `25.00%`；safety false-positive ratio 同时从 `15.80%` 升到 `19.43%`。因此双头
`min` 部署逻辑本身不能保证风险分离：class-balanced safety objective 仍会把部分 hard negative
推到正 margin，且两个 head 在共享 candidate representation 上发生同向误判。后续不应继续只加
串联 veto head；更值得验证的是不平衡校准明确的 negative-first cascade、带置信区间/拒识区间的
风险预测，或让 benefit 与 safety 使用解耦 evidence encoder，并先在 train split 上报告分头
AUROC/precision-recall calibration，再决定是否进入完整 validation。

V15 两轮五项都未刷新 V12。epoch 1 私有 inode `173359264` 的 6 个链接和 epoch 2/latest
私有 inode `173594587` 的 2 个链接已全部删除，正式 config、log 和 JSON receipts 保留。
V12 epoch 2 network-only 最佳、V12 epoch 3 Mask-mIoU 最佳以及历史
`0.582878/0.486012` 后处理 artifacts 已复核未动。

## 22. Absolute-quality delta gate（V16）

V15 的 class-balanced focal safety head 在稀疏正类下没有得到可直接用 `0` 作为决策边界的
posterior；正式验证中 false positive 反而随训练增加。V16 不再用额外二分类 veto 判断相对
switch，而是复用项目中 QueryReranker 的 dense absolute-quality 思路：每个 top-n candidate
输出 box/mask 的 `@0.25`、`@0.50` 和连续 IoU，部署分数为 candidate 预测质量减去 shared
default query 的预测质量。质量聚合使用递增 threshold tier、连续 IoU 以及现有 mask utility
权重 `0.25`，不包含 ScanRefer validation threshold search。

action 为 `topn_absolute_quality_delta`，objective 为
`topn_absolute_quality_calibrated`。二值质量监督使用普通 BCE，避免 inverse-frequency focal
改变 0-logit 的概率含义；连续 IoU 使用 Smooth L1。candidate action target 继续使用
risk-separated fallback/top-n 规则与 false-override row cost。新的 6-output
`absolute_quality_head` 为 `774` 参数且零初始化，因此新旧权重迁移时保持 exact fallback；
声明 V16 action/objective 的 checkpoint 缺少该 head 时 fail closed。

实现覆盖 `models/source_moe.py`、`models/losses.py`、`main_utils.py`、训练脚本、reranker
入口和集成测试。聚焦测试为 `136 passed` 与 `24 passed`，全套为
`2929 passed, 3 warnings`。实际 gate-only allowlist 为 `452,250` 个参数；模型 state 中
`452,252` 个 fallback-gate 数值还包含 2 个非训练 threshold buffer。

A100 smoke `1785685637` 完成 32 train + 32 eval batches，首轮 128-row panel 上 switch
`26` 次、beneficial/harmful 为 `1/25`。精确 optimizer continuation `1785685989` 继续到
epoch 3 后，switch 数收缩为 `9 -> 7 -> 5`，precision 提高为
`11.11% -> 28.57% -> 40.00%`。epoch 2/3 learned REC 都为
`0.500000/0.453125`，相对 fixed default `0.492188/0.437500` 增加 `1/2` 个命中；Mask 为
`0.500000/0.406250/0.350474`。candidate oracle `0.585938/0.539062` 表明结构仍有充足上限，
而重复 pass 后净指标转正，支持进入全量两轮 fresh-optimizer 正式验证。

smoke checkpoint 审计确认所有模型/optimizer 张量 finite，与 V12 共享参数的变化严格限制在
`fallback_gate`，非 gate 主干逐元素不变。三个私有 inode `13999350`、`4406554475`、
`4406554474` 的 15 个 checkpoint 链接已删除，仅保留 config、log、metrics、diagnostics 与
retention receipts。V12 两个 retained 权重及历史后处理最佳 artifacts 未删除。

正式 run `1785686848` 从 V12 epoch 2 network-only best 启动两轮 fresh-optimizer
gate-only 训练，tmux 为 `mcln_v16_absolute_quality_e1_e2`。启动合同确认全量
`36665/9508` train/val、V12 checkpoint 完整加载、allowlist `452,250`，batch 500 实测约
`1.0-1.2 it/s`。epoch 1 JSON 收据窗口按本次吞吐估算为 `01:31-01:37 CST`，中途不轮询。

epoch 1 收据于 `01:31:22 CST` 落盘，并在 `01:35:40 CST` 检查：REC `0.5804586/0.4645562`，Mask
`0.5972865/0.4906395/0.4180273`，均未刷新 V12 retained。candidate oracle 为
`0.6301009/0.5500631/0.4517077`；实际 switch `17` 次，beneficial/harmful `9/8`，
precision `52.94%`，oracle-query match `33.15%`。该轮显示绝对质量头的误切换风险已受控，
但 switch recall 偏低，epoch 2 仍需完成。

epoch 1 训练实际耗时 `3526.66s`，收据于 `01:31:22 CST` 落盘；按该真实周期，epoch 2
检查窗口估算为 `02:45-02:50 CST`，期间不轮询中间日志。

epoch 2 训练耗时 `3428.66s`，收据于 `02:45:07 CST` 落盘并在 `02:49:59 CST` 检查。
REC 为 `0.5806689/0.4647665`，Mask 为 `0.5974968/0.4908498/0.4181498`；相对 epoch 1
五项均改善，但相对 V12 epoch 2 retained，REC 少 `4/1` hits，Mask threshold hits 少
`4/3`，mIoU 低 `0.0002288`，没有刷新全局 best。

gate switch 从 epoch 1 的 `17` 增至 `18`，beneficial/harmful 从 `9/8` 改善为 `11/7`，
precision 达 `61.11%`；oracle-query match 为 `386/1101=35.06%`，oracle switch recall
仍只有 `1.00%`。因此 V16 的 dense absolute-quality 监督确实解决了一部分二分类边界校准和
harmful override 问题，但当前 action 只在极少行执行，主要瓶颈已从 precision 转为 recall。
后续结构应在保持绝对质量校准的同时学习置信区间或 pairwise ranking margin，而不是通过全局
validation threshold 放宽开关。

正式 checkpoint 审计确认模型 `1206` 个张量和 optimizer `54` 个张量均 finite；与 V12
共享的 `1198` 个 tensor 中，变化全部限制在 `fallback_gate`。epoch 1 私有实体已由 retention
自动清理；epoch 2 私有 inode `4382073512` 的 7 个链接因未刷新任何 retained 指标已删除，
config/log/两轮 JSON receipts 保留。V12 epoch 2、V12 epoch 3 与历史后处理最佳 artifacts
复核未动。

## 23. V12-anchor absolute-quality correction cascade（V17）

V16 证明 dense absolute quality 能提高 false-override precision，但直接替换 V12 pairwise
verifier 后只保留了约 `1%` oracle-switch recall。V17 不再训练另一个平行 gate，而把 V12
决策本身作为动态 anchor。stage 1 完整复用并冻结 V12：`utility_head` 在原 top-n 中给出
proposer query，`pairwise_switch_head` 决定保持 shared default 还是采用 proposer。stage 2
以该逐样本结果为 fallback action，对剩余 top-n alternatives 和显式加入的 shared default
逐一计算 correction margin。

stage 2 输入为 anchor/candidate quality hidden、difference、Hadamard product、6 维绝对质量
evidence delta、聚合质量 delta、冻结 direct-utility delta 和冻结 V12 verifier margin。
`cascade_correction_head` 的最终线性层与 `absolute_quality_head` 均为零初始化，因此所有
correction margin 初始为 0；部署使用严格 `margin > 0`，所以 V17 初始 selected query 必然
等于 V12 stage-1 anchor。若 V12 选错，shared default 可作为正 correction 撤销该切换；若另一
候选更好，也可直接从 V12 anchor 提升，不需要先退回 default。

训练 objective `cascade_absolute_quality_calibrated` 以动态 anchor 而非固定 shared query
生成 threshold-transition target。correction verifier 接受 risk-separated setwise loss：存在
正收益 query 时 fallback target mass 为 0，否则为 exact fallback；neutral utility 下移一个
`setwise_temperature` safety gap。另对每个 correction margin 回归相对 anchor utility，并对
候选与 anchor 都训练 box/mask `@0.25`、`@0.50` 和连续 IoU 的 dense absolute quality。

V17 只训练三个模块：`absolute_quality_head`、`cascade_quality_adapter`、
`cascade_correction_head`。hidden dim 128 时分别为 `774`、`16,768`、`67,201` 参数，总计
`84,743`。`--source_moe_gate_new_heads_only` 对 optimizer 使用精确模块白名单；训练态先将整网和
旧 fallback gate 置 eval，再只将三个新模块置 train。旧 hidden、direct utility 与 V12 margin
在 correction 路径全部 detach，从结构和优化器两侧阻断旧参数更新。

迁移合同要求 V12 checkpoint 必须包含 `utility_head` 与 `pairwise_switch_head`，但允许缺少三个
V17 模块；V17 checkpoint 若缺任一新模块则拒绝加载。新模块构造被包在 CPU RNG fork 中，避免
仅因多出随机初始化就改变 `num_workers=0` 的数据增强流。evaluator 的 candidate oracle 同样
改为补回动态 action anchor，而旧 action 自动回退原 shared default，保持历史兼容。

行为测试覆盖零初始化逐元素保持 V12、撤销到 shared default、提升另一候选、动态-anchor
target、仅新头获得梯度、`84,743` allowlist、CLI/config/checkpoint fail-closed 与 RNG 流等价。
全套结果为 `2943 passed, 3 warnings`。真实 128-row 对照中，V12 与 RNG-fixed V17 的 REC
hits 都为 `64/57`，Mask threshold hits 都为 `64/52`，candidate oracle 都为 `75/69`，且 V17
correction 为 0；跨进程 Mask mIoU 差为 `2.51e-5`。

32 train + 32 eval smoke `1785699777` 的 gate loss 从训练均值 `1.6027` 降到评估
`1.5276`，所有张量 finite，32 步内 correction 仍为 0。与 V12 共享的 `1198` 个 tensor 全部
逐元素不变，optimizer 只包含 12 个 V17 参数 tensor。smoke 权重 inode `4299477133` 的 7 个
链接已删除，收据保留。正式两轮 run `1785700377` 从 V12 epoch 2 启动，tmux
`mcln_v17_cascade_e1_e2`，首轮收据预计 `05:12-05:23 CST`。

正式 run 已于 `06:35:49 CST` 完整结束。epoch 1/2 learned REC 都为
`0.5810896/0.4650820`（`5525/4422` hits），Mask 都为
`0.5979175/0.4911653/0.4184406`（`5685/4670` hits）。REC@0.25 和 Mask threshold hits 保持
V12，REC@0.5 比 V12 epoch 2 多 2 hits，mIoU 比 V12 epoch 3 高 `0.0000281`；然而两轮
correction 均为 0，因此不能把这两个微小形式刷新解释为 cascade 的净贡献，也没有达到
`0.59/0.49` 目标。

top-8 gate candidate oracle 稳定为 `0.6296803/0.5500631/0.4517358`。动态 anchor 的 oracle
switch 行为 `1087/9508=11.43%`，oracle-query match 从 `419/1087=38.55%` 提高到
`441/1087=40.57%`，但 predicted switch 和 oracle-switch recall 始终为 0。训练确实更新了
全部 12 个 V17 tensor，平均最大 correction margin 也从 `-1.0022` 上升到 `-0.9339`，只是
仍远低于严格的 0 部署边界。这把瓶颈定位为 row-level opportunity 的类别不平衡和动作边界
校准，而不是 absolute-quality 表征或 candidate recall。

正式 checkpoint 审计确认每份模型 `1216` 个 tensor、optimizer 12 个参数状态和 24 个状态
tensor 全部 finite。与 V12 共享的 `1198` 个 tensor 全部逐元素不变；12 个 V17 tensor 共
`84,743` 参数，与 optimizer 一一对应且两轮间全部更新。当前代码还保存 6 个未被 V17 action
使用的冻结 `safety_switch_head` tensor，它们两轮间也完全不变。

epoch 1 retained inode `2194044680` 的 6 个 best/epoch 链接保留。epoch 2 指标完全打平且没有
correction，inode `2194044681` 的 2 个链接已删除，JSON receipts、config、log 和 retention
manifest 保留。V12 epoch 2/3 retained inode 及历史 `0.582878/0.486012` artifacts 再次核验
未动。

V18 应继续使用动态 V12 anchor，但把 correction objective 拆成两个经过独立归一化的层级：
先判断当前行是否存在任何正收益 query，再只在正行内学习 query 排序和幅度；正行与 fallback
行应分别求均值后等权组合，避免 `88.57%` fallback 行把所有 margin 推到负区间。部署仍使用
统一的 train-split 风险校准和严格 fallback，不引入 ScanRefer validation 专用阈值，才能继续
用于单阶段 ScanRefer、Nr3D 和 Sr3D。

## 24. Opportunity-balanced hierarchical correction（V18）

V18 将 V17 的 correction 拆成“行级 opportunity”和“正行内 query ranking”两层。stage 1
继续完整冻结并复用 V12 的 `utility_head + pairwise_switch_head`，输出逐样本动态 anchor。
stage 2 的 `cascade_correction_head` 为候选 query 生成相对排序 logit；新增的
`cascade_opportunity_head` 接收 anchor 特征、候选集合 masked mean/max、最强候选及其差分，
判断这一行是否存在值得部署的 correction。mean/max pooling 对候选排列不敏感，因而不会把
ScanRefer 特定的候选顺序编码成数据集捷径。

部署规则是严格的两级 fallback：只有 `opportunity_margin > 0` 才切换，并选取 rank logit
最大的有效候选；否则保持 stage-1 anchor。opportunity 最后一层零初始化，迁移初始 margin 为 0，
所以初始输出按严格大于零规则逐元素保持 V17/V12 anchor。该规则不包含 validation threshold
sweep，后续单阶段 ScanRefer、Nr3D 和 Sr3D 使用同一训练分布风险边界。

objective `cascade_opportunity_balanced_calibrated` 对 positive/fallback 行分别归一化后等权组合
opportunity loss，消除 11.43% 正行被 88.57% fallback 行压制的问题。query setwise ranking 和
relative utility regression 只在至少包含一个 beneficial correction 的行上训练；absolute box/mask
quality 辅助监督继续覆盖候选与 anchor。对应 action 为
`cascade_opportunity_quality_correction`。

新增 opportunity head 含 6 个 parameter tensor、`33,281` 参数；连同 V17 的三个 correction
模块，V18 allowlist 共 18 个 tensor、`118,024` 参数。gate-only 训练先将整网置 eval/frozen，
再只启用这四个模块。V17 checkpoint 允许缺少 opportunity head，但必须完整包含 V17 correction
模块；声明 V18 action 的 checkpoint 若缺少任一 V18 模块则 fail closed。focused tests 为
`182 passed`，全套为 `2952 passed, 3 warnings`。

真实零初始化 run `1785712186` 从 V17 retained checkpoint 加载，128-row REC 为 `64/58`，Mask
为 `64/52/0.3506319`，candidate oracle 为 `75/69`，实际 correction 为 0。对照 V17 run
`1785711999` 的 REC `64/57`、Mask `64/52/0.3505157`、oracle `75/69`；一个 `@0.50` hit 的
差异也存在于 fixed default，属于独立 GPU/点采样评估非确定性，而不是 opportunity 分支改变选择。

学习 smoke `1785712574` 完成 32 train + 32 eval batches。gate loss 为
`0.9377/0.9378`，18 个 optimizer 参数状态及 36 个 Adam moment tensor 全部 finite，moment
numel 与 allowlist 同为 `118,024`。与 V12 共享的 `1198` 个模型 tensor 全部逐元素不变；相对
V17 仅 12 个既有 correction tensor 更新，6 个 opportunity tensor 为新增状态。32 步后的
opportunity/correction 仍为 0，保守启动符合零初始化预期，但需要完整数据判断 class-balanced
objective 是否能形成有效边界。smoke inode `39860010` 的 7 个权重链接已清理，收据保留。

正式两轮 run `1785713059` 于 `07:24:14 CST` 启动，路径为
`output/source_moe_cascade_opportunity_train_v18/scanrefer/ssq_moe_e73_cascade_opportunity_e2/1785713059/`，
tmux 为 `mcln_v18_opportunity_e1_e2`。首轮/次轮收据按既往实测周期预计分别在
`08:41-08:52 CST` 和 `10:02-10:13 CST`，窗口前不读取中间训练日志。历史后处理
`0.582878/0.486012`、V12 retained 和 V17 retained 权重均已复核未动。

V18 epoch 1 的正式行为证明两级拆分仍缺一层约束。learned REC 为
`0.5515356/0.4216449`，Mask 为 `0.5942364/0.4868532/0.4148749`；candidate oracle 仍为
`0.6296803/0.5496424/0.4516072`。opportunity 预测切换 `2651` 行，beneficial/harmful 仅
`265/2386`，precision `10.00%`，而 oracle-switch recall 已达到 `24.38%`。因此 V18 从
“完全没有 recall”跨到了“有 recall 但缺乏 action precision”，继续增加正类权重或调整部署阈值
都不是结构性解决方案。

## 25. Selected-query safety verification（V19）

V18 的 opportunity label 是 `any(candidate utility > 0)`，而部署 action 是
`argmax(rank_logit)`。只要 rank head 没选中正 utility query，row label 即使预测正确也会执行
错误候选。V19 为此加入 `cascade_candidate_safety_head`，对每个 candidate 的 correction hidden
输出独立 safety margin。它不是第二个 row classifier，而是验证将要执行的具体 query。

训练保持三项任务解耦：row opportunity 使用正/fallback 行平衡，conditional rank 只在正行内
计算，candidate safety 则在所有有效候选上使用保留真实 class prior 的 cost-sensitive focal
loss。非正 utility 候选的 regression target 至少下压到 `-setwise_temperature`，正候选回归真实
threshold utility；false-positive cost 继续由统一的 `false_override_weight` 控制，不使用任何
validation-set threshold。

推理先按 rank logit 得到 selected query，再计算：

```text
deployed_margin = min(row_opportunity_margin, selected_query_safety_margin)
switch = has_candidate and deployed_margin > 0
```

因此 row recall head 不能绕过 candidate verifier，candidate verifier 也不能在无 opportunity 行
单独触发。safety 最终层零初始化保证 V18 -> V19 的初始 deployed margin 为 0，严格保持动态 V12
anchor。新 action/objective 是 `cascade_opportunity_verified_correction` 和
`cascade_opportunity_verified_calibrated`。

hidden dim 128 时 safety head 为 `Linear(128,128) + LayerNorm + GELU + Linear(128,1)`，共
6 个 parameter tensor、`16,897` 参数；V19 new-head-only 总计 24 个 tensor、`134,921` 参数。
迁移合同允许 V12 缺少 cascade/opportunity/safety、V17 缺少 opportunity/safety、V18 只缺少
safety；声明 V19 action 后任一 V19 模块缺失均拒绝加载。行为测试覆盖零初始化、selected-query
veto、safety 正负梯度、prior-preserving loss、optimizer/train-mode allowlist 和 V12/V17/V18
迁移，定向结果 `189 passed`，全套 `2959 passed, 3 warnings`。

V18 epoch 2 最终 REC 为 `0.5536390/0.4280606`，Mask 为
`0.5935002/0.4868532/0.4144721`，candidate oracle 为
`0.6296803/0.5496424/0.4516072`。虽然 oracle-switch recall 增到 `29.44%`、oracle-query match
增到 `50.05%`，`3004` 次部署切换中仍有 `2684` 次有害，precision 只有 `10.65%`。这进一步
验证 V18 的瓶颈是 selected action 没有独立安全证明，而不是训练轮数不足。V18 epoch 1 的 4 个
checkpoint 链接和 epoch 2 的 3 个冗余链接已清理，只保留 epoch 2 inode `4299478647` 的
`ckpt_epoch_2.pth` 作为 V19 初始化；历史 V12/V17 与 `0.582878/0.486012` artifacts 未动。

V19 零初始化 run `1785723597` 在固定 128-row panel 上得到 REC `64/57`、Mask
`64/52/0.3505157`、candidate oracle `75/69/0.4406735`。旧 opportunity-positive ratio 为
`29.69%`，但 safety-positive 和 deployed correction 均为 0，证明
`min(row_margin, selected_safety_margin)` 在迁移时严格回退动态 anchor。

学习 smoke `1785723824` 完成 32 train + 32 eval batches，gate loss 为 `0.8471/0.8438`，
safety loss 为 `0.1918/0.1589`。24 个 optimizer state 和 48 个 Adam moment tensor 均 finite
且全部非零，step 均为 32；与 V12 共享的 `1198` 个 tensor 全部不变，V18 已有的 18 个允许
tensor 全部更新，safety 输出层也已脱离零值。smoke 权重的 7 个硬链接已删除，收据保留。

正式两轮 run `1785724401` 从唯一 V18 epoch 2 初始化权重启动，路径为
`output/source_moe_cascade_opportunity_verified_train_v19/scanrefer/ssq_moe_e73_cascade_opportunity_verified_e2/1785724401/`，
tmux `mcln_v19_verified_e1_e2`。只训练五个 correction/verifier 模块的 24 个 tensor、
`134,921` 参数；按前序真实周期，epoch 1/2 收据分别只在 `11:55-12:05 CST` 和
`13:15-13:25 CST` 窗口检查。

V19 两轮正式结果为：epoch 1 REC `0.5810896/0.4652924`、Mask
`0.5981279/0.4912705/0.4185490`；epoch 2 REC `0.5811948/0.4653976`、Mask
`0.5982331/0.4913757/0.4186131`。epoch 2 五项均刷新 retained，candidate oracle 保持
`0.6296803/0.5500631/0.4517077`。V19 最终仅部署 5 次 correction，beneficial/harmful 为
`2/3`，precision `40.00%`，但 oracle-switch recall 只有 `0.18%`。因此 candidate safety
确实解决 V18 的误切爆炸，却把问题推到 verifier coverage collapse。

正式 checkpoint 的 V12 共享 `1198` 个 tensor 全部不变，V18 共有 tensor 的变化严格限制在
18 个允许模块；24 个 optimizer state、48 个 Adam moment tensor 均 finite、非零且 step 为
`6110`。epoch 2 inode `34391215` 以 7 个 retention 链接和 1 个 protected 链接保留；epoch 1
与临时 V18 初始化权重已清理。当前 network-only REC 最高为
`0.5811948/0.4653976`，Mask 最高为 `0.5982331/0.4913757/0.4186131`；历史后处理 REC
`0.582878/0.486012` 仍安全保留。

## 26. Candidate-level joint risk action（V20）

V19 的部署顺序仍是 `argmax(rank) -> safety veto`。rank head 的 oracle-query match 约 50%，
所以它选到非正 utility query 时，safety 即使判断正确也只能整行 fallback，不能改选同一候选集
中的正 utility query。V20 将 selection 和 verification 合并为可学习的逐候选部署边界：

```text
joint_input[q] = concat(correction_hidden[q],
                        candidate_rank_margin[q],
                        candidate_safety_margin[q],
                        row_opportunity_margin)
candidate_joint_margin[q] = cascade_joint_action_head(joint_input[q])
selected_query = argmax_valid(candidate_joint_margin)
switch = candidate_joint_margin[selected_query] > 0
```

联合头为 `Linear(hidden_dim+3, hidden_dim) + LayerNorm + GELU + Linear(hidden_dim,1)`。最终层
零初始化，使 V19 -> V20 迁移时所有 joint margin 为 0，严格回退动态 V12 anchor。row
opportunity 与 candidate safety 在 V20 中是可学习证据而非 hard veto；部署边界固定为显式
fallback logit `0`，不读取旧 `decision_margin` 或 ScanRefer validation threshold。

训练不再对大量 candidate 独立做未归一化 prior regression，而在每行加入显式 fallback logit，
用 risk-separated fallback-plus-candidates target 监督最终 action。positive row 的概率质量只分配给
正 utility 候选，fallback row 的质量全部给 fallback；正/fallback 行分别归一化。训练 batch 的
opportunity prior 使用 Beta(1,1) 平滑，再与 `false_override_weight` 一起形成 detached log-odds
校正，使 raw joint margin 的 0 仍是 Bayes cost 对齐的部署边界。该结构让 loss、argmax 和部署
action 严格一致，同时 prior 来自训练分布而非 validation threshold，能够迁移到单阶段
ScanRefer、Nr3D 与 Sr3D。

新 action/objective 为 `cascade_joint_risk_correction` / `cascade_joint_risk_calibrated`。hidden
dim 128 时联合头含 6 个 parameter tensor、`17,281` 参数；连同 absolute-quality、quality
adapter、rank correction、row opportunity 和 candidate safety，new-head-only 总计 30 个
tensor、`152,202` 参数。迁移合同允许 V12 缺少六个模块、V17 缺少后三个、V18 缺少后两个、
V19 只缺 joint head；声明 V20 action 后任一模块缺失均 fail closed。行为测试覆盖 CLI、零初始化、
安全候选改选、先验校准梯度、optimizer/train-mode 白名单和四代迁移，相关回归为
`243 passed, 2 warnings`。

真实 128-row 零评估 run `1785735825` 从受保护 V19 epoch 2 权重迁移，REC 为 `64/57`、Mask
为 `64/52/0.3504906`、candidate oracle 为 `75/69/0.4407775`；joint-positive、predicted
correction 和 harmful correction 均为 0，验证 V19 -> V20 的实际 checkpoint/GPU 合同。

32 train + 32 eval smoke run `1785736073` 的 joint action loss 从零评估 `3.4084` 降到
`2.9193`，gate loss 为 `1.2567`；32 步后仍无 correction/harmful switch。checkpoint 与 V19
共有的 `1228` 个 tensor 中只更新 24 个既有 allowlist tensor，joint head 新增 6 个，总计
30 个 tensor、`152,202` 参数；30 组 Adam 一二阶矩全部非零、step 均为 32，所有模型和
optimizer tensor 均 finite。debug inode `2162713570` 的 7 个权重链接已清理，只保留收据。

最终全套测试为 `2968 passed, 3 warnings`。正式 run `1785736801` 于 `14:00:01 CST` 从 V19
protected epoch 2 启动，两轮均使用完整 `36665/9508` train/validation，只训练 30 个 V20
allowlist tensor；路径为
`output/source_moe_cascade_joint_risk_train_v20/scanrefer/ssq_moe_e73_cascade_joint_risk_e2/1785736801/`，
tmux 为 `mcln_v20_joint_risk_e1_e2`。按前序真实周期，epoch 1/2 只在
`15:17-15:25 CST` / `16:32-16:42 CST` 收据窗口检查。

### 26.1 V20 结果门禁与跨数据集静态审计

V20 正式结果在收据产生前固定采用以下口径，不根据 validation 结果改阈值：REC 目标为
`>=5610/9508`（Acc@0.25）且 `>=4659/9508`（Acc@0.50）；Mask 以 V19 同一 network
checkpoint 的 `5688/4672/0.4186131` 为保护参照。每轮同时报告 candidate oracle、实际
correction、beneficial/harmful correction、precision、oracle-switch recall 和 joint-action
target match，不能用 oracle 或 128-row panel 代替 learned 结果。

对当前实现的跨数据集静态复核确认：`SourceMoE` 本身没有 ScanRefer/Nr3D/Sr3D 名称分支，
router 会对每个 `[B,Q]` query 在 routed experts 中自适应 top-k，V20 的 target 也只依赖相对
fallback 的 IoU threshold transition 和训练分布 prior。仍有一个明确局限：routed experts
之间的选择是逐 query 的，但 shared 与 routed 分支的残差强度仍由单个全局
`routed_scale` 控制，不是逐 query 权重。它不影响本次只训练 action heads 的 V20 合同，
但不能把当前结构描述成“所有源权重都已逐目标自适应”。

正式结果按证据分流：candidate oracle 达标而 learned 未达标时，继续优化候选集合内的联合
action/校准，不扫描数据集阈值；candidate oracle 自身不足时，才评估零初始化的逐 query
shared-routed residual scale；REC 达标但 Mask 保护失败时，转向 query-specific mask fusion。
任何新分支都保持同一实现迁移到单阶段 ScanRefer、Nr3D 和 Sr3D，并以跨数据集正式结果而非
源组合 sweep 证明泛化性。

迁移边界也已静态核对：`sample_dataset` 的监督 mask 会保留 `scanrefer/nr3d/sr3d`、只排除
多目标 `scannet` 行，三源 adapter 依赖的 decoder query、contrastive token 和 mask 输出均在
MCLN forward 内构造。当前尚不能直接宣称跨数据集 runnable，因为正式 launcher 仍固定
`--butd --test_dataset scanrefer --expected_eval_sample_count 9508`。单阶段 ScanRefer 必须显式
移除 `--butd` 并先验收 checkpoint/source tensor 形状；Nr3D/Sr3D 必须使用各自 dataset 参数和
validation 样本数。若某一模型变体不能生成 `mask_text` 所需张量，应 fail closed 或显式定义
新 source contract，不能静默换成该数据集上手选的源组合。

正式 checkpoint 的可复现只读审计已固化为
`scripts/audit_source_moe_checkpoint.py`。它将 V20 与受保护 V19 initializer 逐 tensor 比较，
要求共有/变化/新增 tensor 精确为 `1228/24/6`，所有变化位于六个 new-head allowlist，
optimizer 精确含 30 个 state、`152,202` 个参数的非零 finite Adam moments，并验证 epoch 1/2
step 分别为 `3055/6110` 以及 action/objective/new-head-only 合同。任一条件失败时不允许据此
创建 protected alias 或删除权重。审计器与原 oracle/retention 的聚焦回归为 `18 passed`。

V20 epoch 1 正式训练耗时 `4487.88s`，验证约 `22:51`，收据于 `15:46:52 CST` 落盘。
learned REC 为 `5524/4421 = 0.5809844/0.4649769`，Mask 为
`5685/4670/0.4183627`；相对 V19 network-best 分别少 `2/4` REC hits、`3/2` Mask hits，
mIoU 低 `0.0002504`，五项均未刷新。candidate oracle 为
`5987/5228 = 0.6296803/0.5498528`、mIoU `0.4516537`，因此 action headroom 充分。

关键部署诊断是 oracle switch `1090`、oracle-query match `522`，但 predicted、beneficial、
harmful correction 全部为 `0`；joint-action target match 为约 `88.50%`，最大 margin 均保持
负值。V20 epoch 1 因此不是 false override，而是 joint boundary 的 coverage collapse，当前
行为退回动态 V12 anchor。按用户要求仍完整训练 epoch 2；按首轮真实总周期重新估算，最终
收据窗口为 `17:23-17:27 CST`，期间不轮询 batch 日志。

### 26.2 若 epoch 2 仍零切换：V21 预注册方向

epoch 1 的静态 loss 审计显示，最终实际部署的 `joint_action_loss` 只是 selection、row switch、
utility regression、absolute quality、candidate safety、joint action 六个 decision terms 之一；
其余旧目标仍共同更新 joint evidence。`88.50%` 的 action target match 恰好主要来自 fallback
rows，因此不能解释成联合头已学好。若 epoch 2 仍为零/近零 correction，下一版不延长同配置、
不扫 margin，而采用“V19 action 作为显式 fallback + 新 joint action 为主目标”的增量结构：

1. 零初始化时回退 V19 已部署 action，而不是退回 V12 anchor，保证第 0 步精确保留当前
   network-best；候选 utility 也改为相对该 action 计算。
2. 冻结 V19 的 24 个已训练 evidence tensor，只训练新的 fallback-plus-candidates set head，
   避免五个旧辅助目标继续移动 joint 输入；fallback 自身作为集合 token，最终 margin 为
   candidate logit 减 fallback logit，部署边界仍固定为 0。
3. primary loss 使用 risk-separated setwise action；另加训练分布驱动的正行 coverage
   lower-bound 或 class-count margin，直接阻止 all-fallback trivial solution。coverage 只来自
   train oracle prior，不从 validation 选择阈值。
4. 先在 32-batch probe 量 joint/auxiliary gradient cosine；只有确认负冲突时才引入 PCGrad，
   不默认承担多次 backward 的训练成本。

开源对照固定到具体提交：SelectiveNet `geifmany/selectivenet@a6d0a8f` 的 coverage penalty、
PCGrad `WeiChengTseng/Pytorch-PCGrad@e987ac6`、LDAM-DRW
`kaidic/LDAM-DRW@2536330`，以及 noisy top-k MoE
`davidmrau/mixture-of-experts@f662999`。SelectiveNet 原实现需要人工 coverage 和 post-hoc
calibration，不能原样照搬；LDAM 的 class-count margin 和 SelectiveNet 的 differentiable
coverage 只作为训练分布自适应约束参考。当前 expert usage 已约 50/50 且 candidate oracle
充分，所以 noisy top-k router 不是 V21 首要改动。

### 26.3 V20 最终结论

V20 epoch 2 与 epoch 1 的完整 9,508-row 结果逐项相同：REC 为
`5524/4421 = 0.5809844/0.4649769`，Mask 为
`5685/4670 = 0.5979175/0.4911653`，mIoU 为 `0.4183627`；candidate oracle 仍为
`5987/5228 = 0.6296803/0.5498528`，mIoU 为 `0.4516537`。第二轮仍然没有任何 predicted、
beneficial 或 harmful correction，最大 joint margin 保持负值。因此 V20 的问题已确认是
部署 coverage collapse，不是预训练未加载、训练轮数不足、候选覆盖不足或偶然的单轮波动。
两轮低指标 checkpoint 已在审计后清理，JSON、config 和 log 收据保留；受保护 V19 与历史
`0.582878/0.486012` 三组件未动。

## 27. V19-fallback token set correction（V21）

V21 不再把未训练的 action head 放在 V12 anchor 前面，而是把完整 V19 已部署 query 作为集合
中的显式 fallback。V19 的 rank、row opportunity 与 selected-query safety 路径原样执行，得到
`v19_fallback_query`；新 head 只学习“保留该 query，还是从剩余候选中改选”的相对动作：

```text
fallback = deployed_v19_query
candidates = v19_correction_candidates - {fallback}
if V19 changed V12 anchor:
    candidates += {v12_anchor}  # 可撤销有害的 V19 correction
margin[q] = set_logit[q] - set_logit[fallback]
selected = argmax_q margin[q] if max(margin) > 0 else fallback
```

`FallbackTokenSetActionHead` 对 fallback token 与最多 8 个候选做 input projection、无位置编码的
multi-head self-attention、residual LayerNorm、FFN 和共享线性评分。评分层零初始化，所以迁移后
每个 candidate margin 都精确为 0，逐 query、逐 score 保持 V19 输出；共享评分和无位置编码也
保证 query permutation equivariance。输入由 V19 correction hidden、rank margin、safety
margin 和 row opportunity margin 构成并整体 detach，24 个 V19 evidence parameter tensors
全部冻结。hidden dim 128 时新 head 含 15 个 parameter tensors、`149,504` 个可训练参数。

新 action/objective 固定为 `cascade_v19_fallback_set_correction` /
`cascade_v19_fallback_set_risk_calibrated`。utility 和 oracle target 均相对已部署 V19 fallback
计算，而非 V12 anchor。主目标为 fallback-plus-candidates 的 prior-corrected setwise risk loss；
另加 positive-row 与 fallback-row 分开平均的 deployment-boundary loss，使正行在零边界获得
向上的梯度、fallback 行获得向下的梯度，避免 88.5% fallback target 仅靠类别数量形成
all-fallback 解。温度 `0.25` 和类别信息只来自训练 batch，不使用 ScanRefer validation threshold。

迁移合同是 fail closed：完整 V19 只能缺少新 set head，完整 V21 可以精确恢复；V12、V17、
V18、V20 或缺失任一 V19 evidence tensor 的 checkpoint 均拒绝作为 V21 initializer。审计 profile
要求 V19/V21 的 common/changed/new tensor 精确为 `1228/0/15`，optimizer 只含 15 个 state、
`149,504` parameter numel。行为测试覆盖 V19 exact identity、撤销有害 correction、query
permutation equivariance、正/负边界梯度、相对 V19 target、optimizer allowlist 和迁移拒绝。

成功的 32 train + 32 eval smoke run `1785751933` 只训练上述 `149,504` 个参数。train/eval
gate loss 为 `4.1962/3.9136`，joint loss 为 `3.5153/3.2434`，boundary loss 为
`0.6810/0.6702`；所有 batch、模型和 optimizer tensor 均 finite。checkpoint 精确审计通过：
`1228/0/15`、15 个 optimizer state、30 个非零 Adam moment tensors、step 32。128-row REC
`0.5000/0.4453125`、Mask `0.5000/0.40625/0.350190` 仅是稳定性 panel，不作为正式指标。
smoke inode `37694830` 的 7 个权重链接已删除，audit/config/log/metrics 收据保留。首次 smoke
因 debug tensor 名包含 `loss` 被通用标量-loss finite guard 拒绝；键名改为
`moe_gate_supervision_fallback_query` 后问题消除。最终全项目回归为
`2988 passed, 3 warnings`。

V21 正式两轮 run `1785752995` 于 `18:29:45 CST` 启动，tmux 为
`mcln_v21_v19_fallback_set_e1_e2`，目录为
`output/source_moe_v19_fallback_set_train_v21/scanrefer/ssq_moe_e73_v19_fallback_set_e2/1785752995/`。
它从 protected V19 直接初始化，使用 `batch_size=12`、`num_workers=4`、完整
`36665/9508` train/validation、gate LR `3e-4`、setwise temperature `0.25`，只训练 V21
set head。以最近同配置 V20 的启动、训练和验证实测周期估算，epoch 1 收据窗口为
`20:15-20:25 CST`，epoch 2 为 `21:47-21:57 CST`；窗口前只做一次初始化合同核验，不读取
中间 batch 日志。

### 27.1 V21 正式结果与失败归因

正式 run 实际在 `18:38:59 CST` 完成数据加载，epoch 1/2 训练分别耗时
`3248.84s/3205.42s`，两轮收据于 `19:48:18/20:54:53 CST` 落盘。epoch 1 REC 为
`5158/3884 = 0.5424905/0.4084981`，Mask 为
`5607/4586 = 0.5897139/0.4823307`、mIoU `0.4114272`；epoch 2 REC 为
`5168/3878 = 0.5435423/0.4078671`，Mask 为
`5632/4604 = 0.5923433/0.4842238`、mIoU `0.4131095`。两轮均显著低于 protected V19。

候选 oracle 两轮保持 `5989/5227 = 0.6298906/0.5497476`、mIoU `0.4516835`，但部署
switch 从 epoch 1 的 `3549` 墙加到 epoch 2 的 `3911`。beneficial/harmful 分别为
`362/3187` 和 `346/3565`，precision 从 `10.20%` 降到 `8.85%`；真实 opportunity 始终只有
`1088/9508 = 11.44%`。因此不是候选覆盖、预训练、训练轮数或 checkpoint 漂移问题，而是
等权 deployment-boundary loss 抹掉真实 row prior，把 action 推回 V18 同类的过切解。

两份 checkpoint 审计均通过：相对 V19 的 common/changed/new 固定为 `1228/0/15`，optimizer
为 15 states、`149,504` numel，step 分别为 `3055/6110`，模型与 Adam moments 全部 finite
且非零。两个低指标 inode `37694825/16720963` 的共 8 个 `.pth` 链接已删除，释放约
1.21 GB；config、log、两轮 metrics/diagnostics、retention 和 checkpoint audit JSON 保留。
protected V19 inode `34391215` 仍为 8 links，历史 `0.582878/0.486012` 三组件未动。

V22 不应只删除 boundary 后重复一次弱 evidence 训练。当前 SourceMoE gate 虽包含 64 维 query
projection、pooled text、归一化 box、三源 rank、288 维 decoder query 和三源 raw score，却
缺少已在冻结 parent/geometry reranker 中验证有效的 target-text projection、main/modifier/
pronoun/relation/other 分量分数、scene-normalized box、top-two margin、objectness、query-specific
mask confidence/foreground/text-query Dice 与 target cosine。下一结构应把这套无 GT 的 152 维
rich evidence 作为共享的跨候选输入，使用 fallback-token set encoder 联合学习 query reorder；
动作只采用一个保留经验 row prior 与 false-switch cost 的 fallback-plus-candidates proper risk
loss，避免再让两个目标对同一零边界给出相反梯度。该输入和目标不依赖 ScanRefer validation
阈值，缺失可选 source 时必须显式 mask，才能迁移到单阶段 ScanRefer、Nr3D 与 Sr3D。

## 28. Rich-evidence empirical-risk set correction（V22）

V22 已实现上述结构分流。action/objective 固定为
`cascade_v19_rich_set_correction` / `cascade_v19_rich_set_empirical_risk`。在线 MCLN 直接复用
`rec-query-v1` 的 152 维、无 GT per-query state：64 维 normalized query projection、64 维
target-text projection，以及 scene-normalized box、五类语言分量分数、default/contrastive
score 与 rank、两源 top-two score/margin、objectness、mask confidence/foreground/
text-query Dice 和 target cosine。该 state 只由模型输出、输入文本映射和点云范围构造，未读取
validation label、最佳阈值或固定源组合。

`RichFallbackTokenSetActionHead` 先单独 LayerNorm rich state，再与 V19 的 correction hidden、
rank margin、candidate safety margin 和 row opportunity margin 拼接，并送入无位置编码的
fallback-token self-attention set scorer。完整 V19 action 是显式 fallback；候选不包含 fallback，
但 V19 曾改写 V12 anchor 时会把 anchor 放回集合以允许撤销旧 correction。共享 score 层零初始化，
因此所有 candidate 相对 fallback 的 margin 初始精确为 0，固定部署边界 `>0` 在第 0 步严格
保持 V19。V19 的 24 个 evidence tensor 和送入新头的 action/rich state 全部冻结、detach，只训练
新 rich set head 的 17 个 parameter tensors、`169,264` 个参数。

V22 只保留一个 proper empirical set-risk objective。fallback logit 固定为 0，target 在有益候选间
按训练 utility/temperature 分布；loss 直接在真实 sample rows 上取平均，不再使用 V21 的
positive/fallback 分组等权 boundary，也不平移部署 logit。false-switch cost `2.0` 只加权真实
fallback rows，因而保留训练分布的 row prior，同时仍显式惩罚有害切换。迁移合同 fail closed：
只接受完整 V19 或完整 V22；V20/V21、旧版本及缺失任一 V19 evidence tensor 的 checkpoint 均拒绝。
审计 profile 要求 V19->V22 common/changed/new 为 `1228/0/17`，optimizer 恰有 17 states、
`169,264` parameter numel，正式 epoch 1/2 step 为 `3055/6110`。

真实 protected V19 到 V22 的 128-row 零初始化 run `1785764542` 得到 REC `64/57`、Mask
`64/52/0.3504906`，correction switch 为 0，与同一 panel 的 V19 完全一致。32 train + 32 eval
smoke run `1785764713` 完成后仍全部 finite；checkpoint 审计为 `1228/0/17`、17 optimizer
states、`169,264` numel、step 32，且所有 Adam moments finite/nonzero。smoke 的 7 个 `.pth`
硬链接已删除，JSON/config/log/audit 收据保留。相关实现、迁移、loss、CLI 和行为测试并入全量
回归，结果为 `3006 passed, 3 warnings`。

正式两轮 run `1785765110` 从 protected V19 启动，完整使用 `36665/9508` train/validation、
`batch_size=12`、`num_workers=4`、gate LR `3e-4`、temperature `0.25`。epoch 1 REC 为
`5521/4420 = 0.5806689/0.4648717`，Mask 为
`5682/4671 = 0.5976020/0.4912705`、mIoU `0.4182182`；相对 V19 network-best 分别少
`5/5` REC hits、`6/1` Mask hits，mIoU 少 `0.0003949`，未刷新最佳。candidate oracle 为
`5987/5230 = 0.6296803/0.5500631`、mIoU `0.4517544`，说明候选覆盖仍足够；实际只切换
168 rows，其中 beneficial/harmful 为 `22/146`，precision `13.10%`、oracle-switch recall
`2.03%`。epoch 1 checkpoint 审计通过 `1228/0/17`、17 states、step 3055。按用户要求继续
完成 epoch 2，不用首轮结果提前停止；最终 retention 与清理只依据完整第二轮收据和审计。

### 28.2 V22 正式结论与保留策略

epoch 2 已完整完成（训练 step `6110`，9,508-row validation receipt 于 `2026-08-04 00:09:23
CST` 写入）。learned REC 为 `5521/4418 = 0.5806689/0.4646613`；Mask 为
`5680/4670 = 0.5973917/0.4911653`，mIoU `0.4181528`。相较 epoch 1，REC@0.25 不变、
REC@0.50 少 2 hits，Mask 三项均略降；相较 protected V19 network-best
`5526/4425 = 0.5811948/0.4653976` 与 `5688/4672/0.4186131`，分别少 `5/7` REC hits、
`8/2` Mask hits，mIoU 少 `0.0004603`。因此“预训练未加载”与“训练轮数不够”均已被完整两轮
收据排除。

V22 candidate oracle 仍为 `5987/5230 = 0.6296803/0.5500631`、mIoU `0.4517544`，远高于
目标；瓶颈是 learned action 没有把 oracle 候选转成部署切换。epoch 2 只预测 104 次额外
switch（`1.09%`），其中 beneficial/harmful 为 `14/90`，precision `13.46%`；真实 oracle
opportunity 为 `1085/9508 = 11.41%`，oracle-query match `500/1085 = 46.08%`，switch recall
仅 `1.29%`。这确认 empirical risk head 仍然 under-switch 且候选级排序/质量表征不足，不应
继续原 objective 或用 validation threshold 补救。

epoch 2 checkpoint 审计严格通过：相对 protected V19 的 common/changed/new 为 `1228/0/17`，
17 个 optimizer state、`169,264` parameter numel、step `6110`，所有模型和 Adam moments
finite/nonzero。由于两轮均未刷新任何保留指标，正式目录内 8 个 `.pth` 硬链接已删除，释放约
`1.2 GiB`；两轮结构化 receipts、config、log 和 audit JSON 保留。protected V19 inode
`34391215` 仍为 8 links，历史系统级 `0.582878/0.486012` 三组件未动。

当前最高指标口径保持：系统级历史后处理 REC `0.582878/0.486012`；network-only V19
`0.5811948/0.4653976`；Mask 最高仍为 V19 `0.5982331/0.4913757/0.4186131`。V22 未进入
protected best。下一步按 28.1 的预注册 `dense-quality adaptive source MoE` 分流推进，先做
32-step dense-gradient/identity smoke，再决定正式训练；不把 V22 坏权重作为初始化。

### 28.1 未达标时的预注册分流：dense-quality adaptive source MoE

若 V22 epoch 2 learned REC 仍低于 `0.59/0.49`，同时 candidate oracle 继续高于目标，则不延长
相同 empirical action loss，也不在 validation 上搜索 margin。现有证据表明 V22 只用约 11% 的
positive rows 学一个最终动作，而历史 parent reranker 的有效信号来自每个候选的 threshold/IoU
密集监督；现有 source router 虽逐 query 选择 routed expert，shared default 与 routed 分支的
混合强度仍是全局标量。下一版固定处理这两个结构瓶颈：

1. 保留并冻结完整 V19 deployed action，作为不可退化 fallback；新增 head 的最终 residual scale
   和 action score 零初始化，第 0 步必须逐 query 精确保留 V19。
2. 在同一 rich set encoder 上为 fallback 与每个候选预测 box `IoU>0.25`、`IoU>0.50`、连续
   IoU，以及对应 mask quality；所有有效 candidate 都参加 dense BCE、Huber 和 tier-pairwise/
   listwise loss，不再只靠 rare row action label 学 representation。
3. 把 default 设为 shared expert，contrastive/mask 设为 routed experts；新增逐 query
   `shared-vs-routed` residual mixing，而不是沿用一个全局 `routed_scale`。router 输入 rich evidence，
   source 不可用时由显式 `[B,Q,S]` validity mask 排除，不能用零值冒充有效 source。
4. 部署 margin 由 candidate 与 V19 fallback 的单调 expected-quality/risk 差构成，并保留固定
   fallback logit 0；训练期 false-switch cost 只约束反事实风险，不引入 ScanRefer validation
   threshold。opaque residual action 只能做有界修正，不能绕过 dense quality。
5. 正式训练前必须通过 V19 identity、source permutation/absence、query permutation、dense
   gradient coverage、单调质量、全 tensor finite、new-head optimizer allowlist 和跨数据集
   adapter contract 测试。只有 train-only smoke 显示 dense heads、router 和 residual scale 均有
   非零梯度，才启动下一次 9,508-row 正式验证。

该分流对应用户提出的“共享源 + 目标自适应单源/多源”结构：固定的是 source schema 和风险
函数，不是 ScanRefer 上人工找到的最佳源组合。单阶段 ScanRefer、Nr3D、Sr3D 复用同一模块，
只通过 source validity 和 dataset sample mask 表达可用输入。

## 29. Dense-quality adaptive source MoE（V23）

V23 已按 28.1 的预注册方向实现。action/objective 固定为
`cascade_v23_dense_quality_correction` / `cascade_v23_dense_quality_risk`，并且只接受完整 V19
或精确 V23 checkpoint；V20/V21/V22 及残缺 V19 初始化均 fail closed。完整 V19 action、相关
evidence 参数及输入保持冻结/detach，作为不可退化 fallback。

`AdaptiveSourceMixer` 使用 rich evidence、query state 与各 routed source state，为每个 query
预测 routed-source 分布及 shared/routed residual mixing。default 是 shared source，
contrastive/mask 是 routed sources；显式 `[B,Q,S]` `source_validity` 严格排除缺失 source。
router 和 mixing 输出层均零初始化，模块构造放在 `torch.random.fork_rng` 中，因此迁移第 0 步
逐 query 保持 V19 router、mix scale、candidate score 和最终选择。无效的共享 softmax bias 已
删除，避免一个理论上恒等的参数产生永久全零 Adam moments。

`DenseQualityFallbackSetActionHead` 将完整 V19 fallback 与最多 8 个候选送入无位置编码的
permutation-equivariant set attention。每个有效 token 密集预测 box/mask 的 `IoU>0.25`、
`IoU>0.50` 和连续 IoU；训练组合 threshold BCE、IoU SmoothL1、box threshold-aware listwise、
box-tier-constrained mask listwise，以及相对 V19 fallback 的 empirical set action risk。部署
margin 只由 candidate quality 减 fallback quality 构成，fallback logit 固定为 0，质量到部署
分数的映射对每个质量坐标单调递增，不使用 ScanRefer validation threshold。

最终生产合同含 39 个新 parameter tensors、`588,603` 个参数；optimizer 只允许
`adaptive_source_mixer` 与 `cascade_dense_quality_set_head`。聚焦回归为 `251 passed`，全项目
回归为 `3017 passed, 3 warnings`，覆盖 V19 identity、source/query permutation、缺源 mask、
dense gradient、单调质量、精确 V23 resume、旧版本迁移拒绝及 optimizer allowlist。

真实 protected V19 的 128-row identity run `1785776794` 得到 REC `64/57`、Mask hits
`64/52`，predicted correction 为 0，和同 panel 的 V19 阈值命中完全一致。Mask mIoU
`0.3506319` 相对历史同 panel 有约 `1.4e-4` CUDA 浮点波动，但 query 与阈值命中未变化。
32 train + 32 eval smoke run `1785776991` 后严格 checkpoint 审计通过：V19->V23
common/changed/new 为 `1228/0/39`，optimizer 39 states、`588,603` numel、step 32，78 个
Adam moment tensors 全部 finite/nonzero。smoke inode `10744414654` 的 7 个 `.pth` 链接已
删除，config、log、metrics、diagnostics、retention 与 `v23_audit.json` 保留。protected V19
inode `34391215` 仍为 8 links，历史后处理 `0.582878/0.486012` 三组件未动。

### 29.1 V23 正式两轮结果与结论

正式 run `1785777457` 从 protected V19 启动，使用完整 `36,665/9,508` train/validation、
`batch_size=12`、`num_workers=4`、gate LR `3e-4`，epoch 1/2 训练耗时分别为
`2978.88s/3002.69s`，receipt 分别于 `02:29:30/03:32:43` 写入。

epoch 1 learned REC 为 `5507/9508=0.5791965`、`4418/9508=0.4646613`，Mask 为
`5684/9508=0.5978124`、`4671/9508=0.4912705`、mIoU `0.4182359`。epoch 2 REC 为
`5516/9508=0.5801430`、`4420/9508=0.4648717`，Mask 为
`5681/9508=0.5974968`、`4672/9508=0.4913757`、mIoU `0.4181260`。epoch 2 的 Mask@0.50
与 V19 的 `4672/9508=0.4913757` 并列但未刷新；REC@0.25/0.50 仍低于 V19 的
`0.5811948/0.4653976`，也没有达到 `0.59/0.49`。

候选 oracle 两轮仍为 `5993/5231` 与 `5992/5228`，说明候选覆盖不是瓶颈；learned action
epoch 1/2 分别切换 `522/249` 行，beneficial/harmful 为 `57/465`、`21/228`，precision
降至 `10.92%/8.43%`，oracle-switch recall 为 `5.26%/1.94%`。因此 V23 的 dense quality 表示
收到梯度且候选质量可用，但部署风险排序仍产生高 false-switch，第二轮也未恢复，不能归因于
预训练未加载或训练轮数不足；不应把这次权重用于下一版初始化，也不应以 validation threshold
补救。

epoch 1/2 审计均通过：V19->V23 common/changed/new `1228/0/39`，39 optimizer states、
`588,603` numel、step `3055/6110`，模型及 78 个 Adam moment tensors 全部 finite/nonzero。
由于 V23 没有刷新 REC 或严格更高的 Mask best，正式目录内两轮共 8 个 `.pth` 硬链接已删除；
`eval_metrics_epoch_{1,2}.json`、diagnostics、retention、config、log、
`v23_audit_epoch_{1,2}.json` 保留。protected V19 inode `34391215` 仍为 8 links，历史
`0.582878/0.486012` 三组件仍完整。

## 30. Relative-risk fallback set correction（V24）

V24 在 V23 的 dense absolute quality supervision 上增加 permutation-equivariant
`RelativeRiskFallbackSetActionHead`，action/objective 固定为
`cascade_v24_relative_risk_correction` / `cascade_v24_relative_risk`。候选集合仍显式包含
V19 fallback；relative utility target 允许在没有正收益候选时保持 fallback，避免“只要存在
正候选就强制切换”。set head 不使用 query position，query 重排只由候选内容决定；V19
初始化保持零输出、部署边界仍为 `margin > 0`，第 0 步保持 identity。V23 dense quality head
继续提供绝对 box/mask 监督，新增 relative-risk head 学习候选相对 fallback 的收益和风险。

V24 通过聚焦回归 `116 passed`、Python compile 检查和 128-row GPU smoke。smoke 训练成功
更新 759,167 个参数，optimizer 有 60 个 state；审计为 `1228/0/60`，所有模型参数及
Adam moments finite/nonzero。正式 run 为 `1785787527`，使用 protected V19、完整
`36,665/9,508` train/validation、`batch_size=12`、两轮训练；epoch 1/2 训练耗时分别为
`2999.57s/2968.19s`。

正式收据结果如下（均为 learned selector，分母 9,508）：

| epoch | REC @0.25 | REC @0.50 | Mask @0.25 | Mask @0.50 | Mask mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `5516/9508=0.580143` | `4414/9508=0.464241` | `5675/9508=0.596866` | `4657/9508=0.489798` | `0.417244` |
| 2 | `5519/9508=0.580459` | `4419/9508=0.464767` | `5678/9508=0.597181` | `4670/9508=0.491165` | `0.417964` |

epoch 2 的 candidate oracle 为 `5987/5226`（约 `0.63031/0.54964`），但实际 action 只切换
84 行，其中 beneficial/harmful 为 `10/74`，false-switch rate `88.10%`，oracle-switch
recall `0.93%`。epoch 1 为 `279` 次切换、`50/229` beneficial/harmful、false-switch `82.08%`、
oracle recall `4.59%`。候选覆盖明显高于 learned 结果，瓶颈仍是风险校准而不是预训练加载、
训练轮数或候选不足；V24 不应继续用相同 objective 或用 validation threshold 后处理补救。

正式 epoch-2 checkpoint 审计通过：相对 protected V19 的 common/changed/new 为
`1228/0/60`，optimizer `60` states、`759,167` parameter numel、step `6110`，action/objective
和 new-head-only 合同均匹配。metric retention 自动移除了未成为 best 的 epoch-1 `.pth`，
由于 epoch-2 也未超过受保护 global best，正式目录已移除全部 7 个低指标 `.pth`；它们被
移入该 run 的 `quarantine_low_metric/`，可恢复但不作为发布或下一版初始化。正式目录保留
metrics/diagnostics、config、log、retention 和 `v24_audit_epoch_2.json`。128-row smoke 的
低指标权重同样位于 `source_moe_v24_smoke/.../quarantine_low_metric/`；protected V19 和
历史后处理 `0.582878/0.486012` 三组件均未删除或覆盖。

V24 之后当前最高指标口径不变：系统级历史后处理 REC `0.582878/0.486012`；network-only
V19 REC `0.5811948/0.4653976`；Mask 最高为 V19
`0.5982331/0.4913757/0.4186131`。下一轮若继续，应优先改进跨候选风险校准并在
ScanRefer/Nr3D/Sr3D 上复用 source-validity contract，而不是为 ScanRefer 搜索固定源组合或
部署阈值。

### 30.1 V24 dense-quality deployment ablation

为区分 V24 的 absolute dense-quality 表示与 relative-risk action，本轮从 V24 epoch-2 临时构造
只读诊断 checkpoint：移除 relative-risk head 的部署作用，直接恢复 V23 dense-quality margin，
独立跑完全部 `9,508` 行 validation；临时 checkpoint 随后删除。正式诊断目录为
`output/source_moe_v24_dense_quality_ablation_eval/scannet,scanrefer/ssq_moe_v24_epoch2_dense_quality_ablation/1785796728/`。

official receipt 的 REC 为 `5474/4431 = 0.5757257/0.4660286`，Mask 为
`5657/4640 = 0.5949727/0.4880101`、mIoU `0.4155182`。结构化 action diagnostics 记录
`4030/9508 = 42.39%` 次 correction，其中 beneficial/harmful 为 `319/3711`，precision
`7.92%`、false-switch `92.08%`；oracle opportunity 为 `1086`，beneficial recall
`29.37%`。这与 V24 epoch-2 只切换 84 行形成两端：dense-quality 直接部署严重过切，V24
relative risk 严重欠切，而且两者的剩余切换仍以 harmful 为主。因此候选表征含有部分 @0.50
信号，但固定零边界下的连续效用校准才是下一结构必须直接解决的瓶颈。

## 31. Pairwise calibrated fallback risk（V25）

V25 action/objective 固定为 `cascade_v25_pairwise_calibrated_correction` /
`cascade_v25_pairwise_calibrated_risk`，仍只从 protected V19 初始化，不继承 V24 低指标权重。
它保留 V23 的逐 query adaptive source mixer 和 dense box/mask quality head，但用
`CalibratedPairwiseRiskSetActionHead` 替换 V24 的共享无偏置 token scorer。

新 head 先用无位置编码的 self-attention 编码显式的 V19 fallback 与 candidate set，再为每个
candidate 构造 `candidate hidden / fallback hidden / difference / elementwise product / dense
quality evidence delta` 的角色明确比较。最终 utility head 直接输出 candidate 相对 fallback 的
连续部署效用，包含可学习 bias 以表示经验 switch prior；辅助 benefit head 单独学习正收益判别。
两个输出层 weight/bias 均零初始化，所以固定部署规则仍为 `utility > 0`，第 0 步严格保持 V19，
不需要 ScanRefer validation threshold。集合编码没有 query position，保持 query permutation
equivariance；source absence 继续由 `[B,Q,S]` validity mask 表达。

训练目标同时包含五项：candidate-vs-fallback SmoothL1 utility regression、对效用高估加权的
false-override cost、保留经验样本 prior 的 auxiliary benefit focal risk、显式
fallback-plus-candidates empirical setwise action risk，以及 V23 的 dense absolute box/mask
quality 与 listwise ranking。V24 的关键接线缺陷已修复：V25 utility regression 在 dense-quality
分支之前显式进入总 loss，不再出现日志恒为 0 的伪校准目标。

生产合同为 69 个新 parameter tensors、`825,997` 个参数，其中 pairwise head 为 30 tensors、
`237,394` 参数；optimizer 只允许 adaptive mixer、dense-quality head 和 pairwise head。测试覆盖
V19 exact identity、query permutation、utility/benefit 正负梯度、CLI、optimizer/train-mode
allowlist、V19/V25 fail-closed migration 和 audit profile。全项目回归为 `3023 passed, 2 warnings`。

真实 128-row smoke run `1785799106` 完成 10 train steps 和完整 debug validation；REC 为
`0.5000/0.4453125`，Mask 为 `0.5000/0.40625/0.3504906`，新 correction 为 0。checkpoint
审计严格通过 V19->V25 `1228/0/69`，optimizer 为 69 states、`825,997` numel、step 10，
138 个 Adam moment tensors 全部 finite/nonzero。该 panel 只作为结构与可学习性门禁，不与
9,508-row 正式指标比较。

V25 正式两轮 run `1785799635` 已从 protected V19 启动，目录为
`output/source_moe_v25_pairwise_calibrated_train_v25/scanrefer/ssq_moe_e73_v25_pairwise_calibrated_e2/1785799635/`。
合同为完整 `36,665/9,508` train/validation、`batch_size=12`、gate LR `3e-4`、temperature
`0.25`、new-head-only。按最近实测，epoch 1/2 receipt 窗口为 `08:38-08:43 CST` /
`09:42-09:47 CST`；只在预计完成窗口检查，不读取分钟级中间日志。

### 31.1 V25 epoch 1 正式收据

epoch 1 训练耗时 `3132.72s`，official REC 为
`5525/4420 = 0.5810896/0.4648717`；Mask 为
`5687/4671 = 0.5981279/0.4912705`、mIoU `0.4184664`。相对 protected V19 只少
`1/5` 个 REC hits、`1/1` 个 Mask hits，mIoU 少 `0.0001467`；没有刷新 best，但明显比 V24
更接近不可退化 fallback。candidate oracle 仍为 `5987/5226 = 0.6296803/0.5496424`、mIoU
`0.4516471`。V25 额外 correction 只有 10 行，beneficial/harmful `2/8`、precision `20%`、
oracle-switch recall `0.18%`，说明连续 utility 已减少错误过切，但首轮仍明显欠切。

epoch-1 checkpoint 审计通过 V19->V25 `1228/0/69`，optimizer 69 states、`825,997` numel、
step `3055`，模型与 138 个 Adam moments 全部 finite/nonzero。epoch 2 按预注册合同继续，不因
首轮未刷新 best 提前停止；依据本轮实耗，最终 receipt 窗口修正为 `09:47-09:49 CST`。

### 31.2 V25 最终结果、审计与清理

epoch 2 训练/验证已完整结束。official REC 为
`5525/4421 = 0.5810896/0.4649769`；Mask 为
`5687/4670 = 0.5981279/0.4911653`、mIoU `0.4184430`。相对 epoch 1 只有 REC@0.50
增加 1 hit，Mask@0.50 少 1 hit；相对 protected V19 仍少 `1/4` REC hits、`1/2` Mask hits，
mIoU 少 `0.0001702`，未刷新任何 global best。candidate oracle 为
`5988/5228 = 0.6297854/0.5498528`、mIoU `0.4516748`，候选覆盖仍远高于目标。

V25 最终新增 correction 为 0，真实 oracle opportunity 为 `1086/9508 = 11.42%`。validation
聚合日志中 candidate benefit target 约 `4.93%`，但 benefit predicted-positive ratio 和 utility
positive-candidate ratio 均为 0，utility max-margin mean 约 `-1.3369`。checkpoint 中
utility/benefit bias 仅为 `-0.0718/-0.0372`，说明 collapse 不是单个手工 intercept，而是经验
candidate imbalance、false-switch cost 与 setwise fallback risk 共同把整套 pair representation
推向负半轴。V25 证明显式成对表示和连续回归能消除过切，却没有恢复 rare-benefit recall；继续
相同 loss 或增加 epoch 没有依据。

epoch-2 审计严格通过 V19->V25 `1228/0/69`，optimizer 69 states、`825,997` numel、step
`6110`，模型和 138 个 Adam moments 全部 finite/nonzero。formal 与 smoke 根目录中的所有
`.pth` 已移入各自 `quarantine_low_metric/`，不作为发布或下一版初始化；metrics、diagnostics、
config、log、retention 和 epoch-1/2 audit 收据保留。protected V19 inode `34391215` 仍为
8 links、SHA-256 `2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`，
历史 `0.582878/0.486012` 三组件也保持完整。

下一分流不搜索 validation threshold，也不延长 V25。应把 auxiliary benefit 改成真正参与部署
的 prior-restored likelihood margin：训练时对 rare positive/negative 做 class balance，同时把
训练集即时 empirical prior 的 log-odds 明确加回 raw margin，使固定零边界仍表示 cost-aware
posterior，而不是 balanced posterior。continuous utility 只作为独立辅助回归，candidate 内部
排序使用 shift-invariant listwise loss，避免 empirical setwise fallback loss 再次把同一部署边界
推负。该改动仍复用 V25 pairwise set encoder、dense quality 和 source-validity contract，从 V19
零初始化并保持跨 ScanRefer/Nr3D/Sr3D 的固定规则。

## 32. Prior-restored pairwise benefit（V26）

V26 action/objective 为 `cascade_v26_prior_restored_pairwise_correction` /
`cascade_v26_prior_restored_pairwise_risk`。它不新增数据集专用 source 组合或 validation threshold，
而是复用 V25 的 permutation-equivariant pairwise set encoder、V23 dense-quality evidence 与
逐 query adaptive source mixer。source validity 仍显式进入 `[B,Q,S]` 路由，因此同一结构可继续
迁移到单阶段 ScanRefer、Nr3D 和 Sr3D。

部署 margin 与回归 margin 在 V26 被明确拆开：`benefit_margin` 学 candidate 是否相对 V19
fallback 有 cost-aware 正收益，并直接用固定规则 `max(benefit_margin) > 0` 决定是否修正；
`utility_margin` 只对连续 `decision_utility` 做 overestimate-weighted SmoothL1 辅助回归。候选间
排序使用 shift-invariant listwise loss，不改变整组候选相对零边界的位置。dense absolute
box/mask quality 与 dense listwise supervision 保留；empirical fallback setwise risk 只作为诊断，
不进入 V26 总 loss，避免重复计入 fallback prior。

benefit 训练采用 class-balanced BCE。对当前训练 batch 的 active candidates 统计正负样本，并用
Beta(1,1) 平滑计算 `prior_shift = log((negative_count + 1) / (positive_count + 1))`；该 shift
只加在训练 logit 上，部署仍使用未平移的 raw margin 和固定零阈值。正负类分别逆频率加权，负类
再乘 `false_override_weight`；不使用 focal gamma，以保留 likelihood-ratio 的校准解释。V26 总
decision loss 为 benefit、continuous utility、candidate listwise、dense absolute quality 与 dense
quality ranking 五项之和。

参数和迁移合同与 V25 形状一致：69 tensors、`825,997` trainable numel；V19 common/changed/new
应为 `1228/0/69`。V19->V26 的零初始化严格保持原部署输出；exact V26 可恢复 optimizer，V24/V25
初始化显式拒绝，避免同形状但错误 objective 的静默加载。专项测试覆盖 raw-zero identity、先验恢复、
正负类梯度平衡、utility/deployment 梯度分离、query permutation、CLI/launcher、optimizer allowlist、
migration 与 audit；全套测试结果为 `3033 passed, 3 warnings`。

128-row smoke run `1785811045` 完成 10 个 optimizer steps：REC
`0.500000/0.4453125`，Mask `0.500000/0.406250/0.3504906`，candidate oracle `75/69`，
predicted correction 为 0。audit profile `v26` 通过 `1228/0/69`，optimizer 69 states、
`825,997` numel、step 10，全部模型 tensors 和 Adam moments finite/nonzero。smoke checkpoint
已删除，只保留可复核收据。正式训练必须从 protected V19 启动，不继承 V25；epoch 1/2 的预期
optimizer step 分别为 `3055/6110`，每轮完成后同时比较 official REC、Mask 与 correction
diagnostics，再按 global-best policy 保留或删除权重。

正式 run `1785811744` 已从只读 protected V19 启动，使用完整 `36,665/9,508` 样本、两轮、
`batch_size=12`、LR `3e-4`、temperature `0.25` 和 new-head-only 合同。run 目录为
`output/source_moe_v26_prior_restored_train/scanrefer/ssq_moe_e73_v26_prior_restored_pairwise_e2/1785811744/`。
按最近实测，epoch 1/2 只在约 `12:01 CST` / `13:06 CST` 的预计完成窗口读取 receipt；中间不做
分钟级日志轮询。

### 32.1 V26 epoch 1 收据

epoch 1 的 `3055` steps 用时 `3627.11s`。official REC 为 `5522/9508 = 0.5807741/0.4646613`，
Mask 为 `5684/9508 = 0.5978124/0.4911653`、mIoU `0.4183568`；candidate oracle 为
`5981/5205`，而固定零边界下 predicted correction 为 0。相对 protected V19 的轻微下降与
零误切一致，说明 V26 首轮尚未释放 rare-benefit recall，而不是加载了错误 checkpoint。

`v26_audit_epoch_1.json` 通过：common/changed/new `1228/0/69`、69 optimizer states、
`825,997` trainable numel、step `3055`，all tensors/moments finite and nonzero。epoch 2
继续使用同一 run 和 optimizer，不从 epoch 1 另起实验。

### 32.2 V26 epoch 2 最终结果与归因

epoch 2 official REC 为 `5516/9508 = 0.5801430/0.4637148`；Mask 为
`5676/9508 = 0.5969710/0.4903239/0.4177342`。candidate oracle 已升到
`6664/5436/0.4768359`，oracle opportunity `1444/9508 = 15.19%`，但 predicted correction
仍为 0，benefit target positive 约 `7.08%`，deployed max-margin mean 约 `-5.8224`。这是一致的
all-fallback collapse：oracle 覆盖不断增加而固定零边界没有任何 rare-benefit recall，不能靠多跑
epoch 或 validation threshold 解决。

epoch 2 `v26_audit_epoch_2.json` 通过 common/changed/new `1228/0/69`、69 optimizer states、
`825,997` numel、step `6110`，所有 tensors 和 moments finite/nonzero。epoch 1/2 均低于
protected V19，因此本次 run 的 8 个 checkpoint hardlink 已删除；两轮 metrics、diagnostics、
config、log、retention 与 audits 保留。后续不从 V26 checkpoint 继续训练，下一版仍应从
protected V19 重新设计 benefit 边界校准。

### 32.3 Optional row-wise boundary calibration smoke

V26 all-fallback 后增加了一个保持部署规则不变的可选损失：对每个 query 行聚合 candidate
 margin 的 log-mean-exp，并根据该行是否存在正 utility 分别施加跨零和保持负值的约束。行权重
 做 class balance，false-positive cost 仍由现有配置控制；默认
 `source_moe_gate_boundary_loss_weight=0.0`，因此旧配置和跨数据集迁移不会产生行为变化。该
 loss 只使用 candidate validity/utility，不引入 ScanRefer 专用阈值或 source 组合。

128-row、10-step smoke 的 weight `1.0` 结果为 boundary loss `0.6764`、positive-row ratio
`0.0985`、margin mean `-0.1673`、correction `0`。weight `4.0` 结果为 boundary loss
`0.6738`、positive-row ratio `0.0985`；完整 debug validation 的 predicted-positive ratio
仍为 `0`、max-margin mean `-0.1804`，REC `0.500000/0.4453125`，Mask
`0.500000/0.406250/0.3504906`。candidate oracle 为 `0.5859375/0.5390625`，所以失败点
仍是部署边界校准而非候选覆盖。两次 smoke 不足以支持正式训练，未启动 full run。

weight `4.0` checkpoint 已用 `scripts/audit_source_moe_checkpoint.py --profile v26` 审计，
结果为 `1228/0/69`、optimizer step `10`、69 states、`825,997` numel，全部 tensors/moments
finite/nonzero。低指标 smoke 的 7 个 `.pth` hardlink 在审计后删除，receipts 保留；protected
V19 与历史后处理 best 始终保持不变。后续若继续研究，应先让 boundary calibration 在独立
synthetic/训练 split 上产生稳定的正负 margin，再考虑完整 ScanRefer 或 Nr3D/Sr3D 训练。

实现复核发现 row-wise loss 不应再叠加 deployment false-positive cost：原调用把
`false_override_weight=2.0` 同时用于 candidate benefit 和 row fallback class，令约 10% 的正
行在零边界附近受到负向偏置。现改为 row boundary 内部只做 class balance，candidate benefit
likelihood 仍保留 false-positive cost。修正后的 weight `4.0` smoke（`1785825891`）将
validation max margin mean 从 `-0.1804` 改善到 `-0.1618`，但 10 steps 内仍无正 margin 或
correction，故不把它误判为可正式训练信号；审计仍为 V19->V26 `1228/0/69`，低指标权重已清理。
该修正后的全量测试为 `3035 passed, 3 warnings`。

为排除 smoke 步数过短，又运行了 5 epochs/128 rows（50 optimizer steps）。epoch 2 才出现
约 `0.19%` positive candidate，epoch 5 达到 predicted-positive `2.18%`、oracle recall
`9.09%`，但 precision 只有 `4.55%`、false-switch `31.82%`，debug REC
`0.500000/0.453125`，max-margin mean `-0.5826`。因此 row boundary calibration 能让少量
候选穿过零点，却没有形成可泛化的 source rerank，正式大规模训练没有启动。epoch-5
checkpoint audit 为 V19->V26 `1228/0/69`、optimizer step `50`；所有低指标权重已删除，只
保留五轮 receipts。

## 33. V27 进入条件：dense quality 可分，但风险边界需独立校准

V26 的 5-epoch boundary smoke 已证明把少量候选推过固定零边界会同时引入大量 harmful
switch，因此下一版不能简单提高 boundary loss 或继续延长 V26。对现有 epoch-71 candidate
cache 做的 scene-disjoint holdout 分析提供了更具体的结构证据：`mask_text_query_dice` 对
`IoU>0.25` 的候选 AUROC 为 `0.8383`、对 `IoU>0.50` 为 `0.6990`，`mask_foreground_ratio`
为 `0.6657/0.7080`，`rank_default` 为 `0.6519/0.6080`。因此候选质量表示是可学习的，失败
点是把质量预测转换成固定 `margin > 0` 动作时的 prior/risk 校准，而不是 backbone、候选覆盖
或训练轮数。

V27 的实现前置约束：

1. 先用当前 protected V19 重新生成 fresh-runtime train cache；旧 e71 cache 的 SHA 仅用于
   可分性研究，不得训练或发布 V27 artifact。
2. 训练 head 继续覆盖 fallback 与所有有效 candidate 的绝对 box/mask quality（threshold
   BCE、连续 IoU、tier/listwise），并报告 candidate-level AUROC/AP 与 row-level top-k oracle，
   避免只看稀疏 benefit label。
3. 部署动作单独使用 permutation-equivariant fallback-plus-candidates set head；quality head
   与 action head 解耦，固定 fallback logit 为 0，新增输出零初始化，保证 V19 第 0 步逐元素
   identity。
4. action loss 使用训练场景的 empirical prior 和 false-switch cost，但不把 class-balanced
   posterior 直接当作部署 posterior；每次 smoke 必须报告 positive-row coverage、selected
   action precision、oracle recall 和 max-margin 分布，任何 all-fallback/过切都停止，不进入
   9508-row formal run。
5. source validity、query permutation、single-stage ScanRefer/Nr3D/Sr3D 的缺源路径继续
   fail-closed；不重新引入固定源组合或 ScanRefer validation threshold。

V26 boundary smoke 的低指标权重已经按 inode 审计删除，正式 V27 只能从受保护 V19 初始化。

## 34. V27 uncertainty-quality risk 实测与下一版约束

实际 V27 复用了 V23 `cascade_v23_dense_quality_correction` action path，并将训练 objective
固定为 `cascade_v27_uncertainty_quality_risk`。dense quality head 的 Bernoulli variance 作为
query uncertainty，运行时采用
`risk_quality = predicted_quality - uncertainty_weight * uncertainty`；本轮 uncertainty weight
为 `0.5`。零初始化时各有效 query 的 uncertainty 一致，risk margin 精确为 0，因此 protected
V19 初始化仍是逐元素 identity。V19 checkpoint 缺省值与命令行显式值的恢复顺序已经修正，
cache 和 reranker provenance 也记录 uncertainty 配置。

V27 没有通过 smoke 进入 formal。10-step cost-aware 版本 run `1785833888` 的 switch
precision 仅 `8.00%`、false-switch `92.00%`；row-max boundary ablation run `1785834292`
进一步降到 `4.76%/95.24%`，因此 row-max loss 已回退。5-epoch/50-step 延长 smoke run
`1785834751` 的最好 debug REC/Mask 出现在 epoch 4，分别为
`0.5078125/0.4609375` 和 `0.5078125/0.4140625/0.3581983`，但最好 switch precision 也只有
`15.79%`，epoch 5 又回到 precision `10.00%`、false-switch `90.00%`、oracle recall
`13.33%`。这些是 128-row debug receipts，不能与 9,508-row V19 official 指标直接比较。

epoch-5 audit 严格通过 V19->V27 `1228/0/39`、39 optimizer states、`588,603` numel、step
`50`，所有模型 tensors 和 78 个 optimizer moments finite/nonzero；故失败不是没加载预训练、
optimizer 未更新或训练实际未运行。延长训练只重复暴露 overcut，未达到 formal 进入条件。本轮
8 个低指标 `.pth` hardlink 已物理删除，五轮 receipts 和 audit JSON 保留；protected V19 的
inode、8 links、只读权限和 SHA-256 未变。

下一版的网络约束据此收紧：

1. dense box/mask quality 和 adaptive source mixer 继续作为候选表示与排序监督，不直接决定
   fallback override。
2. 新增独立的 permutation-equivariant abstention/risk head，输出层零初始化；fallback 固定为
   0，确保 V19 第 0 步 identity。
3. risk head 使用训练场景的 row-level beneficial/harmful switch 分布做校准，并显式记录
   precision、false-switch、oracle recall 与 margin quantiles；固定部署边界不得由 ScanRefer
   validation threshold 选择。
4. smoke 的进入条件必须同时排除 all-fallback 与 overcut。未达到预注册的 precision/recall
   条件时不启动 9,508-row formal，也不以增加 epoch 代替结构诊断。
5. source validity 和缺源路径继续 fail-closed，使相同模块能够用于单阶段 ScanRefer、Nr3D 与
   Sr3D，而无需逐数据集搜索最强单源或 source 组合。

截至本轮，发布基线仍是 network-only V19 REC `0.5811948/0.4653976`、Mask
`0.5982331/0.4913757/0.4186131`；系统后处理最高仍为 REC `0.582878/0.486012`。

## 35. Selected-candidate abstention risk（V28）

V28 action/objective 为 `cascade_v28_selected_abstention_correction` /
`cascade_v28_selected_abstention_risk`。它保留 adaptive source mixer、dense quality evidence
和 permutation-equivariant pairwise candidate encoder，但明确拆开两个决策：pairwise head
只负责候选集合内部排序；新增 `SelectedCandidateAbstentionHead` 读取选中候选与 fallback 的
成对 evidence，只输出 row-level risk，决定是否允许该候选覆盖 V19。最终 candidate margins
按行构造成最大值严格等于 row risk，部署边界恒为 `max(margin)>0`，不搜索 validation threshold。

row head 最终线性层零初始化；V19->V28 migration 只允许新增 head 缺失，并拒绝错误 action /
objective。第 0 步完整 SourceMoE 输出逐元素等于 V19，query permutation、source validity、
缺源路径和 single-stage fail-closed 语义保持不变。生产合同为 V19 common/changed/new
`1228/0/75`、75 optimizer states、`876,174` trainable numel 和 150 个 Adam moments。
实现同步覆盖 launcher、main loss allowlist、checkpoint audit profile、cache/reranker provenance；
初始专项回归为 `245 passed`，compile 与 shell syntax 检查通过。

首次 smoke run `1785836663` 暴露外层 loss objective 白名单漏接，在首 batch 前按合同失败，
没有 optimizer step 或 checkpoint；修复后 10-step run `1785836950` 为 debug REC
`0.500000/0.453125`、Mask `0.500000/0.406250/0.3504906`、0 correction，audit 通过
`1228/0/75`、step `10`。这排除了“V28 未加载预训练”或“新增参数未进入 optimizer”。

5-epoch/50-step 延长 smoke run `1785837268` 的 REC@0.25 五轮均为 `0.500000`，REC@0.50
在 epoch 1--3 为 `0.4453125`、epoch 4--5 为 `0.453125`；Mask 始终为
`0.500000/0.406250/0.3504906`。前三轮 0 switch，后两轮均为 1 beneficial、0 harmful，故
precision `100%`、false-switch `0%`；oracle opportunity 为 13 rows，实际 recall 只有
`7.69%`。candidate oracle `0.58594/0.53906` 显示候选覆盖充足，但 candidate benefit target
只有约 `1.5%->3.8%`，低于 row target `10.16%`：pairwise ranking 没有把多数正收益候选
送给 row abstention head。V28 解决了 V27 overcut，却落在低 recall 的 undercut 侧。

epoch-5 audit 严格通过 `1228/0/75`、75 states、`876,174` numel、step `50`，模型和 optimizer
moments 全部 finite/nonzero。两个成功 smoke run 的全部 16 个低指标 `.pth` 已删除；metrics、
diagnostics、config、log、retention 与 audit JSON 保留。失败启动没有权重。protected V19 复核
仍为 inode `34391215`、8 links、mode `0444`、SHA-256
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。

V28 未通过 formal 进入条件，因此不直接扩到 70--80 epochs。下一结构必须联合优化 candidate
selection 和 row risk：候选排序需显式提高 row-oracle query 的入选率，row head 仍保持独立
零初始化、固定零边界，并分别约束 harmful-switch precision 与 oracle recall。只有 128-row
5-epoch smoke 同时保持低 false-switch 且显著提高 recall，才启动 9,508-row validation 的正式
训练；该门禁不依赖 ScanRefer 特有 threshold 或固定 source 组合，可原样迁移至单阶段
ScanRefer、Nr3D 和 Sr3D。

最终定向回归加入 adapter/evaluator/reranker/cache 后为 `339 passed in 19.73s`；V28 相关
Python compile 与 `scripts/train_scanrefer_source_moe.sh` shell syntax 检查均通过。

### 35.1 Full-data training verification

按完整训练要求，V28 正式两轮 run `1785838268` 已于 `2026-08-04 18:11 CST` 从 protected
V19 启动；目录为
`output/source_moe_v28_selected_abstention_train_v28/scanrefer/ssq_moe_e73_v28_selected_abstention_e2/1785838268/`，
tmux 为 `mcln_v28_selected_abstention_e1_e2`。合同为非 debug `36,665/9,508`
train/validation、batch `12`、两轮各约 `3,055` steps、new-head-only、LR `3e-4`、固定零边界。
V19 基于 epoch-71 主干且其 checkpoint 局部 epoch 为 2，所以这里的局部 epoch 1/2 是全局约
e74/e75 的新增 head 训练；不是从随机初始化只训练两轮，也无需把 gate 的局部 epoch 人为写成 74。

该 formal run 用于检验 50-step smoke 与 `6,110` full-data steps 的数量级差异，不能在结果前
预先视为发布候选。预期 epoch 1/2 receipt 窗口为 `19:10--19:20` / `20:15--20:30 CST`；
按窗口检查，最终同时审计 official REC、Mask、switch precision/recall、V19 tensor identity 和
optimizer step，再执行 global-best retention/低指标清理。

### 35.2 V28 失败分解与 V29 预注册分流

V28 epoch-5 smoke 可把 end-to-end recall 分成两个乘法瓶颈。13 个 row-oracle opportunities
中，当前 hard candidate selection 约只有 5 行选到正 utility query，机会捕获约 `38.5%`；
independent abstention 又只放行其中 1 行，条件放行约 `20%`，最终才是 `1/13=7.69%`。
因此只提高 row-head loss 权重会把错误候选推过边界并重现 V27 overcut；只延长 pairwise
regression 则没有直接约束“正收益候选概率质量”。

若 V28 formal 仍未刷新，V29 预注册为 counterfactual selected-risk，保持以下跨数据集合同：

1. candidate policy 仍为 permutation-equivariant query set head，但新增 positive-mass loss：
   对存在正 utility 的 row，直接最大化 softmax policy 落在任一正收益 candidate 上的概率和；
   对无正收益 row 不伪造候选标签。它与现有连续 utility listwise/regression 并行，使 top-1
   opportunity recall 成为显式训练目标，而不是 KL 排序的间接副产物。
2. abstention head 改为共享参数的 counterfactual candidate-conditioned risk：同一 row context
   分别与 top-k candidate 配对，训练时可监督每个候选被选中后的 utility；部署仍只 gather
   candidate policy 的 hard top-1 risk，不使用 oracle、GT 或 validation threshold。
3. risk 校准按 row 等权，而不是把 8 个候选扁平化后使用 candidate prior。正 opportunity row
   以 oracle-best 正候选和 hardest negative 构成一组，无 opportunity row 只用 policy-hardest
   negative；经验 row prior 显式恢复到 raw margin，使固定 `risk>0` 仍有跨 ScanRefer/Nr3D/Sr3D
   的统一含义，避免 V26 flattened-candidate all-fallback。
4. 所有新增输出层保持零初始化，最终 action margin 的行最大值仍严格等于 selected risk；V19
   第 0 步 identity、source validity、query permutation 和缺源 fail-closed 不变。
5. smoke 门禁分别报告 policy opportunity capture、conditional abstention precision/recall 和
   end-to-end switch precision/recall。只有前两层均改善且 harmful switch 不回升，才进入 formal。

该分流目前只是依据 V28 smoke 预注册的下一步，不会改变正在运行的 `1785838268` 进程或其
checkpoint 合同。若正式 full-data steps 已经把两层 recall 学起来，则优先保留更简单的 V28，
不为版本号本身增加模块。

为使后续复评直接验证上述分解，loss diagnostics 新增三个纯统计字段：
`source_moe_gate_policy_selected_positive_count`、
`source_moe_gate_policy_opportunity_capture_ratio` 和
`source_moe_gate_abstention_conditional_recall_ratio`。该改动不参与 loss、没有新参数，也不改变
正在运行的 formal 进程；专项回归为 `2 passed`。正式 checkpoint 可用更新后的代码复评并得到
两层 recall。随后完整 SourceMoE/迁移/audit/adapter/evaluator/reranker/cache 回归已复核为
`339 passed in 6.56s`。

### 35.3 V29 implementation handoff

The preregistered V29 branch is now wired as an isolated action/objective pair:
`cascade_v29_counterfactual_selected_correction` /
`cascade_v29_counterfactual_selected_risk`. `CounterfactualSelectedRiskHead` scores every
valid candidate with shared parameters during training, while deployment gathers only the hard
top-1 policy candidate. Its output layer is zero-initialized and the deployed candidate margin
has an exact row-wise maximum equal to the selected risk, preserving V19 identity at step 0.

The loss combines positive-candidate probability mass with row-balanced positive/hard-negative
counterfactual risk and explicit row-prior restoration. V19 migration, new-head-only optimizer
selection, launcher/provenance allowlists, and audit profile are complete; the contract remains
`1228/0/75`, 75 optimizer states, and `876,174` trainable parameters. Focused V29 tests pass.
Do not launch a formal V29 run until the V28 full-data result is audited and a short V29 smoke
confirms policy opportunity capture improves without harmful-switch regression.

### 35.4 V28 formal result

The two-epoch full-data V28 run `1785838268` used `36,665/9,508` train/validation rows and
`6,110` optimizer steps from the protected V19 initializer. Epoch 1 reached REC
`0.5810896/0.4652924` and Mask `0.5981279/0.4914809/0.4185229`; epoch 2 regressed to REC
`0.5805637/0.4647665` and Mask `0.5974968/0.4909550/0.4181306`. Both epochs had zero beneficial
and harmful switches and zero oracle-switch recall, so the V28 abstention head did not learn a
deployable correction and the target `0.59/0.49` was not reached.

Both checkpoints passed the `1228/0/75`, 75-state, `876,174`-parameter audit at steps `3055` and
`6110`. Low-metric epoch-2 and duplicate epoch-1 links were removed; only the epoch-1
`ckpt_best_mask_acc050.pth` remains because its Mask Acc@0.50 `0.4914809` is marginally above the
protected V19 `0.4913757`. Receipts, diagnostics, audit JSON, config, and logs remain available.
V29 should proceed only through the preregistered short smoke, using the protected V19 initializer,
not this under-recalling V28 checkpoint.

### 35.5 V29 smoke gate and deployment-aligned risk calibration

The first V29 smoke (`1785847541`) raised candidate opportunity capture to roughly `7/12`, but the
counterfactual selected-risk head still had only one beneficial switch and about `8--9%` conditional
recall. The cost=1 ablation (`1785848144`) reached one beneficial and zero harmful switches at best
(precision `100%`, recall `7.14%`), with epoch-5 REC `0.492188/0.453125`. Neither run satisfies the
formal-training gate. The cost=1 epoch-5 checkpoint passed the exact V29 audit and all nine temporary
checkpoint links were removed after the audit; receipts and diagnostics remain.

The V29 risk objective had a calibration mismatch: class-balanced training used
`candidate_risk + prior_shift`, while deployment used raw selected risk `>0`. The implementation now
keeps the prior-restored BCE and averages it with an unshifted auxiliary BCE over the same
counterfactual positives and hard negatives. This gives positive raw risk a direct gradient toward
the fixed deployment boundary without selecting a dataset-specific threshold. The new unit test
asserts the raw positive risk gradient is negative and raw negative risk gradients are positive.

Next gate: run a short V30/V29 calibration smoke from protected V19. Proceed to full ScanRefer only if
candidate opportunity capture and conditional abstention recall both improve while harmful switches
remain zero or below the existing precision guard. Reuse the same module and fixed boundary for
single-stage ScanRefer, Nr3D and Sr3D; do not choose a source combination or threshold separately per
dataset.

### 35.6 V30 raw-risk boundary smoke result

After adding the unshifted raw-risk BCE, smoke `1785850647` (protected V19, 128 rows, 5 epochs,
50 steps) produced REC `0.500000/0.460938` at best and Mask
`0.500000/0.406250/0.3506068`. Epochs 4--5 each had one beneficial and zero harmful switches,
so switch-count precision was `1/1=100%`, while conditional recall was only `8.33%`. The raw boundary
is now reached by a small number of candidates, but candidate policy ranking remains the dominant
bottleneck. Do not promote this smoke to full-data training.

The epoch-5 checkpoint passed the V29 audit (`1228/0/75`, 75 states, `876,174` trainable parameters,
step `50`) and all nine temporary `.pth` links were removed. A first launch failed only because debug
evaluation retained the formal expected sample count (`9508` instead of `128`); its step-10 checkpoint
was audited and removed as well.

Next architecture iteration should improve positive candidate policy selection/ranking before adding
more boundary pressure. Keep the shared counterfactual risk head, zero initialization, source-validity
fail-closed behavior and raw deployment boundary unchanged so the design remains reusable across
ScanRefer, Nr3D and Sr3D.

### 35.7 V31--V33 hard-policy and raw-only objective

V31 added a hard top-1 positive-vs-negative margin beside the positive probability-mass objective, so
the query selected by the deployed policy receives direct ranking pressure. Its epoch-5 whole-eval
diagnostics captured `9/14=64.29%` candidate opportunities, but counterfactual risk released only
`1/14=7.14%`; REC was `0.500000/0.453125`, with one beneficial and zero harmful switches. V32 removed
prior-shift logits from the optimized counterfactual classification objective and retained them only
as diagnostics. V33 also removed the duplicated selected-row shifted BCE from the V29 aggregate loss.
Both ended at `9/13=69.23%` candidate capture, `1/13=7.69%` conditional recall, one beneficial and zero
harmful switches, and REC `0.500000/0.453125`.

All three epoch-5 checkpoints passed the unchanged V29 audit contract (`1228/0/75`, 75 optimizer
states, `876,174` trainable parameters, step `50`) and their temporary `.pth` files were removed. These
runs supersede the historical V30 description that averaged shifted and raw BCE: optimized
counterfactual classification now uses only the deployment-aligned raw-zero boundary, while the
prior-shift loss is diagnostic.

The remaining supervision gap was candidate-level. The shared risk head scored every candidate, but
only the oracle-best positive candidate in each row received a positive risk target. A hard policy can
select another beneficial candidate from the same row, leaving its risk untrained and negative even
when row-level candidate opportunity capture is high.

### 35.8 V34 dense-positive supervision result

V34 closes that gap by supervising every beneficial candidate in a positive row with raw-zero BCE and
utility regression. Candidate losses are averaged within each row before aggregation, so a row with
more beneficial queries does not acquire a larger empirical weight. Negative supervision remains the
policy-hardest negative per row. This changes no parameters or deployment behavior and preserves the
V19 identity, permutation, validity, and fail-closed contracts. The new multi-positive BCE/regression
gradient test passes; the focused SourceMoE, integration, and audit suite is `253 passed`.

The protected-V19 smoke `1785853677` used 128 rows, five epochs, and 50 optimizer steps. Epoch-5
whole-eval diagnostics showed candidate capture `9/15=60.00%`, conditional recall `1/15=6.67%`, one
beneficial switch, zero harmful switches, and `100%` switch precision. Batch-weighted training stats
reported `56.82%/9.09%`. End-to-end REC was `0.500000/0.453125`; Mask was
`0.500000/0.406250/0.3504906`. Dense positive coverage therefore did not improve both policy capture
and risk recall relative to V33 and must not be promoted to a full-data or 70--80 epoch run.

The epoch-5 artifact passed the exact V29 audit (`1228/0/75`, 75 states, `876,174` parameters, step
`50`). All nine V34 checkpoint links were then removed while audit, metrics, diagnostics, config, log,
and retention receipts were kept. The protected V19 and the three `0.582878/0.486012` post-processing
components remain read-only and unchanged.

### 35.9 V35 utility-cost-once result

V35 removed a duplicated risk cost from the dense counterfactual objective. Transition break cost is
already represented in `decision_utility`, so raw-boundary classification and utility regression no
longer multiply the same false-positive cost again. Positive and negative regression are averaged as
two classes rather than pooled by candidate count. The diagnostic prior-shift objective remains
excluded from optimization, and deployment still compares raw selected risk with zero.

The protected-V19 smoke `1785854661` captured `8/13=61.54%` policy opportunities. It released six
switches: two beneficial and four harmful, giving `15.38%` oracle-switch recall, `33.33%` precision,
and a `66.67%` false-switch rate. REC was `0.500000/0.453125`; Mask was
`0.500000/0.406250/0.3518690`. Removing the duplicated cost improved release recall but made the
fixed boundary unsafe, so V35 is not eligible for full-data training. Its epoch-5 checkpoint passed
the unchanged V29 audit contract and all temporary checkpoint files were removed.

### 35.10 V36 symmetric deployment gap result

V36 added a symmetric safety gap around the unchanged raw-zero deployment boundary. Positive
candidate BCE is evaluated at `risk - temperature`, while the policy-hardest negative BCE is
evaluated at `risk + temperature`. This pushes both classes away from an ambiguous zero with equal
margin pressure without introducing a dataset-specific inference threshold. Dense positive
row-normalization, utility-cost-once regression, prior-shift diagnostics, V19 identity, permutation,
source validity, and fail-closed behavior remain unchanged. The focused regression suite is
`255 passed`.

The five-epoch smoke `1785855327` captured `8/13=61.54%` opportunities but released only one
beneficial and zero harmful switches: end-to-end recall `1/13=7.69%`, precision `100%`. Whole-eval
counts are authoritative; batch-weighted log ratios `48.48%/9.09%` are not used for promotion. REC
was `0.500000/0.453125`; Mask was `0.500000/0.406250/0.3504906`. The symmetric gap repaired V35's
precision failure but restored the under-release failure, so V36 also fails the full-data gate.

The epoch-5 artifact passed the exact audit (`1228/0/75`, 75 optimizer states, `876,174` trainable
parameters, step `50`, finite/nonzero model and moments). All eight checkpoint files actually present
in the run were removed after audit; receipts and diagnostics remain. The protected V19 inode,
read-only mode, eight links, and SHA-256 remain unchanged.

The V35/V36 pair narrows the next design requirement: a single global cost or symmetric margin trades
recall directly against harmful-switch precision. The next iteration should estimate candidate-level
calibration or uncertainty from shared source/query evidence and adapt its confidence internally,
while retaining one fixed deployment rule across ScanRefer, single-stage ScanRefer, Nr3D, and Sr3D.
Do not promote another run merely by extending epochs unless a short smoke first improves beneficial
release beyond one switch without reintroducing V35-like harmful switches.

### 35.11 V37--V38 decomposed evidence heads

V37 decomposed candidate risk into benefit and hazard evidence with deployment
`ReLU(benefit) - ReLU(hazard) > 0`. Positive candidates primarily supervise benefit and hard
negatives supervise hazard; the final layer is zero-initialized for exact V19 step-0 identity. Smoke
`1785857843` ended at REC `0.500000/0.4453125`, Mask
`0.500000/0.406250/0.3506068`, and candidate capture `8/13`, but all five epochs produced zero
beneficial, harmful, and total switches. The decomposition was safe but unusably conservative.

V38 interpreted the two outputs as complementary log-odds and deployed `benefit - hazard > 0`, using
symmetric focal classification and normalized `+1/-1` net-risk regression. Smoke `1785859359`
ended at REC `0.500000/0.4609375`, Mask `0.500000/0.406250/0.3504906`, capture `7/12`, and one
beneficial/zero harmful switch (`8.33%` recall, `100%` precision). Its final biases were approximately
`-0.0041518/+0.0041518`; equal weight norms and opposite weights show that the two outputs collapsed
to a stricter single scalar instead of learning complementary evidence.

Both runs passed the exact `1228/0/75`, 75-state, `876,303`-parameter, step-50 audit. V37's eight and
V38's nine checkpoint files were removed after audit; all receipts remain. Neither branch qualifies
for full-data or 70--80 epoch training.

### 35.12 V39 hazard-residual correction

V39 restores V35's release-capable raw gain path and constrains hazard to a non-negative residual
veto. The fixed deployment rule is `candidate_gain - ReLU(candidate_hazard) > 0`. Gain uses raw-zero
BCE plus utility regression; hazard uses focal classification to suppress unsafe candidates. Break
cost remains represented only once in `decision_utility`. No dataset-specific threshold or fixed
source combination is introduced, and V19 identity, query permutation, source validity, and
fail-closed checkpoint contracts remain intact.

The isolated action/objective pair is `cascade_v39_hazard_residual_correction` /
`cascade_v39_hazard_residual_risk`. Model outputs, loss wiring, MCLN construction, CLI/launcher,
reranker provenance, V19 migration, new-head-only optimizer selection, and audit profile are complete.
Focused SourceMoE/integration/audit tests report `273 passed`; the full suite reports `3067 passed`
plus three pre-existing geometry-cache provenance failures caused by a test backbone/manifest
mismatch outside the changed files. Python compilation and shell syntax checks pass.

Protected-V19 smoke `1785861677` completed five epochs, 128 rows, and 50 optimizer steps. Epochs 1--4
held REC `0.500000/0.4453125` and Mask `0.500000/0.406250/0.3504906` with zero correction switches.
Epoch 5 released one beneficial and zero harmful switch, giving `1/13=7.69%` recall and `100%`
precision, but REC was `0.4921875/0.4453125` and Mask fell to
`0.4921875/0.3984375/0.3426781`. Candidate opportunity capture reached `9/13`, so the residual veto,
not candidate availability, remained the limiting stage. Final gain/hazard biases were
`+0.0004505/+0.0053369`, with weight norms `0.01451/0.04859`; hazard was about 3.35 times stronger.

The epoch-5 artifact passed `1228/0/75`, 75 optimizer states, `876,303` parameters, and step `50`, with
finite/nonzero model and moments. Because beneficial release did not exceed the one-switch V36/V38
baseline, V39 fails the formal-training gate. All eight checkpoint files were removed while audit,
metrics, diagnostics, config, log, and retention remain. The protected V19 inode, eight links,
read-only mode, and SHA-256 are unchanged.

The highest results therefore remain post-processed REC `0.582878/0.486012`, network-only V19 REC
`0.5811948/0.4653976`, and V19 Mask `0.5982331/0.4913757/0.4186131`. V37--V39 were all initialized
from the protected pretrained V19 and had verified optimizer updates, so their under-release is not a
missing-pretraining or short-run loading error. The next branch should calibrate relative gain/hazard
learning strength or their shared evidence scale before another formal run, using whole-eval
beneficial/harmful switch counts as the promotion gate.

### 35.13 V19 post-processing probe and joint REC/mask target

The V19 parent/geometry experiment is a one-off diagnostic, not a replacement research branch. Old
sidecars are cryptographically bound to epoch 71, so the fair probe regenerates V19 candidate caches,
re-trains the parent and geometry scorers from V19 train data only, and evaluates exactly 9,508 val
rows once. Geometry extraction now has an explicit portable-provenance mode that binds checkpoint
SHA/epoch, complete split caches, the train-only audit panel, and its train-cache path. The original
epoch-71 path remains strict. The missing `source_moe_gate_uncertainty_weight` provenance field was
also restored; the focused extractor suite is `28 passed`.

The first full-152D hierarchical pair-evidence probe used 29,349 fit and 7,316 held-out train rows.
Expected utility released `45 beneficial / 36 harmful` switches; the 0.25-sigma lower bound released
`13/9`, and 0.50 sigma released only `1/0`. Projection features improve separability over the scalar
probe, but this still fails the formal-training gate. Break-cost and capacity controls are being run
in parallel before choosing the next shared-evidence implementation.

The follow-up signed-utility evidence probe completed all five scene-modulus folds without validation
access. Train-calibrated conformal coverage at 95%, 99%, and 100% released aggregate beneficial/harmful
counts of `592/794`, `193/197`, and `2/1`; three of the five 100% folds released no beneficial switch.
A fixed 1.5-scale lower bound was also unsafe at `141/139`. This is the same coverage collapse in a
different parameterization, so neither evidence probe is eligible for the V40 network path and no
validation threshold search is allowed.

The 256-expression V19 train audit measured default `0.74219/0.48047`, regressed Top-16 oracle
`0.97656/0.88672`, and combined seven-variant geometry oracle `0.98047/0.92578`. Portable extraction at
`batch/workers/shard = 36/4/252` exposed two deeper contracts that the CLI-only test had missed: the
cache validator still fixed `12/2/252`, and batch-dependent top-k ties changed fresh compact query
identity. The manifest now accepts strictly typed positive runtime values with shard/batch alignment,
while non-portable CLI remains fixed at `12/2/252`. Portable extraction canonicalizes query, box, and
default Top-1 identity from the complete bound base cache before target parity, then extracts geometry
from those exact checkpoint queries. This preserves query-consistent box/mask identity. The focused
cache and durability suite is `175 passed`; full V19 train/val geometry extraction restarted on GPU0/1
at `36/4/252` and remains a one-off diagnostic.

The deployment target now explicitly includes the official segmentation reference: REC
`0.59/0.49`, Mask Acc@0.50 `0.5070`, semantic mIoU `0.4472`, with Mask Acc@0.25 not below V19's
`0.5982331`. The existing joint adapter already supplies box tiers, mask tiers, continuous mask IoU,
and query-specific text/query logit calibration, but its v1 selector regressed train calibration and
must not be promoted. V40 will reuse that supervision/cache path with one query-consistent shared
evidence trunk: conservative box evidence vetoes unsafe switches; mask tier and IoU evidence rank the
remaining candidates; the chosen box and mask always share the same parent query. Risk strength is
fit only on scene-disjoint train calibration, with one fixed zero deployment boundary across
two-stage ScanRefer, single-stage ScanRefer, Nr3D, and Sr3D.

### 35.14 Geometry throughput and multi-GPU launch contract

The geometry extraction bottleneck was kernel granularity rather than capacity. Model-forward phases
can reach `88--100%` SM utilization, whereas the old per-sample, per-query quantile and validity loops
drop the geometry phase to `0--35%`. Raising the batch to 63 or 84 consumed approximately 29.0 or
32.5 GB but did not improve first-shard wall time. Query AABB reductions are now vectorized within a
scene, geometry outputs make one batched device-to-host transfer, and every shard reports elapsed time
and rows/s. The isolated A100 quantile segment fell from `10.55ms` to `1.33ms`.

The first two optimized 252-row shards were compared field by field with the original production
cache. Boxes, geometry features, and IoUs all had maximum difference `0.0`; identity, validity,
rejection, parity, and provenance fields were bit-exact as well. The focused suites remain `42 passed`
for mask geometry and `175 passed` for cache/durability. Consecutive-shard steady-state measurements
were `18.90s` for batch36, `19.41s` for batch42, and `17.42s` for batch63, versus a `29.46s` tail mean
for the original batch36 producer. Batch63 was not adopted because immutable production metadata was
already bound to batch36 and changing it would discard 52 completed shards. The train producer safely
resumed at row `13,104/36,665` with vectorized batch36. The complete 9,508-row val sidecar has 38 shards
and content digest
`75e34f1f57062ddea8928ae6cfd41f557d1b69a67cd9f41421dfd7b1056497b6`.

The SourceMoE launcher also no longer hard-codes one distributed rank. It derives
`--nproc_per_node` from `CUDA_VISIBLE_DEVICES` unless `NPROC_PER_NODE` is explicitly set, and rejects
non-positive values or counts larger than the visible device set. Dry runs verify automatic four-rank
and explicit two-rank launches. Full training can therefore use all four GPUs without silently
falling back to one rank; independent smoke variants can still be assigned one per GPU when preserving
the original per-device optimization trajectory is more useful than one larger DDP batch.

The next query-wise mask-calibration change is an end-to-end interface migration, not only a new MCLN
head. A zero-initialized query residual may refine the V19 sample-level alpha, but the grounding
evaluator must reshape `[Q]` weights to `[1,Q,1]`, mask loss must gather alpha with the Hungarian
query indices, candidate/source adapters must not collapse it back through `.mean()`, and geometry
extraction must gather by original query identity. Full training is gated on identity and shape tests
across all of these consumers so training, official mask evaluation, and query-consistent geometry use
one fusion definition.

### 35.15 Query-wise mask fusion implementation

`QueryMaskFusionCalibrator` now predicts a bounded per-query residual around the protected V19 scalar
alpha from final query features, pooled language context, and normalized predicted boxes. Its output
layer is zero-initialized, so enabled step-0 inference is bit-exact to V19. Base evidence and alpha are
detached during the first head-only phase; only 92,429 calibrator parameters are optimized. Loading a
plain V19 checkpoint permits exactly the 12 new calibrator state keys to be missing and rejects any
other state difference.

All mask consumers use `models/mask_fusion.py` as the single scalar/query-wise broadcasting contract.
Hungarian supervision gathers weights on matched detector query IDs; official evaluation, SourceMoE
mask ranking, mask-text source scoring, REC candidate statistics, geometry extraction, and joint cache
labels preserve the same IDs and fuse logits before sigmoid. Focused identity, permutation, batched
shape, detach, gradient, and existing integration suites pass (`236 passed`; separate launcher and
optimizer coverage `148 passed`).

The dedicated launcher supports both independent single-GPU sweeps and automatic multi-rank DDP.
Debug annotation parsing now truncates before scene-graph parsing, and the legacy per-sample NumPy mask
loss weight was replaced by an on-device Torch calculation (maximum float32 difference
`2.3841858e-07`). A batch-56 capacity run used about 25.4GB and completed two train batches in 18.8s;
the comparable three-way smoke uses batch64 so all 128 rows are trained each epoch while occupying
roughly 29GB per A100. GPU1--3 test learning-rate/residual-cap variants while GPU0 completes the one-off
V19 geometry cache. Full-data training remains gated on end-to-end REC/mask non-regression plus finite,
nonzero optimizer state and demonstrably query-varying alpha.

### 35.16 Frozen-state correction and four-GPU full run

The first completed head-only smokes were invalid for promotion. Although their optimizers contained
only the 12 calibrator states, the outer `model.train()` also put frozen BatchNorm modules into training
mode; the exact V19-to-candidate audit was `1228/204/12` common/changed/new. Query-only mode now calls
`eval()` on the complete model and re-enables training only on `query_mask_fusion_calibrator`. A real
backward and Adam update test verifies that frozen BatchNorm and Dropout remain in evaluation mode and
that every non-calibrator state tensor remains bit-exact. The focused mask-fusion suite passes 7 tests,
the training/checkpoint integration suite passes 117 tests, and Python compilation succeeds.

Four corrected batch-64, five-epoch, 128-row smokes used approximately 32.66GB per A100 and produced
epoch-5 residual mean/max/query-std of `0.000823/0.001236/0.000144` for `1e-4/0.10/d0`,
`0.005810/0.008260/0.000746` for `3e-4/0.10/d0`, `0.021414/0.032972/0.003425` for
`5e-4/0.15/d0.1`, and `0.137633/0.180511/0.005342` for `1e-3/0.20/d0.1`. Every checkpoint now passes
the exact `1228/0/12` audit: all 12 new tensors, 12 Adam states, and 24 moment tensors are finite and
nonzero at optimizer step 10. The small debug panel selected epoch 1 for all variants at REC
`0.500000/0.445312` and Mask `0.500000/0.406250/0.350516`; those values are a stability screen, not an
official-validation comparison.

All invalid and non-promoted smoke checkpoint files were unlinked while logs, configs, metrics, and
retention records were preserved, recovering free disk from about 1.5GB to 6.0GB. The protected V19
artifact remains mode 0444 with eight hard links and SHA-256
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`.

The complete V19 geometry train cache has published 36,665 rows. Its fixed
`hidden=256/dropout=0.1/lr=3e-4/seed=0` train-only scorer selected epoch 27 with calibration
`0.95310/0.91697` and geometry weight 0.80. A separate immutable V19 one-shot claim consumed all 9,508
validation rows exactly once: the frozen parent scored `0.580143/0.465713`, while parent plus geometry
scored `0.579617/0.482436`. Geometry therefore lost `0.000526` at Acc@0.25 and gained `0.016723` at
Acc@0.50, but remained below the historical post-processed `0.582878/0.486012`. Selection, claim, and
record are mode 0444; the frozen record SHA-256 is
`8d871a493c24b0723cf7a9bd8e004fc2c6c688815cb8e0d8812b94063608a11b`. This diagnostic geometry branch
is closed without validation-driven retuning.

In parallel,
`mcln_qmask_full80_4gpu` has started four-rank DDP with per-rank batch64, global batch256, 80 epochs,
and `lr=1e-4/max_delta=0.10/dropout=0`. Its output is
`query_mask_fusion/scanrefer/qmask_full80_b64x4_lr1e4_delta010_d0_fixedbn/1785875641`. Promotion still
requires official REC/mask metrics, non-saturated query variance, and the exact `1228/0/12` state
contract at retained checkpoints.

### 35.17 Four-GPU utilization and exact query-head resume

All four DDP ranks are alive and hold approximately `35.4/31.8/32.0/38.0GB`; the earlier impression
that only GPU2 was active came from bursty point-in-time sampling, not a missing process. A continuous
15-second sample measured mean SM utilization of about `33.5%/37.7%/29.9%/38.7%` (roughly 35% overall),
with ranks alternating between idle and fully busy. GPU3 is already close to the 40GB limit, so raising
the per-rank batch above 64 is not the safe throughput lever.

The launch environment had inherited both `OMP_NUM_THREADS=40` and `MKL_NUM_THREADS=40`. Sixteen total
DataLoader workers consequently owned 42 threads each and consumed roughly 2.1--2.3 CPU cores per
worker on a 40-core allocation. The resulting hundreds of runnable threads, 29--31% system CPU, and
millions of context switches per second explain the irregular GPU feed. The query-mask launcher now
sets OMP, MKL, OpenBLAS, and NumExpr from one `CPU_THREADS_PER_PROCESS` value defaulting to 1, disables
tokenizer parallelism, and makes worker/prefetch counts explicit. The first candidate used eight
workers per rank with prefetch factor 1, preserving the previous number of outstanding batches while
replacing oversubscribed thread pools with 32 single-threaded workers. Train workers can optionally persist across epochs, each worker also calls
`torch.set_num_threads(1)`, and query-only DDP disables the unnecessary unused-parameter graph walk.
The container cgroup is capped near 288GB despite the host reporting 629GB, so production resume leaves
persistent workers disabled until an observed cgroup peak proves that idle train workers can coexist
with validation workers safely.

Autograd inspection also shows that the query calibrator receives gradients only from
`adaptive_weight_loss_mask` and `adaptive_weight_loss_dice`. Box, classification, semantic, and
correspondence terms consume frozen outputs; SourceMoE's mask-IoU target is explicitly evaluated under
`no_grad()`. Query-only training therefore has a gradient-equivalent fast path that performs one
last-layer Hungarian match and computes only those two fusion losses. It skips six redundant decoder
matching passes, frozen detector/semantic losses, correspondence geometry, and MoE ranking diagnostics.
A matched-query unit fixture reports `torch.equal` for both full/fast losses and their adaptive-weight
gradients; a second test proves that the full criterion is not invoked and that mask scaling reaches the
same gradient. Loss, SourceMoE integration, and checkpoint suites total `190 passed`.

An exact `query_mask_fusion_resume_optimizer` path was added before changing the live run. It is valid
only for query-mask-only non-eval training without `reduce_lr`; it requires the saved enable flags,
learning rate, hidden size, dropout, and max delta to match exactly, requires the requested epoch to be
checkpoint epoch plus one, and refuses missing optimizer or scheduler state. Exact restoration,
non-contiguous-epoch rejection, and configuration-drift rejection pass together with the existing
checkpoint and mask tests (`28 passed`); Python compilation and shell syntax also pass. Epoch 1 remains
uninterrupted until its atomic checkpoint and full 9,508-row validation finish. The run can then resume
at epoch 2 under the corrected CPU settings, after which measured epoch throughput and utilization will
decide whether the worker count needs another adjustment. No new model metric is claimed by this
throughput-only change.

The original process exited during epoch-1 validation before writing a 9,508-row metrics receipt; it
was not manually interrupted, all ranks released, and the cgroup reports no kernel OOM kill. The atomic
epoch-1 checkpoint is valid under the new `qmask` audit profile: `1228/0/12`, 12 optimizer states,
92,429 parameter elements, Adam step 143, and finite/nonzero model tensors and 24 moments. It is valid
for exact epoch-2 continuation, but epoch 1 makes no metric claim because validation was incomplete.
The resumed launcher mirrors stdout/stderr to a durable log so a repeated evaluator failure is
diagnosable.

The first fast resume captured the missing failure evidence. Eight workers per rank simultaneously
built batch-64 point-mask payloads, increased cgroup `memory.max` events from 52,737 to 60,283, and one
rank-3 DataLoader worker was killed before the first batch. The failed run contains only config/log
receipts and no checkpoint. Production now uses four single-threaded workers per rank with prefetch 1,
halving in-flight batches relative to the old `4 x prefetch 2`; persistence remains disabled. The
expected speedup now comes from gradient-equivalent loss elimination and removal of thread-pool
oversubscription, not a larger host-memory queue.

The production continuation is
`qmask_full80_b64x4_lr1e4_delta010_d0_fastresume_w4p1/1785879774`. All four ranks report exact
optimizer/scheduler restoration and continue at epoch 2. The first 50 steps after dataset readiness
took 238 seconds versus 851 seconds in the old run, a 3.58x throughput improvement; steady state is
approximately 3.9--4.0 seconds per step. A 15-second sample measured per-GPU mean SM utilization of
`64.1%/63.9%/64.1%/75.7%` (67.0% overall), versus about 35.0% overall before the changes, while device
memory remains 33.8--38.1GB. The cgroup is using about 190GiB of 288GiB and its `memory.max` count has
stayed at 60,283, so workers and prefetch will not be raised further. Fast-path logs intentionally show
only the two adaptive mask losses as nonzero; all zero-valued detector/MoE losses are frozen terms that
were proven not to contribute query-head gradients.

### 35.18 Epoch-2 receipt and confirmed resource ceiling

The fast-resumed epoch 2 completed in 583.21 seconds and published a complete 9,508-sample receipt.
Learned-selector REC Acc@0.25/0.50 is `0.580669/0.464767`; Mask Acc@0.25/0.50/mIoU is
`0.597602/0.491060/0.418006`. These values do not exceed the protected V19 network checkpoint
(`0.581195/0.465398` REC and `0.598233/0.491376/0.418613` Mask), so epoch 2 is trajectory evidence and
does not replace V19. Epoch 3 started automatically and the 80-epoch run remains active.

The epoch-2 `qmask` checkpoint audit passes at epoch/Adam step `2/286`: common/changed/new state is
`1228/0/12`, and all 12 optimizer states, 92,429 parameter elements, and 24 moment tensors are finite
and nonzero. The epoch, last, and five metric-best filenames are seven hard links to one 605MB inode;
retention therefore preserves every metric alias without consuming seven checkpoint copies.

Near the end of validation, instantaneous SM utilization reached `97%/100%/81%/79%`, with
`37.4/38.1/33.9/38.0GB` allocated. Per-rank batch64 (global 256) therefore leaves too little headroom
to increase batch size safely. Eight workers per rank already caused a cgroup-memory worker kill, so
the production setting remains four workers per rank with prefetch 1. Brief zero-utilization samples at
epoch or validation transitions are input/synchronization phases and are not evidence of a missing DDP
rank.

### 35.19 Epochs 3--5 and V41 joint query quality reranking

The four-rank query-mask run remains uninterrupted and has entered epoch 6. Epoch 3 preserved the
epoch-2 REC and mask threshold hit counts exactly, with only mIoU moving from `0.4180057` to
`0.4180083`. Epoch 4 produced REC `0.580669/0.464661` and Mask
`0.597602/0.491060/0.4180105`; epoch 5 preserved the threshold metrics and reduced mIoU slightly to
`0.4180094`. Retention therefore keeps epoch 2 for four threshold bests, epoch 4 for mask mIoU, and
epoch 5 only as latest. Epoch 3 was automatically unlinked after no best alias referenced it.

The epoch-4 qmask audit passes at Adam step 572 with exact `1228/0/12` common/changed/new tensors,
12 optimizer states, 92,429 parameter elements, and finite nonzero moments. The protected V19 remains
mode 0444 with eight hard links and SHA-256
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`. Current best claims therefore
remain V19 network REC `0.581195/0.465398`, V19 Mask `0.598233/0.491376/0.418613`, and historical
post-processed REC `0.582878/0.486012`.

V41 adds `JointQueryQualityReranker` above the trained V19 SourceMoE output. It consumes each query's
152-dimensional rich candidate feature plus its normalized parent rank and row-standardized parent
score, applies query-set
self-attention and a shared FFN, and jointly predicts box 0.25/0.50 hits, box IoU, mask 0.25/0.50 hits,
and mask IoU. A zero-initialized bounded residual directly reranks all queries around the protected
`selected_source_scores`; enabled step-0 scores are bit-exact to V19. Its target quality uses a box-tier
stride of four, so no mask gain can promote a lower box tier over a higher one; mask evidence only
orders candidates within the transferable box-first constraint.

Joint-only training now has a gradient-equivalent fast loss. It computes only last-layer query box
IoUs, evaluator-consistent fused mask IoUs, and the V41 listwise/multitask/anchor objectives, skipping
proposal plus six decoder layers of frozen Hungarian, detector, semantic, and mask losses. Full/fast
tests compare the total, all four V41 sub-losses, and every trainable parameter gradient. The broader
regression suite passes 363 tests and the focused V41/audit suite passes 144 tests.

A real CPU construction against the protected V19 confirms 1,248 current state tensors with exactly
20 expected V41 keys missing, no unexpected keys, and bit-exact loading of all 1,228 V19 tensors. The
20 V41 parameter tensors contain 153,531 parameters; legacy optimizer and scheduler state remain
untouched in joint-only initialization. The audit tool now has a `v41` profile requiring
`1228/0/20`, 20 Adam states, and 153,531 optimized elements. PointNet initialization maps its legacy
CUDA checkpoint to CPU before normal rank placement.

The V41 launcher uses four-rank DDP, per-rank batch64, four single-threaded workers, prefetch 1, and
non-persistent train workers. The earlier eight-worker configuration is intentionally forbidden as a
default because it caused a real cgroup-memory worker kill. GPU0 already peaks near 37GB and every rank
repeatedly reaches 100% SM, so neither batch size nor host prefetch is increased. V41 GPU training waits
behind the user-requested uninterrupted 80-epoch qmask run; its first real gate is a one-epoch debug
checkpoint with finite nonzero optimizer moments, nonzero query-wise residual variance, exact frozen
V19 state, and evaluator use of V41 scores.

### 35.20 Epoch-6 receipt and automatic V41 smoke handoff

Epoch 6 published a complete 9,508-row qmask receipt. Learned-selector REC Acc@0.25/0.50 is
`0.580669/0.464661`; Mask Acc@0.25/0.50/mIoU is `0.597602/0.491165/0.418021`. Epoch 2 remains the
retained best for both REC thresholds and Mask@0.25, while epoch 6 now owns Mask@0.50, mask mIoU, and
latest. The protected V19 remains the network-level reference at REC `0.581195/0.465398` and Mask
`0.598233/0.491376/0.418613`; the qmask trajectory does not replace it. Epoch 7 started automatically
and all four ranks remain synchronized at approximately 3.85 seconds per training step.

V41 now exports residual absolute mean, residual absolute maximum, and cross-query residual standard
deviation through the normal evaluator aggregation. These diagnostics distinguish a useful query
reranker from a head that only adds one sample-level offset. Debug mode defaults to 128 validation
rows, while non-debug mode defaults to the official 9,508-row ScanRefer validation set. The focused
summary, V41, and checkpoint-audit tests pass 29 cases, and the summary CLI compiles successfully.

`scripts/queue_v41_smokes_after_qmask.sh` provides an event-driven handoff without polling the training
log. It waits on the current qmask process, validates the final epoch-80 receipt/checkpoint and the
protected V19 digest, then starts four independent three-epoch V41 debug variants concurrently, one
per GPU at batch64. Each result must pass the `v41` checkpoint profile and show finite, nonzero residual
mean and cross-query variance. `scripts/summarize_v41_smoke_panel.py` writes the final panel atomically.
The initial queue stopped after smoke selection. After the quality-coupled revision passed the expanded
regression gates, the handoff was upgraded to launch one pre-registered formal configuration rather
than select hyperparameters from the 128-row debug metrics; the exact contract is recorded below.

### 35.21 Quality-coupled V41 deployment scoring

The pre-smoke audit found that the first V41 draft trained six absolute quality outputs only as an
auxiliary task while an independent residual head exclusively controlled deployed Top-1 scores. It
also retained parent rank but discarded the parent score spacing that indicates whether V19's current
choice is decisive or ambiguous. Both gaps reduce the value of shared box/mask supervision and make
unnecessary switches harder to distinguish from useful ones.

The revised module adds a row-standardized parent-score channel alongside parent rank. Box and mask
threshold outputs use an ordinal factorization, enforcing `P(IoU>0.50) <= P(IoU>0.25)` by construction.
Their six-task joint quality is centered over valid queries and contributes directly to the bounded
residual logit together with the free residual head. Zero-initialized quality weights produce one
constant quality for every valid query, so the centered term is exactly zero and step-0 deployment
remains bit-identical to V19. The mask utility weight is restricted below 0.8, the exact bound needed
to prevent the maximum within-tier mask utility from crossing a box tier.

The additional standardized input changes the projection width from 153 to 154 and adds only 130
parameters. V41 still has 20 state tensors; its parameter and optimizer contract is now 153,531.
Tests cover ordinal nesting, score identity, invalid-query masking, permutation equivariance, and direct
listwise gradients into both quality and residual heads even when auxiliary quality loss is disabled.
The focused, integration/evaluator, and checkpoint groups pass `32`, `136`, and `26` tests respectively;
Python compilation and both launcher syntax checks pass. The active qmask run is untouched, and the
event-driven four-GPU smoke queue will consume this revised architecture. If all four smoke receipts,
20-state/153,531-parameter audits, and nonzero residual-variation gates pass, it immediately launches
the fixed formal configuration on four ranks: batch64 per rank, 80 epochs, learning rate `3e-4`,
dropout `0.1`, mask weight `0.25`, quality-score weight `1.0`, temperature `0.25`, and anchor weight
`0.5`. This choice is pre-registered and independent of debug validation ranking. Any failed smoke
gate prevents formal training; a completed run must additionally pass the epoch-80, Adam-step-11,440
V41 checkpoint audit.

### 35.22 First all-metric qmask improvement at epoch 7

The complete epoch-7 receipt improves every qmask trajectory metric: REC is
`0.581090/0.465187`, and Mask is `0.598128/0.491271/0.418549`. Retention therefore points all five
metric aliases to epoch 7. The result remains narrowly below protected V19, by approximately one/two
REC hits, one/one Mask hits, and `6.46e-5` mask mIoU, but it is a material departure from the epoch-6
plateau and justifies the uninterrupted 80-epoch schedule. Epoch 8 reached residual
mean/max/query-std `0.032737/0.095969/0.035570`, saved at 07:16:21 CST, and entered full validation.

The epoch-7 checkpoint passes the qmask audit at Adam step 1,001 with exact `1228/0/12` state,
12 optimizer states, 92,429 optimized elements, and finite nonzero moments. Its epoch filename and all
five best aliases are six links to one checkpoint inode. Protected V19 remains digest-identical.

Epoch 8 reports REC `0.580879/0.464872` and Mask `0.598023/0.491376/0.418471`. Only Mask@0.50
improves over epoch 7: `4672/9508=0.4913757`, exactly tying protected V19. Retention keeps epoch 7 for
both REC thresholds, Mask@0.25, and mIoU; epoch 8 owns Mask@0.50 and latest. Its audit passes at Adam
step 1,144 with the same exact `1228/0/12`, 12-state, 92,429-element contract.

Retention has physically removed obsolete epoch-2 through epoch-6 weight files. Only two 605MB inodes
remain: five links to epoch 7 and three links to epoch 8. The protected V19 inode remains mode 0444 with
eight links and its original SHA-256.

### 35.23 Four-GPU memory and throughput ceiling audit

The live epoch-9 audit confirms four healthy DDP ranks bound to GPUs 0--3, with approximately
`37.0/34.5/32.4/34.5GB` resident memory. Every GPU reached 100% SM repeatedly during a continuous
25-second sample, and peak board power reached the configured 250W ceiling. High-compute windows are
staggered across ranks because dynamic ScanRefer input preparation and synchronization expose rank
stragglers. All GPU pairs have PCIe `PHB` topology and no NVLink, so a single low-utilization snapshot
is not evidence that only one rank is training.

The active throughput controls are already enabled: batch64 per rank (global 256), four single-threaded
workers per rank, prefetch 1, pinned host memory, non-blocking host-to-device copies, TF32, cuDNN
benchmarking, and the gradient-equivalent query-only fast loss. Cgroup memory is approximately
202.5/309.2GB decimal (188.6/288.0GiB). Eight workers per rank previously crossed this hard limit, while
GPU0 now retains only about 3.9GB nominal headroom before accounting for dynamic-batch peaks. Batch
size, worker count, and prefetch depth therefore remain fixed. Allocating otherwise unused VRAM would
not increase throughput and would remove the remaining peak-memory safety margin.

The qmask run remains uninterrupted and numerically unchanged. The V41 handoff uses all four GPUs as
four concurrent one-GPU smoke jobs, then returns to four-rank DDP for the formal run with the same
validated batch64/worker4/prefetch1 ceiling. Further resource increases require a measured isolated
benchmark rather than an instantaneous utilization reading.

### 35.24 Formal training completion gate

`scripts/audit_training_completion.py` replaces log-text completion claims with a reusable structured
gate. It reuses `metrics_from_receipt` to require the exact `mcln-retrain-metrics-v1` schema and sample
count, valid learned/fixed REC and Mask hit counters, `hits050 <= hits025` for every metric family, and
a finite Mask mIoU consistent with `iou_sum/sample_count`. It then loads the latest checkpoint on CPU,
requires a dictionary payload, and requires its epoch to equal the requested final epoch. A successful
run atomically publishes an `mcln-training-completion-audit-v1` receipt.

The gate now protects both the qmask epoch-80 handoff and formal V41 epoch-80 completion. V41 runs the
complete receipt/epoch gate before its 20-state, 153,531-element, Adam-step-11,440 checkpoint-content
audit. The focused completion, existing oracle, and smoke tests pass 19 cases; Python compilation, CLI
execution, and queue shell syntax also pass. A real 9,508-row qmask epoch-8 receipt/checkpoint dry run
passed. Only the supervisor tmux session was restarted to load the new script, and it rebound to the
uninterrupted qmask PID 151429 at 07:29:45 CST.

### 35.25 Epoch-9 qmask mIoU increment

Epoch 9 trained in 570.29 seconds and published its complete receipt at 07:31:47 CST through the
event-driven wait. REC is `0.580879/0.464767`, and Mask is `0.598023/0.491165/0.418554`. Neither REC
threshold nor either Mask threshold improves its retained qmask best. Mask mIoU is
`0.4185541973`, approximately `5.69e-6` above epoch 7, so only the mIoU alias advances to epoch 9. It
remains below the protected V19 Mask mIoU of `0.418613` and does not replace the network best.

The epoch-9 qmask audit passes with common/changed/new state `1228/0/12`, epoch/Adam step `9/1287`,
12 optimizer states, 92,429 optimized elements, and 24 finite nonzero moment tensors. Retention now
requires three physical checkpoint inodes: epoch 7 for both REC thresholds and Mask@0.25, epoch 8 for
Mask@0.50, and epoch 9 for Mask mIoU plus latest. No low-value inode is currently eligible for deletion,
and training has continued into epoch 10.

### 35.26 V42 in-network query-mask calibration

The mask ceiling is not explained by immutable frozen logits. The earlier strict train-only oracle
panel `20260723_stage0_panel64x16_identity/summary.json` evaluates 1,024 samples over queries,
text/query/fused mask sources, and fixed logit thresholds while forbidding a lower box tier. It still
finds `+0.058594` Mask@0.50 and `+0.056269` mIoU headroom. The old `JointBoxMaskAdapter` converts this
space into a hard offline selector, but all five official metrics regress under scene-disjoint train
calibration. It is therefore excluded from the formal queue and is not treated as a transferable
architecture.

After excluding argmax ties, boundary thresholds still strictly improve mask IoU over the interior
`{-0.5,0,0.5}` grid for 318/1,024 rows, totaling approximately 7.879 IoU. A pure text or query source
strictly beats the existing fused source for 306/1,024 rows, totaling approximately 6.674 IoU. The old
alpha-delta `0.5` and bias `1.0` bounds can only approach a pure source and threshold magnitude 1 at
the saturated limit of `tanh`. The formal V42 bounds are therefore alpha delta `1.0` and bias `2.0`,
placing both targets near the healthy `atanh(0.5)` operating point. This evidence uses only the
train-only panel and does not access validation.

V42 instead extends V41's shared query-set attention with continuous per-query mask calibration. The
base mask alpha becomes a third scalar evidence input. The same hidden state predicts a bounded alpha
residual and a bounded mask-logit bias through one zero-initialized two-output head. The bias is added
to both text and query mask sources, making the later fusion mathematically equivalent to
`base_fused_logits + bias`. Formal bounds are alpha delta `1.0` and logit bias `2.0`. Step-0 box scores,
fusion weights, source logits, and fused masks remain exact V19 identities. Both calibration residuals
are zeroed on invalid queries, query permutation equivariance is preserved, and the box-tier priority
still prevents mask utility from crossing a localization tier.

The joint-only fast path calls the official `SetCriterion.forward_query_mask_fusion` and adds
`mask_loss_scale * (10*focal + 2*dice)` to the V41 listwise, multitask, and anchor objective. Both alpha
and bias therefore learn from training masks without a ScanRefer validation threshold scan. Disabling
calibration preserves V41's 20-state, 153,531-parameter contract. V42 has 22 state tensors and 153,919
parameters, only 388 more than V41.

A real CPU construction uses `prepare_source_moe_gate_checkpoint_config` to inherit the protected
V19 three-source, router, and fallback-gate dimensions before loading. Its structured receipt is
`output/joint_query_quality/v42_protected_v19_initialization_audit.json`: the complete V42 model has
1,250 tensors, all 1,228 protected V19 tensors load bit-exactly, exactly 22 V42 tensors are missing,
and there are no unexpected or changed common keys. The trainable module contains exactly 153,919
parameters. The new `v42` checkpoint profile requires `1228/0/22`, 22 Adam states, and 153,919
optimizer elements and requires the exact `1.0/2.0` alpha/bias bounds; the `v41` profile explicitly
rejects enabled mask calibration.

Tests cover exact box/mask initialization identity, nonzero gradients into both calibration channels,
bias/fusion equivalence, invalid-query masking, permutation equivariance, real fast-mask-loss
backpropagation, exact V41/V42 state contracts, V42 checkpoint auditing, and non-collapsed smoke
diagnostics. The focused group passes 50 tests, and the related SourceMoE, training-group, checkpoint,
evaluator, and completion group passes 321, for 371 passing tests total. Python compilation and both
launcher syntax checks pass.

The existing event-driven `scripts/queue_v41_smokes_after_qmask.sh` filename is retained so the tmux
handoff stays stable, but its current contents launch V42. After the qmask epoch-80 completion gate,
four one-GPU, three-epoch batch64 smoke variants run concurrently. Each enables mask calibration and
must pass its 128-row receipt, the exact V42 checkpoint audit, nonzero residual mean/query variation,
and nonzero alpha-residual, bias-mean, and mask-weight query variation. Passing all four gates launches
the fixed four-rank 80-epoch formal experiment
`v42_qualitycoupled_maskcal_full80_b64x4_lr3e4_mw025_t025_a05_mad10_mlb20`, followed by the epoch-80
completion and V42 checkpoint audits.

The production resource ceiling remains batch64 per rank (global 256), four workers per rank, and
prefetch 1. During a 20-second sample every GPU repeatedly reached 99--100% SM with roughly
`37.0/34.5/32.4/34.5GB` allocated. Alternating low samples are dynamic-input and PCIe-DDP
synchronization windows. GPU0 has only about 3.9GB nominal headroom, and eight workers per rank already
crossed the 288GiB cgroup limit; synthetic VRAM reservation or larger batch/worker settings would
reduce reliability rather than throughput.

The complete qmask epoch-10 receipt is REC `0.580984/0.464977` and Mask
`0.598023/0.491271/0.418554`. It does not improve any epoch-7/8/9 retained best, and training has entered
epoch 11. Retention has removed the unreferenced epoch-10 filename. Protected V19 remains mode 0444,
eight hard links, and digest
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`. Future metric checks follow the
measured 12--13 minute epoch window rather than minute-scale polling.

Epoch 11 published at 07:56:34 CST with REC `0.580879/0.464872` and Mask
`0.598023/0.491271/0.418463`. It improves no retained metric, so no low-value checkpoint audit is run.
Epoch 11 remains only as the current latest recovery inode and will be unlinked after epoch 12 saves a
new latest checkpoint. Training has entered epoch 12. The V42 supervisor remains bound to qmask PID
151429 and has not launched any additional GPU workload.

### 35.27 V43 source-aware query-mask calibration

V42 can adjust alpha and threshold per query, but its shared 152D rich state contains only fused-mask
confidence, fused foreground ratio, and text-query soft Dice. It observes the result of fusion without
directly observing which source is confident or how each source is distributed around the decision
threshold. Expanding `rec-query-v1` itself would invalidate the established 152D candidate caches,
geometry reranker, joint box-mask adapter, and hierarchical-reranker contracts.

V43 therefore preserves that public schema and adds a private optional 10D source-evidence branch only
inside joint query mask calibration. Text and query masks each contribute probability mean,
probability standard deviation, confidence, and foreground ratio. Probability L1 disagreement and
hard-mask disagreement complete the ten dimensions. Every value is target-free and bounded in
`[0,1]`; no validation threshold, category prior, or dataset-specific best source is encoded. The same
construction is deployable for single-stage ScanRefer, Nr3D, and Sr3D. Base alpha, normalized and
standardized parent scores, and the source evidence feed the existing query-set attention before its
joint box/mask quality, IoU, localization-residual, alpha-residual, and logit-bias heads.

`--joint_query_quality_use_source_mask_evidence` enables the branch and fail-closed requires mask
calibration. It is off by default, so V41/V42 state shapes, old checkpoints, and all shared 152D
artifacts remain compatible. Enabling V43 grows the projection input from 155D to 165D while retaining
22 state tensors. Parameter count is 155,219, only 1,300 above V42. All quality, localization-residual,
and mask-calibration output heads remain zero initialized. Step-0 parent scores, selected queries,
fusion alpha, source logits, and fused masks are therefore bit-exact V19 identities for arbitrary
source evidence; invalid-query masking and query permutation equivariance remain intact.

`scripts/audit_joint_query_initialization.py` now makes the real protected-checkpoint initialization a
reusable gate. The V43 receipt at
`output/joint_query_quality/v43_protected_v19_initialization_audit.json` reports 1,250 target tensors,
all 1,228 V19 tensors bit-equal, exactly 22 missing joint-query tensors, no changed common, unexpected,
or shape-mismatched tensors, zero output heads, and exactly 155,219 trainable module parameters. The
`v43` checkpoint profile requires `1228/0/22`, 22 finite nonzero Adam states, 155,219 optimizer
elements, exact alpha/bias bounds `1.0/2.0`, and the source-evidence flag. The V42 profile explicitly
rejects that flag to prevent contract ambiguity.

V43 smoke validation additionally requires nonzero source-evidence query variation and nonzero source
disagreement, alongside the existing localization residual, alpha residual, bias, and mask-weight
variation gates. Tests cover exact source statistics, variable superpoint counts, input validation and
detachment, V43 identity, permutation equivariance, exact V41/V42/V43 parameter contracts, checkpoint
auditing, and smoke collapse rejection. The focused group passes 58 tests; the broader shared-schema,
SourceMoE, training-group, and checkpoint group passes 376. Compilation, launcher syntax, and the real
CPU initialization audit pass.

The stable handoff filename `scripts/queue_v41_smokes_after_qmask.sh` now launches four concurrent V43
smokes and, only after all gates pass, the four-rank formal run
`v43_sourceaware_maskcal_full80_b64x4_lr3e4_mw025_t025_a05_mad10_mlb20`. It retains the measured
batch64/rank, worker4/rank, prefetch1 production ceiling and finishes with epoch-80 completion plus V43
checkpoint audits.

Qmask epoch 13 published at 08:21:40 CST. REC is `5522/4420 = 0.580774/0.464872`; Mask is
`5684/4673 = 0.597812/0.491481` with mIoU `0.418374`. Only Mask@0.50 improves, by one hit over protected
V19 and epoch 8. Retention binds that best and latest to epoch 13. Its audit passes exact
`1228/0/12`, Adam step 1,859, 12 states, 92,429 optimized elements, and finite nonzero moments. The
protected V19 digest remains
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`.

Only the successor supervisor was restarted at 08:25:11 CST. The stable tmux name remains
`mcln_v41_after_qmask`, while its V43 event log and lock are `v43_after_qmask_queue.log/.lock`. It has
rebound to the uninterrupted qmask PID 151429 and has not launched any smoke GPU process early.

### 35.28 V44 candidate-query mask supervision

A gradient-path audit found a supervision-density mismatch in V43. Post-calibration fused-mask IoUs
are intentionally built under `no_grad`, so all-query quality and reranking targets are detached and
well defined. However, the alpha-residual and logit-bias outputs receive direct focal/dice gradients
only from the few queries selected by Hungarian matching. Adding source evidence without broadening
this direct objective leaves most query-specific calibration outputs weakly supervised.

V44 retains the exact V43 architecture and adds an optional train-only candidate-mask objective. For
each grounding row it unions Top-K queries under detached deployed scores with Top-K queries under
detached GT box IoU, excludes invalid queries and Scannet detection rows, and applies the standard
focal/dice losses to their actual calibrated fused logits. The deployed half trains masks that can be
selected now; the box-oracle half supplies useful alternatives that the reranker should learn to
select. Restricting supervision to this union avoids pushing all 256 unrelated background queries
toward one referring mask. GT is used only to construct training candidates and masks; inference has
no additional input or computation.

The objective is
`candidate_weight * mask_loss_scale * (10 * focal + 2 * dice)`. New CLI controls are
`--joint_query_quality_candidate_mask_loss_weight` and
`--joint_query_quality_candidate_mask_top_k`, defaulting to zero and 16. The zero default preserves
every V41/V42/V43 numerical and checkpoint contract. Formal V44 uses weight 0.25 and K=16, at most 32
distinct candidates per row. Smoke gating additionally requires a positive finite
`joint_query_quality_candidate_mask_query_ratio`. Tests cover deployed/oracle union semantics,
invalid and non-grounding exclusion, strict zero gradients for unselected queries, alpha/bias
gradients for selected queries, and fast-path integration. The related regression set passes 343
tests; the complete repository passes 3,138 tests with three pre-existing PyTorch gradcheck/scheduler
warnings. Compilation and launcher syntax checks pass.

The candidate implementation was then optimized without changing its objective. Instead of fusing
all 256 queries and gathering afterward, it now gathers at most 32 text/query logit rows and alpha
values before fusion. An independent dense reference proves bit-exact total loss, alpha gradients,
and bias gradients with `rtol=0, atol=0`; the optimized related set passes 138 tests. This removes an
unneeded all-query autograd fusion graph from V44 while preserving candidate selection and deployment.

The event queue now pre-registers four concurrent V44 smokes with weight/K pairs `0.25/16`, `0.10/8`,
`0.25/32`, and `0.50/16`. They retain the V43 155,219-parameter audit profile because V44 adds no
state. Passing all receipt, checkpoint, residual, source-evidence, calibration, and candidate-coverage
gates launches
`v44_candidate_mask_full80_b64x4_lr3e4_mw025_cmw025_k16` on four DDP ranks for 80 epochs.

Resource settings remain batch64 per rank, global batch256, four workers per rank, and prefetch1. A
30-second sample showed every GPU repeatedly reaching 100% SM with approximately
`37.0/34.5/32.4/34.5GB` resident. Every GPU pair is PCIe PHB with no NVLink. The cgroup currently uses
about 208.7 of 309.2GB and has already recorded 60,283 limit events from the rejected eight-worker
configuration. GPU0 has only about 3.9GB dynamic headroom, so larger batches, workers, prefetch, or
synthetic VRAM reservation are not production-safe throughput improvements.

The V44 supervisor restarted at 08:44:08 CST and rebound to the uninterrupted qmask PID 151429. Its
event log is `v44_after_qmask_queue.log`; it shares the existing lock so V43 and V44 queues cannot run
concurrently, and it performs no log polling or GPU work while waiting. Qmask epoch 14 completed at
08:34:22 CST with REC `0.580879/0.464872` and Mask `0.597707/0.491376/0.418466`, improving no retained
metric. Retention removed `ckpt_epoch_14.pth`; only epoch 7, 9, and 13 best inodes plus the epoch-15
latest inode remain. Protected V19 is still mode 0444 with eight links and its original digest.

Qmask epoch 15 published at 08:46:47 CST with REC
`5525/4422 = 0.581091/0.465082`, Mask `5684/4669 = 0.597812/0.491061`, and mIoU `0.418522`. It improves
none of the epoch-7/9/13 retained bests, and epoch 16 has started. The next receipt check window is
approximately 09:02--09:04 CST.

### 35.29 V45 Lovasz-Jaccard candidate objective

The official evaluator thresholds fused logits at zero (`sigmoid > 0.5`) and then computes point-mask
Jaccard. V44 focal and Dice losses train alpha and bias, but neither directly optimizes the sorted hard
IoU error. V45 adds an optional binary Lovasz hinge following the open-source
`bermanmaxim/LovaszSoftmax` formulation: margin errors are sorted and weighted by the discrete Jaccard
extension gradient. It encodes no ScanRefer threshold, class, or preferred-source prior and transfers
unchanged to single-stage ScanRefer, Nr3D, and Sr3D.

`--joint_query_quality_candidate_lovasz_loss_weight` defaults to zero and constructs the sorting graph
only when positive, preserving all V41--V44 default numerical and runtime contracts. It reuses the V44
deployed-Top-K/box-oracle-Top-K union and acts on network-calibrated fused logits. There is no new state
or inference work. Tests cover the exact single-pixel margin and gradient, zero loss past the margin,
point permutation invariance, finite fast-path calibration gradients, and invalid inputs. The related
set passes 347 tests, compilation passes, and launcher syntax passes. GPU benefit remains unproven until
the smoke panel; CPU contracts are not treated as metric evidence.

The four successor smokes now form a strict single-variable ablation. Lovasz weights are
`0/0.05/0.10/0.20`; every other setting is fixed at `lr=3e-4`, dropout 0.1, mask weight 0.25,
temperature 0.25, anchor 0.5, candidate focal/dice weight 0.25, and K=16. Any nonfinite run, incomplete
128-row receipt, V43 parameter-contract failure, or collapsed candidate/source/calibration diagnostic
blocks formal launch. The pre-registered formal candidate is
`v45_lovasz_candidate_mask_full80_b64x4_lr3e4_mw025_cmw025_clw010_k16` with Lovasz weight 0.10 and
the unchanged V43 22-state, 155,219-parameter checkpoint profile.

The complete V45 repository regression passes 3,142 tests with the same three pre-existing PyTorch
gradcheck/scheduler warnings. The waiting supervisor restarted at 09:00:20 CST and bound to the
uninterrupted qmask PID 151429. Its event log is `v45_after_qmask_queue.log`; it does no polling and
uses no GPU while waiting. Protected V19 again passes its digest, mode-0444, and eight-link checks.

Qmask epoch 16 published at 08:59:17 CST with REC
`5523/4420 = 0.580879/0.464872`, Mask `5686/4669 = 0.598023/0.491061`, and mIoU `0.418370`. It improves
no retained best. The epoch-16 inode is only latest and will be removed after epoch 17 saves. The
measured interval between complete receipts is 12 minutes 30 seconds, placing the next check window at
approximately 09:11--09:13 CST.

The V45 smoke summary gate now additionally requires a finite, strictly positive
`joint_query_quality_candidate_lovasz_loss` for the nonzero-weight g1/g2/g3 variants; the zero-weight g0
control is exempt. Because the 09:00:20 CST supervisor retained the pre-gate script inode, only the
waiting queue was stopped and restarted under the same tmux name at 09:08:06 CST. It rebound to the
uninterrupted qmask PID 151429 and launched no GPU job early.

A 12-second per-second sample during epoch 17 showed every GPU repeatedly reaching 98--100% SM, with
resident memory still approximately `37.0/34.5/32.4/34.5GB`. Short troughs alternate across ranks due
to dynamic point-cloud batches and PCIe DDP synchronization; this is not a GPU2-only run. GPU0 retains
only about 3.9GB dynamic headroom, and eight workers per rank previously hit the 309.2GB cgroup limit.
The production setting therefore remains batch64/rank, four workers/rank, and prefetch1; larger batches,
more host prefetch, or synthetic VRAM reservation would increase failure risk without demonstrated
throughput benefit.

Qmask epoch 17 published at 09:11:53 CST with REC
`5525/4422 = 0.581091/0.465082`, Mask `5687/4672 = 0.598128/0.491376`, and mIoU `0.418523`. It did not
strictly improve the epoch-7/9/13 per-metric bests. Retention removed the low-value epoch-16 inode and
now keeps the epoch-7, epoch-9, and epoch-13 best inodes plus the epoch-17 latest inode. At the measured
roughly 12-minute-30-second full cycle, the next receipt check window is 09:24--09:26 CST.

### 35.30 V46 fallback-gate evidence joint reranker

Epoch-17 diagnostics separate two ceilings: the oracle over fixed-source Top-1 predictions is only
`0.58551/0.46992`, whereas the fallback-gate candidate-set oracle reaches `0.62842/0.54933`. V43--V45
receive rich query state and source-mask statistics but do not explicitly receive the eligibility,
dynamic-anchor, and trained quality outputs that define this higher-value candidate set. V46 adds the
default-off `--joint_query_quality_use_gate_evidence` path and injects 24 target-free, deployment-time
gate features into the same set-attention quality head: four candidate/default/selected/action-anchor
indicators; candidate-score rank and standardized confidence plus expected utility, direct utility,
and action margin; break/neutral/fix probabilities for each of two box and two mask thresholds; and
three fallback/neutral/override decision probabilities. Every channel lies in `[0,1]`. None encodes a
ScanRefer class, GT annotation, dataset threshold, or preferred source, so the contract transfers to
single-stage ScanRefer, Nr3D, and Sr3D.

The option only widens V43's first input layer by 24 channels. All output heads remain zero initialized,
so loading protected V19 preserves step-zero REC, mask alpha, and mask bias exactly. V46 retains 22
state tensors and increases trainable parameters from 155,219 to 158,339. Tests cover normalized
probability contracts, missing/invalid rejection, query-permutation equivariance, detached gate inputs,
exact identity, and parameter count. Checkpoint audit adds a `v46` profile; the smoke gate requires
finite positive gate-evidence query variation and candidate coverage. Joint/SourceMoE integration,
checkpoint-audit, and summary focused regressions pass 162 tests; Python compilation and launcher
syntax pass.

The successor four-GPU panel is a `2x2` design: g0=`gate0/clw0`, g1=`gate0/clw0.10`,
g2=`gate1/clw0`, and g3=`gate1/clw0.10`. All other settings remain fixed at LR `3e-4`, dropout 0.1,
mask weight 0.25, temperature 0.25, anchor 0.5, candidate focal/Dice weight 0.25, and K=16. The two V45
controls use the V43 parameter audit; the two gate-evidence variants use V46. Both candidate/Lovasz and
noncollapse summaries must pass before the pre-registered four-GPU 80-epoch
`v46_gate_evidence_lovasz_candidate_mask_full80_b64x4_lr3e4_mw025_cmw025_clw010_k16` launch. The new
supervisor bound to uninterrupted qmask PID 151429 at 09:29:58 CST and logs to
`v46_after_qmask_queue.log` without launching GPU work early.

Qmask epoch 18 published at 09:24:20 CST with REC
`5521/4419 = 0.580774/0.464767`, Mask `5683/4673 = 0.597707/0.491481`, and mIoU `0.418320`. It strictly
improves no epoch-7/9/13 best. Retention replaced the epoch-17 latest inode with epoch 18, and epoch 19
is training normally. The next receipt window is approximately 09:36--09:38 CST.

### 35.31 V46 initialization preflight and qmask epoch 19

Metric provenance is now an explicit reporting invariant. `0.582878/0.486012` belongs to the historical
epoch-71 backbone plus its SHA-bound parent/geometry sidecars; it is not a V19 post-processing result.
Network-only V19 is `0.581195/0.465398`. The separately regenerated and train-only-fitted V19 one-shot
parent+geometry result is `5511/4587 = 0.579617/0.482436`. Future tables must keep historical-system,
network-only, and same-checkpoint-retrained-sidecar results in separate columns; sidecars cannot be
transferred across checkpoint hashes.

The standalone protected-initialization auditor now includes V46. On the real protected V19 it builds
the full mask-calibration, source-mask-evidence, and 24-channel gate-evidence model and verifies source/
target states `1228/1250`, common/changed/missing/unexpected/shape-mismatch
`1228/0/22/0/0`, exactly 22 joint-head states, 158,339 parameters, and zero-initialized quality,
localization-residual, and mask-calibration output heads. A focused test constructs the same real
`JointQueryQualityReranker` contract. The V46-focused suite passes 163 tests; the complete repository
passes 3,151 tests with the three existing PyTorch warnings and no failures.

The successor queue now runs real protected-V19 V43 and V46 initialization audits before allocating
the GPUs for the smoke panel. Any common-tensor drift, wrong missing set, parameter count, or feature
flag fails closed. The updated supervisor rebound to uninterrupted qmask PID 151429 at 09:41:47 CST
without launching extra GPU work. Protected V19 remains mode 0444 with eight hard links and digest
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`.

Qmask epoch 19 published at 09:36:58 CST with REC
`5523/4421 = 0.580984/0.464977`, Mask `5685/4671 = 0.597918/0.491271`, and mIoU `0.418374`.
It improves no epoch-7/9/13 retained best. Epoch 20 continues normally, with the next complete receipt
expected around 09:49--09:51 CST rather than through minute-level polling.

A capacity preflight found only 3.2 GB free on `/root/autodl-tmp`, below the worst-case retention peak
for four parallel smokes. Three documented failed, single-link mask-head checkpoints were removed:
evalmode epoch 72 (`0.595499/0.486748/0.415837`), fullmaskhead epoch 73
(`0.595288/0.485591/0.414901`), and small-LR epoch 72 (`0.591186/0.481700/0.412972`). Their logs and
configs remain, and about 1.9 GB was reclaimed. The V28 Mask@0.50 checkpoint remains protected because
its `0.491481` ties the current per-metric best; protected V19 and qmask epochs 7/9/13 were untouched.
V28 now also has the read-only hardlink
`protected_mcln_artifacts/scanrefer_mask050_best_v28_0.491481.pth`; both paths share inode 6502719492,
two links, mode 0444, and SHA-256
`2b72aa4d7d4feb4d5423e7d7061032d88909c9a82094fd4573621cd785f808b8`.

V46 panel and formal outputs now default to `experiment_output/joint_query_quality` on the workspace
filesystem, currently with about 9.6 GB free. After qmask exits, the queue requires at least 8 GiB on
that filesystem before using a GPU. Once all four 128-row smoke checkpoints and both summaries pass,
only the four debug run roots have their `.pth` files removed before formal V46 starts; metrics,
audits, configs, and logs remain. The updated supervisor rebound to qmask PID 151429 at 09:48:45 CST;
launcher syntax and the related gate suite pass 29 tests.

Qmask epoch 20 published at 09:49:28 CST with REC
`5522/4420 = 0.580774/0.464872`, Mask `5683/4673 = 0.597707/0.491481`, and mIoU `0.418296`.
Mask@0.50 only ties epoch 13 and therefore creates no strict best; all other metrics are lower. Epoch 21
continues, with the next complete receipt expected around 10:01--10:03 CST.

Monitoring now uses four-complete-epoch windows, approximately every 45--50 minutes. Epochs 21--24
published at 10:02:03, 10:14:44, 10:27:09, and 10:39:38 CST:

| epoch | REC@0.25 / @0.50 | Mask@0.25 / @0.50 / mIoU |
| ---: | ---: | ---: |
| 21 | `5524/4422 = 0.580984/0.465082` | `5683/4670 = 0.597707/0.491165/0.418518` |
| 22 | `5524/4422 = 0.580984/0.465082` | `5684/4669 = 0.597812/0.491060/0.418364` |
| 23 | `5524/4422 = 0.580984/0.465082` | `5684/4669 = 0.597812/0.491060/0.418469` |
| 24 | `5525/4422 = 0.581090/0.465082` | `5685/4667 = 0.597918/0.490850/0.418513` |

None strictly improves the epoch-7/9/13 per-metric bests. Retention only replaces latest with epoch 24
and creates no new best inode. All four GPUs reported 100% SM at the 10:40:43 CST window; qmask and the
event-driven successor queue remain healthy. The next inspection is scheduled for approximately
11:28--11:31 CST, without intermediate epoch-25--27 checks.

At the 11:29:07 CST window, epochs 25--27 had complete receipts while epoch 28 had only atomically saved
its checkpoint and remained in validation; no partial epoch-28 metric was read:

| epoch | REC@0.25 / @0.50 | Mask@0.25 / @0.50 / mIoU |
| ---: | ---: | ---: |
| 25 | `5525/4422 = 0.581090/0.465082` | `5685/4669 = 0.597918/0.491060/0.418553` |
| 26 | `5523/4421 = 0.580879/0.464977` | `5683/4674 = 0.597707/0.491586/0.418380` |
| 27 | `5523/4421 = 0.580879/0.464977` | `5683/4670 = 0.597707/0.491165/0.418448` |

Epoch 26 strictly raises the Mask@0.50 network best from `4673/9508=0.491481` to
`4674/9508=0.491586`; the other four metrics do not improve. Its read-only protected hardlink is
`protected_mcln_artifacts/scanrefer_qmask_best_mask050_epoch26_0.491586.pth`. The source, best alias,
and protected path share inode 6451674584, three links, mode 0444, and SHA-256
`b825ee71f5d8b810307a6c139d54d1d03e7c2ee140c47d67ff0f7c460053de2e`. Retention now keeps epochs
7, 9, and 26 plus latest. The next four-epoch window is approximately 12:18--12:21 CST, without a
separate epoch-28--31 check.

At the 12:18:38 CST window, epochs 28--31 had complete receipts; epoch 32 had only saved its checkpoint
and remained in validation, so no partial epoch-32 metric was read:

| epoch | REC@0.25 / @0.50 | Mask@0.25 / @0.50 / mIoU |
| ---: | ---: | ---: |
| 28 | `5522/4420 = 0.580774/0.464872` | `5682/4668 = 0.597602/0.490955/0.418302` |
| 29 | `5522/4420 = 0.580774/0.464872` | `5683/4670 = 0.597707/0.491165/0.418360` |
| 30 | `5523/4422 = 0.580879/0.465082` | `5682/4670 = 0.597602/0.491165/0.418334` |
| 31 | `5522/4421 = 0.580774/0.464977` | `5682/4672 = 0.597602/0.491376/0.418123` |

No per-metric best changed; retention continues to protect epochs 7, 9, and 26 plus latest. The four
GPUs and V46 waiting queue remain healthy. The next inspection is approximately 13:08--13:11 CST,
with no separate epoch-32--35 check.

At the 13:07:46 CST window, epochs 32--35 had complete receipts while epoch 36 had only saved its
checkpoint and remained in validation:

| epoch | REC@0.25 / @0.50 | Mask@0.25 / @0.50 / mIoU |
| ---: | ---: | ---: |
| 32 | `5523/4421 = 0.580879/0.464977` | `5683/4669 = 0.597707/0.491060/0.418253` |
| 33 | `5522/4421 = 0.580774/0.464977` | `5683/4675 = 0.597707/0.491691/0.418240` |
| 34 | `5523/4422 = 0.580879/0.465082` | `5683/4668 = 0.597707/0.490955/0.418362` |
| 35 | `5524/4422 = 0.580984/0.465082` | `5684/4671 = 0.597812/0.491271/0.418303` |

Epoch 33 raises Mask@0.50 by one more hit to `4675/9508=0.491691`; all other bests remain unchanged.
The new protected path is
`protected_mcln_artifacts/scanrefer_qmask_best_mask050_epoch33_0.491691.pth`, sharing inode 6542431840,
three links, mode 0444, and SHA-256
`20a1a877875cc194356fe1b0781528cd9af8d4e591288f6c91a19c1f71b61b46` with its run aliases. The
superseded epoch-26 external hardlink was removed, reclaiming about 605 MB while preserving metrics and
hash records; V19 was untouched. The next window is approximately 13:56--13:59 CST, without a separate
epoch-36--39 check.
