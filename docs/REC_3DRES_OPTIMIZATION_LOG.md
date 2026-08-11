# ScanRefer REC / 3DRES 优化交接记录

更新日期：2026-08-06

## 目标与保护基线

| 指标 | 当前保护结果 | 目标 |
| --- | ---: | ---: |
| REC Position Acc@0.25 | 58.2878%（5542/9508） | >= 59.00%（>= 5610/9508） |
| REC Position Acc@0.50 | 48.6012%（4621/9508） | >= 49.00%（>= 4659/9508） |
| 3DRES Mask Acc@0.25 | 59.6971% | 不下降并争取提升 |
| 3DRES Mask Acc@0.50 | 49.0324% | > 50.70% |
| 3DRES Mask semantic mIoU | 41.7676% | > 44.72% |

### 当前验收状态（2026-08-06 01:52 CST）

| 验收项 | 当前权威结果 | 状态 |
| --- | --- | --- |
| 双阶段 REC 目标 | 历史受保护系统 `0.582878/0.486012`；目标 `0.59/0.49` | 未达标，分别还差 68/38 hits |
| 网络内最好 | V19 REC `0.581195/0.465398`；Mask `0.598233/0.491376/0.418613` | 已保护，未达到 REC 与 Mask@0.50/mIoU 目标 |
| 候选覆盖上界 | V19 完整 query oracle `0.629680/0.550063`，超过目标 377/571 hits | proposal 充分，主问题是 query 排序和安全覆盖 |
| 单阶段 ScanRefer | epoch 28；retained best REC `0.571939/0.461927`，Mask `0.588241/0.483382/0.412678` | 100 epoch 正式训练中 |
| 自适应多源架构 | V49 三源逐 query mixer；V50 增加 `sacr_structured` 第四源和 query-focus | 已实现并通过合同测试，等待正式训练 |
| 分割专项架构 | 联合六维 Box/Mask 质量、query alpha/bias、source evidence、空间 residual、候选 Mask 损失 | 已实现，V48--V50 队列待运行 |
| 跨数据集合同 | ScanRefer/Nr3D/Sr3D train SACR join 与 token/relation 对齐 | 三个数据集均通过，正式指标待 ScanRefer 主目标后执行 |

当前执行链固定为：单阶段完整 100 epoch -> parent/geometry/JointBoxMask v2 -> 单阶段 Joint Query
-> V48 spatial Mask -> V49 adaptive source smoke/formal -> V50 SACR 四源 formal。历史 `0.582878`、
V19、parent、geometry 和 Mask best 权重均为只读保护；各活动 run 使用 metric retention 自动清理
低指标中间权重。

保护输入均为只读 `0444`，禁止覆盖：

- Backbone：`/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth`
  - SHA-256：`3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`
- Parent reranker：`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/artifacts/reranker_h256_d010_lr1e3_seed0_final_contract.pth`
  - SHA-256：`f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b`
- Geometry reranker：`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_artifacts/selected_geometry_reranker.pth`
  - SHA-256：`835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f`

## 已核验实验

| 实验 | 数据/规模 | Position @0.25 / @0.50 | Mask @0.25 / @0.50 / mIoU | 结论 |
| --- | --- | --- | --- | --- |
| Epoch-71 + parent + geometry 正式验证 | ScanRefer val，9508 | 58.2878% / 48.6012% | 59.6971% / 49.0324% / 41.7676% | 当前保护基线；@0.25 还差 68 hits |
| Joint box-mask Stage-0 smoke | train，1 scene × 4 expressions，重复两次 | 仅 smoke，不作总体指标 | oracle 相对 legacy：@0.50 +25.0pp，mIoU +23.42pp | 两次 rows/summary 一致；样本太少，不能外推 |
| Joint box-mask Stage-0 正式 panel 首次运行 | train，计划 64 × 16 | 未完成 | 未完成 | 在实际第 287 行前因旧 cache query identity 漂移而 fail-closed |

Stage-0 provenance 修复后的可复现实验：

| 实验 | 数据/规模 | Position @0.25 / @0.50 | Mask @0.25 / @0.50 / mIoU | 结论 |
| --- | --- | --- | --- | --- |
| Joint box-mask smoke（重复运行 1） | train，1 scene × 4 expressions | 仅 smoke | 仅 smoke | gate 通过 |
| Joint box-mask smoke（重复运行 2） | 同上 | 仅 smoke | 仅 smoke | `rows.pt`、`selection.json` 与去掉耗时后的 summary 完全一致 |
| Joint box-mask approved panel | train，64 scenes × 16 expressions，1,024 rows，426 replay batches | 0.93164 / 0.86133（legacy），0.98145 / 0.93457（joint oracle） | 0.91309 / 0.80762 / 0.69715（legacy），0.97363 / 0.86621 / 0.75342（joint oracle） | Stage-0 gate 通过；这是 train-only oracle/selection，不是官方 validation |
| Joint adapter v1 | train calibration，56 scenes / 3,625 rows | 95.4759% / 91.4483%（baseline），95.2828% / 91.3655%（adapter） | 95.3379% / 88.1379% / 75.6422%（baseline），95.2276% / 87.9724% / 75.4876%（adapter） | 七项 gate 均未通过；`selection=baseline`、`deployable=false`，未发布权重，也未访问 validation |

两次 smoke 的 `rows.pt` SHA-256 均为
`c778c69bb0faff2c0168eda2b2a4346d81bd32f0e96ed1fb205fec1144`。
正式 panel 的 cache manifest SHA-256 为
`c8858036c3da0b25183f262c763e947a3dac77544ee3073623172716878cfabc`。

旧实验目录中 43 次完整 mask evaluation 的本地历史上限约为：Mask Acc@0.25 `59.7076%`、Acc@0.50 `49.0429%`、semantic mIoU `41.7754%`，没有现成权重满足三个 3DRES 目标。

## 已定位问题

1. checkpoint 保存只按 Position 指标挑选，mask 三项没有进入保存规则。
2. parent/geometry reranker 只使用 box IoU 监督。
3. Position 使用 geometry 选择结果，但两个 mask evaluator 仍各自按 legacy semantic/contrastive score 选 query，box 与 mask 可能来自不同 query。
4. text mask 对 256 个 query 共用，只有 superpoint query mask 随 query 变化；当前融合权重还是 sample 级单标量，query-specific calibration 能力不足。
5. 旧 train candidate cache 生成于 2026-07-14，而相关运行时代码随后修改。已复现一个 Top-K 边界 query ID 漂移；同一当前 runtime 的两次 fresh forward（score、endpoint、RNG、buffer、module eval 状态）逐位一致，因此根因是 provenance drift，不是推理随机性。

## 当前创新方向

- 单 backbone、单 forward 的 query-consistent joint box-mask selector：候选身份固定为 `(parent query, geometry variant)`，所选 box 始终使用其 parent query 的 mask。
- 冻结 backbone，先训练轻量 multi-task quality adapter，同时预测两个 box tier、mask IoU/两个 mask tier和不确定性。
- 对 text/query mask logits 学习有界的 query-specific 权重、温度、bias 和 threshold；disabled 路径必须精确还原现有融合。
- 用 train scene-disjoint calibration 做风险门控，只有两个 box tier 的保守下界均不下降时才允许切换候选。
- 若 train-only oracle headroom 不足，则仅解冻 mask-specific head，保持 detector、box regression、parent/geometry reranker 冻结。

设计与执行计划：

- `docs/superpowers/specs/2026-07-23-scanrefer-joint-box-mask-pareto-design.md`
- `docs/superpowers/plans/2026-07-23-scanrefer-joint-box-mask-pareto.md`

## 当前状态与下一步

- Stage-0 纯逻辑、审计 CLI、identity-only provenance 规则和运行时同-query 映射均已实现；相关 CPU 回归 suite 通过（全套 `2521 passed`，2026-07-23 重测）。
- 旧 candidate cache 只用于 immutable identity/历史 panel 绑定；基于它生成的 36,665-row joint cache 缺少最终发布所需的 fresh-runtime provenance，不能作为最终 artifact 输入。validation cache 不参与训练、选择或 gate。
- Stage-1 joint adapter 已按 train-only gate 回退到 baseline；当前进入条件式 Stage-2。
- **2026-07-23 启动**：eval-mode mask-head fine-tune（仅训练 x_mask + x_query，整个 model 保持 eval() 模式，BN 运行统计不更新，box/REC 路径完全冻结）。
  - 脚本：`scripts/run_mask_head_evalmode_ft.py`、启动脚本：`scripts/launch_mask_head_evalmode.sh`
  - 日志目录：`/root/autodl-tmp/DATA_ROOT/output/mask_head_finetune/evalmode_20260723T150113Z/`
  - 配置：lr=2e-4，batch=12，max_epoch=80，warmup=2ep，decay@50/70，weight_decay=5e-4
  - 可训练参数：1,329,984（共 149M），仅 x_mask.{0,2,4}.{weight,bias} 和 x_query.{0,2,4}.{weight,bias}
  - 状态：**训练中**，epoch 1，GPU 12.5 GiB / 23% 利用率
- 每 epoch 保存 checkpoint + 做全量 val 评估（9508 表达式），目标：Mask mIoU > 44.72、Acc@0.50 > 50.70%；同时监测 REC Position 是否不变。
- 新权重、源码快照、命令、环境、输入/输出 SHA-256 和最终指标将在达标后补到本文件。

## Eval-mode mask-head 实验进展（2026-07-23 起）

设计：整个 model 永远 eval()，只训练 x_mask + x_query（1.33M 参数），BN 运行统计与 box/position 路径完全冻结。脚本 `scripts/run_mask_head_evalmode_ft.py` + `scripts/launch_mask_head_evalmode.sh`。
run_id：`evalmode_20260723T150113Z`，lr=2e-4，batch=12，warmup=2ep，decay@50/70。

**REC Acc 有两套数，勿混淆：**
- 正式 REC 指标（加 parent+geometry reranker）：Acc@0.25=**58.28%**、Acc@0.50=**48.60%**（0.48601）。这是要保护/超越的目标。
- 纯 plain（不加 reranker）：Acc@0.25=57.99%、Acc@0.50=**46.35%**（0.46350）。reranker 约贡献 +2.25pp。
- 微调 run 为省时不加载 reranker，日志里打印的是 plain 46.35，仅用作"box 路径是否被冻结"的探针；只要 plain 逐位不变，加同一套 reranker 后的 48.60/58.28 也必然不变。

| epoch | Position @0.50 (plain 探针) | Mask overall25 | Mask overall50 | mask_sem mIoU | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 基线（保护权重，plain 探针） | 0.46350 | 0.59697 | 0.49032 | 0.41767 | 参照（加 reranker 时 REC 为 58.28/48.60） |
| 72（旧 run，只训 x_mask+x_query，warmup） | 0.46350 | 0.59550 | 0.48675 | 0.41584 | plain 探针逐位一致 ✓；但 mask 分支大部分被冻死，三项微降；已废弃 |
| 72（新 run fullmaskhead，2.66M 参数，warmup ep） | 0.46350 | 0.59529 | 0.48822 | 0.41631 | plain 探针逐位一致 ✓；warmup 首 epoch 三项仍略低于基线 |
| 73（fullmaskhead，第一个满 lr=2e-4 epoch） | 0.46350 | 0.59529 | 0.48559 | 0.41490 | plain 探针逐位一致 ✓；**三项继续下降**，决策规则触发：lr=2e-4 过高，已停止，重启 lr=5e-5 |

## 根本原因分析与方法修正（2026-07-24）

**eval-mode frozen mask-head 失败根因：**
epoch-71 保护权重是用 2 个 pair-sweep epoch 针对 REC Position 优化的，其 transformer decoder 特征已对 box-localization 适配。mask head (x_mask/x_query/swa 等) 的输出依赖 decoder 输出的语义特征。当 decoder 冻结时，mask head 能接受的输入质量已由 REC 训练固定，仅靠末端几层无法恢复 mask-semantic alignment——与将 mIoU 从 41.77% 提升至 44.72% 所需的深度特征变化相比，mask head 的自由度不够。

**正确方法：小 lr 全量续训（已于 2026-07-24 03:52 启动）**

- 从 epoch-71 保护权重出发，用 `--small_lr` 参数全量续训
- 参数学习率分组：mask head (x_mask/x_query) = lr=2e-4；其他 decoder = 2e-4 × 0.01 = 2e-6；backbone/text_encoder ≈ 0
- `--reduce_lr`：不加载 epoch-71 优化器状态，使用新优化器（REC 历史动量不干扰）
- 全量训练损失（box + mask + contrastive）保持，decoder 可以重新学习 mask-semantic alignment
- 预期：mask 指标逐渐上升；REC position 由于 decoder 学习率极小 (2e-6) 不会大幅下降，结合 reranker 的 +2.25pp 缓冲应仍满足 ≥48.60

| epoch | Position @0.50 (plain) | Mask overall25 | Mask overall50 | mask_sem mIoU | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 基线（ep71，plain） | 0.46350 | 0.59697 | 0.49032 | 0.41767 | 参照 |
| 72（fullmaskhead lr=2e-4，warmup） | 0.46350 | 0.59529 | 0.48822 | 0.41631 | 冻结 decoder，mask 微降 |
| 73（fullmaskhead lr=2e-4，满lr） | 0.46350 | 0.59529 | 0.48559 | 0.41490 | 三项下降，已停；方法无效 |
| **small_lr 续训 ep1** | 0.45888（↓！） | 0.59119 | 0.48170 | 0.41297 | 所有指标下降，REC @0.50 已跌破保护线；已停止 |

**关键诊断（2026-07-24 完成）：**

通过对 epoch-70（pair-sweep 前）和 epoch-71（保护权重）做 plain eval 并汇总所有实验，发现：

| checkpoint | pos@0.25 | pos@0.50 | overall25 | overall50 | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch-70（pair-sweep 前） | 0.57899 | 0.45856 | 0.59361 | 0.48443 | 0.41458 |
| **epoch-71（保护权重）** | **0.57993** | **0.46350** | **0.59697** | **0.49032** | **0.41767** |
| eval-mode mask-head ep72 | 0.46350 | 0.46350 | 0.59529 | 0.48822 | 0.41631 |
| eval-mode mask-head ep73 | 0.46350 | 0.46350 | 0.59529 | 0.48559 | 0.41490 |
| small_lr 续训 ep1 | 0.57615 | 0.45888 | 0.59119 | 0.48170 | 0.41297 |

**核心结论：**
1. epoch-71 是整个训练历史中 mask 质量最好的 checkpoint（pair-sweep 轻微改善了 mIoU）。
2. **目标 mIoU=44.72% 在这次训练中从未达到过。** 这不是退化恢复问题，而是该 checkpoint 从未被训练到 44.72% 的 mask 质量。
3. MCLN 官方 release checkpoint（REC=57.17%, mIoU=44.72%）可以同时满足两个目标，但 Google Drive 链接在本环境不可达。
4. 所有后训练微调方法（eval-mode frozen, small_lr continuation）均使指标进一步下降——说明 epoch-71 的 decoder 特征已高度对 box-localization 特化，无法通过局部微调逆转。

## 最终诊断结论与下一步（2026-07-24）

**问题根因明确：**
- MCLN 官方 release 同时达到 REC Acc@0.25=57.17%、mIoU=44.72%（README 表格）
- 我们从原始训练出发做了 pair-sweep 优化（2 个 epoch），将 REC 从 57.17% 提升到 **58.28%**
- 代价是 mIoU 从 44.72% 退化到 **41.77%**——对于 3DRES 目标，pair-sweep 对 mask 有害
- 因此：所有基于 epoch-71 保护权重的微调实验均无法达到 mIoU>44.72%，因为该 checkpoint 从未被训练到这个水平

**唯一可行路径：从头全量重训**，在标准训练目标下（不做 pair-sweep），同时保存 REC 最优和 mask-mIoU 最优两个 checkpoint：
- 预期：重训完整 70+ epoch 后，在 REC Acc@0.25 ≥58.28% 同时 mIoU ≥44.72% 时保存最优 checkpoint
- 关键修改：在 `main_utils.py` `evaluate_one_epoch` 里增加 mask mIoU 追踪和 best-mask checkpoint 保存
- **已于 2026-07-24 06:15 UTC 启动** `scripts/train_scanrefer_mcln_sp.sh`（标准训练，save_freq=1，val_freq=1）

**官方 checkpoint 状态：** Google Drive 链接在本环境网络不可达，无法直接下载。如果之后能连接网络，下载 `1oBUWrTEj3kYyx-DT0HAvAcDUQe4nQgYz` 直接验证即可跳过重训。
2. 原始 MCLN release 的 RES mIoU 就是 44.72（README）；当前保护权重是 REC 优化的 2ep pair-sweep checkpoint，mIoU 掉到 41.77——即用 mask 质量换了 REC。目标 44.72 本质是恢复原始 mask 质量。
3. epoch-1 mask 轻微下降属 warmup 正常扰动；需观察 warmup 结束（epoch>74）lr 稳定后的趋势，并考虑降低 lr。

## MCLN epoch-54 完整重训 Optuna 正式实验（2026-07-24）

目标：从受保护的 epoch-54 checkpoint 出发，仅用 train scene-disjoint calibration split 完成 20 个有效的两 epoch Optuna trial；按预先固定的 Pareto/约束规则选出候选后，自动续训至 epoch 100，最后才执行正式 validation。调参阶段禁止使用官方 validation。

启动前门禁：

- 最终 CPU 全量回归：`2762 passed, 2 warnings in 151.76s`；两条 warning 均为既有 PyTorch scheduler deprecation，命令退出码为 0。
- 启动脚本 `scripts/tuning/run_optuna_mcln_complete_retrain20.sh` 经 `bash -n` 检查通过。
- 真实 A100 单 batch GPU smoke：`/root/autodl-tmp/DATA_ROOT/output/tuning/mcln_complete_smoke_20260725Tpostfix2/smoke_receipt.json`，退出码为 0，`total_loss=6.176612377166748`，未生成 `.pth`，strace 未发现官方 validation 数据集访问。
- smoke 的四组梯度全部 finite：decoder 607 tensors / norm `0.0882193095547`；backbone 48 / `0.00922733801588`；mask head 52 / `0.0461485307347`；selector 9 / `0.00158730195303`。
- epoch-54 输入：size `793041121`，SHA-256 `a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d`；正式启动前改为 mode `0444`。
- PointNet 输入 SHA-256：`9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2`。
- 启动预检时 A100 40 GB 空闲；`/root/autodl-tmp` 可用 `11800571904` bytes；未发现同类 Optuna/long-train 进程。

固定启动绑定：

- `RUN_ID=20260724T170116Z`
- `RUN_NAME=mcln_complete_retrain20_20260724T170116Z`
- `OUTPUT_ROOT=/root/autodl-tmp/DATA_ROOT/output/tuning/mcln_complete_retrain20_20260724T170116Z`
- `STUDY_NAME=mcln_complete_retrain20_20260724T170116Z`
- SQLite：`/root/autodl-tmp/DATA_ROOT/output/tuning/mcln_complete_retrain20_20260724T170116Z/optuna.db`
- orchestrator PID receipt：`/root/autodl-tmp/DATA_ROOT/output/tuning/mcln_complete_retrain20_20260724T170116Z/control/orchestrator.pid`
- launcher stdout：`/root/autodl-tmp/DATA_ROOT/output/tuning/mcln_complete_retrain_launcher.log`

正式输出目录是新目录，不覆盖或删除旧 best result。启动脚本先发布源码/环境/输入的不可变 provenance，再创建 study contract、执行恰好 3,625-row 的 train-only baseline、收集 20 个合法 COMPLETE trial，并耐久派发 epoch-100 长训。

### 该 study 的最终结果：0 feasible，已于 2026-07-26 停止

按用户要求停止（orchestrator PID 28949 及全部子进程已终止）。**13 个 COMPLETE trial + trial_0013 中途 kill，可行 trial 数为 0**，`best.json` 的 `selection_status` = `no_feasible_trial`，`feasible_trial_count` = 0，epoch-100 长训从未派发。约 40 小时机时零产出。

根因两条（读 `scripts/tuning/mcln_optuna_contract.py:192-240` 确认）：

1. **五项联合非退化门槛不可能通过。** `assess_trial_metrics` 要求 position025 / position050 / mask025 / mask050 / mask_miou 五项 delta **同时 ≥ 0**。任一项为负即 infeasible，objective 落到 `_infeasible_penalty` = `-1000 - 100×deficit - len(failures)`。13 个 trial 的 objective 全部在 -1001 ~ -1010 之间，TPE 收不到有效梯度，等价于随机游走。

2. **calibration split 取自模型已拟合的 train 场景。** baseline 在该 3625 样本 split 上 pos@0.25 = 0.9095、mask mIoU = 0.7233 —— 这是记忆而非泛化。在饱和天花板上再训 2 epoch 只有噪声级波动，正负各半，必然触发五项之一。

| | pos@0.25 | pos@0.5 | mask@0.25 | mask@0.5 | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (ep54) | 0.9095 | 0.8381 | 0.9233 | 0.8422 | 0.7233 |
| trial_0000 ep56 | 0.9053 | 0.8281 | 0.9211 | 0.8370 | 0.7212 |

**关于 mask loss 的核查（结论：正常，无问题）：** trial receipt 中 `0head_`~`4head_` 与 `proposal_` 前缀下的 `loss_mask` / `loss_dice` / `sp_loss_*` / `corresponding_*` / `adaptive_weight_*` 全为 0.0，但这是**设计如此**——`models/losses.py:890` 只在 `prefix == 'last_'` 时把 mask 输出挂进 criterion，其余 decoder 层本就不算 mask 损失。实际 `last__loss_dice=0.792151`、`last__loss_mask=0.011271`、`last__sp_loss_dice=0.195702`、`last__adaptive_weight_loss_dice=0.195780` 均非零，聚合项 `loss_dice=0.792151`、`sp_loss_dice=0.195702` 同样非零。**mask 分支的监督信号一直正常**，该 study 的失败与 mask 损失无关，仅由上述两条根因造成。

**不要按原样重启该 study。** 后续方向改为架构创新，见 `docs/SOURCE_MOE_RERANK_DESIGN.md`。

## SourceMoE + 训练内 Query 重排实施记录（2026-07-27）

本节取代早期“SourceMoE 仅有 balance loss但已经完成”的结论。实际审计发现
旧接线中 `routed_scale=0` 且 balance loss 不依赖它，真实 backward 后
`routed_scale.grad=None`；离散 rank 也阻断源分数梯度。因此旧版本一直输出
`default`，不能作为有效 MoE 实验。

已完成修复：

- `models/source_moe.py`：共享 default 锚点、query 级 top-1 稀疏路由、
  straight-through rank、归一化 balance loss、一层 self-attention query 重排；
- `models/losses.py`：真实 box threshold-aware listwise loss，加 box-tier 约束的
  mask listwise 辅助项，并接入 shared-query 0.25/0.50 阈值锚点损失；只监督
  grounding 的 slot-0 root GT；
- `src/grounding_evaluator.py`：REC 与 position/semantic 两个 mask evaluator 使用
  同一个 learned query，joint geometry parent mapping 仍有更高优先级；
- `main_utils.py` / `train_dist_mod.py`：MoE-only 冻结主干并保持 eval，接入配置、
  optimizer、日志、checkpoint 和 retrain metric receipt；
- candidate/geometry/reranker provenance：新 artifact 记录全部 MoE 结构字段；
  旧 artifact 缺字段时严格解释为 `use_source_moe=False`；
- `scripts/train_scanrefer_source_moe.sh`：router/joint 两阶段；router LR 默认
  `3e-4`；joint 必须显式提供 router checkpoint，不能意外从 epoch-71 重启；
  关键 MoE 超参均可由环境变量覆盖，避免为每组实验改动源码；最后 epoch 已验证
  时跳过一次无参数变化的重复 validation，同时保留整数 epoch checkpoint。

当前专家清单固定为 `default,contrastive_text,mask_text`。不使用含人工
0.05/0.10 常数的 rank-blend 源，避免把 ScanRefer 特调组合带到 Nr3D/Sr3D。

组合边界：当前 `rec-query-v1` 不读取 `selected_source_scores`，geometry sidecar
也会覆盖 MoE query。仅按新 checkpoint SHA 重建旧 sidecar 不会利用 MoE；若
router-only 通过门禁，必须新增 learned-score-aware `rec-query-v2` 后才能重新训练
parent/geometry 并报告组合结果。

### 真实 GPU smoke

目录：
`/root/autodl-tmp/DATA_ROOT/output/source_moe_smoke/scanrefer/ssq_moe_debug/1785085628/`

- 32 train batches + 两次 128-row eval 全部 finite；
- 训练后 `routed_scale=0.0152313`，query score head norm `0.0693894`；
- epoch-71 的 1,135 个共同非-MoE tensors 逐元素完全相同，仅 23 个 MoE
  tensors 发生更新；
- 小切片 learned 对 default 的 fix=break=1.5625%；mask position/semantic
  mIoU 相同，证明同-query 评估生效；
- 这些数只证明链路正确，不能作为正式精度结论。

### 正式 router-only 实验

固定配置：1 epoch，batch 12，完整 ScanRefer train，MoE LR `3e-4`，主干冻结，
每轮完整 9,508-row validation。输出根目录：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_formal/scanrefer/ssq_moe_router_lr3e4_e1/`

启动时 GPU 空闲，保护 checkpoint 为 0444，SHA-256 仍为
`3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`。
启动前完整 CPU 回归为 `2800 passed, 3 warnings in 151.74s`，退出码 0；加入
锚点及训练流程修复后的最终回归为 `2806 passed, 3 warnings in 147.78s`，
退出码仍为 0；
原 parent、geometry artifact 和 train/val candidate manifest 均通过新旧 schema
兼容性实载校验。
该首轮 control 在锚点损失实现前已启动，配置中不含 `L_anchor`。两次完整
validation 完全一致，正式结果如下：

| | default | learned | fix | break | learned delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| REC @0.25 | 5512 (57.9722%) | 5406 (56.8574%) | 126 | 232 | -106 hits |
| REC @0.50 | 4407 (46.3504%) | 4281 (45.0252%) | 210 | 336 | -126 hits |

learned 同-query mask 为 `5647/9508 (59.3921%)`、`4626/9508 (48.6538%)`、
mIoU `41.5661%`，三项也低于保护结果。训练累计 expert usage 为约
contrastive 47.4% / mask 52.6%，entropy 0.6687，说明失败不是路由塌缩；
rerank abs max 0.2456 接近 0.25 上限，属于过度重排。23 个 MoE tensors 全部
finite，1,135 个共同非 MoE tensors 与保护 backbone 逐元素相同。

正式 checkpoint：
`1785087235/ckpt_epoch_1.pth`，size `607936805`，SHA-256
`86bab6dcb96ce0a95cd2a8cf71ae707788e9aef6032abb6489242c12529cd6f2`。
结论：**拒绝进入 joint**。第二组从保护 backbone 重新初始化，使用
anchor weight `2.0`、margin `0.05`、temperature `0.2`、max delta `0.10`；
脚本默认 weight 仍为 `1.0`，本实验用环境变量显式覆盖并由 config 留痕。

完整设计、公式、门禁及跨数据集计划见
`docs/SOURCE_MOE_RERANK_DESIGN.md`。此前“唯一可行路径是从头全量重训”的判断
只适用于当时尝试的局部 mask-head/普通续训，不排除本节新增的可训练选择架构。

### 锚点保护组正式结果与结构决策（2026-07-27）

目录：
`/root/autodl-tmp/DATA_ROOT/output/source_moe_formal/scanrefer/ssq_moe_router_anchor_w2_t02_d010_e1/1785092585/`

配置为完整 36,665-row train、9,508-row validation、router-only 1 epoch、
LR `3e-4`、anchor weight `2.0`、margin `0.05`、temperature `0.2`、
max delta `0.10`。最终 receipt：

| | default | learned | fix | break | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| REC @0.25 | 5512 (57.9722%) | 5464 (57.4674%) | 104 | 152 | -48 hits |
| REC @0.50 | 4408 (46.3609%) | 4320 (45.4354%) | 127 | 215 | -88 hits |

learned 同-query mask 为 `5651/9508 (59.4342%)`、
`4638/9508 (48.7799%)`、mIoU `41.5278%`。expert usage 为
contrastive 48.47% / mask 51.53%，entropy 0.2159；rerank abs mean/max
为 0.0970/0.1000。anchor 显著缩小 control 的退化，但两个阈值仍是
`fix < break`，故按预注册门禁拒绝 joint，也不再继续对 validation 做软约束扫参。

checkpoint `ckpt_epoch_1.pth` size `607936933`，SHA-256
`c73c79529537e106f3d85096fafcf8a8ae2996222bcc49210ed8ae788a372788`。
23 个 MoE tensor 全 finite；1,135 个共有非 MoE tensor 与保护 backbone 逐元素
相同；epoch-1 与验证后 epoch-last 的模型权重逐元素相同。

**理论上限诊断：**只在 learned top-1 与 default 之间做完美 gate，最多得到
`5512+104=5616` hits@0.25 和 `4408+127=4535` hits@0.50；后者仅 47.70%，
无法达到目标 49%。但该 learned 排名的 Top-5 为 61.18%/52.74%，候选集合有
足够 headroom。因此已实现网络内 top-k `SelectiveFallbackGate`：

- pre-gate MoE top-8 query 逐 query 与 default 配对；
- REC 0.25/0.50 分别预测 `break/neutral/fix`，阈值权重 2:1；
- 同结构 mask 辅助 head 保护 mask 指标；
- 以 `P(fix) - 2*P(break)` 的风险收益决定是否切换，否则硬回退 default；
- head 全零初始化，第 0 步严格等于 default；
- class-balanced focal 自动适应各数据集类别比例，并额外提高 false override 代价；
- 新增独立 gate-only 阶段，只训练 `source_moe.fallback_gate.*`，候选 MoE 和主干
  均 frozen/eval。

这仍是训练期网络模块，不是 ScanRefer validation 后处理。新旧 candidate/reranker
provenance 已按完整门控结构字段 fail-closed；旧 artifact 缺门控字段时只解释为
门控关闭。正式 gate 训练结果完成后继续追加到本节。

### 首版概率 utility gate 正式结果与拒绝结论（2026-07-27）

目录：
`/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_formal/scanrefer/ssq_moe_top8_safe_gate_e1/1785098730/`

完整 36,665-row train 和 9,508-row validation 均正常完成，但最终
`switch_ratio=0`、`positive_candidate_ratio=0`。正式 receipt：

| | fixed default | gated | delta |
| --- | ---: | ---: | ---: |
| REC @0.25 | 5512 (57.9722%) | 5512 (57.9722%) | 0 |
| REC @0.50 | 4406 (46.3399%) | 4406 (46.3399%) | 0 |
| Mask @0.25 | - | 5672 (59.6550%) | - |
| Mask @0.50 | - | 4657 (48.9798%) | - |
| Mask mIoU | - | 41.7390% | - |

训练 step 500/1000/1500/2000/2500/3000 的 gate loss 为
`0.4900/0.4619/0.4419/0.4329/0.4254/0.4213`，但最大 utility 均值仅从
`-0.7366` 变为 `-0.6637`，始终没有正候选。根因是 class-balanced focal
刻意改变类别先验，输出 softmax 不是校准概率；再用这些概率重建
`P(fix)-2P(break)` 与训练目标不一致。该版本拒绝进入 joint，也不做
decision-margin validation sweep。

checkpoint SHA-256：epoch-1 为
`5d17535d551ccfd897b6b585de380b3ea260ee23ffc37f66d4c042bcccde0b26`，
epoch-last 为
`3ea250d910cb50115582a6e1e5c0bb236cb3dedcc18e7b37c8d34316b51a8bd2`。
两者文件哈希因 metadata 不同，但 1,169 个 model tensor 逐元素相同；11 个 gate
tensor 全 finite，且输入候选 checkpoint 的 1,158 个共有 tensor 全部逐元素不变。

### 联合 decision-margin gate 修复与验收（2026-07-27）

已新增一个零初始化 `decision_head`，直接预测联合
`break/neutral/fix`；部署 margin 为
`logit_fix-max(logit_neutral, logit_break)`，只在严格大于 0 时覆盖 default。
联合 target 将 REC 0.25/0.50 的真实阈值收益按 2:1 合成，break 代价为 2，并以
0.25 权重加入同 query mask 阈值收益。除候选级 class-balanced focal 外，新增
显式 `fallback + top-8 query` 行级选择 loss：存在正收益时监督最佳 query，否则
监督 fallback。因此训练动作与推理动作一致，同时保持第 0 步 exact identity。

代码完成后的完整 CPU 回归为 `2812 passed, 3 warnings in 166.00s`。真实 128-row
GPU smoke：
`/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_smoke_v2/scanrefer/ssq_moe_gate_joint_decision_debug/1785110631/`

- 72,207 个 gate-only trainable parameters，13 个 gate tensor 全 finite；
- decision/box/mask 三个 head 均获得非零更新；
- 与输入候选 checkpoint 共有的 1,158 个 tensor 全部逐元素相同；
- 10 个更新步后仍安全回退 default，128-row receipt 完整且进程退出码为 0；
- 额外 overfit sanity 在约 80 step 开始输出正 margin，到 epoch 10 的 train
  `switch=11.67%`、REC fix=4.17%、break=1.67%，证明正 margin 和选择动作可学；
  该进程随后主动停止且未保存 checkpoint，不把 train-overfit 数字作为精度证据。

唯一正式 v2 按固定 `margin=0`、top-8、LR `3e-4` 完成：
`/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_formal_v2/scanrefer/ssq_moe_top8_joint_decision_e1/1785111311/`。
完整 9,508-row receipt：

| | fixed default | gated | fix / break | delta |
| --- | ---: | ---: | ---: | ---: |
| REC @0.25 | 5512 (57.9722%) | 5462 (57.4464%) | 50 / 100 | -50 hits |
| REC @0.50 | 4407 (46.3504%) | 4381 (46.07699%) | 60 / 86 | -26 hits |
| Mask @0.25 | - | 5650 (59.4247%) | - | 未保护 |
| Mask @0.50 | - | 4631 (48.7064%) | - | 未保护 |
| Mask mIoU | - | 41.5125% | - | 未保护 |

validation `switch_ratio=6.01%`，说明直接动作监督解决了 v1 的“永不切换”，但
选择精度仍不足；两个阈值均为 `fix < break`，故正式拒绝 joint，也不续跑同结构
epoch 2。checkpoint epoch-1/last SHA-256 分别为
`4ab392305e592ef43049566db3606f81af96c315dbb13ad908baafeaee2d060b` 和
`5a95299ee1991e0f08aa42d895ca57773f0ccfff80489c209ad26480c388503e`；
两者 1,171 个 model tensor 逐元素一致。13 个 gate tensor 全 finite，输入 anchor
checkpoint 的 1,158 个共有 tensor 全部逐元素不变。

结论：当前 gate 输入只含 64 维 contrastive query 投影、rank、归一化 box 和 pooled
text；overfit 可学但 validation 不可分。v3 只增强通用证据：加入冻结 decoder 的
288 维完整 query 表征和每个源跨 query 标准化后的原始置信度，不调 validation
margin、break cost 或 mask utility。

### V3 enriched-evidence gate：实现、异常与正式拒绝（2026-07-27）

V3 加入冻结 decoder 的 288 维 query feature 和逐源标准化 raw confidence，
gate-only 可训练参数为 183,951，仍是 13 个 gate tensor。真实 debug smoke：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_smoke_v3/scanrefer/ssq_moe_top8_enriched_evidence_debug/1785116508/`

- checkpoint SHA-256：
  `0bdfb25a6247a2b0c7183c96a6a9b19b79cb2232d6e402618c8b557ab38869ac`；
- 13 个 gate tensor finite，1,158 个 anchor/common tensor 全不变，三个 head
  均更新；debug switch 13.64%，fix/break 为正；
- 训练、保存和验证链路成功，但第一次 harness 仍硬编码期望 9,508 条，随后新增
  `EXPECTED_EVAL_SAMPLE_COUNT`，debug 固定为 128。

第一次 formal 目录 `.../1785116965/` 在约 step 130 主动停止且无 checkpoint。
原因不是 NaN，而是 raw-score tie 时 gate、loss 和 evaluator 各自重新求 default，
得到不同 query。修复后 `moe_shared_query` 成为 gate、loss、evaluator diagnostics
和旧输出 fallback 的唯一事实来源；不能用 raw `argmax` 替代 rank/argsort 语义。
修复后的完整 CPU 回归为 `2823 passed, 3 warnings`。

唯一完成的 V3 formal：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_formal_v3/scanrefer/ssq_moe_top8_enriched_evidence_e1/1785118054/`

| | fixed default | gated | fix | break | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| REC @0.25 | 5512 (57.9722%) | 5419 (56.9941%) | 113 | 206 | -93 |
| REC @0.50 | 4408 (46.3609%) | 4305 (45.2787%) | 148 | 251 | -103 |

同-query mask 为 `5634/9508 (59.2554%)`、`4618/9508 (48.5696%)`、
mIoU `41.3727%`；switch ratio 14.71%。更多证据使 gate 过度切换，所有保护指标
均下降。checkpoint `ckpt_epoch_last.pth` size `604018734`，SHA-256
`adea0787e997ab54691dc65df2872241d975533ffad2dcdae0aacbfd3ba01302`。
13 个 gate tensor 全 finite，1,158 个共有 tensor 与 router anchor 逐元素相同，
三个 head 都有非零更新。结论：V3 正式拒绝，瓶颈是 selective calibration，
不是继续堆 feature。

### V4 evaluator 语义修正与 calibrated utility（2026-07-27）

V4 没有新增网络参数，做两项可泛化修正：

- `default` 源先把主目标 `positive_map > 0` 二值化，严格复刻官方 evaluator；
- 新增 `--source_moe_gate_objective calibrated_utility`，直接用 SmoothL1 把部署
  margin 回归到连续真实 utility；margin 高估按 `false_override_weight=2` 加罚，
  decision/row CE 使用固定 cost，不再按 batch 类别逆频率扭曲先验。

默认 objective 仍是 `balanced_focal`，V1-V3 路径保持不变；已有 gate 续训从
checkpoint config 继承 objective，旧 checkpoint 缺字段时回退 legacy。新增测试
覆盖 evaluator-compatible 二值化、CLI 默认/显式 objective、legacy 等价、finite
梯度、utility 高估加罚和旧 checkpoint 兼容。完整 suite：
`2828 passed, 3 warnings in 134.80s`。

关键训练源码 SHA-256：

- `models/source_choice_adapter.py`：
  `3adea9899c5f0b84204de22005f360dd4d98c80691901a654dafb177b920be49`
- `models/source_moe.py`：
  `4c1e1803b1cd8d6f400b0de7060eefaa99e106a28dc6bd2f476c68bcd1faf372`
- `models/losses.py`：
  `7c477ebe40dd4941909da7fc671b11eeeadf929a9071f023511e34d40ce9f245`
- `main_utils.py`：
  `a1505685961646c5f56700e8ee3718a36d27ec42803052df121256b1c0f83b48`
- `scripts/train_scanrefer_source_moe.sh`：
  `a75db60206d86a52636f690226086c05b7a5939c35f0b36b6336d019c2b71aae`

128-row GPU smoke：

`output/source_moe_gate_smoke_v4/scanrefer/ssq_moe_top8_calibrated_utility_debug/1785123480/`

fixed default 为 `63/128 (49.2188%)`、`57/128 (44.5313%)`；gated 为
`66/128 (51.5625%)`、`60/128 (46.8750%)`，两个阈值均 `fix=3, break=0`，
switch ratio 2.27%。checkpoint SHA-256：
`4ad15e242d3b0d01a80efb08e6b7ecb018be99620171676206f7dc7c94bce2d0`。
13 个 gate tensor 全 finite；box/mask/decision head 与 encoder 均更新；输入 anchor
的 1,158 个共有 tensor 全逐元素不变。该结果只通过链路与方向门禁，不作为精度。

唯一 formal 已按 margin 0、top-8、LR `3e-4`、break cost 2、mask utility 0.25
完成，输出：

`output/source_moe_gate_formal_v4/scanrefer/ssq_moe_top8_calibrated_utility_e1/1785123878/`

完整 9,508-row receipt：

| | fixed default | V4 gated | fix | break | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| REC @0.25 | 5514 (57.9933%) | 5516 (58.0143%) | 4 | 2 | +2 |
| REC @0.50 | 4408 (46.3610%) | 4411 (46.3925%) | 4 | 1 | +3 |

同-query mask 为 `5677/9508 (59.7076%)`、`4662/9508 (49.0324%)`、
mIoU `41.7754%`。validation switch ratio 仅 `0.16%`：V4 消除了 V3 的过度
切换，在两个 REC 阈值均取得严格正收益且保护 mask，但 2/3 hits 的幅度仍远低于
`0.59/0.49` 目标。

checkpoint `ckpt_epoch_last.pth` size `604018862`，SHA-256：
`aaeb4b9ca091e5393f55f29c91b618207e4dda6d231a706576fc73fed9eb5022`。
13 个 gate tensor 与 optimizer state 全 finite；三个 head 与 encoder 均更新；输入
anchor 的 1,158 个共有 tensor 全部逐元素不变。为衔接真实总轮次 72-80，另生成
只把 epoch metadata 改为 71 的 launch-only 副本，1,171 个模型 tensor 逐元素不变，
SHA-256：`e9677636a35d4f33cfc19a1e8d8fb0a1b57966a25611681cae9c5fbfb09bef45`。

### V4 模型级联合训练到总轮次 80（2026-07-27）

实际运行目录：

`/dev/shm/source_moe_joint_full_v4/scanrefer/ssq_moe_v4_joint_e72_e80/1785128173/`

已从上述 launch-only 副本成功加载，实际 `config.json` 确认：

- `joint_det=true`，训练集合为 ScanNet detection + ScanRefer，共用 ScanRefer validation；
- 真实轮次 `72-80`，每轮保存 checkpoint 并跑 9,508-row validation；
- router：top-1、`T=0.2`、anchor weight 2、`max_delta=0.10`；
- gate：top-8、enriched evidence、`calibrated_utility`、break cost 2、mask utility 0.25；
- LR：decoder `2e-5`、PointNet `2e-4`、text encoder `3e-6`、SourceMoE `3e-4`；
- 不做 validation margin/源组合 sweep；逐轮同时记录 fixed/gated REC 和全部 mask 指标。

原方案把 checkpoint 暂存于 36 GB `/dev/shm`。该进程在 dataset loading 阶段被
环境中断，未完成 epoch 72，未生成联合 checkpoint 或 validation receipt；随后
`/dev/shm` 已清空。以上配置核验只证明命令构造正确，不能视为已完成训练。

### V4 持久化重启与权重清理（2026-08-01）

不再生成 epoch-71 临时副本，直接加载唯一 V4 formal checkpoint：

`output/source_moe_gate_formal_v4/scanrefer/ssq_moe_top8_calibrated_utility_e1/1785123878/ckpt_epoch_last.pth`

新增 `--checkpoint_start_epoch 72`，用于首次 joint 初始化时覆盖 checkpoint 中的
非数字 `epoch="last"`。首次 joint 仍用 `--reduce_lr` 建立正确的四组 fresh
optimizer；若中断，则用 `JOINT_RESUME=1` 加载数值 epoch 的 model、optimizer、
scheduler，并自动从下一轮继续，不再强制 `--reduce_lr`。

新增持久化 checkpoint retention：每个完成轮次先原子写入，`ckpt_epoch_last.pth`
硬链接到最新轮；验证后分别维护 learned REC@0.25、REC@0.50、mask@0.25、
mask@0.50 和 mask mIoU 五个最佳硬链接。只删除既非 latest、也不是任何指标最佳
的普通 epoch 权重。输出根目录改为：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_full_v4_persistent/`

已实际启动的持久 run（tmux 会话 `mcln_v4_joint_72_80`）：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_full_v4_persistent/scannet,scanrefer/ssq_moe_v4_joint_e72_e80/1785523835/`

启动验收：完整加载 48,655 条 train 与 9,508 条 validation，V4 checkpoint 明确
打印 first requested epoch 72；前 100 个 batch 的 loss/gradient finite 检查全部通过。
batch-100 均值：total loss `9.8007`、MoE rank `1.5580`、gate `1.3232`、utility
regression `0.6289`、anchor `0.0574`。A100 占用约 26.35 GiB。

同类联合 checkpoint 历史大小约 794 MB；即使 latest 和五项 best 恰好来自六个
不同 epoch，唯一数据约 4.8 GB，另加一次原子写临时空间后仍低于启动时 11 GB
可用空间。best/last 硬链接不重复计费。

本次清理严格按已拒绝实验的完整文件名执行，共删除 15 个 `.pth`、
`9,065,199,334` bytes（约 8.44 GiB）。所有日志、配置、receipt 保留；以下保护资产
均仍存在且未改动：

- `0.58288/0.48601` 系统的 epoch-71 backbone；
- parent reranker 与 selected geometry reranker；
- router anchor `ckpt_epoch_1.pth`；
- V4 formal checkpoint，SHA-256 仍为
  `aaeb4b9ca091e5393f55f29c91b618207e4dda6d231a706576fc73fed9eb5022`。

新增 5 个聚焦测试，完整 CPU suite 为 `2833 passed, 3 warnings in 140.59s`。

### Joint batch 的 MoE 监督污染修复（2026-08-01）

上述持久 run 在 epoch 72 的 `497/4054` 主动停止；没有 checkpoint，也没有正式
validation。停止原因不是数值异常，而是训练中只读审计发现样本身份契约错误：

- `src/joint_det_dataset.py` 返回的 `language_dataset` 对 joint batch 中所有样本都
  是 `scanrefer`，包括由 `anno['dataset']=='scannet'` 生成的检测 prompt；
- `models/losses.py` 原先据此执行 `name != 'scannet'`，所以实际 sample mask 全为
  True；
- ScanNet detection prompt 包含多个目标框，但 SourceMoE 的 referring ranking
  仅用 root GT slot-0，约四分之一训练样本因此提供错误的 query/gate 监督。

修复采用两个互不混淆的字段：`language_dataset` 继续表示 benchmark loss 配方，
`sample_dataset` 表示每条 annotation 的真实来源。SourceMoE 的 rank、fallback
gate、anchor、mask utility 和 balance loss 只在 `sample_dataset != scannet` 上
生效；缺少逐样本字段时直接报错。日志新增
`source_moe_supervised_sample_ratio`，用于确认联合 batch 不再全部参与 MoE 监督。

代码 SHA-256：

- `src/joint_det_dataset.py`：`924e609be2e893c2b39e79865af61e02da049bc52c84b5c1857634c97eea53d2`
- `models/losses.py`：`5db5a5dc65235b20f6441db8e1922204ce61ce13c4ab4efb6c0884d5a6057ce6`
- `tests/test_source_moe_integration.py`：`a2112ff303afeb31b8e1a362b3c3ba247006344460f8c8c829d2ac6a6a76e39b`

验证：定向 `115 passed in 2.94s`；完整 suite
`2835 passed, 3 warnings in 136.55s`。无效 run 的日志保留作审计，但禁止恢复；
修复后的完整训练从 V4 formal checkpoint 重新开始。

修复后的正式 joint run：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_full_v4_samplemask/scannet,scanrefer/ssq_moe_v4_joint_e72_e80_samplemask/1785525545/`

tmux：`mcln_v4_joint_72_80_samplemask`。配置仍为 epoch 72-80、batch 12、每轮
9,508-row validation 和五指标独立 retention。ScanRefer/ScanNet 样本数为
36,665/11,990，理论 supervised ratio `0.753571`；batch 25、50、100、125 累计
实测为 `0.7100/0.7167/0.7367/0.7453`。batch-125 total/rank/gate loss 为
`9.4204/1.2179/1.2474`，全部 finite。

### Fresh joint scheduler 的全局轮次错位修复（2026-08-01）

第二次审计发现上述 sample-mask run 仍不能作为最终实验：`--reduce_lr` 不加载旧
optimizer/scheduler，新的 `MultiStepLR` 从 step 0 开始，但命令仍传全局 milestone
`50 75`。因此总轮次 72-80 内不会触发任何衰减，原训练应在 epoch 76 使用的第二次
0.1x LR 被遗漏。该 run 在 epoch 72 约 `353/4054` 停止，目录中 `.pth` 数量为 0。

修复：`scripts/train_scanrefer_source_moe.sh` 新增经校验的
`LR_DECAY_EPOCHS` 环境变量。用当前 PyTorch 和每 epoch 2-step 的同构模拟确认，
相对 milestone `3` 在 epoch 75 最后一步后将全部参数组降为 0.1x，epoch 76 首步
即使用新 LR。最终启动固定 `LR_DECAY_EPOCHS=3`。

代码 SHA-256：

- launcher：`3aae9ca9db549bae71e79372638cda8c5588204728a45c621d24300a062d19d1`
- integration test：`6c35bbe60e72e747a807bee10625d064f26c34653573f2d8d4ab0f7ce5fbdefe`

验证：`bash -n` 通过；非法 `3 invalid` fail-closed，退出码 2；聚焦测试
`52 passed, 2 warnings`；完整 suite `2836 passed, 3 warnings in 147.20s`。

最终有效 run：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_full_v4_valid/scannet,scanrefer/ssq_moe_v4_joint_e72_e80_samplemask_sched3/1785527065/`

tmux：`mcln_v4_joint_72_80_valid`。实际配置已确认 `lr_decay_epochs=[3]`、总轮次
72-80、sample-mask 修复、9,508-row 每轮验证及五指标独立 retention。数据初始化
于 `03:52:43` 完成，训练/验证条目数为 `48,655/9,508`；V4 formal 权重成功加载，
启动日志明确覆盖 checkpoint 的 `epoch="last"` 并从 epoch 72 开始。首个
batch 25/50/75/100 的 supervised ratio 为 `0.7100/0.7167/0.7467/0.7367`，
正向理论 `0.753571` 收敛。batch 100 的 total/rank/gate loss 为
`9.4031/1.2057/1.2595`，全部 loss/gradient finite；A100 显存约 `26.34 GiB`。
该 run 已通过启动与前 100 个真实 batch 验收，继续训练至 epoch 80。

batch 550 的累计统计显示 supervised ratio `0.7483`，candidate fix-break 净值约为
`+1.49pp@0.25/+1.39pp@0.50`；但 gate fix 与 break 都约 `0.09%`，实际 switch
`0.33%`，远低于 oracle switch `24.82%` 和 decision-target fix `10.97%`。这是早期
under-switch 信号，但当前只完成 epoch 72 的 13.6%，训练 loss/gradient 仍完全 finite，
因此不做中途超参干预；等待 9,508 条完整 validation 再判断是校准速度还是结构瓶颈。

### 预备架构：候选集上下文 fallback gate（等待 V4 完整验证后决定）

现有 gate 对每个候选仅相对 shared/default 做独立 MLP 打分；虽然上游 reranker 有
query attention，最终 override 决策并不直接看到 top-k 候选集之间的竞争关系。现有
calibrated utility 的 row-level selection 还把所有正 utility 候选压成单一 `argmax` 目标，
当多个 query 跨过相同 IoU tier 时会引入任意的 tie label。

如果当前正式 run 仍未达到 learned REC `0.59/0.49`，下一项模型实验固定为：从 MoE score
取 default + K 个非 default alternative，在该小集合上跑 zero-init residual self-attention，
并以 fallback utility=0、候选 calibrated box+mask utility 构成 tie-aware soft action
distribution。训练使用 setwise KL/soft CE，同时保留当前 per-candidate quality、break cost
和过高 utility 预测惩罚；推理不新增数据集阈值或 source 组合。实现验收包括 exact default
identity、候选排列等变性、tie target、sample-mask 与 finite gradient。该设计会作为
ScanRefer 单阶段、Nr3D、Sr3D 的同一可迁移模块，而不是 ScanRefer 专用后处理。

### Gate candidate-set oracle 诊断补齐（2026-08-01）

历史日志中的 `oracle` 只在 `default/contrastive_text/mask_text` 三个源各自 top-1
query 之间取最大 IoU，并不表示 fallback gate 的 default + top-k query action space
上限。验证器现新增：

- `gate_candidate_oracle`：对 default query 与 `moe_gate_candidate_mask` 并集取真实
  box IoU oracle，并报告 Acc@0.25、Acc@0.50、mean IoU；
- `gate_oracle_headroom`：default 在阈值下错误但该 candidate-set oracle 正确的比例。

这两个量决定下一步应优化 contextual decision，还是先改善/扩大 candidate set。改动仅限
验证统计，不改变模型 forward、loss、checkpoint 或五指标 retention receipt。聚焦回归
`96 passed`；完整回归 `2838 passed, 3 warnings in 158.88s`。实现 SHA-256：

- evaluator：`9d0006af6cd30a90d660f60906ea4d30cdd1182e6aa354b30299ce2292cd0109`
- test：`222fa0810aa51fa544520a84cc95f7e6ffb8595cbd7e03bd19f85c36b4d7e825`

正式训练进程在补丁前已加载旧 evaluator，因此不重启、不干扰 epoch 72-80；首个 checkpoint
生成后用当前代码独立 eval 才会得到新 oracle 字段，原五项指标仍由内置验证正常产出。

保护资产复核：历史 `0.582878/0.486012` 的 backbone、parent reranker、geometry
reranker 均存在且权限为 `0444`。其 SHA-256 依次为
`3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`、
`f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b`、
`835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f`。

### Contextual gate 完成与 epoch-72 中途状态（2026-08-01）

候选集 contextual fallback gate 已完成默认关闭实现和边界验收。开启后 K 是 K 个非
default alternatives，default 作为独立 fallback action；小集合 self-attention 使用
zero-init residual scale。soft setwise target 以 fallback utility=0 和正 candidate utility
构造分布，invalid/negative/zero utility candidate 概率为零，temperature=0 保留旧 hard
target。补充了单 query/零 alternative、非法 context 配置、排列等变、集合隔离、finite
gradient、tie target、checkpoint context mismatch 测试。

同时补上 joint optimizer resume 合同：恢复 optimizer/scheduler 前逐项比较 checkpoint 与
runtime 的 SourceMoE source/query/gate/rank/anchor/setwise 配置。尤其
`source_moe_gate_objective`、`source_moe_gate_setwise_temperature` 或 gate loss weight 不一致
会立即报错；有意换目标必须使用 fresh optimizer。

独立 eval 的非 tensor 配置恢复和结构化 oracle receipt 也已完成：SourceMoE eval 在模型
构建前继承 checkpoint 的 source/top-k/query/gate contract；candidate-set oracle 另写
`source_choice_diagnostics_epoch_<N>.json`，不改变原五指标 schema。专用 launcher 在 GPU
存在 compute process 时默认以退出码 3 拒绝启动。真实 V4 checkpoint 的 CPU 配置恢复
审计通过。

评估完成后 launcher 自动运行五项联合审计：REC `0.59/0.49` 达标但任一 mask 指标相对
V4 退化时返回 `repair_mask_tradeoff`；candidate oracle 达标但 learned 未达标时返回
`train_contextual_gate`；oracle 自身不足时返回 `improve_candidate_generation`。最终完整回归
`2870 passed, 3 warnings in 161.72s`。

正式 tmux `mcln_v4_joint_72_80_valid` 未重启；观察到 epoch 72 batch `1525/4054`：
supervised ratio `0.7538`，total/rank/gate loss `9.5497/1.2431/1.2414`，gate switch
`0.0320`，oracle switch `0.2855`，全部 finite。这里的 selected train-batch 累计诊断
`0.6005/0.4360` 不是 9,508-row validation，不用于宣称达到目标。epoch 72 完成后仍按
五指标 receipt 和独立 gate candidate-set oracle 判断下一步。

V4 formal 与 router anchor 也已改为 `0444`：

- V4：`aaeb4b9ca091e5393f55f29c91b618207e4dda6d231a706576fc73fed9eb5022`
- router：`c73c79529537e106f3d85096fafcf8a8ae2996222bcc49210ed8ae788a372788`

若 candidate-set oracle 足够而 learned gate 仍不足，V5 固定从无 gate 的 router anchor
训练 `context_layers=1, heads=4, top_k=8, setwise_temperature=0.25`，先做两轮 gate-only
完整验证，再决定是否进入 epoch 72-80 联合训练；当前 V4 正式 run 占用 GPU，禁止并发。

epoch 72 的独立评估固定使用：

```bash
CHECKPOINT_PATH=/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_full_v4_valid/scannet,scanrefer/ssq_moe_v4_joint_e72_e80_samplemask_sched3/1785527065/ckpt_epoch_72.pth \
EVAL_EPOCH=72 \
LOG_DIR=/root/autodl-tmp/DATA_ROOT/output/source_moe_joint_v4_oracle_eval \
bash scripts/eval_scanrefer_source_moe_checkpoint.sh
```

只有在正式训练释放 GPU 后运行；launcher 自动要求两个 JSON receipt 的
`sample_count == 9508`，并生成 `source_moe_oracle_audit.json`。

### V4 对照复核、联合训练止损与 V5 决策（2026-08-01）

正式 joint run 的 epoch 72 完整验证确认发生 REC 分支崩塌：learned/fixed REC
分别仅为 `0.000526/0.000105`，而 mask 为
`0.585612/0.475494/0.406370`。epoch 73 前 100 batch 仍接近零，因此停止该 run，
不恢复其 optimizer。参数审计无 NaN/Inf，但 backbone、decoder、prediction heads 和
SourceMoE 相对 V4 分别变化约 `2.59%/2.62%/4.84%/43.89%`，同时 BN running stats
明显漂移，证明全模型 joint update 破坏了原有 box/REC 表征。

随后用当前 evaluator 对只读 V4 checkpoint 做独立 9,508-row 对照，结果为：

- learned REC：`0.580143/0.464030`；fixed default：`0.579933/0.463715`；
- mask：`0.596761/0.490324/0.417768`；
- gate candidate-set oracle：`0.630206/0.549642`，mIoU `0.451155`；
- 相对 default 的 oracle headroom：`+5.027pp@0.25/+8.593pp@0.50`；
- 自动审计结论：`train_contextual_gate`。

这同时证明 evaluator 合同正确、候选集合上限足够，瓶颈是 V4 逐候选 gate 的决策能力
（实际 switch 约 `0.16%`），不是 proposal 不足。下一实验固定从无 fallback gate 的
只读 router anchor 初始化，仅训练新建的 contextual fallback gate：一层、四头、top-8、
evidence features、calibrated utility、setwise temperature `0.25`。backbone、decoder、
mask head 和原 SourceMoE 全部冻结；通过两轮完整 ScanRefer 验证后才允许进入新的
epoch 72-80 joint 实验。

V4 对照收据：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_v4_contract_eval/scannet,scanrefer/ssq_moe_v4_contract_eval/1785534811/`

已按低指标清理策略删除崩塌 epoch 72 的七个 checkpoint 硬链接（同一 inode，约
`804 MB`），保留 `log.txt`、`config.json`、validation receipt 和 retention receipt
用于审计。历史 `0.582878/0.486012` 三件套、V4 和 router anchor 均保持只读且未删除。

V5 gate-only 正式 run 已启动：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_contextual_v5/scanrefer/ssq_moe_context_top8_setwise_t025_e2/1785535922/`

tmux 为 `mcln_v5_gate_contextual_e1_e2`。运行时配置复核为 ScanRefer
`36,665/9,508`、epoch 1-2、batch 12；router anchor 成功加载。optimizer 输出确认仅
`source_moe.fallback_gate.*` 可训练，共 `316,432` 参数。gate-only 训练循环先将整网设为
eval，再仅启用 fallback gate 的 train mode，所以冻结参数、BN running stats 和主干
dropout 状态均不会漂移。

### Balanced calibrated utility 预备目标（2026-08-01）

V5 epoch 1 的 batch 500/1000/1500/2000 显示 gate loss 从
`1.3465 -> 1.2893 -> 1.2616 -> 1.2483`，context residual scale 从
`0.0243 -> 0.0363 -> 0.0429 -> 0.0479`，说明 contextual 模块在获得梯度；但 learned
switch 始终为 `0`，而 row-level oracle switch 稳定约 `17%`，mean max margin 约
`-0.95`。这表明 fixed-cost calibrated objective 对稀有正 switch 的先验仍过于保守。

为缩短 V5 完成后的迭代，新增默认关闭的
`balanced_calibrated_utility`：decision 三分类和 row-level setwise action 都使用 batch
内逆频率平衡，break 类继续乘 false-override cost，同时保留 calibrated utility regression。
原 `balanced_focal` 默认值、`calibrated_utility` 数值路径和 V5 已加载进程均不改变。
聚焦单元/集成测试为 `83 passed in 6.73s`。只有 V5 两轮正式验证确认 under-switch 后，
才从同一 router anchor 启动该目标，不能把它混入当前 run。

V5 epoch 1 的 9,508-row 正式验证结果：learned REC
`0.580248/0.463820`，fixed default `0.579933/0.463504`，mask
`0.596550/0.490114/0.417634`。相对 V4 learned REC 仅
`+0.0105pp/-0.0105pp`，三项 mask 轻微下降；deployed switch `0.09%`，而 gate
candidate-set oracle 为 `0.633361/0.552798/0.454365`，headroom 达
`+5.343pp/+8.929pp`。审计收据为 `source_moe_oracle_audit_epoch_1.json`，结论仍是
candidate pass、learned fail。V5 epoch 2 已自动开始，用于严格区分训练时长与目标函数问题。

为后续完整训练增加了 `source_moe_train_only` 续训保护：从已训练 SourceMoE
checkpoint 继续训练时，启动器会在构建模型前恢复 `query_max_delta`、source/top-k、
fallback gate、context 和 objective 等非张量合同；从普通 MCLN checkpoint 首次创建
SourceMoE 的路径保持原行为。该模式只将 `source_moe.*` 设为可训练并令主干保持 eval，
可作为 V6 gate-only 通过后的 72-80 轮安全延伸，避免旧 joint run 的 BN/主干漂移。

当前磁盘审计还确认，旧记录中的 V4 formal checkpoint
`output/source_moe_gate_formal_v4/.../1785123878/ckpt_epoch_last.pth` 已不存在，且没有
进程持有该删除文件；V4 的独立评测收据仍保留，不能再把该路径当作可加载保护权重。
历史 `0.582878/0.486012` 三件套和 router anchor 仍完整，并在
`/root/autodl-tmp/DATA_ROOT/protected_mcln_artifacts/` 中另置有额外只读硬链接保护。

### V5 完成、V6 启动与首轮中途诊断（2026-08-01）

V5 epoch 2 已完成 9,508-row 正式验证。learned REC 为
`0.580143/0.463715`，fixed default 为 `0.579933/0.463504`，mask 为
`0.596550/0.490219/0.417589`；candidate-set oracle 仍为
`0.633361/0.552798/0.454365`。deployed switch 约 `0.13%`，oracle switch 约
`11.91%`，自动审计结论仍为 `train_contextual_gate`。因此两轮 control 已排除“只差一轮
训练”的解释，固定代价 calibrated objective 确实欠切换。

V5 的三个约 605 MB checkpoint 已从实验目录移动到可恢复隔离区：

`/root/autodl-tmp/DATA_ROOT/quarantine_low_quality/source_moe_v5_contextual_e2_1785535922/`

原实验目录只保留配置、日志、两轮正式指标、candidate oracle 和审计收据。隔离区暂未永久
删除，历史最佳三件套与 router anchor 不在任何清理扫描范围内。

V6 已从只读 router anchor 启动：

`/root/autodl-tmp/DATA_ROOT/output/source_moe_gate_balanced_v6/scanrefer/ssq_moe_context_top8_setwise_t025_balanced_calibrated_e2/1785544218/`

tmux 为 `mcln_v6_balanced_gate_e1_e2`。运行配置确认只有
`source_moe.fallback_gate.*` 的 `316,432` 个参数可训练；checkpoint 中的
`query_max_delta=0.10` 已在模型构建前恢复。其余固定为 context 1 层/4 头、top-8、
evidence features、setwise temperature `0.25`、gate LR `3e-4`、每轮 9,508-row 验证和
五指标 checkpoint retention。

V6 batch 500/1000 的累计 gate loss 为 `1.2557/1.2057`，context scale 为
`0.0258/0.0409`，oracle switch 为 `17.22%/16.83%`。部署 switch 为
`0.07%/0.03%`，最大 margin 为 `-0.5667/-0.5749`；相较 V5 同期约 `-0.95` 已明显
接近决策边界，但真实有益 switch recall 仍为零。该中途统计只用于诊断，不替代完整验证；
必须等待 epoch 1 的正式收据后再决定是否进入安全的 epoch 72-80
`source_moe_train_only` 续训。

### V6 epoch 1 正式结果与 V5 权重清理（2026-08-01）

V6 epoch 1 已完成全量 9,508-row 验证。learned REC 为
`0.578881/0.462663`，fixed default 为 `0.579933/0.463610`，mask 为
`0.595499/0.488851/0.416660`。相对现存 V4 合同评测基线，learned REC 分别下降
`0.1262pp/0.1367pp`，三项 mask 分别下降 `0.1262pp/0.1472pp/0.1108pp`，因此
learned target 和 mask guard 均未通过。

同一候选集合的 gate candidate oracle 仍达到
`0.633572/0.552798/0.454344`，相对当前 learned selector 的 headroom 为
`+5.364pp/+8.919pp`。这再次排除了候选能力不足，但 balanced objective 的部署 switch
仅约 `1.24%`、oracle switch 约 `11.92%`，验证集上仍表现为低 recall 且误切换偏多。
自动审计结论为 `train_contextual_gate`。epoch 2 继续完整运行，作为训练时长对照；该结果
不满足进入 epoch 72-80 `source_moe_train_only` 的原性能门禁。

epoch 1 checkpoint 的 tensor 审计确认 1,158 个非 gate tensor 全部与 router anchor
逐项一致，无缺失、新增或非有限值；optimizer 只包含 fallback gate 状态，因此上述退化不是
主干漂移或预训练权重未加载。五指标 retention 已为 epoch 1 建立同 inode 硬链接。正式收据
位于 V6 run 目录的 `eval_metrics_epoch_1.json`、
`source_choice_diagnostics_epoch_1.json` 和 `source_moe_oracle_audit_epoch_1.json`。

V6 epoch 1 收据和 checkpoint 完整性确认后，已永久删除隔离区内三个独立 V5 低质量
checkpoint，释放约 `1.69 GiB`；V5 的配置、日志、两轮指标和审计收据继续保留。删除前后
均核对 inode：历史 `0.582878/0.486012` 的 backbone、parent reranker、geometry reranker
及 router anchor 与 V5 文件无硬链接关系，四个保护资产仍为 `0444`、link count `2`，
未被修改。

用户随后明确要求完成总轮次 70-80 的训练，因此原“V6 先通过性能门禁才长训”约束被
实验时长对照需求覆盖，但安全门禁不变：V6 epoch 2 完成并产生 9,508-row 收据后，从
两轮中 `REC@0.25` 最佳 retained checkpoint 启动 epoch 72-80。该 V7 使用 fresh
optimizer，只训练 `source_moe.*`，主干保持 eval；loss 固定为 rank `1.0`、mask-rank
`0.25`、anchor `2.0`、temperature `0.2`、balanced gate `1.0`，相对 scheduler
milestone 为 `3`。每轮全量验证并只保留 latest 与五项独立最佳，不恢复曾导致 REC 崩塌的
full-joint optimizer。等待会话为 `mcln_v7_source_only_e72_e80`。

V6 epoch 2 正式结果：learned REC `0.580143/0.463715`，fixed default
`0.579933/0.463610`，mask `0.596761/0.490219/0.417715`。相对 epoch 1 的
`0.578881/0.462663`，REC 和三项 mask 均有提升，但仍未达到 `0.59/0.49`；candidate
oracle 保持 `0.633572/0.552798/0.454344`。retention 五项最佳均已切换到 epoch 2，
V7 实际从 `ckpt_best_rec_acc025.pth`（epoch 2）启动。

V7 已于 `10:45:45` 通过配置恢复验收并开始加载 ScanRefer：
`output/source_moe_continue_v7/scanrefer/ssq_moe_context_balanced_source_only_e72_e80/1785552345/`。
启动配置确认 `source_moe_train_only=true`、`start_epoch=72`、`max_epoch=80`、
`query_max_delta=0.1`、fallback/context/objective 从 V6 checkpoint 恢复，且只建立
SourceMoE 参数组；首批日志出现后继续做 finite loss/gradient 和主干冻结审计。

`10:54:10` 数据与模型初始化完成：train/val 为 `36,665/9,508`，日志明确加载 V6
epoch 2 checkpoint；optimizer allowlist 为完整 `source_moe.*`，共 `1,082,004` 个可训练
参数，主干参数不在 optimizer。epoch 72 已开始，首批 GPU 占用约 `8.2 GB`。DDP 打印的
`find_unused_parameters=True` 警告仅提示额外 autograd 遍历，不是 loss/gradient 异常。

Epoch 72 batch 500 的首次 finite 审计通过：total/rank/anchor/gate loss 为
`12.0598/0.9383/0.0307/1.1093`，router balance `1.0073`，全部有限。部署 switch
约 `0.52%`，oracle switch 约 `16.83%`，context scale `0.0991`，routed scale
`0.5255`；完整 SourceMoE 参数正在获得梯度。GPU 约 `8.21 GB`，磁盘余量约 `9.9 GB`，
足够支撑 latest + 五项 best retention。

独立 SourceMoE 评测启动器的默认 baseline 已从不存在的 V4 formal 路径改为现存的
`source_moe_v4_contract_eval/.../1785534811/eval_metrics_epoch_1.json`。shell 语法检查、
真实 baseline auditor 和集成测试均通过，集成测试为 `25 passed`；V6 正在运行的进程不受
该修改影响。

### V7 epoch 72 正式结果（2026-08-01）

V7 epoch 72 已完成完整 9,508-row 验证。learned REC 为 `0.5806689/0.4643458`，fixed
default 为 `0.5799327/0.4637148`；Mask 为 `0.5970761/0.4907446/0.4178385`。
相对 V6 epoch 2，REC 仅增加 `+0.0005259/+0.0003155`，Mask 三项均小幅增加，仍未达到
最终 `0.59/0.49` 目标。候选 oracle 为 `0.6349390/0.5553218`、mIoU `0.4557727`，
说明候选集合仍有足够上限，但 learned gate 没有把上限转化为部署增益；审计结论为
`candidate_oracle_target_pass=true`、`learned_target_pass=false`、`mask_guard_pass=true`。

epoch 72 验证诊断中，gate candidate oracle 相对 learned headroom 为
`+0.055006/+0.091607`，learned gate 的实际 switch 约 `0.12%`；训练/验证中的 neutral
decision 占多数，提示 utility regression 和三分类决策对大量 neutral 候选仍有保守偏置。
继续完成 epoch 73-80，用高学习率阶段和 epoch 76 后低学习率阶段区分“训练轮数不足”和
“gate 校准目标欠切换”，不在运行中改变实验合同。

正式收据位于：
`/root/autodl-tmp/DATA_ROOT/output/source_moe_continue_v7/scanrefer/ssq_moe_context_balanced_source_only_e72_e80/1785552345/`。

在不干扰 V7 进程的前提下，新增了可选的网络内 `expected_utility` action mode：复用已训练
的 box/mask threshold heads 计算期望 break/neutral/fix 收益，作为 gate 的部署 margin。
它没有新增参数、默认保持 legacy `decision`，用于完整训练结束后对同一 checkpoint 做泛化性
更强的 action 消融，而不是 ScanRefer 专用后处理。实现和配置兼容性已通过全套
`2881 passed, 3 warnings`；V7 当前未启用该模式。

V7 epoch 73 的 9,508-row 正式结果为 learned REC
`0.5798275/0.4648717`、Mask `0.5972865/0.4908498/0.4181844`，candidate oracle
`0.6296803/0.5500631/0.4517465`。相对 epoch 72，REC@0.25 下降 `0.0008414`、
REC@0.50 上升 `0.0005259`；Mask 三项均刷新 V7 best。retention 因此保留 epoch 72 的
REC@0.25 权重和 epoch 73 的其余四项权重，实际只有两个 checkpoint inode。auditor 仍为
`candidate pass / learned fail / mask guard pass`，继续 epoch 74。

V7 epoch 74 正式结果为 learned REC `0.5801430/0.4639251`、Mask
`0.5964451/0.4902188/0.4174829`，candidate oracle
`0.6314682/0.5522718/0.4532405`。五项均未刷新 epoch 72/73 best，且
`mask_guard_pass=false`；retention 最佳映射保持 epoch 72 REC@0.25、epoch 73 其余四项。

V7 epoch 75（最后一轮 `3e-4` 高学习率阶段）完成全量 9,508-row 验证。learned REC 为
`0.5799327/0.4637148`，Mask 为 `0.5963399/0.4896929/0.4172467`；candidate oracle
仍达到 `0.6339924/0.5546908/0.4550207`。该轮没有刷新任何 retained best，
checkpoint retention 继续保留 epoch 72 的 REC@0.25 `0.5806689` 和 epoch 73 的
REC@0.50 `0.4648717` 及三项 Mask 最佳。

epoch 75 gate 共预测切换 `120/9508=1.262%`，其中有益 `36`、有害 `84`，precision
为 `30.00%`，oracle switch recall 为 `3.16%`。候选空间连续四轮显著超过
`0.59/0.49`，但高学习率训练到 epoch 75 仍没有形成稳定净增益，因此“仅训练轮数不足”
已经不是主要解释；epoch 76 起按既定合同将 SourceMoE LR 降为 `3e-5`，继续完成
76-80 的低学习率对照，运行中不修改 gate objective。

V7 epoch 76 是首轮低学习率正式对照，learned REC 为
`0.5801430/0.4635044`，Mask 为 `0.5960244/0.4896929/0.4170740`，五项均没有刷新
retained best。candidate oracle 仍为 `0.6255785/0.5466975/0.4492696`，继续超过目标；
gate 预测切换 `102/9508=1.073%`，其中有益 `31`、有害 `71`，precision `30.39%`、
oracle switch recall `2.93%`。降低 LR 没有立即改善 under-switch 或错误切换，继续按合同
完成 epoch 77-80，不用单轮波动提前终止。

V7 epoch 77 正式结果为 learned REC `0.5799327/0.4635044`、Mask
`0.5961296/0.4899032/0.4171509`，candidate oracle
`0.6240008/0.5456458/0.4482024`；五项 retained best 仍不变。gate 预测切换
`116/9508=1.220%`，其中有益 `33`、有害 `83`，precision `28.45%`、oracle switch
recall `3.18%`。低学习率第二轮同样没有提升净部署效果，继续完整运行 epoch 78-80。

V7 epoch 78 正式结果为 learned REC `0.5795120/0.4631889`、Mask
`0.5956037/0.4893774/0.4167333`，candidate oracle
`0.6242112/0.5451199/0.4481257`。gate 预测切换 `145/9508=1.525%`，但有益仅
`37`、有害 `108`，precision `25.52%`、oracle switch recall `3.54%`；错误切换随
部署比例上升，五项 retained best 均保持 epoch 72/73。继续完成最后两轮，以保留完整
训练时长和最终 checkpoint 审计证据。

V7 epoch 79 正式结果回升至 learned REC `0.5803534/0.4640303`、Mask
`0.5966554/0.4903239/0.4174338`，但仍未刷新 epoch 72/73 retained best。candidate
oracle 为 `0.6227387/0.5451199/0.4473105`；gate 预测切换 `106` 次，有益 `32`、有害
`74`，precision `30.19%`、oracle switch recall `3.09%`。继续最后一轮 epoch 80，完成后
冻结训练结论并在 GPU 空闲时评测 retained REC checkpoint 的 `expected_utility` action。

V7 已完整完成 epoch 72-80。最终 epoch 80 learned REC 为
`0.5803534/0.4642406`，Mask 为 `0.5965503/0.4903239/0.4175141`，candidate oracle
为 `0.6224232/0.5442785/0.4470129`。gate 预测切换 `101` 次，有益 `30`、有害 `71`，
precision `29.70%`、oracle switch recall `2.93%`。九轮正式训练没有达到 `0.59/0.49`，
且低学习率阶段没有改变 under-switch/false-switch 瓶颈，因此可排除“只是没加载预训练或
轮数不够”的解释。

最终 retention 仍选择 epoch 72 的 REC@0.25 `0.5806689`，以及 epoch 73 的
REC@0.50 `0.4648717` 和三项 Mask `0.5972865/0.4908498/0.4181844`；对应仅两个
checkpoint inode。训练退出后 GPU 无 compute process。历史 `0.582878/0.486012` 三件套
与 router anchor 再次核对为 `0444`、link count `2`，未被 V7 retention 修改。下一步在
epoch 72 retained checkpoint 上以其 legacy 收据为 baseline 做 `expected_utility` 网络内
action 消融；完成消融后再删除 V7 非 retained 的 epoch 80 latest 权重。

### V7 retained expected-utility 消融与 V8 修正训练（2026-08-01）

epoch 72 REC@0.25 retained checkpoint 的 `expected_utility` 全量评测为 REC
`0.5805637/0.4643458`、Mask `0.5968658/0.4905343/0.4178205`。相对同 checkpoint
legacy 为 REC `-0.0001052/0`；gate 只切换 `18` 次，有益 `11`、有害 `7`，precision
`61.11%` 但 oracle switch recall 仅 `0.95%`。

epoch 73 REC@0.50/Mask retained checkpoint 的同合同评测为 REC
`0.5807741/0.4645562`、Mask `0.5970761/0.4906395/0.4179896`。相对该 checkpoint
legacy 为 REC `+0.0009466/-0.0003155`；gate 切换 `22` 次，有益 `13`、有害 `9`，
precision `59.09%`、recall `1.18%`。这证明 quality heads 的方向比 legacy decision
更精确，但 expected utility 平均 margin 约 `-0.376`，仍严重 under-switch。

代码审计发现 `expected_utility` 部署使用 box/mask heads 的 action margin，但 calibrated
utility regression 仍固定校准 legacy `decision_head` margin。已将回归和 overestimate
统计改为实际部署的 `selection_margin`；decision 模式下它与旧 margin 完全相同，故 legacy
路径不变。新增梯度测试证明 expected-utility 模式下 regression 梯度到达 box/mask heads，
SourceMoE focused tests 为 `92 passed`。

V8 从 epoch 73 retained checkpoint 启动两轮 gate-only 训练，主干、router 和 query
reranker 均保持 eval。配置为 `action_mode=expected_utility`、修正后的 action calibration、
`balanced_calibrated_utility`、setwise temperature `0.25`、gate LR `3e-4`。同时将
`false_override_weight` 从 `2.0` 改为 `1.0`：break cost `2.0` 已在 utility target 中编码
错误切换代价，旧配置在 class/row/regression 三处重复加罚，是 under-switch 的结构性来源。
V8 run 为 `output/source_moe_expected_utility_train_v8/scanrefer/ssq_moe_e73_expected_utility_calibrated_fow1_e2/1785590518/`。

V8 epoch 1 正式结果显示，去掉重复 false-override 惩罚后 action margin 从过度保守变为
过度切换：REC `0.5801430/0.4642406`，Mask `0.5907657/0.4844342/0.4137051`；gate
切换 `108` 次，有益 `28`、有害 `80`，precision `25.93%`、oracle switch recall
`2.54%`。候选 oracle 仍为 `0.6296803/0.5500631`，故该结果不是候选空间退化，而是
`false_override=1.0` 的校准强度不足；epoch 2 作为完整训练时长对照继续运行，随后改测
中间风险权重和直接 setwise utility。

V8 epoch 2 为 REC `0.5800379/0.4638199`、Mask
`0.5909760/0.4847497/0.4137946`；gate 切换 `118` 次，有益 `30`、有害 `88`，
precision `25.42%`、oracle switch recall `2.72%`。两轮均明显弱于 V7 retained
checkpoint，因此 V8 所有 checkpoint 链接已按两个 inode 永久删除，config、log、两轮
metrics/diagnostics receipts 保留。同步删除了 V7 非最佳 epoch 80/latest inode；V7 仅保留
epoch 72/73 两个 retained inode。清理后磁盘可用空间由约 `7.0 GiB` 增至 `8.7 GiB`，
历史 `0.582878` 三件套和 router anchor 仍为 `0444`、link count `2`。

新增 V9 `direct_utility` action mode：fallback gate 增加一个零初始化的 129 参数 scalar
head，直接预测每个候选相对 shared fallback 的 threshold-aware utility。该 margin 同时接受
setwise fallback/candidate loss 和 calibrated utility regression，避免从 class-balanced
break/neutral/fix 概率反推期望值。旧 checkpoint 只允许缺失新增 head 的 weight/bias；若
checkpoint 自称 direct-utility 却缺失它，或缺失任何其他 gate tensor，仍会拒绝加载。

CLI、launcher、checkpoint config、cache/reranker 合同和测试均已扩展到
`direct_utility`。focused tests 为 `95 passed`，全套为 `2885 passed, 3 warnings`。V9 从
V7 epoch 73 retained checkpoint 启动两轮 gate-only 训练，运行时确认只新增 129 个可训练
参数，总计 `316,561`；`false_override_weight=2.0`、break cost `2.0`、setwise
temperature `0.25`，其余主干/router/query reranker 保持冻结。

V9 已按完整两轮 gate-only 合同结束。epoch 1 正式结果为 REC
`0.5795120/0.4638199`、Mask `0.5963399/0.4899032/0.4173530`；epoch 2 为 REC
`0.5799327/0.4640303`、Mask `0.5967606/0.4901136/0.4174779`。二者都低于 V7
retained best（epoch 72 REC@0.25 `0.5806689`、epoch 73 REC@0.50 `0.4648717`、Mask
`0.5972865/0.4908498/0.4181844`），因此 V9 不能作为保留权重。

V9 epoch 2 的 source-choice 结构化诊断显示，candidate oracle 仍有
`0.6296803/0.5500631` 与 mIoU `0.4517495`，但 deployed gate 只切换
`143/9508=1.5040%`，其中 `33` 次有益、`110` 次有害，precision `23.08%`、oracle
switch recall `2.99%`。问题不是候选空间，而是候选排序 margin 同时承担“是否切换”与
“切到哪个 query”，candidate max 会放大 noisy positive utility；下一步改为两级 gate。

轮询策略已按用户要求改为基于实测耗时估算。V9 epoch 1 从训练日志看，训练约 `52` 分钟、
训练结束到 `eval_metrics_epoch_1.json` 写入约 `14` 分钟；epoch 2 因此只在预计收据窗口
`02:09-02:11 CST` 后检查，正式收据实际于 `02:09:27 CST` 写入。低质量 V9 的 7 个
checkpoint 名称全部指向同一 inode `2170237654`，已删除；`log.txt`、`config.json`、
两轮 `eval_metrics` 与 `source_choice_diagnostics` 收据保留。历史 `0.582878/0.486012`
三件套、router anchor、V7 retained epoch 72/73 权重均未删除。

新增 V10 `hierarchical_utility` action mode：保留 direct utility candidate head 只做 top-8
候选排序；新增 `row_switch_head` 对 fallback hidden、候选集合均值和差值做行级判断，只有
row margin 大于 `decision_margin` 才允许离开 default。loss 同步拆分为候选选择 loss、
row-level switch BCE 和 row utility regression，避免 candidate utility max 直接触发部署切换。
所有新增最终层零初始化，旧 checkpoint 迁移初始仍精确 fallback；若 checkpoint 自称
`hierarchical_utility` 却缺少 `utility_head` 或 `row_switch_head`，合同会拒绝加载。

V10 实现范围包括 `models/source_moe.py`、`models/losses.py`、`main_utils.py`、
`scripts/train_scanrefer_source_moe.sh` 和 `scripts/train_rec_reranker.py`。测试新增两级 gate
零初始化、row veto、候选/row 梯度分离、CLI 解析和 checkpoint 迁移/拒绝回归。聚焦测试为
`99 passed`，sidecar/cache 合同测试为 `74 passed`，全套为 `2889 passed, 3 warnings`。
下一步从 V7 epoch 73 retained checkpoint 启动两轮 V10 gate-only 训练，仍冻结 MCLN 主干、
router 和 query reranker，只更新 fallback gate。

V10 已于 `2026-08-02 02:38:17 CST` 启动，正式目录为
`output/source_moe_hierarchical_utility_train_v10/scanrefer/ssq_moe_e73_hierarchical_utility_fow2_e2/1785609497/`，
tmux 为 `mcln_v10_hierarchical_utility_gate_e1_e2`。启动日志确认加载 V7 epoch 73，
train/val 为 `36,665/9,508`，gate-only allowlist 为 `366,226` 个参数，精确等于原 gate
`316,432` + candidate utility head `129` + row switch head `49,665`。首批运行正常；按
约 `1.07 it/s` 估算训练于 `03:35 CST` 结束，加上一轮实测约 `14` 分钟验证，下一次正式
检查窗口为 `03:49-03:51 CST`。

V10 两轮完整结果确认 `balanced_calibrated_utility` 不适合 row-level switch。epoch 1 REC 为
`0.5522718/0.4276399`、Mask 为 `0.5931847/0.4861170/0.4141952`；epoch 2 REC 为
`0.5522718/0.4260618`、Mask 为 `0.5922381/0.4858014/0.4138547`。两轮都远低于 fixed
default 和 V7 retained，因此不保留任何 V10 checkpoint。

V10 epoch 1/2 的 candidate oracle 都稳定在 `0.6296803/0.5499580`、mIoU
`0.4516721`，说明候选与 query ranking 空间仍然充足。实际 row gate 却分别切换
`2453/9508=25.80%` 与 `2923/9508=30.74%`；有益/有害为 `275/2178`、`320/2603`，precision
仅 `11.21%/10.95%`。这不是双阶段主干或 mask 崩塌，而是 balanced row BCE 的 inverse-
frequency 正权重把 oracle switch `11.61%` 的真实先验夸大，导致大规模 false override。

V10 的 6 个 epoch-1/best 名称指向 inode `2163933766`，2 个 epoch-2/latest 名称指向
inode `2164013807`；两个 inode 都仅由 V10 引用，8 个 checkpoint 已删除。保留
`config.json`、`log.txt`、两轮 `eval_metrics`、两轮 `source_choice_diagnostics` 和
retention receipt。清理后磁盘可用空间为约 `8.7 GiB`；历史 `0.582878/0.486012` 只读
三件套、router anchor 和 V7 epoch 72/73 retained checkpoint 均再次确认未动。

下一版不改变 V10 的 architecture，而新增独立训练合同
`hierarchical_risk_calibrated`：box/mask/auxiliary candidate decision 保留 inverse-frequency
balanced supervision，候选 utility 仍只学习 switch 行内 query；只有 row switch BCE 改为
固定成本（fallback false override 乘 `false_override_weight`）且保持 row best-utility
regression。`break_cost` 继续只定义 utility target。该
分离是跨数据集的 risk calibration，而不是 ScanRefer 阈值后处理；新合同名避免把 V10 的
失败配置与新行为混为同一可复现记录。

`hierarchical_risk_calibrated` 已实现并测试：candidate quality 和 auxiliary decision 仍沿用
batch-balanced focal，row switch BCE 独立固定为 `[false_override_weight, 1]`，不再受
oracle-switch inverse frequency 影响；row best utility regression 仍按 overestimate 加权。
若该目标未收到 hierarchical row head，会 fail closed。新增 prior/gradient 回归覆盖其正
switch 梯度小于 V10 balanced objective、fallback 负类风险梯度更大。聚焦测试 `101 passed`，
全套为 `2891 passed, 3 warnings`。

V11 将从同一 V7 epoch 73 retained checkpoint 启动，不加载任何 V10 权重：
`action_mode=hierarchical_utility`、`objective=hierarchical_risk_calibrated`、top-8、
context 1 层/4 heads、evidence features、setwise temperature `0.25`、
`false_override_weight=2.0`、gate LR `3e-4`。因此该消融只改变 row prior calibration，
可直接和 V10 对比，且不会变成 ScanRefer 专用后处理。

V11 启动审计发现并修复两处 fail-closed 合同漏项。首次目录 `1785618301` 虽由 shell 显式
请求 risk objective，但 checkpoint config prepare 无条件恢复成 V10 的
`balanced_calibrated_utility`，在 batch 23 前人工停止且无 checkpoint。修复为 CLI 未指定时
继承、显式指定时保留 fresh gate-only objective，并新增
`source_moe_gate_objective_explicit` 收据字段。随后目录 `1785620042` 在首 batch 由
`models/losses.py` 上层白名单拒绝新字符串；保留 stderr 后取得完整 traceback，再同步该
白名单。两个失败目录都只有 config/log，无权重。

新增回归分别覆盖 explicit objective override 与 `compute_hungarian_loss` 接受新合同；最终
focused/full tests 为 `103 passed`、`2893 passed, 3 warnings`。正式 V11 目录为
`output/source_moe_hierarchical_risk_train_v11/scanrefer/ssq_moe_e73_hierarchical_risk_calibrated_e2/1785620910/`，
tmux `mcln_v11_hierarchical_risk_gate_e1_e2_final`。启动日志确认 objective 正确、加载 V7
epoch 73、gate-only 参数 `366,226`，首 batch 正常。按速度估算首轮收据窗口为
`06:58-07:00 CST`。

V11 两轮已完整结束。epoch 1 learned REC 为 `0.5648927/0.4481489`，Mask 为
`0.5961296/0.4884308/0.4163539`；row gate 切换 `1021/9508=10.738%`，其中有益
`139`、有害 `882`，precision `13.61%`、oracle-switch recall `12.61%`。epoch 2
进一步退化为 REC `0.5623685/0.4469920`、Mask `0.5949727/0.4881153/0.4158356`；
切换增至 `1380/9508=14.514%`，其中有益 `177`、有害 `1203`，precision `12.83%`、
recall `16.06%`。两轮 candidate oracle 均保持 `0.6296803/0.5500631/0.4517543`，
因此候选空间没有退化，失败点仍是 row switch 对有益切换的可分性不足：固定风险 BCE
虽然消除了 V10 的 `25.8%-30.7%` 极端过切换，但错误切换仍显著多于收益。

轮询采用首轮实测耗时：训练 `2968.35 s`（约 `49:28`），验证约 `13:23`；epoch 2 于
`07:00:03 CST` 开始，预计收据为 `08:03-08:04 CST`，只在该窗口检查一次，正式收据于
`08:03:25 CST` 落盘。V11 retention 五项均停留在 epoch 1，但仍全面低于 V7 retained，
所以 inode `13706257` 的 6 个 epoch-1/best 链接和 inode `44743462` 的 2 个
epoch-2/latest 链接全部删除；config、log、两轮 metrics/diagnostics 与 retention receipt
保留。清理前再次确认历史 `0.582878/0.486012` 三件套及 router anchor 为 `0444`、
link count `2`，V7 epoch 72/73 retained 权重也不属于上述 inode。

### V12 candidate-conditioned pairwise verifier（2026-08-02）

V11 暴露出两个训练/部署错位。第一，`row_switch_head` 输入是 top-8 candidate hidden 的均值，
但部署动作实际切换到 `utility_head.argmax` 的单个 candidate；均值无法表示该 candidate 的
反事实质量。第二，旧 row target/regression 使用整行 oracle best utility：即使排序头当前选中
有害 candidate，只要同一行另有正 candidate，row head 仍收到 switch-positive 标签。这会
直接训练出“允许错误 candidate 覆盖 fallback”的行为，与 V11 的低 precision 一致。

V12 新增 `pairwise_verifier` action mode 和 `pairwise_risk_calibrated` objective。candidate
utility head 先选出实际 proposed query；新的 pairwise switch head 对
`[fallback, proposed, proposed-fallback, proposed*fallback, candidate_margin]` 做二分类与
utility regression，训练和部署共用同一 row margin。最终层零初始化，迁入 V7 时精确 fallback。
candidate ranking 改为在所有有候选的行学习相对 utility 分布，包括没有正 candidate 的行；
因此它也会学习选择“最小伤害”候选，而不是在 fallback 行保持随机。row target 与 regression
均绑定当前 proposed candidate 的真实 threshold-aware utility，候选排序仍由全候选 oracle
utility 监督。

迁移合同只允许旧 action mode 缺失新 `pairwise_switch_head`；声明为 `pairwise_verifier` 的
checkpoint 若缺少该 head 或 `utility_head` 会 fail closed。结构化 diagnostics 新增可选的
`row_target_switch_count/rate`，用于区分 oracle candidate coverage 与当前 proposed candidate
可切换率，同时保持旧收据兼容。SourceMoE/配置/缓存聚焦测试 `163 passed`，全套
`2901 passed, 3 warnings`。

V12 将从 V7 epoch 73 retained checkpoint 启动两轮 gate-only 训练，冻结主干、router 与
query reranker，配置为 top-8、context 1 层/4 heads、evidence features、setwise temperature
`0.25`、false override weight `2.0`、gate LR `3e-4`。预期 gate-only 参数为旧 V11
`366,226` 加 pairwise verifier `66,177`，合计 `432,403`；启动日志必须精确匹配后才纳入实验。

V12 正式 run 已于 `08:24:56 CST` 创建，目录为
`output/source_moe_pairwise_verifier_train_v12/scanrefer/ssq_moe_e73_pairwise_verifier_e2/1785630296/`，
tmux 为 `mcln_v12_pairwise_verifier_e1_e2`。启动配置确认 objective/action 均为新合同，
显式加载 V7 epoch 73，可训练参数精确为 `432,403`；首批速度约 `1.10 it/s`。据此估算
epoch 1 训练于 `09:20-09:21 CST` 结束，验证收据窗口为 `09:34-09:35 CST`，只在该窗口
读取正式 JSON。

V12 epoch 1 实际训练 `49:05`，全量收据于 `09:36:21 CST` 落盘。learned REC 为
`0.5805637/0.4637148`，相对 fixed default 增加 `6/2` 个命中，但仍未刷新 V7 REC retained；
Mask 为 `0.5973917/0.4903239/0.4177016`，其中 Mask@0.25 的 `5680` hits 比 V7 retained
多 1 个，暂时保留该 inode。candidate oracle 为 `0.6296803/0.5497476/0.4516247`。

精确 gate diagnostics：oracle switch `1103`，当前 proposer 所选 candidate 为正的 row target
仅 `408`；verifier 实际切换 `109` 次，其中有益 `31`、有害 `78`，precision `28.44%`、
oracle recall `2.81%`。相比 V11，pairwise 对齐将大规模 false override 压回可控范围并恢复
baseline，但 proposer 只把正 candidate 放到 top-1 的约 `34.0%` oracle 行，verifier 又只部署
少量 proposal，当前瓶颈转为 candidate ranking recall 与 under-switch。epoch 2 于
`09:36:21 CST` 开始，按实测 `49:05` 训练 + `13:06` 验证，仅在 `10:38-10:39 CST`
检查最终收据。

V12 epoch 2 正式结果继续提升：learned REC `0.5810896/0.4648717`，Mask
`0.5979175/0.4911653/0.4183786`。REC@0.25 比 V7 network retained 增加 4 hits，成为新的
network-only 最佳；REC@0.50 与 V7 精确同为 `4420/9508`。三项 Mask 分别为
`5685/4670` hits 和更高 mIoU，均刷新 V7。candidate oracle 保持
`0.6296803/0.5497476/0.4516247`。gate 切换降至 `37` 次，有益 `18`、有害 `19`，precision
升至 `48.65%`；proposer-positive row 从 `408` 增到 `422`，oracle-best-tier match 从
`375` 增到 `391`。这说明第二轮仍在改善，不应立即更换结构。

retention 已将 epoch 1 私有 inode 自动删除，当前 7 个 best/epoch2/latest 名称仅指向
inode `4306091917`；该权重因刷新 REC@0.25 与三项 Mask 必须保留。历史
`0.582878/0.486012` 只读三件套、router anchor 和 V7 epoch 72/73 retained 再次核对未动。

为避免继续训练时重置 Adam，新增显式 `source_moe_gate_resume_optimizer` 合同：只允许非 eval
的 gate-only 模式，配置必须与 checkpoint 完全一致，requested start 必须等于
checkpoint epoch + 1，且禁止 `reduce_lr/checkpoint_start_epoch`。通过后原样恢复 optimizer
moment 与 scheduler step；其他 train-only 行为保持 fresh optimizer。恢复路径与全套测试为
`56 passed`、`2906 passed, 3 warnings`。下一 run 从 epoch 2 权重精确恢复 step `6110`，
完整训练 epoch 3-4。

V12 精确续训正式 run 为
`output/source_moe_pairwise_verifier_continue_v12/scanrefer/ssq_moe_e73_pairwise_verifier_e3_e4_exact_resume/1785639639/`，
tmux `mcln_v12_pairwise_verifier_e3_e4_resume`。启动日志明确报告加载 epoch 2、
`resumed exact gate-only optimizer and scheduler state`，可训练参数仍为 `432,403`，首批约
`1.09 it/s`。epoch 3 收据窗口估算为 `12:12-12:13 CST`。

V12 精确 optimizer 续训已完整跑完 epoch 3-4，并按用户要求只在估算完成窗口检查正式收据。
epoch 3 实际收据于 `12:12:28 CST` 落盘，检查时间为 `12:12:30 CST`；epoch 4 根据
epoch 3 的真实训练+验证周期重新估算，收据于 `13:15:09 CST` 落盘，检查时间为
`13:15:20 CST`。中间未进行几分钟一次的训练日志轮询。

epoch 3 learned REC 为 `0.5808793/0.4646613`（`5523/4418` hits），Mask 为
`0.5977072/0.4911653/0.4184125`（`5683/4670` hits）。candidate oracle 保持
`0.6296803/0.5496424/0.4516072`。gate diagnostics：oracle switch `1103`，
row target switch `413`，predicted switch `55`，其中 beneficial/harmful 为 `21/34`，
precision `38.18%`，oracle switch recall `1.90%`，oracle-query match `377/1103`。
该轮 REC 比 V12 epoch 2 分别少 `2/2` hits，Mask@0.25 少 2 hits、Mask@0.50 持平，
但 Mask mIoU 从 `0.4183786` 小幅升至 `0.4184125`，因此作为当前 V12 mask-mIoU
retained 权重保留。

epoch 4 learned REC 为 `0.5807741/0.4645562`（`5522/4417` hits），Mask 为
`0.5973917/0.4909550/0.4180954`（`5680/4668` hits），五项均低于 epoch 3。
gate diagnostics：row target switch `427`，predicted switch `30`，beneficial/harmful
`16/14`，precision `53.33%`，oracle switch recall `1.45%`，oracle-query match
`389/1103`。precision 虽继续提高，但部署 switch 进一步收缩，净 REC 与 Mask 全部回落。

checkpoint retention 保持 epoch 3 为该续训 run 的五项 best。低质量 epoch 4 的
`ckpt_epoch_4.pth` 与 `ckpt_epoch_last.pth` 指向同一私有 inode `2170239165`，已删除；
保留 `config.json`、`log.txt`、epoch 3/4 的 `eval_metrics` 与
`source_choice_diagnostics`、`checkpoint_retention.json`。剩余 6 个 best/epoch3 链接均指向
inode `2170239162`。历史后处理最佳 `0.582878/0.486012` 三件套、router anchor 与 V7
retained epoch 72/73 权重再次核验未删除。

当前最高指标口径更新：系统级历史后处理最佳仍为 REC `0.582878/0.486012`；network-only
REC 最高仍是 V12 epoch 2 的 `0.5810896/0.4648717`，Mask mIoU 最高为本次 epoch 3 的
`0.4184125`。继续同配置 optimizer resume 已出现 plateau/回落，不建议盲目延长；下一步应
集中提升 proposer ranking recall，例如让 verifier 对 top-n proposed candidates 逐一评分并共同参与
selection，或在 listwise utility 里加入 hard-negative/positive recall 约束，而不是继续调全局
switch threshold。

V13 两轮正式结果未刷新 V12。epoch 1 REC 为 `0.5806689/0.4642406`，Mask 为
`0.5974968/0.4906395/0.4180548`；candidate oracle 为
`0.6297854/0.5498528/0.4516363`，row target switch `1108`，predicted switch `86`，
beneficial/harmful `28/58`，precision `32.56%`，recall `2.53%`。epoch 2 进一步降到
REC `0.5800379/0.4638199`，Mask `0.5969710/0.4903239/0.4176694`；predicted switch
`90`，beneficial/harmful `26/64`，precision `28.89%`。虽然 oracle-query match 从
V12 epoch 2 的约 `35.45%` 提高到 `37.91%/39.98%`，净指标没有转化。

V13 失败点不是候选覆盖，而是训练目标校准：row-wise soft target 仍给 fallback 一部分概率，
正候选 margin 难以稳定推过 0；同时逐候选 regression 把 neutral utility 拟合到 0，容易在
部署 `margin > 0` 时产生 false switch。V13 的 epoch1/best inode `6734434168` 与
epoch2/latest inode `6734434169` 已全部删除，保留 config/log/JSON receipts。历史
`0.582878/0.486012` 权重、V12 epoch 2 network-only best 和 V12 epoch 3 mask-mIoU
retained 权重均未删除。

### V13 top-n pairwise verifier（2026-08-02）

V12 的 candidate oracle 稳定在约 `0.6297/0.5496`，但单 proposer top-1 只有约
`35%` oracle-best-tier match；继续训练时 verifier precision 上升而 switch recall 下降。
V13 因此不再只验证 proposer 的一个 query，而对 top-8 每个候选复用同一个
`pairwise_switch_head`，为每个 fallback/candidate 对生成独立 verifier margin。部署直接选择
最大正 verifier margin 的候选，否则精确回退 shared default。该路径不增加参数，gate-only
allowlist 仍为 `432,403`。

训练目标同时保留两层职责：`utility_head` 继续接受全候选 listwise utility 排序监督；
verifier 使用 `[fallback logit=0, top-n verifier margins]` 的 row-wise setwise loss，正行把
真实正收益候选推过 fallback，负行监督 exact fallback；另对每个候选 margin 回归其相对
fallback 的 threshold-aware box+mask utility，并对过估计施加 false-override 风险权重。
训练和部署使用同一 verifier margin，没有 validation 阈值搜索或 ScanRefer 专用后处理。

实现涉及 `models/source_moe.py`、`main_utils.py`、
`scripts/train_scanrefer_source_moe.sh`、`scripts/train_rec_reranker.py` 与相关测试。
聚焦测试为 `164 passed, 2 warnings`，全套为 `2910 passed, 3 warnings`。真实 A100
debug smoke 从 V12 epoch 2 权重迁移，完成 32 个 train + 32 个 eval batch；checkpoint
完整加载、allowlist 为 `432,403`，128-row panel 上只切换 1 次且为有益切换。smoke 的
7 个 checkpoint 名称均指向调试 inode `2206241830`，已全部删除，config/log/JSON
receipts 保留。

正式 V13 run 为
`output/source_moe_topn_pairwise_verifier_train_v13/scanrefer/ssq_moe_e73_topn_pairwise_verifier_e2/1785650097/`，
tmux `mcln_v13_topn_pairwise_e1_e2`。从 V12 epoch 2 network-only 最佳权重启动 fresh
gate optimizer，top-8/context-1/evidence/setwise temperature `0.25`/false override
`2.0`/gate LR `3e-4` 保持不变。首批实测 `1.06 it/s`，epoch 1 正式收据预计
`15:05:20-15:05:50 CST`，只在 `15:06 CST` 检查。

### V14 risk-separated top-n objective（2026-08-02）

V13 的覆盖提升没有转化为净指标，定位到训练目标和部署 0-margin 边界不一致。V14 不增加
参数、不改 top-n pairwise 部署结构，只新增 `topn_risk_calibrated` objective：positive row
把 fallback target 置零，negative/neutral row 使用 exact fallback；逐候选 regression 中
positive/negative utility 保持原值，neutral 从 0 下移至 `-setwise_temperature`。这样同时增强
positive candidate 越过 0 的动力，并减少 neutral false switch。

实现涉及 `models/source_moe.py`、`main_utils.py`、`models/losses.py` 及 SourceMoE 测试；另修复
`row_switch_margin` 校验缩进。聚焦测试为 `124 passed`、checkpoint/诊断测试为
`30 passed, 2 warnings`，全套为 `2917 passed, 3 warnings`。

真实 A100 smoke run 为
`output/source_moe_v14_smoke/scanrefer/ssq_moe_v14_topn_risk_debug_e0/1785663192/`，从 V12
epoch 2 最佳加载，allowlist 精确为 `432,403`，完成 32 train + 32 eval batches。128-row
diagnostics 中 switch `1` 次且为 beneficial，harmful 为 0。smoke 私有 inode `48606427`
的 7 个 checkpoint 链接已删除，config/log/JSON receipts 保留；V12 epoch 2、V12 epoch 3
和历史 `0.582878/0.486012` artifacts 复核未动。

正式 V14 run 为
`output/source_moe_topn_risk_calibrated_train_v14/scanrefer/ssq_moe_e73_topn_risk_calibrated_e2/1785663614/`，
tmux `mcln_v14_topn_risk_e1_e2`。启动于 `17:40:14 CST`，数据初始化于 `17:48:50`
完成；首批约 `1.08 it/s`，allowlist 为 `432,403`。按该吞吐只在预计收据窗口检查：epoch 1
训练 `49:05`，收据于 `18:51:03` 落盘；epoch 2 训练 `48:50`，收据于 `19:53:12`
落盘。中间未按几分钟频率轮询。

epoch 1 learned REC 为 `0.5803534/0.4642406`（`5518/4414` hits），Mask 为
`0.5971813/0.4905343/0.4178422`（`5678/4664` hits）。candidate oracle 为
`0.6297854/0.5500631/0.4517465`；switch `80` 次，beneficial/harmful `26/54`，precision
`32.50%`，oracle-query match `433/1102=39.29%`。

epoch 2 learned REC 进一步降到 `0.5798275/0.4638199`（`5513/4410` hits），Mask 为
`0.5967606/0.4902188/0.4175319`（`5674/4661` hits）。candidate oracle 不变；switch
`85` 次，beneficial/harmful `23/62`，precision `27.06%`，oracle-query match 升到
`450/1102=40.83%`。覆盖继续改善而 precision 恶化，证明 V14 的 hard fallback target 与
neutral safety gap 不足以区分 harmful hard negatives，不能刷新 V12。

V14 epoch 1 五项 best inode `10814120961` 的 6 个链接和低质量 epoch 2/latest inode
`10814120962` 的 2 个链接已全部删除；config、log、两轮 JSON receipts 和 retention 记录
保留。V12 epoch 2 inode `4306091917`、V12 epoch 3 inode `2170239162` 和历史只读
`0.582878/0.486012` artifacts 再次核验未动。当前最高仍为历史后处理 REC
`0.582878/0.486012`；network-only REC 仍为 V12 epoch 2 的
`0.5810896/0.4648717`，Mask mIoU 仍为 V12 epoch 3 的 `0.4184125`。

### V15 dual evidence verifier（2026-08-02）

V14 证明 coverage 不是主要瓶颈，hard-negative precision 才是。V15 保留 V12/V13 的
pairwise benefit head，新增显式 safety head，输入 candidate hidden 以及 box/mask/decision
的 threshold-transition 概率、expected/direct utility。部署使用两者 margin 的最小值，只有
benefit 与 safety 同时为正才覆盖 fallback。训练新增 candidate-wise class-balanced focal
safety loss，对 neutral/break 全部施加 veto 监督；该规则不含 ScanRefer validation 阈值搜索。

实现新增 action `topn_dual_evidence_verifier`、objective
`topn_dual_risk_calibrated`、CLI/checkpoint/fail-closed 合同与双 head 梯度测试。聚焦测试为
`130 passed`、`54 passed, 2 warnings`，全套 `2923 passed, 3 warnings`。新 head 为
`19,073` 参数，gate-only allowlist 精确为 `451,476`。

A100 smoke run 为
`output/source_moe_v15_smoke/scanrefer/ssq_moe_v15_dual_evidence_debug_e0/1785672783/`，从 V12
epoch 2 最佳加载，完成 32 train + 32 eval batches。零初始化 safety veto 在 128-row panel
保持 0 switch，符合 exact-fallback 起点。smoke 私有 inode `8606111000` 的 7 个 checkpoint
链接已删除，config/log/JSON receipts 保留；保护权重未动。

V15 正式 run 为
`output/source_moe_dual_evidence_verifier_train_v15/scanrefer/ssq_moe_e73_dual_evidence_verifier_e2/1785673198/`，
tmux `mcln_v15_dual_evidence_e1_e2`。它从 V12 epoch 2 network-only 最佳权重启动 fresh gate
optimizer，top-8、context 1 层/4 heads、LR `3e-4`、temperature `0.25`、false override
weight `2.0` 保持不变。run 于 `20:19:58 CST` 启动，数据初始化于 `20:28:35 CST` 完成。

epoch 1 训练耗时 `3294.53s`，正式收据于 `21:39:11 CST` 落盘；按预估窗口只在
`21:48 CST` 检查。learned REC 为 `0.5805637/0.4642406`（`5520/4414` hits），Mask 为
`0.5976020/0.4905343/0.4178641`（`5682/4664` hits）。candidate oracle 为
`0.6297854/0.5500631/0.4517248`。部署 switch `98` 次，其中 beneficial/harmful 为
`30/68`，precision `30.61%`、oracle switch recall `2.71%`；oracle-query match 为
`509/1105=46.06%`。

根据 epoch 1 的真实训练和验证周期，将 epoch 2 检查窗口估算为 `22:50-22:52 CST`，期间未
读取训练进度。epoch 2 训练耗时 `3365.47s`，收据于 `22:51:21 CST` 落盘并在 `22:52 CST`
检查。learned REC 降到 `0.5803534/0.4642406`（`5518/4414` hits），Mask 降到
`0.5974968/0.4901136/0.4176414`（`5681/4660` hits）；candidate oracle 不变。switch 增至
`140` 次，beneficial/harmful 为 `35/105`，precision 降到 `25.00%`，oracle-query match
降到 `463/1105=41.90%`。validation 中 safety false-positive ratio 也从 `15.80%` 升至
`19.43%`，说明新增 safety head 没有形成稳定 hard-negative veto；第二轮反而与 benefit head
共同放行了更多有害候选。

两轮五项均低于 V12 retained。V15 epoch 1 的 6 个 checkpoint 名称全部指向私有 inode
`173359264`，epoch 2/latest 的 2 个名称全部指向私有 inode `173594587`；link count 与目录内
名称数严格相等。8 个低质量链接已全部删除，约释放 `1.13 GiB`，仅保留 config、log、两轮
metrics/diagnostics 和 retention receipt。删除后复核 V12 epoch 2 inode `4306091917`（7 links）、
V12 epoch 3 inode `2170239162`（6 links）以及历史 `0.582878/0.486012` 只读三件套均未改动。
当前最高仍为历史后处理 REC `0.582878/0.486012`；network-only REC 仍为 V12 epoch 2 的
`0.5810896/0.4648717`，Mask mIoU 仍为 V12 epoch 3 的 `0.4184125`。

### V16 absolute-quality delta gate（2026-08-03）

V15 的 safety 正类很稀疏，而 inverse-frequency focal 会把 `logit > 0` 的部署边界推到远低于
真实 50% posterior 的位置；这与正式验证中 safety false positive 随训练增加一致。V16 改为
`topn_absolute_quality_delta` action 和 `topn_absolute_quality_calibrated` objective：对每个
候选直接预测 box/mask 的 `@0.25`、`@0.50` 与连续 IoU 共 6 个绝对质量值，再用候选质量减去
shared/default query 质量作为部署 margin。阈值项使用普通 BCE 保留经验先验，连续 IoU 使用
Smooth L1；最终 `absolute_quality_head` 零初始化，旧 checkpoint 迁移时精确回退 default。

新 head 只有 `774` 个参数。V15 已有的 safety head 仍在结构状态中，但 V16 gate-only
allowlist 的实际可训练参数为 `452,250`；checkpoint 中另有 2 个非训练 threshold-weight
buffer。实现及 checkpoint fail-closed/梯度/目标合同测试已通过：SourceMoE 聚焦测试
`136 passed`，reranker 合同测试 `24 passed`，全套 `2929 passed, 3 warnings`。

首个 128-row smoke run `1785685637` 完成 32 train + 32 eval batches，epoch 0 learned REC
与 fixed default 同为 `63/128`、`56/128`，但部署了 `26` 次 switch，仅 `1` 次 beneficial。
随后 run `1785685989` 精确恢复 Adam/scheduler，继续训练 epoch 1-3。switch rate 依次收缩到
`7.03%`、`5.47%`、`3.91%`；epoch 2/3 learned REC 达到 `64/128`、`58/128`，相对 fixed
default 增加 `1/2` 个命中，Mask 为 `0.500000/0.406250/0.350474`。epoch 3 的 switch 为
`5` 次，其中 beneficial/harmful `2/3`，precision `40%`；candidate oracle 为
`75/128`、`69/128`，仍有明显可学习 headroom。

三份 smoke checkpoint 实体完成张量级审计：模型和 optimizer 中所有浮点张量均 finite；与
V12 epoch 2 共享的 `1198` 个模型张量里，变化只出现在 `fallback_gate`，主干、router 与
query reranker 逐元素完全一致。smoke 的 15 个 checkpoint 名称分别指向 inode
`13999350`（7 links）、`4406554475`（6 links）和 `4406554474`（2 links）；它们仅用于
调试、不刷新 retained 指标，已全部删除，config/log/JSON receipts 保留。V12 epoch 2 inode
`4306091917`、V12 epoch 3 inode `2170239162` 与历史 `0.582878/0.486012` artifacts 未进入
清理范围。

V16 正式 run 于 `00:07:23 CST` 启动，目录为
`output/source_moe_absolute_quality_train_v16/scanrefer/ssq_moe_e73_absolute_quality_e2/1785686848/`，
tmux 为 `mcln_v16_absolute_quality_e1_e2`。它从 V12 epoch 2 network-only best 启动两轮
fresh-optimizer gate-only 训练；全量 train/val 为 `36665/9508`，checkpoint 加载成功且
allowlist 精确为 `452,250`。batch 500 于 `00:26:32 CST` 落日志，实测约
`1.0-1.2 it/s`；据此首轮训练预计 `01:17-01:21 CST` 结束，正式验证收据预计
`01:31-01:37 CST`，只在该窗口检查。

epoch 1 收据于 `01:31:22 CST` 落盘，并在预估窗口内的 `01:35:40 CST` 检查。learned REC 为 `0.5804586/0.4645562`
（`5519/4417` hits），Mask 为 `0.5972865/0.4906395/0.4180273`
（`5679/4665` hits），五项均低于 V12 retained。candidate oracle 为
`0.6301009/0.5500631/0.4517077`；gate 实际 switch `17` 次，其中 beneficial/harmful
`9/8`，precision `52.94%`，oracle-query match `365/1101=33.15%`，oracle switch recall
`0.82%`。V16 已将 harmful override 压低，但当前部署偏保守，仍需等待 epoch 2 判断净收益。

epoch 1 训练实际耗时 `3526.66s`，验证收据于 `01:31:22 CST` 落盘；据此 epoch 2 的正式
检查窗口估算为 `02:45-02:50 CST`，期间不读取中间训练日志。

epoch 2 训练实际耗时 `3428.66s`，正式收据于 `02:45:07 CST` 落盘，并在预估窗口内的
`02:49:59 CST` 检查。learned REC 为 `0.5806689/0.4647665`（`5521/4419` hits），Mask
为 `0.5974968/0.4908498/0.4181498`（`5681/4667` hits）。相对 epoch 1，五项全部改善；
相对 V12 epoch 2 retained，REC 仍少 `4/1` hits，Mask@0.25/@0.50 少 `4/3` hits，mIoU
低 `0.0002288`，因此没有刷新任何全局 retained 指标。

epoch 2 candidate oracle 保持 `0.6301009/0.5500631/0.4517077`。gate switch 为 `18` 次，
其中 beneficial/harmful `11/7`，precision 从 epoch 1 的 `52.94%` 提高到 `61.11%`，
oracle-query match 从 `33.15%` 提高到 `35.06%`；但 oracle switch recall 仅 `1.00%`。
结论是 absolute-quality supervision 明显改善了 false-override precision，但部署过于保守，
尚未把 candidate oracle 的 headroom 转成 retained 提升。

正式 epoch 2 checkpoint 完成张量审计：模型 `1206` 个张量、optimizer `54` 个张量全部
finite；与 V12 共享的 `1198` 个张量中，所有变化都严格位于 `fallback_gate`，非 gate 主干、
router 与 query reranker 逐元素不变。retention 已自动清理 epoch 1 私有实体；epoch 2 的
7 个名称均指向私有 inode `4382073512`。由于五项均未刷新，这 7 个链接已全部删除，保留
config、log、两轮 metrics/diagnostics 与 retention receipt，约释放 `578 MiB`。

清理后再次核验：V12 epoch 2 inode `4306091917` 仍为 7 links，V12 epoch 3 inode
`2170239162` 仍为 6 links，历史 `0.582878/0.486012` 三个后处理组件和 router anchor 均在。
当前系统级最高仍为历史后处理 REC `0.582878/0.486012`；network-only REC 最高仍为 V12
epoch 2 的 `0.5810896/0.4648717`，Mask mIoU 最高仍为 V12 epoch 3 的 `0.4184125`。

### V17 V12-anchor absolute-quality cascade（2026-08-03）

V16 的 precision 已升到 `61.11%`，但只召回 `1.00%` oracle switch；同时它用新的绝对质量
delta 完全替换了 V12 已学到的 pairwise decision。V17 因此改为两阶段 action
`cascade_absolute_quality_correction`：第一阶段原样执行冻结的 V12 `utility_head +
pairwise_switch_head`，得到逐样本动态 anchor；第二阶段只在其余 top-n query 与 shared default
之间学习 correction。shared default 被显式放回候选集，所以第二阶段既能撤销 V12 的有害切换，
也能从动态 anchor 提升到另一 query。correction margin 等于 0 时严格保持 V12 选择。

新 objective 为 `cascade_absolute_quality_calibrated`。它以动态 V12 anchor 重算 box/mask
threshold utility，组合 risk-separated setwise target、neutral safety gap、相对 utility
regression 与 6 维 dense absolute-quality 辅助监督。新增模块只有 `absolute_quality_head`、
`cascade_quality_adapter`、`cascade_correction_head`；hidden dim 128 时共 `84,743` 个可训练参数。
旧 gate 全程保持 eval 且冻结，新 loss 中输入 correction 的 V12 utility/margin 也显式 detach。

checkpoint 合同允许 V12 pairwise 权重缺少上述三个新模块，但要求旧 `utility_head` 和
`pairwise_switch_head` 完整存在；声明 V17 action 的 checkpoint 若缺任一新模块则 fail closed。
CLI 新增 `--source_moe_gate_new_heads_only`，只允许 gate-only + V17 action/objective。实现覆盖
`models/source_moe.py`、`models/losses.py`、`main_utils.py`、训练脚本、reranker config 与
evaluator；全套测试为 `2943 passed, 3 warnings`。

真实 V12 debug 基线 `1785698982` 为 REC `64/57`、Mask `64/52`、mIoU `0.3504906`，
candidate oracle `75/69`。首次 V17 零评估发现新增模块构造消耗 CPU RNG，使
`num_workers=0` debug augmentation 改变；固定 default 自身也变化，证明不是 selector
回归。构造新模块时保存/恢复 RNG 后，run `1785699528` 的 REC `64/57`、Mask threshold
`64/52`、candidate oracle `75/69` 与 V12 逐项一致，correction 为 0；Mask mIoU 仅有
跨进程 CUDA 浮点差 `2.51e-5`。

学习 smoke `1785699777` 完成 32 train + 32 eval batches，allowlist 精确为 `84,743`。
所有 batch loss finite，gate loss 的训练均值为 `1.6027`，评估为 `1.5276`；32 步后
correction 仍为 0，因此没有 harmful correction，也说明该结构的启动明显比 V16 保守。
checkpoint 审计显示与 V12 共享的 `1198` 个 tensor 全部逐元素不变，V17 optimizer 只含
12 个新参数 tensor，模型/optimizer 全部 finite。smoke inode `4299477133` 的 7 个链接已删除，
config/log/JSON receipts 保留；V12 两个 retained inode 与历史 `0.582878/0.486012` artifacts
未改动。

正式 run 于 `03:52:49 CST` 启动，目录为
`output/source_moe_cascade_absolute_quality_train_v17/scanrefer/ssq_moe_e73_cascade_absolute_quality_e2/1785700377/`，
tmux 为 `mcln_v17_cascade_e1_e2`。它从 V12 epoch 2 network-only best 启动两轮 fresh-optimizer
训练，仅更新 `84,743` 个 V17 参数。按 V16 同规模真实周期估算，epoch 1 JSON 收据窗口为
`05:12-05:23 CST`，窗口前不读取中间训练日志。

正式两轮已完整结束。epoch 1 训练耗时 `3406.89s`，收据于 `05:14:57 CST` 落盘；epoch 2
训练耗时 `3750.63s`，收据于 `06:35:49 CST` 落盘。两轮五项逐项完全相同：learned REC 为
`0.5810896/0.4650820`（`5525/4422` hits），Mask 为
`0.5979175/0.4911653/0.4184406`（`5685/4670` hits）。相对 V12 epoch 2，REC@0.25 持平，
REC@0.5 多 2 hits，Mask 两个 threshold hits 持平；相对 V12 epoch 3 的最高 Mask mIoU，
形式上高 `0.0000281`。但两轮 correction 都是 `0/9508`，所以这些微小差异不能归因于 V17
第二阶段，更合理的解释是完整重评估中的数值或数据流微差。V17 没有把 REC 推到目标
`0.59/0.49`。

candidate oracle 两轮保持 `0.6296803/0.5500631/0.4517358`（`5987/5230` hits），动态 anchor
上存在 `1087/9508=11.43%` 的 oracle correction 行。oracle-query match 从 epoch 1 的
`419/1087=38.55%` 提高到 epoch 2 的 `441/1087=40.57%`，说明候选排序和绝对质量表征仍在
学习；但 predicted switch、beneficial switch、harmful switch 和 oracle-switch recall 始终
都是 0。评估中的平均最大 margin 仅从 `-1.0022` 改善到 `-0.9339`，主要失败点是稀疏正行下
的 correction 边界校准，而不是候选覆盖。

两份正式 checkpoint 均完成张量级审计。每份模型含 `1216` 个 tensor，optimizer 含 12 个
参数状态和 24 个状态 tensor，全部 finite；与 V12 共享的 `1198` 个 tensor 全部逐元素不变。
12 个实际训练的新头 tensor 共 `84,743` 参数，与 optimizer 形状逐项一致，并且 epoch 1 到
epoch 2 全部发生更新。另有 6 个由当前模型迁移时保存、但 V17 不使用的冻结
`safety_switch_head` tensor，两轮间也逐元素不变。

retention 将五项 best 都指向 epoch 1 inode `2194044680`，共 6 个硬链接，已保留。epoch 2
与 epoch 1 指标完全相同且 correction 仍为 0，因此 inode `2194044681` 的
`ckpt_epoch_2.pth`、`ckpt_epoch_last.pth` 两个链接已删除，释放 `604653697` bytes，config、
log、两轮 metrics/diagnostics 和 retention receipt 全部保留。清理后 V12 epoch 2 inode
`4306091917` 仍为 7 links，V12 epoch 3 inode `2170239162` 仍为 6 links；历史后处理
`0.582878/0.486012` 的三个组件和 router anchor 均未改动。

当前系统级 REC 最高仍是后处理 `0.582878/0.486012`。network-only REC@0.25 最高仍为
`0.5810896`，V17 的 REC@0.5 形式上更新为 `0.4650820`；Mask threshold 最高仍为
`0.5979175/0.4911653`，Mask mIoU 形式上更新为 `0.4184406`。下一版不应搜索 ScanRefer
validation 阈值，也不应只延长同一不平衡 objective；应保留 V12 动态 anchor 和 top-n 候选，
在 train split 上将“该行是否存在正 correction”和“正行内选择哪个 query”分开归一化训练，
显式消除 11.43% 正行被大量 exact-fallback 行压制的问题，再用跨数据集共享的风险校准部署。

### V18 opportunity-balanced correction cascade（2026-08-03）

V18 保留 V12 的动态 stage-1 anchor 和 V17 的 top-n correction 候选，但把训练与部署拆成
两个显式决策。`cascade_correction_head` 只负责正 opportunity 行内的 query 排序；新增
`cascade_opportunity_head` 使用候选集合的 masked mean/max pooling 判断当前行是否值得修正。
部署仅在 `opportunity_margin > 0` 时采用 rank-logit 最大的候选，否则严格回退 stage-1 anchor。
新 opportunity 输出层为零初始化，所以迁移时不会直接改变已有选择。

新 action/objective 分别为 `cascade_opportunity_quality_correction` 和
`cascade_opportunity_balanced_calibrated`。训练中 positive/fallback 行分别求均值后等权组合，
query ranking 只在确有 beneficial correction 的行上计算；这样训练边界不再由 88.57% fallback
行的数量主导。集合特征使用排列不变聚合，不依赖 ScanRefer validation 阈值，可直接复用于
单阶段 ScanRefer、Nr3D 和 Sr3D。实现涉及 `models/source_moe.py`、`models/losses.py`、
`main_utils.py`、训练脚本、reranker config 与相关测试；focused suite 为 `182 passed`，全套为
`2952 passed, 3 warnings`。

真实零初始化对照从 V17 retained checkpoint 启动。V17 debug run `1785711999` 为 REC
`64/57`、Mask `64/52/0.3505157`、candidate oracle `75/69`；V18 run `1785712186` 为 REC
`64/58`、Mask `64/52/0.3506319`、candidate oracle `75/69`，correction 为 0。差异同时出现在
fixed default 的一个 `@0.50` 边界样本，且 opportunity 分支没有执行，因此归因于两次独立
GPU/点采样评估的轻微非确定性，不是零初始化路由回归。

32 train + 32 eval smoke `1785712574` 从 V17 retained checkpoint 启动，仅训练
`absolute_quality_head`、`cascade_quality_adapter`、`cascade_correction_head` 和
`cascade_opportunity_head`。allowlist 恰好包含 18 个 tensor、`118,024` 参数；训练/评估
gate loss 为 `0.9377/0.9378`，所有 batch 与 checkpoint/optimizer tensor 均 finite。32 步后
`opportunity_positive_ratio` 与 correction 仍为 0，说明启动仍然保守，但该规模只覆盖 128 条
样本，不能替代完整正/负 opportunity 学习。张量审计确认与 V12 共享的 `1198` 个 tensor 全部
逐元素不变；相对 V17 的 12 个变化 tensor 全部位于允许的新头，新增 opportunity head 为 6 个
tensor、`33,281` 参数。smoke inode `39860010` 的 7 个 checkpoint 链接已删除，config、log 和
JSON receipts 保留。

V18 正式 run 于 `07:24:14 CST` 启动，目录为
`output/source_moe_cascade_opportunity_train_v18/scanrefer/ssq_moe_e73_cascade_opportunity_e2/1785713059/`，
tmux 为 `mcln_v18_opportunity_e1_e2`。它从 V17 retained checkpoint 进行两轮全量 ScanRefer
fresh-optimizer gate-only 训练，只更新上述 `118,024` 个参数。按 V16/V17 实测周期估算，epoch 1
收据窗口为 `08:41-08:52 CST`，epoch 2 为 `10:02-10:13 CST`，窗口前不轮询中间日志。

启动后再次核验：V17 inode `2194044680` 仍为 6 links，V12 epoch 2/3 inode
`4306091917/2170239162` 仍为 `7/6` links，历史后处理 `0.582878/0.486012` 的三个保护组件仍在
`protected_mcln_artifacts/`。当前最高指标尚未变化：系统级 REC 仍为后处理
`0.582878/0.486012`，network-only 为 `0.5810896/0.4650820`。

V18 epoch 1 收据于 `08:52:38 CST` 落盘，训练耗时 `3718.11s`。正式 learned REC 降为
`0.5515356/0.4216449`（`5244/4009` hits），Mask 为
`0.5942364/0.4868532/0.4148749`（`5650/4629` hits）；五项均未刷新 retained 指标。
candidate oracle 仍有 `0.6296803/0.5496424/0.4516072`，说明候选覆盖没有消失。

V18 确实解决了 V17 完全不切换的问题：oracle-switch recall 提高到
`265/1087=24.38%`，oracle-query match 为 `517/1087=47.56%`。但预测了 `2651/9508=27.88%`
行切换，其中 beneficial/harmful 为 `265/2386`，precision 仅 `10.00%`，false-switch rate
达到 `90.00%`。根因不是简单阈值偏移，而是监督与部署动作不一致：row opportunity 的 target
表示“行内存在任意 beneficial query”，部署却采用 rank head 实际选中的单个 query；当排序未
命中 oracle query 时，正确的 row 判断仍会执行有害动作。batch 内逆频率平衡同时移除了真实
class prior，进一步放大了正预测比例。

### V19 selected-query verified opportunity（2026-08-03）

V19 在 V18 的 row opportunity 与 conditional rank 后增加候选级
`cascade_candidate_safety_head`。新 action/objective 为
`cascade_opportunity_verified_correction` / `cascade_opportunity_verified_calibrated`。
rank head 仍只在正 opportunity 行学习候选排序；safety head 则对每个候选直接监督其相对动态
anchor 的真实 threshold utility，并使用保留经验 class prior 的 cost-sensitive focal loss 与
负 safety-gap utility regression。这样不会通过 ScanRefer validation 阈值恢复 precision。

部署先按 rank logit 选 query，再取该 query 的 safety margin；最终 correction margin 为
`min(row_opportunity_margin, selected_query_safety_margin)`，两个条件都严格大于 0 才执行。
safety 最终层零初始化，因此从 V18 checkpoint 迁移时即使旧 opportunity 已大面积为正，最终
margin 仍为 0，逐元素回退 V12 动态 anchor。该分解直接修复“存在好候选”和“实际所选候选安全”
之间的 label/action mismatch，也适用于单阶段 ScanRefer、Nr3D 与 Sr3D。

V19 新 safety head 含 6 个 parameter tensor、`16,897` 参数；连同 V18 四个模块，new-head-only
allowlist 共 `24` 个 tensor、`134,921` 参数。checkpoint 合同允许 V12/V17/V18 迁移时缺少对应
后续模块，但声明 V19 action 的 checkpoint 缺少 safety head 时 fail closed。实现覆盖模型、
loss、CLI、训练脚本、reranker config、optimizer/train-mode allowlist 与迁移校验。定向测试
`189 passed`，全套为 `2959 passed, 3 warnings`。

V18 epoch 2 已按用户要求完整结束：训练耗时 `3603.72s`，正式收据于 `10:10:54 CST` 落盘。
learned REC 为 `5264/9508=0.5536390`、`4070/9508=0.4280606`，Mask 为
`5643/9508=0.5935002`、`4629/9508=0.4868532`、mIoU `0.4144721`；candidate oracle 保持
`5987/9508=0.6296803`、`5226/9508=0.5496424`、mIoU `0.4516072`。预测切换进一步增加到
`3004`，其中 beneficial/harmful 为 `320/2684`，precision `10.65%`、oracle-switch recall
`29.44%`，oracle-query match `544/1087=50.05%`。因此增加训练时长只提高了 recall，没有修复
约 `89.35%` 的 false-switch，V19 的 selected-query verifier 是必要的结构修正。

V18 清理按“只保留一个 V19 初始化实体”执行。epoch 1 inode `4299478646` 的 4 个链接全部
删除；epoch 2 inode `4299478647` 的 latest/两项 REC-best 共 3 个冗余链接删除，只保留
`ckpt_epoch_2.pth`，SHA-256 为
`126b28dc2db2b2fc89743ecb1da56e0074facc3d1b1db8dbedde7452d6b7eb10`。config、log、两轮
metrics/diagnostics 和 retention receipt 均保留。复核后 V12 epoch 2/3 inode
`4306091917/2170239162` 仍为 `7/6` links，V17 inode `2194044680` 仍为 6 links，历史
`0.582878/0.486012` 三个保护组件未动。

V19 零初始化评估 run `1785723597` 从该 V18 epoch 2 权重迁移，使用 `batch_size=4`、
`num_workers=0` 的固定 128-row panel。checkpoint 加载逻辑将收据标签推进为 epoch 3，但没有
执行训练。REC 为 `64/128`、`57/128`，Mask 为 `64/128`、`52/128`、mIoU `0.3505157`，
candidate oracle 为 `75/128`、`69/128`、mIoU `0.4406735`。旧 row opportunity 有 `29.69%`
为正，但新 safety-positive、predicted correction、beneficial/harmful switch 全部为 0，逐元素
保持动态 V12/V17 anchor，零初始化部署合同成立。

32 train + 32 eval smoke run `1785723824` 的 gate loss 为 `0.8471/0.8438`，训练/评估
safety loss 为 `0.1918/0.1589`；32 步后仍为 0 次 correction 和 0 次 harmful switch，说明
初始 verifier 保守但损失已正常参与优化。checkpoint 审计确认 V12 的 `1198` 个共享 tensor
全部逐元素不变，V18 已存在的 18 个允许 tensor 全部更新；optimizer 恰好包含 24 个 state、
48 个非零 Adam moment tensor，step 均为 32，moment numel 与 allowlist 同为 `134,921`，
模型与 optimizer 全部 finite。smoke inode `4311962973` 的 7 个权重链接已删除，释放
`605267869` bytes，config、log 和 JSON receipts 保留。

V19 正式两轮 run `1785724401` 于 `10:33:15 CST` 启动，路径为
`output/source_moe_cascade_opportunity_verified_train_v19/scanrefer/ssq_moe_e73_cascade_opportunity_verified_e2/1785724401/`，
tmux 为 `mcln_v19_verified_e1_e2`。它从唯一 V18 epoch 2 初始化权重启动 fresh optimizer，完整
验证样本数为 `9508`，只更新 24 个新头 tensor。按 V18 首轮总周期约 88 分钟、次轮约 78 分钟
估算，epoch 1 收据窗口为 `11:55-12:05 CST`，epoch 2 为 `13:15-13:25 CST`；窗口前不读取
中间训练日志。

V19 epoch 1 实际训练耗时 `3210.24s`，收据于 `11:49:36 CST` 落盘。REC 为
`5525/9508=0.5810896`、`4424/9508=0.4652924`，Mask 为
`5687/9508=0.5981279`、`4671/9508=0.4912705`、mIoU `0.4185490`。只预测 6 次 correction，
beneficial/harmful 为 `2/4`，precision `33.33%`、oracle-switch recall `0.18%`；与 V18 的
`3004` 次切换相比，selected-query verifier 已把过度切换压住，但开始明显欠切换。

按首轮真实周期把 epoch 2 收据窗口修正为 `12:56-13:02 CST`，最终收据于 `13:04:26 CST`
落盘，训练耗时 `3559.98s`。REC 进一步达到 `5526/9508=0.5811948`、
`4425/9508=0.4653976`，Mask 达到 `5688/9508=0.5982331`、
`4672/9508=0.4913757`、mIoU `0.4186131`；五项均刷新 network-only/Mask retained 最佳。
candidate oracle 仍为 `5987/9508=0.6296803`、`5230/9508=0.5500631`、mIoU
`0.4517077`。最终只切换 5 行，其中 beneficial/harmful 为 `2/3`，precision `40.00%`、
oracle-switch recall `2/1085=0.18%`，oracle-query match `540/1085=49.77%`。

最终 checkpoint 审计通过：V12 的 `1198` 个共享 tensor 全部逐元素不变；与 V18 共有的
`1222` 个 tensor 中只变化 18 个允许的新头 tensor，没有 allowlist 外变化。V19 allowlist 恰好
24 个 tensor、`134,921` 参数；optimizer 含 24 个 state、48 个非零 Adam moment tensor，step
均为 `6110`，模型和 optimizer 全部 finite。epoch 2 inode `34391215` 的 7 个 retention 链接
全部保留，并在 `protected_mcln_artifacts/` 增加同 inode 的保护别名
`scanrefer_network_best_v19_rec025_0.581195_mask025_0.598233.pth`，当前共 8 links。epoch 1
inode `38114394` 因五项均被 epoch 2 超过已由 retention 自动删除；临时 V18 初始化 inode
`4299478647` 的最后一个链接也在审计后删除，config/log/JSON 收据继续保留。

当前系统级 REC 最高仍为历史后处理 `0.582878/0.486012`，三个保护组件未动；network-only
最高更新为 V19 的 `0.5811948/0.4653976`，Mask 最高更新为
`0.5982331/0.4913757/0.4186131`。按 9,508 条样本计算，network-only 距 `0.59/0.49` 仍缺
`84/234` hits；历史后处理距目标仍缺约 `68/38` hits。候选 oracle 已充分超过目标，因此瓶颈
仍是把候选覆盖转化为可靠 action，而不是继续扩大候选池或扫描 validation 阈值。

### V20 candidate-level joint risk action（2026-08-03）

V20 修复 V19 的剩余 action mismatch。V19 先用 rank head 选一个 query，再只验证该 query；若
首选不安全，即使同一行还有安全候选也直接 fallback。最终实现没有采用早期草案中的
`min(opportunity, safety)` 硬规则，而是为每个候选拼接 correction hidden、candidate rank
margin、candidate safety margin 和 row opportunity margin，交给可学习的
`cascade_joint_action_head` 输出最终 `candidate_joint_margin[q]`。部署在全部有效候选上联合
argmax，再与固定 fallback logit `0` 比较；旧 `decision_margin` 不参与 V20 边界，避免重新引入
数据集阈值。

新 action/objective 为 `cascade_joint_risk_correction` / `cascade_joint_risk_calibrated`。训练
使用 fallback-plus-candidates 的 row-wise risk-separated target：正行只向正 utility 候选分配
概率，非正行全部回到 fallback；正/fallback 行分别归一化，再使用训练 batch 的 Beta(1,1)
平滑 opportunity prior 和 `false_override_weight` 做 detached log-odds 校正。最终联合层零初始化，
所以 V19 -> V20 第 0 步严格不做 correction，回退动态 V12 anchor。

hidden dim 128 时联合头新增 6 个 parameter tensor、`17,281` 参数；V20 new-head-only 白名单
合计 30 个 tensor、`152,202` 参数。checkpoint 合同允许 V12/V17/V18/V19 分别缺少其后新增
模块，但声明 V20 action 的 checkpoint 缺任一 V20 模块都会 fail closed。相关模型、loss、CLI、
训练脚本、reranker config、optimizer/train-mode 与迁移测试已接通；SourceMoE 相关回归为
`243 passed, 2 warnings`。

真实 V19 protected epoch 2 权重迁移后的 128-row 零评估 run `1785735825` 得到 REC `64/57`、
Mask `64/52/0.3504906`、candidate oracle `75/69/0.4407775`。联合 margin、predicted correction、
beneficial/harmful correction 均为 0，证明零初始化部署合同成立；V19 protected inode
`34391215` 仍为 8 links，历史后处理 `0.582878/0.486012` 三个组件未动。

32 train + 32 eval smoke run `1785736073` 从同一 V19 protected 权重启动。联合 action loss
由零评估的 `3.4084` 降到 smoke eval 的 `2.9193`，总 gate loss 为 `1.2567`；32 步后仍为
0 次 correction，因此没有 harmful switch。checkpoint 审计确认：V19 共有的 `1228` 个模型
tensor 中恰好只改变 24 个允许 tensor，新增 joint head 恰好 6 个 tensor；V20 allowlist 合计
30 个 tensor、`152,202` 参数。optimizer 含 30 个 state，60 个 Adam moment tensor 全部非零，
step 均为 32，moment numel 为 `152,202`，模型/optimizer 全部 finite。smoke inode
`2162713570` 的 7 个 `.pth` 链接已删除，config、log、metrics/diagnostics 和 retention receipt
保留；V19 protected inode 与历史三件套再次复核未动。

最终全项目回归为 `2968 passed, 3 warnings`。V20 正式两轮 run `1785736801` 于
`14:00:01 CST` 启动，目录为
`output/source_moe_cascade_joint_risk_train_v20/scanrefer/ssq_moe_e73_cascade_joint_risk_e2/1785736801/`，
tmux 为 `mcln_v20_joint_risk_e1_e2`。它从 V19 protected epoch 2 启动 fresh optimizer，完整
训练/验证样本数为 `36665/9508`，只更新 30 个 V20 allowlist tensor。按 V19 实测总周期估算，
epoch 1 收据窗口为 `15:17-15:25 CST`，epoch 2 为 `16:32-16:42 CST`；窗口前不轮询训练日志。

V20 收据产生前已预先冻结验收口径：REC 必须同时达到 `5610/9508` 和 `4659/9508`；Mask
以 V19 network checkpoint 的 `5688/4672/0.4186131` 为保护参照，并同时审计 oracle、
beneficial/harmful correction、precision、recall 与 joint-action target match。静态泛化复核
确认 routed-expert top-k 已是逐 query 自适应，但 shared/routed 混合强度仍是全局
`routed_scale`，因此不把它夸大为完整的逐目标源权重。若 V20 oracle 达标而 learned 未达标，
下一步仍处理联合 action；oracle 不足时才引入零初始化的逐 query residual scale；REC 达标但
Mask 回退时转向 query-specific mask fusion。上述分流不使用 validation threshold 或固定源组合
sweep，并作为后续单阶段 ScanRefer、Nr3D、Sr3D 的同实现迁移合同。

跨数据集接口复核确认 MoE 的 `sample_dataset` mask 已支持 ScanRefer/Nr3D/Sr3D，当前未完成项
在 launcher：现有正式脚本仍硬编码 `--butd`、`test_dataset=scanrefer` 和 validation `9508`
样本。因此后续迁移必须新增单阶段及 Nr3D/Sr3D 启动合同并验收 source tensor，不直接复用
ScanRefer 命令或在缺源时静默选择固定组合。

新增只读 checkpoint 审计器 `scripts/audit_source_moe_checkpoint.py`，用于正式收据落盘后一次性
验证 V19 -> V20 的 `1228` 个共有、`24` 个允许变化、`6` 个新增 tensor，30 个 optimizer
state、`152,202` moment numel、epoch 1/2 step `3055/6110`、全 tensor finite/nonzero 和
action/objective/new-head-only 配置。聚焦 oracle/retention/checkpoint 测试为 `18 passed`；
审计通过前不创建 protected best alias，也不删除本次正式权重。

V20 epoch 1 的训练/验证实际耗时约 `74:48/22:51`，正式收据于 `15:46:52 CST` 落盘。
learned REC 为 `5524/4421 = 0.5809844/0.4649769`，Mask 为
`5685/4670/0.4183627`；相对 V19 network-best 少 `2/4` REC hits、`3/2` Mask hits，mIoU
低 `0.0002504`，未刷新任何最佳。candidate oracle 仍达
`5987/5228 = 0.6296803/0.5498528`、mIoU `0.4516537`。

gate 的 oracle switch 为 `1090`、oracle-query match 为 `522`，但 predicted/beneficial/harmful
correction 均为 `0`，precision/recall 也为 0；joint-action target match 约 `88.50%`。因此
epoch 1 的失败点是联合风险边界过度保守而非候选不足或 harmful switch。仍按用户要求完成
epoch 2，不提前停止；依据首轮约 `97:39` 的完整周期，epoch 2 收据 ETA 修正为
`17:23-17:27 CST`。最终权重清理等待两轮指标与 checkpoint 审计完成后执行。

epoch 1 后预注册的结构分流是：若 epoch 2 仍零/近零 correction，不继续同配置或扫 threshold；
V21 以 V19 已部署 action 作为零初始化 fallback，只训练新的 fallback-token set action head，
把 joint risk 设为主目标，并用 train-prior coverage lower-bound/class-count margin 防止
all-fallback collapse。开源参考固定为 SelectiveNet `a6d0a8f`、PCGrad `e987ac6`、
LDAM-DRW `2536330` 和 sparse MoE `f662999`；前两者分别只在不引入 validation coverage、
且 probe 证明确有梯度冲突时采用。当前 candidate oracle 和 expert balance 已通过，所以不优先
重写 router。

V20 epoch 2 于 `17:19:21 CST` 产出正式收据，结果与 epoch 1 逐项相同：REC
`5524/9508=0.5809844`、`4421/9508=0.4649769`；Mask
`5685/9508=0.5979175`、`4670/9508=0.4911653`、mIoU `0.4183627`；candidate oracle
`5987/9508=0.6296803`、`5228/9508=0.5498528`、mIoU `0.4516537`。第二轮仍为 0 次
predicted/beneficial/harmful correction，oracle switch 为 1090，因此两轮共同确认
joint-action coverage collapse。问题不是没有加载预训练或只训了一轮；继续相同 objective 也
没有证据能恢复覆盖。V20 两轮低指标 `.pth` 已清理，日志、两轮 metrics/diagnostics 和 checkpoint
审计保留。V19 protected inode `34391215` 与历史 `0.582878/0.486012` 三组件未删除。

### V21 V19-fallback set action（2026-08-03）

V21 已按预注册方向实现，action/objective 固定为
`cascade_v19_fallback_set_correction` / `cascade_v19_fallback_set_risk_calibrated`。它完整复现
V19 已部署 query 作为 fallback，从 correction set 移除该 query；若 V19 做过 correction，则把
V12 anchor 重新加入候选，使新 head 可以撤销有害的旧 correction。新 head 是无位置编码的
fallback-token set attention，输出候选 logit 相对 fallback logit 的 margin；score 层零初始化，
因此第 0 步逐 query、逐 score 精确保留 V19。utility、loss target 与 evaluator oracle 均相对
V19 fallback 计算。

V19 的 24 个 evidence tensors 及其输入全部冻结/detach，只训练新 set head 的 15 个 parameter
tensors、`149,504` 个参数。loss 组合 prior-corrected fallback-plus-candidates setwise risk 与
正行/fallback 行分别平均的 deployment-boundary loss，直接给零边界两侧提供梯度，不从
ScanRefer validation 搜 threshold。迁移 fail closed：只接受完整 V19 或完整 V21；V12/V17/
V18/V20 以及缺任一 V19 evidence tensor 均拒绝。

实现覆盖 `models/source_moe.py`、`models/losses.py`、`main_utils.py`、evaluator、正式 launcher、
reranker config、checkpoint auditor 与行为测试。首次 smoke run 在首 batch 前发现
`moe_gate_loss_fallback_query` 被通用 finite guard 误判为标量 loss；改名为
`moe_gate_supervision_fallback_query` 后解决，该失败 run 没有权重。成功 smoke run
`1785751933` 完成 32 train + 32 eval：train/eval gate loss `4.1962/3.9136`、joint loss
`3.5153/3.2434`、boundary loss `0.6810/0.6702`，所有 batch finite。128-row REC 为
`0.5000/0.4453125`，Mask 为 `0.5000/0.40625/0.350190`；该 panel 只验证可学习性与稳定性，
不与 9,508-row 正式指标比较。

smoke checkpoint 审计严格通过：相对 protected V19 的 common/changed/new 为
`1228/0/15`，optimizer 为 15 states、`149,504` numel、step 32，30 个 Adam moments 均 finite
且非零。inode `37694830` 的 7 个 smoke `.pth` 硬链接已删除，释放约 606 MB；config、log、
metrics、diagnostics、retention 与 audit JSON 保留。审计 debug epoch 0 的 fixture 修正后聚焦
测试为 `8 passed`，最终全项目回归为 `2988 passed, 3 warnings`。

当前最高指标没有变化：系统级 REC 仍为受保护的后处理 `0.582878/0.486012`；network-only
REC 仍为 V19 `0.5811948/0.4653976`；Mask 仍为 V19
`0.5982331/0.4913757/0.4186131`。V21 正式结果必须使用完整 9,508-row validation 后再判断，
不得用 smoke panel 或 candidate oracle 代替 learned 指标。

V21 正式两轮 run `1785752995` 已于 `18:29:45 CST` 从受保护 V19 启动，tmux
`mcln_v21_v19_fallback_set_e1_e2`，路径为
`output/source_moe_v19_fallback_set_train_v21/scanrefer/ssq_moe_e73_v19_fallback_set_e2/1785752995/`。
正式 config 已核对：`36665/9508` 完整 train/validation、`batch_size=12`、`num_workers=4`、
gate LR `3e-4`、temperature `0.25`、V19 evidence/context 参数一致，且
`source_moe_gate_new_heads_only=true`。按最接近的 V20 真实周期估算，epoch 1/2 收据只在
`20:15-20:25 CST` / `21:47-21:57 CST` 窗口检查，不做几分钟级轮询。

V21 正式训练已完整结束。实际数据加载于 `18:38:59 CST` 完成，epoch 1/2 训练耗时
`3248.84s/3205.42s`，正式收据分别在 `19:48:18/20:54:53 CST` 写入。epoch 1 REC
`5158/3884 = 0.5424905/0.4084981`，Mask
`5607/4586 = 0.5897139/0.4823307/0.4114272`；epoch 2 REC
`5168/3878 = 0.5435423/0.4078671`，Mask
`5632/4604 = 0.5923433/0.4842238/0.4131095`。第二轮只是 Mask 和 REC@0.25 小幅回升，
REC@0.50 继续下降，两轮都未接近 V19。

candidate oracle 两轮固定为 `5989/5227 = 0.6298906/0.5497476`、mIoU `0.4516835`。
epoch 1 switch `3549`，beneficial/harmful `362/3187`、precision `10.20%`；epoch 2 switch
进一步增至 `3911`，beneficial/harmful `346/3565`、precision `8.85%`。真实 oracle switch
始终为 `1088=11.44%`，所以 balanced boundary 将经验 prior 抹平后制造了 `37.33% -> 41.13%`
过切，完整第二轮已经排除“训练不够”的解释。

epoch 1/2 checkpoint 审计均通过：V19→V21 common/changed/new `1228/0/15`，optimizer
15 states、`149,504` numel，step `3055/6110`，全部 finite 且 Adam moments 非零。低指标
inode `37694825/16720963` 的 8 个权重链接已全部删除，约释放 1.21 GB；所有结构化收据保留。
V19 protected inode `34391215` 仍为 8 links，历史后处理三组件仍完整。

V22 的结构分流已据此收紧：不调 validation threshold、不改一个全局 class weight，也不再只在
同一 428 维 gate evidence 上添加 veto。代码审计确认在线 SourceMoE 缺少冻结 sidecar 已证明
有效的 152 维 deployable rich evidence，包括目标文本投影、五类语言成分得分、scene-normalized
box、source top-two margin、objectness、mask confidence/foreground/text-query Dice 和 target
cosine。下一版将这套无 GT 特征接入 fallback-token set query reorder，并只使用一个保留真实 row
prior 与 false-switch cost 的 proper set risk objective；不再同时使用 prior-corrected loss 与等权
boundary。缺失 source/feature 必须用显式 validity mask，保持同一模块可用于单阶段 ScanRefer、
Nr3D 和 Sr3D。

### V22 rich-evidence empirical set risk（2026-08-03）

V22 已完成实现。`cascade_v19_rich_set_correction` 将 `rec-query-v1` 的 152 维无 GT rich
evidence 接入 V19-fallback token set head；完整 V19 action 仍是 fallback，最后共享评分层零初始化，
部署边界固定为 0。所有 V19 evidence 参数和输入均冻结/detach，只训练新 head 的 17 个 tensor、
`169,264` 个参数。目标 `cascade_v19_rich_set_empirical_risk` 取消 V21 的 balanced boundary，
保留经验 row prior；fallback logit 固定为 0，false-switch cost `2.0` 仅作用于 fallback rows。
该训练目标不使用 ScanRefer validation threshold。

迁移合同和 checkpoint auditor 同步收紧：只接受完整 V19 或完整 V22，拒绝 V20/V21 与残缺
V19；V19->V22 必须为 common/changed/new `1228/0/17`、17 optimizer states、`169,264`
numel。128-row 零初始化 run `1785764542` 的 REC 为 `64/57`、Mask
`64/52/0.3504906`，0 次 correction，与 V19 同 panel 精确一致。成功 smoke run
`1785764713` 完成 32 train + 32 eval，审计为 `1228/0/17`、17 states、step 32，Adam moments
全部 finite/nonzero；7 个 smoke 权重链接已清理，结构化收据保留。全项目回归为
`3006 passed, 3 warnings`。

正式 run `1785765110` 使用 protected V19、完整 `36665/9508` 样本、`batch_size=12`、
`num_workers=4`，共训练两轮。epoch 1 REC
`5521/4420 = 0.5806689/0.4648717`，Mask
`5682/4671 = 0.5976020/0.4912705/0.4182182`；相对 V19 network-best 少 `5/5` REC hits、
`6/1` Mask hits，尚未刷新指标。candidate oracle 为
`5987/5230 = 0.6296803/0.5500631/0.4517544`；learned switch 只有 168，beneficial/harmful
`22/146`、precision `13.10%`、oracle recall `2.03%`。epoch 1 checkpoint 审计严格通过：
`1228/0/17`、17 states、`169,264` numel、step 3055。epoch 2 按用户要求完整训练和验证，
不根据首轮结果提前终止；最终指标、失败归因与权重清理等待 epoch 2 收据后补录。

#### V22 epoch 2 正式结果与清理（2026-08-04）

epoch 2 已完整跑完 3,055 个 train batches 和 793 个 validation batches。正式 receipt 于
`2026-08-04 00:09:23 CST` 落盘：REC learned `5521/4418 = 0.5806689/0.4646613`，Mask
`5680/4670 = 0.5973917/0.4911653/0.4181528`。epoch 1 为 `5521/4420 =
0.5806689/0.4648717`、Mask `5682/4671/0.4182182`；第二轮没有带来有效提升。

candidate oracle 两轮均为 `5987/5230 = 0.6296803/0.5500631`、mIoU `0.4517544`。epoch 2
learned 只切换 104 行，beneficial/harmful `14/90`，precision `13.46%`；真实 oracle
opportunity `1085/9508 = 11.41%`，oracle-query match `500/1085 = 46.08%`，switch recall
`1.29%`。因此候选集本身足够，V22 的失败是 learned set action 的 under-switch/候选质量排序
瓶颈，不是预训练加载或训练轮数不足。

`ckpt_epoch_2.pth` 审计通过：V19->V22 common/changed/new `1228/0/17`，optimizer 17 states、
`169,264` numel、step `6110`，模型与 moments 全部 finite/nonzero。epoch 1/2 的 8 个正式
`.pth` 链接已清理，约释放 `1.2 GiB`；`eval_metrics_epoch_{1,2}.json`、
`source_choice_diagnostics_epoch_{1,2}.json`、`checkpoint_retention.json`、config、log 和
`v22_audit_epoch_{1,2}.json` 保留。清理后 protected V19 inode `34391215` 仍为 8 links，
历史 `0.582878/0.486012` 三组件仍为只读 2 links。

V22 未刷新当前任何 best：系统级最高仍为后处理 REC `0.582878/0.486012`，network-only 最高
为 V19 `0.5811948/0.4653976`，Mask 最高为 V19 `0.5982331/0.4913757/0.4186131`。下一步
执行已预注册的 dense-quality adaptive source MoE：用所有候选的 box/mask quality 密集监督
训练 rich set 表示，并将全局 routed scale 改为带 validity mask 的逐 query shared/routed
mixing；先通过 identity、排列、缺源、梯度覆盖和 32-step smoke 后再正式训练。

若 epoch 2 仍未达 `0.59/0.49` 而 candidate oracle 保持达标，下一实验已预注册为
`dense-quality adaptive source MoE`，不继续同一 loss 或扫描 validation margin。它冻结 V19
作为零初始化 fallback，在 rich set encoder 上对所有有效 candidates 密集预测 box/mask 的两个
threshold 与 IoU，并以 dense BCE/Huber/tier-listwise loss训练表示；同时把当前全局
`shared-vs-routed` scale 改成 rich-evidence 条件化的逐 query residual mixing。default 为 shared
expert，contrastive/mask 为 routed experts，缺失 source 使用显式 `[B,Q,S]` validity mask。
最终仍以 candidate 相对 V19 fallback 的零边界 risk margin 部署，不使用数据集专用阈值。
该分流只有在 identity、dense gradient coverage、source absence/permutation、query permutation、
optimizer allowlist 和 32-step smoke 全部通过后才允许正式训练。

### V23 dense-quality adaptive source MoE（2026-08-04）

V23 已实现并通过正式训练门禁。固定 action/objective 为
`cascade_v23_dense_quality_correction` / `cascade_v23_dense_quality_risk`。新结构包含
rich-evidence 条件化的逐 query `AdaptiveSourceMixer`，以及对 fallback 和最多 8 个候选密集
预测 box/mask `IoU>0.25`、`IoU>0.50`、连续 IoU 的 set attention quality head。default 为
shared source，contrastive/mask 为 routed sources；缺失 source 由显式 `[B,Q,S]` validity
mask 排除。部署仍以 candidate quality 相对完整 V19 fallback 的零边界 margin 决策，不搜索
ScanRefer validation threshold。

V19 evidence 和输入全部冻结/detach，只训练 adaptive mixer 与 dense quality head。输出层
零初始化并用 `torch.random.fork_rng` 隔离新增模块初始化，保证第 0 步 V19 identity。删除了
对所有 routed logits 同加常数、理论上恒等的 router bias 后，生产合同为 39 个新 tensor、
`588,603` 参数。checkpoint 只允许完整 V19 迁移或精确 V23 resume，拒绝 V20/V21/V22。

测试结果为聚焦 `251 passed`、全项目 `3017 passed, 3 warnings`。128-row identity run
`1785776794` 的 REC `64/57`、Mask hits `64/52`，0 次 correction，与 V19 panel 阈值命中一致；
Mask mIoU 的约 `1.4e-4` 差异属于 CUDA 浮点波动。32 train + 32 eval smoke run
`1785776991` 完成后，checkpoint 审计严格通过 `1228/0/39`、39 optimizer states、
`588,603` numel、step 32，模型及 78 个 Adam moment tensors 全部 finite/nonzero。

smoke inode `10744414654` 的 7 个 `.pth` 硬链接已清理，结构化收据保留。清理后 protected V19
inode `34391215` 仍为 8 links，历史系统级 `0.582878/0.486012` 三组件仍完整。下一步从该
protected V19 启动两轮完整 `36,665/9,508` train/validation；只在基于真实首轮耗时估算的
收据窗口检查，不做几分钟级轮询。两轮均未刷新 best 时，审计后删除 V23 正式低指标权重；
刷新任一受保护指标时先创建独立 protected alias，再清理冗余硬链接。

#### V23 正式两轮结果、失败归因与清理（2026-08-04）

正式 run `1785777457` 从 protected V19 启动，完整使用 `36,665/9,508` train/validation、
`batch_size=12`、`num_workers=4`、gate LR `3e-4`。epoch 1/2 训练耗时为
`2978.88s/3002.69s`，receipt 于 `02:29:30/03:32:43` 写入。

epoch 1 learned REC `5507/9508=0.5791965`、`4418/9508=0.4646613`，Mask
`5684/9508=0.5978124`、`4671/9508=0.4912705/0.4182359`；epoch 2 learned REC
`5516/9508=0.5801430`、`4420/9508=0.4648717`，Mask
`5681/9508=0.5974968`、`4672/9508=0.4913757/0.4181260`。epoch 2 Mask@0.50 与 V19
`4672/9508=0.4913757` 并列但未刷新；REC 仍低于 V19 network-best
`0.5811948/0.4653976`，目标 `0.59/0.49` 未达到。

candidate oracle 两轮为 `5993/5231`、`5992/5228`，仍明显高于 learned 结果。learned switch
分别为 `522/249` 行，beneficial/harmful `57/465`、`21/228`，precision `10.92%/8.43%`，
oracle-switch recall `5.26%/1.94%`；第二轮没有消除 false-switch。结论是候选覆盖、预训练加载、
optimizer 或训练轮数都不是主要瓶颈，V23 的 dense quality/action risk 排序仍不够可靠，不能
继续同配置或做数据集专用 margin 搜索。

epoch 1/2 checkpoint 审计均通过 `1228/0/39`、39 states、`588,603` numel、step `3055/6110`，
模型和 Adam moments 全部 finite/nonzero。由于没有刷新任何严格 best，正式目录的 8 个 V23
权重链接已删除；两轮 metrics、diagnostics、retention、config、log 和
`v23_audit_epoch_{1,2}.json` 保留。protected V19 inode `34391215` 仍为 8 links，历史系统级
`0.582878/0.486012` 三组件仍完整。当前最高口径仍为：后处理 `0.582878/0.486012`，
network-only REC `0.5811948/0.4653976`，Mask `0.5982331/0.4913757/0.4186131`。

### V24 relative-risk fallback set（2026-08-04）

V24 在 V23 dense quality head 后加入 permutation-equivariant
`RelativeRiskFallbackSetActionHead`，固定 action/objective 为
`cascade_v24_relative_risk_correction` / `cascade_v24_relative_risk`。候选集合显式保留
V19 fallback，relative utility target 在没有正收益候选时监督 fallback，避免新增模块把
所有样本强制切换到候选。V19 初始化为零输出并保持部署边界 `>0`，V23 absolute quality
监督保留；新增 head 不使用 query position，支持 query permutation。

实现门禁：聚焦回归 `116 passed`，Python compile 通过；128-row GPU smoke 成功，训练参数
`759,167`、optimizer states `60`，审计 `1228/0/60`，模型与 Adam moments 全部 finite/nonzero。
正式 run `1785787527` 从只读 protected V19 启动，完整 train/validation 为 `36,665/9,508`，
`batch_size=12`，两轮训练耗时 `2999.57s/2968.19s`。

| epoch | REC @0.25 | REC @0.50 | Mask @0.25 | Mask @0.50 | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `5516/9508=0.580143` | `4414/9508=0.464241` | `5675/9508=0.596866` | `4657/9508=0.489798` | `0.417244` |
| 2 | `5519/9508=0.580459` | `4419/9508=0.464767` | `5678/9508=0.597181` | `4670/9508=0.491165` | `0.417964` |

epoch 2 candidate oracle 为 `5987/5226`（约 `0.63031/0.54964`），但 learned gate 只切换
84 行，beneficial/harmful `10/74`，false-switch `88.10%`，oracle-switch recall `0.93%`。
epoch 1 为 `279` 次切换、`50/229`、false-switch `82.08%`、recall `4.59%`。这再次说明
候选覆盖充足而风险排序不可靠；V24 仍低于 `0.59/0.49`，不能归因于未加载预训练或训练
轮数不足。

epoch-2 checkpoint 审计通过：V19->V24 `1228/0/60`，60 optimizer states、`759,167`
parameter numel、step `6110`，contract 完整。metric retention 先清理了正式 epoch-1 权重；
由于 epoch-2 也未超过受保护 global best，随后将正式 run 的全部 7 个 `.pth` 移入该 run 的
`quarantine_low_metric/`，便于恢复但不再混入可发布目录。smoke 低指标权重同样移入
`source_moe_v24_smoke/.../quarantine_low_metric/`；protected V19 及历史
`0.582878/0.486012` 后处理组件保持完整。

因此当前最高仍为：系统级后处理 REC `0.582878/0.486012`；network-only V19
`0.5811948/0.4653976`；Mask V19 `0.5982331/0.4913757/0.4186131`。V24 权重不进入
protected best，也不作为下一版初始化。

#### V24 dense-quality action 消融（2026-08-04）

从 V24 epoch-2 临时剥离 relative-risk deployment，恢复 dense-quality margin 后，独立完成
`9,508` 行 validation；临时 checkpoint 已删除。official REC 为
`5474/4431 = 0.5757257/0.4660286`，Mask 为
`5657/4640 = 0.5949727/0.4880101/0.4155182`。结构化 diagnostics 的 correction 为
`4030/9508 = 42.39%`，beneficial/harmful `319/3711`，precision `7.92%`、false-switch
`92.08%`、oracle recall `29.37%`。因此 V23 dense margin 是过切端，V24 relative margin 是
欠切端；两端都证明核心问题是零部署边界附近的连续效用校准，而不是 candidate oracle 覆盖。

代码复核还发现 V24 虽属于 calibrated objective，但先命中 dense-quality loss 分支，实际总 loss
没有包含 `_calibrated_utility_regression_loss`，对应训练日志一直为 0；同时共享 token scorer 的
`candidate_logit - fallback_logit` 没有 bias，也没有显式 candidate/fallback 角色比较。这两点
成为 V25 的固定修复项。

### V25 pairwise calibrated fallback risk（2026-08-04）

V25 已实现 `cascade_v25_pairwise_calibrated_correction` /
`cascade_v25_pairwise_calibrated_risk`。新 `CalibratedPairwiseRiskSetActionHead` 对显式 V19
fallback 和 candidates 做 permutation-equivariant set encoding，然后逐候选拼接 candidate、
fallback、difference、elementwise product 与 dense-quality evidence delta。带 bias 的 utility
head 直接回归部署 margin，辅助 benefit head 学正收益；两者零初始化，固定 `>0` 边界在第 0 步
精确保留 V19。V23 dense box/mask supervision 与逐 query adaptive source mixer 均保留，
source absence 仍由显式 validity mask 处理。

总 loss 现在明确包含 empirical setwise action risk、overestimate-weighted SmoothL1 utility、
empirical-prior benefit focal risk、dense absolute quality 和 listwise quality；utility regression
不会再被 dense 分支截断。生产 optimizer 合同为 69 tensors、`825,997` 参数，其中新 pairwise
head 为 30 tensors、`237,394` 参数。专属 identity/置换/正负梯度/迁移/optimizer 测试和全项目
回归均通过，最终为 `3023 passed, 2 warnings`。

128-row smoke run `1785799106` 得到 REC `0.5000/0.4453125`、Mask
`0.5000/0.40625/0.3504906`，V25 correction 为 0；审计通过 `1228/0/69`，optimizer 69 states、
`825,997` numel、step 10，所有 138 个 Adam moments finite/nonzero。smoke 只验证稳定性和
梯度覆盖，不作为正式指标。

正式 run `1785799635` 已从 protected V19 启动，完整使用 `36,665/9,508` train/validation，
两轮、`batch_size=12`、gate LR `3e-4`、temperature `0.25`、new-head-only。目录为
`output/source_moe_v25_pairwise_calibrated_train_v25/scanrefer/ssq_moe_e73_v25_pairwise_calibrated_e2/1785799635/`。
基于 V24 实测周期，epoch 1/2 receipts 预计在 `08:38-08:43 CST` / `09:42-09:47 CST`，
不做分钟级轮询。正式指标与清理结论待两轮完成后追加；protected V19 和历史
`0.582878/0.486012` 三组件均未改动。

#### V25 epoch 1 正式结果（2026-08-04）

epoch 1 训练 `3132.72s` 后，official REC 为
`5525/4420 = 0.5810896/0.4648717`，Mask 为
`5687/4671 = 0.5981279/0.4912705/0.4184664`。相对 protected V19 只少 `1/5` REC hits、
`1/1` Mask hits，尚未刷新 best。candidate oracle 为 `5987/5226`、mIoU `0.4516471`；新增
correction 仅 10 行，beneficial/harmful `2/8`、precision `20%`、oracle recall `0.18%`。
V25 已显著抑制 V24/dense ablation 的有害过切，但首轮仍处于 under-switch 端。

epoch-1 审计严格通过 `1228/0/69`、69 optimizer states、`825,997` numel、step `3055`，
所有 tensors 和 Adam moments finite/nonzero。epoch 2 已继续运行，最终 receipt 预计
`09:47-09:49 CST`，不做中间 batch 轮询。

#### V25 epoch 2、失败归因与清理（2026-08-04）

epoch 2 official REC 为 `5525/4421 = 0.5810896/0.4649769`；Mask 为
`5687/4670 = 0.5981279/0.4911653/0.4184430`。相对 protected V19 仍少 `1/4` REC hits、
`1/2` Mask hits，未刷新 best。candidate oracle 为 `5988/5228`、mIoU `0.4516748`，但新增
correction 为 0，oracle opportunity 为 `1086`。validation 的 candidate benefit target 约
`4.93%`，benefit predicted-positive 与 utility positive-candidate 均为 0，utility max-margin
mean `-1.3369`。完整第二轮因此确认 V25 是 calibrated all-fallback collapse，不是预训练、
optimizer、数值稳定或训练只跑一轮的问题。

epoch-2 checkpoint 审计通过 `1228/0/69`、69 states、`825,997` numel、step `6110`，全部
finite 且 moments 非零。formal 与 smoke 的低指标 `.pth` 已移入各自
`quarantine_low_metric/`；根目录仅保留 receipts。protected V19 inode `34391215` 仍为 8 links，
SHA-256 为 `2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`；历史
`0.582878/0.486012` 三组件未移动或删除。

下一版预注册为 prior-restored pairwise benefit：复用 V25 set encoder，但让 cost-aware benefit
margin 直接部署。训练 logits 使用 class balance 提升 rare positive 表示，同时用 train-batch
empirical prior log-odds 将 raw fixed-zero boundary 恢复到真实分布；continuous utility 仅作独立
辅助回归，candidate 排序采用不改变绝对边界的 listwise loss，不再叠加会整体推负的 empirical
fallback set risk。不使用 validation threshold，不继承 V25 权重，仍从 protected V19 初始化。

### V26 prior-restored pairwise correction（2026-08-04）

V26 已按上述预注册方案实现，固定 action/objective 为
`cascade_v26_prior_restored_pairwise_correction` /
`cascade_v26_prior_restored_pairwise_risk`。结构继续复用 V25 的 pairwise set encoder、逐 query
adaptive source mixer 和 dense box/mask quality head，共 69 个可训练 tensor、`825,997` 个参数；
部署决策改为 `benefit_margin > 0`，`utility_margin` 仅保留为连续效用辅助回归，不再控制固定零边界。

rare-benefit loss 使用 Beta(1,1) 平滑的 class-balanced BCE，并在训练 logit 上加入
`log((negative_count + 1) / (positive_count + 1))` 恢复经验先验；false-positive cost 仍由
`false_override_weight` 显式表达。candidate 内部采用 shift-invariant listwise ranking，dense
box/mask quality supervision 保持启用；empirical fallback set risk 只记录诊断，不再二次推动部署
margin 整体变负。所有新输出层保持零初始化，因此从 V19 初始化时第 0 步严格等价于 V19，且不需要
ScanRefer validation threshold。迁移策略 fail-closed：只接受完整 V19 初始化或 exact V26
checkpoint，明确拒绝 V24/V25 权重。

验证结果为 SourceMoE `136 passed`、integration `88 passed`、audit/reranker `36 passed`，全项目
`3033 passed, 3 warnings`，Python compile 与 launcher syntax 均通过。128-row GPU smoke run
`1785811045` 完成 10 个 optimizer steps，REC 为 `64/57 = 0.500000/0.4453125`，Mask 为
`64/52 = 0.500000/0.406250`、mIoU `0.3504906`；candidate oracle 为 `75/69`，部署 correction
为 0。checkpoint audit 通过 V19->V26 `1228/0/69`，optimizer 为 69 states、`825,997`
numel、step 10，模型 tensors 与 138 个 Adam moments 全部 finite/nonzero。smoke 的 7 个
checkpoint hardlink 已在审计后删除，config、log、metrics、diagnostics、retention 和
`v26_audit.json` 保留。

空间清理中，V24/V25 formal 与 smoke 已确认低于 protected V19 的隔离 `.pth` 被物理删除；
对应日志、正式指标、diagnostics、retention 与 audit 收据全部保留。protected V19 inode
`34391215` 仍为 8 links，SHA-256 仍为
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`；历史后处理
`0.582878/0.486012` 三组件未移动、未删除。

V26 正式两轮 run `1785811744` 已于 `2026-08-04 10:49 CST` 从 protected V19 启动，目录为
`output/source_moe_v26_prior_restored_train/scanrefer/ssq_moe_e73_v26_prior_restored_pairwise_e2/1785811744/`。
合同为完整 `36,665/9,508` train/validation、`batch_size=12`、gate LR `3e-4`、temperature
`0.25`、false-override weight `2.0`、new-head-only。参考最近完整 run，初始加载约 8 分钟，
训练约 52 分钟/epoch，验证约 13 分钟/epoch；epoch 1/2 receipt 预计在约 `12:01 CST` /
`13:06 CST` 检查，不做分钟级轮询。正式指标、audit 与最终 retention 结论待运行完成后追加。

#### V26 epoch 1 正式收据

epoch 1 训练耗时 `3627.11s`，official REC 为
`5522/9508 = 0.5807741/0.4646613`；Mask 为
`5684/9508 = 0.5978124/0.4911653`、mIoU `0.4183568`。相对 protected V19
`0.5811948/0.4653976` 与 `0.5982331/0.4913757/0.4186131` 均未提升。candidate oracle
为 `5981/5205`、mIoU `0.4505760`，oracle opportunity 为 `1064/9508 = 11.19%`，但
learned gate 仍为 0 次 correction；因此没有 false switch，但 oracle-switch recall 也是 0。

epoch-1 checkpoint audit 已通过 V19->V26 `1228/0/69`，optimizer 69 states、`825,997`
numel、step `3055`，138 个 Adam moments 全部 finite/nonzero。epoch 2 按预注册合同继续，
不因首轮未刷新 best 提前停止；本轮 epoch 2 的实际检查窗口改按 epoch 1 的 `3627s` 训练和
约 13 分钟验证重新估算。

#### V26 epoch 2、失败归因与清理

epoch 2 训练/验证已完整结束。official REC 为
`5516/9508 = 0.5801430/0.4637148`；Mask 为
`5676/9508 = 0.5969710/0.4903239`、mIoU `0.4177342`。相对 protected V19 仍少
`10/16` 个 REC hits、`12/10` 个 Mask hits，mIoU 少 `0.0008789`；相对本 run epoch 1
也全面下降。candidate oracle 为 `6664/5436`、mIoU `0.4768359`，oracle opportunity
增至 `1444/9508 = 15.19%`，但 learned gate 仍 0 次 correction、benefit target positive
约 `7.08%`、oracle-switch recall `0`，训练日志中的 deployed max margin mean 约 `-5.8224`。

这确认 V26 的 prior-restored benefit head 仍是 all-fallback collapse：候选覆盖和 oracle
headroom 明显存在，且没有 false switch，但 rare positive 没有穿过固定零边界。问题不再是预训练
加载、训练轮数、候选缺失或数值稳定；继续相同 V26 loss 没有依据，应回到 benefit prior/边界校准
与 source evidence 的独立诊断，而不是引入 ScanRefer 专用后处理阈值。

epoch-2 audit 通过 V19->V26 `1228/0/69`，optimizer 69 states、`825,997` numel、step
`6110`，模型和 138 个 Adam moments 全部 finite/nonzero。由于 epoch 1/2 均未超过 protected
global best，本次 run 的 8 个 `.pth`（epoch 1/2、latest 与 metric aliases）已物理删除；
两轮 metrics、diagnostics、config、log、retention 和 `v26_audit_epoch_{1,2}.json` 保留。
protected V19 inode `34391215` 仍为 8 links，SHA-256 未变；历史
`0.582878/0.486012` 后处理三组件也未移动或删除。

### V26 row-wise deployment-boundary calibration smoke（2026-08-04）

为验证 all-fallback 是否只是训练目标没有给固定零边界足够梯度，新增可选的
`_rowwise_boundary_calibration_loss`。它按 query 行计算 candidate margin 的 log-mean-exp：有
正 utility 的行被推向 `>0`，无正 utility 的行被推向 `<0`，并用 class-balanced row weight 与
false-positive cost。默认权重保持 `0.0`，只有显式设置
`SOURCE_MOE_GATE_BOUNDARY_LOSS_WEIGHT` 才启用，因此不改变既有 V26 合同。

128-row debug smoke 从 protected V19 各训练 10 steps。权重 `1.0` 的 boundary loss 为
`0.6764`，positive-row ratio `0.0985`，部署 correction 为 `0`，margin mean `-0.1673`；
权重 `4.0` 的 run 目录为
`output/source_moe_boundary_calibration_smoke_w4/scanrefer/ssq_moe_e73_v26_row_boundary_w4_smoke/1785824932/`，
boundary loss `0.6738`、positive-row ratio `0.0985`，validation predicted-positive ratio
仍为 `0`，max-margin mean `-0.1804`，REC `0.500000/0.4453125`，Mask
`0.500000/0.406250/0.3504906`。candidate oracle 仍为 `0.5859375/0.5390625`，说明候选有
headroom，但短 smoke 尚不足以把 rare positive 推过固定边界；因此没有启动正式 boundary-loss
训练，也不把 smoke 指标与完整 validation 比较。

权重 `4.0` 的 checkpoint 审计通过 V19->V26 `1228/0/69`，optimizer 69 states、`825,997`
numel、step `10`，模型和 Adam moments 全部 finite/nonzero。审计完成后该 smoke 的 7 个
`.pth` hardlink 已删除，仅保留 config、log、metrics、diagnostics、retention 和
`checkpoint_audit_epoch_1.json`；protected V19 inode、hash 以及历史 `0.582878/0.486012`
后处理组件均未移动或删除。当前最高仍是 postprocessed REC `0.582878/0.486012`，network-only
V19 REC `0.5811948/0.4653976`。

随后修正了 row-wise loss 的一个目标冲突：它原先再次使用 `false_override_weight=2.0` 给
fallback 行加权，导致 rare positive 行在零点附近仍受到整体向负侧的梯度。现在 boundary
calibration 只做正/负行均衡，false-positive cost 仅保留在 candidate benefit likelihood 中。
修正后的 weight `4.0` smoke（run `1785825891`）边界 loss `0.6964`，validation
`max-margin-mean=-0.1618`，相对未修正的 `-0.1804` 有改善，但 predicted-positive 和
correction 仍为 `0`，REC/Mask 仍为 `0.500000/0.4453125`、`0.500000/0.406250/0.3504906`。
审计通过后该 run 的 7 个低指标 `.pth` 也已删除；一次只因 DATA_ROOT 缺少末尾斜杠而失败的空
启动 run 仅保留 config/log，没有生成权重。修正后的完整回归为 `3035 passed, 3 warnings`，
因此暂不启动正式 boundary-loss 训练。

为区分“10 步太短”和“目标本身不可靠”，又做了 5-epoch/128-row 延长 smoke（50 optimizer
steps，run `1785826279`）。epoch 2 首次出现 `0.19%` positive candidate、oracle recall
`4.55%`；epoch 5 为 predicted-positive `2.18%`、oracle recall `9.09%`、precision
`4.55%`、false-switch `31.82%`，debug REC `0.500000/0.453125`，max-margin mean
`-0.5826`。边界校准确实能释放少量 correction，但大部分是 harmful，不能作为正式 ScanRefer
训练配置。epoch-5 audit 通过 V19->V26 `1228/0/69`、step `50`，随后该 run 的全部低指标
`.pth` 已删除，五轮 receipts 保留。

### V27 前置可分性检查与低指标权重清理（2026-08-04）

为决定下一版是否继续复用现有 QueryReranker 的 dense quality head，对只读缓存
`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/train` 做了 train-only
scene-disjoint 检查：固定每 5 个场景抽 1 个场景作为 holdout，共 `29,349` rows、`469,584`
个有效候选；缓存绑定的 backbone SHA 是旧 epoch-71 的
`3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`，因此本结果只用于
结构可分性判断，不能直接作为 V19 发布 artifact 的训练输入。

候选标签正例比例为 `IoU>0.25: 0.8613`、`IoU>0.50: 0.7376`。不使用 validation 阈值调参时，
单特征 AUROC 如下：

| 特征 | `IoU>0.25` | `IoU>0.50` |
| --- | ---: | ---: |
| `mask_text_query_dice` | 0.8383 | 0.6990 |
| `mask_foreground_ratio` | 0.6657 | 0.7080 |
| `rank_default` | 0.6519 | 0.6080 |
| `contrastive_top_score` | 0.6556 | 0.6119 |
| `query_objectness` | 0.6254 | 0.6145 |

`mask_text_query_dice` 与候选 IoU 的 Pearson 相关为 `0.6262`，`score_default` 仅 `0.1255`，
`score_contrastive` 为 `0.1619`。这支持 V27 继续使用 dense absolute quality、阈值头和
query set rerank，但部署动作必须单独做训练分布风险校准；不能把 V26 的 all-fallback 解释为
预训练未加载或 epoch 不足，也不能用 validation 后处理阈值补救。

本次审计并清理的低指标目录：
`output/source_moe_boundary_calibration_smoke/scanrefer/ssq_moe_e73_v26_row_boundary_smoke/1785824561/`。
其中 7 个 checkpoint 原来全部硬链接到 inode `8632056461`，指标为 REC
`0.500000/0.4453125`、Mask `0.500000/0.406250/0.3504906`；已物理删除这些 `.pth`，并保留
`checkpoint_retention.json`、`config.json`、`log.txt`、`eval_metrics_epoch_1.json` 和
`source_choice_diagnostics_epoch_1.json`。受保护 V19 inode `34391215`（SHA-256
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dbcbb6e55ececbe`）以及历史系统级
`0.582878/0.486012` 三组件未触碰。

随后用受保护 V19 做 fresh-runtime cache smoke：
`output/rec_reranker/v19_fresh_runtime_smoke/train/`，`--limit 128`、`batch=12`、
`num_workers=4`、`max_candidates=16`。完整数据集初始化耗时约 10 分钟，最终 manifest
正确绑定 V19 SHA、`source_moe_gate_action_mode=cascade_opportunity_verified_correction`、
三源 schema 和 `feature_dim=152`；生成 1 个 `shard_000000.pt`，无残留训练进程。该 smoke
的 default `Acc@0.25/0.50=0.96875/0.72656`，candidate oracle `1.00000/0.96094`，只用于
验证当前 runtime 与受保护 checkpoint 的候选身份一致，不作为 validation 或发布指标。

### V27 uncertainty-aware dense quality 实现与 smoke 结论（2026-08-04）

V27 已实现 `cascade_v27_uncertainty_quality_risk`，继续使用 V23 的逐 query adaptive source
mixer 和 dense box/mask quality set head。quality Bernoulli variance 用作 query uncertainty，
部署质量为 `predicted_quality - uncertainty_weight * uncertainty`，本轮显式设置 uncertainty
weight `0.5`。所有有效 query 在零初始化时 uncertainty 相同，因此 risk margin 为 0，加载
protected V19 后第 0 步保持 identity；旧 V19 缺少 uncertainty 配置时也保持兼容。action
regression 使用 cost-aware `decision_utility` 的固定零边界，连续 box/mask quality 只用于候选
排序。cache/reranker provenance 和 checkpoint audit 都已纳入 V27 uncertainty 合同。

专项 compile、SourceMoE、integration 与 audit 回归通过，定向测试为 `241 passed`，更广的
SourceMoE/integration/reranker/cache 回归此前为 `265 passed`。V27 audit profile 的合同为
common/changed/new `1228/0/39`、39 optimizer states、`588,603` trainable numel。一次首 batch
前因 loss API wiring 失败的 run `1785832913` 未产生 checkpoint，修复后未再复现。

三次 128-row/10-step smoke 均显示明显过切。原始连续质量边界 run `1785833267` 为 29 次
switch、beneficial/harmful `2/27`、precision `6.90%`、false-switch `93.10%`；REC
`0.500000/0.4453125`，Mask `0.500000/0.40625/0.350079`。cost-aware candidate regression
run `1785833888` 为 25 次 switch、`2/23`、precision `8.00%`、false-switch `92.00%`、oracle
recall `15.38%`，REC/Mask 为 `0.500000/0.4453125`、
`0.500000/0.40625/0.350045`。row-max boundary ablation run `1785834292` 进一步恶化到 42 次
switch、`2/40`、precision `4.76%`、false-switch `95.24%`，REC
`0.492188/0.437500`、Mask `0.492188/0.398438/0.342377`；该 loss 已从代码回退。三组
checkpoint 审计后共删除 21 个 `.pth` hardlink，receipts 均保留。

为排除 10 steps 不足，又完整运行 5 epochs/128 rows、每轮 10 steps 的延长 smoke，run
`1785834751`。各轮 REC 为 `0.492188/0.437500`、`0.500000/0.4453125`、
`0.500000/0.4453125`、`0.5078125/0.4609375`、`0.492188/0.4453125`；对应 Mask 最好出现在
epoch 4，为 `0.5078125/0.4140625/0.3581983`。switch precision 五轮依次为
`5.56%/8.70%/6.25%/15.79%/10.00%`，false-switch 为
`94.44%/91.30%/93.75%/84.21%/90.00%`，oracle recall 为
`15.38%/15.38%/6.67%/20.00%/13.33%`。因此延长训练虽在 epoch 4 有短暂改善，仍未学到可用
的 abstention 边界，不能进入 9,508-row formal，更不能据此启动 70--80 epoch 完整训练。

epoch-5 checkpoint audit 通过：common/changed/new `1228/0/39`、39 states、`588,603`
numel、step `50`，模型与 78 个 Adam moments 全部 finite/nonzero。该 run 的 8 个低指标
`.pth`（epoch 4 的 6 个 metric aliases/epoch link，以及 epoch 5/latest 的 2 个链接）已按明确
文件名物理删除；五轮 metrics、diagnostics、config、log、retention 和
`v27_audit_epoch_5.json` 保留。protected V19 仍为 inode `34391215`、8 links、mode `0444`，
SHA-256 仍为 `2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。

当前最高没有变化：系统后处理 REC `0.582878/0.486012`，network-only V19 REC
`0.5811948/0.4653976`，V19 Mask `0.5982331/0.4913757/0.4186131`。V27 的下一结构方向应保留
dense quality 与 adaptive source mixer，但部署不能直接使用 absolute-quality difference；需要
独立、零初始化、按训练分布校准的 abstention/risk head，在固定部署规则下同时约束 switch
precision 与 oracle recall，避免 V26 all-fallback 和 V27 overcut 两个极端。

### V28 selected-candidate abstention：实现与 smoke 结论（2026-08-04）

V28 已按 V27 结论拆开“选哪个候选”和“是否离开 V19 fallback”两个职责。action/objective
固定为 `cascade_v28_selected_abstention_correction` /
`cascade_v28_selected_abstention_risk`：pairwise head 只对候选内部排序，独立 row abstention
head 只判断当前选中候选能否覆盖 fallback；部署时每行 candidate margin 的最大值严格等于
row risk，边界固定为 `>0`。row head 输出层零初始化，因此从 protected V19 加载后的第 0 步
严格保持 V19 identity，不搜索 ScanRefer validation threshold，也不固定最强单源或源组合。

V28 代码合同覆盖 `models/source_moe.py`、`models/losses.py`、`models/mcln.py`、
`main_utils.py`、训练 launcher、checkpoint audit 和离线 reranker/cache provenance。相对 V19
的 common/changed/new 为 `1228/0/75`；只训练 75 个新增 parameter tensors，共 `876,174`
个参数。专项测试在 smoke 前为 `245 passed`，Python compile 和 launcher shell syntax 均通过。

首次启动 run `1785836663` 因外层 loss objective 白名单遗漏，在首 batch 和 optimizer step
之前 fail closed；只生成 config/log，没有 checkpoint。补齐白名单后的 10-step run
`1785836950` 完成，debug REC 为 `0.500000/0.453125`，Mask 为
`0.500000/0.406250/0.3504906`，correction 为 0。epoch-1 audit 通过
`1228/0/75`、optimizer step `10`，证明预训练已正确加载且新增 head 实际进入 optimizer。

为排除 10 steps 太短，又完整运行 5 epochs/128 rows、每轮 10 steps，run `1785837268`。
REC 五轮依次为 `0.500000/0.4453125`、`0.500000/0.4453125`、
`0.500000/0.4453125`、`0.500000/0.453125`、`0.500000/0.453125`；Mask 五轮均为
`0.500000/0.406250/0.3504906`。epoch 1--3 不切换，epoch 4--5 各释放 1 个 beneficial、
0 个 harmful switch，precision `100%`、false-switch `0%`，但 oracle recall 仅
`1/13=7.69%`。candidate oracle 为 `75/69=0.58594/0.53906`，说明 V28 已消除 V27 的
overcut，但仍明显 undercut；row head 只能对 pairwise head 当前选中的候选作 abstention，候选
排序没有稳定选中正收益 query。

epoch-5 checkpoint audit 已通过：common/changed/new `1228/0/75`、75 optimizer states、
`876,174` numel、step `50`，模型与 150 个 Adam moment tensors 全部 finite/nonzero。
两个成功 smoke run 的 16 个低指标 `.pth` 链接均已物理删除，只保留 config、log、metrics、
diagnostics、retention 和 `v28_audit_epoch_1/5.json`；失败启动本来就没有权重。删除后 protected
V19 仍为 inode `34391215`、8 links、mode `0444`，SHA-256 仍为
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。

因此当前没有启动 V28 的 70--80 epoch/full-data 训练。这里的停止依据不是训练轮数猜测，而是
结构诊断：selected abstention 的 positive target 约从 `1.5%` 升到 `3.8%`，仍远低于 row
oracle `10.16%`，完整长训会把主要算力投入一个受 candidate selection 限制的目标。下一版应
联合提升 candidate selection 与 row risk，但继续保留 V28 的独立固定边界、V19 identity、
source-validity fail-closed 和零 harmful-switch 约束；通过同样 5-epoch smoke 门禁后再进入正式
训练。当前最高仍为系统后处理 REC `0.582878/0.486012`，network-only V19 REC
`0.5811948/0.4653976`，V19 Mask `0.5982331/0.4913757/0.4186131`。

文档更新后的完整定向回归覆盖 SourceMoE、CLI/迁移、checkpoint audit、candidate adapter、
evaluator、reranker 与 cache，共 `339 passed in 19.73s`；相关 Python compile 和 launcher
`bash -n` 同时通过。

在上述 smoke 风险判断保留的前提下，按“先做一次完整训练”的要求，已于 `2026-08-04
18:11 CST` 启动 V28 两轮 full-data 验证 run `1785838268`，tmux 为
`mcln_v28_selected_abstention_e1_e2`，目录为
`output/source_moe_v28_selected_abstention_train_v28/scanrefer/ssq_moe_e73_v28_selected_abstention_e2/1785838268/`。
配置已确认非 debug、expected validation `9,508`、batch `12`、V19 initializer、75 个新增
head tensors only、gate LR `3e-4`、固定零边界和五指标 retention。V19 checkpoint 自身是建立
在 epoch-71 主干上的局部 epoch-2 V19 artifact，因此本 run 的局部 epoch 1/2 对应继续推进
全局约 e74/e75，不代表只从头训练两轮；预计每轮 `3,055` optimizer steps，完整两轮为
`6,110` steps，是 50-step smoke 的 122 倍。

依据 V25/V26 同硬件实测，epoch 1 receipt 预计在 `19:10--19:20 CST`，epoch 2 在
`20:15--20:30 CST`。只在预计完成窗口轮询。正式 run 的作用是验证大量 full-data steps 能否
提升 V28 undercut recall；它不撤销 precision/Mask/global-best 门禁，未刷新指标的权重仍将在
最终 checkpoint audit 后清理。

等待正式结果期间完成了 V28 supervision 分解，不读取运行中 batch 日志。epoch-5 smoke 的
13 个 row-oracle opportunities 中，hard candidate policy 约只捕获 5 个正收益 query（约
`38.5%`），independent abstention 再放行 1 个（条件 recall 约 `20%`），所以最终为
`1/13=7.69%`。若 full-data formal 仍未刷新，下一版不简单增加 boundary 权重，而按设计文档
35.2 节实现 positive-mass candidate policy 与 row-prior-calibrated counterfactual selected risk；
当前 V28 训练代码和进程不受该预注册分流影响。

同时补充了三个不参与训练的 V28 分层统计：selected-positive count、candidate policy
opportunity capture 和 abstention conditional recall。代码没有新增参数或改变 action/loss，
专项测试 `2 passed`；完整定向回归为 `339 passed in 6.56s`。这些字段将在 formal checkpoint
复评时使用，当前已加载模型的训练进程不受影响。

### 35.3 V29 counterfactual selected-risk wiring（待 V28 formal 结果后 smoke）

V29 已按上述预注册合同接入代码，但尚未启动正式训练，也不影响运行中的 V28。
action/objective 固定为 `cascade_v29_counterfactual_selected_correction` /
`cascade_v29_counterfactual_selected_risk`。新增 `CounterfactualSelectedRiskHead` 对所有有效
候选共享计算 risk，部署时仍只 gather candidate policy 的 hard top-1；candidate margin 行最大值
严格等于 selected risk，输出层零初始化。loss 增加 positive-candidate probability-mass 监督和
row-balanced positive/hard-negative counterfactual risk，保持 V19 fallback、固定 `>0` 边界和
跨数据集 fail-closed 合同。

V29 的 V19 migration、new-head-only optimizer、launcher、reranker provenance、checkpoint audit
profile（`1228/0/75`、75 states、`876,174` numel）及 query-permutation/zero-init 测试均已接通；
定向回归 `4 passed`，Python compile 与 launcher `bash -n` 通过。正式 V29 必须先完成短 smoke，
并与 V28 formal 的 REC、Mask、switch precision/recall 及保护权重 hash 一起复评。

### 35.4 V28 full-data formal 结果与清理（2026-08-04）

正式 run `1785838268` 已完成两轮 full-data gate-only 训练：训练集 `36,665`、验证集
`9,508`、每轮 `3,055` steps，总计 `6,110` steps；V19 initializer 和 75 个 new-head-only
optimizer tensors 均经审计通过。epoch 1 指标为 REC `0.5810896/0.4652924`、Mask
`0.5981279/0.4914809/0.4185229`；epoch 2 降为 REC `0.5805637/0.4647665`、Mask
`0.5974968/0.4909550/0.4181306`。两轮都 `beneficial_switch=0`、`harmful_switch=0`、
`oracle_switch_recall=0`，所以没有达到 `0.59/0.49`，也不支持“只是训练轮数不足”的判断。

epoch 1/2 checkpoint audit 分别通过 `1228/0/75`、75 optimizer states、`876,174` 参数、
step `3055/6110`，所有模型张量和 Adam moments finite/nonzero。按 global-best retention，
删除 epoch 2 及 epoch 1 的低指标 alias，只保留
`ckpt_best_mask_acc050.pth`（Mask Acc@0.50 `0.4914809`，相对 protected V19 的
`0.4913757` 有微小提升）；删除共 7 个 hardlink，metrics/diagnostics/audit/log/config 均保留。
protected V19 复核仍为 inode `34391215`、8 links、mode `0444`、SHA-256
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。V29 仍按预注册设计等待
短 smoke，不直接复用 V28 的低 recall 权重。

### 35.5 V29 counterfactual-risk smokes and raw-boundary calibration（2026-08-04）

V29 短 smoke `1785847541`（5 epochs、128 validation rows、50 optimizer steps）把 candidate
policy opportunity capture 提升到约 `7/12=58.3%`，但 selected-risk head 最好仍只释放 1 个
beneficial switch，conditional abstention recall 约 `8--9%`；REC 最高仅
`0.500000/0.460938`，未进入 formal。仅设置
`SOURCE_MOE_GATE_FALSE_OVERRIDE_WEIGHT=1.0` 的 cost=1 消融 `1785848144` 也只达到 1 个
beneficial、0 个 harmful switch（epoch 4 precision `100%`、recall `7.14%`），epoch-5 REC
为 `0.492188/0.453125`，因此 false-positive cost 不是主要 undercut 原因。

cost=1 epoch-5 checkpoint 已通过 V29 精确审计（`1228/0/75`、75 optimizer states、
`876,174` trainable parameters、step `50`），随后该 run 的 9 个临时 `.pth` hardlink 全部
删除，只保留 metrics、diagnostics、config、log、retention receipt 和 audit JSON。

代码复核确认 V29 训练 BCE 使用了 `candidate_risk + prior_shift`，而推理仍以 raw selected
risk `>0` 为固定边界，导致正候选可能在训练上被视为正、部署时却仍回退。现在 risk loss 在
保留 row-prior-balanced BCE 的同时，加入同一正样本/hard-negative 集合上的 unshifted raw-risk
BCE，并对两者等权平均；不引入 ScanRefer 专属阈值。新增单测验证 raw positive/negative risk
梯度方向，已通过。下一步必须先做短 calibration smoke，再决定是否 full-data 训练。

当前最高指标未变化：后处理 REC `0.582878/0.486012`，network-only V19 REC
`0.5811948/0.4653976`，V19 Mask `0.5982331/0.4913757/0.4186131`。protected V19 inode
`34391215`、mode `0444`、SHA-256 `2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`
以及三份 `0.582878/0.486012` 后处理组件均保持不变。

### 35.6 V30 raw-risk boundary smoke 结果与清理（2026-08-04）

V29 的 raw-boundary BCE 修正后，V30 smoke `1785850647` 使用 protected V19、128-row debug
split、5 epochs/50 steps 和固定部署边界 `risk>0`。第 1--3 轮 REC 均为
`0.500000/0.453125`；第 4--5 轮提升为 `0.500000/0.460938`，Mask 始终为
`0.500000/0.406250/0.3506068`。第 4--5 轮各有 1 个 beneficial、0 个 harmful switch，
precision `100%`、oracle/conditional recall `8.33%`；第 5 轮 raw positive risk 比例约
`0.76%`。按整轮 switch 计数，precision 仍为 `1/1=100%`、recall 仅 `8.33%`；训练日志中
按 row 平均的 predicted-positive ratio 约 `0.76%`，说明 raw boundary 已能释放少量候选，却
还不能稳定提高机会覆盖。

该 run 的 epoch-5 checkpoint 审计通过 V29 合同（`1228/0/75`、75 states、`876,174` 参数、
step `50`），随后 9 个临时 `.pth` hardlink 全部删除，metrics、diagnostics、config、log、
retention 和 `audit_epoch_5.json` 保留。一次先行启动因忘记把 debug expected sample count 设为
128，在评估写 diagnostics 时被 `128 != 9508` fail-closed；其 epoch-1 权重也已审计（step `10`）
并清理，不影响正式结果。

V30 没有达到 formal gate，因此不启动 9,508-row 或 70--80 epoch 训练；当前最高仍为后处理
REC `0.582878/0.486012`、network-only V19 REC `0.5811948/0.4653976`。下一步应优先改进
candidate policy 的正收益 query 选择/排序，避免继续只把风险头推过零边界；该方向仍保持跨
ScanRefer、Nr3D、Sr3D 的自适应 source MoE 和固定部署规则。

### 35.7 V31--V33 hard-policy 与 raw-risk 目标清理（2026-08-04）

V31 `1785851559` 在 positive probability-mass 之外加入 hard top-1 positive-vs-negative
margin，使部署实际选择的 query 直接参与排序监督。epoch 5 的整轮 diagnostics 捕获
`9/14=64.29%` opportunity，但 risk 只放行 `1/14=7.14%`；1 beneficial、0 harmful，REC
`0.500000/0.453125`，Mask `0.500000/0.406250/0.3506068`。V32 `1785852121` 将真正参与优化的
counterfactual 分类完全改为 raw-zero BCE，prior-shift BCE 只保留为 diagnostics；捕获
`9/13=69.23%`、recall `1/13=7.69%`，REC 仍为 `0.500000/0.453125`。V33 `1785852709`
进一步移除 V29 objective 中重复叠加的 selected-row prior-shifted BCE，结果仍为捕获
`9/13=69.23%`、recall `1/13=7.69%`、1 beneficial、0 harmful，REC
`0.500000/0.453125`。

三次 epoch-5 checkpoint 都通过 V29 审计（`1228/0/75`、75 states、`876,174` 参数、step
`50`），所有临时 `.pth` 均已删除。复核发现 counterfactual risk 虽对所有候选前向计算，正类
训练实际只抽取每行 oracle-best 一个候选；policy 可选中同一行另一个正收益候选，但该候选的
risk 没有正类监督。这解释了 candidate opportunity capture 已到约 `69%` 而部署 recall 仍只有
约 `8%`。当前实现以 raw-zero 分类作为唯一优化目标，prior-shift 只用于诊断，历史 35.5 节中
“两种 BCE 等权”的描述仅对应 V30 之前的中间状态。

### 35.8 V34 dense-positive counterfactual risk（2026-08-04）

V34 将每个 positive row 的全部正收益候选都纳入 raw-zero BCE 和 utility regression，先在行内
平均再跨行聚合；负类仍只取 policy-hardest negative。该改动不增加参数，不改变 `risk>0`
部署边界、V19 step-0 identity、query permutation、source validity 或 fail-closed 合同。新增
测试同时验证同一 row 的多个正候选都收到正向越界的 BCE/回归梯度，聚焦 SourceMoE、integration
和 audit 回归为 `253 passed`。

5-epoch/128-row smoke `1785853677` 从 protected V19 启动。epoch 5 整轮 diagnostics 为
candidate capture `9/15=60.00%`、conditional recall `1/15=6.67%`、1 beneficial、0 harmful、
precision `100%`；日志的 batch-weighted 对应统计为 `56.82%/9.09%`。REC 为
`0.500000/0.453125`，Mask 为 `0.500000/0.406250/0.3504906`。相对 V33，dense supervision
没有让 policy capture 与 risk recall 同时改善，因此不进入 full-data/70--80 epoch 训练。

V34 epoch-5 checkpoint 审计通过（`1228/0/75`、75 states、`876,174` 参数、step `50`），随后
9 个临时 `.pth` 全部删除，run 目录只保留约 `140K` 的 audit、metrics、diagnostics、config、
log 和 retention receipt。当前最高仍为后处理 REC `0.582878/0.486012`、network-only V19 REC
`0.5811948/0.4653976`、V19 Mask `0.5982331/0.4913757/0.4186131`。protected V19 仍为 inode
`34391215`、mode `0444`、SHA-256
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`，三份后处理组件也未改动。

### 35.9 V35 utility-cost-once smoke（2026-08-04）

V35 修正 dense counterfactual objective 中的 cost 重复计入：break/false-positive 代价已经编码在
`decision_utility`，raw-zero BCE 和 utility regression 不再额外乘同一代价；正负 regression
先各自求均值再等权聚合，避免负例数量主导风险尺度。prior-shift loss 仍只作诊断，部署仍使用
raw selected risk `>0`，参数量、V19 identity、query permutation、source validity 和
fail-closed 合同均不变。

5-epoch/128-row smoke `1785854661` 的 epoch-5 整轮 diagnostics 显示 candidate capture
`8/13=61.54%`；模型预测 6 次 switch，其中 2 beneficial、4 harmful，recall
`2/13=15.38%`、precision `33.33%`、false-switch rate `66.67%`。REC 为
`0.500000/0.453125`，Mask 为 `0.500000/0.406250/0.3518690`。V35 虽把 recall 从单次放行提高到
2 次，但代价是不可接受的 harmful switch，因此不进入 full-data/70--80 epoch 训练。

epoch-5 checkpoint 通过 V29 精确审计（`1228/0/75`、75 states、`876,174` 参数、step `50`），
该 run 的临时 `.pth` 已全部清理，audit、metrics、diagnostics、config、log 和 retention receipt
保留。

### 35.10 V36 symmetric deployment-gap smoke（2026-08-04）

V36 在固定 raw-zero 部署边界两侧加入对称安全间隔，但不移动推理阈值：正类 BCE 使用
`risk - temperature`，负类 BCE 使用 `risk + temperature`，促使安全候选和危险候选以相同压力
远离零点。dense-positive 行内平均、policy-hardest negative、utility-cost-once 和正负 regression
等权聚合均保留。新增对称间隔和多正候选梯度合同后，聚焦回归为 `255 passed`。

5-epoch/128-row smoke `1785855327` 的 epoch-5 整轮 diagnostics 为 candidate capture
`8/13=61.54%`、最终 recall `1/13=7.69%`、1 beneficial、0 harmful、precision `100%`；训练日志中
batch-weighted 的 `48.48%/9.09%` 只用于在线观察，不替代整轮计数。REC 为
`0.500000/0.453125`，Mask 为 `0.500000/0.406250/0.3504906`。对称间隔恢复了 precision，却把
V35 的 2 次有效放行退回 1 次，仍未解决 risk undercut，因此不启动 full-data 或 70--80 epoch。

V36 epoch-5 checkpoint 已通过审计：common/changed/new 为 `1228/0/75`、75 optimizer states、
`876,174` trainable parameters、step `50`，全部模型张量和 Adam moments finite/nonzero。随后严格
删除 run 目录中实际存在的 8 个低指标 `.pth`，保留 `audit_epoch_5.json`、五轮 metrics/
diagnostics、config、log 和 retention receipt。protected V19 复核仍为 inode `34391215`、
8 links、mode `0444`、SHA-256
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。

V35/V36 共同说明当前矛盾不是简单训练轮数不足：降低重复 cost 可以提高 release recall，但会放大
harmful switches；增加对称 gap 又会回到过度保守。下一结构应让校准依据候选证据强度/不确定性
自适应，而不是继续修改 ScanRefer 专属阈值或固定 source 组合。当前最高保持不变：后处理 REC
`0.582878/0.486012`，network-only V19 REC `0.5811948/0.4653976`，V19 Mask
`0.5982331/0.4913757/0.4186131`。

### 35.11 V37--V38 双头风险分解结果（2026-08-05）

V37 将单一 risk 拆为 candidate benefit/hazard 两个证据头，部署规则为
`ReLU(benefit) - ReLU(hazard) > 0`；正候选主要学习 benefit，hard negative 主要学习 hazard，
输出层零初始化以严格保持 V19 step-0 identity。5-epoch/128-row smoke `1785857843` 的 epoch 5
REC 为 `0.500000/0.4453125`，Mask 为 `0.500000/0.406250/0.3506068`，candidate capture
`8/13=61.54%`，但五轮均为 0 beneficial、0 harmful、0 switch。V37 解决了 V35 的危险过切，
却完全没有释放能力。

V38 将 benefit/hazard 改为互补 log-odds，部署使用 `benefit - hazard > 0`，分类采用对称 focal，
净风险回归目标归一化为 `+1/-1`，避免 utility 绝对尺度直接决定边界。smoke `1785859359` 的
epoch 5 REC 为 `0.500000/0.4609375`，Mask 为 `0.500000/0.406250/0.3504906`，capture
`7/12=58.33%`，只释放 1 个 beneficial、0 harmful，recall `8.33%`、precision `100%`。
最终双头 bias 为 `-0.0041518/+0.0041518`，权重范数相同且符号近乎相反，实际退化成更保守的
单标量，而没有形成可解释的互补证据。

V37/V38 epoch-5 checkpoint 均通过精确审计：common/changed/new `1228/0/75`、75 optimizer
states、`876,303` trainable parameters、step `50`，模型和 Adam moments finite/nonzero。
V37 的 8 个、V38 的 9 个低指标 `.pth` 均已删除，只保留 audit、metrics、diagnostics、config、
log 和 retention。两者均未进入 full-data 或 70--80 epoch 训练。

### 35.12 V39 gain 主路 + hazard residual veto（2026-08-05）

V39 回到 V35 可释放的 raw gain 主路，同时把 hazard 限制为非负 residual veto：部署固定为
`candidate_gain - ReLU(candidate_hazard) > 0`。gain 使用 raw-zero BCE 与 utility regression；
hazard 使用 focal 分类，仅学习压制危险候选；break cost 只保留在 `decision_utility`，不重复乘权。
该设计仍无 ScanRefer 专属阈值，保持 V19 零初始化 identity、query permutation、source-validity
fail-closed 和统一零边界。action/objective 为 `cascade_v39_hazard_residual_correction` /
`cascade_v39_hazard_residual_risk`。

实现已完整接入模型 forward/output、loss、MCLN、CLI、launcher、reranker provenance、V19 migration、
new-head-only optimizer 和 checkpoint audit。专项 SourceMoE/integration/audit 回归为 `273 passed`；
全项目为 `3067 passed`，另有 3 个既有 geometry-cache provenance 测试因测试 backbone config 与
缓存 manifest 不一致而失败，未涉及本次修改文件。Python compile 与 launcher `bash -n` 通过。

protected-V19 smoke `1785861677` 完成 5 epochs/128 rows、50 optimizer steps。epoch 1--4 REC
均为 `0.500000/0.4453125`，Mask 为 `0.500000/0.406250/0.3504906`，且没有 correction switch；
epoch 5 只释放 1 个 beneficial、0 harmful，oracle recall `1/13=7.69%`、precision `100%`，REC
为 `0.4921875/0.4453125`，Mask 降为 `0.4921875/0.3984375/0.3426781`。candidate capture
为 `9/13=69.23%`，但部署仍明显 under-release。最终 gain/hazard bias 为
`+0.0004505/+0.0053369`，权重范数为 `0.01451/0.04859`，hazard 分支约为 gain 的 3.35 倍，
说明 residual veto 仍主导边界。

epoch-5 审计通过 `1228/0/75`、75 states、`876,303` 参数、step `50`，全部 finite/nonzero。
由于 beneficial switch 没有超过 V36/V38 的 1 次，V39 不满足正式训练门禁，没有启动 full-data
或 70--80 epoch；run 中 8 个 `.pth` 已删除，JSON/log/audit 保留。protected V19 再次核验为 inode
`34391215`、8 links、mode `0444`、SHA-256
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。

当前最高指标仍为后处理 REC `0.582878/0.486012`；network-only 最高仍为 V19 REC
`0.5811948/0.4653976`，对应 Mask `0.5982331/0.4913757/0.4186131`。V37--V39 表明问题不是
“未加载预训练”或“只训练几轮”：每次均从受保护 V19 完整迁移且 optimizer 实际更新。下一版应
直接校准 gain 与 hazard 的相对学习速度或共享证据尺度，并继续以短 smoke 的 beneficial/harmful
整轮计数作为正式长训门禁。

### 35.13 V19 后处理快速复测、V40 全特征证据与分割联合目标（2026-08-05）

按用户要求，V19 + parent/geometry 后处理只做一次快速指标摸底，完成后立即回到网络内 V40
主线。旧 parent/geometry artifact 明确绑定 epoch-71 SHA，不能直接作为 V19 正式结果；因此已用
受保护 V19 生成完整 train/val candidate cache，并为 geometry extractor 新增显式
`--portable-provenance`。该模式要求 checkpoint SHA/epoch、完整 train/val cache、train-only audit
panel 及 audit train-cache path 全部一致；默认模式仍严格锁定原 epoch-71 保护输入。同期修复
geometry provenance 漏记 `source_moe_gate_uncertainty_weight` 的既有问题，专项回归为
`28 passed`。V19 复测结束前不得宣称旧 `0.582878/0.486012` 是 V19 组合指标。

V40 的完整 152 维 `hierarchical_pair_evidential` train-only scene-disjoint probe 已完成：fit/holdout
为 `29,349/7,316` rows，candidate-oracle positive rows 为 `731`。普通期望策略为
`45 beneficial / 36 harmful`；`0.25 sigma` 下界为 `13/9`；`0.50 sigma` 只有
`1 beneficial / 0 harmful`。相较 24 标量特征的 `12/9`，projection 特征明显增加了可分性，
但仍未达到“beneficial > 1 且 harmful = 0”的正式训练门禁。三组 break-cost/hidden-dim
train-only 复核正在四卡并行，不访问 validation。

后续五折 `signed_utility_evidence` 已覆盖全部 `scene_modulus=5` holdout，仍只使用 train。
train-calibration conformal 95% 五折合计释放 `592 beneficial / 794 harmful`；99% 为
`193/197`；100% 才收缩到 `2/1`，其中三折没有 beneficial，仍是 coverage collapse。
固定 `1.5 scale` 也只有 `141/139`，不能提供稳定安全边界。因此该 signed-utility 结构与前述
hierarchical evidence 一并否决，不接入 V40 网络，也不继续搜索 validation 阈值。

V19 train-only geometry audit 已完成 `256` expressions：默认候选为
`0.74219/0.48047`，regressed Top-16 oracle 为 `0.97656/0.88672`，七种 geometry 联合 oracle
为 `0.98047/0.92578`。首次以 `batch=36/workers=4/shard=252` 启动 portable 提取时暴露两个
深层合同遗漏：cache manifest 仍硬编码 `12/2/252`，且 batch 改变后的 top-k tie 漂移触发 fresh
query identity parity。现已把通用 manifest 合同改为严格类型、正值及 shard/batch 整除；默认
非 portable CLI 仍固定 `12/2/252`。portable parity 则以完整 base cache 的 query、box 和 default
Top-1 为规范身份，再从当前 checkpoint 对应 query 提取 geometry，避免 box/mask query 错配。
专项 cache/durability 回归为 `175 passed`。完整 V19 geometry train/val 已分别在 GPU0/GPU1
按 `36/4/252` 重启；该项仍只是一次 V19 后处理摸底，完成后回到网络内联合证据主线。

分割目标同步提升为联合硬门槛：REC `Acc@0.25 >= 0.59`、`Acc@0.50 >= 0.49`，Mask
`Acc@0.50 >= 0.5070`、semantic mIoU `>= 0.4472`，并保持 Mask@0.25 不低于 V19 的
`0.5982331`。现有 `JointBoxMaskAdapter` 已具备 box tier、mask tier、连续 mask IoU 和
query-specific text/query logit calibration，但旧 v1 在 train calibration 上五项均退化，不能直接
复用其 selector。下一版复用其标签/cache 链路，改为同一 shared evidence trunk 的
query-consistent 多任务质量选择：box 与 mask 始终来自同一个 parent query，box 两档的风险下界
负责 veto，mask 两档与连续 IoU 负责在安全候选中排序；禁用路径必须逐位恢复 V19。部署不能使用
ScanRefer validation 阈值，门禁和风险强度只由 scene-disjoint train calibration 确定，以便迁移到
单阶段 ScanRefer、Nr3D 和 Sr3D。

### 35.14 四卡 geometry 吞吐优化与自动 DDP 启动（2026-08-05）

完整 geometry 提取的低 GPU 利用率不是显存不足：模型前向阶段 SM 可达 `88--100%`，但旧
mask-to-AABB 实现会对每个样本的 16 个 query 逐个发起 quantile kernel 并执行同步有效性判断，
几何阶段会降到 `0--35%`。`batch=63/84` 的首 shard 总耗时分别约 `235/257s`，显存约
`29.0/32.5GB`，均未优于 `batch=36`；因此不能把“占满显存”当作加速依据。

`mask_logits_to_point_aabbs` 已改为同一场景内 query 维并行的 masked reduction/nanquantile，
geometry row 的 GPU-to-CPU 传输也从逐字段逐行同步改为整批传输，并新增每 shard 的耗时与
rows/s 日志。A100 微基准中 quantile 段由 `10.55ms` 降到 `1.33ms`。优化后的前两个
`252`-row shard 与原生产 cache 逐字段比较，boxes/features/IoUs 最大误差均为 `0.0`，所有
整数、布尔、身份、provenance 与 parity 字段也逐位一致。`42` 个 mask 测试和 `175` 个
geometry cache/durability 测试全部通过。

四卡同起的连续两-shard 实测稳态为：batch36 `18.90s`（`13.33 rows/s`）、batch42
`19.41s`（`12.98 rows/s`）、batch63 `17.42s`（`14.46 rows/s`）。旧生产 batch36 尾段均值为
`29.46s/shard`。虽然 batch63 略快，但生产 manifest 已不可变地绑定 batch36；改为63必须从头
丢弃已完成的52个 shard，不值得。train 生产 cache 因此在 `13,104/36,665` rows 处安全中断，
保留全部原子提交的52个 shard，并用向量化 batch36 原地恢复。val cache 已完整发布
`9,508` rows/38 shards，content digest 为
`75e34f1f57062ddea8928ae6cfd41f557d1b69a67cd9f41421dfd7b1056497b6`。

MoE 主训练 launcher 过去把 `--nproc_per_node` 写死为1，这是“给出多张可见卡但只有一张在跑”
的直接原因。现已按 `CUDA_VISIBLE_DEVICES` 自动推导进程数，并允许显式
`NPROC_PER_NODE` 覆盖；非正整数或超过可见 GPU 数会在启动前失败。shell 语法、自动四 rank
和显式两 rank dry-run 均通过。正式长训可直接给出 `CUDA_VISIBLE_DEVICES=0,1,2,3`，短 smoke
仍优先四卡各跑独立配置，以保持单卡优化轨迹并提高实验吞吐。

受保护 V19 再次核验为 mode `0444`、8 hardlinks、SHA-256
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`；历史
`0.582878/0.486012` 三组件未删除。当前指标没有因本节吞吐优化发生变化。

下一网络版的 query-wise mask calibration 必须作为端到端接口修改：MCLN 可在 V19 样本级
alpha 上叠加 zero-init query residual，但 `grounding_evaluator` 需把 `[Q]` 显式 reshape 为
`[1,Q,1]`，mask loss 需按 Hungarian `idx0` gather 对应 alpha，`source_choice_adapter` 和
`rec_candidate_adapter` 不得再 `.mean()` 回样本标量，`rec_mask_geometry` 则需按原始 query index
gather alpha。以上消费者全部通过 identity/shape 测试前，不启动 full train；否则即使模型头正确，
训练与官方 mask 评估也会使用不同融合语义。

### 35.15 Query-wise mask fusion calibrator 与四卡 smoke（2026-08-05）

已实现网络内 `QueryMaskFusionCalibrator`：输入最后层 query、masked-mean 文本上下文和归一化
box center/size，在 V19 样本级 alpha 上添加有界 query residual。输出层权重/bias 为零初始化，
step-0 的 256 个 query 权重与 V19 alpha 逐位相同；query/text/box/base-alpha 默认 detach，首阶段只
更新新头，避免破坏 V19 定位与 MoE 路径。新头共 `92,429` 个参数，checkpoint loader 只允许
V19 缺少对应的 12 个 state-dict keys，其他 missing/unexpected key 均 fail closed。

共享实现位于 `models/mask_fusion.py`，统一 scalar、`[Q]`、`[Q,1]` 及 batched query weight 的
规范化、gather 和 logit-space fusion。Hungarian mask loss 按 `idx0` gather alpha；官方
`grounding_evaluator`、SourceMoE mask ranking、source-choice mask score、REC candidate features、
mask geometry 及 joint box-mask cache/audit 均已迁移。旧 `.mean()` 和 `reshape(1)` 标量假设已
移除。identity/query-index/shape/detach/gradient/非零 query variance 门禁连同现有消费者回归为
`236 passed`，另一次 CLI/optimizer 集成为 `148 passed`，Python compile 与 launcher dry-run 通过。

新增 `scripts/train_scanrefer_query_mask_fusion.sh`，支持按 `CUDA_VISIBLE_DEVICES` 自动推导 DDP
ranks，也支持单卡独立超参 smoke。debug 数据集原来在截断为128条之前解析完整 split，现已在
ScanRefer/Sr3D/Nr3D 的场景图解析前截断，双数据集初始化由数分钟降到约25秒。mask loss 中每样本
GPU->NumPy->GPU 的高斯权重同步也改为纯 Torch；10组随机 float32 对比最大差
`2.3841858e-07`。batch56 单卡容量标定峰值约 `25.4GB`，2 train batches 为 `18.8s`；正式三组
smoke 统一使用 batch64，使每 epoch 完整覆盖128 rows，并预计占用约29GB/卡。

当前 GPU0 继续顺序发布 V19 geometry train；GPU1/2/3 分别运行
`lr/max_delta = 3e-4/0.10, 1e-3/0.20, 3e-3/0.25` 的 5-epoch smoke。最初三组暴露 Hungarian
CPU index 与 CUDA mask 的 `index_select` 设备不一致，修复为显式把 query index 移到 mask
device；随后单卡首轮已确认 checkpoint 合同、forward/backward、非零 residual/query std 和保存
路径均有效。只有完成整轮指标与 checkpoint finite/nonzero 审计后才决定 full-data 四卡长训，
不会因显存占用高而跳过质量门禁。

### 35.16 冻结状态修复、清理与四卡完整训练（2026-08-05）

首次完成的 head-only smoke 不能用于选型：optimizer 虽然只含 query calibrator 的 12 个 state，
但 `model.train()` 把冻结主干的 BatchNorm 也切到了 train mode，V19 与候选 checkpoint 的
common/changed/new 实际为 `1228/204/12`。`BaseTrainTester._set_source_moe_train_mode` 现已在
query-only 模式先对整网执行 `eval()`，再只对 `query_mask_fusion_calibrator` 执行 `train()`；一次
真实 backward/Adam step 的回归测试确认顶层模型、冻结 BatchNorm/Dropout 保持 eval，所有非
calibrator state 逐位不变。新增 mask-fusion 测试为 `7 passed`，训练接线和 checkpoint 集成为
`117 passed`，Python compile 通过。

修复后四卡各自运行 batch64、5 epoch、128-row debug smoke，显存稳定约 `32.66GB/GPU`，四卡
均观测到有效计算峰值。四组 epoch-5 residual mean/max/query-std 分别为：`1e-4/0.10/d0`
`0.000823/0.001236/0.000144`，`3e-4/0.10/d0` `0.005810/0.008260/0.000746`，
`5e-4/0.15/d0.1` `0.021414/0.032972/0.003425`，`1e-3/0.20/d0.1`
`0.137633/0.180511/0.005342`。四组 checkpoint 审计均为 `1228/0/12`，12 个新增 tensor、
12 个 Adam state 和 24 个 moments 全部 finite/nonzero，optimizer step 为10。debug 最佳都停在
epoch 1：REC `0.500000/0.445312`，Mask `0.500000/0.406250/0.350516`；该128样本结果只用于
稳定性筛选，不能与完整9508-row validation 指标横向比较。

所有旧的无效 smoke 和修复后未晋级 smoke `.pth` 已逐文件 unlink，日志、config、metrics、
retention JSON 均保留；磁盘可用空间由约 `1.5GB` 恢复到 `6.0GB`。受保护 V19 仍为 mode
`0444`、8 hardlinks，SHA-256 仍为
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。

V19 geometry train cache 已完整发布 `36,665` rows。规定配置
`hidden=256/dropout=0.1/lr=3e-4/seed=0` 的 train-only scorer 在 epoch 27 early-stop 选中，
calibration 为 `0.95310/0.91697`、geometry weight `0.80`。独立 V19 one-shot claim 随后消费
完整 `9,508`-row validation sidecar，frozen parent 为 `0.580143/0.465713`，parent+geometry 为
`0.579617/0.482436`：Acc@0.25 净降 `0.000526`，Acc@0.50 净升 `0.016723`，但仍低于历史
后处理最高 `0.582878/0.486012`。selection/claim/record 均为 mode `0444`；record SHA-256 为
`8d871a493c24b0723cf7a9bd8e004fc2c6c688815cb8e0d8812b94063608a11b`。该 geometry 后处理分支
到此关闭，不再根据 validation 调参。

同时已启动
`mcln_qmask_full80_4gpu`：4-rank DDP、每卡 batch64、global batch256、80 epoch、
`lr=1e-4/max_delta=0.10/dropout=0`，输出目录为
`query_mask_fusion/scanrefer/qmask_full80_b64x4_lr1e4_delta010_d0_fixedbn/1785875641`。选择保守
学习率是因为高学习率在仅10 step 时已明显逼近 residual 上限；完整训练仍必须逐 epoch 检查
Mask 三指标、query variance、饱和率和 `1228/0/12` checkpoint 合同。

### 35.17 四卡利用率诊断与无损续训提速（2026-08-05）

完整训练确认四个 DDP rank 均存活，显存分别约为 `35.4/31.8/32.0/38.0GB`；因此“只有 GPU2
运行”是瞬时利用率观测造成的误判，而不是 rank 缺失。连续 15 秒采样的四卡平均 SM 利用率约为
`33.5%/37.7%/29.9%/38.7%`，总平均约 `35.0%`，各 rank 经常交替出现 `0%` 和 `100%`，说明
主要问题是输入/同步等待，不是显存不足。GPU3 已接近 40GB 上限，继续提高 batch64 会显著增加
中途 OOM 风险，不能用“占满显存”替代吞吐分析。

根因是启动环境把 `OMP_NUM_THREADS` 和 `MKL_NUM_THREADS` 都继承为40。当前每卡4个 DataLoader
worker、共16个 worker，每个 worker 实际拥有42个线程并占用约 `2.1--2.3` 个 CPU 核；在仅40个
可用核上形成数百 runnable threads、约 `29--31%` system CPU 和每秒数百万次 context switch。
launcher 现改为由 `CPU_THREADS_PER_PROCESS` 统一强制设置 OMP/MKL/OpenBLAS/NumExpr，默认1，
并关闭 tokenizer 内部并行；query-mask launcher 将 workers/prefetch 显式化，首次候选为每 rank
8 workers、prefetch factor 1，理论 outstanding batch 数与旧 `4 workers x prefetch 2` 相同。
训练 loader 支持显式常驻跨 epoch；每个 worker 初始化时也显式 `torch.set_num_threads(1)`。
但容器 cgroup 上限约 `288GB`，明显低于主机 `free` 显示的629GB；为了避免 validation 时 train/test
两组大 dataset workers 叠加，正式恢复默认关闭 persistent workers，待实测 cgroup 峰值后才允许开启。
query-head-only DDP 已关闭无意义的 unused-parameter autograd 图遍历。

进一步沿 autograd 路径审计确认，query calibrator 只从
`adaptive_weight_loss_mask/adaptive_weight_loss_dice` 接收梯度；box/CE/semantic/corresponding
损失均连接冻结输出，SourceMoE mask IoU target 也明确位于 `no_grad()`。因此 query-only 模式新增
梯度等价 fast path：只执行最后层一次 Hungarian matching，并只计算上述两项融合 mask loss，跳过
其余6层重复 matching、冻结定位/语义损失、mask correspondence 几何循环和 MoE ranking 统计。
合成 matched-query 测试中 fast/full 的两项 loss 与 adaptive-weight 梯度均 `torch.equal`；fast-path
入口另有测试确认完整 criterion 不会被调用且缩放梯度正确。loss/SourceMoE/checkpoint 专项合计
`190 passed`，Python compile 通过。

为避免因提速重启清空 Adam，新增严格 `query_mask_fusion_resume_optimizer` 合同：只允许
query-mask-only、非 eval、非 reduce-lr 模式；checkpoint 的启用状态、lr、hidden、dropout 和
max-delta 必须逐项一致；起始 epoch 必须严格等于 checkpoint epoch+1；optimizer 与 scheduler
缺失或参数漂移均 fail closed。恢复、非连续 epoch 拒绝和配置漂移拒绝回归连同既有 checkpoint/
mask 基础测试为 `28 passed`，Python compile 与 launcher shell syntax 通过。当前 epoch 1 不被中断，
待其原子 checkpoint 和完整 9,508-row validation 完成后，再从 epoch 2 使用新 CPU 配置续跑并
实测吞吐；本节尚未产生新的模型指标。

原进程在 epoch-1 checkpoint 发布后进入 validation，但在 metrics receipt 之前无 stderr 留存地
退出；四卡和全部 rank 随后释放，非人工中断，kernel cgroup 计数没有 OOM kill。epoch-1 checkpoint
已用新增 `qmask` 审计 profile 验证：`1228/0/12`、12 states、`92,429` numel、Adam step143，
所有模型与24个 moments finite/nonzero，故可安全续训。恢复 launcher 将 stdout/stderr 同步写入
独立日志；epoch1 因缺完整9508-row receipt 不产生指标 claim。

首次 fast resume 实际捕获了退出根因：每 rank 8 workers 虽保持理论 outstanding 数不变，但32个
worker 会同时构造 batch64 的点级 mask，cgroup `memory.max` 事件由 `52,737` 增至 `60,283`，
rank3 worker 被系统 `Killed`，训练在首 batch 前 fail closed。该失败 run 只有 config/log、没有
checkpoint。生产配置因此改为每 rank 4 个单线程 workers、prefetch1，相比旧 `4x2` 把在途 batch
减半；persistent 仍关闭。GPU加速主要依靠梯度等价 fast loss 和消除640线程级过度并发，而不是
继续放大 host-memory 队列。

最终恢复 run 为
`qmask_full80_b64x4_lr1e4_delta010_d0_fastresume_w4p1/1785879774`，四个 rank 均打印
`resumed exact query-mask optimizer and scheduler state`，从 epoch2 连续训练。数据集就绪到 step50
为 `238s`，旧 run 同区间为 `851s`，端到端前50步吞吐提升 `3.58x`；warmup 后稳定在
`3.9--4.0s/step`。15秒四卡平均 SM 为 `64.1%/63.9%/64.1%/75.7%`，总体约 `67.0%`，
相对旧配置总体约 `35.0%` 接近翻倍；显存约 `33.8--38.1GB`。cgroup 当前约
`190GiB/288GiB`，`memory.max=60,283` 未再增加，因此不再提高 workers/prefetch。fast path 的
日志只保留两项有梯度的 adaptive mask loss，其余冻结 loss 显式为0，这是预期行为而非损失缺失。

### 35.18 Epoch 2 正式回执与资源上限确认（2026-08-05）

fast resume 的 epoch 2 用时 `583.21s`，完整 `9,508` 样本 validation 已发布原子 metrics receipt：
learned-selector REC Acc@0.25/0.50 为 `0.580669/0.464767`，Mask Acc@0.25/0.50/mIoU 为
`0.597602/0.491060/0.418006`。本轮未超过受保护 V19 网络指标
REC `0.581195/0.465398`、Mask `0.598233/0.491376/0.418613`，因此只作为长训轨迹记录，不能替换
V19 最佳权重。epoch 3 已自动开始，80-epoch 计划不中断。

epoch-2 checkpoint 的 `qmask` 审计为 pass：epoch/Adam step 为 `2/286`，模型
common/changed/new 为 `1228/0/12`，optimizer 含12个 state、`92,429` 个参数元素和24个
finite/nonzero moment tensors。retention manifest 同步记录五项指标；当前
`ckpt_epoch_2`、`ckpt_epoch_last` 和五个 best 名称均为同一 inode 的7个 hardlinks，只实际占用
约605MB。后续指标不佳的 epoch 文件可原子替换/删除而不影响受保护 V19。

验证末段的四卡即时 SM 曾达到 `97%/100%/81%/79%`，显存为
`37.4/38.1/33.9/38.0GB`；这确认四个 rank 都在有效工作。每卡 batch64、global batch256 已使至少
一张卡只剩约2.8GB余量，不能再安全增大 batch。8 workers/rank 又已经在288GiB cgroup 内存上限
触发 worker kill，所以生产 run 保持4 workers/rank、prefetch1。短时0%利用率主要出现在
epoch/validation 切换和输入等待阶段，不能据单点采样判断某张卡未运行。

### 35.19 Epoch 3--5 轨迹与 V41 joint query quality（2026-08-05）

80-epoch query-mask run 保持四卡连续运行并已进入 epoch 6。epoch 3 的 REC 与 epoch 2 完全相同，
Mask threshold hits 也完全相同，仅 mIoU 从 `0.4180057` 增至 `0.4180083`。epoch 4 的 REC 为
`0.580669/0.464661`，Mask 为 `0.597602/0.491060/0.4180105`；epoch 5 的 REC 与 threshold
指标不变，mIoU 回落至 `0.4180094`。因此 retention 以 epoch 2 保存 REC@0.25、REC@0.50、
Mask@0.25、Mask@0.50 四项 best，以 epoch 4 保存 mIoU best，epoch 5 仅作为 latest。epoch 3
在不再被任何 best alias 引用后已自动删除，属于预期的低价值权重清理。

epoch-4 checkpoint 的 `qmask` 审计通过：common/changed/new 为 `1228/0/12`，Adam step 为
`572`，12 个 state、92,429 个参数元素及24个 moments 全部 finite/nonzero。受保护 V19 仍为
mode `0444`、8 hard links，SHA-256 为
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`，未被 retention 或清理
触碰。当前网络最高仍是 V19 REC `0.581195/0.465398`、Mask
`0.598233/0.491376/0.418613`；历史后处理 REC 最高仍是 `0.582878/0.486012`。

下一网络分支 V41 为 `JointQueryQualityReranker`。它以 V19/SourceMoE 的
`selected_source_scores` 为锚，输入每个 query 的152维 rich candidate feature、归一化基线
rank 和行内标准化父分数，经 query-set self-attention 和共享 FFN，同时预测 box 0.25/0.50、box IoU、mask
0.25/0.50、mask IoU，并用零初始化 residual head 直接重排全部 query。训练目标严格以 box tier
优先：stride 4 保证任何 mask 增益都不能把低 box tier query 提到高 tier query 之前；mask 质量
只在同一 box tier 内提供泛化友好的联合证据。step-0 的最终 score 与 V19 逐位一致。

V41 joint-only 现有梯度等价 fast loss：只计算最后层 query box IoU、正式 fused mask IoU 和
V41 多任务/listwise/anchor loss，跳过 proposal+6层 decoder 的冻结 Hungarian、检测、语义和
分割损失。合成 full/fast 对照要求总 loss、四个 V41 子 loss 和每个参数梯度一致；相关回归为
`363 passed`，V41/audit 专项为 `144 passed`。真实受保护 V19 CPU 初始化也已验证：现模型
state count `1248`，只缺预期20个 `joint_query_quality_reranker.*` tensor；V41 共153,531个参数，
旧 optimizer/scheduler 完全不加载，冻结 V19 state 逐 tensor 相等。PointNet checkpoint 改为先
`map_location="cpu"` 加载，再由训练流程统一搬到 rank 对应 GPU。

V41 launcher 保持每卡 batch64、四卡 DDP、4个单线程 workers、prefetch1、persistent off。
此前8 workers/rank 已实际触发 cgroup worker kill，故不能用增加 host prefetch 来换取表面显存
占用；当前 GPU0 峰值约37GB且四卡均频繁达到100% SM，batch 也不再上调。V41 尚未启动真实 GPU
训练，因为四卡正用于不中断的 qmask 80-epoch run；待该 run 完成或用户明确调整优先级后，先做
1个正式 debug epoch，要求20个 Adam moments finite/nonzero、残差跨 query 非零、V41 evaluator
确实消费重排 score，再进入完整训练。

### 35.20 Epoch 6 与 V41 自动接力门禁（2026-08-05）

qmask epoch 6 的完整 `9,508`-row receipt 已发布。learned-selector REC Acc@0.25/0.50 为
`0.580669/0.464661`，Mask Acc@0.25/0.50/mIoU 为
`0.597602/0.491165/0.418021`。retention 继续由 epoch 2 保存 REC@0.25、REC@0.50 和
Mask@0.25；epoch 6 刷新 Mask@0.50 与 mIoU best，并同时作为 latest。该结果仍低于受保护 V19
网络最好 REC `0.581195/0.465398`、Mask `0.598233/0.491376/0.418613`，因此不替换 V19。
训练已自动进入 epoch 7，warmup 后四个 rank 稳定约 `3.85s/step`，单点 GPU 利用率差异来自
DataLoader/DDP 同步窗口，不代表缺 rank。

V41 新增 residual 诊断 `residual_abs_mean/residual_abs_max/residual_query_std`，由模型写入
`end_points` 并进入完整 evaluation 汇总，用于确认 reranker 不仅整体偏移分数，而且确实产生
query 间差异。debug launcher 合同明确：`DEBUG=1` 默认验证128条，正式模式默认9,508条。
专项测试与 checkpoint audit 合计 `29 passed`，汇总器 Python compile 通过。

新增 `scripts/queue_v41_smokes_after_qmask.sh` 作为无日志轮询的自动接力：它绑定当前 qmask 主进程，
等待其自然退出后验证 epoch 80 receipt/latest checkpoint 和 protected V19 SHA-256，再等待四卡无
compute process，随后每张卡并行运行一个3-epoch、batch64的 V41 debug 变体。四个 checkpoint 必须
通过 `v41` profile 审计，且 residual mean/query std 必须 finite 且非零；结果由
`scripts/summarize_v41_smoke_panel.py` 原子汇总。原始队列只完成 smoke 和候选筛选；35.21 的
结构修正通过扩大回归后，队列已升级为门禁通过即进入预注册的完整训练，具体合同见下节。

### 35.21 V41 质量驱动部署排序修正（2026-08-05）

在首个真实 GPU smoke 前完成目标函数到部署路径的审计，发现初版 V41 有两个结构性信息断点。
第一，六任务 quality head 只接受辅助 BCE/regression 监督，最终 `selected_source_scores` 完全由另一条
residual head 输出；因此质量预测即使学准，也不能直接贡献 Top-1。第二，输入只加入父分数 rank，
丢失 V19 Top-1 与替代 query 的相对置信间隔，使网络较难判断何时应保持受保护父选择。

修正版保持 query-set attention 和同一 parent query 的 box/mask 一致性，但把行内标准化
`selected_source_scores` 作为额外输入；box 与 mask 的两档阈值头改为 ordinal 分解，显式保证
`P(IoU>0.50) <= P(IoU>0.25)`；六任务预测生成的联合质量在每个样本内中心化后，直接加入 residual
logit，再与 direct residual 共同经过有界 `tanh`。quality head 零初始化时所有有效 query 的联合质量
严格相等，中心化项为零，所以启用模块的 step-0 score、Top-1 和 V19 逐位一致。mask weight 被限制
为 `<0.8`，从数学上保证任何 mask 收益不能跨越 box tier；队列中的 `0.25/0.50` 均满足该合同。

输入从153维增至154维，只增加 LayerNorm 2个和 projection 128个参数；state tensor 仍为20个，
总参数及 optimizer numel 精确为 `153,531`，`v41` checkpoint audit 已同步。新增测试证明 ordinal
概率嵌套、permutation equivariance、无效 query 屏蔽和 step-0 identity；并在 auxiliary quality loss
权重为0时确认 listwise 梯度仍直接到达 quality head 与 residual head。定向/接线/checkpoint 三组回归
分别为 `32/136/26 passed`，Python compile 与两个 launcher 的 shell syntax 均通过。当前 qmask
80-epoch run 未被修改或中断，事件驱动 V41 四卡 smoke 队列将直接使用本修正版。为避免 qmask
结束后四卡空等，四组 smoke 全部通过后将自动启动固定而非 debug 指标选择的正式配置：4-rank、
每卡batch64、80 epoch、`lr=3e-4/dropout=0.1/mask_weight=0.25/quality_weight=1.0/`
`temperature=0.25/anchor=0.5`。任一 smoke receipt、20-state/153,531-numel audit、residual mean 或
query std 门禁失败都会阻止正式启动；正式结束后还必须通过 epoch80、Adam step11440 的 V41
checkpoint audit。该自动化不使用 ScanRefer validation 调参，只减少实验间空闲时间。

### 35.22 qmask epoch 7 首次整体刷新（2026-08-05）

epoch 7 完整 9,508-row receipt 为 REC `0.581090/0.465187`，Mask
`0.598128/0.491271/0.418549`，五项均超过 epoch 2--6 的 qmask 轨迹并由 retention 同时晋级。
相对 protected V19 仍分别少约1/2个 REC hits、1/1个 Mask hits，mIoU 低约 `6.46e-5`，但已从
epoch 6 的 `0.580669/0.464661` 和 `0.597602/0.491165/0.418021` 明显脱离平台，证明完整训练不应
提前停止。epoch 8 训练 residual mean/max/query-std 已增长到
`0.032737/0.095969/0.035570`，仍在 `max_delta=0.10` 合同内，并于 `07:16:21 CST` 保存后进入
完整验证。

epoch-7 qmask checkpoint 审计通过：common/changed/new=`1228/0/12`，Adam step=`1001`，12个
optimizer states、92,429个参数元素及24个 moments 全部 finite/nonzero。epoch7 文件与 REC/Mask
五个 best alias 为同一 inode、6个 hard links，只占一个约605MB权重；protected V19 SHA-256 仍为
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。

epoch 8 完整回执为 REC `0.580879/0.464872`、Mask
`0.598023/0.491376/0.418471`。除 Mask@0.50 外均低于 epoch 7；Mask@0.50 为
`4672/9508=0.4913757`，与 protected V19 精确并列并刷新 qmask best。因此 retention 当前以
epoch 7 保存 REC 两档、Mask@0.25和mIoU，以 epoch 8 保存 Mask@0.50及latest。epoch 8 审计为
`1228/0/12`、Adam step1144、12 states/92,429 numel，所有 moments finite/nonzero。

权重清理已随 retention 原子完成：epoch 2--6 的低价值 `.pth` 不再存在，目录中只剩 epoch7 inode
的5个链接和 epoch8 inode 的3个链接，实际仅占两个约605MB checkpoint。protected V19 仍为
mode `0444`、8 links、原 SHA-256，未被本轮清理触碰。

### 35.23 四卡显存与吞吐上限复核（2026-08-05）

epoch 9 训练期间再次核对运行态，四个 DDP rank 均存活且分别绑定 GPU0--3；显存常驻约
`37.0/34.5/32.4/34.5GB`。连续25秒采样中四卡都多次达到 `100% SM`，计算高峰功耗达到或短时
超过250W配置上限。不同 rank 的高负载窗口会交错，原因是 ScanRefer 动态样本的数据准备与
PCIe-only DDP 同步等待；拓扑中任意两卡均为 `PHB`，没有 NVLink。单次 `nvidia-smi` 看到一张卡
100%、其余卡较低不能解释为只有一张卡参与训练。

当前吞吐相关设置已经全部生效：每 rank batch64、global batch256、4个单线程 workers、
prefetch1、pinned memory、non-blocking H2D、TF32、cuDNN benchmark，以及 query-only 梯度等价
fast loss。cgroup 内存实测约 `202.5/309.2GB`（十进制，即约 `188.6/288.0GiB`）；此前
8 workers/rank 已实际触发该上限。GPU0 只剩约3.9GB名义余量，且动态 batch 峰值尚需安全空间，
因此 batch、worker 和 prefetch 均不再上调。主动分配未使用显存本身不会提高吞吐，反而会缩小
峰值容错空间。

当前 qmask 长训不重启、不改变数值合同。自动接力的四个 V41 smoke 会一张卡一个任务并行运行；
正式 V41 继续使用四卡 DDP、batch64/rank、4 workers/rank、prefetch1。该配置是在 GPU OOM、
host OOM 与吞吐之间的已验证生产上限，不再根据瞬时利用率做未经基准验证的放大。

### 35.24 正式训练完成门禁（2026-08-05）

新增 `scripts/audit_training_completion.py`，将“训练结束”从日志判断升级为结构化门禁。脚本复用
`metrics_from_receipt`，要求 schema=`mcln-retrain-metrics-v1`、精确样本数、REC learned/fixed 与
Mask hits 均为合法整数、各自 `hits050 <= hits025`，并要求 Mask mIoU 在 `[0,1]` 且与
`iou_sum/sample_count` 在严格误差内一致。随后以 CPU 加载 latest checkpoint，要求 checkpoint
为字典且 epoch 与预期最终轮完全一致；通过后原子发布
`mcln-training-completion-audit-v1` 收据。

门禁已同时接入 qmask epoch80 交接和正式 V41 epoch80 结束处，V41 必须先通过完整指标/epoch
门禁，之后才执行20-state、153,531-numel、Adam step11440 的 checkpoint 内容审计。定向测试连同
已有 oracle/smoke 测试为 `19 passed`，compile、CLI help 和 queue shell syntax 均通过；真实 qmask
epoch8 的9,508-row receipt + epoch8 checkpoint 演练也通过。运行中的 V41 supervisor 已只重启队列
会话以加载新脚本，并于 `07:29:45 CST` 重新绑定未中断的 qmask PID 151429。

### 35.25 qmask epoch 9 mIoU 微幅刷新（2026-08-05）

epoch 9 训练耗时 `570.29s`，完整回执通过事件等待于 `07:31:47 CST` 发布。REC 为
`0.580879/0.464767`，Mask 为 `0.598023/0.491165/0.418554`。其中 REC 两档、Mask 两档均低于
各自 qmask best；Mask mIoU 为 `0.4185541973`，比 epoch7 的 `0.4185485118` 高约
`5.69e-6`，因此只有 mIoU best 晋级 epoch9。该结果仍低于 protected V19 Mask mIoU
`0.418613`，不能替换受保护网络最好权重。

epoch9 qmask 审计通过：common/changed/new=`1228/0/12`、epoch/Adam step=`9/1287`，12个
optimizer states、92,429参数元素、24个 moments 均 finite/nonzero。retention 现在保留三个实际
checkpoint inode：epoch7 承担 REC 两档与 Mask@0.25，epoch8 承担 Mask@0.50，epoch9 承担 mIoU
与 latest；没有可删除的低价值 inode。训练已继续进入 epoch10。

### 35.26 V42 网络内 query-mask 联合校准与后继队列（2026-08-05）

分割瓶颈不能仅从 qmask 当前曲线判断为“冻结 mask logits 不可提升”。旧 train-only 严格 oracle
面板 `20260723_stage0_panel64x16_identity/summary.json` 在1,024条训练样本上同时搜索 query、
text/query/fused mask 源和固定 logit 阈值；在不降低 box tier 的约束下，Mask@0.50 仍有
`+0.058594`、mIoU 有 `+0.056269` 的可用空间。旧 `JointBoxMaskAdapter` 把该空间离散化为离线硬
selector，但 scene-disjoint train calibration 的五项正式指标全部退化，因此该 adapter 不进入正式
队列，也不把 validation 阈值搜索包装成结构创新。

对该面板进一步排除 argmax 并列后，仍有318/1,024条样本的边界阈值相对内部
`{-0.5,0,0.5}` 严格提高 mask IoU，累计 IoU 增益约7.879；另有306/1,024条样本的纯text或query
源严格优于既有fused源，累计增益约6.674。原先 alpha delta `0.5` 和bias `1.0`虽在理论极限能逼近
纯源/阈值1.0，但要求`tanh`进入饱和区。V42正式合同因此改为alpha delta `1.0`、bias `2.0`，使
纯源和等效阈值±1在约`atanh(0.5)`的非饱和位置即可到达；这来自train-only结构覆盖分析，不读取
validation。

V42 在 V41 的共享 query-set attention 内增加连续、逐 query 的 mask 校准。它把原始 mask alpha
作为第三个标量证据输入，并由同一个 hidden state 输出两个零初始化量：有界
`mask_alpha_residual` 调整 text/query 融合比例，有界 `mask_logit_bias` 同时平移两路 mask logits。
后者在融合后严格等价于对 fused logits 加一个网络预测的逐 query 分割阈值偏置。正式上限固定为
alpha delta `1.0`、logit bias `2.0`，step 0 的定位 score、融合 alpha、两路 mask logits 和最终
fused mask 均保持 V19 identity。无效 query 的两个残差都强制为0，set attention 的 query
permutation equivariance 保持不变；定位侧继续使用 box-tier 优先目标，不允许 mask 收益跨 box tier。

joint-only fast path 现直接复用正式 `SetCriterion.forward_query_mask_fusion`，将
`mask_loss_scale * (10*focal + 2*dice)` 加入 V41 的 listwise/multitask/anchor loss。这样 alpha 和
bias 都由原始训练 mask 监督获得梯度，不扫描 ScanRefer validation 阈值。V41 关闭该功能时仍保持
20个 state、153,531参数；V42 只新增一个2输出 head并把输入扩一维，共22个 state、153,919参数，
相对 V41 只增加388个参数。

真实 CPU 构造使用与 launcher 相同的 `prepare_source_moe_gate_checkpoint_config` 继承 V19 的
三源/router/fallback 合同，审计收据写在
`output/joint_query_quality/v42_protected_v19_initialization_audit.json`。完整 V42 state count 为
`1250`：受保护 V19 的`1228`个 tensor 全部逐位相等，恰好只缺22个
`joint_query_quality_reranker.*` tensor，unexpected/changed common 均为0，可训练参数精确为
153,919。checkpoint audit 新增 `v42` profile，要求正式权重为 `1228/0/22`、22个 Adam state和
153,919个 optimizer elements，并精确要求alpha/bias上限为`1.0/2.0`；V41 profile反向要求mask
calibration未启用。

测试覆盖 V42 的 box/mask step-0 identity、alpha/bias 双通道梯度、mask bias 融合等价、无效 query
屏蔽、permutation equivariance、fast mask loss 到 calibration head 的真实反传、V41/V42精确参数
合同、V42 checkpoint audit和smoke非零门禁。V42定向组为`50 passed`，相关 SourceMoE、训练分组、
checkpoint、评估器和完成回执扩展组为`321 passed`，合计`371 passed`；Python compile和两个
launcher的`bash -n`均通过。

`scripts/queue_v41_smokes_after_qmask.sh` 的当前内容已升级为 V42。它仍以文件事件绑定 qmask PID，
不做分钟级日志轮询；qmask epoch80完成门禁通过后，四张卡各并行运行一个3-epoch、batch64 smoke。
每个候选都强制 `JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION=1`，必须通过128条完整 receipt、
`v42` checkpoint合同，以及非零 residual mean/query std、alpha residual、bias mean和mask weight
query std门禁。四项均通过后自动进入固定四卡DDP 80轮正式配置
`v42_qualitycoupled_maskcal_full80_b64x4_lr3e4_mw025_t025_a05_mad10_mlb20`，最终再执行epoch80
完成门禁和V42 checkpoint审计。

资源配置继续为batch64/rank、global256、4 workers/rank、prefetch1。20秒连续采样中四卡均多次
达到99--100% SM，显存约`37.0/34.5/32.4/34.5GB`；短时低谷在不同rank间交替，是动态样本和
PCIe DDP同步窗口。GPU0只剩约3.9GB且8 workers/rank已真实触发288GiB cgroup上限，故不通过
无效显存占位或提高batch/worker冒险换取表面利用率。

qmask epoch10完整回执为REC `0.580984/0.464977`、Mask
`0.598023/0.491271/0.418554`，没有刷新epoch7/8/9的任何分项best，训练已进入epoch11。
retention 已移除无best引用的epoch10命名权重；受保护V19仍为mode `0444`、8 hard links，SHA-256
保持 `2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。后续按每轮约12--13分钟
完成窗口检查，不恢复分钟级轮询。

队列复核期间 epoch11 回执于 `07:56:34 CST` 按预估窗口发布：REC
`0.580879/0.464872`、Mask `0.598023/0.491271/0.418463`，同样未刷新任何 best，故不额外运行
低价值 checkpoint 审计。epoch11 只保留为当前 `last` 恢复点；epoch12 checkpoint 保存后 retention
会自动清理其无 best 引用的 inode。训练已进入 epoch12，V42 supervisor 保持绑定 qmask PID
151429，没有提前占用 GPU。

### 35.27 V43 源感知 query-mask 校准（2026-08-05）

V42 已能逐 query 调 alpha 和阈值，但其152维 rich feature 只包含 fused mask confidence、fused
foreground ratio 和 text-query soft Dice。它能观察融合结果，却不能直接区分“text源置信度高”与
“query源置信度高”，也缺少两源各自阈值附近的分布证据；这会迫使一个共享 hidden state 从融合后
统计反推最佳源与阈值。直接扩展公共 `rec-query-v1` 不可接受：该152维 schema 已被候选缓存、
geometry reranker、joint box-mask adapter 和 hierarchical reranker 固化，改维度会破坏已有实验
证据和部署合同。

V43 因此保留公共152维输入不变，只在 joint query mask calibration 内增加一个显式可选的10维、
无GT源级 evidence 分支。对 text/query 两路分别计算 probability mean、probability std、
confidence 和 foreground ratio 共8项，再加入两路 probability L1 disagreement 与 hard-mask
disagreement 共2项。所有量都由当前样本的网络输出得到并严格落在 `[0,1]`，不使用 ScanRefer
validation、数据集类别先验或固定最佳源，可原样迁移到单阶段 ScanRefer、Nr3D 和 Sr3D。base
alpha、父分数 rank/standardization 与这10项 evidence 一起进入原 query-set attention；同一 hidden
state仍联合预测 box/mask 两档质量、IoU、定位 residual、mask alpha residual 和logit bias。

新功能由 `--joint_query_quality_use_source_mask_evidence` 单独控制，且 fail-closed 要求 mask
calibration 同时启用。默认关闭时 V41/V42 state shape、旧缓存和旧 checkpoint 完全不变。V43
启用后输入投影从155维增至165维，仍只有22个 state，参数从153,919增至155,219，仅增加1,300个
LayerNorm/Linear投影参数。quality、定位 residual 和 mask calibration 输出头继续全零初始化，故
任意10维 evidence 下 step 0 的父定位分数、query选择、alpha、两源 logits 与 fused mask 都逐位保持
V19 identity；invalid query屏蔽和query permutation equivariance也保持。

新增 `scripts/audit_joint_query_initialization.py` 将真实整网初始化检查固化为可复用门禁。受保护V19
上的 V43 收据为
`output/joint_query_quality/v43_protected_v19_initialization_audit.json`：完整目标仍为1,250个
tensor，V19的1,228个tensor全部逐位一致，恰好缺22个`joint_query_quality_reranker.*` tensor，
shape mismatch/unexpected/changed common均为0，输出头全零，子模块参数精确为155,219。checkpoint
audit 新增 `v43` profile，要求`1228/0/22`、22个非零finite Adam states、155,219 optimizer
elements、alpha/bias上限`1.0/2.0`，并要求source-mask-evidence开关为true；V42 profile反向拒绝
误启该开关。

smoke 还新增两项运行时非塌缩诊断：10维 evidence 的query std与最后两项source disagreement
mean都必须finite且大于0。测试覆盖源统计数值/边界/可变superpoint数、V43 step-0 identity、输入
校验与detach、permutation equivariance、精确V41/V42/V43参数合同、V43 checkpoint合同和smoke
门禁。V43定向组为`58 passed`，SourceMoE、公共152维schema、训练分组和checkpoint扩展回归为
`376 passed`；Python compile、两个launcher的`bash -n`和真实V19 CPU审计均通过。

事件驱动脚本文件名 `scripts/queue_v41_smokes_after_qmask.sh` 为保持运维入口不变而保留，但内容已
升级为 V43。qmask epoch80完成后四张卡并行跑4个 V43 smoke，每个都强制启用source mask
evidence并通过128-row receipt、`v43` checkpoint合同、定位/校准/evidence非塌缩门禁；全部通过
才启动四卡DDP正式实验
`v43_sourceaware_maskcal_full80_b64x4_lr3e4_mw025_t025_a05_mad10_mlb20`，资源上限仍为
batch64/rank、4 workers/rank、prefetch1，结束后执行epoch80与V43双审计。

qmask epoch13于`08:21:40 CST`发布完整9,508-row回执：REC为
`5522/4420 = 0.580774/0.464872`，Mask为
`5684/4673 = 0.597812/0.491481`，mIoU `0.418374`。其中只有Mask@0.50刷新：4673 hits比
protected V19/epoch8多1 hit，retention已将Mask@0.50 best与latest绑定epoch13。该权重审计通过
`1228/0/12`、Adam step1859、12 states/92,429 elements及全部finite/nonzero moments；REC两档、
Mask@0.25和mIoU best仍分别为epoch7/7/9，受保护V19 SHA-256仍为
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。

后继 tmux 名称继续保留为 `mcln_v41_after_qmask`，仅该 supervisor 于`08:25:11 CST`重启以加载
V43脚本，并已重新绑定未中断的qmask PID 151429；新事件日志/锁分别为
`v43_after_qmask_queue.log/.lock`。重启后四张卡仍由qmask四个rank占用，没有提前启动smoke。

### 35.28 V44 候选 query 分割密集监督（2026-08-05）

对 V43 训练路径的梯度审计确认：校准后的 fused mask IoU 在 `no_grad` 中生成，因此 box/mask
quality 与 query rerank 头能获得全部有效 query 的正确 detached target；但 alpha residual 与 logit
bias 只通过 `SetCriterion.forward_query_mask_fusion` 的 Hungarian 匹配 query 获得直接 focal/dice
梯度。校准头增加了源级证据，却仍只有每个样本极少数 query 的直接分割监督，这是结构输入与目标
密度不匹配，而不是继续增加 evidence 维度能解决的问题。

V44 不增加参数，而是为 V43 增加一个默认关闭的 train-only candidate mask objective。每个 grounding
样本分别取当前部署分数 Top-K 与 GT box IoU Top-K 的并集，屏蔽 invalid query 和 Scannet detection
样本，然后对这些 query 的真实校准后 fused logits 计算标准 focal/dice。部署分数、box IoU 和候选
索引全部 detach；GT 只用于训练监督，推理不增加输入或分支。并集同时覆盖“当前会被选中”的 query
和“定位正确但尚未被选中”的 query，避免对全部256个背景 query 使用同一 referring mask 导致 bias
塌缩。正式 objective 为
`cmw * mask_loss_scale * (10 * candidate_focal + 2 * candidate_dice)`，预注册
`cmw=0.25, K=16`，每行最多32个不同候选。

新参数为 `--joint_query_quality_candidate_mask_loss_weight` 和
`--joint_query_quality_candidate_mask_top_k`；默认分别为0和16，所以 V41/V42/V43 的数值、state、参数
和 checkpoint 合同完全不变。新诊断
`joint_query_quality_candidate_mask_query_ratio` 必须 finite 且大于0，确保 launcher 没有只传参数却
实际跳过该目标。定向测试覆盖部署/box-oracle并集、invalid与非grounding屏蔽、未选query严格零梯度、
alpha/bias候选梯度和fast train-only集成；相关 SourceMoE/mask/checkpoint扩展回归为`343 passed`，
完整仓库回归为`3138 passed, 3 warnings`；warnings均为已有PyTorch gradcheck/LR scheduler提示。
Python compile与两个launcher的`bash -n`均通过。

候选mask实现随后做了计算图等价优化：旧写法先融合全部256个query再gather候选，新写法先gather
最多32个候选的text/query logits与alpha再融合，避免构造无用的全query额外融合图。独立dense
reference证明总loss、alpha梯度和bias梯度在`rtol=0, atol=0`下逐位相等；优化后相关回归
`138 passed`。该变化只降低V44训练期算力/显存开销，不改变候选、目标或部署路径。

事件队列已预注册四个并行 V44 smoke：`cmw/K` 分别为`0.25/16`、`0.10/8`、`0.25/32`、
`0.50/16`。它们继续使用 V43 的155,219参数 checkpoint profile，同时额外要求候选覆盖率非零；全部
通过后才启动四卡80轮
`v44_candidate_mask_full80_b64x4_lr3e4_mw025_cmw025_k16`。资源合同保持batch64/rank、global256、
4 workers/rank、prefetch1。30秒运行态采样中四卡均多次达到100% SM，显存约
`37.0/34.5/32.4/34.5GB`；四卡互联均为PCIe `PHB`、没有NVLink。cgroup当前约
`208.7/309.2GB`且历史max事件60,283次，GPU0动态余量约3.9GB，因此不增加batch/worker/prefetch，
也不做无效显存占位。

V44 supervisor 于`08:44:08 CST`重启并重新绑定未中断的qmask PID 151429；新事件日志为
`v44_after_qmask_queue.log`，共享旧lock以阻止V43/V44重复队列，等待期间不轮询日志也不占GPU。
qmask epoch14于`08:34:22 CST`完成：REC `0.580879/0.464872`，Mask
`0.597707/0.491376/0.418466`，未刷新任何分项best；retention已删除低价值
`ckpt_epoch_14.pth`。当前只保留epoch7、9、13三个best inode以及epoch15 latest inode，protected
V19仍为mode `0444`、8 links且SHA-256不变。

qmask epoch15于`08:46:47 CST`发布：REC `5525/4422 = 0.581091/0.465082`，Mask
`5684/4669 = 0.597812/0.491061`，mIoU `0.418522`，仍未刷新epoch7/9/13的best；epoch16已开始。
下一次回执检查窗口为约`09:02--09:04 CST`。

### 35.29 V45 Lovasz-Jaccard 候选分割目标（2026-08-05）

正式评估器对 fused logits 执行`sigmoid > 0.5`，即logit零阈值硬分割，再计算point-mask Jaccard；
V44的focal/dice虽能训练alpha与bias，但不直接优化排序后的硬IoU误差。V45按开源
`bermanmaxim/LovaszSoftmax`的二值Lovasz-hinge公式增加可选候选mask辅助项：按margin error降序，
用离散Jaccard扩展梯度加权hinge error。该目标不含ScanRefer阈值、类别或最佳源先验，可原样用于
单阶段ScanRefer、Nr3D和Sr3D。

新参数`--joint_query_quality_candidate_lovasz_loss_weight`默认0，只有大于0时才构造排序图，故
V41--V44默认数值与速度合同不变；它复用V44的部署Top-K/box-oracle Top-K并集，直接作用于网络预测
的alpha/bias校准后logits，不增加state或推理计算。单像素margin/梯度、超过margin零损失、点排列
不变性、fast path真实反传和非法输入均已覆盖；相关扩展回归`347 passed`，compile和launcher
syntax通过。真实GPU效果仍需smoke证明，当前不把CPU合同当成指标收益。

后继四卡smoke改为严格单变量消融：除Lovasz权重`0/0.05/0.10/0.20`外，四组均固定
`lr=3e-4, dropout=0.1, mask_weight=0.25, temperature=0.25, anchor=0.5, cmw=0.25, K=16`。
任一组出现非finite训练、128-row回执不完整、V43参数合同失败或候选/源证据/校准输出塌缩都会阻止
正式启动。预注册正式候选为
`v45_lovasz_candidate_mask_full80_b64x4_lr3e4_mw025_cmw025_clw010_k16`，Lovasz权重0.10；V45仍
沿用V43的22-state、155,219参数审计合同。

V45完整仓库回归为`3142 passed, 3 warnings`，三项仍是已有PyTorch gradcheck/scheduler提示。
等待supervisor于`09:00:20 CST`重启并绑定未中断的qmask PID 151429，事件日志为
`v45_after_qmask_queue.log`，无轮询、无提前GPU占用；protected V19的SHA-256、0444权限和8 links
再次核对不变。

qmask epoch16于`08:59:17 CST`发布：REC `5523/4420 = 0.580879/0.464872`，Mask
`5686/4669 = 0.598023/0.491061`，mIoU `0.418370`，未刷新任何best。epoch16 inode当前仅由latest
和epoch名引用，epoch17保存后会由retention自动移除。相邻完整回执间隔实测12分30秒，下一次检查
窗口为约`09:11--09:13 CST`。

V45 smoke汇总门禁进一步要求非零权重组g1/g2/g3的
`joint_query_quality_candidate_lovasz_loss`必须finite且严格大于0；g0零权重对照不作该要求。由于
`09:00:20 CST`启动的supervisor仍持有修改前脚本inode，已仅终止等待队列并于`09:08:06 CST`
用同名tmux重新启动；新进程再次绑定未中断的qmask PID 151429，没有提前启动GPU任务。

epoch17训练期间的12秒逐秒采样显示四卡均反复达到`98--100% SM`，常驻显存仍约
`37.0/34.5/32.4/34.5GB`；瞬时低谷在不同rank间交替，而非只有GPU2工作。GPU0动态余量约3.9GB，
且8 workers/rank此前已触发309.2GB cgroup上限，因此继续保持batch64/rank、4 workers/rank、
prefetch1，不增加batch/worker/prefetch，也不以无效显存占位冒充吞吐优化。

qmask epoch17于`09:11:53 CST`发布：REC `5525/4422 = 0.581091/0.465082`，Mask
`5687/4672 = 0.598128/0.491376`，mIoU `0.418523`。它没有严格超过epoch7/9/13的分项best；
retention已删除低价值epoch16，仅保留epoch7、9、13 best inode与epoch17 latest inode。按约
12分30秒完整周期估算，下一次回执检查窗口为`09:24--09:26 CST`。

### 35.30 V46 fallback-gate evidence joint reranker（2026-08-05）

epoch17诊断中，固定源Top-1 oracle仅为`0.58551/0.46992`，但fallback gate候选集合oracle达到
`0.62842/0.54933`。V43--V45的joint reranker能看到rich query和mask源统计，却没有显式看到
fallback gate用来定义这组高价值候选的资格、动态锚点和已训练质量输出。V46因此增加默认关闭的
`--joint_query_quality_use_gate_evidence`，把24维无GT、部署时已存在的gate evidence送入同一set
attention质量头：candidate/default/selected/action-anchor四个指示；candidate score秩、标准化置信度、
expected utility、direct utility和action margin五项；两个box阈值与两个mask阈值各自的
break/neutral/fix概率共12项；fallback/neutral/override decision概率3项。所有输入均在`[0,1]`，
不包含ScanRefer类别、GT、数据集阈值或最佳源先验，可原样迁移单阶段ScanRefer、Nr3D和Sr3D。

该开关只把V43第一层输入加宽24维；输出head仍零初始化，所以从protected V19加载后的step0 REC、
mask alpha和mask bias严格保持原输出。V46保持22个state tensor，训练参数从155,219增至158,339。
合同测试覆盖概率归一化、非法/缺失输入拒绝、query置换等价、gate输入detach、step0 identity和精确
参数数目；checkpoint audit新增`v46` profile，smoke gate要求gate evidence query std与candidate
ratio均finite且大于0。joint/source-MoE集成、audit与summary定向回归共`162 passed`，Python compile
和launcher syntax通过。

为同时验证V45 Lovasz目标和V46架构，后继四卡smoke改成`2x2`：g0=`gate0/clw0`，
g1=`gate0/clw0.10`，g2=`gate1/clw0`，g3=`gate1/clw0.10`；其余统一固定
`lr=3e-4, dropout=0.1, mask_weight=0.25, temperature=0.25, anchor=0.5, cmw=0.25, K=16`。
g0/g1按V43参数合同审计，g2/g3按V46合同审计；两组candidate、Lovasz和非塌缩诊断全部通过后，
启动预注册四卡80轮
`v46_gate_evidence_lovasz_candidate_mask_full80_b64x4_lr3e4_mw025_cmw025_clw010_k16`。
新supervisor于`09:29:58 CST`绑定未中断的qmask PID 151429，事件日志为
`v46_after_qmask_queue.log`，没有提前占GPU。

qmask epoch18于`09:24:20 CST`发布：REC `5521/4419 = 0.580774/0.464767`，Mask
`5683/4673 = 0.597707/0.491481`，mIoU `0.418320`，未严格刷新epoch7/9/13 best；retention已用
epoch18 latest替换epoch17 latest，epoch19正常训练。下一次回执检查窗口按实测周期为约
`09:36--09:38 CST`。

### 35.31 V46 初始化前门禁、完整回归与 epoch19（2026-08-05）

指标报告口径再次固化：`0.582878/0.486012` 是 epoch-71 backbone 加其 SHA 绑定的历史
parent/geometry 后处理，不是 V19 加后处理。network-only V19 为
`0.581195/0.465398`；为 V19 重新生成完整 train/val cache、仅用 train calibration 重训 scorer
后的 one-shot parent+geometry 结果为 `5511/4587 = 0.579617/0.482436`。后续任何结果表必须把
`historical system`、`network-only` 和 `same-checkpoint retrained sidecar` 三列分开，不允许把旧
artifact 跨 checkpoint 复用或用同一数值重复申报新实验。

独立初始化审计 `scripts/audit_joint_query_initialization.py` 已增加 V46 profile。它在真实受保护
V19 上构建启用 mask calibration、source-mask evidence 和24维 gate evidence 的完整模型，结果为
source/target state `1228/1250`、common/changed/missing/unexpected/shape-mismatch
`1228/0/22/0/0`，joint head 恰有22个state、`158,339`参数，quality/residual/mask calibration
输出头均为全零。新增单元测试使用真实 `JointQueryQualityReranker` 构造同一合同；V46相关定向
回归为`163 passed`，完整仓库为`3151 passed, 3 warnings`且零失败。

等待队列现会在释放GPU并启动smoke之前，依次对V43控制组和V46实验组运行上述真实protected-V19
初始化审计；任一公共tensor变化、缺失集合、参数量或开关不符都会fail closed。更新后的supervisor
于`09:41:47 CST`重新绑定未中断的qmask PID `151429`，没有启动额外GPU任务。protected V19仍为
mode `0444`、8 hardlinks、SHA-256
`2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。

qmask epoch19于`09:36:58 CST`发布：REC
`5523/4421 = 0.580984/0.464977`，Mask
`5685/4671 = 0.597918/0.491271`，mIoU `0.418374`。它没有刷新epoch7的REC/Mask@0.25、
epoch13的Mask@0.50或epoch9的mIoU；retention因此继续只保护epoch7/9/13及当前latest，epoch20
正常训练。按最近完整周期估算，下次回执窗口约为`09:49--09:51 CST`。

启动容量审计发现`/root/autodl-tmp`只余`3.2GB`，不足承载四个并行smoke的最坏checkpoint
retention。已逐文件删除三个文档中明确失败、且均为单链接的旧mask-head权重：evalmode epoch72
（Mask `0.595499/0.486748/0.415837`）、fullmaskhead epoch73
（`0.595288/0.485591/0.414901`）和small-lr epoch72
（`0.591186/0.481700/0.412972`），共释放约`1.9GB`；对应日志/config全部保留。V28的
`ckpt_best_mask_acc050.pth`虽然REC较低，但Mask@0.50为`0.491481`并列当前分项best，明确保留；
protected V19和qmask epoch7/9/13均未触碰。V28该权重已增加只读保护hardlink
`protected_mcln_artifacts/scanrefer_mask050_best_v28_0.491481.pth`；两路径同属inode
`6502719492`、links `2`、mode `0444`，SHA-256为
`2b72aa4d7d4feb4d5423e7d7061032d88909c9a82094fd4573621cd785f808b8`。

V46 panel与正式run默认迁移到工作区文件系统
`experiment_output/joint_query_quality`，当前空闲约`9.6GB`。队列在qmask结束后强制要求该文件系统
至少还有`8GiB`，不足则在占GPU前退出；四个128-row smoke完成checkpoint审计与双summary后，
只删除这四个debug run根目录下的`.pth`再启动正式V46，保留全部metrics/audit/config/log。
更新后的等待进程于`09:48:45 CST`绑定原qmask PID，launcher语法与相关门禁回归为`29 passed`。

qmask epoch20于`09:49:28 CST`发布：REC
`5522/4420 = 0.580774/0.464872`，Mask
`5683/4673 = 0.597707/0.491481`，mIoU `0.418296`。Mask@0.50只与epoch13并列，未产生
严格best；其余指标更低，retention不新增inode并进入epoch21。下一回执窗口按实测周期约为
`10:01--10:03 CST`。

按降低轮询频率的要求，后续改为每四个完整epoch、约`45--50`分钟做一次窗口检查。epoch21--24
回执时间为`10:02:03/10:14:44/10:27:09/10:39:38 CST`，结果如下：

| epoch | REC@0.25 / @0.50 | Mask@0.25 / @0.50 / mIoU |
| ---: | ---: | ---: |
| 21 | `5524/4422 = 0.580984/0.465082` | `5683/4670 = 0.597707/0.491165/0.418518` |
| 22 | `5524/4422 = 0.580984/0.465082` | `5684/4669 = 0.597812/0.491060/0.418364` |
| 23 | `5524/4422 = 0.580984/0.465082` | `5684/4669 = 0.597812/0.491060/0.418469` |
| 24 | `5525/4422 = 0.581090/0.465082` | `5685/4667 = 0.597918/0.490850/0.418513` |

四轮均未严格超过epoch7/9/13分项best，retention只以epoch24替换latest，没有新增best inode。
`10:40:43 CST`窗口采样四卡均为`100% SM`，qmask主进程和事件队列正常；下一窗口安排在
约`11:28--11:31 CST`，中间不检查epoch25--27。

`11:29:07 CST`窗口中，epoch25--27已完整发布，epoch28仅完成checkpoint、仍在validation，故不
读取其中间结果。完整回执为：

| epoch | REC@0.25 / @0.50 | Mask@0.25 / @0.50 / mIoU |
| ---: | ---: | ---: |
| 25 | `5525/4422 = 0.581090/0.465082` | `5685/4669 = 0.597918/0.491060/0.418553` |
| 26 | `5523/4421 = 0.580879/0.464977` | `5683/4674 = 0.597707/0.491586/0.418380` |
| 27 | `5523/4421 = 0.580879/0.464977` | `5683/4670 = 0.597707/0.491165/0.418448` |

epoch26的Mask@0.50以`4674/9508=0.491586`严格超过epoch13/V28的`4673/9508=0.491481`，
刷新当前Mask@0.50 network best；其他四项仍未刷新。该inode已增加只读保护hardlink
`protected_mcln_artifacts/scanrefer_qmask_best_mask050_epoch26_0.491586.pth`，源路径、best alias和
保护路径同属inode `6451674584`、links `3`、mode `0444`，SHA-256
`b825ee71f5d8b810307a6c139d54d1d03e7c2ee140c47d67ff0f7c460053de2e`。retention保留epoch7、
epoch9、epoch26和当前latest；下一窗口按四轮周期安排在约`12:18--12:21 CST`，中间不单查
epoch28--31。

`12:18:38 CST`窗口中，epoch28--31已完整发布；epoch32已保存checkpoint但仍在validation，未读
其中间结果：

| epoch | REC@0.25 / @0.50 | Mask@0.25 / @0.50 / mIoU |
| ---: | ---: | ---: |
| 28 | `5522/4420 = 0.580774/0.464872` | `5682/4668 = 0.597602/0.490955/0.418302` |
| 29 | `5522/4420 = 0.580774/0.464872` | `5683/4670 = 0.597707/0.491165/0.418360` |
| 30 | `5523/4422 = 0.580879/0.465082` | `5682/4670 = 0.597602/0.491165/0.418334` |
| 31 | `5522/4421 = 0.580774/0.464977` | `5682/4672 = 0.597602/0.491376/0.418123` |

四轮没有刷新任何分项best；retention继续保护epoch7、epoch9、epoch26和latest，四卡训练与V46
等待队列均正常。下一检查窗口约为`13:08--13:11 CST`，中间不单查epoch32--35。

`13:07:46 CST`窗口中epoch32--35已完整发布，epoch36只完成checkpoint并仍在validation：

| epoch | REC@0.25 / @0.50 | Mask@0.25 / @0.50 / mIoU |
| ---: | ---: | ---: |
| 32 | `5523/4421 = 0.580879/0.464977` | `5683/4669 = 0.597707/0.491060/0.418253` |
| 33 | `5522/4421 = 0.580774/0.464977` | `5683/4675 = 0.597707/0.491691/0.418240` |
| 34 | `5523/4422 = 0.580879/0.465082` | `5683/4668 = 0.597707/0.490955/0.418362` |
| 35 | `5524/4422 = 0.580984/0.465082` | `5684/4671 = 0.597812/0.491271/0.418303` |

epoch33将Mask@0.50 best再提高1 hit至`4675/9508=0.491691`，其余分项不变。新best保护路径为
`protected_mcln_artifacts/scanrefer_qmask_best_mask050_epoch33_0.491691.pth`，与run内epoch/best
路径同属inode `6542431840`、links `3`、mode `0444`，SHA-256
`20a1a877875cc194356fe1b0781528cd9af8d4e591288f6c91a19c1f71b61b46`。被同架构同指标严格
取代的epoch26外部保护单链接已删除，释放约605MB；其metrics/hash记录保留，V19未动。下一窗口
约为`13:56--13:59 CST`，中间不单查epoch36--39。

## 单阶段 ScanRefer 迁移实验（2026-08-05）

用户要求先把历史最高后处理网络和方法迁移到单阶段 ScanRefer。历史三件套严格绑定的
epoch-71 backbone SHA 为 `3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`，
plain REC 为 `0.57993/0.46350`，后处理（parent + geometry）为
`0.582878/0.486012`。本实验使用该 backbone 初始化，保留
`default,default_rank_blend_contrastive010` 双源 selector 和同一训练目标，但运行时显式
关闭 `butd/butd_gt/butd_cls`，并保留官方单阶段 `joint_det` 的 ScanNet+ScanRefer 训练构成。

新增 [audit_scanrefer_single_stage_transfer.py](../scripts/audit_scanrefer_single_stage_transfer.py)
在训练前构建 `butd=false` MCLN 并做张量契约审计：共享 `1078` 个 tensor 全部形状一致，仅允许
丢弃检测框流的 `66` 个 tensor；任何其他 missing/unexpected/mismatched 参数都会 fail-closed。
四卡 launcher 为 [train_scanrefer_mcln_sp_single.sh](../scripts/train_scanrefer_mcln_sp_single.sh)，
默认 batch `18/卡`、fresh optimizer、`1--100` epochs、每 epoch 验证并保留五项分项 best。

单阶段 checkpoint 不能复用双阶段 SHA 绑定的 parent/geometry artifact。训练结束后必须用该单阶段
checkpoint 重新生成 train/val candidate cache，再以 portable provenance 训练同构 parent + geometry
sidecar；因此下阶段报告分开记录 network-only 与重新绑定的后处理结果，不能直接套用
`0.582878/0.486012`。

正式 run 于 `13:28:02 CST` 启动，目录为
`output/single_stage_best_postprocess/scanrefer/mcln_epoch71_parent_geometry_single_stage_e1_e100_b18x4/1785907694`。
首次 joint 数据初始化实测约 `8:31`，确认 train/val 样本为 `48,655/9,508`；进入训练后四卡
显存约 `35.4--35.7GiB/卡`、采样窗口均为 `100% SM`，batch 18/卡 无 OOM。按 676 train
steps 和完整验证估算，epoch-1 收据窗口为 `14:03--14:10 CST`，到该窗口前不轮询指标。

为让单阶段重训同一 geometry 方法，portable provenance 现允许两种合法输入：标准双阶段
`butd=true` 和单阶段 `butd=false`；两者仍一律拒绝 `butd_gt/butd_cls`，并继续严格绑定 checkpoint
SHA、完整 cache、152D feature schema、Top-16 candidate rule 和 parent artifact。相关定向回归为
`142 passed`。qmask 在 epoch-37 完整 checkpoint/验证收据后暂停，V46 自动接管队列已关闭；
后续可从 epoch 37 的 optimizer/scheduler 严格续训。

磁盘清理收据：qmask retention 已压缩为 epoch33（Mask@0.50=`0.491691`）和 epoch37（latest、
REC@0.25=`0.581195`）两条记录，删除 epoch7/9 的两个私有 checkpoint inode，共释放约 `1.21GB`。
清理后 `/root/autodl-tmp` 可用约 `6.2GB`；V19、qmask epoch33 Mask@0.50、历史后处理三件套均已
重新核验 SHA、inode、权限和 hardlink，未被删除或改写。

事件驱动的 [queue_single_stage_best_postprocess.sh](../scripts/queue_single_stage_best_postprocess.sh)
已绑定训练 launcher PID `357439`，不读取中间日志；训练完成后依次执行 completion audit、REC@0.25
best checkpoint 选择、单阶段 train/val candidate cache、parent reranker、mask geometry audit、
portable train/val geometry cache、geometry reranker 和最终 parent+geometry evaluation。所有 cache
和 sidecar 输出放在工作区 `experiment_output/single_stage_best_postprocess/1785907694`，避免占用
`/root/autodl-tmp` 的模型权重空间。

### 单阶段 epoch 1 与 V47 跨 stage 训练内重排（2026-08-05）

epoch 1 训练耗时 `1413.60s`，完整 `9,508` 条验证收据于 `14:04:01 CST` 发布。network-only
REC 为 `5332/4221 = 0.560791/0.443942`，Mask 为
`0.576146/0.474548/0.403539`。这是从双阶段 epoch-71 权重迁移后首轮适配结果，尚不能作为
单阶段最终指标。训练加验证的首个完整周期约 `27` 分钟，后续只按完整 epoch 窗口检查。

代码审计确认本轮训练只启用了旧二源 `SourceChoiceSelector`；`source_moe_*` 与
`joint_query_quality_*` 损失均为零。epoch-1 的两个固定源仅为
`0.560791/0.443942` 与 `0.560055/0.443206`，source-choice oracle 也只有
`0.561632/0.444783`，说明仅做样本级二选一没有足够上限。另一方面，双阶段 V19 的 query
候选 oracle 已达约 `0.6297/0.5501`，因此后继仍应优化逐 query 选择和分割校准，而不是搜索
ScanRefer validation 上的固定源组合。

V47 首先修正跨 stage 的结构耦合：`JointQueryQualityReranker` 现在可接任意已训练 source
arbiter，即 `SourceMoE` 或 `SourceChoiceSelector`。只有 24 维 fallback-gate evidence 仍严格
要求 SourceMoE fallback gate；selector 路径只能使用部署时通用的 152 维 query state、源分数
和两路 mask evidence。这样同一个 set-attention query 重排、逐 query mask alpha/bias 校准、
candidate focal/dice 及 Lovasz-Jaccard 目标可以直接迁移到单阶段 ScanRefer，并保留后续
Nr3D/Sr3D 的无数据集先验接口。

checkpoint 预处理同步支持 selector-backed joint-only 训练，并从 checkpoint 强制继承 source
列表和 hidden dim；若运行时错误启用 SourceMoE 或 gate evidence 则 fail closed。新入口为
[train_scanrefer_single_stage_joint_query_quality.sh](../scripts/train_scanrefer_single_stage_joint_query_quality.sh)，
公共 launcher 新增 `MODEL_STAGE=single` 与 `SOURCE_ARBITER=selector` 合同；原双阶段默认仍为
`MODEL_STAGE=two`、`SOURCE_ARBITER=moe`，数值路径不变。

已对正在运行的真实单阶段 epoch-1 checkpoint 执行 V43 selector 初始化审计，收据为
`experiment_output/single_stage_joint_query_contract/v43_selector_epoch1_initialization_audit.json`：
source/target state 为 `1078/1100`，common/changed/missing/unexpected/shape-mismatch 为
`1078/0/22/0/0`，新 head 恰有 `155,219` 参数，quality/residual/mask calibration 输出头
全部为零。由此证明 step 0 不改变当前网络输出。扩大回归分别为 `180 passed` 与
`171 passed`，launcher dry-run、Python compile 和 shell syntax 均通过。

新增事件队列
[queue_single_stage_joint_query_after_postprocess.sh](../scripts/queue_single_stage_joint_query_after_postprocess.sh)，
tmux 为 `mcln_single_stage_joint_query_after_post`。它于 `14:24:54 CST` 绑定现有 postprocess
supervisor PID `368662`，只用 `tail --pid` 等待，不读取训练日志、不占 GPU。parent+geometry
完整结束并通过 SHA/产物门禁后，四张卡分别并行跑 3-epoch/128-row 的 loss panel：base、
`cmw=0.10/K=8`、`cmw=0.25/K=16`、`cmw=0.25/clw=0.10/K=16`。四组必须通过
`v43_selector` 的 `1078/0/22` tensor、22 optimizer state、非零 residual、mask calibration、
source evidence、candidate coverage 和 Lovasz 门禁；debug 权重随后删除，只保留日志与收据。
最后预注册四卡、batch `64/卡`、global batch `256`、80 epoch 的正式 V47 训练。该正式实验尚未
产生指标，不能提前申报为提升。

epoch 2 完整收据于 `14:31:16 CST` 发布，REC 为
`5389/4321 = 0.566786/0.454459`，Mask 为
`5531/4559 = 0.581721/0.479491`，mIoU 为 `0.407376`；五项均超过 epoch 1。retention 已删除
被全面覆盖的 epoch-1 checkpoint inode；epoch-2 inode `4374907969` 由 latest 与五项 best 共
7 个 hardlink 保护，实际只占一份 `751,924,425` bytes。epoch 1/2 收据间隔 `27:14`，下一完整
窗口为约 `14:58:30--15:00 CST`，此前不轮询。

全仓测试在 GPU 可见时首先被无 skip 条件的 `pointnet2/pointnet2_test.py` 强制 CUDA 测试触发
OOM，随后同一 pytest 进程的 CUDA context 污染造成级联假失败；没有中断或缩减四卡训练。隐藏
GPU 后重跑结果为 `3153 passed, 11 skipped, 1 failed`，唯一失败仍是该文件强制 `.cuda()` 导致
的预期“无 CUDA GPU”错误，其余仓库合同全部通过。本次变更相关的 CUDA 初始化路径另由真实
epoch-1 checkpoint 的 `1078/0/22` 审计覆盖。

### Nr3D/Sr3D 迁移启动合同审计（2026-08-05）

对真实 CSV 和官方 split 做结构化计数后，当前代码过滤规则对应的验证样本数为 Nr3D `7,899`
（130 个 test scan）和 Sr3D `17,726`（255 个 test scan）。数据盘实际目录是
`${DATA_ROOT}/refer_it_3d`，原加载器只查 `${DATA_ROOT}/ReferIt3D`；现已加入这两种公开布局的
确定性查找，均不存在时列出检查过的路径并 fail-closed，不创建数据软链接。

公共 joint-query launcher 新增显式 `LANGUAGE_DATASET`/`TEST_DATASET` 合同，允许
`scanrefer|nr3d|sr3d` 且训练/验证语言数据集必须一致。ScanRefer 默认行为仍使用历史 V19 和
`9,508` 条验证；非 ScanRefer 不再继承这两个默认值，必须显式提供对应 checkpoint 和
`EXPECTED_EVAL_SAMPLE_COUNT`，防止用错误分母发布指标。路径、shell 语法、单阶段队列和
SourceMoE/selector 扩大定向回归结果为 `110 passed`。该改动只是迁移前置合同，尚未启动
Nr3D/Sr3D 实验，也不能视为跨数据集性能结论。

epoch 3 完整收据于 `14:58:39 CST` 发布，REC 为
`5386/4329 = 0.566470/0.455301`，Mask 为
`5528/4545 = 0.581405/0.478019`，mIoU 为 `0.407761`。相对 epoch 2，REC@0.50 增加
8 hits、mIoU 增加约 `0.000385`，REC@0.25 和 Mask 两个阈值项小幅回落。因此 retention 保留
epoch 2（REC@0.25、Mask@0.25/0.50）与 epoch 3（REC@0.50、mIoU）两个 checkpoint inode，
没有保留 epoch 1 或其他差权重。epoch 2/3 收据间隔 `27:23`，下一完整检查窗口约为
`15:26--15:28 CST`。

epoch 4 完整收据于 `15:26:01 CST` 发布，REC 为
`5409/4330 = 0.568889/0.455406`，Mask 为
`5567/4539 = 0.585507/0.477387`，mIoU 为 `0.409214`。除 Mask@0.50 外四项刷新本 run
best；Mask@0.50 仍由 epoch 2 的 `4559/9508 = 0.479491` 保持。retention 已删除不再承担
任何 best/latest 的 epoch-3 checkpoint，只保留 epoch 2 与 epoch 4 两个实际 inode。历史
`0.582878` backbone/parent/geometry、V19 和 qmask epoch33 的只读保护 inode 均再次核验存在。
epoch 3/4 收据间隔 `27:22`，下一完整检查窗口约为 `15:53--15:55 CST`。

epoch 5 完整收据于 `15:53:18 CST` 发布，REC 为
`5384/4306 = 0.566260/0.452882`，Mask 为
`5536/4534 = 0.582247/0.476862`，mIoU 为 `0.407828`，五项均未刷新。epoch 5 当前只作为
`latest` 保留以支持中断恢复；下一轮 checkpoint 成功发布后，如果它仍不承担任何分项 best，
retention 会自动删除该 inode。run best 仍为 epoch 4 的 REC 两档、Mask@0.25/mIoU，以及
epoch 2 的 Mask@0.50。相邻回执间隔 `27:17`，下一完整检查窗口约为 `16:20--16:22 CST`。

### ScanRefer Unique/Multiple 正式指标合同（2026-08-05）

按新增报告要求，后续正式结果除 overall REC/Mask 外，必须同时报告 ScanRefer Unique 与
Multiple 的 Acc@0.25/0.50。现有 evaluator 原本已经用部署时最终 position score 精确累计并在
日志打印这些 subgroup，但 `eval_metrics_epoch_*.json` 只保存 overall。现已将
`position_subgroups.unique|multiple` 加入同一原子 receipt，每组保存
`sample_count/hits025/hits050/acc025/acc050`。导出时强制要求两组分母之和等于 overall
sample count，两个阈值的 subgroup hits 之和分别等于 learned-selector overall hits；任一比率与
hits/分母不一致均 fail-closed。V47 正式 epoch-80 completion audit 已启用
`--require-position-subgroups`，相关 evaluator、receipt、completion、queue 定向回归为
`92 passed`。

当前正在运行的 Python 进程不会热加载新代码，因此 epoch 1--5 的原始 v1 receipt 不作事后改写，
其日志中的 exact subgroup counters 作为权威证据。当前 overall 最佳 epoch 4 对应：Unique
Acc@0.25/0.50 为 `1200/1419 = 0.845666`、`1022/1419 = 0.720226`；Multiple 为
`4209/8089 = 0.520336`、`3308/8089 = 0.408950`。后处理正式评估与 V47 都是之后新启动的
进程，会直接发布包含这些字段的结构化 receipt。

### V48 query-superpoint 空间 Mask 残差（2026-08-05）

V42/V43 只能按 query 调整两路 Mask 的融合权重和统一 logit bias，同一个 query 内所有
superpoint 获得相同偏移，无法直接修正边界或局部漏分。V48 在保留训练内 set-attention query
重排、box/mask 联合质量预测、逐 query alpha/bias 以及候选 Focal/Dice/Lovasz 目标的基础上，
新增低秩 query-superpoint 动态残差：query 与每个样本的 superpoint feature 分别经 LayerNorm
和两层投影，在低秩空间做缩放点积，再以 `2*tanh` 限幅。残差同时加入 text-mask 与
query-mask logits，因此融合后的最终 logit 精确增加同一空间残差，不改变原有两源权重语义。
输入默认 detach，joint-only 训练仍冻结 MCLN 主干。

空间头的 query 投影末层权重和 bias 均为零初始化，所以 V19 权重加载后的 step 0 严格保持
原网络定位与分割输出。真实受保护 V19 初始化审计收据位于
`/root/autodl-tmp/DATA_ROOT/output/v48_contract_audit/protected_v19_initialization.json`：原有
`1,228` 个张量全部相等，common/changed/missing/unexpected/shape-mismatch 为
`1228/0/34/0/0`；V48 joint head 恰有 `34` 个 state、`176,979` 个参数，零残差输出头审计通过。

新增真实 `MCLN.forward` 集成测试，而不只测试孤立 refiner：step 0 的最终
`last_pred_masks/sp_last_pred_masks` 保持不变；激活空间投影后最终 Mask 发生非零变化，候选
Focal/Dice/Lovasz 损失可从这两个 evaluator/criterion 共用字段反传至空间头。V48 checkpoint
profile 同时强制校验 mask calibration、source-mask evidence、空间头开关、hidden dim `32`、
最大残差 `2.0`、34 个 optimizer state 和 `176,979` optimizer numel。smoke profile 要求空间
残差 mean/max 均为有限正数，防止空间头虽然存在但训练塌缩为恒等映射。包含 SourceMoE、MCLN
接线、Mask 融合、冻结优化器、初始化/checkpoint/smoke 审计和单阶段队列的扩大回归为
`383 passed`。

V48 当前只是通过结构与训练路径合同，尚无真实验证集提升，不能替代已预注册的 V47 正式队列。
晋级条件是先完成真实小样本 smoke 并通过非零梯度、非塌缩空间残差、完整 checkpoint 和指标
收据门禁；随后才可占用四卡进行正式对照。对比时固定同一 V19 initializer、相同 source
arbiter、batch/epoch/seed，至少同时报告 Overall/Unique/Multiple REC@0.25/0.50 以及 Mask
Acc@0.25/0.50/mIoU，避免只凭分割单项波动选型。

单阶段 epoch 6 完整收据于 `16:20:52 CST` 发布，REC 为
`5431/4330 = 0.571203/0.455406`，其中 Acc@0.25 刷新本 run best，Acc@0.50 与 epoch 4
精确并列。Mask 三项全部刷新为 `5587/4586 = 0.587610/0.482331`、mIoU `0.411946`。
同一最终 position score 的 Unique Acc@0.25/0.50 为
`1208/1419 = 0.851304`、`1027/1419 = 0.723749`；Multiple 为
`4223/8089 = 0.522067`、`3303/8089 = 0.408332`。Mask 的 Unique 两档为
`1249/1419 = 0.880197`、`1012/1419 = 0.713178`，Multiple 两档为
`4338/8089 = 0.536284`、`3574/8089 = 0.441835`。

retention 在 epoch 6 checkpoint 原子发布后删除了无任何 best/latest 职责的 epoch-5 inode。
epoch 6 inode 由 latest、REC@0.25 和 Mask 三项共 6 个 hardlink 保护；epoch 4 inode 仅继续承担
REC@0.50 best。相邻 epoch-5/6 收据间隔约 `27:35`，下一完整检查窗口约为
`16:48--16:50 CST`，此前不轮询。

V48 四卡 smoke 接力入口为
[queue_v48_spatial_mask_smokes_after_v47.sh](../scripts/queue_v48_spatial_mask_smokes_after_v47.sh)。
它于 `16:28:15 CST` 在 tmux `mcln_v48_spatial_smoke_queue` 启动，主 PID `441502`，显式绑定
V47 supervisor PID `385263` 并通过 `tail --pid` 事件等待，不读取训练日志、不占 GPU，也不会与
现有单阶段训练、后处理或 V47 抢卡。V47 全部退出且 GPU 空闲后，四卡分别运行
`cmw/clw/K = 0.10/0/8`、`0.25/0/16`、`0.25/0.05/16`、`0.25/0.10/16`；所有组固定
V19、V48 hidden `32`、空间残差上限 `2.0`，因此只比较候选 Mask 监督强度。

每组必须通过含 Unique/Multiple 的完整训练收据、V48 `1228/0/34` checkpoint 合同、34 个
optimizer state、非零 query 重排/alpha/bias/source evidence/空间残差、候选覆盖及相应 Lovasz
门禁。summary 成功发布后仅删除四组 debug `.pth`，保留日志、指标和审计；队列未预注册 V48
正式长训，避免在 smoke 尚无真实提升证据时占用四卡。新增队列静态合同、shell 语法和 V48
相关审计/forward 回归为 `76 passed`。

V48 端到端消费路径审计同时发现 V47 selector 接线缺口：MCLN 已能用 selector 生成
`selected_source_scores` 和 joint reranker 输出，但 joint-only fast loss 的通用锚点合同仍沿用
`moe_shared_source/moe_shared_query/moe_valid_mask`，旧 selector 输出没有这三个字段。因此 V47
若按原代码启动，会在首个训练 batch 以缺少 `moe_shared_source` fail-closed，而不是静默训练；
当前 V47 尚处于事件等待状态，未产生坏权重，正在运行的单阶段 epoch 1--6 也没有启用 joint
reranker，不受该问题影响。

现已让 selector 从同一 `source_choice_batch` 发布通用训练合同：共享源固定为配置中的
`source_moe_shared_source`（当前为 `default`），共享 query 是该源在有效 query 内的 argmax，
valid mask 与 joint reranker 使用的 mask 逐位相同；缺源、shape/device、有限性或空有效行均立即
报错。真实轻量 `MCLN.forward` 输出随后直接进入正式
`compute_hungarian_loss(..., joint_query_quality_train_only=True)`，完整 joint listwise/quality、
候选 Focal/Dice/Lovasz 均执行并反传至空间头。SourceMoE、selector、冻结优化器、Mask、两级队列
和审计扩大回归为 `376 passed`。这项修复会由之后新启动的 V47/V48 进程加载。

正式 subgroup 收据进一步覆盖 Mask：`mask.position_subgroups.unique|multiple` 与 REC 使用同一
`sample_count/hits025/hits050/acc025/acc050` 结构。导出器分别强制 Unique+Multiple 分母等于
Overall `sample_count`，Mask 两档 hits 之和等于 Overall Mask hits；完成门禁启用
`--require-position-subgroups` 时会同时要求 REC 和 Mask 两套结构，任何缺字段、比例与命中数不符、
两档非嵌套或分组不能还原 Overall 都 fail-closed。当前长训进程不热加载该变更，因此 epoch 1--6
继续以日志 exact counts 为 Mask subgroup 权威证据；后续新进程直接写结构化 JSON。相关导出、
解析、completion、queue 回归为 `54 passed`。

V48 smoke 的空间激活门禁不再只检查 residual mean/max：新增
`mask_spatial_superpoint_std_mean`（每个 query 内跨 superpoint 的局部变化）与
`mask_spatial_query_std_mean`（同一场景跨 query 的条件变化），两项都必须 finite 且严格大于0。
这样统一常数偏移不能冒充空间修正，也不能重复 V42 已有的 query logit bias。真实 MCLN fast-loss
测试使用通道与点位置均非退化的确定性 feature，证明两种方差非零且候选损失仍反传；另有反例
确认 residual 非零但 superpoint 方差为0时 V48 smoke 必须失败。相关端到端与队列回归为
`205 passed`。

单阶段 epoch 7 完整收据于 `16:48:10 CST` 发布，五项同时刷新本 run best：REC 为
`5438/4344 = 0.571939/0.456878`，Mask 为
`5593/4596 = 0.588241/0.483382`、mIoU `0.412678`。定位 Unique Acc@0.25/0.50 为
`1214/1419 = 0.855532`、`1037/1419 = 0.730796`，Multiple 为
`4224/8089 = 0.522191`、`3307/8089 = 0.408827`。Mask Unique 两档为
`1249/1419 = 0.880197`、`1022/1419 = 0.720226`，Mask Multiple 为
`4344/8089 = 0.537026`、`3574/8089 = 0.441835`。

epoch-7 checkpoint inode `4374907969` 由 latest 和五项 best 共 7 个 hardlink 保护。由于五项均
严格超过先前 best，epoch 4 与 epoch 6 不再承担保留职责，retention 已删除它们；run 目录当前
只有 epoch-7 一个实际 checkpoint inode。历史 `0.582878` backbone/parent/geometry、V19 和
qmask epoch33 仍位于只读保护目录。epoch-6/7 收据间隔约 `27:18`，下一完整检查窗口约为
`17:15--17:17 CST`。

### V49 联合质量驱动的逐 query 自适应多源融合（2026-08-05）

V23--V39 的 dense mixer 会直接替换 V19 的源融合路径，历史真实实验均未超过 V19；V48 又只在
最终 query 重排和 Mask 头使用联合质量，不能改变每个 query 对各源的依赖。V49 因此采用较小的
零残差扩展：保持 V19 fallback/parent score 不动，在 V48 set-attention hidden 与六维绝对质量
证据（box/mask 两档概率和各自 IoU）上，对同一 query 的全部 source rank 使用共享编码器。源身份
只由 shared-source flag 表示，routed sources 共用参数，因此交换 routed source 顺序时最终修正
严格等变，不把 ScanRefer 上的固定源编号写进策略。

router 为每个 query 输出 dense source weights；权重熵的指数作为有效源数诊断，既能收敛到近似
单源，也能保留多源组合。mixed rank 与 V19 parent rank 的差只通过零初始化 strength head 写入
联合 reranker residual logit。因此加载 V19 后 source correction、query residual、Mask alpha/
bias 和空间残差均严格为零，最终 REC/Mask 输出保持 step-0 identity；第一步 strength head 可获
非零梯度，更新后 source router 也获得非零梯度。无效源权重强制为零，每个有效 query 至少一个
有效源、每个样本必须存在 shared source，否则 fail-closed。

V49 在 V48 的 34 个 state/`176,979` 参数上新增 11 个 state/`52,481` 参数，合计 45 个 state、
`229,460` 个 joint-only 可训练参数。受保护 V19 真实初始化收据位于
`/root/autodl-tmp/DATA_ROOT/output/v49_contract_audit/protected_v19_initialization.json`：
common/changed/missing/unexpected/shape mismatch 为 `1228/0/45/0/0`，所有旧 tensor 逐元素相等，
新输出头零初始化，审计通过。模块 identity、两步梯度、routed-source 置换等变、真实轻量
`MCLN.forward`、SourceMoE 历史路径、checkpoint profile、分组收据与 queue 扩大回归为
`395 passed`；Python compile 与相关 shell 语法均通过。

V49 四卡 smoke 入口为
[queue_v49_adaptive_source_mix_smokes_after_v48.sh](../scripts/queue_v49_adaptive_source_mix_smokes_after_v48.sh)，
tmux 为 `mcln_v49_adaptive_source_mix_after_v48`，初始主 PID `471624`。它通过 `tail --pid=441502`
绑定 V48 队列，不读取日志、不占 GPU；V48 完整退出且 GPU 空闲后才在四卡分别运行 V48 的四组
候选 Mask 监督配置，并额外启用 adaptive source mixing。每组必须通过含 REC/Mask
Unique/Multiple 的完整收据、V49 `1228/0/45` checkpoint 合同、45 个 optimizer state、非零
query/Mask/空间/source-mix 诊断；其中 source mix residual、learned router residual 和跨 query
weight std 必须大于0，effective source count 必须至少为1。该门禁允许策略按 query 收敛到近似
单源或保留多源，不能用固定权重冒充自适应路由。smoke 只提供真实可训练性与方向筛选证据，未
预注册正式长训。

单阶段 epoch 8 完整收据于 `17:15:27 CST` 发布，全部低于 epoch 7：REC 为
`5422/4324 = 0.570256/0.454775`，Mask 为
`5568/4571 = 0.585612/0.480753`、mIoU `0.410142`。定位 Unique Acc@0.25/0.50 为
`1211/1419 = 0.853418`、`1028/1419 = 0.724454`，Multiple 为
`4211/8089 = 0.520584`、`3296/8089 = 0.407467`。Mask Unique 两档为
`1246/1419 = 0.878083`、`1002/1419 = 0.706131`，Mask Multiple 为
`4322/8089 = 0.534306`、`3569/8089 = 0.441216`；分组 hits 分别精确还原 Overall。

retention 未改写任何 best：epoch-7 inode `4374907969` 继续由五项 best 和 epoch-7 文件共 6 个
hardlink 保护；epoch-8 inode `4374906354` 仅由 `ckpt_epoch_8.pth` 与 `ckpt_epoch_last.pth`
两个 hardlink 承担恢复职责，下一轮发布后若不刷新 best 会自动清理。epoch-7/8 收据间隔
`27:17`，下一完整检查窗口约为 `17:42--17:44 CST`，此前不轮询。

### 短 smoke 优先的实验队列重排（2026-08-05 17:34 CST）

原队列会在 V47 四组 smoke 后立即占用四卡进行 80 epoch 正式长训，V48/V49 smoke 因而至少延后
一天；V48/V49 均从受保护双阶段 V19 独立初始化，不消费 V47 正式权重，所以这项串行依赖没有
技术必要。为优先筛选网络架构并减少无效长训，现将
`RUN_FORMAL_AFTER_SMOKE` 默认值从 `1` 改为 `0`：当前单阶段 100 epoch 与既定后处理完成后，依次
执行 V47 四组短 smoke、V48 四组短 smoke、V49 四组短 smoke，收齐真实验证证据后再选择下一次
正式长训，不再预先锁定 V47。

队列同时升级为 fail-closed 证据链：V48 必须读到 V47 base/candidate 两份 summary 且均
`pass=true` 才启动；V49 必须读到 V48 summary 且 `pass=true` 才启动。三个纯等待 supervisor 已
从下游到上游安全重启，未触碰 PID `357439` 的四卡训练或 PID `368662` 的后处理等待器；新 PID
分别为 V47 `473915`、V48 `473933`、V49 `473965`，绑定关系为
`368662 -> 473915 -> 473933 -> 473965`。随后补齐 V47 每组 smoke 的
`--require-position-subgroups` completion audit 并再次加载脚本，当前有效 PID 为 V47 `475702`、
V48 `475719`、V49 `475752`，最终绑定关系为
`368662 -> 475702 -> 475719 -> 475752`。相关 completion/receipt/queue 回归为 `48 passed`，shell
syntax 通过。

单阶段 epoch 9 完整收据于 `17:42:55 CST` 发布。REC 为
`5412/4366 = 0.569205/0.459192`：Acc@0.25 低于 epoch 7，但 Acc@0.50 超过 epoch 7 的
`0.456878`，刷新本 run best。定位 Unique Acc@0.25/0.50 为
`1210/1419 = 0.852713`、`1043/1419 = 0.735025`，Multiple 为
`4202/8089 = 0.519471`、`3323/8089 = 0.410805`。Mask 为
`5561/4557 = 0.584876/0.479281`、mIoU `0.410001`；Mask Unique 两档为
`1244/1419 = 0.876674`、`1014/1419 = 0.714588`，Mask Multiple 为
`4317/8089 = 0.533688`、`3543/8089 = 0.438002`。REC/Mask 的 Unique+Multiple hits 均精确
还原 Overall。

retention 已删除不承担职责的 epoch-8 checkpoint inode。epoch-7 inode `4374907969` 继续由
REC@0.25、Mask 三项 best 与 epoch-7 文件共 5 个 hardlink 保护；epoch-9 inode `4382675967`
由 REC@0.50 best、epoch-9 与 latest 共 3 个 hardlink 保护。历史受保护的 `0.582878` backbone、
V19、qmask epoch33 等均未改变。epoch-8/9 收据间隔约 `27:28`，下一完整检查窗口约为
`18:10--18:12 CST`，此前不轮询。

### 训练期间的旧权重清理（2026-08-05 17:52 CST）

为保证 100 轮单阶段训练完成后后处理队列的磁盘门禁（至少 7 GiB 可用），清理了已结束且没有活动进程依赖的旧实验目录中的 `.pth` 副本：旧 query-mask smoke/full runs，以及 V7、V12、V17、V19、V28 的历史 SourceMoE runs。日志、JSON 收据和审计文件保留，便于比较；当前单阶段 run、`preserved_best`、`rec_reranker` 和所有保护目录未触碰。删除前确认没有进程引用这些目录。

保护核验通过：`scanrefer_best_backbone_acc025_0.582878_component.pth`、对应 parent/geometry reranker、V19 `0.581195/0.598233`、qmask `Mask@0.50=0.491691`、V28 `Mask@0.50=0.491481` 均仍存在；清理后 `/root/autodl-tmp` 可用约 9.3 GB。后续 retention 继续只保留当前 run 的 latest、各项 best 及恢复所需 inode。

单阶段 epoch 10 完整日志收据于 `18:10:04 CST` 发布。REC 为
`5433/4348 = 0.571414/0.457299`；定位 Unique Acc@0.25/0.50 为
`1213/1419 = 0.854827`、`1045/1419 = 0.736434`，Multiple 为
`4220/8089 = 0.521696`、`3303/8089 = 0.408332`。Mask 为
`5586/4582 = 0.587505/0.481910`、mIoU `0.411510`；Mask Unique 两档为
`1245/1419 = 0.877378`、`1011/1419 = 0.712474`，Mask Multiple 为
`4341/8089 = 0.536655`、`3571/8089 = 0.441464`。REC/Mask 的 Unique+Multiple
hits 均精确还原 Overall。该长训进程在 subgroup 导出器更新前启动，因此 epoch 10 的
JSON 只保留主计数；日志中的 subgroup exact counts 已核对，训练结束后的新进程官方评估
将直接写入包含 `position_subgroups` 和 `mask.position_subgroups` 的结构化收据。

单阶段 epoch 11 完整日志收据于 `18:37:35 CST` 发布。REC 为
`5414/4336 = 0.569415/0.456037`；定位 Unique Acc@0.25/0.50 为
`1212/1419 = 0.854123`、`1042/1419 = 0.734320`，Multiple 为
`4202/8089 = 0.519471`、`3294/8089 = 0.407220`。Mask 为
`5566/4562 = 0.585402/0.479806`、mIoU `0.410263`；Mask Unique 两档为
`1246/1419 = 0.878083`、`1010/1419 = 0.711769`，Mask Multiple 为
`4320/8089 = 0.534059`、`3552/8089 = 0.439115`。REC/Mask 的 Unique+Multiple
hits 均精确还原 Overall；retention 仍保护 epoch 7 的 REC@0.25、Mask 三项 best 和
epoch 9 的 REC@0.50。

单阶段 epoch 12 完整日志收据于 `19:05:09 CST` 发布。REC 为
`5414/4334 = 0.569415/0.455827`；定位 Unique Acc@0.25/0.50 为
`1201/1419 = 0.846371`、`1033/1419 = 0.727977`，Multiple 为
`4213/8089 = 0.520831`、`3301/8089 = 0.408085`。Mask 为
`5562/4569 = 0.584981/0.480543`、mIoU `0.410610`；Mask Unique 两档为
`1243/1419 = 0.875969`、`1009/1419 = 0.711064`，Mask Multiple 为
`4319/8089 = 0.533935`、`3560/8089 = 0.440104`。REC/Mask 的 Unique+Multiple
hits 均精确还原 Overall，retention 仍未改写任何 best。

### V49 smoke 后正式双阶段续跑队列（2026-08-05 19:17 CST）

新增 `scripts/queue_double_stage_v49_formal_after_smoke.sh`，独立等待 V49 smoke
summary，不提前占用 GPU。只有 V49 `pass=true` 且四个变体均通过完整 REC/Mask
Unique/Multiple 收据、V49 checkpoint 合同和非零 source-mix 诊断时才继续。队列按
smoke 的 learned REC@0.25/0.50 综合比例优先、Mask 四项和 mIoU 次优选择变体，并从
变体名解析 candidate Mask/Lovasz 权重及 top-k；不会把固定 ScanRefer 源编号写入正式
模型。随后从只读 V19 初始化 `MODEL_STAGE=two`、`SOURCE_ARBITER=moe`、adaptive
source mixing、Mask calibration 和 spatial refiner，四卡训练 80 epoch，最终强制
`--require-position-subgroups` completion audit。supervisor 当前 PID `517499`，只
等待 V49 PID `475752`，尚未启动 GPU 训练。

### 单阶段长训 epoch 13 分组收据（2026-08-05 19:32 CST）

epoch 13 验证已完成。REC 为 `5399/4344 = 0.567840/0.456880`；定位 Unique
Acc@0.25/0.50 为 `1206/1419 = 0.849894`、`1048/1419 = 0.738548`，Multiple 为
`4193/8089 = 0.518358`、`3296/8089 = 0.407467`。Mask 为
`5549/4541 = 0.583614/0.477598`，mIoU `0.408628`；Mask Unique 两档为
`1243/1419 = 0.875969`、`1005/1419 = 0.708245`，Mask Multiple 为
`4306/8089 = 0.532328`、`3536/8089 = 0.437137`。两套 Unique+Multiple hits 均精确
还原 Overall。该轮仍由 subgroup 导出器更新前启动的长训进程执行，因此旧版
`eval_metrics_epoch_13.json` 只含主计数；日志中的 exact subgroup counters 已完成校验，后续
新评估进程会写入 `position_subgroups` 与 `mask.position_subgroups`。

### V49 正式训练架构终检补强（2026-08-05 19:43 CST）

审计正式双阶段队列后确认，V19 checkpoint 会在模型构建前强制继承三源 schema：共享源
`default`，以及 routed sources `contrastive_text`、`mask_text`；因此 V49 并非使用 launcher
中的两源默认占位值，而是对共享源和两个可路由源进行逐 query 自适应组合。

正式队列原有 completion audit 能证明 epoch 80、9508 个验证样本以及 REC/Mask 两套
Unique/Multiple 收据完整，但不能单独证明 V49 新模块确实完成了全部优化步骤。现增加第二道
架构终检：从正式日志分别提取 epoch 1 与最终 epoch 的真实 steps-per-epoch，要求两者唯一且
一致，再计算精确 optimizer step；随后以只读 V19 为 baseline 执行 V49 checkpoint audit，强制
检查 `1228/0/45` tensor 合同、45 个 optimizer state、`229460` 个 moment 参数、全部 finite/
nonzero，以及 mask calibration、空间 Mask refiner、自适应 source mixing 的配置合同。最终收据为
`formal_v49_checkpoint_audit.json`；任一条件不满足时正式队列 fail closed，不会宣称实验完成。

### 单阶段长训 epoch 14 分组收据（2026-08-05 19:59 CST）

epoch 14 REC 为 `5383/4299 = 0.566155/0.452146`；定位 Unique Acc@0.25/0.50 为
`1210/1419 = 0.852713`、`1047/1419 = 0.737844`，Multiple 为
`4173/8089 = 0.515886`、`3252/8089 = 0.402027`。Mask 为
`5527/4521 = 0.581300/0.475494`，mIoU `0.407398`；Mask Unique 两档为
`1241/1419 = 0.874560`、`999/1419 = 0.704017`，Mask Multiple 为
`4286/8089 = 0.529855`、`3522/8089 = 0.435406`。REC/Mask 两套分组 hits 均精确还原
Overall。本轮没有刷新 retention；epoch 13 的无职责 checkpoint 已删除，epoch 7 继续承担
REC@0.25 和 Mask 三项 best，epoch 9 承担 REC@0.50 best，epoch 14 仅承担 latest/恢复职责。
历史 `0.582878` 受保护权重未改变。

### V50-style 自适应源质量对齐监督（2026-08-05 20:03 CST）

V49 原自适应 mixer 具有共享 source encoder、逐 query source weights、single/multi-source
effective count 和 step-0 identity，但 source router 主要通过最终 query listwise loss 间接学习。
为增强跨数据集泛化，新增可消融的 `source mix alignment loss`：训练时先用真实 Box/Mask IoU
构造 box-tier-first 的联合 query quality，再比较每个源的 query rank 与联合质量 rank 的一致性；
一致性越高，该 query 上该源的目标权重越大。目标仅依赖 rank agreement 和 source validity，
不使用源名称或 ScanRefer 样本类别，因此交换 `contrastive_text` 与 `mask_text` 后损失严格不变，
也可直接迁移到 Nr3D/Sr3D。默认 loss weight 为 0，旧 V41--V49 配置行为不变。

新增训练参数：`joint_query_quality_source_mix_loss_weight` 和独立的
`joint_query_quality_source_mix_alignment_temperature`；后者与 mixer router softmax 的
`joint_query_quality_source_mix_temperature` 分离，当前分别使用 `0.25` 与 `0.5`。针对性测试证明
即使模型仍处于 step-0 identity，alignment loss 也能向 source router 产生 finite、nonzero
gradient；fast training path 和 routed-source permutation invariance 均通过。

V49 四卡面板同时重新设计：V48 完成后先按 REC@0.25/0.50 主目标、Mask/mIoU 次目标自动选择
最佳 candidate Mask/Lovasz/top-k 配置，V49 固定该分割配置，只并行比较 source-mix loss 权重
`0.00/0.10/0.25/0.50`，避免再次重复扫描 Mask 超参。每个 checkpoint audit 还会核验实际 loss
weight 与 alignment temperature；正式双阶段队列从 V49 summary 继承胜出权重。V49 CPU
supervisor 最终以 PID `541439` 重新绑定 V48 PID `475719`，正式 supervisor PID `541453` 绑定
新 V49 PID，未触碰四卡训练或 V47/V48 队列。

同一轮参数继承审计还修复了一个旧的量级错误：smoke 变体 `clw005/clw010` 实际训练权重是
`0.05/0.10`，原正式选择脚本却按 `/1000` 解析为 `0.005/0.01`。V49 从 V48 继承配置以及正式
双阶段从 V49 继承配置现统一按 `/100` 解析，确保正式训练精确复现 smoke 胜出的 Lovasz 权重，
不会再被静默缩小十倍。

### 改进版 BUTD 三创新与 MCLN/MoE 的继承边界（2026-08-05 20:30 CST）

本项目所称“改进版 BUTD 三创新”固定指 `SACR/RAPF/QAHNL`；代码中的正式缩写是
`QAHNL`（参数 `use_qahnl`），与口头使用的 `AQHNL` 指同一模块。它们不是 BUTD-DETR
原论文的 box stream、detection prompt、span alignment 三项，也不是更早的
Type-Embedding/Span-Direct 或 S2S/ACD/DHC 叙事。

代码审计确认，不能表述成“MCLN 原样移植 SACR、RAPF、QAHNL 后再增加 MoE”：

- BUTD 的 SACR 由父项目 `models/sacr_head.py::SACRHead` 实现，读取 structured slots，生成
  target-attribute 与 relation-anchor structured score。当前 `models/mcln.py` 没有导入
  `SACRHead`/`StructuredSlotBuilder`，三源 `default/contrastive_text/mask_text` 中也没有真正的
  SACR structured source。因此 SACR 尚未直接迁移；MCLN 的 relation/modify token 聚合不能冒充
  anchor-conditioned compositional reasoning。
- BUTD 的 RAPF 由 `models/reliability_fusion.py::ReliabilityFusion` 实现，用 entropy、top-1
  margin、base/structured disagreement、JS divergence、parse confidence 和 anchor reliability
  预测 gate，再把 structured residual 注入 base/quality anchor。MCLN 没有复用该类；V19
  `SourceMoE + fallback gate` 和 V49 adaptive source mixer 继承了“可靠时才覆盖共享源”的思想，
  但把 sample/source fusion 升级为逐 query 三源路由、共享 default anchor 和保守 fallback，属于
  MCLN-specific 后继实现，不是 RAPF 代码移植。
- BUTD 的 QAHNL 由 `QualityHead` 与 `_qahnl_losses` 实现，预测候选 Box IoU，并在 source top-k
  中按 IoU gap 构造 hard negatives 和自适应 margin。MCLN 没有调用这两个实现；V47--V49 的
  Joint Query Quality 改为同时预测 Box/Mask 的 `>0.25`、`>0.50` 与连续 IoU，使用 box-tier-first
  listwise、anchor protection、candidate Mask Focal/Dice/Lovasz 和 source-rank alignment。这是
  QAHNL 的质量/困难候选思想在联合 REC/RES 上的扩展重写。
- 真正从 BUTD/UCRA 直接迁入 MCLN 的是通用 source-choice 接口：
  `source_choice_adapter.py`、`source_choice_selector.py`、训练期 GT-IoU source target 与推理期
  deployable-source 选择。后续 SourceMoE 在这个接口上从“每样本选一个完整 source”扩展到
  “每个 query 自适应单源或多源”，V49 再用联合 Box/Mask 质量直接监督 source weights。

因此当前安全的论文关系是：SACR 负责在 BUTD 中生成 structured expert；RAPF 是早期可靠性
融合基础；QAHNL 提供质量和 hard-negative 学习原则；MCLN 先迁移统一 source arbitration
接口，再用 SourceMoE 与 Joint Query Quality 对 RAPF/QAHNL 原则做 query-wise、REC/RES 联合
扩展。若要声称三模块全部跨 backbone 迁移，后续还必须把真正的 SACR structured source 接入
MCLN/MoE，并进行独立消融；当前代码证据不支持该表述。

同一时间窗口发布的单阶段 epoch 15 完整收据为：REC Overall
`5381/4302 = 0.565944/0.452461`，Unique 为
`1214/1419 = 0.855532`、`1049/1419 = 0.739253`，Multiple 为
`4167/8089 = 0.515144`、`3253/8089 = 0.402151`。Mask Overall 为
`5551/4533 = 0.583824/0.476756`，mIoU `0.409038`；Mask Unique 为
`1248/1419 = 0.879493`、`1011/1419 = 0.712474`，Multiple 为
`4303/8089 = 0.531957`、`3522/8089 = 0.435406`。两套分组 hits 均精确还原 Overall。
本轮未刷新 best；retention 已删除无职责 epoch 14 权重，只保留 epoch 7、epoch 9 和 epoch 15
latest。历史 `0.582878` 受保护权重未改变，epoch 16 已自动开始。

## 最新 MCLN-MoE 网络框架完整说明（2026-08-05）

### 1. 版本定义与结论边界

当前“最新网络框架”特指下面这条实际代码路径：

```text
two-stage MCLN backbone
  -> three-source score pool
  -> SourceMoE + protected V19 fallback parent
  -> Joint Query Quality reranker
  -> query-wise adaptive source mixing
  -> query-wise Mask calibration + spatial Mask residual
  -> shared final query for REC and RES
```

实验脚本仍将该架构命名为 `V49`。随后加入的 source-mix alignment loss 在记录中称为
“V50-style supervision”，但它已经合并进 V49 smoke 和正式训练选择流程，不是另一个已经发布的
checkpoint。论文中可暂称 **MCLN-JQMoE**，版本号只用于内部实验管理。

需要严格区分三种状态：

1. V49 网络代码、损失、checkpoint 合同、step-0 identity 和梯度测试已经实现并通过。
2. V49 四卡 smoke 与 80 epoch 双阶段正式训练仍在排队，尚未产生正式全验证指标，因此不能把它
   写成“已经优于 V19”。
3. 当前可发布的 network-only parent 仍是 V19：REC `0.5811948/0.4653976`，Mask
   `0.5982331/0.4913757/0.4186131`；历史后处理最好 REC
   `0.582878/0.486012` 另行保护。

当前 V49 正式模型从只读 V19 checkpoint 初始化，继承的真实 source schema 为：

```text
shared source: default
routed source 1: contrastive_text
routed source 2: mask_text
```

这不是 launcher 中两源默认值。正式队列在模型构建前从 checkpoint 恢复三源 schema，并用
checkpoint audit 强制验证。

### 2. 任务输入、输出与双阶段含义

输入包括：

- 点云 `P in R^(N x (3+C))`，当前 ScanRefer 使用坐标与颜色；
- 指代表达文本 `T`；
- 双阶段模式下的外部检测框、检测类别和有效性 mask；
- 训练时的目标框、目标点级 Mask、语言 token positive maps；
- 点到 superpoint 的映射。

模型对每个样本产生 `Q=256` 个共享候选 query。每个 query 同时对应：

- 一个 3D box `b_q=(cx,cy,cz,w,h,d)`；
- 一个 REC 排序分数；
- 一个 superpoint Mask；
- Box/Mask 两组质量估计；
- 三个 source 的逐 query 权重。

最终 REC 与 RES 必须选择同一个 query，避免“定位选 query A，分割却取 query B”造成评价身份错位。
双阶段的含义不是 SourceMoE 有两个训练阶段，而是 MCLN 使用外部 detector 产生的 box/class stream；
单阶段则使用 `--joint_det`，不依赖该外部 box stream。后面的 SourceMoE、Joint Query Quality 和
Mask 校准接口都建立在最终 query 集上，结构上可以复用于两种模式。

### 3. 总体前向流程

```text
Point cloud ----------------> PointNet++ ----------------------> 1024 visual seeds, 288-d
                                                                   |
Text -----------------------> frozen RoBERTa -> projector --------+--> 3-layer bi-modal encoder
                                                                   |
Detected boxes/classes -----> geometry/class embedding -----------+   (two-stage only)
                                                                        |
                                                                        v
Objectness top-k sampling --------------------------------------> 256 initial queries
                                                                        |
                                                                        v
                                      6-layer multimodal decoder <---- points/text/detections
                                             |             |
                                      256 boxes       query features
                                             |             |
                                             +------+------+
                                                    |
                    +-------------------------------+-----------------------------+
                    |                               |                             |
             default score                contrastive_text score          mask_text score
                    |                               |                             |
                    +-------------------------------+-----------------------------+
                                                    |
                                      SourceMoE + V19 safe fallback
                                                    |
                                            V19 parent scores
                                                    |
                                  Joint Query Quality set attention
                         +--------------------------+--------------------------+
                         |                          |                          |
                 adaptive source mix        Box/Mask quality heads       Mask calibration
                         |                          |                    + spatial residual
                         +--------------------------+--------------------------+
                                                    |
                                       final shared query ranking
                                                    |
                                      same query -> REC box + RES Mask
```

### 4. 基础 MCLN 主干

#### 4.1 点云、文本和检测框编码

视觉分支使用 PointNet++，从原始点云产生 `1024` 个 seed，特征维度为 `288`。文本分支使用冻结的
RoBERTa，token 特征经线性投影、LayerNorm 和 Dropout 映射到 `288` 维。冻结 RoBERTa 可减少后续
小数据微调造成的语言漂移，但文本 projector 仍可训练。

双阶段检测框流把每个检测框拆成两部分：

- 6 维框几何经 learned position embedding 得到 `128` 维；
- 检测类别经预存类别语义 embedding 和线性层得到 `160` 维。

两者拼接为 `288` 维 detected feature。三层双向跨模态 encoder 在视觉 seed、文本 token 和检测框
之间交互，同时使用点间空间位置编码。

#### 4.2 Query 初始化与 Box decoder

视觉 seed 先经过 objectness head，在 `1024` 个 seed 中选择 top `256` 作为初始 query。proposal
head 给出初始中心和尺寸，然后进入 6 层 multimodal decoder。每层 query 都读取点云、文本和双阶段
检测框信息，并输出中间框；最后一层产生正式的 `256` 个 box、token semantic logits、288 维
decoder query 和 64 维归一化 contrastive query。

#### 4.3 原始双源 Mask 头

Mask 不是简单从最终 box 裁剪得到，而是并行生成两个 logit source：

1. **query Mask**：最终 decoder query 经 `x_query`，与 superpoint feature 点积，得到
   `[Q,S]` 的 query-specific Mask logits。
2. **text Mask**：文本 token 经 3 层 SWA 从 superpoint feature 中吸收视觉信息，选择最相关 token
   后生成文本 Mask；原 MCLN 将其扩展到所有 query。

superpoint feature 由 seed Mask feature、半径邻域聚合和相对坐标编码共同构造。原始融合为：

```text
M_q = alpha * M_text,q + (1-alpha) * M_query,q
```

原始 `alpha` 主要是样本级标量。V49 将它升级为逐 query 可学习权重，并额外学习 logit bias 和
query-superpoint 空间残差。

### 5. 三源候选评分池

#### 5.1 `default` 共享源

`default` 使用最后一层 semantic token probability。语言被映射为 main target、modifier、pronoun、
relation 和 other entity 五类 token map：

```text
map_default = main + modifier + pronoun + relation - other
s_default(q) = sum_t p_sem(q,t) * map_default(t)
```

它是 MCLN 原生、最稳定的排序来源，因此被定义为永远存在的 shared source，而不是普通 routed
expert。

#### 5.2 `contrastive_text` 路由源

64 维 query 与 64 维 token 均做 L2 normalization，得到 cosine similarity matrix。它使用与
default 相同的 signed token map 聚合：

```text
s_contrastive(q) = sum_t cos(z_q, z_t) * map_default(t)
```

它更强调跨模态表示相似性，与 semantic soft-token classifier 的误差模式不同。

#### 5.3 `mask_text` 路由源

该源衡量同一 query 的 text Mask 与 query Mask 是否一致：计算 soft Dice、query Mask 平均置信度，
再使用原自适应 Mask 权重组合。它不是 GT Mask oracle，推理时只依赖网络自身输出，因此可部署。

#### 5.4 为什么先转为 rank

三个 source 的原始量纲不同，不能直接相加。每个 source 在每个场景内部转成 `[0,1]` rank。训练
使用 straight-through rank normalization：前向保持精确排序，反向用平滑 proxy 传梯度。缺失源的
validity 强制为 false、权重强制为 0；每个样本必须至少有一个有效 shared-source query，否则
fail closed。

### 6. SourceMoE 与 V19 安全 parent

#### 6.1 Shared-source sparse MoE

每个 query 的 router 输入为：

```text
z_q = [64-d query, 64-d pooled text, 6-d normalized box, all source ranks]
```

router 只对 routed experts 分配 top-k 权重，共享 `default` 不参与竞争。基础融合可写成：

```text
routed(q) = sum_s gate(q,s) * rank_s(q)
s_moe(q) = rank_default(q)
           + tanh(scale) * (routed(q) - rank_default(q))
           + delta_context(q)
```

其中 `delta_context` 来自跨 query 的轻量 self-attention reranker。`scale` 零初始化，因此新 MoE 在
第 0 步不会破坏 shared default。router 使用 load-balance 诊断，防止所有 query 永久塌缩到同一
expert，但最新 V49 joint-only 阶段冻结整个 SourceMoE，不再更新该 parent。

#### 6.2 V19 分层安全回退

历史实验证明“存在更好候选”和“模型实际选中的候选确实更好”是两件不同的事。V19 将切换拆为：

1. proposer/rank head 在 top candidate set 内提出候选；
2. row opportunity head 判断该样本是否存在值得切换的机会；
3. selected-query safety head 单独验证即将执行的那个 query；
4. 只有两个 margin 都为正才允许覆盖动态 anchor。

部署边界为：

```text
deployed_margin = min(row_opportunity_margin, selected_query_safety_margin)
switch = has_candidate and deployed_margin > 0
```

否则严格保持 parent query。该规则不扫描 ScanRefer validation threshold，所以比手工后处理更适合
迁移。V19 的问题是过于保守，正式验证只执行 5 次 correction，说明安全性高但 coverage 很低。
V49 不改写这条已训练路径，而把其输出作为 `parent_scores` 再做零残差联合质量学习。

### 7. V49 Joint Query Quality reranker

#### 7.1 输入状态与 set encoder

每个 query 构造 `2*64+24=152` 维 deployable rich state，主要包含：

- 64 维 query projection 和 64 维 target-text projection；
- 归一化 box center/size；
- main/modifier/pronoun/relation/other 分量分数；
- default/contrastive 原分数、rank、top-1 和 margin；
- seed objectness；
- Mask confidence、foreground ratio、text-query Mask Dice；
- query 与 target text cosine。

再拼接 parent rank、标准化 parent score、基础 Mask alpha 和 10 维无 GT Mask-source evidence，经过
`hidden=128` 的 projection、1 层 4-head self-attention 和 FFN。self-attention 没有注入 query
编号捷径，使模型可以比较同一场景中全部候选，而不是逐 query 独立打分。

#### 7.2 六维 Box/Mask 绝对质量头

每个 query 同时预测：

```text
P(Box IoU > 0.25), P(Box IoU > 0.50), predicted Box IoU,
P(Mask IoU > 0.25), P(Mask IoU > 0.50), predicted Mask IoU.
```

两档概率采用 ordinal 参数化，强制 `P(>0.50) <= P(>0.25)`。这比独立二分类更符合 IoU 阈值的
单调关系。

训练 target 使用 Box tier 优先的词典序质量：

```text
t_box  = 1[IoU_box>0.25] + 1[IoU_box>0.50]
t_mask = 1[IoU_mask>0.25] + 1[IoU_mask>0.50]
quality_target = 4*t_box + IoU_box
                 + lambda_mask*(2*t_mask + IoU_mask)
lambda_mask = 0.25
```

系数 4 保证低 Box tier 的 query 不能仅靠 Mask 好而越级覆盖更可靠的定位 query；在同一 Box tier
内，Mask 才参与细排。这直接对应本项目 REC `0.25/0.50` 主目标和 Mask 三项保护目标。

#### 7.3 最终 query 重排

模型预测一个直接 residual，并把中心化联合质量和 adaptive source-mix residual 一起注入：

```text
residual(q) = 1.25 * tanh(
    direct_logit(q)
    + centered_predicted_quality(q)
    + source_mix_residual(q)
)
s_final(q) = s_V19_parent(q) + residual(q)
```

最终层零初始化，所以加载 V19 时 `residual=0`，第 0 步的 REC query、Mask query 和五项指标都应
保持 parent。训练后才允许通过网络分数改变 query，而不是在 evaluator 中添加数据集专用规则。

### 8. 逐 query 自适应多源融合

这是相对原 V19 固定 source path 的核心新增模块。对于每个 query 和每个 source，使用同一个
source encoder 读取：

```text
[joint hidden, six quality values, source rank,
 source rank - shared rank, shared-source flag]
```

除 shared flag 外不提供 source ID embedding，因此两个 routed source 交换顺序时输出严格等变。
这避免模型记住“ScanRefer 上第二个 source 通常更好”之类的数据集捷径，并允许以后增删 expert。

router 产生 dense 权重：

```text
w(q,s) = softmax(rank(q,s)/0.5 + learned_residual(q,s))
mixed_rank(q) = sum_s w(q,s) * rank(q,s)
mix_delta(q) = strength(q) * (mixed_rank(q) - parent_rank(q))
```

`strength` 决定本 query 是否真正采纳 source mix。权重熵的指数作为 effective source count：接近
1 表示自适应单源，大于 1 表示多源组合。模型不强迫所有样本使用固定组合。

训练新增 source-rank alignment：先由真实 Box/Mask IoU 产生 joint-quality rank，再按各 source
rank 与该 target rank 的接近程度构造 soft target：

```text
p_target(q,s) proportional to
    exp(-abs(rank_s(q)-rank_quality(q))/tau_align), tau_align=0.25
L_mix = CE(p_target, w)
```

该 target 只依赖 rank agreement 和 source validity，不依赖 source 名称、Unique/Multiple 标签或
数据集名，因此可迁移到单阶段 ScanRefer、Nr3D 和 Sr3D。V49 smoke 比较
`lambda_mix in {0,0.10,0.25,0.50}`，正式训练继承胜出值。

### 9. 分割头专项优化

#### 9.1 Query-wise Mask source evidence

对 text Mask 与 query Mask 分别提取均值、标准差、置信度、前景比例，再计算两源概率 L1 差和硬
分歧率，共 10 维。它们不读取 GT，可在推理时直接使用。

#### 9.2 Alpha 与 bias 校准

联合 hidden 为每个 query 预测：

```text
alpha_q = clamp(alpha_base + delta_alpha_q, 0, 1)
delta_alpha_q in [-1,1]
bias_q in [-2,2]
```

同一个 `bias_q` 加到两个 Mask source，`alpha_q` 决定 text/query Mask 的组合。这样不同候选可按
自身证据选择更可靠的 Mask source，不再共享一个样本级 alpha。

#### 9.3 Query-superpoint 空间 residual

288 维 query 和 288 维 superpoint feature 分别投影到 32 维，点积产生 `[Q,S]` 的低秩空间修正，
经 `2*tanh` 限幅后同时加到两个 Mask source。query projection 的末层零初始化，因此初始空间
residual 为 0。

#### 9.4 候选级 Mask 监督

只监督 Hungarian matched query 会让最终 reranker 可能选到“box 尚可但 Mask 没训练好”的候选。
因此 V49 对以下集合的并集施加 Mask 监督：

```text
top-K by current final score UNION top-K by true Box IoU
```

对这些候选的融合 Mask 使用 Focal + Dice，并消融 Lovasz hinge。候选 Focal/Dice 权重、Lovasz
权重和 `K in {8,16}` 先由 V47/V48 smoke 选择，再交给 V49；这部分旨在同时提升 Mask@0.25、
Mask@0.50 和 mIoU，而不是只修最终阈值。

### 10. 最新训练目标

V49 joint-only 的核心损失为：

```text
L_joint = L_listwise
          + L_absolute_quality
          + 0.5 * L_anchor_protection
          + lambda_mix * L_source_alignment

L_mask = 10 * L_calibration_focal + 2 * L_calibration_dice
         + lambda_candidate * (10 * L_candidate_focal + 2 * L_candidate_dice)
         + lambda_lovasz * L_candidate_lovasz

L_total = L_joint + L_mask
```

- `L_listwise` 让最终 query 分布拟合 Box-tier-first 的联合质量分布；
- `L_absolute_quality` 同时监督四个阈值概率和两个连续 IoU；
- `L_anchor_protection` 在 parent query 已过 `0.25/0.50` 时，要求它至少以 margin `0.05` 压过错误
  query，专门抑制 false override；
- `L_source_alignment` 直接训练逐 query source weights；
- Mask losses 训练 alpha/bias、空间 residual 以及候选 Mask。

正式 V49 使用 `joint_query_quality_train_only`：MCLN 主干、三源 SourceMoE 和完整 V19 parent 全部
保持 eval/frozen，只训练 45 个 Joint Query Quality state、`229,460` 个参数。优化器为 AdamW，
LR `3e-4`、weight decay `5e-4`，四卡每卡 batch `12`，80 epoch，LR milestones 为 50/75。
冻结 parent 的目的是把结构贡献与已有最佳模型严格隔离；若 V49 不增益，可以无损回退 V19。

### 11. V49 与改进版 BUTD 三创新的关系（历史状态）

改进版 BUTD 的三项创新固定指：

1. **SACR，Structured Anchor-Compositional Reasoning**：把文本拆成 target、attribute、relation、
   anchor slots，先判断候选是否符合目标/属性，再显式计算目标候选与参照物候选之间的关系和几何，
   输出 structured score。
2. **RAPF，Reliability-Aware Probabilistic Fusion**：根据 parse confidence、score entropy、top-1
   margin、base/structured 分歧、JS divergence 和 anchor 置信度预测 gate，只在结构源可靠时把
   structured residual 注入 base score。
3. **AQHNL，代码名 QAHNL/QA-HNL，Quality-Aware Hard Negative Learning**：按 Box IoU 选择正
   query，并在低 IoU 候选中选择模型当前打分最高的 hard negatives，用 IoU gap 决定 adaptive
   margin，迫使正确候选压过最容易混淆的错误候选。

V49 MCLN 与它们的关系不是“原样移植三模块后再加 MoE”：

| 对比项 | 改进版 BUTD | 最新 MCLN-MoE | 作用差异 |
|---|---|---|---|
| 结构推理 | SACR 显式 target/attribute/relation/anchor composition | 当前三源没有真正 SACR expert | MCLN 目前缺显式 anchor-conditioned 关系源，后续应作为第四源接入 |
| 源融合 | RAPF 主要在 base 与 structured residual 间学习可靠性 gate | shared default + 多 routed sources，逐 query dense/sparse routing，并有 V19 fallback | 从固定两源可靠融合升级为任意多源、逐 query 的选择或组合 |
| 困难候选 | AQHNL 主要用 Box IoU hard-negative margin 训练某个 score source | listwise query ranking + anchor protection + Box/Mask 六维质量 + candidate Mask loss | 从 Box-only 辅助排序扩展为直接参与部署的 REC/RES 联合质量学习 |
| 分割优化 | 三创新核心不直接修 Mask | alpha/bias、Mask-source evidence、空间 residual、Focal/Dice/Lovasz | 显式优化 Mask@0.25、Mask@0.50 和 mIoU |
| 泛化方式 | 依赖 structured parser validity 与选定 score source | source-name-free shared encoder、rank agreement target、无验证阈值 sweep | 更适合更换数据集和增删 source |
| 安全迁移 | RAPF 对不可靠 structured source 关 gate | shared-source anchor、V19 verifier、zero residual、missing-source fail closed | 多层保护已有最优 parent，减少 learned override 破坏正确 query |

V49 阶段可以安全写成：MCLN-MoE **继承并扩展了 RAPF 的可靠性融合原则和 AQHNL 的质量感知困难候选
原则**，将其重写为多源逐 query 路由以及 Box/Mask 联合质量学习；**SACR 尚未直接接入当前 MCLN**。
若论文要声称三项均完成跨 backbone 迁移，必须把 `SACRHead + StructuredSlotBuilder` 接为第四个
`sacr_structured` expert，并做缺解析 fail-closed、三数据集数据合同和独立消融。

### 12. V49 汇报用简明版（历史状态，当前请以第 13 节 V50 为准）

#### BUTD 三创新各自做什么

- **SACR**：把“目标是什么、有什么属性、和哪个参照物是什么关系”拆开推理，产生结构化候选分数。
- **RAPF**：判断结构化分数靠不靠谱，靠谱才融合，不靠谱就保留基础分数。
- **AQHNL**：专门找模型最容易混淆的错误 query，用真实 IoU 和自适应 margin 把正确 query 拉到前面。

#### 最新 MCLN-MoE 的主要创新点

- **多源 Query-MoE**：每个 query 自己决定依赖 default、contrastive text 还是 mask-text，也可组合
  多源，解决“每个数据集手工找最佳单源/组合”泛化差的问题。
- **共享源加安全回退**：default 永远作为稳定 anchor，V19 同时检查“是否值得切换”和“要切的
  query 是否安全”，降低错误覆盖。
- **联合 Box-Mask 质量重排**：同时预测 Box/Mask 的 `0.25/0.50` 命中概率和连续 IoU，再按
  Box-tier-first 重排，直接服务 REC 目标并保护 RES。
- **质量驱动的自适应源权重**：路由策略不记固定源编号，而根据每个 source 的 rank 与联合质量
  一致性学习权重，可自然退化成单源或形成多源组合。
- **分割头专项增强**：逐 query 学习 Mask 融合权重和 bias，增加空间 residual，并监督 top-K 困难
  候选 Mask，目标是一起提升 Mask@0.25、Mask@0.50 和 mIoU。
- **网络内训练而非数据集后处理**：所有最终分数和 Mask 修正均在 forward 中产生，不依赖验证集
  阈值扫描，便于迁移到单阶段 ScanRefer、Nr3D 和 Sr3D。

一句话区别：**BUTD 的三创新是“构造结构源、判断结构源是否可靠、用 Box hard negatives 训练
排序”；最新 MCLN-MoE 是“让每个 query 在多个可部署源之间自适应选择，并用联合 Box/Mask 质量
直接控制最终 query 和分割结果”。**

### 13. V50 SACR 四源扩展与正式队列（2026-08-05 22:16 CST）

> 本节取代第 11、12 节中“SACR 尚未直接接入 MCLN”的旧状态判断。旧段落保留用于记录当时的
> V49 代码审计结论；截至本节时间，SACR 已完成网络接入和 CPU 合同测试，但尚无正式验证指标，
> 因而只能写成“已实现”，不能提前写成“已取得增益”。

#### 13.1 最终继承边界

V50 没有把新 SACR 参数插入已经训练好的 V19 `SourceMoE`，避免改变 V19 state dict 和推理路径：

```text
V19 frozen parent SourceMoE:
  default + contrastive_text + mask_text
                         |
                         v
                  V19 parent scores

Separate Joint Query Quality source pool:
  default + contrastive_text + mask_text + sacr_structured
                         |
                         v
             query-wise adaptive source mixer
                         |
                         v
               joint Box/Mask reranker
```

SACR 的真实实现由 `StructuredSlotBuilder + SACRHead` 构成。文本被池化为
target/attribute/relation/anchor slots；候选目标与参照物之间使用 11 维相对几何；最终
`sacr_structured = default + tanh(scale) * parse_confidence * structured_score`。解析缺失、target
无效或 span 对齐失败时，该源 fail closed，不参与四源融合。

相对改进版 BUTD 三创新，当前准确关系为：

- SACR 已从 BUTD 的 base/structured 两路结构残差迁成 MCLN 的第四个可部署 expert；
- RAPF 的“可靠时才覆盖”原则被扩展为逐 query 多源权重、shared default、V19 opportunity/safety
  verifier 和 missing-source 回退；
- AQHNL 的 Box hard-negative 原则被扩展为 Box/Mask 六维绝对质量、box-tier-first listwise 重排、
  anchor protection、source-rank alignment 和候选级 Mask 监督；
- MCLN 额外加入逐 query Mask alpha/bias、Mask-source evidence、query-superpoint 空间 residual 以及
  Focal/Dice/Lovasz，属于 BUTD 三创新之外的 RES 专项扩展。

#### 13.2 三数据集结构数据合同

完整 validation 审计收据为
`experiment_output/v50_sacr_contract/sacr_structured_data_val_audit.json`：

| 数据集 | 原始样本 | sidecar 命中 | SACR 可用 | 有效 relation-anchor |
|---|---:|---:|---:|---:|
| ScanRefer | 9508 | 9508 | 9336 | 6302/6302 |
| Nr3D | 7899 | 7899 | 7824 | 4119/4119 |
| Sr3D | 17726 | 17726 | 17726 | 18882/18882 |

Sr3D 原 CSV 是权威样本表。spaCy sidecar 只有 17678 个唯一键，缺少的是官方 CSV 中 48 个重复行；
loader 现在复用对应 sidecar，而不是把 17726 条验证数据静默缩成 17678 条。三个数据集所有存活
target 的字符 offset 和 RoBERTa token alignment 均为 100% 有效。

训练集 smoke 前置合同固定权威样本数为 ScanRefer `36665`、Nr3D `32919`、Sr3D `65846`。
键级预审计已证明三者 lookup 零缺失；Sr3D 的 `65693` 个唯一 sidecar 行由原 CSV 的 153 个重复行
复用后恢复到 `65846`。完整 train offset/token/relation 对齐仍由 V50 队列执行，当前不提前标记通过。

#### 13.3 训练梯度修复

四源 mixer 最初使用纯 `argsort` rank，forward 正确但 SACR score 路径不可导；只关闭 input detach
仍不能让梯度进入 SACR。现改为 straight-through rank normalization：forward 保持完全相同的硬
rank，backward 使用标准化 sigmoid 平滑代理。回归测试证明第 0 步：

- 最终分数与受保护 V19 parent 完全相等；
- source alignment loss 能训练四源权重；
- 梯度能到达 `sacr_residual_scale`、`SACRHead` 和 `StructuredSlotBuilder`。

受保护 V19 初始化审计通过：共同 tensor `1228` 个、改变 `0` 个、新增/可训练 state `66` 个，
新增可训练参数 `1,150,390`，输出头为零初始化。V50 相关语法检查、Python compile 和聚焦回归结果
为 `244 passed, 4 warnings`。

补充静态梯度审计发现，最初版本包含 8 个被 span/anchor softmax 或 source rank 严格抵消的标量
bias：五个 slot attention bias，以及 target-attribute、global、anchor 三个 scorer 的末层 bias。
这些参数数学上不可辨识，会让“所有 Adam moment 非零”的 smoke 合同必然失败，也不会增加模型
表达能力。V50 在正式启动前删除这 8 个 dead parameters，并加入逐参数非零梯度回归；因此合同由
`74/1,150,398` 修正为 `66/1,150,390`，不影响任何 V19 tensor 或第 0 步 identity。

#### 13.4 四卡训练队列

`scripts/queue_double_stage_v50_sacr_after_v49.sh` 已在 tmux
`mcln_v50_sacr_after_v49` 中启动，队列 PID `592290`，并通过 `tail --pid=541453` 绑定正式 V49
队列，不做日志轮询、不提前占用 GPU。前序完成后执行：

1. 验收正式 V49 completion/checkpoint receipt；
2. 对三个数据集重跑 train/val 两套 SACR 数据合同，并执行 V19 identity initialization audit；
3. 进行四卡 V50 smoke，要求 66 个可训练 state 都有非零 optimizer moment；
4. smoke 通过后删除其 `.pth`，只保留收据；
5. 从受保护 V19 初始化四卡 80 epoch 正式 V50，并保留正式 latest 和各指标最优权重。

队列最初在 PID `586259` 启动；加入 dead-bias 修复和 train/val 双数据合同后，为避免运行中的 Bash
继续使用已缓冲的旧脚本内容，仅重启了这个空闲等待队列为 PID `592290`。前序训练和
V47/V48/V49 队列均未中断。

#### 13.5 同期单阶段收据

单阶段 ScanRefer epoch 19 完整收据：REC Overall `5416/9508=0.569625`、
`4392/9508=0.461927`；REC Unique `0.855532/0.751233`，Multiple
`0.519471/0.411176`。Mask Overall `0.587610/0.482331`、mIoU `0.412258`；Mask Unique
`0.880197/0.714588`，Multiple `0.536284/0.441587`。当前 REC 最优仍是 epoch 7 的
`0.571939`，REC@0.50 当前最优更新为 epoch 19 的 `0.461927`；该训练尚未完成，不能作为最终结果。

#### 13.6 单阶段 epoch 20 收据与下一检查窗口

epoch 20 的权威收据为：REC Overall `5418/9508=0.569836`、
`4353/9508=0.457825`；REC Unique `1213/1419=0.854827`、
`1056/1419=0.744186`，Multiple `4205/8089=0.519842`、
`3297/8089=0.407591`。Mask Overall `5569/9508=0.585717`、
`4558/9508=0.479386`、mIoU `0.409957`；Mask Unique `0.881607/0.715292`，
Multiple `0.533811/0.438002`。本轮未刷新任一历史最优。

learned selector 仍 100% 选择 `default`，两源 oracle 相对 default 的 REC@0.25/0.50
余量只有 `0.00084/0.00095`。这表明当前单阶段二源候选池本身缺少足够的可纠正样本；继续提高
selector 覆盖率无法产生目标所需的增益，后续单阶段架构实验应复用 V50 的四源候选池和联合
Box/Mask query reranker。

epoch 7、19、20 分别是当前综合最优、REC@0.50 最优和 latest 三个物理 checkpoint inode；其余
低指标 epoch 权重已由 retention 清理，best 名称均为硬链接。epoch 8--20 的收据间隔约
`27.1` 分钟，epoch 20 在日志时间 `22:42:58` 发布，所以下一次只在约 `23:10` 的 epoch 21
完成窗口检查，不做分钟级轮询。

#### 13.7 V50 排队期间代码复审

在不占用 GPU、也不读取活动训练日志的前提下，重新核对了 V50 的实际训练链路：

- `joint_query_quality_train_only` 同时解冻 `joint_query_quality_reranker`、
  `StructuredSlotBuilder`、`SACRHead` 和 `sacr_residual_scale`，其余 V19 parent 保持冻结；
- 四源 `source_mix_alignment_loss` 已纳入 joint supervision，总权重由正式 selection 继承且
  对 SACR 强制不低于 `0.25`；
- query listwise、六维绝对质量、anchor protection、Mask calibration 和候选级
  Focal/Dice/Lovasz 均在 joint-only 分支进入最终标量 loss；
- SACR source 使用 straight-through rank，反向梯度可到达 slot builder、SACR scorer 和 scale；
  缺解析行通过 source validity 置零，不会污染其余三个源。

当前工作树重新执行 SACR/Joint Query/V50 queue 聚焦回归 `71 passed, 2 warnings`，SourceMoE
回归 `266 passed`，合计 `337 passed`；V50 queue/train Shell 语法检查及相关 Python compile
全部通过。警告仅为 CPU 测试环境关闭 CUDA autocast，不影响合同。该复审没有修改模型代码或
正在等待的队列配置。

#### 13.8 候选上界与 V50 结构方向确认

已完成的 V19 正式验证给出 candidate-set oracle REC `5987/9508=0.629680`、
`5230/9508=0.550063`，Mask oracle mIoU `0.451708`。相对 `0.59/0.49` 的最低命中数
`5610/4659`，现有 query candidate set 分别多出 `377/571` hits，证明达到目标不需要先生成
新的 box proposal。相比之下，历史固定 source Top-1 oracle 只有约 `0.58551/0.46992`，连目标
本身都覆盖不到；所以“每个源先取 Top-1，再让模型选择源”的 sample-level selector 在结构上
不可能完成目标。

结构化前置收据已写入
`experiment_output/v50_sacr_contract/v19_candidate_headroom_audit.json`：
`candidate_oracle_target_pass=true`、`learned_target_pass=false`、决策为
`train_contextual_gate`，与上述人工核算一致。

据此，V50 当前方向保持不变：四个 source 必须保留 `[Q,S]` 的逐 query 分数，先进行 query-wise
自适应融合，再由 set attention 和 Box/Mask 联合质量对完整候选集合重排。现阶段不提前加入
query/box refinement，以免在候选覆盖已经充分时扩大训练变量。只有正式 V50 仍未达标，且新
四源 action oracle 或候选质量诊断证明 proposal 上界不足，才进入关系条件 box refinement；
若 oracle 达标而 learned 指标不足，则继续修正 listwise/质量校准和安全覆盖率。

#### 13.9 单阶段后处理队列吞吐优化

复审 `queue_single_stage_best_postprocess.sh` 确认：训练结束后固定读取 retained
`ckpt_best_rec_acc025.pth`，先审计其 `butd/butd_gt/butd_cls` 全为 false，再重新生成 train/val
候选与 geometry cache；最终收据强制包含 REC 和 Mask 的 Overall、Unique、Multiple 精确计数。

原队列在两源 candidate cache 完成后，依次执行 GPU2 parent reranker 和 GPU3 train-only mask
geometry audit；两者都只依赖同一个只读 train cache，彼此没有数据依赖。现已改成两个独立进程
并行运行并统一 wait/fail gate。geometry cache 两路仍并行使用 GPU0/1，其余单模型训练阶段
保持单卡，不做无效显存占位。

静态复审发现 geometry runtime 有严格的权威执行合同：`world_size=1`、本地 `cuda:0`、
`args.batch_size=12`，非末 batch 也必须恰有 12 行。因而最终 official parent+geometry evaluation
不能改成四卡；队列原有的单卡 `batch_size=18` 同样会 fail closed，现已修正为单卡、batch 12。
相关 queue/geometry runtime/completion/metrics 回归在隔离 CUDA 后为 `114 passed, 1 skipped`，
Shell 语法检查通过。第一次运行测试时未隔离 CUDA，唯一 CUDA device-contract case 在训练已占满
显存时 OOM；它没有产生 checkpoint 或新训练进程，随后已用 `CUDA_VISIBLE_DEVICES=''` 完整重跑。

为使正在 wait 的 Bash 使用最终脚本 inode，只重启了空闲 tmux
`mcln_single_stage_postprocess_queue`；最终 pane PID `599319` 继续通过 `tail --pid=357439` 绑定
原单阶段训练，不读取日志、不提前占 GPU，训练进程未中断。安全提速仅保留 GPU2/3 两项独立任务
并行，不以破坏 runtime provenance 的方式强行四卡化。

#### 13.10 V50 跨数据集监督合同复审

V50 核心模块 `joint_query_quality.py`、`sacr_head.py`、`structured_slots.py` 和
`source_choice_adapter.py` 中没有 ScanRefer/Nr3D/Sr3D 名称分支。joint/SACR supervision 使用
逐样本 `sample_dataset`：ScanRefer、Nr3D、Sr3D 行均为 true，仅联合训练中没有单一 referring
target 的 Scannet detection 行为 false；它不会错误使用 batch 级 `language_dataset` 把混合
batch 全部纳入或全部排除。

source mixer 除 shared flag 外没有 source-ID embedding，routed sources 换序时输出权重同步换序、
最终 mixed score 不变；source-rank alignment target 同样对 routed-source 置换不变。结合第
13.2 节三个数据集的 sidecar/offset/token 合同，这证明当前架构和监督路径具备真实的跨数据集
接口，而不是只在文档中宣称可迁移。dataset/SACR/sample-mask/source-permutation 聚焦回归为
`23 passed, 2 warnings`，警告仍仅来自 CPU 环境关闭 CUDA autocast。

#### 13.11 单阶段 epoch 21 收据

epoch 21 于日志时间 `23:09:57` 发布。REC Overall 为
`5392/9508=0.567101`、`4347/9508=0.457194`；REC Unique 为
`1216/1419=0.856942`、`1058/1419=0.745595`，Multiple 为
`4176/8089=0.516257`、`3289/8089=0.406602`。Mask Overall 为
`5539/9508=0.582562`、`4547/9508=0.478229`、mIoU `0.408754`；Mask Unique 为
`0.880197/0.710359`，Multiple 为 `0.530350/0.437508`。本轮没有刷新任何 retained best。

selector 仅在 `1/9508` 行选择非 default，且没有产生 threshold fix 或 break；两源 oracle
headroom 进一步只有 `0.00074/0.00084`。retention 已删除 epoch 20 inode，当前仍只有三个物理
checkpoint：epoch 7 负责 REC@0.25 与三个 Mask best，epoch 19 负责 REC@0.50 best，epoch 21
仅作为 latest。epoch 20 到 21 的完整收据间隔为 `26m59s`，所以下次只在约 `23:37 CST`
检查 epoch 22，不做中间轮询。

#### 13.12 后处理队列重启后的依赖链恢复

为加载第 13.9 节修正后的后处理脚本，旧后处理 PID `368662` 被空闲重启。原下游 V47 仍绑定
旧 PID；旧 PID 结束后，它按设计发现 `training_completion.json` 尚不存在并 fail closed。随后
V48、V49 smoke、V49 formal 和 V50 也分别因缺少上游 summary/selection 而退出，没有绕过门禁，
也没有启动任何 GPU 作业。该行为保护了实验正确性，但意味着重启上游等待器后必须显式重建
整个事件链。

现已从最终后处理 PID 逐级重建并核对每层事件日志：

```text
599319 single-stage parent+geometry postprocess
   -> 602504 single-stage V47 joint query
   -> 602614 V48 spatial-mask smoke
   -> 602767 V49 adaptive-source smoke
   -> 602866 formal double-stage V49
   -> 603021 V50 four-source SACR
```

六个进程的 `/proc/<pid>/wchan` 均为 `do_wait`；每个下游日志都明确记录上述直接 predecessor PID，
并使用 `tail --pid`，没有日志轮询。`nvidia-smi` 中仅有当前单阶段训练的四个 compute PID，重建的
等待链未占显存。当前训练 PID `357439` 从始至终未中断。

#### 13.13 严格劣势旧权重清理

在不读取活动训练中间指标的 epoch 22 等待窗口，对历史大权重执行了指标、硬链接、脚本依赖和
`lsof` 四项审计。以下两个物理 inode 均被同类保护基线同时支配，且无活动进程或启动脚本依赖，
因此删除不可恢复的 checkpoint 文件、保留其 JSON/CSV/log 指标收据：

- inode `10739770061`：`mcln_pair_rank005_rank010_default005_2ep_best_acc025_epoch71_0.57930.pth`，
  REC 为 `0.57930/0.46256`；被同类保护权重 `0.57993/0.46340` 同时支配，删除 `794125897` bytes。
- inode `10739948061`：Optuna trial 0 的
  `best_trial_acc025_epoch72.pth` 与 `best_optuna_rec_acc025.pth` 两个硬链接，REC 为
  `0.57699/0.46066`；删除最后两个链接后释放 `794125833` bytes。

合计释放 `1588251730` bytes，约 `1.48 GiB`；`/root/autodl-tmp` 删除后可用空间为
`10788028416` bytes。旧 epoch 68 权重 `0.57793/0.46140` 暂不删除，因为
`run_optuna_mcln_source_choice_continue20.sh` 仍显式将其作为 `ACC50_CKPT` 默认输入；在重写或退役
该旧实验入口前，删除会制造悬空依赖。

清理后再次核对：`scanrefer_best_backbone_acc025_0.582878_component.pth` 与保护的 `0.57993`
基线仍共享 inode `10739770064`、link count 为 2；V19 最优权重 inode `6496464367` 仍独立存在。
`0.582878` 后处理组件、V19、Mask@0.50 最优和当前单阶段 retained checkpoints 均未触碰。

#### 13.14 单阶段 epoch 22 收据

epoch 22 在预计窗口内于 `23:36:58 CST` 发布。REC Overall 为
`5416/9508=0.569626`、`4382/9508=0.460875`；REC Unique 为
`1219/1419=0.859056`、`1068/1419=0.752643`，Multiple 为
`4197/8089=0.518853`、`3314/8089=0.409692`。Mask Overall 为
`5586/9508=0.587505`、`4567/9508=0.480332`、mIoU `0.411434`；Mask Unique 为
`1255/1419=0.884426`、`1017/1419=0.716702`，Multiple 为
`4331/8089=0.535418`、`3550/8089=0.438868`。

本轮没有刷新任何 retained best。当前单阶段 best 仍为：epoch 7 的 REC@0.25
`0.571939` 和 Mask `0.588241/0.483382/0.412678`，epoch 19 的 REC@0.50
`0.461927`。retention 已删除 epoch 21 inode，当前物理 checkpoint 仅为 epoch 7、epoch 19 和
epoch 22 latest。四卡在检查时分别占用约 `39.3--39.8 GiB`，GPU 利用率为
`100/80/56/95%`，训练与六级后续队列均存活。

由于活动进程启动早于当前 subgroup receipt 代码更新时间，它产生的原始 epoch JSON 仍只有
Overall；本轮从同一次 official evaluator 日志读取 exact subgroup counters，并另存
`experiment_output/single_stage_epoch22_full_metrics.json`。当前代码的完整 Overall/Unique/Multiple
收据合同已由隔离 CUDA 的 `34 passed` 回归覆盖，后续新进程会直接写入这些分组字段。

epoch 21 到 22 的回执间隔为 `27m01s`，下次只在约 `2026-08-06 00:04 CST` 检查 epoch 23。
按剩余 78 个 epoch 和当前速度粗估，100 epoch 训练约在 `2026-08-07 11:10 CST` 完成；实际时间
仍以之后每轮真实回执间隔滚动修正。

#### 13.15 V50 高质量 Query 聚焦源对齐

复审发现原四源 `source_mix_alignment_loss` 对全部 256 个 valid query 等权平均；而 REC/RES
最终指标只由 Top-1 及其附近少量候选决定，大量背景 query 会稀释 SACR 和 source router 的有效
监督。现增加
`joint_query_quality_source_mix_query_focus_weight`，默认 `0.0`，保持 V49 行为；V50 明确使用
`0.75`。训练时 75% 的 query 权重来自 detached、Box-tier-first 的 listwise relevance，剩余
25% 保留均匀权重，避免只学习单个 oracle query。该设计不使用 source 名称、数据集名称或验证集
阈值，仍满足 routed-source 置换等变和跨 ScanRefer/Nr3D/Sr3D 的迁移合同。

实现链路已贯通 CLI、loss、checkpoint args/audit、V50 selection、smoke 和正式训练。V50 的
`source_mix_loss_weight` 仍继承 V49 selection 并强制不低于 `0.25`，聚焦权重固定为 `0.75`；旧
checkpoint 缺该字段时按 `0.0` 解释。新增精准回归证明：

- `focus=0.0` 与原 V49 均匀 source-alignment 目标按完整浮点运算顺序逐值一致；
- `focus=0.75` 在 step-0 identity 状态下仍不改变 V19 parent 输出，但梯度非零地到达 adaptive
  source router；
- 同一个 `focus=0.75` 损失可继续穿过 straight-through source rank，到达
  `StructuredSlotBuilder`、`SACRHead` 和 `sacr_residual_scale`，没有 dead parameter；
- routed sources 换序时 loss 和 target Top-1 指标保持不变，非法聚焦权重 fail closed。

隔离 CUDA 的首轮聚焦回归为 `180 passed`；补齐 V49 精确兼容和真实 SACR 全链路聚焦梯度后，
Joint Query/SACR/SourceMoE/V50 聚焦集为 `196 passed, 2 warnings`，扩大到 SourceMoE、数据合同、
训练参数组、初始化和 checkpoint/queue audit 后为 `386 passed, 2 warnings`。两条 warning 仅为
CPU 环境自动关闭 CUDA autocast。测试未占用活动训练 GPU，也未修改 PID `357439` 或六级等待链。

#### 13.16 单阶段 epoch 23 收据

按预计窗口仅检查一次，epoch 23 于 `2026-08-06 00:04:00 CST` 发布。REC Overall 为
`5401/9508=0.568048`、`4369/9508=0.459508`；REC Unique 为
`1219/1419=0.859056`、`1065/1419=0.750529`，Multiple 为
`4182/8089=0.516998`、`3304/8089=0.408456`。Mask Overall 为
`5548/9508=0.583509`、`4550/9508=0.478544`、mIoU `0.409290`；Mask Unique 为
`1247/1419=0.878788`、`1013/1419=0.713883`，Multiple 为
`4301/8089=0.531710`、`3537/8089=0.437260`。

本轮没有刷新任何指标最优。当前单阶段 retained best 仍为 epoch 7 的 REC@0.25
`0.571939` 和 Mask `0.588241/0.483382/0.412678`，以及 epoch 19 的 REC@0.50
`0.461927`。物理 checkpoint 仅有三个 inode：epoch 7 同时承担 REC@0.25 和三个 Mask best，
epoch 19 承担 REC@0.50 best，epoch 23 与 `ckpt_epoch_last.pth` 共享 latest inode；epoch 22 已由
retention 删除。受保护的历史 `0.582878` 组件和 V19 权重均存在，未被触碰。

learned selector 本轮 `100%` 选择 default，没有产生 fix 或 break；两源 oracle headroom 仅为
REC@0.25 `0.00084`、REC@0.50 `0.00074`，继续支持单阶段后续采用 V50 四源逐 query 联合重排，
而不是继续优化当前二源 sample selector。四卡检查时显存约 `39.3--39.8 GiB`，利用率为
`100/57/100/96%`；训练 PID `357439` 与六级后续队列全部存活。

完整收据为 `experiment_output/single_stage_epoch23_full_metrics.json`。epoch 22 到 23 的发布间隔为
`27m02s`，下次只在约 `2026-08-06 00:31 CST` 检查 epoch 24。按剩余 77 个 epoch 和当前速度
粗估，epoch 100 约在 `2026-08-07 10:46 CST` 完成，后续继续按真实间隔滚动修正。

#### 13.17 V50 运行中脚本版本审计

V50 等待进程 PID `603021` 启动早于第 13.15 节 query-focus 修改，因此额外核对运行中 Bash 是否
会采用最终脚本。`/proc/603021/fd/255` 与当前
`scripts/queue_double_stage_v50_sacr_after_v49.sh` 均为 inode `6914648722`、size `15786`、mtime
`2026-08-05 23:46:36 CST`；运行 fd 当前 offset 为 byte `4411`，而新增
`source_mix_query_focus_weight=0.75` 的第一处内容位于 byte `6538`。因此 Bash 尚未读取该配置段，
上游 V49 完成后会从同一最终 inode 继续读取新配置，不存在旧脚本缓存歧义。

据此不重启 PID `603021`，避免无必要改动已经核对的 PID 依赖链。其子进程仍为
`tail --pid=602866 -f /dev/null`，V50 锁文件继续由当前进程持有；训练 PID `357439` 和上游五级
队列均未触碰。

#### 13.18 旧 source-choice epoch 70 权重退役

磁盘审计发现 inode `6447584600` 的两个硬链接共占 `794127241` bytes，对应旧
source-choice epoch 70，REC 为 `0.57920/0.45877`。它的直接续训后继 epoch 71 为
`0.57993/0.46340`，两项严格更高；两者 model state 均为 1144 个 key，key 集和 tensor shape
完全一致，后继 checkpoint 的 config 也明确记录从旧权重初始化。后继已由
`scanrefer_best_backbone_acc025_0.582878_component.pth` 保护，SHA-256 为
`3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`。

删除前唯一运行时依赖是遗留
`scripts/tuning/run_optuna_mcln_source_choice_continue20.sh` 的 `ACC25_CKPT` 默认值。现已将默认值
迁到上述受保护后继，同时保留环境变量覆盖能力；新增回归 `1 passed` 且 Shell syntax 通过。
全项目代码/测试/文档不再引用旧 basename，`lsof` 也没有活动引用后，删除以下两个硬链接：

- `.../1781953653/best_rec_acc025_epoch70.pth`；
- `.../preserved_best/mcln_source_choice/current_best_rec_acc025_epoch70_0.57920.pth`。

删除不可恢复，历史 JSON/log 指标收据保留。`/root/autodl-tmp` 可用空间从 `10532808 KiB` 增至
`11308204 KiB`。清理后再次核对，历史 `0.582878` 保护 inode `10739770064`、link count 2，V19
保护 inode `6496464367` 均未变化。完整清理收据为
`experiment_output/weight_cleanup_20260806_0015.json`。

#### 13.19 单阶段 epoch 24 收据

按估算窗口检查，epoch 24 于 `2026-08-06 00:30:58 CST` 发布。REC Overall 为
`5388/9508=0.566681`、`4343/9508=0.456773`；REC Unique 为
`1215/1419=0.856237`、`1058/1419=0.745595`，Multiple 为
`4173/8089=0.515886`、`3285/8089=0.406107`。Mask Overall 为
`5551/9508=0.583824`、`4559/9508=0.479491`、mIoU `0.408955`；Mask Unique 为
`1249/1419=0.880197`、`1021/1419=0.719521`，Multiple 为
`4302/8089=0.531833`、`3538/8089=0.437384`。

本轮没有刷新任何 best。retention 已删除 epoch 23 inode，当前物理 checkpoint 仍只有 epoch 7、
epoch 19 和 epoch 24 latest；受保护的 `0.582878` 组件、V19 和 Mask best 均未删除。当前两源
selector 仍 `100%` 选择 default，fix/break 均为 `0`，REC oracle headroom 为 `0.00116/0.00105`，
说明单阶段后续应继续等待 V50 四源 query-wise 重排，而不是依赖当前 sample-level 二源 selector。

epoch 23 到 24 的发布间隔为 `26m58s`，下次只在约 `2026-08-06 00:58 CST` 检查 epoch 25。按
当前滚动速度，epoch 100 预计约在 `2026-08-07 10:40 CST` 完成。完整收据为
`experiment_output/single_stage_epoch24_full_metrics.json`。

#### 13.20 V50 三数据集 train split SACR 合同

为避免正式队列数日后才暴露 sidecar 或 token 对齐错误，提前在 CPU 上执行 V50 原定的 train split
全量合同审计，输出为
`experiment_output/v50_sacr_contract/sacr_structured_data_train_audit.json`，总结果 `pass=true`：

| 数据集 | 原始/Join 样本 | SACR 可用 | 有效 relation-anchor | offset/token 有效率 |
|---|---:|---:|---:|---:|
| ScanRefer | `36665/36665` | `35997` | `30235` | `1.0/1.0` |
| Nr3D | `32919/32919` | `32545` | `17269` | `1.0/1.0` |
| Sr3D | `65846/65846` | `65846` | `70006` | `1.0/1.0` |

三个数据集 lookup missing 和 sidecar extra 均为 0，relation-anchor pair valid rate 均为 `1.0`。
Sr3D base 中 153 个重复 key 按既定合同复用唯一 sidecar 行，最终样本数保持 `65846`，没有静默丢样。
审计使用 `CUDA_VISIBLE_DEVICES=''`，仅占约一个 CPU 核和 2.1 GiB 内存；没有新增 GPU 进程或
checkpoint。V50 队列仍会在正式启动时重跑 train/val 两份合同，确保输入在实际启动时未漂移。

#### 13.21 单阶段 epoch 25 收据

epoch 25 于 `2026-08-06 00:57:59 CST` 发布。REC Overall 为
`5405/9508=0.568469`、`4353/9508=0.457825`；REC Unique 为
`1217/1419=0.857646`、`1060/1419=0.747005`，Multiple 为
`4188/8089=0.517740`、`3293/8089=0.407096`。Mask Overall 为
`5567/9508=0.585507`、`4558/9508=0.479386`、mIoU `0.410079`；Mask Unique 为
`1252/1419=0.882311`、`1014/1419=0.714588`，Multiple 为
`4315/8089=0.533440`、`3544/8089=0.438126`。

本轮没有刷新 retained best；epoch 24 latest 已删除，物理 checkpoint 仍只有 epoch 7、epoch 19
和 epoch 25 latest。当前最好保持 REC `0.571939/0.461927`、Mask
`0.588241/0.483382/0.412678`。两源 selector 仍没有 fix/break，oracle headroom 为
`0.00137/0.00137`，不足以改变单阶段架构方向。

epoch 24 到 25 的发布间隔为 `27m01s`，下次只在约 `2026-08-06 01:25 CST` 检查 epoch 26。
按剩余 75 个 epoch 计算，epoch 100 当前预计约在 `2026-08-07 10:44 CST` 完成。完整收据为
`experiment_output/single_stage_epoch25_full_metrics.json`。

#### 13.22 JointBoxMask 真正的 Mask 策略接线

代码复审确认，原单阶段 `parent+geometry` 队列只启用了
`eval_use_rec_reranker_scores/eval_use_rec_geometry_reranker_scores`。即使后续简单增加旧的
`eval_use_rec_joint_box_mask`，原 `apply_rec_joint_box_mask_runtime_policy()` 也只返回 flat geometry
score；`JointBoxMaskAdapter.calibration_head` 的五个输出既没有进入 `train_adapter()` 损失，也没有
传给 Evaluator。Evaluator 只能让旧 fused Mask 跟随 geometry winner，不能声称完成 learned Mask
校准。

现将该链路升级为 `rec-joint-box-mask-adapter-v2`：

1. `JointBoxMaskAdapter` 增加逐 parent-query 的 `3 x 5=15` 路 Mask policy head，三个可部署源为
   `text/query/fused`，五个 logit 阈值为 `[-1,-0.5,0,0.5,1]`。head 为零初始化，并加入极小的
   `fused@0` identity prior，因此 step 0 精确选择原 Mask 策略。
2. 现有完整 train-only cache 已包含每个 query 的 `[3,5]` Mask IoU，不增加 GT runtime 字段或大
   体积 raw-logit cache。训练同时优化 policy cross-entropy 和 expected-IoU regret；原 Box tier、
   Mask IoU、Mask threshold 和 ranking 头继续联合训练。
3. train calibration gate 现在用同一个最终 flat query，先映射 parent query，再用 learned source/
   threshold 计算 Mask@0.25、Mask@0.50 和 mIoU；只有 REC 两档不退且 Mask 门槛通过才发布 v2
   adapter。
4. runtime payload 明确携带 final flat index、parent position、policy/source/threshold index 和精确
   threshold。每项都检查 shape、dtype、device、范围、policy 分解、final score Top-1 和 parent
   mapping；缺字段、索引漂移或阈值不一致均 fail closed。
5. MCLN Evaluator 不再只重用旧 fused Mask。它在最终 parent query 上选择 learned text/query/fused
   logits并应用 learned 阈值，再映射到 point mask；REC 与 RES 因而使用同一 query，同时 Mask
   内容本身也真正改变。推理 payload 不允许 GT/IoU target 字段。

正在等待的 `queue_single_stage_best_postprocess.sh` 没有重启。Joint cache、30-epoch adapter、train
gate 和 joint official subgroup audit 作为第二分支追加在原 baseline official receipt 之后；train
gate 返回 baseline 时保留原结果并继续下游，只有发布 adapter 才进行 joint official eval。追加后
脚本和 `/proc/599319/fd/255` 仍为 inode `6914648716`，fd offset `2075` 位于追加段之前，运行中的
Bash 会读取新分支，不影响 PID 等待链。

聚焦回归覆盖模型/loss、完整 train-only cache、trainer/artifact、实际 geometry builder、runtime、
Evaluator、official runner、queue 和既有 geometry/parent 回归；结果为 `163 passed, 1 skipped`。
skip 为既有无 CUDA 条件分支。Shell syntax 和四个修改 Python 文件的 compile 均通过。

#### 13.23 单阶段 epoch 26 收据

按预计窗口仅检查一次，epoch 26 于 `2026-08-06 01:25:04 CST` 发布。REC Overall 为
`5430/9508=0.571098`、`4387/9508=0.461401`；REC Unique 为
`1225/1419=0.863284`、`1068/1419=0.752643`，Multiple 为
`4205/8089=0.519842`、`3319/8089=0.410310`。Mask Overall 为
`5585/9508=0.587400`、`4586/9508=0.482331`、mIoU `0.412116`；Mask Unique 为
`1261/1419=0.888654`、`1020/1419=0.718816`，Multiple 为
`4324/8089=0.534553`、`3566/8089=0.440846`。

本轮仍未刷新 retained best：REC 保持 epoch 7 的 `0.571939` 和 epoch 19 的 `0.461927`，Mask
保持 epoch 7 的 `0.588241/0.483382/0.412678`。retention 已删除 epoch 25 latest，物理训练权重
继续只保留各指标最优和 epoch 26 latest；历史双阶段 `0.582878` 与 V19 保护权重没有触碰。完整
收据为 `experiment_output/single_stage_epoch26_full_metrics.json`。

epoch 25 到 26 的发布间隔为 `27m05s`，下次只在约 `2026-08-06 01:52 CST` 检查 epoch 27；按
剩余 74 个 epoch 粗估，epoch 100 约在 `2026-08-07 10:50 CST` 完成。

#### 13.24 单阶段 epoch 27 收据与 V50 部署链复核

按上一轮预计窗口仅检查一次，epoch 27 于 `2026-08-06 01:52:42 CST` 发布。REC Overall 为
`5409/9508=0.568889`、`4363/9508=0.458877`；REC Unique 为
`1215/1419=0.856237`、`1068/1419=0.752643`，Multiple 为
`4194/8089=0.518482`、`3295/8089=0.407343`。Mask Overall 为
`5548/9508=0.583509`、`4543/9508=0.477808`、mIoU `0.408871`；Mask Unique 为
`1254/1419=0.883721`、`1018/1419=0.717407`，Multiple 为
`4294/8089=0.530844`、`3525/8089=0.435777`。

本轮未刷新 retained best，当前单阶段最好仍为 REC `0.571939/0.461927` 和 Mask
`0.588241/0.483382/0.412678`。retention 已删除 epoch 26 latest；物理训练权重继续只有 epoch 7、
epoch 19 和 epoch 27 latest 三个 inode。历史双阶段 `0.582878` 主干、V19、parent、geometry、
V28 Mask@0.50 和 QMask epoch 33 权重均保持只读保护。learned selector 仍 `100%` 选择 default，
fix/break 均为 0，两源 oracle headroom 只有 `0.00105/0.00105`，没有理由改变后续四源逐 query
MoE 方向。

等待窗口内复核 V50 当前工作树：四源权重以每个 source 的 query rank 与 Box-tier-first 联合质量
rank 的一致性监督，`query_focus_weight=0.75` 将大部分权重放在最终 Top-1 附近，同时保留 25%
均匀 query 权重；梯度可穿过 straight-through rank 到 `StructuredSlotBuilder`、`SACRHead` 和
SACR residual scale。Mask alpha/bias 会直接覆盖 `last_pred_masks/sp_last_pred_masks/adaptive_weights`，
空间 residual 随后再次写回点级 Mask，最终结果和候选级 Focal/Dice/Lovasz 监督都消费修改后的
Mask，不存在旧 JointBoxMask v1 的“训练头但 evaluator 未消费”问题。隔离 CUDA 的 Joint Query、
Mask、SACR、数据合同、V50 queue/checkpoint 聚焦回归为 `211 passed, 2 warnings`；warning 仅为
CPU 环境自动关闭 CUDA autocast，活动 GPU 未被测试占用。

epoch 26 到 27 的正式发布间隔为 `27m38s`，下次只在约 `2026-08-06 02:20 CST` 检查 epoch 28。
按剩余 73 个 epoch 粗估，epoch 100 约在 `2026-08-07 11:29 CST` 完成；后续继续按真实发布间隔
滚动修正。检查时四卡显存约 `39.3--39.8 GiB`，瞬时利用率为 `87/89/100/99%`；训练 PID
`357439` 和五级后续队列均存活。完整收据为
`experiment_output/single_stage_epoch27_full_metrics.json`。

#### 13.25 V51 预注册：RAPF-style 匿名源分布可靠性

V50 保留为 rank-only 四源对照，不在正式启动前改变其结构。进一步逐项对照改进版 BUTD 的
`ReliabilityFusion` 后确认，V50 mixer 已看到逐 query 的 source rank、相对 shared rank gap 和
联合 Box/Mask quality，但没有显式看到 RAPF 使用的整行 score distribution 可靠性。对
`contrastive_text/mask_text/sacr_structured` 这类尺度和尖锐度不同的源，只凭 rank 可能无法区分
“稳定的 Top-1 证据”和“近似均匀、margin 很小的偶然 Top-1”。

因此预注册 V51 可选开关
`joint_query_quality_use_source_distribution_reliability`。它对每个 `[B,Q,S]` source 使用完全共享的
无源名函数，增加六维证据：

1. 源内标准化 query score；
2. 源内 query probability；
3. 按有效 query 数归一化的 entropy；
4. 有界 Top-1/Top-2 margin；
5. 与 shared source 的 Top-1 disagreement；
6. 与 shared source query distribution 的归一化 JS divergence。

该设计不读取 source 名称、数据集名称、validation 阈值或 GT；解析不可用的 SACR source 仍由
`source_validity` 整列置零。routed source 置换时六维特征和最终权重同步置换，增删源时仍复用同一
encoder，因而保留 ScanRefer/Nr3D/Sr3D 的迁移合同。它比直接搬运 BUTD RAPF 更一般：不硬编码
parse confidence、generic target 或 SACR anchor 字段，而是把可迁移的 entropy/margin/disagreement/
JS 原则推广到任意多源逐 query MoE。

兼容边界严格固定：开关默认 false；V50 的 `source_encoder` 输入仍为 `H+9`，state 数和
`1,150,390` 参数均不变。V51 开启后输入为 `H+15`，只增加 `6x128=768` 个参数，总参数
`1,151,158`，state 数仍为 66；router 与 strength 最终层继续零初始化，所以 step 0 精确回退 V19
parent。真实受保护 V19 初始化审计为 common/changed/missing/unexpected/shape-mismatch
`1228/0/66/0/0`、`pass=true`，收据为
`experiment_output/v51_rapf_source_reliability/v51_protected_v19_initialization.json`。

当前核心、CLI、MCLN、训练 launcher、初始化/正式 checkpoint audit 已接线。V49/V50 profile 明确
要求该开关为 false，运行中 V50 queue 的 smoke/formal 也各显式设置一次 `=0`；queue 文件和
`/proc/603021/fd/255` 同为 inode `6914648722`，fd offset `4411` 尚未读到新增配置段。隔离 CUDA
的 Joint Query、parser、初始化、checkpoint、V50 queue 和跨数据集 launcher 聚焦回归为
`194 passed`。

V51 进入条件预注册为：正式 V50 learned 指标仍未达到 `0.59/0.49`，同时完整 query/action oracle
继续达到目标，且 source diagnostics 显示 entropy/margin/JS 与 beneficial/harmful switch 存在可分
信息。若 V50 已达标则不运行 V51；若 V50 oracle 也不足则不运行本模块，而按第 13.8 节进入关系
条件 proposal refinement。这样 V51 只修复“源可靠性校准不足”，不把它当作任何失败类型的通用
补丁。

#### 13.26 单阶段 epoch 28 收据

按预计窗口仅检查一次，epoch 28 于 `2026-08-06 02:19:50 CST` 发布。REC Overall 为
`5405/9508=0.568469`、`4341/9508=0.456563`；REC Unique 为
`1228/1419=0.865398`、`1064/1419=0.749824`，Multiple 为
`4177/8089=0.516380`、`3277/8089=0.405118`。Mask Overall 为
`5551/9508=0.583824`、`4531/9508=0.476546`、mIoU `0.408109`；Mask Unique 为
`1257/1419=0.885835`、`1015/1419=0.715292`，Multiple 为
`4294/8089=0.530844`、`3516/8089=0.434664`。

本轮没有刷新 retained best，当前单阶段最好继续为 REC `0.571939/0.461927` 和 Mask
`0.588241/0.483382/0.412678`。learned selector 仍 `100%` 选择 default，fix/break 为 0，两源
oracle headroom 仅 `0.00095/0.00105`。retention 已用 epoch 28 latest 替换并删除 epoch 27 latest，
历史双阶段保护权重未触碰。检查时四卡显存均约 `39--40 GiB`，三个 rank 瞬时利用率为
`96--100%`；GPU0 的 `6%` 是单次采样中的同步波谷，四个 DDP rank 与主 PID 均存活。

epoch 27 到 28 的发布间隔为 `27m08s`，下次只在约 `2026-08-06 02:47 CST` 检查 epoch 29。按
剩余 72 个 epoch 粗估，epoch 100 约在 `2026-08-07 10:54 CST` 完成。完整收据为
`experiment_output/single_stage_epoch28_full_metrics.json`。
