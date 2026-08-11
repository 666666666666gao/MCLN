# ScanRefer 完整创新模型 Optuna 重训设计

## 状态与决策

本设计于 2026-07-24 经用户分段确认。

本轮 Optuna 的模型范围固定为可端到端训练的 `MCLN +
source-choice selector`。parent reranker、geometry reranker 和 joint
box-mask selector 不参与 20 组短训；它们与 backbone checkpoint 指纹绑定，
只在长训选出候选 backbone 后按新权重重新生成。

采用单锚点、train-only、约束 TPE：20 个 trial 全部从同一份无 selector
的 epoch-54 checkpoint 开始，每个 trial 严格训练两个 epoch。Optuna 只使用
ScanRefer train 的 scene-disjoint calibration 指标。选出参数后重新加载
epoch-54，在完整 ScanRefer train 上执行长训，不续接短训 checkpoint。

当前目录没有 Git 元数据，禁止为了本实验初始化仓库。可复现性由源码快照、
逐文件 SHA-256、环境清单、完整命令、数据划分和 artifact 关系清单保证。

## 最终目标

最终发布必须在同一次 9,508 条 ScanRefer official validation 运行中满足：

| 指标 | 要求 | 精确计数 |
| --- | ---: | ---: |
| REC Position Acc@0.25 | `>= 59.00%` | `>= 5,610 / 9,508` |
| REC Position Acc@0.50 | `>= 48.60%` | `>= 4,621 / 9,508` |
| 3DRES Mask Acc@0.25 | `> 58.70%` | `>= 5,582 / 9,508` |
| 3DRES Mask Acc@0.50 | `> 50.70%` | `>= 4,821 / 9,508` |
| 3DRES Mask semantic mIoU | `> 44.72%` | `> 0.4472` |

Position 和 mask 指标必须来自同一系统，不允许拼接不同 checkpoint 的最优值。
推理不得使用 ground-truth box、mask、target ID 或 IoU。

## 已核验事实

### 初始化 checkpoint

统一初始化权重：

`pretained model/ckpt_epoch_54.pth`

- SHA-256：`a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d`
- 大小：793,041,121 bytes
- checkpoint epoch：54
- 包含 1,135 个 model keys，不包含 source-choice selector 参数
- 与当前 MCLN 的 1,135 个公共参数名和 shape 全部匹配
- 原始 config 使用旧的 ThreeDRefTR_SP 模型别名、batch 10、训练到 100 epoch

该文件不是 README 官方结果的可靠复现物。在当前代码和环境上的完整验证为：

| 指标 | 当前实测 |
| --- | ---: |
| Position Acc@0.25 | `56.910%`，`5,411 / 9,508` |
| Position Acc@0.50 | `45.383%`，`4,315 / 9,508` |
| Mask Acc@0.25 | `58.4455%` |
| Mask Acc@0.50 | `47.5915%` |
| Mask semantic mIoU | `40.6728%` |

因此，本设计只把它视为所有 trial 的公平公共锚点，不声称它复现 README 的
44.72% mIoU。训练前将文件模式设为 `0444`，之后每次启动都校验 path、size、
mode 和 SHA-256，任何不一致均 fail closed。

### 旧 Optuna 的问题

旧目录
`/root/autodl-tmp/DATA_ROOT/output/tuning/mcln_source_choice_continue_optuna20_20260628_005200`
只完成 trial 0。该 trial 训练四个 epoch，旧 runner 因 Python 3.7 不支持
`Path.unlink(missing_ok=True)` 在清理阶段中止。

旧设计不能直接复用：

- 搜索 4/6/8 epoch，不满足固定两个 epoch；
- 目标只有 `0.8 * REC@0.25 + 0.2 * REC@0.50`；
- 不把 Mask Acc@0.50 和 mIoU 纳入选择；
- 固定了历史上较差的 `default,mask_text` source pair；
- 每个 trial 留一份约 0.79 GB 权重会超过当前磁盘预算。

### 已确认的训练接线缺陷

完整模型联合训练时，selector 参数当前进入普通 decoder optimizer group，
实际使用 `--lr`，而不是 `--source_choice_selector_lr`。只有 selector-only
训练路径正确使用 selector LR。

`models/losses.py` 已支持 `mask_loss_scale` 和
`consistency_loss_scale`，但主 CLI 和 `_compute_loss` 尚未暴露和转发它们。
直接运行旧脚本会得到含无效维度的 Optuna 结果，必须先修复并测试接线。

## 范围

本轮范围包含：

- MCLN backbone、decoder、box heads 和完整 mask head 的联合续训；
- source pair `default,default_rank_blend_contrastive010`；
- source-choice selector 的联合训练和部署评分；
- selector、mask head、普通 decoder 和 backbone 的互斥 optimizer groups；
- mask、consistency 和 selector loss 权重调节；
- train-only scene-disjoint Optuna、固定两个 epoch、20 个成功 trial；
- 一次从 epoch-54 重新开始的完整 train 长训；
- 长训后针对候选 backbone 重建 sidecars；
- 五指标、权重、源码、命令和环境的可复现归档。

本轮不包含：

- 在 20 个 trial 中加载或训练旧 parent/geometry/joint sidecars；
- 把 epoch-71 保护权重作为第二个 Optuna 初始化锚点；
- 从随机初始化开始两 epoch 搜索；
- 用 official validation 选择 Optuna 参数；
- 改动 ScanRefer annotations、split membership 或 metric 定义；
- 覆盖、删除或修改已有保护权重和 sidecar 内容。

## 数据划分与隔离

复用项目已有的 seed-0、90/10 authoritative scene split：

- fit：506 scenes，33,040 条 ScanRefer expressions；
- calibration：56 scenes，3,625 条 ScanRefer expressions；
- 总计：562 train scenes，36,665 条 ScanRefer expressions。

划分按 `scan_id` 进行。同一 scene 的 ScanRefer expressions、joint detection
rows 和任何派生记录不得跨 fit/calibration。scene ID 列表、原始顺序、数量和
canonical JSON SHA-256 写入 run manifest。

fit loader 保留正式训练的数据增强和 joint detection 合同。calibration loader
关闭增强、固定顺序、只包含 ScanRefer expressions。Optuna runner 不创建
official validation dataset，也不得打开 `val_v3scans.pkl`、ScanRefer val
annotation 或历史 validation cache。文件访问 smoke audit 需要证明这一点。

## 总体数据流

1. 校验 epoch-54、数据和代码快照。
2. 在 56 个 calibration scenes 上对 epoch-54 fixed-default 路径执行一次基线评估。
3. 每个 trial 重新加载同一 model state，重新初始化 optimizer 和 selector，重置 RNG。
4. 在 506 个 fit scenes 上严格训练两个 epoch。
5. 每个 epoch 后评估 3,625 条 calibration expressions；只用 epoch 2 选优。
6. 20 个成功 trial 完成后按预声明约束和目标生成唯一 `best.json`。
7. 重新加载 epoch-54，在全部 562 train scenes 上按 best 参数长训。
8. 为选中的新 backbone 重新生成 parent、geometry 和 joint sidecars。
9. 对最终完整系统执行 official validation 和发布 gate。

短训 checkpoint 不作为长训初始化，因为它只见过 fit scenes。长训重新加载
epoch-54，使用全部 train scenes 和新 optimizer。

## Optimizer 参数组

optimizer group 必须互斥、覆盖全部 `requires_grad=True` 参数，并按稳定名称排序。
任何参数重复、遗漏或落入错误 group 都终止启动。

1. `source_choice_selector.*`：使用独立 `selector_lr`。
2. 完整 mask head：使用 `decoder_lr * mask_head_lr_multiplier`。
3. `backbone_net.*`：使用 `decoder_lr * 10`。
4. `text_encoder.*`：保持冻结，不进入 optimizer。
5. 其他可训练参数：使用 `decoder_lr`。

完整 mask head 的精确前缀为：

- `x_mask.`
- `x_query.`
- `rel_encoder.`
- `swa_layers.`
- `swa_ffn_layers.`
- `out_norm.`
- `out_score.`

`text_query_proj` 当前不在已验证的运行时 mask-head 数据路径中，不纳入专用
mask group；它稳定归入普通 decoder group，即使随后观察到梯度也不改变本轮
optimizer group contract。

## 搜索空间

固定项：

- sampler：`TPESampler(seed=0, n_startup_trials=5)`；
- 成功 trial 数：20；
- 每个 trial：严格两个 epoch，不 pruning；
- batch size：18；
- num workers：4；
- weight decay：`5e-4`；
- clip norm：`0.1`；
- selector hidden dim：288；
- source pair：`default,default_rank_blend_contrastive010`；
- default source：`default`；
- selector target：`precision_gain_default_sourcewise_focal_bce`；
- RNG seed：0；
- optimizer：fresh AdamW，不加载 epoch-54 optimizer/scheduler state。

搜索维度：

| 参数 | 分布 |
| --- | --- |
| `decoder_lr` | log-uniform `[5e-6, 4e-5]` |
| `mask_head_lr_multiplier` | categorical `{1, 2, 4}` |
| `selector_lr` | log-uniform `[2e-4, 2e-3]` |
| `mask_loss_scale` | log-uniform `[0.5, 4.0]` |
| `consistency_loss_scale` | log-uniform `[0.1, 2.0]` |
| `selector_loss_weight` | log-uniform `[0.1, 1.0]` |
| `selector_min_iou_gap` | categorical `{0.02, 0.03, 0.05, 0.08}` |

前三个 trial 为预声明 seed presets，计入 20 个成功 trial：

| preset | decoder LR | mask LR 倍率 | selector LR | mask scale | consistency scale | selector weight | IoU gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical balanced | `2e-5` | 1 | `5e-4` | 1.0 | 1.0 | 0.5 | 0.03 |
| mask-focused | `1e-5` | 4 | `5e-4` | 2.0 | 0.5 | 0.5 | 0.03 |
| conservative balanced | `8e-6` | 2 | `1e-3` | 2.0 | 0.25 | 0.2 | 0.05 |

## 指标与 Optuna 选优

每次 calibration 评估直接从 evaluator counters 生成结构化 JSON，不依赖日志中
五位小数的正则解析。必须记录：

- `learned_selector` Position @0.25/@0.50 exact hits 和 denominator；
- 同一 trial 的 `fixed_default` Position exact hits；
- Mask @0.25/@0.50 exact hits 和 denominator；
- Mask semantic IoU sum、denominator 和 mIoU；
- source fix/break counts、selector source ratio 和所有 loss 均值。

epoch-54 calibration baseline 的 Position 指标使用 fixed-default 路径。对 trial
定义五个增量：trial learned-selector Position 两项减 baseline fixed-default 两项，
trial mask 三项减 baseline mask 三项。

trial 只有同时满足以下条件才可行：

1. 完整评估 3,625 条 calibration expressions，所有数值 finite；
2. learned-selector Position @0.25 hits 不低于同一 trial fixed-default hits；
3. learned-selector Position @0.50 hits 不低于同一 trial fixed-default hits；
4. 相对 epoch-54 calibration baseline 的五个增量均不小于零。

对可行 trial 的单目标分数为：

`100 * (min(delta_5) + 0.25 * mean(delta_5))`

该目标优先改善五项中的最弱项，再比较整体平均改善，禁止用某一项的大幅提升
掩盖另一项下降。同分时依次比较 Position @0.25 hits、Position @0.50 hits、
Mask @0.50 hits、mIoU 和较小 trial number。

如果 20 个成功 trial 中没有可行项，runner 写出 `selection_status=no_feasible_trial`
并停止，不启动长训。不得把最小退化项描述为“最优可发布参数”。

## Trial 生命周期与恢复

Optuna study 使用 SQLite 持久化。`--n-trials 20` 的语义是数据库中最终存在
20 个 `COMPLETE` 且通过结构校验的独立 trial；`FAIL`、`RUNNING` stale 或被中断
的 trial 不计入 20。恢复时计算缺少的成功数量，而不是再盲目追加 20 个。

每个 trial：

1. 创建唯一临时输出目录并写 command/config；
2. 校验输入、split、GPU、磁盘和已有 study contract；
3. 训练 epoch 1 和 epoch 2；
4. 原子写入两个 epoch metrics 和 trial receipt；
5. 只把当前全局 best trial checkpoint 硬链接到稳定路径；
6. 删除该 trial 的其他 `.pth`，保留日志、config 和 metrics；
7. 再次校验 epoch-54 未变化。

临时中断不得删除可诊断日志。非有限 loss、OOM、指标缺失、row count 不匹配、
schema drift、split digest drift 或 checkpoint drift 都把 trial 标为失败。

## 长训合同

只有存在可行 `best.json` 才启动长训。长训重新加载 epoch-54，fresh optimizer，
使用 best trial 参数和全部 562 train scenes，训练 global epoch 55 至 100，
共 46 个 continuation epochs。

短训和长训使用相同的 46-epoch cosine LR 因子：

`factor(t) = 0.5 * (1 + cos(pi * t / T))`

其中 `T = 46 * steps_per_epoch`，各 optimizer group 乘同一 factor，保持 LR 比例。
短训只执行该 schedule 的前两个 epoch，而不是构造一个两 epoch cosine；这样短训
观察到的更新尺度与长训开头一致。

official validation 只在预声明 global epochs
`60,65,70,75,80,85,90,95,100` 运行，不参与 Optuna。每次运行都输出完整五指标
JSON。长训使用原子覆盖的 `latest.pth` 恢复文件，另最多保留三个互不支配的
五指标 Pareto checkpoints。

Pareto retention 先删除被支配 checkpoint；若仍超过三个，依次保留：

1. 距离五个最终 target 的总短缺最小者；
2. Position @0.25 最大者；
3. Mask @0.50 与 mIoU 平衡分最大者。

相同路径的重复身份只保留一次。删除前必须确认稳定 hardlink/copy 和 receipt 已
落盘。长训结束后从排名第一的 backbone 开始生成新 sidecars；只有该候选最终
失败时才处理下一个 Pareto checkpoint。

## Sidecar 重建与最终评估

旧 parent、geometry 和 joint artifacts 继续只读保存，但不得接到新 backbone。
对候选 backbone 执行：

1. 基于当前 runtime 重新提取完整 train candidate cache；
2. 重新训练 parent reranker；
3. 重新生成 geometry cache 并训练 geometry reranker；
4. 重新生成 joint box-mask cache，并按既有 train-only gate 训练 joint selector；
5. 校验每个 artifact 内的 backbone SHA-256 和上游 artifact SHA-256；
6. 对完整系统运行一次 9,508 条 official validation 发布评估。

任何 sidecar train-only gate 失败都回退到它声明的 baseline，不得发布失败权重。
最终 receipt 必须声明 `inference_uses_ground_truth=false`。

## 磁盘与 checkpoint 保留

当前 `/root/autodl-tmp` 约有 12 GiB 可用，单个 MCLN checkpoint 约 0.79 GB。
每次 trial 和长训启动前要求至少 8 GiB 可用，否则 fail closed。

稳定保留上限：

- 原始 epoch-54：一份，位于仓库现有路径；
- Optuna 全局 best trial：一份；
- 长训 `latest.pth`：一份；
- 长训 Pareto checkpoints：最多三份；
- sidecar artifacts、结构化 receipts 和源码快照。

trial 的 epoch 1、非 best epoch 2 和重复 `last` checkpoint 在 receipt 写完后删除。
不得删除当前保护基线的任何 artifact。

## 测试与验证

实现采用 TDD，至少覆盖：

- selector LR 在联合训练中确实进入独立 group；
- mask-head group 前缀、LR 倍率、参数互斥和完整覆盖；
- `mask_loss_scale`、`consistency_loss_scale` 从 CLI 精确传入 loss；
- 非法 scale、重复/遗漏 optimizer 参数 fail closed；
- seed-0 scene split 的 506/56 scenes、33,040/3,625 rows 和 digest；
- tuning 期间禁止构造或访问 official val；
- 固定两个 epoch、20 个成功 trial、stale/failed trial 恢复；
- exact counters、五项 delta、feasibility 和 tie-break；
- checkpoint 只保留 best/latest/Pareto 合同；
- Python 3.7 兼容的清理实现；
- dry-run command 和 source pair；
- 单 batch GPU forward/backward，确认四个主要 LR group 都有 finite gradients；
- 一个最小两 epoch smoke study 能恢复并发布结构化 best receipt。

代码完成后先运行 focused tests，再运行项目完整 CPU regression suite。正式启动前
执行 GPU smoke、epoch-54 checksum recheck、磁盘 preflight 和 official-val
file-access audit。

## 可复现产物

每次 study 根目录必须保存：

- `optuna.db`；
- `study_contract.json`；
- `baseline_metrics.json`；
- `trials.csv` 和逐 trial JSON receipt；
- `best.json` 或明确的 no-feasible receipt；
- 完整 shell command 和 resolved argv；
- Python、PyTorch、CUDA、Optuna、GPU 和 package 清单；
- train annotation、scene split、epoch-54 和输出 checkpoint SHA-256；
- 排除 weights/data/cache 后的源码快照及 manifest SHA-256；
- stdout/stderr、PID、hostname、开始/结束时间和退出状态。

`docs/REC_3DRES_OPTIMIZATION_LOG.md` 在以下节点追加简要交接记录：代码修复与测试、
study 启动、20 组完成与最优参数、长训启动、每个正式候选、sidecar 重建和最终
五指标。记录失败实验，但不需要粘贴完整训练日志。

## 完成条件

本阶段“启动完成”的条件是：实现和测试通过、epoch-54 已保护、train-only baseline
已生成、Optuna study 已以正确合同运行且可以从 SQLite 恢复。

整个目标只有在最终新系统同时通过五项 official validation gate、发布权重和代码
快照已校验、实验日志已更新后才算完成。仅完成 20 组搜索或仅启动长训都不能宣称
指标目标已经达成。
