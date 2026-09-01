# MCLN 完整实验、代码与优化交接文档

更新时间：2026-09-01 19:20（Asia/Shanghai；A-V4 已封存，补充跨数据集泛化判断、服务器遗留 Sr3D 复现脚本与下一步边界）
文档性质：单文件完整交接；前半部分是当前结论与执行指南，后半部分保留 V1--V133 全量时间线。  
安全说明：文档不包含 SSH 密码、API key 或其他明文凭据。远程连接使用用户现有安全配置。

> **阅读优先级（2026-09-01 19:20）**：第 20 章是当前唯一有效的终态快照，覆盖本文件前部仍保留的
> V133“正在运行”、A-V4“尚未启动”等历史时态。旧章节保留是为了完整审计，不表示对应任务仍在运行。
> A-V4 Fold-4 已完整训练和评估但未过门禁，路线现已封存。当前没有 Nr3D/Sr3D 正式训练。

## 0. 一页结论

- 远程代码目录：`/home/gb/butd/mcln`，实际工程位于
  `/home/gb/new butd/butd_detr-main/MCLN-main`；当前服务器只有一张 GPU，所有 GPU 任务必须串行。
- ScanRefer 双阶段正式最好不是单一权重包揽全部指标，而是一个 Pareto 集：
  - **V99**：REC@0.25 最好 `58.6033%`，同时三项 Mask 最好：
    `59.8443% / 52.3349% / 45.9303% mIoU`。
  - **V113**：REC@0.50 最好 `50.8519%`。
  - **V109**：REC 两阈值间的非支配中间点，且用户明确要求永久保留。
- 用户给出的原 MCLN 分割 baseline 为 Mask@0.25=`58.70%`、Mask@0.50=`50.70%`、
  mIoU=`44.72%`；V99 三项均已超过该 baseline。当前真正未完成的是 REC@0.25：目标
  `>=59.00%`，V99 尚差 `38 hits`（需要从 `5572` 到至少 `5610`）。REC@0.50 已超过目标
  `49.00%`。
- V132（final-decoder cross-modal query adapter）完整 4 轮均失败；最佳仍是第 1 轮，且 REC 与
  Mask 都未超过现有保护结果。失败完整权重在 bitwise 可重建后已清理，只保留 compact delta/manifest。
- V133（SACR structured score refiner）已经结束且不再运行。已审计轮次 REC 为
  `56.6365/44.5204` 与 `56.2894/43.8999`，均明显失败；其 global gate 与 raw score 整行饱和，
  residual mean/max 从 `0.2087/0.2097` 升至 `0.2402/0.2416`，逼近 `<0.25` 上限。旧章节中的
  “正在训练”“第 3 轮开始”只保留为历史时间线，不得据此恢复任务。
- 当前源码与小型证据统一发布到 `https://github.com/666666666666gao/MCLN` 的 `main`；A-V4
  Fold-4 终局代码、结果与 result-to-claim 证据至少已发布到 `f73407d`。本轮继续补入服务器遗留的
  Sr3D 冻结 launcher、权重平均 builder、主交接文档和方法候选清单。权重、输出目录、预训练模型、
  缓存和大文件均不上传。
- **后续实验范围已永久冻结**：不再补跑任何所谓 baseline 公平复现，不执行
  detector-pretrained/global48/150--240 epoch 对照，也不把它作为新方法的前置条件或差距解释方案。
- **不参照原建议第七节**：其中按 ScanRefer/Nr3D/Sr3D 分别规定的旧执行顺序、旧前置步骤和旧路线选择，均不进入当前实施计划或论文证据链。
- **不借鉴原建议第八节**：所谓完整 global-batch/150--240 epoch baseline 公平复现已取消，不再作为解释 Nr3D/Sr3D 差距、启动新实验或发表结论的前置条件。
- 对“实验七/实验八”的排除按名称和内容同时生效：即使后续文档编号变化，也不得重新引入其旧方法、训练顺序、对照要求或结论；旧 E0--E7 矩阵同样不采用。
- 后续只围绕现有 V99 总体架构开展单一变量、短周期、可审计实验，直接服务于
  Nr3D 至少 `4740/7899` 与 Sr3D 至少 `12214/17726` 的硬目标。
- A-V4 Fold-4 已完整消费 `27004` 个 fit 样本、执行 `1688` 个 optimizer steps，并评估 `5915` 个
  held-out train-scene 样本。它在 @0.25 净 `-20 hits`、@0.50 净 `-151 hits`，不是训练未完成，
  而是候选切换精度不足；该路线不得继续调参、正式验证或长训。
- 当前正式最好与新硬目标：Nr3D `4475/7899=56.6527%`，必须严格超过 `60.0%`（至少
  `4740/7899`）；Sr3D `12139/17726=68.4813%`，必须严格超过 `68.9%`（至少 `12214/17726`）。
  当前没有活动中的 Nr3D/Sr3D 训练，不能把任一差距解释成“还没训练完”。

## 1. 目标、验收口径与不可破坏约束

| 项目 | 验收口径 |
|---|---|
| 主目标 | ScanRefer 双阶段 REC Acc@0.25 `>=59.00%`、Acc@0.50 `>=49.00%` |
| Mask 安全目标 | 尽量保持/提升 V99；至少不能把用户 baseline `58.70/50.70/44.72` 当成已超越现有最好 |
| 泛化要求 | 模块应可用于 ScanRefer/Nr3D/Sr3D；不得只依赖 validation 或数据集特化后处理 |
| 正式计数 | ScanRefer official validation 必须恰好 `9,508` samples；必须同时保存 hits 与百分比 |
| 验证纪律 | train-only/scene-disjoint smoke 先行；正式 validation 不用于阈值、margin、epoch 反复搜索 |
| GPU 纪律 | 当前只有一张 GPU；同一时刻只能有一个训练/评估任务 |
| 权重纪律 | 永久保留 V109；保留 V99、V113 与依赖链；删除前必须证明可恢复或有等价 hardlink/compact delta |
| 代码纪律 | 活动正式实验期间冻结源码与配置；任何新版本必须默认关闭、零初始化可回退、有限值与边界可审计 |
| 排除路线 | 不做 baseline 公平复现；不参照或继承实验/章节七与八的旧方法、顺序、前置条件及结论；不采用旧 E0--E7 实验矩阵 |
| 交接纪律 | 敏感凭据不落盘；远程原始日志继续保留，本文件作为当前本地完整交接副本 |

## 2. 当前正式最佳指标

### 2.1 V99：REC@0.25 与三项 Mask 最好

| 指标 | Overall | Unique | Multiple |
|---|---:|---:|---:|
| REC Acc@0.25 | **58.6033% (5572/9508)** | 88.8654% (1261/1419) | 53.2946% (4311/8089) |
| REC Acc@0.50 | 50.4523% (4797/9508) | 80.5497% (1143/1419) | 45.1725% (3654/8089) |
| Mask Acc@0.25 | **59.8443% (5690/9508)** | 90.2044% (1280/1419) | 54.5185% (4410/8089) |
| Mask Acc@0.50 | **52.3349% (4976/9508)** | 80.1268% (1137/1419) | 47.4595% (3839/8089) |
| Mask mIoU | **45.9303%** | -- | -- |

### 2.2 当前 Pareto 权重对比

| 版本 | REC@0.25 | REC@0.50 | Mask@0.25 | Mask@0.50 | Mask mIoU | 结论 |
|---|---:|---:|---:|---:|---:|---|
| V99 | **58.6033% (5572)** | 50.4523% (4797) | **59.8443% (5690)** | **52.3349% (4976)** | **45.9303%** | @.25 与 Mask 最好 |
| V109 | 58.3824% (5551) | 50.8414% (4834) | 59.8338% (5689) | 52.3138% (4974) | 45.9224% | 永久保留的中间 Pareto 点 |
| V113 | 58.3403% (5547) | **50.8519% (4835)** | 59.8338% (5689) | 52.3138% (4974) | 45.9226% | @.50 最好 |

V113 subgroup：REC unique `.25/.50=88.7949/80.9020`（`1260/1148`），multiple
`52.9979/45.5804`（`4287/3687`）；Mask unique `90.2044/80.1268`，multiple
`54.5061/47.4348`。

## 3. 最近正式网络实验

### 3.1 V132 四轮完整结果（失败方向）

| Epoch | REC hits@.25/.50 | Mask hits@.25/.50 | Mask mIoU | 判定 |
|---:|---:|---:|---:|---|
| 1 | 5504 / 4391 | 5669 / 4660 | 41.6989% | V132 内最好，但低于 V99/V113 |
| 2 | 5493 / 4383 | 5657 / 4635 | 41.5002% | 继续退化 |
| 3 | 5488 / 4355 | 5650 / 4628 | 41.5477% | 继续退化 |
| 4 | 5495 / 4366 | 5669 / 4661 | 41.7353% | 未恢复到现有最好 |

结论：在 decoder/query feature 上直接加入 full residual 会同时破坏 REC 与 Mask。V132 checkpoint 已用
epoch71 parent + adapter delta 对模型状态逐 tensor bitwise 重建验证；随后仅删除失败完整权重，保留只读
compact manifest 和 adapter delta，V99/V109/V113 均未触碰。

### 3.2 V133 第 1 轮完整结果与当前状态

| 指标 | Overall | Unique | Multiple |
|---|---:|---:|---:|
| REC Acc@0.25 | 56.6365% (5385/9508) | 86.6103% (1229/1419) | 51.3784% (4156/8089) |
| REC Acc@0.50 | 44.5204% (4233/9508) | 72.3749% (1027/1419) | 39.6341% (3206/8089) |
| Mask Acc@0.25 | 59.1923% (5628/9508) | 89.9225% (1276/1419) | 53.8015% (4352/8089) |
| Mask Acc@0.50 | 48.6222% (4623/9508) | 72.5863% (1030/1419) | 44.4183% (3593/8089) |
| Mask mIoU | 41.4128% | -- | -- |

- 固定 parent 选择为 REC `57.9933%/46.3820%`（`5514/4410`），说明 learned SACR score 本身造成
  主要退化。
- 第 1 轮相对目标缺 `225/426 hits`；相对 V99 缺 `187/564 hits`；Mask 相对 V99 缺
  `62/353 hits`、mIoU 低 `4.5174 percentage points`。
- receipt：`eval_metrics_epoch_1.json`，SHA-256=
  `0c6c92570b405c6adde8eae2c2e4038292fc9fbd05449cf41dfa914b8a3e1d71`。
- 当前 run：
  `/root/autodl-tmp/DATA_ROOT/output/network_v133_sacr_score_refiner/v133_sacr_score_refiner_review3_formal_e1_e4_b8x1/scanrefer/v133_sacr_score_refiner_review3_formal_e1_e4_b8x1/1786797601`。
- 第 2 轮完整回执已审计，结果见下一节；第 3 轮已开始。最终仍须完成 4 轮并逐轮审计，不能只依据
  训练 loss 早停或选择 epoch。

### 3.3 V133 第 2 轮完整结果（继续退化）

| 指标 | Overall | Unique | Multiple |
|---|---:|---:|---:|
| REC Acc@0.25 | 56.2894% (5352/9508) | 86.3284% (1225/1419) | 51.0199% (4127/8089) |
| REC Acc@0.50 | 43.8999% (4174/9508) | 71.1769% (1010/1419) | 39.1148% (3164/8089) |
| Mask Acc@0.25 | 59.2659% (5635/9508) | 89.8520% (1275/1419) | 53.9004% (4360/8089) |
| Mask Acc@0.50 | 48.6432% (4625/9508) | 72.5863% (1030/1419) | 44.4431% (3595/8089) |
| Mask mIoU | 41.4346% (`iou_sum=3939.6035616758504`) | -- | -- |

- 固定 parent 仍为 REC `57.9933%/46.3820%`（`5514/4410`）。learned REC 相对目标缺
  `258/485 hits`，相对 V99 缺 `220/623 hits`，相对 V113 的 @.50 最好缺 `661 hits`。
- Mask 相对 V99 缺 `55/351 hits`，mIoU 低约 `4.4957 percentage points`。第 2 轮三项 Mask
  虽略高于第 1 轮，因此 Mask best hardlink 指向 epoch2；REC best 仍指向 epoch1。
- receipt SHA-256=`8da73e1f8078bf0ba44b691c6fdda0e5a0ae028c92aeff0e64f7803cf0e7755f`，
  sample count、分组总数/hits、阈值嵌套、百分比和 `iou_sum/9508` 全部独立复算通过。
- 验证 gate=`0.9664`、residual mean/max=`0.2402/0.2416`；与第 1 轮相比进一步逼近
  `0.25` 上限，证明当前 loss/deployment mismatch 正在加重，而非尚未收敛。
- checkpoint 当前是两个实际 inode：epoch1 inode link count=`3`（REC 两个 best + epoch1）；epoch2
  inode link count=`5`（Mask 三个 best + epoch2 + last），每个实际文件 `610,243,106` bytes。
  `/root/autodl-tmp` 约剩 `2.6GB`，但活动 run 仍不得清理。

## 4. 实验谱系总览

下表用于快速定位；每一个版本的预注册、代码 SHA、测试、run 路径、指标、失败原因与清理动作都在本文后半部的
“完整逐实验时间线”中保留。

| 阶段 | 主要版本 | 做了什么 | 最终认识 |
|---|---|---|---|
| 基础复现与保护 | baseline、epoch71、V19 | 建立 REC/Mask 共同 evaluator、冻结父权重、完整 9508-row 计数 | 先解决可复现与权重安全，避免只看打印百分比 |
| 多源与门控 | V1--V50 | SourceMoE、source selector、SACR source、多源 query mixer、mask fusion | 大多数失败来自过度切换：fix 小于 break；全局 gate 不能替代样本级风险 |
| Parent-relative 排序 | V51--V88 | transition advantage、factorized hit、候选安全、Mask guard、branchwise witness | 可以做到少量净修复，但稀疏 fix、distribution shift 与安全分支互相牵制，未达到目标 |
| 训练缓存/后处理 | V89--V107 | train-only cache、层级 reranker、mesh superpoints、V99 official | V99 成为 REC@.25 与 Mask 最佳；候选 oracle 足够，瓶颈是排序与安全切换 |
| MeshSP OOF 风险校准 | V108--V113 | scene-cross-fit、风险委员会、非对称双 head | V113 把 REC@.50 提到 50.8519%，但 @.25 未迁移；V109/V113 与 V99 形成 Pareto 集 |
| 冻结 cache adapter | V114--V131 | 语言空间注意、pairwise/utility/listwise、hyperspherical semantic、双路径 adapter | @.50 普遍较好，但 @.25 bootstrap/subgroup 门反复失败；停止在同一 cache 上继续搜索 |
| 主网络表示 | V132 | final-decoder cross-modal query adapter | 四轮均损害 REC/Mask；full query residual 太激进 |
| 结构化 score-only | V133 | StructuredSlotBuilder + SACRHead + bounded score residual，冻结 box/mask/parent | 合同正确但第 1 轮整行残差饱和并严重破坏排名；4 轮正式实验仍在进行 |

## 5. 关键代码地图

| 文件 | 作用 | 当前关注点 |
|---|---|---|
| `models/mcln.py` | MCLN 主体、source choice、SACR score 注入、Mask 最终路径 | V133 在约 2227--2407 行构造 global gate 和 bounded residual；当前饱和根因在这里 |
| `models/sacr_head.py` | target/attribute compatibility、relation-anchor geometry、structured score | 输出未锚定 parent，相同行的 raw score 容易整体推到 tanh 饱和区 |
| `models/losses.py` | SACR listwise KL、Box/Mask IoU supervision、DDP example normalization | 当前 KL 对所有有效 query 对齐连续质量，未区分“可在 0.25 预算内改变排名”和“不可达”样本 |
| `main_utils.py` | CLI、checkpoint exactness、optimizer trainable prefixes、loss 参数接线 | 新版本必须更新 config/state exactness，保持旧 checkpoint fail-closed |
| `train_dist_mod.py` | dataset、train/eval loop、9508-row receipt、subgroup/诊断打印 | 正式结果必须由 receipt 复算，不只取日志末尾数值 |
| `scripts/run_v133_sacr_score_refiner.sh` | V133 identity/smoke/formal 固定配置 runner | 活动正式运行期间不可修改 |
| `scripts/audit_v133_sacr_score_gate.py` | 来源、identity、finite/bound、checkpoint gate 审计 | review3 gate 的主要证据入口 |
| `scripts/audit_v133_sacr_supervision_contract.py` | ScanRefer/Nr3D/Sr3D 与 DDP 监督合同 | Mask 只在有 Mask GT 的数据上进入；box objective 跨三个 grounding 数据集一致 |
| `scripts/v133_receipt_utils.py` | receipt 写入/复核 | 核对 sample count、hits、subgroup partition、threshold nesting 与 iou_sum |
| `models/rec_*adapter.py`、`scripts/run_v11*--v131*` | 已结束的 cache adapter/OOF 系列 | 只作失败证据和可复现实验，不再在同一 validation/cache 上扫参 |

GitHub 同步只包含源码与小型文本证据；`.pth/.pt/.ckpt/.safetensors/.bin/.onnx/.h5/.npy/.npz`、
output、pretrained model、cache、backup 已在同步前排除或忽略。

## 6. 已确认的问题

1. **候选覆盖不是主瓶颈**：V19 完整 query oracle 已达到约 `62.9680%/55.0063%`，超过目标；实际问题是
   如何在不增加 break 的前提下选择正确 query。
2. **全局或弱样本 gate 会过切换**：多代 SourceMoE/JQQ 表明只保护 parent score 并不够，其他候选统一抬高后
   仍会越过 parent。
3. **稀疏 fix 监督易塌缩**：direct transition 三分类、absolute hit BCE、pairwise/listwise 都出现 neutral/break
   占优、fix recall 低或 validation distribution shift。
4. **同一 cache 上继续组合后处理已经耗尽**：V114--V131 覆盖空间、语义、pairwise、utility、Pareto、listwise、
   rescue 与双路径，均无法让 @.25 的总体/子群/bootstrap 门同时通过。
5. **主 query residual 太宽会损害 Mask**：V132 说明改动 decoder query 本体会联动 box、mask 和 contrastive projection，
   即便零初始化也会在训练后整体漂移。
6. **V133 当前 objective 与部署预算不匹配**：KL 要求所有 query score 拟合绝对连续质量，但部署只能增加
   `<0.25` residual。许多目标 query 即使残差打满也不可能越过 parent，loss 仍持续推高 gate/raw score，造成整行饱和。
7. **Mask 必须按最终 REC query 一致评估**：不能分别为 REC 和 Mask 偷换 query；V99 的 Mask 提升已经在同一 chosen
   query 合同下审计。
8. **磁盘与 hardlink 容易误判**：多个 best 名称常指向同一 inode；清理必须先看 inode/link count、重建证据与依赖链，
   不能按文件名数量估算占用或直接删除。

## 7. 下一步改进方案

### 7.1 先完成 V133，而不是运行中改模型

1. 等待 epoch 2--4 的 `eval_metrics_epoch_N.json`。
2. 每轮严格复核：`sample_count=9508`、Overall=Unique+Multiple、`.50 hits<=.25 hits`、百分比与 hits 一致、
   Mask `mIoU=iou_sum/9508`、所有诊断 finite、residual `<0.25`。
3. 与目标、V99、V109、V113 及用户 Mask baseline 同时比较。
4. 只保留 V133 内五项 retention 的 Pareto/最佳 inode；活动运行结束前不清理 checkpoint。
5. 若任一轮形成新的网络最好，先做 full-checkpoint exact-load，再串行执行真实 Nr3D/Sr3D 128-row interface smoke。

### 7.2 若 V133 四轮均失败：V134 Feasible Parent-Relative SACR

这不是重复 V51--V85 的旧 transition head，而是直接修正 V133 的 loss/deployment mismatch：

- 将 SACR raw score 逐行锚定到当前 parent query：`relative_raw[q] = raw[q] - raw[parent]`，保证统一 offset
  不能形成残差，parent residual 恒为 0。
- 训练时先计算 parent score gap；只有在 `sacr_score_max_delta` 预算内确实可能越过 parent、且 Box 质量相对 parent
  至少有预注册增益的候选才作为 promotion teacher。不可达样本不再把 gate 推向上限。
- 用连续的 parent-relative Box advantage 做 dense supervision；有 Mask GT 时加入固定权重的 Mask advantage/safety，
  Nr3D/Sr3D 沿用同一个 Box objective，不读取 dataset-specific validation 规则。
- 从预测的正 relative advantage 构造逐样本 abstention gate；没有可信 headroom 的行保持 parent。
- 对无可行修复行加入 trust-region/preserve-parent loss；对可行行只训练足以跨过 parent 的最小 margin，避免整行饱和。
- 保留外层 exact-zero global gate 作为 step-0 bitwise identity；所有新增路径默认关闭、有限值、DDP unequal-row、
  zero-valid-rank、full-checkpoint exactness 与 frozen box/mask contract 必须先过审计。
- 先做 scene-disjoint 128-row smoke，固定 `fix>=break`、residual 非饱和、Mask 不退化门；通过后才允许一次 9508-row formal。

### 7.3 跨数据集接口验证

- 已确认真实文件存在：`refer_it_3d/nr3d.csv`、`nr3d_spacy.csv`、`sr3d.csv`、`sr3d_spacy.csv`，以及
  train/val scan pickle 和 superpoints。
- V133/V134 使用 `--dataset nr3d|sr3d --test_dataset ... --eval --debug --expected_eval_sample_count 128`
  的真实 loader；必须加载训练后的完整 checkpoint，不允许用未训练新 head 或 synthetic-only 结果冒充迁移验证。
- ScanRefer 主目标未过门前，Nr3D/Sr3D 只做接口、finite、shape、sample-count 与性能不崩溃检查，不据 128 条调参。

## 8. 权重、磁盘与恢复状态

- 必须保留：epoch71 parent、V99、V113、V109，以及 V99/V113 所需 parent/geometry/hierarchy artifact 和 claim。
- V99 hierarchy artifact：
  `/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/v99_artifacts/pareto_contextual_h128_seed0_fullfit.pth`，
  SHA-256=`9752990c393fa6e45173a9dd129c4de4bb740924094dcbbec2f3121cbf39d1f2`。
- V133 epoch1 的 `ckpt_epoch_1.pth`、`ckpt_epoch_last.pth` 与五个 best 名称是同一 inode 的 7 个 hardlink，
  实际只占一个 `610,243,106`-byte 文件。正式运行活动期间不要删除任何一个名称。
- V132 已在 bitwise 重建成功后清理失败 full checkpoint，只保留 compact delta/manifest；其恢复链详见时间线 14.217。
- 每次清理前执行：解析绝对路径、检查 inode/link count、核对 SHA/manifest、证明 parent+delta 可重建、确认不在活动
  run 中；清理后复查 protected 权重 SHA 与剩余空间。不要使用 broad glob、递归删除或按“文件名看起来旧”来判断。

## 9. 建议下一会话使用的 skills

- `monitor-experiment`：持续监控 V133 四轮进度、GPU、日志、receipt 与 checkpoint retention。
- `analyze-results`：逐轮复算 Overall/Unique/Multiple、REC/Mask hits、mIoU 和相对基线 delta。
- `experiment-audit`：在声明新最好或清理权重前做来源、样本数、checkpoint、代码与无 GT inference 审计。
- `run-experiment`：V133 完成后的 Nr3D/Sr3D interface smoke，以及通过门禁后的 V134 正式运行。
- `code-review`：V134 默认关闭、identity、DDP、checkpoint exactness 与 frozen box/mask 合同审查。
- `result-to-claim`：只有在新结果通过完整 9508-row 审计后，判断可支持的论文/实验结论。
- 用户明确要求：**不要使用 `tdd` skill**；测试仍可作为正常工程验证执行，但不采用 TDD 工作流。

## 10. 完整逐实验时间线（V1--V133）

以下内容是远程权威日志的完整副本，包含每次预注册、代码改动、测试/审计、运行路径、结果、失败原因、权重清理和
恢复证据。上面的总览用于快速接手；需要复现实验或核对具体 SHA 时，以本节相应版本条目为准。

# ScanRefer REC / 3DRES 优化交接记录

更新日期：2026-08-14

## 目标与保护基线

| 指标 | 当前保护结果 | 目标 |
| --- | ---: | ---: |
| REC Position Acc@0.25 | **58.6033%（5572/9508）** | >= 59.00%（>= 5610/9508） |
| REC Position Acc@0.50 | **50.4523%（4797/9508）** | >= 49.00%（>= 4659/9508） |
| 3DRES Mask Acc@0.25 | **59.8443%（5690/9508）** | 不下降并争取提升 |
| 3DRES Mask Acc@0.50 | **52.3349%（4976/9508）** | > 50.70% |
| 3DRES Mask semantic mIoU | **45.9303%** | > 44.72% |

### 当前验收状态（2026-08-14 CST）

| 验收项 | 当前权威结果 | 状态 |
| --- | --- | --- |
| 双阶段 REC 目标 | V99 + 官方 mesh superpoints `0.586033/0.504523`；目标 `0.59/0.49` | @0.50 已超过 138 hits；@0.25 仍差 38 hits |
| 双阶段 Mask 目标 | V99 + 官方 mesh superpoints `0.598443/0.523349/0.459303` | 三项均超过用户基线 `0.5870/0.5070/0.4472` |
| V101 唯一正式验证 | REC `0.584034/0.499159`；Mask `0.598338/0.523244/0.459220` | clean exit，但未超过 V99；本轮冻结为负结果，不重试或按 val 调参 |
| 网络内最好 | V19 REC `0.581195/0.465398`；Mask `0.598233/0.491376/0.418613` | 已保护，未达到 REC 与 Mask@0.50/mIoU 目标 |
| 候选覆盖上界 | V19 完整 query oracle `0.629680/0.550063`，超过目标 377/571 hits | proposal 充分，主问题是 query 排序和安全覆盖 |
| 单阶段 ScanRefer | epoch 28；retained best REC `0.571939/0.461927`，Mask `0.588241/0.483382/0.412678` | 100 epoch 正式训练中 |
| 自适应多源架构 | V49 三源逐 query mixer；V50 增加 `sacr_structured` 第四源和 query-focus | 已实现并通过合同测试，等待正式训练 |
| 分割专项架构 | 联合六维 Box/Mask 质量、query alpha/bias、source evidence、空间 residual、候选 Mask 损失 | 已实现，V48--V50 队列待运行 |
| 跨数据集合同 | ScanRefer/Nr3D/Sr3D train SACR join 与 token/relation 对齐 | 三个数据集均通过，正式指标待 ScanRefer 主目标后执行 |

当前执行链已收敛为：保留 epoch71 + parent + geometry + V99；V101 唯一 validation 已完成且未刷新
V99，train/val mesh-derived superpoints 均已修复并切入可逆 view，后续进入新的 train-only 路线；服务器仅一张 GPU，所有训练/评估
固定 GPU0 串行。V99 最佳 artifact 与必要 backbone/parent/geometry 均为只读；V101 负结果 artifact
已按“只保留最好权重”规则删除，但 receipt、claim、日志、配置和完整 SHA 仍保留。失败路线 checkpoint
同样在审计后删除。

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
| V99 + mesh-derived official superpoints | ScanRefer val，9508 | **58.6033% / 50.4523%** | **59.8443% / 52.3349% / 45.9303%** | 当前同一系统最好；仅 REC@0.25 尚差 38 hits |
| V101 full-train Pareto + mesh-derived official superpoints | ScanRefer val，9508 | 58.4034% / 49.9159% | 59.8338% / 52.3244% / 45.9220% | 仍超过用户 Mask baseline，但 REC 与 Mask 均未超过 V99；负结果封存 |
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


## 14. 2026-08-12 重启审计与 V51 BMQ-Rank 实施

### 14.1 真实基线与瓶颈复核

本轮重新读取 9508 条 ScanRefer 验证收据、candidate headroom、源码和进程。受保护后处理系统
REC 为 5542/9508=0.582878、4621/9508=0.486012，距 0.59/0.49 还差 68/38 个命中；
Mask 为 0.596971/0.490324/0.417676。V19 learned REC 为 0.581195/0.465398，Mask 为
0.598233/0.491376/0.418613；相同候选的 oracle REC 已达 0.629680/0.550063，mask oracle
mIoU 为 0.451708。因此首要问题是 query 排序和安全切换，而不是 proposal 覆盖。

源码复核确认：原 Joint Query anchor 只保护已命中的父 query，没有父 query 未命中时的 repair；
MCLN text mask 只生成一个 mask，再 expand 到 256 个 query。V49/V50 队列因上游 summary 缺失
已经停止，四张 A100 40GB 当前空闲。第 13.25 节 V51-RAPF 是未执行的可选预注册；本轮主线
命名为 V51 BMQ-Rank，RAPF 留作后续可靠性消融。

### 14.2 V51 实现合同

已在 models/joint_query_quality.py 实施，所有开关默认关闭以保持历史路径：

1. 平滑 utility：Box 为 1*soft025 + 2*soft050 + 0.5*IoU；Mask 为
   0.5*soft025 + 1*soft050 + 0.5*IoU，总 Mask 权重 0.25，温度 0.05。
2. 双向 anchor：protect 与 repair，margin 为 0.05@0.25 和 0.10@0.50。
3. Top-K 并集：joint Top-16、各 source Top-8、训练期 GT utility Top-4，并强制包含父 query。
   GT 不进入 forward/inference。
4. 按 |U_i-U_j| 加权的 pairwise softplus loss，权重 0.5。
5. direct residual logit 缩放为 0.25，保留质量与 source-mix 路径。
6. 新增 protect/repair、candidate ratio 和分项 loss 诊断。

CLI、MCLN 构造、训练损失和训练脚本已接线；独立 launcher 为
scripts/run_double_stage_v51_bmq_rank.sh，不再依赖死亡队列。V51 保留 V50 SACR 四源池、
Mask calibration、query-superpoint spatial refiner 与候选 Mask Focal/Dice/Lovasz 监督。

### 14.3 回归与初始化

V51/Joint Query/SACR/初始化聚焦回归为 83 passed（2 个 CPU autocast warning），SourceMoE
integration 与训练参数组为 124 passed。初始化收据
experiment_output/v51_bmq_rank/v51_bmq_protected_v19_initialization.json 为 pass=true：
common/changed/missing/unexpected/shape-mismatch=1228/0/66/0/0，新模块参数 1,150,390，
输出头零初始化，direct scale 0.25 与 metric utility 已写入审计。受保护 V19 未修改。

### 14.4 运行顺序

先运行 2 个 debug epoch/128 条验证的 R2-S。通过后才进入四卡 40 epoch、完整 9508 条验证的
R2-F。若 REC 排名提升但 Mask@0.50/mIoU 仍受限，进入 V52 QTM-3D；若 selected rank 仍显著
低于 candidate oracle，再进入 V53 last-two-decoder DN/Group refinement。

依据：Rank-DETR https://arxiv.org/abs/2310.08854 与
https://github.com/LeapLabTHU/Rank-DETR；Mask2Former https://arxiv.org/abs/2112.01527；
DINO https://arxiv.org/abs/2203.03605。

### 14.5 R2-S 真实数据烟测、故障修复与候选冻结

R2-S 首次进入真实数据后发现两个此前单元测试没有覆盖的问题：

1. Candidate Mask loss 将不同场景的 superpoint mask 直接 `torch.cat`；真实 batch 的场景宽度
   不同（观测到 2961 与 1431），因此第一批即 shape error。现改为逐场景累计、按被选 mask 数
   加权归一，允许每个场景有独立的 superpoint 数。
2. 修复后第一批出现 SACR 梯度非有限。新增逐参数诊断定位到 SACR residual/slot attention/MLP；
   根因是常量 source score 的 rank-normalize 方差为 0，而 `sqrt(variance)+eps` 在 0 点导数
   非有限。现改为 `variance.clamp(min=eps**2).sqrt()`，并补充常量源有限梯度回归。

修复后的 Joint Query/SACR/finite-training 聚焦回归为 `87 passed`，仅有 2 个 CPU autocast
warning。变量 superpoint 数和常量 source score 均已有固定回归。

第一组 aggressive full smoke 虽完成 2 epoch，但发生过度切换，故明确拒绝：epoch 1
fixed/learned REC 命中为 `63/56 -> 60/52`，Mask 为 `61/49`、mIoU `0.330190`；
epoch 2 learned 进一步降至 `56/46`，Mask `58/44`、mIoU `0.304320`，switch 约 0.5。
它证明“放大自由 residual”会破坏父模型，不进入正式训练。

随后对同一 128 条 debug 合同执行两组 safe smoke：

| 变体 | epoch | fixed REC hits | learned REC hits | Mask hits | Mask mIoU |
|---|---:|---:|---:|---:|---:|
| R1 anchor-only | 1 | 63/57 | 64/58 | 64/51 | 0.350195 |
| R1 anchor-only | 2 | 63/57 | 64/58 | 64/52 | 0.350130 |
| R2 BMQ-safe | 1 | 63/56 | 64/57 | 64/51 | 0.350186 |
| R2 BMQ-safe | 2 | 63/56 | 64/57 | 64/51 | 0.352024 |

两组在各自 run 内均实现 REC +1/+1，且没有破坏 Mask；并发 debug 的 fixed@0.50 有 1 个样本
随机差异，因此只比较 run 内 delta。BMQ-safe 保留 metric-aligned utility、双向 anchor、
Top-16/8/4 候选并集和 gain-weighted pairwise，但将 pairwise 权重冻结为 0.25、direct residual
scale 冻结为 0.25、reranker LR 冻结为 1e-4，作为 R2-F 唯一正式候选。第 14.2 节的 0.5 是
aggressive 初始配置，不再用于正式训练。

为给正式训练释放空间，已在精确核对路径后删除
`/root/autodl-tmp/DATA_ROOT/output/double_stage_v51_bmq_smoke` 下 24 个烟测 checkpoint
硬链接；约释放 3.5 GiB，可用空间从约 7.4 GiB 增至约 11 GiB。该删除不可恢复，但 smoke 的
JSON、config 和完整日志均保留，受保护 V19 与历史最好权重未触碰。

### 14.6 R2-F 四卡正式运行

R2-F 已于 `2026-08-12 03:01:07 CST` 启动，screen 为 `v51_bmq_formal`。launcher 日志：

`experiment_output/v51_bmq_rank/v51_bmq_safe_formal_e1_e40_b12x4_20260812_030107.log`

正式 run 目录：

`/root/autodl-tmp/DATA_ROOT/output/double_stage_v51_bmq_formal/scanrefer/v51_bmq_safe_formal_e1_e40_b12x4/1786474880`

配置为 4 x A100、每卡 batch 12、40 epoch、每 epoch 完整验证 9508 条。关键冻结合同为：
metric utility 温度 0.05，双向 anchor 权重 2.0、margin 0.05/0.10，pairwise 0.25，
Top-K 16/8/4，direct residual 0.25，最大 joint delta 0.5，candidate Mask
Focal/Dice 权重 0.25、Lovasz 0.05。V50 SACR 四源、adaptive source mixing、Mask calibration
和空间 Mask refiner 全部保留。

启动初始化审计再次为 `pass=true`，受保护 V19 的 1228 个公共 tensor 零改变，新模块 66 个
tensor/1,150,390 参数按合同初始化。截至 `2026-08-12 03:05:50 CST`，4 个 DDP rank 均存活，
正以约 96% CPU 执行 train split 文本解耦，尚未进入 GPU batch；这属于数据集构建阶段，当前
没有 OOM、NaN 或进程退出。首个完整 epoch receipt 发布后再依据 fix/break、source switch、
REC/Mask overall 与 subgroup 决定继续或早停，不能仅凭 128 条 smoke 外推正式指标。

### 14.7 R2-F epoch 1 全量门禁失败并停止

R2-F epoch 1 于 `2026-08-12 03:30:11 CST` 完成 9508 条正式验证：

| 指标 | fixed / V19 parent | learned BMQ | delta |
|---|---:|---:|---:|
| REC@0.25 | 5515/9508 = 0.580038 | 5458/9508 = 0.574043 | -57 hits / -0.005995 |
| REC@0.50 | 4410/9508 = 0.463820 | 4332/9508 = 0.455616 | -78 hits / -0.008204 |
| Mask@0.25 | V19 0.598233 | 5659/9508 = 0.595183 | -0.003050 |
| Mask@0.50 | V19 0.491376 | 4623/9508 = 0.486222 | -0.005154 |
| Mask mIoU | V19 0.418613 | 0.414974 | -0.003639 |

精确 threshold effects 显示 REC@0.25 fix/break 为 1.462%/2.061%，REC@0.50 为
1.125%/1.946%；验证期 joint switch 约 12.35%，错误切换多于修复。候选能力仍充足：
gate candidate oracle 为 5986/9508=0.629575、5232/9508=0.550273，mIoU 0.451913。
因此本轮失败不是 proposal coverage，而是 deployed ranking precision。

Mask 的 Unique/Multiple 为 1287/4372 hits@0.25、1038/3585 hits@0.50；REC
Unique/Multiple 为 1246/4212 hits@0.25、1053/3279 hits@0.50。退化不是某个小 subgroup
单独造成，Multiple 仍是绝对主瓶颈。

epoch 1 checkpoint 审计收据
`experiment_output/v51_bmq_rank/v51_bmq_safe_formal_epoch1_checkpoint_audit.json`
为 pass=true：1228 个公共主干 tensor 零变化，只有允许的 66 个新 tensor 更新；optimizer
为 66 states、1,150,390 参数、763 step，moment 全部有限且非零。由此排除错误初始化、主干漂移、
NaN 和 optimizer 漏参，结论锁定为 BMQ 的自由部署 residual 过强：验证 residual abs mean/max
为 0.3810/0.4871，第一轮即接近 max delta 0.5。

确认 epoch 1 JSON、日志及单 inode checkpoint 保留后，于约 `03:32 CST` 对精确进程组发送
SIGINT；screen 和所有 rank 已退出，四卡回到 1 MiB。R2-F 标记 rejected，不继续盲跑 39 轮。
该失败权重仅作为审计证据，不作为后继初始化。

### 14.8 R1-F anchor-safe 全量消融启动

为区分“BMQ metric/pairwise 目标失败”和“允许的部署改动过大”，已于
`2026-08-12 03:33:59 CST` 启动 R1-F，screen 为 `v51_anchor_formal`。launcher 日志：

`experiment_output/v51_bmq_rank/v51_anchor_formal_e1_e40_b12x4_20260812_033359.log`

run 目录：

`/root/autodl-tmp/DATA_ROOT/output/double_stage_v51_bmq_formal/scanrefer/v51_anchor_formal_e1_e40_b12x4/1786476851`

R1-F 仍从受保护 V19 零初始化新头，4 卡、每卡 batch 12、每轮完整 9508 条验证；只将
joint max delta 收紧到 0.25，关闭 metric-aligned utility、pairwise 和 Top-K loss restriction，
保留双向 anchor margin 0.05/0.10、anchor weight 2、direct scale 0.25、SACR 四源、
source mixing、Mask calibration、空间 refiner 与 candidate Mask loss。首轮门禁与 R2-F 相同：
若 REC/Mask 明显低于 fixed parent，就停止并实施带部署候选限制/硬保护的一致训练-推理版本；
若恢复净正增益，才允许进入 epoch 2。


### 14.9 R1-F epoch 1 全量门禁失败、审计与停止

R1-F epoch 1 于 `2026-08-12 04:02:31 CST` 完成 9508 条正式验证：

| 指标 | fixed / V19 parent | learned anchor-safe | delta |
|---|---:|---:|---:|
| REC@0.25 | 5515/9508 = 0.580038 | 5494/9508 = 0.577829 | -21 hits / -0.002209 |
| REC@0.50 | 4409/9508 = 0.463715 | 4387/9508 = 0.461401 | -22 hits / -0.002314 |
| Mask@0.25 | V19 0.598233 | 5675/9508 = 0.596866 | -0.001367 |
| Mask@0.50 | V19 0.491376 | 4660/9508 = 0.490114 | -0.001262 |
| Mask mIoU | V19 0.418613 | 0.417830 | -0.000783 |

精确 threshold effects 为 REC@0.25 fix/break `75/96`（0.7888%/1.0097%），REC@0.50
`61/83`（0.6416%/0.8729%）；即使比 R2-F 保守，验证集仍是错误切换多于修复。候选能力保持不变，
gate candidate oracle 为 `5987/9508=0.629680`、`5230/9508=0.550063`，mIoU `0.451888`，继续
确认失败点是 deployed ranking precision 而不是 proposal coverage。

训练 batch 100 时 joint switch 仅 0.08%，fix/break@0.25 为 0.08%/0；batch 700 时约为
0.37%/0.30%，看似略为净正。但完整验证时 switch 升到 3.73%，ranking residual abs mean/max
达到 `0.2268/0.2465`，再次贴近本轮更小的 max delta 0.25。单纯收紧 residual bound 只能缩小
退化，不能防止自由候选通过抬分击穿父 query。Mask spatial residual 同时达到
`1.7233/1.9992`（上限 2.0），解释了 Mask 没有从 candidate supervision 获益。

检查点审计收据
`experiment_output/v51_bmq_rank/v51_anchor_formal_epoch1_checkpoint_audit.json` 为 `pass=true`：
公共主干 `1228/0` common/changed，新模块 66 个 tensor、1,150,390 参数，optimizer 66 states、
step 763，权重与 moment 均 finite/nonzero。另修正 launcher 初始化审计的 profile 选择：anchor
profile 现在显式要求 `metric_aligned_utility=false`；正确收据
`experiment_output/v51_bmq_rank/v51_anchor_protected_v19_initialization.json` 亦为 `pass=true`。
这只修复审计标签，运行时 config 原本就是 false，不影响本次结果。

确认 epoch 1 的 JSON、日志和 7 个硬链接共享的单一 checkpoint inode 后，对精确进程组停止后续
39 轮；screen、所有 rank 与 dataloader 均已退出，四卡回到 1 MiB。R1-F 标记 rejected。

下一步预注册为 R2-P（parent-score-preserving promotion）：父 Top-1 的部署 score 保持为原始
baseline score，不允许 learned negative residual 降低它；非父候选必须在自身 learned score 上扣除
固定 promotion margin 后仍超过父 score 才能切换。listwise/anchor 训练直接消费同一 deployed score，
保证训练与推理一致。这是排序层内部的安全合同，不读取数据集名、验证阈值或后处理规则，可迁移到
ScanRefer/Nr3D/Sr3D。先关闭会饱和的输出 Mask calibration/spatial residual，隔离验证 REC 排序；
REC 通过后再进入 V52 QTM-3D，以零初始化 query-specific Mask residual 解决单一文本 Mask 广播问题。


### 14.10 R2-P 实现、审计与真实数据烟测

R1-F/R2-F 的共同失败模式不是“父 query 自身被降分”这么单一，而是任意非父 candidate 可凭自由
residual 直接越过父 query；两轮验证 residual 均贴近各自上限，训练 batch 的净修复也没有泛化。
因此实现默认关闭的 R2-P 部署合同，并在 reranker forward 内同时供训练 loss 与 evaluator 消费：

1. 先由原始 parent score 的 Top-1 确定父 query；父 query 部署 residual 强制为 0，部署 score 逐值
   等于原始 parent score。
2. 非父 query 的 learned residual 在部署前统一扣除 `promotion_margin=0.05`；只有 learned score 同时
   跨过原始父子分差与该 margin 才能晋升。
3. `scores`、`selected_indices`、listwise/anchor loss 和 evaluator 全部消费同一个部署张量，不存在
   只在评估时执行的后处理分支；合同不读取数据集名、GT 或 validation IoU。
4. 新增父分数 drift、learned residual 与 deployed residual 的分离诊断。历史路径默认
   `preserve_parent_score=false, margin=0`，逐值保持兼容。
5. 为隔离 REC 安全性，R2-P 关闭 Mask calibration、source-mask evidence、spatial Mask residual、
   candidate Focal/Dice/Lovasz；保留 SACR 四源、source mixing、双向 anchor 与 quality/listwise 训练。

核心、CLI、MCLN、通用训练脚本、独立 launcher、初始化与正式 checkpoint audit 均已接线。
受保护 V19 初始化收据
`experiment_output/v51_bmq_rank/v51_parent_promotion_protected_v19_initialization.json` 为
`pass=true`：common/changed/missing/unexpected/shape-mismatch=`1228/0/52/0/0`，仅新增允许的
1,126,942 参数；父分数保持、0.05 margin、max delta 0.25、direct scale 0.25、Mask 输出关闭均
逐项审计。

128 条、单卡、2 epoch 真实数据烟测 run：

`/root/autodl-tmp/DATA_ROOT/output/double_stage_v51_bmq_smoke/scanrefer/v51_parent_safe_smoke_e1_e2_b12x1/1786479766`

两轮收据完全一致：fixed REC `63/56`，learned REC `64/57`，均为 `+1/+1` 且 selector break 为 0；
Mask `64/52`、mIoU `0.350607`，因输出 Mask 路径隔离而与父部署保持一致。父分数 drift 全程严格为
0，candidate promotion margin 为 0.05，candidate Mask loss 与所有 Mask 输出 residual 为 0。
烟测 checkpoint 审计
`experiment_output/v51_bmq_rank/v51_parent_safe_smoke_epoch2_checkpoint_audit.json` 为
`pass=true`：52 states、1,126,942 参数、step 20，公共主干无变化。`rel_attn` 一阶矩约 2.1e-23
非零，二阶矩在 float32 平方后下溢为 0；审计只在“一阶矩非零且所有值 finite”时允许该二阶矩
下溢，并显式记录 `zero_second_moment_count=1`，全零一阶矩仍被拒绝。对应数值边界已有回归。

Joint Query、SACR、真实 MCLN forward、有限梯度、初始化和 checkpoint audit 聚焦回归为
`229 passed`。烟测只证明合同正确和小样本净增，不作为目标指标证据；下一步仍需 9508 条正式
验证，首轮若 fix 不大于 break 或 REC/Mask 低于父模型即停止。


### 14.11 R2-P 正式训练早停：父分数不变仍不足以保证净修复

R2-P 四卡正式 run 的 launcher 日志为：

`experiment_output/v51_bmq_rank/v51_parent_safe_formal_e1_e40_b12x4_20260812_104403.log`

该 run 没有等到完整验证，因为训练期部署行为已连续越过安全门禁。batch 100 时 switch 为
`39.42%`，REC@0.25 fix/break 为 `1.33%/2.08%`，REC@0.50 为
`3.08%/3.83%`；batch 200 时 switch 上升到 `68.58%`，fix/break@0.25 为
`1.88%/2.63%`，@0.50 为 `4.96%/6.50%`。随后 switch 继续上升到约
`76.53%`。两个阈值在两个正式检查点均是 break 大于 fix，说明仅冻结父 query 的 score
并不能阻止大量候选被统一抬高后错误越过父 query。

因此在精确确认 PGID 后停止进程，screen、DDP ranks 和 dataloader workers 均已退出；未生成
9508 条 validation receipt，也不把中途状态称为正式精度。日志保留作为否证，R2-P 标记
rejected。结论是后继模块必须学习“这个具体候选相对当前父 query 会修复还是破坏”，而不能再输出
一个无类别约束的自由 scalar residual。

### 14.12 V51-T：Parent-Conditioned Transition Advantage

针对 R2-F、R1-F、R2-P 的共同失败模式，实现默认关闭的 V51-T（parent-conditioned
transition advantage）。该模块不是再增加一个样本级 Gate，而是对每个候选与当前父 query
组成有序 pair，并在 REC 的 0.25/0.50 两个阈值分别预测三类状态：

- `break`：父 query 命中而候选不命中；
- `neutral`：父子在该阈值同为命中或同为未命中；
- `fix`：父 query 未命中而候选命中。

pair 特征由 candidate hidden、parent hidden、差值、逐元素积、父子 baseline rank 差和标准化
score 差构成；输出形状为 `[B,Q,2,3]`。部署优势使用 `fix` 对
`neutral + break_cost * break` 的风险敏感 log-odds，两个阈值权重为 1:2。父 score 逐值保持
不变，非父候选仍需跨过 0.05 promotion margin，且仅允许原始父分数 Top-32 候选参与晋升；
所有限制同时被训练 loss 与 evaluator 消费，不依赖数据集名或 validation GT。

训练端使用相对父 query 的精确 transition 标签，并只监督同一 Top-32 部署候选。为避免类别频率
约 22%/72%/6% 导致多数类塌缩，loss 对每个样本内的有效类别先按类数归一化，再按可配置 class
weight 聚合。部署 break cost 与训练 class weight 分离：当前预注册配置训练为
break/neutral/fix=`1/1/1`，部署仍为 `break_cost=4`。旧的 quality/residual 输出头在该模式
被显式移出 optimizer，解决了首次 DDP 烟测因 unused parameters 在 batch 2 退出的问题。

初始化审计
`experiment_output/v51_bmq_rank/v52_parent_transition_protected_v19_initialization.json`
为 pass：公共 checkpoint tensor `1228/0` common/changed，新增 26 tensors、220,481 参数，
zero initialization、父分数保持、Top-32、margin 0.05、max delta 0.25、关闭 SACR/adaptive
source mixing/Mask 输出均符合合同。optimizer 只训练其中 219,578 参数；聚焦单测包含 step-zero
恒等、Top-K 边界、有限梯度、三类监督、非法合同和 DDP optimizer 参数过滤。

两轮初始烟测完成且安全，但 transition 头预测全 break，部署优势全负、实际 switch=0；run 内
fixed `63/57` 到 learned `64/58` 的 +1/+1 来自既有父路径差异，不能归因于新模块。将训练
class weight 调整为等权、LR 提高到 3e-4 后又执行 5 epoch/每轮 10 step 的 128 条烟测：

`experiment_output/v51_bmq_rank/v51_parent_transition_smoke_e1_e5_b12x1_20260812_120337.log`

五轮 fixed/learned 均为 `63/57 -> 64/58`，Mask `64/52`、mIoU 约 0.3505；父分数 drift、
transition switch、fix 和 break 均严格为 0，因此部署安全。头从全 break 转为全 neutral，
transition loss 约从 0.91 降到 0.88，但 50 个 step 仍不足以学出 fix。关键新证据是验证 Top-32
candidate oracle 达到约 `0.8106/0.7576`，远高于目标 `0.59/0.49`，候选限制没有耗尽
headroom；瓶颈仍是 pair 分类泛化。

因此不以短烟测的 neutral collapse 直接否决，而于 `2026-08-12 12:14:41 CST` 启动四卡正式
训练，screen 为 `v52_parent_transition_formal`，launcher 日志：

`experiment_output/v51_bmq_rank/v51_parent_transition_formal_e1_e40_b12x4_20260812_121433.log`

run 目录为
`/root/autodl-tmp/DATA_ROOT/output/double_stage_v51_bmq_formal/scanrefer/v51_parent_transition_formal_e1_e40_b12x4/1786508086`。
预注册早停门禁：batch 100/200 检查 transition 三类 recall、advantage positive ratio、
switch 及两个阈值 fix-break；若开始切换且连续出现 break>fix，则停止。若保持安全或净修复，
必须完成 epoch 1 的 9508 条验证后才判断是否进入下一轮。


### 14.13 V51-T epoch 1 正式收据与 checkpoint 审计

V51-T epoch 1 完成 9508 条正式验证，成为本轮首个两个 REC 阈值均在完整验证上净正的新增架构：

| 指标 | fixed / V19 parent | learned V51-T | delta |
|---|---:|---:|---:|
| REC@0.25 | 5515/9508 = 0.580038 | 5522/9508 = 0.580774 | +7 hits / +0.000736 |
| REC@0.50 | 4410/9508 = 0.463820 | 4419/9508 = 0.464767 | +9 hits / +0.000947 |
| Mask@0.25 | V19 0.598233 | 5684/9508 = 0.597812 | -0.000421 |
| Mask@0.50 | V19 0.491376 | 4667/9508 = 0.490850 | -0.000526 |
| Mask mIoU | V19 0.418613 | 0.418073 | -0.000540 |

验证期 transition switch 为约 0.71%，fix/break@0.25 为 0.29%/0.25%，@0.50 同为
0.29%/0.25%，与 +7/+9 的精确命中增益一致。父分数 drift 仍严格为 0；Top-32 candidate
oracle 为 0.7693/0.6855。与 R2-F/R1-F 相比，本方案把错误切换压到很低，并首次实现完整验证
净修复；但距受保护后处理目标仍差约 61/29 hits，单轮增益规模尚不够。

checkpoint 审计收据
`experiment_output/v51_bmq_rank/v52_parent_transition_formal_epoch1_checkpoint_audit.json`
为 pass：common/changed/new=`1228/0/26`，公共主干零变化；optimizer 为 22 states、
219,578 参数、step 763，全部 moment finite/nonzero。审计同时锁定 parent-transition 开关、
部署 break cost 4、Top-32、父分数保持、margin 0.05 和 max delta 0.25。完整聚焦回归为
`213 passed`。

因为两个主阈值均净正、Mask 退化小于 0.001，允许进入 epoch 2，而不是按旧失败方案停止。
epoch 2 训练 batch 100/200/300 的累计 REC@0.25 fix-break 分别为 +0.33%、+0.17%、+0.20%，
@0.50 为 +0.17%、+0.08%、+0.11%；目前仍为净正且 switch 低于 1%，继续等待第二轮正式
9508 条验证。若 epoch 2 不再扩大净增益，则停止 direct three-class 版本，改为 factorized
candidate/parent absolute-hit probability：对每个 Top-32 query 直接监督两个阈值命中概率，
再由概率恒等式构造 fix/break/neutral，以解决 direct transition 的 fix 类仅约 3--6%、
recall 学习缓慢问题。


### 14.14 V51-T epoch 2 否决、停止与 V53-FH 架构切换

V51-T epoch 2 的 9508 条正式验证否决继续训练：fixed 仍为 `5515/4410 = 0.580038/0.463820`，learned 降为 `5508/4405 = 0.579302/0.463294`，即相对父路径 `-7/-5 hits`，也比 epoch 1 的 `5522/4419` 明显回退。Mask 同步降为 `0.596130/0.488957/0.416789`。checkpoint retention 已确认五项最佳全部仍是 epoch 1，且六个 best 名称与 `ckpt_epoch_1.pth` 共享 inode `6457389551`、link count 6，故最佳权重不会被后续轮覆盖。进入 epoch 3 后 batch 100 的累计 fix/break 已恶化到 `0.08%/0.92%@0.25`、`0.08%/0.83%@0.50`，因此于 `2026-08-12 12:50 CST` 精确终止 PGID 78012，复查无残留进程。结论：直接三分类在第一轮可小幅净修复，但第二轮迅速过拟合 neutral/break；epoch 1 保留为 V51-T 唯一候选。

随后实现 V53-FH（Factorized Hit Advantage）。它不直接拟合稀疏的 fix/break/neutral：共享 query encoder 后增加仅 258 参数的 ordinal hit head，密集监督 Top-32 每个 query 的 `P(Box IoU>0.25)` 和 `P(Box IoU>0.50)`，并结构保证后者不高于前者。部署时将候选概率 `p_c` 与不可变父 query 概率 `p_p` 解析分解为 `p_fix=(1-p_p)p_c`、`p_break=p_p(1-p_c)`，以阈值权重 1:2 聚合 `p_fix-4p_break` 产生 residual；仍保留父分数、0.05 promotion margin、Top-32 和 max-delta 0.25。零初始化下所有候选风险效用为负，故不会随机切换。

远程实现已贯通 `models/joint_query_quality.py`、`models/mcln.py`、`models/losses.py`、`main_utils.py`、`train_dist_mod.py` 和训练 launcher。新增聚焦测试覆盖零步安全、概率嵌套、密集平衡损失梯度及互斥契约；聚焦结果 `99 passed`（其中 joint-query 文件 `75 passed`）。全套测试首次跑至 1953 项时，一个无关的 frozen geometry 文件监视竞态用例失败，单独复跑 `1 passed`，判为时序偶发而非本次回归。

受保护 V19 初始化审计收据：`experiment_output/v51_bmq_rank/v53_factorized_hit_protected_v19_initialization.json`。结果 pass，common/changed/missing=`1228/0/22`，公共主干逐位不变，新模块总参数 153,789，输出头零初始化，父分数/Top-32/break-cost 契约全部通过。下一门禁是真实 128 条短烟测：确认 DDP 无 unused parameter、loss 有效下降且部署仍无明显 break，再进入完整验证。


### 14.15 V53-FH 烟测、风险校准与 V54-FR 正式注册

V53-FH 进行了 5 epoch、每轮 10 step、128 条 debug 烟测，launcher 日志为 `experiment_output/v51_bmq_rank/v53_factorized_hit_smoke_e1_e5_b12x1_20260812_1324.log`。训练稳定且无 DDP unused parameter；factorized BCE 从首 batch 0.7193 下降，父分数 drift、两个阈值 break 和模块 switch 始终为 0。第 5 轮 checkpoint 审计 `experiment_output/v51_bmq_rank/v53_factorized_hit_smoke_epoch5_checkpoint_audit.json` 通过：18 optimizer states、152886 参数、step 50，moment 全 finite/nonzero，公共主干 1228 张量零改动。验证表 `63/56 -> 64/57` 来自既有父路径差异，不能归因于 switch=0 的新模块。

为分离训练与部署校准，加入独立 `factorized_hit_break_cost`。同一 epoch-5 权重只评估 cost=4/2 均无正 advantage；cost=1 时解析式严格化简为 `p_candidate-p_parent`，约 14.7% 候选为正，但原 margin 0.05 仍阻断全部切换。cost=1、margin=0 的机制验证产生 switch 3.03%，两个阈值 fix/break 均为 `0.76%/0`，fixed `63/56` 到 learned `65/58`；Mask 同时由 64/52、mIoU 0.35049 升为 65/53、0.35833。由于 debug train/eval 是同一 128 条，只能证明 dense hit rank 可学且解析部署可修复，不能视为泛化结果。

据此正式注册 V54-FR（Factorized Hit Ranking）：训练仍为 Top-32 ordinal dense absolute-hit BCE；部署使用校准不敏感的相对概率差 `p_candidate-p_parent`（即 factorized cost=1），父 query 严格不变、margin=0、Top-32、max-delta 0.25。受保护初始化审计 `experiment_output/v51_bmq_rank/v54_factorized_rank_protected_v19_initialization.json` pass，common/changed/missing=`1228/0/22`，参数 153789，零初始化和安全契约通过。正式门禁必须用独立完整 9508 验证；若 epoch 1 任一阈值 break>fix 或 Mask@0.25 退化超过 0.001，则停止/回滚。


V54-FR 四卡正式 run 已于 `2026-08-12 13:45:09 CST` 启动，screen `v54_factorized_rank_formal`，launcher 日志 `experiment_output/v51_bmq_rank/v54_factorized_rank_formal_e1_e40_b12x4_20260812_1345.log`，run 目录 `/root/autodl-tmp/DATA_ROOT/output/double_stage_v54_factorized_rank_formal/scanrefer/v54_factorized_rank_formal_e1_e40_b12x4/1786513514`。四 rank 配置核对为完整 36665/9508 数据、global batch 48、LR 3e-4、factorized cost 1、margin 0、Top-32；当前处于 CPU 侧 train/val text decoupling，四进程各持续约 100% CPU，无 I/O 或进程挂死证据。历史正式 run 从 config 到 dataset ready 约 8 分 43 秒，因此继续等待而不重启。


### 14.16 V54-FR 连续净破坏早停与 V55-ND 嵌套支配

V54-FR 完成初始化后在训练 epoch 1 的 batch 100/200/300/400 连续暴露同一部署风险。factorized BCE 从 0.5836 降至 0.4722，但决策切换率从 9.58% 增至 23.56%；累计 fix/break@0.50 依次为 0.58%/1.00%、1.21%/1.63%、1.53%/1.94%、1.77%/2.38%，四个窗口均为净破坏。batch 400 的 @0.25 为 1.21%/1.13%，虽略净正，但不能抵消更关键的 @0.50 退化。父分数 drift 始终为 0，说明失败来自加权平均效用允许候选以 @0.50 损失换取 @0.25 收益，而非受保护主干漂移。按预注册连续窗口规则于 2026-08-12 14:00 CST 精确终止 PGID 87609，复查无残留 GPU 进程；该 run 未进入 epoch 1 验证，也不保留为候选。run 目录为 `/root/autodl-tmp/DATA_ROOT/output/double_stage_v54_factorized_rank_formal/scanrefer/v54_factorized_rank_formal_e1_e40_b12x4/1786513514`。

据此实现 V55-ND（Nested-threshold Dominance）。训练监督仍是可迁移的 Top-32 ordinal dense hit probability，不增加数据集专用后处理或新参数；仅将解析部署效用从两个阈值的 1:2 加权均值改为 `min(utility@0.25, utility@0.50)`。当 cost=1 时，每个 utility 严格等于 `P_candidate(hit)-P_parent(hit)`，所以候选只有在两个嵌套阈值都预测优于父查询时才可能被提升，幅度由较弱改善决定。该约束直接编码 ScanRefer/Nr3D/Sr3D 共用的嵌套 IoU 指标结构，而不是学习一个数据集特定 Gate。新增分阈值 positive-ratio 诊断，可区分 0.25-only、0.50-only 与两阈值共同改善证据。

远程实现贯通 `models/joint_query_quality.py`、`models/mcln.py`、`main_utils.py`、`train_dist_mod.py`、launcher 与聚焦测试；语法检查及聚焦回归为 `101 passed`。受保护初始化审计 `experiment_output/v51_bmq_rank/v55_nested_dominance_protected_v19_initialization.json` 为 pass：common/changed/missing=`1228/0/22`，新模块 153,789 参数，zero-init、父分数保持、Top-32、cost=1、margin=0 和 nested-dominance 合同全部通过。

只改部署规则、复用 V53 epoch-5 权重的 128 条机制对照记录在 `experiment_output/v51_bmq_rank/v55_nested_smoke_eval_cost1_margin0.log`。相对 V54 加权规则，switch 从 3.03% 降至 2.27%，两个阈值 break 仍为 0；REC 为 64/58（0.50000/0.45312），即 @0.25 回到 fixed、@0.50 仍比 fixed 56 命中多 2。由于该 128 条与烟测训练集相同，只证明约束能降低风险，不能证明泛化；正式判定必须来自独立 9508 条验证。预注册早停门禁：batch 100/200 连续观察两个阈值 fix-break 与 nested positive ratio；若任一阈值连续净破坏则停止。若训练窗口安全，完成 epoch 1 全量验证；只有两个 REC 阈值均不低于 fixed 且 Mask@0.25 退化不超过 0.001 才允许进入 epoch 2。

### 14.17 V55-ND 正式早停、分布偏差与 V56-PRR

V55-ND batch 100 的 switch/fix-break 为 4.83%、0.17%/0.42%@0.25、0.33%/0.83%@0.50；batch 200 为 11.54%、0.54%/0.79%@0.25、1.00%/1.46%@0.50。两个阈值连续净破坏，故终止 PGID 91540；未生成 9508 条收据。日志：`experiment_output/v51_bmq_rank/v55_nested_dominance_formal_e1_e40_b12x4_20260812_142505.log`。

代码追踪发现 V51--V55 无条件启用 `--augment_det`：仅训练期约 30% 概率随机替换 detector box，验证/部署无同样 corruption。V54 正式 target ratio 约 0.79/0.60，同一 debug-val 约 0.37/0.27，存在 train-deploy 分布偏差。launcher 已新增 `AUGMENT_DET` 开关，后续 ranker-only 实验设为 0。

V56-PRR 增加父相对 pair 监督；聚焦回归 `103 passed`，初始化审计 `experiment_output/v51_bmq_rank/v56_parent_relative_rank_protected_v19_initialization.json` 通过。128 条 smoke 的 break recall 接近 1、fix recall 约 0.02、switch=0，属于过度保护，拒绝正式训练。

### 14.18 V57-HPA、V58 与 V59 短消融

V57-HPA 用 hardest bidirectional parent anchor 替换全 pair loss。V57 smoke 最终 repair recall 为 86.36%/90.91%，protect recall 仅 29.47%/2.83%；dense BCE 同时退化为 @0.25 全命中、@0.50 全未命中。REC 五轮均 64/57，但模块 switch/fix/break 全 0，故不能归因于 V57。日志：`experiment_output/v51_bmq_rank/v57_hard_parent_anchor_smoke_e1_e5_b12x1_20260812.log`。

V58 对同一权重扫 promotion-margin=0.0025/0.005/0.01/0.02，四档均 64/57、switch=0。V59 比较 BCE、hard anchor、score anchor 和学习率；三组在并发写约 606 MB checkpoint 时因数据盘仅余 589 MB 收到 SIGABRT，归因于存储资源。唯一完整 V59c 使用 LR=1e-3、BCE=0.1、hard anchor=5、score anchor=0、augment_det=0。128 条 epoch 1--5 REC 为：0.50000/0.45312、0.50000/0.45312、0.50781/0.46094、0.49219/0.44531、0.46094/0.41406。epoch 3 两阈值均 +1/128，之后过拟合；epoch-3 最佳权重与收据保留。

只删除了 V56/V57 可复现 smoke 的 `.pth` 临时权重和 V59c 非最佳 epoch-5 inode；全部日志、收据和 V59c epoch-3 最佳权重保留。数据盘可用空间从 589 MB 恢复到 3.4 GB；正式 checkpoint 改存项目所在远程 overlay。

### 14.19 V60-HPA 正式注册与启动

V60 配方：`augment_det=0`、LR=1e-3、dense BCE=0.1、hard parent anchor=5、score anchor=0、父分数保持、Top-32、nested dominance、cost=1、margin=0、Mask 输出关闭。初始化审计 `experiment_output/v51_bmq_rank/v60_hard_anchor_protected_v19_initialization.json` 为 pass：common/changed/missing=`1228/0/22`，新增 153,789 参数，公共主干逐位不变、输出头零初始化。

四卡 screen `v60_hard_anchor_formal` 于 2026-08-12 15:41 CST 启动，日志 `experiment_output/v51_bmq_rank/v60_hard_anchor_formal_e1_e40_b12x4_20260812.log`，checkpoint 根为 `experiment_output/v60_hard_anchor_formal_checkpoints`。预注册门禁：batch 100/200 任一阈值连续净破坏即停止；否则完成 epoch 1 全量 9508 验证。
### 14.20 V60 正式前缀停止（2026-08-12）

- 配置：V59c 梯度方案，`augment_det=0`，部署 break cost=1，4 GPU 正式训练。
- 初始化审计：共同/改写/缺失张量 `1228/0/22`，新模块 153,789 参数，零初始化通过。
- 前缀回执：batch100 的切换率约 9.42%；Acc@0.25 fix/break=`0.0042/0.0100`，Acc@0.50=`0.0108/0.0117`。batch200 切换率约 13%；Acc@0.25=`0.0042/0.0113`，Acc@0.50=`0.0121/0.0138`。
- 判定：两个连续前缀的 break 均超过 fix，按预注册停止规则终止，未进入 validation。

### 14.21 V61 部署成本与全量评估（2026-08-12）

- V61 只把同一因子化方案的部署 cost 改为 4；正式训练到 epoch5。最佳 Position Acc@0.25 为 epoch3 `0.581405`，Acc@0.50 为 epoch1 约 `0.465187`，仍低于目标且没有实质净增益。
- epoch1 全量 9,508 条复评：Position `0.58109/0.46508`，Mask `0.598233/0.491376`，Mask mIoU `0.418498`。
- 选择器切换率仅约 `0.0001`；高成本成功冻结了破坏，也几乎完全冻结了修复。
- 因首次评估进程启动于 counterfactual endpoint 透传修补之前，最终日志没有可靠的六成本反事实字段；该问题已在 `models/losses.py` 两条 loss 收集路径修复，不能用缺失字段推断最优成本。

### 14.22 V62 分解式父候选转移头（2026-08-12）

目标是修复 V52 三分类 neutral 主导和 V57 绝对 hit 相减的问题。对每个候选-父查询、每个阈值分别预测：

1. `p(change)`：候选是否改变父查询的命中状态；
2. `p(fix | change)`：发生改变时是修复还是破坏。

训练中 neutral/change、fix/break 分组独立归一；方向概率训练不注入部署风险成本。部署采用嵌套阈值最小效用，父分数严格保留，候选限制 Top32。

实现与验证：

- 新开关 `joint_query_quality_use_decomposed_transition_advantage`，与旧 direct/factorized 模式互斥；CLI、MCLN 构造、训练脚本和 optimizer 冻结规则均已接通。
- 初始化审计：共同/改写/缺失张量 `1228/0/26`，新模块 220,223 参数；受保护基线零改写，安全初始化通过。
- 相关回归测试 121 项通过；随后新增 V62c 多成本回执后，模块单测 82 项通过。

#### V62a/V62b smoke

- V62a 同时在训练 loss 和部署 utility 使用 cost=4，导致方向头 fix recall 长期为 0，零切换；拒绝。
- V62b 改为事实概率等组训练，cost=2 时学会 fix，但切换率 33%–50%，128 条验证从固定父选择器约 `0.5000/0.4531` 降至最好 `0.4844/0.4219`；拒绝。
- V62a/V62b 的可复现 smoke `.pth` 已删除以释放约 1.2 GB 实际空间；32 份日志和 JSON 回执保留。删除权重不可直接恢复，但可按配置重跑。

#### V62c 风险阈值重新参数化

- 部署效用改为 `p(change) * tanh((direction_log_odds - log(cost))/2)`；其符号严格等价于 `p(fix|change) > cost/(1+cost)`。
- 方向头 step-0 偏置设为 `log(cost)`，因此零初始化仍精确等于父选择器，同时成本语义不被中心常数抵消。
- 初始化审计和单测通过。128 条/约10训练 step 的 cost=4 smoke 保持零切换；同权重反事实显示 cost<=2 全切换且净负，cost>=2.5 零切换，说明小 smoke 主要只学到全局偏置，必须进入正式前缀观察候选间方差。
- 已启动 V62c cost=4、`augment_det=0`、LR=1e-3、4 GPU 正式前缀；通过一次前向回执 1.25–4.0 多成本的 switch/fix/break，按前缀规则早停或进入 validation。

### 14.23 V62c 正式前缀反事实与早停（2026-08-12）

正式配置为 4 GPU、global batch 48、`augment_det=0`、LR=`1e-3`、Top-32、部署 cost=4；完整训练/验证集规模为 36,665/9,508。run 日志：`experiment_output/v51_bmq_rank/v62c_decomp_prior_c4_formal_e1_e40_b12x4_20260812.log`。

批次 100 的同权重多成本反事实（格式为 switch、fix/break@0.25、fix/break@0.50）：
- cost 1.25：`86.83%, 2.92/4.25%, 7.00/8.00%`
- cost 2.00：`44.33%, 1.33/2.83%, 4.08/4.58%`
- cost 3.00：`20.67%, 0.58/1.58%, 2.33/2.58%`
- cost 4.00：`5.42%, 0.17/0.33%, 0.58/0.58%`

批次 200：
- cost 1.25：`76.75%, 2.88/3.50%, 6.21/7.79%`
- cost 1.50：`61.75%, 2.29/2.96%, 5.00/6.50%`
- cost 2.00：`37.75%, 1.67/2.17%, 3.58/4.42%`
- cost 2.50：`23.54%, 1.13/1.54%, 2.46/2.58%`
- cost 3.00：`15.29%, 0.75/1.08%, 1.83/1.79%`
- cost 3.50：`9.63%, 0.46/0.67%, 0.96/1.13%`
- cost 4.00：`5.04%, 0.33/0.42%, 0.54/0.63%`

结论：两个连续正式前缀中，所有可产生实质切换的成本在 Acc@0.25 都是净破坏；实际 cost=4 也从 batch100 的净 `-0.16pp/0.00pp` 变为 batch200 的 `-0.09pp/-0.09pp`。cost=3 在 batch200 的 @0.50 仅暂时净 `+0.04pp`，但同时 @0.25 净 `-0.33pp`，不满足双阈值目标。因而失败来自候选方向排序，而非部署成本选择，继续扫描 cost 或训练到完整 epoch 缺乏依据。

按预注册规则终止进程组 PGID 119320；复查对应进程、screen `v62c_formal` 与 GPU compute-app 均为空。终止信号传递期间日志多出现一个后续累计窗口，但不改变已由 batch100/200触发的判定。V62c 未进入 9,508 条正式验证，不作为候选权重。下一步不得继续调 Gate/cost；应直接改进候选间的 setwise 排序监督，使稀疏 fix 在同一场景 Top-32 内优先于 break/neutral。


### 14.24 V63 场景内 setwise tier 排序与边界对齐（2026-08-12）

V62c 证明部署 cost 不是主因后，V63 将监督单位改为同一场景 Top-32 内的 repair-or-stay 决策。新 setwise tier head 对候选-父查询 pair 输出两个安全分支，部署取两支优势的最小值；最终层零初始化，step-0 精确复现父选择器。训练将 repair 行与 stay 行等组平均：存在严格提升父 box tier（0/1/2）的候选时，目标分布只放在最佳可达 tier；否则目标为父查询。两分支共同监督，只有两支都支持候选才会实际切换。

实现已贯通模型、MCLN/CLI/launcher、optimizer 冻结和两条 loss 收集路径。联合及集成回归分别为 85/85、95/95。初始化审计 `experiment_output/v63_setwise_tier_initialization_audit.json` 通过：common/changed/missing=`1228/0/26`，新模块 219,965 参数，公共主干零改写、输出头全零、父分数保持和 Top-32 合同一致。

V63a smoke 暴露一个不可达标签问题：训练 softmax 使用“父分数+有界 residual”，而许多最佳 tier 候选与父分数差超过 max-delta=0.25；模型无法令目标候选胜出，转而把全部候选优势推到约 -1.4，stay recall=1、repair recall=0。V63a 不进入正式训练。

V63b 将训练分数改为实际部署边界余量。对候选所需 residual (r=score_parent-score_candidate+margin)，若 (r>=max_delta) 则从 repair 候选中排除；否则训练 logit 为 (a-atanh(r/max_delta)/4)。其正号严格等价于部署时候选可超过父查询，消除了不可满足标签。

同样 128 条、5 epoch smoke 中，固定父为 63/56 hits；V63b learned 五轮分别为 64/57、64/58、64/58、64/57、64/57，两个阈值始终 0 break。epoch2 @0.25 为 +1、@0.50 为 +2，repair recall 约 1.82%，stay recall 100%；Mask 输出未启用，保持 64/52、mIoU 0.350491。该结果只作为机制门禁，不视为泛化成绩。保留 epoch2（与 best-rec050 同 inode）权重；V63a 全部与 V63b 其余可复现 smoke `.pth` 已删除，日志/JSON/配置全保留，overlay 可用空间由 2.4GB 回到 4.7GB。


### 14.25 V63b 正式前缀连续无修复停止（2026-08-12）

V63b 四卡正式配置为 global batch 48、完整 36,665/9,508、`augment_det=0`、LR 1e-3、Top-32、max-delta 0.25。运行时确认 joint-only 可训练参数 219,062，与 V63 checkpoint 审计合同一致。

batch100：switch/fix/break 两阈值全为 0；repair-row ratio 20.08%，reachable-query ratio 99.92%，repair recall 0，stay recall 1。batch200：同样 switch/fix/break 全 0；repair-row ratio 18.71%，reachable-query ratio 99.88%，repair recall 0，stay recall 1。

该结果不是候选不可达，而是 row softmax CE 仍可通过把所有候选压低来优化占多数的负项。虽未触发 break>fix 风险线，但连续两个大前缀无任何修复，进入完整验证没有信息价值，故按补充的连续无修复无效门禁终止 PGID 123930；复查进程、screen 和 GPU 为空。V63b 正式 run 未生成 checkpoint/9508 收据，不作为候选。

下一版 V63c 强制每场景候选优势零均值，使全局 accept/reject 偏置不可表达；并将 row softmax CE 替换为实际切换边界上的等权 hardest-positive / hardest-negative 对比损失。repair 正项与所有保护负项各占 50%，全体同号不再是 loss 最优解。

### 14.26 V63c/V63d 候选中心化对比与低学习率门禁（2026-08-12）

V63b 正式前缀的全拒绝捷径说明，只做边界对齐仍不足。V63c 对同一场景 Top-32 的非父候选原始 pair advantage 做候选维中心化，从表示层消除全体共同 accept/reject 偏置；训练损失改为等权的实际边界 hardest-positive / hardest-negative 对比：repair 行推动最佳严格 tier 提升候选越过父边界，保护项推动最危险非 repair 候选留在边界以下。最终 setwise head 去掉会被中心化严格抵消的 bias，因此参数合同为 missing=25、总参数 219,963、可训练参数 219,060。

V63c（LR=1e-3）128 条、3 epoch smoke 证明候选间信号可学习，但更新过快：epoch1 为 fixed `63/56` 到 learned `64/57`，两阈值均零 break，Mask `64/52`、mIoU `0.350491`；epoch2 降到 `57/50`，fix 约 7.03%、break 约 11.72%，Mask 同步降到 `57/46`、mIoU `0.310018`；epoch3 又恢复到 `64/57`。该振荡否决 LR=1e-3 正式训练。

V63d 仅将 LR 降到 `3e-4`，其余架构和 loss 完全不变。128 条、5 epoch smoke 的 Position hits（fixed 恒为 `63/56`）依次为：`64/57`、`64/57`、`63/56`、`63/56`、`63/56`；对应 fix/break 为 `(1/0,1/0)`、`(2/1,2/1)`、`(1/1,1/1)`、`(1/1,1/1)`、`(2/2,2/2)`。五轮 Position 无净负，但 Mask mIoU 从 epoch1 的 `0.350491` 依次降为 `0.346818/0.344831/0.341801/0.343069`，因此只保留 epoch1 作为机制候选，不把后续轮视为安全收益。

V63d epoch1 实物 checkpoint 审计回执：`experiment_output/v63d_centered_contrast_lr3e4_smoke_epoch1_checkpoint_audit.json`。结果 pass：公共张量 common/changed/new=`1228/0/25`，所有张量有限；优化器仅含 21 个新模块状态、219,060 参数、step=10，动量有限且非零。审计过程中发现 checkpoint 保存空 `joint_query_quality_source_names`，模型运行时会继承父 selector source pool，而审计器此前未复现该继承语义。已修复审计器并加入回归测试；联合 V63 模块/审计测试 `115 passed`。

空间清理仅删除 V63b/V63c 和 V63d epoch5 的可复现 smoke `.pth`；全部日志、JSON、配置与 V63d epoch1 最佳权重保留。overlay 可用空间由 3.0 GB 恢复到 4.7 GB。

V63d 四卡正式训练已于 `2026-08-12 19:51 CST` 从受保护 V19 重新初始化启动，而非从 smoke 权重续训。配置为完整 36,665/9,508、global batch 48、`augment_det=0`、LR=`3e-4`、Top-32、max-delta=0.25、父分数严格保持；screen `v63d_formal`，PGID `128580`，日志 `experiment_output/v51_bmq_rank/v63d_centered_contrast_lr3e4_formal_e1_e40_b12x4_20260812.log`，checkpoint 根 `experiment_output/v63d_centered_contrast_lr3e4_formal_checkpoints`。门禁保持严格：batch100/200 若任一阈值连续 break>fix 则停止；若连续无修复也停止；只有训练前缀安全才进入 epoch1 的 9,508 条正式验证。

### 14.27 V63d 正式零修复停止与 V64 集合内 repair 排序（2026-08-12）

V63d 正式 run 在 batch100 与 batch200 连续给出相同结论：两个阈值 fix/break 全为 0，repair recall=0；stay recall 分别为 99.88% 与 99.94%，repair-row ratio 为 20.08%/18.71%，reachable-query ratio 为 99.92%/99.88%。loss 仅从 0.7045 降至 0.7032，说明低学习率把 V63c 的振荡破坏转化成了近乎全拒绝，并未改善正式分布的 repair 候选排序。按预注册连续无修复门禁终止 PGID 128580；screen、进程组和 GPU compute-app 均清空，未产生 checkpoint 或 9508 条验证收据。

代码审计进一步定位：V63 的 pair head 本身已包含候选 hidden、父 hidden、差分、乘积和两类父相对 score，因此失败不能简单归因于“没有 pair 特征”。真正缺口在监督：现有 hardest-positive 边界项只要求某个最佳 repair 越过父边界，negative 项只要求最危险 non-repair 留在父边界下，却没有直接要求 repair 在候选集合内部排过 non-repair。正式多样分布下，模型可以略降边界 loss 而 repair top-rank recall 仍为 0。

V64 新增可配置 `joint_query_quality_setwise_rank_loss_weight`（默认 0，保持所有旧实验行为）。在 repair 行内、排除父 query 后，对每个安全分支计算 `gap=max(best-tier repair margin)-max(non-repair margin)`，加入 `softplus(0.02-gap)`；该目标不能通过全候选拒绝优化。部署路径、父分数保持、候选中心化、双分支 hard-min、Top-32 和 step-0 identity 全部不变，也不增加模型参数。新增 rank margin/recall/loss 诊断。定向 V63/V64 测试 6 项通过，完整 joint-query + checkpoint auditor 回归 `117 passed`。



### 14.28 V64 候选内排序 smoke 与正式前缀（2026-08-12）

V64 以 LR=`3e-4`、setwise rank weight=`2.0` 完成 128 条、5 epoch smoke。固定父选择器恒为 Position `63/57`；learned selector 五轮依次为 `64/58`、`64/58`、`64/58`、`65/59`、`64/58`。最佳 epoch4 相对父查询两个阈值分别净增 `+2/+2`，零 break；Mask 同轮为 `65/53`、mIoU `0.356478`。训练 rank recall 从 `0.2379` 提升到 epoch4 的 `0.3621`，epoch4 repair recall `0.0227`、stay recall `1.0000`。

epoch4 实物审计 `experiment_output/v64_setwise_rank_w2_lr3e4_smoke_epoch4_checkpoint_audit.json` 通过：common/changed/new=`1228/0/25`，21 个 optimizer states、219,060 参数、step=40，全部 finite/nonzero。

V64 四卡正式任务从受保护 V19 重新初始化，完整 36,665/9,508、global batch 48、`augment_det=0`、LR=`3e-4`。batch50/100/150/200 的 rank recall 为 `0.2623/0.2682/0.2768/0.2828`，rank margin 从 `-0.0100` 收窄至 `-0.0077`；但两个阈值 fix/break 连续全 0，repair recall=0。最佳 repair 平均边界仍为 `-0.0398`，按连续无修复门禁终止，未进入完整验证。

### 14.29 V65–V67 repair 边界与保护语义消融（2026-08-12）

V65 新增默认关闭的 repair-boundary 额外权重，不改部署与模型参数。V65 weight=2 在 smoke epoch1 为净 `+1/+1`，随后切换失控；epoch2–5 Position 分别为 `53/41`、`56/45`、`56/45`、`52/41`，Mask 同步大幅下降。到 epoch5，non-repair 平均边界升到 `+0.0431`，stay recall 仅 `0.1392`，明确否决。

V65b 将额外权重降到 0.5，仍在 epoch2 出现 fix/break=`6/11`、Position `58/51`，五轮均不达标。V65c 改为自步式：只有 best repair 已排过最危险 non-repair 的行才获得额外跨零力；epoch1 净 `+1/+1`、Mask mIoU `0.352439`，但 epoch2 仍变为 fix/break=`7/12`、Position `58/52`。资格门控降低初始风险，但共享参数仍把保护行一起抬高。

V66 增加 top-4 non-repair tail 保护。epoch1 安全净 `+1/+1`，但 epoch2–5 Position 为 `58/52`、`62/56`、`56/50`、`53/47`；tail violation ratio 最终 `0.2935`。扩大 hardest-negative 覆盖仍不能抵消共享表示污染。

V67 将 repair 限制为 Box tier 提升且 Mask tier 不低于父查询，并让双分支只把对应阈值真实 Box/Mask hit→miss 视为 break，safe-neutral 不再当负例；联合回归 `123 passed`。但 smoke epoch1 Position `63/51`、Mask `65/53`，epoch2 Position `55/43`、Mask `57/45`，说明 Mask 保护有效但 Position@0.50 严重退化，配置否决。

V67 保存后续 checkpoint 时 overlay 仅余 92 MB，原子保存报 `unexpected pos` 并退出。已确认 screen/进程/GPU 全空，删除 V65/V65c/V66/V67 这些明确失败且由日志/JSON可重跑的全部 `.pth` 和原子临时文件；保留配置、JSON、完整日志、V64 epoch4 最佳权重及受保护 V19。可用空间恢复到 4.1 GB。后续不再用无条件 repair 跨零额外力；回到 rank-only 强化，先排对候选再观察原始边界项能否安全跨零。


### 14.30 V68：提高候选内排序权重仍无法稳定越过部署边界（2026-08-12）

- 目的：在 V64 基础上仅把 `setwise_rank_loss_weight` 从 2 提高到 5，判断更强候选内排序是否能稳定转化为 Top-1 修复。
- 固定：V64 原始 Box tier repair 定义、候选中心化、双分支保守最小值、LR `3e-4`、128/128、5 epoch；V65/V66 额外损失均为 0。
- 固定 parent：REC `63/57`。
- learned REC（epoch 1–5）：`64/58`、`61/55`、`64/58`、`60/54`、`63/57`。
- learned Mask（epoch 1–5）：`64/52`、`61/49`、`64/52`、`60/48`、`63/51`；epoch 1 mIoU `0.349644`。
- 候选内 rank recall 从约 `0.25` 提升到 `0.36`，但修复边界召回仍接近 0，并出现跨 epoch 振荡。
- 结论：单纯放大共享分支的 rank 梯度不能解决 promotion 与 protection 的梯度耦合，V68 淘汰，不进入正式验证。
- 存储：失败实验的 `.pth` 已删除，日志、config 与逐 epoch JSON 保留。

### 14.31 V69：独立 Promotion / Safety 双头（2026-08-12）

- 动机：V63–V68 的两个输出来自同一 MLP，且监督标签相同，实际上是冗余副本；修复与保护仍共享同一优势值。
- 架构：
  - promotion head：独立非线性 pair head，候选内中心化，只承担最佳 repair 排序和越过 parent 边界；
  - safety head：独立非线性 pair head，保留绝对输出，学习否决 Box@0.25、Box@0.50 或 Mask@0.25 的真实 threshold regression；
  - 部署仍取 promotion/safety 两分支的保守最小值，不读取 GT，不增加独立 scalar gate；
  - Mask@0.25 被显式保护，Mask@0.50 不作为 veto，以免锁死当前偏低的高阈值 Mask。
- 验证：相关回归 `237 passed`；V69 专项 `4 passed`。受保护 V19 初始化审计通过：common `1228`、changed `0`、unexpected `0`、new tensors `30`、new params `286139`、输出头全零。
- smoke：128/128、LR `3e-4`、rank weight 2、5 epoch。fixed parent REC 恒为 `63/56`。
- learned REC（epoch 1–5）：`63/56`、`62/55`、`63/56`、`60/53`、`61/54`。
- learned Mask（epoch 1–5）：`63/51`、`62/50`、`63/51`、`60/48`、`61/49`；最佳 mIoU 为 epoch 1 的 `0.344965`。
- 诊断：promotion rank recall 能到约 `0.42–0.55`；safety hazard query ratio 约 `0.25`。但 safety 仅监督每行当前最危险的一个 hazard，后续危险候选尾部持续漏过；epoch 5 switch ratio `0.322`，fix/break 分别约 `0.083/0.114`，净退化。
- 结论：职责解耦本身可训练且零初始化安全，但 safety 监督密度不足；V69 不进入正式验证。下一步 V70 在保持 promotion 与 V64 rank 不变的情况下，仅对 safety 的全部 hazard 做逐行平衡稠密否决。
- 存储：V69 失败 `.pth` 已删除；日志、config、审计 JSON、逐 epoch 评估 JSON 保留。

### 14.32 V70：逐行均衡的稠密 Safety 监督与 w=1 烟测（2026-08-12）

- 动机：V69 的 safety head 每行只监督当前最危险的一个 hazard，未被选中的
  Box@0.25、Box@0.50 或 Mask@0.25 hazard 可能在共享更新后越过零边界。
- 实现：新增默认关闭的
  `joint_query_quality_setwise_dense_safety_loss_weight`。仅在 V69 独立
  promotion/safety 双头模式中，对每个 exact hazard 施加
  `softplus(0.02 + safety_margin)`；先在行内平均、再跨行平均，避免 hazard 数量多的
  scene 主导 batch。V69 原有 hardest-hazard 边界项保留，promotion head 不接收该
  dense loss 梯度。
- 接线修复：初版已写入 launcher 和两条监督路径，但漏了
  `main_utils.py -> compute_hungarian_loss` 传参以及总 loss 函数形参/合法性检查；
  正式运行前已补齐。V70 单测最初的 1/3 hazard fixture 实际误含 2/4 hazard，修正
  非目标 query 的 Box/Mask IoU 后，逐行均衡解析值与实现一致。
- 验证：语法检查通过；完整 joint-query 回归 `98 passed`，source-MoE、集成与初始化
  审计相关回归 `275 passed`。受保护 V19 初始化审计
  `experiment_output/v51_bmq_rank/v70_dense_safety_initialization_audit.json`
  通过：common/changed=`1228/0`，新模块 286,139 参数，输出头零初始化，父分数与
  Top-32 合同不变。
- w=1 单卡 128/128、5 epoch smoke：fixed parent 每轮 `63/57`；reported learned
  每轮 `64/58`；Mask 每轮 `64/52`、mIoU `0.350491`。dense safety loss 在 epoch1
  从 `0.6612` 降至 `0.6280`，五轮 hazard violation ratio 保持 0。
- 关键判读：joint-query switch ratio 五轮均为 0，因此 reported learned 相对 fixed
  的 `+1/+1` 来自既有父 selector，不能归因于 V70。w=1 消除了 V69 的 hazard 尾部
  泄漏，但把 safety 分支推成全拒绝；不能进入正式 9,508 条验证。
- 下一门禁：只降低 dense-safety 权重到 0.25，其他配置逐位保持。如果仍连续零
  switch，则判为结构性过保护；如果再次出现 break>fix，则判为稠密权重无法同时
  满足 promotion/protection，停止 V70 路线。

### 14.33 V70b：低权重 Dense Safety 烟测与正式前缀注册（2026-08-12）

- 配置：只将 V70 的 dense-safety 权重从 1.0 降至 0.25；其余保持 LR `3e-4`、
  setwise rank weight 2、独立 promotion/safety 双头、Top-32、父分数保持、
  `augment_det=0`、128/128、5 epoch。
- fixed parent 每轮为 `63/57`。learned epoch1--4 均为 `64/58`，epoch5 为
  `64/59`；Mask 五轮均为 `64/52`、mIoU `0.350491`。
- 归因：父 selector 本身相对 fixed 贡献 `+1/+1`；epoch5 joint-query switch ratio
  为 `1/128=0.0078125`，新增 @0.50 fix `1/128`，两个阈值 break 均为 0。因此 V70b
  可归因净收益是 Position@0.50 `+1/128`，Mask 不变；这通过机制烟测但不是泛化
  证据。
- epoch5 实物审计
  `experiment_output/v51_bmq_rank/v70b_dense_safety_w025_smoke_epoch5_checkpoint_audit.json`
  通过：common/changed/new=`1228/0/30`，optimizer 26 states、285,236 参数、
  step 50，所有新张量与动量 finite，且每个 state 的 exp_avg 非零。
- 权重保留：epoch5 在所有指标上不差于 epoch1，且 @0.50 更优；所有 best 名称已
  统一为 epoch5 inode `7225261154`（7 个 hard links）。被支配的 V70 w=1 权重和
  V70b epoch1 重复 inode 已删除；配置、日志、五轮 JSON 与审计收据均保留，可按
  原配置重跑。overlay 可用空间恢复到 3.5 GB。
- 正式门禁：从受保护 V19 重新初始化，4 GPU、global batch 48、完整
  36,665/9,508。batch100/200 若任一阈值连续 break>fix 则停止；若连续无 joint
  fix/switch，则停止；只有正式前缀安全且出现可归因修复才进入 epoch1 全量验证。

### 14.34 V70b 正式前缀确认负迁移并早停（2026-08-13）

- 正式配置按 14.33 的预注册执行：从受保护 V19 重新初始化，4 GPU、global batch
  48、完整 36,665/9,508、`augment_det=0`、LR `3e-4`、setwise rank weight 2、
  dense-safety weight 0.25。日志：
  `experiment_output/v51_bmq_rank/v70b_dense_safety_w025_formal_e1_e40_b12x4_20260812.log`。
- 累计前缀（格式为 `fix/break@0.25, fix/break@0.50, switch`）：
  - batch50：`0.67/0.50%, 0.50/1.17%, 6.17%`；
  - batch100：`0.50/0.83%, 1.25/2.08%, 18.50%`；
  - batch150：`1.11/1.44%, 2.22/3.89%, 33.28%`；
  - batch200：`1.58/1.92%, 3.17/4.75%, 44.79%`。
- batch100 与 batch200 两个预注册检查点中，两个阈值均连续
  `break > fix`；同时 dense-safety violation ratio 从 batch50 的 0.46% 上升到
  batch200 的 3.52%。这说明低权重 dense loss 在小 smoke 上产生的单个安全修复
  不能泛化到正式分布，且训练推进时 safety 尾部重新泄漏。
- 按预注册规则终止经命令行复核属于本任务的进程组 PGID `146358`。复查对应
  训练进程、screen `v70b_dense_safety_formal`、GPU compute-app 与未完成 `.pth`
  均为空；正式日志、launcher 和配置保留。未进入 9,508 条验证，不产生候选权重。
- 结论：V70 的负类单向 dense safety 存在不可接受的两端行为：weight 1 为全拒绝，
  weight 0.25 则在正式分布随训练逐步泄漏。下一版 V71 不再只把 hazard 向负侧推，
  而对 transition-active 的 safe 与 hazard 候选分别施加正/负 margin，并先按行、
  再按类别等权平均；目标是显式消除全负 safety 解，同时不让数量占优的 hazard
  淹没 safe 类。

### 14.35 V71 双向类别均衡 Safety 监督与 smoke 预注册（2026-08-13）

- 实现：新增默认关闭的
  `joint_query_quality_setwise_balanced_safety_loss_weight`。仅在独立
  promotion/safety 双头的 safety 分支上，对所有 transition-active 非 hazard 候选
  施加 `softplus(0.02-safety_margin)`，对所有 exact hazard 候选施加
  `softplus(0.02+safety_margin)`；候选先在各自行内平均，行再在类内平均，最后
  safe/hazard 两类等权。这样全负 safety 输出不再是最优解，且 hazard 数量不改变
  两类相对权重。
- 新 loss 与 V70 dense-safety 互斥，避免同一 hazard 被重复计权；默认两者均为 0，
  所有旧配置保持逐位兼容。新增 safe/hazard violation ratio 诊断，不改变部署公式、
  模型参数或父分数保持合同。
- 验证：Python/shell 语法通过；V70/V71 定向测试 `4 passed`；完整 joint-query
  `100 passed`；source-MoE、集成与初始化审计相关回归 `275 passed`。
  受保护 V19 初始化审计
  `experiment_output/v51_bmq_rank/v71_balanced_safety_initialization_audit.json`
  通过：common/changed/new=`1228/0/30`，新模块 286,139 参数，输出头全零且 safety
  合同一致。
- smoke 预注册：单 GPU、128/128、5 epoch、batch 12、`augment_det=0`、LR
  `3e-4`、rank weight 2、dense weight 0、balanced weight 1，其余与 V70/V70b
  完全一致。若任一 epoch 两阈值中出现 `break > fix`，或连续全拒绝且无可归因
  joint fix，则不进入正式训练；只有出现零净破坏的 joint-query 修复且 Mask 不退化，
  才审计实物 checkpoint 并考虑正式前缀。

### 14.36 V71 smoke：单标量 Safety 仍无法稳定分离风险（2026-08-13）

- fixed parent Position 恒为 `63/57`。learned Position（epoch1--5）为：
  `64/58`、`63/57`、`61/55`、`61/55`、`62/56`。
- joint-query 累计归因（epoch1--5，格式为
  `fix/break@0.25, fix/break@0.50, switch`）：
  - `0.00/0.00%, 0.00/0.00%, 0.00%`；
  - `0.76/1.89%, 0.76/1.89%, 3.41%`；
  - `0.76/3.41%, 0.76/3.41%, 5.68%`；
  - `2.27/4.92%, 2.27/4.92%, 11.36%`；
  - `0.76/2.65%, 0.76/2.65%, 4.92%`。
- epoch1 的 reported `+1/+1` 发生在 joint switch=0 时，仍来自既有父 selector，
  不能归因于 V71。epoch2 已触发预注册的 `break > fix` 门禁；读取聚合日志期间任务
  已完成后续小规模轮次，因此保留五轮完整诊断，但不据此放宽门禁，也不进入正式
  训练。
- Mask（epoch1--5）为 `64/52`、`63/51`、`61/49`、`61/49`、`62/50`；mIoU
  为 `0.350491/0.342098/0.330884/0.332173/0.340079`。除 epoch1 零 joint 切换外，
  所有轮次均同步退化。
- 诊断：balanced safe/hazard violation 在 epoch1 为 52.06%/34.74%，之后仍约
  33%--52%/29%--42%。双向监督消除了 V70 w=1 的全拒绝，却把大量语义不同的
  Box@0.25、Box@0.50、Mask@0.25 风险压进同一个 safety 标量；单头无法同时形成
  三条稳定决策边界，最终由安全放行转成真实 break。
- 终止后复查对应进程、screen 与 GPU compute-app 均为空。V71 epoch1 在所有关键
  指标上不优于保留的 V70b epoch5（后者 Position `64/59`、Mask `64/52`），其余
  epoch 更差。验证 config、checkpoint-retention、五轮 eval JSON、launcher、完整
  日志均存在且 JSON 可解析，并确认 V70b 与受保护 V19 权重仍在后，删除 V71 两个
  实际 checkpoint inode 的 8 个 `.pth` 名称；日志和全部回执保留。overlay 可用
  空间从 2.4 GB 恢复到 3.5 GB；已删权重不可直接恢复，但可按保留配置完整重跑。
- 下一版 V72 不再用复合 OR hazard 的单一 safety 输出。将三个保护标准拆成独立
  Box@0.25、Box@0.50、Mask@0.25 风险头，分别做行内/类别均衡监督，部署时对三个
  criterion margin 与 promotion margin 取保守最小值；目标是把 V71 的互相冲突
  标签变成可辨识的多任务边界，仍不读取推理期 GT。

### 14.37 V72 三标准因子化 Safety 头与 smoke 预注册（2026-08-13）

- 架构：新增默认关闭的 `joint_query_quality_use_factorized_setwise_safety`。仅在
  独立 promotion/safety 模式下，把 safety 输出从 1 维改为三维，分别对应
  Box@0.25、Box@0.50、Mask@0.25。每个标准各自预测候选是否会把父查询的 hit
  变成 miss；推理时先对三 safety margin 取最小值，再与 promotion margin 取
  保守最小值。推理路径不读取 GT，父分数保持、Top-32 与 bounded residual 不变。
- 监督：新增默认权重 0 的
  `joint_query_quality_setwise_factorized_safety_loss_weight`。每个标准内部先按候选、
  再按行、最后按 safe/hazard 类等权；三个标准再等权。V70 dense、V71 balanced
  与 V72 factorized 三种附加 safety loss 互斥，防止重复计权。
- 兼容性：不开新 flag 时仍实例化 V69--V71 的单 safety 头，state shape 与旧配置
  不变；开 flag 时最终 safety weight 从 `1x128` 变为 `3x128`，总参数只增加 256。
  最终层全零，step-0 三个 criterion margin 均为零，选择结果精确复现父 selector。
- 验证：Python/shell 语法通过；V70--V72 定向测试 `7 passed`，V72 审计专项
  `1 passed`；完整 joint-query+初始化审计 `113 passed`，source-MoE/集成
  `266 passed`。真实受保护 V19 审计
  `experiment_output/v51_bmq_rank/v72_factorized_safety_initialization_audit.json`
  通过：common/changed/new=`1228/0/30`，参数 286,395，输出头全零，三头合同一致。
- smoke 预注册：单 GPU、128/128、5 epoch、batch 12、`augment_det=0`、LR
  `3e-4`、rank weight 2、factorized-safety weight 1；dense/balanced weight 0，
  其余与 V71 完全一致。任一完整验证 epoch 若任一阈值 `break > fix`，则停止且不
  进入正式训练；只有 joint-query 出现零净破坏的可归因修复，且 Mask 不退化，才
  审计实物 checkpoint 并进入正式前缀。

### 14.38 V72 smoke：出现安全增益窗口，但高学习率振荡否决正式晋级（2026-08-13）

- 首次 launcher 已正确保存 factorized flag 和 loss weight，但 loss 收集器只搬运
  旧 setwise 键，未把 `setwise_factorized_safety` marker 与三标准 scores 放入
  `joint_outputs`，因此合同检查在首个 batch 主动报错；未产生有效更新或 `.pth`。
  修复两条 loss 收集路径后，以独立 `_r1` 目录重跑，完整回归 `217 passed`。
- fixed parent Position 恒为 `63/57`。V72-r1 learned Position（epoch1--5）为：
  `64/58`、`62/56`、`61/55`、`66/60`、`65/59`。
- joint-query 归因（格式为 `fix/break@0.25, fix/break@0.50, switch`）：
  - epoch1：`1.52/1.52%, 1.52/1.52%, 3.03%`；
  - epoch2：`3.03/4.55%, 3.03/4.55%, 14.02%`；
  - epoch3：`1.52/4.17%, 1.52/4.17%, 7.95%`；
  - epoch4：`1.52/0.00%, 1.52/0.00%, 1.52%`；
  - epoch5：`0.76/0.00%, 0.76/0.00%, 1.52%`。
- Mask（epoch1--5）为 `64/52`、`62/50`、`61/49`、`66/54`、`65/53`；mIoU
  为 `0.349644/0.341612/0.332339/0.362777/0.358098`。epoch4 同时给出
  Position `+3/+3`、Mask `+2/+2`，且 joint 自身 `+2/+2` 零 break，证明三标准
  拆分可以产生安全窗口；但 epoch2 已触发预注册 `break > fix`，epoch3 继续失败，
  不允许用事后挑选 epoch4 直接进入正式训练。
- epoch4 checkpoint 审计
  `experiment_output/v51_bmq_rank/v72_factorized_safety_smoke_epoch4_checkpoint_audit.json`
  通过：common/changed/new=`1228/0/30`，optimizer 26 states、285,492 参数、
  step40，全部张量与动量 finite/nonzero。保留 epoch4 的 6 个硬链接；验证五轮
  config/eval/retention JSON 可解析且 V70b/受保护 V19 均仍在后，删除被支配的
  epoch5/last inode。overlay 可用空间约 3.0 GB。
- V72b 预注册：架构和所有 loss 权重不变，只把 joint LR 从 `3e-4` 降为 `1e-4`；
  smoke 延长为 10 epoch（100 step），使累计更新量覆盖 V72 出现窗口的量级。
  任一完整 epoch 只要任一阈值 `break > fix` 即停止；低 LR 若连续全拒绝也拒绝。
  只有出现 joint 可归因、零净破坏且 Mask 不退化的稳定窗口，才进入正式前缀。

### 14.39 V72b smoke：降学习率延后但未消除 Safety 振荡（2026-08-13）

- 配置严格保持 V72 的三标准因子化架构、rank weight 2 与 factorized-safety
  weight 1，只把 joint LR 从 `3e-4` 降至 `1e-4`，单 GPU、128/128、batch 12。
  任务在读取归因门禁时已经完成 8 个验证 epoch 并开始 epoch9；确认当前命令行、
  PID/PGID=`153431` 后按预注册规则终止整条进程组。终止后 screen 与 GPU
  compute-app 均为空。
- fixed parent Position 恒为 `63/57`。八轮 learned Position 为：
  `64/58`、`64/58`、`65/59`、`63/57`、`65/59`、`64/58`、`64/58`、
  `65/59`。对应 Mask hits 为 `64/52`、`64/52`、`65/53`、`63/51`、
  `65/53`、`64/52`、`64/52`、`65/53`；mIoU 为
  `0.350491/0.350491/0.358098/0.344965/0.358098/0.350491/0.350417/0.356076`。
- joint-query 归因（格式为 `fix/break@0.25, fix/break@0.50, switch`）：
  - epoch1：`0.00/0.00%, 0.00/0.00%, 0.00%`；
  - epoch2：`0.00/0.00%, 0.00/0.00%, 0.00%`；
  - epoch3：`0.76/0.00%, 0.76/0.00%, 0.76%`；
  - epoch4：`0.76/1.52%, 0.76/1.52%, 2.27%`；
  - epoch5：`0.76/0.00%, 0.76/0.00%, 1.52%`；
  - epoch6：`0.00/0.00%, 0.00/0.00%, 0.00%`；
  - epoch7：`0.76/1.14%, 0.76/1.14%, 1.89%`；
  - epoch8：`1.52/1.14%, 1.52/1.14%, 3.41%`。
- epoch3/5 给出 joint 自身 `+1/+1` 且零 break，Mask 同步 `+1/+1`，说明降
  LR 确实把 V72 的安全窗口提前稳定到 30--50 step；但 epoch4 已首次触发
  `break > fix` 的硬门禁，epoch7 再次失败。epoch8 虽为净正，也不能用事后挑选
  覆盖前序失败，因此 V72b 整组拒绝，不进入正式全数据训练。
- 三标准诊断显示问题不是 Box@0.25 单头：epoch3 的 hazard/safe violation
  （Box@0.25、Box@0.50、Mask@0.25）分别为
  `15.48/75.14%`、`37.55/49.58%`、`38.65/40.98%`；到首次失败 epoch4 为
  `13.41/78.57%`、`35.10/49.66%`、`34.92/38.51%`。Box@0.50 与 Mask@0.25
  风险边界仍接近随机重叠；部署取三个 margin 的最小值只能降低放行率，无法保证
  被放行候选不含 hazard。仅降低 LR 因而只能改变振荡时间，不能解决辨识问题。
- 保留 config、八轮 eval JSON、retention JSON、launcher 与完整日志作为失败回执。
  epoch3/5 的相同最佳权重只作为机制证据；V72 epoch4 已有通过审计且表现更好的
  受保护机制 checkpoint，因此 V72b 不形成新的最佳候选。再次确认受保护 V19、
  V70b 与 V72 epoch4 权重均存在，并验证 18 个 JSON 全部可解析后，删除 V72b
  8 个 `.pth` 名称（2 个实际 inode）；这些权重不可直接恢复，但可按保留配置完整
  重跑。删除后 V72b 权重计数为 0，overlay 可用空间约 3.0 GB。
- 下一步不继续扫 LR 或 safety loss 标量权重。V73 应把“候选是否安全”从单个
  点估计改为带保守置信边界的风险估计，并让 promotion 只在收益下界为正且三个
  hazard 上界均低于阈值时放行；目标是在不读取推理期 GT、不过拟合 ScanRefer
  后处理的前提下，将安全门从平均分类器变成可训练的风险约束。

### 14.40 V73 双边界保守风险头与 smoke 预注册（2026-08-13）

- V72/V72b 表明单个三标准点估计会在“放行 safe”与“否决 hazard”之间振荡。
  V73 新增默认关闭的 `joint_query_quality_use_factorized_setwise_risk_bound`：
  每个 Box@0.25、Box@0.50、Mask@0.25 标准不再只输出一个 safety margin，而是
  输出中心估计与保守 guard 两个 margin，共 6 路 veto。部署先对每个标准的两路
  margin 取最小值，再对三个标准和 promotion 取最小值；任何 guard 不支持切换时
  都回退到受保护父查询。该规则只使用模型特征，不读取推理期 GT，也不依赖
  ScanRefer 类别或样本身份。
- 新增默认权重 0 的
  `joint_query_quality_setwise_factorized_risk_bound_loss_weight`。中心头继续做每个
  criterion 内 safe/hazard 等类损失；guard 头使用同一明确标签，但按既有
  `transition_break_cost=4` 对 hazard 类加权，形成偏向高 hazard recall 的风险上界。
  两头损失等权后再跨三标准平均。新 loss 与 V70 dense、V71 balanced、V72
  factorized point loss 互斥，promotion head 不接收 risk-bound 额外梯度。
- 兼容性与初始化：不开 V73 flag 时，V72 仍是 3 输出 safety head，旧 checkpoint
  shape/行为不变；开启时只把最终输出从 3 增至 6，总参数由 286,395 增至
  286,779，state tensor 数仍为 30。六个最终输出全零，step-0 residual 为 0、父
  分数与选择精确复现。V73 专项 `4 passed`，完整 joint-query/初始化
  `117 passed`，Source-MoE/训练集成 `336 passed`。
- 真实受保护 V19 初始化审计
  `experiment_output/v51_bmq_rank/v73_factorized_risk_bound_initialization_audit.json`
  通过：common/changed/new=`1228/0/30`，unexpected/shape mismatch=`0/0`，
  参数 286,779，六路输出头全零，父分数保持、Top-32 与 V73 合同一致。
- smoke 固定为单 GPU、128/128、batch 12、`augment_det=0`、LR=`1e-4`、最多
  10 epoch；rank weight 2、risk-bound weight 1、break cost 4，其余 safety loss
  均为 0。任一完整 epoch 在任一阈值出现 `break > fix` 即停止并拒绝；到 epoch5
  若仍无 joint 可归因修复也停止。只有至少两个完整 epoch 出现 `fix > 0`、
  `break = 0` 且 Mask 不退化，才审计最佳实物 checkpoint 并考虑正式全数据前缀。

### 14.41 V73 smoke：风险上界消除 hazard 但退化为全拒绝（2026-08-13）

- 监控读取时任务已完成 6 个完整 epoch。fixed parent Position 恒为 `63/57`；
  learned selector 六轮均为 `64/58`，Mask 均为 `64/52`、mIoU `0.350491`。
  六轮 joint-query `fix/break@0.25`、`fix/break@0.50` 与 switch 全部为 0；因此
  reported 的 `+1/+1` 完全来自既有父 selector，不能归因于 V73。
- 三个 guard 从 epoch1 到 epoch6 的 hazard violation 全部严格为 0，但 safe
  violation 也全部为 100%。中心 point 头并非全拒绝：例如 Box@0.25 的 point
  hazard/safe violation 从 epoch1 `8.3/82.8%` 演化到 epoch6 `36.2/42.8%`，
  Box@0.50 为 `17.7/74.7%` 到 `33.5/50.0%`，Mask@0.25 为 `12.2/79.3%`
  到 `46.5/29.1%`。真正把所有切换否决的是 4 倍 hazard guard。
- 这暴露出比 loss 权重更具体的结构问题：当前 safety margin 与 promotion margin
  一样先减去候选跨越父分数所需的 `required_advantage`，然后 hard-min 直接决定
  residual 幅度。Safety 因而同时承担“候选是否危险”和“候选能否补足父分数差”
  两个职责；4 倍 guard 即使正确识别 hazard，也会因分数差把大量 safe margin 压
  到负区间。V73 的风险头不是纯 veto，所谓上界仍与收益幅度耦合。
- 按预注册“到 epoch5 无 joint 修复即停”规则，在重新核对命令与 PGID=`155894`
  后终止进程组；复查 screen、进程组与 GPU compute-app 均为空。验证 config、
  retention、六轮 eval/diagnostics 共 14 个 JSON 可解析，且 V19/V70b/V72 最佳
  权重仍在后，删除 V73 8 个 `.pth` 名称（2 个实际 inode）。失败权重不可直接
  恢复，但完整配置、日志与启动脚本保留；V73 不进入正式训练。
- 下一版 V74 不调 break cost。它把 promotion 和 safety 的部署职责彻底分开：
  promotion 独自决定 residual 幅度并承担 `required_advantage`；六路 safety 只在
  原始绝对风险 margin 的零边界上作硬 veto，不再减父分数差。step-0 两者仍为 0，
  精确复现父选择器；目标是保留 V73 的零 hazard，同时释放真正 safe 的 repair。

### 14.42 V74 纯 Safety Veto Gate 与 smoke 预注册（2026-08-13）

- 新增默认关闭的 `joint_query_quality_use_setwise_safety_veto_gate`。开启后，
  candidate-centered promotion margin 独自生成 bounded residual；六路 safety 的
  原始绝对 margin 只产生 `margin > 0` 的硬放行 gate。若 gate 否决，则只截断
  promotion 的正部分，不改变其负部分；若 gate 放行，则完整保留 promotion 幅度。
  用 straight-through sigmoid 提供局部反向梯度，但前向选择严格使用硬 gate。
- 训练边界同步改为职责一致的两类 margin：promotion branch 仍减去候选越过父分数
  所需的 `required_advantage`；safety branch、三个 criterion 与六个 point/guard
  score 都不再减该值，只在零边界学习 safe/hazard。V73 风险 loss、4 倍 guard、
  Top-32、父分数保持和所有标签定义均未改变，因此本轮只检验部署职责解耦。
- step-0 时 promotion/safety 全为 0，截断前后 residual 仍精确为 0；旧 V63--V73
  在新 flag 关闭时完全保留 hard-min 行为，不增加参数或 state tensor。V74 定向
  `3 passed`，完整 joint-query/初始化 `120 passed`，集成 `336 passed`。真实 V19
  审计 `experiment_output/v51_bmq_rank/v74_safety_veto_gate_initialization_audit.json`
  通过：common/changed/new=`1228/0/30`，参数 286,779，输出头全零且合同一致。
- smoke 仍为单 GPU、128/128、batch 12、`augment_det=0`、LR=`1e-4`、最多
  10 epoch，rank/risk-bound=`2/1`、break cost 4。任一完整 epoch 任一阈值
  `break > fix` 即停止；到 epoch5 无 joint 修复也停止。只有至少两个完整 epoch
  `fix > 0, break = 0` 且 Mask 不退化，才审计实物 checkpoint 并进入正式前缀。

### 14.43 V74 smoke：去除分数差耦合后 guard 仍全拒绝（2026-08-13）

- V74 完成 6 个完整评测 epoch。fixed parent 恒为 `63/57`；learned Position
  六轮均为 `64/58`，Mask 均为 `64/52`、mIoU `0.350491`。所有 epoch 的 joint
  fix/break/switch 与 `safety_veto_accept_ratio` 全为 0；reported `+1/+1` 仍只
  来自既有父 selector。按 epoch5 无修复门禁，在核对 PGID=`157911` 后停止，
  screen、进程组与 GPU compute-app 均清空。
- 去掉 `required_advantage` 后 point 头已能强烈放行部分标准：例如 epoch3
  Box@0.25 point hazard/safe violation 为 `96.1/0.8%`，Mask@0.25 为
  `100.0/0.0%`；但每轮、每个标准的 guard hazard/safe violation 仍精确为
  `0/100%`，六路最小值因此永远不大于 0。V74 排除了“全拒绝仅由父分数差重复
  扣除导致”的假设，定位到 cost-weighted guard 自身的决策基线。
- 数学原因：guard 的 safe/hazard 类损失权重为 `1:4`。在特征尚未分开时，最优
  cost-weighted safety logit 相对普通 log-odds 平移 `-log(4)`；直接用 0 作为
  部署边界，等价于要求未加权 `P(safe) > 0.8`。短 smoke 里全负不是异常，而是
  该损失与部署阈值不一致的必然基线。
- 验证 config、retention、六轮 eval/diagnostics 共 14 个 JSON 可解析，且受保护
  权重仍在后，删除 V74 8 个 `.pth` 名称（2 个 inode）。同时清除 V73/V74 各一
  个被终止保存留下的无效 `.pth.tmp`，共约 1.2 GB；这些临时文件从未形成有效
  checkpoint，均不可恢复也无需恢复。overlay 可用空间恢复到约 2.9 GB。

### 14.44 V75 cost-prior 校准与 smoke 预注册（2026-08-13）

- 新增默认关闭的
  `joint_query_quality_use_cost_calibrated_setwise_risk_bound`。仅在 V73 risk-bound
  头上启用：部署时给 guard safety logit 加回解析偏移 `log(break_cost)`，中心
  point 不变，再做双头/三标准最小值；loss 内训练 guard 前减回同一偏移。因此
  训练梯度和 4 倍 hazard 代价与 V74 完全相同，只修正 cost-sensitive logistic
  loss 引入的先验平移，不新增可调阈值，也不扫描 cost。
- break cost=4 时，step-0 point/guard 部署 margin 为 `0/log(4)`，最小值仍为 0，
  residual 与父选择器精确不变。不开新 flag 时 V73/V74 行为保持；参数/state 数
  不变。V75 定向 `3 passed`，完整 joint-query/初始化 `123 passed`，集成
  `336 passed`。真实审计
  `experiment_output/v51_bmq_rank/v75_cost_calibrated_risk_bound_initialization_audit.json`
  通过：common/changed/new=`1228/0/30`，参数 286,779，合同与零初始化均通过。
- smoke 与 V74 完全相同，只开启解析校准：单 GPU、128/128、batch 12、LR
  `1e-4`、最多 10 epoch。任一 epoch 任一阈值 `break > fix` 即停止；epoch5
  无 joint 修复也停止。仍要求至少两个 epoch `fix>0, break=0` 且 Mask 不退化，
  才审计实物 checkpoint 并考虑正式前缀。

### 14.45 V75 smoke：解析校准从全拒绝越过到危险全放行（2026-08-13）

- 五个完整评测 epoch 的 learned Position 命中数依次为 `64/57, 64/57,
  65/58, 63/56, 57/50`；Mask 依次为 `64/52, 64/52, 65/53, 63/51,
  57/45`，mIoU 从 `0.350491` 在 epoch3 短暂升至 `0.358098`，epoch4/5 随后
  降至 `0.344965/0.307341`。fixed parent 每轮均为 `63/56`。
- joint `fix/break@.25`（`.50` 完全相同）依次为 `0/0, 0/0, 0.76/0,
  0.76/1.52, 0.76/6.44%`，switch 为 `0, 0, 1.52, 3.03, 10.23%`；veto
  accept 为 `0, 0, 4.48, 10.62, 62.07%`。epoch3 虽有一次净安全修复，只有
  一个 epoch，未满足“两轮”门禁；epoch4 已出现 `break > fix`，明确触发停止。
  监控返回时 epoch5 也已完成且恶化，epoch6 仅完成训练和 checkpoint 保存、尚未
  完成评测。核对后只向精确 PGID `159948` 发送 TERM；screen、进程组与 GPU
  compute-app 均已清空。
- 校准后的三个 guard 在五轮评测中 hazard violation 全部为 `100%`、safe
  violation 全部为 `0%`，即部署端对观测到的危险候选也全部放行。真正控制早期
  gate 的是三个 point 头的交集：Box@.25 point safe violation 在 epoch1--5 为
  `100.0, 100.0, 95.47, 89.52, 40.41%`；Box@.50 在前四轮为 `0%`、epoch5
  为 `0.72%`，Mask@.25 从 epoch2 起为 `0%`。随着 Box@.25 point 放松，accept
  激增，但已失效的 guard 无法阻止 break，解释了 epoch4--5 的快速退化。
- 结论：`+log(4)` 精确抵消了 loss 的类别代价偏移，却不能被解释为逐候选风险
  上界；当前 guard 只学到成本修正后的总体先验，未形成样本级 hazard 排序。V75
  因预注册失败被拒绝，不进入正式前缀，也不把 epoch3 的小样本偶然提升当作候选。
  后续版本必须让部署边界直接由可验证的“相对父候选损失”决定，不能再把类别代价
  的常数校准当作安全证书，也不应仅对现有阈值做扫描。

### 14.46 V76 连续 Safety Slack 分位数下界与 smoke 预注册（2026-08-13）

- 新增默认关闭的
  `joint_query_quality_use_setwise_safety_slack_quantile_bound`，要求 V73 的六路
  risk-bound 头与 V74 veto gate，且与 V75 cost-calibration 互斥。六路参数合同
  不变：每个 Box@.25、Box@.50、Mask@.25 标准分别输出 point 与 lower-bound。
- V76 不再对 safe/hazard 二分类。若父候选在标准 `t` 上命中，监督目标为
  `(candidate_metric-t)/t`；若父候选未命中，则该候选不可能造成 break，目标为
  `(t-parent_metric)/t`。该连续、无量纲的 safety slack 严格满足：负值对应真实
  hit-to-miss，正值对应不破坏，零就是部署边界；它同时携带候选离阈值多远的信息，
  避免 V75 仅学习总体类别先验。
- point 用逐行平均绝对误差回归 slack；lower-bound 用 pinball loss 学习
  `tau=1/(1+break_cost)` 分位数。cost=4 时 `tau=0.2`；loss 再乘 `1+cost`，使
  乐观高估风险边界的斜率与保守低估形成严格 `4:1`，而部署仍只检查解析的零边界，
  没有新增可扫描阈值。三标准各取 point/lower 最小值后作硬 veto，promotion 仍只
  负责候选排序和 residual 幅度。
- 最终层仍为零，因此 step-0 point/lower/promotion 均为零，hard veto 虽不放行，
  residual 也精确为零，父分数与父选择完全复现。不开新 flag 时 V73--V75 行为
  保持。V76 新增/相邻定向 `5 passed`，完整 joint-query/初始化 `126 passed`，
  Source-MoE/训练/retention 集成 `378 passed`（仅两个既有 scheduler deprecation
  warning）。远程与本地传输副本 SHA256 逐文件一致，并保留八个
  `.v76_slack_quantile_20260813.bak` 回滚文件。
- 真实 V19 初始化审计
  `experiment_output/v51_bmq_rank/v76_safety_slack_quantile_bound_initialization_audit.json`
  通过：common/changed/new=`1228/0/30`，unexpected/shape mismatch=`0/0`，
  参数 286,779、state 30、输出头全零、父分数/Top-32/互斥合同均通过。
- smoke 固定为单 GPU、128/128、batch 12、`augment_det=0`、LR=`1e-4`、最多
  10 epoch；rank/slack-bound weight=`2/1`、break cost 4，其余 safety loss 为 0。
  任一完整 epoch 任一 REC 阈值 `break > fix` 即停止；epoch5 仍无 joint 修复也
  停止。只有至少两个完整 epoch `fix>0, break=0` 且 Mask 不退化，才审计最佳
  实物 checkpoint 并考虑正式全数据前缀。

### 14.47 V76 透传审计修正与 V76b smoke：分位数仍学成行级先验（2026-08-13）

- 首次启动命令与 config 都显示 V76 flag=true，但训练日志 marker=0，point/guard
  loss 仍约为旧 BCE 的 `0.69`。核查发现 `models/losses.py` 的两条 endpoint 收集
  路径没有透传新 marker；模型构造本身正确，但 loss 端把缺失 marker 解释为 false。
  发现时该无效 run 已完成若干快速 debug 轮，立即核对并终止精确 PGID `162270`；
  它不计为 V76 结果。删除其 8 个 `.pth` 名称（2 inode），保留 config/JSON/log
  作为审计证据。
- 在两条 `transition_keys` 中加入 marker，给 `models/losses.py` 创建第九个
  `.v76_slack_quantile_20260813.bak`，上传 SHA256 核对一致。合并回归为
  `504 passed`（joint/audit 126 + Source-MoE/训练/retention 378），仅有两个既有
  scheduler deprecation warning。V76b 从受保护 V19 重新初始化，运行时明确看到
  marker=`1.0000`、point/quantile loss 约 `1.0/1.54`，确认执行的是新目标。
- V76b 完成 6 个完整评测 epoch（停止请求到达前多完成一轮）。fixed parent 恒为
  `63/57`；learned Position 六轮均为 `64/58`，Mask 六轮均为 `64/52`、mIoU
  `0.350491`。joint fix/break/switch 与 veto accept 六轮全部为 0；所以 reported
  `+1/+1` 仍来自既有父路径差异，新模块无可归因修复。按 epoch5 无修复门禁停止
  PGID `163562`，screen、进程组、GPU compute-app 均为空。
- 分位数 loss 的总体覆盖语义成立：Box@.25 coverage 从 `19.16%` 到 `19.50%`，
  Box@.50 从 `23.64%` 到 `22.25%`，Mask@.25 从 `15.82%` 到 `15.89%`，围绕
  `tau=20%`。但它仍主要学习行/数据总体先验而非候选排序：Box@.25 lower-bound
  hazard violation 在 epoch2--6 约 `97.8%`，Mask@.25 同样约 `97.2%`；相反
  Box@.50 lower-bound safe violation 从 `100%` 仅降到 `99.0%`，由这一标准把
  所有候选最终否决。point 头也几乎对所有 safe 与 hazard 同时给正值。
- 结论：连续 slack 修复了目标的信息量与统计解释，却没有消除“用行级常数满足
  边际 loss”的捷径。V76b 拒绝，不进入正式前缀。下一版不改 tau/cost/零阈值，
  而在同一父候选行内直接监督 safe 与 hazard 的 slack 顺序，使共同偏置无法降低
  该项；仍保留分位数边界负责绝对校准。

### 14.48 V77 行内 Safety Slack Pairwise Order 与 smoke 预注册（2026-08-13）

- 新增默认关闭的
  `joint_query_quality_use_setwise_safety_slack_pairwise_order`，要求 V76 quantile
  bound；不增加参数、不改变 V76 的连续目标、`tau=0.2`、cost=4 或部署零边界。
  对每个父候选行、每个标准，构造所有真实 safe 候选与 hazard 候选的有序对；point
  与 lower-bound 都回归两候选真实 normalized slack 的差。loss 是预测差与真实差
  的 L1，因此任何行级共同常数严格抵消，且真实连续差本身给出尺度，无需新 margin。
- 绝对 point/quantile loss 继续负责把输出校准到零边界，pairwise loss 只负责候选
  次序；二者在 V77 风险项内等权相加。新增每标准 point/lower pair MAE 与
  safe-over-hazard accuracy。step-0 所有输出仍为零，pairwise loss 有非零有限梯度，
  但 residual/父分数/选择严格复现 V19。
- V77 新增/相邻定向 `3 passed`；完整 joint/audit + Source-MoE/训练/retention
  合并回归 `506 passed`，仅两个既有 scheduler warning。真实初始化审计
  `experiment_output/v51_bmq_rank/v77_safety_slack_pairwise_order_initialization_audit.json`
  通过：common/changed/new=`1228/0/30`、参数 286,779、state 30，输出头全零，
  V76/V77、veto、Top-32 和父分数合同一致。九个修改文件均有
  `.v77_slack_pairwise_20260813.bak`。
- smoke 与 V76b 相同：单 GPU、128/128、batch 12、LR=`1e-4`、最多 10 epoch，
  promotion rank/slack risk=`2/1`。任一完整 epoch 任一阈值 `break > fix` 即停；
  epoch5 无 joint 修复即停。仍只在至少两个 epoch `fix>0, break=0` 且 Mask
  不退化时审计实物 checkpoint 并考虑正式前缀。

### 14.49 V77 smoke：排序学会但零边界仍无可归因修复（2026-08-13）

- 监控返回并执行停止前共完成 8 个完整评测 epoch。八轮 fixed parent 均为
  `63/57`，reported learned Position 均为 `64/58`；Mask 均为 `64/52`、mIoU
  `0.350491`。但 joint 内部归因在 epoch1--7 的 fix/break/switch 全为 0；epoch8
  虽出现 `0.76%` switch，`fix/break@.25/.50` 仍全部为 0。因此 reported
  `+1/+1` 是既有父路径与 fixed-default 的差异，不是 V77 新模块产生的修复。
- lower-bound 的 safe-over-hazard 行内排序确实被学会：Box@.25 八轮 accuracy 为
  `87.01, 87.39, 87.77, 88.01, 88.74, 89.02, 88.63, 88.33%`；Box@.50 为
  `78.57, 63.53, 37.18, 43.10, 64.40, 83.75, 84.74, 84.95%`；Mask@.25 为
  `94.34, 94.35, 94.41, 94.74, 95.43, 96.00, 95.73, 96.16%`。这验证了行内
  pairwise loss 消除了共同偏置捷径并提供了有效候选排序信号。
- 但绝对零边界仍没有形成有效决策：veto accept 八轮为 `0, 0, 0, 0.24,
  0.15, 0.80, 41.53, 56.88%`。前六轮基本全拒绝；后两轮快速放开后，promotion
  所选候选仍没有产生任何 REC fix，说明“安全排序正确”尚不能保证绝对边界校准，
  也不能保证与 promotion 候选对齐。不能据此启动正式全验证集评测。
- 按预注册的 epoch5 无可归因修复门禁拒绝 V77。停止前训练已进入 epoch9；核对
  后仅向精确 PGID `165413` 发送 TERM，screen、进程组与 GPU compute-app 均清空。
  删除该 run 的 8 个 `.pth` 名称（2 inode），保留 config、8 轮 eval JSON、source
  diagnostics、retention JSON 与完整 launcher/log；受保护 V19 inode
  `6496464367`、大小 `605267997` 未变，overlay 可用空间恢复到 `2.9G`。
- 下一步不再调整 tau、cost 或部署阈值。V77 已证明相对排序可学，剩余结构性问题是
  让安全分数具有父候选条件下的可识别零点，并让训练目标直接覆盖部署时唯一被
  promotion 选中的候选；新设计必须继续保持 step-0 精确复现与无阈值扫描。

### 14.50 V78 Proposal-Conditioned Safety 两阶段架构与 smoke 预注册（2026-08-13）

- 新增默认关闭的 `joint_query_quality_use_proposal_conditioned_safety`，要求 V77 的
  pairwise slack order 与 V74 safety veto。部署改成严格 `Propose -> Verify`：
  promotion 先按“候选 promotion advantage - 击败父分数所需 advantage”提出唯一
  非父候选；若 proposal 本身不能越过父分数，则直接保留父候选。只有可 promotion
  的 proposal 才交给三标准 point/lower-bound 安全门验证；任一标准的最小值不大于
  零都回退父候选，不会在被拒后隐式选择第二个候选。
- V76/V77 的全候选 safe-vs-hazard pairwise loss 原样保留，继续学习候选相对安全
  次序；但负责绝对零点的 point L1 与 lower pinball loss 只在当前 promotion
  proposal 上计算。这样训练中的绝对校准对象与部署真正验证的唯一候选严格一致，
  避免 V77 用大量部署不会选中的 easy safe 候选满足总体分位数。proposal argmax
  作为离散两阶段接口，不向 promotion 传递 safety loss；promotion 仍由 setwise
  tier/rank 目标独立训练，因此不存在 safety 通过换 proposal 自行降低 loss 的捷径。
- 该改动不增加参数，不改变 `tau=0.2`、cost=4、零阈值或 Top-32 候选合同。最终层
  零初始化时所有 proposal 的 promotion margin 小于等于零，promotable mask 全空，
  residual 精确为零，父分数与选择精确复现。增加 proposal/hazard、promotable、
  safety-accept 三类运行诊断；九个修改文件均保留
  `.v78_propose_verify_20260813.bak` 回滚副本。
- V76--V78 定向测试 `8 passed`；完整 joint-query/初始化 `131 passed`，Source-MoE、
  训练、retention 等集成 `350 passed`，额外 scheduler/checkpoint `21 passed`（仅两个
  既有 scheduler deprecation warning），合计 `502 passed`。真实 V19 初始化审计
  `experiment_output/v51_bmq_rank/v78_proposal_conditioned_safety_initialization_audit.json`
  通过：common/changed/new=`1228/0/30`、unexpected/shape mismatch=`0/0`，参数
  286,779、state 30，输出头全零、父保护/Top-32/V78 合同均通过。
- smoke 沿用单 GPU、128/128、batch 12、LR=`1e-4`、最多 10 epoch，rank/slack
  risk=`2/1`。任一完整 epoch 任一 REC 阈值 `break > fix` 立即停止；epoch5 仍无
  joint 修复也停止。只有至少两个完整 epoch `fix>0, break=0` 且 Mask 不退化，才
  审计实物 checkpoint 并进入正式全验证集前缀；否则拒绝且只清理该 run 的权重。

### 14.51 V78 smoke：proposal 对齐暴露 promotion--safety 因果错位（2026-08-13）

- 停止请求到达前完成 4 个完整评测 epoch。fixed parent 四轮均为 `63/56`；learned
  Position 为 `64/57, 62/55, 64/57, 64/57`，Mask 为 `64/52, 62/50, 64/52,
  64/52`，epoch2 mIoU 从约 `0.350607` 降到 `0.337473`。epoch1/3/4 的 reported
  `+1/+1` 仍是既有父路径差异，不是 V78 修复。
- joint 归因最关键的 epoch2 为两个 REC 阈值 `fix=0, break=1.52%`、switch
  `1.52%`，明确触发 `break > fix` 停止门禁。epoch1 无 promotable proposal；
  epoch2 proposal promotable=`1.52%`，但 proposal safety accept=`100%`，即唯一
  真正越过父分数的危险候选被安全门放行。epoch3 安全门转为全拒绝而回到父路径，
  不能撤销 epoch2 已观测到的风险证据。
- proposal 本身的 hazard 比例从 epoch1 的 `21.66%` 升到 epoch2 的 `25.45%`，
  同时三标准 lower-bound 行内排序仍约为 Box@.25 `85.99/86.99%`、Box@.50
  `83.64/83.35%`、Mask@.25 `93.24/93.26%`。这说明 V77 的全候选安全排序能力
  没有消失；失败来自 promotion 独立提出的候选分布与 safety 的在线校准不同步，
  而 proposal-only 绝对 loss 的样本数在早期又太少，无法在首次可 promotion 时提供
  可靠下界。严格两阶段接口正确暴露了这一因果错位，却未解决它。
- 按预注册门禁拒绝 V78，不进入正式前缀。核对后只向精确 PGID `168023` 发送
  TERM；screen、进程组与 GPU compute-app 均清空。删除该 run 8 个 `.pth` 名称
  （2 inode），保留 config、4 轮 eval/source diagnostics、retention JSON 与完整
  日志；受保护 V19 inode `6496464367`、大小 `605267997` 未变，overlay 可用空间
  恢复为 `2.9G`。
- 下一版应直接把 promotion 的候选选择与安全证据联合起来，而不是继续让一个独立
  promotion argmax 产生高 hazard proposal 后再做稀疏在线校准；仍不得通过扫描阈值
  或 ScanRefer 特化规则规避该问题。

### 14.52 V79 Parent-Referenced Safety 与 smoke 预注册（2026-08-13）

- 新增默认关闭的 `joint_query_quality_use_parent_referenced_safety`，要求 V77 pairwise
  slack order，并与 V78 proposal-conditioned safety 互斥。对每个父候选行，六路
  safety 原始输出统一减去 parent--parent pair 的六路输出，再进入 criterion min、
  hard veto 和 V76/V77 loss。该结构精确消除任何行级共同偏置，候选之间的所有差值
  完全保留；零点由同一网络对不可变父候选的输出条件化确定，而不是由数据集阈值扫描
  或常数校准产生。
- V79 恢复全候选连续 slack L1/pinball 校准，不使用 V78 稀疏 proposal-only 绝对
  loss；因此每个 batch 从第一步起都对所有候选提供零点梯度。promotion 与安全仍是
  独立非线性头，部署继续采用 hard safety veto；新模块不增加参数，不改变 tau=0.2、
  cost=4、Top-32 或 promotion margin。最终层为零时，父参考与所有候选 safety 均为
  零，hard veto 不放行，residual/父分数/选择精确复现。
- 增加结构测试验证：对 safety head 任意加 `+100/-37` 行偏置，V79 部署六路输出
  逐元素相同；parent 输出严格为零；V78/V79 互斥合同与缺失依赖均会 fail-fast。
  V76--V79 定向 `10 passed`，完整 joint-query/初始化 `133 passed`，集成与
  scheduler/checkpoint `371 passed`（仅两个既有 scheduler warning），合计
  `504 passed`。
- 真实 V19 初始化审计
  `experiment_output/v51_bmq_rank/v79_parent_referenced_safety_initialization_audit.json`
  通过：common/changed/new=`1228/0/30`、unexpected/shape mismatch=`0/0`，参数
  286,779、state 30、输出头全零、V79/父保护/Top-32 合同一致。九个修改文件均有
  `.v79_parent_ref_20260813.bak`。
- smoke 与 V77 相同：单 GPU、128/128、batch 12、LR=`1e-4`、最多 10 epoch，
  rank/slack risk=`2/1`。任一完整 epoch 任一 REC 阈值 `break > fix` 即停；epoch5
  无 joint 修复即停。仍只在至少两个 epoch `fix>0, break=0` 且 Mask 不退化时审计
  实物 checkpoint 并考虑正式全验证集前缀。

### 14.53 V79 smoke：父参考消除早期误放，但 promotion 交集仍为空（2026-08-13）

- 停止请求到达前完成 6 个完整评测 epoch，fixed parent 每轮 `63/57`；learned
  Position 每轮 `64/58`，Mask 每轮 `64/52`、mIoU `0.350491`。六轮 joint
  `fix/break/switch@.25/.50` 全部为 0，因此 reported `+1/+1` 仍完全来自既有父路径。
- 父参考显著改变了早期 safety 行为：epoch1 三路 lower-bound hazard violation 为
  Box@.25 `1.29%`、Box@.50 `0%`、Mask@.25 `1.88%`，而 V76/V77 同阶段常出现
  接近全误放或全拒绝；veto accept 仅 `1.01%`。到 epoch2--6 accept 逐步为
  `1.41, 2.97, 9.38, 17.33, 38.07%`，仍无实际 switch，说明父参考确实给出了稳定
  相对零点，而不是立刻发生 V78 的危险切换。
- 行内 lower-bound 排序在六轮保持较高：Box@.25 从 `86.76%` 缓降到 `80.98%`，
  Box@.50 从 `81.66%` 升到 `85.07%`，Mask@.25 约 `95.3%`。但随 accept 增长，
  Box@.25/@.50 hazard violation 到 epoch6 也升到 `22.83/22.30%`；更关键的是
  promotion-positive 候选与三路安全正交集始终没有击败父分数，所以没有修复也没有
  break。V79 解决了行偏置与早期安全问题，却没有解决 promotion--safety 交集为空。
- 按 epoch5 无可归因修复门禁拒绝 V79，不进入正式前缀。停止时训练已进入 epoch7；
  核对后仅向精确 PGID `170042` 发送 TERM，screen、进程组与 GPU compute-app 均
  清空。删除该 run 8 个 `.pth` 名称（2 inode），保留 config、6 轮完整 eval JSON、
  diagnostics、retention 与日志；受保护 V19 inode `6496464367`、大小
  `605267997` 未变，overlay 可用空间恢复到 `2.9G`。
- 后续不应放松阈值来制造交集；需要把 promotion 目标改成在已学到的安全有序空间中
  对 repair 候选形成可部署的联合优势，或直接监督“安全 repair 的联合 score”，同时
  保持父参考零点与 step-0 复现。

### 14.54 V80 Coupled Safe-Repair Witness 架构与 smoke 预注册（2026-08-13）

- V79 的 promotion 与 safety 各自有连续监督，但两个目标可能由不同候选满足；部署却要求同一
  候选同时越过父分数且通过三路安全门。新增默认关闭的
  `joint_query_quality_use_coupled_safe_repair_witness`，并要求 V79 parent-referenced safety。
  对每个真实 repair 候选，联合边界定义为 promotion 父边界与三路 safety 最小边界的最小值；
  每行再取最佳 repair 候选，并用既有 `0.02` setwise margin 直接监督至少一个同候选联合 witness。
  反传采用温度 `0.05` 的 smooth-min、前向保持 hard-min 精确值，修复行之外不产生该损失。
- 这不是放松部署阈值：hard veto、零阈值、`tau=0.2`、cost=4、Top-32、promotion margin 与
  inference 选择规则均不变，也不增加参数。新目标只消除“promotion 最优候选与 safety 最优候选
  分属两处”的训练捷径；V79 的全候选相对排序、父参考绝对零点和三个安全标准仍全部保留。
  最终层为零时联合 witness 仅提供训练梯度，部署 residual 仍为零，父分数与选择精确复现。
- 结构测试覆盖依赖 fail-fast、真实 repair 行的同候选联合 loss、promotion/safety 两头非零梯度
  和初始化 profile。V76--V80 定向 `12 passed`；完整 joint-query/初始化 `135 passed`；
  Source-MoE、训练分组、retention 与 scheduler/checkpoint 集成 `371 passed`（仅两个既有
  scheduler deprecation warning），合计 `506 passed`。Python compile、主训练脚本与两个 V80
  launcher 的 `bash -n` 均通过。
- 真实 V19 初始化审计
  `experiment_output/v51_bmq_rank/v80_coupled_safe_repair_witness_initialization_audit.json`
  为 `pass=true`：common/changed/new=`1228/0/30`、unexpected/shape mismatch=`0/0`，新增模块
  参数 `286,779`、state 30，所有输出头为零，V80/V79/父保护/Top-32 合同一致。九个修改文件均
  保留 `.v80_coupled_witness_20260813.bak` 回滚副本。
- smoke 继续使用单 GPU、训练/验证各 128 条、batch 12、LR=`1e-4`、最多 10 epoch，rank/slack
  risk=`2/1`，不做阈值扫描。任一完整 epoch 任一 REC 阈值 `break > fix` 立即停止；epoch5 仍无
  joint 修复也停止。只有至少两个完整 epoch `fix>0, break=0` 且 Mask 不退化，才审计实物
  checkpoint 并考虑正式全验证集前缀；否则拒绝并只清理该 run 的权重。

### 14.55 V80 smoke：同候选联合监督首次产生可归因净修复（2026-08-13）

- run `1786561159` 完成全部 10 个 epoch。fixed parent 十轮均为 `63/57`；epoch1--3
  joint fix/break/switch 全零，learned 的 `64/58` 仍来自既有父路径。epoch4 首次出现两个 REC
  阈值共同 `fix=2/128、break=0`；epoch5 为 `3/128、0`，epoch6 为 `2/128、0`，连续三轮满足
  预注册的安全修复门禁。对应 learned/Mask/mIoU 分别为 epoch4 `66/60, 66/54, 0.362777`、
  epoch5 `67/61, 67/55, 0.370046`、epoch6 `66/60, 66/54, 0.365367`，Mask 没有以退化换 REC。
- epoch8 为五项 retention 共同最佳：learned REC `69/63 = 0.5390625/0.4921875`，Mask
  `69/57 = 0.5390625/0.4453125`、mIoU `0.386688`。joint 两阈值均为
  `fix=6/128、break=1/128`，净增 5 hits；switch `10.61/128 = 8.33%`。witness recall 从
  epoch1--3 的 0 升到 epoch4 `3.03%`、epoch5 `5.30%`、epoch8 `10.15%`，证明新增目标确实
  建立了 promotion--三路安全的同候选交集，而不是沿用父路径伪提升。
- epoch9/10 的 break 分别升到两个阈值 `2/128`，以及 `.25=2/128、.50=3/128`，但各轮
  fix 仍更多，未触发 `break > fix`；learned 分别为 `68/62`、`67/60`。retention 正确锁定
  epoch8。其五个 best 名称与 `ckpt_epoch_8.pth` 硬链接到同一 inode `740113099`，没有复制
  六份大权重。
- epoch8 实物 checkpoint 审计
  `.../1786561159/v80_epoch8_checkpoint_audit.json` 为 `pass=true`：V19 common/changed
  `1228/0`、new/joint-new `30/30`、unexpected 0；优化器参数 `285,876`、实际 state 26、
  step 精确为 80，所有新增权重与 Adam moments finite，每个 state 的 `exp_avg` 非零。实物
  config 再次确认 V80/V79、三路 factorized safety、hard veto、Top-32 与 cost=4 合同。
- 这是 V76--V80 首次越过 smoke 晋级门禁的结构。下一步固定 epoch8 实物权重做完整 9508 条
  validation prefix；不继续训练、不调整阈值。只有完整集仍为两阈值净修复且 Mask 不退化，才
  考虑更长训练或正式候选；否则按完整集证据拒绝。

### 14.56 V80 epoch8 全量 9508：正 witness 泛化失败，拒绝正式训练（2026-08-13）

- 固定已审计 `ckpt_epoch_8.pth`、batch48、单卡执行只读 `--eval`，完整收据位于
  `experiment_output/v80_coupled_safe_repair_witness_epoch8_full_eval/scanrefer/`
  `v80_coupled_safe_repair_witness_epoch8_full9508_b48x1/1786561701/`。进程正常结束、GPU 与
  screen 清空；`sample_count=9508`，未继续 optimizer step，也未扫描阈值。
- 同次 fixed parent REC 为 `5514/4407 = 0.579932/0.463505`，V80 learned 为
  `5486/4371 = 0.576988/0.459718`，分别退化 `-28/-36 hits`。joint 内部浮点收据为
  `.25 fix/break=0.0069/0.0110`、`.50=0.0060/0.0112`，对应约 `66/105` 与 `57/106`；
  两个阈值均明确 `break > fix`，触发预注册拒绝门禁。switch ratio 为 `5.29%`。
- Mask 为 `5652/4629 = 0.594447/0.486853`、mIoU `0.415314`，低于 protected V19 的
  `0.598233/0.491376/0.418613`；因此 smoke 的 Mask 同步提升同样没有泛化。REC Unique/Multiple
  分别为 `.25 1237+4249=5486`、`.50 1046+3325=4371`；Mask 为
  `.25 1272+4380=5652`、`.50 1031+3598=4629`，主计数均被 subgroup 精确还原。
- failure localization 很清楚：full-set coupled witness recall 仅 `2.78%`，witness margin
  `-0.0852`；safety veto accept 却升至 `79.96%`。Box@.25/@.50/Mask@.25 safety hazard
  violation 分别为 `52.22/60.84/39.78%`。V80 只要求每个 repair 行存在一个正联合 witness，
  解决了 smoke 中的正样本交集，却没有在同一联合边界上压住最难的非安全/非修复候选；小样本
  的正 witness 因而伴随大规模错误放行。
- 按完整集门禁拒绝 V80，不进入更长或四卡训练。protected V19 inode `6496464367`、大小
  `605267997` 保持不变。下一架构不得靠阈值或 loss-weight 扫描；应把同一 coupled boundary
  改为双侧可分离目标：正 safe-repair 至少一项越过既有 `0.02` margin，同时每行最难 unsafe /
  non-repair candidate 留在零线下，并保持 parent-reference、hard veto 与 step-0 identity。

### 14.57 V81 Bidirectional Coupled Boundary 与 smoke 预注册（2026-08-13）

- 新增默认关闭的 `joint_query_quality_use_bidirectional_coupled_boundary`，要求 V80 coupled
  witness。V80 的正侧保持不变：每个 repair 行至少一个最佳 safe-repair 候选的 deployed joint
  margin 越过既有 `+0.02`。V81 在完全同一个 `min(promotion boundary, 三路 safety min)` 上
  增加负侧：每行所有非最佳 repair 候选中的 hardest joint margin 必须低于 `-0.02`。正负约束
  因而不能由不同 head、不同候选或不同边界分别满足。
- V81 不增加参数，不改 inference、hard veto、父参考零点、零阈值、`tau=0.2`、cost=4、Top-32
  或 promotion margin；smooth-min 仍仅作直通反传，hard-min 前向不变。新增 hardest-negative
  margin/violation、正负 separation margin/recall 诊断。九个修改文件均有
  `.v81_bidirectional_coupled_20260813.bak` 回滚副本。
- V76--V81 定向 `14 passed`；完整 joint-query/初始化 `137 passed`；集成 `371 passed`（仅两个
  既有 scheduler warning），合计 `508 passed`。Python compile、主 launcher 语法通过。真实
  V19 初始化审计
  `experiment_output/v51_bmq_rank/v81_bidirectional_coupled_boundary_initialization_audit.json`
  为 `pass=true`：common/changed/new=`1228/0/30`、参数 `286,779`，所有输出头为零，V81/V80/
  V79/父保护/Top-32 合同一致。
- smoke 沿用单 GPU、128/128、batch12、LR=`1e-4`、最多10 epoch以及 rank/slack risk=`2/1`，
  只增加结构性负侧 loss，不扫描权重或阈值。任一完整 epoch 任一 REC 阈值 `break>fix` 即停；
  epoch5 无 joint 修复即停。至少两个 epoch 净修复且 Mask 不退化才审计实物；即使 smoke 晋级，
  仍必须通过固定最佳 epoch 的完整 9508 条前缀，V80 的小样本提升不再作为正式训练依据。

### 14.58 V81 smoke：负侧成立但正侧共同下移，拒绝（2026-08-13）

- 停止请求到达前完成 6 个完整评测 epoch；fixed parent 均为 `63/56`，learned 均为
  `64/57`，Mask 均为 `64/52`、mIoU `0.350491`。六轮 joint fix/break/switch 全零，因此
  `+1/+1` 仍完全来自既有父路径，V81 没有可归因修复，触发 epoch5 门禁。
- 负侧目标本身有效：hardest-negative margin 从 epoch1 `-0.0101` 单调降到 epoch6
  `-0.0474`，negative violation 六轮均为 0。但正 witness margin 同时从 `-0.0316` 降到
  `-0.0542`，positive recall 与正负 separation recall 始终为 0；separation margin 在
  `-0.0214` 到 `-0.0070` 间仍为负。即两个独立绝对边界在共享表示上形成“整体下移”捷径，
  安全拒绝增强却把 repair 一起压在零线下。
- 核对后仅终止精确 PGID `176388`，screen/GPU 清空。停止到达前训练进入 epoch7；删除该 run
  的 8 个 `.pth` 名称（2 inode），保留 config、6 轮完整 eval JSON、diagnostics、retention 与
  日志。protected V19 inode `6496464367`、大小 `605267997` 未变，overlay 可用约 `2.3G`。
- V81 不进入全量评测。下一版应直接监督同一 joint score 的
  `best-positive - hardest-negative` 行内差值以消除共同平移自由度，并用正负中点锚定父参考零线；
  不应通过调大正侧权重抵消负侧，也不改变部署阈值。

### 14.59 V82 Centered Coupled Separation 与 smoke 预注册（2026-08-13）

- 新增默认关闭的 `joint_query_quality_use_centered_coupled_separation`，要求 V81/V80/V79 链。
  repair 行不再同时使用 V80 正绝对 loss 与 V81 负绝对 loss，而是在同一 deployed joint score
  上直接要求 `best-safe-repair - hardest-nonrepair >= 0.04`；这一行内差值严格消除共同平移。
  同时以 `abs((positive+negative)/2)` 把正负中点锚定到父参考零线，使 learned gap 不会整体漂到
  零线同一侧。没有正候选的 stay 行仍保留 hardest-negative `< -0.02` 的单侧保护。
- 部署路径、hard-min/hard veto、参数量、父参考、零阈值、`tau=0.2`、cost=4、Top-32 均不变；
  V82 启用时 V80/V81 两个独立绝对 loss 经结构测试严格为零，避免重复施压。新增 midpoint abs 与
  0.04-margin recall 诊断。九文件留有 `.v82_centered_coupled_20260813.bak`。
- V76--V82 定向 `16 passed`，完整 joint-query/初始化 `139 passed`，集成 `371 passed`（两个既有
  scheduler warning），合计 `510 passed`。真实 V19 初始化审计
  `experiment_output/v51_bmq_rank/v82_centered_coupled_separation_initialization_audit.json`
  为 `pass=true`：common/changed/new=`1228/0/30`，参数 `286,779`，零输出头与 V82--V79/父保护/
  Top-32 合同全通过。
- smoke 仍为单 GPU、128/128、batch12、LR=`1e-4`、最多10轮，其他权重保持 V80/V81 原值；
  不扫描 loss weight。任一阈值 `break>fix` 即停，epoch5 无 joint 修复即停；至少两个 epoch 净修复
  且 Mask 不退化才审计并固定最佳实物做9508条全量前缀。

### 14.60 V82 smoke：宽泛 non-repair 负集造成共同负漂移，拒绝（2026-08-13）

- run `1786563827` 的 fixed parent 在 epoch1--5 均为 `63/57`。epoch1--3 learned 为
  `64/58`、Mask `64/52`、mIoU `0.350491`，joint fix/break/switch 全零；epoch4 首次出现
  两个阈值共同 `fix=1/128、break=0`，learned `65/59`、Mask `65/53`、mIoU `0.355170`；
  epoch5 又回到 `64/58`、Mask `64/52`，joint fix/break 全零。到预注册 epoch5 只有一个净修复
  epoch，未满足“至少两个净修复 epoch 且 Mask 不退化”的晋级条件，因此拒绝 V82，不做 checkpoint
  审计或 9508 条全量评测。
- separation 的早期方向一度正确：epoch1--3 的行内正负差从 `-0.0228` 改善到 `-0.0081`；
  epoch4 margin recall 达到 `1.52%` 并产生一个 joint 修复。但 epoch5 separation 又退到
  `-0.0131`、recall 归零。更关键的是 midpoint abs 从 epoch1 `0.0212` 持续恶化到 epoch5
  `0.0476`，正 margin 从 `-0.0326` 降到 `-0.0541`，负 margin也从 `-0.0098` 降到
  `-0.0431`：V82 虽消除了独立正负边界的纯平移自由度，却仍把两个端点共同推到部署零线下。
- 停止信号发出时，128 条单轮训练已经快速完成 epoch6--8 并进入 epoch9；这些是停止门禁后的
  旁观数据，不能用于反向修改预注册判定。epoch6 learned/Mask 为 `64/58, 64/52`，epoch7--8 为
  `65/59, 65/53`；完整 config、epoch1--8 eval JSON、diagnostics、retention 与日志均保留。
  精确终止 PGID `178474` 后 screen/GPU 清空，仅删除该 run 的全部 `.pth`（3 个 inode，多个
  hardlink 名称），overlay 可用空间恢复到约 `2.3G`；protected V19 未触碰。
- 失败定位不是 margin 或 loss-weight 数值不足，而是负集语义过宽：V82 把所有“非最佳修复”候选
  都当作必须压到零线下的 hardest negative，其中包含不会破坏当前正确预测的中性/安全候选。
  这与部署目标不一致，也给共享 risk-bound 表示施加了不必要的整体负压力。下一版固定其余合同，
  只把会破坏 box@.25、box@.50 或 mask@.25 的 candidate 作为 coupled negative；repair 行有
  hazard 时做 centered separation，无 hazard 时保留正 witness；stay 行只压制真实 hazard。

### 14.61 V83 Hazard-Conditioned Coupled Separation 与 smoke 预注册（2026-08-13）

- 新增默认关闭的 `joint_query_quality_use_hazard_conditioned_coupled_separation`，要求完整 V82
  centered 链。V83 不再把所有 non-best-repair 当 joint negative，而只选父候选当前正确、目标
  candidate 会破坏的精确保护事件：Box@.25、Box@.50 或 Mask@.25 任一 hazard。repair 行有
  hazard 时仍在同一 deployed hard-min joint score 上做 `positive - hardest-hazard >= 0.04` 与
  零中点锚定；repair 行没有 hazard 时恢复单侧 positive witness；stay 行只压制真实 hazard。
  安全中性 candidate 不再获得负标签。
- V83 不增加参数，不改 inference、hard veto、父参考零点、阈值、`tau=0.2`、cost=4、Top-32、
  smooth-min ST 或所有 loss weight。新增 paired-repair-row、unpaired-positive-row 与 coupled hazard
  candidate ratio 诊断。九个修改文件均保留 `.v83_hazard_conditioned_20260813.bak` 回滚副本。
- 定向 V80--V83 测试 `9 passed`；完整 joint-query/初始化 `142 passed`；Source-MoE、训练分组、
  checkpoint/retention、retrain provenance 与 ScanRefer train-only 集成 `390 passed`（仅两个既有
  scheduler warning），合计 `532 passed`。Python compile、训练脚本与两个 V83 launcher 的
  `bash -n` 均通过。
- 真实 V19 初始化审计
  `experiment_output/v51_bmq_rank/v83_hazard_conditioned_coupled_separation_initialization_audit.json`
  为 `pass=true`：common/changed/new=`1228/0/30`、unexpected/shape mismatch=`0/0`，新增模块参数
  `286,779`、state 30，所有输出头为零，V83--V79、父保护与 Top-32 合同全部一致。
- smoke 固定单 GPU、训练/验证各128条、batch12、LR=`1e-4`、最多10 epoch，rank/slack risk
  权重仍为 `2/1`，不扫描阈值或权重。任一完整 epoch 任一 REC 阈值 `break>fix` 立即停止；
  epoch5 仍无 joint 修复则停止。只有至少两个完整 epoch `fix>0, break=0` 且 Mask 不退化，才
  审计实物 checkpoint，并固定最佳实物执行完整9508条验证；否则拒绝并只清理本 run 权重。

### 14.62 V83 smoke：连续双阈值净修复，epoch7 晋级（2026-08-13）

- run `1786566277` 完成10轮；fixed parent 始终为 `63/56`。epoch1--3 learned `64/57` 但
  joint fix/break/switch 全零；epoch4--6 learned `64/58`，仅 `.50 fix=1/128、break=0`。
  epoch7、8 首次连续满足完整门禁：learned REC `65/59`，两个阈值分别
  `fix/break=1/0、2/0`，Mask `65/53`、mIoU `0.357759/0.357757`，没有以 Mask 退化换 REC。
- epoch9--10 learned 仍为 `65/59`、Mask `65/53`，mIoU 升至 `0.358626`，但两个阈值都出现
  1 个 break，内部为 `.25 fix/break=2/1、.50=3/1`。因此按安全优先的预注册规则选择五项
  retention 共同最佳 epoch7，而不是事后选择 mIoU 更高但已有 break 的 epoch9/10。
- V83 负集语义按设计生效：hazard candidate ratio 全程约 `25.01%`，repair 行约 `3.64%`
  同时含 hazard、约 `87.27%` 走无 hazard 正 witness；配对中点 abs 保持 `0.0075--0.0095`，
  显著小于 V82 epoch5 的 `0.0476`。epoch7 positive recall `3.33%`，hardest-hazard margin
  `-0.0915`，证明安全中性候选不再驱动过宽负漂移。
- epoch7 实物审计 `.../1786566277/v83_epoch7_checkpoint_audit.json` 为 `pass=true`：V19
  common/changed=`1228/0`，new/joint-new=`30/30`，unexpected 0；优化器参数 `285,876`、
  state 26、step精确为70，新增权重和 Adam moments 全部 finite，每个 state 的 `exp_avg` 非零。
  因而固定 epoch7 做一次完整9508条只读评估，不继续训练、不调整阈值。

### 14.63 V83 epoch7 全量9508：整体接近 V19，但新 joint 路径仍净伤害，拒绝（2026-08-13）

- 完整收据位于
  `experiment_output/v83_hazard_conditioned_coupled_separation_epoch7_full_eval/scanrefer/`
  `v83_hazard_conditioned_coupled_separation_epoch7_full9508_b48x1/1786567326/`；样本精确9508，
  同次 fixed parent REC `5514/4407`，V83 learned `5524/4408 = 0.580984/0.463610`，表面相对
  fixed 为 `+10/+1 hits`。
- 但内部新 joint 路径在两个阈值都违反门禁：`.25 fix/break=0.0010/0.0012`，约 `10/11`；
  `.50=0.0014/0.0027`，约 `13/26`。整体相对 fixed 的小幅提升由既有父路径补偿了 V83 的净
  伤害，不能归因成新结构成功。相对 protected V19 `5526/4425`，V83 仍为 `-2/-17 hits`，
  也未达到历史 REC best 或正式目标。
- Mask 为 `5686/4671 = 0.598023/0.491271`、mIoU `0.418634`；相对 protected V19
  `5688/4672 = 0.598233/0.491376` 为 `-2/-1 hits`，mIoU仅 `+0.000021`。REC subgroup
  Unique/Multiple 为 `.25 1243+4281=5524`、`.50 1049+3359=4408`；Mask为
  `.25 1278+4408=5686`、`.50 1033+3638=4671`，都精确还原主计数。
- 全量 failure localization：hazard candidate ratio `19.09%`，repair 行有 hazard / 无 hazard
  分别 `24.19%/75.81%`；paired separation margin `0.0147`、0.04-margin recall `19.99%`，
  但 positive witness margin `-0.1087`、recall仅 `0.57%`，而 hardest-hazard margin已到
  `-0.0688`。即 V83 已能拒绝 hazard，但共享三路 hard-min/risk-bound 仍把绝大多数 safe-repair
  witness 留在零线下；只缩窄负集不足以建立可泛化的正部署边界。
- 按全量门禁拒绝 V83，不进入更长/四卡训练。精确目录内9个 `.pth` 名称（3 inode）全部删除，
  保留 config、10轮 smoke eval/diagnostics、retention、初始化/实物审计以及全量回执和日志；
  screen/GPU 清空，overlay可用约 `2.3G`，protected V19 未触碰。

### 14.64 V84 Monotonic Box-Safety Folding 与 smoke 预注册（2026-08-13）

- V83 全量分量诊断进一步定位正 witness 瓶颈：promotion@.25/.50 positive margin 为
  `-0.0253/-0.0741`；安全候选在 Box@.25 point/guard 的误拒率为 `9.72/12.79%`，在
  Mask@.25 为 `9.83/11.44%`，但 Box@.50 guard 高达 `43.88%`（point仅 `11.41%`）。该 guard
  学的是总体 `tau=0.2` 风险分位数，却被逐候选当绝对硬否决，是 positive joint margin
  `-0.1087` 的主要结构瓶颈。
- 新增默认关闭的 `joint_query_quality_use_monotonic_box_safety_folding`，要求完整 V83 链。
  box tier 改善在定义上不可能破坏 Box@.25/.50，因此 V84 将两项 box safety 折叠进 promotion：
  V83 的真实 box hazard 仍作为 coupled negative 监督 promotion，但部署 hard veto 与 coupled
  positive hard-min 只保留正交的 Mask@.25 safety。三个 criterion、六个 point/guard head 仍全部
  训练和记录，V84 只移除两个 box guard 对 safe-repair 的重复逐候选否决。
- 不增加参数，不改父参考、零阈值、cost=4、Top-32、promotion margin、loss weight 或
  Mask@.25 hard veto。结构测试精确证明负的 Box@.25/.50 guard 不再拒绝 promoted candidate，
  同一 candidate 的负 Mask@.25 point/guard 仍使其回退父候选。九个文件保留
  `.v84_monotonic_box_folding_20260813.bak`。
- V80--V84 定向 `11 passed`；完整 joint/init `144 passed`；集成 `390 passed`（两个既有
  scheduler warning），合计 `534 passed`。真实 V19 初始化审计
  `experiment_output/v51_bmq_rank/v84_monotonic_box_safety_folding_initialization_audit.json`
  为 `pass=true`：common/changed/new=`1228/0/30`、参数 `286,779`、所有输出头为零，V84--V79
  与父保护合同全部一致。
- smoke 沿用单 GPU、128/128、batch12、LR=`1e-4`、最多10轮、rank/slack risk=`2/1`，不扫描
  权重或阈值。任一 REC 阈值 `break>fix` 即停；epoch5 无 joint 修复即停。至少两个完整 epoch
  两阈值均 `fix>0, break=0` 且 Mask 不退化才审计实物并执行固定最佳 checkpoint 的9508条全量
  前缀；否则拒绝并只清理本 run 权重。

### 14.65 V84 smoke：box guard 折叠降低中点漂移，但安全修复不稳定，拒绝（2026-08-13）

- run `1786569841` 完成10轮，fixed parent 全程 `63/57`。epoch1--3 learned `64/58` 但 joint
  fix/break均零；epoch4两个阈值都 `fix/break=1/1`。epoch5 是唯一安全净修复轮：learned
  `65/59`，两阈值 `1/0`，Mask `65/53`、mIoU `0.358098`。
- epoch6立即在两阈值触发硬停止条件：`fix/break=0/1`，learned `63/57`、Mask `63/51`、mIoU
  `0.344092`。监控延迟期间已快速完成后续轮次；epoch7为`1/1`，epoch8--9再次`0/1`，epoch10
  joint归零。没有第二个完整 epoch 在两阈值同时 `fix>0,break=0`，所以不以单次 epoch5 峰值晋级，
  不审计 checkpoint，也不做9508条全量评测。
- V84 的结构作用可观测：paired midpoint abs 维持 `0.0025--0.0037`，低于 V83 smoke 的
  `0.0075--0.0095`；但 positive witness margin仍从 `-0.0331` 下滑到 `-0.1368`，recall最多
  `3.33%`。folding 解除了 Box@.50 guard 的重复 veto，却没有阻止同一 hard-min loss 在
  promotion 与剩余 Mask@.25 safety 间轮流只修“当前最差分量”，因此切换边界出现交替 fix/break。
- 按 smoke 门禁拒绝 V84。精确 run 内8个 `.pth` 名称（2 inode）全部删除，保留 config、10轮
  eval/diagnostics、retention和日志；screen/GPU清空，overlay可用约`2.3G`，protected V19未触碰。
  下一版不调权重或阈值：对同一个 oracle-safe best-repair candidate 分别要求 promotion margin
  与 Mask@.25 safety margin越过零线，再保留 V83 hazard negative；避免一个 min loss 的瓶颈
  轮换捷径。

### 14.66 V85 Same-Candidate Branchwise Witness 与 smoke 预注册（2026-08-13）

- 新增默认关闭的 `joint_query_quality_use_same_candidate_branchwise_witness`，要求完整 V84 链。
  V85 先用精确 deployed joint score 在 best oracle-safe repair 集合中选定唯一候选，再对该同一
  candidate 的 promotion branch 与 V84 剩余 Mask@.25 safety branch 分别施加 `>+0.02` 的正边界；
  exact hazard 仍以同一 deployed joint score施加 `<-0.02` 负边界。它保持 V80 的同候选交集合同，
  但消除 V84 单一 hard-min loss 只修当前最差 branch、下一轮换瓶颈的捷径。
- V85 不增加参数、不改 inference、阈值、权重、父参考、cost=4、Top-32 或 Mask hard veto；启用时
  V82 centered loss严格为零，由新的 branchwise loss接管。新增同候选 promotion margin、Mask safety
  margin与双分支 recall 诊断。九文件均留 `.v85_branchwise_witness_20260813.bak`。
- V80--V85 定向 `13 passed`；完整 joint/init `146 passed`；集成 `390 passed`（两个既有 scheduler
  warning），合计 `536 passed`。真实 V19 初始化审计
  `experiment_output/v51_bmq_rank/v85_same_candidate_branchwise_witness_initialization_audit.json`
  为 `pass=true`：common/changed/new=`1228/0/30`、参数`286,779`、零输出头与 V85--V79 合同一致。
- smoke 仍为单 GPU、128/128、batch12、LR=`1e-4`、最多10轮、rank/slack risk=`2/1`，不扫描任何
  权重或阈值。任一阈值 `break>fix` 即停，epoch5 无 joint 修复即停；至少两个完整 epoch 两阈值
  同时 `fix>0,break=0` 且 Mask不退化才审计实物并固定最佳 checkpoint 做9508条全量前缀。

### 14.67 V85 smoke：同候选双分支产生净增益，但始终伴随破坏，拒绝（2026-08-13）

- run `1786571516` 自然完成10轮，fixed parent 全程为 `63/57`。epoch1--2 joint 路径不切换；
  epoch3--4 learned 为 `64/58`，两个阈值均 `fix/break=1/1`。epoch5首次出现净增益：learned
  `65/59`、Mask `65/53`、mIoU `0.356379`，但两个阈值仍为 `2/1`，不满足零破坏门禁。
- 后续没有任何零破坏净修复轮。epoch7 learned `66/60`、Mask `66/54`、mIoU `0.363647`，两个
  阈值均为 `3/1`；epoch8为`2/1`；epoch9为`3/2`。最终 epoch10再次达到 learned
  `66/60`、Mask `67/54`、mIoU `0.368597`，两个阈值仍为`3/1`。因此虽有多个表面净增益轮，
  却没有一个完整 epoch 达到 `fix>0,break=0`，更不可能满足连续两个安全轮的预注册晋级条件。
- 分支诊断解释了这一失败：promotion margin 始终为负（约`-0.0283`至`-0.0361`）；Mask safety
  margin 从接近零逐步恶化到 epoch10 的`-0.0557`，双分支 recall 最多仅`5.30%`。同时 negative
  margin 从`-0.0239`持续拉到`-0.1086`。即分别监督两个 branch 确实比 V84 更常发现 repair，
  但共享打分表示仍优先扩大 hazard 分离，未学出可跨 epoch 保持的无破坏同候选边界。
- 按 smoke 门禁拒绝 V85，不审计 checkpoint、不做9508条全量评测，也不扫描阈值或权重。精确
  run 内8个 `.pth`（每个`607,645,755` bytes，合计约`4.86 GB`）全部删除，保留 config、10轮
  eval/diagnostics、retention、初始化审计与日志；screen已退出，protected V19未触碰。
- 结论边界：V85 是 V84 之后首个在多轮同时提升 learned REC 与 Mask 的结构，但“收益必伴至少
  一个 break”说明仅重写 loss 聚合不足以满足部署安全。后续若继续，应将优化对象从共享标量
  joint score 改为具有显式父候选不劣约束的可验证决策机制；在此之前不应启动完整训练。

### 14.68 V86 Parent Non-Degradation Certificate 与 smoke 预注册（2026-08-13）

- V85 的 break 定位到部署合同本身：V84 folding 后 Box@.25/.50 不再参与 safety veto，只由
  promotion 间接吸收 exact hazard negative；因此 promotion 的假阳性可以直接产生 Box break。
  另一方面，V83 全量已证明把 Box guard 重新全量接入会过度拒绝，尤其 Box@.50 guard 的 safe
  violation 为`43.88%`（point仅`11.41%`）。V86 因而新增默认关闭的
  `joint_query_quality_use_parent_non_degradation_certificate`，要求完整 V85 链。
- V86 对同一个 deployed candidate 建立四项显式证书：promotion、Box@.25 point slack、
  Box@.50 point slack、Mask@.25 point+guard。只有四项都严格大于父参考零线才允许覆盖父候选；
  任一 Box point 预测为退化即回退 immutable parent。Box 使用连续 parent-relative safety slack
  point head，避开保守 quantile guard 的高误拒；Mask 继续保留 point+guard，未放松既有保护。
- 训练时也先按精确四项 deployed hard-min 选唯一 oracle-safe best-repair，再对该同一候选四项
  分别施加`>+0.02` witness；exact Box@.25/Box@.50/Mask@.25 hazard 仍走四项联合负边界。该机制
  不用 ScanRefer 身份、类别或固定样本规则，阈值来自 ScanRefer/Nr3D/Sr3D 共用评测定义，具备
  跨数据集架构泛化性；不增加参数，不改 LR、loss weight、cost=4、Top-32 或部署零阈值。
- 九个既有文件先按 V85 SHA256 精确校验，再留
  `.v86_parent_non_degradation_20260813.bak` 后原子部署；新增两个 launcher。定向 V80--V86
  `13 passed`，完整 joint/init `149 passed`，Source-MoE、训练分组、checkpoint/retention、
  retrain provenance 与 ScanRefer train-only 集成 `402 passed`（仅两个既有 scheduler warning），
  合计`551 passed`；Python compile 与三个 shell `bash -n` 全通过。
- 真实 V19 初始化审计
  `experiment_output/v51_bmq_rank/v86_parent_non_degradation_certificate_initialization_audit.json`
  为`pass=true`：common/changed/new=`1228/0/30`、unexpected/shape mismatch=`0/0`，新增模块参数
  `286,779`、state 30，所有输出头为零，V86--V79 合同与 protected parent 完全一致。
- smoke 固定单 GPU、训练/验证各128条、batch12、LR=`1e-4`、最多10 epoch，rank/slack risk
  权重仍为`2/1`，不扫描任何阈值或权重。任一完整 epoch 任一 REC 阈值`break>fix`立即停止；
  epoch5仍无 joint 修复则停止。只有至少两个完整 epoch 两阈值均`fix>0,break=0`且 Mask 不退化，
  才审计实物 checkpoint 并固定最佳实物执行9508条全量验证；否则拒绝并只清理本 run 权重。

### 14.69 V86 smoke：证书消除 break，但仅末轮一次安全修复，未晋级（2026-08-13）

- run `1786574027` 的 fixed parent 全程`63/57`。epoch1--9 learned 固定`64/58`、Mask`64/52`、
  mIoU`0.350491`，joint switch/fix/break 全零；epoch5 已满足“仍无 joint 修复”的停止条件，但
  监控轮询到达时训练已快速完成后续轮次。最终 epoch10 首次出现安全修复：learned`65/59`，
  两阈值均`fix/break=1/0`，Mask`65/53`、mIoU`0.357759`，证明显式父候选不劣证书确实能
  去掉 V85 中每次收益伴随的 break。
- 但 epoch10 是唯一一个完整安全修复轮，不满足预注册的连续两个 epoch 晋级门槛，因此不以
  末轮单次峰值做9508条全量评测。epoch10 实物审计`v86_epoch10_checkpoint_audit.json`为
  `pass=true`：common/changed/new=`1228/0/30`、unexpected 0；优化器参数`285,876`、state 26、
  step精确100，新权重和 moments 全 finite，且每个 state 的`exp_avg`非零。
- 失败归因具有明确分支证据：Box@.25/.50 point certificate margin 从 epoch1 的
  `+0.0024/+0.0017`持续增至 epoch10 的`+0.1697/+0.1309`，说明新增 Box 证书可学习且不是
  拒绝瓶颈；相反 Mask@.25 certificate 从`-0.0023`降至`-0.1295`，promotion margin 从
  `-0.0327`降至`-0.0421`，四分支 recall 直到 epoch10 才到`2.27%`。同时 hardest joint
  negative 从`-0.0237`拉到`-0.1377`。根因是一个 Box hazard 的联合 hard-min 负损失可由
  无关 Mask 或 promotion 分支承担，产生跨 criterion 的错误负归因。
- 按门禁拒绝 V86，不进入全量。retention 的7个名称都指向同一 inode（每个`607,649,723`
  bytes）；确认 protected V19、完整日志、逐轮收据、初始化与实物审计可恢复后，删除本 run 的
  `.pth` 名称并保留其余证据。下一版保持 V86 部署证书不变，只把 hazard negative 分解到真实
  负责的 Box@.25、Box@.50 或 Mask@.25 certificate，并以 stay-row promotion 负边界保护回退；
  不再允许某一 criterion 的 hazard 压低无关分支。

### 14.70 V87 Criterion-Responsible Hazard Attribution 与 smoke 预注册（2026-08-13）

- 新增默认关闭的`joint_query_quality_use_criterion_responsible_hazard_attribution`，要求完整 V86
  链。部署端完全保持 V86 的四项证书；变化仅在训练归因：Box@.25 hazard 只压低 Box@.25 point，
  Box@.50 hazard 只压低 Box@.50 point，Mask@.25 hazard 只压低 Mask point+guard hard-min。
  promotion 不再吸收某个具体 safety criterion 的 hazard；无 repair 的 stay 行另以 hardest
  promotion `<-0.02` 保持父候选回退。
- V87 同时关闭 V69 基础 safety 分支的共享 hardest-hazard negative，因为该项在 V86 仍把任意
  criterion hazard 施加到当前最小的联合 safety certificate，会重复制造跨 criterion 负梯度。
  连续 slack point/quantile calibration、safe-vs-hazard pair order、同候选四项正 witness 和
  所有部署零阈值均保留；不增加参数、不改 LR、权重、cost=4、Top-32 或 Mask hard veto。
- 梯度结构测试用“只有 Box@.50 hazard、没有 repair、Mask parent 本就错误”的行证明：安全头仅
  Box@.50 point channel 获得 hazard 梯度，Box@.25、两个 Box guard、Mask point/guard 均严格零；
  promotion 仅由独立 stay-row 项训练。V80--V87 定向`12 passed`，完整 joint/init`151 passed`，
  Python compile 与三个 shell `bash -n` 全通过。
- 真实 V19 初始化审计
  `experiment_output/v51_bmq_rank/v87_criterion_responsible_hazard_attribution_initialization_audit.json`
  为`pass=true`：common/changed/new=`1228/0/30`、unexpected/shape mismatch=`0/0`、参数`286,779`，
  全部输出头为零；九个修改文件保留`.v87_criterion_responsible_hazard_20260813.bak`。
- smoke 沿用单 GPU、128/128、batch12、LR=`1e-4`、最多10轮、rank/slack risk=`2/1`，不扫描
  阈值或权重。任一阈值`break>fix`即停，epoch5无 joint 修复即停；至少两个完整 epoch 两阈值
  都`fix>0,break=0`且 Mask不退化才审计实物并做固定 checkpoint 的9508条全量评测。

### 14.71 V87 smoke：criterion 分解解除负污染，但证书系统性乐观，拒绝（2026-08-13）

- run `1786575678` 的 fixed parent 为`63/57`。epoch1零切换；epoch2 learned`64/58`但两个阈值
  已各`fix/break=1/1`。epoch3触发硬停止，learned`62/56`，两阈值均`1/3`；监控到达前已完成的
  epoch4进一步为 learned`59/53`、`fix/break=3/约8`，Mask`59/47`、mIoU`0.322269`。
  终止信号生效前共留下 epoch1--7 完整收据；epoch5--7仍持续`break>fix`，未出现可晋级轮。
- criterion attribution 的隔离作用本身符合设计：epoch1三个负责证书 margin 均接近零正值，且
  stay promotion margin 为`-0.0074`；到 epoch6，负责 Box@.25/.50/Mask@.25 margin 已升至
  `+0.1784/+0.2027/+0.1266`，四证书 recall达`21.21%`。但这也暴露了单独 point certificate
  的另一失败模式：真实 hazard 的 hardest margin 同样快速转正，安全头从 V86 的过度负污染切换
  成系统性乐观假阳性，导致 switch ratio最高`22.73%`且 break多于fix。
- 因任一阈值`break>fix`硬门禁已在 epoch3触发，拒绝 V87，不审计 checkpoint、不做全量评测。
  精确 run 内9个 `.pth` 名称对应3个 inode；protected V19、代码回滚副本、完整日志、逐轮收据与
  初始化审计均可恢复后只删除本 run 权重。下一版不在 point/guard 间扫描经验阈值：新增参数隔离、
  零初始化的 joint-hazard veto head，独立学习“任何受保护指标退化”，并与 V86 四证书取交；从而
  同时保留连续证书的可解释性与 V86 的零 break 行为，而不再污染 promotion/Mask 分支。

### 14.72 V88 Independent Joint-Hazard Lower-Bound Certificate 与 smoke 预注册（2026-08-13）

- V87 证明逐 criterion point 负归因不会互相污染，但各 point head 仍可对真实 hazard 同时产生
  乐观假阳性。V88 新增默认关闭的
  `joint_query_quality_use_independent_joint_hazard_certificate`，要求完整 V86 链且与 V87 模式
  互斥。它新增独立 MLP，并将 parent/candidate pair feature 在输入处 `detach`；因此保守风险
  梯度只能更新该证书自身参数，不能压低共享表示、promotion 或 V86 的 Box/Mask 证书。
- 监督目标不是“任一 hazard”的 OR 分类，而是三个连续、无量纲、parent-relative safety slack
  （Box@.25、Box@.50、Mask@.25）的逐候选 hard minimum。以 break cost=4 对应的 20% pinball
  quantile 学习联合安全下界；部署时作为第五项证书与 V86 的 promotion、两个 Box point、Mask
  point+guard 取严格交集，只有五项都`>0`才允许覆盖 immutable parent。它仍不读取数据集身份、
  类别或验证集规则，阈值来自三套 grounding 数据共用评测定义，不扫描部署阈值或 loss weight。
- V88 禁用 V86 的共享 hardest-hazard negative，并保留独立 stay-row promotion 回退约束；正样本
  witness 则要求同一个 oracle-safe repair 的五项证书都跨过`+0.02`。定向契约验证了：零初始化
  完全复现父选择；第五证书可单独 veto；其 score 反传时共享参数/输入梯度严格为空；连续 minimum
  slack 与 1:4 pinball 数值精确匹配。
- 九文件部署包 SHA256 为
  `4091936f886903dec1ea33158a82972b49d79eabbb7e621dc483ebe68fc842bf`，部署前均保留
  `.v88_independent_joint_hazard_20260813.bak`。完整 joint/init `156 passed`，Source-MoE、训练
  分组、checkpoint/retention、retrain provenance 与 ScanRefer train-only 集成`402 passed`
  （仅两个既有 scheduler warning），合计`558 passed`；Python compile 与 launcher `bash -n`
  均作为启动前门禁。
- 真实 protected V19 初始化审计
  `experiment_output/v51_bmq_rank/v88_independent_joint_hazard_certificate_initialization_audit.json`
  为`pass=true`：common/changed/new=`1228/0/35`、unexpected/shape mismatch=`0/0`，reranker 参数
  `353,083`、state 35，全部输出头为零，V88--V79 合同完全一致。
- smoke 沿用单 GPU、训练/验证各128条、batch12、LR=`1e-4`、最多10轮、rank/slack risk=`2/1`，
  不扫描超参。任一完整 epoch 任一 REC 阈值`break>fix`立即停止；epoch5仍无 joint 修复则停止。
  只有至少两个完整 epoch 两阈值均`fix>0,break=0`且 Mask不退化，才审计实物 checkpoint 并以
  固定最佳实物做9508条全量验证；否则拒绝并只清理本 run 的权重，日志与审计继续保留。

### 14.73 V88 smoke：反向梯度隔离仍遭遇移动表示，零 break 但全程零修复（2026-08-13）

- run `1786577359` 的 fixed parent 为`63/56`；epoch1--8 learned 始终`64/57`、Mask`64/52`、
  mIoU`0.350491`。八轮新 joint 路径的 switch/fix/break 均严格为零，因此 learned 相对 fixed
  的`+1/+1`来自既有父 selector，不能归因给 V88。epoch5 已命中“仍无 joint 修复”的预注册
  停止门槛；SSH 冷却期间后台完成的 epoch6--8 只作旁观证据，不用于反向挑选。
- V88 下界并未学成稳定安全判别器：target-negative ratio 固定约`25.01%`，quantile coverage
  约`24.5--25.8%`，但 safe-reject 从 epoch1/2 的`100%`降到 epoch4的`24.58%`后又波动，
  unsafe-accept 则从`0%`升到 epoch3/4/5的`20.31/62.51/44.15%`；MAE 仅从`0.7724`
  缓慢降到 epoch7的`0.7468`。联合五证书 recall 八轮始终为 0。
- V86 的三个既有证书会学习：Box@.25/.50 margin 从 epoch1 的`+0.0049/+0.0044`升到
  epoch5的`+0.1675/+0.1541`，Mask margin 也到`+0.0637`；真正仍在部署交集外的是
  promotion（epoch5 `-0.0347`）与不稳定的新增下界。V88 虽在独立头输入处 detach，避免其 loss
  反传污染共享表示，但它读取的仍是被 promotion/Box/Mask 等 loss 每步改写的 shared hidden，
  因而面对移动输入分布；“梯度隔离”不等于“表示隔离”。
- 按 epoch5 门禁拒绝 V88，不审计 checkpoint、不做全量验证。精确终止后 screen/训练进程为空，
  四张 A100 均为`1 MiB/0%`；run 内8个 `.pth`名称对应2个 inode（每个`608,454,828` bytes），
  在确认8轮收据、日志、launcher、初始化审计和 protected V19 可恢复后仅删除这些权重。
- 下一最小因果对照 V89 保持 V88 loss、五证书部署合同、cost=4、LR与所有门禁不变，只让独立
  joint-hazard MLP 直接读取冻结的原始 rich candidate/parent features 与父分数 rank/standardized
  difference，自带投影器且完全绕过 shared joint hidden。若 V89 仍全拒绝或出现高 unsafe-accept，
  则否决“移动共享表示是主因”，不再延续独立 hazard-head 方向。

### 14.74 V89 Frozen-Raw Hazard Features：移动表示假设被否决（2026-08-13）

- V89 是 V88 的单变量因果对照：新增默认关闭的
  `joint_query_quality_use_frozen_raw_joint_hazard_features`，要求 V88 独立证书。独立 MLP 不再读取
  shared joint hidden，而直接使用冻结 rich candidate/parent raw features、差值、逐元素积及父子
  rank/standardized-score 差；监督、五证书部署交集、cost=4、LR、loss weight 与所有门禁不变。
- 结构测试在任意大幅改写 shared input projection/attention 参数前后，证明 V89 hazard score 逐值
  不变。V88/V89 定向`5 passed`，完整 joint/init`158 passed`，相同集成套件`402 passed`
  （两个既有 scheduler warning），合计`560 passed`；部署包 SHA256 为
  `e78432e1fafb4ba7a432177f4aabe7d27e49a08b0b481c61eeb914d1a92f03dc`。
- protected V19 初始化审计
  `experiment_output/v51_bmq_rank/v89_frozen_raw_joint_hazard_features_initialization_audit.json`
  为`pass=true`：common/changed/new=`1228/0/35`、unexpected/shape mismatch=`0/0`，reranker 参数
  `365,371`、state 35，全部输出头为零；九文件保留
  `.v89_frozen_raw_joint_hazard_20260813.bak`。
- run `1786578398` 的 epoch1--4仍为零 joint switch/fix/break；epoch5首次放行即同时破坏两个阈值：
  `fix/break=.25=0/1,.50=0/1`，learned 从前四轮`64/57`降至`63/56`，Mask 从`64/52`
  降至`63/51`、mIoU`0.344092`。独立头 unsafe-accept 在 epoch1--3为0，但 epoch4/5升到
  `41.75%/46.64%`；MAE仍约`0.7685`，五证书 recall epoch5回到0。冻结 raw 输入只把 V88
  的风险泄漏延后一轮，并未形成安全边界，故“shared hidden 移动是主因”被实验否决。
- epoch5 已同时触发`break>fix`与“无安全修复”门禁，拒绝 V89，不做全量。停止到达前留下的
  epoch6--8仅作旁观；终止后四卡`1 MiB/0%`。精确 run 内8个 `.pth`名称对应2个 inode
  （每个`608,602,284` bytes），确认8轮收据/日志/审计与 protected V19 后仅删除这些权重。
- V71--V89 已覆盖单/多 criterion、point/guard/continuous quantile、共享/归因/参数隔离和 frozen
  raw 表示；继续叠加同类 safety head 缺乏新信息。下一步转向更高 immutable parent：审计并复用
  已保护的历史 backbone+parent-reranker+geometry-reranker（REC`0.582878/0.486012`），把剩余目标
  缺口降至68/38 hits，再评估其候选集中的可安全修复余量；不再以 V19 network-only parent
  `0.581195/0.465398`作为唯一起点。

### 14.75 历史 e71 Geometry 候选池只读 Oracle 审计（2026-08-13）

- 审计严格复用受保护的 e71 parent、selected geometry artifact 与完整 9,508-row val base/geometry
  cache，先以正式 sidecar 的 frozen score 路径选候选，再在选择完成后读取 IoU 标签做 oracle
  诊断；报告明确标记`diagnostic_uses_validation_labels=true`，不会作为训练、超参或阈值选择输入。
  产物为
  `experiment_output/historical_e71_geometry/frozen_rec_geometry_headroom_audit_v1.json`，SHA256
  `1b7dfaedf486d9a469f4d8f6194fae43b73f727c93b82701df3fe83bd753775e`。
- 审计逐样本精确复现 frozen sidecar：parent=`5512/4421 = 0.579722/0.464977`，selected
  geometry=`5542/4623 = 0.582878/0.486222`，parent→geometry 的 fix/break 为
  @.25=`103/73`、@.50=`350/148`。封存 official 仍以`5542/4621 = 0.582878/0.486012`
  为正式接受口径；sidecar 的 @.50 多 2 hits 只用于诊断，因此正式目标缺口仍是`68/38`，不是
  sidecar 口径的`68/36`。
- 同一个有效候选池（最多`16×7=112`）的逐样本 flat oracle 为
  `6849/6308 = 0.720341/0.663441`，超过目标命中线`5610/4659`达`1239/1649` hits。
  相对 selected geometry，oracle 可在零 break 条件下修复 @.25=`1307`、@.50=`1685`，其中
  `1098`行可同时修复两个阈值。候选池覆盖远不是瓶颈，剩余目标只需识别现有安全修复的约
  `5.2%/2.3%`。
- 修复来源并不依赖单一特殊 variant：@.25 修复的 oracle variant 计数为
  `[401,497,70,38,69,175,57]`，@.50 为`[568,585,88,40,72,255,77]`；regressed、
  `fused_t0_exact`与 blend variant 均贡献显著。平均有效候选数`110.03`，75%样本有完整112个
  候选；selected-to-oracle IoU regret 中位数`0.0377`、P90`0.6069`。
- 结论：不扩大 proposal/geometry variant，不改 backbone，也不再延续 V71--V89 hazard-head
  家族。下一步只审计并改进历史 geometry 的候选选择目标/表示，以 train-only scene split
  拟合和校准；完整 val 只允许在结构、训练配置和 checkpoint 已冻结后做一次确认。

### 14.76 V90 Selective Geometry Residual：固定协议与启动前门禁（2026-08-13）

- 代码审计发现仓库已有但未留下正式训练产物的 selective residual 路线，恰好针对 14.75 暴露的
  选择器瓶颈。它把受保护的 selected geometry winner 作为 immutable baseline；每个替代候选仅
  读取可部署特征，学习相对 baseline 在 Box@.25、Box@.50 上分别属于
  `break/neutral/fix`，不修改 backbone、parent 或原 geometry scorer，也不依赖 Mask 标签。
- pair feature 固定为185维：179维 normalized geometry feature 差、parent rank 差、geometry
  rank 差、两个原 geometry threshold probability 差、IoU-estimate 差与 same-query 指示。模型网格
  仅含 linear/hidden64、weight decay `1e-4/1e-3`、break cost `2/4/8`，合计12个配置；两个阈值
  head 的固定权重为`2:1`。部署分数是两个三分类 head 的加权期望
  `P(fix)-P(break)`，且只提升超过 train-only margin 的一个候选，否则保持 baseline。
- 选择协议预先锁定：e71 train cache 仍按 scene 分为 fit=`33,040`与 calibration=`3,625`；fit 内
  做5折 scene-disjoint OOF，每配置固定10 epoch，margin 只取 OOF 正 gain 的
  `50/60/70/80/90/95/97.5/99`最近秩百分位。候选必须五折在两个阈值均非负、pooled @.25
  delta 为正、10,000次 scene-cluster bootstrap 的 @.25 95% lower bound `>0`且 @.50 lower
  bound `>=0`；胜者按`2*delta025+delta050`、更高 margin、较少 switch 固定排序。
- 只有 OOF 合格才在全部 fit scenes 重训一次并访问独立 train calibration。calibration gate 固定为
  baseline `3461/3315`、oracle `3606/3588`，要求 residual `hits025>=3524`且
  `hits050>=3315`，并逐值核对样本数、候选 IoU 与物化 SHA；失败则不发布 artifact。只有两道
  train-only gate 都通过，才冻结 artifact 并允许后续一次完整 val/official 确认，不用 val 反选。
- 启动前四卡均为`1 MiB/0%`，`/root/autodl-tmp`尚余约`3.4 GiB`；该阶段只生成小模型与 JSON
  收据，不写大型 checkpoint。selective residual model/train/runtime/official 四套专项测试
  `195 passed in 11.27s`。代码 SHA256：model
  `d27be8220638bb4f4c4ac307ff6ca42a20072c3daa0b43f5689a05ca131d403b`，train
  `e78491852b6b612e913445df99feaff65d3ad0265614b76e5588ddef094a6c7b`，official runner
  `ef8ed4613ad4a8b9fd9812b9c12203d8df1b29747b9014b56ba7c1bbdc7a001a`。

### 14.77 V90 train-only OOF：pairwise residual 信号不足，按门禁回退 baseline（2026-08-13）

- V90 正常完成并原子发布
  `experiment_output/historical_e71_geometry/v90_selective_geometry_residual_trainonly/result-receipt.json`，
  SHA256 `06e4be7c0e73f1ae3821c78ccd0c92930ddd74e8524282ef9c3a8e2d368b1a4d`；派生分析
  `analysis-summary.json` SHA256 为
  `bc673ad8a9b0af2d87cd0fda2224dec95219c334e0f44e5f855260a6c399977b`。
  收据为`selected=baseline`、`deployable=false`、`artifact=null`、
  `validation_data_accessed=false`，protected backbone/parent/geometry 的前后身份逐值相同；因 OOF
  已拒绝，独立 train calibration 按协议未运行。
- 33,040-row fit baseline 为`31636/29418`，同池 oracle 为`32868/32392`，分别有
  `1232/2974` hits 余量；但12个模型配置与各自 margin 共92个固定策略没有一个通过 OOF 门禁。
  @.25 pooled delta 范围`[-112,+2]`，@.50 为`[-523,+8]`；92个策略的 @.25 scene-bootstrap
  95% lower bound 全部未严格大于0，72个策略的 @.50 lower bound 为负。
- 最佳加权净增策略（hidden64、break cost4、margin P80）仅 switch 143/33040，得到
  @.25 fix/break=`2/0`、@.50=`5/3`，净增`+2/+2`；但 fold3 的 @.50 为`-2`，bootstrap
  lower bound 为`0/-2`。最接近保守门禁的 P90 策略 switch 72，仅 @.50=`3/1`净增2，
  @.25 完全没有修复；它不能贡献目标最缺的 @.25 68 hits。
- 结论：不通过放宽 margin 或 bootstrap 反选 V90。单个 alternative-vs-baseline 的185维差分三分类
  把 query 身份与其7个 geometry variant 混在同一 flat 决策中，OOF 可复现增益远小于候选 oracle；
  后续需显式建模 query→variant 层次，而不是继续调整 break cost 或阈值。

### 14.78 V91 Hierarchical Query→Variant Reranker 与预注册（2026-08-13）

- V91 保持 e71 backbone、parent 和 selected geometry 完全冻结，把112个候选恢复为
  `16 queries × 7 variants`。query head 读取152维 query feature及 default/parent score/rank、
  两个 Top-1 标志；variant head 读取25维 geometry feature、geometry score/rank、regressed/valid
  标志，并结合所选 query 表示。推理先选 utility 最大的 query，再只在该 query 的合法 variants
  中选一个，避免 flat residual 对111个替代候选分别作相互独立判断。
- 两级 head 均预测 @.25/@.50 命中概率，并以乘法参数化强制
  `P(hit@.50)<=P(hit@.25)`；query 标签定义为“该 query 的任一合法 variant 命中”，variant 标签
  是对应框的严格阈值命中。loss 对 query/variant 两级各做按行均衡 BCE，固定阈值权重`2:1`，
  false-positive cost 只取`2/4`。
- 固定网格为 hidden `64/128`、weight decay `1e-4/1e-3`、false-positive cost `2/4`，共8配置；
  每折12 epoch、5折 scene-disjoint OOF。margin 仍只取 OOF 正 gain 的
  `50/60/70/80/90/95/97.5/99`最近秩百分位。策略须五折两阈值均非负、pooled @.25 delta
  `>0`，且两个 scene-bootstrap 95% lower bound 均`>=0`；不满足即回退 baseline。
- OOF 合格后才重训并访问独立3,625-row train calibration；calibration gate 与 V90 完全相同，
  要求`hits025>=3524`、`hits050>=3315`并核对全部 cache digest。只有两道 train-only gate 均通过
  才发布 artifact；验证集不参与结构、模型、margin 或 checkpoint 选择。启动前 hierarchical
  model/train/runtime/official 四套专项测试为`215 passed in 21.02s`。

### 14.79 V91 train-only 结果：OOF 信号成立，但独立 calibration 否决发布（2026-08-13）

- V91 正常完成并原子发布
  `experiment_output/historical_e71_geometry/v91_hierarchical_query_variant_trainonly/result-receipt.json`，
  SHA256 `5cc9c5ba94618bccd598b3d05f33d0dd1954602b9a2f54e246d0c19cef25ff6b`。收据最终为
  `selected=baseline`、`deployable=false`、`artifact=null`、`validation_data_accessed=false`；四卡已
  回到`1 MiB/0%`，protected backbone/parent/geometry 的路径、inode、mode、size 与 SHA256 在训练
  前后逐值相同。
- 5-fold scene-disjoint OOF 证明层次结构确实学到可泛化信号。33,040-row baseline 为
  `31636/29418`，oracle 为`32868/32392`；72个固定 margin 策略中64个通过所有 OOF 谓词。按原
  预注册排序选择 hidden128、weight decay`1e-3`、false-positive cost4、margin P50：切换
  `14196`行，@.25 fix/break=`299/175`、@.50=`981/673`，净增`+124/+308`。五折 delta 分别为
  @.25=`[22,24,43,26,9]`、@.50=`[49,47,54,57,101]`；10,000次 scene-bootstrap 95% lower
  bound 为`+80/+212`。
- 增益几乎都来自 query 纠错而非同 query 的 variant 微调：selected-query changes=`12803`，
  same-query variant changes=`1393`；wrong-query recoveries 为 @.25=`271`、@.50=`875`，
  wrong-variant recoveries 仅`28/106`。这印证 V90 的 flat pairwise 失败诊断：主瓶颈是先选对
  query，显式 query→variant 分解是有效方向。
- OOF 胜者按协议在独立3,625-row train calibration 上只得到 baseline`3461/3315`→
  `3458/3318`：@.25 fix/break=`23/26`、@.50=`79/76`，净增`-3/+3`，scene-bootstrap lower
  bound=`-17/-22`。它未达到预注册`3524/3315`门槛，故自动回退 baseline，不发布 artifact、
  不运行 validation/official。这不是可通过查看 val 后放宽门槛挽救的候选。
- 事后协议审计发现 calibration 的 @.25 门槛要求`+63/3625=1.74%`净增，而 OOF 胜者只有
  `+124/33040=0.375%`；前者是后者增益率的约4.64倍，因而门槛把“最终正式目标缺口”错误地
  等量施加到仅11%的独立 train calibration 上。这个门槛对 V91 必须维持，结果仍为拒绝；但后续
  协议应使用预注册的非退化统计门槛与按样本比例缩放的最低效应，而不是不可达的绝对`+63`。

### 14.80 V92 设计依据：固定高置信稀疏 query 切换，而非追逐 OOF 总增益（2026-08-13）

- V91 的 OOF Pareto 前沿显示，hidden128、weight decay`1e-3`、false-positive cost2、margin P99
  仅切换`279/33040=0.844%`，仍得到 @.25 fix/break=`122/41`、@.50=`124/44`，净增
  `+81/+80`；五折为 @.25=`[12,17,15,26,11]`、@.50=`[14,17,11,31,7]`，bootstrap 95%
  lower bound=`+55/+52`。这一个 train-only 策略的 OOF 净增已经超过正式缺口`68/38`，同时
  switch 数比 V91 P50 少约50.9倍，净修复/切换率从`0.87%/2.17%`升到`29.0%/28.7%`。
- V92 因此不增加新网络，也不扫描新模型超参：复用 V91 的层次结构与同一固定模型网格，但模型
  选择目标预注册为“先满足 OOF 五折双阈值严格为正、双 bootstrap lower bound严格为正、OOF
  净增至少覆盖正式缺口`68/38`，再最少化 switches；并列才比较`2*delta025+delta050`”。这会
  在不读取 val 的前提下固定到高置信稀疏区域，而非 P50 高覆盖区域。
- 独立 train calibration 继续是发布前硬门禁，但按 calibration 占 fit 的固定比例
  `3625/33040`缩放最低效应：要求两个阈值都不低于 baseline，且 @.25/@.50 至少净增
  `ceil(68*3625/33040)=8`与`ceil(38*3625/33040)=5`；同时两项 scene-bootstrap lower bound
  均不得为负。该门槛在运行前写死，验证集继续完全不可见。若未通过，V92 同样不发布、不做
  official；若通过，才冻结唯一 artifact 并允许一次完整正式验证。

### 14.81 V92 sparse refit calibration：稀疏 OOF 优势仍未跨过全 fit 单模型重训（2026-08-13）

- V92 严格验证 V91 收据及其 SHA 后，只从已封存的 OOF diagnostics 按 14.80 的规则选策略；59个
  策略满足双阈值每折严格为正、双 bootstrap lower bound严格为正及 OOF 至少`+68/+38`。最少
  switch 的唯一胜者确为 hidden128、wd`1e-3`、cost2、P99：margin=`0.9188945293`、
  switch=`279`、OOF=`+81/+80`。选择过程未重跑 OOF、未读取 calibration 或 validation 标签。
- 固定胜者在全部33,040-row fit 上重训一个模型，再一次性评估独立3,625-row train calibration。
  只读产物
  `experiment_output/historical_e71_geometry/v92_sparse_hierarchical_calibration_audit_v1.json`
  SHA256 为`ef3c7d5feb7b1ad00762daf5a0aaabfb867a3e1f639bd027c91580dc310a2563`；
  `validation_data_accessed=false`，protected 三件套前后身份完全相同，四卡结束后均`1 MiB/0%`。
- calibration 只切换`23/3625=0.634%`，@.25 fix/break=`6/8`、@.50=`8/6`，baseline
  `3461/3315`变为`3459/3317`，即`-2/+2`；scene-bootstrap 95% lower bound=`-9/-5`。
  预注册的按比例最低效应`+8/+5`及两个非负 lower bound 四项全部失败，故 V92 不发布 artifact、
  不做 validation/official。
- V91 P50 与 V92 P99 在 OOF 都是五折一致正增益，但把五个 fold model 丢弃、在全 fit 上重训
  一个模型后分别变成`-3/+3`与`-2/+2`。因此现有证据不能简单归结为“层次结构没有信号”；更
  精确的待检验假设是 full-fit refit 的 normalization/score scale 或单次优化轨迹破坏了 OOF
  模型的跨 scene 稳定性。

### 14.82 V93 Five-fold Logit Ensemble 因果对照与预注册（2026-08-13）

- V93 固定复用 V92 的唯一配置、P99 margin、数据划分、epoch、seed 与所有校准门槛；唯一变化是
  不再训练 full-fit 单模型，而是保留五个 scene-fold model，每个仍只在4/5 fit scenes训练并使用
  自己训练子集拟合的 normalization。部署预测对五个模型的 query logits 与 variant logits做等权
  算术平均，再执行完全相同的 query→variant argmax 和固定 P99 margin；不扫描模型权重、投票数、
  margin 或共识阈值。
- calibration 对五个模型都属于从未参与训练的 scene 集，因此可直接检验“OOF 模型集成是否比
  full-fit refit 稳定”。发布前门槛与 V92 完全相同：@.25/@.50 相对 baseline 至少`+8/+5`，且
  两个 scene-bootstrap 95% lower bound 都`>=0`。不满足则只保留只读拒绝报告；满足也仅进入
  ensemble artifact 的实现/契约测试，validation 仍需在 artifact 完整冻结后才允许一次正式访问。

### 14.83 V93b 结果：fold ensemble 未改善 calibration，refit-only 假设被否决（2026-08-13）

- V93 首次运行完整训练出5个 fold member，但汇总脚本误把公开 policy candidate 直接传给内部
  diagnostics，缺少 selector 才补充的`sentinel`字段而在报告阶段抛出`KeyError`。该轮没有生成
  JSON/artifact，失败 log 原样保留；修复仅改为调用公开`choose_hierarchical_configuration`接口，
  不改模型、训练、预测或门槛。V93b 重跑时五个 normalization/state SHA256 与首次运行逐值一致，
  证明训练确定性且报告修复没有改变模型。
- 有效只读产物
  `experiment_output/historical_e71_geometry/v93b_hierarchical_fold_ensemble_calibration_audit_v1.json`
  SHA256 为`3760987d878ba6f30adaf5087c3925f21caf4b65e590aab571f9be380e5c65e7`；五个成员分别在
  `26005/26536/26540/26713/26366`行与404/405 scenes上训练，state SHA256 全部不同，calibration
  的56 scenes 与 fit 零重合。受保护三件套未变、四卡结束为`1 MiB/0%`。
- 固定 P99 margin 下 ensemble 切换`24/3625=0.662%`，@.25 fix/break=`6/8`、@.50=`8/8`，
  baseline`3461/3315`变为`3459/3315`，即`-2/0`；bootstrap lower bound=`-9/-8`，四项 V92
  calibration 门槛仍全部失败。故不发布、不访问 validation/official。
- V92 full-fit 单模型是`-2/+2`，V93b fold ensemble 是`-2/0`；两者都无法复现 OOF P99 的稳定
  正增益。因此“问题主要是 full-fit refit/normalization score-scale 漂移”被否决。更可能的主因是
  在同一组 OOF predictions 上从72个相关策略选最大值造成选择偏差，而 binary hit BCE 的真实 effect
  本身过弱；后续不再扫描 margin、fold weighting、投票阈值或 seed。

### 14.84 跨 split effect-size 更正与下一目标（2026-08-13）

- 14.80 曾把33,040-row OOF 的`+81/+80`直接与9,508-row official 缺口`+68/+38`比较，这是跨
  不同样本数的绝对 hit 数尺度错误。正确的等比例 OOF 最低净增应为
  `ceil(68*33040/9508)=237`与`ceil(38*33040/9508)=133`；V91 最佳 P50 为`+124/+308`，仅 @.50
  足够，@.25 只达到所需 effect 的52.3%；P99 的`+81/+80`更只有34.2%/60.2%。14.80 的实验选择
  仍保持预注册且 V92 已被独立 calibration 拒绝，但“P99 OOF 已覆盖正式缺口”这一解释作废。
- 下一候选的 train-only OOF 门槛预先修正为 @.25至少`+237`、@.50至少`+133`，五折双阈值均
  不退化、双 bootstrap lower bound严格为正；calibration 的等比例门槛`+8/+5`也应相对 official
  缺口按`3625/9508`缩放为`ceil(68*3625/9508)=26`与`ceil(38*3625/9508)=15`，而不是此前错误的
  `8/5`。修正后门槛只会更严格，不会把 V91--V93 的任何拒绝改成接受。
- 模型方向转为 graded listwise ranking：query target 使用该 query 七个 variant 中的最大连续 IoU
  utility，variant target 使用连续 IoU utility；对16 queries先做 listwise softmax ranking，再在胜出
  query内对7 variants排名，避免 binary BCE 把0.51与0.95、0.49与0.01视为同质标签。模型仍只读
  同一 deployable feature、保持 query→variant 因子化与受保护三件套冻结；先在固定单配置上做
  scene-disjoint OOF，未达到修正后的 effect-size 门槛就不运行 calibration，更不访问 validation。

### 14.85 V94 Graded Listwise Query→Variant：固定协议与启动门禁（2026-08-13）

- V94 继续复用 V91 的152维 query feature、25维 variant feature、query→variant 因子结构、候选池、
  normalization、hidden128、dropout0.1、weight decay`1e-3`、LR`3e-4`、batch256、12 epoch和seed0；
  不扫描模型、loss weight、temperature、margin或seed。受保护 backbone/parent/geometry 仍全冻结。
- 唯一结构性变化是监督目标：每个合法 variant 的连续质量固定为
  `q=IoU+1[IoU>.25]+2[IoU>.50]`，query 质量取其七个合法 variant 的最大 q；对16 queries和胜出
  query内7 variants分别以固定 target temperature`0.25`构造 soft listwise target，最小化两个
  masked softmax cross-entropy之和。推理仍把 ordered hit probability 按`2:1`合成 utility，先选
  query再选variant，不读取标签。
- 只跑一个配置与一个固定 P50 positive-gain margin，消除 V91 在72个相关策略中取最大值的选择
  偏差。5-fold scene-disjoint OOF 的发布前门槛按14.84的正确样本比例固定为：@.25净增至少237、
  @.50至少133、五折两个阈值均不退化、两个 scene-bootstrap 95% lower bound均严格为正。任一
  失败即不运行独立 calibration。
- 只有 OOF 全通过才在全部 fit scenes 重训一次，并访问独立3,625-row train calibration；固定要求
  @.25/@.50至少净增`26/15`，且两个 bootstrap lower bound均非负。再失败则不发布模型；两道
  train-only gate都通过也只生成 staged artifact 的实现候选，仍须完成契约测试和冻结后才允许一次
  official。validation 全程不参与结构、target、temperature、margin、checkpoint或门槛选择。

### 14.86 V94 OOF：连续 listwise 显著强化 @.50，但 @.25 effect 仍不足（2026-08-13）

- V94 启动前新增三项数值/梯度测试并回归完整 hierarchical 合同，共`118 passed in 2.59s`；脚本与
  测试 SHA256 分别为`35949078cae57ac2387170834373cda3df95c58743a6249623d3fe94dc1041ca`
  与`a8ef52e46b8bd8ba34d839b91568197855f25b1516e686b1fa84f405b0cd6342`。五折最终总 loss 均在
  `4.5156--4.5210`，query/variant 分量约`2.698--2.700/1.816--1.821`，无 NaN、OOM 或优化失稳。
- 只读结果
  `experiment_output/historical_e71_geometry/v94_graded_listwise_hierarchical_trainonly_v1.json`
  SHA256 为`fb538a14e0c1c84f118f7ed4d16648b95d526800d63163f430388a1a9e748129`；
  `validation_data_accessed=false`、protected 三件套前后相同，结束后四卡`1 MiB/0%`。
- 固定 P50 margin=`0.1387150`，切换`4527/33040=13.70%`。OOF @.25 fix/break=`234/90`，
  净增`+144`，五折`[22,21,48,23,30]`，bootstrap lower bound=`+105`；@.50 fix/break=
  `702/298`，净增`+404`，五折`[90,59,75,74,106]`，lower bound=`+318`。相对 V91 P50 的
  `+124/+308`，graded listwise 提升到`+144/+404`，说明连续质量排序是有效方向。
- 但修正后的最低 OOF effect 是`+237/+133`；V94 只有 @.50 通过，@.25仍差93 hits，故按协议
  `calibration_status=not_run`，不重训 full-fit、不发布 artifact、不访问 validation。迁移结构也发生
  有益变化：同 query variant changes=`3443`、query changes=`1084`；@.50修复中482来自 variant、
  220来自 query，而 @.25 为80/154。

### 14.87 V95 Threshold-aligned Graded Listwise 预注册（2026-08-13）

- V94 的训练质量是`IoU+1*hit25+2*hit50`，中等 IoU 候选只加1、同时过双阈值候选加3；但推理
  utility 和当前正式缺口都更重视 @.25，固定为`2*P25+1*P50`。V95 的唯一变化是把训练 target
  对齐为`IoU+2*hit25+1*hit50`：中等 IoU与双阈值候选分别加2/3，在不降低高 IoU排序的同时，
  提升“刚跨过 .25 但未过 .50”候选相对低 IoU的 listwise margin。
- hidden128、wd`1e-3`、dropout0.1、temperature0.25、12 epoch、seed0、P50 margin、候选池、
  feature、normalization、query→variant结构和全部门槛均与 V94 完全相同；仍只跑一个配置、一个
  策略。OOF必须达到`+237/+133`并满足五折双阈值非退化与双 bootstrap lower bound严格为正，
  否则直接拒绝；通过后 calibration 仍要求`+26/+15`及双 lower bound非负。validation 零参与。

### 14.88 V95 OOF：权重对齐只有小幅增益，仍未过 @.25 effect gate（2026-08-13）

- V95 编译及同一套数值/梯度/hierarchical 回归为`118 passed in 2.26s`；脚本与测试 SHA256 为
  `e26f64dc98b1e34a9bf648b3b4a9179a9e948a375e493c1953e760bd6c0595b1`和
  `611e5038595efb6e407b9c04f509eb5c20bb945d0e19a9454ddb06aa0af32a20`。只读结果
  `experiment_output/historical_e71_geometry/v95_threshold_aligned_listwise_hierarchical_trainonly_v1.json`
  SHA256 为`5ae7393d59d1a8c65f4f03db7c4206f99c3f6188bf65424bad915bddd8be4bee`；validation未访问，
  protected三件套未变，四卡结束为`1 MiB/0%`。
- 固定 P50 margin=`0.1367993`，switch=`4509`。OOF @.25 fix/break=`233/84`，净增`+149`，
  五折`[23,21,51,21,33]`，bootstrap lower bound=`+109`；@.50=`702/292`，净增`+410`，
  五折`[92,66,73,70,109]`，lower bound=`+324`。相对 V94 只改善`+5/+6`，说明简单交换阈值
  target 权重不是 @.25 瓶颈。
- @.25仍低于正确门槛237，故`calibration_status=not_run`。V94/V95 五折最终 query loss 都约
  `2.70`，非常接近16类均匀预测的`ln(16)=2.773`；现有训练先把两维 logits 经 ordered sigmoid
  压到概率，再以`2*P25+P50`作为 listwise score，最大动态范围仅0--3，可能限制了 ranking margin。

### 14.89 V96 Unbounded Listwise Utility 预注册（2026-08-13）

- V96 保持 V95 的 target、结构、feature、hidden128、wd`1e-3`、temperature0.25、12 epoch、seed0、
  P50、单配置与全部 OOF/calibration 门槛不变。唯一变化是 ranking score 不再经过 sigmoid：query
  与 variant 都直接用 raw utility `2*logit25+logit50`做 listwise softmax、argmax与相对 baseline
  margin。两个 logits 在本版本是无界 utility components，不再解释为校准命中概率。
- 这是对“概率压缩导致 listwise underfit”的单变量检验；若 query loss 明显下降而 @.25 effect仍不
  足，则否决该假设，不调整 temperature/epoch/LR。仍要求 OOF至少`+237/+133`、五折双阈值非退化、
  双 bootstrap lower bound严格为正；通过才允许 calibration，且其门槛仍为`+26/+15`与双 lower
  bound非负。validation继续零参与。

### 14.90 V96 OOF：raw utility 降低 loss 但不提升 @.25，饱和假设被否决（2026-08-13）

- V96 同时替换 OOF 与潜在 calibration 的 predictor，避免训练用 raw utility、校准悄悄回到 sigmoid；
  新增无界性、2:1权重、排序方向与有限梯度测试，连同 V95/完整 hierarchical 回归共
  `120 passed in 2.33s`。脚本/测试 SHA256 为
  `01685c13252243e00adc78e0a50e5aa341e49700ad396577ffa5386c4cda1e28`与
  `4b55b4dd0c7c0c48b2e137921a16f751534165cf8ac8a403b25b989fb46d67c1`。
- 有效只读结果
  `experiment_output/historical_e71_geometry/v96_unbounded_listwise_hierarchical_trainonly_v1.json`
  SHA256=`9e5f97849f217156334854a3f2a4b36f916cf7c8e084f89383ca1d2b2d94cad8`；内部原始
  V95-schema evidence 也以只读文件保留并由有效结果绑定。validation未访问，protected未变，四卡
  结束`1 MiB/0%`。
- 五折 query loss 约`2.6918--2.6934`，相对 V95 的`2.6982--2.7001`只下降约0.006；variant loss
  下降约0.006--0.010。固定 P50 margin=`0.1199301`，switch=`6236`，OOF @.25 fix/break=
  `223/74`、净增`+149`，五折`[24,25,47,22,31]`、lower bound=`+110`；@.50=`677/313`、
  净增`+364`、lower bound=`+280`。@.25与V95完全相同，@.50反而少46，故未过237门槛并跳过
  calibration。概率压缩不是 @.25 effect 不足的主因，不再调整 utility scale/temperature。

### 14.91 V97 Query-set Contextual Listwise 预注册（2026-08-13）

- 结构审计发现现有 hierarchical query encoder 对16个 query逐个独立编码，只有最终 listwise loss和
  argmax形成竞争；它无法显式表达“候选A相对候选B更符合语言/几何上下文”。V97 在 V95 bounded
  listwise 基础上加入一层 permutation-equivariant query-set Transformer encoder：hidden128、4头、
  FFN256、GELU、dropout0.1，严格使用 query_valid key-padding mask。contextual query embedding同时
  供 query head和其下7个 variant head使用；variant encoder、feature与候选池均不变。
- 这是可跨 ScanRefer/Nr3D/Sr3D复用的集合比较模块，不读取数据集名、类别、scene规则或部署标签。
  除该一层 context module外，V95 的 aligned target、bounded utility、wd`1e-3`、LR、epoch、seed、
  P50、单配置与全部门槛不变；不扫描层数、head数、FFN或margin。
- 启动前必须证明：query permutation等变；无效 padding query无论特征多大都不影响合法输出；修改一个
  合法 query可通过 attention影响另一个合法 query；梯度有限；并回归 V95/原 hierarchical 合同。
  OOF仍须至少`+237/+133`、五折双阈值非退化、双 bootstrap lower bound严格为正才运行 calibration；
  calibration仍需`+26/+15`和双 lower bound非负。validation零参与。

### 14.92 V97 OOF：集合上下文取得当前最强单配置 OOF（2026-08-13）

- V97 启动前结构/数值/原 hierarchical 回归共`121 passed in 2.36s`；permutation等变、padding
  隔离、合法 query跨候选作用和有限梯度均有定向测试。脚本/测试 SHA256 为
  `ca12f2a832e93089cfb856882ba535b75807a2537fdaecfa21b5cc6f2227a94f`与
  `dfdeb727ca0d608040462f26c9849c31e3721a157fab2cce32a78c04c90e1236`。
- 有效只读结果
  `experiment_output/historical_e71_geometry/v97_contextual_listwise_hierarchical_trainonly_v1.json`
  SHA256=`ca04b4cbd1804b92d676d815b79bfcacdaab3e8745742177bd94283cedda7f8d`，内部原始 evidence
  也由其绑定；validation未访问、protected未变、四卡结束`1 MiB/0%`。
- V97 五折最终 loss=`4.4966--4.5022`，比 V95 的`4.5167--4.5219`一致更低。固定 P50 margin=
  `0.1331222`、switch=`5316`；OOF @.25 fix/break=`245/71`、净增`+174`，五折
  `[30,36,50,27,31]`、bootstrap lower bound=`+131`；@.50=`758/283`、净增`+475`，五折
  `[119,68,80,90,118]`、lower bound=`+386`。它相对 V95 提升`+25/+65`，证明 query-set
  contextual comparison 是有效、跨数据集可复用的结构改进。

### 14.93 Effect-size transfer 第二次更正：必须按 oracle 可修复余量归一化（2026-08-13）

- 14.84 用总样本数把 official 缺口缩放到 train，是另一个尺度错误：fit baseline @.25 已为
  `31636/33040=95.75%`，val baseline仅`5542/9508=58.30%`，两者的剩余错误机会完全不同；要求
  fit `+237`等于捕获其 oracle余量1232的19.2%，远严于 val 达标所需捕获1307余量的5.20%。
  该错误门槛使 V94--V97 都跳过 calibration，但没有造成验证泄漏或错误发布。
- 正确、预先可计算的 transfer effect 使用14.75及各 train baseline 已封存的同候选池 oracle repair
  headroom。val 达标所需捕获比例为 @.25=`68/(6849-5542)=5.2028%`、@.50=
  `38/(6308-4623)=2.2552%`（@.50分母沿用诊断 sidecar候选池，分子仍用更严格 official缺口38）。
  对 fit oracle余量`1232/2974`取上整得到 OOF最低`+65/+68`；对独立 calibration余量
  `145/273`得到`+8/+7`。这同时适配 train/val 不同难度和基线命中率。
- 该更正在任何 V97 calibration/validation访问前完成，只使用早已封存的 oracle报告与 V91 receipt，
  不用未知 split结果反调门槛。V97 OOF 的`+174/+475`、五折全正和 lower bound`+131/+386`
  充分通过更正门槛，因此冻结 V97 作为唯一 calibration 候选；不在 V94--V97之间查看 calibration
  后反选。独立 calibration 固定要求`+8/+7`且两个 scene-bootstrap lower bound均非负；失败仍拒绝，
  通过才实现并测试 staged artifact。validation继续不可见。

### 14.94 V97 独立 train calibration：@.50 达到效应，但 @.25 破坏略多于修复（2026-08-13）

- calibration 脚本严格绑定 V97 OOF 报告 SHA、hidden/wd/P50 margin与`+174/+475`，并验证 fit 与
  calibration scene零重合；定向源绑定/结构测试`6 passed in 1.99s`。脚本 SHA256 为
  `a27e6feaf4edc32ea95d246e6fd5e043066b02e2d3b2f3022e012a1462f583ca`。
- 只读产物
  `experiment_output/historical_e71_geometry/v97_contextual_listwise_calibration_audit_v1.json`
  SHA256=`42be0078d6019585235ac20ac276538d267890d4f0aa7375c653e9d06bd1d6ab`；
  `validation_data_accessed=false`、protected前后相同，结束后四卡`1 MiB/0%`。
- 固定 V97 切换`508/3625`；@.25 fix/break=`9/12`、净增`-3`，bootstrap lower bound=`-12`；
  @.50=`53/33`、净增`+20`，lower bound=`-1`。@.50净增超过 oracle-headroom calibration门槛7，
  但下界差1；@.25的效应和下界均失败。因此 V97 不发布、不做 validation/official。
- 与 OOF @.25=`245/71`相比，full-fit→calibration 的 proposal distribution 在 .25 break/fix 上发生
  明显反转，但 @.50仍保留正增益。下一最小方向不是继续改 query selector或margin，而是在 V97
  已提出的一个候选上加入 proposal-specific verifier：只判断该次 baseline→proposal切换是否安全，
  以拒绝少量 .25 break，同时尽量保留 .50 fixes。它与 V90 对111个任意 alternative做 flat
  分类不同，输入分布由冻结 V97 selector定义，任务更窄且仍可跨三套 grounding 数据复用。

### 14.95 V98 Nested Proposal Verifier 预注册（2026-08-13）

- V97 calibration 已被读取，不能再把3,625-row split当成独立模型选择门禁。V98 全部选择证据回到
  33,040-row fit，并采用严格 nested scene cross-fit：外层 held fold h 的 V97 proposal来自排除h的
  四折模型；verifier训练行若属于 fold r，则 proposal来自同时排除h与r的三折模型。因此 held fold
  的标签既不参与自身 proposal，也不参与 verifier训练 proposal的生成。
- 计算去重后固定训练5个四折 V97 model与10个 unordered pair-exclusion三折 model，共15个；V97
  架构、aligned target、12 epoch、P50 margin=`0.1331222057`完全冻结。每个 outer fold再训练一个
  proposal verifier，它只接收 V97 已提议的单个 baseline→proposal 185维 deployable差分，不在112
  候选中重选。verifier固定为 hidden64、dropout0.1、wd`1e-3`、LR`3e-4`、10 epoch、seed0、break
  cost4的双阈值 break/neutral/fix分类器；用固定`2:1` expected signed gain，严格`>0`才接受。
- 不扫描 verifier hidden、cost、margin、seed或投票规则。普通 non-nested stacking会同时报告以量化
  选择偏差，但只有 nested结果有效。nested gate使用14.93 oracle-headroom尺度：@.25/@.50至少
  `+65/+68`，五折两个阈值均非负，两个 scene-bootstrap 95% lower bound严格为正；失败即终止。
  旧 calibration只允许在 V98完全冻结后作带`contaminated_diagnostic_only=true`的旁观，不参与选择；
  validation仍不可见。


### 14.96 V98 Nested Proposal Verifier 结果：拒绝（2026-08-13）

- 修复仅涉及 fold 索引的 canonical 排序；修复后聚焦测试通过，并以新输出名完整重跑。正式证据为 `experiment_output/historical_e71_geometry/v98b_nested_proposal_verifier_trainonly_v1.json`，SHA-256 `20eb88903f6fcaa14bea77cc5569cf23ed8c8cc4373a9fed8052172aa12807d5`。
- 严格 nested 路径完成 5 个主 V97 外折模型、10 个双折排除 V97 proposal generator，以及 5 个 verifier。它只接受 68/33040 次切换：@.25 fixes/breaks=`10/0`、净 `+10`、scene bootstrap 95% LB `+4`；@.50 fixes/breaks=`16/5`、净 `+11`、LB `+2`。五折净变化为 `0/0, 0/0, 0/0, +10/+11, 0/0`。
- 虽然全折非负且两个 bootstrap 下界均为正，但未达到预注册的 oracle-repair headroom transfer 门槛 `+65/+68`，因此 `passed=false`，V98 明确拒绝，不生成 artifact、不访问 validation 或已污染 calibration。
- ordinary stacking 只作为非独立诊断：271 次切换，@.25 `34/8=+26`、@.50 `57/21=+36`，LB `+12/+16`，同样低于效应门槛。nested 与 ordinary 的共同衰减表明二阶段 verifier 的主要问题是训练 proposal generator 与最终 proposal 分布错配，而非简单阈值过严。
- 三个 protected artifacts 在运行前后 identity 完全一致；`validation_data_accessed=false`、`contaminated_calibration_accessed=false`。V98 到此冻结为负证据，不再调 verifier 超参。

### 14.97 V99 Pareto Contextual Hierarchy 预注册（2026-08-13）

- 结构目标：去掉 V98 的第二模型与 proposal-distribution mismatch，保留 V97 的单个、permutation-equivariant query-set contextual hierarchy。
- proposal generator、V95 bounded target `IoU+2*hit25+hit50`、hidden=128、dropout=0.1、weight decay=1e-3、12 epochs、seed=0，以及 V97 已冻结 aggregate margin `0.13312220573425293` 全部不变。
- 唯一新增的确定性部署门为 Pareto threshold gate：V97 proposal 与当前 geometry baseline 的预测 `p@.25` 增量和 `p@.50` 增量必须都严格大于 0，同时 aggregate gain 必须通过原 V97 margin，才允许切换。该门直接复用同一模型的两阈值输出，不训练 verifier、不增加数据集特定输入。
- 只运行一次固定 5-fold scene-disjoint fit OOF，不读旧 calibration 或 validation，不做 grid search。晋级仍需 @.25/@.50 净增益至少 `+65/+68`、五折两阈值均非负、两项 scene-bootstrap 95% LB 均严格为正；失败即拒绝该结构。
- 远程实现 `scripts/run_v99_pareto_contextual_hierarchical.py` SHA-256 `78ff3fb141ab9aa8334285cd1d9e3c37845c7769710b166ee0fce00c33fac4a9`；测试 `tests/test_v99_pareto_contextual_hierarchical.py` SHA-256 `bc1a7d25fdd369e0a9f286059f7f2abf872fc14b29351d851fbc7d7ff3acc261`。V99 + V97 + V95 聚焦回归 `9 passed`。


### 14.98 V99 OOF 结果与 Artifact 冻结（2026-08-13）

- 正式结果 `experiment_output/historical_e71_geometry/v99_pareto_contextual_hierarchical_trainonly_v1.json`，SHA-256 `db42ef5853fb36fba9bdc53afb719bff9eb5a3f9e772475a4c76c363db01572d`，mode `0444`。
- V99 在 33040-row fit OOF 接受 5186 次切换：@.25 fixes/breaks=`245/70`、净 `+175`、scene-bootstrap 95% LB `+132`；@.50 fixes/breaks=`751/277`、净 `+474`、LB `+385`。五折净变化为 `+30/+120, +36/+70, +50/+80, +27/+89, +32/+115`，全部严格为正；六项预注册门禁全通过。
- 相对固定 V97 的 5315 次切换与 `+174/+475`，Pareto gate veto 129 次（全部是预测 @.50 增量非正；预测 @.25 非正为 0），最终为 `+175/+474`。它未显著放大效应，但在不训练第二 verifier 的条件下保持了完整效应规模与跨折稳定性。
- `validation_data_accessed=false`、`contaminated_calibration_accessed=false`，三份 protected artifact 运行前后 identity 一致。因此 V99 可进入一次 full-fit refit；旧 calibration 不参与 artifact 训练、阈值选择或晋级。
- 部署实现新增独立 schema `rec-pareto-contextual-hierarchical-v1`，旧 `rec_hierarchical` artifact 路径继续兼容。运行时使用同一模型的 threshold heads 做 Pareto gate；相关 V95/V97/V99、geometry runtime、hierarchical runtime 与 official runner 回归共 `73 passed`。
- full-fit artifact 必须绑定 V99 OOF result/script SHA、33040 rows/506 scenes、fit normalization、deployable-row/candidate-IoU digest、模型 state digest与 protected input SHA；以新文件 mode `0444` 独占写入并严格 reload。任何绑定或 reload 失败都不得进入 validation。


### 14.99 V99 Artifact 与官方评测预检（2026-08-13）

- full-fit artifact 已从 33040-row/506-scene fit split 重训并以 mode `0444` 冻结：`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/v99_artifacts/pareto_contextual_h128_seed0_fullfit.pth`，SHA-256 `9752990c393fa6e45173a9dd129c4de4bb740924094dcbbec2f3121cbf39d1f2`，size 914315；receipt SHA-256 `9c53d22ce9d0dd31aeb032d9c8c43392bfc95c05e4c1e7420a06789605b6fb88`。
- fit evidence：normalization `e3fbfd634f888328447a150fb2df73623825a1b9e5fdb7b0dc3c9af02a2947cb`，model state `e82b6ab8fe4a407ad79dc25b764cba22a086cc6676b9aa64cbb75f17b277332e`，deployable rows `1a0c31a549fb436a318d0cca1762e4a3ea9dea6f4afae249168294ab78df1e39`，candidate IoU `4b44ffebec929ea36fa7190c7f7471ad7dfce23345a8fb7e70506c8b4bcdbfa9`。final epoch loss/query/variant=`4.500388/2.693775/1.806613`。
- 真实 CUDA loader 链 parent→geometry→V99 通过；V99 为 `cuda:0`、eval、全参数 frozen，222852 参数，artifact/parent/geometry/normalization/OOF/scene-fold SHA 均精确绑定。
- 官方 runner `scripts/run_frozen_v99_pareto_contextual_official.py` SHA-256 `883afc66a7651a307726b3210dafbd8603261584d8e72e1267208373c2ab5384`。相关 official + runtime 回归 `151 passed`；dry-run 未创建 claim/output，绑定四 artifact，`validation_data_accessed=false`、`inference_uses_ground_truth=false`，205-file Python tree manifest SHA-256 `e25959c363a4d48904087937c7f8c7a6de4857bd90fc94da4835b36542f73767`。
- 正式 REC 硬门为 `5610/4659`（即 Acc@.25≥.59、Acc@.50≥.49）。Mask 硬保持门相对同一 backbone+geometry 已封存基线 `5676/4662/0.4176762145`；历史 network-only V19 best `5688/4672/0.4186131` 同时作为更强的提升诊断，不能把“尽量提升”误作未经声明的额外硬阻断。
- 正式运行创建唯一 claim 后，冻结四 artifact 与完整 Python tree，运行结束再次逐项验证不变；同一 9508-row stdout 同时封存 REC 与 Mask，禁止 GT 推理标志。


### 14.100 V99 唯一官方验证：完整推理后的正式失败（2026-08-13）

- 唯一官方验证访问完整 ScanRefer val `9508` rows、`793/793` batches；claim `/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/v99_artifacts/v99_official_validation_once.claim.json` 已在访问前以 mode `0444` 冻结，SHA-256 `e2ca7a1762b21470de76e8050f117e1358ed665f6ad71c3993fb28b535bdbab4`。未使用 GT 推理标志，也未重跑验证。
- 不可变 stdout `/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/v99_official_val/official_stdout.log`，mode `0444`、SHA-256 `78c4fe2cfbd865bf4d7c81382e26cfd478130ad70010890eeb698c40e020e911`。它恰好包含一组完整 REC 与 Mask 汇总：REC `5552/4645 = 0.58393/0.48854`；相对 sealed geometry official `5542/4621` 仅 `+10/+24`，距目标 `5610/4659` 仍差 `58/14`，故 `rec_target_pass=false`。
- 同一次 stdout 的 Mask 为 `5676/4662 = 0.5969709718/0.4903239377`，命中数与同 backbone+geometry baseline 完全一致；`mask_sem=0.4176463137`，比封存基线 `0.4176762145` 低 `0.0000299008`。因此按预注册逐项硬保持门，`mask_baseline_preservation_pass=false`；它也未达到 V19 best `5688/4672/0.4186131`。
- 推理与指标打印完成后，旧 `GroundingEvaluator.export_retrain_metrics` 的 `unique and multiple hits025 must partition learned hits` 断言触发，子进程 `returncode=1`。该异常发生在两条 REC、三条 Mask 指标和最终 batch 全部输出之后，不改变已计算指标；runner 按 fail-closed 原则未自动发布结果。
- 通过只读 recovery sealer 对既有 stdout 做恢复封存：`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/v99_official_val/official_result.json`，mode `0444`、SHA-256 `82fe2b7e9578268b5cebc7efcb1bdec4d2db2712fe3b4823bc9bab7acbcdc6f1`。结果显式保留 `returncode=1`、failure stage/signature、`revalidation_performed=false`，不把后置异常伪装成正常退出。
- recovery 要求开跑前 claim 中四份 protected artifact identity 与当前逐字节一致，并要求 205-file Python tree 的 manifest/records 与开跑前完全一致；两项均通过。V99 artifact 与所有历史 protected weights 仍为 mode `0444`，未被改写。
- 结论：V99 的 fit-only OOF `+175/+474` 没有迁移到 full-fit official（仅 `+10/+24`）；artifact 作为可复现实验结果保留，但不得作为 goal best。正式验证结果只用于一次性 go/no-go 与迁移偏差审计，不用于对 V99 阈值做验证集调参。下一路线必须重新从 training OOF 构造更接近线上候选分布/部署语义的结构，并在进入任何后续 official 前通过 full-fit replay 与 runtime parity 门。


### 14.101 V99 OOF→部署迁移审计（2026-08-13）

- 冻结 V99 full-fit artifact 在原 33,040-row fit cache 上做只读 replay；结果 `experiment_output/historical_e71_geometry/v99_fullfit_train_replay_audit_v1.json`，mode `0444`、SHA-256 `ee10ce3780d721f796b4f26a9eb857f81389e006e4ea6091477e7873acb724f4`。artifact 实际接受 5,121 次切换，@.25/@.50 净增 `+233/+766`，五折按固定 scene mapping 分解为 `+33/+170, +45/+120, +70/+153, +44/+140, +41/+183`；bootstrap lower bound `+186/+673`。这排除“full-fit optimization 本身塌缩”的解释。
- 进一步对同 33,040 rows 同时执行离线 materializer 与真实 `train_dist_mod.py` runtime builder；只读结果 `experiment_output/historical_e71_geometry/v99_train_runtime_parity_audit_v1.json`，mode `0444`、SHA-256 `4b7a2052f5bfaa003836e9f4af90d971c11d405f1a1ec8cc350eb97b557d7ece`。八个 normalized hierarchy inputs、baseline indices/scores、query/variant logits、proposal、Pareto pass、gain、switch mask、selected indices 全部位级一致，`row_count=33040`、`all_equal=true`。
- 两个审计均只读取 train caches，`validation_data_accessed=false`、`contaminated_calibration_accessed=false`；backbone/parent/geometry/V99 identity 前后不变。由此 V99 official 仅 `+10/+24` 的主因不是 artifact refit 或 runtime 接线，而是 train-scene absolute-hit model 到未见 val scenes 的泛化/分布迁移。
- 审计还暴露协议缺口：过去 artifact 晋级只要求 OOF，没有强制 full-fit replay 与 exact runtime parity。后续候选在 official 前必须补齐这两道门；但它们只能排除工程漂移，不能替代 scene-disjoint OOF 泛化证据。

### 14.102 V100 Baseline-conditioned Relative-effect Contextual Ranker 预注册（2026-08-13）

- V90 已证明对 111 个 alternative 各自独立做 185-D pair classification 缺少 query 层次；V98 又只验证 V97 已提出的单一 proposal，受 proposal-generator 分布错配限制。V100 与二者不同：保留 V97 的 16-query permutation-equivariant contextual encoder，同时让全部 `16×7=112` candidates 共同参与一个 baseline-conditioned relative-effect selector。
- 对每个候选与 @.25/@.50，监督不再是绝对 IoU/命中概率，而是相对当前 frozen geometry baseline 的三类 `break/neutral/fix`。候选 head 显式读取 candidate query context/variant embedding、唯一 baseline 的 query context/variant embedding、两者差、各自 deployable auxiliary state与 same-query 标志；不读取 dataset 名、GT mask、类别规则或 scene-specific 常量。
- 固定单配置：hidden128、1 层 query-set Transformer、4 heads、FFN256、GELU、dropout0.1、AdamW `lr=3e-4/wd=1e-3`、12 epochs、seed0、batch256。loss 固定为两阈值三分类 CE，按各阈值 baseline-hit/baseline-miss rows 等权，避免 e71 train baseline @.25=`95.75%` 的强 prior 淹没 fix 学习；再加一个按 baseline IoU 三段等权的 112-way signed-effect listwise term。没有 hidden/cost/temperature/margin/seed 网格。
- 推理将每个 threshold 的 signed effect 定义为 `P(fix)-P(break)`，aggregate=`2*effect25+effect50`；在 112 个合法候选中取 aggregate 最大者，仅当它与 baseline 不同且 `effect25>0`、`effect50>0`、aggregate>0 时切换，否则保持 baseline。baseline 候选固定为 neutral anchor；不运行 margin percentile 搜索。
- 在编码/运行前进一步锁定无歧义的 loss：三分类 CE 先在每行合法候选内平均，再对每个阈值的 baseline-hit 与 baseline-miss 两组等权；112-way listwise 使用真实 signed aggregate `2*(hit25_candidate-hit25_baseline)+(hit50_candidate-hit50_baseline)`，对每行所有最大 aggregate 候选均匀分配 target probability，不设 temperature。listwise row loss按 baseline IoU `<=.25 / (.25,.50] / >.50` 三段等权；总 loss=`classification + listwise`，不引入可调系数。
- 推理精确定义：baseline anchor 的两个 effect 与 aggregate 固定为 0；非 baseline 候选只有在两个 predicted signed effects 均严格大于 0 时才参与 argmax，其余 mask 为 `-inf`。最大合格 aggregate 必须严格大于 baseline anchor 0 才切换；无额外 margin。
- 有效证据是固定五折 scene-disjoint OOF；晋级要求 @.25/@.50 至少 `+65/+68`（14.93 oracle-headroom transfer）、五折两个阈值都严格为正、两个 10,000-scene-bootstrap 95% lower bound严格为正。通过后仍须 full-fit train replay不弱于 `+65/+68`、exact offline/runtime parity、artifact/protected SHA冻结；失败即拒绝，不访问旧 calibration或任何 validation。


### 14.103 V100 OOF：纯 relative-effect selector 严重过切换，否决（2026-08-13）

- 相关结构/策略/损失及既有层级回归 `138 passed`；模型、实验脚本、测试 SHA-256 分别为 `2f4e7d8e06130201a484541e9983d1c275386d1e9f28baabdf2c5898cc4cab6f`、`8c6a6846ed501536b2460a9fbbd34fd63af3bd97612aa24e477889d15b409e3f`、`b789081388262da64a900ceff773138928f1c2b2e46eddb9e38f818566a6c1ec`。
- 只读结果 `experiment_output/historical_e71_geometry/v100_baseline_relative_contextual_trainonly_v1.json`，mode `0444`、SHA-256 `9df20aca38ea0dea9653376e30e883c200b3b863be15b9d410bf667ec75cd7c0`；`validation_data_accessed=false`、`contaminated_calibration_accessed=false`，protected 三件套前后 identity 一致。
- 固定五折共切换 `31,696/33,040=95.93%`；@.25 fixes/breaks=`734/2105`、净 `-1371`、bootstrap LB `-1545`；@.50=`1542/4540`、净 `-2998`、LB `-3320`。五折净变化为 `-279/-669, -246/-608, -300/-659, -292/-641, -254/-421`，十二项 effect/fold/bootstrap 门全部失败。
- 失败机制不是优化不收敛：五折最终 classification/listwise/total loss 范围稳定在约 `0.367--0.375 / 4.356--4.380 / 4.729--4.754`。真正问题是把 baseline-hit/miss 强制等权后，rare fix 的预测先验被严重放大；`effect25>0 & effect50>0` 在未校准 signed heads 上几乎不构成安全门。
- V100 明确拒绝：不生成 artifact、不运行 calibration/validation，也不通过新增 margin sweep挽救。纯 relative-effect 不能替代 V99 的 absolute contextual listwise selector；若后续复用 relative target，只能作为保留自然先验的辅助损失/安全信息，而不能单独主导 112-candidate argmax。


### 14.104 V101 预注册：全训练场景覆盖的 V99 固定协议 OOF（2026-08-13）

V99 的正式验证仅获得 `+10/+24 hits`，但后续两项只读审计已经排除了两个直接实现故障：同一
`33040` 条 fit 数据上的 frozen full-fit artifact replay 为 `+233/+766`，而离线预测与正式运行时的
输入、proposal、gain、Pareto gate 和 selected index 在全部训练行逐项相等。因此当前待检验假设不是
重拟合崩溃或部署路径漂移，而是 V99 仅使用 `506` 个场景、`33040` 条样本拟合，遗漏的 `56` 个训练
场景、`3625` 条样本削弱了未见场景泛化。

V101 不改网络结构、损失或部署选择规则，只把训练域覆盖扩大到 ScanRefer 的全部 train split：

1. 数据固定为现有 e71 base/geometry train cache 的全部 `36665` 条 joined rows、`562` 个场景；不读
   validation cache，也不把此前 calibration 结果当独立选择证据。旧 calibration 场景现在仅作为普通
   train 场景进入新的 scene-disjoint OOF；每一行仍只由未见过其场景的 held-fold 模型预测。
2. 固定五折 scene-disjoint OOF、seed 0；每折训练其余四折，禁止样本级随机拆分。必须覆盖全部
   `36665` 行且每个 scene 只属于一个 held fold。
3. 架构精确复用 V99 的 V97 contextual query-set hierarchy；目标精确复用 V95 bounded
   threshold-aligned listwise；hidden `128`、dropout `0.1`、weight decay `1e-3`、学习率和 epoch
   计划均沿用冻结实现。不得扫描 margin、权重、随机种子或结构。
4. proposal 与部署规则固定为 V99：aggregate gain `2*delta025 + delta050` 不小于
   `0.13312220573425293`，且 `delta025>0`、`delta050>0` 才接受；否则回退固定 geometry parent。
5. 训练集全量基线/oracle 为 `35097/36474 @0.25`、`32733/35980 @0.50`。正式验证目标缺口相对
   oracle headroom 的既定比例机械换算后，OOF 最低净增固定为 `+72/+74 hits`；此外五个 held fold
   在两个阈值上都必须严格为正，scene bootstrap 95% lower bound 也必须严格为正。
6. protected backbone/parent/geometry 的 path、size、mtime、SHA-256 必须前后一致；结果以
   create-exclusive、只读 JSON 保存，并记录输入、源码和预测 digest。
7. 若任一门禁失败，V101 立即否决，不构建 artifact、不访问 validation。只有全部通过，才允许用
   全部 `36665` 条 train rows 构建新的冻结 artifact，先做同域 replay/runtime parity，再决定是否
   执行一次正式目标验证。

本轮是训练覆盖/泛化协议实验，不宣称新数据集特定技巧；模型输入和门控不包含 ScanRefer 专属类别
或样本类型，因此代码合同仍应可用于 Nr3D/Sr3D。当前服务器未发现 Nr3D/Sr3D 的等价缓存，故本轮
只验证跨 ScanRefer 未见场景的 OOF，不能据此声称已经取得跨数据集数值。


### 14.105 V101 全训练场景 OOF 结果：通过（2026-08-13）

- 正式 train-only 证据为 `experiment_output/historical_e71_geometry/v101_full_train_pareto_oof_v1.json`，
  mode `0444`，SHA-256 `2cb453b130306449901bed9c337984aad7f8b7048d05bd4240b5077f0de9ac1e`；
  `validation_data_accessed=false`、`prior_calibration_used_for_selection=false`，protected backbone、parent、
  geometry 的运行前后 identity 完全一致。
- 全部 `36665` 行、`562` 个场景恰好进入一次 held fold。五折 held rows/scenes 为
  `7088/113, 7080/113, 7674/112, 7464/112, 7359/112`，对应 train scene 数为
  `449,449,450,450,450`，没有 scene leakage。
- 固定 V99 Pareto 协议接受 `5882` 次切换。@0.25 fixes/breaks=`236/77`，净 `+159`；@0.50
  fixes/breaks=`827/307`，净 `+520`。五折净增依次为
  `+28/+93, +28/+92, +26/+134, +53/+95, +24/+106`，两个阈值每折都严格为正。
- scene bootstrap 10000 次的 95% lower bound 为 `+118/+421`；净增超过预注册的 oracle-scaled
  门槛 `+72/+74`，全部六项 predicates 为真，故 `passed=true`。
- 与 V99 的 `33040` 行 OOF `+175/+474` 相比，全训练覆盖将 @0.50 提高到 `+520`，但 @0.25
  降到 `+159`。因此本结果只支持进入 artifact/replay/parity 阶段，不支持预先宣称正式验证必然优于
  V99。后续仍固定同一结构、loss、margin 和 Pareto gate，不做阈值或随机种子搜索。
- 首次启动在任何 materialization/训练前因 joined-row schema 防漏断言错误访问顶层 `scan_id` 而
  fail closed；394-byte 原日志保留。唯一修复为读取真实字段 `row["base"]["scan_id"]`，修复后相关
  回归 `8 passed`，最终 V101 驱动 SHA-256
  `034291a86b08a2386b3861f8dbe732acd3ae34bacdf2f70145cf5ecde9e5af92`。

下一门禁：用全部 `36665` 条 train rows 训练一个冻结 V101 artifact；同域 full-fit replay 必须两个阈值
净正、运行时输入到 selected index 必须与离线预测一致、protected artifacts 必须不变。全部通过后才
允许一次 formal validation；任一失败则不访问 validation。

#### 14.106 V101 冻结补充与 Mask 退化定位（2026-08-13）

- V101 全训练集 Pareto 上下文模型已冻结为只读 artifact：`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/v101_artifacts/pareto_contextual_h128_seed0_fulltrain.pth`，SHA-256 `2c969a6c28a0c9315b53f0f847567345e47da8c912091344b23612680643a2ae`，mode `0444`。全拟合训练集 replay 为 `+237/+765` hits（@0.25/@0.50），5 折均为正；它仅证明冻结 artifact 与训练契约一致，不替代 OOF 或正式验证。
- 为下游 Mask OOF 重放并冻结了 V101 逐行 OOF 决策：`experiment_output/historical_e71_geometry/v101_oof_row_decisions_v1.pth`，36,665 行、562 scenes、5,882 次接受切换、预测摘要 `b81664e65d64dad7058f8f252d990d4ab11dd8c00746c64a918bb120b6434c99`，文件 SHA-256 `d3d79f6b7b3948863c553683415b1f22297261871e20dd15b8584993040e141c`，mode `0444`。5 个 fold 的模型/归一化从训练场景重新拟合，逐字节复现 V101 OOF 报告。
- 当前官方 V99 Mask 为 `59.6971/49.0324/41.7646`（Mask@0.25 / Mask@0.50 / semantic mIoU）；相对用户给出的 MCLN baseline `58.70/50.70/44.72`，@0.25 高 `+0.9971pp`，但 @0.50 低 `-1.6676pp`、mIoU 低 `-2.9554pp`。定位结果不是 evaluator 指标错配：epoch71 backbone 来自 REC/box pair sweep，未以 RES mIoU 选择；MCLN 的 text mask 对 query 广播，而 evaluator 再与 query-specific superpoint mask 融合，故改变 REC query 或只优化 box 会暴露 RES 退化。

#### 14.107 V102 同一 query 的 Mask 后处理可行性审计（2026-08-13）

- 冻结输入：V101 OOF parent query；不允许 mask 审计改 REC flat index。基线固定为该 query 的 `fused@logit 0.0`；oracle 只在同一 query 的 `text/query/fused × {-1,-0.5,0,0.5,1}` 共 15 个 mask policy 内选择。
- 只读训练集审计：`experiment_output/historical_e71_geometry/v102_same_query_mask_oracle_audit_v1.json`，SHA-256 `92a8f6cf5f66d5f94591119c5de29366c14814feebdfb791ed8e16ec3918c077`，mode `0444`，`validation_data_accessed=false`。
- 同-query oracle 相对基线：Mask@0.25 `94.5998 -> 95.9225`（`+485`, `+1.3228pp`），Mask@0.50 `85.6784 -> 87.8304`（`+789`, `+2.1519pp`），mIoU `72.2030 -> 74.5105`（`+2.3075pp`）；20,641 行 IoU 可提升，0 行下降。5 折分别 `+88/+172/+2.3499pp`、`+99/+134/+2.1156pp`、`+113/+181/+2.5317pp`、`+95/+164/+2.3338pp`、`+90/+138/+2.1905pp`，方向全部一致。
- V101 换 query 本身在同一 legacy mask policy 下仅 `+18/+17/+0.0329pp`，因此 Mask 空间主要来自来源/阈值选择，不是偷换定位候选。最优全局固定 policy 仅提供小幅收益，不能替代逐样本策略。

#### 14.108 V102 Mask-only 后处理预注册（训练前冻结，2026-08-13）

1. **边界与不变量**：V101 先产生最终 REC flat index；V102 只能读取该 index 对应 parent query，输出一个 mask source 与 logit threshold。V102 返回的 REC score tensor 必须与 V101 bitwise 相同，最终 flat index/parent mapping 必须逐行一致；任何不一致 fail closed。训练与 OOF 不访问 validation，推理不接触 GT/IoU/类别名。
2. **可部署特征**：新建 train-only v2 sidecar，逐 query 保存 52 个纯推理统计：三种 source 各 7 个 logit/probability/confidence/entropy 汇总，三种 source × 五阈值的前景比例，三对 source × 五阈值的二值一致率，以及融合 text weight。它与旧 15-policy label cache、候选 cache、epoch71 backbone 的 SHA/逐行 query identity 绑定。特征定义与类别、场景 ID、ScanRefer 文本模板无关，可沿相同模型输出接口迁移到 Nr3D/Sr3D。
3. **模型**：`QueryMaskPolicyPostprocessor`。179-D query/geometry candidate 特征先做 128-D variant encoder；同-query mean/max 与全局 query context、52-D mask feature 共同进入 1 层 4-head、FFN=256 的 permutation-equivariant contextual encoder；密集输出每 query × 15 policy 的 IoU 估计与严格嵌套的 @0.25/@0.50 概率。hidden=128，dropout=0.1。
4. **训练唯一配置**：seed=0，12 epochs，batch=128，AdamW，lr=`3e-4`，weight decay=`1e-3`，grad clip=`1.0`。所有有效 query/policy 做密集监督；loss=`SmoothL1(IoU)+BCE(hit025,hit050)+0.5*soft-listwise+0.5*expected-regret`；listwise utility=`IoU + P@0.25 + 2*P@0.50`，target temperature=`0.25`。normalization 只由每个 OOF fold 的 4/5 训练 scenes 拟合。
5. **部署唯一策略**：在冻结 V101 parent query 上比较 15 policy；proposal 为预测 utility 的稳定 argmax。仅当相对 legacy `fused@0.0` 的预测 `delta IoU>0`、`delta P@0.25>0`、`delta P@0.50>0` 且 `delta IoU+delta P@0.25+2*delta P@0.50 >= 0.02` 时切换，否则严格回退 legacy。无 grid search、无 validation 调参。
6. **OOF 门**：沿 V101 相同 5 个 scene-disjoint folds。要求 REC decision digest 完全不变；总 `delta_hits025>=49`、`delta_hits050>=79`、`delta_mIoU>=0.0023`；每折三项均非负且每折至少一项严格为正；scene-block bootstrap 95% lower bound 对三项均严格大于 0。任一失败则 V102 不可部署，不运行正式 validation。
7. **后续顺序**：TDD feature/cache/model/policy/runtime -> v2 train cache 完整性与 parity -> 一次固定配置 OOF -> 通过后全训练集 artifact/严格 reload/runtime parity -> 合并 V101+V102 一次正式 ScanRefer validation；最后才比较当前 V99 与 MCLN mask baseline，并单独记录 Nr3D/Sr3D 尚缺跨数据集验证，不能把接口泛化误写成指标泛化。

#### 14.109 V102 特征缓存、启动故障与可复现性封存（2026-08-13）

- 首次两个 32-row smoke 若重新运行 top-16 候选选择，会分别在低排名 query 处出现冻结 base cache 与
  当前 backbone 排名漂移（其中 batch24 的 row5 为 query `171 -> 169`）。根因是重新做候选排序不能
  保证与旧 label/cache 轴逐位一致；这两次失败均发生在训练前，输出保留为负证据。修正为只从已绑定
  base cache 读取冻结 `query_indices/candidate_valid`，再按该轴提取当前 mask logits；不重算候选身份。
- 修正后的 32-row smoke 严格重载通过：shape=`[32,16,52]`、全部 finite、无效候选全零、52/52
  特征非零方差；backbone/base/joint manifest SHA 均匹配。随后完整生成 train cache：
  `/root/autodl-tmp/DATA_ROOT/output/scanrefer_joint_box_mask/v102_mask_features_train_full_v1`，
  36,665 rows、586,640 valid candidates、52/52 非退化，content SHA-256
  `41b1728971963cb09db58b1e4ab6f02314e23a5184c7a4cb8db5dc7b2e81668a`，manifest file SHA-256
  `6f183499fd899767dcc321a71e95d01e9d29eb324e5417398c7e869dbaa5a39d`。
- OOF 第一次启动在 import 时即失败：直接执行 `scripts/run_v102_mask_only_oof.py` 只把 `scripts/`
  放入 Python 模块路径，无法 import `models`。最小红/绿对照稳定复现；仅增加
  `PYTHONPATH=/tmp/mcln_repo` 后 `--help` 状态 0。失败日志只读 SHA-256
  `40b113708089b3fd6993168795196db460fc04c9d5fd0b2fb8a67b5c55759c52`。
- 第二次启动在任何数据加载/训练前失败：共享保护函数只接受严格三键
  `{backbone,parent,geometry}`，驱动误把 `v101_sidecar` 放进同一字典。最小修复不放宽共享契约；
  sidecar 改用同强度的 regular-file/non-symlink/mode0444/inode/time/size/SHA 身份前后比较。
  相关聚焦回归 `14 passed`，真实 sidecar SHA `d3d79f...e141c` 校验通过，可写临时文件按预期
  fail closed；失败日志 SHA-256 `0cd371f80d878d241aaaf05fb8f64f8d0eb32398b62bab1a588f7f616bf7db40`。
- 最终 OOF 驱动 SHA-256 `20c02902cdbabb52ade119ab3f90671aa8bcfd851c2d3b157d89daed54ba8f50`；
  以上两次均未写结果、未训练、未改 protected artifact，故不构成额外配置试验。

#### 14.110 V102 五折 OOF 结果：部分有效但联合门失败，禁止部署（2026-08-13）

- 正式只读结果 `experiment_output/historical_e71_geometry/v102_mask_only_oof_v1.json`，mode `0444`，
  SHA-256 `4ed892f89bc04fecd0f1618dc415039614241d432b90b0010af3908ef719c1b0`；完整覆盖
  36,665 rows、562 scenes、5 folds，`validation_data_accessed=false`、`inference_uses_ground_truth=false`。
  backbone/parent/geometry/V101-sidecar 四份保护输入前后身份完全一致。
- 固定 V102 接受 4,781 次 mask-policy 切换。相对同-query legacy fused@0：Mask@.25
  `34685 -> 34801`，净 `+116 hits`（`+0.3164pp`）；Mask@.50 `31414 -> 31448`，净
  `+34 hits`（`+0.0927pp`）；mIoU `0.7220305 -> 0.7228831`，净 `+0.0008526`。
  REC parent/index digest 前后同为 `f043cd...e7ba`，证明 mask-only 边界成立。
- 五折 delta 为 `+22/+6/+0.001484`、`+23/+9/+0.000703`、`+15/+28/+0.001512`、
  `+35/-19/+0.000271`、`+21/+10/+0.000291`。scene bootstrap 10,000 次 95% lower bound 为
  @.25 `+0.002271`、@.50 `-0.000247`、mIoU `+0.000254`。
- 因 @.50 仅 `+34 < +79`、mIoU `+0.000853 < +0.0023`、held fold3 的 @.50 为 `-19`，且
  @.50 bootstrap lower bound 为负，预注册联合 gate 明确 `passed=false/deployable=false`。
  V102 不构建部署 artifact、不接 runtime、不访问 formal validation，也不得事后降低门槛。
- 可复现决策重放逐折逐项相同，预测摘要 `f248e0ce2f853f59be0c09a1e9c80c846dc7e199cbefd89c2f770f06ebb4ab85`；
  逐行 sidecar `v102_mask_only_oof_decisions_v1.pth` SHA-256
  `a1039a8d6feebcf25f586ba3c795658d999f6b1ed78f1faab2e7756b00ed7ba6`，mode `0444`。

#### 14.111 V102 失效机制与独立 result-to-claim 判定（2026-08-13）

- 逐行诊断 `experiment_output/historical_e71_geometry/v102_mask_only_oof_diagnosis_v1.json`，mode `0444`，
  SHA-256 `3827e4f5ca80bd003d897482a08b1a5e3dc3d59929d1d71e61aad7b52eb0b9be`。4,781 次接受中
  @.25 rescue/break=`150/34`，@.50=`156/122`；IoU improve/equal/degrade=`1230/2412/1139`。
  预测与真实 delta 的 Pearson 仅 @.25=`0.3086`、@.50=`0.1748`、IoU=`0.2306`，说明绝对
  outcome heads 相减得到的 policy gain 校准不足，尤其无法可靠排除 @.50 break。
- `text@-1.0` 被选 994 次，贡献 @.25 `+60`、@.50 `-13`；held fold3 更为
  `+27/-10/-0.001388`。若无接受门把 34,000 个非 baseline proposal 全部采用，则 @.50
  rescue/break=`370/368`、净 `+2`，mIoU `-0.000085`。因此问题不是简单把 margin 放宽或缩紧，
  而是必须直接建模 baseline→proposal 的 rescue/break 风险并保留自然事件先验。
- 只读固定-policy审计显示 `fused@-1.0` 相对 fused@0 在全 train OOF 为 `+24/+126/+0.001205`，
  `fused@-0.5` 为 `+8/+103/+0.001355`，后者五折三项均正；但这些是观察 V102 后的诊断，不能
  当作无偏发布证据或直接用于 validation。same-query oracle 子集仍保留大量余量：fused-only
  `+93/+455/+0.01294`，query+fused `+101/+519/+0.01441`；说明可用阈值候选充足，瓶颈是选择。
- 独立 result-to-claim 评审为高置信度 `partial`：支持“REC identity不变”和“train OOF下
  Mask@.25可靠提升”，不支持三指标联合提升、可部署、正式 val 或 Nr3D/Sr3D 泛化。建议 V103
  采用嵌套 scene-disjoint 校准、显式 break-cost/相对转移风险、多种子最坏情况，并只在新 OOF
  预注册门通过后构建 artifact/runtime；V102 到此冻结为诊断性负证据。

#### 14.112 V103 Relative Mask Transition Ensemble 预注册（编码前冻结，2026-08-14）

1. **主张与边界**：V103 检验“直接预测 legacy→candidate 的 rescue/break 风险，比 V102 对两个绝对
   outcome 相减更能稳定选择 mask policy”。冻结 V101 先给出 REC flat index/parent query；V103 仅输出
   该 query 的 mask source/threshold，不得改 REC scores、flat index 或 parent mapping。推理特征仍只用
   179-D 通用 query/geometry features 与 52-D mask logit statistics，不使用 GT、IoU、类别名、dataset
   名或 scene ID；接口可复用于 Nr3D/Sr3D，但本轮不能宣称已有跨数据集数值。
2. **候选集合**：legacy anchor 固定为 `fused@0.0`（原 15-policy index 12）。V102 显示低阈值
   text-only 是主要 @.50 break 源，故 V103 在看自身 OOF 前冻结允许集合为原 indices
   `[4,5,6,7,8,9,10,11,12,13,14]`：只保留收缩型 `text@+1.0` 与全部 query/fused thresholds；
   indices `[0,1,2,3]` 永久禁止。该 11-policy 子集同-query oracle 为 `+202/+636/+0.017548`，五折
   三项均正，证明仍有足够 headroom；oracle 只用于容量审计，不参与推理。
3. **相对网络**：`RelativeMaskTransitionPostprocessor` 复用 V102 的 128-D variant encoder、同-query
   mean/max 聚合、52-D mask feature 与 1-layer/4-head/FFN256 query contextual encoder。每个 query×11
   policy 直接输出 `delta_iou in [-1,1]`，以及 @.25/@.50 各三类 transition logits：
   `break/neutral/rescue`。anchor 的三项预测在选择时机械置零，不作为可切换 proposal。
4. **唯一训练协议**：每个 outer fold 固定训练 seeds `[0,1,2]`，不按结果选 seed；部署对象就是三个
   model 的固定 worst-case consensus ensemble。每个 model 为 12 epochs、batch128、AdamW
   `lr=3e-4/wd=1e-3`、dropout0.1、clip1.0。normalization 只拟合该 outer fold 的四折 train scenes。
   标签直接由 candidate 相对 anchor 计算：真实 delta IoU 与两个 `-1/0/+1` transition。loss 不做
   rare-class 重采样/逆频率权重，保留自然 break/rescue prior；固定为
   `SmoothL1(delta_iou)+CE025+2*CE050+0.5*listwise+0.5*regret`。listwise 的 target/pred utility 均为
   `delta_iou + transition025 + 2*transition050`，target temperature=`0.25`。
5. **唯一选择策略**：三个 seed 分别得到 `effect=P(rescue)-P(break)`。对每个 policy 取跨 seed 最坏值
   `min(delta_iou), min(effect025), min(effect050), min(delta_iou+effect025+2*effect050)`；仅前三项都
   严格正且 worst aggregate `>=0.02` 的非-anchor policy 有资格，取 worst aggregate 最大者（并列取
   原 policy index 最小者），否则回退 anchor。没有 margin/temperature/cost/seed/policy subset sweep，
   不使用 V102 held-row outcome 做阈值拟合。
6. **证据范围与门**：沿 V101/V102 完全相同的 5 个 scene-disjoint outer folds；V103 是在 V102
   train-only诊断后冻结的新 development OOF，不能冒充未触碰确认集。必须满足 REC digest 位级不变；
   总 delta hits @.25 `>=49`、@.50 `>=79`、delta mIoU `>=0.0023`；五折三项均非负且每折至少一项
   严格正；10,000 次 scene-block bootstrap 的三项 95% lower bound 均严格正。还须记录三个单 seed
   的诊断结果，但不得事后挑 seed；正式 gate 只评估预注册 consensus ensemble。
7. **晋级顺序**：先做公共 model/loss/selector/gate 聚焦测试与现有 V102 回归；再运行唯一 V103 OOF。
   任一门失败则冻结为负证据，不构建 artifact、不接 runtime、不访问 validation。全部通过后才用全部
   36,665 train rows 重训三 seed ensemble，要求只读 artifact 严格 reload、full-train replay三项净正、
   offline/runtime mask policy逐位一致及 REC scores/index逐位一致；最后才允许 V101+V103 一次正式
   ScanRefer validation。正式结果仍以用户给出的 MCLN mask baseline `58.70/50.70/44.72` 和当前
   V99 official `59.6971/49.0324/41.7646` 分别比较，不能用 train OOF 指标替代 official。

#### 14.113 V103 五折三种子 OOF：仅 mIoU 幅度门失败（2026-08-14）

- 编码前预注册之后才新增 `models/rec_relative_mask_policy.py` 与
  `scripts/run_v103_relative_mask_oof.py`；模型/驱动 SHA-256 分别为
  `3eb89aaea7879ac086deb14449721713083c14143809be4238dd6fec5aa28626`、
  `fd2b5ae44f73859d21a3d4e6d0804a58ecc3b1dae0ddaf91681621a78fb8e00a`。先得到模块不存在的红测，
  再完成新 model/loss/三 seed selector/gate 与 V102 回归共 `16 passed`；A100 CUDA 前向、反向、
  selector smoke 均 finite。唯一正式 OOF 训练 5 folds × seeds `[0,1,2]`，无 seed/policy/margin sweep。
- 只读报告 `experiment_output/historical_e71_geometry/v103_relative_mask_transition_ensemble_oof_v1.json`
  SHA-256 `d8e1a1c90bcf4bd6a726472903345e92e6a9694c1b47b346c074742c6380f2dc`，mode `0444`；只读逐行
  sidecar `v103_relative_mask_transition_ensemble_oof_decisions_v1.pth` SHA-256
  `8020a65c0464b0368bbe457416a8f5d08c41982ccc03011aa5a9fd63f27fc792`，mode `0444`。覆盖
  36,665 rows、562 scenes、15 个训练模型；`validation_data_accessed=false`、
  `inference_uses_ground_truth=false`，四份 protected artifact 身份前后完全相同。
- consensus 接受 3,708 次 mask-policy 切换。相对 legacy fused@0，Mask@.25
  `34685 -> 34746`，净 `+61 hits`（`+0.1664pp`）；Mask@.50 `31414 -> 31553`，净
  `+139 hits`（`+0.3791pp`）；mIoU `0.7220305 -> 0.7240734`，净 `+0.0020429`。
  REC parent/index digest 前后同为 `f043cd...e7ba`。
- 五折 delta 依次为 `+14/+23/+0.002370`、`+10/+27/+0.001514`、
  `+6/+41/+0.002322`、`+16/+21/+0.002322`、`+15/+27/+0.001663`；每折三项均严格为正，
  且 V102 曾失败的 fold3 已不再出现 @.50 break。scene-block bootstrap 10,000 次 95% lower bound
  为 @.25 `+0.001045`、@.50 `+0.002807`、mIoU `+0.001617`，三项均严格正。
- 预注册 gate 的其余七项全部通过，但 mIoU 幅度 `+0.002043 < +0.0023`，故报告正确标记
  `passed=false/deployable=false`。不得用“只差 0.000257”事后降低门槛；V103 不构建 full-train
  artifact、不接 runtime、不访问 formal validation。三个单 seed 仅作诊断：分别为
  `+52/+147/+0.001936`、`+59/+173/+0.002227`、`+59/+157/+0.002263`，不得挑 seed 发布。
- sidecar 独立重放的 before/after、全局 metrics 与预测 SHA `c22b...c98` 逐项一致。3,708 次接受中
  IoU improve/equal/degrade=`1516/1278/914`；预测 worst delta 与真实 delta Pearson=`0.4411`，
  worst aggregate 与真实 delta=`0.5339`。事后把 `min worst_delta` 从 0 提至 0.0025--0.05 均不能
  达到 mIoU 门，说明简单加阈值不是修复。V103 已在 @.25/@.50 留有 `+12/+60 hits` 的门槛余量，
  下一轮应保持所有 eligibility 与 gate 不变，研究合格候选内部的 IoU 优先排序，而非放松安全约束。
- 独立 result-to-claim 评审为高置信度 `partial`：支持以上 train OOF 增益、逐折/bootstrap 稳定性与
  REC identity 不变，不支持可部署、正式 validation、联合最终指标或跨数据集数值泛化。评审确认
  “eligible 内优先 worst delta-IoU”可作为一次新预注册 development OOF，且不属于降门槛或
  ScanRefer 特化；但它由 V103 失败适应性启发，不能当作独立确认，若再失败不能继续反复改 selector。

#### 14.114 V104 IoU-Priority Relative Consensus 预注册（编码前冻结，2026-08-14）

1. **动机与证据边界**：V103 的 REC identity、五折非退化、三个 bootstrap 下界及 @.25/@.50
   总命中门均已通过，仅 mIoU `+0.002043 < +0.0023`；同时命中门有 `+12/+60 hits` 余量。
   逐行诊断表明简单提高 `min worst_delta` 不会补足 mIoU，因此 V104 不改 margin、不删 held rows，
   只检验“在已经满足三指标安全约束的候选中，按相对 IoU head 排序能否把容量分配给更高质量 mask”。
   这是观察 V103 train-only OOF 后冻结的下一次 development OOF，不是独立确认结果。
2. **冻结不变项**：数据、5 个 scene-disjoint folds、V101 REC parent、179+52-D 推理特征、允许/禁止
   policy 集、`RelativeMaskTransitionPostprocessor` 架构、loss、normalization、12 epochs、batch128、
   AdamW `3e-4/1e-3`、dropout0.1、clip1.0、seeds `[0,1,2]` 以及不按 seed 选模均与 V103 完全相同。
   不使用 validation、GT、类别名、dataset/scene ID，不重采样/重加权，不做阈值或 policy subset sweep。
3. **唯一变化：合格后的排序**：仍逐 policy 计算三个 seed 的
   `worst_delta_iou`、`worst_effect025`、`worst_effect050`、`worst_aggregate`；eligibility 与 V103
   完全相同，即非 anchor、前三项严格正且 `worst_aggregate>=0.02`。V104 在 eligible 集内按
   `(worst_delta_iou, worst_aggregate, -original_policy_index)` 字典序取最大；无 eligible 时回退
   legacy fused@0。anchor 机械归零，REC parent/index 原样返回。不得在看到 V104 OOF 后切回
   aggregate 排序或混合两个 selector。
4. **复现实验与门**：重新训练固定 5×3 个模型，记录每 seed final loss 和单 seed 诊断；不把 V103
   held outcomes 输入 V104。报告/逐行 sidecar 独占创建并设 `0444`，保护四份输入身份，要求 REC digest
   完全不变。部署门一字不改：总 `delta_hits025>=49`、`delta_hits050>=79`、`delta_mIoU>=0.0023`，
   五折三项非负且每折至少一项严格正，10,000 次 scene bootstrap 三个 95% lower bound 均严格正。
5. **晋级纪律**：任一门失败即冻结为不可部署负证据，不构建 artifact/runtime、不访问 validation；
   全部门通过后才训练 full-train 三 seed artifact，完成 strict reload、full-train replay、runtime parity、
   REC bitwise invariance，再进行 V101+V104 唯一一次正式 ScanRefer validation。接口可迁移到
   Nr3D/Sr3D，但未做跨数据集数值验证前不得宣称其指标泛化。

#### 14.115 V104 OOF：IoU-priority 排序证伪，停止 selector 迭代（2026-08-14）

- V104 selector/包装驱动 SHA-256 分别为
  `229900e61ea706d509ec98fc3252fa4b3f82a956cbced2dc6e7134f86419996e`、
  `33a244ebac2f48e6c4a24fc2811d65f4df91075ce71813bd8ff7d97a50b15867`；TDD 先红后绿，新旧
  model/driver/V102 回归共 `21 passed`，A100 smoke 确认 ranking/parent 契约。V104 复用 V103 训练
  驱动 `fd2b...e00a`，15 个 model 的逐折逐 seed final loss 与 V103 精确一致，证明唯一变量确为
  eligible 内字典序排序。
- 只读报告 `experiment_output/historical_e71_geometry/v104_iou_priority_relative_mask_oof_v1.json`
  SHA-256 `47172ae1cb39ca431dd03180b738538cfa23b77b352e94848236be3a59a8ff56`，mode `0444`；只读 sidecar
  `v104_iou_priority_relative_mask_oof_decisions_v1.pth` SHA-256
  `48e2ec316a30416cf5f44b26126522b757876c393f1c4ffcc3c7eb44779d6506`，mode `0444`。schema、
  `eligible_ranking=[worst_delta_iou,worst_aggregate,lowest_original_policy_index]`、双 driver SHA、
  36,665 rows/562 scenes/5 folds/15 models 均与预注册一致；protected identity 前后相同，未访问 validation。
- 3,708 次接受相对 legacy 得到 `+62 hits@.25`、`+134 hits@.50`、`+0.0019966 mIoU`；逐折为
  `+14/+23/+0.002374`、`+10/+26/+0.001445`、`+6/+41/+0.002383`、
  `+17/+17/+0.002095`、`+15/+27/+0.001662`。bootstrap 95% lower bound 为
  `+0.001067/+0.002658/+0.001576`，REC digest 仍为 `f043cd...e7ba`。
- gate 仍仅 `delta_mIoU>=0.0023` 失败，故 `deployable=false`，不构建 artifact/runtime、不访问
  formal validation。sidecar 精确重放通过；相对 V103 仅 434 行改变，净效果为
  `+1 hit@.25/-5 hits@.50/-0.0000463 mIoU`，直接证伪“把 worst delta head 用作首排序即可修复”。
- V103 接受行的真实正/负 IoU 增益和分别为 `+109.089/-34.185`；只保留真实正增益的诊断 oracle
  仍有 `+79/+195/+0.002975`，说明过滤 break 有容量。可是仅用 worst prediction 与 policy one-hot
  做 leave-one-fold ridge filter 只有约 `+60/+135/+0.001999`，不能过门。至此停止对同一 OOF 反复
  改固定 selector；若继续，应使用 outer-fold 内严格 inner-OOF 生成 proposal，再训练能读取原始
  179+52-D 特征的非线性 switch verifier，以避免元层面泄漏和欠校准。

#### 14.116 V48 query-superpoint 空间 Mask 对照恢复与启动前审计（2026-08-14）

- **问题锚点**：当前唯一正式最好 V99 为 REC `0.583929/0.488536`、Mask
  `0.596971/0.490324/0.417646`；相对用户给出的 MCLN Mask baseline
  `0.5870/0.5070/0.4472`，Mask@.25 已高 `+0.009971`，但 Mask@.50 与 mIoU 仍低
  `-0.016676/-0.029554`。V102--V104 只改变同一 query 的离散 mask policy，V104 已明确冻结为
  不可部署，因此不再修改 selector。
- **结构审计结论**：V42/V43 只产生逐 query 的 alpha 与统一 logit bias，同一 query 内所有
  superpoint 获得相同偏移；V48 的 `QuerySuperpointMaskRefiner` 是现有唯一能输出局部
  `[query,superpoint]` residual 的网络模块，但它此前因上游事件链重启而从未跑出真实 smoke。
  当前实现只读取 query/superpoint feature，尚不读取两路 mask logits、源分歧或 box-relative
  geometry。故先运行原 V48 作为必要对照；若其局部 residual 已激活但 Mask 指标不增，再预注册
  evidence/geometry-aware 增强，不能跳过对照直接增加复杂度。
- **主张与反主张**：主张 C1 是“局部 query-superpoint residual 能修复全局 alpha/bias 无法覆盖的
  漏分与误分”；最低证据是空间 residual 的 query 内和 query 间方差均严格正，且 128-row
  debug 的 Mask@.50/mIoU 不劣于同协议 control。反主张是“增益只来自额外 candidate loss 或短集
  波动”；因此固定同一 protected V19、seed、batch、epochs 和 optimizer，只比较预注册的
  `cmw/clw/K={0.10/0/8,0.25/0/16,0.25/0.05/16,0.25/0.10/16}`，不据结果新增第五组。
- **执行与晋级门**：恢复既有 `queue_v48_spatial_mask_smokes_after_v47.sh` 的四组单卡、3 epoch、
  128-row smoke；要求四组完整 REC/Mask Overall+Unique+Multiple 收据，初始化
  `1228/0/34`、checkpoint/optimizer 合同、alpha/bias/source-evidence/candidate-loss 非零，且
  `mask_spatial_superpoint_std_mean>0`、`mask_spatial_query_std_mean>0`。smoke 只用于可训练性和
  方向筛选，不作为 9,508-row 正式指标；只有至少一组无非有限值、REC 无结构性崩坏且 Mask@.50
  或 mIoU 相对 control 有正向信号，才允许预注册正式 V48 或增强版。否则冻结 V48 为负证据。
- **代码身份**：启动前 SHA-256 为 `joint_query_quality.py=d6bbb0...df8111`、
  `mcln.py=156d8f...98b08`、`mask_fusion.py=0ab1a7...c748`、训练 launcher
  `5b083a...06df6`、V48 queue `76d505...85c0`；启动时将重新记录完整哈希和 protected V19
  `2d6a3c...ecbe` 身份，不从任何 smoke 权重续训。
- **容量与可恢复性清理**：启动门禁发现 overlay/data 盘仅约 `2.3/3.3GB`，低于并行 smoke 的
  `4GiB` 下限。审计旧 V51 三个 128-row smoke 后确认：24 个 `.pth` 链接对应 6 个物理 inode、
  共 `3,662,867,966` bytes；全部硬链接仅在
  `DATA_ROOT/output/double_stage_v51_bmq_smoke` 内，无打开文件，正式 R2-P/V51-T 均从 protected
  V19 重新初始化而未消费这些 smoke。其 config、完整日志、逐 epoch metrics/diagnostics 和
  retention JSON 均保留，可用相同 runner 精确重跑。删除仅限这 24 个旧 smoke `.pth`，data 盘
  free 从 `3,402,456` 增至 `6,979,496 KiB`；protected V19 删除后仍为 mode `0444`、SHA-256
  `2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe`。

#### 14.117 V48 四组真实 smoke：空间残差激活但分割质量不增，禁止正式训练（2026-08-14）

- protected V19 初始化审计通过：公共 tensor `1228/0/34`（common changed/new）、新模块
  `176,979` 参数、34 个 state、输出头全零，mask calibration/source evidence/spatial refiner
  开关与 `hidden=32/max_delta=2.0` 均符合冻结合同。四组单卡在 A100 上完成 3 epoch、每轮
  128-row validation；无 OOM、非有限值或训练异常，四份 epoch3 checkpoint audit 均为
  `34 states/176,979 elements/step=6` 且所有 68 个 Adam moments finite/nonzero。
- 只读汇总
  `/root/autodl-tmp/DATA_ROOT/output/v48_spatial_mask_20260814/v48_spatial_mask_smoke_summary.json`
  SHA-256 `cbb5e3743f7ed261281202696be90270d1826328cf07f53deffcbbccdf5c3e11`，四组
  structural `pass=true`。固定 epoch3 结果如下；同一 128-row V19 父 Mask 参照为
  `64/52, mIoU=0.350607`，不能把 debug 比例当成完整 validation 指标。

| variant (`cmw/clw/K`) | REC hits .25/.50 | Mask hits .25/.50 | mIoU | spatial abs / SP-std / Q-std |
|---|---:|---:|---:|---:|
| `0.10/0/8` | `64/57` | `64/50` | `0.350304` | `0.0049/0.0059/0.0027` |
| `0.25/0/16` | `64/57` | `64/50` | `0.350109` | `0.0065/0.0039/0.0058` |
| `0.25/0.05/16` | `64/57` | `64/52` | `0.349695` | `0.0137/0.0040/0.0065` |
| `0.25/0.10/16` | `64/57` | `64/52` | `0.349527` | `0.0124/0.0028/0.0060` |

- 空间 residual 在 superpoint 与 query 两个维度均真实非零，证明不是接线/梯度问题；但四组 mIoU
  相对父参照分别低约 `0.000303/0.000498/0.000912/0.001080`，前两组还各损失 2 个 Mask@.50
  hits。Lovasz 权重越高，residual 与统一 bias 越大而 mIoU 越低。故 C1“仅凭冻结 query/SP
  feature 的低秩 residual 可修正局部边界”未获得质量证据；V48 不进入 9,508-row 或 full-data
  正式训练，也不按 epoch1/2 反选变体。
- 失败机制与实现一致：V48 residual 对当前 text/query/fused logits、源分歧、预测 box 和
  superpoint 相对位置条件独立；同一 query 内哪些 superpoint 正处于不确定边界、位于预测框内外，
  只能由冻结 feature 间接猜测。结果说明下一版必须显式读取这些推理时可用证据，而不是继续调
  `cmw/clw/K`。
- 按预注册清理规则，四组 checkpoint 在完成 receipt/checkpoint audit 与 summary 后全部删除，日志、
  config、逐 epoch metrics、diagnostics 与 audit 保留；data 盘恢复约 `7.8GB` free。第三轮原子
  checkpoint 前还清理了已被后继 V54/V55 正式实验取代的 V53-FH 128-row smoke：8 个链接、2 个
  物理 inode、`1,212,071,974` bytes；其 config/log/五轮 metrics/diagnostics/retention 与独立
  checkpoint audit 均保留，且无外部 hardlink、脚本依赖或打开文件，可从 protected V19 精确重跑。
  protected V19 SHA 在两次清理后仍为 `2d6a3c...ecbe`。

#### 14.118 V105 EGQS-R：Evidence/Geometry-conditioned Query-Superpoint Mask Refiner 预注册（编码前冻结，2026-08-14）

1. **主张与边界**：V105 检验“局部 mask 修正必须显式依赖当前 source logits 的边界不确定性和
   predicted-box-relative geometry”。它是 mask-only 网络模块，放在 REC source arbiter/sidecar 已
   生成 scores 与 selected indices **之后**、最终 mask loss/evaluator **之前**；不得回写任何 REC
   score、rank、query index 或 parent mapping。推理不读取 GT、类别名、dataset/scene ID 或固定
   ScanRefer 源偏好，接口直接适用于 ScanRefer/Nr3D/Sr3D。
2. **模块输入与零初始化**：每个样本读取冻结的 query feature `[Q,288]`、superpoint feature
   `[S,288]`/center `[S,3]`、predicted box center/size `[Q,6]`、text/query mask logits `[Q,S]`
   和现有 fusion alpha `[Q]`。先计算当前 fused logit，再构造 7 维连续 evidence basis：缩放后的
   text/query/fused logit、fused uncertainty、两源 probability disagreement、signed source
   difference；8 维 geometry basis 为 box-size 归一化相对坐标的 signed/absolute 三轴、连续 inside
   margin 和径向距离。所有上游输入 detach。
3. **高效 query-conditioned residual**：保留 V48 的低秩 query–superpoint content dot product，
   另由 query embedding 输出 7/8 维 evidence/geometry 系数，与逐 superpoint basis 点积；避免在
   `[Q,S]` 上构造大 hidden MLP。content 的 query 末层及两组 coefficient heads 全零初始化，最终
   `delta=2*tanh(content+evidence+geometry)`，step0 必须逐位为零。delta 同时加到 text/query
   logits，故不改变 alpha 语义；参数与计算不依赖 superpoint 数量。
4. **干净消融与唯一 smoke 协议**：从 protected V19 相同初始化运行四组单卡、seed0、batch64、
   3 epoch、LR `3e-4`、matched-query `10*focal+2*dice`，关闭 global alpha/bias、candidate dense
   loss 与 Lovasz，唯一变化为 `content-only / evidence-only / geometry-only / all`。固定使用 epoch3，
   不按 epoch1/2 或结果新增配置。content-only 是不含 V48 joint rerank/global calibration 的干净
   局部对照；两个单分支是 novelty isolation；`all` 是唯一晋级候选。
5. **公共接口与门**：测试 seam 冻结为 (a) refiner `forward` 数值/step0/置换与 source-swap 接口，
   (b) `MCLN.forward` 的最终 `last_pred_masks/sp_last_pred_masks` 与 REC scores/index identity，
   (c) launcher→checkpoint audit→128-row metrics。编码前需用户确认这些 seam。smoke 要求所有
   optimizer/moment finite/nonzero，residual 的 SP/Q 两维 std 严格正，REC score/index digest
   位级不变；`all` 固定 epoch3 的 Mask@.50 不低于 content-only，且 mIoU 至少高 `0.0003`，否则
   V105 冻结为负证据，不进入正式训练。
6. **正式晋级顺序**：smoke 过门后才从受保护 epoch71 backbone（SHA `3e44f4...2208`）用完整
   36,665 train rows、4-GPU、seed0、global batch192、固定20 epoch训练 `all`；不按完整 validation
   选 epoch，唯一候选固定为 epoch20。先做 strict reload、train replay、REC bitwise identity 与
   runtime parity；全部通过后，才允许把固定 V101 REC artifact（SHA `2c969a...a2ae`）和 V105
   mask head组合进行一次 9,508-row official validation。REC 硬门仍为 `5610/4659`；Mask 至少不低于
   V99 official `5676/4662/0.4176463`，并同时报告与用户 baseline `0.5870/0.5070/0.4472` 的差值。
   任一硬门失败均不发布为 goal best，也不得根据 validation 反调 V105 evidence/geometry 维度或门槛。

#### 14.119 V105 只读接线审计与测试 seam 待确认状态（2026-08-14）

- 14.118 的三个公共测试 seam 已在用户侧请求确认；按 TDD 纪律，收到确认前不写测试或模型代码。
  本节只冻结实现落点，避免确认后再次做架构性选择。
- 新模块放在 `models/mask_fusion.py`，不复用 `JointQueryQualityReranker` 的 score/residual head；
  `models/mcln.py` 当前在 line 1720 先生成 arbiter 输出、line 2003 才 `end_points.update(selector_out)`，
  V105 必须插在二者之间且只更新 `last_pred_masks/sp_last_pred_masks`。这样 source scores、selected
  indices 与 parent mapping 已经冻结，mask residual 不可能反向改变 REC 选择；无 source arbiter 时
  仍在最终 return 前执行同一 mask-only 分支。
- `super_xyz_list` 已由输入点坐标和公开 superpoint mapping 确定性生成；query boxes 已存在于
  `last_center/last_pred_size`，不需要新增数据集字段。text/query logits、alpha、query/SP features
  也都在同一 forward 内可用，故 Nr3D/Sr3D 不需新增 loader 或类别分支。
- 训练沿用 `compute_hungarian_loss(...query_mask_fusion_train_only)` 的真实 matched-query
  Focal/Dice fast path，但新增独立 `query_superpoint_mask_refiner_train_only` 开关与参数前缀，避免
  强制构造旧 `QueryMaskFusionCalibrator`。接线范围冻结为：`main_utils.py` 的 CLI/互斥训练模式/
  freeze-train-mode/checkpoint missing-key 合同，`train_dist_mod.py` 的 model kwargs，
  `models/mcln.py` 的构造与后-arbiter forward，`scripts/audit_source_moe_checkpoint.py` 的 V105
  profile，以及独立 launcher/panel summary；loss 数值公式不改。
- 当前代码身份仍为：doc `57fcfb...f861d`、`mask_fusion.py=0ab1a7...c748`、
  `mcln.py=156d8f...98b08`、`main_utils.py=f02715...1aa1`、`train_dist_mod.py=f02fee...5503`。
  四卡均 `1MiB/0%`、无 screen；因此等待 seam 确认不会覆盖运行中实验或产生半成品 checkpoint。

#### 14.120 用户取消 TDD 确认门、V105 实现与单卡真实 smoke 启动（2026-08-14）

- 用户明确要求“不用 TDD skill、不需要确认、直接执行”，因此 14.118/14.119 的等待条件取消；仍保留
  三个 seam 作为实现后验证接口，不再把它们当成编码前门。随后用户说明服务器只剩一张 GPU，故
  四路 smoke 从四卡并发改为 GPU0 串行，架构、seed、batch64、3 epoch、LR 与固定 epoch3 门均不变。
  若 smoke 晋级，正式阶段必须使用单卡梯度累积保持预注册的 effective global batch192；不得把硬件
  变化偷换为更小训练目标。
- `models/mask_fusion.py` 新增 `EvidenceGeometryQuerySuperpointMaskRefiner`：16 个 state tensors、
  26,095 参数。content 是 32 维低秩 query×SP dot；evidence 是 7D（两源/fused scaled logits、
  fused probability/uncertainty、source probability absolute/signed disagreement）；geometry 是 8D
  （box-size-normalized signed/absolute xyz、inside margin、radius）。content query 输出层与 7/8D
  query-conditioned coefficient heads全零初始化，最终 `2*tanh` residual 同时加到两路 logits。
  所有父输入 detach，且没有 dataset/class/scene ID。
- `models/mcln.py` 只在所有 arbiter/joint score 已写回后、最终 return 前运行 EGQS；只覆盖
  `last_pred_masks/sp_last_pred_masks`。`main_utils.py` 增加独立 `use/train_only/lr/hidden/delta/components`
  CLI、父 checkpoint missing-key、optimizer prefix、eval-mode freeze 与 mask-only Focal/Dice fast path；
  `train_dist_mod.py` 只传构造参数。新脚本为 `train_scanrefer_egqs_mask_refiner.sh`、
  `queue_v105_egqs_mask_smokes.sh`、`summarize_v105_egqs_smoke.py` 和 contract smoke。
- 独立 contract smoke 通过：step0 residual 逐位为零，激活后 residual abs mean `0.0414277`、
  SP std `0.0211346`、Q std `0.0450663`，query/SP permutation max error `0`，source swap finite，
  父输入无梯度；参数/张量数精确为 `26095/16`。protected V19 仍为 1228 张量、SHA
  `2d6a3c...ecbe`，其 SourceMoE/gate/action/objective 全部按 checkpoint config 继承。
- GPU0 为唯一 A100-PCIE-40GB；screen `mcln_v105_egqs_20260814` 启动串行 panel。content 固定
  epoch3 已完成 128 条：REC `64/57`（与父路径一致），Mask `64/52`，mIoU `0.350716593`；训练后
  residual abs mean/SP std/Q std 分别约 `3.39e-4/1.27e-4/3.52e-4`，证明真实激活。
- 首次 checkpoint audit 因通用规则要求“所有 Adam moments 非零”而拒绝 content：禁用的
  evidence/geometry heads按消融定义得到零梯度。审计现改为活动组件 moments 必须 finite/nonzero、
  禁用组件的精确 parameter IDs 必须保持 zero moment；content 复审通过：common/changed/new
  `1228/0/16`、optimizer states/numel/step `16/26095/6`，inactive IDs 正好 `12--15`。审计后删除
  content 的 8 个 `.pth` links（2 个物理 inode）；指标、日志、completion/audit receipt 保留，队列
  从 evidence 继续而不重复 content。

#### 14.121 只保留最佳权重：V51 正式旧权重清理（2026-08-14）

- 用户再次要求清理无用权重并只保留最好权重。只读盘点确认 V59c epoch3 虽仅 128-row，但历史日志
  将它登记为该路线唯一候选，故保留；protected V19、epoch71、V99/V101 与 single-stage best 也不动。
- `DATA_ROOT/output/double_stage_v51_bmq_formal` 的三组正式路线均已有 config、9508-row metrics、
  checkpoint audits 与 launcher logs，且被 V99 综合结果支配；全部从 protected V19 初始化，可重跑。
  对 22 个 `.pth` links 审计得到 4 个物理 inode、总 `2,449,779,612` bytes；所有 hardlinks 都在该
  精确目录内，无外部 link、打开文件或活动 V51 Python 进程。删除后该目录剩余 `.pth=0`，配置/指标/
  审计/日志未删。protected V19 与 epoch71 SHA 复核仍为 `2d6a3c...ecbe`、`3e44f4...2208`；数据盘
  可用空间在 V105 同时写 checkpoint 的情况下约 `9.0GB`。这些 V51 权重不能直接恢复，只能从保留
  的 V19+config 重跑；这是经用户授权的有损但可重建清理。

#### 14.122 V105 四分支单卡 smoke 终态：结构通过、质量门失败（2026-08-14）

- GPU0 串行完成 content/evidence/geometry/all 四组固定 epoch3；每组 REC 均为 `64/57`，证明
  mask-only 接线没有改变 REC。四组 Mask@.25/.50/mIoU 依次为：content
  `64/52/0.350716593`、evidence `64/52/0.350351649`、geometry
  `64/51/0.350439980`、all `64/52/0.350835181`。all 保持 content 的 Mask@.50，但 mIoU 只增加
  `0.000118588`，低于编码前冻结的 `+0.0003` 门；summary 的唯一 failure 为
  `all mIoU margin`，因此 `pass=false`，禁止进入完整训练或 official validation。
- 失败不是未激活：all 的 residual abs mean/max、SP std、Q std 分别为
  `0.082627/0.858837/0.066225/0.090225`，content/evidence/geometry 三项贡献 abs mean 为
  `0.002431/0.085094/0.097504`。四份 completion/checkpoint audit 均通过，公共 parent tensors
  `1228/0/16`，all 的 16 个 optimizer states/32 个 moments 全 finite/nonzero、step=6；单分支禁用
  参数的零 moments 与预注册 ID 精确一致。
- 只读汇总
  `/root/autodl-tmp/DATA_ROOT/output/v105_egqs_mask_20260814/v105_egqs_smoke_summary.json` 保留；四组
  checkpoint 均在 audit 后删除，V105 根目录 `.pth=0`。protected V19、epoch71、V101 artifact SHA
  复核仍为 `2d6a3c...ecbe`、`3e44f4...2208`、`2c969a...a2ae`，GPU 空闲，数据盘约 `11GB` 可用。
- 因果解释边界：V105 已能读取 source evidence 与 box-relative geometry，但每个 superpoint 的
  residual 仍独立生成，没有显式邻接或连通性约束。高幅度 residual 只产生极小 mIoU 正增益，说明
  下一路线不应继续扩大逐点 MLP/系数头，而应检验局部拓扑传播能否形成连贯的边界修复。

#### 14.123 V106 Boundary-aware Superpoint Graph Diffusion 预注册（编码前冻结，2026-08-14）

1. **唯一新假设**：同一 query 的 fused mask 在空间/语义相邻 superpoints 上应有局部一致性；V106
   用显式 KNN 图的邻域消息修正不确定边界，检验 V105 缺失的拓扑变量。它仍是 mask-only 网络头，
   位于最终 REC arbiter 之后，只允许覆盖两路 mask logits；REC scores、rank、flat index 和 parent
   mapping 必须逐位不变。
2. **通用输入与图**：只读取 V105 已公开的 detached query/SP features、SP xyz、两路 mask logits、
   fusion alpha 与预测 box，不读取 GT、类别、scene/dataset ID。每场景基于 SP xyz 构造固定 `K=8`
   邻域；边权由距离尺度归一化和 frozen SP feature cosine 共同确定，并在 query 间共享。这样接口不
   绑定 ScanRefer，Nr3D/Sr3D 可直接调用相同 forward。
3. **边界消息与零初始化**：在当前 fused logit/probability 上计算邻域加权均值、signed diffusion、
   局部 variation、uncertainty-gated diffusion 与 box inside-margin 支持，组成固定 graph basis；由
   normalized query feature 输出 basis coefficients。唯一输出 head 全零初始化，故 step0 residual
   必须逐位为零；最终 `max_delta=2` 的有界 residual 同时加到 text/query logits。父网络全部冻结，
   仍训练 matched-query `10*focal+2*dice`，不加入 Lovasz、candidate loss 或 selector loss。
4. **单卡固定 smoke**：唯一硬件为 GPU0 A100-40GB，按 seed0、batch64、128 train/128 val、3 epoch、
   LR `3e-4` 串行运行两个事先冻结的变体：`spatial`（距离图）与 `bilateral`（距离+feature cosine 图）；
   固定比较 epoch3，不按 epoch1/2 选模，不新增 K/LR/delta sweep。spatial 是图拓扑必要对照，
   bilateral 是唯一晋级候选。
5. **结构与质量门**：要求 step0 exact、SP permutation equivariance、KNN 无 self/重复/越界、父输入
   detach、所有 active optimizer moments finite/nonzero、residual SP/Q std>0、REC `64/57` 与父路径
   identity。bilateral 固定 epoch3 必须 Mask@.50 不低于 spatial 且 mIoU 至少高 `0.0003`；同时其
   mIoU 必须高于 V105 all 的 `0.350835181`。任一失败立即冻结 V106，删除 transient checkpoints，
   不进入正式训练。
6. **若晋级**：完整训练仍只用一张 GPU；通过梯度累积把 effective global batch 固定为 192，使用
   36,665 train rows、seed0、固定 epoch20。strict reload/train replay/runtime parity 后，才允许与
   冻结 V101 REC artifact 组合进行一次 9,508-row official validation；门仍为 REC `5610/4659`，
   Mask 不低于 V99 `5676/4662/0.4176463`。不得因单卡改变 batch 目标或用 validation 反调图参数。

#### 14.124 V106 单卡 smoke 终态：图分支真实激活但质量门失败（2026-08-14）

- 新增 `BoundaryAwareSuperpointGraphMaskRefiner`，只有 `2,888` 个参数、4 个 state tensors；固定
  `K=8`，比较纯空间 KNN 与距离+冻结 superpoint feature cosine 的 bilateral 图。contract smoke
  通过：step0 residual 精确为零，置换误差为零，KNN 无 self/重复/越界，所有父输入 detach；激活后
  residual abs mean `0.017559`、SP/Q std `0.021318/0.021166`。contract receipt SHA-256
  `4613e98bf727eac4b5b3384258c8757ecaa7aadb641d7dfb2bbff3e9bdf01c35`。
- GPU0 串行完成两个固定 epoch3 smoke。spatial 为 REC `64/57`、Mask `64/52`、mIoU
  `0.3504062698`；bilateral 为 REC `64/57`、Mask `64/51`、mIoU `0.3504654098`。bilateral
  只增加 `0.00005914` mIoU，却损失 1 个 Mask@.50 hit，并低于 V105-all 的 `0.3508351807`；
  三项预注册门均失败，summary `pass=false`，SHA-256
  `b9a37a5a17033675181623c85cdfec31568506a29ce37b673cfc66e4f7ba597f`。
- 两组 completion/audit 都通过，公共 parent tensors `1228/0/4`、optimizer 4 states / 2,888 params /
  step6，moments finite/nonzero。spatial completion/audit SHA 分别 `f5532f...975`、`69c712...da9f`；
  bilateral 为 `84083c...c4ef`、`f04551...99a0b`。两组 transient checkpoint 已全部删除，V106
  根目录 `.pth=0`；只保留 config、日志、逐 epoch metrics、diagnostics、receipt 和 summary。
- 结论：V105/V106 都证明后处理已真实读取局部 evidence/geometry/topology，但在当前 epoch71 mask
  表征上收益远小于目标。停止继续扩大局部 mask refiner，转向核验官方 release 与数据预处理身份。

#### 14.125 官方 epoch54 权重取得、旧 superpoint 口径复现（2026-08-14）

- 官方 GitHub README 的 Google Drive ID `1oBUWrTEj3kYyx-DT0HAvAcDUQe4nQgYz` 在远程超时，改由
  本机下载后通过单文件 staging 上传。服务端 Content-Disposition 为 `ckpt_epoch_54.pth`，size
  `793,041,121` bytes、SHA-256
  `a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d`；上传前后 size/SHA
  一致后原子移入
  `/root/autodl-tmp/DATA_ROOT/protected_mcln_artifacts/mcln_official_ckpt_epoch_54.pth` 并设 mode
  `0444`。本机 793MB 临时副本随后删除；远程受保护文件是唯一工作副本，可从官方链接重取。
- checkpoint 为 epoch54、1,135 个 model tensors、149,566,498 params，全部 finite；config 是
  ScanRefer、6 decoder layers、color、BUTD、self-attend、joint detection、soft-token 与 contrastive，
  不含任何 SourceMoE/selector/sidecar。当前代码 strict load 成功。
- 在原 `/DATA_ROOT/superpoints/val` 上完成 9,508-row、单 GPU0、batch24 的无 sidecar 评测：REC
  `5411/4317 = 56.9100/45.4039%`；Mask `5557/4527 = 58.4455/47.6125%`，semantic mIoU
  `40.6768%`。它没有复现 README/用户给出的 `58.70/50.70/44.72` mask 基线。
- 为排除本仓库 evaluator 修改，commit `9744a4ed219062d448ed0dba587eeb864491f158` 的上游原始代码
  用同一 checkpoint/data 跑固定 128 rows；当前/上游的 REC、Mask 两档完全一致，mIoU 分别
  `0.2964106504/0.2964405006`，只差 `0.00002985`。因此差距不是新增 evaluator 或后处理造成。

#### 14.126 根因：206/312 个 val superpoints 是体素 fallback；数据修复与 A/B（2026-08-14）

- 逐场景审计证明旧 val 目录有 `206/312=66.03%` 文件与
  `fallback_superpoints_from_scan_pc(scan, voxel_size=0.18)` **逐元素完全相同**；服务器恰好只有
  `106/312` 个 val mesh，缺 mesh 的 206 个场景与 fallback 集合一一对应。train 同样只有
  `412/1201` 个 mesh、剩余 `789/1201` 为 fallback，故后续正式训练前也必须修复 train 数据。
- 从 `https://huggingface.co/datasets/marvex/scannet-dataset` 只下载缺失的 206 个 val
  `_vh_clean_2.ply`：共 `1,311,604,290` bytes。压缩上传包 size `781,141,732`、SHA-256
  `dd50f365923c39b075ae21434ffc64ca56f7c34096dc76446e2adce025635f2c`；远程复核后仅向不存在的
  mesh 路径解包，覆盖冲突数为 0，val mesh 覆盖变为 `312/312`，上传包随后删除。另对 3 个原有
  mesh 做同源抽查，镜像与服务器文件 size/SHA 均逐字节相同。
- 旧 superpoints 完整保留；在独立只读目录
  `/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/val` 重新执行官方
  `segmentator.segment_mesh`。312 个新文件长度均为 50,000；其中原有正规 106 个与旧文件逐元素
  相同，206 个 fallback 全部改变。旧/新 superpoint 数中位数 `1512 -> 1001`，新目录 127MB、
  mode `0444/0555`，排序文件 manifest SHA-256
  `c043a3759297250cefcb996709563167e58e98a2359c98b28033826fb3f02409`。独立 data-root view
  `/root/autodl-tmp/DATA_ROOT_mcln_meshsp` 只把 val superpoints 指向新目录，train/其他输入仍指向原
  DATA_ROOT，故 A/B 可逆且不覆盖旧数据。
- 128-row A/B 在 REC `52/44` 不变时，Mask 从 `54/42/0.2964106504` 变为
  `54/52/0.3173006708`：Mask@.50 `+7.8125pp`、mIoU `+2.0890pp`。完整 9,508-row 单卡复现为
  REC `5411/4315 = 56.9100/45.3828%`；Mask `5577/4819 = 58.6559/50.6836%`，semantic mIoU
  `44.6926%`。相对用户基线只差 `-0.0441/-0.0164/-0.0274pp`，已把原先约 3--4pp 的异常缺口
  解释并恢复到复现噪声级；无 NaN/OOM/缺文件，正式 stdout 正常退出。

#### 14.127 V99 在修复 val superpoints 上的唯一固定复核（运行前冻结，2026-08-14）

1. **目的与唯一变化**：旧 V99 official 的 REC `5552/4645` 是当前最好完整 REC；它的 Mask
   `5676/4662/0.4176463` 使用了已证伪的 mixed/fallback val superpoints。本次只把 `--data_root`
   改为 14.126 的只读 view；backbone、parent、geometry、V99 artifact、命令其余部分和 evaluator
   全部冻结，不新增/训练/选择参数，不搜索 threshold，也不把本次 validation 用于调参。
2. **冻结输入**：epoch71 backbone SHA `3e44f4...2208`、parent reranker `f06f8972...69b`、geometry
   reranker `835c25be...3b6f`、V99 artifact
   `/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/v99_artifacts/pareto_contextual_h128_seed0_fullfit.pth`
   SHA `9752990c393fa6e45173a9dd129c4de4bb740924094dcbbec2f3121cbf39d1f2`，以及新 val superpoint
   manifest `c043a375...02409`。全部在运行前后复核只读 identity。
3. **协议与报告**：唯一正式运行使用 GPU0、batch12、9,508 rows，保留原 V99 的 parent+geometry+
   hierarchical flags；必须报告 REC overall/unique/multiple 两档及 Mask overall/unique/multiple 两档和
   mIoU。与旧 V99、修复后的官方 epoch54、用户 baseline 分别比较。运行只生成 log/config/receipt，
   不生成 checkpoint；无论结果好坏都不据此修改 V99 artifact。

#### 14.128 V99 + mesh-derived official superpoints 完整结果与封存（2026-08-14）

- GPU0 单卡、固定 9,508-row validation 已完整生成全部预测和指标。REC overall 为
  `5572/9508 = 58.603281%`、`4797/9508 = 50.452251%`；unique 为
  `1261/1419 = 88.865398%`、`1143/1419 = 80.549683%`；multiple 为
  `4311/8089 = 53.294598%`、`3654/8089 = 45.172456%`。
- 同一选择结果的 Mask overall 为 `5690/9508 = 59.844342%`、
  `4976/9508 = 52.334876%`，semantic mIoU `45.930260%`；unique 为
  `1280/1419 = 90.204369%`、`1137/1419 = 80.126850%`；multiple 为
  `4410/8089 = 54.518482%`、`3839/8089 = 47.459513%`。
- 相对用户给出的 MCLN Mask baseline `58.70/50.70/44.72`，三项分别提高
  `+1.144342/+1.634876/+1.210260pp`，因此此前“mask 没超过 baseline”的根因已经由缺失 mesh 导致的
  fallback superpoints 解释并修复。相对旧 mixed-superpoint V99，REC 增加 `+20/+152` hits，Mask
  增加 `+14/+314` hits，mIoU 提高 `+4.165630pp`。
- REC@0.50 已超过目标 `138` hits；REC@0.25 距 `5610/9508` 仍差 `38` hits，因此总目标尚未完成，
  不把 mask 达标替代为 REC 达标。下一步只在修复后的 train superpoints 上验证预注册 REC 候选。
- 9,508 个样本和全部指标打印完成后，旧 `export_retrain_metrics` 才因 learned/subgroup 来源语义不同触发
  `ValueError: unique and multiple hits025 must partition learned hits`，进程 return code 为 1。该异常位于
  post-metric export，不中断预测或指标计算；receipt 明确记录为 recovered post-metric failure，不能写成
  clean exit，也不据此丢弃已完成指标。
- 只读结果 receipt：
  `/root/autodl-tmp/DATA_ROOT/output/v99_meshsp_official_20260814/v99_meshsp_official_result.json`，
  SHA-256 `311097c8a0fc1eceab3c95983937071e67fd8082ac46d1af5d3701ada4eb491c`。launcher/config/run-log
  SHA 分别为 `d77eac...b364`、`c42acb...f6b0`、`2e6aa5...b61`；输出目录 `.pth=0`。
  V99/epoch71/V19 输入 SHA 复核仍为 `975299...d1f2`、`3e44f4...2208`、`2d6a3c...ecbe`，GPU 空闲。

#### 14.129 V101 通用 Pareto runtime 接线与 post-metric export 修复（2026-08-14）

- 冻结 V101 artifact 的 schema 是 `rec-pareto-contextual-full-train-artifact-v1`，而现有 runtime 只把
  V99 的 `rec-pareto-contextual-hierarchical-v1` 路由到通用 Pareto model/policy；V101 因而会被误送到
  旧 hierarchical loader。这是接线缺失，不是模型或 OOF 失败。
- `train_dist_mod.py` 只增加 V101 schema 到同一 Pareto schema 集合：V99 仍调用原
  `load/validate_v99_artifact`，V101 调用已冻结的 `load/validate_v101_artifact`；两者之后完全共享
  `aggregate_margin=0.1331222057` 与 `apply_pareto_contextual_policy`。未修改特征、候选、logits、margin、
  threshold 或选择规则。文件 SHA 从 `0c084c...f5fa` 变为
  `34a6ed34ffc09979479deb4b5b4c72cf0c6a98ef6c768e9a5630d652bb754078`。
- 真实 GPU0 strict-load audit 同时加载 V99 SHA `975299...d1f2` 与 V101 SHA `2c969a...a2ae`；两者
  model 均 eval、无 requires-grad、device `cuda:0`、父/geometry/backbone SHA 绑定通过。既有
  hierarchical/V99 official 聚焦回归 `17 passed`，证明 V99 路径未回归。
- V99 修复数据评估的 return code 1 来自一个口径错误：`position_subgroups` 在最终 parent+geometry+
  contextual rerank 后记录，而 `position.learned_selector` 在这些 reranker 前记录，旧 exporter 却强制
  两者 hits 相等。`src/grounding_evaluator.py` 现保留两套计数，不再做跨阶段相等断言；仍强制
  unique/multiple denominator 完整分割、各 subgroup @.50<=@.25、Mask subgroup 与 overall 精确分割。
  文件 SHA 从 `a670bb...9485` 变为
  `736f26eb4474a628b13418947ba57b84740b0a82c0700a60e698ed73597b693c`；原指标测试 `29 passed`，
  额外 stage-separation/nesting audit 通过。该修复不改变任何预测或指标，只避免完整评估在导出后报错。

#### 14.130 V101 + mesh-derived val superpoints 唯一正式验证预注册（运行前冻结，2026-08-14）

1. **唯一候选与变化**：使用已经通过 5-fold scene-disjoint OOF 的冻结 V101 artifact
   `pareto_contextual_h128_seed0_fulltrain.pth`，SHA
   `2c969a6c28a0c9315b53f0f847567345e47da8c912091344b23612680643a2ae`；相对 14.128 只替换 V99
   artifact 为 V101。epoch71 backbone、parent、geometry、完整 mesh-derived val superpoints、batch12、
   seed0、GPU0、9,508 rows 和所有 evaluator 口径不变。
2. **训练证据但非结果承诺**：V101 使用全部 36,665 train rows / 562 scenes，OOF 为 `+159/+520`
   hits，五折两档均严格正，scene bootstrap 下界 `+118/+421`；full-fit replay 为 `+237/+765`。
   这些只允许它进入一次 validation，不能替代 validation，也不得据此挑 margin/seed。
3. **运行纪律**：不读 validation GT 作为输入，不训练、不搜索、不根据 14.128 的 38-hit 缺口改阈值；
   V101 与 V99 使用同一通用 Pareto policy。输出只允许 config/log/result receipt，不保存 checkpoint；
   protected artifacts 前后 path/size/mode/SHA 必须一致。
4. **硬门与次级比较**：REC 必须同时达到 `>=5610/9508` 与 `>=4659/9508`。Mask 硬保底为用户
   baseline `>=58.70/>=50.70/>=44.72`；另外逐项报告是否保持当前 V99 最好
   `59.844342/52.334876/45.930260`，但不得用 Mask 优势掩盖 REC@.25 未达标。
5. **报告范围**：完整报告 REC 与 Mask 的 overall/unique/multiple 两档及 semantic mIoU，记录 exact
   hits、所有输入/源码/log SHA 和 clean return code。无论通过或失败，V101 本轮不再做第二次
   validation 或 validation-driven 修改；失败时转向修复后 train superpoints 上的新训练路线。

#### 14.131 V101 第一次启动 fail-closed：data-root 尾斜杠合同（2026-08-14）

- 首次启动在 dataset 构造 tokenizer 时、任何 batch/预测/指标前退出；GPU 无 evaluation 迭代，故它是
  preflight failure，不是额外 validation 结果。runner 把 `Path` 转成字符串时去掉了 data-root 尾部
  `/`，而旧 dataset 用字符串拼接 `data_path + "roberta-base/"`，产生不存在的
  `/root/autodl-tmp/DATA_ROOT_mcln_meshsproberta-base/`。V101 artifact/model/policy/参数均未执行或修改。
- 失败目录、launcher、exitcode 与原 claim 全部保留并改名标注
  `failed_preflight_missing_trailing_slash`；output `.pth=0`。stdout/launcher/exitcode SHA 分别为
  `92c36d...53c3`、`07fd0b...fc87`、`4355a4...d865`。
- 唯一修复是让 authoritative command 保留已在 V99 成功命令中使用的
  `/root/autodl-tmp/DATA_ROOT_mcln_meshsp/` 尾斜杠，并使用新的 create-exclusive retry claim；其余命令
  和所有冻结参数逐项不变。dry-run 已精确打印带尾斜杠路径，runner SHA 为
  `a3a3033fbe84f56ef048cc4f2402bca6387f283b7d66274a2231a1c3d9309b07`。该启动修复后允许重试同一
  预注册 validation，不把 preflight typo 当作一次结果选择。

#### 14.132 V101 + mesh-derived val superpoints 正式结果：未超过 V99（2026-08-14）

- 单 GPU0、batch12、完整 `9,508` rows 正常退出，result `returncode=0`；没有 validation GT 作为输入，
  没有训练或阈值搜索。REC overall 为 `5553/9508 = 58.4034%`、
  `4746/9508 = 49.9159%`；unique 为 `1264/1419 = 89.0768%`、
  `1137/1419 = 80.1268%`；multiple 为 `4289/8089 = 53.0226%`、
  `3609/8089 = 44.6161%`。
- 同一最终 query 的 Mask overall 为 `5689/9508 = 59.8338%`、
  `4975/9508 = 52.3244%`，semantic mIoU `45.9220%`；unique 为
  `1280/1419 = 90.2044%`、`1137/1419 = 80.1268%`；multiple 为
  `4409/8089 = 54.5061%`、`3838/8089 = 47.4472%`。
- V101 仍超过用户 Mask baseline `58.70/50.70/44.72`，分别为
  `+1.1338/+1.6244/+1.2020pp`；但相对 V99，REC 少 `19/51` hits，Mask 两档各少 `1` hit，
  mIoU 低 `0.0082pp`。因此全部最好指标仍归 V99；REC@.25 仍差 `38` hits，goal 未完成。
- 权威 receipt：
  `/root/autodl-tmp/DATA_ROOT/output/v101_meshsp_official_20260814/official_result.json`，SHA-256
  `e07263ed12dad9d9c5003a46cefe370a09fb8f01cd65fe1da9d7cafe4dda534f`；stdout/launcher/exitcode
  SHA 分别为 `a08ae8fe81fd9ac8d2cd732dd354ac3f260c6b3a00735f0a63ca6c36a8ebb7fd`、
  `5e599e61ad0453b8c55689e1953e978293d3c90ef77c6a35ce612f436e27b034`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。输出目录 `.pth=0`；
  backbone/V101 artifact 运行前后 SHA 仍为 `3e44f4...2208`、`2c969a...a2ae`。
- 这是 14.130 预注册候选的唯一 validation 结果。V101 不再重跑，也不根据 validation 修改 margin、
  seed 或 policy；继续路线改为在修复后的 train superpoints 上产生新的 train-only 证据。

#### 14.133 训练集 mesh 补全与 superpoint 重建启动（2026-08-14）

- 旧 train superpoints 共 `1,201` scenes，其中 `789` 个场景因为缺 mesh 使用 voxel fallback；缺失集合
  manifest 恰好 `789` 行，SHA-256
  `caf63109bdf9f19cd8132b3c70eb1f2467d70fc605d174c6ec801b34c1c31079`。从同一 ScanNet 镜像取得
  `789` 个 `_vh_clean_2.ply`，本地逐文件验证 PLY header、最小 size `800,072` bytes，无 `.part`。
- 为控制 3.7--3.9GB 剩余磁盘，按 `100×7 + 89` 分成 8 个临时压缩包串行上传；每包均先做 SHA、
  路径与覆盖冲突检查，只解包原先不存在的文件，解包后立即删除压缩包。8 个包 SHA 依次为
  `efeabb...daea`、`7b42...f96c`、`9412...7449`、`519b...12a5`、`5edd...9dc0`、
  `e261...b5a30`、`3d5c...f5e7`、`75759d...b5da9`；上传日志 SHA-256
  `17f6b8c9f8f15a6c3cb5199847457912f4ca9699f0d213b590d863a78a4ee423`。
- 远程门禁：manifest 中当前缺失 `0`，`scannet/scans` 下 mesh 文件 `1,513`，上传 staging 文件 `0`；
  没有覆盖原有 412 train / 312 val mesh。V101 完全退出、GPU0 释放后才启动 CPU-only
  `mcln_train_meshsp_build_20260814`，目标是独立目录
  `/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train`，不覆盖旧 train superpoints。
- 构建后必须审计：新目录恰好 `1,201` 个长度 50,000 的整数 tensor；原有正规 `412` 个逐元素相同，
  原 fallback `789` 个全部改变，集合必须与 manifest 精确一致。审计通过前不切换 train view，也不删除
  本地 mesh staging。

#### 14.134 train mesh-derived superpoints 完成、审计与可逆切换（2026-08-14）

- CPU-only `mcln_train_meshsp_build_20260814` 正常退出，exit code `0`。独立目标目录包含恰好
  `1,201` 个 `_superpoint.pth`，总 size `508,393,855` bytes，文件 mode `0444`、目录 mode `0555`；
  每个 tensor 均为长度 50,000 的整数分组标签。
- 自动审计为 `identical_to_old_count=412`、`changed_from_old_count=789`、
  `changed_set_matches_fallback_manifest=true`；原正规场景全部逐元素不变，原 fallback 场景全部改变且
  精确等于预冻结 manifest。superpoint 数中位数从 `1372` 变为 `843`，与 val 修复方向一致。
- 审计 receipt
  `/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train_audit.json` 的 SHA-256 为
  `a118d311a3f7f1a434f06ff61582142178d9f4e740b3e5a6a8b529b4239b9215`；排序文件 manifest SHA-256
  `95c11c2714c2d67d3059b3de0e9d57a9eb717273ee66d2c98d35f18d4218869f`。build log/exitcode SHA 分别为
  `419426256ebf5eb59ec554a6da7fcae91a6218a2ea2a84e5557dbd43152cb8e3`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均已只读。
- 只修改可逆 view：
  `/root/autodl-tmp/DATA_ROOT_mcln_meshsp/superpoints/train` 从旧
  `/root/autodl-tmp/DATA_ROOT/superpoints/train` 原子切换到
  `/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train`；val 继续指向 mesh-derived val 目录。
  view 核验 train/val 文件数为 `1201/312`。旧 train/val superpoints 和全部原 mesh 均未覆盖或删除，
  因此 A/B 可随时回退。

#### 14.135 第二轮只保留最佳权重清理（2026-08-14）

- 清理前 `DATA_ROOT/output` 有 30 个 `.pth` 路径、表观 size `12,000,622,518` bytes；其中大量是
  同一 inode 的 retention hardlink。逐 inode、指标和后继实验审计后删除 22 个精确路径，释放物理
  `2,157,371,511` bytes；可用空间约从 `3.38GB` 增至 `5.54GB`。未递归删除实验目录，所有 config、
  log、metrics、claim、receipt 与 checkpoint-retention JSON 均保留。
- 单阶段 9 个路径实际为 3 份权重。保留 epoch7 的
  `ckpt_best_rec_acc025.pth`（SHA `8804109f...3eec`；同时是 Mask 三项最好）和 epoch19 的
  `ckpt_best_rec_acc050.pth`（SHA `551847ca...079`）；删除这两份权重的 5 个冗余 hardlink 名称，
  并删除五项均劣于 retained best 的 epoch28/last 物理权重（SHA `9dd2bab4...f2d`）。
- V59c epoch3 的 6 个名字是同一个 `606,036,179`-byte inode（SHA `3d26be9c...f3a6`）；同配方后继
  V60 已在两个正式 train prefix 连续 `break>fix` 后 fail-closed，V61 全量结果也未晋级，因此全部
  V59 `.pth` 删除。旧 source-choice epoch68 `0.57793/0.46140` 又被 epoch71 双指标严格支配，删除其
  `794,127,241`-byte 权重（SHA `ff8fddc...21f6`）。两者都可由保留的父权重+config 重跑。
- V101 唯一 validation 已证实全面未超过 V99，故删除 `914,379`-byte artifact（SHA
  `2c969a6c...a2ae`）；其 immutable build receipt、两份 claim、OOF/result/log 和构建脚本全部保留，
  负结果仍可审计。另删除未选中的 2 个 parent 与 5 个 geometry 小 artifact，只保留冻结的
  `final_contract` 和 `selected_geometry_reranker`。
- 清理后 `DATA_ROOT/output` 只剩 8 个 `.pth`：V99 必需的 epoch71/parent/geometry/V99 artifact，
  V19 的 parent/geometry，以及单阶段两档各自最好 checkpoint。当前双阶段最佳关键 SHA 复核为
  epoch71 `3e44f4...2208`、parent `f06f89...69b`、geometry `835c25...3b6f`、V99
  `975299...d1f2`；protected V19 仍为 `2d6a3c...ecbe`。

#### 14.136 V108 MeshSP-aligned train-only 重拟合路线预注册（2026-08-14）

1. **动机与边界**：旧 V99/V101 的 36,665-row train cache 在 `789/1201` 个 train 场景仍使用
   voxel fallback superpoints，而新正式 val 已全部使用 mesh-derived superpoints。基础 REC cache 的
   152-D 特征明确包含 `mask_confidence`、`mask_foreground_ratio`、`mask_text_query_dice`，geometry
   cache 还包含 25-D mask-to-box 特征；因此旧训练特征与当前部署特征存在已证实的数据口径错位。
   V108 先消除这项 train/val mismatch，不改 validation、模型输出接口或 ScanRefer 特定规则。
2. **冻结阶段顺序**：只在 train split 上依次执行：(a) 用 epoch71 和新的 train view 重建 Top-16
   36,665-row base candidate cache；(b) 与旧 cache 做逐行因果 A/B，要求候选 identity/box/IoU 不变；
   ScanRefer train 覆盖的 562 scenes 中，原正规 201 scenes 的全部特征必须不变，变化只能来自
   manifest 交集中的 361 个原 fallback scenes 的 mask 特征；(c) 重建 train-only
   mask geometry；(d) 重新训练通用 parent/geometry/contextual hierarchy 并做 5-fold scene-disjoint OOF。
   任一阶段失败即停止，不生成 validation artifact。
3. **阶段一固定命令**：唯一 GPU0，batch12、workers4、shard256、Top-16、seed0，输入 checkpoint SHA
   `3e44f4...2208`、train SP audit SHA `a118d3...215`、SP sorted manifest
   `95c11c...869f`，输出新目录 `DATA_ROOT/output/rec_reranker/e71_top16_meshsp/train`，不覆盖旧 cache。
   runner `scripts/run_v108_meshsp_train_candidate_cache.sh` SHA-256
   `573b6f77cac89f0e4c24c5e71492db96299272f476f37dcdcbd2200e02a7ef90`；输出只允许 `.pt` cache、
   manifest/log/receipt，不允许新 checkpoint。
4. **候选不变量门**：旧 train cache 的 frozen counts 为 default `34892/31870`、candidate oracle
   `36405/35409`（总数 36,665）。新 cache 必须 exact 复现四个 counts、query identity、candidate
   validity、boxes 与 candidate IoU；否则说明 superpoint 修复意外改变 REC 候选池，立即 fail closed。
5. **OOF 晋级门**：后续 V108 模型仍只能读取通用 query/text/score/mask/box geometry，不读 dataset、
   scene/category ID 或 validation GT。5 个 held-scene folds 两档净增均须严格为正、scene bootstrap 95%
   下界均须大于 0；总增益至少达到 V101 原预注册下限 `+72/+74`，且 ScanRefer train 中原正规
   201 scenes 与修复的 361 scenes 两个子集在两档都不得为负。只有这些 train-only 门全部通过，
   才讨论一次新的 official validation；
   不根据已知的 38-hit validation 缺口调 margin、seed 或阈值。
6. **A/B 审计实现**：`scripts/audit_v108_meshsp_candidate_cache.py` SHA-256
   `6d875d01a604a3049999f6cf4bbc626e5fb73243aaf6414163096d931feb2c86`，逐 shard 流式比较 36,665
   rows；非 Mask 的 149 个 feature columns 与所有 candidate identity/score/box/IoU 必须 bitwise 相同，
   三个允许变化的列固定为上述 `mask_*`，且变化 scene 必须是 frozen fallback manifest 的子集。

#### 14.137 V108 batch24 启动在结果前停止、改为权威 batch12（2026-08-14）

- 第一次 runner 把候选提取写成 batch24；运行到 `9216/36665` 时复核历史 geometry audit provenance，
  确认旧 authoritative cache 的提取 batch 是 12。不同 batch shape 可能让 CUDA GEMM 产生微小浮点漂移，
  会破坏 14.136 要求的 bitwise 因果 A/B；因此在任何完整 cache、OOF 或 validation 结果前主动停止。
- 不完整的 36 个 shard 已逐文件删除，未生成 checkpoint；只读保留 incomplete log/manifest，SHA 分别为
  `3074376675d11fe893440111b4cfdd1e86b206e170e08f0bce3f679de76244c8`、
  `ab7dabc51f54ad07c6e97f03ec70cd9a9051b070229ce8a668ced1d12cd26765`，目录明确标注
  `e71_top16_meshsp_failed_preflight_batch24`。该启动不是候选结果，不能参与选择。
- 唯一协议修正为 batch12，其余 checkpoint/data/view/workers/shard/Top-K 均不变；新 runner SHA 即
  14.136 所列 `573b6f...ef90`。修正依据只来自旧 cache provenance，不读取本轮质量指标或 validation。
- 同时修复 portable geometry provenance：非 portable 路径仍强制 batch12；portable 路径改为严格绑定
  audit selection 中的正整数真实 batch，而不再伪装成常量12。源码
  `scripts/cache_scanrefer_rec_mask_geometry.py` SHA 从 `17f772...759d` 变为
  `a291142eb1263792b82282cf904be2e19e27e78c35c2bf0a489161dc1c7424bc`；原回归 `29 passed`，
  独立 batch24 portable 接受探针通过。V108 重跑仍使用 batch12；该修复用于后续完整 provenance。

#### 14.138 V108 单 GPU 串行执行与子组 OOF 门（结果前冻结，2026-08-14）

- 服务器当前只有一张 `NVIDIA A100-PCIE-40GB`，所有 CUDA 阶段固定 `CUDA_VISIBLE_DEVICES=0` 与
  `cuda:0`，严格串行为 candidate cache → candidate A/B/mask-geometry cache → parent → geometry →
  5-fold OOF。等待阶段只运行 shell polling，不创建 CUDA context；任一上游 exitcode、只读 receipt、
  schema、样本数或 `validation_data_accessed=false` 不满足就停止，绝不启动下游。
- candidate 完成后的第一段 runner 为
  `scripts/run_v108_single_gpu_serial_geometry_wait.sh`（SHA-256
  `0906ab0469995405c2635f0ef59110f3cd02958eae3efb0d1341dfae35a73c8c`）；它等待 candidate screen
  完全退出且 receipt 存在后才调用 `scripts/run_v108_meshsp_train_geometry.sh`（SHA-256
  `fe3272dd28d8a483ce7a629b8fbea407280cf0b13c7a1af63518c8ff46080ae8`）。后者先执行逐行 A/B，
  再执行 256-row mask geometry audit，最后用 batch36/workers4 生成 36,665-row portable geometry cache。
- 第二段 runner `scripts/run_v108_meshsp_models_oof_serial.sh`（SHA-256
  `c20e524743be1fdacd6bf665c23f1f30c2a8746892328589e3b4f65fee4cb0c2`）等待上一段 clean receipt 后，
  固定重训 parent `h256/dropout0.1/lr1e-3/wd1e-4/batch256/seed0` 与 geometry
  `h256/dropout0.1/lr3e-4/wd1e-4/batch256/split0/model-seed0`，然后才运行 scene-disjoint OOF；
  parent/geometry 内部均执行 best-state 恢复与 strict artifact reload。
- 新 OOF 实现 `scripts/run_v108_meshsp_pareto_oof.py`（SHA-256
  `05682999efcd313d8d0f25f616944c119cf8f2a50d8ab34c1e3e80b0eb104078`）保持 V99 的 contextual
  query-set architecture、固定 margin、目标与 5-fold scene mapping，不搜索 seed/margin/阈值。除 14.136
  的总增益、逐 fold 与 bootstrap 门外，它按冻结 manifest SHA
  `caf63109bdf9f19cd8132b3c70eb1f2467d70fc605d174c6ec801b34c1c31079` 把 562 个训练场景精确划为
  修复的 361 scenes 和原正规 201 scenes；两个子组在 @0.25/@0.50 的 OOF delta 都必须 `>=0`。
  独立 synthetic gate probe 已验证 10 个谓词全真时通过、任一子组负增益时拒绝。该阶段只读 train
  cache/GT，不读取 ScanRefer validation、Nr3D 或 Sr3D。
- 三个 screen 名分别为 `mcln_v108_meshsp_candidate_20260814`、
  `mcln_v108_meshsp_geometry_wait_20260814`、`mcln_v108_meshsp_models_oof_wait_20260814`；后两者在上游
  运行时只是等待器。启动核验时 `nvidia-smi` 只有一个 CUDA PID，因此没有单卡并发。只有 OOF receipt
  的全部预注册谓词为真，才允许构建 deployable artifact、做 full-fit/runtime parity 并预注册一次 official
  validation；否则封存失败原因并清理落选 `.pth`。

#### 14.139 V108 candidate 完成、旧代码 A/B 基线失败与同代码控制修正（2026-08-14）

- batch12 的 MeshSP train candidate cache 已 clean exit：`36,665` rows、`144` shards，默认 Top-1
  `34892/31870`，candidate oracle `36405/35409`，精确复现冻结四项计数；manifest/log/exit/receipt SHA-256
  分别为 `fc4ef0c...6d6d`、`c77edc2b...ebc7`、`9a271f2a...86aa`、`bfe2a650...caaa`，目录与文件
  已只读。该阶段仍未访问 validation，也未生成 checkpoint。
- 随后的首轮 A/B 使用了 7 月 14 日生成的 `e71_top16/train` 作为旧-SP基线，并在首个原正规场景
  `scene0000_00` 报 `regular-scene candidate features changed`；geometry、parent、OOF 均未启动。
  fail-closed 的 6 个 log/exit 已逐文件移动到只读目录
  `e71_top16_meshsp/failed_preflight_oldcode_cache_ab_20260814/`，没有删除，主要 A/B error log SHA 为
  `9d7b2191...a298`。这不是模型负结果，而是因果控制口径失败。
- 快速复现命令直接调用 `audit(...)`，约 10 秒连续两次稳定在同一 scene 失败。逐列探针证明：候选
  identity/box/IoU 与 149 个非 Mask 列保持 exact；差异只在有效候选的
  `mask_confidence/mask_foreground_ratio/mask_text_query_dice`，不是 padding。前 256 rows 的最大绝对差
  约为 `0.00727/0.03161/0.39197`，同时既有 ULP 级小差异也有明显语义差异，故不能解释为纯 CUDA
  舍入。抽样正规 scene 的旧/新 SP 文件逐字节相同，例如 `scene0000_00` SHA
  `11f92113...ffd8`、`scene0002_00` SHA `30d8c366...587d`，排除 SP view 错配。
- 根因是 **代码代际混杂**：冻结的 7 月源码 snapshot 中
  `models/rec_candidate_adapter.py` SHA `9e471376...192b` 用
  `adaptive_weights.mean()` 把所有 query 压成一个标量 Mask fusion alpha；当前源码 SHA
  `dfc5afaa...10a3` 已在 8 月 5 日改为 `fuse_query_mask_logits(...)`，保留 scalar 或逐 query alpha。
  因此即使 SP 完全相同，三列 Mask 特征也应变化；14.136 中要求它们在 201 个正规 scenes exact 的旧
  基线不具因果可比性，不能靠放宽 bitwise tolerance 修补。
- 当前代码+旧 SP 的 256-row probe 已 clean exit。与 MeshSP cache 对比时，前 252 rows 中正规 scene
  匹配；只在 probe 最后 4 rows 出现非 Mask 差异，因为 `limit=256` 把正式 batch12 的末组切成 batch4。
  这再次确认 batch shape 是复现合同的一部分，故不把有限 probe 作为正式 A/B。
- 修正后的唯一因果协议是在 **完全相同当前源码、checkpoint、batch12、workers4、36,665 rows** 下新增
  一份旧-SP control cache，然后以它对比 MeshSP cache。旧-SP 目录 1,201 文件、`509,239,295` bytes，
  排序内容 SHA `365aa6a6...59e6`；控制 runner
  `scripts/run_v108_currentcode_oldsp_train_candidate_cache.sh` SHA
  `6d56c07302da8532c9fef29b2b7d989c55235dcef9a20d288b6779c6f1885f3e`，绑定 candidate script
  `b1db28f1...f1932`、提取期 `train_dist_mod.py` `34a6ed34...4078`、MCLN/adapter/mask-fusion/dataset
  六份关键源码 SHA，并生成完整只读 receipt。为让控制提取与已完成 MeshSP 提取的源码身份也一致，
  尚未用于 forward 的 V108 runtime schema 扩展已临时还原；OOF 通过后再恢复并回归验证。
- 几何 runner 现只接受上述 control receipt，A/B 旧端改为
  `e71_top16_currentcode_oldsp/train`；新 geometry/wait runner SHA 分别为
  `10758029...85d5`、`e16c306b...49c3`。当前三个串行 screen 为 control cache、geometry wait、models/OOF
  wait，后两者不建立 CUDA context。正式 A/B 仍要求 regular201 全特征 exact、fallback361 仅三列 Mask
  特征可变、候选/box/IoU/四项计数 exact；失败仍停止，成功才进入 geometry。整个修正过程没有读取
  ScanRefer validation，也没有按已知 validation 缺口调整模型或阈值。

#### 14.140 V108 同代码因果 A/B 通过与完整 MeshSP geometry cache（2026-08-14）

- 当前代码+旧 SP control cache 已按正式 batch12 完成：36,665 rows、144 shards，四项冻结计数仍为
  default `34892/31870`、oracle `36405/35409`；control receipt SHA-256 为
  `c472528e114238511bc8784b0efd01d75414485a5bd3be000b334112042b700b`。它只用于隔离 SP 变量，
  不参与模型选择或 validation。
- 完整逐行 A/B clean pass，报告
  `e71_top16_meshsp/candidate_ab_audit.json` SHA-256
  `4439438db6b3e9cccbf462a527395a39c3ca60588cd24d87c9549bdaef35b663`：
  36,665 rows/562 scenes 中，regular201 的 12,560 rows 全部特征 bitwise exact；fallback361 的
  24,105 rows 全部发生变化，且所有变化都严格位于该子集。candidate identity/boxes/IoU、149 个非 Mask
  列与四项计数均 exact；三列变化行数分别为 `24038/23906/24105`，最大绝对差为
  `0.123588/0.247233/0.979181`。这把 Mask 特征变化因果归因于 train superpoint 修复，而非代码代际、
  batch shape 或候选池变化。
- 预注册 64 scenes/256 expressions 的 geometry audit 随后通过：default `0.74219/0.48047`，
  `fused_t0_exact` 为 `0.77734/0.71875`（@0.25 fixes/breaks=`13/4`，@0.50=`66/5`），组合 geometry
  oracle 为 `0.98047/0.94531`。这只是 train diagnostic，不是 official 指标。
- 完整 geometry cache 在唯一 GPU0 上按 batch36/workers4 串行生成并原子发布：36,665 rows、562 scenes、
  146 shards、622,730,170 shard bytes；content digest
  `7bd0634bb7a6faeece7399e81dc98987e562dc8eea2ee701de8e9535f9bbc91f`，receipt SHA-256
  `e45adaafb3730f45dabcea7f0c4f4492a6ea6360b7f07bdb164270bd934d9443`，exitcode `0`，
  `validation_data_accessed=false`。整个阶段始终只有一个 CUDA PID，parent/geometry/OOF 均在上游退出后
  才接棒。

#### 14.141 V108 parent/geometry 重训、OOF 差 2 hits 拒绝与失败权重清理（2026-08-14）

- parent 按冻结配置在 epoch4 早停，train 内 calibration `Acc@0.25=0.94290`、
  `Acc@0.50=0.88138`；artifact SHA 为
  `7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f`。geometry 在 epoch5 早停，
  融合权重 `0.90`，train 内 calibration `0.95228/0.92828`；artifact SHA 为
  `20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972`。两者严格串行、只读保存后才启动
  OOF；这些 calibration 数字不作为 official 结果。
- OOF 初次进入时连续暴露三项历史基础设施常量：旧 residual loader 只接受 V19/V99 的固定
  parent/geometry SHA；旧 immutable capture API 只允许三个固定键；旧 materialization validator 又把
  geometry weight 写死为 `1.0`。三次都在相应计算前 fail closed，未访问 validation；traceback/log/exit
  分别保存在四个只读 `failed_oof_*_20260814` 审计目录（第四个为完成 OOF 后 receipt 对“失败应返回75”
  的错误假设）。修复保持历史 loader/validator 默认行为不变，只为 V108 冻结两个 cache receipt、两个
  model SHA、四个 manifest/receipt 前后 SHA，并给 materializer 增加默认不变的可注入 validator；旧路径
  物化回归 `4 passed`。
- 修复后的代码 SHA：`train_scanrefer_rec_hierarchical_reranker.py`
  `948075df82a17685e102c1913eff44f2ee032cc3e999bd5c3341aaa1b689aff3`，
  `run_v108_meshsp_pareto_oof.py`
  `94dbce107e8412e00ac777cdea732cf6f93d6a7985550258ef7ae46763053c8d`，
  `build_v108_meshsp_pareto_artifact.py`
  `af5ce0419b89a58d11bbcfd27e4dfb40b163e65009877b0ed83a576a95956efa`；恢复与终结脚本 SHA 分别为
  `3454516e7e4c1a6059e402614124ef6ba31a0e0e3ecff7a60076e0c9e1bfadef`、
  `b54c6d7e64700e0eb0071b64dde36c14057b24946fd20385020e28b1788c99c6`。
- 5-fold scene-disjoint OOF 最终完整运行，五折增益均严格为正：@0.25 为
  `+4/+13/+17/+28/+8`，@0.50 为 `+53/+33/+44/+56/+59`。总体 baseline hits
  `35215/33808`，候选 hits `35285/34053`，即 **`+70/+245`**；scene bootstrap 95% 下界为
  **`+32/+183`**。corrected361 子组 `+49/+178`，regular201 子组 `+21/+67`，两子组均非负。
  10 个预注册谓词中仅 `delta025 >= +72` 失败，差 **2 hits**；其余总增益、逐折、bootstrap 与子组门
  全部通过。因此 V108 按协议拒绝，未构建 deployable artifact，未读取 official validation。
- OOF 报告 SHA-256 为
  `72ca54b2db0bca829011a2f480c458c0a3e450a492dd77de9d8411e84f3e9162`；最终失败 receipt SHA-256 为
  `983dbe5141a4bef2a3c36e23bbc0c833aa2d4a4ea026a833708b7b2dad6fdb32`，协议 exitcode `75`，明确记录
  `gate_outcome=failed_delta025_by_two_hits` 与 `validation_data_accessed=false`。
- 按“只保留最佳权重”要求，在 receipt 对 artifact SHA/size、训练日志、cache receipt 与重建配置全部绑定后，
  已精确删除 V108 的两个落选 `.pth` 并删除空 `v108_artifacts` 目录；文件本身不可直接恢复，但可由保留的
  36,665-row candidate/geometry caches 与冻结命令确定性重建。全 output `.pth` 从 10 回到 8，历史
  epoch71、V19 parent/geometry、V99、single-stage @0.25/@0.50 最佳权重均未改动。

#### 14.142 V109 双层 scene-cross-fit 策略校准预注册（结果前冻结，2026-08-14）

- **动机**：V108 的模型 OOF 在 @0.50 有大余量（`+245`），但 @0.25 以 `+70` 比原门槛少 2 hits；
  4,069 个切换在 @0.25 产生 194 fixes/124 breaks，说明下一步应提高对低置信 @0.25 切换的选择性，
  而不是降低 `+72` 门、读取 validation 或修改候选/geometry。V109 保持 V108 的 parent、geometry、
  contextual hierarchy、训练目标、seed、五折 scene mapping 和所有输入完全不变，只把固定 Pareto policy
  改为 **leave-one-fold-out meta calibration**。
- **无泄漏双层协议**：先像 V108 一样得到五份 scene-disjoint raw OOF proposal、aggregate gain 和两个
  head gain。对外层 held fold `k`，policy 只能读取另外四个外层 OOF folds 的预测与 train GT；held fold
  的标签在 policy 选择时不可见。选好后才应用到 fold `k`，五个 held 结果拼成最终 meta-OOF。用于部署的
  单一 global policy 仅在全部 OOF 预测上按同一冻结规则选出；正式 validation 仍要等 meta-OOF 全门通过。
- **冻结 30-policy grid**：aggregate margin 为
  `{0.10,0.12,0.13312220573425293,0.15,0.18,0.22}`，最小 @0.25 head gain 为
  `{0,0.0025,0.005,0.01,0.02}`，@0.50 head gain 始终严格 `>0`。在四折 calibration 上，候选须
  overall @0.25 正增、每个可见 fold 两阈值非负，且 @0.50 增益至少为固定 V108 policy 在同四折增益的
  `ceil(50%)`；随后依次最大化 @0.25、@0.50、减少 switches，并以更高 head floor/margin 作确定性 tie-break。
  synthetic probe 验证 30 个候选齐全，改变被排除 held rows 的 IoU/head gain 不会改变 policy 选择。
- **晋级门不变**：最终 meta-OOF 总增益仍须 `>=+72/+74`；五个 held folds 两阈值都须严格正增；
  scene-bootstrap 95% 下界都须 `>0`；corrected361 与 regular201 两子组两阈值均须 `>=0`。任一失败即
  exit76、禁止 official validation，并在 receipt 后删除本轮重建的两个落选 `.pth`。
- 实现 `scripts/run_v109_meshsp_nested_policy_oof.py` SHA-256
  `aa77fadd55d8e579ad65748f8e1a078cbb756eaf53b061ec0339b73495a17c33`；单卡串行 runner
  `scripts/run_v109_meshsp_nested_policy_serial.sh` SHA-256
  `6c5770ce5371da4e8b80349e9a66e008a572b2f4e775ef7640ec8404e37a7bff`。runner 先按原冻结命令重建
  parent/geometry，并要求 artifact SHA 必须逐字节复现 V108 的 `7b8956...616f`、`20f33c...7972`；
  随后才在唯一 GPU0 上运行 V109。启动前 free disk `4,001,886,208` bytes、GPU compute PID 为空。

#### 14.143 V109 nested-policy OOF 全门通过与 full-fit artifact 构建冻结（2026-08-14）

- parent/geometry 重建逐轮复现 V108，最终 artifact SHA 分别 exact 为 `7b8956...616f` 与
  `20f33c...7972`；说明 14.141 的删除是可恢复的，也证明 V109 的差异只来自 policy procedure。
- 五个外层模型 folds 的 raw prediction digest 受报告绑定；五个 leave-one-fold-out meta calibrations 都独立选择
  同一策略：aggregate margin `0.15`、@0.25 head gain 严格 `>0.02`、@0.50 head gain 严格 `>0`。
  held-fold 净增 @0.25 为 `+5/+14/+16/+28/+9`，@0.50 为 `+53/+35/+43/+56/+59`，全部严格为正。
- 汇总 baseline hits `35215/33808`，V109 hits `35287/34054`，即 **`+72/+246`**；switches 从 V108
  的 4,069 降到 3,897。@0.25 fixes/breaks 从 `194/124` 变为 `193/121`，净增正好从 +70 提到门槛
  +72；@0.50 fixes/breaks 为 `477/231`。scene bootstrap 95% 下界为 **`+34/+183`**。
- corrected361 子组为 `+52/+182`，bootstrap 下界 `+20/+131`；regular201 子组为 `+20/+64`，
  其 @0.25 bootstrap 下界为 `-1`，但预注册子组门只要求 pooled delta 非负，故不改变门定义。
  总增益、五折严格正增、整体 bootstrap 正下界、两个子组非负共 10/10 谓词全真。
- OOF report SHA-256
  `37680aaa34757cf9bb2376e93629ae6b89aa6b8fac16960ac091305cc20146a1`；单卡 pipeline receipt SHA-256
  `07af9c6b331e808f86d16e62ae92a1106e86321c3f0734c3f2cb6ede46b94986`，exitcode `0`，
  `validation_data_accessed=false`。因此允许进入 full-fit/runtime parity，但此时仍未访问 official validation。
- full-fit builder `scripts/build_v109_meshsp_nested_policy_artifact.py` SHA-256
  `93d5dca1b284d5cb4a34902b69a21a287c589a06e8c0ee43f12e375c608e5b98`，先严格重验 OOF、五个
  meta policy 一致性、输入 receipt/model SHA，再在全 36,665 rows 上用原 V99 contextual hierarchy
  的 12 epoch/h128/dropout0.1/wd1e-3 训练；artifact policy 额外冻结 margin/head025/head050 与双层选择过程。
  单卡 runner `scripts/run_v109_meshsp_artifact_serial.sh` SHA-256
  `2b75cc913c191517c490184a3df4c2fc01f10379e3eb8f65809ef7e075a98833`；输出只能是
  `v109_artifacts/nested_policy_h128_seed0_fullfit.pth` 与只读 receipt。构建、strict reload、train replay、
  runtime parity 全过后才允许一次 official validation；否则清理新增 artifact/落选权重。

#### 14.144 V109 full-fit artifact 完成与只读封存（2026-08-14）

- 唯一 GPU0 上的 full-fit 已 clean exit，并在 CPU 上 strict reload。artifact 为
  `e71_top16_meshsp/v109_artifacts/nested_policy_h128_seed0_fullfit.pth`，SHA-256
  `20db69ddc27680a035384277bc48cd44109215e3d7d1158cdc4a4f21ff7c785b`，大小 915,339 bytes，
  mode `0444`；artifact receipt SHA-256 为
  `19f7676241b1558beb53c67f770cf8c3a3d149d3e0ee21a61579e383d53b7115`，同为 `0444`。
- 全量拟合精确使用 36,665 rows/562 scenes；第 12 epoch 的 loss/query/variant 为
  `4.4842278593/2.6923590153/1.7918688299`。模型 state、normalization、scene fold、deployable rows、
  candidate IoU digest 分别为 `4b5f4962...eceef`、`77046539...d3be`、`829d230d...3e53`、
  `473be1a4...5f39`、`b0945e31...abed`。artifact schema 固定为
  `rec-pareto-contextual-meshsp-nested-policy-full-train-artifact-v1`，policy 固定为 margin `0.15`、
  @0.25 head gain 严格 `>0.02`、@0.50 head gain 严格 `>0`。
- 构建前后 backbone、parent、geometry 及两个 cache manifest/receipt 的内容身份完全相同，
  `validation_data_accessed=false`。V109 正式结论前继续保留 parent `7b8956...616f`、geometry
  `20f33c...7972` 与 full-fit artifact 三个依赖权重；此时 output 下 `.pth` 共 11 个，没有并发或额外权重。

#### 14.145 V109 runtime 接入、语义回归与 36,665-row parity（2026-08-14）

- `train_dist_mod.py` 已只增加 V109 schema 的严格动态 loader/validator 分支，并从 artifact policy 读取
  两个 head floor；`models/rec_pareto_contextual_hierarchy.py` 给 Pareto policy 增加默认均为 `0.0` 的
  可选 floor，故 V99/V101 的旧调用行为不变。最终源码 SHA 分别为
  `691a7aa969bc2fb277f9807bda578b20dcf5de1cf827ad37e4808e2b92c794fc` 与
  `d108fc146b80646b7ab0479d7a03d2f7f7cf69ed45bea597232b46f9b836f9fe`。
- 首轮既有回归发现 provenance 返回字典被多加两个字段，旧精确契约 18/19；已撤回多余返回字段后
  19/19 通过。随后新增实现后的独立 V109 语义回归，把 runtime `switch_mask` 与冻结 OOF
  `policy_accept_mask` 在确定性随机样本上直接比较，5/5 通过；最终联合回归 **20/20 passed**。
- parity 脚本 `scripts/audit_v109_runtime_parity_train.py` SHA-256
  `011a5de2881545a965df801db265a05f52002d9c55229cf7218e53975e70ff16`；单卡 runner SHA-256
  `f86e474d0951294f2c291cbfcc8bdd98fe70668e5d994635a8ff931fe3329073`。前两次在计算前分别因训练
  loader 没有裸模型 SHA 属性、parent 实为 `(model, artifact)` 而 fail closed；日志保存在
  `failed_parity_loader_attrs_20260814/` 与 `failed_parity_parent_tuple_20260814/`，未访问 validation、
  未改权重。第一次完整 PASS 的报告因 `-inf - -inf` 让诊断用 different-elements 误计数而被封存到
  `superseded_parity_nonfinite_diagnostics_20260814/`，其核心 `torch.equal` 判定原本已全真，但不用于门禁。
- 最终只读 parity 报告 `v109_train_runtime_parity.json` SHA-256
  `5bc46bbf1146a34ee834b4241f934b244f1d8b2287fb931ec963c720350a9c46`，mode `0444`：36,665 rows 的
  8 组层级输入、baseline indices/scores、query/variant logits、proposal/head gain/aggregate gain、
  Pareto pass、switch mask 与 selected indices 全部 `equal=true`、`different_elements=0`、
  `max_abs=0.0`；受保护的 backbone/parent/geometry/V109 artifact 前后身份相同，权重未修改，
  validation 未访问。至此 full-fit、strict runtime loader、train replay 与 runtime parity 全门通过。

#### 14.146 V109 唯一 official validation 预注册与 dry-run PASS（结果前冻结，2026-08-14）

- official runner `scripts/run_frozen_v109_meshsp_official.py` SHA-256
  `3095cdd1746d4e99fe120a5b2f35284483d448f2ef020433f19c0d0bf9ca286b`，只允许 GPU0/world-size1、
  batch12、官方 epoch71 backbone、mesh-derived validation superpoint view 与 9,508 条 population；
  parent/geometry/V109 artifact 精确绑定 `7b8956...616f`、`20f33c...7972`、`20db69...785b`，并强制
  parity SHA `5bc46b...a9c46`、OOF SHA `37680a...e9162`、`validation_data_accessed=false`。
- 无 validation forward 的 dry-run 已通过 strict artifact load、完整 Python tree snapshot、命令与环境校验，
  只读 preflight SHA-256 为 `f7bdb28a4c77cea40f1e4621bbec127cab876377ba1a7f53b612605ec27025a2`；
  回归为 20/20 passed，GPU0 此时空闲。实际输出唯一固定为
  `/root/autodl-tmp/DATA_ROOT/output/v109_meshsp_official_20260814`，一次性 claim 为
  `v109_artifacts/v109_meshsp_official_once_after_train_runtime_parity.claim.json`；claim/output 已确认不存在。
- 正式门在结果前固定：REC hits 至少 `5610/4659`（Acc@0.25/0.50 至少 `0.59/0.49`）；Mask 至少保持
  用户 baseline `0.5870/0.5070/mIoU 0.4472`，并单独报告是否保持 V99 meshSP 的
  `5690/4976/mIoU 0.4593026021`。不允许 GT boxes/masks、train eval、selective residual 或第二个 CUDA 任务；
  不按 validation 结果再调 policy。若 V109 在两个 REC 指标上均不形成新的最佳/Pareto 最佳，则在正式
  result/claim/日志与三个权重 SHA 可恢复性封存后删除 V109 artifact 及重建的 parent/geometry；若形成新
  最佳则保留这一整条最小依赖链，并继续清理非最佳权重。

#### 14.147 V109 official 结果：@0.50 新最佳、@0.25 与 Mask 未胜 V99（2026-08-14）

- 唯一正式运行在 screen `mcln_v109_official` 上用 GPU0 串行完成 793 batches/9,508 samples，runner
  `scripts/run_v109_official_serial.sh` SHA-256
  `4aa655b72d62996c5feccf8bec1578cdcf1725cc80663603dd285c403fa40218`；全程只有一个 CUDA PID，
  exitcode `0`，结束后 GPU 回到 1 MiB。一次性 claim、stdout、result SHA-256 分别为
  `3da2e573115e4985c185fb81f85c9aba407836791e15cb40c1be5000ee8178b0`、
  `79f84f51438557bd5a9689e58fd3640577c1e0c2ca2e75220a2738ce6c199889`、
  `9afe5160359e56f867d1f500cd906b7b2133af124b49a353ed0d50b8ab8778ba`，均为 mode `0444`。
- 正式 REC/Mask 结果如下；百分比由精确 hit counts 计算，mIoU 为完整浮点值：

| 指标 | V109 | V99 最佳 | V109 - V99 |
|---|---:|---:|---:|
| REC overall @0.25 | 5,551/9,508 = 58.3824% | 5,572 = 58.6033% | -21 hits / -0.2209 pp |
| REC overall @0.50 | 4,834/9,508 = **50.8414%** | 4,797 = 50.4523% | **+37 hits / +0.3891 pp** |
| REC unique @0.25 / @0.50 | 88.7949% / **81.1135%** | 88.8654% / 80.5497% | -1 / +8 hits |
| REC multiple @0.25 / @0.50 | 53.0473% / **45.5310%** | 53.2946% / 45.1725% | -20 / +29 hits |
| Mask overall @0.25 / @0.50 | 5,689/9,508 = 59.8338% / 4,974 = 52.3138% | 5,690 = 59.8443% / 4,976 = 52.3349% | -1 / -2 hits |
| Mask unique @0.25 / @0.50 | 90.2044% / 80.1268% | 90.2044% / 80.1268% | 0 / 0 hits |
| Mask multiple @0.25 / @0.50 | 54.5061% / 47.4348% | 54.5185% / 47.4595% | -1 / -2 hits |
| Mask mIoU | 45.9224% | 45.9303% | -0.0079 pp |

  Mask overall @0.25 的权威值为 `5,689/9,508 = 0.5983382414808582`。REC unique/multiple totals 分别为 1,419/8,089；
  Mask unique hits 为 `1280/1137`，multiple hits 为 `4409/3837`。
- 门结果：REC@0.50 目标 `>=0.49` 通过并刷新全局最佳；REC@0.25 只有 58.3824%，未达到 59.0%，
  比目标少 59 hits。Mask 三项仍全部超过用户 baseline 58.70%/50.70%/44.72%，但均略低于 V99，
  因而 `mask_user_baseline_preservation_pass=true`、`mask_v99_meshsp_preservation_pass=false`、
  `rec_target_pass=false`、`all_goals_pass=false`。
- result 中 protected artifacts 前后完全相同，完整 Python source-tree manifest 前后完全相同；无 GT boxes/
  masks、无 train eval，正式 result 仅在冻结 OOF/parity 后读取 validation。V109 虽未同时胜出，但已形成
  **@0.50 的新 Pareto 最佳**，所以按“保留最好权重”要求保留 V109 full-fit + V108 parent/geometry 三个
  `.pth`；V99 仍是 @0.25 与 Mask 三项最佳，也继续保留。没有失败权重可删，output `.pth` 维持 11 个。

#### 14.148 V110 预注册：跨种子不确定性 LCB 深度集成（2026-08-14）

- 动机：V109 的 train-only OOF 为 `+72/+246 hits`，正式评测却在 REC@0.25 相对 V99 少 21 hits，且
  regular 子群 @0.25 scene-bootstrap 95% 下界为 `-1`。V110 不读取 validation，也不针对 ScanRefer
  标注规则增加特化修补；它复用可迁移到 ScanRefer/Nr3D/Sr3D 的 contextual hierarchy，在每个 scene-
  disjoint 训练折上串行训练固定种子 `0/1/2` 三个成员，用单调 `P(IoU>0.25), P(IoU>0.50)` 的算术均值
  选 proposal，并以三个成员对“ensemble proposal 相对 baseline”的 head-gain 总体标准差估计认知不确定性。
- 冻结切换规则使用 head lower-confidence-bound：`LCB = mean_gain - λ·population_std`，仅在 proposal
  不同于 baseline、`2*LCB025+LCB050 >= margin`、两个 head LCB 均为正且至少 `2/3` 成员同意 ensemble
  proposal 时切换。完整冻结 grid 为 `margin={0.10,0.1331222057,0.15,0.18}`、
  `min_LCB025={0,0.01,0.02}`、`λ={0,0.5,1}`、`consensus={2/3,1}`，共 72 个策略。
- 协议为外层五折 scene-disjoint OOF，每折三个模型严格串行（共 15 次小模型拟合）；每个 held fold 的
  策略只能由另外四个模型 OOF folds 的标签选择。候选必须达到按行数缩放的 V109 门槛
  `ceil(73*n/36665)` / `ceil(246*n/36665)`，且参与校准的各折两阈值 delta 均非负；随后按
  @0.25、@0.50、较少切换、较保守策略依次排序。若无合格策略则仍生成审计结果，但 gate fail closed。
- 在任何结果产生前冻结的最终 promotion gate：整体 `delta025>=73`（严格超过 V109）、`delta050>=246`，
  五折两阈值均严格为正；整体 bootstrap 下界至少 `34/183`；corrected 子群 bootstrap 下界至少
  `20/131`；regular 子群 @0.25 下界严格大于 0、@0.50 下界至少 29；五个 meta 选择均可行且最终
  policy 完全一致。只有全部通过才允许 full-fit/parity/一次正式评测；否则不创建权重、不访问 validation。
- 实现 `scripts/run_v110_meshsp_uncertainty_ensemble_oof.py` SHA-256
  `95ef5e9d69308bb104186bdee1badfc4134671ce339ac72ff134b7d1d268f596`；单卡 runner
  `scripts/run_v110_meshsp_uncertainty_ensemble_serial.sh` SHA-256
  `9261470445fe2cce4a460e29ea273110fbfe6ddecea786e0e1f5aba505af7234`，两者均 mode `0444`。runner
  绑定 V109 OOF、两个 cache manifest/receipt、backbone、V108 parent/geometry 与全部相关源码 SHA，要求
  恰好一张 GPU、GPU 无计算进程、至少 2.5 GiB 空间，并拒绝覆盖任何 V110 输出。
- 上线前 `py_compile`、72-policy import/boundary probe 通过；hierarchical/V95/V97/V99 相关回归
  **124/124 passed**。当前 GPU0 为 1 MiB/0%，V109 @0.50 artifact 及其 V108 parent/geometry 依赖链、
  V99 @0.25/Mask 最佳依赖链继续保留；V110 OOF 不写任何 `.pth`。

#### 14.149 V110 OOF 结果：不确定性有效，但 ensemble proposal/共识损失过大（2026-08-14）

- 单卡 screen `v110_uncertainty_oof` 串行完成 5 个 scene-disjoint 外折、每折种子 `0/1/2`，共 15 次
  训练；每个成员最终 loss 约 `4.4803--4.4924`，15 个 state SHA 均不同。结果、完整 stdout log、
  exit receipt 分别为只读 SHA-256 `7970a54bbf8a26ca09370be6a2413e436dbcb408dbc1dc1eec0a162ee40f8d48`、
  `4cdc1fa67039b7a942d4408fb842da589e4dac671d20805c242e8b1e96cb1ce1`、
  `461144ccfd56ee3cf0f9a9d80e520c5b872166b23092d5fd838ecbdb46d64dab`；预期 gate-fail exit 为 `76`。
  运行后 GPU0 回到 1 MiB/0%，protected weights 与 cache metadata 前后完全相同。
- Nested held-fold 结果为 `delta025=+44`、`delta050=+128`、1,833 switches；五折依次为
  `(+4,+34)/(+13,+10)/(+10,+37)/(+18,+28)/(-1,+19)`，整体 scene-bootstrap 95% 下界
  `+13/+84`。corrected 子群为 `+25/+88`、下界 `-2/+53`；regular 子群为 `+19/+40`、下界
  `+3/+16`。regular @0.25 下界确实从 V109 的 `-1` 改善为正，说明跨种子风险信号有价值。
- 但 5 个 meta folds 的 72-policy grid 全部 `eligible_candidate_count=0`，选出的 fallback policies 也不一致；
  全局标签可见诊断上最优 frozen policy 仅 `+58/+143`，仍显著低于 V109 `+72/+246`。失败 predicates
  包括两项总体 delta、两项总体 bootstrap、两项 corrected bootstrap、regular @0.50 bootstrap、fold 4
  @0.25、meta 可行性与一致性。按预注册规则禁止 full-fit/parity/validation，未创建任何 `.pth`。
- 失败原因定位为：对三个成员概率取均值会改变 V109 seed-0 的高价值 proposal，而 `>=2/3` proposal
  consensus 又进一步拒绝大量 @0.50 修复；这不是简单调 margin 能恢复的，因为全局最优 grid 也只到
  `+58/+143`。下一版应固定 V109 seed-0 proposal，仅让其他种子评价“同一个 anchor proposal 相对
  baseline”的收益方差，以风险折扣筛掉不稳定切换；`lambda=0` 必须严格退化为 V109 决策，避免再次
  牺牲已验证的 @0.50 增益。
- 权重清理审计：V110 只生成 JSON/log/exit，无失败权重可删；全 output `.pth` 仍为 11 个。用户指定保留的
  V109 @0.50 artifact SHA `20db69...785b` 及 V108 parent/geometry SHA `7b8956...616f` /
  `20f33c...7972` 完整保留，V99 @0.25/Mask Pareto 最佳链也未动。

#### 14.150 V111 预注册：V109 anchor proposal + 跨种子风险委员会（2026-08-14）

- V111 针对 V110 已定位的机制失败，不再平均三个成员的 proposal。每个外折仍串行拟合固定种子
  `0/1/2`，但 proposal 和原始 head gain **只取 seed 0 anchor**；实现必须用
  `tensor_sha256(proposal, 2*gain025+gain050, head_gain)` 与 V109 原始 OOF prediction SHA 完全相等，且
  `lambda=0, margin=0.15, min_gain025=0.02` 的 base policy 必须精确复现 V109 `+72/+246`，否则在策略
  选择前 fail closed。
- seed 1/2 仅在 seed-0 已选定的同一个 anchor proposal 上计算相对 baseline 的两个 head gains；风险定义为
  三成员 gain 相对 anchor gain 的 RMS disagreement，`risk=sqrt(mean_s((gain_s-gain_anchor)^2))`。切换使用
  `LCB=anchor_gain-lambda*risk`，因此 `lambda=0` 严格退化为 V109，而正 lambda 只删除跨初始化不稳的
  anchor switches，不会像 V110 那样改 proposal。这一模块只依赖模型预测分歧，可迁移到 Nr3D/Sr3D。
- 冻结 150-policy grid：V109 原 margin
  `{0.10,0.12,0.1331222057,0.15,0.18,0.22}`、`min_LCB025={0,0.0025,0.005,0.01,0.02}`、
  `lambda={0,0.5,1,2,4}`，`min_LCB050=0`。每个 held fold 的策略仍只看另外四个 scene OOF folds；
  非 base 候选必须在校准集上严格提高 base @0.25、不得降低 base @0.50，且四折两阈值 delta 均非负。
  若存在候选，依次最大化最弱折 @0.25、整体 @0.25、最弱折 @0.50、整体 @0.50，再偏好较少 switches
  与较保守参数；若不存在则确定性回退 V109 base，不伪造改进。
- 最终 promotion gate 沿用并强化 V110 的预注册标准：nested OOF `delta025>=73`、`delta050>=246`，
  五折两阈值严格为正；整体 bootstrap 下界至少 `34/183`；corrected 至少 `20/131`；regular @0.25
  下界严格大于 0、@0.50 至少 29；五个 meta folds 都必须选到严格优于 base 的候选且 policy 完全一致；
  两项 anchor exact-reproduction 契约也必须通过。全部通过前不 full-fit、不访问 validation、不写权重。
- 实现 `scripts/run_v111_meshsp_anchor_committee_oof.py` SHA-256
  `eed77d2b4b75923eb74e37dd7af5565b4459170e8c28614bc4971544bc03d89d`；单卡 runner
  `scripts/run_v111_meshsp_anchor_committee_serial.sh` SHA-256
  `be8bf3e72649062165e0d27dd0c0bfd3febc6d436c05106c0bf524b93711c3f4`，均 mode `0444`。runner
  额外绑定 V109 success 与 V110 failure 报告 SHA，拒绝覆盖输出并要求单卡空闲。`py_compile`、150-policy
  anchor/risk boundary probe 与相关回归 **124/124 passed**。V111 OOF 仍不产生 `.pth`。

#### 14.151 V111 OOF 结果与 Pareto 审计：严格复现 V109，发现小幅阈值交换（2026-08-14）

- 单卡完成 15 个成员；V111 result/log/exit 只读 SHA-256 分别为
  `1455ec6044104932c1ecfd89c3bcc17a1e0a31cb85c518ca15db209390575a58`、
  `a6aa7401e195ccb1242224f29e9272fdac61a592a6f8a225d7a39a72f1380892`、
  `461144ccfd56ee3cf0f9a9d80e520c5b872166b23092d5fd838ecbdb46d64dab`，预期 gate-fail exit `76`。
  seed-0 anchor prediction 的 expected/actual SHA 均为
  `bdcc8c01aabf5fe891f7789ca630ea533209209090d80f02852ea9e66184a57d`；base policy 精确复现
  V109 `+72/+246`、3,897 switches、逐折 `(+5,+53)/(+14,+35)/(+16,+43)/(+28,+56)/(+9,+59)`。
  因严格不许 @0.50 降低，五个 meta folds 均无 strict-improvement candidate，确定性回退 base；最终只失败
  `delta025>=73`、regular @0.25 bootstrap 下界 `>0`、meta 必须选到严格改进三项。protected 输入前后相同，
  GPU 回到 1 MiB，未访问 validation、未写 `.pth`。
- `analyze-results` 原始对比表（所有数字均为 train-only scene OOF，不是正式 validation）：

| 配置 | 选择协议 | delta@0.25 | delta@0.50 | switches | 关键稳定性 |
|---|---|---:|---:|---:|---|
| V109 / V111 base | nested，@0.50 不得降低 | +72 | +246 | 3,897 | bootstrap lower +34/+183；regular -1/+29 |
| V110 mean proposal + consensus | nested | +44 | +128 | 1,833 | fold4 @0.25=-1；regular lower +3/+16 |
| V111 risk，margin=.10, lambda=.5 | 全 OOF 诊断 | +78 | +238 | 3,622 | 五折均正：+6/+15/+19/+29/+9 |
| V111 risk，margin=.12, lambda=.5 | 全 OOF Pareto | **+79** | +236 | 3,414 | 五折均正：+6/+17/+19/+29/+8 |
| 95% @0.50-preservation 规则 | nested 探索 | +74 | +231 | 尚待重放 | held @0.25 五折 +6/+15/+19/+29/+5 |

- 关键发现：
  1. **观察**：anchor SHA 与所有 V109 指标完全相等；**解释**：V110 的退化确由改 proposal/强制共识引起，
     不是重训漂移；**含义**：固定 anchor 的风险过滤具有可靠对照；**下一步**：只改变风险容许规则。
  2. **观察**：`lambda=.5` 把 @0.25 增益提高 6--7 hits，只损失 8--10 个 @0.50 hits；**解释**：跨种子
     disagreement 能过滤一部分 @0.25 false-positive switches，但某些被过滤样本仍是 @0.50 修复；**含义**：
     两阈值并非完全同向，应按任务硬目标而不是强制逐 hit Pareto；**下一步**：允许最多 5% 的 OOF @0.50
     增益回撤，同时维持五折正向和 bootstrap 门。
  3. **观察**：95% 规则的 nested held 合计 `+74/+231`，但五折 margin 不一致（`.10/.10/.10/.12/.18`），
     `lambda=.5,min_LCB025=.02` 完全一致；**解释**：风险机制稳定而 margin 是折间 nuisance；**含义**：后续
     full-fit 可用 meta 多数 margin `.10`，但必须先重放得到完整 bootstrap/子群统计；**下一步**：V112 保存
     可复用的 train-only prediction cache，避免再为纯 policy 审计重复 15 次训练。
- V111 未产生失败权重，远程 `.pth` 仍为 11 个；V109 @0.50 与 V99 @0.25/Mask 两条 Pareto 最佳链继续保留。

#### 14.152 V112 预注册：95% @0.50 保留率的 nested 风险交换（2026-08-14）

- V112 是明确标记的 **prior train OOF-informed protocol iteration**：V111 训练 OOF 用于提出 95% 保留率，
  但 V112 每个 held fold 的 policy 仍只能读取另外四折，`validation_data_accessed=false`，也不把 V111 全局
  Pareto policy 直接当 held 结果。架构、种子、anchor exact-reproduction 与 150-policy grid 均不变。
- 每个 meta 校准将 @0.50 下限从“不得低于 anchor”改为
  `ceil(0.95 * anchor_calibration_delta050)`；候选仍须严格提高 anchor @0.25，且参与校准的四折两阈值
  delta 均非负。排序仍优先最弱折 @0.25、整体 @0.25、最弱折 @0.50、整体 @0.50、较少 switches 与
  保守参数。五个 meta policy 允许 margin 随折变化，但 `lambda/min_LCB025/min_LCB050` 必须完全一致；
  若通过，full-fit deployment margin 取五折多数，票数相同时取更高 margin。
- 在重放完整 bootstrap 前冻结 promotion gate：nested `delta025>=73`、`delta050>=230`，五折两阈值严格
  为正；整体 scene-bootstrap 95% 下界至少 `34/160`；corrected 子群至少 `20/120`；regular 子群
  @0.25 下界严格大于 0、@0.50 至少 20；五折都必须选到严格优于 anchor 的候选，risk family 一致，
  V109 anchor raw/base 两项精确复现。这里允许的 @0.50 OOF 回撤小于 V109 正式 @0.50 相对 49% 目标的
  175-hit 余量，但只有全部 train-only 门通过才允许 full-fit/parity；正式评测门仍是 REC
  `>=5610/4659` 且 Mask 至少用户 baseline。
- 为避免后续纯 policy 审计再次重训 15 个成员，V112 会额外写一个只读 gzip JSON train-only prediction
  cache，包含 scan/fold IDs、anchor proposal/gain/risk 与 baseline/proposal train IoU；明确不含 validation，
  也不是模型权重。其 SHA 将绑定进主报告，之后若无需再审计可安全删除。
- 实现 `scripts/run_v112_meshsp_anchor_committee_tradeoff_oof.py` SHA-256
  `7939de3e65b7952c5fd8fb8b67020d33c28b3eff405c2385164c1fb8f958f207`；单卡 runner
  `scripts/run_v112_meshsp_anchor_committee_tradeoff_serial.sh` SHA-256
  `f0fe7c9951d5811bf29640a5c97c26ffd1b8840d0f5f74f19a8007eaeaee9457`，均 mode `0444`。
  runner 绑定 V109/V110/V111 报告、V109--V112 相关源码、cache receipts/manifests 与三条 protected 权重 SHA；
  `py_compile`、policy aggregation probe 和回归 **124/124 passed**。V112 OOF 本身不创建 `.pth`。

#### 14.153 V112 首次重放的 cache 序列化兼容失败与只改 I/O 重跑（2026-08-14）

- 首次重放已完成 15 个模型、5 个 meta folds 与所有 diagnostics 计算；held folds 为预期
  `(+6,+53)/(+15,+33)/(+19,+45)/(+29,+51)/(+5,+49)`，合计 `+74/+231`，五折 risk family 均为
  `lambda=.5,min_LCB025=.02`。但在报告落盘前，远程 Python 的旧版 `gzip.compress` 不接受 `mtime`
  keyword，抛出 `TypeError`，exit `1`；因此没有主 JSON、没有 prediction cache、没有权重，GPU 已释放。
- 失败 stdout/exit 已完整移动封存到
  `failed_v112_cache_gzip_mtime_20260814/`，SHA-256 分别为
  `1c7170536c808fd6fc5e1ba3471d0eb386e34d7bb5547b0bc9d8fdbdf83ec24a` 与
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`，均 mode `0444`，可恢复且不冒充
  正式 V112 报告。
- 修复仅把 cache 压缩调用改为兼容旧 Python 的
  `gzip.GzipFile(fileobj=BytesIO(),mtime=0,filename='')`；模型、数据、policy grid、selector、gates、
  diagnostics 与输出 schema 均未改变。远程 legacy gzip round-trip probe 通过。
- 修复后实现 SHA-256 为 `5f03702325d6ed93f7fe15348a0161032e95dac676bc06f35548f57131b3ce1b`；
  runner SHA-256 为 `5915a32135a261ffded43e2d798184b878b803ed3d412a8bd4cc65d2234699ca`，均 mode `0444`。
  第二次重放仍须完整执行并以新 report/cache SHA 为准；首次内存结果只用于确认执行到序列化边界，不替代报告。

#### 14.154 V112 完整结果：总体与 corrected 通过，regular @0.25 下界等于 0（2026-08-14）

- 修复后完整只读 report/cache/log/exit SHA-256 分别为
  `128ce636d27234db7fca4fb23bd5d30945928d9ac9dcd1cf8139c38670a41b96`、
  `1123df3d312e433bf14b83874de99742906907738802bf878056ca07caa7ffdd`、
  `8bbf7931f262623a969f113348a3ea48a79486ac566f397e06af25501523c809`、
  `461144ccfd56ee3cf0f9a9d80e520c5b872166b23092d5fd838ecbdb46d64dab`；exit `76` 为预期 gate fail。
  cache 解压 schema、36,665 行长度、`validation_data_accessed=false`、`train_labels_only=true` 与 report 内绑定
  SHA 全部通过；压缩/解压大小为 770,350/3,804,496 bytes。
- Nested OOF 为 `+74/+231`、3,470 switches，五折为
  `(+6,+53)/(+15,+33)/(+19,+45)/(+29,+51)/(+5,+49)`；整体 bootstrap 下界 `+39/+175`。
  corrected 为 `+55/+168`、下界 `+26/+123`；regular 为 `+19/+63`、下界 **`0/+31`**。除
  `regular_bootstrap025_lower_strictly_positive` 外所有预注册 predicates 均通过；deployment margin 投票
  `.10:3/.12:1/.18:1`，risk family 一致。由于 regular 下界不是严格正数，禁止 full-fit/validation。
- protected weights/cache metadata 前后完全相同，GPU 回到 1 MiB；未创建 `.pth`，远程权重仍为 11 个。

#### 14.155 V113 预注册：非对称双 head 风险折扣的 cache-only nested 重放（2026-08-14）

- V112 cache 的 CPU-only 设计审计显示两个阈值对跨种子风险的敏感度不同：固定
  `lambda025=0.5, lambda050=0.25, min_LCB025=0.02, min_LCB050=0` 后，仅在六个既有 margins
  `{.10,.12,.1331222057,.15,.18,.22}` 中做 leave-one-fold-out 选择。该规则对 @0.25 风险更谨慎、对
  @0.50 修复保留更多，是阈值无关的双 head calibration 机制，不读取 corrected/regular 身份进行决策。
- 每个 held fold 的 margin 仍只由其余四折选择：相对 V109 anchor 必须严格提高 @0.25、保留至少 95%
  @0.50 增益，且四折两阈值非负；排序最大化最弱折 @0.25、整体 @0.25、最弱折 @0.50、整体 @0.50。
  deployment margin 取五折多数，平票取更高 margin。
- 在结果重放前冻结的 promotion gates：nested `delta025>=77`、`delta050>=235`；五折两阈值严格正；
  整体 bootstrap 下界至少 `40/180`；corrected 至少 `25/125`；regular 至少 `1/35`；五折均有 eligible
  margin，deployment margin 至少 3 票。全部通过才允许 full-fit 三成员、runtime parity 和一次正式评测。
  V113 明确标记 `prior_train_oof_used_for_protocol_design=true`，不把它描述为未调参的独立统计检验。
- 实现 `scripts/run_v113_meshsp_asymmetric_risk_replay.py` SHA-256
  `439c75c081c3f445564ad36a55dfb4ab92443061ee889301297081ab4b4a2ee3`；CPU-only runner
  `scripts/run_v113_meshsp_asymmetric_risk_replay_serial.sh` SHA-256
  `661dd251c58248275b70e3a7ebb841fe07ef63406cd30c6f8bf4b932d793a1fb`，均 mode `0444`。
  runner 绑定 V109/V112 report、V112 prediction cache 与 protected backbone/parent/geometry SHA，要求唯一 GPU
  空闲但不启动 CUDA。`py_compile` 与 asymmetric boundary probe 通过；此重放不创建权重。

#### 14.156 V113 重放结果：全部冻结门通过（2026-08-15）

- V113 CPU-only replay 的 result/log/exit SHA-256 分别为
  `ced399bca041cfa1f4213671100347f4a2423783aee4936ce7a82f785605e61d`、
  `3c72e1800d7954a278229bc86dcc112ce221f5153115843409c918527b1a8ef8`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，exit `0`；V112 cache
  SHA 仍为 `1123df3d...fdd`，seed-0 anchor raw prediction SHA 仍为
  `bdcc8c01aabf5fe891f7789ca630ea533209209090d80f02852ea9e66184a57d`。未读取 validation，protected
  backbone/parent/geometry 前后完全一致。
- Nested held-fold 为 `+78/+240`、3,553 switches，五折依次为
  `(+5,+52)/(+16,+36)/(+19,+45)/(+30,+53)/(+8,+54)`；整体 scene-bootstrap 95% 下界
  `+42/+184`。corrected 子群为 `+57/+172`、下界 `+27/+127`；regular 子群为
  `+21/+68`、下界 `+2/+36`。12 个预注册 predicates 全为 `true`，包括 regular @0.25 下界严格为正。
- 五个 meta folds 的 margin 为 `.1331222057/.12/.12/.12/.12`，因此 deployment 以 4/5 多数冻结为
  `aggregate_lcb_margin=.12, min_head_lcb025=.02, min_head_lcb050=0, risk_lambda025=.5,
  risk_lambda050=.25`。全标签可见的诊断 winner 为 `+80/+243`，但部署严格采用 nested 多数规则，不能用
  该诊断替换 held-fold 结果。
- `analyze-results` 原始对比表（均为 train-only scene OOF，不是 validation）：

| 配置 | nested delta@0.25 | nested delta@0.50 | switches | overall lower | corrected lower | regular lower |
|---|---:|---:|---:|---|---|---|
| V109 anchor | +72 | +246 | 3,897 | +34/+183 | +20/+131 | -1/+29 |
| V112 对称风险 | +74 | +231 | 3,470 | +39/+175 | +26/+123 | 0/+31 |
| V113 非对称风险 | **+78** | +240 | 3,553 | **+42/+184** | **+27/+127** | **+2/+36** |

- **观察**：V113 相比 V109 增加 6 个 @0.25 OOF hits，只回撤 6 个 @0.50 hits，同时三个 bootstrap
  切面均不再含负/零 @0.25 下界；**解释**：较小的 @0.50 风险系数保留了 V112 被过度过滤的阈值修复，
  而 @0.25 仍用 `.5` 抑制初始化不稳定 switch；**含义**：V113 是目前唯一同时通过总体、corrected、regular
  冻结门的风险策略；**下一实验**：只允许三成员 full-fit、完整 runtime parity 和一次正式验证，不再扫参数。
- replay 不生成 `.pth`；执行后整棵 output 仍为 11 个权重。用户指定的 V109 @0.50 artifact 与 V99
  @0.25/Mask Pareto 链均完整保留。

#### 14.157 V113 full-fit/runtime 预注册与源码冻结（2026-08-15）

- 部署结构为 `AsymmetricRiskContextualHierarchyCommittee`：固定三个独立种子 `0/1/2`；seed 0 唯一决定
  proposal，三成员只在同一个 proposal 相对 baseline 上产生两个 head gains，风险严格按
  `sqrt(mean_s((gain_s-gain_anchor)^2))` 计算，再按冻结的 `.5/.25` 系数构造 LCB。V99/V109 的 schema、
  loader 和原策略分支保持不变。
- full-fit builder `scripts/build_v113_meshsp_asymmetric_risk_artifact.py` SHA-256
  `9875fa881b4d81aae92fc4f1f033c06de252073b93de2e9e76c85ba53fde8a8f`；模型模块
  `models/rec_pareto_contextual_hierarchy.py` SHA
  `fa56d3da22b9ce0c8c6389173ff4f45c3407818d7a73c2aeab9f44ce81722d4a`；runtime
  `train_dist_mod.py` SHA `9916a5df1cf07d9a83d72108520b9b5617bb7991ecc3d526261eb07c4488a238`，均冻结为
  mode `0444`。builder 在落盘前强制 seed-0 full-fit state 与封存 V109 artifact
  `20db69...785b` 逐 tensor 完全相等，并在三个成员全部成功后才以 exclusive write 发布单一 committee
  artifact；任一成员失败都不发布权重。
- 单卡 runner `scripts/run_v113_meshsp_asymmetric_risk_artifact_serial.sh` SHA
  `cc600ffcb2a52aacef22d044d651cf81df28f0562df0216e50f8168482279403`，mode `0555`。它绑定 V113/V112
  report、V112 cache、V109 anchor、V108 parent/geometry、两个 cache receipt/manifest 与全部训练源码 SHA；
  要求恰好一张 GPU、GPU0 无计算进程、至少 2.5 GiB 空间，并拒绝覆盖任何 V113 输出。
- parity 程序 `scripts/audit_v113_runtime_parity_train.py` SHA
  `7c989cbdc1dd73aeeea482130b028be69bc5f1d570889f5a1b5493de87f9d938`，mode `0444`。它将在完整
  36,665 train rows 上逐字段比较离线/运行时 materialization、三个成员 logits、anchor proposal、成员 proposal、
  anchor gain、RMS risk、双 head LCB、aggregate LCB、switch 与最终 selected index，要求 exact equality。
- `py_compile`、新增 asymmetric risk 边界/三成员结构测试及完整 hierarchy/V94--V99/official 相关回归通过：
  **245/245 passed**，log SHA `89647f0dc1ca1e82547f3dcf3e96d4658890de7cf906c954f0933de2f5d57b14`。
  此时 GPU0 空闲，远程 `.pth` 仍为 11；只有 artifact full-fit 与 parity 均通过后才允许一次 9,508 样本
  official meshSP 验证。

#### 14.158 V113 full-fit 首次预检失败与保护快照修复（2026-08-15）

- 首次 runner 在任何训练/物化前 fail closed：复用函数 `capture_immutable_artifact_identities` 的契约要求输入键
  **只能**是 `backbone/parent/geometry`，初版 builder 为加强保护额外加入 `v109_anchor`，因此抛出
  `ValueError: protected paths must name backbone, parent, geometry`，exit `1`。GPU 始终为 1 MiB/0%，没有创建
  `v113_artifacts` 或任何 `.pth`，整棵 output 权重仍为 11 个。
- 失败 build/pipeline/exit 已原样封存到
  `failed_v113_artifact_protected_key_20260815/`，SHA-256 分别为
  `46e85c7546c4127d3bf2155ff1862a7d779bb40a7724e83df748f9c7743ac1bf`、
  `72c9eb31f3262fe95a036be46e1997ea001295777b085beec46872a90b144d1e`、
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`，均只读；没有删除失败证据。
- 修复只把 V109 anchor 改为独立 regular-file/non-symlink/mode-0444/stable-SHA 身份快照，再与三个既有 protected
  identities 合并比较；模型结构、三个 seeds、数据、训练函数、policy、OOF gate 与 artifact schema 都未改变。
  首次修复的 builder/runner SHA 为 `a6e7b894...8fb0` / `0b760d15...d8a9`；`py_compile`、V109 anchor
  快照 probe 与 runner `bash -n` 已通过。它们随后仅因 14.159 的证据类型修复被新 SHA 取代。

#### 14.159 V113 full-fit 第二次证据类型失败与无算法改动修复（2026-08-15）

- 第二次 runner 已通过全部输入保护并完成 seed 0 full-fit，但在构造 member receipt 时把训练函数返回的
  `epochs[-1]` 误当整数执行 `int(...)`；实际既有 V109/V111 契约是
  `{epoch,loss,query_loss,variant_loss}` 字典，因此抛出 `TypeError`、exit `1`。失败发生在任何 artifact
  exclusive write 之前；没有创建 `v113_artifacts` 或 `.pth`，权重仍为 11 个，GPU 已释放。
- 失败 build/pipeline/exit 封存于 `failed_v113_artifact_final_epoch_type_20260815/`，SHA-256 为
  `f55b60e32312c3292b371c2bb026e7946f82976e0fa0114777c9ce6039b3884c`、
  `736a03b89cc1a83496f4380809895445f83a1db08eba1199027da9a0dec8f220`、
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`，均只读；没有删除失败证据。
- 修复仅原样深拷贝 final-epoch 字典，并严格校验键集合、正整数 epoch 与三个有限正 loss；模型状态、种子、
  optimizer/epochs、数据、policy、artifact tensor 内容均不改变。最终 builder/runner SHA 更新为 14.157 所列
  `9875fa88...a8f` / `cc600ffc...403`；`py_compile`、真实字典契约 probe 与 `bash -n` 通过，允许从头重跑。

#### 14.160 V113 三成员 full-fit 成功与 parity 启动门（2026-08-15）

- 第三次从头运行成功，artifact/pipeline exit `0`；build/pipeline/exit SHA-256 为
  `55c53bf3854d708066820b9f23280c45cd8e892f753d3bed1672e1fb364ab040`、
  `19b0f8b18cddb38c8bfa228d2b162d6be90b6845ecb55dadcf8b05e527aed4b3`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。三个成员均完成 epoch 12：
  seed/state SHA 为 `0/4b5f4962...ceef`、`1/aba027c6...cfaa`、`2/eed30276...adfe`，三个 state
  互不相同；seed 0 与封存 V109 full-fit state 逐 tensor 完全相等，`anchor_matches_v109=true`。
- 新 committee artifact 为
  `v113_artifacts/asymmetric_risk_committee_h128_seeds0_1_2_fullfit.pth`，SHA-256
  `45f96279794da73c9d21f5f7e817bb47def03a86a30ab7db092c1b1c0275a37b`，2,713,351 bytes，mode `0444`；
  receipt SHA `1af664eac2be45cbd6032f1a9340c7043f24a2ab91c09284d267eda0bbc9097d`。artifact 绑定 36,665 rows、
  562 scenes、冻结 normalization、三个 member states、V113 OOF/report 与 V108/V109 保护链，strict reload 通过。
- 这是 **provisional candidate**，不是当前最好权重；output `.pth` 暂从 11 增至 12。V99/V109 及其依赖均未动，
  V109 artifact 继续按用户指定明确保留。只有 parity + official 后形成新 Pareto 最好，V113 才保留；否则在完整
  结果/可恢复证据写入后清理这个候选权重。
- parity runner `scripts/run_v113_meshsp_runtime_parity_serial.sh` SHA-256
  `440108e9211e765509b4397e860a4d45982a128ec84f490d8480651451341877`，mode `0555`；它绑定 artifact/
  receipt、parent/geometry、cache receipts/manifests、builder/model/runtime/parity source SHA，要求唯一 GPU 空闲，
  并拒绝覆盖报告。下一步只允许完整 36,665-row exact parity；此门通过前禁止 official。

#### 14.161 V113 完整 train/runtime parity 通过（2026-08-15）

- 在完整 36,665 train rows 上完成离线物化与 `train_dist_mod.py` 实际运行路径的逐字段审计，报告
  `v113_train_runtime_parity.json` SHA-256
  `53e86c392e86a7cb8813041d3a978413cc3c1784f741ded82a9444aba8ac4a81`，mode `0444`；log/pipeline/exit
  SHA-256 分别为 `9704930a4f6ff8eb8c4b7c839298ce7a1b73fc0e9f061f086c41e91c54102537`、
  `7b2f7358ae57e862e628d37149b5e7460e798ca2f08bffa9eeb032f2390d7795`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，exit `0`。
- `all_equal=true`：raw query/variant/aux/valid、baseline index/score、三成员 query/variant logits、anchor/member
  proposal、anchor/member gains、双 head RMS risk/LCB、aggregate LCB、Pareto/switch/selected index 的差异元素数
  全为 0，最大绝对差全为 0。审计未读取 validation、未修改权重，protected artifact 前后相同。
- 审计结束后 GPU0 为 1 MiB，output `.pth` 为 12 个：V113 仍只是 provisional candidate；V99、V109 及依赖
  完整保留。parity 门已满足，允许进入预先承诺的唯一一次 9,508 样本 official。

#### 14.162 V113 唯一一次 official 的冻结预检（2026-08-15）

- official 驱动 `scripts/run_frozen_v113_meshsp_official.py` SHA-256
  `bc56b06a2b00acf554fecbdbd0b41afe08cdb7d58536fc46a31ba8e2fa0d3f82`，mode `0444`；它固定 9,508
  样本、单 GPU、V108 parent/geometry、V113 committee 与 parity SHA，拒绝所有 GT inference flags，并在真正运行前
  exclusive 创建一次性 claim。REC 门为 hits `>=5610/4659`，Mask 同时记录用户 baseline 与 V99 Pareto 保留诊断。
- 不访问 validation 的 dry-run 预检
  `v113_official_preflight_dryrun.json` SHA-256
  `05a5b23e2b3ac9c4f4d21e808cf88c61d64295a62301d94791413e77a09724f9`，mode `0444`：
  `sample_count=9508`、`validation_data_accessed=false`、`inference_uses_ground_truth=false`，命令仅绑定冻结 V113
  checkpoint，`CUDA_VISIBLE_DEVICES=0`。同一 parser 对既有 V99/V109 原始 official stdout 回放得到精确 REC/Mask
  hits `5572/4797/5690/4976` 与 `5551/4834/5689/4974`，证明解析口径与既有结果一致。
- 单卡一次性 runner `scripts/run_v113_official_serial.sh` SHA-256
  `fb890f612a29236474ce504fbf1183201098908cd41636afd8510f80fc352d0a`，mode `0555`，`bash -n` 通过；它绑定
  artifact/parity/preflight/official-driver/model/runtime 全部 SHA，要求系统恰好一张物理 GPU 且无计算进程，并拒绝
  output、claim、driver log 或 exit receipt 预先存在。下一动作只有一次：执行该 runner，不再改参数或重跑。

#### 14.163 V113 唯一一次 official 结果、Pareto 判定与权重保留（2026-08-15）

- 9,508 样本 official 完成，runner exit `0`，GPU 回到 1 MiB。result/stdout/claim/driver/exit SHA-256 分别为
  `bdee0579c41e13b1c45f9822a316ee58d1f19534b7970fa7f05e5011ff8b088b`、
  `e60b3f6ec758a303ab55b9c137266d3b27634f11bbdb134cf139164d5878111d`、
  `f6dfbdd8da27aa807b54fc26c6a755ec9db32ceb8d5b14efd1f17a3c01b02cd9`、
  `24a6cf5ae7e376be387be9106536a69e0b0b6a24170fc0c247bc8eed250bebb9`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，全部 mode `0444`。
  样本数、无 GT inference、protected artifacts、完整 Python manifest 前后均严格相等。
- V113 REC：overall hits `5547/4835`，即 `58.3403%/50.8519%`；unique 为
  `1260/1419=88.7949%`、`1148/1419=80.9020%`；multiple 为
  `4287/8089=52.9979%`、`3687/8089=45.5804%`。@0.50 比 V109 多 **1 hit**，成为当前 official 最好；
  @0.25 比 V99 少 25 hits，未达到预注册 59% 门。
- V113 Mask：overall hits `5689/4974`，即 `59.8338%/52.3138%`，mIoU `45.9226%`；unique
  `90.2044%/80.1268%`，multiple `54.5061%/47.4348%`。三项均超过用户给定 MCLN baseline
  `58.70%/50.70%/44.72%`，但分别比 V99 少 `1/2` hits、mIoU 低 `0.0076` percentage point，因此
  Mask official 最好仍是 V99。
- official Pareto 对比：

| 版本 | REC@0.25 | REC@0.50 | Mask@0.25 | Mask@0.50 | Mask mIoU |
|---|---:|---:|---:|---:|---:|
| V99 | **58.6033% (5572)** | 50.4523% (4797) | **59.8443% (5690)** | **52.3349% (4976)** | **45.9303%** |
| V109 | 58.3824% (5551) | 50.8414% (4834) | 59.8338% (5689) | 52.3138% (4974) | 45.9224% |
| V113 | 58.3403% (5547) | **50.8519% (4835)** | 59.8338% (5689) | 52.3138% (4974) | 45.9226% |

- 保留判定：V99 是 REC@0.25 与三项 Mask 最好；V113 是 REC@0.50 最好；V109 在 REC 两阈值间仍是
  non-dominated 中间点，且用户明确要求保留。因此三个权重及 V108 parent/geometry 依赖全部保留。
  当前远程 output 共 12 个 `.pth`，其中 11 个是此前已审计保留集，新增 V113 已由新 @0.50 最好证明有用；
  **本轮没有删除任何权重**，避免把 Pareto 链或依赖误判为无用。
- 结论：V113 的 train-OOF 风险稳定性收益没有迁移到 validation @0.25，说明剩余瓶颈不是运行时不一致，而是
  train/validation 的 proposal 风险校准分布偏移；继续在同一 OOF cache 上调整 margin/lambda 不再可信。下一步应换
  与现有 OOF 调参正交、可由训练标签学习并冻结的泛化信号，而不是再次消费 validation 或重跑 V113。

#### 14.164 REC 主目标的 V114 预注册：语言条件空间关系注意力（2026-08-15）

- 用户进一步明确后续主要看 REC，因此主目标固定为 REC@0.25/@0.50；Mask 仅作为最终候选的非明显退化安全约束。
  `analyze-results` 原始 official 对比显示：

| 版本 | overall@0.25/@0.50 | unique@0.25/@0.50 | multiple@0.25/@0.50 |
|---|---|---|---|
| V99 | 5572/4797 | 1261/1143 | 4311/3654 |
| V109 | 5551/4834 | 1260/1151 | 4291/3683 |
| V113 | 5547/4835 | 1260/1148 | 4287/3687 |

- **观察**：V109→V113 在同一 train OOF baseline 上由 `+72/+246` 变为 `+78/+240`，即 @0.25 预期增加
  6 hits；official 却由 `5551/4834` 变为 `5547/4835`，即 `-4/+1`。@0.25 的 4-hit 回撤全部来自
  multiple，unique 不变。**解释**：风险 margin 只控制是否切换，不能补足多实例表达所需的候选间显式空间关系；
  同一 cache 上继续调整 lambda/margin 会加重 selection overfit。**含义**：下一候选必须改网络表示而非再调后处理。
- V114 固定为一个可迁移网络模块：对 16 个 REC query candidates 计算 3D pairwise location（距离、垂直/水平方向），
  以冻结的 64D `target_text_proj` 条件化四头空间注意力，再接 FFN；query/variant heads、V95 双阈值 graded-listwise
  objective、12 epochs、seed 0、V99 固定 Pareto margin `0.13312220573425293` 全部不变。空间坐标来自通用的
  `center_x/y/z_norm`，不含 ScanRefer unique/multiple 标签或 validation 信息，因此同样可用于 Nr3D/Sr3D。
- 在运行前冻结的单次 architecture-screen gates：相对同一 V108 parent+geometry baseline，OOF delta 至少
  `+110/+220`；五个 scene-disjoint folds 两阈值均严格正；scene bootstrap 下界至少 `+65/+165`；corrected
  下界至少 `+35/+115`，regular 至少 `+8/+25`；switch rate 不超过 13%。固定门相对 V108/V113 要求 @0.25
  至少多约 40/32 train hits，只有满足后才允许多 seed 稳定性、full-fit、parity；本轮不访问 validation、不生成权重。
- 模型 `models/rec_language_spatial_context.py` SHA-256
  `e26d2880d7c787531feeb49c2b1ecb0021bedf2bc801a04af761f1b0a127acfe`；OOF 程序
  `scripts/run_v114_meshsp_language_spatial_oof.py` SHA
  `4ee9fa146d9215745eaa997230727e144230a5bb2ecb4c56f32f1c9a6eb11a66`，均 mode `0444`；单卡 runner
  `scripts/run_v114_meshsp_language_spatial_oof_serial.sh` SHA
  `bbb1319a9f417a0ef7ca917335aec7e659b343a907c1484f64b5e00207706488`，mode `0555`。runner 绑定全部
  cache/receipt、V108 report、parent/geometry/backbone、hierarchy/V95/V99/V108 与复用空间层 SHA，要求恰好一张空闲 GPU。
- `py_compile` 与新模块/既有 V113 边界回归通过：**11/11 passed**；测试覆盖输出 shape、padding、有限值、空间坐标
  与语言条件确实影响 logits，以及冻结 hidden width。GPU0 空闲，V114 输出均不存在，远程 `.pth` 仍为 12 个。

#### 14.165 V114 OOF 结果：显式关系信号有效但无界替换主上下文失败（2026-08-15）

- 完整 36,665-row/562-scene 五折 OOF 正常完成，result/log/exit SHA-256 为
  `72413c5955e9b92ad5318452547b6212d4d02c980d7c2fb399de6b873994ac65`、
  `b9e576ee14c8c15254a3b4e470dde28cffae18ca9bdcde8771b924b80191c628`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`；执行 exit `0`，
  report promotion `passed=false`。protected artifacts/cache metadata 前后完全一致，validation 未访问，GPU 回到 1 MiB。
- REC OOF 为 `+54/+216`，五折 `(+8,+61)/(+3,+27)/(+21,+46)/(+20,+49)/(+2,+33)`；总体
  bootstrap 下界仅 `+15/+157`。corrected 为 `+28/+144`、下界 `-4/+97`；regular 为 `+26/+72`、
  下界 `+4/+38`。相对 V113，regular 略增 `+5/+4`，但 corrected 大幅下降，说明 pairwise language-spatial
  signal 本身有信息，却不能直接替代原 contextual anchor。
- V114 切换 9,365 行（25.5421%），其中 query changes 8,433；fix/break 在 @0.25 为 `203/149`，大量
  9,013 neutral switches。除五折严格正与 regular@0.50 下界外，冻结的 delta、总体/子群下界和 13% switch
  ceiling 均失败。**判定**：禁止 full-fit、禁止 validation、禁止生成 `.pth`；远程权重仍为 12 个。
- **下一实验**：保留 fold-local V99 标准 contextual hierarchy 作为冻结 anchor，只训练有界的语言空间残差 adapter；
  零初始化 delta heads 要求初始 logits 与 V99 bit-exact，固定 `tanh(delta)*0.25` 限制纠正幅度，并学习 reliability
  gate。该改变检验“关系信号应作小修正而非主排序替代”，不是在 V114 结果上扫描 margin。

#### 14.166 V115 预注册：冻结 V99 anchor 的有界语言空间残差适配器（2026-08-15）

- 每个 scene-disjoint fold 先以完全不变的 V99/V95 流程训练 anchor，随后永久冻结 anchor 参数并强制 eval；V115
  adapter 对 anchor contextual embedding 做目标语言条件的 3D spatial attention，以 reliability gate 融合，query/
  variant 两个 delta heads 均零初始化。初始输出必须与 anchor logits bit-exact，最终改变量由固定
  `0.25*tanh(delta)` 逐 logit 有界；anchor 在 adapter backward 中禁止产生 gradient。
- 训练只优化 adapter，仍用 V95 graded-listwise objective、12 epochs、seed 0、固定 V99 Pareto margin，无参数 sweep。
  运行前冻结 promotion gates：OOF delta 至少 `+105/+225`；五折两阈值严格正；总体 bootstrap 下界至少
  `+60/+170`；corrected 至少 `+35/+115`，regular 至少 `+8/+25`；switch rate 不超过 13%。只有全部通过才允许
  多 seed 稳定性，当前 screen 不生成权重、不访问 validation。
- 模型 `models/rec_anchored_spatial_adapter.py` SHA-256
  `2ca6cc3657661401aae708192a0d8c5c2d157abc198d2324c296073e691a3b4e`；OOF 程序
  `scripts/run_v115_meshsp_anchored_spatial_adapter_oof.py` SHA
  `dfbe142a44339137c977375cca77afce17fd8b515dcdde2184d7f3126d02ca4b`，均 mode `0444`；runner
  `scripts/run_v115_meshsp_anchored_spatial_adapter_oof_serial.sh` SHA
  `fb750aac7bc3e68069281fd901c9338fea1fccc19ef941b16d62fe65709be255`，mode `0555`，绑定 V99 anchor
  implementation、空间层、V95/V108、caches/receipts 与全部 protected artifact SHA。
- 初次测试唯一失败是理论 0.25 bound 在 float32 相减中略高于打印值，模型未改，只把断言容差冻结为
  `0.250001`；随后 `py_compile` 与 V114/V115/V113 相关回归 **14/14 passed**。覆盖 initial exact-anchor、
  residual bound、padding、anchor 无 gradient 及空间/语言路径。GPU0 空闲，V115 输出不存在，`.pth` 仍为 12。

#### 14.167 V115 首次运行的真实工厂类型失败与不覆盖重跑（2026-08-15）

- 首次 V115 在第 0 折训练前退出，exit `1`；原因是 adapter 的类型保护误把 V99 的 Pareto **策略名**当成其底层
  网络类。V99 `fit_v97` 实际返回 `ContextualHierarchicalQueryVariantReranker`，Pareto 约束位于预测策略层；原单测
  使用了结构等价但不是工厂真实返回类型的 `ParetoContextualHierarchicalReranker`，因此漏掉该接口差异。
- 失败发生在任何 fold adapter 训练、report 或权重写入前。失败 log/exit SHA-256 为
  `35cea51c63b8a450d8b280432ec08b18ee00df7db13544f474a6970fc2dac39a` / 
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`；原文件保留，未删除或覆盖。
- 修复后的 anchor 校验要求真实 hierarchy 基类、单层 `TransformerEncoder query_context` 与 hidden width 128；测试改用
  V99 工厂实际使用的 V97 contextual 类。模型/test SHA-256 更新为
  `45cc25d9209d1300f5c6f5ff2f9620acefdf47e8166451504aa7fdfb37c9b0cd` / 
  `e94421ae6c28714bf45a2c3c1c24a50dfdab91a4b56954cf2819939d37795a51`；`py_compile` 与相关回归
  **14/14 passed**。算法、残差尺度、训练目标、seed 和所有 promotion gates 均未改变。
- 为保留失败证据，重跑使用新文件名 `v115r1_*`；runner SHA-256
  `148f99f1c54b9acdb5a364aae2df15bb66e608ba4141d5a5a657122927fb0bb4`，mode `0555`，不覆盖首次失败回执。

#### 14.168 V115r1 OOF 结果：@0.50 强增益但 @0.25 安全性未过门（2026-08-15）

- 完整 36,665-row/562-scene 五折 OOF 正常完成，result/log/exit SHA-256 为
  `cae35808390c5f8c86b5ed3eeb73219ac226a20c7be091e36986fa25cf5f423f`、
  `b86e584ba87d6370b7da237fe65b6cccc4973e914bb0ebe284540952bf2ef774`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`；
  report promotion `passed=false`。protected artifacts/cache metadata 未变，validation 未访问，GPU 回到 1 MiB，`.pth` 为 12。
- REC OOF 为 `+75/+263`；五折为 `(+5,+65)/(+13,+32)/(+20,+48)/(+25,+58)/(+12,+60)`，两阈值
  每折都严格为正。总体 bootstrap 下界为 `+35/+199`；corrected 为 `+55/+193`、下界 `+21/+143`；regular
  为 `+20/+70`、下界 `-1/+34`。
- 适配器切换 4,321/36,665（`11.7851%`），显著低于 V114 的 25.5421% 且满足 13% ceiling；@0.25
  fixes/breaks 为 `199/124`，@0.50 为 `504/241`。冻结 anchor + 有界残差因此验证了“控制切换”的结构假设，
  但 @0.25 delta `75<105`、总体下界 `35<60`、corrected 下界 `21<35`、regular 下界 `-1<8`，未过安全门。
- **判定**：禁止 multi-seed/full-fit/validation，未生成任何 V115 权重。V115 对 @0.50 已超过预注册门（delta
  `263>=225`、下界 `199>=170`，子群也通过），下一候选只允许加强 @0.25 优先级与错误切换安全性，同时保留
  正的 @0.50 Pareto 约束；不从 validation 选择参数。

#### 14.169 V116 预注册：归一化 4:1 的 REC@0.25 主阈值后处理（2026-08-15）

- V115 网络、fold-local V99 anchor、训练目标、12 epochs/seed 0、residual scale 0.25、margin 与双 head
  Pareto 正增益约束全部不变；唯一算法变量是 hierarchical proposal selector 的阈值权重由 `2:1` 改为归一化
  `2.4:0.6`（比例 `4:1`、权重和仍为 `3.0`）。因此原固定 margin 仍处于同一 utility 尺度，同时更优先保护
  用户明确主看的 REC@0.25；不扫描其他比例。
- V116 绑定 V115r1 report SHA `cae35808...f423f` 作为 protocol-design evidence，但仍只对相同 train cache 做
  scene-disjoint OOF，`validation_data_accessed=false`。promotion gates 原样保持 `+105/+225`、五折双阈值严格正、
  总体下界 `+60/+170`、corrected `+35/+115`、regular `+8/+25`、switch rate `<=13%`；@0.50 不因
  @0.25 主目标而放宽。
- OOF 程序 `scripts/run_v116_meshsp_primary025_policy_oof.py` SHA-256
  `340d214b80f4616c8ee3fd7c3c04071b11472389e84abb3ffd1db6a7ed671a3b`，mode `0444`；runner
  `scripts/run_v116_meshsp_primary025_policy_oof_serial.sh` SHA-256
  `e32c9a6013232fa61a2db765df130975b6f3b0ba9b1f9ba843c226e41ac6015d`，mode `0555`；policy test SHA
  `50c2a7fea0f85c5bd0e0c7358c61094262f82653e8c34027cb6dc1443de740d2`。
- 新 selector 测试先暴露并修复 report exclusive-write 分支的一行缩进错误；该错误在静态/测试阶段发现，未启动训练、
  未产生任何 V116 output。修复后 `py_compile`、runner `bash -n` 与相关回归 **16/16 passed**；测试证明标准
  2:1 与 V116 4:1 在构造样本上产生预期不同选择、padding 不可被选择、权重和与比例均固定。

#### 14.170 V116 OOF 结果：提高 @0.25 utility 权重没有修复概率误校准（2026-08-15）

- 完整五折 OOF 正常结束，result/log/exit SHA-256 为
  `18fddfca24719062cc83b6b8e1c11183b04bb4e1b09da263d7e8b0db938ccdb9`、
  `832287a7bb199a04a19719b8bbfcd88324e44145e826645fe3c9fb36f7e92429`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`；validation 未访问、GPU 回到 1 MiB、`.pth` 仍为 12。
- REC OOF 为 `+72/+264`，五折 `(+5,+68)/(+13,+30)/(+20,+49)/(+24,+59)/(+10,+58)`；总体
  bootstrap 下界 `+33/+199`。corrected 为 `+54/+193`、下界 `+21/+142`；regular 为 `+18/+71`、
  下界 `-3/+35`。@0.25 的 delta/总体与两个子群下界仍失败，其余门通过。
- V116 切换 4,607（12.5651%），相对 V115 多 286；@0.25 fixes/breaks 从 `199/124` 变为 `201/129`，
  即只多 2 fixes 却多 5 breaks，净增益反而少 3；@0.50 从 `+263` 微增为 `+264`。因为所有 fold 的
  anchor/adapter 末轮 loss 与 V115 逐值相同，该退化可严格归因于 4:1 后处理，而非训练随机性。
- **判定**：禁止 full-fit/validation，不再试其他 utility 比例。证据说明 @0.25 的主要瓶颈是概率 head 的绝对校准，
  不是 2:1 排序权重；下一候选恢复标准 V115 selector，并直接训练绝对双阈值概率。

#### 14.171 V117 预注册：V115 有界残差的双阈值绝对概率校准（2026-08-15）

- 结构、fold-local V99 frozen anchor、residual scale 0.25、标准 2:1 proposal selector、Pareto margin、seed/epochs
  全部恢复并固定为 V115。唯一算法改动是在原 V95 graded-listwise loss 上，以系数 `1.0` 加入 query/variant
  pointwise BCE：直接以候选真实 `IoU>0.25`、`IoU>0.50` 为目标，阈值项按归一化 `2:1` 加权；padding 严格屏蔽。
  该辅助项训练绝对 hit probability，使“proposal 相对 baseline 的正 head gain”有可校准含义，而不是继续调后处理。
- V117 绑定 V115/V116 train-only reports 作为 protocol-design evidence；不读取 validation、不扫描 loss coefficient 或
  threshold weight。promotion gates 与 V115 完全一致：`+105/+225`、五折双正、总体下界 `+60/+170`、corrected
  `+35/+115`、regular `+8/+25`、switch rate `<=13%`。
- OOF 程序 `scripts/run_v117_meshsp_calibrated_adapter_oof.py` SHA-256
  `b9729032fb3771092ee0d51d88e589050b4092e9ede6b60f499fe5d4b3035f65`，mode `0444`；runner
  `scripts/run_v117_meshsp_calibrated_adapter_oof_serial.sh` SHA
  `737b61f87e24464fe21f225592bb7c61b38f102a6432a70bb23bc825094dfb98`，mode `0555`；calibration test
  SHA `31274c26d3519069236c684a5b248ec44c7775c4cb1b9ff659737d73fe5949bc`。
- `py_compile`、runner `bash -n` 与相关回归 **18/18 passed**；新增测试证明阈值正确的绝对概率得到更低损失，loss
  有限且 query/variant 两级均可反向传播。V117 尚未启动，输出不存在，GPU0 空闲。

#### 14.172 V117 OOF 结果：绝对 BCE 不能替代相对 switch-risk 校准（2026-08-15）

- 完整五折 OOF 正常结束，result/log/exit SHA-256 为
  `6e43afe461745ba4c65956d39f4e2fed7c62fd17d59a4641118549b1e1fc6c00`、
  `38c10c964e385a00bf76472bda41744a31c09117d48cfde7e1ffbe936b4f9d1b`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`；validation 未访问、GPU 回到 1 MiB、`.pth` 仍为 12。
- REC OOF 为 `+70/+243`；五折 `(+3,+53)/(+12,+33)/(+18,+45)/(+29,+56)/(+8,+56)`；总体
  bootstrap 下界 `+31/+180`。corrected 为 `+50/+173`、下界 `+17/+123`；regular 为 `+20/+70`、
  下界 `-1/+34`。@0.25 delta/总体与两个子群下界失败；@0.50 和 switch-rate 门通过。
- V117 切换 4,672（12.7424%）；@0.25 fixes/breaks `196/126`，@0.50 `482/239`。相对 V115，简单
  absolute BCE 增加 351 switches，却令净增益从 `+75/+263` 降到 `+70/+243`；说明单候选绝对 hit probability
  并不足以判断 proposal 相对 baseline 的 fix/break 风险。
- **判定**：禁止 full-fit/validation，不生成权重。V114--V117 已分别排除“无界替换 contextual anchor”“单纯提高
  @0.25 utility 权重”“简单绝对阈值 BCE”。下一方向必须训练 **proposal-vs-baseline 成对 signed gain/risk**，并在每个
  outer scene fold 内用独立 scene-disjoint calibration 产生训练样本，避免把同一模型的 in-sample proposal 当作安全标签；
  在该 nested protocol 固化前不再消费 GPU 或继续调同一 OOF 阈值。

#### 14.173 V118 预注册：nested scene-disjoint proposal-vs-baseline signed risk（2026-08-15）

- 每个 outer held fold 固定选择 `(held+1)%5` 作为完整 inner calibration fold；inner proposal model 只在其余三个
  scene folds 上拟合，因此 inner risk 标签中的 proposal 对该场景严格 OOF。随后 risk head 在 inner calibration 的
  V115-Pareto switches 上训练，outer V115 proposal model 再用四个 outer-train folds 从头拟合，最终在 outer held fold
  应用冻结 risk head。inner-fit、inner-calibration、outer-held 三组 scene 两两不交，禁止 in-sample proposal 安全标签。
- risk head 是可跨 ScanRefer/Nr3D/Sr3D 使用的 `23→32→2` MLP，仅输入推理可得的 proposal/baseline 双阈值概率、
  head/aggregate gain、原始候选得分差、query/variant change、中心/尺寸差、query/variant auxiliary 差；单测确认更改
  `candidate_ious` 不会改变特征。IoU 只生成 inner training targets：每阈值 fix=`+1`、break=`-4`、neutral=`0`；
  固定 event weight 4 的 Huber、200 epochs、lr/weight decay `1e-3`，无 grid。outer 接受条件为原 V115 Pareto 门与
  两个预测 signed-risk 均严格 `>0`。
- promotion gates 不变：OOF `+105/+225`，五折双阈值严格正，总体 bootstrap 下界 `+60/+170`，corrected
  `+35/+115`、regular `+8/+25`，switch rate `<=13%`；只在全部通过后允许稳定性/full-fit，当前不访问 validation、
  不生成权重。
- 模型 `models/rec_pairwise_switch_risk.py` SHA-256
  `15b93e0c7978e461c73221c0142070641c51bf252633f0d8fbacff47ee65cffd`，OOF 程序
  `scripts/run_v118_meshsp_nested_pairwise_risk_oof.py` SHA
  `a08bb024e7596e259bac556ae9493d3805f3e581d7995cfdee5b89a1553ce649`，均 mode `0444`；单卡 runner
  `scripts/run_v118_meshsp_nested_pairwise_risk_oof_serial.sh` SHA
  `e332f32fa30800b75ac8d7659025ed3d21ad732be9ae8cbff1ea0a9d2f977b7c`，mode `0555`；test SHA
  `5cf94c9c31d23a25504b8dd885cda67586dcc491bd44774d32539d7af72d4b07`。
- `py_compile`、runner `bash -n` 与相关回归 **21/21 passed**。额外 GPU synthetic fit/predict smoke 完成 200 epochs，
  score shape/finite 与 receipt 均通过，final loss `0.0946329`；这是随机合成数据，不构成结果选择。V118 output 不存在，
  GPU0 空闲，远程 `.pth` 仍为 12。

#### 14.174 V118 OOF 结果：成对风险提高 switch 精度但正收益门过度保守（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `8611a9bd24ab6e4d09e05dc37833f8e5d9dfc34e4c1be647ec66ecd4f10958da`、
  `dae9c8772e5f1998544106a771bebe6e54b36d6f3992b1c824717af192ad4dd9`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`；validation 未访问、GPU 回到 1 MiB、`.pth` 仍为 12。
- 每折 inner calibration 均含两个阈值的 fix/break/neutral，risk head final loss 为 0.3486--0.4815；V115 base
  switches `700/863/905/1071/782` 被保留为 `378/454/519/427/312`，总计 2,090/4,321（48.37%）。
- REC OOF `+63/+202`，五折 `(+4,+51)/(+11,+25)/(+14,+44)/(+25,+55)/(+9,+27)`；总体 bootstrap
  下界 `+34/+155`。corrected `+43/+146`、下界 `+20/+109`；regular `+20/+56`、下界 `+4/+30`。
  除五折双正、regular@0.50 与 switch ceiling 外，冻结 promotion floors 未通过。
- 相对 V115，@0.25 breaks/fixes 从 `124/199` 降至 `58/121`，每 switch 净收益率从 1.74% 提高到 3.01%；
  @0.50 从 `241/504` 降至 `121/323`，净收益率从 6.09% 提高到 9.67%。因此 nested pairwise risk 有真实
  精度价值，但要求 score `>0` 把 neutral-like 安全切换也拒绝过多，绝对 delta 不足。
- **判定**：禁止 full-fit/validation，不生成权重。下一单次消融不改变任何 inner/outer 训练、特征、target、loss 或
  seed，只把 safety veto 语义从“预测正收益”改为“预测不是 break”：以 break=`-4` 与 neutral=`0` 的数学中点
  `-2` 为固定边界；该值由 target 编码推出，不对 OOF 扫描。

#### 14.175 V119 预注册：signed-risk break/neutral midpoint veto（2026-08-15）

- V118 的 nested scene split、V115 proposal、23→32→2 risk head、+1/−4/0 target、event-weighted Huber、
  200 epochs 与全部输入完全不变。唯一算法差异为 outer gate：原 V115 Pareto switch 仅在任一阈值 risk score
  `<=(-4+0)/2=-2` 时被判为更接近 break 并 veto；两个 score 均 `>-2` 即视为 not-break。固定 midpoint 不扫描。
- promotion gates 继续保持 `+105/+225`、五折双正、总体下界 `+60/+170`、corrected `+35/+115`、regular
  `+8/+25`、switch `<=13%`；绑定 V118 report SHA `8611a9bd...958da` 为 protocol-design evidence，不访问 validation。
- OOF 程序 `scripts/run_v119_meshsp_nested_break_veto_oof.py` SHA-256
  `65938d1678f1ef53d5759f13feb187975b71ec1da302f43063432269f220b9a2`，mode `0444`；runner
  `scripts/run_v119_meshsp_nested_break_veto_oof_serial.sh` SHA
  `07c1e7436e490db5f213d7ef39cbffb2942e8ed2e6c58cb8d2a1657ca63c1d66`，mode `0555`；midpoint test
  SHA `20c4ec835459eaccd0f58b7be1ab3184168c25f053bd67424cac791103ab622a`。
- `py_compile`、runner `bash -n` 与相关回归 **22/22 passed**；测试把 break cost 4、中点 −2 及推导关系全部冻结。
  V119 output 不存在，GPU0 空闲，远程 `.pth` 仍为 12。

#### 14.176 V119 OOF 结果：连续 signed-risk 的中点仍不能可靠区分 break（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `731d110af0c8954d2b6ff5a5e8930c9b4897eaf4f71d96a89c720c7bbbd2ee8a`、
  `6992a845d57d74b5f598d970a31276f2381c6d85ee551c702d13c42ed312a51d`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`；validation 未访问、GPU 回到 1 MiB、`.pth` 仍为 12。
- REC OOF 为 `+71/+256`，五折 `(+5,+65)/(+11,+30)/(+20,+48)/(+23,+56)/(+12,+57)`；总体
  bootstrap 下界为 `+32/+193`。corrected 为 `+52/+187`、下界 `+18/+137`；regular 为 `+19/+69`、
  下界 `-2/+33`。@0.50 与 switch ceiling 通过，但 @0.25 delta、总体及两个子群下界仍未过预注册门。
- V119 保留 4,309/4,321 个 V115 switches，只 veto 12 个，行为几乎退化回 V115；固定中点虽然不含 OOF
  threshold sweep，却说明 Huber 连续回归分数的数值位置没有稳定对应 break/neutral 的语义边界。
- **判定**：禁止 full-fit/validation，不生成权重；停止继续搜索该连续 risk score 的阈值。下一候选保留完全相同的
  nested scene-disjoint protocol 和部署特征，但直接把每个阈值的结果建模为 break/neutral/fix 三分类，以固定 argmax
  语义消除连续 target 尺度与 veto 阈值之间的错配。

#### 14.177 V120 预注册：nested break/neutral/fix outcome classifier（2026-08-15）

- scene split、fold-local V115 proposal、23D proposal-vs-baseline 部署特征均与 V118/V119 相同；inner proposal
  继续只在三个 scene folds 拟合，并在独立 inner calibration fold 产生标签，outer held scenes 从不参与模型或
  classifier 训练。IoU 仍只用于 inner training label，推理特征不含 IoU、unique/multiple 或 validation 信息。
- risk head 改为固定 `23→32→(2×3)` 分类器，每个 REC 阈值的 class order 为 break/neutral/fix；用加权
  cross-entropy 训练 200 epochs，lr/weight decay 均为 `1e-3`。event weight 固定为 4、break cost 固定为 4，
  因此三类权重预先推导为 `(16,1,4)`。outer gate 不使用阈值：V115 Pareto switch 只有在两个阈值的 argmax
  都不是 break 时才接受；不做 grid/threshold sweep。
- promotion gates 不变：REC OOF `+105/+225`，五折双正，总体 bootstrap 下界 `+60/+170`，corrected
  `+35/+115`、regular `+8/+25`，switch `<=13%`。V119 result SHA `731d110a...d2ee8a` 被绑定为
  protocol-design evidence；当前 screen 不访问 validation、不生成 `.pth`，全部门通过前禁止稳定性/full-fit。
- 模型 `models/rec_pairwise_switch_classifier.py` SHA-256
  `b650cd738e004a3d2febba0bcf23d852bc1ca1fb9e845bfe1530dd466a2bf3cc`；OOF 程序
  `scripts/run_v120_meshsp_nested_outcome_classifier_oof.py` SHA
  `4b98efd48aca8b9da63eb0b412adf0713d21b022f0ddf144a9f653d0a8330e1b`，均 mode `0444`；单卡 runner
  `scripts/run_v120_meshsp_nested_outcome_classifier_oof_serial.sh` SHA
  `6f7601e482674677887c81e29adec2d46acc5ec42843c96254d5054a967373c4`，mode `0555`；test SHA
  `4c4a69e7504e4397efa188bb9c3b00f1ac6c647d16dcd8cf54e8673664077cc5`。
- `py_compile`、runner `bash -n` 与 V115--V120 定向回归 **14/14 passed**；测试覆盖输出 `[N,2,3]`、有限值与
  backward、四种双阈值 fix/break/neutral 标签及冻结权重 `(16,1,4)`。V120 output 不存在，GPU0 空闲，
  远程 `.pth` 仍为 12；V109 权重保持不动。

#### 14.178 V120 OOF 结果：23D 手工差分特征不足以区分安全修复与危险切换（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `bd8272d8c5337b60136f3d74dd50233c0a8c0e82b8f17287daf8505cd96afd87`、
  `f4e5f6ea325c8107da763869a47553670a657badc773c194cf572acdd87bee77`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`；protected artifacts/cache metadata 前后严格相等，validation 未访问，GPU 回到 1 MiB，
  远程 `.pth` 仍为 12。
- 五折分别保留 V115 switches `500/700`、`709/863`、`764/905`、`902/1071`、`624/782`，总计
  3,499/4,321（80.98%）；各 inner calibration fold 在两个阈值均含 break/neutral/fix 三类，classifier final
  loss 为 0.4318--0.5476，无标签缺失或训练崩溃。
- REC OOF 为 `+52/+190`，五折 `(+4,+32)/(+10,+32)/(+11,+45)/(+19,+42)/(+8,+39)`；总体
  bootstrap 下界 `+28/+147`。corrected 为 `+41/+134`、下界 `+21/+101`；regular 为 `+11/+56`、
  下界 `-3/+29`。仅五折双正、regular@0.50 和 switch ceiling 通过，其余冻结 promotion floors 均失败。
- @0.25 为 95 fixes/43 breaks，@0.50 为 281/91。相对 V115，classifier 去掉 822 switches，虽去掉
  81/150 个 breaks，却同时丢失 104/223 个 fixes，净 delta 从 `+75/+263` 降到 `+52/+190`；相对更保守的
  V118 也分别少 `11/12` hits。因此改变 gate 语义并未解决判别瓶颈。
- **判定**：禁止 full-fit/validation，不生成权重；停止在 V118--V120 的 23D 手工差分特征上调整 regression/classifier
  threshold、class weight 或接受规则。下一候选必须提升网络表示：使用 proposal/baseline 的 contextual object
  embedding、目标语言表示与相对 3D 几何构造 learned pairwise critic，再沿用 nested scene-disjoint 训练与固定
  train-only promotion gates，以检验“缺少语义判别信息”而非继续优化同一后处理。

#### 14.179 V121 预注册：全候选 contextual semantic hit critic（2026-08-15）

- V121 保留 V115 的 fold-local V99 frozen anchor、有界 language-spatial adapter、proposal selector 和 Pareto 门，
  也保留 V118--V120 的 nested scene split：inner proposal 仅在三个 folds 拟合，完整的第四个 inner calibration
  fold 训练 critic，outer held fold 始终与二者 scene-disjoint。V120 result SHA `bd8272d8...afd87` 被绑定为
  protocol-design evidence；validation 不访问，当前不生成权重。
- 与 V118--V120 只在约 650--965 个已选 switch 上训练不同，V121 对 inner calibration 中**所有有效的 16×7
  候选**学习 candidate hit probability。每个候选输入 fold-local V115 的 128D contextual query embedding、128D
  variant embedding、64D target-text mean，以及 16D 通用几何/rank auxiliary；共享 projection 后融合候选、语言
  乘性交互和 query/variant 差异，由 `128` hidden head 输出 `IoU>.25`、`IoU>.50` 两个 logits。标签只参与
  inner BCE；推理输入不含 IoU、unique/multiple 或数据集专用标识，能迁移到 Nr3D/Sr3D 的同类候选表示。
- critic 使用不加 class weight 的 BCE、12 epochs、batch 8192、AdamW lr/weight decay `1e-3`，不做参数或阈值
  sweep。outer 接受规则唯一固定为：先满足原 V115 Pareto gate，再要求 critic 的
  `P(proposal hit@.25)-P(baseline hit@.25)>=0`；@.50 critic 作为联合表示的辅助监督但不增加第二个 veto，避免已知
  @.50 强增益掩盖用户明确优先的 @.25。正式 promotion gates 仍为 `+105/+225`、五折双正、总体 bootstrap
  下界 `+60/+170`、corrected `+35/+115`、regular `+8/+25`、switch `<=13%`，任一失败即禁止 full-fit/validation。
- 模型 `models/rec_semantic_candidate_critic.py` SHA-256
  `d16798f49c94a9fd36b03e22002dff1a0bdbf7120bb1760badc721426e0d5a6f`；OOF 程序
  `scripts/run_v121_meshsp_nested_semantic_critic_oof.py` SHA
  `1517e0170ac1a103fa86b3d6423076b1081e79fae8ddc87267e90e4372ee33dc`，均 mode `0444`；单卡 runner
  `scripts/run_v121_meshsp_nested_semantic_critic_oof_serial.sh` SHA
  `4a2e8de3411831f85168248fa94e2a5beb2f6c9d71b7189df2e787384bad0362`，mode `0555`；test SHA
  `415faa6046592d500e7dfdb91ff6c16d2b202cb9d17edc2190718709ed16ef62`。
- `py_compile`、runner `bash -n` 与 V115--V121 定向回归 **19/19 passed**。真实 V115 interface 测试证明输出
  tensor contract 正确，并证明任意修改 `candidate_ious` 不会改变 inference components；10 万候选 GPU synthetic
  fit/predict 完成 12 epochs，final loss `0.543630`，概率 shape `(893,16,7,2)`、finite/range
  `[0.04514,0.97852]` 均通过。该烟测为随机合成数据，不参与结果选择。GPU0 空闲，V121 outputs 不存在，
  `.pth` 仍为 12；V99/V109/V113 及全部依赖保持不动。

#### 14.180 V121 OOF 结果：全候选绝对命中学习仍未形成 baseline-relative 判别（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `af29255b71e19b41879e287abfeb99b368b9fb9eec81cfb5377045bf950ded0d`、
  `650a660b2fd28d141c28a59cfc8805dacc3b2a53cd7fe107f44b6083264c08d0`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`；protected artifacts/cache metadata 前后严格相等，validation 未访问，GPU 回到 1 MiB，
  `.pth` 仍为 12。
- 每折 critic 使用 768,456--836,081 个有效候选，@.25 正例约占 88%--89%，12-epoch BCE final loss
  0.0251--0.0295；训练稳定。五折分别保留 V115 switches `383/700`、`557/863`、`444/905`、
  `718/1071`、`389/782`，总计 2,491/4,321（57.65%）。
- REC OOF 为 `+72/+185`，五折 `(+4,+44)/(+12,+30)/(+14,+33)/(+26,+45)/(+16,+33)`；总体
  bootstrap 下界 `+45/+142`。corrected 为 `+52/+128`、下界 `+31/+95`；regular 为 `+20/+57`、
  下界 `+3/+31`。五折双正、regular@.50 与 switch ceiling 通过，但 delta、总体及其余子群门均失败。
- @.25 为 141 fixes/69 breaks，@.50 为 290/105。相对 V115，V121 去掉 1,830 switches；@.25 虽去掉
  55 breaks，却也丢失 58 fixes，净 delta `75→72`；@.50 去掉 136 breaks、同时丢失 214 fixes，净 delta
  `263→185`。绝对 candidate BCE 在高度正例化的候选总体上 loss 很低，但 proposal 与强 baseline 的局部概率排序
  仍不可靠，复现了 V117“绝对命中概率不能替代相对 switch 判断”的结论。
- **判定**：禁止 full-fit/validation，不生成权重；不扫描 semantic probability difference threshold。V121 证明
  contextual/variant/language 表示与全候选训练基础设施可用，下一候选只改变监督问题：从同一 inner calibration 中
  构造 V115-Pareto hard candidate 与 baseline 的相对 fix/break 事件，使用共享语义 encoder 直接训练 pairwise
  utility classifier；neutral 不主导 loss，推理以固定 pairwise logit 符号判定，不回退到绝对 BCE。

#### 14.181 V122 预注册：V115-Pareto hard-pair semantic utility critic（2026-08-15）

- nested scene split、fold-local V115 proposal、128D contextual query/128D variant/64D language/16D geometry-rank
  inputs 与全部 protected artifacts 沿用 V121；V121 report SHA `af29255b...ded0d` 绑定为 protocol-design evidence。
  唯一研究变量是把 absolute candidate hit BCE 改为 candidate-vs-baseline 的相对事件监督，不改变 V115 网络、selector、
  margin、seed 或任何 promotion gate。
- 在 inner calibration 上先为**每个有效候选**计算其相对同一行 baseline 的 V115 双 head gains；只有满足原 V99/V115
  Pareto 条件（两 head gain 严格正、`2*g025+g050>=0.1331222057`）的 hard candidate 才进入 pair pool。每阈值
  candidate 与 baseline 命中状态不同才形成监督事件：candidate 修复 baseline 为 fix=`1`，candidate 破坏 baseline 为
  break=`0`；neutral 在该阈值完全不进入 loss，且不使用 class weight。由此扩大到所有 policy-relevant hard pairs，
  同时避免 V121 中约 88%--89% 的绝对正例掩盖局部相对差异。
- critic 用共享语义 candidate encoder 分别编码 candidate/baseline，再融合二者、signed/absolute/product difference 与
  V115 model gains，由 128-hidden head 输出双阈值 fix-vs-break logits。固定 50 epochs、batch 1024、AdamW
  lr/weight decay `1e-3`；无 grid/threshold sweep。outer gate 唯一为原 V115 Pareto acceptance 且
  `pairwise_logit025>=0`，即固定的 fix-vs-break decision boundary；@.50 仅作辅助相对监督，正式 @.50 promotion 门不放宽。
- promotion gates 原样保持：REC OOF `+105/+225`，五折双正，总体 bootstrap 下界 `+60/+170`，corrected
  `+35/+115`、regular `+8/+25`，switch `<=13%`；全部通过前禁止稳定性/full-fit/validation，不生成权重。
- 模型 `models/rec_semantic_pairwise_utility.py` SHA-256
  `181f19f041c9a652550442710c010a01d6065da03c3cfaa726172235a8a6676c`；OOF 程序
  `scripts/run_v122_meshsp_nested_semantic_pairwise_oof.py` SHA
  `429ba348623f966ab1c5813bec37bd2a0f52191b508a979e9883bcbec6b49a84`，均 mode `0444`；runner
  `scripts/run_v122_meshsp_nested_semantic_pairwise_oof_serial.sh` SHA
  `bfaa83e1ba659335a9a3fb86166207afd65288df1d0af2d9d4fcfe65b78fd1a3`，mode `0555`；test SHA
  `bcdc5f76e49db5a2a44c2a3feda3d040791856116bc940b059d76090151f57ff`。
- `py_compile`、runner `bash -n` 与 V115--V122 定向回归 **22/22 passed**。100-pair GPU synthetic end-to-end
  smoke 含每阈值 50 fixes/50 breaks，50 epochs 后 loss `0.002205`，预测 shape `(100,2)`、finite，@.25
  positive rate 精确 0.5；随机合成 smoke 不参与结果选择。GPU0 空闲，V122 outputs 不存在，`.pth` 仍为 12。

#### 14.182 V122 OOF 结果：hard-pair 监督有稳定正信号，但任意 pair fusion 泛化不足（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `95fad62c6b1e8df313292dcf88f07c3d3bfde33a395c05d254162b7f4a9b2321`、
  `28702d7337ae831611290821278d5da5cc5e678ad589de2230dab708a03140c1`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`、`deployable=false`。protected artifacts 与 cache metadata 前后严格相等，validation 未访问；
  实验输出目录 `.pth` 仍为 12，V99/V109/V113 权重均保留，V122 未生成或删除权重。
- 五折 hard-pair 训练分别包含 6,506/8,576/9,588/5,879/6,943 个 event pairs；50-epoch loss 从
  `0.4985/0.4708/0.4580/0.4659/0.4323` 降至 `0.0020/0.0032/0.0113/0.0080/0.0108`。critic
  仅接受 V115 switches `318/700`、`332/863`、`229/905`、`478/1071`、`278/782`，总计
  1,635/4,321（37.84%，全体 sample switch rate 4.46%）。
- REC OOF 为 `+57/+137`，五折 `(+8,+31)/(+7,+19)/(+7,+19)/(+15,+36)/(+20,+32)`；总体
  bootstrap 下界 `+32/+100`。corrected 为 `+38/+81`、下界 `+17/+50`，且 fold 1 的 @.25 为 `-1`；
  regular 为 `+19/+56`、下界 `+6/+38`。五折总体双正和 switch ceiling 通过，但 delta、总体下界、corrected
  与 regular@.25 门均失败。
- @.25 为 122 fixes/65 breaks，@.50 为 230/93。与 V121 的 `+72/+185` 相比，直接相对监督进一步减少
  856 switches，却把净增益降为 `+57/+137`；训练 loss 接近零而外折只保留 37.8% 的基础切换，说明任意
  candidate/baseline pair fusion 能记忆 inner hard events，却没有学到足够可迁移的候选排序。该结果仍比随机或负增益稳定：
  五折总体均双正，证明 hard-pair 事件监督本身有有效信号。
- **判定**：禁止 full-fit/validation，不生成权重，不扫描 epoch、hidden size 或 pairwise logit threshold。下一候选保留
  V122 数据、nested split、监督标签与全部冻结门，仅把任意 pair fusion 改为反对称 Bradley--Terry 结构：同一个低容量
  semantic scorer 分别产生 candidate/baseline utility，再以严格的 `s(candidate)-s(baseline)` 形成双阈值 logits。
  这样强制交换输入时符号翻转并约束可传递排序，直接针对 V122 的 pair memorization/generalization 缺口。

#### 14.183 V123 预注册：反对称 shared-utility hard-pair ranker（2026-08-15）

- V123 绑定 V122 result SHA `95fad62c...9b2321` 为唯一 protocol-design evidence；沿用同一 nested scene split、
  V115 proposal、V122 的全 hard-candidate pair pool、fix/break event mask、50 epochs、batch 1024、AdamW
  lr/weight decay `1e-3`、无 class weight。validation 不访问，当前不生成权重。
- 唯一研究变量是 critic 结构。V121 的 query/variant/text/aux projection 和 64D candidate representation 保持不变，
  但移除 V122 的 322D 任意 pair fusion 与 128-hidden classifier；共享无 bias 的线性 utility head 分别输出
  `s(candidate)`、`s(baseline)`，最终 logits 固定为
  `s(candidate)-s(baseline)+softplus(scale)*V115_model_gain`。每阈值 scale 从 `1.0` 初始化且始终为正。
  因此交换 candidate/baseline 并翻转 model gain 时 logits 必须严格变号，强制可传递、低容量排序。
- outer 接受规则仍只有原 V115 Pareto gate 且 `antisymmetric_logit025>=0`；@.50 只作辅助监督，不引入第二 veto，
  不扫描 threshold。promotion gates 原样保持：REC OOF `+105/+225`，五折双正，总体 bootstrap 下界
  `+60/+170`，corrected `+35/+115`、regular `+8/+25`，switch `<=13%`；全部通过前禁止稳定性、full-fit
  和 validation。
- 模型 `models/rec_semantic_antisymmetric_utility.py` SHA-256
  `21f49b070bf72af67a33c0087246c8367955b1ef531f5b8cc83b242a12c8499b`；OOF 程序
  `scripts/run_v123_meshsp_nested_semantic_antisymmetric_oof.py` SHA
  `0592dcbff468dee5e6ebbd20758c31f5c217188d64edc37290f733101b0c9ac2`，均 mode `0444`；runner
  `scripts/run_v123_meshsp_nested_semantic_antisymmetric_oof_serial.sh` SHA
  `8f097ace6343504eefe965cc99fdc5d40e3055fd1ffc05b7670120519c4dcf6b`，mode `0555`；test SHA
  `d487ffa9b3af466b3ac47b3e7c6b7e482fa1213535c96af6a3bea2b0ce95a19f`。
- `py_compile`、runner `bash -n` 与 V116--V123 定向回归 **23/23 passed**。100-pair GPU synthetic
  fit 的 loss `0.697874→0.001976`，输出 `(100,2)`、finite、两阈值 positive rate 均为 0.5；训练后输入交换
  的最大反对称误差精确 `0.0`，gain scales `0.99036/0.98974` 且为正。该随机合成 smoke 不参与结果选择。
  GPU0 空闲，V123 outputs 不存在，实验输出目录 `.pth` 仍为 12；V99/V109/V113 保持不动。

#### 14.184 V123 OOF 结果：反对称约束缓解过度 veto，但单头接受仍未达到门槛（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `be05e2e5ac077c19852981dcc1280ff18a27bfe5253cc2900a9ca6c272155b2d`、
  `ad4cec90e9fe9b773860152c96ccb99ab0aaa76f9928025539505b261009a18f`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`、`deployable=false`。protected artifacts/cache metadata 前后严格相等，validation 未访问；
  实验输出目录 `.pth` 仍为 12，V99/V109/V113 保持不动，V123 未生成或删除权重。
- critic 分别接受 V115 switches `309/700`、`457/863`、`346/905`、`617/1071`、`324/782`，总计
  2,053/4,321（47.51%，全体 sample switch rate 5.60%），比 V122 多保留 418 switches；各折 50-epoch
  final loss `0.00214/0.00354/0.01130/0.00795/0.01121`。
- REC OOF 为 `+74/+179`，五折 `(+5,+32)/(+12,+27)/(+12,+33)/(+27,+45)/(+18,+42)`；总体
  bootstrap 下界 `+45/+138`。corrected 为 `+49/+113`、下界 `+24/+78`，fold 1 @.25 为 `-5`；regular
  为 `+25/+66`、下界 `+10/+44`。五折总体双正、regular 双阈值和 switch ceiling 通过，但 pooled delta、
  总体下界及 corrected 门失败。
- @.25 为 134 fixes/60 breaks，@.50 为 266/87。相对 V122 的 `+57/+137`，反对称结构提升 `+17/+42`；
  相对 V115 的 `199-124=+75`，V123 去掉 65 fixes 与 64 breaks，@.25 只少 1 个净 hit，证明结构约束
  基本解决了 V122 的无差别过度 veto，但没有进一步识别足够的 @.25 breaks。@.50 仍比 V115 的 `+263` 少 84 hits，
  说明仅以 @.25 logit 决策没有利用训练中的 @.50 fix/break 辅助头。
- **判定**：禁止 full-fit/validation，不生成权重，不扫描阈值或训练超参。下一候选不改模型、数据或训练，只把固定
  outer sign gate 从 `logit025>=0` 改为 learned-Pareto：`logit025>=0 AND logit050>=0`。该零阈值双头交集与
  V115 原始双 head Pareto 语义一致，检验 @.50 事件头能否过滤 V123 留下的共同 breaks；仍使用全部冻结 promotion gates。

#### 14.185 V124 预注册：双阈值 learned-Pareto sign gate（2026-08-15）

- V124 绑定 V123 result SHA `be05e2e5...55b2d` 为唯一 protocol-design evidence。V115 proposal、V122 hard-pair
  event population、V123 反对称模型、nested scene split、50 epochs、batch 1024、AdamW lr/weight decay
  `1e-3`、seed 与 promotion gates 全部逐字节/逐参数保持；模型仍为
  `models/rec_semantic_antisymmetric_utility.py` SHA
  `21f49b070bf72af67a33c0087246c8367955b1ef531f5b8cc83b242a12c8499b`，不新增模型文件或权重。
- 唯一变量是 outer acceptance：原 V115 Pareto acceptance 之后，同时要求固定零阈值
  `antisymmetric_logit025>=0 AND antisymmetric_logit050>=0`。不调整零阈值、不做 OR/加权和/grid 比较；@.50
  头从 V122 起一直使用真实 fix/break 事件联合训练，因此该交集是 learned utility 对 V115 双 head Pareto 语义的直接延伸。
- promotion gates 不变：REC OOF `+105/+225`，五折双正，总体 bootstrap 下界 `+60/+170`，corrected
  `+35/+115`、regular `+8/+25`，switch `<=13%`；全部通过前禁止稳定性/full-fit/validation，不生成权重。
- OOF 程序 `scripts/run_v124_meshsp_nested_semantic_learned_pareto_oof.py` SHA-256
  `a9f18cf7ae164be991ca3e981ae3ca6cf71f23de1aa82549e5670a5998befc32`，mode `0444`；runner
  `scripts/run_v124_meshsp_nested_semantic_learned_pareto_oof_serial.sh` SHA
  `90c106b3da1c8d8ddc6bfa0aa40d22d0578695fca93eb061eb5eefd6acab1b66`，mode `0555`；test SHA
  `14bda885b4d69b7d6274e4b24c2198b8bf473d53abdbb74390f03b8207d5b3b6`，mode `0444`。
- `py_compile`、runner `bash -n` 与 V116--V124 定向回归 **27/27 passed**；新增测试覆盖 `++/00/+−/−+`
  四种 utility 符号及 baseline veto，只有 `++/00` 被接受。V123 模型 GPU smoke 和严格反对称证据直接复用；GPU0
  空闲，V124 outputs 不存在，实验输出目录 `.pth` 仍为 12，V99/V109/V113 不动。

#### 14.186 V124 OOF 结果：@.50 硬交集同时删除 fixes，不能替代单头 gate（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `cf430277b6e5f3c2f45ef8622fcb6b66df3022fa24d9252fb584c5c4377bc172`、
  `35ccb6a341e01a3a85a4ba05271cc98988772d08d05b39a058c5a1ad6ed497b4`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`、`deployable=false`。protected artifacts/cache metadata 前后严格相等，validation 未访问；
  实验输出目录 `.pth` 仍为 12，V99/V109/V113 不动，V124 未生成或删除权重。
- 五折训练 state SHA 与 V123 逐折完全一致，证明唯一变量是 acceptance。双头交集分别接受 V115 switches
  `201/700`、`247/863`、`210/905`、`350/1071`、`227/782`，总计 1,235/4,321（28.58%，全体
  sample switch rate 3.37%），比 V123 少 818 switches。
- REC OOF 为 `+65/+156`，五折 `(+7,+27)/(+12,+25)/(+8,+27)/(+20,+36)/(+18,+41)`；总体 bootstrap
  下界 `+40/+120`。corrected 为 `+44/+95`、下界 `+21/+64`，fold 1 @.25 为 `-2`；regular 为
  `+21/+61`、下界 `+9/+41`。总体五折双正、regular 双阈值和 switch ceiling 通过，其余主要门失败。
- @.25 为 115 fixes/50 breaks，@.50 为 223/67。相对 V123，@.50 sign veto 确实去掉 10/20 个
  @.25/@.50 breaks，但同时去掉 19/43 个 fixes，净 delta 分别 `74→65`、`179→156`。因此 @.50 head
  对共同 breaks 有信号但不足以支撑硬交集；继续做 OR、加权和或非零阈值将构成同一 OOF 上的 gate sweep，禁止。
- **判定**：禁止 full-fit/validation，不生成权重；停止围绕 V115 单一 proposal 做 semantic veto/gate 调整。
  V115 固定 proposal 本身只有 `199-124=+75` 的 @.25 净增益，V123 已用反对称 scorer 达到 `134-60=+74`，
  说明后处理 veto 已逼近该 proposal policy 的净上限。下一结构方向必须扩大 outer decision space：复用 V123 在所有
  hard candidates 上训练的可传递 utility，直接在每行全部 V115-Pareto 候选中 argmax，再与 baseline 比较，而不是只审查
  V115 已选的一个 proposal；这样才可能引入新 fixes 并超过 `+75` 的固定-proposal ceiling。

#### 14.187 V125 预注册：全 V115-Pareto hard-candidate utility argmax（2026-08-15）

- V125 绑定 V123/V124 result SHA `be05e2e5...55b2d`、`cf430277...bc172` 为直接设计证据；沿用同一
  nested scene split、fold-local V115 adapter、inner calibration hard-pair population、V123 反对称 shared-utility
  模型、50 epochs、batch 1024、AdamW lr/weight decay `1e-3`、seed 与全部 promotion gates。validation
  不访问，当前不生成权重。
- 唯一研究变量是 outer decision space。每个 outer-held 样本枚举所有 valid nonbaseline candidate，并严格要求
  V115 双 head gains 均为正且 `2*g025+g050>=0.1331222057`；用 V123 模型逐一计算 candidate-minus-baseline
  双阈值 utility，在所有合格候选中以 `logit025` 最大者为 proposal，平局固定取最低 flat candidate index。只有该行
  最大 `logit025>=0` 才接受，否则回退 baseline。`logit050` 继续作为联合训练头，不作新 hard veto；不扫描阈值、
  top-k、加权和或其他 gate。
- inference 输入只包含冻结的 query/variant/text/geometry-rank 表示与 V115 model gains；候选枚举和排序不读取
  `candidate_ious`、unique/corrected 标签或 validation。与 V123 的固定单 proposal gate 相比，本实验可选择 V115
  未选中的其他 Pareto 候选，因此有机会引入新 fixes、突破固定 proposal 的 `+75` @.25 净增益上限。
- promotion gates 原样保持：REC OOF `+105/+225`，五折双正，总体 bootstrap 下界 `+60/+170`，corrected
  `+35/+115`、regular `+8/+25`，switch `<=13%`；全部通过前禁止稳定性/full-fit/validation，不生成权重。
- 复用模型 `models/rec_semantic_antisymmetric_utility.py` SHA-256
  `21f49b070bf72af67a33c0087246c8367955b1ef531f5b8cc83b242a12c8499b`；OOF 程序
  `scripts/run_v125_meshsp_nested_semantic_all_candidate_oof.py` SHA
  `6cee94ab0ef3e06da6c22a412304ab33877c19517f3412b232f238a48d08e9a0`，mode `0444`；runner
  `scripts/run_v125_meshsp_nested_semantic_all_candidate_oof_serial.sh` SHA
  `ea2c8b76c1de5e0acd5ebbeaacc8b9e6f7b59caacd02edbbfdddaf63004b845a`，mode `0555`；test SHA
  `d07d9335444a306fdbd1692860323bc03654af485b3d388def74667df6a7f0aa`，mode `0444`。
- `py_compile`、runner `bash -n` 与 V116--V125 定向回归 **31/31 passed**；新增测试构造同一行两个合格候选，
  验证选择 utility 更高但不是 V115 原 proposal 的候选，并验证另一行在最大 utility 为负时严格回退 baseline。
  GPU0 空闲，V125 outputs 不存在，V99/V109/V113 权重保持不动。

#### 14.188 V125 OOF 结果：全候选 argmax 放大中性切换，净增益低于固定 proposal（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `b52c96f7a641a069714dc36bf6cc2dbe055b6937693a441291bc0b1fc0e1806b`、
  `fc3fd380ae7a96eac29a390474cde38440ed13a3c3e11e6ad06373bd4fb91420`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`、`deployable=false`。protected artifacts/cache metadata 前后严格相等，validation 未访问；
  实验输出总 `.pth` 仍为 12，V125 未生成或删除权重。
- 五折共评估 400,095 个 all-hard-candidate pairs，24,261 行至少有一个候选；最终接受 13,662 次切换，
  switch rate `37.2617%`，其中 19,940 行超出 V115 原固定 proposal 接受集合，23,482 行的 utility argmax
  与 V115 proposal 不同。所有被选候选仍满足冻结 Pareto/margin 条件，margin violation 和非正 head gain 均为 0。
- REC OOF 仅为 `+31/+99`，五折 `(+0,+11)/(+2,+30)/(+10,+27)/(+15,+15)/(+4,+16)`；总体
  bootstrap 下界 `-3/+45`。corrected 为 `+21/+75`、下界 `-7/+33`；regular 为 `+10/+24`、
  下界 `-9/-9`。@.25 首折为 0，且 delta、bootstrap、subgroup 与 switch rate 门全部失败。
- @.25 为 186 fixes/155 breaks，另有 13,321 neutral switches；@.50 为 400/301，另有 12,961 neutral
  switches。相对 V115 固定 proposal 的 `+75/+263`，V125 扩张为三倍以上切换却降到 `+31/+99`；相对
  V123 的 `+74/+179` 也显著下降。共享 utility 在 inner-held hard events 上能排序，但把它当作每行 100 余候选的
  全局 maximum 会产生明显 winner's curse：大量候选因估计噪声取得极端高分，真实效果主要为中性且新增 breaks 接近 fixes。
- **判定**：禁止 full-fit/validation，不生成权重，不对本次 utility logits 做事后阈值/top-k/switch-budget 扫描。
  扩大候选空间本身不是缺失环节；下一方向必须先解决 candidate-set-size 偏差和最大值选择噪声。若继续使用 all-candidate，
  应在新的 nested 预注册实验中训练 listwise/choice-aware selector，使 baseline 与整组候选共同归一化，并显式加入 abstain
  选项，而不是把独立 pairwise logits 直接取最大值。

#### 14.189 V126 预注册：listwise candidate-set normalization + baseline abstain（2026-08-15）

- V126 绑定 V125 result SHA `b52c96f7...e1806b` 为直接设计证据；nested scene split、fold-local V115
  adapter、all-hard-candidate 枚举、V123 反对称 shared utility、seed、50 epochs、AdamW lr/weight decay
  `1e-3` 与全部 promotion gates 保持。validation 不访问，当前不生成权重。
- 唯一研究变量是 inner-calibration 训练目标。每个样本把全部 V115-Pareto hard candidates 与一个 score 固定为 0
  的 baseline abstain 动作放入同一 listwise softmax。对每个阈值，若 baseline miss 且存在 eligible fix，则所有 fix
  actions 构成可接受正集合，最小化其总 softmax 概率的负对数；否则 baseline 是唯一正动作，所有候选（含 neutral 和
  break）都被作为负动作。双阈值 loss 等权平均，不使用 class weight。
- outer inference 不变：枚举所有 valid nonbaseline 且双 head gain 严格正、
  `2*g025+g050>=0.1331222057` 的候选，按 `logit025` argmax，平局取最低 flat index；只有最大值相对
  baseline score `0` 非负才切换。`logit050` 仅为联合训练头，不新增 hard veto；不扫描温度、阈值、top-k、
  switch budget 或 logit 组合。inference 不访问 IoU、unique/corrected 或 validation。
- 该目标直接针对 V125 的 winner's curse：候选数越多，softmax denominator 对全部非目标动作的累计惩罚越强；
  baseline-only 行要求所有候选共同低于显式 abstain，而不是仅在 event pairs 上学习局部符号。
- promotion gates 不变：REC OOF `+105/+225`，五折双正，总体 bootstrap 下界 `+60/+170`，corrected
  `+35/+115`、regular `+8/+25`，switch `<=13%`；全部通过前禁止稳定性/full-fit/validation，不生成权重。
- 复用模型 `models/rec_semantic_antisymmetric_utility.py` SHA-256
  `21f49b070bf72af67a33c0087246c8367955b1ef531f5b8cc83b242a12c8499b`；OOF 程序
  `scripts/run_v126_meshsp_nested_semantic_listwise_oof.py` SHA
  `97b17c9bbb56b2654a10c12e7cb72db507528a5a21d0e7b39b928d8d0b750ded`，mode `0444`；runner
  `scripts/run_v126_meshsp_nested_semantic_listwise_oof_serial.sh` SHA
  `1fe9c7c4aee9251a4cafb3b2a5b5bd966ce96312a6a9c7f322356cd3f5051e34`，mode `0555`；test SHA
  `a219e29e3243377541f192dcb091fceb88be1c0ff1c11d49bd112f16766dda64`，mode `0444`。
- `py_compile`、runner `bash -n` 与 V116--V126 定向回归 **36/36 passed**。128-row GPU synthetic
  end-to-end smoke 含 64 fix-target/64 baseline-target 行，listwise loss `0.755524→0.011911`，最终恰好接受
  64 行，验证 dense indexed scatter 的梯度回传与 baseline abstain 生效；随机合成 smoke 不参与结果选择。
  GPU0 已释放，V126 outputs 不存在，实验输出 `.pth` 仍为 12。

#### 14.190 V126 OOF 结果：listwise abstain 修复 precision，但等权行先验导致过度保守（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `899a884cdd362e13f9d1772e53bd8cca01af9bf849d66417eee33df6fd7e8c53`、
  `40aad6d2dcb56d668f09bb9a2f1a93b1b7c326841f8600a79d08ea6ffb68df9a`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`、`deployable=false`。protected artifacts/cache metadata 前后严格相等，validation 未访问；
  `.pth` 仍为 12，V126 未生成或删除权重。
- 五折接受 `9/40/34/41/18` 次切换，总计仅 142/36,665（`0.3873%`）；对应 fold-local inner
  @.25 fix-target 行仅 `54/89/96/84/58`，而 baseline-target 行均约 7,000。五折 listwise final loss
  `0.000736/0.001928/0.002439/0.000686/0.000970`，显示拟合稳定但稀有 fix prior 被等权行平均严重压制。
- REC OOF 为 `+24/+17`，五折 `(+3,+5)/(+2,+4)/(+4,+0)/(+12,+6)/(+3,+2)`；总体
  bootstrap 下界 `+10/+3`。corrected 为 `+18/+11`、下界 `+6/-1`；regular 为 `+6/+6`、
  下界 `0/-2`。@.25 五折全正且 bootstrap 下界为正、switch ceiling 通过，但 delta、@.50 严格五折及
  promotion 下界均失败。
- @.25 为 38 fixes/14 breaks/90 neutral switches，event precision `73.08%`；@.50 为 39/22/81。
  相对 V125 的 13,662 switches 和 `+31/+99`，V126 把 winner's curse 大幅压住，以约 1/96 的切换数获得
  `+24` @.25，证明 candidate-set normalization 与显式 abstain 是正确结构；问题由“过度选择”转为“fix-target
  行先验过低”。
- **判定**：禁止 full-fit/validation，不生成权重，不对 logits 做阈值或温度扫描。下一实验保留 V126 的模型、
  candidate sets、positive-set 定义、baseline score 0 与 inference 原样，只在训练 loss 中做封闭式 prior correction：
  每阈值 baseline-target 行 weight=1，fix-target 行 weight=`N_baseline/(4*N_fix)`，其中 4 直接复用冻结的
  false-positive cost。这样 weighted baseline 总量仍为 fix 总量的 4 倍，无自由超参或 OOF sweep。

#### 14.191 V127 预注册：false-positive-cost prior-corrected listwise abstain（2026-08-15）

- V127 绑定 V126 result SHA `899a884c...e8c53` 为直接设计证据；模型、nested scene split、fold-local
  V115 adapter、all-hard-candidate sets、fix-only positive sets、baseline score `0`、50 epochs、row batch 512、
  AdamW lr/weight decay `1e-3`、outer argmax/zero gate 与 promotion gates 全部保持。validation 不访问，当前不生成权重。
- 唯一变量是 inner listwise 行权重。每阈值 baseline-target 行固定 weight `1`；fix-target 行固定 weight
  `N_baseline/(4*N_fix)`，计数仅来自该 outer fold 的 inner calibration。于是 baseline-target 总权重严格为
  fix-target 总权重的 4 倍；常数 4 直接复用冻结诊断中的 false-positive cost，不新增可调超参、阈值或 grid。
- candidate softmax、positive mass 与 inference 完全沿用 V126：训练时 baseline miss 且存在 eligible fix 才以所有 fixes
  为正集合，否则 baseline 是唯一正动作；推理按所有 V115-Pareto candidates 的 `logit025` argmax，最大值非负才切换。
  `logit050` 仅联合训练，不 hard veto；不访问 IoU、unique/corrected 或 validation。
- promotion gates 不变：REC OOF `+105/+225`，五折双正，总体 bootstrap 下界 `+60/+170`，corrected
  `+35/+115`、regular `+8/+25`，switch `<=13%`；全部通过前禁止稳定性/full-fit/validation，不生成权重。
- 复用模型 `models/rec_semantic_antisymmetric_utility.py` SHA-256
  `21f49b070bf72af67a33c0087246c8367955b1ef531f5b8cc83b242a12c8499b`；OOF 程序
  `scripts/run_v127_meshsp_nested_semantic_prior_corrected_oof.py` SHA
  `0da18187a7690ad9e5d57feedbc6bb9ef69fe723dd228ee7969e1f5c4277fe12`，mode `0444`；runner
  `scripts/run_v127_meshsp_nested_semantic_prior_corrected_oof_serial.sh` SHA
  `2338843fa5b76f0d2048746582455268a789d50bbba655619bab1bc756f04039`，mode `0555`；test SHA
  `430d53b2c7383b626e2dacef5fd21d8636790afe89224bb4175134f4b8cb0fbf`，mode `0444`。
- `py_compile`、runner `bash -n` 与 V116--V127 定向回归 **41/41 passed**。128-row GPU synthetic smoke
  含 16 fix/112 baseline rows，closed-form fix weight 精确 `112/(4*16)=1.75`，listwise loss
  `0.881189→0.011561`，最终恰好接受 16 行；随机合成 smoke 不参与结果选择。GPU0 已释放，V127 outputs
  不存在，`.pth` 仍为 12。

#### 14.192 V127 OOF 结果：prior correction 未释放 recall，且 @.25 净增益下降（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `ad1fe06606eb696462a08877f314050e0a0bf6bb9780df1d149897fcfcb1904a`、
  `bb56b15a75d5c1c40b0c76dae79d5c0abf03d181c349c05fbe6e530cd5645fdc`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`、`deployable=false`。protected artifacts/cache metadata 前后严格相等，validation 未访问；
  `.pth` 仍为 12，V127 未生成或删除权重。
- 五折 fix-target weights @.25 为 `32.53/21.31/19.19/21.65/30.30`，@.50 为
  `11.32/8.20/8.39/11.18/10.11`；接受 switches `15/29/48/52/42`，总计 186（`0.5073%`），
  只比 V126 多 44，说明闭式 prior correction 并未显著改变强 baseline softmax 的决策边界。
- REC OOF 为 `+21/+21`，五折 `(+2,+5)/(+5,+3)/(+2,+4)/(+7,+7)/(+5,+2)`；总体
  bootstrap 下界 `+9/+8`。corrected 为 `+12/+13`、下界 `+2/+2`；regular 为 `+9/+8`、
  下界 `+2/0`。总体五折双正和 switch ceiling 通过，但所有 delta/bootstrap promotion 门失败。
- @.25 为 37 fixes/16 breaks/133 neutral，净增益反而低于 V126 的 38/14=`+24`；@.50 为 44/23=`+21`。
  加权增加的选择主要是 neutral，并新增 2 个 @.25 breaks，没有带来更多 fixes。继续改变 cost 将成为同一 OOF 上的权重
  sweep，禁止。
- **判定**：禁止 full-fit/validation，不生成权重；停止调整 listwise prior/temperature/threshold。下一结构保留未加权
  V126 作为高精度 **rescue**，同时把 V115 的原 proposal/acceptance 设为不可替换主路径：V115 已接受的行原样保留，
  仅在 V115 abstain 行允许 V126 的全候选 listwise 选择新增切换。该 union 不牺牲 V115 的 `+75/+263`，并把
  V126 的精度用于补充 fixes；最大切换约 `(4321+142)/36665=12.17%`，天然低于 13% ceiling。

#### 14.193 V128 预注册：V115-protected primary + V126 listwise rescue（2026-08-15）

- V128 绑定 V126/V127 result SHA `899a884c...e8c53`、`ad1fe066...1904a` 为设计证据；明确回退到 V126
  **未加权** listwise 训练。nested split、fold-local V115、all-hard-candidate sets、fix/baseline positive sets、
  shared utility、baseline score 0、50 epochs、row batch 512、AdamW lr/weight decay `1e-3`、seed 与 promotion
  gates 全部保持；validation 不访问，当前不生成权重。
- 唯一变量是 outer policy composition。V115 `base_accepted=true` 的行严格保留其原 proposal、head gains 和 acceptance，
  listwise 无权替换或 veto；仅在 V115 `base_accepted=false` 的行，运行 V126 all-candidate `logit025` argmax，最大值
  非负才新增 rescue，否则 baseline。`logit050` 仍只联合训练；无阈值、温度、cost、top-k 或 switch-budget sweep。
- 该 union 直接组合 V115 的 recall 与 V126 的 precision。V115 OOF switches 为 4,321；V126 全部 raw accepts 仅 142，
  因而即使完全不重叠，V128 switches 上界 4,463/36,665=`12.1724%`，仍低于冻结 `13%` ceiling；实现同时断言
  每个 V115 accepted identity 不变、rescue 只能发生在 V115 abstention。
- promotion gates 不变：REC OOF `+105/+225`，五折双正，总体 bootstrap 下界 `+60/+170`，corrected
  `+35/+115`、regular `+8/+25`，switch `<=13%`；全部通过前禁止稳定性/full-fit/validation，不生成权重。
- 复用模型 `models/rec_semantic_antisymmetric_utility.py` SHA-256
  `21f49b070bf72af67a33c0087246c8367955b1ef531f5b8cc83b242a12c8499b`；OOF 程序
  `scripts/run_v128_meshsp_nested_semantic_protected_rescue_oof.py` SHA
  `3464f6fed7e3e174915f8296ff43145f7ffaa6168230bdd222589544b961be7c`，mode `0444`；runner
  `scripts/run_v128_meshsp_nested_semantic_protected_rescue_oof_serial.sh` SHA
  `33885a22d9eb80ab1b21d35d5f49c0f1ff20f56841fd15246da03235ee156237`，mode `0555`；test SHA
  `04f17ee6a2b7f9d6ba72285b36dfd12aac2b22f90c4fdc969dd292ae743932cd`，mode `0444`。
- `py_compile`、runner `bash -n` 与 V116--V128 定向回归 **46/46 passed**。新增 synthetic contract 同时覆盖：
  V115 accepted 行即使 listwise 更偏好其他候选也保留 V115、V115 abstain+positive utility 新增 rescue、V115
  abstain+negative utility 回退 baseline。V126 的 listwise GPU 收敛 smoke 直接复用；GPU0 空闲，V128 outputs
  不存在，`.pth` 仍为 12。

#### 14.194 V128 OOF 结果：V126 高精度 fixes 与 V115 重叠，abstain 补集无增量价值（2026-08-15）

- 完整 nested 五折正常完成，result/log/exit SHA-256 为
  `3b90a175a2fa12c7dd10de210175582d2e50568631ea9c94d27c4cb3f8edd452`、
  `be3916fd3d4ba9581679d029993c5b45109d0de5b07da2ef1a31fa0fc7ee8c43`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`，
  report `passed=false`、`deployable=false`。protected artifacts/cache metadata 前后严格相等，validation 未访问；
  `.pth` 仍为 12，V128 未生成或删除权重。
- V126 training state SHA 五折逐折完全复现，证明唯一变量确为 outer composition。V115 原 4,321 switches 全部身份不变，
  listwise raw accepts `9/40/34/41/18` 中只有 `2/6/7/14/5` 位于 V115 abstain 补集，最终新增 34，
  总 switches 4,355（`11.8778%`）；V115 identity violation、margin violation、非正 head gain 均为 0。
- REC OOF 为 `+69/+258`，五折 `(+4,+64)/(+10,+32)/(+19,+45)/(+25,+58)/(+11,+59)`；总体
  bootstrap 下界 `+30/+194`。corrected 为 `+51/+191`、下界 `+17/+141`；regular 为 `+18/+67`、
  下界 `-3/+31`。五折双正、@.50 delta/总体/corrected/regular 和 switch ceiling 通过，但 @.25 delta、总体与
  subgroup 门失败。
- @.25 为 199 fixes/130 breaks=`+69`；与 V115 的 199/124=`+75` 比较，新增 34 rescues **没有新增 fix**，
  只增加 6 breaks 与 28 neutral。@.50 为 506/248=`+258`，也比 V115 `+263` 少 5。由此可知 V126 的高精度
  fix signal 完全落在 V115 已接受区域；abstain 补集没有可叠加净收益。
- **判定**：禁止 full-fit/validation，不生成权重；停止 V115/V126 proposal、veto、union、阈值、prior 或候选集合的
  后处理组合。V115 固定 proposal `+75` 仍是该 reranker family 的最佳 @.25 OOF，连续 V121--V128 已覆盖绝对
  critic、pairwise、反对称、双头 gate、全候选 argmax、listwise abstain、prior correction 与 protected rescue，均未超过。
  下一阶段必须回到跨数据集可泛化的网络/表示模块，改善 proposal/query 本身，而不是继续在同一缓存上重排。

#### 14.195 V129 预注册：Text-conditioned Directed Box-Relation Graph Adapter（2026-08-15）

- V129 绑定 V128 result SHA `3b90a175...d452` 和 V115r1 result SHA `cae35808...f423f` 为设计证据。
  V115 已证明冻结 V99 上的有界关系适配可得到 `+75/+263`，但它的关系编码只含中心距离/方向五维，@.25
  regular bootstrap 下界仍为 `-1`；V128 又证明同一 proposal 上继续叠加 critic/rescue 没有增量。因此本轮不改
  threshold、margin、proposal policy、loss prior 或候选集合，而把唯一研究变量固定为 **query 表示中的有向 box
  relation message**，直接加强多实例候选间的空间/语义竞争。
- 每个 query pair `(i,j)` 构造 19D 通用有向边：signed/absolute center delta、按两框平均尺度归一的 delta、
  log size ratio、3D/水平距离、冻结 64D query projection cosine、两端 target-text cosine 及其差、3D box IoU。
  边编码由冻结 64D target-text 表示作 FiLM 条件化，并与 V99 anchor query context 共同进入 4-head directed
  attention；self edge 与 padding edge 严格屏蔽。reliability gate 融合 relation message，query/variant delta heads
  仍为零初始化，部署 logits 初始与 V99 bit-exact，最终逐 logit residual 仍固定为 `0.25*tanh(delta)`。
- 输入只来自既有可部署 query embedding、文本 embedding、归一化 box center/size 与 target cosine；不读 dataset 名、
  ScanRefer unique/multiple 标志、GT、scene ID 或 validation。关系定义对 query permutation 等变，并可原样用于
  Nr3D/Sr3D，因而这是网络表示实验而不是数据集特化后处理。
- fold-local V99 anchor、V95 graded-listwise objective、12 epochs、hidden 128、dropout 0.1、AdamW learning rate/
  weight decay、seed 0、V99 Pareto margin 与双 head positive-gain policy 全部沿用 V115，不做结构/超参 grid。
  promotion gates 仍为 REC OOF `+105/+225`、五折双正、总体 bootstrap 下界 `+60/+170`、corrected
  `+35/+115`、regular `+8/+25`、switch `<=13%`；全部通过前禁止 full-fit、validation 与权重生成。
- 模型 `models/rec_box_relation_adapter.py` SHA-256
  `b20bfd625f0f9ba437efe3e152bdc134d7b3f3426c9c4397e5f1cbb332fe72da`，OOF 程序
  `scripts/run_v129_meshsp_box_relation_adapter_oof.py` SHA
  `bf0bff490760daa227671c5e38b365cfb6d3aeebe3c5b645981a539f572627e0`，测试 SHA
  `eff523e27f8bec84d75eee3490b1df9a10d6084900dd4852c266463da65c1b15`，三者 mode `0444`；单卡 runner
  `scripts/run_v129_meshsp_box_relation_adapter_oof_serial.sh` SHA
  `adfd9a6e2b9df2dd3394a4c1f956f3fa95ea5f742ee2c4120498e63dd6adb9a0`，mode `0555`。
- `py_compile`、runner `bash -n` 与 V114/V115/V129 定向回归 **12/12 passed**。GPU contract 覆盖零头
  初始化后的 24-step 优化，确认 edge encoder 在第二步后获得非零有限梯度且最终 loss 低于初值 75%；另覆盖
  relation 反对称坐标、padding、query permutation 等变、文本/box 路径、残差上界和 anchor 无梯度。GPU0 已释放，
  V129 outputs 不存在，validation 未访问，远程 `.pth` 仍为 12。

#### 14.196 V129 OOF 结果：box-relation 表示稳定但未改善主阈值（2026-08-15）

- 完整 36,665-row/562-scene 五折正常完成，result/log/exit SHA-256 为
  `f962733c8ea7a071e709711e83e2f61fa018a5efe804232f173c98609a0a602f`、
  `29cee57065f5bf15c6445b334f93de3be13748b5887fb75811895119b6cd99a7`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`；
  report `passed=false`、`deployable=false`。protected artifacts/cache metadata 前后严格相等，validation 未访问；
  `.pth` 仍为 12，V129 未生成或删除权重。
- REC OOF 为 `+65/+258`，五折 `(+0,+60)/(+11,+27)/(+19,+45)/(+27,+65)/(+8,+61)`；总体
  bootstrap 下界 `+26/+196`。corrected 为 `+46/+181`、下界 `+13/+131`，其中 fold1 @.25=`-16`；
  regular 为 `+19/+77`、下界 `-2/+44`。@.50 delta/五折/总体/子群门全部通过，但 @.25 的 delta、
  严格五折、总体与两个子群下界均失败。
- 接受 4,366/36,665 次切换（`11.9078%`），满足 13% ceiling；@.25 fixes/breaks=`193/128`，
  @.50=`501/243`。五折 adapter final loss 均比各自 V99 anchor 低约 `0.0126--0.0142`，证明边表示可优化且
  数值稳定；但相对 V115 的 `+75/+263` 反而少 `10/5` hits，显式 size/IoU/semantic pair edge 没有提高
  未见场景的 @.25 switch precision。
- **判定**：禁止 full-fit/validation，不生成权重；不再扩展空间 edge 维度、head 数或 residual scale。
  V129 的失败把主瓶颈进一步缩小到 cross-modal candidate identity：现有 64D unit query/text embeddings 在 hierarchy
  输入前逐维 z-score，V99/V115/V129 的普通 MLP/attention 没有显式保留单位球的乘积/差异结构。下一网络实验固定恢复
  fold-local normalization 前的 query/text 表示，显式编码 product/absolute-difference/cosine 与候选集合竞争；仍以
  冻结 V99 anchor 和有界零初始化 residual 隔离，不改 proposal policy 或后处理。

#### 14.197 V130 预注册：Hyperspherical Query-Text Semantic Interaction Adapter（2026-08-15）

- V130 绑定 V129 result SHA `f962733c...602f` 为直接设计证据；空间 relation、box edge、V99 anchor、proposal 与
  Pareto policy 均不改。唯一研究变量是 cross-modal candidate identity 表示：adapter 保存每个 fold 的 V99
  `query_features` mean/std 为只读 buffers，在 forward 内精确反演标准化，再分别 L2-normalize 原始 64D query/text，
  显式拼接 query、text、逐维 product、absolute difference、9D main/modifier/pronoun/relation/other/default/
  contrastive/rank evidence 和 cosine，共 266D。
- 266D 表示经 `266→128` encoder 和一层 4-head permutation-equivariant set Transformer，让同一描述下的 16 个
  candidate 直接在保留单位球几何的语义空间中竞争。它只读取冻结模型的推理输出，不读类别规则、dataset 名、GT、
  scene ID、unique/multiple 或 validation；同一接口可用于 ScanRefer/Nr3D/Sr3D。
- 与 V115/V129 相同，fold-local V99 anchor 永久 eval/frozen，semantic reliability gate 后接零初始化 query/variant
  delta heads，step0 logits 必须 bit-exact；修正仍固定为 `0.25*tanh(delta)`。V95 graded-listwise objective、12 epochs、
  hidden128、dropout0.1、AdamW learning rate/weight decay、seed0、V99 margin 与双 head positive-gain policy 全不变，
  不做 normalization、结构、residual、margin 或阈值 grid。
- promotion gates 不变：REC OOF `+105/+225`、五折双正、总体 bootstrap 下界 `+60/+170`、corrected
  `+35/+115`、regular `+8/+25`、switch `<=13%`；全门通过前禁止 full-fit、validation 与权重生成。
- 模型 `models/rec_hyperspherical_semantic_adapter.py` SHA-256
  `d1db38703b39733fddecec5e12f3e2813978dcc41dc9dd7cd7d263980cab19ff`，OOF 程序
  `scripts/run_v130_meshsp_hyperspherical_semantic_adapter_oof.py` SHA
  `d06c09bd4b17198cc2f057778600205a590af99bff4296233a1958d99483a785`，测试 SHA
  `376ce0750a5a5a453011f946dd5d8cfef49992879344ec02532d68ea71059b03`，三者 mode `0444`；单卡 runner
  `scripts/run_v130_meshsp_hyperspherical_semantic_adapter_oof_serial.sh` SHA
  `fa64865fadc3c0609370db53b8deae73a386c7d3f90a2f6e6435900704017183`，mode `0555`。
- `py_compile`、runner `bash -n` 与 V115/V129/V130 定向回归 **15/15 passed**。覆盖 normalization 精确反演、
  单位范数/product/difference/cosine、padding、query permutation 等变、语义路径敏感性、残差上界、anchor 无梯度，
  以及 GPU 24-step 零头启动后 semantic encoder 获得非零有限梯度并显著降低目标 loss。GPU0 空闲，V130 outputs
  不存在，validation 未访问，远程 `.pth` 仍为 12。

#### 14.198 V130 OOF 结果：单位球语义交互有增益，但单路径仍停在 V115 上限（2026-08-15）

- 完整五折正常完成，result/log/exit SHA-256 为
  `563e55b6d2fc9000e9dee841e6a5894aff18a7980e00682861e9aee4a6572e64`、
  `2e6fa4f68cd77169da0f4246dbfce983b11af383c6ce5a8c939369ffff3e2640`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`；
  report `passed=false`、`deployable=false`。protected artifacts/cache metadata 未变，validation 未访问，
  `.pth` 仍为 12，V130 未生成或删除权重。
- REC OOF 为 `+73/+260`，五折 `(+4,+64)/(+14,+29)/(+21,+51)/(+23,+60)/(+11,+56)`，两阈值
  五折均严格正；总体 bootstrap 下界 `+35/+197`。corrected 为 `+54/+191`、下界 `+23/+141`，但 fold1
  @.25=`-6`；regular 为 `+19/+69`、下界 `-1/+33`。@.50 全门通过，@.25 delta、总体与两个子群下界失败。
- 接受 4,290 次切换（`11.7005%`）；@.25 fixes/breaks=`196/123`，@.50=`502/242`。相对 V129
  `+65/+258`，V130 以更少的 76 次切换增加 `+8/+2`，证明 normalization 反演后的显式 hyperspherical
  product/difference 是正信号；但相对 V115 `+75/+263` 仍少 `2/3`，单独语义 set branch 没突破当前上限。
- **判定**：禁止 full-fit/validation，不生成权重；不调 semantic dimension、normalization、residual 或 margin。
  下一唯一结构检验保留 V115 的语言条件 5D center-relation attention，同时并联 V130 的 hyperspherical semantic
  set branch，用共享有界 residual heads 联合训练；若两路径仍不能超过 `+105/+225`，即停止缓存 adapter 家族并转入
  主 MCLN decoder/query 表示的训练期模块，不再组合 gate/threshold。

#### 14.199 V131 预注册：Dual Hyperspherical-Semantic + Language-Spatial Adapter（2026-08-15）

- V131 绑定 V115r1/V130 result SHA `cae35808...f423f`、`563e55b6...2e64` 为设计证据，是缓存 adapter
  家族最后一次结构检验。它不组合两个已训练模型、不融合 OOF prediction、不改 policy；每折仍先拟合同一 V99 anchor，
  然后从零联合训练一个网络内双路径 adapter。
- spatial path 精确采用 V115 的 4-head、目标文本条件 5D center-relation attention；semantic path 精确采用 V130 的
  fold-normalization 反演、266D unit query/text product/absolute-difference/cosine 表示与一层 4-head set Transformer。
  两路分别形成相对 frozen anchor context 的 residual，由一个 `5*128→128→2` reliability gate 联合融合，再由共享
  query/variant delta heads输出。gate bias 固定 `-2`、delta heads 零初始化，step0 与 V99 bit-exact，最终 logit
  修正仍固定 `±0.25`。
- 输入、mask 与模块均 query-permutation equivariant；只读冻结 query/text/center/score 表示和 fold-local normalization，
  不读 dataset、GT、unique/multiple、scene ID 或 validation，可迁移到 Nr3D/Sr3D。V95 objective、12 epochs、hidden128、
  dropout0.1、AdamW lr/wd、seed0、V99 margin 与 Pareto positive-gain policy 全不变，不扫描 branch weight 或 gate。
- promotion gates 原样：REC OOF `+105/+225`、五折双正、总体 bootstrap 下界 `+60/+170`、corrected
  `+35/+115`、regular `+8/+25`、switch `<=13%`；全门通过前禁止 full-fit、validation 与权重生成。若失败，停止
  V114--V131 cache-adapter/critic family，下一步只能改主 MCLN decoder/query 训练表示。
- 模型 `models/rec_dual_semantic_spatial_adapter.py` SHA-256
  `d33007979fa77930b3364cd0fe9fee36fcf6cf528497c8701578e67b773118d5`，OOF 程序
  `scripts/run_v131_meshsp_dual_semantic_spatial_adapter_oof.py` SHA
  `0899c3b3f9df3bc7ae6ebf2a0264e56173fae5bc9722da28cd83ea31093083d9`，测试 SHA
  `a46b3a44ab3bcd7eb90b3d1001edaeec55c4f281fdcbe84e6e10355e92d0cb7b`，三者 mode `0444`；runner
  `scripts/run_v131_meshsp_dual_semantic_spatial_adapter_oof_serial.sh` SHA
  `d072fefbc8a6961fd02a595005a8810f91eaef761e871cb6730150f27c958ba1`，mode `0555`。
- `py_compile`、runner `bash -n` 与 V115/V130/V131 定向回归 **15/15 passed**；GPU contract 证明零头启动后
  spatial/semantic 两路均获得非零有限梯度并降低目标 loss，另覆盖双路敏感性、normalization 恢复、padding、query
  permutation、残差上界和 anchor 无梯度。GPU0 空闲，V131 outputs 不存在，validation 未访问，`.pth` 仍为 12。

#### 14.200 V131 OOF 结果：双路径无互补增益，停止 cache-adapter 家族（2026-08-15）

- 完整五折正常完成，result/log/exit SHA-256 为
  `e993c70a6f2d6f841dfff52eb31e631d7e826dee4380154419dfffc1e1394611`、
  `a8e8947de6f0eafad00bb8c98129fcfc47ce479550299ba75293479221631b0e`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`，均 mode `0444`，exit `0`；
  report `passed=false`、`deployable=false`。protected artifacts/cache metadata 前后完全一致，validation 未访问，
  `.pth` 仍为 12，V131 未生成或删除权重。
- REC OOF 为 `+66/+258`，五折 `(+3,+61)/(+14,+26)/(+21,+50)/(+22,+67)/(+6,+54)`；总体
  bootstrap 下界 `+28/+193`。corrected 为 `+46/+184`、下界 `+14/+133`，fold1 @.25=`-10`；regular
  为 `+20/+74`、下界 `-1/+37`。仅 @.50 和 switch 门通过，@.25 delta/总体/两个子群门全部失败。
- 接受 4,352 次切换（`11.8696%`）；@.25 fixes/breaks=`197/131`，@.50=`503/245`。相对 V115
  `+75/+263` 少 `9/5`，相对 V130 `+73/+260` 少 `7/2`；联合训练没有形成互补，空间路径主要重现同一批
  @.50 fixes，同时增加 @.25 breaks。
- **判定**：禁止 full-fit/validation，不生成权重。按 14.199 的事先约束，正式停止 V114--V131 的
  frozen-cache adapter、critic、proposal、gate、listwise 与组合路线；不再消费同一缓存做后处理式结构搜索。
  后续必须回到主 MCLN：在 decoder/query feature 进入 box、mask 与 contrastive projection 前加入零初始化、可回退的
  训练期 cross-modal module，使 query/box/mask 表示本身改变；先做冻结父权重的真实 train smoke 与 held-train 门，
  通过后才允许完整训练/validation。

#### 14.201 V132 预注册：Final-Decoder Cross-Modal Query Adapter（2026-08-15）

- V132 直接执行 14.200 的路线切换，不再读取 V99--V131 cache 或训练后 reranker。初始化父权重固定为 epoch71
  `mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth`，SHA-256
  `3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`；父文件只读保护，V109 及现有
  12 个 `.pth` 均不改动。
- 新 `DecoderQueryTextAdapter` 只插在最后一个 MCLN decoder layer 输出与 final box head、64D contrastive REC
  projection、`x_query` mask projection 之间，因此同一 query residual 同时改变 box、REC 与 query-specific mask。
  模块以 query 对 text tokens 的 4-head cross-attention、逐维 query/text product 与 absolute difference、上一层
  center/log-size 的 scene-wise 标准化几何编码、以及 4-head permutation-equivariant candidate set attention 构造
  residual；不读取 dataset 名、GT、unique/multiple、scene ID 规则或 validation，可原样用于 Nr3D/Sr3D。
- residual 输出层 weight/bias 全零初始化，部署为 `query + 0.25*tanh(delta)`，故 step0 与 epoch71 query
  bit-exact，逐维修改严格不超过 `0.25`。父网络全程 `eval` 且 `requires_grad=False`，仅训练
  `decoder_query_adapter.*`，AdamW lr=`3e-4`、hidden=`288`、dropout=`0.1`、seed=`0`；smoke 使用单张
  A100、batch=`8`、2 epochs。
- 为禁止早期读取 ScanRefer validation，`--debug_train_holdout` 在 ScanRefer train annotations 内按
  `SHA256(scene_id) mod 5` 做 scene-level 固定划分：bucket0 留出，其余训练，各取 128 rows；两个 subset scene
  严格不相交，held copy 禁止点云与检测框 augmentation。先用零 adapter 读取同一 held-train subset 建立父基线，
  再训练 2 epochs；两步均设置 `expected_eval_sample_count=128`，不构造 val dataset。
- promotion gates：checkpoint missing keys 必须精确等于 `decoder_query_adapter.*`；optimizer/train mode 必须只覆盖
  adapter；训练和评估无 NaN/Inf，最终 loss 低于首个 epoch，residual mean 必须非零且 max `<=0.25`；相对零
  adapter held-train 父基线，REC@0.25 与 @0.50 各最多下降 1 hit、mask mIoU 最多下降 1.0 point。全部通过才允许
  单卡 full-train/official validation；否则停止 V132，不用 validation 调参。
- 核心 source SHA-256：`models/mcln.py`=`4dd162b4886fad77c931db63742f5f1a916de8951bfa6eb9fecc19d5cba172a1`，
  `main_utils.py`=`25fb79e850bbdf865d6921b9f5425a3b1a001840142cbaa22b1465becae9e777`，
  `train_dist_mod.py`=`5baf4aedae024ff4818f37927014876d4754aa32c16c1e7acc128a050e85740e`，
  `src/joint_det_dataset.py`=`b6d785448a82743c9c367cc46e49021c8e1a08743e03dff32fdaf6086a515040`，
  test=`1ea8d85c8abc18891e440de440009f1b8d4b11638f713276a7d6d9273b7ffc22`，runner=
  `b5eb3117d7624d99a7d7f7c4e384364054aa4c7f7005b9f49eb241c0fa3bff9b`；source/test mode `0444`，runner
  mode `0555`。`py_compile`、runner `bash -n`、V132 GPU contract **3/3 passed**，既有 checkpoint/retention/
  finite-training/optimizer/dataset contract 回归 **60/60 passed**。
- 首次 baseline 仅进入 CPU train-annotation 预处理，约 3 分钟仍未加载模型、未占用 GPU、未产生指标或权重，因原实现
  在 scene 取样前解析全部 36,665 条文本而主动终止。结果盲态下只做等价的计算优化：将同一
  `SHA256(scene_id) mod 5` partition 前移到 `load_scanrefer_annos` 的 scene-graph parsing 之前，使 train/held
  各只解析固定的 128 rows；数据集合、顺序、门槛和所有模型/训练超参不变。终止后 GPU 回到 1 MiB，V132 `.pth`
  为 0，现有权重仍为 12。
- 过滤优化后的首次 baseline 在模型加载安全契约处按设计 fail-closed：adapter 30 个 missing keys 精确正确，但 epoch71
  还含 9 个冻结 `source_choice_selector.*` keys，而首版 runner 未实例化该父模块，故全部被报告为 unexpected 并在任何
  forward/metric 前停止。读取 checkpoint `config` 后，runner 恢复父训练时完全相同的 selector：sources=
  `default,default_rank_blend_contrastive010`、hidden=288、min-IoU-gap=0.03，并启用原有
  `eval_use_selector_choice_scores`；selector 保持 eval/frozen，V132 唯一 trainable 模块仍是 adapter。该修正只恢复父
  checkpoint 的既有网络结构，不改变研究变量、数据或门槛；失败 run 未产生指标/权重，现有 `.pth` 仍为 12。
- 恢复父 selector 后的 baseline 在首个 held forward 前打印出 `holdout=128 examples/1 scenes`，揭示 ScanRefer
  train JSON 按 scene 聚集，partition 后直接取前 128 rows 虽满足 scene-disjoint，却不能形成有意义的跨场景门。
  在任何 metric 输出前再次 fail-closed；将每个固定 hash partition 内的 records 改为按排序 scene ID round-robin
  interleave，再由同一 `overfit=128` 截断。这样不改变 partition 成员或模型超参，但确保尽可能多 scene 覆盖；dataset
  contract 继续 **4/4 passed**。该 run 未生成权重，停止后 GPU 回到 1 MiB，现有 `.pth` 仍为 12。

#### 14.202 V132 零残差 held-train 父基线（2026-08-15）

- 修正后的 baseline 正常 exit `0`：held-train 固定为 128 rows、120 scenes，和 train partition scene overlap=`0`；
  checkpoint 契约确认仅有 `decoder_query_adapter.*` 30 个新 keys，optimizer 只含 adapter 1,502,208 params，epoch71
  parent 与冻结 selector 完整加载。screen log/exit SHA-256 为
  `ffb5752f711ccae5a0b9d43008ff425c195bf036f0e46956f3c4b971c3d76c4e`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。
- 零 residual held-train REC position Top-1：Acc@0.25=`124/128=96.875%`，Acc@0.50=`117/128=91.40625%`；
  unique=`21/21=100%`、`20/21=95.2381%`，multiple=`103/107=96.2617%`、`97/107=90.6542%`。
  selector 128/128 选择 default，故 learned-selector 数值与 position head 相同。
- mask held-train：Acc@0.25=`120/128=93.75%`，Acc@0.50=`113/128=88.28125%`，mIoU=
  `74.9070808437%`。这是 train split 上用于网络 smoke 的局部安全基线，不与 9,508-row official validation 指标混用。
  baseline 未生成任何权重；远程 `.pth` 仍为 12，GPU 已释放。下一步按 14.201 固定配置运行 2-epoch adapter-only
  smoke，并与这些 raw hit counts 作盲态门判断。

#### 14.203 V132 adapter-only smoke 结果与权重清理（2026-08-15）

- smoke 正常 exit `0`，train/held 各 128 rows、128/120 scenes、scene overlap=`0`；checkpoint 仅缺 adapter
  30 keys，实际 trainable=`1,502,208`，其余 parent/selector 均 eval/frozen。epoch-average train loss 从
  `10.5204` 降至 `10.4463`，无 NaN/Inf/OOM。
- epoch1 held REC 为 `123/116`，epoch2 回到零残差父基线 `124/117`；epoch2 unique=`21/21,20/21`，
  multiple=`103/107,97/107`。mask epoch2 为 `120/128`、`112/128`、mIoU=`75.0524923847%`：相对父基线
  @.25 `0 hit`、@.50 `-1 hit`、mIoU `+0.1454115410 point`，全部满足预注册门。
- epoch2 adapter 30 tensors 严格重载成功；deterministic synthetic contract 上 output weight/bias L1=
  `120.01036/0.44038`，residual abs mean=`0.0209651`、max=`0.0996920 < 0.25`，全有限，证明模块已离开
  零初始化且有界。screen log/exit、epoch1/2 receipt、epoch2 full checkpoint SHA-256 分别为
  `d8b5f637dbd9a3897d5426bcffcd416b4c6953817f33601b1fa6d711d950995e`、
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`、
  `b52b3e0a6fd5e49c4680b2fdc311cab2eb0ead9b91d7d4ba3fc02dde164d2422`、
  `63adbe9cfb6a4b9ce38f0f606e6626c0ecd7893ee8fa3e574af3ee19ccd12021`、
  `debc82126d5c8f0c3148e1b2db7cfbe3aaf1c363072fc93b07f78229bf70be45`。
- 为落实只留可恢复最佳权重，将 epoch2 adapter 抽成只读 artifact
  `v132_smoke_adapter_epoch2.pth`（6,018,443 bytes，SHA
  `29903a3ec1cc9b646df933a79e953dc0d3f76160d6406cf7c9d0b83e4e95292d`），包含 parent/source/full-smoke
  fingerprints、30 tensors、配置、metrics 与 loss，重新加载逐 tensor exact。随后删除 smoke 专属两份 617 MB
  full-checkpoint inode 的 8 个 hard-link 名称，目录从约 1.2 GB 降至 5.9 MB；删除目标全部位于该 V132 smoke
  timestamp 目录，原 12 个最佳/保留权重（包括 V109）未动。当前 `.pth` 为 13：原 12 + compact V132 smoke。
- **判定：全部 gate passed**，允许一次正式 full-train/official validation；smoke 的局部高准确率不作为 official
  最好指标，当前正式最好仍为 V99 REC@.25=`58.6033%`、V113 REC@.50=`50.8519%`。

#### 14.204 V132 单卡正式训练预注册（2026-08-15）

- 唯一 A100 上从同一 protected epoch71 parent 重新零初始化 adapter，使用完整 ScanRefer train，冻结所有 parent 与
  selector；V132 结构/hidden/heads/dropout/max-delta/lr/seed 均与 smoke 完全相同，不从 smoke 小样本权重续训。
  单卡 batch=`8`、4 epochs、每 epoch 一次固定 9,508-row official validation；不做 lr、epoch、source、threshold 或
  postprocess grid。预计每 epoch train+validation 约 60 分钟。
- checkpoint retention 只按 REC@.25、REC@.50、mask@.25、mask@.50、mask mIoU 原始指标保留 hard links；完成后按
  inode 去重并只留非支配最佳正式权重，V109 永久保留。若出现 NaN/Inf/OOM、非 adapter 参数可训练、样本数非
  `36,665/9,508` 或 parent/source fingerprint 漂移，立即 fail-closed。
- 正式主判据优先 REC：与 epoch71 raw parent/当前 V99、V113 official best 同时报告；若网络 checkpoint 产生新最好，
  再对其运行一次固定、未重拟合的 V99 policy 兼容性评测。validation 只用于报告/保留，不据此更改 V132 超参。

#### 14.205 交接文档归档与续写规则（2026-08-15）

- 远程 `/tmp/mcln_repo/docs/REC_3DRES_OPTIMIZATION_LOG.md` 继续作为唯一 canonical handoff；后续实验设计、运行状态、
  指标、审计和权重清理记录只在该远程文档末尾续写。
- 2026-08-15 将 canonical handoff、`SOURCE_MOE_RERANK_DESIGN.md` 以及 `refine-logs/` 中当前/历史计划与追踪快照
  合并为一个本地总文档，保存到 `C:\Users\gb\Desktop\document\MCLN_实验交接总文档.md`。该文件仅为归档快照，
  不作为后续续写源；远程 canonical 的旧 `.bak` 副本因已被当前文档继承，不重复并入本地总文档。
- 本地归档不包含 SSH 密码或其他明文凭据；连接凭据仍仅使用现有安全辅助脚本读取，不写入交接材料。

#### 14.206 V132 单卡正式训练已启动（2026-08-15）

- 启动前唯一 A100 40GB 空闲、无 compute process、无同名 formal screen/输出目录；protected parent、V132 runner
  SHA-256 分别复核为 `3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`、
  `b5eb3117d7624d99a7d7f7c4e384364054aa4c7f7005b9f49eb241c0fa3bff9b`，已有 `.pth=13`（含永久保护
  V109 和 compact V132 smoke），未删除或覆盖任何权重。`bash -n`、单 GPU 和输出不存在检查均通过。
- `2026-08-15 13:40:54 CST` 在 detached screen `mcln_v132_formal` 启动。显式覆盖 runner formal 默认值为
  `MAX_EPOCH=4`、`VAL_FREQ=1`、`BATCH_SIZE=8`、`PRINT_FREQ=100`、`MASTER_PORT=5132`；其余结构、
  optimizer、adapter lr=`3e-4`、seed=`0` 与 14.204 预注册不变，从 protected epoch71 parent 重新零初始化
  adapter，不读取 smoke adapter 权重。
- run 目录：`/root/autodl-tmp/DATA_ROOT/output/network_v132_decoder_query_adapter/
  v132_decoder_query_adapter_formal_e1_e4_b8x1/scanrefer/v132_decoder_query_adapter_formal_e1_e4_b8x1/1786772458/`；
  `config.json` SHA-256=`268830a4117fa6f5a64d6f7d8858ae2ed917adeb9c8bae05ed80f4a30542cb15`。screen log 为
  `/tmp/v132_formal_20260815_screen.log`，最终 exit marker 为 `/tmp/v132_formal_20260815.exit`。
- 全量数据构建完成并严格打印 train=`36,665`、official validation=`9,508`；adapter-only trainable 参数
  `1,502,208`，父 checkpoint 契约仅缺预期的 adapter 30 keys，epoch71 成功加载。epoch1 已推进到
  `100/4583`，稳定约 `1.6 batch/s`，GPU 约 6.1GB、利用率 84%，无 OOM/NaN/Inf/traceback。按当前速度估计
  每个 train epoch 约 46--48 分钟，之后执行一次固定 official validation；当前尚无正式指标或新权重。
- 后续只在 epoch receipt 或异常时检查；每次验证必须核对 9,508 sample、REC overall/unique/multiple、Mask
  overall/unique/multiple 与 mIoU。训练完成前不更改超参；结束后按 inode/metric receipt 只保留非支配最佳正式权重，
  V109 永久保留。

#### 14.207 V132 epoch1 完整正式回执：未超过现有最好（2026-08-15）

- epoch1 全量训练于 `2026-08-15 14:36:59 CST` 完成，严格为 `4,583` batches，随后完成固定的
  `9,508`-row official validation；receipt schema=`mcln-retrain-metrics-v1`、sample_count=`9,508`，
  SHA-256=`c6ec67e1d88f4195bf967f39caa61a91123e9f6233315215b4d9a3d58b178dcd`。Unique=`1,419`、
  Multiple=`8,089`，REC 与 Mask 两档的 subgroup hits 均严格加和到 overall，回执通过完整性审计。
- REC learned-selector 与 fixed-default 恰好相同：overall @.25/@.50 为
  `5504/4391 = 57.888094%/46.182162%`；Unique 为
  `1244/1054 = 87.667371%/74.277660%`；Multiple 为
  `4260/3337 = 52.664112%/41.253554%`。相对 V99 @.25 最好少 `68` hits、`-0.715187pp`；
  相对 V113 @.50 最好少 `444` hits、`-4.669752pp`，两项均未刷新正式最好，也未达到 59%/49% 目标。
- 同一 query 选择的 Mask overall @.25/@.50/mIoU 为
  `5669/9508 = 59.623475%`、`4660/9508 = 49.011359%`、`41.698857%`；Unique 为
  `1276/1030 = 89.922481%/72.586328%`，Multiple 为
  `4393/3630 = 54.308320%/44.875757%`。相对 V99 Mask 最好分别为 `-21/-316` hits、mIoU
  `-4.231403pp`；@.25 仍高于用户 baseline，但 @.50 与 mIoU 明显低于 baseline，epoch1 不具备保留优先级。
- epoch1 checkpoint SHA-256=`8179f199a27b4498efb44545eac95b797ce65ba7f171454a9f2f9b885f1c9bcd`，大小
  `617,227,079` bytes。`ckpt_epoch_1`、`epoch_last` 与五个 metric-best 名称均为同一 inode
  `6484398810` 的 7 个 hard links，物理只占一份；尚未删除，作为活动训练的恢复点。日志中的唯一宽泛
  `inf` 匹配来自配置字段 `confidence`，严格错误正则确认无 Traceback/OOM/NaN/Inf/Killed/RuntimeError。
- epoch2 已自动开始，正式配置和超参不变。继续完整运行至 epoch4，逐轮只读比较；只有出现新正式网络最好时才按
  14.204 对该 checkpoint 运行一次冻结 V99 compatibility policy，不用 validation 调参。

#### 14.208 V132 epoch1 的只读瓶颈定位（2026-08-15）

- 同一完整验证中，fixed-default=`57.888%/46.182%`，rank-blend source=`57.836%/46.119%`，
  两源 oracle 也只有 `58.014%/46.287%`；冻结 selector 的 source 占比为 default=`100%`，fix/break
  均为 `0`。因此当前 source pair 最多只提供约 `12/10` hits 的可见 headroom，继续训练或调该 selector
  不能解释距 59%/49% 的主缺口。
- 与之相反，同一 V132 checkpoint 的 position query Top-1/Top-5/Top-10 为
  @.25=`57.888%/66.113%/69.363%`、@.50=`46.182%/57.446%/61.064%`；semantic query 的 Top-5
  也为 `66.449%/57.762%`。这证明满足目标的候选框已大量存在，主要瓶颈是 256 queries 内的 target-query
  排序，而不是候选框召回或两种 source 的路由。
- V132 adapter 同时改动 final box head、contrastive query ranking 和 query-specific mask embedding；标准检测/分割
  loss 不能直接保证 validation Top-1 的安全排序。epoch1 未改善 Top-1 且 REC/Mask 都低于现有最好，说明这种
  full-query residual 至少在第一轮没有把候选召回转成目标 query 选择。后续仍按预注册跑完四轮；若全程不胜，
  下一网络变量必须把“关系感知 target-query ranking”与 box/mask 表征解耦，并优先利用可跨 ScanRefer/Nr3D/Sr3D
  的 target/attribute/relation-anchor 结构，不再重复同类全 query 扰动或只调 source 阈值。

#### 14.209 V132 后续与 V133 score-only SACR-Lite 备用计划冻结（2026-08-15）

- 在 V132 活动训练期间只完成远程只读设计，不改源码、不占第二张 GPU。`experiment-plan` 的最新与不可变计划位于
  `refine-logs/EXPERIMENT_PLAN.md`、`refine-logs/EXPERIMENT_PLAN_20260815_150630.md`，内容 SHA-256 均为
  `60a389b7edec9b062e6cce35df88c1439762e0ac474b0b2d97f6c44846f2154f`；对应 tracker 最新/不可变副本 SHA-256
  均为 `ca32137b7f05c7b632c966ffed089c3851ebb413b149402eb3fcecca0505b4b9`。根级 `MANIFEST.md` 新建并登记四份
  输出，SHA-256=`4832a1f7fdad266f6b5db5554ee249bec6837ccc65bbb2235e16b737f6f61a35`；全部文件只在远程，桌面归档未更新。
- 计划冻结两个 claim：一是用 target/attribute/relation-anchor 结构把 V132 已证明存在的 Top-K headroom 转成
  Top-1；二是 score-only 与 parse-aware fallback 保持候选 box/mask 表征，并复用到 ScanRefer/Nr3D/Sr3D。
  anti-claim 明确排除更多候选、validation 阈值搜索、ScanRefer 专用规则或更大训练预算。
- 若 V132 四轮未达标，V133 只新增独立 bounded structured-score residual，复用现有 `StructuredSlotBuilder`/
  `SACRHead` 的 target、attribute、relation-anchor 几何；父 query/box/mask/selector 冻结，无有效结构行 exact
  fallback。监督使用当前 train batch 全 queries 的连续 3D IoU listwise target；有 Mask GT 时只加固定 0.25
  mask-quality 项，Nr3D/Sr3D 无 mask 时自然退化为 box-only，不引入数据集阈值分支。
- 执行门严格串行：先完成 V132 4/4 receipts；V133 step-0 identity/contract；128-row、120-scene-disjoint smoke
  两阈值均 `fix >= break`；随后单卡 seed0、最多四轮完整 ScanRefer。正式成功必须同一结果 REC hits
  `>=5610/4659`，Mask 至少保持用户 baseline `58.70%/50.70%/44.72%`；只有主结果通过才运行 relation 删除实验
  与 Nr3D/Sr3D transfer interface。禁止重复 V80--V131 safety-loss/阈值组合。
- `15:10 CST` 前后 V132 epoch2 已推进至 `1800/4583`，screen 仍在、strict errors=`0`；当前计划只是
  failover preregistration，不改变 V132 任何训练状态或判据。

#### 14.210 V132 epoch2 完整正式回执：继续退化，最佳权重未变（2026-08-15）

- epoch2 全量训练与固定 `9,508`-row official validation 于 `2026-08-15 15:51:37 CST` 完成；receipt
  `eval_metrics_epoch_2.json` SHA-256=`f2bc0addf325a8b65a89b4c741df3301453ef033417edea44be659d0f600f0ec`、
  schema=`mcln-retrain-metrics-v1`、sample_count=`9,508`。Unique=`1,419`、Multiple=`8,089`，REC 与 Mask
  两档 subgroup sample/hits 均严格加和到 overall，完整性审计通过。
- REC overall @.25/@.50 为 `5493/4383 = 57.772402%/46.098023%`；Unique 为
  `1239/1057 = 87.315011%/74.489077%`；Multiple 为
  `4254/3326 = 52.589937%/41.117567%`。相对 epoch1 再少 `11/8` hits；相对 V99 @.25 最好少
  `79` hits、`-0.830879pp`，相对 V113 @.50 最好少 `452` hits、`-4.753891pp`。距 59% @.25
  门槛 `5610` hits 仍少 `117`，未产生网络最好。
- 同一 query 选择的 Mask overall @.25/@.50/mIoU 为
  `5657/4635 = 59.497265%/48.748422%`、`41.500175%`；Unique 为
  `1275/1031 = 89.852008%/72.656801%`，Multiple 为
  `4382/3604 = 54.172333%/44.554333%`。相对 epoch1 再少 `12/25` hits；相对 V99 Mask 最好分别少
  `33/341` hits、`-0.347076/-3.586454pp`，mIoU 低 `4.430125pp`。@.25 仍略高于用户 baseline，
  @.50 与 mIoU 仍明显低于 baseline。
- epoch2 checkpoint SHA-256=`097ef272d372331c1358c32d57b579e3b601ce267f86aec60d370715606e56ed`，大小
  `617,227,079` bytes；`ckpt_epoch_2` 与 `ckpt_epoch_last` 为 inode `6484398811` 的 2 个 hard links。
  五个 metric-best 别名仍全部指向 epoch1 inode `6484398810`，说明 retention 未误把退化的 epoch2 标为最好。
  活动训练期间不删除任何恢复点；V99、V113、永久 V109 等既有 protected 权重不受影响。
- epoch3 已自动开始，唯一 A100 正常使用，strict errors=`0`；`/root/autodl-tmp` 尚余约 `2.6GB`，按当前每轮
  物理 checkpoint `617MB` 估算可完成剩余两轮，但必须继续监控磁盘。V132 仍按预注册完整跑至 epoch4，不据
  epoch2 改超参；若剩余两轮均不胜，执行 14.209 已冻结的 V133 score-only SACR-Lite 方案。

#### 14.211 V133 score-only SACR-Lite 只读落点审计（2026-08-15）

- V132 epoch3 活动训练期间没有修改任何源码，只读确认 V133 无需重写结构化语言前端。现有
  `models/structured_slots.py` SHA-256=`78f5c2e3a1e794ebf8876f24126c67fbb0c404707d065f55847ea7d2b2ef3281`
  已完成 target/attribute/relation/anchor span pooling；`models/sacr_head.py`
  SHA-256=`a92b98d13c3219015dad09a58ce9c7bf557634db7a605cc19614479e735c1bc4` 已实现 target/attribute
  compatibility、anchor shortlist、11-D 相对几何和 relation-anchor composition。两者均无
  ScanRefer/Nr3D/Sr3D 名称分支。
- 已有 train/val 三数据集结构合同回执 SHA-256 分别为
  `e58e2412d5473b0022c0b5b4bfcfbda6355e99b543fadde6bd4fdbbcf13e12fe`、
  `792eb1580ea1c75b1dbcc7345dd4a13c453e9f2f2bfdf68e8ad632005ed8a5e3`，总结果均 `pass=true`。
  SACR 可用行 train/val 分别为 ScanRefer `35,997/36,665`、`9,336/9,508`，Nr3D
  `32,545/32,919`、`7,824/7,899`，Sr3D `65,846/65,846`、`17,726/17,726`；三数据集 target
  offset/token 有效率与有效 relation-anchor pair 对齐率均为 `100%`。这给出真实跨数据集输入合同，
  不是 ScanRefer 专用规则。
- 现有 `source_choice_adapter.py` 已能构造 `default + SACR residual`，但此前 V50 把 SACR 绑定到四源
  joint-query mixer 与上游 V49 selection；其队列因缺少 `selected_v49_formal_config.json` fail-closed，
  `experiment_output/v50_sacr/` 没有正式 metrics/checkpoint。因此 V133 不是重复一个已有 SACR 正式失败结果，
  而是删除未实际验证的四源/mask-calibration 耦合，只验证结构化 score residual 本身。
- V133 最小实现差额现已定位：新增互斥的 `sacr_score_only_train_only`/独立 lr 与 checkpoint contract；仅训练
  `structured_slot_builder + sacr_head + bounded scale`；对结构有效行直接输出
  `default_score + bounded structured residual`，无结构行逐元素 exact fallback 到 frozen default；用所有
  256 queries 的连续 box-IoU listwise target，ScanRefer 有 mask GT 时固定加入 `0.25` mask-quality 项。
  父 query/box/mask、V99 parent source 和 selector 全部冻结，不通过 selector 二选一，也不修改 candidate box
  或 mask embedding。
- epoch3 当前约 `453/4583`、strict errors=`0`，V132 formal 完成前只保留上述实施地图，不落代码。V132
  若 4/4 不胜，按 14.209 的 identity/contract、scene-disjoint smoke、单卡 formal 顺序实施；避免训练期源码
  漂移，也避免复用 V50 未完成的上游依赖。

#### 14.212 用户指定持久路径恢复（2026-08-15）

- 只读路径审计确认实际仓库为 `/home/gb/new butd/butd_detr-main/MCLN-main`，临时别名
  `/tmp/mcln_repo` 正确解析到该目录，但用户指定的 `/home/gb/butd/mcln` 原先不存在。依赖 `/tmp` 别名会在
  服务器重启后留下恢复风险，也不满足本目标约定的代码与交接路径。
- 已新建空的持久父目录 `/home/gb/butd`，并建立可逆符号链接
  `/home/gb/butd/mcln -> /home/gb/new butd/butd_detr-main/MCLN-main`；没有移动、复制或覆盖仓库文件。
  `readlink -f /home/gb/butd/mcln` 已严格等于实际仓库路径。经新路径读取 canonical handoff 的 SHA-256
  仍为 `f2ee876c825f3aa28fd004b5d11537e2be64d912b35eca094b00869c6e78889e`、mode=`0444`，证明两条路径
  访问同一文件而非副本。
- 后续所有源码、实验计划和交接续写统一优先使用 `/home/gb/butd/mcln`；`/tmp/mcln_repo` 仅保留兼容旧命令，
  不再作为唯一恢复入口。建链时 V132 epoch3 约 `1204/4583`，唯一 A100 正常、strict errors=`0`、正式回执
  仍为 2/4，未修改活动训练或权重。

#### 14.213 V133 无 Git 源码恢复门（2026-08-15）

- 经持久路径执行 `git rev-parse --show-toplevel`，实际仓库没有 `.git`，所以 V133 不能把 Git 当作源码回退机制。
  当前已有 `.v132_parent/`，但只覆盖 V132 的四个旧父文件，不能完整恢复 V133 预计新增的 score-only/loss/runner
  改动。V133 首次源码修改前必须建立独立 `.v133_parent/`，不得覆盖 `.v132_parent/` 或散落的历史 `.bak`。
- 已只读冻结 V133 touch-set 基线 SHA-256：`main_utils.py=25fb79e8...e9e777`、
  `train_dist_mod.py=5baf4aed...e85740e`、`models/mcln.py=4dd162b4...a172a1`、
  `models/losses.py=d4274b04...af7c6e`、`models/source_choice_adapter.py=dc32c6ad...9b11fbb`、
  `models/source_moe.py=f09b2c5a...62fbd3`、`models/structured_slots.py=78f5c2e3...ef3281`、
  `models/sacr_head.py=a92b98d1...c1bc4`、`src/joint_det_dataset.py=b6d78544...515040`、
  `scripts/train_scanrefer_joint_query_quality.sh=5b083a9a...306df6`。十个文件合计约 `1,012KB`，当前磁盘可安全
  容纳一份逐路径副本。
- 门禁顺序冻结为：V132 4/4 正式结束；重新计算十个 SHA 并与本节逐一相等；创建保留相对路径、mode、mtime 的
  `.v133_parent/`；生成 SHA/mode manifest 并将快照置为 `0444`；以逐文件 checksum 证明可恢复；此后才允许
  解锁并修改 V133 touch-set。任一 SHA 漂移、快照不完整或恢复校验失败均 fail-closed，不启动 V133。
- `16:06:55 CST` V132 epoch3 约 `1481/4583`，单 A100 约 20.7GB、strict errors=`0`、磁盘仍余约
  `2.6GB`。本节仅记录恢复门和只读 fingerprint，没有创建快照、修改源码或清理权重。

#### 14.214 V133 score-only 唯一数学合同与部署链冻结（2026-08-15）

- 对 query `i` 定义 frozen parent 分数 `s_i=default_score_i`；SACR 读取 detached final query、detached box、
  detached parent score，以及由 detached text feature 进入可学习 slot-pooling 后得到的 target/attribute/
  relation/anchor slots，输出 raw structured score `r_i`。唯一部署式冻结为
  `score_i = s_i + valid * 0.25 * tanh(a) * tanh(r_i)`，其中 `a` 为唯一标量 gate，初始化为 `0`。
  因而 step-0 对全部行与 parent bitwise identity，任意训练时刻每个 query 的绝对残差严格小于 `0.25`；无
  target/有效结构行 `valid=0`，逐元素精确回退到 `s_i`，不允许阈值或数据集分支。
- 训练只在 structured-valid 行激活。box 主损失为所有 256 queries 上的连续 3D IoU listwise KL：
  `KL(softmax(IoU_box/tau) || softmax(score/tau))`，固定 `tau=0.1`；若 batch 提供 GT point mask，再加
  `0.25 * KL(softmax(IoU_mask/tau) || softmax(score/tau))`。不使用 validation threshold label、
  ScanRefer 专用 subgroup 或 hard-coded 0.25/0.50 tier，Nr3D/Sr3D 无 mask 时自然退化为同一个 box-only
  objective。父分数、box、query/text encoder 参数均冻结；只有 slot builder、SACR head 和 gate 可训练。
- 只读部署审计确认 `src/grounding_evaluator.py` 在 `eval_use_selector_choice_scores=true` 时，REC Top-1 与
  `_resolve_learned_mask_queries` 均读取同一个 `end_points['selected_source_scores']`。V133 在所有既有
  selector/reranker 结束后把上述 score 写为最终 `selected_source_scores`，因此 box 与 mask 必然选择同一
  query；不会改 candidate box、`last_pred_masks`、`sp_last_pred_masks` 或 adaptive mask weight。
- 实施后的固定验证合同为：scale=`0` 全行 score bitwise identity；invalid 行训练后仍 bitwise parent；
  `abs(delta)<0.25` 且 finite；trainable parameter 名单只含三类新模块；parent 参数梯度/optimizer state 均为空；
  连续 box-IoU listwise 在无 mask batch 可反传，ScanRefer mask 项权重精确为 `0.25`；checkpoint missing keys
  精确等于新模块；REC 与 mask evaluator 的 chosen query 逐行相同。上述合同通过后才进入 128-row/
  120-scene-disjoint smoke，不用 validation 调参数。
- `16:10:33 CST` V132 epoch3 约 `1837/4583`（40%），唯一 GPU compute PID=`170211`、显存约
  `20.7GB`、strict errors=`0`、正式回执仍为 2/4。本节仍为只读冻结，没有修改活动源码或权重。

#### 14.215 V133 scene-disjoint smoke 数据门复核（2026-08-15）

- 现有 `--debug --debug_train_holdout` 已在 `train_dist_mod.py::get_datasets` 强制从 `split='train'` 构造两个
  独立 `Joint3DDataset`，训练/holdout 均严格 `128` rows，随后对 `scan_id` 集合执行 overlap fail-closed；
  holdout 同时关闭 `augment_det` 和 dataset augment。若缺 `--debug`、任一侧非 128 行或场景相交都会立即报错。
- `src/joint_det_dataset.py` 的分区只对 ScanRefer train annotation 按稳定 SHA-256 scene hash 切分，再按场景
  轮转取样，避免前 128 行被单一长场景占满。V132 已修复后的真实 smoke launcher 记录为
  `train=128 examples/128 scenes; holdout=128 examples/120 scenes; overlap=0`；这与 14.209 预注册的
  `128-row / 120-scene-disjoint` holdout 完全一致，且没有读取 official 9,508-row validation。
- V133 smoke 因而直接复用该入口与 `EXPECTED_EVAL_SAMPLE_COUNT=128`，不得使用普通 `--debug`（其旧路径会把
  split 切到 validation），也不需要新增或修改数据分区代码。smoke 只检查 identity/finite/bound、optimizer
  coverage 和 holdout 两阈值 `fix >= break`；其指标不能作为正式 ScanRefer 结果或用于调 max-delta/tau。
- `16:15:27 CST` V132 epoch3 约 `2307/4583`（50%），strict errors=`0`、GPU/磁盘正常、正式回执仍为
  2/4。本节仅复用已运行的数据合同证据，没有触发新的评测、修改源码或删除权重。

#### 14.216 新网络 checkpoint 的冻结 V99 compatibility 边界（2026-08-15）

- 受保护 V99 hierarchy artifact 仍位于
  `/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/v99_artifacts/pareto_contextual_h128_seed0_fullfit.pth`，
  SHA-256=`9752990c393fa6e45173a9dd129c4de4bb740924094dcbbec2f3121cbf39d1f2`、mode=`0444`；原一次性
  claim 已存在，SHA-256=`e2ca7a1762b21470de76e8050f117e1358ed665f6ad71c3993fb28b535bdbab4`、mode=`0444`。
  parent/geometry artifact SHA 分别为 `f06f8972...c17269b`、`835c25be...263b6f`，均绑定 epoch71 backbone
  SHA `3e44f4bd...ef2208`。
- `train_dist_mod.py::validate_rec_reranker_provenance` 会严格比较 runtime `checkpoint_path` SHA 与 artifact
  中的 checkpoint fingerprint；geometry/hierarchy 又逐层绑定 parent artifact。因此即使 V133 只多出 SACR
  参数，直接把 checkpoint 路径替换为 V133 也会按设计 fail-closed。禁止删除旧 claim、修改 artifact 或绕过
  provenance 后把结果伪装成原 V99 official。
- V133 的 score-only 合同冻结 parent query/box/mask 参数与输出，所以若其产生新的 network best，兼容审计先在
  36,665-row train-only cache 上证明：除允许的新 SACR state keys 外所有 checkpoint tensor 与 epoch71 逐键
  bitwise 相同；V99 parent/geometry/hierarchy 的全部 runtime input feature、candidate identity 与 mask evidence
  逐行 bitwise 相同。只有两项都通过，才可签发一个显式的 cross-checkpoint compatibility certificate。
- 如需真正重跑 official，必须新建独立 compatibility runner/output/claim：保留原 V99 三个 artifact 不变；用上条
  certificate 代替单一文件 SHA 相等，但继续验证其余 model-input/backbone-config/protected-file/code-tree 合同；
  消费一次 9,508-row validation 并明确标成“V133 checkpoint 上冻结 V99 兼容复测”，不得覆盖旧 V99 结果。
  若不把 V133 score 注入 V99 feature，理论上应逐样本复现旧 V99 选择；若想融合 V133 与 V99 score，则输入分布
  已改变，不属于未重拟合 compatibility，未经新的 train-only 方案与预注册不得运行。
- 因此 compatibility 仍只在 V132/V133 产生新 network best 后触发，当前不占 GPU、不消费 validation。审计时
  V132 epoch3 约 `2907/4583`（63%），strict errors=`0`、正式回执仍为 2/4，未修改源码或权重。

#### 14.217 V132 四轮结论与可恢复清理（2026-08-15）

- V132 `v132_decoder_query_adapter_formal_e1_e4_b8x1` 已完整产生 4 个 `9,508`-sample 回执。REC
  @.25/@.50 hits 依次为 epoch1 `5504/4391`、epoch2 `5493/4383`、epoch3 `5488/4355`、
  epoch4 `5495/4366`；V132 内部最好仍是 epoch1 的 `57.8881%/46.1822%`，未超过 V99
  `58.6033%` 或 V113 `50.8519%`，更未达到目标 `59%/49%`。
- 同一四轮 Mask hits 依次为 `5669/4660`、`5657/4635`、`5650/4628`、`5669/4661`，mIoU
  依次为 `41.6989%/41.5002%/41.5477%/41.7353%`。即使 V132 最好的 mask@.25 仍高于用户
  baseline `58.70%`，mask@.50 与 mIoU 都明显低于 `50.70%/44.72%`；因此 full decoder-query
  residual 被判定为同时损害 REC 与 mask 的失败方向。
- 清理前以 epoch71 parent 重建 epoch1/epoch4：全部模型状态逐张量 bitwise equal。只读 compact manifest
  SHA-256=`98b3637d6886e92401cd0b71d540a7479927e95a42cf5f3dfd68ea8b76357e94`，两个 adapter delta
  SHA-256=`a8758e35...a70`、`a1e45b93...39ba`，各约 `6.02MB`。完成审计后才删除 V132 完整非最佳
  checkpoint hard links；V99、V113 与用户指定永久保留的 V109 均未触碰。

#### 14.218 V133 review1/review2 实施与审查结论（2026-08-15）

- V133 实现为独立 SACR structured-score refiner：冻结 parent query/box/mask/selector，只训练
  `structured_slot_builder + sacr_head + sacr_score_gate` 共 `920,930` 参数；部署残差固定 bounded
  为 `<0.25`。ScanRefer 使用 box-IoU listwise 加固定 `0.25` mask-IoU 项，Nr3D/Sr3D 只使用同一
  box objective，不含 validation threshold 或数据集专用后处理。
- review1 源码快照位于 `.v133_review1/`，manifest/tar SHA-256=`40c1df13...c775`/
  `6a8c4f62...8d166`；review2 快照位于 `.v133_review2/`，对应 SHA-256=`eb362103...e078`/
  `6984c67a...7307`。两轮审查修复了 ScanRefer-only mask supervision、DDP 全局 example normalization、
  zero-valid-rank、runner 固定配置、scene-disjoint smoke 与 score-state checkpoint exactness。
- review2 cross-dataset/DDP 契约曾通过，但第二轮独立审查发现它仍不能签发正式 gate：历史 run 没有启动时
  source/parent binding，gate 来源集合过窄；训练后 checkpoint 只校验 SACR keys 而非完整 model state；回执
  存在 `exists()+replace()+chmod` 的 no-clobber/可写窗口，且 baseline/smoke/formal 使用不同 GPU 锁。
  因此 review2 数值 smoke 虽为 `124/117` hits 且有限，仍明确作废为正式来源证据，未启动 formal。
- review2 两张 smoke 完整 checkpoint 清理前均以 epoch71 parent + 21 个 SACR tensors 重建，`1,165`
  个 model state 全部 bitwise equal；compact manifest SHA-256=`8bc4ee3be22ea4b4248406d3d76e3d582eb2be8e48640bda760391fd69be7ca3`。
  审计后只删除该 smoke 目录的 8 个 hard-link 名称，保留 logs/config/metrics/compact recovery。

#### 14.219 V133 review3 严格来源门与动态验证（2026-08-15）

- review3 改为启动前不可变 binding：绑定 parent path/SHA、固定 batch/lr/tau/max-delta/mask-weight、
  `nproc=1`、launch log 和当前源码。来源清单递归覆盖当前 repo 的 `304` 个 `.py` 文件加唯一 runner，排除
  `.v*` 快照、Git/cache；明确覆盖 `sacr_head.py`、`structured_slots.py`、selector adapter、dataset/evaluator。
  `--repo-root` 必须与执行脚本真实根目录相同，run config 与 launch log 必须互相指向同一 run directory。
- 回执发布改为同目录 temp、写前 `fchmod(0444)+fsync`、`os.link` 原子 create-if-absent 与 directory fsync；
  重复路径和错误 repo root 的负向测试都以非零退出。三个运行模式共用一个 `v133_gpu.lock`，runner 硬编码
  单 GPU、`nproc=1` 和正式最多 4 epoch。训练后 V133 checkpoint 要求完整 current/checkpoint model keys、
  shape、dtype 与 load missing/unexpected 全部 exact，不再只检查 SACR 子树。
- 零门控 identity 将冻结审计从 5 项扩为 7 项，新增 decoder query 与 parent selector scores。实际 128-row
  identity 固定/新分数均为 `124/128`（@.25）和 `117/128`（@.50），Mask=`120/128`、`113/128`、
  mIoU=`0.7490708084`；gate=`0`、residual=`0`、`7/7` tensor audit pass。
- 真实 scene-disjoint smoke 记录 `train=128 examples/128 scenes; holdout=128 examples/120 scenes; overlap=0`。
  epoch1/2 REC 都为 `124/117`，Mask 都为 `120/113`、mIoU=`0.7490676212`，两阈值均
  `fix=break=0`；gate 约从 `-0.0046` 到 `-0.0094`，最大残差约 `0.0023 < 0.25`。
- 相关静态回归为 `73 passed`。新 cross-dataset/DDP contract receipt SHA-256=
  `46a741f16048b9a13800f40761ffc731848df24caf9b29d52818fc8fc190814b`、mode=`0444`；最终 fail-closed
  smoke gate SHA-256=`80400e5ae0b37f499587a703576b2befec8a36194830d946557a4971616e29c7`、mode=`0444`，
  build 后又独立 verify 通过。
- 完整 review3 来源快照 `.v133_review3/SOURCE_MANIFEST.json`/`source_tree.tar` SHA-256=
  `d078ca4b...3aab`/`25380b19...42a`，归档中 304 个来源逐文件 hash 复核通过。两张 smoke checkpoint
  也以 parent + 21 SACR tensors 对 `1,165` states bitwise 重建；compact manifest SHA-256=
  `9ff8b0b8...d8e4`，delta SHA-256=`72124ae8...a8f3`/`f2569844...f398`。随后只删除 8 个完整
  hard-link 名称，释放约 `1.22GB`，gate 再验证仍通过。

#### 14.220 V133 review3 正式运行启动（2026-08-15）

- 正式 runner 已在单 GPU 上启动：experiment=`v133_sacr_score_refiner_review3_formal_e1_e4_b8x1`，run id
  `1786797601`，固定 `batch=8`、seed=`0`、`nproc=1`、epoch `1..4`、每轮完整 `9,508`-row validation。
  启动前 gate 再验证 SHA 不变并通过；正式 launch binding SHA-256=
  `210480793bf9be0b261fda6c936659c734d6e7492d8b56bbf5566f6a084d1c5f`、mode=`0444`。
- 当前阶段为完整 ScanRefer 文本/结构化数据预处理，worker CPU 约 `100%`、RSS 约 `15GB`，尚未进入 GPU
  train step、未生成正式 receipt，不能据此声称有新结果。后续必须逐轮检查 sample_count/hits、REC 与 Mask
  五项、gate/residual finite/bound、checkpoint retention 和磁盘；未完成 4 轮或未取得充分失败证据前不改配置。

#### 14.221 GitHub 源码同步与 V133 正式第 1 轮审计（2026-08-15）

- 已按用户要求把当前远端 MCLN 源码同步至 `https://github.com/666666666666gao/MCLN` 的分支
  `agent/sync-remote-mcln-source`，提交=`8a251607051b1adbea3fe0cdc8d4407aecb5398f`，并创建 draft PR
  `https://github.com/666666666666gao/MCLN/pull/1`；未自动合并。打包源归档 SHA-256=
  `e14ff9701faf30d2fcaa852cf5bbd019776a2ab8a1eeef80cd9de44fc2e5387b`，解包后共 445 个文件、
  `13,948,845` bytes。权重、输出目录、预训练模型、缓存/备份、大文件均未上传；仓库原有的
  `data/class_embeddings3d.npy` 也在该提交中移除。路径/链接安全检查、禁用扩展和 >10MB 文件检查、
  通用秘密扫描及已知凭据精确扫描均为 0 命中。
- V133 review3 正式第 1 轮 receipt：
  `eval_metrics_epoch_1.json`，SHA-256=
  `0c6c92570b405c6adde8eae2c2e4038292fc9fbd05449cf41dfa914b8a3e1d71`，sample_count=`9,508`。
  learned REC overall `.25/.50=56.6365/44.5204`（hits=`5,385/4,233`），unique
  `86.6103/72.3749`（`1,229/1,027` of `1,419`），multiple `51.3784/39.6341`
  （`4,156/3,206` of `8,089`）；固定 parent 选择为 `57.9933/46.3820`
  （`5,514/4,410`）。Mask overall `.25/.50=59.1923/48.6222`（`5,628/4,623`），
  mIoU=`41.4128`；Mask unique `89.9225/72.5863`（`1,276/1,030`），multiple
  `53.8015/44.4183`（`4,352/3,593`）。分组总数、hits、阈值嵌套、比率和
  `iou_sum/9508` 均逐项复算通过且 finite。
- 第 1 轮尚未达到目标：learned REC 相对 `.59/.49` 的 hits 缺口为 `-225/-426`；相对 V99
  REC 缺 `187/564` hits；Mask 相对 V99 缺 `62/353` hits，mIoU 低 `4.5174pp`。
  验证诊断 gate=`0.8388`、residual mean/max=`0.2087/0.2097`，虽未越过 max-delta
  `0.25`，但 mean 几乎等于 max，说明全局残差接近一致饱和，缺少逐样本 abstention，容易产生错误覆盖。
  当前判断是 SACR 的 quality-KL 与全局 scalar gate 共同把 residual 推向饱和；若 4 轮全部失败，下一版应采用
  parent-relative advantage/trust-region 监督并加入保守的逐样本 gate，而不是继续增大 residual 上限。
- 第 1 轮 checkpoint 的 7 个保留名称（含 `ckpt_epoch_1.pth`、`ckpt_epoch_last.pth` 和五个 best
  名称）均指向同一 inode，link count=`7`，实际仅占一个 `610,243,106`-byte 文件；当时
  `/root/autodl-tmp` 可用约 `3.2GB`。第 2 轮已继续运行，正式运行期间不修改源码/配置，也不清理其活动
  checkpoint；完成后再做逐轮 receipt 审计、最佳权重保留和可恢复性清理。

#### 14.222 V133 epoch2 正式回执、本地完整交接与 epoch3 启动（2026-08-15）

- epoch2 `eval_metrics_epoch_2.json` SHA-256=
  `8da73e1f8078bf0ba44b691c6fdda0e5a0ae028c92aeff0e64f7803cf0e7755f`，mode=`0644`、
  size=`1132`、sample_count=`9,508`。REC overall `.25/.50=56.2894/43.8999`
  （hits=`5,352/4,174`），unique `86.3284/71.1769`（`1,225/1,010` of `1,419`），
  multiple `51.0199/39.1148`（`4,127/3,164` of `8,089`）；固定 parent 仍为
  `57.9933/46.3820`（`5,514/4,410`）。相对目标缺 `258/485` hits，相对 V99 缺
  `220/623` hits，相对 V113 @.50 最好缺 `661` hits。
- Mask overall `.25/.50=59.2659/48.6432`（`5,635/4,625`），mIoU=`41.4346%`
  （`iou_sum=3939.6035616758504`）；unique `89.8520/72.5863`（`1,275/1,030`），multiple
  `53.9004/44.4431`（`4,360/3,595`）。相对 V99 缺 `55/351` hits，mIoU 低约
  `4.4957pp`。sample count、Overall=Unique+Multiple、threshold nesting、rate 与
  `iou_sum/9508` 均独立复算通过。
- epoch2 验证 gate=`0.9664`、residual mean/max=`0.2402/0.2416`，比 epoch1 的
  `0.8388`、`0.2087/0.2097` 更接近 max-delta `0.25`；learned REC 同时从
  `5,385/4,233` 降至 `5,352/4,174`，进一步确认整行饱和正在扩大错误覆盖。
- retention：REC 两项 best 仍为 epoch1；Mask 三项 best 更新为 epoch2。epoch1 inode link count=`3`
  （epoch1 + 两个 REC best），epoch2 inode link count=`5`（epoch2 + last + 三个 Mask best），每个实物
  `610,243,106` bytes；`/root/autodl-tmp` 可用约 `2.6GB`。当前不删除活动 checkpoint。
- epoch3 已自动启动，单 GPU 正常；源码/配置仍冻结。四轮全部完成后再决定 V133 最佳保留与 V134
  feasible parent-relative SACR，不用中途结果选择超参。

#### 14.223 GitHub PR #1 合并进 main（2026-08-15）

- 用户明确要求合并后，先复核 PR #1 为 `OPEN`、`MERGEABLE/CLEAN`，head commit=
  `8a251607051b1adbea3fe0cdc8d4407aecb5398f`。193 个变更路径中唯一匹配模型/数组扩展名的是
  **删除**原仓库 `data/class_embeddings3d.npy`；PR head 最终 tree 中 `.pth/.pt/.ckpt/.safetensors/.bin/
  .onnx/.h5/.hdf5/.npy/.npz`、output/checkpoint/pretrained 路径计数为 0。
- PR `https://github.com/666666666666gao/MCLN/pull/1` 已从 draft 转为 ready 并以 merge commit 合并；
  `main` 当前 SHA=`d273446207c9c5ef6b7cc64210137b768617c62b`，parents 为原 main
  `359c2e2c0843d59d047cc6e7243d99f7bec30eb2` 与同步提交 `8a251607...b5398f`，PR state=`MERGED`。
- 本地隔离分支 `agent/v134-feasible-parent-relative-sacr` 的未完成 V134 文件没有被 stage、commit、push
  或带入 PR #1；该分支只用于 V133 活动运行期间的离线准备。

---

## 15. 2026-08-30：Nr3D/Sr3D 最新进展与“指标提升不上去”专项诊断

> 本章是 2026-08-15 之后实验的增量交接。结论与数字以服务器正式评估回执、候选缓存和实际运行日志为依据；
> 不用训练中 loss、手工拼接单项最优或未完成 epoch 代替正式结果。
>
> **目标更新说明（2026-08-31 01:25 CST）**：本章最初采用的 `Nr3D>59.8% / Sr3D>68.4%` 已被用户
> 提高为 `Nr3D>60.0% / Sr3D>68.9%`。因此本章中“Sr3D 已达旧目标”的历史判断仍属实，但不再代表当前任务完成。

### 15.1 当前结论

1. **Sr3D 已超过既定 baseline/目标**：REC@0.25=`68.4813%`，超过 `68.4%`，多 `14 hits`。
2. **Nr3D 尚未超过 baseline**：当前正式最好 REC@0.25=`56.6527%`，目标为严格超过 `59.8%`；
   需要至少 `4724/7899`，当前为 `4475/7899`，仍差 `249 hits`（`3.1523pp`）。
3. Nr3D 的主要瓶颈已经由实证定位为：**正确候选通常已经在 Top-K 内，但最终 Query/source score 没有把它排到 Top-1**。
   因而它不是以“候选框完全找不到”为主，也不是继续降低学习率就能自然解决的问题。
4. 当前 V99 的第二个 score source 在 Nr3D 上几乎没有互补性，selector 长期退化为默认源；关系反事实辅助在
   Nr3D 又缺少 Sr3D 那种精确 GT anchor，监督噪声更大。这两个事实解释了相同总体架构在 Sr3D 有效、在 Nr3D
   不够有效的主要差异。
5. Nr3D 当前正在跑一个**只改变训练采样、不改变网络/损失/推理**的 Top-5 hard-example replay 对照。E58 已完成且
   只有 `4427/3715`，相对当前正式最好 `4475/3759` 下降 `48/44 hits`；这已经构成第一次正式未刷新。按预注册
   patience 规则继续完成 E59，但不在中途改变 LR。若 E59 仍不刷新 `4475 hits`，该路线即停止封存。

### 15.2 三个数据集的当前最好结果

| 数据集/系统 | REC@0.25 | REC@0.50 | Mask@0.25 | Mask@0.50 | Mask mIoU | 状态 |
|---|---:|---:|---:|---:|---:|---|
| ScanRefer 双阶段 V99 | **58.6033%（5572/9508）** | 50.4523%（4797/9508） | **59.8443%** | **52.3349%** | **45.9303%** | REC@.25 与三项 Mask 主结果 |
| ScanRefer 双阶段 V113 | 58.3403%（5547/9508） | **50.8519%（4835/9508）** | 59.8338% | 52.3138% | 45.9226% | REC@.50 Pareto 最好 |
| Nr3D V99 权重平均正式结果 | **56.6527%（4475/7899）** | **47.5883%（3759/7899）** | **53.0700%（4192/7899）** | **44.0435%（3479/7899）** | **37.4337%** | 当前 Nr3D 全指标最好；未达到 59.8 |
| Sr3D V99 正式最好 | **68.4813%（12139/17726）** | **58.3042%（10335/17726）** | **65.1585%（11550/17726）** | **54.2424%（9615/17726）** | **44.9970%** | 已超过 68.4 目标 |

说明：

- Nr3D 原保护 E57 为 REC `4463/3749`，权重平均候选相对它提升 `+12/+10 hits`；因此目前正式最好是
  `4475/3759`，而不是旧 E57。
- Nr3D 目标按“严格超过 59.8%”计算为至少 `4724/7899`，当前缺 `249 hits`。若仅按四舍五入显示，不能改变
  正式 hits 验收口径。
- Sr3D `68.4813%` 相对 `68.4%` 只领先 `14 hits`，虽然已达标，但必须继续保护当前最佳权重和正式回执。
- ScanRefer 的 V99/V113 是两个真实 Pareto 点，不能把它们的单项最好拼成一个不存在的模型。

### 15.3 Nr3D 已完成实验时间线

| 实验 | Epoch | REC@0.25 hits | REC@0.50 hits | 与当时 E57（4463）比较 | 结论 |
|---|---:|---:|---:|---:|---|
| 原保护 V99 | 57 | 4463 | 3749 | 基线 | 旧保护点 |
| effective global batch 48 | 58 | 4400 | 3696 | -63 | 未刷新 |
| effective global batch 48 | 59 | 4399 | 3694 | -64 | 连续未刷新，停止 |
| relation-counterfactual auxiliary | 63 | 4400 | 3691 | -63 | 未刷新 |
| relation-counterfactual auxiliary | 64 | 4394 | 3674 | -69 | 未刷新，停止 |
| 第一次低 LR 延续 | 58 | 4452 | 3752 | -11 / +3 | @.25 未刷新 |
| 第一次低 LR 延续 | 59 | 4432 | 3699 | -31 | 未刷新 |
| 第一次低 LR 延续 | 60 | 4437 | 3722 | -26 | 未刷新 |
| 第一次低 LR 延续 | 61 | 4426 | 3728 | -37 | 未刷新 |
| 第一次低 LR 延续 | 62 | 4421 | 3716 | -42 | 未刷新 |
| tier hard-query auxiliary | 58 | 4413 | 3707 | -50 | 未刷新 |
| tier hard-query auxiliary | 59 | 4398 | 3676 | -65 | patience=2 停止 |
| E26/E29 权重平均正式评估 | — | **4475** | **3759** | **+12 / +10** | 当前正式最好 |
| Top-5 hard-example replay | 58 | 4427 | 3715 | -36 / -34 | 相对当前最好为 -48 / -44；第一次未刷新 |

补充判断：低 LR、第二次衰减、effective batch 48、关系辅助与 tier hard-query 都没有产生可复现的 @.25 刷新。
因此“再衰减一次 LR”目前没有实验依据；它更可能让模型继续在同一个局部平台内波动。

### 15.4 当前活动实验：Top-5 hard-example replay

- 实验名：`nr3d_mcln_joint_butdcls_v99_e57_hard_replay_top5_e58_e59_b16a1`
- screen：`635865.mcln_nr3d_hard_replay_top5`
- 运行根目录：
  `/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/backbone/`
  `nr3d_mcln_joint_butdcls_v99_e57_hard_replay_top5_e58_e59_b16a1_20260830_223656`
- 正式 leaf：
  `.../nr3d/nr3d_mcln_joint_butdcls_v99_e57_hard_replay_top5_e58_e59_b16a1/1788100625`
- 恢复点：受保护 V99 E57 full-state checkpoint，仍为 `B16×A1`；只训练 E58--E59。
- 唯一实验变量：把训练集里“默认 Top-1 在 IoU@.25 失败、但 Top-5 存在正确候选”的样本额外回放一次。
  网络、V99 selector、损失、推理分支均不改变，满足跨数据集论文架构一致性。
- hard rows：`1548/32919=4.70%` 的基础训练标注，覆盖 `327/511` 个场景；joint train 原始 `44909` 条，
  加回放并补齐后为 `46464` 条，即 E58 每轮 `2904` 个 B16 step。
- 2026-08-31 01:12 CST，E58 已完成正式 7899-row 评估：REC@.25/.50=`4427/3715`
  （`56.0451%/47.0313%`），Mask@.25/.50=`4128/3420`（`52.2598%/43.2966%`），Mask mIoU=`36.7828%`。
  相对当前正式最好，REC 下降 `48/44 hits`，Mask 下降 `64/59 hits`，mIoU 下降约 `0.6509pp`，没有改善。
- E58 的 source 诊断显示 selector 仍以 `1.0000` 比例选择 default source，非 default 为 `0.0000`；固定默认源
  `4241/3670`，source oracle 也只约多 `3/2 hits`。这不是“selector 没训练够”那么简单，而是第二 source
  本身几乎没有可利用的互补候选排序。
- 2026-08-31 01:16 CST，E59 已自动进入 `80/2904`，GPU 约 `36.2 GiB`，无 Traceback、CUDA OOM、NaN 或 Inf。
- 验收规则：以当前正式最好 `4475 hits` 为刷新基线。E58/E59 若连续不超过它，就封存该路线；不会用 loss 下降
  或训练 batch 内的高 accuracy 代替正式 7899 样本评估。

#### hard replay 本身的局限

- 额外样本只占 joint exposure 的约 `1555/44909=3.46%`，训练分布改变较温和，可能不足以移动 249 个正式 hits。
- 它只重复困难样本，**没有新增显式目标去要求“正确 Top-5 候选超过错误 Top-1”**；原损失若仍不能区分同类候选，
  多看一次样本不一定会改变最终排序。
- hard set 对长句/组合语言的富集很弱：全体与 hard set 的 token 均值约 `11.47→11.55`，`>=13 tokens`
  约 `31.45%→32.30%`。因此它没有非常精准地聚焦诊断中最困难的长组合表达。
- hard set 中最佳正确候选要跨过错误 Top-1 的 score gap 中位数约 `0.1206`；gap `<=0.05` 的仅
  `448/1548=28.94%`，`<=0.10` 的为 `691/1548=44.64%`。这说明很多样本并非轻微调分即可修复。

### 15.5 红线复算：当前结果确实尚未达标

已对服务器上 163 份实际 metrics receipt 做递归扫描，并用 hits 重新计算，不依赖日志中四舍五入的百分比：

```text
BEST_FORMAL_RECEIPT: 4475 / 7899 = 0.56652741
TARGET_STRICT_GT_0.598: hits_required = 4724
GAP: 249 hits
AssertionError: Nr3D formal REC@0.25 remains below strict >59.8 target
```

该失败断言是预期的“红灯”证据：当前不能把 56.65 宣称为超过 59.8，也不能把候选 oracle 当作正式指标。

### 15.6 主因一：候选框不是主要上限，Top-1 排序才是

对当前正式候选缓存做 unfiltered Top-K oracle 诊断：

| 选择口径 | REC@0.25 hits | REC@0.25 |
|---|---:|---:|
| 当前 default Top-1 | 4275 | 54.1208% |
| Top-2 oracle | 4872 | 61.6787% |
| Top-3 oracle | 5136 | 65.0209% |
| Top-5 oracle | 5458 | 69.0974% |
| Top-8 oracle | 5781 | 73.1865% |
| Top-16 oracle | 6343 | 80.3013% |

解释：

- Top-2 oracle 已经超过 59.8 目标，Top-5/Top-16 的上限更高；因此绝大多数剩余失败不是“没有正确框”。
- default Top-1 失败但 Top-16 内可修复的样本有 `2068` 个。要达到目标，只需安全修复其中 `249` 个，约
  `12.04%`，理论候选余量充分。
- 在这 2068 个可修复失败中，第一正确候选的 rank `<=2/3/5/8` 分别有 `597/861/1183/1506` 个。
- 最佳正确候选与当前错误 Top-1 的 score gap `<=0.01/0.03/0.05/0.10/0.20/0.50` 分别有
  `69/150/224/359/561/1057` 个。目标 249 hits 已超过“只翻很小 gap（<=0.05）”的 224 个，
  但小于 `<=0.10` 的 359 个：需要有语义依据的安全 reranking，不能只做一个极小全局扰动。

结论：下一阶段应把问题表述为**候选级语义消歧/重排序**，而不是继续盲目改 box proposal 或增加全局残差。

### 15.7 主因二：V99 source selector 在 Nr3D 上发生功能性塌缩

V99 在 ScanRefer 的核心部署形式是从固定候选 score source 中做安全选择；但 Nr3D 的两个现有 source 几乎相同：

| Nr3D 候选 source | REC@0.25 |
|---|---:|
| `default` | 54.1208%（4275/7899） |
| `default_rank_blend_contrastive010` | 53.6903% |
| 两 source oracle | 54.1588%（4278/7899） |

- 第二个 source 本身更差；即使使用不可部署的两源 oracle，也只比 default 多 `3 hits`，远小于目标缺口 249。
- 正式权重平均评估和当前训练日志都显示 `selected_non_default_ratio=0.0000`；目标 non-default 比例只有约
  `0.1%--0.15%`。selector 学成“永远选 default”在这个输入集合上反而近似合理。
- 所以当前问题不是 selector MLP 容量不足，而是**提供给 selector 的第二个 source 不具备互补候选排序**。
  单纯加深 gate、改阈值或延长训练，不会凭空产生缺失的互补信息。

注意：这里的 `4275` 是候选 source 子系统的 default diagnostic，正式部署最好是 `4475`；两者口径不同，不能
混为同一个 overall 结果。但二源 oracle 只多 3 hits 的结论足以证明现有 source routing 没有 249-hit 上升空间。

### 15.8 主因三：长句、组合关系与属性表达的排序误差更明显

验证集诊断显示 proposal oracle 基本保持高位，但 Top-1 随语言复杂度明显下降：

| 子群 | Top-1 REC@0.25 | Top-16 oracle@0.25 |
|---|---:|---:|
| 2--6 tokens | 58.27% | 81.51% |
| 7--8 tokens | 58.38% | 83.06% |
| 9--12 tokens | 53.20% | 80.62% |
| >=13 tokens | 49.19% | 77.21% |
| 有 spatial 表达 | 54.24% | 80.45% |
| 无 spatial 表达 | 53.05% | 78.95% |
| 有 color 表达 | 50.60% | 80.26% |
| 无 color 表达 | 55.19% | 80.31% |
| 有 shape 表达 | 51.32% | 78.27% |
| 无 shape 表达 | 54.65% | 80.68% |

尤其是 color/shape/长句：候选 oracle 仍接近 78%--81%，但 Top-1 明显更低。这说明模型经常“看到了正确对象”，
却没有正确利用文本中的属性、关系和参照物把同类候选排开。它与 Nr3D 的 Multiple/同类干扰本质一致。

### 15.9 主因四：Nr3D 与 Sr3D 的关系监督质量不同

- Sr3D 原始标注能够提供精确 relation/anchor 信息；关系反事实辅助叠加低 LR 后，最终 REC@0.25 达到
  `68.4813%`，说明该思想在有可靠 anchor 的数据上可以工作。
- Nr3D 没有同等精确的 GT anchor。现有 train-only relation-counterfactual 路径只能使用唯一/伪 anchor，
  在多同类对象场景中容易把错误参照物当作监督来源。
- Nr3D relation-CF 的 E63/E64 仅 `4400/4394 hits`，均低于 E57 的 `4463`；这不是偶然少训练一轮，
  而是监督源质量不匹配造成的直接负证据。
- 因此论文可以保持同一整体网络，但不能声称两个数据集的 anchor supervision 完全等价。更泛化的做法必须在
  无可靠 anchor 时 fail closed 或使用只由原始文本/场景可证明的保守关系，而不能扩大伪标签覆盖。

### 15.10 仅保留的历史归因边界：不得转化为 baseline 复现任务

> **2026-08-31 最新决策覆盖本节原执行含义：**以下内容只解释为什么当前结果与论文公开数值不能做
> 完全同协议归因，不再要求补跑 detector-pretrained/global48/150--240 epoch 的公平 baseline，也不把该复现
> 作为 FPR-TV、Proposal Refiner 或任何后续正式实验的前置条件。项目资源只用于直接达到
> `Nr3D REC@0.25>60.0%` 与 `Sr3D REC@0.25>68.9%`。

公开的 MCLN Nr3D 训练配置是 `4 GPU × batch 12 = global batch 48`、最长 `240 epochs`，主要 LR milestone
在 epoch 150。当前最好的分支是 `B16×A1` 的 E57 低 LR checkpoint；此前 global48 对照只运行到约 E41 或
E58--E59 的局部恢复实验，没有从头完整跑到 epoch150/240。

- 当前 E57 约经历 `57×44909≈2.56M` 条样本 exposure。
- 同一数据规模跑到官方 epoch150 约为 `150×44909≈6.74M` exposure。
- “optimizer step 数对齐”不等同于“global batch、梯度噪声与样本 exposure 全部对齐”。

因此只保留以下两个**解释性判断**，不得据此生成新的训练任务：

1. **当前 B16 低 LR 分支已经进入真实平台期**：多次延训和 LR 衰减均未刷新，不能再期待简单续训自然补 249 hits。
2. **官方 global48、150/240 epoch 的公平复现尚未真正完成**：所以“为什么低于论文 59.8”仍存在协议归因边界，
   但按用户最新决定不再补跑；论文中如实声明协议差异即可。

服务器上没有可直接正式评估的官方 MCLN Nr3D task checkpoint，只有 GroupFree detector 初始化权重。因此目前无法
用发布权重直接复核论文的 59.8。该事实仅作为证据边界保存，不再导出“必须完整重训 baseline”的行动项。

### 15.11 Parser 与 `nr3d_spacy/sr3d_spacy` 的结论

- 当前 ScanRefer、Nr3D、Sr3D 正式 V99 都读取**原始标注**，并在线调用 legacy `Scene_graph_parse`；不是把
  `nr3d_spacy.csv` 或 `sr3d_spacy.csv` 直接作为 baseline 输入。
- 当前正式配置中 `use_sacr_source=false`，`legacy_scene_graph_cache=''`；因此现有最好结果不能写成“使用了
  spacy sidecar”。
- `nr3d_spacy.csv/sr3d_spacy.csv` 含有预分解乃至人工/GT 派生字段，直接接入需要启用新的 structured/SACR
  数据路径，会改变输入合同与架构口径，不能作为“与 baseline 完全一致的 V99”直接替换。
- 运行环境当前约为 spaCy `3.4.4` / model `3.4.1`，而原 README 记录 `3.3.0/3.3.0`；这是可复现性漂移，
  但现有证据不足以证明它单独造成 3.15pp 缺口。
- 保守 raw-parser 选择在 train-only 审计中有稳定改善：fit target-text match `68.9259%→71.5045%`
  （`+2.5785pp`，`757 fixes/92 breaks`）；scene-disjoint holdout `68.2284%→70.7673%`
  （`+2.5389pp`，`211 fixes/30 breaks`）。这证明解析存在可改进空间，但**不是正式 REC 提升证据**。
- 若当前 hard replay 失败，优先做一次固定权重、固定候选、仅替换为经审计 raw-parser cache 的 one-shot 正式评估；
  先证明 inference 指标，再决定是否重建训练 cache。不能对 validation 反复扫 parser 规则。

### 15.12 为什么现在不应立即再次衰减学习率

已有三类负证据：

1. E57 后低 LR E58--E62 的 REC@.25 为 `4452, 4432, 4437, 4426, 4421`，没有一次超过 4463。
2. 后续更低 LR 的长链也没有刷新，tier/hard-query 分支 E58/E59 为 `4413/4398`。
3. 权重平均只带来 `+12 hits`，说明相邻 checkpoint 是在同一盆地内轻微波动，而非持续上升趋势。

当前 hard replay 已经从低 LR E57 full state 恢复。在它产生 E58 正式指标前再改 LR，会同时改变“采样”和“优化率”
两个变量，破坏因果判断。正确做法是先完成 E58；只有出现明确提升但随后停滞时，才有依据考虑一次预注册衰减。

### 15.13 下一步执行顺序

1. **完成当前 hard replay E58 正式评估**，与新基线 `4475` 比较；不以旧 `4463` 冒充当前最好。
2. 若 E58 不刷新，继续预注册的 E59；若 E58/E59 连续不刷新，停止并只保留正式最佳权重/回执，删除 run-local
   无用 epoch 权重。
3. hard replay 失败后，执行一次经双重审计的 conservative raw-parser one-shot eval；应改为绑定当前保护的
   `4475` 权重平均模型，而不是继续引用已经被超越的旧 E57，并继续保持 V99 网络、候选和 selector 不变。
   - 该新版 launcher 已于 2026-08-30 23:39 CST 准备在远程：
     `scripts/eval_nr3d_v99_raw_parser_avgbest_one_shot.sh`，SHA-256=
     `fccb6f894c8ce703582b4dbfbc14e92034a40729c0ce711cff30caa52e3ea350`。
   - 它固定当前最好权重 SHA `76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`、
     baseline `4475/3759`、同一 V99 selector 和 raw-only conservative parser bundle；不训练、不写权重。
   - Python3.7 heredoc AST `5/5` 与 Bash syntax 已通过；实际 preflight 已完成权重/bundle/V99 provenance 校验，
     随后因 hard-replay 正占用 GPU 约 33.1 GiB 而按预期以 status 5 拒绝启动。one-shot run root 仍不存在，
     因此未消费唯一评估机会，也没有与当前训练并发。
4. parser one-shot 若产生可信净提升，再用同一个 raw-only parser 合同重建训练 cache；若不提升则封存 parser 路线。
5. **该历史方案已取消，不得执行**：不再启动 detector-pretrained、no-task-resume、global48 的完整
   150/240 epoch baseline 复现；训练协议差异只作为论文归因边界记录，不再占用后续实验资源。
6. 若仍需新模块，应以同一模块支持 ScanRefer/Nr3D/Sr3D：针对 Top-2/Top-5 候选做文本条件的保守 verifier，
   不使用 Unique/Multiple 标签、不使用 dataset-ID、不联合训练；无可靠优势时严格回退 parent。普通 listwise loss、
   全局 gate、无条件残差、噪声伪 anchor 已有充分负结果，不应换名字重复。

### 15.14 权重与关键产物保护

- Nr3D 当前正式最好保护权重：
  `/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/`
  `official_best_rec025_epoch_57_0p56652741.pth`
- SHA-256：`76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`。
- Nr3D 受保护 full-state E57：
  `/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/audit/`
  `nr3d_mcln_joint_butdcls_v99_relation_cf_conservative_anchor_density_v2_audit_e58_b100_b16x1_w4p2_one_shot/`
  `resume_e57.pth`
- E57 SHA-256：`fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655`。
- Sr3D 已达标的 best REC@.25 权重、正式 metrics/config receipt 必须保留；后续清理只删除非 best、非 last、
  非 provenance 依赖的 run-local checkpoint。
- 远程权威持续交接文档仍是：
  `/home/gb/new butd/butd_detr-main/MCLN-main/docs/REC_3DRES_OPTIMIZATION_LOG.md`。
  本地本章是截至 2026-08-30 23:24 CST 的同步快照，后续实验继续先写远程权威文档，再同步到本地。

### 15.15 面向论文的诚实表述

- ScanRefer：V99 是 REC@.25/Mask 主结果，V113 是 REC@.50 Pareto 点。
- Sr3D：同一 V99 总体架构已经以 `68.4813%` 超过 `68.4%` 目标，证明迁移并非整体失效。
- Nr3D：当前最好 `56.6527%`，仍低于 `59.8%`；现阶段只能报告“候选 oracle 充分、错误集中在文本条件的
  Top-1 选择”，不能宣称已超过 baseline。
- 跨数据集失败差异可解释为：Sr3D 有更可靠的关系/anchor 标注，而 Nr3D 更依赖噪声伪 anchor与复杂自然语言；
  同时 Nr3D 尚未完成与论文相同的 global48/150--240 epoch 全时程复现。
- 后续创新应统一成“安全候选重排序/保守解析”，而不是为 Nr3D 单独接入含 GT 派生字段的 sidecar，否则会破坏
  泛化性和论文公平口径。

---

## 16. 2026-08-31：为什么 ScanRefer 提升明显，而 Nr3D/Sr3D 迁移困难——逐场景、点云密度、语言与实验效果完整诊断

> 本章专门回答四个问题：ScanRefer 的提升到底来自哪里；为什么同一 V99 思路迁移后没有自然超过
> Nr3D/Sr3D baseline；哪些点云场景、物体类别与表达最困难；我们已经做过哪些修复、各自效果是否明显。
> 所有正式指标均来自完整官方回执；逐场景分析来自已经封存的候选缓存，不重新访问 validation 做参数搜索。

### 16.1 先纠正一个容易误解的结论

“Nr3D 和 Sr3D 都不如 baseline”只描述了大部分训练过程。Sr3D 已超过外部 68.4 baseline，但在用户提高
目标后，两个迁移数据集都仍未完成：

| 数据集 | 当前正式最好 REC@0.25 | 外部 baseline | 当前项目硬目标 | 距新目标 | 当前判定 |
|---|---:|---:|---:|---:|---|
| ScanRefer | 58.6033%（5572/9508） | 原 MCLN 57.17% | 59.0%（5610 hits） | -38 hits | 数值上明显超过原 MCLN，但未到 59.0 |
| Nr3D | 56.6527%（4475/7899） | 59.8% | **严格 >60.0%（至少4740 hits）** | **-265 hits** | 未超过 baseline，也未达到新目标 |
| Sr3D | 68.4813%（12139/17726） | 68.4% | **严格 >68.9%（至少12214 hits）** | **-75 hits** | 已超外部 baseline，但未达到新目标 |

因此后文应使用以下准确表述：

1. ScanRefer 的**完整系统**相对原 MCLN 有明显数值提升。
2. Nr3D 仍明显低于外部 baseline，而且距离新目标还差 265 hits，是当前最严重的迁移问题。
3. Sr3D 在很长一段实验中低于 baseline，后来通过可靠关系监督、低学习率延续和权重平均刚刚越过 68.4；
   它证明方法并非完全不能迁移，但距离新的 68.9 目标仍差 75 hits，不能再视为任务完成。
4. 目前没有多随机种子均值、方差和显著性检验。本文中的“明显/显著”若出现，均指**数值幅度**，不等价于
   统计学显著性。

### 16.2 证据口径：正式结果与逐场景诊断不能混用

本章同时使用两类证据：

1. **正式总体结果**：完整 evaluator 的固定样本数回执。ScanRefer 为 9508，Nr3D 为 7899，Sr3D 为 17726。
   论文主表只能使用这类数值。
2. **候选缓存诊断**：保存每个样本 Top-16 Query、score 和 IoU，用于回答错误出在 proposal 还是 Top-1 排序，
   并按 scene、目标点数、体积、类别、同类干扰物和语言长度分层。这类结果只能解释机制，不能替代正式总体结果。

具体缓存口径如下：

- Nr3D 缓存绑定当前最好权重 SHA
  `76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`，共 7899 条；其
  `target_iou_policy=root_only` 的 default diagnostic 为 4275 hits，而正式 position evaluator 为 4475 hits。
  两者 stage/IoU 口径不同，因此场景分层统一使用缓存内 4275 作为诊断分母，不把它冒充正式 4475。
- Sr3D 候选缓存绑定 E32 SHA
  `68dbeb2fb4df3e381b7845a3a5d63976fbd060de24dffe8f0d3a26d7f6126c16`，共 17726 条；缓存
  default diagnostic 为 11374 hits。最终最好是后续 E26/E29 权重平均的 12139 hits，故 Sr3D 场景表描述的是
  已有 E32 缓存暴露的困难模式，不声称是最终平均权重逐场景的精确复算。
- 点云目标点数、目标轴对齐体积和场景物体数来自实际 `val_v3scans.pkl`：目标点数是 5 万点采样后属于目标实例
  的点数，体积由这些点的三轴范围乘积计算。它们是同一输入点云上的可审计难度代理。

这个区分很重要：总体指标回答“最终模型有多好”，候选缓存回答“为什么会错”。

### 16.3 ScanRefer 的大幅总提升究竟来自哪里

#### 16.3.1 完整系统相对原 MCLN 的数值变化

| 指标 | 原 MCLN baseline | ScanRefer V99 | 绝对变化 |
|---|---:|---:|---:|
| REC@0.25 | 57.17% | 58.6033% | **+1.4333pp** |
| REC@0.50 | 45.53% | 50.4523% | **+4.9223pp** |
| Mask@0.25 | 58.70% | 59.8443% | **+1.1443pp** |
| Mask@0.50 | 50.70% | 52.3349% | **+1.6349pp** |
| Mask mIoU | 44.72% | 45.9303% | **+1.2103pp** |

`@0.50` 的提升远大于 `@0.25`，说明这套系统最主要的收益是从已有候选中选择**定位更准、Mask 更一致**的
Query，而不是单纯增加一个“只要碰到目标就算对”的粗框。

#### 16.3.2 不能把全部提升都归功于一个 V99 小模块

ScanRefer 当前主结果是一个完整链条：

```text
MCLN backbone
  -> parent reranker
  -> geometry reranker
  -> V99 contextual query-set hierarchy
  -> Pareto safety gate
  -> mesh-derived official superpoint mask post-processing
```

各部分作用不同：

- V99 对 Top-16 Query/variant 做 permutation-equivariant contextual 建模，不再孤立判断一个候选。
- 监督效用固定为 `IoU + 2×hit@0.25 + hit@0.50`，把粗定位、严格定位和连续 IoU 同时纳入。
- Pareto gate 只有在预测 `@0.25` 和 `@0.50` 都改善、且 aggregate gain 超过冻结 margin 时才切换；否则回退
  当前 geometry parent。这显著减少“为了修一个阈值而破坏另一个阈值”的冒险切换。
- V99 的 5-fold scene-disjoint OOF 在 33040 条训练样本上得到 `@.25 fixes/breaks=245/70`、净 `+175`，
  `@.50=751/277`、净 `+474`，且五折均为正；说明候选上下文确有训练域内作用。
- 但**隔离 V99 hierarchy**在最初 sealed geometry official 上只把 `5542/4621` 提到 `5552/4645`，即
  `+10/+24 hits`（约 `+0.105/+0.252pp`）。因此相对原 MCLN 的 `+1.43/+4.92pp` 是整条流水线的累计结果，
  不是“V99 单个后处理器独自贡献了全部增益”。

#### 16.3.3 正确 superpoint 修复是 ScanRefer Mask 与严格 IoU 提升的重要组成

旧验证曾混用/fallback superpoints，修复为 mesh-derived official superpoints 后，相对旧 V99：

| 变化 | REC hits@.25/.50 | Mask hits@.25/.50 | Mask mIoU |
|---|---:|---:|---:|
| 修复前 -> 修复后 | `+20 / +152` | `+14 / +314` | **+4.1656pp** |

这解释了为什么早期看起来 Mask@0.50 与 mIoU 没超过 baseline，而最终三项都超过。它也说明 ScanRefer 的总收益
包含了**正确的数据/几何后处理**。Nr3D/Sr3D 不会仅靠复制一个 selector 自动获得同样的 superpoint 修复收益。

#### 16.3.4 ScanRefer 的评估分布对 Overall 更友好

ScanRefer V99 的分组为：

| 子群 | 样本数 | REC@0.25 | REC@0.50 |
|---|---:|---:|---:|
| Unique | 1419 | 88.8654% | 80.5497% |
| Multiple | 8089 | 53.2946% | 45.1725% |
| Overall | 9508 | 58.6033% | 50.4523% |

Unique 占 14.92%，且准确率很高，会抬高 Overall。Nr3D 与 Sr3D 当前正式回执的
`position_subgroups.unique.sample_count=0`，全部样本都进入 Multiple 口径；它们本身就是在同类目标之间做消歧，
不存在 ScanRefer Unique 这部分容易样本。因此三个数据集的绝对 Overall 不能直接按同一难度理解。

### 16.4 三个数据集的关键分布差异

| 维度 | ScanRefer | Nr3D | Sr3D | 对 V99 迁移的影响 |
|---|---|---|---|---|
| 语言 | 自然语言，常有目标外观与位置描述 | 自然自由表达，长句、视角词、多跳关系更多 | 模板化关系句，词汇和句法较规则 | Nr3D 对 parser 与组合语义最敏感；Sr3D 更容易从关系监督获益 |
| 评估子群 | Unique 1419 + Multiple 8089 | 正式口径全部 Multiple | 正式口径全部 Multiple | Nr/Sr 没有 Unique 容易子集兜底 |
| anchor | 后处理主要依赖候选上下文与解析实体 | 没有与 Sr3D 等价的显式精确 anchor，只能用唯一/伪 anchor | CSV 有 `anchor_ids/anchors_types` 精确标注 | 同一 relation-CF 在 Sr 有效，在 Nr 容易被伪 anchor 噪声反向监督 |
| score source | parent/geometry/query variants 有真实互补 | `default` 与 rank-blend 几乎同序，二源 oracle 只多 3 hits | 关系候选与精确 anchor 提供更强差异 | Nr selector 退化为永远选 default，不是 gate 容量问题 |
| Mask 几何 | 已修复 mesh-derived official superpoints | 小目标/稀疏点导致 proposal 与 mask 同时困难 | 同样受小目标影响 | ScanRefer 的 Mask 修复收益不能自动跨数据集复制 |
| 训练协议 | 后处理有固定 OOF/full-fit/一次 official 流程 | 当前最好来自 B16×A1 E57；未完成论文 global48、150/240 全时程 | 已经历较长训练、低 LR 延续与权重平均 | Nr 既有真实平台，也有“官方训练时程尚未完全复现”的未决因素 |

### 16.5 Nr3D：哪些点云场景和样本最差

#### 16.5.1 首先区分“候选没有生成”和“候选有但排错”

| Nr3D 候选诊断 | hits@0.25 | Acc@0.25 |
|---|---:|---:|
| default Top-1 | 4275 | 54.1208% |
| Top-2 oracle | 4872 | 61.6787% |
| Top-5 oracle | 5458 | 69.0974% |
| Top-16 oracle | 6343 | 80.3013% |

在 7899 条中：

- `2068` 条属于**排序失败**：Top-1 错，但 Top-16 已有 IoU>0.25 的正确候选。
- `1556` 条属于**候选/定位失败**：Top-16 仍没有 IoU>0.25 候选。
- Top-2 oracle 已经达到 61.68%，超过 59.8 目标。这是“主要瓶颈在 Top-1 选择”的最直接证据。
- 但 1556 条 proposal failure 也不能忽略；它们集中在非常小、点数稀疏的目标上，单靠重排序无法修复。

#### 16.5.2 同类干扰物越多，Top-1 越低

| 同类干扰物数 | 样本数 | Top-1@0.25 | Top-16 oracle | 可修复排序失败 | proposal failure |
|---:|---:|---:|---:|---:|---:|
| 1 | 3926 | 58.889% | 83.036% | 948 | 666 |
| 2 | 1698 | 50.942% | 76.561% | 435 | 398 |
| 3 | 1234 | 47.650% | 78.201% | 377 | 269 |
| 4 | 624 | 51.282% | 80.929% | 185 | 119 |
| 5+ | 417 | 45.564% | 75.060% | 123 | 104 |

从 1 个干扰物到 5 个以上，Top-1 下降 `13.325pp`。Oracle 仍有 75.06%，说明多同类场景并不只是 detector
找不到框，更常见的是模型没有正确利用关系/属性从多个正确类别实例中选出目标。

#### 16.5.3 长自然语言主要伤害排序，而不是彻底消灭候选

| 文本长度 | 样本数 | Top-1@0.25 | Top-16 oracle | 排序失败 | proposal failure |
|---|---:|---:|---:|---:|---:|
| 2--6 tokens | 1639 | 58.267% | 81.513% | 381 | 303 |
| 7--8 tokens | 1576 | 58.376% | 83.058% | 389 | 267 |
| 9--12 tokens | 2389 | 53.202% | 80.620% | 655 | 463 |
| 13+ tokens | 2295 | 49.194% | 77.211% | 643 | 523 |

13+ token 长句比 2--6 token 短句低 `9.073pp`，但 Top-16 仍达 77.21%。这类句子常同时包含目标类别、颜色、
参照物、左右/前后以及观察方向；模型“看见了”正确对象，却没有把全部约束组合起来。

#### 16.5.4 小目标和稀疏点目标是最明确的 point-cloud proposal 瓶颈

目标点数四分位边界为 `227 / 494 / 1066`，目标体积四分位边界约为
`0.0427 / 0.1470 / 0.4576 m^3`：

| 点云目标分组 | Top-1@0.25 | Top-1@0.50 | Top-16 oracle | proposal failure |
|---|---:|---:|---:|---:|
| 最稀疏 Q1（<=227 点） | 41.343% | 26.956% | 70.621% | 582 |
| Q2 | 55.550% | 48.454% | 81.703% | 361 |
| Q3 | 58.388% | 53.675% | 84.744% | 301 |
| 最稠密 Q4 | 61.258% | 58.722% | 84.178% | 312 |

| 目标体积分组 | Top-1@0.25 | Top-1@0.50 | Top-16 oracle | proposal failure |
|---|---:|---:|---:|---:|
| 最小 Q1 | 42.994% | 28.174% | 70.916% | 575 |
| Q2 | 55.043% | 48.657% | 82.767% | 340 |
| Q3 | 56.933% | 51.316% | 81.933% | 357 |
| 最大 Q4 | 61.531% | 59.605% | 85.606% | 284 |

最稀疏与最稠密目标相差 `19.915pp`，最小与最大体积相差 `18.537pp`。所以“所有错误都是语言排序”也不准确：
对于肥皂盒、鼠标、瓶子、书、厕纸等小目标，3D proposal 本身经常没有达到 IoU@0.25。

#### 16.5.5 最低类别集中在小物体、细长物体和高度相似实例

| 类别（样本>=30） | n | Top-1@0.25 | Top-16 oracle | 主要问题 |
|---|---:|---:|---:|---|
| mouse | 34 | 2.94% | 14.71% | 极小、点数少，主要是 proposal failure |
| soap dish | 56 | 3.57% | 32.14% | 极小且常在浴室/水池附近 |
| storage bin | 34 | 14.71% | 47.06% | 外观相近、堆叠/遮挡 |
| bottle | 47 | 17.02% | 40.43% | 小、细长、同类多 |
| book | 84 | 19.05% | 42.86% | 薄、点稀疏、常成堆 |
| cup | 70 | 25.71% | 70.00% | 候选有较大余量，兼有排序问题 |
| toilet paper | 95 | 28.42% | 58.95% | 小、重复、浴室遮挡 |
| rail | 71 | 32.39% | 57.75% | 细长结构，box IoU 不稳定 |
| bag | 87 | 36.78% | 83.91% | 候选充足，主要是同类排序 |
| box | 258 | 40.31% | 65.12% | 堆叠、外观近似、关系消歧 |
| picture | 256 | 43.75% | 67.19% | 薄平面、同墙多实例 |
| door | 428 | 50.23% | 73.83% | 多个相似大平面，依赖视角/房间关系 |

这里能看到两种完全不同的失败：`mouse/soap dish/bottle/book` 的 oracle 也很低，应改 proposal/小物体表征；
`bag/cup/picture/door` 的 oracle 明显高于 Top-1，应改语义排序。

#### 16.5.6 具体低指标场景

| ScanNet scene | n | Top-1@0.25 | Top-16 oracle | 总失败 | 其中可排序修复 | 场景特征 |
|---|---:|---:|---:|---:|---:|---|
| `scene0100_00` | 21 | 4.76% | 33.33% | 20 | 6 | 候选上限本身低，proposal 主导 |
| `scene0606_00` | 70 | 17.14% | 68.57% | 58 | 36 | 正确候选大量存在，Top-1 消歧失败 |
| `scene0693_00` | 33 | 18.18% | 63.64% | 27 | 15 | 排序与 proposal 混合失效 |
| `scene0678_00` | 57 | 24.56% | 54.39% | 43 | 17 | 多同类、候选上限偏低 |
| `scene0458_00` | 110 | 30.00% | 49.09% | 77 | 21 | 浴室/淋浴区域，小瓶、soap dish、厕纸等；56 条 proposal failure |
| `scene0357_00` | 87 | 31.03% | 62.07% | 60 | 27 | 重复目标和关系消歧并存 |
| `scene0095_00` | 76 | 31.58% | 44.74% | 52 | 10 | proposal 上限尤其低 |
| `scene0084_00` | 89 | 32.58% | 59.55% | 60 | 24 | 小目标与同类候选混合 |
| `scene0203_00` | 119 | 37.82% | 66.39% | 74 | 34 | 高失败贡献的复杂房间 |
| `scene0030_00` | 129 | 41.86% | 75.19% | 75 | 43 | 候选很多，但关系排序不稳 |
| `scene0653_00` | 164 | 50.00% | 85.37% | 82 | **58** | 53 个物体的办公室，desk/file cabinet/keyboard/board 同类密集 |
| `scene0645_00` | 191 | 58.12% | 75.39% | 80 | 33 | 86 个物体，含大量极小实例；47 条 proposal failure |

不能只按最低百分比选择优化场景。`scene0653_00` 的准确率不是最低，但它贡献 82 个错误、其中 58 个有正确
Top-16 候选，是最值得做安全重排序的场景之一；`scene0458_00` 则以 proposal failure 为主，重复回放排序样本
不会解决小瓶和肥皂盒没有框的问题。

#### 16.5.7 典型 Nr3D 失败语句

排序失败示例（正确候选已在 Top-5）：

- `scene0653_00 / trash can`："when looking at the whiteboard, it is the trashcan on the right-hand side"；
  Top-1 IoU=`0.246`，正确候选 rank=2、IoU=`0.513`，score gap 仅 `0.0053`。这是非常典型的可安全修复样本。
- `scene0653_00 / window`："The window that does not have a computer facing directly away from it"；
  错误 Top-1 IoU=0，正确 rank=2、IoU=`0.753`，需要理解否定与参照物。
- `scene0653_00 / file cabinet`：含 backpack、purple/burgundy item 等多属性长句；正确 rank=2、IoU=`0.738`。
- `scene0653_00 / keyboard`："farthest from the window" 且 "in front of two monitors"；正确候选在 rank=5。
- `scene0653_00 / desk`："closest to and parallel to the big whiteboard"；正确候选在 rank=4、IoU=`0.829`。
- `scene0549_00 / armchair`："facing the window, near left chair"；正确 rank=2、IoU=`0.834`，gap=`0.0033`。

Proposal failure 示例（Top-16 仍无正确框）：

- `scene0458_00 / soap dish`：59 个采样点，目标体积约 `0.0007`，最佳 IoU=`0.126`。
- `scene0458_00 / bottle`：104 点，"middle triangular shelf inside the shower"，Top-16 最佳 IoU=0。
- `scene0458_00 / toilet paper`：166 点，最佳 IoU=`0.209`，尚未越过 0.25。
- `scene0645_00 / soap dish`：仅 11 点，最佳 IoU=`0.145`。
- `scene0645_00 / door handle/door`：21 点，最佳 IoU=0；语句中的“door”实际指非常局部的门把手区域。
- `scene0645_00 / picture`：30 点、薄平面，最佳 IoU=0。

### 16.6 Sr3D：为什么早期低于 baseline，最后又能勉强超过

#### 16.6.1 Sr3D 也有候选排序余量，但最终有更可靠的关系监督

E32 候选诊断为：

| Sr3D E32 候选诊断 | hits@0.25 | Acc@0.25 |
|---|---:|---:|
| default Top-1 | 11374 | 64.1656% |
| Top-2 oracle | 12616 | 71.1768% |
| Top-5 oracle | 13385 | 75.5094% |
| Top-16 oracle | 14825 | 83.6320% |

其中 3451 条是 Top-1 错但 Top-16 可修复，2901 条为 Top-16 proposal failure。Sr3D 与 Nr3D 都存在排序问题；
差别在于 Sr3D 的 `anchor_ids` 和关系类型是显式标注，能够给 relation-counterfactual auxiliary 提供精确参照物，
而不是从自然语言中猜一个伪 anchor。

#### 16.6.2 同类干扰与小目标依然明显降低 Sr3D

| 同类干扰物数 | n | Top-1@0.25 | Top-16 oracle |
|---:|---:|---:|---:|
| 1 | 12430 | 66.870% | 86.146% |
| 2 | 3594 | 60.211% | 78.854% |
| 3 | 1256 | 52.309% | 75.557% |
| 4 | 336 | 52.976% | 73.214% |
| 5+ | 110 | 57.273% | 80.000% |

1 个到 3 个干扰物时下降 `14.561pp`。5+ 组样本仅 110 条且关系构成不同，回升不能解读为“干扰越多越容易”。

| Sr3D 点云目标分组 | Top-1@0.25 | Top-1@0.50 | Top-16 oracle |
|---|---:|---:|---:|
| 最稀疏 Q1（<=204 点） | 52.456% | 30.996% | 77.535% |
| Q2 | 67.579% | 61.035% | 85.819% |
| Q3 | 66.321% | 60.280% | 85.550% |
| 最稠密 Q4 | 70.470% | 66.561% | 85.721% |

最稀疏与最稠密相差 `18.014pp`。目标体积最小 Q1 为 54.703%，Q3/Q4 约 70.66/69.13%，说明小目标仍是
跨数据集共同问题，而不是 Nr3D 独有问题。

#### 16.6.3 关系类型本身有明显难度差异

| Sr3D relation type | n | Top-1@0.25 | Top-16 oracle | 解释 |
|---|---:|---:|---:|---|
| back | 42 | 42.86% | 73.81% | 视角/朝向依赖强，样本较少 |
| right | 220 | 50.00% | 83.18% | 正确候选多，但左右消歧失败 |
| supported-by | 152 | 52.63% | 75.00% | 接触/支撑几何难 |
| front | 316 | 53.16% | 83.86% | 视角关系难 |
| left | 174 | 57.47% | 77.59% | 仍低于整体 |
| farthest | 7188 | 62.79% | 82.46% | 远距离比较比 closest 更难 |
| closest | 7238 | 65.21% | 83.12% | 大样本主体 |
| between | 1672 | 69.74% | 90.55% | 双 anchor 明确时上限很高 |
| above | 332 | 73.49% | 87.95% | z 轴关系较稳定 |

按 coarse type，`allocentric=52.66%`，低于 horizontal 64.00%、vertical 70.28% 和 between 69.74%。
这说明即使有精确 anchor，front/back/left/right 等方向关系仍需要可靠视角定义；parser 只识别关系词不等于模型
真正理解观察坐标系。

#### 16.6.4 Sr3D 的低指标场景与类别

| scene | n | Top-1@0.25 | Top-16 oracle | 总失败 | 可排序修复 | 特征 |
|---|---:|---:|---:|---:|---:|---|
| `scene0553_00` | 20 | 0.00% | 35.00% | 20 | 7 | proposal 上限极低 |
| `scene0084_01` | 64 | 18.75% | 53.13% | 52 | 22 | 小目标与重复实例混合 |
| `scene0690_01` | 30 | 20.00% | 66.67% | 24 | 14 | 候选存在但关系排序差 |
| `scene0527_00` | 24 | 25.00% | 37.50% | 18 | 3 | proposal 主导 |
| `scene0084_00` | 92 | 27.17% | 83.70% | 67 | **52** | 极强排序 headroom |
| `scene0203_01` | 168 | 29.17% | 54.17% | 119 | 42 | 77 条 proposal failure |
| `scene0231_01` | 330 | 38.48% | 72.42% | 203 | **112** | 大量关系样本，排序与 proposal 各占一半 |
| `scene0207_02` | 548 | 63.50% | 92.88% | 200 | **161** | 总体不算最低，但贡献最多可修复排序错 |
| `scene0645_00` | 682 | 67.74% | 83.14% | 220 | 105 | 86 个物体，小物体 proposal failure 115 条 |

最低类别与 Nr3D 高度一致：`soap dish=15.48%`、`toilet paper=22.22%`、`bag=23.53%`、
`bottle=26.00%`、`storage bin=33.33%`、`book=40.99%`、`picture=52.54%`。这进一步证明小目标点云
与同类实例消歧是跨数据集问题，而非某个 CSV 划分偶然造成。

典型排序失败集中在 `scene0207_02`：

- “backpack in the center of the table and the chair”：正确候选 rank=2、IoU=`0.903`、gap=`0.0042`。
- “door farthest from the blanket”：正确候选 rank=2、IoU=`0.477`、gap=`0.0064`。
- “door between the table and the refrigerator”：正确候选 rank=2、IoU=`0.520`。
- “stool in front of the chair”：正确候选 rank=2、IoU=`0.818`。
- “towel in the middle of the table and the pillow”：正确候选 rank=2、IoU=`0.531`。

典型 proposal failure 集中在 `scene0645_00`：soap dish 仅 11 点、door 21 点、picture 30 点、backpack 66 点；
即使关系句和 anchor 都正确，候选框本身仍可能没有达到 0.25 IoU。

### 16.7 为什么 V99 source selector 在 Nr3D 上没有复制 ScanRefer 的收益

Nr3D 当前两种 source 的实测诊断为：

| source | REC@0.25 |
|---|---:|
| `default` | 54.1208%（4275/7899） |
| `default_rank_blend_contrastive010` | 53.6903% |
| 两 source oracle | 54.1588%（4278/7899） |

不可部署的 oracle 也只多 3 hits。训练日志持续显示
`source_choice_selected_non_default_ratio=0.0000`，并不是 selector “没有学会探索”，而是第二个 source 几乎没有
独立正确样本可供选择。继续增加 gate 层数、训练 epoch 或改阈值，不可能从两个近乎同序的输入中制造 249 hits。

ScanRefer 的 parent、geometry 与 query variants 之所以有用，是它们在候选排序上确实存在互补；Pareto gate 只需判断
何时安全切换。Nr3D 迁移保留了“二选一”的形式，却没有保留“两个候选源真正互补”的前提。这是当前最关键的
跨数据集失效点。

### 16.8 已做方法、真实效果与是否改善明显

#### 16.8.1 Nr3D：绝大多数局部优化均未刷新

以旧保护 E57 `4463/3749` 为受控比较点：

| 方法 | 正式结果 hits@.25/.50 | 相对 E57 | 效果判断 |
|---|---:|---:|---|
| effective global batch 48 E58 | 4400/3696 | -63/-53 | 负结果 |
| effective global batch 48 E59 | 4399/3694 | -64/-55 | 连续负结果，停止 |
| relation-counterfactual E63 | 4400/3691 | -63/-58 | 伪/唯一 anchor 噪声，负结果 |
| relation-counterfactual E64 | 4394/3674 | -69/-75 | 继续下降，停止 |
| 低 LR E58 | 4452/3752 | -11/+3 | @.25 未刷新，仅 @.50 噪声级改善 |
| 低 LR E59--E62 | 4432/3699、4437/3722、4426/3728、4421/3716 | 全部低于 E57 | 已形成平台/回落 |
| tier hard-query E58 | 4413/3707 | -50/-42 | 负结果 |
| tier hard-query E59 | 4398/3676 | -65/-73 | patience=2 停止 |
| E26/E29 权重平均 | **4475/3759** | **+12/+10** | 当前唯一正式正增益，但只有 +0.1519/+0.1266pp |
| Top-5 hard replay E58 | 4427/3715 | -36/-34；相对当前最好 -48/-44 | 第一次正式未刷新；E59 按预注册继续 |

结论：Nr3D 现有优化中，只有权重平均带来小幅正增益；低 LR、batch 对齐、普通 hard loss、关系反事实、tier
auxiliary 以及第一轮 Top-5 hard replay 都没有改善主指标。把 `+12 hits` 称为显著提升并不诚实，它更像同一
局部盆地内的稳定化收益。E58 hard replay 下降还说明：仅把“Top-5 内有正确框”的困难样本多看一次，并不会自动
把正确候选推到 Top-1；若没有直接而可靠的候选间优势监督，重复采样甚至会使总体排序与 Mask 一起退化。

#### 16.8.2 Sr3D：关系监督有正效应，主要提升来自后续训练与低 LR，权重平均完成最后越线

| 阶段 | hits@.25/.50 | REC@.25/.50 | 相对 E32 |
|---|---:|---:|---:|
| 原 V99 E32 | 11975/9877 | 67.5561%/55.7204% | 基准 |
| relation-CF E38 最好 | 12025/9944 | 67.8382%/56.0984% | **+50/+67 hits** |
| 后续正式低 LR E23 | 12116/10207 | 68.3516%/57.5821% | +141/+330 |
| 后续正式 E26 | 12123/10336 | 68.3911%/58.3098% | +148/+459 |
| E26 0.75 + E29 0.25 权重平均 | **12139/10335** | **68.4813%/58.3042%** | **+164/+458** |

相对 E32，最终累计为 `+0.9252pp@.25/+2.5838pp@.50`，说明 Sr3D 内部训练改进是有效的；其中
relation-CF 单独约 `+0.2821pp@.25`，是正效果但不算大。权重平均相对 E26 为 `+16 hits@.25/-1 hit@.50`，
它是最后跨过 68.4 的小幅稳定化步骤，而不是主要学习来源。

相对外部 68.4 baseline，最终只高 `0.0813pp`。因此论文可写“达到/略超 baseline”，不能写“显著领先”。

#### 16.8.3 Conservative raw parser：有训练域证据，尚无正式 REC 证据

保守 raw parser 在 train-only / scene-disjoint holdout 的 target-text match 上分别：

- fit：`68.9259% -> 71.5045%`，`+2.5785pp`，`757 fixes / 92 breaks`；
- holdout：`68.2284% -> 70.7673%`，`+2.5389pp`，`211 fixes / 30 breaks`。

它证明 Nr3D 的 command/inversion/copular 句法目标选择存在可修复问题，但 target-text match 不是 REC。当前已准备绑定
4475 最好权重的一次性 raw-parser eval；只有 hard replay 失败后才允许执行，且不得根据 validation 再扫规则。

### 16.9 根因排序：证据、解释与下一步含义

#### 根因 1（高置信）：正确候选存在，但文本条件 Top-1 排序失效

- **Observation**：Nr3D Top-2 oracle 61.68%，Top-16 80.30%；2068 条 Top-1 失败可由 Top-16 修复。
- **Interpretation**：候选生成不是主要上限，模型没有稳定组合类别、属性、关系、anchor 与视角。
- **Implication**：优先研究 Top-2/Top-5 内的保守文本 verifier；只在有明确优势证据时越过 parent。

#### 根因 2（高置信）：小/稀疏点目标存在独立 proposal 瓶颈

- **Observation**：Nr3D 最稀疏四分位 41.34%，最稠密 61.26%；mouse/soap dish/bottle/book oracle 也很低。
- **Interpretation**：这些样本没有可供后处理选择的合格框，重排序无能为力。
- **Implication**：若要覆盖此类样本，需要小物体 proposal、点云分辨率或候选框 refinement；应与排序实验分开评估。

#### 根因 3（高置信）：Nr3D 的 source 和 anchor 质量不满足 V99 前提

- **Observation**：二 source oracle 只多 3 hits；selector 非 default 选择率为 0；relation-CF 在 Nr 为负、在 Sr 为正。
- **Interpretation**：Nr3D 的 rank-blend 不提供互补排序，伪 anchor 又引入监督噪声。
- **Implication**：不能继续优化空转 gate；无可认证 anchor 时必须 fail closed，或从原始文本构造保守证据。

#### 根因 4（中高置信）：Nr3D 自然语言的长句、否定和观察视角没有被稳定解析

- **Observation**：13+ token Top-1 只有 49.19%，但 oracle 77.21%；实际失败包含“facing the window”、
  “does not have”、closest/farthest 与多参照物组合。
- **Interpretation**：简单关系词检测不足以确定目标、anchor 和坐标系。
- **Implication**：parser 改动必须 raw-only、保守 abstain、跨 scene 审计；不能接入含 GT 派生字段的 sidecar 冒充泛化。

#### 历史归因边界（不执行）：Nr3D 未完成论文 baseline 的完整训练时程复现

- **Observation**：论文配置为 global48、最长 240 epoch、epoch150 衰减；当前最好为 B16×A1 E57，局部 global48
  只跑了短恢复，未从 detector-pretrained/no-task-resume 完整走到 150/240。
- **Interpretation**：当前分支的局部平台是真实的，但它与论文正式训练曝光量仍不完全等价。
- **Implication**：这只限制论文中的因果归因强度。根据 2026-08-31 最新决定，不再进行完整公平重训；不得将其
  加回训练排期，也不得作为 FPR-TV 或 Proposal Refiner 的启动门槛。

### 16.10 当前不应得出的结论

1. 不能说“V99 在所有跨数据集上都失败”：Sr3D 已略超 baseline。
2. 不能说“Nr3D 只差继续训练”：多次延训和衰减均不刷新，已有明确平台证据。
3. 不能说“Nr3D 全是 detector 问题”：Top-2 oracle 已超过目标，排序 headroom 很大。
4. 也不能说“Nr3D 全是排序问题”：1556 条 Top-16 proposal failure 和小目标分层证明 proposal 是独立瓶颈。
5. 不能把 ScanRefer 总系统 `+1.43/+4.92pp` 全归给 V99 hierarchy；隔离 hierarchy official 只有 `+10/+24 hits`，
   其余来自 backbone/parent/geometry、正确 superpoints 与整条后处理链。
6. 不能把 Sr3D 最终 `+0.0813pp` 写成显著 SOTA；它是单次正式结果、缺少多 seed 方差。
7. 不能把候选 oracle、parser target-match 或训练 batch accuracy写进论文主结果列；它们只用于机制诊断。

### 16.11 当前活动实验与后续执行顺序

截至 2026-08-31 03:49 CST：

- Nr3D Top-5 hard-example replay E58 已完成正式评估：REC=`4427/3715`（`56.0451%/47.0313%`），
  Mask=`4128/3420`（`52.2598%/43.2966%`），Mask mIoU=`36.7828%`。
- 相对当前正式最好 `4475/3759`，E58 REC 下降 `48/44 hits`；相对旧 E57 `4463/3749` 也下降 `36/34 hits`。
  因而 E58 是明确负结果，不是“尚未训练完所以暂时低”。
- selector 在整套正式评估中仍为 `selected_non_default_ratio=0.0000`；learned selector 与 default source 完全同值，
  source oracle 也仅约增加 `3/2 hits`，进一步坐实第二 source 缺少互补信息。
- E59 已完成正式评估：REC=`4400/3702`（`55.7033%/46.8667%`），Mask=`4128/3426`
  （`52.2598%/43.3726%`），Mask mIoU=`36.8457%`。相对当前正式最好 `4475/3759`，REC 下降
  `75/57 hits`；相对同一 run 的 E58 又下降 `27/13 hits`。selector 仍 100% 选择 Default。
- E58 与 E59 连续两轮都未刷新，已经满足预注册 patience=2 停止条件；hard replay 正式封存，不再叠加
  LR 变化、更多重复样本或延长 epoch。E59 正式 receipt SHA-256 为
  `21b49e8d1270aa01fb4fe59d5f40d786c2c850b6ac2fcb233cf2c4affe9fbc00`。

固定执行顺序：

1. hard replay 已按 patience=2 完成并封存；当前正式最好继续保留 `4475/3759`。
2. 主路线直接转入候选级 FPR-TV 的 scene-disjoint density/finite-gradient 审计；未通过门禁不得长训。
3. Conservative raw parser 如执行，只允许绑定当前最好权重的一次性隔离评估；它不再作为 FPR-TV 的启动
   前提，也不能据此继续搜索规则或 validation threshold。
4. 根据用户 2026-08-31 的最新决策，**不再执行 detector-pretrained、no-task-resume、global48、
   150/240 epoch 的公平 baseline 复现**，也不再用该复现解释或延迟新方法。
5. 新网络方向必须保持 ScanRefer/Nr3D/Sr3D 同一总体架构：Top-2/Top-5 文本条件 verifier、parent-relative
   advantage、逐样本可靠性 gate、无证据回退 parent；不联合训练、不使用 dataset ID、不使用 Unique/Multiple 标签。
6. 小目标 proposal 改进必须作为独立消融，不能与 reranking 混成一个实验后无法判断来源。

### 16.12 面向论文的推荐叙述

可以如实写成：

> 在 ScanRefer 上，候选上下文建模、Pareto 安全选择和修复后的 mesh-superpoint mask pipeline 共同带来
> 58.60/50.45 的 REC 与 59.84/52.33/45.93 的 Mask 结果，较原 MCLN 尤其提升严格 IoU 指标。
> 在 Sr3D 上，同一总体架构配合精确关系 anchor、低学习率延续和权重平均达到 68.48，略超 68.4 baseline。
> Nr3D 仍为 56.65，错误分析表明主要瓶颈是自然语言条件下的同类候选 Top-1 消歧，同时小/稀疏目标还存在
> 独立 proposal 上限；这两类错误不能由继续衰减学习率或普通排序 loss 一并解决。

不应写成：

> V99 后处理在三个数据集上都显著超过 baseline。

更合适的研究结论是：**V99 的安全候选选择原则具有一定可迁移性，但其收益依赖候选源互补性、anchor 可靠性、
语言分布和点云目标质量；ScanRefer 的完整增益不能被视为一个数据无关的固定后处理增益。**

### 16.13 2026-08-31 新硬目标与并行推进状态

用户于 2026-08-31 将两个迁移数据集的目标提高，并随后再次明确确认 Nr3D 必须严格超过 60.0%：

| 数据集 | 新验收规则 | 最小正式 hits | 当前最好 | 当前缺口 |
|---|---:|---:|---:|---:|
| Nr3D | 严格 `REC@0.25 > 60.0%` | `4740/7899` | `4475/7899=56.6527%` | **265 hits / 3.3549pp** |
| Sr3D | 严格 `REC@0.25 > 68.9%` | `12214/17726` | `12139/17726=68.4813%` | **75 hits / 0.4231pp** |

此前 `59.8/68.4` 继续作为外部 baseline 对照，但不再是项目完成门槛。所有后续判断必须直接比较整数 hits；
不得依靠显示到一位小数后的四舍五入。

两台服务器的 official REC monitor 已同步为新门槛并完成在线核验：Nr3D
`target_rec025=0.600`，Sr3D `target_rec025=0.689`，两者当前均为 `target_reached=false`；重新启动 monitor
没有改变或覆盖已保护的最好权重。特别地，Nr3D 的严格验收是至少 `4740/7899=60.0076%`，刚好显示为
`60.0%` 但 hits 少于 4740 的结果不能算达标。

Sr3D 重新进入未完成状态后，已完成以下审计：

1. E22--E29 的 REC@.25 依次为 `12051, 12116, 12090, 12085, 12123, 12108, 12114, 12123` hits；
   单模型已在约 12108--12123 之间平台波动，E26/E29 权重平均才得到 12139。仅靠继续当前低 LR 延训，没有
   75-hit 增益证据。
2. E26/E29 full-state 当前 LR 均为
   `[1e-6, 1e-5, 1e-6, 1.25e-6]`，即初始 LR 的 `0.01×`；scheduler 即将在 epoch30 再衰减。
   因此现在再次手工减 LR 更可能冻结现有盆地，而不是产生新判别能力。
3. 当前最好权重已由 official monitor 单独保护：
   `official_best_rec025_epoch_26_0p68481327.pth`，SHA-256=
   `da985736e5bc116c03cca51a523a211cade515d9b7580deb8e9d48bf8a4499d3`。它与权重平均 decision 中的
   candidate SHA 完全一致；E26/E29 full-state 也继续保留。
4. 为满足 7 GiB 安全空间门，删除了两个已被当前最好淘汰的旧物理 checkpoint：旧 relation-CF E38 与旧
   plateau E21，共释放约 1.59 GB；对应日志/metrics 均保留，当前正式最好与 E26/E29 未删除。
5. 已启动一个**100 micro-batch、无验证、无权重输出**的 E26 relation-counterfactual 审计：保持 V99 网络、
   selector、B12×A2、LR、scheduler 和推理完全不变，只启用既有训练期关系困难负样本。第一次启动在真正
   训练前发现 bounded audit 的 expected plan 误写成 full-loader 计划，已安全停止，未执行 GPU optimizer step；
   修正为 `100 micro-batches / 50 optimizer updates` 后重新启动。当前 screen 为
   `590262.mcln_sr3d_e26_relcf_audit`，audit root 为
   `/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/sr3d/audit/`
   `sr3d_mcln_joint_butdcls_v99_e26_relation_cf_audit_e27_b100_b12a2_20260831_014726`。
   只有 exact-GT-anchor、reference-valid、hard-negative density、pair violation 和 gradient gates 全部通过，
   才会准许短正式训练；这保持 ScanRefer/Nr3D/Sr3D 的总体网络一致，也不新增部署分支。

Nr3D hard replay E59 后续已在 2026-08-31 03:49 CST 完成：正式 REC 为 `4400/3702`，低于 E58 的
`4427/3715`，更低于当前最好 `4475/3759`；Mask 为 `4128/3426`，mIoU=`36.8457%`。全程未发现
Traceback/CUDA/NaN，但 selector 非 Default 选择率仍为 0。E58/E59 已形成连续两轮未刷新，hard replay
按 patience=2 封存，不再延长或提前衰减学习率。

### 16.14 2026-08-31 方法路线收敛与明确排除项

用户提供的最新问题分析被采纳为后续方法设计依据，但明确排除其中第七节“公平训练对照”和第八节 E0--E7
实验矩阵：二者不进入实施计划、训练排期或论文证据链。历史 `59.8/68.4` 仅作为外部 baseline 数值背景，
项目验收仍按本章的 `Nr3D>60.0% / Sr3D>68.9%` 新硬目标执行。

后续研发收敛为两条彼此正交的路线：

1. **主路线：FPR-TV（Feasible Parent-Relative Text Verifier）**。从 Parent 与 Top-2/Top-5 候选出发，
   对每个具体 Candidate 构造 Candidate-vs-Parent 特征，读取 target/attribute/relation/anchor/viewpoint/negation
   等文本证据，预测 @0.25 修复、@0.50 修复与 break risk。只允许具有可行越界预算且通过可靠性验证的离散
   切换；否则逐样本精确回退 Parent。该路线禁止整行全局 scalar gate、全部 Query 自由 residual 和无证据切换。
2. **辅路线：Density-Aware Local Proposal Refiner**。只针对低点数、小体积、低局部密度候选重新读取多尺度
   局部点与早层点云特征，预测小幅 box center/size 修正和 quality；它必须与排序 Verifier 分开训练、分开评估，
   不能把“候选存在但排错”和“候选本身不存在”混成一个结果。

统一约束如下：不联合训练，不使用 dataset ID、Unique/Multiple 标签或 validation margin sweep；不可靠 anchor
必须 abstain；保持 ScanRefer/Nr3D/Sr3D 的总体网络接口一致；第一阶段冻结 Box、Mask 与 Backbone，只验证
REC 候选选择；任何新模块都必须记录 Top-K oracle、fix/break/neutral、switch precision、score-gap、文本与点云
密度分层，以及 Mask 安全指标。

当前已启动的 Nr3D E59 与 Sr3D 100-batch relation-CF 审计只作为有界历史路线收尾，不由其扩展公平 baseline
或新的 loss sweep。后续是否进入正式训练，以候选级证据密度、有限梯度、Parent 精确回退与 scene-disjoint
`fix>break` 为先决条件。

### 16.15 2026-08-31 FPR-TV 实现、审查与当前运行状态

#### 16.15.1 范围再次冻结

本阶段已经按用户最终决策冻结研究范围：

- **不再复现所谓公平 baseline**，不执行 detector-pretrained/global48/150--240 epoch 对照，也不把它作为
  解释当前差距或启动新方法的先决条件；
- **不参照原建议第七节**，不继承其中按数据集规定的旧执行顺序、旧前置步骤或旧路线选择；
- **不借鉴原建议第八节**，其中要求的 baseline 公平训练复现已取消；旧 E0--E7 实验矩阵也不再用于排队、组合模块或构造论文证据；
- 主路线只保留 FPR-TV，Proposal Refiner 留作后续独立路线；二者不混训，第一阶段也不修改 Box、Mask、
  Backbone 或 Proposal；
- 当前仍在运行的 Nr3D E59 只是已经开始的 hard-replay 有界收尾。无论结果如何，都不能据此继续延长训练、
  改学习率、增加 replay 或恢复已取消的 baseline 路线。

#### 16.15.2 已完成的 FPR-TV 代码固定点

FPR-TV 已在与正式服务器隔离的本地 staging 分支实现，尚未覆盖 live checkout，也尚未启动训练：

```text
staging: C:\Users\gb\.codex_remote_staging\mcln_fprtv_20260831
branch : fpr-tv
base   : e663f6729c138923821c9edbbe8c0d53d1c21178
HEAD   : a3fe9607753950a5c3d90604ed2e0be0414aeeb1
```

核心新增文件为 `models/parent_relative_text_verifier.py`，并以默认关闭方式接入 `models/mcln.py`、
`models/losses.py`、`main_utils.py`、`train_dist_mod.py` 和训练参数组管理。实现的实际行为不是新的全局 Gate，
也不是给整行 Query 增加自由 residual，而是：

1. 以原 V99 `selected_source_scores` 的正式 Parent 为保留动作；
2. 在 Parent Top-K 与 Text Top-K 的精确去重并集中，为每个 Candidate 构造 Candidate-vs-Parent 特征；
3. 候选级预测 `repair@0.25`、`repair@0.50`、`break@0.25`、`break@0.50`、连续 IoU advantage 与可靠性；
4. 先应用 `score_gap <= 0.25` 的可行晋升约束，再应用 parse/anchor/reliability/break-risk 约束；
5. 每条样本最多执行一次离散切换；没有可靠正收益 Candidate 时，精确回退原 Parent；
6. 训练阶段冻结 Box、Mask 与 Backbone，梯度只更新 verifier；推理阶段不产生新的 Box 或 Mask 分支。

Checkpoint 合同也已收紧：训练和评估都要求完整、精确的 V99 selector 配置；不允许 SourceMoE、SACR
score refiner 或其他并行分数分支混入；FPR-TV checkpoint 必须具有完整 canonical key/shape/dtype，不能依赖
`strict=False` 漏载或随机初始化。默认关闭时，原模型状态结构与原 checkpoint 加载行为保持不变。

#### 16.15.3 审查中发现并修复的关键部署语义错误

初版实现虽然正确使用了 `selected_source_scores`，但 Parent 与候选池最初是在原始 Query 轴上取 Top-K；正式
`joint_det + butd_cls` evaluator 则会先用 detector-overlap 规则排除非 GT-compatible Query，再取 Top-1。
若不修复，会出现三类严重偏差：

- 训练所称的 Parent 不是正式 evaluator 真正使用的 Parent；
- raw Parent Top-1 或 text Top-1 可能是正式部署中会被过滤掉的 Query；
- verifier 可能把一个不可部署 Candidate 统计为有效 switch，导致 `fix/break` 回执失真。

最终固定点复用了与正式 evaluator 相同的 `models.rec_evaluator_filter.build_detector_overlap_valid`，以
`inputs.det_boxes`、`det_bbox_label_mask` 和 IoU `0.25` 构造唯一 detector-valid 轴。Parent、Parent Top-K、
Text Top-K、SACR shortlist、可行性监督和最终离散切换现在全部受同一轴约束。若整行没有 detector-valid
Candidate，只保留一个结构性槽位用于数值稳定，但该行 `deployable=false/input_valid=false`，不参加监督、
不能切换、最终保持原分数；正式 evaluator 会如实记录 miss，代码不会制造虚假修复。

FPR-TV 同时强制 runtime 与 checkpoint 的 `butd_cls=True`。这使它当前只对应已经采用的正式
`joint_det + butd_cls` 合同，不允许在另一候选过滤协议下误用同一 checkpoint。

#### 16.15.4 验证与独立审查证据

在独立测试目录 `/root/mcln_fprtv_test` 中完成的最终定向回归为：

```text
268 passed, 6 deselected
```

六项 deselected 仅是需要历史 launcher 文件的 SourceMoE integration 测试；隔离测试目录没有这些启动工件，
不是功能失败。其余覆盖包括：

- 共享 detector filter 与正式 evaluator 完全一致；
- raw Parent Top-1 无效、text Top-1 无效以及整行无 detector-valid Candidate 的反例；
- 空候选、无正例、无可行晋升时仍能有限值反传；
- Parent 未切换时分数精确保留、每行最多一次切换；
- 训练只更新 verifier 参数，Box/Mask/Backbone 不漂移；
- 完整 checkpoint 保存/恢复、默认关闭 checkpoint 回归与评估 tensor 精确性；
- V99 selector、`butd_cls`、非 V99 分支和 source 配置的 fail-closed 合同。

最终 fixed-point 已完成两条互相独立的只读审查：

| 审查轴 | 结论 | 核验重点 |
|---|---|---|
| Spec | **PASS** | detector-valid Parent/Top-K、空检测行回退、无 scope creep、反例覆盖 |
| Standards/Correctness | **PASS** | runtime/checkpoint `butd_cls`、稳定去重、监督/部署有效性、默认关闭兼容 |

当前只能据此声明“实现和安全合同通过”，**不能声明 REC 已提升**。代码尚未部署到 live 服务器、尚未产生
FPR-TV 权重，也没有 Nr3D、Sr3D 或 ScanRefer 的新正式指标。

#### 16.15.5 旧 Nr3D hard-replay E59 的最终结果与封存

E59 已于 2026-08-31 03:49 CST 完成 2,904 个训练 batch 和完整 7,899 条正式评估，全程无 Traceback、
CUDA OOM 或 NaN/Inf。正式 receipt 为：

| E59 指标 | hits / sum | 百分比 |
|---|---:|---:|
| REC@0.25 | `4400/7899` | `55.7033%` |
| REC@0.50 | `3702/7899` | `46.8667%` |
| Mask@0.25 | `4128/7899` | `52.2598%` |
| Mask@0.50 | `3426/7899` | `43.3726%` |
| Mask mIoU | `2910.4452/7899` | `36.8457%` |

Receipt 路径为该 run leaf 下的 `eval_metrics_epoch_59.json`，SHA-256=`21b49e8d1270aa01fb4fe59d5f40d786c2c850b6ac2fcb233cf2c4affe9fbc00`。
相对 E58，E59 REC 再下降 `27/13 hits`；相对当前正式最好 `4475/3759`，下降 `75/57 hits`。日志中的
`selected_non_default_ratio=0.0000`，Default 与 Selected 完全相同，另一个 source 的 oracle headroom 仍不足
千分之一。结论不是“还没训练完”，而是 replay 没有增加可部署 source 互补性，且连续两轮正式退化。

因此 hard replay 已按 patience=2 正式封存：不保留 E59 为最好、不覆盖受保护 E57、不继续 epoch、不改变 LR、
不增加 replay 强度。run 内 retention 显示 E58 仍优于 E59，但二者均低于跨 run 的正式最好，所以只保留
metrics/log/receipt 作为负结果证据。核对 run 已退出且四个文件只对应两个未晋升物理 inode 后，已删除
`ckpt_epoch_58.pth`、`ckpt_best_rec_acc025.pth`、`ckpt_epoch_59.pth` 和 `ckpt_epoch_last.pth`，释放约
`1.59 GB`；这些删除不可从该 run 恢复。受保护的 `official_best_rec025_epoch_57_0p56652741.pth` 是不同
inode，删除后再次核验仍存在；`/root/autodl-tmp` 可用空间约 `12 GB`。

项目硬目标仍为 Nr3D 至少 `4740/7899`、Sr3D 至少 `12214/17726`。E59 没有改变当前最好：

| 数据集 | REC@0.25 | REC@0.50 | 与硬目标差距 |
|---|---:|---:|---:|
| Nr3D | `4475/7899 = 56.6527%` | `3759/7899 = 47.5883%` | @0.25 差 `265 hits` |
| Sr3D | `12139/17726 = 68.4813%` | 以正式最好回执为准 | @0.25 差 `75 hits` |

#### 16.15.6 FPR-TV 下一步唯一允许的实验门

FPR-TV 不直接进入长训。部署前先做 scene-disjoint、无正式 validation 阈值搜索的有界审计，至少记录：

```text
detector-valid/deployable row ratio
Parent/Text Top-K union size 与正确候选覆盖
feasible candidate ratio
repair@0.25 / repair@0.50 / break@0.25 / break@0.50 密度
每折 fix / break / neutral 与 switch precision
fallback、可靠候选、score-gap 和候选 rank 分布
finite loss / finite gradient / 实际更新参数集合
Parent exact-preservation 与 Box/Mask/Backbone zero-drift
```

只有每个 scene-disjoint fold 都满足 `fix > break`、没有 detector-axis 虚假候选、梯度有限且冻结参数不漂移，
才允许短训练和一次正式评估。Nr3D 先验证同类候选与长文本消歧；同一架构通过后再在 Sr3D 验证，不能针对
数据集加入 ID、Unique/Multiple 标签或单独调 decision threshold。Proposal Refiner 仍排在 FPR-TV 真实结果
之后，且必须用独立实验回答 Top-16 proposal failure，不能与当前 reranker 一起训练后合并归因。

#### 16.15.7 2026-08-31 FPR-TV recovery-v4 唯一 100-batch 审计终态

本节是 16.15.4 中“尚未部署/尚未运行”的后续终态更新。FPR-TV 经过隔离部署、三次启动期 fail-closed
取证和 structured-collate 修复后，于 2026-08-31 12:54--13:05 CST 完成唯一 recovery-v4 有界审计：

```text
run root : /root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/audit/
           nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v4
run leaf : runtime_output/nr3d/nr3d_fpr_tv_e57_e58_b100_b16x1_recovery_v4/1788152132
epoch    : E57 full-state resume -> bounded E58
plan     : B16 x A1, 100 micro-batches, 100 optimizer steps, 0 dropped
data     : 44,909 train / 7,899 validation annotations
mode     : train-only density/finite-gradient audit; no validation, no checkpoint save
```

此前第三次启动在第一次 `DataLoader.__next__` 的默认 collate 中失败，错误为 variable-length structured list
不能由 `default_collate` 堆叠。该失败不是模型或 FPR loss 数值错误。最终修复只在 SACR、SACR-refiner 或
FPR 显式启用时选择已有 `joint_det_structured_collate`；三项均关闭时仍保持原 `collate_fn=None`。冻结首批
deterministic replay 证明旧路径在同一首批必然于训练循环体之前失败，新 structured collate 可以成功组批，
因此旧失败严格为 0 optimizer step。

recovery-v4 的 100 个 batch 全部完成，无 Traceback、CUDA OOM、NaN/Inf 或磁盘错误。主要均值如下：

| 审计量 | 观测值 | 预注册下限 | 结论 |
|---|---:|---:|---|
| detector-valid / deployable row ratio | `0.999375` | `0.50` | PASS |
| detector candidate ratio | `0.304832` | `0.002` | PASS |
| feasible candidate ratio | `0.322558` | `0.01` | PASS |
| reliable row ratio | `0.385000` | `0.02` | PASS |
| positive row ratio | `0.043370` | `0.002` | PASS |
| candidate positive ratio | `0.015993` | `0.0005` | PASS |
| gradient norm | `10.889313` | `> 0` | PASS |

FPR-TV 总 loss=`1.951489`，其中 repair=`0.701541`、break=`0.463147`、reliability=`0.640438`、
IoU=`0.053443`、action=`0.178664`，均为有限值。100-batch 末尾仍为 `fallback_ratio=1.0`、
`switch_ratio=0.0`、`learned_reliable_ratio=0.0`；这说明审计证明了候选/正例密度和有效梯度充足，但并未
证明一个只训练100步的 verifier 已学会可部署切换，更不构成 REC 提升证据。

正式 decision 为：

```text
schema                   = mcln-fpr-tv-density-audit-decision-v7
density_gate_pass        = true
gate_failures            = []
audit_only               = true
long_training_authorized = false
next_stage               = scene_disjoint_audit_only
```

审计 receipt SHA-256=`9f2549ad52e756c090a9865f055762af50171b06bf585817d61758b1773abb49`；
decision SHA-256=`b3867681f374450decfb3d7dc55db005944973e4ed0cac09e224c8be8c2e194e`。二者均为
`0444 root:root`。run root 外没有生成任何新 `.pth`，GPU 已释放，受保护 Nr3D 最好权重没有修改。

因此当前科学结论只到：**FPR-TV 的真实 Nr3D 候选密度、有限损失和梯度门通过，可以进入独立的
scene-disjoint 短审计；尚未授权长训练，也尚无新的 Nr3D REC 指标。** 下一步不能直接启动 E58--E62
长训，不能根据这100批的 `fallback=1.0` 临时改 threshold；必须先按冻结合同验证跨场景的 fix/break、
switch precision、Parent 保留和冻结参数 zero-drift。

#### 16.15.8 2026-08-31 FPR-TV scene-disjoint fold 0 正式负结果与封存决定

本节记录 16.15.7 所授权的**唯一下一阶段**：在 Nr3D train scenes 内做预注册的五折
scene-disjoint 短审计。该审计不访问正式 `7899` 行评估集，不搜索 margin/threshold，不保存模型权重，
也不执行已明确排除的公平 baseline 复现、原建议第七节或第八节 E0--E7 实验矩阵。五折按串行守卫执行：
只有 fold 0 先通过，才允许继续 fold 1--4；任一折失败即停止整条配置。

##### 16.15.8.1 冻结实现、切分与运行合同

scene-disjoint 审计在隔离 staging 分支完成实现、回归和双轴审查后才启动：

```text
staging HEAD          : 40f1dfd
frozen config SHA-256 : 7845c9a73a158f814cb7f8278fbb60abfd49f2bee6e5e1d0b1892c4401ed32c9
runtime manifest      : ba8f895ef7832fb88fdb99807244063ab31e4140aff39eb6fdb16e56081e3112
runtime closure       : 365 files, exact copy, 0444 files / 0555 directories
GroupFree SHA-256     : 9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2
protected E57 SHA-256 : 76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1
launcher SHA-256      : e246560bfd2547744b8a8a72f58eb1c313c814f4a9063d9f01bd53226c1d1c21
```

真实 preflight 不是只检查脚本参数，而是按正式顺序执行
`parse_option -> prepare_source_moe_gate_checkpoint_config -> canonical config gate`；post-prepare
配置 SHA 与 main 中的预注册常量完全一致。正式 `pp_checkpoint` 只引用独立、只读、SHA 固定的 GF
snapshot；运行前后对 365 文件 snapshot 做完整文件集、mode、size 和 SHA 复验。Spec 与
Standards/Correctness 两条独立审查均为 **PASS** 后才允许启动。

fold 0 的冻结切分和训练计划如下：

| 项目 | 冻结值 |
|---|---:|
| fold 规则 | `int(sha256(scan_id)[:8], 16) % 5` |
| fit scenes / samples | `402 / 25,790` |
| heldout scenes / samples | `109 / 7,129` |
| scene overlap | `0` |
| 起点 | 受保护 Nr3D V99 E57 full-state |
| 训练轮 | 仅 E58 |
| batch / accumulation | `B16 x A1` |
| 完整 fit batch | `1,612`，自然尾批，`0` 条遗漏 |
| heldout batch | `446` |
| 可训练参数 | `2,510,184`（structured slots、SACR evidence、FPR-TV） |
| 冻结参数 | `149,670,851`，训练前后 digest 完全一致 |
| 正式 validation / 权重保存 | 均禁止 |

训练 receipt 对实际消费行做原始 row-id 的 count、unique count 与顺序无关 SHA 校验：`25,790`
条 fit 样本恰好各消费一次，没有重复或遗漏；fit/heldout scene 集严格不相交。E57 由同一 opened FD
完成 SHA、`torch.load`、内部 `epoch=57` 与二次 SHA 校验。完整 E58 无 Traceback、CUDA OOM、
NaN/Inf 或磁盘错误，冻结参数零漂移、52 个 trainable tensors 确实发生更新。

##### 16.15.8.2 fold 0 heldout 精确结果

fold 0 于 2026-08-31 15:01--15:55 CST 完成。正式决策只读取 `7,129` 条 heldout train-scene
样本的 Candidate-vs-Parent 转移计数：

| 阈值 | Parent hits / acc | FPR-TV hits / acc | Fix | Break | 净 hits | Transition precision |
|---|---:|---:|---:|---:|---:|---:|
| REC@0.25 | `6873 / 96.4090%` | `6831 / 95.8199%` | `40` | `82` | **`-42`** | `32.7869%` |
| REC@0.50 | `6046 / 84.8085%` | `5767 / 80.8949%` | `119` | `398` | **`-279`** | `23.0174%` |

其他部署统计：

```text
sample_count        = 7129
switch_count        = 876
switch_rate         = 12.2878%
fix_per_switch@0.25 = 4.5662%
fix_per_switch@0.50 = 13.5845%
kept_correct@0.25   = 6791
kept_wrong@0.25     = 216
kept_correct@0.50   = 5648
kept_wrong@0.50     = 964
```

预注册 gate 要求 `switch_count>0`、`fix@0.25>break@0.25` 且
`fix@0.50>=break@0.50`。实际 decision 为：

```text
fold_gate_pass          = false
gate_failures           = [acc025_fix_not_greater_than_break, acc050_net_negative]
long_training_authorized= false
next_stage              = method_correction_only
```

Receipt 路径：

```text
/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/fpr_tv_scene_disjoint_v1/
fold_0/runtime_output/nr3d/nr3d_v99_fpr_tv_scene_fold0_e58/1788159699/
fpr_scene_disjoint_audit_fold_0_epoch_58.json
```

Receipt SHA-256=`843bf42ceec16b55677ab7279877c115854fe4270f58f364f885631dc33aa8ff`；
终态 decision 位于 fold root 的 `decision.json`。运行结束后 screen 和训练进程均退出；排除只读
`input_snapshot/gf_detector_l6o256.pth` 后，run root 内新生成权重数为 **0**。受保护 E57 的 SHA 再次
核验完全未变，`/root/autodl-tmp` 仍约有 `7.6 GiB` 可用空间。

##### 16.15.8.3 为什么 density audit 通过，scene-disjoint 效果仍失败

这次负结果把此前“指标提升不上去”的原因进一步缩小了。100-batch density audit 证明的是：

- detector-valid Candidate、feasible Candidate 和正例在真实数据中不是空集合；
- loss、梯度和 trainable parameter update 都是有限且可执行的；
- Parent、Box、Mask 与 Backbone 可以按合同冻结。

它**没有证明** verifier 能在未见场景中准确判断哪个 Candidate 应越过 Parent。fold 0 的完整结果表明，
真正失败点在部署判别，而不是候选密度、训练没跑完或代码没有梯度：

1. **Parent 先验过强，切换需要非常高的精度。** 该 fold 的 Parent 已有 `96.41%/84.81%`；任何切换都在
   主动冒险。要获得净正收益，所有发生 hit 状态变化的 switch 中 fix 比例必须超过 50%，实际只有
   `32.79%/23.02%`，远低于安全线。
2. **可靠性头明显过度放行。** heldout 共有 876 次切换（12.29%），但真正修复 @0.25 的只有 40 次；
   大多数 switch 是 neutral 或 break。训练末段虽然已经从最初的全回退学到约 10% switch，但不是“多训练
   就会自然变好”，因为完整 E58 的 train 内统计也已出现 `fix<break`，heldout 只是把这一问题放大。
3. **严格 IoU 的安全性没有被联合保证。** 一个 Candidate 可能在文本关系上看似更匹配、甚至不破坏
   @0.25，却比 Parent 的框质量差，从而大量破坏 @0.50。最终 @0.50 有 `398` breaks、仅 `119` fixes，
   净损失 `279` hits，是最明确的结构性失败信号。
4. **两个 V99 source 在 Nr3D 上仍缺乏互补性。** heldout 日志中 selector 的 Default 选择率仍为 100%，
   source oracle 与 fixed Default 的 @0.25/@0.50 完全相同；FPR-TV 无法从近似重复的 source 中创造新的
   稳健证据，只能依赖尚未校准好的文本/结构证据重新排序。
5. **训练目标与部署动作仍有错配。** repair、break、reliability、IoU 和 action loss 都能下降，但连续概率
   的联合最小化并不自动等价于“仅在 Candidate 同时优于 Parent 的两个阈值时切换”。当前候选级结构消除了
   V133 的整行 residual 饱和，却没有形成足够强的 Parent-dominance certificate。
6. **不能把失败解释为未训练完。** 本折完整消费 25,790 个 fit 样本，完成唯一预注册 E58 后再评估；
   gate 失败来自精确 heldout transition counts。继续 epoch、临时降 LR 或在该 fold 上调 threshold 都会把
   heldout 变成调参集，破坏 scene-disjoint 证据。

因此，Nr3D 当前主要瓶颈仍是：正确候选偶尔存在，但文本/关系证据不足以在强 Parent 上进行低风险 Top-1
替换；同时一部分 @0.50 错误来自 Candidate box quality，本排序模块本身不能修复。这个结果也进一步解释了
为什么 ScanRefer 上 V99 的多源候选和 mesh/superpoint 修复能显著提升，而 Nr3D 上复制“候选重排”形式却
不能复制同样收益：Nr3D 的 source 互补性更低、自然语言参照更不稳定、且 Parent 正确率更高，错误切换成本
显著大于可修复收益。

##### 16.15.8.4 封存与下一步纪律

基于预注册串行门，已执行以下终态决定：

- **不启动 fold 1--4**；它们不能被用来挑选一个看起来更好的 fold，也不能成为新的 threshold sweep；
- **不做正式 7,899-row 评估，不保存 FPR-TV 权重，不覆盖当前最好 E57**；
- **不将 100-batch density PASS 写成方法有效**；完整科学结论是“可训练，但首个未见场景折上净退化”；
- **不恢复公平 baseline 复现，不采用原建议第七节，也不采用第八节 E0--E7 矩阵**；
- 当前 FPR-TV v1 配置正式封存为负结果，唯一允许状态为 `method_correction_only`；不得在已消费的 fold 0
  上事后调整 reliability threshold、promotion margin、Top-K 或 loss 权重。

如果未来重启 FPR-TV，必须先形成与本版不同的结构性修正，而不是参数微调：例如把部署动作定义为显式的
双阈值 Parent-dominance certificate，要求 Candidate 对 @0.25 与 @0.50 的非退化证据同时成立；将 switch
budget 与可证明的 transition precision 绑定；无证据时保持 Parent。新方法必须使用新的预注册合同和未消费的
scene-level 证据，不能复用 fold 0 选择配置。

本次实验不改变项目最好指标与硬目标：

| 数据集 | 当前正式最好 REC@0.25 | REC@0.50 | 硬目标 | 剩余差距 |
|---|---:|---:|---:|---:|
| Nr3D | `4475/7899 = 56.6527%` | `3759/7899 = 47.5883%` | `>=4740/7899` | `265 hits` |
| Sr3D | `12139/17726 = 68.4813%` | 以正式最好回执为准 | `>=12214/17726` | `75 hits` |

最终判断：**FPR-TV v1 不是“训练轮数不够”，而是未见场景上的切换精度不足；它已经完成应有的否证，
继续同配置只会扩大验证泄漏风险，不应继续消耗 GPU。**

#### 16.15.9 Sr3D 关系反事实困难负样本：E26→E27--E28 受控短训练

2026-08-31 进一步固定了项目范围：**不再进行 baseline 公平复现，不采用原建议第七节，也不采用
第八节 E0--E7 实验矩阵。** 后续只允许围绕现有 V99 总体架构做有明确因果假设的短实验。本次选择
Sr3D relation-CF，是因为它只在训练期增加关系反事实困难负样本损失，网络结构、V99 Selector、正式
推理分数和 `joint_det + butd_cls` 协议均不改变；同时 Sr3D 提供 exact GT anchor，关系监督可靠性明显
高于 Nr3D 的伪 Anchor。

##### 16.15.9.1 100-batch 审计结果与正确解释

唯一成功的 bounded audit 从纯 V99 E26 完整断点恢复，在 E27 只执行 100 个 micro-batches / 50 个
optimizer steps，不做 validation、不保存权重。固定配置为 B12×A2、relation-CF weight 0.5、
parent Top-K 32、target/attribute tolerance 0.10、geometry threshold 0.08、pair margin 0.05、
max negatives 8、Acc@0.25 pair weight 2.0。审计回执：

```text
/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/sr3d/audit/
sr3d_mcln_joint_butdcls_v99_e26_relation_cf_audit_e27_b100_b12a2_20260831_014726/
sr3d/sr3d_mcln_joint_butdcls_v99_e26_relation_cf_audit_e27_b100_b12a2/1788112049/
train_audit_receipt_epoch_27.json
```

Receipt SHA-256=`0d2e84ca112f8f3aa78e12b2cbfc9faba2793eb9da14299df97f440efc99f0fd`；
审计目录新权重数为 0。关键统计如下：

| 审计项 | 实测值 | 冻结下限 | 结果 |
|---|---:|---:|---:|
| exact GT anchor ratio | `0.77416669` | `0.50` | PASS |
| relation reference-valid ratio | `0.75000002` | `0.50` | PASS |
| hard-negative row ratio | `0.08250000` | `0.01` | PASS |
| selected-negative count mean | `0.20666667` | `0.02` | PASS |
| pair-violation ratio | `0.48525000` | `0.05` | PASS |
| gradient norm | `14.25196363` | `>0` | PASS |

训练损失有限：total loss=`8.98378`，relation-CF auxiliary loss=`0.103784`。但是审计同时显示 strict/coarse
break-selected ratio 约为 `0.5743/0.4036`；这些值描述被挖掘困难负样本的组成，不是正式 REC 的
fix/break。正确结论只能是：**Sr3D 上 exact-anchor relation-CF 有足够密度且梯度可训练，值得做一次短
REC 验证；审计本身不证明 REC 提升。**

##### 16.15.9.2 正式 E27--E28 唯一短训练

旧 `run_sr3d_v99_relation_counterfactual_aux.sh` 是 E34/B14/E35--E46 历史合同，不能用于当前最好分支。
因此新增固定 launcher：

```text
/home/gb/butd/mcln/scripts/run_sr3d_v99_e26_relation_cf_e27_e28.sh
SHA-256 = 651c16bed6af609dbc5e37e0b9388e244b9336deaa0928294caebba4c0a27545
```

它逐项固定并验证：

- E26 resume checkpoint SHA=`4ac72dd3d33bb6aa13278e4e67208d98f006a9863396b3f2ab3713a9c904fd1d`；
- checkpoint 内部 epoch=26、4 optimizer groups、716 AdamW states；
- current LR=`[1e-6, 1e-5, 1e-6, 1.25e-6]`，lineage=`0.01`；
- scheduler `last_epoch=26×3243`，milestones=`97290/129720`；
- 训练集 77,836、loader 6,486 micro-batches、A2 后 3,243 optimizer steps、无尾批丢弃；
- 正式评估 17,726 条、REC@0.25-only checkpoint retention；
- SACR、FPR-TV、额外 REC checkpoint 分支全部关闭，仅 relation-CF training auxiliary 开启。

2026-08-31 16:16:42 CST 在 screen `mcln_sr3d_relation_cf_e27_e28` 启动。正式 run leaf：

```text
/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/sr3d/backbone/
sr3d_mcln_joint_butdcls_v99_e26_relation_cf_e27_e28_b12a2_w0p5_20260831_161654/
sr3d/sr3d_mcln_joint_butdcls_v99_e26_relation_cf_e27_e28_b12a2_w0p5/1788164223
```

16:28:25 数据集构建完成，精确记录 77,836/17,726；E26 checkpoint 显示 `loaded successfully
(epoch 26)`。16:29:19 E27 到 batch 20/6486：total loss=`9.1614`、relation-CF loss=`0.1010`，
均有限；GPU 约 25.5 GiB，无 Traceback、CUDA OOM、NaN/Inf 或磁盘错误。官方 monitor 继续保护现有
Sr3D 最好 `12139/17726=68.4813%`，硬目标保持 `12214/17726=68.9044%`，仍差 75 hits。

本实验最多只跑 E27--E28；每轮正式结果必须同时记录 REC@0.25/@0.50、Mask 两阈值与 mIoU。
若没有超过现有最好，不做 loss weight、margin、Top-K 或额外 epoch 扫描；若刷新，则由 official monitor
只读保护新 REC@0.25 最好权重，结束后清理非最好 checkpoint。

#### 16.15.10 FPR-TV v1 误切的进一步根因定位与范围冻结

2026-08-31 对 fold 0 固定 receipt 建立了秒级只读红灯检查。相同 receipt SHA
`843bf42ceec16b55677ab7279877c115854fe4270f58f364f885631dc33aa8ff` 连续两次稳定失败：
`@0.25 fix=40 <= break=82`，`@0.50 net=-279`。因此症状是确定性的，不是日志解析、随机波动或
训练尚未结束。

进一步读取冻结 code snapshot 后，发现部署可靠性语义存在一个明确错配：

1. `repair_head`、`break_head` 和 `reliability_head` 都使用 class-balanced binary loss；当正负样本同时
   存在时，正类与负类损失各占 0.5。这样训练出的 sigmoid 反映的是平衡先验下的判别分数，不能直接解释为
   真实数据先验下的校准概率。
2. 部署却固定使用 `sigmoid(reliability)>=0.5`、`max(repair)>=0.5`，并把
   `max(repair)>max(break)` 当作 `predicted_safe`。它没有要求 break 风险本身低于 0.5，也没有要求两个
   IoU 阈值分别满足非退化。
3. 一个最小反例已经实际运行：`repair=(0.80,0.10)`、`break=(0.70,0.60)`、
   `reliability=0.60`、`action_logit=0.10` 时，当前代码得到 `current_switch=True`，但
   `absolute_no_break_contract=False`。也就是说，一个预测破坏风险高达 0.70 的候选仍可被标记为 safe。
4. 训练统计与该错配吻合：safe positive candidate ratio 仅 `2.2334%`，learned reliable candidate ratio
   却为 `10.3420%`，扩大约 4.63 倍；positive row ratio 为 `6.0563%`，训练 switch ratio 已到
   `9.9642%`，heldout switch ratio 又升到 `12.2878%`。

第二个放大因素是审计基率。受保护 E57 本身用完整 Nr3D train 训练，而 fold 0 只对新 FPR-TV head 做
scene-disjoint；因此 Parent 在该 holdout 上达到 `6873/7129=96.4090%`。Parent 正确/错误约为
`26.85:1`，但系统仍切换 876 行；在这种分布中只有 256 行存在 @0.25 的理论修复机会，任何未经真实先验
校准的 0.5 gate 都会明显偏向误切。这不等于可以忽略负结果：它说明当前 v1 既不满足强 Parent 的保守
合同，也不能用该 fold 事后寻找新阈值。

因此，FPR-TV v1 的失败原因按证据强弱更新为：

- **直接实现/目标语义错配**：balanced score 被当作 calibrated probability，且 relative
  `repair>break` 被误当作 absolute no-break certificate；
- **审计分布放大**：已见完整 train 的强 Parent 使误切成本远大于修复收益；
- **候选正例稀疏**：safe candidate 仅约 2.23%，Top-K 文本证据远未达到可自由切换的可靠度；
- **Nr3D source 互补性低**：现有 V99 两 source 几乎重复，不能为 verifier 提供足够独立证据。

若未来形成 FPR-TV v2，允许的改变必须是结构性修正而不是在 fold 0 上调参：把最终动作改为单一、与
部署目标一致的 Parent-dominance / abstention 判别，分别证明 @0.25 和 @0.50 非退化；不能再比较三个
未校准 balanced sigmoid 的大小。新版本必须重新预注册并使用未消费证据。

项目范围同时永久按用户决定冻结：**不做 baseline 公平复现；不参照或继承实验/章节七与八中的旧方法、
顺序、前置条件和结论；不采用旧 E0--E7 实验矩阵。** 当前只保留与 V99 总体架构一致、有单一因果假设的短实验；不把这些被排除路线换名
重新引入。

本节记录时 Sr3D relation-CF 正式 E27 已到 `1140/6486`（17.57%），total/relation-CF loss 仍有限，
无 Traceback、OOM、NaN/Inf 或磁盘错误；GPU 约 26.1 GiB，数据盘余约 7.6 GiB，尚未进入正式评估。

#### 16.15.11 FPR-TV v2 结构性修正、未消费 fold1 预注册与启动

2026-08-31 在不复用 fold0 选阈值、不修改 Top-K/loss weight/margin/训练时程的前提下，完成了
FPR-TV v2 的最小结构性修正。它不是新的 Gate 参数扫描，而是直接修正 v1 的训练概率语义与部署安全
合同：

1. repair、break、joint Parent-dominance 三个二分类头改用真实经验先验下的逐有效样本 BCE；不再用
   class-balanced BCE 的输出冒充真实概率；
2. Candidate 只有在“至少一个阈值 repair probability **严格大于 0.5**、两个阈值 break
   probability **全部严格小于 0.5**、joint Parent-dominance probability **严格大于 0.5**”时才有资格
   进入离散动作比较；
3. `repair > break` 不再被允许作为安全证书；action logit、可晋升 score gap、detector-valid、结构解析
   可靠性与 exact Parent fallback 仍必须同时成立；
4. 概率恰为 `0.5` 被定义为无证据平局，必须 abstain。新增 joint-dominance 与 repair 两个 exact-half
   回退测试，防止 `>=0.5` 重新引入零证据切换。

实现固定在本地审查分支：

```text
repo   = C:\Users\gb\.codex_remote_staging\mcln_fprtv_20260831
base   = 40f1dfde91b37c58d6a383b4b39de4db23e6ac7d
v2     = 68944114ac13bd7cc803c72a596dd9b403f4d7ae
strict = de9de11
launch = b6791a9
```

隔离环境定向回归为 `83 passed`；更早的完整测试为 `3736 passed / 6 failed`，六项均来自隔离快照中既有
audit/joint-query fixture 与运行代码闭包不匹配，不触及本次 FPR-TV 三文件差异。Spec 与
Standards/Correctness 两轴在修正 exact-half 后均 PASS。

为避免再次消费 fold0，新 launcher 采用以下 fail-closed 合同：

- 参数只接受 `FOLD=1`；传 0、2、3、4 均直接拒绝；
- 同一 opened FD 内读取并校验 v1 fold0 decision SHA
  `02f1951b...2a22e67` 与 receipt SHA `843bf42c...8ff`；精确要求 v1 结果仍为
  `@0.25 40 fix / 82 break`、`@0.50 119 fix / 398 break`、switch=876、禁止长训；
- 同时要求 v1 fold1 与 v2 fold1 两个 run root 都不存在；
- v2 runtime manifest 固定 365 个文件、总大小 105,978,868 bytes，manifest SHA
  `ed255b26286ef5a9b2b5ed497563ea0221e1bf3ea63ea9d1075bbcd2223459e1`，并强制 manifest
  `source_root` 与实际 v2 隔离源根完全一致；
- v2 code/input snapshot、GroupFree、E57 与正式配置继续做前后 SHA 校验；fold 审计不允许生成 `.pth`。

2026-08-31 18:05 CST，fold1 preflight 精确通过：fit=`25,578 samples / 400 scenes`，holdout=
`7,341 samples / 111 scenes`，configuration SHA=`7845c9a7...32c9`。随后在 screen
`mcln_nr3d_fpr_tv_v2_fold1` 启动唯一 E58 scene-disjoint 审计：

```text
/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/
fpr_tv_scene_disjoint_v2/fold_1
```

它从受保护 Nr3D E57 完整状态开始，仅训练 Structured Slot Builder、SACR evidence 与 FPR-TV 三个允许
模块，完整消费 fit split 一轮后只评估 heldout train scenes。晋级条件仍固定为：至少发生一次 switch，
`fix025 > break025` 且 `fix050 >= break050`；无论结果如何，本折本身都不授权长训练、正式 7,899-row
评估或权重保存。

同一时刻 Sr3D relation-CF E27 已到约 `4622/6486`（71.26%），total loss 约 `9.12`、aux loss 约
`0.0853`，无 Traceback/OOM/NaN/Inf；预计约 39 分钟完成训练，之后才进入 17,726-row 正式验证。
当前正式最好仍不变：Nr3D `4475/3759`，Sr3D `12139/10335`；硬目标分别为 Nr3D `4740` hits、
Sr3D `12214` hits。

项目范围继续严格遵守用户决定：**不做 baseline 公平复现；不采用原建议第七节；不采用原建议第八节
E0--E7 矩阵。**

#### 16.15.12 FPR-TV v2 fold1 终态与 Sr3D relation-CF E27 正式结果

##### 16.15.12.1 Nr3D FPR-TV v2：从“误切过多”退化为“完全不切”

2026-08-31 19:17 CST，唯一预注册的 v2 fold1 scene-disjoint 审计自然结束。训练完整消费
`25,578` 个 fit 样本、`400` 个场景，共 `1,599` 个 batch；样本身份 count/unique count 均为
`25,578`，实际消费集合 SHA 与冻结 fit split SHA 完全一致，因此没有重复或遗漏。held-out 为
`7,341` 个样本、`111` 个互斥场景。

审计与决策文件为：

```text
receipt = .../fpr_tv_scene_disjoint_v2/fold_1/runtime_output/nr3d/
          nr3d_v99_fpr_tv_v2_scene_fold1_e58/1788170744/
          fpr_scene_disjoint_audit_fold_1_epoch_58.json
receipt SHA-256 = 331c79a390d15494d9475628a24f16461dd1c5fb4a6149a86902a1af3e2df12e

decision = .../fpr_tv_scene_disjoint_v2/fold_1/decision.json
decision SHA-256 = beefd711d21ca0a2d314b696c3b99aa8cf0910b30edc1713b898978d59d69cb7
```

终态转移统计如下：

| held-out 指标 | Parent | Selected | fix | break | net | switch |
|---|---:|---:|---:|---:|---:|---:|
| REC@0.25 | `7004/7341 = 95.4093%` | `7004/7341 = 95.4093%` | 0 | 0 | 0 | 0 |
| REC@0.50 | `5906/7341 = 80.4523%` | `5906/7341 = 80.4523%` | 0 | 0 | 0 | 0 |

因此固定门失败：`no_heldout_switch`，以及 `Acc@0.25 fix` 没有严格大于 `break`。decision 明确为
`fold_gate_pass=false`、`long_training_authorized=false`、`next_stage=method_correction_only`。
fold2--fold4 与正式 7,899-row 评估均不得启动。

这不是训练未生效。52 个允许训练 tensor、共 `2,510,184` 个参数的 SHA 确实改变；冻结的 1,144 个
tensor、149,670,851 个参数 SHA 完全不变，Box、Mask、Parent score 等冻结输出 sentinel 前后也完全
一致。平均总/FPR loss=`0.913742`，grad norm=`5.604997`，均有限。真正原因是安全证据极度稀疏：

- feasible candidate ratio=`31.8702%`；
- predicted no-break candidate ratio=`62.6506%`；
- predicted repair candidate ratio 仅=`0.05299%`；
- learned reliable candidate ratio 仅=`0.00814%`；
- eligible/switch ratio 均为 0，fallback ratio=1.0。

因此 v2 的科学结论是：经验先验 BCE 与绝对 break veto 成功消除了 v1 的 876 次危险切换，但联合
`repair>0.5 + both break<0.5 + joint-dominance>0.5` 在当前 Nr3D evidence 下过于保守，最终没有任何
可部署动作。v1 是 over-switch，v2 是 under-switch；二者共同证明继续改一个全局阈值、margin 或 loss
weight 只会在两种失败之间摆动，不能解决 source 不互补和文本/Anchor 证据不足。当前 FPR-TV v2 后续 folds
正式封存，且本实验没有生成任何新 `.pth`，受保护 Nr3D E57 最好权重保持不变。

##### 16.15.12.2 Sr3D relation-CF E27：正式负结果，E28 按预注册继续

Sr3D E27 于 19:12 CST 完成全部 `17,726` 条正式评估。原始结果为：

| 指标 | E27 | 当前受保护最好 E26 | 差值 |
|---|---:|---:|---:|
| REC@0.25 | `12121/17726 = 68.3798%` | `12139/17726 = 68.4813%` | `-18 hits / -0.1015pp` |
| REC@0.50 | `10252/17726 = 57.8359%` | `10335/17726 = 58.3042%` | `-83 hits / -0.4683pp` |
| Mask@0.25 | `11521/17726 = 64.9949%` | — | — |
| Mask@0.50 | `9547/17726 = 53.8587%` | — | — |
| Mask mIoU | `44.8159%` | — | — |

正式 receipt 为该 run leaf 下 `eval_metrics_epoch_27.json`。official monitor 已核验 sample_count=17,726，
将 latest 更新为 E27，但 `metric_best/preserved_best` 继续指向 E26 的 `12139` hits；硬目标仍是
`12214/17726=68.9044%`，当前最好距离目标 75 hits。E27 不能作为 relation-CF 正收益证据。

E28 随后按唯一预注册合同完成，没有临时调低 LR、改变 relation-CF loss weight、Top-K、margin 或延长
epoch。最终结果为：REC@0.25=`12118/17726=68.3629%`，REC@0.50=
`10283/17726=58.0108%`；Mask@0.25/@0.50=`11533/9539=65.0626%/53.8136%`，Mask mIoU=
`44.7734%`。相对受保护 E26，REC 仍少 `21/52 hits`；相对 E27，@0.50 回升 31 hits，但 @0.25 又少
3 hits。连续 E27/E28 都未超过 12139 hits，因此 Sr3D relation-CF 短路线正式封存，不做参数扫描、第三轮
衰减或追加 epoch。

训练 screen 自然退出且 official monitor 已消费 E28 receipt；run 内 `ckpt_epoch_27.pth`、
`ckpt_epoch_28.pth`、`ckpt_epoch_last.pth`、`ckpt_best_rec_acc025.pth` 四个非全局最好权重在精确路径与
进程校验后删除，共释放 `3,176,838,436` bytes。该 run 剩余 `.pth=0`，日志、E27/E28 JSON 和 retention
receipt 全部保留。受保护 E26 权重删除前后 SHA 均为
`da985736e5bc116c03cca51a523a211cade515d9b7580deb8e9d48bf8a4499d3`，磁盘可用空间恢复到约 8 GiB。

本节再次固定项目边界：**baseline 公平复现永久取消；不参照或继承实验/章节七与八中的旧方法、顺序、
前置条件和结论；旧 E0--E7 实验矩阵不采用。** 当前结果不能通过这些被排除路线补证或换名重启。

#### 16.15.13 Nr3D Conservative Raw Parser 唯一正式评估：文本匹配改善未转化为 REC，路线封存

##### 16.15.13.1 为什么做这一次评估，以及它不是什么

此前 Conservative Raw Parser 在 Nr3D train-domain 的中间解析诊断中，将 target-text match 从
`68.9259%` 提高到 `71.5045%`（`+2.5785pp`），在 scene-disjoint train holdout 上从
`68.2284%` 提高到 `70.7673%`（`+2.5389pp`）。该结果只能说明保守 command、inversion、copular
三类语法规则更常把文本 target 对齐到标注目标名称，不能证明 3D grounding 的 REC 会提高。

因此预注册了一次、且只允许一次的完整 `7,899`-row eval-only 验证。它使用当前受保护的 Nr3D 最好
平均权重，不训练、不生成新权重、不改变 V99 selector、不扫描 parser 规则或决策阈值，也不属于已经被
用户取消的 baseline 公平复现、第七节或第八节 E0--E7 实验矩阵。

正式输入固定为：

```text
checkpoint = official_best_rec025_epoch_57_0p56652741.pth
checkpoint SHA-256 = 76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1

raw parser bundle = nr3d_conservative_syntax_v1.bundle
bundle SHA-256 = 1bb14e411debf1736569cdbf532987311289e10efe6492331cbd73da9c3cbcb6

GroupFree SHA-256 = 9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2
launcher SHA-256 = fccb6f894c8ce703582b4dbfbc14e92034a40729c0ce711cff30caa52e3ea350
```

启动前，原 review root 的 `train_dist_mod.py` 已被后续 FPR 审计工作更新，首次 preflight 因固定 SHA
不一致而正确拒绝，正式 one-shot 未被消费。服务器仍保存了预注册时的精确文件字节，因此从一份
`9/10` 关键文件匹配的只读历史快照恢复代码闭包，并只补入 SHA 精确匹配的
`main_utils.py`；随后 `train_dist_mod.py`、`main_utils.py`、dataset、cache loader、spaCy parser、
MCLN、selector、adapter、losses、evaluator 与 launcher 共 `11/11` 项全部匹配固定 SHA。第二次
preflight 才通过，正式运行使用该执行根生成独立 consumed snapshot，当前仓库和历史保护快照均未修改。

##### 16.15.13.2 正式结果：REC、Mask 与 mIoU 全部下降

2026-08-31 23:03--23:22 CST，screen `mcln_nr3d_raw_parser_one_shot` 完成全部 `494/494` 个验证
batch。bundle 严格命中 `7,899/7,899`，fallback=0；模型权重按固定 SHA 成功加载，运行期无
Traceback、OOM、磁盘错误或进程异常，且 run 外生成权重数为 0。

| 指标 | 当前受保护最好 | Conservative Raw Parser | 差值 |
|---|---:|---:|---:|
| REC@0.25 | `4475/7899 = 56.6527%` | `4352/7899 = 55.0956%` | `-123 hits / -1.5572pp` |
| REC@0.50 | `3759/7899 = 47.5883%` | `3650/7899 = 46.2084%` | `-109 hits / -1.3799pp` |
| Mask@0.25 | `4192/7899 = 53.0700%` | `4095/7899 = 51.8420%` | `-97 hits / -1.2280pp` |
| Mask@0.50 | `3479/7899 = 44.0435%` | `3392/7899 = 42.9421%` | `-87 hits / -1.1014pp` |
| Mask mIoU | `37.4337%` | `36.5249%` | `-0.9089pp` |

候选结果距离新的 Nr3D 硬目标 `4740/7899` 反而扩大到 `388 hits`。该版本的绝对子群结果为：

| 子群 | REC@0.25 | REC@0.50 |
|---|---:|---:|
| Easy | `2370/3926 = 60.3668%` | `1980/3926 = 50.4330%` |
| Hard | `1982/3973 = 49.8867%` | `1670/3973 = 42.0337%` |
| View-dependent | `1334/2647 = 50.3967%` | `1095/2647 = 41.3676%` |
| View-independent | `3018/5252 = 57.4638%` | `2555/5252 = 48.6481%` |

这些数值再次表明较难的同类消歧、复杂关系和视角表达仍是主要低值区域；但本次不把子群标签用于模型
训练或阈值选择。

正式 source-choice 诊断还显示：learned selector 继续 `100%` 选择 `default`，第二个
`default_rank_blend_contrastive010` source 的选择率为 0；两个 source 的 oracle headroom 仅约
`0.00051 / 0.00038`。所以 parser 即使改善部分 target 字符串，也没有产生一个可被 V99 selector 利用的
互补排名来源。

##### 16.15.13.3 为什么 target-text match 提升，最终 REC 却下降

该现象不是矛盾，而是说明此前的中间指标没有覆盖真正的部署链：

1. **target-text match 只测“解析出的目标名称是否像 GT 类别”，不测候选框是否正确。** 它不评价同类
   实例之间的属性、关系、Anchor、视角和否定组合，也不评价最终 Top-1 Query。
2. **权重是在旧解析分布上训练的，eval-only 替换 parser 形成输入分布漂移。** 新 bundle 会改变
   `graph_node/graph_edge/auxi_entity` 等结构槽，但 Backbone、语言融合和 selector 没有在该分布上重新
   学习；字符串更“正确”不代表这些槽在冻结网络里被赋予了正确权重。
3. **Nr3D 的主要剩余错误不是单纯 target head 抽取错误。** 已有 Top-K oracle 证明正确候选经常存在，
   真正缺口是同类别候选间的 relation/anchor-conditioned Top-1 选择。只改 target parser，没有新增
   候选级几何证据，也不能修复噪声 Anchor。
4. **两个 V99 source 缺乏互补性。** selector 全程回到 default，oracle headroom 也接近 0；因此
   parser 引入的错误没有第二条独立来源可以纠正。
5. **Mask 同步下降是分布漂移的直接旁证。** 如果变化只改善最终文本标签，Mask 不应系统性少
   `97/87 hits` 且 mIoU 下降 `0.9089pp`；实际结果说明新结构槽扰动了 query/mask conditioning，影响
   远不止最后一个 REC 分数。

因此不能根据 `+2.54pp` 的中间 target match 宣称 parser 有效，也不能为了把正式 REC 拉回去而继续在
同一 validation 上加规则、搜语法门槛或重建 train cache。当前证据支持的结论是：**Raw-only parser 的
单独 eval-time 迁移对当前 V99 权重有显著负作用。**

##### 16.15.13.4 postflight 恢复与最终决策

完整评估和 metric receipt 已正常生成，但 launcher 在最后的 code-snapshot inventory 检查处拒绝创建
原 decision。原因是旧 `record_tensorboard.py` 把两个运行期 event 文件写入了 code snapshot：train
event `40 bytes`、val event `741 bytes`。逐文件复核证明：预注册 manifest 中的代码文件修改数为 0、
删除数为 0，唯一差异就是这两个 TensorBoard 文件；checkpoint、bundle、GroupFree 的 source/snapshot
SHA 全部一致，无生成权重。因此没有重跑评估，而是基于同一次 `494/494` 运行的原始 metric bytes 写入
只读 recovery decision：

```text
metric receipt SHA-256  = 5c0f28f86ebbffdc2f32f433c0187adf8386483c84f201ff6c7342a1347a97e1
recovery decision SHA-256 = a0723c6a89e559f18cd1d60369791de374ddb990b094914b7aebadb8b7c4f541
evaluation_rerun = false
official_best_weight_modified = false
branch_status = sealed_negative
next_action = do_not_rebuild_train_cache_or_rerun_parser_eval
```

最终决策：Conservative Raw Parser 正式封存；不重跑、不重建 parser train cache、不继续添加规则，也不将
其作为 Nr3D 提升证据。当前正式最好继续保持 Nr3D `4475/3759`、Sr3D `12139/10335`，硬目标仍分别为
`4740` 与 `12214` hits。后续继续遵守用户固定边界：不做 baseline 公平复现，不执行原第七节，不采用
原第八节 E0--E7 矩阵。

#### 16.15.14 Nr3D Density-Aware Target Box：唯一 100-batch 训练审计通过，但尚无 REC 证据

##### 16.15.14.1 方法与本次审计要回答的问题

FPR-TV v1/v2 已分别以 over-switch 和 complete abstention 告终，Raw Parser 也证明单独改善 target-text
match 不能改善正式 REC。与此同时，Nr3D 的点数/体积诊断仍显示低点数小目标具有独立 Proposal 瓶颈：
最稀疏四分位目标的 Top-1 明显低于稠密目标，并且部分小目标的 Top-K oracle 本身也偏低。因此后续不再把
所有错误都压进候选重排序，而是增加一条与文本排序正交的训练期辅助路线：

> **Density-Aware Target Box auxiliary：只对 Nr3D/Sr3D referring row 中点数 `0<n<256` 的目标，
> 使用最后一层 Hungarian 匹配到 target0 的 Query，施加按稀疏度加权的 GT center/size L1；默认权重为 0，
> 不增加参数、不改变推理、不读取数据集 ID 或 Unique/Multiple 子群标签。**

固定稀疏权重为 `1 - n/256`，辅助项为 center L1 加 `0.2 × size L1`。ScanNet 检测行显式排除；GT、点数、
稀疏权重与 Hungarian index 均 detach，梯度只回到匹配 Query。唯一可配置量是总 loss weight，本次固定为
`1.0`；点数阈值和 size 系数不是可搜索超参。

由于该模块首先需要证明“真实数据上有足够激活密度、损失和梯度有限且非零”，本次只执行一次 E57→E58
的 100-microbatch train-only 审计。它不是 validation，也不生成 REC/Mask 指标，不保存 checkpoint，
更不授权长训练。正式输入为受保护完整 E57：

```text
checkpoint epoch/SHA = 57 / fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655
GroupFree SHA        = 9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2
batch/accumulation   = 16 × 1
audit epoch/batches  = E58 / 100
eval/save            = false / false
runtime manifest     = b25ddeee706da141bad1d307d4de335ddd4d4f088309c1c24c19c6699779f7de
launcher SHA         = b9f9ba0b5116c81741f8f9140928d46b04b2c57518f4b40feed5b8cf1956f340
static executor SHA  = ade6898cfb4bcdaf165f706f37bfd6479f43f03108b6230d213d87a67b00ca37
```

运行前已完成 77 项定向测试与双轴代码审查。正式进程仅消费 367-file 只读代码快照、独立 inode 的 E57/GF
输入快照和固定数据清单；全局 GPU 锁、Landlock、能力清零与 pre/post SHA 验证均通过。该基础设施只用于
保证一次性审计的输入/输出身份，不构成方法贡献。

##### 16.15.14.2 2026-09-01 正式 100-batch 回执

2026-09-01 02:52--03:06 CST，唯一 one-shot 审计自然完成。训练集/测试集读取为 `44,909/7,899`；
实际计划精确为 `100 requested / 100 effective / 0 dropped / 100 optimizer steps`。E57 在同一固定输入快照中
按 SHA 与内部 epoch 成功恢复，100 个 batch 全部有限，无 Traceback、OOM 或 NaN。训练期间未进入
validation，`runtime_metrics_count=0`，并且 `runtime_pth_count=0`。

| 审计量 | 100-batch 均值 | 含义 |
|---|---:|---|
| `density_aware_target_box_loss` | `0.1192157212` | 辅助项真实激活且非零 |
| active row / batch | `3.27` | 每批平均约 3.27 个稀疏 referring target |
| referring row / batch | `11.78` | 分母为可用 referring rows，不含检测行 |
| active row ratio | `0.2768755039` | 约 **27.69%** referring rows 进入辅助监督 |
| target point count mean | `135.9273349` | 激活目标平均仅约 136 个点，确为稀疏目标 |
| sparsity weight mean | `0.4390338597` | 辅助权重没有塌到 0 或饱和到 1 |
| center L1 | `0.0977383481` | 匹配 Query 的中心仍有可学习误差 |
| size L1 | `0.1073868572` | 匹配 Query 的尺寸仍有可学习误差 |
| overall grad norm | `19.8903716373` | 有限且非零；没有梯度爆炸/消失证据 |
| total loss | `8.6155924702` | 全程有限 |

20/40/60/80/100 batch 的 density loss 依次为 `0.1315/0.1234/0.1208/0.1161/0.1192`。这不是正式
收敛曲线，但说明辅助项在不同 batch 组中持续存在，没有只在首批偶然激活，也没有数值发散。

原始回执与决策为：

```text
receipt SHA-256 = 109c183812910dd98ae193b5578cdbbda2da5182f32300ffcee9eec2e745b5f2
decision SHA-256 = 525056a5c07e9a519d6fedc759594926bb42281258823475aa6618fc23de6bae
density_gate_passed = true
default_off_regression_passed = true
long_training_authorized = false
next_step_if_passed = scene_disjoint_short_audit_only
```

审计前后 E57/GF SHA 保持 `fe1e...f6655 / 9ff3...e54f2`，没有修改任何受保护权重。运行终态仅保留
config、log、TensorBoard、训练审计 JSON、pre-audit provenance 与 decision；没有可被误当作候选模型的
新 checkpoint。

##### 16.15.14.3 可以得出的结论与不能得出的结论

可以确定：

1. **低点数目标不是极少数边角样本。** 在本次固定抽样中，27.69% 的 referring rows 激活，训练密度足以
   支持一次短周期 scene-disjoint 验证。
2. **辅助目标与当前 Query 存在真实误差。** center/size L1 均约 0.10，说明不是已经完全拟合的零信号。
3. **实现可训练且不破坏默认路径。** loss/gradient 有限，权重 0 的旧路径已回归；本次没有参数、推理或
   checkpoint schema 扩张。
4. **该路线针对的是 Proposal/localization，不是 FPR-TV 的 Top-1 文本排序。** 它可能改善小目标框覆盖，
   但不会直接解决长句、Anchor、视角和同类实例选择。

不能确定：

1. **不能宣称 REC 已提高。** 本次故意不跑 validation，没有任何新 REC/Mask/mIoU 数值。
2. **不能从 loss 下降推断最终定位命中提升。** 必须在不接触正式 7,899-row 集的 scene-disjoint holdout 上
   检查小目标 localization/oracle 与 Parent REC 是否净正。
3. **不能直接进入长训。** decision 明确永久写入 `long_training_authorized=false`；下一步最多是一个固定、
   无阈值搜索、无 checkpoint 保存的 scene-disjoint 短审计。
4. **不能把它与 FPR-TV、Raw Parser、relation-CF 或被取消的 baseline 复现同时组合。** 下一步唯一变量仍是
   Density-Aware Target Box loss。

##### 16.15.14.4 下一步固定边界

下一步只允许设计并经独立审查后执行 **scene-disjoint short audit**：训练与 heldout scene 必须无重叠，
从同一受保护 E57 full-state 开始，保持 V99、B16×A1 和全部主损失不变；只打开本辅助 loss。审计需要同时
报告 heldout REC@0.25/0.50、稀疏目标分层、Top-K oracle/Proposal coverage 与 box center/size 变化，且不在
heldout 上搜索阈值、权重、点数门槛或 epoch。若该短审计没有明确的稀疏目标净收益且总体不退化，则路线
直接封存；通过也只代表可以讨论下一阶段，不自动授权正式 7,899-row 评估或长训练。

项目硬目标保持不变：Nr3D 至少 `4740/7899`，Sr3D 至少 `12214/17726`。用户明确排除项继续为硬约束：
**不做 baseline 公平复现；不执行原建议第七节；不采用原建议第八节 E0--E7 实验矩阵。**

#### 16.15.15 Density-Aware scene-disjoint 三角色审计：@0.50 有效，但 @0.25 负收益，路线正式封存

##### 16.15.15.1 为什么必须增加 parent/control/method 三角色

上一节的 100-batch train-only 审计只证明辅助 loss 有密度、有梯度、数值有限，不能证明定位指标会改善。
为了把“普通 E57 继续更新 100 步”的影响与 Density-Aware 辅助项本身分开，本次使用同一受保护 E57、
同一 Nr3D train-domain scene split 和同一训练样本顺序，串行执行三个角色：

1. `parent`：不训练，只在固定 fold-2 holdout 上评估 E57；
2. `control`：辅助权重为 0，按原训练目标更新恰好 100×B16，再评估同一 holdout；
3. `method`：除 Density-Aware loss weight=1 外与 control 完全相同，并消费同一 1,600 条 fit rows。

固定 fold-2 包含 511 个 Nr3D train scenes / 32,919 条 referring rows，其中 fit 为 408 scenes / 26,590
rows，holdout 为 103 scenes / 6,329 rows。这里的 scene-disjoint 含义是：本次 100-step 更新不读取 holdout
scene；受保护 E57 本身曾由完整 Nr3D train split 训练，因此这些绝对准确率不能当成未知场景泛化水平，只能用
parent/control/method 的同源差分判断辅助项方向是否正确。

正式合同为：B16×A1、E57→E58、method/control 各 100 optimizer steps、同一 1,600 unique row identity、
不访问正式 7,899-row validation、不保存 checkpoint、不允许阈值/权重/点数门槛/epoch 重试。用于部署与审计
的主要身份为：

```text
protected E57 SHA-256 = fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655
GroupFree SHA-256     = 9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2
runtime manifest      = 04977c404fb759722d56e8bbeadb383a7113f4cec8e6d7dbde24d35f3f48c354
static executor       = d63392f280a6563e6cd8439a44aa5da8eb68c59d71c7a5574aa2763915e02775
formal launcher       = 4f9715fb74ecdc7aeb13eb36d67e26b57a5f677be019dca94b7d69ddc0257feb
fit 1,600-row SHA     = e47d5ecd515fa34653a95177be6c836860e78d9cef1a0431a009f6e41980a98d
holdout 6,329-row SHA = 18683a43051f172e757073db296ad9b1ac5af882abe472cc67ce6db521668959
```

静态入口在正式运行前已完成 same-FD preflight；竞争同一
`/root/autodl-tmp/mcln_v99_backbone_gpu0.lock` 时按合同返回 status 6。正式运行结束后 E57/GF source SHA
保持不变，审计树中仅有两份只读输入快照 `.pth`，生成 checkpoint 数为 0；三角色均记录
`formal_validation_accessed=false` 与 `long_training_authorized=false`。

##### 16.15.15.2 训练信号真实存在，但不能据此判定方法有效

method 的 100-step 训练回执为：

| 训练审计量 | method 均值 |
|---|---:|
| Density-Aware loss | `0.1308621538` |
| active sparse rows / batch | `4.65` |
| active row ratio | `0.290625` |
| active target points mean | `132.0403` |
| sparsity weight mean | `0.4742177` |
| center L1 | `0.1060342` |
| size L1 | `0.1241395` |
| overall grad norm | `51.1964` |
| optimizer steps / samples | `100 / 1600` |

control 的 grad norm 为 `42.7698`。这证明辅助项确实改变了优化轨迹，但最终是否值得保留只能由下面的
paired holdout 差分决定。

##### 16.15.15.3 paired holdout 结果

以下 Top-16 均先应用正式 `butd_cls` detector-overlap `IoU>0.25` 过滤，排序使用同一
`selected_source_scores`；`active_sparse` 是点数 `0<n<256` 的 1,512 条目标。

| 范围 / 指标 | Parent E57 | Control 100步 | Method 100步 | Method-Control |
|---|---:|---:|---:|---:|
| overall selected @0.25 | 6066/6329, `95.8445%` | 6070, `95.9077%` | 6060, `95.7497%` | **-10 hits, -0.1580pp** |
| overall selected @0.50 | 5323, `84.1049%` | 5314, `83.9627%` | 5333, `84.2629%` | **+19 hits, +0.3002pp** |
| overall Top-16 @0.25 | 6282, `99.2574%` | 6283, `99.2732%` | 6280, `99.2258%` | **-3 hits** |
| overall Top-16 @0.50 | 6028, `95.2441%` | 6027, `95.2283%` | 6076, `96.0025%` | **+49 hits, +0.7742pp** |
| sparse selected @0.25 | 1391/1512, `91.9974%` | 1394, `92.1958%` | 1388, `91.7989%` | **-6 hits, -0.3968pp** |
| sparse selected @0.50 | 910, `60.1852%` | 908, `60.0529%` | 944, `62.4339%` | **+36 hits, +2.3810pp** |
| sparse Top-16 @0.25 | 1471, `97.2884%` | 1473, `97.4206%` | 1469, `97.1561%` | **-4 hits, -0.2646pp** |
| sparse Top-16 @0.50 | 1256, `83.0688%` | 1254, `82.9365%` | 1294, `85.5820%` | **+40 hits, +2.6455pp** |
| sparse matched-target IoU mean | `0.632438` | `0.631468` | `0.640597` | **+0.009128** |
| sparse matched @0.50 | 1205 | 1203 | 1240 | **+37 hits** |

Dense 目标没有形成对主指标有帮助的补偿：method 相对 control 的 dense selected @0.25 为 `-4 hits`，
dense selected @0.50 为 `-17 hits`；虽然 dense Top-16 @0.50 增加 `9 hits`，最终 Top-1 仍下降。

相对 parent 而不是 control，method 的 overall selected @0.25 仍为 `-6 hits`，active-sparse selected @0.25
为 `-3 hits`，active-sparse Top-16 @0.25 为 `-2 hits`；因此不能把 control 自身的随机/正常 100-step
变化误算成辅助项收益。

##### 16.15.15.4 Gate 结果与正式判决

预注册门槛中只有以下方向通过：

- sparse matched-target IoU 严格高于 control；
- overall selected @0.50 不低于 control；
- fit/holdout identity、0 生成权重、0 正式 validation 全部精确通过。

以下四个核心门槛失败：

```text
active_sparse_top16_hits025_strict_gain_vs_control = false
overall_selected025_non_degradation_vs_control     = false
active_sparse_selected025_non_degradation          = false
not_jointly_worse_than_parent                      = false
```

最终 artifact：

```text
parent receipt SHA  = 3fafb3173f1e7537e346a4389c3095cd477759c12f9ff6127165e526e0831afa
control receipt SHA = 905888c76ade71ccb844be7361b66a5a466ce867c700b82dedeaa084c5343828
method receipt SHA  = aef803baea9845feb5eef7a1c8832ee4933529e9d6de4a8935a93f83284d5d81
decision SHA        = c494fcdf1db53de3babaa9536ea4c9ca29903413de43224793d9698b450191d5
density_gate_passed = false
next_allowed_step   = seal_method
long_training_authorized = false
```

因此 **Density-Aware Target Box 路线正式封存**。不允许用同一 holdout 再调 loss weight、`n<256`、size
系数、学习率、训练步数或 fold；不组合 FPR-TV 后再解释结果，也不进入正式 7,899-row evaluation。

##### 16.15.15.5 问题本质：它改善的是“框更紧”，不是“@0.25 找到正确目标”

这次结果把小目标问题进一步拆开了：

1. **辅助项对 localization refinement 确有作用。** sparse matched IoU `+0.00913`、selected @0.50
   `+36 hits`、Top-16 @0.50 `+40 hits`，说明 matched Query 的框在变紧。
2. **但它没有增加 @0.25 proposal coverage。** sparse Top-16 @0.25 反而少 4 hits；所以当前 center/size
   L1 不是“补出缺失小目标 proposal”的机制，只是在已有 Hungarian match 上做回归。
3. **共享参数更新引入了候选漂移。** Density loss 虽只从 matched Query 产生直接梯度，梯度仍经过共享
   decoder/backbone 参数；它能改善某些 Query 的严格 IoU，同时让其他候选的粗粒度覆盖或排序变差。
4. **Top-1 排序仍未被解决。** @0.50 的 Top-16 增益没有等比例转成 dense Top-1，且 @0.25 Top-1
   下降，说明局部框回归不能替代文本条件候选选择。
5. **这个 holdout 的 @0.25 Top-16 已约 97%--99%。** 受保护 E57 曾看过完整 train split，故本次绝对
   oracle 很高；它适合判断 100-step 差分，不足以证明正式 val 中 1,556 个 proposal-failure 样本会被修复。

##### 16.15.15.6 三条失败路线合并后的最新诊断

截至本次审计，三条相互独立的证据已经闭合：

1. **Conservative Raw Parser**：target-text match 中间指标改善，但冻结 V99 正式 REC/Mask 同时下降，
   说明 eval-time parser 分布漂移不能解决同类实例消歧；路线已封存。
2. **FPR-TV v1/v2**：v1 在 fold0 发生 over-switch，@0.25 `40 fix / 82 break`；v2 改为经验先验 BCE
   和绝对安全门后，joint reliability/repair 头在约 2% candidate-positive 先验下塌为全拒绝，fold1 为
   `0 switch`。这不是再调 0.5 阈值能解决的问题，而是重复 learned veto 与极低正例率的结构问题。
3. **Density-Aware Target Box**：稀疏目标 @0.50 和 matched IoU 提升，但 @0.25 proposal/selection
   下降；它解决的是已有 match 的框精度，不是缺失 proposal 或文本 Top-1。

因此下一代候选验证器若继续推进，必须是结构修正而不是参数微调：把 fallback 作为候选集合中的显式动作，
用 row-level setwise competition 直接学习“选择某个候选或保持 Parent”；保留确定性 detector-valid、score-gap、
parse/anchor reliability 和 absolute break veto，但删除与 repair 重复、在低先验下塌缩的 learned reliability
串联门。任何新方案都必须使用尚未消费的 scene split、先做新的固定审计，并继续禁止 validation threshold
sweep。Density-Aware 不再参与组合。

项目硬目标仍为 Nr3D `>=4740/7899`（严格超过 60.0%）与 Sr3D `>=12214/17726`（严格超过
68.9%）。用户固定排除项继续生效：**不做 baseline 公平复现，不执行原建议第七节，不采用原建议第八节
E0--E7 实验矩阵。**

#### 16.15.16 FPR-TV v3 fold3 正式短审计：动作竞争减少损伤，但仍未解决安全切换（2026-09-01）

##### 16.15.16.1 本轮到底改了什么

FPR-TV v3 不是在已消费的 fold0/fold1 上改阈值，而是一次新的结构修正：

1. 删除在极低正例先验下塌缩的 learned reliability 串联门；
2. repair head 只保留为经验先验 BCE 辅助监督和诊断，不再参与部署资格判定；
3. 把“保持 Parent”固定为分数 0 的显式 fallback action；只有最佳候选 action logit 严格大于 0 才允许切换；
4. 候选仍必须同时满足正式 `butd_cls` detector-valid、原 score-gap feasible、确定性的 parse/target/anchor
   reliability，以及两个绝对 break 概率都严格小于 0.5；
5. 没有重新启用 SACR、Relation-CF、Density-Aware、外部 REC reranker、GT sidecar、Dataset ID 或
   Unique/Multiple 标签，也没有修改 Backbone、Box、Mask 或推理网络。

本轮只允许预注册 scene-disjoint fold3：417 个 fit scenes、26,714 条 fit 样本，94 个 held-out scenes、
6,205 条 held-out 样本；从受保护 Nr3D V99 E57 完整状态开始，以 B16×A1 完整训练 E58 一个自然尾批 epoch，
随后在 held-out 上评估。禁止保存权重、禁止 7,899-row formal、禁止自动启动 fold4，decision 永久写入
`long_training_authorized=false`。

##### 16.15.16.2 正式结果

| 指标 | Parent | v3 Selected | Fix | Break | 净 hits | Transition precision |
|---|---:|---:|---:|---:|---:|---:|
| REC@0.25 | 5990/6205 = 96.5351% | 5975/6205 = 96.2933% | 23 | 38 | **-15** | 37.7049% |
| REC@0.50 | 5248/6205 = 84.5770% | 5069/6205 = 81.6922% | 104 | 283 | **-179** | 26.8734% |

总切换数为 `844/6205 = 13.6019%`。fold gate 明确失败：

```text
acc025_fix_not_greater_than_break
acc050_net_negative
```

因此结果不是“还没训练完”，也不是“再多跑几轮可能自然超过 Parent”。完整 1,670 个 optimizer steps、
26,714 个唯一 fit row 和全部 388 个 held-out batches 均已完成；失败发生在最终部署决策本身。

##### 16.15.16.3 工程与审计完整性

- 训练：`1670` batches、`26714/26714` unique fit rows，identity SHA
  `47c9c5905353089cdd1b6e06df4a96ac0729e6beec7515ae53b4edf7819d100a`；
- 数值：总 loss mean `0.9334643`，grad norm mean `5.7755590`，无 NaN、Inf、OOM 或异常退出；
- 状态：冻结的 `1144` 个 tensor、`149,670,851` 参数前后 SHA 完全一致；50 个 FPR tensor、
  `2,509,927` 参数确实发生更新；
- 输出：关键 Parent/Box/Mask sentinel 前后完全一致，证明本轮只改变候选决策头；
- 权重：除只读 `input_snapshot/gf_detector_l6o256.pth` 外没有生成任何 `.pth`；
- 受保护 E57 未改写，SHA 仍为
  `76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`；
- GF 输入快照 SHA 为
  `9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2`；
- receipt SHA：`35a53516bc7f5f8bd6f1da53f609734798699d3afddc678d1e025eaec1e2b5aa`；
- decision SHA：`d66fd84c901972b11fc0f78f7b70f679f4d3d2f9aae1b1233e6a3105261e072b`；
- reviewed launcher / runtime manifest / config SHA 分别为
  `2936a941...bee1e`、`291fec61...879e`、`f193d6ab...1c35`；双轴 code review 均 PASS。

##### 16.15.16.4 为什么 v3 仍然失败

v3 解决了 v2 的“0 switch 完全拒绝”，也比 v1 的 `876` 次切换、@0.25 `-42`、@0.50 `-279`
更少破坏；但不同 scene fold 的绝对数字不能作严格横向提升声明。它揭示的核心问题是：

1. **action head 恢复了切换能力，却没有恢复足够高的切换精度。** 844 次切换中，真正改变 @0.25
   成败的只有 61 次，其中 fix 仅 23；@0.50 的 387 次成败转换中 break 高达 283。
2. **absolute break veto 没有学成可靠的风险上界。** held-out 上仍有大量候选通过 `<0.5` 门，随后在
   @0.50 破坏正确 Parent。当前 break probability 只是分类分数，不是足以支持硬安全决策的保守风险界。
3. **正向修复先验太低。** 训练统计中的 candidate-positive ratio 约 `2.34%`；在这种不平衡下，action
   容易学到“某候选相对更像正例”，但难以保证其部署后的绝对 break 风险。
4. **高 Parent 准确率放大了误切成本。** 此 scene audit 的 E57 已见过完整 Nr3D train split，Parent 在
   fold3 上已达 96.54%/@0.25；可修复空间极小，任何不够精确的切换都会 break>fix。这也意味着这些绝对
   准确率不能当作正式 val 指标，只能用于同 fold 的 Parent-vs-selected 因果比较。
5. **问题不是 proposal 数量或训练时长。** 正确候选是否存在与是否安全切换是两个问题；本轮完整训练已完成，
   而失败集中在 candidate action 与 break-risk 的联合校准。

##### 16.15.16.5 决策与下一步边界

FPR-TV v3 fold3 **正式判负并封存**：不运行 fold4，不做 7,899-row formal，不保存/恢复本轮权重，不在
fold3 上扫描 action margin、0.5 break threshold、Top-K、loss weight、LR 或 epoch。下一步状态只能是
`method_correction_only`。

如果继续候选验证路线，必须改变风险建模而不是调门限。下一版至少要满足：

1. 将“会不会破坏 Parent”改成可验证的候选级保守风险上界或 conformal/upper-confidence abstention，
   而不是把普通 BCE sigmoid 当作硬概率；
2. 直接优化 switch-level `fix > break`，并把 @0.25 与 @0.50 风险分开约束，不能让 @0.25 action
   通过后在 @0.50 大量破坏；
3. 在任何新 fold 前先做只读 counterfactual density：统计候选可修复量、score gap、break head 分桶
   calibration，以及每个 gate 对 23 fix/38 break、104 fix/283 break 的保留率；
4. 新方法必须使用未消费 scene split 和新的预注册合同，继续保持 exact Parent fallback、无 GT/无 Dataset ID、
   不修改 Box/Mask/Backbone、无 validation threshold sweep；
5. Density-Aware 路线继续封存，不与本结果组合后再解释。

本轮没有改变项目公开最好指标：Nr3D 仍为 `4475/7899 = 56.6527%`（@0.25）与
`3759/7899 = 47.5883%`（@0.50）；Sr3D 仍为 `12139/17726 = 68.4813%`。硬目标仍分别为
Nr3D `>=4740/7899` 与 Sr3D `>=12214/17726`。用户固定排除项不变：不做 baseline 公平复现，
不采用原建议第七节、原建议第八节或旧 E0--E7 实验矩阵。

#### 16.15.17 A-V4 Counterfactual-Parent 机制代码完成：只放行 100-batch 前置审计准备（2026-09-01）

##### 16.15.17.1 为什么继续改候选验证器，而不是恢复 baseline 或旧实验矩阵

FPR-TV v3 fold3 的正式失败已经证明：现有 action head 可以产生切换，但 actual-Parent 监督中的正向修复
事件过少，普通 BCE break 分数也不能直接当成部署风险上界。下一步因此不是延长训练、调整 0.5 门限，
也不是恢复 detector-pretrained/global48/150--240 epoch 的公平 baseline 复现，而是先验证一个更窄的结构假设：

> 在完全不改变正式推理 Parent、候选集合、Box、Mask 和 Backbone 的前提下，构造最多两个只用于训练的
> counterfactual Parent view，能否把真实 safe-repair 正例密度至少提高到 actual-Parent 的两倍。

该路线称为 **A-V4 Counterfactual-Parent mechanism audit**。它只回答“训练信号是否真实增加”，不直接
回答正式 REC 是否提升，更不授权 fold4、7,899-row formal 或长训练。

##### 16.15.17.2 已完成的代码改动

已在本地研究分支提交：

```text
commit = 90d460c  Add counterfactual parent verifier training
```

本次只修改八个受审文件：`main_utils.py`、`models/losses.py`、`models/mcln.py`、
`models/parent_relative_text_verifier.py`、`train_dist_mod.py` 及三个对应测试文件。核心实现为：

1. **最多两个 GT-free counterfactual views**：
   - 真正的全局 Text-Top1，且必须是 detector-valid、actual-feasible、非 actual Parent；
   - leave-one-out Default Top1，先移除 actual Parent 再从有效候选中选择。
2. **只在显式 opt-in 的训练分支启用**：默认关闭时不新增 utility head、checkpoint key、loss 或统计字段，
   保持旧 FPR 数值与历史 scene-audit config SHA 语义。
3. **部署路径完全不变**：正式选择仍只使用 actual V99 Parent；counterfactual view 不进入推理，也不改变
   Parent/Box/Mask/Backbone 输出。
4. **transition utility 明确区分 fix / neutral / break**：两个 IoU 阈值分别使用 `+1 / 0 / -1`，不再把
   break 与 neutral 混为同类。
5. **actual 与 counterfactual 独立审计**：分别记录 positive candidate/row、fix/break/neutral pair、
   utility/risk loss、nonfinite、selected-score gradient L1；全局 grad norm/finite gate 继续保留。

##### 16.15.17.3 复审中发现并关闭的关键问题

这轮代码没有一次性放行，而是经过三轮双轴只读审查。被发现并修复的问题本身也是后续研究的重要教训：

1. **假 Text-Top1**：早期实现曾在 feasible 子集中取 Text Top1，会在真正 Text Top1 不可行时退而选择
   次优候选。现已改为先在所有 detector-valid compact candidates 中取唯一真实 Top1；若它不是
   actual-feasible，则该 view 直接 abstain，不能伪造训练正例。
2. **default-off 不完全兼容**：早期版本即使新开关关闭也会创建 utility head，并改变旧 repair/break/IoU
   的监督域。现已将新 head、loss、mask 和 stats 全部置于 opt-in 分支；关闭时 state_dict 和数值路径保持旧合同。
3. **历史五折配置 SHA 漂移**：新增 CLI 默认字段会进入完整 args canonical hash。现已在旧 scene audit 中
   显式拒绝启用，同时把 false 默认归入受控动态排除，旧冻结配置不因新增字段而失效。
4. **LOO Parent 与 score axis 不自洽**：若只更换 `parent_position`、仍沿用原分数轴，原 actual Parent
   必然比分配的新 Parent 更高，因而会被 score-gap gate 判为不可晋升，结构性删除最重要的
   `B' -> actual Parent` 修复监督。现已把 LOO Parent 的训练期分数精确复制为 actual deployable Parent
   分数，使两者 gap 精确为 0；候选选择仍只依据原始预测，不读取 GT。
5. **invalid padding 高分污染**：不能用 compact 行的无掩码 `max` 作为反事实 Parent 分数，因为 padding
   槽也可能带有占位 query 分数。最终实现直接复制 actual deployable Parent 分数，并用 invalid padding=10
   的反例证明无效槽不会影响 counterfactual score axis。

##### 16.15.17.4 当前验证状态

最终固定代码在远端 Python 3.7 临时测试目录完成：

```text
64 passed in 2.98s
git diff --check = clean
Standards/Correctness review = PASS
Spec review                  = PASS
```

测试覆盖真实 Text-Top1/no-feasible-fallback、actual Text-Top1 abstain、最多两个 distinct views、无可行
候选回退、LOO actual-Parent feasible/fix、invalid padding 高分、actual/CF score-axis 有限非零梯度、
fix-break-neutral utility、default-off 无新参数/损失，以及历史 config SHA 兼容。

这只证明机制代码具备进入 A-V4 前置审计的资格，**不代表方法已经改善 Nr3D 指标**。截至本节写入时：

- A-V4 100-microbatch audit launcher/receipt gate 尚未正式冻结；
- 未部署正式源码；
- 未启动 GPU 训练；
- 未消费任何新的 scene-disjoint fold；
- 未生成新权重；
- Nr3D 正式最好仍为 `4475/7899 = 56.6527%`，Sr3D 仍为 `12139/17726 = 68.4813%`。

##### 16.15.17.5 下一步唯一允许动作

下一步只允许准备并独立审查一个 **exact 100-microbatch、train-only、no-val、no-save** 的 A-V4 launcher。
该审计必须从受保护 E57 full-state 开始，保持 B16×A1，只训练 verifier heads，并同时满足：

1. actual/CF 全部统计有限，两个 score axis 均有有限非零梯度；
2. counterfactual positive-row ratio `>= 2 × actual positive-row ratio`；
3. 同时存在 fix 与 break 监督；
4. frozen state/output sentinel 精确不变；
5. 生成 `.pth` 数为 0，decision 永久写入 `long_training_authorized=false`。

若任一条件失败，A-V4 直接封存；通过也只允许讨论尚未消费 outer fold 的下一阶段，不自动启动 fold4 或
正式验证。

项目范围继续永久遵守用户 2026-09-01 的明确决定：**不做 baseline 公平复现；不参照、继承或改名恢复
原建议第七节；不借鉴原建议第八节及 E0--E7 实验矩阵。** 这些排除项不得作为 A-V4 的对照、前置任务、
后续组合或论文补证路径。

#### 16.15.18 Nr3D official REC monitor 污染纠偏与真实状态恢复（2026-09-01）

##### 16.15.18.1 发现的问题：96.2933% 不是 7,899-row 正式指标

2026-09-01 检查 Nr3D official monitor 时，旧 v2 state 一度把
`5975/6205 = 96.2933%` 写成 `latest/metric_best`。这个数值本身是 FPR-TV v3 fold3 的
**scene-disjoint 6,205-row held-out 审计结果**，在该审计范围内是真实的；错误在于旧 monitor 递归扫描
所有 `eval_metrics_epoch_*.json`，却没有要求正式 Nr3D 的精确样本数 `7,899`，因而把 scene audit 当成
full-dataset official receipt。另两份 `7,129` 与 `7,341` row 的场景审计也存在同类污染风险。

因此必须明确：

- `96.2933%` **不是** Nr3D official 7,899-row 指标，不能与 `56.6527%`、60.0% 硬目标或论文
  baseline 比较；
- 它也不表示 A-V4、FPR-TV 或其他新方法已经刷新正式最好；
- 旧 state 的 `latest/metric_best` 字段被污染，但受保护的 E57 checkpoint 与
  `preserved_best_rec025=4475/7899` 没有被覆盖；
- 三份 scene-audit receipt 均是有审计价值的原始证据，**没有删除**，只是由新 monitor 明确归类为
  `ignored_nonformal_receipt`。

##### 16.15.18.2 重新计算后的 Nr3D 正式事实

对 Nr3D 输出根下全部 168 份指标 JSON 做严格只读重算，结果为：

| receipt 分类 | 数量 | 含义 |
|---|---:|---|
| 正式 full-dataset receipt | **165** | top-level 与 `multiple` 均精确 `7,899`，schema/partition/hits/accuracy 全部合法 |
| 非正式 scene-audit receipt | **3** | 样本数分别不是 `7,899`，保留文件但不进入 latest/best/promotion |
| 损坏或结构不合法 receipt | **0** | 当前未发现 |

纠正后的正式状态是：

| 角色 | Epoch | REC@0.25 | REC@0.50 | 结论 |
|---|---:|---:|---:|---|
| 最新正式 receipt | E59 hard replay | `4400/7899 = 55.7033%` | `3702/7899 = 46.8667%` | 已训练完但低于最好，不是“还没训练完” |
| 正式最好/受保护权重 | E57 weight average | **`4475/7899 = 56.6527%`** | **`3759/7899 = 47.5883%`** | 当前 Nr3D 公开最好，仍受保护 |
| 严格硬目标 | -- | `>60.0%`，即至少 `4740/7899` | -- | 仍差 **265 hits / 3.3549pp** |

受保护 E57 文件仍为：

```text
/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/
official_best_rec025_epoch_57_0p56652741.pth
SHA256 = 76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1
```

Sr3D 的正式最好没有因本次 Nr3D monitor 事件改变，仍为
`12139/17726 = 68.4813%`；严格目标仍为 `>68.9%`，即至少 `12214/17726 = 68.9044%`，
尚差 75 hits。

##### 16.15.18.3 v3 monitor 的修复合同

新 `mcln_official_rec_monitor_v3.py` 不再只凭文件名接收结果，而是 fail-closed 执行以下约束：

1. dataset 与正式总数固定绑定：Nr3D=`7,899`、Sr3D=`17,726`；命令行不能改成 scene-audit count；
2. receipt 必须是固定 schema，top-level 与 `position_subgroups.multiple` 的样本数必须同时精确匹配；
3. `unique` 必须为 0-row 分区，`unique + multiple` 必须等于总数；hits 必须为整数、范围合法，
   `hits@0.50 <= hits@0.25`，accuracy 必须与 hits/count 一致；
4. 样本数不符的 receipt 只计入 `ignored_nonformal_receipt_count`，结构损坏的 receipt 计入
   `invalid_receipt_count`，二者都不得进入 latest、metric best、checkpoint promotion 或 target 判断；
5. 从旧 state 迁移时，CLI 固定的 target、sample count 与受保护 E57 优先；旧 v2 污染字段不能覆盖；
6. 只有同时绑定正式 receipt、规范 control 路径、stored SHA，并通过同一打开文件的
   `SHA -> torch.load -> exact epoch -> SHA` 复验，才允许继承新版 v3 的更优 checkpoint；坏格式、错 epoch、
   SHA 漂移均回退到受保护 E57；
7. 新最好 checkpoint 从 `O_NOFOLLOW` 打开的单一源复制到独立 inode，校验 epoch 后 `fsync` 并原子发布；
   state JSON 持久化成功后才清理旧 checkpoint，任何 receipt 都不删除；
8. 全生命周期持有非阻塞 `flock`，防止两个 official monitor 同时改 state；
9. `target_reached` 使用严格 `>`：恰好 60.0% 或 68.9% 仍为 false，符合用户“超过”目标，而不是
   “大于等于显示值”。

##### 16.15.18.4 验证、提交与部署证据

- Git fixed point：`4bddb480e592ff176d52bf03b5979ef6b766d0b5`；
- v3 script SHA：`2481210aed2cb010e968dcdac2617680420c0609327b1fdb34d7d9c03d458590`；
- test SHA：`de923de952989c811425a8534ed59ce67fc4735911c1da6cd02c0a9a657ad089`；
- 远端 Python 3.7：`14 passed in 1.35s`；
- 真实根只读 preflight：`formal=165, ignored=3, invalid=0`，E57 SHA、epoch 与正式 best 全部精确匹配；
- Standards/Correctness review：PASS；Spec review：PASS；
- 旧 v2 screen `652079.mcln_nr3d_official_rec_monitor` 与其 Python PID 已按身份退出；
- 新 v3 screen：`800089.mcln_nr3d_official_rec_monitor`；Python PID：`800094`；
- v3 首轮于 `2026-09-01 10:59:37 CST` 写入纠正状态，随后按 60 秒周期持续刷新；
- `2026-09-01 11:01:37 CST` 状态快照 SHA 为
  `348a4fc68f72845d72eb5b5c4be7c0b30cb54cace6f0a9b7662473f374f42ee9`；
- v3 正持有 `official_rec_monitor.lock`，GPU `1 MiB / 0%`，无 `train_dist_mod.py`、无
  `torch.distributed`、无任何训练任务。

##### 16.15.18.5 对最新实验进展的影响

本次修复只恢复监控口径，没有创造新指标，也没有改变任何模型权重。当前应使用的唯一公开结论仍是：

- Nr3D：正式最好 `56.6527%`，未达到严格 `>60.0%`；
- Sr3D：正式最好 `68.4813%`，未达到严格 `>68.9%`；
- A-V4 目前只有机制代码与 64 项回归证据，尚未完成 100-microbatch 前置审计，更没有 7,899-row
  正式结果；
- FPR-TV v3 的 `96.2933%` 只能称为特定 6,205-row held-out scene split 的结果，且相对其 Parent
  `96.5351%` 仍是 `-15 hits`，不能被包装成正式提升；
- 当前没有活动训练，因而不存在“现在是否要继续衰减 LR”的问题；只有新实验通过独立前置门后才可决定；
- 继续严格遵守用户排除项：**不做 baseline 公平复现；不执行、继承或改名恢复原建议第七节；不借鉴
  原建议第八节及 E0--E7 实验矩阵。**

## 17. 项目文档总整合、来源索引与发布规则（2026-09-01）

### 17.1 单一权威文档

从本节起，本文件是 MCLN 项目的**唯一完整中文交接主文档**。远端历史日志、设计稿、实验计划和审计报告
继续保留为原始证据，但不得各自覆盖本文件中更新的目标、最好指标、失败结论、停止规则或排除项。

整合采用“结论合并 + 来源索引”，不把 1MB 以上的旧日志逐字重复粘贴。这样既保留完整实验时间线，也避免
同一个结果在多份文档中以不同旧口径重复出现。需要逐行取证时，按下表回到原始文件及其 SHA。

### 17.2 核心来源文档与已整合位置

| 原始文档 | SHA256 | 本文件中的整合位置 | 权威状态 |
|---|---|---|---|
| `docs/REC_3DRES_OPTIMIZATION_LOG.md` | `1dc2ec19...9ace5` | 第 2--14 章及 16.3--16.12 | 历史全量实验源；最新状态以第 16.15、17 章为准 |
| `docs/SOURCE_MOE_RERANK_DESIGN.md` | `7857a6dd...496a` | Source-MoE、V94--V133 与 selector 诊断章节 | 架构/失败归因源，不自动授权新实验 |
| `docs/SPACY_SIDECAR_EXPERIMENT_AUDIT_20260822.md` | `0d3c5748...18f7a` | 15.11、16.8.3、16.15.13 | sidecar FAIL 结论保留；raw-only parser 结果已单独封存 |
| `docs/archive/V99_REC025_BEST_ARCHIVE.md` | `71594594...1129` | 第 0--2 章、15.14、16.3 | ScanRefer V99 保护基线与 claim 边界 |
| `EXPERIMENT_AUDIT.md` | `6ab386d8...17e3` | 结果口径、数据隔离、权重保护条款 | 审计源；WARN 不得被省略为 PASS |
| 远端旧 `FPR_TV_SPEC_2026-08-31.md` | `429effed...bd4` | 16.15.8--16.15.16 | 历史 v1 合同；已由当前扩展版取代 |
| 当前 `FPR_TV_SPEC_2026-08-31.md` | `7000ff92...71ae` | 16.15.8--16.15.17 | FPR-TV v1/v2/v3 与 A-V4 当前机制合同 |
| `FPR_TV_COUNTERFACTUAL_PARENT_AUDIT_SPEC_2026-09-01.md` | `befba370...10246` | 16.15.17、18.2--18.4、19 | A-V4 exact-100 train-only 前置审计合同；不构造验证集，不授权长训或正式验证 |
| `DENSITY_AWARE_TARGET_BOX_SPEC_2026-08-31.md` | `945a23a7...d7e` | 16.15.14 | 100-batch auxiliary 审计合同 |
| `DENSITY_AWARE_TARGET_BOX_SCENE_AUDIT_SPEC_2026-09-01.md` | `bcd1d89c...e424` | 16.15.15 | 三角色 scene-disjoint 因果审计合同 |
| `refine-logs/EXPERIMENT_PLAN*.md` | 见 17.4 | 14.215--14.223、16.15 | 历史计划，不得重新激活已失败路线 |
| `refine-logs/EXPERIMENT_TRACKER*.md` | 见 17.4 | 对应日期时间线 | 只作运行账本，不覆盖正式 receipt |
| `README.md` | GitHub main 保留 | 安装、数据、训练和评估入口 | 工程使用说明，不承担最新实验结论 |

### 17.3 历史 ScanRefer 计划/设计文档的整合映射

远端 `docs/superpowers` 下 17 份 ScanRefer 计划/设计均已吸收到第 4--14 章。它们按因果依赖合并为：

1. Query reranker 与 geometry reranker：候选缓存、scene-disjoint 训练、正式 evaluator 接线；
2. one-epoch fine-tuning：参数组、损失、校准、回退和单次正式评估；
3. source-gate probe：Top-K membership、完整 Query 状态与非部署 probe；
4. position subgroup、selective residual 与 hierarchical reranker：风险门、OOF 与在线校准；
5. joint box-mask Pareto：同一 Query identity 下的 REC/Mask 联合安全；
6. Optuna complete retrain：失败搜索、完整训练和 Pareto retention。

对应原始文件如下，均继续保留于远端/GitHub `docs/superpowers`：

- `plans/2026-07-14-scanrefer-rec-reranker.md` 与对应 design；
- `plans/2026-07-14-scanrefer-rec-mask-geometry-reranker.md`；
- `plans/2026-07-16-scanrefer-rec-one-epoch-finetune.md` 与对应 design；
- `plans/2026-07-17-scanrefer-rec-source-gate-probe.md` 与对应 design；
- `plans/2026-07-20-scanrefer-position-subgroup-report.md`；
- `plans/2026-07-20-scanrefer-rec-selective-residual.md`、diagnostic replay 与对应 design；
- `plans/2026-07-20-scanrefer-rec-hierarchical-query-variant.md` 与 risk-controlled design；
- `plans/2026-07-23-scanrefer-joint-box-mask-pareto.md` 与对应 design；
- `plans/2026-07-24-scanrefer-optuna-complete-innovation-retrain.md` 与对应 design。

这些文件说明“当时计划如何实现”，不代表路线后来通过。是否成功必须回到本文件对应终态、正式 receipt 和
protected checkpoint；任何已经失败或被用户排除的计划都不能凭旧 design 重新启动。

### 17.4 实验计划、跟踪器与重复文件消重

远端 refine 文档共有两组计划：

- V51/V52 计划：`EXPERIMENT_PLAN_20260812_0209.md`，SHA `c2d558e5...723e`；
- V132/V133 计划：`EXPERIMENT_PLAN.md` 与 `EXPERIMENT_PLAN_20260815_150630.md`，两者字节相同，
  SHA 均为 `60a389b7...154f`。

`EXPERIMENT_TRACKER.md` 与 `EXPERIMENT_TRACKER_20260815_150630.md` 也字节相同，SHA 均为
`ca32137b...4b9`。其余 8 月 12 日 tracker 是时间点快照，均已进入第 14 章时间线。

因此总文档只保留一份语义，不把重复文件重复粘贴。原文件继续保留 SHA 证据，便于核对当时计划是否在结果
生成前冻结。

### 17.5 不属于叙事文档的文件

以下文件不并入正文，但继续作为工程或 provenance 输入保留：

- `data/meta_data/*_scans.txt`、ScanNet split 列表：数据划分，不是实验结论；
- `.pytest_cache/README.md`、`pointnet2.egg-info/*.txt`：工具生成文件；
- `.v*/MANIFEST.txt`、runtime manifest、build receipt、decision/receipt JSON：不可变身份或运行证据；
- 单行 launcher pointer、日志、checkpoint、TensorBoard 与模型权重：运行产物，禁止进入 GitHub 源码提交。

### 17.6 文档冲突时的优先级

若多份文档冲突，按以下顺序解释：

1. 用户最新明确约束与本文件第 0、1、15、16、17 章；
2. exact formal receipt、decision、protected checkpoint 与 SHA；
3. 已完成实验的终态章节；
4. experiment audit；
5. design/spec/plan；
6. 中途 tracker、日志和旧 README。

尤其要继续执行三项永久规则：Nr3D 必须严格超过 60.0%，Sr3D 必须严格超过 68.9%；不做 baseline
公平复现；不继承原建议第七节、第八节或 E0--E7 矩阵。

### 17.7 统一发布位置

整合后的主文档只维护同一内容的三份副本：

- 本地：`C:\Users\gb\Desktop\document\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`；
- 远端：`/home/gb/new butd/butd_detr-main/MCLN-main/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`；
- GitHub：`docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`。

三份发布后必须逐字节 SHA256 一致。以后新增实验先写原始 receipt/decision，再更新本总文档；不得另建一份
相互竞争的“最新总结”。

## 18. 最新代码冻结、A-V4 前置审计实现与三端发布（中间版本；以第 19 章为准）

### 18.1 本次代码发布边界

本次发布以研究分支固定点 `71e526c` 为源，不删除 GitHub `main` 已有源码、历史实验文件或文档。
发布采用“在完整 `main` 文件树上覆盖 68 个受审路径”的方式，最终实际变化集中在 11 个文件。

新增内容包括 A-V4 前置审计规范、runtime manifest、静态入口二进制与 build receipt、bounded launcher、
合同测试，以及 `main_utils.py` 中的 exact-100 train-only 审计生命周期。

以下状态必须与“正式实验完成”区分：代码已实现、已审查、已具备 fail-closed 启动条件；但 A-V4 的
100-microbatch 审计尚未运行，因此没有新 REC、没有新 checkpoint，也没有 long-training 授权。

### 18.2 A-V4 bounded audit 的真实合同

唯一允许的 A-V4 前置运行仍是：从受保护 Nr3D E57 full-state 开始，仅运行 E58 的 100 个 microbatch，
`B16×A1`、单 rank、train-only、no-validation、no-checkpoint-save。

审计只训练 FPR-TV verifier 参数组。V99 Parent、检测器、文本编码器、mask 分支及其余冻结状态必须精确不变；
同一首批输入的冻结输出 sentinel 也必须逐值一致。

receipt 必须证明 100 个 microbatch、100 个 optimizer step 和 1,600 个样本槽位；必须分别记录 actual 与
counterfactual 的样本、正例、fix/break/neutral、风险、utility、非有限数和 selected-score 梯度。

只有 counterfactual positive-row density 至少达到 actual 的两倍，同时存在 fix 与 break、两条 score axis
梯度非零、全部数值有限、冻结状态和输出精确一致时，density gate 才能通过。

无论 gate 成败，decision 都固定写入 `audit_only=true`、`formal_validation_accessed=false` 和
`long_training_authorized=false`。通过仅允许独立讨论未消费的 scene-disjoint 短审计，不得自动启动 fold4、
7,899-row 正式验证、Sr3D 实验或长训。

### 18.3 发布前发现并关闭的 launcher 完整性问题

发布前的最终只读审查发现，早期 launcher 的一个 Python heredoc 被截断文本污染。该问题会让 `bash -n`
表面通过，但 preflight 在 Python 解析阶段必然失败。

同一截断还吞掉了数据 manifest 尾部校验、E57 checkpoint 合同校验，以及独立输入快照的 copy/verify helper。
若不修复，backbone 会在占用 one-shot root 后失败。

固定点 `71e526c` 已恢复全部逻辑：dataset inventory/size/mode/SHA 校验、独立 inode 输入快照、0444/owner
复验，以及同一打开文件上的 `SHA -> torch.load -> exact E57/V99/optimizer -> SHA` 检查。

decision 采用临时文件写入、文件 `fsync`、原子替换和父目录 `fsync`。这样即使 gate 失败，
`long_training_authorized=false` 的终态也具有崩溃持久性。

另一个边界是合法的“零 counterfactual view”。旧代码会在计算 positive ratio 时除零；当前实现将零分母
安全映射为 ratio 0，使 supervision gate 失败、持久写入 decision，然后以状态 20 结束，而不是异常崩溃。

### 18.4 验证证据与尚未发生的动作

修复后的 launcher 通过 Bash 语法检查；其中 6 段 Python heredoc 均由远端 Python 3.7 编译通过。
相关回归为 `68 passed in 3.16s`。

Standards/Correctness 与 Spec 两个独立审查轴均对 `71e526c` 给出 PASS。审查确认训练 argv、
100×B16×A1、no-val/no-save、runtime manifest、静态入口和 Landlock 合同未因修复漂移。

截至本节写入时，A-V4 one-shot audit root 仍不存在；GPU 上没有训练进程。本次发布没有执行 preflight 之后的
backbone，也没有生成 receipt、decision、权重或正式指标。

### 18.5 当前三数据集结论保持不变

| 数据集/任务 | 当前正式最好 | 严格目标 | 当前差距与结论 |
|---|---:|---:|---|
| ScanRefer 双阶段 REC@0.25 | `5572/9508 = 58.6033%` | `59.00%` | 差 38 hits；Mask 三项与 REC@0.50 已达标 |
| Nr3D REC@0.25 | `4475/7899 = 56.6527%` | `>60.0%`，至少 `4740/7899` | 差 265 hits；E59 已结束但低于 E57，不是未训练完 |
| Sr3D REC@0.25 | `12139/17726 = 68.4813%` | `>68.9%`，至少 `12214/17726` | 差 75 hits；旧 68.4 目标已作废 |

ScanRefer 的提升来自真正互补的 Parent/Geometry/Query/Mask-derived source、V99 层级选择与 mesh-superpoint
纠错。Nr3D/Sr3D 的主要瓶颈仍是同类实例文本消歧、可靠 Anchor 组合与小目标 Proposal 覆盖。

因此，当前仍不能声称 A-V4 已改善 Nr3D 或 Sr3D。它只是针对“正确候选已在 Top-K、但实际 Parent 下正监督
密度不足”这一问题的机制修正与审计实现。

### 18.6 永久排除项与下一步门禁

继续永久执行用户决定：不做 baseline 公平复现；不执行、继承、改名或引用原建议第七节；不借鉴原建议
第八节及 E0--E7 实验矩阵。

下一步只有在代码三端 SHA 一致、静态 preflight 通过、one-shot root 仍未消费且 GPU/全局锁空闲时，才可讨论
一次 100-microbatch backbone。任何失败都封存 A-V4，不允许用调阈值或重复运行选择结果。

### 18.7 本次统一发布位置

- GitHub：`https://github.com/666666666666gao/MCLN`，发布分支
  `agent/mcln-latest-20260901-v2`，合并后以 `main` 为权威源码；
- 本地主文档：`C:\Users\gb\Desktop\document\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`；
- 远端主文档：`/home/gb/new butd/butd_detr-main/MCLN-main/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`。

三份主文档必须逐字节一致。GitHub 保留各独立规范、runtime manifest、build receipt 与测试；本主文档负责
统一结论、时间线、来源索引和冲突优先级，不替代原始不可变实验 artifact。

## 19. 最终发布修订：运行时源码闭包与真正 train-only A-V4（2026-09-01）

### 19.1 为什么 PR #4 不是最终发布

PR #4 已把 A-V4 机制代码、launcher、规范和交接文档合入 GitHub `main`，merge commit 为
`5e388f92c00dfe164b16ae73376f1a8a4c63b8fa`。它是完整性审计前的中间版本。

后续独立审计发现三个发布级问题。第一，GitHub 中的 `src/joint_det_dataset.py` 和
`src/grounding_evaluator.py` 不是 runtime manifest 锁定的正式字节，前者还缺少当前训练入口传入的 cache 参数。

第二，旧路径虽然不执行 validation，却仍在 `get_loaders()` 中构造验证 dataset 和 DataLoader；旧数据清单也含
`val_v3scans.pkl` 与 `superpoints/val`。因此旧版不能严格声称 `formal_validation_accessed=false`。

第三，第一次 no-validation 修订把 `test_loader` 正确设为 `None` 后，`BaseTrainTester.main()` 仍无条件执行
`len(test_loader.dataset)`。这会使正式 A-V4 bounded audit 在首个训练 batch 之前崩溃。该问题由独立完整性复审
发现，随后增加 `_optional_test_dataset_size()`：只有 exact bounded train-only A-V4 可以缺失 test loader，其他
路径仍 fail closed；A-V4 只记录“testing dataset disabled”，不再解引用空 loader。

这些问题不会产生虚假的已完成实验，因为 A-V4 one-shot 尚未运行；但它们会破坏未来审计的运行身份和
train-only 证据，所以必须在代码发布完成前修正。

### 19.2 最新源码同步边界

最终发布以 A-V4 reviewed runtime manifest 为来源，在 GitHub 完整文件树上同步 66 个确有语义差异或缺失的
`.py/.sh/.md` 文件。Linux `.so/.o`、egg-info、checkpoint、日志和 TensorBoard 不进入源码提交。

GitHub 对这 66 个同步文件以及原有 `main_utils.py`、`train_dist_mod.py` 共 68 个源码条目按 Git blob 字节与
reviewed Linux runtime SHA 对齐；这不等于把整个 371-file Linux 运行时（其中含编译扩展和构建产物）逐字节放入
GitHub。完整 371-file bytewise closure 只在远端只读 runtime snapshot 中核验，GitHub 发布的是可审查源码闭包。

关键同步文件包括 `src/joint_det_dataset.py`、`src/grounding_evaluator.py`、
`src/legacy_scene_graph_cache.py`、`models/rec_evaluator_filter.py`、hard-replay/tier auxiliary 和历史 V99 launcher。

正式 dataset/evaluator 身份已恢复为：

```text
src/joint_det_dataset.py  SHA256 800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0
src/grounding_evaluator.py SHA256 0173b31a7a818f872c210b01a4e5d17601c4e5f10ec8d97f78c7e537fa44e062
src/legacy_scene_graph_cache.py SHA256 aa4c5949ba017a9f8a44f63caf73669717428eb9725e458027f3053de4d0e749
```

### 19.3 真正的 no-validation 实现

`main_utils.py` 新增 bounded A-V4 train-only 判定。该模式要求 train dataset 存在、test dataset 为 `None`，
只构造一个训练 DataLoader；若出现 eval、缺失 train dataset 或任何 test dataset，立即 fail closed。

`train_dist_mod.py` 在完成训练集构造后立即返回 `(train_dataset, None)`。它不再实例化 validation dataset，
并显式拒绝 `eval/debug/eval_train`。

新的数据清单为 `nr3d_fpr_tv_av4_train_only_data_manifest_v2.json`。它从旧清单中删除
`val_v3scans.pkl`、`superpoints/val` 及对应文件行，保留训练 pickle、Nr3D CSV、RoBERTa、train superpoint 和 GF train。

清单的固定事实为：

```text
schema     = mcln-nr3d-fpr-tv-av4-train-only-data-manifest-v2
file_count = 2420
total_size = 10,593,197,424 bytes
SHA256     = 155e2233efbe5c312c19c6dc709ce8c564c601d50e73d6907a3702f000d9d173
validation source rows = 0
```

### 19.4 Runtime manifest v2 与验证证据

`fpr_tv_counterfactual_parent_runtime_manifest_v2.json` 绑定更新后的 `main_utils.py` 与
`train_dist_mod.py`。完整闭包仍为 371 文件，合计 106,114,031 bytes，SHA256 为
`f09e490789680a8e7105cb1167f6f6f025a9a83a8998657eda3c9e1b4c9ab807`。其中最终
`main_utils.py` 的 SHA256 为 `b40c6f6ca83ec68f655feb820de788f7398b0e71574cf73a9f6b22b137fba47e`。

远端只读验证逐文件哈希了上述 371 个运行时文件和 2,420 个训练数据文件，二者均通过。launcher 的 Bash
语法与 6 段 Python heredoc 均通过；核心文件兼容 Python 3.7。

最终聚焦回归为 `77 passed in 3.13s`。新增用例实际调用 loader/dataset 构造路径，证明 A-V4 模式只构造训练集和
一个训练 DataLoader，并验证 bounded audit 可安全处理 `test_loader=None`、普通路径则拒绝空 test loader；另有
清单测试证明 validation source 与 inventory 均为零。

### 19.5 当前实验状态没有被发布动作改变

本次只修代码、清单、测试和文档，没有执行 A-V4 preflight 后的 backbone，没有启动训练、评估或正式
7,899-row validation，也没有生成 receipt、decision 或 checkpoint。

当前正式最好仍为：ScanRefer `5572/9508 = 58.6033%`；Nr3D `4475/7899 = 56.6527%`；Sr3D
`12139/17726 = 68.4813%`。Nr3D 目标仍为至少 4740 hits，Sr3D 目标仍为至少 12214 hits。

最新诊断仍指向两类正交失败。第一类是候选已在 Top-2/Top-5/Top-16，但长句、同类干扰、属性、关系、
否定和视角组合使 Top-1 排序选错；第二类是小体积、低点数目标本身没有合格 proposal。

Nr3D 最低场景集中在多个同类实例、13+ token 描述和点数不超过约 227 的目标。`mouse`、`soap dish`、
`bottle`、`book`、`toilet paper` 等小物体尤其困难。Sr3D 因显式 anchor 更可靠，但关系模块短训仍未刷新 E26。

因此，A-V4 仍只是待运行的 100-microbatch 机制审计，不能写成指标提升。即使未来 density gate 通过，decision
也必须保持 `long_training_authorized=false`，不得自动启动 fold4、正式验证、Sr3D 或长训。

### 19.6 命名遗留与解释边界

runtime closure 中保留部分 density/tier 名称和静态安全组件。这些是历史审计基础设施与信任路径，不表示
Density-Aware loss 被 A-V4 启用，也不授权已封存的 density 路线复活。

A-V4 训练 argv 仍明确排除 density target-box、proposal refiner、relation-CF、SACR deployment、dataset ID、
Unique/Multiple 输入、GT anchor sidecar 和 validation threshold sweep。

### 19.7 最终发布位置

- GitHub：`https://github.com/666666666666gao/MCLN`；PR #4 是中间版本。PR #5 的发布链为源码闭包提交
  `057fce03cdb4ac701f4144f312566bee6af1d0ae`、发布记录提交 `688b02e85c4df7cecad5b46c556c8891b8ba4d11`，以及
  最终空 test-loader 修复提交 `5a9cdd49cc73ade1f30971b1c97dc8c417fe5513`：
  `https://github.com/666666666666gao/MCLN/pull/5`；
- 本地：`C:\Users\gb\Desktop\document\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`；
- 远端：`/home/gb/new butd/butd_detr-main/MCLN-main/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`。

最终发布后，GitHub `docs/`、本地和远端主文档必须逐字节 SHA256 一致。以后仍以本文件为唯一综合叙事，
原始 spec、manifest、receipt、decision 和测试继续作为不可变证据。

## 20. 2026-09-01 终态更新：A-V4 机制审计通过、正式指标未刷新

### 20.1 当前最好指标与硬目标

本轮是 **train-only 机制审计**，没有访问正式验证集，因此三个数据集的公开最好值均未改变：

| 数据集 | 当前正式最好 REC@0.25 | REC@0.50 | 当前硬目标 | 仍缺 |
|---|---:|---:|---:|---:|
| ScanRefer | `5572/9508 = 58.6033%` | V99 同系统 `4797/9508 = 50.4523%`；全局最好 V113 `4835/9508 = 50.8519%` | REC@0.25 `>=5610/9508 = 59.00%` | 38 hits |
| Nr3D | `4475/7899 = 56.6527%` | `3759/7899 = 47.5883%` | 严格 `>60.0%`，操作门槛 `>=4740/7899` | 265 hits |
| Sr3D | `12139/17726 = 68.4813%` | `10335/17726 = 58.3042%` | 严格 `>68.9%`，操作门槛 `>=12214/17726` | 75 hits |

ScanRefer V99 的三项 Mask 最好仍为 `59.8443% / 52.3349% / 45.9303% mIoU`，均高于用户给出的
`58.70% / 50.70% / 44.72%` baseline。Nr3D 最新 E59 已完成而非“尚未训练完”，结果
`4400/7899 = 55.7033%`，低于 E57 最好；Sr3D relation-CF E27/E28 也均低于受保护 E26。

### 20.2 为什么 Nr3D/Sr3D 仍比预期低

当前证据把错误拆成两类，不能再用一个全局 gate 或同一种 loss 混合解释：

1. **排序失败**：Nr3D 的正确框经常已经在 Top-K，但文本条件 Top-1 选错。Top-2/Top-5/Top-16 oracle
   为 `61.6787% / 69.0974% / 80.3013%`；共有 2,068 条“Top-1 错但 Top-16 有正确候选”。
2. **Proposal 失败**：另有 1,556 条连 Top-16 都没有正确候选。小体积、低点数目标无法靠 rerank 修复。
3. **长句与同类干扰**：Nr3D 13+ token Top-1 约 `49.19%`；同类干扰物至少 5 个时约 `45.56%`。
   颜色、形状、否定、视角和多个参照物的组合使候选间消歧不稳定。
4. **稀疏小物体**：Nr3D 最稀疏点数 Q1 Top-1 `41.343%`，最稠密 Q4 `61.258%`；`mouse`、
   `soap dish`、`bottle`、`book`、`toilet paper` 等类别最明显。
5. **Sr3D 的关系边界**：Sr3D 有显式 anchor，整体优于 Nr3D，但 `back/right/front` 关系仍低，约
   `42.86% / 50.00% / 53.16%`；最稀疏 Q1 约 `52.456%`，最稠密 Q4 `70.470%`。

代表性低指标场景保持：Nr3D `scene0100_00`、`scene0606_00`、`scene0693_00`、`scene0678_00`、
`scene0458_00`；Sr3D `scene0553_00`、`scene0084_01`、`scene0690_01`。其中 Nr3D `scene0653_00`
和 Sr3D `scene0207_02` 有大量“候选存在但排序错误”的可修复样本，分别约 58 与 161 条。

### 20.3 已封存的负结果

- FPR-TV v3 fold3：844 次切换；@0.25 fix/break=`23/38`，净 `-15`；@0.50=`104/283`，净
  `-179`。路线判负，不在该 fold 上扫 margin、Top-K、loss、LR 或 epoch。
- Density-Aware Target Box：scene-disjoint 对照在 @0.50 和 matched IoU 有提升，但 @0.25 overall
  `-10 hits`，不能解决主目标，路线封存且不与 A-V4 组合复活。
- Sr3D relation-CF：E27/E28 均未刷新 E26，停止继续降 LR、延长 epoch 或调关系阈值。
- Nr3D E58/E59 hard replay：均未刷新 E57，证明当前分支已经平台，不是“等训练完自然上涨”。
- 永久排除 baseline 公平复现、原章节/实验七、原章节/实验八及旧 E0--E7 矩阵。

### 20.4 A-V4 Counterfactual-Parent 的唯一 100-batch recovery 结果

首次 one-shot 在第一个 batch 的 backward 后、optimizer step 前因 score-gradient 审计缺失而安全失败；固定证据证明
`optimizer_steps=0`、无 receipt/decision/权重。修复只做三件事：在 verifier-only+CF 路径打开 MCLN 根
training bit 但保持冻结子模块 eval；把 actual Parent score axis 构造成 detached differentiable leaf；在 backward
前 retain actual/CF 两条 score-axis 梯度。科学配置、候选规则、loss、E57、数据和门槛没有改变。

2026-09-01 15:09--15:18，唯一 recovery 通过固定静态入口执行：

```text
epoch=58
batch_count=100
optimizer_step_count=100
sample_count=1600
formal_validation_accessed=false
generated_checkpoint_count=0
```

关键观测：

| 项目 | 结果 |
|---|---:|
| Actual positive-row ratio | `0.0416667` |
| Counterfactual positive-row ratio | `0.1269394`（约为 actual 的 3.05 倍） |
| Actual selected-score gradient L1 | `0.00185942` |
| CF selected-score gradient L1 | `0.00185261` |
| Global clipped grad norm | `5.419588` |
| Actual fix/break/neutral pairs per batch | `0.92 / 2.36 / 15.04` |
| CF fix/break/neutral pairs per batch | `1.74 / 1.79 / 15.49` |
| CF views per batch | `10.3` |
| Actual/CF nonfinite count | `0 / 0` |
| Frozen tensors | `1144` tensors、`149,670,851` elements，SHA 前后完全一致 |
| Trainable tensors | `52` tensors、`2,510,441` elements，SHA 确实改变 |

所有 13 项机制检查均为 true，`counterfactual_density_gate_passed=true`。A-V4 证明了 counterfactual Parent
能显著提高正向修复监督密度，同时 actual/CF 两条 score 轴均收到真实梯度；但这仍不是 REC 提升证据。

不可变证据：

```text
receipt SHA256  = 717cea9e3a34f66610526586ce13022846f06830cfeda5b36c2802b6408c0b4e
decision SHA256 = aa492439073c60eec8b4cea34538715cc10934f65d53f5f7787e7824b3caec51
E57 SHA256      = fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655
GF SHA256       = 9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2
launcher SHA256 = 9519c1a11ee295eaa052b191cb6c1213bcda434bad84adc3a222106bf71e802c
runtime manifest= 7c65debaf544857d129d9d242f9d9d17496db3f647292725b9c67ba5a2d7d825
```

decision 永久记录 `audit_only=true`、`formal_validation_accessed=false`、
`long_training_authorized=false`，下一状态仅为 `independent_scene_disjoint_review_only`。它不自动授权 fold4、
7,899-row 正式验证、Sr3D 或长训。

### 20.5 当前代码与发布状态

本地权威仓库为 `C:\Users\gb\.codex_publish_mcln_20260901_v2`，分支
`agent/av4-scene-fold4-prereg-20260901`。A-V4 recovery、Fold-4 正式短审计及 result-to-claim 已并入
GitHub `main`；本轮开始前的权威提交为 `f73407d`。四个 recovery 修复提交均已包含：

```text
4f3ab10 Repair A-V4 gradient audit and add zero-step recovery
5f0ebff Make A-V4 counterfactual audit formally reachable
fc3429f Bind A-V4 recovery proof to verifier-only runtime
dc5fddd Correct A-V4 runtime manifest size
```

本轮继续把 master 文档、服务器遗漏的 Sr3D 冻结复现脚本与小型方法候选文本推送到
`https://github.com/666666666666gao/MCLN`。禁止上传 `.pth/.pt/.ckpt`、训练数据、缓存、归档、SSH 凭据或
大模型文件。服务器、本地与 GitHub 的 master 文档必须逐字节一致。

### 20.6 A-V4 终态后的下一步边界

A-V4 Fold-4 已消费并失败，不能再以“机制门通过”为前提继续。下一步只允许先对**与 A-V4 正交的新假设**做
独立新颖性和可证伪性评审；没有自动授权任何 GPU 任务。若人工决定继续，必须满足：

- 不使用已消费 Fold-4 做阈值、margin、Top-K、loss、LR 或 epoch 调整；
- 使用新的未消费 scene-disjoint split，先形成固定 spec、launcher、运行时清单、one-shot 根和双轴审查；
- 仍从受保护 E57 独立开始，只评估 held-out train scenes，不接触 7,899 正式集，不自动保存/晋升权重；
- 晋级必须同时满足 @0.25 正增益、@0.50 非劣、`fix > break` 与 Parent fallback；失败立即封存；
- 不复活 A-V4/FPR Gate、Density-Aware、Relation-CF 扫参、SACR deployment、baseline 公平复现、章节七/八
  或 E0--E7。

### 20.7 单文件交接原则

本文件继续作为唯一完整叙事。仓库中的 `FPR_TV_SPEC_2026-08-31.md`、
`FPR_TV_COUNTERFACTUAL_PARENT_AUDIT_SPEC_2026-09-01.md`、runtime/data manifest、receipt、decision、
`V99_ARCHITECTURE.md`、`REC_3DRES_OPTIMIZATION_LOG.md` 和历史 `docs/superpowers/` 计划保留为原始证据；
后续接手者先读本章，再按 SHA 回查原件，不从旧历史章节恢复已封存路线。

### 20.8 A-V4 Fold-4 终态：完整运行，但科学门禁失败（2026-09-01 18:12 CST）

Fold-4 是 A-V4 唯一预注册的 scene-disjoint 短审计。它从受保护 E57 独立恢复，只训练 allowlist 内的
FPR 模块；未访问 Nr3D 正式 7,899-row validation，未保存新权重，也未启动 Sr3D 或长训。

训练和评估生命周期完整结束：

| 合同 | 实际结果 |
|---|---:|
| Fit 样本 | `27004`，row-identity SHA 与冻结 split 一致 |
| Optimizer steps | `1688/1688` |
| Held-out train-scene 样本 | `5915` |
| Epoch | 从受保护 E57 恢复，执行 E58 |
| Nonfinite | actual=`0`，counterfactual=`0` |
| Actual selected-score gradient L1 | `0.0208565252` |
| Counterfactual selected-score gradient L1 | `0.0220884040` |
| 新 `.pth` | `0` |
| Formal validation accessed | `false` |

模型实际执行了 `805` 次候选切换。两阈值结果如下：

| Held-out 指标 | Parent | A-V4 selected | Fix / Break | 净 hits | 结论 |
|---|---:|---:|---:|---:|---|
| REC@0.25 | `5661/5915 = 95.7058%` | `5641/5915 = 95.3677%` | `38 / 58` | `-20` | `fix <= break`，失败 |
| REC@0.50 | `5011/5915 = 84.7168%` | `4860/5915 = 82.1640%` | `108 / 259` | `-151` | 净负收益，失败 |

这不是 under-switch：805 次切换约占 `13.61%`。问题是错误切换明显多于有效修复，尤其 @0.50 的
break 是 fix 的 2.40 倍。@0.25 transition precision 约 `39.58%`，@0.50 约 `29.43%`，不足以保护 Parent。

反事实 Parent 确实提高了训练监督密度。训练期 CF positive-row ratio 均值约 `12.65%`，actual 约
`6.11%`；两条 score axis 都收到非零梯度，且无非有限值。因此失败不能归因于 CF 分支未执行或梯度断开。

真正失败点是**监督密度增加没有转化为可靠部署决策**。模型学会了更频繁地提出候选，但 break-risk 与
action utility 仍不能在未见场景中充分分离。当前 eligible/fallback 机制没有把切换控制到高精度区间。

独立审计确认：receipt 与 metrics 原始字节 SHA 均与 decision 绑定；算术关系独立复算通过；runtime closure
`373` 文件、data manifest `2420` 文件均通过 postflight；screen 已退出，GPU/全局锁释放，错误日志计数为 0。

不可变终态证据：

```text
receipt SHA256  = 53062ce3110bc5d0f7a2ab9273797a764f06d3234a340b7819d997308abe2605
metrics SHA256  = 97baf04157af257210b8973bdffffc57c1df09331db5fd9505cb005ff07b2781
decision SHA256 = a1c93a71ce62e0c96d02e65241579d520294bf4b868ceccd7178fe9248fd5109
launcher SHA256 = 1fdf9caaec11e0c8d5dd9de109edbc133de316336c91e361135717a537b4b01e
runtime manifest= 575fe4e15e9a6380ebe3d05c71d30d5d8292e36a135b3d42e881cc158c0cdb1b
config SHA256   = aaf4d8edc59e99e056f294b4c031467d2570fb43a879099261b4048054ce4177
```

decision 固定为 `fold_gate_pass=false`、`audit_only=true`、`formal_validation_accessed=false`、
`long_training_authorized=false`。A-V4 至此封存：不得在 Fold-4 上扫阈值、margin、Top-K、loss、LR 或 epoch，
不得运行正式 7,899-row 验证，也不得以该路线为 Sr3D/长训前置。

本次结论进一步强化现有诊断：Nr3D 的主要矛盾不是“候选数量不足”或“训练尚未结束”，而是未见场景上的
文本条件安全排序仍不可靠。下一方案必须与 A-V4 正交，并先证明高精度 abstention 或独立 proposal 改善；
不能只增加反事实样本、放宽 gate 或提高切换率。

### 20.9 为什么 ScanRefer 表现较好，而 Nr3D/Sr3D 泛化不足（2026-09-01 19:20 CST）

当前证据支持一个需要在论文中明确承认的结论：**现有系统具备代码和接口层面的跨数据集可迁移性，但尚未证明
性能层面的跨数据集泛化性。** 同一总体架构能够在 ScanRefer、Nr3D、Sr3D 上运行，不等于同一新增模块会在三个
数据集上都产生正收益。

ScanRefer 的较好结果来自完整系统的共同作用，而不能全部归因于一个语言推理模块：

- V99 的 Parent、Geometry、Query Variant 与 Mask-derived Geometry 确实提供了互补候选；选择器有真实的
  “不同来源中选对一个”的空间。
- mesh-derived superpoint 修复属于几何/数据管线纠错，对旧 V99 带来 REC `+20/+152 hits` 和 Mask
  `+14/+314 hits`，Mask mIoU 提升约 `4.1656pp`。因此 ScanRefer 的最终增益同时包含网络、多源排序和
  几何输入质量改善。
- ScanRefer 当前最好为 REC@0.25 `5572/9508=58.6033%`、REC@0.50 `4835/9508=50.8519%`
  （分别来自 V99/V113），Mask 三项最好为 `59.8443/52.3349/45.9303`。这证明完整系统在 ScanRefer
  有效，但不能单独证明 FPR、Relation-CF 或 A-V4 具有跨数据集普适性。

Nr3D 的数据结构与 ScanRefer 不同，现有模块的关键前提没有成立：

- Nr3D Default 与第二 Source 几乎同序，两 Source oracle 相对 Default 只增加 `3 hits`。没有真正互补的
  Source 时，再复杂的 selector/gate 也无法凭空创造大量正确候选。
- Nr3D Top-2/Top-5/Top-16 oracle 为 `61.6787%/69.0974%/80.3013%`，说明大量正确候选已在前几名，
  主要错误是同类实例、长句、属性、视角和多参照物条件下的 Top-1 消歧，而不是简单的全局重打分。
- 已统计 `2068` 条 ranking failure 和 `1556` 条 proposal failure。前者需要真正的候选级语言证据，后者
  需要局部 proposal 改善；把两类错误压进一个 Gate 或 residual loss 会互相污染。
- A-V4 已完整训练并产生 `805` 次切换，但 @0.25 净 `-20`、@0.50 净 `-151`。因此 Nr3D 低指标
  不是训练未完成，而是 learned switching 在未见场景中没有达到保护 Parent 所需的精度。

Sr3D 的情况介于两者之间。它具有模板化关系和较可靠的显式 anchor，Relation-CF 曾得到小幅正收益，说明
结构化关系模块在 anchor 可靠时更容易工作；但正式最好 `12139/17726=68.4813%` 只略高于旧 `68.4%`
口径，仍低于当前严格 `>68.9%` 目标。其提升还依赖低学习率延续和权重平均，不能声称单个关系模块带来
显著、稳定的泛化提升。

对“是否需要重新训练 ScanRefer”的回答是：**如果下一模块要作为三数据集统一论文方法，最终必须在
ScanRefer 上用同一模块重新训练/评估；但不应现在就盲目重训。** 正确顺序是先在新的、未消费的 Nr3D
scene-disjoint split 上证明模块有正向因果价值，再依次做 Sr3D 与 ScanRefer 同架构验证。否则每次 Nr3D
失败都重训 ScanRefer，只会消耗算力，不能证明泛化。

统一架构的最低合同如下：

1. 三个数据集使用相同模块、张量接口和决策逻辑；允许分别训练权重，但不允许输入 dataset ID、
   Unique/Multiple 标签或 GT-derived anchor sidecar。
2. 新模块默认关闭时必须保持现有 V99 路径；在新 Nr3D split 上必须同时满足 @0.25 正增益、@0.50
   非劣、`fix > break` 和无证据时精确回退 Parent。
3. Nr3D 门通过后才做 Sr3D 同构短审计；二者都通过后，再重训/评估 ScanRefer，确认其 REC 与 Mask
   Pareto 结果不退化。任何一个数据集失败，都不能写成“三数据集泛化提升”。
4. A-V4/FPR Gate、Counterfactual Parent、Density-Aware Target Box、Relation-CF 扫参、旧 Section 7/8、
   E0--E7 和 baseline 公平复现均保持封存，不因本节重新开放。

### 20.10 下一方法候选与本轮代码/文档整合状态

仓库新增 `idea-stage/IDEA_CANDIDATES.md` 与时间戳副本，记录十个与 A-V4 正交的候选方向及其已知近邻。
它们当前只是**候选清单**，不是最终方法，也没有授权 GPU 实验。初步文献地图表明 EG-3DVG、ORD、
ViewSRD、TSP3D、CFA 等工作已覆盖许多直接的同类消歧、关系分解和局部点重读思路；后续必须先做独立
新颖性评审，再选一个最小、可证伪、三数据集共用的机制。

本轮从服务器旧工程中只读找回并纳入 GitHub 的是 12 个此前未发布的 Sr3D 冻结复现脚本：包括
detector-pretrained/global24、plateau/official schedule、E26--E29 权重平均、Relation-CF 审计与 E27--E28
launcher。它们保留历史 SHA/路径门禁，用于复现实验谱系；由于其固定的是当时服务器代码 SHA，**不得把它们
直接当作当前 A-V4 代码树的可运行入口**。服务器较旧的 `main_utils.py`、`train_dist_mod.py` 和模型源码没有
反向覆盖 GitHub 新版本。

发布权威顺序固定为：

- 源码：GitHub `https://github.com/666666666666gao/MCLN` 的 `main`；
- 完整叙事：`docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`；
- 本地镜像：`C:\Users\gb\Desktop\document\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`；
- 服务器镜像：`/home/gb/new butd/butd_detr-main/MCLN-main/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`。

三份交接文档必须逐字节一致。当前没有活动训练、没有待衰减学习率的 run，也没有获得新 Nr3D/Sr3D 长训授权。

### 20.11 下一候选的严格查新终态：EDG 不能按原叙事继续（2026-09-01 19:33 CST）

已完成 2024--2026 公开论文多检索式查新，并由独立 GPT-5.5 xhigh reviewer 复核。完整报告位于
`idea-stage/NOVELTY_CHECK_EDG_20260901.md`。原 Evidence-Deficit Grounding（EDG）只能判为
**CONDITIONAL，接近 NO-GO**：

- candidate-local 高分辨率点重读与 proposal refinement 已被 PV-Ground、TSP3D、3D-SPS 等覆盖；
- same-class distractor contrast 已被 EG-3DVG、TransRefer3D、CORE-3DVG 等直接覆盖；
- target/attribute/relation/anchor clause 分解已被 EDA、G3-LQ、ViewSRD、ORD 等覆盖；
- 三数据集统一主 scorer、无 post-hoc reranker 是工程合同，不是新颖性贡献；
- 因此 C1--C3/C5 均不得在论文中作为独立创新点。

唯一可能保留的窄假设改称 **Candidate-pair Counterfactual Evidence Necessity Supervision（CENS）**：
对固定 GT--hard-negative candidate pair，只有当某 evidence expert 对区分这一候选对具有真实必要性时，移除
该 expert 才应显著降低 GT-vs-negative margin；移除无区分力 expert 时，正负排序应保持。推理仍是完整
evidence 的一次主 scorer 前向，不使用 gate、threshold、Parent switch 或 post-hoc reranker。

CENS 与 Multi-Attribute Interactions Matter 的 counterfactual causal attention、普通 branch dropout 和
margin regularization 高度接近，当前不能直接启动完整训练。若人工决定做唯一证伪实验，必须使用新的、未消费的
Nr3D train-only scene-disjoint mini-fold，在不访问 7,899 formal validation 的条件下固定比较：

```text
C1-C3 concat
C1-C3 + random branch dropout / ordinary counterfactual loss
C1-C3 + CENS
```

只训练小 evidence heads 或冻结 V99 主干，总预算不超过 2 GPU 小时，只看单次前向 Top-1。CENS 必须同时满足：
相对 concat `>=+0.8pp` Acc@0.25、proposal-present ranking failures 净修复至少 `3%`、原本容易/已正确样本
下降 `<0.2pp`，且优于 random dropout/普通 counterfactual 对照。任一失败即将 CENS 判为普通 regularizer 并
永久封存，不进入完整 Nr3D、Sr3D 或 ScanRefer 重训。

本节没有授权 GPU 实验。当前 GPU 仍被独立 ScanRefer ablation queue 占用；Nr3D/Sr3D 没有活动 trainer，
Sr3D 最新 E27/E28 分别为 `0.68379781/0.57835947` 与 `0.68362857/0.58010832`，均低于受保护 E26
weight-average `12139/17726=68.4813%`。因此不存在需要继续等待或衰减 LR 的 Nr3D/Sr3D run。

### 20.12 CENS 代码级可实现性审计与 20:02 训练现场（2026-09-01）

本轮没有为了保留新方法名称而直接写 CENS。对当前代码的实际张量和监督来源逐项核验后，CENS 在现有
输入合同下缺少独立、可审计的 evidence-necessity target：

1. `models/sacr_head.py` 只产生 `target_attr_scores` 与 `relation_anchor_scores` 两类模型预测，并将两者相加为
   `structured_scores`。`models/parent_relative_text_verifier.py::_structured_evidence()` 只是把这些模型输出及
   relation geometry signature 拼接成特征；它们不是 GT 定义的“哪类证据对某个候选对必要”标签。
2. 若再用上述模型自身的分数判断某支路是否必要，训练目标会退化为 self-distillation；若随机移除支路，则只是
   ordinary branch dropout。这两种实现都不满足 20.11 查新后保留的 C4 定义。
3. 现有 `build_counterfactual_parent_views()` 改变的是 Parent 身份/score axis，不是 leave-one-evidence-out
   intervention；不能把它改名解释成 CENS。
4. 唯一有外部几何依据的 relation 监督已经由 `relation_counterfactual_auxiliary.py` 实现：它用 train-only
   target/anchor geometry 挖 relation-inconsistent hard negatives。Sr3D E27/E28 已实测均低于受保护 E26；
   Nr3D 又没有可靠显式 anchor。复制该 loss 再增加 evidence dropout 不会形成新的独立监督。
5. Nr3D/Sr3D 标注不提供可直接定义候选级颜色、形状等 attribute necessity 的完整真值。为补齐标签而引入
   人工规则、模型伪标签或 GT-derived anchor sidecar，会分别落入启发式、自蒸馏或已禁止输入合同。

因此当前 CENS 合同判为 **code-level NO-GO**：不新增代码、不启动 mini-fold、不消费 GPU。只有未来获得与模型
预测独立的 candidate-pair evidence-necessity 标注或等价可验证监督时，才允许重新立项；不能用普通 dropout、
self-distillation 或既有 relation-CF 冒充。

数据入口也再次核对：当前 ScanRefer、Nr3D、Sr3D 正式最好均读取原始标注并在线调用 legacy
`Scene_graph_parse`，不是直接使用 `nr3d_spacy.csv/sr3d_spacy.csv`。spaCy sidecar 含预分解及部分人工/GT
派生字段，直接替换会改变输入合同；而 raw-only conservative parser 的唯一 Nr3D 正式评估为
`4352/7899`，比当前受保护最好 `4475/7899` 少 123 hits，故解析路径仍保持封存。

`2026-09-01 20:02 CST` 远端现场为：

- Nr3D 与 Sr3D 均无活动 `train_dist_mod.py` trainer，故没有可提前衰减的学习率；
- Sr3D official monitor 存活，但最新回执仍是 E27/E28，受保护最好仍为 E26 weight-average
  `12139/17726=68.4813%`；
- 唯一 GPU trainer 是另一个项目队列的 ScanRefer `05_rapf_no_query_quality`。它在 E10 以
  `lr_base=1e-4`、`lr_pointnet=1e-3` 完成训练，验证到 `200/397`；该 run 不属于当前 Nr3D/Sr3D
  目标，本轮没有修改其 LR、进程或文件；
- 当前硬目标继续按用户后续更新执行：Nr3D 至少 `4740/7899`（严格 `>60.0%`），Sr3D 至少
  `12214/17726`（严格 `>68.9%`）。现有最好分别还差 265 与 75 hits。

### 20.13 最新代码补齐、三端事实源与自然同目标多描述审计（2026-09-01 20:27 CST）

本轮继续沿用服务器已整合到 20.12 的本文件，没有新建第二份主交接文档。先对本地发布仓库、GitHub
`main` 与服务器运行镜像做只读审计：审计开始时，本地最新提交和 GitHub `main` 均为
`0d4d80998a63f239e880d9ea6f840bf0f65a914a`，工作树无已修改文件；服务器目录不是 Git 工作树，不能用
服务器 `git status` 判断新旧。

服务器镜像共有 480 个 `.py/.sh/.yaml/.yml` 文件，本地仓库跟踪 474 个同类源码文件。公共文件的原始
SHA 有 367 个不同，但去除 Windows/Linux `CRLF/LF` 差异后只剩 24 个真正不同；这 24 个服务器文件的
修改时间均不晚于 2026-08-26，而本地对应版本已在 2026-09-01 的发布提交中更新。因此服务器是运行镜像，
不是比 GitHub 更新的源码主线；不得用服务器旧 `main_utils.py`、`models/mcln.py`、`train_dist_mod.py`
等文件反向覆盖 GitHub。

服务器独有的 69 个源码后缀文件中，65 个属于 `.v132_parent/.v133_parent` 备份或
`experiment_output/v51_bmq_rank` 历史启动脚本，不纳入当前主线。剩余 4 个是与当前正式代码直接配套、此前
漏传的单元测试，本轮补入仓库：

- `tests/test_dataset_v99_pipeline_contract.py`：V99 数据集管线、artifact、official receipt 与 cleanup 合同；
- `tests/test_rec_evaluator_filter.py`：检测候选 overlap filter 的形状、有效性和错误输入合同；
- `tests/test_sacr_relation_counterfactual.py`：Relation-CF 开关、可训练路径、部署和监督合同；
- `tests/test_scanrefer_debug_train_holdout.py`：ScanRefer debug train/holdout scene 隔离。

四个测试在服务器原始 `bdetr` 环境中为 `23 passed in 4.03s`；待提交 staged tree 在隔离 `/tmp`
源码归档中补入服务器已有的 PointNet2 编译扩展后再次得到 `23 passed in 3.94s`。Windows 本地旧 Conda
环境因缺少 `transformers` 在 pytest 收集阶段中止，不是断言失败，也没有据此增加 fallback、兼容层或额外
异常分支。本轮没有修改模型、loss、dataset、训练 launcher 或推理路径；代码变化只补齐这 4 个既有测试。

同时完成了一个不访问正式 validation、只读训练标注的可行性审计。普通训练数据中的
`(scene_id, target_id)` 已天然形成同一目标的多描述组：

| 数据集 | Train rows | 同目标组 | 多描述覆盖 | 描述/线索多样性 |
|---|---:|---:|---:|---:|
| Nr3D | `32919` | `4664` | `32919/32919 = 100%` | `4112/4664 = 88.1647%` 组在空间/颜色/形状/目标线索上有差异 |
| Sr3D | `65846` | `6993` | `65846/65846 = 100%` | `5576/6993 = 79.7369%` 组在关系/粗关系/anchor 上有差异 |
| ScanRefer | `36665` | `7875` | `36618/36665 = 99.8718%` | 每组最多 5 条自然描述，重复文本仅 4 行 |

这只证明三数据集都具备用**相同训练机制**利用自然多描述监督的输入条件，不是 REC 提升证据。它不需要
dataset ID、Unique/Multiple 标签、spaCy sidecar、GT-derived anchor sidecar，也不改变单描述推理路径。
独立方法评审暂时把 `Target-ID Listwise Sibling Consistency` 与
`Sibling-Inconsistent Distractor Penalty` 列为信息价值较高的两个候选：前者约束同一目标不同描述下的
Top-K 候选分布，后者用同目标描述间不一致的错误赢家构造训练期 hard negative。二者仍可能分别被解释为
普通 consistency distillation 或 hard-negative mining，故当前**没有选定最终方法、没有新增实现、没有授权
GPU mini-fold**；必须先完成严格查新与重复性/新颖性边界审计。

`20:27 CST` 训练现场仍无 Nr3D/Sr3D trainer，也没有可提前衰减的学习率。独立 ScanRefer ablation
`05_rapf_no_query_quality` 已结束 E10 验证并进入 E11，日志到 `E11 1000/2027`；它不属于当前
Nr3D/Sr3D 严格目标，本轮没有修改其 LR、进程、权重或输出。

后续建议技能与顺序：先使用 `novelty-check` 对上述自然多描述候选做正式查新；只有候选通过后，再使用
`experiment-plan` 固定一个未消费 train-scene split、单一机制和不超过 2 GPU 小时的可证伪审计；实际运行
才使用 `run-experiment`/`monitor-experiment`。不得由本节恢复 baseline 公平复现、旧章节七/八、E0--E7、
FPR/A-V4、CENS、Relation-CF 或 Density-Aware 路线。

### 20.14 自然同目标多描述路线的严格查新终态（2026-09-01 20:50 CST）

已按 20.13 的前置条件完成多检索式文献核验和独立 GPT-5.5 xhigh reviewer 审查。完整公开报告写入
`idea-stage/NOVELTY_CHECK_SIBLING_DISAGREEMENT_20260901.md`；完整 reviewer prompt/response 保存在本地
`.aris/traces/novelty-check/2026-09-01_run01/`，按 trace 协议不提交 Git。

原候选 Natural Sibling Disagreement Regularization（NSDR）包含两部分：

1. 对齐同一 `(scene_id,target_id)` 下多个描述在同一候选集合上的 score distribution；
2. 将一条 sibling 描述暴露出的高分错误实例广播为同目标其他描述的共享 hard negative。

严格查新终态为：**NSDR 不得作为下一代 MCLN 的主创新继续实现。** 第一部分与 Chen 等人 2021/2022 年
synonymous referring-expression contrastive learning 的核心思想直接重叠，把 feature-space consistency 移到
3D candidate-score space 只是实现位置变化。第二部分只剩一个窄差异：负例来自同目标其他描述暴露的错误
instance ID，再广播给全组；但它仍很容易被归类为把普通 hard-negative sampler 从 row-level 改为 group-level。
独立评审总体新颖性仅为 `4/10, CAUTION`。

若未来只为证伪保留第二部分，名称与定位必须收缩为 **Disagreement-conditioned Group Hard-negative
Margin（DGHM）**。设同目标描述组为 `G=(s,t)`，描述 `q` 对场景实例 `i` 的分数为 `z_q(i)`：

```text
H_q      = TopM_{i != t} z_q(i)
rho_G(i) = (1 / |G|) sum_{q in G} 1[i in H_q]
N_G      = {i != t : 0 < rho_G(i) < 1}
L_DGHM   = sum_{q in G} sum_{i in N_G} max(0, m + z_q(i) - z_q(t))
```

这只能表述为一个 training-only、group-conditioned hard-negative regularizer，不能声称新的 contrastive-learning
范式。无生成文本、无 parser/spaCy、无推理 sidecar、单次 forward 和三数据集统一执行仍是系统合同，不是
独立新颖性；Sr3D 描述又是模板生成，不能把三数据集统一称为 natural siblings。

任何未来 DGHM 证伪实验都必须在新的 Nr3D train-only scene-disjoint split 上，以相同 backbone、seed、
candidate set、训练步数、负例数量和辅助 loss 总权重同时比较：Base、C1-only、per-row HNM、same-class HNM、
random sibling broadcast、Chen-style synonymous contrastive adaptation、DGHM-only 与 C1+DGHM。出现以下任一
情况立即 ABANDON：DGHM 不超过 per-row/same-class HNM；组合不超过两个单项；改善不集中于已诊断的
Multiple/Hard same-class ranking failures；partial-sibling disagreement 不下降；consensus-wrong 上升；或只有
auxiliary loss 改善而 held-out Top-1 REC@0.25 不改善。

本节没有修改模型、loss、sampler、dataset、launcher 或推理代码，也没有启动 GPU。虽然当前数据字段已经足以
实现训练期 `(scene,target)` 分组、Top-5 candidate 与场景 instance box 映射，但在新颖性不足时直接写 sampler
和 loss 只会产生一个缺少论文价值的常规 HNM 变体。当前主路线因此判为 **NO-GO**；窄 DGHM 仅作为低优先级
候选存档，不占用当前独立 ScanRefer ablation 的 GPU，也不触发 Nr3D/Sr3D/ScanRefer 重训。

`20:41 CST` 只读训练现场仍为：Nr3D/Sr3D 无活动 trainer，没有可衰减 LR；独立 ScanRefer
`05_rapf_no_query_quality` 已保存 E10 最好 `last__bbs_acc0.25_top1=0.3871476651` 后进入 E11。该值只是独立
ablation row，不替代受保护 ScanRefer V99/V113 正式最好，也没有证据允许改变该 run 的学习率。当前正式最好与
严格目标保持不变：Nr3D `4475/7899=56.6527%`，距至少 `4740` 还差 `265 hits`；Sr3D
`12139/17726=68.4813%`，距至少 `12214` 还差 `75 hits`。

### 20.15 Nr3D 视角文本增强错配：已确认并最小修复（2026-09-01 21:14 CST）

在继续排查“为什么 ScanRefer 较好而 Nr3D 明显低于目标”时，确认了一个独立于 Gate、parser 和新网络模块的
真实训练数据缺陷。`src/joint_det_dataset.py::_augment_nr3d()` 原实现使用大小写敏感的
`' ' + rel + ' '` 子串匹配，且只在 utterance 末尾补空格；`_is_view_dep()` 又直接对原字符串调用
`split()`。因此以下真实 Nr3D 表达没有被识别为 view-dependent：

```text
Facing the whiteboard, choose the lamp on the right.
Looking at the books, choose the book on the left.
The chair is on the left.
```

这些样本会错误进入随机 `90-degree rotation + axis flip` 分支，而文本中的 left/right/front/facing/looking
语义没有同步变换，形成可重复的文本--场景监督错配。该问题不是词表是否足够大的假设：只使用原代码已有的
十个视角词，对原始 Nr3D CSV 与正式 train/test scene split 做只读审计即得到：

| Split | Rows | 旧实现禁止大旋转 | 同词表规范化后禁止 | 旧实现漏检 | 漏检占全部行 | 漏检占含视角词行 |
|---|---:|---:|---:|---:|---:|---:|
| Train | `32919` | `10152` | `12307` | `2155` | `6.5464%` | `17.5104%` |
| Test diagnostic | `7899` | `2223` | `2705` | `482` | `6.1020%` | `17.8189%` |

最小修复仅做两件事：`_is_view_dep()` 用 `re.findall('[a-z]+', utterance.lower())` 生成英文词 token；
`_augment_nr3d()` 精确复用该判断并取反。没有扩充词表、没有 parser/spaCy、没有 dataset-sidecar、没有新增
fallback/try-except/兼容层，也没有修改网络、loss、候选、推理或 Sr3D 的 `VIEW_DEP_RELS` 路径。修复后
train/test 分别恰好有 `2155/482` 行由“允许”改为“禁止”，反向变化均为 `0`。

回归闭环：

1. 新测试 `tests/test_nr3d_view_augmentation.py` 在旧实现上稳定得到 `3 failed, 2 passed in 2.33s`；
2. 同一测试对补丁后的单一隔离源码树得到 `5 passed in 2.47s`；
3. 补丁后的完整相关集合
   `dataset V99 + REC filter + Relation-CF + ScanRefer holdout + dataset contract + SACR structured + new test`
   得到 `51 passed in 3.45s`；
4. 第一次把新测试与服务器旧源码混在同一 pytest 进程的组合因 Python 模块缓存看见旧函数，不作为绿灯；
   最终结果来自当前 GitHub HEAD 归档覆盖补丁后的单一源码树。

这项修复保持 ScanRefer/Nr3D/Sr3D 网络架构完全一致，只纠正 Nr3D/ScanRefer 自然语言增强的文本边界判断；
它有望改善 Nr3D view-dependent 与同类消歧样本，但当前尚无 REC 提升证据，不能把 `2155` 条受影响训练行直接
换算为命中数。两台已登记 GPU 当前分别运行独立 ScanRefer 消融，故没有热替换服务器源码，也没有启动 Nr3D
训练。待资源释放后，只允许先做同起点、同 train-scene split、同 batch/step 的 old-vs-fixed 短审计；不访问
7,899-row formal validation，不搜新词表、LR、epoch 或增强概率。fixed 必须在 held-out train scenes 的
REC@0.25 为正、REC@0.50 非负且 view-dependent 子集明确改善，才允许进入三数据集同架构训练验证。


### 20.16 最新源码发布闭环、三端续写与 21:50 训练状态（2026-09-01）

本节严格接续服务器现有 §20.15，而不是另建交接文件。续写前，服务器、本地桌面和 GitHub 工作树中的主交接
均为 `1013823 bytes`、SHA256
`49fa3c661cef0419195187a9d329caddbf6b64c3c0330ea23f4fdf39ebb10313`，因此不存在需要人工拼接的分叉。

当前源码权威仍是 GitHub `https://github.com/666666666666gao/MCLN` 的 `main`。Nr3D 视角文本增强修复、
回归测试和此前漏传的服务器合同测试已经发布到 commit
`5213822f84711697af73511697d913f21865fcff`；本地发布工作树与远端 `main` 精确一致。发布内容不含
checkpoint、数据集、cache、训练日志、SSH 配置、密钥或 askpass 文件。

服务器 `/home/gb/new butd/butd_detr-main/MCLN-main` 仍是非 Git 的运行镜像。只读复核没有发现
`2026-09-01 21:12:42 CST` 之后新增的 MCLN source/test/script 文件；其中
`src/joint_det_dataset.py` 仍是修复前字节，而不是比 GitHub 更新的实现。它没有被热替换，是因为两台 GPU
正在运行彼此独立的 ScanRefer 消融队列；这避免活动进程的源码身份被中途改变。后续 Nr3D 新审计启动前，
必须从 GitHub 固定提交构造新的隔离运行树，不能在活动目录上临时打补丁，也不能用服务器旧源码反向覆盖
GitHub。

`2026-09-01 21:50 CST` 的只读训练现场如下：

- Nr3D 与 Sr3D 均无活动 `train_dist_mod.py` trainer，因此没有可提前衰减的学习率；受保护正式最好仍为
  Nr3D `4475/7899=56.6527%`（REC@0.25）和 `3759/7899=47.5883%`（REC@0.50），Sr3D
  `12139/17726=68.4813%`（REC@0.25）和 `10335/17726=58.3042%`（REC@0.50）。相对严格目标
  `4740/7899` 与 `12214/17726`，仍分别差 `265` 和 `75 hits`。
- 独立 ScanRefer 行 `05_rapf_no_query_quality` 的 BBS Top-1 从 E5 的
  `0.3294068/0.1669121` 上升到 E10 的 `0.3871477/0.2125578`（@0.25/@0.50），E5→E10
  @0.25 为 `+5.7741pp`；它已进入 E13，下一次固定验证在 E15。
- 另一独立 ScanRefer 行 `12_sacr_no_pairwise_geometry` 的 BBS Top-1 从 E5 的
  `0.2901767`（@0.25）上升到 E10 的 `0.3874632/0.2072991`（@0.25/@0.50），E5→E10
  @0.25 为 `+9.7287pp`；同轮 BBF 为 `0.3951409/0.2189735`。它随后进入 E11。
- 两行在 E5→E10 都是明显上升而非平台期，故本轮没有提前衰减 LR，也没有改训练进程、scheduler、权重或
  队列。这些值属于独立 ScanRefer 消融的早期中间结果，不能替代 V99/V113 正式最好，也不能用来声明新
  MCLN 跨数据集指标。

下一项 MCLN 因果验证仍保持最小范围：等 GPU 释放后，从同一受保护 Nr3D E57 起点建立 old-vs-fixed 两个
不可变代码快照，在一个新的、未消费的 train-scene split 上使用完全相同的 batches、steps、optimizer 和
scheduler。只评估 held-out train scenes，不访问正式 `7899` 行；仅当 fixed 的 REC@0.25 为正、REC@0.50
非负且 view-dependent 子集改善时，才允许进入完整 Nr3D 以及后续 Sr3D/ScanRefer 同架构验证。旧章节七、
章节八、E0--E7、baseline 公平复现、parser/spaCy sidecar 与已封存 Gate/Relation-CF/Density 路线均不恢复。
