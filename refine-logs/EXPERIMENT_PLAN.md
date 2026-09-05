> **2026-09-05 生效修订**：下文 full-state/E57→E58 描述保留为历史预注册，已由 master §20.30
> 的 payload 实证纠错覆盖。锁定 SHA 是 eval-only E57/E69 平均权重，无 optimizer/scheduler。
> 当前唯一执行合同是同权重、同新 AdamW、四组 LR 1e-5/1e-4/1e-5/1.25e-5、完整 1611 steps，
> 其余 split 和科学门不变；通过后的下一步按 master §20.18 为 G1 准备。G0 old已完成、fixed运行中，当前进度见master §20.36与Tracker。
> 此holdout只排除新增更新，底层checkpoint已见过完整train；不称为整个系统从未见过的新场景。

> **2026-09-05 执行口径核实（old回执产生后）**：下文2,155行是raw CSV上的旧增强允许→新增强禁止，
> 经当前实际文本清洗与必要parser前缀后是325行（fit253、holdout72）。G0 holdout关闭增强，
> 因而实际训练干预为253行。old允许增强16,432行与源码重算精确一致；fixed预期16,179行待完成核验。
> 该核实未改变G0参数、分组或科学门，详见master §20.36.4；不要再把6.5464%的raw差异当作实际干预比例。

# MCLN Nr3D 视角文本增强修复的配对因果审计计划

**Problem**：已确认旧 Nr3D 视角词匹配漏掉 2,155/32,919 条训练表达，使其在文本语义不变时仍可能接受 90° 旋转和翻转；当前尚无 REC 增益证据。
**Method Thesis**：只纠正这一训练标签错配、保持 V99 网络和推理完全不变，应首先在未访问正式验证集的 scene-disjoint train 审计中改善 view-dependent REC。
**Date**：2026-09-01
**Status**：预注册；未启动 GPU、未访问 7,899-row formal validation。
**范围约束**：不复现 baseline；不恢复旧章节七/八、E0--E7、FPR/A-V4、Relation-CF、Density、parser/spaCy 或新网络模块。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：规范化视角词边界能消除真实训练监督错配，并改善未见 train scenes 的 Nr3D REC | 该缺陷直接破坏 left/right/front/facing 等语言与点云几何的一致性，且影响 6.5464% 训练行 | fixed 相对 old：held-out train scenes 的 REC@0.25 严格增加、REC@0.50 非负，且 view-dependent 子集 REC@0.25 严格增加；两边同起点、同 split、同更新合同 | B0、B1、B2 |

**Anti-claim**：本审计不证明完整 Nr3D 已超过 60.0%，不把 2,155 行换算为 hits，不引入新模型贡献，也不使用正式 validation 选配置。

## Paper Storyline

- 主文可支持：数据增强必须保持空间语言语义等变；最小修复不改变统一网络架构。
- 附录可支持：旧/新谓词逐行审计、split 绑定、view-dependent/view-independent 分组结果。
- 明确切除：词表扩充、增强概率扫描、LR/epoch 搜索、多个 salted split、正式 7,899-row 试跑、Sr3D/ScanRefer 提前重训。

## Experiment Blocks

### B0：输入、代码与 split 闭包

- Claim tested：old 与 fixed 是否只在已确认的视角词谓词上不同。
- Dataset / split / task：Nr3D 原始 train 32,919 rows / 511 scenes；不读取 test/formal validation。
- Compared systems：
  - old：未来审计提交的完整代码树，但 `src/joint_det_dataset.py` 精确替换为 `5213822^` 中的旧版本；
  - fixed：同一完整代码树，使用 `5213822` 及其后续审计代码中的修复版本。
- Frozen split：
  - salt：`MCLN-NR3D-VIEW-AUG-PAIR-V1-20260901`；
  - assignment：`int(sha256(salt + "\0" + scan_id)[:8], 16) % 5`；
  - holdout fold：`0`；
  - salt、hash、fold 在观察 counts/metrics 前固定，不根据样本数或结果重选。
- Frozen census（在上述固定之后只读计算）：
  - fit：`404 scenes / 25,768 rows`，sample identity SHA
    `1cd8a48e901d5e4a67ba82185c576e4639697d86cd26d6665e6de698ea4f16ff`；
  - holdout：`107 scenes / 7,151 rows`，sample identity SHA
    `8ea5315099343e93a0513eecf4ff18c1f62f788153f1aa9c1962f320c8231967`；
  - view-dependent rows：fit/holdout `9,589/2,718`；
  - old allow→fixed block：fit/holdout `1,693/462`；合计仍为已审计的 `2,155`；
  - overlap=0，scene union=511，row union=32,919。
- Required identity checks：
  - 两树所有训练相关文件 SHA 相同，唯一允许不同的是 `src/joint_det_dataset.py`；
  - 该文件 diff 必须精确等于 5213822 中已审计的修复；
  - E57 checkpoint SHA 必须为 `76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`；
  - fit/holdout scene overlap=0，union=511 scenes，row union=32,919，sample identity SHA 固定。
- Metrics：split counts、sample IDs、old/fixed allow→block 差异，正式验证访问计数。
- Success criterion：全部 identity/data gates 通过。
- Failure interpretation：不启动 GPU，先修闭包；不得换 salt/fold。
- Priority：MUST-RUN。

### B1：old 与 fixed 的单轮配对训练

- Claim tested：修复本身能否改善未见 train scenes，而不是依赖新模块。
- Dataset / split / task：B0 固定 fit scenes 训练；固定 holdout train scenes 只评估。
- Setup details：
  - 同一受保护 Nr3D E57 full-state 起点；
  - V99 统一网络、同 seed、B16×A1、同 sampler 顺序、同 batches、同 optimizer/scheduler、同一完整 fit epoch；
  - old 和 fixed 串行运行；不提前停止、不改 LR、不保存可晋升 checkpoint；
  - holdout `augment=False`；禁止读取 `nr3d_test_scans.txt` 和 7,899-row formal receipt；
  - 不增加 fallback、兼容 flag、parser、sidecar 或 dataset-specific 模型输入。
- Decisive metrics：Overall REC@0.25、REC@0.50；view-dependent REC@0.25/0.50。
- Diagnostic metrics：view-independent REC、fix/break、训练中触发大旋转的行数、optimizer steps、sample identity SHA。
- Success criterion：
  1. fixed Overall REC@0.25 hits > old；
  2. fixed Overall REC@0.50 hits >= old；
  3. fixed view-dependent REC@0.25 hits > old；
  4. 两边训练 row identity、batch count、optimizer-step count完全一致；
  5. formal validation access=0、generated persistent weight=0、无 NaN/OOM。
- Failure interpretation：若任一科学门失败，只能保留“修复了标签一致性”的工程结论，不能声称 REC 有效，也不进入完整 Nr3D 训练。
- Priority：MUST-RUN。

### B2：一次性决策与后续授权

- Claim tested：B1 是否足以授权完整同架构验证。
- Compared systems：只比较预注册 old 与 fixed；不加第三个变体。
- Success criterion：机械重算 B1 五项门；不使用小数阈值或事后 margin。
- If PASS：仅授权一次完整 Nr3D fixed 训练；达到严格 `>=4740/7899` 后，才依次做 Sr3D `>=12214/17726` 与 ScanRefer 同架构复核。
- If FAIL：封存本次性能路线；不得扫描 salt、fold、词表、LR、epoch、增强概率或正式 validation。
- Table / figure target：数据管线纠错消融；不作为新网络主创新。
- Priority：MUST-RUN。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| VA0 | 冻结 split 与代码差异 | CPU-only closure | B0 全部门通过 | <30 min CPU | 旧/新树差异超出一个文件则停止 |
| VA1 | 配对 old 角色 | old E57→E58 + heldout train eval | 完整一轮、无正式访问、无持久权重 | 约 1.5–2.5 GPUh | GPU 队列占用；不抢占 |
| VA2 | 配对 fixed 角色 | fixed E57→E58 + 同 heldout eval | 与 old 的 rows/batches/steps 完全一致 | 约 1.5–2.5 GPUh | RNG 分支消费不同；只声称端到端数据管线效应 |
| VA3 | 机械决策 | no-GPU comparator | C1 全部门同时通过 | <5 min CPU | 任一负门即封存 |
| VA4 | 完整 Nr3D | 仅 VA3 PASS 后 | 正式 `>=4740/7899` | 条件运行 | 不以中间 train 审计冒充正式结果 |

## Compute and Data Budget

- 必需预算：约 3–5 A100 GPU-hours，old/fixed 串行；当前两张 GPU 均被独立 ScanRefer 队列占用，不抢占。
- 数据：仅 Nr3D train；固定 split 已记录为 25,768 fit / 7,151 holdout rows，未据此更改 salt/fold。
- 最大瓶颈：6.5% 行级纠错能否在一个完整 fit epoch 内形成可检测的 held-out Top-1 改善。

## Risks and Mitigations

- 随机流会因 old/fixed 的 rotate 分支消费不同而分叉：这是实际端到端数据管线效应；计划只要求相同 seed、sampler、batch identity 和步数，不虚假声称逐点噪声 bit-exact。
- E57 已在完整 train 上训练：因此 B1 只证明继续训练时的增量因果效应，不声称从零泛化；完整正式训练必须等 B1 PASS。
- 修复可能只改善 view-dependent 子集而整体信号过小：仍要求 Overall @0.25 正增益，避免用子群结果包装整体失败。

## Final Checklist

- [x] 单一主 claim 与 anti-claim 已冻结
- [x] salt、hash、fold 在 counts/metrics 前冻结
- [x] old/fixed 唯一允许源码差异已定义
- [x] baseline 公平复现和旧路线明确排除
- [x] B0 split counts / identity SHA 闭包
- [ ] B0 old/fixed source-tree 闭包
- [ ] VA1 old 完成
- [ ] VA2 fixed 完成
- [ ] VA3 决策完成
- [ ] 条件式完整 Nr3D/Sr3D/ScanRefer 验证
