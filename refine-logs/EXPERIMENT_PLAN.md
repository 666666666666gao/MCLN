# MCLN V132 后续与 V133 关系感知 Query Ranking 实验计划

**Problem**：ScanRefer 双阶段模型已有充足 Top-K 候选，但 Top-1 query 排序仍未达到 REC@0.25 ≥ 59%、REC@0.50 ≥ 49%。
**Method Thesis**：在保持父模型 box 与 mask 表征不变的前提下，用可跨 ScanRefer/Nr3D/Sr3D 复用的 target/attribute/relation-anchor 结构直接校准 query 排序，比继续扰动整条 final-query 表征更可靠。
**Date**：2026-08-15
**约束**：单张 A100；V132 完成前不改源码、不启动第二个 GPU 任务；validation 不用于调参；V109 永久保留。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：结构化 target-anchor ranking 能把已有 Top-K 候选转成更高 Top-1 REC | V132 epoch1 的 query Top-5 已达 66.113%/57.446%，但 Top-1 仅 57.888%/46.182% | ScanRefer 9,508-row official 同一 checkpoint 同时达到 ≥5610/≥4659 REC hits，且无 validation 调参 | B0–B3 |
| C2：score-only、parse-aware fallback 能保持 box/mask 表征并具备跨数据集复用性 | V132 full-query residual 同时扰动 box/REC/mask；旧复杂 safety/reranker 在 full validation 泛化失败 | step-0 identity、无结构行 exact fallback、候选 box/mask tensor 不变；Nr3D/Sr3D 同 schema smoke 可运行 | B1、B2、B4 |

**Anti-claim**：收益不是更多候选、validation 阈值搜索、ScanRefer 专用规则或更大训练预算造成的。

## Paper Storyline

- Main paper must prove：Top-K 召回充分时，结构感知的 query scoring 能改善 Top-1，并保持候选 box/mask 本身不变。
- Appendix can support：target/attribute-only 删除实验、parser coverage、Nr3D/Sr3D schema/transfer smoke。
- Experiments intentionally cut：继续扫描 source 阈值、V80–V131 safety-loss 组合、再次扩大 full-query residual、validation 上选 margin。

## Experiment Blocks

### B0：完成 V132 冻结基线

- Claim tested：V132 的 full-query residual 是否随 epoch2–4 反转 epoch1 的轻微退化。
- Why：不能用 epoch1 提前否定已预注册的四轮网络实验。
- Dataset / split / task：ScanRefer train 36,665；official val 9,508；双阶段 REC/RES。
- Compared systems：epoch71 parent、V99/V113 official best、V132 epoch1–4。
- Metrics：REC overall/Unique/Multiple @.25/.50；Mask overall/Unique/Multiple @.25/.50；mIoU；Top-K/oracle；residual magnitude。
- Setup：完全沿用已启动的 V132 配置，单卡、4 epochs、每轮固定 validation。
- Success criterion：任一网络 checkpoint 先刷新网络 REC；只有刷新时才运行一次冻结 V99 compatibility policy。
- Failure interpretation：四轮均不胜说明全 query residual 不是正确优化轴。
- Table target：主结果表的 V132 行。
- Priority：MUST-RUN。

### B1：V133 结构与 identity gate

- Claim tested：score-only SACR-Lite 可在不改变候选 box/mask 的情况下提供结构化 query score。
- Why：隔离 V132 的表示扰动与旧复杂 reranker 的 policy overfit。
- Dataset / split / task：合成边界样例 + ScanRefer held-train 128 rows/scene-disjoint 120 scenes。
- Compared systems：父模型 fixed-default、V132 best、V133 zero-step、V133 两轮 smoke。
- Metrics：step-0 score/box/mask bit-exact；structured/fallback coverage；REC fix/break；Mask；梯度与 residual 范围。
- Setup：复用 `StructuredSlotBuilder`/`SACRHead` 的 target、attribute、relation-anchor 几何；新增独立 bounded score residual。父 query、box head、mask head、selector 全冻结；无有效结构行 exact fallback；只训练该 score head。监督为当前 train batch 中所有 queries 的连续 3D IoU listwise target，ScanRefer 有 mask GT 时仅加固定 0.25 权重的 mask quality，其他数据集自然退化为 box-only。
- Success criterion：contract/finite/identity 全过；held smoke 两阈值均 `fix >= break`，Mask 不发生 tensor-level 改写。
- Failure interpretation：若 relation coverage/梯度无效，拒绝 formal；不通过 validation 调 margin。
- Table target：结构安全门表。
- Priority：MUST-RUN。

### B2：V133 ScanRefer 正式训练与唯一判断

- Claim tested：关系感知 query scoring 是否将 Top-K headroom 转成 ScanRefer Top-1。
- Why：这是主目标的决定性证据。
- Dataset / split / task：完整 ScanRefer train/official val，双阶段。
- Compared systems：epoch71 fixed-default、V99、V113、V132 best、V133。
- Metrics：首要 REC @.25/.50 raw hits；其次 Unique/Multiple、Mask、mIoU、Top-K、fix/break、parser coverage。
- Setup：单卡，固定 seed0、最多 4 epochs、每轮一次完整 9,508 validation；不做 lr/margin/source grid。只有 score head 可训练；checkpoint retention 按五项指标硬链接去重。
- Success criterion：同一 V133 checkpoint 或其一次冻结 V99 compatibility 结果达到 `REC hits >=5610/4659`；Mask 至少保持用户 baseline 58.70%/50.70%/44.72%，并报告相对 V99 的差值。
- Failure interpretation：若 Top-K 不变而 Top-1 仍不升，结构 slot/监督缺少判别信息；停止 SACR-Lite，不做阈值补救。
- Table target：主 ScanRefer 结果表。
- Priority：MUST-RUN。

### B3：最小删除实验

- Claim tested：relation-anchor 分支而非额外参数本身带来提升。
- Why：排除“只是更大 MLP”的解释。
- Dataset / split / task：仅在 B2 达标后，固定 ScanRefer protocol。
- Compared systems：完整 V133；删除 relation-anchor、只留 target/attribute；等参数无结构 MLP。
- Metrics：REC、Mask、mIoU、结构覆盖和参数量。
- Setup：相同 epoch/seed/预算，不新增超参。
- Success criterion：完整 V133 优于两个对照，或明确 relation 分支只在 Multiple 子集有效。
- Failure interpretation：若无差异，采用更简单的 target/attribute head。
- Table target：主文或附录消融。
- Priority：MUST-RUN IF B2 PASS。

### B4：Nr3D/Sr3D 迁移接口与 smoke

- Claim tested：模块不是 ScanRefer 专用后处理。
- Why：支撑跨数据集泛化定位。
- Dataset / split / task：Nr3D、Sr3D 的现有 train/val schema；先做构建与小样本 smoke。
- Compared systems：父模型与 V133；不迁移 ScanRefer 阈值。
- Metrics：加载/coverage/finite、REC 主指标、fallback 比例。
- Setup：同一模块和连续 IoU 监督；数据集仅提供 box 标签时自动关闭 mask term。
- Success criterion：同一代码路径可训练、可评估且无数据集分支阈值；若预算允许再做完整评测。
- Failure interpretation：接口不通则 C2 未成立，先修通再宣称泛化。
- Table target：附录 transfer 表。
- Priority：MUST-RUN interface；full evaluation NICE-TO-HAVE。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 收完 V132 | B0 epoch2–4 | 4/4 receipt、9508 rows、无错误 | 约 3 GPU-hours remaining | 磁盘不足；每轮检查物理 inode |
| M1 | V133 静态/identity | B1 contract tests | zero-step/fallback/box-mask identity 全过 | 0 GPU，约 30–60 min | 运行中源码漂移；必须等 V132 结束后编辑 |
| M2 | scene-disjoint smoke | B1 smoke | fix≥break、finite、无 mask tensor 改写 | <0.2 GPU-hour | 小样本假阳性；只作晋级门 |
| M3 | 主正式实验 | B2 | 同一结果达到 5610/4659 hits | 约 4 GPU-hours | ranking 泛化失败；无 validation 调参 |
| M4 | 机制与迁移 | B3、B4 | 删除实验和跨数据集接口成立 | 1–5 GPU-hours | parser coverage 不一致；启用 exact fallback |

## Compute and Data Budget

- 单张 A100，所有 GPU 任务串行；V132 未结束前不启动 V133。
- V133 必需阶段估计约 4.2 GPU-hours；只有主结果通过才运行消融和完整 transfer。
- 数据准备：复用现有 structured slots、box/mask GT 和 scene-disjoint smoke；不访问额外 validation 标签做校准。
- 最大瓶颈：query ranking 的 train-to-validation 泛化，而非候选召回。

## Risks and Mitigations

- 结构 parser 覆盖不足：无结构行严格退回父 default score，并单独报告 coverage。
- score head 再次过拟合：只使用连续 IoU listwise 监督、一个固定配置、scene-disjoint smoke；不叠加旧 safety policy。
- Mask 因 query 选择改变而下降：候选 mask tensor 完全冻结；固定可选 mask-quality 项和 baseline/V99 双重报告。
- 磁盘不足：active run 只保留 metric-best/latest 硬链接；清理前核对 inode、SHA、恢复依赖并永久保护 V109。

## Final Checklist

- [x] Main result、novelty、simplicity 与 transfer evidence 已映射
- [x] V132 与 V133 的变量隔离
- [x] 单 GPU 串行和 stop/go gate 明确
- [ ] V132 四轮审计完成
- [ ] V133 identity/smoke 通过
- [ ] V133 ScanRefer 正式达标
- [ ] 非劣权重与 V109 保留审计完成

