# 固定视觉记忆 E：训练方案、实现与启动证据审计

日期：2026-09-17。审计人：gpt-6-astra / max，fresh native Codex reviewer `/root/fixed_memory_training_audit`；模型与推理配置由主执行器确认。`review_independence: same-family`，`acceptance_status: provisional`。

**总体结论：WARN / provisional。修订后的方案及源码未发现静态启动阻塞。初审 17:41:59 快照只有启动和参数计数；17:49:00 容量补充确认真实 E batch8 通过，17:52:17 不可变快照进一步记录 initial 进度 2048/6887，详见第 11–12 节。完整 initial、训练终态、实际 terminal restore 和正式结果仍待完成。不得将 D 接口测试、CPU 序列化夹具或进程启动替代 E 的对应运行证据。**

审计只读检查本地源码与已归档证据，没有读取凭据、连接远端、启动模型、执行优化器或删除 checkpoint。除本报告和主执行器明确授权的审计追踪文本外，未修改其他文件。第 1–9 节保留初审 **2026-09-17 17:41:59 +08:00** 快照及当时待办状态；第 11 节保留 **17:49:00** 容量补充，第 12 节绑定 **17:52:17** 不可变快照，第 10 节机器摘要反映最新范围。均不代表审计人独立实时查询了远端。

## 1. 阻塞项与处理状态

### B1：原统一 256 MiB 导出预算不足 — 源码修复已核验

原预算把训练导出和双臂正式验证混用同一个 256 MiB 额度。仅 float32 数组就需要：

- 训练 initial + terminal：`2 × 6887 × 256 × (6 + 2) × 4 = 112,836,608` 字节。
- 正式 published_parent + fit_terminal：`2 × 9508 × 256 × (6 + 2) × 4 = 155,779,072` 字节。
- 合计 `268,615,680` 字节，超过 `256 MiB = 268,435,456` 字节，还未计入 NPY 头、JSONL、日志和协议文件。

证据：`scripts/run_pvground_scanrefer_fixed_memory.py:273-274`；`scripts/evaluate_pvground_scanrefer_task_observation.py:265-266,330-331`。这一问题已在启动前报告主执行器。

已部署补丁将两阶段容量分开：训练保留两份 delta、256 MiB 导出和 768 MiB 余量；独立 endpoint 审计成功后，仅回收 E 自己的 `latest.pth`；正式推理前要求至少 1 GiB 空闲。实际 `audit_queue.py:38-52` 核验审计状态、固定视觉状态、训练 receipt SHA、terminal SHA，限制解析后删除路径的父目录为 E 训练目录，先写路径/SHA/字节计划，再删除并写完成收据。实际 `formal_queue.py:23-37` 在审计、非退化门槛、实际终点 CPU strict restore 后检查空闲容量，再进入 GPU 锁。

证据目录：`refine-logs/pvground_scanrefer_endpoint_audit_20260917_fixed_memory_v1/` 与 `refine-logs/pvground_scanrefer_formal_20260917_fixed_memory_v1/`。清理实际执行、清理收据与正式前余量仍待终态；当前结论是**已部署源码闭合原预算缺口**，不是声称已经回收了空间。

### B2：实际新模式容量与终态证据 — 尚未产生，不作 PASS

本地 E 归档已有 `memory_policy.json`，截至上述快照没有 `capacity.json`、`initial/receipt.json`、`terminal.pth`、训练终态 receipt 或实际 `terminal_restore.json`。因此“真实 batch8 完整 loss/backward 已通过”“固定骨干训练后无漂移”“3723 步完成”“正式结果满足晋级”均未获支持。这不构成源码启动阻塞，但阻止对应完成性和性能结论。

## 2. 唯一训练机制变量、参数计数与优化器

**PASS（源码范围核对）；运行时终态验证待完成。** E driver 与当前 D driver 的差异仅为：记录原 D checkpoint 参数键集合；完整冻结 `backbone_net`；训练模式下单独将骨干切为 eval；检查和记录固定 state；保存兼容 D 的全 delta 键集合。未见数据、增强、loss、reader 架构、LR、训练行数、输出规则或终点选择的额外变化。

D 当前 driver 与 D 已完成归档 `train.py` 均为 SHA256 `819e723d2dac40a191f7e057a9ba1b340519454c4f241f2458a40b431376a872`。E 已归档 `train.py` 与本地 E driver 字节相同。三个 reader 模块与 D spec 的绑定哈希一致；实际 `pv_ground.py`、`main_utils.py`、`models/losses.py` 缓存哈希均与 D/E 共享的 `source_port.json` 一致。

E 在安装 reader 及拷贝初始新 state 后保存原 D 参数集合，先核验 D 的 820 个 trainable tensors / 28,883,227 个参数元素，然后对整个骨干执行 `requires_grad_(False)`；每次进入训练通过 `model.train(); model.backbone_net.eval()`。没有把父值 BN 单独混回一个已训练失败终点。证据：`scripts/run_pvground_scanrefer_fixed_memory.py:111-161,248-255,338-345`。

实际 E `memory_policy.json` 与静态预期一致：

| 项目 | D | E | 证据性质 |
|---|---:|---:|---|
| 可训练参数张量 | 820 | 718 | D 接口收据；E 启动时实际枚举 |
| 可训练参数元素 | 28,883,227 | 25,397,851 | 同上 |
| 冻结参数张量 | 199 | 301 | 同上 |
| 冻结参数元素 | 124,645,632 | 128,131,008 | D 值由 E 总量及骨干差额推导 |
| 本次固定的骨干参数张量 | — | 102 | D 参数分组及接口实际变更参数列表 |
| 本次固定的骨干参数元素 | — | 3,485,376 | D/E 可训练参数元素之差 |
| 固定视觉 state 张量（参数 + buffer） | — | 204 | E 实际 memory policy；D 的 102 参数 + 102 buffers |

证据：`refine-logs/pvground_task_observation_interface_20260917_v1/results/receipt.json:1062-1063,1620-1627`；`refine-logs/pvground_parameter_direction_20260917_v1/parameter_groups.json:3-106,888-889`；E `memory_policy.json`。参数计数是初始化后的实际枚举，尚不是训练成功证据。

真实 `main_utils.py` 的优化器在 `frozen=False, small_lr=False` 分支中，对普通参数、骨干参数、文本参数三组都按 `p.requires_grad` 过滤。E 将骨干和原文本参数排除；骨干组与文本组为空，AdamW 不为它们建立状态。仍是 LR `1e-5`、weight decay `0.0005`、clip `0.1`，没有新增 scheduler、降 LR 分支或替代 loss。证据：实际缓存 `D:/Program Files/UserCache/gb/codex/tmp/pvg_loss_20260917/main_utils.py:339-366`；driver `148-158,228-246`。原模型只逐参数冻结文本编码器，E 保持其原有规则：实际缓存 `pvg_parameter_20260917/pv_ground.py:173-176`。

此实验将“骨干可学习性”和“骨干 train/eval 行为”一起改变；冻结后全局梯度裁剪所见梯度集合也随之改变。结果只能评价这个完整训练策略，不能分别归因于 BN、骨干权重、某个 reader 或新架构。方案正文已明确这一边界。

## 3. 新模式容量、buffer 不漂移与初始恢复

**PASS（检查实现）；实际容量 PENDING。** 容量检查在 E 完整冻结策略之后运行，使用真实 fit DataLoader 的 batch8、在线点云/检测框增强、训练 DataProcessor、完整原生 criterion 与 backward。`update=False` 时不执行 optimizer step。检查所有骨干模块处于 eval、骨干参数无 grad、204 个视觉 state 与父初值相等，然后才记录容量 PASS。证据：driver `205-264`。

容量后严格恢复完整模型初始 state，清除全部梯度，并重新设置 Python / NumPy / Torch / CUDA RNG。初始评估完成后再次核对全部 state 与初值一致，并在固定训练前重置 RNG。冻结动作位于 reader 初始化之后，不增加此前的随机操作；共享模型和模块哈希、初始化顺序及 seed2027 与 D 相同。证据：driver `77-84,111-147,261-269,338-341`。

每步检查骨干 eval 与无梯度；每次保存和训练最终 receipt 前比较完整视觉 state；所有 frozen 参数在 terminal 保存前与初值比较。独立 endpoint auditor 会重新加载父 checkpoint 和实际 terminal 的字节，对父 checkpoint 的每个 `backbone_net.*` state 做 `torch.equal`，没有仅采信训练 receipt 的布尔值。证据：driver `239-240,344-368,390-392`；`scripts/audit_pvground_fixed_memory.py:95-105`。

不漂移保证针对模型 state。在线增强输入和训练阶段的数据预处理仍按 D 执行，不能把“固定记忆函数”解释成每次输入特征完全相同。容量前初始化已成功、D 历史 batch8 容量通过、E CPU restore 夹具通过，都不能替代本轮 E 真实训练容量回执。

## 4. Checkpoint 格式与 strict restore

**PASS（键集合与准备夹具）；实际终点恢复 PENDING。** E 的保存集合使用冻结前的 D 参数键集合，加全部原有持久 buffer，保留 820 个原 D 可训练参数键及 252 个 buffer，共 1072 个 delta state。模型含父 1234 个 state 加 37 个 reader state，共 1271。即使 102 个骨干参数在 E 不再训练，它们仍保存在 delta 中；只有 optimizer 状态减少。证据：driver `134-136,342-354`。

既有 `terminal_state()` 按未冻结的完整 D 模型收集 expected delta 集合，要求键集合相等、形状/dtype 一致、数值有限，再 strict load。因此 E 保存集合与既有格式相符。证据：`scripts/evaluate_pvground_scanrefer_task_observation.py:53-88`。E 并未试图让一个只含 718 个可训练参数的精简 delta 静默补齐缺失骨干。

本轮正式准备已实际运行 CPU 序列化夹具，收据为 `status=pass`、1072 delta / 1271 expanded / 37 added tensors、缺失 reader state 被拒绝、CUDA 未初始化、model forwards=0、optimizer steps=0。其 scope 明确为 `in-memory serialization fixture changing all 37 reader tensors; not trained weights`，terminal SHA 为 null。它证明**当前正式恢复代码的格式接口**，没有加载 E 终点。

实际正式队列在进入 GPU 推理前调用 `restore_probe.py --terminal`。该路径额外绑定 3723 步、完整 fit 行集合、parent/spec/module/source-port 哈希和 terminal 字节 SHA。证据：`scripts/check_pvground_task_observation_restore.py:59-85`；实际 E formal queue `34-37`。没有执行 optimizer resume；当前 optimizer 参数组改变并不影响推理恢复，但本报告不宣称可用 D optimizer 格式恢复 E 训练。

## 5. 初值、RNG、固定训练预算和 D/E 比较

**PASS（协议和代码绑定）；实测初始一致性/完成预算 PENDING。** Driver 断言父 checkpoint SHA、seed2027、batch8、单遍和两个 LR；scene 前缀哈希划分形成 29,778 fit / 6,887 holdout，fit 与 holdout 场景、行均不相交。严格 DataLoader 顺序和固定 generator 保持既定合同。最后一批应为 2 行，3722 个满 batch 加 1 个末批，共 3723 步。

证据：driver `40-71,164-207,356-380`；auditor `80-87,107-137`。训练终点独立审计按训练日志检查逐步序号、每批行数、所有 fit 行恰好一次、holdout 零使用及 finite loss/gradient 记录。初始/终态完整评估各一次，未见按 loss 提前选择权重或更换终点的分支。

D/E comparer 绑定共同父权重、seed、batch、fit passes、LR、input manifest、env 和 source port，按逐行身份、point hash、GT 做匹配。终点比较另要求 3723 步逐批 row order 相同，报告各自起点、终点差、修复/破坏和变化量之差。初始输出字节是否相同会真实记录，代码没有预填相等。证据：`scripts/compare_pvground_fixed_memory_control.py:18-54,73-105`。比较队列只读取完整独立审计后的输出，不运行模型、不更新优化器、不修改晋级门槛。

当前唯一已完成 D 对照的 bbs 起点为 `6147/5549`，终点为 `6136/5547`，净变化 `-11/-2`；来自指定 D `receipt.json:19-25,50-56,76-85`。这些是 6887 模块留出计数，不能当成正式 9508 验证或 E 的结果。D 的已完成 status、3723 步和负结果均保留。

## 6. Experiment-audit A–F

| 检查 | 状态 | 证据与解释 |
|---|---|---|
| A. GT 来源 | PASS（源码） | 数据集从 ScanRefer annotation 的 object_id 与 token 建立样本；GT boxes 来自对应 ScanNet 实例框，GT masks 来自实例点集合。driver 从 batch 的 center_label / size_gts / gt_masks 取 GT。不是用预测生成评价 GT。 |
| B. 分数归一化 | PASS | REC 是 IoU 阈值命中计数；Mask mIoU 是 GT 交并比之和除以固定样本数。预测 softmax 只用于原生候选排序，未用模型自身最大/均值去归一化最终性能。 |
| C. 结果文件存在性 | WARN | D 完整负结果存在；E 当前只有启动、计数与正式准备回执，没有 capacity/initial/terminal/正式结果。不支持 E 性能结论。 |
| D. 指标调用链 | PASS（源码）；运行覆盖待终态 | evaluate 调用 native loss 和官方 GroundingEvaluator，并将逐行框/分数/IoU 写入回执；auditor 重新选框、重算 IoU、核对逐行命中及 transitions。比较函数由队列明确调用，未见新增死指标被冒充结果。 |
| E. 评价范围 | WARN / 明确限定 | 单 seed、单 pass、预训练已见场景的模块 holdout；不能声称新场景泛化、统计显著性或多次独立复现实验。 |
| F. 类型 | real_gt | REC/Mask 评价使用真实数据集 GT；原有 Mask corresponding consistency 为训练中的自监督辅助项；CPU 恢复为 serialization fixture，不是性能评价。 |

GT 源码证据：`src/joint_det_dataset.py:633-641,1085-1119,1387-1392`。本地该文件 SHA `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d` 与实际 input manifest 所绑 source manifest 完全一致；本地 source manifest SHA 为 `75cd5f87a8e715c15b458ed1964c2e6f1d35046bb6beffca1b3388d5903bbe36`，与 D/E 指定数据源相符。

Loss 证据：实际 `main_utils.py:268-280` 保留 boxes/labels/masks/contrastive_align；实际 `models/losses.py:856-886` 从真实 batch 建立目标，`:574-627` 保留 GT mask 与原预测对应一致性项，`:943-955` 保留原权重。E 未新增代理 GT；必须继续把辅助自一致性损失与正式 real_gt 指标区分。

评价证据：driver `278-330`；auditor `31-73,117-145`。独立审计会从已导出的全部 256 个 raw boxes 和 native scores 核对 REC；**Mask 审计只重计导出的每行 Mask IoU，没有重新加载二值预测 mask 和 GT mask**，脚本明确在输出 `mask_audit_scope` 声明这一限制。raw256 oracle 只作 GT 上界诊断，comparer 明示不是合法过滤后的召回或可部署选择结果。

## 7. 正式验证与晋级路径

**PASS（已部署队列门槛）；正式结果 PENDING。** 真实 formal queue 等待本轮原审计 controller 退出，要求训练和审计退出码为 0、`integrity_pass`、`fixed_visual_memory_verified`、正确 receipt SHA、bbs 对自身 initial 在 0.25 与 0.50 两阈值均不退化。失败保留 `skipped_primary_rec_regression`，formal rows=0。成功路径必须先通过实际 E terminal CPU restore 和磁盘检查，再锁 GPU 进行固定双臂验证。

每臂固定 9508 行，主输出 bbs；parent 关闭 reader，terminal 开启已训练 reader；bbf 只作诊断。既有正式 evaluator 固定 V99 REC 下界 `5572/4797`、Mask `58.70/50.70/44.72`；严格绑定数据身份、完整每臂行数与 parent/terminal 权重哈希。正式 evaluator 不自行修改阈值或使用伸展目标替换下界。证据：`scripts/evaluate_pvground_scanrefer_task_observation.py:43-50,99-147,191-241,255-349`。

formal queue 在评估返回 0 后运行独立正式 audit；生成的 formal auditor 绑定 source/observation/task modules、两臂启用状态、source port 和实际 terminal_restore 回执。准备时 6887 行旧输出 CPU 重计只证明当前 recount 路径可用，其 `formal_rows=0`；没有运行新正式推理。证据：E formal `preparation_receipt.json`、`restore_preparation.json`、`audit.py`、`formal_queue.py:43-48`。

本轮没有预先启动 Nr3D/Sr3D 任务。Nr/Sr REC 只能在真实正式回执和独立正式 audit 确认全部既定门槛通过后决定；训练接口、6887 holdout 或正式 queue 启动不构成晋级。

## 8. 实际部署、容量和进程证据

本地已归档四个启动回执，且四份 spec 声明的全部文件共 24 项均完成 SHA 校验，所有 Python 字节均可 compile；E train.py 与本地已审 driver 字节完全相同。生成后 actual audit/formal/comparison 队列哈希与审计人在本地只读重构的字节相同。

| 组件 | 启动回执中的原 controller PID | 回执时间（CST） | 当前证据含义 |
|---|---:|---|---|
| E training | 8666 | 17:40:55.204078 | 已启动；17:41:59 快照在文本解耦阶段 |
| E endpoint audit | 8673 | 17:40:57.622993 | 已启动等待；首次检查 18:00:55，之后 300 秒 |
| D/E comparison | 8866 | 17:41:38.792513 | 已启动等待；初始等待 audit，terminal 首查约 20:41:36 |
| E formal continuation | 8872 | 17:41:43.741175 | CPU 准备通过后已排队；首查约 20:41:11，之后 300 秒 |

E spec 记录启动前空闲 `1,800,552,448` 字节，超过训练明确阈值 `1,758,340,958` 字节，差额 `42,211,490` 字节。17:41:59 观察快照记录空闲 `1,800,208,384` 字节及 GPU 进程占用 `2426 MiB`；这些只属于该快照，不能用来替代 formal 前的重查。单份 delta 采用 D `342,299,567` 字节上界；E 将少保存骨干 AdamW 状态，因此该上界是保守估算，最终文件大小仍应以实际文件为准。

当前严格区分：已验证源码与部署字节、实际参数枚举、CPU 夹具、进程快照；尚未验证新模式 capacity、完整初始重计、完整训练预算、terminal 不漂移、实际 terminal strict restore、formal 指标。不能把正在运行写成训练完成。

## 9. 必须在结果报告前闭合的事项

1. 收集 E `capacity.json`：必须来自当前 driver、真实 batch8 完整 loss/backward、optimizer_steps=0、204 visual state 无漂移；随后确认完整 initial receipt 与独立 initial audit。
2. 收集 E terminal、训练 receipt/log、原 controller 退出码及 endpoint audit：3723 步、29778 行恰好一次、相同逐批行序、204 visual state 与父权重实际字节相等；保留完整两阈值修复/破坏与负结果。
3. 核验 D/E initial 和 terminal 比较输出，明确实际起点是否字节相同。若不同，原样报告差异，不将描述性 difference-in-changes 当作独立机制因果证明。
4. 核验 `cleanup_planned.json` / `cleanup_receipt.json` 对应 E 自己的 superseded latest；保留 terminal。若候选符合既定门槛，收集 actual `terminal_restore.json`、formal 前容量通过、9508/arm 完整正式输出及独立 formal audit；否则保留跳过决定。
5. 最终叙述限于单 seed 的训练策略对照；不得宣称新架构贡献、BN 单独贡献、统计显著性或预训练固定策略普遍有效/无效。

## 10. 审计输入哈希与归属

本报告只对下列字节及上述实际归档快照有效；后续改动需明确复核范围。没有生成跨模型族接受结论。

**哈希范围：** 下表 7 个 `scripts/` 项目绑定审计时的原始文件字节，不能直接拿 Git 后续换行规范化后的工作树 SHA 冒充同一字节身份。已审字节的精确副本保存在 `refine-logs/pvground_scanrefer_finetune_20260917_fixed_memory_v1/audited_sources/`，7 份文件均已独立按 `manifest.json` 核对 SHA 并 compile；该 manifest 的 SHA256 为 `de8bc31adf215dbbf239404506e9ad575d190d302491a97595920cb89f57ffd0`。实际已部署 driver/queue 继续受各自不可变 spec 绑定，与发布时的本地换行处理分开。

观察脚本初审 SHA 为 `41379f5de38d42aeb71f497930439a4e08b3a3b05cee8d709e91a5906d6007f0`；容量归档时增加了 D/E comparison 目录，以及 cleanup/比较输出收集。此最终版本已重新逐行审阅，改动限于观察范围，没有修改训练或晋级规则，下表及精确归档使用最终补审 SHA `f5c399d5…`。其他 6 份脚本的审计字节未变。

| 文件 | SHA256 |
|---|---|
| `docs/PVG_FIXED_VISUAL_MEMORY_CONTROL_2026-09-17.md` | `d19945c3ad9410f3d3f5b07f08f61344ef18f330781a691aa4d1410fd046fd93` |
| `scripts/run_pvground_scanrefer_fixed_memory.py` | `f03aa1fcb6c4e495c4cff191b3189de2b17644396a684839a928a3c3e7402925` |
| `scripts/launch_pvground_fixed_memory_training.py` | `68813563ebd8a176caa90c396b3f6b42faad71f297c2bec03373e537d5ccc095` |
| `scripts/audit_pvground_fixed_memory.py` | `a7c8f915a6ce13ca16707cd47214a59977d1349de71f2844b8bff9ae2a7ca98f` |
| `scripts/compare_pvground_fixed_memory_control.py` | `9c83a59ac2502aa16cd010dd628e6d6dfb4bd2984190aef54b0e9c6b3705c184` |
| `scripts/prepare_pvground_fixed_memory_formal.py` | `b770895e544fb3191afd1a0b7216f511a39dc787e37b745d06c5552e85328d38` |
| `scripts/prepare_pvground_fixed_memory_comparison.py` | `ff2232d48b66d3c7310d56ca315cfacf25c4e49467036674ec674f6d55249cd9` |
| `scripts/observe_pvground_fixed_memory_training.py`（容量补充时最终版） | `f5c399d58bd03c10c8c30da42f4e5bc9ebc00e1b6d3af0aebbca85d25395bb10` |
| 实际 E training `spec.json` | `0babc37d4ecb3fb5e033d6180cccc007c51235d1d52c63316dfd788c8a3fd467` |
| 实际 E `audit_queue.py` | `1527285d3489f0d36adaa56d9d615b806e45815baebab17d27db2743865052b8` |
| 实际 E `formal_queue.py` | `31ebbab7ec114c53fc28ed970aec1600f64cef22ad29c9486ada3209cc1dcdce` |
| 实际 E comparison `queue.py` | `993059a9353db19dd089133dc08161cdf343a0dc1a018478e18c0da2a317e9bf` |
| 实际 source `models/pv_ground.py` 缓存 | `20c353512939ac9087cd8f3d07b82a648bcefe42510563e26afc37235facdb7c` |
| 实际 source `main_utils.py` 缓存 | `14bd101eb78ec3d729968aff0a975684efa8388574583ca9e4b0f512877b4f5c` |
| 实际 source `models/losses.py` 缓存 | `920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de` |

```json
{
  "audit_skill": "experiment-audit",
  "reviewer_model": "gpt-6-astra",
  "reviewer_reasoning": "max",
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "overall_verdict": "WARN",
  "integrity_status": "warn",
  "static_launch_blockers_remaining": 0,
  "resolved_findings": ["combined_export_disk_allowance_insufficient"],
  "evaluation_type": "real_gt",
  "initial_runtime_snapshot_cst": "2026-09-17T17:41:59.466677+08:00",
  "runtime_snapshot_cst": "2026-09-17T17:52:17.495578+08:00",
  "runtime_snapshot_file": "refine-logs/pvground_scanrefer_finetune_20260917_fixed_memory_v1/observation_20260917_175217.json",
  "runtime_snapshot_sha256": "0250d844105142746e9f141a717950a1d259a3ca0fa4bca0cc9f072d41a0617e",
  "runtime_log_file": "refine-logs/pvground_scanrefer_finetune_20260917_fixed_memory_v1/run_20260917_175217.log",
  "runtime_log_sha256": "c4e6dcac0a77ef4ccd14364167b4e7ca15c26275f881aad37911d95a7b6827cb",
  "initial_rows_observed": 2048,
  "initial_rows_expected": 6887,
  "initial_evaluation_complete": false,
  "actual_new_mode_capacity_verified": true,
  "capacity_optimizer_steps": 0,
  "capacity_visual_state_tensors_verified": 204,
  "capacity_sha256": "5930a09745a2a39a872e4db8b37b31dedfa66c80f3d4f3e1f00b85f7435d659f",
  "actual_terminal_restore_verified": false,
  "actual_fixed_visual_memory_terminal_verified": false,
  "formal_performance_result_available": false,
  "claim_ceiling": "single-seed fixed-backbone training strategy; capacity only, no performance result"
}
```

## 11. 17:49:00 补充：真实 E 新模式容量通过

**容量检查 PASS（已归档实际运行证据）；总体仍 WARN / provisional。** 本节闭合初审 B2 中的“新模式容量尚未产生”部分，不改写初审时间线，也不闭合完整 initial、3723 步训练终态或正式评价。

独立读取的文件均来自 `refine-logs/pvground_scanrefer_finetune_20260917_fixed_memory_v1/`：

- `capacity.json:1-31`：`status=pass`，时间 **2026-09-17T17:48:15.204309+08:00**，batch_size=8，optimizer_steps=0，fixed_visual_memory=true，visual_state_unchanged=true，visual_state_tensors=204。
- `run.log:8-9`：完整数据输入检查记录 fit/holdout `29778/6887`，物理场景 `456/106`，4 行固定输入 fixture；随后记录唯一一次 `PVG_BATCH8_CAPACITY_PASS`。日志中的完整 JSON 与 `capacity.json` 逐项相等。
- `observation_latest.json:1-47`：**17:49:00.007053** 快照中的容量内容与独立容量文件逐项相等；仍绑定原 E controller PID 8666。
- `train.py` SHA 仍为 `f03aa1fcb6c4e495c4cff191b3189de2b17644396a684839a928a3c3e7402925`，与 training spec 绑定及此前审过的源码一致。

容量 batch 的 8 行是 `[14307,26871,13547,1622,9672,18692,29949,23417]`，与 D 的首个容量 batch 行序完全相同。loss `13.184042930603027`、clip 前 grad_norm `67.60269165039062` 及所记录各损失均 finite。峰值 allocated 显存为 **14,962,948,608 字节（约 13.9353 GiB）**；此 batch 的计时为 `4.37521767616272` 秒。未据此挑选权重、修改配置或预测最终性能/完整训练时长。

718 / 25,397,851 可训练张量/参数元素、301 / 128,131,008 冻结张量/参数元素与独立 `memory_policy.json` 完全一致。由于已绑定源码在写容量回执前实际执行 `step(..., update=False)` 与 `check_visual_memory()`，这份回执支持：**新冻结模式下在线增强 batch8 的完整原生 loss/backward 已完成，骨干无梯度、全部子模块 eval、204 个视觉 state 与父初值相等，且没有 optimizer 更新**。证据调用位置：driver `248-260`。

范围仍有限：这不是 E optimizer 更新成功或 3723 步训练成功的证明；容量回执在之后的全初值恢复/RNG 重置代码之前写出，当前快照没有完整 initial receipt，因而没有把容量后的完整初始评估报告为已完成；也没有 actual terminal、terminal_restore、终态不漂移或正式指标。CPU 序列化夹具与此真实 capacity 是不同证据，继续分开记录。

17:49:00 当时核对的证据 SHA256（`capacity.json` 保持不变；后两项原可变文件已被后续观察覆盖，只保留历史身份，不能当作当前同名文件 SHA；当前不可变证据见第 12 节）：

| 文件 | SHA256 |
|---|---|
| E `capacity.json` | `5930a09745a2a39a872e4db8b37b31dedfa66c80f3d4f3e1f00b85f7435d659f` |
| E `observation_latest.json`（17:49:00 快照） | `ada5f80c58d6d7b9fbdb4e4767dd1737700f43b836611ec03b309e06baf015f0` |
| E `run.log`（截至 capacity 行） | `8d15f624cc6f8732b164e9e871b2d25de29a155f5372a75ad2433f033d8ef7a5` |

## 12. 17:52:17 快照跟进：绑定不可变副本

主执行器后续观察覆盖了 `observation_latest.json` 和 `run.log`。审计人仅重新读取本地已固定命名副本 `observation_20260917_175217.json` 与 `run_20260917_175217.log`，没有新轮远端查询或新的模型执行。第 11 节两个旧哈希是当时核对记录，本节的固定文件及哈希才是当前可重新读取的快照证据。

固定观察副本 `:1-47` 记录时间 **2026-09-17T17:52:17.495578+08:00**，仍绑定原训练 controller PID 8666；容量内容与 17:48:15 的 `capacity.json` 一致。固定日志 `:8-9` 保留同一输入/容量通过记录，`:10-13` 依次报告 initial `512、1024、1536、2048 / 6887` 行。最终进度为 **2048/6887，204.23065543174744 秒**，不是完整初始评估，也不是 fit 训练步数。

这份进度证明程序已进入容量后初始评估的前向流程；按照已绑定 driver 的执行顺序，先执行完整 state 恢复、相等断言与 RNG 重置，再开始 initial。但没有完整 initial receipt、独立 initial audit、3723 步终态、actual terminal restore 或正式验证结果，相关完成状态仍为 false / pending。总体结论维持 WARN / provisional，容量通过的有限结论不变。

| 当前不可变文件（均位于 E training 归档目录） | SHA256 |
|---|---|
| `observation_20260917_175217.json` | `0250d844105142746e9f141a717950a1d259a3ca0fa4bca0cc9f072d41a0617e` |
| `run_20260917_175217.log` | `c4e6dcac0a77ef4ccd14364167b4e7ca15c26275f881aad37911d95a7b6827cb` |

本节只修正快照绑定和已观察进度，不改变训练、正式评价、晋级或清理逻辑；后续不得将这些不可变快照中的空闲容量或进度写成当前实时值。
