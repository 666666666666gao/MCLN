**Native matcher Mask 职责：实际 CPU 证据审查**

审查时间：2026-10-11T02:47:45.9956298+08:00。结论 **WARN，0 个阻断项**。执行证据范围为 `ACTUAL_NATIVE_MATCHER_CONTROLLED_CPU_ONLY`，评估类型为 **simulation_only**。

这次本地 fresh 审查支持报告：指定 9 个合成 fixture 的原生 matcher CPU 检查已完成，实际回执与原始 payload 字节一致。它不提供完整 benchmark 精度、正常训练掉点因果、贡献有效性或 GPU/训练准入。

请求模型 gpt-6-astra、reasoning_effort=max；无实际身份凭证，因此记录 actual_identity_attestation=UNATTESTED、review_independence=same-family、acceptance_status=provisional。审查只读本地证据并写本报告；没有 SSH、项目导入、检查脚本重跑、模型/数据集/GPU/训练调用，也没有修改活动源码。

**A. GT 来源：PASS**

目标框、positive_map 与二值目标 Mask 在 fixture 中人工定义；之后故意把目标框放入 query 3/17，其他框远离目标，分类 logits 全零。这是可预期的工程对照，目标不是数据集 GT，匹配正确也不是模型精度。此构造在 [check_native_matcher_cpu.py:45](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:45)、[check_native_matcher_cpu.py:53](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:53)、[check_native_matcher_cpu.py:64](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:64) 可直接追溯；实际结果明确 dataset_rows=0、formal_accuracy=null、simulation_only，见 [MATCHER_CPU_RESULT.json:3](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/cpu_execution/MATCHER_CPU_RESULT.json:3)。

**B. 成本、原生调用与捕获：PASS**

check 从固定 warm_source/models/losses.py 用 importlib 加载真实模块，构造 HungarianMatcher(1,0,2,True)。原生 factory 的前三个权重也是 1/0/2；本次没有执行 factory 或独立复核活动训练的全部 flags。见 [check_native_matcher_cpu.py:17](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:17)、[check_native_matcher_cpu.py:31](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:31)、[main_utils.py:277](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/main_utils.py:277)。

capture_assignment 先复制实际传入的完整 tensor，再把同一个原 tensor 交给原 SciPy linear_sum_assignment，返回值也原样透传。预期 query 3/17 是真实调用后的断言，没有替代计算结果。见 [check_native_matcher_cpu.py:33](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:33)、[check_native_matcher_cpu.py:37](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:37)、[check_native_matcher_cpu.py:84](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:84)；原生最终成本与按 batch 分配的调用见 [losses.py:372](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:372)、[losses.py:380](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:380)。

直接 Mask 项仅读取 pred_masks，按 >0 二值化、superpoints 映射后计算点级 L1，乘 0.0002。它不读取 sp_pred_masks 或 adaptive_weights；其他项是原生 class softmax/positive_map 与 GIoU，框 L1 权重为 0。没有以模型自身最大值归一化的性能分数。见 [losses.py:291](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:291)、[losses.py:324](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:324)、[losses.py:328](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:328)、[losses.py:342](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:342)、[losses.py:353](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:353)。

**C. 实际文件、payload 与原生绑定：PASS**

CPU 回执记录 2026-10-11 02:43:49.093757—02:43:51.432678 +08:00，时间差 2.338921 秒；CPU 子进程和传输退出码均为 0，两 stderr 文件均为 0 字节。见 [CPU_EXECUTION.json:2](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/cpu_execution/CPU_EXECUTION.json:2)、[TRANSPORT_EXIT.json:1](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/cpu_execution/TRANSPORT_EXIT.json:1)。本轮用本地 JSON/base64 与 SHA256 工具独立核对以下字节，未运行项目代码：

| 原始 payload 字段 | 落盘产物 | 字节数 | 独立字节比较 |
|---|---|---:|---|
| stdout_base64 | CPU_STDOUT.txt | 147 | 完全一致 |
| stderr_base64 | CPU_STDERR_PRIVATE.txt | 0 | 完全一致 |
| result_base64 | MATCHER_CPU_RESULT.json | 5165 | 完全一致 |

原始 payload 在 [RAW_STDOUT.json:1](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/cpu_execution/RAW_STDOUT.json:1)；解码落盘代码在 [run_authorized.py:65](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/run_authorized.py:65)。结果 SHA256 为 `7eb917935e2457d7dc47ff52a3c86c8111a3bfbdc2262c1b86db9fe2964417aa`；另外 7 个非 payload 字段逐项匹配 CPU_EXECUTION，后者保存的 source_review_sha256 也匹配本次读取的 SOURCE_REVIEW.json。

三份原生源本地完整 SHA256/字节数与 spec、实际结果摘要及部署清单相符；actual_losses_path 与 warm_source 一致。实际 utils 和 utils.scatter_util 路径/摘要匹配导入后的断言和 spec。见 [CHECK_SPEC.json:3](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/CHECK_SPEC.json:3)、[check_native_matcher_cpu.py:25](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:25)、[MATCHER_CPU_RESULT.json:14](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/cpu_execution/MATCHER_CPU_RESULT.json:14)、[NATIVE_SOURCE_PORT.json:147](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/NATIVE_SOURCE_PORT.json:147)、[NATIVE_SOURCE_PORT.json:188](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/NATIVE_SOURCE_PORT.json:188)、[NATIVE_SOURCE_PORT.json:215](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/NATIVE_SOURCE_PORT.json:215)。

正确 complete_fit/PV-Ground/utils 的四份本地源文件均匹配历史部署清单及 source review 的封存值；但 lr_scheduler/logger 没有本次独立运行时路径/摘要回执，第三方包也未完整封存。utils 顶层只导入 scheduler/logger；所读源码中的日志写入须调用 setup_logger，此检查没有该调用。见 [utils/__init__.py:6](C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/utils/__init__.py:6)、[utils/lr_scheduler.py:7](C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/utils/lr_scheduler.py:7)、[utils/logger.py:33](C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/utils/logger.py:33)、[utils/logger.py:76](C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/utils/logger.py:76)、[NATIVE_SOURCE_PORT.json:626](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/NATIVE_SOURCE_PORT.json:626)、[NATIVE_SOURCE_PORT.json:668](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/NATIVE_SOURCE_PORT.json:668)。env 摘要在相关记录间一致，本轮未另读 env 内容重算其规范化摘要；不声称完整环境证明。

既有 SOURCE_REVIEW 是执行前的 SOURCE_ONLY 报告，其“结果尚不存在”属于当时状态；本审查没有把该报告的 verdict 当作实际成功证据。当前实际判断来自上面的产物、payload、独立源码阅读与哈希核验。见 [SOURCE_REVIEW.md:5](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/source_review/SOURCE_REVIEW.md:5)、[SOURCE_REVIEW.md:36](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/source_review/SOURCE_REVIEW.md:36)。

pv_ground.py 与 main_utils.py 在本检查只校验文件摘要，没有导入/运行；实际被调用的是 losses.HungarianMatcher。

**D. 实际调用数、分配与成本结果：PASS；归档强度有保留**

结果有 9 条记录：7 个 B=1 fixture 和 2 个 B=2 fixture。结合每次 len(captured)==batch 的断言和成功回执，可对应 9 次 matcher 调用、11 次 SciPy 分配、16 个 GT 配对；这不是 9 个真实场景，也不是额外 profiler 记录。所有单 GT 项分配 query [3]→gt [0]，双 GT 项分配 query [3,17]→gt [0,1]。见 [check_native_matcher_cpu.py:75](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:75)、[check_native_matcher_cpu.py:93](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:93)、[MATCHER_CPU_RESULT.json:31](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/cpu_execution/MATCHER_CPU_RESULT.json:31)。

| 检查 | 回执中的总成本差摘要 |
|---|---|
| single_own / multi_own / mixed_batch_own | 4 个对应矩阵的最大绝对差均为 0 |
| single_shared_text | 首行列偏移 +0.0003999471664428711；最大列内偏移散布 1.1920928955078125e-7 |
| two_gt_shared_text | 首行列偏移 [+0.0003999471664428711, -0.0004000663757324219]；最大列内散布 2.384185791015625e-7 |
| single_mask_component | 偏移 0，散布 0；原 Mask 成本为零的阴性对照 |

这些实际摘要位于 [MATCHER_CPU_RESULT.json:214](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/cpu_execution/MATCHER_CPU_RESULT.json:214)。改变公共 Text Mask 的两项非零偏移与 4 点二值 L1 的源码推导约 ±0.0004 相符；散布小于脚本的 1e-6 阈值。single_mask_component 的 Text Mask 原本恰等于唯一 GT，因此删除 key 的 0 差值不能单独证明非零 Mask 项。见 [check_native_matcher_cpu.py:56](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:56)、[check_native_matcher_cpu.py:64](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:64)、[check_native_matcher_cpu.py:109](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:109)。

**W1：完整成本矩阵只在内存捕获。** 落盘只有分配和差值摘要，见 [check_native_matcher_cpu.py:89](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:89)、[check_native_matcher_cpu.py:106](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:106)、[check_native_matcher_cpu.py:118](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:118)、[check_native_matcher_cpu.py:130](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:130)。本轮能核对真实 payload 字节和检查路径，不能声称从归档全矩阵独立重算过数值或再次求解过匹配。

**E. 实际 scope 与新 claims：WARN，非阻断**

实际范围是 B=1/2、GT=1/2、Q=T=256、P=4 的确定性合成输入，无真实数据/随机采样或多 seed、PV 前向、criterion、backward、优化器更新。fixture 固定框/logits，而且框的几何对应被故意做得清楚；其不变分配不构成浮点近并列情形的一般运行证明。见 [check_native_matcher_cpu.py:45](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:45)、[check_native_matcher_cpu.py:93](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:93)、[MATCHER_CPU_RESULT.json:3](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/cpu_execution/MATCHER_CPU_RESULT.json:3)。

在数学层面，广播 Text Mask 在每一 GT 列添加 query 无关常数；当 GT<=Q 且每个 GT 都匹配时，每个完整分配得到相同实数总偏移。源码的广播与匹配支持这一条件推导，但它与“所有真实 float32 输入都逐位保持分配”不同。见 [pv_ground.py:548](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/pv_ground.py:548)、[losses.py:333](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:333)、[losses.py:380](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:380)、[SOURCE_FINDING.md:17](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/SOURCE_FINDING.md:17)。

**W2：固定 matcher 输入没有覆盖整个模型。** 所读 pv_ground.py 仍包含几何参考与 span 调用及末层框改写；这些框随后进入 matcher。所读 loss_masks 在匹配后按 idx0 使用 Query Mask，Text Mask 也有原生监督。本轮没有执行这些分支，不能把“直接 Mask 项不读 sp_pred_masks”升级为“整个模型没有使用 Query Mask”或“这是续训掉点原因”。见 [pv_ground.py:593](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/pv_ground.py:593)、[pv_ground.py:604](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/pv_ground.py:604)、[losses.py:574](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:574)、[losses.py:623](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:623)、[losses.py:899](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:899)。

**W3/W4：对照与运行证明的边界。** 单 no-mask 对照为零成本；非零作用由 shared-text 对照给出。gpu_calls=0 应理解为本检查无 GPU 运算/初始化，脚本仍调用 torch.cuda.is_initialized()；零训练/模型等字段主要是源码范围加回执声明，并非全机独立监控。runner 显式置空 CUDA_VISIBLE_DEVICES，以 -B 执行隔离检查；没有活动训练 observer 或源码写入路径。见 [run_authorized.py:36](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/run_authorized.py:36)、[check_native_matcher_cpu.py:122](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/check_native_matcher_cpu.py:122)。原生 loss_masks 中存在 .cuda()，但本次没有调用它，见 [losses.py:620](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/models/losses.py:620)。

SOURCE_FINDING 的直接成本与非因果/非精度限定没有超出本次证据。第13行数据接口 132 上限、第29行历史负实验、第35行前作不属于本次独立实际复核对象；第33行是后续比较原则，未由本审查实施或准入。见 [SOURCE_FINDING.md:13](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/SOURCE_FINDING.md:13)、[SOURCE_FINDING.md:25](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/SOURCE_FINDING.md:25)、[SOURCE_FINDING.md:29](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/SOURCE_FINDING.md:29)、[SOURCE_FINDING.md:33](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/SOURCE_FINDING.md:33)。历史部署清单也不是当前训练存活见证，见 [NATIVE_SOURCE_PORT.json:821](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/NATIVE_SOURCE_PORT.json:821)。

**F. 可使用的结论与阻断项**

- 可报告：9 个指定合成 fixture 的原生 matcher CPU 检查实际完成；原始 payload 与落盘产物一致；固定框/logits 时，所列 Query Mask 交换的成本差为零，公共 Text Mask 改变产生近似列常数偏移，所列分配不变。
- 不支持：归档全成本矩阵已独立重算、所有输入的浮点分配等价、Query Mask 对整个模型无作用、掉点因果、正式 benchmark 精度、第三个有效贡献、训练完成或 GPU 准入。
- 阻断项为空；无需为这份有限工程结论修改活动源码或补跑训练。保持上述限定即可引用本回执。

完整 21 个项目输入及 5 个使用的技能/政策文件的绝对路径与 SHA256（无前缀）见 ACTUAL_REVIEW.json。该 JSON 另保存 payload 字节核验、实际摘要、全部限制及逐条 file:line 证据。

