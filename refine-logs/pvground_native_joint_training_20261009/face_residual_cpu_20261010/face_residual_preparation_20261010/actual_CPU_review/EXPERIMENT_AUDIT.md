# Face residual 合成 CPU 实际结果审查

**结论：PASS；bounded_check_verdict：PASS；阻断问题：0。** 实际保存的 CPU 执行证据支持本报告限定的模块工程结论，不构成正式方法有效性验收。未发现需要修改待审输入或重跑 CPU 的问题。

日期：2026-10-10。审查代理：`/root/pvg_face_residual_cpu_actual_20261010`。请求路由为 `gpt-6-astra / max / fork_turns=none`；无独立身份凭据，实际模型与推理强度均为 `UNATTESTED`，`review_independence: same-family`、`acceptance_status: provisional`。

以下 `F` 指 `C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_residual_preparation_20261010`，`S` 指其父目录的 `source`。审查直接读取实际源码、原始传输 stdout、子进程 stdout、结果和历史审查输入；没有执行待审模块、SSH、GPU 操作或当前训练查询。仅新增本报告和指定 trace。

## A. Ground Truth Provenance — PASS

本任务没有数据集 GT，也没有基准精度评估。`F/check_face_residual_modules_cpu.py:63` 起用随机 query、support、points、logits 和框生成合成输入；`F/check_face_residual_modules_cpu.py:95` 起另设固定框、0.5 gate 和人工目标，损失定义在第 109 行。零输出检查以已加载 prior 的输出作为工程等价参照，不能作为预测正确性的 GT。计划已明确标为合成模块检查（`F/CPU_CHECK_PLAN.md:5`、`:7`、`:9`），结果也报告 `real_loader_rows: 0` 与 `formal_accuracy: null`（`F/cpu_execution_attempt2/CPU_MODULE_RESULT.json:36`、`:43`）。

分类为 **synthetic_proxy / synthetic_module_engineering**；这里的 PASS 表示参照来源及用途声明正确，不表示存在真实 GT 性能证据。

## B. Score Normalization — PASS

实际输出的是参数/状态数量、精确张量等价断言、原始梯度 norm 和人工损失，没有将预测的最大值、均值或其他自身统计量作为精度分母。`F/check_face_residual_modules_cpu.py:109`、`:120` 直接报告人工 MSE 和梯度 norm。`S/extremal_span_mixer.py:76`、`:85` 的 count 权重归一化，以及 `F/face_residual_span_mixer.py:42` 的几何尺度归一化属于模块输入计算，未被报告成准确率。

## C. Result File Existence and Provenance — PASS

七份 attempt2 原始/派生文件均存在。新的只读核对确认：

- `F/cpu_execution_attempt2/RAW_STDOUT.json:1` 中三项 base64 解码后分别与 `CPU_STDOUT.txt`、`CPU_STDERR.txt`、`CPU_MODULE_RESULT.json` **字节相同**；stdout 解析出的 JSON 与结果 JSON 相同。
- `TRANSPORT_EXIT.json:1`、`CPU_EXECUTION.json:4` 和原始传输 receipt 均为 exit 0。transport stderr 与 child stderr 均为空。
- `CPU_EXECUTION.json` 恰好等于原始 receipt 的非 base64 字段加 launcher 写入的 `source_review_sha256`，符合 `F/run_face_residual_cpu_authorized.py:79` 至第 88 行的实际收集路径。
- 子进程时间为 **2026-10-10 07:54:00.788183–07:54:02.740352 +08:00**，用时 **1.952169 秒**。实际 `torch.__version__` 输出为 **1.10.2+cu111**（checker 第 158 行、结果第 3 行）。
- receipt 所绑定 R2 SHA 为 `a4bd33432972ae7bfb168037a462ff9ff94a3f7ef61536aa826ddc964636b1d3`，与当前 canonical、R2、075336 timestamped JSON 均相同。R2 finalization 记录时间 07:53:36.143373 早于 CPU 开始时间，且其记录的 canonical digest 相同。
- R2 列出的 **30 个实际输入摘要全部匹配**。R1 的计划、模块、checker、初始化配置仍匹配历史摘要。R1 launcher 按归档 `transport_attempt1/REVIEWED_LAUNCHER_ATTEMPT1.py` 核对，其 SHA 为 `af231e30a21a7c06dab27c50350928423187e430d936081a6d050e7ac0d73a00`；当前 R2 launcher 为 `599161fe51f152d71d3348ad98cba6718b0122d1e6a478cfa5b4457782d3da4f`。没有把修改后的活路径说成 R1 原件。

旧 A/B/geometry 三个本地源码摘要均与 `../NATIVE_SOURCE_PORT.json` 一致。其 `model_source`、环境 canonical digest 与 receipt 一致。checker 在第 28–31 行对三份旧源码和两份 prepared 源码核对 SHA，在第 39–52 行对固定 d06/f989 checkpoint 核对 SHA 并 strict load；完整结果在这些操作之后生成。

权重绑定分别为 d06 `d06adb8ed227d98d46afa8e3970dcb9fbe31ab2613c675883b80162f830f4cb6`、f989 `f9898c5dc7a5e97bf152421efc9072c23ef255618c64477a4bfacf996f2b3b19`（`F/source_conditioned_init.json:9`、`:11`）。这是保存的已审 launcher/child 执行链提供的运行证据；审查者没有再次打开远端权重，也没有独立的后端执行身份凭据。

首轮 exit 255、0 字节 stdout、133 字节私有 stderr 与归档逐字节一致；私有 stderr 仅在内存核对主机密钥校验失败类别，没有复制或引用其文本。07:47:14.188669 的静态记录为固定 CPU root/receipt 不存在且 transport exit 0（`F/STATIC_CPU_ROOT_READ.json:2`、`:3`、`:4`、`:6`）。结合 host-key failure 和先建目录后启动 child 的顺序，支持首轮限定 checker 未启动。R1/R2 报告及失败记录均保留。证据不证明 OpenSSH 内部精确解析根因。

## D. Executed Paths and Engineering Assertions — PASS

已逐行审读 checker 和 wrapper。下列调用均位于结果写入（checker 第 169 行）之前的实际顺序路径，而非未调用的展示函数。

| 支持的限定结论 | 实际证据 |
|---|---|
| d06 A 的 10 项状态与 f989 B 的 14 项状态 strict load 完成 | checker 第 39–52 行；child exit 0 且生成最终结果 |
| 保留的 axis prior 29793 参数，新增 head 23425 参数，总计 53218 参数；模块 20 项 state，新增 6 项 | wrapper 第 18–21 行；checker 第 56–61 行；结果第 11–15 行；本审查另以整数公式核对 |
| source-visible 与 source-hidden 模式在零输出时中心/尺寸与已加载 prior 精确相等 | checker 第 73–90 行分别调用 prior 和两个 wrapper，并执行 `torch.equal`；结果第 16 行。结果称其为 bitwise equal，本报告按实际 `torch.equal` 张量相等检查表述，未另外获得浮点内存字节转储 |
| 新 head 的隐藏源输入列确实清零 | wrapper 第 37–41 行；checker 第 80–89 行的 pre-hook 直接检查列 0–31 与 324；结果第 17 行。共享旧 axis prior 仍使用来源信息 |
| 独立内部 gate 夹具首步输出层有梯度，第一次更新后首层有梯度 | checker 第 94–122 行；结果第 18–30 行；仅新 head 的 AdamW 参数被更新 |
| 给定残差实际进入 Torch 倒置端点解码路径 | checker 第 126–142 行给定 x 两面残差 +0.5/-0.5，执行 `refine_one` 并断言 raw size=-3、输出 size=3；wrapper 第 52–62 行；结果第 32 行 |
| 独立模块 state 在内存 strict reload 后恢复相同 `refine_one` 中心/尺寸输出 | checker 第 144–156 行使用 `BytesIO`、strict load 与 `torch.equal`；结果第 33 行 |

两个梯度记录分别为：step 0，output norm **0.028049109503626823**、encoder norm **0**、loss **0.024999991059303284**；step 1，output norm **0.02770828641951084**、encoder norm **0.00021599180763587356**、loss **0.02476266585290432**。这些是两次反传、两次内存更新的合成记录；未作训练效果或收敛推断。

A 实际调用的 `whole_mask_range.member_statistics` 已审读，本地副本 SHA 匹配 port（`S/mask_support_corrector.py:9`、`:59`；`C:\Users\gb\.codex\tmp\pvground_whole_mask_range_20261003\whole_mask_range.py:12`）。child 直接 rehash 的旧源码为 A/B/geometry 三份；该传递依赖的运行时 `__file__` 和独立运行时 digest 未输出。因此这里不声称每一个环境依赖都获得了独立执行来源证明。

## E. Scope Assessment — PASS

证据范围是一次 CPU child、固定 seed 2027、一批 256 queries、8 superpoints、50000 随机点，以及两个初始 source 模式。它不是 256 个真实样本、50000 个验证条目或多次独立实验。

梯度夹具另建 prior 并直接供给 0.5 gate（checker 第 94–103 行），没有验证已加载 f989 的实际 gate 饱和模式或其训练梯度。倒置测试替换 head 为给定残差，不证明网络预测过这些残差。roundtrip 中的 prior 属于该独立夹具，恢复后只调用 `refine_one`，不证明 f989 wrapper 全 forward、optimizer、RNG 或完整 PV 恢复。

CPU 边界由 reviewed launcher 的空 `CUDA_VISIBLE_DEVICES`、单 CPU 线程设置（第 52 行），CPU checkpoint 加载（checker 第 42–43 行）、源码中的 CPU 构造和最终 `torch.cuda.is_initialized()` 断言（第 157 行）支持。结果中的 `GPU_calls`、查询数等零值是按程序范围写入的声明字段，不是全机硬件或进程监测统计；审读路径中没有 GPU 计算或活动训练查询调用。

**不支持的范围：**真实 loader/场景、native criterion、1301-state native factory、完整 PV 或完整 optimizer 恢复、GPU 执行、正式 REC/精度增益、已证明的模块有效贡献、GPU 训练准入、整体目标完成。当前主训练和 C-off-first 顺序不在本次审查中改动，也未查询其状态。

## F. Evaluation Type — synthetic_proxy

更具体的用途是 `synthetic_module_engineering`。所有准确性类结果均为空，实际证据只支持已列出的工程性质。限定范围内没有发现虚构 GT、自归一化性能、缺失结果、未执行断言或范围扩张。

## 处置及封存

`blocking_issues: []`，`nonblocking_issues: []`。上述限制是本次声明边界，不要求新增 fallback、防御分支、兼容层或无关重构。此次 PASS 不触发任何后续运行授权。

本审查的 **36 项确定性核对全部通过**，60 份实际输入路径及裸 SHA256 写入 `EXPERIMENT_AUDIT.json`。完整请求、响应、只读核对脚本和实际工具结果存于 `F/.aris/traces/experiment-audit/ACTUAL_CPU`。`OUTPUT_SHA256.json` 以其所在 `actual_CPU_review` 目录为相对路径基准封存报告与该 trace；封存不含自身或任何私有 SSH stderr 原件。
