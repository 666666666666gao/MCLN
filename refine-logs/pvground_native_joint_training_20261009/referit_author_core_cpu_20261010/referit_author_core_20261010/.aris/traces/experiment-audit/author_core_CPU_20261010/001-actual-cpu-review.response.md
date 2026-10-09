# 实际 CPU 作者核心加载复核

**限定 CPU 证据 PASS；总体 WARN；本次审计范围内阻塞问题 0。** 两次实际 `TrainTester.get_model` 构造及各自作者核心严格加载有完整、一致的执行记录支持。此结论不放行 Nr3D/Sr3D 训练、对象输入协议或 REC 精度。

审计对象为 2026-10-10 07:13:28.638211—07:13:53.032772 CST 的单次执行，墙钟 24.394561 秒。复核者 `/root/pvg_referit_author_cpu_actual_20261010` 与执行者分离；请求模型/推理档为 `gpt-6-astra / max`，实际两者均 **UNATTESTED**。路线标记为 `same-family / provisional`，不声称跨品牌接受。委派消息包含执行者摘要；本报告以直接读取的源码、原始回执和本地重算为依据，不声称盲审或严格零上下文。

## A. 参考来源与检查含义：PASS（工程检查适用；科学 GT 不适用）

检查目标来自对应作者 checkpoint，经初始化器实际计算 SHA256 后 `torch.load(..., map_location='cpu')`，并严格载入全部 1,235 项状态；不是从模型预测生成标签。Nr3D 为 `d2d9afaf9c293c54977f3555a46c7bb2a72d9f80163a3f602dd8426032d7fa5d`，Sr3D 为 `a4a14b0090947177a648703ad6de094246891d89174fffa56fe454f730dbe3dc`（`source/native_model_initialization.py:13-15,30-36`；`nr3d/init.json:4-5`；`sr3d/init.json:4-5`）。本审计未重新读取远端 checkpoint；哈希执行证据来自已绑定的检查路径和成功回执。

加载后的核心与同一作者 checkpoint 的再次读取作精确比较。这验证目标 state 原样进入新工厂模型，不是独立参考数值计算、科学正确性证明或性能复现（`../referit_author_core_20261010_check_cpu.py:43-54`）。两份旧 `strict_load.py` 的实际文件哈希与其历史 JSON 一致，历史 checkpoint 哈希与当前 spec 一致；旧工厂记录只作来源归属，不替代本次结果。

三份原始 inventory 已用 stdlib 重新比较：ScanRefer 1,234 项，Nr/Sr 各 1,235 项，唯一新增项为 `text_encoder.embeddings.position_ids`，shape `[1,514]`、dtype `torch.int64`，所有公共项 shape/dtype 一致。此 inventory 比较本身没有张量值证据；本次执行提供了保留核心的值比较证据。重算细节在 trace 的 `deterministic_verification.json`。

## B. 分数归一化：PASS（未计算性能分数）

本次没有准确率、模型输出指标或归一化分母。`core_key_shape_dtype_and_values_exact` 是覆盖全部目标核心 state 的布尔断言结果，不能转换成“100% REC”或泛化质量结论（`../referit_author_core_20261010_check_cpu.py:50-59,77-80`）。

## C. 文件、原始回执与终态：PASS

对 `cpu_execution/RAW_STDOUT.json:1` 内三项 base64 使用严格解码，再与各自落盘文件逐字节比较，全部一致：

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| CPU_STDOUT.txt | 3,676 | `40bef6a4a028d2030c5794a146d78e27d8b6f22abb948577046e95f766683e20` |
| CPU_STDERR.txt | 1,486 | `305add9a7767b087422925300c01ac24f5b215384d67d7c56aedc939acdacf99` |
| AUTHOR_CORE_CPU_RESULT.json | 4,542 | `af6bd6632437239c730e39b292b13b7b0145ec89d6ff99309bb726056ab56d6f` |

stdout JSON 与独立保存的 result JSON 语义完全相等；这是同一次执行的两种输出，不是两次独立实验。raw metadata 与本地 `CPU_EXECUTION.json` 除新增 source-review 哈希外一致。传输退出码、raw 子进程退出码和本地执行退出码均为 **0**；传输 stderr 为空。源审查绑定值为 `87f68880f93a224aaf89fc4f5bfc67504b135c1d1187d4ae97ed85c3f0e4e287`，与 R3 文件实测一致（`cpu_execution/TRANSPORT_EXIT.json:1`；`cpu_execution/CPU_EXECUTION.json:2-11`；`../run_referit_author_core_cpu_authorized.py:54-61,73-86`）。

`CPU_STDERR.txt:1-6` 是两次 RobertaModel 构造报告的未使用语言模型头权重；源码随后执行对应作者 state 的 strict load，最终退出 0。这里没有发现被忽略的作者核心缺键/多键失败。会话 `43694 closed0` 仅由委派消息提供；审计未查询该会话，不将其当作另一个独立终态证据。

`SOURCE_PREPARATION.json` 与 `PLAN.md` 的 CPU_PENDING 是封存的执行前状态，实际完成状态来自后续 CPU 回执。为保持历史证据，本审计不修改这些旧状态文件。

## D. 实际调用和断言覆盖：PASS（限定范围）

checker 在隐藏 CUDA 后导入 torch，记录版本 **1.10.2+cu111**，并在构造前及每例构造后检查 CUDA 未初始化。实际路径为 `/root/autodl-tmp/pvground_referit_author_core_cpu_20261010/PV-Ground`；通过 `inspect.getfile` 约束真实 `TrainTester` 和 `configure_native_model` 来源。`TrainTester.__new__` 绕过训练器初始化，随后确实调用真实工厂，循环仅含 nr3d、sr3d；每例种子均为 2027（checker `:14-42,62,77-80`；`source/train_dist_mod.py:100-129`；result `:121-126`）。

严格加载路径先要求并加载作者 1,235 项（含 position_ids），再验证位置缓冲区等于确定性 arange，之后改为 nonpersistent。保留核心状态为 1,234 项；安装 G 后 1,271 项，安装 A/B 后 1,295 项。没有丢弃未知作者键或放宽 strict（`source/native_model_initialization.py:27-48`）。“1,295 项”是内存中的 `state_dict()` 计数，本次没有保存并重载模型 checkpoint。

核心检查覆盖每例全部 1,234 项的 key 存在性、shape、dtype、CPU device 和 `torch.equal` 值相等，共两例。另有 position_ids 相等且不在最终 state、文本编码器冻结、G/C 标志和 A/B 输出层 state 全零断言。结果文件只在这些断言之后生成；成功终态及与源码一致的输出支持这些检查经过执行（checker `:43-82`；result `:7-13,65-71`）。这不是本审计重跑，也没有逐算子的外部探针日志。

必须保留以下精度边界：

- **A/B 是输出层 weight/bias 状态为零**，没有运行 A/B forward 或模型 forward，不能声称实际前向输出已测得为零（checker `:57-59`；`source/mask_support_corrector.py:20-22`；`source/extremal_span_mixer.py:25-27`）。
- G/A/B 是按本次作者核心新安装的模块；“fresh”不等于每个 G 参数独立随机初始化。G 的投影/attention 按源码复制本次已加载核心，另含新零值参数（`source/pvground_source_query.py:18-37`；`source/pvground_observation_query.py:130-142`；`source/pvground_task_observation_query.py:15-23,56-62`）。本次没有额外的 G 全状态独立数值参考比较。
- 两份 spec 的 G/A/B checkpoint 都为 null，初始化器断言该条件并仅载入对应作者核心，再安装模块。这支持没有加载 ScanRefer 训练 checkpoint 或模块 overlay；架构字典里的 false 不单独构成证据。构造仍使用公共离线 RoBERTa 和 class embeddings 等既有依赖（两份 spec `:6-9`；initializer `:21-22,30-48`；`source/models/pv_ground.py:168-190`）。
- 0 forward/criterion/optimizer/loader/GPU/训练查询由本次静态调用边界与回执共同支持；零计数字段并非动态 hook 计数器。本审计没有扩展验证整个训练 pipeline。

## E. 输入连续性与范围：WARN

R3 当前 **114/114** 输入 SHA256 与封存清单一致。R2 的 14 份源码与 2 份 spec、8 份固定/时间戳 R1/R2 报告均保持原字节；R3 相较 R2 仅 `native_model_initialization.py` 不同。原生本地 source 在 R3 已列文件范围内亦哈希一致。runner 读取 port 声明的 116 份源文件、逐个检查哈希并复制到新隔离目录，再覆盖 14 份本轮源码；写入路径不指向活动原生 source。审计未访问远端当前状态，也不将本地哈希表扩展成远端当前状态证明（runner `:13-17,21-54`；`SOURCE_PREPARATION.json`）。

R2 原 76 项输入中 75 项一致。唯一历史例外是 live `refine-logs/EXPERIMENT_TRACKER.md` 从 `03bca0f2ce19ac70a59c5a38449dcf322b24ee0d46895f51181b00dfccb7343a` 变为 `f662c38d6eb125544c38ae54e7b2ff855f52a1d1a854d5b619f0ae183fcf28fb`。`R2_CLOSED_TRACKER_INPUT.md` 仍精确匹配旧哈希，`R2_SEALED_TRACKER_SNAPSHOT.json` 的 C:/D: 路径别名经同文件检查通过。此例外已在 R3 trace 记录；不能声称全部 live R2 输入从未变化。原始报告和归档未改写。

实际覆盖是 **Nr3D、Sr3D 各一次 CPU 工厂构造，单种子 2027，真实数据行 0**。两例均记录以下四项协议差异（result `:33-53,91-111`）：

| flag | 作者训练配置 | 本次 CPU 构造 |
| --- | --- | --- |
| butd | false | true |
| butd_cls | true | false |
| joint_det | true | false |
| detect_intermediate | true | false |

`get_model` 将 butd/butd_gt/butd_cls 取逻辑或，因此此处结构可兼容；dataset 构造却分别使用这些 flags，且 joint_det/detect_intermediate 控制数据与检测设置。这解释了“装载成功”不能证明真实输入协议等价（`source/train_dist_mod.py:61-80,83-96,121`）。不能据此认定 Nr/Sr 正式训练已准备完毕。

## F. 类型与可支持主张

类型为 **engineering_initialization_check**；科学 GT 评价分类不适用。既不是 real_gt 性能评估，也不是 synthetic_proxy 或仿真性能实验。

支持的主张：已绑定 R3 源码和各自作者 checkpoint 的两次真实 CPU 工厂构造完成；严格作者核心加载、1,234 项保留核心精确一致、位置缓冲区处理、1,295 项内存状态及 A/B 输出层零状态检查有一致回执支持。

不支持的主张：模型或 A/B 前向数值、完整 criterion、真实 loader、optimizer 更新、保存/恢复或冷恢复、GPU 运行、真实对象协议与作者一致、正式训练效果、REC 精度、多种子稳健性及跨基准科学结论。实际 reviewer 模型/推理档未验证。

## 后续与审计记录

本次没有提出 CPU 源码修补要求。最终 Nr/Sr 训练仍需同一方法定型、真实数据/对象/检测协议检查及其另行授权的工程验证。本报告不给予训练放行。

本审计只进行了局部源码/JSON/base64/哈希/AST 检查，没有 SSH、模型导入/构造、GPU、训练状态查询、CPU 重跑或输入修改。未读取 MEMORY、SOUL、USER、日记、AUTH 或 askpass 内容；R1 bootstrap 差异只来自现有 R3 metadata。标准库校验脚本退出 0；137 个审计输入实际哈希收录于 JSON，原始 CPU 文件完整复制至 trace。完整请求、报告响应、身份限制、执行证据和输出哈希在 `.aris/traces/experiment-audit/author_core_CPU_20261010` 与 `actual_CPU_review/OUTPUT_SHA256.json`。
