# Nr3D／Sr3D 原生源码审查 R1

**结论：FAIL；2 个阻塞项、1 个非阻塞文档问题。** 这是 experiment-bridge Phase 2.5 的本地源码审查，不能作为 CPU 执行、模型初始化、部署或精度验收。

请求路由为 `gpt-6-astra`／`max`；当前 host 未证明实际模型及 effort，二者记录为 **UNATTESTED**。审查者 `/root/pvg_referit_native_source_20261010`，`review_independence: same-family`，`acceptance_status: provisional`。本审查不能声称跨模型族或完全零上下文验收。

源码路径以 `referit_native_preparation_20261010/` 为根；历史 import artifacts 位于上层 `pvground_native_joint_training_20261009/`，完整绝对路径见 JSON。

## 阻塞项

### B1：验证与 best 保存仍使用 ScanRefer 的样本数和门槛

`source/train_dist_mod.py:252–257` 仍断言两个原生评价分母都是 9508，并返回 `rows=9508`；`source/main_utils.py:172–176` 再次要求 `rows==9508`。`main_utils.py:365–370` 在第一轮训练前就调用 E0 评价。因此，实际验证集条目数不等于 9508 的 Nr/Sr 运行会在 E0 结束时终止，无法进入正常训练。这是当前源码的必经路径，不需要启动 GPU 才能发现。

同一文件 `main_utils.py:167–169,382–384` 还按 ScanRefer 的 5658／4850 命中门槛排序并保存 best。只删除 9508 断言，仍会留下错误的分母记录和跨基准 best 选择规则。

最小修正：原生 evaluator 和 bbs 计算保持原样；使用各自真实完整验证集分母，并在记录处核对该分母；明确定义 Nr/Sr 的原生 REC best 排序，去掉 ScanRefer 命中门槛。此修正需要窄范围修改 `main_utils.py`，因此如实把计划和变更清单由 5 文件更新为 6 文件。不要仅为维持“5 文件”而保留已发现的问题。

### B2：CPU 执行器漏掉已经验证过的运行环境绑定

`check_native_targets_cpu.py:9–13` 导入原生 `models.losses`。提供的真实 `models/__init__.py:7–10` 先导入 PVGround；其 SHA 与 `native_joint_v2/NATIVE_SOURCE_PORT.json` 中的原生文件一致。导入链会经过 `pvground_observation_query.py` 到 `pcdet`。

这里有现成失败证据：`runtime_import_attempt1/NATIVE_RUNTIME_IMPORTS_CPU_STDERR.txt` 的链路最终是 `ModuleNotFoundError: No module named 'pcdet'`，对应 exit 1。后续已成功的 `check_native_runtime_imports_r2_authorized.py:16–19` 先读取既有 runtime `env_spec.json`，写入 `env` 并绑定 `PYTHONPATH`；`NATIVE_RUNTIME_IMPORTS_CPU.json:81–90` 记录了其实际导入结果。

新 `run_native_targets_cpu_authorized.py:45–48` 只继承环境、设置 CUDA 与线程变量，然后启动检查器，没有复用这段已证明必要的绑定。因此在执行目标／梯度断言之前仍有已知导入缺口。**本次没有运行新检查器，不把历史失败说成新检查器的实际失败，也不把历史成功说成当前运行已通过。**

最小修正：复用原有已核对 runtime 环境及 PYTHONPATH 绑定，在其后明确保持 `CUDA_VISIBLE_DEVICES=''`，保留“原生 SetCriterion 对照准备版 G/C”的来源关系。无需安装依赖、增加 fallback 或兼容层。

## 非阻塞问题 W1：B 的初始门控说明错误

`refine-logs/EXPERIMENT_PLAN.md:14` 写初值 0.5；实际 `extremal_span_mixer.py:25–27` 将 output 权重和 bias 清零，`:97–102` 使用 `raw_gate.clamp(0,1)`，所以 fresh B 的初始门控是 **0**，初始框来自 Mask reference。初始化器不会加载 span 状态。这既不是固定半混合，也不能称作者原模型 E0。

如果遵守本轮保持原结构的要求，最小修正是将文档改成实际的 gate 0／完整 Mask reference，不要为了匹配文字偷偷修改计算。

## 已核对正确的范围

- 14 对源码逐字节 SHA 对比，恰好是准备清单中的 5 个文件改变；所有 current／prepared 哈希都吻合所给 manifest。两个 `original/` G/C 文件也分别等于 current source。9 个其余文件相同，包括 forward、读模块、A/B、Mask geometry 和 root bbs。这里证明本地来源及源码连续性，未查询远端活动源码，也未实际构造模型验证状态字典。
- G 使用首个有效 root；对所有匹配 Query 建立保护掩码；真实 `sample_dataset=scannet` 被排除。`language_dataset` 只决定原生任务配方。ScanRefer／Nr 的 0.6／0.2／0.2／0.1 与 Sr 的 0.625／0.125／0.125／0.125，分别吻合实际 `SetCriterion.loss_pos_align`。既定单 rank、6 Decoder 时外部系数分别为 0.5/7 和 1/7，eos 权重、非归一目标质量和 entropy 项均保留。
- 实际数据源码快照 `038_joint_det_dataset.py:1086–1119` 将表达 root 放在 GT 槽 0，可选 anchor 放在后面，有效槽连续；`:1264–1272,1402–1403` 明确区分 benchmark weighting 与 annotation source；`:1055–1081` 给出 root／anchor 文本 map 规则。原 criterion 的有效槽过滤不改顺序。这支持源码接口契约，未检查真实样本。
- C 在增加目标前跳过 matched-root、matched-other 与 detection 行；只给原生 bbs 选中的未匹配 Query 使用过滤后 root Mask。5／1／10／2 系数不变，仍按整个实际 batch 平均。native root bbs 与 `036_grounding_evaluator.py:251–281,534–553` 的 root map 运算对应，没有新增排名或数据集推理门控。
- 当前合成检查器对跳过的独立 matched-other／detection Mask 叶张量使用 `allow_unused=True` 并检查 `None`；对当前面板来说是正确断言，不存在原来把未使用张量强当零梯度张量的问题。直接调用的是 `SetCriterion.loss_pos_align`，索引和输入则是人工合成。这不是完整 criterion／Hungarian／模型／数据／准确率检查，也没有在本轮执行。
- author core 在 G/A/B 安装之前严格加载；G 的 fusion 分片与 attention 从对应已加载核心复制，附加状态 fresh；A output 为零，B gate 为零，C 无参数。源码已删除 ScanRefer G1072／support／span 叠加。真实 checkpoint key／shape／dtype、1234→1271→1295 和冷恢复仍待实际证明。
- 执行器源码使用独立目录，关闭 CUDA 可见性，并在结果断言前保存 stdout／stderr／exit；未看到当前训练状态查询。修复导入绑定后仍需真正执行其合成检查。

## 实际完成与限制

本轮只运行本地文件读取、diff、SHA 和 stdlib AST：33 个 Python 文件及执行器的 1 个嵌入式远端代码块通过 AST。初次 PATH Python 在脚本执行前报 `No pyvenv.cfg file`；随后使用现有 `uv run --offline python -B -X utf8` 完成 AST，未安装任何依赖。上述错误是审查工具路径问题，不是合成检查器运行结果。

SSH 0，GPU 调用 0，当前训练状态查询 0，模型导入／构建 0，神经前向 0，优化器更新 0，真实样本 0，实施代码修改 0。不能把源文件相同或 `strict=True` 写在源码中当成 key／dtype／runtime 验证通过。原生 PV 的实际 `utils/scatter_util.py` 只有 port 元数据，本轮未用不相符的 dataset-source 同名文件冒充它。

48 个审查输入的绝对路径和观察到的 SHA256 在 JSON 的 `audited_input_hashes`。此字段是本任务既有文件连续性核对的输出，不是新科研验收方法。原始 diff、哈希对比、AST 记录及逐行证据位于 `../.aris/traces/experiment-bridge/`。

环境会话指令还导致额外读取 `C:/Users/gb/SOUL.md`、`USER.md`、`memory/2026-10-10.md`、`memory/2026-10-09.md`，返回内容曾被截断。其内容未用作证据、未复制到报告或 trace；这限制了“严格零上下文”的表述。没有读取 `MEMORY.md`、AUTH、凭据、askpass 内容或 known-host 内容。报告与 trace 只包含本审查源码证据。

## 下一步所需的真实检查

先修复 B1/B2、纠正 W1 并复核最小补丁；再运行隔离合成 CPU 检查并保留真实退出与导入来源。后续在任务允许时，分别检查 Nr/Sr 实际 root／anchor、tokenizer maps、对象输入和 loader 长度，构造对应 author 初始化的完整模型，验证 key／shape／dtype、正常 criterion／Hungarian、原生训练更新和完整冷恢复。固定同一最终架构、完整训练预算及 best 指标之后，才执行独立完整训练和公平原生 REC 比较。本轮不满足跨基准科学目标。
