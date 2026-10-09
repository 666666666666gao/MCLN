# Nr3D／Sr3D 原生源码复核 R2

**限定源码结论 PASS；总体 WARN，0 个阻塞项、0 个未解决代码／文档问题。** WARN 表示真实 CPU 检查及原生模型运行仍未执行，不表示已发现新的代码缺陷。源码放行范围仅为已经授权的隔离合成 CPU 目标／梯度工程检查，不包括完整 PV 模型、GPU、训练、活动训练查询或结果验收。

本次是原 R1 审查者的阻塞修复复核。请求路由 `gpt-6-astra`／`max`，实际模型及 effort 均 **UNATTESTED**；`review_independence: same-family`，`acceptance_status: provisional`。

## R1 问题已在源码层关闭

| 问题 | 实际修正与证据 | 结论 |
|---|---|---|
| B1：ScanRefer 分母及 best 规则残留 | `train_dist_mod.py:217` 在 tqdm 包装前取得 `len(test_loader.dataset)`；`:254–257` 核对两个真实 evaluator 分母并返回该值。`main_utils.py:167–176` 改为 `(hits050, hits025)`，记录处不再要求 9508。 | CLOSED_SOURCE_ONLY |
| B2：CPU 导入缺少已验证环境 | runner `:45–50` 将现有 runtime spec 的 env（包括 PYTHONPATH）传给子进程，然后显式关闭 CUDA；checker `:13–15` 用 `inspect.getfile(SetCriterion)` 核对实际原生来源。runner `:13–17` 指向独立 R2 报告。 | CLOSED_SOURCE_ONLY |
| W1：fresh B 初值写成 0.5 | 两份计划均改为零输出经 clamp 得到 gate 0、初始采用 Mask reference；mixer 计算文件与 R1／当前 ScanRefer 来源逐字节一致。 | CLOSED_SOURCE_ONLY |

进度表的旧“5 文件”文字也在封存前已改为“6 文件”，并重新读取核对。当前源准备固定 JSON 与 `SOURCE_ADAPTER_PREPARATION_R2.json` 完全相同。计划明确：strict REC 后 wide REC 是同一 best 的选择规则，仍报告固定终点；该规则没有取代两项相对对应 baseline 的科学目标。

## 实際差异与保留范围

14 个当前准备文件全部读取并做 AST／SHA 对比。相对 R1，仅 `main_utils.py` 和 `train_dist_mod.py` 两个计算源码改变；其余 12 个相同。相对当前 ScanRefer 来源，实际恰好有 6 个变更：main_utils、train_dist_mod、models/losses、native_model_initialization、G、C。所有 14 个当前 ScanRefer 来源哈希仍等于 R1 所审与原 manifest，未被本复核修改；没有查询远端活动状态。

CPU runner 和 checker 的变化也逐项与 `revision1_before_metric_correction/` 对比：环境绑定、R2 报告路径及 inspect 来源核对均已落实；原始 stdout／stderr／exit 仍在结果断言之前保存。手工 indices 和 toy 输入仍明确标为合成，调用的是原生 `loss_pos_align`，没有把它包装成完整 criterion、Hungarian、数据或准确率检查。

G/C 数学与保护逻辑相对 R1 无改变：Nr／Sr 原生权重、6 层分母、真实 annotation detection 排除、全部 matched Query 保护和 C 实际 batch 预算保留。当前合成面板的 `allow_unused=True`／`None` 断言保持正确。对应作者 core 的 fresh G/A/B 初始化路径和两份 init spec 也未改变，没有加载 ScanRefer G1072、support 或 span 状态。forward、模块、native bbs 运算均未改动，没有新增排名、数据集推理门控、fallback 或无关重构。

## 验证范围与保留记录

使用已存在的 `uv run --offline python -B -X utf8`，只做文件读取、diff、SHA 和 stdlib AST。49 个 Python 文件及 1 个嵌入远端代码块解析通过；未导入实施模型、未执行 loss 或合成检查。SSH、GPU、活动训练查询、神经前向、优化器更新及实施源码修改均为 0。

76 个实际审查输入的路径与观察哈希在 R2 JSON 的 `audited_input_hashes`。原 R1 固定报告、时间戳报告、原始 trace 和第一版快照均保留；新证据单独写入 `../.aris/traces/experiment-bridge/R2/`，没有覆盖 R1。原固定 JSON SHA 为 `e00e51fff481624376633f3058f213c2decd269ca02e15f130873d48f07d32f8`，MD SHA 为 `5895e7c7a4e34aae7c62c1487b79632b3f8d251230d9ebe979ff9ffe403a8227`。

R2 没有额外 bootstrap 读取。R1 会话曾按环境指令读取 SOUL、USER 和 2026-10-10／09 daily memory，且返回曾截断；这一上下文限制继续披露，其内容没有用作审查证据或复制到报告。没有读取 MEMORY.md、AUTH、凭据、askpass 或 known-host 内容。不能宣称严格零上下文或已证明实际 Astra/max 身份。

后续仍须取得真实合成 CPU 检查回执；再按任务安排验证 Nr/Sr 真数据与 loader、作者状态 key／shape／dtype、完整模型构造、criterion／Hungarian、正常训练及冷恢复。完整跨基准训练和公平原生评价仍未完成，本报告不能代替这些证据。
