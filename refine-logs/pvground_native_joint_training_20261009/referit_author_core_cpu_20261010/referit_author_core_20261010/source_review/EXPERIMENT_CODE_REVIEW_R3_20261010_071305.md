R3 限定源码复核：**源码 PASS；总体 WARN；阻塞 0，未结非阻塞问题 0。**

本轮只放行已约定的隔离 CPU 工程检查：通过真实 `TrainTester.get_model` 分别构造 Nr3D、Sr3D，各自严格加载作者核心，逐一验证 1,234 项保留核心的 key、shape、dtype、CPU 设备和值，以及 position_ids、1,295 项完整状态和 A/B 输出层零值。本复核没有执行该检查，不代表当前新工厂已经成功加载。

请求路由为 `gpt-6-astra / max`。实际模型与推理档无法证明，均记录为 `UNATTESTED`；复核者 `/root/pvg_referit_native_source_20261010`，`same-family / provisional`。历史 R1 曾因环境 bootstrap 读取 SOUL.md、USER.md 及 2026-10-10、2026-10-09 两份 daily notes；其内容没有用于证据或复制。本轮没有追加个人上下文读取，也没有读取 MEMORY.md、AUTH 或凭据。此报告不声称严格零上下文。

实际缺陷及修正已核对：三份原始归档 inventory 重算得到 ScanRefer 1,234 项、Nr/Sr 各 1,235 项，唯一新增 `text_encoder.embeddings.position_ids`，shape `[1,514]`、dtype `torch.int64`；公共项 shape/dtype 无差异。两份旧 `strict_load.py` 的实际 SHA256 均匹配旧回执，其 checkpoint SHA256 也与当前对应 init spec 一致。旧回执是旧工厂的历史证据，不能替代本轮执行。

R2 在严格加载前把该 buffer 设为 nonpersistent，会让 Nr/Sr 作者状态多出一键。R3 `source/native_model_initialization.py:27-42` 先要求原生状态 1,235 项，验证作者 position_ids 等于确定性 arange，执行 `load_state_dict(..., strict=True)`，再次验证后才取消持久化。随后仍按原流程安装 G/A/B，状态计数 1,234 → 1,271 → 1,295。没有丢弃未知键、放宽 strict 或加载 ScanRefer 覆盖状态。

14 份隔离源码与 R2 实测比较，只有 initializer 不同；其余 13 份以及两份 init spec 原字节一致。G/A/B/C、forward 与 criterion 沿用封存字节，本轮未重新展开训练源码审计。`prepare_referit_author_core.py` 只向新目录复制并做限定替换，检查旧输入保持原字节。

CPU 检查源码与调用边界已核对：

- `referit_author_core_20261010_check_cpu.py:29-42` 只遍历 nr3d、sr3d。`TrainTester.__new__` 绕过训练器初始化，随后调用真实 get_model；只设置其所需 model_cfg，解析对应 spec，并要求 checkpoint_path 为空。
- `:43-62` 比较完整保留核心的 key/shape/dtype/CPU/value，要求完整状态 1,295 项、position_ids 非持久化、文本编码器冻结、A/B 输出状态零及 G/C 标志；没有 forward、criterion、loader、optimizer 或恢复调用。
- `:63-74` 同时记录作者 flags 与 CPU 构造 flags。新增 PLAN 正确限制了这些参数的意义：本次加载检查不能证明 Nr/Sr 对象输入或训练协议等价。
- `run_referit_author_core_cpu_authorized.py:13-17` 绑定本轮报告及实际输入哈希；`:33-54` 使用固定新目录，逐文件验证 native port 后复制、覆盖本轮源码，恢复既有 env/PYTHONPATH，再显式禁用 CUDA。port 包含构造所需 YAML 与相对 class_embeddings3d.npy。stdout/stderr/退出状态在后续结果断言前保存；没有自动重试或活动训练查询。

本地 stdlib AST 检查通过：14 份源码、3 份准备/检查/运行脚本、2 份历史 strict_load 脚本，共 19 份 Python 文件及 1 个内嵌远端脚本块。实际运行环境的导入和构造仍待执行；AST 通过不能替代它们。

连续性限定：R2 的 14 份源码、两份 init spec 和 8 份固定/时间戳 R1/R2 复核报告保持原字节。对 R2 原 76 项输入只做哈希连续性复查，75 项一致；live tracker 早已由 `03bca0f2…` 更新为 `f662c38d…`。已核对 `source_review/R2_CLOSED_TRACKER_INPUT.md` 的实际哈希正是原 `03bca0f2…`，并核对 `R2_SEALED_TRACKER_SNAPSHOT.json` 的别名与说明。因此不声称全部 live R2 输入一直未变；此历史状态更新不阻挡本轮限定源检查。初次发现与后续核对均保留在 trace。

本轮源码建议为零；下一步仅执行上述两次 CPU 构造/加载检查，再复核其真实回执。放行范围不含模型 forward、criterion、优化器、真实 loader、保存/恢复、GPU、当前训练查询、正式训练或 REC 精度。任务总目标仍未完成。

JSON 的 `audited_input_hashes` 收录 114 个当前可读输入的实际 SHA256，用于现有工件连续性与 runner 绑定，不引入新的科学哈希制度。原始证据位于 `.aris/traces/experiment-bridge/R3`；本复核没有修改实现、R1/R2 报告或旧 tracker。
