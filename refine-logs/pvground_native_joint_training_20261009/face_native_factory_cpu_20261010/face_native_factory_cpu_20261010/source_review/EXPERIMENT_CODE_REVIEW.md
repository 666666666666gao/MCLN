结论：**PASS，0 个阻断问题，0 个非阻断缺陷。** 本结论仅通过 `PLAN.md` 所定义的一次 CPU 子进程、两次真实 `TrainTester.get_model` 构建检查的源码门；本审查没有执行该检查，不构成 GPU 或正式训练准入。

请求路由为 `gpt-6-astra / max`，实际模型与推理强度没有宿主独立凭据，均记为 `UNATTESTED`。`review_independence: same-family`、`acceptance_status: provisional`、`execution_scope: SOURCE_ONLY`。以下 `F` 为本报告父目录，`S` 为 `F.parent/source`。只新增审查报告及私有审查记录；未读取认证 helper、known-host 内容或私有执行 stderr，未查询当前训练。

1. **改动与导入路径正确。** 原始字节比较确认，`F/source/train_dist_mod.py` 仅将 `S/train_dist_mod.py:128` 的 import 和第 129 行的返回调用替换为 face 初始化扩展；其他字节不变。隔离 source 目录恰好三份 Python 文件，两份模块与先前已审 CPU 准备版本逐字节一致。checker 第 16–30 行把隔离目录和既有 warm 源放在搜索路径前两位，并在未来执行时断言 factory、扩展、原生初始化与 parser 的来源。`models/__init__.py` 导出真实 `PVGround`，`S/models/pv_ground.py:161` 使用原始 PV backbone；没有 fake model 或替代网络。借用现有环境内的 PointNet/OpenPCDet 等依赖仍受原环境合同约束。本次确认的是源码路由及保存的既有记录，不冒充新 CPU 的实际 import receipt。

2. **初始化与 E0 身份/schema 一致。** 两份配置除 `face_residual_mode` 外完全相同，均固定 ScanRefer、seed 2027、extremal prior；普通训练参数均存在于原 parser，且不带 resume/pointnet checkpoint 参数。扩展先完整调用原生初始化，保留 official 严格加载、G state-delta 合并、A/B 严格加载的 1234→1271→1295 流程。`NORMAL_E0_IDENTITY.json` 与已保存 normal summary 和 cold-recovery 的 best 身份相同：615023752 字节、epoch 0、1295 states、9508 行及 5677/4920 命中。远端 wrapper 在 child 前流式核对该权重身份；checker 第 34–38 行再检查 payload 字段。这里引用命中数仅为父 checkpoint 身份核对，没有评估新候选精度。

3. **状态、对象与可训练性检查覆盖限定目标。** 扩展第 16–20 行把原先的 prior 对象放入 `axis_prior` 并断言对象相同。checker 第 55–63 行只对 `candidate_span_mixer.` 前缀显式改名，逐项检查全部 1295 个保留 tensor 的 shape、dtype、CPU device 与 `torch.equal`；配合总数 1301 和新增六项 face-residual state 检查，覆盖完整 state 集。参数公式为 23425 新参数、29793 prior 参数、合计 53218。输出层 weight/bias 全零、face 与 prior 全部可训练、原生 text encoder 冻结均有结果写入前的实际断言；原生初始化还保留原 trainable core 检查。本检查的 `face_output_zero` 指输出层零初始化，未执行 forward；实际零输出框等价性属于此前独立合成 CPU 证据。

4. **两模式比较公平且声明有限。** 每次构建前重置 Torch seed 2027；两个配置、模型结构和参数构造顺序相同，模式差异只保存为布尔开关。第一次完整 state 被 clone，第二次对 1301 项全部使用 `torch.equal`，不会因为两个 state 字典共享模型存储而得到空洞相等。隐藏模式只屏蔽额外 face head 的 tokens/fraction，旧 axis prior 仍共同使用来源信息；模块文档与架构说明正确限定此控制范围。保留 prior 证据键的 wrapper 输出与现有 PV forward 读取接口一致，但真实 forward/criterion/梯度仍未由本检查验证。

5. **CPU 与所有权边界正确。** child 固定空 `CUDA_VISIBLE_DEVICES`、OMP/MKL 线程 1，既有环境的 OpenBLAS 线程亦为 1；checker 导入 Torch 前断言可见设备为空，再设 Torch CPU 线程 1，并检查 CUDA 未初始化。使用 `TrainTester.__new__` 避开 logger、数据处理器和训练初始化，仅设置工厂所需 `model_cfg`。没有调用 loader、forward、criterion、optimizer、DDP、训练入口或权重保存。两个 Python 启动均有 `-B`，显式输出均位于新 CPU 根或本地独立 attempt。完整 E0 payload 和第一份 state clone 会留在内存，构建结束后删除模型/state 引用并收集；没有 forward 计算图或两个完整模型的有意并存。本审查未测量峰值 RAM，也没有现有证据表明这一顺序构建存在内存/所有权阻断缺陷。冻结权重、配置和 class embedding 的只读加载不等同于数据集/loader 构造。

6. **传输与失败记录保持既有约束。** launcher 先要求 SOURCE_ONLY PASS/WARN、零 blocker 和审查输入原始 SHA 全匹配，再要求本地 attempt 不存在。远端核对现有 env canonical SHA、port 的全部 116 个既有文件和 E0 身份后，要求专属固定 CPU 根不存在才写入三份隔离源与检查配置。只有一次 SSH 和一次 child 调用；没有 retry、fallback、当前训练查询、停止/重启或源覆盖路径。现有严格 host-key、固定已协商算法及无空格 askpass/known-host 字面路径保留；密码不进入待审 argv/JSON/打印内容。失败保留原始 stdout/stderr 和 exit 记录，不因等待超时自动重跑。结果中 GPU/查询/步骤为零属于该程序范围声明，并非全机硬件监测统计。

7. **静态核对通过，实际结果仍待产生。** 本次 42 项确定性核对全部通过，19 份 Python 源及一份内嵌远端程序通过 Python 3.7 AST 解析；12 份已读 warm 源/配置与 port 的既有 SHA 和大小匹配。39 个实际审查输入均以解析后的绝对路径和原始文件 SHA256 写入配套 JSON，封存前再次核对未变。全量请求、报告响应、原始输入快照、两行 diff 与检查记录仅位于 `source_review/private`。没有新增实现修补、兼容层或哈希方案。

`blocking_findings: []`，`non_blocking_findings: []`。不支持的结论包括：新 CPU 工厂已实际通过、1301-state 完整 checkpoint/optimizer 恢复、真实场景或 criterion 执行、GPU 梯度/训练、REC 精度增益或整体研究目标完成。未来限定 CPU 执行仍须通过已有运行时断言并审查实际结果；当前 C-off 训练及其观察器/顺序不在本次审查中变更。
