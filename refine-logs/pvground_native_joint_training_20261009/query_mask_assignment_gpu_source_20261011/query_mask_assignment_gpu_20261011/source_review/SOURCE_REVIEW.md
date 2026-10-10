# Query Mask 真实批次 GPU 预检准备：SOURCE_ONLY 审查

结论：**WARN，0 个源码阻塞项**。本轮限定修改的静态正确性检查为 PASS；WARN 保留未执行真实 GPU 预检、未验证运行时输入一致性和恢复结果的边界。未发现需要补丁的实际源码缺陷。

- `execution_scope: SOURCE_ONLY`
- `review_independence: same-family`；`acceptance_status: provisional`
- 请求的审查路由：`gpt-6-astra / max`；实际模型与 effort 身份：`UNATTESTED`，不据请求参数声称身份已验证。
- 仅读本地输入和历史审查记录，运行标准库字节比较、SHA256 与 AST 解析；没有导入项目、运行 CPU/GPU 检查、模型前向、criterion、数据 loader、反向、optimizer、SSH、网络或训练查询。
- 只写本目录报告及 `trace/`。未修改任何所审输入、活动源、protocol 准入状态、控制器或观察器。

## 静态核验

1. **来源与可重建性正确。** GPU 目录的 14 份源与 `query_mask_assignment_20261011/source` 逐字节一致；两份 manifest、两份 protocol 的 source map、旧 SOURCE_ONLY 审查的 14 个源 pin 全部一致。CPU 实际审查中的 losses pin 与 summary→actual-review 绑定亦一致。独立从准备脚本的 AST 提取文字替换，重建出的 helper 与所审 helper 字节完全相同。原 `native_c_off_preflight.py` 的 **60 个断言全部保留**，现 helper 共 71 个；`group`、`observed_loss`、`TwoNativeBatches` 的 AST 不变。未重新审查或运行整个历史 GPU 实验。

2. **共同 protocol 与门禁正确。** `text.json` / `query.json` 的唯一差异是 `matcher_mask_source`。共同 arguments 与原 C-off protocol 完全一致：seed2027、B8、256 queries、LR 1e-6 / 1e-6 / 1e-5、weight decay 0.0005、clip 0.1；没有 debug、joint_det、detect_intermediate 或 augment_det。init 与原 C-off init 字节一致，G 开启、C 关闭。两份 `serial_gpu_preflight_admitted=false`；helper 第 26 行断言先于第 31 行输出目录创建及第 35 行 Torch 导入。准备脚本只生成本地文件，没有部署器、GPU 控制器或排队动作。14 份局部源并非完整可运行 PV 仓库，计划已明确要求另行绑定完整运行时。

3. **观察器保持实际训练匹配。** helper 第 155–198 行先取得配置来源的原 matcher 结果；没有 `pred_masks` 的输出直接返回它。原 `compute_hungarian_loss` 当前仅 `last_` 带 Mask，六个其余 prefix 不做额外比较；辅助 matcher 调用仍注释。`observed_loss` 先递增当前 step，再进入原 criterion，因此末层记录的 step 是 1、2，且末尾断言共两条。静态预期每种模式 14 次配置 matcher 调用、2 次替代来源诊断调用；这不是已观测的运行计数。

4. **索引、设备和 GT 配对正确。** 原 matcher 返回 CPU int64 索引，沿用原 native loss 允许的高级索引方式读取 GPU 张量；`superpoints` 与 Mask 张量处于同一模型输入设备，供 gather 使用。两个来源各自按 GT 索引排序后比较 Query，并核对唯一 Query／GT 数量覆盖全部有效 GT。IoU 对排序后的最终 box 与对应 GT box 取对角线，Mask gather 使用同一原始点→superpoint 映射。记录的 `gt` 是过滤后 target 列表的位置，未把非 root GT 映射到槽 0。排序仅重绑局部变量，返回值仍为原 `configured` 列表，未改写输出、GT、损失或原匹配结果。

5. **成本诊断命名和公式正确。** 原 matcher cost 保留 class=1、bbox=0、GIoU=2、Mask=0.0002，且 helper 明确核对。`own_mask_intersection/union` 与 `own_query_mask_cost` 始终使用被匹配 Query 自身的 Mask；`matching_mask_source` 与 `matching_mask_cost` 按 text/query 分支使用实际匹配来源。原 `_to_gpu` 已断言 GT Mask 为二值；`>0`、gather、0.0002 加权不一致点数与原 Mask L1 项语义一致。Text 控制的 own-query 诊断没有冒充 Text matcher 实际 Mask 成本。替代 matcher 和诊断均为 no_grad，构造与观察逻辑不消耗 RNG。

6. **正常训练和恢复检查保留。** helper 使用同一个真实 train loader 前两个增强 batch，共 16 行，经原 `train_one_epoch`、完整 native criterion、反向、clip、AdamW 与 scheduler 更新。保留原 core/backbone/G/A/B 参数组、RoBERTa 参数冻结、G 开启、C 附加监督缺席的检查。初态要求 1295 个状态与保留 E0 逐项相同，仅在 architecture 比较中将 C 标志改为关闭；E0 的历史 C 适配已披露。之后检查梯度、各组参数变化、每项 Adam 状态 step=2、完整模型/optimizer/scheduler/Python/NumPy/Torch/CUDA RNG 保存及对象重建后的恢复。

7. **匹配模式恢复表述准确。** 显式 protocol→CLI 参数在 helper 第 45 行传入，factory 使用它构建 matcher；第 278 行核对 checkpoint config，恢复后第 293–294 行重新调用 `get_criterion` 并核对保存模式。`load_checkpoint` 本身不覆盖 CLI，这一限制已在计划中说明。这里是同进程销毁并重建模型与 optimizer 的恢复检查定义；没有执行跨进程恢复或下一步数值连续性验证。

## 非阻塞限制与后续边界

- 旧 CPU 记录仅支持构造输入上的 33 次 matcher、44 个 assignment problem、66 对 GT 处理和 6 次 boxes-only criterion/backward；本轮只读其有限结论，不将其升级为真实 loader 或完整 GPU criterion 成功。
- 相同 seed、sampler epoch 与 generator seed 支持相同输入规则。当前报告未比较两次真实增强输入；helper receipt 仅记录 step/batch_row 与匹配诊断，不能据此声称已证明跨模式输入逐项相等。计划明确将实际一致性留给后续真实执行核对。本轮不要求为此增加猜测性兼容层或额外 hash 机制。
- 末层替代来源是在各自训练轨迹的同一输出上做局部诊断；第二步的两个模式已可能具有不同参数，不能把跨模式差异归结为同一固定模型上的纯 matcher 变化。
- 诊断使用训练增强输入，不是 9508 条正式精度、候选身份判定或独立贡献评价。Mask 成本的诊断公式与原成本语义一致，不声明浮点逐位重算验证。
- 当前准备不是作者完整混合 ScanNet 检测配方，未验证检测混合行、Nr3D 或 Sr3D。没有证明三项有效贡献、训练增益或全目标完成。
- 必须先等待当前 C-off 三轮终态并判断，再独立决定 GPU 预检准入。本报告不授予 GPU、正常训练或最终方法准入。原观察器 52851 / PID 51540、下一次观察约定 2026-10-11 05:00:40.968644 +08:00 仅作为任务约束保留；本轮没有查询其活动状态。

不需要源码补丁。后续真实预检应依据现有计划先完成准入，再对实际输入、两步输出和完整恢复记录做实际执行审查。

全部 46 个审查输入的绝对路径与纯 SHA256 在 `SOURCE_REVIEW.json`。字节比较、原断言保留与模板重建证据见 `trace/VERIFICATION.json`、`trace/PREFLIGHT_DIFF.txt`。标准库检查首次因 PATH 中的 Python 缺少 `pyvenv.cfg` 在解释器启动阶段失败；使用已安装 CPython 3.11.14 的绝对路径后通过，未安装环境或运行项目。
