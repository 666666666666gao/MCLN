# PV-Ground＋G / G＋P2 实现审查

- 日期：2026-10-02
- reviewer：`/root/pvg_g_p2_review`
- model：`gpt-6-astra`；reasoning_effort：`max`；fork_turns：`none`（主任务已确认实际启动参数）
- review_independence：`same-family`
- acceptance_status：`provisional`
- 结论：**PASS（修复后静态代码审查）**。当前没有未解决的 BLOCKING 问题。
- 本审查没有运行远程 GPU、训练或正式评估，也没有验证实际终态二进制的加载结果。实际 CPU 恢复、batch8 GPU 两步预检、显存容量和正式结果仍须由执行回执确认。

## 审查范围

读取计划 `docs/PVG_G_P2_PLAN_2026-10-02.md`、P2 模型、新源码准备器、单臂入口、串行双臂控制器，以及现有 source/observation/task reader 和 semantic-assignment 模块。对照下载的实际 D/G `pv_ground.py`、`encoder_decoder_layers.py`、`modules.py`、`losses.py`，并核对已有同源 `main_utils.py`、数据集、官方 evaluator、体素处理配置、G spec/receipt/resource/input manifest。

## 已发现并修复的阻断问题

原单臂入口把既有 D/G interface 回执的 `source_port_sha256` 与新 P2 公共源码的 `source_port.json` 比较。新准备器必然生成不同清单，因此原实现会在模型加载前失败；改回旧清单又会与改动后的源码不符。

主任务已作最小修复，并经本 reviewer 复读：`scripts/run_pvground_g_p2.py:68` 起，将既有 interface 校验指向 `spec['parent_source_port']`；新公共源码仍通过 `spec['source_port']` 单独校验。部署 spec 必须按主任务确认的方案分别填写原 D/G 清单和新公共源码清单。本问题已关闭。

## 正确性结论

1. **G 起点恢复正确。** 入口先严格加载作者 1234 项核心，再安装既有 D/C task reader 的 37 项状态；G delta 被要求精确覆盖当前可训练参数及原有 buffers，共 1072 项，逐项检查形状和 dtype，再严格加载。没有继承 G 的 optimizer。P2 只新增 `decoder.5.candidate_evidence_read.*`，不会替换既有 D/G reader 参数。

2. **P2 输入形状、文本 padding 和几何路径相符。** 原 decoder 提供 `[Q,B,288]` query、`[B,L,288]` 文本与 `[B,L]` bool padding mask；P2 的 `[B,Q,L]` logits 按该 mask 屏蔽 padding。六源内容宽度保持 `(128,128,128,128,256,256)`，观测宽度保持 `(10,13,13,13,13,13)`。候选相对坐标为 `[B,Q,K,3]`，与候选尺度归一化坐标、扩展的观测状态组成各源 `6+n` 维位置输入。候选中心和尺寸取上一 decoder 层预测，沿用原生 detach；模型路径没有读取 GT 几何。

3. **插入位置保持 D/G 语义与几何分流。** P2 的共享实例残差加到既有两个 task residual 后，继续调用原 `finish_task_queries`、共享 dropout、norm、FFN 和原生 prediction heads。零初始化输出层使初始残差为零。没有新排名器、教师、P3、TGS 改动或旧 MCLN 模块移植。

4. **原生 loss 与 G 替换一致。** 实际 criterion 的七次 matcher 调用顺序是 `proposal_`、`last_`、`0head_` 至 `4head_`，因此入口使用 `matching[1]` 正确。G correction 仅替换末层未匹配且 root IoU 大于 0.5 的 eos token CE，保留 eos 权重 0.1，并使用 ScanRefer 原生 `0.5/7` 总损失系数；资格框和目标均 detach。匹配、box/giou、mask、contrastive 等其余目标保持原样。原生源码中的 root token 权重与 correction 一致。

5. **同预算和随机性控制合理。** 两臂在正式 fit 前重置 seed2027，使用同一个显式 seeded DataLoader、两个 workers、相同增强和 batch_size8；`drop_last=False` 配合 row Counter 检查保证 29778 行消费一次，共 3723 更新，末批为两行。P2 forward 不产生额外随机抽样，初始化消耗的 RNG 在 fit 前被重置。两臂使用新 AdamW，core/backbone/P2 LR 均为 1e-5，weight_decay 5e-4、clip 0.1，冻结文本参数保持冻结。不同 P2 激活造成的真实优化差异属于待比较的方法差异。

6. **预检覆盖所要求的执行路径。** 同一 P2 模型分别启用和暂时禁用新模块，完整 forward 前重置 RNG，以对齐原生 Gumbel 抽样；比较末层语义分数、中心和尺寸。随后重置 G＋零残差初始状态，实际执行两次 batch8 反向/更新，并检查输出层梯度以及第二步的文本条件、位置分支梯度。BytesIO 序列化包含全部可变模型状态和 optimizer，重新严格加载模型及 optimizer 后核对模型值和 step=2。此逻辑正确；实际运行结果尚未由本审查获得。

7. **留出与正式评估路径正确。** fit/holdout 使用既有 29778/6887 行划分，并验证物理场景不相交。formal 分支严格恢复本臂 3723 步 terminal，重新建立 ScanRefer `split='val'` 数据集并要求完整 9508 行；detect_intermediate、butd、颜色、height、batch 和 workers 与既有正式入口一致。IoU 使用数据集 `center_label`、`size_gts` 和 `gt_masks`，同时与官方 evaluator 汇总交叉核对；没有拿另一模型的输出作标签。主指标为既定 bbs，bbf 为原有并列记录，没有新增部署选择器。

8. **双臂串行控制器满足无成绩筛选的要求。** `scripts/run_pvground_g_p2_pair.py` 要求实际两步预检 PASS，按 G train → G＋P2 train → G formal → G＋P2 formal 执行，使用已有 GPU lock；没有根据 holdout 涨跌选择正式参评模型。每次训练前按预检实测序列化大小预留两份原子替换空间及日志空间。训练仅保留一份 latest，结束后改名 terminal；未新增额外终态副本。

## 实际完成的本地验证

- 八个审查范围内的 Python 文件通过内存 `compile`，未写入 pyc。
- 从准备器 AST 读取实际四处替换常量，在下载的真实 D/G 源码中逐一确认唯一命中；替换后两份源码均通过内存 `compile`。
- 核对已有同源 optimizer/criterion 和官方 evaluator 的实际代码；确认控制器所需 `PYTHONPATH` 与 GPU lock 在现有 env spec 中存在。

没有新增测试、保护框架、fallback 或无关改动。审查只写本报告。后续按既定流程完成 CPU 严格恢复及真实 GPU 预检后，再运行双臂完整预算与两端正式评估；本报告不构成精度或显存已通过的声明。
