本次初始审查结论：**WARN，阻断项 0；非阻断测量缺口 1。** 已直接审阅生成的 runner、检查模块、两份实际 spec、复制依赖及封存的原生 PV／criterion 调用路径。就限定的工程预检而言，模型装配、Mask 范围读入、最终框损失路由、分支梯度隔离与内存重载代码是正确的；没有发现会确定性阻断首次完整模型预检的源码缺陷。尚未执行新预检，不能把本结论写成运行通过。

审查范围为 `SOURCE_ONLY`。审查者为新建原生 Codex 代理 `/root/pvg_whole_model_preflight_review`；父代理确认启动参数为 `gpt-6-astra`、`max`、`fork_turns=none`。工具未单独证明后端 SKU。归属为 `review_independence: same-family`、`acceptance_status: provisional`，不宣称跨模型家族验收。审查者未修改任何实现。

**N1：峰值显存及耗时没有覆盖计划所称的完整流程（非阻断，中等）。**

[runner 第 353 行](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/run_whole_mask_preflight.py:353)开始的有精修器完整前向、两次缓存输入重放、无精修器完整前向及成员映射检查，都发生在[第 409 行](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/run_whole_mask_preflight.py:409)的 `torch.cuda.reset_peak_memory_stats()` 之前。之后读出的 `peak_allocated_bytes` 不能作为明确覆盖这些阶段的全流程峰值；CUDA 缓存可能保留较高 reserved 值，也不能代替完整阶段的统计范围。现有 `steps[*].seconds` 从数据准备计时，涵盖 criterion、多次梯度诊断、反传和更新，属于带诊断的步骤总耗时。起点完整前向及整个预检没有单独计时。

这不会使模型和梯度检查本身失效。若要完成计划中的测量，应将峰值统计起点放到纳入测量的首个 GPU 阶段之前，并记录同步后的起点前向及全程耗时；已有步骤时间应继续明确标作步骤总耗时。无需改变模型、损失、优化器或科学协议。本报告不对实现打补丁。

**已核对的正确实现。**

- 真实工厂先严格加载 1234 个官方状态，处理已知的非持久化 `position_ids`，再装入 37 状态／923616 参数的旧任务读取器；原 G 的 1072 个 delta 键、形状和 dtype 都有严格断言，随后才装新头。未打开任何权重文件。
- 新头的五个 Linear 层对应十个持久状态、400614 参数：输入合计 1302 维，为 288 Query＋896 局部池化＋3 粗框布局尺寸＋6 场景归一化粗框条件＋109 完整范围信息。输出层保持零初始化。
- 封存原生源码实际产生 Text Mask `[1,256,S]`、Query Mask `[256,S]` 和每样本标量 alpha。局部及完整范围分支均使用 `alpha*Text+(1-alpha)*Query` 的 logit 融合。完整范围用实际增强后的 50000 点统计，并以原生 SP ID 索引 Mask 槽位，未误用压缩后的 `np.unique` 序号。全部 256 候选保留。
- 精修在原生 Mask 生成后调用，覆盖 `last_center/last_pred_size`。原生 criterion 从这两个最终张量组装 `last_` 框，并使用真实数据 GT。七次原生 Hungarian 的次序为 `proposal_,last_,0head_...4head_`，所以 `matching[1]` 用于 G 替换是正确的。
- G 资格基于脱离梯度的最终框与真实根目标、IoU>.5 且未匹配槽位；替换公式保留原生 ScanRefer 权重和 .5/7 系数。直接语义／G 梯度及八项原生 Mask 损失到新头的梯度检查与实际计算图一致；Mask 项逐项检查。
- 捕获实际前向传给新头的同一组参数后，在同一零输出头上切换完整范围开／关，检查两组结果及粗框完全相同；然后恢复所选条件。另一次重置 RNG 的无新头原生前向和完整调用顺序也会被检查。
- `whole_range_loss_routes` 先求定位项对完整范围张量的梯度，再以脱离梯度的上游梯度求该张量到 Text／Query／alpha 的 VJP，能隔离新增范围通路。关闭条件要求该张量不在定位图内；第二次前反向（已经历一次更新）要求三组非零梯度，允许零头第一步的零梯度。
- 实际优化器工厂是 AdamW。内存序列化后严格检查全模型持久状态、所有已有 optimizer state 的 step=2，以及参数组、状态 ID、各状态键和全部标量／张量值，包括两类 moments。序列化目标为 `BytesIO`。
- 接口只允许 `cpu/preflight`。实际生成文件已截去评估和 3723 步正式拟合／磁盘 checkpoint 写入部分；残留 `if formal` 分支对允许的 CLI 不可达。没有新增正式训练或权重文件输出路径。
- 人工源码检查未发现新增 Python 3.7／Torch 1.10 不兼容语法或 API。此项不代表导入、编译或环境测试通过。

**结论的适用范围。**

原 standalone CPU receipt 明确使用构造的 Query、Mask、粗框和目标，未运行 PVGround／CUDA；旧 tail_fused preflight 属于旧头。它们都不能替代这次新入口的实际运行证据。即使未来通过，新预检也只覆盖一个 batch8 和两次更新，不构成完整数据精度、收敛、正式训练或 9508 表达评估结果。

同为 400614 参数不代表有效功能容量相同；局部对照将 109 个输入置零。直接语义／Mask 损失到新头梯度为零，也不代表训练后语义或 Mask 不变：共享上游会更新，框变化也可改变离散 Hungarian／G 资格。缓存重放的精确性只限定同输入零头；不证明跨进程或全 6887 行逐位一致。

分支梯度诊断使用原生 bbox＋GIoU 的未加权和，并汇总每组梯度范数；实际更新仍使用完整原生加权损失＋G。它证明通路存在，不能证明所有样本／Query 都有非零梯度或能提高精度。成员映射的运行时显式比对仅选择每个样本的 Query 0、42、255。内存重载不证明磁盘恢复、RNG 恢复或恢复后继续训练等价。

`verify_native_replacement` 的更强 CE 重建／梯度见证仍在 `update=False` 分支，而本入口两次调用均为 `update=True`。替换公式源码正确，但不能称新预检会重新执行那个可选见证。

正式共同父权重和损失选择仍待当前归一化结果分析。本报告不查询或修改当前训练，不代替部署／资源安排，也不授权正式训练。

**源码身份及未执行项。**

生成 runner：`27288` 字节，SHA256 `95f31c2ec5d87de86d9ada8fe600f37a7b01d036617bc4fe6763c7d737109f11`，与准备记录一致。九个复制模块均与其记录来源的字节数及 SHA256 一致。封存原生 PV 为 `cc17e6c235075c4be24c752bb76dcafe0273c7a036f3dcf6bedb3696379aaa58`，losses 为 `920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de`，与旧 imports／source_port 相符。Dataset 的单独 namespace 路径与旧 imports 相符，没有将它误认为模型目录中的 Dataset。

完整 38 项文本／源码的绝对路径、字节数、SHA256 和实际阅读范围保存在 [EXPERIMENT_CODE_REVIEW.json](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/EXPERIMENT_CODE_REVIEW.json) 的 `reviewed_files`。并行新增的发布／请求记录脚本不在本次审查范围内。

未运行准备器、runner、模型／helper 导入、AST／编译、任何测试、新 CPU 模式、CUDA 预检、前反向或更新；未加载权重／二进制 fixture，未使用 SSH／GPU、查询作业或凭据，未启动 launcher、正式训练、评估、发布或其他代理。仅进行了文本／源码读取、现有文件身份核对及本报告写入。准备记录中的 `local_AST_pass` 是生产者历史记录，不是本审查新增的执行证据。

