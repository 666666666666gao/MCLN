本次计时／显存补丁复核结论：**PASS，阻断项 0，未关闭的非阻断项 0。原 N1 已在源码层面关闭。** 这是首次审查者对精确补丁的续审，不是第二次独立新审查。新完整模型预检仍未执行。

审查范围继续为 `SOURCE_ONLY`，归属为 `same-family / provisional`。原生代理启动配置由父代理确认为 `gpt-6-astra / max / fork_turns=none`；没有独立后端 SKU 证明。审查者未导入、运行或修改实现。

**原 N1 的关闭依据。**

[runner 第 94 行](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/run_whole_mask_preflight.py:94)现在仅在 preflight 模式重置一次 CUDA allocator 峰值，位置早于原生工厂导入／构建及模型放入 CUDA；两步更新前的旧重置已移除。原生有精修器起点前向、缓存开／关两次头重放、原生无精修器前向及两次更新中的完整前向，都在开始计时前和工作完成后同步 CUDA，再以 `perf_counter` 计算墙钟耗时。最后记录从 `main()` 进入到模型／优化器重载核验结束的时间。[builder 第 84 行](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/prepare_preflight.py:84)起的对应替换与实际生成 runner 一致。

对照封存原文，新增内容限于这些计时、同步、峰值统计及记录字段。原 38 项已审阅文本中，只有 runner 与 builder 的字节／SHA256 改变；其余 36 项，包括两份 spec、模型、损失和所有 helper 均保持一致。因此 N1 不再需要补丁，未发现新增确定性阻断缺陷。

**计量字段的准确范围。**

| 字段 | 实际范围 |
|---|---|
| `peak_allocated_bytes / peak_reserved_bytes` | 当前 CUDA 设备的 PyTorch allocator 峰值，覆盖工厂之前的重置至内存重载核验，包含起点前向、缓存重放、映射检查及两步诊断更新；不等于 NVML 的进程／整卡显存或 allocator 外部占用。 |
| `initial_full_forward_seconds`、`native_without_refiner_forward_seconds`、`steps[*].full_model_forward_seconds` | 同步后的 `observed_forward` 墙钟耗时，包含模型、CPU 几何统计及 Python 调用记录；不包含此前输入准备或其后 criterion／反传／更新，不是纯 GPU 内核时间。 |
| `cached_information_pair_replay_seconds` | False、True 两次头部重放及条件恢复的合计时间；不包含上游 PV 前向，也不包含随后精确性断言。 |
| `steps[*].seconds` | 原有步骤总耗时：准备、前向、criterion、梯度诊断、反传、裁剪、更新及同步；外层额外梯度范数／断言不在该区间。 |
| `runner_wall_seconds_through_restore` | 从进入 `main()` 到重载核验后的墙钟时间，包含参数／来源检查、加载、工厂、数据准备及各预检阶段；不含 Python 进程启动和最终 receipt 写盘／打印。 |

这些字段尚无新测量值，不能据此宣称吞吐、性能、整卡显存、精度或跨组耗时等价。

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

**版本与保留记录。**

当前 runner 为 `28513` 字节，SHA256 `485a8e04fecbcb83a10beee73af52e46d5fee206884eaae1b148820003699bff`；builder 为 `10677` 字节，SHA256 `5a8e103b624fa84332f236ef6c2830fa177022a3cf40520ad8db30506a2c66b4`。[instrumentation_patch.json](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/instrumentation_patch.json) 为 `778` 字节，SHA256 `12da07f527d0364e3c112b5584369f55a6495600209a123753a9fde3029b02b9`，准确连接补丁前后两版。

`source_preparation.json` 保留原 runner 的 `95f31c2e...109f11` 身份，这是原始准备凭证；本报告采用补丁 receipt 对应的新身份，没有将旧记录冒充新记录。首次 WARN 报告及原 runner／builder 已在 [原审查记录目录](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/.aris/traces/experiment-bridge/2026-10-03_whole_model_preflight_run01)封存，字节／SHA256 与首次审查一致，本次未改动。

全部 43 项已审阅文本的绝对路径、字节数、SHA256、实际阅读范围，以及已关闭 N1、逐项源码核对与计量范围，保存在 [EXPERIMENT_CODE_REVIEW.json](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/EXPERIMENT_CODE_REVIEW.json)。原有源码正确性结论与适用范围全部保留；尤其 `verify_native_replacement` 仍不会被两个 `update=True` 步骤执行。

本审查仅读取源码／文本、核对身份并写两份最新报告。未运行准备器、patcher、runner、模型／helper 导入、AST／编译或测试；未加载二进制 fixture／权重，未使用 SSH／GPU、查询作业／凭据、启动其他代理或部署。生产者报告的局部 AST 解析是其准备过程记录，不是本审查新增的运行证明；实际模型、CUDA、内存／时间指标及精度仍待执行。
