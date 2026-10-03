# 完整 Mask 成员范围表示：首次执行前源码审查

审查日期：2026-10-03。结论：**FAIL**。存在 **1 个执行阻断问题**；核心成员统计、点数加权和原生 scalar-alpha logit 融合公式正确。另有 **1 个非阻断的参考独立性限制**。修正阻断问题并完成补丁复查之前，不应运行当前启动器。

本审查为 `SOURCE_ONLY`，由新建 Codex 审查代理直接阅读指定文件完成；`review_independence=same-family`，`acceptance_status=provisional`。这是同系列模型的独立上下文审查，不是跨模型系列验收。审查代理未执行实现、导入实现模块、加载 fixture 二进制、发起 SSH、查询 GPU 或活动训练、读取认证包装器或凭据。只写本目录的两份审查报告。以下运行行为均为源码推导，不是已发生的运行结果。

阅读范围为本目录的 `CODE_REVIEW_REQUEST.txt`、`CPU_WITNESS_PLAN.md`、三个实现文件、`fixture_inventory.json`、`fixture_schema.json`，以及指定的 `normalized_spec.json` 和两个已导入的原生源码快照。完整路径见 JSON 的 `reviewed_files`。审查遵循 experiment-bridge 的首次部署前审查要求及其同系列、临时验收归属规则。

## 阻断问题 B1：正常导入产生的字节码目录会使成功核验被启动器拒绝

- 位置：`run_cpu_witness_authorized.py:30-31,47-48`；触发导入位于 `cpu_witness.py:11`。
- 证据：启动器创建一个新的可写目录，上传 `whole_mask_range.py` 和 `cpu_witness.py`，然后只以 `python -u` 运行后者。后者导入前者。启动器未传 `-B`，也未设置禁止写字节码的环境变量。正常 Python 导入会在该新目录创建 `__pycache__`。
- 后果：即使所有数学断言通过、远端 `receipt.json` 已写入且远端进程退出码为 0，行 47 仍要求目录恰好只有两个 `.py` 和 `receipt.json`，因而拒绝额外的 `__pycache__`。行 48 的本地 `cpu_receipt.json` 和后续 `cpu_execution.json` 不会生成。这是当前启动路径与自身输出契约的冲突，不需要异常数据或恶意用户才能触发。未读取远端环境，不能假定存在外部配置替当前命令关闭字节码写入。
- 最小修正：在此远端 Python 命令加入 `-B`，保留现有三文件断言。例如将解释器参数改为 `..., '/venv/bin/python', '-B', '-u', ...`。不需要删除缓存、扩大目录白名单、添加重试或修改活动源码。
- 复查要求：只需确认启动命令已明确禁止字节码写入且其他范围不变；随后由获授权的执行代理运行首次 CPU 核验。当前审查没有实际执行来确认运行结果。

## 非阻断问题 N1：成员极值的密集参考复用了被测摘要

- 位置：`cpu_witness.py:67-75`，相应生产统计位于 `whole_mask_range.py:28-33`。
- 证据：密集参考的 `member_lower_mean_normalized` 和 `member_upper_mean_normalized` 直接把 `geometry['lower']`、`geometry['upper']` 按成员展开，随后做逐点加权。两条路径因此共享已经算好的成员极值。行 47 只另行检查这些极值有限。
- 实际覆盖：该比较确实独立核对了“按 SP 点数加权”与“展开到每个成员后加权”的等价性，以及这些摘要对 Text、Query、alpha 的梯度；它没有独立验证极值是否由原始成员坐标正确生成。极值生成的源码本身正确使用 `np.minimum.at` / `np.maximum.at`，本审查没有发现其实现缺陷，所以这不是首次数学聚合运行的阻断项。
- 最小改进：若要把运行结果表述为“独立验证了成员极值构造”，在参考分支中按原始 SP ID 对成员坐标直接 `min` / `max`，再展开和加权，不要读取被测 `geometry['lower']` / `geometry['upper']`。不做此改动时，明确保留“极值摘要构造由源码检查，运行比较验证其聚合”的限定。

## 方法和原生接口核对

`models.pv_ground.py:528-552` 显示每个场景只选一个 Text token 的 Mask；`pred_masks.expand(1,256,M)` 产生共享候选维的 Text logits；Query logits 经 `squeeze(0)` 为 `256 x M`；`sigmoid(pred_scores.squeeze())` 得到每场景一个零维 alpha。`evaluator.py:596-603` 和 `685-692` 均先计算 `alpha * Text + (1-alpha) * Query`，再 sigmoid，并按原生 SP ID gather 到全部点。因此新实现的融合顺序和标量 alpha 假设正确。

新函数刻意接收单场景的二维 Text/Query 张量；当前构造输入通过 `shared_text.expand_as(query)` 正确模拟原生重复 Text，并保留对共享 Text 的梯度累加。未来接入时原生 Text 需要去掉其开头的长度为 1 的维度，当前尚无模型适配器，也没有声称已完成模型集成。未把 alpha 误当作 256 个候选权重，未在 sigmoid 后融合，也未按粗框或候选分数剪裁。

`np.unique(..., return_inverse=True)` 只压缩有成员的 SP，且保留排序后的原生 ID；Mask 读取使用 `fused[:, native_ids]`，没有把压缩索引误当作原生槽号。原生源码按 `superpoint` 做 `scatter_mean` 并按同一 ID gather；此处使用 `max(ID)+1` 构造数学 Mask 槽数与该索引语义一致。fixture schema 给出的 SP dtype 是 `torch.int64`。

设 SP 的成员数为 `n_s`，其每轴桶计数为 `H_sab`，融合概率为 `p_qs`，总支撑质量为 `Z_q = sum_s(n_s * p_qs)`。实现中的

`softmax(logsigmoid(logit_qs) + log(n_s)) = n_s * p_qs / Z_q`

是正确的点级测度权重。随后使用 `H_sab / n_s`，因此输出方向 profile 为 `sum_s(p_qs * H_sab) / Z_q`，恰好是逐点 soft support 分布；这没有丢掉大 SP 的成员数，也没有把点数重复乘两次。连续一、二阶矩以相同权重合并，`second - mean**2` 为每轴方差，`origin + mean * span` 恢复物理坐标均值，`support_fraction = Z_q / N` 正确。

成员极值摘要是 `sum_s(n_s * p_qs * min_or_max_s) / Z_q`。这是对各成员所属 SP 的极值取软加权均值，既不是全场景支撑的真正最小/最大值，也不是实例边界。计划、字段命名和 receipt 限制已经正确说明这一点。32 桶保存的是每轴边缘分布，不保存完整三维联合形状；连续矩与成员范围不能恢复任意缺失细节。未发现需要在本任务加入阈值、空槽 fallback 或额外边界分支的证据。

## 密集前向和梯度参考核对

除 N1 指出的极值复用外，参考路径直接按每个点的原生 SP ID 读取 logits、sigmoid、除以逐点概率和，使用 `scatter_add_` 累加桶概率，并从逐点坐标计算一、二阶矩和平均概率。它没有复用压缩路径的 softmax 权重、桶计数、均值或二阶矩，足以核对点数权重、原生槽映射和对应前向聚合。两边共享场景归一化原点、跨度和同一桶约定，所以该比较也不是对归一化配置本身的独立检验；这些公式在源码中已核对正确。

计划指定的 256 候选输出及 profile 各轴质量守恒均有断言；密集等价和梯度比较只针对固定 `[0,31,127,255]` 四个 Query，符合计划。前向比较遍历全部返回字段，绝对误差门限为 `1e-9`。梯度比较针对包含 profile、连续矩、范围摘要和支撑质量的一个具体标量 objective，分别求共享 Text、完整 Query 张量和 scalar alpha 的梯度；`expand` 和候选索引仍在 autograd 图中，没有 detach。它是这四个 Query、这一 objective 的梯度比较，不是完整 Jacobian、上游网络梯度或所有可能 logits 的验证。

所有数学 logits 都是确定性的构造值，alpha 固定为可求导的 `0.35`。这不涉及随机采样或模型预测。源坐标从 CPU float32 fixture 转为 float64；计算和比较按 CPU double 进行。前向及梯度误差的断言会拒绝非有限差异。这里尚无实际误差数值或 PASS receipt。

## fixture 来源、隔离及输出语义

提供的 inventory 根目录与 `normalized_spec.json` 的 `reference_fixtures` 完全一致。现有 receipt 记录了按既有有序 fit row ID 选择前四个不同物理场景、无 augmentation、无 formal rows、无模型 forward 和优化步；行对应关系为：

| 文件 | training row | scan | receipt 的 superpoint_count |
|---|---:|---|---:|
| row_00000.pt | 0 | scene0000_00 | 667 |
| row_00173.pt | 173 | scene0001_00 | 710 |
| row_00237.pt | 237 | scene0002_00 | 2329 |
| row_00455.pt | 455 | scene0004_00 | 1441 |

schema 报告 `point_clouds` 为 `[1,50000,6]` float32，`superpoint` 为 `[1,50000]` int64。审查读取的是已完成元数据检查的记录，并未自己重新加载四个 `.pt`。计划中的 witness 会对每个文件核对现有 receipt 的完整文件 SHA-256，并分别断言输入形状；该现有 provenance 检查实现正确，没有发现需要新建哈希框架的问题。

fixture 字典和 receipt 包含 text、检测框或 target_id 等元数据，但本 witness 的数值运算只读取 XYZ 和 SP ID；没有用目标标签、GT 框/Mask、检测框或 text 内容构造支撑或监督。`GT_used=False` 对当前数值计算是正确的。使用全部 50000 点意味着使用该 fixture 的全部采样成员，不等于读取了原始扫描的所有物理点。

启动器只读取本地归一化 spec 中的已有解释器与 fixture 路径，使用新的远端目录，不导入或写入活动模型源码，不读取 checkpoint，也没有 GPU 状态、训练进程或控制器查询命令。远端命令由 `shlex.join` 对每个参数引用；环境赋值 `CUDA_VISIBLE_DEVICES=` 的形式正确。SSH 使用已知主机键加载及默认拒绝未知键策略，没有发现此命令边界的注入问题。未检查认证包装器、密钥、密码内容或实际连接状态。

CPU witness 只导入 NumPy、Torch 及本地几何模块，使用 `map_location='cpu'`，没有 CUDA 张量操作，结束前检查 `torch.cuda.is_initialized()`。这些源码支持计划中的 CPU-only 范围；运行是否成功仍要由实际 receipt 证明。零模型 forward、零 optimizer update、零权重文件及未集成模型的声明符合源码。

正常远端退出时，启动器先保存 stdout、stderr 和退出码，只有零退出码才读取并校验 receipt。新目录创建和源文件排他写入避免覆盖既有任务。B1 是已找到的成功收集路径缺陷。此次审查没有发现证据支持增加重试、兼容层、额外异常捕获或改造输出机制。

`compressed_geometry_bytes` 仅统计返回 NumPy 数组的 `nbytes`，不包括构建临时数组、Torch 转换、全部候选激活、autograd 图、密集参考或进程开销，不能作峰值内存或 CUDA 容量结论。SP 中心替代的 L1 指标是三轴、逐 SP 的原始桶计数差异；它不是预测 Mask 精度、边界误差或模型增益。四个训练场景、构造 logits 和 CPU double 的数学比较不支持 REC、泛化、完整模型可运行、结构方法有效或最终边界已验证等结论。

当前门结论保持 **FAIL / 1 blocker**。最小必要变更仅是 B1 的 `-B`；N1 可通过独立原始成员极值参考改进，或在后续报告保留明确的覆盖范围限定。未修改任何实现或活动实验文件。
