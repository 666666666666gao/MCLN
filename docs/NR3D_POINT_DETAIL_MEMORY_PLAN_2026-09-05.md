# 局部点细节直接进入 superpoint Mask：独立预检方案

2026-09-05。状态：模块和预检入口已准备；4 项服务器 CPU 测试通过。
尚未执行原生 CUDA / 真实数据预检，尚未训练或取得质量结果。
L1 按原定设置完成后，才执行本预检。该模块不叠加 L1 中间权重。

## 问题与本次范围

完整合法 256 Query 中已有合格框的当前 Nr3D 错误占 87.55%。本次改变服务于
Mask 的局部证据来源，不能据此预期解决主要的复杂语言排序错误。
M2 的 828 条有好框、superpoint 多数标签也足以表达目标的难例中，767 条所有
原始 Mask 仍不过 0.5；M4 则实际观察到空球邻域返回远处 seed0。M5 最近邻
对照未通过双阈值要求。这些证据支持独立检验细节传递，而非继续改变融合权重。

保留原 PointNet++ 全局路径、同一批 50,000 点、候选与对象协议，以及现有全部
Box/Mask 头。本次只新增一个 superpoint 特征残差，不替换主干、不改邻域算法、
不加任务 token、新损失或候选评分器。对象外观记忆和共享身份任务查询是后续独立实验。

这首先是一个**点特征控制分支**，不是稀疏体素实现。原 SA1 已提供 2,048 个位置
及 128 维特征，最终 Mask 路径只有 1,024 个 seed。先利用可用中间特征，避免
在证据尚未成立时同时更换输入密度、预训练和计算环境。

## 实际实现

`scripts/nr3d_point_detail_memory.py` 独立实现以下操作：

1. 用现有 PointNet++ 三近邻逆距离插值，将 SA1 特征映射回同一批输入点。
2. 每点拼接 128 维特征、相对本 superpoint 中心的 XYZ、现有输入 RGB，共 134 维。
3. `Linear(134,128) → ReLU` 在点上编码，再按真实 superpoint 成员求平均。
4. 用无 bias 的 `Linear(128,288)` 得到每个 superpoint 的残差。该投影零初始化。
5. 把相同残差加到原 grouper 的两个邻居通道。原位置编码和 max pooling 继续执行；
   两个原生 Mask 分支都读取补充后的特征，Query Mask 仍逐候选生成。

共 **54,144** 个新增参数。原始 XYZ/RGB 不经过额外采样，不使用 GT 实例 Mask
清理输入，也不把目标标签或 IoU 放入特征。GT 仅通过原生损失进行监督。
复用 `deterministic_scatter_mean_dim0` 的确定性分组归约；未改变该工具。
中间语义特征仍受 2,048 个锚点限制；每点连续坐标和颜色提供独立的细节入口。
因此不能称为已恢复未观测的表面，或已验证体素优于点表示。

插值保留现有 `PointnetFPModule` 的 `1 / (distance + 1e-8)` 约定。
这里零距离确实存在：SA1 锚点就是从输入点中采样得到的点。
没有新设缺邻居 fallback 或 CPU 插值替代实现。

附件通过临时 hooks 接入；不注册到 parent state，不修改原生 `models/mcln.py`。
本轮没有更改正在运行的 L1 目录、源码、参数或训练进程。

## 已完成的 CPU 检查

服务器既有 Python 3.7.11 / PyTorch 1.10.2+cu111；CUDA 不可见。
4 项测试通过，耗时 1.31 秒，证据在 `refine-logs/point_detail_cpu_20260905_v1/`：

- 零初始化输出为零；第一步只连接输出矩阵，固定合成更新后内部编码器也取得有限非零梯度。
- 一个 superpoint 的颜色变化不影响其他 superpoint；缺号槽保持零；点顺序变化不影响聚合。
- 一起平移点与 superpoint 中心保持相对几何输出。
- 合成原生路径中，多样本 hooks 顺序正确；移除附件恢复原结果，父模型状态不变。

这些是合成 CPU 样例。测试中的插值函数替身仅检验 hook 路由，**不验证 CUDA 内核**。
没有真实训练行、GPU forward 或优化器更新；合成测试显式执行了一次输出矩阵更新以检查梯度路径。

## 下一步：16 条真实 fit 输入预检

入口 `scripts/run_nr3d_point_detail_preflight.py`，使用与 M3 相同的 16 条表达和
16 个训练场景；同一 split salt、原始 CSV 顺序、B4、seed0、无增强、0 workers。
受保护 E57 平均权重、冻结源码 manifest、所有使用的数据和附件均须绑定 SHA-256，
运行前后复核。父模型保持 eval 且参数冻结，原生 Hungarian 与 Mask 损失不变。

每批运行原生、零残差、固定小残差共三次，即 12 次真实 forward，0 optimizer 更新。
固定残差为 288×128 矩阵的矩形对角线乘 0.001，只用于梯度/连接检查，不能作为效果结果。

通过条件：

- 原生三近邻 CUDA 在包含精确锚点的合成输入上有限、正确；实际 SA1 是 B×128×2048。
- 输入点 SHA 与 M3 一致；零残差时种子、候选、Query、Box、分数、两种原始 Mask 和 alpha 完全一致。
- 非零残差实际改变原始 Query Mask，Mask 损失对全部新增参数取得有限非零梯度；
  REC Query、框、分数和种子仍完全一致。
- 父模型参数/缓冲区和文件不变；结束时新增模块也恢复初始状态。
- 记录实际峰值显存与每批 forward 耗时。原生 no-grad 与附件 autograd 计时用途不同，不能直接当部署开销比。

CPU 检查不足以保证上述条件；GPU 预检尚未运行。通过后才固定一次独立训练对照的
样本、步数、控制组与质量门槛，再读取模块留出终态。不得用 16 行扰动 Mask 结果挑结构或阈值。
任务查询分支应在局部信息贡献明确后单独比较，最终统一实例选择仍需要完整训练与正式验证。

## 同时补充的实例身份诊断

`scripts/summarize_nr3d_query_identity.py` 只读取已封存 M2 的 7,899 行，不运行新 forward。
它核验压缩/解压行哈希，按原生 `IoU > threshold` 和逐行浮点累加复现
4,194 / 3,482 Mask hits 与原生 IoU 总和。

7,898 行有合法 REC Query，其中 **533 行**的 REC/Mask 选择不同；另 1 行无合法 REC Query，
不为该行虚构替换 Mask。对 7,898 行固定读取 REC Query 对应的已有融合 Mask：

| 条件 | 原生 Mask 选择 | REC Query 条件 Mask | 修复 / 破坏 / 净增 |
|---|---:|---:|---:|
| Mask@0.25 | 4,194 | 4,241 | 64 / 17 / +47 |
| Mask@0.50 | 3,482 | 3,509 | 36 / 9 / +27 |
| Mask mIoU（同一 7,898 行） | 37.4640231% | 37.7524401% | +0.2884171 个百分点 |

这支持核验共享身份，但也直接显示强制同 Query 会破坏已有命中。
它是已有验证输出的反事实诊断，不是训练结果、正式最好、选择规则部署或泛化证据。
证据 `refine-logs/mask_branch_diagnostic_20260905_v1/query_identity_analysis.json`。

## 参考代码边界

本轮只读核对 [PV-Ground 官方代码](https://github.com/AaNnWwTt/PV-Ground/tree/262e2592589baec7bb83a0d46aae6542d4ccedfb)。
其 `pv_backbone.py` 确认 MeanVFE、VoxelBackBone8x、HeightCompression、VoxelSetAbstraction
组成主干；`pv_ground.py` 仍使用 `QueryAndGroup(radius=0.2, nsample=2)` 构造 Mask 特征。
因此只移植体素主干不能视为消除了当前 Mask 读取瓶颈。
本实现独立编写，未复制上游代码，也未安装其新版本 spconv/运行环境。

三数据集正式目标仍未完成。L1 结果、局部信息质量、任务查询互补性和后续正式泛化结果
分别记录；不得把 CPU 测试或既有验证诊断写成新结构有效性结论。
