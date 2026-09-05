# C1：候选 Mask Query 读取 superpoint 记忆的隔离原型

本文件记录实现与 CPU 检查，尚非训练方案或有效性结论。B 的 native/detail 配对
仍按原清单运行；本原型不加载 B 的训练中参数，不修改其代码和 GPU 任务。

## 要验证的增量

原生 Mask Query 经 `x_query` 投影后直接与 superpoint 特征点积。旧 V48 及后续
Mask refiner 已有 Query/superpoint 的低秩逐对分数、几何和原始 Mask 证据残差，
因此不能重新添加相同形式并称为任务查询创新。

C1 让每个候选先从同一场景的 superpoint 集合中读出上下文，再更新自己的 Mask
Query，随后继续原生 Mask 点积。它改变信息的读取过程，没有新增原始观测：

```text
原生 Decoder 身份 Query ──→ 原生 Box / token / contrastive 路径
           │
           └──→ 原生 x_query ──→ 候选位置条件的 SP 读取 ──→ Query Mask
                                      ↑
                           原生 SP 特征 + SP 中心 + 该候选框
```

对一个场景，记已有 Mask Query 为 q，SP 特征为 F，中心为 x，候选框为 (c,s)：

```text
Q = Wq LayerNorm(q)
K = Wk LayerNorm(F)
V = Wv LayerNorm(F)
relative[i,j] = (x[j] - c[i]) / (0.5 * clamp(s[i], min=1e-4))
A[i,j] = softmax_j(Q[i]·K[j]/sqrt(64) - 0.5*sum(relative[i,j]^2))
q_mask_new = q + Wout(A V)
```

空间项是固定的轴对齐框距离偏置；没有硬裁剪记忆或新增空邻域处理。
正尺寸约定沿用现有 Mask 几何代码，框输入本身不被修改。它没有提供对象朝向，
也不表示学会了多个参照物的语言逻辑。真实输入上注意力是否过于集中，仍需预检。

Wout 零初始化。模块共 **74,880 参数、8 个张量**。其他投影初始可计算上下文，
但在 Wout=0 时自身梯度为零；输出矩阵更新后，这些梯度路径才打开。

## 接入范围

代码 `scripts/nr3d_mask_query_memory.py`：隔离 attachment 保存原
`_seg_seeds_prediction`，捕获原 grouper 的 SP 中心，只在原生 Query Mask 方法
调用处更新该场景的 Mask Query，并继续调用原 writer。

原生 Box 和身份 Query 路径保留；Text Mask/alpha 的计算不经过此包装。
新增模块不注册进 parent state dict，移除 attachment 恢复原方法。
这些是实现的边界；真实 native 输出是否保持，尚待 GPU 预检确认。

这只实现 C 中缺失的 Mask 任务记忆读取，并非已经实现全部下一代联合架构。
它保留已有 Box 读出，未增加可被已有 bias 吸收的固定 Box token；没有更改 REC/Mask
的最终选择规则，也没有宣称改善 REC 排名或解决全数据集的任务梯度冲突。

## 当前验证

服务器 Python3.7.11、PyTorch1.10.2+cu111 的 4 项 CPU 合成测试通过：

1. 零初始化输出相同；输出矩阵固定微扰后全部参数梯度有限且非零。
2. 两个候选能按各自框位置读取不同记忆。
3. 固定特征下平移坐标、置换 SP 或候选索引保留相应读出。
4. 不同 SP 数量的两场景包装保持原 writer、其他模拟输出和移除后行为。

**0 数据集行、0 原生模型 forward、0 GPU forward、0 optimizer 更新。**
固定权重扰动只用于合成测试，不是训练结果。证据
`refine-logs/mask_query_memory_cpu_20260906_v1/receipt.json`。

下一步先完成 B 的终态和独立 artifact 检查，再确定后续执行顺序。C1 需要真实
fit 输入的零起点/固定扰动/梯度预检及同输入控制，才具备学习实验入口；当前没有
C1 长训队列、质量门槛或正式模型 artifact。
