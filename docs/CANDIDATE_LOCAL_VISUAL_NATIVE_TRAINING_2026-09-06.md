# 候选局部视觉：原生训练接入

本次将 ScanRefer 正在检验的同一条候选局部读取路径接入原生模型工厂、优化器和
checkpoint 加载。Nr3D/Sr3D 训练待 Scan 正式通过后启动；当前 Scan 固定实验仍使用
自己的 614 文件隔离源码和预注册设置。

## 接口

原生 `train_dist_mod.py` 使用以下新增选项：

```text
--use_candidate_local_visual
--candidate_local_visual_lr 0.0001
```

开关默认关闭。开启时，原 `MCLN` 六层模型在完成原模块构造后，给最后一层安装
`CandidateLocalVisual`。它与 Scan 实验的插入位置、原始点/SA1 输入及 145008 个
新增参数完全相同。Nr/Sr 保持现有 `butd_cls`、预测类别输入、原生 selector 输出
和合法候选过滤协议。

新分支有独立的 `candidate_local_visual` 优化器分组，记录完整 `parameter_names`；
原 decoder/backbone/mask_head/selector 分组继续使用各自配置。分组不重叠，也没有
遗漏可训练参数。不开启时保留已有分组。原生训练的解冻范围和增强仍由原训练配置
控制，本次接入不把 Scan 首轮的“只训练最后层”隐式应用到整个原生训练器。

## 预训练与恢复

现有 Nr3D 平均权重是 evaluation-only 文件，没有 optimizer/scheduler。作为新训练
起点应使用已有的 `--model_only_initialization --checkpoint_start_epoch 1`，创建新
优化器；这属于权重初始化，不是恢复原 E57 的完整训练状态。

原加载器会跳过形状不符张量，并允许只加载部分主干。因此，局部模型在该宽松路径
之前校验完整模型身份：从旧核心初始化时仅允许缺少整个新分支的 10 个张量；不能
缺少其他核心张量，不能只载入部分新分支，也不能接受形状或 dtype 不符的核心。
已训练的局部模型要求完整匹配。加载带局部分支的 checkpoint 而未开启结构时直接
报错，避免忽略新增张量后继续评估。使用原生 DDP state 命名；Scan 实验专用的
plain-state 文件仍由它的专用正式入口读取。

完整的原生新 checkpoint 可以保留五组优化器/调度器状态。旧核心权重不能被当成
五组新优化器的完整恢复文件。没有新增自动降级或部分加载行为。

## 已完成验证

2026-09-06 17:45:13 CST，原 Python3.7/Torch1.10.2 环境完成：

- 新增及已有加载/优化器回归检查共 55 项通过，包括新五组优化器与调度器恢复、
  核心尺寸不符时在修改模型前拒绝、禁止静默丢弃已训练局部分支。
- 真实 Nr3D 受保护权重在 Nr3D、Sr3D 两种原生配置下完成实际 `load_checkpoint`。
  1144 个原张量逐位等于输入权重，新增 10 个张量保持各自初始化，输出投影为零。
- 两种配置均为 1154 个 state 张量、149779949 个总参数，其中新分支145008参数。
  优化器覆盖原 decoder625/backbone48/mask_head58/selector9，加 local10 个参数张量。
  这是优化器包含的张量数，不是已取得梯度或完成更新的数量。
- 真实模型优化器/调度器加载前后相同，GPU前向0、真实数据行0、模型更新0、
  完整模型 checkpoint 写入0。小型单测使用临时测试权重和一次性优化器更新。

这里 Sr3D 只验证 Nr3D 权重可以作为兼容的预训练输入，不代表恢复了历史 Sr3D
最好权重，也不代表这种初始化已经取得 Sr3D 性能收益。

隔离源码：
`/root/autodl-tmp/mcln_candidate_local_native_preparation_20260906_v1/model_source`
，616文件；manifest SHA
`4af68cad46b52c9e250de17872a193485d75529378db983ad037779232e500fc`。
证据：`refine-logs/candidate_local_native_preparation_20260906_v1/`。

ScanRefer 9508行正式底线通过后，再以实际 Nr/Sr 输入完成 GPU 前向、候选映射及
梯度预检，锁定各自短期训练配置。当前检查不替代这一步，不恢复已取消的长期
baseline 队列，也不将旧240/140 epoch脚本作为此次训练预算。Nr/Sr Mask 不设晋级门。
