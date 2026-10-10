## §20.376.150 — Query Mask匹配的真实正常训练预检入口：源码准备

已将§148/149的隔离匹配改动接入现有native_c_off_preflight入口，仍仅SOURCE_ONLY。14份原生源与§148 Query版本逐字节相同；本次新增的是工程检查入口和两份模式配置，没有修改活动C-off源码、执行GPU或创建部署器/排队任务。完整PV依赖仍须从既有运行时绑定部署，14份局部源不能冒充完整仓库。

text/query两份配置唯一差异为matcher_mask_source。两者均保持同一保留E0模型状态、fresh优化器、G保留/C关闭、B8/seed2027、核心骨干LR1e-6/新增1e-5/WD0.0005/clip0.1；原生RoBERTa冻结，核心、骨干及G/A/B正常联合更新。父状态包含历史C适配，不是从头无C消融。此处仍是ScanRefer表达微调，不改称作者完整混合检测配方。

每种模式计划用真实原训练loader的两个增强batch，共16行和2次更新，调用原train_one_epoch与完整原生criterion。保留原模板已有梯度/参数更新、初始1295模型状态与保留E0逐项相等、完整Adam/scheduler/全部RNG保存及冷恢复检查。作为初始一致性参照的checkpoint继续使用受保护的C-on/E0 best.pth（ed8455dd…6faf0c4），不使用活动C-off训练终点，也不使用预检更新后的状态进入正式训练。

新增只读匹配观察器在同一实际前向输出与GT上计算另一Mask来源的匹配，并始终返回配置来源的原结果给loss。记录全部有效GT、Query分配变化、对应最终Box IoU、Own Query Mask交并数/二值成本，以及实际匹配来源的Mask成本。Own成本与Text控制实际成本分开标注；没有改变目标、候选、分数或添加损失。无Mask的早期prefix不新增比较，预计末层每步一次。

匹配模式写入checkpoint config，并在恢复后通过显式CLI参数新建criterion核对。原load_checkpoint不自动以config覆盖CLI，因此正式恢复须携带原matcher_mask_source；不能只因模型状态恢复就声称监督模式自动恢复。

两个protocol的serial_gpu_preflight_admitted均为false，入口在导入Torch之前拒绝执行。必须先等当前C-off三轮终态及判断，之后才考虑独立完整源、父权重、空闲A100/GPU锁、磁盘保存空间与真实工程预检。两个模式输入是否实际一致、成本竞争和匹配变化幅度仍未实测。

旧C-off同类实际预检总体647.08秒，两次更新25.90秒。后续每模式约11分钟仅作初始估计，查看按整体完成时间安排，逾时才每240秒继续；不能将两步计算耗时当成全部加载/恢复耗时，也不在现在创建新观察器。

限定源码审查WARN/0阻断，14源/模板及两份配置差异已核对；same-family/provisional、实际identity UNATTESTED。源码准备不是GPU通过、训练启动、新精度或三有效机制证明。新正式控制是否能复用当前text行为等价的C-off终态，要在完整终态和真实预检后决定，不预先重复启动三轮控制。

原观察器52851/PID51540仍按Oct11 05:00:40检查正常第2轮。保留最好5677/4920及必要父链/V99。没有新增ScanRefer/Nr3D/Sr3D指标，目标ACTIVE_UNMET。
