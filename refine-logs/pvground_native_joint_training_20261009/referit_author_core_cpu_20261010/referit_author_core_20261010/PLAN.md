# Nr/Sr 作者核心加载顺序检查

本轮补充已有 B4 准备，解决真实输入差异：两份作者权重均包含 `text_encoder.embeddings.position_ids`，总数1235；ScanRefer作者权重没有该持久缓冲区，总数1234。当前R2初始化若先设nonpersistent再strict加载，会多出一键。已归档作者 inventory 和旧 strict_load 的证据保留；不能将旧工厂的通过当作当前新工厂已通过。

改动仅在新的隔离初始化文件：先校验作者确定性buffer并严格加载1235项，再取消buffer持久化；正常全模型保存仍为1234核心＋G/A/B状态，合计1295。没有删任意未知键、放宽strict、加载ScanRefer状态或加入猜测性兼容分支。R2源码和报告全部保持原字节；活动ScanRefer源码不变。

一次CPU工程检查使用当前真实`TrainTester.get_model`与各自作者权重，比较全部1234项核心的key、shape、dtype和值，验证1295全状态、新A/B输出零，以及CUDA未初始化。完整框架文件按现有port逐文件SHA核对后复制到单独CPU目录，不查询活动训练。采用现有正常入口的构造参数只用于加载/构造检查，**不是宣称Nr/Sr作者训练与对象输入协议已核对一致**；作者与本次构造flag将同时记录。

不运行forward、criterion、真实loader、优化器、保存/恢复、GPU或正式训练，不产生REC精度。源代码复核通过后才执行；结果封存后再独立复核其实际覆盖。完整跨基准训练仍须等待ScanRefer最终版本和单卡空闲，并核对真实数据、协议与冷恢复。

当前阶段：SOURCE_PREPARED；CPU_PENDING；正式训练未启动；总目标ACTIVE_UNMET。
