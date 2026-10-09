## 20.376.133 Nr／Sr 作者核心严格加载的实际差异与修正（2026-10-10）

继§132继续核对作者初始化时，发现了新的真实加载问题。已有作者tensor inventory显示：Nr与Sr均为1235项，ScanRefer为1234项；唯一额外键为`text_encoder.embeddings.position_ids`，shape为[1,514]、dtype为int64，公共参数shape/dtype均相同。旧作者strict-load记录确实校验过该确定性buffer等于arange。但§132新初始化先设nonpersistent，再按1234结构strict加载，会多出该键；旧工厂成功不能替代当前工厂检查。

本轮在`referit_author_core_20261010`单独修正加载顺序：先验证position_ids并严格载入1235项作者状态，再取消该buffer的持久化，按原有1234→1271→1295构建新增结构。只改初始化文件；R2的14份源码、两份初始化spec、固定报告及活动ScanRefer代码均保持原字节。旧R2 tracker在前次CPU入库时已从03bca变为f662，其原字节归档与别名继续保留；不能笼统声称所有live输入从R2起完全不变。R3源码复核为PASS／总体WARN、0阻塞，仅放行这次CPU构造与作者核心加载。

实际CPU子进程于 **07:13:28.638211—07:13:53.032772（北京时间）** 执行完成、退出0。分别调用隔离副本的真实`TrainTester.get_model`，完整新模型均有1295项状态；1234项保留核心的key、shape、dtype和值全部与各自作者权重相同，确定性位置buffer校验通过，新A/B输出层为零。上游G/A/B新初始化，没有加载ScanRefer的G/d06/f989；CPU构造完成不代表这些新增状态已经训练有效。CUDA未初始化，模型forward、optimizer、真实loader和GPU调用均为0，没有REC精度或冷恢复结果。

这次还记录了一个尚待真实接口核对的协议差别：Nr/Sr作者config使用`butd_cls=True`、`butd=False`、`joint_det=True`、`detect_intermediate=True`；这次只用于CPU构造的共用参数是`butd=True`、`butd_cls=False`、`joint_det=False`、`detect_intermediate=False`。两者都能构造相同形状的对象分支，不代表对象输入与训练职责相同。后续必须核对对应作者对象/检测/anchor接口，让公平控制与完整方法使用同一起点和相同真实协议；不能将本次CPU通过当成跨基准训练放行。

证据目录：`refine-logs/pvground_native_joint_training_20261009/referit_author_core_cpu_20261010/`。原始回执、源码复核、实际结果与结果复核分别保留；实际模型身份无法独立核实，按UNATTESTED、same-family、provisional记录。没有新增权重文件。

当前ScanRefer正常训练未查询或修改；原定唯一观察仍为08:19:04.678479。下一步先取得其真实E0／epoch结果、验证必要控制和三项贡献，固定完整架构；Nr/Sr随后还须完成真实loader、对象协议、GPU forward／loss／梯度、完整保存恢复，再各自正常训练。目标保持同一ScanRefer模型Acc@0.25>59.5%、Acc@0.5>51%，三项有效机制及完整模型跨基准验证；ACTIVE_UNMET。

实际结果复核已封存：限定CPU证据PASS，总体WARN、范围内0阻塞。137项输入、26项输出及原始回执字节核验通过；A/B仅输出层状态为零，不是本轮执行forward的证明。作者对象/混合训练flags与本次构造flags的差异明确保留，真实训练未放行；完整参数加载与真正训练效果继续分开。源码PLAN/SOURCE_PREPARATION保留执行前历史状态，最新闭合状态见ACTUAL_AUTHOR_CORE_CPU_AUDIT_CLOSURE.json。
