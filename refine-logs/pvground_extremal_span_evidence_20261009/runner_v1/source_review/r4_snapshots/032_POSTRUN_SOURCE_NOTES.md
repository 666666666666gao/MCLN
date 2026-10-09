## §20.376.116 — 在既有训练期间准备终态核验，不修改活动实验

承接§115。本轮原生fit已于12:43:37 CST定时启动观察确认真实第1更新：B8，matched控制loss0.3012793660，selected策略总loss1.1429555416，其中2条赢家完全未匹配而获得额外Mask监督。该记录不是当前更新数、验证结果或精度收益。当前原控制器1339与配置保持不变；当前最好5599/4859仍保留，5658/4850及三个有效贡献目标仍未达成。

Windows持久结束观察器PID26516在12:46:12真实进程/CmdLine核对为存活，至此remote_queries_performed=0；首次远端完成观察仍为17:14:27，预计17:19:27结束，之后240秒。没有提前重复查询GPU或修改正在运行的源码。

三份独立postrun源码已准备并经新上下文R3 SOURCE_ONLY PASS、0阻断；尚未执行实际终态收集/重算/CPU恢复。collector只接受本地原观察器和远端fit.exit/status已成功关闭，流式核对字节/SHA，暂收两份小terminal用于恢复后再按用户授权保留最佳、清理无用。analyzer仅针对真实9508终态，重算三组parent/content/selected的全部256候选，共7,302,144个Box IoU；核对3723步、29778 fit ID各一次、实际root/other/unmatched职责及额外监督预算。native/native和CPU/CPU的合格未选中统计分开，所有CPU/native阈值差异保留，不自动晋级。

同帧parent是未加学习式corrector的Mask参考父路径，不是受保护5599/4859头；与当前最好比较属于独立前向，输入身份按row/scantarget/rootbox/pointSHA逐行对齐，不能将独立数值漂移写成新增模块效果。Mask指标仅重算保存的点Mask IoU，本脚本不重构原始ScanNet GT，也不重放整个点Mask。R1混合口径已修正；R2发现遗漏旧W已有的CPU RNG重放及既有intake绑定，R3已沿用修正。CPU恢复仅核对两份terminal及CPU同种重建中的1314状态；未存入checkpoint的零输出R隐藏随机状态只在CPU构建间对齐，不宣称原formal GPU所有状态逐位重现；不执行NN、不更新优化器、不删除文件。

Nr/Sr仅本地缓存准备清点：已有对应作者权重SHA与分区，旧正式接口Nr7899、Sr17726条，butd_cls实例框/预测类别，与ScanRefer对象输入分别标注；Sr文本权重及Nr/Sr语言倍率不同，联合检测行按sample_dataset保护。未查询本日远端可用性、未启动新Nr/Sr训练。后续完整方法固定后从各自作者核心初始化，不能用ScanRefer G完整状态覆盖它们的核心。缓存两步spec不等于新最终方法训练结果；仍先完成ScanRefer目标与贡献验证。
