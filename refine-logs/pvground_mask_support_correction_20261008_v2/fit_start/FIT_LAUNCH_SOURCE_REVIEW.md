# FIT_LAUNCH_SOURCE_REVIEW — SOURCE_ONLY PASS

同上下文、同系列、临时接受；实际后台/model/effort 均为 **UNATTESTED**。没有阻塞项或非阻塞发现。本报告不表示 v2 M0 通过，也不授权现在启动 fit。

正式启动器先要求观察器真实 closed、controller 不存活、exit0、完整 M0 receipt；两组均须完成两步、零头等价、冻结状态、1314 状态 CPU 重建、GPU 新头/Adam 恢复及同帧 native Mask/Box 精确对照。远端真实退出码和状态会再次检查。此次审查未查询运行中的 M0 或读取它的结果。

正式流程依次运行完整 initial9508、29778 一遍/每组3723更新、完整 terminal9508。每阶段按现有代码构建新进程；训练新建头和优化器，不继承 M0 更新。全部256候选、单一原生 bbs、同 Query 的 Box/Mask 与原生评估核对保持。只保存两份新 head delta、Adam 状态及恢复元数据。

数组预算与代码的2378个 NPZ 对应：三份全部候选 Box、评分、GT及行ID，外加容器余量、128 MiB 文本/日志及2 MiB 两份delta，总要求546015744字节（520.72 MiB）。这是启动前必须通过的真实空间检查，不是已测容量结论。

最新正式观察器已纳入绑定：估计19267秒，首查18967秒，即终点前300秒；之后必要时240秒。它只观察同一controller，真实关闭后写receipt，不启动NN或collector；超两倍估计只停止该观察器并保留原任务。

22个实际文件绑定见JSON。两个更新后的正式管理脚本和两段嵌入程序的Python3.7 AST检查通过；本轮两份审查合计15个当前v2 Python源文件及3段嵌入程序。此次仅本地源码/摘要/AST操作，未执行SSH、GPU、torch、模型、权重读取或启动。
