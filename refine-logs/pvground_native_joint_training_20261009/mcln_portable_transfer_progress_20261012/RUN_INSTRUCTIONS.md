# MCLN迁移三路配对执行

仅使用既有region-9.autodl.pro:33476及warm Py3.7/Torch1.10.2 runtime，不构建新环境。全路径使用C:/别名，避免既有known-host路径空格故障。凭据由run_mcln_thread_credential_20261011.py私有注入，不写入源文件或结果。

1. fresh Astra/max源码审查通过后，launch_transfer_stage_authorized.py --stage cpu：隔离MCLN源码与现有PointNet编译扩展，实际原生工厂严格加载E71和已验证A/B参数。没有前向、没有更新。
2. 当前PV full BN诊断实际终态0、GPU空闲后，--stage sanity：真实首batch8条、native/Mask-reference/full-transfer共享输入及RNG，原生评价对数据集GT。全部1159状态不变、语义分数完全相同。
3. sanity实际通过后，--stage full：同样三路完整9508，原生last/bbs。每128个batch写进度；根据真实sanity及后续128batch速度估时，临近终点首查，此后240秒。不是训练，无optimizer/新权重，不用GT决定推理候选。

每一步是显式单次准入，不自动重试、不抢占PV任务、不改父权重，不启动尚未定义预算的正式训练。失败时先保存child.exit、status.json、stage.log与controller.log，再针对实际错误修复。

MCLN保留E71既有数据/对象输入配置，mesh超点目录与原生文本权重；PV专属六源G读取未迁移。A/B跨骨干直接加载属于迁移探测，并非MCLN正常训练结果。实际GPU/全量/训练状态必须分别记录。
