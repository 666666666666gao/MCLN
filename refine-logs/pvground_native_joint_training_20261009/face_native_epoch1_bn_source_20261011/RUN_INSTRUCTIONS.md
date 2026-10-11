# 执行入口与结果解释

源码审查已完成：SOURCE_ONLY_PASS / WARN，0项阻断；这是同家族、独立上下文、provisional的源码审查，实际模型身份UNATTESTED。首批GPU预检及9508条重放尚未执行。当前训练观察器43551/PID5940首次读数仍为2026-10-11 21:16:47.338197北京时间，不能为启动诊断提前查询或改动训练。

当前source_conditioned三轮训练实际结束后，先运行sanity。launcher会核对三轮终态、检查点、现有源码/环境及GPU空闲，并在相同GPU锁下先冷恢复best/latest的模型、优化器、scheduler和全局RNG，再执行首个实际8条batch的两分支重放。

本地必须沿用已有Python3.11.14。裸python为3.7，缺少launcher使用的shlex.join；远端模型脚本仍使用已有Python3.7/PyTorch1.10环境，不新增兼容层或重装环境。

```powershell
& 'C:\Users\gb\AppData\Roaming\uv\python\cpython-3.11.14-windows-x86_64-none\python.exe' -B -X utf8 'C:\Users\gb\.codex\tmp\run_mcln_thread_credential_20261011.py' 'C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\batchnorm_buffer_diagnostic_20261011\launch_buffer_diagnostic_authorized.py' --stage sanity
```

实际sanity终态、冷恢复和动态断言通过后，再显式调用同一命令并将最后参数改为`--stage full`。没有自动重试、自动插入训练或预检权重复用。若观察超时，查询原controller/child handle；不要因为观察超时重新启动。

远端结果根目录为`/root/autodl-tmp/pvground_face_bn_diagnostic_20261011/{sanity,full}`，各阶段的`diagnostic_status.json`与`replay/RESULT.json`分别记录进程终态和动态结果。预检预计20分钟，完整重放暂估60分钟，按实际吞吐修正预计时间；首次查在预计结束前几分钟，后续240秒。

逐行的交叉IoU是同一Query**槽位**在不同BN统计下的几何结果，不证明两次表示对应同一物理实例，也不单独隔离评分的因果作用。原生Mask mIoU分别标明position/bbs与semantic/bbf；原生overall_mask命中属于semantic/bbf，逐行自身/融合Mask则对应各分支bbs所选槽位。

诊断只是固定已训练参数后的前向干预，不是最终方法。即使替换统计提高准确率，也不覆盖历史正式成绩，不晋升混合状态权重；后续仍要用正常训练配方的直接对照证明指标增益。
