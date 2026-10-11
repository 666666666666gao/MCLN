# 连续两轮严格指标下降：隔离源码检查

本目录只是未来ScanRefer正常训练入口的停止规则补丁。当前远端source_conditioned三轮、模型源码、01:41:56观察器及BN诊断源码全部不改。没有部署、GPU前向、优化器更新或新增训练预算。

用户明确确认按Acc@0.5相对上一轮下降计算，连续两次才停；持平或回升重置。当前真实4920→4599→4625不触发。该规则不是“连续低于起点就停”，也不看Acc@0.25或训练loss。

修改只涉及三个源码文件：原生main_utils.py在完整验证记录和best/latest保存后检查；native_decline_policy.py从原生指标日志计算；controller接受正常的提前终止并记录实际轮数。继承原生同实验恢复路径、原选模、权重保存、架构、模型、loss、数据和优化器。规则状态从已有native_metrics.jsonl读取，不新增checkpoint格式或缺失状态fallback。

源码基于实际Face-v2 main_utils.py（5b66b8ff...）。第一次准备误指向旧V/source/main_utils.py，原有身份断言阻止了写入；已改为实际Face-v2/source，未放宽断言或修改活动源码。

执行下面的CPU控制流程检查；它直接抽取补丁后的原生循环AST，使用训练/评估/保存回调，检查是否提前退出以及latest保存是否先于停止。它不执行真实模型、不写权重、不证明GPU训练或冷恢复通过。模拟轮次只检验控制流，不能当实验精度。

```powershell
& 'C:\Users\gb\AppData\Roaming\uv\python\cpython-3.11.14-windows-x86_64-none\python.exe' -B -X utf8 'C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\native_decline_policy_20261012\check_native_decline_loop.py' --output-dir 'C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\native_decline_policy_20261012\reviewer_cpu_fixture'
```

输出目录是新目录，首次执行后不要原地重复；保留RESULT.json。初次执行的cpu_fixture_1已通过，四份Python文件也通过实际Python3.7 AST检查。

未来部署还须把同一个native_decline_policy.py放在模型源码目录和normal controller目录，纳入已有helper/source清单。实际预算另行选择并记录；controller的planned_epochs和--max_epoch必须一致，提前结束时报告实际样本量与更新数，不能仍称完成了原固定预算。本目录未提供已准入的启动器，不可用它覆盖当前运行或旧审查结果。后续源码审查与真实运行/保存/恢复仍需按相应范围验证。
