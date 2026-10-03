# PV-Ground候选监督一致性执行表

| Run ID | Purpose | Variant | Status | Evidence |
|---|---|---|---|---|
| C0 | 原生函数CPU检查 | 控制/目标替换 | PASS | cpu_test.json；无资格时精确等价，有资格时公式/梯度allclose，保护其他匹配 |
| C1 | 真实GPU两步预检 | 原G＋对比目标替换 | PASS_ENGINEERING | preflight.json，batch8/两步更新/保存恢复通过；多正例数值含义另做复核 |
| C2 | 同预算继续训练控制 | 原G | COMPLETE_MODULE_HOLDOUT | 3723次更新，fit29778条各一次；11:28终态6887留出6144／5561；control_complete/ANALYSIS.json；非正式9508成绩 |
| C3 | 同预算继续训练实验 | 原G＋对比目标替换 | RUNNING_OBSERVED | 11:33实际g_consistent/train，控制器320934／子进程341235存活；没有新正式成绩；有效权重变化WARN保留 |
| C4 | 正式9508评估/候选与修复分解 | 两组 | TODO | 无新指标 |
