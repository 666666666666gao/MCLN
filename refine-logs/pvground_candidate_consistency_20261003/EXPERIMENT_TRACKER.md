# PV-Ground候选监督一致性执行表

| Run ID | Purpose | Variant | Status | Evidence |
|---|---|---|---|---|
| C0 | 原生函数CPU检查 | 控制/目标替换 | PASS | cpu_test.json；无资格时精确等价，有资格时公式/梯度allclose，保护其他匹配 |
| C1 | 真实GPU两步预检 | 原G＋对比目标替换 | PASS_ENGINEERING | preflight.json，batch8/两步更新/保存恢复通过；多正例数值含义另做复核 |
| C2 | 同预算继续训练控制 | 原G | RUNNING_OBSERVED | 2026-10-03 11:15真实观察：g_control/train，3648/3723步；control_boundary_observation_000.json；无新正式精度 |
| C3 | 同预算继续训练实验 | 原G＋对比目标替换 | QUEUED_IN_PAIR | 已启动串行配对中的第二训练阶段，11:15尚未开始；负对比目标及有效权重变化WARN保留 |
| C4 | 正式9508评估/候选与修复分解 | 两组 | TODO | 无新指标 |
