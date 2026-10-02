# PV-Ground候选监督一致性执行表

| Run ID | Purpose | Variant | Status | Evidence |
|---|---|---|---|---|
| C0 | 原生函数CPU检查 | 控制/目标替换 | PASS | cpu_test.json；无资格时精确等价，有资格时公式/梯度allclose，保护其他匹配 |
| C1 | 真实GPU两步预检 | 原G＋对比目标替换 | PASS_ENGINEERING | preflight.json，batch8/两步更新/保存恢复通过；多正例数值含义另做复核 |
| C2 | 同预算继续训练控制 | 原G | NOT_STARTED | 保存空间待解决；不复用历史控制当作本次终态 |
| C3 | 同预算继续训练实验 | 原G＋对比目标替换 | NOT_STARTED | 无新指标；282新增对应下原生扩展损失为负，解释及尺度仍待复核 |
| C4 | 正式9508评估/候选与修复分解 | 两组 | TODO | 无新指标 |
