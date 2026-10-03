# PV-Ground候选监督一致性执行表

| Run ID | Purpose | Variant | Status | Evidence |
|---|---|---|---|---|
| C0 | 原生函数CPU检查 | 控制/目标替换 | PASS | cpu_test.json；无资格时精确等价，有资格时公式/梯度allclose，保护其他匹配 |
| C1 | 真实GPU两步预检 | 原G＋对比目标替换 | PASS_ENGINEERING | preflight.json，batch8/两步更新/保存恢复通过；多正例数值含义另做复核 |
| C2 | 同预算继续训练控制 | 原G | COMPLETE | 3723更新／29778条各一次；6887留出6144／5561；9508原生bbs5588／4423；complete/与analysis/ |
| C3 | 同预算继续训练实验 | 原G＋对比目标替换 | COMPLETE_NOT_PROMOTED | 3723更新／29778条各一次；6887留出6153／5610；9508原生bbs5596／4457；较原G−19／−38 |
| C4 | 正式9508评估/候选与修复分解 | 两组 | COMPLETE_WITH_LIMITS | bbs净+8／+34；严格修复379／破坏345；起点非逐位一致；TERMINAL_EXPERIMENT_AUDIT.md与analysis_initial_comparison/ |
