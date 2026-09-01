# MCLN Nr3D 视角增强修复审计 Tracker

Updated: 2026-09-01 22:06 CST

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| VA0 | B0 | 冻结新 salted split 与代码差异 | CPU closure | Nr3D train only | counts/SHA/overlap/diff | MUST | SPLIT PASS / CODE PENDING | 404/107 scenes，25768/7151 rows，overlap0；old/fixed tree closure 待实现 |
| VA1 | B1 | 旧增强谓词配对角色 | old snapshot, E57→E58 | salted fit→holdout train | REC .25/.50 + view-dependent | MUST | BLOCKED BY GPU QUEUE | 不抢占独立 ScanRefer 队列 |
| VA2 | B1 | 修复增强谓词配对角色 | fixed snapshot, E57→E58 | same salted fit→holdout train | same metrics | MUST | BLOCKED BY VA1/GPU | rows/batches/steps 必须与 VA1 相同 |
| VA3 | B2 | 机械比较并作一次性决策 | no-GPU comparator | VA1 vs VA2 | hit deltas/gates | MUST | PENDING | 禁止阈值、salt、fold、LR、epoch 扫描 |
| VA4 | B2 | 完整 Nr3D 同架构验证 | fixed only | formal 7,899 | REC@0.25/@0.50 | CONDITIONAL | NOT AUTHORIZED | 仅 VA3 全门 PASS 后授权 |
