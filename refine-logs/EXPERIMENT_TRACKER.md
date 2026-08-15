# MCLN Optimization Experiment Tracker

Updated: 2026-08-15 15:06 CST

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R132-E1 | M0 | full-query adapter official | V132 epoch1 | ScanRefer 9508 | REC/Mask/mIoU | MUST | COMPLETE-REJECT | REC 5504/4391；Mask 5669/4660；mIoU 0.416989 |
| R132-E2 | M0 | duration/trend | V132 epoch2 | ScanRefer train+9508 | all metrics | MUST | RUNNING | 2026-08-15 15:04 为 1300/4583，无错误 |
| R132-E3 | M0 | duration/trend | V132 epoch3 | ScanRefer train+9508 | all metrics | MUST | QUEUED | same frozen config |
| R132-E4 | M0 | final V132 gate | V132 epoch4 | ScanRefer train+9508 | all metrics | MUST | QUEUED | same frozen config |
| R133-I | M1 | identity/contract | score-only SACR-Lite | synthetic + held-train | bit-exact/finite/coverage | MUST | PENDING V132 | V132 完全退出后才改源码 |
| R133-S | M2 | scene-disjoint gate | V133 2-epoch smoke | held-train 128/120 scenes | fix/break, REC, Mask | MUST | PENDING | formal gate: both thresholds fix≥break |
| R133-F | M3 | main result | V133 formal | ScanRefer 36665/9508 | REC hits, Mask, mIoU | MUST | PENDING | target REC ≥5610/4659 |
| R133-A | M4 | novelty isolation | no-relation/equal-param | ScanRefer | same official metrics | CONDITIONAL | PENDING B2 PASS | no new hyperparameter grid |
| R133-X | M4 | cross-dataset interface | same V133 module | Nr3D/Sr3D smoke | coverage/finite/REC | MUST-INTERFACE | PENDING B2 | box-only fallback where no mask GT |

