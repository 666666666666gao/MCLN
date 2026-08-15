# MCLN Optimization Experiment Tracker

Updated: 2026-08-12 03:06 CST

| ID | Variant | Status | Key setting | Result / receipt |
|---|---|---|---|---|
| R0 | protected/V19 baselines | complete | protected artifacts | learned 0.581195/0.465398; oracle 0.629680/0.550063 |
| R1 | V51 anchor-safe smoke | complete | 2 debug epochs; margins 0.05/0.10 | both epochs fixed 63/57 -> learned 64/58; Mask preserved |
| R2-S | V51 BMQ smoke | complete | safe Top16/8/4; pairwise 0.25; direct 0.25 | both epochs +1/+1 REC; Mask mIoU 0.350186/0.352024 |
| R2-F | V51 BMQ formal | running | 40 epochs; 4xA100; batch 12/GPU; 9508 eval | started 03:01:07; dataset text decoupling |
| R3 | V52 QTM-3D | pending R2 | query-specific text masks | pending |
| R4 | V53 DN-Group | pending oracle audit | last-two decoder layers | pending |
