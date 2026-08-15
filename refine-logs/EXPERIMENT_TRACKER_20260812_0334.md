# MCLN Optimization Experiment Tracker

Updated: 2026-08-12 03:34 CST

| ID | Variant | Status | Key setting | Result / receipt |
|---|---|---|---|---|
| R0 | protected/V19 baselines | complete | protected artifacts | learned 0.581195/0.465398; oracle 0.629680/0.550063 |
| R1-S | V51 anchor-safe smoke | complete | 2 debug epochs; margins 0.05/0.10 | both epochs fixed 63/57 -> learned 64/58; Mask preserved |
| R1-F | V51 anchor-safe formal | running | max delta 0.25; metric/pairwise/TopK off; 4xA100 | started 03:33:59; dataset loading |
| R2-S | V51 BMQ-safe smoke | complete | Top16/8/4; pairwise 0.25; max delta 0.50 | both smoke epochs +1/+1 REC |
| R2-F | V51 BMQ-safe formal | rejected | one full epoch / 9508 validation | learned 0.574043/0.455616; Mask 0.595183/0.486222/0.414974; stopped |
| R3 | V52 QTM-3D | pending R1 evidence | query-specific text masks | pending |
| R4 | V53 DN-Group | pending oracle audit | last-two decoder layers | pending |
