# Archived P3 selected-query refinement magnitude

This is an offline description of the completed raw-point P3 model. It does not evaluate the active same-tail raw/fused experiment and does not change a historical score. Source row hashes are checked against the original intake; `SELECTED_REFINEMENT_MAGNITUDE.json` records the executed analysis and script hash.

| Saved stage, native bbs | Rows | Median maximum face shift | 99th percentile maximum face shift | Largest maximum face shift | Coarse / final hits at 0.50 | Repairs / damages |
|---|---:|---:|---:|---:|---:|---:|
| Module holdout terminal | 6887 | 3.267 mm | 7.311 mm | 9.808 mm | 5603 / 5614 | 23 / 12 |
| Formal development validation | 9508 | 3.261 mm | 7.305 mm | 9.311 mm | 4403 / 4401 | 15 / 17 |

The maximum face shift is the largest absolute change among the six axis-aligned faces computed from the stored coarse and final center/size for the SAME selected Query. The original evaluation runner clamps BOTH coarse and final sizes to at least 1e-6 before recording; these statistics describe saved evaluated box geometry, not unclamped head residuals. Positive saved sizes do not prove the raw head always produces valid dimensions. Units inherit the ScanNet evaluation protocol and are not independently calibrated here. All 9508 selected formal predictions have maximum face displacement below 1 cm; this does not describe all 256 candidates. The formal median of maximum face shift divided by the corresponding GT axis size is 0.005282 (about 0.53%); this uses GT only for offline diagnosis, not inference.

On the 326 formal rows whose coarse IoU lies in [0.45, 0.50), the refiner repairs 15 strict errors, while the mean continuous IoU change is -0.000414. The 348 rows in [0.50, 0.55) have 16 strict damages. Formal mean IoU changes by +0.000337 over all rows, yet strict hits fall by 2. A small increase in continuous average therefore did not deliver the required threshold improvement.

Interpretation: the saved evaluated selected-query correction is mostly a millimetre-scale adjustment under this run. This is consistent with its small internal threshold effect; it does not establish whether the cause is inadequate instance support, parameter learning rate, gradient clipping, optimization duration, or another training interaction. It does not measure unclamped regression output or how much joint training changed upstream boxes or candidate ranking, and it is not a trained-without-P3 ablation.

Next step remains the already running fixed-budget same-tail raw versus predicted fused-Mask support pair. Compare both formal REC and these direct correction statistics after completion. Do not alter the active raw run or expand residuals/LR merely from this observation. If support is ineffective, use the completed pair to choose the next mechanism rather than claiming Mask evidence automatically transfers V99's benefit.

Evidence limits: terminal and formal contain different scenes and cannot prove a causal generalization mechanism; full candidate arrays, local member points and regression logits were not stored. Coordinates, GT boxes and IoUs are saved evaluation artifacts. Original P3 formal bbs remains 5594 / 4401; the same-Query coarse 4403 is an internal intermediate, not an independent baseline. No new GPU forward, training update or Nr3D/Sr3D evaluation occurred.
