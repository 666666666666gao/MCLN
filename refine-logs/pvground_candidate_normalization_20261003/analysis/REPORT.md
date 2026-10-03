# PV-Ground candidate normalization: completed three-arm comparison

All arms start from original G with fresh AdamW, seed2027, batch8, LR1e-5, weight decay5e-4 and clip0.1; 29778 fit rows are consumed once in identical order (3723 updates).
Existing G CE replacement, original model, native score and 256 candidates remain unchanged. Only the expanded final contrastive denominator changes from N to N+A relative to g_consistent.

| Set / mode | Rows | G control hits | Expanded N hits | Expanded N+A hits | N+A delta vs control | N+A delta vs expanded N |
|---|---:|---:|---:|---:|---:|---:|
| initial / bbs | 6887 | 6176 / 5602 | 6176 / 5602 | 6176 / 5602 | +0 / +0 | +0 / +0 |
| initial / bbf | 6887 | 6206 / 5647 | 6206 / 5647 | 6206 / 5647 | +0 / +0 | +0 / +0 |
| terminal / bbs | 6887 | 6144 / 5561 | 6153 / 5610 | 6153 / 5552 | +9 / -9 | +0 / -58 |
| terminal / bbf | 6887 | 6168 / 5589 | 6137 / 5595 | 6115 / 5461 | -53 / -128 | -22 / -134 |
| formal / bbs | 9508 | 5588 / 4423 | 5596 / 4457 | 5575 / 4426 | -13 / +3 | -21 / -31 |
| formal / bbf | 9508 | 5599 / 4427 | 5572 / 4403 | 5596 / 4399 | -3 / -28 | +24 / -4 |

| Formal bbs comparison | Threshold | Repairs | Damages | Net |
|---|---:|---:|---:|---:|
| N+A vs g_control | @25 | 247 | 260 | -13 |
| N+A vs g_control | @50 | 383 | 380 | +3 |
| N+A vs g_consistent | @25 | 229 | 250 | -21 |
| N+A vs g_consistent | @50 | 356 | 387 | -31 |

| Formal mode / arm | Mask @0.25 hits | Mask @0.50 hits | Mask mIoU % |
|---|---:|---:|---:|
| bbs / g_control | 5821 | 5145 | 47.09092428 |
| bbs / g_consistent | 5812 | 5125 | 47.04721234 |
| bbs / normalized | 5801 | 5102 | 46.85124199 |
| bbf / g_control | 5820 | 5151 | 47.06966849 |
| bbf / g_consistent | 5825 | 5121 | 47.05420757 |
| bbf / normalized | 5834 | 5125 | 47.04000644 |

Normalized delta from original G (5615/4495): -40/-69 hits. Development target 5615/4754 passed: False.
Initial threshold bitmaps all equal: True. Continuous differences and query/coverage differences are retained in SUMMARY.json.
Metric-best for weight retention: original_g (5615/4495 hits). Original G remains a necessary parent. This analyzer deletes no weights.

Interpretation limits:

- Same original-G weight, fresh optimizer and fit order; cross-process continuous starting outputs can differ. Actual initial REC/Mask bitmap and candidate/query differences are reported, without assuming parity.
- Normalization changes the scale of the entire expanded contrastive term, including original and background contributions; it is not a pure label-only or extra-positive-only change.
- The 6887 module holdout has author-pretraining-seen scenes; the 9508 formal set is development validation.
- GT candidate coverage is offline evidence, not a deployable filter or verified instance identity.
- Both boxes and scores can change through shared training. Repairs are not fixed-box reranking gains.
- All 256 candidates remain available; bbs and bbf are separate outputs, with bbs primary.
- One fixed seed supplies no cross-seed significance estimate; no new Nr3D or Sr3D results exist.
- This analysis recounts the saved native IoU/Mask-IoU records; it does not independently reconstruct all Masks or full candidate geometry from raw data.
