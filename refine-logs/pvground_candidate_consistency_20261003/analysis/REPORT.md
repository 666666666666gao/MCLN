# PV-Ground G / G+consistent fixed-budget comparison

Primary mode: `bbs`. Both arms reuse trained G, start fresh AdamW, and consume 29778 fit rows once (3723 updates).
Seed2027, batch8, core/backbone LR1e-5, weight decay5e-4, clip0.1. The shared G adaptation history is additional to this pass.
Only final contrastive correspondence replacement changes; both arms retain G CE replacement and all 256 candidates.

| Set / mode | Rows | G @0.25 / @0.50 | G+consistent @0.25 / @0.50 | Delta hits |
|---|---:|---:|---:|---:|
| initial / bbs | 6887 | 6176 / 5602 | 6176 / 5602 | +0 / +0 |
| initial / bbf | 6887 | 6206 / 5647 | 6206 / 5647 | +0 / +0 |
| terminal / bbs | 6887 | 6144 / 5561 | 6153 / 5610 | +9 / +49 |
| terminal / bbf | 6887 | 6168 / 5589 | 6137 / 5595 | -31 / +6 |
| formal / bbs | 9508 | 5588 / 4423 | 5596 / 4457 | +8 / +34 |
| formal / bbf | 9508 | 5599 / 4427 | 5572 / 4403 | -27 / -24 |

| Formal bbs threshold | Repairs | Damages | G coverable errors | Consistent coverable errors |
|---|---:|---:|---:|---:|
| @25 | 244 | 236 | 3316 | 3297 |
| @50 | 379 | 345 | 3357 | 3307 |

The ScanRefer development target preserves historical G loose hits (5615) and requires strict hits >=4754 (50%).
Target passed: False. Same-budget strict increment preserving loose: True.

Score leader for retention: original_g (5615 / 4495 hits). Original G remains a required parent. No weights were deleted by this analysis.

Interpretation limits:

- The original exact starting-summary assertion failed and is archived. Inputs/GT and initial REC/Mask threshold bitmaps match; selected boxes, some queries, continuous Mask IoU and Top-k oracle vectors differ across processes. This is a qualified same-budget empirical comparison, not bitwise-paired evidence.
- The 6887 module holdout includes scenes seen by author pretraining.
- The 9508 formal set is development validation.
- Candidate coverage uses GT only for offline diagnosis.
- Both frames and scores can change across these trained models; paired repairs are not fixed-box reranking gains.
- bbs and bbf are reported separately; primary selection remains bbs.
- One seed provides no cross-seed significance estimate.
- Expanding positive correspondences with the original matched denominator also changes effective objective weight.
- No candidates are dropped from forward inference; training qualification is not a deployable GT filter.
