# PV-Ground G / G+P2 fixed-budget comparison

Primary mode: `bbs`. Both arms reuse trained G, start fresh AdamW, and consume 29778 fit rows once (3723 updates).
Seed2027, batch8, core/backbone/P2 LR1e-5, weight decay5e-4, clip0.1. The shared G adaptation history is additional to this pass.

| Set / mode | Rows | G @0.25 / @0.50 | G+P2 @0.25 / @0.50 | Delta hits |
|---|---:|---:|---:|---:|
| initial / bbs | 6887 | 6176 / 5602 | 6176 / 5602 | +0 / +0 |
| initial / bbf | 6887 | 6206 / 5647 | 6206 / 5647 | +0 / +0 |
| terminal / bbs | 6887 | 6144 / 5552 | 6153 / 5579 | +9 / +27 |
| terminal / bbf | 6887 | 6175 / 5600 | 6162 / 5591 | -13 / -9 |
| formal / bbs | 9508 | 5600 / 4452 | 5613 / 4419 | +13 / -33 |
| formal / bbf | 9508 | 5628 / 4465 | 5615 / 4401 | -13 / -64 |

| Formal bbs threshold | Repairs | Damages | G coverable errors | P2 coverable errors |
|---|---:|---:|---:|---:|
| @25 | 238 | 225 | 3302 | 3283 |
| @50 | 334 | 367 | 3303 | 3387 |

The ScanRefer development target preserves historical G loose hits (5615) and requires strict hits >=4754 (50%).
Target passed: False. Same-budget strict increment preserving loose: False.

Interpretation limits:

- The 6887 module holdout includes scenes seen by author pretraining.
- The 9508 formal set is development validation.
- Candidate coverage uses GT only for offline diagnosis.
- Both frames and scores can change across these trained models; paired repairs are not fixed-box reranking gains.
- bbs and bbf are reported separately; primary selection remains bbs.
- One seed provides no cross-seed significance estimate.
