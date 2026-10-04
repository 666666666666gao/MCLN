# PV-Ground frozen-G local / whole complete range-source comparison

Original G fixed in eval mode, fresh AdamW, seed2027, batch8, head LR1e-5, weight decay5e-4, clip0.1.
Each arm consumes 29778 fit rows once in identical order: 3723 optimizer updates. All original-G parameters and persistent buffers stay fixed in eval mode; only the same 400614-parameter head learns. Native+G losses are computed, and native final-box losses train the head.

| Set / mode | Rows | Local hits @.25 / .50 | Whole hits @.25 / .50 | Whole minus local |
|---|---:|---:|---:|---:|
| initial / bbs | 6887 | 6176 / 5602 | 6176 / 5602 | +0 / +0 |
| initial / bbf | 6887 | 6206 / 5647 | 6206 / 5647 | +0 / +0 |
| terminal / bbs | 6887 | 6159 / 5537 | 6165 / 5539 | +6 / +2 |
| terminal / bbf | 6887 | 6189 / 5591 | 6194 / 5584 | +5 / -7 |
| formal / bbs | 9508 | 5588 / 4447 | 5589 / 4456 | +1 / +9 |
| formal / bbf | 9508 | 5621 / 4474 | 5622 / 4480 | +1 / +6 |

| Formal bbs threshold | Whole repairs vs local | Whole damages vs local | Net |
|---|---:|---:|---:|
| @25 | 6 | 5 | +1 |
| @50 | 26 | 17 | +9 |

| Formal arm / mode | Mask @.25 hits | Mask @.50 hits | Mask mIoU % |
|---|---:|---:|---:|
| local_range / bbs | 5812 | 5133 | 47.10806707 |
| local_range / bbf | 5847 | 5156 | 47.36791481 |
| whole_range / bbs | 5812 | 5133 | 47.10777331 |
| whole_range / bbf | 5847 | 5156 | 47.36762105 |

| Formal bbs arm | Same-Query coarse/final @.50 | Repairs / damages | Full256 good / missing errors | Median max-face move mm |
|---|---:|---:|---:|---:|
| local_range | 4495 / 4447 | 81 / 129 | 3534 / 1527 | 16.683310 |
| whole_range | 4495 / 4456 | 84 / 123 | 3518 / 1534 | 16.843475 |

Retained metric best: original_g (5615/4495). Target5615/4754 passed: False.
Original-G deltas: {'local_range': [-27, -48], 'whole_range': [-26, -39]}. Controller retention is verified separately; this analyzer deletes no weights.

Interpretation limits:

- Both arms start from original G with fresh AdamW. The only planned source change is the 109-D whole-instance range versus zeros; all 256 candidates remain available.
- The actual cross-process initial query, continuous box/Mask-IoU and threshold differences are recorded. Same-forward zero-head exactness does not establish cross-process bitwise pairing.
- The 6887 module holdout has scenes seen by author pretraining; the 9508 formal set is development validation.
- Full-256 coverage uses GT offline and does not certify physical instance identity or deployable achievable accuracy.
- All original-G parameters and persistent buffers remain fixed in eval mode; only the existing 10-tensor/400614-parameter refiner learns. Native stochastic sampling and numerical differences still require actual output comparison. Between-model repairs are not an independent reranking ablation.
- Within-model coarse/final comparison fixes the selected Query and native upstream forward. The coarse box comes from frozen original G in that forward, not a separately evaluated historical baseline.
- CPU reconstruction verifies selected box thresholds. Mask and full-candidate coverage recount saved native values; raw Masks and all candidate boxes are not independently reconstructed.
- Terminal weights were restored by the formal runner and hashes were checked before controller retention. This analyzer does not replay model or optimizer tensors, download or delete weights.
- One fixed seed supplies no cross-seed significance estimate. No new Nr3D/Sr3D, six-face distribution, geometry-quality readback or V99 teacher result exists.
