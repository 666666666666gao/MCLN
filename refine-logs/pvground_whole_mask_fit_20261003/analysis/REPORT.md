# PV-Ground local / whole complete range-source comparison

Original G, fresh AdamW, seed2027, batch8, LR1e-5/core and backbone, weight decay5e-4, clip0.1.
Each arm consumes 29778 fit rows once in identical order: 3723 optimizer updates. Both use the same 400614-parameter head and native+G losses.

| Set / mode | Rows | Local hits @.25 / .50 | Whole hits @.25 / .50 | Whole minus local |
|---|---:|---:|---:|---:|
| initial / bbs | 6887 | 6176 / 5602 | 6176 / 5602 | +0 / +0 |
| initial / bbf | 6887 | 6206 / 5647 | 6206 / 5647 | +0 / +0 |
| terminal / bbs | 6887 | 6160 / 5591 | 6156 / 5559 | -4 / -32 |
| terminal / bbf | 6887 | 6180 / 5600 | 6178 / 5587 | -2 / -13 |
| formal / bbs | 9508 | 5603 / 4428 | 5594 / 4461 | -9 / +33 |
| formal / bbf | 9508 | 5637 / 4431 | 5603 / 4458 | -34 / +27 |

| Formal bbs threshold | Whole repairs vs local | Whole damages vs local | Net |
|---|---:|---:|---:|
| @25 | 247 | 256 | -9 |
| @50 | 383 | 350 | +33 |

| Formal arm / mode | Mask @.25 hits | Mask @.50 hits | Mask mIoU % |
|---|---:|---:|---:|
| local_range / bbs | 5806 | 5114 | 46.93181071 |
| local_range / bbf | 5823 | 5137 | 47.06934219 |
| whole_range / bbs | 5811 | 5106 | 46.92723284 |
| whole_range / bbf | 5805 | 5110 | 46.87424777 |

| Formal bbs arm | Same-Query coarse/final @.50 | Repairs / damages | Full256 good / missing errors | Median max-face move mm |
|---|---:|---:|---:|---:|
| local_range | 4428 / 4428 | 14 / 14 | 3323 / 1757 | 2.705010 |
| whole_range | 4471 / 4461 | 13 / 23 | 3311 / 1736 | 2.828427 |

Retained metric best: original_g (5615/4495). Target5615/4754 passed: False.
Original-G deltas: {'local_range': [-12, -67], 'whole_range': [-21, -34]}. Controller retention is verified separately; this analyzer deletes no weights.

Interpretation limits:

- Both arms start from original G with fresh AdamW. The only planned source change is the 109-D whole-instance range versus zeros; all 256 candidates remain available.
- The actual cross-process initial query, continuous box/Mask-IoU and threshold differences are recorded. Same-forward zero-head exactness does not establish cross-process bitwise pairing.
- The 6887 module holdout has scenes seen by author pretraining; the 9508 formal set is development validation.
- Full-256 coverage uses GT offline and does not certify physical instance identity or deployable achievable accuracy.
- Both boxes and scores can change during shared training. Between-model repairs are not fixed-box reranking gains.
- Within-model coarse/final comparison fixes the selected Query in a jointly trained model; the coarse box is not an independent baseline.
- CPU reconstruction verifies selected box thresholds. Mask and full-candidate coverage recount saved native values; raw Masks and all candidate boxes are not independently reconstructed.
- Terminal weights were restored by the formal runner and hashes were checked before controller retention. This analyzer does not replay model or optimizer tensors, download or delete weights.
- One fixed seed supplies no cross-seed significance estimate. No new Nr3D/Sr3D, six-face distribution, geometry-quality readback or V99 teacher result exists.
