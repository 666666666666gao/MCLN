# PV-Ground frozen-G residual / distribution boundary comparison

Original G fixed in eval mode, fresh AdamW, seed2027, batch8, head LR1e-5, weight decay5e-4, clip0.1.
Each arm consumes 29778 fit rows once in identical order: 3723 optimizer updates. All original-G parameters and persistent buffers stay fixed in eval mode; only the declared boundary head learns. Both use whole-range support; distribution adds six-face distribution targets with a separate1/7 coefficient. Native+G final-box losses remain.

| Set / mode | Rows | Residual hits @.25 / .50 | Distribution hits @.25 / .50 | Distribution minus residual |
|---|---:|---:|---:|---:|
| initial / bbs | 6887 | 6176 / 5602 | 6176 / 5602 | +0 / +0 |
| initial / bbf | 6887 | 6206 / 5647 | 6206 / 5647 | +0 / +0 |
| terminal / bbs | 6887 | 6168 / 5535 | 6174 / 5616 | +6 / +81 |
| terminal / bbf | 6887 | 6200 / 5592 | 6203 / 5654 | +3 / +62 |
| formal / bbs | 9508 | 5589 / 4446 | 5616 / 4506 | +27 / +60 |
| formal / bbf | 9508 | 5624 / 4469 | 5653 / 4526 | +29 / +57 |

| Formal bbs threshold | Distribution repairs vs residual | Distribution damages vs residual | Net |
|---|---:|---:|---:|
| @25 | 43 | 16 | +27 |
| @50 | 126 | 66 | +60 |

| Formal arm / mode | Mask @.25 hits | Mask @.50 hits | Mask mIoU % |
|---|---:|---:|---:|
| residual / bbs | 5812 | 5133 | 47.10762848 |
| residual / bbf | 5847 | 5156 | 47.36762105 |
| distribution / bbs | 5812 | 5133 | 47.10767130 |
| distribution / bbf | 5847 | 5156 | 47.36762105 |

| Formal bbs arm | Same-Query coarse/final @.50 | Repairs / damages | Full256 good / missing errors | Median max-face move mm |
|---|---:|---:|---:|---:|
| residual | 4495 / 4446 | 80 / 129 | 3516 / 1546 | 16.642533 |
| distribution | 4495 / 4506 | 20 / 9 | 3384 / 1618 | 5.602553 |

| Offline GT-volume rank quartile | Rows | Residual @.50 | Distribution @.50 | Difference |
|---|---:|---:|---:|---:|
| 1 | 2377 | 857 | 881 | +24 |
| 2 | 2377 | 1043 | 1073 | +30 |
| 3 | 2377 | 1106 | 1110 | +4 |
| 4 | 2377 | 1440 | 1442 | +2 |

GT-volume quartiles are offline diagnostics, with row_id tie breaks; no GT volume enters inference, and the grouping does not establish a causal size effect.

Retained metric best: distribution (5616/4506). Target5615/4754 passed: False.
Original-G deltas: {'residual': [-26, -49], 'distribution': [1, 11]}. Controller retention is verified separately; this analyzer deletes no weights.

Interpretation limits:

- Original G is fixed in eval mode and each arm starts from its protected parent with fresh AdamW; all256 candidates remain.
- Both use identical1302-D whole/global plus local support. Residual400614 versus distribution456102 parameters, face decoding and DFL/7 change together: representation-plus-supervision adaptation, not an isolated DFL effect or completed face-conditioned token method.
- Both apply the common1e-6 reference/final size floor. It preserves zero-head agreement with the same numerical protocol; it is not exact preservation of untreated negative native dimensions.
- Actual cross-process initial Query, continuous box/Mask-IoU and threshold differences are recorded, not assumed bitwise equal.
- The6887 module holdout scenes were seen by author pretraining;9508 formal rows are development validation. Single seed2027 gives no multi-seed significance estimate.
- Boundary targets use every actual final native Hungarian correspondence and training GT; none enter the inference head. Out-of-range targets saturate at the declared endpoints and are counted.
- Final native box/GIoU losses supervise the deployed frame. The first experiment keeps native scoring unchanged; no geometry-quality callback, P2, contrastive expansion, V99 teacher or dual ranking.
- Same-Query coarse/final evidence comes from this frozen-G forward; between-arm repairs are not an independently trained reranking ablation.
- CPU reconstruction verifies selected boxes and thresholds. Raw Masks and full candidate boxes are not reconstructed; their recorded native scalar IoUs and coverage are only recounted.
- Distribution entropy is a model statistic, not calibrated localization quality. Face offsets describe this parameterization, not guaranteed true instance boundaries.
- Formal runner restored the terminal delta and parent identity before retention. This analyzer never replays model/optimizer tensors, downloads or deletes weights.
- No new Nr3D or Sr3D result exists. Three complete proposed contribution claims and the full target remain pending actual evidence.
