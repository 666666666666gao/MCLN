# PV-Ground semantic-only P2 continuation

Same original G start, fresh AdamW, seed2027, batch8, LR1e-5, 3723 updates/29778 fit rows once.

| Set / mode | G control | Joint P2 | Semantic-only P2 | Delta vs G |
|---|---:|---:|---:|---:|
| initial / bbs | 6176 / 5602 | 6176 / 5602 | 6176 / 5602 | +0 / +0 |
| initial / bbf | 6206 / 5647 | 6206 / 5647 | 6206 / 5647 | +0 / +0 |
| terminal / bbs | 6144 / 5552 | 6153 / 5579 | 6160 / 5571 | +16 / +19 |
| terminal / bbf | 6175 / 5600 | 6162 / 5591 | 6182 / 5606 | +7 / +6 |
| formal / bbs | 5600 / 4452 | 5613 / 4419 | 5588 / 4439 | -12 / -13 |
| formal / bbf | 5628 / 4465 | 5615 / 4401 | 5623 / 4460 | -5 / -5 |

| Formal bbs comparison | Threshold | Repairs | Damages | Net |
|---|---|---:|---:|---:|
| vs_g_control | @25 | 228 | 240 | -12 |
| vs_g_control | @50 | 346 | 359 | -13 |
| vs_joint_p2 | @25 | 232 | 257 | -25 |
| vs_joint_p2 | @50 | 389 | 369 | +20 |

| Formal bbs model | Full-256 @0.25 / @0.50 |
|---|---:|
| g_control | 8902 / 7755 |
| joint_p2 | 8896 / 7806 |
| semantic_p2 | 8893 / 7795 |

bbs_iou_groups_vs_g: reference rows, semantic-only P2 columns.

| Reference IoU | >0.50 | (0.25, 0.50] | <=0.25 |
|---|---:|---:|---:|
| strict_gt050 | 4093 | 206 | 153 |
| loose_only_025_to050 | 222 | 839 | 87 |
| miss_le025 | 124 | 104 | 3680 |

IoU endpoint groups do not establish physical instance identity or a causal loss mechanism.

bbs_iou_groups_vs_joint_p2: reference rows, semantic-only P2 columns.

| Reference IoU | >0.50 | (0.25, 0.50] | <=0.25 |
|---|---:|---:|---:|
| strict_gt050 | 4050 | 217 | 152 |
| loose_only_025_to050 | 274 | 815 | 105 |
| miss_le025 | 115 | 117 | 3663 |

IoU endpoint groups do not establish physical instance identity or a causal loss mechanism.

Historical G remains5615/4495; development target preserves5615 loose hits and requires4754 strict hits.
Target passed: False. Same-budget strict increment preserving loose: False.

Interpretation limits:

- All three trained arms start from the original G endpoint with fresh AdamW.
- The sealed G/joint-P2 arms are reused; no G control was retrained.
- Both boxes and scores can change across trained arms; this is not a fixed-box comparison.
- The semantic-only routing removes direct P2 geometry residual; shared/upstream parameters still train.
- Initial REC identity is checked, while historical initial Mask differences remain unexplained.
- Module holdout scenes were seen by author pretraining; formal9508 is development validation.
- Candidate oracle uses GT for offline diagnosis, and bbf stays secondary.
- One seed does not establish cross-seed significance or Nr3D/Sr3D effectiveness.
