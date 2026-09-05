# MCLN Nr3D experiment tracker

Updated: 2026-09-05 14:40 CST. Detailed evidence: master §§20.37–20.38.

| Run | Status | Evidence / next action |
|---|---|---|
| G0 old | COMPLETE | 1,611 updates, 25,768 fit rows; 7,151 heldout rows, REC hits 6,805 / 5,839 |
| G0 fixed | COMPLETE | Same updates and row order; REC hits 6,816 / 5,955; augmentation permissions 16,179 match normalized-text audit |
| G0 decision | SCIENTIFIC FAIL; INTEGRITY PASS | Overall +11 / +116; view-dependent @.25 is 0 net (20 fixes / 20 breaks). Both 612-file snapshots verified unchanged |
| Original G0→G1 | SEALED | Keep data fix; no promotion of failed augmentation performance route or restoration of canceled work |
| P1 padding identity | COMPLETE | Query ordering explains large raw-axis differences; four selected physical seeds and binary Masks stable. SWA token sensitivity remains; PR #7 stays draft |
| P1 candidate contract | V2 COMPLETE | v1 omitted formal size clamp; minimal correction and 5 CPU tests passed. Four real rows have 43/40/29/39 legal Queries and incomplete object memory coverage |
| Pair/global prototype | PASS; NO ACCURACY CLAIM | 10 scorer + 3 adapter tests, protected-weight CPU and real four-row GPU gradient probes passed; zero optimizer updates |
| Independent P2 v1 | LAUNCHED 14:28:37 CST | User's later relation comparison, separately registered before counts/results. Same frozen backbone forward; one epoch, two heads |

P2: screen `9428.mcln_p2_global_pair_train`, Python PID 9430, shared GPU lock
`/root/autodl-tmp/mcln_v99_backbone_gpu0.lock`. Addon:
`/root/autodl-tmp/mcln_g0_view_pair_20260905/pair_readout_train_v1`.
Log `training.log`; results only in new `results/`. At 14:40 both heads had 1,000 updates, 4,000 fit rows, 3,956 covered rows,
and finite losses; fit elapsed 357.476 s. Expected fit end ~15:14, evaluation
completion ~15:20–15:25. A read-only collector is scheduled for 15:18 CST
(screen mcln_p2_completion_window_1518), with no polling while it waits.

Fixed P2 split: 413 fit scenes / 26,747 rows (6,687 batches); 98 module-holdout
scenes / 6,172 rows (1,543 batches). Input manifest SHA:
`7a6d276f4c9bd745bc5d28fd7e7b12803a6a9dd485c2cddd7ae1d1d7aa701535`.
Five contract CPU tests and a first-fit-batch supervised GPU smoke passed:
four covered rows, finite loss/gradients, zero updates, no holdout evaluation.
No intermediate holdout or checkpoint selection. Save only the two final addon
heads; no completed P2 accuracy result is available at this update.

All train-scene holdouts exclude only the corresponding new updates. The
protected backbone already saw these scenes, so these audits do not establish
unseen-system generalization. New formal 7,899-row evaluations: zero. Protected
Nr3D remains 4,475 / 3,759 REC hits; protected ScanRefer/Sr3D are unchanged.
The overall three-benchmark objective is not complete.

Historical boundaries: raw CSV augmentation-permission changes total 2,155;
normalized text gives 325 (fit 253 / holdout 72). Fit 253 counts direct permission
changes, not every numerical input affected by worker RNG. The frozen G0
auxiliary logging defect did not affect its independent per-row decision.
