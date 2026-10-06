# PV-Ground terminal experiment audit

**Verdict: WARN. No unresolved blocking finding in the bounded reported experiment.** The four formal results independently recount correctly. The best strict result is the **fused-reference step-0 architecture: 5,598 / 4,848 hits**, not the trained fused terminal. New fused training reduces strict hits by16.

Review date: 2026-10-06T13:53:02.639856+00:00. Fresh reviewer context, continued after a communication recovery. Reviewer: `/root/pvg_mask_reference_terminal_integrity`; requested route `gpt-6-astra`, reasoning `max`; actual backend **not attested**. **Same-family; provisional.** This is a terminal source-and-actual-artifact review, not a historical SOURCE_ONLY verdict.

Paths below are relative to `D:\Program Files\UserCache\gb\codex\tmp\pvground_mask_reference_20261006`. `imported/` means `C:\Users\gb\.codex_mcln_g0_20260905\refine-logs\pvground_candidate_normalization_20261003\complete\source\imported`; `runtime_bundle/` means the sibling `pvground_final_quality_20261005/runtime_bundle`. Exact absolute paths and real-byte SHA256 identities are in `EXPERIMENT_AUDIT.json`.

## Verified results and scope

| State | New optimizer steps | Hits > .25 | Hits > .50 | Acc > .50 |
|---|---:|---:|---:|---:|
| Protected geometry parent | 3,723 in its prior stage | 5,616 | 4,511 | 47.444257% |
| Native initial | 0 | 5,615 | 4,495 | 47.275978% |
| Native trained | 3,723 | 5,617 | 4,510 | 47.433740% |
| Fused initial | 0 | 5,598 | 4,848 | 50.988641% |
| Fused trained | 3,723 | 5,593 | 4,832 | 50.820362% |

Each formal state has9,508 expressions,141 scans/physical scenes and2,068 scan-target pairs. The independent CPU checker verified4,756 NPZ files containing9,736,192 candidate rows and29,208,576 prior/reference/final box vectors. All4,819 intake members (647,499,194 bytes) match their manifests; all69 requested inputs exist. Every selected Query has the stored row's maximum native score, with no top-score ties. Recomputed float64 IoUs produce zero selected-threshold flips and zero full256 oracle-label mismatches. The largest selected IoU difference from stored float32 values is1.1153e-5; the check does not claim bitwise IoU equality.

Both actual fits contain29,778 unique rows in the same order and3,723 optimizer steps, batch8, final batch2. The6,887-row module holdout spans106 scans. Its initial/terminal hits are6176/5602 to6176/5612 for native and6149/5667 to6147/5665 for fused. This holdout is in the already trained parent universe. Evidence: `complete_fit/*/train.jsonl`, `complete_fit/*/receipt.json`, `analysis/terminal_review_scratch_20261006/checks.json`.

## A. Ground-truth provenance — WARN (source path passes)

The evaluated target is real dataset GT. ScanRefer split JSON supplies scene/object IDs (`imported/src.joint_det_dataset.py:585`); masks use annotated object-member indices and boxes use `scan.get_object_bbox` (`:1086`), with box jitter restricted to augmented training (`:1112`). The model receives points, text, superpoints and detector proposals; GT boxes/masks/identity are not forward inputs (`complete_fit/run_geometry_fit.py:268`). `butd_gt` and `butd_cls` are false (`:208`). Native bbs uses caption-derived token maps, whose parser consumes caption text (`imported/src.joint_det_dataset.py:1875`); it does not use GT geometry to select a Query.

Additional geometry eligibility intentionally consults **training** root GT and excludes all native matched candidates (`complete_fit/query_supported_geometry.py:27`). The spatial reference itself consumes predicted fused logits and member geometry only (`complete_fit/mask_reference.py:10`). Prediction-derived reference boxes are model inputs, not evaluation targets.

The underlying raw annotation/scan pickles are not among the69 local inputs. Thus this is a source/receipt-grounded provenance check, not independent raw-dataset reauthentication. The collected dataset, model, loss, preprocessing, main utility and native evaluator sources exactly match both execution import receipts.

## B. Score normalization — PASS

REC counts strict `IoU > .25`/`IoU > .50`, divided by9,508 expressions. Box IoU uses intersection/geometric union (`complete_fit/run_geometry_fit.py:291`; `imported/models.losses.py:70`). Mask IoU uses logical intersection/union and mIoU is its expression mean (`imported/evaluator.py:897`; runner`:345`,`:369`). Accuracy is not divided by the model's maximum or any predicted statistic. Model softmax, support-feature normalization and training losses do not alter the evaluation denominator.

The native-score Query indexes both Box and Mask (`run_geometry_fit.py:339`); the row hit totals and Mask sum are explicitly reconciled with the pinned native evaluator (`:365`). The evaluator's source hash is `39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677`.

## C. Actual results and state identity — WARN (numeric checks pass)

All formal receipts/rows, train logs, closure evidence, stored arrays and the later CPU restoration receipt exist and agree. The controller completed all six modes with exit0 (`complete_fit/fit_status.json:1`), and the observer records actual process closure (`fit_wait.json:3`). Paired rows match row ID, scan, target, root box and point hash. No phantom formal result was found.

Both arms retain8 hidden geometry tensors and reset the same2 output tensors (`complete_fit/mask_reference.py:69`), with only10 geometry tensors/456,102 parameters trainable (`run_geometry_fit.py:131`). Parent/G/Mask/semantic/R parameters remain frozen by the executed source and state-equality receipts. Initial and terminal checkpoints are separate (`:245`,`:468`,`:538`); two preflight updates are not carried into fresh formal/train processes (`controller.py:38`). The declared hidden history is11,169 prior updates and14,892 at the new terminal; output history is0 initial/3,723 terminal. The new logs verify3,723 directly; all ancestral PV/G/geometry histories were not re-audited here.

The selected receipt records `fused_mask_reference/initial.pth`, step0,1,828,167 bytes, SHA256 `2301e90a9391af4e4cd56edae09ce49db1cc8f67463e1c2568352ee2dafaff61`, empty Adam state,10 geometry tensors and equality of all1,304 full-model states against original construction (`selected_candidate_CPU_restore.json:105`). Its spec/factory/checker and summary hashes match actual local bytes (`:148`). The direct factory restores all geometry tensors without requiring the old geometry file (`postrun/selected_mask_reference_factory.py:46`); the old file is used by the checker only for a comparison witness (`postrun/restore_candidate_state.py:102`). These are audited receipt claims: weight bytes were not copied locally and this reviewer loaded no weights or models.

One concrete metadata defect remains: `analysis/SUMMARY.json:571` still says `selected_checkpoint_restore_pending=true`, while the later actual receipt and narrative say complete. Use the bound restoration receipt as authority and reconcile status in a later versioned administrative record. This does not change the checked counts and is not a blocker for the declared selection. Audit/retention pending statuses describe their earlier snapshot; retention has not been executed by this reviewer.

## D. Executed versus unused metric code — WARN

Every **claimed** REC/Mask metric is called and appears in saved results: controller→active runner→`evaluate`→native evaluator (`controller.py:36`, `run_geometry_fit.py:331`,`:468`; `imported/evaluator.py:194`). The inherited evaluator also runs BBF/subgroup routines, whose outputs this runner does not export. Its standalone NumPy `softmax` (`evaluator.py:37`) and the DIoU helper (`models.losses.py:103`) are unused on this path. Their presence is not evidence of additional measured metrics.

The bundled `run_final_quality_fit.py` and `native_final_quality.py` describe a different readback-quality training path. The current controller never invokes that runner. No learned-quality-ranking conclusion follows from those files. No unrelated cleanup or refactor is required.

## E. Experimental scope — WARN

This is one seed2027, one repeatedly used ScanRefer development split, two reference arms and two evaluated states per arm. The best state is selected on that same split (`postrun/analyze_mask_reference_formal.py:167`). The narrative accurately excludes new Nr3D/Sr3D performance, multi-seed robustness, zero-shot generalization, learned ranking and teacher transfer (`analysis/NARRATIVE_REPORT.md:21`). Keep those qualifiers in every downstream claim.

Parent→selected fused initial at strict.50 has713 repairs/376 damages/net337; at.25 it has174 repairs/192 damages/net−18. Initial native→initial fused has724/371/net353 at.50 and174/191/net−17 at.25. Trained native→trained fused has703/381/net322 at.50. Fused initial→trained has13/29/net−16 at.50 and4/9/net−5 at.25. These are independently verified paired development outcomes, not a successful new-training claim.

Equal point hashes and frozen weights do not establish identical fresh-process outputs. There are2 selected-Query differences between the two initial arms,3 between each initial and trained state, and3 between the parent and each initial state. The two trained arms have0 Query differences. The source of drift is not causally isolated by this audit (`analysis/SUMMARY.json:398`,`:426`).

## F. Evaluation type — PASS: real_gt

Primary REC and native Mask outputs are real-GT evaluations. The training support criterion is a GT-based eligibility proxy. Predicted extents remain model references. Preflight fixtures and CPU state restoration are integrity checks, not extra accuracy evaluations.

## Proof boundaries and claim impact

- **Supported:** the reported four formal hit pairs; full256 stored candidate retention; sole native-bbs selection; same-Query Box/Mask source route; declared fused initial metric-best state.
- **Supported with scope:** selected step0 CPU state reconstruction, using the actual bound receipt; the strict development improvement over the protected parent, while reporting its.25 regression and Query differences.
- **Unsupported:** improvement caused by the additional fused training; all9508 raw Mask-member reconstruction; a new GPU validation after restoration; multi-benchmark or multi-seed generalization.

The raw-member witness in preflight checks all256 references on the actual **eight-row batch**, reused for two updates per arm (`complete_fit/mask_reference.py:83`; `run_geometry_fit.py:484`; actual preflight receipts). It does not cover raw membership for every formal row. The separate39-case fixture contains recorded empty-support logits; this reviewer independently verified all39 have no positive fused logit (largest−0.1764220893), consistent with retaining the prior. Full formal NPZs contain boxes, scores, root GT and validity, not raw masks. The current CPU audit therefore cannot regenerate native scores from semantic logits or raw Mask IoUs/extrema. The selected receipt explicitly states `GPU_forward_replayed=false` and `raw_Mask_support_recomputed=false` (`selected_candidate_CPU_restore.json:144`).

**No unresolved blocking finding was identified for these bounded claims.** The WARN verdict preserves the missing raw/binary provenance, development-scope limits, inactive metrics and stale status field. It does not attest a model backend or authorize executing deletion. No SSH, GPU work, model/weight load, optimizer update, package installation or deletion occurred. Only these reports and scratch evidence under `analysis` were written. Previous SOURCE_ONLY verdicts were not used as terminal evidence.
