# Cached geometry experiment integrity review

**Overall verdict: WARN. Blocking issues: none for qualified factual reporting of these cached analyses.**

All three new summaries reproduce exactly from the retained cache. This supports descriptive geometry statistics for the old completed fused P3 checkpoint. It does not establish training cause, physical-instance identity, recoverable accuracy, or efficacy of the active G candidate-consistency pair.

Reviewer: gpt-6-astra, reasoning effort max, fresh native agent /root/pv_cached_geometry_integrity. Parent-confirmed spawn used fork_turns=none. Review independence: **same-family**. Acceptance: **provisional**. Review timestamp: 2026-10-03 03:09:27 UTC.

## Scope and method

Path aliases for exact source references:

- W = C:/Users/gb/.codex/tmp/pvground_candidate_consistency_20261003
- D = C:/Users/gb/.codex/tmp/pvground_fused_support_20261002/candidate_audit/full_gt_scope_v2
- S = C:/Users/gb/.codex/tmp/pvground_fused_support_20261002/complete_tail_fused_retry/source/imported

I read every line of the three new analysis scripts and D/run.py. I independently processed **9,508 JSONL rows and all 1,189 NPZ files, comprising 2,434,048 candidates and 129,459,547 array-file bytes**. The first 1,188 chunks contain 8 rows each; the final chunk contains 4. IDs are exactly 0–9507. There are **141 scan IDs/physical scenes and 2,068 scene-target pairs**. No candidate was removed.

The independent implementation did not import or rerun the author's analysis scripts. It checked array shapes, finite values, IoU/probability ranges, file sizes, row identity, selected-query fields, score maxima, root matching, root/scene overlap booleans, per-row qualification and rank bounds. It recomputed all floor counts, all 55 floor examples, all quantiles, overlap groups, joint-query statistics, and float64 box IoUs. Every new reported count agrees. All first-qualified rank score bounds coincide: no tie ambiguity affects any rank or >32 count. None of the selected scores has a tie.

The JSON companion contains actual counts, the independent CPU source, and **1,211 audited input SHA-256 hashes**, including all 1,189 chunks. All summary source, row and chunk digests match current bytes. Local dataset/evaluator source hashes match D/imports.json:11,16.

Only these two review files were created. **No SSH, network, GPU forward, optimizer update, model/runtime mutation, candidate pruning or new paired accuracy result occurred.** The old cache was read on CPU; no checkpoint was deserialized and no training state was queried live.

## A. Ground truth provenance — PASS

ScanRefer expressions and target IDs come from annotation JSON (S/src.joint_det_dataset.py:585–648). Root boxes come from the target object's bbox; root masks come from annotated instance point membership, not predictions (S/src.joint_det_dataset.py:1086–1119). These helpers are called during dataset retrieval and returned as center_label, size_gts and gt_masks (S/src.joint_det_dataset.py:1324–1334,1387–1392).

D/run.py:238–241 loads validation with butd_gt/butd_cls false. Forward inputs contain point/voxel data, text, detector boxes and superpoints; spatial GT is added after the model forward for criterion/evaluation (D/run.py:269–286,308–336). Root box IoU uses the expression target (D/run.py:335–337); root masks use batch gt_masks (D/run.py:363–375).

The comparison scene list is restricted to detection-vocabulary objects and the native object-slot limit (S/src.joint_det_dataset.py:1121–1157). Root GT is independently scored, including **87 rows** where the root is absent from that list (D/run.py:344–355; D/receipt.json:16). The overlap summary explicitly limits physical-identity interpretation (W/audit_cached_error_proxies.py:83–88).

This verifies source provenance and matching archived source hashes. The original annotation files, point memberships and full scene boxes are not in the audited cache. I did not reconstruct every mask/scene IoU from raw GT or re-verify current remote data/checkpoint bytes. The source-defined real-GT evaluation is not a model-generated reference.

## B. Prediction-dependent score normalization — PASS

Box IoU is intersection/union (D/run.py:299–305). Mask IoU uses point intersection/union after the native fused-logit threshold (D/run.py:357–375; S/evaluator.py:594–605,897–902). Counts use strict **>0.25** and **>0.5**, not >=. No reported metric divides by a prediction maximum, minimum or mean.

Native softmax and text-component score addition/subtraction are candidate scoring, not normalization of reported accuracy (D/run.py:324–328; S/evaluator.py:254–303). The new scripts report direct counts/quantiles (W/audit_cached_box_floor.py:50–87; W/audit_cached_error_proxies.py:45–78; W/audit_cached_selected_box_mask.py:21–41).

## C. Result existence and count agreement — WARN

**All new summary arithmetic passes independent verification.** All listed inputs/results exist. The old cache has exit 0 and its completion-log JSON exactly equals receipt.json (D/run.exit:1; D/run.log:26). It has **5,566 selected hits@0.25 and 4,406 hits@0.5**, leaving **3,942** and **5,102** errors. These are the old completed fused-cache results, not a new paired-training result (D/receipt.json:11–21).

### C1. Evaluator-clamped size floor

All cached coarse/final size components are >= float32(1e-6). The counts exactly match W/cached_p3_box_floor/SUMMARY.json:8–21:

| Count | Recomputed |
|---|---:|
| candidates | 2434048 |
| coarse_floor_candidates | 561889 |
| final_floor_candidates | 563465 |
| introduced_floor_candidates | 1576 |
| matched_root_final_floor_rows | 66 |
| qualified50_final_floor_candidates | 0 |
| removed_floor_candidates | 0 |
| rows_with_final_floor_candidate | 9441 |
| selected_coarse_floor_rows | 53 |
| selected_final_floor_rows | 55 |
| selected_floor_strict_errors | 55 |
| selected_floor_with_qualified50_alternative | 6 |
| selected_introduced_floor_rows | 2 |

All **55** saved selected-floor examples match the actual arrays and rows. Every row has one query matched to GT slot 0; therefore summing matched-root floor candidates also equals 66 affected rows in this cache.

Selected minimum-size quantiles exactly match W/cached_p3_box_floor/SUMMARY.json:23–30:

| Quantile | Recomputed minimum size |
|---|---:|
| 0 | 9.999999974752427e-7 |
| 0.01 | 0.04553420074284077 |
| 0.1 | 0.15181758999824524 |
| 0.5 | 0.4420914649963379 |
| 0.9 | 0.7780136704444887 |
| 0.99 | 1.0879816031455998 |
| 1 | 1.9154678583145142 |

The generator clamps last_pred_size before caching and clamps coarse size separately (D/run.py:319–323). These are **evaluator-clamped sizes**. A floor value cannot distinguish an original negative size, zero, or a small positive value. “563,465 raw negative boxes” would be unsupported. The supplied script correctly discloses missing preclamp sizes (W/audit_cached_box_floor.py:89–94).

Only **2** selected floor events are newly present at the final stage; **53** already existed at the coarse stage. The marker covers **55 of 5,102** strict errors, leaving **5,047** strict errors without a selected floor hit. Neither this frequency nor the 6 floor-error rows with a qualified alternative establishes that eliminating a floor event would correct selection.

### C2. Strict-error overlap proxies

The exact mutually exclusive selected-query predicates are:

1. root IoU > 0 and root IoU >= the maximum included scene-GT IoU;
2. the included scene-GT maximum strictly exceeds root IoU;
3. both values are zero.

These are implemented at W/audit_cached_error_proxies.py:41–53 and independently checked against arrays. All-zero ties are correctly separate. “Nearest” means **largest box IoU**, not physical distance. The first group permits ties: **1,313 of 1,333 rows** tie the scene maximum. In the second group all 3,750 best-scene IDs differ from the target ID, which still does not establish a predicted box's physical identity.

The three columns are positive root maximum / strictly larger scene maximum / all-zero within the included annotation scope. Counts match W/cached_p3_error_proxies/SUMMARY.json:7–44:

| Count | Root maximum, positive | Scene maximum larger | All-zero |
|---|---:|---:|---:|
| strict_errors | 1333 | 3750 | 19 |
| has_qualified_box | 761 | 2639 | 8 |
| no_qualified_box | 572 | 1111 | 11 |
| has_unmatched_qualified_box | 716 | 2486 | 7 |
| has_qualified_box_and_mask_same_query | 669 | 2417 | 7 |
| has_unmatched_qualified_box_and_mask_same_query | 635 | 2278 | 6 |
| selected_loose_hit | 1057 | 103 | 0 |
| first_qualified_rank_beyond32 | 159 | 1749 | 8 |
| root_absent_from_scene_detection_GT | 20 | 44 | 1 |

Of 5,102 strict errors, **216** selected queries are root matched and **4,886** unmatched. Native evaluation matching is not a physical foreground/background label (D/run.py:339–343,429–433). “No annotated overlap” means no overlap with the independent root or the included restricted scene list, not with every scene object.

Overall, **3,408** strict-error rows have any qualified box; **3,209** have an unmatched qualified box; **1,694** have none among all 256. **1,916** have their first qualified box ranked after 32. These are descriptive oracle-availability counts, not measured rescues.

### C3. Same-query box/mask conditions

The selected-query table matches W/cached_p3_selected_box_mask.json:7–11:

| Selected query | Mask >0.5 | Mask <=0.5 |
|---|---:|---:|
| Box >0.5 | 4,075 | 331 |
| Box <=0.5 | 1,008 | 4,094 |

Each selected box and mask IoU was independently matched to the **same selected_query** in its NPZ row. The 1,008 box misses with mask hits are evaluation facts, not evidence of a specific training conflict.

For alternative candidates, the script uses an elementwise conjunction before reducing over queries (W/audit_cached_error_proxies.py:55–64). Exactly **3,093** strict-error rows contain a query with box>0.5 and mask>0.5; **2,919** contain an unmatched query satisfying both. Separate existential checks for any qualifying box and any qualifying mask would produce **3,169** rows, overcounting the actual same-query condition by 76. The supplied summary uses the correct conjunction.

Selected-mask subgroups match W/cached_p3_selected_box_mask.json:13–31:

| Count | Root maximum, positive | Scene maximum larger | All-zero |
|---|---:|---:|---:|
| strict_box_errors | 1333 | 3750 | 19 |
| selected_mask_hit50 | 940 | 68 | 0 |
| zero_box_overlap_mask_hit50 | 0 | 7 | 0 |
| selected_mask_hit50_with_qualified_box | 613 | 40 | 0 |

**Required distinction:** selected_mask_hit50_with_qualified_box totals **653** (613+40). It means the selected query's mask succeeds while an alternative query's box qualifies. The code combines selected mask_hit with existence of a qualified box (W/audit_cached_selected_box_mask.py:26–38). These rows are strict selected-box errors: **zero** of the 653 is a same-selected-query box/mask success. The top-level same_selected_query flag describes the 2x2 selected table; do not extend it to this auxiliary field.

### C4. Preserved CPU/GPU discrepancy

Independent float64 geometry reproduces the existing warning exactly (D/CPU_RECOUNT.json:9–24; D/CPU_THRESHOLD_DIFFERENCES.json:5–31):

- Maximum absolute CPU/GPU box-IoU difference: **5.394267800662433e-06**.
- One threshold mismatch: **row 3499/query 189**, an unselected unmatched candidate, at 0.25.
- Native IoU **0.24999968707561493**; CPU IoU **0.2500001376978458**.
- @0.25 qualified total: native **379,732**, CPU **379,733**. Unmatched: **370,849** versus **370,850**; root matched **8,883**, other matched **0**.
- @0.5 qualified total: **257,085** on both. Root matched **7,738**, unmatched **249,347**, other matched **0**.
- All selected .25/.5 decisions, every candidate .5 decision and all error-row “has alternative” counts agree.

No tolerance erased this discrepancy or changed native decisions. C is WARN to preserve that numeric limit; it is not a failure of the new summary counts. Comparisons against archived formal outputs in D/CPU_RECOUNT.json:75–79 were not re-audited here; this review covers the supplied cache and the three new analyses.

## D. Called versus dead code — PASS

Each new script has one main function containing its aggregation and an executed module entry-point call (W/audit_cached_box_floor.py:9–106; W/audit_cached_error_proxies.py:9–100; W/audit_cached_selected_box_mask.py:7–57). Every reported aggregation was independently reproduced. There is no unused new metric advertised as a result.

The original cache path calls model forward, native loss/matcher and official evaluator, then writes root box/mask and scene-overlap arrays (D/run.py:308–412). box_iou is called for root, coarse and scene boxes (D/run.py:299–305,335–348). Its successful receipt comes after state-preservation/evaluator-agreement assertions (D/run.py:419–435). The archived evaluator reaches its box/mask methods (S/evaluator.py:194–206).

The original runner accepts only formal mode (D/run.py:37,46), leaving its inherited CPU-mode and non-formal FitDataset branches unreachable (D/run.py:176–178,191–208,244–263). These are not new metric functions and receive no result credit. No CPU model-restore claim follows from that branch.

AST inspection finds no optimizer construction/update, backward or model.train call in the three analyses or cache runner. GPU_forward_executed=false correctly refers to **these new CPU analyses**. The old cache itself required a GPU forward (D/run.py:179,310); do not conflate these execution scopes.

## E. Scope, cause and active-experiment separation — WARN

This is **one old completed tail_fused P3 checkpoint, seed2027, one ScanRefer development-validation set, 141 scenes and 9,508 expressions** (D/run.py:80–90,210–243; D/receipt.json:11,20). It supports descriptive full-candidate statistics. It does not establish multiseed stability, unseen-test generalization, a deployed selector's gains, a training cause, or full physical/semantic correctness of IoU-qualified candidates.

The analyses explicitly read the old fused cache (W/audit_cached_box_floor.py:10,92; W/audit_cached_error_proxies.py:10,88; W/audit_cached_selected_box_mask.py:8,49). The active G pair has distinct roots and a semantic_consistency false/true contrast (W/g_control_spec.json:2,32–35; W/g_consistent_spec.json:2,32–35). The plan excludes P3/fused-tail use from that pair (W/EXPERIMENT_PLAN.md:13–20). **No count here evaluates the active control/treatment, and no new paired accuracy is available from these files.**

A nonblocking **status-document freshness correction** is needed: W/EXPERIMENT_PLAN.md:3 and W/EXPERIMENT_TRACKER.md:7–9 still say formal training NOT_STARTED. The retained observation records g_control/train and a logged **step1088** at **2026-10-03 09:36:15 CST** (W/pair_checkpoint_observation.json:3–6,25,55). Those lines are stale relative to existing evidence. This was a file read, not a new live query. The observation does not establish current progress, treatment completion or a formal pair result. The parent confirmed it would preserve the audited original versions until delivery.

## F. Evaluation type — PASS: real_gt with diagnostic proxies

Root box/mask measurements are **real_gt**. Identity interpretations of maximum scene-box overlap are explicitly limited **GT-based geometry proxies**, not model-generated synthetic ground truth (D/run.py:335–375,429–433; W/audit_cached_error_proxies.py:83–88). Floor incidence is a prediction-geometry diagnostic with real-GT REC outcomes attached, not a standalone performance gain.

## Required claim qualifications and corrections

There is no blocking numerical/code issue for reporting the actual cached counts. Retain these limits:

- Say evaluator-clamped floor incidence, not recovered raw negative-size incidence.
- Define restricted maximum-IoU proxy groups, ties and the separate all-zero case; do not call them confirmed physical-instance or language-error categories.
- Distinguish **653 selected mask successes plus an alternative qualified box** from **3,093 jointly qualified candidate rows** and **2,919 unmatched jointly qualified candidate rows**.
- Preserve the one native/CPU @0.25 discrepancy and both raw candidate totals.
- Keep old fused P3 diagnostics separate from active G-pair training; no training-cause or new-accuracy conclusion follows.
- Update plan/tracker status from the dated observation after preserving audited input hashes; do not extrapolate later progress.
- Do not claim raw reconstruction of candidate masks, scene maxima, Hungarian costs or preclamp sizes: the necessary inputs are absent.

Deterministic count agreement is evidence within this scope. Semantic interpretation remains same-family/provisional; this is not cross-family acceptance.

