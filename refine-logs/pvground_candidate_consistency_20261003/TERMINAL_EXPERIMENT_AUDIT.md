# Terminal experiment integrity audit

**Overall verdict: WARN.** The stored metrics, row identities, source hashes and training budgets agree with the primary artifacts. The result supports a qualified, one-seed ScanRefer development comparison: bbs gains 8/34 hits over the continuing control, while bbf, Mask mIoU and candidate coverage regress. It does not meet the declared historical-G or development target.

Date: 2026-10-03T15:36:11.167559+08:00. Reviewer: fresh native **gpt-6-astra / max**, `fork_turns=none`, task `/root/pvg_consistency_terminal_integrity`. Review independence: **same-family**. Acceptance: **provisional**. The actual request is recorded in `.aris/traces/experiment-audit/2026-10-03_terminal_run01/001-terminal-integrity.request.json`. The parent executor is recorded as `codex-current-session`; its specific model was not independently attested.

Context disclosure: the native review started with paths/checklist only. Workspace policy also required reading startup notes; those notes were not accepted as experiment evidence. Judgments and numbers in this report derive from the cited primary artifacts and reviewer CPU checks.

Paths below are relative to `C:\Users\gb\.codex\tmp\pvground_candidate_consistency_20261003`. No experiment source/evidence was modified; the only outputs are this audit and reviewer CPU-analysis artifacts under the trace directory. No SSH, GPU work, checkpoint copying or deletion was performed.

**Deterministic verification.** All **169** enumerated checks passed. The reviewer independently wrote a stdlib-only recomputation program without importing the author analysis. It read and hashed **63 files**, recounted **46,564 records in six evaluation JSONLs** and **7,446 logged training steps**, and recalculated **93,128 selected-box IoUs** from the saved boxes/root boxes. There were **zero** disagreements at either strict threshold; the largest continuous difference from recorded float32 IoU was `3.7845209630571475e-6`. Mask scalars and oracle flags were recounted and structurally checked; unavailable raw masks/full candidate tensors were not reconstructed.

The six evaluation JSONLs contain repeated evaluations of the same examples: two arms, initial and terminal on 6,887 holdout rows, plus two formal passes on 9,508 development rows. They are not 46,564 independent examples.

| Check | Verdict | Meaning |
|---|---|---|
| A. GT provenance | WARN | Dataset-to-evaluator source path supported; raw dataset/mask authenticity not locally reverified. |
| B. Normalization | WARN | Metric denominators pass; expanded semantic correspondence also changes effective loss weight. |
| C. Evidence/numerics | WARN | All deterministic checks pass; stale status and provenance/reporting limits remain. |
| D. Executed code | PASS | Claimed correction and evaluations have actual call paths and output witnesses. |
| E. Scope | WARN | One seed, one dataset, inherited G history, development use and failed exact start parity limit claims. |
| F. Evaluation type | PASS | `real_gt`, with GT-oracle quantities restricted to offline diagnosis. |

**Independently recounted results.** All accuracies use `IoU > threshold`, not `>=`. REC columns are integer hits; Mask mIoU is percent. Table entries reproduce both native receipts and `analysis/SUMMARY.json`.

| Stage / mode | N | Control REC @.25 / @.50 | Consistent REC @.25 / @.50 | Delta hits | Control / consistent Mask mIoU |
|---|---:|---:|---:|---:|---:|
| initial / bbs | 6887 | 6176 / 5602 | 6176 / 5602 | +0 / +0 | 74.19823767 / 74.19939023 |
| initial / bbf | 6887 | 6206 / 5647 | 6206 / 5647 | +0 / +0 | 74.43898288 / 74.44013545 |
| terminal / bbs | 6887 | 6144 / 5561 | 6153 / 5610 | +9 / +49 | 73.89662059 / 73.89023847 |
| terminal / bbf | 6887 | 6168 / 5589 | 6137 / 5595 | -31 / +6 | 74.01355129 / 73.84542307 |
| formal / bbs | 9508 | 5588 / 4423 | 5596 / 4457 | +8 / +34 | 47.09092428 / 47.04721234 |
| formal / bbf | 9508 | 5599 / 4427 | 5572 / 4403 | -27 / -24 | 47.06966849 / 47.05420757 |

Formal bbs accuracy is **58.77156079% / 46.51872108%** for control and **58.85570046% / 46.87631468%** for consistent, a **+0.08413967 / +0.35759361 percentage-point** difference. Formal Mask hit counts are **5821/5145 → 5812/5125** for bbs and **5820/5151 → 5825/5121** for bbf. Mask mIoU changes are **−0.04371195 pp** and **−0.01546092 pp**, respectively.

| Formal interface / threshold | Repairs | Damages | Net | Control / consistent coverable errors |
|---|---:|---:|---:|---:|
| bbs / .25 | 244 | 236 | +8 | 3316 / 3297 |
| bbs / .50 | 379 | 345 | +34 | 3357 / 3307 |
| bbf / .25 | 248 | 275 | -27 | 3305 / 3321 |
| bbf / .50 | 366 | 390 | -24 | 3353 / 3361 |

“Coverable” is determined from saved GT-qualified oracle flags. It is not a count of guaranteed policy repairs. Across trained arms the candidates and scores both change. The original-G reference **5615/4495** is supplied by the contract, not independently recounted in this intake. Conditional on that reference, consistent is **−19/−38 hits** and misses the **4754** strict-hit development target by **297**. Neither continuation replaces that reference leader.

| Formal interface | Threshold | Control Top-16/32/64/256 oracle hits | Consistent Top-16/32/64/256 oracle hits |
|---|---|---|---|
| bbs | .25 | 6217 / 6687 / 7701 / 8904 | 6151 / 6596 / 7616 / 8893 |
| bbs | .50 | 5395 / 5865 / 6810 / 7780 | 5343 / 5782 / 6729 / 7764 |
| bbf | .25 | 6376 / 6788 / 7721 / 8904 | 6170 / 6622 / 7635 / 8893 |
| bbf | .50 | 5540 / 5948 / 6825 / 7780 | 5336 / 5807 / 6765 / 7764 |

**Initial-comparison failure and replacement analysis.** The reviewer reran `analyze_complete.py`: it exits 1 at line 132 with the preserved exact-summary assertion and creates no output directory. The qualified analyzer exits 0, reproducing `analysis/SUMMARY.json` and `analysis/REPORT.md` byte for byte in a separate reviewer directory. Its diff replaces exact-summary equality with equality of eight REC/Mask threshold bitmaps and adds the diagnostic/hash and limitation. This is a post hoc relaxation of the analysis admission rule, not repaired functional equivalence.

| Initial field | bbs | bbf |
|---|---:|---:|
| Selected-query changes | 0 | 1 |
| Selected-box changes | 6886 | 6887 |
| Maximum selected-box coordinate difference | 0.0008502006530761719 | 0.02062702178955078 |
| Selected IoU changes | 5939 | 5966 |
| Selected Mask IoU changes | 2 | 2 |
| Maximum Mask IoU difference | 0.059084534645080566 | 0.059084534645080566 |
| Changed oracle .25 / .50 vectors | 2 / 1 | 1 / 0 |
| REC/Mask .25/.50 bitmap differences | all 0 | all 0 |

All 13,773 changed mode-records and every field of `analysis_initial_comparison/SUMMARY.json` were independently reproduced. Recorded row IDs, scene/target IDs, root boxes and point hashes match, including cross-arm terminal/formal alignment. This is not proof of all raw inputs or Mask GT equality. The cause of cross-process differences was not identified here.

**Training and checkpoint provenance.** Each arm has 3,723 sequential records (3,722 batches of eight and one of two), 29,778 unique fit IDs, identical batch order and no overlap with the 6,887 holdout IDs; their union is exactly `range(36665)`. Source `run.py:423` creates fresh AdamW after initial evaluation; `run.py:313` performs the optimizer step before the row is returned/written. There is no AMP skip path. All logged numeric values are finite. These are source/log-supported step counts, not a reviewer inspection of each terminal optimizer tensor.

The fit logs report 456 physical fit scenes; saved holdout records independently enumerate 106 scenes/1,479 scene-target pairs, and formal records enumerate 141 scenes/2,068 scene-target pairs. The saved holdout/formal scene sets do not overlap. The reviewer could not independently map all fit rows to raw scenes because the annotation/split manifest was not included.

The consistent arm records 1,022,313 additional qualified correspondences across 29,778 fit expressions, versus 938,378 CE-qualified queries in the continuing control. Same qualification logic does not mean identical qualification sets after the models diverge. Every consistent step has nonzero contrastive correction, ranging from −12.58745288848877 to +5.242063045501709. Its native final-head reconstruction error is exactly zero in the saved scalars. Expanded last-head values are negative in 3,638 steps; total losses remain finite and range 1.8724746704101562–20.465845108032227.

Checkpoint inheritance is author checkpoint → required G delta → this continuation delta. The run verifies/loads the author and G parents (`run.py:126`, `run.py:149`) and formal mode strictly loads the final delta (`run.py:204`). The collector hashes terminal files and matches the training receipt (`collect_complete.py:42`). The table below quotes those **remote witnesses**; this reviewer did not possess/hash/reload checkpoint bytes.

| Recorded remote artifact | Bytes | Recorded SHA256 |
|---|---:|---|
| g_control/terminal.pth | 342299951 | `a7c463cf22bca194650da3b997cca242843c7a327275e044ac0367168680120a` |
| g_consistent/terminal.pth | 342299951 | `2bbe2ae88e6dc993ad8c607fa24a7f9453cac781384e65ab1ad683e742762133` |

The declared G parent SHA256 is `0575dfae333dabdf288a470fc09d8967f9867d1853a5f61d9cfc3cbcf7964522`; author checkpoint SHA256 is `6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3`. These are not part of the local audited-input hash inventory. The two-step preflight receipt documents save/reload engineering checks only; it does not establish terminal optimizer reload. `load.json` is overwritten on formal entry, before terminal load, and its unconditional `fresh_optimizer` field is not such evidence. Formal completion supports that the terminal model load path ran, but no repeat-on-holdout output equality or terminal optimizer reload was performed by the reviewer.

All 39 collected-download hashes/sizes and all 11 recorded imported-module hashes match the local copies. The six separately collected imported sources match their intake. These are exact file checks; they do not certify the complete upstream release, raw dataset, CUDA dependencies or unavailable source manifests.

**A. Ground-truth provenance: WARN.**

PASS for recorded fields, selected-box geometry and the inspected source path; raw dataset provenance not independently verified.

- ScanRefer scene_id/object_id come from ScanRefer_filtered_{train,val}.json. Target boxes and masks come from scan.get_object_bbox(tid) and scan.three_d_objects[tid][points] in train/val_v3scans.pkl, with root at index 0.
- prepare() passes point/voxel tensors, text, GroupFree detection boxes/classes and superpoints into model.forward; center_label, size_gts and gt_masks are added only after forward for native_loss and evaluation. butd_gt and butd_cls are false; model-derived GroupFree detections are inputs, not the reference target.
- The native evaluator receives dataset center/size/mask targets. No undisclosed model-derived evaluation reference was found in the inspected path. Language alignment token maps are derived from the expression parser and are part of the native score interface. Native Mask training also includes prediction-to-prediction corresponding terms (models.losses.py:592, 612, 621); these are auxiliary training objectives, not the evaluator GT.
- Training qualification uses detached predicted boxes against training root GT and excludes all matched queries. Formal/holdout Top-16/32/64/256 oracle flags are GT-dependent offline diagnostics, not a deployable filter or demonstrated recovery policy.
- No raw annotation JSON, scan pickle/object implementation, raw masks, complete model dependency bundle or actual point arrays are in this intake. Exact equality of row_id, scan_id, target_id, root_box and point_sha256 does not establish equality of all inputs, token maps, detection arrays or masks.

Evidence: `complete/source/imported/src.joint_det_dataset.py:208`, `complete/source/imported/src.joint_det_dataset.py:594`, `complete/source/imported/src.joint_det_dataset.py:634`, `complete/source/imported/src.joint_det_dataset.py:1086`, `complete/source/imported/src.joint_det_dataset.py:1204`, `complete/source/imported/src.joint_det_dataset.py:1361`, `complete/source/run.py:262`, `complete/source/run.py:274`, `complete/source/run.py:355`, `complete/source/run.py:377`, `complete/source/pvground_semantic_assignment.py:9`, `complete/source/imported/evaluator.py:535`, `complete/source/imported/evaluator.py:879`, `analyze_complete.py:56`, `inspect_initial_difference.py:28`, `analysis/REPORT.md:28`.

**B. Score and loss normalization: WARN.**

PASS: metric denominators are evaluation-row counts or geometric/point unions, never prediction maxima, minima or means.

- REC is the count of selected-box IoU strictly greater than .25/.5 divided by 6887 or 9508; Mask mIoU is the unweighted mean of per-expression point intersection/union, multiplied by 100. bbs uses semantic-class softmax; bbf uses normalized query/token dot-products, fixed temperature .07 and token softmax. These are prediction interfaces, not result renormalization.
- The exact changed objective appends eligible unmatched queries with root target index 0 only for last-layer loss_sem_align. The original native matching indices remain in box/Mask losses. correction=(expanded_native-original_native)*(.5/7) replaces, rather than double-counts, the native final semantic term.
- Expanded correspondence also changes eos=.1 query weights to matched weight 1 and changes both query-to-token and token-to-query terms while retaining the original matched-count denominator. It is not a weight-controlled isolation of label consistency.
- All 3723 experimental updates record nonzero contrastive correction; 3638 expanded last-layer values are negative (minimum -148.15162658691406). This follows the reused weighted native formula; it is not a conventional guaranteed-nonnegative InfoNCE quantity and not evidence of accuracy. All logged numeric values are finite.
- The native logged loss_sem_align is the sum of seven pre-correction head values. loss_contrastive_correction and total loss carry the added replacement; do not present the former field alone as the final trained semantic objective.

Evidence: `complete/source/run.py:292`, `complete/source/run.py:303`, `complete/source/run.py:368`, `complete/source/run.py:385`, `complete/source/run.py:390`, `complete/source/run.py:403`, `complete/source/pvground_candidate_consistency.py:18`, `complete/source/pvground_candidate_consistency.py:27`, `complete/source/imported/models.losses.py:646`, `complete/source/imported/models.losses.py:697`, `complete/source/imported/models.losses.py:731`, `complete/source/imported/models.losses.py:749`, `complete/source/imported/models.losses.py:777`, `complete/source/imported/models.losses.py:943`, `complete/source/imported/evaluator.py:298`, `complete/source/imported/evaluator.py:897`, `complete/g_consistent/train.jsonl:1`, `EXPERIMENT_PLAN.md:31`, `analysis/REPORT.md:35`.

**C. Evidence existence and numerical agreement: WARN.**

PASS: 169 explicit checks, all six evaluation files, all recorded metric summaries, geometry thresholds, hashes, batch order and the initial diagnostic agree. Documentation/provenance limitations remain.

- Read and SHA256-hashed 63 requested input files. Recounted 46564 evaluation records, independently calculated 93128 selected-box IoUs from saved centers/sizes, and obtained zero threshold disagreements. Maximum float64-vs-recorded-float32 IoU difference is 3.7845209630571475e-6. Mask scalars and saved oracle flags were recounted, not recomputed from unavailable masks/all-candidate geometry.
- Both arms have 3723 sequential logged optimizer-step calls, 29778 distinct fit row IDs, 3722 batches of eight and one batch of two, identical batch order, and zero fit/holdout row overlap. Fit plus holdout IDs exactly cover range(36665). All four stage exits are zero and pair_status is complete.
- Training source creates fresh AdamW after initial evaluation, calls ordinary optimizer.step with no AMP skip path, and saves the final step rather than selecting a holdout-best checkpoint. The source/log chain supports completion; the reviewer did not deserialize optimizer state or prove every parameter updated on every step.
- Initial threshold bitmaps agree for REC and Mask in both modes, but output tensors are not identical: bbs boxes differ on 6886 rows; bbf boxes differ on 6887 rows and one bbf selected query changes. Continuous Mask IoU changes on two rows in each mode. The 13773 archived changed mode-records and all diagnostic fields were independently reproduced.
- Re-executed original analyzer locally: exit 1 at its line 132 assertion, no output directory created. Re-executed qualified analyzer in reviewer trace space: exit 0 and both generated files byte-identical to analysis/. The patch changes the analysis admission criterion and disclosure only; it does not restore exact functional-start parity.
- Collected endpoint hashes agree with training receipts, and formal source strictly loads the terminal delta before evaluation. The actual checkpoint bytes and parent weights were not downloaded, opened or reloaded by this auditor. load.json/imports.json are written on each invocation; the collected load timestamps are formal-stage, and load.json fresh_optimizer is not an optimizer-reload witness.
- EXPERIMENT_PLAN.md:3 and EXPERIMENT_TRACKER.md:7-9 are stale preterminal status snapshots despite completed primary receipts. analysis/REPORT.md discloses the failed initial assertion and limits, but Inputs/GT match needs the narrower saved-field qualifier. Mask degradation is present in SUMMARY.json/receipts but not tabulated in REPORT.md.
- Historical G 5615/4495 is a supplied contract constant, not independently recounted historical evaluation in this input set. Comparison/retention against it is conditional on that prior reference; current two-arm metrics are fully recounted.

Evidence: `complete/pair_status.json:2`, `complete/pair_status.json:28`, `complete/g_control/train.jsonl:3723`, `complete/g_consistent/train.jsonl:3723`, `complete/g_control/train.log:82`, `complete/g_consistent/train.log:82`, `complete/g_control/formal.log:26`, `complete/g_consistent/formal.log:26`, `complete/source/run.py:204`, `complete/source/run.py:307`, `complete/source/run.py:420`, `complete/source/run.py:426`, `complete/source/run.py:440`, `collect_complete.py:42`, `collect_complete.py:68`, `collect_imported_sources.py:25`, `complete/INTAKE.json:198`, `analyze_complete.py:132`, `prepare_initial_qualified_analysis.py:14`, `analyze_complete_initial_qualified.py:132`, `analysis_initial_comparison/SUMMARY.json:130`, `analysis_initial_comparison/SUMMARY.json:178`, `EXPERIMENT_PLAN.md:3`, `EXPERIMENT_TRACKER.md:7`, `analysis/REPORT.md:28`.

**D. Executed functions and dead-code claims: PASS.**

Claimed evaluation and training-correction routes are supported by code plus execution records; no claimed phantom metric or proposed module execution found.

- pair.py executes control/train, consistent/train, control/formal, consistent/formal. train mode calls evaluate(initial), step(update=True) 3723 times, then evaluate(terminal); formal mode loads terminal and calls evaluate(formal). All stages have corresponding outputs.
- Native loss head order is proposal_, last_, 0head_..4head_, so captured matching[1] is the intended final-layer matching. The consistent branch invokes candidate_consistency_correction, calls native loss_sem_align for original and expanded correspondences, adds its correction before backward and records nonzero values in every update.
- GroundingEvaluator.evaluate calls both bbox and both Mask functions. The runner checks native REC hit totals and Mask sums against its row writer. Top-5/10 and subgroup accumulators exist/execute but are not retained as reported audited metrics. print_stats and visualization branches are not execution evidence for additional results.
- verify_consistency_replacement and verify_native_replacement run only in the no-update preflight path, not on all training updates. The preflight receipt is a two-update engineering witness, not independent reviewer gradient/reload reproduction. p2 is asserted false; its unreachable installation branch is not claimed as executed.
- SourceReader/ObservationReader/TaskReader weights and architecture pre-exist in both G arms; the current comparison adds no new model state. Absence of direct box gradient from eligibility selection does not mean geometry predictions remain fixed after shared-parameter training.

Evidence: `complete/source/pair.py:36`, `complete/source/run.py:40`, `complete/source/run.py:281`, `complete/source/run.py:296`, `complete/source/run.py:323`, `complete/source/run.py:351`, `complete/source/run.py:417`, `complete/source/run.py:455`, `complete/source/imported/models.losses.py:852`, `complete/source/imported/models.losses.py:917`, `complete/source/imported/evaluator.py:194`, `complete/preflight/preflight.json:9`, `complete/preflight/preflight.json:52`, `complete/g_consistent/train.jsonl:1`.

**E. Scope and claim strength: WARN.**

The completed scope is two continuation arms at seed 2027 on one dataset, six evaluation passes; no cross-seed or cross-dataset claim is supported.

- Each arm consumes 29778 fit rows once; logs report 456 physical fit scenes. This latter scene count comes from the run log/source assertion, not independent raw annotation enumeration. The saved 6887-row module holdout contains 106 distinct physical scenes and 1479 scene-target pairs. The 9508-row development validation contains 141 scenes and 2068 scene-target pairs. Holdout/formal saved scene sets are disjoint.
- Both initial and terminal passes reuse the same holdout; 46564 records are repeated arm/stage evaluations, not 46564 independent examples. The module holdout is explicitly acknowledged as seen by author pretraining. The inherited G training history is additional to the matched continuation pass.
- Formal bbs improves by 8 loose and 34 strict hits (+.084139672 and +.357593605 percentage points) relative to the continuing control, but bbf loses 27/24 hits and both formal Mask mIoUs decrease. Candidate coverage also decreases. This is a mixed, small empirical result.
- The method remains 19/38 hits below the stated historical G 5615/4495 and 297 strict hits below the 4754 development target. It has not met the prespecified advancement criterion.
- The initial equivalence check was relaxed after seeing a failure, exact functional starts differ, eligibility distribution and effective loss weight change, and both boxes and scores can move. The evidence does not isolate pure label-consistency causality, fixed-box reranking, robust/significant gains, resolution of the sole failure cause, or physical-instance correctness of all IoU-qualified candidates.
- No new Nr3D or Sr3D evaluation, independent repetitions, cross-seed significance estimate or unbiased held-out test result is provided. The analysis already discloses most of these limits.

Evidence: `complete/g_control/spec.json:9`, `complete/g_consistent/spec.json:11`, `complete/g_control/train.log:8`, `complete/g_consistent/train.log:8`, `complete/g_control/initial/rows.jsonl:1`, `complete/g_control/formal/rows.jsonl:1`, `EXPERIMENT_PLAN.md:20`, `EXPERIMENT_PLAN.md:27`, `research_contract.md:5`, `analysis/REPORT.md:11`, `analysis/REPORT.md:13`, `analysis/REPORT.md:28`, `analysis/REPORT.md:34`.

**F. Evaluation type: PASS.**

real_gt by inspected dataset-to-evaluator provenance; raw-dataset authenticity remains outside this local audit.

- All six REC/Mask evaluation passes use ScanRefer-referred object boxes and ScanNet object point-membership masks, not a teacher/baseline reference, synthetic scene or human score.
- GT-dependent candidate-oracle diagnostics are real_gt offline upper-bound/coverage diagnostics. They are not deployable accuracy, selected-query policy performance, proof of instance identity, or a synthetic/self-supervised reference.
- The CPU analysis replay evaluates saved real-GT records. The two-update preflight is engineering validation on the recorded training batch and does not constitute another benchmark result.

Evidence: `complete/source/imported/src.joint_det_dataset.py:634`, `complete/source/imported/src.joint_det_dataset.py:1086`, `complete/source/run.py:372`, `complete/source/run.py:392`, `complete/source/run.py:395`, `complete/source/imported/evaluator.py:544`, `complete/source/imported/evaluator.py:888`.

**Blocking issues and reporting corrections.** `blocking_issues=[]` for reporting the narrow, qualified descriptive result. There is no observed evidence-integrity failure in the supplied records. This does not clear promotion, generalization/causal claims, raw-data certification, checkpoint deletion or deployment. The following corrections should accompany terminal reporting:

- **W1 (WARN):** Plan and tracker retain an 11:33 running/TODO snapshot although the pair completed at 14:58:13.702207+08:00. Update the terminal publication/status while preserving the old snapshot and the failed exact-comparison evidence. Evidence: `EXPERIMENT_PLAN.md:3`, `EXPERIMENT_TRACKER.md:7`, `EXPERIMENT_TRACKER.md:8`, `EXPERIMENT_TRACKER.md:9`, `complete/pair_status.json:28`.
- **W2 (WARN):** Inputs/GT exact is broader than the fields actually compared; exact starting outputs failed and the relaxed criterion was post hoc. Say recorded row IDs, scene/target IDs, root boxes and point hashes match; REC/Mask threshold bitmaps match; full inputs/raw GT and bitwise outputs were not independently verified. Evidence: `analyze_complete.py:56`, `inspect_initial_difference.py:28`, `prepare_initial_qualified_analysis.py:14`, `analysis/REPORT.md:28`.
- **W3 (WARN):** Mask decreases are in SUMMARY.json and native receipts but absent from the compact REPORT.md table. Include both formal Mask mIoUs and bbf negatives alongside bbs gains in the terminal-facing summary. Evidence: `research_contract.md:5`, `complete/g_control/formal/receipt.json:5`, `complete/g_consistent/formal/receipt.json:5`, `analysis/REPORT.md:7`.

**Claim impacts.**

| Claim | Impact | Limit / reason |
|---|---|---|
| Two arms completed 3723 logged steps over the same 29778 unique fit rows and six complete evaluation passes exist. | supported | Source/log/row evidence; not independent terminal optimizer-state deserialization. |
| Formal bbs consistent-control difference is +8/+34 hits. | supported_with_qualifiers | Descriptive one-seed development result; +.08414/+.35759 pp. No stable, statistically significant or pure-label causal gain established. |
| The current policy beats historical G or passes the development target. | unsupported | Using the declared historical reference, it is -19/-38 hits; 4457<4754. Historical baseline itself was not independently recounted here. |
| There is an across-interface or Mask improvement. | unsupported | Formal bbf -27/-24 hits; Mask mIoU decreases in both interfaces. |
| The comparison starts from bitwise-identical functional outputs and proves pure label-consistency causality. | unsupported | Initial exact analyzer fails; expanded correspondences change effective weight and shared-parameter geometry can change. |
| Full-256 coverage represents an executable inference filter or guaranteed repair count. | unsupported | GT-only offline diagnosis; raw full-candidate tensors were not provided to independently regenerate coverage. |
| All reported dataset-derived references and full raw inputs are independently verified. | unsupported | Source provenance and saved-field alignment are verified; raw dataset/masks/dependencies are not. |
| The candidate-consistency correction really ran and native box/Mask matching was not replaced. | supported_with_qualifiers | Correct final-head call path and nonzero logged corrections in all 3723 updates; direct gradient witness is preflight only. |
| Nr3D/Sr3D or robust multi-seed generalization is established. | unsupported | Only ScanRefer, seed2027, two trained endpoints. |
| Recorded endpoint hashes establish independently verified terminal optimizer reload and archived checkpoint availability. | unsupported | Remote collection receipts attest hashes. Checkpoint bytes were not in the intake; no reviewer reload/archive verification. |

**Audit limitations.**

- No SSH, network data access, GPU forward, training, checkpoint copy, deletion or mutation of the 63 audited inputs was performed by this reviewer.
- Raw ScanRefer annotations, scan/instance geometry, raw point tensors, masks, complete candidate coordinates/scores, parser caches and source manifests referenced remotely are absent from the supplied intake. Their runtime checks are source/log evidence, not independent local revalidation.
- Saved selected-box geometry is independently recomputed; Mask IoU scalars and candidate-oracle flags are only independently recounted and structurally checked. bbs/bbf selection rankings cannot be independently rerun from saved selected boxes alone.
- Checkpoint and parent file hashes quoted below are recorded remote witnesses, not hashes computed from checkpoint bytes by this reviewer. No terminal optimizer state inspection or checkpoint reload repeat was performed.
- Collected imports.json and load.json correspond to the last (formal) invocation. Source-port/custom-module hashes in training receipts support the training chain, but no immutable per-stage full import snapshot or whole dependency bundle is supplied.
- The claim that the evaluator is official is supported by the recorded native path/import hash and source inspected here, not an independent comparison to an upstream release or commit.
- The historical G baseline and prior author/G training history are declared inputs to this audit, not independently reconstructed historical runs.
- One seed, post hoc analysis-gate relaxation and changed effective objective weight limit causal/generalization conclusions. The same-family fresh reviewer route is provisional, not cross-family acceptance.

**Reproducible reviewer artifacts.** `reviewer_recompute.py`, `reviewer_recompute_results.json`, `reviewer_check_analyses.py` and `reviewer_analysis_replay.json` are under `.aris/traces/experiment-audit/2026-10-03_terminal_run01/`. The original-analyzer traceback, qualified replay and analysis diff are retained there. Recompute source SHA256: `a2a2c14eb13121e86de42a4412b0cc4b532651965b1328386fcb224fb82fe331`. All 63 source/evidence hashes were rechecked unchanged immediately before this report was written.

**Actual audited input SHA256 inventory.** These values were computed from the local bytes actually read, not copied from an author manifest. The machine-readable report includes file sizes and all independently recomputed metrics/check outcomes. Remote checkpoint witnesses are intentionally separate.

| Audited input | SHA256 |
|---|---|
| `analysis/REPORT.md` | `4baefb0a1ad6f0bc6329e6795d892d0a74a0f6f7f7bdece06f7a66546a32dff4` |
| `analysis/SUMMARY.json` | `907abe7d3f0cff4706ba0d4cbb70dae1cc4f5f16621d70149a8ececefa388519` |
| `analysis_initial_comparison/CHANGED_ROWS.jsonl` | `a85628d60d8381fbafcbd48e2be30387348cca5d2767fe645c93d0d8fd376fd5` |
| `analysis_initial_comparison/failed_analyze_complete.py` | `26dc25549f3999a8330c529a639082d516cea1b85e80d8d78d580d28af4259cd` |
| `analysis_initial_comparison/FAILED_ATTEMPT.json` | `8b197cfae0dbf6866e2ca09bd6b5e69486a8b59804b0fdb8d2a3bcef5f08fc0d` |
| `analysis_initial_comparison/SUMMARY.json` | `b3258e66ecad30b6051d8733d210bc8fd32d49ad2e92ae112be95d916e438aaf` |
| `analyze_complete.py` | `26dc25549f3999a8330c529a639082d516cea1b85e80d8d78d580d28af4259cd` |
| `analyze_complete_initial_qualified.py` | `a5d01e0cf5babeda9f440f8a3ab815ffb27b6a9606a01f89c9b6cd402589ce07` |
| `collect_complete.py` | `7df5d89ee423c92a39acb14e5995e0e321a9eb555de428c735812d9cf5538255` |
| `collect_imported_sources.py` | `90e159326973a882eb4cd093e30f5f3e422b49200ce13aa85f18d6126ea3a269` |
| `complete/g_consistent/formal/receipt.json` | `1cde0d7debdeaf2033fb71a622f1f9c72e2fa563b86589a44b77255a015aa36e` |
| `complete/g_consistent/formal/rows.jsonl` | `7b309abf838b5fc1c40632dc72d157d6b9abcb499ab3971dff6f82e1a09ae828` |
| `complete/g_consistent/formal.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `complete/g_consistent/formal.log` | `924e1d2eb58181e0866cd402309b5ae43817f6b96ab2ecc4c909ab0481fca603` |
| `complete/g_consistent/imports.json` | `c267bd1d40c062d0d3fccf30043085e6f772a6a53f1b7367e6fbdd5837216dac` |
| `complete/g_consistent/initial/receipt.json` | `33ae41151c9c5022c933210080ab16021073556af0f6f01c90b2bf3a24489825` |
| `complete/g_consistent/initial/rows.jsonl` | `5090806948ee43bb727aa4f73d8079968a43297b0524ad9836fa2d5e38f011f0` |
| `complete/g_consistent/load.json` | `cc52d6889dff9bca139402a2a710aab80f680c824249fe3745ce3ebbc7ac6a60` |
| `complete/g_consistent/receipt.json` | `1292350ad95b54274a5615cf0512b7cbb851d2d03a4342e3b065fd72d4a7b773` |
| `complete/g_consistent/spec.json` | `676e39b61b0451aa69fee82cac1d5c921207f77127ed8fa4bd988df1f321d156` |
| `complete/g_consistent/terminal/receipt.json` | `4a36b0b70e53af3a77aa1252b9212379223914086a98350848fb87ac5a814035` |
| `complete/g_consistent/terminal/rows.jsonl` | `4668eff89949aca74f19a52fe794c705bff4266402a7fda77cefdb14dd155deb` |
| `complete/g_consistent/train.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `complete/g_consistent/train.jsonl` | `dff3d5215083167d63c535bac209ccf556e098c1206ce2d77a32e0942013a034` |
| `complete/g_consistent/train.log` | `43540276351bb82a5fbc2a67462704478ae6514b6d6ceaf0bae7e78a95280590` |
| `complete/g_control/formal/receipt.json` | `9b430f63432cb414bdcb3fd29a4c5b9cef23e03d366ce8f3547fe3d4809a8777` |
| `complete/g_control/formal/rows.jsonl` | `d362a766d1b937da38ce2b7bb7285c3403ce2452f74322beddad773a2d87afc2` |
| `complete/g_control/formal.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `complete/g_control/formal.log` | `16b5c4ef6e8ea9174cc1ef8a6ce9153a6d7f6d1573cc53cf37e76eb4f5b3f775` |
| `complete/g_control/imports.json` | `c267bd1d40c062d0d3fccf30043085e6f772a6a53f1b7367e6fbdd5837216dac` |
| `complete/g_control/initial/receipt.json` | `f194404683a1a05c1ed5aff4495945f238bdcbbcb5d3acf6c6e2d5f92a57d138` |
| `complete/g_control/initial/rows.jsonl` | `a5ae84a9078a7ea678c597b03c4c23a4d461082e084d15c547ea6669e3b72f03` |
| `complete/g_control/load.json` | `922717546f3cab1f2120017b10c635aa2a53e7371a222e110b79efabc1096bd5` |
| `complete/g_control/receipt.json` | `c0fb48ed70ff5c1daee12357ad93bb737390a49ba209a3a414767145a3254d43` |
| `complete/g_control/spec.json` | `ddc5466d40f86274019b04e4d2fa0c3049d56df47767386048063a352b22786e` |
| `complete/g_control/terminal/receipt.json` | `c25ddc7e6972cf4b79517bb1cda1058adddfa471ba336ab630e1aee3a0bcba47` |
| `complete/g_control/terminal/rows.jsonl` | `f12936faf45c951b4dfe8e4b1d8cd40fb30c9cef27da743f93d623b08a1fc529` |
| `complete/g_control/train.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `complete/g_control/train.jsonl` | `8d69fe201c88af0aedaa11baee81432346658a29da42135f5237f4f107789202` |
| `complete/g_control/train.log` | `779f0c35dda08a632dc111ba73f957184964e48154f4e2b43aa2451797c4166a` |
| `complete/INTAKE.json` | `f649d2ef263d2a92e6689916be0f9da5cb9142ab0dff27743767e0cf64c5b062` |
| `complete/pair_status.json` | `c6713acdb3837f01ec7d473528ac9ee80900ab594c46ebcad38a0f9c4a91958b` |
| `complete/preflight/preflight.json` | `061c7de80812e170caf2be84b4c017456018aa25fb1fe706480e11134af6a260` |
| `complete/source/imported/evaluator.py` | `39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677` |
| `complete/source/imported/INTAKE.json` | `2785cb844ee75a68d5d19fba6bbb772d5834e4b86567be46f549661c64419a0d` |
| `complete/source/imported/main_utils.py` | `14bd101eb78ec3d729968aff0a975684efa8388574583ca9e4b0f512877b4f5c` |
| `complete/source/imported/models.losses.py` | `920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de` |
| `complete/source/imported/models.pv_ground.py` | `7fe369d7aaefda3f4e6eb389393799aedbe95ace56dbc107a7d26b0e3a3ea235` |
| `complete/source/imported/prepare_data.py` | `3a3de8bd54f675c5abac638be404cd47902951ae505caded34094f733838bd94` |
| `complete/source/imported/src.joint_det_dataset.py` | `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d` |
| `complete/source/pair.py` | `d6d428b1c574dd9e0e40b056ab83c34d5d07b4d92ddd52ac34880bafa9b65a60` |
| `complete/source/pvground_candidate_consistency.py` | `dc25727853c178f312db05f5f66c49fb56ecc0b72d390e91aa28dcd6f9085186` |
| `complete/source/pvground_observation_query.py` | `cc92906f9b6f388c8da702f555a099faf07d619fefa2772bfb229693ebae7742` |
| `complete/source/pvground_semantic_assignment.py` | `3bf9c5f50095e7e447af91e7f61a34c930bf04ec8590f7e4da54134b859cf773` |
| `complete/source/pvground_source_query.py` | `e0fcc6a30402ebeba7a9ffa1be907515350b64e408602a8b47b702e83a5d05ab` |
| `complete/source/pvground_task_observation_query.py` | `39349bf1fe8f3f51149f0872911020ecd0ede98fc9a8278cecb5ae466ed8643d` |
| `complete/source/run.py` | `919fc36d61b8f21a7285f55c153717f05baa4e969ca43ccbc25171e165bc7392` |
| `EXPERIMENT_PLAN.md` | `43516b3a0930b56fed0576544a854ea577a9b0f985c48cdeacdb687a6bdd54b2` |
| `EXPERIMENT_TRACKER.md` | `d481ef730fe8c24f3a140cbfcb94b58c9f1fc1ec145c78a54fdcae79d83f9814` |
| `inspect_initial_difference.py` | `80427da362ccae30acfaa9cf96d43685afd7b498b70616b53d70b2a77da4a922` |
| `prepare_initial_qualified_analysis.py` | `45d54a8ee56f0ad6fbe74e76fe773900da82488bcbf153c447490699015f0b11` |
| `research_contract.md` | `e2b0daa42dff9a092b8a1ed90ccf0073c63cfcd09b4e7a5cd3705389c5d0925e` |
| `WEIGHT_RETENTION.md` | `fc1de402f15d1067d2e90037342a485d1cb8728000574a9a7369fab24407e072` |
