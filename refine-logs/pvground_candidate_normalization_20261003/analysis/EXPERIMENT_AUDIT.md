# PV-Ground normalization terminal experiment audit

**Overall verdict: WARN. Blocking issues: none for the narrow, qualified result and the reviewed fixed-target retirement boundary.**

All supplied terminal numbers agree with the independently recounted saved records. The normalized endpoint is nonleading: formal `bbs` is **5575 / 4426**, a difference of **−21 / −31** from expanded-N and **−40 / −69** from the declared historical original-G reference. The development target is not met. The prepared retirement script passes the static single-file boundary review; it was not executed.

**Review class:** fresh native reviewer context; `review_independence=same-family`, `acceptance_status=provisional`. Native agent `/root/pvg_normalization_terminal_integrity`. Archived native spawn arguments verify `model=gpt-6-astra`, `reasoning_effort=max`, `fork_turns=none`; this does not independently attest an underlying backend SKU or cross-family independence. Attribution: `N/.aris/traces/experiment-audit/2026-10-03_terminal_run01/001-terminal-integrity.request.json` (SHA256 `d797cd52ca729225214d1444f31352e5920573818882450295276c765944c585`). This additional attribution file is not scientific evidence.

**Evidence roots:** `N` = `C:\Users\gb\.codex\tmp\pvground_candidate_normalization_20261003`; `C` = `C:\Users\gb\.codex\tmp\pvground_candidate_consistency_20261003`. References below are exact file:line locations within these roots.

## Execution and evidence scope

65 requested files (65,751,121 bytes) were read and fingerprinted. The companion JSON records every absolute path, byte size, physical line count, SHA256 and read mode. All 26 JSON files, 12 NDJSON files, 17 Python ASTs, three exit markers and seven other text files were read. The evaluator/runner/analysis paths were reviewed directly; general model/data modules received focused source and complete static AST inspection. The historical comparator audit was treated as a document to check, not as proof of correctness. Required workspace startup notes were not used as experimental evidence.

Reviewer-written standard-library Python processed **69,846 evaluation rows**, **139,692 selected box/root-box pairs**, and **11,169 training records**. It imported no author analyzer, model or project module. It made no CUDA/SSH/job calls, opened no weight or credential bytes, spawned no agents, and changed no implementation or primary result. Only these audit reports were written.

The CPU report is [AUDIT_CPU_RECOUNT.json](AUDIT_CPU_RECOUNT.json), SHA256 `40a2a4e7ab3239e11441a9387734b45ba7dd08cebff84834cd771e86dce99b05`. All 33 collected-file manifests and 11 normalized import hashes match. The three arms have identical logged fit batch order; each has 3,722 batches of eight and one of two, with 29,778 distinct IDs consumed once. All 59 normalized stdout training-progress records match their corresponding train.jsonl records.

Selected-box IoU reconstructed from saved coordinates differs from saved float32 IoU by at most **3.7845209630571475e−6**, with **zero changed @.25/@.50 decisions**. This is an independent selected-box arithmetic check. Mask-IoU and oracle values were recounted and structurally checked; **no independent all-Mask reconstruction, full-256 geometry/ranking reconstruction, qualification replay or terminal optimizer replay was performed**.

## Recomputed formal results

All formal denominators are 9,508 expressions. REC and Mask threshold tests use strict `>`; percentages use the fixed row count.

| Arm | bbs REC @.25 / @.50 | bbf REC @.25 / @.50 | bbs Mask mIoU % | bbf Mask mIoU % |
|---|---:|---:|---:|---:|
| g_control | 5588 / 4423 | 5599 / 4427 | 47.09092428 | 47.06966849 |
| g_consistent | 5596 / 4457 | 5572 / 4403 | 47.04721234 | 47.05420757 |
| normalized | 5575 / 4426 | 5596 / 4399 | 46.85124199 | 47.04000644 |

| Normalized formal bbs comparison | Threshold | Repairs | Damages | Net |
|---|---:|---:|---:|---:|
| vs g_control | .25 | 247 | 260 | -13 |
| vs g_control | .50 | 383 | 380 | +3 |
| vs g_consistent | .25 | 229 | 250 | -21 |
| vs g_consistent | .50 | 356 | 387 | -31 |

Evidence: `N/complete/normalized/formal/rows.jsonl:1` through record 9508; corresponding native receipts; `N/analysis/REPORT.md:12`, `:17`, `:24`. All nine phases and both interfaces, including full Mask counts and offline oracle totals, are reproduced in the JSON reports.

Normalized fit completed at 19:04:41.440033 +08:00; terminal training receipt is 19:15:44.700248; the original controller completed formal evaluation at **2026-10-03 19:32:52.555688 +08:00**. `controller.exit`, `train.exit`, and `formal.exit` are zero. These are saved terminal observations; no new live check was made (`N/complete/normalized/train.log:82`, `:97`; `N/complete/status.json:16`).

## A–F checks

### A. Ground-truth provenance: WARN

PASS for inspected dataset-to-evaluator dataflow and saved selected-box geometry; raw dataset provenance not independently certified.

- ScanRefer_filtered_train/val.json supplies scene_id and object_id. Dataset targets are scan.get_object_bbox(tid) and scan.three_d_objects[tid][points]; center_label, size_gts and gt_masks are returned separately from model inputs. GroupFree predictions are inputs, not evaluation references.
- prepare() passes point/voxel tensors, expression text, detection boxes/classes/mask and superpoints; model.forward occurs before native_loss merges dataset GT. butd_gt and butd_cls are false, and the evaluator disables filter_non_gt_boxes. No prediction-derived evaluation GT was found.
- G qualification uses detached predicted training boxes versus the training root GT and excludes every Hungarian-matched query. This is supervised training. Oracle@16/32/64/256 uses GT solely for offline diagnosis.
- All row_id/scan_id/target_id/root_box/point_sha256 fields align within stages across the three arms. These five saved fields do not prove equality of complete input tensors, token maps, detection arrays, superpoints or raw masks. Raw annotation/scan files, parser dependencies and full source manifests are outside this 65-file intake.

Evidence: `N/complete/source/imported/src.joint_det_dataset.py:208`; `N/complete/source/imported/src.joint_det_dataset.py:594`; `N/complete/source/imported/src.joint_det_dataset.py:634`; `N/complete/source/imported/src.joint_det_dataset.py:1086`; `N/complete/source/imported/src.joint_det_dataset.py:1204`; `N/complete/source/imported/src.joint_det_dataset.py:1387`; `N/complete/source/run.py:262`; `N/complete/source/run.py:355`; `N/complete/source/pvground_semantic_assignment.py:9`; `N/complete/source/imported/evaluator.py:535`; `N/complete/source/imported/evaluator.py:879`.

### B. Prediction-derived metric normalization and training denominator: PASS

- REC uses selected-box IoU > .25/.5 and a fixed evaluation-row denominator, 6887 or 9508. Mask mIoU is the arithmetic mean of dataset-mask intersection/union, multiplied by 100. No reported metric is divided by a prediction maximum, minimum or mean.
- bbs semantic-class softmax and bbf query/token softmax at temperature .07 are native ranking interfaces, not metric rescaling. All 256 predictions remain available; the recorded Top-k oracle flags are not an inference filter.
- The native criterion orders heads proposal_, last_, 0head_...; captured matching[1] is the last head. The replacement computes old native last loss and new expanded last loss, then adds (new-old)*(.5/7). The new loss uses N+A and retains the same native formula, root-token maps, qualification rule and CE correction. No duplicate full contrastive term is added.
- Normalized records show sum N=29778, sum A=998995, sum(N+A)=1028773. Per-update (N+A)/N is 9.5 to 71.75, so for fixed tensors the whole expanded term is multiplied by N/(N+A)=0.0139373 to 0.1052632 relative to the expanded-N term. The per-update mean scale is 0.0313546, distinct from the aggregate count ratio 0.0289452. Original, background and added-correspondence contributions all scale; this is not an added-positive-only intervention.
- All 3723 correction records are nonzero and reconstruct the recorded formula within 2.725e-7. Native reconstruction error is zero. The native expanded loss is negative in 3628 steps; the unchanged native weighted formula permits negative values, and the full recorded losses remain finite. This is not evidence of score normalization fraud.
- The same qualification rule does not imply identical realized correspondence sets across training trajectories: expanded-N has 1022313 extra correspondences, normalized has 998995. No qualification or gradient replay was executed by this audit.

Evidence: `N/complete/source/pvground_candidate_consistency.py:12`; `N/complete/source/pvground_candidate_consistency.py:28`; `N/complete/source/imported/models.losses.py:646`; `N/complete/source/imported/models.losses.py:778`; `N/complete/source/imported/models.losses.py:852`; `N/complete/source/imported/models.losses.py:945`; `N/complete/source/run.py:281`; `N/complete/source/run.py:384`; `N/complete/source/run.py:403`; `N/complete/normalized/train.jsonl:1`; `N/complete/normalized/train.jsonl:3723`.

### C. Result existence, terminal identity and record agreement: WARN

PASS: all supplied results exist and the terminal numerical claims match; WARN: living plan/tracker retain launch-time states.

- All 65 listed files were read and hashed. All 26 JSON and 12 NDJSON files were parsed completely; all 17 Python files were parsed statically without importing project code. All 33 intake file size/hash entries and 11 normalized imported-source hashes match actual local bytes.
- All 69846 evaluation records and 11169 training records were independently recounted with standard-library CPU code. Nine evaluation receipts, three training receipts/transitions, both result summaries and all 59 normalized train-progress records agree. The three recorded training arms each contain steps 1..3723, 29778 unique fit IDs, 3722 batches of eight and one batch of two, in exactly the same batch order.
- Normalized controller/train/formal exit files are zero; status is complete at 2026-10-03T19:32:52.555688+08:00. INTAKE agrees and records controller absent, no GPU forward/optimizer updates during collection, and no checkpoint downloads/deletions. These are saved observations, not new live process queries.
- Selected-box IoU was independently recomputed from every saved box/root_box pair using Python float64. Maximum absolute difference from saved float32 IoU is 3.7845209630571475e-6; all .25/.5 decisions match. Raw Mask reconstruction and full-256 geometry/ranking reconstruction were not possible from these files.
- EXPERIMENT_PLAN.md:3 still says training started/no new precision, and EXPERIMENT_TRACKER.md:8-9 say RUNNING/NOT_STARTED. They are stale status text, not missing terminal results. Preserve launch.json as historical launch evidence and update the living status before terminal publication.
- The original-G 5615/4495 reference and earlier 16-weight deletion claim are historical assertions in supplied documents, not independently recounted/reverified primary events here. CPU-formula artifacts named in the tracker and comparator full import/load/exit files are not included in these 65 files; no independent certification of those unprovided artifacts is claimed.

Evidence: `N/complete/status.json:2`; `N/complete/controller.exit:1`; `N/complete/normalized/train.exit:1`; `N/complete/normalized/formal.exit:1`; `N/complete/normalized/receipt.json:45`; `N/complete/normalized/train.log:82`; `N/complete/normalized/train.log:97`; `N/complete/normalized/formal.log:26`; `N/analysis/REPORT.md:8`; `N/EXPERIMENT_PLAN.md:3`; `N/EXPERIMENT_TRACKER.md:8`; `N/WEIGHT_RETENTION.md:11`; `C/analyze_complete_initial_qualified.py:148`.

### D. Dead-code use and scorer call path: PASS

- run.evaluate calls GroundingEvaluator.evaluate, which calls both native bbox and both native Mask scorers. Native REC top-1 counts and Mask IoU sums are compared to recorded row metrics before a pass receipt is written; analysis helpers actually feed the published summary.
- Training calls native_loss and then the CE and normalized contrastive correction before backward, clipping and optimizer.step. The correction is nonzero in all normalized training records. Finite-gradient and restore witnesses are actual preflight records, not equivalent to an independent terminal optimizer replay.
- Legacy visualization helpers, evaluator print_stats/synchronization and unreported top-5/top-10/subgroup accumulators are not used for these reported claims. No claimed metric was found to depend on an uncalled function. The inactive p2 branch is rejected by the runner spec guard.

Evidence: `N/complete/source/run.py:40`; `N/complete/source/run.py:289`; `N/complete/source/run.py:313`; `N/complete/source/run.py:367`; `N/complete/source/run.py:405`; `N/complete/source/imported/evaluator.py:194`; `N/complete/source/imported/evaluator.py:897`; `N/analyze_complete.py:138`; `C/analyze_complete_initial_qualified.py:14`.

### E. Scope, comparison fairness and claim limits: WARN

- One ScanRefer continuation seed (2027), three trained endpoints: one new normalized endpoint and two reused completed comparators. Each consumes 29778 fit IDs once. The normalized log records 456 fit physical scenes; each initial/terminal file independently contains 6887 rows in 106 scenes, and each formal file contains 9508 rows in 141 scenes.
- Specs share original-G/official parent hashes, fit seed/budget/batch/LR and model/source settings after declared arm fields are removed. Runner and receipt hashes support the common implementation. However, comparator complete source/import/load histories and all raw training inputs are not separately supplied. Prior G adaptation and author pretraining are additional history, not part of this 3723-step budget.
- Initial threshold bitmaps all match. Against G control, normalized selected queries/boxes/REC IoUs match, but two selected Mask IoUs differ per interface (maximum 0.0204778). Against expanded-N, bbs 6886 and bbf 6887 selected boxes differ, bbf has one query change, two Mask-IoU changes per interface, and some oracle-vector changes. The comparison is not bitwise paired.
- Formal normalized bbs is 5575/4426: -13/+3 versus G control, -21/-31 versus expanded-N, and -40/-69 versus the declared historical G. bbf is 5596/4399. Both normalized formal Mask mIoUs are below both comparator mIoUs. These support a qualified negative/descriptive result, not useful stable gain, target success or promotion.
- The 6887 module holdout is declared to contain author-pretraining-seen scenes; the 9508 formal set is development validation. One seed supplies no cross-seed robustness/significance, no new Nr3D/Sr3D evidence, no fixed-box reranking interpretation and no deployable oracle filter. REPORT.md already discloses these key limits and includes Mask/bbf negatives.

Evidence: `N/normalized_spec.json:9`; `N/complete/source/run.py:149`; `N/complete/source/run.py:185`; `N/complete/normalized/train.log:8`; `N/analysis/REPORT.md:10`; `N/analysis/REPORT.md:24`; `N/analysis/REPORT.md:31`; `N/analysis/REPORT.md:37`; `C/complete/g_control/spec.json:9`; `C/complete/g_consistent/spec.json:9`.

### F. Evaluation type classification: PASS

Classification: `real_gt`.

- All nine REC/Mask evaluations are real_gt by the inspected source path: reference boxes and instance masks come from dataset objects. This class describes the path; it does not assert raw-data certification.
- Oracle coverage is a GT-dependent offline diagnostic within the real_gt evaluation. It is not synthetic GT, a deployable inference policy or a guaranteed number of future recoveries. Native prediction-to-prediction Mask consistency terms exist in training but are not evaluation ground truth.

Evidence: `N/complete/source/imported/src.joint_det_dataset.py:1100`; `N/complete/source/imported/evaluator.py:544`; `N/complete/source/imported/evaluator.py:888`; `N/complete/source/imported/models.losses.py:592`; `N/complete/source/run.py:395`.

## Prepared nonbest-endpoint retirement: PASS_STATIC_ONLY

The sole proposed target is `/root/autodl-tmp/pvground_candidate_normalization_20261003/normalized/terminal.pth`, recorded as **342,299,951 bytes**, SHA256 `01f928653357ab5bcc8bff2cb86a33db109d5f5a3a8c3d7c6b8cb25a6ac60cb0`. This reviewer did not open its bytes or delete it.

- Local gates require audit PASS/WARN with no blockers, original_g score leader, normalized_endpoint_keep false, complete intake, controller exit 0 and consistent finish time.
- Only one remote unlink call exists. The item path must exactly equal the fixed normalized/terminal.pth under the fixed experiment root, its resolved path must match, and its size/SHA must match the collected terminal item and training receipt. Remote controller absence and complete train/formal states (3723 steps/9508 rows) are rechecked.
- Original G is hashed before and after deletion. No parent/V99/log/result/source/archive path is a deletion target; no new archive is created. The remote retirement receipt is written before stdout and then fetched/compared locally, so loss of the local observation after completed remote execution leaves a recoverable remote receipt.
- This is a static boundary check. No SSH, checkpoint-byte hashing, deletion or live recheck was performed. It does not promise an atomic unlink-plus-receipt transaction under arbitrary process crash; no observed requirement or evidence justifies adding unrelated recovery logic.
- Original G is the leader under the declared historical 5615/4495 reference, which was not independently recounted here. Normalized is also nonleading among independently recounted endpoints because expanded-N has 4457 versus normalized 4426 strict hits. Original G remains the necessary parent irrespective of that historical performance reference.

Evidence: `N/retire_completed_nonbest.py:12`; `N/retire_completed_nonbest.py:23`; `N/retire_completed_nonbest.py:31`; `N/retire_completed_nonbest.py:42`; `N/retire_completed_nonbest.py:49`; `N/retire_completed_nonbest.py:51`; `N/retire_completed_nonbest.py:59`; `N/retire_completed_nonbest.py:74`.

No concrete deletion-boundary defect was found. The script targets neither the original G, official parents, MCLN/V99 chain, logs, result rows nor source files. A completed remote deletion writes its remote receipt before local observation, so a lost local receipt can be recovered from the remote file without re-executing deletion. This is a static finding, not an assertion that deletion occurred or that all historical archives remain available.

## Nonblocking actions and claim limits

- **W1 — Plan and tracker retain pre-completion states despite verified terminal records.** Update current terminal status/links while retaining historical launch and raw logs. Evidence: `N/EXPERIMENT_PLAN.md:3`; `N/EXPERIMENT_TRACKER.md:8`; `N/complete/status.json:16`.
- **W2 — Same correspondence set wording can only mean the same qualification rule. Realized extra-correspondence totals differ (1022313 versus 998995), and the denominator scales the entire expanded term.** Retain the whole-term-scale qualification and describe a common qualification rule, not a frozen identical realized set or pure added-positive effect. Evidence: `N/EXPERIMENT_PLAN.md:1`; `N/EXPERIMENT_PLAN.md:9`; `N/complete/source/pvground_candidate_consistency.py:14`; `N/analysis/REPORT.md:38`.
- **W3 — Raw dataset/Mask/full-256 tensors, terminal optimizer bytes and original-G primary formal rows are outside the supplied review scope. Initial continuous outputs differ across processes.** Report a one-seed development result with explicit saved-record and historical-reference limits. No broad raw-data/optimizer/bitwise-parity certification. Evidence: `N/analysis/REPORT.md:37`; `N/analysis/REPORT.md:44`; `N/analyze_complete.py:157`.

The stale status can be corrected in the terminal publication while preserving the original launch snapshot. No implementation change, new training, extra compatibility/fallback/retry layer, hashing framework or unrelated refactor is requested.

| Claim | Assessment |
|---|---|
| Three saved training arms each completed the common 3723-step/29778-row continuation budget, with one normalized arm newly run. | supported_with_qualifiers — Full log/receipt/source agreement; no terminal optimizer-byte replay. |
| Normalized formal bbs is 5575/4426 and its difference from expanded-N is -21/-31. | supported — Exact saved-record recount; one-seed development result. |
| Normalized improves original G, satisfies 5615/4754, or is the best retained endpoint. | unsupported — 5575/4426 fails both declared historical-G preservation thresholds and the target; strict score also trails audited expanded-N. |
| N+A normalization changes only the last expanded contrastive denominator while keeping native evaluation/CE/regression/Mask formula wiring. | supported_with_qualifiers — Static source and nonzero logged replacement checks; all expanded-term contributions scale, and realized qualification sets evolve separately. |
| Exact functional starting parity, raw-mask/full-256 reconstruction, complete optimizer replay or robust causal/generalization acceptance is established. | unsupported — These are explicitly outside the saved evidence and execution scope. |
| Prepared retirement is bounded to this one completed nonleading normalized endpoint and retains a remote receipt after successful remote execution. | supported_static_only — Script inspected, not executed; no deletion or weight availability claim. |

The historical 5615/4495 original-G score is declared rather than independently recounted in this intake. Even without that historical reference, normalized is not best among the three independently recounted endpoints. Original G remains a required parent. The report does not independently certify the earlier 16-weight cleanup or upstream official-evaluator identity.

All 65 original file identities were rechecked unchanged before these reports were written. The machine-readable verdict, exact evidence paths and complete identities are in [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json).
