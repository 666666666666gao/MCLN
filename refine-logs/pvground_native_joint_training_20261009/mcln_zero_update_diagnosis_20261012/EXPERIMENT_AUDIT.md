# Experiment audit: cached MCLN zero-update diagnosis

**Verdict: PASS for the final qualified diagnostic. No blocking findings.**  
**Execution scope:** REAL_CACHED_MCLN_ZERO_UPDATE_DIAGNOSIS  
**Review:** fresh same-family Codex reviewer; provisional. Actual model and effort: UNATTESTED.  
**Date:** 2026-10-12

Independent local Python/NumPy recomputation matched all **346 numeric diagnostic values** in each result version exactly (692 comparisons). All **28,524 MCLN selected-box IoUs** match float32 geometric recomputation exactly. The PV size-product formula reproduces every threshold decision; its largest scalar difference is 5.960464477539062e-7, so PV IoUs are not claimed bit-exact.

The initial expression-pairing label was overstated and was corrected during review. Numeric tables did not change. Use the [qualified RESULT](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/zero_update_diagnosis_qualified_20261012/RESULT.json). The original source and result remain historical evidence.

## F1: cross-model expression identity was overstated — RESOLVED

The original source checked row index, scene/target, point hash and GT, then published “same_expression_order_scene_target_GT_box_and_pointcloud_sha=true.” PV rows do not retain text or annotation identity. There are only **141 unique point hashes** among 9,508 records, and **2,052 of 2,068 scene/target objects** have multiple distinct MCLN expressions. MCLN indices 0, 31 and 32, for example, have different expressions but identical scene, target and point hash. Matching these fields cannot independently certify expression identity.

Evidence: [original assertions:16](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/zero_update_diagnosis_20261012/analyze_zero_update_transfer_r1_unqualified.py:16), [original label:54](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/zero_update_diagnosis_20261012/analyze_zero_update_transfer_r1_unqualified.py:54), [original RESULT:8](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/zero_update_diagnosis_20261012/RESULT.json:8), [PV record writer:316](C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/paired_span_loop.py:316).

The final source/result separates verified row/scene/target/GT/point alignment from “exact_cross_model_text_or_annotation_pairing_verified=false” and labels cross-model intersections as aligned saved-row comparisons. This is sufficient for the bounded task; it requires no new inference.

Evidence: [final source:54](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/analyze_zero_update_transfer_20261012.py:54), [qualified RESULT:8](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/zero_update_diagnosis_qualified_20261012/RESULT.json:8).

## Checks A–F

### A. Ground-truth provenance — PASS with local-evidence limits

The loader reads ScanRefer validation annotations. Target boxes and masks originate from dataset object boxes and point membership, not predictions. The MCLN replay attaches batch labels after forward and records root GT separately. PV formal GT is taken from dataset center/size labels. All 9,508 root boxes agree between PV/MCLN and across the three MCLN arms.

Evidence: [dataset annotations:585](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/source/src/joint_det_dataset.py:585), [dataset GT:1085](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/source/src/joint_det_dataset.py:1085), [MCLN labels:119](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/source/evaluate_mcln_portable_pair.py:119), [PV GT:296](C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/paired_span_loop.py:296).

Original dataset archives and raw point masks were not reopened. Provenance is supported by retained loading/evaluation code and records, not a new dataset hash verification. Native text parsing maps are present; they must not be described as absent. Spatial GT is used for evaluation and offline grouping.

### B. Score normalization and denominators — PASS

Native semantic softmax/text-map terms rank all 256 queries. The selected query is not selected by GT IoU. Both the native evaluator and analyzer use strict greater-than thresholds. Whole-population accuracy uses 9,508 rows. Repairs come from initial misses; breaks from initial hits. Every net equals repairs minus breaks and after-hits minus before-hits. No accuracy normalization uses prediction statistics.

Evidence: [MCLN score:79](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/source/evaluate_mcln_portable_pair.py:79), [native threshold:1747](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/source/src/grounding_evaluator.py:1747), [comparison:31](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/analyze_zero_update_transfer_20261012.py:31), [PV native score:4](C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/source_review/snapshots/033_native_root_bbs.py:4).

MCLN uses clamped sizes and boundary-difference volumes; PV's saved helper uses size-product volumes. The independent checker respects both formulas. No selected IoU in the four series equals exactly 0.25 or 0.5.

Evidence: [MCLN IoU:712](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/source/models/losses.py:712), [PV IoU:252](C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/run_span_pair.py:252).

### C. Result existence and fidelity — PASS for the final diagnostic

Both raw files, both results, original-source snapshot and final source exist and are hashed by exact resolved absolute path. Raw hashes match the declared result hashes; the PV raw hash also matches its formal receipt. Every aggregate, transition, quartile, support group, geometry median, error count and cross-model support count reproduces. The saved MCLN replay result records completion and zero optimizer updates. This task is a completed reduction of cached records; it is not a training-completion claim.

Evidence: [final serialization:130](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/analyze_zero_update_transfer_20261012.py:130), [completed replay RESULT:2](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/full_first_boundary_observation/replay/RESULT.json:2), [PV receipt](C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/formal/receipt.json), and INDEPENDENT_CPU_EVIDENCE.json.

The prior replay audit's post-clamp “raw size” label and missing earlier CPU source-review version remain historical WARN findings. No raw-preclamp claim or certification of that missing review version is made here. This PASS does not replace the earlier audit verdict.

Evidence: [prior finding 1:13](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/actual_integrity_review/EXPERIMENT_AUDIT.md:13), [prior finding 2:25](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/actual_integrity_review/EXPERIMENT_AUDIT.md:25).

### D. Dead code — PASS

The analyzer's comparison and faces functions are called by the emitted tables. The replay calls row_evidence and the native evaluator, checks counters, and serializes results. The independent checker imports no experiment module and executes no neural code.

Evidence: [functions:31](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/analyze_zero_update_transfer_20261012.py:31), [table calls:63](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/analyze_zero_update_transfer_20261012.py:63), [geometry calls:107](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/analyze_zero_update_transfer_20261012.py:107), [replay calls:130](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/source/evaluate_mcln_portable_pair.py:130).

### E. Scope — PASS for the qualified claim

Scope is 9,508 expressions, 141 scenes, 2,068 intended scene/target objects, one validation benchmark and recorded seed 2027. There are 9,506 unique scene/target/text tuples; MCLN rows do not retain annotation IDs. All MCLN saved query indices and GT boxes agree across the three arms. Same batch, restored RNG, unchanged semantics and model states are recorded runtime-assertion evidence; logits, before/after state tensors and remote checkpoint bytes were not independently rerun or rehashed.

Evidence: [paired replay:112](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/source/evaluate_mcln_portable_pair.py:112), [state assertion:141](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/source/evaluate_mcln_portable_pair.py:141).

GT-volume quartiles are expression-weighted. Cutoff ties belong to the lower group. Support groups use the selected full-transfer masks, so they are outcome-conditioned descriptions. Good mask overlap, candidate coverage and box IoU do not establish the predicted physical instance's identity or the cause of an error.

The saved checkpoint identities identify different MCLN/PV cores, despite matching portable support/span hashes. Cross-model comparisons cannot isolate a module effect. The cross-model marginal differences reproduce, but intersections retain F1's missing-expression qualification.

Evidence: [MCLN spec](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/TRANSFER_SPEC.json:2), [PV restored identity](C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/extremal_support/formal_restore.json:7), [final interpretation:363](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/zero_update_diagnosis_qualified_20261012/RESULT.json:363).

Zero-update geometry evidence cannot explain/predict formal epoch-1 training, prove A and B independently effective, or claim G transfer. The final result explicitly retains formal_epoch1_explained=false. This review made **zero neural forwards, optimizer steps, GPU calls, SSH calls, active-job queries and new accuracy experiments**.

Evidence: [analyzer scope:57](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/analyze_zero_update_transfer_20261012.py:57), [epoch-1 qualification:362](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/zero_update_diagnosis_qualified_20261012/RESULT.json:362).

### F. Evaluation type — PASS: real_gt

Selected-box outcomes use dataset GT. Saved mask overlap and Full256 max-IoU are GT-conditioned support/coverage diagnostics, not synthetic references, physical-instance labels or deployed oracle selection. Their scalar values were re-aggregated; raw masks and all 256 MCLN candidate boxes were not available for geometric recomputation.

Evidence: [row evidence:94](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/source/evaluate_mcln_portable_pair.py:94), [candidate counts:116](C:/Users/gb/.codex/tmp/mcln_portable_best_transfer_20261012/analyze_zero_update_transfer_20261012.py:116).

## Reproduced results

All accuracies below use 9,508 records and strict thresholds.

| Arm | Hits >0.25 | Accuracy % | Hits >0.5 | Accuracy % |
|---|---:|---:|---:|---:|
| MCLN native | 5515 | 58.003786 | 4409 | 46.371477 |
| MCLN reference | 5459 | 57.414809 | 4767 | 50.136727 |
| MCLN full | 5473 | 57.562053 | 4777 | 50.241902 |
| PV extremal support | 5677 | 59.707615 | 4920 | 51.745898 |

| Transition | Repairs / breaks / net >0.25 | Repairs / breaks / net >0.5 |
|---|---:|---:|
| native → reference | 138 / 194 / −56 | 694 / 336 / +358 |
| native → full | 138 / 180 / −42 | 695 / 327 / +368 |
| reference → full | 18 / 4 / +14 | 23 / 13 / +10 |

At >0.5, native-based repairs use 5,099 initial misses and breaks use 4,409 hits; reference → full uses 4,741 misses and 4,767 hits. Full versus native changes accuracy by +3.870425 percentage points at >0.5 and −0.441733 at >0.25. Full versus reference adds +0.105175 and +0.147244 points respectively.

| GT-volume group, smallest first | Rows | Native → reference net >0.5 | Native → full net >0.5 | Reference → full net >0.5 |
|---|---:|---:|---:|---:|
| 1 | 2377 | +199 | +201 | +2 |
| 2 | 2380 | +125 | +131 | +6 |
| 3 | 2374 | +26 | +26 | 0 |
| 4 | 2377 | +8 | +10 | +2 |

Cutoffs are 0.09070646915287309 / 0.2477334084009524 / 0.5180449836317556 m³. Of 4,731 full-transfer misses at >0.5, 3,388 have a different saved candidate above threshold and 1,343 do not; 488 have both selected mask IoUs >0.5. These are geometric/support descriptions, not causal or physical-instance diagnoses.

Qualified aligned-row cross-model counts at >0.5 are: both hit 3,724; MCLN-only hit 1,053; PV-only hit 1,196; both miss 3,535. Net +143 PV hits matches the marginal difference. At >0.25 the counts are 4,499 / 974 / 1,178 / 2,857, net +204 PV hits. Only marginal differences are independent of the unverified expression pairing.

## Action and acceptance boundary

No blocking correction remains. Preserve the expression, physical-instance/proxy, different-checkpoint and zero-update/E1 qualifications whenever using these tables. No new engineering work or model run is required.

The semantic review remains same-family/provisional. Deterministic acceptance applies only to completed local calculations. Model/effort identity is UNATTESTED; policy preferences are not runtime attestation.

EXPERIMENT_AUDIT.json, INDEPENDENT_CPU_EVIDENCE.json and trace/ contain the resolved hashes, original finding, verifier source and request/response trace. Only this review directory was written by the reviewer.

