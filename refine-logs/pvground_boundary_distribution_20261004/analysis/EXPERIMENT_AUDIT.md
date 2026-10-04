# Experiment Audit Report

Date: 2026-10-04. Auditor: fresh native Codex reviewer, same family; **provisional**. The installed route requests `gpt-6-astra` / `max`; provider backend SKU is not independently attested.

**Overall verdict: WARN. Integrity status: warn. Blocking integrity issues: none.**

The saved evidence supports the qualified two-arm result. Independent CPU reconstruction and recount confirm formal `last/bbs` residual **5589/4446** versus distribution **5616/4506** at strict IoU > .25 / > .50 over 9508 development rows. The distribution-minus-residual strict difference is **+60 hits, +0.63104754 percentage points**. The full target remains unmet. WARN records the observed numerical differences, limited experimental scope and evidence limits below; it does not indicate a fabricated result.

## Scope and evidence identities

Execution scope: **LOCAL_CPU_RECORD_RECOUNT_AND_STATIC_SOURCE_REVIEW_WITH_SAVED_RUNTIME_RECEIPTS**. I read the 89 requested artifacts and 11 supplemental primary artifacts, parsed all 46,564 evaluation rows and 7,446 training rows, and AST-parsed 39 supplied Python files. The 65 terminal intake hashes and 34 module/native-import declarations match. All 100 input hashes were checked again before reporting. No experiment source or primary result was changed.

Paths below use these exact roots:

- `R` = `D:/Program Files/UserCache/gb/codex/tmp/pvground_boundary_distribution_20261004`
- `N` = `D:/Program Files/UserCache/gb/codex/tmp/pvground_fused_support_20261002/complete_tail_fused_retry/source/imported`
- `P` = `C:/Users/gb/.codex/tmp/pvground_boundary_distribution_20261004/complete_preflight`
- `G` = `C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_scanrefer_formal_20260918_semantic_assignment_v1`

`R/analysis/AUDIT_READ_MANIFEST.json` records every actual path, byte size, line count and SHA256. `R/analysis/AUDIT_CPU_RECOUNT.json` contains the deterministic results. Hash agreement identifies supplied files; it is not remote attestation.

## A. Ground truth provenance: PASS

The loader reads ScanRefer scene/object annotations (`N/src.joint_det_dataset.py:585-648`). Target masks use annotated ScanNet object-member indices and target boxes use the corresponding object bounds (`:1086-1119`), exported as `center_label`, `size_gts`, `gt_masks` (`:1387-1393`). Evaluation reads those labels (`R/run_boundary_fit.py:518,544`). GT is not generated from model predictions.

The model input is point/voxel data, language, detector proposals and superpoints (`R/run_boundary_fit.py:285-295`). Both perfect-detector branches are disabled (`:254-257`); `filter_non_gt_boxes=False` is explicit (`:498`). Native token maps are text-parse outputs (`N/src.joint_det_dataset.py:981-1082,1874-1888`). GT volumes and oracle hits are computed after inference, not used to choose a Query. Predicted Mask probabilities are whole/local support features, not reference labels. Classification: **real_gt**, with raw dataset-file verification outside this local audit.

## B. Denominators and normalization: PASS

Box IoU is intersection divided by union; decisions are strict `>` (.25 and .50), not `>=` (`R/run_boundary_fit.py:525-550,565-571`; `N/evaluator.py:289-303`). Mask IoU uses logical intersection/union; mIoU divides its sum by the complete row count (`N/evaluator.py:897-902`; runner `:543-545,568-571`). No result metric is divided by the model's own maximum or average output.

Predicted-mask member weights, scene spans and softmax probabilities are internal feature normalization (`R/complete/distribution/whole_mask_range.py:12-67`). They do not normalize reported accuracy. DFL uses a mean over native matched boxes times six faces, followed by a separate coefficient 1/7 (`R/pvground_boundary_box_refiner.py:40-65`; runner `:314-328`). The native seven-head denominator and G semantic replacement coefficient remain separate (`N/models.losses.py:947-955`; `R/complete/distribution/pvground_semantic_assignment.py:40-47`).

## C. Actual artifacts, metrics and terminal closure: PASS

Each arm has 3723 ordered steps, 29,778 distinct fit row IDs exactly once, 3722 batches of 8 and one final batch of 2. The complete fit order matches across arms; fit and initial/terminal holdout row IDs are disjoint and cover all 36,665 training-pool IDs. Numeric losses, boundary values and gradient norms are finite in all saved training records. Evidence: `R/complete/{residual,distribution}/train.jsonl:1-3723`; runner `:616-651`.

Both train/formal exits and controller exit are zero. `R/complete/status.json:2-28` records all four finished stages, ending at 2026-10-04 16:14:46 +08:00. The collector's 16:20:31 snapshot records the controller absent (`R/complete/INTAKE.json`). The earlier plan/launch documents retain their preparation-time state; they are not the terminal status source.

| Phase / mode | Rows per arm | Residual @.25 / .50 | Distribution @.25 / .50 |
|---|---:|---:|---:|
| Initial / bbs | 6887 | 6176 / 5602 | 6176 / 5602 |
| Initial / bbf | 6887 | 6206 / 5647 | 6206 / 5647 |
| Terminal / bbs | 6887 | 6168 / 5535 | 6174 / 5616 |
| Terminal / bbf | 6887 | 6200 / 5592 | 6203 / 5654 |
| Formal / bbs | 9508 | 5589 / 4446 | 5616 / 4506 |
| Formal / bbf | 9508 | 5624 / 4469 | 5653 / 4526 |

All six `rows.jsonl` files and receipts, plus `R/analysis/SUMMARY.json` and `REPORT.md:8-13`, agree. Independent scalar CPU IoU reconstruction covers 93,128 selected box/mode records and their coarse boxes: **zero .25/.50 decision changes**. Maximum final IoU arithmetic difference is 4.25526282e-6. Distribution selected-face decoding also agrees with saved boxes to at most 4.42258312e-7 in formal evaluation.

Formal Mask @.25/.50 hits are 5812/5133 in bbs and 5847/5156 in bbf for both arms. bbs mIoU is 47.1076284839% residual versus 47.1076712955% distribution; bbf is 47.3676210549% for both. These are **saved native scalar recounts**, not reconstruction of raw binary masks. Full256/oracle coverage and rank buckets likewise recount saved flags; full candidate boxes/scores are not available here for fresh replay.

## D. Actual called paths and dead code: WARN

The active refiner runs after native Mask generation and replaces `last_center`/`last_pred_size` (`N/models.pv_ground.py:551-564`). Native final-box L1/GIoU consumes this deployed frame (`N/models.losses.py:518-544,898-917`). Evaluation calls all four native Box/Mask branches (`N/evaluator.py:194-206`; runner `:510-513`) and compares saved Box hits/Mask sums with native totals (`:565-571`). No reported metric is phantom or unused.

The scorer has one selected candidate per mode. `bbs` is fixed as primary; `bbf` is separately reported, not mixed into the primary choice. Within each mode the selected Query indexes both the box and Mask (`R/run_boundary_fit.py:536-557`; native evaluator `:281-303,641-653,746-758`). No GT-based inference filter or dual ranking is active.

One existing metric helper, `N/models.losses.py:103` (`calculate_diou_3d`), has no active caller in the supplied sources; its integration at `:546-557` is commented out. No current reported metric uses it. This is an informational dead-code warning, with no requested unrelated refactor.

## E. Scope and claim ceiling: WARN

This is one seed (2027), two configurations, one dataset. Logs report 456 physical fit scenes; saved holdout rows contain 106 scenes, and formal rows contain 141. The 6887 holdout rows come from the upstream author-training pool; the 9508 formal rows are development validation. The result does not establish multi-seed significance, independent-test performance or new-scene pretraining generalization. Evidence: `R/complete/{residual,distribution}/train.log:8`; specs `:9-15`; runner `:201-267`; `REPORT.md:50`.

Residual400614 versus distribution456102 changes output parameter count, representation and explicit DFL supervision together. It supports a package comparison, not isolated DFL causality. This adapter is not a completed direction-conditioned token decoder. Boundary entropy is uncalibrated. No new Nr3D/Sr3D, teacher, quality callback or three-contribution completion is evidenced (`R/EXPERIMENT_PLAN.md:13-34,69-78`; `REPORT.md:47-57`).

The primary development target is 5615/4754. Distribution reaches 5616/4506, **248 strict hits short**. The broader goal is not achieved (`R/complete/status.json:40,76`).

## F. Evaluation classification: PASS — real_gt

Initial, terminal and formal Box/Mask metrics use dataset GT. Same-Query repair/damage counts, Full256/top-K oracle coverage and GT-volume quartiles are offline GT diagnostics. They are not deployable candidate selection, calibrated quality or model-generated ground truth. The four formal GT-volume quartiles each contain 2377 rows; strict distribution-minus-residual counts are +24,+30,+4,+2. This does not establish a causal size effect (`R/analyze_terminal.py:93-119`; `REPORT.md:32-39`).

## Requested protocol checks

**Frozen state and ten tensors.** The protected official PV parent is loaded, then original G's1072-tensor delta; only the ten candidate-head tensors are trainable (`R/run_boundary_fit.py:137-199`). Model eval state is preserved while the head trains (`:595`); all core persistent state is compared with its initial value before endpoint save (`:629`). Checkpoints contain only the selected head delta plus optimizer/RNG state (`:599-614`). Both completed receipts record all ten changed tensors and unchanged G state. These are source/runtime-receipt checks, not fresh tensor replay.

The supplemental real preflight receipts record two batch8 updates per arm, ten positive second-step head gradients, exact AdamW moments/steps/groups restoration, unchanged G state and zero same-cached-input/common-floor deltas. I independently checked all four saved call-order witnesses per arm. Native box-loss output gradients are residual7.58567/9.84650 and distribution0.196345/0.217636; distribution DFL-alone output gradients are0.448557/0.495409. Evidence: `P/{residual,distribution}/preflight.json:540-675`; runner `:332-353,443-488`. No GPU probe was rerun.

**Matching, directions and support.** Native prefixes are proposal,last,0head…4head (`N/models.losses.py:852-853`), with one active matcher call each (`:817,828-829`). The runner captures seven and uses `matching[1]` for distribution targets (`R/run_boundary_fit.py:314-326`), with exactly the same GT-mask filtering and target indices as native loss. This is each forward's actual native assignment, not an assertion that assignments are identical between arms. The saved distribution run matches29,778 boxes/178,668 faces: one GT match per example in this run; broader multi-GT cases were not empirically exercised.

Negative-face offset increases outward low-face displacement, and positive-face offset increases outward high-face displacement; the decode/target formulas are inverse before final-size flooring (`R/pvground_boundary_box_refiner.py:25-37`). All33 knots are symmetric and increasing. Independent scalar checks cover knots, midpoints and both saturated endpoints (67 cases), plus three target/decode orientation roundtrips. Raw target finiteness is asserted before clipping; 387 faces are outside[-4,4] over222 steps (e.g. distribution `train.jsonl:9`). These are counted clipped training targets, not hidden invalid values.

Both arms share the same1302-dimensional aggregate input (288 query +896 local +3 size +6 coarse/global coordinates +109 whole-mask summary) and112 local member slots, with all256 candidates retained (`R/pvground_boundary_box_refiner.py:77-95`; support modules). Saved evaluation point/root/row identities and training row order match. Cross-arm training tensors and all aggregate feature values were not archived; identical construction/protocol is supported, bitwise feature equality is not.

**Actual floor statistics.** Both arms use1e-6 for reference and final size; the count predicate is raw_size<=floor (`R/pvground_boundary_box_refiner.py:25-30,100-110`). Preflight observes1281 native coarse-size elements below the floor per batch. Training axis-count sums are372,721 residual and1,136,612 distribution. These are repeated candidate-axis exposures, including equality to floor, not distinct objects or exclusively crossed-face incidents. Formal selected bbs flooring is4 rows/6 axes residual versus6 rows/10 axes distribution; bbf is1/1 versus12/22. Zero-head equality therefore concerns the common floor protocol, not unmodified negative native sizes.

**Observed numerical limits.** The saved initial comparison is exactly reproduced. At initial line4920/row26603, bbs Query changes68→225 and bbf176→225. Maximum selected coordinate differences are0.1902718544m/bbs and0.0741221905m/bbf; most other initial differences are continuous numerical differences. Line3803/row18482 has a0.02047777176 Mask-IoU difference. Initial Box threshold decisions are all unchanged. In formal evaluation, all9508 selected Query IDs and coarse boxes/IoUs match between arms, while bbs line6375/row6374 has Mask IoU0.6595962048→0.6636667252. Threshold counts are unchanged. No learned Mask improvement or bitwise paired-start claim follows.

**Separate effects.** Formal between-arm strict repairs/damages are126/66 (+60). Same-Query coarse/final residual is4495→4446 (80 repairs/129 damages); distribution is4495→4506 (20/9). Formal Full256 strict coverage is7884→7962 residual and7884→7890 distribution, but it is only an offline oracle. Median maximum face movement is16.642533mm versus5.602553mm. These are distinct quantities, not interchangeable demonstrations of historical-G improvement.

Historical original-G5615/4495 is supported by the supplemental publication receipts/source, whose hashes connect terminal SHA0575dfae… to official parent6f24c67c… (`G/receipt.json`, `G/fit_terminal/receipt.json`, `G/evaluate.py:252-346`). Its saved-row hash is2d1edf5b…; those raw historical rows were not supplied/read. The +1/+11 historical comparison is receipt-linked, not a fresh replay or row-level comparison with that old run.

**Restore and retention.** Formal mode strictly restores the3723-step delta, head keys, spec and parent identity before evaluation (`R/run_boundary_fit.py:220-237`). The controller waits for9508-row formal success and CPU threshold agreement before retaining/deleting (`R/controller.py:91-139`). Residual's4,921,221-byte endpoint SHA468fded6… is recorded deleted after formal verification; distribution's5,587,141-byte endpoint SHA79e35068… is the only retained experiment weight. `complete/INTAKE.json:408-426` confirms this collected state and zero downloads. No `.pth` exists under the local complete packet.

Deletion is restricted to this experiment's residual/distribution terminal.pth paths; original G is rehashed and explicitly excluded. Official PV and V99 paths cannot enter that deletion branch. This supports protection by bounded code and collected G identity; it is not a fresh V99 checkpoint hash verification. Active recovery is replaced in place, and verified nonbest endpoints are retired; transient write `.tmp` files are distinct from steady-state retained weights. No failed local weight archive was created by the collector.

## Actions and limitations

No blocking correction or rerun is required to report the qualified result. Preserve the scope, numerical, floor/saturation and package-attribution qualifiers. Do not claim the full target, additional benchmarks, calibrated entropy, isolated DFL causality or a completed directional decoder.

The audit imported no model/project code and used no Torch/CUDA, SSH, weight reads, job queries/restarts or implementation changes. It did not inspect credentials/auth-wrapper files or raw goal files. The explicitly listed collector source was read only; no connection or environment-secret lookup was executed. Raw dataset contents, raw Masks, full candidate arrays, historical G rows and live remote state remain outside verification. Saved preflight/training receipts are evidence of recorded runtime assertions, not independently repeated execution.

Review independence is **same-family** and acceptance status **provisional**. The substantive WARN must not be promoted to cross-family acceptance or independently verified provider-backend identity.
