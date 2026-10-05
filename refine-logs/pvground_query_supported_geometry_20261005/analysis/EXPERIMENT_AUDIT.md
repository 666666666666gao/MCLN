# Experiment audit: closed Query-supported geometry continuation

**Verdict: PASS. Blocking findings: none.** This applies to the bounded closed result; the research target remains unmet.

Date: 2026-10-05T14:18:26.564492+08:00. Fresh delegated same-family review; acceptance is **provisional**. Requested routing: `gpt-6-astra`, effort `max`. Actual backend, model and effort are **not attested**. Prior source/launch PASS records are attribution records, not terminal result evidence.

Root: `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005`. Only `analysis/EXPERIMENT_AUDIT.json` and this report were written.

## Verified result

Native last/bbs top1; strict IoU thresholds; denominator 9508 development expressions.

| System | Hits@.25 | Hits@.50 | Acc@.25 (%) | Acc@.50 (%) |
|---|---:|---:|---:|---:|
| protected_geometry_parent | 5616 | 4506 | 59.06604964 | 47.39167017 |
| control | 5616 | 4499 | 59.06604964 | 47.31804796 |
| query_supported | 5614 | 4509 | 59.04501472 | 47.42322255 |

Strategy versus control: 22 repairs / 12 damages / net +10 at .50 (+0.10517459 percentage points). Versus parent: 16 / 13 / +3 (+0.03155238 points). The strategy's same-Query coarse-to-final refinement is 33 / 19 / +14.

The actual metric-best candidate is `query_supported`, 5614/4509. The same-model target >=5615 AND >=4754 is **false** for all three systems; this candidate is short by 1 and 245 hits. Parent and strategy threshold results must not be combined. Checkpoint receipt identity: `0b37986d4448b125405d573272282ef37396a781ba970789dfc9b080445cec51`; this audit did not load or retain weights.

## Deterministic validation

- All 73 INTAKE entries match current byte lengths and SHA256; all 17 sealed helper hashes match specs. Active model/dataset/loss/preprocessing local source hashes match recorded imports/manifests.
- All 7446 training records checked: each arm 3723 updates, 3722 full batches of 8 plus one batch of 2, 29778 unique fit rows exactly once, exact cross-arm order and fit partition. Specs differ only in root and extra-loss weight 0/1.
- All 56072 evaluation-row records checked (six new stages and parent formal). NumPy float64 recomputed 112144 selected/coarse Box/GT IoU pairs: zero threshold flips at .25/.50; maximum new-stage CPU/GPU absolute difference 3.137222141180729e-6.
- All SUMMARY stages, training aggregates, paired counts, four 2377-row volume groups, formal CSV values and rounded RESULTS claims agree. Masks/full256 oracle coverage were checked as saved scalar/indicator recounts.
- Native controller and all four child exits are 0; terminal status/observer/INTAKE agree. Both formal restore receipts match train checkpoint hashes and exact optimizer/model restore assertions.

Checks used cached `uv run --offline --with numpy python -B -X utf8 -`; actual inline checks exited 0. No project training or analyzer module was imported/executed. The JSON contains exact computed results and all current input hashes.

## Checks A-F

### A. Ground truth provenance: PASS

- The active dataset source is the local joint_det_dataset.py whose current SHA256 equals both arms' imports.json and appearance_source_manifest. It reads ScanRefer scene/object annotations and dataset ScanNet objects. Root GT masks come from annotated instance point indices; GT boxes come from those dataset instance points, not any current or parent prediction.
- The visual_data_handlers.py copy under pv_ground_source_20260905 matches the dataset manifest's 6d1d4f... hash. The runtime bundle's different visual handler is not substituted for the active dataset implementation.
- The neural forward receives points/voxels/text/superpoints and detected GroupFree boxes; butd_gt and butd_cls are false. Native bbs uses the existing parsed text maps, without a GT Box/Mask eligibility gate at inference.
- Extra-candidate supervision reads root GT only in the training step. Eligibility requires own Query Mask IoU > .5 AND fused Mask IoU > .5 AND final Box IoU <= .5, then excludes every native final-layer matched Query. The final-layer matching index is matches[1] because native prefix order is proposal,last,0head,...,4head. Public Text Mask alone cannot qualify.
- This is a training support/overlap proxy for extra responsibility, not physical object identity proof. The reported Box/Mask evaluations themselves use real dataset GT.

Evidence:

- `../pvground_g_p2_20261002/complete/source/joint_det_dataset.py:203-209,585-649,1085-1119,1387-1392` — Dataset loading, annotated target IDs, instance membership masks and root Box labels.
- `../pv_ground_source_20260905/src/visual_data_handlers.py:129-163,195-197,225-259` — Dataset aggregation/segmentation membership and instance bounding boxes.
- `complete/control/imports.json:7,14` — Active dataset source and actual imported hash; query_supported matches.
- `run_geometry_fit.py:45-74,152-203,242-255,299-311` — Pinned inputs and data split; forward inputs exclude target geometry/masks; GT used in scoring.
- `query_supported_geometry.py:27-58` — Both Mask requirements, Box-poor condition, all-match exclusion and root supervision.
- `run_geometry_fit.py:360-369,417-419` — Final-layer matching and training-only extra loss; formal calls evaluate only.
- `../pvground_runtime_bundle_20260908_v1/PV-Ground/models/losses.py:849-886,891-917` — Native matcher prefix order and all actual GT assignments.

Evidence limit: Dataset loading and source identity were traced directly; no remote dataset pickle, full raw annotation corpus, or raw Mask tensor replay was performed.

### B. Score and loss denominators / normalization: PASS

- Reported Box Acc is 100 * strict-threshold hits / actual expression count (9508 formal or 6887 holdout). Standard IoU is intersection / union. Mask mIoU is the arithmetic mean of saved per-expression IoUs times 100. No performance score is normalized by a model's own maximum/minimum or mean prediction statistic.
- The native total remains native + G correction + matched boundary loss / 7; only the strategy adds its extra block with weight 1 (control weight 0). Existing G and native supervision/matching denominators are unchanged.
- For each expression with q extra candidates, existing native center L1 + half-weight size L1 and GIoU are divided by q; distribution CE averages its 6*q faces. (10*L1 + 2*GIoU + distribution)/7 is then averaged over actual batch size, including empty expressions. The final batch has B=2. This is loss weighting, not inflated evaluation normalization.
- The six-face range, 33 existing knots, REG_SCALE=4 and clipping remain unchanged. Out-of-range training faces are recorded, not silently removed. Strategy: 558553 eligible-candidate roles, 5793 empty expressions, 200912 outside faces; control diagnostics: 558599, 5795, 200917.
- Feature softmaxes and whole-Mask weighted scene statistics are model inputs, not reported performance denominators.

Evidence:

- `geometry_result_metrics.py:5-10,23-45,73-77` — Direct Box IoU, actual row denominator and Mask scalar mean.
- `run_geometry_fit.py:329-341,366-369` — Native receipt agreement, counts and separate extra block.
- `query_supported_geometry.py:24-25,49-64` — Empty expressions, per-expression candidate mean and actual batch denominator.
- `../pvground_runtime_bundle_20260908_v1/PV-Ground/models/losses.py:518-544,831-844,943-956` — Native regression coefficients and denominator.
- `../pvground_final_quality_20261005/runtime_bundle/pvground_boundary_box_refiner.py:9-15,33-65` — Fixed knots/target formulas, six-face mean and clipping counter.
- `../pvground_final_quality_20261005/runtime_bundle/pvground_semantic_assignment.py:34-50` — Existing G correction denominator.
- `complete/control/train.jsonl:1-3723` — All loss components finite and arithmetic consistent within float rounding.
- `complete/query_supported/train.jsonl:1-3723` — All extra counts/face counts/batch sizes checked.
- `analysis/SUMMARY.json:662-737` — Training aggregate counts and means reproduced.

### C. Result existence, exact counts and claims: PASS

- All 73 files declared in complete/INTAKE.json exist and match current byte lengths and SHA256. All 17 sealed helper files match both arm specs. The root plan/specs/controller/runner/qualification code/launch receipt/source review match collected copies. Both train-log hashes and all seven evaluation-row hashes (six new stages plus parent formal) match their receipts.
- All 7446 train records and all 56072 evaluation-row records were parsed. A separate local NumPy float64 recount recomputed selected final and coarse Box/GT IoUs (112144 pairs): zero changes at .25 or .50. Maximum absolute CPU-vs-saved-GPU discrepancy across the new stages is 3.137222141180729e-6.
- Every per-stage summary, saved Mask scalar recount, monotonic top16/32/64/256 oracle count, Box/Mask quadrant, displacement statistic, paired repair/damage count, training aggregate and GT-volume group matches SUMMARY; formal CSV and rounded RESULTS table match.
- Formal native bbs parent/control/strategy are 5616/4506, 5616/4499 and 5614/4509. Strategy vs control: .50 repairs22/damages12/net+10; vs parent: repairs16/damages13/net+3. Same selected Query coarse-to-final refinement is repairs33/damages19/net+14.
- The primary metric-best candidate is the actual query_supported checkpoint with 5614/4509. The same-model >=5615 AND >=4754 target is false for every model; strategy is short by 1 and 245 hits. No cross-model or bbf/bbs metric combination is accepted.
- Historical draft/launch status surfaces are identified separately; actual closed status is established by terminal artifacts, not the old tracker or review PASS.

Evidence:

- `complete/INTAKE.json:3-295` — All 73 declared file byte/hash checks.
- `complete/control/formal/rows.jsonl:1-9508` — Every control formal row.
- `complete/query_supported/formal/rows.jsonl:1-9508` — Every strategy formal row.
- `../pvground_boundary_distribution_20261004/complete/distribution/formal/rows.jsonl:1-9508` — Parent comparison with exact row/scan/target/GT/point alignment.
- `complete/control/formal/receipt.json:3-19` — Control 5616/4499, denominator9508.
- `complete/query_supported/formal/receipt.json:3-19` — Strategy 5614/4509, denominator9508.
- `../pvground_boundary_distribution_20261004/complete/distribution/formal/receipt.json:13-25` — Parent bbs5616/4506; bbf is a different mode.
- `analysis/RESULTS.md:7-19` — All quantitative headline claims and their stated limits.
- `analysis/FORMAL_METRICS.csv:1-4` — Exact formal output table.
- `analysis/SUMMARY.json:538-660,811-830` — Paired counts, volume groups, best model and target false.
- `analyze_closed_formal.py:54-101` — Claim generation and same-model target decision.

### D. Actual metric calls, process closure, budget, restore and frozen-state evidence: PASS

- Controller starts the actual run_geometry_fit.py train then formal for each arm, waits for each child, writes native exits, and fails on nonzero code. Four child exits and fit_controller.exit are 0; collected status lists all four phases complete at 2026-10-05T13:53:50.086084+08:00, elapsed16290.719s. fit_wait records the controller no longer alive and matches INTAKE/status.
- Evaluation actually calls observed_readback_forward -> native_root_bbs -> native GroundingEvaluator.evaluate. Row hit totals are asserted equal to native bbs counters and Mask sum to native mask_pos. Completed formal.log entries agree with receipts. Offline geometry_result_metrics.paired/summarize are actually called by the closed analyzer. fit_body.py is generation source embedded exactly in the executed runner, not a separate executed program.
- The same initial three checkpoint identities, seed2027, architecture and specs are used; the only spec differences are output root and extra_geometry_weight. Both use fresh AdamW lr1e-5/WD.0005/clip.1. Every arm has step1..3723, 3722*8+2=29778 unique rows exactly matching fit partition, with identical order. Each head thus has3723 continuation updates,7446 including its3723 parent updates. Preflight's2 updates per arm were in a separate process/in-memory serialization and not reused for the fresh fits.
- The runner freezes all parameters then unfreezes only candidate_box_refiner, explicitly asserts10 trainable tensors/456102 parameters, puts parents/R in eval, checks no other parameter gradient, compares all non-head state tensors against start, and records complete receipts. Both formal processes restore the exact head delta and optimizer, require all optimizer steps3723 and verify tensor/moment/group equality; restored file hashes match train receipts.
- The source retains all256 candidates and one deployed final semantic subhead after refiner/R. Hooks assert the call order and exact same-forward geometry/Mask invariance. Every saved new evaluation row records one native head call and zero diagnostic replay calls. Both training logs also record one call. Box and fused Mask selection share the same bbs Query.
- The two actual GPU preflight receipts record gradient isolation to qualified outputs, nonzero isolated head gradient, empty expressions and clipped endpoints, cached upstream bbs/Mask invariance after a head update, and exact optimizer serialization. These are runtime receipts from the checked code, not a fresh GPU replay.
- Unused bundled quality diagnostics and unused general evaluator display helpers are not treated as experimental results. No reported metric was found to originate from an uncalled path.

Evidence:

- `controller.py:36-60` — Child command, wait, exit and completion sequence.
- `fit_launch.json:2-5,15-29` — Original process584730 and declared budget.
- `fit_wait.json:2-37` — Closed observer/controller and exit0.
- `complete/fit_status.json:2-23` — Four complete phases and finished timestamp.
- `complete/fit_controller.exit:1` — Native controller exit0.
- `complete/control/train.exit:1` — Train exit0.
- `complete/control/formal.exit:1` — Formal exit0.
- `complete/query_supported/train.exit:1` — Train exit0.
- `complete/query_supported/formal.exit:1` — Formal exit0.
- `run_geometry_fit.py:117-151,221-240,257-344,346-415,453-511` — Model construction, fresh/restore optimizer, called evaluation, training and frozen-state assertions.
- `complete/control/train.jsonl:1,3723` — First and last actual step; all intervening lines checked.
- `complete/query_supported/train.jsonl:1,3723` — Same first/tail IDs and complete budget.
- `complete/control/formal_restore.json:2-9` — Exact model/optimizer restore and terminal hash.
- `complete/query_supported/formal_restore.json:2-9` — Exact model/optimizer restore and terminal hash.
- `complete/control/formal.log:7-25` — Actual progress and closed formal receipt.
- `complete/query_supported/formal.log:7-25` — Actual progress and closed formal receipt.
- `../pvground_geometry_readback_20261004/revision2/source_preview/PV-Ground/models/pv_ground.py:506-575` — Deferred native head, Mask generation, all-candidate refiner then semantic readback.
- `../pvground_geometry_readback_20261004/revision2/source_preview/PV-Ground/models/modules.py:135-179` — Deferred original semantic call prevents double scoring.
- `../pvground_final_quality_20261005/runtime_bundle/readback_preflight_checks.py:19-68` — Observed order/one call and same-forward invariance.
- `../pvground_final_quality_20261005/runtime_bundle/readback_model_factory.py:15-75` — Pinned parent reconstruction and256 Queries.
- `complete/control/preflight.json:3-21,33-53,89-109` — Actual two-step invariance/gradient/serialization witness.
- `complete/query_supported/preflight.json:3-21,33-53,89-109` — Actual two-step invariance/gradient/serialization witness.
- `analyze_closed_formal.py:9,54-76` — Called offline metric functions.
- `fit_body.py:1-166` — Exact embedded training fragment.

Evidence limit: Frozen-state and checkpoint restore evidence is source-linked runtime assertions/receipts, not a new local tensor comparison or proof of current remote weight retention. Training augmented tensors and full score vectors were not saved for full replay.

### E. Scope and claim ceiling: PASS

- One seed2027, two full fresh same-start continuation fits;29778 fit expressions per arm. Each initial/terminal stage has6887 expressions across106 scan IDs and is a pretrained-seen training-split module holdout. Each formal stage has9508 expressions across141 scan IDs and is development validation, not an unseen final test.
- Identity/GT/point hashes align across paired stages and parent comparisons. Initial hit sets and selected Queries are equivalent, but bitwise whole-forward equivalence is not established and is empirically false for some stored values (O1). This limits mechanistic or causal interpretation of a very small single-seed gain.
- Training eligibility and evaluation volume quartiles use GT offline only. Oracle candidate coverage is an upper-bound diagnostic, not deployed selection accuracy or physical identity recognition.
- The current RESULTS/SUMMARY explicitly state single seed, seen6887/development9508, scalar Mask/full256 limits, no physical-identity proof, no Nr3D/Sr3D new result and no complete three-module novelty proof. These qualifiers are consistent with the audited evidence.
- This PASS establishes integrity of the bounded closed report; it does not establish significance, robust generalization, the experiment's target achievement, or completion of a broader research goal.

Evidence:

- `EXPERIMENT_PLAN.md:5-15` — Bounded head continuation and target.
- `run_geometry_fit.py:39-53,155-203` — Seed, fit/holdout split, formal validation.
- `analysis/RESULTS.md:17-19` — Existing scope qualifiers.
- `analysis/SUMMARY.json:818-830` — All target booleans false and explicit scope.
- `analyze_closed_formal.py:77-85,89-101` — Offline GT grouping and actual-model target predicate.
- `complete/control/initial/rows.jsonl:1-6887` — 106 scan IDs; paired identity equivalence.
- `complete/control/formal/rows.jsonl:1-9508` — 141 scan IDs; aligned development expressions.

### F. Evaluation-type classification: PASS

- No synthetic-proxy, self-supervised-proxy, simulation-only or human-evaluation result is relabeled as dataset accuracy.

Evidence:

- `run_geometry_fit.py:299-341` — Prediction compared against dataset Box/Mask.
- `query_supported_geometry.py:18-21,27-64` — Training support proxy.
- `complete/control/preflight.json:2,14` — Engineering-only two-step test.
- `complete/query_supported/preflight.json:2,14` — Engineering-only two-step test.
- `analysis/RESULTS.md:19` — Classification and scope stated.

| Evaluation | Classification | Limit |
|---|---|---|
| initial6887 and terminal6887 selected Box/Mask | `real_gt` | Pretrained-seen module holdout; CPU Box recomputation, Mask scalar recount |
| formal9508 selected Box/Mask for both arms and parent | `real_gt` | Development validation; CPU Box recomputation, Mask scalar recount |
| all256 candidate availability and GT volume quartiles | `real_gt` | Offline GT oracle/diagnostic; availability verified as saved indicator recount |
| extra geometry qualification | `not_an_accuracy_evaluation` | Training GT-conditioned support/overlap proxy; not model-generated GT and not physical-identity truth |
| two-update preflight | `engineering_check` | Runtime gradient/serialization/invariance test; accuracy_result=false |

## Nonblocking observations

**O1: Same-start state recipe and same rows/order do not imply identical full-forward predictions.**

Initial selected Query IDs and both Box hit sets match exactly, but 6885/6887 selected boxes differ numerically (maximum coordinate difference 0.0008503198623657227 m), maximum selected-IoU difference is 0.0019326508045196533, two mask-IoU scalars differ, and one oracle25 and one oracle50 row differ. Formal control/strategy selected Query IDs differ on rows 958, 2217 and 2283; none of these three rows changes the .25 or .50 hit outcome. Existing RESULTS scope explicitly disclaims cross-CUDA full-forward bitwise equality.

Evidence: `complete/control/initial/rows.jsonl:1-6887`; `complete/query_supported/initial/rows.jsonl:1-6887`; `complete/control/formal/rows.jsonl:959,2218,2284`; `complete/query_supported/formal/rows.jsonl:959,2218,2284`; `analysis/RESULTS.md:19`.

**O2: The plan and tracker are pre-run/launch snapshots; closed status comes from terminal evidence.**

EXPERIMENT_PLAN still says IMPLEMENTATION_DRAFT and EXPERIMENT_TRACKER is explicitly timestamped 09:23 with fits pending. They must not be presented as the current closure record. The actual fit_status, native exit files, stage receipts, and saved rows establish completion at 13:53:50 CST. This is not evidence of an unfinished experiment.

Evidence: `EXPERIMENT_PLAN.md:3`; `EXPERIMENT_TRACKER.md:3,11-12`; `complete/fit_status.json:2-23`.

**O3: controller_body.py does not exist in the requested root.**

The executed controller.py exists, was reviewed directly, and is byte-identical to the collected controller.py. No content or verdict was fabricated for the absent optional file.

Evidence: `controller.py:1-60`.

## Limits and claim impact

- No GPU, SSH, remote process query, browser, credential-wrapper inspection, experiment-source mutation, checkpoint loading, deletion, retention or archive action was performed.
- Remote dataset corpus and checkpoint bytes were not newly replayed; provenance/frozen-state/restore conclusions are limited to matching local source identities and closed runtime artifacts.
- Saved rows contain selected/coarse Boxes and GT Boxes, but do not contain raw predicted/GT Masks, all256 Boxes/scores, or per-training-step candidate IDs. Mask/full256 availability is a scalar/indicator recount; training qualification is source-path verification with logged counts and actual preflight witnesses.
- Same-start identity and exact training row order are supported; full-forward bitwise equality is contradicted by observed numerical drift. Training augmentation tensor equality was not directly saved/recounted.
- Single seed, development validation, pretrained-seen holdout and small net gains do not establish statistical significance, out-of-distribution generalization, Nr3D/Sr3D results, or broad novelty.
- The reports identify a metric-best candidate and target failure; they do not attest that weight-retention actions have occurred.
- Model routing is requested gpt-6-astra/max only; backend actual model and effort are unattested. Semantic review remains same-family/provisional.
- The dynamically loaded runtime evaluator was not re-read remotely. Its available local source copy matches the source-port hash, and the recorded run asserts native-counter agreement; this audit's strongest direct validation is the complete selected Box/GT recount.

| Claim | Impact |
|---|---|
| The two geometry-head continuation runs and9508-expression evaluations actually closed. | supported: Exit0/status, complete rows, source-linked runtime receipts and local hashes. |
| Strategy4509 exceeds same-budget control4499 by10 and parent4506 by3 at native bbs Acc@.50. | supported_with_qualifiers: Exact single-seed development-row counts; no significance/generalization claim. |
| Only the existing456102-parameter geometry head trained; parents and zeroR were frozen. | supported_by_source_and_runtime_receipts: Exact trainability/gradient/state assertions and complete train/restore evidence; no local weight tensor replay. |
| The same-model5615/4754 target has been met. | unsupported_false: Actual metric-best5614/4509 is below both thresholds; existing SUMMARY correctly reports false. |
| All Masks/full256 candidate geometry were replayed independently on CPU. | unsupported: Only saved scalar/indicator recount is possible; existing RESULTS does not make this claim. |
| Own Query support proves physical identity, broader task novelty or Nr3D/Sr3D success. | unsupported: No such claim is supported or made by the audited closed report. |

## Disposition

- No blocking code or numerical correction is required for the narrowly scoped closed report.
- Keep the existing scope qualifications when reporting results; treat the plan/tracker as timestamped prelaunch history and terminal artifacts as the closure record.
- Keep metric-best selection separate from same-model target success; the audited target is unmet.

Current-hash manifest: 127 reviewed inputs are listed in `EXPERIMENT_AUDIT.json`, including explicit review extent. `controller_body.py` is absent. SOURCE_REVIEW and LAUNCH_REVIEW exist, remain SOURCE_ONLY/same-family/provisional/backend-unattested records, and were not substituted for this terminal audit.
