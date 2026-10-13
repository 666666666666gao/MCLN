# Experiment audit: selected-query box family and task paths

**Overall verdict: WARN.** Final saved-data arithmetic, fixed-family maximization and bound source-path statements pass within scope. **No scoped blocking findings remain.** Strict numerical wording and an ambiguous intermediate revision were corrected before sealing. This is not broader model acceptance.

Fresh context; requested **gpt-6-astra / max**; actual model/effort **UNATTESTED**. Independence **same-family**, acceptance **provisional**. Reviewer actions: no remote/GPU query, model construction/forward, autograd/optimizer update, checkpoint/original-dataset load, production-source edit or training action.

| Epoch | Actual >0.25 | Family >0.25 | Actual >0.5 | Family >0.5 | Support-good box-bad >0.5 reachable / unreachable |
|---|---:|---:|---:|---:|---:|
| E0 | 5677 | 5839 | 4920 | 5379 | 271 / 223 |
| E1 | 5679 | 5859 | 4752 | 5335 | 378 / 254 |
| E2 | 5588 | 5793 | 4623 | 5208 | 382 / 267 |

There are 9,508 rows per epoch. At >0.25, support-good/box-bad reachable versus unreachable counts are 27/48,53/55,57/61. The 28,524 rows reuse one validation set: 141 scenes, 2,068 scan-target pairs, 9,506 distinct scan-target-utterance hashes. E0/E1/E2 are snapshots of one seeded run, not three independent runs.

## A ground truth provenance: WARN

Dataset GT path is source-supported; only saved GT coordinates were read. Original annotations/cache/points/masks were not reloaded or byte-attested.

Evidence: [103_joint_det_dataset.py:205](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/103_joint_det_dataset.py:205>), [103_joint_det_dataset.py:596](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/103_joint_det_dataset.py:596>), [103_joint_det_dataset.py:1086](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/103_joint_det_dataset.py:1086>), [103_joint_det_dataset.py:1387](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/103_joint_det_dataset.py:1387>), [105_visual_data_handlers.py:129](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/105_visual_data_handlers.py:129>), [selected_geometry_eval_diagnostics.py:35](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_overlay_v4/selected_geometry_eval_diagnostics.py:35>), [selected_family_capacity_20261014.py:114](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014.py:114>).

## B score normalization: PASS

IoU is intersection/geometric union; raw accuracy is strict hits/9508. No own-output maximum/mean normalizer. GT maximization is explicitly an oracle diagnostic.

Evidence: [selected_family_capacity_20261014.py:12](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014.py:12>), [selected_family_capacity_20261014.py:135](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014.py:135>), [102_grounding_evaluator.py:296](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/102_grounding_evaluator.py:296>), [recount_selected_rows_20261014.py:99](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/recount_selected_rows_20261014.py:99>).

## C existence and numbers: PASS

All28524 rows, family CSV numeric/witness/support fields, family count fields, original recount epoch-summary fields and intake/source hashes reproduce. Incorrect strict wording was corrected before sealing.

Evidence: [REPORT.md:20](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/task_path_and_family_capacity_20261014/REPORT.md:20>), [selected_family_capacity_20261014.py:151](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014.py:151>), [RESULT.json:39](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_geometry_cpu_recount_20261014/RESULT.json:39>), [native_metrics.jsonl:1](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/OBS_selected_geometry_rows_20261014_files/logs/scanrefer/selected_geometry/1791872741/native_metrics.jsonl:1>), [REPORT.md:30](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/task_path_and_family_capacity_20261014/REPORT.md:30>).

## D called vs dead: WARN

Primary family/evaluator paths are called and reproduce artifacts; native objective/model calls are wired. DIoU helper is inactive and not a reported result. Historical chunk IDs are author metadata, not freshly attested execution.

Evidence: [selected_family_capacity_20261014.py:125](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014.py:125>), [selected_family_capacity_20261014.py:175](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014.py:175>), [107_train_dist_mod.py:234](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/107_train_dist_mod.py:234>), [055_losses.py:917](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/055_losses.py:917>), [055_losses.py:546](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/055_losses.py:546>).

## E scope: WARN

One closed-C configuration, seed2027, one ScanRefer validation set, E0/E1/E2 snapshots:9508 rows each,141 scenes, 2068 scan-target pairs, 9506 distinct scan-target-utterance hashes. Three snapshots are not three runs. Query-slot changes are not instance identities.

Evidence: [OBS_selected_geometry_rows_20261014_INTAKE.json:3](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/OBS_selected_geometry_rows_20261014_INTAKE.json:3>), [native_rows_epoch000.jsonl:1](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/OBS_selected_geometry_rows_20261014_files/logs/scanrefer/selected_geometry/1791872741/native_rows_epoch000.jsonl:1>), [013_initialization_reference.json:2](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/013_initialization_reference.json:2>), [REPORT.md:16](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/task_path_and_family_capacity_20261014/REPORT.md:16>), [REPORT.md:25](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/task_path_and_family_capacity_20261014/REPORT.md:25>).

## F evaluation type: WARN

Source-supported real_gt saved-coordinate recount plus GT-conditioned oracle capacity diagnostic; not independently reloaded real_gt, not model-generated GT. GT chooses coefficients, so these are not deployable predictions. Synthetic/random fixtures are numerical checks only.

Evidence: [selected_family_capacity_20261014.py:59](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014.py:59>), [selected_family_capacity_20261014.py:164](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014.py:164>), [recount_selected_rows_20261014.py:112](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/recount_selected_rows_20261014.py:112>), [REPORT.md:32](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/task_path_and_family_capacity_20261014/REPORT.md:32>).

## G mathematical completeness: PASS

Four crossings plus endpoints are complete for positive endpoint/GT sizes. Compactness and coordinatewise affine-fractional monotonicity prove the event Cartesian maximum; flat/tangent/constant cases included.

Evidence: [selected_family_capacity_20261014.py:23](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014.py:23>), [124_extremal_span_mixer.py:98](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/124_extremal_span_mixer.py:98>).

## H precision and thresholds: WARN

Threshold counts stable; smallest maximum-to-threshold margin4.091914164128241e-5. Saved rounded outputs can exceed ideal real-family maxima by9.685783429056241e-9; final text correctly uses one-sided+1e-5 tolerance. Nine rational checks confirm decisions. Failed floor assertion is independently reproduced.

Evidence: [REPORT.md:30](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/task_path_and_family_capacity_20261014/REPORT.md:30>), [FAILED_SOURCE.py:117](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014_attempt1_size_assertion/FAILED_SOURCE.py:117>), [selected_family_capacity_20261014.py:117](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_family_capacity_20261014.py:117>), [native_rows_epoch002.jsonl:7685](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/OBS_selected_geometry_rows_20261014_files/logs/scanrefer/selected_geometry/1791872741/native_rows_epoch002.jsonl:7685>).

## I source targets and graph: PASS

Source-only: nine key sources match current/closed-C stages. A native supervision, B graph, G eos 0.1/outer 0.5/7, C actual-batch normalization and raw-logit threshold distinction verified; no actual gradient claim.

Evidence: [058_pv_ground.py:568](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/058_pv_ground.py:568>), [055_losses.py:945](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/055_losses.py:945>), [130_pvground_semantic_assignment.py:34](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/130_pvground_semantic_assignment.py:34>), [135_selected_support_geometry_objective.py:59](<C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/span_lr_actual_pair_review_20261014/snapshots/135_selected_support_geometry_objective.py:59>).

## Independent numerical and mathematical result

The independent verifier uses scalar IoU and Cartesian products of per-axis overlap lengths/widths with sorted in-range events. It does not import project modules. Across 85,572 saved final/native/mask box pairs, scalar values exactly match the original CPU64 family-CSV values. Across 28,524 family rows, maxima/witness differences are at most 2.631228568361621e-14. Every original recount epoch-summary field and every saved threshold decision reproduces. There are 83 primary+59 supplemental deterministic checks, nine exact rational boundary checks, and a text-only AST check that reproduces the final report from its generator byte-for-byte.

[MATHEMATICAL_REVIEW.md](MATHEMATICAL_REVIEW.md) proves event completeness, positive dimensions/union and coordinatewise maximization, including constant/degenerate-overlap cases. The original six analytic/24 random 13^3 cases reproduce; ten new analytic and forty 17^3-grid cases use numerical seed 419 (not a training seed). Grids are lower-bound sanity checks, not the maximality proof or new experiment scores.

## Resolved precision finding and revision history

Initial REPORT.md:30 claimed every saved final IoU was no greater than the ideal family maximum. E2 row_index 7684 contradicts strict dominance: 0.45504065938941557 versus 0.45504064970363217. Exact saved-coordinate rational arithmetic confirms excess 9.685783429056241e-9. Final wording correctly states saved_final_IoU <= ideal_family_max_IoU + 1e-5 and explicitly rejects near-equality/strict-dominance interpretations. The nearest maximum-to-threshold distance is 4.091914164128241e-5; no actual hit loses threshold membership.

The reviewer changed no external report/source. The parent revised the report and its generator. INPUT_REVISION_RECORD.json preserves original/intermediate/final hashes and exact corresponding byte copies. Of the original 46 inputs, only REPORT.md changed; the final generator was added by follow-up, yielding 47 final input bindings. All scientific data/source inputs remained unchanged. “All initial inputs unchanged” is deliberately false.

## Source findings and claim boundaries

[SOURCE_TASK_PATH_REVIEW.md](SOURCE_TASK_PATH_REVIEW.md) details call/target/weight evidence. A reaches matched Query/fused Mask losses. B retains coarse/probability/content graph paths but mask membership is discrete. G retains eos 0.1 and outer 0.5/7. C adds (10*L_bbox+2*L_giou)/7, with actual-batch denominator, for selected unmatched support-qualified root targets; it does not replace CE/contrastive targets. Connectivity does not certify nonzero realized gradients through clamps, selection or support.

Nine key source versions match both current and closed-C stage SHA/bytes; six additional current sources match their manifest. Current main_utils/diagnostics differ from closed-C, so the old diagnostic was separately bound. This does not certify all 118 stage entries or live remote imports. The previous audit report is hash-bound to its seal but its conclusions do not substitute for this primary recount.

Remaining limits: raw annotation/cache/point data and point masks were not reloaded; full256 box/score arrays and original Hungarian eligibility are absent; terminal checkpoint bytes, optimizer state and cold recovery were not tested. Historical CPU chunk IDs are author-recorded metadata, not fresh platform attestation; numerical products were independently reproduced here. The current low-LR run and broader goal are outside scope.

GT-selected coefficients do not show that shared features can learn them. Training changes endpoints/representations/selections, so the fixed-family bound is not a full-model ceiling. Query-slot changes are not stable instance identities. Target differences do not prove gradient conflict, a bug or decline cause. Passing assertions alone prove neither safety nor gain. No deployment, full-model-capacity, causal, identity, realized-gradient or cold-recovery claim is accepted.

## Reproduction and seal

EXPERIMENT_AUDIT.json records checks and claim impacts. INDEPENDENT_RECOUNT.json, DETERMINISTIC_TRACE.json, SUPPLEMENTAL_CHECKS.json, RATIONAL_BOUNDARY_CHECKS.json, three independent CSVs and EXECUTION_RECEIPTS.json preserve calculations. INPUTS_BEFORE/AFTER.json and SOURCE_BINDINGS.json identify exact bytes. REVIEW_COMPLETE.json is written last and binds every final input/output except its recursively impossible self-hash; the seal hash is reported externally. No broader acceptance follows.
