# Experiment audit: actual native Face preflight pair

**Verdict: WARN. Blocking findings: 0. Engineering checks: PASS within the stated scope.**

No blocker to the stated actual initialization/gradient/save-restore engineering claims. Both real runs completed exit 0 with identical recorded collated inputs. WARN preserves remote-payload, same-process and non-accuracy limits.

Execution scope: `ACTUAL_NATIVE_FACE_PREFLIGHT_PAIR`. Assurance: `same-family` / `provisional` / reviewer identity `UNATTESTED`. Policy requested gpt-6-astra at max effort; actual identity is not attested.

Reviewed: 2026-10-11T12:11:04.464164+08:00. All 38 declared file digests match; six additional original source bodies match the exact 118-entry source catalog. The manifest, nine mandatory instruction/context reads and generated deterministic check record are also hash-bound (55 total). Context reads are not experiment evidence.

Actions: local reading, hashing, 17 Python AST parses, four embedded-script AST parses and review-output writes. No SSH, GPU query, Torch import, data/model load, training, source change or deletion.

## Actual pair

| Field | without_additional_source | source_conditioned |
|---|---:|---:|
| started_cst | 2026-10-11T11:19:57.699848+08:00 | 2026-10-11T11:32:50.453065+08:00 |
| completed_cst | 2026-10-11T11:30:26.537009+08:00 | 2026-10-11T11:43:08.923157+08:00 |
| elapsed_seconds | 628.8369479253888 | 618.4698750153184 |
| exit_code | 0 | 0 |
| actual_input_rows | 16 | 16 |
| actual_updates | 2 | 2 |
| full_state_tensors | 1301 | 1301 |
| restored_optimizer_states | 826 | 826 |
| face_output_gradients | [0.3264526128768921, 0.06731867045164108] | [0.3265463411808014, 0.06723404675722122] |
| second_face_internal_gradient | 2.0552264686557464e-05 | 2.06050190172391e-05 |
| face_internal_changed_elements | 20176 | 21694 |
| face_output_changed_elements | 33 | 33 |
| checkpoint_bytes | 841963854 | 841963854 |

Both runs record these exact full collated-input SHA-256 values:

- `8e775089188d7c475658a15b344cace39b6b2c3c13b8326b3b91f97382ce0d30`
- `293d97ae2ad630e3a8293a188a84459fcfcc448a4a8b484c258735752a4eabf7`

The clock-derived durations agree with controller elapsed time within 0.22 milliseconds. Nine intake file entries per arm match directly read bytes and SHA values; status, diagnostic and receipt fields agree. The native logs show both updates and checkpoint save. Evidence: both `actual_engineering_*/preflight.log:13`, `:15`, `:18`, `:22` and `witness/NATIVE_M0_RECEIPT.json:12`, `:16`, `:23`, `:43`, `:121`.

## A ground truth provenance — PASS

Real Joint3DDataset training batches supply center_label, size_gts, gt_masks and language maps after model.forward. Prediction-only mask references are model geometry, not GT. Native corresponding_loss terms also use predicted query masks as auxiliary consistency targets; they are training losses, not reported accuracy. Raw dataset contents and original batch tensors were not independently reloaded by this reviewer.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\train_dist_mod.py:55-97; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\train_dist_mod.py:132-175; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\main_utils.py:443-458; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\source\models\losses.py:856-917; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\source\models\losses.py:579-627; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\source\native_mask_geometry.py:5-43.

## B score normalization — PASS

Neither preflight computes formal performance. Raw per-step losses agree with cumulative loss averages in native logs. saved_retained_metrics and formal_accuracy are null. The normal evaluator source uses dataset GT hit counts, not normalization by the maximum of model predictions; the normal entry requires both denominators 9508. This evaluator was not executed in these preflights.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:285-350; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\main_utils.py:473-501; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\train_dist_mod.py:237-257; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\normal_controls_20261010\normal_source_review\private\audited_inputs\runtime_binding\model_source\src\grounding_evaluator.py:289-303; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\normal_controls_20261010\normal_source_review\private\audited_inputs\runtime_binding\model_source\src\grounding_evaluator.py:535-553.

## C result existence — PASS

All 38 manifest input digests match. Each arm has nine directly checked intake file bindings, an exit-0 status/log, receipt, matching training diagnostic and initial comparison. Remote checkpoint inventory hashes and lengths equal their receipts, but model bytes were not independently downloaded or locally loaded. Original protocol timestamps/runtime-false fields are source-preparation snapshots; actual controller status and intake establish completion.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\actual_pair_review\REVIEW_INPUTS.json:1; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\ENGINEERING_without_additional_source_INTAKE.json:94-171; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\ENGINEERING_source_conditioned_INTAKE.json:94-171.

## D dead code and actual calls — WARN

The helper calls native train_one_epoch; its observation wrapper calls and returns the original criterion loss. The native method performs forward, GT attachment, backward, clipping, AdamW and scheduler steps, supported by logs and changed parameters. No evaluate_one_epoch call occurs in the helper. The dormant DIoU function has only commented call sites in the inspected loss; it has no attributed result. Future evaluator source presence is not credited as an executed result.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:203-249; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\main_utils.py:428-470; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\actual_engineering_without_additional_source\preflight.log:13-22; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\actual_engineering_source_conditioned\preflight.log:13-22; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\source\models\losses.py:103-133; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\source\models\losses.py:546-557.

## E scope — WARN

Two configurations at seed 2027, same two B8 collated inputs, 16 row presentations and two optimizer updates per arm. Constructed train/validation lengths are 36665/9508; 9508 is not a completed evaluation. Unique scenes and unique annotation count are not established by stored hashes alone. Evidence supports initialization, gradient reachability, parameter changes and same-process save/restore. It does not establish accuracy, new best, module efficacy, complete training, complete source removal, cold-process or next-batch continuity.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:65-98; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:285-350; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\face_residual_span_mixer.py:35-41.

## F evaluation type — WARN

Classification: real_gt supervised training engineering preflight, with native auxiliary prediction-consistency losses. Formal accuracy evaluation was not executed. No model-generated reference is presented as benchmark accuracy.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\source\models\losses.py:856-886; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:285-350.

## Engineering checks

### E1 — PASS

Both actual input_batch_sha256 lists are equal. Original protocols differ only in native_init_spec and face_residual_mode; initializer specs differ only in face_residual_mode. Admitted protocol copies change only status and serial_gpu_preflight_admitted. Equality covers recorded collated inputs, not bitwise cross-process trajectories.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:65-98; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\actual_engineering_without_additional_source\witness\NATIVE_M0_RECEIPT.json:16-19; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\actual_engineering_source_conditioned\witness\NATIVE_M0_RECEIPT.json:16-19.

### E2 — PASS

Strict official/G/support/span initialization precedes Face wrapping. The executed helper asserts that the 1295 retained E0 states match exactly after explicit axis-prior prefix remapping; six new Face state entries yield 1301. Face adds 23425 parameters and complete span has 53218. Both admissions record retained E0 SHA ed8455ddc67e4d17018e9cf499197140e15e35db3f5daee55e0eec6846faf0c4. That ancestor had C enabled; this C-off configuration is not never-C-trained.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\query_mask_assignment_gpu_20261011\source\native_model_initialization.py:18-74; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\native_face_model_initialization.py:9-29; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:107-130.

### E3 — PASS

Each arm records bitwise initial center/size equality against its own retained axis formula, zero Face movement, zero differing size elements, 46 sizes below 1e-6 and minimum 9.99999883788405e-7. Both native/reference inputs are floored at 1e-6 and gates are bounded. The minimum is exactly one binary32 ULP below binary32(1e-6)=9.999999974752427e-7, supporting float32 convex-mix rounding. This does not compare full output tensors between processes or independently run a baseline forward.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:163-187; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\face_residual_span_mixer.py:51-67; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\query_mask_assignment_gpu_20261011\source\extremal_span_mixer.py:98-105; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\source\native_mask_geometry.py:35-43; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\actual_engineering_without_additional_source\witness\INITIAL_OUTPUT_COMPARISON.json:1-10; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\actual_engineering_source_conditioned\witness\INITIAL_OUTPUT_COMPARISON.json:1-10.

### E4 — PASS

Native Text-mask matching uses class/bbox/GIoU/mask costs 1/0/2/0.0002. Native query-mask and corresponding/fused-mask losses remain. G correction is enabled and records 193 then 293 reassigned queries. Only C extra selected-query-mask supervision is off; helper asserts the C extra-loss keys are absent.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\main_utils.py:277-289; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\source\models\losses.py:267-390; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\source\models\losses.py:908-970; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:206-232.

### E5 — PASS

Both Face output steps have positive pre-clipping gradient norms. With the zero output layer, first-step Face internal gradient is zero; second-step internal gradient is positive and both Face groups change parameters. Core/backbone/support have positive gradients each step and retained span groups also have gradients and changes. The combined original_core_and_G_reader gradient is not isolated G causal evidence.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:137-153; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:196-202; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:256-282; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\actual_engineering_without_additional_source\witness\NATIVE_M0_RECEIPT.json:23-78; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\actual_engineering_source_conditioned\witness\NATIVE_M0_RECEIPT.json:23-78.

### E6 — PASS

Both saves contain 1301 model states and 826 Adam states, all Adam steps equal 2. Rebuilt model/optimizer/scheduler are compared for exact model values, optimizer keys/groups/values, scheduler and Python/NumPy/Torch/CUDA global RNG. This occurs within one process; engineering epoch 0 restores start_epoch 1. DataLoader generator/worker/sampler progress, next batch and cold-process trajectory recovery are not tested.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:285-329; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\main_utils.py:133-164.

### E7 — PASS

All 118 staging hash entries equal SOURCE_HASHES and both protocol catalogs; paths match the isolated PV root. Six locally read original source bodies match those pins. Nine recorded PV module origins map into the catalog; pcdet.config is recorded under the established external runtime. The controller rehashes every source before execution and the observer rehashes them after closure. This review does not freshly inspect all remote source bodies or the external runtime payload.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\SOURCE_STAGE_RECEIPT.json:1; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\face_engineering_controller.py:19-35; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\observe_engineering_authorized.py:49-63; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\native_face_preflight.py:348-349.

### E8 — PASS

without_additional_source zeros only added Face residual tokens and source_fraction. The retained source-aware axis prior, prior gates, geometry, query and parameter capacity are common. This is control of additional direct Face source evidence, not a source-free model or complete source ablation.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\face_residual_span_mixer.py:35-49; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\face_residual_span_mixer.py:69-77; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\query_mask_assignment_gpu_20261011\source\extremal_span_mixer.py:61-105.

### E9 — PASS

The existing normal launcher requires this exact ACTUAL_NATIVE_FACE_PREFLIGHT_PAIR audit, bound intakes/receipts and its SOURCE_ONLY review. Its normal child forbids checkpoint_path and loads fresh native initializer weights, not M0 model/Adam. This audit satisfies only the engineering-audit prerequisite; live source/environment/ancestor/process/GPU/disk gates and a separate launch action remain necessary.

Evidence: D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\normal_training\launch_normal_authorized.py:20-60; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\normal_training\launch_normal_authorized.py:65-120; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\normal_training\normal_joint_controller.py:34-38; D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_native_gpu_v2_20261011\source\train_dist_mod.py:128-129.

## Unscored remote checkpoint inventory

- without_additional_source: 841963854 bytes, SHA-256 `a657f014073b6fb6fc6f3ff094069e4438abee8e02f33c408848ea1fe9044a5e`.
- source_conditioned: 841963854 bytes, SHA-256 `26cf4b0885f5aee3ba23fc07405c1e75739f5f30bb336b754d839c862c8e31f1`.

These hashes are in the saved helper receipts and closed-run intake tables. Model payload bytes were not independently opened locally. No cleanup has been performed or verified by this review.

## Nonblocking qualifications

- **W1 REMOTE_PAYLOAD**: Engineering checkpoint bytes are receipt-bound remote artifacts, not independent local payload audits. Each is 841963854 bytes and unscored. Cleanup is not verified or performed.
- **W2 RESTORE_SCOPE**: Same-process object reconstruction and global state equality only; not cold-process, DataLoader/worker state restoration, next-batch or continuous replay.
- **W3 CLAIM_SCOPE**: No formal 9508 accuracy, new best, three effective modules, normal three-epoch completion or Nr3D/Sr3D result.
- **W4 CONTROL_SCOPE**: Both arms retain source through the old axis prior/common network. Only added Face direct evidence is suppressed; C-off also retains C-trained ancestor states.
- **W5 GRADIENT_ATTRIBUTION**: Pre-clipping group gradients and parameter changes do not isolate every G parameter or establish efficacy.
- **W6 REVIEW_ASSURANCE**: Same-family/provisional; actual reviewer model/effort UNATTESTED. Fresh invocation was not blinded: executor summaries and mandatory session context were supplied. Primary experiment artifacts were independently inspected and hashed.
- **W7 PAIR_EQUALITY**: Recorded collated input identity and per-arm prior-formula equality do not prove bitwise trajectories. First-step losses differ by about 9.54e-7.

## Normal launcher boundary

The current reviewed launcher SHA-256 is `f4f7b2d1f62ca0d5774e0892299ca9df7d17b03470878ac7f2b71fe0e986a74c`. Its SOURCE_ONLY review remains WARN with zero source blockers. This report fulfills only its engineering-audit prerequisite. Full normal-training admission and launch have not occurred in this review. The live source/environment/ancestor/process/GPU/disk gates remain required; no M0 model or optimizer state is reused.

## Actions

- Retain all seven scope/assurance qualifiers. No new GPU recomputation is required for these bounded engineering claims.
- Use the unchanged reviewed normal launcher for any separately authorized next admission; its live gates remain required.
- Claim exact checkpoint cleanup only after a separately evidenced action.
- Obtain complete formal runs, cold recovery and an actual-result audit before accuracy or efficacy claims.

All absolute input paths, source line references, exact hashes and deterministic comparison results are in `EXPERIMENT_AUDIT.json` and `DETERMINISTIC_CHECKS.json`.
