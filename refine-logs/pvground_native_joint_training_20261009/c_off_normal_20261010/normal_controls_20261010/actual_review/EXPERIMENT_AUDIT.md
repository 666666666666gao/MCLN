# Experiment Audit: actual native C-off preflight

**Verdict: WARN; blocking findings: 0.** The supplied execution evidence satisfies the bounded two-update engineering contract. Same-family/provisional review. Formal C-off precision and live normal-launch admission remain pending.

Date: 2026-10-10T19:54:12.699209+08:00. Scope: ACTUAL_NATIVE_C_OFF_PREFLIGHT. Requested reviewer: gpt-6-astra, effort max. Actual model/effort: **UNATTESTED**.

Evidence paths are relative to `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2`; `../runtime_binding` is its sibling. The JSON records 41 resolved input paths and SHA-256 values. Full request, response and snapshots under `.aris/traces/` remain PRIVATE and must not be published.

## A–F checks

### A. Ground-truth provenance — PASS

The byte-matched deployed reader loads ScanRefer split JSON/TXT and object_id, constructs masks from scan.three_d_objects point membership and boxes from scan.get_object_bbox, and returns center_label/size_gts/gt_masks. The training loop appends those fields after model.forward with collision assertions. Raw corpus files were not independently inspected.

Group-free predicted detections are model inputs: butd=true, butd_gt=false and butd_cls=false. They do not replace the supervised target boxes.

The objective is mixed. Box and own/fused-mask losses use dataset GT. Native corresponding_mask/dice losses instead use a thresholded predicted query mask as an auxiliary synthetic consistency target. The model's predicted-mask geometric reference is also internal model-derived evidence. Neither is external benchmark GT.

The initially supplied model_source dataset is historical and does not match the deployed digest. The explicitly added dataset_source copy matches the port and is authoritative for this audit.

Evidence: `../runtime_binding/dataset_source/src/joint_det_dataset.py:585`; `../runtime_binding/dataset_source/src/joint_det_dataset.py:1086`; `../runtime_binding/dataset_source/src/joint_det_dataset.py:1189`; `../runtime_binding/dataset_source/src/joint_det_dataset.py:1387`; `source/train_dist_mod.py:163`; `source/main_utils.py:449`; `source/models/losses.py:856`; `source/models/losses.py:579`; `source/models/losses.py:592`; `source/models/pv_ground.py:593`; `NATIVE_SOURCE_PORT.json:564`.

### B. Score normalization and log arithmetic — PASS

No reported accuracy is normalized by prediction extrema. The source-defined future evaluator divides hit counts by GT expression counts and computes box/mask IoU using intersection over geometric union.

Softmax, standard Dice denominators and prediction-dependent auxiliary consistency weights are training/ranking operations, not rescaled accuracy claims.

The two raw total losses are 13.838889122009277 and 14.264213562011719. Their mean is 14.051551342010498, matching the second cumulative log value 14.0516. The log does not claim that 14.0516 is the second raw loss. step_seconds=25.904242292046547 covers both updates.

Evidence: `source/main_utils.py:415`; `source/main_utils.py:473`; `source/main_utils.py:481`; `normal_controls_20261010/actual_engineering/preflight.log:15`; `normal_controls_20261010/actual_engineering/preflight.log:18`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:60`; `source/models/losses.py:70`; `source/models/losses.py:393`; `../runtime_binding/model_source/src/grounding_evaluator.py:296`; `../runtime_binding/model_source/src/grounding_evaluator.py:897`; `source/train_dist_mod.py:237`; `normal_controls_20261010/native_c_off_preflight.py:175`.

### C. Result existence and consistency — PASS

All 41 supplied and explicitly added inputs exist and were snapshotted and hashed. Eight supplied actual text artifacts and both executed helpers match intake/admission bytes and SHA. Nine authoritative supplied deployed code copies match the port. Intake embeds the same 116-entry port as NATIVE_SOURCE_PORT.

Launch/read exits and preflight.exit are zero; the last observation records controller closure and complete equality/recovery status. Diagnostic and receipt have identical gradients, deltas and raw losses. Checkpoint size and SHA agree across receipt and intake.

The exact executed assertion path supports two updates, sixteen training rows and full 1295-tensor recovery. Engineering retained_metrics and formal_accuracy are null. Parent E0 counts 5677/4920 identify the earlier checkpoint; they are not new C-off accuracy.

The reviewer did not load checkpoint binaries or independently rerun remote/neural work. These are source and execution-artifact findings, not independent full-benchmark reproduction.

Evidence: `normal_controls_20261010/ENGINEERING_INTAKE.json:29`; `normal_controls_20261010/ENGINEERING_INTAKE.json:56`; `normal_controls_20261010/ENGINEERING_INTAKE.json:113`; `normal_controls_20261010/ENGINEERING_READ_EXIT.json:1`; `normal_controls_20261010/ENGINEERING_LAUNCH_EXIT.json:1`; `normal_controls_20261010/actual_engineering/preflight.exit:1`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:7`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:88`; `normal_controls_20261010/native_c_off_preflight.py:209`; `normal_terminal_recovery_20261010/attempt2/RECOVERY_RESULT.json:5`.

### D. Executed versus inactive code — WARN

The required engineering path is live: native factory, PVGround.forward, compute_hungarian_loss, backward, AdamW/scheduler updates, save_checkpoint and load_checkpoint. The observation wrapper returns native loss unchanged and records two callback visits.

G stays enabled and records 193/293 reassigned queries. C is intentionally disabled; its extra-loss and extra-row keys are asserted absent. The selected-query-loss implementation is not executed in this arm.

The initial no-grad forward checks the mixing formula on the first training batch. GroundingEvaluator is not invoked by this preflight. Its source presence and the unused commented DIoU branch supply no actual accuracy/DIoU result. This is a scope warning, not an absent required engineering operation.

Evidence: `normal_controls_20261010/native_c_off_preflight.py:113`; `normal_controls_20261010/native_c_off_preflight.py:144`; `normal_controls_20261010/native_c_off_preflight.py:172`; `source/main_utils.py:450`; `source/main_utils.py:460`; `source/models/losses.py:917`; `source/models/losses.py:959`; `source/models/losses.py:546`; `source/selected_query_mask_objective.py:7`; `source/train_dist_mod.py:181`; `../runtime_binding/model_source/src/grounding_evaluator.py:194`.

### E. Scope and launch prerequisites — WARN

Actual scope is one C-off/extremal_support configuration, seed 2027, one rank, batch 8, two training batches and sixteen consumed rows. Unique scene count is not recorded. Dataset construction counts 36665 train and 9508 validation rows; 4583 is full-loader length, not actual completed updates.

No formal C-off validation or three-epoch continuation is completed here. The receipt marks formal_accuracy=null and actual_normal_epoch_training_started=false. The prepared 13749-update recipe belongs to a later stage.

The official/G/support/span parent chain includes C-adapted support and 3723 prior span updates. This control can test incremental C during matched continuation; it cannot establish a never-C history, three effective contributions or Nr3D/Sr3D success.

Preparation/protocol/plan and embedded port pending fields are dated snapshots. The newer terminal receipt controls actual engineering status; old pending fields neither cancel completion nor establish formal precision.

The actual-preflight prerequisite is satisfied within this provisional review. Normal launch still requires the existing fresh source/parent/helper/environment, live storage/GPU and flock checks. Newer recorded cleanup arithmetic projects 28700204 bytes above reserve after retiring M0; that is not current admission.

Evidence: `normal_controls_20261010/native_c_off_preflight.py:64`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:7`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:108`; `normal_controls_20261010/NORMAL_NATIVE_RUN_PROTOCOL.json:70`; `normal_controls_20261010/NORMAL_NATIVE_RUN_PROTOCOL.json:98`; `normal_controls_20261010/CONTROL_PREPARATION.json:42`; `normal_controls_20261010/EXPERIMENT_PLAN_SOURCE.md:40`; `normal_controls_20261010/init.json:17`; `normal_controls_20261010/launch_c_off_normal_authorized.py:45`; `normal_controls_20261010/launch_c_off_normal_authorized.py:65`; `normal_degradation_assessment_20261010/OLD_INITIAL_ARRAY_DEDUPLICATION.json:73`.

### F. Evaluation type — PASS

real_gt describes the dataset-supervised components of the two updates and the source-defined future evaluator. It does not mean a held-out benchmark was run here.

Native corresponding-mask consistency uses synthetic_proxy training references. Predicted mask geometry is an internal forward reference. Neither substitutes for formal GT.

Initialization equality, gradients/deltas and recovery are engineering state checks, not task-accuracy evaluation.

Evidence: `source/models/losses.py:579`; `source/models/losses.py:612`; `source/models/pv_ground.py:593`; `../runtime_binding/model_source/src/grounding_evaluator.py:535`; `../runtime_binding/model_source/src/grounding_evaluator.py:879`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:100`.

## Exact engineering assertions

**INITIALIZATION — PASS.** Native factory loads four hash-bound parent components. It compares all 1295 entries to normal E0 using torch.equal, with full key coverage and architecture differing only in C=false. This is tensor-value equality, not serialized-file equality or a new 9508-row output test.

Evidence: `source/native_model_initialization.py:18`; `normal_controls_20261010/native_c_off_preflight.py:73`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:4`; `normal_terminal_recovery_20261010/attempt2/RECOVERY_RESULT.json:5`.

**CORE_TRAINABILITY — PASS.** Original trainability is retained; RoBERTa parameters have requires_grad=false, no text parameters enter AdamW and every trainable parameter identity is covered once by optimizer groups.

Evidence: `source/models/pv_ground.py:173`; `source/native_model_initialization.py:60`; `source/main_utils.py:292`; `normal_controls_20261010/native_c_off_preflight.py:88`.

**TWO_NATIVE_UPDATES — PASS.** Two callback visits, native backward/optimizer/scheduler steps and saved optimizer-state step==2 are asserted before the receipt. Raw losses agree with diagnostic and logged running means.

Evidence: `normal_controls_20261010/native_c_off_preflight.py:144`; `normal_controls_20261010/native_c_off_preflight.py:176`; `normal_controls_20261010/native_c_off_preflight.py:215`; `source/main_utils.py:460`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:10`.

**GRADIENTS_AND_DELTAS — PASS.** Hooks collect finite pre-clipping gradient norms. All reported groups have positive norms on both observed steps and nonzero final deltas. Changed parameter counts: core/G aggregate 693, backbone 102, support 10, span-internal 12, span-output 2. This does not prove every core parameter or individual G submodule changed.

Evidence: `normal_controls_20261010/native_c_off_preflight.py:130`; `normal_controls_20261010/native_c_off_preflight.py:183`; `normal_controls_20261010/native_c_off_preflight.py:200`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:17`.

**C_OFF_G_ON — PASS.** C=false and G=true are checked on the constructed model. C keys must remain absent after the unchanged native loss. Receipt records C-disabled flags and G reassignment 193/293.

Evidence: `normal_controls_20261010/native_c_off_preflight.py:74`; `normal_controls_20261010/native_c_off_preflight.py:146`; `source/models/losses.py:959`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:60`.

**SAVE — PASS.** Native saver includes model, AdamW, scheduler, architecture, epoch and Python/NumPy/Torch/all-CUDA RNG. Loadback checks 1295 tensors, nonempty AdamW state and step 2; retained_metrics=null. Checkpoint size 841675936 and SHA match receipt/intake.

Evidence: `source/main_utils.py:151`; `normal_controls_20261010/native_c_off_preflight.py:209`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:91`; `normal_controls_20261010/ENGINEERING_INTAKE.json:113`.

**RECOVER — PASS.** A newly constructed architecture is loaded through the native loader. Comparisons cover model keys/values, optimizer groups and every state entry, scheduler state, start_epoch=1, Python RNG, the full NumPy RNG tuple, Torch RNG and all CUDA entries. Receipt reports 820 optimizer states. No subsequent update or bitwise full-run reproduction is claimed.

Evidence: `source/train_dist_mod.py:128`; `source/main_utils.py:133`; `normal_controls_20261010/native_c_off_preflight.py:221`; `normal_controls_20261010/native_c_off_preflight.py:228`; `normal_controls_20261010/native_c_off_preflight.py:242`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:94`.

**NO_CARRYOVER_OR_FORMAL_SCORE — PASS.** Actual stage is unscored and normal epochs have not started. Future normal command forbids checkpoint_path/frozen and creates a fresh optimizer/scheduler from protected initialization components. The future source route is verified; its execution is not asserted.

Evidence: `normal_controls_20261010/native_c_off_preflight.py:209`; `normal_controls_20261010/native_c_off_preflight.py:269`; `normal_controls_20261010/normal_joint_controller.py:31`; `source/main_utils.py:321`; `normal_controls_20261010/actual_engineering/witness/NATIVE_M0_RECEIPT.json:108`.

## Recorded numeric evidence

| Quantity | Value and interpretation |
|---|---|
| Actual training | 2 updates, 2 batches, 16 consumed rows, seed 2027, batch 8 |
| Constructed datasets | 36665 train, 9508 validation; validation loop not run |
| Full-loader length | 4583 updates per planned epoch; 36664 rows after drop_last; not completed here |
| Raw total losses | 13.838889122009277; 14.264213562011719 |
| Second logged loss | 14.0516 = cumulative mean 14.051551342010498 |
| Timing | 25.904242292046547 s for both updates; 647.0771946460009 s controller elapsed |
| State | 1295 model entries; 820 optimizer states; each saved optimizer step 2 |
| Engineering checkpoint | 841675936 bytes; SHA-256 c3f151d0fed401a79dede2f9e052e7572a24d8a9f9b16b7b725efa856e5aaa65; retained_metrics=null |
| Formal precision | null; no C-off benchmark result |

## Later-launch prerequisites

The actual receipt and this audit can satisfy the existing local actual-stage gate. Separately reviewed source input hashes still match. Current remote source/environment/parent/helper equality, closed-controller, disk, idle-GPU and flock checks remain required by the existing launcher. This report does not replace them. Evidence: `normal_controls_20261010/launch_c_off_normal_authorized.py:15`; `normal_controls_20261010/launch_c_off_normal_authorized.py:45`; `normal_controls_20261010/launch_c_off_normal_authorized.py:50`; `normal_controls_20261010/launch_c_off_normal_authorized.py:65`; `normal_controls_20261010/launch_c_off_normal_authorized.py:100`.

Earlier engineering admission free capacity was 2464931840 bytes, 140220020 below the normal reserve. Newer deduplication records 1792176128 free bytes; adding the recorded M0 checkpoint 841675936 gives 2633852064, a historical projected margin of 28700204 over 2605151860. This incorporates newer evidence without claiming live admission. Evidence: `normal_controls_20261010/actual_engineering/admission.json:6`; `normal_degradation_assessment_20261010/EXACT_E3_WEIGHT_RETIREMENT.json:20`; `normal_degradation_assessment_20261010/OLD_INITIAL_ARRAY_DEDUPLICATION.json:73`; `normal_controls_20261010/launch_c_off_normal_authorized.py:65`.

No new experiment, source patch, benchmark rerun or defensive framework is requested by this audit. Later comparison remains the planned matched continuation, fixed E3 and predefined best, with historical C exposure disclosed.

## Limitations

- No SSH, neural execution, model/experiment-module import, checkpoint deserialization, installation, deletion or executor-file edits were performed by the reviewer.
- Raw ScanRefer/ScanNet corpus and checkpoint binaries were not independently inspected. Remote execution, equality and recovery are supported by byte-bound source and terminal receipts, not independent neural reproduction.
- Nine supplied authoritative deployed code copies were compared to the 116-entry port. The launcher/observer sources enforce the remote manifest; this reviewer did not inspect all 116 files locally.
- The historical model_source dataset differs from deployed code. The added dataset_source copy matches SHA 3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d and is authoritative for provenance.
- Gradient/update evidence is aggregate by group. Equality follows the actual torch.equal/state-dictionary checks, not serialized checkpoint byte equality.
- Recovery checks state immediately after load; it does not certify subsequent full-run numerical reproducibility. Engineering epoch 0 is an unscored two-update test convention, not a completed normal epoch.
- The 9508 validation rows are only a constructed dataset count here. Formal C-off accuracy, effect, fair continuation comparison, three effective mechanisms and Nr3D/Sr3D success are unestablished by this stage.
- Historical cleanup/archive receipts do not establish current storage/GPU admission. Binary archives were not reopened.
- Requested routing is gpt-6-astra/max; actual model/effort runtime attestation is unavailable and remains UNATTESTED.
- Mandatory workspace startup files were read under AGENTS.md and excluded from audit evidence. Findings rely on the explicitly supplied artifacts and private byte snapshots.

## Claim impact

Supported within supplied-artifact limits: two native updates, C disabled/G active, initial tensor-value equality, group gradients/deltas, native save/load and exact serialized-state recovery. Unsupported by this stage: new C-off accuracy/benefit, never-C history, three effective contributions, Nr3D/Sr3D results or independent full-benchmark reproduction.

