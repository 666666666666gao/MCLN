# Experiment audit: closed control recount and checkpoint archive

**Verdict: WARN. Blocking findings: 0.** The independent saved-artifact checks pass. The qualifications below limit what the evidence establishes.

Fresh Codex reviewer, same-family / provisional. Requested model: `gpt-6-astra`; requested effort: `max`. Actual model and effort: **UNATTESTED** because runtime attestation was not available. No cross-family acceptance is claimed.

Date: 2026-10-13T14:57:37.680122+08:00. Primary artifact root: `C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012`. All output is under `control_terminal_audit/`; this path resolves through the existing filesystem mapping to `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\control_terminal_audit`.

The audit read primary source, raw captured JSONL, CSVs, compressed transport stdout, receipts, and local CPU checkpoints. It executed no SSH, GPU query, model forward, training, restart, or deletion. No current remote process or disk state is inferred.

## Independent results

The independent scalar implementation reproduced all 28,524 records and every submitted CSV IoU value. All 242 main deterministic checks passed; the additional source/identity checks and matcher-note binding checks also passed. CPU environment: Python3.7.6, Torch1.13.1+cpu, NumPy1.21.6. The author’s recount script was read but not executed.

| Epoch | Hits >0.25 / 9508 | Acc >0.25 | Hits >0.5 / 9508 | Acc >0.5 | Repairs / breaks vs E0 at0.5 |
|---|---:|---:|---:|---:|---:|
| E0 | 5677 | 59.707615% | 4920 | 51.745898% | 0 / 0 |
| E1 | 5655 | 59.476231% | 4620 | 48.590660% | 406 / 706 |
| E2 | 5616 | 59.066050% | 4596 | 48.338241% | 388 / 712 |

Each evaluation covers141 scenes and2068 scene-target pairs. All9508 saved point-cloud/mask fingerprint pairs, row order, scan/target/text identities, and GT coordinates match E0. The reduced `(scan,target,text-hash)` key has9506 distinct values: row pairs2991/3009 and5798/5801 repeat that key and GT. Unique row indices preserve pairing; annotation IDs and raw membership were not independently reloaded. Query-slot changes8915 and8995 are not instance-identity changes.

There are zero CPU64-versus-logged threshold disagreements at either threshold for final/native/mask endpoints. The largest logged-versus-CPU coordinate-IoU difference is about6.335e-6; no logged selected endpoint IoU exceeds1. E0 remains the retained best, and4920→4620→4596 satisfies the two-strict-decline rule. The9166-update record is consistent with2×4583 batches and one optimizer step per batch; the retired latest optimizer-state counters were not independently recounted.

The independently reproduced final-versus-native repairs/breaks at0.5 are686/261,444/59,596/72. Final-versus-mask-endpoint values are103/64,285/494,271/492. These are comparisons of saved predictions at the same selected slot within each epoch, not retrained ablations.

## A–F integrity checks

### A ground truth provenance: PASS

ScanRefer annotation object IDs identify ScanNet objects. Dataset boxes and masks come from dataset object points, and validation disables training augmentation. The diagnostic root_gt_box reads center_label/size_gts slot0. Neither selected predictions nor mask-derived endpoints are used as GT. Native bbs ranking uses the existing parsed text maps; root box/mask GT is used for scoring/diagnostics. This is verification of the repository native protocol, not external certification against an independently run official benchmark implementation.

Evidence: `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\runtime_binding\dataset_source\src\joint_det_dataset.py:585`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\runtime_binding\dataset_source\src\joint_det_dataset.py:1086`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\runtime_binding\dataset_source\src\joint_det_dataset.py:1387`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\runtime_binding\dataset_source\src\visual_data_handlers.py:225`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v4\selected_geometry_eval_diagnostics.py:35`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\source\native_root_bbs.py:4`.

### B score normalization: PASS

The formal denominator is 9508 dataset records. Accuracy is hits/9508, and box IoU is intersection/(prediction volume + GT volume - intersection). Strict >0.25 and >0.5 decisions were independently reproduced; no threshold disagreement or above-one logged selected IoU was found. Semantic softmax supplies candidate scores, not a prediction-derived denominator for formal accuracy. Mask support uses 2*intersection>union, a strict0.5 rule.

Evidence: `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v2\models\losses.py:70`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\runtime_binding\model_source\src\grounding_evaluator.py:296`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v4\train_dist_mod.py:255`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v4\selected_geometry_eval_diagnostics.py:96`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\control_terminal_audit\INDEPENDENT_RECOUNT.json:1`.

### C result existence and identity: PASS

All required primary files exist. ct3 intake files match their size/SHA records; original compressed payloads decompress byte-for-byte to the local rows; ct3 and e2v2 terminal statuses agree. Independent scalar IoU reproduces every submitted CSV value and all claimed hit/repair/break totals. Checkpoint/witness/archive/retirement joins pass. Earlier pending fields remain historical facts; this audit adds later evidence without changing the original receipts.

Evidence: `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\control_terminal_audit\DETERMINISTIC_CHECKS.json:1`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\control_terminal_audit\AUDITED_INPUT_HASHES.json:1`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\control_terminal_audit\INDEPENDENT_CHECKPOINT.json:1`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\control_terminal_audit\SUPPLEMENTAL_EVIDENCE.json:1`.

### D call chain and dead code: WARN

The claimed path is connected: controller invokes train_dist_mod; BaseTrainTester evaluates E0 and each completed epoch; evaluate_one_epoch invokes _main_eval_branch, native GroundingEvaluator.evaluate and diagnostics.append/finish. GroundingEvaluator dispatches bbs to evaluate_bbox_by_pos_align; native_root_bbs reproduces its root text-score definition for diagnostics. Saved rows and metric agreement bind the diagnostic path. The unclaimed DIoU helper is dormant; no result is attributed to it.

Evidence: `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\normal_run_controller_v3.py:58`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v3\main_utils.py:378`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v3\main_utils.py:549`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v4\train_dist_mod.py:209`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v4\train_dist_mod.py:233`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\runtime_binding\model_source\src\grounding_evaluator.py:194`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v2\models\losses.py:103`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v2\models\losses.py:546`.

### E scope and claims: WARN

One control configuration, one seed2027, 141 validation scenes, 2068 scene-target pairs and 9508 records/evaluation; E0 plus E1/E2. Retained best remains E0. Two strict Acc@0.5 declines support the recorded stop. Saved strategy evidence is an admitted-child snapshot with no completed metric. Full256 coverage/mask support cannot be independently reconstructed from this compact record. No robustness, new-best, strategy improvement, causal attribution, or goal-completion claim is justified.

Evidence: `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\native_control_protocol.json:58`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v3\main_utils.py:167`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\OBS_selected_geometry_sw_INTAKE.json:15`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\control_terminal_audit\INDEPENDENT_RECOUNT.json:1`.

### F evaluation type: PASS

Formal bbs accuracy and saved endpoint IoU comparisons use dataset GT. Candidate-coverage values are GT-referenced oracle diagnostics, not deployed accuracy; prediction-derived native/mask boxes remain predictions. The supplemental matcher note is source/algebraic analysis, not an additional runtime evaluation.

Evidence: `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v4\selected_geometry_eval_diagnostics.py:29`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\source_overlay_v4\selected_geometry_eval_diagnostics.py:69`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\recount_control_rows_20261013.py:92`; `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\control_terminal_audit\MATCHER_SOURCE_NOTE_AUDIT.json:1`.

## Compact diagnostic limits and causal claims

Saved full256 coverage at0.5 is8234/8309/8428, with3314/3689/3832 wrong selected answers despite a logged qualified candidate. The audit verifies these sums and scalar count/rank consistency. It cannot recompute all256 candidate IoUs or native ranking without the missing boxes/logits. Likewise, support-qualified counts5114/5118/5090 and wrong-box counts494/801/804 at0.5 are arithmetic on logged mask intersections/unions; masks were not reconstructed.

Support-qualified is not equivalent to eligibility for the selected-geometry training objective: original matching assignments are absent. `selected_support_geometry_objective.py:26` excludes all originally matched queries, while `pvground_semantic_assignment.py:23` qualifies unmatched boxes using final BoxIoU>0.5. The source responsibility map accurately distinguishes geometry and CE/contrast duties, but it is not a measured gradient conflict or failure explanation. A’s corrected own mask is passed into native mask losses (`models/pv_ground.py:568`; `models/losses.py:584`). The mixer’s deployed gate is clamped to[0,1] (`extremal_span_mixer.py:100`), so a negative raw gate is not a negative deployed coefficient.

Neither the control decline nor endpoint comparisons establish strategy efficacy, a unique failure cause, or a general result across datasets/seeds. The supplied selected-geometry receipt/intake records an admitted child at2026-10-13T14:27:43+08:00 with `actual_metrics:null`; it supplies no strategy result and makes no statement about current runtime state.

## Checkpoint content and retirement

Both local checkpoint files exist and were hashed and loaded on CPU. All1295 model keys/order/dtypes/shapes/tensor-byte hashes agree with each other, the protected manifest, and the recorded original remote witness. The original91,175-byte non-model payload exactly matches the decoded remote stdout payload. The reconstructed checkpoint preserves `config`, `optimizer`, `scheduler`, `epoch`, `architecture`, `retained_metrics`, `rng`, and original state-dict `_metadata`. Its epoch is0, optimizer state is empty with four parameter groups, and retained metrics are5677/4920.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| Protected earlier E0 |615023752|`ed8455ddc67e4d17018e9cf499197140e15e35db3f5daee55e0eec6846faf0c4`|
| Historical remote control best |615023880|`fd47120083ef457c1ad978ea11f52b75852284e887cb5c2bfa3f710a7b075f39`|
| Reconstructed control best |614925967|`d686f1bbb1eb2a5092777a953ef63d63752f73182eb464b9837a4f7007ad264b`|
| Original non-model payload |91175|`6da6afab2b571ae5b108b474a3adec544e06738290a15cd0efc5ab4fcd9245c5`|

This establishes content preservation conditional on the captured original-content witness; the deleted original ZIP cannot now be independently reread locally. It does not establish an original byte copy or cold normal recovery. No model was constructed or evaluated. The source even asserts `not args.checkpoint_path` in the ordinary training main (`source_overlay_v3/main_utils.py:374`), and the protocol declares `cold_resume_supported:false`.

The retirement source restricts deletion to the exact closed-control parent directory and the two hashed names (`retire_completed_control_weights_20261013.py:24–46`). Receipt, original stdout, exit0, capacity hashes, archive receipt, and terminal-receipt hash agree. Recorded removal is615023880-byte best plus841677024-byte losing latest, totaling1456700904 bytes. The losing latest was intentionally not archived; preserved E0 does not preserve the final E2 learned/optimizer state. All three original row files were preserved by the retirement record and subsequently captured with the same hashes. The local incomplete22,577,152-byte SCP partial is absent; it was never a complete original-best archive.

The historical free-space observations are1199345664 before and2656051200 after. They are observation-time values, not current availability. No claim is made that every dependency remains present today. The original status and earlier pending flags remain unchanged historical records.

## Transport and source binding

ct2 exited1 with the preserved `gzip.compress(..., mtime=0)` signature error. The v2 collector removes that argument, ct3 exits0, and the original compressed stdout payloads hash correctly and decompress byte-for-byte to all three saved row files. The collector copies the earlier terminal small files only after verifying their hashes. These are transport recovery records, not neural reruns. Earlier local reconstruction attempts and SCP interruption are described by the failure ledger; they were not replayed, and the SCP termination cause remains unattested.

Fifteen distinct inspected model/data/evaluation source files and the controller helper match the recorded R4 hashes. The terminal admission contains the same complete118-file stage receipt. This audit did not independently rehash all118 local sources or inspect current remote files. The static protocol/initialization documents’ prepared/pending fields are not interpreted as current status.

## Supplemental matcher-mask source note

**PASS within source/algebraic scope.** The four note source bindings match R4. The current proposal/decoder head is `ClsAgnosticPredictHead` (`models/pv_ground.py:234,252`); its implementation writes boxes/scores and no `proposal_pred_masks` (`models/modules.py:135–179`). The last-layer loss therefore passes the broadcast Text mask as `pred_masks` and the own mask separately as `sp_pred_masks` (`models/losses.py:854,911–914`). `HungarianMatcher.forward` reads `pred_masks`, thresholds logits at0, expands to points, and uses0.0002 times L1/Hamming (`models/losses.py:294,328–342,373–377`).

For fixed scene b and GT column j, every Query q receives the same repeated Text mask m_b. Its mask cost0.0002·sum_p|1[m_b(p)>0]−GT_j(p)| has no q dependence. This term cannot distinguish those query rows; class1/GIoU2 costs can still do so, with bbox coefficient0 (`main_utils.py:291`). Query-mask focal/Dice losses remain present (`models/losses.py:584–590`). This supports the note’s limited statement. It provides no measured runtime cost matrix, accuracy gain, novelty result, or unique causal explanation.

## Findings and permitted claim boundary

**Blocking findings: none.** Warnings:

- **W1 — Saved-box verification has explicit raw-data and candidate limits:** The audit recomputed selected final/native/mask-endpoint IoU from saved coordinates. Raw ScanRefer/ScanNet objects, point clouds, masks, 256 candidate boxes/logits, and original matching assignments were not reloaded. Coverage, qualified counts/ranks, and mask support use recorded scalars. Input fingerprint equality is checked from saved hashes, not by recomputing hashes of raw arrays.
- **W2 — Content preservation is distinct from original byte copy and cold recovery:** The retained artifact reconstructs the original E0 state using identical protected model tensors and the original non-model payload. It has a different ZIP hash and length from the retired original. No model construction, strict architecture reload, forward replay, or normal training continuation was executed. The audited normal main explicitly rejects checkpoint resume. The losing E2 latest was intentionally not archived.
- **W3 — One control run does not establish a causal mechanism or strategy benefit:** Scope is one ScanRefer validation setting, seed 2027, E0 and two completed training epochs. Endpoint comparisons are observational comparisons at the selected slot, not retrained module ablations. Coverage gaps, support-qualified errors, and a source responsibility table do not prove the source of decline, extra-loss eligibility, or effectiveness of the selected-geometry strategy. Strategy intake supplied here has no completed metrics.
- **W4 — Record identity is stronger than the reduced text key:** Each epoch has 9508 consecutive unique row indices but 9506 unique (scan_id,target_id,utterance_sha256) tuples. Two repeated normalized identity groups are recorded. Annotation IDs/raw dataset membership were not independently checked. Pairing by row index plus the saved identities and GT is exact across epochs; 8915/8995 query-slot changes do not mean instance-identity changes.
- **W5 — A dormant metric helper does not substantiate an additional result:** calculate_diou_3d is defined in losses.py but its loss_boxes invocation is commented out. The primary last/bbs IoU path is active and its outputs are present. No DIoU training/evaluation effect is supported by these artifacts.
- **W6 — Update count is corroborated by terminal history and source arithmetic:** 9166 equals two completed epochs times 4583 batches with one optimizer.step per batch. This audit did not inspect the retired latest optimizer-state counters, so it is not an independent state-based update recount.

The evidence supports the recorded negative control outcome, saved-coordinate recount, qualified diagnostic summaries, original E0 checkpoint-content preservation, historically scoped retirement, and the matcher note’s source fact. It does not support strategy improvement, completion of the target, independently recomputed full256 coverage, loss-eligibility/gradient conflict frequencies, original-ZIP backup, or verified cold normal recovery.

Evidence files: `DETERMINISTIC_CHECKS.json`, `INDEPENDENT_RECOUNT.json`, three `independent_epoch*_rows.csv`, `INDEPENDENT_CHECKPOINT.json`, `INDEPENDENT_TENSOR_MANIFEST.json`, `SUPPLEMENTAL_EVIDENCE.json`, `MATCHER_SOURCE_NOTE_AUDIT.json`, and `AUDITED_INPUT_HASHES.json`. `EXPERIMENT_AUDIT.json` includes the complete primary input hash ledger. `REVIEW_COMPLETE.json` is written last to bind the frozen report bytes.
