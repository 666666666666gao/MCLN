# Closed experiment integrity audit

**Verdict: WARN. Blocking findings: none.**

The actual closed comparison is supported by source-bound receipts and an independent full-row recount. Native-target control achieves 4511/9508 strict IoU>0.50 hits, member-target 4502, and the protected parent 4509. The member-target intervention loses nine hits versus the same-budget control. No blocking integrity finding was identified; this remains a single-seed development comparison, with limited raw-array replay and observable cross-run numerical drift.

Execution scope: ACTUAL_CLOSED_TERMINAL_AUDIT. Requested reviewer: gpt-6-astra / max; fresh context. Backend model identity is not attested. Review independence: same-family; acceptance: provisional.

## Independently recounted primary result

All values below use 9508 formal ScanRefer development expressions and strict IoU > threshold.

| Model | Hits @0.25 | Acc@0.25 (%) | Hits @0.50 | Acc@0.50 (%) |
|---|---:|---:|---:|---:|
| protected_geometry_parent | 5614 | 59.04501472 | 4509 | 47.42322255 |
| control | 5616 | 59.06604964 | 4511 | 47.44425747 |
| member_target | 5616 | 59.06604964 | 4502 | 47.34960034 |

Control is the measured strict-primary winner: +2 hits over the protected parent. Member-target loses 9 hits against the same-budget control and 7 against the parent. Neither current arm meets the dual target (5615/4754); the best is 243 hits short at 0.50. These are development-selection observations, not significance or generalization findings.

The recount covers 56072 evaluation rows and 7446 training records. Both arms use the identical full 29778-expression order (3722 batches of 8, one batch of 2; 3723 optimizer steps). The 6887-expression module holdout was seen during previous pretraining and is not an independent test set.

## Evidence and method

- Treated author analyses and all older review verdicts as untrusted. Recomputed file identities and calculations independently without importing or executing the author analyzer.
- Read the active runner, auxiliary-loss implementation, actual imported model/loss/dataset copies, native evaluator, closure/preflight/restore receipts, all 7446 training JSONL records and all 56072 evaluation JSONL records. Reused hash equality for duplicated source copies.
- Matched every primary stage to its receipt and intake hash; verified all 153 packet identities twice. Parsed all 52 listed Python artifacts for syntax without executing model code.
- Independently recomputed selected final/coarse AABB IoUs in Python double precision, strict threshold counts, transitions, candidate-coverage scalar aggregations, mask scalar statistics, quartiles, displacement statistics, training counts and row order.
- Read the eight original fixed-cohort NPZ files as NPY arrays with standard-library zip/struct parsing. Independently recomputed the 64-row target-jitter counterfactual from stored boxes, mask intersection/union counts and native matched slots; this is a separate historical diagnostic, not formal-array replay.
- Verified all 24 historical cohort-intake and all 9 target-jitter-intake file identities. These additional inputs support the diagnostic provenance check; their earlier review judgments do not substitute for this audit.

All 205 reviewed input files are individually bound by bytes and SHA-256 in EXPERIMENT_AUDIT.json. All 153 packet files, the 51-file current intake, the 24-file historical cohort intake and the 9-file target-jitter intake match their saved identities.

reviewed_files binds actual bytes read, including files hash-checked for identity and inactive/older evidence consulted only for context. It is not a claim that every bundled routine executed or that earlier review judgments were adopted. Runtime remote files and weights were not accessed by this reviewer.

Selected-box CPU recount can rederive IoUs from the stored selected/coarse box and native GT coordinates. Formal mask and full-candidate metrics can only be recounted from their saved scalar values: no raw formal point masks, full candidate box arrays or complete ranking scores are available. The reviewer did not access remote data/weights, execute a model, alter an experiment file, delete/archive weights, read personal memories or open credentials.

## A. Dataset ground-truth provenance — PASS

ScanRefer object IDs and utterances come from dataset annotation JSON. Native root masks mark ScanNet annotation object-member point indices; native boxes enclose those members. They are not derived from predicted boxes or masks.

The member target is captured from scan.get_object_bbox on the same augmented 50000-point scan representation before the existing independent 0.95+0.1*U box jitter. The hook calls the original target constructor and does not consume RNG. It changes neither native center_label/size_gts nor gt_masks.

The mode-dependent auxiliary_roots argument is used both in Box IoU<=0.5 eligibility and extra L1/GIoU/DFL targets. Own Query-mask AND fused-mask IoU>0.5, all current native matched-query exclusions, and the native GT loss/matching/evaluation recipe are unchanged. 'Unchanged matching' means the algorithm and targets, not identical assignments after model outputs change.

Text masks are a single expression mask expanded over 256 candidates; Query masks are candidate specific. Requiring own Query support prevents treating shared Text support alone as candidate identity. Neither support overlap nor a query index proves physical instance identity.

Provenance is verified from bound code, saved identities and execution assertions. The reviewer did not reload the remote ScanNet annotation/point files and therefore does not claim an independent end-to-end reconstruction of each dataset label.

Evidence:

- Dataset annotation loading and object_id mapping: [C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py:585-649](<C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py:585>).
- Annotation-member masks, member AABB and independent native box jitter: [C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py:1086-1119](<C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py:1086>).
- Fixed 50000-point representation, annotation segments/objects, bounding boxes: [C:/Users/gb/.codex/tmp/pv_ground_source_20260905/src/visual_data_handlers.py:84-259](<C:/Users/gb/.codex/tmp/pv_ground_source_20260905/src/visual_data_handlers.py:84>).
- Capture member reference while preserving original native return: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:168-181](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:168>).
- Native matching, G and matched DFL use original batch; only auxiliary_roots is mode dependent: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:370-381](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:370>).
- Own/fused support, root-coordinate qualification, match exclusion and extra losses: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/query_supported_geometry.py:31-65](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/query_supported_geometry.py:31>).
- Shared Text mask versus candidate-specific Query mask construction: [C:/Users/gb/.codex/tmp/pvground_geometry_readback_20261004/revision2/source_preview/PV-Ground/models/pv_ground.py:532-556](<C:/Users/gb/.codex/tmp/pvground_geometry_readback_20261004/revision2/source_preview/PV-Ground/models/pv_ground.py:532>).

## B. Metric denominators and strict thresholds — PASS

Reported Acc@0.25 and Acc@0.50 are 100 times the number of expressions with selected-box IoU strictly greater than the threshold, divided by the actual stage row count: 9508 formal or 6887 module holdout. Equality is a miss. No model-max, model-mean or prediction-relative accuracy normalization is used.

The native evaluator is called with only_root=True, last_, bbs, thresholds [0.25,0.5], filter_non_gt_boxes=False; independently recorded selected-box hit counts must equal its bbs counters. The evaluator also computes native bbf metrics, but those are not used to select the reported result.

Mask support uses sigmoid(logit)>0.5 and point-count intersection/union. Auxiliary eligibility requires both Query and fused mask IoU>0.5 but box IoU<=0.5. The extra geometry term averages candidates within each expression, then averages all batch expressions including empty rows; its denominator is not a prediction score statistic.

Every selected final/coarse-box CPU threshold decision agrees with the recorded GPU IoU decisions across all 56072 reviewed evaluation rows. Full-formal maximum CPU/GPU IoU differences are 2.8856048953640467e-6 (control) and 2.8749439336395177e-6 (member).

Mask mIoU and full-256 coverage match saved scalar values. The full formal rows contain no raw point masks, all-256 box arrays or semantic-score vectors, so those scalar recounts do not independently rederive pixel/point membership, candidate ranking or oracle coverage.

Evidence:

- Native evaluator configuration and full evaluation loop: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:283-352](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:283>).
- AABB IoU computation, strict thresholds and fixed row denominator: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/geometry_result_metrics.py:5-77](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/geometry_result_metrics.py:5>).
- Native bbs ranking and strict IoU>t counters: [C:/Users/gb/.codex/tmp/pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py:281-303](<C:/Users/gb/.codex/tmp/pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py:281>).
- Point-mask fusion and intersection/union evaluation: [C:/Users/gb/.codex/tmp/pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py:594-653](<C:/Users/gb/.codex/tmp/pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py:594>).
- Point-count mask IoU: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/query_supported_geometry.py:6-12](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/query_supported_geometry.py:6>).
- Qualification and per-expression/batch loss normalization: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/query_supported_geometry.py:40-65](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/query_supported_geometry.py:40>).

## C. Artifact existence, execution and numerical claims — PASS

All 153 packet files exist and match actual bytes/SHA-256. All 51 current closed-intake files match their collection identities. Stage-row SHA values, training-log SHA values, specification SHA values, source-import SHA values and terminal restore/fit identities are consistent.

Both preflight, train and formal processes, plus both phase controllers, have exit code 0. Each final log record equals the corresponding JSON receipt. The controller records completion at 2026-10-05T21:49:56.894938+08:00, and the closed observer records controller_alive=false.

All seven evaluation files were parsed without dropping rows: parent formal 9508, and each new arm initial 6887, terminal 6887 and formal 9508. Formal IDs are exactly 0..9507. Pairing validates row_id, scan_id, target_id, root_box and point_sha256 throughout.

Independent recount reproduces all SUMMARY stage statistics, raw receipt metrics, transition counts, formal volume quartiles, training summaries, metric-best identity and target checks. CSV/RESULTS table values agree. Member versus control at 0.50: 9 repairs, 18 damages, net -9; member versus parent: 20 repairs, 27 damages, net -7. Control versus parent: 18 repairs, 16 damages, net +2.

Formal restoration is supported by the executed strict model restore and optimizer key/moment/step/group comparison, bound to each terminal hash. This reviewer did not download or open the remote weights; the terminal identities remain execution-receipt evidence, not a new reviewer-side weight restoration.

Historical PREPARED/pending records are not terminal execution reports. Current weight_retention.json, CLOSED_RESOURCES.json and terminal_publication.json are absent; their prepared tools are reviewed separately as unexecuted.

Evidence:

- Artifact paths, lengths and hashes verified against actual bytes: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/TERMINAL_AUDIT_PACKET.json:1](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/TERMINAL_AUDIT_PACKET.json:1>).
- 51 closed downloaded text/source/result artifacts: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/INTAKE.json:1](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/INTAKE.json:1>).
- All four train/formal children closed successfully and parents preserved: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/fit_status.json:2-23](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/fit_status.json:2>).
- Observer confirms controller is no longer alive: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/fit_wait.json:8-34](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/fit_wait.json:8>).
- Actual formal completion record: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/formal.log:25](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/formal.log:25>).
- Actual formal completion record: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/formal.log:25](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/formal.log:25>).
- Actual fit completion record: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/train.log:94](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/train.log:94>).
- Actual fit completion record: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/train.log:94](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/train.log:94>).
- Strict state/optimizer restoration and terminal identity: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/formal_restore.json:2-9](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/formal_restore.json:2>).
- Strict state/optimizer restoration and terminal identity: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/formal_restore.json:2-9](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/formal_restore.json:2>).
- Reported parent/control/member numeric table: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/analysis/FORMAL_METRICS.csv:2-4](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/analysis/FORMAL_METRICS.csv:2>).
- Reported metrics and transition claims: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/analysis/RESULTS.md:7-19](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/analysis/RESULTS.md:7>).
- Executed formal checkpoint reconstruction and step verification: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:232-247](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:232>).

## D. Actual called code and inactive diagnostics — WARN

The closed controller executes run_geometry_fit.py with the corresponding arm specification, sequentially in preflight/train/formal modes. Launch source checks bind the deployed files; complete/run_geometry_fit.py is byte-identical to the audited runner. imports.json binds the actually imported dataset, model, losses, main_utils and DataProcessor source copies.

The actual model call is observed through hooks: refiner, readback, then one native semantic head. The active training loop calls native criterion, semantic G correction, matched boundary DFL and query_supported_geometry_loss. Current full evaluation rows all record one native head call and zero diagnostic head replays. There is no synthetic row/result generator in this path.

The current runner passes only native points/voxels/text/superpoints/detector predictions to the model. pre_jitter_root_box, GT boxes, GT masks and qualification are outside the forward input dictionary. Native bbs uses the existing language-token maps; no new spatial-GT scoring, candidate pruning or ranking head is added.

Some bundled routines are intentionally inactive for this comparison: native_final_quality_loss/run_final_quality_fit are not called, and repeated_forward_differences is not called by this runner. Their existence and old draft docstrings are not evidence that their diagnostics, quality fit, or a cross-forward equality test ran. This is a nonblocking call-scope warning.

The runtime evaluator is loaded explicitly from runtime/PV-Ground/src/grounding_evaluator.py. Its inspected local snapshot has SHA-256 39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677, equal to the source-port evaluator copy, but imports.json records only five other modules and the current runner does not separately hash the runtime evaluator file at load time. This limits exact per-run evaluator-byte attestation; selected-box scoring/threshold totals were independently rederived from every saved row.

Evidence:

- Actual sequential subprocess commands: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/controller.py:36-58](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/controller.py:36>).
- Remote/local preflight and executable source-byte agreement: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/launch_geometry_fit_authorized.py:44-49](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/launch_geometry_fit_authorized.py:44>).
- Imported path and SHA bindings: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/imports.json:2-14](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/imports.json:2>).
- Active imports: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:103-116](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:103>).
- Forward input whitelist excludes spatial GT/auxiliary references: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:252-273](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:252>).
- Active observed forward and training losses: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:363-405](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:363>).
- Actual complete-forward call order and same-forward identity checks: [C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/readback_preflight_checks.py:19-68](<C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/readback_preflight_checks.py:19>).
- Defined independent-full-forward diagnostic is not invoked here: [C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/readback_preflight_checks.py:71-87](<C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/readback_preflight_checks.py:71>).
- Bundled quality loss exists but is inactive in this comparison: [C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/native_final_quality.py:41-53](<C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/native_final_quality.py:41>).
- Existing language-map bbs score, no added spatial-GT rank: [C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/native_root_bbs.py:4-9](<C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/native_root_bbs.py:4>).
- Explicit runtime evaluator import without a separate per-run evaluator-file SHA receipt: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:224-228](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:224>).
- Stored source-port evaluator identity matches the inspected local runtime snapshot; it is not a new attestation of the executing runtime file: [C:/Users/gb/.codex/tmp/pvground_geometry_readback_20261004/revision2/complete_preflight/source_port.json:1](<C:/Users/gb/.codex/tmp/pvground_geometry_readback_20261004/revision2/complete_preflight/source_port.json:1>).

## E. Experimental scope, initialization and claim ceiling — WARN

The two current specifications differ only in output root and auxiliary_target_mode. They start from the same hash-bound protected geometry parent, official PV checkpoint and original G state, with identical seed 2027 and fresh AdamW optimizers. The preflight's two memory-only updates do not initialize either formal fit.

Each arm has exactly 3723 sequential update records: 3722 batches of eight and one of two, covering the same 29778 unique fit IDs exactly once in the same order. Those IDs equal the raw split protocol; the 6887 holdout IDs are disjoint and ordered exactly as that protocol. Source/runtime assertions check physical-space separation for the local fit/holdout split.

Only the existing 456102-parameter, 10-state geometry head is trainable. Parent model parameters, buffers, masks, semantics and the zero-output R adapter are frozen/eval. End-of-fit assertions compare every non-head state exactly with initialization; current full evaluation has no R training.

The parent geometry lineage already contains 7446 updates (3723 boundary-head fit plus 3723 prior query-supported continuation); these new terminals have 11169 cumulative geometry updates. Original G and official PV pretraining/adaptation precede that history. A fresh optimizer does not mean training from scratch.

The 9508 formal expressions cover 141 scans and are development validation used for selection. Each 6887 module holdout covers 106 scans; the split protocol explicitly records previous pretraining exposure. There is one seed, one fit per current arm and one dataset. No independent test-set, multi-seed stability, significance, Nr3D/Sr3D generalization or complete three-module novelty result is established.

Actual cross-run differences disprove any blanket bitwise identity claim. Across initial arms, 6885 coarse boxes differ and one saved mask IoU differs (row 16804, JSONL line 3439: 0.950276255607605 versus 0.8911917209625244), despite equal selected queries/hit counts. Across formal arms, 9503 coarse boxes and three selected queries differ. The three query-change rows have box/mask IoU zero in both arms. Same-forward snapshots and cached-head replays establish only their stated local identities.

The extra-loss direct-output gradient is zero outside qualified candidates in the exercised real batch, but the shared head's parameters can change other candidates on later forwards; native matching is recomputed. Qualification and target coordinates changed together. The observed -9 comparison is not an isolated loss-label-only causal estimate, and the control's +2 hits is a measured selection result rather than evidence of a stable improvement.

Evidence:

- Same official/G/geometry parents and fixed budget/hyperparameters: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/control_spec.json:2-55](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/control_spec.json:2>).
- Only scientific intervention: member_gt mode: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/member_target_spec.json:53](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/member_target_spec.json:53>).
- Parent reconstruction, freeze/unfreeze and 456102/10-state assertions: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:118-148](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:118>).
- Fresh AdamW construction: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:231](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:231>).
- Two-update memory-only preflight, not a saved initialization: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:439-474](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:439>).
- One exact full fit and frozen-state assertion: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:481-533](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:481>).
- Independent iteration over all 3723 training records: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/train.jsonl:1-3723](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/train.jsonl:1>).
- Independent iteration over all 3723 training records: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/train.jsonl:1-3723](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/train.jsonl:1>).
- Raw split IDs, physical-space counts and pretrained-seen holdout declaration: [C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/split_protocol.json:1](<C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/split_protocol.json:1>).
- Parent's saved cumulative geometry history: [C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/run_geometry_fit.py:464-473](<C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/run_geometry_fit.py:464>).
- Observed cross-run mask difference: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/initial/rows.jsonl:3439](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/initial/rows.jsonl:3439>).
- Observed cross-run mask difference: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/initial/rows.jsonl:3439](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/initial/rows.jsonl:3439>).
- First of three formal selected-query differences; also lines 2239 and 2284: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/formal/rows.jsonl:959](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/control/formal/rows.jsonl:959>).
- First of three formal selected-query differences; also lines 2239 and 2284: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/formal/rows.jsonl:959](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/complete/member_target/formal/rows.jsonl:959>).
- Qualified-output gradient check and cached-upstream replay limits: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:384-419](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:384>).

## F. Evaluation classification — PASS

Classification: **real_gt**.

The closed 9508/6887 evaluations are real_gt: predictions are scored against dataset-provided object masks and annotation-member boxes under the native evaluation contract.

The historical augmented fit64 jitter counterfactual is an annotation-derived real_gt diagnostic, explicitly not an accuracy result, optimizer run or simulation. Its stored mask support is prediction-derived eligibility measured against real GT, not model-generated replacement ground truth.

Independent NPZ/JSON recount reproduces 3479 supported unmatched candidates, native-qualified 1090, member-qualified 861, removed 395 and added 166; in the fixed native cohort outside-range candidates change 201 to 87. These are correlated training-candidate diagnostics, not additional formal validation rows or proof of the later nine-hit cause.

No synthetic_proxy, self_supervised_proxy, simulation_only or human_eval performance claim is supported or used here.

Evidence:

- Actual ScanRefer val loader: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:198-203](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:198>).
- Stored native root GT and same-query Box/Mask evaluation: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:309-351](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:309>).
- Annotation-derived pre-jitter root capture: [C:/Users/gb/.codex/tmp/pvground_target_jitter_20261005/run_label_probe.py:68-82](<C:/Users/gb/.codex/tmp/pvground_target_jitter_20261005/run_label_probe.py:68>).
- Closed CPU64 diagnostic; zero optimizer/model steps: [C:/Users/gb/.codex/tmp/pvground_target_jitter_20261005/complete/receipt.json:2-17](<C:/Users/gb/.codex/tmp/pvground_target_jitter_20261005/complete/receipt.json:2>).
- Native matched slots and GT-based mask intersections/unions: [C:/Users/gb/.codex/tmp/pvground_query_geometry_cohort_20261005/complete/run_cohort_probe.py:294-354](<C:/Users/gb/.codex/tmp/pvground_query_geometry_cohort_20261005/complete/run_cohort_probe.py:294>).
- All 64 rows paired to target-jitter labels: [C:/Users/gb/.codex/tmp/pvground_query_geometry_cohort_20261005/complete/rows.jsonl:1-64](<C:/Users/gb/.codex/tmp/pvground_query_geometry_cohort_20261005/complete/rows.jsonl:1>).

## Nonblocking qualifications

These are claim and reproducibility limits, not evidence of an invalid primary result.

- **W1 — Single-seed development selection:** One seed and development validation cannot establish stable/statistically significant gains or out-of-dataset generalization; 6887 is pretrained-seen module holdout.
- **W2 — Raw evidence replay boundary:** Selected final/coarse boxes can be independently rescored against saved GT. Formal masks/full-candidate coverage can only be recounted from saved scalars, and remote raw dataset/weight contents were not independently replayed.
- **W3 — Cross-forward reproducibility:** Initial and formal raw rows show numerical/selection drift. Cached-head and same-forward checks do not establish cross-CUDA complete-forward bitwise equality.
- **W4 — Intervention interpretation:** Auxiliary eligibility and targets change together; native matching protocol is unchanged but assignments may change. Local direct-gradient exclusion does not freeze other outputs of the shared trainable head.
- **W5 — Per-run evaluator source binding:** The explicitly loaded runtime evaluator lacks its own per-run SHA record. A local runtime snapshot matches the stored source-port copy, and the full selected-box calculation was independently recounted, but exact remote evaluator-byte identity is not newly attested.

## Prepared strict-best retention and publication

**Source review only; neither stage has executed.** No current retention, resource-closure or publication receipt exists at audit time.

Maximize strict native last/bbs Acc@0.50 count. On an equal primary count, keep the protected parent before considering Acc@0.25. If both new arms tie above the parent, compare Acc@0.25; a complete new-arm tie follows the table's control-first order.

Retain the current control's full ten-state geometry delta and its optimizer; remove only the closed parent geometry terminal and member-target geometry terminal. Preserve official PV, original G and V99 paths. This describes the reviewed plan, not an action performed by this auditor.

No model meets the same-model dual target >=5615 at .25 and >=4754 at .50. The measured best remains 243 hits below 4754.

analyze_closed_formal.py orders candidates by (rec_hits50, is_protected_parent, rec_hits25). Independent row recount yields control as winner. The remote retention source also rejects a non-parent winner unless its primary count strictly exceeds the parent.

The wrapper requires actual closed observer state, a received terminal review, PASS/WARN with empty blocking_findings, a reviewed retention source hash and no conflict between the dual-threshold target and metric-best winner.

Before deletion, the retention source checks exact root/parent paths, terminal SHA agreement with fit/restore receipts, formal count/hits, head shape/key/parameter identity, update history, parent identities and all ten optimizer steps. Each retained delta stores every geometry state, so reconstruction needs official PV plus original G plus the winner, not the older geometry ancestor.

The publisher requires the completed audit, retention receipt and resource verification; checks parent publication/clean repository heads and identical handoff bytes; carries actual metrics and scope qualifications into the appended section; verifies copied bytes and push destination. check_closed_resources.py hashes official/G/winner weights and verifies removed paths and idle GPU before publication.

No weight_retention.json, CLOSED_RESOURCES.json or terminal_publication.json exists for this comparison at audit time. No deletion, resource postcondition, publication commit, remote byte identity or successful push is attested by this report.

Evidence:

- Strict primary metric and protected-parent tie priority: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/analyze_closed_formal.py:88-103](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/analyze_closed_formal.py:88>).
- Closed/audit/review prerequisites and decision forwarding: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/retain_metric_best_authorized.py:11-26](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/retain_metric_best_authorized.py:11>).
- Fixed target paths and header/identity checks: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/retain_metric_best.py:12-49](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/retain_metric_best.py:12>).
- Strict improvement over parent and exact nonwinner deletion set: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/retain_metric_best.py:50-67](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/retain_metric_best.py:50>).
- All ten selected head states and full optimizer saved: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:487-500](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/run_geometry_fit.py:487>).
- Requires audit, retention and resource receipts: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/publish_terminal.py:16-41](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/publish_terminal.py:16>).
- Prepared result interpretation and explicit scope limitations: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/publish_terminal.py:64-87](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/publish_terminal.py:64>).
- Prepared exact-byte publication and repository destination checks: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/publish_terminal.py:93-185](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/publish_terminal.py:93>).
- Planned protected-weight hash and removed-path checks: [C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/check_closed_resources.py:20-36](<C:/Users/gb/.codex/tmp/pvground_auxiliary_target_20261005/check_closed_resources.py:20>).

## Claim checks

| Claim | Assessment |
|---|---|
| Closed parent/control/member strict formal hits are 4509/4511/4502 of 9508. | supported |
| Member-target improves against the current same-budget native-target control. | unsupported; the observed difference is -9 hits (-0.09465713083718973 percentage points). |
| Control is the metric-best candidate under the specified strict Acc@0.50 promotion rule. | supported conditionally for retention; actual retention not executed |
| The control's two-hit advantage over the parent is a stable causal improvement. | unsupported |
| Native/member coordinates change auxiliary qualification and localization targets, leaving native target protocol unchanged. | supported |
| Full point-mask/full-candidate replay, physical identity proof or cross-CUDA bitwise equality was established. | unsupported and not claimed by the qualified current RESULTS scope |

## Follow-through

- Keep the current development/single-seed, scalar-recount and numerical-drift qualifications attached to the result.
- Use the negative member-versus-control result as the observed outcome; do not turn its within-model coarse-to-final +7 hits into an independent-baseline gain.
- If the already authorized workflow proceeds, execute the reviewed retention/header checks and record actual resource/publication receipts before marking those stages complete. This audit itself performs none of those actions.

The executor must preserve the actual native reviewer call, response and tool trace. This report does not assert cross-family independence or attest the requested backend model identity.

Packet SHA-256: 866bbf8ec17f769b0b84a9ace75d7a3b0ad935b60926d2836f7f5cc1f88e9866.

