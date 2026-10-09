# Actual span preflight experiment audit

Date: 2026-10-09. Execution scope: ACTUAL_PREFLIGHT. Fresh reviewer context: true.

Overall verdict: **WARN — no blocking findings.** The sealed artifacts support this two-step engineering preflight. They do not establish new ScanRefer accuracy, an effective module, three effective contributions, statistical stability, or a completed research target.

Requested reviewer route: Codex / gpt-6-astra / max. Actual backend, model and effort are UNATTESTED because this agent has no exposed route attestation. Review independence is same-family; acceptance status is provisional. This new auditor is distinct from the historical source reviewer. Historical R3/R5 reports are source-only attribution, not execution evidence.

## Evidence boundary and closure

The auditor waited for the executor's sealed-intake message before opening any witness NPZ. All 41 collected files, totaling 22,794,801 bytes, match their sealed intake sizes and SHA-256 hashes. The 11 collected computational sources equal both the current canonical sources and pair_spec.json. The six previously collected actual imported Python modules match the current imports.json module paths and hashes. Stage-one snapshots remained unchanged.

The original controller and child both exited 0; preflight_wait.json records the controller no longer alive and contains the exact collected preflight_status.json. The native run finished at 18:25:16 CST, and intake was sealed at 18:39:10 CST. This audit performed no SSH, GPU access, Torch/project-module import, model replay, optimizer step, pickle load, package download, source change, weight action, deletion, promotion or training launch. Numeric checks used the already-cached offline native uv NumPy environment.

Binding SHA-256 values:

- Current pair_spec.json: ceaa2bbeeaa124f86aa46b091aa63a3c9f00587fd4a291f8f73a088d52c2527e.
- Current preflight_wait.json: 78514915f1d3b5e4abce837502d74e53ae795066ab83e8e92ae46b2d244cdea1.
- Actual preflight_complete/preflight.json: 629a478aad3f5536cd4451463e4ed1fa7f8be1e4fdcf9df6289b09e02ff76d83.
- Sealed INTAKE.json: e39b3eae7360f78df9dacedfa7b7ac1ee634085ab44c8db1fc3800415e0c88ff.
- Explicit retained parent terminal identity: d06adb8ed227d98d46afa8e3970dcb9fbe31ab2613c675883b80162f830f4cb6.

The parent decision, checkpoint inspection, prior CPU recount and audit hashes match FINAL_SPEC_RECEIPT.json and pair_spec.json. The declared parent is the completed selected_query terminal, with historical 5606/4881 hits. Those historical hits are used only to identify the parent; they are not span results. The earlier parent audit predates its later completed CPU inspection, so its historical “pending inspection” statement is not treated as the current state. Evidence: FINAL_SPEC_RECEIPT.json:3; pair_spec.json:57; preflight_wait.json:6; preflight_complete/preflight_status.json:2; parent postrun_results/RETAINED_PARENT_DECISION.json:2 and checkpoint_inspection.json:2.

## A. Ground-truth provenance — PASS

The actual loss targets come from dataset center_label, size_gts and gt_masks after box_label_mask filtering. Native ScanNet object membership and object boxes produce those labels; the training path applies the existing native geometric augmentation and up-to-5% box jitter. Model outputs are used to define Hungarian correspondence, not as the target box or target mask.

All 16 saved row-step cases have one actual GT box/mask, with nonempty nontrivial binary masks. Their box coordinates are consistent with raw GT-member extents under the native jitter bounds. Raw points, GT boxes and GT masks are identical for each row between the two steps. This checks the saved evidence and source chain; the auditor did not independently reload the original ScanNet annotation database.

Inference inputs contain points, text, SP IDs and detector outputs, with butd_gt/butd_cls disabled. GT enters matching, loss and witness capture after the shared parent/mixer forward. No GT box or GT mask is passed to support selection or mixing. The bbs score uses the existing linguistic token maps and semantic logits, without an IoU-based selector.

Evidence: run_span_pair.py:152, 164, 201, 237; matched_mask_objective.py:6, 20; paired_span_loop.py:126; imported src_joint_det_dataset.py:1086, 1268, 1325, 1354, 1387; native_root_bbs.py:4. Native proof: GT_COORDINATE_VERIFICATION.json.

## B. Loss normalization and geometric construction — PASS

The native Box loss is center L1 plus 0.5 times size L1, together with GIoU. Each term is divided by the count of actual targets, eight here, which equals the original matched-target count in these measured cases. The span objective preserves coefficients 10 and 2 and divides by seven original outputs. It adds no score loss or expanded-positive query loss. No reported accuracy is normalized by prediction statistics.

For both neutral first-step arms, independent float64 reconstruction from the saved raw matched boxes and GT gives L1 0.16207887046039104 and GIoU loss 0.39366257770882723. Errors relative to native float32 receipts are 1.862645149230957e-9 and 1.134075057751538e-7. Both steps' recorded total losses reconcile with (10 L1 + 2 GIoU)/7.

All 4096 candidate-row-step cases were reconstructed from the actual 50,000-point clouds and SP IDs. Counts, origins, spans, 32-bin histograms, means, second moments, minima and maxima match exactly. All256 reference validity, centers, sizes and the whole/extremal source fractions also match exactly, including dtype conversion to native float32 coordinates. The 23 recorded selected-plus-matched query cases have exactly the required query union, and their raw text/query logits and scalar alpha reproduce the stored foreground decisions. The remaining candidates have full foreground/member evidence, not full raw logits.

The whole arm pools all valid foreground SPs weighted by member counts. The extremal arm retains the SPs producing each axis minimum/maximum, comparing the same reduced float32 coordinates. Source equality preserves all tied SPs. There were no actual multi-SP extremal ties in this batch, so tie handling is a static contract rather than an empirically exercised case. There were 2087 empty references and zero nonempty degenerate references. Existing invalid-reference native priors and size flooring are retained; the span module adds no reference fallback, candidate pruning or ranking rule.

The zero output layer produces gate zero and exact same-forward Mask-reference boxes at step one. At step two the recorded gates remain in [0, 1], with maxima 3.111776095465757e-5 (whole) and 2.7826423320220783e-5 (extremal). The source uses the same convex coefficient for each axis's center and positive size. Native executed assertions check positive output sizes. Final head boxes/gates are not exported per candidate, so the auditor's independent array replay covers references and source fractions; final interpolation is supported by the executed gate summaries, assertions and source formula.

Evidence: imported models_losses.py:518, 831, 948; main_utils.py:268; matched_span_objective.py:5; extremal_span_mixer.py:38, 45, 51, 64, 72, 79, 103; mask_reference.py:10; paired_span_loop.py:68. Native proof: NUMERIC_VERIFICATION.json.

## C. Receipt existence, identity and restoration — PASS

The actual preflight records two optimizer steps per separate 29,793-parameter, 14-state span head over one shared frozen 1314-state parent. The source checks nonaliasing head parameters, equal initial values, empty distinct AdamW optimizers, no parent gradients, no other-head gradients and no other-head mutation during each update. All Mask and score objects remain those of the same parent forward. All parent state tensors remain exact to the initial snapshot.

For each arm, the executed native restoration serializes only to memory, rebuilds the complete CPU model, compares all 1328 states to the expected 1314 parent plus 14 trained-head states, and restores the Adam parameter group, 14 moment/step states and CPU/CUDA/NumPy/Python RNG state. Mode and the multi-part parent identity are checked. The restored wrapper is actually called with the native input dictionary on GPU; the observed call order is native semantic head, completed parent, span mixer, once each. Its updated outputs are equal to the original trained head evaluated on the same captured parent cache.

This is not a cold-GPU reconstruction of the original parent, nor a claim of bitwise identity across separate full parent forwards. The record explicitly marks cold GPU reconstruction false. No checkpoint files were written by this preflight; no weight files are in the intake. A later train phase constructs fresh heads/optimizers in a new process; only formal mode restores trained terminals. Nothing in this audit starts that phase.

Evidence: span_refinement_model.py:24, 41, 47, 55; run_span_pair.py:124, 134; paired_span_loop.py:38, 61, 126, 212, 361; span_controller.py:48; preflight_complete/load.json:2; preflight_complete/preflight.json:8; INTAKE.json:212.

## D. Actually called paths and gradients — WARN for unused legacy metric only

The native objective, parent wrapper and both span heads are actually exercised by the closed preflight path; these conclusions are not taken from source-only R3/R5 reports. Direct loss gradients on final center/size outputs are zero for every unmatched query in all four arm-step cases, and nonzero on the eight matched queries.

At step one, output weight/bias norms are nonzero and every recorded upstream norm is zero, as expected from zero output initialization. At step two the raw per-parameter norms independently establish positivity for each of the four recorded upstream groups. Each value below is a sum of its parameter L2 norms, not a joint L2 norm:

| Step-two group | Whole support | Extremal support |
|---|---:|---:|
| Query projection | 2.936540553832856e-6 | 2.4436029804064674e-6 |
| Support projection | 7.612340269247397e-8 | 6.551519504682801e-8 |
| Face encoder | 5.513398306788986e-7 | 4.841217986495394e-7 |
| Axis decoder | 8.507443794769642e-6 | 7.532824270128913e-6 |

This group claim comes from all 14 saved per-parameter norms and their independently checked sums, not from the aggregate upstream assertion alone. Direct unmatched-output gradient zero does not imply those outputs stay numerically unchanged after shared-parameter updates.

The formal evaluator is imported and hash-bound but not executed for accuracy in preflight. The legacy calculate_diou_3d helper remains defined with its loss call commented out; no DIoU or full-evaluation result is attributed to it.

Evidence: paired_span_loop.py:126, 138, 151, 165, 246, 278, 369; preflight_complete/preflight.json:90, 142, 413, 465; imported models_losses.py:103, 546.

## E. Scope — WARN, nonblocking

This is one seed (2027), one physical batch of eight distinct training rows from eight scenes, reused for two steps. It is not 16 independent samples. All measured matching cases have one GT. Multi-GT behavior, nonempty degenerate references and tied extremal SPs are not empirically covered.

Frozen parent state is not frozen output across independent forwards: all eight selected winners and six matched queries differ between the two steps despite identical raw points and GT. The native model uses Gumbel sampling in its forward. Each arm pair still uses the same parent cache within its step. No separate-forward numerical identity is claimed.

Raw logits cover 23 selected/matched query cases; foreground/member arrays cover all 4096 candidates. Loss gradients and restoration are verified from hash-bound native assertions/receipts; the auditor did not rerun Torch or reload a model. Complete updated per-candidate head boxes and raw gradient tensors were not exported. These are limits of the engineering evidence, not blockers for this narrowly declared preflight.

No 9508-row validation, training-effectiveness claim, novelty claim, multi-seed stability, three-module claim, Nr3D/Sr3D claim, or automatic promotion is supported. The preflight explicitly records accuracy_result=false. No further code change or model run is required by this audit.

Evidence: pair_spec.json:103; paired_span_loop.py:369; preflight_complete/preflight.json:2, 247, 570; imported models_pv_ground.py:313, 610; NUMERIC_VERIFICATION.json.

## F. Evaluation type — PASS

Classification: real_gt for the actual supervised native Box objective, with engineering diagnostics for support provenance, gradients, frozen-state checks and serialization. These diagnostics are not a benchmark accuracy evaluation or synthetic-proxy accuracy.

## Artifacts and trace

Machine report: EXPERIMENT_AUDIT.json, including zero blocking findings and the complete reviewed_files path/hash bindings. The full review is preserved as EXPERIMENT_AUDIT.raw.md and in the original trace directory:

.aris/traces/experiment-audit/2026-10-09_actual_preflight_run01/

The trace preserves the exact request, sealed-intake follow-up, current source/receipt snapshots, all sealed intake files, imported source snapshots, native tool receipts, verifier source, independent numeric proofs and full response. Some initial display reads were clipped; the complete source snapshots and missing targeted windows were reviewed. No auditor command failed.
