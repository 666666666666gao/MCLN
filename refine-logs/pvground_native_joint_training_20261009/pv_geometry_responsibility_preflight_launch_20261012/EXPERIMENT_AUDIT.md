# Experiment audit — retained superpoint boundary pilot

**Verdict: WARN. Blocking findings: 0.** All reported counts are independently reproducible; the warnings limit interpretation. Review independence is **same-family**, acceptance is **provisional**, and actual runtime model/effort attribution is **UNATTESTED**. The skill requests gpt-6-astra/max; that request is not an attestation.

Primary execution scope: `REAL_SAVED_SUPERPOINT_BOUNDARY_TRAINING_PILOT`. Supplemental scope: `REAL_RETAINED_9508_SAME_FORWARD_GEOMETRY_CONTROL_COUNTS`. Generated 2026-10-12T15:14:05.901272+00:00.

Only E:/anaconda/python.exe and NumPy array arithmetic were used. No model load, neural forward, optimizer update, GPU, SSH, remote/active-job query, install, or private MEMORY access occurred. All audit outputs are in this review directory. The original analysis and responsibility scripts were read, not run.

## Verification and source provenance

- **A — GT provenance: PASS, with a physical-purity limitation.** The exact historical dataset-source SHA-256 `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d` matches `imports.json`. [joint_det_dataset.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/runtime_binding/dataset_source/src/joint_det_dataset.py:1099>) builds GT masks from the annotated object's point indices; it does not read model predictions or project these masks to majority-superpoint labels. [matched_mask_objective.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/matched_mask_objective.py:6>) copies batch masks/boxes into targets, and [paired_span_loop.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/paired_span_loop.py:103>) saves those targets alongside predicted logits. The later majority-superpoint target at matched_mask_objective.py:41 belongs to a separate loss and is not the saved point GT. Raw annotation/scan files were not reopened; this confirms the retained source/data path, not the physical correctness of every annotation.
- **B — Score normalization: PASS.** [analyze_saved_superpoint_boundary_pilot.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/analyze_saved_superpoint_boundary_pilot.py:58>) uses point intersection/union and strict >0.5. The supplement uses raw geometric intersection/union against a row's stored GT box. No prediction-maximum normalization or new score scaling was applied.
- **C — Existence, hashes and numbers: PASS.** All 23 audited input files are freshly hashed in the JSON report with resolved absolute paths and raw 64-character SHA-256 values. Eight NPZ byte lengths, NPZ hashes and raw-point-byte hashes match their receipt. Both complete output JSONs reproduce exactly, including all **381 primary + 39 supplemental numerical leaves**. The final re-read confirms no audited input changed.
- **D — Code path: PASS.** Capture is called from preflight step at [paired_span_loop.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/paired_span_loop.py:126>). The analyses use top-level computations, and their reported data-dependent outputs reproduce independently. No new claim is made that the separate torch gradient/protection probe or full training path was rerun.
- **E — Scope: WARN.** These are the same eight distinct row IDs/scenes in one retained augmented training batch, step 1. The responsibility checker loops steps 1 and 2 ([check_saved_responsibility_cpu.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/check_saved_responsibility_cpu.py:35>)), but this pilot neither reads step 2 nor counts it as new samples. The 9508-row historical supplement is a separate scope and cannot supply missing point-level incidence.
- **F — Evaluation type: real_gt.** Both scopes use retained dataset GT, with the diagnostic and historical-comparison limits below.

The independent checks pass **770 primary + 51 supplemental checks**. All 13 saved observed-query logit pairs reconstruct their stored masks exactly. Each selected query is the unique maximum of saved scores. All eight selected Mask boxes reconstruct exactly from their foreground point extrema; all 1,013 valid candidate reference extents also reconstruct exactly, and all 2,048 saved validity flags agree with raw support validity. Logits were saved for observed queries only; this is not an all-256-logit validation.

## Primary pilot: what the numbers support

Seven of eight selected masks have point IoU > 0.5. Among those seven, five include false-positive points from provided-GT-background superpoints, four extend outward beyond the GT member-point extent, and seven outward faces have background-superpoint extrema. Mixed-superpoint counts are zero.

| Training row | Query | Point mask IoU | False positives | Max outward vs GT members (m) | GT-member/training-box max face gap (m) |
|---|---:|---:|---:|---:|---:|
| 27845 | 87 | 0.507936508 | 54 | 0.004329324 | 0.032986999 |
| 23527 | 243 | 0.957001880 | 34 | 0.000000000 | 0.061347842 |
| 27285 | 120 | 0.000000000 | 578 | 2.272642374 | 0.025558680 |
| 9781 | 88 | 0.908629442 | 18 | 0.032246351 | 0.038985729 |
| 3314 | 106 | 0.989247312 | 0 | 0.000000000 | 0.033110082 |
| 8523 | 22 | 1.000000000 | 0 | 0.000000000 | 0.035257220 |
| 15720 | 3 | 0.971264368 | 10 | 0.028165936 | 0.141958952 |
| 32718 | 78 | 0.783926219 | 139 | 0.072776079 | 0.044553399 |

The independent set-based check finds **zero mixed-label superpoints among all 12,423 present row-superpoint instances**, not merely among false positives. Therefore the pilot has no mixed-label cases with which to test that mechanism. “Pure background” means no provided root-GT member in that superpoint; it does not mean a physically verified pure scene region. Keep that qualification whenever reusing these counts.

Each of the 48 selected faces has exactly one extremal point; there are no within-face ties or cross-category ties in this cache. The bad-mask row 27285 uses point 47338 at two different faces: six point-face incidences represent five unique points. The seven outward faces among good masks have seven distinct witnesses. The code correctly counts a face once per category, includes all exact extremal ties, and does not incorrectly assert disjoint category-presence counts in general ([analyze_saved_superpoint_boundary_pilot.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/analyze_saved_superpoint_boundary_pilot.py:47>), [analyze_saved_superpoint_boundary_pilot.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/analyze_saved_superpoint_boundary_pilot.py:84>)).

Training boxes and GT-member extents are different quantities in all eight rows. The original dataset code explicitly jitters training boxes ([joint_det_dataset.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/runtime_binding/dataset_source/src/joint_det_dataset.py:1105>)). The pilot correctly reports both references instead of substituting one for the other. Outward member extent is not equivalent to training-box residual or formal box error.

The geometry is the saved A-corrected **Mask endpoint**, before final B. The capture saves parent Mask boxes ([paired_span_loop.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/paired_span_loop.py:109>)); final B boxes are not present. The responsibility script explicitly acknowledges this ([check_saved_responsibility_cpu.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/check_saved_responsibility_cpu.py:63>)). These examples do not establish a causal explanation of final B failures, a validation boundary-failure rate, improved model accuracy, or three independently useful contributions.

## Separate supplement: retained 9508-row geometry controls

[summarize_retained_geometry_controls.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/summarize_retained_geometry_controls.py:14>) recounts existing outputs from [rows.jsonl](<D:/Program Files/UserCache/gb/codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/formal/rows.jsonl>). I independently recomputed 47,540 selected-box IoUs from the saved boxes and each row's root GT, checked exact fixed-half box construction, and recomputed paired transitions using row-ID set differences. Both float32 and float64 give every saved strict 0.25/0.5 threshold decision. The source uses a single selected Query and shared parent forward for the controls/arms ([paired_span_loop.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/paired_span_loop.py:290>) and [paired_span_loop.py](<D:/Program Files/UserCache/gb/codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/paired_span_loop.py:307>)). No new neural accuracy result was generated.

| Existing output | Hits >0.25 | Hits >0.5 |
|---|---:|---:|
| native | 5615 | 4495 |
| mask | 5606 | 4881 |
| fixed_half | 5675 | 4857 |
| whole_support | 5676 | 4921 |
| extremal_support | 5677 | 4920 |

| Before → extremal support | >0.25 repairs / breaks / net | >0.5 repairs / breaks / net |
|---|---:|---:|
| native | 158 / 96 / +62 | 686 / 261 / +425 |
| mask | 87 / 16 / +71 | 103 / 64 / +39 |
| fixed_half | 33 / 31 / +2 | 156 / 93 / +63 |
| whole_support | 1 / 0 / +1 | 3 / 4 / -1 |

The retained learned mixture has **+2 / +63 hits** relative to fixed-half geometry at 0.25 / 0.5. Extremal versus whole support is **+1 / −1**, so there is no strict overall advantage for extremal features. These are existing same-row comparisons; preserve historical model selection and do not infer multi-seed significance or an independent contribution from the feature choice.

Continuous saved IoUs are not bitwise identical to all NumPy calculations: maximum float32 recomputation discrepancy is 6.556510925292969e-7. Historical Mask IoU exceeds one in 112 rows, with maximum 1.0000050067901611, due to float32 coordinate/volume rounding; float64 reconstruction is bounded by one and leaves all counts/transitions unchanged. No scores were clipped, normalized or overwritten.

## Actionable use and audit trace

Retain the eight-row/augmented-training/step-1 qualifier, provided-GT-relative purity, Mask-before-B qualifier, member-extent versus training-box distinction, and historical-output qualifier on the supplement. Do not promote these artifacts into a validation incidence, causal diagnosis, new neural performance claim, strict extremal-feature advantage, or proof of three effective contributions. No source correction or new acceptance gate is required for the reported numbers.

Two exploratory verifier assumptions were corrected transparently. An extra all-candidate check initially equated invalid reference sizes with raw native sizes; the 399 differences are confined to invalid references whose saved sizes empirically equal max(native size, 1e-6). All selected/valid references were exact throughout. An extra strict float32 IoU ≤1 check encountered historical rounding and was replaced by finite-float32 plus exact float64 threshold verification. The original scripts/results and diagnostics remain in this directory; no audited source/result was edited.

Machine report: [EXPERIMENT_AUDIT.json](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/superpoint_boundary_pilot_20261012/review/EXPERIMENT_AUDIT.json>). Full recomputation evidence: [recomputed_evidence.json](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/superpoint_boundary_pilot_20261012/review/recomputed_evidence.json>) and [supplement_recomputed_evidence.json](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/superpoint_boundary_pilot_20261012/review/supplement_recomputed_evidence.json>). Trace: [001-cached-audit.request.json](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/superpoint_boundary_pilot_20261012/review/trace/2026-10-12_run01/001-cached-audit.request.json>) and [001-cached-audit.response.md](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/superpoint_boundary_pilot_20261012/review/trace/2026-10-12_run01/001-cached-audit.response.md>).
