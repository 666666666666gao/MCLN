# Fixed-endpoint coordinate-projection audit

Date: 2026-10-13. Verdict: **WARN**, with **0 blocking findings within the stated diagnostic scope**. This is a fresh-context Codex, same-family review with provisional acceptance. Requested reviewer: `gpt-6-astra`, effort `max`; actual reviewer model and effort: **UNATTESTED**. The available agent inventory identifies this agent and its status but exposes no model/effort attestation.

Scope: `RETAINED_SELECTED_QUERY_FIXED_ENDPOINT_GT_COORDINATE_PROJECTION_DIAGNOSTIC`.

The reported fixed-endpoint coordinate diagnostic is numerically supported. It is not evidence of a new method's accuracy, attainable REC gains, an IoU optimum, a whole-network capacity limit, or completion of the research goal. No source or result change is required for this bounded diagnostic. Preserve its existing claim restrictions and the warnings below.

## Direct evidence and deterministic checks

The eight requested inputs were read directly. Their byte hashes before and after verification agree. RESULT's three declared hashes match the archived rows, current mixer source and CSV. The C: paths and RESULT's resolved D: paths refer to the same files; this is a filesystem alias, not a different input. Full hashes are in `deterministic_verification.json`.

- Rows: exactly 9,508, ordered IDs 0–9,507; 141 scenes and 2,068 scene/target pairs. All 12 CSV fields are populated, all numeric CSV values are finite, and row/scene/target/selected-Query identity agrees with the source archive.
- All 1,189 saved NPZ batches were opened. Every saved GT and each of the five selected box variants equals its JSONL counterpart. Every selected Query is the unique maximum of the saved 256 scores; zero score ties were found. This preserves the historical selected Query and endpoints. It does not re-evaluate the current trainable model.
- All 1,246 files in the historical `complete_fit/INTAKE.json` match their recorded SHA256 and byte count. All 11 historical runner source hashes match the saved specification. The formal exit is 0, and the formal receipt binds the same rows SHA256.
- All 36 reported group statistics agree with an independent computation. Each CSV numeric field was compared on all 9,508 rows. Maximum error for shared minimax values and coupling gaps is `1.7763568394002505e-15` m; independent values and copied IoU/gate fields agree exactly.
- The selected-box reconstruction discrepancy is `6.46173880625156e-7` m, agreeing with RESULT. There are 129 axes with both endpoint faces identical, six selected zero-gate axes in rows 763 and 2982, and no selected one-gate axes.
- Float64 box-IoU recomputation reproduces every historical 0.25/0.5 threshold count: native 5615/4495, Mask 5606/4881, fixed-half 5675/4857, whole-support 5676/4921, and extremal-support 5677/4920. These remain historical scores.

## Independent mathematical verification

For one axis, let the Mask/native lower faces be `mL,nL`, upper faces `mU,nU`, and GT faces `tL,tU`. Because center and size share the same coefficient, the actual faces are `mL + g(nL-mL)` and `mU + g(nU-mU)`, with `0 <= g <= 1`. The objective is the maximum of the two absolute face-coordinate residuals.

The reviewed analyzer enumerates endpoints, individual residual zeros, and crossings of signed residuals (`analyze_fixed_endpoint_axis_coupling_20261013.py:33–51`). This is valid for the convex piecewise-linear objective. Its division masks for zero denominators omit non-isolated breakpoints; they do not invent a box or a fallback prediction.

The independent implementation did not import or execute that analyzer and did not reuse its breakpoint list. For a proposed error `e`, each nonconstant face supplies the closed gate interval solving `|a*g-b| <= e`; a constant face supplies the feasibility condition `|b| <= e`. Their intersection with `[0,1]` is nonempty exactly when that error is attainable. Sixty-four bisection steps were applied to all 28,524 axes, from zero to the error at `g=0`. The largest final bracket width is `1.7763568394002505e-15` m. Taking the maximum over the three independent axis minima gives the box minimax value.

For independent face coefficients, the nearest attainable coordinate is the median of the GT coordinate and the two endpoint coordinates. This derives the projection independently of the analyzer's clipping expression. Each face's interval is independent, so these six nearest points attain the minimax error. The common-gate feasible set is contained in the independent-face feasible set, establishing the error inequality.

Ordering is also proved, not assumed. Each upper coordinate is at least its corresponding lower coordinate plus `delta = min(Mask size, native size, GT size) > 0`. The median operator is monotone in its three inputs and commutes with translation; therefore projected upper minus projected lower is at least delta. All 28,524 widths satisfy this bound within `1e-14` numerical rounding; the smallest projected width is `9.999999974752427e-7` m. Arbitrary independent face choices can invert faces, as the old face model notes, but these nearest-GT projections remain ordered.

## Recomputed statistics

All counts use strict inequalities exactly as the analyzer does. Groups overlap and must not be added together. There is no confidence interval, significance test, or repeated-seed robustness claim.

| Saved-row group | Rows | Shared median error (m) | Independent median error (m) | Gap >1 cm | Gap >2 cm | Shared >2 cm, independent <=2 cm |
|---|---:|---:|---:|---:|---:|---:|
| All | 9508 | 0.1125853062 | 0.1110517234 | 740 | 339 | 279 |
| Selected strict errors | 4588 | 1.4832327068 | 1.4832327068 | 186 | 94 | 15 |
| Saved fused-mask IoU >0.5 and selected-box error | 499 | 0.1904582083 | 0.1904582083 | 36 | 29 | 12 |
| Mask endpoint correct and final box wrong | 64 | 0.2002135217 | 0.2002135217 | 8 | 6 | 4 |

The coupling-gap median is zero in all four groups. Counts above a coordinate-error threshold are not predicted accuracy gains. In particular, 339, 94, 29 and 6 are counts of fixed-endpoint coordinate gaps, not additional correct boxes.

## A. Ground-truth provenance — PASS within the cached-data scope

`complete_fit/paired_span_loop.py:296–318` constructs root GT from dataset batch `center_label[:,0,:3]` and `size_gts[:,0]`, and computes the historical fused-mask IoU against `gt_masks[:,0]`. The formal arrays and JSONL GT agree exactly. GT is not generated from either endpoint or final prediction. The reviewed analyzer loads `root_box` solely for an explicitly offline GT-conditioned projection (`analyze_fixed_endpoint_axis_coupling_20261013.py:19–23,81–102`).

The original dataset/point-mask bytes were not independently reconstructed in this audit. All 9,508 saved point SHA256 strings exist and are well formed, but their point-array preimages are absent from the inspected formal NPZ format and were not rehashed. Saved fused-mask IoUs are not independently recomputed from masks. Evaluation type: **real_gt, historical coordinate diagnostic**; it has no new benchmark-performance claim.

## B. Normalization and numerical integrity — WARN

The coordinate errors are unnormalized absolute face differences. No metric divides by the model's own output maximum, mean or range. Medians and threshold counts are ordinary descriptive statistics (`analyze_fixed_endpoint_axis_coupling_20261013.py:60–80`).

The historical Mask-box IoU field has 112 values slightly greater than one, with maximum `1.0000050067901611`. The first independent verification rejected this range, and that attempt is preserved. Float64 geometric recomputation differs from the stored Mask-box IoUs by at most `6.6200876713828904e-6` and leaves every threshold decision unchanged. This is consistent with floating-point computation, has no effect on the present group counts, and was neither clamped nor rewritten. Final-box IoU and saved fused-mask IoU stay in `[0,1]`.

## C. Existence, hashes and source/runtime binding — WARN, no scoped blocker

The diagnostic exists and all its reported fields match. The current mixer SHA256 is `5ab81a56ffcd635eb61ff245f6771c40d20027b7708656b09e8d92bf3f5ac087`, identical to the R4 source-stage entry (`SOURCE_STAGE_R4_RECEIPT.json:630–634`). It computes `raw_gate`, clamps to `[0,1]`, and applies the same gate to center and size (`source/extremal_span_mixer.py:97–104`).

117 of the 118 R4 source entries have exact local byte preimages. The remaining Linux `pointnet2/_ext.cpython-37m-x86_64-linux-gnu.so` preimage was not found in the inspected local artifacts. R4 review/helper hashes and the canonical environment-spec hash are verified. This audit does not certify all 118 remote source files or the current live runtime.

The R4 receipt alone says staging/GPU pending; it is not a runtime receipt. Separately, all eight saved paired-preflight intake files match their hashes and sizes. Their admission's source-stage object equals R4 exactly, and its six helper hashes match local files, including `native_selected_geometry_preflight.py`. Both saved actual M0 receipts report the initial formula check; the bound preflight code performs exact tensor equality at lines 131–138. These are historical runtime witnesses, not a new forward performed by this audit.

The diagnostic directory did not supply the original analyzer invocation/exit transcript. The audit hashes the currently supplied analyzer and independently reproduces its numerical outputs; this establishes consistency and reproducibility, not proof of that original command or every action in the original executor session. The analyzer's zero-forward/no-remote declarations are consistent with its imports and code.

The 2026-10-09 saved diagnostic endpoints were produced by the frozen-parent runner, whose mixer SHA256 is `523203b7a5304ba41ca3bfb66c6911777adb0b55287806ddfed57d8ae9566adf`, not the current joint-training mixer bytes. A full AST comparison establishes equality after removing only the differing module docstring, `.detach()` calls and resulting identity assignments. Thus their frozen-value forward algebra agrees, while gradient flow and trainability differ. The report does not present archived endpoints as outputs of the current trainable network.

## D. Active computation and execution scope — PASS

Both defined analyzer metric functions (`faces`, `summary`) are called; their outputs are represented in the CSV/JSON. Its only imports are csv/hashlib/json/pathlib/numpy. This reviewer did not execute the analyzer, training source, a model forward, a training optimizer, SSH, a GPU query, or any network call. Only audit/forensic files under this review directory were written. The requested eight inputs remain byte-identical.

## E. Scope and claim boundaries — PASS with mandatory qualifiers

1. GT chooses these coefficients offline; they are not available at deployment. The 9,508 rows concern one retained historical run, selected Query and fixed endpoint pair. No new method accuracy or seed robustness is established.
2. Coordinate minimax error is not IoU. Neither the minimizing gates nor coordinate counts are IoU-optimal or attainable REC gains.
3. A fixed-endpoint restriction does not bound the trainable model: native regression, Mask support and selected Query can change during training. The current joint-training port explicitly restores gradient flow.
4. The saved archive contains clamped gates, not raw gates, for every one of its 9,508 rows. Six zeros do not distinguish raw `<0` from raw `==0`. Preflight raw-gate counts cover all 256 Queries in other augmented training batches (`native_selected_geometry_preflight.py:163–168`); they do not establish chosen-C clamp-gradient incidence on these retained validation rows.
5. The 499-row saved-mask subgroup is not current C eligibility. Current C requires the selected Query to be unmatched and both own and fused support IoUs to exceed 0.5 (`source_overlay_v2/selected_support_geometry_objective.py:13,26–56`). The historical field provides only fused-mask IoU and lacks the current matcher role/own-mask qualification.
6. Independent face freedom is already present in `face_residual_span_mixer.py:18–21,51–66`: six face residuals modify six coefficients and its absolute size orders arbitrary outputs. The saved `face_native_epoch3_closeout_20261012/CLOSEOUT.json:6–47` records strict hits of 4920 at E0 followed by 4599, 4625 and 4476. This supports the limited historical statement that that tested normal-training run did not improve the start. It does not establish that face freedom is new, impossible to train, or a proven cause of the decline. That historical run's full result integrity is not re-audited here.

## Warnings, action and provenance

There are no blocking findings for publishing the existing qualified coordinate diagnostic. Keep the reported scope/limitations, retain the 112-value numeric warning, and do not claim all source/preimage hashes or reviewer model identity were attested. Broader deployment, causal, raw-gate or whole-network conclusions remain unsupported.

The numerical verifier's first failure (out-of-range historical Mask-box IoU) and the source AST verifier's first failure (identity assignments remaining after removing detach) were preserved with scripts, stdout/stderr and exit records. The latter was a verifier normalization defect; the revised verifier removes only those no-op identity assignments. No experiment file was repaired or result rerun.

Evidence files: `deterministic_verification.json`, `historical_binding_verification.json`, `admission_binding_verification.json`, `r4_source_hash_verification.json`, `historical_intake_hashes.json`, and `saved_npz_hashes.json`. The complete request/response and model-attribution limitations are stored under `.aris/traces/experiment-audit/2026-10-13_run01/`.
