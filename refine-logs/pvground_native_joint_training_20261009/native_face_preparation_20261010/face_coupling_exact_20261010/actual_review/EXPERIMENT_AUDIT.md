# Closed face-coupling experiment audit

Date: 2026-10-10 (Asia/Shanghai). Execution scope: `CLOSED_OFFLINE_GEOMETRY_DIAGNOSTIC`.

**Overall verdict: WARN. Blocking findings: none.** The closed-row arithmetic and continuous coupled-family maximum are supported. The warnings concern residual field names and the limits of a GT-informed geometry diagnostic. No new performance result or model acceptance follows.

Requested reviewer: `gpt-6-astra`, reasoning effort `max`, native Codex route. Actual model, effort, and backend: **UNATTESTED**; this native response does not certify them. Reviewer context is fresh, `review_independence: same-family`, `acceptance_status: provisional`; there is no cross-family acceptance. The initial request supplied expected counts, so this was not a numerically blind review. Those counts were independently recomputed from the frozen rows, rather than accepted as evidence.

## Evidence and independent execution

The raw `complete_fit/formal/rows.jsonl` has SHA256 `7b29b1646e52e8d3d253b3167b6895df9d789e7325022b0d9440814ec143c7b9`. It agrees with both the formal receipt and `CPU_RECOUNT.json`. The receipt SHA256 agrees with the recount, and the recount SHA256 agrees with the diagnostic. The reviewer verified these bindings before consuming the rows.

The independent implementation is `independent_closed_geometry_audit.py`. Unlike the submitted diagnostic, it builds sorted unique in-range crossings, pads with an endpoint, and evaluates a broadcast product of interval lengths/intersections directly. It does not import or call the diagnostic. It verified all 9508 CSV row identities, query indices, numerical columns, subgroup flags, and aggregate values. Source rows are exactly IDs 0 through 9507, spanning 141 scenes and 2068 scene-target pairs. Each row retains its original selected Query for all five boxes.

The single successful verification used cached NumPy 2.5.3 through `uv run --offline --with numpy python -B -X utf8 ...`; no network access or permanent installation was requested. Two earlier interpreter discovery attempts failed before the diagnostic ran (`No pyvenv.cfg file`, then missing NumPy); both are preserved in the trace. There was no torch import, SSH call, GPU/model execution, checkpoint read, live training-status query, source change, cleanup, restart, or publication. The original bounded diagnostic and all primary inputs retain their before/after hashes.

## A. Ground-truth provenance: PASS within the closed evidence scope

`diagnose_closed_face_coupling_exact_cpu.py:24` reads `root_box` from the hash-bound original rows. In the closed runner, `paired_span_loop.py:296` obtains that box from dataset batch `center_label` and `size_gts`, then records it at line 317. `run_span_pair.py:194` constructs the ScanRefer validation dataset, with 9508 annotations checked at lines 183 and 198. The GT is not generated from a prediction or normalized from its output.

The reviewer did not reload the original dataset or rerun its loader. Thus this audit verifies the stored GT binding and source provenance, not a new independent dataset-authenticity certification. GT directly selects the oracle/projection geometry, so its result is expressly offline.

## B. Metric and normalization: PASS

The function at `diagnose_closed_face_coupling_exact_cpu.py:28` is ordinary axis-aligned 3D intersection-over-union, with union `predicted volume + GT volume - intersection`. It never divides a score by the model's maximum, minimum, or average. Strict deployed hits use `IoU > 0.25` and `IoU > 0.5`; oracle/projection hits use `IoU > threshold + 1e-6`. The denominator for the reported hit counts is always 9508.

All stored native, Mask, fixed-half, whole-support, extremal-support, and GT boxes are finite with positive sizes. The independent computation also checked positive unions. Float32 deployed counts and float64 recomputation give the same threshold decisions for every stored box.

## C. Result existence, binding, and numbers: PASS

Both result artifacts exist and all aggregate values match the independent calculation. The following are existing selected-Query box counts, not new formal evaluations:

| Stored geometry | Hits > 0.25 | Hits > 0.5 |
|---|---:|---:|
| Native | 5615 | 4495 |
| Mask | 5606 | 4881 |
| Fixed half | 5675 | 4857 |
| Whole support | 5676 | 4921 |
| Extremal support | 5677 | 4920 |

The GT-informed continuous coupled oracle has **5839 / 5379** hits and the feasible independent-face GT projection has **5877 / 5472** hits, with the declared `1e-6` qualification margin. Neither computation has a value within `1e-6` of 0.25 or 0.5. The largest discrepancy between the independently evaluated exact oracle and the submitted CSV is `2.63123e-14`; the largest projected-bound discrepancy is `4.10783e-15`. Float32 CSV text discrepancies are at most `4.63257e-8`, consistent with decimal serialization.

## D. Execution and dead-code assessment: PASS

The IoU routine is used to verify all stored boxes, the independent projection, and every enumerated coupled combination. Result-writing code is reached, with a complete 9508-row CSV and the matching JSON. The reviewer independently executed a separate NumPy verifier successfully. `prepare_exact_face_diagnostic.py:7` corrects the original group name, and line 31 changes the evidence condition from the loose bound to the exact oracle. Neither preparation script nor original diagnostic was executed or changed by this review.

The exact diagnostic's `best_gate` array is calculated but not exported. This does not invalidate its maximum or counts; the reviewer exported maximizing gates for the 93 witnesses in `independent_witnesses.csv`.

## Continuous three-axis maximum: PASS

The reviewed current mixer applies the same gate to center and size (`source/extremal_span_mixer.py:100` through line 102); the closed original mixer uses the same formula at lines 103 through 105. Thus, for each axis, lower and upper faces are

`L(g) = L_mask + g (L_native - L_mask)` and `U(g) = U_mask + g (U_native - U_mask)`, for `0 <= g <= 1`.

The width is the convex combination of two positive endpoint widths, so it stays positive over the complete gate interval. For fixed other-axis gates, let `A` be the predicted cross-sectional area, `B` the intersected cross-sectional area, and `V_gt` the fixed GT volume. Then

`IoU(g) = B ell(g) / [A (U(g)-L(g)) + V_gt - B ell(g)]`,

where `ell(g) = max(0, min(U(g), G_high) - max(L(g), G_low))`. Its formula can change only when a moving face crosses one of the two GT faces. Within each segment, the numerator and denominator are affine in `g`, and the denominator is strictly positive. The derivative of an affine-over-affine function has constant sign (or is zero), so a segment maximum is attained at an endpoint.

The full gate cube is compact and IoU continuous. Starting from a global maximizer, each coordinate can be replaced by one of that axis's endpoint/crossing candidates without lowering the value. These candidate sets do not depend on the other two coordinates. Repeating for all three axes yields a global maximizer in their Cartesian product: at most six candidates per axis, hence **216 combinations**, including duplicates. This is a structural enumeration of the continuous maximum, not a coarse gate grid or coordinate-ascent heuristic.

There are **129 constant lower-face row-axis entries and 129 constant upper-face entries**, with the same 129 axes constant on both sides. Their slopes are zero, so they introduce no isolated crossing; endpoints suffice. The diagnostic's masked division at line 54 assigns a duplicate endpoint in those cases. Independent nonzero-slope root construction agrees. Across all source, candidate, and projected widths, the minimum is `9.999999974752427e-7`.

The separate minimum of the three optimized 1D IoUs is also a valid upper bound: each 3D IoU is no greater than the corresponding projected 1D IoU, since the intersected area in the other axes is no larger than either box's area. It can be loose. The exact 3D maximum lies below this bound to numerical precision (maximum excess `5.55112e-16`).

Both learned stored arms have gates in `[0,1]`; their float64 center/size reconstruction residuals from stored endpoints are at most `5.83254e-7` (whole) and `6.46174e-7` (extremal), consistent with original float32 arithmetic. The independently computed exact oracle bounds every observed native, Mask, fixed-half, whole-support, and extremal-support IoU, with maximum positive excess **0**. The loose projected bound's largest excess is `3.33067e-16`.

## Independent faces and robust witnesses: PASS with the stated scope

The GT-clipped lower and upper faces lie inside their separate native/Mask endpoint intervals. The reviewer reconstructed feasible independent gate weights, checked they lie in `[0,1]`, and checked all projected boxes satisfy `upper > lower`. This establishes a feasible choice for this geometry family. The audit makes no claim that this projection maximizes IoU over independent faces.

There are **4588** deployed extremal strict errors. Of these, **93** satisfy both `maximum coupled IoU <= 0.5 - 1e-6` and `independent projected IoU > 0.5 + 1e-6`. For those rows, no continuous three-axis coupled gate can qualify the same stored Query/source boxes, while a valid six-face choice does. The weakest exclusion gap is `7.42490e-5` below 0.5; the weakest independent qualification gap is `1.59523e-4` above 0.5. Their smallest projection-minus-coupled gap is `0.00326438`. These are numerical-margin witnesses under float64 computation, not interval-arithmetic certificates or statistical robustness claims.

Among the **499 FUSED-Mask-qualified strict errors**, **44** are such witnesses. In the closed runner, `mask_iou` is explicitly computed from the alpha-weighted text/own fused mask (`paired_span_loop.py:312` through line 315). It does not establish qualification of the own Query mask alone. The initial diagnostic's `own_selected_fused_mask_qualified_strict_errors` name was incorrect; the exact version corrects the group at line 97 and discloses the correction at line 129. The old artifact remains preserved.

The source geometry also contains actual ordering hazards for unrestricted independent gates: native and Mask intervals are strictly disjoint on **12 x-axes, 8 y-axes, and 11 z-axes**, or **31 row-axis instances across 30 expressions**. Selecting opposite endpoints can therefore produce `lower > upper`. This is an observed reason any prospective independent-face implementation would need to ensure ordering. It does not affect the current GT projection, whose boxes all pass the ordering check; no network design or modification is approved or implemented here.

## E. Scope and findings: WARN, no blockers

**W1 — Residual field-name ambiguity.** `rows.csv` calls its final flag `bound_below_half_and_independent_above_half`, while the exact code at line 94 uses the exact 3D oracle. The corresponding JSON count name also ends in `excluded_by_bound`. Only **9** of the 93 witnesses are excluded by the loose minimum-of-1D bound; **84** additionally require the exact 3D maximum. Consumers must use the exact-oracle definition, not infer that all 93 were certified by the loose bound. The original inputs were preserved rather than renamed.

**W2 — Claim ceiling and provenance.** This is one closed selected-Query diagnostic over stored dataset GT, not a new normal-training run, benchmark evaluation, generalization test, or learned-policy experiment. The initial request exposed the target counts, and reviewer routing is not attested. Fresh-context same-family review remains provisional. The input's `fresh_diagnostic_review_pending: true` is preserved historical state; this separate sealed report records the completed review.

Allowed claims are the exact stored-count reproduction, the continuous coupled-family maximum proof, the finite-row geometric witness counts, and the observed ordering hazard. These require the same stored selected Query, the same native/Mask endpoints, the strict threshold convention, and the stated numerical margins.

Unsupported claims include an attainable learned gain of 93 hits, deployable oracle performance, new formal accuracy, prediction identity improvement, own-mask qualification, efficacy of three modules, or efficacy of the ongoing joint training. Changing source boxes or selected Queries changes the diagnosed family and can remove the stated impossibility; the result is not a universal lower bound on all future models.

No rerun or input mutation is required to accept the qualified deterministic geometry findings. Downstream prose and parsers should explicitly name the exact-oracle condition and maintain the GT/offline qualifiers. No active run should be restarted, modified, or promoted on the basis of this audit.

## F. Evaluation classification

`real_gt`, specifically **closed offline representational geometry analysis using stored dataset GT**. The GT is real provenance evidence, but this oracle's use of GT to choose geometry limits the claim to representational feasibility. It is neither deployable performance nor a newly trained method.

## Artifacts

- `INDEPENDENT_RECOMPUTATION.json`: full deterministic results and margins.
- `independent_witnesses.csv`: all 93 witness identities and maximizing coupled gates.
- `independent_closed_geometry_audit.py`: independent CPU implementation.
- `EXPERIMENT_AUDIT.json` and `EXPERIMENT_AUDIT.seal.json`: machine verdict and hash seal.
- `../../.aris/traces/experiment-audit/2026-10-10_face_coupling/`: raw request, response, metadata, interpreter failures, execution output, and before/after input hashes.
