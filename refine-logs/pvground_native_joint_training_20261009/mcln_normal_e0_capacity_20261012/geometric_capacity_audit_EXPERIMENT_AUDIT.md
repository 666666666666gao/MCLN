# Experiment audit: saved MCLN geometric capacity

Date: 2026-10-12. Overall verdict: **WARN**, limited to scope and provenance qualifications. Deterministic CPU checks: **PASS**. No numerical correction or integrity failure was found in the stated offline geometric diagnosis. There is no blocker to reporting that qualified diagnosis.

This is a fresh-context, same-family, provisional review. The requested reviewer was `gpt-6-astra`, reasoning effort `max`, `fork_turns: none`; the host returned only the canonical task `/root/mcln_geometric_capacity_audit_20261012`. Actual model identity and actual reasoning effort are **UNATTESTED**. Requested identity is not treated as runtime attestation. No cross-family acceptance is claimed.

Only local saved artifacts and code were read. Verification ran NumPy on CPU, with zero SSH calls, GPU queries, neural forwards, optimizer updates, model edits, or changes to active experiments. All writes are beneath `geometric_capacity_audit/`. Input hashes were checked after recomputation.

## Verified numbers and scope

The 15,766,406-byte paired-row file has SHA-256 `9562bb3a4bea8fef04fc2bd7adb9aa3110224c1193bf2b4f6bc81436582ea417`, matching `full_ROWS_TRANSFER.json:3-4` and both probe outputs. It contains 9,508 contiguous, unique row indices; 141 scans; 2,068 scan/target pairs; 9,506 distinct scan/target/utterance triples; and 141 point-cloud hashes. Two repeated annotation triples are present; rows must not be described as 9,508 independent scenes or unique objects. Annotation IDs and the original annotation files are not in the row archive, so this audit does not independently authenticate the official annotation inventory.

The archived replay is one ScanRefer validation configuration, one checkpoint combination and seed 2027, with three paired branches. It is not a multi-seed or cross-dataset experiment. The same selected query is used across all three branches on every row; all three saved GT boxes also agree on every row. Query indices range from 0 to 251 within the 256-query axis. No probe reselects the query by GT IoU.

Counts use strict `IoU > threshold`, denominator 9,508:

| Saved output or offline geometry choice | >0.25 | >0.5 |
|---|---:|---:|
| Native saved model output | 5,515 | 4,409 |
| Uncorrected Mask-reference saved output | 5,459 | 4,767 |
| Full-transfer saved A+B output | 5,473 | 4,777 |
| GT choice between the two whole endpoint boxes | 5,653 | 5,103 |
| GT 729-point axis grid, float32 | 5,687 | 5,219 |
| Same 729-point grid, independent float64 check | 5,687 | 5,219 |
| GT continuous tied-axis knots, float64 | 5,687 | 5,222 |
| GT independent-face projection, float64 | 5,715 | 5,310 |

Both probe scripts were replayed with only their final output-file write and print expressions omitted from the parsed AST. Their resulting JSON objects matched the saved JSON objects exactly, including all 12 examples in each file. A separate implementation enumerating distinct in-domain knots, with a different interpolation order, agreed with the continuous probe within `1.4099832412739488e-14` IoU and matched both threshold classifications on every row. The separate implementation used 8-64 distinct candidate combinations per row; the author's fixed 216 combinations include duplicate and clipped knots, which is harmless.

Evidence: `probe_axis_mixture_capacity.py:9-20,36-66,79-89`; `probe_continuous_axis_face_capacity.py:10-17,48-91,104-113`; `AXIS_MIXTURE_CAPACITY_PROBE.json:16-42`; `CONTINUOUS_AXIS_FACE_CAPACITY_PROBE.json:6-28`; `evaluate_mcln_portable_pair.py:35-69,112-160`; `CPU_VERIFICATION.json`.

## A. Ground-truth provenance — PASS, with archive boundary

The probe target is `native.root_gt_box`, not the Mask-reference model output (`probe_axis_mixture_capacity.py:14-19`; `probe_continuous_axis_face_capacity.py:15-17`). The producer obtains that field from `GroundingEvaluator._parse_gt`, selecting the first/root dataset box (`evaluate_mcln_portable_pair.py:79-109`). `_parse_gt` concatenates dataset `center_label` and `size_gts` (`source/src/grounding_evaluator.py:1992-2010`). The dataset builds those fields from the annotated target object's point membership and axis-aligned extent (`source/src/joint_det_dataset.py:1086-1119,1324-1329,1387-1392`; `source/src/visual_data_handlers.py:195-197,225-227,246-259`). ScanRefer annotation loading and validation selection are explicit (`source/src/joint_det_dataset.py:585-597,633-645`; `source/train_dist_mod.py:2938-2950`).

This supports **real dataset GT**, subject to the saved-archive provenance boundary. The remote raw dataset, checkpoint bytes, and original execution were not reacquired or rerun. Local code and saved files cannot supply an independent cryptographic attestation of those remote inputs. The native repository evaluator is used and its counts are asserted against the row counts (`evaluate_mcln_portable_pair.py:130-150`); this is not a new run of an independently obtained official benchmark evaluator.

Spatial GT fields enter the producer after the model forward (`evaluate_mcln_portable_pair.py:118-124`). The new geometry modules do not take GT-box/GT-mask/IoU inputs. The original `_get_inputs` does retain text annotation maps/spans (`source/train_dist_mod.py:3454-3485`), so an annotation-free-interface claim would be wrong. GT is deliberately used by the offline probes to choose geometry, and the result files label this explicitly.

## B. Score normalization — PASS

The metric is intersection divided by geometric union. There is no division by a model's maximum, mean, best observed score, or another artificial score normalizer. Dependence of the ordinary IoU denominator on predicted box volume is legitimate IoU, not self-normalized accuracy. The probes preserve the native `1e-6` size floor, boundary-difference volumes and strict thresholds (`probe_axis_mixture_capacity.py:23-37,54-66`; `probe_continuous_axis_face_capacity.py:20-31,81-91`; `source/models/losses.py:712-752`). Near-perfect GT-projected values would be oracle geometry, not evidence of near-perfect learned accuracy.

## C. Result existence and numerical agreement — PASS

All requested files exist. Input byte length/hash, row count, query equality, GT equality, table counts, and all saved examples were verified. The archived `full_first_boundary_observation/replay/RESULT.json` has the SHA-256 named by `full_ROWS_TRANSFER.json:8`; its saved branch counts and completion state agree with the rows. The archived controller status is `complete`, exit code 0. `GEOMETRIC_PARAMETERIZATION_DIAGNOSIS.md:9-14,24` agrees with the recomputation, including 28/88 extra face-projection hits over tied-axis geometry.

Float32 reconstruction agrees **bit-for-bit** with every saved selected IoU for native, Mask reference and full transfer. Float64 geometry is numerically different: maximum absolute differences from saved float32 IoUs are approximately `2.074e-6`, `6.203e-6`, and `8.889e-6`, respectively. All three arms have zero threshold disagreements at 0.25 and 0.5. The 729-point float32/float64 grids also have zero threshold disagreements. Float64 is not a reproduction of network arithmetic or a formal interval-arithmetic certificate.

The continuous search's three additional >0.5 rows are indices 6202, 8686, 8693 (JSONL lines 6203, 8687, 8694). Their float64 grid/continuous IoUs are approximately 0.493444/0.509979, 0.495504/0.553617, and 0.486156/0.545813. These are finite-grid omissions, not float32/float64 threshold flips. The closest continuous and projected IoUs to either reported threshold remain at least `1.178e-4` away; this is an observed numerical margin, not a proof of a universal floating-point error bound.

## D. Actual code paths — PASS for the requested scope

Both probe metric functions are called and their outputs reach the saved tables and examples; the CPU replay verified those complete paths. The continuous script's `coarse` JSON load at line 79 is unused; it creates no metric claim and does not substitute the grid for the continuous calculation.

The source mixer uses one clamped gate per axis for both center and size (`source/extremal_span_mixer.py:97-103`). The model routes non-native arms through `finish_portable_geometry` (`source/models/mcln.py:3167-3170`). That function applies A to query-mask logits, rebuilds the Mask geometry, then invokes B (`source/mcln_portable_geometry.py:18-35`). Thus the algebraic tied-axis family matches B's **type of parameterization**. The saved reference branch is uncorrected by A, however. The producer saves only final selected boxes and GT, not `span_mask_center`/`span_mask_size` (`evaluate_mcln_portable_pair.py:109`; `source/mcln_portable_geometry.py:32-34`).

This audit did not execute a neural path. Historical neural execution is evidenced by archived producer output; only the geometric probe paths were re-executed here.

## E. Scope — WARN

The new diagnosis already states the decisive qualifications (`GEOMETRIC_PARAMETERIZATION_DIAGNOSIS.md:16,22,24,28-32`). Keep them. These are selected-query, saved-endpoint, GT-conditioned capacities on one cached validation replay. They do not establish learned gate quality, physical-instance correctness beyond box overlap, six-face-module validity, generalization, seed robustness, a training gain, or completion of the broader research goals.

The saved-endpoint continuous optimum is **not** an upper bound on the deployed A+B outputs. The independently recomputed saved full-transfer IoU exceeds this axis optimum on 43 rows; full transfer is correct while the saved-endpoint axis family is below threshold on 1 row at 0.25 and 3 rows at 0.5. Consequently a difference of total hit counts is not the number of recoverable deployed errors. At 0.5 the continuous family has 448 hits outside the full-transfer hit set and full transfer has 3 hits outside the family, for a net count difference of 445.

`MCLN_INTERFACE_AND_TRAINING_NOTES.md:3,17,21,27` is an earlier 8-row/pre-completion snapshot; its statements that the full 9,508 replay is pending are superseded by the archived completed result. If reused as a current status document, add a historical timestamp/supersession note. This does not require changing model code or numerical results. Claims in the diagnosis about active training status, earlier PV negative experiments, PV counts 5677/4920, and unfinished Nr3D/Sr3D goals are outside this geometry audit and are not newly certified here.

## F. Classification — real_gt, offline GT-conditioned geometry diagnostic

The target provenance is `real_gt`. The evaluation subtype is `offline_GT_selected_geometric_capacity_on_saved_predictions`. It is neither synthetic reference consistency nor a new neural/model-accuracy evaluation. GT selection imposes a stronger claim ceiling than the generic `real_gt` label: these oracle values cannot become deployable scores or model-performance claims. The saved native/reference/full-transfer baseline counts remain historical model-output measurements, with the separate provenance boundary above.

## Independent mathematical verification

**Continuous tied-axis knot sufficiency: valid in real arithmetic.** For axis j, write lower and upper faces as affine functions `l_j(t)=r_lj+t(n_lj-r_lj)` and `h_j(t)=r_hj+t(n_hj-r_hj)`, `t in [0,1]`. Positive endpoint widths give positive widths throughout. Partition the interval at 0, 1, and the crossings of either predicted face with either GT face. On each piece, the overlap length is affine (including the identically-zero case): the four face crossings include changes in min/max branches and the onset/loss of overlap. Holding the other axes fixed, predicted volume and intersection are affine in t. Thus IoU has form `(a*t+b)/(c*t+d)`, with strictly positive union. Its derivative has constant sign on that piece, or is identically zero. A maximizing coordinate can therefore be moved to a piece endpoint without decreasing IoU.

The partition points depend only on the corresponding axis, not the other two gates. Starting from a global maximizer on compact `[0,1]^3`, successively move each coordinate to a knot without decreasing the value. The resulting Cartesian knot tuple is still a global maximizer. The six padded knots per axis, hence 216 combinations, contain such a tuple. Zero face displacement needs no crossing; clipping out-of-domain crossings merely duplicates the 0/1 endpoints. This proves the real-arithmetic family statement in `probe_continuous_axis_face_capacity.py:43-69`; it does not make its float64 outputs exact real numbers.

**Independent six-face GT projection: legal and dominating in the stated family.** On one axis let lower-face permitted bounds be `[A,B]` and upper-face permitted bounds be `[C,D]`. Since both endpoint boxes are ordered, `A<C` and `B<D`. For GT bounds `L<H`, the projections are `p=min(max(L,A),B)` and `q=min(max(H,C),D)`. Strict coordinatewise monotonicity of min/max gives `p<q`. Each projected coordinate lies in its endpoint interval, hence has an independent mixing coefficient in `[0,1]` (an equal-endpoint interval needs no unique coefficient). This proves legality for every ordered endpoint pair and positive GT extent, not merely the observed rows.

For any feasible box with positive intersection, move a face toward the corresponding GT face as far as its permitted interval allows. Removing an exterior slab reduces predicted volume with unchanged intersection, so IoU cannot decrease. Adding a missing interior slab changes predicted volume and intersection at rates `A_other` and `C_other`. If the current axis width is s and overlap is d, then `V=A_other*s`, `I=C_other*d`, `d<=s`; the IoU derivative numerator is `C_other*(V+T)-A_other*I = A_other*C_other*(s-d)+C_other*T >= 0`, where T is GT volume. The moves retain positive overlap and valid ordering. Repeating the moves reaches the projected box. Zero-intersection boxes cannot outperform its nonnegative IoU. The projection is therefore a maximizer over valid boxes in the independent-face endpoint intervals, and in particular dominates the tied-axis subfamily.

The CPU check verifies all 9,508 projected boxes lie in their permitted intervals and have strictly positive extents, with minimum extent `0.009714454412460327`; the minimum `face_iou-axis_iou` is 0. This does **not** imply arbitrary six independently predicted gates preserve validity: in 10 saved rows, `max(lower endpoints) >= min(upper endpoints)` on at least one axis (first example row index 740). A learned six-face parameterization needs its own validity definition and evaluation. No such module was added or tested.

**Finite grid versus exact family.** In real arithmetic, evaluating the 729-point grid provides reachable witnesses, hence a lower bound on per-row maximum IoU in the continuous family; it does not provide an upper bound. The recorded float32 and float64 numbers are approximations, not certified arithmetic bounds, so exact pointwise inequalities across those precisions are not asserted. The knot theorem characterizes the continuous family exactly in real arithmetic; the implementation evaluates it numerically in float64. The independent-face family is larger. Neither continuous nor face-projection family is the exact actual-B family when its A-corrected endpoints were not saved.

## Required corrections, blockers and claim impact

- No numerical correction is required for the two probe JSONs or the new geometric diagnosis table.
- Preserve the existing GT-only, saved-endpoint and float64 qualifiers. Do not turn extra oracle hits into promised learned gains, gains attributable to A or B, or a deployed-model upper bound.
- Label the earlier interface/training note as historical if it is presented as current; its pending-full-replay statements are stale. This is the only documentation correction identified within the inspected context.
- No blocker remains for the explicitly scoped offline diagnosis. Exact deployed-B capacity remains blocked by missing A-corrected pre-B endpoints; learnability/model-accuracy conclusions remain unsupported without corresponding neural evaluation; exact reviewer-model attribution remains blocked by missing host attestation. These do not block the local geometric result.

The shared-axis parameterization claim, knot sufficiency, projection legality/dominance, saved hit counts and 28/88 extra reachable hits are supported within their stated family. Model accuracy improvement, deployability, physical-instance identity, active-training success and broader-project completion are unsupported by these probes.

## Reproduction and trace

Run `uv run --offline --no-project --with numpy --python 3.13 python geometric_capacity_audit/verify_geometry.py` from the parent directory. The recorded verification used Python 3.13.12 and NumPy 2.5.3; it completed in approximately 2.8 seconds of script time. Only the audit directory receives verification outputs.

Machine result: `EXPERIMENT_AUDIT.json`. Deterministic evidence: `CPU_VERIFICATION.json`, `CPU_VERIFIED_GEOMETRY.npz`, `PROVENANCE_VERIFICATION.json`, and `INPUT_HASHES.json`. The complete received task and follow-up, review response, requested-versus-actual model identity and input-hash trace are in `.aris/traces/experiment-audit/2026-10-12_run01/`. The full semantic review remains same-family/provisional; the reproducible arithmetic checks are separately marked deterministic/accepted.
