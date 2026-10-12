# Experiment audit: retained-source PV geometry and training responsibility

Date: 2026-10-12. **Overall verdict: WARN (scope/provenance qualifications); deterministic checks: PASS.** No numerical correction is required. The revised `RETAINED_PV_PROBLEM_NOTE.md`, SHA-256 `821bd6835c5d8a77d47014d7cbbc2fc9acdc99ba15b9b595644d61103581cb60`, closes the sample-independence, precision and overly broad repair-scope wording issues. There is no remaining blocker to its narrowly scoped offline diagnosis.

This is a fresh-context, **same-family/provisional** semantic review. Requested reviewer identity: `gpt-6-astra`, reasoning effort `max`; actual model and actual reasoning effort: **UNATTESTED**. The host-visible reviewer identifier is `/root/pv_retained_geometry_audit_20261012`, a canonical task name, not model attestation. No cross-family acceptance is claimed. Reproducible numerical checks are separately classified as deterministic/accepted.

Only local artifacts were read. The audit ran NumPy on CPU, with zero SSH/remote calls, GPU calls, neural forwards, backward passes, optimizer updates, model edits or promotion. All audit writes are in this `integrity_review` directory. The original producer, probe scripts, result JSONs and archived arrays were preserved. The existing diagnostic scripts were replayed only after removing their terminal output-write/print expressions from their parsed AST; the neural producer was never imported.

For references below, `probe/` is this directory's parent; `producer/` is `C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/`; `earlier/` is `C:/Users/gb/.codex/tmp/pvground_selected_mask_training_20261009/complete_fit/`. Two additional source copies were verified against `producer/imports.json`: `losses` is `C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/models/losses.py`; `dataset` is `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/runtime_binding/dataset_source/src/joint_det_dataset.py`. `native_root_bbs` is the copy in `C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/`, verified against the original `pair_spec.json`.

## Verified scope and numbers

The source is the **20261009 frozen-parent retained-source cache**, not a new evaluation and not an independently established rowwise copy of the later native E0 replay. `formal/rows.jsonl` contains 9,508 contiguous unique row IDs, 141 scans, 2,068 scan/target pairs and 141 point-cloud hashes. No utterance or annotation ID is saved in these rows, so distinct-expression identities cannot be independently counted here. The 1,189 NPZ files cover every row exactly once, with 256 finite candidate boxes per row; 1,188 batches contain eight rows and the last contains four. The selected Query ranges from 0 to 255. It is the unique maximum of the saved native score on every row, and the same Query is used for the native endpoint, A-corrected Mask endpoint, final B box and selected fused Mask.

All hit counts use strict `IoU > threshold`, denominator 9,508:

| Saved output / GT-conditioned geometry | >0.25 | >0.5 |
|---|---:|---:|
| Native endpoint | 5,615 | 4,495 |
| A-corrected Mask-reference endpoint | 5,606 | 4,881 |
| Actual saved B output | 5,677 | 4,920 |
| GT choice between the two whole endpoints | 5,788 | 5,229 |
| GT continuous tied-axis family, float64 | 5,839 | 5,379 |
| GT independent-face projection, float64 | 5,877 | 5,472 |
| Any of the 256 actual saved B candidates, GT overlap | 8,970 | 8,234 |

The continuous family reaches 162/459 actual B errors; actual B has zero threshold-qualified rows outside that family in this cache. Independent-face projection adds 38/93 rows beyond the tied-axis family. These are GT-selected geometric capacities, not new model accuracies or promised gains. The reported smallest-half counts also reproduce: 106/278 axis-reachable B errors and 19/57 extra face-reachable errors. The groups are formed from the saved float32 GT-volume product and quantile cuts, so they are row-weighted size groups, not independent-object cohorts.

The full, mutually exclusive failure partition is:

| Among actual B errors | >0.25 criterion | >0.5 criterion |
|---|---:|---:|
| Selected axis family can qualify; some actual candidate qualifies | 108 | 277 |
| Selected axis family can qualify; no actual candidate qualifies | 54 | 182 |
| Selected axis family cannot qualify; another actual candidate qualifies | 3,185 | 3,037 |
| Neither selected axis family nor any actual candidate qualifies | 484 | 1,092 |
| Total actual B errors | 3,831 | 4,588 |

Thus 3,293/3,314 errors have another saved candidate above threshold, and 538/1,274 have none. The sets “axis repair possible” and “another actual candidate qualified” overlap; they must not be summed as disjoint recoverable gains. All256 qualification is only geometric overlap against the same GT: it does not certify physical-instance identity, complete language satisfaction, a learnable scorer, or a deployable selection rule. This audit also verified that float64 all256 flags agree with the original saved float32 `oracle25/oracle50` all256 flags on every row.

At strict 0.5, the selected final-B-box versus selected fused-Mask matrix is:

| | Mask >0.5 | Mask <=0.5 |
|---|---:|---:|
| Final B box >0.5 | 4,628 | 292 |
| Final B box <=0.5 | 499 | 4,089 |

The Mask marginal is 5,127 and the box marginal is 4,920. The NPZ cache does not retain full validation point masks; this matrix independently recounts saved per-row Mask IoUs whose producer uses dataset point-mask GT. It is not a new point-mask recomputation of the 9,508 validation rows.

## A. Ground-truth provenance — PASS, within the archived-input boundary

The geometric target is `root_box`/`root_gt`, not either predicted endpoint (`probe/probe_retained_geometry.py:22-28,125-136`). The producer builds the root box from dataset `center_label[:,0,:3]` and `size_gts[:,0]` after forwarding the model, and the selected Mask target is `gt_masks[bid,0]` (`producer/paired_span_loop.py:287-317`). `run_span_pair.py:180-199` instantiates the ScanRefer validation dataset; `paired_span_loop.py:278-280` disables augmentation for formal evaluation.

The dataset source hash exactly matches the original archived import receipt. Its annotation loader reads the ScanRefer split and annotation files (`dataset:585-597`); target masks use the annotated object's point membership, target boxes use `scan.get_object_bbox`, and these populate `center_label`, `size_gts` and `gt_masks` (`dataset:1086-1119,1324-1329,1387-1392`). The **training** branch additionally applies multiplicative box jitter when train augmentation is enabled (`dataset:1112-1113`); this does not execute for formal validation. The saved training-witness boxes below are actual augmented/jittered training targets and should not be conflated with clean validation boxes.

The deployed B module receives predictions/support geometry, not spatial GT (`producer/span_refinement_model.py:8-21,41-44`; `extremal_span_mixer.py:110-120`). Its native score does use existing text annotation maps (`native_root_bbs.py:4-9`), so “no annotation information enters the benchmark score” would be false. It does not use the root box, target mask or IoU to choose the selected Query. GT is explicitly used by the **offline diagnostic** to choose alternative geometry.

Local rows, source and collection receipts cannot independently authenticate the original remote raw dataset, annotations, checkpoint bytes or execution. Those were not reacquired here. The label is real dataset GT with this archive provenance boundary, not unconditional external dataset attestation.

## B. Score normalization — PASS

Geometric IoU is intersection divided by union (`probe/probe_retained_geometry.py:33-42`; `producer/run_span_pair.py:252-258`). There is no normalization by the model's maximum, minimum, mean, best case or confidence. The ordinary union's dependence on prediction volume is part of IoU. The nominal `1e-6` size floor is an existing geometric definition, not score rescaling. Training normalization by actual valid target count and the original `(10,2)/7` loss coefficients is not metric normalization (`producer/matched_span_objective.py:5-17`). The witness Mask checks use integer point intersection/union counts.

## C. File existence, hashes and precision — PASS

`formal/rows.jsonl` is 17,678,554 bytes, SHA-256 `7b29b1646e52e8d3d253b3167b6895df9d789e7325022b0d9440814ec143c7b9`; `formal/receipt.json` is 947 bytes, SHA-256 `81f2818ea99e8b21d54e39c545852b360a683ed368457a27bea03fe9738a6f5f`. They match the producer receipt, the reported probe and the **pre-existing** `MANIFEST_STDOUT.json`/`INTAKE.json`. The original collection is recorded at `2026-10-10T00:30:23.217742+08:00`. The manifest and intake have identical 1,246-entry file lists. All 1,189 currently audited NPZ bytes match both those original entries and the new `ARRAY_BINDINGS.json`. The sixteen training-witness arrays and their two receipts also match original collection hashes.

This is stronger than merely hashing today's local arrays: it is a comparison with preserved original collection evidence. It remains a local archived chain, not a fresh independent attestation of remote bytes. All eleven `new_runner_files` hashes, the final-spec-to-pair-spec hash, the training completion receipt, and the duplicate earlier/current matching source agree. The archived formal log/receipt records completion and reconciles row counts with the native evaluator (`paired_span_loop.py:337-359`; `run_span_pair.py:220-227`). This audit did not reacquire or rerun an external official evaluator.

All saved numerical fields, all twelve geometry examples, the complete all256 partitions and both added diagnostic JSONs reproduce. A distinct implementation enumerates only unique in-domain knots and interpolates as `reference + gate*(native-reference)`. It agrees with the probe within `2.631228568361621e-14` IoU and has identical threshold classes. It uses 8–64 distinct combinations per row; the probe's 216 padded combinations include harmless duplicates.

**Precision is not bitwise uniform across metric formulas.** NumPy float32 boundary-volume IoU differs from the saved producer IoU by at most `2.4437904357910156e-6` (native), `5.185604095458984e-6` (Mask endpoint), and `3.4570693969726562e-6` (B). Even reproducing the producer's center/size volume formula in NumPy is not bitwise identical to archived GPU IoU: maximum differences are `4.172325134277344e-7`, `6.556510925292969e-7`, and `5.960464477539062e-7`. Float64 boundary-volume differences from saved values are at most about `2.840e-6`, `6.620e-6`, and `4.024e-6`. Every route has **zero** threshold disagreements at 0.25 and 0.5.

By contrast, B's saved center and size are **bitwise exactly reconstructed in float32** as `(1-gate)*Mask + gate*native`. Its float64 face reconstruction maximum error is `4.903858519256232e-7`. The different IoU and box-reconstruction assertions must remain separate. The continuous and face calculations reinterpret saved float32 endpoints and apply the nominal size floor in float64; they are numerical approximations to the stated real-arithmetic family, not exact network arithmetic or certified interval bounds. The closest continuous IoU is approximately `6.793e-4` from 0.25 and `3.967e-5` from 0.5; the closest projected IoU margins are `4.812e-4` and `7.536e-5`. These are observed margins, not universal error bounds.

## D. Live code paths and direct responsibility — PASS for the specified archived paths

The probe's bounds/IoU routines feed its knot enumeration, projection, per-file all256 calculations, tables and saved examples (`probe/probe_retained_geometry.py:33-179`). Complete replay and an independent implementation verify those paths. The two supplement scripts likewise read real saved arrays/rows and feed their reported summaries. There are no phantom metric files or uncalled metric functions supporting the audited numbers.

The producer obtains one frozen-parent forward and applies both heads to that same dictionary (`producer/paired_span_loop.py:68-83`). Native boxes, the parent A-corrected Mask boxes and the B outputs are saved in the same NPZ (`paired_span_loop.py:291-299`). The source B clamps one gate per axis and uses it for both center and size (`extremal_span_mixer.py:100-105`); `span_refinement_model.py:12-14` explicitly records the pre-B Mask endpoint. Therefore this PV cache supplies the actual pre-B endpoint pair for this archived run. This differs from the separate MCLN reference audit, where the saved Mask-reference branch was not necessarily the A-corrected pre-B endpoint. Results cannot be transferred across those cache definitions.

B training gets frozen-parent actual-GT Hungarian assignments (`matched_mask_objective.py:6-30`; `paired_span_loop.py:126-136`). The hash-matched native `loss_boxes` explicitly indexes only those source Query indices and corresponding target boxes (`losses:518-544,781-787`). The active B objective is `(10*loss_bbox + 2*loss_giou)/7`, with valid-GT normalization, no extra root-box target and no score loss (`matched_span_objective.py:5-17`). Parent/Mask/score tensors are frozen in this archived training path (`span_refinement_model.py:33-44`; `paired_span_loop.py:68-76,146-149`).

The original preflight differentiates B's final center/size tensors and asserts zero direct gradients outside the matcher-selected Query set (`paired_span_loop.py:137-145`). Both saved steps report eight matched queries and zero unmatched nonzero gradients in both arms; these records match the original manifest. This audit did not run backward. Shared head parameters can still alter unmatched Query outputs after a step; zero direct output gradient does **not** mean an unmatched candidate is permanently unchanged or has no indirect parameter sharing.

The earlier A mask objective also indexes Query-mask logits by matched Query indices and target masks by the paired actual target IDs (`earlier/matched_mask_objective.py:33-57`). This supports the explicitly inspected terms' responsibility statement. It does not establish what every later end-to-end loss or matcher does. In particular, no claim about current MCLN matching, full-dataset unmatched frequency, or the effect of expanding positives follows from these two saved preflight observations.

## E. Scope, supplements and narrative — WARN; narrow wording issues closed

The formal evidence is one ScanRefer validation cache, one checkpoint combination, seed 2027, from the frozen-parent head-training experiment. It is not multi-seed, cross-dataset, a new architecture test, a learned oracle, or a later native-E0 identity audit. Count equality with 5,677/4,920 cannot prove equal row identities, inputs or outputs for E0. The revised note preserves this at `RETAINED_PV_PROBLEM_NOTE.md:3`.

**Training observations.** Both step receipts, all sixteen witness byte hashes, selected score maxima, exact actual matcher/GT indices, raw point hashes and reconstructed point-level own/fused Mask IoUs verify. There are 16 observations, **eight distinct training rows/scans**, and two forwards/steps of the **same loaded augmented batch**. Raw points, superpoints, actual GT boxes and actual GT masks are identical between the two steps. The producer explicitly reuses the same `batch_cpu` (`paired_span_loop.py:367-380`). Seven selected-Query observations are unmatched; fourteen have both own and fused Mask IoU >0.5; five satisfy both conditions. These are observations, not five independent expressions. The supplement's explicit correction is necessary when interpreting the original `summary.rows=16,batches=2`; it resolves that ambiguity without changing original evidence bytes (`inspect_prediction_and_repeated_batch.py:12-28,51-69`; revised note:38).

The independent Mask calculation uses superpoint membership counts and integer intersection/union. The stored fused foreground also agrees exactly with the zero-logit threshold; the smallest observed absolute selected logits are about 0.00698 for own and 0.00360 for fused, with no threshold ambiguity here. Each witness has exactly one actual GT and `matched_gt_ids=[0]`.

Mask quality alone does not label these five observations “Mask good / B box bad.” The witness lacks the selected final B box. As an additional bounded check, this audit evaluated the saved native and Mask-reference **endpoints** against the actual augmented training box: only two of the five qualified-unmatched observations have Mask-reference endpoint IoU <=0.5. Both are the same training row 27845 at steps 1/2, with IoU approximately 0.415923/0.456291. The native endpoint is approximately 0.511166/0.460508 there. The other three observations have Mask-reference endpoint IoU >0.5. These endpoint facts are not a new final-B measurement, an unaugmented-box diagnosis, or full-dataset prevalence. The revised note correctly avoids that stronger classification (`RETAINED_PV_PROBLEM_NOTE.md:38`).

**Prediction range.** The actual selected gates have median approximately 0.255170/0.258063/0.266780, maximum 0.425905/0.425163/0.426653, and standard deviation 0.049336/0.049753/0.049813 across x/y/z. Six axis coefficients are exactly zero; none is one. This is an observed prediction distribution. B's parameterization still permits `[0,1]`; the saved distribution does not identify an optimization cause, saturation problem or learnability guarantee.

At 0.5, 348 rows have a qualified native endpoint and unqualified Mask endpoint, with 261 still unqualified after B. The reverse endpoint group has 734 rows, of which B makes 64 unqualified. B repairs 103 and breaks 64 relative to Mask, net +39. At 0.25, the analogous numbers are 182/96, 173/16 and 87/16. All reproduce. GT endpoint classes are diagnostic labels, not inputs to a proposed inference rule (`inspect_prediction_and_repeated_batch.py:30-69`; revised note:30-32).

**Narrative closure.** The revised note expressly states strict thresholds and float64 qualification (line 18); limits the 3,037-row impossibility to the same endpoints and existing tied-axis coefficients, leaving six-face/new-endpoint possibilities open (line 24); separates the reused eight-row batch (line 38); and places current MCLN state outside this PV audit, referring it to separately published records (line 40). These changes close the identified wording issues. The goal arithmetic in line 46 is correct: 60%/52% of 9,508 requires 5,705/4,945 integer hits, gaps 28/25 from 5,677/4,920. Current retained-checkpoint status, separate §20.376.162 contents, active training progress and outstanding broader-project work are contextual assertions, **not newly authenticated by this cache audit**. The note does not count oracle values as goal attainment or three effective contributions.

## F. Classification — real_gt, with a GT-oracle claim ceiling

Primary subtype: `offline_GT_conditioned_geometric_capacity_on_saved_predictions`. Supplement subtypes: `saved_prediction_distribution` and `saved_augmented_training_witness_diagnostic`. The targets are dataset-derived (`real_gt`), including the training transform qualification above. They are not model-generated synthetic references. The oracle geometry counts nevertheless cannot be reported as new model accuracy, deployable inference, expected learned gains or successful contribution ablations. Baseline selected-output counts are historical model-output measurements only.

## Mathematical verification

**Tied-axis knot sufficiency is valid in real arithmetic.** On each axis let `l(t)=r_l+t(n_l-r_l)` and `h(t)=r_h+t(n_h-r_h)`, with `t in [0,1]`. Positive endpoint widths imply positive widths throughout. Partition at 0, 1 and crossings of either predicted face with either GT face. Between crossings, overlap length is affine or identically zero. With the other axes fixed, predicted volume and intersection are affine in t, while union remains strictly positive. IoU therefore has the form `(a*t+b)/(c*t+d)` on each piece, whose derivative has constant sign or is zero. A maximizing coordinate can move to a knot without decreasing IoU.

The knots of one axis do not depend on the other gates. Starting with a global maximizer on the compact cube and moving its coordinates successively to their knots yields a Cartesian knot maximizer. Six padded knots per axis, 216 triples, therefore suffice; zero endpoint displacement needs no crossing, and clipping an out-of-range crossing only repeats a boundary. This proves the family characterization, not exactness of its float64 implementation or a learned network's ability to realize arbitrary per-row GT gates.

**Independent-face projection is legal and dominating over valid boxes in its endpoint intervals.** Let allowed lower bounds be `[A,B]` and upper bounds `[C,D]` on an axis. Ordered endpoint boxes imply `A<C` and `B<D`. For GT `L<H`, the projections are `p=clamp(L,A,B)` and `q=clamp(H,C,D)`. Equivalently these are medians of `(A,L,B)` and `(C,H,D)`; all three paired coordinates are strictly ordered, hence `p<q`. The projected faces are feasible convex mixtures of their corresponding endpoint faces.

For any valid candidate with positive intersection, moving a face toward its corresponding GT face within the allowed interval cannot decrease IoU. Removing an exterior slab reduces prediction volume at fixed intersection. Adding a missing interior slab changes prediction and intersection at rates `A_o` and `C_o`; with axis width s, overlap d, GT volume T, the derivative numerator is `C_o*(V+T)-A_o*I = A_o*C_o*(s-d)+C_o*T >= 0`. Positive overlap preserves face ordering along these moves. Repeating gives the projection; zero-intersection boxes cannot exceed its nonnegative IoU. Thus the projection maximizes over the valid independent-face family, which contains the tied-axis family.

All 9,508 projected boxes are inside their face intervals with positive extent; minimum extent is approximately `9.999999999e-7`, and minimum projected-IoU minus tied-axis-IoU is zero. **Arbitrary** independently predicted face coefficients need not preserve ordering: 30 rows have `max(lower endpoints) >= min(upper endpoints)` on at least one axis. That empirical check and the projection theorem do not validate a six-face neural head. No such head was tested.

## Claim impact, blockers and next diagnostic priority

- **Supported, with stated scope:** saved counts; same selected Query/endpoints; actual-B float32 reconstruction; continuous-knot and legal-face-projection mathematics; qualified reachability counts; all256 coverage; saved Mask/box table; corrected eight-row training-witness interpretation; archived direct-gradient responsibility; observed gate/endpoint groups.
- **No numerical corrections required.** The repeated-batch ambiguity is resolved by the added JSON and final note; the original result files remain unchanged. No wording blocker remains for the revised narrow diagnosis.
- **Unsupported extensions:** later E0 rowwise identity; whole-training-set unmatched prevalence; current MCLN objective/assignment status; learned selection or recovery rates; six-face architecture effectiveness; an optimization failure cause inferred solely from gate range; new model accuracy; three effective contributions; cross-dataset performance; completion of the research goals. These require their own evidence. Actual reviewer model/effort remain unattested.
- **Priority judgment: yes, qualified.** The existing tied-axis family reaches 459 strict-threshold errors, versus 93 additional strict-threshold cases gained by releasing faces within the same endpoints. Another 3,037 strict errors lack selected-axis repair while an actual saved candidate geometrically qualifies. Together with the archived objective's matching-only direct responsibility and the small witness, this supports prioritizing objective, supervision/selection and learning checks before attributing failure to coupling or enlarging the architecture. It does not establish which cause dominates, prove that expanded positives will help, prescribe validation-GT inference, or authorize changing a running experiment.

## Reproduction and trace

Run `verify_geometry.py`, then `verify_training_winners.py`, with Python 3.13.12 and NumPy 2.5.3 on CPU. The verification used the existing interpreter at `D:/Program Files/UserCache/gb/uv/archive-v0/159xpsy_Kro0-gJX4848_/Scripts/python.exe` with `-B`; no environment was installed. The main check took approximately 9.7 seconds and the supplemented training check approximately 1–2 seconds. They write only into this directory.

Evidence: `CPU_VERIFICATION.json`, `CPU_VERIFIED_GEOMETRY.npz`, `PROVENANCE_VERIFICATION.json`, `TRAINING_WINNER_VERIFICATION.json`, `INPUT_HASHES.json`, and the retained final narrative snapshot. `EXPERIMENT_AUDIT.json` is the machine summary. The complete received task and five follow-up messages, full review, exact final response, requested-versus-actual identity and input-hash manifest are recorded under `.aris/traces/experiment-audit/2026-10-12_run01/`. All input files were rehashed unchanged at finalization.
