# Readback first-preflight failure review

Date: 2026-10-04, Asia/Shanghai. Verdict: **FIX_REQUIRED — preflight harness; runtime acceptance remains unproven.**

This is a fresh same-family rescue/source review with provisional acceptance status. The requested Astra/max backend and effort are not independently attested by this reviewer. No remote call, model execution, launch, credential access, training, or primary-source edit was performed. Only the two requested review reports are written.

## Observed outcome

The collected controller is closed, failed with exit 1, and finished at 22:18:31+08:00. Its sole completed phase is evidence_hidden/CPU. CPU construction and strict loading passed: 1072 G replacement tensors, 10 protected geometry states / 456102 parameters, and 23 new readback states / 96672 parameters.

The first hidden-arm GPU preflight raised `AssertionError: last_center` at `run_readback_preflight.py:200`, inside `fixed_geometry_and_masks` at `readback_preflight_checks.py:35`. The optimizer is constructed at runner line 209, so this attempt performed **zero optimizer updates**. The zero-query equality at lines 202–203, both training steps, strict optimizer restoration, and final PASS receipt were not reached. The visible arm was not reached. There are no native accuracy rows or new disk weights in the collected intake.

The available trace supplies neither the differing values nor their magnitude, nonfinite status, input mutation witnesses, or a repeated disabled-readback baseline. It establishes inequality, not its cause.

## Blocking finding: the comparator conflates two requirements

At runner lines 191–200 the code runs the entire model with readback disabled, resets RNG, and then runs the entire model again with readback enabled, using the same mutable `inputs` dictionary. It demands bitwise equality between these separate realizations. Lines 212–215 repeat this cross-forward requirement during training.

That check combines (1) repeatability of the inherited entire CUDA pipeline, (2) any effects of reusing its input dictionary, and (3) whether R changes protected outputs in its own forward. A failure cannot distinguish them. This is an observed preflight blocker, not proof that R moved a box. Keep the repeated-forward evidence as a repeatability diagnostic and replace the R-isolation acceptance test with a strict same-forward witness.

The source supports the intended causal boundary:

- Native contrastive projections are computed before R (PVGround lines 501–504).
- Both native Mask paths and adaptive weights are computed before the frozen refiner (lines 523–557).
- The refiner writes final center/size before R (lines 559–565).
- R receives these already-created outputs and returns a semantic Query residual; only final semantic logits are subsequently assigned (lines 567–574).
- The final prediction head defers only semantic computation; center/size use the same geometric features (modules.py lines 135–179).
- R uses out-of-place `semantic_query + residual`, zero-initializes its output weight/bias, and contains no geometry/Mask assignment (readback module lines 27–29, 33–75).

No direct geometry-writing defect was found in the reviewed R implementation. This source finding does not substitute for its failed real preflight.

## Cause boundaries

The matched deployed source calls `scatter_mean` for superpoint coordinates (PVGround line 430); its matched helper implements the reduction through `scatter_add_` (scatter_util.py lines 17–55). The source also calls `F.gumbel_softmax` even in eval mode (PVGround lines 597–610); the runner already resets Python, NumPy, CPU Torch, and CUDA RNG before each comparison. The backbone receives the original input mapping (PVGround line 286).

These are concrete paths to inspect. **Neither CUDA reduction nondeterminism nor input mutation is proven to have caused this failure.** Nor does the trace prove a harmless numerical scale: NaNs are not excluded by `torch.equal` failing. The declared inherited runtime is Python 3.7.11 / Torch 1.10.2+cu111; this review did not import it.

The local pv_utils.py snapshots inspected while tracing the input mapping do not match the collected deployed source-port identity. Their writes cannot be presented as proof of what this deployed run did. The matched pv_backbone.py and scatter_util.py identities were checked against the existing collected manifest; no new identity framework is proposed.

## Minimal real diagnostic before another acceptance claim

Use the same failed batch, all 256 candidates, existing provider/weights/modes, and seed 2027. No optimizer or model change is needed for this diagnostic.

1. Retain a pristine copy of the original raw input fields assembled at runner lines 186–190. For each diagnostic forward, use a new dictionary and cloned tensor fields from that pristine fixture; copy the text list. Snapshot the original field values before and after a forward, and report new mapping keys separately. Added feature keys are not by themselves evidence that raw point values changed.
2. Run the disabled-readback model twice, A0 and A1, with the same reset RNG and identical frozen state. Record exact equality, shapes/dtypes, differing element counts, finite/NaN/Inf counts, and finite maximum absolute difference for each protected field. Compare already-available intermediate witnesses: source features/seed coordinates, query coordinates/features, superpoint-coordinate lists, coarse boxes, masks, boundary logits/evidence, and final boxes. Name only the first *captured stage* that differs, not an unobserved first kernel.
3. Run one enabled, zero-initialized R forward Z with the same pristine input and the same-forward witnesses below. Persist the diagnostic JSON before any acceptance assertion; avoid another trace containing only the first key.
4. If A0 and A1 differ with R absent, the cross-forward premise has an actual counterexample. Do not attribute it to one kernel without localization. If original input leaves change, report their exact names and values/difference summaries; an additional reused-input control is justified by that evidence. If A0/A1 are exact but Z differs across runs, inspect where the paths diverge, including semantic-head deferral, instead of labeling the discrepancy benign.
5. If the evidence localizes to superpoint coordinates, a small isolated repeat of the actual matched `scatter_mean` on the captured unchanged coordinate/index tensors is a targeted next diagnostic. Do not replace the inherited kernel, change precision, add a fallback, or relax a threshold merely to make this preflight pass.

All cross-forward differences must remain recorded. A successful same-forward R check is not a claim that the whole inherited pipeline is bitwise deterministic.

## Correct strict zero-init and fixed-geometry acceptance check

Implement this in the preflight observer/helper and its caller. The reviewed model algorithm needs no change for this correction.

- In the R forward-pre-hook, capture `semantic_query.detach().clone()` and independent cloned snapshots of the existing protected end_points fields. Use these seven tensor fields: `last_center`, `last_pred_size`, `p3_coarse_center`, `p3_coarse_size`, `boundary_logits`, `whole_mask_range_evidence`, `last_proj_queries`. Clone every tensor in `last_pred_masks`, `sp_last_pred_masks`, and `adaptive_weights`. A reference to the mutable dictionary or a detached alias is not a snapshot.
- Keep the existing actual call sequence assertion: refiner, R, native final semantic head. Assert one native final semantic-head call in each real model forward and eval mode for frozen parents/BN/Dropout.
- Capture R output and the native semantic head's real input/output. At zero init, require finite tensors and exact equality of the captured original semantic Query and R output. Require the real head input to equal the correctly transposed/contiguous R output, and its captured output to equal the published `last_sem_cls_scores`.
- For a direct zero-start score comparison, after the actual forward and after removing its counting hooks, evaluate only the existing frozen native head on the captured original Query. Compare those diagnostic replay logits exactly with the actual Z logits. Label this explicitly as a separate isolated diagnostic head replay; the deployed/model-forward head is still called once. It does not rerun the backbone, Gumbel sampling, masks, or geometry. Failure remains a failure; do not silently downgrade to a tolerance.
- After each full model forward returns, compare the saved pre-R geometry/evidence/contrast/Mask snapshots to that same forward's final output with `torch.equal`. Preserve full B×256 center/size and candidate ordering and every Mask entry, not only the selected box or a sample of queries. Perform this at zero init and for each of the two update forwards.
- Keep the existing frozen-state equality against `initial`, trainable-only parameter set, absent upstream gradients, isolated CE+G gradients, native all256 bbs score/rank/logit-gradient witness, two AdamW steps, and strict in-memory model/optimizer restore. No acceptance of these currently unreached checks is granted here.

This separates an exact causal invariance test from inherited forward repeatability; it does not remove the geometry requirement or accept a nonzero geometric discrepancy from R.

The older B preflight at `run_boundary_fit.py:383–400` already uses captured adapter arguments for a same-input replay before its separate full-model comparison. That is a source precedent for isolating a zero-output property. Its earlier success is not evidence that the current protected, trained provider must produce identical values on every repeated full-model forward.

## Evidence and validation limits

The five principal root helper/runner files are byte-identical to their runtime_bundle copies and match the collected hidden spec. Both source_preview model files match the collected source port, and pv_ground.py also matches the actual imports receipt. Checks were local reads and existing-source identity comparisons only; no runtime/tensor test was run by this reviewer.

Still unproven: the cause and magnitude of the first difference; original input mutation; repeated disabled-forward determinism; same-forward all256 geometry/Mask preservation in the real runtime; zero-start semantic equality; actual training gradients and two updates; both arms' successful preflights; restoration, memory and timing results; any accuracy improvement. Review of a changed implementation must be followed by actual diagnostic/preflight receipts.

Exact experiment-artifact read inventory, including partial and rejected source reads, is in the accompanying JSON. Session instruction/context files are not experiment evidence.

