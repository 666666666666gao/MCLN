# Isolated span runner source follow-up — R2

Date: 2026-10-09. Scope: STATIC_ONLY. Same continuing reviewer: /root/pvg_extremal_span_runner_source_20261009.
Backend/model/effort: UNATTESTED; the original R1 request named Codex/gpt-6-astra/max, which is not runtime attestation.
Review independence: same-family. Acceptance status: provisional.

**Overall/source verdict: WARN.** The new provenance arrays and gradient summaries are statically consistent with the native tensor contracts. The R1 membership-evidence omission is addressed at source level. One receipt selector still differs from the formal native selector; this limits its exact “native winner” claim. No actual tie, selector disagreement, NPZ capture, M0, gradient, restore, fit or metric was observed. No launch approval follows.

This report binds the captured paired_span_loop.py SHA256 c6b8328aee7ae7369d5636b11ce31eb06b53267af1358e4667cc3e8e88df0d5d. During report finalization, the executor announced a separate R3 revision. R2 remains a review of its preserved snapshots; it does not claim its hashes are still the latest live version.

## Preserved evidence and actual verification

INPUT_MANIFEST_R2.json contains 26 exact source/input snapshots, including the live paths and captured hashes for all 11 bound Python files, pair_spec_template.json and RUNNER_PREPARATION.json. These 13 publication-bound files are identifiable by their runner_v1 parent. r2_snapshots/ preserves the captured bytes. R2_EXACT_REQUEST.txt preserves the exact request.

R2_STATIC_VERIFICATION.json records a real exit-0 check under the already discovered Python 3.7.6 interpreter E:\\anaconda\\python.exe with -I -S -B. No NumPy, Torch or project module was imported. The verifier:
- confirmed 57 R1 artifacts/snapshots unchanged and all 26 captured R2 originals/snapshots equal at verification time;
- parsed 15 Python input files with native Python 3.7;
- matched all 11 current source hashes against both manifests;
- verified the other ten runner Python files are byte-identical to R1;
- verified the template changed only the paired-loop hash, and preparation metadata changed only that hash and time;
- proved the complete loop AST equals R1 after removing the new witness method, its preflight-only assignment/receipt field, and the gradient-summary receipt keyword;
- re-evaluated the two isolated parent predicates, which still reject PENDING before neural/child execution.

R2_LOOP.diff records the exact source delta. The R2 verification invocation succeeded first time; R1's three failed invocations and original failed verifier remain unchanged. No dependency synchronization, installation or retry of the known uv metadata failure was attempted. R2_NATIVE_TOOL_RECEIPTS.json and R2_NATIVE_VERIFICATION_ATTEMPT_1.json contain actual commands/results.

## Native shapes and indexing — PASS, static

The native implementation appends Text predictions after expand(1,256,S), whereas Query predictions are squeezed to (256,S), then corrected without changing shape (pv_ground.py:544-570; mask_support_corrector.py:24-26,49-67). Consequently the new witness correctly uses last_pred_masks[bid][0] and sp_last_pred_masks[bid] as two (256,S) tensors (paired_span_loop.py:95-108). Indexing them by observed_queries returns (K,S), not a slice of a missing singleton dimension. alpha is the native scalar.

The witness's inputs['points'][:,1:].reshape(B,50000,6) is the same layout consumed by the native support corrector/refiner (paired_span_loop.py:89; pv_ground.py:564,573). Its saved raw-point bytes are checked against the batch point-cloud bytes (paired_span_loop.py:100-102). It retains the (50000,) native SP IDs, full (256,S) foreground, (256,) reference validity, (256,) scores, all (256,3) native/Mask boxes, and each arm's (256,3,2) source fractions.

Each geometry dictionary contains actual native_ids and count, origin/span, histogram, mean/second moment and lower/upper arrays. K_present, the number of observed native SP IDs, need not equal S, the logits' native slot extent. Saving native_ids is essential to that distinction (whole_mask_range.py:12-37; paired_span_loop.py:103-114). The archive retains the dictionary rather than assuming densely occupied slots.

The archived GT boxes/masks come directly from the actual targets returned by frozen_parent_assignments. Those targets select batch GT using box_label_mask; no predictions generate them (matched_mask_objective.py:6-30; paired_span_loop.py:129-130). matched_gt_ids index rows of the saved valid-target arrays; they are not asserted to be original ScanNet object IDs. Scene/training row IDs and the per-NPZ hash bind the record and its GT contents. GT is written as provenance only and does not enter the mixer.

All .numpy() calls on frozen-parent data occur on no-grad outputs. The new arm source-fraction copies explicitly detach. No new parent/mixer forward, target mutation, parameter mutation or random sample occurs in the witness.

## Membership, extrema, ties and invalid support — source coverage PASS

When actual files exist, the saved fields suffice to reconstruct the requested member selection without a neural replay:

1. Use raw_points[:,:3] and superpoints to recompute the observed native ID partition, counts and member statistics with the native float64 statistics path; compare with member_*.
2. Index foreground_sp by member_native_ids. Keep this unmasked foreground when distinguishing absent support from present-but-degenerate support. It is saved before reference_valid is applied (paired_span_loop.py:97,103-105).
3. Recreate member coordinates as origin + normalized_bound * span, then cast to the native prediction dtype before extrema/equality checks, as mask_reference.py:20-24 and extremal_span_mixer.py:45-53 do. Reducing uncast float64 coordinates would not reproduce the specified native comparisons.
4. Empty support means no present foreground members. Degenerate support means some foreground exists but high>low does not hold on every axis. Compare the resulting validity with reference_valid.
5. Apply saved validity for the mixer. Whole-support selects all active members; extremal-support selects every member exactly equal to the relevant lower/upper extreme. Count all tied SP sources and their actual point-member counts, then recompute each source fraction against the archived (256,3,2) arrays (extremal_span_mixer.py:64-90).

This is a statement about the source's planned evidence sufficiency, not an executed reconstruction or observed tie/empty/degenerate count. No witness NPZ has been generated by this audit.

The raw-logit subset is explicit: observed_queries is the sorted union of all original matched Queries and the witness-selected maximizer, and raw_logits_scope says so (paired_span_loop.py:93-94,106-108,116-121). All256 refers to foreground/members/boxes/scores, not all256 raw Text/Query logits. The subset can check its own fusion against foreground. Outside it, the future audit can reconstruct membership/extrema from archived binary foreground, but cannot independently rederive every foreground bit from unsaved raw logits. The source does not claim otherwise.

## Gradient interpretation and execution isolation — PASS, static

The four new summaries cover query_projection, support_projection, face_encoder and axis_decoder. Each is the sum of its constituent parameter L2 gradient norms, not the L2 norm of a concatenated group (paired_span_loop.py:151-158). They are observations for later inspection. No new assertion says every group is positive, and no current evidence claims that outcome. The existing zero-first-step/positive-second-step aggregate checks are unchanged. The complete per-parameter norms remain available.

The sole support_input_witness call is conditional on preflight=True, before the loss/backward, and its return value is only attached to the receipt (paired_span_loop.py:126-180). Normal fit gets support_input_witness=None in its log; the only extra fit effect is this null receipt field. It does not write support archives or perform the new score/membership work. Formal evaluation never calls it.

The matcher/objective, paired forwards, cloned initialization, separate optimizers, updates, full1328 restore contracts, RNG restoration and formal9508 logic are unchanged by executable AST comparison. The neural source itself is unchanged. The capture adds actual future M0 I/O and therefore may affect measured diagnostic elapsed time; it is not a fit algorithm change.

## Remaining finding

**W_R2_1 — receipt selector is a different operation from the native formal selector.** paired_span_loop.py:93 uses argmax(), while formal selection uses argsort(descending=True) followed by ranked[0] at lines310-311. The former guarantees a maximum-scoring Query; exact identity with the latter's tied winner is not established by source equality. If the actual maxima are unique they coincide. No actual score tie or disagreement has been observed or simulated.

This affects the receipt's exact “native winner” description and whether that precise formal winner is guaranteed to be in the raw-logit subset. It does not change training, inference, all256 foreground, or matched-query inclusion. The minimal source alignment is the existing formal expression scores[bid].argsort(descending=True)[0] in the new receipt; no branch, fallback, tie policy or gradient change is needed. This is a source-contract warning, not a claimed runtime failure. The announced R3 change is outside R2's verdict.

R1's full-membership recording omission is otherwise resolved in source; raw named gradient interpretation is clarified by explicit group observations. The stale historical proposal wording and pending launcher/output-directory provisioning noted in R1 remain preparation limits, not new neural defects.

## A–F integrity disposition and claim ceiling

- A / GT provenance: PASS static; only actual batch GT is archived, with no new inference GT route.
- B / score normalization: PASS static; original bbs scoring and no own-output metric normalization. Exact winner identity has the limited warning above.
- C / results: PASS as source preparation only; no witness archive or experimental result is represented as existing.
- D / called paths: PASS static; the new helper is reachable only in preflight and has no neural replay.
- E / scope: WARN; no actual B8 M0, gradients, reconstruction or formal9508 result.
- F / type: STATIC_ONLY_SOURCE_FOLLOWUP; future experiments remain real_gt, not executed here.

Actual backend/model/effort remain UNATTESTED. This is the same reviewer continuing R1, not a new independent context or a different model family. No current experiment, parent decision, GPU reservation, checkpoint retention or global note was changed. Source assertions and planned receipts do not approve an M0 run, deployment, accuracy claim or launch.

