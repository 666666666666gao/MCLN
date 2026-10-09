# Isolated span runner source audit — R1

Date: 2026-10-09. Scope: STATIC_ONLY. Reviewer: /root/pvg_extremal_span_runner_source_20261009.
Requested route: Codex / gpt-6-astra / max. Actual backend, model and effort: UNATTESTED.
Review independence: same-family. Acceptance status: provisional.

Overall verdict: WARN. Core source verdict: PASS within the static scope below. No concrete blocking defect was established in the native call, paired optimization, objective, checkpoint identity or formal metric paths. This is neither an actual M0 result nor execution/deployment/launch approval. The pending parent template must remain rejected.

## Scope and evidence handling

I read the supplied request and all 11 bound runner Python files, the three runner metadata files, the proposal and M0 plan, the preceding source review, the preceding native entry/spec/actual-preflight review, the native PVGround/loss sources, and the two restore/readback helpers. I additionally inspected the local copies of the native evaluator, bbs helper, member-statistics provider, boundary decoder, dataset target builder, criterion constructor and entry preparation helper. Prior review verdicts were historical context, not substitute runtime evidence for this runner.

INPUT_MANIFEST.json and snapshots/ preserve 38 exact input files, including the exact request and governing audit policy. EXACT_REQUEST.txt is the unmodified request. All 38 original files still matched their snapshots when the successful verifier ran. Every one of the 11 current runner hashes equals both pair_spec_template.json:new_runner_files and RUNNER_PREPARATION.json:files. The source tree was not edited. No main MEMORY, credentials, SSH/GPU query, Torch import, project-module import, model simulation/build, optimizer update, checkpoint load, package installation or experimental launch occurred.

The current experiment's GPU reservation was respected as instructed; its live status was not independently queried. The reviewed sources remain sealed. The preparation metadata correctly says the runner has not been deployed or executed (RUNNER_PREPARATION.json:2,24-32).

## Actual local verification, including failures

1. Offline uv discovery returned E:\\anaconda\\python.exe for Python 3.7.
2. The first uv run failed before the verifier ran: existing pyodbc-4.0.0_unsupported.dist-info has an invalid version. Exit 1 is preserved in NATIVE_VERIFICATION_ATTEMPT_1.json.
3. The second uv run added --no-sync. uv warned that it has no effect with --no-project and failed on the same metadata. Exit 1 is preserved in NATIVE_VERIFICATION_ATTEMPT_2.json. No package repair was attempted.
4. The exact discovered interpreter was invoked directly with -I -S -B to exclude site packages. That invocation exposed a reviewer-only AST name-walker TypeError. Its exit 1, original verifier source and traceback are preserved in NATIVE_VERIFICATION_ATTEMPT_3.json and VERIFY_ATTEMPT_3_ORIGINAL.py. This was an audit-script failure, not a runner failure.
5. After correcting only the review-folder AST helper's representation of non-name call roots, the same isolated Python 3.7.6 invocation exited 0. STATIC_VERIFICATION_R1.json and NATIVE_VERIFICATION_ATTEMPT_4.json preserve the actual result.

The successful checks parsed 23 Python input files under native Python 3.7, compared all input snapshots, checked the 11 bound hashes and supplied dependency hashes, bound native/wrapper signatures using AST-derived inspect.Signature objects, evaluated only the two static parent-selection predicates, compared copied protocol ASTs and spec fields, reconstructed the entry text in memory, and counted the declared Linear parameters. No experiment module was imported or executed. Successful deterministic checks do not promote the semantic review beyond same-family/provisional and do not establish neural behavior.

## A. Ground-truth provenance — PASS, static

The native dataset target builder derives object masks from scan.three_d_objects[tid]['points'] and boxes from scan.get_object_bbox(tid); train-only target jitter is preserved (joint_det_dataset.py:1086-1119). The entry imports the manifest-bound native dataset, verifies its source manifest/superpoints/split, reconstructs the 29,778 fit and 6,887 holdout IDs, and checks physical-scene disjointness (run_span_pair.py:50-58,98-116,149-220). The separate formal branch loads split='val', asserts 9,508 rows and uses no optimizer updates (run_span_pair.py:180-205; paired_span_loop.py:315-320).

FitDataset, FormalDataset, loader, prepare and box_iou are executable-AST identical to the supplied prior native entry. The actual train fixture path, B8 batch index, row order, point hash and noisy root-box checks are preserved; augmentation is enabled for fit/M0 and disabled for formal (paired_span_loop.py:321-334,232-235). These are future assertions, not evidence that the new fixture has run.

The mixer receives predictions/features/member geometry, not dataset GT or the GT-conditioned offline diagnostic. Model inputs are built only from points/voxels/text/SPs/detected inputs; GT stays in the separate batch used for matching and evaluation (run_span_pair.py:237-250; extremal_span_mixer.py:29-120).

native_mask_targets retains all valid GT instances. frozen_parent_assignments calls the original matcher once on the frozen parent before either arm's loss (matched_mask_objective.py:6-30; paired_span_loop.py:85-94). The criterion uses native HungarianMatcher(1,0,2,soft_token) (main_utils.py:268-280). No extra positive Query or GT-derived inference box is introduced.

matched_span_loss calls the native loss_boxes with the shared assignment. It preserves center L1 plus half-weight size L1, GIoU, valid-GT normalization, coefficients 10/2 and division by seven (matched_span_objective.py:5-17; losses.py:518-559,831-844,849-955). In the supplied dataset contract, valid targets fit within the native 256 Query capacity; there is no denominator based on prediction quality.

## B. Score normalization and selection — PASS, static

native_root_bbs.py:4-9 matches the supplied official position-alignment reduction: semantic softmax, root token evidence, modifier/pronoun/relation additions and other-entity subtraction (grounding_evaluator.py:217-303). This is the existing native score, not normalization by the model's own maximum/mean accuracy.

The span head changes only center/size. The original semantic output and both Mask object lists remain identical objects (span_refinement_model.py:8-21; paired_span_loop.py:68-83). All 256 Queries remain. source_mode changes only the support-membership selection branch at extremal_span_mixer.py:69-72. Member-count normalization is feature pooling, not metric rescaling.

Formal evaluation selects one native-bbs Query and uses it for every Box control and the fused Mask (paired_span_loop.py:245-284). fixed_half is exactly the predeclared 50% formula. There is no coefficient sweep, own-output-normalized score, new ranking head, candidate pruning or multiseed loop.

## C. Result existence and attribution — PASS for preparation; runtime unavailable

The 11 prepared sources and declared hashes exist. No actual M0, fit endpoint or new full9508 result is provided or claimed by RUNNER_PREPARATION.json. The preceding PREFLIGHT_ACTUAL_REVIEW.json describes a different 27,841-parameter Mask pair; it cannot witness this 29,793-parameter span pair.

The historical ENTRY_PROVENANCE.json hash is not the current hash. Deterministic reconstruction verified the distinction exactly: its nine replacements produce a40dfae3b1cd084c39b516a022545123e4cf77297f1c91364869d5454073639d; the current ten-replacement helper additionally replaces the hardcoded starting score with candidate_gate_hits=[5658,4850], producing the currently bound ab44f57fd062c6a5ba03da1740a3a1e000523321e020bb8b48509b3fb90eba2c. See ENTRY_FROM_PRIOR.diff and HISTORICAL_ENTRY_TO_CURRENT.diff. Neither text reconstruction executes the entry.

The template's parent_selection_status remains PENDING_PRIOR_PAIR_CLOSURE_AND_RETAINED_PARENT_DECISION_DO_NOT_LAUNCH (pair_spec_template.json:58). Both source predicates reject that value: run_span_pair.py:43 before Torch import, and span_controller.py:18 before child creation. The audit evaluated those isolated predicates; it did not run either entry. No finalized pair_spec.json or launcher/deployment was added.

## D. Called paths, paired independence and restoration — PASS, static; actual behavior pending

The native PVGround and wrapper both use forward(self, inputs). SpanRefinementModel calls parent(inputs) once (span_refinement_model.py:41-44; pv_ground.py:355). forward_pair calls observed_readback_forward once, then applies both heads to that one returned cache (paired_span_loop.py:68-83). The helper performs model(dict(inputs)) and checks original semantic-head order/call count (readback_preflight_checks.py:19-57). The native source routes support correction, Mask-reference boxes, zero-output readback and the deferred original semantic head in that order (pv_ground.py:560-589).

The parent is reconstructed from the declared official/G/selected zero-output Mask-reference/support dependencies, with strict load/key/shape contracts (selected_mask_reference_factory.py:13-71; mask_support_model_factory.py:9-30; run_span_pair.py:119-139). All parent parameters are frozen; exact parent state, including buffers, is compared against the initial 1,314-state snapshot. No parent parameter belongs to either new optimizer (paired_span_loop.py:38-66).

The declared seven Linear layers total 29,793 parameters and 14 state tensors. The two instantiated heads clone identical initial values, assert independent parameter storage and create separate fresh AdamW optimizers (paired_span_loop.py:41-57). Backward checks the other arm and parent have no gradients; the preflight snapshots the other arm's values and checks them after each update (paired_span_loop.py:90-130).

The two-step path checks same-cache neutral Box equality at step zero, finite losses/parameter gradients, nonzero output.weight gradient, zero first-step upstream aggregate and positive second-step upstream aggregate. It differentiates loss directly with respect to final center/size and requires every unmatched output gradient to be zero (paired_span_loop.py:78-82,95-119,334). See the evidence-coverage qualification below.

The small checkpoint stores source_mode, all mixer state, optimizer, step and exact parent identity; load checks mode/identity before strict head loading (span_refinement_model.py:47-61). The runner adds spec hash, complete seen-row list and four RNG families (paired_span_loop.py:136-144). M0 serializes this small payload in memory, rebuilds the parent on CPU and verifies all 1,328 composed state tensors; it does not serialize a standalone full parent checkpoint. Both actual CPU and GPU optimizer restore comparisons are wired, including all keys/moments/steps/groups (paired_span_loop.py:166-199; whole_model_preflight_checks.py:30-44).

The integration path calls deployed(dict(inputs)) and captures the actual parent output. It then compares restored and updated heads on that same captured cache. Original semantic/parent/mixer call counts and Mask identity are checked (paired_span_loop.py:200-228). It explicitly disclaims cold-GPU-parent reconstruction and repeated-backbone bitwise identity. This matters because the native Gumbel sampling remains stochastic in eval (pv_ground.py:610-624).

Formal loading requires step 3723, arm/source_mode, exact spec, exact fit-row multiset, parent identity and optimizer step states (paired_span_loop.py:152-164). The fit path checks 3723 updates and all 29,778 rows exactly once before writing its completion receipt (paired_span_loop.py:343-361). Actual serialized restoration and optimizer states remain untested.

## E. Formal evaluation scope and failure preservation — WARN: future execution only

The source routes formal restore into evaluate. For each batch it saves row IDs, dataset root GT, native scores and all256 boxes for native, Mask, fixed-half and both learned arms. Raw row records include the selected Query, selected boxes/IoUs, point hash and oracle diagnostics; the latter are not used as a deployed score (paired_span_loop.py:232-290).

After exactly 9,508 rows, the learned-arm strict >0.25/>0.50 hit totals must equal the official evaluator's bbs Top1 counts. The raw bbs Mask-IoU sum is checked against mask_pos, with the stated 1e-3 total tolerance; native/fixed controls are recounted from raw rows (paired_span_loop.py:291-313). The evaluator file hash matches the template. Its extra bbf diagnostics do not replace the declared bbs primary metric. Actual counts and independent archive recounts do not exist yet.

The controller uses exclusive child-log creation, records the actual exit file immediately after child.wait, writes failed status and raises before entering success bookkeeping (span_controller.py:47-68). It has no retry loop or automatic successor. Checkpoint save uses a temporary file and rename; this runner contains no deletion/retention action (paired_span_loop.py:146-150). New small checkpoints remain consumer-protected under the stated end-audit/retention decision.

Resource exclusion is an unimplemented future launcher responsibility, not something this source audit approves. The two arm output directories must be provisioned by that eventual preparation/launch step or by a simple explicit setup before fit: save writes output/<arm>/terminal.pth.tmp, while this loop creates only its formal directory (paired_span_loop.py:146-150,236-237). No completed launcher contract was supplied, so no self-contained deployment readiness is claimed.

## F. Evaluation classification — WARN: no actual model evaluation

This review is STATIC_ONLY_SOURCE_AUDIT. The proposed training and formal evaluation use real_gt. The separately described validation-GT least-squares analysis is a real-GT-conditioned offline representational diagnostic, not a deployed metric, training-label dataset or IoU upper bound. No diagnostic output is consumed by this runner.

Executed scope: local stdlib AST/signature/hash/text/metadata verification only. Neural forwards=0; optimizer updates=0; actual checkpoint saves/restores=0; formal rows evaluated=0; new weights=0.

## Nonblocking findings and preparation limits

W1 — The future preflight receipt does not alone establish every M0 detail promised in M0_AND_CONTROL_PLAN.md:17-28 and METHOD_PROPOSAL.md:46. It records all individual parameter gradient norms, but the automatic second-step gate checks only their combined upstream sum; it can pass without proving each query/SP projection and face-encoder group is nonzero (paired_span_loop.py:109-119). It also records invalid-reference totals and source-fraction minima/maxima, but not actual native SP mapping, empty-versus-degenerate membership or per-face extremal ties. Those details are available transiently in member geometry/mixer selection, then absent from the written receipt (extremal_span_mixer.py:42-88; paired_span_loop.py:113-119,131-134,336-341).

The per-parameter raw norms are sufficient for that later group-by-group inspection: absence of separate positivity assertions is not, by itself, a required source repair. The remaining concrete receipt omission is the source-membership/tie evidence, which the current raw norms cannot reconstruct.

This is an evidence-coverage limit, not proof that a gradient or selection is wrong. Before calling the complete planned M0 witness satisfied, the minimal extension is to record the actual source-membership/tie statistics and fixture/GT identity, and inspect or gate the recorded norms by the named upstream groups. Do not claim these witnesses from the current aggregate flag. No algorithm change, fallback or compatibility layer is indicated.

W2 — METHOD_PROPOSAL.md:48 still says the runner has not been made. That statement predates the now-present isolated runner. The current RUNNER_PREPARATION.json correctly distinguishes prepared source from deployment/execution. A later documentation update should preserve that distinction; this review did not rewrite the historical proposal.

W3 — Exact frozen tensors, B8 capacity, first/second-step neural gradients, real CPU/GPU optimizer restore, RNG restore and full formal counts remain deferred requirements. Source assertions are not executed witnesses. The current template and absent launcher deliberately prevent treating preparation as launch-ready.

There are no required changes to the core neural computation established by this audit. There is no permission to finalize the parent, modify the active experiment, bypass the pending gate or start M0.

## Claim impact

- “The isolated native runner is prepared and statically consistent with the bound sources”: supported within this audit's scope.
- “The two arms have equal declared capacity and differ in support-source selection”: statically supported; actual tensors/storage/gradients still require M0.
- “Full planned M0, actual restored model equivalence, capacity or optimizer integrity passed”: unsupported by this review.
- “Accuracy improved, targets5658/4850 were met, or three contributions are effective”: unsupported.
- “Fixed50% is a new method contribution”: not supported; it is the labeled prior control.
- “The parent is finalized, the GPU is available, or launch is approved”: false for this preparation state.

Full raw reviewer text: RAW_RESPONSE_R1.md. Machine record: SOURCE_REVIEW_R1.json. Deterministic result: STATIC_VERIFICATION_R1.json. All native tool receipts, including failed invocations, are retained in this review folder.

