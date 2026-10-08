# Experiment audit — actual closed warm support pair

**Overall verdict: WARN. Integrity status: warn. Blocking findings: none.**

The archived closed run supports the reported **5599/4859 Box hits out of 9508** for both trained arms. An independent offline CPU recount verified every archived byte/hash and recomputed all **14,604,288** candidate Box IoUs across the initial and terminal formal stages. It found **zero selected-threshold or saved-oracle label disagreements**. This supports retaining content as a tied metric-best trained candidate under the historical rule; it does not establish a benefit from the signed-log geometry treatment, three effective contributions, or completion of the current joint target.

Date: 2026-10-08. Reviewer: fresh native Codex agent `/root/pvg_compressed_support_closed_actual_audit`. Requested routing: `gpt-6-astra`, reasoning effort `max`. Actual backend/model/effort: **UNATTESTED**; requested routing is not host attestation. Review independence: **same-family**. Acceptance: **provisional**. Execution scope: **ACTUAL_CLOSED_TRAINED_PAIR**. No SSH, neural forward, optimizer update, checkpoint deserialization, source mutation, promotion, cleanup, or publication was performed by this auditor.

Path abbreviations below are exact local directories:

- `W` = `C:/Users/gb/.codex/tmp/pvground_compressed_geometry_support_20261008`
- `P` = `C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground`
- `H` = `C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle`
- `D` = `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py`
- `N` = `C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2`

## A. Ground-truth provenance — WARN

The active ground truth is dataset-derived, not generated from the evaluated predictions. The imported dataset loads ScanRefer scene/object annotations (`D:585`, `D:634`), builds point-instance masks from `scan.three_d_objects[tid]['points']`, and obtains target boxes through `scan.get_object_bbox` (`D:1086`, `D:1099`, `D:1105`). The evaluation loader uses the validation split, disables augmentation, and fixes 9508 row IDs (`W/run_mask_support_pair.py:175`, `W/run_mask_support_pair.py:189`). The root GT is copied from `center_label[:,0]` and `size_gts[:,0]` into both the NPZ and row archive (`W/paired_support_loop.py:304`, `W/paired_support_loop.py:324`). All 9508 row IDs, scan IDs, target IDs, root boxes and point hashes align exactly across the historical retained model, initial formal pass and final formal pass.

The actual imported dataset is the detection-aligned source with SHA256 `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d`, matching `W/complete_fit/imports.json:16`. The dataset file inside P has a different hash and is not the imported dataset. Independent AST comparison found identical target-box/mask, scene-object and token-map implementations, but a real difference in detection augmentation order: the actual imported source flips before rotating (`D:1228`), while P rotates before flipping (`P/src/joint_det_dataset.py:845`). The runner's import binding and fixtures target the former (`W/run_mask_support_pair.py:99`, `W/run_mask_support_pair.py:205`). This distinction must survive reproducibility descriptions.

The archived Scan helper traces boxes to aligned sampled instance points (`P/src/visual_data_handlers.py:84`, `:129`, `:195`, `:225`, `:246`). **This audit did not rebuild the original raw ScanNet/ScanRefer dataset, original preprocessing pickle, detector files, superpoint files, language parser, or original imported Scan helper.** The archived helper implementation was examined, but its identity with the original dataset-source helper is not separately hash-bound by the supplied imports receipt. Row identities and saved GT are internally verified, not a replacement for raw-data provenance reconstruction.

The native protocol includes annotated object/target handling and dataset-side language maps. Position alignment selects using positive/modify/pronoun/relation/other-entity maps (`H/native_root_bbs.py:4`; `P/src/grounding_evaluator.py:251`, `:281`, `:535`). Scene/auxiliary GT object data are also built by the dataset (`D:1121`, `D:1161`). Under this runner, neural inputs use detector proposals with `butd=True`, `butd_gt=False`, `butd_cls=False` (`W/run_mask_support_pair.py:189`, `:242`; `D:1189`, `D:1360`). The new support head adds no GT/oracle input; **the whole native evaluation must not be advertised as globally GT-free**.

The fused-mask extent is a model-derived **prediction reference**, not evaluation ground truth (`W/mask_reference.py:10`, `:18`, `:24`). Its absent/degenerate-support branch retains the coarse prior. The actual formal rows contain 40 invalid selected references per arm. This existing geometric validity branch is not a GT-quality gate.

## B. Scores, denominators and candidate scope — PASS

The headline denominator is the fixed expression count 9508; hits use strict `IoU > 0.25` and `IoU > 0.5` (`W/paired_support_loop.py:351`; `P/src/grounding_evaluator.py:296`). Box IoU uses intersection divided by union, and Mask IoU uses point-set intersection/union (`W/run_mask_support_pair.py:247`; `W/paired_support_loop.py:331`). No reported accuracy is divided by a maximum, mean, minimum or range of model scores.

The native `last_/bbs` score is the native language-map reduction, with no added quality score, external rank or treatment-specific score mixing (`H/native_root_bbs.py:4`; `W/support_pair_forward.py:14`). Both heads reuse the same parent's logits. All NPZ score arrays have 256 columns and all three box arrays have 256 candidates. Every saved selected Query has the maximum score; there were no top-1 score ties. Native top-5/top-10 evaluation and top-16/32/64/256 oracle diagnostics do not truncate the deployed candidate pool. The 16 nearest point members in the inherited box helper concern point support, not Query pruning.

Box and Mask are extracted for the same saved Query (`W/paired_support_loop.py:321`, `:331`, `:336`). The active training losses use actual valid GT count, majority-superpoint GT and coefficients **5/1/10/2**, without division by seven (`W/matched_mask_objective.py:35`, `:41`, `:51`; `P/models/losses.py:393`, `:411`, `:562`, `:947`). The earlier semantic-assignment helper and its normalization are not invoked by this pair.

## C. Results and archive integrity — WARN

Deterministic result checks passed. `W/complete_fit/INTAKE.json:1` contains 2423 unique files totaling **230030167 bytes**, including **2378 NPZ files**. Every listed local file has the exact declared size and SHA256; the only additional file inside complete_fit is INTAKE itself. The remote manifest and recovered intake agree. The recorded partial transfer's 2129 retained files plus 294 resumed files reconcile to the complete inventory (`W/collector_recovery/PARTIAL_CLASSIFICATION.json:8`; `W/collector_recovery/RECOVERY_COMPLETE.json:4`). The auditor did not contact the remote host or independently reenact the historical timeout.

The controller and all three fit modes have successful exit receipts and complete status (`W/complete_fit/fit_status.json:2`; `W/complete_fit/fit_controller.exit:1`). Formal log completion records match the actual JSON receipts. Both stages have 1189 contiguous NPZ batches, exactly 9508 rows and 7,302,144 independently recomputed candidate IoUs per stage. SUMMARY, receipts, saved rows and independent Box counts agree.

| Closed model/pass | Box hits >0.25 | Box hits >0.50 | Percentages |
|---|---:|---:|---:|
| Historical protected content / both initial warm arms | 5598 | 4856 | 58.876735 / 51.072781 |
| Final content | 5599 | 4859 | 58.887253 / 51.104333 |
| Final signed-log box_conditioned | 5599 | 4859 | 58.887253 / 51.104333 |
| Same-forward frozen parent, both formal passes | 5598 | 4848 | 58.876735 / 50.988641 |

The CPU float64 versus saved-native selected hit disagreement count is **0 for every stage, arm and threshold**. Saved oracle labels at 16/32/64/256 also have **0 disagreements**. The small non-top-1 score-boundary ties do not change these observed oracle labels. Full-256 terminal oracle coverage is content **8929/8157**, geometry **8931/8159**, parent **8925/8163**; these are oracle availability diagnostics, not deployed accuracy.

There is a real numerical qualification: saved native selected Box IoUs reach **1.0000050067901611**. At initialization, 107 parent rows and 108 rows per corrected arm exceed 1; at terminal, 107 rows per model do. Independent float32 arithmetic reproduces excursions, while float64 recomputation stays at or below 1. Maximum selected CPU64/native absolute difference is **6.6200876713828904e-6**. These differences do not alter any audited threshold/oracle label. Values were neither clamped nor normalized. The initial auditor script's over-strict range assertions and their correction are retained in `W/analysis/AUDIT_initial_execution.json:1`.

Mask metrics were independently **recounted from saved selected Mask IoUs only**: initial hits 5812/5126 and mIoU 47.082638%; final hits 5816/5131, content mIoU 47.086931% and geometry 47.086953%. The NPZ archives do not contain raw masks/GT masks, so **no raw Mask IoU recomputation was performed**. The native Mask computation and its runtime equality assertion were reviewed (`W/paired_support_loop.py:331`, `:354`).

## D. Executed paths, updates and restoration — WARN

All 3723 training records were parsed. Each arm saw 29778 unique IDs exactly once: 3722 batches of eight and one batch of two. These IDs and the 6887 module-holdout IDs form a disjoint exhaustive partition of the 36665 archived training-row ID space. Every record has one frozen-parent forward, one final semantic-head call, original matcher assignments, independent-head flags and zero expanded positives (`W/paired_support_loop.py:101`, `:155`, `:416`; `W/complete_fit/train.jsonl:1`, `:3723`). All logged loss components and gradient norms are finite/nonnegative. Reconstructing each loss from its four components differs by at most **8.57e-8**. Clipped-update counts are **324 content / 360 geometry**; gradient maxima are **0.359512 / 0.359576**. This checks actual records, not the full historical GPU gradient tensors.

The path is one `observed_readback_forward` followed by two heads and two independent AdamW updates (`W/paired_support_loop.py:60`, `:68`, `:107`; `H/readback_preflight_checks.py:19`). The content state is a genuine 3723-step warm start; only its unused geometry columns are zeroed and copied into the treatment. Optimizers are newly initialized. Terminals therefore have **3723 additional / 7446 cumulative support updates**, plus earlier parent training history, not a fresh 3723-update model (`W/paired_support_loop.py:35`, `:47`, `:160`; `W/pair_spec.json:98`).

GroundingEvaluator's evaluation methods, Box recounts and Mask statistics are actually called, and their outputs reach the closed receipts (`W/paired_support_loop.py:317`, `:352`; `P/src/grounding_evaluator.py:194`). Legacy functions such as `semantic_assignment_correction`, `distribution_loss`, `reference_bounds_witness`, `whole_range_loss_routes`, and `calculate_diou_3d` are not active claimed metric paths in this pair. Their presence cannot support new runtime/module claims.

The constructor's support-output zero initialization is superseded by the nonzero warm head (`W/paired_support_loop.py:46`; `W/complete_fit/load.json:8`). The inherited Box output and R output remain zero and frozen; they are not newly effective learned modules (`W/mask_support_corrector.py:73`; `W/support_pair_forward.py:16`; `H/pvground_boundary_evidence_readback.py:27`). The fused-mask geometric reference still changes the Box prediction even when the learned Box output is zero.

Actual M0 evidence covers two discarded updates per arm, 1314-state CPU reconstruction, exact same-cache native GPU head integration, and state/optimizer in-memory restore; it explicitly does **not** cover a cold GPU parent (`W/complete_fit/preflight.json:8`; `W/paired_support_loop.py:205`). Formal receipts record restored 10-state AdamW/head checkpoints and terminal hashes (`W/complete_fit/content/formal_restore.json:1`; `W/complete_fit/box_conditioned/formal_restore.json:1`; `W/paired_support_loop.py:183`). This auditor has not loaded those remote terminal weights.

The corrected postrun CPU inspector is still **pending**, not a completed result. Its RNG replay can prove equality only between its own cold CPU constructions (`W/postrun/inspect_closed_terminals_rng_replay_authorized.py:59`, `:81`, `:96`). It does not attest the uncheckpointed zero-R hidden initialization of the historical formal GPU run or 9508-row cold GPU inference. `W/STORAGE_SOURCE_REVIEW_RNG_REPLAY.json` is explicitly SOURCE_ONLY and was not used as runtime evidence.

## E. Comparison scope, selection and cleanup — WARN

The declared formal scope is **9508 expressions / 141 scans / 141 physical scenes**, one seed 2027, two paired continuation heads. The 6887 module-holdout rows come from the training split and are not a new independent benchmark. No new Nr3D/Sr3D training or multi-seed evidence is established. Single seed is intentional; no extra seeds are requested.

The final same-forward content-to-geometry comparison has **0 repairs and 0 damages at both thresholds**. There are real all-candidate geometry differences (24,612 coordinate values across 3506 expressions) and tiny saved Mask-mIoU differences, so “identical selected hit labels” is warranted; “identical models” is not. The signed-log geometry treatment has **no demonstrated selected Box accuracy benefit over content**.

Each terminal versus its same-forward parent has repairs/damages **13/12 at 0.25** and **33/22 at 0.50**, net +1/+11. Historical/initial-to-final comparisons give repairs/damages **7/6 and 19/16**, net +1/+3, but they use independent passes. Between initial/final frozen-parent passes, **3 selected Queries and 40 selected parent boxes change**. Across all256 candidates, scores change at 2,387,463 values and parent coordinates at 5,092,286 values, touching all 9508 expressions; maxima are 0.501911 and 2.146236 respectively. Native Gumbel sampling remains present in eval (`P/models/pv_ground.py:610`). The data prove cross-pass drift; they do not isolate all its causes or make the historical +1/+3 wholly attributable to support training.

The immutable pair rule is historical **5620/4764 joint-gate first, then hits@0.50, then hits@0.25, ties preserve the protected incumbent** (`W/EXPERIMENT_PLAN.md:17`; `W/pair_spec.json:5`; `W/postrun/analyze_compressed_geometry_formal.py:200`). Neither trained arm passes that joint gate, but both rank above historical 5598/4856 by the next criterion. The two new arms tie; the analyzer's existing candidate order picks content. Thus content is **a tied metric-best eligible terminal**, not a unique geometry win. The rule specifies incumbent ties but does not separately prescribe ties between new arms. Retaining content is consistent with the observed ranking and simpler no-geometry deployment; do not invent a more specific predeclared new-arm tie rule.

Current user targets are separately **strictly >59.5% / >51.0%**, requiring **5658/4850 hits** (`W/CURRENT_RESEARCH_GOALS.json:8`). The best audited terminal is **59 wide-threshold hits short**, while its strict-threshold count passes. The publisher explicitly distinguishes these latest targets from historical selection (`W/postrun/publish_actual_results_cleanup_authorized.py:65`, `:69`). SUMMARY's old field name `passes_updated_dual_target` refers to the pair's historical 5620/4764 calculation (`W/postrun/analyze_compressed_geometry_formal.py:145`); it must not be reinterpreted as the new goal or retroactively rewritten to change selection. Three effective contributions and the full joint goal remain unestablished.

Pending cleanup is bounded to two exact remote checkpoints (the tied unselected geometry arm and superseded old content copy) and 2378 hash-matched locally archived remote NPZs (`W/postrun/retire_closed_nonbest_authorized.py:24`, `:59`, `:65`). The old warm-start local archive was independently hashed and matches its recorded 446789 bytes and SHA. The final factory accepts full support delta without loading that old checkpoint (`W/mask_support_model_factory.py:9`); the pair runner's warm-start constructor still needs the old checkpoint for reproduction, so its local archive must remain.

The required official Scan/Nr/Sr weights, original G, selected mask-reference dependency and new retained content total six protected weights. These dependencies remain necessary even where their standalone metrics are lower (`W/CLEANUP_FOLLOWUP_20261008_1645.json:4`; `W/postrun/retire_closed_nonbest_authorized.py:38`, `:40`). Cleanup source validates exact paths, hashes, archive size/hash and completed CPU inspection before deletion. **This is a source-level safety assessment plus verified local archival evidence, not certification that cleanup/retention has happened.** Postrun inspection, new-best archival, cleanup and publication receipts were absent during this audit. No source change is required by this audit to perform the already bounded postrun chain once its actual prerequisites pass.

## F. Evaluation classification — WARN / real_gt

Primary Box and native Mask evaluation: **real_gt**, under the disclosed native dataset/object/language-map protocol. Predicted-mask extents are inference geometry, and top-k/full256 oracle coverage is a GT-based diagnostic; neither is a replacement ground-truth score. The audit's independent deterministic Box verification is accepted within its saved-data scope. Its semantic integrity verdict remains same-family/provisional. Raw Mask reconstruction, raw dataset reconstruction, terminal checkpoint loading and formal cold GPU validation were not performed.

## Claim impact and follow-through

- **Supported, with scope:** this actual single-seed closed pair produced 5599/4859 for both terminals; all saved candidate boxes and claimed selected counts reconcile; content is an eligible tied metric-best terminal under the historical rule.
- **Supported, with comparator named:** terminal correction improves +1/+11 over the same-forward frozen parent in this pass. Historical +1/+3 is an independent-pass observation, not isolated causal training gain.
- **Unsupported:** a selected-accuracy benefit from signed-log geometry, a third effective contribution, global GT-free deployment/evaluation, raw independently recomputed Mask performance, unique metric superiority of content, or completion of the updated joint research goal.
- **Pending execution:** corrected CPU terminal inspection, verified new content archive, exact authorized remote cleanup, and publication. Preserve the lower-metric restoration dependencies and old warm-start local archive. These pending tasks are not fabricated blockers in the completed-data integrity verdict, and this report is not their completion receipt.

## Evidence and exact input binding

`W/analysis/AUDIT_independent_cpu.py` and its actual `AUDIT_independent_cpu.json` contain the independent full archive/array/training checks. `AUDIT_source_checks.py/json` binds imported sources, preflight/log receipts and pending cleanup scope. `AUDIT_initial_execution.json` retains the auditor's two numerical-range assertion failures. `AUDIT_EXECUTION.md` records successful commands and exit evidence. `EXPERIMENT_AUDIT.json` contains every genuinely read input path and exact SHA256, with scope labels separating semantic review, complete data checks, and archive hash-only verification. Hash-only files are not represented as independently understood scientific evidence.

Designated trace: `C:/Users/gb/.codex_mcln_g0_20260905/.aris/traces/experiment-audit/2026-10-08_run07`. The executor is responsible for saving this agent's actual final response there. Requested model/effort and actual UNATTESTED identity remain distinct in the trace metadata.
