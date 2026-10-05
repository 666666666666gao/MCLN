# Experiment audit: separate Query/Text/fused mask qualification

**Verdict: WARN. Blocking findings: none.** The saved diagnostic and proposed publication support the bounded training-GT qualification claim. They do not establish physical candidate identity, validation prevalence, or a useful new geometry loss.

Date: 2026-10-05. Fresh reviewer: `/root/pvg_mask_branch_terminal_integrity`. Requested route: `gpt-6-astra`, reasoning `max`, `fork_turns: none`; actual backend/model/effort are unattested. Review independence is **same-family** and acceptance is **provisional**.

## Independently verified result

NumPy/stdlib checks verified all 24 intake files (624,252 bytes), all eight NPZs, all 64 row records, the batch-log totals, receipt, terminal status, and observer closure. There are 60 distinct scan IDs and 60 physical-scene prefixes, with 256 candidates per row (16,384 total). Every native target-slot list is `[0]`.

| Qualification on these saved augmented training rows | Candidates |
| --- | ---: |
| Fused mask IoU > 0.5, box IoU <= 0.5, native-unmatched | 1,112 |
| Above pool and own Query mask IoU > 0.5 | 1,090 |
| Above pool, Query IoU <= 0.5, Text IoU > 0.5 | 22 |
| Above pool with both individual branch IoUs <= 0.5 | 0 |
| Query IoU > 0.5, box-poor and unmatched, regardless of fusion | 1,100 |

The 1,090 subset spans 52/64 rows and is 98.0216% of the conditional 1,112-candidate pool. Text itself qualifies on 48/64 rows. This fraction is a conditional candidate share, not an accuracy or prevalence estimate; candidates within a row are dependent. The 22 category is a branch-overlap result, not a causal claim that Text alone produced those fused masks.

Saved integer intersections/unions independently reproduce all mask qualifications using `2 * intersection > union`; maximum difference from the stored float IoUs is below 2.98e-8. Independently computed float64 box IoU preserves every box qualification; its maximum difference from saved CUDA float32 IoU is 2.3695575378512856e-6, so numerical equality is not claimed.

Against the closed D probe, row IDs, scan IDs, saved point identities and root boxes agree. There are zero native-match changes, zero Box > 0.5 changes, zero fused-mask > 0.5 changes and zero selected-query changes. The nonzero prior/current maximum Box-IoU difference is **1.0728836059570312e-6**; maximum fused-IoU difference is zero. Prior arrays were only read and remain unchanged.

Evidence: `analysis/AUDIT_CPU.json`, `analysis/audit_cpu.py`, `analysis/SUMMARY.json:6`, `complete/receipt.json:30`, `complete/rows.jsonl`, and `complete/batch_00.npz` through `batch_07.npz`.

## A. Ground-truth provenance — PASS

The loader obtains ScanRefer annotation `object_id` from the dataset and creates the target mask from `scan.three_d_objects[tid]['points']`; boxes come from the annotated object's bounding box, with the existing training augmentation. This is dataset GT in the augmented training interface, not a reference generated from predictions. The probe reads root slot zero from `batch['gt_masks']` and `center_label`/`size_gts`.

The reused factory restores official PV, then the original G delta, then the protected 4506 boundary delta, and installs a fresh zero-output R. The runner freezes all parameters and calls `eval()` after the factory. The executed source is identical to the reviewed local runner, and actual import identities match the relevant local source snapshots. No broader historical accuracy claim was re-audited.

Evidence: `pvground_g_p2_20261002/complete/source/joint_det_dataset.py:633`, `joint_det_dataset.py:1086`; `run_mask_branch_probe.py:44`, `run_mask_branch_probe.py:113`, `run_mask_branch_probe.py:132`, `run_mask_branch_probe.py:157`; `probe_body.py:43`, `probe_body.py:56`; `pvground_final_quality_20261005/runtime_bundle/readback_model_factory.py:18`, `readback_model_factory.py:39`, `readback_model_factory.py:45`; `pvground_boundary_evidence_readback.py:28`; `complete/imports.json`.

## B. Score normalization and candidate semantics — PASS

Mask IoU uses point-weighted intersection divided by point-weighted union. No model maximum or prediction-statistic denominator normalizes the metric. Text logits are a shared expansion across all 256 candidates; Query logits are candidate-specific. The fused mask uses the native logit formula `sigmoid(alpha * Text + (1-alpha) * Query) > 0.5`, matching the deployed evaluator. Saved Text counts are equal across each row's candidates.

Native matching captures the `last_` call after `proposal_`, restores native slot IDs through `valid[targets]`, and excludes all matched candidates. Native signed bbs retains main/modify/pronoun/relation evidence and subtracts other-entity evidence. Saved bbs have no top ties and reproduce each selected Query.

Evidence: `probe_body.py:27`, `probe_body.py:46`, `probe_body.py:51`, `probe_body.py:65`; `pvground_geometry_readback_20261004/revision2/source_preview/PV-Ground/models/pv_ground.py:543`; `pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py:596`; `models/losses.py:852`, `models/losses.py:872`; `pvground_final_quality_20261005/runtime_bundle/native_root_bbs.py:4`.

## C. Result existence and closed execution — PASS

All intake entries are present and match the recorded raw sizes and identities. Per-row counts, totals and summary agree. Controller and probe exit files are zero; status is complete; the saved observer records the controller as absent and closes successfully. Source and receipt establish no optimizer construction, no backward, no new weight output, unchanged model state and absent model gradients. The controller's terminal record confirms protected-parent verification.

Evidence: `complete/INTAKE.json:3`, `complete/status.json:2`, `wait.json:18`, `complete/probe.log:7`, `complete/receipt.json:16`; `probe_body.py:20`, `probe_body.py:106`; `controller.py:35`; `analyze_probe.py:24`, `analyze_probe.py:38`.

## D. Executed metric path — PASS

The generated runner actually calls the native criterion, then computes and writes each branch's counts/IoU on every batch. The eight cumulative log entries and terminal receipt match independent recounts. `BaseTrainTester.get_criterion` returns `compute_hungarian_loss`; the saved metrics are not outputs of an unused helper. AST and generation-recipe checks passed without model imports or execution.

Evidence: `run_mask_branch_probe.py:195`, `probe_body.py:13`, `probe_body.py:65`, `probe_body.py:97`, `probe_body.py:100`; `pvground_runtime_bundle_20260908_v1/PV-Ground/main_utils.py:268`; `analysis/AUDIT_CPU.json`.

## E. Scope — WARN, no blocking finding

The actual scope is one protected checkpoint, seed 2027, eight fixed shuffled augmented fit batches, 64 rows/60 scenes, all root-only. Query IoU > 0.5 is an overlap-support proxy. It neither proves physical identity nor demonstrates a nonempty multi-target responsibility case. There was no holdout or full validation evaluation, no optimizer update and no geometry-loss efficacy test.

Raw point masks were not archived. CPU independently recounts the saved all-candidate integer intersections/unions and recomputes box geometry. The selected Query's three branches were separately expanded to points and checked in the successful GPU run; that source/runtime witness is not a CPU replay of raw masks. The proposed publisher states this distinction explicitly.

Evidence: `EXPERIMENT_PLAN.md:5`, `EXPERIMENT_PLAN.md:9`; `probe_body.py:70`; `complete/receipt.json:15`; `analysis/SUMMARY.json:16`; `publish_branch.py:34`, `publish_branch.py:38`, `publish_branch.py:42`.

## F. Evaluation type — real_gt, diagnostic support proxy

GT provenance is `real_gt`. The evaluated claim is a training-GT support qualification diagnostic, not physical identity or formal accuracy. Calling it a proxy does not mean its GT was synthesized from model output.

## Proposed publisher — PASS for reviewed source, unexecuted

`publish_branch.py` requires the closed observer, CPU recount, returned audit and no blocking findings. The proposed text retains the training-only/identity/prevalence limits and labels the next geometry strategy as a draft. That future strategy was not reviewed or accepted here.

The publisher compares local/remote evidence as raw bytes, uses a data-root symlink for `complete`, and compares Git text blobs after CRLF-to-LF normalization while preserving NPZ bytes. Read-only Git clean-filter checks passed for 49 then-existing payloads in each of the two evidence repositories; all three heads and four raw handoff copies match the prior publication, and the repositories were clean. The convention is intentional: recorded intake SHA identities refer to raw collected files, not normalized Git text blobs. No Git mutation, remote call or publisher execution occurred in this audit.

Evidence: `publish_branch.py:16`, `publish_branch.py:62`, `publish_branch.py:65`, `publish_branch.py:89`, `publish_branch.py:94`; `analysis/AUDIT_PUBLISHER.json`, `analysis/audit_publisher.py`. Subsequent reviewer-report/trace files are not included in that 49-file measurement; the publisher will enumerate its final payloads when executed.

## Claim impact and action

- **Supported:** the exact bounded branch-count and coverage statements above, shared Text versus candidate Query semantics, and unchanged prior-probe qualifications.
- **Requires the existing qualifier:** “Query-confirmed” means Query IoU-qualified support on this saved training sample.
- **Unsupported by this audit:** physical identity proof, formal-validation prevalence/accuracy, multi-target runtime coverage, or benefit/acceptance of the future geometry strategy.
- **Action:** retain these qualifiers in publication. No experiment-code correction or rerun is required by the audited evidence.

Reviewed file identities and structured findings are in `EXPERIMENT_AUDIT.json`. Independent verification artifacts are `AUDIT_CPU.json` and `AUDIT_PUBLISHER.json`.
