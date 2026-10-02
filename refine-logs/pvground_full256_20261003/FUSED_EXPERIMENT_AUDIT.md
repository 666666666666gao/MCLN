**Experiment audit — fused-support retry**

**Overall verdict: WARN. Integrity status: warn.** The completed retry is supported as a negative, single-seed ScanRefer development result. The saved final boxes independently reproduce **5566/4406** bbs hits at IoU > 0.25/> 0.50, versus **5594/4457** for the same-tail raw control. No fabricated GT, self-normalized accuracy, or phantom completed result was found in the audited artifacts. Exact functional-start parity, complete candidate/Mask replay, and current parent-checkpoint retention are not established by this local audit.

Date: 2026-10-03, Asia/Shanghai. Reviewer: **gpt-6-astra**, reasoning effort **max**, fresh context (`fork_turns=none`), native Codex. Agent: `/root/pvg_fused_retry_result_integrity`. **review_independence=same-family; acceptance_status=provisional.** This is not cross-family assurance.

Only local files and CPU calculations were used. No SSH, network, GPU, model forward, training, deletion, or experiment-artifact modification was performed. This review writes only `EXPERIMENT_AUDIT.md` and `EXPERIMENT_AUDIT.json`; the full Markdown is returned for parent-managed immutable tracing.

**Evidence paths**

The following aliases resolve the exact file:line references below:

- `B` = `C:/Users/gb/.codex/tmp/pvground_fused_support_20261002`
- `R` = `B/complete_tail_fused_retry`
- `W` = `B/complete_tail_raw`
- `F` = `B/failed_tail_fused_initial`
- `G` = `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/g_control`
- `A` = `C:/Users/gb/.codex/archives/pvg_tail_fused_retry_20261002`
- `O` = `C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_scanrefer_formal_20260918_semantic_assignment_v1`
- `S` = `C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_semantic_assignment_comparison_20260918_v1/summary.json`
- `H` = `C:/Users/gb/.codex_mcln_g0_20260905/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`

The machine-readable companion contains SHA-256 hashes for all **128 experimental input files**, including the four subsequently supplied original-G records. Prior audits were treated as records to check, not as proof of their own conclusions. Historical handoff claims outside the supplied experiment and its comparisons were not re-audited.

| Check | Status | Finding |
|---|---|---|
| A. Ground-truth provenance | PASS | Dataset annotations and scene object geometry/membership feed the native evaluator. |
| B. Score normalization | PASS | Strict IoU thresholds and expression-count denominators; no prediction-based accuracy normalization. |
| C. Result existence and evidence | WARN | Completed results, rows, hashes, exits and checkpoint agree; some replay and parent-retention evidence is unavailable locally. |
| D. Live versus dead code | WARN | Reported REC/Mask metrics are live; an inherited CE/gradient witness is unreachable in this run. |
| E. Scope and attribution | WARN | One seed, one dataset, development evaluation; amended E0 gate does not prove functional equivalence. |
| F. Evaluation classification | PASS | `real_gt`; GT-assisted candidate coverage remains an offline diagnostic. |

**A. Ground-truth provenance — PASS**

The loader reads `ScanRefer_filtered_<split>.txt/.json`, maps the annotation's `object_id` to `target_id`, and obtains target boxes and masks from the scan's object records. Boxes come from `scan.get_object_bbox(tid)`; mask membership comes from `scan.three_d_objects[tid]['points']`. These become `center_label`, `size_gts` and `gt_masks`. Evidence: `R/source/imported/src.joint_det_dataset.py:594`, `:636`, `:1086`, `:1101`, `:1105`, `:1387`. Evaluation disables augmentation before loading batches: `R/run.py:430`–`:432`.

The actual path imports the archived native `GroundingEvaluator` from an explicit runtime file, records its hash, instantiates it with `only_root=True`, thresholds `[.25,.5]`, and `filter_non_gt_boxes=False`, then calls `evaluator.evaluate(predictions,'last_')` for each batch. It asserts that saved selected-box counts equal native evaluator counters and that Mask IoU sums agree within 1e-3. Evidence: `R/run.py:116`–`:126`, `:433`, `:448`, `:493`–`:499`; `R/source/imported/evaluator.py:194`–`:206`, `:543`–`:550`, `:887`–`:901`. The imported evaluator's SHA is `39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677`, matching `R/arm/imports.json` and its archived bytes.

Predicted support is an input, not a replacement target: four channels contain Text/Query probabilities, their disagreement, and sigmoid of the native weighted fused logits. The raw arm gathers the same channels and zeros them. Forward inputs contain points, text, detected boxes/classes and superpoints; GT is attached afterward for loss/evaluation. Evidence: `R/source/pvground_tail_support_box_refiner.py:15`–`:31`, `:45`–`:53`; `R/run.py:263`–`:278`, `:439`–`:453`.

Limit: the original annotation files, scene pickle and raw point/mask arrays are remote paths, not supplied local dataset bytes. Thus provenance is supported by archived source and execution records; this review does not independently reconstruct GT from raw ScanNet files or authenticate the upstream repository over the network.

**B. Score normalization — PASS**

REC uses ordinary axis-aligned 3D intersection-over-union and strict `>` decisions at 0.25 and 0.50. Accuracies divide hits by 6887 or 9508 expression rows. Mask mIoU divides the sum of per-row intersection/union values by the row count, then multiplies by 100. The softmax/0.07 operations rank candidates; they do not normalize reported accuracy by model-output extrema. Evidence: `R/run.py:449`–`:480`, `:491`–`:501`; `R/source/imported/evaluator.py:296`–`:303`, `:485`–`:491`, `:897`–`:902`.

Independent standard-library CPU arithmetic recomputed every saved selected box for fused/raw/G-continuation and the failed E0, and every saved coarse box where present. All row-level REC threshold decisions agree with the stored IoUs. Across the fused stages and both modes, the largest continuous selected/coarse IoU discrepancy is **3.55474025e-6**; this is agreement of threshold decisions, not bitwise GPU replay. All receipt counts, Mask sums and rates match. The two existing CPU_RECOUNT files also match 120 directly compared numerical fields plus coverage, transitions and paired counts.

Both coarse and final sizes were floored at 1e-6 before recording. The audited geometry is therefore deployed evaluator geometry; positive saved sizes do not establish positivity of the unclamped head output. Evidence: `R/run.py:444`–`:452`; the preflight reports negative coarse-size elements in `R/preflight/preflight.json`. No model-output normalization is used to disguise this floor.

**C. Result existence and evidence — WARN**

All **44** fused intake entries, **42** raw entries and **9** failed-attempt entries have matching file sizes and SHA-256 hashes. Imported sources, split protocol, source port, spec, runner and training-log hashes agree with their recorded references. The two completed arms have preflight/train/formal exit codes 0. Fused training finishes at 3723 updates, terminal holdout evaluation completes, and formal evaluation completes on 9508 rows; this conclusion uses terminal records, not the launch flag. Evidence: `R/arm/train.exit:1`, `R/arm/formal.exit:1`, `R/preflight/preflight.exit:1`, `R/arm/train.log:83`, `:97`–`:98`, `R/arm/formal.log:26`, `R/status.json:2`–`:17`. The training receipt's `formal_rows:0` is a training-stage field; the separate formal receipt has 9508.

Every row file has the expected unique IDs in order: the exact 6887-row holdout list or formal IDs 0–9507. Both completed tail arms and G-continuation have exactly 3723 training records, 29778 unique fit IDs each once, no fit/holdout-ID overlap, and the same complete batch-row order. Evidence: `R/run.py:538`–`:549`; `R/arm/train.jsonl:1`–`:3723`; corresponding raw/G logs and `R/source/split_protocol.json:1`. The last batch has two rows.

The following values were independently recounted from saved predictions:

| Evaluation | Rows | bbs hits @.25/.50 | bbf hits @.25/.50 |
|---|---:|---:|---:|
| Fused initial holdout | 6887 | 6176 / 5602 | 6206 / 5647 |
| Fused terminal holdout | 6887 | 6155 / 5586 | 6159 / 5608 |
| Fused formal development | 9508 | 5566 / 4406 | 5593 / 4436 |
| Same-tail raw formal development | 9508 | 5594 / 4457 | 5597 / 4472 |
| Historical G-continuation formal development | 9508 | 5600 / 4452 | 5628 / 4465 |

Evidence: `R/arm/initial/receipt.json:5`–`:23`, `R/arm/terminal/receipt.json:5`–`:23`, `R/arm/formal/receipt.json:5`–`:23`; `W/arm/formal/receipt.json:5`–`:23`; `G/formal/receipt.json` and all associated rows.

Fused bbs formal accuracy is **58.54017669% / 46.33992427%**. Paired versus raw, @.25 fixes/breaks are **230/258**, net **-28**; @.50 fixes/breaks are **335/386**, net **-51**. bbf nets are **-4/-36**. Saved input identities match across these formal pairs. Fused formal bbs Mask hits are **5782/5083**, mIoU **46.73816640%**; bbf is **5809/5115**, **46.98313272%**. Mask values were recounted from stored IoUs, not recomputed from raw binary masks. Evidence: `R/arm/formal/rows.jsonl:1`–`:9508`, `W/arm/formal/rows.jsonl:1`–`:9508`; `B/FUSED_GEOMETRY_PROFILE.json:21`–`:44`.

Original G is separately identified by terminal SHA `0575dfae333dabdf288a470fc09d8967f9867d1853a5f61d9cfc3cbcf7964522`, and its supplied formal receipts report **5615/4495** bbs. The original-G aggregate receipt, stage receipt, audit and comparison summary agree, including their linked hashes. This is receipt-level verification; the original-G formal row/box arrays are absent locally, so they were not freshly recounted. It must not be confused with G-continuation's 5600/4452. Fused differences are **-49/-89** versus original G and **-34/-46** versus G-continuation. Evidence: `O/receipt.json:19`–`:20`, `:44`, `:61`; `O/fit_terminal/receipt.json:20`–`:26`; `O/audit.json:132`; `S:1889`–`:1910`.

The archived terminal is **346718993 bytes**, SHA `d33790155342b6dc5abdcdf451fdf74c3e0cc2e9d8846741b0d1844513035bec`, matching the training receipt and archive receipt. A restricted, inert CPU parser inspected the PyTorch ZIP metadata without importing/executing torch or model code; all ZIP CRCs pass. It contains 1082 delta tensors, 10 refiner tensors totaling 367494 parameters, 29778 recorded row IDs, RNG states, and 806 optimizer states, all at step 3723. The refiner output weights/bias are finite and nonzero. The stored spec and parent/module hashes match the run. Evidence: `A/terminal.pth` ZIP member `archive/data.pkl`; `A/archive_receipt.json:2`–`:8`; `R/arm/receipt.json:77`–`:80`; save/load path `R/run.py:204`–`:215`, `:523`–`:536`.

This checkpoint is a delta, not a standalone full model. It requires the author checkpoint SHA `6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3` and the original-G SHA above. The completed path checks those hashes before loading, and the handoff records protection of the parents. However, their current retained bytes were not supplied locally and were not queried remotely. Current parent retention and complete independent model restoration remain **unverified**, not failed or certified. Evidence: `R/run.py:82`–`:83`, `:128`–`:159`; `H:21502`, `:21527`.

The first fused attempt is preserved as exit 1, with a selected-box equality failure after initial evaluation and before creation of the training optimizer. It is not a trained negative result. Evidence: `F/arm/train.exit:1`, `F/arm/train.log:23`–`:28`, `F/run.py:515`, `:524`; `F/status.json:2`–`:8`. A `fresh_optimizer:true` field in a load receipt by itself does not prove optimizer creation or updates.

The earlier publication records are time-bounded: `fused_retry_publication.json` explicitly says no new training result; its historical handoff prefix hash matches. The candidate-retention publication's full handoff hash matches the audited document and explicitly says its GPU diagnostic was not executed. The raw saved flags independently reproduce candidate bins **966/508/980/886/1711**, strict wrong count **5051**, and alternative strict-coverage count **3340**. The 1137 selected boxes in (.25,.50] split into 675 with another strict-qualifying candidate and 462 without one. These are saved-flag diagnostics, not a completed new all-candidate inference. No later candidate diagnostic is certified by this audit. Evidence: `B/fused_retry_publication.json:17`–`:19`; `B/candidate_retention_publication.json:17`–`:19`; `H:21511`–`:21529`.

**D. Live versus dead code — WARN**

The reported REC and Mask functions are executed through the evaluator's four-method dispatcher. Selected/coarse box IoUs, oracle flags and the values used for receipts are generated inside the same active evaluation loop; native counter equality assertions connect the copied row statistics to that loop. The deployed final `last_center/last_pred_size` are overwritten by the tail refiner after native Mask generation, then used for loss and evaluation. Evidence: `R/source/imported/models.pv_ground.py:518`–`:562`; `R/run.py:440`–`:499`; `R/source/imported/evaluator.py:203`–`:206`.

An inherited `verify_native_replacement` CE/gradient reconstruction witness is guarded by `if not update`, whereas the preflight and training call sites both pass `True`. It is not a live witness for this run and is not present as such in its step records. The handoff already acknowledges this limitation. Evidence: `R/run.py:322`–`:323`, `:391`, `:542`; `R/source/pvground_semantic_assignment.py:53`–`:93`; `H:21433`.

The imported evaluator's standalone NumPy softmax helper, presentation/visualization paths and distributed synchronization are not sources of the reported results. Top-5/10 and auxiliary native counters are computed but not exported in these receipts. The recount script also contains an unreachable `tail_raw` branch under a parser accepting only `tail_fused`; its actually selected comparison is the raw control. Evidence: `B/recount_fused_retry.py:10`–`:19`. No result is credited to these inactive paths.

**E. Scope, initialization and attribution — WARN**

The dataset is ScanRefer only; seed is **2027**, batch size **8**, one fit pass, 3723 updates, LR/backbone LR **1e-5**, AdamW weight decay **0.0005**, clipping **0.1**. Both arms use the same archived model/loss/module bytes, 14-dimensional member inputs, 367494 refiner parameters, parent hashes and training row order. The spec diff is confined to output root, arm/support flag and comparison-root bookkeeping. Runner differences add the explicit initial zero-refinement assertions and amended comparison/reporting gate; the controller's additional disk reserve changes from 128 MiB to 64 MiB. Evidence: `R/arm/spec.json:10`–`:15`; `R/run.py:84`–`:85`, `:175`–`:177`, `:518`–`:520`; byte comparison against `W`.

The intended support intervention is real, but it changes the geometry-to-Mask gradient route as well as forward support values. The real two-step preflight records zero direct semantic/Mask-loss-to-refiner gradients and a nonzero fused geometry-to-Mask-output gradient on step two (**2.39137296e-6**). It does not establish isolation of all tasks during joint training. Evidence: `R/run.py:301`–`:321`, `:390`–`:399`; `R/preflight/preflight.json:596`; `R/source/pvground_tail_support_box_refiner.py:47`–`:61`.

The actual E0 gate is weaker than exact cross-process functional equivalence. It verifies saved row/scan/target/GT/point identities, identical REC threshold decisions, and within-forward zero refinement; all-candidate zero boxes are checked by executed assertions, while saved rows independently verify the selected boxes. The real-batch bypass test covers one batch's center, size and semantic outputs. Neither test supplies full cross-process ranks, all detector/text tensors, or raw Mask equality for every row. Evidence: `R/initial_zero_output_comparison.py:3`–`:23`; `R/run.py:347`–`:362`, `:441`–`:443`, `:509`–`:518`. Four input fixtures cover extra detector/text fields (`R/run.py:247`–`:257`), not every evaluation row.

Independent comparison with raw E0 reproduces bbs Query change at row **26603** and bbf changes at **26603, 26641**. bbs/bbf selected boxes differ in **6882/6881** rows; maximum coordinate differences are **0.1895313263 / 0.0748627186** in stored coordinate units. Mask IoUs differ at rows **16804, 18482** in both modes. All REC threshold decisions remain equal. These include Query switches and must not all be described as negligible floating-point noise. Evidence: `R/arm/initial_control_comparison.json:12844`–`:12848`, `:25697`–`:25717`; `R/arm/receipt.json:34`–`:35`. The change of gate is explicitly documented in `H:21485`–`:21498`; the single-batch probe described there is not independently available in the supplied artifact set and is not promoted to whole-dataset equivalence.

The module holdout has **6887 expressions**, **106 physical scene identifiers**, and **1479 (scan,target) pairs**. Formal development evaluation has **9508 expressions**, **141 physical scene identifiers**, and **2068 (scan,target) pairs**. Fit/holdout separation and 456/106 physical-space counts are recorded and enforced by source; all fit row coverage was independently checked. The holdout was seen during prior pretraining, explicitly recorded in `R/source/split_protocol.json:1`; it is not an unseen-scene test of this method. Formal validation is also repeatedly used in this development history, so no untouched-test, multi-dataset, multi-seed robustness or physical-instance-identity claim follows. Evidence: `R/run.py:185`–`:257`; `R/arm/train.log:8`; supplied row files and `H:21472`–`:21489`.

Both geometry profiles were independently reproduced, including all modes and GT-volume groups. Fused bbs selected-query coarse-to-final @.25 goes **5583→5566** (1 fix, 18 breaks); @.50 goes **4412→4406** (15 fixes, 21 breaks). Raw @.50 goes **4448→4457** (24 fixes, 15 breaks). These are internal diagnostics within each jointly trained model, not separately trained no-refiner ablations.

For fused bbs, median center movement is **1.898482 mm**; median maximum single-face movement **3.064364 mm**, p90 **5.261618 mm**, maximum **8.902013 mm**, with **0/9508** exceeding 10 mm. Raw has median **3.727801 mm** maximum-face movement and **6/9508** exceeding 10 mm. These statistics cover saved selected boxes after size flooring, not every candidate or unclamped residuals; millimetres assume the inherited metre coordinate convention. Evidence: `B/FUSED_GEOMETRY_PROFILE.json:21`–`:81`; `B/RAW_GEOMETRY_PROFILE.json:21`–`:81`; computation `B/profile_completed_fused_geometry.py:31`–`:90`.

GT-volume cuts are **0.0907064692, 0.2477334084, 0.5180449836 m³**, giving **2377/2380/2374/2377** expression rows. Fused within-query @.50 nets by ascending quartile are **-5/-4/+1/+2**; fused-versus-raw nets are **-27/-39/-2/+17**. Small GT volume is not evidence of sparse observed support. Grouping uses GT only offline. Evidence: `B/FUSED_GEOMETRY_PROFILE.json:12`–`:16`, its `by_GT_volume` records; `B/profile_completed_fused_geometry.py:31`–`:35`, `:113`–`:118`.

Accordingly, the supported comparison is a **single-seed same-tail, same-budget empirical result under a documented amended initialization gate**. It does not isolate every changed hit as an exclusive causal effect of Mask support. Historical G comparisons additionally change the tail/source protocol.

**F. Evaluation classification — PASS: real_gt**

REC and Mask metrics compare predictions against dataset-derived target boxes and object memberships. Predicted Mask support does not turn the evaluation into a model-generated-reference proxy. Coarse/refined and GT-volume analyses are offline diagnostics using real GT. Full-256 coverage is a GT-assisted oracle from saved GPU flags: fused formal **8923/7814**, raw **8934/7797** at the two thresholds. Those are not deployable selected accuracy, semantic identity verification, or independent CPU reconstruction of all 256 candidates. Evidence: `R/run.py:453`–`:485`; `B/recount_fused_retry.py:93`–`:100`; `B/FUSED_GEOMETRY_PROFILE.json:679`. No completed new matching/identity diagnostic is included in this classification.

**Actionable outcome and claim limits**

- **No integrity blocker for reporting the completed negative result with these qualifications.** Counts, selected geometry, fit coverage and local archive integrity pass deterministic checks.
- **Performance promotion is unsupported:** 4406 strict hits is **348 below 4754/9508 (50%)** and 51 below the raw control. Preserve the original G and protected reference results. This audit does not authorize replacement, deletion or Nr3D/Sr3D promotion.
- Describe the same-tail comparison with its amended E0 gate and observed Query/box/Mask differences. Exact functional-start parity or exclusive support-mechanism causality requires additional evidence; do not infer either from equal initial hit counts.
- Before claiming independently restorable archival preservation, verify both retained parent checkpoints against their named hashes. This review confirms the child archive and execution-time loading evidence only.
- Keep raw Mask replay, full-candidate reconstruction and the planned matching diagnostic explicitly unavailable until actual data, completed receipts and independent checks exist. Do not relabel a launch, script, partial run or failed diagnostic as a completed result.
- Do not claim the inherited CE/gradient reconstruction witness ran here. Its unused branch does not invalidate the live metric computation.
- Append the completed result after the dated launch/pending sections; preserve those historical records and the failed first attempt. Positive mechanism, robustness, unseen-scene and cross-dataset claims remain unsupported.

**Selected input hashes**

All values below are SHA-256. The complete absolute-path inventory and independent numerical results are in [EXPERIMENT_AUDIT.json](C:/Users/gb/.codex/tmp/pvground_fused_support_20261002/complete_tail_fused_retry/EXPERIMENT_AUDIT.json).

| Input | SHA-256 |
|---|---|
| R/run.py | `54f9df009cc5b67f27ab52745143e85d1fcd5301432b69f664f5e944999c22e6` |
| R/arm/spec.json | `234c661dfab8ad193949042dff08eaa52465793d66e8af37766ba709e102dcf9` |
| R/arm/formal/rows.jsonl | `0b6b63512e819890747eff146abff9a54ae8581cf636da117bcf859a6c0fecbf` |
| W/arm/formal/rows.jsonl | `019210a51f6f6c54fea52d4bf0b67c56154d4840c9d96459857b904ca6910f79` |
| B/FUSED_GEOMETRY_PROFILE.json | `04e1c81c52aff73d5701ccf66ff25488b2d66d8405205b2417cfbc5f7bc12860` |
| B/RAW_GEOMETRY_PROFILE.json | `8c82f109b9f322780b0603134012a121849494367bc03ca6095f3a2124c0d414` |
| A/terminal.pth | `d33790155342b6dc5abdcdf451fdf74c3e0cc2e9d8846741b0d1844513035bec` |
| O/receipt.json | `60af173412696ee53ac13e60116460afa6bae5050cd96af84249088dba296374` |
| H, audited complete bytes | `7cd5d9c21ecef1e66323e7aeff44ef025402b087e089ac8a21e0362eb64fd860` |
