# Native target-box jitter terminal integrity audit

**Verdict: PASS. Blocking findings: 0. Unresolved nonblocking findings: 0.**

Actual closed CPU diagnostic and publication-template source review. Fresh Codex context; requested gpt-6-astra / max; backend and effort unattested. Attribution: **same-family / provisional**. This is not independent or external acceptance.

Generated: 2026-10-05T16:11:44.293994+08:00

## Verified result

The diagnostic completed at **2026-10-05T15:28:47.939361+08:00**, elapsed **422.92293848097324 seconds**, exit **0**. Nine collected files total **61,608 bytes**. All intake hashes/sizes, copied sources and historical source-review hashes match. Eight log events cover 8 through 64 rows; the final log event equals receipt.json and the wait log.

A separately written read-only NumPy recount used cached NumPy 2.5.3 via `uv run --offline --with numpy python -B -X utf8 -`. It imported neither audited metric module nor experiment entrypoint. All **414 target SUMMARY scalar leaves** match exactly: 410 numeric, 2 boolean and 2 string. All **442 referenced cohort SUMMARY leaves** also match: 433 numeric, 3 boolean and 6 string. Status/counter/provenance leaves were checked against source and closure records; numerical reductions came from JSON and eight NPZ archives.

## A–F checks

### A. Ground truth provenance: PASS

ScanRefer object_id and ScanNet segmentation/aggregation supply the target identity and point membership. Both box definitions are dataset-derived; no prediction, detector output or predicted mask becomes GT.

The override captures scan.get_object_bbox(root) after native common point augmentation and immediately before the native six-coordinate center/size multiplier. Its lookup and arithmetic consume no RNG, mutate no scan data and delegate to the native method exactly once.

Pre-jitter means the bbox of annotated members in the native 50000-point scan representation. It is not a raw full-resolution physical-object envelope. The current publisher line58 explicitly states that scope.

The supplied runtime-bundle visual_data_handlers file differs from the manifest-pinned file only by CRLF/LF bytes. The exact pinned local file was additionally inspected and its SHA matches the dataset manifest; normalized text and AST are identical.

Evidence:

- `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:594-648` — Dataset annotations and object_id.
- `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1086-1119` — Masks, bbox lookup and native independent box jitter.
- `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1324-1329` — Common point augmentation precedes bbox capture.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\run_label_probe.py:68-82` — Pre-jitter capture and unchanged prior bool-mask wrapper.
- `C:\Users\gb\.codex\tmp\pv_ground_source_20260905\src\visual_data_handlers.py:84-126` — Native 50000-point scan representation.
- `C:\Users\gb\.codex\tmp\pv_ground_source_20260905\src\visual_data_handlers.py:129-160` — Dataset annotation membership.
- `C:\Users\gb\.codex\tmp\pv_ground_source_20260905\src\visual_data_handlers.py:195-259` — Pure member lookup and bbox arithmetic.
- `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\appearance_source_manifest.json:466-470` — Pinned source SHA values.

### B. Normalization and metric formulas: PASS

Reported box IoU divides 3D intersection volume by prediction volume + dataset-target volume - intersection. No output-max/min/mean normalization is used.

Mask support uses saved int64 intersections/unions and strict 2*intersection>union for both query and fused masks, plus matched_slot<0. Success is strictly IoU>0.5 (paired summaries also use >0.25); qualification uses parent IoU<=0.5.

Face error/shift is maximum absolute low/high face displacement in metres. Boundary target is ((coarse_center - GT_low)/coarse_size)*4-2 for negative faces and ((GT_high - coarse_center)/coarse_size)*4-2 for positive faces. Cached coarse sizes already use the native float32 1e-6 floor; outside means target<-4 or target>4, excluding equality.

Coarse-size normalization is the existing coordinate/loss parameterization, not normalization of a reported quality score. All reported group counts are actual candidate denominators; selected_query contains one BBS argmax per row.

Evidence:

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\cohort_metrics.py:6-34` — Physical IoU, face conversion and strict paired transitions.
- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:309-327` — Cached size floor, integer Mask support and node bounds.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\analyze_labels.py:36-87` — Eligibility, target formulas, groups and reductions.
- `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_boundary_box_refiner.py:9-37` — Scale=4, floor=1e-6, endpoints [-4,4] and exact target formula.

### C. Artifact existence, status and exact numbers: PASS

All nine intake entries have exact recorded size/SHA; total 61608 bytes. No unmanifested complete artifact exists except INTAKE.json. Copied runner/spec/plan/source-review artifacts equal their local counterparts; all 23 historical reviewed-file hashes remain valid.

The fresh recount checks all 414 target summary scalar leaves exactly (410 numeric, 2 bool, 2 string). The referenced cohort summary also checks exactly for all 442 leaves (433 numeric, 3 bool, 6 string). Numerical reductions were recalculated from JSON/NPZ; status/counter/provenance leaves were checked against source and closure records.

64 row IDs/order, scan IDs, point-byte hash records, all noisy root coordinates, valid native GT slots [0], NPZ row/root coordinates, saved matching, selected BBS queries and per-row qualification/outside counts are consistent.

Finished receipt, final JSON log event and wait.log are identical; run.exit is 0; progress is exactly 8,16,...,64. Elapsed runtime is 422.92293848097324 seconds. The 2-5 minute source estimate was an estimate, not the actual duration.

Prior-summary DFL means are exactly reproduced from saved per-candidate GPU scalars. Raw logits are absent, so the underlying DFL loss is not independently recalculated.

Evidence:

- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\complete\INTAKE.json:1-54` — Nine exact saved files and zero-weight intake.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\complete\run.log:12-20` — All batch events and terminal receipt.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\complete\run.exit:1` — Successful exit.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\complete\receipt.json:2-17` — Closed diagnostic scope/counters.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\complete\rows.jsonl:1-64` — All replay rows.
- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\rows.jsonl:1-64` — Existing cohort rows.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\analysis\SUMMARY.json:1-632` — All 414 leaves checked.
- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\analysis\SUMMARY.json:1-640` — All 442 leaves checked; historical audit-pending metadata remains historical.

### D. Metric functions actually used: PASS

analyze_labels imports cpu_iou/faces/paired and calls each in the target/arm/group loops. Its local face_target is called for both target definitions. All outputs are present in the saved target SUMMARY.

The shared summarize function is a prior-cohort metric, called by analyze_probe.py:41 and represented in the existing cohort SUMMARY. It is not an unused promised target-jitter metric.

Actual new runtime reads data and saves boxes only. It imports the dataset/data contract, not the PV-Ground models package; CUDA is hidden and asserted uninitialized. torch.load within the dataset loads saved superpoint data tensors, not model weights.

Evidence:

- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\analyze_labels.py:12-12` — Imported metric functions.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\analyze_labels.py:39-98` — Every metric call reaches saved summary.
- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\analyze_probe.py:7-49` — Prior summarize execution and output.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\run_label_probe.py:37-47` — CPU dataset import.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\run_label_probe.py:132-144` — Runtime guards and terminal zero model/optimizer/weight counters.
- `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:195-226` — Local tokenizer and data-superpoint loading.

### E. Scope, counterfactual matching and claims: PASS

Exactly one seed (2027), eight shuffled fit batches of eight, 64 distinct expressions, 60 distinct scan IDs/scene prefixes and 256 correlated candidates per expression. All 64 row IDs and scene-salt classifications belong to the hash-matched 29778-row fit partition; the 6887-row holdout is disjoint.

The supported-unmatched set (3479 candidates), native noisy-label Hungarian assignment, query/fused Mask counts, cached boxes and native BBS ranking are held fixed. Only reference boxes and derived targets/IoUs change. Clean-qualified 861 is named separately; paired head comparisons use each explicitly defined fixed group.

There is no formal-accuracy, augmentation-bug, unseen-scene, multi-seed, physical-identity or causal-deficit conclusion. The source and publication template explicitly reject those inferences and do not recompute DFL without logits.

Publication numbers in the fixed-1090 table, jitter counts, completion status, old formal hit/percentage deltas and deleted-byte count match their saved sources. Current line72 explicitly separates the old 4506 parent's 964 cases from the retained 4509 head's 961 cases.

The existing formal9508 and initial/terminal holdout6887 paths explicitly set augment=False and augment_det=False before fetching fresh evaluation batches. This data-label diagnostic does not change those paths.

Evidence:

- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\run_label_probe.py:49-107` — One seed, native RNG/order and fixed fit loader.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\analyze_labels.py:36-98` — Counterfactual keeps matching/Mask support fixed and separates groups.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\publish_closed_labels.py:58-76` — Current bounded claims and explicit historical-head attribution.
- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\run_geometry_fit.py:175-204` — Formal/train dataset definitions.
- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\run_geometry_fit.py:273-287` — Augmentation disabled before evaluation loader.
- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\run_geometry_fit.py:417-419` — Formal evaluation call.
- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\run_geometry_fit.py:453-455` — Holdout then augmented fit.
- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\run_geometry_fit.py:491-495` — Terminal holdout evaluation.

### F. Evaluation classification: PASS

This is a bounded annotation-derived, fixed-training-panel target counterfactual. Annotation geometry is not generated from model predictions, so synthetic_proxy would misdescribe GT provenance.

The subset is model-conditioned through offline support/box eligibility. Candidate hit counts and geometry changes are within-panel diagnostics, not newly accepted formal performance metrics. No separate fresh formal evaluation occurred in this review.

Evidence:

- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\EXPERIMENT_PLAN.md:3-13` — Diagnostic purpose and claim ceiling.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\analysis\SUMMARY.json:2-17` — accuracy_result=false and bounded-target metadata.
- `C:\Users\gb\.codex\tmp\pvground_target_jitter_20261005\publish_closed_labels.py:70-76` — Old formal results distinguished from offline label substitutions.

## Recounted numbers

The panel contains 64 distinct fit expressions, 60 scan IDs/scene prefixes, seed 2027, eight batches and 256 correlated candidates per row. All 64 row IDs and scene-salt classifications belong to the hash-matched 29,778-row fit partition; holdout 6,887 is disjoint.

| Quantity | Native noisy | Pre-jitter annotation |
|---|---:|---:|
| Supported, unmatched candidates (fixed) | 3,479 | 3,479 |
| Box≤0.5 qualified candidates | 1,090 | 861 |
| Rows with a qualified candidate | 52 | 51 |
| Fixed 1,090 cohort: outside candidates | 201 | 87 |
| Fixed 1,090 cohort: outside faces | 482 | 265 |
| Fixed 1,090: parent / control / query-supported hits>0.5 | 0 / 3 / 5 | 395 / 388 / 382 |
| Fixed 1,090: parent mean IoU | 0.2736403096341624 | 0.4189426279673801 |
| Fixed 1,090: control mean IoU | 0.27183079870865834 | 0.41745845720247315 |
| Fixed 1,090: query-supported mean IoU | 0.2724319541905488 | 0.41731409837226197 |

Replacing the target removes 395 candidates from the qualification and adds 166. The fixed noisy cohort splits into 1,003 inside and 87 outside the clean range. Maximum per-row face shift has median **0.053923383355140686 m**, maximum **0.14195909723639488 m**; all 64 rows exceed 1 cm and 37 exceed 5 cm.

Every arm has zero CPU-versus-saved-GPU threshold disagreements at 0.25 and 0.5. Native outside-range classification has zero disagreements. Coarse sizes use the saved float32 floor (9.999999974752427e-7). The float64 boundary recount can differ numerically at very large normalized coordinates, but no outside membership changes. Prior-summary DFL means are reductions of saved GPU scalars only; raw logits are absent.

## Source and actual scope

SOURCE_REVIEW.json and SOURCE_REVIEW_CALL.json remain historical **SOURCE_ONLY** records. This terminal audit additionally checks actual exit/log/receipt/intake/row evidence. The new CPU replay has zero model forwards, optimizer steps, model weights loaded/created and initialized CUDA. The earlier GPU cohort that produced cached candidate arrays remains a separate executed experiment.

Native row order, scan IDs, point-hash records, noisy root coordinates and valid slots [0] match the preceding cohort. Saved NPZ rows/roots, selections and matching are consistent. The added annotation lookup has no RNG/mutation and invokes native target construction once. Loader seed, shuffle, workers, batch and the prior boolean-mask conversion are unchanged. Original remote scans/masks and no-augmentation fixtures were not reloaded by this reviewer; exact replay is witnessed by successful runtime assertions and saved row identity.

The supplied visual_data_handlers.py SHA is `8858aa1d7019aa5f06065bd13fcc35fe130e347787c088269c3535939806060a`. The exact manifest-pinned file at `C:\Users\gb\.codex\tmp\pv_ground_source_20260905\src\visual_data_handlers.py` has SHA `6d1d4f0c792e8b86f238d88c02f1933f945313a3831d0bf9eabb9bf4580bb177`. Decoded text and AST are identical; only LF/CRLF bytes differ. The actual source manifest matches input_manifest.json.

## Publication-template check

Fixed-cohort numbers, target shifts, eligibility changes and closure claims match evidence. Existing formal 5,614/4,509 out of 9,508 is 59.0450%/47.4232%; versus the old 5,616/4,506 parent this is strict +3 and loose −2. Saved row threshold tallies and receipt digests were checked only to verify the template; no fresh formal model evaluation occurred.

One attribution ambiguity was found and resolved by the parent's source-only wording edit: the old 4,506 parent has **964** saved formal Mask>0.5/Box≤0.5 cases; the retained 4,509 head has **961**. Current line72 explicitly identifies both. Current line58 also explicitly limits geometry to the native **50,000-point scan representation**, not original full-resolution points.

Final reviewed publisher SHA256: `54786a8cf4043a2c25f013bc610bbe02fbe6d550694736cbe58678f5e944ff28`.

Existing receipts support the complete retained 10-state head and two already-deleted nonbest heads totaling 11,173,002 bytes. No fresh retention/deletion or current remote-file revalidation occurred. Formal9508/holdout6887 augmentation settings were traced to executed source/closed receipts. No target protocol change or formal-accuracy causal inference is accepted.

The publisher was **not executed by this reviewer**. No handoff, continuation state, repository or remote artifact was modified.

## Limits

- Native annotation bbox is based on the existing 50000-point scan representation and common augmentation; pre-jitter removes only the final independent six-coordinate target multiplier.
- The reviewer compared saved point hashes and successful exact-input runtime assertions; original remote scan points/masks and no-augmentation fixtures were not reloaded locally.
- Native returned values/field behavior are preserved by the inspected delegation path; the pre-existing boolean gt_masks conversion is unchanged from the prior probe. All fields enumerated in actual replay assertions are witnessed; unsaved fields are not separately byte-recounted.
- One fixed augmented fit panel and correlated candidates cannot establish formal-accuracy improvement or a cause of a formal error population.
- Existing noisy-label matching/support are not recomputed for clean target boxes. This is the stated counterfactual, not a new complete matching/evaluation protocol.
- No new DFL from logits, new model forward, training, checkpoint operation or remote current-state validation occurred.
- The historical source review remains SOURCE_ONLY; the present terminal verdict is same-family/provisional with unattested requested model/effort.
- Publication helper execution and publication completion remain outside this review.

## Actions and inventory

No remaining correction is required for the audited diagnostic. This reviewer changed no experiment source, raw result, SUMMARY, weight or dataset; only the two audit reports were written. The parent corrected the two publication phrases, whose final source SHA is recorded above.

The first read-only recount attempt compared the stored float32 coarse floor to a float64 literal. Correcting that reviewer assertion to the actual dtype yielded the full passing recount; this was not an experiment failure and changed no experiment artifact.

The JSON contains exact paths, sizes, SHA256 values and review scopes for all **62 evidence files** under `reviewed_files`, along with full recount checks and per-target/arm results. No credential wrapper or secret was read.

