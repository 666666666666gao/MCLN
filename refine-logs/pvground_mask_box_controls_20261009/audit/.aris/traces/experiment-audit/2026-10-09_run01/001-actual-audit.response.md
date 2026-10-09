# Fresh actual experiment-integrity review

Date: 2026-10-09. Agent: `/root/pvg_fixed_geometry_actual_audit_20261009`.

Overall verdict: **WARN**. The closed cached-prediction arithmetic **PASSES** independent deterministic verification. There is **no blocking finding for the narrowly stated October 6 fixed-query geometry comparison**. The cross-forward extension is accepted only as a row-aligned arithmetic hypothesis. Neither extension establishes a new deployable model, weight-promotion eligibility, three effective innovations, or goal completion.

Attribution: `fresh_context=true`; `requested_model=gpt-6-astra`; `requested_reasoning_effort=max`; actual backend, actual model and actual reasoning effort are all `UNATTESTED`. The local native reviewer route is same-family, with `review_independence=same-family` and `acceptance_status=provisional`. Requested routing is not evidence of the actual serving backend. No cross-family acceptance is asserted.

This reviewer read the primary scripts, rows, receipts and arrays directly and wrote an independent NumPy verifier. It did not import or execute either executor analysis script or any model code. It used the existing cached Python 3.10.19 / NumPy 1.23.5 environment via offline uv. No SSH, GPU, model forward, training, package installation, checkpoint deserialization, cleanup, active-source change, credential access or main MEMORY.md access occurred. All reviewer writes are inside this audit directory.

## Evidence locations

The following aliases are exact path prefixes used by the line references below:

- `D` = `C:/Users/gb/.codex/tmp/pvground_mask_box_controls_20261009`
- `O` = `C:/Users/gb/.codex/tmp/pvground_mask_reference_20261006`
- `S` = `C:/Users/gb/.codex/tmp/pvground_mask_reference_20261006/complete_fit/fused_mask_reference`
- `C` = `C:/Users/gb/.codex/tmp/pvground_compressed_geometry_support_20261008`
- `L` = `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py`
- `V` = `C:/Users/gb/.codex/tmp/pv_ground_source_20260905/src/visual_data_handlers.py`
- `E` = `C:/Users/gb/.codex/tmp/pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py`
- `H` = `C:/Users/gb/.codex/tmp/pvground_geometry_readback_20261004/revision2`
- `P` = `C:/Users/gb/.codex/tmp/pvground_novelty_20261009`

Every reviewed input has a SHA256 and byte size in `REVIEWED_FILE_HASHES.json`. All 1189 original NPZ files, independently read and hashed, are listed in `VERIFIED_ARRAY_HASHES.json`; each matches both the executor's new `ARRAY_MANIFEST.json` and the original October 6 `complete_fit/INTAKE.json`. The source arrays total **115,804,171 bytes**. The deterministic record is `DETERMINISTIC_VERIFY.json`, SHA256 `7d1efc9b2d2a54b512aac295f05e15f57e5e1a953caaa08e27b893ba9da39f36`; the supplementary record is `SUPPLEMENTARY_VERIFY.json`, SHA256 `1238e6b79b12ba108c231b7c0d662fb6ba6c1fda2fb34e3d3559f931fd6cfbf7`.

## A. Ground-truth provenance — WARN, with real-GT source tracing passed

The targets are not generated from predictions. The runner constructs the validation dataset with `split='val'`, requests exactly 9508 rows, disables augmentation in `evaluate`, and saves `root_gt` from `batch['center_label'][:,0,:3]` and `batch['size_gts'][:,0]` (`O/run_geometry_fit.py:194-213,299-325,332`). Model inputs at `268-281` contain points, text, superpoints and detected boxes, not root GT boxes or masks. The predicted reference uses fused predicted logits and member-coordinate extents, not GT (`O/mask_reference.py:10-28,42-66`).

The recovered original loader's SHA256 is `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d`, exactly matching `S/imports.json:8,16` and the bound appearance source manifest. It loads `val_v3scans.pkl` at `L:205-209`, loads the ScanRefer validation scene list and annotations at `585-597`, obtains `target_id` from annotated `object_id` at `634-648`, and constructs root GT from the annotated target's `scan.get_object_bbox(tid)` at `1086-1119`. It returns those center/size tensors at `1387-1392`. Validation does not execute the training-only box jitter at `1112-1113`.

The exact scan-handler source was also found, with SHA256 `6d1d4f0c792e8b86f238d88c02f1933f945313a3831d0bf9eabb9bf4580bb177`, matching `appearance_source_manifest.json`. It loads ScanNet mesh coordinates and alignment, samples 50,000 points with its inherited fixed sampling seed 1184 (`V:84-125`), and associates points with annotated segment groups (`V:129-178`). `get_object_bbox` returns the AABB of annotated object member points (`V:195-197,225-227,245-259`). This is dataset-annotation-derived GT, rather than a prediction-generated target. The inherited sampling seed is distinct from the experiment seed 2027; it is not an extra experimental run.

Qualification: this audit did not load the original `val_v3scans.pkl`, source mesh/aggregation files or original annotation JSON and rederive every box from those bytes. Nor does the source receipt contain a complete stage-specific dataset/checkpoint attestation. The source implementation derives boxes from the cached sampled scan members; the audit does not certify equivalence to an alternative full-resolution official GT box implementation. The valid claim is **real_gt under the archived native PV/ScanRefer data and evaluation contract**. This provenance ceiling is not a reason to rerun a GPU to check the cached arithmetic.

## B. Metric normalization and numerical behavior — PASS, with a recorded precision warning

`D/analyze_fixed_query_boxes.py:25-32` computes standard axis-aligned 3D intersection over union. The denominator is predicted volume plus dataset-GT volume minus intersection, not the model's score maximum, mean, best-IoU row or a learned normalizer. Accuracy is strict `IoU > 0.25` / `IoU > 0.5`, divided by the fixed count 9508 (`35-41,109-115`). The native evaluator uses the same strict thresholds (`E:289-303`); no row lies exactly on either threshold in the audited conditions.

Independent verification reconstructed all 28,524 selected IoUs per precision directly from the original NPZ arrays. Every saved CPU float32 and float64 IoU equals the independently computed value, and all summary counts and transition tables match exactly. No float32/float64 threshold decision differs in any of the three conditions. Native/mask saved GPU versus CPU threshold decisions also have zero flips; their maximum absolute IoU difference is `4.172325134277344e-7`. Recomputing the average itself in float64 also gives 5671/4839.

There is a real, small numerical issue: 94 pure-Mask float32 IoUs exceed 1, with maximum `1.0000050067901611` (for example source row 981, `SELECTED_BOX_ROWS.jsonl:982`). This follows the unmodified coordinate-intersection / stored-size-volume arithmetic in float32. All values are finite and nonnegative; the float64 maximum is exactly 1.0. The maximum Mask float32/float64 difference is `6.6200876713828904e-6`. This does not alter either reported accuracy threshold. The auditor retained the raw values and did not clamp, normalize or rewrite the source results. The diagnostic and the failed overly strict auditor assertion are preserved. Do not claim that the raw float32 IoU range is strictly bounded by 1 or that its values are bitwise equal to GPU/float64.

## C. Existence, hashes, counts and snapshot binding — PASS for cached results

There are exactly 9508 sequential, unique row IDs 0 through 9507, covering 141 scenes and 2068 scene/target pairs. There are 1189 contiguous NPZ shards, with batches of 8 except the final batch of 4; no extra or missing source shard exists. Each has 256 candidates and the expected seven fields. All predictions and root boxes are finite with positive dimensions. All shapes and file hashes are checked in `audit/independent_verify.py:65-146`.

The original source row SHA is `40672da53d767bc2262fa280a9904589c813a98b0c419d1cbdac161378144a23`, bound by `S/initial_formal/receipt.json:16` and by the exact completion message at `S/initial_formal.log:25`. `S/initial_formal.exit` is 0. `O/complete_fit/fit_status.json:1-27` records the completed source stages; the controller launches the actual runner and mode, waits for each child, and only records completion after exit 0 (`O/complete_fit/controller.py:36-58`). Source receipt values 5598/4848 agree with both the selected Mask array recount and the official-evaluator assertion in the runner (`O/run_geometry_fit.py:365-378`).

The new `SUMMARY.json:6-10` binds the source rows, source receipt, analysis script, array manifest and selected output rows. All five hashes match. The selected output SHA is `db8ad8f8869218aef4444f71465b9d3d59a6cc284f1328645d47c59a1342b976`. No October 8 corrected output is used by the October 6 primary control.

The original stage is a zero-output-offset reference state, not a newly learned geometry state. `O/mask_reference.py:69-80` resets both output tensors to zero; `O/run_geometry_fit.py:468-479` evaluates the initial state without an optimizer step. The independently checked `final` and `reference` arrays are equal for every candidate in every shard. The historical retention receipt maps `fused_mask_reference/initial_formal` to step 0 and SHA256 `2301e90a9391af4e4cd56edae09ce49db1cc8f67463e1c2568352ee2dafaff61` (`O/weight_retention.json:4-15`).

Provenance boundary: `S/load.json:21-23` is timestamped 17:35, after the initial-formal completion at 15:10, and is overwritten by the later formal runner. Its matching spec/source hashes are corroboration, not a distinct initial-stage state digest. The old stage's receipt binds rows, while the archived intake separately binds arrays/source bytes. No complete original GPU state was loaded or newly attested here. The independent CPU output establishes arithmetic over the archived snapshot, not a new neural execution.

## D. Executed versus dead code; fixed-query mechanism — PASS

The executor analyzer reaches `main()` at `D/analyze_fixed_query_boxes.py:153-154`, reads every shard, evaluates all three box conditions (`61-105`), invokes both precision implementations (`90-94`), and writes the exact observed rows/summary (`120-150`). Its transition function is consumed in all three pair comparisons (`129-132`). The auditor independently executed `audit/independent_verify.py`, ending with actual process exit 0 (session 53451, completion chunk `4fa41d`). The supplementary verifier also exited 0 (chunk `de1cf4`). The auditor did not relabel mere source inspection as an executed experiment.

The original runner calls `observed_readback_forward` once per batch and saves native boxes, reference boxes, final boxes, scores and GT from that same batch before loss/evaluator consumption (`O/run_geometry_fit.py:312-325`). The observed hook executes a single model call and verifies the refiner/readback/native-head order and geometry/mask equality within that call (`H/readback_preflight_checks.py:19-68`). Every archived row reports one native-head call and zero diagnostic head replays (`O/run_geometry_fit.py:355-356`); all 9508 records agree.

The native score helper uses semantic probabilities and text token maps (`H/native_root_bbs.py:4-9`), matching the original evaluator's `bbs` score and arg-sort (`E:217-286`). The relevant maps come from parsed text spans (`L:981-1082,1342-1348`); there is no selection based on box GT IoU. The runner computes and saves scores before computing the per-row IoU, chooses `ranked[0]` (`O/run_geometry_fit.py:315,339-340`), and the auditor checks that every saved Query is the maximum of the 256 saved scores. **There are zero maximum-score ties.**

Native, Mask and blended boxes use that identical Query, row, source input hash and GT. The blend is globally fixed to `(native + mask) * float32(0.5)` (`D/analyze_fixed_query_boxes.py:81-102`); no coefficient loop, score search or GT-conditioned per-row choice appears. The source row's `oracle25/oracle50` fields are diagnostics (`O/run_geometry_fit.py:353-354`), never used by the control selector. `reference_bounds_witness` is explicitly a preflight-only raw-member check (`O/mask_reference.py:83-117`); it was not freshly rerun on every formal mask and is not claimed to have been.

Invalid references follow the already existing no-support/degenerate-support rule `O/mask_reference.py:24-26`, which depends only on predicted support. There are 1,162,190 invalid candidate references among 2,434,048 candidates, including 39 selected references. For every invalid candidate the archived reference equals the native prior exactly; all selected invalid rows therefore have identical native/Mask/blended geometry. This audit verifies the archived handling, not a new fallback or a candidate-coverage performance claim.

## E. Scope and claim ceiling — WARN

The primary comparison is one archived model state, one complete 9508-row ScanRefer validation pass, three deterministic representations of each selected Query, and seed 2027. The pass consists of 1189 original batch forwards; “same forward” means the three boxes for each row share its original batch forward, not that all 9508 rows were processed in one batch. The present work performs zero neural forwards. The 256-candidate arrays are preserved, but the tested metric is top-1 selected-box geometry, not a new all-candidate recall study, robustness study or independent multi-dataset trial.

| October 6 condition | Hits @0.25 / @0.5 | Accuracy @0.25 / @0.5 |
|---|---:|---:|
| Native regression | 5615 / 4495 | 59.0555% / 47.2760% |
| Pure fused-Mask reference | 5598 / 4848 | 58.8767% / 50.9886% |
| Fixed half blend | 5671 / 4839 | 59.6445% / 50.8940% |

Relative to pure Mask, the blend repairs/damages 121/48 at 0.25 and 207/216 at 0.5, net +73/-9. Its strict count is 11 short of the minimum 4850 for a value strictly above 51%. No primary condition reaches the joint minimum 5658/4850. The Mask and blend columns cannot be mixed into a fictitious single model.

EG-3DVG's original supplied paper specifies a mask-derived box and a 0.5 weighted average with its initial regression box (`P/EG-3DVG.txt:208-247`). It also has separate PECA/GMA/ECL mechanisms (`101-119,191-207`). This work applies only that prior geometry equation to PV outputs. It is not an EG-3DVG architecture/training reproduction and is not a novel contribution by itself. The reported limits say so explicitly (`D/SUMMARY.json:137-143`; `D/FINDINGS.md:35`). `NOVELTY_REVIEW.md:23-31,64-71,85-89` calls for this narrow control and does not establish three effective modules. This audit does not independently certify the broader novelty review.

## F. Evaluation type — PASS: real_gt, with distinct prediction scopes

The primary control uses **real_gt**, specifically dataset-annotation-derived targets under the archived native contract. A predicted Mask-derived box is a prediction, not a target; its use does not make the metric a synthetic-proxy GT evaluation. The later hybrid likewise compares constructed predictions to real dataset GT, but it is a **cross_forward_geometry_hypothesis**, not a same-forward current-model benchmark or an independently executed new complete model. Both distinctions must remain in downstream tables and prose.

## Additional requested cross-forward audit

All 9508 rows match exactly across the archived selected rows and current source rows in row ID, scan ID, target ID, Query, root GT, point-cloud SHA and uncorrected Mask extent (`D/compare_retained_with_archived_prior.py:18-38`, independently checked at `audit/independent_verify.py:202-245`). Current source rows and receipt match their stated hashes and the source receipt's 9508 rows. Both configurations match the official base checkpoint hash, G checkpoint hash, environment hash, input-manifest path, native-evaluator hash, seed, batch size, reference mode, retained-hidden prior updates and all 17 helper hashes. The old source-port hash equals the current old-source-port hash; mask-reference hashes also match. These are dependency/spec equality checks, not a fresh bitwise equality proof of all original GPU state.

The retained content checkpoint's actual local archive is 447,109 bytes with SHA256 `6f9109cb690399880b8dc3bcf6e4b98d1053145732ee2a6b504bfaecbd1eaac2`, independently hashed without loading it. Its retained receipt agrees with the cross-forward result and current spec. The old zero-update reference checkpoint identity agrees between `O/weight_retention.json` and `C/pair_spec.json:74-75`. `C/postrun/checkpoint_inspection.json:49-59` explicitly says original formal GPU all-state identity was not verified; this audit preserves that limit. The main numeric verifier did not hash checkpoints; the separate supplementary verifier supplies the actual archive hash check.

| Cross-forward condition | Hits @0.25 / @0.5 | Accuracy @0.25 / @0.5 |
|---|---:|---:|
| Saved retained pure-Mask output | 5599 / 4859 | 58.8873% / 51.1043% |
| October 6 regression plus retained Mask, fixed half blend | 5676 / 4840 | 59.6971% / 50.9045% |

The hybrid repairs/damages 124/47 at 0.25 and 195/214 at 0.5, net +77/-19. Float32/float64 threshold flips are zero. It fails the strict count by 10 rows and therefore does not even meet the numerical joint gate. **The current same-forward native regression boxes were not saved.** Equal inputs, selected Queries and uncorrected Mask extents do not prove identity of those unsaved boxes. Current-model deployment performance and promotion remain unsupported regardless of the numerical result. The extension is honestly labeled at `D/RETAINED_CROSS_FORWARD_HYPOTHESIS.json:12-13,75-84` and `D/FINDINGS.md:19-30`; the labels must not be removed.

`D/FINDINGS.md:38` correctly describes 3074 as zero **box** overlap, rather than logically deriving zero Mask intersection from the box statistic. The supplemental recount confirms 3074 zero box intersections and no saved Mask IoU above 0.5 in that subset. It additionally observes that all those *saved scalar* Mask IoUs equal zero. No raw mask membership was reconstructed in this audit, so this is a source-scalar recount, not a new all-mask intersection proof. The training-status and future-plan sentences at `FINDINGS.md:36-40` are outside this closed CPU execution audit; no claim that the active run was checked is made.

## Actual invocation failures and corrections

No failed invocation is omitted or converted into a scientific PASS:

1. Initial requested `O/pair_spec.json` was absent. The numbered read exited 1 (chunk `c06209`). The parent corrected the request to the existing `O/fused_mask_reference_spec.json`, read successfully (chunk `e04a27`). The error excerpt is in `INVOCATION_ERRORS.json`. This is a request-path correction.
2. The first verifier failed before array computation because `Path.resolve()` followed the `.codex` directory junction to `D:/Program Files/UserCache/gb/codex`, while manifest strings use `C:/Users/gb/.codex`. Counts were 1189/1189 and filesystem entries were the same. `verify_001.log`, its invocation and diagnostic JSON, and `independent_verify_attempt001.py` are retained. The verifier now preserves the specified absolute alias. The same invocation also printed uv's warning that `--no-sync` has no effect with `--no-project`; later invocations omit that redundant flag. No package installation occurred.
3. The second verifier completed all original shard/hash/geometry checks but failed its overly strict raw float32 `IoU <= 1` assertion. `verify_002.log`, invocation/completion/diagnostic JSON and `independent_verify_attempt002.py` are retained. The third version reports out-of-range values without clipping them and verifies the requested thresholds. It exited 0 and produced the deterministic receipt.
4. A read-only `rg --files ... -g '*.pth'` search in the two experiment directories returned exit 1 because those directories contain no checkpoint files (chunk `cd9aaf`). This is a normal no-match discovery result, not a data corruption or scientific failure. The separately recorded retained checkpoint archive exists and hashes correctly.

The raw verification output files and tool result receipts are the actual local invocation records; no tool-attested model identity is inferred from them.

## Qualified claim decisions and blockers

- **Supported:** exact closed October 6 fixed-native-winner comparison and the reported paired transitions, qualified to the archived native data contract and one snapshot.
- **Supported:** the fixed blend improves the loose count relative to pure Mask in this snapshot while reducing the strict count; it fails the joint gate.
- **Supported only as an arithmetic hypothesis:** the row-aligned old-regression/current-Mask hybrid and its negative strict-threshold tradeoff.
- **Unsupported:** full EG-3DVG reproduction, a new current same-forward model result, weight promotion, broad robustness/generalization, three effective innovations, or completion of the full ScanRefer plus Nr3D/Sr3D objective.

`blocking_findings=[]` for accepting the narrow cached arithmetic. The separate **claim blockers** are missing same-forward current native regression outputs for a current-model blend claim, absence of the required executed module/dataset evidence for the full goal, and the fact that both blends fail the stated joint numerical gate. No reviewer verdict may replace these missing executions. Further GPU work is not required to accept this CPU recount, and was not requested or performed by this audit.

The complete machine-readable judgment is `EXPERIMENT_AUDIT.json`. The sealed executor `SUMMARY.json` is unchanged, including its historical pending-review field; this separate report records the actual completed review.
