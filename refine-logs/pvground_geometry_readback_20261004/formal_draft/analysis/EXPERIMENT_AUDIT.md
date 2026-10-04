# PV-Ground frozen4506 readback terminal experiment audit

Date: 2026-10-05. **Overall verdict: WARN. Integrity status: warn. No blocking finding for reporting this completed, negative, single-seed comparison.**

Fresh delegated Codex audit, `review_independence=same-family`, `acceptance_status=provisional`. Astra / max was requested; **actual backend model and actual reasoning effort are unknown**, with no authoritative attestation available. This is an actual terminal audit, not a promotion of the earlier source review or preflight to accuracy proof.

The original result rows, receipts, sources and `SUMMARY.json` were read without modification. No training, inference, remote access, credential-wrapper read, weight load, weight deletion, or author analyzer execution was performed. A separate stdlib CPU checker independently reconstructed the saved Box statistics.

## Result supported by the audited evidence

| System | Formal rows | Box hits >.25 | Box hits >.50 | Acc@.25 | Acc@.50 |
|---|---:|---:|---:|---:|---:|
| Protected frozen4506 geometry parent | 9508 | 5616 | 4506 | 59.06604964% | 47.39167017% |
| Evidence hidden | 9508 | 5616 | 4475 | 59.06604964% | 47.06562894% |
| Evidence visible | 9508 | 5615 | 4477 | 59.05553218% | 47.08666386% |

The visible arm repairs 5 and damages 3 hidden-arm outcomes at strict IoU >.50: **+2 / 9508 = +0.02103492 percentage points**, with 63 selected-query changes. Relative to the protected parent, hidden repairs/damages = 67/98 (net -31), visible = 66/95 (net -29). The same-frame pre/post-readback diagnostic has those same net counts, but is a distinct comparison. Neither arm improves on the protected geometry parent or meets 5615/4754. The protected best remains 248 strict hits below the target. Evidence: `complete/evidence_hidden/formal/receipt.json:16`, `complete/evidence_visible/formal/receipt.json:16`, `complete/status.json:30`, `controller.py:135`, `analysis/AUDIT_CPU_CHECK.json`, and `analysis/AUDIT_PAIRED_CHANGES.csv`.

## A–F checks

| Check | Status | Judgment |
|---|---|---|
| A. Ground-truth provenance | PASS | Executed loader source derives target Box/Mask from dataset object annotations and points. No model-produced target was found. Raw scene data were not independently re-extracted. |
| B. Score normalization | PASS | Accuracy denominator is evaluated examples; IoU uses geometric union. No accuracy divided by a prediction maximum, minimum, or mean. |
| C. Existence, identity, numbers, coverage | PASS | All 40 terminal intake artifacts match size and SHA256; both full row sets, training traces, exits, CSV and summary agree with independent recorded-data reconstruction. |
| D. Invoked runtime and dead code | WARN | Terminal metric, restoration and retention routes are reachable and recorded as executed; inherited unused helpers and unexported tensor evidence limit independent replay. |
| E. Scope and causal interpretation | WARN | Two arms, seed 2027, ScanRefer developer validation only. Module holdout is pretrained-seen. Full independent forwards are demonstrably not bitwise identical. |
| F. Evaluation classification | PASS | `real_gt` for initial/terminal holdout and formal Box/Mask evaluation. Fixed-frame replay is a real-GT forward diagnostic, not an independently trained ablation. |

### A. Dataset GT and inference inputs

The imported dataset file is pinned as SHA256 `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d` in both actual `complete/*/imports.json:8` and `:16`; the independently found local archive matches. It loads `DATA_ROOT_mcln_meshsp/{train,val}_v3scans.pkl` (`joint_det_dataset.py:205–209`) and `ScanRefer/ScanRefer_filtered_{train,val}.txt/.json` (`:585–648`). The target ID comes from the annotation's `object_id` (`:637`). `_get_target_boxes` selects that object's dataset point memberships and `scan.get_object_bbox`, then converts min/max to center/size (`:1086–1119`). The returned GT is `center_label`, `size_gts`, `gt_masks` (`:1388–1392`), not predictions.

Here `joint_det_dataset.py` means `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py`, whose identity was checked against actual imports. Its dataset-side visual helper is **not** the initial PV runtime helper: the exact `appearance_source_manifest.json:470` SHA `6d1d4f0c792e8b86f238d88c02f1933f945313a3831d0bf9eabb9bf4580bb177` matches `C:/Users/gb/.codex_mcln_g0_20260905/src/visual_data_handlers.py`. That source loads ScanNet mesh coordinates and annotation segment/aggregation JSON (`:95–105`, `:129–163`); dataset bounding boxes are axis-aligned extents of the sampled object points (`:225–227`, `:246–258`). Thus these are the installed dataset/native sampled-point GT boxes, not a separately verified full-mesh box convention.

The inference dictionary contains points/voxels, expression, superpoints, and GroupFree detected boxes/classes (`run_readback_fit.py:240–253`). The latter come from prediction `.npy` input files, separate from GT (`joint_det_dataset.py:1189–1223`); `butd_gt=False`, `butd_cls=False` are enforced (`run_readback_fit.py:180–190`, `runtime_bundle/readback_model_factory.py:15–23`). GT is joined to predictions **after** the model call for losses/evaluation (`run_readback_fit.py:255–261`, `:285–301`). The bbs text-role maps are generated from parsed expression spans (`joint_det_dataset.py:981–1082`, `:1872–1899`), not from predicted boxes or candidate GT-IoU gates. `filter_non_gt_boxes=False` is explicit (`run_readback_fit.py:277–278`). Full256 oracle flags and GT volume groups are evaluation-only diagnostics (`:305–321`; `analyze_closed_formal.py:133–140`).

**Limit:** the original `.pkl`, scene meshes/segment annotations, text annotation JSON and raw Mask tensors are not in the supplied/local export. Local searches found no matching original train/val scene caches or ScanRefer annotation files. GT provenance is verified in the identity-matched source and recorded GT consistency, not independently re-extracted from those remote original datasets. This limits authenticity attestation; it is not evidence of synthetic GT.

### B. Exact native bbs and normalization

For final semantic logits `z[q,t]`, let `p[q,t] = softmax_t(z[q,t])`. The deployed root score is, in the actual addition order:

```text
s[q] = sum_t p[q,t] * 1[positive_map[0,t] > 0]
     + sum_t p[q,t] * modify_positive_map[0,t]
     + sum_t p[q,t] * pron_positive_map[0,t]
     + sum_t p[q,t] * rel_positive_map[0,t]
     - sum_t p[q,t] * other_entity_map[0,t]
q* = argsort_descending(s[0:256])[0]
```

Evidence: `runtime_bundle/native_root_bbs.py:4–9`; identity-matched native `grounding_evaluator.py:218–303`, `:535–553`. Only the main positive map is binarized for evaluation; modifier/pronoun/relation/other maps retain their native values. The training CE target's .6/.2/.2/.1 coefficients are **not** substituted into inference scoring. This is `bbs` / position alignment, not the separately computed `bbf` contrastive score.

The selected final Box and Mask share `q*`. Mask uses `sigmoid(alpha * text_mask_logit[q*] + (1-alpha) * query_mask_logit[q*]) > .5`, then native superpoint-to-point indexing (`run_readback_fit.py:307–323`; native evaluator `:594–650`). Box correctness is strict `IoU > threshold`; reported accuracy is `100 * hits / rows`, and mean Mask IoU is `100 * sum(row_mask_iou) / rows` (`run_readback_fit.py:334–342`, `analyze_closed_formal.py:78–81`). IoU divides intersection by geometric union (`run_readback_fit.py:263–269`; native `models/losses.py:35–75`).

The boundary probability centering and Mask support normalization are internal feature/decode calculations (`runtime_bundle/pvground_boundary_box_refiner.py:18–22`; `whole_mask_range.py:40–67`), not normalization of reported performance by prediction statistics. No second quality ranking, teacher, inference GT gate, or candidate subset was found. Native evaluator also computes `bbf` as a separate diagnostic, but the selected output/retention path uses bbs only (`grounding_evaluator.py:194–206`; `controller.py:100–117`).

### C. Actual artifacts and independent CPU reconstruction

The checker `analysis/audit_cpu_check.py` does not import or execute the training/evaluation/analyzer modules. It uses separate scalar float64 box-intersection arithmetic and reads every NDJSON record. Its actual output is `analysis/AUDIT_CPU_CHECK.json`; full input identities are `analysis/AUDITED_INPUT_SHA256.json`.

- **40/40 terminal intake artifacts, 51,378,753 bytes** match the original `complete/INTAKE.json:3–164`; no listed artifact is missing and no unlisted terminal artifact is hidden. Five exit files are `0`; all four controller stages completed. `status.json`, the parsed controller log, `wait.json` and intake terminal status agree. Actual completion is `2026-10-05T03:43:16.306947+08:00` (`complete/status.json:2–28`, `:72`; `wait.json:78–101`).
- **56,072 evaluation records** were reconstructed: 2 × (6887 initial + 6887 terminal + 9508 formal), plus 9508 parent rows. Selected bbs Box, saved coarse Box, and both arms' saved bypass Box were checked against their recorded dataset GT boxes. There are **zero @.25/.50 threshold disagreements**, including coarse and bypass checks. Float64 and recorded GPU float32 IoU values can differ by a few millionths; they are not reported as exact floating matches.
- Each formal row ID is exactly `0…9507`, in order. Each holdout stage has exactly the protocol's 6887 IDs. Pairing verifies `row_id`, `scan_id`, `target_id`, `root_box`, and `point_sha256`, not merely list length. Formal covers 141 recorded physical scenes; holdout covers 106. Recorded holdout/formal physical scene overlap is zero. All 106 recorded holdout scene hashes satisfy the specified salt/fold rule.
- Actual training is **29778 = 3722 × 8 + 2**, one pass, 3723 updates, no accumulation or dropped tail. The two 3723-line traces have exact step sequences and identical 29778-element sample order, with no repeat or omission and exact membership in fit IDs. Order SHA256 is `1ab6f45572237ad8be0fc003f4a1e8cd872a4f41f6e7af66b300364e03c2810e`; first IDs `[14307,26871,13547,1622,9672,18692,29949,23417]`, final IDs `[4176,17320]`. Every stored loss/gradient/timing is finite, and all 59 progress rows in each train log match the full trace. Evidence: `run_readback_fit.py:236–238`, `:383–423`; `complete/*/train.jsonl:1` and `:3723`.
- Recomputed hits, all direct and between-arm repairs/damages, recorded oracle aggregates, and GT volume quartiles agree with `SUMMARY.json` and CSV. At formal @.50, hidden/visible full256 stored coverage is 7890/7890; errors with a recorded qualifying candidate are 3415/3413, without one 1618/1618. Top16/32/64/256 stored oracle counts are hidden `[5524,6027,7001,7890]`, visible `[5524,6029,7000,7890]`. These counts re-aggregate **saved boolean flags**, not raw full256 boxes.

| GT volume quartile | Rows | Hidden >.50 | Visible >.50 | Repairs / damages | Net |
|---|---:|---:|---:|---:|---:|
| 1, smallest | 2377 | 879 | 879 | 1 / 1 | 0 |
| 2 | 2377 | 1058 | 1059 | 2 / 1 | +1 |
| 3 | 2377 | 1099 | 1101 | 2 / 0 | +2 |
| 4, largest | 2377 | 1439 | 1438 | 0 / 1 | -1 |

Quartiles sort **recorded dataset GT volume**, then row index to resolve ties; each has 2377 rows. They do not define an inference policy. The eight strict between-arm changes are individually recorded in `AUDIT_PAIRED_CHANGES.csv`: repairs at row IDs 165, 2287, 5330, 8331, 8963; damages at 2465, 4637, 6491. Corresponding original JSONL lines are ID+1, e.g. `complete/evidence_visible/formal/rows.jsonl:166` vs hidden `:166` changes CPU IoU 0.4807496677 → 0.7898661141.

**Source identity:** both actual import receipts match the locally inspected dataset, PV model, loss, main utility, preparation and native evaluator bytes. All 16 runner/helper payloads match both arm specs; top-level and payload runner copies match; module source matches the pinned source port. The 28 actual corrected-V2 preflight intake files also match their archived hashes; both real two-update receipts exist and are separately classified as preflight evidence. They are not used as terminal score proof.

One identity trap was resolved without changing experiment evidence: initial `pvground_runtime_bundle_20260908_v1/env_spec.json` has canonical SHA `c73eed09…` and is **not** the executing environment's spec. The exact pinned canonical SHA `966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c` was independently located at `C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_runtime_20260908_v1/witness_repair_v3/env_spec.json`. The first checker assertion exposed the stale candidate; the original checker and error/resolution trace are retained. This is a resolved archive-selection issue, not a mismatch between the executed receipt and final selected archive.

### D. Runtime, original G, restoration and retention

The factory reconstructs official PV → G → frozen4506 distribution geometry in strict state-key/shape/dtype order, then freezes all parent parameters and installs R (`runtime_bundle/readback_model_factory.py:18–69`). Both arms have **96672 trainable parameters / 23 tensors** and the same parents, seed, optimizer, full expression, query and six face roles. The only spec differences are output root, preflight root and `use_geometry_evidence`. The visible/hidden intervention zeros only the 44-channel evidence before its encoder (`pvground_boundary_evidence_readback.py:33–75`); it does not remove the evidence branch or enlarge the other arm.

Final semantic scoring is actually deferred (`revision2/source_preview/PV-Ground/models/modules.py:135–178`; `models/pv_ground.py:507–515`), with native Mask production before frozen geometry refinement, R, then a **single final native semantic head** (`pv_ground.py:523–576`). `observed_readback_forward` hooks the three real modules and asserts order, exact semantic input/output, and cloned geometry/Mask/contrastive preservation within that forward (`runtime_bundle/readback_preflight_checks.py:19–68`). These assertions execute on every training and evaluation forward; trace rows record one head call and fixed geometry. A separately labelled cached pre-R semantic-head replay is called once only in evaluation (`run_readback_fit.py:288–291`, `:324–325`). It is not a second deployed ranking or a second backbone inference.

Original G is byte-identical to the archived September 18 G helper (SHA `3bf9c5f50095e7e447af91e7f61a34c930bf04ec8590f7e4da54134b859cf773`). G obtains detached final Box/root GT IoU, marks **all Hungarian matched queries** as excluded, and qualifies only unmatched `IoU > .5` (`pvground_semantic_assignment.py:9–23`). It replaces that slot's original eos-weighted CE with the native unnormalized .6/.2/.2/.1 root token target, subtracting old CE and applying `.5/7` once (`:26–50`). It does not alter matched slots, regression targets or contrastive positives. The hook's `matching[1]` is justified by native matcher invocation order proposal, last, 0head…4head (`models/losses.py:808–846`, `:849–917`; `run_readback_fit.py:389–396`). AdamW lr `1e-5`, decay `.0005`, clipping `.1` and parent-eval/R-train modes are explicit (`run_readback_fit.py:43–44`, `:215`, `:360–404`).

All persistent parent states are compared exactly with their initial values before terminal serialization (`run_readback_fit.py:421–424`), while every update checks no parent gradient (`:398–402`). A **new formal process** reconstructs parents, checks terminal/parent/spec identities and fit rows, strictly loads complete model state, restores AdamW and checks every model tensor plus all optimizer moments/steps/groups (`:216–234`; `whole_model_preflight_checks.py:30–44`). Both original `complete/*/formal_restore.json:2–11` report 23 optimizer states, one group, step 3723 and strict exactness. Their terminal hashes agree with fit and retention receipts.

Retention runs only after successful train/formal exits, complete receipts, 9508 rows and CPU threshold checks (`controller.py:72–109`). Selection requires strictly more hits@.50; deletion is confined to SHA-verified `terminal.pth` under the two owned arm directories and excludes protected parents (`:112–131`). Receipts record deletion of hidden SHA `61119a2019d565f2755617cbd1ecd8311b6eebecb4c76e39b2f50cf30c041a38` and visible SHA `9a2307c23191d75d5f6ef630f394b7479fc8ac088b77fbe9f969b6418936c0bb`, **1,284,112 bytes each / 2,568,224 total**. Official PV/G/4506 parent hashes are checked before and after deletion (`controller.py:39–44`, `:127`). Required reconstruction parents remain preserved according to the executing controller receipts (`complete/*/weight_retention.json:9–25`). The audited program has no V99 deletion path.

**D warnings:** model/AdamW tensor equality, raw Mask/full256 inference and remote deletion/parent existence were not independently re-executed: the authorized audit has text artifacts, and the two inferior R weights have already been removed. The exported rows omit full logits/maps, raw masks, all candidate boxes and per-step tensors. Hence selected Box and outcome arithmetic can be independently checked, while ranking generation, Mask IoU, candidate-oracle geometry and gradient qualifications remain source + original runtime-witness evidence. `readback_preflight_checks.py:71`, `:90`, `:103`, `:126` are imported into the formal runner but not called by its terminal path; they are called by the separately audited V2 preflight (`revision2/run_readback_preflight.py:200–242`). Inherited `distribution_loss` (`pvground_boundary_box_refiner.py:40`) is unused in frozen-provider R fit, and native `calculate_diou_3d` (`models/losses.py:103`) is not in the active loss. No result here is attributed to these unused helpers. The source's old “unexecuted draft” docstrings are stale; actual matching terminal receipts establish execution.

### E. Scope, observed numerical limits and claims

The run is a **ScanRefer, seed-2027, fixed-parent, one-pass R comparison**. The frozen geometry parent already received its separate fit; these arms reuse its learned geometry rather than train the entire architecture from a fresh official baseline. The holdout protocol has 29778 fit IDs / 456 reported physical scenes and 6887 holdout IDs / 106 scenes and explicitly says pretraining has seen the development holdout (`pvground_g_p2_20261002/complete/source/split_protocol.json:1`). The runtime reconstructs the scene salt and asserts physical disjointness before training (`run_readback_fit.py:146–156`, `:187–195`). The reviewer independently checked fit-ID membership and saved holdout scenes; original full training annotations were unavailable for a second 456-scene reconstruction. Formal val is a separate no-update process over all 9508 installed rows / 141 recorded scenes (`:166–185`, `:356–358`). Neither module holdout accuracy nor formal developer accuracy is an untouched test-set claim.

Frozen state equality and same-forward preservation **do not imply complete-forward bitwise repeatability**. Direct row comparisons establish:

- Initial hidden/visible choose the same query on all 6887 rows, yet **951** saved selected boxes differ in coordinates (maximum coordinate difference `5.230307579040527e-6`).
- Formal hidden/visible choose the same query on **9445** rows; **1441** of those saved boxes differ (maximum `4.470348358154297e-6`).
- Each arm's pre-R bypass chooses the same query as the historical parent on all 9508 rows, yet respectively **1460 / 1453** saved boxes differ (maximum `6.9141387939453125e-6`).

These are independently measured from the actual recorded boxes (`AUDIT_CPU_CHECK.json.recorded_same_query_cross_forward_drift`), consistent with actual V2 `disabled_repeat_differences.exact=false` and `zero_cross_forward_differences.exact=false` (`revision2/complete_preflight/evidence_visible/preflight.json:4`, `:472`). They do not negate the within-forward tensor snapshot checks. They do prevent an unqualified claim of identical full numerical candidate tensors across arms/history. No saved paired strict hit difference was caused by these roundoff discrepancies in the CPU reconstruction.

The statement “geometry evidence is beneficial” is not established as a stable effect by +2 hits from one seed, especially with both arms below the parent. Fixed-frame head replay diagnoses trained-R intervention on a cached query; it is not an independently trained no-R ablation. GT volume/oracle analyses are post hoc diagnostic views, not deployed decision rules. No Nr3D/Sr3D result, cross-dataset generalization, stable multi-seed improvement, three successful contributions, or novelty priority is demonstrated. No external literature novelty audit was performed. The supplied `analysis/RESULTS.md:18–22` and handoff tail (`MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md:22252–22280`) already preserve these limitations or timestamp their earlier partial state.

### F. Classification

`initial`/`terminal` holdout and `formal` Box/Mask evaluation: **real_gt** (installed ScanRefer annotations + ScanNet object membership/geometry). Full256 oracle/volume diagnostics also use real GT, but are offline diagnostics. The cached-head bypass has real-GT outcomes and is classified as a **fixed-frame forward diagnostic**, not synthetic proxy truth and not an independently trained ablation. The preflight's zero-residual/score/gradient equalities are implementation checks, not accuracy evaluations (`revision2/complete_preflight/*/preflight.json:2`; `run_readback_fit.py:288–291`).

## Findings and claim impact

**Blocking findings for qualified reporting of this terminal comparison: none.** No mismatch, fabricated metric denominator, prediction-derived GT substitution, or phantom terminal result was found in the evidence examined.

| ID | Severity | Finding / required qualifier |
|---|---|---|
| W1 | Nonblocking WARN | Original raw datasets, all-candidate logits/boxes and raw masks were not exported. Preserve source/runtime vs independent-CPU evidence labels; do not call this a complete independent inference rerun. |
| W2 | Nonblocking WARN | Inferior R weights were already deleted under the controller policy. Restore/deletion/parent-preservation exactness is supported by original runtime assertions and receipts; no new model/optimizer restore or current remote filesystem inspection occurred. |
| W3 | Nonblocking WARN | One seed, one dataset, pretrained-seen module holdout, repeatedly used developer validation. +2 hits is a descriptive outcome, not a reliable gain or causal proof beyond this controlled run. |
| W4 | Nonblocking WARN | Observed cross-forward coordinate drift prevents bitwise complete-forward identity claims; same-frame preservation remains separately supported. |
| W5 | Nonblocking WARN | Several imported preflight or inherited loss helpers are unused in formal R fit. Cite their actual preflight execution only; do not infer terminal diagnostics from their mere presence. |
| R1 | Resolved | Initial archived environment spec was stale; exact pinned witness-repair-v3 archive was located and hash-checked. Keep that identity in the reproduction manifest. |

The negative result, exact training budget/order, recorded native bbs metrics, and protected-parent nonpromotion are supported. A stronger end-to-end replay claim requires original datasets/tensors/weights that this audit did not have. Stable performance, novelty and cross-dataset claims remain unsupported by this experiment. No new training, broad refactor or forced rerun is required to report the qualified negative result.

## Identities, artifacts and forensic trace

All **117 read experiment evidence files** have full path, byte length and SHA256 in `analysis/AUDITED_INPUT_SHA256.json`; the machine audit embeds the same mapping. Key identities:

| Artifact | SHA256 |
|---|---|
| Formal runner | `773213e3035688df304d4dd88d63b8b27bbf636fec8af2f49be47f3339564a08` |
| Controller | `0ee58b61242d89a43be34fd43b5c2a3f77bd8bd6f0291db766967579e7c31777` |
| Native evaluator | `39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677` |
| Native losses | `920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de` |
| Executed model source | `8d2aad665ce7d5e19f7423309ef0ba05ba406f302159c6460a36323cbae1d1c9` |
| Delayed-semantic module source | `1f262505e4c9d6fd186609c6da1497bf5cf6f62f982fa7746c7d4f5400d8f2df` |
| Hidden formal rows | `998ff0180434374f63f5523167512dad3786b4d8a5014a27929d35ee517edfc4` |
| Visible formal rows | `f7e9e3306d6eb497ea2c53878ea93b5f2713b1720174b1ab6b05884b0468556f` |
| Parent formal rows | `5bcdaed7649f833ef52251ac5d481628cec0bcd3b9830b6f6b6736367abc6a0e` |
| Hidden train trace | `50d070711d065efd1f4af0d6799f36ccbc8c62a8dcf7246778f5819d18cc2e5a` |
| Visible train trace | `42fbfea82eb4708f98cbf29c8f3811464124c4402a9038ab1dafa14bffb698ad` |

Reviewer-owned deterministic artifacts: `analysis/audit_cpu_check.py`, `analysis/AUDIT_CPU_CHECK.json`, `analysis/AUDIT_PAIRED_CHANGES.csv`, `analysis/AUDITED_INPUT_SHA256.json`. Deterministic saved-data arithmetic passed within its explicit scope; the semantic experiment audit remains **same-family / provisional / WARN**.

Private trace: `formal_draft/.aris/traces/experiment-audit/2026-10-05_readback_terminal_run01/`. It contains the actual request, full audit response, route metadata with unattested backend fields, original failed checker/error and resolution, final checker/stdout, and check/input hashes. This audit does not overwrite `SUMMARY.json`'s original `integrity_review_pending` field; this separate report is the terminal review artifact.
