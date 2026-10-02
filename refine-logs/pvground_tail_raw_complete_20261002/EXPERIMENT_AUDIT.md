**Overall verdict: WARN. Blocking findings: none for the completed raw result and the exact corrected publisher reviewed below.** Raw formal `bbs` is **5594/4457** at strict IoU >0.25/>0.50. The paired fused attempt **failed after E0, before optimizer creation: zero updates and no post-training result**.

Reviewer: **gpt-6-astra / max**, fresh evidence review, `review_independence=same-family`, `acceptance_status=provisional`. Audit date: **2026-10-02**. The later failed-attempt evidence was incorporated at the coordinator's explicit direction. This is a local source/artifact and independent standard-library CPU audit. I performed no SSH, model/training/diagnostic forward, publisher execution, repository mutation or checkpoint deserialization. Only these two requested reports were written.

The audit reads and hashes **73 actual files**: the 60 originally listed artifacts, the request, three failure-supplement records and all nine files referenced by the failure intake. The original raw evidence remains intact. The new publisher has ten review-bound inputs and thirteen explicit payloads under the authorized scope update.

| Check | Verdict | Result |
|---|---|---|
| A — GT and object inputs | PASS | Native GT path and predicted-box input protocol traced |
| B — Arithmetic, budget and bindings | PASS | Full independent recount, pairing, fit order and archive hashes agree |
| C — Execution and current claims | WARN | Raw complete; fused failed at E0 with zero optimizer updates |
| D — Invocation, order and gradients | WARN | Actual paths verified; preflight and dormant-code limits retained |
| E — Scientific scope | WARN | One seed, saved/clamped geometry and partial replay scope |
| F — Publisher | PASS, static only | Corrected failure status, bounded writes and guarded nonforce publication |

Path aliases in the exact file:line references and inventory:

- `C` = `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\complete_tail_raw`
- `Z` = `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\failed_tail_fused_initial`
- `S` = `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002`
- `H` = `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\g_control`
- `G` = `C:\Users\gb\.codex_pvground_cs_20261002\refine-logs\pvground_scanrefer_formal_20260918_semantic_assignment_v1`
- `A` = `C:\Users\gb\.codex\archives\pvg_tail_raw_20261002`
- `P` = `C:\Users\gb\.codex\tmp\pvground_p3_next_20261002`

**A — PASS: dataset GT and object-input protocol.**

ScanRefer annotation `object_id` becomes `target_id`; the first target is the referred instance, `scan.get_object_bbox(tid)` supplies its box, and instance-point indices supply `gt_masks`. The loader emits `center_label`/`size_gts`; the native criterion uses these targets, the evaluator takes the root box, and the runner saves that same target as `root_box`. Evidence: `C\source\imported\src.joint_det_dataset.py:596`, `:634`, `:1086`, `:1102`, `:1106`, `:1388`; `C\source\imported\models.losses.py:857`, `:872`; `C\source\imported\evaluator.py:535`; `C\run.py:450`, `:455`.

The model receives XYZ/RGB, text, superpoints and detected object boxes/categories. The source explicitly loads GroupFree prediction files, uses `butd=True`, and disables `butd_cls`/`butd_gt` substitutions. Target boxes and masks enter the criterion/evaluator after the model forward, rather than the input dictionary. This is an object-assisted input protocol. Evidence: `C\source\imported\src.joint_det_dataset.py:1204`, `:1361`, `:1367`; `C\run.py:132`, `:232`, `:263`, `:275`; `C\source\imported\models.pv_ground.py:377`.

G's detached unmatched-query qualification, `IoU > 0.5`, is a geometric training proxy. It excludes Hungarian-matched queries and supplies token-label replacement; it does not redefine dataset GT or directly supervise the box refiner. Observed support counts, nearest-point distances and predicted Mask probabilities also do not establish instance identity or teacher knowledge. Evidence: `C\source\pvground_semantic_assignment.py:9`, `:23`, `:34`; `C\source\pvground_observation_query.py:19`; `C\source\pvground_tail_support_box_refiner.py:15`.

This is a source-to-saved-target trace. Original annotations, instance point assets, detector arrays and retained parent checkpoint bytes were not independently reopened within this local audit.

**B — PASS: independent arithmetic, denominators, pairing and hash bindings.**

I independently parsed all 46,564 evaluation records across the three raw and three historical-control stages, plus all 7,446 training records. A standard-library corner-intersection implementation replayed 139,692 saved selected/coarse box IoUs. Every raw and historical selected strict `>0.25`/`>0.50` decision agrees with its saved IoU; every raw coarse decision also agrees. The largest raw selected/coarse continuous discrepancy is **3.6777091214634794e-6**. All result fields checked against `C\CPU_RECOUNT.json` reproduce; this is not bitwise GPU arithmetic replay.

REC accuracy is row hits divided by 6,887 or 9,508, multiplied by 100. Mask mIoU is the arithmetic mean of the saved per-row IoUs multiplied by 100. There is no scene-macro or batch-mean denominator. Both REC and saved Mask hits use strict thresholds. The native evaluator and runner agree on root-only top-1 selection; `bbs` uses token-position scores, while `bbf` uses normalized query/token similarity at temperature 0.07. Evidence: `C\run.py:433`, `:446`, `:468`, `:488`; `C\source\imported\evaluator.py:281`, `:296`, `:325`, `:485`, `:649`, `:897`.

| Stage | Mode | Rows | REC hits >.25/>.50 | REC % >.25/>.50 | Mask hits >.25/>.50 | Mask mIoU % |
|---|---|---:|---:|---:|---:|---:|
| initial | bbs | 6887 | 6176/5602 | 89.67620154/81.34165820 | 6180/5818 | 74.19939023 |
| initial | bbf | 6887 | 6206/5647 | 90.11180485/81.99506316 | 6208/5833 | 74.44013545 |
| terminal | bbs | 6887 | 6157/5579 | 89.40031944/81.00769566 | 6139/5770 | 73.60350559 |
| terminal | bbf | 6887 | 6171/5610 | 89.60360099/81.45781908 | 6152/5779 | 73.72184153 |
| formal | bbs | 9508 | 5594/4457 | 58.83466554/46.87631468 | 5796/5112 | 47.06109268 |
| formal | bbf | 9508 | 5597/4472 | 58.86621792/47.03407657 | 5801/5108 | 47.04073307 |

For the same selected query, the independently replayed coarse-to-final changes are:

| Stage | Mode | Coarse hits >.25/>.50 | >.25 repairs/damages/net | >.50 repairs/damages/net |
|---|---|---:|---:|---:|
| initial | bbs | 6176/5602 | 0/0/0 | 0/0/0 |
| initial | bbf | 6206/5647 | 0/0/0 | 0/0/0 |
| terminal | bbs | 6162/5579 | 2/7/-5 | 25/25/0 |
| terminal | bbf | 6171/5604 | 5/5/0 | 29/23/+6 |
| formal | bbs | 5595/4448 | 4/5/-1 | 24/15/+9 |
| formal | bbf | 5597/4463 | 5/5/0 | 23/14/+9 |

These are within-forward intermediate comparisons, not independently trained no-refiner baselines. Evidence: `C\run.py:441`, `:478`; `C\arm\formal\rows.jsonl:1` through its 9,508 records; `C\CPU_RECOUNT.json:207`.

Initial-to-terminal holdout changes are `bbs`: **106/125/-19** at .25 and **259/282/-23** at .50; `bbf`: **95/130/-35** and **252/289/-37**. Initial raw and historical-control selected query, box, IoU, row/scene/target/root-box identities and point hashes agree exactly. The retained initial Mask difference is row **18482** in both modes; equality cannot be extended to all outputs. Evidence: `C\arm\initial_control_comparison.json:2`; `C\arm\receipt.json:79`; `C\run.py:506`.

The paired historical-continuation comparisons independently reproduce:

| Stage | Mode | >.25 repairs/damages/net | >.50 repairs/damages/net |
|---|---|---:|---:|
| terminal | bbs | 132/119/+13 | 315/288/+27 |
| terminal | bbf | 114/118/-4 | 280/270/+10 |
| formal | bbs | 234/240/-6 | 345/340/+5 |
| formal | bbf | 211/242/-31 | 331/324/+7 |

All paired row, scene, target, root-box and point-hash identities match. Historical formal `bbs` is **5600/4452**. The separately supplied original-G aggregate receipts agree on **5615/4495**, giving raw deltas **-21/-38**; original-G per-row predictions were not supplied for a new replay. Evidence: `H\formal\receipt.json:18`; `G\receipt.json:14`, `:61`; `G\audit.json:25`; `C\CPU_RECOUNT.json:300`.

All 3,723 fit batches match the historical control in order and row content: 3,722 batches of 8 and one of 2, ending with rows **4176, 17320**. Each of the 29,778 fit IDs occurs exactly once; no holdout ID appears, and fit plus holdout partition IDs 0–36,664. All logged numbers are finite. Saved loss equals native loss plus correction within float serialization precision, and the correction agrees with `(new_ce-old_ce)*0.5/7`. Evidence: `C\arm\train.jsonl:1`, `:3723`; `H\train.jsonl:1`, `:3723`; `C\source\split_protocol.json:1`; `C\run.py:544`.

Native CE retains the unnormalized ScanRefer token mixture **0.6/0.2/0.2/0.1**, EOS weight **0.1**, and matched-box denominator. Geometry losses use matched-box normalization; native Mask focal loss averages superpoint positions then divides by matched boxes, Dice uses its +1 numerator/denominator smoothing and the same box denominator, and Mask training targets are majority-pooled instance GT; native layer aggregation uses seven heads and ScanRefer's **0.5** CE/contrastive coefficient. G replaces only qualified unmatched CE terms with the same EOS/box denominator and `0.5/7` multiplier. The current run is batch 8, seed 2027, one fit pass, LR/backbone LR 1e-5, weight decay 0.0005 and clip norm 0.1, using fresh AdamW. Inherited manifest values such as batch 12/2,482 steps are not this run's executed budget. Evidence: `C\source\imported\models.losses.py:393`, `:437`, `:579`, `:484`, `:504`, `:831`, `:943`; `C\source\pvground_semantic_assignment.py:26`, `:40`; `C\arm\spec.json:9`; `C\run.py:174`; `C\source\imported\main_utils.py:339`.

All **42** intake-listed files match both byte counts and SHA256. Stage-row hashes, training-log hash, run/spec hashes, custom-module hashes, split hash and all six actually imported source hashes match their receipts. The actual imported dataset intentionally comes from the separate detection-aligned source tree; its hash is the imported-file hash, not the different dormant dataset copy listed in the model-tree port. Evidence: `C\INTAKE.json:6`; `C\arm\imports.json:3`, `:10`; `C\source\source_port.json:94`; `C\run.py:104`; `S\collect_complete_arm.py:84`.

The archived binary is **346,718,993 bytes**, SHA256 **50961f4be97d5e02cef158b32f3c85300308f7ce044b7668def07277b3df85bd**, matching training, intake and archive receipts. I hashed it without deserializing it. The writer/receipts identify a state delta plus optimizer/RNG requiring retained author and G parents; this audit does not certify a self-contained deployable model or revalidate all upstream manifest blobs. Evidence: `A\archive_receipt.json:3`; `C\arm\receipt.json:75`; `C\run.py:527`; `S\collect_complete_arm.py:94`.

**C — WARN: raw completion is verified; the fused attempt failed before training updates.**

All requested raw artifacts exist. The raw preflight, train and formal exit files contain **0**. Every JSON-bearing line in the saved raw logs was parsed; progress records match training rows, and final logged objects exactly match the stage/training receipts. Training completed 3,723 updates, followed by terminal holdout evaluation and a separate formal evaluation; controller completion is **2026-10-02 22:27:22.704685 CST**. The training receipt's `formal_rows=0` describes its own phase, while the subsequent formal receipt records 9,508. Evidence: `C\arm\train.exit:1`; `C\arm\formal.exit:1`; `C\preflight\preflight.exit:1`; `C\arm\train.log:83`, `:97`, `:98`; `C\arm\formal.log:26`; `C\status.json:17`.

The launch receipt records a fused attempt at **22:30:29.233635 CST**, controller 267118, and the capacity comparison **835,325,952 >= 827,444,774 bytes**. The later **22:52:00.724601** boundary supersedes any running-state interpretation: it records failure/exit **1** after E0 completed at **22:49:53.389742**, with `AssertionError: ('bbs', 'box')`. The archived fused spec's actual comparison target is **same-tail raw**. Failure occurs at `run.py:515`; optimizer creation follows at `:524`, proving **zero optimizer updates in this attempt**. The earlier `load.json` flag `fresh_optimizer=true` was emitted before creation and is only a setup declaration. Evidence: `S\tail_fused_training_launch.json:2`, `:7`; `S\tail_fused_training_boundary_225200.json:4`, `:50`, `:52`, `:63`; `Z\arm\spec.json:59`; `Z\arm\train.log:22`, `:28`; `C\run.py:165`, `:515`, `:524`.

I additionally read/hash-checked all **nine** files named in the failed-attempt intake, parsed its full 6,887-row E0 file/log, and independently reproduced every DIFF changed-ID list, count, maximum, threshold count and example for both references. The archived runner bytes match the raw runner. Fused E0 selected/coarse box threshold replay and Mask-IoU recount match its receipt. The extra saved rows bring the total evaluated records inspected to **53,451**.

Against same-tail raw, all saved input/GT identities agree, but `bbs` changes query **225 -> 68** at row **26603**, and `bbf` changes **203 -> 116** at row **26641**. Maximum selected-box coordinate changes are **0.1963639259338379** and **0.014290809631347656** in inherited coordinate units. **6,775/6,887 bbs** and **6,802/6,887 bbf** maximum coordinate changes are <=1e-5; the query-switch cases are materially larger. All four per-row strict REC decisions remain unchanged, leaving E0 counts **6176/5602** and **6206/5647**. Same-tail Mask IoU differs at row **16804** in both modes; versus historical G it differs at **16804 and 18482**. Identical hit counts therefore do not establish exact forward equality or justify blanket tolerance relaxation. Evidence: `Z\DIFF.json:25793`, `:25806`, `:51439`, `:51446`, `:51452`; `Z\arm\initial\rows.jsonl:1`; `Z\INTAKE.json:42`.

The **current** publisher correctly records this failure, withdraws the old finish estimate, preserves the failed evidence and gives no fused post-training result. Its raw counts, percentages, repairs/damages, coverage totals, 42-file count, parameter count, archive scope and original/historical-G deltas match the inspected artifacts. Its failure counts and query changes match the independent supplemental recount. Evidence: `S\publish_raw_result_and_fused_launch.py:53`, `:57`, `:62`, `:64`, `:66`, `:68`, `:157`.

The appendix's statement that fixed-input/weight/RNG runtime diagnosis is ongoing is executor status, not independently evidenced by this supplied file set. No diagnostic model execution, fix, relaunch or diagnosis result is credited here. Launch-time GPU/retention statements are saved observations, not fresh live attestations; metre labels inherit the dataset convention. Nr/Sr and retained V99 receive no new execution/performance credit. These limits do not block publishing the verified raw result and explicitly failed attempt.

**D — WARN: actual call order is supported; preflight and dormant paths have narrower meaning.**

The runner actually imports the recorded evaluator and calls `evaluate(..., 'last_')` with root-only, thresholds [.25,.5], topks [1,5,10] and `filter_non_gt_boxes=False`. All four native box/Mask evaluation methods run. Optional detector-overlap filtering and visualizations are dormant; they do not define the reported score. The new tail executes after the proposal, six decoder heads, query projection, native Text/Query Mask generation and adaptive weights, then changes only final center/size tensors. Evidence: `C\run.py:116`, `:433`, `:445`; `C\source\imported\evaluator.py:194`, `:240`, `:67`; `C\source\imported\models.pv_ground.py:454`, `:505`, `:520`, `:551`, `:556`.

Both support arms have a 14-dimensional member input and **367,494 refiner parameters**, independently calculated from the five Linear layer shapes. Both perform the native Mask gather; raw replaces the four gathered support channels with zeros. Fused uses Text probability, Query probability, absolute probability difference and sigmoid of native weighted logit fusion. The native Text mask is expanded across the 256 queries; the Query mask supplies query-specific support. This is not the earlier 10-input refiner protocol. Evidence: `C\source\pvground_candidate_box_refiner.py:23`; `C\source\pvground_tail_support_box_refiner.py:13`, `:20`, `:45`; `C\source\imported\models.pv_ground.py:540`.

The saved raw preflight contains four independently checked event lists: refiner-enabled/disabled evaluation forwards and two update forwards. Each has one proposal and each of the six heads, one query projection, five Text-mask calls and one Query-mask call per sample, three SWA/FFN passes per sample, then the optional single refiner. The two optimizer steps use the same eight-row probe batch and show direct geometry-output gradient norms **6.54449749/7.25900793**; all eight separately tested native Mask losses and the direct semantic loss have zero/unused gradients to the refiner. Raw geometry-to-Mask-output gradient is zero. The zero output layer makes internal refiner gradients zero at the first update; nonzero member/condition/aggregate gradients appear at the second. Logged parameter gradients are captured after clipping. Evidence: `C\source\pvground_tail_preflight.py:13`, `:24`; `C\run.py:300`, `:328`, `:390`; `C\preflight\preflight.json:543`, `:565`, `:594`, `:616`.

Zero-init equality covers only `last_sem_cls_scores`, `last_center` and `last_pred_size` on that one reset-RNG batch. The member-mask mapping check covers all eight samples but query IDs 0,42,255 only. It is not universal equality of every output or a precision result. Shared trainable upstream parameters, discrete matching changes and later updates can still alter semantic/Mask behavior even with zero direct autograd routes to the refiner. No fused gradient witness is inferred from raw preflight evidence. Evidence: `C\run.py:340`, `:360`, `:374`; `C\preflight\preflight.json:656`.

The `verify_native_replacement` reconstruction diagnostic is only called under `if not update`, while the current preflight and fit call `step(..., True)`. It receives no execution credit. Its loss algebra was inspected statically. DIoU helper/commented alternatives are also not the active loss; the source computes a query-generation quantity but does not add it to the shown total. Evidence: `C\run.py:322`, `:391`, `:546`; `C\source\imported\models.losses.py:103`, `:936`, `:947`.

**E — WARN: scientific and replay ceiling.**

The supported outcome is one seed-2027 ScanRefer raw-control continuation and its saved results. The 6,887-row module holdout contains 106 physical scenes and is explicitly marked previously seen by parent pretraining. It is distinct from the 9,508-row formal development validation on 141 physical scenes. Their saved physical-scene sets are disjoint; the full fit/holdout scene separation is asserted by the runner and recorded in the split. None of this establishes unseen-scene generalization or statistical significance. Evidence: `C\source\split_protocol.json:1`; `C\run.py:185`, `:232`, `:244`.

Both coarse and final saved sizes are floored at 1e-6. Formal `bbs` contains **49 coarse and 51 final size elements** at that floor; preflight records **1,281 negative coarse size elements**. The audit validates emitted boxes, not validity of unclamped head outputs. XYZ offsets, distances and residuals retain the upstream coordinate convention; no new metric-scale calibration was inspected. Radius-normalized observation statistics remain geometric proxies. Evidence: `C\run.py:441`, `:444`; `C\preflight\preflight.json:646`; `C\source\pvground_tail_support_box_refiner.py:37`; `C\source\pvground_observation_query.py:26`.

Mask summaries were independently recounted from saved IoUs, with source tracing to native instance-mask GT; binary masks were not saved for independent replay. Saved Full-256 coverage flags give formal raw `bbs` **8934/7797**, with **3340/3340** geometrically qualified but unselected rows. I verified flag domains, monotonicity, sums and consistency with selected hits. All 256 boxes and score tensors were not saved, so CPU work cannot independently reproduce full-candidate coverage or query ranking. Evidence: `C\run.py:474`, `:480`; `C\CPU_RECOUNT.json:210`, `:221`.

Historical G continuation is a descriptive matched-row reference, not the same-tail source-isolation control. Same-selected-query coarse/final repair counts are an internal forward comparison. The actual intended support-source comparison requires completed paired tail_fused versus tail_raw results under their shared protocol; the fused attempt represented by the newer boundary failed before optimizer updates. No causal gain, teacher transfer, robust Mask improvement, Nr3D/Sr3D performance or promotion follows from these artifacts. The primary raw strict hit count is **297 below 4,754**, the count required for 50% of 9,508. Evidence: `C\CPU_RECOUNT.json:299`, `:359`; `S\tail_fused_training_boundary_225200.json:4`, `:52`; `C\run.py:515`, `:524`.

Evidence classification: native training targets and saved REC/Mask evaluation originate in **real_gt**. G eligibility is a **GT-derived geometric proxy**; support counts/distances and predicted soft Mask support are **observational/prediction proxies**. Full-256 flags are **saved GPU GT-referenced diagnostics**. Preflight, archive receipts, launch and failure boundaries are **operational evidence**, with no automatic performance or execution credit beyond the recorded stage.

**F — PASS, static only: current bounded publisher and claim-preserving transport.**

The reviewed publisher is **12,367 bytes**, SHA256 **f545b377772b21ec46cc029c0af423e68506ad739a4c315bb375fb0c8dc69c4f**. Its initial seven substantive review inputs remain hash-bound; the authorized failure supplement adds the failure boundary, DIFF and failure intake, making **ten** bound inputs. The exact current publisher/collector/recount/result/intake/launch/completion-boundary and all three additions are in this inventory. It requires PASS/WARN with no blockers, checks each reviewed hash, then checks the raw numbers and actual fused failure. Evidence: `S\publish_raw_result_and_fused_launch.py:16`, `:18`, `:28`, `:35`.

The script requires prior handoff SHA256 **da59b197f226d82e0c2928a5cf7a8c32a36da2a4d212f7dc6230ace9dbd851f2**, the three recorded HEADs, four identical local copies and clean repositories. It verifies the old remote document before mutations, forms `new = old + UTF-8 appendix`, checks the exact prior prefix and a unique section marker, and rechecks all local/remote document bytes afterward. These are inspected runtime guards; this auditor did not open the live handoffs or query Git/SSH to assert that the future preconditions currently pass. Evidence: `P\selected_refinement_publication.json:4`, `:10`; `S\publish_raw_result_and_fused_launch.py:40`, `:45`, `:72`, `:102`, `:125`.

The authorized update expands the initial ten payloads to **thirteen explicit payloads**, adding three failure records. Writes are bounded to four local handoffs, the remote handoff, those payloads in the first two repositories/remote evidence directory, appended manifests/attributes, the synchronization guard's expected hash, and the publication receipt. There are no active experiment-source/checkpoint edits or deletions. The failure raw row/log files remain privately archived and hash-described; the thirteen published payloads do not include their raw data. Evidence: `S\publish_raw_result_and_fused_launch.py:75`, `:103`, `:110`, `:120`, `:148`, `:161`.

Scoped staging rejects unexpected status paths, runs the cached whitespace check and verifies every staged payload against source bytes. Repositories must be clean afterward. The single push is **git push origin HEAD:main**, with no force flag, followed by remote-ref verification. Committed document/payload blob IDs are compared across the first two repositories. The guard update requires exactly one old-hash occurrence. Evidence: `S\publish_raw_result_and_fused_launch.py:130`, `:138`, `:142`, `:145`, `:150`.

Remote reads use a quoted `cat --` SSH command, a 65,536-byte chunk loop and successful exit-status validation; they are the selected transport, not a fallback. Saved stdio-read evidence predates the latest prior handoff and has a different document hash/length, so it supports transport compatibility only. The source and witness do not certify future successful writes, commits, push or publication. Evidence: `S\publish_raw_result_and_fused_launch.py:94`; `P\publication_stdio_read.json:2`; `P\selected_refinement_publication.json:10`.

Publication receipt fields now distinguish historical launch from `failed_E0`, **0** optimizer updates and no formal result. Saved observer fields can be null and remain historical metadata, not liveness. This auditor executed no publisher, SSH, model, training, diagnostic run or repository mutation. **Blocking findings: none for this exact corrected publisher and the bounded archived claims.**

**Actual inspected-file SHA256 inventory**

All hashes below were calculated from the current bytes read. Full absolute paths, byte counts and hashes are also in the JSON report. Neither generated report is included in this inventory; the JSON separately records the Markdown SHA256.

| File | Bytes | SHA256 |
|---|---:|---|
| `C\arm\formal\receipt.json` | 685 | `485cac46e227a3d71596c9b62506dba60b5e3722d7b7d376a049b4403b5483f0` |
| `C\arm\formal\rows.jsonl` | 11,465,618 | `019210a51f6f6c54fea52d4bf0b67c56154d4840c9d96459857b904ca6910f79` |
| `C\arm\formal.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `C\arm\formal.log` | 3,225 | `0082771b80aaec9929791c544ce824fedab27c45d20ad2ade13237648afc23f2` |
| `C\arm\imports.json` | 1,256 | `f9bb4be0be4d61f3ca26c9c0c6b69a25a1d94b970f039815515debfa13edf5fd` |
| `C\arm\initial\receipt.json` | 683 | `3c264a7c99d707dc654d59b1ed61cbd74ab28164cca315c57af8d69bdf9fd444` |
| `C\arm\initial\rows.jsonl` | 8,460,712 | `ab71095b3f658471776e7c6af63c5908edea02169eee10b9f73a9c26657dd109` |
| `C\arm\initial_control_comparison.json` | 130 | `a62610b78752ebf6e7bad78dd9ddcae2b8a24bb029945944a6f14b86cc69a44c` |
| `C\arm\load.json` | 347 | `41cbb2499c94f308e030baee68120df885d5fb12c015d406ae8f97fd6c48b697` |
| `C\arm\receipt.json` | 3,341 | `1014f97a5c8a069173716fb9e03297b8464d94657a2529c9fc744429653e7af9` |
| `C\arm\spec.json` | 3,710 | `30f8a11ae02f0af3cae1ae3db6d5f225820b13b426c44466615cca01cf5caf13` |
| `C\arm\terminal\receipt.json` | 684 | `41e1ed2148d4ce7ef72de2b5927e6d936f1f012575f388fdbf624b964ec9d810` |
| `C\arm\terminal\rows.jsonl` | 8,459,584 | `c006df1980841b3d359e0bc988ace837dbb969c4376df60be21e26ca0e7405cc` |
| `C\arm\train.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `C\arm\train.jsonl` | 2,235,828 | `67cc528aa67679ce24a38d3ef53bf88f623efc72790bd402b5af45191426bfec` |
| `C\arm\train.log` | 44,410 | `f414af6c88caeb6d70b35cbb4f1cef89320e090a3fb8c1e2459dfeb2974a1796` |
| `C\controller.py` | 4,120 | `21774197eb1ee5a9c4e7742953c1d496e75d6192c80c82de24eb6084398bdcb6` |
| `C\CPU_RECOUNT.json` | 10,090 | `08fe504affc8ec90f3c37a5ef6f950bfa77a875b8c7c88a4a8ce77c49f3e676a` |
| `C\INTAKE.json` | 11,095 | `f5233d4c89e4bca0d4acb8fa7303057d4a7b39b5e22ac83be3d367cd68a551bb` |
| `C\preflight\imports.json` | 1,256 | `f9bb4be0be4d61f3ca26c9c0c6b69a25a1d94b970f039815515debfa13edf5fd` |
| `C\preflight\load.json` | 347 | `26d0e62ad5ce7537c3bc0f68d3d9e8b2fde1f49acb6961600bcfb32b1cf4452f` |
| `C\preflight\preflight.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `C\preflight\preflight.json` | 14,485 | `24b51e90a387fff1c938612f6eab1f73e3c59c51bbaba6c2a5170754ab47457e` |
| `C\preflight\preflight.log` | 10,594 | `a947d4b337a8a2c50a312777eff54d059ab0e38d2335acd229ac8f0dca816bb8` |
| `C\preflight\spec.json` | 3,720 | `667bd2c4984d550c89eab0150eaafcc4f687f163b9dc2103768889ea96d463fc` |
| `C\prepare_source.py` | 2,218 | `7487405bfd6865838720613c046a6c055d13f4a8d107cc5d26ea66b86e493f8e` |
| `C\run.py` | 38,567 | `483a7db4e0d11d8b79af63ea6e02f6d9455bed218190efc17d42e48929648346` |
| `C\source\imported\evaluator.py` | 41,473 | `39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677` |
| `C\source\imported\main_utils.py` | 24,599 | `14bd101eb78ec3d729968aff0a975684efa8388574583ca9e4b0f512877b4f5c` |
| `C\source\imported\models.losses.py` | 40,468 | `920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de` |
| `C\source\imported\models.pv_ground.py` | 25,386 | `cc17e6c235075c4be24c752bb76dcafe0273c7a036f3dcf6bedb3696379aaa58` |
| `C\source\imported\prepare_data.py` | 19,658 | `3a3de8bd54f675c5abac638be404cd47902951ae505caded34094f733838bd94` |
| `C\source\imported\src.joint_det_dataset.py` | 75,981 | `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d` |
| `C\source\input_manifest.json` | 161,881 | `14d03c98380779f9a632a5cf395bce6bb2f63af8358900e616fabb4081552f89` |
| `C\source\pvground_candidate_box_refiner.py` | 3,767 | `e971346230d0e0547139823c106e48f970c58a0b0ef26ef46b75beaa672f6212` |
| `C\source\pvground_observation_query.py` | 7,683 | `cc92906f9b6f388c8da702f555a099faf07d619fefa2772bfb229693ebae7742` |
| `C\source\pvground_semantic_assignment.py` | 5,065 | `3bf9c5f50095e7e447af91e7f61a34c930bf04ec8590f7e4da54134b859cf773` |
| `C\source\pvground_source_query.py` | 3,025 | `e0fcc6a30402ebeba7a9ffa1be907515350b64e408602a8b47b702e83a5d05ab` |
| `C\source\pvground_tail_preflight.py` | 2,898 | `823f27d14e333a2fb8f15e6bd41a81abe3717c34a4adcb77aebbbb050ebcdbdb` |
| `C\source\pvground_tail_support_box_refiner.py` | 3,655 | `665e94c150492a9fc3d52ddca7da2b7dca7d7ddbd981e759640f15846bb3f757` |
| `C\source\pvground_task_observation_query.py` | 2,770 | `39349bf1fe8f3f51149f0872911020ecd0ede98fc9a8278cecb5ae466ed8643d` |
| `C\source\source_port.json` | 10,485 | `3b2b44241f0c1b52b5e42f6ea84e656abb1dc3f3b062c87d49f0bcbf94b2f7bc` |
| `C\source\split_protocol.json` | 246,139 | `06d0b20a848be97827a4e0257b074afcfb91a04fc8e448f7de34eef453294461` |
| `C\status.json` | 404 | `86824e4c1fb15a24deaa494eeaf757415af6b791f93ef1fa34528ecb1b38c473` |
| `S\collect_complete_arm.py` | 6,134 | `d9ec4305afef6c066be16bc9ce9655356d240935bb33ba092ba04e36088bbe6c` |
| `S\recount_complete_arm.py` | 8,537 | `23724c51a50a46fadcad75fdff662c22572223e2131b731924721329d6410e94` |
| `S\publish_raw_result_and_fused_launch.py` | 12,367 | `f545b377772b21ec46cc029c0af423e68506ad739a4c315bb375fb0c8dc69c4f` |
| `S\tail_fused_training_launch.json` | 987 | `f92de620ee214e024b45d948f328c4d1127801ef3d49c6d6d169f8ad6a23562d` |
| `S\tail_raw_training_boundary_222900.json` | 13,666 | `34ee5ff2709f9ccf3c4a1ecfe83ef941f99bb6084c83ff882d258aad3a9c6089` |
| `H\train.jsonl` | 2,236,096 | `f7170d5f16fb027f061d52b60c585ad7273c3586b8f72b33ac4b7034c31201d4` |
| `H\initial\rows.jsonl` | 5,510,292 | `bcb49aa285cad9ca594f755f151b4ace454ab216e8633a1161a5766aa757e455` |
| `H\terminal\rows.jsonl` | 5,510,801 | `f8eefd766dab7656b219bce36ad2422b6785a2d1fbdb0e578fdce23ee99e6c41` |
| `H\formal\rows.jsonl` | 7,467,535 | `83c2ff101759b5a24fd3b7b792b39ded4ff26b8054565f6bc00085859375aa38` |
| `H\formal\receipt.json` | 685 | `ce178e23eee1d85665bd3c290b9da95b6c061416de2a7c3e6893e0e57da2e859` |
| `G\receipt.json` | 2,535 | `60af173412696ee53ac13e60116460afa6bae5050cd96af84249088dba296374` |
| `G\audit.json` | 2,842 | `eaf159d8092eea1383c69550729c753dfd5e58dea788bf9df346682aed3df704` |
| `A\archive_receipt.json` | 424 | `451a27073a5ee45836867691ab1b2a0a117eb693ae9856919fda6c099b08178a` |
| `A\terminal.pth` | 346,718,993 | `50961f4be97d5e02cef158b32f3c85300308f7ce044b7668def07277b3df85bd` |
| `P\selected_refinement_publication.json` | 694 | `4dcd493c2e9b9906a955a7d7b8ba3d24099d89485d4c1e888e50ac88e87c347b` |
| `P\publication_stdio_read.json` | 264 | `a4a9b05e7fb95e3e912dd1c393af92da9221bfc80db1eb0d153f87ff768cef15` |
| `S\tail_fused_training_boundary_225200.json` | 2,962 | `a23f423e3693b8243d7d471b0449fcd6900cf49e409ab4b2e9f7bb1c94ea856a` |
| `Z\DIFF.json` | 812,918 | `0de306713d2189eb4bac6df78b06142f026b4d1f359883a7772fbf3ceddbfdec` |
| `Z\INTAKE.json` | 1,983 | `e40cb75d141221e2f60961aba92f590dd4158be947e0ff3c4144e8f1c67452f0` |
| `Z\status.json` | 202 | `02f7bf48a7fc82d6466d393f5df903aaaf75e83cac84c8e699cf05aca5108b86` |
| `Z\run.py` | 38,567 | `483a7db4e0d11d8b79af63ea6e02f6d9455bed218190efc17d42e48929648346` |
| `Z\arm\spec.json` | 3,731 | `03c6d64f322210998e2cc35b71f42dd42628465d5d4303588a8e269c90311c76` |
| `Z\arm\load.json` | 348 | `1211d7caa114d4364fa56256979467235a762b510d46b75310ffe5fd678dcbae` |
| `Z\arm\imports.json` | 1,256 | `f9bb4be0be4d61f3ca26c9c0c6b69a25a1d94b970f039815515debfa13edf5fd` |
| `Z\arm\train.log` | 3,191 | `387d217122a1dc86c8cdafe0579ed0685a09d98421e262f48c891c487e704cad` |
| `Z\arm\train.exit` | 2 | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |
| `Z\arm\initial\receipt.json` | 682 | `3e033e23e997464fae9a3d5db6d640ea90f662a24038903fd93a4db76b2838d1` |
| `Z\arm\initial\rows.jsonl` | 8,461,047 | `1749aaf81d15d70867e3fd842804f60f91a4b5826ac779e3c92fe9ec3c8fc5e8` |
| `S\RAW_RESULT_REVIEW_REQUEST.md` | 8,891 | `3be7a5875a9abf9c667c2a0ff6a2137bd6843086327c02b403e72e8e45ae6044` |

