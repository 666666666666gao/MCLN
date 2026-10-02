**Overall verdict: WARN. No blocking finding prevents publication of the bounded P3 result in the revised draft.** The completed experiment supports a negative result: formal native `last_/bbs` accuracy did not improve.

Auditor attribution: fresh Codex reviewer, Astra/max, `review_independence: same-family`, `acceptance_status: provisional`. I inspected artifacts directly and performed independent CPU calculations. I modified no artifacts and launched no GPU work.

The reviewed publication script has SHA256:

`b2cb69aed9cb8e49ca28f9f1501de399d6987e91022caa6b0a6158583cf6b6b6`

For the exact references below:

- `P` = `C:\Users\gb\.codex\tmp\pvground_p3_next_20261002`
- `C` = `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\g_control`
- `H` = `C:\Users\gb\.codex_pvground_cs_20261002\refine-logs\pvground_scanrefer_formal_20260918_semantic_assignment_v1`
- `A` = `C:\Users\gb\.codex\archives\pvg_p3_20261002`

| Check | Status | Finding |
|---|---|---|
| A. Ground-truth provenance | PASS, with evidence limit | Dataset-derived root boxes and instance masks; no model-generated reference substituted |
| B. Prediction-derived normalization | PASS | Standard IoU, threshold counts and sample means |
| C. File existence and numerical agreement | PASS | Completed receipts exist; P3 and continuation-control metrics independently agree |
| D. Executed versus dead evaluation code | PASS for reported metrics | Native REC and Mask evaluation execute; unused diagnostic branches are distinguished below |
| E. Scope and claims | WARN | One seed, one local ScanRefer development-validation protocol; replay and control-equivalence limits remain |
| F. Evaluation classification | PASS | `real_gt`, with two-stage detector inputs; oracle coverage is GT-assisted diagnostic evidence |

**A — Ground-truth provenance**

The loader reads ScanRefer annotations, maps `object_id` to the target, and builds targets from dataset scan objects. GT masks use `scan.three_d_objects[tid]['points']`; GT boxes use `scan.get_object_bbox(tid)`. These become `center_label`, `size_gts` and `gt_masks`. The runner’s saved `root_box` and Mask truth come directly from those batch fields.

Evidence: `P\complete\source\imported\src.joint_det_dataset.py:596`, `:634`, `:1086`, `:1102`, `:1106`, `:1387`; `P\complete\run.py:401`, `:407`, `:427`; `P\complete\source\imported\evaluator.py:535`, `:879`.

The prediction forward receives points, text, superpoints and detector inputs before labels are merged for native loss/evaluation. P3 itself reads raw observations and the geometry query; it receives no target box, GT mask or teacher output. G’s GT-dependent semantic assignment remains a training loss correction.

Evidence: `P\complete\run.py:253`, `:265`, `:275`, `:283`; `P\complete\source\pvground_candidate_box_refiner.py:42`; `P\complete\source\pvground_semantic_assignment.py:9`, `:34`.

**Limit:** I verified the source-to-saved-target path, not an independent reconstruction of every target from original ScanNet annotation/mesh files. Those raw files were not part of this local audit bundle.

**B — Metric normalization**

REC uses axis-aligned intersection-over-union and strict `IoU > 0.25` / `IoU > 0.5` decisions. Accuracies divide hit counts by the evaluated row count. Mask mIoU averages stored per-example intersection/union values. No reported accuracy is divided by the model’s maximum, mean or other prediction statistic.

Evidence: `P\complete\run.py:408`, `:428`, `:441`; `P\complete\source\imported\models.losses.py:70`; `P\complete\source\imported\evaluator.py:897`; `P\recount_complete.py:51`, `:64`.

Semantic softmax and the contrastive temperature are part of native candidate ranking, not performance normalization. The runner’s ranking formula matches the imported evaluator: main + modifier + pronoun + relation − other-entity scores, with GT box filtering disabled.

Evidence: `P\complete\run.py:384`, `:397`, `:420`; `P\complete\source\imported\evaluator.py:222`, `:281`, `:329`, `:384`.

**C — Result files, numerical agreement and identity**

I verified all **34 collected files** against `INTAKE.json`, the imported-source and module hashes, all P3/control row-file receipt hashes, and both training-log hashes. The controller status and final logs substantiate completion.

Evidence: `P\complete\INTAKE.json:5`; `P\complete\status.json:2`; `P\complete\p3\imports.json:10`; `P\complete\p3\train.log:83`, `:98`; `P\complete\p3\formal.log:26`.

Independent calculations produced:

| Evaluated output | Rows | bbs hits >0.25 | bbs hits >0.5 | bbf hits >0.25 | bbf hits >0.5 |
|---|---:|---:|---:|---:|---:|
| P3 initial holdout | 6,887 | 6,176 | 5,602 | 6,206 | 5,647 |
| P3 terminal holdout | 6,887 | 6,172 | 5,614 | 6,184 | 5,643 |
| P3 formal validation | 9,508 | **5,594** | **4,401** | 5,601 | 4,423 |
| Continued-G terminal holdout | 6,887 | 6,144 | 5,552 | 6,175 | 5,600 |
| Continued-G formal validation | 9,508 | **5,600** | **4,452** | 5,628 | 4,465 |

Evidence: respective `P\complete\p3\initial\receipt.json:18`, `terminal\receipt.json:18`, `formal\receipt.json:18`; `C\terminal\receipt.json:18`; `C\formal\receipt.json:18`, plus their complete row files.

P3 formal bbs is **58.8346655% / 46.2873370%**. Against continued G, the paired counts are:

- >0.25: **240 fixes, 246 breaks, net −6**.
- >0.5: **330 fixes, 381 breaks, net −51**.

Every paired P3/control row has matching stored row/scan/target identity, root GT and point hash. Independent control-box recalculation also preserved every threshold decision.

Historical original-G **5615/4495** and author-parent **5579/4381** are present in historical receipts, with author and original-G checkpoint hashes matching P3’s starting identities. All 9,508 historical protocol scene/target identities match P3 formal ordering.

Evidence: `H\receipt.json:19`, `:37`, `:44`, `:61`; `H\fit_terminal\receipt.json:20`; `H\published_parent\receipt.json:20`; `H\protocol.json:14`; `P\complete\p3\spec.json:8`, `:50`.

Thus P3’s historical original-G delta is **−21/−94**, and the **4754/9508** strict target is unmet. Historical raw prediction arrays were not present in that local receipt directory; this audit verifies those historical references against receipts and protocol, not a fresh historical prediction recount.

**Original versus revised numerical checks**

The only recount-code change replaces the two `< 2e-6` maximum-error assertions with **per-row threshold-equivalence assertions** for both selected and coarse boxes, retaining reported maximum discrepancies.

Evidence: `P\recount_complete_strict_2e6_failed.py:67`; `P\recount_complete.py:67`.

The old bound is genuinely violated by the stored data:

- Initial selected/coarse maximum discrepancy: `2.8997857486334766e-6`.
- Formal bbs selected/coarse: `2.8116331026728503e-6` / `2.6527254894936902e-6`.
- Largest P3 discrepancy overall: formal bbf coarse, `2.935893036542676e-6`.

I independently reproduced the diagnostic maxima and confirmed **zero threshold flips across every P3 stage, mode and selected/coarse comparison**. A separate float32 arithmetic reconstruction brought discrepancies down to at most approximately `5.37e-7`, consistent with finite-precision arithmetic. The worst absolute-error example includes a size clamped near `1e-6`.

Evidence: `P\complete\RECOUNT_DIFFERENCE_DIAGNOSIS.json:6`, `:402`, `:549`; `P\complete\p3\initial\rows.jsonl:3243`; `P\complete\p3\formal\rows.jsonl:254`.

**Judgment:** the revision is acceptable for the reported threshold accuracies and repair/damage counts. It does **not** establish bitwise equality, the original absolute tolerance, or exact continuous-IoU replay. No altered count or hidden threshold disagreement was found.

**D — Actual execution and mechanism evidence**

The runner invokes the imported evaluator after the model forward and native loss. The evaluator invokes both REC and both Mask branches. REC counts are checked against native accumulators, and Mask IoU sums against native accumulated sums before receipts are written.

Evidence: `P\complete\run.py:391`, `:396`, `:442`, `:445`; `P\complete\source\imported\evaluator.py:194`; completion evidence at `P\complete\p3\formal.log:26`.

P3 is wired into the final deployed `last_center` and `last_pred_size`. Its module uses seven locations × sixteen input indices, pooled observation features, and one six-dimensional additive residual. A direct comparison with the original-G parent model showed only the refiner initialization and final-box wiring changes; all 98 source-port entries agree with the original-G manifest except that model file.

Evidence: `P\complete\source\imported\models.pv_ground.py:277`, `:513`; `P\complete\source\pvground_candidate_box_refiner.py:19`, `:37`, `:55`, `:68`; `P\complete\prepare_source.py:19`.

For formal bbs, recalculation of the **same selected query’s** coarse/final boxes gives:

| Threshold | Coarse hits | Final hits | Fixes | Breaks |
|---|---:|---:|---:|---:|
| >0.25 | 5,598 | 5,594 | 3 | 7 |
| >0.5 | 4,403 | 4,401 | 15 | 17 |

Evidence: `P\complete\CPU_RECOUNT.json:196`; box capture at `P\complete\run.py:392`, `:429`.

These are internal comparisons inside the jointly trained P3 model. They do not isolate the causal training contribution of adding P3.

A dormant diagnostic remains: `verify_native_replacement` is called only under `if not update`, whereas the P3 preflight and fit call `step(..., True)`. Therefore this completed run does not supply a fresh execution of that particular reconstruction witness. The separate preflight geometry/semantic-gradient assertions do execute.

Evidence: `P\complete\run.py:286`, `:299`, `:349`, `:495`. This does not invalidate a reported metric or require unrelated code cleanup.

**E — Budget, scope and remaining warnings**

I independently checked all **3,723 training records**, not just a prefix:

- 29,778 distinct fit IDs, each seen exactly once.
- Exact batch-by-batch order agreement with continued G.
- 3,722 batches of eight and one batch of two.
- No fit ID in the 6,887-row holdout.
- Finite recorded loss and gradient statistics.

Evidence: complete `P\complete\p3\train.jsonl:1` through final record `:3723`; `C\train.jsonl:1`; `P\complete\source\split_protocol.json:1`; executable checks at `P\complete\run.py:493`.

The protocol is one fit pass, seed 2027, fresh AdamW, LR/backbone LR `1e-5`, weight decay `5e-4`, clipping `0.1`. The two disposable preflight updates are separate from the freshly restored training run.

Evidence: `P\complete\p3\spec.json:9`; `P\complete\run.py:169`, `:345`, `:474`; `P\complete\controller.py:48`.

The recorded scene scope is 456 fit physical spaces and 106 holdout spaces; I independently counted 106 holdout and 141 formal-validation scenes. Holdout/formal scenes are disjoint. Physical fit/holdout separation is also asserted in the executed loader and recorded in its log. The split explicitly says previous pretraining has seen the development holdout.

Evidence: `P\complete\source\split_protocol.json:1`; `P\complete\run.py:234`; `P\complete\p3\train.log:8`.

The material nonblocking limitations are:

1. **E0 equality is bounded.** All 6,887 stored point hashes, identities, GT boxes and selected bbs/bbf query/box/IoU records match control. This is not equality of all rankings, logits, detector/text inputs or masks. Detector/text fixtures cover four rows.  
   Evidence: `P\complete\run.py:237`, `:461`; revised `P\publish_complete.py:68`.

2. **E0 Mask differences remain unexplained.** Rows 2058 and 18482 differ in both modes. Their P3/control IoUs are respectively `0.9277961254 / 0.9480887055` and `0.8976109028 / 0.8771331310`. Cancellation in aggregate does not establish equivalence.  
   Evidence: `P\complete\p3\initial\rows.jsonl:368`, `:3803`; corresponding `C\initial\rows.jsonl` lines; `P\complete\p3\initial_control_comparison.json:2`.

3. **Mask recount is not raw Mask replay.** Formal bbs stored-IoU recount gives 5786/5113 hits and mIoU **46.8973071%**. The point masks/logits needed for independent raw replay are absent from row records.  
   Evidence: `P\complete\run.py:426`, `:429`; `P\complete\p3\formal\receipt.json:14`.

4. **Full256 coverage remains saved execution evidence.** Formal coverage flags total 8946/7848, with bbs top16 totals 6228/5456. All-256 boxes and scores were not saved; I checked flag consistency and totals, not independent all-candidate geometry.  
   Evidence: `P\complete\run.py:431`; `P\complete\CPU_RECOUNT.json:210`.

5. **The archive is a delta checkpoint.** I independently verified its 346,715,793-byte file and SHA `5dce3f…8395`. Binary inspection found 1,082 model-delta tensors, all ten P3 tensors, 367,238 P3 parameters, Adam state for 806 tensors—all at step 3723—RNG states and the exact fit-row order. Frozen text-encoder weights are absent. This proves archive content, not a fresh full-model/optimizer restore. The runner’s restoration path depends on retained parents.  
   Evidence: `A\archive_receipt.json:3`; checkpoint construction at `P\complete\run.py:478`; restoration at `:121`, `:145`, `:198`; revised `P\publish_complete.py:70`.

6. **The evidence is one seeded local development experiment.** It supports the recorded negative outcome, not robustness, statistical significance, unseen-test generalization or Nr3D/Sr3D improvement.

**F — Evaluation classification and publication judgment**

REC and Mask are **`real_gt`** evaluations. The model uses **two-stage inputs**: offline GroupFree boxes/classes are loaded and passed into inference, with `butd=True`, `butd_gt=False`, `butd_cls=False`. Those detector predictions are inputs, not evaluation truth.

Evidence: `P\complete\source\imported\src.joint_det_dataset.py:1204`, `:1361`; `P\complete\run.py:222`, `:260`; `P\complete\source\imported\models.pv_ground.py:378`.

Full256/top-k oracle coverage is a **GT-assisted diagnostic upper bound**, not deployable REC accuracy. Preflight equality, gradients and serialization are engineering evidence, not accuracy results.

The revised draft’s P3 claims are supported or appropriately qualified:

- **Supported:** completion, full fixed budget, matched control order, 5594/4401 formal bbs, negative deltas, unmet target.
- **Supported with stated limits:** same-query refinement counts, stored Mask statistics, executed coverage flags, bounded E0 comparison and delta-checkpoint archive.
- **Unsupported extensions:** P3 improved strict localization, full E0 functional equivalence, independently replayed raw Masks/all256 coverage, standalone full-model archive, V99 capability internalization, or cross-dataset gains.

The revised wording at `P\publish_complete.py:46`, `:58`, `:64`, `:66`, `:68`, `:70` retains the necessary qualifications. **Blocking findings: none for publishing that bounded P3 result.** Future-arm statements and separate old-file archival/deletion operations at `:74` onward are outside this completed-P3 audit and receive no certification from it.
