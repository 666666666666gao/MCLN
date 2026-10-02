# Experiment Audit: PV-Ground semantic-only P2 continuation

**Overall verdict: WARN. Integrity status: warn. Acceptance: provisional.**

The saved results and complete training row budgets/orders pass independent deterministic checks. The semantic-only arm fails the registered scientific promotion criteria. No fabricated GT, self-normalized reported metric or mismatched reported performance count was found in the audited evidence. The warnings concern an additional initial Mask-IoU discrepancy, bounded checkpoint evidence, dormant diagnostic code, and the one-seed development scope.

Reviewer: **gpt-6-astra**, reasoning **max**, fresh agent `/root/pvg_semantic_complete_integrity`. Review independence: **same-family**. No cross-family or robustness assurance. Date: **2026-10-02**.

Path aliases used in exact file:line references:

- `S` = `C:/Users/gb/.codex/tmp/pvground_p2_semantic_20261002/complete`
- `P` = `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete`
- `T` = `C:/Users/gb/.codex/tmp`

Only this report and `EXPERIMENT_AUDIT.json` were written. No SSH, model forward, GPU execution, optimizer update, checkpoint-tensor load, deletion or evidence modification was performed. The auditor used independent standard-library arithmetic, not the supplied analyzer functions. All 79 requested inputs were read/hashed. Ten additional local files explicitly listed in the prior supplied INTAKE were used to finish inventory/split/source checks, giving 89 hashed audit inputs.

| Check | Status |
|---|---|
| A. Ground truth provenance and evaluator | PASS |
| B. Score normalization | PASS |
| C. Result existence, exact values and completion | WARN |
| D. Actually invoked metric code and dormant helpers | WARN |
| E. Actual scope and claim ceiling | WARN |
| F. Evaluation type | PASS |

## A–F evidence

### A. Ground truth provenance and evaluator: PASS

The executed ScanRefer loader obtains target IDs from dataset annotations, boxes from annotated ScanNet instances, and masks from their point memberships. GroupFree prediction inputs are separate; butd_gt and butd_cls are disabled. The pinned runtime GroundingEvaluator is explicitly imported, called and its REC totals/Mask sums cross-checked against raw-row accumulation. Local copies match the six recorded import hashes, including the evaluator and adapted loader. This verifies the supplied code/row provenance chain, not independent inspection of remote dataset tensors or a fresh upstream vendor-source comparison.

Evidence: `S/source/imported/src.joint_det_dataset.py:585-648`; `S/source/imported/src.joint_det_dataset.py:1086-1119`; `S/source/imported/src.joint_det_dataset.py:1189-1226`; `S/source/imported/src.joint_det_dataset.py:1325-1392`; `S/run.py:224-265`; `S/run.py:107-117`; `S/run.py:381-432`; `S/semantic/imports.json:3-16`.

### B. Score normalization: PASS

REC is a raw strict IoU hit count and any percentage uses the fixed row count. Mask IoU uses intersection/union and mIoU is sum/N*100. Softmax and feature normalization are scoring operations, not rescaling the reported metric by the model maximum. No model-output max/min/mean is used as a reporting denominator. Candidate coverage is explicitly GT-assisted offline diagnosis.

Evidence: `S/run.py:388-430`; `S/source/imported/evaluator.py:296-303`; `S/source/imported/evaluator.py:485-491`; `S/source/imported/evaluator.py:897-902`; `P/source/analyze.py:25-49`; `T/analyze_pvg_semantic_complete_20261002.py:97-105`.

### C. Result existence, exact values and completion: WARN

All 79 requested inputs exist. All 40 copied new intake artifacts and all 42 prior intake artifacts match declared SHA-256 and size. Nine evaluation files (69,846 records) independently reproduce every SUMMARY metric, paired result, coverage count and three-bin matrix. All training logs and exits are consistent with completion. WARN: the report only names historical initial Mask differences, but the semantic arm adds a difference at row 2058; terminal weights are remote attestations, and historical original-G formal counts are supplied reference constants rather than independently recounted original-G rows in this intake.

Evidence: `S/analysis/REPORT.md:7-58`; `S/analysis/SUMMARY.json:563-860`; `S/semantic/initial/rows.jsonl:368`; `S/semantic/initial/rows.jsonl:3439`; `S/semantic/initial/rows.jsonl:3803`; `P/g_control/initial/rows.jsonl:368`; `P/g_control/initial/rows.jsonl:3439`; `P/g_p2/initial/rows.jsonl:3803`; `S/INTAKE.json:205-215`; `S/semantic/receipt.json:68-71`; `S/status.json:2-20`; `S/preflight/preflight.exit:1`; `S/semantic/train.exit:1`; `S/semantic/formal.exit:1`.

### D. Actually invoked metric code and dormant helpers: WARN

All claimed REC and Mask paths execute: run.evaluate calls GroundingEvaluator.evaluate, which calls both bbox and both mask methods, and the run asserts agreement before writing each receipt. The fresh preflight geometry-gradient branch executes with update=True. WARN is limited to dormant diagnostics: calculate_diou_3d is defined but its loss invocation is commented out; verify_native_replacement is only reached with update=False, whereas the supplied preflight/training calls use True. No reported result is attributed to these helpers; do not claim a new DIoU result or executed replacement-gradient witness.

Evidence: `S/run.py:267-307`; `S/run.py:340-349`; `S/run.py:371-435`; `S/source/imported/evaluator.py:194-206`; `S/source/ported/models/losses.py:103-133`; `S/source/ported/models/losses.py:546-557`; `S/source/pvground_semantic_assignment.py:53-93`; `S/CODE_REVIEW.md:24`.

### E. Actual scope and claim ceiling: WARN

One seed (2027), one newly trained semantic-routing arm and two reused completed controls; each has 29,778 fit IDs once over 3,723 steps. There are 6,887 module-holdout expressions in 106 scenes and 9,508 development-validation expressions in 141 scenes. The retained split protocol records 456 fit scenes and author-pretraining exposure of the module holdout. These are not three seeds or a fresh simultaneous three-arm trial. The report correctly limits generalization, but cannot support robustness, significance, physical-instance identity from IoU bins, a causal gradient-conflict/overfit explanation, or new Nr3D/Sr3D effectiveness.

Evidence: `S/semantic/spec.json:9-14`; `S/PLAN.md:22-34`; `P/source/split_protocol.json:1`; `S/semantic/train.jsonl:1-3723`; `P/g_control/train.jsonl:1-3723`; `P/g_p2/train.jsonl:1-3723`; `S/semantic/formal/rows.jsonl:1-9508`; `S/analysis/REPORT.md:35-59`.

### F. Evaluation type: PASS

Initial, terminal and formal performance evaluations use dataset-provided target-instance GT: real_gt. Candidate-oracle statistics are GT-assisted offline diagnostics over model proposals, not attainable deployment metrics or synthetic references. Preflight is an engineering/gradient witness on a real batch, not an additional benchmark result.

Evidence: `S/source/imported/src.joint_det_dataset.py:1086-1119`; `S/source/imported/evaluator.py:535-553`; `S/source/imported/evaluator.py:879-902`; `S/run.py:392-430`; `S/preflight/preflight.json:1-13`.

## Independent recount

All nine raw evaluation files contain the expected unique IDs: six holdout evaluations of 6,887 rows and three formal evaluations of 9,508 rows, totaling 69,846 records. All 18 mode/stage/arm metric dictionaries, all paired fields, all candidate-coverage arrays and all three-bin matrices exactly match `S/analysis/SUMMARY.json`. Receipt integer fields and floating sums/mIoUs also match. The primary threshold rules are **IoU > 0.25** and **IoU > 0.50**.

| Phase / mode | G control hits @25 / @50 | Joint P2 | Semantic-only P2 | Semantic minus G |
|---|---:|---:|---:|---:|
| initial / bbs | 6176 / 5602 | 6176 / 5602 | 6176 / 5602 | +0 / +0 |
| initial / bbf | 6206 / 5647 | 6206 / 5647 | 6206 / 5647 | +0 / +0 |
| terminal / bbs | 6144 / 5552 | 6153 / 5579 | 6160 / 5571 | +16 / +19 |
| terminal / bbf | 6175 / 5600 | 6162 / 5591 | 6182 / 5606 | +7 / +6 |
| formal / bbs | 5600 / 4452 | 5613 / 4419 | 5588 / 4439 | -12 / -13 |
| formal / bbf | 5628 / 4465 | 5615 / 4401 | 5623 / 4460 | -5 / -5 |

Evidence: `S/semantic/{initial,terminal,formal}/rows.jsonl:1-6887/9508`, each adjacent `receipt.json:3-26`; `P/g_control/{initial,terminal,formal}/rows.jsonl:1-6887/9508`; `P/g_p2/{initial,terminal,formal}/rows.jsonl:1-6887/9508`; reported table `S/analysis/REPORT.md:7-12`. The complete numeric dictionaries and individual artifact hashes are in the JSON companion.

Formal semantic-only `bbs` is **5588/9508 = 58.771560790912915%**, **4439/9508 = 46.68700042069836%**. Its delta is **-12/-13** versus continued G and **-25/+20** versus joint P2. Against the supplied historical reference 5615/4495, the arithmetic difference is **-27/-56**. It is **315 strict hits below 4754**. `preserves_historical_g_both_thresholds`, `target_pass` and `strict_increment_preserving_loose_vs_control` are all correctly false. The historical reference is not a newly recounted original-G formal result in this intake (`S/analysis/SUMMARY.json:843-853`; `T/analyze_pvg_semantic_complete_20261002.py:92-96`).

| Formal comparison | Mode | Threshold | Repairs | Damages | Net |
|---|---|---|---:|---:|---:|
| Semantic vs g_control | bbs | >0.25 | 228 | 240 | -12 |
| Semantic vs g_control | bbs | >0.50 | 346 | 359 | -13 |
| Semantic vs g_control | bbf | >0.25 | 239 | 244 | -5 |
| Semantic vs g_control | bbf | >0.50 | 340 | 345 | -5 |
| Semantic vs joint_p2 | bbs | >0.25 | 232 | 257 | -25 |
| Semantic vs joint_p2 | bbs | >0.50 | 389 | 369 | +20 |
| Semantic vs joint_p2 | bbf | >0.25 | 247 | 239 | +8 |
| Semantic vs joint_p2 | bbf | >0.50 | 407 | 348 | +59 |

Paired identities (`row_id`, `scan_id`, `target_id`), root GT boxes and point SHA strings align exactly for every compared row, including initial-to-terminal within each arm. These are paired expression outcomes, not a fixed-box or fixed-score intervention. Saved selected REC boxes were independently re-evaluated against saved GT boxes in double precision across all 139,692 arm/stage/mode records; maximum absolute IoU difference from the recorded float values was **4.0932876225197035e-6**, with **zero threshold disagreements**. This numerical check is independent of the supplied PyTorch evaluator.

### Candidate coverage

Each entry is @25 / @50 hit count. The four prefixes are top 16, 32, 64 and all 256 candidates under that output mode.

| Formal model / mode | Top 16 | Top 32 | Top 64 | Full 256 |
|---|---:|---:|---:|---:|
| g_control / bbs | 6174 / 5337 | 6619 / 5788 | 7598 / 6711 | 8902 / 7755 |
| g_control / bbf | 6314 / 5467 | 6726 / 5874 | 7622 / 6714 | 8902 / 7755 |
| joint_p2 / bbs | 6188 / 5365 | 6617 / 5816 | 7600 / 6733 | 8896 / 7806 |
| joint_p2 / bbf | 6331 / 5504 | 6747 / 5927 | 7661 / 6784 | 8896 / 7806 |
| semantic_p2 / bbs | 6254 / 5446 | 6735 / 5935 | 7774 / 6879 | 8893 / 7795 |
| semantic_p2 / bbf | 6430 / 5591 | 6860 / 6035 | 7798 / 6890 | 8893 / 7795 |

The full-256 flags agree between `bbs` and `bbf` on every row in all nine files. At @50, semantic-only has 40 more covered rows than G but 13 fewer selected hits; coverable selected errors are 3356 versus 3303. At @25, full coverage is 8893 versus 8902. Coverage is an offline GT oracle, not observed deployable accuracy. Raw files retain flags, not all candidate boxes, so this audit independently verifies flag aggregation and consistency rather than all-256 geometric calculations (`S/run.py:399-416`; `P/source/analyze.py:29-48`).

### Three-bin IoU transitions

Rows are the reference model; columns are semantic-only P2. Bins are strictly **>0.50**, **(0.25, 0.50]**, **<=0.25**. Every matrix totals 9508.

Reference: G control

| Reference bin | >0.50 | (0.25, 0.50] | <=0.25 |
|---|---:|---:|---:|
| >0.50 | 4093 | 206 | 153 |
| (0.25, 0.50] | 222 | 839 | 87 |
| <=0.25 | 124 | 104 | 3680 |

Reference: joint P2

| Reference bin | >0.50 | (0.25, 0.50] | <=0.25 |
|---|---:|---:|---:|
| >0.50 | 4050 | 217 | 152 |
| (0.25, 0.50] | 274 | 815 | 105 |
| <=0.25 | 115 | 117 | 3663 |

The historical G-to-joint matrix also independently matches `S/../previous_pair_iou_transition.json:8-28`: `[[4085,235,132],[209,846,93],[125,113,3670]]`. No matrix establishes a physical instance ID, a localization-only cause, gradient conflict or overfitting. Prediction query indices are slots; the retained GT `target_id` identifies the target, not the physical identity of the selected prediction.

## Initial REC parity versus Mask discrepancies

All 6,887 initial rows match across all three arms in identity, GT, point SHA, `bbs` and `bbf` selected query/box/IoU, and candidate-oracle flags. Initial REC is exactly 6176/5602 (`bbs`) and 6206/5647 (`bbf`) for all three. This does **not** mean whole-output or Mask-IoU equality.

The following values occur identically for `bbs` and `bbf` within each listed row:

| Training row ID | Initial file line | Scan / target | G Mask IoU | Joint P2 Mask IoU | Semantic-only Mask IoU |
|---:|---:|---|---:|---:|---:|
| 2058 | 368 | scene0036_00 / 5 | 0.9480887055397034 | 0.9480887055397034 | 0.9277961254119873 |
| 16804 | 3439 | scene0320_00 / 34 | 0.950276255607605 | 0.8911917209625244 | 0.8911917209625244 |
| 18482 | 3803 | scene0358_00 / 14 | 0.8771331310272217 | 0.8976109027862549 | 0.8771331310272217 |

Each table line points to that exact line in `P/g_control/initial/rows.jsonl`, `P/g_p2/initial/rows.jsonl` and `S/semantic/initial/rows.jsonl`. Historical G-vs-joint differences are **16804 and 18482**. New semantic-vs-G differences are **2058 and 16804**. New semantic-vs-joint differences are **2058 and 18482**. Thus there are two differing rows per pair and **three distinct affected rows** across the complete comparison.

Initial Mask @25/@50 hit counts remain equal: `bbs` 6180/5818 and `bbf` 6208/5833. Initial Mask mIoU (%) differs: G **74.19909289362919 / 74.43983810884491**; joint P2 **74.1985323191717 / 74.43927753438744**; semantic-only **74.19794032916319 / 74.43868554437893**, in `bbs/bbf` order. All these values are correctly recorded in the receipts/SUMMARY; the missing point is the narrative disclosure of newly affected row 2058 (`S/analysis/REPORT.md:56`; `S/analysis/SUMMARY.json:859`).

The causes are unresolved. Do not call these rounding errors or claim that semantic routing caused them. The real preflight verifies only `last_sem_cls_scores`, `last_center`, and `last_pred_size`; all recorded maximum differences are zero, but Mask output is not included in that test (`S/run.py:329-332`; `S/preflight/preflight.json:212-216`).

## Training budget, source bindings and completion

For each of G control, joint P2 and semantic-only, independent parsing of every training log line verifies steps **1–3723** exactly once, constant `total_steps=3723`, **29778 unique fit IDs exactly once**, **3722 batches of 8 plus 1 batch of 2**, and exact matching of the retained split protocol. Every batch row list is identical across all three arms. The first batch is `[14307,26871,13547,1622,9672,18692,29949,23417]`; the final batch is `[4176,17320]`. There is no holdout ID in training. Logged numeric values are finite and logged total loss equals native loss plus CE correction within 2e-6. Initial/terminal GT/point identities remain unchanged.

Evidence: `S/semantic/train.jsonl:1-3723`; `P/g_control/train.jsonl:1-3723`; `P/g_p2/train.jsonl:1-3723`; `P/source/split_protocol.json:1`. All holdout scan IDs satisfy the registered salted physical-scene fold rule. The 106 holdout and 141 formal scene sets are disjoint. The split manifest records 456 fit physical scenes and zero fit/holdout physical overlap; the runner independently asserts that full annotation partition at runtime (`S/run.py:179-188`, `S/run.py:232-247`). The local training logs contain row IDs, not all training scene/augmented tensor contents, so do not enlarge this into a new per-tensor replay claim.

The actual config is seed 2027, batch 8, one fit pass, fixed LR/core/backbone 1e-5, native wd 5e-4 and clip 0.1. `BaseTrainTester.get_optimizer` constructs fresh AdamW at training start. The scientific training process never loads an old optimizer. The preflight is a separate process with two disposable updates of the same eight-row batch, plus in-memory mutable-state/Adam restore. Those updates do not enter the saved training start (`S/run.py:168-176`, `S/run.py:312-367`, `S/run.py:440-480`; `S/source/imported/main_utils.py:339-366`; `S/controller.py:31-45`).

The three run specs agree on original-G SHA, author checkpoint SHA, env spec SHA, seed, batch, LR, fit passes, primary mode, CE threshold, and G reader/assignment source hashes. Every training receipt binds its local run.py, spec and training-log SHA correctly. Six `semantic/imports.json` hashes match the copied modules. Both `source_port.json` hashes match. The imported dataset also matches `P/source/appearance_source_manifest.json:466`; that manifest and the split protocol match the hashes in the new input manifest. The new input manifest is byte-identical to the retained previous one.

Among the nine compared G/P2 model/helper files, only `encoder_decoder_layers.py` differs, and the complete difference is exactly the single intended replacement at line 534. The new decoder SHA is `8cc6731f3086a2f074e6905bb9e24a3c321b4b180f2cd43cf6b642c31e54995b`; runner SHA is `06c1cba7249a00897324092e2d20faffd0597b37b036b7b8f95ad5e8ab5d07ee`. The native losses and the semantic-assignment correction are byte-identical to the controls. All listed Python inputs parse.

Actual new exits are all **0** (`S/preflight/preflight.exit:1`, `S/semantic/train.exit:1`, `S/semantic/formal.exit:1`). The status contains the three completed stages in order and finishes **2026-10-02 13:29:53.808799 +08:00** (`S/status.json:2-20`). The earlier reused pair status is complete with all four ordered stages (`P/pair_status.json:2-28`); its individual exit files are not in the local prior intake. Its receipts and controller completion guards support the reported completion, but this audit does not claim to have independently read those absent exit files.

## Findings and evidence boundaries

### W1. Additional initial Mask-IoU discrepancy must be disclosed (WARN)

Historical G versus joint-P2 differences are row IDs 16804 and 18482. Semantic-only versus G differences are row IDs 2058 and 16804. Semantic-only versus joint-P2 differences are row IDs 2058 and 18482. Every pair has two differing rows in both bbs and bbf; the union is three rows. REC query, selected box, IoU, oracle flags, identity, GT and point SHA are exactly equal at all initial rows. Mask hit counts at 0.25/0.50 remain equal, but Mask sums/mIoUs do not. The source preflight compares semantic scores, centers and sizes only. Cause remains unresolved; do not label it rounding or nondeterminism without evidence.

Evidence: `S/semantic/initial/rows.jsonl:368`; `S/semantic/initial/rows.jsonl:3439`; `S/semantic/initial/rows.jsonl:3803`; `P/g_control/initial/rows.jsonl:368`; `P/g_control/initial/rows.jsonl:3439`; `P/g_control/initial/rows.jsonl:3803`; `P/g_p2/initial/rows.jsonl:368`; `P/g_p2/initial/rows.jsonl:3439`; `P/g_p2/initial/rows.jsonl:3803`; `S/run.py:329-332`; `S/analysis/REPORT.md:56`.

### W2. Checkpoint assurance is bounded to runtime evidence and remote attestation (WARN)

The 344,535,837-byte terminal checkpoint is not in the local intake. Its remote SHA-256 96faa68d0ebc02f2d21018eec3ccbfbb135d0b2a643108265c8440a6c237fecf matches the training receipt. Formal code loads terminal.pth, checks original-G hash/P2 flag/step 3723/exact selected key set, then strict-loads the merged state; successful formal exit/receipt supports execution of that path. load.json is written before terminal loading, so it certifies original-G plus fresh P2 reconstruction, not terminal tensors or Adam state. Formal code does not compare the file hash to receipt.terminal_sha256, check p2_routing/spec/source hashes in the payload, inspect Adam state, or explicitly assert terminal tensor dtype/finiteness. No independent local terminal tensor inspection or optimizer-state audit occurred.

Evidence: `S/run.py:142-162`; `S/run.py:198-207`; `S/run.py:437-439`; `S/INTAKE.json:205-215`; `S/semantic/receipt.json:68-71`; `S/semantic/load.json:1-9`; `S/semantic/formal/receipt.json:22-26`; `S/semantic/formal.exit:1`; `T/collect_pvg_semantic_complete_20261002.py:53-84`.

### W3. Scientific promotion fails on development evidence (WARN)

Semantic-only formal bbs is 5588/4439 of 9508 (58.77156079091291% / 46.68700042069836%). Versus same-budget G this is -12/-13; versus joint-P2 it is -25/+20. Relative to supplied historical-G reference 5615/4495 it is -27/-56, and it falls 315 strict hits short of target 4754. Both stated development gates are false. Positive strict change versus joint-P2 alone does not establish improvement over continued or original G.

Evidence: `S/semantic/formal/receipt.json:13-19`; `P/g_control/formal/receipt.json:13-19`; `P/g_p2/formal/receipt.json:13-19`; `S/analysis/SUMMARY.json:843-853`; `S/PLAN.md:49-53`.

### W4. Raw-row recomputation is narrower than full tensor replay (WARN)

Selected REC IoUs were independently reconstructed from saved predicted/GT boxes in double precision: maximum absolute discrepancy across all arms/stages/modes is 4.0932876225197035e-6, with zero threshold-classification changes. Candidate coverage is recounted from per-row oracle flags; all 256 candidate boxes are not stored. Mask aggregates are recounted from per-row Mask-IoU scalars; point-level predicted masks/GT masks are not stored. All paired identity/root-GT/point-SHA fields align, but underlying remote dataset/point tensors were not independently hashed by this auditor. Training logs prove row order/budget, not a separate per-step bitwise comparison of augmented inputs.

Evidence: `S/run.py:397-417`; `S/semantic/formal/rows.jsonl:1`; `S/run.py:251-265`; `S/run.py:470-481`.

### I1. Semantic routing is verified but does not freeze geometry learning (INFO)

Only decoder line 534 differs from sealed joint-P2 model source: residuals=(residuals[0]+evidence,residuals[1]). G reader returns semantic then geometry; native CE/contrastive outputs consume semantic query, and center/size plus Query Mask consume geometry query. Native matching and loss bytes remain unchanged. P2 reads full available text tokens (max_length=256), six predicted/observed source memories, and detached previous predicted candidate centers/sizes, not GT boxes. The preflight witnesses zero/unused gradients from loss_bbox+loss_giou to P2 on two updates of one batch, and nonzero P2 objective gradients. Shared/upstream parameters and discrete Hungarian selection remain routes through which training can alter geometry.

Evidence: `S/source/ported/models/encoder_decoder_layers.py:526-535`; `S/source/pvground_task_observation_query.py:25-53`; `S/source/ported/models/pv_ground.py:293-295`; `S/source/ported/models/pv_ground.py:480-520`; `S/source/ported/models/modules.py:143-178`; `S/source/pvground_expression_evidence.py:37-62`; `S/source/ported/models/losses.py:298-390`; `S/run.py:288-307`; `S/preflight/preflight.json:1-13`; `S/preflight/preflight.json:122-192`.

### I2. Legacy input-manifest fields are not the current training configuration (INFO)

input_manifest.json retains old batch_size=12, steps_per_arm=2482, core_learning_rate=1e-6, readouts_frozen and MCLN artifact fields. The runner only takes its data/source/split contract for this purpose and explicitly uses semantic/spec.json plus native config for batch8, 3723 updates, LR1e-5, wd5e-4 and clip0.1. The original author checkpoint comes from env_spec and the original G delta from base_terminal; the legacy MCLN artifacts are not loaded in the audited runner. Do not quote the old manifest configuration as this run.

Evidence: `S/source/input_manifest.json:1-53`; `S/run.py:45-79`; `S/run.py:119-170`; `S/semantic/spec.json:9-14`; `S/source/imported/main_utils.py:339-366`.

## Claim impact

| Claim | Audit assessment |
|---|---|
| Recorded same-budget semantic-only formal bbs performance is 5588/4439 and all reported comparisons are numerically reproduced. | **supported** |
| All three arms have the same starting REC rows and exact same fit row order/input count. | **supported**. Identity is for saved REC/input fields and row budget; it is not whole-output bitwise parity, equal parameter count or identical overall development cost. |
| P2 evidence addition is routed only to the semantic residual. | **supported**. Static source plus bounded runtime preflight; shared/upstream geometry training and discrete assignment may still change. |
| Semantic-only P2 improves on same-budget G, preserves original G, or achieves the development target. | **unsupported** |
| Strict-bin losses prove the same physical instance only became slightly less localized, or prove gradient conflict/overfitting. | **unsupported** |
| Independent local terminal tensor/Adam inspection or bitwise initial Mask parity was completed. | **unsupported** |
| Only the two historical Mask discrepancies exist. | **unsupported**. The three-arm initial union contains row IDs 2058, 16804 and 18482. |
| Cross-seed robustness/significance, unseen-holdout generalization, test-set/SOTA performance or new Nr3D/Sr3D effectiveness. | **unsupported** |
| Historical original-G formal counts 5615/4495. | **needs_qualifier**. These are the supplied registered historical reference constants, not newly independently recounted original-G formal rows in this intake. |
| Full-256 candidate coverage and Mask aggregates were independently recounted. | **supported**. From saved oracle flags and Mask-IoU scalars; full candidate tensors and point masks were not replayed. |

## Action items

1. Add a clearly labeled disclosure beside the result, preserving the audited evidence: initial Mask differences by all three row IDs/arms/modes and values; keep REC parity and unchanged Mask hit counts distinct from Mask-IoU parity.
2. Keep this result as a failed promotion under the registered last/bbs development criteria; retain the reused-control, one-seed, pretraining-seen-holdout and no-new-Nr3D/Sr3D qualifiers.
3. Describe checkpoint evidence as copied-file hash verification plus runtime formal strict-load evidence and remote terminal attestation. Do not describe load.json or this audit as an independent local terminal tensor/Adam audit.
4. Do not infer physical instance identity, isolated semantic scoring gains, gradient-conflict causality or overfitting from the endpoint IoU matrices.
5. If a future claim requires stronger terminal provenance, bind the terminal digest/spec/routing to a dedicated load receipt and inspect retained tensor/optimizer metadata without a model forward. This is not a prerequisite for reporting the present qualified negative result.
6. Do not attribute any result to the dormant DIoU or update=False replacement-gradient helper. No retraining, evaluator replay or automatic robustness claim is requested by this audit.

## audited_input_hashes

The JSON companion stores the full absolute-path map under `audited_input_hashes`, plus every independently recomputed metric and diagnostic. The table below uses the exact path aliases defined above. All digests are SHA-256 of local input bytes; the remote-only terminal checkpoint hash is deliberately excluded from this independently hashed input map.

| Input | SHA-256 |
|---|---|
| T/analyze_pvg_semantic_complete_20261002.py | 8136e66a029c9be6af780b34ef514d61ef7b9ffd274f744cb734e6029de5420e |
| T/collect_pvg_semantic_complete_20261002.py | 6f143731cec1621c13a1a070e1cada7679f6872c99062d5cf4413af3dbe240a7 |
| P/g_control/cpu_terminal_audit.json | 32737b8d8ce84fcc9e931db76c4f9b0eab236d4aa0fae24baf685586dda7774f |
| P/g_control/formal/receipt.json | ce178e23eee1d85665bd3c290b9da95b6c061416de2a7c3e6893e0e57da2e859 |
| P/g_control/formal/rows.jsonl | 83c2ff101759b5a24fd3b7b792b39ded4ff26b8054565f6bc00085859375aa38 |
| P/g_control/initial/receipt.json | ae014c5c70be185ab203c050cb2cc9d5ff67c4787a2123ca1031c9f5db947e9a |
| P/g_control/initial/rows.jsonl | bcb49aa285cad9ca594f755f151b4ace454ab216e8633a1161a5766aa757e455 |
| P/g_control/load.json | 9d9b96629db093981eb6c35ead1b8316633495c8e721d98e4d9f01e9711ca7a0 |
| P/g_control/receipt.json | 7b76c10b0dae74bf11e4d04438ebb76bc2bfa748d0990f93f2edf55917995b4a |
| P/g_control/spec.json | 375440282ab23854794b1e59b44daae0bc3e52a0bbe90342d0eb721528de4e12 |
| P/g_control/terminal/receipt.json | e967790186cd82735d420e055e990133dc602040ca195436c8594cbd1b348eec |
| P/g_control/terminal/rows.jsonl | f8eefd766dab7656b219bce36ad2422b6785a2d1fbdb0e578fdce23ee99e6c41 |
| P/g_control/train.jsonl | f7170d5f16fb027f061d52b60c585ad7273c3586b8f72b33ac4b7034c31201d4 |
| P/g_p2/cpu_terminal_audit.json | e252739ad019471863a9250a3b0c96ecf4eb103229aabc54885584d2f4563b73 |
| P/g_p2/formal/receipt.json | b622f90ebec456bc7ecc885b31c2c6a80f6fc8fa06ad6c9fbdc115dbbb3fc9d5 |
| P/g_p2/formal/rows.jsonl | fc22b21b63424503a45fa0daa98e07acdae05ba6f2994366f49757d3efc0c448 |
| P/g_p2/initial/receipt.json | 4c6339f9c8d6c24cf521dff683b71f3ad25bb22def38a61ded9ea16924483ed4 |
| P/g_p2/initial/rows.jsonl | 4d205d9d149b6f9f4c9a63758884b90fa2101b8b5ebbeac5363fccc97a691e0c |
| P/g_p2/load.json | 1d643687199248532073b7aba530329691c7a53e2bcbac62ec2b6d0bf3cea0b5 |
| P/g_p2/receipt.json | 97587391d12dc77e664191ea1a83f978041c7adf4cf3eb194de621712733b82e |
| P/g_p2/spec.json | 4049ce1675f58f173f57b3908e7a48a4e5889f103a90d8083536df6096be88db |
| P/g_p2/terminal/receipt.json | 64cbf83bdebd3b96d64009938eb843b34f5e012cf003e84458f4501572484558 |
| P/g_p2/terminal/rows.jsonl | 7b364a63d4bf6c7e29b426d0d00299124600f91c5f39d4ba855bdebbe4967c69 |
| P/g_p2/train.jsonl | a99c59dfa6628c05b8e95f9e1fbc12d3e9b2935a3c4db8bef0addf0ab8590f58 |
| P/INTAKE.json | 62db82401dc11f112e50581aed065e4e56b7e20fcf2e87de491521cfaee5cd22 |
| P/pair_status.json | a8e49ec9bbf8d658b1788ec45eff137d213cf4f0c62a3198640679877f0be45f |
| P/REPORT.md | 20c4fb99027b738ebd2a93db17467e07d3acfda1772354eda5da8f458da7fa85 |
| P/source/analyze.py | 38a1673ce556ec70c7565fa280edfd0d2cbc998bf10f5029ce9354532e6db89a |
| P/source/appearance_source_manifest.json | 75cd5f87a8e715c15b458ed1964c2e6f1d35046bb6beffca1b3388d5903bbe36 |
| P/source/grounding_evaluator.py | 39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677 |
| P/source/input_manifest.json | 14d03c98380779f9a632a5cf395bce6bb2f63af8358900e616fabb4081552f89 |
| P/source/joint_det_dataset.py | 3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d |
| P/source/pair.py | a7e9c4b08a6198c9d70091d45cd9b9b3ad68dd06c4933e2df4fa81d62a58f20c |
| P/source/ported/models/encoder_decoder_layers.py | 4ba0b8a95114d01987def21a42c846f03b5f32d462f5a7b77f168baddc555874 |
| P/source/ported/models/losses.py | 920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de |
| P/source/ported/models/modules.py | 1609b3608e017669ae0ff63e7618f2078e5f72f20955da02e7ff230481b846fc |
| P/source/ported/models/pv_ground.py | 7fe369d7aaefda3f4e6eb389393799aedbe95ace56dbc107a7d26b0e3a3ea235 |
| P/source/pvground_expression_evidence.py | fe9dbc61f4a12252a1f9da3d00d2dc75f5d97c2f9c52d2c9dd2a3a84dd48d6db |
| P/source/pvground_observation_query.py | cc92906f9b6f388c8da702f555a099faf07d619fefa2772bfb229693ebae7742 |
| P/source/pvground_semantic_assignment.py | 3bf9c5f50095e7e447af91e7f61a34c930bf04ec8590f7e4da54134b859cf773 |
| P/source/pvground_source_query.py | e0fcc6a30402ebeba7a9ffa1be907515350b64e408602a8b47b702e83a5d05ab |
| P/source/pvground_task_observation_query.py | 39349bf1fe8f3f51149f0872911020ecd0ede98fc9a8278cecb5ae466ed8643d |
| P/source/run.py | 9ffb44c9d06677fc4e7a2014211ebf228678940465a03db6fbfc4d03f89bf533 |
| P/source/split_protocol.json | 06d0b20a848be97827a4e0257b074afcfb91a04fc8e448f7de34eef453294461 |
| P/SUMMARY.json | 596de7c6d2506a1e974c3763209cd43eb37a2aee48babb19ca39b5157185ff1f |
| S/analysis/REPORT.md | 3a060d5095a2352b1288b33e33637375b1f19aa7f6a3c4793f738e90de045c14 |
| S/analysis/SUMMARY.json | 809fa8938689ff2c6ab3568660e7bf15935cd1efac14af797deccc1e8eca8d0e |
| S/CODE_REVIEW.json | cb9eab2db6185601a4ef6d95bbb7d9a62cb27737f2850dbb337f1c956c48eb63 |
| S/CODE_REVIEW.md | 0e90693420e156292d62fe4e98e7172e8d13cab86400e00e1b3c47cae19dc87a |
| S/controller.py | 4029ca10e4c5731c26f5858ccba9cd281b32ff14f996556af6c25f8c42520016 |
| S/INTAKE.json | 1edd9fa832b065dec4767e48fde072932fa59ee711422740e27f4d709dc98067 |
| S/PLAN.md | f54d244ae93a8df480517e77c5c8cad2203ad72dae4f33f03bedd17ef18ec5ce |
| S/preflight/preflight.exit | 9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa |
| S/preflight/preflight.json | a6bec0c1796ad0f85c5eea22b45a957acc6f74a5288539bf8a6ac71010aff781 |
| S/preflight/spec.json | 88ba7bb244daba1c8a3a7418fcb7decf1e1c37acf564eea7ff2e0be977bbe08f |
| S/prepare_source.py | 6221f5a2e52cc2c895e73cee990aa839e521f996613c5bd5960d036ef9e7b9d9 |
| S/run.py | 06c1cba7249a00897324092e2d20faffd0597b37b036b7b8f95ad5e8ab5d07ee |
| S/semantic/formal.exit | 9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa |
| S/semantic/formal/receipt.json | 8cd17f8a1136cb9e238c92f8d2557710660b366394e3878dfa3ad582a0c4ae04 |
| S/semantic/formal/rows.jsonl | e777df4bd3f67a79ee23b230b122c698224a58492755515e0c7a562d9282a884 |
| S/semantic/imports.json | 1031d82d109512ebfaaa27c0f42086dba882c820f9e1bb2bc3be24f7531acc64 |
| S/semantic/initial/receipt.json | 240b3c8e416c9a99446483d9c54df7f2378b18bbc7dfba9454c72ee9463d8315 |
| S/semantic/initial/rows.jsonl | 8e16d7a2881da47a11dbff72f17b80580123ce5af3b3ffb5da1f9c7c0157d79c |
| S/semantic/load.json | 907e3de43f564e0e530d7977010e14f9a1733bd77758237b7a3d5f5f07e48e07 |
| S/semantic/receipt.json | 3d13d29ce004472f9d6d142e35feddb8b131f70a9f74bd6da9116d95fba0d038 |
| S/semantic/spec.json | 45f402df1d70eb569cccb0902f0cac0a491548be543811bf37a1de77d4211862 |
| S/semantic/terminal/receipt.json | 3e6fd70436def780f830d1028a2476062e23ce9cfea38061acee0a2692cfac03 |
| S/semantic/terminal/rows.jsonl | c6ea30c18106901e62701f18f6c89fd1317aa6ab8a83e64c368a1639b74dadea |
| S/semantic/train.exit | 9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa |
| S/semantic/train.jsonl | 1ecd0e285473f02179e5bfa141ae40f503079a052aeeb907a061156079b07354 |
| S/source/imported/evaluator.py | 39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677 |
| S/source/imported/main_utils.py | 14bd101eb78ec3d729968aff0a975684efa8388574583ca9e4b0f512877b4f5c |
| S/source/imported/models.losses.py | 920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de |
| S/source/imported/models.pv_ground.py | 7fe369d7aaefda3f4e6eb389393799aedbe95ace56dbc107a7d26b0e3a3ea235 |
| S/source/imported/prepare_data.py | 3a3de8bd54f675c5abac638be404cd47902951ae505caded34094f733838bd94 |
| S/source/imported/src.joint_det_dataset.py | 3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d |
| S/source/input_manifest.json | 14d03c98380779f9a632a5cf395bce6bb2f63af8358900e616fabb4081552f89 |
| S/source/ported/models/encoder_decoder_layers.py | 8cc6731f3086a2f074e6905bb9e24a3c321b4b180f2cd43cf6b642c31e54995b |
| S/source/ported/models/losses.py | 920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de |
| S/source/ported/models/modules.py | 1609b3608e017669ae0ff63e7618f2078e5f72f20955da02e7ff230481b846fc |
| S/source/ported/models/pv_ground.py | 7fe369d7aaefda3f4e6eb389393799aedbe95ace56dbc107a7d26b0e3a3ea235 |
| S/source/pvground_expression_evidence.py | fe9dbc61f4a12252a1f9da3d00d2dc75f5d97c2f9c52d2c9dd2a3a84dd48d6db |
| S/source/pvground_observation_query.py | cc92906f9b6f388c8da702f555a099faf07d619fefa2772bfb229693ebae7742 |
| S/source/pvground_semantic_assignment.py | 3bf9c5f50095e7e447af91e7f61a34c930bf04ec8590f7e4da54134b859cf773 |
| S/source/pvground_source_query.py | e0fcc6a30402ebeba7a9ffa1be907515350b64e408602a8b47b702e83a5d05ab |
| S/source/pvground_task_observation_query.py | 39349bf1fe8f3f51149f0872911020ecd0ede98fc9a8278cecb5ae466ed8643d |
| S/source/source_port.json | b9d71fd2f130579f095b5e693b988fc6f9b7d1de2710464756fa50a8f3d22bd2 |
| S/status.json | 33d553b08cca2038566136debf9a7cc8838e4cf2fdbd21078265a18652743479 |
| T/pvground_p2_semantic_20261002/previous_pair_iou_transition.json | a4c4ac7072bd7c1fd53c5eefdff1d823d13c7eab0438d2a6a1cf01c02af38533 |
