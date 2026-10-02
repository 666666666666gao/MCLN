# PV-Ground P2 fixed-box experiment integrity audit

Date: 2026-10-02. Reviewer: fresh Codex agent `/root/pvg_fixed_box_complete_integrity`, invoked as `gpt-6-astra` with `max` reasoning and `fork_turns=none` (invocation confirmed by the delegating parent).
Review independence: **same-family**. Acceptance status: **provisional**.

## Overall verdict: PASS for the bounded diagnostic

No concrete integrity blocker or required source correction was found. The independently verified result is **actual 5613/4419 versus semantic bypass 5612/4416 out of 9508**, at strict IoU >0.25/>0.50. That is **+1/+3 observed hits**, with **99 changed selected queries**. It is a small direct final-semantic forward effect at one already-trained P2 state. This PASS does not promote P2, establish a trained no-P2 ablation, or explain a training/gradient mechanism.

The reviewer read local artifacts and used independent standard-library arithmetic. No model code was imported, no checkpoint loaded, no neural/evaluator run or training performed, and no network/SSH used. Only these two audit reports were written.

Exact references below use **B = C:/Users/gb/.codex/tmp/pvg_p2_fixed_boxes_20261002/complete** for unqualified paths, **H = C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete**, and **A = C:/Users/gb/.codex/tmp/analyze_pvg_fixed_box_complete_20261002.py**. A compressed JSONL reference such as `full/rows.jsonl.gz:1476` means its decompressed line 1476, containing zero-based row_id 1475.

## A. Ground-truth provenance: PASS

The ScanRefer loader reads `ScanRefer_filtered_<split>.json` and uses its scene/object identifiers (`H/source/joint_det_dataset.py:594-597`, `:634-648`). Its target box comes from `scan.get_object_bbox(tid)`; the associated mask marks the annotated instance's points (`:1086-1119`). Dataset return fields are `center_label`, `size_gts` and `gt_masks` (`:1388-1392`). The native evaluator concatenates those GT center/size fields and restricts them to the root object (`H/source/grounding_evaluator.py:535-553`).

The current runner constructs model inputs explicitly from point/voxel tensors, text, predicted detector boxes/classes/validity and superpoints (`run.py:261-271`). It calls the model at `:312` before appending the GT-bearing batch at `:320-321`; root GT for scoring is read at `:336`. P2 receives text, observed features and previous-layer predicted candidate centers/sizes, rather than target boxes (`H/source/pvground_expression_evidence.py:37-62`; `H/source/ported/models/pv_ground.py:460-461`, `:480-495`; `H/source/ported/models/encoder_decoder_layers.py:526-535`).

Independent comparison of every retained row found exact equality with the historical formal record for row ID, scan ID, target ID, point SHA-256 and all six root-GT coordinates. Historical predictions are replay references, not evaluation targets (`run.py:340-348`; `full/rows.jsonl.gz:1-9508`; `H/g_p2/formal/rows.jsonl:1-9508`).

This is a source-and-record provenance PASS. Original dataset JSON/scene/instance binaries were outside the assigned set; their bytes were not independently reopened or reconstructed.

## B. Score normalization and actual native bbs: PASS

The native label **bbs** means the evaluator's *position/token alignment* path. It softmaxes `last_sem_cls_scores`, sums probabilities under the binarized main-object text map, adds modifier/pronoun/relation map terms and subtracts the other-entity map term. It ranks candidates by that score (`H/source/grounding_evaluator.py:217-286`, `:535-553`). This is reproduced at `run.py:353-363`. **bbf** is a separate contrastive query/token dot-product path with temperature 0.07 (`H/source/grounding_evaluator.py:325-332`); the current intervention does not recompute or claim a bypass bbf result.

The metric counts selected-box IoU **strictly greater than** 0.25 or 0.50. The percentage denominator is the actual 9508 rows, not a prediction maximum/minimum/mean (`run.py:379-390`; `H/source/grounding_evaluator.py:289-303`; `REPORT.md:8-15`; `A:99-100`). Softmax is part of the native ranking score, not a self-normalized accuracy. Saved composite scores span -0.44726741313934326 to 1.8659247159957886, consistent with signed sums of map-weighted probabilities rather than a claimed probability/accuracy.

IoU uses prediction/GT intersection over union (`run.py:345-348`; `H/source/ported/models/losses.py:35-75`). Full256 coverage asks whether any saved candidate has GT IoU above the threshold; this is explicitly an offline oracle diagnostic. It is not actual top-1 accuracy.

## C. Result existence, independent arithmetic and execution: PASS

All **33 assigned artifacts** exist and have fresh SHA-256 records in the JSON audit. All **13 assigned artifacts also listed in INTAKE** match its exact bytes/hashes. The intake's embedded receipt/status equal the separate files; both receipt row hashes, current runner/spec hashes, controller receipt hashes, prior formal-row hash and split/task-module bindings pass (`INTAKE.json:4-109`, `:112-173`; `full/receipt.json:23-27`; `pilot/receipt.json:23-27`; `status.json:7-15`; `H/g_p2/formal/receipt.json:23`).

The complete gzip files were decompressed and every row parsed independently, without executing A. Full row IDs are exactly 0–9507; all 2,434,048 candidate boxes have six finite coordinates and positive sizes, with finite valid IoUs. Both score arrays contain 256 finite values, each selected query maximizes its own score, and selected IoU equals the saved candidate IoU. Every top-1 maximum is unique.

| Semantic path | Hits >.25 | Hits >.50 | Accuracy >.25 | Accuracy >.50 |
|---|---:|---:|---:|---:|
| P2 actual | 5613 | 4419 | 59.0345% | 46.4767% |
| P2 semantic bypass | 5612 | 4416 | 59.0240% | 46.4451% |

The observed deltas are +0.01051746/+0.03155238 percentage points. The counts agree with `full/receipt.json:9-17`, `SUMMARY.json:5-37` and `REPORT.md:10-15`.

| Strict IoU threshold | Repairs (actual hit, bypass miss) | Damages | Net | Repair row IDs | Damage row IDs |
|---|---:|---:|---:|---|---|
| >.25 | 2 | 1 | +1 | 2434, 8013 | 8055 |
| >.50 | 4 | 1 | +3 | 1475, 3138, 4178, 7438 | 8537 |

These eight transition rows were checked directly: `full/rows.jsonl.gz:1476`, `:2435`, `:3139`, `:4179`, `:7439`, `:8014`, `:8056`, `:8538`. They match `full/receipt.json:30-40` and `SUMMARY.json:39-51`.

Ranking and coverage were independently recomputed from the saved score arrays and reconstructed IoUs, rather than merely summing saved oracle bits:

| Path / threshold | Top16 | Top32 | Top64 | All256 |
|---|---:|---:|---:|---:|
| Actual >.25 | 6188 | 6617 | 7600 | 8896 |
| Bypass >.25 | 6191 | 6612 | 7600 | 8896 |
| Actual >.50 | 5365 | 5816 | 6733 | 7806 |
| Bypass >.50 | 5367 | 5813 | 6735 | 7806 |

Every row's oracle bits agree, not just the totals (`SUMMARY.json:9-35`). The existing score ties crossing Top16/32 at row 2467 and Top16 at row 2538 cannot change coverage at either threshold; all possible choices from those boundary tie groups give the same coverage (`full/rows.jsonl.gz:2468`, `:2539`). Full256 equality follows the fixed geometry.

**Historical replay:** all 9508 current actual selections equal historical bbs selections. All 9508 historical bbs boxes and all 9508 historical bbf boxes equal the corresponding current candidate coordinates exactly. Their saved scalar IoUs are also exactly equal; the maximum historical selected-IoU difference is **0**. All actual oracle arrays equal the historical bbs arrays. All stored input/GT identities match. The combined point-identity sequence hash is `2e936597ac382e3345e55868e43f11732ac0672a67dd3fe37f59ffff27db6342` (`SUMMARY.json:52`; `full/rows.jsonl.gz:1-9508`; `H/g_p2/formal/rows.jsonl:1-9508`). Historical rows do not contain all 256 boxes, so this is not a claim of complete historical candidate-array equality.

**Precision check:** independently reconstructed all **2,434,048** full candidate IoUs in float64. The largest absolute difference from saved float32 IoU is **5.484273699285502e-6**, at row 9477/query 10: 0.5965253108236518 versus 0.5965307950973511 (`full/rows.jsonl.gz:9478`). There are **zero >.25 and zero >.50 classification differences across all candidates**, including the closest stored values 0.2499990165233612 and 0.5000004768371582 (`:9102`, `:9128`). A separate standard-library reconstruction rounding each operation to float32 follows the native corner-volume formula for **19,016 selected actual/bypass pairs**. Its largest difference is 2.205371856689453e-6, with **zero selected threshold flips** and the same four hit counts. Thus numerical representation/formula differences do not alter this result; the native/manual formulas are not asserted to be bitwise identical.

**Pilot:** independently read all eight rows and 2048 candidates. Both paths have 5/5 hits, zero selection changes and zero repairs/damages. The eight complete parsed records equal the first eight full records exactly. Pilot full256 coverage is 8/7; it covers one scene and six scan-target pairs (`pilot/rows.jsonl.gz:1-8`; `pilot/receipt.json:3-40`). It is an engineering overlap, not an independent repeat.

**Completion, gate order and time:** the receipt is saved with `status=replay_mismatch` if historical mismatch count is nonzero; only after the assertion passes is `FIXED_BOX_COMPLETE` emitted (`run.py:391-406`). The controller checks child exit code, receipt status and row count before recording completion (`controller.py:42-54`), and requires the pilot's complete/replay status before full (`:26-34`). Here `full.exit:1` is **0**; `full.log:26` and `:27` each contain the exact final receipt, and `status.json:2` is complete. Its 18 progress records increase from 512 to 9216 rows (`full.log:8-25`).

The full child-process wall time is **1024.590033 seconds** (about 17 min 5 sec); the receipt's **897.713277 seconds** measures the diagnostic loop and final bookkeeping, excluding model/data initialization. Pilot child/loop times are **131.938504/4.600161 seconds**. The controller spans **1156.530151 seconds** from 09:15:23.890754 to 09:34:40.420905 CST. These are distinct clocks, not conflicting runtime claims (`controller.py:37-51`; `run.py:307`, `:401`; `status.json:5-20`; `full/receipt.json:5`; `pilot/receipt.json:5`).

## D. Executed metric paths and fixed predictions: PASS

The runner invokes one full model forward per batch, then computes two final semantic heads (`run.py:308-318`). The source-reader hook captures the original D/G residual tuple before P2's out-of-place addition. The wrapper computes the actual and bypass tails, keeps only the bypass semantic value, and returns the actual semantic/geometry pair to the model (`run.py:290-303`; `H/source/ported/models/encoder_decoder_layers.py:526-535`). The task tail processes semantic and geometry values separately using shared parameters; in eval mode its dropout contributes no random mask (`H/source/pvground_task_observation_query.py:33-53`; `run.py:280`).

The counterfactual is a shallow copy with **only** `last_sem_cls_scores` replaced. Centers, clamped sizes and both mask structures are explicitly the same objects; projections, adaptive weights and superpoints also remain shared. Candidate geometry is checked unchanged around both evaluator calls (`run.py:324-335`). The unused bypass geometry never reaches prediction output. The normal model sends actual geometry into box and mask paths (`H/source/ported/models/pv_ground.py:496-517`, `:519-552`).

Both native evaluator calls are live (`run.py:332-333`); the dispatcher executes bbs, bbf and both mask paths (`H/source/grounding_evaluator.py:194-206`). Native bbs hit counts must equal the manual saved-IoU counts before a receipt is written (`run.py:379-385`). All reported transitions and coverages are produced inside the live row/receipt path. New bbf/mask values are not reported by this diagnostic. Unused legacy softmax, print/synchronization/visualization helpers, and inherited unreachable fit/CPU branches produce no claimed metric.

No optimizer construction/update, backward call or checkpoint save exists in the diagnostic runner. An AST call scan finds one model call site and zero optimizer/backward/checkpoint-save calls. Evaluation is under `model.eval()` and `torch.no_grad()`; named buffers and the terminal file hash must remain unchanged (`run.py:277-280`, `:304-308`, `:375-378`). This supports **zero new optimizer updates**, with the execution-evidence limits below. `full/load.json:3-5` records no fresh optimizer, 1072 G delta states and 70 P2 states; it is written before the later strict P2 terminal restoration (`run.py:167-169`, `:204-213`), so it is not a standalone certificate of that final restore.

## E. Evaluation scope and object-input protocol: PASS, with a narrow claim ceiling

Independently counted full scope is **9508 expressions, 141 scan/physical-scene IDs and 2068 scan-target pairs**. It uses one P2 step3723 checkpoint, seed2027 and batch8 on ScanRefer **development validation** (`run.py:86`, `:204-235`, `:257-259`, `:277-284`; `H/g_p2/spec.json:9-14`; `REPORT.md:3-6`). The source enforces val row count/order; this is not a two-scene pilot promoted to a full result.

Inputs are **GroupFree detector-assisted / two-stage**, with `butd=True, butd_cls=False, butd_gt=False` (`run.py:230-233`). Predicted detector arrays come from `group_free_pred_bboxes_<split>/<scan>.npy` (`H/source/joint_det_dataset.py:1188-1226`); GT-object proposal replacement branches are disabled (`:1361-1372`). The 256 evaluated output boxes are the model's actual predictions conditioned on those detector inputs.

The report explicitly states no independent repeat and no trained no-P2 comparison, and rejects a gradient-conflict explanation (`REPORT.md:13-27`). Those qualifications are necessary and accurately match the evidence. Features, D/G reader, head and geometry have already been trained with P2; the intervention only estimates the direct semantic contribution under that fixed state. These limitations are not new correctness findings.

The separate matched-continuation comparison remains control **5600/4452** versus P2 **5613/4419**, or **+13/-33** (`H/g_control/formal/receipt.json:13-19`; `H/g_p2/formal/receipt.json:13-19`). Historical G **5615/4495** and strict target **4754** are inherited references in `H/SUMMARY.json:418-427` and `H/EXPERIMENT_AUDIT.md:81`; the historical G raw run was not newly recounted. Current 4419 remains 335 hits below 4754, and the report correctly does not promote P2 (`REPORT.md:29-34`). The earlier paired-run audit's warnings, including its initial-mask discrepancy, are not erased by this narrower fixed-state audit.

## F. Evaluation type: PASS — real_gt

Both current semantic paths use dataset-provided root GT and are classified **real_gt**. Candidate Top-K/full256 coverage is a **GT-assisted offline oracle diagnostic**. Historical model predictions are used only to verify replay; they do not become reference labels for accuracy (`run.py:336-365`, `:391-402`). The CPU audit and eight-row pilot are engineering/integrity checks; they are not extra benchmark configurations. No synthetic_proxy, self_supervised_proxy, simulation_only or human_eval claim is present.

## Practical evidence limits and action items

No required correction or rerun is identified for the reported bounded arithmetic. Preserve the explicit checkpoint/seed/development-set/two-stage qualifications in any later use.

- This is same-family/provisional review, not cross-family acceptance.
- Original dataset, detector and superpoint bytes were not independently reopened. Current rows bind point hashes and root GT; they do not independently bind every text/detector/superpoint tensor to the old run.
- Historical records lack full candidate arrays, logits/text and masks. Current arrays allow all256 coverage reconstruction, and historical selected bbs/bbf box/IoU comparisons pass, but full historical tensor identity is unavailable.
- Current rows do not retain mask arrays. Fixed masks are supported by the reviewed shared-object execution path and successful completion, not a new independent mask-byte or mask-IoU reconstruction.
- Checkpoint binaries were not loaded or hashed by this auditor; remote processes and runtime imports were not inspected. Buffer equality, checkpoint immutability, strict restore, and actual-logit replay are supported by source gates plus retained execution records. A separate native-accumulator/import dump was outside the supplied set.
- The assigned intake subset excludes `launch.json`, `controller.log`, `pilot.log`, `pilot.exit` and `pilot/load.json`; only the 13 listed assigned entries were independently byte-verified against INTAKE.
- Robustness across seeds/benchmarks, total P2 training effects, and the cause of historical-G regression remain unestablished. No claim of gradient conflict follows from these overall metrics.

## Artifact hashes and trace

`EXPERIMENT_AUDIT.json` contains all 33 freshly calculated input hashes/byte sizes, the exact 99 changed row IDs, all repairs/damages, complete independent counts, precision checks, tie checks, runtime checks and evidence references.

Key hashes:

- Executed diagnostic runner: `8e3e598f8cb8444e4c553e2f0de9bea0b129c59d25eef6df2be52521a3531bb7`.
- Full compressed rows: `5828e307aef52f1026e0d48dfdd511ff7e0cb351ca2645123255941c88bf7e80`.
- Historical P2 formal rows: `fc22b21b63424503a45fa0daa98e07acdae05ba6f2994366f49757d3efc0c448`.
- CPU analysis source: `988805d54825975462f3289d3cb83a38fb61c79292156f1442861cf238973957`.

The delegating parent retains the review request/response transcript. This reviewer was explicitly restricted to these two report files and did not create another trace directory.
