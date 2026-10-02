# PV-Ground G / G+P2 experiment integrity audit

Date: 2026-10-02. Auditor: gpt-6-astra; reasoning effort: max; agent: /root/pvg_g_p2_complete_integrity.
Review independence: same-family. Acceptance status: provisional. No cross-family assurance is claimed.

## Overall verdict: WARN

The recorded paired experiment is complete and its reported arithmetic is correct. The formal primary `bbs` result is G **5600 / 4452** versus G+P2 **5613 / 4419** out of 9508: **+13 / -33 hits**, or **+0.136727 / -0.347076 percentage points** at strict IoU >0.25 / >0.50. The registered development target and the same-budget strict-improvement condition both fail.

No confirmed fabricated GT, prediction-normalized metric, missing current result, numerical reporting mismatch, or incomplete fit-row consumption was found. WARN concerns the research claim ceiling, bounded provenance evidence, and a real two-row initial Mask IoU discrepancy. An unsuccessful experiment is not itself an integrity failure.

The auditor used only local reads and standard-library arithmetic. No experiment module was imported, no model/checkpoint was loaded, no model forward, training, evaluation replay, CUDA operation, or remote command was run. Only this report and EXPERIMENT_AUDIT.json were written.

References below are exact file:line locations. Unqualified paths are relative to **B = C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete**. Additional aliases: **P = C:/Users/gb/.codex_mcln_g0_20260905/docs/PVG_G_P2_PLAN_2026-10-02.md**; **U = C:/Users/gb/.codex/tmp/pvg_training_interface_source_20260908/main_utils.py**; **V = C:/Users/gb/.codex_mcln_g0_20260905/src/visual_data_handlers.py**; **H = C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_scanrefer_formal_20260918_semantic_assignment_v1**. `../` refers to B's parent directory.

## A. Ground-truth provenance: PASS

ScanRefer targets originate in `ScanRefer_filtered_<split>.json`: `source/joint_det_dataset.py:594`, `:596`, `:636`, `:637`. Ground-truth boxes use the annotated target's object points, and masks mark the same annotated instance: `source/joint_det_dataset.py:1086`, `:1100`, `:1103`, `:1105`, `:1389`, `:1392`. The matching source copy V has the exact SHA-256 recorded in `source/appearance_source_manifest.json:470`; V:131 and V:143 read ScanNet segmentation/aggregation JSON, V:152 constructs the annotated instances, and V:225/V:246 derive axis-aligned boxes from their points. This is dataset GT, not a model's prediction used as its own target.

The scorer takes root GT from batch `center_label`/`size_gts` (`source/run.py:383`) and GT masks from `gt_masks` (`:403`). The native evaluator follows the same sources (`source/grounding_evaluator.py:535`, `:544`, `:879`, `:888`).

Inference input construction is explicit: voxelized 50,000-point XYZ/RGB, text, detected boxes/classes/validity and superpoints (`source/run.py:253`–`:263`). The model executes before the GT batch is appended for loss/scoring (`:265`–`:269`, `:372`–`:378`). The loss source appends loss outputs; it does not replace the evaluated predicted boxes/masks with GT (`source/ported/models/losses.py:849`–`:973`).

The object-input protocol must be called **GroupFree detector-assisted / two-stage ScanRefer**, with `butd=True, butd_cls=False, butd_gt=False` (`source/run.py:222`–`:225`, `:230`–`:232`). Detection inputs are loaded from `group_free_pred_bboxes_<split>/<scan>.npy` (`source/joint_det_dataset.py:1188`–`:1226`). The GT-object-input replacement branches at `:1361` and `:1367` are disabled. A point/text-only, single-stage input claim is not supported.

P2 receives previous-layer predicted centers/sizes, detached by the native decoder (`source/ported/models/pv_ground.py:460`, `:493`, `:514`), and reads text/content/observed support (`source/pvground_expression_evidence.py:37`–`:62`; `source/ported/models/encoder_decoder_layers.py:529`–`:535`). It receives no target GT geometry. G's GT-qualified CE correction is training-only (`source/run.py:283`; `source/pvground_semantic_assignment.py:9`–`:47`). Text scoring maps derive from caption parsing with first-object selection, not predicted reference labels (`source/joint_det_dataset.py:981`–`:1082`, `:1784`, `:1875`). GT-based candidate coverage is used only after the model's ranking for offline diagnosis (`source/run.py:400`–`:407`).

Scope of this PASS: source-level provenance and the supplied records. Original annotation/scene arrays were not available in this audit, so their individual dataset bytes and per-row GT masks were not independently reconstructed.

## B. Score normalization: PASS

REC is the raw count of rows satisfying **IoU >0.25 / >0.50**. Mask hits use the same strict thresholds; Mask mIoU is the sum of per-row mask IoUs divided by the actual row count, multiplied by 100 (`source/run.py:390`–`:421`; `source/analyze.py:35`–`:46`). IoU uses geometric intersection over prediction/GT union (`source/ported/models/losses.py:70`–`:75`; `source/grounding_evaluator.py:897`–`:902`). No metric denominator is a model's own maximum, minimum, mean score, or best result.

Softmax and L2 normalization are internal ranking/attention operations, not reported-performance rescaling. The large full-256 oracle coverages are explicitly GT-assisted diagnostic upper bounds, not deployed accuracy.

Independently reconstructed 93,128 selected-box IoUs from saved boxes/root boxes using ordinary floating-point arithmetic. The maximum absolute difference from the stored float32 IoU was 0.000004093288; there were **zero threshold-decision mismatches**. All stored REC/Mask IoUs were finite and within [0,1].

## C. Result existence, completion and arithmetic: PASS

All **42 INTAKE-listed files, 42,246,644 bytes**, matched their listed sizes and SHA-256 values. INTAKE itself, the three requested documents and 14 supplemental original local artifacts were also hashed: **60 audited input hashes** are recorded in EXPERIMENT_AUDIT.json. Supplied primary inputs were not changed.

The completed controller lists G train → P2 train → G formal → P2 formal and finishes at 2026-10-02T08:04:50.418657+08:00 (`pair_status.json:2`–`:28`). This matches the unconditional sequence and successful child-exit gate in `source/pair.py:33`–`:60`; there is no holdout-based filtering of which arm receives formal evaluation.

Read and recounted all **46,564 raw evaluation rows**: four 6887-row initial/terminal files and two 9508-row formal files. Every row ID is unique within its file and follows the appropriate partition order. Every phase receipt's row hash, counts, mask sum and mIoU, every SUMMARY metric and paired repair/damage/coverage value, and every training receipt's initial-to-terminal REC transition agrees with the independent recount. See each `{arm}/{phase}/receipt.json:5`–`:25`, `SUMMARY.json:6`, `:143`, `:280`, and raw `rows.jsonl` lines 1 through 6887 or 9508.

| Set / mode | Rows | G REC >.25 / >.50 | G+P2 REC >.25 / >.50 | Delta hits |
|---|---:|---:|---:|---:|
| initial / bbs | 6887 | 6176 / 5602 | 6176 / 5602 | 0 / 0 |
| initial / bbf | 6887 | 6206 / 5647 | 6206 / 5647 | 0 / 0 |
| terminal / bbs | 6887 | 6144 / 5552 | 6153 / 5579 | 9 / 27 |
| terminal / bbf | 6887 | 6175 / 5600 | 6162 / 5591 | -13 / -9 |
| formal / bbs | 9508 | 5600 / 4452 | 5613 / 4419 | 13 / -33 |
| formal / bbf | 9508 | 5628 / 4465 | 5615 / 4401 | -13 / -64 |

| Set / mode | G Mask hits >.25 / >.50 | G+P2 Mask hits >.25 / >.50 | G mIoU (%) | G+P2 mIoU (%) |
|---|---:|---:|---:|---:|
| initial / bbs | 6180 / 5818 | 6180 / 5818 | 74.199092894 | 74.198532319 |
| initial / bbf | 6208 / 5833 | 6208 / 5833 | 74.439838109 | 74.439277534 |
| terminal / bbs | 6160 / 5782 | 6172 / 5793 | 73.815412229 | 73.978294426 |
| terminal / bbf | 6187 / 5810 | 6181 / 5799 | 74.145932593 | 74.043119512 |
| formal / bbs | 5825 / 5136 | 5817 / 5119 | 47.086036077 | 47.077437489 |
| formal / bbf | 5837 / 5169 | 5802 / 5112 | 47.251785728 | 46.948502157 |

Formal paired transitions are independently recounted from `g_control/formal/rows.jsonl:1` through `:9508` and `g_p2/formal/rows.jsonl:1` through `:9508`; `source/analyze.py:53`–`:74` defines the reported REC comparison.

| Formal metric / mode | >.25 repairs / damages / net | >.50 repairs / damages / net |
|---|---:|---:|
| REC / bbs | 238 / 225 / 13 | 334 / 367 / -33 |
| REC / bbf | 223 / 236 / -13 | 312 / 376 / -64 |
| Mask / bbs | 178 / 186 / -8 | 203 / 220 / -17 |
| Mask / bbf | 156 / 191 / -35 | 183 / 240 / -57 |

At formal bbs >.25, control/P2 full-256 coverage is 8902/8896, and their coverable errors are 3302/3283; 234 repairs came from control-coverable errors and 4 from control-missing-candidate errors. At >.50, coverage is 7755/7806 and coverable errors 3303/3387; repairs split 325/9. These are recorded-candidate diagnostics only. Full arrays, all phases, both modes and within-arm transitions are in EXPERIMENT_AUDIT.json.

Both 3723-line training logs have continuous step numbers, finite numeric records, the identical sequence of every batch's row IDs, 3722 batches of eight plus one batch of two, and precisely one consumption of every one of the 29,778 fit IDs. No holdout row ID is consumed. The flattened row-order SHA-256 (compact JSON encoding) is `1ab6f45572237ad8be0fc003f4a1e8cd872a4f41f6e7af66b300364e03c2810e` for both arms. Sources: `g_control/train.jsonl:1`, `:3723`; `g_p2/train.jsonl:1`, `:3723`; `source/split_protocol.json:1`; `source/run.py:449`–`:462`.

Specs are identical except `root` and `p2` (`g_control/spec.json:9`–`:51`; `g_p2/spec.json:9`–`:51`). Fresh AdamW, LR1e-5, wd5e-4 and clip0.1 follow `source/run.py:165`–`:175`, `:433`–`:434`, and the hash-matching optimizer source U:340–365. This is matched **fit-sample/update budget**, not identical parameter count, wall time, or total development cost. P2 adds 70 state/parameter tensors (180,966 parameters by static layer-size arithmetic); G/P2 logged fit-loop times are 8603.58/8761.62 seconds. P2's two-step disposable engineering preflight is separate.

The shared G checkpoint reference is `0575dfae...64522`; current terminal hash receipts agree between the train receipt and CPU audit for G (`834284f2...a0fc6a`) and P2 (`60371c5e...92d9e9`). The local binaries themselves were not inspected. Do not mistake inherited `spec.files.train.py` for the executed runner: the current train receipts' `script_sha256` matches `source/run.py` exactly (`g_control/receipt.json:37`; `g_p2/receipt.json:37`). `formal_rows:0` in training receipts describes their training/holdout scope; the separate formal receipts correctly record 9508.

Historical G 5615/4495 is supported by the original local receipt H/receipt.json:19–20, bound to the same terminal hash at H/receipt.json:61 and the earlier G training receipt at H/receipt.json:64. It was not newly evaluated in this audit. The current formal scene/target order matches all 9508 entries in H/protocol.json's `parsed_identities`; current row records omit text, so exact current text equality is not independently attested by them.

## D. Executed metric paths / dead-code detection: PASS for reported metrics

Read the complete current runner (491 lines), controller (65), analyzer (150) and native grounding evaluator (902), plus the model/loss/input paths relevant to these metrics. The runner explicitly imports the native evaluator (`source/run.py:106`–`:110`) and calls it on every evaluation batch (`:378`). The evaluator dispatches both REC modes and both Mask modes (`source/grounding_evaluator.py:194`–`:206`); its hit and mask accumulators are populated at `:302`, `:490`, `:653`, `:758`.

Before writing any successful phase receipt, the runner asserts exact native REC count agreement and Mask IoU-sum agreement within 1e-3 (`source/run.py:415`–`:424`). Receipts and all raw rows exist and agree with the separately recomputed totals. The native source SHA-256 `39c8de92...931677` also matches `../g_source_port.json:100`.

No claimed REC/Mask metric is an uncalled function. Legacy NumPy softmax, visualization/stat-printing helpers and the commented-out DIoU loss path do not produce any claimed result. The separate `verify_native_replacement` gradient witness is behind `if not update` (`source/run.py:286`); this pair's preflight/train calls use update=True (`:333`, `:452`). Thus this audit does **not** claim that the G gradient-witness helper was executed by the current pair.

Execution agreement is evidenced by the asserted writer path and its completed artifacts, not by this auditor replaying the native evaluator. A separate native-accumulator dump and the current `imports.json`/formal logs are not in the intake.

## E. Scope, initialization and claim ceiling: WARN

Only **ScanRefer, two arms, one seed (2027)** were tested. Each holdout has 6887 expressions over **106 scans/physical spaces and 1479 targets**; each formal evaluation has 9508 expressions over **141 scans/physical spaces and 2068 targets**, independently counted from raw rows. Observed holdout/formal physical-space overlap is zero. The protocol reports 456 fit spaces and 106 holdout spaces (`source/split_protocol.json:1`); the runtime verifies all scene-derived row partitions and fit/holdout physical separation (`source/run.py:177`–`:187`, `:233`–`:235`). A complete fit-row-to-scene map is not in the training logs, so the fit-scene count is protocol/runtime evidence rather than an independent raw-log recount.

The holdout is held out from this module fit, but the protocol expressly states prior author pretraining saw it (`source/split_protocol.json:1`; P:31). The 9508 set is development validation, not an untouched final test (`REPORT.md:25`–`:26`). G already underwent an earlier 3723-update/29,778-row adaptation (`../g_receipt.json:6`, `:61`–`:64`). This pass is additional to that history and author training. Do not present it as a one-pass-from-scratch method or equal total budget against original PV-Ground.

The terminal holdout bbs gain against the continuation control is +9/+27, but both arms lose bbs accuracy against their own shared starting point: G -32/-50 and P2 -23/-23 (`g_control/receipt.json:82`–`:91`; `g_p2/receipt.json:82`–`:91`). Formal P2 is -2/-76 hits relative to the historical G 5615/4495 reference and is 335 strict hits below the 4754 target. `SUMMARY.json:417`–`:428` correctly leaves every promotion condition false and Nr3D/Sr3D new-method results unavailable.

**Observed initial Mask nonidentity:** the two initial runs have exactly equal selected query IDs, selected boxes, REC IoUs and all saved oracle bits, but their Mask IoUs differ at two rows in both modes:

| Initial rows.jsonl line / row_id | G Mask IoU | P2 Mask IoU |
|---|---:|---:|
| 3439 / 16804 | 0.950276255607605 | 0.8911917209625244 |
| 3803 / 18482 | 0.8771331310272217 | 0.8976109027862549 |

Evidence: `g_control/initial/rows.jsonl:3439`, `:3803`; `g_p2/initial/rows.jsonl:3439`, `:3803`. These cause -0.000560574457 percentage points of initial mIoU for P2 and **no Mask threshold-hit changes**. The preflight compares centers, sizes and semantic scores, not masks (`source/run.py:322`; `../preflight_preflight.json:210`–`:214`). Their cause is unresolved by the retained records; do not label it proved harmless nondeterminism or claim bitwise mask parity.

Neither the +13 loose hits nor any coverable-error counts establish a robust or statistically significant improvement. One paired run cannot isolate effects of the P2 text, positional and observation branches, and both trained checkpoints change their features/boxes/scores. There is no fixed-box reranking or per-module causal gain claim available (`REPORT.md:28`; `source/ported/models/pv_ground.py:506`–`:517`).

## F. Evaluation type: PASS — real_gt

Initial holdout, terminal holdout and formal development-validation REC/Mask evaluations are **real_gt**. Candidate coverage is a **GT-assisted offline oracle diagnostic**, not a deployment metric or a proxy for actual accuracy. The CPU restore/serialization and two-step preflight are engineering checks, not additional benchmark results. No evaluation was classified as synthetic_proxy, self_supervised_proxy, simulation_only or human_eval.

## Evidence limits, separate from confirmed defects

No confirmed integrity defect was found. The following evidence is unavailable or insufficient for stronger assurance:

- Original ScanRefer JSON/ScanNet scene arrays, GT mask tensors, GroupFree detections and superpoint binaries were not in the audited intake. The matching dataset/Scan sources establish the route; they do not prove every underlying data byte.
- Raw rows retain selected boxes and scalar Mask IoU plus oracle flags. They omit complete masks, all 256 boxes, scores and per-row text/detector/superpoint hashes. Recorded Mask IoUs and oracle flags can be aggregated, but mask intersection/union and full candidate coverage cannot be reconstructed without those omitted tensors. Recorded point hashes, row/scene/target IDs and root boxes match across arms and initial/terminal; this is not a complete tensor-identity proof.
- Train logs prove all row order/consumption. They do not hash every augmented tensor. The common seeded loader/augmentation source supports the intended same-input protocol; actual per-step augmented tensor equality is not separately witnessed.
- Author/G/current terminal binary checkpoints were not loaded locally. CPU audit reports, not this auditor, witness state counts, finite optimizer states and terminal optimizer steps. Full author-pretraining provenance was not re-audited.
- Current formal receipts bind their row files but do not themselves record the loaded terminal binary hash. The execution `imports.json`, child logs/exits and a separate native-accumulator dump were not collected here. The new P2 module hash is recorded by collection; unlike the G modules, the run spec/receipt does not separately attest that module's runtime hash. These are provenance limits, not evidence that a different file was used.

## Action items

1. Preserve the negative formal outcome: primary bbs +13/-33, target fail, no historical-G retention claim. Keep bbf separate.
2. State detector-assisted/two-stage inputs, development-validation status, one seed and shared G's earlier training budget alongside any reported result.
3. Record the two-row initial Mask discrepancy; restrict zero-residual equality claims to the outputs actually compared. Do not infer a cause from these records.
4. If stronger replay/initialization assurance is required, archive the existing execution imports/logs, checkpoint hash/restore evidence and underlying GT/prediction tensors with the run. No rerun is required to accept the verified arithmetic of this bounded negative result.
5. Additional independent seeds/benchmarks and an appropriate fixed-candidate experiment would be needed before claims of robustness, Nr3D/Sr3D transfer or a particular reranking/component mechanism.

## Claim impact

| Claim | Judgment |
|---|---|
| Both arms completed a matched 29,778-row / 3723-update continuation and 9508-row formal evaluation | Supported by supplied records and independent recount; binary provenance remains bounded |
| Formal bbs is 5600/4452 versus 5613/4419; repairs/damages are 238/225 and 334/367 | Supported |
| P2 improves strict formal REC, passes the registered target, or retains historical G at both thresholds | Unsupported; actual records contradict it |
| P2 improves holdout bbs by +9/+27 versus this continuation control | Supported only with holdout/pretraining and own-start regression qualifiers |
| P2 improves formal Mask quality | Unsupported here: bbs Mask hits -8/-17; mIoU -0.008598588 percentage points |
| Initial outputs are entirely identical / Mask path is deterministically identical | Unsupported; two actual Mask IoU differences |
| Repairs demonstrate fixed-box reranking or causal benefit of one P2 component | Unsupported |
| Robust cross-seed or ScanRefer/Nr3D/Sr3D success | Unsupported; one ScanRefer seed only |

## Hash and trace record

EXPERIMENT_AUDIT.json records all 60 exact audited input SHA-256 values, 42 intake hash/size checks, all raw metric recounts and paired transitions. Supplemental source copies were accepted only where their hashes match the sealed manifests. A stale runtime-env candidate was not used; the accepted environment's canonical hash is the exact spec value `966235b2...0c82c`.

Full reviewer request/response trace belongs to the delegating executor. This auditor was restricted to the two report files and did not create a separate trace directory.
