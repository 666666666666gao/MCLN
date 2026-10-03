# Experiment integrity audit: completed local-range / whole-range pair

**Overall verdict: WARN. No blocking integrity defect found.** The records support the completed, bounded comparison. They do not establish improvement over original G or completion of the ScanRefer target.

Date: 2026-10-04T05:23:37.875639+08:00. Reviewer: fresh delegated `gpt-6-astra` / `max` requested native route, `review_independence=same-family`, `acceptance_status=provisional`. Backend SKU was not independently attested. Execution scope: local source inspection and deterministic CPU checks of saved terminal evidence. No model/Torch import, CUDA, SSH, network, checkpoint read, credential/auth-wrapper read, job change, sub-agent, primary edit or weight deletion occurred.

`R` below is `C:/Users/gb/.codex/tmp/pvground_whole_mask_fit_20261003`. `native/` means the six files in `C:/Users/gb/.codex/tmp/pvground_fused_support_20261002/complete_tail_fused_retry/source/imported/`. Every native copy's SHA256 matches **both current arms'** `complete/{arm}/imports.json:10-16`; they are not treated as current merely because they came from a previous run.

## Deterministic evidence

All **81 requested primary files / 63,476,687 bytes** were read; every JSON/NDJSON record was parsed. All **63 files named by INTAKE** match both bytes and SHA256, and all **34 Python sources** parse without import. **343 checks pass** (303 record/source checks and 40 diagnostic/log checks), with no failed check. Detailed identities and results are in `AUDIT_INPUT_READS.json`, `AUDIT_CPU_RECOUNT.json` and `AUDIT_DIAGNOSTICS_RECOUNT.json`.

The six evaluation files contain **46,564 records**, and both train logs contain **3,723 records**. Independent arithmetic reconstructed **186,256 selected final/coarse box IoUs** against saved root boxes. There are **zero disagreements at either strict threshold** with stored native IoUs. This is stronger than recounting stored IoU values, but it does not reconstruct model predictions. Mask values, full-256 oracle flags and top-k coverage remain **saved native-value recounts**: raw masks, all candidate boxes and score/logit tensors are absent. No optimizer state or checkpoint was replayed.

## A. Ground-truth provenance — PASS

ScanRefer annotation JSON supplies scene/object/text identity (`native/src.joint_det_dataset.py:585-649`). Root boxes are scan instance boxes, and GT Masks are membership in `scan.three_d_objects[tid]['points']` (`:1086-1119`); they are returned as `center_label`, `size_gts`, `gt_masks` (`:1324-1393`). They are not derived from model predictions. The native evaluator reads those targets (`native/evaluator.py:535-553,879-902`); the runner saves the same root and computes selected Mask IoU against the dataset Mask (`complete/run_whole_mask_fit.py:503-531`). All six evaluations are `real_gt`.

The model's actual input dictionary contains points/voxels, text, predicted detector boxes/classes and superpoints, not target boxes, target Masks or dataset identity (`complete/run_whole_mask_fit.py:275-285`). `butd=True`, `butd_gt=False`, `butd_cls=False` select external group-free **predictions**, not perfect GT detections (`:139-144,244-254`; dataset `:1189-1257,1354-1372`). This remains a detector-input protocol; these data do not establish single-stage performance.

G's unmatched-query qualification uses detached predicted boxes and dataset root IoU, excludes Hungarian-matched queries, and changes only the existing training CE responsibility (`complete/local_range/pvground_semantic_assignment.py:9-50`; runner `:304-313`). It is not an inference GT gate. Native model-to-model Mask correspondence targets at `native/models.losses.py:592-621` are training auxiliaries; they are not the target for the reported real-GT Mask metrics. The original dataset payloads were not independently loaded in this audit, so provenance is established from pinned code and saved identities rather than a new dataset reconstruction.

## B. Metric denominators and normalization — PASS

The runner counts strict `IoU > .25` and `IoU > .5`; denominator is 6,887 or 9,508 examples. Mask mIoU is the native selected-IoU sum divided by that row count, multiplied by 100 (`complete/run_whole_mask_fit.py:510-551`). Geometric intersection/union is the native box definition (`native/models.losses.py:35-75`), and Mask IoU uses logical intersection/union (`native/evaluator.py:897-902`). No reported score is divided by its own maximum, minimum, mean or accuracy.

Token softmax in bbs/bbf and soft foreground-mass normalization in `whole_mask_range.py:40-67` are scoring/feature construction, not metric inflation. Both arms perform that range computation; local replaces the 109 evidence channels with zeros (`pvground_whole_mask_box_refiner.py:49-54`). G retains the matched-GT denominator, EOS coefficient .1 and native .5/7 loss scaling (`pvground_semantic_assignment.py:34-50`). In all 7,446 train records, `matched_queries` equals actual batch size (8, or final batch2); added qualified queries do not enter that denominator.

## C. Result existence, numbers and completion — PASS

| Evaluation / mode | Rows per arm | Local .25 / .50 hits | Whole .25 / .50 hits |
|---|---:|---:|---:|
| Initial / bbs | 6,887 | 6,176 / 5,602 | 6,176 / 5,602 |
| Initial / bbf | 6,887 | 6,206 / 5,647 | 6,206 / 5,647 |
| Terminal development / bbs | 6,887 | 6,160 / 5,591 | 6,156 / 5,559 |
| Terminal development / bbf | 6,887 | 6,180 / 5,600 | 6,178 / 5,587 |
| Formal development / bbs | 9,508 | 5,603 / 4,428 | 5,594 / 4,461 |
| Formal development / bbf | 9,508 | 5,637 / 4,431 | 5,603 / 4,458 |

Each table entry is recounted from the full corresponding `complete/{arm}/{initial,terminal,formal}/rows.jsonl`, checked against its receipt and `analysis/SUMMARY.json#/phases`. The formal counts are at `complete/local_range/formal/receipt.json:5-19` and `complete/whole_range/formal/receipt.json:5-19`. Formal bbs whole-versus-local repairs/damages are **247/256 at .25 and 383/350 at .50**, giving **-9/+33**. Formal bbs Mask hits .25/.50 and mIoU are **5806/5114/46.93181070875354%** local and **5811/5106/46.92723284231716%** whole. All report tables match the saved evidence.

`complete/status.json:2-28` records the four completed stages in serial order. `controller.exit:1`, both `train.exit:1` and both `formal.exit:1` are 0. Train stdout records actual fit completion at local `train.log:82` and whole `train.log:83`; full formal receipts are logged at each `formal.log:26`. All logged terminal/evaluation receipts agree with the saved JSON; no Traceback/AssertionError/RuntimeError/OOM appears in these logs. INTAKE records controller closed and no remaining run-owned weight at `:85-86,404-415`.

Historical original G **5615/4495** is consistently labeled history, tied to the protected parent SHA. Its original 9,508 prediction rows are not in this packet, so that historical number is **not independently freshly recounted here**. Both new endpoints are below it; target **5615/4754 is false** (`status.json:30-40,81-82`; `analysis/REPORT.md:32-45`). Prelaunch plan/launch fields are dated snapshots; the final status, receipts and report establish completion.

## D. Called scorer paths and selected-frame identity — PASS

The runner explicitly imports the current SHA-tied native evaluator (`complete/run_whole_mask_fit.py:119-133`), calls `evaluator.evaluate` (`:498`) and asserts selected row counts/Mask sums against native accumulators (`:541-547`). `native/evaluator.py:194-206` calls all four box/Mask scorers. bbs uses token-softmax probabilities; bbf uses projected Query/token similarity divided by .07, followed by token softmax. Both use the same parsed main/attribute/pronoun/relation-minus-other text weighting and ranked top Query (`native/evaluator.py:218-303,315-390`; runner `:499-535`). `filter_non_gt_boxes=False` prevents the optional geometry filter from selecting candidates with GT.

Native Masks are generated before the range refiner, which then overwrites `last_center/last_pred_size` (`native/models.pv_ground.py:520-564`). The criterion consumes those final fields (`native/models.losses.py:898-917`), as does the evaluator. Coarse and final stored boxes use the same selected Query; the selected Mask uses that Query and the same native scalar-alpha fusion (`runner:521-535`). The saved rows consistently agree when bbs/bbf select the same Query, and their full-256 coverage flags agree regardless of ranking mode. Per-row `row_id`, `scan_id`, `target_id`, `root_box` and `point_sha256` match across arms and between initial/terminal stages.

`analyze_terminal.py:99-103,173-204` calls the inspected saved-row helper functions and refinement diagnostics. Their counts and all reported diagnostic values were independently checked. Uncalled legacy helper main functions, optional visualization/stat printing and commented DIoU paths are not current reported metrics. No phantom called metric was found. Native ranking itself cannot be rerun from this packet because logits/scores and all candidate tensors are not stored; selection evidence is the pinned called source plus successful runtime assertions and saved records.

## E. Configuration, fit and claim scope — WARN

The only spec differences are `root`, `support_arm`, `use_whole_range`, `preflight_root`. All 11 shared arm Python copies are byte-identical. Both use the same protected G SHA, official-parent identity, six native import identities, common dataset/fixture/environment paths, fresh AdamW, seed2027, nominal batch8, core/backbone LR1e-5, weight decay5e-4 and clip0.1 (`complete/run_whole_mask_fit.py:86-89,135-192,571-608`; `native/main_utils.py:268-366`). No preflight optimizer state is loaded into formal fitting.

Every arm's `train.jsonl:1-3723` contains **3,722 batches of8 and one batch of2**, totaling **29,778 unique examples in exactly the same order**. The first batch is `[14307,26871,13547,1622,9672,18692,29949,23417]`; the last is `[4176,17320]`. Those IDs plus the 6,887 development IDs exactly partition `0..36664`, with no overlap. Both logs report **456 physical fit scenes / 106 holdout scenes** (`train.log:8`), matching the source's physical-disjointness assertion. Saved formal rows cover **141 scenes**. Training augmented tensors are not saved, so identical batch IDs and seed are not a tensor-equality certificate.

All recorded loss components, corrections, gradients and timing numbers are finite. Local total loss spans **10.6055851..21.3386230**, whole **10.6813841..24.3000374**. Every update records a nonzero G correction. This verifies the recorded fit budget and active loss path; optimizer tensor replay was not performed. Controller train-mode times include dataset loading, initial/terminal evaluation, serialization and fitting, so **12,753.09 / 12,817.98 seconds are not pure optimizer/GPU times** (`status.json:8,20`; train fit markers report ~10,239.58 / 10,343.48 seconds).

Actual initial outputs are not bitwise equal. For both modes, **6,881 boxes and 5,943 native IoUs differ**; bbs has0 changed Query IDs and2 changed Mask-IoU rows, bbf has1 changed Query and3 changed Mask-IoU rows. Initial REC threshold bitmaps are identical, but each mode also has2 changed `oracle25` vectors and1 changed `oracle50` vector. Example: `complete/{local_range,whole_range}/initial/rows.jsonl:4920`, row26603, uses bbf Query225 versus176. The saved initial comparison and fit/analysis copies agree exactly with this audit; maximum selected-box coordinate difference is **0.00683259964 m bbs / 0.0734105110 m bbf**. Same-forward zero-head exactness does not imply full-process pairing.

The equal nominal **400,614-parameter** head includes109 channels zeroed only in local; equal count does not prove equal effective capacity. Both endpoints jointly update shared modules. Formal bbs same-selected-query coarse/final strict hits are **4428/4428** local (14 repairs/14 damages) and **4471/4461** whole (13/23). Median maximum face displacement is **2.705010 / 2.828427 mm**; all9,508 selected bbs moves are below1 cm. A whole-range repair example is formal row line1234; a damage example is line534. These data do not establish a positive direct final-residual effect. Coarse is a within-model diagnostic, not a separately trained baseline.

Whole gains33 formal strict hits over local but loses9 loose hits and remains **21/34 below historical G**. The 6,887-row holdout includes author-pretrained scenes and the9,508-row set is development validation. One seed supports no cross-seed significance claim. Nr3D/Sr3D, six-face distributions, quality feedback and V99 transfer are future work, not results (`FORMAL_RANGE_CONTROL_PLAN.md:7-13`; `FUTURE_METHOD_PLAN.md:3-43`; `analysis/REPORT.md:35-45`). The current report already states the main limits; WARN preserves that ceiling rather than alleging fabricated numbers.

## F. Evaluation-type classification — PASS (`real_gt`)

All six box/Mask evaluations use dataset GT. Full-256 coverage is an offline GT-assisted diagnostic: at formal bbs .50, local has3323 failed selections with a good stored full-256 candidate and1757 without; whole has3311/1736. This is not deployable accuracy or verified physical-instance identity (`runner:533-535`; `analyze_terminal.py:60-74`). Predicted Mask support is a model feature, and native Mask-to-Mask correspondence is a training auxiliary; neither changes the evaluation target to a synthetic reference.

## Retention and protected artifacts — PASS within supplied record scope

The controller waits for train and formal exits0, checks full budgets/row hashes, reconstructs selected box thresholds, verifies endpoint and protected-G hashes, and only then deletes an exact owned `terminal.pth` (`controller.py:68-134`). It selects by strict bbs@.50 and keeps the incumbent on ties. Original G remains the recorded best.

| Deleted recorded endpoint | Bytes | SHA256 | Receipt |
|---|---:|---|---|
| local_range/terminal.pth | 347116945 | `313c69e4dff9afa1dd32181e301a0a6012998ac59110c27a538eff2ea0e61b4e` | `complete/local_range/weight_retention.json:27-36` |
| whole_range/terminal.pth | 347116945 | `ed77338c4dfd64efacbd2bc5eb38d2528c1271cc0d22aab2ad45c0ae930b563b` | `complete/whole_range/weight_retention.json:27-36` |

Total recorded retirement is **694233890 bytes**. Both receipts agree with fit endpoint identities and formal row hashes, report no local weight archive, and retain protected G SHA `0575dfae333dabdf288a470fc09d8967f9867d1853a5f61d9cfc3cbcf7964522`. The post-run collector verifies named deletion paths absent and zero remaining run-owned weights (`collect_terminal.py:25-54`); its saved INTAKE contains no retained run weights and zero downloads. All63 collected source/log/row/receipt files remain exact. Neither controller nor collector contains a mutation path to V99 or unrelated protected parents; V99 is not an experimental teacher in this run. This is bounded source/receipt evidence, not a new remote V99 hash or a search for hidden archives elsewhere. No audit action touched weights.

## Blocking issues and disposition

`blocking_issues: []`. No code change, model rerun, defensive fallback, new hashing infrastructure or unrelated refactor is required to archive this experiment accurately.

Keep these nonblocking qualifications: single-seed development scope and prior training history; non-bitwise starts and unverified training tensor equality; saved-value limits for Masks, full candidates and ranking; joint-adaptation/effective-capacity limits; historical-G and bounded retention attribution. Neither endpoint should be promoted as meeting the original-G preservation/ScanRefer target. The full goal remains unmet.

The machine report contains all input SHA256 identities, A-F decisions, execution counters and exact audit-owned report paths. Workspace-required session notes were consulted during setup but were not evidence for any verdict; prior source-review conclusions were checked rather than adopted. `imports.json`/`load.json` are overwritten by the formal invocation; training identity additionally relies on fit script/spec/module/source-port hashes and successful source assertions. Remote datasets/manifests, transitive parser/backbone/CUDA dependencies, model/optimizer tensors and upstream protocol equivalence were not independently re-executed or reconstructed.
