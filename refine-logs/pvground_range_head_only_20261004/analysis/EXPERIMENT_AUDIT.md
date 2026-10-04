# Experiment integrity audit — frozen-G range-head terminal pair

**Overall verdict: WARN. Integrity status: warn. Blocking issues: none.**

The saved evidence supports a completed, qualified negative ScanRefer development experiment. The local and whole arms have native bbs hits **5588/4447** and **5589/4456** at IoU > .25/.50. Whole versus local is +1/+9, with 26 strict repairs and 17 damages. Both remain below the recorded historical original G (5615/4495); the development target 5615/4754 is unmet. No result-changing integrity defect was found in the audited packet. This does not establish a stable source gain, generalization, or successful boundary refinement.

- Review date: 2026-10-04.
- Reviewer task: `/root/pvg_head_only_terminal_integrity`.
- Route: requested/configured native Codex `gpt-6-astra`, reasoning `max`, `fork_turns=none`; **same-family / provisional**. Backend SKU was not independently attested.
- Execution: local source/record reads and stdlib-only deterministic CPU checks. No project-code imports, model execution, CUDA, SSH, network, job queries, weight reads, credential/auth-wrapper reads, or primary-artifact edits.
- Exact input inventory, byte counts and SHA256s: `AUDIT_CPU_RECOUNT.json#/inputs`. All **84** requested files were read; every saved JSON/NDJSON was parsed. The **63** collected files match INTAKE. Six native-source copies match both arms' current recorded imports.
- CPU verification: **874 checks, zero failures**, all **46,564 evaluation rows** and **7,446 training records**. Selected final/coarse box IoUs were independently reconstructed for both scoring modes, with zero threshold disagreements in any phase. Exact verifier source is preserved in `AUDIT_CPU_METHOD.json`; focused row differences are in `AUDIT_CPU_DIAGNOSTICS.json`.

Paths below are relative to `C:/Users/gb/.codex/tmp/pvground_range_head_only_20261004` unless prefixed **N**. **N** means `C:/Users/gb/.codex/tmp/pvground_fused_support_20261002/complete_tail_fused_retry/source/imported`. NDJSON line numbers are one-based record numbers.

## A. Dataset ground-truth provenance — PASS

ScanRefer annotations supply scene and target object identity; the dataset reads `ScanRefer_filtered_<split>.json`, maps `object_id` to `target_id`, and obtains target boxes and binary masks from the scan object's box and point-membership annotations. The runner records the first annotated object's center/size as `root_box` and evaluates against `gt_masks[:,0]`. These evaluation targets are not generated from model predictions.

Evidence: **N**`/src.joint_det_dataset.py:585`, `:634`, `:1086`, `:1101`, `:1327`, `:1387`; `run_range_head_only.py:499`, `:504`, `:525`. The source identities are bound by `complete/local_range/imports.json:10`, `complete/whole_range/imports.json:10` and independently recomputed SHA256s in the CPU report.

Inference receives points/voxels, utterance, detected boxes/classes and superpoints. It does not receive target boxes, target masks or the target object ID. Both `butd_gt` and `butd_cls` are false; detected inputs come from saved GroupFree detections. Text maps used by the native scorer come from utterance parsing; the commented code that would append the annotated object name is inactive. Auxiliary GT metadata exists in the dataset, but the reported bbs/bbf score does not use the auxiliary-box map.

Evidence: `run_range_head_only.py:249`, `:257`, `:280`; **N**`/src.joint_det_dataset.py:681`, `:981`, `:1189`, `:1354`; **N**`/evaluator.py:281`, `:384`.

The matched-G semantic replacement uses training GT IoU > .5 to qualify unmatched slots, detached from geometry, with the native matched-target denominator. This qualification is not an inference filter. Here the semantic/Mask paths are frozen, so the correction is computed but does not train the head; native final-box losses do. Native Mask correspondence losses also use predicted masks as training consistency targets, but these are not reported as dataset-GT evaluation scores.

Evidence: `complete/local_range/pvground_semantic_assignment.py:9`, `:34`; `run_range_head_only.py:309`, `:319`; **N**`/models.losses.py:562`, `:849`.

Limit: original dataset annotation/point files and transitive data-loader dependencies were not supplied for independent data reloading. This is a source-and-saved-record provenance finding, not a new authentication of the dataset bytes.

## B. Metric denominators and native scoring — PASS

Box IoU is intersection divided by prediction/GT union; Mask IoU uses binary intersection/union. Hit rates use the actual row count (6887 or 9508), and Mask mIoU is the sum of saved per-example IoUs divided by that count. Raw hit counts and Mask sums remain available. There is no performance normalization by the model's own maximum, minimum or average output.

Evidence: **N**`/models.losses.py:35`, `:70`; **N**`/evaluator.py:297`, `:485`, `:897`; `run_range_head_only.py:506`, `:524`, `:537`; both arms' six `*/receipt.json` files and corresponding `rows.jsonl` records 1–6887 / 1–9508.

The saved bbs query uses token-softmax position-alignment scores; bbf uses the projected-query/token dot product, temperature .07 and token softmax. Both sum main/modify/pronoun/relation scores and subtract other-entity scores, as in the bound native evaluator. The runner then uses the same selected query for its final box, coarse box and fused native Mask. It preserves all 256 candidates; `filter_non_gt_boxes=False`. Final sizes and coarse sizes use the native 1e-6 evaluation floor.

Evidence: **N**`/evaluator.py:194`, `:217`, `:281`, `:325`, `:384`, `:535`, `:594`, `:683`; `run_range_head_only.py:479`, `:490`, `:495`, `:517`, `:539`.

The whole-range module's probability-weighted spatial summaries and point-cloud normalization are model input features, not evaluation-score denominators. The 109 range channels are zeroed only in the local control; both arms execute the same provider and retain the same local support and six normalized coarse-box channels.

Evidence: `complete/local_range/whole_mask_range.py:12`, `:40`; `complete/local_range/pvground_whole_mask_box_refiner.py:16`, `:49`.

## C. Files, numbers and actual completion — PASS

The complete status, controller log and INTAKE status agree. Controller plus both train/formal exits are zero, with exactly four ordered completed stages. Both training logs contain steps 1–3723 and exactly 29,778 unique fit IDs: 3722 batches of eight plus one batch of two. The two complete batch sequences are identical. All recorded numeric losses and gradient norms are finite. The fit/holdout ID sets are disjoint and together cover IDs 0–36664.

Evidence: `complete/status.json:2`, `:4`, `:81`; `complete/controller.log:1`; all five `*.exit:1`; `complete/local_range/train.jsonl:1` through record 3723 and `complete/whole_range/train.jsonl:1` through record 3723. Actual fit-completion events are `complete/local_range/train.log:82` and `complete/whole_range/train.log:83`.

| Evaluation / scoring | Rows per arm | Local hits .25 / .50 | Whole hits .25 / .50 |
|---|---:|---:|---:|
| Initial / bbs | 6887 | 6176 / 5602 | 6176 / 5602 |
| Initial / bbf | 6887 | 6206 / 5647 | 6206 / 5647 |
| Terminal / bbs | 6887 | 6159 / 5537 | 6165 / 5539 |
| Terminal / bbf | 6887 | 6189 / 5591 | 6194 / 5584 |
| Formal / bbs | 9508 | 5588 / 4447 | 5589 / 4456 |
| Formal / bbf | 9508 | 5621 / 4474 | 5622 / 4480 |

The independent recount matches every phase receipt and the corresponding SUMMARY metrics, paired repair/damage counts, training transitions and all formal selected-box diagnostics. Both final and coarse IoUs were rebuilt from saved six-coordinate boxes and saved root GT, across both modes and all six sets: **186,256 box-IoU reconstructions**, **zero .25/.50 threshold disagreements**. Mask sums/threshold hits and full-256 oracle bits were recounted from saved values, not reconstructed from raw predictions.

Formal native closure: `complete/local_range/formal.log:26` and `complete/whole_range/formal.log:26`. Claims checked: `analysis/REPORT.md:8`, `:17`, `:22`, `:29`, `:32`; `NEXT_EXPERIMENT_DECISION.md:5`, `:7`, `:9`, `:24`. Machine verification: `AUDIT_CPU_RECOUNT.json#/evaluations` and `#/cross_arm_pairs`.

The original plan's “planned” and launch's “result unavailable” fields are historical snapshots, not terminal status. The decision explicitly says the next experiment is not implemented or trained. The publisher is conditional on the actual audit result; it was inspected but not executed here. Historical original-G 5615/4495 is consistently labelled and arithmetically used, but its original evaluation rows were not in this request.

## D. Called metrics and unused code — WARN

All currently claimed REC and Mask metrics have executable call paths: `evaluate()` calls all four native scoring routines; the runner calls that evaluator, checks its aggregate REC and Mask totals, writes per-row evidence and produces the matching receipts. The analysis calls `read_evaluation`, `compare_rows` and its own refinement routine. There is no phantom claimed metric.

Evidence: **N**`/evaluator.py:194`; `run_range_head_only.py:494`, `:537`; `analyze_terminal.py:175`, `:206`; `C:/Users/gb/.codex/tmp/pvground_candidate_consistency_20261003/analyze_complete_initial_qualified.py:14`, `:54`.

The inherited `calculate_diou_3d` metric/loss helper is defined but has **zero active calls**; its proposed loss wiring is commented out. The old `whole_range_loss_routes` and `native_mask_loss_routes` diagnostics are also not called by this frozen-head runner. Its retained `verify_native_replacement` branch requires `update=False`, whereas both actual runner call sites pass true. They cannot be cited as newly executed head-only checks.

Evidence: **N**`/models.losses.py:103`, `:546`; `complete/local_range/whole_model_preflight_checks.py:5`; `complete/local_range/pvground_tail_preflight.py:13`; `run_range_head_only.py:335`, `:425`, `:595`. `AUDIT_CPU_DIAGNOSTICS.json#/static_scope_checks` records the AST call checks. This is nonblocking: the current reports do not claim DIoU training or executed Mask-gradient diagnostics. No cleanup/refactor is required to accept these records.

## E. Scope, comparability and claim ceiling — WARN

There are two source conditions, one seed (2027), and one additional fit pass per condition from the same trained original-G parent. The 6887 initial/terminal records cover **106 physical scenes / 1479 scene-target pairs**; formal9508 covers **141 physical scenes / 2068 scene-target pairs**. Saved holdout and formal physical-scene IDs do not overlap. Training logs report 456 fit scenes; fit scene identities were not independently reconstructible from the ID-only training log. The 6887 set is declared pretrained-seen module holdout, and the 9508 set is development validation.

Evidence: `complete/local_range/train.log:8`, `complete/whole_range/train.log:8`; both arms' evaluation rows; `analysis/REPORT.md:39`, `:45`; `run_range_head_only.py:198`, `:235`.

Only `root`, `support_arm` and `use_whole_range` differ between specs. All eleven copied local/whole Python modules are byte-identical. The head has ten tensors and 400614 parameters by explicit layer dimensions. Source sets only that head trainable, sets the full model to eval and then the head to train, checks original persistent state equality after fitting, and requires absent frozen gradients. Formal reload restores only the head delta over the original-G state.

Evidence: `local_range_spec.json:9`, `whole_range_spec.json:9`; `run_range_head_only.py:135`, `:182`, `:217`, `:343`, `:567`, `:601`; **N**`/main_utils.py:339`. The formal/fit receipts attest the result; this audit did not compare weight or optimizer tensors. `load.json`/`imports.json` are overwritten by the later formal process, so their `fresh_optimizer` flag alone is not a training optimizer-state witness. Fresh training AdamW is supported by the actual training code and complete records.

Starting outputs are **not bitwise paired** despite identical saved inputs/GT and identical REC threshold decisions. Initial bbs changes query on one row, boxes on 6882 rows and Mask IoU on two; bbf changes query on two rows, boxes on 6881 and Mask IoU on two. The maximum selected-box coordinate differences are 0.1895313263 m (bbs) and 0.0748627186 m (bbf), so do not dismiss every selected-box difference as a tiny rounding error.

Exact examples: `complete/{local_range,whole_range}/initial/rows.jsonl:4920` (row_id26603, bbs query225→68; bbf225→176), and record4958 (row_id26641, bbf203→116). Initial comparison arrays reproduce exactly in the independent recount.

Within each arm, initial→terminal selected query and coarse box/IoU are identical for both modes. Local Mask IoUs are identical; whole Mask IoU differs on record3439 / row_id16804 (0.9502762556→0.8911917210). Across formal arms, queries match on all9508 rows, but Mask IoU differs on record5003 / row_id5002 (0.8592000008→0.8312693238). These observed output differences must remain distinguished from the source/receipt assertion that parameters and persistent buffers are fixed. This audit did not identify the specific runtime numerical mechanism.

The within-forward bbs strict refinement is **4495→4447**, 81 repairs/129 damages (local), and **4495→4456**, 84 repairs/123 damages (whole). Whole median maximum face movement is **16.843475 mm**, p99 **51.607381 mm**. These support the stated failure to preserve selected-box quality, not a unique explanation of optimization failure. Whole full256 strict coverage rises **7884→7974**, but the final selected result declines by39. Coverage is a GT-assisted oracle diagnostic; it does not certify physical instance identity or achievable deployed ranking.

The report/decision already preserves these limitations. The +9 between arms is a descriptive single-seed development delta, not established statistical robustness or a clean ranking-only effect. No new Nr3D, Sr3D, six-face decoder, quality readback or V99-teacher performance is supported.

## F. Evaluation-type classification — PASS

- Primary REC and Mask results, all phases: **real_gt**.
- Selected coarse/final box comparisons: **real_gt**, within-forward paired geometry diagnostics.
- Full256/top-k coverage and error decomposition: **real_gt-backed offline oracle diagnostics**, not deployed accuracy and not model-generated GT.
- Native predicted-Mask correspondence terms: training consistency losses, not evaluation metrics.
- Human, simulation-only or synthetic-reference benchmark results: none claimed or accepted.

Evidence: dataset/evaluator paths in A–B; `run_range_head_only.py:506`, `:529`; `analysis/REPORT.md:40`, `:43`.

## Retention and preservation boundaries

The controller evaluates a restored terminal head, checks row hashes and CPU-selected-box threshold counts, compares strict bbs against the recorded incumbent, and deletes only a nonbest `terminal.pth` under the two owned arm directories. It explicitly excludes the protected parent and rehashes that parent after deletion. Atomic rolling saves retain a single recovery endpoint; the terminal analysis does not delete weights.

Evidence: `controller.py:88`, `:98`, `:109`, `:117`, `:123`, `:130`; `run_range_head_only.py:573`, `:600`. Both `complete/*/weight_retention.json:27` record exactly one owned endpoint, **4,921,029 bytes each**, total **9,842,058 bytes**. Their hashes match the fit receipts and their formal-row hashes match the collected rows. The saved collector checked deletion absence and original-G SHA; INTAKE records no retained owned weights and zero downloads (`complete/INTAKE.json:404`, `:411`, `:416`; `collect_terminal.py:37`, `:41`, `:43`).

The 63 collected text/source/row artifacts remain hash-exact; no weight file exists under the local complete packet by filename inspection. The deletion code's allowed paths exclude original PV/G and V99. However, this audit did not inventory the live server, other directories or V99 weights. Thus protected-chain preservation and no extra archives are accepted only within the recorded operation's source/receipt/collection boundary, not as an independent machine-wide proof.

## Nonblocking issues and required qualifiers

1. **N1 — Incomplete reconstruction scope.** Raw Masks, all candidate boxes/logits, optimizer states, actual checkpoint tensors and full raw dataset are outside the packet/audit. Report source/receipt checks, selected-coordinate reconstruction and saved-scalar recounts separately.
2. **N2 — Output nondeterminism is observed.** Preserve actual initial and repeated-Mask differences. Frozen parameters do not imply byte-identical predictions.
3. **N3 — Limited research scope and unmet target.** One seed, two arms, pretrained-seen holdout and development validation support a bounded negative result. Stable improvement, independent reranking gain and three-benchmark success are unsupported.
4. **N4 — Inactive inherited checks.** Do not claim DIoU, old Mask-gradient diagnostics or the unreachable semantic-gradient witness executed in this run.
5. **N5 — Historical and retention proof boundary.** Original-G score provenance and protected V99/live-filesystem state were not independently replayed or inspected. Use the saved records with the stated boundary.

No source change, fallback, compatibility layer, extra training or weight operation is required to preserve the verified current claims. Parent-owned review tracing and any publication remain separate actions; this reviewer wrote only the two audit reports and identified audit-owned CPU reports.

## Claim impact

| Claim | Finding |
|---|---|
| Both fixed-budget head-only arms completed; quoted current metrics are correct | Supported by source, closed records, hashes and independent recount |
| Only the existing head learns while original persistent G state stays fixed | Supported by source/runtime assertions and receipts; no independent tensor replay |
| Whole is +9 strict hits over local on this development run | Supported descriptively, with one-seed and starting-output qualifiers |
| The head preserves or improves original-G selected accuracy / reaches target | Unsupported; current evidence is negative and reports it as negative |
| Full256 coverage is deployable performance or physical-instance correctness | Unsupported; offline oracle only |
| Stable generalization or completed new Nr3D/Sr3D/V99/six-face results | Unsupported and not claimed as completed |

The reviewer started as a fresh non-forked task. Workspace-mandated identity/daily-memory initialization files were also read at session start; they were not used as result evidence. No prior review verdict was adopted as this audit's verdict.

