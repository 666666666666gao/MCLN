# MCLN transfer replay — experiment integrity audit

**Date:** 2026-10-12. **Overall verdict: WARN. Integrity status: warn.**

**Reviewer route:** fresh Codex auditor; requested `gpt-6-astra`, reasoning effort `max`; actual model and effort **UNATTESTED** because the host does not expose them. `review_independence: same-family`; `acceptance_status: provisional`. Deterministic calculations below are completed local CPU checks, not independent model-family acceptance.

**Audited project:** `C:\Users\gb\.codex\tmp\mcln_portable_best_transfer_20261012`. Evidence references below are relative to that directory. The audit wrote only `actual_integrity_review/`; it ran no SSH, GPU queries, neural inference, training, or experiment-source/result edits.

The saved full replay exists and supports the reported three-arm, seed-2027 comparison on 9508 ScanRefer validation expressions from 141 scenes. All 28,524 selected-box IoUs were independently recomputed from saved GT/prediction coordinates in float32 and match the recorded IoUs **exactly**. Reported hits, percentages, transitions, and size-group counts also match. Two concrete recording defects remain: a post-clamp size statistic is called “raw,” and the CPU admission names an earlier source-review hash whose report is not retained in the enumerated package. Neither establishes an error in the full selected-box accuracy.

## Concrete findings

### F1 — The “raw nonpositive size” statistic is measured after clamping

**Severity: nonblocking WARN; affects a diagnostic claim, not the reproduced accuracy.**

`source/evaluate_mcln_portable_pair.py:125` replaces `last_pred_size` with `clamp_min(1e-6)` before `row_evidence` saves `selected_box` at lines 83–109 and 130–136. `analyze_paired_rows.py:49–50` nevertheless names a statistic from those saved boxes `raw_predicted_nonpositive_size_rows`; the three zeros appear in `FULL_PAIR_ANALYSIS.json:10`, `:27`, and `:44`.

The issue is observable, not hypothetical: `full_paired_rows.jsonl:2983` (zero-based index 2982, scene0304_00, target 11, selected query 75) has all three native dimensions equal to float32 `1e-6`. Its original three size values were not saved. The saved zero therefore cannot establish that raw native sizes were all positive. The audit preserves this record in `actual_integrity_review/CLAMPED_NATIVE_ROW_2982.json`.

The clamp itself agrees with the existing native evaluation path: `source/main_utils.py:7897–7899` clamps prediction sizes before evaluation, and `source/models/losses.py:712–722` uses the same box-size floor. The native selected IoU for this row is independently reproduced as 0; all three arms remain misses at both thresholds. No metric correction is indicated.

**Minimal action:** rename the existing diagnostic to `saved_post_clamp_nonpositive_size_rows` and label the raw pre-clamp count unavailable. If a later authorized replay needs raw-size diagnostics, record those before clamping. Do not infer a raw count or rerun inference solely to repair this wording.

### F2 — The CPU admission cannot be matched to the retained source-review report

**Severity: nonblocking WARN; affects the early review trail.**

`cpu_first_boundary_observation/admission.json:5` records source-review SHA256 `9530dcc8d4bd1a6b9c3c85ed399500c3eb856e42b0d298647b6586c8814a1268`. The current `source_review/SOURCE_REVIEW.json` hashes to `24ca3c6177aa87828a91cfbc72fdc5226a3024184b2487d9828c5fe13e4e3bf0`. The enumerated `source_review/` contains the final report, request, narrative trace, checker, checker result, and report writer, but no report bytes matching the CPU admission's earlier digest. The current report's `reviewed_at` is 08:47:47 CST; `cpu_LAUNCH.json` records launch at 08:47:35 CST. These are recorded timestamps, not an inference about what the earlier report said.

Sanity and full admissions both match the retained report exactly (`sanity_first_boundary_observation/admission.json:5`; `full_first_boundary_observation/admission.json:5`). All 753 retained source-review input hashes and all 736 source-manifest entries match current files. CPU, sanity, and full saved status/result/log bytes match their original observer transport payloads, and all child exits are 0. Thus the mismatch does not demonstrate failed CPU execution or a wrong full deployment; it leaves the earlier review version uninspectable.

**Minimal action:** preserve immutable review versions alongside admission hashes. Recover the earlier report if its bytes still exist; otherwise retain this explicit gap. Do not reconstruct a report and present it as the original.

## Checks A–F

### A. Ground-truth provenance — PASS, with local-evidence limits

The loader reads ScanRefer validation scene IDs and annotations from `ScanRefer_filtered_val.txt/.json` (`source/src/joint_det_dataset.py:585–597`, `:633–649`), loads dataset scans from `val_v3scans.pkl` (`:203–209`), and builds target masks from object point membership and boxes from `scan.get_object_bbox(tid)` (`:1085–1119`). Validation construction explicitly uses split `val` with `eval_train=False` and `debug=False` (`source/train_dist_mod.py:2938–2970`; `source/evaluate_mcln_portable_pair.py:47`, `:65–67`). There is no model-generated reference in this path. Predicted detector input is used; `butd_gt` and `butd_cls` are asserted false (`source/evaluate_mcln_portable_pair.py:55`; dataset substitutions are at `:1361–1372`).

The original input interface already forwards **text parsing maps/spans**: `positive_map`, modifier/pronoun/other/relation maps, and applicable structured text fields (`source/train_dist_mod.py:3454–3485`). These must not be described as absent from model inputs. Main text maps come from utterance parsing and token spans (`source/src/joint_det_dataset.py:981–1082`); their token normalization is at `:1624–1651`.

**Spatial GT** (`center_label`, `size_gts`, `gt_masks`, target boxes) is attached only after `model(inputs)` returns (`source/evaluate_mcln_portable_pair.py:119–124`). Native loss consumes those labels for evaluation at `source/models/losses.py:2229–2311`; no optimizer update follows. The added support, geometry and span modules read prediction features, masks, alpha, points, superpoint members and native predicted boxes, not spatial GT or GT-IoU eligibility (`source/mcln_portable_geometry.py:9–37`; `source/mask_support_corrector.py:24–66`; `source/extremal_span_mixer.py:29–121`; `source/native_mask_geometry.py:5–43`). Their literal-key audit is saved in `SOURCE_CONTRACT_CHECKS.json`.

Saved root GT boxes are equal across all three arms for every row and constant across repeated scene/target identities. The original dataset archives, canonical annotation JSON and raw masks are not in this local transfer package; this audit therefore establishes provenance from the actual loader/recorded execution plus saved GT, not a new hash verification of the remote dataset contents.

### B. Score normalization and native evaluation — PASS

Primary scoring follows the supplied native `last/bbs` path exactly: softmax `last_sem_cls_scores`; `_parse_gt` binarizes **only the main map**; main + modifier + pronoun + relation − other; default query-axis descending argsort; all 256 candidates remain eligible; select top 1 (`source/src/grounding_evaluator.py:935–986`, `:989–1017`, `:1624–1688`, `:1732–1754`, `:1992–2009`). `row_evidence` uses those same native parse/ranking methods and term order (`source/evaluate_mcln_portable_pair.py:79–109`). No selector/reranker or detector-overlap filter is enabled (lines 49–50, 69, 85–86).

The native evaluator uses strict `IoU > threshold`, not `>=` (`source/src/grounding_evaluator.py:1747–1754`). Before writing RESULT, the replay asserts each native evaluator denominator is 9508 and its integer hits equal the paired-row counts (`source/evaluate_mcln_portable_pair.py:139–150`). Reported accuracy is `100 * hits / 9508`; mIoU is the mean over all 9508 expressions. There is no metric division by prediction maxima, minima, means, or selected subsets. Text softmax and module feature/weight normalizations are not accuracy normalizations.

Independent CPU box computation uses center ± clamped-size/2, **boundary-difference volumes**, intersection/union, float32, and strict thresholds, matching `source/models/losses.py:712–752`. Maximum saved/recomputed IoU difference is 0 for all arms. The audit additionally checked float64 threshold sensitivity; detailed counts are in `INDEPENDENT_CPU_EVIDENCE.json`. F1 limits only what can be claimed about sizes before this native clamp.

### C. Result existence, execution, deployment, and unchanged states — WARN (F1, F2)

The files, observation receipts and results are real local artifacts, enumerated in `LOCAL_FILE_INVENTORY.txt` and hashed in `AUDITED_INPUT_HASHES.json`. The full paired file has 15,766,406 bytes and SHA256 `9562bb3a4bea8fef04fc2bd7adb9aa3110224c1193bf2b4f6bc81436582ea417`; the saved full RESULT hashes to `e15731bfbf868b118ec162eec0cbba53850f3fea2d1dbde39f500b9ca9ebabae` (`full_ROWS_TRANSFER.json:2–8`). Both hashes were recomputed locally.

CPU factory completed at 08:47:51 CST, sanity at 08:51:50, full at 10:14:55, with exit code 0 in each saved status and `child.exit`. Full started at 08:58:59, took 4556.04 seconds including setup, and was observed at 11:13:49. Its complete saved stage log contains progress through 9216 rows and the final 9508-row RESULT (`full_first_boundary_observation/stage.log.tail.txt:8–17`). The observer's original base64 file payloads, extracted status/result/admission/exit files, and entire saved logs agree byte-for-byte; the “tail” files happen to contain these complete short logs. `DEPLOYMENT_EVIDENCE.json` records all comparisons. The earlier source-only import/asset warnings did not prevent actual CPU, sanity, or full completion.

Checkpoint identities used by the replay are:

| Role | Recorded SHA256 | Runtime enforcement |
|---|---|---|
| MCLN E71 core | `3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208` | `checked_payload`, remove exactly 9 SourceChoice states, strict 1135-state core and tensor equality |
| A support terminal | `d06adb8ed227d98d46afa8e3970dcb9fbe31ab2613c675883b80162f830f4cb6` | hash, `selected_query` / zero-geometry identity, exact 10-key prefix, strict load |
| B span terminal | `f9898c5dc7a5e97bf152421efc9072c23ef255618c64477a4bfacf996f2b3b19` | hash, `extremal_support` identity, matching A parent hash, strict load |

The exact checkpoint paths are in `TRANSFER_SPEC.json:2–10`. Enforcement is at `source/mcln_transfer_initialization.py:11–47`; the successful CPU RESULT and stage log are actual execution evidence, not a claim that the auditor reopened remote checkpoints. No checkpoint binaries are in the local transfer package. The spec's expected E71 byte count is declarative; SHA256 is the actual enforced identity check.

The CPU run asserts the 1135 core tensors remain equal after installing A/B, confirms 27,841 and 29,793 module parameters, preserved core trainability and frozen RoBERTa (`source/check_mcln_transfer_cpu.py:29–44`). The replay clones all 1159 state tensors before evaluation and asserts exact equality afterwards before RESULT (`source/evaluate_mcln_portable_pair.py:61–63`, `:141`, `:160–162`). All branches use one once-read raw batch, deep copies, and restored Python/NumPy/Torch CPU/all-CUDA RNG; native semantic logits are asserted equal for every batch (`:73–77`, `:112–136`). All 9508 saved selected-query indices agree across arms, and the eight sanity records equal the first eight full records exactly. Before/after tensors, full logits and RNG snapshots are not archived locally, so unchanged states and semantic equality are **completed runtime-assertion evidence**, not an independent local tensor replay.

### D. Dead code and active metric paths — PASS for reported metrics

The executable replay calls the native evaluator and `row_evidence` for each arm and batch, then checks native counters before serializing RESULT (`source/evaluate_mcln_portable_pair.py:130–161`). `GroundingEvaluator.evaluate` calls position/semantic box and mask evaluators (`source/src/grounding_evaluator.py:786–799`); `last/bbs` is the reported primary result. Added modules are connected at the end of MCLN forward (`source/models/mcln.py:3167–3172`). Compared to the supplied baseline origin, the only MCLN source changes are three attributes and that tail call; the original evaluator, loss, trainer and main utility bytes are unchanged. The four portable modules match their listed origins byte-for-byte (`SOURCE_CONTRACT_CHECKS.json`; `MCLN_DIFF.patch`).

Actual saved selected boxes change on 9462 rows for native→Mask reference and 9495 rows for reference→full transfer. Full transfer changes selected own/fused mask IoU on 609/517 rows. These outcomes are consistent with the active geometry/support paths. `whole_mask_range.mask_range_evidence` is an unused pre-existing helper in this replay; `member_statistics` is the called provider. No published metric is attributed to that unused helper. Other native evaluator statistics are computed but not exported as claims in this replay.

### E. Scope assessment — PASS for the explicitly bounded comparison

Full scope is 9508 indexed expressions, 141 scenes, one dataset, seed 2027, one complete three-arm replay. Sanity is eight expressions, not a second full run or an accuracy claim. There are 9506 unique scene/target/text tuples, with repeated text identities at indices [2991,3009] and [5798,5801]. `ann_id` is not saved; repeated text is not evidence of an invalid duplicate. Exact canonical annotation multiset membership cannot be independently rechecked from this package.

This is **direct transfer of the specified PV A/B terminal states to an MCLN E71 core**, with zero optimization updates. It is not MCLN adaptation training, an independent A-versus-B ablation, three effective modules, or a multi-seed/generalization result. PV's six-source G core is not loaded; no Face branch is installed (`source/mcln_transfer_initialization.py:19–47`; `TRANSFER_SPEC.json:12–13`). `normal_training_completed`, `best_promoted`, and `full_goal_complete` are all false in the actual full RESULT. Those limits are also explicitly retained in `EXPERIMENT_PLAN.md:7–13` and `FULL_PAIR_ANALYSIS.json:145–154`.

### F. Evaluation type — PASS: `real_gt`

The full and sanity metrics compare real predictions to dataset-derived spatial labels. Full-256 maximum IoU is a **GT geometric coverage diagnostic**, not deployed selection accuracy or model input. The CPU factory stage has no neural forward and no accuracy; the present NumPy audit is a deterministic recomputation, not an additional experiment. There is no synthetic, self-supervised or simulation reference in the audited metric path.

## Independently verified result and denominator

Each percentage below uses all **9508** expressions and strict thresholds.

| Arm | Hits @0.25 | Acc @0.25 | Hits @0.5 | Acc @0.5 | Saved Full256 coverage @0.5 |
|---|---:|---:|---:|---:|---:|
| native | 5515 | 58.003786% | 4409 | 46.371477% | 7886 |
| mask_reference | 5459 | 57.414809% | 4767 | 50.136727% | 8159 |
| full_transfer | 5473 | 57.562053% | 4777 | 50.241902% | 8165 |

| Comparison | @0.25 repaired / broken / net | @0.5 repaired / broken / net |
|---|---:|---:|
| native → mask_reference | 138 / 194 / −56 | 694 / 336 / +358 |
| native → full_transfer | 138 / 180 / −42 | 695 / 327 / +368 |
| mask_reference → full_transfer | 18 / 4 / +14 | 23 / 13 / +10 |

The net full-transfer change versus native is −0.441733 percentage points at 0.25 and +3.870425 points at 0.5. Incremental A/B versus Mask reference contributes +0.147244 and +0.105175 points. These are paired counts, not proof that A and B are independently beneficial. At 0.5, native-based repairs are drawn from 5099 initial misses and breaks from 4409 initial hits; reference→full repairs are drawn from 4741 misses and breaks from 4767 hits. The JSON includes the corresponding 0.25 denominators and every repair/break index.

Saved own/fused mask mIoU re-aggregates to native/reference 0.4590117541 / 0.4589964131 and full transfer 0.4594650310 / 0.4592812710. Full256 and mask numbers are checked **only by re-aggregation of saved scalar IoUs**. Raw 256-candidate coordinates, point masks and mask intersections/unions were not retained, so this audit does not claim independent geometric recomputation of those quantities.

The subsequently supplied `diagnose_reference_regressions.py` and `REFERENCE_REGRESSION_DIAGNOSIS.json` were also read directly. A separate local verifier derived break/repair membership from the audit's recomputed IoUs and independently reproduced all four comparisons, quartile counts, volume-ratio medians and normalized face-error medians exactly. For example, the 194 native→reference breaks at 0.25 have median predicted/GT volume ratio 4.082160; 79 retain saved fused-mask IoU >0.5. This is a GT-conditioned offline description, not proof of a physical-instance error or a cause. The source explicitly retains `cause_proven=false` (`diagnose_reference_regressions.py:23–48`); the audit agrees with that limit. Evidence is in `REGRESSION_DIAGNOSTIC_CPU_EVIDENCE.json`.

## Claim impact and follow-up

- **Supported:** the saved full three-arm, same-input transfer comparison and the selected-box counts above; the improvements/regression must both be reported.
- **Supported with the stated evidence boundary:** exact named checkpoint hashes and all-state invariance were enforced during successful recorded execution. Local checkpoint/tensor reinspection was unavailable.
- **Needs correction:** “raw nonpositive predicted-size rows = 0.” The measured value is post-clamp; pre-clamp values are unavailable.
- **Needs a trace qualifier:** CPU-stage source approval refers to an unretained earlier report; sanity/full approval hashes do match.
- **Unsupported by this replay:** MCLN training completion, separate effectiveness of A and B, effectiveness of three modules, transfer of G, meeting 60%/52%, or replacing the retained PV best result.

No further inference or training is required to substantiate the selected-box arithmetic or to correct the two recording issues. Training-policy and budget decisions are outside this read-only audit.

## Audit artifacts and limits

`INDEPENDENT_CPU_EVIDENCE.json`, `full_selected_box_recomputation.csv`, `sanity_selected_box_recomputation.csv`, `DEPLOYMENT_EVIDENCE.json`, `SOURCE_CONTRACT_CHECKS.json`, `AUDITED_INPUT_HASHES.json`, and `.aris/traces/experiment-audit/2026-10-12_run01/` contain calculations, row indices, hashes, request and report trace. The verifier is `verify_saved_evidence.py`; its final execution exited 0 on local Python 3.7.6 / NumPy 1.21.6. It does not import experiment code or PyTorch.

The first checker attempt stopped on F2 and is preserved as `CPU_CHECK_ATTEMPT01_*`; the second reached output serialization and exposed an audit-script NumPy-int JSON conversion error, preserved as `CPU_CHECK_ATTEMPT02_*`. The final script stores the real F2 mismatch and explicitly converts that count for JSON; it does not relabel the mismatch as PASS. Primary source/results were unchanged.

This audit reviews the listed replay/initialization/adapter/launch/observer/analyzer code and the active native factory, input, dataset, loss and evaluator paths. The other packaged files received identity checks, not a claim of exhaustive semantic review. No claim is made to have independently rehashed remote data/checkpoints, verified prior PV training history, or established backend model identity.
