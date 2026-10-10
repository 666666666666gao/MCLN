# Experiment Audit Report

Date: 2026-10-10. Project: PV-Ground native normal joint training, extremal_support, seed 2027.

**Overall verdict: WARN. Integrity status: warn.** The supplied artifacts support a completed three-epoch run and the recorded E0–E3 `last/bbs` counts. They show no gain from this normal fine-tuning run: the prescribed rule retains the already-trained initialization at E0. A subsequent actual attempt2 receipt supports exact state restoration of the normal E0 best and E3 latest checkpoints, after a narrowly corrected verifier-schema failure. It does not reevaluate accuracy. Independently reconstructed per-expression scores, three effective contributions, and Nr3D/Sr3D efficacy remain unestablished. No affirmative evidence of fabricated GT or prediction-statistic metric normalization was found in the inspected code.

Reviewer: fresh native Codex agent `/root/pvg_normal_terminal_integrity_20261010`. Requested route: `gpt-6-astra`, reasoning `max`. Actual runtime model and reasoning level: **UNATTESTED**; the request is not runtime attestation. `review_independence: same-family`; `acceptance_status: provisional`. The semantic review is provisional. The separately recorded deterministic equality/hash/arithmetic checks passed within their stated local scope.

All 20 original requested artifacts were directly read and SHA-snapshotted, followed by all eight recovery files supplied in two follow-up messages. The full 902-line evaluator was inspected, together with the complete executable content of the supplied training, loss, model and dataset files. The whole training-tail file was scanned; progress-only lines were also checked for continuous E3 evaluation coverage. Six directly relevant supplemental files were read: the normal controller, extremal initialization manifest, support corrector, span mixer, task-query installer and semantic-assignment objective. This reviewer performed no SSH, network request, GPU operation, model import, training, checkpoint load, deletion or credential/private-transport file read. The recovery evidence describes an executor-run operation; the auditor only read its code and artifacts. Audit outputs reside only in this review directory. Scope does not include the raw dataset corpus, imported third-party implementations, full original training log, or direct inspection of remote checkpoint contents.

## Evidence locations

References below are file:line references relative to these explicitly defined roots:

- `R` = `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2`
- `O` = `R/normal_epoch3_boundary_observation`
- `E` = `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/runtime_binding/model_source/src/grounding_evaluator.py`
- `D` = `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/runtime_binding/dataset_source/src/joint_det_dataset.py`
- `M` = `O/logs/scanrefer/extremal_support/1791573559/native_metrics.jsonl`
- `Q` = `R/normal_terminal_recovery_20261010`; `Q2` = `Q/attempt2`

Machine evidence: [DETERMINISTIC_EVIDENCE.json](DETERMINISTIC_EVIDENCE.json), [RECOVERY_EVIDENCE.json](RECOVERY_EVIDENCE.json), [input SHA before](input_sha256.before.json), [input SHA after](input_sha256.after.json), [supplemental SHA](supplemental_input_sha256.json), and `recovery_input_sha256.before/after.json`. Original input snapshots remain unchanged. The audit's standalone stdlib verifiers are [audit_deterministic.py](audit_deterministic.py) and [audit_recovery.py](audit_recovery.py). The first deterministic file records the original boundary packet's recovery-pending state; `RECOVERY_EVIDENCE.json` records the subsequently supplied recovery completion and is the current recovery finding.

## A. Ground-truth provenance — WARN (source path passes; corpus attestation absent)

The active source loads dataset GT, not model-generated reference labels. ScanRefer annotations and the split scene list are read from `ScanRefer_filtered_{train,val}.json/.txt`; `object_id` becomes the target ID (`D:585–648`). Scene data are loaded from `{split}_v3scans.pkl` (`D:203–209`). Target masks come from the annotated object's point membership and target boxes from `scan.get_object_bbox(tid)` (`D:1086–1119`); these become `center_label`, `size_gts`, and `gt_masks` (`D:1325–1348`, `D:1387–1393`). The evaluator reconstructs GT boxes from those batch labels and keeps target slot zero under `only_root=True` (`E:535–553`); masks likewise come from `gt_masks` (`E:879–895`). The GT source has no dependency on predictions in the inspected dataset code.

Model inputs are scene points/voxels, utterance, superpoints and detector proposals (`R/source/train_dist_mod.py:132–175`). Dataset GT is attached to `end_points` **after** model forward (`R/source/main_utils.py:449–458`, `R/source/main_utils.py:521–534`). Detector input boxes come from Group-Free prediction files (`D:1189–1226`), while GT-proposal substitution and filtering switches are prohibited by the active entry assertions (`R/source/train_dist_mod.py:406–410`; `D:1361–1372`). The positive/modify/pronoun/relation maps used for the native query score come from parsed expression text (`D:981–1082`, `D:1784–1900`), not a selected prediction's identity or IoU.

Prediction-derived mask extents are **predictions**, not GT: `native_mask_geometry` uses detached predicted mask logits and observed scene geometry (`R/source/native_mask_geometry.py:5–23`), then the model combines them with predicted regression boxes inside forward (`R/source/models/pv_ground.py:567–617`). The supplemental support corrector and span mixer use predicted/query features and observed geometry, with no GT argument (`R/source/mask_support_corrector.py:54–67`; `R/source/extremal_span_mixer.py:107–121`). Dataset GT is used for legitimate training-only extra supervision (`R/source/models/losses.py:959–969`; `R/source/selected_query_mask_objective.py:7–37`; `R/source/pvground_semantic_assignment.py:9–48`). These extra objectives are disabled in eval because `native_joint_training` is `self.training` (`R/source/models/pv_ground.py:615`; evaluator sets `model.eval()` at `R/source/train_dist_mod.py:198`).

Limit: the dataset JSON/TXT/PKL/superpoint/proposal bytes, unique expression IDs, scene inventory and train/val disjointness receipt were not supplied. A 9508-count assertion is not an independent proof that those are exactly 9508 unique official validation expressions. `E` is the retained model-source evaluator, SHA-matched to the source-port manifest (`R/NATIVE_SOURCE_PORT.json:557–570`); its parity with an independently obtained official ScanRefer evaluator was not demonstrated in this packet. The raw-corpus/official-parity portion therefore remains unverified, while source-level GT provenance passes.

## B. Score normalization and metric identity — PASS

The primary metric is top-1 axis-aligned box success using strict `IoU > threshold`, for thresholds 0.25 and 0.5. Standard intersection-over-union is implemented as intersection divided by the union of predicted and dataset boxes (`R/source/models/losses.py:35–75`). The evaluator ranks candidates by the native text-map score, obtains IoU against dataset GT, increments an integer success count, and increments the GT count by the number of target expressions (`E:217–303`). No benchmark score denominator comes from the model's maximum, minimum or mean prediction score. Softmax and loss normalizations are not normalization of reported benchmark accuracy.

The run returns the same evaluator/model's two raw `last_/bbs/top1` counters, asserts both denominators equal 9508, and appends these counts (`R/source/train_dist_mod.py:252–257`; `R/source/main_utils.py:172–176`). Each recorded epoch is internally one model state; the counts must not be mixed across epochs. The file's `same_complete_model: true` flag is authored by this return path, not independent checkpoint attestation.

| Epoch | Hits @0.25 / 9508 | Acc @0.25 | Hits @0.5 / 9508 | Acc @0.5 | Δ hits vs E0 (@0.25 / @0.5) | Both count gates |
|---|---:|---:|---:|---:|---:|---|
| E0 | 5677 | 59.707615% | 4920 | 51.745898% | 0 / 0 | Yes |
| E1 | 5652 | 59.444678% | 4596 | 48.338241% | −25 / −324 | No |
| E2 | 5552 | 58.392932% | 4552 | 47.875473% | −125 / −368 | No |
| E3 | 5576 | 58.645351% | 4488 | 47.202356% | −101 / −432 | No |

Source: `M:1–4`. Percentages and deltas were independently recomputed from integers. The 5658/4850 count gates in `R/source/main_utils.py:167–169` correspond to strictly exceeding 59.5%/51.0% over 9508 rows. E0 exceeds those count minima by 19/70 hits. E3 is 1.062263/4.543542 percentage points below E0.

The E3 log's `position alignment` Top-1 0.58645/0.47202 matches 5576/9508 and 4488/9508 at its printed precision (`O/train_tail.txt:648–649`). The name is established by `E:125–139`: `bbs` is printed as position alignment. `semantic alignment` is `bbf`, and `mask@kiou overall25/overall50` 0.616533/0.538389 is mask success, not the primary box result (`E:160–161`, `E:773–800`; `O/train_tail.txt:650–651`, `O/train_tail.txt:695–701`). Substituting those larger mask values for `last/bbs` would be invalid.

## C. File existence, execution, selection and checkpoint evidence — WARN

All 20 requested paths exist. The four JSONL rows occur exactly once with epochs `[0,1,2,3]`; all have 9508 rows and `last/bbs`. Parsed metrics are identical in JSONL, `normal_status.json`, `controller.log`, and both copies in the boundary observation. The status/controller bodies are equal. Admission matches the launch record; both controller and protocol bytes match admission's hashes. The observed child argv is exactly the protocol argv plus the prescribed init/log/experiment arguments. There is no `--checkpoint_path`, `--frozen`, debug, eval-train or alternate-data argument in that argv (`O/normal_status.json:11–74`; `R/NORMAL_NATIVE_RUN_PROTOCOL.json:6–56`).

The completed-run evidence is consistent: `train.exit:1` is zero; the controller reports completion at `2026-10-10T17:57:10.138008+08:00`, elapsed 52680.562 seconds, with the four metrics (`O/normal_status.json:75–110`). The boundary observation at 18:03:39 reports controller and child absent (`R/NORMAL_EPOCH3_BOUNDARY_OBSERVATION.json:3–5`). The tail reaches E3 evaluation batch 1189/1189 and logs a latest-checkpoint save (`O/train_tail.txt:646–711`). The supplemental controller waits for the training subprocess, requires exit zero, requires all four metric rows, and hashes both existing checkpoint files before writing `complete` (`R/normal_joint_controller.py:37–63`). This supports the narrow claim that the recorded bounded run completed; copies of the same controller output are not independent execution witnesses.

The 40,000-byte tail begins mid-evaluation and contains only the later portion of E3 validation; the boundary record says the full log was 1,085,691 bytes (`R/NORMAL_EPOCH3_BOUNDARY_OBSERVATION.json:195`). E1/E2 raw logs, the actual training-dataset constructor count and per-update diagnostics are absent from the supplied results. `36665` training expressions and `36664` rows/epoch remain protocol projections requiring actual loader confirmation (`R/NORMAL_NATIVE_RUN_PROTOCOL.json:74–83`). The later recovery receipt does provide stronger update evidence: 820 populated latest-checkpoint Adam states all have step 13749, consistent with 3×4583. That does not prove unique train-row coverage or a nonzero value change in every enabled parameter.

**Selection is E0.** The source evaluates and saves E0 before the first normal optimizer update (`R/source/main_utils.py:365–370`), then compares the tuple `(both gates, gate050, hits025, hits050)` after each epoch and saves `latest` independently (`R/source/main_utils.py:167–169`, `R/source/main_utils.py:371–386`). Executing only this isolated pure ordering function on the recorded rows selects E0. No trained epoch can replace it. This initially inferred identity is also explicitly observed in the subsequent recovery receipt: `best.pth` epoch 0; `latest.pth` epoch 3 (`Q2/RECOVERY_RESULT.json:12`, `Q2/RECOVERY_RESULT.json:33`). The auditor did not directly load those payloads.

Source binding has real but limited evidence. All 10 supplied Python sources match their exact SHA and byte length in `NATIVE_SOURCE_PORT.json`; the five supplemental model/init artifacts also match. All 116 manifest paths and all historical source-path fields were checked structurally. The five collected files with intake hashes all match their recorded digests and sizes (`R/NORMAL_EPOCH3_BOUNDARY_OBSERVATION.json:168–193`). This verifies local bytes against the supplied deployment/intake receipts. It does **not** independently attest the exact imported module origins or all runtime source bytes at training completion: the supplied source-port record itself has `actual_import_origins_verified: false` (`R/NATIVE_SOURCE_PORT.json:811`), and the reviewed normal controller checks helpers/env but does not produce a terminal full-source rehash. Historical preparation flags in source-port/init/protocol and the launch-time `NOT_COMPLETED` status are historical records, not contradictions of the later terminal record.

Checkpoint hashes are controller-reported:

- `best.pth`: 615,023,752 bytes; `ed8455ddc67e4d17018e9cf499197140e15e35db3f5daee55e0eec6846faf0c4`.
- `latest.pth`: 841,676,832 bytes; `a000a1d3ea56db3ec9bcb44b49c1db68933dfaaae7c807c351ee0e5e76c2ca2f`.

Source: `O/normal_status.json:112–125`. Save code includes model, optimizer, scheduler, architecture, retained metrics and Python/NumPy/Torch/CUDA RNG state; load code requests strict model loading and restores those states on resume (`R/source/main_utils.py:132–164`). At the original terminal observation, recovery was explicitly pending and that observation had neither reloaded nor rehashed weights (`R/NORMAL_EPOCH3_BOUNDARY_OBSERVATION.json:130`, `R/NORMAL_EPOCH3_BOUNDARY_OBSERVATION.json:221`). The earlier M0 flags cannot resolve this. The subsequently supplied normal-checkpoint recovery evidence below does resolve **state-restoration** verification for those exact identities, without rewriting the historical pending observation or asserting a new score.

No paper or experiment tracker was supplied. Therefore no manuscript-number agreement or independent tracker `DONE` check beyond the supplied run status is claimed.

### C follow-up: actual normal-checkpoint recovery — PASS for state restoration; no accuracy reevaluation

The original verifier failed at `saved['retained_metrics'] == retained` (`Q/failed_attempt1/recovery.log:1–4`; `Q/recover_normal_terminal.py:39–54`), and its `RECOVERY_EXIT.json:2` is 1. In training, `args._retained_native_metrics` is the untagged evaluator dictionary; only JSONL serialization adds `epoch` (`R/source/main_utils.py:172–176`, `R/source/main_utils.py:367–370`). The failed verifier compared that saved metric dictionary against an epoch-tagged JSONL row. This schema mismatch is confirmed by source, not inferred from the author's correction label. The assertion precedes model construction and `load_checkpoint`; it does not establish failed checkpoint recovery, corrupted training, or a failed original run. Original `O/train.exit:1` remains 0.

The attempt2 source changes exactly this selection/schema fragment: it computes `retained_row`, confirms it is E0, and excludes only the recording-only `epoch` key before comparing **all** saved metric fields exactly (`Q2/recover_normal_terminal.py:39–55`). A normalized textual diff confirms no other code change; see `recovery_helper.diff`. It retains hash/size verification before payload load, epoch/state assertions, native factory and native checkpoint loader calls, and exact equality checks for model tensors, optimizer parameter groups and state, scheduler, next epoch and Python/NumPy/Torch/CUDA RNG (`Q2/recover_normal_terminal.py:43–92`). It does not call model forward, train_one_epoch, backward or optimizer.step.

An actual result exists; it is not a plan-only claim. `Q2/RECOVERY_EXIT.json:2` is 0. `Q2/RECOVERY_RESULT.json:2–48` records both exact normal checkpoint identities and the following states:

| Checkpoint | Saved epoch | Model state tensors | Populated Adam states | Saved Adam step values | Restored next epoch |
|---|---:|---:|---:|---|---:|
| best | 0 | 1295 | 0 | empty | 1 |
| latest | 3 | 1295 | 820 | 13749 for every populated state | 4 |

Both retained-metric dictionaries equal the E0 5677/4920 dictionary. This confirms that `latest.retained_metrics` is the remembered best score, **not E3's own validation score**. The native loader also checks architecture equality and strict state load (`R/source/main_utils.py:133–145`). The recovery receipt states native factory/loader use, zero new forward/optimizer/training-batch counts, accuracy not rerun, correct import paths for four core modules, unchanged normal configuration, and no training restart (`Q2/RECOVERY_RESULT.json:50–70`). Its 14 source hashes in the plan match the local computational source bytes; protocol/checkpoint/env identities agree with the original run. These are actual supplied recovery witnesses, still subject to the same-family/provisional audit attribution and the fact that the reviewer did not independently execute them.

The `4583` scheduler constructor length is hardcoded in the verifier (`Q2/recover_normal_terminal.py:66`) and is not a new measured loader count. The saved Adam step values provide separate corroboration of 13749 normal optimizer steps for the 820 populated state entries. Empty best Adam state is appropriate for a checkpoint saved before any normal update. Exact state restoration does not prove a successful next update, bitwise prediction replay, model accuracy on a new pass, or nonzero updates for every enabled parameter.

**Nonblocking provenance warning:** `Q2/SCHEMA_CORRECTION.json:10` labels `6ad98e3019ce3c34cb0c456a6569ffe2ffc3c4086b1a3f27318ac40b9fb7c46f` as the corrected helper SHA. That is the SHA of LF-normalized text. The actual file contains 104 CRLF line endings and has raw SHA `9787b25f693fb7ca7f773e2c419dbfda3dae479b12b30b82dc0052ae8e3b3097`, exactly matching `Q2/RECOVERY_PLAN.json:19`. The original helper raw SHA matches the correction record. Both raw and normalized identities are saved in `RECOVERY_EVIDENCE.json`; the correction record's 6ad98e digest must not be presented as the raw deployed byte hash. This discrepancy does not change the inspected schema fix or the plan-bound recovery result.

## D. Actual call path and dead code — WARN (primary path passes)

The live metric route is concrete: `BaseTrainTester.main` calls E0 and each epoch evaluator (`R/source/main_utils.py:367`, `R/source/main_utils.py:380`); `_main_eval_branch` invokes the model, then the criterion (`R/source/main_utils.py:510–549`); `TrainTester.evaluate_one_epoch` invokes `GroundingEvaluator.evaluate` for `last_`, synchronizes counters and returns `bbs` integers (`R/source/train_dist_mod.py:208–257`). `GroundingEvaluator.evaluate` calls all four box/mask methods (`E:194–206`); these update the printed accumulators and the primary returned counters. The E3 tail is numerically consistent with that route. There is no phantom primary metric function.

Nonblocking presentation/dead-code findings:

1. Prefixes `proposal_` and `0head_`–`4head_` are allocated and printed, but explicitly skipped during evaluation (`R/source/train_dist_mod.py:200–206`, `R/source/train_dist_mod.py:225–228`; `E:123–143`). Their logged zeros (`O/train_tail.txt:652–675`) are unmeasured empty counters, not actual zero-accuracy results.
2. `calculate_diou_3d` is defined (`R/source/models/losses.py:103–133`) but its candidate use is commented out (`R/source/models/losses.py:548–556`); no active call was found in the inspected source tree. It does not support a DIoU metric claim. The evaluator's standalone NumPy `softmax` helper (`E:37–42`) is likewise unused in the active route; Torch softmax calls are used instead.
3. `mask_pos` is printed with the `mask_sem` count (`E:152–153`). Both mask methods are unconditionally invoked once per batch expression in this run (`E:203–206`, `E:650–653`, `E:755–758`), so this shared denominator does not alter this run's printed mask mean; it should not be described as an independently normalized primary metric.

## E. Scope, training attribution and claim ceiling — WARN

The actual documented scope is one extremal-support configuration, one fixed seed (2027), ScanRefer validation with 9508 counted expressions at E0 and E1–E3, three epochs of normal fine-tuning. E0–E3 are related checkpoints, not four independent runs. The number of distinct evaluated scenes is not established by the packet. No alternate-arm normal run, multi-seed comparison or Nr3D/Sr3D result is present (`O/normal_status.json:4–9`; `R/NORMAL_NATIVE_RUN_PROTOCOL.json:84–88`).

The architecture is integrated into the normal forward/criterion route. Core/backbone/addition trainable parameters are included in AdamW; text parameters remain frozen by the model's native policy (`R/source/main_utils.py:292–306`, `R/source/main_utils.py:428–470`; `R/source/native_model_initialization.py:60–74`). The active recipe is a ScanRefer-only low-LR fine-tune: core/backbone LR 1e−6, additions LR 1e−5, batch 8, one rank; `joint_det`, `detect_intermediate` and `augment_det` are false (`R/NORMAL_NATIVE_RUN_PROTOCOL.json:34–39`, `R/NORMAL_NATIVE_RUN_PROTOCOL.json:60–86`). That document explicitly says the current recipe is not the author's mixed ScanNet-detection training recipe. This audit independently verifies the current flags/source behavior, not the correctness of the document's historical claim about every author flag.

Initialization already contains learned weights. The supplemental initialization manifest lists official, G, support and span checkpoints; support prior updates are 11169 and span prior updates 3723 (`R/init_manifests/extremal_support.json:4–19`, `R/init_manifests/extremal_support.json:26–27`). The model initialization code SHA-checks and loads these payloads (`R/source/native_model_initialization.py:29–59`). E0 is zero updates **in this normal run**, not zero training or a newly initialized architecture. The ordinary optimizer is constructed fresh and the argv has no resume checkpoint (`R/source/main_utils.py:327–352`; `R/normal_joint_controller.py:31–35`). The reported E0 success therefore cannot be attributed to the subsequent three normal epochs.

All trained checkpoints have lower primary counts than E0; the highest post-training result is E1 and it still misses both prescribed gates. Thus “normal joint fine-tuning improved the initialized model” and “the terminal E3 model meets both targets” are contradicted by these observations. No significance, confidence interval, per-expression repaired/regressed identity, root-cause attribution, robustness or three-independent-effective-mechanisms conclusion follows from aggregate counts. This audit neither invents per-row changes nor treats the earlier engineering preflight as terminal recovery. These are claim limits, not a demand to run unauthorized new training.

## F. Evaluation classification — PASS: real_gt

The benchmark box and mask metrics are `real_gt`: dataset-provided targets are compared with model predictions (`D:1086–1119`; `E:289–303`, `E:535–553`, `E:897–902`). They are not synthetic-proxy agreement scores. Prediction-derived mask geometry is part of the prediction path. The `corresponding_loss_*` auxiliary losses compare text masks with thresholded predicted query masks (`R/source/models/losses.py:592–621`), which is a model-derived training consistency target; those losses are not benchmark ground truth and do not change the classification of the reported `last/bbs` metric. No simulation-only or human-evaluation result is supplied.

## Supported and unsupported claims

| Claim | Assessment |
|---|---|
| The bounded ordinary native run completed and recorded E0–E3 metrics with exit 0. | Supported; later saved Adam counters are 13749 for 820 populated states. Unique train-row coverage is not independently observed. |
| The E0 initialized complete model is recorded at 5677/4920 over 9508, numerically exceeding both gates. | Supported as an in-process, single-seed recorded validation result; independent row-level recount and checkpoint replay remain unverified. |
| The prescribed selection retains E0 as best; E3 is the terminal/latest checkpoint. | Supported by source/metric ordering and the subsequent actual recovery receipt's saved epoch fields. |
| Normal joint training improved accuracy over E0. | Contradicted: best trained E1 is −25/−324 hits; E3 is −101/−432. |
| Terminal E3 satisfies both benchmark gates. | Contradicted: E3 is 5576/4488. |
| Exact state restoration of normal E0 best and E3 latest, including optimizer/scheduler/RNG, is evidenced. | Supported by the actual attempt2 source and successful receipt bound to the original normal hashes; no new forward or accuracy pass. |
| Recovery reevaluated accuracy or proved a successful next training update. | Unsupported; both are expressly outside the recovery operation. |
| Every trainable module/parameter received a nonzero normal update. | Unsupported without actual normal-run parameter/optimizer deltas. Source enables joint optimization, which is a weaker supported statement. |
| The run reproduces the author's mixed detection training protocol. | Unsupported and inconsistent with the current documented flags. |
| Three effective mechanisms, per-row repairs, statistical significance, multi-seed robustness, or Nr3D/Sr3D efficacy are established. | Unsupported/out of scope. |
| The complete research goal is achieved. | Unsupported. |

## Blocking versus nonblocking items and next evidence

There are **zero demonstrated hard integrity failures** in the inspected primary metric/GT code. WARN is not an approval to promote a checkpoint or expand the claims. No item blocks a qualified descriptive report of the recorded run and its regression.

**State-restoration evidence gap resolved by follow-up:** the actual normal E0/E3 attempt2 receipt supports state-only cold recovery for the listed hashes, as detailed above; no old M0 inference is used. There is no remaining blocker to that qualified restoration statement. A claim of a fresh accuracy pass, deterministic prediction replay or successful next update remains unsupported because those actions were not performed. This audit does not prescribe a redundant rerun to restate the existing in-process scores.

**Blocks an independently reproduced official benchmark claim:** supply the actual data/split/proposal manifests and unique validation row IDs, together with prediction/GT arrays or a replay receipt sufficient to recompute hits, confirm row coverage and verify evaluator parity. The present packet supports a source-level `real_gt` route and internally consistent raw counters, not an independent raw-geometry recount.

**Nonblocking for the descriptive report:** retain explicit single-seed/one-configuration scope; name `last/bbs` and E0 selection; exclude unmeasured prefix zeros from tables; do not substitute mask results; distinguish author-recipe assertions from the current verified flags; preserve the absent full training log/per-row coverage and training-time runtime-origin attestation as limitations. Record the schema-correction helper digest as LF-normalized versus the actual plan-bound CRLF digest. The source-port/init/protocol and boundary `pending` fields remain historical, with the later recovery receipt separately cited.

The claim of a positive normal-training gain is already contradicted, not merely awaiting a favorable audit. Any later mechanism-efficacy, per-row repair or significance claim would require its own appropriate evidence. No new GPU work, extra seed, rerun or training continuation is authorized by this report.

## Trace and execution notes

`001-integrity.request.json` preserves the supplied full review request; `001-integrity.delegation.txt` preserves the actual task message. `002-terminal-recovery.request.json` preserves both follow-up messages and their added paths. `001-integrity.response.md` and `002-terminal-recovery.response.md` are byte-identical copies of this full integrated substantive response. Run/call metadata preserve requested/attested model distinctions; `events.jsonl` remains local to this allowed directory. `execution_notes.json` retains local command failures and their successful corrected checks; the original scientific verifier failure is independently preserved in the read-only input and recovery discussion. The final handoff is separately retained in `001-integrity.handoff.md`.
