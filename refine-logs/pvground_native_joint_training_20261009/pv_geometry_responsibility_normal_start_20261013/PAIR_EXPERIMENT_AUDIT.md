# Actual native PV preflight integrity audit

**Verdict: WARN — bounded engineering preflight passes; no blocking findings.**

The final actual pair completed successfully. This review accepts only the saved two-step native engineering evidence. It does not establish formal accuracy, a full epoch, a scientific improvement, or cold ordinary resume.

Review route requested gpt-6-astra/max. Actual model and reasoning identity are **UNATTESTED**; the review is **same-family / provisional**. This fresh delegated reviewer read primary artifacts directly and verified the handoff assertions. The handoff included factual summaries, so this is not presented as a blind, executor-narrative-free review. Reviewer activity: 0 SSH, 0 GPU queries, 0 model forwards, 0 optimizer updates, 0 progress queries, 0 installs. No private MEMORY was read; all writes are confined to this review directory.

## Bound evidence

- Scope: `ACTUAL_SELECTED_GEOMETRY_NATIVE_PREFLIGHT_PAIR`.
- Exact final intake: `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\OBS_preflight_paired_terminal_1_INTAKE.json`.
- Raw SHA256: `2babf6208183c9c455d54eebbd0e3e52adc756fcc41615c1bacfbe37ea7a20d1`.
- The C: workspace alias resolves to the D: path above; `audited_input_hashes` uses that resolved absolute key and a raw 64-hex value as required by the launcher.
- Original observer exit 0; collection **2026-10-13 00:16:02.264706 +08:00**, after due **00:05:19**. Pair complete/0 at **00:03:09.078501**. Controller **296485** and child **297059** were absent in the actual observation.
- All **8 decoded files**, **3 saved log tails**, receipt/diagnostic fields and intake reconstruction match original stdout bytes. **45 input files** are hash-bound; **11 local executing-source files** match the R4 manifest, which contains **118** remote source entries. This reviewer did not re-query the remote machine.

## A–F integrity checks

**A — GT provenance: PASS.** The staged dataset source reads ScanRefer annotation `object_id`, then builds masks from annotated scan-object points and boxes from the corresponding scan object. Native criterion targets are the loader's `center_label`, `size_gts`, `gt_masks`, and valid slots. The strategy never substitutes predictions for GT. This verifies the source chain and actual successful factory receipt; it does not re-download or certify dataset files.

**B — score normalization: PASS.** Extra geometry uses native L1/GIoU, `(10*L1+2*GIoU)/7`, with actual batch denominator **8**. Original matching denominators remain untouched. Own/fused mask eligibility is integer **2I>U**, with no predicted-score normalization and no Box-IoU eligibility threshold. Native bbs arg-sort remains the selector.

**C — result existence: PASS.** Both actual arm receipts and before-acceptance diagnostics exist, match the original byte transport, report two actual updates, and have child exit 0. The first incomplete intake was not accepted; its control receipt is identical to the final one.

**D — executed path: PASS within scope.** Actual `TrainTester.get_loaders/get_criterion/get_model/get_optimizer` calls lead to native `train_one_epoch`, forward, native criterion plus G, backward, clipping, and AdamW. Extra geometry records and direct output gradients establish that the new training hook ran in the strategy arm. Full formal evaluator/diagnostic-finish functions are source-wired but were not executed here and receive no result credit.

**E — scope: WARN.** One seed (**2027**), two serial arms, **2 updates and 16 row presentations per arm**: **4 updates / 32 presentations total**, on the same two B8 batch dictionaries. Unique-expression/scene counts are not exported. Dataset lengths **36665 train / 9508 val** and normal loader length **4583** are factory observations, not completed epochs or 9508-example accuracy results.

**F — type: real_gt, actual native training engineering preflight.** Formal accuracy is null. This is neither a synthetic proxy result nor a formal benchmark score.

## Initialization, gradients and geometry responsibility

Both protocols share all common arguments. Their only differences are arm, initialization-spec path and geometry flag; the initialization specs differ only in the geometry flag. Both logged `checkpoint_path=None`, seed2027, ScanRefer-only, `joint_det=False`, `detect_intermediate=False`, `augment_det=False`, B8, 256 queries, six decoder layers, lr core/backbone1e-6, A/B1e-5, WD5e-4 and clip0.1. Both load the same SHA-checked official/G/A/B payload identities and construct **1295** state entries. RoBERTa remains frozen. There is no old-Adam inheritance claim.

Exact paired input SHA256 arrays:

1. `8e775089188d7c475658a15b344cace39b6b2c3c13b8326b3b91f97382ce0d30`
2. `293d97ae2ad630e3a8293a188a84459fcfcc448a4a8b484c258735752a4eabf7`

These hash the real augmented DataLoader dictionaries. No cross-arm initial tensor digest or separate fingerprint of every later voxel tensor is exported; source/spec/checkpoint binding supports the common fresh initialization, while recovery equality is checked separately for each saved checkpoint.

All task-gradient norms below are positive and finite. The JSON keeps full precision and maximum parameter changes.

| Group | Control step1 / step2 norm | Strategy step1 / step2 norm | Changed elements control / strategy |
|---|---:|---:|---:|
| Original core and G reader | 28.2905369 / 93.7258606 | 28.8251762 / 94.4219437 | 22,783,050 / 22,796,323 |
| Backbone | 1.4587525 / 5.3572121 | 1.4828081 / 5.3877883 | 3,197,174 / 3,196,977 |
| A support corrector | 0.0734098 / 0.4114381 | 0.0734093 / 0.4121856 | 27,263 / 27,265 |
| B span internal | 0.1580389 / 0.0769783 | 0.1505849 / 0.1301914 | 29,759 / 29,760 |
| B span output | 0.6669713 / 0.1465556 | 0.5487776 / 0.1516804 | 33 / 33 |

Changed parameter counts for these groups are respectively **693/102/10/12/2** in both arms. Control total losses are **13.83889102935791 / 14.264206886291504**; strategy **14.218085289001465 / 14.540265083312988**. G reassignment counts are **193/293** in each arm. The old selected-mask extra is off in both. These loss/gradient numbers do not prove accuracy improvement.

| Strategy step | Extra rows / nonzero direct-gradient queries | Extra loss | Direct output gradient L1 | Selected roles root / other / unmatched |
|---|---:|---:|---:|---:|
| 1 | 7 / 7 | 0.37919801473617554 | 7.955854892730713 | 1 / 0 / 7 |
| 2 | 6 / 6 | 0.2760017514228821 | 6.566103935241699 | 0 / 0 / 8 |

Step1 qualified selected queries: **226,108,166,157,136,15,133**; selected query115 was already matched to root and received no extra target. Step2 qualified queries: **253,188,245,64,1,177**; query118 had **0/1539** own/fused support and query84 **143/295**, so neither qualified. All 15 unmatched row qualifications were recomputed from saved integer intersections/unions; 13 pass strict own+fused IoU>.5.

The actual preflight takes `autograd.grad(extra, [last_center,last_pred_size])` and asserts **exactly zero** at every location outside the eligible set before the successful receipt can be written. The nonzero counts equal the 7/6 eligible queries. This is direct-output supervision protection: shared parameter updates can still affect future predictions. Matched-other selection and an entirely empty eligible batch were not observed in these two batches; their behavior is source-covered only. Original matching objects and normal denominators are not replaced.

## Recovery, prior failure, and formal-launch boundary

Each arm saves **1295** model-state entries and **820** Adam states with step2, destroys the live model/optimizer/scheduler, constructs replacements in the same process, and verifies exact model tensors, Adam fields and groups, scheduler and Python/NumPy/CPU/CUDA RNG against the saved checkpoint. No following restored optimizer step was tested. Cold normal resume, DataLoader worker position and continuation of ordinary early-stop state remain unverified; native normal main explicitly requires a fresh run.

Each engineering checkpoint is **841676000 bytes**. The terminal observer reread the checkpoint bytes and matched its receipt hash:

- Control: `6bade79f415d64a51e081fe6883834163267c358b2fa868ddbad24241d731f74`
- Strategy: `2a3dc220b503ba6b9ba02aaaa61fd558ef8ce819a43790e678c8fb3788eb6fdf`

No weights were downloaded or loaded by this reviewer. Both have null retained/formal metrics and `preflight_weights_used_for_formal_training=false`. The observer's zero new-forward/GPU/update counters describe observation activity; the experiment itself performed four optimizer updates.

MCLN had already completed/0 with two strict declines at **2026-10-12 23:11:36.120282 +08:00**. Initial controller **291554** then failed at disk admission before a native child, status file or update. Its failure and archive remain intact. The successful controller296485 used the exact same R4 stage object and helper hashes after cleanup. The admission failure is not a scientific training failure.

The reviewed v4 launcher requires the actual complete/0 pair, absent controller/child, 4 total updates, matching input hashes, the correct audit scope, no blockers, and this exact final-intake SHA. It then rechecks actual receipts (2 updates / 1295 states each), uses fresh initialization without preflight checkpoints, verifies source files under the GPU lock, and schedules strategy only after native control finishes. This review launches nothing. Full E0/9508 validation and later formal results remain separate runtime evidence under the existing plan; no new acceptance gate is introduced here.

## Source anchors and audit artifacts

[Actual factory, input capture and native training call](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/native_selected_geometry_preflight.py:61>); [Direct output gradient assertion](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/native_selected_geometry_preflight.py:171>); [Same-process recovery comparisons](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/native_selected_geometry_preflight.py:232>).

[Inherited native train_one_epoch](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_overlay_v3/main_utils.py:449>); [Fresh normal-loop requirement](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_overlay_v3/main_utils.py:374>).

[SHA-checked fresh initialization](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_overlay_v2/native_model_initialization.py:13>); [Dataset target construction and actual loss hooks](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_overlay_v2/models/losses.py:856>).

[Selected-only qualification and native geometry loss](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_support_geometry_objective.py:9>); [Actual ScanRefer dataset factory](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_overlay_v4/train_dist_mod.py:56>).

[Observed processes, raw byte capture and checkpoint hashes](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/observe_study_authorized.py:43>); [Exact actual-pair audit gate](<D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/launch_normal_arm_authorized_v4.py:20>).

`DETERMINISTIC_VERIFICATION.json` contains exact numerical and byte checks. `EXPERIMENT_AUDIT.json` is the launcher-consumable report. The full request/follow-up, response, metadata, input snapshots and verifier receipts are in `.aris/traces/experiment-audit/2026-10-13_run01/`.

One local verification attempt hit Windows path length while saving a long snapshot filename. Its original script and failure receipt are preserved; changing only snapshot names to numbered `.bin` files let the standard-library verifier complete with exit0. No experiment source was modified and no scientific result was rerun.
