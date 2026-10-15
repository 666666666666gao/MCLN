# Experiment Audit — Actual Matched Endpoint Native Preflight Pair

**Date:** 2026-10-15  
**Overall verdict:** WARN — no blocking experiment-integrity finding for the captured engineering pair.  
**Scope:** `ACTUAL_MATCHED_ENDPOINT_NATIVE_PREFLIGHT_PAIR`  
**Review:** fresh independent-context Codex reviewer; same-family, provisional. Requested Astra/max route: **UNATTESTED** by the available host evidence.

The captured evidence supports two actual native optimizer updates in each arm and exact same-process recovery checks. It supports continuing through the existing normal-run admission gate. Formal training, formal accuracy and the larger project objectives remain unproven by this pair. This review issued no remote command, model call, GPU/process query or training update, and changed no audited input.

Paths below use **K** = `D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\coarse_endpoint_training_20261014` and **S** = K's parent `/span_lr_actual_pair_review_20261014/snapshots`. **F/C receipt** means the terminal capture's `final_duplicate/coarse_endpoint/witness/NATIVE_M0_RECEIPT.json`. The C: workspace path resolves through a junction to the physical D: path above. All 70 reviewed input byte hashes are recorded in the machine report; the terminal intake hash is `3b426779b8883cb5c6aa6bc2715cb15c673cf01191c3574de4b066bfd9ba873b`.

## A. Ground truth provenance — PASS, within the stated boundary

`S/103_joint_det_dataset.py:585-649` loads ScanRefer annotations and their dataset `object_id`; `:205-225` loads scan objects and superpoints. `:1086-1119` obtains boxes from `scan.get_object_bbox(tid)` and masks from the object's point indices, with the native training box augmentation. `:1324-1392` returns `center_label`, `size_gts`, `gt_masks` and `box_label_mask`. The criterion constructs targets from those fields (`K/source/models/losses.py:856-886`).

The extra loss reuses the actual last-head Hungarian indices and those same targets. It never substitutes a prediction as a target or rematches (`losses.py:917-919,959-966`; `matched_endpoint_geometry_objective.py:14-26`). The actual indices are recorded in both receipts: each step has 8 matches, every target is slot 0, and `other_GT_slots_matched=0` (F/C receipt:6-157).

The native predicted-mask extent and B mixed box are prediction-derived geometry inputs (`S/126_native_mask_geometry.py:5-43`; `S/124_extremal_span_mixer.py:107-121`). They are not the matched GT. The existing corresponding-mask consistency terms intentionally compare predictions during training (`losses.py:592-621`); they do not supply a claimed accuracy or dataset GT.

**Boundary:** the reviewer traced source-bound loading and the actual factory/import/receipt chain. Raw annotations, point clouds, superpoint files, actual batch tensors and GT numerical coordinates were not reread. Preservation of supplied multi-GT indices is supported by code, not an empirical multi-GT run. With `joint_det=false` and `detect_intermediate=false`, root-only targets are expected in this actual ScanRefer recipe.

## B. Score normalization — PASS

The added term is exactly
`(10 * L_bbox + 2 * L_giou) / (num_decoder_layers + 1)`,
using native center L1 plus 0.5 size L1, and native GIoU. Both losses divide by the actual original last matching count, 8 in these batches (`matched_endpoint_geometry_objective.py:20-25`; `losses.py:518-544,831-844`). The single-rank execution makes the helper's local matched count agree with the native denominator.

Neither an accuracy nor a localization success rate is manufactured by normalization against the model's own maximum, mean or minimum. Dice/IoU overlap formulas and B geometry feature scaling are not accuracy renormalization. No formal accuracy is reported.

The native trainer logs running means, not the second batch's raw loss (`S/050_main_utils.py:442-450,510-524`). All three checked log keys—total, matched auxiliary, and selected-geometry loss—match receipt running means to the printed four decimal places.

| Arm | Auxiliary losses, steps 1 / 2 | Total losses, steps 1 / 2 | Direct auxiliary endpoint gradient L1 |
|---|---|---|---|
| final_duplicate | 0.33187044 / 0.20065124 | 14.54995918 / 14.74105835 | 9.05347729 / 9.43088913 |
| coarse_endpoint | 0.56998408 / 0.36514166 | 14.78807545 / 14.90603352 | 7.96809721 / 9.05811501 |

Equal coefficients are a controlled recipe choice; they do not imply equal loss magnitudes or gradient directions.

## C. Actual result existence — WARN for stale tracking; actual artifacts PASS

The first cached observation at **22:46:13+08** records the final arm complete and the coarse arm running. The terminal observation at **22:56:16+08** records both children/controller closed; pair completion is **22:47:43+08**, exit 0, and four total native updates. Both arm exit files are `0`. The complete captured logs contain both native train steps, checkpoint save, receipt completion and no terminal exception. Shared first/terminal final-arm artifacts and admission are byte-identical.

The checker validated all 13 intake file rows across the two observations, six captured complete-log lengths, embedded/cached receipt equality, status equality, and both intake exits. The diagnostic and receipt gradients, updates and losses are identical for each arm. This conclusion is based on receipts, diagnostics, logs, source and capture consistency together.

| Engineering checkpoint | Bytes | SHA256 recorded by harness and terminal intake |
|---|---:|---|
| final_duplicate | 841,676,128 | `53230e3cef210e1fab504f39ff9136c49c61fcda4300cf08f6108251b6af94b3` |
| coarse_endpoint | 841,676,128 | `b2f15d9fa7446f47e134cb75f38c9aa1d3aa5c563f992084e7b26c67058282fa` |

See F/C receipt:630-650 and `K/OBS_preflight_actual_pair_terminal_expected_INTAKE.json:1405`. These are unscored engineering checkpoints. The reviewer compared captured hash/size evidence and did not load their payloads.

`K/EXPERIMENT_TRACKER.md:7-8` still says both preflights have not started, and `:22-24` only records launch with witness pending. These dated states lag the terminal artifacts. Append current operator state after the audit seal; retain the historical preparation/launch record. No actual accuracy number is contradicted or fabricated.

## D. Execution and dead code — WARN for unused helpers; claimed path PASS

The exact runtime calls `TrainTester` from `train_dist_mod`, uses its loaders, factory, criterion and optimizer, and invokes the inherited native `train_one_epoch` for two real batches (`K/runtime_preparation/native_matched_endpoint_preflight.py:35-38,66-116,234-247`). `S/052___init__.py:7-10` binds the factory and criterion exports. `S/107_train_dist_mod.py:101-130` constructs actual `PVGround` and calls `configure_native_model`; `S/050_main_utils.py:473-508` forwards, attaches dataset targets after forward, backpropagates the total loss, clips, and steps AdamW and the scheduler.

Both receipts list the isolated arm's actual imports (F/C receipt:643-650). These source hashes match the staged files. The 118-file parent becomes 120 files per arm: 117 unchanged, `models/losses.py` modified by exactly 7 inserted lines, plus the objective and endpoint config. All other criterion lines are byte-identical to the prior snapshot. The two arm source copies differ only in endpoint config; their protocol differences are `arm`, `model_source`, `matched_endpoint`.

The auxiliary observation wrapper executes twice, records the actual original indices, and checks direct gradients are zero outside matched queries. Every step has 8 nonzero matched queries and zero unmatched direct-gradient L1 (harness:171-216; F/C receipt:6-157). This is executed evidence, not unused helper existence.

| Full-objective parameter group | Changed elements, final / coarse |
|---|---:|
| original_core_and_G_reader | 22,796,049 / 22,786,363 |
| backbone | 3,196,793 / 3,197,341 |
| support_corrector (A) | 27,263 / 27,264 |
| span_internal (B) | 29,756 / 29,755 |
| span_output (B) | 33 / 33 |

All five recorded groups have positive finite gradient norms in both steps and positive updates (F/C receipt:189-230). G label reassignment also executes on 193/293 queries in each arm. **The core/G reader group is aggregate evidence; individual G-reader parameter changes are not separately reported.** Parameter hooks measure the total objective. The direct auxiliary endpoint gradient is separately measured; original-only gradients and gradient angles are not numerically isolated. Source preserves the original objective, including existing G and selected-support geometry terms. The original final-box objective already has paths through B to coarse geometry; this audit does not establish a previously missing gradient.

Legacy `calculate_diou_3d` is defined but its criterion invocation is commented (`losses.py:103,546-557`). `S/130_pvground_semantic_assignment.py:53` defines a separate verification helper that this preflight never calls. The inspected `S/123_scanrefer_data_contract.py` is absent from the actual staged parent source list. None of these supplies empirical evidence in this audit; no deletion or speculative defensive machinery is requested.

## E. Scope, shared state and recovery — WARN

Two arms, seed **2027**, two batches of eight presentations per arm, two optimizer steps per arm. The same two augmented-batch fingerprints and initial full-state fingerprint match exactly (F/C receipt:5,158-188):

- Initial full state: `18780eb15c725014b8f4aededd9d0084cfd2297f6c0f3c179f601544ab4ce317`.
- Input batch 1: `8e775089188d7c475658a15b344cace39b6b2c3c13b8326b3b91f97382ce0d30`.
- Input batch 2: `293d97ae2ad630e3a8293a188a84459fcfcc448a4a8b484c258735752a4eabf7`.
- LRs: `[1e-6, 1e-6, 1e-5, 1e-5, 1e-6]`.
- Trainable group elements: `[25397851, 3485376, 0, 27841, 29793]`; the text encoder group is empty/frozen.

Unique expression and scene counts are not recoverable from these records. Same initial tensors and data do not warrant a bitwise-forward-determinism claim: first-step raw gate minima differ at floating-point scale. Both arms use all 256 candidates in the native model; no candidate-quality or coverage performance is established here.

The harness saves the complete state, asserts 1,295 model state entries and Adam steps equal 2, creates a new architecture without official/G/A/B reloading, and checks exact equality of model tensors, optimizer parameter groups and every state value, scheduler, and Python/NumPy/CPU torch/all CUDA RNG states (`native_matched_endpoint_preflight.py:281-323`; `S/050_main_utils.py:134-165`). Both receipts record **820 restored optimizer states** and `full_recovery_exact=true`. This is an actual same-process restore assertion chain; cold-process restore and resumed-update equivalence were not tested.

Recorded loop times are **30.581 s / 26.956 s** for the two updates, not full epoch throughput. Peak allocated bytes are **22,181,649,408 / 22,181,634,048**, with **25,218,252,800** reserved for each; the pair's overall recorded elapsed time is **1,225.965 s**. These are captured engineering measurements, not a performance comparison.

The admission binds the unchanged warm interpreter and canonical env hash `966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c`. The actual launch uses `flock --no-fork -n` on the existing GPU-0 lock. The reviewed controller checks prior terminal status, prior process absence, one A100, idle device, reserve and staged bytes before running children serially; its child environment fixes device 0 and world/rank sizes to one (`matched_endpoint_preflight_pair_controller.py:15-37,45-66`). The completed chain is consistent with these invariants; this reviewer made no fresh hardware/process observation.

Formal training must still pass its existing locked admission. The normal controller launches the original trainer with the shared initialization spec and prohibits `--checkpoint_path` (`normal_matched_endpoint_controller.py:25-27,63-67`). The factory loads official/G/A/B payloads with their declared hashes on fresh initialization (`S/127_native_model_initialization.py:29-60`); the normal path creates fresh AdamW and scheduler. **No M0/preflight checkpoint is a normal initialization source.** Both actual receipts report normal training false and preflight weights unused. This is verified source intent and captured preflight state, not a future normal-run result.

## F. Evaluation type — real_gt engineering sanity

The matched endpoint objective is dataset-supervised `real_gt` training sanity. Native corresponding-mask consistency terms are `self_supervised_proxy` training components, and native mask/B geometry is prediction-only inference computation. The pair does not execute a formal evaluator or produce an accuracy. Dataset lengths 36,665/9,508 and the 4,583-update normal-loader length verify construction/configuration only.

No claim is supported about a 9,508-row gain, retained capability improvement, three independent mechanisms, target thresholds, Nr3D/Sr3D, or whole-project completion.

## Deterministic verification and action items

`actual_pair_review/audit_cached_evidence.py` ran with `python -B -X utf8`, exit 0, tool chunk `eb4c91`. **11 checks passed**, including all 70 input hashes, syntax parsing of 24 source files without importing them, source lineage and exact insertion, captured evidence consistency, shared state/inputs/LRs, actual gradients/updates, checkpoint intake agreement and log arithmetic. Details are in `DETERMINISTIC_CHECKS.json`. No project/torch/NumPy import, dataset load, payload load, live query, remote action or neural execution occurred.

There are **zero blocking findings** for this engineering-pair scope. Keep the nine machine-readable warnings with downstream claims. Append the actual terminal state to the tracker after sealing; use the existing serial normal admission and fresh initialization. Do not label this review cross-family accepted, a verified Astra/max execution, formal accuracy evidence, cold recovery, or empirical multi-GT validation.

