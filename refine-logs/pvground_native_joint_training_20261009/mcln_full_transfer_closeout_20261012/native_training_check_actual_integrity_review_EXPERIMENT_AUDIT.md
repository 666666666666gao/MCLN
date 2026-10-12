# Experiment Audit — native MCLN two-update engineering check

Date: 2026-10-12. Overall **WARN**, **0 blocking findings**. The narrow engineering claim is supported: two native supervised updates completed and the program verified saved-state restoration in the same process. The WARN preserves the validation-construction note and the limits below; it is not a formal training or accuracy result.

Requested reviewer: `gpt-6-astra`, effort `max`. Actual model, effort and backend: **UNATTESTED**. Review route: **same-family / provisional**. This is the same designated integrity auditor continuing into this new scope, not a fresh context. The request supplied expected counts and requested zero blockers; those statements were treated as claims and checked against primary artifacts. The earlier full9508 audit was not repeated.

`TC` below is `native_training_check`; `source/` and `TRANSFER_SPEC.json` below are in its parent transfer directory. Full input byte snapshots and hashes are in [INPUT_MANIFEST.json](INPUT_MANIFEST.json); the actual file inventory is [LOCAL_INPUT_INVENTORY.json](LOCAL_INPUT_INVENTORY.json).

## Independently verified local evidence

The stdlib-only [verifier](verify_saved_check.py) passed **70 checks**, with zero failures. It decoded the original observer payload and independently verified the three transferred files' exact bytes, lengths and SHA256 values, then checked the two complete log tails against their raw payloads. `status`, standalone `RESULT` and the final JSON log line agree exactly; the two preceding JSON lines agree with the two update records. These are consistent copies from one execution, not independent training repetitions. See [INDEPENDENT_CPU_EVIDENCE.json](INDEPENDENT_CPU_EVIDENCE.json).

| Artifact | Bytes | SHA256 |
|---|---:|---|
| status.json | 3568 | `9c0aa0d4ba582f8aafd5a904d2463c51cdf37ec9938055fa0416ddb7ed3cb1c4` |
| admission.json | 270 | `19c63cd2e2269c4b725b164f88eef13426330a9c9df5bd53ce0bfb70d4e199ec` |
| check/RESULT.json | 2547 | `4eb11c083ee4f1435d6687a3ea5c87be4e62a3f9eabe0acb6f5054a72433a501` |

The whole check log is 4,278 bytes and the controller log is empty; both are shorter than the observer's 16,000-byte tail limit, so these saved tails cover their complete recorded files. The controller source sends child output to check.log and itself prints nothing. No `child.exit` exists in the saved file inventory; the actual child return code is recorded through `process.wait()` in status, not inferred from a missing exit file.

Launch was at **13:46:27.103127 +08:00**, child PID **271102**. The child actually exited **0** at **13:56:17.934245 +08:00**, elapsed **590.796276547 seconds**. RESULT was written at **13:56:10.147456**. The successful observation at **13:59:50.908641** reports both controller and child absent. Sources: `check_controller.py:27`, `LAUNCH.json:14`, `native_training_check_corrected_due_boundary_observation/status.json:20`, original `native_training_check_corrected_due_boundary_STDOUT.json:1`.

## A. Ground-truth provenance — PASS

The check calls `TrainTester.get_datasets(args)` and iterates only `Subset(train, range(16))` (`check_native_mcln_portable_training.py:93`). Native train construction uses `split=train` with the inherited joint-det/detect-intermediate/augment-det/BUTD settings (`source/train_dist_mod.py:2606`, `:2892`). Dataset annotation JSON supplies object identities and text (`source/src/joint_det_dataset.py:585`, `:633`); masks come from annotated object point membership, boxes from `scan.get_object_bbox`, with the native train-time box jitter (`:1086`). They are not produced from model predictions.

The distinction between two kinds of supervision matters: native text maps and spans already enter `_get_inputs` (`source/train_dist_mod.py:3463`). Spatial `center_label`, `size_gts`, `gt_masks` and batch targets are attached only **after** `predictions = model(inputs)`, then consumed by native loss (`check_native_mcln_portable_training.py:114`, `:115`, `:118`; `source/models/losses.py:2229`, `:2297`). It would be false to say that no annotation-derived information enters native forward at all. The added A/B adapter receives query/support features, points, predicted logits and predicted geometry; it has no spatial-GT argument/read (`source/mcln_portable_geometry.py:9`, `source/mask_support_corrector.py:54`, `source/extremal_span_mixer.py:107`, `source/native_mask_geometry.py:5`).

This is a source/dataflow verification backed by the successful native execution. Original batch tensors and the remote dataset itself were not locally reloaded in this audit.

## B. Score normalization — PASS

The two numbers are unnormalized native scalar losses from `_compute_loss`, printed as `float(loss.detach())`; gradient diagnostics are sums of absolute tensor elements before clipping (`check_native_mcln_portable_training.py:101`, `:118`, `:128`, `:134`). No benchmark score or accuracy is calculated. Native mask/feature normalization and the span coefficient clamp are internal model operations, not a division that inflates reported performance.

| Update | Rows | Recorded raw loss | A output / internal gradient L1 | B output / internal gradient L1 | Trainable-core gradient L1 |
|---|---:|---:|---:|---:|---:|
| 1 | 8 | 13.5062255859375 | 0.5426797271 / 5.0308179930 | 4.3816020489 / 24.4119589310 | 37947.2922194073 |
| 2 | 8 | 12.062352180480957 | 0.1718717813 / 2.7148427945 | 0.2241130173 / 0.8610909802 | 18642.9861614232 |

The loss values belong to different batches; their ordering is not convergence or accuracy evidence. The audit independently reconciled the recorded L1 components, not their absent gradient tensors.

The invalid-reference counts are **1351/2048 (65.966796875%)** and **1209/2048 (59.033203125%)**, with denominator `8*256` query references per update. They are not failed training rows. Gate fractions have denominator `8*256*3=6144` axes per update. Recorded raw gate ranges are `[0.0204575695, 0.0993678793]` and `[0.0215257090, 0.0925630182]`; both strictly lie inside `[0,1]` and agree with zero reported below-zero/above-one clamping fractions. These source-derived denominators and arithmetic are in [INDEPENDENT_RECORDED_STEP_CHECKS.csv](INDEPENDENT_RECORDED_STEP_CHECKS.csv). No pass/fail quality threshold is defined for invalid-reference frequency by this engineering check.

## C. Execution and result existence — PASS

The raw receipts, checksums, existing files, successful return code and completion boundary establish a completed recorded execution. All quantities cited above exist in `native_training_check_corrected_due_boundary_observation/check/RESULT.json:2` and are corroborated by complete log content. No result is inferred from executor prose.

The current R2 gate exactly matches its archived `SOURCE_REVIEW_20261012_134532.json`; the archived R1 matches `reviewed_v1`. R2 precedes launch. Independently rehashed **25/27** listed static inputs match R2, including the check, corrected controller, launcher and relevant native source. The two excluded transport/environment files were not opened, as required by scope. This is local continuity verification, not a new remote hash attest. The launcher contains reviewed-source and prior-stage-completion admission checks before it starts the controller (`launch_check_authorized.py:12`, `:23`, `:28`).

The runtime recorded a saved **793,579,034-byte** engineering checkpoint, **1159 model state_dict entries**, and **731 AdamW state entries**. These are runtime metadata, not bytes/tensors independently fetched or compared by this audit. No local engineering checkpoint is present in the enumerated task inputs and no saved checksum of that checkpoint is provided. Its exact contents cannot be independently recounted from JSON alone.

## D. Active implementation — PASS

The model is native MCLN, initialized from the hash-checked E71 core and the exact PV support/span states (`check_native_mcln_portable_training.py:40`, `:74`; `source/mcln_transfer_initialization.py:11`). The loader removes the nine selector entries, strictly loads the 1135-entry original core and checks equality; it then strictly loads only A/B and checks 1159 combined entries. Transfer identities in `TRANSFER_SPEC.json` are E71 `3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`, A `d06adb8ed227d98d46afa8e3970dcb9fbe31ab2613c675883b80162f830f4cb6`, B `f9898c5dc7a5e97bf152421efc9072c23ef255618c64477a4bfacf996f2b3b19`. The auditor verified the loader and spec, without opening those remote checkpoint files.

Native `model.forward` invokes the active full-transfer adapter before returning (`source/models/mcln.py:3167`). A corrects superpoint mask logits and B modifies the final center/size; native loss consumes those final outputs (`source/mcln_portable_geometry.py:19`, `:30`; `source/models/losses.py:3194`). The hard reference extent is detached/discrete; this check proves the observed active gradients, not differentiability through every hard membership operation.

The check obtains native criterion/AdamW, places A/B in one explicit LR group, asserts no duplicate optimizer parameter IDs and exact coverage of all `requires_grad` parameters, and performs backward/clip/optimizer/scheduler calls twice (`check_native_mcln_portable_training.py:81`, `:89`, `:110`; `source/main_utils.py:5187`, `:5801`, `:6653`). Native frozen RoBERTa parameters are preserved and asserted (`source/models/mcln.py:1364`; check `:80`). Actual recorded gradients are nonzero for both internal and output parts of both modules on both steps; the trained A/B parameter comparison reports **27265/27841** and **29793/29793** changed elements (`RESULT.json:60`). The denominators were independently recomputed from the source's Linear dimensions using AST, without importing a model. Changed-element numerators were not independently recounted from absent tensors.

Core gradients are aggregated over non-A/B trainable parameters and are nonzero in both updates. This supports participation of the trainable native core in the update path. The script stores before/after tensors only for A/B, so it does not measure a delta for every original-core parameter; that stronger claim is unsupported.

## E. Scope and restoration limits — WARN

Actual training scope is **one run, seed 2027, two batch-8 updates, 16 subset rows**. **48655** is the length of the constructed training dataset, not the number of rows trained. `joint_det`, `detect_intermediate`, `augment_det` and `butd` are recorded true (`RESULT.json:5`). The dataset source forwards and uses these flags, and native augmentation is active for train. Exact sample/scene identities were not saved; this audit cannot assign a scene count. The script's `max_epoch=8` configures the scheduler object; its executed loop is the two-batch subset, not eight epochs.

**Nonblocking scope note W1:** the native dataset factory also constructs an unused validation dataset (`source/train_dist_mod.py:2938`). The saved complete log says `Loading val files` at line 8. Only the returned train object is iterated; there is no validation forward, accuracy or validation-based selection. A zero-validation-file-access claim would be false. Retain this qualification; no source change is needed to close the narrow completed check.

The program writes model, optimizer, scheduler and Python/NumPy/CPU/CUDA RNG to an actual checkpoint and reads it back (`check_native_mcln_portable_training.py:164`). It deliberately changes one B output bias and advances CUDA RNG, then restores and asserts exact recursive equality of every saved structure (`:168`–`:181`). The successful result occurs only after these assertions. This supports **runtime-tested saved-state exact restoration in the same process**. It does not establish cold restart, process reconstruction, dataloader/sampler continuation or a post-restore training step. Optimizer/scheduler are reloaded and compared but are not deliberately perturbed. The comparison is to the saved trained state, not to the original pre-training E71/A/B states.

No training tensors, before/after tensors, gradient arrays or RNG states were transferred locally. This audit therefore does **not** claim an independent local tensor equality check. The runtime checkpoint metadata and assertions are the available evidence.

## F. Evaluation type — PASS: real_gt supervised engineering check

The check uses real dataset targets in native training. It is a two-update integration and serialization check, with `formal_accuracy=null`, `cold_restart_verified=false`, `formal_training_completed=false`, and no best promotion (`RESULT.json:68`; corrected observation). It is distinct from both the already completed direct PV-trained A/B transfer evaluation and a formal MCLN adaptation run. It imports E71 plus the portable A/B modules; the PV-only six-source G and its alternate supervision are not transferred (`source/mcln_transfer_initialization.py:19`, `:31`; `TRANSFER_SPEC.json:12`).

## Repaired defects and retained history

1. **Controller environment omission:** R1's controller omitted explicit inherited runtime environment. The R2 source adds the existing environment mapping/PYTHONPATH and passes `env=variables` to its only child `Popen`. The independent diff matches the reviewed diff, and the unchanged in-scope reviewed inputs match. See [INDEPENDENT_CONTROLLER_R2.diff](INDEPENDENT_CONTROLLER_R2.diff), `check_controller.py:11`, `:23`.
2. **Observer payload missing stage:** the failed `due_recovery` receipt is exit 1, empty stdout and `KeyError: 'stage'`. In the embedded code, `b['stage']` is read before status. The only source repair is serializing `dict(launch,stage=args.stage)` at line 66. The corrected observer exited 0 and returned the actual already-completed status; no training or GPU work was introduced. See [INDEPENDENT_OBSERVER_STAGE.diff](INDEPENDENT_OBSERVER_STAGE.diff) and the preserved failure/success receipts. The initial first-boundary owner file alone is not an execution-result receipt and is not treated as one.

## Claim impact and closure

The two-update native training integration and same-process saved-state recovery claims are supported with the evidence qualifications above. Cold restart, formal adaptation accuracy, convergence, best-model improvement, and independent local checkpoint-tensor comparison are unsupported and are not asserted. There are **0 blocking findings** for this engineering claim and one retained scope note.

PLAN and admission explicitly exclude these engineering weights from formal initialization and model selection. The saved receipts, source snapshots, request, report and full trace are archived here. Cleanup under the existing user authorization is a separate action after closure; this auditor performed no cleanup and does not attest that it occurred.

All audit execution was local CPU/stdlib only: **0 SSH calls, 0 GPU queries, 0 neural forwards, 0 optimizer updates, 0 experiment-file changes, 0 local checkpoint tensor comparisons**. Semantic review remains same-family/provisional with actual identity UNATTESTED. Deterministic acceptance applies only to the completed saved-byte, source-hash, AST and arithmetic checks.
