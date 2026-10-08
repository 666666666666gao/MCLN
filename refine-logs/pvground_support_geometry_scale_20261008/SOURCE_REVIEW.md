# Geometry-scale diagnostic source review

**Verdict: WARN — no blocking findings in the final reviewed revision.**

**Execution scope: SOURCE_ONLY. Diagnostic status: NOT_EXECUTED.** This is a fresh-context, same-family review with provisional acceptance. The requested route was `gpt-6-astra`, reasoning effort `max`; actual backend, model and effort are **UNATTESTED**. Requested routing is not backend attestation.

All 19 supplied project files were read and are bound to their byte SHA256 in `SOURCE_REVIEW.json`. The reviewer made no SSH calls, neural forwards, optimizer steps, package installations, external reviewer calls or source edits. Only the requested review artifacts were written. The executor's capacity-check revision to the launcher was reread; the bound launcher hash is `98194401fa63e98ab94fdcdf5a3b4d8f6a4b4f9a00e5dc30b48269527268a6d3`.

References below are relative to `C:/Users/gb/.codex/tmp/pvground_support_geometry_scale_20261008/` unless a different path is shown. The JSON manifest contains every complete path.

## A. Ground-truth provenance — PASS

The visible supervision route uses annotation-derived valid boxes, semantic labels, masks and text maps. `matched_mask_objective.py:6` constructs targets from `box_label_mask`, `center_label`, `size_gts`, `gt_masks` and the other dataset fields. `matched_mask_objective.py:20` matches frozen parent predictions against these valid GT targets, and `matched_mask_objective.py:40` derives majority-superpoint target masks from the annotated masks.

The data wrapper obtains object boxes from the scan annotation (`run_mask_support_pair.py:159`), checks binary GT masks (`run_mask_support_pair.py:170`), and sends raw points, text, superpoints and detector proposals/classes to the parent (`run_mask_support_pair.py:232`). GT boxes and masks are not parent inference inputs. Matched IDs in the feature hook affect reporting, not which inputs receive the signed-log transformation (`diagnostic_scale_run.py:50`).

The predicted-Mask extent in `mask_reference.py:10` is a model-side geometry reference. It is not substituted for supervision. This distinction prevents confusing a model-derived inference feature with fake GT.

## B. Metric normalization — PASS

The supervised loss uses the actual valid-GT count as its denominator and explicitly combines the four terms with coefficients 5/1/10/2 (`matched_mask_objective.py:35`, `matched_mask_objective.py:51`). The diagnostic records raw parameter and total L2 gradient norms, without clipping (`diagnostic_scale_run.py:70`).

The coarse-size division is the geometric input construction under investigation (`mask_support_corrector.py:32`); it does not normalize a reported accuracy metric. The count feature uses a fixed 50,000-point denominator (`mask_support_corrector.py:43`). The hook reports raw and used feature maxima, explicitly labels the signed-log transform, and computes ordinary distribution quantiles (`diagnostic_scale_run.py:50`, `diagnostic_scale_run.py:77`). No result metric is divided by its own maximum, minimum or mean.

## C. Artifact existence and execution status — WARN

All 19 supplied artifacts exist. The entire archived `C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/train.jsonl` was parsed: 3,723 sequential steps, 29,778 rows and 29,778 unique row IDs. Every batch except the final one has eight rows; the final batch has two. Both arms' archived matched-query and valid-GT counts equal their batch sizes.

The panel exactly matches `TRAINING_SCALE_ANALYSIS.json:784` and the archived log at lines 1, 2, 522, 523, 524, 3211, 3722 and 3723. Row order and both original gradient values match for every selected batch. Independent arithmetic on the archived log also reconciles both arms' step counts, clipped-step counts, gradient median/p99/maximum, and loss mean/median/maximum with the analysis table. This is a reconciliation of existing log data, not a new model measurement.

The diagnostic's new `diagnostic.json`, progress, import/load receipts, GPU measurements and successful state assertions are **prospective**. They were not observed in this review. `diagnostic_scale_run.py:92` writes the completion record only after the loop and final state assertion, while `EXPERIMENT_PLAN.md:13` correctly requires later independent collection of actual completion.

Remote checkpoint/data/helper availability was not inspected. Source hash assertions for these inputs exist in `run_mask_support_pair.py:45` and `run_mask_support_pair.py:76`, but their future execution is not an observed PASS. Historical accuracy counts in `EXPERIMENT_PLAN.md:13` and the 9,508/39 comment in `mask_reference.py:13` are contextual claims outside the supplied formal-result evidence.

## D. Active and inactive source paths — WARN

The actual entry binds `ScaleDiagnosticRun as PairedSupportRun` at `run_mask_support_pair.py:111`, constructs that class and calls its overridden `run` at `run_mask_support_pair.py:255`. The override asserts preflight mode and diagnostic-only scope (`diagnostic_scale_run.py:18`).

The active route calls the observed frozen parent forward, GT matcher, support corrector and matched Mask loss (`diagnostic_scale_run.py:33`, `diagnostic_scale_run.py:63`, `diagnostic_scale_run.py:68`). Its only backward-related actions are derivative calculation and gradient clearing. The inherited constructor does create two AdamW objects (`paired_support_loop.py:42`); that does not perform an update.

The inherited clipping/update path (`paired_support_loop.py:109`), checkpoint save path (`paired_support_loop.py:139`), restore/integration paths and accuracy evaluation (`paired_support_loop.py:244`) are not reached by this override. There is no active optimizer step or checkpoint creation.

The WARN is an evidence boundary: inherited `native_mask_witness` (`paired_support_loop.py:64`) and `GroundingEvaluator` are inactive here. Their existence must not be presented as a new numerical native-criterion reconciliation or validation result. The visible helper has the declared coefficients and GT denominator; this source review does not claim an executed native-loss equality witness.

## E. Scope — PASS

The planned panel has eight selected historical batches and **58 distinct training expressions**, with sizes `[8,8,8,8,8,8,8,2]`. It plans one seed (2027), eight fresh frozen-parent forwards, 14,848 candidate inputs, and three identically initialized derivative probes. The number of physical scenes is not established by the supplied row IDs and should not be reported as 58 scenes.

The runtime checks panel length, row count, uniqueness, fit membership and loaded batch order (`diagnostic_scale_run.py:20`, `diagnostic_scale_run.py:31`). All 256 queries are retained by the corrector shape checks (`mask_support_corrector.py:25`).

Fresh augmentation is explicitly enabled and RNG reset for this panel (`diagnostic_scale_run.py:26`). The plan correctly states that these are not the historical augmented inputs, Gumbel RNG sequence or learned support-head states (`EXPERIMENT_PLAN.md:5`). The output scope excludes historical spike reproduction, trained efficacy and formal accuracy (`diagnostic_scale_run.py:96`). No training run, holdout pass, formal 9,508-row validation or research-gate completion follows from this diagnostic.

## F. Evaluation classification — PASS

**Classification: `real_gt`, limited to training-panel supervised loss and local derivatives.** The feature-scale counts themselves are tensor diagnostics without a GT score. The signed-log branch is an input transformation at the same zero-output initialization, not a trained model or a newly selected deployed module.

At zero output weights, gradients of the preceding layers are expected to be zero: backpropagation through the output layer multiplies by those zero weights. A difference in reported norms therefore primarily measures output-layer sensitivity at this initialization. Even a subsequently observed smaller signed-log gradient would not establish that compression improves trained accuracy or caused the historical spikes.

## Additional execution checks

- **Routing and imports — source PASS.** All nine supplied `new_runner_files` match their spec hashes. The supplied `C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/readback_preflight_checks.py` matches the declared helper digest. Dataset/base-module origin assertions exist at `run_mask_support_pair.py:97` and `run_mask_support_pair.py:112`. The remaining remote dependencies were not imported during review.

- **Frozen parent and probe state — source PASS.** The factory freezes the parent (`mask_support_model_factory.py:13`); the entry snapshots the full state, asserts no trainable parent parameters and sets eval mode (`run_mask_support_pair.py:128`). Parent forwards use `no_grad`, backward requires parent gradients absent, each batch compares probe state tensors exactly, and the final full parent-state comparison is active (`diagnostic_scale_run.py:33`, `diagnostic_scale_run.py:74`, `diagnostic_scale_run.py:83`, `diagnostic_scale_run.py:91`; implementation at `paired_support_loop.py:58`). These assertions have not been executed here.

- **Identical initialization, Masks and losses — source PASS.** The output weights/biases start at zero (`mask_support_corrector.py:20`), content/raw heads are independent deep copies with identical state (`paired_support_loop.py:35`), and the log head is another exact copy (`diagnostic_scale_run.py:23`). Each arm checks exact equality to parent query Masks, and all three scalar losses must be identical (`diagnostic_scale_run.py:66`, `diagnostic_scale_run.py:82`).

- **Hooks and derivatives — source PASS.** The pre-hook sees the last nine channels at the first member Linear, records raw/used per-query maxima, matched maxima and raw outlier counts, and changes only those nine channels for the log arm. Capture count must equal expression count and the hook is removed before backward (`diagnostic_scale_run.py:50`–`diagnostic_scale_run.py:71`). The supplied readback helper checks the refiner/readback/native-semantic order and one final semantic-head call (`readback_preflight_checks.py:19`–`readback_preflight_checks.py:57`, full helper path above).

- **Warm runtime — WARN, runtime untested.** Canonical JSON hashing independently reproduces `966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c`, matching the declared Python 3.7/Torch 1.10.2+cu111 environment. Manual source review identified no incompatibility in the remote snippets or diagnostic route. `shlex.join` is used by the local launcher/observer, not their remote Python 3.7 snippets. The launcher reuses the existing venv and environment without installation or rebuilding (`deploy_diagnostic_authorized.py:19`, `deploy_diagnostic_authorized.py:49`). Imports, kernels and actual memory requirements remain untested.

- **Single GPU capacity and lock — source PASS after revision.** The final launcher checks no compute processes, queries index/name/used/total memory, requires one GPU0/A100 with 40,000–45,000 MiB total and under 500 MiB used, checks disk reserve, and records the queried capacity (`deploy_diagnostic_authorized.py:29`–`deploy_diagnostic_authorized.py:39`). It preserves `CUDA_VISIBLE_DEVICES=0` and acquires the documented nonblocking GPU flock around the neural command (`deploy_diagnostic_authorized.py:49`–`deploy_diagnostic_authorized.py:51`; `C:/Users/gb/.codex/tmp/pvground_cs_restart_20261002/env_spec.json:6`, `:35`). This checks physical capacity before launch; actual batch peak allocation is only known after execution.

- **Single launch and failure preservation — source PASS.** A local launch receipt guard and exclusive new remote root prevent automatic re-launch into the same task (`deploy_diagnostic_authorized.py:11`, `:25`, `:36`). There is one screen launch, with original stdout/stderr and exit status preserved (`:52`–`:54`). The observer checks the original PID, closes on terminal exit, asserts if it disappears without a receipt, and never restarts neural work (`observe_diagnostic_authorized.py:20`, `:45`, `:60`).

- **420/240-second observation schedule — source PASS.** The first target is launch-record time plus 420 seconds (`deploy_diagnostic_authorized.py:58`, `:62`). The observer waits for that timestamp and sets subsequent live checks to observation completion plus 240 seconds (`observe_diagnostic_authorized.py:16`, `:35`, `:61`). Successful artifacts are read back with byte comparisons (`:53`–`:57`). The schedule itself was not run.

## Validation limits and disposition

A local stdlib-only AST/data-check command could not start because the default `python` command returned `No pyvenv.cfg file`. No project module ran. Source review was manual; full JSON-log reconciliation and SHA256 verification completed through PowerShell. No AST, compilation, import or runtime PASS is claimed.

No blocking source changes are requested for the exact revision bound in `SOURCE_REVIEW.json`. Keep the new run labeled NOT_EXECUTED until actual completion is collected. Any later result supports only the declared feature-scale and zero-head derivative scope. Historical-spike causality, trained compression efficacy, formal accuracy and the active unmet research gates remain unsupported by this diagnostic.
