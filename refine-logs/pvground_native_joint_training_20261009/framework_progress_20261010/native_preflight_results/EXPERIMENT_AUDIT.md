# Actual native M0 experiment audit

**FAIL — whole_support ran two real native training iterations, then failed its addition-gradient activity assertion.**

Date: 2026-10-10. Fresh reviewer context; requested Astra/max/Codex, actual model/effort/backend attestation **UNATTESTED**. Review class: **same-family / provisional**.

## Actual failure and blocking findings

**B1. Gradient activity failed after real training.** The collected whole_support log records Train [1][1/2] and [1][2/2], then an AssertionError at native_joint_preflight.py:161:

assert step['support_corrector'] > 0 and step['span_output'] > 0

Evidence: native_preflight_failed_attempt1/whole_support.log:14–25. The helper computed gradient_norms but did not print it. The log cannot identify the failed category or step. The preceding core/backbone positivity assertion passed for the same checked iteration; it does not establish positivity across both checked steps because the loop stops at the failure.

**B2. Parameter-delta and full recovery checks were not reached.** The failure occurs before the delta loop at helper line162 and save/reconstruction/load at lines176–217. The serial controller aborts at line40 before extremal_support (collected controller.log:2–4). Both successful arm receipts and native_preflight_complete/INTAKE.json are absent. The failed intake records no other file metadata, including no checkpoint binary. The failed-run intake is actual failure evidence, not a successful complete intake.

This is an **actual failed M0**, separate from the earlier upstream archive attempt saved as PRE_M0_UNAVAILABLE_ATTEMPT1.

## Evidence established by the real run

All 11 collected file byte counts and SHA256 values match native_preflight_failed_attempt1/INTAKE.json. Collection exit is0 and stderr empty. The collected preflight helper, controller and protocol exactly match reviewed R5 bytes.

With that source binding, reaching the two trainer records establishes execution past the real-loader/model assertions: 36,665 training rows, 9,508 validation rows, 4,583 normal batches, two B8 ScanRefer batches and an initial 1,295-tensor model (collected helper:61–77). This is assertion/control-flow evidence, not an independent remote dataset recount. RoBERTa freezing and native optimizer membership checks also passed before training. The observed-loss wrapper twice checks native training and differentiable deployed last boxes/scores before calling the ordinary criterion (helper:124–134; source/models/losses.py:899–917).

Both trainer records occur after loss.backward, clip_grad_norm_, optimizer.step and scheduler.step (source/main_utils.py:460–482). Thus two optimizer.step calls happened. Required nonzero parameter changes were **not verified**: their later probe was not reached. The result is not a successful both-arm M0 or formal training.

## Training numbers and clipping diagnosis

The first printed total loss is 14.5394. The second printed 14.9367 is the **two-batch cumulative mean**, because the native logger divides accumulated statistics by batch_idx+1 (main_utils.py:473–483). It is not the second batch's raw loss; the rounded values imply approximately 15.3340 for that batch, not an exact raw measurement.

The probe's gradient hooks execute during loss.backward and append values **before clipping** (helper:114–120; main_utils.py:461–470). Therefore clipping cannot directly zero the already-recorded current-step hook values. It can affect the optimizer update and later gradients. The configured clip_norm is 0.1, but the returned global norm and post-clip per-group norms are not logged. Finite loss and finite gradient elements do not establish a finite accumulated norm or positive activity in every group.

There is a concrete path to investigate: span output weights/bias are zero initialized, then raw_gate is hard-clamped to [0,1] (source/extremal_span_mixer.py:25–27,97–102). An inactive clamped branch is a source-grounded candidate. Actual raw gates and the failed group/step are missing, so **gate saturation and clipping are not established causes**.

Read-only repair recommendations:

1. Emit existing per-step gradient_norms/loss_records and groupwise parameter deltas before the failing assertions. Preserve that diagnostic even when the assertion fails.
2. Record the native returned grad_total_norm, per-category gradients before/after the existing clip, and each step's raw span gate range and <=0/>=1 counts. Keep B8, seed2027, learning rates, clipping and objectives unchanged during diagnosis.
3. Do not remove the assertion, disable clipping or change the gate just to pass. If measurements establish a legitimately inactive clamped second batch while the first has nonzero activity and the two-step group delta is nonzero, assess activity across the two-batch probe instead of demanding positive activity in every batch. That is a conditional acceptance-design question, not a confirmed fix.
4. Apply a numerical/dataflow repair only after the diagnostic identifies the actual cause. No source was edited or model rerun by this reviewer.

## Integrity checks

| Check | Status | Evidence and limit |
|---|---|---|
| Ground-truth provenance | PASS, whole_support | The exact deployed-hash loader reads ScanRefer annotations and scan instance boxes/points (runtime_binding/dataset_source/src/joint_det_dataset.py:594–597,1101–1106); the bound actual helper reaches two real batches. This is not a full data-byte audit. |
| Score normalization | PASS | formal_accuracy stays null. Native prediction-derived corresponding-mask auxiliaries are training consistency targets, not evaluation GT. |
| Result existence | FAIL | Real failed logs/intake are verified; successful both-arm receipts and complete-success intake are absent. |
| Executed paths | WARN | Ordinary trainer runs twice; parameter-delta/save/cold-recovery code is after the failing assertion and not reached. |
| Scope | PASS | Actual scope is a failed whole_support two-batch engineering check; extremal_support is not started. No full epoch or accuracy claim. |
| Evaluation classification | real_gt | Training inputs/targets; engineering preflight, not an accuracy evaluation. |

## Recovery, ownership and claim limits

No actual model+AdamW+scheduler+four-RNG save/cold-load equality is established; the run never reaches that stage. The reviewer did not reopen a checkpoint binary. Future direct receipt execution evidence must be distinguished from independent binary reopening. State equality would not prove next-step equivalence or mid-epoch data-loader determinism; aggregate changes would not prove every parameter or three useful contributions.

The reviewer independently verified collector 53060 creation/argv, held the same Windows handle from 01:03:24.905991 and observed it signaled at 01:23:24.315263 with exit 1. The first yielded-wait check occurred at 01:24:21, after the scheduled first observation, without intervening remote/NN queries. Training records are timestamped 01:08:01 and 01:08:06; the later terminal observation is not the actual failure timestamp. Launch/admission and the collected serial controller bind one-rank GPU0 configuration and prior-run closure, not proof of global GPU exclusivity throughout the run.

The source-only R5 identities still match; the local dataset-source scheduler is a different hash from the retained PV scheduler and is not substituted for it. The ordinary future controller omits --checkpoint_path, but no later formal execution or checkpoint retirement is established.

Actual M0 acceptance, formal-training admission and checkpoint retirement are **withheld**. The actual failed intake, logs, wait and helper files are bound with resolved D: paths and bare SHA256. Missing success-intake/receipt paths receive no placeholder hash.

The reviewer performed no SSH/GPU/model/NN/training/credential operations, changed no original input and did not publish or promote anything. Raw native tools, immutable input snapshots, request/steering, report and seal are preserved under native_preflight_results.
