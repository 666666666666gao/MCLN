# FPR-TV A-V4 Counterfactual-Parent Audit Specification

## 1. Purpose

This audit tests one mechanism only: whether at most two GT-free,
training-only counterfactual Parent views increase safe-repair supervision for
the existing Parent-relative text verifier. It does not test formal REC and it
does not authorize a scene-disjoint fold, a 7,899-row evaluation, or long
training.

## 2. Immutable scientific contract

- Dataset: Nr3D, `joint_det + butd_cls`, original V99 source-choice selector.
- Source: the protected full-state E57 checkpoint, exact SHA-256 and internal
  epoch 57, with four optimizer groups and 716 AdamW states.
- Runtime: one GPU/rank, B16 x accumulation 1, exactly 100 microbatches and 100
  optimizer steps in E58.
- Trainable scope: `structured_slot_builder`, `sacr_head`, and
  `parent_relative_text_verifier` only. Box, mask, backbone, source-choice
  selector, and all other parameters remain frozen.
- The counterfactual flag is explicitly enabled. Deployment selection remains
  the actual V99 Parent; counterfactual views are training-only.
- The validation dataset and validation DataLoader must not be constructed.
  The train-only data manifest must exclude `val_v3scans.pkl` and
  `superpoints/val`.
- No validation/evaluation, checkpoint save, retention, LR scan, threshold
  search, or downstream launch is allowed.

The candidate, action, risk, utility, parse-reliability, score-gap, and
fallback constants are the reviewed A-V4 defaults. The launcher exposes no
scientific hyperparameter argument.

## 3. Input and execution integrity

The formal entry must be a reviewed static executor rooted under a root-owned,
non-writable trust path. It must:

1. verify the exact launcher/static-source/static-binary identities from the
   same opened launcher file descriptor;
2. hold the shared GPU0 lock for the complete preflight/formal lifetime;
3. build independent-inode E57, GroupFree, data-manifest, and full runtime-code
   snapshots;
4. bind the data snapshot to the reviewed train-only v2 manifest and reject
   any validation source or inventory row;
5. verify the complete runtime manifest before and after execution;
6. execute from the code snapshot under a clean environment, Landlock write
   allowlist, no-new-privileges, and empty capability sets; and
7. preserve fail-closed process-group supervision for TERM/HUP/INT/KILL parent
   failures.

The formal process may write only its one-shot runtime output/home and the
minimal `/tmp`, `/dev`, and `/proc` roots required by CUDA/PyTorch.

## 4. Required bounded receipt

The only accepted training artifact is one
`train_audit_receipt_epoch_58.json` with:

- schema `mcln-train-loss-epoch-v1`;
- epoch 58, `max_train_batches=100`, `batch_count=100`,
  `optimizer_step_count=100`, and `sample_count=1600`;
- the exact immutable input-snapshot E57 path;
- finite, non-empty `loss_means` and `stat_means`;
- `audit_only=true`, `formal_validation_accessed=false`, and
  `long_training_authorized=false`;
- exact frozen-state and frozen-output-sentinel equality; and
- a changed verifier-trainable state.

No `eval_metrics*.json` and no generated `.pth` may exist outside the immutable
input snapshot.

## 5. Mechanism gates

All required actual-Parent and counterfactual statistics must be finite.
The audit passes the mechanism-density gate only when all of the following are
true:

1. actual and counterfactual selected-score gradient L1 means are both finite
   and strictly positive;
2. the global clipped optimizer gradient norm is finite and strictly positive;
3. actual and counterfactual nonfinite counts are exactly zero;
4. both actual and counterfactual supervised/sample counts are positive;
5. at least one counterfactual view is exercised;
6. the counterfactual positive-row ratio is at least twice the actual
   positive-row ratio, with a strictly positive counterfactual numerator;
7. counterfactual fix-pair and break-pair counts are both strictly positive;
8. frozen state and frozen output sentinel are exact, while the verifier
   trainable partition changes; and
9. no evaluation receipt or generated checkpoint exists.

The decision is durable and always records
`long_training_authorized=false`. A pass only permits an independent review of
whether a future unconsumed scene-disjoint fold should be preregistered.

## 6. Failed first attempt and one-shot recovery

The first formal attempt consumed its original one-shot root and failed on the
first batch after `backward()` but before `optimizer.step()`. The actual Parent
score axis is a non-leaf gathered tensor; the audit read its `.grad` without
retaining that gradient first. The frozen log remains at `0/2806`, and the
original root contains no training receipt, decision, or generated weight.

The zero-step proof must not rely on the initial tqdm line alone. The verifier
must reject any positive `N/2806` counter and must also bind the old source
closure proving the failure is input-independent: verifier-only mode left the
MCLN root in eval mode, the training-only differentiable score-axis branch was
therefore unreachable, the compact gathered score was not retained, and the
unconditional post-backward check preceded every `optimizer.step()`.

The original root must never be deleted, renamed, reused, or completed in
place. A recovery attempt is permitted only when a reviewed verifier proves,
from exact file SHA-256 values and the old runtime closure, that the failure
was zero-step, pre-optimizer, non-evaluating, and weight-free. The recovery
must use a new trust root, launcher identity, experiment name, and one-shot
audit root. It must snapshot the reviewed failure-evidence manifest and verify
the old failure both before and after execution.

The only scientific-code corrections permitted by this recovery are to:

1. activate the MCLN root training semantic required to build the already
   specified actual/CF score axes, while keeping every frozen child module in
   eval mode and only the three allowlisted modules in train mode;
2. create the actual audit score axis as a detached differentiable leaf before
   verifier/loss forward, without unfreezing or modifying the V99 Parent;
3. retain the actual and counterfactual score-axis gradients before backward.

These corrections must not change the A-V4 heads or losses, trainable groups,
data, E57 input, batch count, learning rates, candidate rules, mechanism gates,
deployment/default-off behavior, or permanent prohibition on long training.

## 7. Permanent exclusions

This audit must not introduce or revive:

- fair baseline reproduction;
- the previously rejected Section/Experiment 7;
- the previously rejected Section/Experiment 8 or E0--E7 matrix;
- a proposal-refiner, density-target-box loss, SACR deployment branch,
  relation-CF branch, dataset-id input, Unique/Multiple input, GT-derived
  anchor sidecar, or validation-tuned margin; or
- automatic fold4, 7,899-row formal evaluation, long training, or Sr3D launch.

The hard project target remains strictly above 60.0% on Nr3D. It remains
strictly above 68.9% on Sr3D. This bounded audit cannot itself satisfy either
target.
