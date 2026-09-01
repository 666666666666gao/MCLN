# FPR-TV A-V4 Scene-Disjoint Fold-4 Specification

## 1. Purpose and authorization boundary

This is the only preregistered scene-disjoint generalization audit authorized
by the successful A-V4 Counterfactual-Parent 100-microbatch mechanism audit.
It tests whether the fixed A-V4 training correction improves deployed
Parent-relative switching on one previously unconsumed Nr3D train-scene fold.

It is not a baseline reproduction, a validation run, a formal 7,899-row
evaluation, a full-fit replay, a long-training experiment, or an Sr3D launch.
Passing this fold only permits an independent review of a later short-training
proposal. It never authorizes that proposal automatically.

## 2. Immutable history and one-shot identity

Before preflight or execution, the launcher must validate from exact immutable
JSON bytes and SHA-256 values that:

1. FPR-TV v1 fold 0 was consumed and failed its fixed gate;
2. FPR-TV v2 fold 1 was consumed and failed by complete abstention;
3. fold 2 was consumed by the sealed negative Density-Aware Target Box audit;
4. FPR-TV v3 fold 3 was consumed and failed with 844 switches,
   `fix/break=23/38` at REC@0.25 and `104/283` at REC@0.50;
5. the unique A-V4 recovery audit completed exactly 100 optimizer steps and
   1,600 samples, passed all fixed mechanism-density checks, accessed no
   validation data, saved no weight, and retained
   `long_training_authorized=false`; and
6. every earlier fold-4 root and this experiment's one-shot root are absent.

Only fold number 4 is accepted. The root is consumed atomically before any
formal training process starts and is never deleted, renamed, reused, or
completed in place after failure.

## 3. Frozen scientific contract

- Dataset: Nr3D train scenes only, `joint_det + butd_cls`, original V99
  source-choice selector, no official validation/test rows.
- Split: immutable
  `int(sha256(scan_id)[:8], 16) % 5`; exactly 511 source train scenes and
  32,919 grounding rows.
- Fold 4: exactly 417 fit scenes / 27,004 fit rows and 94 held-out scenes /
  5,915 held-out rows, with zero scene overlap.
- Source: the same protected full-state E57 used by the successful A-V4
  mechanism audit, SHA-256
  `fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655`,
  internal epoch 57.
- Training: E58 only, one GPU/rank, B16 x accumulation 1, complete fit epoch,
  natural final batch of 12, exactly 1,688 optimizer steps, no dropped row.
- Trainable scope: `structured_slot_builder`, `sacr_head`, and
  `parent_relative_text_verifier` only. Every other model tensor is frozen.
- A-V4 Counterfactual-Parent training is explicitly enabled. At most two
  GT-free training-only counterfactual Parent views may contribute loss.
  Held-out deployment uses only the actual V99 Parent and the fixed verifier.
- Candidate Top-K, score-gap feasibility, structured reliability, action
  margin, absolute break veto, optimizer, scheduler, LR, loss weights, seed,
  and every other behavior-affecting argument are fixed by one preregistered
  canonical configuration SHA. No fold-specific override is accepted.
- Density target-box, proposal refinement, relation-CF, SACR deployment,
  hard replay, raw-parser cache, dataset-ID input, Unique/Multiple input, and
  GT-derived anchor sidecars are disabled.

Every fit source-row identity must be consumed exactly once. Count, unique
count, and the order-independent identity SHA must match the frozen fit set.

## 4. Runtime and provenance contract

The formal entry must use a separately reviewed static executor installed
under a root-owned, non-writable trust-root ancestry. It must bind the exact
launcher, static source, static ELF, shared GPU0 lock, protected E57,
GroupFree checkpoint, complete runtime-code manifest, and the five immutable
history artifacts.

The launcher must:

1. reject ambient loader, shell-function, Python-home, and user-site inputs;
2. verify the complete reviewed runtime closure by manifest before copying;
3. copy code, E57, GroupFree, and history evidence to independent inodes;
4. verify exact file set, size, mode, and SHA before and after execution;
5. execute from the code snapshot under a clean environment, Landlock write
   allowlist, no-new-privileges, empty capability sets, and closed inherited
   descriptors;
6. hold the common GPU0 lock for its entire lifetime; and
7. preserve fail-closed process-group supervision across normal exit,
   TERM/HUP/INT, and parent failure.

Only the isolated runtime output/home and the minimum CUDA/PyTorch temporary
roots may be writable. The protected inputs and official monitor roots are
read-only and outside the experiment output tree.

The short no-save audit requires at least 5 GiB free on the output/data
filesystem. Its three independent immutable copies (E57, GroupFree, and the
reviewed code closure) total approximately 0.893 GiB; the remaining capacity
is reserved for the runtime environment, logs, receipts, and fail-closed
headroom. This is a fixed preflight safety gate, not a tunable experiment
parameter.

## 5. Required receipt and advancement gate

The only accepted scientific artifacts are one fold-4 scene-audit receipt,
one eval-metrics receipt for the 5,915 held-out train rows, and one durable
wrapper decision. No generated `.pth` is allowed.

The receipt must prove:

- internal source epoch/SHA `57/fe1e...f6655`;
- exact fold/split counts and zero scene overlap;
- 27,004 fit samples, 27,004 unique source-row identities, exact identity SHA,
  1,688 microbatches and optimizer steps, including the natural 12-row tail;
- exactly 5,915 held-out evaluations on train scenes only;
- unchanged frozen tensors and frozen Box/Mask/V99 Parent sentinels;
- changed A-V4 trainable tensors; and
- exact integer parent/selected/fix/break/kept-correct/kept-wrong partitions at
  REC@0.25 and REC@0.50.

The fold gate passes only when all integrity checks pass and:

1. at least one deployed switch occurs;
2. REC@0.25 `fix_count > break_count`; and
3. REC@0.50 `fix_count >= break_count`.

No threshold, Top-K, margin, loss, LR, scheduler, epoch count, or decision
rule may be changed after observing this fold. Failure seals A-V4. Success is
evidence for independent review only. The decision always records
`audit_only=true`, `formal_validation_accessed=false`, and
`long_training_authorized=false`.

## 6. Permanent exclusions and project targets

This work must not introduce or revive fair baseline reproduction, the
previously rejected Section/Experiment 7, the rejected Section/Experiment 8,
or the E0--E7 matrix. It must not automatically run fold 4 again, official
Nr3D, Sr3D, full-fit, long training, or any downstream experiment.

The formal project targets remain strictly above 60.0% on Nr3D
(`>=4740/7899`) and strictly above 68.9% on Sr3D (`>=12214/17726`). This
train-scene fold cannot itself satisfy either formal target.
