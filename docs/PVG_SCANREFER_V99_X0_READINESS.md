# ScanRefer V99 X0 Readiness

## Current State: 2026-11-06 12:42 UTC

The12:42UTC current-only read-back found the sole controller still live,
GPU idle, and no preflight completion, optimizer-update witness, or A/B
completion.
The first bounded preflight had failed before an optimizer update because
the isolated archive omitted `models.pvground_span_supervision.py`; the
exact repository-owned helper has since been restored and SHA-registered.
The live controller remains the only process authorized to retry it.

Capacity remains below the recorded reserve: 3,978,037,815 bytes required
versus 821,837,824 free (short3,156,199,991 bytes). The last persisted
queue-state receipt is timestamped08:38UTC and says
`requires_storage_capacity`; its age means it is not evidence of a newer
queue transition. Do not stop/duplicate the controller, modify its recipe,
or manually delete protected files. The paired official/Reader-free E0
acceptance result remains5435/4327, not a training score.

## Previous State: 2026-11-06 11:32 UTC

The sole controller remained live and GPU-idle without preflight/update or
A/B completion. Capacity was 821,846,016 bytes free against the3,978,037,815
reserve. The existing controller owns any retry; protected files remain.

## Previous State: 2026-11-06 10:41 UTC

The sole sequential controller remains live. Its latest read-back has no
preflight completion, optimizer-update witness, or Stage A/B completion.
The prior isolated-source import failure was repaired by adding the
repository-owned Span supervision helper and registering its SHA; the
controller, not a manual retry, owns continuation.

Storage is still the active gate: 3,978,037,815 bytes required versus
821,854,208 free at10:41UTC, a deficit of3,156,183,607 bytes. GPU was
idle. Do not duplicate or stop the controller, modify its recipe, or
delete protected inputs/checkpoints. Paired E0 remains5435/4327 for both
official and Reader-free identity on9508; it is an acceptance check, not
a training metric.

## Previous State: 2026-11-06 09:36 UTC

The legacy predecessor completed all8 epochs. Its full9508 semantic
epoch8 result is5380/4450; trial best epoch2=5398/4512. The immutable
B2 comparator remains5483/4656. The epoch8 checkpoint passed the
08:02UTC CUDA-off CPU audit: own epoch8,1168 model tensors, nonempty
optimizer/scheduler and SHA
`6d6d6ca609867d835cc68573166740ac02b7d904a39aa1f71517a21ffcd72a74`.
This legacy rank-loss result is not V99 QILA/X1.

The isolated X0 step-zero process passed the paired full9508 E0 gate:
official and Reader-free identity both scored5435/4327. Point-order,
identity, official54 checkpoint SHA, seed0 and backend matched; identity
added24 declared tensors. The controller then began the bounded
12-microbatch/two-step preflight. At08:41:59UTC its process was live in
dataset/text-decoupling setup; no first-update witness or preflight
completion receipt was confirmed. This is preflight, not Stage A/B training.
The sequential Stage A/B controller remains alive, but its storage gate
requires3,978,037,815 free bytes; the08:56UTC read observed821,882,880.
The documented capacity expansion remains necessary for formal training.
Do not launch a duplicate process or manually delete protected checkpoints.

The bounded preflight attempt at08:50UTC failed before any optimizer
update with the confirmed packaging error
`ModuleNotFoundError: models.pvground_span_supervision`. The loss module
imports this repository-owned helper when the configured Span gate loss
is nonzero, but the isolated Stage B overlay archive omitted it. The
preflight run directory contains only config/log/tensorboard; no training
audit or update witness exists. The exact existing helper was copied into
the current isolated source and its SHA
`330e0bb03fe8fd5c9f0b485410af522d92909ecf51a29d7db4b9e80d48b95abd`
was added to the preparation manifest at08:59:54UTC. The X0 preparation
script now also includes that module in its required source/hash list.
The original queue controller was live and waiting; the repair did not
start a retry or alter any live trainer. Let the existing controller
retry its bounded preflight after the slot becomes available, then require
the actual first/second update witnesses and completion receipt.

Failure and repair evidence:
`results/mcln_pvground_20261101/scanrefer-x0-preflight-module-failure-repair-20261106T0859.json`.

## Previous State: 2026-11-06 08:42 UTC

The paired E0 identity gate passed; the first bounded preflight process
was still loading data and had no first-update witness at observation.
Formal A/B training remained storage-gated.

## Previous State: 2026-11-06 08:33 UTC

Official E0 had completed at5435/4327; Reader-free identity was592/1189
and pending.

## Previous State: 2026-11-06 08:25 UTC

The official E0 evaluation was at799/1189 batches; no result or X0
optimizer update was yet recorded. Storage remained below the Stage A/B
queue reserve.

## Prepared and Queued: 2026-11-06 07:26 UTC

The isolated trial is now prepared at
`/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_v99_x0_20261106`.
Its source is an independent 25MB copy of the reconciled deployed Stage B
source, with only the X0 controls and dedicated tools overlaid. The live
predecessor source remains unchanged.

Remote CPU checks now also constructed the actual MCLN model and ran the
actual official model-only loader: all1134 native tensors matched
official54 exactly; total1158 tensors,24 trainable Support/Span parameter
tensors, Reader absent, fresh optimizers and BN eval in both phases.
Actual parser checks passed the preflight/A/B contracts. No GPU forward,
actual optimizer step, formal validation or X0 training has run yet.
Receipts: `scanrefer-x0-preparation-20261106T0716.json` and
`scanrefer-x0-cpu-model-preflight-20261106T0720.json`.

The single sequential controller is live in screen
`pvg_scanrefer_v99_x0_20261106`, screen PID344492, launched07:23:57UTC.
It polls every300 seconds. It requires:

1. Registered predecessor PID328001 gone and its authoritative receipt
   proving full8 epochs completed; canonical history must contain all8
   full9508 rows.
2. An idle GPU and adequate capacity; no concurrent GPU trainer.
3. Predecessor own-CPU best/latest audit, then paired full9508 official
   e54 and Reader-free step-zero evaluations. The frozen published e54
   reference is5435/4327; X0 identity must match it exactly, along with
   checkpoint, point/order hashes, seed and backend. This is distinct
   from the B2 global-best comparator5483/4656 and from the historical
   same-checkpoint B2 Reader-bypass inference5504/4673.
4. A12-microbatch/two-optimizer-step preflight, then fresh official54
   initialization for Stage A3 and model-only A3 initialization for B8.
   Both stages have no early stopping. Rank loss is zero in X0.

The actual queue read-back at07:24UTC confirmed its screen process live,
reason `awaiting_predecessor_natural_full_budget_completion` and no X0
training started. Do not launch another controller. Narrow check:
`tools/check_scanrefer_x0_preparation.sh`.

## Capacity Action Required

At07:49UTC the Scan host's50GiB filesystem had820,989,952 bytes free
while the predecessor was completing epoch8 validation. This is a
transient in-run observation, not the post-completion reserve.
At07:05UTC the filesystem had1,643,368,448 bytes free;
current best/latest checkpoints were795,607,563 bytes each. After source
preparation, free space was1,617,629,184 bytes. The queue reserves
3,978,037,815 bytes for two retained warmup checkpoints plus joint
best/latest and a new save, conservatively bounded using that measured
joint checkpoint footprint. Preparation was small; the active run still
continues.

Before X0 can enter GPU acceptance/training, provide at least3GB more
free space after the predecessor retention transition;5GB expansion is
recommended. The controller will proceed automatically
once its predecessor, GPU and capacity conditions pass. No manual cleanup,
extra retention watcher or deletion of protected historical runs/inputs
was performed. Capacity waiting is not a current training failure or a
claim that target accuracy is achieved.

Capacity evidence:
`results/mcln_pvground_20261101/scanrefer-current-storage-capacity-20261106T0706.json`.
Queue evidence: `scanrefer-x0-queue-started-20261106T0722.json` and
`scanrefer-x0-queue-readback-20261106T0722.json` in the same directory.

The plan's E0 identity is the official epoch54 full9508 reference
5435/4327, not the B2 best5483/4656. Historical same-checkpoint B2
Reader-bypass inference is5504/4673 and was not a trained ablation. The
queue's E0 assertion now requires both official epoch54 hits to equal
5435/4327 and identity hits to match exactly. This resolves an ambiguity
without describing a function-preservation result as achieved before the
GPU E0 runs.

## Verified Source Reconciliation: 2026-11-06

The approved wrapper read and SHA256-hashed the current trial runner and
its deployed `stage_b_source` at 06:54:40 UTC. All four files matched the
local checkout before this patch. Evidence:
`results/mcln_pvground_20261101/scanrefer-v99-source-readiness-20261106T0655.json`.

Confirmed gaps against the user's November 6 plan:

1. `use_pvground_innovations` implicitly attaches Reader, so simply omitting
   `use_candidate_local_visual` does not create the intended X0.
2. Joint training calls `model.train()` and had no explicit BatchNorm
   freeze control. Module-only training already uses the frozen native
   model in eval mode.
3. The current runner explicitly sets legacy BBF rank-loss weight 0.3.
   It does not implement the plan's instance-aware QILA objective.
4. Span gate supervision exists but defaults to zero. X0 must explicitly
   request `--pvground_span_gate_loss_weight 0.1`.

The live trial remains untouched. At 06:51 UTC its epoch8 was at
2011/6081, with finite first-update loss and Reader/Support/Span updates
10/10/14. Complete epoch7 semantic hits were 5362/4418 over9508; trial
best2=5398/4512 and global B2=5483/4656 remain distinct.

The current runner SHA is
`58bc4cacc4d4a2468307796cdade5cb71c9db8c41413ef10dab2156e183e464d`.
Its recovery source now uses `epoch == 1` when checking initialization
metadata. Historical traceback text using a different predicate is not
proof of a current live failure. This source read does not test recovery
after authorized checkpoint retention; do not claim general resume
correctness from it.

## Implemented Controls

All new options default off, preserving current NR/SR and historical
ScanRefer behavior:

- `--disable_pvground_reader`: require innovations, suppress Reader
  construction, retain Support/Span, skip Reader train-mode activation
  and first-update probes. Explicit Reader enablement conflicts with
  removal and is rejected.
- Reader-free checkpoint validation permits only the existing declared
  new Support/Span keys during model-only official initialization.
  Exact Reader-free resume remains strict; a checkpoint containing
  Reader keys is rejected rather than silently dropping them.
- `--freeze_batchnorm`: apply eval mode to all BatchNorm subclasses
  after the existing training-mode selection, including SyncBatchNorm.
  Affine parameters retain their existing trainability; this control
  freezes running statistics and batch-statistics behavior.

These changes were uploaded only to the isolated controls directory:
`/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_v99_x0_controls_20261106`.
Neither the active trial's source nor its runner was replaced.

## Remote Verification and Limits

`tools/check_scanrefer_x0_controls.py` ran remotely with GPUs hidden,
through `tools/verify_scanrefer_x0_controls.sh` and the approved wrapper.
It compiles the three changed source files and executes only the actual
extracted mode/attachment methods and checkpoint-key validator.

Passed:

- Default Reader attachment and Reader-free Support/Span attachment.
- Frozen joint-training BN buffers with live affine/native gradients.
- Module-only mode without Reader and with BN kept in eval.
- Official-to-Reader-free model-only key validation and exact resume keys.
- Rejection of unexpected Reader keys; legacy checkpoint validation.

Receipt:
`results/mcln_pvground_20261101/scanrefer-x0-controls-verification-20261106T0700.json`.
This is not a complete MCLN construction, checkpoint load, dataset test,
forward pass, optimizer-update witness, or full9508 evaluation. There is
no new accuracy result or demonstrated gain from these code changes.

## Next Sequential Experiment

Let the current eight-epoch run finish naturally and collect its complete
history, own-epoch CPU checkpoint audit, retention and terminal receipts.
Then prepare X0 in a separate source directory from the reconciled Stage B
source and these controls. Do not run a second GPU job concurrently.

X0 follows the plan: official54 model-only initialization, fresh optimizer,
three-epoch module warmup and then low-LR joint training, Reader removed,
Support/Span retained, Span gate loss0.1, BN frozen throughout, effective
batch48. Preserve the immutable B2 comparator.

Before the new training launch, prove complete checkpoint-key accounting,
step-zero full9508 protocol identity, actual Support/Span optimizer
updates, frozen native weights in Stage A and frozen BN buffers in both
stages. Archive exact configs, source hashes and evidence. Respect the
user's no-stop instruction; no new implicit early-stop policy.

True X1 requires the plan's instance-aware QILA implementation and its
separate objective/gradient/data-isolation checks. The old rank-loss trial
must not be relabeled as X1. X2/RCR follows only after the documented X1
comparison gate passes. NR/SR official scratch recipes remain unchanged.
