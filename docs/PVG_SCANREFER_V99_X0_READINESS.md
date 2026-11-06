# ScanRefer V99 X0 Readiness

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
