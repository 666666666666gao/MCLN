# Clean Baseline Transition

## User Direction

On 2026-10-31, the user paused the previous optimization goal and
requested stopping all three module fine-tuning jobs, removing three
innovation components, and training cleaner baselines starting from
the three dataset-specific pretrained checkpoints.

## Confirmed Stop

At approximately 11:58 America/New_York:

- ScanRefer screen pvg_scanrefer_module_ft_semantic exited; no training
  process or GPU compute process remained; GPU memory 0 MiB.
- NR3D screen pvg_nr3d_module_ft exited following SIGINT; no training
  process remained; GPU memory 0 MiB.
- SR3D screen pvg_sr3d_module_ft exited following SIGINT; no training
  process remained; GPU memory 1 MiB.

The latest complete validations remain ScanRefer25, NR3D9, SR3D4.
Interrupted ScanRefer26, NR3D10 and SR3D5 are not complete results.
Existing logs, saved checkpoints and selected checkpoints were preserved.
This stop was user-requested, not automatic early stopping.

## Architecture Decision Pending

The phrase "remove three innovations" needs its component boundary
confirmed before changing architecture or launching new jobs. The
official PV-Ground architecture and our three added modules are
different sets of components. Our added modules are the source-query
reader, candidate-support corrector, and candidate-span mixer.

Read-only inspection confirmed the author repository origin is
AaNnWwTt/PV-Ground at commit
262e2592589baec7bb83a0d46aae6542d4ccedfb. Its model uses a pv_backbone,
text-guided attention and sampling, and mask-related paths, separately
from our three additions. This is source inventory, not a claimed
mapping of the paper's three contributions.

No clean-baseline model has been modified or launched. After the user
confirms the component boundary, implement only those removals, audit
matching and excluded checkpoint tensors, initialize fresh optimizer
and scheduler state, then verify forward/loss and training updates.
Do not restart the stopped module fine-tuning jobs or run previously
queued geometry diagnostics under the paused objective.
