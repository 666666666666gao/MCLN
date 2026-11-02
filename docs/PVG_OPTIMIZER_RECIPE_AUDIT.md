# Saved Optimizer And Scheduler Audit

Recorded2026-11-02, remote CPU only; no local tests or model forward.
The helper `tools/audit_mcln_optimizer_recipe.sh` loaded fixed preserved
checkpoints and checked their own epochs,optimizer states and scheduler.
All three receipts report CUDA uninitialized.

| Dataset/checkpoint | Steps per epoch | Optimizer state step | Scheduler last_epoch | Saved milestones |
| --- | ---: | ---: | ---: | --- |
| ScanRefer9 | 6081 | 54729 | 54729 | 66891,188511 |
| NR3D22 | 936 | 20592 | 20592 | 141336 |
| SR3D8 | 3243 | 25944 | 25944 | 100533,132963 |

Every optimizer state carrying a step agrees within each checkpoint.
Steps equal checkpoint epoch times optimizer steps per epoch. The current
trainer constructs the scheduler using accumulation-aware optimizer steps,
and calls scheduler.step immediately after optimizer.step. This supports
correct accumulation scaling for the inspected saved states.

Nonempty optimizer groups retain the configured rates:
Scan1e-5/backbone2e-5,NR/SR1e-4/backbone1e-3. NR/SR have a third group
with LR1e-5 but zero parameters; that group is not evidence of an incorrect
task learning rate. Inspect parameter counts before interpreting it.

## Inherited Decay Convention

Inspected primary source:
`qzp2018/MCLN`,commit `9744a4ed219062d448ed0dba587eeb864491f158`,
`utils/lr_scheduler.py:get_scheduler` and `main_utils.py` training loop.
Official repository: https://github.com/qzp2018/MCLN .
Source was read,not executed.

Both the official and current scheduler use
`(decay_epoch - warmup_epoch) * steps_per_epoch`. The official loop starts
at epoch1 and steps the scheduler after each optimizer update. With
warmup_epoch=-1,the saved milestones land at the end of epochs11/31
for Scan,151 for NR,and31/41 for SR. The following epoch's updates use
the decayed rate; the last update at the milestone uses the previous rate.
This is one epoch later than a literal decay10/30,etc interpretation.
The NR current config was independently read and confirms -1/start1.
For Scan/SR the measured milestone arithmetic matches the same formula;
the three receipts do not separately export their warmup config field.

This is an inherited recipe convention,not evidence of a new regression
or the cause of the accuracy gap. No running scheduler,source,optimizer
or configuration was changed. Do not silently relabel the runs as having
already decayed at the configured epoch number.

Raw receipts under `docs/results/mcln_pvground_20261101/`:

- `scanrefer-e9-optimizer-recipe.json`
- `nr3d-e22-optimizer-recipe.json`
- `sr3d-e8-optimizer-recipe.json`

This audit does not measure real Reader residuals,Support corrections,
Span gates,or matched bypass metrics. It cannot establish module benefit,
a convergence diagnosis or a performance gain. Those require separate
remote model observations and complete same-checkpoint validation.
