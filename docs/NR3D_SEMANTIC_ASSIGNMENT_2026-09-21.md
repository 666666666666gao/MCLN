# Nr3D MCLN semantic assignment experiment

## Scope

User priority is Nr3D on the protected MCLN family. ScanRefer V99 and Sr3D
protected artifacts remain unchanged. EG training is stopped. This experiment
adds no backbone, Query, Gate, inference score, or trainable parameters.

The hypothesis is that some geometrically qualified unmatched Nr3D Queries can
benefit from root text supervision instead of the native no-object target.
PV G is prior evidence, not a demonstrated Nr3D result. Raw-box IoU is a
geometric qualification proxy, not proof of semantic instance identity.

## Exact intervention

At the last layer only, select Queries with raw-box IoU strictly greater than
0.5 against root GT and unmatched to every GT. Restrict this to samples whose
actual `sample_dataset` is `nr3d`; joint `scannet` prompts are untouched even
when `language_dataset` is `nr3d`.

Replace the native no-object target with the native nonnormalized root target:
0.6 positive + 0.2 modifier + 0.2 pronoun + 0.1 relation. Keep target entropy,
eos weight 0.1, and the criterion's actual normalization denominator. Apply
the CE difference inside `loss_pos_align`, so native decoder/contrastive
coefficients are retained rather than copied into a separate auxiliary loss.
Qualification uses detached boxes. All matched Queries, earlier layers,
Hungarian assignments, geometry/Mask losses, and contrastive targets remain
native. No ground truth is introduced at inference.

## Execution order

1. Synthetic CPU test against the actual repository native CE method: explicit
   relabel values and gradients, matched-other-object protection, joint prompt
   protection, strict IoU boundary, empty selection, last-layer-only routing,
   geometry detachment, and removal restoring native behavior.
2. Real-data zero-update probe on 16 previously fixed Nr3D fit rows, protected
   E57, existing bdetr environment and fixed source snapshot. Verify total-loss
   scaling and explicit relabel gradients; record actual selected counts.
   This does not establish optimizer behavior, accuracy, or generalization.
3. After the probe, lock the paired training plan and run a real optimizer
   preflight before full training. Use one fixed seed (2027), the same protected
   start, identical data/order/update budget, fresh optimizers, and unchanged
   native inference. No long baseline reproduction or multi-seed sweep.

E57 is an evaluation-only weight average without optimizer state. New training
must not be described as exact optimizer continuation. The paired update scope
and budget are recorded below; no full training is launched by the zero-update probe.

## Evaluation

Primary task remains complete Nr3D REC on 7899 rows. Preserve historical
4475/3759 (56.6527/47.5883), and report the actual same-source zero-update
result separately. Earlier same-weight source audit 4478/3763 is not a method
gain. Strictly exceeding the project's 59.82/51.38 reference requires at least
4726/4059. Mask has no promotion gate; native Mask computation is retained.

Report same-budget native-versus-replacement results, repairs/breaks at both
thresholds, candidate coverage and eligible-candidate ranks. Do not attribute
the normal adaptation gain to label replacement. Any module holdout from
pretrained training scenes is a development screen, not unseen-scene evidence.

## Locked paired adaptation budget

Both arms start from protected E57 with fresh AdamW and the same native
architecture, trainable parameter set and losses, including the existing source
selector loss weight 0.5. Update all parameters that the native model exposes
as trainable, with uniform LR 1e-5, weight decay 0.0005, clipping 0.1. This is
a common low-LR adaptation protocol, not reproduction of the historical E57
training schedule. No new model parameters are added.

Use all 32919 Nr3D train expressions once, batch 4 (8230 updates per arm),
seed 2027. Disable geometric augmentation and exclude joint detection prompts
in both arms for this Nr-only adaptation. Preserve intermediate-object labels.
Both arms consume the same sampled input tensor per step and the same seeded
model/loss randomness. Differences after updates are part of the paired run.

First execute 2 optimizer steps per arm on 8 previously fixed fit rows, verify
the shared initial training predictions, finite gradients, actual parameter
changes, and saving/reloading model tensors. These preflight weights are not
used for full training. Then measure the actual parent on all 7899 validation
expressions, train one fixed pass from the protected weights, and evaluate both
terminal arms on the same 7899 expressions regardless of which is better.
Keep native selected-source scores and detector support filtering. This is a
development evaluation, not an untouched test. No intermediate best selection.

Recovery checkpoints every 512 updates preserve model/optimizer/step; automatic
resume is not implemented by this initial runner. At the terminal update each
recovery file is refreshed then renamed as its terminal checkpoint, retaining
two arm files rather than four duplicate files. All per-step records are saved.
Checkpoint files use `/root/mcln_nr_semantic_states_20260921_v1`; logs and
receipts remain in the experiment directory on the data disk.

The first optimizer preflight completed two steps in each arm (716 parameter
tensors with finite gradients, nonzero gradient norms, identical first training
predictions) but failed saving its second checkpoint with ENOSPC. It is not a
passed preflight. The data disk had only 1.7 MB available; the system disk had
2.7 GB. The attempted local archive stalled and was not completed or hash verified.
Only this run's two disposable preflight checkpoint files were removed after
retaining the failure log and update records; temporary_cleanup.json records
the exact paths and sizes. The partial local archive is not a valid checkpoint. Historical protected and
EG checkpoint files are untouched. Retry uses a separate `optimizer_preflight_v2`
output, the system-disk checkpoint directory and one reusable preflight file.
The launch checks available space on both disks before running.

## Status

Implementation, synthetic checks and the real 16-fit-row zero-update probe passed.
14 rows contain 69 eligible unmatched Queries. Max explicit relabel CE error
2.3841858e-7, max logit-gradient error 3.7252903e-9; native aggregate scaling
and unchanged model state verified. Nr3D weight is 1/7, unlike ScanRefer's
0.5/7: the wrapper follows the native aggregator automatically. Probe timing
after dataset initialization was 4.0844 seconds, peak allocation 1.956 GB.
Evidence: `refine-logs/nr3d_semantic_assignment_20260921_v1`.

Fresh same-family provisional code review passed for this bounded probe. An
initial synthetic fixture's float64/float32 denominator mismatch was corrected;
the probe draft's ScanRefer coefficient was corrected to Nr3D's 1/7 before
launch. Neither required changing the replacement module.

Two optimizer steps per arm occurred in the first, storage-failed preflight;
they are not a full training result and are discarded before the retry/full
run. No formal accuracy or full-training result exists yet. Complete the
corrected storage/reload preflight before the paired experiment.


Update: optimizer_preflight_v2 passed actual model and optimizer reload after two steps per arm. Full paired entry launched as PID43284; baseline evaluation precedes training. No new formal accuracy exists yet. See handoff section20.270.
