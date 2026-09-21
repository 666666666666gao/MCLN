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
must not be described as exact optimizer continuation. Detailed update scope
and budget are not yet frozen; no full training is launched by the probe.

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

No optimizer steps, formal accuracy results or full training have occurred.
Next: freeze update scope and budget, run genuine optimizer preflight, then
the paired native/replacement experiment. Existing protected metrics remain.
