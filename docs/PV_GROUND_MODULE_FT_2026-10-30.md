# PV-Ground Module Fine-Tuning: ScanRefer, NR3D, and SR3D

Date: 2026-10-30 (America/New_York)

## Targets and Protocol

| Dataset | Target |
| --- | ---: |
| ScanRefer Acc@0.25 | 60.0% |
| ScanRefer Acc@0.50 | 52.0% |
| NR3D Acc@0.25 | 59.8% |
| SR3D Acc@0.25 | 69.4% |

All values below are native semantic-alignment `last/bbf` top-1 accuracies. A completed row is accepted only when the evaluation covers the full validation set and reports `same_complete_model=true`. Epoch 0 is the untrained initialization and is shown for reference, never as a fine-tuning result.

The runs use batch size 10 and seed 111, with `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and LR decay epochs `[10,30]`. The pretrained core is frozen; the three PV-Ground modules are trained. NR3D and SR3D use their respective paper-repository dataset flags, including `--butd_cls` and `--self_attend`. The active runs are monitored independently; their metric definitions are not mixed.

## ScanRefer

Validation size: 9,508.

| Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 5606 | 4843 | 58.961% | 50.936% |
| 1 | 5559 | 4542 | 58.467% | 47.770% |
| 2 | 5552 | 4538 | 58.393% | 47.728% |
| 3 | 5619 | 4641 | 59.098% | 48.812% |
| 4 | 5580 | 4581 | 58.687% | 48.180% |
| 5 | 5598 | 4600 | 58.877% | 48.380% |
| 6 | 5597 | 4629 | 58.866% | 48.685% |
| 7 | 5623 | 4716 | 59.140% | 49.600% |
| 8 | 5613 | 4720 | 59.034% | 49.642% |
| 9 | 5608 | 4739 | 58.982% | 49.842% |
| 10 | 5617 | 4776 | 59.077% | 50.231% |

Best trained evaluation by Acc@0.50 is epoch 10. Relative to the recorded baseline (59.518% / 46.992%), epoch 10 is -0.441 / +3.239 percentage points. Target minima are 5705/9508 at Acc@0.25 and 4945/9508 at Acc@0.50, leaving gaps of 88 and 169 hits. Acc@0.25 remains below baseline and target; a localization/box-precision bottleneck is plausible but unconfirmed.

Epoch 10's full checkpoint was verified against its native metric row and retained as `best_trained_epoch10.pth`; epoch 11 is training.

## NR3D

Validation size: 7,899.

| Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 4739 | 4132 | 59.995% | 52.310% |
| 1 | 4786 | 3918 | 60.590% | 49.601% |
| 2 | 4780 | 3909 | 60.514% | 49.487% |

Epoch 2 is +0.519/-2.823 points versus initialization at Acc@0.25/Acc@0.50 and remains 0.714 points above the 59.8% Acc@0.25 target. It is below epoch 1 by 0.076/0.114 points at the two thresholds. Epoch 1 remains the best trained checkpoint by the Acc@0.50-first rule; its checkpoint is preserved as `best_trained_epoch1.pth`. The minimum target count is 4,724 hits; epoch 2 exceeds it by 56.


## SR3D

Validation size: 17,726.

| Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 11620 | 10023 | 65.553% | 56.544% |
| 1 | 11944 | 10166 | 67.381% | 57.351% |

Epoch 1 is +1.829/+0.807 points versus initialization at Acc@0.25/Acc@0.50, and 2.019 points below the 69.4% Acc@0.25 target. The minimum target count is 12,302; epoch 1 is short by 358 hits. Its verified full checkpoint is preserved as `best_trained_epoch1.pth`; epoch 2 is training.

## Verified Operational Issue

The ScanRefer checkpoint audit showed that `best.pth` retention includes the epoch-0 initialization. This left ScanRefer's selected `best.pth` at epoch 0 even after trained rounds had become the best fine-tuned evaluations. `latest.pth` is overwritten as training advances, and in the audited ScanRefer run its retained-metric field described the historical retained best rather than the latest checkpoint's own evaluation.

For ScanRefer epochs 8 and 9 and NR3D epoch 1, the full checkpoint epoch was matched to `native_metrics.jsonl` and the model, optimizer, scheduler, and RNG payload was checked before a same-filesystem hard link was created to preserve the trained checkpoint. This is an operational preservation measure; it does not modify the training process or weights. Future result reporting must select among trained epochs separately from initialization and join each checkpoint to the native metric row for its own epoch.

## Three Module Roles and Testable Hypotheses

The deployed model contains three modules whose intended roles are:

1. **Task-observation query reader:** separate semantic and geometry query transforms read shared observed memories, then use the decoder's shared attention parameters.
2. **Candidate mask-support corrector:** adds candidate-specific residuals to superpoint mask logits from query/support features and current mask state.
3. **Extremal span mixer:** aggregates per-axis face evidence from predicted mask support and mixes mask-derived and native box centers/sizes with bounded axis-wise coefficients.

These are implementation roles, not novelty claims. The current metric pattern motivates, but does not yet prove, the following tests:

- **ScanRefer:** separate Unique/Multiple performance and inspect whether semantic candidate selection is correct while final box IoU misses 0.50. Compare predicted-mask and native-box IoU and inspect per-axis span-mixer gate distributions before changing the corrector or mixer.
- **NR3D:** track whether the first-epoch Acc@0.25 gain and Acc@0.50 drop persist; distinguish candidate selection errors from box-extent calibration errors before dataset-specific internal-parameter tuning.
- **SR3D:** obtain trained-round metrics before attributing its initialization gap to any module.
- Across all datasets, use controlled ablations or internal-parameter sweeps while keeping the overall network and three-module boundaries fixed. Prefer a dataset-specific internal setting only when the corresponding dataset diagnostics support it; do not infer cross-dataset benefit from one dataset.

Potential ideas to study, without copying implementations or comparing incompatible headline metrics:

- [ConcreteNet](https://github.com/ouenal/concretenet): inter-instance attentive fusion and contrastive grounding; its Unique/Multiple split motivates subgroup diagnostics. The repository has a non-commercial-use license, so code reuse requires a separate licensing decision.
- [3D-SPS](https://github.com/fjhzhixi/3D-SPS): language-guided coarse selection followed by target-oriented progressive mining; a conceptual reference for testing staged evidence use in the query-reader path. MIT licensed.
- [3DRefTR](https://github.com/Leon1207/3DRefTR): a unified REC/RES model with superpoint mask prediction, relevant to mask-support diagnostics. Its reported REC metric is not interchangeable with this experiment's native Acc@IoU.

No architectural or module change is justified by the current aggregate metrics alone. First complete and audit the next full validation rows, then use component-level and subset-level evidence to choose a narrow experiment.

## Checkpoint and Monitoring References

- ScanRefer run root: `/root/autodl-tmp/pvground_scanrefer_module_ft_20261030`; native metrics: `logs_semantic/scanrefer/scanrefer_module_ft_semantic/1793345477/native_metrics.jsonl`.
- NR3D run root: `/root/autodl-tmp/pvground_nr3d_module_training_20261030`; native metrics: `logs/nr3d/nr3d_module_ft/1793382384/native_metrics.jsonl`.
- SR3D run root: `/root/autodl-tmp/pvground_sr3d_module_training_20261030`; native metrics: `logs/sr3d/sr3d_module_ft/1793380711/native_metrics.jsonl`.

This record is a snapshot, not a final result. Append later full evaluations with exact row counts, metric provenance, checkpoint epoch, and target deltas.

## Follow-up status: 15:35 America/New_York

No new complete native validation rows were present. ScanRefer epoch 10 was at 2,470/4,865 training batches, NR3D epoch 2 at 1,985/4,490, and SR3D epoch 1 at 6,308/7,783. All three screens and GPUs were active and the checked log tails showed no errors. The latest complete evaluations remain ScanRefer epoch 9, NR3D epoch 1, and SR3D initialization epoch 0. No run parameters or processes were changed.

## Follow-up status: 15:40 America/New_York

No new complete native validation rows were present. ScanRefer epoch 10 was at 2,906/4,865 training batches, NR3D epoch 2 at 2,425/4,490, and SR3D epoch 1 at 6,757/7,783. All three screens and RTX 4090 GPUs remained active; checked logs contained no errors. The latest full-set `last/bbf`, `same_complete_model=true` evaluations remain ScanRefer epoch 9 (5,608/9,508 at IoU 0.25; 4,739/9,508 at IoU 0.50), NR3D epoch 1 (4,786/7,899; 3,918/7,899), and SR3D initialization epoch 0 (11,620/17,726; 10,023/17,726). Commands retain `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and decay epochs `[10,30]`. No run settings or processes were changed.

## Follow-up status: 15:44 America/New_York

No new full evaluation row had appeared. ScanRefer epoch 10 was at 3,197/4,865 batches, NR3D epoch 2 at 2,725/4,490, and SR3D epoch 1 at 7,047/7,783. All three screens and GPUs remained active; checked log tails showed no errors. Metric histories and checkpoint state remain unchanged from the 15:40 audit.

## Follow-up status: 15:55 America/New_York

SR3D epoch 1 training had completed and its native validation was underway at 122/1,773 batches. ScanRefer epoch 10 was at 4,018/4,865 training batches and NR3D epoch 2 at 3,570/4,490. All screens and GPUs were active; checked logs showed no errors. No new full evaluation row is available yet. The SR3D epoch-0 initialization remains its only completed evaluation; after epoch-1 validation, match the native metric row to the checkpoint epoch and preserve the full trained state before it can be overwritten.

## Follow-up status: 16:04 America/New_York

SR3D epoch 1 native validation progressed to 884/1,773 batches. ScanRefer epoch 10 reached 4,649/4,865 training batches and NR3D epoch 2 reached 4,224/4,490. All three jobs and GPUs remained active; checked log tails showed no errors. There were no new complete native metric rows or checkpoint updates at this poll.

## Follow-up status: 16:14-16:20 America/New_York

SR3D epoch 1 completed its full native validation: hits@0.25/0.50=11944/10166 of 17726, Acc=67.381%/57.351%, `primary_score=last/bbf`, `same_complete_model=true`. Compared with its untrained epoch-0 initialization (65.553%/56.544%), this is +1.829/+0.807 percentage points; Acc@0.25 remains 2.019 points below the 69.4% target. The epoch-1 checkpoint epoch and full model/optimizer/scheduler/RNG payload were verified, then retained as `best_trained_epoch1.pth` using a same-filesystem hard link.

ScanRefer epoch 10 completed full validation: hits@0.25/0.50=5617/4776 of 9508, Acc=59.077%/50.231%, `last/bbf`, same complete model. Relative to baseline 59.518%/46.992%, the changes are -0.441/+3.239 points; gaps to targets 60.0%/52.0% are 0.923/1.769 points. Epoch 10 is the best trained evaluation by Acc@0.50. Its complete checkpoint was matched to the native row and retained as `best_trained_epoch10.pth`.

NR3D epoch 2 completed full validation: hits@0.25/0.50=4780/3909 of 7899, Acc=60.514%/49.487%, `last/bbf`, same complete model. Versus untrained epoch 0 (59.995%/52.310%), the changes are +0.519/-2.823 points. Acc@0.25 is 0.714 points above its 59.8% target. Epoch 1 remains the best trained checkpoint under the Acc@0.50-first selection rule.

At approximately 16:20, the next rounds were active: ScanRefer epoch 11 at 210/4865 batches, NR3D epoch 3 at 279/4490, and SR3D epoch 2 at 488/7783. All screens and GPUs were active and checked log tails were clean. The verified commands still use `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and LR decay epochs `[10,30]`; no process or setting was changed.

At 16:30 America/New_York, no new validation rows had appeared. ScanRefer epoch 11 reached 965/4865 batches, NR3D epoch 3 reached 1056/4490, and SR3D epoch 2 reached 1258/7783. All three screens and GPUs remained active, checked logs contained no errors, and requested learning-rate/decay settings were unchanged.

Arithmetic correction at 16:32 America/New_York: raw NR3D hit counts and checkpoint selection are unchanged. Exact hits/7899 conversions are epoch 1 60.5899%/49.6012% and epoch 2 60.5140%/49.4873%; previous conversions for epoch-1 Acc@0.25 and epoch-2 Acc@0.50 were incorrect and are superseded.

## Follow-up status: 16:48 America/New_York

No new complete validation rows were available. ScanRefer epoch 11 was at 2298/4865 training batches, NR3D epoch 3 at 2425/4490, and SR3D epoch 2 at 2613/7783. All three screens and RTX 4090 GPUs were active; checked log tails showed no errors. Live arguments remain `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and LR decay epochs `[10,30]`; no settings or processes were changed.

Latest completed native `last/bbf` semantic-alignment results remain ScanRefer epoch 10: hits@0.25/0.50=5617/4776 of 9508 (59.077%/50.231%); NR3D epoch 2: 4780/3909 of 7899 (60.514%/49.487%); SR3D epoch 1: 11944/10166 of 17726 (67.381%/57.351%). All rows have `same_complete_model=true`. The exact NR3D percentage correction above is reflected in the current status and this branch.

## Follow-up status: 17:00 America/New_York

All three runs remained active; no new full native validation rows or checked log errors. Progress: ScanRefer epoch 11 at 3167/4865 batches, NR3D epoch 3 at 3320/4490, and SR3D epoch 2 at 3485/7783. The live configurations still use `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and decay epochs `[10,30]`; no settings or processes were changed.

Latest full `last/bbf`, `same_complete_model=true` results remain ScanRefer epoch 10 5617/4776 of 9508 (59.077%/50.231%), NR3D epoch 2 4780/3909 of 7899 (60.514%/49.487%), and SR3D epoch 1 11944/10166 of 17726 (67.381%/57.351%). Based on observed batch rates, NR3D full validation is expected around 17:25-17:30 and ScanRefer around 17:35; SR3D is later. The next check should be near the validation window.
