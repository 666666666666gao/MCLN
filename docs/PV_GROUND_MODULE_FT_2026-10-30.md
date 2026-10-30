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

Best trained evaluation by Acc@0.50 is epoch 9. Relative to the recorded baseline (59.518% / 46.992%), epoch 9 is -0.536 / +2.850 percentage points. The remaining gaps to target are 1.018 points at Acc@0.25 and 2.158 points at Acc@0.50. The stricter-threshold metric has improved across recent rounds, while Acc@0.25 remains below baseline and target. This is consistent with a localization/box-precision bottleneck, but does not establish its cause.

At the 15:15 status sample, epoch 10 was training at 1,048/4,865 batches. Complete full-set evaluations were available through epoch 9.

## NR3D

Validation size: 7,899.

| Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 4739 | 4132 | 59.995% | 52.310% |
| 1 | 4786 | 3918 | 60.615% | 49.601% |

Epoch 1 is the first trained evaluation. Compared with initialization, Acc@0.25 increased by 0.621 points and exceeds the 59.8% target by 0.815 points; Acc@0.50 decreased by 2.709 points. The target for this run is Acc@0.25, and the stricter-threshold change is diagnostic rather than a stated acceptance criterion. One trained epoch is insufficient to conclude whether this trade-off persists.

At the 15:15 status sample, epoch 2 was training at 527/4,490 batches. The epoch-1 checkpoint was verified against its full native metric row and preserved as `best_trained_epoch1.pth` before `latest.pth` could be overwritten.

## SR3D

Validation size: 17,726.

| Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 11620 | 10023 | 65.553% | 56.544% |

The initialization is 3.847 points below the 69.4% Acc@0.25 target and is not a fine-tuning result. At the 15:15 status sample, epoch 1 was training at 4,855/7,783 batches; no trained evaluation was yet complete.

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
