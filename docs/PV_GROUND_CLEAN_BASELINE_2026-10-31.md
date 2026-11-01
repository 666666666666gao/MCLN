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

## Confirmed Architecture

The user explicitly confirmed removal of the original PV-Ground paper's
innovations, rather than just our three added modules.

The actual CVPR paper was downloaded and extracted locally on 2026-10-31.
Section 4.3 and Table 4 specify two architectural ablations:

- No PVI: replace the sparse voxel backbone and Voxel Set Abstraction
  with the standard PointNet++ backbone.
- No TGS: replace Gumbel soft sampling with supervised Top-K selection
  on FPS seed points.

The third contribution bullet is experimental evidence, not a third
independent module. Section 3.4 explicitly identifies the decoder and
multi-task box/mask prediction heads as inherited from MCLN; these stay.
Our Reader / Support Corrector / Span Mixer are absent from the clean
baseline. There is no new mask-to-box replacement.

Primary reference: [PV-Ground paper, Sections 3.4 and 4.3, Table 4](
https://openaccess.thecvf.com/content/CVPR2026/papers/Shang_PV-Ground_Text-Guided_Point-Voxel_Interaction_for_3D_Visual_Grounding_CVPR_2026_paper.pdf).
Official repository: AaNnWwTt/PV-Ground,
commit 262e2592589baec7bb83a0d46aae6542d4ccedfb.
Original model SHA256:
fca2dcb6b5dd1dd59b9f4e12f999664f1aa6730374aef6ed09a66fe0a2f0a2f7.
Downloaded primary paper PDF SHA256:
ad96d075ca310f5cc48d875074cf5dc6af594938904811be3d734df0416287d2.
Patched model SHA256:
3b43604a96adc2eb7cb37fa78aea3846d3e6e3c138009b964843182e2852951f.
All three machines matched these model hashes at deployment.

## Weight Transfer And Training

Each dataset starts from its own original author checkpoint, with its
SHA256 checked before any optimizer update. Every retained tensor must
match name and shape. Only replaced backbone/TGS tensors and the
previously audited obsolete RoBERTa position_ids buffer are excluded.
Every newly initialized tensor must belong to PointNet++ or the Top-K
objectness classifier. The complete lists are written to load_audit.json.

No PointNet++ detector checkpoint was found in the inspected runtime.
Therefore PointNet++ and the Top-K classifier use seeded fresh
initialization. This is partial transfer from PV-Ground, not a complete
pretrained point-only baseline and not a from-scratch architecture-neutral
comparison. Retained parameters still carry PV-Ground pretraining history.
An initial accuracy drop after this replacement must not be described as
a failed reproduction of the intact author checkpoint.

The Top-K classifier restores the inherited MCLN/EDA query-generation
focal-loss term with coefficient 8; PV-Ground computed this term but
omitted it from the total when using TGS. Otherwise author loss, decoder,
heads, dataset and grounding evaluator are preserved.

Training uses fresh AdamW and scheduler state:

- LR 1e-5; PointNet++ LR 2e-5; batch 10; seed 111.
- Decay [10,30], warmup_epoch=0. The official default -1 shifts actual
  scheduler milestones by one epoch, hence it is explicitly overridden.
- All retained non-RoBERTa parameters are trainable, with no module-only
  freeze. RoBERTa keeps the author's frozen configuration.
- ScanRefer uses BUTD, self_attend, augment_det, joint_det.
- NR3D/SR3D use BUTD-class filtering, self_attend, joint_det.
- Maximum trained epochs: ScanRefer/NR3D 100, SR3D 140, per author scripts.
- Automatic early stop excludes epoch0: both 3-epoch rolling means must
  decrease for two consecutive comparisons.

A real batch-10 preflight requires finite loss, nonzero gradients and
parameter changes in PointNet++, Top-K classifier, decoder, box heads
and mask projection. Parameters, buffers, optimizer state and process RNG
are reset after the probe; it is not counted as a training epoch.

Epoch0 is a full untrained validation of the changed architecture.
All metrics use last/bbf semantic alignment and exactly ScanRefer9508,
NR3D7899, SR3D17726 expressions. Same-checkpoint pairs are mandatory.
ScanRefer selection is Acc050-first; NR3D/SR3D are Acc025-first, with
the other threshold as tie breaker. best.pth may be epoch0;
best_trained.pth excludes it. latest.pth stores its own metrics.

## Deployment

Three isolated runtime roots:
/root/autodl-tmp/pvground_clean_baseline_20261031.
The original author tree and old run directories are retained.

| Dataset | Endpoint | Screen |
| --- | --- | --- |
| ScanRefer | connect.bjb2.seetacloud.com:48073 | pvg_scanrefer_clean_baseline |
| NR3D | connect.nmb1.seetacloud.com:32749 | pvg_nr3d_clean_baseline |
| SR3D | connect.nmb2.seetacloud.com:45914 | pvg_sr3d_clean_baseline |

Log: train_clean.log. Results:
results/DATASET/clean_baseline/RUN_ID/.
First attempts began approximately 12:12 EDT. The current successful
runs started at12:37EDT and passed real-update verification below.
The first startup attempts were interrupted before training to correct
the audited position_ids exclusion and scheduler milestone configuration.
The next attempt exposed an overly strict assertion on the missing-key
list: PyTorch automatically supplies new BatchNorm num_batches_tracked
buffers. The loader now constructs the fully audited merged state and
uses strict=True. Name/shape checks remain mandatory. Checkpoint auditing
now happens before dataset preprocessing, so an invalid checkpoint cannot
waste another dataset load. The PointNet++ B,C,N output is explicitly
transposed to the original PV cross-encoder's B,N,C layout.
All three actual CUDA backbone witnesses passed (seed111, input
2x50000x6, output2x1024x288, finite, peak353683456 bytes).
The12:23 attempts were superseded by the12:37 corrected runs below.

The old pv-ground heartbeat no longer existed (verified by update result);
one replacement heartbeat, pv-ground-clean-baseline, was created in this
chat, every 15 minutes. Do not create another monitor.

At12:28EDT all three original checkpoint SHA256 values matched. Each
model loaded1022 retained tensors and freshly initialized112 replacement
tensors. ScanRefer excluded212 author tensors; NR3D/SR3D excluded213
(including their obsolete RoBERTa position_ids buffer). NR3D's actual
batch10 update passed: loss32.88343048, nonzero gradients and changed
parameters in all five audited paths, peak16798477312 bytes.
ScanRefer/SR3D real-data updates were still pending at that observation.

The 12:23 attempts subsequently passed actual updates for ScanRefer and
NR3D, then failed initial validation because this runner had omitted the
author's existing size-clamping evaluation step. The correction directly
reuses BaseTrainTester._main_eval_branch, including the original
pred_size clamp and statistics; it does not invent a new postprocessor.
All three corrected jobs restarted at12:37EDT. Runner SHA256:
382e23750334d77846c01b2281dd6657c38c074cefa5726b69aec542d696b423.
All three current-run update checks and evaluation progress are verified.
Monitoring/export select the latest config.json run and never substitute
an older attempt's first_update.json or load_audit.json as its evidence.

Do not restart old module fine-tuning or previously queued geometry
diagnostics. Prior optimization goal remains paused.

## Verified Startup: 2026-10-31 12:55 EDT

| Dataset | Current run | Batch10 loss | Complete epoch0 Acc025 / Acc050 |
| --- | --- | ---: | --- |
| ScanRefer | 1793464620 | 34.795635 | 1.640724% / 0.147244% |
| NR3D | 1793464625 | 32.883430 | 8.570705% / 2.278769% |
| SR3D | 1793464628 | 36.046616 | 12.439355% / 4.620332% |

All five checked paths have nonzero gradient norms and a parameter
actually changed by AdamW. Probe states were reset before epoch0.
Current first_update.json files prove these current runs, not an older
failed attempt. Peak allocated GPU bytes: Scan16814440448,
NR16798477312, SR16545724416.

At12:51, ScanRefer and NR3D were training epoch1. The SR3D snapshot
at12:54 shows initial validation601/1773; no partial accuracy is reported.
Scan epoch0 has156/14 hits among9508; NR epoch0 has677/180 among7899.
Both rows use last/bbf and same_complete_model=true.
CPU best.pth/latest.pth audits match each checkpoint's own native epoch0
row and PointNet++_TopK_noPVI_noTGS architecture; each contains1134
model tensors and an empty optimizer state, as expected before training.
These low scores are untrained replacement-architecture results and
must not be presented as intact author-checkpoint reproduction scores.

Startup evidence and native rows:
docs/results/pvground_clean_baseline_20261031/*-startup.json.
CPU metadata: *-checkpoint-metadata.json in the same directory.
The three source checkpoint SHA values are recorded in each load audit.
Every model retained1022 tensors and initialized112 tensors; exclusions
are212 for ScanRefer and213 for NR3D/SR3D.

## SR3D Initial Validation Complete: 2026-10-31 13:13 EDT

SR3D epoch0 completed at13:04:42EDT:2205/819 hits among17726,
Acc02512.4393546203% and Acc0504.6203317161%, last/bbf,
same_complete_model=true. It is the untrained changed architecture.
CPU best/latest metadata was checked against this exact native row;
both epoch0 with1134 modeltensors and emptyoptimizerstate.
All three screens and GPUs are active; all are trainingepoch1.
At13:13 NR reached2324/4490 and SR567/7783. No trained complete
epoch exists yet. Based on current rates, NR1 full~13:45-13:50,
Scan1~13:55-14:00 and SR1~15:10-15:25; use actual progress to
refine these estimates. Do not immediately repoll or reload metadata.

## Monitoring Snapshot: 2026-10-31 13:28 EDT

All metric rows below are complete last/bbf evaluations with
same_complete_model=true. The CSV contains all21 complete rows.
Checkpoint audits match Scan best/latest epoch7, NR epoch8 and SR epoch3
to each checkpoint's own native row.

| Dataset | Epoch | Hits025 / Hits050 | Acc025 / Acc050 |
| --- | ---: | ---: | ---: |
| ScanRefer | 0 | 156 / 14 | 1.6407% / 0.1472% |
| ScanRefer | 1 | 3855 / 2352 | 40.5448% / 24.7371% |
| ScanRefer | 2 | 4194 / 2653 | 44.1102% / 27.9028% |
| ScanRefer | 3 | 4412 / 2882 | 46.4030% / 30.3113% |
| ScanRefer | 4 | 4533 / 3020 | 47.6756% / 31.7627% |
| ScanRefer | 5 | 4630 / 3111 | 48.6958% / 32.7198% |
| ScanRefer | 6 | 4690 / 3163 | 49.3269% / 33.2667% |
| ScanRefer | 7 | 4683 / 3190 | 49.2533% / 33.5507% |
| NR3D | 0 | 677 / 180 | 8.5707% / 2.2788% |
| NR3D | 1 | 3872 / 2495 | 49.0189% / 31.5863% |
| NR3D | 2 | 4110 / 2879 | 52.0319% / 36.4477% |
| NR3D | 3 | 4220 / 3068 | 53.4245% / 38.8404% |
| NR3D | 4 | 4256 / 3150 | 53.8802% / 39.8785% |
| NR3D | 5 | 4312 / 3220 | 54.5892% / 40.7647% |
| NR3D | 6 | 4395 / 3263 | 55.6400% / 41.3090% |
| NR3D | 7 | 4374 / 3242 | 55.3741% / 41.0432% |
| NR3D | 8 | 4405 / 3316 | 55.7666% / 41.9800% |
| SR3D | 0 | 2205 / 819 | 12.4394% / 4.6203% |
| SR3D | 1 | 10883 / 8298 | 61.3957% / 46.8126% |
| SR3D | 2 | 11285 / 8845 | 63.6635% / 49.8985% |
| SR3D | 3 | 11423 / 9018 | 64.4421% / 50.8744% |

## SR3D Initialization Audit

The low epoch0 result is complete and protocol-consistent:17726 rows,
last/bbf, same_complete_model=true. CPU best/latest match the epoch0
native row. The source checkpoint transfer audit found96 newly
initialized PointNet++ backbone tensors and16 Top-K tensors because
the supplied PV-Ground checkpoint has no compatible pretrained
counterparts. The other1022 matching tensors were loaded.

The change recovers after optimization: epoch1=61.3957%/46.8126%,
epoch2=63.6635%/49.8985%, epoch3=64.4421%/50.8744%.
At the 2026-11-01 00:27 UTC snapshot, epoch3 validation progress
was1701/1773. This supports the new-backbone initialization explanation
but does not establish final convergence to intact-PV-Ground results.
All21 complete rows are in
docs/results/pvground_clean_baseline_20261031/all-epochs.csv.
