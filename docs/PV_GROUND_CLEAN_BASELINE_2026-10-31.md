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

## NR3D Epoch 9 Complete: 2026-11-01 01:04 UTC

NR3D epoch9 is a full7899-row last/bbf same-complete evaluation:
4446/3280 hits, Acc02556.2856058%, Acc05041.5242436%.
CPU audit confirms both latest.pth and best.pth are epoch9 and match
this native row, with the expected full checkpoint state.
Latest-three trained means (epochs7-9) are55.809%/41.516%;
epochs6-8 were55.595%/41.444%, so both increased and the early-stop
decline streak remains zero.

At the export snapshot, ScanRefer was training epoch8 (~33%),
NR3D was validating epoch9 at701/790, and SR3D was training epoch4
(~88%). No intervention or parameter change was made.
The complete-epoch CSV now contains22 rows.

## SR3D Epoch 4 Complete: 2026-11-01 01:34 UTC

SR3D epoch4 completed with17726 rows: hits025/050=11634/9230,
Acc02565.6324044%, Acc05052.0704051%, last/bbf,
same_complete_model=true. CPU latest.pth and best.pth both match
epoch4 and its exact metric row. Both thresholds improved over epoch3,
so the SR3D trained rolling means increased; early-stop decline streak
is zero.

The all-datasets CSV now contains23 complete rows. ScanRefer epoch8
was training at4334/4865 (~89%) during the prior check. SR3D had
started epoch5 training at105/7783. NR3D remains active; its latest
completed validation is epoch9.

## ScanRefer Epochs 8-9 and NR3D Epoch 10: 2026-11-01 01:50 UTC

| Dataset | Epoch | Count | Hits025 / Hits050 | Acc025 / Acc050 |
| --- | ---: | ---: | ---: | ---: |
| ScanRefer | 8 | 9508 | 4729 / 3255 | 49.7371% / 34.2343% |
| ScanRefer | 9 | 9508 | 4776 / 3267 | 50.2314% / 34.3605% |
| NR3D | 10 | 7899 | 4493 / 3322 | 56.8806% / 42.0560% |

All three rows are complete last/bbf evaluations with
same_complete_model=true. CPU best/latest metadata for ScanRefer matches
epoch9 and its exact native row; NR3D best/latest matches epoch10.

ScanRefer's latest three-round means (epochs7-9) increase on both
thresholds. NR3D's epochs8-10 means are56.311%/41.853%, above
epochs7-9 at55.809%/41.516%. Both early-stop decline streaks remain zero.

At the 2026-11-01 01:49 UTC export, ScanRefer validation was901/951
and NR3D validation was701/790. SR3D epoch5 is training; its latest
completed validation remains epoch4. The combined CSV now has26 rows.

## ScanRefer Epoch 10 and NR3D Epoch 11: 2026-10-31 22:48 EDT

| Dataset | Epoch | Count | Hits025 / Hits050 | Acc025 / Acc050 |
| --- | ---: | ---: | ---: | ---: |
| ScanRefer | 10 | 9508 | 4872 / 3306 | 51.2411% / 34.7707% |
| NR3D | 11 | 7899 | 4472 / 3310 | 56.6148% / 41.9040% |

Both are complete `last/bbf` rows with `same_complete_model=true`.
CPU audit confirms ScanRefer best/latest both select epoch10 and match
that native row. NR3D best remains epoch10 (Acc025-first); latest is
epoch11. Both checkpoints were CPU-audited against their own native rows,
with1134 model tensors and707 optimizer-state entries.

NR3D's epochs9-11 three-round mean is56.5937%/41.8281%, versus
56.3109%/41.8534% for epochs8-10. Only Acc050's rolling mean declined,
so the simultaneous-two-threshold early-stop streak remains zero.
ScanRefer epochs8-10 means are50.4032%/34.4552%, up from epochs7-9
at49.7406%/34.0485%. No early stop is due.

At the latest live snapshots, ScanRefer had started epoch11, NR3D epoch12,
and SR3D remained in epoch5. All screens and GPUs were active. The complete
metric table now contains28 rows; raw native metrics and the new ScanRefer
epoch10 / NR3D epoch11 CPU checkpoint audits are saved alongside it.

## ScanRefer Epoch 11, NR3D Epoch 12, SR3D Epoch 5: 2026-10-31 23:47 EDT

| Dataset | Epoch | Count | Hits025 / Hits050 | Acc025 / Acc050 |
| --- | ---: | ---: | ---: | ---: |
| ScanRefer | 11 | 9508 | 4876 / 3316 | 51.2831% / 34.8759% |
| NR3D | 12 | 7899 | 4491 / 3336 | 56.8553% / 42.2332% |
| SR3D | 5 | 17726 | 11510 / 9124 | 64.9329% / 51.4724% |

All rows are complete last/bbf evaluations with `same_complete_model=true`.
CPU checkpoint audits match ScanRefer best/latest to epoch11, NR3D best
epoch10/latest epoch12, and SR3D best epoch4/latest epoch5 to each
checkpoint's own native metric row. ScanRefer uses Acc050-first selection;
NR3D and SR3D use Acc025-first selection.

The latest trained three-round means increased on both thresholds for
every dataset: ScanRefer epochs8-10 to9-11 are50.40%/34.46% to
50.92%/34.67%; NR3D epochs9-11 to10-12 are56.59%/41.83% to
56.78%/42.06%; SR3D epochs2-4 to3-5 are64.58%/50.95% to
65.00%/51.47%. No automatic early stop is due.

At the snapshot, all jobs/screens/GPUs remained active: ScanRefer epoch12
training94/4865, NR3D epoch13 training1102/4490, and SR3D epoch6
training439/7783. The combined CSV contains31 complete rows. Updated
native snapshots and ScanRefer epoch11 / SR3D epoch5 CPU checkpoint
audits are saved in `docs/results/pvground_clean_baseline_20261031/`.

## ScanRefer Epoch 12 Complete: 2026-11-01 04:46 UTC

ScanRefer epoch12 completed all9508 expressions with hits025/050
4854/3326 and Acc025/Acc050 51.0517459%/34.9810686%. The row uses
last/bbf, `same_complete_model=true`, and the expected clean architecture.
CPU audit at04:59 UTC matched both `best.pth` and `latest.pth` to this
epoch's own native metric row; each checkpoint contains1134 model tensors
and707 optimizer states. ScanRefer's Acc050-first selected checkpoint is
now epoch12.

The trained rolling means for epochs10-12 increased on both thresholds
over epochs9-11, so the automatic early-stop streak remains zero. At the
05:00 UTC live snapshot, ScanRefer was training epoch13 at1293/4865,
NR3D epoch14 at2508/4490, and SR3D epoch6 at5278/7783. All screens and
GPUs were active; no configuration or process changes were made.

The combined all-epoch CSV now contains33 complete rows: ScanRefer13,
NR3D14, and SR3D6. The updated ScanRefer native snapshot and
`scanrefer-checkpoint-metadata-epoch12.json` are saved beside the CSV.

## NR3D Epoch 14 Complete: 2026-11-01 05:25 UTC

NR3D epoch14 completed all7899 expressions with hits025/050 4492/3334
and Acc025/Acc050 56.8679580%/42.2078744%. The row is complete
`last/bbf` with `same_complete_model=true`. CPU audit confirms latest.pth
matches its own epoch14 native row; best.pth remains epoch10 by the
Acc025-first rule. Both checkpoints have1134 model tensors and707
optimizer states.

The epochs12-14 trained rolling means increased on both thresholds over
epochs11-13, leaving the early-stop streak at zero. At05:30 UTC,
ScanRefer was training e13 at2400/4865, NR3D e15 at469/4490, and SR3D
e6 at7344/7783. All screens and GPUs were active, with no training
configuration or process changes.

The combined CSV now contains34 complete rows: ScanRefer13, NR3D15,
and SR3D6. Updated NR3D native metrics and epoch14 CPU checkpoint
metadata are saved in `docs/results/pvground_clean_baseline_20261031/`.

## ScanRefer Epoch 14 Complete: 2026-11-01 06:51 UTC

ScanRefer epoch14 completed all9508 expressions: hits025/050=4896/3340,
Acc025/Acc050=51.4934792%/35.1283130%. The full row is `last/bbf` with
`same_complete_model=true`. CPU audit confirms best.pth and latest.pth
both match this epoch's own row; the Acc050-first best advances to14.
The epochs12-14 three-round means rise over epochs11-13 on both metrics,
so no early-stop decline streak is active.

At06:50 UTC, ScanRefer epoch15 was training305/4865, NR3D epoch16
2713/4490, and SR3D epoch7 4061/7783. All runs remain active. The combined
all-epoch CSV contains38 complete rows: ScanRefer15, NR3D16, SR3D7.
The ScanRefer native snapshot and epoch14 CPU checkpoint metadata are
saved with the experiment results.

## ScanRefer Epoch 13 and SR3D Epoch 6: 2026-11-01 05:56 UTC

| Dataset | Epoch | Count | Hits025 / Hits050 | Acc025 / Acc050 |
| --- | ---: | ---: | ---: | ---: |
| ScanRefer | 13 | 9508 | 4889 / 3333 | 51.4199% / 35.0547% |
| SR3D | 6 | 17726 | 11590 / 9305 | 65.3842% / 52.4935% |

Both are complete `last/bbf` rows with `same_complete_model=true`.
CPU checkpoint metadata matches ScanRefer best/latest to their own epoch13
row. ScanRefer's Acc050-first selected checkpoint advances to epoch13.
SR3D latest matches epoch6; its Acc025-first selected checkpoint remains
epoch4 at65.6324%/52.0704%.

NR3D's latest complete row remains epoch14:4492/3334 of7899,
56.8680%/42.2079%; CPU metadata matches latest epoch14 and best epoch10.
The automatic early-stop condition is not met: ScanRefer's e11-13
three-epoch means rise over e10-12; SR3D's e4-6 means rise over e3-5.
NR3D e12-14 means also rise over e11-13; epoch15 is still training.

At the05:56 UTC live snapshot, ScanRefer epoch14 was training833/4865,
NR3D epoch15 was training2807/4490, and SR3D epoch7 was training406/7783.
All screens and GPUs were active. The combined all-epoch CSV contains36
complete rows: ScanRefer14, NR3D15, SR3D7. Native snapshots and CPU
checkpoint audits for ScanRefer epoch13 and SR3D epoch6 are saved in
`docs/results/pvground_clean_baseline_20261031/`.

## NR3D Epoch 15 Complete: 2026-11-01 06:24 UTC

NR3D epoch15 completed all7899 expressions with hits025/050 4503/3340
and Acc025/Acc050 57.0072161%/42.2838334%. It is a complete `last/bbf`
row with `same_complete_model=true`. CPU audit matches both best.pth and
latest.pth to epoch15's own native metric row (1134 model tensors and707
optimizer states); the Acc025-first selected checkpoint advances to15.

The trained rolling means for epochs13-15 rise on both thresholds over
epochs12-14, so the automatic early-stop decline streak remains zero.
At06:23 UTC, NR3D had started epoch16 at285/4490; ScanRefer epoch14 was
training3318/4865, and SR3D epoch7 was training2256/7783. All three
screens and GPUs remained active; no settings or processes were changed.

The combined all-epoch CSV now contains37 full rows: ScanRefer14, NR3D16,
and SR3D7. NR3D native metrics and epoch15 CPU checkpoint metadata are
saved in `docs/results/pvground_clean_baseline_20261031/`.

## NR3D Epoch 16 Complete: 2026-11-01 07:17 UTC

NR3D epoch16 completed all7899 expressions with hits025/050=4473/3317
and Acc025/Acc050=56.6274212%/41.9926573%. The row is complete
`last/bbf` with `same_complete_model=true`. CPU audit verifies latest.pth
matches its own epoch16 row; best.pth remains epoch15 under Acc025-first
selection. Both checkpoints contain1134 model tensors and707 optimizer
states.

The epochs14-16 three-round means (56.8342%/42.1615%) declined on both
thresholds from epochs13-15 (56.8764%/42.2037%). This is the first
consecutive dual-decline comparison; it does not yet meet the required
two consecutive comparisons. At07:16 UTC, NR3D epoch17 was training99/4490.
ScanRefer epoch15 and SR3D epoch7 remained active; their latest complete
rows are e14 and e6 respectively. The combined CSV now has39 rows:
ScanRefer15, NR3D17, SR3D7.

## ScanRefer Epoch 15 Complete: 2026-11-01 07:50 UTC

ScanRefer epoch15 completed all9508 expressions with hits025/050=4918/3359
and Acc025/Acc050=51.7248633%/35.3281447%. It is a full `last/bbf` row
with `same_complete_model=true`. CPU audit verifies best.pth and latest.pth
both match epoch15's own row; Acc050-first selected checkpoint advances
to15. Its epochs13-15 rolling means rise over epochs12-14.

At07:49 UTC, ScanRefer was training epoch16 at232/4865; NR3D epoch17
at3098/4490; and SR3D epoch7 validation at200/1773. All jobs remain
active. The all-epoch CSV now contains40 rows: ScanRefer16, NR3D17, SR3D7.

## SR3D Epoch 7 Complete: 2026-11-01 08:02 UTC

SR3D epoch7 completed all17726 expressions with hits025/050=11609/9352
and Acc025/Acc050=65.4913686%/52.7586596%. This is a complete `last/bbf`
row with `same_complete_model=true`. CPU audit matches latest.pth to its
own epoch7 row; Acc025-first best remains epoch4 (65.6324%/52.0704%).
The epochs5-7 rolling means are mixed versus epochs4-6, so no early-stop
decline is counted.

At08:01 UTC, ScanRefer epoch16 was training1273/4865 and NR3D epoch17
4145/4490. Both jobs and GPUs remained active. The combined table
contains41 rows: ScanRefer16, NR3D17, SR3D8.

## NR3D Epoch 17 Complete: 2026-11-01 08:10 UTC

NR3D epoch17 completed all7899 `last/bbf` rows with hits025/050
4489/3334 and Acc025/Acc050=56.8299785%/42.2078744%.
`same_complete_model=true`. CPU audit matches `latest.pth` to its
own epoch17 native row (1134 model tensors,707 optimizer states);
`best.pth` remains epoch15 under Acc025-first selection.

The epochs15-17 three-round means are56.8215%/42.1615%, compared with
56.8342%/42.1615% for epochs14-16. Acc025 declined slightly while
Acc050 was unchanged, so this is not a dual-threshold decline and the
automatic early-stop streak resets to zero. At08:23 UTC, NR3D epoch18
was training1205/4490; ScanRefer epoch16 was3321/4865 and SR3D epoch8
was1496/7783. All three jobs remained active.

The combined all-epoch CSV contains42 full rows: ScanRefer16, NR3D18,
and SR3D8. The epoch17 native metrics snapshot and CPU checkpoint audit
are saved in `docs/results/pvground_clean_baseline_20261031/`.

## ScanRefer Epoch 16 Complete: 2026-11-01 08:47 UTC

ScanRefer epoch16 completed all9508 `last/bbf` rows with hits025/050
4894/3361 and Acc025/Acc050=51.4724443%/35.3491796%.
`same_complete_model=true`. CPU audit matches both `best.pth` and
`latest.pth` to their own epoch16 row (1134 model tensors,707 optimizer
states); the Acc050-first selected checkpoint advances to epoch16.
The epochs14-16 three-round means rise over epochs13-15, so no
dual-decline is counted.

At08:53 UTC, ScanRefer epoch17 was training604/4865, NR3D epoch18
3925/4490, and SR3D epoch8 3491/7783. All three jobs remained active.
The combined CSV contains43 full rows: ScanRefer17, NR3D18, SR3D8.
The epoch16 native metrics snapshot and CPU checkpoint audit are saved
in `docs/results/pvground_clean_baseline_20261031/`.

## NR3D Epoch 18 Complete: 2026-11-01 09:05 UTC

NR3D epoch18 completed all7899 `last/bbf` rows with hits025/050
4505/3351 and Acc025/Acc050=57.0325358%/42.4230915%.
`same_complete_model=true`. CPU audit matches both `best.pth` and
`latest.pth` to their own epoch18 native row (1134 model tensors,707
optimizer states); the Acc025-first selected checkpoint advances to18.

The epochs16-18 three-round means rose to56.8300%/42.2079% from
56.8215%/42.1615% for epochs15-17, clearing the early-stop decline
streak. At09:00 UTC, ScanRefer epoch17 was training1178/4865, NR3D
epoch19 was training11/4490, and SR3D epoch8 was4288/7783. All jobs
remained active. The combined CSV contains44 full rows: ScanRefer17,
NR3D19, SR3D8. Native metrics and epoch18 CPU checkpoint metadata are
saved in `docs/results/pvground_clean_baseline_20261031/`.

## ScanRefer Epoch 17 and NR3D Epoch 19: 2026-11-01 10:04 UTC

| Dataset | Epoch | Count | Hits025 / Hits050 | Acc025 / Acc050 |
| --- | ---: | ---: | ---: | ---: |
| ScanRefer | 17 | 9508 | 4902 / 3357 | 51.5565839% / 35.3071098% |
| NR3D | 19 | 7899 | 4495 / 3341 | 56.9059375% / 42.2964932% |

Both rows are complete `last/bbf` evaluations with
`same_complete_model=true`. CPU audits match ScanRefer `latest.pth` to
its epoch17 row while `best.pth` remains epoch16 (Acc050-first). NR3D
`latest.pth` matches epoch19 while `best.pth` remains epoch18
(Acc025-first). Every audited checkpoint contains1134 model tensors and
707 optimizer-state entries.

NR3D rolling means for epochs17-19 are56.9228%/42.3092%, above the
epochs16-18 means of56.8300%/42.2079%; the automatic early-stop condition
is not met. At10:01 UTC, ScanRefer epoch18 was training1318/4865, NR3D
epoch20 was training61/4490, and SR3D epoch8 validation was301/1773.
All jobs remained active. The combined CSV contains46 complete rows.
Exact native metric rows and CPU audits are stored in
`docs/results/pvground_clean_baseline_20261031/`; no training or process
settings were changed.
