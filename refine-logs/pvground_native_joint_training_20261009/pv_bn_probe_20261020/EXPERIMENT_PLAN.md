# BN statistics probe (2026-10-20)

## Evidence that motivates it

Full ScanRefer 9508, native last/bbs, hits @0.25/@0.5. All runs start from the same PV+G+A+B E0 (5677/4920), coarse_endpoint recipe, seed 2027, B8.

| LR scale (core/G/backbone, A, B) | E1 | E2 | E3 |
|---|---:|---:|---:|
| 1.0x (1e-6, 1e-5, 1e-6) | 5678/4742 | 5657/4722 | - |
| 0.5x | 5669/4784 | 5634/4732 | - |
| 0.1x (1e-7, 1e-6, 1e-7) | 5653/4759 | 5649/4760 | 5642/4770 |

The @0.5 drop after one epoch is ~140-180 hits and essentially unchanged across a 10x LR range. Fine-tuning drift that does not scale with LR points at state updated without gradients: BatchNorm running statistics (momentum 0.1 in `models/pv_ground.py::init_bn_momentum`, re-estimated from single-GPU batch-8 augmented training data; the author checkpoint was trained with batch 24).

E0 rank analysis (`native_rows_epoch000.jsonl`): of 4588 @0.5 errors, 3514 select a box with IoU<0.1 (wrong object); only 104 of the 757 loose-box errors (0.25<=IoU<0.5) have a qualified candidate at bbs rank 2-5. Selection/quality reranking was not pursued again: the handoff records the F competition loss, candidate consistency/normalization, final-box quality supervision and SupportIdentityReadout as negative, and its rules exclude new quality heads, second rankings and post-hoc reranking.

## Code change

`bn_probe_main_utils.py` = `bounded_main_utils.py` (pv_bounded_lr_20261017) plus an opt-in `--freeze_bn` flag (`bn_probe_FREEZE_BN_DIFF.patch`): after `model.train()`, every `_BatchNorm` module is put back in eval mode so running statistics stay at their E0 values. Default off; existing recipes are unchanged. Server stage: `/root/autodl-tmp/pvground_bn_probe_20261020/PV-Ground` (copy of the 0.1x source stage; only `main_utils.py` and `scripts/launch_bn_probe.sh` differ).

## Arms (serial, single A100)

1. `bn_probe_lr0`: identical recipe, all LRs 0; stopped by hand after the E1 evaluation (the trainer asserts max_epoch == 8). Parameters cannot move (AdamW decay is lr-scaled). If E1 still loses ~150 @0.5 hits, the drop is BN-statistics drift.
2. `bn_probe_freeze_0p1`: 0.1x recipe with `--freeze_bn`. Tests whether training can then improve on E0 instead of starting ~150 hits below it.

Launch: `bash scripts/launch_bn_probe.sh <run_name> <overrides>`; see the script header. Target unchanged: 5724/4945 on the same complete model.

## Status

0.1x run (`lr_scale_0p1_retry2`) E0-E3 metrics: `lr_scale_0p1_retry2_native_metrics_E0_E3.jsonl`. No probe results yet.

2026-10-20 01:04: 0.1x run stopped by user decision during E4 (~25%), its E3 `latest.pth` deleted, E0 `best.pth` and logs kept (`USER_STOP.json`). The first lr0 launch with `--max_epoch 1` hit `assert max_epoch == 8`; relaunched 01:13 (PID 49788) with the default 8 epochs.

## Result: arm 1 (lr0) — BN running statistics cause the drop

| lr0, all LRs 0 | @0.25 | @0.5 |
|---|---:|---:|
| E0 | 5677 | 4920 (exact reproduction) |
| E1 | 5632 | 4749 (-45/-171) |
| E2 | 5638 | 4766 (-39/-154) |

Checkpoint check (E2 `latest.pth` vs E0 `best.pth`, same run): all 1043 non-BN tensors bit-identical; all 84 BatchNorm layers have changed `running_mean`/`running_var`/`num_batches_tracked`. With weights frozen, BN statistic re-estimation alone reproduces the ~150-170 @0.5 loss seen at every LR scale. The earlier fine-tuning "declines" were therefore not under-training. Stopped 2026-10-20 ~10:20 after E2 (`STOP_RESULT.json` written, both lr0 checkpoints deleted). Next: arm 2 `bn_probe_freeze_0p1` (`--freeze_bn`).

## Arm 2 (freeze_bn, 0.1x) — interim, launched 2026-10-20 10:25 (PID 57399)

| | @0.25 | @0.5 | old 0.1x (BN drifting) |
|---|---:|---:|---:|
| E0 | 5677 | 4920 | 5677/4920 |
| E1 | 5678 | 4884 | 5653/4759 |
| E2 | 5674 | 4888 | 5649/4760 |

Freezing BN removes most of the loss (+25/+125 over the drifting run at E1), but @0.5 is still 32-36 below E0, so the weight updates themselves cost a little strict localization. E3 in progress.
