# MCLN V51/V52 Experiment Plan

Timestamp: 2026-08-12 02:09 CST
Scope: remote-only worktree /home/gb/new butd/butd_detr-main/MCLN-main

## Primary claims

1. V51 BMQ-Rank can close the ScanRefer two-stage REC selection gap by aligning query scores with Acc@0.25/0.50, repairing missed parent queries as well as protecting correct ones, and focusing supervision on a deployable Top-K union.
2. If V51 preserves REC but mask@0.50/mIoU remain limited, V52 QTM-3D can improve RES by replacing the single text mask broadcast to all 256 queries with query-specific text-conditioned masks.

## Evidence before intervention

| system | REC@0.25 | REC@0.50 | Mask@0.25 | Mask@0.50 | Mask mIoU |
|---|---:|---:|---:|---:|---:|
| protected postprocessed | 0.582878 | 0.486012 | 0.596971 | 0.490324 | 0.417676 |
| V19 learned | 0.581195 | 0.465398 | 0.598233 | 0.491376 | 0.418613 |
| V19 candidate oracle | 0.629680 | 0.550063 | — | — | 0.451708 |
| target | 0.590000 | 0.490000 | preserve/improve | preserve/improve | preserve/improve |

The 9508-sample oracle clears both REC targets. The learned selector made only 5 corrections (2 useful, 3 harmful), so candidate coverage is not the principal bottleneck.

## Experiment blocks

### R1 — bidirectional anchor

Protect a correct parent and repair a wrong parent when a hit candidate exists. Margins are 0.05 at IoU 0.25 and 0.10 at IoU 0.50.

### R2 — V51 BMQ-Rank

- Smooth utility: box weights (1,2,0.5), mask weights (0.5,1,0.5) times 0.25, sigmoid temperature 0.05.
- Candidate union: deployed Top-16, each source Top-8, GT utility Top-4 during training, always parent.
- Gain-weighted pairwise loss weight 0.5 plus listwise loss.
- Direct residual scale 0.25.
- Retain V50 SACR, mask calibration, spatial refiner and candidate mask losses.
- Smoke: 2 debug epochs and 128 validation samples.
- Formal entry: finite smoke, checkpoint, evaluation receipt, no protected-weight mutation.

### R3 — V52 QTM-3D

After R2 full validation, replace the broadcast text mask with 256 query-specific text masks conditioned on target/attribute/relation/anchor slots and a soft proposal prior. Add iterative masked cross-attention and quality-weighted mask supervision.

### R4 — V53 DN-Group Refiner

Only if the candidate oracle remains high but selected rank stays deficient. Unfreeze the last two decoder layers at lower LR and add denoising/one-to-many groups.

## Required receipts

REC overall/unique/multiple at 0.25 and 0.50; mask overall/unique/multiple at 0.25 and 0.50 plus mIoU; parent hit, fix, break, net; candidate oracle; selected-query oracle rank; score/IoU Spearman; calibration; source entropy; switch ratio.

## Stop rules

Reject incomplete 9508-sample formal evaluation, NaN/Inf, checkpoint-contract failure, protected-weight mutation, or a run that gains one threshold by causing a larger break regression at the other. Keep R2 separate from QTM-3D. The older unexecuted V51-RAPF preregistration remains only an optional reliability ablation.
