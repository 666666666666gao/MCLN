# Density-Aware Target Box Scene-Disjoint Paired Audit Spec

Date: 2026-09-01  
Status: frozen before implementation  
Scope: Nr3D train-domain evidence only; audit-only; never authorizes long training

## 1. Objective

Determine whether the fixed Density-Aware Target Box auxiliary improves sparse-target proposal localization beyond the effect of 100 additional ordinary training microbatches, while preserving the deployed V99 ranking behavior on a scene-disjoint Nr3D holdout.

This audit does not reproduce an external baseline, does not use the rejected experiment/section 7 or 8 routes, and does not use the old E0--E7 matrix.

## 2. Immutable method

- Active target rows are exactly Nr3D rows with `0 < n_target_points < 256`.
- Density weight is exactly `1 - n_target_points / 256` and is detached.
- The supervised query is the final-layer Hungarian match for GT target slot 0.
- Per-row auxiliary is center L1 sum plus `0.2 * size L1 sum`.
- The only exposed method variable is the existing auxiliary loss weight: control `0.0`, method `1.0`.
- The module has no parameters and no inference branch.

The threshold, size coefficient, weight, batch budget, fold, seed, optimizer, scheduler and learning rate must not be changed after observing results.

## 3. Frozen data split

- Source: Nr3D train-domain annotations only, exactly 32,919 rows in 511 scenes.
- Fold mapping: first 32 bits of SHA-256(`scan_id`) modulo 5.
- Audit fold: exactly fold 2.
- Fit: 408 scenes, 26,590 rows.
- Holdout: 103 scenes, 6,329 rows.
- Fit and holdout scenes must be disjoint and their union must equal the frozen Nr3D train scene set.
- The model command retains the V99 `joint_det+butd_cls` architecture contract, but the audit dataset views contain no ScanNet detection rows.
- Holdout augmentation is disabled. Fit augmentation retains the existing training behavior.

Every consumed fit and holdout row must carry its immutable source-row identity. Receipts must contain count, unique count and an order-independent SHA-256 over the exact row set.

## 4. Frozen roles

All roles use the same reviewed code/input snapshot, protected full-state E57 checkpoint SHA-256 `fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655`, GroupFree checkpoint SHA-256 `9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2`, V99 selector configuration, B16 x A1, seed 0 and single rank.

1. `parent`: load E57 full state and evaluate only the fold-2 holdout.
2. `control`: load the same E57 full state, consume exactly the first 100 fit microbatches / 1,600 unique rows / 100 optimizer steps with auxiliary weight 0, then evaluate the holdout.
3. `method`: repeat the exact control contract and fit-row identities with auxiliary weight 1, then evaluate the holdout.

Control and method may differ only in role/output identity and `density_aware_target_box_loss_weight`.

## 5. Exact deployment-aligned metrics

For every holdout row:

- Build final candidate boxes from `last_center + last_pred_size`.
- Rank with `selected_source_scores` from the existing V99 selector.
- Apply the same `butd_cls` detector-overlap validity filter as the formal evaluator at IoU strictly greater than 0.25.
- A row with no valid candidate is a miss at every selected/Top-16 threshold.
- GT is exactly target slot 0.

Report for `overall`, `active_sparse` (`0<n<256`), `dense` (`n>=256`) and `zero_point` (`n=0`):

- sample count and target-point-count sum/mean;
- selected Top-1 hits and accuracy at IoU 0.25 and 0.50;
- Top-16 oracle hits and accuracy at IoU 0.25 and 0.50;
- Hungarian target-0 matched-query IoU sum/mean and hits at 0.25/0.50;
- matched-query center L1 sum/mean and size L1 sum/mean.

All values must be finite. Counts must be nested (`hits050 <= hits025 <= sample_count`). Holdout row identity must equal the frozen fold-2 holdout identity for all three roles.

## 6. Success gate

All conditions must pass:

1. Method active-sparse Top-16 oracle@0.25 hits strictly exceed control.
2. Method active-sparse Hungarian matched-target IoU mean strictly exceeds control.
3. Method overall selected REC@0.25 and REC@0.50 are each not lower than control.
4. Method active-sparse selected REC@0.25 is not lower than control.
5. Relative to parent, method must not simultaneously lower overall selected REC@0.25 and active-sparse Top-16 oracle@0.25.
6. Control and method consume exactly the same 1,600 unique fit rows; every role evaluates exactly the same 6,329 unique holdout rows.
7. The run is finite, produces zero `.pth`, accesses zero formal 7,899-row validation samples and leaves protected E57/GF SHA unchanged.

Any failure seals the method at this budget. No retry with another fold, seed, weight, threshold, size coefficient, learning rate, epoch or batch budget is allowed.

## 7. Execution and publication boundary

- One fixed O_EXCL outer root contains the three serial roles and one final paired decision.
- One global GPU0 lock covers snapshot construction, all three roles and postflight verification.
- Each role writes one raw audit receipt. The decision binds all receipt bytes/SHA, code/input manifests, commands and protected source SHA.
- Training returns before validation/checkpoint-save logic. Parent returns before training logic. Generated `.pth` count must be zero.
- The decision always writes `audit_only=true` and `long_training_authorized=false`, whether the gate passes or fails.
- A pass authorizes only independent review of whether a later experiment should be proposed. It does not authorize long training or formal Nr3D validation.

## 8. Explicit exclusions

- No external baseline fair reproduction.
- No experiment/section 7 or 8 method, schedule, prerequisite or conclusion.
- No old E0--E7 matrix.
- No FPR-TV, raw parser, relation-CF, SACR or dataset-ID/subgroup-label combination.
- No formal Nr3D 7,899-row validation, no Sr3D/ScanRefer evaluation, no checkpoint retention.
- No learning-rate decay decision is made from this short audit.
