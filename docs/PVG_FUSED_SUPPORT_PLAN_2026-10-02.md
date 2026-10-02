# PV-Ground: same-tail raw versus predicted fused-Mask support

Status: preparation only. Current raw-point P3 continues unchanged. No new GPU run, terminal REC, teacher, quality loss, or boundary-distribution result exists.

User direction: use PV-Ground as baseline, prioritize ScanRefer Acc@0.5, transfer useful predicted spatial support and geometry into training; do not restore V99 rankings or inference side chains. Original verified G5615/4495 remains the common start. P2 endpoints are negative historical evidence and are not initialization.

## One controlled change

Two fresh adaptations start from the same original G, with the same added head initialization, native G loss, batches, optimizer, budget and final native bbs score. Both execute the refiner once AFTER native Text Mask, Query Mask and alpha generation. Both use the same 14-channel member encoder; tail_raw supplies zero in the four support slots, while tail_fused supplies predicted Text probability, Query probability, their absolute disagreement and native fused-logit probability. The four slots are the sole between-arm input difference. This matches parameter capacity and execution order. The older 10-channel pre-Mask P3 is an additional historical control, not the independent fused-support comparison.

Preserve current center+six face locations, 16 nearest distinct input indices, RGB/relative positions/distance, mean/max pooling, geometry Query conditioning, additive six-dimensional box residual, zero output initialization and native last/bbs. No radius search, hard threshold/crop, extra score, evidence readback, boundary distribution, teacher or new loss. Preserve coarse-box regression prior and all native semantic/contrastive/Mask call counts and order. Neighborhood locations are detached as in P3; predicted Mask values remain differentiable. Thus geometric loss gains a path to native Mask parameters in the fused arm; this is part of the specified intervention and must be recorded, not called complete task isolation.

## Exact source dependency

Copy sealed September17 original G/D source into a NEW isolated directory. Save last geometry_query and the undetached coarse last_center/last_pred_size. Native loop may continue its existing detached base tensors, but the final refiner reads the original end_points values so native final-box loss still trains the original box head. After native prediction_masks, sp_pred_masks, adaptive_weights and superpoint IDs exist, run the refiner once and overwrite native last_center/last_pred_size. Do not regenerate Mask from refined Query or rerun semantic heads.

Native fusion is alpha*last_pred_masks[batch][0,query,superpoint] + (1-alpha)*sp_last_pred_masks[batch][query,superpoint]. Here last_pred_masks is the expanded Text branch. Alpha follows its actual native tensor shape. Map only selected raw-point indices through the corresponding per-input superpoint IDs; do not allocate 256x50000 Mask tensors or use GT masks/GT identities as network inputs. The original augmented points and membership mapping must remain aligned.

## Fixed protocol and acceptance

seed2027, physical/effective batch8/8, fresh AdamW, core/backbone LR1e-5, weight decay5e-4, clip0.1; original29778 fit rows once/3723 updates;6887 module holdout and9508 development validation. The current long-used9508 split is development validation. No new Nr3D/Sr3D run until the structure succeeds and is fixed.

Each arm records full E0 chosen Query/box/IoU and input/GT identities, initial Mask differences, all fit-batch IDs, selected coarse/refined boxes, repair/damage and executed coverage flags. Do not treat the old G continuation as the sole control: the two same-tail arms are required. Do not change the already-running P3 protocol or retrospectively select its checkpoints.

Real preflight after the current P3 ends: native call order/count preserved, strict original G restore, zero box/semantic parity, exact selected-member Mask/logit mapping versus the same native tensors, no GT input, true loss on refined native outputs, direct geometric gradients to refiner and (once internal weights update) Mask support, no direct Mask/semantic loss to the refiner, save/reload, actual peak CUDA and serialized size. Only actual passed runtime gates justify training. Disk space is measured from serialized size; no deletion is authorized by this plan.

Primary result is Acc@0.5 against same-tail raw and original G; report Acc@0.25 and native Mask with repairs/damages. Aim4754/9508=50.0% and preserve5615 loose hits; these are unmet targets, not promised effects. Historical V994797 strict hits is reference only. Final attribution requires completed results and fresh integrity review.

## Deferred

Boundary-distribution regression, final-quality supervision, G/contrastive responsibility changes and V99 training teacher are separate later experiments, conditional on actual support evidence. They are not enabled here.
