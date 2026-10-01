# PV-Ground pretrained core + CS improvements

**Superseded before formal training.** The later user clarification selects
the verified PV-Ground+G endpoint and a G vs G+P2 same-budget comparison; see
`PVG_G_P2_PLAN_2026-10-02.md`. This wholesale-port draft passed a two-step GPU
engineering probe only. It has no trained accuracy and is not the active plan.

User decision (2026-10-02): replace the MCLN E71 starting core with the author's
PV-Ground pretrained ScanRefer weights and adapt our improvements. Old MCLN
experiments remain historical records; their active training can be superseded
once this isolated implementation passes its launch checks.

## Fixed first experiment

- Official PV-Ground ScanRefer checkpoint, epoch 81, 1234 core tensors. Fresh
  optimizer and epoch 0; do not load the historical PV+D+G adapted endpoint.
- Preserve PV-Ground's voxel and point backbones, language encoder, queries,
  decoder, box/semantic heads, text and query segmentation and native mask fusion.
- M1: zero-residual seed and superpoint enhancement using the six native VSA
  feature sources (BEV, raw points, x_conv1..4), actual raw/FPS superpoint
  membership, and encoded seed features. Counts describe member availability;
  they are not calibrated VSA neighborhood quality or original MCLN SA1/SA2.
- M2: original full-scene plus nearest-16-superpoint query reader, last two layers.
- M3: original Query-Mask support box refiner, last two layers.
- R: existing three-role geometry evidence readback, final semantic head only;
  contrastive projections and final masks use the query before R. The semantic
  head executes once. Keep an independent `cs` arm for a later R control.
- Original PV-Ground losses only. No G, quality loss, teacher, dual ranking,
  hard mask-to-box replacement, or inference sidechain in this first experiment.
- Both native and improved arms use the same deterministic superpoint centers.
  M1 alone has the added deterministic raw/FPS member-mean branches. The shared
  center numerical change is not credited to R or M1.
- Single A100 40GB, seed 2027, 21 epochs, effective batch 12 (capacity must be
  measured). Same LR recipe: new/core/backbone = 1e-4/2e-5/2e-6 times sqrt(B/16),
  weight decay 0.0005, global gradient clip 0.1. Epoch 1 core/backbone LR 0.1x;
  epoch 2 onward uses the existing cosine schedule. Frozen RoBERTa remains frozen.
- ScanRefer + joint ScanNet detection training, original intermediate object
  supervision and two-stage detected-object input. Full 9508 expression validation,
  native last/bbs top-1 IoU against GT. Report a newly measured E0, every epoch,
  preset best and fixed endpoint. Do not inherit paper numbers as measured results.
- Keep the existing preset best comparator: min(hits025/5572, hits050/4797), then
  total hits. These are V99 reference scales, not a loss or the development gate.
- Save mutable model state and optimizer as an exact delta against the immutable
  official parent, retaining best/latest only; restore parent then complete delta.

## Run order

1. Strict CPU parent-load and source/shape checks; fresh code review.
2. Preserve R weights/logs, stop its superseded training and queued diagnosis.
3. Real GPU zero-init comparison with identical RNG (native Gumbel sampling is
   stochastic even in eval), two update steps, module gradients, actual batch
   capacity, and delta/optimizer save + reload. No formal result from this probe.
4. Start from the official parent again. Full E0 validation, then 21-epoch training.
5. Evaluate gain versus its own PV-Ground E0 and corresponding controls. ScanRefer
   development gate remains 58.3%/50.0% (5544/4754 hits). Fix the method before
   independently training Nr3D and Sr3D. No promise of reaching that gate.

Runtime: reuse the verified isolated PV-Ground Torch1.10.2/spconv runtime and
the existing ordered-VSA source. No new environment or backbone replacement.
Historical V99 and MCLN weights are protected. This baseline switch supersedes
the previous restriction requiring MCLN as the only starting core.
