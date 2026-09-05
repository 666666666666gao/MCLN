# Existing object-input appearance availability: CPU-only audit

R1's actual object memory contains128D box-position and160D predicted-class
features, without independent object appearance pooling. R1 failed its fixed
screens; this audit does not reopen its trained pair-readout variant.

The original ReferIt3D listener and MVT encode individual objects' colored
point clouds before language fusion. Their segmented-object inputs cannot be
silently added to our current MCLN protocol. Inspect only point crops inside
the already supplied butd_cls object boxes; instance memberships are used
solely to measure crop contamination, never as new model inputs.

Use the same16 preselected M3/M4 fit expressions and immutable source/data.
Reconstruct the original CPU dataset inputs with seed0 and augmentation off,
and require their point hashes to equal M3. Reuse the already measured M4
seed indices/coordinates instead of running a backbone. No model, optimizer,
GPU forward, holdout data, crop-margin tuning or new scoring head.

For every valid object slot record existing box/predicted class, raw point
count inside exact float32 bounds, original1024-seed count in that box, and
mean/std of native input RGB when the crop is nonempty. Audit instance-point
count, crop purity, and true-instance seed membership separately. Count empty
raw crops, empty seed crops that still contain at least32 raw points, and
nonpositive box dimensions. These are availability/contamination diagnostics,
not learned-feature quality or model accuracy.

Verify that the first four rows reproduce R1's190 object slots and155 correct
predicted classes. A result may support a new object-appearance experiment
using existing inputs; it cannot establish its REC benefit. Preserve M5,
protected source/checkpoints, candidate Query axes and benchmark protocols.

Primary implementation references:

- https://github.com/referit3d/referit3d/blob/eccv/referit3d/models/referit3d_net.py
- https://github.com/sega-hsj/MVT-3DVG/blob/main/referit3d/models/referit3d_net.py
