# Coarse geometry input-scale diagnostic

No new accuracy claim and no optimization step. The completed support-pair experiment and content checkpoint remain unchanged.

Eight previously recorded training batches (steps1,2,522,523,524,3211,3722,3723) provide58 unique fit rows. Fresh augmented inputs and fresh frozen-parent forwards are used; this does not reproduce the old augmentation/Gumbel RNG, learned geometry-head state or original step523 spike.

The exact existing official/G/Mask-reference parent is rebuilt in its warm runtime. Every batch has one frozen parent forward and actual parent Hungarian-to-valid-GT correspondences. All256 candidates are retained. Three identically initialized zero-output derivative probes use content-only, original9 geometry channels and signed-log compressed9 channels respectively. No optimizer.step, clipping, checkpoint write, candidate ranking change, teacher or new GT inference input.

Measure actual coarse-size floors for all and matched Queries; each Query's largest geometric feature, matched geometric feature amplitudes, native5/1/10/2 Mask loss and parameter gradient norms. Assert identical zero-output Masks, identical losses, finite gradients and unchanged parent/head states. The compression is an input diagnostic, not a selected deployed module.

Existing canonical environment SHA966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c, Python3.7/Torch1.10.2cu111, one A10040GB, no packages installed. Actual capacity and lock required before launch. First observation420seconds after launch; if still live,240second intervals. Preserve primary failures; never restart on observer timeout.

Required source check occurs before executing. Actual completion will be collected independently. No training or formal9508 validation is authorized by this diagnostic script. Best content5598/4856 and required author/official/G weights remain protected. Research gates5620/4764,3effective contributions, then fixed full model author-pretrained independent Nr/Sr remain ACTIVE_UNMET.
