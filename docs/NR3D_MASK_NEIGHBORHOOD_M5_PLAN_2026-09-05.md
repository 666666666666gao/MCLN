# M5: matched short training of the existing Mask feature projections

M4 identified31 majority-foreground superpoints with no seed inside radius.2;
they read seed0 at3.09–6.16m while a nearest seed was.200–.285m away. Its16-row
inference gain was concentrated:6 Masks improved,6 worsened,4 were unchanged.
This warrants a controlled learning screen, not a model promotion. The original
upstream MCLN has the same grouping configuration, so this is not an identified
regression introduced by the added modules.

Two arms start from the exact protected averaged checkpoint and frozen source:
original ball grouping and nearest-two grouping from M4. Train only existing
x_query, x_mask and rel_encoder parameters (16 tensors); freeze all remaining
parameters and buffers, including Decoder, Box heads, SWA and score selector.
Use model.eval() with autograd enabled so frozen dropout/normalization stays
fixed. There are no new heads, GT inputs to forward, fusion changes, extra
losses, radius/K choices, or candidate-selection changes.

Keep the existing P2/R1 scene split salt and all6172 held-out expressions from
98 training scenes. Select the first2048 fit expressions in CSV order, before
new predictions. The held-out scenes were seen by the frozen base network;
call this a Mask-module holdout, never whole-system unseen-scene generalization.
Fit and holdout scenes must be disjoint. No formal validation is constructed.

Use the same batches for both arms: batch4, no augmentation, two epochs,
epoch-specific seed0/1 shuffling, exactly1024 optimizer updates per arm. Use
fresh AdamW, fixed LR1e-5, saved-checkpoint weight decay and gradient clipping;
record their numeric values before any update. The actual TrainTester loss and
Hungarian matching are unchanged. Grounding loss terms have no parameter path
to the trainable Mask projections. Verify all trainable gradients are connected
and finite before stepping. Neither arm's frozen parameters/buffers may change.

Before training, evaluate both arms once on the full fixed6172-row holdout.
After the fixed1024 updates, evaluate both once again, with identical input
order, batch16, seed1000 and native thresholds/selection. Save per-expression
REC, selected Mask IoU and raw REC/input hashes. Require exact Box/score parity
between arms and exact grounding hashes from before to after training.
Save only trained Mask projection states and optimizer states in the isolated
remote directory; never overwrite or publish a protected/full checkpoint.

Screen the fixed terminal nearest-two arm against BOTH the terminal original
control and the starting protected-original arm: selected fused Mask mIoU must
improve by at least.002 absolute IoU (.2 percentage points), and Mask@.25/.50
hit counts must not decrease. Integrity checks and REC identity must all pass.
Report fixes/breaks and per-scene results even if the screen fails. Do not select
an earlier epoch, sweep hyperparameters, or substitute GT-oracle scores.

A passing result only supports a larger reproducibility study. This Mask-only
screen cannot raise Nr3D REC past60%; the REC relation/representation objective
remains a separate unresolved requirement. R1/P2 failures stay sealed. No
ScanRefer/Sr3D production path is altered by this isolated experiment.
