# Fixed ScanRefer empty-neighborhood pooling control

This is a zero-new-parameter control, not a validated coverage-aware method.
The corrected PV-Ground VSA source and aligned mesh/detection inputs remain fixed.
For each actual StackSAModuleMSG radius, keep the real CUDA empty-neighborhood mask
and zero its pooled vector after the original MLP/BN/ReLU/max pooling. Supported
neighborhoods retain the original computation. No source-ranking or candidate gate
is introduced. BN behavior, native REC and Mask losses, evaluator, and data remain native.

Evidence before launch: the 16-fit-scene census found actual augmented sparse-source
empty neighborhoods while raw-point neighborhoods retained support; zero grouped
input can produce nonzero pooled output through the pretrained MLP. This may encode
absence and is not by itself a numerical bug. Real CUDA supported/empty tests,
nonempty full-model exact replay, and two ephemeral full-native-loss updates passed.

Start from the official ScanRefer epoch81 parent, never from ephemeral check weights
or a failed endpoint. Use seed2027, 29778 fit expressions, 6887 fixed module-holdout
expressions, batch8, LR/backbone LR1e-5, one fit pass/3723 updates. No seed search,
extra epochs, loss changes, or branch-specific hyperparameter search. The prior
completed VSA-corrected native trial is the same-budget control (terminal bbs6119/5537).
Report both the new arm's own initial-to-terminal change and its difference from that
control; initial differences must not be described as training gain. Holdout scenes
were seen during upstream pretraining and do not establish unseen-scene generalization.

Primary output remains native bbs; bbf is diagnostic. Independent CPU audit recounts
all 6887 rows/candidate arrays. Existing own-initial two-threshold nonregression gates
the queued formal evaluation; formal acceptance still requires V99 REC5572/4797 of9508
and ScanRefer Mask58.70/50.70/44.72 percent. Nr/Sr training waits for Scan acceptance;
Nr/Sr Mask is not an acceptance gate but its native training loss is retained.

The control's boolean is not a state_dict entry. Train spec, checkpoint, receipt and
formal evaluator must bind the module SHA and explicit architecture flag. Formal
published-parent arm disables this control; terminal arm enables it. State keys and
parameter shapes remain exact. Inference still uses no MCLN Parent/Geometry/V99 sidecars.

Disk: reuse all parent weights and source directories. Reserve 1.25GiB for two peak
331MB delta checkpoints plus holdout/formal arrays and logs, and 1GiB free reserve.
Only one latest checkpoint is overwritten atomically; preserve protected parents and
fixed endpoints. No duplicate full parent or optimizer history files are created.
Queue checks every300s after estimated milestones; never start duplicate jobs.

Runtime is warm-reused under the unchanged env_spec SHA966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c.
No packages, compiled kernels, or base runtime files change. The module preserves its
OpenPCDet Apache2.0 source attribution (see licenses/OpenPCDet-LICENSE).
