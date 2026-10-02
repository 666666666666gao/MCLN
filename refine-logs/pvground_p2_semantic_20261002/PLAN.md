# P2 semantic routing control

This is one minimal routing experiment on the verified PV-Ground+G start.
The completed joint-P2 training result was formal bbs5613/4419 versus the
same-budget G control5600/4452. Fixed actual P2 boxes gave only+1/+3 hits for
P2 semantic reading over bypass, with99/9508 selected-query changes. It does
not establish why adaptation lost original-G capability. Original G5615/4495
remains protected and is the starting model, not the new P2 endpoint.

Only change the last decoder routing:

    joint P2:    (G_semantic + P2_evidence, G_geometry + P2_evidence)
    this control:(G_semantic + P2_evidence, G_geometry)

The unchanged P2 reads full text and existing six-source observation memories.
It still feeds native semantic CE and contrastive outputs. Geometry and Query
Mask use G's existing geometry path. This removes direct P2 evidence addition
to that path; it does not freeze upstream/shared parameters or prove that
semantic gradients cannot affect their training. No new module, P3, teacher,
TGS/RoBERTa change, loss-temperature change, score or dataset inference gate.

Use the original G endpoint SHA0575dfae...7964522, fresh AdamW, seed2027,
batch8, LR/backboneLR1e-5, weightdecay5e-4, clip0.1. Same29778 fit rows each
once,3723updates; same6887 module holdout, then9508 ScanRefer development
validation under GroupFree predicted-object-assisted two-stage last/bbs.
Keep G's exact CE label-replacement rule and original native matching/Mask loss.

Reuse the completed, sealed G control and joint-P2 comparison from the SAME
start/config/budget; do not rerun them. Verify new full training row order and
initial evaluation against those saved controls. A failed input/order check
invalidates the comparison instead of being silently fixed. Initial Mask
differences previously observed remain disclosed; their cause is unresolved. No whole-output
bitwise identity claim. Reusing a control is explicitly recorded, not a new
three-arm simultaneous trial.

Fresh code review precedes deployment. Real2-update preflight must restore G,
test zero-residual starting behavior, show P2 semantic-task gradients and
zero bbox/giou gradients to P2 parameters under the new routing, and serialize
and reload mutable state/Adam IN MEMORY. That gradient probe establishes the
current direct computational routing, not protection of all shared geometry.
Only after preflight passes and measured checkpoint-replacement space is
available can training start. Use the existing single-A100 flock lock.

Store the new run on the data disk: /root/autodl-tmp/pvground_p2_semantic_20261002.
Known09:46 free bytes: data876249088; system501379072. Retain all existing
G/V99/paired endpoints and diagnostic rows; no cleanup or deletion. Reserve
two measured mutable-state copies plus128MiB for actual atomic replacement.

Criteria: report both terminal holdout and full formal bbs against originalG,
continuedG and jointP2; do not choose an output mode or checkpoint post hoc.
Promotion requires preserving originalG5615@0.25/4495@0.50 and positive strict
gain against continuedG; ScanRefer progression target remains at least
58.3%/50.0%(5544/4754). P3 and Nr3D/Sr3D remain dependent on actual evidence.

Estimate3h for training including initial/terminal holdout, plus18min formal
evaluation from the previous actual timings. Check the short preflight after
about5min, then initial/training startup at a meaningful boundary. For long
training, schedule an observer near its estimated completion; if unfinished,
wait180-300s between observations. No unchanged retries or parallel GPU jobs.
