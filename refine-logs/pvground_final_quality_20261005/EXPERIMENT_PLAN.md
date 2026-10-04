# Actual final geometry quality on the native PV-Ground decision

Status: implementation draft; no new GPU preflight or training yet.

Protected parent: official PV -> original G -> frozen distribution geometry,
5616/4506 on the installed 9508-row ScanRefer developer validation. Preserve
all reconstruction parents and historical V99. Single deployed native bbs;
all 256 queries; selected Query supplies both Box and Mask.

The completed, frozen-parent evidence-visible readback control is 5615/4477.
Its 96672-parameter readback was trained with native+G only. The final-quality
arm starts afresh from the same 4506 parent, the same seed-2027 zero-output R,
and the same AdamW. It does not resume the removed negative terminal.
No R expansion, teacher, contrastive-positive expansion or denominator sweep.

One new term fits centered native bbs differences to centered detached final
Box/root IoU differences. Native bbs is signed token evidence, not a class
probability. For each expression, the pool is the actual Hungarian matched
root plus exactly the unmatched IoU>.5 queries accepted by original G. All
queries matched to other GTs stay outside this term. This eligibility is a
training geometry proxy; it is not a claim of independently known semantic
identity and is never used by inference.

For pool C, loss per sample is mean_{i in C}[(s_i-u_i)-mean_C(s-u)]^2.
It equals half the mean squared error over ordered pairwise differences.
The per-sample reduction gives the new term an independent weight budget;
native CE/G/contrastive normalization is unchanged. Weight is fixed at 1.0,
chosen before any new validation. All256 candidates remain available.

Control reuse: the actual completed evidence-visible/native+G fit, not its
historical single best column. Compare the new arm to that full 3723-update
trace and also to protected4506. Training row order must match exactly;
geometry/Mask parent states remain frozen/eval. Floating outputs from
independent complete forwards are not promised bitwise identical.

Budget: batch8, accumulation1, 29778 fit rows once, 3722 full batches plus a
2-row tail =3723 AdamW updates; lr1e-5, decay.0005, clip.1. Same 6887
pretrained-seen module holdout, then independent native9508 evaluation.
Estimated ~110 minutes fit+holdout and ~25 minutes formal, based on the actual
completed identical-structure control. No batch/LR/budget change.

Before formal launch: fresh source review, real batch two-update preflight,
independent pairwise loss/gradient equivalence, exact zero gradient outside
pool, high-vs-low IoU direction, isolated quality gradient reaching R,
same-frame geometry/Mask preservation, strict in-memory model/AdamW restore,
and capacity/storage check. Preflight makes no accuracy claim and saves no
weight. Warm existing environment, no package rebuild.

Primary selection: native Acc@.50. Record .25 and same-Query Mask, fixes and
damages, direct cached-head diagnostic and candidate availability. Only a
completed superior terminal is retained; nonbest owned delta is removed
after formal restore/CPU checks, without creating another weight archive.
Target remains same model5615/4754. Nr/Sr and novelty claims remain unproven.
