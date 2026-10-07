# Actual-face member reading: focused ScanRefer contrast

Status: sampler source prepared; shared-forward runner, SOURCE review and actual
two-step PV GPU preflight still required. No training result or module claim.

Keep PV-Ground/original G, the actual retained5598/4848 predicted fused-Mask
reference, the456102-parameter/10-state distribution geometry head, its109-D
range evidence, the native score, all256 queries, and same Query Box/Mask.
Do not load either failed reference-keep terminal. Keep weight is0; existing
native+G, matched DFL and Query-supported extra geometry remain common.

## One changed structural input

- Control `face_center`: existing center and six face-center KNN,16 members each.
- Method `face_region`: same center KNN; six faces read16 points nearest the
  bounded face rectangle. Normal distance plus tangential distance outside the
  rectangle determines selection. Inside and nearby outside observations are
  both allowed. The same XYZ/RGB/relative positions/Mask channels and Euclidean
  face-center distance are then encoded by the unchanged14-D member MLP.
- No new parameter, radius scan, probability threshold, proposal pruning,
  reference-selection rule, text attention, quality score, teacher, loss or LR.

The retained version lost191 wide-threshold hits relative to its own native
coarse boxes;128 of those cover>=95% GT volume but have>4x GT volume. This is
evidence of overextended references, not proof that face-center KNN caused all
errors, that particular points are outliers, or that128 hits are recoverable.
The trial asks whether observations across a face help the current decoder
learn useful corrections while preserving the strong reference.

## Paired frozen-parent protocol

The actual closed reference-keep evidence contains cross-forward all256 reference/validity
and a few Mask-IoU differences despite frozen states. Thus the new contrast
must share one actual frozen parent forward per augmented fit batch and per
validation batch. Each geometry head has its own optimizer and own losses;
the same query, raw points, native coarse prediction, Mask/support and native
score are supplied to both. Same-forward does not mean identical geometry
features: differing face members are the intended method variable.

The runner must explicitly verify the all-zero R, detach the frozen shared
semantic score from the control geometry graph where necessary, preserve
the original head invocation/semantic order, and avoid cross-head gradients.
The exact endpoint dependencies must be read from actual PV code, then checked
in the real two-update GPU preflight. No guessed recursive output compatibility
layer or alternate scoring path. This shared experimental parent is not a
second deployed head: each saved model rebuilds only its declared sampler.

## Engineering gates before fit

1. SOURCE review of sampler, paired runner and actual dependency/order graph.
2. Independent rectangle-selection check on actual raw points, exact original
   center member selection, neutral output/reference identity, same native
   score/Mask, correct two loss/optimizer graphs, nonzero head gradient/update,
   parent/R protection, actual save/restore, measured memory/throughput.
3. Fresh initial heads from the retained state; never continue preflight state.
4. Actual available disk reserve. Existing candidate arrays are archived, but
   a new precise array-deletion approval is still required before removal.

## Fixed fit and claim scope

After gates, each head sees the same29778 expressions exactly once and makes
3723 updates. B8, accumulation1, LR1e-5, WD5e-4, clip0.1, seed2027. Same-forward
validation of6887 seen-scene holdout and9508 development expressions; record
parent reference and both finals, all256 coverage, repairs/damages, Mask and
sampling distances. Budget/history are disclosed. Do not reinterpret seen
training-scene holdout as formal generalization or claim a new inference score.
Persist raw all256 arrays for both9508 formal stages in one paired NPZ per batch.
The6887 holdout retains selected-row JSON and all256 coverage summaries, following
the existing control protocol; no holdout raw-array collection is added.

First compare method versus same-budget control, then versus the same-forward
neutral reference and retained5598/4848 state. Both user thresholds must hold
for one final model: at least5620/4764. Three genuinely effective contributions
remain required; this source or a neutral-output pass establishes none. Only
then freeze the complete method and independently train Sr3D/Nr3D from each
author's corresponding weights, with corresponding native control and seed2027.
