# M3: native Mask matching, feature and gradient probe

This is a training-data-only mechanism check following the completed M2 audit.
It does not train a new model or reopen the failed R1 pair-readout screen.

Use the first16 distinct scenes encountered in the existing P2/R1 fit partition,
one expression per scene in CSV order. The split salt and scene fold remain
unchanged. Select these rows before any new predictions, using only identity
metadata. No fit/holdout reshuffling, formal validation, or outcome-based row
selection. Save all selected IDs and input hashes.

Run four B4 batches through the protected averaged checkpoint and immutable
source. Use seed0, no augmentation, model.eval(), with autograd enabled.
Evaluation mode keeps protected buffers/dropout fixed; this probes the native
loss graph on training data, not stochastic training dynamics. No optimizer is
constructed, no parameter updated and no checkpoint saved.

Call the actual TrainTester criterion and loss wrapper. Observe the last-layer
criterion inputs and its returned Hungarian indices without replacing them.
Keep native REC/Mask selection and GT-best legal Box identities separate.

Record per row:

- Number/identity of actual targets and matched Queries; native matched Box and
  three Mask IoUs, alongside native selections and good legal Box Queries.
- All256 Query Box/Mask qualities and which raw Query logits receive nonzero
  direct gradients from the existing combined Mask loss. Count good Box Queries
  without direct Mask supervision, without calling ordinary one-to-one matching
  a bug or automatically recommending extra positive labels.
- Mask-loss and native grounding-loss gradients at the actual final Decoder
  Query tensor: per-Query norms, cosine where both gradients are nonzero, and
  projection parameter/feature connections. Observe disconnected gradients
  explicitly; do not fabricate zeros as evidence of a connected path.
- Gradients through the shared seed Mask projection and relative-coordinate
  encoding, plus the final Box head as a separate path. Gradient existence is
  not evidence of successful learning or generalization.
- Majority GT superpoint targets, target-point counts, and the existing
  superpoint grouper's actual seed-neighbor indices. For target-bearing/majority
  positive superpoints, count neighborhoods without a seed center in the target.
  Seed-center membership is not receptive-field coverage.

Mask objective uses the original text/query/fusion focal and Dice terms and
consistency coefficients. Grounding objective uses the original accumulated
classification, Box and contrastive terms, divided by decoder_layers+1. Neither
loss, its weighting nor its targets are changed. The probe observes the actual
graph rather than injecting new differentiable leaf features into the network.

Assertions check finite forward/loss/connected gradients, Query-logit equality
to the observed query-feature/superpoint dot product, dataset identities, and
protected parameters/buffers plus source/data hashes before and after. CPU
tests cover direct gradient support versus Hungarian indices and neighborhood
membership counting; the complete real-data run remains the required check.

Do not infer population-level frequencies from16 selected training examples.
If the existing gradient and indexing paths work, record that fact and use the
remaining evidence to formulate one matched structural/loss experiment. A probe
pass alone does not establish an improved model. Preserve all three benchmark
best artifacts and keep the overall objective active until they truly improve.
