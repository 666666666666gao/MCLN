# R1: reference-memory and relation-readout factorial control

Fixed before any new head updates or holdout scoring. P2 v1 remains failed.
The current P1 native-path diagnosis does not confer a pass on that variant.

## Evidence and one bounded structural question

The complete P1 diagnostic found qualifying Full256 boxes for 2995 of 3421
current REC errors, while legal Query memory covers only 32.72% of the
expression-repeated object slots. This is not labelled text-anchor recall.
A four-fit-row probe on original P2 fit IDs 0/1/3/4 confirmed that the actual
Decoder object features and masks match the existing input branch exactly.
Its 190 active object slots have 155 correct predicted classes, while full
Query memory covers 38 slots. The input features are 128D box-position plus
160D predicted-class encoding, not separately pooled object appearance.

R1 asks whether the missing reference evidence limits the candidate-pair
readout. Use the existing `butd_cls` object features and boxes as an alternative
reference memory, without supplying new GT classes, anchor assignments or
target labels to inference. Original allowed object-box inputs remain the
same benchmark protocol. The model is not described as GT-object-free.

## Four shared-input arms

| Arm | Reference memory | Relation text context |
|---|---|---|
| query_global | All legal Queries + null | Masked global sentence pool |
| query_pair | All legal Queries + null | Full token sequence per ordered pair |
| object_global | All valid input object slots + null | Masked global sentence pool |
| object_pair | All valid input object slots + null | Full token sequence per ordered pair |

The primary candidate is `object_pair`; the other arms are predefined controls.
Do not select whichever control scores best after the run as a replacement
primary candidate. Report all four arms even when the primary fails.

`CandidateReferenceScorer` retains the exact P2 parameters and scoring function,
but accepts a separate reference axis. Query-memory behavior must match the
original scorer on synthetic and real fit tensors. CPU tests also verify
memory-slot permutation, masked padding, null reference, and feature gradients.
Counts remain 347953 (global) and 931729 (pair). A memory comparison within each
readout is parameter matched. Global versus pair still includes the additional
pair-reading parameters, so do not claim a parameter-count-matched text ablation.

## Fixed data, optimization, and output contract

- Protected checkpoint SHA `76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`;
  unchanged immutable G0 fixed-source manifest `dcf333b0e1868a7eeaafaf7f0a7abdb664a34dda65966defc1ad244ce762b15d`.
- Reuse the P2 v1 scene split without a new salt: 26747 fit rows / 413 scenes,
  6172 holdout rows / 98 scenes. These are new-head holdout data already seen
  by the frozen backbone, not unseen-system or official validation.
- The same frozen backbone forward serves all four arms. The actual last
  Decoder's `detected_feats` and `detected_mask` are observed without replacing
  outputs. Candidate features, boxes, post-encoder tokens, legal Default
  Top32 targets and five-dimensional pair geometry remain unchanged.
- One full fit epoch, B4, four workers, no augmentation, seed9 initialization,
  loader seed0, AdamW LR1e-4, weight decay1e-4, clip1, dropout0, no scheduler.
  Common parameters initialize identically; spatial weights come from the
  protected final Decoder layer. All four arms receive the same update count.
- Preserve the P2 IoU-weighted listwise CE on valid candidates above .25.
  Uncovered rows have no ranking loss; an uncovered batch skips every arm.
  No new loss, feature encoder, residual score, switch gate or box/Mask update.
- One uncalibrated scalar score per arm; same legal Query mapping in train
  and evaluation. Mask evaluation uses that arm's selected Query; protected
  Mask retains its native selection. Save final addon heads only.
- Use native mask-before-sort Query selection and float64 point-mask IoU in
  every evaluation arm, incorporating P1's resolved output details. The Query
  controls reproduce the old readout function, but this run is not claimed
  to reproduce historical P2 aggregate metrics if tied-score selection changes.
- First run one actual fit-batch forward/backward with zero optimizer steps,
  checking finite gradients and old Query-readout equivalence. Stop for an
  actual integrity/nonfinite failure. No automated restart or parameter sweep.
- Evaluate every holdout expression exactly once after the full fit epoch,
  seed1000. No intermediate score inspection, epoch choice or formal validation.

## Decision fixed before results

Require all three screens for `object_pair` to advance:

1. Versus `query_pair`: positive REC@.25 net hits overall, in CSV-token length
   >=13, and in actual 2+ distractors; nonnegative overall REC@.50.
2. Versus `object_global`: the same requirements. This tests the full pair
   reading design against the simpler readout with the same object memory.
3. Versus protected output: positive overall REC@.25, nonnegative REC@.50,
   and no lower mean Mask IoU, with its actual protected Mask path.

All source/state/order/update and evaluator parity checks must also pass.
Controls are diagnostic even if one wins. A failed primary is sealed without
silently changing these gates or promoting a control. These exploratory screens
do not establish statistical significance, formal improvement or cross-dataset
generalization. The overall three-benchmark goal remains separate and unmet.

Report all REC/Mask metrics and fix/break counts. Use the raw CSV token column
for length groups. Add paired whole-scene bootstrap intervals (2000 draws,
seed20260905) for memory effects within each readout, readout effects within
each memory, the interaction, and primary versus protected. Keep all expressions
of a sampled scene and calculate expression-weighted ratios. Do not change
the above screens based on interval selection.

Use the existing GPU lock and an isolated experiment directory. Estimate
roughly 60–75 minutes from the earlier P2 throughput, then update from the
first logged fit milestone and poll near completion. Original model source,
protected checkpoints, P2 v1 archives and ScanRefer/Sr3D results stay intact.

## Terminal status, 2026-09-05 18:38 CST

All four arms completed6687 updates and6172 holdout rows. Independent verification PASS; all three registered object_pair screens FAIL. No Decoder integration or formal promotion. See `NR3D_REFERENCE_MEMORY_R1_RESULT_2026-09-05.md` and sealed terminal evidence. The initial CPU verifier cwd error and corrected v2 are both retained; no model rerun.
