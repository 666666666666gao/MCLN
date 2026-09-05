# P2 v1: one frozen-backbone relation-readout comparison

This plan is fixed before P2 split counts, supervised head updates or held-out
scores are observed. It is a separate response to the user's narrowed relation
proposal. G0 failed its own augmentation gate and remains closed.

The code contract is `scripts/nr3d_pair_readout_contract.py`. Scene assignment
uses salt `MCLN-NR3D-PAIR-READOUT-V1-20260905`, the first eight hexadecimal digits
of SHA256(salt + NUL + scan_id), modulo five; fold zero is holdout. Only the
32,919 official train expressions participate. The protected backbone has seen
these scenes: this is holdout for the new head, not unseen-system evaluation.
No salt, fold or sample count is selected after examining results.

## Fixed comparison

- Start from protected evaluation-only checkpoint SHA
  `76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`.
  Keep the immutable G0 fixed-source snapshot and its complete source manifest.
- Freeze the whole backbone in eval mode with no augmentation. Each batch has
  exactly one backbone forward shared by global and pair heads. The existing
  `butd_cls` object inputs and REC detector-overlap rule remain the protocol.
- Common 288D Queries, text_memory, final clamped boxes, Default legal Top-32
  targets, full legal Query memory plus null; unchanged five-dimensional geometry.
- Initialize both common submodules with seed 9 and the protected last spatial
  attention weights. Pair-specific modules are additional parameters, not a
  parameter-count-matched control. Both final scalar heads initialize equally.
- One complete fit epoch, batch four, four workers, fixed shuffle/worker seed 0.
  Both heads receive the same batch and one AdamW step per supervised batch:
  LR 1e-4, weight decay 1e-4, clip norm 1, no scheduler, dropout zero.
- The sole loss is listwise cross entropy with target mass proportional to IoU
  **only on legal Top-32 candidates whose root IoU is strictly greater than .25**.
  Other legal candidates are negatives. Uncovered rows contribute no ranking
  loss; if an entire batch is uncovered, both updates are skipped and counted.
  There is no bad-box positive, base residual, switch gate, replay or auxiliary loss.
- Evaluation uses the same single scalar logit, original Query mapping and
  legal candidate set. A row with no legal candidate is a miss, matching the
  existing evaluator; no fallback selection is added.
- Save only the two final addon heads in a new experiment directory. Do not
  modify protected weights or production inference. No intermediate holdout,
  epoch selection, LR/margin sweep or formal validation. Evaluate once after
  training with seed 1000 and include every held-out expression.

## Measurements and fixed interpretation

Report REC hits at strict IoU >.25 and >.50 for protected selector, Default,
global and pair, on the same frozen per-batch predictions. Validate protected
row hits against the existing evaluator. Report fixes/breaks both against the
global control and against protected selection. Split by raw CSV token count
(13+ for the long group) and actual dataset distractors (2+ for the hard group).
These names do not imply isolated causal effects of language length/distractors.

Record pre/post-filter Top-16/32/64/256 target coverage, zero-legal rows, target
point counts, selected-Query Mask IoU, correct-box-Query Mask IoU and the full
Query Mask oracle. Report object coverage for the full legal Query memory and
target Top-32 as an availability proxy, not annotated text-anchor recall.
Existing protected Mask selection is evaluated separately from its REC Query.
Pair and global use their one selected Query for both box and Mask.

The mechanism screen requires pair minus global: overall REC@.25 >0 net hits,
REC@.50 >=0, and REC@.25 >0 in both long and hard groups. The practical screen
requires pair minus protected: overall REC@.25 >0, REC@.50 >=0, and mean Mask IoU
no lower than protected Mask selection. Both screens and input/update integrity
must pass before advancing this variant to a limited Decoder experiment.
These are screening rules, not statistical significance claims or a guarantee
of benefit on 7,899-row formal validation. A failure ends this fixed variant;
it does not prove every relation model ineffective. Formal promotion is always
false for this train-scene audit. Any later experiment requires a new named
contract rather than silently adjusting this one.

Single-seed exploratory results cannot establish cross-benchmark validity.
If this mechanism advances, later independent runs need multiple seeds,
scene-cluster uncertainty, and the same structure on all three benchmarks.

## Operational checks

Use the existing shared GPU flock and require the completed source-bound P1
receipts plus G0 integrity receipt. G0's scientific pass is not claimed or required
for this independently requested experiment. First run a single fit-batch loss
and backward probe, zero updates, before launching the full fixed run. An actual
nonfinite loss/gradient or identity failure stops the job; no automatic restart,
parameter fallback or partial result promotion. Check launch once, use early
logged throughput to estimate duration, then poll near the estimated finish.
