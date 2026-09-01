# FPR-TV implementation contract (2026-08-31)

## Objective

Implement the first isolated ranking route for MCLN:

`Parent + compact Top-K candidates -> candidate-vs-parent text verifier -> one discrete switch or exact parent fallback`.

The first phase changes only final REC selection. Box, Mask, the detector,
the decoder, and the existing V99 parent scores remain frozen.

## Required behavior

1. The parent is the exact argmax of the deployed V99 parent scores after the
   formal `butd_cls` detector-overlap filter. Raw Top-1 queries outside that
   evaluator-valid axis are never parents or promotion candidates.
2. The compact candidate set contains the parent plus the union of fixed
   Top-K parent-score and text/contrastive-score candidates. Candidate order
   and de-duplication are deterministic, and both Top-K lists are restricted to
   the same formal detector-valid candidate axis.
3. Every non-parent candidate is evaluated relative to that parent. The model
   predicts repair evidence, break risk, IoU advantage, and reliability for
   each candidate; it does not add a free residual to every query.
4. A candidate is deployable only when all fixed feasibility and reliability
   gates pass. Detector-invalid candidates are ineligible for supervision and
   switching. At most one candidate can replace the parent.
5. With no reliable positive action, initialization, invalid rows, or no valid
   candidates, the selected query and the complete score row are exactly the
   parent result. The parent score is never modified.
6. Training uses parent-relative targets and an explicit fallback action.
   Neutral candidates receive a negative margin rather than a zero target.
7. Joint-detection ScanNet rows are excluded from grounding supervision.
8. Phase-one training is fail-closed unless the dedicated train-only mode is
   enabled. Only the structured slot builder, SACR evidence head, and FPR-TV
   verifier may receive gradients.
9. The new path is opt-in. With it disabled, prior model, loss, optimizer,
   checkpoint, and inference behavior remain unchanged.
10. The implementation exposes detached statistics for switch rate,
    feasibility, reliability, repair, break, fallback, and loss components.

## Method correction after the consumed v1 fold

The first preregistered scene fold showed that v1 was structurally unsafe:
the safe-positive candidate rate was about 2.23%, while a class-balanced
reliability head marked about 10.34% of candidates reliable. Deployment then
treated balanced-loss sigmoid scores as calibrated probabilities and accepted
`max(repair) > max(break)`. This could promote a candidate with repair score
0.80 and break score 0.70.

FPR-TV v2 therefore has the following non-tunable deployment contract:

1. Repair, break, and joint Parent-dominance heads that use a fixed 0.5
   deployment boundary are trained with empirical-prior binary log loss, not
   class-balanced binary loss.
2. A candidate is eligible only if at least one threshold predicts repair,
   both thresholds predict absolute break probability below 0.5, and the
   joint Parent-dominance head predicts a safe repair above 0.5.
3. Relative `repair > break` is forbidden as a safety certificate. The
   action-vs-fallback logit, feasibility mask, deterministic structured gate,
   and exact Parent fallback remain required.
4. These rules are a semantic correction, not a threshold sweep. The
   consumed fold 0 may be used only as fixed failure evidence and never to
   select v2 thresholds, loss weights, Top-K, or margins. Any v2 experiment
   requires a new preregistered launcher and previously unconsumed scenes.

## Method correction after the consumed v2 fold

The only preregistered v2 fold showed complete abstention. On 7,341 held-out
rows the verifier made zero switches. During fit, safe-repair candidates were
about two percent of candidates, while empirical-prior BCE produced a learned
joint-reliability positive ratio of 0.00814% and a repair-positive ratio of
0.05299%. Requiring both rare-event heads above 0.5 duplicated the same
safe-repair decision already represented by the action-vs-fixed-fallback
softmax and made the deployment conjunction structurally unreachable after
one complete fit epoch.

FPR-TV v3 therefore makes one non-tunable structural correction:

1. The explicit setwise action head is the only learned repair/action gate.
   Its target is still exactly the best feasible safe-repair candidate, or
   the fixed fallback when no such candidate exists. Positive and neutral
   action margins remain fixed at 0.25.
2. The empirical repair head remains an auxiliary training/diagnostic head,
   but its `0.5` probability is not a second deployment veto. The redundant
   learned joint-reliability head is removed.
3. Deployment still requires the unchanged detector-valid candidate axis,
   score-gap feasibility, deterministic parse/target/anchor reliability,
   action logit strictly above the fixed zero fallback, and both absolute
   break probabilities strictly below 0.5. No relative repair-vs-break rule
   and no tunable probability threshold is introduced.
4. Fold 0 and fold 1 remain immutable failure evidence. Fold 2 was consumed
   by the separate Density-Aware scene audit and is also unavailable. A v3
   experiment may use only previously unconsumed fold 3 followed by fold 4,
   with identical frozen configuration. Both must pass the existing
   fix/break gate before any later proposal can be reviewed.

## Verification before any formal run

- Syntax and focused regression tests pass in an isolated copy of the remote
  runtime.
- Default-off compatibility and train-only parameter confinement are tested.
- Parent fallback and parent-score identity are tested.
- A raw Top-1 outside the formal detector-valid axis is tested to ensure that
  the deployed filtered parent is used and invalid candidates cannot switch.
- No-candidate, unreliable-row, padded-candidate, and empty-grounding-batch
  cases are finite and return a valid differentiable loss.
- No code is deployed over a live training checkout.

## Scene-disjoint advancement gate

The 100-microbatch density audit can authorize only this gate. It cannot
authorize long training or a formal 7,899-row evaluation.

1. Nr3D training scenes are assigned to five immutable folds by
   `int(sha256(scan_id)[:8], 16) % 5`. The complete source is exactly 511
   train scenes and 32,919 Nr3D grounding samples; ScanNet auxiliary rows and
   all official val/test rows are excluded.
2. Each fold starts independently from the protected E57 checkpoint, trains
   only the three allowed FPR modules for exactly one complete fit epoch at
   A1 with local batches of at most 16 and one natural remainder batch,
   evaluates only its held-out train scenes, saves no checkpoint, and never
   tunes a threshold between folds. The receipt must prove every frozen fit
   sample was consumed exactly once by matching the complete unique source-row
   identity set and its order-independent SHA, not merely a batch-size total.
   The complete V99/FPR optimizer, candidate, gate, margin, and reliability
   configuration is validated against one fixed contract and every receipt
   must carry the same canonical configuration SHA; fold-specific tuning is a
   hard failure.
3. The preregistered fit/holdout counts are:

   - fold 0: 402/109 scenes, 25,790/7,129 samples;
   - fold 1: 400/111 scenes, 25,578/7,341 samples;
   - fold 2: 408/103 scenes, 26,590/6,329 samples;
   - fold 3: 417/94 scenes, 26,714/6,205 samples;
   - fold 4: 417/94 scenes, 27,004/5,915 samples.

4. Every fold must have at least one deployed switch, strict
   `fix025 > break025`, and non-negative `fix050 >= break050`.
5. Exact state and sentinel receipts must prove that all frozen model tensors,
   Box outputs, Mask outputs, and parent V99 score rows are byte-identical
   before and after fit, while the allowed FPR state changes.
6. Exact integer transition counts must partition every held-out sample into
   fix, break, kept-correct, or kept-wrong at both thresholds. Batch-averaged
   ratios are not advancement evidence.
7. All five folds must pass before a separate short-training proposal can be
   reviewed. Even five passes do not themselves authorize long training or
   formal test evaluation.

## Explicit non-goals

- No fair baseline reproduction and no global-batch/150--240 epoch baseline
  rerun.
- Do not use the previously proposed Section 7 baseline plan.
- Do not use or rename the previously proposed Section 8 E0--E7 matrix.
- No joint multi-dataset training, dataset-ID feature, Unique/Multiple label,
  validation threshold sweep, global scalar gate, free all-query residual, or
  simultaneous Box/Mask/Backbone modification.
- The density-aware local Proposal Refiner is a separate later route and must
  not be mixed into the first FPR-TV experiment.

## Dataset targets

- Nr3D formal REC@0.25 must be strictly greater than 60.0%: at least
  4,740/7,899 hits.
- Sr3D formal REC@0.25 must be strictly greater than 68.9%: at least
  12,214/17,726 hits.
- Existing best weights remain immutable until a same-dataset formal result
  exceeds the corresponding protected best.
