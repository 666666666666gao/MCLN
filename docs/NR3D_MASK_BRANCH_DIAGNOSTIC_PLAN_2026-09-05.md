# M2: observe native Mask branches at fixed thresholds

This plan is fixed before reading any M2 branch outputs. Start only after the
original R1 four-arm run and its terminal verification have completed. M2 is a
read-only diagnosis of the protected model, with zero optimizer steps and no
experimental R1 or P2 head loaded.

## Question and necessary new observation

The sealed P1 observer retained fused Query Masks, but did not retain raw text
and raw Query Mask logits or their IoUs. The CPU majority-label audit therefore
cannot distinguish poor raw predictions from damage caused by their fusion.
Recovering this missing evidence requires one additional native forward pass;
this is not a score-selected historical reproduction retry.

Keep the original 1105 good-box/bad-fused-Mask rows and their 154/123/828
representation/majority-label/prediction partition fixed from P1 and the CPU
audit. Do not redefine cohort membership using this run's predictions.

## Fixed inputs and computation

- Protected averaged checkpoint SHA-256:
  `76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`.
- Frozen current-source manifest SHA-256:
  `dcf333b0e1868a7eeaafaf7f0a7abdb664a34dda65966defc1ad244ce762b15d`.
- Same native validation dataset/order, 7899 rows, B16, four workers, one rank,
  historical CLI/config and 627-file data provenance as P1. Only the output
  directory and experiment name change. Historical core source is still not
  fully recovered, so equality to the historical 4475/3759 receipt is separate
  from diagnostic validity.
- The runner is a minimal derivative of the sealed P1 observer. Model, loss,
  candidate legality, source scores, Query/Box/Mask outputs, native evaluator,
  sampling, and thresholds remain unchanged.
- Record original text logits, original Query logits, and actual native fused
  logits, all thresholded at logit greater than zero. Record the actual scalar
  alpha; do not search alpha values, thresholds, gates, or score combinations.
- Assert the text Mask is broadcast across Queries, reconstructed fused point
  Masks exactly equal native evaluator Masks, and point-count float64 IoUs
  equal the existing fused-Mask observer.
- Keep native REC Query, native Mask Query, and GT-best legal Box Query separate.
  Also record branch oracles over all Queries and over legal Queries whose Box
  IoU exceeds .5. A good Mask on the wrong Box Query is not REC/Mask alignment.
- Check every protected parameter/buffer before and after evaluation, plus
  immutable source, observer, checkpoint, and data hashes. No checkpoint saves.

## Analysis fixed in advance

First compare all 7899 identities and the existing P1 prediction fields with
the sealed rows, reporting any difference instead of treating it as a method
gain. Analyze the original cohorts regardless of whether predictions differ.
The runner calls the old misnamed sentence-length field
`normalized_token_count`; use the existing CSV enrichment for actual raw
token lengths if a sentence-length analysis is needed.

For each fixed cohort, report whether either raw branch can exceed .5, whether
fusion exceeds .5, and the corresponding counts restricted to good legal Box
Queries. Preserve native selection results as separate columns. Report alpha
distribution descriptively, without using validation outcomes to choose a new
value. All branch oracles use GT and are diagnostic only.

If raw branches contain good Masks but the actual fusion loses them, this
supports investigating the fusion/training connection. If both raw branches
fail, it still does not by itself isolate RSA, backbone, sampling, or the loss.
No new module, training run, or formal weight promotion follows automatically.

Two CPU tests already pass in the actual Py3.7/Torch1.10.2 environment: exact
point-count IoU including unused superpoint IDs and zero logits; and separation
of native selections, good Masks on wrong Box Queries, and legal good-Box
oracles. These tests do not constitute a completed GPU or benchmark audit.
