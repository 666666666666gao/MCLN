# Nr3D Mask supervision and superpoint-target audit

The fixed validation-input diagnosis now separates 1105 good-box/bad-Mask
expressions into 154 limited by the superpoint representation, 123 additionally
limited by majority-label construction, and 828 whose majority GT Mask already
exceeds .5 while every predicted fused Query Mask remains at or below .5.
This supports inspecting raw text, raw Query, and fused Mask predictions before
choosing another Mask architecture. It does not establish which raw branch fails.

## Scope and fixed inputs

This is a CPU-only extension of the completed P1 diagnostic, with zero model
forwards, zero optimizer steps, no new model scores, and no training-rule change.
It applies the existing training-label construction rule to P1's fixed validation
inputs: 7899 expressions, 1213 unique scene/target pairs, and 130 scenes.
Do not describe these as observed failing training examples or a new model result.
The 131 consumed scan/superpoint files exactly match P1's data hashes. Every
target-point count and GT optimal-superpoint IoU reproduces the sealed P1 output.

## Existing-source checks

The local model, loss and dataset ASTs match the frozen runtime source. The
author repository was pinned at `9744a4ed219062d448ed0dba587eeb864491f158`.

- The text branch selects an SWA-refined **text token**, creates one text Mask,
  and broadcasts it over the REC Query axis. Here `q_idx` is a text-token index,
  not the final REC candidate index. This design already exists in the author
  [model implementation](https://github.com/qzp2018/MCLN/blob/9744a4ed219062d448ed0dba587eeb864491f158/models/mcln.py#L533)
  and is already described in the master architecture section.
- Hungarian Mask cost reads this broadcast text Mask. Its value is constant
  across Queries for each GT column. With every GT column assigned once,
  that term supplies no Query-disambiguation evidence; changing its coefficient
  alone does not add Mask-quality information to the assignment. This is a
  consequence of the existing broadcast path, not a measured gain or a newly
  introduced defect. The current active scalar fusion path preserves the broadcast.
- The consistency pseudo-label threshold `sp_src_masks > 0.5` acts on logits;
  final fused inference uses logit > 0. The same consistency threshold is in
  the author [loss implementation](https://github.com/qzp2018/MCLN/blob/9744a4ed219062d448ed0dba587eeb864491f158/models/losses.py#L544).
  This audit does not change it or attribute the paper gap to it.
- The active Nr3D annotation loader initializes an empty Anchor-ID list and
  keeps scalar root targets. Thus the concern that one text Mask is supervised
  against both root and Anchor GT masks does not apply to this Nr3D path.
- Protected configuration has `mask_loss_scale=1` and `consistency_loss_scale=1`;
  inactive Mask calibrators/refiners are not treated as active model components.

## Point-space quality of the existing training labels

The current loss marks a superpoint foreground only when strictly more than
half its sampled points belong to the target. We expand those labels back to
points and measure their GT IoU. Separately, the already-tested optimal-union
oracle chooses the best union of whole superpoints using GT.

| Fixed validation-input construction | Mask hits > .25 | Mask hits > .50 | Point-space mean IoU |
|---|---:|---:|---:|
| Current strict-majority GT labels | 7645 / 7899 | 7291 / 7899 | 81.956425% |
| GT optimal union of superpoints | 7843 / 7899 | 7476 / 7899 | 83.374977% |

These are label/oracle values, not deployed predictions. Their 1.418553
percentage-point mean-IoU difference is neither a promised training gain nor
an upper bound on how changing supervision could affect learned predictions.

There are 153 expression rows, representing 23 unique scene/target pairs, whose
majority construction has no foreground superpoint; 146 of those rows have at
most 227 sampled target points. Examples by category include toilet paper
41/95, soap dish35/56, and bottle21/47 expression rows. These categories are
descriptive and must not become validation-derived training exceptions.

## Refinement of the 1105 good-box/bad-Mask cases

Here good box means some legal Query box has root IoU > .5. Bad Mask means the
best **fused** Mask among all 256 Queries has root Mask IoU <= .5. This is an
IoU-defined diagnostic, not independent GT instance-identity classification.

| Mutually exclusive condition | Expressions | Share of 1105 |
|---|---:|---:|
| Even the optimal superpoint union has IoU <= .5 | 154 | 13.94% |
| Optimal union > .5, but current majority GT Mask <= .5 | 123 | 11.13% |
| Current majority GT Mask > .5, but all predicted fused Masks <= .5 | 828 | 74.93% |

All 951 cases with an optimal bound above .5 have at least one majority-positive
superpoint. The 828 cases in the last row remain unexplained by representational
capacity or hard-label construction alone. A raw branch could still be good and
then be damaged by fusion; the present artifact contains fused predictions only.
Therefore it does not justify directly assigning the cause to RSA, the shared
backbone, one raw branch, or the scalar fusion weight.

## Consequence for ongoing work

R1's four-arm comparison and all advancement screens remain fixed. Its training
and scheduled terminal analysis are untouched. After collecting the complete
R1 result, the next Mask-specific diagnosis should record raw text, raw Query,
and existing fused Mask IoUs at the existing threshold, plus the native scalar
fusion weight, while keeping Query identity explicit. This is an evidence
collection step, not a selector revival, threshold sweep, or new Mask-head trial.

The first CPU launcher stopped before statistics because its data-provenance
argument incorrectly included `results/`. The corrected launcher reads the
original audit-root manifest; the numerical script and settings are unchanged.
Both launch records and the original error log are retained. The corrected
controller completed with exit0. Two focused CPU tests cover empty majority
labels and a case where the majority construction misses .5 while the optimal
superpoint union exceeds .5.

Evidence: `refine-logs/mask_training_target_audit_20260905_v1/`.
Full result SHA-256: `79ce884363d1ae8e79590659b57d4544732003d5dde2f4e56ad4c938d3744d77`.
Runtime source manifest: `dcf333b0e1868a7eeaafaf7f0a7abdb664a34dda65966defc1ad244ce762b15d`.
No protected benchmark result was replaced; the three-benchmark objective
remains incomplete.
