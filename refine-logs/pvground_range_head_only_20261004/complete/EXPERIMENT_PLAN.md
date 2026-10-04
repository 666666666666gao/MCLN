# Stable G / range-head learning control

Status: planned, not launched. Previous source pair evidence lives at
`../pvground_whole_mask_fit_20261003/analysis/REPORT.md`.

The completed whole source improves strict native bbs by33 versus local, but
remains34 below original G. Its same-selected-Query final refinement is
4471->4461 (13 repairs,23 damages), with2.83mm median maximum face movement.
This does not establish effective boundary refinement or identify a unique
optimizer failure. First test whether the existing head can learn useful
corrections while the original G representation is stable.

Use the same original official PV+G, unchanged400614-parameter/1302-input head,
109 full-range statistics, local fused support and native scoring. Freeze every
original G parameter and keep all original modules in eval mode, including
running statistics. Train only the10 existing refiner parameters. Keep original
native geometry losses; semantic/Mask outputs are frozen observations. There is
no geometry-to-Mask training route in this control, intentionally. No teacher,
new loss, six-face distribution, quality readback, P2 or dual inference ranking.

First run one real whole-range batch8 twice to verify zero-head preservation,
real head geometry gradients, all original state unchanged after both updates,
and exact in-memory head/AdamW restoration. Reuse unchanged model modules and
the completed factory/evaluator. This is a new freeze-mode sanity check, not a
repeat of the already successful joint-training preflight. No disk weights.

After the sanity result, compare refiner-only local_range versus whole_range.
Each starts again from original G, freshAdamW, seed2027, effectivebatch8,
LR1e-5,WD5e-4,clip0.1,29778fit rows once/3723updates. Preserve source and all256
candidates. Separate6887 pretrained-seen initial/terminal from9508 development
validation. Compare both source arms and originalG5615/4495; goal5615/4754.
Record direct sameQuery refinement and paired repairs/damages; shared-training
results are historical adaptation comparisons, not independent source gains.

Only one active recovery file. Keep the strict-bbs metric best and protected
parents/V99; delete our verified nonbest endpoints, preserve rows/code/logs and
create no failed full-weight archives. No batch/schedule/budget changes mixed
into this control. Original-score and coarse-candidate equality is checked from
actual outputs; stochastic/non-bitwise differences remain reported, not hidden.

If this control still cannot learn effective corrections, move to the user's
direction-preserving six-face support decoder and explicit boundary supervision.
If it helps, evaluate finite joint adaptation separately. Quality readback and
training-only teacher are later independent changes. No preclaim of three
validated contributions or Nr/Sr generalization.
