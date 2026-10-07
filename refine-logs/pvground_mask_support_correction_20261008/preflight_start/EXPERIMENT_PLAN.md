# Mask support correction — implemented source, not deployed or validated

The full goal remains one complete ScanRefer model with at least5620/4764 hits and three effective contributions, followed by independent author-initialized Sr3D/Nr3D training. Current protected best5598/4848 remains the recovery source. No new score or effective-module claim is made here.

## Evidence and target

The coordinate-only analysis of the128 old overextended cases found at least one background-only possible extremal member in every case. The63 missing-range cases have164 missing faces with observed target members beyond them. These are selected historical development errors, not dataset incidence, recovered historical Masks or training examples. The fresh subset does not reproduce old full-sequence Gumbel RNG. Pure-background and missing-support errors motivate learning candidate support, rather than repeating the failed ordinary Box residual head.

## Smallest structural comparison

Keep PV/G, native semantic scoring, all256 candidates, exact member-based Mask reference and the zero-output Box head fixed. Add a zero-initialized residual correction to each candidate's own Query superpoint Mask logit, then use the unchanged native scalar-alpha Text/Query logit fusion. Both arms use the same candidate/superpoint content and existing Text/Query/fused Mask evidence. The treatment also receives superpoint position/range relative to the original coarse Box; the control receives zeros in exactly those9 geometry channels. Common native features already contain spatial information; this isolates additional explicit coarse-Box relations, not all geometry. Parameters, initialization, optimizer and budget are shared. Capture coarse geometry before Mask-reference replacement; do not use GT geometry at inference. Corrected support must produce both the delivered Mask and its one reference Box for the same Query.

This tests whether the coarse prior helps remove non-target extent and restore missing target support. It is not an adaptive inference threshold, a second ranking or a guaranteed new contribution. Preserve native invalid-support behavior; do not add a new fallback.

## Training responsibility to specify from actual code

Use only the existing training split, with the actual native instance/member labels. Establish candidate-to-GT correspondence from the frozen parent before correcting support. Each matched candidate retains its actual GT, including other instances; do not substitute root slot0. Use only native matched candidates; no expanded positive set. Reuse native majority-superpoint labels and the four active gradient terms:5 Query focal +1 Query Dice +10 fused focal +2 fused Dice. Other native terms cannot update this frozen-parent Mask-only head; no extra quality ranking loss is introduced. The reference threshold/extrema path stays detached; do not claim Box-loss gradients end-to-end through it.

## Execution gates

Before launching, read the real PV forward/evaluator/Mask criterion, implement the smallest two-arm change and obtain source review. A real two-step GPU precheck must verify zero-output equivalence, task gradients, frozen-parent state, restore and measured capacity. Reset the formal optimizer and state after precheck. Complete the full original9508 batch order for initial and final evaluations; do not replay only selected batches or reuse the subset RNG as historical equivalence. Fixedseed2027, batch8 and the established one-pass29778/3723-update budget are the initial control recipe. Estimate duration from actual precheck/closed timing; no early NN polling.

Both arms must be compared with each other and the protected5598/4848 start. Record paired repairs/damages, Mask metrics and original scoring invariance. A win over a degraded control alone is insufficient. Keep only the selected best learned state and dependencies after closure; archive results before retiring nonbest weights. Do not deploy, promote three contributions, or start Nr/Sr from this design note alone.

## Actual implementation state

CandidateMaskSupportCorrector (27841parameters/10state tensors), matched native Mask objective, paired-forward helpers and an isolated PV forward overlay have been written under pvground_mask_support_correction_20261008. The overlay preserves the original source and integrates the correction before native Mask reference. GPU precheck, full runner, source acceptance, deployment and formal scores remain pending. This note is not an executed-experiment result.

## Current runner implementation

The paired runner now implements native-parent matching, native active Mask terms, two independent heads/optimizers, full original evaluation order, CPU reconstruction plus GPU head/optimizer restoration and native PV integrated-forward checks. These checks are source only until the actual preflight exits successfully. No new GPU precheck, fit, formal metric or effective contribution is claimed.

## Source review correction with actual repeatability evidence

The first fresh SOURCE_ONLY report is FAIL with two blockers and stays preserved with its source snapshots. Existing recorded same-RNG full-forward Query Mask drift reaches 1.2e-4, and the previous complete6887 frozen-parent passes differ in the reference Box at row16804 by0.03451145m. Thus an independent repeated native forward cannot be an implementation-equivalence witness. The corrected M0 captures the new head's uncorrected inputs and output in one actual native integrated forward, then replays both pair heads on that exact cache and requires exact corrected-Mask and final-Box equality. Full CPU reconstruction and actual GPU new-head/optimizer restoration remain required; no cold GPU parent reconstruction is claimed. No tolerance was increased and no fallback was added.

Independent initial/terminal traversals still require identical input/GT identities and frozen parent states. They preserve both parent outputs and report Query/Box drift instead of asserting threshold-derived parent Box equality. Initial9508 requires each zero head to match that same forward's parent count; if the parent differs slightly from historical5598/4848, the historical best stays unchanged and that difference is not head benefit. Incremental REC evidence uses each same-forward parent and the paired control, with historical differences also reported. The full original source port contains98declared files; all remain, with only the PV forward overlay replaced. Corrected source acceptance is pending in SOURCE_REVIEW_SUPPLEMENT, while the original FAIL remains visible.
