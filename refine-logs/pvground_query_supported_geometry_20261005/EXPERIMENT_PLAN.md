# Additional geometry responsibility, conditional on candidate Query support

Status: IMPLEMENTATION_DRAFT. Do not deploy until the separate Query/Text/fused fixed64 probe closes and validates a nonempty Query-confirmed target pool. This is a training strategy, not a fourth model module or a claimed gain.

Current protected best: PV official -> original G -> supervised boundary distribution head, 5616/4506. Reconstruct that exact model with the existing zero-output R source port, freeze/eval every parent and R, train only the existing 456102-parameter geometry head. Keep all256 candidates, one native bbs score and same-query Box/Mask. No teacher/new inference GT gate or geometry-range change.

Control: unchanged native+G+matched boundary loss (1/7). Strategy: control loss plus one separately normalized block for unmatched root-support-qualified Box-poor candidates. Training-only qualification requires BOTH candidate Query Mask IoU>0.5 and native fused Mask IoU>0.5, final Box IoU<=0.5, and exclusion of all actual native Hungarian-matched Queries (root and other instances). Public Text Mask alone never qualifies a candidate. This is a support/overlap proxy, not physical identity truth.

Each extra candidate receives existing native final Box L1/GIoU and existing six-face distribution target formulas. Mean over extra candidates within each expression; empty observed rows contribute zero; average over actual batch size; independent block weight1 with native final-layer 1/7 scale. Original supervision/matching/denominators unchanged. Current knots/range/clipping unchanged and out-of-range targets recorded.

Same initial checkpoint, data and seed2027; each arm fresh AdamW, lr1e-5, WD0.0005, clip0.1, micro/effectiveB8, accumulation1, 29778 fit rows once ->3723 updates (3722 full B8 + last B2). Original4506 cannot substitute for the new head-continuation control. No repeated P2, G denominator, R quality, local probability concatenation or teacher-coordinate experiment.

Engineering gate: fresh source review; actual two GPU updates per arm; extra geometry loss isolated gradients to qualified output/head only; zero direct responsibility for nonqualified or already matched outputs; parent/R state and native bbs/Mask exact under head updates; memory serialization+optimizer reload; actual empty-row and endpoint cases from current tensors; no disposable disk weights.

Then one full fit/holdout/formal9508 comparison, repair/damage/sameQuery refinement and candidate coverage. Control and strategy terminal weights are small head deltas. Keep active recovery and only metric-best weights after verified results; no negative-weight archives. Target remains same-model >=5615/4754; strict target gap248, ACTIVE_UNMET.
