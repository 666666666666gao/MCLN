# Current research contract — 2026-09-21

User priority: return to protected MCLN/V99 family; improve Nr3D beyond native
MCLN while preserving ScanRefer and Sr3D protected results. No multi-seed or
long baseline reproduction. Mask is diagnostic, not a promotion gate.

Current bounded experiment is specified in
`docs/NR3D_SEMANTIC_ASSIGNMENT_2026-09-21.md`. Its proposed claim is that
training-only final-layer root-token replacement for geometrically qualified,
unmatched Nr3D candidates can improve native grounding. This is unproven.

Evidence required: real native-loss integration and optimizer preflight,
same-start/same-budget/same-seed native-versus-replacement training, complete
7899-row REC with unchanged inference, separate candidate coverage and selection
analysis. Synthetic checks alone do not establish a training or accuracy claim.

Protect root/other-GT matched Queries and joint detection prompts. Do not
introduce GT at inference or claim box IoU proves semantic identity. Preserve
the original geometric assignment and all other losses. E57 has no optimizer
state, so this is fresh adaptation from protected weights.

Current native Nr3D reference is 59.82/51.38; strict exceed requires 4726/4059
hits. Protect historical MCLN 4475/3759; obtain the actual same-source start
separately. Do not combine single-metric bests from different models.

The bridge skill's referenced research-contract template is absent from the
installed skill bundle; this record states the applicable project contract
directly rather than blocking execution on that missing template.
