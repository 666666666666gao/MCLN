# Auxiliary-target comparison source review

PASS. No blocking or unresolved nonblocking source defects found.

Execution scope: **SOURCE_ONLY**. Fresh delegated Codex review; requested `gpt-6-astra` / `max`, backend unattested, same-family / provisional. No source edit, deployment, remote call, model forward, optimizer update or weight load was performed.

The two arms differ only in the auxiliary reference: native noisy root versus the annotated member root captured from the same augmented native 50000-point scan before independent native box jitter. The chosen root is used consistently for auxiliary qualification and L1/GIoU/DFL. Native matching, native GT and losses, G correction, detection inputs and official evaluation remain unchanged. This is a qualification-plus-target comparison, not a loss-only change.

The root-only auxiliary DFL shapes are correct: one valid slot and one center/size pair, with every extra target index zero. The original full native GT mask is retained only for native matched DFL. Every native matched query is excluded from extra supervision. Own-query and fused Mask IoU must both exceed .5; per-expression averaging and empty-zero batch averaging are preserved.

Both arms reconstruct from official PV, original G and the retained4509 full10-state geometry head. The factory does not need deleted4506/control weights. Only the existing456102 geometry parameters train; parent, Mask, semantic and zeroR states stay frozen. Each arm has3723 new updates over29778 rows once, B8/tail2, for11169 cumulative geometry updates.

The new preflight input checks compare row IDs, point identity, native noisy GT and member target with the actual closed CPU64 records before two updates. The code verifies isolated qualified-output gradients, frozen parent states, cached bbs/Mask, finite gradients and strict memory-only optimizer/model restore. Formal fit rebuilds in a fresh process.

Read-only validation passed: five Python AST checks, five preparation digests, all17 warm-helper bindings, exact arm-spec difference, unchanged native function ASTs, capture-method AST equality to the completed CPU probe, and saved64 row/point/native-target identities. Saved4509 batch1 under the previously fixed matches contains zero-qualification rows and outside-range faces for both modes (126 native;40 member). This supports the selected batch but does not substitute for the new GPU preflight.

The required CPU-label audit is now PASS with no blockers. Actual two-arm GPU preflight remains pending and must pass before formal fit. The reviewer did not open weights; full-state reconstruction is supported by the actual completed training/retention evidence and factory source. The predecessor receipt is at `complete/query_supported/receipt.json`; there is no `train/receipt.json` at the suggested alternate path.

Exact reviewed paths, SHA256 values, coverage notes and evidence are in `SOURCE_REVIEW.json`. Some warm dependencies were checked for digest binding only; substantive helper source coverage is separately marked.
