Actual R1 failure diagnosis — source-only, cause unresolved.

The collected Nr3D-native run failed at `referit_model_preflight.py:141`, the Query Mask allclose assertion (`rtol=atol=1e-5`). The collected runner is byte-identical to sealed R1, SHA256 `512f21026646458ad68dae8b4b1d37cf18ed3b1bfdb91dec9045a0a08cce2b34`. Both initial forward calls returned; the preceding original-state equality and semantic comparison passed. Criterion/optimizer construction and backward had not been reached. Controller and child exit are 1, completed runs are empty, and the collected observer is closed.

No numerical mask differences were emitted, so the current artifacts cannot establish magnitude, sign changes, or cause. Source-only R1 PASS was not runtime success.

Concrete source path: `pv_ground.py:430` computes superpoint centroids via `scatter_mean`, implemented with device `scatter_add_` at `scatter_util.py:30`. Grouping, relative-coordinate encoding, and max pooling feed Query Mask through `x_query` and the final mask einsum. The semantic decoder does not use this superpoint mask-feature branch. Geometry refinement and boundary readback execute afterward and cannot directly overwrite masks already generated. The zero-output task reader and task transforms show no new deterministic structural defect in the inspected source.

Native floating-point variation is therefore plausible but unproven. The separately reviewed A/B/C control is appropriate: native A and native B keep the same branch, installed C changes only the intended installation; all use one fixed batch and identical RNG, with fresh input dictionaries. It records unchanged-tolerance comparisons and per-row own/fused sign and raw-logit differences, plus centroid and projected-query diagnostics, then ends without any optimizer or weight write.

A/B variation would establish native cross-forward variation for this batch, not prove scatter causation or equivalence of C. No tolerance relaxation or four-run retry is supported by the existing evidence. Exact diagnostic-source review is in `initial_comparison/SOURCE_REVIEW.json` and `.md`.

Same-context, same-family, provisional review; actual backend not attested. No SSH/network/GPU/model or project-code execution was performed by this reviewer. Original R1 source and audit were not edited.

Evidence SHA256: run.log `24716e5f46b079a8d72e952add38c381460ce1a9faac155c5398496f75a13e24`; imports.json `5fe75f6dc6c1a7be5ebcc5846986d415e5d9e4cea67eb5df9e0946cdb6edcb32`. Full reviewed identities are in the separate diagnostic source review.

