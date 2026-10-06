Source-only review: PASS; no blocking findings.

This is a same-context, same-family, provisional review. The actual backend is not attested. I inspected local source and collected R1 artifacts directly; I did not launch code, access SSH/network/GPU, load a model, install packages, or change source.

R1 actually failed at `referit_model_preflight.py:141`: Query Mask allclose at the unchanged `rtol=atol=1e-5`. The collected runner matches sealed R1 (`512f21026646458ad68dae8b4b1d37cf18ed3b1bfdb91dec9045a0a08cce2b34`). The preceding semantic comparison and original-state check passed. The failure occurred before criterion/optimizer creation, so there were zero optimizer updates and no completed M0 run. The log supplies no mask deltas or cause.

The separate diagnostic is correctly bounded. A and B retain the same native decoder path with observed VSA groupers; C reattaches the saved task reader and installs the zero-initialized heads. Each forward uses the same fixed four-referring/four-detection augmented batch, an identical RNG reset, eval/no_grad, and a fresh input dictionary. The rendered source contains exactly three model calls and ends before criterion, optimizer, backward, or weight serialization.

All 1235 official initial tensors are checked before A and again after A/B before C; the installed model has 70 additional states. This is an initialization witness, not an after-C restoration claim. The corresponding Nr3D official checkpoint is pinned; no Scan trained state is loaded.

All A/B/C pairs record global semantic, box, x_query, and query-point maxima; per-row mask max/mean/RMS, unchanged allclose, own/fused logit-sign changes, text/alpha differences, and superpoint-centroid differences. The fusion formula matches production. Semantic maximum is global, as intended for this bounded measurement. Sign-change counts use logits > 0.

The source gives a concrete location to investigate: `pv_ground.py:430` forms superpoint centroids through `scatter_mean`, whose sum uses device `scatter_add_` (`scatter_util.py:30`); grouping, relative coordinates, and max pooling then feed Query Mask. The semantic decoder does not consume that mask-feature branch. Geometry refinement and boundary readback run after Query Mask formation. This makes native numerical variation a plausible explanation, but current evidence does not identify its cause or exclude an installed-path difference.

A/B drift would establish that this native comparison is not exactly repeatable for the fixed batch. It would not by itself prove scatter causation or installed-path equivalence. A single triplet cannot establish a new tolerance. The diagnostic preserves the original tolerance and supplies measurements without repeating the four M0s.

The launcher checks the closed prior observer and failed R1 exit, no receipts, reviewed identities, the pinned existing environment, GPU availability, and one absent isolated root. It performs one launch and exclusive file writes without changing R1. The original R1 audit/source remains unchanged.

`SOURCE_REVIEW.json` records all 24 reviewed path/byte/SHA256 identities and precise check locations. This source PASS does not report a successful diagnostic run or resolve R1's runtime failure.

