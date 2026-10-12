# MCLN portable transfer: source review

**Verdict: WARN; source gate PASS, zero remaining blockers.** Scope is **SOURCE_ONLY**. Requested reviewer: `gpt-6-astra`, reasoning `max`; actual model/effort identity **UNATTESTED**. Same-family review; acceptance is provisional.

The corrected source is ready for the actual CPU factory check. This review does not establish successful imports, GPU execution, accuracy, or experiment integrity.

## Findings

- **B1 resolved:** the original paired-row scorer used normalized main-token weights while native `GroundingEvaluator._parse_gt` binarizes them. The replacement calls the native parser, accumulates the five terms in native order and uses the native default-axis ranking. It is now source-equivalent to the `last/bbs` readout.
- **B2 resolved:** the first archive omitted the model's required class embeddings and the parser mapping/text assets. The final manifest includes the five local parser assets; `REMOTE_ASSETS.json` and the launcher explicitly check and copy existing class embeddings and the warm PointNet extension. Remote existence is deferred to the actual CPU stage.
- **N1 nonblocking:** local launcher and observer use `shlex.join`. Invoke them through the existing **local Python 3.11 credential runner**; shell-default Python here is 3.7.6. Remote/controller/evaluator code is compatible with Python 3.7 syntax. No fallback is needed.

## Source conclusions

The MCLN diff contains only three attributes and the forward-tail call. A uses native mask-query features; B uses the pre-mask decoder query. Both are 256 queries by 288 dimensions, and support features use the matching 288-by-SP layout. Strict E71 loading preserves the 1135-state native core after removing nine SourceChoice states; A/B have the intended 27841/29793 parameters. The four portable modules are unchanged from their supplied source. G and Face are excluded.

The adapter consumes only observed points, model features and predicted masks/boxes. GT is attached after forward and used by the original loss and evaluator. The three arms share one raw batch and restored Python/NumPy/Torch CPU/CUDA RNG, preserve native semantic scores, and must pass 1159-state equality at completion. Evaluation uses the native validation dataset, actual GT and strict IoU thresholds. The code requires 8 sanity rows or 9508 full rows and native integer-hit equivalence before writing completion.

The replay has no optimizer or checkpoint-writing path. The launcher checks preceding actual success, BN completion and GPU admission before taking the existing serial lock. The controller records the child exit and result. The observer only reads existing status/process identities/logs at its supplied due time. SSH staging uses the configured real destination, host-key witness and private credential route.

## Checks and limits

- All 736 packaged files match the source manifest; exact reviewed input hashes are in `SOURCE_REVIEW.json`.
- 121 packaged core/entry Python files and both embedded remote bodies parse under actual local Python 3.7.6. Inactive legacy script fragments were excluded from syntax claims.
- The original MCLN diff and four unchanged portable source comparisons passed.
- Executed: local source reads, hashes, syntax parsing and a pure arithmetic scoring counterexample. Experiment imports, SSH, GPU queries, neural forwards and optimizer updates: **zero**.

Next: actual CPU factory/checkpoint verification, then real 8-row paired sanity on successful admission, then full 9508 only after sanity passes. Transfer effectiveness and normal MCLN training remain unmeasured. `REVIEW_REQUEST.json`, `REVIEW_TRACE.md`, `run_source_checks.py` and `STATIC_CHECKS.json` preserve the request, original defects, corrections and checks.
