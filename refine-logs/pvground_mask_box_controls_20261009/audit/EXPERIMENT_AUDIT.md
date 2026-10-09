# Experiment Audit Report

Date: 2026-10-09. Overall verdict: **WARN**. **Independent deterministic verification: PASS.**

Fresh context; requested gpt-6-astra / max. Actual backend, model and reasoning effort: UNATTESTED. Same-family review; acceptance provisional. No blocking finding prevents accepting the narrowly stated cached CPU geometry comparison.

| Check | Status | Finding |
|---|---|---|
| A. GT provenance | WARN | Dataset-derived GT and exact original loader/scan-handler hashes verified; original raw scan/annotation bytes were not rederived. |
| B. Normalization | PASS | Standard IoU, strict thresholds and fixed denominator 9508; no output-dependent metric normalization. |
| C. Result existence | PASS | 9508 rows, 141 scenes, 2068 targets by scene, 1189 shards; all counts, transitions and hashes match. |
| D. Executed code | PASS | Independent offline CPU checker actually exited 0; fixed native-score Query, coefficient 0.5 and existing invalid handling verified. |
| E. Scope | WARN | One October 6 snapshot plus a separately qualified cross-forward hypothesis; neither blend satisfies both goals. |
| F. Evaluation type | PASS | real_gt under the archived native data contract; cross-forward predictions remain a hypothesis. |

| October 6 same-source condition | Hits @0.25 / @0.5 | Accuracy @0.25 / @0.5 |
|---|---:|---:|
| Native regression | 5615 / 4495 | 59.0555% / 47.2760% |
| Pure Mask reference | 5598 / 4848 | 58.8767% / 50.9886% |
| Fixed half blend | 5671 / 4839 | 59.6445% / 50.8940% |

Blend versus Mask: repairs/damages 121/48 at 0.25, 207/216 at 0.5, net +73/-9. All 28,524 selected IoUs per precision were reconstructed. Float32/float64 and saved GPU/CPU threshold flips are zero. There are 39 selected invalid references and zero native-score winner ties.

The existing float32 formula yields 94 Mask IoUs slightly above 1, maximum 1.0000050067901611. Float64 is at most 1.0; accuracy counts are unaffected. Values were retained without clipping. This is a numerical warning, not score normalization.

The separate old-regression/current-Mask hybrid verifies as **5676/4840**, net +77/-19 versus saved retained 5599/4859. All9508 input-hash/GT/query/uncorrected-Mask identities align. Old/current core dependency records match, and the retained 447109-byte checkpoint archive hashes to 6f9109cb690399880b8dc3bcf6e4b98d1053145732ee2a6b504bfaecbd1eaac2. Current same-forward native boxes are absent, so this is not new current-model performance or promotion evidence.

GT evidence: old run_geometry_fit.py:194-213,312-338; hash-matching joint_det_dataset.py:1086-1119,1387-1392; original visual_data_handlers.py:129-178,225-259. Query and execution evidence: old run_geometry_fit.py:312-356,365-378; native_root_bbs.py:4-9; readback_preflight_checks.py:19-68. Control evidence: analyze_fixed_query_boxes.py:61-119,137-146; independent_verify.py:65-245. Novelty/scope evidence: EG-3DVG.txt:208-247; FINDINGS.md:7-40. Exact absolute path definitions and full per-check references are in RAW_REVIEW_REPORT.md and the JSON report.

The original loader uses annotated sampled object members; full-resolution GT equivalence and complete original GPU state were not newly attested. The later load receipt is not a separate initial-stage state seal. No new GPU replay is needed to validate the complete cached arithmetic.

`blocking_findings=[]` for the closed recount. Claim blockers remain: unsaved current same-forward regression boxes; neither blend meets 5658/4850; no three-effective-module or Nr3D/Sr3D execution evidence supplied by this diagnostic. It is only an EG-3DVG geometry-formula control, not full model reproduction or a novel contribution. No verdict replaces missing experiments.

The missing request path, Windows junction assertion failure, float32-range assertion failure, and normal no-checkpoint search result are preserved in INVOCATION_ERRORS.json with raw logs and invocation receipts. Successful numerical execution: verify_003.log / completion chunk 4fa41d. Supplementary narrative/provenance checks: verify_004.log / chunk de1cf4. The 3074 zero-overlap count is a box statistic; raw Mask membership was not rederived.

Full report: RAW_REVIEW_REPORT.md. Machine-readable judgment: EXPERIMENT_AUDIT.json. Complete SHA256 evidence: REVIEWED_FILE_HASHES.json and VERIFIED_ARRAY_HASHES.json. Deterministic receipts: DETERMINISTIC_VERIFY.json and SUPPLEMENTARY_VERIFY.json. All scientific inputs and active training/source state remain unchanged.
