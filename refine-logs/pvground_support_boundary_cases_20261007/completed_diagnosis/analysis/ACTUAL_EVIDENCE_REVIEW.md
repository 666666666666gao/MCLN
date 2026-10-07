# Final actual-evidence audit

**Verdict: WARN.** The closed evidence supports two bounded descriptions: fresh-forward member support on a historically selected error cohort, and conservative provenance of old cached face coordinates. **Historical prediction/support recovery remains failed and rejected.** There is no blocker for those two qualified descriptions; this does not accept the rejected claims.

Execution scope: `ACTUAL_ARTIFACTS`. Reviewer task: `/root/pvg_closed_support_final_audit`. Additional fresh context; same-family and provisional. Actual serving backend/model and reasoning effort: **UNATTESTED**. Timestamp: 2026-10-08 01:19:03 +08:00. The task briefing supplied artifact paths, earlier-auditor status and claim boundaries; this was not a blind zero-information review. I checked those boundaries against the actual sources and evidence.

## What was actually audited

I read the collector and both analysis/helper chains, both independent CPU checker sources, actual model/dataset source counterparts, the declared manifests and source reviews, closure/intake/cleanup records, all 191 current support records, all 191 analysis records, all 191 historical-coordinate records, the 191 primary cohort records and the 9508-row historical identity file.

A new private checker rehashed **all 205 retained intake files, 127,881,001 bytes**, and found exact size/SHA matches. All **191 NPZ remain local**. It matched six source files to actual imported-source hashes, checked two additional bound runner sources, reconciled every row and every summary aggregate, verified the exact cleanup manifest, and decoded three targeted NPZ cases. It did not rerun either completed full point-level CPU audit.

The original auditor, `/root/pvg_support_boundary_actual_audit`, produced:

- `PRELIMINARY_ACTUAL_EVIDENCE_REVIEW.json/.md` and `PRELIMINARY_CPU_CHECK.json`: 61 complete NPZ at 23:13:58 on October 7, with 130 still missing and historical equivalence rejected.
- `analysis/INDEPENDENT_CPU_AUDIT.json`: all 191 NPZ at 00:27:58 on October 8; 2674 array members, 9,550,000 point labels, 382 role slices and 2286 valid faces.
- `analysis/INDEPENDENT_HISTORICAL_FACE_AUDIT.json`: all 191 NPZ at 00:30:02; 1146 old faces, four raw fields per archive, 57,300,000 point/face comparisons and all 205 intake size/SHA checks.

Those computations belong to the original auditor. Their detailed rows, source coverage and hashes were checked here; their execution counts are not claimed as my new executions. The delegating task reported that the earlier agent remained `pending_init` without a final report; I did not attest host lifecycle state. No final report existed at these output paths before this audit. All earlier reports, checker sources, CPU results and warnings are preserved.

## Findings

| Check | Status | Finding |
|---|---|---|
| GT provenance | PASS, bounded | Dataset object-member labels and boxes; no prediction-derived GT. Raw dataset payload was not reopened. |
| Normalization | PASS | Ordinary IoU and native language/semantic scores; no performance scaling by the model's own maximum. |
| Actual result existence | PASS | Exact 205-file archive; three complete 191-row result sets and actual closed forward logs. |
| Call reachability | PASS | Collector calls the observed model route and support helper; persisted values match complete CPU checks. |
| Scope | WARN | Selected development errors, one seed, no full-validation or effective-module claim. |
| Historical prediction equivalence | **FAIL, rejected** | Fresh numeric slots do not recover old stochastic Queries or support. |
| Old coordinate provenance | PASS, bounded | All coordinate-compatible raw members considered; ambiguous cases remain ambiguous. |
| Cleanup | PASS, operational | Exact redundant remote NPZ manifest, all local evidence retained; no scientific acceptance implied. |

### GT, scores and actual calls

The hash-bound `pvground_g_p2_20261002/complete/source/joint_det_dataset.py:585` reads ScanRefer annotations; `:637` uses `object_id`; `:1103` sets target masks from `scan.three_d_objects[tid]['points']`; `:1106` obtains dataset object boxes. `:1274` resets `scan.pc` from `scan.orig_pc`. The collector selects validation/no augmentation, asserts full input identity and exact GT at `collect_support_cases.py:149` and `:150`, and archives dataset target labels at `:168`.

This is `real_gt` **offline diagnostic** evidence. No point-support GT is supplied in the collector's model input dictionary (`collect_support_cases.py:133`). All 191 row identities and GT boxes match their historical sources. The previous full member audit found a maximum target-member/GT-bound difference of 2.384185791015625e-7 m. The full XYZRGB hash remains collector-recorded: RGB is absent from NPZ, and neither this reviewer nor the original auditor reopened the raw dataset payload.

`native_root_bbs.py:4` uses the existing softmax and language-span reductions across all 256 candidates. Saved scores and true-max selection were checked in the full CPU audit. Original all256 semantic token logits and language maps are absent, so the score itself cannot be regenerated from those missing inputs. `analyze_closed.py:25` uses ordinary intersection-over-union and `:74` strict `>0.25` / `>0.5` thresholds. `support_evidence.py:49` uses actual target/total member counts. Historical coverage and volume ratios divide by dataset GT volume (`pvground_reference_keep_20261006/postrun/analyze_cached_reference_errors.py:36`). Scene-span normalization in the geometry provider is coordinate representation, with raw coordinates restored, not score inflation.

`collect_support_cases.py:136` calls `observed_readback_forward`; its source requires exactly the refiner, readback and native semantic head in order (`readback_preflight_checks.py:48`). The collector then calls `analyze_case` and saves its result (`collect_support_cases.py:176`). The closed collector log has 161 sequential forward progress records ending at 191 cases, and both exit files are zero. This is evidence of the completed collection, not a claim that this reviewer executed a model. Optional witness/training helpers not reached by this collection are not counted as checks.

### Historical equivalence is still rejected

The actual imported model's `forward` invokes `_generate_queries` (`models/pv_ground.py:442`); the Query generator uses `GumbelSampling` (`:316`), and `F.gumbel_softmax` is unconditional (`:609`). Evaluation mode disables dropout but does not bypass this sampling.

The collector resets RNG before a subset loop (`collect_support_cases.py:117`, `:120`). It executes 161 original B8 contexts / 1288 context rows, beginning at old batch 16; it does not advance through omitted old full-validation batches. Keeping complete batches and input identity therefore does not restore old stochastic Query realizations.

The complete actual evidence records:

- **167/191 changed selected Query numbers**: 108/128 in the historical overextended cohort and 59/63 in the historical missing-extent cohort.
- **191/191 changed old-slot coarse boxes** at 2e-6 m tolerance; 48/191 old-slot reference boxes also change.
- All **24 rows with unchanged selected Query numbers still have changed coarse boxes**.
- Row 128, historical slot 116: **0.027122974395751953 m** maximum coarse-box difference. I directly reconfirmed this from its NPZ and primary cached box. The red-check record's two exit-1 executions remain preserved; they were not replayed.
- Deployed roles have 191 valid ranges. The historical numeric-slot role has 190 valid ranges and one invalid range, row 7599. My spot check found zero active superpoints, no invented faces and the existing native prior retained.

Thus, the label `historical` in immutable per-row output means **a fresh-forward numeric Query slot bearing the old selected index**. The corrected summary states this. It cannot support original paired prediction causality, recovered old foreground or an explanation of the original5598/4848 result at the member-selection level. The historical128/63 groups select expressions; they are not necessarily the geometry groups of the new predictions.

### What the descriptive evidence does support

For the **fresh deployed roles**, 127/128 historical-overextended expressions and 46/63 historical-missing-extent expressions have a pure-background extremal source superpoint. Mixed extremal superpoints occur in 7/128 and 2/63 respectively. These are current-forward descriptive counts on a selected cohort; background/mixed row groups overlap.

The separate old-coordinate helper reads only `xyz`, `target`, `superpoint` and `root_gt`, and takes face coordinates from old cached boxes (`historical_face_provenance.py:14`, `:22`). It includes every raw point whose axial coordinate lies within2e-6 m (`:31`). It does not restrict candidates by fresh Mask/Query information, and it does not reconstruct the old foreground.

| Old expression cohort | Rows | At least one background-certain old face | At least one face with all possible source SPs pure background | Rows with any possible mixed source | Missing old GT faces with observed target beyond |
|---|---:|---:|---:|---:|---:|
| Overextended | 128 | 128 | 128 | 7 | 13/13 |
| Missing GT extent | 63 | 44 | 44 | 2 | 164/164 |

Across **1146 old faces**, 821 have only background-compatible members, 297 only target-compatible members, and **28 remain target/background ambiguous**. There are820 faces whose possible source superpoints are all pure background, and17 with possible mixed sources. Row categories overlap. The nearest-member coordinate error is at most2.384185791015625e-7 m, below the fixed2e-6 m tolerance.

Two targeted checks tested distinctions that matter for interpretation:

- Row128's old `z+` face has two coordinate-compatible members: one target and one background. The output preserves ambiguity.
- Row8368's old `x-` face has a single compatible background member from mixed superpoint676. Background-member certainty does not imply a pure-background source superpoint.

The 177 missing old faces all have actual archived target members beyond them. This shows omission by those extents. It does not establish a learned correction, guaranteed recoverability, or deployment feasibility. Any possible mixed source remains a possibility, not proof the old mask selected that superpoint. These conclusions are conditional on the recorded identical-input provenance chain; no old Mask or Query recovery is implied.

### Archive retirement and preserved audit history

The source at `cleanup_archived_authorized.py:16` requires a closed task; `:30` binds exactly191 archive names; `:33` checks local bytes/SHA; `:37` fixes the remote root; `:48` checks every remote copy before deletion; `:51` unlinks only those validated paths. The selected protected checkpoint hash is checked before and after by that recorded remote operation.

`array_cleanup_receipt.json` lists exactly the intake's191 NPZ entries and126,198,802 released file bytes, with `best_weights_touched=0`, `datasets_touched=0`, and `text_logs_touched=0`. My current archive check confirms all205 local intake files still match, including all191 NPZ. Remote deletion and protected-weight assertions are supported by the reviewed source and saved exit-checked receipt; this audit did not make a new SSH connection or reopen a checkpoint.

Storage retirement follows its own closed-task/archive gate. It is not scientific acceptance. `CLEANUP_SOURCE_REVIEW`, `ANALYSIS_SOURCE_REVIEW` and `HISTORICAL_FACE_SOURCE_REVIEW` were read as source reviews, not substituted for actual evidence.

The only current-versus-previous hash drift is `EXPERIMENT_PLAN.md`. Its later archive-retirement appendix was read; prior review hashes remain unchanged. No collector, model/data source, NPZ, result row or full CPU result changed in those bindings. Earlier planned/pending statements in the plan are temporal preparation history, not evidence of current nonexecution.

## Claim boundaries and disposition

The preliminary missing-archive finding is resolved by the completed intake and full CPU results. Its historical-equivalence finding remains unresolved and rejected. This additional report preserves both the original preliminary record and the failed recovery evidence.

No new9508 score, general validation incidence, multiseed robustness, effective trained module, or one of the required three contributions is established. The retained5598/4848 score is historical context, not newly evaluated here. Validation GT labels, grouping thresholds and oracle decisions stay offline and cannot become training targets or inference rules.

The empty `blocking_findings` list applies only to the two explicitly bounded descriptive claim classes. It does not clear any historical-recovery claim. The JSON records `historical_equivalence_status=FAIL_REJECTED_UNRESOLVED`, the preserved P2 finding and explicit rejected claims.

Reproducible additional checks: `analysis/audit_adjudication_private.py`, SHA256 `ddeca8d82b4356387e635272373d5122a793800ac2ef928fd0bdc51f24573189`, executed once successfully. Its current-input hashes and results are embedded in the JSON. This additional task performed no NN/GPU work, checkpoint or raw-dataset payload read, SSH/network access, deletion, model/data change or publication.

