# Preliminary actual-artifact integrity review

**Verdict: WARN — preliminary only.** Snapshot: 2026-10-07 23:13:58 +08:00. This review does not accept a completed archive or final CPU analysis. It is a fresh-context, same-family, provisional experiment-audit review. Requested reviewer: `gpt-6-astra`, effort `max`; the actual serving backend/effort is unattested.

The collector produced real diagnostic evidence for 191 historically selected development errors, covering 52 scenes and 83 scene/target pairs. I read all 191 collected records, all 191 cohort records, and the 9508-row historical identity source. I independently executed CPU checks on the 61 complete NPZ archives present at the snapshot: 122 role slices, 732 faces, 854 array members, and 3,050,000 point labels. These checks passed. The remaining 130 archives, `complete/INTAKE.json`, and `analysis/SUMMARY.json` were not accepted or assumed present. No `analysis` directory was created by this preliminary review.

The main scientific limitation is substantial prediction drift. Directly counting the 191 JSON records gives **167 changed selected Query numbers**, split 108/128 in the historical overextended group and 59/63 in the historical missing-extent group. Recorded validity is 191/191 for the deployed role and 190/191 for the historical-number role. An earlier observation of three changes cannot describe this collection.

## A. GT provenance — PASS within the inspected source/archive chain

The actual imported `joint_det_dataset.py` hash matches the provided local source. It reads ScanRefer validation annotations at lines 585–597 and takes the object identity from `object_id` at line 637. Object member labels are set from `scan.three_d_objects[tid]['points']`, and boxes come from `scan.get_object_bbox(tid)` at lines 1086–1111; they are not generated from model predictions. The collector saves `batch_cpu['gt_masks'][bid, 0]` and the dataset center/size GT at lines 150–173. Archived member labels were independently enumerated, not inferred from predicted masks. Their GT member bounds agree with archived GT box bounds within 2.087e-7 m in the 61 executed archives.

The raw dataset payload itself was not opened. This is a source-hash and archived-member provenance audit, not a new raw-dataset reconstruction. The archives contain XYZ, membership, and GT target labels but no RGB; full XYZRGB input hashes cannot be regenerated from these archives. The collector's full input identity assertion at lines 148–152 is runtime-recorded evidence, not an independently repeated full-input hash here.

## B. Score and metric path — PASS with a reconstruction limit

`native_root_bbs.py:4–9` applies the existing semantic-logit softmax and language span maps to all candidate scores. It does not use spatial GT boxes or point-support labels. `collect_support_cases.py:138–173` retains all 256 candidates and selects the maximum native score. All 61 available archives contain 256 finite scores, and their deployed Query is an actual maximum. Raw all-256 token logits and language maps are not archived, so this review does not claim independent reconstruction of the score from those inputs.

`analyze_closed.py:25–31` uses ordinary intersection-over-union geometry, and lines 74–76 use strict `> 0.25` and `> 0.5` hit thresholds. No diagnostic performance score is divided by a model-output maximum. Superpoint target fractions divide actual GT member counts by actual member counts. Independently recomputed text/query fusion matched exactly in all 61 archives; fused support and six-face evidence also matched, with maximum reference-bound discrepancy 1.193e-7 m.

## C. Actual execution and file existence — WARN, final intake pending

`wait.json` and `complete/receipt.json` record closed execution, 161 forwards, 1288 original-context rows, 191 saved cases, zero optimizer updates and zero weights created. The receipt's rows hash matches the actual 191-record JSONL. I verified 84 unique input-file hashes, including 61 complete archives, five source files against actual imported-source hashes, and four collector-side files against their executed copies.

The independent CPU checker is `audit_actual_support_independent.py`; its actual snapshot result is `PRELIMINARY_CPU_CHECK.json`. It imports no project evidence helper, torch, checkpoint loader, or network client. No NN forward or training was performed by the reviewer. Missing archives and final summary remain an explicit barrier to final archive acceptance.

## D. Reachability — PASS for collected evidence; final analyzer not claimed executed

`collect_support_cases.py:176–181` calls the support helper and writes its per-row result. The saved records and 61 independently enumerated raw slices agree. `analyze_closed.py` was read as source; a source implementation alone is not evidence of its future successful execution.

## E. Historical identity and scientific scope — WARN, original-prediction attribution unsupported

The actual imported `models/pv_ground.py:311–321` generates Queries through `GumbelSampling`. Its `forward` at lines 597–611 unconditionally calls `F.gumbel_softmax`, with no evaluation-mode bypass. The collector resets the RNG at `collect_support_cases.py:120`, then runs only the subset of 161 retained B8 contexts at lines 117–127. The manifest's first original block is 16. It does not restore the original full-evaluation RNG state for each original batch.

Keeping each B8 context and the exact input point identity therefore does not restore the historical stochastic candidate realization. In this collection, the role named `historical` means **the historical numerical Query slot evaluated in the current fresh subset forward**. Its original prediction or original point support has not been recovered. This is also directly visible in the arrays: at tolerance 2e-6, all 61 available historical-number coarse boxes differ from their archived historical boxes; 23/61 historical-number reference boxes differ. Even all nine rows with an unchanged selected Query number have changed coarse boxes. For example, row 191 retains Query 190 but has a 0.0325923 m maximum center/size difference. Row 2013's historical-number slot 132 differs from the old coarse box by 1.3752844 m and the old reference box by 1.7580727 m.

The data supports descriptive current member/boundary evidence on a cohort selected from historical errors. It does not establish a paired causal explanation of the original retained 5598/4848 result, general validation incidence, a new 9508-expression accuracy score, a successful trained module, or any of the required three effective contributions. The historical 128/63 grouping is cohort provenance, not proof that each current prediction still belongs to that geometric class. Preliminary archives arrive in row order; their support counts must not be generalized to all 191 cases.

## F. Evaluation type

`real_gt` targeted development diagnostics. The predicted reference is an evaluated model output, not the ground truth. GT may be used to describe these cases; no validation GT oracle, threshold, or label is justified for training or deployment by this evidence.

## Open findings

1. Final artifact acceptance waits for all 191 size/hash-matched archives, closed intake, actual CPU summary, and a bounded all-191 independent follow-up.
2. Publication must explicitly describe fresh-subsequence predictions and historical-number slots, and must exclude restoration of historical support or paired original-error causal attribution. No full NN replay or model change is requested by this review.

The report and CPU snapshot must be preserved when a later final review is written.
