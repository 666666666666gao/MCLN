# Experiment audit — actual closed Mask-support pair

Date: 2026-10-08. Overall verdict: **WARN**. Integrity status: **WARN**, with no blocking defect found in the recorded full-9508 Box counts or their current metric-best ordering.

This is a fresh-context, same-family, provisional audit. Requested reviewer: `gpt-6-astra`, effort `max`. Actual backend, model, and effort are **UNATTESTED**; the requested routing is not backend attestation. No external reviewer, overlay, model/optimizer execution, inference replay, SSH, or source modification was used. The reviewer independently read the executed evaluation path and ran an artifact-only CPU verifier. The earlier source-only PASS is not actual-result evidence.

Paths below use:
- `R` = `C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2`
- `C` = `R/complete_fit`
- `D` = `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source`
- `H` = `C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle`
- `P` = `C:/Users/gb/.codex/tmp/pvground_mask_reference_20261006/complete_fit/fused_mask_reference/initial_formal`
- `G` = `C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json`

The JSON report binds substantive inputs by SHA256, including all 2523 intake files. `AUDIT_CPU_CHECK.py` contains the independent verifier; `AUDIT_CPU_EVIDENCE.json` contains its complete output, including repaired/damaged row IDs. `AUDIT_SUPPLEMENT.json` and `AUDIT_PARENT_SOURCE_DIFF.patch` bind the additional source and training checks. The C: workspace resolves to a D: backing directory; these are path aliases, not two artifact sets.

## Actual results

All Box counts below are native last/bbs Top-1 counts at **strict** IoU > 0.25 / > 0.5, with denominator 9508. Independent float64 Box recomputation produced exactly the same selected counts. Mask columns are a recount of stored selected Mask IoUs; raw predicted Masks and GT Masks were not retained in these NPZ files.

| Model / evaluation | Updates per new head | Box hits >.25 | Box hits >.5 | Box accuracy .25 / .5 | Stored Mask hits >.25 / >.5 | Stored Mask mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Historical protected parent | 0 | 5598 | 4848 | 58.8767% / 50.9886% | — | — |
| Initial content | 0 | 5598 | 4848 | 58.8767% / 50.9886% | 5812 / 5133 | 47.107671% |
| Initial box_conditioned | 0 | 5598 | 4848 | 58.8767% / 50.9886% | 5812 / 5133 | 47.107671% |
| Final same-forward parent | — | 5598 | 4848 | 58.8767% / 50.9886% | — | — |
| Final content | 3723 | **5598** | **4856** | **58.8767% / 51.0728%** | 5812 / 5126 | 47.082638% |
| Final box_conditioned | 3723 | 5596 | 4849 | 58.8557% / 50.9992% | 5811 / 5130 | 47.081862% |

Evidence: `C/initial_formal/receipt.json:15`, `C/formal/receipt.json:6`, `C/formal.log:25`, independently verified from all lines of `C/initial_formal/rows.jsonl` and `C/formal/rows.jsonl`, and all 2378 formal NPZ files. Independent aggregate evidence starts at `R/analysis/AUDIT_CPU_EVIDENCE.json:73` and `:235`. Historical counts were recounted directly from `P/rows.jsonl`.

| Comparison (before → after) | .25 repairs / damages / net | .5 repairs / damages / net |
|---|---:|---:|
| Final same-forward parent → content | 6 / 6 / 0 | 14 / 6 / **+8** |
| Final same-forward parent → box_conditioned | 5 / 7 / −2 | 8 / 7 / +1 |
| Final content → box_conditioned | 3 / 5 / **−2** | 2 / 9 / **−7** |
| Independent initial content → final content | 6 / 6 / 0 | 14 / 6 / +8 |
| Independent initial box_conditioned → final box_conditioned | 5 / 7 / −2 | 8 / 7 / +1 |
| Historical protected parent → final content | 6 / 6 / 0 | 14 / 6 / +8 |
| Historical protected parent → final box_conditioned | 5 / 7 / −2 | 8 / 7 / +1 |

Evidence: `C/paired_support_loop.py:319`; independent comparison data and exact affected row IDs at `R/analysis/AUDIT_CPU_EVIDENCE.json:407`. Both independent-pass parent comparisons have **zero repairs and zero damages at either threshold**. Equal threshold counts do not establish equal outputs.

**Metric-best selection is supported within the declared three eligible candidates:** content terminal, box_conditioned terminal, and protected parent. The rule is dual-target gate first, then >.5 hits, then >.25 hits, with an exact tie retaining the protected parent. Here there is no tie: 4856 > 4849 > 4848, while none passes the dual gate. Zero-output initial evaluations are diagnostics and are correctly excluded. This is selection among this audited candidate set, not proof of a global optimum or a fully revalidated standalone deployed checkpoint. Evidence: `R/postrun/analyze_support_correction_formal.py:199`, `:232`; independent result `R/analysis/AUDIT_CPU_EVIDENCE.json:859`.

The required same-model minimum is **5620 / 4764**. Content remains **22 hits short** at .25; box_conditioned is 24 short. Three effective contributions and the subsequent independently trained fixed full models on Nr3D/Sr3D remain unproved. Evidence: `G:9`, `:17`, `:22`, `:38`; `R/postrun/analyze_support_correction_formal.py:236`.

## A. Ground-truth provenance — WARN

The inspected source and hash-bound runtime import support **real dataset GT**. The exact dataset module used at runtime is `D/joint_det_dataset.py` (SHA256 `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d`), **not** the different `C/PV-Ground/src/joint_det_dataset.py` bundled beside the model. The actual runtime import and its digest are recorded in `C/imports.json:8` and `:16`; source import checks occur at `C/run_mask_support_pair.py:49` and `:97`. Its archive matches both runtime import and the appearance-source manifest; this was independently checked, not assumed from directory names.

The GT path is:
1. Read ScanRefer filtered JSON/text split files, retain scene ID and annotated object ID: `D/joint_det_dataset.py:585`, `:634`.
2. Load scan objects from the dataset's split scan pickle: `D/joint_det_dataset.py:205`.
3. Construct target occupancy from `scan.three_d_objects[tid]['points']` and target boxes from `scan.get_object_bbox(tid)`: `D/joint_det_dataset.py:1086`, `:1100`, `:1105`. Box jitter is restricted to augmented train mode at `:1112`; formal evaluation disables augmentation.
4. Return `center_label`, `size_gts`, and `gt_masks`: `D/joint_det_dataset.py:1387`.
5. Native evaluator parses those GT fields: `C/PV-Ground/src/grounding_evaluator.py:534`, `:878`.
6. Formal producer writes the same root GT to NPZ and rows: `C/paired_support_loop.py:263`, `:265`, `:283`. All archived `root_gt` arrays exactly equal their row GT, with row/scene/object/point-digest alignment across initial, final and historical passes.

The new corrector receives predicted query/support features, observed points, predicted native coarse geometry, native Masks, and alpha. It does not receive GT object IDs/boxes/Masks or oracle labels: `C/mask_support_corrector.py:24`, `:52`; `C/support_pair_forward.py:22`. GT-dependent Hungarian matching is training-only at `C/paired_support_loop.py:81` and `C/matched_mask_objective.py:20`. Evaluation computes oracle labels only after native score ranking at `C/paired_support_loop.py:280` and `:298`.

The original input protocol is preserved: scene points/voxels, utterance, superpoints, detector boxes/mask/classes enter the forward at `C/run_mask_support_pair.py:232`; `butd=True`, `butd_cls=False`, `butd_gt=False` are explicit at `:189`. Detection inputs originate in GroupFree prediction files at `D/joint_det_dataset.py:1203`; GT-proposal replacement branches are controlled separately at `:1360`. No new deployed V99 ranker is called.

The no-added-GT conclusion has a precise scope. The existing benchmark batch contains GT-related annotation fields and language-position maps. Native bbs explicitly uses the language maps (`H/native_root_bbs.py:4`; native evaluator `:218`, `:281`); the dataset also computes GT-related auxiliary fields (`D/joint_det_dataset.py:1161`, `:1337`). These facts do not make the new corrector a GT oracle, but they forbid the broader assertion that the entire underlying benchmark protocol is free of GT-related fields.

**Limitation driving WARN:** this audit did not independently regenerate all GT from the original scan/annotation bytes. The referenced actual `src/visual_data_handlers.py` digest is `6d1d4f0c792e8b86f238d88c02f1933f945313a3831d0bf9eabb9bf4580bb177` (`D/appearance_source_manifest.json:470`); the bundled PV version has a different digest and is not substituted as the actual loaded module. Original `val_v3scans.pkl` and raw GT Masks are absent from this collected evidence set. Source/runtime bindings and the entire GT-to-result handoff pass; raw-data provenance beyond those bindings remains a qualification, not evidence of fake GT.

## B. Metric and supervision normalization — WARN

**Box metrics pass.** Native hit counters use GT object count, strict IoU thresholds, and no prediction-statistic denominator (`C/PV-Ground/src/grounding_evaluator.py:289`, `:296`, `:303`). The final reported denominator is 9508 (`C/paired_support_loop.py:307`, `:315`; analysis `:139`). Softmax used to rank predictions and standard Dice loss normalization are not metric rescaling.

Independent CPU computation covered **14,604,288 candidate Box IoUs**: 9508 × 256 × 3 geometries × 2 stages. The maximum selected native-versus-float64 IoU difference was approximately **6.62009e-6**, with **zero** selected >.25 or >.5 threshold flips for every geometry/stage. Every stored selected query had the maximum native score, with **no tied maximum**. Every NPZ selected Box matched the same row/query. All Top-16/32/64/256 oracle labels also agreed with independent CPU geometry checks. Evidence: `R/analysis/AUDIT_CPU_CHECK.py:39`, `:134`; `R/analysis/AUDIT_CPU_EVIDENCE.json:138`, `:299`, `:334`.

**Native active supervision is preserved.** Query focal/Dice and fused focal/Dice carry coefficients **5, 1, 10, 2**, outside the native detector-layer division by seven (`C/PV-Ground/models/losses.py:562`, `:583`, `:623`, `:947`). The matched objective uses actual valid GT count (`C/matched_mask_objective.py:35`), native majority-superpoint GT at `:40`, and the frozen parent final Hungarian assignment. Native text Mask slots are duplicated across all 256 candidates (`C/PV-Ground/models/pv_ground.py:545`), so selecting the first matched-count text slots in the native loss is consistent with this producer.

All 3723 training records have matched count = valid GT = actual batch rows, including the last batch of two. Every one of the 29778 assignments points to actual GT slot 0; all 256 possible query IDs appear over the run. There is **one matched positive Query per expression**, not supervision of all 256 Queries against that same target. Background/other instances contribute zero target membership within the matched object's Mask; there is no new background-query or all-instance supervision branch. Frozen text-Mask/correspondence/Box/semantic branches are not counted as newly optimized losses. Evidence: `C/train.jsonl:1`, `:3723`; `R/analysis/AUDIT_CPU_EVIDENCE.json:26`; `C/matched_mask_objective.py:33`.

The runtime asserts finite gradients for all ten head parameter tensors and absence of parent/other-arm gradients (`C/paired_support_loop.py:98`). Actual two-step preflight witnesses show the zero-initialized output learns first and hidden tensors receive gradients after the first update (`R/preflight_complete/preflight.json:106`, `:274`); actual fit records show nonzero whole-head gradient norm on every update. This establishes tensor-level coverage, not nonzero task gradient in every individual weight coordinate. In the content arm the nine explicit geometry input channels are zero by design.

**WARN 1:** only selected Mask IoUs, not raw Mask arrays, were saved. Stored IoUs exactly reproduce Mask hits/mIoU and were reconciled with native evaluator accumulation during the run (`C/paired_support_loop.py:290`, `:314`). That is not independent raw-Mask recomputation. The NPZ schema at `:265` contains Boxes, scores and root GT only.

**WARN 2, actual stability evidence:** box_conditioned has 346/3723 raw gradient norms above the common 0.1 clip threshold (content: 52/3723). At step 523 its pre-clip norm is **1,206,195.5**, and at step 3211 its loss is **154.9939575** (`C/train.jsonl:523`, `:3211`; `R/analysis/AUDIT_SUPPLEMENT.json:10`). All are finite and clipping executes before the optimizer step (`C/paired_support_loop.py:109`), so these do not invalidate counted outputs. They do preclude a blanket stability claim. The nine geometry channels divide by clamped predicted coarse sizes (`C/mask_support_corrector.py:32`); that is a plausible source of sensitivity, **not a demonstrated cause** without the corresponding runtime activations.

## C. Existence, completion and collection — PASS

The original observer's exit 1 is a transport failure, not a NN failure: `R/observer_original63704_closed_transport_failure.json:81` contains the SSH-banner exception after observing the controller alive. Later inspection still sees the same controller PID 872610 alive and terminal-evaluation progress (`R/fit_observer_transport_recovery_inspection.json:1`). The resumed observer explicitly did not restart training (`R/fit_observer_resumed_wait.json:2`, `:8`).

Authoritative completion is recorded at 10:20:12 CST with the three phases completed and exit 0 (`C/fit_status.json:2`, `:4`, `:17`; `C/fit_controller.exit:1`). All three actual phase exit files also contain 0. The controller writes phase exits only after child wait and marks complete only after all modes and protected-parent hashes pass (`C/controller.py:42`, `:50`, `:58`). The later observation at 10:23:22 sees controller dead, exit 0 and complete status (`R/fit_wait.json:2`, `:6`, `:30`). The prior progress line at 9216 rows is correctly subordinate to final receipt/exit evidence.

Collection completed at 10:28:33 CST and was gated on controller closure and remote complete/exit0, not elapsed time (`R/collect_closed_fit_authorized.py:12`, `:26`). I independently verified all **2523 files / 234674076 bytes**, every declared SHA256 and size, no missing items and no undeclared extra file except INTAKE itself (`C/INTAKE.json:12621`; `R/analysis/AUDIT_CPU_EVIDENCE.json:2`). No checkpoint/temporary weight files were copied, as the collector explicitly excludes them at `:30`.

Actual fit rows are unique, cover the declared 29778 exactly once for each arm, and the 6887 module holdout is disjoint with union 36665. The archived split protocol IDs agree exactly. These are distinct from the formal 9508 validation expressions; formal and module-holdout scans do not overlap. Evidence: `C/train.jsonl:1` through `:3723`; `D/split_protocol.json:1`; `R/analysis/AUDIT_CPU_EVIDENCE.json:26`, `:397`. The 6887 receipt explicitly has `formal_rows=0` (`C/receipt.json:5`).

Separate parameter storage and AdamW optimizers are created at `C/paired_support_loop.py:35`, `:40`, `:42`. Checkpoints save head deltas, full optimizer state, step, row IDs, parent/spec digests and RNG states (`:126`); formal restoration requires step3723, all row IDs, strict head loading, optimizer loading and all optimizer steps3723 (`:145`). Both actual formal_restore receipts have ten moment/step states, exact groups/moments/steps and distinct terminal digests (`C/content/formal_restore.json:2`, `C/box_conditioned/formal_restore.json:2`).

**Scope of PASS:** this establishes actual recorded job completion, artifact integrity, and executed save/restore checks. The auditor did not independently deserialize terminal checkpoint bytes; they were deliberately excluded from intake. It is not a new attestation of current remote checkpoint availability, a retention operation, or a post-audit cold restore.

## D. Active routes, source difference and integration — WARN

Both trained heads are active in actual final outputs. Relative to the same-forward parent, content changes **95105/2434048** candidate Boxes and **533/9508** selected Boxes; box_conditioned changes **525792/2434048** candidate Boxes and **363/9508** selected Boxes. Initial arm arrays are exactly equal to the parent for all candidates. Evidence: `R/analysis/AUDIT_CPU_EVIDENCE.json:235`. This rules out a source-only or disconnected new head as the explanation of the reported changes.

The route is native Query Mask logits → learned support correction → original scalar-alpha logit fusion → hard predicted support extent → frozen zero-offset Mask-reference Box. It retains one native bbs score and one selected Query for Box and Mask. The formal pair computes both heads after one frozen parent forward and retains identical semantic logits, text Masks and alpha (`C/support_pair_forward.py:5`, `:22`, `:36`). The native integration inserts the corrector after native Masks and before final Box reconstruction/readback/native semantic head (`C/PV-Ground/models/pv_ground.py:555`, `:563`, `:572`, `:580`).

No direct new Box loss trains the head: geometry is detached under no_grad (`C/support_pair_forward.py:10`); corrected Mask logits receive matched native Mask losses. The parent, old Box head and readback are frozen. The Box-refiner output and readback output are zero (`C/run_mask_support_pair.py:125`; `C/support_pair_forward.py:16`), so their learned hidden branches do not establish new effective contributions. The Mask-derived reference remains active even with zero learned Box offsets (`C/mask_reference.py:18`, `:60`; `H/pvground_boundary_box_refiner.py:18`, `:105`). Invalid predicted extents use the existing predicted coarse reference, with no IoU/GT gate (`C/mask_reference.py:24`); final invalid selected counts are 40 for both arms.

The arms share exact initial tensor values, head architecture, 27841 parameters/10 state tensors, seed2027, batches, one-pass budget, optimizer settings and frozen-parent forward. The controlled difference is only whether the nine explicit coarse-box-relative channels are zero or enabled (`C/paired_support_loop.py:35`; `C/mask_support_corrector.py:32`). “Content” still sees geometry-bearing backbone/query features and superpoint member counts; it does not mean an entirely geometry-free model.

The locked prior source-port digest matches `C/pair_spec.json:43`. An independent comparison of old and new source-port file maps shows exactly one changed PV tree file, `models/pv_ground.py`; the archived old source bytes match the old manifest. The precise changes add the optional corrector and feature handoff (`R/analysis/AUDIT_PARENT_SOURCE_DIFF.patch:3`, `:11`). All separately declared new runner/helper files were hash-verified. This is a content-manifest comparison, not an invented Git commit or backend execution attestation.

**Limitation driving WARN:** actual two-update preflight rebuilt the full CPU state, restored head/optimizer on GPU and tested native integrated output against the same captured parent inputs (`C/paired_support_loop.py:164`; `R/preflight_complete/preflight.json:8`). It explicitly says **gpu_parent_cold_reconstruction_executed=false** at `:19` and `:50`. Actual final formal evaluation restores step3723 heads and runs the paired application path (`C/paired_support_loop.py:340`), while standalone native integration was witnessed at step2. Thus the record supports active corrected-head results and a source-consistent native integration, but not a new full-9508 cold-GPU replay of the final standalone 1314-tensor model. No fabricated runtime attestation is supplied.

Native bbf/mask-semantic routines are called by the original evaluator (`C/PV-Ground/src/grounding_evaluator.py:194`) but are not promoted to the primary bbs claim. Legacy helper routines such as distribution-loss and cached-head diagnostics are not treated as newly executed training contributions.

## E. Scope and claim ceiling — WARN

This is one seed, one fixed training budget, two arms, 9508 expressions across **141 scans** per formal pass. It provides a paired local experiment, not robustness across seeds or datasets. The separate 6887 module holdout was already seen by prior pretraining according to its split protocol (`D/split_protocol.json:1`); it must not be represented as globally unseen from-scratch test data.

The formal point digest, scene ID, target ID, row ID and root GT match exactly across initial/final/historical passes (`R/analysis/AUDIT_CPU_EVIDENCE.json:397`). Parent state equality does **not** mean stochastic/numerical output equality:
- 4 selected-query changes and 40 selected-parent-Box changes between initial and final.
- All 9508 expressions have at least one changed parent score and one changed candidate-Box element.
- 2388376 score elements differ (maximum absolute change 0.4960959535), and 5099805 Box elements differ (maximum 1.6873032451).

Evidence: `R/analysis/AUDIT_CPU_EVIDENCE.json:841`. The producer uses Gumbel sampling even in the native forward (`C/PV-Ground/models/pv_ground.py:610`, `:622`); no mechanism-specific cause of the observed drift was established. These differences should not be dismissed as harmless epsilon-level noise. In this recorded comparison they happen not to change any parent's selected threshold-hit label. The positive content comparison remains supported because it uses the **same final parent forward**.

All256 availability is an oracle diagnostic, not deployed accuracy:

| Final arm | Top16 oracle .25 / .5 | Top32 | Top64 | All256 |
|---|---:|---:|---:|---:|
| content | 6234 / 5525 | 6733 / 6018 | 7756 / 7008 | 8923 / 8167 |
| box_conditioned | 6233 / 5529 | 6729 / 6019 | 7757 / 7004 | 8939 / 8160 |

For content, 3325 / 3311 rows have a >.25 / >.5 candidate somewhere in all256 but its deployed Top1 fails; 585 / 1341 have no such candidate. Box_conditioned: 3343 / 3311 available-but-unselected and 569 / 1348 with none. These distinguish candidate availability from ranking, but cannot be reported as a deployed gain. Evidence: `C/paired_support_loop.py:298`; `R/analysis/AUDIT_CPU_EVIDENCE.json:335`.

The paired explicit-geometry arm loses **2 / 7** hits to content, despite a small +1 strict hit against the parent. The observed +8 content strict hits, unchanged wide hits, and slightly lower stored Mask mIoU do not establish three effective contributions or a generally better segmentation model. No extra seeds are required by this audit; claims must respect the user's selected single-seed scope. Nr3D/Sr3D results for a fixed full model are not supplied here.

## F. Classification and permitted claims — PASS

| Evidence / claim | Classification or verdict |
|---|---|
| Native selected Box performance | **real_gt**, independently recounted from archived Box predictions and saved dataset root GT |
| Selected Mask performance | **real_gt** by producing source; stored selected-IoU recount only, not independent raw-Mask verification |
| Matched head supervision | **real_gt**, one actual root target per training expression with native majority-SP target |
| TopK/all256 availability | **GT oracle diagnostic** within real_gt evaluation; requires GT to identify a successful candidate and is not the deployed selector |
| Same-forward parent/arm agreement, source digests, restore equality | Integrity/consistency checks, not benchmark-quality metrics |
| Content 5598/4856, +8 strict hits on this paired run | **Supported**, with single-seed and same-forward scope |
| Box-conditioned geometry is more effective than its content control | **Unsupported** by this result (−2/−7 paired net) |
| General Mask-quality improvement | **Unsupported**; stored Mask mIoU decreases and raw Masks are unavailable |
| Content is metric-best among the three eligible candidates | **Supported**, subject to the declared tie rule and checkpoint-restoration limits |
| Both current user thresholds are met | **Unsupported** |
| Three effective contributions are established | **Unsupported** |
| Final complete model has been independently trained on Nr3D and Sr3D | **Unsupported** |
| Independent raw GT/Mask rederivation or final cold-GPU deployment attestation | **Not performed / unproved** |

## Blocking and nonblocking findings

There is **no blocking integrity defect for the as-recorded full9508 Box table or current metric-best candidate**.

The following block stronger acceptance claims: the .25 target is unmet; the explicit-geometry contribution is not supported against its paired control; three effective contributions and fixed-full-model Nr3D/Sr3D training are unproved. A claim of independently rederived raw Masks or a freshly cold-restored final full GPU model would also exceed the evidence.

Nonblocking qualifications are: missing raw GT/Mask evidence beyond the reviewed source/runtime bindings; checkpoints excluded from local intake; actual clipped-gradient spikes in box_conditioned; substantial cross-pass candidate drift; inherited pretraining exposure to the module holdout; and unavailable backend/model/effort attestation. Earlier source-only reviews remain source-only, as stated at `R/postrun/ANALYSIS_AND_INTAKE_SOURCE_REVIEW.json:3`, `:18`.

Use the content terminal as the **metric-best candidate from this completed pair**, retain the explicit scope above, and keep the overall research goal open. This audit performs no checkpoint promotion, deletion, additional training, or deployment.

