# Experiment integrity audit — Mask/Box geometry responsibility

Date: 2026-10-05. Overall verdict: **WARN**. Integrity status: **warn**. **Blocking findings: none.**

The saved fixed-panel results are internally consistent and independently reproducible within the stated evidence limits. They support a current-checkpoint, training-GT diagnostic. They do not establish formal accuracy, intervention effectiveness, historical training responsibility, or prevalence outside this panel. One bookkeeping field requires qualification. A stale draft disk-space statement was corrected during review and the revised wording was read directly.

Reviewer: `/root/pvg_mask_geometry_terminal_integrity`, fresh context. Requested route: `gpt-6-astra`, reasoning `max`, `fork_turns=none`. Actual model/backend/effort are not independently attested. Semantic review is `same-family` / `provisional`; the deterministic local checks are separately identified. The prior source review was read as an artifact, not accepted as a substitute for this review.

Only local file reads and a NumPy CPU verifier were executed. No SSH, GPU, model construction/forward, optimizer, weight loading/saving, or experiment/analyzer replay occurred. The original one-shot `analyze_probe.py` was not run. Original `analysis/SUMMARY.json` remains byte-identical, SHA256 `01e1df8604ce53d6628974bcd7134e3359c685c4f7a8bce4db80dc9d6b733b0a`.

## Evidence paths

- `PROBE` = `C:/Users/gb/.codex/tmp/pvground_mask_geometry_responsibility_20261005`
- `HELPERS` = `C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle`
- `DATA` = `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source`
- `NATIVE` = `C:/Users/gb/.codex/tmp/pvground_runtime_bundle_20260908_v1/PV-Ground`
- `PORT` = `C:/Users/gb/.codex/tmp/pvground_geometry_readback_20261004/revision2/source_preview/PV-Ground`
- `PRIOR` = `C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/complete/quality/train.jsonl`

References below identify these exact files and source lines. Binary evidence is identified by NPZ file and array key; exact file bytes and SHA256 identities are in `EXPERIMENT_AUDIT.json` and `AUDIT_CPU.json`.

## A. Ground-truth provenance — PASS, with source/runtime limits

ScanRefer annotation IDs and expressions are loaded from dataset files (`DATA/joint_det_dataset.py:585-648`). Point membership masks come from the selected dataset object's member indices, and boxes from `scan.get_object_bbox`, not predictions (`DATA/joint_det_dataset.py:1086-1119`). The enabled train loader first transforms the current point cloud (`:895-908`, `:1324-1329`), then also applies its inherited multiplicative **box-label jitter** (`:1112-1113`). Therefore these are augmented training labels, not untouched official validation boxes. This distinction matters when interpreting a Mask/Box disagreement.

The actual loader import SHA matches the local DATA source (`PROBE/complete/imports.json:7,14`). The appearance manifest, split protocol, local loader identity, 29,778 fit / 6,887 holdout disjoint row IDs, and all stored scene IDs' fit-fold membership passed local checks. The runner verifies the source and dataset superpoint contract, enables both augmentations, and prepares train voxels (`PROBE/run_mask_geometry_probe.py:47-55,94-114,138-178,179-195,214-231`). No GT mask/box enters the model input dictionary; GT is attached only after forward for the native loss (`:183-204,235-239`).

The native criterion filters actual labels through `box_label_mask` (`NATIVE/models/losses.py:856-886`). The probe captures the real second matcher call, whose prefix is `last_`, validates its boxes and targets, then maps compressed GT indices through `valid[targets]` (`PROBE/run_mask_geometry_probe.py:232-276`; `NATIVE/models/losses.py:852-853,891-917`). This is not a root-only nearest-IoU replacement. The actual matcher has class/GIoU/text-mask costs (`NATIVE/main_utils.py:268-278`; `NATIVE/models/losses.py:327-390`). Saved assignments and slot mapping are internally exact for every row.

**Limit:** raw dataset point/mask tensors and full matcher cost inputs are not in this local packet. GT generation and the actual Hungarian computation are established by reviewed source plus hash-bound successful runtime receipts, not a fresh independent reconstruction from raw data. A stored point SHA alone does not prove historical tensor equality.

## B. Score normalization — PASS

No reported diagnostic is divided by the model's own maximum or mean. Box IoU is ordinary intersection/union (`PROBE/run_mask_geometry_probe.py:206-212`). Mask support is `sigmoid(alpha * text + (1-alpha) * query) > .5`; actual superpoint member counts and root-positive member counts produce point intersection/union for all 256 candidates (`:277-296`). This matches native fusion and point expansion (`NATIVE/src/grounding_evaluator.py:594-605,897-902`). It is not an unweighted average of superpoint IoUs. The selected candidate is additionally checked by explicit point expansion at runtime.

The signed native bbs rank uses softmax probabilities and the native main/modifier/pronoun/relation/other-entity reductions (`HELPERS/native_root_bbs.py:4-9`; `NATIVE/src/grounding_evaluator.py:222-286,535-553`). This softmax is part of the existing model scoring rule, not post-hoc accuracy normalization. Local selection from saved bbs agrees for all 64 rows; no top-score ties occur. Raw token logits/maps are absent, so bbs generation itself was source-checked, not independently recomputed.

Float64 Box IoU reconstructed from saved boxes agrees within `2.36955754e-6` absolute; saved integer Mask intersections/unions reproduce ratios within `2.97043959e-8`. Both give **zero threshold-classification differences** across all 16,384 candidates. Mask evidence here is an independent recount of saved counts, **not** an independent raw-Mask replay.

## C. Result files, numbers, and closure — WARN for metadata only

All 24 intake-listed files exist and match both recorded size and SHA256 (1,046,473 bytes total). Deployed copies of plan, source gate, runner, controller, spec and launch receipt match the local originals. All 17 helper identities and the five actual imported source identities match. `probe.exit` and `controller.exit` are both 0; status, controller log, terminal probe log and observer closure agree (`PROBE/complete/INTAKE.json:3-125`; `complete/status.json:2-14`; `complete/probe.log:7-15`; `wait.json:3-21`). Completion is an actual terminal record, not the earlier launch status.

The independent verifier reads all eight NPZ archives with `allow_pickle=False`, checks their schema and finite values, recomputes geometry and roles, and checks every JSONL row against its NPZ data and receipt. Results:

| Quantity | Independently confirmed |
|---|---:|
| Input rows / physical scenes / seeds | 64 / 60 / 1 (2027) |
| Retained candidates | 16,384 = 64 × 256 |
| Mask IoU > .5 and Box IoU ≤ .5 | 1,115 |
| Of those: unmatched / root matched / other matched | 1,112 / 3 / 0 |
| Rows with at least one such unmatched candidate | 53 / 64 |
| Selected answers in that group / unmatched | 10 / 10 |
| Mask and Box both > .5 | 2,447, across 58 rows |
| Unmatched candidates with Box IoU > .5 | 2,593 |
| All matched / unmatched candidates | 64 / 16,320 |
| Nonzero saved Box / boundary gradients | 64 / 64, all matched |
| Unmatched candidates with nonzero saved direct gradients | 0 / 0 |
| Exact GT face targets outside [-4,4] in the 1,112 pool | 222 candidates / 546 faces |
| Coarse Box IoU > .5 within that same 1,112 pool | 9 |

The original float32-derived medians are `0.1641764641 m` for maximum absolute GT-face error and `0.0055255890 m` for maximum coarse-to-final face change. Independent float64 reconstruction gives `0.1641763747 m` and `0.0055256337 m`; these differ only at floating-point rounding scale. Counts match `PROBE/complete/receipt.json:209-222` and `analysis/SUMMARY.json:5-29` exactly.

Nonblocking metadata findings:

1. `analysis/SUMMARY.json:20` says `cpu_checks: 548`. Source `analyze_probe.py:36,78,84` computes `8*4 + 64*8 + 4`. It is a manual bookkeeping value, not a count of distinct assertions or actual assertion executions. There are 16 assertion sites and, from the actual loop lengths, 452 evaluations of those statements, including 24 intake hash assertions. Do not advertise “548 independent checks.” Original SUMMARY is preserved; this audit is the provenance-preserving annotation.
2. **Resolved draft wording:** the earlier publisher at SHA256 `add6c096b575bf115fe9f1f28354065bf2057d019f0f45cb8a84eb56a54caeb3`, old `publish_probe_20261005.py:61`, said approximately 48 MiB system-disk space. This run's `resource_check.json:1` records 29,908,992 bytes = 28.5234375 MiB. The executor changed the text to the launch receipt's actual byte count; I read the final revised `publish_probe_20261005.py:65`, SHA256 `d754eb05c094c7eb01aff5558e62574b1daadb1558069ef323e86b0a29bb0b33`. It also annotates the original 548 bookkeeping label (`:61`). This does not affect diagnostic results. Publication execution itself is outside this audit. `AUDIT_CPU.json` preserves the earlier publisher hash as its time-of-check snapshot; the final audit identities record the separately reread revision.

## D. Called code and gradient attribution — PASS for reported probe measures

The executed entry point calls `main`; the bounded loader loop calls actual model forward, native criterion, the two leaf-gradient probes, native bbs, Box IoU and Mask count computation, then saves each batch and terminal receipt (`PROBE/run_mask_geometry_probe.py:228-365,369-370`). All eight batch records and their final cumulative counts appear in `complete/probe.log:7-15`. No reported measure depends on a merely defined but uncalled helper. The inherited native `calculate_diou_3d` function is unused/commented out (`NATIVE/models/losses.py:103-133,546-557`); **DIoU is not an executed or claimed metric** here. Uncalled quality-training/preflight helpers in the copied bundle likewise supply no claimed results.

Final Box gradients use detached output leaves and the actual native `loss_boxes`, including center L1, half-weight size L1, matched-count normalization and `(10*L1 + 2*GIoU)/7` (`PROBE/run_mask_geometry_probe.py:247-253`; `NATIVE/models/losses.py:518-544,947-955`). I independently derived the piecewise analytic NumPy derivative from saved matched boxes/GT for all 64 matches. Its maximum absolute component agrees with saved `box_gradient_max` within `1.85608283e-7`. This independently supports the coefficient, loss, normalization and direct-gradient interpretation; full signed gradient vectors were not saved.

Boundary gradients use a separate detached boundary-logit leaf and `distribution_loss/7`, selecting the same final native matches (`PROBE/run_mask_geometry_probe.py:254-276`; `HELPERS/pvground_boundary_box_refiner.py:40-65`). Their recorded support is zero on every unmatched candidate. **This field contains the distribution-loss derivative only**, not the sum of distribution and decoded-Box gradient routes. Raw boundary logits are unavailable for an independent numerical derivative. Neither leaf probe measures model-parameter gradients, total effects of other losses/layers, future changed assignments, or shared-parameter updates.

The only two autograd calls in the runner target these detached leaves. Full model forward/criterion run under `no_grad`; all parameters are frozen. The runner checks absent parameter gradients and exact before/after state_dict equality before writing its receipt (`:135-138,235-258,349-365`). Controller source checks zero steps/new weights and protected-parent hashes before/after (`PROBE/controller.py:17-20,35-42`), and the terminal receipts satisfy those checks. These are source/runtime witnesses, not a local rehash of the remote model state or weight files.

## E. Scope and interpretation — WARN

The first eight seeded shuffled fit batches exactly match `PRIOR:1-8`, the saved NPZ row IDs and all 64 JSONL records. There are 60 physical scenes, one seed, one reconstructed checkpoint, and one current augmentation per row; all rows belong to the 29,778-row fit partition. The loader/augmentation/voxel path is the executed fit path (`HELPERS/run_final_quality_fit.py:237-254,423-451`; `PROBE/run_mask_geometry_probe.py:179-195,214-231`). The earlier fit log is an **ordering reference**, not historical per-step assignment or augmented-tensor evidence. Current fresh-R model state is not the evolving earlier fit model. The result is neither full-fit coverage nor a formal validation evaluation.

All 64 observed valid target lists are `[0]`. Mapping through all valid native GT slots is implemented, but no nonempty multi-GT case occurred. Protection of another matched instance, Nr/Sr behavior and joint detection responsibilities are **not empirically tested** by this panel (`PROBE/complete/rows.jsonl:1-64`; `analysis/SUMMARY.json:29`).

The checkpoint basis is explicit: official PV SHA `6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3`, original-G delta SHA `0575dfae333dabdf288a470fc09d8967f9867d1853a5f61d9cfc3cbcf7964522`, distribution-geometry delta SHA `79e35068b8787a81c354c5fd3ed2bbc12fc62e06167b9cfbe781f86cbbd36c67`. Source hashes are verified before loading; factory applies strict official state, 1,072 G delta tensors and 10 geometry tensors, then installs fresh R with a zero output projection (`PROBE/spec.json:2-11`; `run_mask_geometry_probe.py:73-76,116-123`; `HELPERS/readback_model_factory.py:15-75`; `pvground_boundary_evidence_readback.py:27-29`). Same-forward query identity is asserted at runtime (`PROBE/run_mask_geometry_probe.py:237-238`). “5616/4506” are protected-parent labels, not newly audited formal accuracy. Inherited spec fields `updates:3723` and factory metadata `fresh_readback_optimizer_required:true` do not describe this probe's executed steps, which are zero.

The executed boundary parameterization uses 33 knots spanning [-4,4], `REG_SCALE=4`, and reference size floor `1e-6` (`HELPERS/pvground_boundary_box_refiner.py:9-37`). Independent CPU algebra checks both target encoding and inverse decoding in the same coarse-box frame. Its out-of-range counts reproduce 222/546; the actual matched targets all have zero out-of-range faces, matching every batch receipt. This tests **exact GT-face representability**. It is not an optimization over representable boxes and provides no upper bound proving IoU > .5 impossible. The nine coarse hits are counted over the entire 1,112-candidate pool, not just its 222 out-of-range candidates.

Native text masks are explicitly expanded to all 256 candidates before fusion (`PORT/models/pv_ground.py:532-556`). Thus a fused-Mask/root-IoU qualification can partly reflect a shared text-mask component; it is not an independent identity certificate for a query, a semantic correctness label, or a new deployment gate. The saved associations warrant a bounded hypothesis about direct training geometry responsibility, not a causal claim that adding targets, expanding range, or changing losses will improve formal accuracy.

## F. Evaluation type — real_gt, training diagnostic

Root Mask/Box qualification, Hungarian roles and encoded boundary targets use dataset-derived, currently augmented training labels. They are **real_gt** diagnostics. Model-produced coarse boxes define the coordinate frame for GT face offsets; they are not substituted GT. The same-forward zero-R and immutable-state checks are implementation invariants, not accuracy measurements. There is no synthetic-reference performance claim, independent human evaluation, or formal validation/effectiveness result.

## Claim disposition and necessary follow-up

- **Supported, with panel scope:** this fixed panel contains 1,112 fused-Mask-qualified, Box-poor, unmatched candidates; saved direct final Box and boundary-distribution gradients are zero for them. Source/runtime evidence and independent saved-array checks agree.
- **Supported as a range diagnostic:** 222 of those candidates have at least one exact GT-face offset outside current knots, 546 faces total. No impossibility or recoverable-accuracy inference follows.
- **Unsupported by this packet:** general prevalence; historical per-step supervision; nonempty multi-GT protection; candidate physical/semantic identity; absence of all shared-parameter/other-loss influence; any formal accuracy improvement or causal effectiveness.
- Preserve the original summary and identify `cpu_checks` as bookkeeping; do not quote 548 as assurance. The revised publisher annotates this label, corrects the disk number and explicitly states the common Text-Mask component (`:57,61,65`). Retain the fixed-panel, gradient-route, raw-Mask and checkpoint qualifiers in any publication. No further training/model replay is required to publish these qualified diagnostic results.

Verifier: `analysis/audit_cpu.py`; actual command: `uv run --offline --with numpy python -B -X utf8 .../analysis/audit_cpu.py`. Results and named check trace: `analysis/AUDIT_CPU.json`. Requested review trace directory: `.aris/traces/experiment-audit/2026-10-05_mask_geometry_terminal_run01/`; executor should attach the actual returned report/response to its native delegation record.
