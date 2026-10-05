# EXPERIMENT_AUDIT

**Verdict: WARN. Deterministic integrity checks: PASS. Blocking findings: 0.**

Fresh-context same-family / provisional review. Requested model and effort: `gpt-6-astra` / `max`; actual model, effort and backend are unattested (`backend_attested: false`). Generated: 2026-10-05 06:42:30 UTC.

The saved diagnostic is internally consistent. All 435 checked summary scalar leaves match a separately written NumPy recount exactly. WARN preserves the training-panel, GPU-DFL and review-attribution limits; no diagnostic source or raw-data correction is required by this audit.

Only `analysis/EXPERIMENT_AUDIT.json` and this report were written. No SSH, credential wrapper, GPU replay, browsing, weight load/deletion, source/raw change or SUMMARY rewrite was performed.

## Actual execution and evidence

- Current intake: 24 files, 2,632,409 bytes, every existing size/SHA entry exact; all eight NPZ archives opened and every field loaded.
- Closed execution: 8 full parent forwards, 8 final native semantic-head calls, 24 cached geometry-head replays, 64 distinct fit rows / 60 scene IDs and 16,384 candidates.
- `probe.exit` and `controller.exit` both contain 0; `status.json`, `controller.log` and `wait.json.status` agree. Eight batch events and the final probe-log receipt match the saved artifacts.
- Current runner/spec/row SHA bindings, 131 existing SOURCE_REVIEW entries, 17 helpers and all five recorded import digests match actual local bytes. The closed pair's 73-file intake (45,176,490 bytes) and actual terminal/spec/strict-restore identities also match.
- The diagnostic receipt reports 461.939561 seconds including model/data preparation; the controller reports 466.309289 seconds. These are not training-time measurements.

Evidence: `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\INTAKE.json:3`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\receipt.json:7`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\probe.log:7`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\status.json:2`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\controller.exit:1`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\probe.exit:1`.

## A. Ground Truth Provenance — PASS

Labels originate from ScanRefer object annotations and ScanNet object point memberships. The dataset source SHA is exactly the one recorded by the actual runner. Predictions do not generate the reference labels. The diagnostic writes `root_box` from `center_label`/`size_gts` slot 0 and evaluates Mask support against `gt_masks` slot 0.

Evidence: `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:584` (annotation IDs); `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1100` (GT membership); `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1387` (returned labels); `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:294` (actual root reference); `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\imports.json:9`.

### Precisely which tensors are augmented

| Tensor/path | Actual operation and distinction | Exact source |
|---|---|---|
| Supervised `gt_bboxes` → `center_label`/`size_gts` → diagnostic `root_box` | First bound the current augmented object points, then independently multiply each of the six center/size coordinates by a factor `0.95 + 0.1*random`, in `[0.95, 1.05)`. This is additional supervised box-label noise. | `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1105`; `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1112`; `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1387`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:296` |
| Points and `gt_masks` | Point rows first undergo permitted flips, z/x/y rotations, per-point positive noise in `[0,0.005)`, shift and scale. Object Mask labels keep the same annotation point-index membership. There is no further point/Mask transform accompanying the later box-only six-dimensional noise. | `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:823`; `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:907`; `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1100` |
| Scene GT `all_bboxes` | Built separately from current object member points and given its own separate six-dimensional factors. It is not the supervised root-box tensor. | `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1140`; `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1152`; `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1426` |
| Two-stage detected-object input `all_detected_boxes` → `det_boxes` | Loaded from GroupFree predictions, transformed by shared flip/rotation/shift/scale, then separately corrupted when `augment_det=True`. This detector-input corruption is not the supervised-box jitter above. `butd_gt=False`/`butd_cls=False` prevent replacing inputs by GT scene boxes. No claim is made that an internal network `proposal_` output is manually jittered. | `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1203`; `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1227`; `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1241`; `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1360`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:157`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:188` |

The hash-matched `Scan` helper reads object points from current `scan.pc` and computes their axis-aligned min/max box: `C:\Users\gb\.codex\tmp\pv_ground_source_20260905\src\visual_data_handlers.py:195`, `C:\Users\gb\.codex\tmp\pv_ground_source_20260905\src\visual_data_handlers.py:225`, `C:\Users\gb\.codex\tmp\pv_ground_source_20260905\src\visual_data_handlers.py:245`. The sample resets `scan.pc` from `orig_pc` before augmentation, and point augmentation precedes `_get_target_boxes`: `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1273`, `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1325`. Thus point/Mask membership shares the augmented coordinate rows, while the subsequent supervised bbox noise has no matching point/Mask transformation.

### Trigger and actual evaluation modes

Supervised root-box and scene-GT box jitter require **both** `self.split == 'train'` and `self.augment`: `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1112` and `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1152`. Detected-input corruption has its separate `self.augment_det` plus train-split condition at `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1241`. `model.eval()` by itself does not control these dataset flags.

| Actual path | Dataset split | augment / augment_det | Supervised bbox noise |
|---|---|---|---|
| Current fixed 64-row fit diagnostic | train | True / True | Enabled |
| Closed pair's 6,887-row initial and terminal module holdout | train | False / False during evaluate | Disabled |
| Closed pair's 9,508-row formal evaluation | val | False / False during evaluate | Disabled |

Current diagnostic enabling: `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:157` and `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:238`. The actual closed-pair runner creates the formal dataset with `split='val'` at `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\run_geometry_fit.py:189`. Its shared `evaluate` explicitly sets both flags False before calling a fresh holdout DataLoader and `prepare(...,'eval')` at `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\run_geometry_fit.py:273`. Formal, initial and terminal call sites are lines 417, 453 and 495 of that same file. Actual control/query-supported train logs each record completed 6,887-row stages at lines 20 and 93; their formal logs record completed 9,508-row stages at line 25. `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\controller.py:36` binds both arms to that runner.

Interpretation: the 64-row IoU, face error, DFL targets and fixed Box≤0.5 condition refer to an additionally noisy native training target; Mask support refers to object membership on the augmented cloud. They do not measure an exact unjittered physical envelope. This report does not assign any measured error/repair to the augmentation. The extra bbox-noise qualifier **does not apply** to the actual closed pair's formal or module-holdout labels, whose flags are disabled. This is a protocol check, not a new formal-metric audit.

## B. Score Normalization — PASS

Box IoU divides intersection volume by the actual 3D union. Mask support divides integer point intersection by integer point union. Group means use actual fixed counts, not a prediction maximum/mean. Native BBS softmax is a ranking calculation; it is not used to normalize reported quality. All thresholds are strict `>`; repairs and damages use the corresponding complementary `<=` side.

Evidence: `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\cohort_metrics.py:6`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\cohort_metrics.py:24`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\cohort_metrics.py:37`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:312`; `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\native_root_bbs.py:4`.

## C. Results and Recount — PASS

Every candidate's CPU float64 IoU, maximum face error and maximum movement from its parent coarse box were recomputed without importing `cohort_metrics.py`. Parent eligibility was reconstructed from saved integer Query/fused intersections/unions, the saved parent GPU Box≤0.5 flag and the parent matched-slot array. All matched-slot arrays were rebuilt from row records. Membership stays fixed across all three heads.

The range condition was rederived from saved parent coarse boxes and GT, with endpoints −4 and +4. All 98,304 per-face inside/outside classifications agree. There are 1,090 qualified candidates on 52 rows, including 201 outside candidates and 482 outside faces. All 64 row IDs belong to the fit partition and match the prior row/scan/point-hash/GT records.

| Group | Count | Parent mean IoU | Control mean IoU | Query-supported mean IoU | Hits >0.5: parent / control / query-supported |
|---|---:|---:|---:|---:|---:|
| all_candidates | 16384 | 0.137442392 | 0.137214979 | 0.137046305 | 2654 / 2647 / 2641 |
| parent_qualified | 1090 | 0.273640310 | 0.271830799 | 0.272431954 | 0 / 3 / 5 |
| qualified_inside | 889 | 0.326996409 | 0.324887115 | 0.325532038 | 0 / 3 / 5 |
| qualified_outside | 201 | 0.037652387 | 0.037168782 | 0.037576360 | 0 / 0 / 0 |
| parent_box_good | 2654 | 0.709049722 | 0.708466705 | 0.707390808 | 2654 / 2644 / 2635 |
| selected_query | 64 | 0.549682523 | 0.549221080 | 0.549485038 | 42 / 42 / 42 |
| selected_qualified | 10 | 0.312986195 | 0.309457052 | 0.310723166 | 0 / 0 / 0 |

All 21 arm/group summaries and 21 pair/group comparisons match, with 435 checked scalar leaves and maximum summary difference 0. The machine-readable report retains the full recounted summaries.

| Comparison | Group | Mean ΔIoU | >0.25 repairs / damages / net | >0.5 repairs / damages / net |
|---|---|---:|---:|---:|
| parent_to_control | all_candidates | -0.000227413 | 1 / 6 / -5 | 3 / 10 / -7 |
| parent_to_control | parent_qualified | -0.001809511 | 0 / 5 / -5 | 3 / 0 / 3 |
| parent_to_control | qualified_inside | -0.002109294 | 0 / 5 / -5 | 3 / 0 / 3 |
| parent_to_control | qualified_outside | -0.000483604 | 0 / 0 / 0 | 0 / 0 / 0 |
| parent_to_control | parent_box_good | -0.000583017 | 0 / 0 / 0 | 0 / 10 / -10 |
| parent_to_control | selected_query | -0.000461444 | 0 / 0 / 0 | 0 / 0 / 0 |
| parent_to_control | selected_qualified | -0.003529144 | 0 / 0 / 0 | 0 / 0 / 0 |
| parent_to_query_supported | all_candidates | -0.000396087 | 0 / 7 / -7 | 6 / 19 / -13 |
| parent_to_query_supported | parent_qualified | -0.001208355 | 0 / 5 / -5 | 5 / 0 / 5 |
| parent_to_query_supported | qualified_inside | -0.001464371 | 0 / 5 / -5 | 5 / 0 / 5 |
| parent_to_query_supported | qualified_outside | -0.000076027 | 0 / 0 / 0 | 0 / 0 / 0 |
| parent_to_query_supported | parent_box_good | -0.001658913 | 0 / 0 / 0 | 0 / 19 / -19 |
| parent_to_query_supported | selected_query | -0.000197485 | 0 / 0 / 0 | 0 / 0 / 0 |
| parent_to_query_supported | selected_qualified | -0.002263030 | 0 / 0 / 0 | 0 / 0 / 0 |
| control_to_query_supported | all_candidates | -0.000168674 | 3 / 5 / -2 | 11 / 17 / -6 |
| control_to_query_supported | parent_qualified | 0.000601155 | 2 / 2 / 0 | 3 / 1 / 2 |
| control_to_query_supported | qualified_inside | 0.000644923 | 2 / 2 / 0 | 3 / 1 / 2 |
| control_to_query_supported | qualified_outside | 0.000407578 | 0 / 0 / 0 | 0 / 0 / 0 |
| control_to_query_supported | parent_box_good | -0.001075896 | 0 / 0 / 0 | 7 / 16 / -9 |
| control_to_query_supported | selected_query | 0.000263958 | 0 / 0 / 0 | 0 / 0 / 0 |
| control_to_query_supported | selected_qualified | 0.001266114 | 0 / 0 / 0 | 0 / 0 / 0 |

CPU-versus-saved-GPU maximum IoU differences are parent `2.180517747252253e-6`, control `2.342442458003191e-6`, query-supported `1.899503906055422e-6`. Every >0.25/>0.5 threshold comparison agrees, and CPU parent qualification changes are 0. Maximum face-error discrepancy is `8.344650268554688e-7` m; maximum face-movement discrepancy is `4.76837158203125e-7` m.

Float64 target values themselves are not claimed bitwise equal to float32 GPU targets: the maximum absolute target difference is 2.4101468697190285 at magnitude about 17.83 million on a size-floor component. Maximum inside-face target difference is `9.612381891344057e-6`; there are no range-classification disagreements. This does not change any reported group.

Evidence: `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\analysis\SUMMARY.json:2` (counts), line 10 (arms), line 246 (pairs); `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\cohort_metrics.py:54`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\analyze_probe.py:26`. Binary evidence is all actual `complete/batch_00.npz` through `batch_07.npz`; binary archives do not have text line numbers.

## D. Called Metrics and Closure — PASS

Each full parent forward captures one geometry-input tuple. It is reused for three head replays, loading all 10 head tensors (456,102 parameters) strictly. Runtime assertions compare parent replay boxes with native parent boxes and check BBS plus every Mask/alpha tensor after each replay. The actual parent match is final-layer `matches[1]`: native criterion prefix order is `proposal_`, `last_`, then earlier heads.

All reported metrics are on an executed save path. The analyzer invokes `summarize`, which invokes the IoU/face/pair functions. GPU DFL uses the inspected clipped nonuniform interpolation formula, but NPZ stores only the per-candidate scalar: group DFL means reproduce those scalars, not a fresh CPU logits-to-loss calculation.

Parent restoration compares every full-model state tensor to the original CPU clone and confirms absent gradients. Controller verifies no local probe weight artifact and unchanged protected parent file hashes. There is no optimizer/backward/update/save call in the diagnostic. `load.json:2` is inherited factory metadata about an optimizer being required for future training; no diagnostic optimizer is constructed.

Evidence: `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:211` (head binding); `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:246` (forward/replay); `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:331` (metrics/save); `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:362` (restore); `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\controller.py:31` (closure); `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\launch_probe_authorized.py:52` (actual controller exit filename); `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\readback_preflight_checks.py:19` (native semantic head hook); `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_boundary_box_refiner.py:33` (DFL definition); `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\cohort_metrics.py:54`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\analyze_probe.py:41`. Native matcher order: `C:\Users\gb\.codex\tmp\pvground_runtime_bundle_20260908_v1\PV-Ground\models\losses.py:849`.

## E. Scope — WARN

Scope is one seed, 64 seen augmented fit inputs, 60 scenes, root-only GT and correlated candidates. This is not an unseen-scene accuracy result, multi-GT protection demonstration or physical-instance identity test. The plan/receipt/summary already state these limits; no scope inflation was found.

Current-versus-prior full-forward drift is real:

- Parent boxes: 76,261 differing components; max absolute difference 0.0013647079467773438 m.
- Parent GPU IoU: 4,081 differing entries; max 0.00021922588348388672.
- BBS: 16,038 differing entries; max 0.0005064010620117188.
- Fused Mask intersections/unions: 6 / 17 differing entries, with maximum integer differences 95 / 155. Query intersection/union counts are exact.
- Matched slots, selected candidates, root GT, row/point identity and all fixed cohort members are unchanged.

Same-cached-input head equality cannot be generalized to separate complete CUDA forward equality. Evidence is the actual current/prior eight-batch arrays and the explicit limit in `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\complete\receipt.json:13`.

Raw point clouds/full GT masks were not reloaded locally; recorded point hashes were matched and eligibility was recomputed from saved integer counts. Runtime source/assertions/closed receipts support same-batch equality and restore checks; this review did not replay a GPU model.

## F. Evaluation Type — PASS: real_gt

Classification is **real dataset GT with native training augmentation and box-label noise, used in a fixed-parent cohort diagnostic**. The offline Mask-support qualification is a GT-support proxy, not a physical identity certificate. It is not a model-generated reference, simulation-only result, self-supervised evaluation or human evaluation.

Evidence: `C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\source\joint_det_dataset.py:1086`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py:294`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\cohort_metrics.py:78`; `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\analysis\SUMMARY.json:632`.

## Claim impact

- Supported: 1,090 fixed qualified candidates; query-supported repairs 5 at >0.5 versus parent, all inside; qualified mean IoU nonetheless falls by 0.001208355444.
- Supported: over all candidates, query-supported versus parent has 6 repairs and 19 damages at >0.5, net −13. Versus control on the qualified group it has 3 repairs and 1 damage, net +2.
- Supported: native selected candidates keep 53 hits at >0.25 and 42 at >0.5 for every head. None of the 10 selected-qualified candidates is repaired at >0.5.
- GPU-scalar-only: qualified DFL means are 4.0189714431762695 / 4.166342735290527 / 3.37739896774292 for parent/control/query-supported. Lower DFL is not sufficient evidence of better localization.
- Unsupported by this diagnostic: general formal accuracy gains, solved physical identity, unseen-scene generalization or nonempty multiple-GT protection. Those claims are not made by the current summary.

Formal metric acceptance and retention/deletion decisions are not assessed. The closed pair's evidence is used for head binding and the requested augmentation-mode trace only.

## Files and provenance

The JSON report lists **184 actual current file paths with SHA256 and review scope**, distinguishing direct semantic/array checks from historical manifest-hash linkage. It includes the full A–F evidence and recounted metrics. Selected sealed artifact hashes:

| File | SHA256 |
|---|---|
| `complete\spec.json` | `748809b0e1c740b3253d992de719b8e8e1bac55518e7e1e0a5a6651932f1de71` |
| `complete\run_cohort_probe.py` | `ff006960b9a1c0754ba6dc3ef163fc997c3594e2232e98c9bc25f0639766dd18` |
| `complete\rows.jsonl` | `0e6236609932eab258ab52b9be725c5161fb1f55378fc565504dcb1386179f3d` |
| `complete\receipt.json` | `c33f90ee770fe6a27f70e1372eee62dba169d979f43957d650767866b2906460` |
| `analysis\SUMMARY.json` | `c83227a15d97b7fafc4307200f5e8e6f9bafe271ecd4db10c27bf175e9c6986c` |
| `complete\batch_00.npz` | `b0fb8880b214544bfbab2c9c525dfce8e11df63b264baa5ba9ec8ee2aabc1f86` |
| `complete\batch_01.npz` | `b4b7698f591a6ac995a030b4c2170ea73e5f4a189e9ded2d70e04c608a9fbd9f` |
| `complete\batch_02.npz` | `99f8a09ab437e898fc68854017d8184fad781ce52420e1ab3a3225abdff60154` |
| `complete\batch_03.npz` | `1fd009cdc0902ef768ebace44a24110fb7cf54a683fb82e91ffa9099b2243caf` |
| `complete\batch_04.npz` | `bb42cbd58c8034aa4472a58ea96cf3e3af60fc8408ab21a0e34a7559fd26dfbd` |
| `complete\batch_05.npz` | `c9f64b5913d752f5b7d3567e64634487566945392ef33de337fe0255bcc0452e` |
| `complete\batch_06.npz` | `1aaef5edc68d4eb6c75d984bc1f65e7e1f3e5dbd25fc16fdf754678007d51cf8` |
| `complete\batch_07.npz` | `4cafe6f27eed2a32a277db377acacfd1108e3a8b6c7d60642e506be4a7ace691` |

The sealed preparation plan/source review and SUMMARY's audit-pending marker were preserved. This report supplies the terminal audit without rewriting those artifacts.
