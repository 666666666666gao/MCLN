# Experiment Audit: Actual Closed Span Result

Date: 2026-10-10 (Asia/Shanghai). Execution scope: `ACTUAL_CLOSED_SPAN_RESULT`.

**Verdict: WARN.** The archived numerical results are supported. Each learned arm independently exceeds both requested ScanRefer thresholds. This is a frozen-parent, two-head mechanism-control experiment; it does not establish ordinary joint training, three effective novel contributions, or the full research goal. No model was promoted by this audit.

Fresh native Codex reviewer: `/root/pvg_span_outcome_20261010`. Requested route: `gpt-6-astra`, reasoning `max`; actual model and reasoning route: **UNATTESTED**. Review independence is `same-family`; acceptance is `provisional`. Deterministic recount passed. No confirmed blocker was found for accepting the archived counts with the scope below. The broader claims listed below remain unsupported.

## Independently verified results

All values below are root `last_`/`bbs`, top-1, strict `IoU > threshold`, with the same native score and selected query for Box and Mask. Denominator: 9,508 validation expressions, 141 scene IDs, seed 2027. The two learned arms are separate models; 5,677/4,921 is **not** a measured single-model result.

| Path | Hits @0.25 | Acc @0.25 | Hits @0.5 | Acc @0.5 | Both gates |
|---|---:|---:|---:|---:|---|
| Native regressor control | 5,615 | 59.0555% | 4,495 | 47.2760% | No |
| Frozen Mask-reference parent | 5,606 | 58.9609% | 4,881 | 51.3357% | No |
| Fixed-half control | 5,675 | 59.6866% | 4,857 | 51.0833% | Yes |
| Whole-support span | 5,676 | 59.6971% | 4,921 | 51.7564% | Yes |
| Extremal-support span | 5,677 | 59.7076% | 4,920 | 51.7459% | Yes |

The requested strict percentage gates require at least 5,658/4,850 hits. Fixed-half also passes, so merely crossing these thresholds does not demonstrate that learned extremal evidence is necessary.

| Paired comparison | Repairs / damages @0.25 | Net @0.25 | Repairs / damages @0.5 | Net @0.5 |
|---|---:|---:|---:|---:|
| Whole versus Mask | 86 / 16 | +70 | 103 / 63 | +40 |
| Extremal versus Mask | 87 / 16 | +71 | 103 / 64 | +39 |
| Whole versus fixed-half | 33 / 32 | +1 | 157 / 93 | +64 |
| Extremal versus fixed-half | 33 / 31 | +2 | 156 / 93 | +63 |
| Extremal versus whole | 1 / 0 | +1 | 3 / 4 | -1 |

Evidence: `complete_fit/formal/rows.jsonl`, `complete_fit/formal/receipt.json`, every `complete_fit/formal/batch_*.npz`, and `RAW_REVIEW/independent_recount.json`. The last comparison does not establish an effective novel extremal-support contribution. It also supplies no evidence that three connected components are three separately effective contributions.

## A. Ground-truth provenance: PASS, with a local-data limitation

GT comes from ScanRefer annotations and stored ScanNet scene objects. Dataset loading reads `ScanRefer_filtered_{train,val}.json` and the corresponding scene list; target IDs select scene-object points and object bounding boxes. Training jitter is restricted to the training split. Formal evaluation constructs `split='val'`, disables augmentation, and takes root GT from `center_label` and `size_gts`. Neither the span head nor the score takes target geometry as an inference input.

Evidence: dataset source lines 585–597, 633–648, 1086–1119 and 1387–1392; `run_span_pair.py:152–208,237–250`; `paired_span_loop.py:278–299`; `matched_mask_objective.py:6–30`; `extremal_span_mixer.py:29–120`. “Mask reference” means a predicted geometric input, not synthetic ground truth.

The reviewer did not have local copies of the raw ScanRefer annotation JSONs, `train_v3scans.pkl`, `val_v3scans.pkl`, or the remote superpoint tensors, and did not rederive GT from those raw files. GT provenance is supported by the hash-matched dataset source, captured GT arrays, and the executed source/contract checks; it is not a new raw-dataset authentication exercise. Missing raw files remain an evidence limit.

## B. Score normalization: PASS

Box IoU is intersection volume divided by geometric union volume; accuracy is an integer hit count divided by 9,508. No accuracy denominator uses the model's maximum, minimum, or mean score. Softmax in the native BBS score is ordinary token-probability scoring, not normalization of reported performance. Point-count normalization inside the mixer is a feature-pooling operation.

Evidence: `run_span_pair.py:252–258`; `paired_span_loop.py:340–355`; `recount_closed_span_cpu.py:33–40,73–95`; helper `native_root_bbs.py:4–9`; evaluator lines 217–303 and 535–553.

The independent checker initially imposed an unnecessary exact `IoU <= 1` floating-point assertion. Actual float32 calculations exceeded one slightly for 6,842 Mask candidates (maximum 1.0000050068) and four extremal candidates (maximum 1.0000002384). The failed checker and native error are preserved. The completed checker records the raw values without clipping or rescaling. All 12,170,240 candidate threshold decisions agree between float32 and float64 at both thresholds; no reported hit count changes.

## C. Result existence and arithmetic: PASS

The intake and remote manifest agree on all **1,246 files / 281,198,905 bytes**, including two terminal checkpoints. Every listed file was independently size- and SHA-256-verified. All 1,189 formal NPZ batches cover validation indices 0–9,507 once and contain 256 scores and 256 six-coordinate boxes for each of five paths. All selected queries are unique score maxima; selected boxes equal their JSONL records exactly. Recomputed selected IoUs agree with saved values within 6.56e-7 and yield identical decisions. All full-256 oracle counts agree too; oracle counts are diagnostic coverage, not deployed accuracy.

Training has exactly 3,723 sequential records per arm, 3,722 batches of eight plus a final batch of two, and 29,778 unique fit rows. Their multiset matches the hash-bound fit split and their order matches both terminal payloads. The stored optimizers each contain 14 finite Adam moment states at step 3,723 with the declared learning rate and weight decay. This was inspected with a restricted tensor-data parser using NumPy, without importing torch or constructing a model.

Evidence: `complete_fit/INTAKE.json`, `MANIFEST_STDOUT.json`, `train.jsonl`, both `terminal.pth` and `formal_restore.json`; `paired_span_loop.py:126–209,389–407`; `RAW_REVIEW/independent_recount.py` and its successful native receipts.

Training and validation integers are separate annotation-enumeration namespaces: 7,674 integers overlap, which does not identify shared examples. The first author recount's disjoint-integer assertion was invalid; its removal and preserved failure are consistent with the source. The source actually reads ScanRefer JSON, so the historical phrase “CSV row IDs” is imprecise. Training fit/holdout disjointness is checked against the separate pinned split. Raw annotation-level train/validation disjointness was not newly rederived locally.

## D. Actually called code and protocol: PASS

The controller executes `train` and then `formal` using `run_span_pair.py`; both terminate successfully. Entry calls `PairedSpanRun.run`, which calls the training steps or terminal restore plus evaluation. Evaluation calls the hash-matched native `GroundingEvaluator.evaluate`, and asserts native BBS hit counts equal saved row counts. Native BBF/mask diagnostic functions also execute inside that evaluator; those alternate diagnostic rankings do not select the reported BBS outputs. Unused helper definitions are not presented as new measurements.

One frozen parent forward feeds both independent span heads and all three controls. `apply_span_mixer` changes only the final center/size while preserving score and Mask objects. The same selected native query supplies each Box and its Mask. All 256 candidates remain available; no V99 or additional learned deployed ranking is introduced. Each arm has 29,793 trainable parameters in 14 tensors, identical initial head state, separate optimizers, and the same 29,778 rows/3,723 updates. The parent receives no gradients. Full formal evaluation uses the shared `apply_span_mixer` function, rather than separately calling each complete wrapper and rerunning the backbone.

Evidence: `span_controller.py:44–67`; `run_span_pair.py:128–139,260–267`; `paired_span_loop.py:38–83,126–180,278–366`; `span_refinement_model.py:8–44`; evaluator lines 194–206; per-row and per-step forward-count witnesses.

## E. Real scope and claim ceiling: WARN

This closed run is a one-pass frozen-parent span adaptation on 29,778 fit expressions, followed by 9,508 ScanRefer validation expressions across 141 scenes. It is not ordinary end-to-end joint optimization of the original trainable PV-Ground core. No 6,887-row local-holdout metric, new Nr3D/Sr3D result, additional seed, or clean new test-set result is established here. Repeated use of this validation set for method development/selection must remain disclosed. The one-seed scope is intentional; this audit does not request a multi-seed run.

The retained research-goal file records normal joint training as not started, three effective contributions as unconfirmed, and full-goal completion as false. Those limitations agree with the actual span code. A new numerical record is supported; a completed scientific objective or automatic promotion is not.

Evidence: `pair_spec.json:15–18,30,46,56–58,98–124`; `run_span_pair.py:147–208`; `current_research_goals.json` (exact inspected SHA in audit JSON); final paired counts above.

## F. Evaluation classification: real_gt

This is real-GT grounding evaluation on the declared ScanRefer validation split. It is neither a model-generated proxy nor a new benchmark-independent test. Mask hits 5,814/5,127 and mIoU 47.0564283% were recounted from saved row metrics and checked against the native receipt. Formal raw prediction/GT masks are not stored in the NPZ box archive, so this audit does not claim an independent raw-mask recomputation.

## CPU terminal recovery: supported assembly and Adam restore; stronger identity unverified

The successful CPU recovery has real stdout, stderr and exit-code-zero receipts. Its source loads the four hash-checked protected checkpoint dependencies, builds each 1,314-state parent and its 14-state span head, strictly loads the actual terminal mixer, restores every Adam key/moment/step/group, and checks step 3,723. The full assembled state has 1,328 tensors. CUDA remains uninitialized, the helper makes zero forward calls, and CUDA RNG is explicitly not replayed. The earlier `KeyError: 'data_root'` attempt and its empty stdout/exit 1 are retained; the corrected script reads `manifest['data_root']`.

**The helper does not establish bitwise identity of all 1,328 tensors against the original run.** It clones a newly constructed parent's state, then compares the assembled wrapper against that same clone plus the saved mixer. `BoundaryEvidenceReadback` contains random hidden initialization and only zeros its output layer; the recovery helper does not reset the original constructor RNG or compare this hidden state to an independent original-parent archive. The selected geometry refiner's ten tensors, in contrast, are explicitly loaded from the selected checkpoint. Both geometry/R output layers are checked zero by the wrapper. The inactive R hidden-state limitation therefore does not itself contradict the archived effective-path metrics, but the name `full_cpu_model_state_exact` must retain this restricted assembly meaning.

The recovery stdout also lacks a fresh actual-import/source-hash transcript for that reconstruction. Current local copies match the run's pinned source, but this is not a separate attestation of imports at recovery time. This audit supports complete CPU construction plus actual head/Adam loading, not a fresh cold GPU inference or an independent original full-state bitwise replay.

Evidence: `restore_closed_span_cpu_authorized.py:21–69`; `selected_mask_reference_factory.py:49–62`; helper `pvground_boundary_evidence_readback.py:18–29`; `span_refinement_model.py:27–32,55–61`; helper `whole_model_preflight_checks.py:30–44`; `CPU_TERMINAL_RECOVERY*.json/txt`.

## Source identity, evidence limits and disposition

The reviewer verified 11 current/archived runner files against `pair_spec`, all 17 pinned helper files, all 98 model-source-port files, and all six recorded actual-import digests. The discovered 625-file dataset-source manifest has 376 locally available files, all hash-exact; all Python files are present. The remaining 249 non-Python files are listed as missing local evidence, not assumed present. The run entry checks the declared remote source, split and superpoint hashes before execution. The complete 1,246-file run intake is fully present; it is not a claim that all original raw data and environment binaries are copied into that intake.

For accepting the **archived numerical result within this scope**, `blocking_findings` is empty. Claims of three effective novel contributions, completed normal joint training, original full-state bitwise replay, fresh terminal cold GPU inference, or completed Nr3D/Sr3D remain unsupported. Before making any such claim, obtain its direct evidence; do not relabel this existing run. Keep both arm metrics, fixed-half control and whole-support control visible. No publication, retention change, checkpoint mutation, new model execution or promotion occurred during this audit.

The report, JSON and seal bind the actual `CPU_RECOUNT`, `INTAKE`, `pair_spec`, terminal recovery and reviewed files by bare SHA-256. Native tool results, the independent CPU checker, its failed first attempt and successful result are in `RAW_REVIEW/`. The original author recount's pending flags are historical: the later CPU assembly receipt exists, with the limitations above.

Source aliases used above: dataset = `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/runtime_binding/dataset_source/src/joint_det_dataset.py`; evaluator = `C:/Users/gb/.codex/tmp/pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py`; helper directory = `C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle/`; CPU helper = `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/restore_closed_span_cpu_authorized.py`. Other unqualified evidence paths are relative to this run's `runner_v1` directory (postrun artifacts relative to `postrun_results`).
