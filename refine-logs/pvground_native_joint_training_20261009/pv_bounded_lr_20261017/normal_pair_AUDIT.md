# Completed matched-endpoint normal-pair integrity audit

**Verdict: WARN.** The sealed local evidence supports a completed, equal-budget normal-training comparison. It does not support a new stronger model, an effective third mechanism, or completion of the current user goal. No confirmed integrity violation invalidating that qualified comparison was found. Evidence limitations below prevent an unqualified reproduction claim.

Date: 2026-10-17T00:38:09.307615+08:00. Execution scope: `COMPLETE_MATCHED_ENDPOINT_NORMAL_PAIR`.
Reviewer: fresh native task `/root/matched_endpoint_normal_pair_audit_20261017`. Requested route: `gpt-6-astra`, effort `max`, `fork_turns=none`. **Actual model and effort: UNATTESTED.** Review independence: **same-family**; acceptance: **provisional**. The native task return identifies a task, not an attested backend. Earlier source-only and two-step M0 audits were read as historical artifacts and were not used as substitutes for this full-normal result review.

Only local snapshots were read. No SSH/network operation, GPU query, neural forward, checkpoint loading, runtime installation, job restart, third-party communication, or indexed source/result mutation was performed. Audit artifacts are confined to this review directory.

## Evidence and seal scope

All **100 physical R2 inputs match their raw SHA256 seals**. `AUDITED_INPUT_HASHES.json` and the `audited_input_hashes` fields use exactly the R2 path keys and raw digest values, without alias fallback. Additional source dependencies and archive receipts have separate hash maps. Critical active trainer, dataset, evaluator, loss, geometry and controller paths were read directly; 85 local Python files were parsed without importing project code.

R1 is retained unchanged, SHA256 `bff97f2e8094fe681fd4655c7b7f4c656174756327bb86c6fdb5628ccdac56c6`. Its 101 entries described 100 physical files: the tracker appeared as both an unprefixed path with stale expected hash `8d051fb500f75a7d587619fedcc8c664d3b3261c0c8edb067ad10db63c07524d` and an extended-path alias with current hash `9e7b0e402593801f117dd24521fac36141a649a0e548c4fd9f42b2edd7d88bc7`. R2 SHA256 `930b128e8eeef0ade7fd09f57555b6c164c25505d3c448a89244f0c980406672` removes only that duplicate and binds the current tracker once. The correction resolves the index seal defect; it does not erase its history or change source/results.

The 240-entry stage manifest binds 120 files per arm. **206 entries have matching local bytes; 34 entries correspond to 17 unavailable unique dependencies.** The missing items include `models/encoder_decoder_layers.py`, `models/modules.py`, nine ScanNet metadata/split files, four parser word lists, `data/fig.png`, and the pointnet2 binary. Two other local copies with the same Python filenames do not match the staged hashes and were not substituted. Exact paths and expected hashes are in `SOURCE_HASH_BINDINGS.json`. Historical manifest agreement is not independent inspection of missing bytes.

For references below, `K` is `C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\coarse_endpoint_training_20261014`, `D` is `C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012`, and `S` is `C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\deployment_geometry_training_20261012\span_lr_actual_pair_review_20261014\snapshots`. `F` denotes `K/OBS_final_duplicate_e2_validation_expected_terminal_files/logs/scanrefer/final_duplicate/1792077835`; `C` denotes `K/OBS_coarse_endpoint_e2_validation_final_files/logs/scanrefer/coarse_endpoint/1792124122`. Machine-readable evidence contains full paths and line numbers.

## A. Ground-truth provenance — WARN, real_gt

The inspected GT path is dataset-derived: ScanRefer `scene_id`/`object_id` and captions select ScanNet objects; object segment point coordinates form boxes and masks. These populate `center_label`, `size_gts`, and `gt_masks`. Targets are not manufactured from the model's predicted boxes. The dataset reads group-free proposal boxes separately, and the normal argv enables `--butd` without `--butd_gt` or `--butd_cls`. Model input construction excludes GT box/mask coordinates; GT is attached after the forward for losses and evaluation. Evidence: `S/103_joint_det_dataset.py:585–649,1086–1119,1189–1256,1260–1450`; `S/105_visual_data_handlers.py:84–163,225–259`; `S/107_train_dist_mod.py:133–176`; `S/102_grounding_evaluator.py:535–553`.

The actual annotation JSON/TXT, cached scan pickle, segmentation/scan payloads and exact split membership were not available for independent byte or identity verification. Scene counts and unique expression IDs were not recounted. Therefore this is a source-level real-GT finding with logged full9508 execution, not certification of all original dataset bytes or official-paper protocol equivalence.

## B. Score normalization — PASS on the inspected active path

Native accuracy counts selected root boxes whose standard IoU is strictly greater than 0.25 or 0.50, divided by the dataset target count, asserted to be 9508. The active path does not divide accuracy by any maximum, minimum or mean of model outputs. Semantic softmax and caption token-map ranking do not normalize the accuracy metric. Both reported percentages are directly reproducible from integer hits/9508. Evidence: `S/102_grounding_evaluator.py:251–303`; `S/107_train_dist_mod.py:240–261`; `S/128_native_root_bbs.py:4–9`; `K/source/models/losses.py:35–75`.

## C. Result existence and execution — WARN for unavailable raw evidence

Both original normal paths are present in the sealed controller/child records:

- Control: `/root/autodl-tmp/pvground_matched_endpoint_20261014/normal/final_duplicate/logs/scanrefer/final_duplicate/1792077835`; controller/child 741474/741477; actual completion 2026-10-16 12:07:04+08.
- Coarse: `/root/autodl-tmp/pvground_matched_endpoint_20261014/normal/coarse_endpoint/logs/scanrefer/coarse_endpoint/1792124122`; controller/child 792357/792360; actual completion 2026-10-17 00:06:38+08.

Both terminal captures report controller and child absent with exit 0; coarse started after control completion. The accepted intakes, captured file hashes, native statuses, metrics, stop records, diagnostic totals, E2 log tails and summary CSV agree. The current tracker has dated COMPLETE entries (`K/EXPERIMENT_TRACKER.md:38–50`); early Not started entries and protocol fields such as `normal_training_started=false` are frozen history, not the execution evidence. Only the last 20000 bytes of each full train log are locally present. All six native raw-row payloads are absent locally despite their recorded hashes/sizes.

| Arm | Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 % | Acc@0.50 % |
|---|---:|---:|---:|---:|---:|
| final_duplicate | E0 | 5677 | 4920 | 59.707615 | 51.745898 |
| final_duplicate | E1 | 5651 | 4735 | 59.434161 | 49.800168 |
| final_duplicate | E2 | 5631 | 4697 | 59.223812 | 49.400505 |
| coarse_endpoint | E0 | 5677 | 4920 | 59.707615 | 51.745898 |
| coarse_endpoint | E1 | 5678 | 4742 | 59.718132 | 49.873790 |
| coarse_endpoint | E2 | 5657 | 4722 | 59.497265 | 49.663441 |

Evidence: `F/native_metrics.jsonl:1–3`, `C/native_metrics.jsonl:1–3`, both `native_stop.json:1–39`, `K/normal_pair_completed_20261017/metrics.csv:1–7`, both native normal-status files and the original terminal intake files.

## D. Live metric path — PASS

The trainer calls `GroundingEvaluator.evaluate` on every validation batch for `last_`, synchronizes it, reads `last/bbs` top-1 counts, checks both denominator counts equal 9508, then writes native metrics. This is an executed metric path corroborated by actual records, not a metric definition without a caller. Existing bbf/top-5/top-10/semantic/mask diagnostics are not the primary reported accuracy. The separate detection/AP branch is inactive for ScanRefer; commented DIoU loss calls establish no DIoU result. Evidence: `S/107_train_dist_mod.py:180–262`; `S/102_grounding_evaluator.py:194–206,217–303`; `S/050_main_utils.py:183–187`; `K/source/models/losses.py:889–925`.

## E. Training scope, source lineage and comparison — WARN

Each arm uses seed2027, batch8, accumulation1, the original 36665-expression training set with native drop-last, 4583 batches/36664 row visits per epoch, and **two actual complete training epochs: 9166 loops and 73328 row visits**. Source has one backward/optimizer/scheduler step per batch, with clipping0.1 and no AMP skip/accumulation path. These are source-implied 9166 optimizer step calls; serialized optimizer counters were not inspected. Validation includes E0/E1/E2, each logged as all9508 with 256 candidates. The stage's max8 is a cap, not eight executed epochs. Evidence: `S/050_main_utils.py:255–278,380–412,473–541`; both `selected_geometry_responsibility.jsonl:1–2`; both native metric histories.

Both stop after two adjacent strict Acc@0.5 declines including E0: control 4920→4735→4697; coarse 4920→4742→4722. Equality/increase resets the count in source, and pure arithmetic boundary fixtures verify this rule. The actual totals happen to match under this common dynamic stop rule. Best ordering is both-goal then strict hits then wide hits; both best-overall records remain E0 and both best-trained records are E1. Wall times are approximately 12.72h control and 11.86h coarse, not equal GPU-time budgets. Evidence: `S/050_main_utils.py:169–180,380–412` and both native stop/status files.

Coarse minus control is **+27/+7 at E1** and **+26/+25 at E2**. Best-trained coarse E1 is **+1/−178 versus E0**; control best-trained E1 is −26/−185. Terminal coarse is −20/−198 versus E0; terminal control −46/−223. Thus the small paired-control advantage does not preserve the strong starting strict accuracy, justify promotion, or prove an independently effective mechanism. Wide and strict metrics must come from the same complete model; coarse E1 wide cannot be combined with E0 strict.

The original parent stage has118 files. Both new stages add the endpoint objective/config and modify only `models/losses.py`. Original losses SHA256 is `b960fd486e4b325f346f3925d98bcda84be59020310d123832a789a525ceefee`; current losses is `28f1bbc92563e5dd5a11c980f2cb8b3068862db12e5939e8aa5653b00cb6b0c4`. The exact diff is seven inserted lines, `losses.py:960–966`. Between the two stage manifests only `matched_endpoint_config.json` differs. Admission records bind the common stage receipt; the 17 missing unique dependencies remain a limit on independently inspecting all source bytes.

The new objective selects native coarse center/size for the coarse arm or final center/size for the duplicate-final arm. It calls the actual native `SetCriterion.loss_boxes` using the original last Hungarian indices, original target dictionaries and original matched-count normalization, with `(10*bbox + 2*GIoU)/(decoder_layers+1)`. It does not rematch or construct prediction-derived targets. Original native/G/selected-geometry objectives remain in place. The two protocols differ only in arm, source path and selected endpoint; learning rates/core trainability and initialization recipe agree. Evidence: `K/source/matched_endpoint_geometry_objective.py:8–26`; `K/source/models/losses.py:808–846,856–886,917–919,959–986`; both protocol files; `AUDITED_SOURCE_DIFF.patch`.

Normal child argv has no checkpoint resume and names the shared official/G/A/B initialization spec. Initialization builds the common pretrained state and then a fresh optimizer; “fresh” does not mean random training from scratch. No M0 state enters this normal path. The M0 pair's equal initial-state/batch hashes and two-step gradients are engineering evidence only. M0 recorded two updates per arm and same-process recovery, not full-normal accuracy or cold recovery. Generic original target preservation is visible in source, but M0 and the normal responsibility counters do not demonstrate non-root target execution. Evidence: `S/127_native_model_initialization.py:13–59`; `S/107_train_dist_mod.py:129–130`; `S/050_main_utils.py:302–321,380–388`; both original normal-status argv and M0 receipts.

The frozen code/protocol target is historical **5705/4945** (60.0%/52.0%). The newer user acceptance target is **5724/4945** (60.2%/52.0%), as recorded in both terminal receipts. All observed checkpoints fall below both thresholds, so this metadata difference did not alter this pair's best selection or stop. The retained5677/4920 is short **47/25** of the current requirement. Future acceptance must use the newer goal; historical source was not edited. This audit does not establish Nr3D/Sr3D results or paper-level matching-protocol superiority.

## F. Evaluation type and deployment boundary — WARN, real_gt

Primary last/bbs is **real_gt** under the inspected implementation. Same-selected-query native/mask/final comparisons, repair/break counts and “best of all256” coverage use dataset GT for diagnosis. They are not deployable selection policies or replacement accuracy. The prediction-only mask extent and support/span path does not consume GT coordinates at inference; caption parsing and token evidence are distinct from GT coordinates. Evidence: `S/136_selected_geometry_eval_diagnostics.py:28–107`; `S/126_native_mask_geometry.py:5–43`; `S/107_train_dist_mod.py:133–176`; `S/102_grounding_evaluator.py:251–303,535–553`.

Cross-arm query identities change after training, so this is not an isolated fixed-query regression-head causal proof. Diagnostic arithmetic was checked, but saved-row coordinates were not recounted. Same-epoch comparison is supported at the logged aggregate level only.

## Specific unresolved evidence limits

1. **Six missing raw-row payloads.** Exact claimed paths, sizes and hashes are in `DETERMINISTIC_CHECK_RESULTS.json` under `limitations.raw_rows`. No independent coordinate/IoU recount, expression-ID uniqueness audit or cross-arm fingerprint comparison was possible.
2. **Dataset and 17 unique dependency bytes.** Dataset provenance was traced through source; original scan/annotation payloads and missing staged dependencies were not independently rehashed. `SOURCE_HASH_BINDINGS.json` preserves every missing path and expected digest.
3. **No bitwise E0-output equivalence.** Both E0 totals are5677/4920, but control row hash is `50ecf0ecc17fb048fef78a81948a5c08612fc2d3a5292d300bd8c956781dd06f` (14953876B) and coarse is `fd19a0338714ed34e98312537c8378eb421364a7cb7652df716c09a8199105c5` (14953818B). Raw gate axis counts also differ by one. The cause is unknown; equal metrics/model manifests do not prove identical forwards.
4. **No trained cold recovery.** No checkpoint was loaded and no serialized trained optimizer/scheduler/RNG state was checked. Loop counts, M0 same-process recovery and retained E0 archives are insufficient to certify cold normal recovery.
5. **Archive scope and current location.** Seven additional receipt inputs were read and hashed separately. Control and coarse remote E0 manifests agree across1295 tensors and have zero optimizer-state entries. Current control reconstruction `E:/codex_archives/pvg_final_duplicate_20261016_reconstructed/best.pth` and coarse reconstruction `E:/codex_archives/pvg_coarse_endpoint_20261017_reconstructed/best.pth` both exist at614926031B. Receipts claim SHA256 `30349b465394cf149ab177126b336a13fa7363c43cf017be9ceeee436a4f8aa3` and `749cce5f57ceb3b2c430ebc21d1b07f0f39e854bfaac79f849d89aa9dbd8bce0`, respectively; this reviewer did not rehash/load those payloads. They are content reconstructions, not original ZIP-byte copies. Historical `C:/Users/gb/.codex/archives/pvg_native_normal_20261010/best.pth` is currently absent. The E-drive control reconstruction is the current coarse source named in receipts. The coarse receipt retains a20261016 filename despite its October17 creation.
6. **Local, one-seed evidence.** Remote source/runtime was not revisited; full original train logs are not present; one seed and one validation dataset do not establish robust generalization. The review is same-family/provisional and backend model/effort remain unattested.

## Deterministic verification and blocking findings

`check_normal_pair.py` uses Python standard-library file parsing, SHA256, AST checks and integer/float arithmetic. It does not import the model. Run with existing `E:/python.exe -B check_normal_pair.py` from this directory. Final result: **66 PASS, 1 WARN, 0 FAIL**. The WARN is explicit incomplete local source-byte coverage. Audit execution history preserves an earlier failed overbroad “all240 bytes locally present” assertion, its original script and its results; absent evidence was not converted into a verification claim. Initial interpreter/long-path read problems are also recorded.

**Confirmed integrity blockers for the qualified completed-pair comparison: none.** This does not authorize another launch, certify all dependencies, approve a model promotion or satisfy the research goal. Claim acceptance remains blocked for the current60.2%/52.0% goal, a successful third mechanism, full raw-coordinate validation, complete source reproduction and cold recovery. Machine-readable checks, exact full-path evidence, limitations, claim effects and input hashes are in `EXPERIMENT_AUDIT.json`; `EXPERIMENT_AUDIT_COMPLETE.json` binds the final artifacts and exact100 R2 input seals.

## Actions and claim impact

- Retain the qualified negative result and E0; do not promote the auxiliary objective as proven effective. The same-epoch deltas are reportable with the single-seed/two-epoch and evidence qualifiers.
- For independent raw validation, provide the six hash-matching saved row files, recount dataset identities and IoUs, and investigate the E0 row/gate differences before asserting exact equivalence.
- For complete reproduction, supply matching missing source dependencies and dataset payload/split hashes. For cold recovery, provide a trained full-checkpoint restoration and optimizer/scheduler/RNG audit. Neither M0 nor an E0 model-content archive closes those gaps.
- Keep the newer5724/4945 target explicit in subsequent acceptance decisions. Three substantive effective mechanisms and matching-protocol Nr3D/Sr3D remain separate unfulfilled evidence obligations.
