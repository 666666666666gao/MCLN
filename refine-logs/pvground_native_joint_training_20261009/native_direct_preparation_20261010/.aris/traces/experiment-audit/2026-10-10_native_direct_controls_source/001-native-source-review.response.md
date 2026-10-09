# Experiment audit: isolated native direct controls

Date: 2026-10-10. Overall verdict: **WARN**. Integrity status: **warn**.

**Blocking source issues before the bounded CPU module checker: 0.** The reviewed code and run bundle support proceeding to that engineering check when the executor has the relevant execution authorization. This is a source review, not execution authorization, a completed model preflight, or approval of accuracy results. **GPU work and formal control results are not admitted by this audit.** The planned full-model checks and result-review prerequisites remain outstanding.

Execution scope: `ISOLATED_NATIVE_CONTROL_SOURCE_NOT_LAUNCHED`.

Requested reviewer route: `gpt-6-astra`, reasoning `max`, fresh Codex reviewer. Actual model and actual effort: **UNATTESTED**; the available task context does not provide an independent actual-model attestation. Review independence: **same-family**. Acceptance status: **provisional**. No cross-family or actual Astra identity claim is made.

## Evidence boundary and reproducible checks

The reviewer directly read the 45 originally listed artifacts and four supplemental inputs: `CPU_BUNDLE.json`, `prepare_direct_control_cpu_bundle.py`, `run_direct_control_modules_cpu_authorized.py`, and the local `034_whole_mask_range.py` snapshot. All 49 input hashes are recorded in `EXPERIMENT_AUDIT.json`; exact byte equality was checked for the 11 duplicated unchanged source files, with their shared text reviewed once. The three changed modules were read in both versions. Four controllers were read and are byte-identical. The four init manifests and protocols were read and compared as parsed JSON.

Only local text reads, hashes, byte comparisons, JSON parsing and Python AST parsing were executed. The deterministic checker parsed 37 Python artifacts and the transport script's embedded remote program without importing or executing them. The first invocation through the PATH `python` failed with `No pyvenv.cfg file`; rerunning the static script with the available `E:/python.exe` succeeded. No project module, Torch operation, model construction, checkpoint load, neural computation, remote connection, credential file or active-training status was accessed by this reviewer.

`DETERMINISTIC_CHECKS.json` records actual deterministic checks. It confirms all original/prepared manifest hashes, the CPU bundle's prepared hashes, the 14 original hashes and the additional member-statistics dependency hash. It also records that local `CPU_MODULE_WITNESS.json` and `CPU_MODULE_EXECUTION.json` do not exist at this audit snapshot. Their absence is not a remote status check.

Citation roots used below:

- `R/`: `D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/`.
- `C/`: `D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/native_direct_controls_20261010/`.
- `P/`: `D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/`.
- `M`: `C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/source_review/snapshots/034_whole_mask_range.py`.

For unchanged files, citations to `R/` also apply to the byte-identical `C/source/` file at the same lines. Per-input hashes bind this observation to this snapshot only.

## A. Ground-truth provenance — WARN

The supplied native training path constructs `Joint3DDataset` with explicit train/validation splits and the common data root (`R/train_dist_mod.py:55–96`). Training targets are drawn from the batch's `center_label`, `size_gts`, `sem_cls_label` and `gt_masks`, filtered by the batch's valid-box mask (`R/models/losses.py:856–885`). The selected-query auxiliary mask objective uses the dataset root mask and aggregates it into superpoints (`R/selected_query_mask_objective.py:13–32`). Its query selection is prediction-dependent, but its mask target is not a model prediction. The semantic-assignment correction uses the root box and root token maps as training labels (`R/pvground_semantic_assignment.py:9–47`) and is invoked only while `native_joint_training` is true (`R/models/losses.py:959–970`; `R/models/pv_ground.py:615–617`).

There is an existing prediction-derived *training consistency target*: `sp_src_masks_2` is derived from predicted masks and supervises corresponding text-mask losses (`R/models/losses.py:592–621`). This is not a dataset accuracy reference and must not be described as an additional independent ground truth.

The geometric reference is also explicitly prediction-derived, with detached mask logits and hard support membership; it is an inference feature, not evaluation ground truth (`R/native_mask_geometry.py:5–23`). The new A/B controls do not receive GT during their forward paths (`C/source/mask_support_corrector.py:26–74`; `C/source/extremal_span_mixer.py:31–121`). Input assembly supplies points, expression text, detector inputs and superpoints; dataset labels are attached after `model(inputs)` for criterion computation (`R/train_dist_mod.py:163–175`; `R/main_utils.py:449–458,521–534`).

The CPU checker uses random tensors and eight synthetic superpoints, and uses original-module outputs as equivalence references (`C/check_direct_control_modules_cpu.py:73–121`). Its labeling explicitly denies scenes, GT and accuracy claims. The local member-statistics source counts sampled rows per superpoint, not deduplicated visible instances (`M:12–37`); the documentation preserves that distinction (`C/DIRECT_CONTROLS.md:23`).

**Limit:** dataset files, `Joint3DDataset`, `GroundingEvaluator`, the complete imported model repository and the official benchmark evaluator are not review inputs. Their dataset provenance or official-evaluator equivalence is not independently certified here. No real dataset was evaluated.

## B. Score normalization — PASS within reviewed source

The reported grounding rates divide hit counts by evaluator ground-truth counts (`R/train_dist_mod.py:237–240`), and the return path retains integer hits and enforces exactly 9508 evaluator entries for both thresholds (`R/train_dist_mod.py:252–257`). No reviewed reporting code divides accuracy by the model's own maximum, mean or other prediction statistic. The formal evaluator's internal implementation remains outside this input set.

`native_root_bbs` uses semantic probabilities and the existing expression maps, without an additional score head (`R/native_root_bbs.py:4–9`). It is used to select the auxiliary training query (`R/selected_query_mask_objective.py:8–9`). A modifies the query-mask output; B modifies center/size and gate diagnostics; neither patch writes semantic logits or introduces deployed reranking (`R/models/pv_ground.py:567–574,604–613`).

Member-count normalization, support pooling, geometric scaling, softmax and bounded gates are model-feature computations, not normalized performance results (`C/source/mask_support_corrector.py:43–53`; `C/source/extremal_span_mixer.py:48–106`). Fixed-half mode sets both effective and reported raw gates to 0.5 (`C/source/extremal_span_mixer.py:99–108`).

## C. Result-file existence and claim matching — WARN

The current control artifacts consistently say source prepared, not constructed, not launched, and no new accuracy (`C/DIRECT_CONTROL_PREPARATION.json:69–93`; all four `init.json:20–23`; all four protocols `:98–102`; `C/EXPERIMENT_TRACKER.md:7–13`). These pending labels agree with the absence of the two local CPU execution/witness files checked during this audit. The CPU checker's success dictionary is a code template written only after assertions; it is not an existing result (`C/check_direct_control_modules_cpu.py:140–161`). The transport script likewise preserves execution status and only copies a witness after successful exit (`P/run_direct_control_modules_cpu_authorized.py:69–84`); it was not executed.

The documentation's historical `5677/4920` literals have no corresponding result files among these inputs (`C/DIRECT_CONTROLS.md:35`). They are explicitly separated from new controls, but this audit does **not** verify or approve those historical numbers. The numerical requirements `5658/4850` are correctly the minimum integer hit counts strictly above 59.5%/51% of 9508. They are targets, not observed outcomes. The coded best-checkpoint ordering is visible in `R/main_utils.py:167–169,365–385`; no selected checkpoint or best score is verified here.

## D. Reachability and dead code — WARN

The source establishes the intended active chain: the model factory calls the initializer (`R/train_dist_mod.py:111–129`); it installs A and B (`C/source/native_model_initialization.py:46–65`); PV forward calls A, constructs the mask reference, then calls B (`R/models/pv_ground.py:567–574,593–613`). Training and evaluation call the criterion (`R/main_utils.py:406–412,449–470,521–534`). Evaluation calls `GroundingEvaluator.evaluate`, and ordinary training records its returned metrics (`R/train_dist_mod.py:208–257`; `R/main_utils.py:367–385`). These paths are reachable by source inspection, not witnessed execution.

Existing dormant helpers must not be credited as completed checks: `calculate_diou_3d` is defined but its loss call is commented out (`R/models/losses.py:103–133,546–557`); `verify_native_replacement` is defined but not called by the audited training/checker artifacts (`R/pvground_semantic_assignment.py:53–93`); `M:40–67` defines `mask_range_evidence`, whereas A imports and calls only `member_statistics` (`C/source/mask_support_corrector.py:9,66–67`). No new numerical claim depends on these dormant helpers, and removing them is not required for this task.

## E. Scope and ablation interpretation — WARN

Actual control-model executions observed: **0**. Actual dataset expressions evaluated in this audit: **0**. Actual training seeds completed for these controls: **0**. Four configurations are prepared; three epochs, seed 2027, batch/effective batch 8, world size 1 and 9508 formal evaluation rows are planned, not measured (`C/common_behavior_check/NORMAL_NATIVE_RUN_PROTOCOL.json:74–88`; `C/DIRECT_CONTROL_PREPARATION.json:80–92`).

The 14-source comparison finds changes only in `extremal_span_mixer.py`, `mask_support_corrector.py`, and `native_model_initialization.py`. Training entry, criterion, ranking helper and PV forward are byte-identical. Across the four manifests only the declared control mode changes; protocols differ only in their `native_init_spec` path. The shared protocol fixes dataset, seed, model dimensions, LR, clipping and three-epoch schedule (`C/common_behavior_check/NORMAL_NATIVE_RUN_PROTOCOL.json:6–56`).

The optimizer includes exactly trainable parameters, separates native core/backbone/text/additions, excludes the frozen text encoder, and asserts nonoverlap (`R/main_utils.py:292–306`). Each controller uses the same entry and explicitly excludes checkpoint resume and full-core freezing (`C/common_behavior_check/normal_joint_controller.py:31–35`, identical in all four arms). A/B freezing changes the effective optimization capacity as intended; this is not an equal-active-capacity comparison.

- **Common:** full A plus learned B preserves the original computations by source inspection. The new constructor flags themselves consume no additional random draws. Actual same-batch PV outputs and loss equivalence remain untested.
- **A bypass:** forward returns the original query-mask tensor while still constructing the same member geometry (`C/source/mask_support_corrector.py:64–74`). Its 27,841 registered parameters are frozen (`C/source/native_model_initialization.py:62–63`). It tests A's incremental continuation effect from a shared trained history, not training that never used A.
- **A content-only:** the five mask-state columns and one count column are zeroed; the two 32-wide content projections remain, and all nine box-position columns are already zero because the factory passes `False` (`C/source/mask_support_corrector.py:39–53`; `C/source/native_model_initialization.py:46`). The output is still a residual added to the original query mask (`:54–57`). Thus only the residual predictor is content-only; the resulting mask and downstream B still retain native mask information. No claim of removing all mask information is supported. The registered parameters remain trainable, but zero columns alter their effective task gradients.
- **B fixed-half:** the learned gate is replaced by 0.5 for both center and size, and all 29,793 B parameters are frozen without deleting state (`C/source/extremal_span_mixer.py:99–108`; `C/source/native_model_initialization.py:64–65`). It tests learned gating against fixed fusion, not the unique value of extremal sources. Mask extents are discrete and retain the existing native prior for invalid references (`R/native_mask_geometry.py:13–23`).

All arms declare the same official/G/support/span parent hashes and supervision flags (`C/common_behavior_check/init.json:4–13`, identical across arms). The documentation correctly discloses inherited trained history, unequal effective capacity, discrete references, single-seed limits, and the distinction between gating and extremal-source claims (`C/DIRECT_CONTROLS.md:10,16–23,35,41`). “Criterion unchanged” means the same loss implementation and flags; changed predictions can change assignments and loss values. It does not mean control losses must equal common losses.

## F. Evaluation type — PASS for explicit classification

The current audit is a **static source/integrity review**. The planned CPU check is `synthetic_module_engineering_fixture_not_accuracy_evaluation`. Within the checklist taxonomy, original-model output equality is a **synthetic_proxy** consistency check, combined with algebraic and local-gradient assertions. It supports only those engineering properties for its fixture. It is neither a real-GT accuracy evaluation nor full PV validation (`C/check_direct_control_modules_cpu.py:73–88,104–138,140–159`).

The planned formal ScanRefer evaluation is an intended **real_gt** route, subject to later verification of the actual dataset/evaluator inputs and complete output records. There is no present formal result to classify as completed real-GT evidence.

## Payload loading, state and checker coverage

The initializer hashes payload bytes before load and preserves strict state loading (`C/source/native_model_initialization.py:13–15,31–61`). It verifies official `module.` prefixes, original 1234 state tensors, installed-G 1271 state tensors, the G delta's 1072 entries and exact shape/dtype, A payload role/zero-geometry/prefix/10 entries, B's source mode and parent-support digest, and final 1295 state tensors (`:27–66`). Frozen branch parameters remain registered. Original trainability and text freezing are asserted (`:67–68`). Architecture metadata includes both new control modes and trainability; recovery compares this metadata and loads model state strictly (`:72–84`; `R/main_utils.py:133–145`). Optimizer, scheduler, RNG and retained metrics are included in the full checkpoint (`R/main_utils.py:151–163`).

These are **source assertions**, not observed payload/constructor/recovery passes. The reviewer did not read the remote checkpoint payloads. The CPU checker actually plans to load only A/B payloads into standalone modules (`C/check_direct_control_modules_cpu.py:44–71`). It does not invoke `configure_native_model`; its freezing calls are manual. Its `strict=True` loads and state-key comparisons do not independently prove payload dtype equality; only the G path contains an explicit dtype equality assertion in the reviewed initializer. Therefore the documentation's planned key/shape/dtype and full-state checks remain to be witnessed, without requiring speculative new loading logic now (`C/DIRECT_CONTROLS.md:28–30`).

Planned checker coverage is meaningful but bounded: common A mask equality; common B box/evidence equality; common member-statistics equality; bypass tensor identity; zero content-only observation columns; fixed gates and fixed-fusion formula; frozen branch flags; common-A query/output gradients; fixed-B native-box gradients (`C/check_direct_control_modules_cpu.py:91–138`). It does **not** compare criterion loss, run native ranking/data paths, test content-only parameter gradients, test actual optimizer membership, construct the full PV model, or save/recover it. The emitted report explicitly marks those major omissions false (`:156–159`).

The fixed-B gradient test varies only `native_coarse_center/size` while retaining an already materialized independent mask reference (`:131–137`). Its exact-half derivative is conditional on that fixture. In full PV forward an invalid mask reference uses the native prior (`R/native_mask_geometry.py:19–21`), so a broader claim that every full-chain native-box derivative is always 0.5 would be false. No such broad conclusion is admitted here. A single random eight-superpoint fixture also does not establish coverage of all invalid/degenerate/tied-support cases or extremal-geometry accuracy.

## CPU input and transport source assessment

`C/CPU_BUNDLE.json:2–18,20–47` supplies the checker's required original path and expected hashes, including the `whole_mask_range.py` import dependency and checker itself. Its prepared hashes match local bytes, and the additional dependency hash matches the directly read local snapshot. `P/prepare_direct_control_cpu_bundle.py:9–20` builds this input structure. The earlier absence of this file from the initial review request is closed by the supplemental artifact; there is no remaining missing-bundle finding.

The optionally reviewed transport script checks the audit scope, verdict and every audited hash before preparing execution (`P/run_direct_control_modules_cpu_authorized.py:13–18`). Its embedded program requires a fresh isolated CPU directory, checks destinations remain below that directory, verifies transported bytes, masks CUDA visibility and runs only the module checker in the pinned environment (`:37–57`). It does not start any of the transported training controllers. Original model source is read for comparison rather than written. These are static observations about the script, not a remote-execution receipt or an authorization decision. No referenced authentication helper, host-key file or transport witness was opened.

## Required next evidence and claim impact

No source repair was found necessary before the narrowly scoped CPU module checker. Preserve its actual exit status, stdout/stderr, Torch version, bundle and payload hashes; a source audit alone is not a pass of that checker.

Before later admitting GPU control training, the documented pending workflow still requires full PV construction/strict loading for the chosen modes; same-input common/native output and native-loss comparison with controlled stochastic state; actual per-arm gradients and optimizer groups; native data/score/criterion checks; full checkpoint save/cold recovery; and the already specified active-normal terminal/result review and necessity decision (`C/DIRECT_CONTROLS.md:25–33`; protocols `:93–106`). The complete repository, evaluator, data and environment bindings must accompany that evidence. This reviewer has not checked the active run and does not infer its status.

Supported now: the sealed source/configuration differences, preservation of the 11 shared source files, declared ablation semantics, and the bounds of the planned checker. Needs actual execution: module equivalence/gradient assertions, full 1295-state construction, parent payload contents/dtypes, native loss equality, optimizer responsibility and recovery. Unsupported now: improved REC/Mask accuracy, completed controls, necessity of A/B/C, unique extremal-source value, target attainment, multi-seed robustness, Nr3D/Sr3D effectiveness, or a training history that never used the inherited components.

The accompanying seal records hashes and unchanged-input verification; it is not a cryptographic identity attestation, remote-state witness, or result approval.
