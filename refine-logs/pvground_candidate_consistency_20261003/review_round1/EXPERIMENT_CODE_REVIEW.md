# PV-Ground candidate-consistency code review

Verdict: **FAIL — do not deploy this source for GPU preflight yet.**

Review time: 2026-10-03 05:25 CST. Reviewer: delegated Codex source reviewer, **same-family / provisional**. This is a fresh review of the supplied local files, not an external author review or an executed model audit. The exact underlying model/version is not independently exposed in this review context.

The intended semantic-target replacement is correct in its selection, loss formula, normalization, and matching boundary. One concrete CPU/CUDA index error prevents the reviewed implementation from reaching the GPU loss. Existing source-provenance assertions also need to be connected to the files actually executed. CPU fixtures and GPU sanity have **NOT_RUN** status; no gain, successful restoration on GPU, or training completion is certified.

All task-relative references below use `C:/Users/gb/.codex/tmp/pvground_candidate_consistency_20261003`. `P` denotes `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source`. `U` denotes `C:/Users/gb/.codex/tmp/pvg_training_interface_source_20260908`.

## Blocking findings

### B1 — CPU Hungarian indices are concatenated with CUDA qualified indices

**Severity: high. Blocks GPU preflight and the consistent arm.**

`pvground_candidate_consistency.py:21-23` creates `additional` with `selected[bid].nonzero()` and concatenates it directly with `queries`; it likewise concatenates `zeros_like(additional)` with `matched_targets`.

The actual native matcher moves its cost matrix to CPU and returns CPU `int64` tensors constructed from SciPy indices (`P/ported/models/losses.py:378-390`). The runner's hook only clones those tensors (`run.py:280-283`). In the real path, the model and predictions are on CUDA (`run.py:167`, `run.py:262-266`); `qualified_unmatched()` computes its boolean selection on the predictions' device (`pvground_semantic_assignment.py:10-23`). Therefore the appended indices are CUDA while the original matcher indices are CPU. Concatenating these nonempty tensors fails before the new native semantic-loss call. This is a source-established device mismatch in the actual runtime path, not a hypothetical edge case.

The CPU fixture uses CPU tensors for both sides (`cpu_test.py:41-53`), so even a successful CPU fixture would not detect this error.

**Minimal correction:** move the additional loss-only indices to the original match-index device before concatenation, keeping the original Hungarian pairs unchanged. Then regenerate the existing module pin/specs and perform the already planned CPU fixture and real two-step GPU preflight. No matcher, regression, or Mask change is needed.

### B2 — Existing provenance checks do not bind the imported modules or the supplied source-port pin

**Severity: medium. Blocks certification of the deployed source as the reviewed, pinned source.**

There are two concrete disconnects in the existing checks:

1. `run.py:43-46` hashes module files under `output`, which is an arm directory, then imports those modules by bare module name. The same pattern applies to the existing observation/task/source modules (`run.py:65-68`). The launcher uploads separate copies to the common root and each arm (`run_preflight_authorized.py:19-27`), invokes the common-root `run.py`, and prepends that common root to `PYTHONPATH` (`run_preflight_authorized.py:33`, `pair.py:25`, `pair.py:44-45`). The early semantic imports therefore use the common-root copies, not the arm copies whose bytes were checked. The imported-module receipt at `run.py:114-119` omits these custom modules. The new consistency-module check inherits this real path mismatch from the parent runner.
2. The specs already supply `source_port_sha256`, but `run.py:76-77` reads the source-port manifest and trusts its `files` mapping without comparing the manifest to that expected pin. The runner contains no reference to `spec['source_port_sha256']`. Recording the observed manifest digest in a later checkpoint/receipt (`run.py:427`, `run.py:469`) does not enforce the supplied expected value.

All five local module files currently match their supplied module pins, and the four reused G/D/C modules are byte-identical to the sealed parent copies. **No tampering or actual local mismatch was observed.** The finding is that the implemented assertions cannot establish what they claim to establish.

**Minimal correction:** apply the existing module checks to the copies that will actually be imported and confirm those import locations; compare the existing source-port digest to the already supplied spec value before trusting its file list. This repairs existing checks; it does not require a new manifest format, hashing framework, or compatibility layer.

## Correct implementation and nonblocking observations

### Semantic selection, replacement, and gradient boundary

- The same `qualified_unmatched()` function is reused by CE and contrastive correction. It detaches prediction and root geometry, uses strict IoU `> 0.5`, and excludes **all** Hungarian-matched queries. A high-IoU query assigned to another target is protected, not relabeled (`pvground_semantic_assignment.py:9-23`).
- Root target index zero is valid before filtering. The expanded pairs retain every original pair and append only selected unmatched queries mapped to root zero (`pvground_candidate_consistency.py:18-24`). The actual matcher output is neither modified nor fed back into regression or Mask losses. The expansion is passed directly and only to `loss_sem_align()`.
- Native `loss_sem_align()` first initializes unmatched not-mentioned positives, then overwrites the positive map at the supplied match indices (`P/ported/models/losses.py:660-671`). Appending the selected semantic correspondences therefore replaces their not-mentioned targets. It does not add a second contradictory target term.
- The code calls the native method for both old and expanded targets, preserving its temperature, positive/eos weights, other-entity term, and bidirectional formula. The **original** match-count denominator is deliberately retained (`pvground_candidate_consistency.py:27-33`; native `losses.py:831-835`). More qualified queries can consequently increase the effective semantic supervision mass; that is part of the stated strategy, not an implementation error.
- `.5 / 7` matches the actual source: ScanRefer weight is 0.5 and the six-decoder loss averages seven heads (`P/ported/models/losses.py:852-853`, `P/ported/models/losses.py:944-954`). Matcher call index 1 is `last_`, because the native order is proposal, last, then heads 0–4. Thus the runner's use of `matching[1]` is correct.
- The total objective is native loss plus the existing G CE correction plus `(expanded_native_semantic - original_native_semantic) * .5 / 7` (`run.py:287-299`). At the mathematical level this replaces the original final semantic term. Intermediate-layer targets and losses retain their original calls.
- With no qualified query, the expanded pair values equal the original pairs and the correction is zero. The prepared CPU test checks zero correction and exact equality of final-query projection gradients, protects an overlapping other-target match, and compares the qualified case to an explicitly constructed native expanded-target loss (`cpu_test.py:60-80`). These are meaningful assertions, but they have **not been executed in this review**.
- The qualified-case fixture uses `allclose` for loss and gradient (`cpu_test.py:76-79`), while its receipt field is named `exact_expanded_native_loss_and_gradient` (`cpu_test.py:84`). That field should be described as agreement within the stated tolerances. Exact equality is asserted only for the no-qualified query-projection gradient case. This wording issue does not change the objective.
- The witness checks for no direct gradient into final box coordinates/sizes. Shared model parameters and token-to-query normalization may still change gradients or later predictions for other queries. The implementation correctly promises unchanged matching/targets for those queries, not unchanged global training gradients.
- High IoU remains geometric qualification, not verified physical-instance identity. The plan correctly treats efficacy as unproven. All 256 queries remain in the native logits and normalization; there is no candidate pruning or unconditional root labeling.

### Architecture, data, metrics, and optimizer

- Comparison of the complete generated `run.py` against the sealed parent finds only P2 exclusion, consistency imports/correction, the disposable preflight replacement, and consistency metadata. The four reused module files are byte-identical to the parent. `p2` is asserted false, the optional P2 installation is unreachable, and no P3/fused tail/teacher/quality head is installed. The existing G/D/C architecture is retained.
- The three specs differ only in output root and the consistency boolean. Both arms use the same author parent checkpoint and original G delta, seed 2027, batch 8, learning rates `1e-5`, and one fit pass. The delta shape/dtype/key set is checked and strictly loaded (`run.py:121-152`). The old optimizer is not loaded into a training arm.
- `BaseTrainTester.get_optimizer()` creates AdamW (`U/main_utils.py:283-365`); the runner invokes it fresh after E0 and checks weight decay 0.0005 and clip norm 0.1 (`run.py:168-172`, `run.py:417-418`). RNG is reset independently of the seeded loader. The last fit batch has two rows because 29,778 is not divisible by eight; `drop_last=False` correctly preserves all rows and gives 3,723 updates.
- The archived split file's existing digest matches the input manifest. Static recount confirms 29,778 unique fit rows, 6,887 unique holdout rows, and zero overlap. The runner also reconstructs the scene-based split and checks physical-scene disjointness, then verifies exact once-only fit coverage (`run.py:180-189`, `run.py:234-249`, `run.py:435-445`). Actual dataset loading and tensor-fixture checks remain unrun here.
- Formal validation uses a separate mode and the 9,508-row validation split, after loading the trained endpoint. Holdout is observed without optimizer updates; formal validation is not used by the qualification function during training. The plan correctly discloses that the author pretraining already saw the module-holdout scenes.
- `prepare()` passes points/voxels, text, detector boxes/classes/masks, and superpoints to the model; training GT is added to criterion inputs only after `model(inputs)` (`run.py:257-284`). Evaluation also predicts before adding GT to the native loss/evaluator. The new correction is not called by `evaluate()`.
- Root scoring remains the native last/bbs rule, with bbf retained as a diagnostic. Root token maps used for scoring come from parsed language (`P/joint_det_dataset.py:981-1060`), and the selected query is determined before GT overlap is used for metrics. `filter_non_gt_boxes=False`; the GT-based candidate oracle is recorded only as an evaluation diagnostic (`run.py:350`, `run.py:379-391`). Row-order checks, strict `>` thresholds, native evaluator hit recounts, and Mask sum checks are retained (`run.py:369-407`). No new validation-geometry gate or GT selection was found.

### Preflight, restoration, SSH, and execution boundaries

- The preflight is designed to run one backward-capacity witness with zero optimizer updates, then exactly two optimizer steps on a real fit batch. It serializes delta and optimizer state in memory, reloads them, checks all model tensors against the reloaded state, and checks step counters (`run.py:318-343`). This is a suitable planned implementation check after B1 is corrected. It is not a completed result or a disk-save test.
- The capacity witness uses `model.train()`, so it can update training buffers and consume RNG even though it does not call `optimizer.step()`. That is confined to the disposable preflight process; each full arm reloads original G separately. The code does not reuse preflight weights for full training.
- The launcher requires an accepting review with no blocking findings, runs CPU first, stops on a failing exit code, and uses the existing GPU lock for preflight. It creates a new isolated remote root and does not launch `pair.py` or delete files. Exclusive local log creation and remote directory creation prevent a silent rerun into the same attempt.
- SSH loads system known-host keys and does not install an accept-unknown-host policy (`run_preflight_authorized.py:13-16`). Paramiko's default rejecting policy remains in force. Passwords come from the environment, and remote command arguments/environment entries are shell-quoted with `shlex.join`. No SSH host-verification bypass or shell-injection defect was found in the reviewed launcher. **No network connection was made by this review.**

## Full-pair resource hold

**The full pair must not launch with the currently reported capacity.** This is an unmet execution prerequisite, separate from the source defects above. `runtime_inventory.json:16-25` records only **437,751,808 free bytes** for the actual model/output directory filesystem. The much larger values obtained by querying individual parent-checkpoint paths do not establish capacity for the output directory.

`pair.py:28` correctly checks the pair root for at least `3 * preflight.serialization_bytes + 256 MiB`; it also rechecks `2 * serialization_bytes + 128 MiB` before each training arm (`pair.py:38-42`). This accounts for a retained first endpoint and a second arm's atomic checkpoint replacement. The reviewed checkpoint path writes a temporary latest file, replaces latest atomically, then renames latest to terminal without retaining another endpoint copy (`run.py:421-448`).

The new serialization size is **not measured yet**. For scale only, the archived original-G terminal is 342,299,695 bytes; using that value in the pair formula gives 1,295,334,541 bytes, well above the reported free space. This is not a substitute for the new preflight measurement. Establish adequate space on the actual output path and let the existing gates pass before full training. This review authorizes **no deletion, cleanup, migration, or pair launch**.

## Scope and verification limits

Read in full: `EXPERIMENT_PLAN.md`, `research_contract.md`, `pvground_candidate_consistency.py`, `cpu_test.py`, `build_runner.py`, generated `run.py`, `prepare_pair.py`, `pair.py`, `run_preflight_authorized.py`, all three specs, and the four reused modules. Compared the complete runner/pair diff to `P/run.py` and `P/pair.py`. Read the native matcher, semantic loss, matching-count calculation, and aggregate loss from `P/ported/models/losses.py`. Inspected relevant model wiring, dataset token-map/input construction, native evaluator scoring, optimizer/criterion construction, existing source/runtime/split manifests, and the supplied resource inventories.

Actually performed: read-only source inspection; complete textual runner/pair comparison; byte equality and existing-pin checks for local modules; static split recount and existing split-pin check; confirmation that the archived runtime environment matches the spec's canonical digest; local Python AST parsing of task Python files. The pinned runtime is Python 3.7.11 / Torch 1.10.2+cu111; local syntax parsing used Python 3.13 and is not target-runtime execution. The `ast.Module(body=...)` fixture construction is consistent with the pinned Python 3.7 AST, so this review does not invent a newer-Python compatibility requirement.

**NOT_RUN:** CPU loss/gradient fixtures; target-runtime imports; GPU forward/backward; two optimizer steps; CUDA device-mismatch reproduction; GPU memory measurement; model/optimizer serialization witness; disk checkpoint save/reload; SSH deployment; fresh remote capacity measurement; E0/holdout/formal evaluation; either full training arm; efficacy or innovation assessment. No implementation file was edited, no GPU job was started, and no file was deleted by this review. The absent local CPU/preflight receipts and TODO tracker agree with this status.

After the minimal B1/B2 corrections, review the revised source, then run the already planned CPU and real GPU preflight. Keep the full pair held until actual output-path capacity is adequate. This review supplies no new experiment or hyperparameter search.
