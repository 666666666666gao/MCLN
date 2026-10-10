# ReferIt3D metadata actual-outcome audit

Date: 2026-10-10T00:43:55Z
Reviewer: fresh delegated Codex agent `/root/pvg_referit_metadata_actual_20261010`.
Requested model/effort: `gpt-6-astra` / `max`; actual model/effort: **UNATTESTED**.
Review independence: **same-family**. Acceptance: **provisional**.

**Overall verdict: WARN. Bounded metadata check: PASS. Blocking issues: 0. Nonblocking issues: 1.**

The stored output supports the explicitly limited CSV-eligibility and selected-scene file-availability claim. It does not establish a loader, model, author-protocol reproduction, REC accuracy, GPU readiness, or training admission. The sole warning concerns unsynchronized cross-host timestamps; absent full-loader/model checks are bounded exclusions, not failures of this metadata task.

File references below use the following roots: reader, plan, results, transport and `source_review/` are relative to `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/referit_metadata_20261010`; `joint_det_dataset.py` is `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/runtime_binding/dataset_source/src/joint_det_dataset.py`; `NATIVE_SOURCE_PORT.json` and the protocol source note are in `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2`.

## Recorded outcome

| Dataset | Runtime split | Metadata split | Rows in scenes | Eligible expressions | Used scenes | Missing superpoints | Missing GroupFree |
|---|---|---|---:|---:|---:|---:|---:|
| Nr3D | train | train | 32,919 | 32,919 | 511 | 0 | 0 |
| Nr3D | val | test | 8,584 | 7,899 | 130 | 0 | 0 |
| Sr3D | train | train | 65,846 | 65,846 | 1,018 | 0 | 0 |
| Sr3D | val | test | 17,726 | 17,726 | 255 | 0 | 0 |

These are recorded metadata counts, not independently recounted remote CSV contents or actual dataset lengths. All four `actual_native_loader_length` fields and `REC_accuracy` are null. Nr3D CSV total is 41,503; Sr3D is 83,572. Train plus validation prefilter rows equal the respective totals; repeated per-dataset CSV hashes/bytes/totals and per-split pickle sizes agree. Evidence: `METADATA_RESULT.json:7`, `:28`, `:49`, `:70`, `:101`; `RAW_STDOUT.json:1`.

## A. Ground truth provenance — PASS

The reader uses existing dataset CSV fields and native scene lists. It does not create a target or reference from a model, compare predictions with GT, or measure agreement. The nonnegative `target_id` assertion is only a metadata check; valid object membership and index range remain unknown. No official performance evaluator was invoked or required for this result.

Evidence: `read_referit_metadata_authorized.py:17`, `:31`, `:38`, `:39`, `:60`; `joint_det_dataset.py:418`, `:495`.

## B. Score normalization — PASS

The output contains row/scene counts, file-presence lists, byte sizes and SHA256 values. Counts come from `len()`; there are no prediction statistics, metric denominators, score rescaling, or reported REC scores.

Evidence: `read_referit_metadata_authorized.py:43` through `:51`, `:60`; `METADATA_RESULT.json:101`.

## C. Result existence and agreement — WARN; bounded check PASS

All ten nominated project artifacts exist. A literal-preserving JSON parse confirms that every value in `RAW_STDOUT.json` equals its counterpart in `METADATA_RESULT.json`; their byte hashes differ only as expected for raw versus pretty-printed representations. The stored transport exit is **0**.

The local native file SHA256 is `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d`. It matches the reader pin, reported dataset source and retained port entry. All four split SHA256 values and absolute paths match the port. The source-review JSON has the exact sealed hash/size; four overlapping project inputs and the already-read policy also match the review/seal. The separate historical experiment-bridge skill and other seal outputs were outside this audit and were not rehashed.

Evidence: `RAW_STDOUT.json:1`; `METADATA_RESULT.json:92`; `TRANSPORT_EXIT.json:1`; `read_referit_metadata_authorized.py:23`; `NATIVE_SOURCE_PORT.json:42`, `:49`, `:119`, `:126`, `:564`; `source_review/EXPERIMENT_CODE_REVIEW.json:18`; `source_review/SOURCE_REVIEW_SEAL.json:34`.

**W1, nonblocking:** remote `observed_cst` is `2026-10-10T08:35:31.961230+08:00`, while local transport started at `08:35:23.862079` and finished at `08:35:26.721448`. The remote label is **5.239782 seconds after** the local finish. Local transport elapsed time is **2.859369 seconds**. The artifacts do not attest clock synchronization or the cause of the discrepancy. Preserve both timestamps and do not claim a synchronized cross-host timeline. No metadata rerun is required for bounded acceptance. Evidence: `METADATA_RESULT.json:3`, `TRANSPORT_EXIT.json:1`; their clocks are read separately at reader lines `56`, `70`, `76`.

## D. Called code and artifact production — PASS

The top-level payload calls the extracted resolver, iterates two datasets and two splits, and emits four records. The local wrapper captures stdout/exit before parsing the result. The actual artifacts agree with that reachable route. Train/test disjointness assertions exist before the final print; this reviewer did not reopen the split contents or rerun those assertions. No uncalled metric function is represented as a result.

Only the resolver AST node is compiled. Its body is limited to constructing known CSV paths and checking files; the native module's top-level torch/model imports and loader construction are not executed by that step. This review did not execute either input Python file.

Evidence: `read_referit_metadata_authorized.py:25` through `:31`, `:34`, `:43`, `:52` through `:55`, `:71` through `:80`; `joint_det_dataset.py:65` through `:78`; `RAW_STDOUT.json:1`, `TRANSPORT_EXIT.json:1`. This is source/artifact corroboration, not an independently signed runtime trace.

## E. Scope and operational boundary — PASS

The reader's Nr3D train branch uses all rows in train scenes; Nr3D validation additionally requires lowercased `correct_guess == 'true'`. It does not add a Nr3D `mentions_target_class` filter. Both Sr3D splits require lowercased `mentions_target_class == 'true'`. Runtime `val` selects the native `test` scene list while retaining `val` for superpoints, GroupFree files and scans pickle. These definitions match the native source, including the base and structured annotation branches.

Evidence: `read_referit_metadata_authorized.py:34` through `:42`, `:49`; `joint_det_dataset.py:402` through `:429`, `:439` through `:453`, `:483` through `:510`, `:533` through `:549`, `:215`, `:1204`, `:1262`, `:1358`.

The inspected remote payload imports only standard-library modules, reads fixed inputs, checks selected-scene paths, and stats pickle sizes. It contains no current-training query, model/CUDA/optimizer call or remote write. Local output writes are explicit. Recorded zero flags agree with that source; they are not independent telemetry about other remote processes. The reviewer made no SSH call, remote write, current-training observation or model/GPU call.

The following remain bounded exclusions:

- **Actual loader length and pickle membership:** only pickle sizes are recorded. The native constructor unpickles scans and loads superpoints for every pickle scene; Nr3D annotation processing also indexes scan objects. Checking CSV-selected scenes cannot prove all constructor inputs are available. Training can multiply/mix annotation lists.
- **NPY/PTH validity:** `is_file()` checks do not validate arrays, tensor contents, shapes, object classes or sample construction.
- **Author protocol:** GroupFree proposal-file existence does not establish the `butd_cls` scene-proposal protocol. The native branch replaces proposal boxes/masks for `butd_cls`. The note's checkpoint flags and separate entrypoint statements were not independently audited beyond the supplied note.
- **Full performance and readiness:** REC accuracy, final architecture, training effectiveness, seeds, GPU behavior, cold resume and training admission are untested and unclaimed.

Evidence: `PLAN.md:3`, `:5`, `:7`; `read_referit_metadata_authorized.py:17` through `:60`, `:73` through `:80`; `METADATA_RESULT.json:93` through `:101`; `joint_det_dataset.py:205` through `:225`, `:249` through `:258`, `:519` through `:525`, `:1204`, `:1269` through `:1275`, `:1367` through `:1372`, `:1619`; `REFERIT_OBJECT_PROTOCOL_SOURCE_NOTE_20261010.md:5`, `:10`, `:12`, `:15`.

## F. Evaluation type — PASS

Classification: **`metadata_file_availability_only`**. Existing dataset CSV metadata is the provenance. This is not a performance evaluation and none of the skill's real-GT/proxy/simulation/human performance labels grants an appropriate claim ceiling here. It is not model-generated GT or a self-normalized score.

Evidence: `PLAN.md:5`; `METADATA_RESULT.json:2`, `:26`, `:47`, `:68`, `:89`, `:98`, `:101`.

## Claim ceiling and follow-through

The four recorded eligible-expression counts and empty missing-path lists are usable with the metadata/selected-scene qualifiers above. Exact loader size, pickle membership, valid NPY/PTH contents, author protocol reproduction and full accuracy are not established; their absence does not block this bounded outcome. No input change or rerun is required. Retain W1's clock qualification when citing chronology.

This audit provides **no GPU or training admission and no whole-goal completion**. Future loader/protocol/model checks belong to separately authorized work after model selection. Neither this review nor the historical source-port status is evidence about current training progress.

## Evidence and trace

The JSON companion lists SHA256 values for every read input: ten project artifacts and six skill/policy files. The reader was reviewed in full; the native dataset and source-port semantics were reviewed only where relevant to this bounded check. Raw remote CSVs/pickles/NPYs/PTHs, private stderr, credentials/AUTH, personal memory and optional handoff were not read.

Trace: `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/referit_metadata_20261010/.aris/traces/experiment-audit/2026-10-10_run01_actual_metadata_004355`.
It preserves the original task, optional follow-up, raw local check outputs, corrected deterministic checks and this full review. A first local timestamp check rounded fractional seconds when PowerShell converted parsed date objects back to text; the raw trace is retained and the final deltas above were recomputed from the original ISO strings. No input artifact was changed.

Seal: `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/referit_metadata_20261010/actual_review/ACTUAL_REVIEW_SEAL.json`. The seal is an ordinary unsigned digest snapshot, not immutable or independent attestation.

