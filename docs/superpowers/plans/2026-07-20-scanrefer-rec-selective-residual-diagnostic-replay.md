# ScanRefer REC Selective-Residual Diagnostic Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay the rejected five-fold selective-residual experiment with its original train-only numerical protocol and publish enough immutable evidence to explain every eligibility failure without opening the held-out calibration rows or any ScanRefer validation source.

**Architecture:** Keep the existing 12-model grid, five scene folds, ten epochs, OOF gains, percentile margins, strict IoU thresholds, and bootstrap seeds unchanged. Add downstream diagnostics to the pure selector and trainer, split the fixed 506/56 training scenes before expensive materialization, and publish either a staged candidate or a diagnostic-only baseline receipt into an exclusively reserved directory whose completion receipt is written last. Harden the strace audit so destructive path operations and unsafe report destinations fail closed.

**Tech Stack:** Python 3.7, PyTorch 1.10.2, CUDA 11.1, pytest, strace, canonical JSON/SHA-256.

---

The workspace has no `.git` metadata. Do not initialize Git or create a
worktree. Replace commit steps with focused RED/GREEN runs and SHA-256 records.
Never modify, chmod, rename, unlink, or overwrite the frozen backbone, parent,
or geometry artifacts.

**Approved design:**
`docs/superpowers/specs/2026-07-20-scanrefer-rec-hierarchical-risk-controlled-reranking-design.md`

## File Map

- Modify `models/rec_selective_residual.py`: retain complete candidate
  diagnostics and named eligibility predicates for every fixed margin.
- Modify `scripts/train_scanrefer_rec_selective_residual.py`: compute fixed
  gain/label summaries, split rows before materialization, publish a v2 result
  receipt with `calibration.status="not_run"` on OOF rejection, and use
  exclusive fresh-only output publication.
- Modify `scripts/audit_rec_finetune_file_access.py`: reject unsafe report
  destinations, verify expected protected hashes, and trace destructive path
  syscalls in addition to opens.
- Modify `tests/test_rec_selective_residual.py`: selector diagnostic tests.
- Modify `tests/test_train_scanrefer_rec_selective_residual.py`: label/gain,
  orchestration, receipt, and publication tests.
- Modify `tests/test_audit_rec_finetune_file_access.py`: destructive syscall,
  protected SHA, and report-path tests.

### Task 1: Preserve Every OOF Candidate Diagnostic

**Files:**
- Modify: `tests/test_rec_selective_residual.py`
- Modify: `models/rec_selective_residual.py`

- [x] **Step 1: Write a failing no-eligible diagnostic test**

Extend the existing rejected-candidate fixture and require the selector to
retain all candidates rather than returning only four summary fields:

```python
choice = choose_selective_configuration(candidates)
assert choice["eligible"] is False
assert choice["selected"] == "baseline"
assert choice["candidate_count"] == len(candidates)
assert choice["eligible_candidate_count"] == 0
assert len(choice["candidate_diagnostics"]) == len(candidates)
record = choice["candidate_diagnostics"][0]
assert set(record["eligibility_predicates"]) == {
    "not_no_switch",
    "all_folds_nonnegative025",
    "all_folds_nonnegative050",
    "pooled_delta025_positive",
    "bootstrap025_lower_bound_positive",
    "bootstrap050_lower_bound_nonnegative",
}
assert record["failed_predicates"] == sorted(
    name for name, passed in record["eligibility_predicates"].items()
    if not passed
)
```

Also require exact baseline/proposed counts, fixes, breaks, neutral switches,
abstentions, switch rate, five fold deltas, and both bootstrap records at each
threshold. The no-switch sentinel must appear explicitly and fail only the
`not_no_switch` and strict-positive predicates implied by zero delta.

- [x] **Step 2: Run RED**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_selective_residual.py::test_configuration_selection_rejects_fold_and_cluster_regressions \
  tests/test_rec_selective_residual.py::test_no_switch_sentinel_is_diagnostic_and_cannot_win -q
```

Expected: FAIL because `candidate_diagnostics`, predicate records, and exact
effect counts are absent.

- [x] **Step 3: Expand `_candidate_diagnostics` without changing eligibility**

Keep the existing eligibility expression byte-for-byte equivalent and derive
named booleans before combining them:

```python
predicates = {
    "not_no_switch": not candidate["sentinel"],
    "all_folds_nonnegative025": all(
        fold["hits025"] >= 0 for fold in fold_deltas.values()
    ),
    "all_folds_nonnegative050": all(
        fold["hits050"] >= 0 for fold in fold_deltas.values()
    ),
    "pooled_delta025_positive": bootstrap025["delta_hits"] > 0,
    "bootstrap025_lower_bound_positive": (
        bootstrap025["lower_bound_95"] > 0
    ),
    "bootstrap050_lower_bound_nonnegative": (
        bootstrap050["lower_bound_95"] >= 0
    ),
}
eligible = all(predicates.values())
```

For each threshold, count baseline/proposed hits, fixes (`0 -> 1`), breaks
(`1 -> 0`), switched-neutral rows, kept-correct, and kept-wrong directly from
the already validated bit arrays. Do not add per-row values to the returned
diagnostic.

- [x] **Step 4: Return the complete table for both success and failure**

`choose_selective_configuration` must always add:

```python
{
    "candidate_count": len(diagnostics),
    "eligible_candidate_count": len(eligible),
    "candidate_diagnostics": diagnostics,
}
```

When an eligible record wins, copy the winning record to the top level for
backward-compatible runtime selection and retain the full table. When none is
eligible, keep `reason="no-eligible-configuration"` and `selected="baseline"`.

- [x] **Step 5: Run GREEN**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_selective_residual.py -q
```

Expected: all selector/model tests pass with unchanged chosen policy in the
eligible fixtures.

Verification (2026-07-20): `83 passed in 4.10s`.
Source SHA-256: `098462b9a19a6902544ada5cd598c5913487846a68ce0b0d29160dfd8b617821`.
Test SHA-256: `aaedbcfde5c100fd80a3df820d3f13fdc184dc222894af078fc2946e8cbf16c8`.

### Task 2: Add Fixed Gain And Label Diagnostics

**Files:**
- Modify: `tests/test_train_scanrefer_rec_selective_residual.py`
- Modify: `scripts/train_scanrefer_rec_selective_residual.py`

- [x] **Step 1: Write failing label-distribution tests**

Add a synthetic record set containing every break/neutral/fix tier for both
thresholds and same-query/different-query alternatives. Require:

```python
summary = summarize_residual_training_labels(records)
assert set(summary) == {"all", "same_query", "different_query"}
for group in summary.values():
    assert set(group) == {"0.25", "0.50"}
    assert set(group["0.25"]) == {"break", "neutral", "fix", "total"}
    assert sum(group["0.25"][name] for name in (
        "break", "neutral", "fix"
    )) == group["0.25"]["total"]
```

The helper must derive targets through `build_selective_pair_targets`; it must
not reproduce threshold logic independently.

- [x] **Step 2: Write failing gain-summary tests**

Define the immutable quantile grid:

```python
RESIDUAL_GAIN_QUANTILES = (
    0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0,
)
```

Require `summarize_oof_pair_gain(pair_gain, pair_valid)` to report valid and
positive counts, finite min/max/mean/population-standard-deviation, and
nearest-rank values for every fixed quantile. Empty positive gains must produce
`count=0` and `statistics=null`, never fabricated zeros.

- [x] **Step 3: Run RED**

Run the two new tests directly. Expected: import failure for the two helpers.

- [x] **Step 4: Implement both pure summaries**

Validate CPU `float32` gain `[N,112]`, CPU bool validity, finite valid values,
and canonical residual records. Use `torch.std(unbiased=False)` and the same
nearest-rank convention as margin construction. JSON output must contain only
Python ints/floats/lists/dicts/`None`.

- [x] **Step 5: Bind summaries into cross-fit records**

For each fixed configuration, add:

```python
{
    "configuration_index": config_index,
    "folds": [{
        "fold": held_out_fold,
        "fit_scene_count": ...,
        "fit_row_count": ...,
        "held_scene_count": ...,
        "held_row_count": ...,
        "training_labels": summarize_residual_training_labels(fit_records),
    }],
    "gain_summary": summarize_oof_pair_gain(oof_pair_gain, pair_valid),
    "oof_pair_gain_sha256": ...,
    "prediction_count": len(records),
}
```

Do not retain model parameters or per-row gains in the published JSON. Keep
the in-memory tensor only until selection and digesting finish.

- [x] **Step 6: Run GREEN**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_train_scanrefer_rec_selective_residual.py::test_cross_fit_is_scene_disjoint_complete_and_exactly_ten_epochs \
  tests/test_train_scanrefer_rec_selective_residual.py -q
```

Expected: every configuration still covers every fit row once and trains the
same number of batches.

Verification (2026-07-20): summary tests `7 passed in 1.77s`, cross-fit test
`1 passed in 5.04s`, full trainer file `59 passed in 5.67s`.
Source SHA-256: `d92ae510c724a725c1c68c51622aa921795f6537e9407479a55bacc54fe9535e`.
Test SHA-256: `653b0fd08f67f50bc0e0fbd34c9df2c386fb8c3541acfa83d8896e9e0b4a16a0`.

### Task 3: Split Before Materialization And Never Evaluate Calibration On OOF Failure

**Files:**
- Modify: `tests/test_train_scanrefer_rec_selective_residual.py`
- Modify: `scripts/train_scanrefer_rec_selective_residual.py`

- [x] **Step 1: Write a failing pre-materialization split test**

Construct joined rows with the authoritative 562-scene mapping and inject a
materializer that records every received `scan_id`. Require the rejected OOF
path to materialize exactly the 506 fit scenes and never receive any of the 56
calibration scenes.

- [x] **Step 2: Write a failing eligible-path sequencing test**

For an eligible synthetic OOF choice, require two materialization calls:
first fit rows, then calibration rows exactly once after the choice is frozen.
The selected configuration and margin must be identical before and after the
calibration call.

- [x] **Step 3: Run RED**

Run both orchestration tests. Expected: FAIL because all 36,665 rows are
currently materialized before `split_residual_records`.

- [x] **Step 4: Add an identity-only joined-row split**

Reuse `AUTHORITATIVE_SPLIT_SEED0` and its exact seed-0 scene membership. Add:

```python
def split_residual_joined_rows(joined_rows):
    # Inspect only dataset_index and scan_id to form canonical fit/calibration
    # lists. Require 506/56 scenes and 33040/3625 rows.
```

Preserve original dataset-index order. Materialize fit rows first. Only when
OOF selection is eligible may the runner materialize the calibration rows and
run the predeclared gate.

- [x] **Step 5: Run GREEN**

Run the orchestration suite. During the formal replay in Task 7, record and
reconcile the replayed fit-row digest in the new receipt. The rejected v1
receipt did not retain a prior fit-row digest, so no historical digest may be
invented; any digest disagreement within the v2 evidence chain is a hard
error, not a reason to regenerate a split.

Verification (2026-07-20): focused isolation tests `4 passed in 5.17s`; full
trainer file `60 passed in 6.16s`.
Source SHA-256: `919c0069c8b1685a81b1851c1685ecfcfee3a6e7406642672be9b1542f8c3eae`.
Test SHA-256: `f545affcc1a6e40afcc5592eb1d139ad574c72f5116c6ece9be8ed8568f2d697`.

### Task 4: Publish A Truthful V2 Diagnostic Receipt

**Files:**
- Modify: `tests/test_train_scanrefer_rec_selective_residual.py`
- Modify: `scripts/train_scanrefer_rec_selective_residual.py`

- [x] **Step 1: Write failing rejected-receipt tests**

Require the OOF-rejected result schema:

```python
assert receipt["schema"] == "rec-selective-residual-result-receipt-v2"
assert receipt["selected"] == "baseline"
assert receipt["deployable"] is False
assert receipt["validation_data_accessed"] is False
assert receipt["calibration"] == {
    "status": "not_run",
    "reason": "oof_selection_rejected",
}
assert receipt["oof"]["choice"]["eligible"] is False
assert len(receipt["oof"]["choice"]["candidate_diagnostics"]) > 0
assert receipt["oof"]["baseline"]["hits025"] > 0
assert receipt["oof"]["baseline"]["hits050"] > 0
```

Explicitly reject `observed_hits025=0`/`observed_hits050=0` placeholders that
could be mistaken for measurements.

- [x] **Step 2: Write failing receipt-tamper tests**

Add `validate_selective_residual_result_receipt`. Mutate every nested digest,
count reconciliation, policy candidate count, fold count, calibration status,
protected snapshot, and policy flag. Each mutation must fail closed.

- [x] **Step 3: Run RED**

Run the new receipt tests. Expected: FAIL on the old v1 gate schema.

- [x] **Step 4: Build the v2 receipt from measured OOF state**

The `oof` section must contain the fixed protocol, scene-fold mapping digest,
fit row/materialization digests, all configuration summaries, the full choice
table, and exact baseline hit counts. The `calibration` section is either
`not_run` or one measured immutable record; there is no all-zero substitute.
Keep `report_only=false`, `eligible_for_model_selection=true` only for OOF
train-only selection, and `validation_data_accessed=false`.

- [x] **Step 5: Run GREEN**

Run the complete residual trainer test file and strict-reload the written JSON
through the validator.

Verification (2026-07-20): receipt and tamper tests `19 passed in 3.43s`;
publication tests `2 passed in 2.44s`; combined selector/trainer suites
`161 passed in 7.35s`. Legacy v1/zero-placeholder scan returned no matches.
Model SHA-256: `d27be8220638bb4f4c4ac307ff6ca42a20072c3daa0b43f5689a05ca131d403b`.
Trainer SHA-256: `445ca413193494088fa427c938c7400695f767b35389db7d63b667e5ff449a1b`.
Selector-test SHA-256: `45555c3a4e922123e14ac7005815e8d37358898f77b7c619c7358f1d9bed6923`.
Trainer-test SHA-256: `f52203f3596c216c85b447f1fa266b18d18be9dc181414d22d0d2dfdd0f7cbca`.

### Task 5: Replace TOCTOU Publication With Fresh-Only Completion-Last Writes

**Files:**
- Modify: `tests/test_train_scanrefer_rec_selective_residual.py`
- Modify: `scripts/train_scanrefer_rec_selective_residual.py`

- [x] **Step 1: Write failing publication-race tests**

Cover pre-existing final output, symlink components, a target created after
reservation, pre-existing artifact/receipt names, interrupted artifact write,
and interrupted receipt write. Require that no existing byte is overwritten
and that a partial directory contains no completion receipt.

- [x] **Step 2: Run RED**

Run only publication tests. Expected: FAIL because the current check-then-
`os.rename` path can replace a racing destination.

- [x] **Step 3: Reserve the final directory exclusively before work**

Use `os.mkdir(output, 0o700)` with no preflight existence check. Capture the
directory device/inode and pass that reservation to publication. A crash may
leave this directory, but it remains nondeployable because the completion
receipt is absent.

- [x] **Step 4: Add exclusive file writers**

Open every final file with
`O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW`, write through the held descriptor, fsync,
`fchmod(0444)` before close, verify the pathname still resolves to the same
device/inode, then fsync the directory. Write `result-receipt.json` last.
Never call `os.replace` or overwrite-capable `os.rename`.

- [x] **Step 5: Run GREEN**

Run publication tests and assert all three protected snapshots remain exactly
equal on every success and injected-failure path.

Verification (2026-07-20): focused publication/reservation tests
`9 passed in 2.78s`; full trainer file `85 passed in 7.00s`; combined
selector/trainer suites `168 passed in 7.14s`. The overwrite-capable syscall
scan found no `os.rename`, `os.replace`, or `shutil` matches.
Trainer SHA-256: `42eb48e4e0695d4de5d82f2483acc4ee3f6e7b46dee1c3d2400044b8d9014862`.
Trainer-test SHA-256: `9b1a0c78d0df23d0ccff9e70ec31683781f30c8302e344530184f4c3bb68f157`.

### Task 6: Harden The File-Access Audit

**Files:**
- Modify: `tests/test_audit_rec_finetune_file_access.py`
- Modify: `scripts/audit_rec_finetune_file_access.py`

- [x] **Step 1: Write failing unsafe-report tests**

For residual mode, pass each protected artifact, either cache root, output
directory, trace prefix, runner-exit path, and an existing report as
`report_path`. Require configuration failure without modifying the target.

- [x] **Step 2: Write failing destructive-syscall tests**

Add synthetic successful traces for `rename`, `renameat`, `renameat2`,
`unlink`, `unlinkat`, `link`, `linkat`, `symlink`, `symlinkat`, `chmod`,
`fchmodat`, and `truncate` against each protected artifact. Require a named
violation even when no write-intent `open` exists.

- [x] **Step 3: Write failing expected-SHA tests**

Replace each protected fixture after its command is bound and require residual
audit failure for backbone, parent, and geometry digest mismatch.

- [x] **Step 4: Run RED**

Run the new audit tests. Expected: unsupported syscalls or false passes.

- [x] **Step 5: Implement fail-closed path mutation parsing**

Extend `STRACE_FILE_ACCESS_SELECTOR` and `SUPPORTED_SYSCALLS` with the exact
destructive syscall list. Normalize every source/destination path using the
same cwd/dirfd rules as opens. Successful mutation is allowed only within the
exclusively reserved residual output or explicitly bound scratch directory;
any protected/cache/source mutation is a violation. Failed calls are still
recorded but do not imply data mutation.

- [x] **Step 6: Enforce fresh safe report publication**

Apply overlap checks to every mode, not only source-gate mode. Write the report
itself with `O_EXCL|O_NOFOLLOW`, `fchmod(0444)` on the open descriptor, stable
identity verification, and parent-directory fsync. Never replace an existing
path, including on configuration-error fallback.

- [x] **Step 7: Verify protected hashes**

For residual mode compare the three stable snapshots with the hard-coded
authoritative SHA-256 values as well as mode `0444`. Record the expected and
observed values in the audit report.

- [x] **Step 8: Run GREEN**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_audit_rec_finetune_file_access.py -q
```

Expected: all existing source-gate and new residual audit tests pass.

Verification (2026-07-20): focused Task 6 matrix `72 passed in 9.46s`.
The complete collected audit file was run as four non-overlapping node-id
chunks because the tool output window drops a single run's footer after about
30 seconds: `80 + 80 + 80 + 67 = 307 passed`. No chunk failed. The audit
source contains no `os.replace`, `os.rename`, or `tempfile` API call.
Audit SHA-256: `139cf77dc954764fccddad798da9d9bfda41e91d58c5f1d666e1280b7ceac0a8`.
Audit-test SHA-256: `7c5bb45131fb332b9f8236c0e1bc5a6255717a05c28688c8de5da5dd7799a8ac`.

### Task 7: Verification And Formal Train-Only Replay

**Files:** all files above; no source changes during the formal replay.

- [ ] **Step 1: Compile changed modules**

```bash
/root/miniconda3/envs/bdetr/bin/python -m py_compile \
  models/rec_selective_residual.py \
  scripts/train_scanrefer_rec_selective_residual.py \
  scripts/audit_rec_finetune_file_access.py
```

- [ ] **Step 2: Run focused and regression suites**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_selective_residual.py \
  tests/test_train_scanrefer_rec_selective_residual.py \
  tests/test_rec_selective_residual_runtime.py \
  tests/test_rec_geometry_runtime.py \
  tests/test_audit_rec_finetune_file_access.py -q
```

Then run the complete suite. Record test counts and hashes of every changed
source/test/plan file.

- [ ] **Step 3: Snapshot frozen inputs and reserve fresh evidence paths**

Require mode `0444`, exact SHA-256, device, inode, size, mtime, and ctime for
the three frozen artifacts. Create a new random audit root; do not reuse either
prior residual audit directory.

- [ ] **Step 4: Run the unchanged numerical OOF protocol under strace**

Use the same cache paths, CUDA device, grid, seed, fold mapping, ten epochs,
batch size, and thresholds as the rejected run. The audit must finish with
`pass=true`, `violations=[]`, and `validation_data_accessed=false`.

- [ ] **Step 5: Validate and interpret the receipt**

Strict-load the v2 receipt, reconcile every margin with its fold/bootstrap
diagnostics, compare OOF gain digests and fixed-protocol metadata, and verify
the fit-row materialization digest plus protected before/after/live snapshots.
If no candidate is eligible,
preserve `selected=baseline`, leave calibration `not_run`, and use the measured
failure modes to motivate the already approved hierarchical query-first model;
do not alter or rerun this residual grid.

- [ ] **Step 6: Update the next-phase plan from train-only evidence**

Create a separate implementation plan for the hierarchical query-then-variant
reranker. Its architecture and closed hyperparameter grid remain those in the
approved design; Phase 2 diagnostics may determine which failure modes to
emphasize in reporting, but may not expand the grid using calibration or
validation evidence.
