# ScanRefer Position Subgroup Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure and seal the immutable best system's true position-alignment Unique and Multiple Top-1 metrics at strict IoU thresholds 0.25 and 0.50 without changing model selection or protected weights.

**Architecture:** Add namespaced position subgroup counters to `GroundingEvaluator`, then launch the exact frozen geometry command through a report-only runner that writes into a newly reserved directory. The runner parses exact hit/denominator lines from both logger output and captured stdout, reconciles subgroups with the authoritative totals, and seals provenance only after pre/post artifact and source checks pass.

**Tech Stack:** Python 3.7, PyTorch 1.10.2, pytest, torch distributed launch, ScanRefer, canonical JSON and SHA-256 provenance.

---

The workspace has no `.git` metadata. Do not initialize Git or create a
worktree. Replace commit steps with focused verification and SHA-256 records.
The approved roadmap is
`docs/superpowers/specs/2026-07-20-scanrefer-rec-hierarchical-risk-controlled-reranking-design.md`.

This plan covers Phase 1 only. It must not modify residual training or create
the hierarchical model.

## File Map

- Modify `src/grounding_evaluator.py`: accumulate and print exact position
  subgroup counts under tuple keys that cannot collide with semantic metrics.
- Modify `tests/test_grounding_evaluator_rec_geometry.py`: verify strict
  thresholds, geometry selection, semantic isolation, exact logging, and
  distributed merging.
- Create `scripts/run_frozen_rec_geometry_position_subgroups.py`: build the
  frozen command, reserve a fresh report directory, capture output, parse and
  reconcile metrics, bind provenance, and publish the report last.
- Create `tests/test_run_frozen_rec_geometry_position_subgroups.py`: verify
  parser, command, no-replace publication, report-only flags, provenance, and
  authoritative/non-authoritative outcomes with synthetic launches.

### Task 1: Position-Only Evaluator Counters

**Files:**
- Modify: `tests/test_grounding_evaluator_rec_geometry.py`
- Modify: `src/grounding_evaluator.py:94-169`
- Modify: `src/grounding_evaluator.py:629-756`

- [x] **Step 1: Write failing subgroup isolation tests**

Extend `_base_end_points` with scalar batch metadata:

```python
"is_unique": torch.tensor([True], device=device),
"is_hard": torch.tensor([False], device=device),
"is_view_dep": torch.tensor([True], device=device),
```

Add a helper and a test that requires the geometry-selected good box to update
position Unique/easy/view-dependent counters at both thresholds while the
semantic `unique` counter remains unchanged:

```python
def _position_group(evaluator, group, threshold):
    key = ("position_subgroup", group, threshold)
    return evaluator.dets[key], evaluator.gts[key]

def test_geometry_updates_namespaced_position_subgroups_only():
    end_points = _attach_geometry(_base_end_points())
    evaluator = _geometry_evaluator()
    evaluator.evaluate_bbox_by_pos_align(end_points, "last_")
    assert _position_group(evaluator, "unique", 0.25) == (1, 1)
    assert _position_group(evaluator, "unique", 0.50) == (1, 1)
    assert _position_group(evaluator, "multiple", 0.25) == (0, 0)
    assert evaluator.dets["unique"] == 0
    assert evaluator.gts["unique"] == pytest.approx(1e-14)
```

Add a second row/evaluator case with `is_unique=False`, `is_hard=True`, and
`is_view_dep=False`; require Multiple/hard/view-independent totals and require
Unique plus their mutually exclusive companions to remain zero.

- [x] **Step 2: Run RED for missing tuple keys**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_grounding_evaluator_rec_geometry.py \
  -k 'position_subgroup' -q
```

Expected: FAIL with `KeyError` for `("position_subgroup", ...)`.

- [x] **Step 3: Implement namespaced counters and exact output**

Add the fixed group order:

```python
POSITION_SUBGROUPS = (
    "unique", "multiple", "easy", "hard",
    "view_dependent", "view_independent",
)
```

Initialize integer counters for every configured threshold in `reset`. Add a
private recorder that maps the three metadata bits to exactly three group
members and increments one expression-level hit/total per threshold:

```python
def _record_position_subgroups(self, end_points, bid, threshold, found):
    if found.numel() != 1:
        raise ValueError("position subgroup reporting requires one root")
    groups = (
        "unique" if bool(end_points["is_unique"][bid]) else "multiple",
        "hard" if bool(end_points["is_hard"][bid]) else "easy",
        "view_dependent" if bool(end_points["is_view_dep"][bid])
        else "view_independent",
    )
    hit = int(bool(found[0].item()))
    for group in groups:
        key = ("position_subgroup", group, threshold)
        self.dets[key] += hit
        self.gts[key] += 1
```

Call it only after the existing `bbs` update when `prefix == "last_"`,
`k == 1`, and `self.only_root`. In `print_stats`, emit one unambiguous line for
each group and threshold:

```python
"position subgroup {} Acc{:.2f}: hits={}, total={}, accuracy={:.12f}"
```

Do not modify the old semantic or mask keys.

- [x] **Step 4: Add strict-boundary and geometry-override tests**

Monkeypatch `_iou3d_par` to return an exact tensor containing 0.25 and 0.50 in
two parameterized cases. Require both the total `bbs` counter and the relevant
position subgroup hit counter to remain zero because the evaluator uses `>`.

Create two otherwise identical endpoints whose geometry score selects either
the good or bad geometry box. Require the position subgroup hit to change and
the semantic subgroup result to remain identical.

- [x] **Step 5: Add logging and distributed-merge tests**

Use a logger that stores messages. Seed exact integer tuple counters, call
`print_stats`, and require each line to include exact hits, total, and the
12-decimal ratio. Monkeypatch `misc.all_gather` with two process dictionaries,
call `synchronize_between_processes`, and require tuple hits/totals to sum.

- [x] **Step 6: Run GREEN and the neighboring evaluator suite**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_grounding_evaluator_rec_geometry.py \
  tests/test_grounding_evaluator_rec_reranker.py \
  tests/test_grounding_evaluator_source_choice.py -q
```

Expected: all tests pass with no warning or collection error.

### Task 2: Exact Report Parser And Frozen Command

**Files:**
- Create: `tests/test_run_frozen_rec_geometry_position_subgroups.py`
- Create: `scripts/run_frozen_rec_geometry_position_subgroups.py`

- [x] **Step 1: Write failing parser and command tests**

Require `parse_position_subgroups(log_text, stdout_text)` to parse exactly the
six groups at both thresholds from both renderings and return integer
`hits`, `total`, full-precision `accuracy`, and a five-decimal token. Tests
must reject a duplicate line, a missing group, a log/stdout mismatch,
`hits > total`, a rendered ratio inconsistent with the exact counts, and
Unique/Multiple totals or hits inconsistent with overall `5542/4621`.

Require `build_report_command(run_root)` to equal
`run_frozen_rec_geometry_official.build_authoritative_command()` except for
the fresh `--log_dir`, experiment name
`epoch71_geometry_position_subgroups`, and report master port `29673`.

- [x] **Step 2: Run RED for the missing report module**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_run_frozen_rec_geometry_position_subgroups.py -q
```

Expected: collection fails because
`scripts.run_frozen_rec_geometry_position_subgroups` does not exist.

- [x] **Step 3: Implement strict parsing and reconciliation**

Define constants for sample count 9,508, authoritative hits 5,542/4,621,
group names, thresholds, experiment, port, and report schema. Use an anchored
regular expression for the exact evaluator line. Require exactly one match per
group/threshold in each rendering and exact equality between renderings.

Implement:

```python
def validate_subgroup_reconciliation(subgroups, totals):
    for threshold, total_name in ((0.25, "hits025"), (0.50, "hits050")):
        unique = subgroups[("unique", threshold)]
        multiple = subgroups[("multiple", threshold)]
        if unique["total"] + multiple["total"] != 9508:
            raise ValueError("position subgroup denominators do not reconcile")
        if unique["hits"] + multiple["hits"] != totals[total_name]:
            raise ValueError("position subgroup hits do not reconcile")
```

The parser verifies the 12-decimal accuracy token against
`hits / float(total)` and derives the five-decimal token from exact integers.

- [x] **Step 4: Implement command construction and contract checks**

Copy the official argv, replace only the three approved values, and compare
all other positions against the original argv. Reuse the official fixed
interpreter, artifact paths, environment builder, stable snapshots, code-tree
manifest, exclusive JSON writer, and preflight validation rather than creating
alternative inference logic.

- [x] **Step 5: Run parser/command GREEN**

Run the focused test file. Expected: parser and command tests pass; synthetic
launcher tests remain deselected until Task 3.

### Task 3: Fresh-Only Report Launch And Sealing

**Files:**
- Modify: `tests/test_run_frozen_rec_geometry_position_subgroups.py`
- Modify: `scripts/run_frozen_rec_geometry_position_subgroups.py`

- [x] **Step 1: Write failing synthetic launch tests**

Bind temporary artifacts and a synthetic `subprocess.run` that creates one
timestamp directory, `config.json`, `log.txt`, and captured stdout. Require:

```python
record["report_only"] is True
record["eligible_for_model_selection"] is False
record["selection_uses_validation"] is False
record["inference_uses_ground_truth"] is False
record["authoritative"] is True
record["overall"]["hits025"] == 5542
record["overall"]["hits050"] == 4621
```

Also require the report file to be canonical JSON, mode `0444`, created last,
and bound to stdout/log/config/source/design/protected snapshots. Add failure
tests for an existing output path, nonzero subprocess, multiple timestamp
runs, protected artifact mutation, source mutation, and total counts that do
not reproduce 5,542/4,621. A total mismatch is sealed with
`authoritative=false`; provenance mutation is a hard error with no completion
report.

- [x] **Step 2: Run RED for the missing launch API**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_run_frozen_rec_geometry_position_subgroups.py \
  -k 'launch or publication or authoritative' -q
```

Expected: FAIL because `run_position_subgroup_report` and report validation are
not implemented.

- [x] **Step 3: Implement exclusive reservation and evidence capture**

`reserve_report_directory` creates the fixed report parent after rejecting
symlink components and then calls `tempfile.mkdtemp` with prefix
`position_subgroups.`. Print the resolved directory before starting the child.
Inside it, create `evaluation_stdout.log` with `O_CREAT|O_EXCL|O_NOFOLLOW`.
Run the child with `stderr=STDOUT`, `cwd` bound to the code root, world size 1,
CUDA 0, OMP 1, and the canonical Python path.

Take pre/post stable snapshots of the backbone, parent, and geometry artifacts
including SHA, mode, and five-field identity. Require their SHAs to equal the
three approved constants, their modes to equal `0444`, and their complete
snapshots to be unchanged. Snapshot the interpreter, code tree, and approved
design before launch and require them unchanged after launch.

- [x] **Step 4: Implement report validation and last-file publication**

After a zero exit, discover exactly one timestamp run, parse `log.txt` and
captured stdout, parse official totals with the existing official parser, and
validate subgroup reconciliation. Build a canonical record containing:

```python
{
    "schema": "rec-geometry-position-subgroup-report-v1",
    "version": 1,
    "report_only": True,
    "eligible_for_model_selection": False,
    "selection_uses_validation": False,
    "inference_uses_ground_truth": False,
    "authoritative": totals == {"hits025": 5542, "hits050": 4621},
    "sample_count": 9508,
    "overall": totals,
    "position_subgroups": nested_exact_counts,
    "artifacts_before": protected_before,
    "artifacts_after": protected_after,
    "code": code_manifest,
    "design": design_snapshot,
    "run": run_evidence,
}
```

Write `position_subgroup_report.json` only after all hard invariants pass,
using the exclusive canonical writer. Read it back through the stable snapshot
API, require mode `0444` and byte-for-byte canonical equality, then return it.

- [x] **Step 5: Run full runner GREEN and regression tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_run_frozen_rec_geometry_position_subgroups.py \
  tests/test_run_frozen_rec_geometry_official.py -q
```

Expected: all report tests and the unchanged official-runner suite pass.

### Task 4: Formal Frozen Evaluation And Verification

**Files:**
- Read only: the three protected `.pth` artifacts
- Create only: a fresh directory under
  `/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_position_subgroup_reports/`

- [x] **Step 1: Run the complete focused CPU verification**

Run the four evaluator/runner test files plus `py_compile` for the two modified
runtime modules. Expected: exit 0, no failed tests, no syntax errors.

- [x] **Step 2: Record pre-launch protected hashes and modes**

Run `sha256sum` and `stat` on the backbone, parent, and geometry files. Require
the approved SHAs and mode `444` before launching.

- [ ] **Step 3: Launch exactly one report-only full evaluation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/root/miniconda3/envs/bdetr/bin/python \
  scripts/run_frozen_rec_geometry_position_subgroups.py
```

Expected: one fresh report root is printed, the 9,508-expression child exits
zero, and the runner prints the canonical sealed report path. Do not retry a
completed run to seek different metrics.

- [ ] **Step 4: Validate and relay the requested metrics**

Load the sealed JSON through the runner's validator. Require
`authoritative=true`, overall counts `5542/4621`, Unique+Multiple denominator
9,508 at each threshold, Unique+Multiple hits equal the corresponding overall
hits, all report-only/no-GT flags exact, and pre/post protected snapshots
identical.

Report to the user, for each of Unique and Multiple at both thresholds, the
exact `hits/total`, full-precision accuracy, and five-decimal accuracy. Also
state the unchanged overall metrics and sealed evidence path.

- [ ] **Step 5: Run post-launch artifact verification**

Repeat `sha256sum` and `stat`; compare path, inode, mode, size, mtime, ctime,
and SHA against the sealed pre-launch snapshots. Expected: all three protected
artifacts are unchanged.
