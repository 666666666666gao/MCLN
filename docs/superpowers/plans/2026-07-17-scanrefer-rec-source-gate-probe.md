# ScanRefer REC Source-Gate Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fail-closed, train-only 306-step probe that updates only the final MCLN semantic score head to improve full-query Top-8 membership while preserving the frozen boxes, parent reranker, geometry reranker, and both calibration thresholds.

**Architecture:** Factor the exact 256-query deployable scores from the existing candidate adapter, train the final semantic classifier with a strict two-threshold eighth-negative membership loss, and reuse the authoritative train split plus existing frozen runtime for step-0/step-306 calibration. The probe publishes only a nondeployable smoke receipt and restores step 0 on any regression.

**Tech Stack:** Python 3.7, PyTorch 1.10.2, CUDA 11.1, pytest, ScanRefer, MCLN, existing REC parent/geometry runtime, strace file-access audit.

---

The directory has no Git metadata. Replace each commit checkpoint below with a
fresh focused test run plus `sha256sum` of the files changed in that task. Do
not initialize a repository or create a worktree.

## File Map

- Modify `models/rec_candidate_adapter.py`: pure full-query state and exact compacting wrapper.
- Create `models/rec_source_gate.py`: target attachment, Top-K membership loss, exact trainability/mode/optimizer contracts, and gate diagnostics.
- Create `scripts/probe_scanrefer_rec_source_gate.py`: train-only initialization, 306-step loop, calibration gate, rollback, and nondeployable receipt.
- Modify `tests/test_rec_candidate_adapter.py`: full-query/compact parity and GT-isolation tests.
- Create `tests/test_rec_source_gate.py`: loss, trainability, and frozen-output tests.
- Create `tests/test_probe_scanrefer_rec_source_gate.py`: runner, receipt, rollback, path, and tamper tests.
- Modify `tests/test_audit_rec_finetune_file_access.py`: exact source-gate command/receipt audit coverage.

### Task 1: Pure Full-Query REC State

**Files:**
- Modify: `models/rec_candidate_adapter.py`
- Modify: `tests/test_rec_candidate_adapter.py`

- [x] **Step 1: Write failing parity and target-isolation tests**

Add tests that construct deterministic synthetic `end_points` with 12 queries,
call `build_full_rec_query_state`, then call
`compact_rec_query_state(full, topk_per_source=3, max_candidates=6)`. Require:

```python
assert full["default_scores"].shape == (2, 12)
assert full["contrastive_scores"].shape == (2, 12)
assert full["boxes"].shape == (2, 12, 6)
assert full["features"].shape == (2, 12, 152)
assert set(full).isdisjoint({
    "center_label", "size_gts", "box_label_mask", "gt_masks",
    "candidate_ious", "threshold_labels",
})
assert torch.equal(compact["query_indices"], legacy["query_indices"])
assert torch.equal(compact["valid_mask"], legacy["valid_mask"])
assert torch.equal(compact["features"], legacy["features"])
assert torch.equal(compact["boxes"], legacy["boxes"])
assert torch.equal(compact["default_scores"], legacy["default_scores"])
assert torch.equal(
    compact["contrastive_scores"], legacy["contrastive_scores"]
)
```

Also inject target-only fields with sentinel tensors and prove they cannot
change any full or compact output.

- [x] **Step 2: Run RED**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_candidate_adapter.py -q
```

Expected: import failures for `build_full_rec_query_state` and
`compact_rec_query_state`.

- [x] **Step 3: Implement the full-state factoring**

Move the existing score, feature, and full-box construction into:

```python
def build_full_rec_query_state(end_points, inputs):
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": feature_names,
        "features": full_features,
        "boxes": full_boxes,
        "default_scores": default_scores,
        "contrastive_scores": contrastive_scores,
        "num_queries": num_queries,
    }

def compact_rec_query_state(full_state, topk_per_source=8,
                            max_candidates=16):
    query_indices, valid_mask = select_candidate_indices(
        full_state["default_scores"],
        full_state["contrastive_scores"],
        topk_per_source=topk_per_source,
        max_candidates=max_candidates,
    )
    # Gather the exact existing compact tensors and return the legacy schema.
```

Make `build_rec_candidate_batch` exactly:

```python
return compact_rec_query_state(
    build_full_rec_query_state(end_points, inputs),
    topk_per_source=topk_per_source,
    max_candidates=max_candidates,
)
```

Do not add target fields or change stable tie ordering.

- [x] **Step 4: Run GREEN and runtime-adjacent parity tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_candidate_adapter.py \
  tests/test_rec_reranker_runtime.py \
  tests/test_rec_geometry_runtime.py -q
```

Expected: all tests pass.

- [x] **Step 5: Record checkpoint hashes**

Run `sha256sum models/rec_candidate_adapter.py tests/test_rec_candidate_adapter.py`
and retain the output in the working log.

### Task 2: Strict Top-8 Membership Loss

**Files:**
- Create: `models/rec_source_gate.py`
- Create: `tests/test_rec_source_gate.py`

- [x] **Step 1: Write failing loss tests**

Define tests for this public API:

```python
loss, stats = compute_rec_source_gate_loss(
    default_scores,
    query_ious,
    query_valid,
    topk=8,
    thresholds=(0.25, 0.50),
    threshold_weights=(2.0, 1.0),
    margin=0.0,
    temperature=1.0,
)
```

Cover all of the following with explicit tensors:

- IoU exactly `0.25` or `0.50` is negative; `0.25001` and `0.50001` are positive.
- The cutoff is the eighth-largest valid negative score, not Top-1 or ninth.
- Invalid queries with very high scores cannot enter the cutoff.
- Fewer than eight negatives produces a differentiable zero for that row.
- No positive produces a differentiable zero and increments `no_positive_rows`.
- Reversing `s+` and `s8-` reverses the gradient directions.
- Two rows receive equal row weight even when their positive counts differ.
- NaN, nonpositive temperature, wrong shapes, empty rows, and nonliteral fixed
  threshold/weight contracts are rejected.

Require stats keys for each threshold:

```python
{
    "informative_rows", "active_violations", "no_positive_rows",
    "too_few_negative_rows", "positive_count", "mean_positive_cutoff_gap",
}
```

- [x] **Step 2: Run RED**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_source_gate.py -q
```

Expected: module import failure.

- [x] **Step 3: Implement loss and detached target attachment**

Add:

```python
def attach_full_query_targets(full_state, end_points, root_only=True):
    boxes = full_state["boxes"]
    gt_boxes = torch.cat([
        end_points["center_label"][..., :3].float(),
        end_points["size_gts"].float(),
    ], dim=-1)
    gt_mask = end_points["box_label_mask"]
    if root_only:
        gt_boxes = gt_boxes[:, :1]
        gt_mask = gt_mask[:, :1]
    return compute_query_ious(boxes, gt_boxes, gt_mask).detach()
```

Implement the exact softplus loss from the design. Use masked `topk`, row-wise
means, fixed float weights, and a graph-connected zero such as
`default_scores[query_valid].sum() * 0.0`.

- [x] **Step 4: Run GREEN and loss compatibility tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_source_gate.py tests/test_rec_reranker.py -q
```

Expected: all tests pass.

- [x] **Step 5: Record checkpoint hashes**

Run `sha256sum models/rec_source_gate.py tests/test_rec_source_gate.py`.

### Task 3: Exact Score-Head Trainability

**Files:**
- Modify: `models/rec_source_gate.py`
- Modify: `tests/test_rec_source_gate.py`

- [x] **Step 1: Write failing trainability and alias tests**

Build a production-shaped fake MCLN with six prediction heads and require the
only trainable names to be:

```python
expected = {
    name for name, _ in mcln.named_parameters()
    if name.startswith("prediction_heads.5.sem_cls_scores_head.")
}
```

Require parent and geometry parameter sets to be frozen, all modules to remain
in eval mode except the final semantic head, cross-boundary aliases to be
rejected, and one optimizer group with exact constant LR/weight decay/clip.

Run one synthetic update and assert:

```python
assert torch.equal(before_center, after_center)
assert torch.equal(before_size, after_size)
assert raw_iou_digest_before == raw_iou_digest_after
assert any(not torch.equal(before[name], after[name]) for name in expected)
assert all(torch.equal(before[name], after[name]) for name in frozen_names)
```

- [x] **Step 2: Run RED**

Run the focused test file and expect missing trainability APIs.

- [x] **Step 3: Implement exact contracts**

Add:

```python
SOURCE_GATE_TRAINABLE_PREFIX = \
    "prediction_heads.5.sem_cls_scores_head."

def configure_rec_source_gate_trainability(mcln, parent, geometry):
    for module in (mcln, parent, geometry):
        module.requires_grad_(False)
    selected = tuple(
        (name, parameter)
        for name, parameter in mcln.named_parameters()
        if name.startswith(SOURCE_GATE_TRAINABLE_PREFIX)
    )
    if not selected:
        raise ValueError("final semantic score head is missing")
    if len({id(parameter) for _, parameter in selected}) != len(selected):
        raise ValueError("source-gate parameters contain aliases")
    for _, parameter in selected:
        parameter.requires_grad_(True)
    return {"source_gate_semantic_head": selected}

def set_rec_source_gate_train_mode(mcln, parent, geometry):
    mcln.eval()
    parent.eval()
    geometry.eval()
    mcln.prediction_heads[5].sem_cls_scores_head.train()

def set_rec_source_gate_eval_mode(mcln, parent, geometry):
    mcln.eval()
    parent.eval()
    geometry.eval()

def build_rec_source_gate_optimizer(parameters, lr=1e-4,
                                    weight_decay=1e-4):
    values = tuple(parameter for _, parameter in parameters)
    return torch.optim.AdamW(
        values, lr=float(lr), weight_decay=float(weight_decay)
    )

def clip_rec_source_gate_gradients(parameters, max_norm=1.0):
    values = tuple(parameter for _, parameter in parameters)
    return torch.nn.utils.clip_grad_norm_(values, float(max_norm))
```

Reject missing prefixes, duplicate/aliased parameters, unexpected trainable
parameters, nonfinite optimizer arguments, and any gradient on frozen tensors.

- [x] **Step 4: Run GREEN plus existing fine-tune trainability tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_source_gate.py tests/test_rec_finetune.py -q
```

Expected: all tests pass and the old joint fine-tune contract remains unchanged.

### Task 4: Source-Gate Calibration Diagnostics

**Files:**
- Modify: `models/rec_source_gate.py`
- Modify: `tests/test_rec_source_gate.py`

- [x] **Step 1: Write failing accumulator and digest tests**

Feed uneven batches with dataset indices, full IoUs, source scores, compact
candidate indices, parent-candidate IoUs, geometry-candidate IoUs, and final
selected IoUs. Require exact counts for:

- raw-query oracle;
- default Top-8 membership oracle;
- contrastive Top-8 membership oracle;
- union Top-16 membership and candidate oracle;
- parent and geometry candidate oracles;
- final default/parent/geometry Top-1;
- gained/lost transitions versus step 0;
- ordered raw-query IoU and selected-IoU SHA-256 digests.

Use strict threshold boundaries and reject duplicate, missing, reordered, or
out-of-range dataset indices.

- [x] **Step 2: Run RED**

Expected: missing accumulator APIs.

- [x] **Step 3: Implement immutable diagnostic records**

Add a small accumulator whose `finalize(expected_sample_count)` returns only
JSON primitives plus canonical SHA-256 strings. It must retain no private IoU
arrays in a published receipt.

- [x] **Step 4: Run GREEN**

Run `tests/test_rec_source_gate.py` and the existing calibration diagnostic
tests in `tests/test_rec_finetune.py`.

### Task 5: Train-Only Nondeployable Probe Runner

**Files:**
- Create: `scripts/probe_scanrefer_rec_source_gate.py`
- Create: `tests/test_probe_scanrefer_rec_source_gate.py`

- [x] **Step 1: Write failing initialization and CLI tests**

The CLI accepts exactly:

```text
--data-root
--backbone-checkpoint
--parent-reranker
--geometry-reranker
--output-dir
--device
--probe-steps (default 306, test-only 1 allowed)
```

Require three immutable inputs, a new nonoverlapping output directory,
`cuda:0`, seed 0, batch size 18, fit/calibration split `33040/3625`, and no
validation dataset/cache objects.

- [x] **Step 2: Run RED**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_probe_scanrefer_rec_source_gate.py -q
```

Expected: runner import failure.

- [x] **Step 3: Implement initialization with a source-gate loader wrapper**

Reuse `load_rec_finetune_initial_state`, `build_train_only_data`,
`build_rec_finetune_inputs`, and frozen parent/geometry runtime validation from
`scripts/train_scanrefer_rec_finetune.py`. Preserve its two workers and worker
seeding, but use a source-gate-only loader factory with `pin_memory=False` for
both fit and calibration after the strace diagnosis. Bind both live loader
settings in the source-gate data contract and receipt. Do not call the legacy
joint optimizer or publication functions, and do not change the legacy loader
defaults. Capture initial code hashes before model/data creation.

- [x] **Step 4: Write failing step-loop and rollback tests**

Inject synthetic model, loaders, calibration, optimizer, clock, and writer.
Require step-0 calibration, exactly `probe_steps` updates, final calibration,
eligibility checks from the design, earliest baseline fallback, exact state
restore, and no deployable `.pth` output.

- [x] **Step 5: Implement the bounded loop**

For each fit batch:

```python
end_points = mcln(build_rec_finetune_inputs(batch))
full = build_full_rec_query_state(end_points, inputs)
query_ious = attach_full_query_targets(full, end_points, root_only=True)
loss, loss_stats = compute_rec_source_gate_loss(
    full["default_scores"], query_ious,
    torch.ones_like(query_ious, dtype=torch.bool),
)
optimizer.zero_grad()
loss.backward()
clip_rec_source_gate_gradients(parameters)
optimizer.step()
```

Calibration must use the exact frozen parent/geometry deployment path and the
new diagnostic accumulator. Save the initial state in CPU memory and restore
it on any gate failure.

- [x] **Step 6: Implement a nondeployable receipt**

Write `smoke-receipt.json` atomically with schema
`rec-source-gate-probe-receipt-v1`. Bind input hashes, code hashes, interpreter,
device, exact trainability/loss/optimizer contracts, step-0 and final metrics,
diagnostics/digests, eligibility decision, restored-state verification, runtime
duration, peak CUDA memory, and `validation_data_accessed=false`. Set mode
`0444` after strict reload. Never write a deployable checkpoint.

- [x] **Step 7: Run GREEN and publication-adjacent tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_probe_scanrefer_rec_source_gate.py \
  tests/test_train_scanrefer_rec_finetune.py \
  tests/test_rec_finetune_runtime.py -q
```

Expected: all tests pass.

### Task 6: File-Access Audit Compatibility

**Files:**
- Modify: `tests/test_audit_rec_finetune_file_access.py`

- [x] **Step 1: Add an exact source-gate runner trace test**

Create a synthetic successful trace whose exact initial argv names
`scripts/probe_scanrefer_rec_source_gate.py` and whose only output is
`smoke-receipt.json`. Pass it through the existing `mode="smoke"` audit and
require zero violations. Change one argv token, add one validation-cache open,
and add one undeclared output in separate cases; each must fail closed.

- [x] **Step 2: Run the focused audit tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_audit_rec_finetune_file_access.py -q
```

Expected: all tests pass without changing the audit implementation. If the
new exact-command test exposes an audit bug, add the smallest failing audit
unit test before changing the implementation; do not widen any allow rule.

- [x] **Step 3: Run audit and runner tests together**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_audit_rec_finetune_file_access.py \
  tests/test_probe_scanrefer_rec_source_gate.py -q
```

Expected: all tests pass.

### Task 7: Full CPU Verification And Audited One-Step GPU Smoke

**Files:**
- No new source files.

- [x] **Step 1: Compile changed modules**

Run `py_compile` on both modified models, the new runner, the audit script, and
all new tests. Expected: exit 0.

- [x] **Step 2: Run the complete CPU suite**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests
```

Expected: zero failures.

- [x] **Step 3: Run one-step probe under strace**

Use the same syscall set and syscall number `437` convention as the prior
audited smoke. Invoke the new runner with `--probe-steps 1` and a fresh output
directory outside all cache/input trees.

Loader root-cause diagnosis completed before this step:

- `workers=0, pin_memory=True`: first calibration batch in 2.129 seconds;
- `workers=2, pin_memory=False`: first calibration batch in 4.221 seconds,
  exact indices `1275..1292`, natural exit zero;
- selected fix: source-gate-only `workers=2, pin_memory=False` so augmented
  fit worker RNG remains unchanged.

- [x] **Step 4: Audit and verify smoke evidence**

Require runner exit 0, audit exit 0, zero violations, no validation paths,
receipt mode `0444`, step-0 exact `3461/3316`, raw-query digest stability,
frozen box/parent/geometry hashes, and restored/staged reproduction equality.

Evidence: `/root/autodl-tmp/source_gate_smoke1_v2_output_parent_ILywpf/result/smoke-receipt.json`
and `/root/autodl-tmp/source_gate_smoke1_v2_control_FuBYbt/audit-report.json`.
The audit covered 36,387 syscalls across 679 trace files with zero violations
or uncertain paths. Step 1 regressed geometry Top-1 from `3461/3316` to
`3458/3313`; step 0 was restored bitwise and reproduced as `3461/3316`.

### Task 8: Audited 306-Step Decision Probe

**Files:**
- No source edits after the one-step audit.

- [x] **Step 1: Freeze the code manifest and command**

Record SHA-256 for every project Python file and PointNet2 shared object used by
the runner. Construct one exact `--probe-steps 306` command with a fresh output
directory and bind it into the audit invocation.

Frozen evidence is under
`/root/autodl-tmp/source_gate_probe306_v2_control_H09cM4`: the 79-entry code
manifest digest is `992c37dd6200366bd5d730c0f24a4cd679d273827bcc272843d419d08fb78416`,
and `run-contract.json` binds the exact runner argv, environment, explicit
supported strace syscall set, fresh output directory, and runtime scratch
directory. The preserved first attempt under
`/root/autodl-tmp/source_gate_probe306_control_SamlYz` used the broader
`%file` selector; its runner and rollback succeeded, but the audit correctly
failed closed on the extra `readlink` syscall, so that trace is not accepted
as Task 8 evidence.

- [x] **Step 2: Run the audited probe**

Run to completion without accessing validation and without modifying source.
Do not leave the strace or runner session active at turn end.

- [x] **Step 3: Apply the design gate**

If either final geometry threshold, either candidate oracle, or the raw-query
digest regresses, retain step 0 and reject the source-gate hypothesis. If all
gates pass and at least one `0.25` membership/oracle/Top-1 count improves,
preserve the read-only nondeployable receipt and write a separate production
artifact design before any full run or official validation.

- [x] **Step 4: Update the active project plan**

Record exact step-0/final hit counts, gained/lost transitions, loss coverage,
runtime, audit counts, and the next evidence-driven action. Do not mark the
project goal complete unless a later 9,508-sample official run proves both
`0.60000/0.47000` gates.

The accepted v2 runner exited zero after `3,029.287` seconds. Its sole output
is the read-only nondeployable receipt at
`/root/autodl-tmp/source_gate_probe306_v2_output_parent_NxpjRw/result/smoke-receipt.json`
(`a57e34b356bf1bc04afdf8a968bc474a7f8e52a54811f8f263c830f302322f2c`).
The read-only audit report is
`/root/autodl-tmp/source_gate_probe306_v2_control_H09cM4/audit-report.json`
(`52034d20e1e8d9fe9b75ca62d113063c2142652baf1a60481e6a36f24e7d3f53`):
51,637 syscalls across 679 trace files, zero violations, zero uncertain paths,
zero denied paths, and zero data-root allow misses.

Step 0 versus step 306 was:

| Diagnostic | Step 0 | Step 306 | Gained/lost at 0.25 | Gained/lost at 0.50 |
| --- | ---: | ---: | ---: | ---: |
| Default Top-8 membership | `3578/3481` | `3573/3479` | `2/7` | `5/7` |
| Parent-candidate oracle | `3599/3521` | `3595/3518` | `2/6` | `2/5` |
| Geometry-candidate oracle | `3606/3588` | `3603/3585` | `2/5` | `2/5` |
| Geometry Top-1 | `3461/3316` | `3454/3297` | `6/13` | `28/47` |

The raw 256-query oracle stayed exactly `3615/3559` with ordered digest
`7a75f033a7afb2b1871e971b2797e544bac8bb59e6a20aa68735eb23842d5751`.
All 306 updates were informative: 10,760 total informative threshold rows,
with 74/193 active violations, mean loss `1.123207`, and mean gradient norm
`0.608414`. The loss therefore optimized a populated objective, but moved the
fixed Top-8 source gate in the wrong direction.

The design gate rejected step 306 and selected step 0. Restore was bitwise
verified and the selected state reproduced exactly as geometry `3461/3316`;
the three immutable inputs retained their original identities, modes, and
SHA-256 values. No checkpoint was written. The source-gate hypothesis is
closed: do not create a deployable artifact or access official validation from
this result. The next action is a separate design review of the existing
joint REC fine-tune fallback using only these train-calibration diagnostics.
