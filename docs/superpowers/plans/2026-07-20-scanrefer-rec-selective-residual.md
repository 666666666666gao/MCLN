# ScanRefer REC Selective Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and gate a scene-cross-fitted selective residual reranker that preserves the frozen geometry policy by default and advances to one official ScanRefer evaluation only after train-only evidence supports `Acc@0.25 >= 0.60000` and `Acc@0.50 >= 0.47000`.

**Architecture:** A pure residual module builds 185D alternative-versus-baseline features from the existing frozen 112-candidate geometry state, predicts threshold-specific break/neutral/fix probabilities, and promotes an alternative only above one out-of-fold calibrated margin. A standalone runner uses five scene-disjoint folds inside the 506 fit scenes, selects without the 56 calibration scenes, refits once, then requires cache calibration hits of `3524/3315` followed by online deterministic calibration hits of `3524/3316` before any runtime integration or official launch.

**Tech Stack:** Python 3.7, PyTorch 1.10.2, CUDA 11.1, pytest, ScanRefer, existing REC candidate/geometry caches, scene-clustered bootstrap.

---

The workspace has no `.git` metadata. Do not initialize Git or create a
worktree. Replace commit steps with focused tests plus SHA-256 records. Never
modify the three read-only baseline artifacts.

**Approved design:** `docs/superpowers/specs/2026-07-20-scanrefer-rec-selective-residual-design.md`

## File Map

- Create `models/rec_selective_residual.py`: feature, target, model, loss,
  policy, fold, bootstrap, gate, and artifact contracts.
- Create `scripts/train_scanrefer_rec_selective_residual.py`: strict train-only
  cache loading, frozen decision materialization, cross-fit grid, refit,
  cache calibration gate, staged artifact publication, and receipt.
- Create `tests/test_rec_selective_residual.py`: pure model/policy/statistical
  contracts.
- Create `tests/test_train_scanrefer_rec_selective_residual.py`: runner,
  boundary, rollback, reproducibility, and artifact tests.
- Modify `main_utils.py`: optional residual checkpoint and enable flag.
- Modify `train_dist_mod.py`: stable-load and apply the residual after the
  frozen geometry scorer while preserving the disabled path exactly.
- Modify `tests/test_rec_geometry_runtime.py`: disabled parity and enabled
  selective promotion tests.
- Create `scripts/run_frozen_rec_selective_residual_official.py`: immutable
  one-shot launcher and result sealer for a calibration-approved artifact.
- Create `tests/test_run_frozen_rec_selective_residual_official.py`: command,
  binding, no-GT, one-shot, metric, and preservation tests.
- Modify `scripts/audit_rec_finetune_file_access.py` and
  `tests/test_audit_rec_finetune_file_access.py`: residual train-only path
  policy and receipt coverage.

### Task 1: Pair Features, Targets, And Baseline-Exact Policy

**Files:**
- Create: `models/rec_selective_residual.py`
- Create: `tests/test_rec_selective_residual.py`

- [ ] **Step 1: Write failing shape, target, and GT-isolation tests**

Add tests using `[B,112,179]` normalized frozen geometry features and exact
frozen scorer outputs. Require these public functions:

```python
pair = build_selective_pair_features(
    normalized_features=features,
    valid_mask=valid,
    baseline_indices=baseline,
    parent_rank=parent_rank,
    geometry_rank=geometry_rank,
    threshold_logits=threshold_logits,
    iou_estimate=iou_estimate,
    query_positions=query_positions,
)
assert pair["features"].shape == (2, 112, 185)
assert pair["valid_mask"].shape == (2, 112)
assert not pair["valid_mask"][0, baseline[0]]

targets = build_selective_pair_targets(
    candidate_ious, valid, baseline, thresholds=(0.25, 0.50)
)
assert targets.shape == (2, 112, 2)
assert set(targets.unique().tolist()) <= {0, 1, 2}
# class 0=break, 1=neutral, 2=fix
```

Inject `center_label`, `size_gts`, and sentinel target fields into unrelated
containers and prove pair features do not accept or inspect them. Test strict
boundaries: IoU equal to `0.25/0.50` is a miss; one ULP above is a hit.

- [ ] **Step 2: Run RED**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_selective_residual.py -q
```

Expected: import failure for `models.rec_selective_residual`.

- [ ] **Step 3: Implement validation and the 185D pair builder**

Define exact constants and build features without targets:

```python
PAIR_FEATURE_DIM = 185
RESIDUAL_THRESHOLDS = (0.25, 0.50)
RESIDUAL_HEAD_WEIGHTS = (2.0, 1.0)
RESIDUAL_CLASS_NAMES = ("break", "neutral", "fix")

def build_selective_pair_features(
        normalized_features, valid_mask, baseline_indices,
        parent_rank, geometry_rank, threshold_logits, iou_estimate,
        query_positions):
    batch_size, candidate_count, feature_dim = normalized_features.shape
    rows = torch.arange(batch_size, device=normalized_features.device)
    baseline_features = normalized_features[rows, baseline_indices]
    feature_delta = normalized_features - baseline_features.unsqueeze(1)
    parent_delta = parent_rank - parent_rank[rows, baseline_indices].unsqueeze(1)
    geometry_delta = (
        geometry_rank - geometry_rank[rows, baseline_indices].unsqueeze(1)
    )
    threshold_probability = threshold_logits.sigmoid()
    threshold_delta = (
        threshold_probability
        - threshold_probability[rows, baseline_indices].unsqueeze(1)
    )
    iou_delta = (
        iou_estimate - iou_estimate[rows, baseline_indices].unsqueeze(1)
    )
    same_query = query_positions.eq(
        query_positions[rows, baseline_indices].unsqueeze(1)
    ).to(normalized_features.dtype)
    pair_features = torch.cat([
        feature_delta,
        parent_delta.unsqueeze(-1),
        geometry_delta.unsqueeze(-1),
        threshold_delta,
        iou_delta.unsqueeze(-1),
        same_query.unsqueeze(-1),
    ], dim=-1)
    positions = torch.arange(candidate_count, device=valid_mask.device)
    pair_valid = valid_mask & positions.unsqueeze(0).ne(
        baseline_indices.unsqueeze(1)
    )
    pair_features = torch.where(
        pair_valid.unsqueeze(-1), pair_features,
        torch.zeros_like(pair_features),
    )
    assert feature_dim == 179 and pair_features.shape[-1] == 185
    return {
        "features": pair_features,
        "valid_mask": pair_valid,
        "baseline_indices": baseline_indices,
    }
```

Validate exact float32/bool/int64 dtypes, shared device, finite valid values,
one valid baseline per row, and shapes `[B,C,179]`, `[B,C]`, `[B,C,2]`.
Return only `features`, `valid_mask`, and `baseline_indices`.

- [ ] **Step 4: Implement detached signed tier targets**

Use the mapping below independently at each threshold:

```python
baseline_hit = (
    candidate_ious.gather(1, baseline_indices[:, None]).squeeze(1) > threshold
)
alternative_hit = candidate_ious > threshold
target = torch.ones_like(candidate_ious, dtype=torch.long)  # neutral
target[baseline_hit.unsqueeze(1) & ~alternative_hit] = 0    # break
target[~baseline_hit.unsqueeze(1) & alternative_hit] = 2    # fix
return torch.stack(targets, dim=-1).detach()
```

Reject non-finite/out-of-range IoUs and ensure invalid/baseline entries remain
neutral but are excluded by the pair mask.

- [ ] **Step 5: Implement the residual model and row-balanced loss**

Expose:

```python
class SelectiveResidualModel(torch.nn.Module):
    def __init__(self, input_dim=185, hidden_dim=64, dropout=0.1):
        super().__init__()
        if hidden_dim == 0:
            self.encoder = torch.nn.Identity()
            width = input_dim
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            width = hidden_dim
        self.head = torch.nn.Linear(width, 6)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, pair_features, pair_valid):
        # Return logits [B,C,2,3], zero-filled for invalid entries.
        encoded = self.encoder(pair_features)
        logits = self.head(encoded).reshape(
            pair_features.shape[0], pair_features.shape[1], 2, 3
        )
        return logits.masked_fill(~pair_valid[:, :, None, None], 0.0)

def compute_selective_residual_loss(
        logits, targets, pair_valid, break_cost,
        threshold_weights=(2.0, 1.0)):
    class_weights = logits.new_tensor([break_cost, 1.0, 1.0])
    informative = pair_valid.any(dim=1)
    if not bool(informative.any().item()):
        zero = logits.sum() * 0.0
        return zero, {
            "informative_rows": pair_valid.sum().detach(),
            "break025": pair_valid.sum().detach(),
            "neutral025": pair_valid.sum().detach(),
            "fix025": pair_valid.sum().detach(),
            "break050": pair_valid.sum().detach(),
            "neutral050": pair_valid.sum().detach(),
            "fix050": pair_valid.sum().detach(),
        }
    head_losses = []
    denominator = pair_valid.sum(dim=1).clamp(min=1).to(logits.dtype)
    for head_index in range(2):
        values = torch.nn.functional.cross_entropy(
            logits[:, :, head_index, :].reshape(-1, 3),
            targets[:, :, head_index].reshape(-1),
            weight=class_weights,
            reduction="none",
        ).reshape_as(pair_valid)
        row_loss = (
            values * pair_valid.to(values.dtype)
        ).sum(dim=1) / denominator
        head_losses.append(row_loss[informative].mean())
    loss = (
        threshold_weights[0] * head_losses[0]
        + threshold_weights[1] * head_losses[1]
    ) / sum(threshold_weights)
    stats = {"informative_rows": informative.sum().detach()}
    for head_index, suffix in enumerate(("025", "050")):
        for class_index, name in enumerate(("break", "neutral", "fix")):
            stats[name + suffix] = (
                pair_valid & targets[:, :, head_index].eq(class_index)
            ).sum().detach()
    return loss, stats
```

`hidden_dim=0` creates one `Linear(185,6)`; `hidden_dim=64` creates the approved
two-layer model. Zero-initialize the final layer. Apply class weights
`[break_cost, 1.0, 1.0]` and return a differentiable zero when no alternative
is valid. Publish per-head break/neutral/fix counts and finite loss values.

- [ ] **Step 6: Implement expected gain and deterministic promotion**

```python
def expected_selective_gain(logits):
    probabilities = logits.softmax(dim=-1)
    signed = probabilities[:, :, :, 2] - probabilities[:, :, :, 0]
    return 2.0 * signed[:, :, 0] + signed[:, :, 1]

def apply_selective_policy(base_scores, pair_gain, pair_valid, margin):
    valid = torch.isfinite(base_scores)
    baseline_indices = base_scores.argmax(dim=1)
    candidate_gain = pair_gain.masked_fill(~pair_valid, -float("inf"))
    best_gain, selected_indices = candidate_gain.max(dim=1)
    switch_mask = pair_valid.any(dim=1) & (best_gain > 0.0) & (
        best_gain >= float(margin)
    )
    selected_indices = torch.where(
        switch_mask, selected_indices, baseline_indices
    )
    scores = base_scores.clone()
    positive_infinity = torch.full_like(scores[:, 0], float("inf"))
    promoted = torch.nextafter(
        base_scores.masked_fill(~valid, -float("inf")).max(dim=1).values,
        positive_infinity,
    )
    rows = switch_mask.nonzero(as_tuple=False).reshape(-1)
    scores[rows, selected_indices[rows]] = promoted[rows]
    return {
        "scores": scores,
        "selected_indices": selected_indices,
        "switch_mask": switch_mask,
        "baseline_indices": baseline_indices,
    }
```

Return `scores`, `selected_indices`, `switch_mask`, and `baseline_indices`.
The no-switch output scores must be bitwise equal to `base_scores`. Promotion
must preserve every non-selected candidate's relative order and invalid `-inf`.

- [ ] **Step 7: Run GREEN and record hashes**

Run the focused test and:

```bash
sha256sum models/rec_selective_residual.py \
  tests/test_rec_selective_residual.py
```

### Task 2: Scene Folds, Cluster Bootstrap, And Gate Selection

**Files:**
- Modify: `models/rec_selective_residual.py`
- Modify: `tests/test_rec_selective_residual.py`

- [ ] **Step 1: Write failing exact-fold tests**

Create 17 synthetic scenes with uneven row counts. Require:

```python
mapping = build_residual_scene_folds(scan_ids, fold_count=5, seed=0)
assert set(mapping) == set(scan_ids)
assert set(mapping.values()) == set(range(5))
assert mapping == build_residual_scene_folds(scan_ids, 5, 0)
assert canonical_scene_fold_sha256(mapping) == expected_sha
```

Prove rows from one scene never cross folds and input row order cannot change
the mapping.

- [ ] **Step 2: Write failing bootstrap and gate tests**

Use paired baseline/proposed hit bits from six scenes. Require seed-0 repeated
results to be exact, row bootstrap to be impossible through the API, and this
selection behavior:

```python
choice = choose_selective_configuration(candidates)
assert choice["eligible"] is True
assert choice["delta_hits025"] > 0
assert choice["fold_deltas"]["0"]["hits050"] >= 0
assert choice["margin_percentile"] == 95.0
```

Test rejection when any fold regresses, either clustered lower bound is
negative, the `0.25` lower bound is not strictly positive for the final choice,
or a non-contract percentile/configuration appears.

- [ ] **Step 3: Implement exact fold and bootstrap contracts**

Use `sorted(set(scan_ids))`, `random.Random(0).shuffle`, and position modulo
five. Implement 10,000 resamples of whole scenes with replacement using a
private `random.Random(0)`. Report exact hit deltas and one-sided 95% lower
bounds as the fifth percentile using the deterministic nearest-rank rule.

- [ ] **Step 4: Implement the closed configuration grid and tie policy**

Define immutable grids:

```python
RESIDUAL_HIDDEN_DIMS = (0, 64)
RESIDUAL_WEIGHT_DECAYS = (1e-4, 1e-3)
RESIDUAL_BREAK_COSTS = (2.0, 4.0, 8.0)
RESIDUAL_MARGIN_PERCENTILES = (50.0, 60.0, 70.0, 80.0, 90.0,
                               95.0, 97.5, 99.0)
```

Choose only from these values. Rank eligible policies by
`2*delta_hits025 + delta_hits050`, then larger margin, fewer switches, linear
model, larger weight decay, larger break cost. A no-switch candidate is a
diagnostic baseline but cannot win because the selected policy requires a
strictly positive clustered `0.25` lower bound.

- [ ] **Step 5: Run focused tests and record hashes**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_selective_residual.py -q
sha256sum models/rec_selective_residual.py \
  tests/test_rec_selective_residual.py
```

### Task 3: Exact Frozen Decision Materialization

**Files:**
- Create: `scripts/train_scanrefer_rec_selective_residual.py`
- Create: `tests/test_train_scanrefer_rec_selective_residual.py`

- [ ] **Step 1: Write failing materialization parity tests**

Build synthetic joined base/geometry rows plus recording frozen parent and
geometry models. Require `materialize_residual_rows` to return canonical CPU
records with:

```python
{
    "dataset_index": int,
    "scan_id": str,
    "target_id": int,
    "pair_features": FloatTensor[112,185],
    "pair_valid": BoolTensor[112],
    "candidate_ious": FloatTensor[112],
    "baseline_index": int,
    "baseline_scores": FloatTensor[112],
    "query_positions": LongTensor[112],
    "variant_indices": LongTensor[112],
}
```

Assert the baseline indices and scores exactly match
the `geometry_weight=1.0` branch of `evaluate_geometry_blends`, all frozen
models remain in eval/no-grad state, and modifying target IoUs changes targets but cannot
change any pair feature or baseline selection.

- [ ] **Step 2: Run RED**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_train_scanrefer_rec_selective_residual.py -q
```

Expected: import failure for the new runner.

- [ ] **Step 3: Implement strict train-only input loading**

Reuse only these public/established helpers:

```python
joined, base_manifest, geometry_manifest, parent = \
    load_geometry_training_data(base_cache, geometry_cache, parent_artifact)
geometry_model, geometry_artifact = load_geometry_reranker_artifact(
    geometry_artifact_path, device="cuda:0", parent_artifact_path=parent_path,
    base_manifest=base_manifest, geometry_manifest=geometry_manifest)
materialize_parent_scores(joined, parent, device="cuda:0",
                          local_batch_size=12)
```

Reject any path component containing `val`, `validation`, `official`, `claim`,
or `receipt`; require manifest split exactly `train`; bind the known base,
geometry, parent, and backbone digests. Do not accept caller-provided expected
digests.

- [ ] **Step 4: Implement streamed frozen feature materialization**

For batches of 256 joined rows, call `build_geometry_training_batch`, normalize
with the immutable geometry artifact statistics, run the frozen geometry model
under `no_grad` and disabled autocast, rebuild the exact parent/learned ranks,
apply the artifact's `geometry_weight`, find the stable baseline Top-1, and call
`build_selective_pair_features`. Copy only the declared record to CPU. Bind an
ordered SHA-256 over row identity, baseline index/scores, pair features, and
candidate IoUs.

- [ ] **Step 5: Add invariance and tamper tests**

Require rejection of row reorder, duplicate/missing index, identity mismatch,
wrong artifact SHA, non-float32 tensors, changed candidate schema, and a frozen
model that enters train mode or acquires gradients. Verify the baseline
selected-IoU digest equals the authoritative step-0 digest on an injected
fixture.

- [ ] **Step 6: Run GREEN and parent/geometry regressions**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_train_scanrefer_rec_selective_residual.py \
  tests/test_train_rec_geometry_reranker.py \
  tests/test_rec_geometry_runtime.py -q
```

### Task 4: Five-Fold Trainer, Fixed Calibration Gate, And Artifact

**Files:**
- Modify: `scripts/train_scanrefer_rec_selective_residual.py`
- Modify: `models/rec_selective_residual.py`
- Modify: `tests/test_train_scanrefer_rec_selective_residual.py`

- [ ] **Step 1: Write failing cross-fit tests**

Use a separable synthetic 15-scene fixture. Spy on every optimizer batch and
assert the held-out fold's scene IDs never enter fitting, all 12 approved grid
configurations produce one OOF prediction per fit row, training is exactly ten
epochs, and configuration/gate selection sees only pooled OOF records.

- [ ] **Step 2: Write failing calibration and rollback tests**

Exercise exact boundary cases:

```python
assert calibration_gate({"hits025": 3524, "hits050": 3315}, baseline).passed
assert not calibration_gate({"hits025": 3523, "hits050": 3315}, baseline).passed
assert not calibration_gate({"hits025": 3524, "hits050": 3314}, baseline).passed
```

Require no artifact on a failed gate, a read-only nondeployable receipt with
`selected="baseline"`, and unchanged SHA/mode/inode evidence for all three
baseline inputs. A successful synthetic run publishes one staged artifact with
`deployable=False` and retains the baseline files.

- [ ] **Step 3: Implement deterministic cross-fit training**

For each grid configuration and fold, initialize seed 0, train ten epochs with
AdamW (`lr=3e-4`, batch 256, clip 1.0), and store only held-out pair gains.
Apply the fixed percentile grid and `choose_selective_configuration`. Refit the
winning configuration on all 506 fit scenes for ten epochs; do not use
calibration during refit.

- [ ] **Step 4: Implement one-pass calibration and diagnostics**

Evaluate the refit model exactly once on the cached original 56 calibration
scenes.
Publish baseline/proposed/oracle hits, fixes/breaks, switches/abstentions,
per-scene deltas, bootstrap bounds, wrong-query/wrong-variant recoveries,
ordered selected-IoU SHA, row materialization SHA, and all fold/config records.
Require `3524/3315`, unchanged oracle counts, and unchanged raw/frozen digests.

- [ ] **Step 5: Implement strict artifact and atomic publication**

Use schema `rec-selective-residual-v1`. Store `deployable=False`, model state,
exact model/grid/gate
configuration, fold mapping/hash, OOF metrics/digest, fixed calibration record,
all immutable input SHAs, feature names, and
`validation_data_accessed=False`. Provide:

```python
save_selective_residual_artifact(path, artifact)
load_selective_residual_artifact(path, device="cpu",
                                 parent_sha256=None,
                                 geometry_sha256=None)
validate_selective_residual_artifact(
    artifact, expected_backbone_sha256, expected_parent_sha256,
    expected_geometry_sha256, expected_feature_names
)
```

Write to a fresh sibling `.building` path, fsync, rename once, reload and
compare every tensor/value. Keep this staged candidate writable only inside its
fresh experiment directory; never overwrite any path.

- [ ] **Step 6: Implement CLI and receipt**

The CLI accepts exactly the four immutable input paths, a fresh output
directory, and `--device cuda:0`. Cache/split/grid/training/gate constants are
not CLI-tunable. A failure receipt is nondeployable; a success receipt names
the artifact SHA and exact calibration evidence.

- [ ] **Step 7: Run focused tests and record hashes**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_selective_residual.py \
  tests/test_train_scanrefer_rec_selective_residual.py -q
sha256sum models/rec_selective_residual.py \
  scripts/train_scanrefer_rec_selective_residual.py \
  tests/test_rec_selective_residual.py \
  tests/test_train_scanrefer_rec_selective_residual.py
```

### Task 5: Runtime Integration Without Disabled-Path Drift

**Files:**
- Modify: `main_utils.py`
- Modify: `train_dist_mod.py`
- Modify: `tests/test_rec_geometry_runtime.py`
- Create: `tests/test_rec_selective_residual_runtime.py`

- [ ] **Step 1: Write failing CLI and disabled-parity tests**

Add `--rec_selective_residual_checkpoint` and
`--eval_use_rec_selective_residual_scores`. Require the enable flag to imply
the existing parent and geometry score flags and all three artifact paths.
When disabled, compare the complete output dict and tensors bitwise against the
current geometry runtime.

- [ ] **Step 2: Write failing enabled-runtime tests**

Inject a recording residual model and require it receives the same normalized
179D features and frozen heads used by training, never receives a GT field,
promotes only the policy-selected flat candidate, preserves remaining rank
order, and attaches no new target/diagnostic tensors to `end_points`.

- [ ] **Step 3: Implement stable residual loading and provenance**

Extend `TrainTester` with `rec_selective_residual` and its artifact. Stable-load
once, validate its backbone/parent/geometry SHAs against the already-open
artifacts, freeze/eval it, and reject partial state. Rehash the residual file
before the first batch.

- [ ] **Step 4: Extend geometry builder with optional residual context**

Add optional `residual_model=None, residual_artifact=None` parameters to
`build_rec_geometry_runtime_outputs`. After frozen geometry blending, call the
same `build_selective_pair_features`, residual forward, and
`apply_selective_policy`. Keep the exact six-field geometry output schema; only
`rec_geometry_scores` changes when a permitted switch occurs.

- [ ] **Step 5: Run runtime and evaluator regressions**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_geometry_runtime.py \
  tests/test_rec_selective_residual_runtime.py \
  tests/test_grounding_evaluator_rec_geometry.py \
  tests/test_main_utils_source_choice_checkpoint.py -q
```

### Task 6: File-Access Audit And One-Shot Official Launcher

**Files:**
- Modify: `scripts/audit_rec_finetune_file_access.py`
- Modify: `tests/test_audit_rec_finetune_file_access.py`
- Create: `scripts/run_frozen_rec_selective_residual_official.py`
- Create: `tests/test_run_frozen_rec_selective_residual_official.py`

- [ ] **Step 1: Add failing train-only audit tests**

Require the residual trainer command to contain only train caches and the new
output root. Reject any syscall touching ScanRefer validation annotations,
`/val`, `geometry_val`, prior official logs/results, claims, or receipts. Require
the receipt's opened-path digest and `validation_data_accessed=False`.

- [ ] **Step 2: Implement residual audit mode**

Reuse the explicit syscall selector:

```text
open,openat,creat,open_by_handle_at,chdir,fchdir,execve,execveat,io_uring_setup,437
```

Do not use `%file`. Bind the exact interpreter, command, process tree, trace
inventory, artifact SHAs, output allowlist, and zero-violation report.

- [ ] **Step 3: Write failing official launcher tests**

Adapt the existing geometry official tests. Require exactly 9,508 rows, gate
hits `5705/4469`, all four artifact SHAs, exact residual selection receipt,
no-GT flags (`butd=true`, `butd_gt=false`, `butd_cls=false`), one exclusive
claim, immutable stdout/log/result, and preservation of the old best weights
regardless of outcome.

- [ ] **Step 4: Implement the fixed launcher and sealer**

Build the existing authoritative geometry command plus:

```text
--rec_selective_residual_checkpoint <fixed-selected-artifact>
--eval_use_rec_selective_residual_scores
```

Expose no tuning, path, metric, or claim override. Parse each official metric
once from independent log/stdout copies, recover exact integer hits, seal the
result read-only, and set `acceptance_gate_pass` only for
`hits025 >= 5705`, `hits050 >= 4469`, sample count 9,508, and no GT.

- [ ] **Step 5: Run audit and launcher tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_audit_rec_finetune_file_access.py \
  tests/test_run_frozen_rec_selective_residual_official.py -q
```

### Task 7: Full CPU Verification Before Training

**Files:** all files above.

- [ ] **Step 1: Compile all changed Python files**

```bash
/root/miniconda3/envs/bdetr/bin/python -m py_compile \
  models/rec_selective_residual.py \
  scripts/train_scanrefer_rec_selective_residual.py \
  scripts/run_frozen_rec_selective_residual_official.py \
  scripts/audit_rec_finetune_file_access.py \
  main_utils.py train_dist_mod.py
```

- [ ] **Step 2: Run the focused REC suite**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_selective_residual.py \
  tests/test_train_scanrefer_rec_selective_residual.py \
  tests/test_rec_selective_residual_runtime.py \
  tests/test_rec_geometry_runtime.py \
  tests/test_grounding_evaluator_rec_geometry.py \
  tests/test_audit_rec_finetune_file_access.py \
  tests/test_run_frozen_rec_selective_residual_official.py -q
```

- [ ] **Step 3: Run the complete suite**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest -q
```

Expected: all existing and new tests pass. Record count, elapsed time, and
SHA-256 of every changed source/test file.

### Task 8: Audited Train-Only Cross-Fit And Calibration Gate

**Files:** no source edits during the run.

- [ ] **Step 1: Rehash and stat immutable inputs**

Require exact expected SHAs and mode `0444` for the backbone, parent, and
geometry artifacts. Abort before output creation on any mismatch.

- [ ] **Step 2: Run an audited synthetic/one-batch smoke**

Use a fresh `mktemp -d` output and trace root. Require zero access violations,
exact cache step-0 baseline `3461/3315`, exact cache digests, finite gradients,
and successful rollback/reload.

- [ ] **Step 3: Run the full five-fold train-only experiment**

Use the four fixed immutable inputs and a fresh output directory. Do not open
or enumerate validation caches. Preserve stdout, stderr, trace audit, runtime,
GPU memory, fold table, OOF digest, calibration record, and receipt.

- [ ] **Step 4: Apply the fixed gate without reinterpretation**

If cache calibration is below `3524/3315`, keep the baseline, publish only a
nondeployable failure receipt, and return to brainstorming for the hierarchical
query-first design. Do not tune from the calibration failure. If it passes,
rehash/reload the staged residual artifact and proceed to the online gate
without changing its bytes.

### Task 9: Online Train Calibration, Runtime Smoke, And Official Evaluation

**Files:** no source edits after the selected artifact is frozen.

- [ ] **Step 1: Run the authoritative online train-calibration gate**

Use the same ordered 56 training scenes as the source-gate step-0 reproduction.
Require baseline `3461/3316`, candidate `hits025 >= 3524`, candidate
`hits050 >= 3316`, identical raw-query and candidate-oracle counts/digests, and
no validation access. A failure leaves the staged artifact nondeployable and
retains the baseline.

- [ ] **Step 2: Seal the artifact and run a one-batch runtime parity smoke**

Require the cache-selected decisions and live MCLN decisions to match exactly,
boxes/oracles/frozen SHAs to remain unchanged, and inference inputs to exclude
GT. Add the online calibration evidence, set `deployable=True`, publish a new
immutable artifact rather than editing the staged artifact, chmod it `0444`,
and audit the complete process tree.

- [ ] **Step 3: Seal the claim and launch one official run**

Launch only through
`scripts/run_frozen_rec_selective_residual_official.py`. Do not inspect or tune
against validation before the artifact, command, code manifest, and claim are
frozen.

- [ ] **Step 4: Verify the actual objective**

Require authoritative evidence for:

```text
sample_count == 9508
hits025 >= 5705        # printed Acc@0.25 >= 0.60000
hits050 >= 4469        # printed Acc@0.50 >= 0.47000
inference_uses_ground_truth == false
```

If the run fails either metric, preserve both old and new artifacts, retain the
old official best, and return to a new train-only design. Do not alter or rerun
the frozen candidate based on validation rows.
