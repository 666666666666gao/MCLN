# ScanRefer REC One-Epoch Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, run, and provenance-bind one train-only ScanRefer REC fine-tuning epoch, then perform one new frozen official validation and prove `Acc@0.25 >= 0.60000` and `Acc@0.50 >= 0.47000` without inference-time ground truth.

**Architecture:** A standalone runner loads epoch-71 weights without optimizer state, trains only the exact MCLN decoder/box allowlist plus the parent and geometry rerankers, and selects among seven predetermined steps using only a scene-disjoint ScanRefer train calibration partition. New online-fine-tune artifact schemas bind the selected backbone and both rerankers without pretending that old frozen caches were regenerated; `train_dist_mod.py` dispatches either old cache-trained artifacts or the new strict artifact family into the same deployable score builders.

**Tech Stack:** Python 3.7, PyTorch 1.10.2, CUDA 11.1, pytest, ScanRefer, MCLN, existing QueryReranker and mask-geometry modules.

**Repository Constraint:** This directory has no `.git` metadata. Do not create commits or worktrees. Preserve unrelated files and record fresh test output after each task.

**Approved design:** `docs/superpowers/specs/2026-07-16-scanrefer-rec-one-epoch-finetune-design.md`

---

## File Map

- Modify `models/losses.py`: backward-compatible supervised-mask and consistency loss scales.
- Create `models/rec_finetune.py`: exact trainability/mode contract, differentiable REC forward, calibration selection, and new artifact schemas.
- Create `scripts/train_scanrefer_rec_finetune.py`: train-only dataset orchestration, weight-only initialization, one-epoch loop, atomic publication, and CLI.
- Modify `train_dist_mod.py`: strict new-artifact loader dispatch; keep old frozen artifact behavior unchanged.
- Create `scripts/run_frozen_rec_finetune_official.py`: one-shot claim/launch/seal/compare launcher for the selected fine-tuned system.
- Create `tests/test_rec_finetune.py`.
- Create `tests/test_train_scanrefer_rec_finetune.py`.
- Create `tests/test_rec_finetune_runtime.py`.
- Create `tests/test_run_frozen_rec_finetune_official.py`.

### Task 1: Parameterize the Approved Hungarian Loss Scales

**Files:**
- Modify: `models/losses.py`
- Create: `tests/test_rec_finetune.py`

- [x] **Step 1: Write a failing loss-composition test**

Monkeypatch the expensive criterion components with scalar tensors and assert
that `mask_loss_scale` multiplies main, superpoint, and adaptive supervised
mask terms while `consistency_loss_scale` multiplies only corresponding terms:

```python
loss, _ = compute_hungarian_loss(
    end_points,
    num_decoder_layers=0,
    set_criterion=fake_criterion,
    mask_loss_scale=0.1,
    consistency_loss_scale=0.1,
)
assert torch.equal(loss, expected_detector + 0.1 * supervised_mask
                   + 0.1 * corresponding_consistency)
```

Add a second assertion that omitting both keywords exactly reproduces the
current total.

- [x] **Step 2: Run RED**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_finetune.py::test_hungarian_loss_scales_are_separate_and_default_compatible -q
```

Expected: `TypeError` for the missing keyword arguments.

- [x] **Step 3: Implement the minimal backward-compatible parameters**

Extend the signature with:

```python
mask_loss_scale=1.0,
consistency_loss_scale=1.0,
```

Build named `supervised_mask_loss` and `corresponding_consistency_loss`
expressions from the existing coefficients, validate both scales as finite
non-negative numbers, and add their scaled values to the unchanged detector
and source-choice terms.

- [x] **Step 4: Run GREEN and existing loss-adjacent tests**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_finetune.py \
  tests/test_source_choice_selector.py \
  tests/test_main_utils_source_choice_checkpoint.py -q
```

Expected: all pass.

### Task 2: Exact Trainability, Mode, Optimizer, and Step Contracts

**Files:**
- Create: `models/rec_finetune.py`
- Modify: `tests/test_rec_finetune.py`

- [x] **Step 1: Write failing allowlist and frozen-mode tests**

Use a small fake MCLN with modules named like production and require:

```python
groups = configure_rec_finetune_trainability(mcln, parent, geometry)
assert groups["mcln_names"] == expected_exact_names
assert all(parameter.requires_grad for parameter in groups["mcln_parameters"])
assert all(not parameter.requires_grad for name, parameter
           in mcln.named_parameters() if name not in expected_exact_names)

set_rec_finetune_train_mode(mcln, parent, geometry)
assert mcln.training is False
assert mcln.decoder.training is True
assert mcln.backbone_net.training is False
assert mcln.text_encoder.training is False
assert mcln.x_mask.training is False
assert parent.training is True and geometry.training is True
```

- [x] **Step 2: Write failing optimizer and step-count tests**

Require one fresh AdamW with exactly three named groups and immutable values:

```python
optimizer = build_rec_finetune_optimizer(groups)
assert [(g["name"], g["lr"], g["weight_decay"]) for g in optimizer.param_groups] == [
    ("mcln_decoder_box", 2e-5, 5e-4),
    ("parent_reranker", 1e-3, 1e-4),
    ("geometry_reranker", 3e-4, 1e-4),
]
assert natural_batch_count(33040, 18) == 1836
assert calibration_steps(1836, 306) == (0, 306, 612, 918, 1224, 1530, 1836)
```

Also spy on `clip_grad_norm_` and require clip values `0.1/1.0/1.0` on the
three disjoint parameter tuples.

- [x] **Step 3: Run RED**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_finetune.py -q
```

Expected: module import failure.

- [x] **Step 4: Implement the pure contracts**

Define constants and these public functions:

```python
MCLN_TRAINABLE_PREFIXES = (
    "decoder.", "decoder_query_proj.",
    "proposal_head.", "prediction_heads.",
)
CALIBRATION_STEPS = (0, 306, 612, 918, 1224, 1530, 1836)

def configure_rec_finetune_trainability(mcln, parent, geometry): ...
def set_rec_finetune_train_mode(mcln, parent, geometry): ...
def set_rec_finetune_eval_mode(mcln, parent, geometry): ...
def build_rec_finetune_optimizer(groups): ...
def clip_rec_finetune_gradients(groups): ...
def natural_batch_count(sample_count, batch_size): ...
def calibration_steps(max_steps, interval): ...
```

Fail closed if a prefix matches no parameter, groups overlap, any unlisted
MCLN parameter remains trainable, or the computed contract differs from the
approved constants.

- [x] **Step 5: Run GREEN**

Run the focused file and expect all tests to pass.

### Task 3: Gradient-Enabled Parent and Geometry Forward

**Files:**
- Modify: `models/rec_finetune.py`
- Modify: `tests/test_rec_finetune.py`

- [x] **Step 1: Write failing GT-isolation and target-detach tests**

Construct synthetic deployable `end_points` and a separate target dictionary.
Spy on `build_rec_candidate_batch` and `build_rec_mask_geometry_candidates` so
they fail if any target field is present. Require:

```python
state = build_rec_finetune_forward(
    end_points_without_gt, inputs, targets,
    parent, parent_artifact, geometry, geometry_artifact,
)
assert state["parent_candidate_ious"].requires_grad is False
assert state["geometry_candidate_ious"].requires_grad is False
assert "center_label" not in state["parent_model_inputs"]
assert "center_label" not in state["geometry_model_inputs"]
```

- [x] **Step 2: Write failing gradient and eval-parity tests**

Backpropagate `parent_loss + geometry_loss`. Require gradients on reranker
parameters and a synthetic allowed box/query tensor, no gradient through target
IoUs, and no gradient on frozen tensors. In eval mode compare the generated
parent query scores, geometry boxes, flat scores, valid mask, and fallback
index exactly against the existing runtime builders.

- [x] **Step 3: Run RED**

Run the two named tests and confirm missing function failures.

- [x] **Step 4: Implement the differentiable forward**

Use only existing low-level production helpers:

```python
candidate_batch = build_rec_candidate_batch(...)
normalized_parent = normalize_parent_features(
    candidate_batch, parent_artifact["feature_mean"],
    parent_artifact["feature_std"])
parent_outputs = parent(normalized_parent, candidate_batch["valid_mask"])
parent_targets = attach_candidate_targets(
    candidate_batch, targets, root_only=True)
parent_loss, parent_stats = compute_rec_reranker_loss(
    parent_outputs, parent_targets["candidate_ious"].detach(),
    candidate_batch["valid_mask"])
```

Build fixed-weight deployed parent state, then geometry candidates and 179D
features before target attachment. Run the geometry reranker directly and call
the same REC loss with detached geometry IoUs. Package the final deployment
state using the existing stable rank-blend/tie helpers. Do not call either
runtime wrapper because they intentionally freeze modules and enter no-grad.

- [x] **Step 5: Run GREEN and runtime regressions**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_finetune.py \
  tests/test_rec_reranker_runtime.py \
  tests/test_rec_geometry_runtime.py -q
```

Expected: all pass.

### Task 4: Scene Split, Calibration, and Regression Stop

**Files:**
- Modify: `models/rec_finetune.py`
- Create: `tests/test_train_scanrefer_rec_finetune.py`

- [x] **Step 1: Write failing exact-split tests**

Build identities from the actual train annotation metadata without loading
point clouds. Require the exact authoritative counts and mapping SHA. Assert
fit and calibration scenes are disjoint and every annotation occurs once.

- [x] **Step 2: Write failing calibration accumulator tests**

Feed synthetic parent/geometry selected IoUs in uneven batches and require
strict `>0.25` and `>0.50` hit counts, fixed sample order, and exact sample
count. Require the acceptance-aware score formula from the design.

- [x] **Step 3: Write failing selection-state tests**

Exercise these cases:

```python
selector = CalibrationSelector(step0_metrics)
assert selector.observe(306, improves_both).action == "continue"
assert selector.observe(612, equal_score).best_step == 306
assert selector.observe(918, below_step0_at_050).action == "stop"
assert selector.best_step == 306
```

Add a separate composite-score regression case and an attempt to observe a
non-contract step.

- [x] **Step 4: Implement and run GREEN**

Reuse the exact seed-0 scene shuffling algorithm and canonical JSON hashing
from `scripts/train_rec_geometry_reranker.py`. Define a small immutable
calibration-state object that stores every observation and deep-copied CPU
snapshots only for eligible improvements. Run the focused tests.

### Task 5: New Fine-Tune Artifact Schemas and Runtime Dispatch

**Files:**
- Modify: `models/rec_finetune.py`
- Modify: `train_dist_mod.py`
- Create: `tests/test_rec_finetune_runtime.py`

- [x] **Step 1: Write failing artifact round-trip and tamper tests**

Require separate exact-schema parent and geometry payloads. Both bind the new
backbone SHA; geometry also binds the serialized parent SHA. Require explicit
initial SHA lineage, fixed normalization tensor hashes, split mapping SHA,
selected step, calibration history, optimizer/loss/allowlist contract, and
`validation_data_accessed=False`.

Mutate each binding independently and require the loader to reject it. Save,
reload, and require identical model tensors and deployment scores.

- [x] **Step 2: Write failing legacy/new dispatch tests**

Spy on loaders and assert that existing artifact versions still use
`load_parent_reranker_snapshot` plus `load_geometry_reranker_artifact`, while
the two new schema identifiers use only `load_rec_finetune_runtime_artifacts`.
Mixed old/new pairs must fail.

- [x] **Step 3: Run RED**

Run the new runtime file and confirm missing schema/dispatch failures.

- [x] **Step 4: Implement strict builders and loaders**

Define:

```python
REC_FINETUNE_PARENT_SCHEMA = "rec-finetune-parent-v2"
REC_FINETUNE_GEOMETRY_SCHEMA = "rec-finetune-geometry-v2"

def build_rec_finetune_parent_artifact(...): ...
def build_rec_finetune_geometry_artifact(...): ...
def validate_rec_finetune_artifact_pair(...): ...
def load_rec_finetune_runtime_artifacts(parent_path, geometry_path, device): ...
```

Reuse the runtime field names consumed by the existing candidate/geometry
builders, but validate the new online-training provenance separately. In
`train_dist_mod.py`, inspect stable snapshots and dispatch the artifact family
before model construction. Keep existing artifact validators byte-for-byte
compatible.

- [x] **Step 5: Run GREEN and the complete old artifact suites**

Run new runtime tests plus all parent/geometry artifact and runtime tests.

### Task 6: Train-Only Runner Initialization and Data Boundary

**Files:**
- Create: `scripts/train_scanrefer_rec_finetune.py`
- Modify: `tests/test_train_scanrefer_rec_finetune.py`

- [x] **Step 1: Write failing CLI and immutable-argument tests**

The CLI accepts only paths, device, and a `--smoke-steps` test escape hatch.
All scientific settings are constants. Reject mismatched input SHA, an output
inside any input/cache/official directory, a non-empty final output, a device
other than CUDA 0 for production, or any max step above 1,836.

- [x] **Step 2: Write failing no-validation-data test**

Monkeypatch `Joint3DDataset` and every cache loader. Run initialization and
assert the only dataset call is:

```python
Joint3DDataset(
    dataset_dict={"scanrefer": 1},
    test_dataset="scanrefer",
    split="train",
    joint_det=False,  # represented by the dataset dictionary
    butd=True,
    butd_gt=False,
    butd_cls=False,
    augment_det=True,
    ...,
)
```

No path containing `/val`, `geometry_val`, or `official_result` may be opened.
Calibration must use a train dataset view with both augmentation flags false.

- [x] **Step 3: Write failing weights-only initialization test**

Provide a synthetic checkpoint containing poisoned optimizer/scheduler states.
Require strict model weight loading, epoch 71 verification, a newly constructed
AdamW with empty state, and no scheduler. Require parent and geometry states to
match their input artifacts before the first update.

- [x] **Step 4: Implement initialization and loaders**

Reuse the exact model configuration preparation and batch-to-device helpers
from `scripts/cache_scanrefer_rec_candidates.py`. Strip `module.` once and
load the MCLN state with `strict=True`. Build fit/calibration index views from
annotation `scan_id`; fit uses shuffled deterministic sampling and
`drop_last=False`, calibration is ordered and unaugmented.

- [x] **Step 5: Run GREEN**

Run the runner test file without CUDA or real point-cloud loading.

### Task 7: One-Epoch Loop, Calibration, and Atomic Publication

**Files:**
- Modify: `scripts/train_scanrefer_rec_finetune.py`
- Modify: `tests/test_train_scanrefer_rec_finetune.py`

- [x] **Step 1: Write a failing synthetic end-to-end loop test**

Use tiny fake datasets/models and a six-step contract. Require step-0
calibration, natural remainder, optimizer step counts, calibration cadence,
first regression stop, restoration of the best snapshot, and no access to the
test sentinel.

- [x] **Step 2: Write failing publication tests**

Interrupt each atomic save and prove no final partial file appears. On success
require publication order `backbone -> parent -> geometry -> selection`, exact
SHA links, strict reload parity, and final modes `0444`. Existing inputs and
sealed official files must retain their inode identity, mode, size, and hash.

- [x] **Step 3: Implement the loop**

For every fit batch:

```python
set_rec_finetune_train_mode(mcln, parent, geometry)
optimizer.zero_grad()
end_points = mcln(inputs_without_gt)
rec_state = build_rec_finetune_forward(...)
end_points.update(target_batch)
hungarian_loss, end_points = compute_hungarian_loss(
    end_points, 6, set_criterion,
    query_points_obj_topk=4,
    source_choice_selector_loss_weight=0.0,
    mask_loss_scale=0.1,
    consistency_loss_scale=0.1,
)
total = hungarian_loss + rec_state["parent_loss"] + rec_state["geometry_loss"]
total.backward()
clip_rec_finetune_gradients(groups)
optimizer.step()
```

At each contract step, run the ordered unaugmented calibration loader under
no-grad, update the selector, save an in-memory CPU snapshot if eligible and
better, and stop on the first defined regression. Restore and reproduce the
selected calibration metrics before publication.

The reproduction gate also hashes every ordered dataset index and exact
selected IoU. Selection, restored state, and staged artifact reload must share
that digest exactly. Validate and retain each pass's aggregate diagnostics
independently; do not require oracle-only diagnostics to be byte-equal. MCLN's
superpoint-center reduction must use the deterministic stable segment mean so
the digest is a meaningful executable contract.

- [x] **Step 4: Run GREEN**

Run both fine-tune unit files and expect all tests to pass.

### Task 8: CPU Verification and One-Batch GPU Smoke

**Files:** all files touched in Tasks 1-7.

- [ ] **Step 1: Compile changed modules**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m py_compile \
  models/losses.py models/mcln.py models/rec_finetune.py \
  utils/scatter_util.py \
  scripts/train_scanrefer_rec_finetune.py train_dist_mod.py
```

- [ ] **Step 2: Run focused and full tests**

Run all fine-tune, REC, evaluator, and artifact suites, then:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest -q
```

Expected: zero failures.

Also run `tests/test_scatter_util.py` on CUDA. Require exact repeated outputs
under `torch.use_deterministic_algorithms(True)` and float64-reference accuracy
for production-shaped coordinates with offsets `0`, `10`, and `1000`.

- [ ] **Step 3: Run a train-only one-update GPU smoke**

Run the production runner with `--smoke-steps 1` into
`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/rec_finetune_smoke1`.
Require finite losses and gradients, correct trainable counts, no OOM, one
remainder-independent update, step-0 and step-1 train-calibration metrics, and
no validation path access. Delete only this disposable smoke output after its
receipt and hashes are recorded.

### Task 9: Launch and Seal the Full Train-Only Fine-Tuning Run

**Output:**
`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/rec_finetune_1ep`

- [ ] **Step 1: Snapshot runtime provenance**

Record SHA-256 and stable file identity for every input and every Python/source
file imported by the runner. Record interpreter/link identity, CUDA/PyTorch
versions, GPU identity, exact environment allowlist, and the no-validation-data
path guard.

- [ ] **Step 2: Run the full contract once**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD:$PWD/pointnet2:${PYTHONPATH:-}" \
/root/miniconda3/envs/bdetr/bin/python \
  scripts/train_scanrefer_rec_finetune.py \
  --data-root /root/autodl-tmp/DATA_ROOT/ \
  --backbone-checkpoint /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth \
  --parent-reranker /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/artifacts/reranker_h256_d010_lr1e3_seed0_final_contract.pth \
  --geometry-reranker /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_artifacts/selected_geometry_reranker.pth \
  --output-dir /root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/rec_finetune_1ep \
  --device cuda:0
```

Require at most 1,836 updates and stop earlier only under the specified
calibration regression rule.

- [ ] **Step 3: Independently verify selected artifacts**

Reload all three deployable files, recompute hashes and calibration metrics,
verify the selection record and initialization lineage, assert every selected
file is `0444`, and confirm no validation file appeared in the recorded access
set.

**Train-only decision evidence (2026-07-20):** The existing configuration has
already been exercised through its first 306-step calibration boundary as a
nondeployable probe. The read-only receipt is
`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/rec_finetune_probe306/smoke-receipt.json`
(`0aff9041923cf3ac8f78cc03f5a664712d9a07ebedae9dcaa4b8a5be5c6de7b2`).
It completed all 306 finite updates with zero frozen-gradient tensors, but
geometry Top-1 regressed from `3461/3316` to `3431/3287`. The declared first
regression rule selected and reproduced step 0, so no deployable artifact was
created. A one-update receipt independently showed an immediate `0.50`
regression from `3316` to `3307` and also restored step 0.

The separately audited final-semantic-head source-gate probe subsequently
preserved the raw-query oracle but still regressed geometry Top-1 to
`3454/3297`, parent-candidate oracle to `3595/3518`, and geometry-candidate
oracle to `3603/3585`; it likewise restored step 0. These two negative controls
close the current joint objective and source-gate objective. Do not launch the
unchanged 1,836-step command or build its official launcher. Retain the three
original read-only artifacts and return to a separate train-only objective
design review before resuming Task 9.

### Task 10: New One-Shot Official Launcher

**Files:**
- Create: `scripts/run_frozen_rec_finetune_official.py`
- Create: `tests/test_run_frozen_rec_finetune_official.py`

- [ ] **Step 1: Write failing synthetic launcher tests**

Adapt the security contract from `tests/test_run_frozen_rec_geometry_official.py`
for the new checkpoint, parent, geometry, and selection record. Cover claim
exclusivity, all input/runtime/code hashes, interpreter symlink lstat identity,
stdout FD identity, environment sanitization, transient tamper detection,
atomic receipt/result/comparison sealing, no-GT config, exact sample count, and
one-shot refusal after a claim exists.

- [ ] **Step 2: Run RED**

Expected: import failure for the new launcher.

- [ ] **Step 3: Implement the launcher**

Reuse reviewed pure helpers from the old launcher, but use a new goal ID,
claim, receipt, output directory, artifact paths, and expected SHA constants.
The launch command must use batch size 12 and set:

```text
butd=true
butd_gt=false
butd_cls=false
eval_use_rec_reranker_scores=true
eval_use_rec_geometry_reranker_scores=true
```

Do not reference, delete, chmod, or reuse the old official claim or result.

- [ ] **Step 4: Verify launcher security**

Run its complete synthetic suite, the full repository suite, `py_compile`, and
an independent bounded review. Do not launch official validation until all are
clean.

### Task 11: Full Official Validation and Completion Audit

- [ ] **Step 1: Create the one-shot claim and launch once**

Use only:

```bash
/root/miniconda3/envs/bdetr/bin/python \
  scripts/run_frozen_rec_finetune_official.py launch
```

Never rerun or delete the claim, even if the metric misses.

- [ ] **Step 2: Seal and compare**

Parse the exact official evaluator output into a read-only result. If a
fine-tuned sidecar was not independently produced before the claim, omit the
sidecar comparison rather than reading validation data a second time. Preserve
the stdout log, config, result, claim, receipt, selection, and all SHA values.

- [ ] **Step 3: Apply the only completion gate**

Require exactly 9,508 samples, `inference_uses_ground_truth=false`, and:

```text
last_ position alignment Acc0.25 Top-1 >= 0.60000
last_ position alignment Acc0.50 Top-1 >= 0.47000
```

Only after all three conditions are proven may the active goal be marked
complete. Otherwise preserve the immutable failure evidence and keep the full
goal active.
