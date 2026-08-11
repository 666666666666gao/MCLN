# ScanRefer REC Hierarchical Query-Variant Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, select, calibrate, and deploy a train-only hierarchical query-then-variant reranker that reaches official position-alignment `Acc@0.25 >= 0.60000` and `Acc@0.50 >= 0.47000` without inference-time ground truth or validation-driven tuning.

**Architecture:** Preserve the immutable epoch-71 backbone, parent reranker, and geometry reranker as the default policy. A new model encodes seven geometry variants per query, pools them into a query representation, selects one of 16 queries, and then selects one of that query's seven variants; an OOF-selected intervention margin abstains to the frozen geometry Top-1 unless the predicted utility gain is sufficiently large. Five scene-disjoint folds inside the 506 fit scenes select one of eight predeclared configurations and one fixed percentile margin before the 56 calibration scenes can be opened.

**Tech Stack:** Python 3.7, PyTorch 1.10.2, CUDA 11.1, pytest, ScanRefer train caches, canonical JSON/SHA-256 provenance, strace file-access audit.

---

The workspace has no `.git` metadata. Do not initialize Git or create a
worktree. Replace commit steps with focused RED/GREEN verification and
SHA-256 records. Do not use subagents in this workspace. Never modify, chmod,
rename, unlink, or replace these protected files:

```text
backbone  3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208
parent    f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b
geometry  835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f
```

**Approved design:**
`docs/superpowers/specs/2026-07-20-scanrefer-rec-hierarchical-risk-controlled-reranking-design.md`

**Phase-2 handoff evidence:** The formal v2 residual replay completed all OOF
fitting and entered the `choice.eligible is not True` baseline-publication
branch. Publication then failed because the receipt validator compared a
pair-label count with an expression-row count. The failed trace is sealed at
`/root/autodl-tmp/rec_residual_audits/crossfit_diag_v2.8IgDW7`; its audit has
`validation_data_accessed=false`, exact protected hashes/modes, runner exit 1,
and no completion receipt. The earlier audited v1 receipt independently records
`selected=baseline`. Do not rerun the residual grid.

## Fixed Tensor Contract

Each row has 16 compact queries and seven query-major variants:

```text
query_features          float32 [16,152]
variant_features        float32 [16,7,25]
query_aux_continuous    float32 [16,4]
  default_score, default_rank, parent_score, parent_rank
query_aux_binary        bool    [16,2]
  default_is_top1, parent_is_top1
variant_aux_continuous  float32 [16,7,2]
  geometry_score, geometry_rank
variant_aux_binary      bool    [16,7,2]
  geometry_is_top1, frozen_baseline_is_top1
query_valid             bool    [16]
variant_valid           bool    [16,7]
candidate_ious          float32 [16,7]  # train labels/diagnostics only
baseline_index          int             # flattened frozen geometry Top-1
baseline_scores         float32 [112]
```

Normalization is fitted on valid entries inside each OOF fit fold separately
for the 152D query features, 25D variant features, four continuous query
auxiliaries, and two continuous variant auxiliaries. Binary fields are not
normalized. The refit artifact stores statistics fitted on all 506 fit scenes.

## File Map

- Create `models/rec_hierarchical_reranker.py`: pure tensor validation,
  monotone query/variant heads, balanced loss, deterministic hierarchical
  proposal, abstention policy, and OOF selector.
- Create `tests/test_rec_hierarchical_reranker.py`: model, mask, monotonicity,
  loss, stable tie, policy, fold, bootstrap, and selector tests.
- Create `scripts/train_scanrefer_rec_hierarchical_reranker.py`: train-only
  loading/materialization, fold-local normalization, fixed-grid cross-fit,
  refit, cache calibration, receipt/artifact validation, and fresh-only
  publication.
- Create `tests/test_train_scanrefer_rec_hierarchical_reranker.py`: cache,
  materialization, normalization isolation, training, calibration, receipt,
  publication, and tamper tests.
- Modify `main_utils.py` and `train_dist_mod.py`: optional hierarchical runtime
  artifact, provenance validation, stable load, and score promotion.
- Create `tests/test_rec_hierarchical_runtime.py`: disabled parity, enabled
  hierarchy, no-GT, and artifact-binding tests.
- Modify `scripts/audit_rec_finetune_file_access.py` and
  `tests/test_audit_rec_finetune_file_access.py`: exact hierarchical runner
  argv and train-only audit profile.
- Create `scripts/run_rec_hierarchical_online_calibration.py` and
  `tests/test_run_rec_hierarchical_online_calibration.py`: deterministic
  56-scene online calibration reproduction.
- Create `scripts/run_frozen_rec_hierarchical_official.py` and
  `tests/test_run_frozen_rec_hierarchical_official.py`: immutable one-shot
  official validation launcher and result sealer.

### Task 1: Pure Hierarchical Model And Stable Proposal

**Files:**
- Create: `models/rec_hierarchical_reranker.py`
- Create: `tests/test_rec_hierarchical_reranker.py`

- [ ] **Step 1: Write failing tensor-contract and GT-isolation tests**

Build two synthetic rows using the fixed tensor contract. Require the public
model API below and verify that adding `candidate_ious`, `center_label`,
`size_gts`, or `gt_masks` to unrelated dictionaries cannot affect model
outputs because none is accepted by `forward`:

```python
model = HierarchicalQueryVariantReranker(hidden_dim=64, dropout=0.1)
outputs = model(
    query_features=query_features,
    variant_features=variant_features,
    query_aux_continuous=query_aux_continuous,
    query_aux_binary=query_aux_binary,
    variant_aux_continuous=variant_aux_continuous,
    variant_aux_binary=variant_aux_binary,
    query_valid=query_valid,
    variant_valid=variant_valid,
)
assert set(outputs) == {
    "query_logits", "variant_logits", "query_embedding",
    "variant_embedding",
}
assert outputs["query_logits"].shape == (2, 16, 2)
assert outputs["variant_logits"].shape == (2, 16, 7, 2)
```

Reject wrong shapes, float64, integer feature tensors, non-bool masks,
non-finite valid values, a valid variant under an invalid query, and a row
without a valid query.

- [ ] **Step 2: Run RED for the missing module**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_hierarchical_reranker.py -q
```

Expected: collection fails because `models.rec_hierarchical_reranker` does not
exist.

- [ ] **Step 3: Implement the encoders and monotone heads**

Define exact constants:

```python
QUERY_COUNT = 16
VARIANT_COUNT = 7
QUERY_FEATURE_DIM = 152
VARIANT_FEATURE_DIM = 25
QUERY_AUX_CONTINUOUS_DIM = 4
QUERY_AUX_BINARY_DIM = 2
VARIANT_AUX_CONTINUOUS_DIM = 2
VARIANT_AUX_BINARY_DIM = 2
HIERARCHICAL_THRESHOLDS = (0.25, 0.50)
HIERARCHICAL_THRESHOLD_WEIGHTS = (2.0, 1.0)
HIERARCHICAL_HIDDEN_DIMS = (64, 128)
```

Implement `HierarchicalQueryVariantReranker`. Its variant encoder is
`Linear(25,H) -> ReLU -> Dropout(0.1)`. Masked mean and masked maximum over
the seven variant embeddings produce `2H` query summary features. The query
encoder is `Linear(152 + 4 + 2 + 2H,H) -> ReLU -> Dropout(0.1)`. The query
head maps `H -> 2`. The variant head is
`Linear(2H + 2 + 2,H) -> ReLU -> Dropout(0.1) -> Linear(H,2)` using the query
embedding, variant embedding, continuous variant auxiliaries, and binary
variant auxiliaries.

The first logit is `logit25`; the second is `conditional50`. Expose:

```python
def monotone_hit_probabilities(logits):
    probability25 = logits[..., 0].sigmoid()
    probability50 = probability25 * logits[..., 1].sigmoid()
    return torch.stack((probability25, probability50), dim=-1)
```

Require `0 <= P50 <= P25 <= 1` without clipping. Zero invalid embeddings and
logits so padding cannot alter masked pooling or loss.

- [ ] **Step 4: Write failing stable hierarchy tests**

Require query selection by `2*P25 + P50`, then variant selection only inside
the selected query. Construct a row where the globally largest variant score
belongs to another query and prove it cannot win after the query decision.
Require lowest query/variant index on exact ties.

- [ ] **Step 5: Implement deterministic proposal selection**

Expose:

```python
def select_hierarchical_proposal(
        query_logits, variant_logits, query_valid, variant_valid):
    query_probability = monotone_hit_probabilities(query_logits)
    variant_probability = monotone_hit_probabilities(variant_logits)
    query_utility = 2.0 * query_probability[..., 0] + query_probability[..., 1]
    variant_utility = (
        2.0 * variant_probability[..., 0] + variant_probability[..., 1]
    )
    selected_query = query_utility.masked_fill(~query_valid, -float("inf")).argmax(1)
    rows = torch.arange(query_logits.shape[0], device=query_logits.device)
    selected_variant = variant_utility[rows, selected_query].masked_fill(
        ~variant_valid[rows, selected_query], -float("inf")
    ).argmax(1)
    selected_flat = selected_query * VARIANT_COUNT + selected_variant
    return {
        "query_indices": selected_query,
        "variant_indices": selected_variant,
        "flat_indices": selected_flat,
        "query_utility": query_utility,
        "variant_utility": variant_utility,
    }
```

Validate all inputs and rely on first-index `argmax` only after preserving the
canonical frozen query-major/variant-minor axis.

- [ ] **Step 6: Run GREEN**

Run the focused file and record SHA-256 for the model and test.

### Task 2: Strict Labels, Row-Balanced Loss, And Risk Policy

**Files:**
- Modify: `models/rec_hierarchical_reranker.py`
- Modify: `tests/test_rec_hierarchical_reranker.py`

- [ ] **Step 1: Write failing strict-label and loss tests**

Require variant targets `candidate_ious > threshold` and query targets equal
`any(valid variant target)` for each query. IoU exactly 0.25 or 0.50 is a miss.
Test rows with unequal valid query and variant counts; duplicating variants in
one query must not increase that query's or row's loss weight.

- [ ] **Step 2: Implement detached labels**

```python
def build_hierarchical_targets(candidate_ious, variant_valid):
    variant_targets = torch.stack(
        tuple(candidate_ious.gt(threshold)
              for threshold in HIERARCHICAL_THRESHOLDS),
        dim=-1,
    ) & variant_valid.unsqueeze(-1)
    query_targets = variant_targets.any(dim=2)
    query_valid = variant_valid.any(dim=2)
    return {
        "query_targets": query_targets.detach(),
        "variant_targets": variant_targets.detach(),
        "query_valid": query_valid,
    }
```

- [ ] **Step 3: Implement false-positive-weighted balanced BCE**

For probability `p` and binary target `y`, use
`-(y*log(p) + false_positive_cost*(1-y)*log1p(-p))` with machine-epsilon
clamping. Average variants inside each query, then valid queries inside each
row, then rows equally. Compute the query loss with the same row balancing.
Combine thresholds with weights 2:1 and return `query_loss + variant_loss`.
Expose exact positive/negative counts for both heads and thresholds.

- [ ] **Step 4: Write failing abstention and selector tests**

Require the frozen baseline when the proposal is identical, invalid, has
non-positive gain, or gain is below the margin. Promotion must alter only the
selected score using `torch.nextafter(max_score, +inf)`. Add five-fold and
scene-bootstrap fixtures for these fixed grids:

```python
HIERARCHICAL_WEIGHT_DECAYS = (1e-4, 1e-3)
HIERARCHICAL_FALSE_POSITIVE_COSTS = (2.0, 4.0)
HIERARCHICAL_MARGIN_PERCENTILES = (
    50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5, 99.0,
)
```

- [ ] **Step 5: Implement the fixed policy and selection gate**

`apply_hierarchical_policy` accepts one proposed flat index and predicted gain
per row. A candidate is eligible only when all five folds have non-negative
exact deltas at both thresholds, pooled `delta_hits025 > 0`, and both
scene-clustered one-sided 95% lower bounds are at least zero. Order eligible
candidates by `2*delta_hits025 + delta_hits050`, larger margin, fewer switches,
smaller hidden dimension, larger weight decay, then larger false-positive
cost. Keep the no-switch sentinel diagnostic but never select it.

- [ ] **Step 6: Run GREEN and residual-selector regressions**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_hierarchical_reranker.py \
  tests/test_rec_selective_residual.py -q
```

### Task 3: Train-Only Materialization And Fold-Local Normalization

**Files:**
- Create: `scripts/train_scanrefer_rec_hierarchical_reranker.py`
- Create: `tests/test_train_scanrefer_rec_hierarchical_reranker.py`

- [ ] **Step 1: Write failing materialization tests**

Use recording frozen parent and geometry models. Require canonical CPU records
with exactly the fixed tensor contract and no target field in any model call.
Changing `candidate_ious` must change only labels/digests, never features,
scores, proposal, or validity.

- [ ] **Step 2: Run RED for the missing trainer module**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_train_scanrefer_rec_hierarchical_reranker.py -q
```

Expected: collection fails for the missing trainer.

- [ ] **Step 3: Reuse strict train-only loading and identity split**

Import `load_residual_training_inputs`, `split_residual_joined_rows`,
`capture_immutable_artifact_identities`, and the authoritative cache/hash
constants from `train_scanrefer_rec_selective_residual`. Load only the base
train cache, geometry-train cache, parent artifact, and geometry artifact.
Split joined identities into 506 fit scenes/33,040 rows and 56 calibration
scenes/3,625 rows before materialization. Do not pass calibration rows to the
materializer until OOF selection is frozen and eligible.

- [ ] **Step 4: Materialize raw deployable hierarchy fields**

For batches of 256, call `build_geometry_training_batch`, run the frozen
geometry scorer under `no_grad`/disabled autocast, and reconstruct the exact
parent prior, ranks, geometry ranks, baseline scores, and stable baseline
index. Slice the first 152 and next 25 raw feature dimensions using the
artifact's exact names; derive the four continuous and two binary query fields
and two continuous/two binary variant fields defined above. Reshape only after
verifying query positions are `[0..15]` repeated seven times and variant
indices are `[0..6]` inside each query.

- [ ] **Step 5: Write failing normalization-isolation tests**

Perturb held-fold features by a large constant and prove fit-fold means/stds
and normalized fit tensors are bitwise unchanged. Require float64 streaming
population statistics, minimum standard deviation `1e-6`, and zero-filled
invalid normalized entries.

- [ ] **Step 6: Implement and bind normalization statistics**

Expose `fit_hierarchical_normalization(records)` and
`normalize_hierarchical_batch(batch, statistics)`. Store counts, float32
means/stds, feature-name arrays, and a canonical SHA-256. Reject statistics
fitted with an empty valid field or applied to a different schema.

- [ ] **Step 7: Run GREEN and cache regressions**

Run the new trainer tests plus `tests/test_train_scanrefer_rec_selective_residual.py`
and `tests/test_train_rec_geometry_reranker.py`.

### Task 4: Fixed Eight-Configuration Five-Fold OOF

**Files:**
- Modify: `scripts/train_scanrefer_rec_hierarchical_reranker.py`
- Modify: `tests/test_train_scanrefer_rec_hierarchical_reranker.py`

- [ ] **Step 1: Write failing cross-fit isolation tests**

Use at least five scenes with unequal row counts. Observe every optimizer
batch and normalization fit. Require eight configurations, five held folds,
12 epochs, one OOF proposal/gain per fit row, no held scene in fitting or
statistics, and these immutable settings:

```python
HIERARCHICAL_EPOCHS = 12
HIERARCHICAL_BATCH_SIZE = 256
HIERARCHICAL_LEARNING_RATE = 3e-4
HIERARCHICAL_GRAD_CLIP_NORM = 1.0
HIERARCHICAL_DROPOUT = 0.1
HIERARCHICAL_MODEL_SEED = 0
```

- [ ] **Step 2: Implement deterministic fit and prediction**

Use AdamW, fixed learning rate, selected weight decay, 12 complete shuffled
epochs, gradient clipping 1.0, and seed 0 for Python/CPU/CUDA. Recreate the
model and fit-fold statistics for every configuration/fold. Save only OOF
query/variant proposal indices, proposal gain, and canonical digests; delete
fold models before advancing.

- [ ] **Step 3: Build every fixed margin candidate**

For each configuration, add one no-switch sentinel and every available
nearest-rank positive-gain percentile. Record exact baseline/proposed hits,
fixes, breaks, neutral switches, abstentions, switch rate, five fold deltas,
10,000-replicate scene bootstraps, selected-query changes, same-query variant
changes, wrong-query recoveries, and wrong-variant recoveries. Labels are
diagnostics only and cannot enter model inputs.

- [ ] **Step 4: Implement selection and truthful rejection**

Select once using the gate from Task 2. If no candidate is eligible, set
`selected=baseline`, `calibration={"status":"not_run",
"reason":"oof_selection_rejected"}`, publish no artifact, and never
materialize the 56 calibration scenes.

- [ ] **Step 5: Run GREEN**

Run the cross-fit tests and require exact repeatability for OOF digests and
selected choice across two CPU synthetic runs.

### Task 5: Refit, Cache Calibration, Receipt, And Fresh Publication

**Files:**
- Modify: `scripts/train_scanrefer_rec_hierarchical_reranker.py`
- Modify: `tests/test_train_scanrefer_rec_hierarchical_reranker.py`

- [ ] **Step 1: Write failing refit/calibration tests**

Require one refit on all 506 fit scenes using all-fit normalization. Only after
the OOF choice is frozen may the 3,625 calibration rows be materialized once.
The fixed cache gate is:

```text
candidate hits025 >= 3524
candidate hits050 >= 3315
baseline == 3461/3315
oracle == 3606/3588
candidate-IoU digest unchanged
row-materialization digest unchanged
```

- [ ] **Step 2: Implement the staged artifact**

Use schema `rec-hierarchical-query-variant-v1`. Store model state, hidden
dimension, weight decay, false-positive cost, margin/percentile, all-fit
normalization, feature names, scene-fold hash, OOF proposal/gain digests,
cache calibration record, immutable input SHAs, `deployable=false`, and
`validation_data_accessed=false`. Strict-load on CPU and require exact tensor
and metadata equality.

- [ ] **Step 3: Implement a strict v1 result receipt**

Use schema `rec-hierarchical-result-receipt-v1`. Bind the authoritative split,
fit identity/materialization/statistics digests, eight configuration records,
all policy diagnostics, choice, calibration status/record, artifact SHA,
protected before/after snapshots, and `validation_data_accessed=false`.
Validate every count, fold partition, bootstrap, eligibility predicate, digest,
and artifact/calibration relationship before publication.

- [ ] **Step 4: Implement completion-last fresh-only publication**

Reuse the residual reservation and exclusive descriptor writers. Reserve the
absolute output directory before loading caches, write the staged artifact
with `O_EXCL|O_NOFOLLOW`, fsync and seal it `0444`, then write
`result-receipt.json` last through a non-overwriting hard link. A failure may
leave an empty/partial `0700` directory but never a completion receipt.

- [ ] **Step 5: Add race/tamper tests**

Cover an existing output, symlink component, racing final filename, interrupted
artifact/receipt write, changed normalization digest, fold count, candidate
count, calibration status, and protected artifact identity. Require no existing
byte to be overwritten.

- [ ] **Step 6: Run GREEN**

Run the complete hierarchical model/trainer suites and strict-reload every
synthetic success/failure receipt.

### Task 6: Runtime Integration With Disabled-Path Parity

**Files:**
- Modify: `main_utils.py`
- Modify: `train_dist_mod.py`
- Create: `tests/test_rec_hierarchical_runtime.py`

- [ ] **Step 1: Write failing parser and disabled-parity tests**

Add `--rec_hierarchical_reranker_checkpoint` and
`--eval_use_rec_hierarchical_reranker_scores`. Require the enable flag to imply
the parent and geometry flags/artifacts and to be mutually exclusive with
`--eval_use_rec_selective_residual_scores`. With hierarchy disabled, compare
the full geometry output dictionary bitwise with the current best runtime.

- [ ] **Step 2: Write failing enabled-runtime tests**

Inject a recording model/artifact. Require the exact raw hierarchy tensors,
artifact normalization, query-first/variant-second proposal, selected margin,
and one score promotion. GT sentinels in `end_points`/`inputs` must remain
unread and absent from outputs. Keep the existing six-field geometry output
schema unchanged.

- [ ] **Step 3: Implement stable artifact loading/provenance**

Stable-hash and load the hierarchy artifact once. Bind its backbone, parent,
geometry, train-cache, feature-name, normalization, OOF, and calibration
digests to the already loaded frozen artifacts. Freeze/eval every parameter and
reject partial state or artifact mutation between batches.

- [ ] **Step 4: Apply hierarchy after frozen geometry scoring**

Inside `_build_rec_geometry_runtime_outputs_float32`, build the same raw query,
variant, and auxiliary tensors used during training, normalize with artifact
statistics, run the model, call `select_hierarchical_proposal` and
`apply_hierarchical_policy`, and replace only `rec_geometry_scores`. Do not
attach probabilities, labels, or diagnostics to `end_points`.

- [ ] **Step 5: Run runtime/evaluator regressions**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_hierarchical_runtime.py \
  tests/test_rec_selective_residual_runtime.py \
  tests/test_rec_geometry_runtime.py \
  tests/test_grounding_evaluator_rec_geometry.py -q
```

### Task 7: Hierarchical File-Access Audit

**Files:**
- Modify: `scripts/audit_rec_finetune_file_access.py`
- Modify: `tests/test_audit_rec_finetune_file_access.py`

- [ ] **Step 1: Write failing exact-command tests**

Add mode `hierarchical` and
`build_hierarchical_training_argv(interpreter, base_cache, geometry_cache,
parent_artifact, geometry_artifact, output_dir)`. Require the exact trainer
path, fixed `--device cuda:0`, two train cache roots, and no other arguments.

- [ ] **Step 2: Write failing access and publication tests**

Require validation/official/claim/receipt path access to fail, as must any
successful destructive syscall outside the reserved output or scratch roots.
Require exact SHA/`0444` for all protected artifacts, runner exit zero,
completion receipt, and `validation_data_accessed=false`.

- [ ] **Step 3: Implement the hierarchical audit profile**

Reuse the hardened parser and syscall set. Generalize only runner identity,
expected argv, receipt schema, and artifact output name. Keep fresh `0444`
audit-report publication and every existing residual/source-gate deny rule.

- [ ] **Step 4: Run the complete audit suite**

Run all collected audit tests, splitting by non-overlapping node IDs only if
the tool output window truncates the footer. Record exact pass totals.

### Task 8: Online Calibration And One-Shot Official Runner

**Files:**
- Create: `scripts/run_rec_hierarchical_online_calibration.py`
- Create: `tests/test_run_rec_hierarchical_online_calibration.py`
- Create: `scripts/run_frozen_rec_hierarchical_official.py`
- Create: `tests/test_run_frozen_rec_hierarchical_official.py`

- [ ] **Step 1: Write failing online-calibration contract tests**

Reuse the ordered 56-scene source-gate calibration view. Require 3,625 rows,
baseline `3461/3316`, candidate `hits025 >= 3524`, candidate
`hits050 >= 3316`, identical raw-query/candidate-oracle counts and digests,
no validation access, and exact hierarchy artifact SHA. A failure leaves the
staged artifact nondeployable.

- [ ] **Step 2: Implement and seal online calibration**

Run the live frozen MCLN/parent/geometry/hierarchy path with no GT-fed model
input. Ground truth may be used only after score/box selection to count IoU.
Write an exclusive `online-calibration.json` with command/environment/code/
artifact/input digests and immutable before/after snapshots. On pass, publish
a new `deployed_hierarchical_reranker.pth` with `deployable=true`; never edit
the staged artifact in place.

- [ ] **Step 3: Write failing official launcher tests**

Clone the fixed geometry command and add only the deployed hierarchy checkpoint
and enable flag. Require exactly 9,508 rows, no GT inference, exact four
artifact hashes, one exclusive claim, immutable stdout/log/result, and these
integer gates:

```text
hits025 >= 5705
hits050 >= 4469
```

- [ ] **Step 4: Implement immutable official launch/sealing**

Preflight the deployed artifact, online/cache calibration receipts, runtime
flags, code manifest, Python/CUDA environment, and protected artifacts before
creating the claim. Parse independent log/stdout metrics exactly once, recover
integer counts, reconcile position subgroups if printed, and publish the
read-only result last. Preserve all old best files regardless of outcome.

- [ ] **Step 5: Run online/official launcher suites**

Run the two new test files plus the existing geometry and residual official
launcher tests.

### Task 9: Formal Verification, Train-Only Selection, And Evaluation

**Files:** all files above; no source edits during formal runs.

- [ ] **Step 1: Compile and run CPU regressions**

Compile every changed Python module. Run hierarchical model/trainer/runtime,
geometry/runtime/evaluator, audit, online calibration, and official launcher
tests, then the complete repository suite. Record exact test totals and
SHA-256 for every changed source/test/plan file.

- [ ] **Step 2: Snapshot frozen inputs and reserve fresh roots**

Require exact protected SHA-256/mode/device/inode/size/mtime/ctime. Reserve a
fresh audit root outside `DATA_ROOT`, a `0700` empty scratch directory, a fresh
experiment path, and a fresh audit report path.

- [ ] **Step 3: Run the eight-configuration OOF under strace**

Use the exact audit-built argv, CUDA device 0, seed 0, five scene folds, 12
epochs, and fixed grid. Require audit `pass=true`, `violations=[]`,
`validation_data_accessed=false`, exact protected hashes/modes, and a strict
completion receipt.

- [ ] **Step 4: Apply OOF and cache gates without reinterpretation**

If no candidate is eligible, preserve baseline, leave calibration `not_run`,
and write a new train-only design before any further search. If OOF is eligible
but cache calibration fails `3524/3315`, preserve the staged evidence and do
not tune on those 56 scenes. Only a passing unchanged artifact advances.

- [ ] **Step 5: Run online calibration once**

Require `candidate >= 3524/3316`, unchanged baseline/oracle/raw-query digests,
no validation access, and a newly sealed deployable artifact. Do not modify or
rerun it after observing online calibration.

- [ ] **Step 6: Launch one official evaluation**

Use only `scripts/run_frozen_rec_hierarchical_official.py`. Success requires
9,508 expressions, position-alignment `hits025 >= 5705`, `hits050 >= 4469`,
`inference_uses_ground_truth=false`, immutable evidence, and unchanged old
best artifacts. A metric failure is retained as one result and cannot be used
to alter/retry the frozen candidate.

## Plan Self-Review

- Spec coverage: hierarchy, monotone heads, row-balanced loss, fixed 8-model
  grid, fold-local normalization, risk abstention, 506/56 isolation, offline
  and online gates, runtime, audit, one-shot official evaluation, and protected
  artifact preservation are each assigned to a task.
- Type consistency: all tasks use 16 queries, seven variants, 152D query
  features, 25D variant features, 4+2 query auxiliaries, 2+2 variant
  auxiliaries, and query-major flat indices `query*7+variant`.
- Search scope: only hidden dimension, weight decay, false-positive cost, and
  the fixed OOF margin percentiles vary. Calibration and validation cannot
  expand or reorder the grid.
- Completion gate: the plan is not complete until exact official counts prove
  both requested thresholds and inference-time GT remains absent.
