# ScanRefer REC Query Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a deployable query-level reranker that raises ScanRefer validation `last_ position alignment` Top-1 to Acc@0.25 >= 0.60000 and Acc@0.50 >= 0.47000.

**Architecture:** Freeze the best MCLN checkpoint, extract a deterministic Top-16 union of default and contrastive queries, cache deployable per-query features and training-only IoU targets, then train a shared MLP with listwise, threshold, and IoU losses. Integrate its full-query scores into the existing evaluator without changing predicted box geometry.

**Tech Stack:** Python 3.7, PyTorch 1.10.2, pytest, ScanRefer, existing MCLN/PointNet++ pipeline.

**Repository constraint:** This directory has no `.git` metadata, so worktree creation and commits are impossible. Preserve unrelated files and record verification output after every task.

---

### Task 1: Pure Candidate Selection And Oracle Utilities

**Files:**
- Create: `models/rec_reranker.py`
- Create: `tests/test_rec_reranker.py`

- [ ] **Step 1: Write failing candidate-selection tests**

```python
def test_select_candidate_indices_unions_sources_without_duplicates():
    default = torch.tensor([[0.9, 0.8, 0.7, 0.1]])
    contrastive = torch.tensor([[0.1, 0.95, 0.6, 0.7]])
    out, valid = select_candidate_indices(
        default, contrastive, topk_per_source=2, max_candidates=4
    )
    assert out.tolist() == [[0, 1, 3, 2]]
    assert valid.tolist() == [[True, True, True, True]]


def test_candidate_oracle_reports_both_thresholds():
    ious = torch.tensor([[0.1, 0.6], [0.3, 0.4]])
    valid = torch.ones_like(ious, dtype=torch.bool)
    metrics = compute_candidate_oracle(ious, valid)
    assert metrics == {"acc025": 1.0, "acc050": 0.5}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_reranker.py -q
```

Expected: import failure because `models.rec_reranker` does not exist.

- [ ] **Step 3: Implement deterministic selection and oracle helpers**

Implement these public APIs in `models/rec_reranker.py`:

```python
def select_candidate_indices(
        default_scores, contrastive_scores,
        topk_per_source=8, max_candidates=16):
    """Return [B,K] query indices and [B,K] validity mask."""


def compute_query_ious(candidate_boxes, gt_boxes, gt_mask):
    """Return maximum target IoU per candidate as [B,K]."""


def compute_candidate_oracle(candidate_ious, valid_mask):
    """Return strict-IoU acc025 and acc050 as Python floats."""
```

Selection order is default Top-K first, contrastive Top-K second, followed by
remaining default-score order until `max_candidates` is filled. Duplicate
query indices are removed per sample. Invalid padding uses index zero and is
marked false.

- [ ] **Step 4: Add strict-threshold and empty-padding tests**

Tests must prove IoU exactly equal to 0.25 or 0.50 is not counted, matching
`GroundingEvaluator`, and invalid padded candidates cannot become oracle.

- [ ] **Step 5: Run focused and existing source-choice tests**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_reranker.py \
  tests/test_source_choice_adapter.py \
  tests/test_source_choice_selector.py -q
```

Expected: all selected tests pass.

### Task 2: Deployable Feature Adapter And Reranker Model

**Files:**
- Create: `models/rec_candidate_adapter.py`
- Modify: `models/rec_reranker.py`
- Create: `tests/test_rec_candidate_adapter.py`
- Modify: `tests/test_rec_reranker.py`

- [ ] **Step 1: Write a failing adapter shape/leakage test**

Construct a synthetic `end_points` batch with four queries, token component
maps, projected queries/tokens, seed logits, query indices, and two mask
streams. Assert:

```python
batch = build_rec_candidate_batch(end_points, inputs, max_candidates=4)
assert batch["features"].shape[:2] == (2, 4)
assert batch["boxes"].shape == (2, 4, 6)
assert batch["query_indices"].shape == (2, 4)
assert batch["valid_mask"].dtype == torch.bool
assert not any("gt" in key or "iou" in key for key in batch["model_inputs"])
```

- [ ] **Step 2: Run adapter tests and verify RED**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_candidate_adapter.py -q
```

Expected: import failure for the missing adapter.

- [ ] **Step 3: Implement the feature adapter**

Expose:

```python
def build_rec_candidate_batch(end_points, inputs, max_candidates=16):
    """Build deployable candidate tensors from an MCLN forward result."""


def attach_candidate_targets(candidate_batch, end_points):
    """Return a copy with training-only candidate_ious and labels."""


def scatter_candidate_scores(candidate_scores, query_indices, valid_mask,
                             num_queries, fill_value=-1e4):
    """Map compact scores back to [B,Q]."""
```

The deployable feature vector must include projected query/text features,
scene-normalized box geometry, five token-component scores, default and
contrastive scores/ranks/margins, gathered query objectness, and fused-mask
confidence/foreground/Dice statistics. Scene normalization clamps every
extent to at least `1e-6`. Target attachment is the only function allowed to
read `center_label`, `size_gts`, or `box_label_mask`.

- [ ] **Step 4: Write failing model and loss tests**

Test these properties:

```python
model = QueryReranker(input_dim=features.shape[-1], hidden_dim=64)
out = model(features, valid_mask)
assert out["ranking_logits"].shape == valid_mask.shape
assert out["threshold_logits"].shape == valid_mask.shape + (2,)
assert out["iou_estimate"].shape == valid_mask.shape
assert torch.all((out["iou_estimate"] >= 0) & (out["iou_estimate"] <= 1))
```

Also assert one optimizer step reduces `compute_rec_reranker_loss` on a fixed
toy batch and padded candidates never win the listwise target.

- [ ] **Step 5: Implement the MLP and aligned loss**

Implement:

```python
class QueryReranker(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1): ...
    def forward(self, features, valid_mask): ...


def compute_rec_reranker_loss(outputs, candidate_ious, valid_mask,
                              listwise_weight=1.0,
                              threshold_weight=1.0,
                              iou_weight=0.5): ...
```

The pointwise input concatenates each candidate with masked per-sample
feature mean and max. The listwise target is selected lexicographically by
`IoU > 0.50`, then `IoU > 0.25`, then IoU. Threshold BCE uses strict labels;
IoU regression uses smooth L1. Return total loss and detached components.

- [ ] **Step 6: Verify Tasks 1-2**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_reranker.py tests/test_rec_candidate_adapter.py -q
```

Expected: all tests pass.

### Task 3: Resumable ScanRefer Candidate Cache

**Files:**
- Create: `scripts/cache_scanrefer_rec_candidates.py`
- Create: `tests/test_rec_candidate_cache.py`

- [ ] **Step 1: Write failing shard and checkpoint-key tests**

Cover stripping one leading `module.` prefix, atomic shard naming, manifest
round trips, refusing to mix checkpoint fingerprints, and resuming after the
last complete shard.

- [ ] **Step 2: Run cache tests and verify RED**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_candidate_cache.py -q
```

Expected: import failure for the missing cache script.

- [ ] **Step 3: Implement extraction**

The script accepts:

```text
--split {train,val}
--data-root PATH
--checkpoint PATH
--output-dir PATH
--batch-size INT
--num-workers INT
--shard-size INT
--max-candidates INT
--limit INT
--device cuda:0
--overwrite
```

It instantiates `Joint3DDataset(dataset_dict={"scanrefer": 1})`, disables
geometric and detector augmentation for deterministic caching, loads the
frozen MCLN checkpoint after stripping DDP prefixes, runs `TrainTester` input
construction, builds candidate features, attaches targets, and saves CPU
tensors in atomic `.pt` shards. Each row stores dataset index, scan ID,
target ID, candidate tensors, default Top-1 query index, and target IoUs.
The JSON manifest records schema version, checkpoint SHA-256, split,
candidate settings, feature dimension, sample count, and shard paths.

- [ ] **Step 4: Add an oracle-only mode and gate**

After extraction print exact strict metrics for default Top-1 and candidate
oracle. Exit with code 2 when `--require-oracle 0.62 0.50` is supplied and
either oracle threshold is missed.

- [ ] **Step 5: Run cache unit tests and one-batch smoke extraction**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_candidate_cache.py -q

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/bdetr/bin/python \
  scripts/cache_scanrefer_rec_candidates.py \
  --split val \
  --data-root /root/autodl-tmp/DATA_ROOT/ \
  --checkpoint /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth \
  --output-dir /root/autodl-tmp/DATA_ROOT/output/rec_reranker/cache_smoke \
  --batch-size 1 --num-workers 0 --shard-size 1 --limit 1 \
  --max-candidates 16 --device cuda:0 --overwrite
```

Expected: one shard, one manifest, finite features, and printed oracle metrics.

### Task 4: Scene-Disjoint Reranker Training

**Files:**
- Create: `scripts/train_rec_reranker.py`
- Create: `tests/test_train_rec_reranker.py`

- [ ] **Step 1: Write failing deterministic split and checkpoint tests**

Assert that all rows from one scene remain in one partition, the same seed
produces the same 90/10 split, feature normalization uses fit rows only, and a
saved artifact restores identical logits.

- [ ] **Step 2: Run training tests and verify RED**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_train_rec_reranker.py -q
```

Expected: import failure for the missing training script.

- [ ] **Step 3: Implement cache dataset, normalization, and trainer**

The script accepts train-cache path, output path, seed, hidden dimension,
dropout, learning rate, weight decay, batch size, max epochs, and patience.
Use AdamW, gradient clipping at 1.0, deterministic scene split, and early stop
on calibration score:

```python
score = min(acc025 / 0.60, acc050 / 0.47) + 0.1 * (acc025 + acc050)
```

Save a single artifact containing model state, feature mean/std, adapter
schema version, input dimension, candidate rule, epoch, and calibration
metrics. The training loader must never load ScanRefer validation shards.

- [ ] **Step 4: Verify on a synthetic learnable cache**

Generate a small cache in the test where one feature identifies the best
query. Require the trained artifact to improve both calibration accuracies
over the default candidate and reproduce scores after reload.

- [ ] **Step 5: Run Tasks 1-4 tests**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_rec_reranker.py \
  tests/test_rec_candidate_adapter.py \
  tests/test_rec_candidate_cache.py \
  tests/test_train_rec_reranker.py -q
```

Expected: all tests pass.

### Task 5: Runtime Integration And Full Metric Gate

**Files:**
- Modify: `main_utils.py`
- Modify: `train_dist_mod.py`
- Modify: `src/grounding_evaluator.py`
- Create: `tests/test_grounding_evaluator_rec_reranker.py`
- Modify: `scripts/train_scanrefer_mcln_sp.sh`

- [ ] **Step 1: Write a failing evaluator override test**

Build an `end_points` fixture where default scores select a wrong box and
`rec_reranker_scores` select the correct box. Assert the default evaluator
misses and `eval_use_rec_reranker_scores=True` records one hit at both
thresholds.

- [ ] **Step 2: Run the evaluator test and verify RED**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest \
  tests/test_grounding_evaluator_rec_reranker.py -q
```

Expected: constructor or behavior failure because the override is absent.

- [ ] **Step 3: Add CLI and runtime loading**

Add:

```text
--rec_reranker_checkpoint PATH
--eval_use_rec_reranker_scores
```

Load the artifact once in `TrainTester`, rebuild deployable features after
each MCLN forward, normalize with stored fit statistics, run the frozen
reranker, scatter compact scores to `[B,256]`, and store them as
`end_points["rec_reranker_scores"]`. Assert adapter schema and checkpoint
candidate rules match the artifact.

- [ ] **Step 4: Add evaluator precedence**

In position alignment only, use `rec_reranker_scores` when the new flag is
set. Give it precedence over source-choice scores. Keep semantic and mask
metrics unchanged. Log fixes, breaks, and candidate oracle as separate
diagnostics without changing the official metric names.

- [ ] **Step 5: Run the complete unit suite**

```bash
/root/miniconda3/envs/bdetr/bin/python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 6: Build full deterministic caches**

Run Task 3 for ScanRefer train and validation under:

`/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/`

Require validation candidate oracle `>= 0.62000/0.50000`. If this fails, stop
before training and create the mask-derived geometry follow-up specified in
the design.

- [ ] **Step 7: Train the reranker and run full validation**

Train only from the train cache, then launch the existing one-GPU distributed
evaluation with the best MCLN checkpoint, reranker artifact, and evaluator
override. Preserve the full `config.json`, `log.txt`, and reranker artifact.

- [ ] **Step 8: Apply the final acceptance gate**

Extract the final official log lines and require:

```text
last_ position alignment Acc0.25: Top-1: >= 0.60000
last_ position alignment Acc0.50: Top-1: >= 0.47000
```

If either fails, keep the project goal active and proceed to mask-derived
geometry, then REC-specific short fine-tuning, in that order.
