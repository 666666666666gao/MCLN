# ScanRefer Joint Box-Mask Pareto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify one reproducible single-backbone ScanRefer system that reaches Position Acc@0.25 >= 59.00%, preserves at least 4,621 Position Acc@0.50 hits, and exceeds 58.70% / 50.70% / 44.72% on the three mask metrics.

**Architecture:** Extend the existing deterministic train replay with query-level mask targets, then train a frozen-backbone multi-task adapter over the existing 16 queries and 112 geometry variants. A conformal Pareto gate may switch away from the protected geometry top-1 only when both box tiers are protected, and the selected geometry variant always uses its parent query's calibrated mask. Mask-head fine-tuning is conditional on the train-only oracle or calibration gate.

**Tech Stack:** Python 3.7, PyTorch 1.12, pytest, existing MCLN/REC candidate and geometry helpers, ScanRefer train replay, JSON/PyTorch sharded receipts.

**Execution:** Inline execution is pre-authorized by the user. Do not stop for plan review. This directory has no Git metadata and must not be initialized as a repository; replace each commit checkpoint with a SHA-256 source snapshot and protected-artifact audit.

---

## File Map

- Create `models/rec_joint_box_mask.py`: pure mask target, calibration, joint-feature, model, conformal, and Pareto-selection APIs.
- Create `scripts/audit_scanrefer_joint_box_mask.py`: deterministic Stage 0 train-only replay and headroom receipt.
- Create `scripts/cache_scanrefer_joint_box_mask.py`: sharded full-train materialization with capacity preflight and atomic publication.
- Create `scripts/train_scanrefer_joint_box_mask.py`: scene-disjoint OOF training, calibration, gate, and artifact publication.
- Create `scripts/run_frozen_rec_joint_box_mask_official.py`: protected formal evaluation launcher and final receipt.
- Create `tests/test_rec_joint_box_mask.py`: pure tensor/model tests.
- Create `tests/test_audit_scanrefer_joint_box_mask.py`: audit/oracle/receipt tests.
- Create `tests/test_cache_scanrefer_joint_box_mask.py`: cache and storage preflight tests.
- Create `tests/test_train_scanrefer_joint_box_mask.py`: split, conformal, policy, and publication-gate tests.
- Create `tests/test_rec_joint_box_mask_runtime.py`: runtime loader, schema, and attachment tests.
- Create `tests/test_grounding_evaluator_rec_joint_box_mask.py`: same-query box/mask evaluator tests.
- Create `tests/test_run_frozen_rec_joint_box_mask_official.py`: command, preservation, and receipt tests.
- Modify `main_utils.py`: joint artifact and evaluation flags.
- Modify `train_dist_mod.py`: load one joint artifact, build inference-only joint outputs, and attach them after geometry scoring.
- Modify `src/grounding_evaluator.py`: consume the selected parent query's calibrated mask while preserving legacy parity diagnostics.
- Modify `models/mcln.py` only in conditional Stage 2: expose or train mask-specific modules without changing the detector path.

## Task 1: Pure Mask Targets And Query Identity

**Files:**
- Create: `models/rec_joint_box_mask.py`
- Create: `tests/test_rec_joint_box_mask.py`

- [ ] **Step 1: Write failing tests for strict mask metrics and source fusion**

```python
def test_compute_mask_candidate_targets_uses_strict_thresholds():
    text = torch.tensor([[[2.0, -2.0], [2.0, -2.0]]])
    query = torch.tensor([[[-2.0, 2.0], [2.0, -2.0]]])
    gt = torch.tensor([[1, 0]], dtype=torch.bool)
    out = compute_mask_candidate_targets(
        text, query, torch.tensor([0.5]), gt,
        torch.tensor([[True, True]]), torch.tensor([0.0]))
    assert out["ious"].shape == (1, 2, 3, 1)
    assert out["source_names"] == ("text", "query", "fused")
    assert out["ious"][0, 1, 2, 0].item() == 1.0
    assert out["hits050"][0, 1, 2, 0].item() is True


def test_flat_variant_maps_to_parent_query():
    flat = torch.tensor([0, 6, 7, 111])
    assert torch.equal(flat_to_parent_query(flat, 7), torch.tensor([0, 0, 1, 15]))
```

- [ ] **Step 2: Run the tests and verify the missing API failure**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_joint_box_mask.py -q`

Expected: collection fails because `models.rec_joint_box_mask` does not exist.

- [ ] **Step 3: Implement validated vectorized helpers**

```python
MASK_SOURCE_NAMES = ("text", "query", "fused")
JOINT_MASK_SCHEMA_VERSION = "rec-joint-box-mask-v1"


def flat_to_parent_query(flat_indices, variant_count):
    if flat_indices.dtype != torch.long:
        raise TypeError("flat_indices must use int64")
    if variant_count <= 0:
        raise ValueError("variant_count must be positive")
    return torch.div(flat_indices, variant_count, rounding_mode="floor")


def fuse_mask_logits(text_logits, query_logits, alpha):
    if text_logits.shape != query_logits.shape or text_logits.dim() != 3:
        raise ValueError("mask logits must share shape [B,K,S]")
    alpha = alpha.to(text_logits).reshape(-1, 1, 1).clamp(0.0, 1.0)
    return alpha * text_logits + (1.0 - alpha) * query_logits
```

Implement `compute_mask_candidate_targets` with boolean intersection/union,
strict `> 0.25` and `> 0.50`, invalid-candidate masking, explicit empty-union
rejection, finite checks, and stable `(B,K,3,T)` output tensors.

- [ ] **Step 4: Add rejection and disabled-parity tests**

Test malformed shapes, non-finite logits, a row with no valid candidate,
non-bool GT, and exact equality between `fuse_mask_logits` and the current
`alpha * text + (1-alpha) * query` expression.

- [ ] **Step 5: Run focused tests**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_joint_box_mask.py -q`

Expected: all tests pass.

- [ ] **Step 6: Record source checkpoint**

Run: `sha256sum models/rec_joint_box_mask.py tests/test_rec_joint_box_mask.py`

Expected: two stable digests copied into the Stage 0 development log.

## Task 2: Joint Oracle And Headroom Summary

**Files:**
- Modify: `models/rec_joint_box_mask.py`
- Modify: `tests/test_rec_joint_box_mask.py`
- Create: `tests/test_audit_scanrefer_joint_box_mask.py`

- [ ] **Step 1: Write a failing synthetic Pareto-oracle test**

```python
def test_joint_oracle_improves_mask_without_breaking_box_tiers():
    box = torch.tensor([[[0.60], [0.55], [0.20]]])
    mask = torch.tensor([[0.30, 0.80, 0.95]])
    out = select_joint_oracle(box, mask, torch.tensor([0]))
    assert out["selected_parent_query"].item() == 1
    assert out["selected_box_iou"].item() > 0.50
    assert out["selected_mask_iou"].item() == pytest.approx(0.80)
```

- [ ] **Step 2: Verify the missing-function failure**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_joint_box_mask.py::test_joint_oracle_improves_mask_without_breaking_box_tiers -q`

Expected: FAIL because `select_joint_oracle` is undefined.

- [ ] **Step 3: Implement exact box-tier preservation**

```python
def iou_tier(iou):
    return (iou > 0.25).long() + (iou > 0.50).long()


def select_joint_oracle(box_ious, mask_ious, baseline_flat_indices):
    # Flatten KxG, retain candidates whose tier is >= the baseline tier,
    # then sort by (-mask_iou, -box_iou, flat_index).
    flat_box = box_ious.reshape(box_ious.shape[0], -1)
    flat_mask = mask_ious.reshape(mask_ious.shape[0], -1)
    baseline_box = flat_box.gather(1, baseline_flat_indices[:, None]).squeeze(1)
    baseline_tier = iou_tier(baseline_box)
    eligible = iou_tier(flat_box) >= baseline_tier[:, None]
    eligible[:, baseline_flat_indices] = True
    ranked_mask = flat_mask.masked_fill(~eligible, float("-inf"))
    selected = torch.argmax(ranked_mask, dim=1)
    selected_mask = flat_mask.gather(1, selected[:, None]).squeeze(1)
    selected_box = flat_box.gather(1, selected[:, None]).squeeze(1)
    return {
        "selected_flat_index": selected,
        "selected_parent_query": torch.div(selected, box_ious.shape[-1], rounding_mode="floor"),
        "selected_box_iou": selected_box,
        "selected_mask_iou": selected_mask,
        "fix025": (selected_mask > 0.25) & ~(baseline_box > 0.25),
        "fix050": (selected_mask > 0.50) & ~(baseline_box > 0.50),
        "break025": (selected_box > 0.25) & (baseline_box <= 0.25),
        "break050": (selected_box > 0.50) & (baseline_box <= 0.50),
    }
```

The implementation must return selected flat index, parent query, box/mask IoU,
and per-threshold fix/break flags. A sample with no alternative retains the
baseline index.

- [ ] **Step 4: Implement aggregate summaries and pre-registered gate**

```python
def stage0_gate(summary):
    return {
        "pass": (
            summary["delta_mask_acc050"] >= 0.03
            and summary["delta_mask_miou"] >= 0.04
            and summary["delta_position_acc025"] >= 0.0
            and summary["delta_position_acc050"] >= 0.0
        ),
        "thresholds": {
            "delta_mask_acc050": 0.03,
            "delta_mask_miou": 0.04,
            "delta_position_acc025": 0.0,
            "delta_position_acc050": 0.0,
        },
    }
```

- [ ] **Step 5: Test ties, strict IoU boundaries, invalid variants, and gate failure**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_joint_box_mask.py tests/test_audit_scanrefer_joint_box_mask.py -q`

Expected: all tests pass.

## Task 3: Deterministic Stage 0 Replay CLI

**Files:**
- Create: `scripts/audit_scanrefer_joint_box_mask.py`
- Modify: `tests/test_audit_scanrefer_joint_box_mask.py`

- [ ] **Step 1: Write failing CLI and provenance tests**

```python
def test_defaults_are_the_approved_panel():
    args = parse_args(["--checkpoint", "base.pth", "--train_cache", "cache",
                       "--output_dir", "out"])
    assert args.scene_count == 64
    assert args.expressions_per_scene == 16
    assert args.selection_seed == 0
    assert args.logit_thresholds == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_receipt_rejects_validation_split():
    with pytest.raises(ValueError, match="train"):
        validate_manifest({"split": "val"}, "a" * 64)
```

- [ ] **Step 2: Run and confirm failure**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_audit_scanrefer_joint_box_mask.py -q`

Expected: FAIL because the new script is absent.

- [ ] **Step 3: Implement CLI by reusing existing replay primitives**

Import deterministic panel and replay helpers from
`scripts.audit_scanrefer_mask_geometry`, candidate construction from
`models.rec_candidate_adapter`, mask-logit gathering from
`models.rec_mask_geometry`, and protected parent/geometry runtime builders from
`train_dist_mod`. Do not duplicate dataset construction or annotation parsing.

For every replay batch:

```python
with torch.inference_mode():
    end_points = model(inputs)
    parent = build_rec_reranker_outputs(end_points, inputs, parent_model,
                                        parent_artifact)
    geometry = build_rec_geometry_runtime_outputs(
        end_points, inputs, parent, geometry_model, geometry_artifact)
    targets = compute_replay_mask_targets(end_points, inputs, parent, geometry)
    rows.extend(compact_joint_rows(end_points, inputs, parent, geometry, targets))
```

Reject any inference feature mapping that contains `gt_masks`,
`candidate_ious`, `center_label`, `size_gts`, or `box_label_mask` before targets
are attached in the explicit target-only boundary.

- [ ] **Step 4: Implement atomic outputs**

Write `selection.json`, `rows.pt`, `summary.json`, `stdout.log`, and
`manifest.json` to a staging directory. Hash every output, publish by rename,
and leave `population_estimate=false`. Include protected artifact snapshots,
dataset/cache digests, selected scene IDs, source snapshot digest, elapsed time,
and `validation_data_accessed=false`.

- [ ] **Step 5: Test interrupted staging, overwrite rejection, and schema drift**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_audit_scanrefer_joint_box_mask.py -q`

Expected: all tests pass.

- [ ] **Step 6: Run the existing REC regression tests**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_mask_geometry.py tests/test_audit_scanrefer_mask_geometry.py tests/test_rec_geometry_runtime.py -q`

Expected: all tests pass unchanged.

## Task 4: Execute Stage 0 Smoke And Approved Panel

**Files:**
- Output only: `/root/autodl-tmp/DATA_ROOT/output/scanrefer_joint_box_mask/<run_id>/stage0_*`

- [ ] **Step 1: Verify protected inputs and free capacity**

Run exact `stat` and `sha256sum` checks for the three protected artifacts and
`df -h /root/autodl-tmp`.

Expected: modes `444`, approved digests, and enough room for smoke outputs.

- [ ] **Step 2: Run a four-expression smoke replay**

Run the new CLI with `--scene_count 1 --expressions_per_scene 4`, batch size 4,
seed 0, and a new output directory.

Expected: four rows, no validation access, exact candidate-cache parity, and a
well-formed non-deployable Stage 0 receipt.

- [ ] **Step 3: Repeat smoke and compare content hashes**

Expected: row tensors, selection, and metric summaries are identical after
excluding timestamps and elapsed time.

- [ ] **Step 4: Run the approved 64x16 panel**

Use the protected epoch-71 checkpoint, parent reranker, and geometry reranker.
Do not pass any validation cache or annotation path.

Expected: up to 1,024 rows and a definitive `stage0_gate.pass` value.

- [ ] **Step 5: Branch only on the pre-registered gate**

If pass, continue to Task 5. If fail, skip Tasks 5-10 and execute Task 11. Do
not change thresholds after seeing the result.

## Task 5: Full-Train Joint Cache

**Files:**
- Create: `scripts/cache_scanrefer_joint_box_mask.py`
- Create: `tests/test_cache_scanrefer_joint_box_mask.py`

- [ ] **Step 1: Write failing capacity and row-schema tests**

```python
def test_capacity_preflight_requires_four_gib_reserve():
    estimate = estimate_cache_capacity(1024, 256 << 20, 10 << 30)
    assert estimate["projected_bytes"] == 9 * (1 << 30)
    assert estimate["can_materialize"] is False


def test_cache_row_has_no_validation_or_runtime_gt_fields():
    assert set(row) == APPROVED_TRAIN_ROW_KEYS
    assert not (set(row) & INFERENCE_FORBIDDEN_KEYS)
```

- [ ] **Step 2: Implement sharded materialization**

Store all 36,665 train expressions in original dataset order. One row contains
float32 joint features and labels, float16 text/query superpoint logits,
compressed bool GT superpoint mask, validity, scan ID, target ID, dataset index,
and candidate/variant identities. A manifest binds every shard to the Stage 0
schema and protected hashes.

- [ ] **Step 3: Implement deterministic streaming fallback**

When projected cache size plus 4 GiB exceeds free space, emit a signed
`storage_decision.json` and use replay-backed iterable batches. Never select a
smaller population based on metrics.

- [ ] **Step 4: Run tests and a one-shard smoke**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_cache_scanrefer_joint_box_mask.py -q`

Expected: all tests pass; smoke shard reloads with exact identities and tensor
shapes.

- [ ] **Step 5: Materialize or register streaming for all train rows**

Expected: exactly 36,665 unique dataset indices, all 562 train scenes, no
validation identities, and complete shard hashes.

## Task 6: Multi-Task Adapter And Mask Calibration

**Files:**
- Modify: `models/rec_joint_box_mask.py`
- Modify: `tests/test_rec_joint_box_mask.py`

- [ ] **Step 1: Write failing model-shape and bounded-calibration tests**

```python
def test_joint_adapter_outputs_exact_contract():
    model = JointBoxMaskAdapter(input_dim=186, hidden_dim=128, dropout=0.1)
    out = model(torch.zeros(2, 16, 7, 186), torch.ones(2, 16, 7,
                    dtype=torch.bool))
    assert set(out) == {"box_logits", "mask_iou", "mask_logits",
                        "log_scale", "calibration"}
    assert out["box_logits"].shape == (2, 16, 7, 2)
    assert out["mask_logits"].shape == (2, 16, 7, 2)
    assert out["calibration"].shape == (2, 16, 5)
```

- [ ] **Step 2: Implement the v1 architecture**

Use per-variant projection `Linear(186,128) -> LayerNorm -> GELU`, masked
query/variant mean and max context, `Linear(384,128) -> GELU -> Dropout(0.1)`,
and separate linear heads. Calibration is per parent query and predicts exactly
five values: residual alpha, two log temperatures, bias, and threshold.

- [ ] **Step 3: Implement exact disabled path and bounded transforms**

```python
fused = weight * (text / t_text) + (1.0 - weight) * (query / t_query) + bias
binary = fused > threshold
```

Bounds: weight `[0,1]`, temperatures `[0.25,4.0]`, bias `[-2,2]`, threshold
`[-1,1]`. Disabled mode must be bitwise equal to current fusion and threshold
zero.

- [ ] **Step 4: Implement multi-task losses**

Return named box/mask BCE, IoU Huber, focal, Dice, ranking, and switch-risk
terms. Tests must prove invalid candidates contribute zero and each term is
finite with backward gradients.

- [ ] **Step 5: Run pure model tests**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_joint_box_mask.py -q`

Expected: all tests pass on CPU and CUDA when available.

## Task 7: Scene-Disjoint Trainer, Conformal Gate, And Artifact

**Files:**
- Create: `scripts/train_scanrefer_joint_box_mask.py`
- Create: `tests/test_train_scanrefer_joint_box_mask.py`

- [ ] **Step 1: Write failing split and policy tests**

Test exact 5-fold scene membership, disjoint 90/10 fit-calibration scenes,
stable seed-0 digests, exactly three policy names, and rejection of validation
manifests.

- [ ] **Step 2: Implement OOF training**

Train one adapter per scene fold with fixed v1 hyperparameters. Record every
epoch's untouched fold metrics but select epoch only through the designated
train calibration partition. Use deterministic seeds and gradient clipping.

- [ ] **Step 3: Implement one-sided conformal box-delta bounds**

```python
eligible = (
    candidate_delta_lcb025 >= 0.0
    and candidate_delta_lcb050 >= 0.0
)
```

Compute residual quantiles from calibration scenes only. Test monotonicity,
finite bounds, stable ties, and exact fallback when no candidate is eligible.

- [ ] **Step 4: Implement the publication gate**

Require calibration deltas: Position@0.25 >= 0, Position@0.50 >= 0,
Mask@0.25 >= 0, Mask@0.50 >= +0.02, mIoU >= +0.03, and non-negative
scene-block bootstrap lower bounds for both Position deltas. A failure writes
`selected="baseline"` and no deployable adapter.

- [ ] **Step 5: Implement exact artifact validation**

Artifact fields include schema, state dict, feature names, normalization,
architecture, loss weights, split digests, OOF metrics, conformal quantile,
policy, protected hashes, source snapshot hash, and
`selection_uses_validation=false`.

- [ ] **Step 6: Run trainer tests and deterministic tiny training**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_train_scanrefer_joint_box_mask.py -q`

Expected: all tests pass; two tiny runs produce identical selections and model
tensor hashes.

- [ ] **Step 7: Train on the full train population**

Expected: OOF receipt, untouched calibration receipt, and either one selected
read-only artifact or an explicit baseline selection.

## Task 8: Runtime Attachment

**Files:**
- Modify: `main_utils.py`
- Modify: `train_dist_mod.py`
- Create: `tests/test_rec_joint_box_mask_runtime.py`

- [ ] **Step 1: Write failing parser and runtime-contract tests**

Add `--rec_joint_box_mask_checkpoint` and
`--eval_use_rec_joint_box_mask_scores`. Test that enablement requires parent,
geometry, and joint checkpoints and both existing reranker flags.

- [ ] **Step 2: Implement one-time loading and provenance validation**

Load the artifact once per process, bind it to actual protected model hashes,
and reject partial state. Keep the adapter in float32 eval mode with gradients
disabled and outer autocast disabled.

- [ ] **Step 3: Build exact inference output**

Return only versioned inference fields: selected flat index, parent query index,
selected box, selected scores, calibrated superpoint logits, threshold,
validity/fallback, and `inference_uses_ground_truth=false`. Reject any
ground-truth-only field before validation.

- [ ] **Step 4: Test malformed fields and exact baseline bypass**

Mutate every output field for shape, dtype, device, NaN, range, and mapping
errors. Disabled mode must not call the adapter and must leave existing
geometry outputs byte-for-byte unchanged.

- [ ] **Step 5: Run runtime regression suite**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_rec_joint_box_mask_runtime.py tests/test_rec_geometry_runtime.py tests/test_rec_reranker_runtime.py -q`

Expected: all tests pass.

## Task 9: Query-Consistent Evaluator

**Files:**
- Modify: `src/grounding_evaluator.py`
- Create: `tests/test_grounding_evaluator_rec_joint_box_mask.py`

- [ ] **Step 1: Write a failing same-query test**

Construct two detector queries where legacy semantic selection chooses query 0,
the joint payload chooses geometry variant 8 whose parent is query 1, and only
query 1 has the correct mask. Assert that Position and formal mask metrics both
use the joint selection while legacy diagnostic counters retain query 0.

- [ ] **Step 2: Implement formal joint mask evaluation**

When enabled on `last_`, validate the joint payload, map calibrated superpoint
mask to points through `superpoints`, apply the per-row strict logit threshold,
and update semantic mIoU, overall25/50, unique/multiple, easy/hard, and
view-dependent counters exactly once. Other decoder prefixes keep legacy
behavior.

- [ ] **Step 3: Preserve metric formulas and prevent double counting**

Test strict IoU comparisons, mean aggregation, distributed counter merge, and
that enabling joint evaluation does not call the legacy formal semantic mask
path a second time.

- [ ] **Step 4: Run evaluator and full focused tests**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_grounding_evaluator_rec_joint_box_mask.py tests/test_grounding_evaluator_rec_geometry.py tests/test_grounding_evaluator_rec_reranker.py -q`

Expected: all tests pass.

## Task 10: Stage 1 Train-Only Gate And Smoke Runtime

**Files:**
- Output only under a new `scanrefer_joint_box_mask/<run_id>` directory.

- [ ] **Step 1: Run a frozen-backbone runtime smoke**

Use four train expressions and compare adapter-off outputs to the protected
baseline exactly. Enable the adapter and verify selected box/query/mask mapping,
finite logits, no GT fields, and deterministic repeat hashes.

- [ ] **Step 2: Evaluate untouched train calibration scenes**

Generate the publication-gate receipt with exact hit counts, deltas, scene-block
bootstrap bounds, policy, and conformal quantile.

- [ ] **Step 3: Branch only on the gate**

If selected artifact exists, continue to Task 12. If `selected="baseline"`,
execute Task 11. Do not inspect formal validation to choose this branch.

## Task 11: Conditional Mask-Head Fine-Tuning

**Files:**
- Conditional modify: `models/mcln.py` only when Stage 1's pre-registered
  train-only gate fails and Stage 2 requires explicit mask-module groups.
- Create: `scripts/train_scanrefer_joint_mask_head.py`
- Create: `tests/test_train_scanrefer_joint_mask_head.py`

- [ ] **Step 1: Write parameter-isolation tests**

Assert that the first experiment enables gradients only for the query-mask
projection/output modules and joint adapter. Detector backbone, transformer,
text encoder, and all box heads must be frozen, and their tensors must remain
bitwise equal after one optimizer step.

- [ ] **Step 2: Train query-mask modules only**

Use mask focal/Dice, text-query consistency, calibration, and box-risk losses.
Rebuild the train feature cache because mask statistics changed; do not reuse
old parent calibration blindly.

- [ ] **Step 3: Apply the same OOF/calibration gate**

If the query-mask experiment fails, run one separately recorded text-mask
branch experiment. Do not unfreeze both branches in the first attempt and do
not tune on validation.

- [ ] **Step 4: Publish only a gate-passing artifact**

Expected: either a selected read-only mask-head plus joint adapter bundle or an
explicit baseline result. Return to Task 8 runtime verification for any new
bundle before Task 12.

## Task 12: Official Runner And Reproducibility Receipt

**Files:**
- Create: `scripts/run_frozen_rec_joint_box_mask_official.py`
- Create: `tests/test_run_frozen_rec_joint_box_mask_official.py`

- [ ] **Step 1: Write failing command and preservation tests**

Assert exact 9,508 validation population, batch/runtime contract, all required
flags, protected checkpoint paths, a new log directory, and rejection of an
existing claim or result receipt.

- [ ] **Step 2: Implement atomic one-shot launcher**

Snapshot protected artifacts and source, create a claim, run official
evaluation, parse exact integer hits and mask sums, record stdout/log/config
hashes, verify `inference_uses_ground_truth=false`, then compare protected
states after completion.

- [ ] **Step 3: Implement the final acceptance audit**

```python
accepted = (
    hits_position025 >= 5610
    and hits_position050 >= 4621
    and hits_mask025 >= 5582
    and hits_mask050 >= 4821
    and mask_miou > 0.4472
    and not inference_uses_ground_truth
    and protected_before == protected_after
)
```

- [ ] **Step 4: Run tests and source snapshot verification**

Run: `/root/miniconda3/envs/bdetr/bin/python -m pytest tests/test_run_frozen_rec_joint_box_mask_official.py -q`

Expected: all tests pass; snapshot manifest covers every changed/imported file.

- [ ] **Step 5: Run the full focused regression suite**

Run all new tests plus existing REC candidate, mask geometry, geometry runtime,
and evaluator suites.

Expected: all tests pass with no protected-file changes.

- [ ] **Step 6: Execute one formal validation for the train-gated artifact**

Do not alter the artifact after this run. If it fails, preserve the report and
return to a documented train-only design amendment; do not tune against the
validation result.

- [ ] **Step 7: Seal successful artifacts**

On full acceptance only, set selected model bundles, source snapshot manifest,
command/config, and final receipt to `0444`. Recompute hashes and produce a
single reproduction command. Keep the previous protected baseline unchanged.

## Final Completion Audit

- [ ] One official receipt proves Position hits >= 5,610 and >= 4,621.
- [ ] The same receipt proves mask hits >= 5,582 and >= 4,821.
- [ ] The same receipt proves semantic mIoU > 0.4472.
- [ ] The same receipt proves no ground-truth inference access.
- [ ] All protected artifact paths, modes, sizes, and SHA-256 digests match.
- [ ] New selected artifacts and reproducibility files are read-only and hashed.
- [ ] The exact source snapshot, environment, data identities, split digests,
      command, and random seeds are present.
- [ ] Focused and regression tests pass from the recorded environment.
- [ ] Only then call `update_goal(status="complete")`.
