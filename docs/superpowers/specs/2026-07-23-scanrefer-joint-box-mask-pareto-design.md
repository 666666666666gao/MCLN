# ScanRefer Joint Box-Mask Pareto Design

## Status And Decision

This design was approved by the user on 2026-07-23.

The selected direction is a single-backbone, single-forward ScanRefer system
that makes box and mask decisions on the same detector query. It adds a
train-only mask headroom audit, a frozen-backbone joint quality adapter, and a
risk-controlled Pareto selector. Mask-head fine-tuning is a gated second stage,
not the first experiment. A second backbone is out of scope unless train-only
evidence later proves that the approved single-backbone design lacks enough
headroom and the user explicitly expands the scope.

The current deployed system remains immutable until a new system passes every
gate in this document. Failed experiments must not replace or modify it.

The repository has no Git metadata. The implementation must not initialize a
repository. Reproducibility therefore uses an immutable source snapshot,
content hashes, configuration, commands, environment metadata, and receipts
instead of a Git commit identifier.

## Formal Objective

The authoritative ScanRefer validation population contains 9,508 expressions.
One final system must satisfy all five requirements in the same official run:

| Metric | Required result | Count form where applicable |
| --- | ---: | ---: |
| Position Acc@0.25 | at least 59.00% | at least 5,610 / 9,508 |
| Position Acc@0.50 | no lower than current best | at least 4,621 / 9,508 |
| Mask Acc@0.25 | strictly greater than 58.70% | at least 5,582 / 9,508 |
| Mask Acc@0.50 | strictly greater than 50.70% | at least 4,821 / 9,508 |
| Mask semantic mIoU | strictly greater than 44.72% | greater than 0.4472 |

The current protected formal result is:

- Position Acc@0.25: 5,542 / 9,508 = 58.2878%.
- Position Acc@0.50: 4,621 / 9,508 = 48.6012%.
- Mask Acc@0.25: 59.6971%.
- Mask Acc@0.50: 49.0324%.
- Mask semantic mIoU: 41.7676%.
- `inference_uses_ground_truth=false`.

The new design therefore needs at least 68 additional Position Acc@0.25 hits,
must lose no Position Acc@0.50 hits, and must materially improve mask Acc@0.50
and semantic mIoU. Passing only a subset of these requirements is failure.

## Protected Baseline Contract

The following artifacts are inputs only and must remain mode `0444` with the
listed SHA-256 digests:

| Artifact path | SHA-256 |
| --- | --- |
| `/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth` | `3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208` |
| `/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/artifacts/reranker_h256_d010_lr1e3_seed0_final_contract.pth` | `f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b` |
| `/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/geometry_artifacts/selected_geometry_reranker.pth` | `835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f` |

Every experiment launcher must fingerprint these files before and after work.
It must abort before training if a digest, mode, size, or path differs. New
artifacts are written to new directories and never to a protected path.

## Evidence And Root Cause

The design addresses four verified structural problems.

1. Checkpoint preservation ranks epochs by Position Acc@0.25 or Acc@0.50.
   Mask metrics do not participate in the preservation decision.
2. Parent and geometry rerankers are supervised by box IoU only.
3. The box evaluator consumes the deployed parent/geometry scores, while the
   two mask evaluators independently select a query from legacy semantic or
   contrastive scores. A selected geometry box and evaluated mask therefore
   need not describe the same query.
4. The text-derived mask is one expression-level mask expanded across all 256
   detector queries. Only the superpoint query mask varies by query, and the
   existing fusion weight is one scalar for the sample. This restricts both
   query-specific mask expressiveness and calibration.

A read-only inventory of all 43 complete local mask evaluations found no
hidden checkpoint that solves the problem. The historical maxima are about
59.7076% Mask Acc@0.25, 49.0429% Mask Acc@0.50, and 41.7754% semantic mIoU.
The official 44.72% checkpoint is linked from the README but its Google Drive
host is unreachable from this environment. The approved design must not depend
on that external file.

## Scope And Non-Goals

In scope:

- One existing MCLN backbone forward pass per expression.
- Existing 16 parent queries and their 112 geometry variants.
- Query-consistent box and mask selection.
- A lightweight joint quality model and mask-logit calibration adapter.
- Optional fine-tuning of mask-specific modules after an explicit headroom
  gate, while preserving the box objective through the joint selector.
- Train-only model selection with scene-disjoint fit and calibration splits.

Out of scope for the first implementation cycle:

- A second backbone or a second full MCLN forward pass.
- Validation-driven threshold, loss-weight, or checkpoint selection.
- Ground-truth boxes, masks, target IDs, or IoUs at inference time.
- Repeating the failed box-only hierarchical reranker experiment.
- Changing ScanRefer annotations, validation membership, or metric formulas.
- Overwriting, renaming, chmoding, or deleting protected artifacts.

## Stage 0: Train-Only Mask Headroom Audit

### Purpose

Before training a new selector, measure whether the existing raw mask logits
contain enough recoverable quality. The audit decides whether Stage 1 can use a
frozen backbone or whether Stage 2 is necessary.

### Population And Leakage Control

The audit reuses the deterministic train-cache replay mechanism. Its first
gate panel uses seed 0, 64 train scenes, and at most 16 expressions per scene,
for at most 1,024 expressions. Selection is baseline-stratified across
easy/hard, unique/multiple, and view-dependent/view-independent groups. Panel
construction and all subsequent splits use scan IDs, so expressions from one
scene cannot cross fit, calibration, or audit partitions.

No validation row, validation cache target, or validation metric may be read by
the audit, trainer, policy selector, or artifact publisher. Existing formal
validation results may be retained only as immutable baseline evidence; they
cannot select a new policy.

### Measurements

For each expression and each of the existing 16 parent queries, stream the
following without retaining full point masks after the row is summarized:

- Text, query, and current adaptive-fused mask logits.
- Fixed logit thresholds `-1.0`, `-0.5`, `0.0`, `0.5`, and `1.0`.
- Mask IoU, hit at 0.25, hit at 0.50, foreground fraction, confidence,
  entropy, text-query Dice, and selected superpoint count.
- Parent query index, all seven geometry variants, box IoU targets, current
  parent score, current geometry score, and current selected geometry variant.
- Current semantic-mask query, current position-mask query, parent top-1 query,
  geometry top-1 query, per-source mask oracle, per-query mask oracle, and a
  joint oracle constrained not to reduce the sample's box threshold tier.

The audit must report aggregate and subgroup metrics, query disagreement rates,
oracle fixes/breaks, Pareto front size, and the distribution of mask gains at a
fixed box tier.

### Stage 0 Gate

Stage 1 is justified only if the joint train-only oracle, relative to the
current deployed selection on the same panel, shows at least:

- 3.0 percentage points of Mask Acc@0.50 headroom;
- 4.0 percentage points of semantic mIoU headroom;
- no reduction in Position Acc@0.25 or Position Acc@0.50 under the oracle's
  per-sample box-tier constraint.

These margins exceed the current formal validation deficits and leave room for
the learned selector to fall short of its oracle. If this gate fails, skip a
frozen-only selector experiment and proceed to the separately gated mask-head
fine-tuning design in Stage 2. The audit itself never publishes a deployable
model.

If the gate passes, materialize compact Stage 1 training rows for all 36,665
train expressions. Store one text-logit vector and the 16 selected query-logit
vectors per expression in sharded float16 form, plus compressed ground-truth
superpoint masks and float32 candidate summaries. Before extraction, estimate
bytes per expression from the gate panel and require the projected cache plus
a 4 GiB safety reserve to fit. If it does not fit, use deterministic streaming
replay for training; do not reduce the train population or substitute
validation rows merely to fit storage.

## Stage 1: Joint Quality Adapter

### Candidate Identity

The deployable candidate identity is `(parent_query_index, geometry_variant)`.
All seven variants of one parent query share the same query-specific mask. A
selected box variant is always returned with the mask derived from its parent
query. This invariant removes the current box-mask query mismatch.

### Inputs

The adapter consumes inference-available data only:

- Existing 152 parent candidate features.
- Existing 25 geometry features for each variant.
- Existing parent and geometry scores and stable ranks.
- Mask diagnostics computed from raw text and query logits without labels:
  foreground fraction, confidence, entropy, text-query Dice, adaptive alpha,
  mask/regressed-box agreement, and source-validity flags.

The runtime schema is named and versioned. Feature order, dtype, shape, and
normalization are stored in the artifact and validated exactly at load time.

### Outputs

Version 1 uses a set-aware shared encoder over the 16 parent queries and seven
geometry variants, followed by separate linear heads. The implementation plan
must fix its exact hidden dimensions, normalization, and dropout before any
training run; those values become part of the artifact schema. The encoder
produces, for each candidate:

- Probability that box IoU exceeds 0.25.
- Probability that box IoU exceeds 0.50.
- Expected mask IoU.
- Probability that mask IoU exceeds 0.25.
- Probability that mask IoU exceeds 0.50.
- A bounded log-scale uncertainty estimate used with calibration residuals by
  the risk gate.
- Bounded mask calibration parameters described below.

The quality heads are multi-task outputs from one candidate representation.
They are not independent selectors.

### Query-Specific Mask Calibration

For a parent query, the calibrated superpoint logit is:

`w * (text_logit / T_text) + (1 - w) * (query_logit / T_query) + bias`.

The adapter predicts bounded residuals around the current behavior:

- `w` is constrained to `[0, 1]` and initialized from the existing adaptive
  alpha.
- `T_text` and `T_query` are positive and bounded.
- `bias` and the final logit threshold are bounded residuals initialized to
  zero.

Disabling the adapter must reproduce the original fusion exactly: current
alpha, both temperatures equal to one, zero bias, and zero logit threshold.
This exact parity path is required for tests and fallback behavior.

### Pareto Selection Policy

Selection begins from the current deployed geometry top-1 candidate. A new
candidate may replace it only when one-sided, scene-calibrated conformal lower
bounds for both predicted box-tier deltas are non-negative. The bounds combine
the learned log scale with residual quantiles computed only from the untouched
train calibration scenes.

Among eligible non-dominated candidates, select one of exactly three
predeclared lexicographic policies using the train calibration partition:

1. Expected mask IoU, then Mask Acc@0.50 probability, then Mask Acc@0.25
   probability.
2. Mask Acc@0.50 probability, then expected mask IoU, then Mask Acc@0.25
   probability.
3. The minimum of the three predicted mask quantities, then expected mask IoU,
   then Mask Acc@0.50 probability, then Mask Acc@0.25 probability.

All policies finish ties with current deployed score and stable flat index.
No additional policy may be introduced after results are seen. The chosen
policy and conformal quantile are selected without validation outcomes. If no
candidate clears the risk gate, retain the current deployed candidate.

### Training

The backbone and all three protected artifacts are frozen. Training uses:

- Huber loss for expected box and mask IoU.
- Binary cross-entropy or focal loss for the four threshold events.
- Differentiable mask focal and Dice losses for the calibration parameters.
- A ranking loss that prefers Pareto improvements over the current candidate.
- A conservative penalty for predicted switches that break either box tier.

Five scene-disjoint folds generate out-of-fold predictions. A separate
scene-disjoint calibration partition selects the finite policy, confidence
margin, and checkpoint. The final artifact is then refit on the permitted
train population with the selected contract frozen.

### Stage 1 Publication Gate

A deployable Stage 1 artifact may be published only when the untouched
calibration scenes show all of the following against the current deployed
system evaluated on exactly the same rows:

- Position Acc@0.25 does not decrease.
- Position Acc@0.50 does not decrease.
- Mask Acc@0.25 does not decrease.
- Mask Acc@0.50 improves by at least 2.0 percentage points.
- Semantic mIoU improves by at least 3.0 percentage points.
- Bootstrap lower confidence bounds for both Position deltas are non-negative.
- No inference payload contains a ground-truth-only field.

If the gate fails, the selected artifact is `baseline`; no failed adapter is
made available to the official evaluator.

## Stage 2: Gated Mask-Head Fine-Tuning

Stage 2 runs only if Stage 0 shows that existing-logit headroom is insufficient
or Stage 1 cannot convert sufficient oracle headroom on untouched train
calibration scenes.

Only mask-specific modules may be unfrozen initially. The detector backbone,
transformer decoder, box prediction heads, text encoder, and query geometry
remain frozen. Because the existing parent reranker includes mask statistics,
any changed raw mask output requires rebuilding its train-only feature cache
and retraining the joint adapter; the old parent scores cannot be assumed to
remain calibrated.

The Stage 2 objective combines mask focal/Dice supervision, text-query
consistency, bounded calibration residuals, and the same joint box-risk loss.
One variable is changed per experiment: first the query mask head, then the
text mask branch only if the query-head experiment fails its train-only gate.
No broad end-to-end fine-tuning is allowed before those isolated experiments
are diagnosed.

Stage 2 uses the same scene-disjoint protocol and publication gate as Stage 1.
Hyperparameter search begins only after the architecture and data flow pass
their tests and train-only gates.

## Runtime And Evaluator Integration

The runtime attachment must expose an exact, versioned payload containing:

- Selected flat geometry index and selected parent query index.
- Selected box and its score.
- Selected calibrated mask logits or binary mask.
- Validity masks and deterministic fallback indices.
- A diagnostic flag proving that inference did not consume ground truth.

The official evaluator uses the selected box for both Position thresholds and
the selected parent query's calibrated mask for all three mask metrics. Legacy
position-mask and semantic-mask query selectors remain available only for
parity reports; they do not define the new system's formal mask prediction.
Mask intersection, union, strict 0.25/0.50 comparisons, and mean-IoU
aggregation remain exactly the current evaluator formulas; only the
inference-produced selected mask changes.

Malformed enabled attachments fail closed. Missing fields, schema mismatches,
NaNs, invalid indices, dtype changes, or non-finite valid scores raise an
error. An explicitly disabled adapter follows the exact protected baseline
path. There is no silent fallback from a malformed enabled artifact.

## Reproducibility Contract

Each experiment uses a new directory below:

`/root/autodl-tmp/DATA_ROOT/output/scanrefer_joint_box_mask/<run_id>/`

It contains:

- `artifacts/`: model state and normalization/schema contract.
- `receipts/`: audit, split, training, selection, and evaluation receipts.
- `logs/`: complete stdout/stderr and metric logs.
- `source_snapshot/`: copies of every implementation and launcher file used.
- `manifest.json`: hashes, sizes, modes, commands, environment, dataset
  identities, scene split digests, random seeds, and artifact relationships.

Publishing is atomic. A selected artifact and its manifest are set to `0444`
after all hashes are recorded. The receipt includes protected-artifact state
before and after the run. Source snapshot hashing replaces unavailable Git
metadata and must cover imports transitively used by the new runtime.

## Testing And Verification

Implementation follows test-driven development. Required tests include:

1. Mask IoU and threshold metrics on synthetic superpoint masks.
2. Text/query/fused logit extraction on original detector query indices.
3. Exact disabled-adapter parity with the current mask fusion.
4. Geometry-variant to parent-query mapping and query-consistent mask use.
5. Stable tie breaking and deterministic CPU/GPU serialization.
6. Feature schema, normalization, dtype, shape, and non-finite rejection.
7. Runtime rejection of ground-truth-only fields.
8. Risk-gate fallback when no candidate is eligible.
9. Evaluator integration proving one selected query drives box and mask.
10. Protected-artifact before/after hash and mode verification.
11. Scene-disjoint split and validation-exclusion tests.
12. End-to-end smoke replay with a tiny train-only panel.

After unit and integration tests pass, verification proceeds in this order:

1. Deterministic train-only Stage 0 audit.
2. Five-fold out-of-fold Stage 1 training.
3. Untouched train-scene calibration gate.
4. Stage 2 only when its trigger condition is documented.
5. One formal 9,508-expression validation run for a gate-passing artifact.
6. Final receipt audit against every metric and preservation requirement.

A formal validation failure is report-only: it cannot change that artifact's
thresholds, policy, weights, or checkpoint. Any later architectural attempt
requires a documented design amendment and new train-only evidence rather
than direct tuning against the failed validation result.

The goal is complete only when the same official validation receipt proves all
five formal metrics and the preservation/reproducibility contract. Until then,
the protected baseline remains the best deployable system.
