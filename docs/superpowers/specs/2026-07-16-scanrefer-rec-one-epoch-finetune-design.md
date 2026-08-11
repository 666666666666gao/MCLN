# ScanRefer REC One-Epoch Fine-Tuning Design

## Status And Scope

This document specializes the already-approved `Fallback 2: REC-Specific
Fine-Tuning` in
`docs/superpowers/specs/2026-07-14-scanrefer-rec-reranker-design.md`.
It does not reopen the frozen geometry experiment or use its validation result
for model selection.

The frozen official run produced 9,508 samples with:

- `Acc@0.25 = 0.58288` (5,542 hits);
- `Acc@0.50 = 0.48601` (4,621 hits);
- no inference-time ground truth.

The new system must recover at least 163 additional strict `IoU > 0.25` hits
while preserving `Acc@0.50 >= 0.47000`. It may train for at most one pass over
the ScanRefer training fit partition. It must not read ScanRefer validation
data until every new deployable artifact has been selected and frozen.

## Immutable Inputs

- Epoch-71 MCLN checkpoint:
  `/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth`
- Checkpoint SHA-256:
  `3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208`
- Parent reranker SHA-256:
  `f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b`
- Geometry reranker SHA-256:
  `835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f`
- Train split: seed 0, 562 scenes, 36,665 expressions.
- Fit partition: 506 scenes and 33,040 expressions.
- Calibration partition: 56 scenes and 3,625 expressions.
- Scene mapping SHA-256:
  `72685aa01285dbe72b9e0331acd5f10457f773e9e158ae4f884b9c4176cf95bd`.

The existing official result, sidecar, claim, receipt, selection record, and
artifacts remain immutable. The fine-tuning run writes to a new output tree.

## Architecture

### Train-only runner

Add a standalone `scripts/train_scanrefer_rec_finetune.py`. It constructs only
`Joint3DDataset(split="train", dataset_dict={"scanrefer": 1})`; it never calls
`BaseTrainTester.main`, never constructs a validation dataset, and never loads
the validation caches. It loads only model weights from epoch 71, then creates
one fresh AdamW with three explicit parameter groups and no scheduler.

The fit loader uses batch size 18, `drop_last=False`, and exactly 1,836 updates.
The fit view retains the epoch-71 ScanRefer augmentation contract, including
detector augmentation. The calibration view uses the same training annotations
but disables point-cloud and detector augmentation and iterates deterministically.
MCLN computes superpoint centers with a stable sorted segment mean using
float64 accumulation for float32 coordinates. The calibration path must not use
CUDA atomic scatter reductions, whose order-dependent mask drift can change a
near-tied parent or geometry Top-1 result.

### Trainable system

Add `models/rec_finetune.py` for pure contracts and gradient-enabled REC
forward helpers. The MCLN allowlist is exact:

- `decoder.*`;
- `decoder_query_proj.*`;
- `proposal_head.*`;
- `prediction_heads.*`.

The parent and geometry `QueryReranker` modules are also trainable. Every other
MCLN parameter is frozen. Training mode is applied by first calling
`mcln.eval()`, then setting only the four allowlisted modules to train mode.
This keeps PointNet++, RoBERTa, mask/SWA, projections, source selector, and
all other frozen dropout or normalization modules in eval mode.

The deployed candidate features and geometry are built before ground-truth
fields are attached. Parent and geometry IoUs are then computed root-only and
detached before entering `compute_rec_reranker_loss`. This prevents gradients
through the target construction while allowing the two REC losses to update
the rerankers and the allowlisted decoder/box path through deployable features.
The discrete Top-K and rank-blend operations are not used as gradient sources.

### Optimization and losses

The matcher costs are class/L1/GIoU = `1/5/2`. The existing Hungarian loss is
extended with backward-compatible `mask_loss_scale` and
`consistency_loss_scale` keyword arguments. Defaults remain 1.0 so existing
training is unchanged.

For the fine-tuning runner:

- supervised main-mask, superpoint-mask, and adaptive-mask terms are multiplied
  by `mask_loss_scale=0.1`;
- corresponding-mask terms are multiplied by
  `consistency_loss_scale=0.1`;
- source-choice loss has weight 0;
- the parent REC ranking loss retains its single-best listwise objective;
- the geometry REC ranking loss uses best-tier pairwise logistic loss, with all
  valid candidates in the highest reachable strict `0.25/0.50` tier treated as
  positives and only lower-tier candidates treated as negatives;
- each geometry positive/negative pair contributes
  `softplus(negative_logit - positive_logit)`; pairs are averaged within each
  row, then informative rows are averaged with equal row weight; a batch with
  no cross-tier pair returns a differentiable zero;
- ranking/threshold/IoU weights remain `1/1/0.5`, and parent and geometry REC
  losses each have global weight 1.

The geometry objective metadata records the strict thresholds, comparison
operator, positive and negative policies, loss direction, both reduction
levels, and the differentiable-zero policy. The parent alpha is a literal
`0.0` and the geometry alpha is a literal `1.0` at the two loss call sites;
published metadata is a fresh defensive value and cannot rewrite the executed
alphas through a shared mutable dictionary.

The AdamW groups are:

| Group | Learning rate | Weight decay | Gradient clip |
| --- | ---: | ---: | ---: |
| MCLN allowlist | `2e-5` | `5e-4` | `0.1` |
| Parent reranker | `1e-3` | `1e-4` | `1.0` |
| Geometry reranker | `3e-4` | `1e-4` | `1.0` |

Learning rates remain constant. The epoch-71 optimizer and scheduler are never
loaded.

The parent and geometry feature mean/std tensors are copied from the initial
artifacts and remain fixed. Their deployed blend weights also remain fixed at
parent `0.9` and geometry `1.0`; the fallback selects a training step, not new
hyperparameters.

## Calibration And Selection

Run an unaugmented train-calibration pass at steps:

`0, 306, 612, 918, 1224, 1530, 1836`.

The batch size is 18 and TF32 follows the deployed CUDA contract. Every pass
uses the same fixed ordering, root-only strict thresholds, fixed normalization,
fixed blend weights, and stable tie policy.

Step 0 defines the baseline. A later step is eligible only when both
calibration accuracies are greater than or equal to the step-0 values. Eligible
steps are ranked by:

```text
min(Acc@0.25 / 0.60, Acc@0.50 / 0.47)
    + 0.1 * (Acc@0.25 + Acc@0.50)
```

Ties keep the earliest step. A calibration regression is either:

- either threshold falling below step 0; or
- the composite score falling strictly below the preceding calibration pass.

The first regression stops training immediately and restores the best eligible
snapshot. The final partial batch counts as the 1,836th update; the runner can
never exceed that bound.

## Artifact And Runtime Contract

The old cache-trained artifact schemas are not relabeled or forged. The runner
publishes, in order:

1. a deployable MCLN checkpoint and its SHA-256;
2. a new fine-tuned parent artifact bound to that checkpoint;
3. a new fine-tuned geometry artifact bound to both the checkpoint and parent
   artifact SHA-256;
4. a canonical train-only selection record binding all initial and final
   artifacts, scene mapping, allowlist, losses, optimizer groups, calibration
   history, selected step, code hashes, and no-validation-data declaration.

The online fine-tune parent artifact, geometry artifact, and selected-step
checkpoint metadata retain their `v2` schemas. The selection record and smoke
receipt use `v3` because they add exact selected-output digests and independent
reproduction diagnostics. No successful older online fine-tune publication
exists.

The code manifest includes the runner, both REC loss/forward modules, candidate
adapter, mask-geometry target builder, geometry input builder, source-choice
IoU kernel, artifact loaders, Hungarian losses, and deployed runtime. It is
captured before model/data initialization and must match again before any smoke
receipt or deployable directory is committed.

The new artifact schemas explicitly describe online one-epoch fine-tuning and
the fixed normalization lineage. `train_dist_mod.py` dispatches to their strict
loader while preserving the existing cache-trained loader unchanged. Both
artifact families feed the same deployable candidate and geometry builders.

All files are written atomically. Inputs and already-frozen outputs are never
overwritten. Selection, restored-state reproduction, and staged-artifact reload
must have exactly equal calibration metrics and exactly equal SHA-256 digests
over ordered `(dataset_index, selected_iou.float.hex())` rows. Their aggregate
diagnostics are each fully validated and retained but are not required to be
byte-equal, because oracle-only floating observations are not deployable
selection output. The staged reload metrics, diagnostics, and digest are written
into the selection record before all files are made read-only.

## Testing

Focused tests must prove:

- exact scene split metadata and 1,836-step natural remainder;
- no construction or access of ScanRefer validation data;
- exact parameter allowlist and frozen eval-mode invariant;
- fresh three-group AdamW, constant learning rates, and per-group clips;
- matcher `1/5/2`, loss scaling, and source-choice weight zero;
- exact pairwise direction and row-balanced reduction, strict threshold tiers,
  invalid-alpha rejection, and detached coverage counts;
- GT isolation, detached IoU targets, and gradient flow only to allowed modules;
- step-0 gate, regression stop, earliest tie, and snapshot restoration;
- exact ordered selected-output digests across selection, restore, and staged
  reload, while independently validating legal oracle-only diagnostic drift;
- exact repeatability and bounded numerical error of the deterministic
  superpoint segment mean on production-shaped CUDA inputs;
- new artifact SHA bindings, tamper rejection, atomic round trip, and exact
  runtime parity with the trainable forward in eval mode;
- compatibility of all existing frozen parent/geometry artifacts and tests.

Training diagnostics publish listwise, best-tier pairwise, and routed ranking
losses plus informative-row, pair, positive, and negative counts for both
rerankers. A run with no geometry informative pair is invalid and cannot be
published even if finite zero gradients are present.

A one-batch GPU smoke must pass before the full training launch. It may use
only training data and writes to a disposable output directory.

## Final Evaluation

After train-only selection, a separate one-shot official launcher binds the new
checkpoint, parent artifact, geometry artifact, selection record, interpreter,
runtime code, configuration, and logs. It uses the same no-GT runtime contract:
`butd=true`, `butd_gt=false`, `butd_cls=false`.

The project is complete only if a new full 9,508-sample official ScanRefer
validation run reports:

- `last_ position alignment Acc0.25 Top-1 >= 0.60000`;
- `last_ position alignment Acc0.50 Top-1 >= 0.47000`;
- `inference_uses_ground_truth=false`.

Train calibration, a smoke run, cache metrics, or the prior official result do
not satisfy this gate.
