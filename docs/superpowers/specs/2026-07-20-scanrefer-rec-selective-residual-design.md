# ScanRefer REC Selective Residual Design

## Status And Decision

This design is the next train-only experiment for the ScanRefer REC objective.
The current immutable deployment remains the best system. The experiment may
publish a new artifact only after every gate below passes; otherwise the
baseline is restored and retained.

The selected approach is a **baseline-preserving selective residual reranker**.
It learns when an existing geometry Top-1 decision should be replaced by one
of the already available candidates. It does not change the MCLN backbone,
candidate construction, predicted boxes, parent reranker, geometry variants,
or the fixed candidate axis during the first experiment.

The formal target remains a fresh 9,508-sample official evaluation with:

- strict `IoU > 0.25` accuracy at least `0.60000`;
- strict `IoU > 0.50` accuracy at least `0.47000`;
- no inference-time ground truth.

The current immutable inputs are the epoch-71 backbone and the existing parent
and geometry artifacts. Their SHA-256 values are bound in the experiment
receipt and their files must remain read-only:

```text
backbone  3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208
parent    f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b
geometry  835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f
```

## Evidence And Motivation

The authoritative train calibration partition has 3,625 expressions. At the
frozen step-0 state, geometry Top-1 obtains `3,461/3,625` hits at `0.25` and
`3,316/3,625` at `0.50`. The geometry candidate oracle obtains `3,606` and
`3,588` hits, so 145 of the 164 `0.25` misses are recoverable from the existing
candidate set. The remaining errors are mostly wrong-query decisions rather
than missing boxes. The raw 256-query oracle is unchanged by the rejected
source-gate probe, which rules out a simple box-coverage explanation.

The older frozen-cache geometry selection record reports `3,461/3,315`; the
authoritative online deterministic calibration reports `3,461/3,316`. This
known one-row replay difference is not silently normalized. Cache cross-fit is
compared with the cache baseline, followed by a separate online calibration
reproduction compared with `3,461/3,316` before publication.

Several ordinary global rerankers and a joint fine-tune have already shown
regressions. The next method therefore limits its intervention surface: the
frozen geometry selection is the default action, and a learned residual may
only override it when an out-of-fold calibrated confidence rule predicts a
positive net change at both thresholds.

## Data Boundary And Cross-Fit Protocol

Only the completed `train` candidate cache and `geometry_train` sidecar may be
opened. Both are joined by contiguous `dataset_index` and `scan_id`. No
validation annotation, validation cache, official result, or validation log is
available to the diagnostic or training process.

The existing 562-scene seed-0 split is retained:

- 506 fit scenes / 33,040 expressions;
- 56 calibration scenes / 3,625 expressions.

The 56-scene calibration partition is never used to fit residual parameters,
choose a feature set, or tune a threshold. Within the 506 fit scenes, sort the
scene IDs, shuffle them with Python `random.Random(0)`, and assign them to five
folds by shuffled position modulo five. For each fold, fit the residual model
on four scene groups and produce predictions for the held-out group. Pooling
these out-of-fold predictions is the only source for selecting the model
configuration and single intervention gate. The final residual is then refit
on all 506 fit scenes with those fixed choices and evaluated once on the
untouched 56-scene calibration partition.

All rows remain in their canonical dataset-index order for diagnostics and
digests. Scene-level bootstrap intervals, rather than row-level intervals, are
used when reporting uncertainty so expressions from one scene cannot be treated
as independent evidence.

## Model And Objective

For each row, the frozen geometry runtime supplies a baseline candidate `b`
from the existing flat rank blend and a set of valid alternatives from the
same 16-query by 7-variant axis. The residual model receives only deployable
quantities:

- the normalized 179D alternative-minus-baseline feature difference;
- alternative-minus-baseline frozen parent and geometry ranks;
- alternative-minus-baseline frozen threshold probabilities at `0.25/0.50`;
- alternative-minus-baseline frozen IoU estimate;
- a same-query indicator.

This produces a fixed 185D pair feature. The model is either a linear head or
`Linear(185,64) -> ReLU -> Dropout(0.1) -> Linear(64,6)`. The final six logits
form two three-class heads (`break`, `neutral`, `fix`) for the signed hit change
at `IoU > 0.25` and `IoU > 0.50`. Their final layer is initialized to zero, so
the initial expected change `P(fix) - P(break)` is zero and the selection policy
is exactly the baseline.

The target for each pair is the exact signed tier change computed from detached
train-only IoUs. Cross entropy is averaged over alternatives within a row and
then over rows, preventing rows with many valid variants from dominating. The
`0.25` and `0.50` head losses have fixed weights `2` and `1`. A small
predetermined grid is evaluated only through out-of-fold predictions:

- model: linear or 64D hidden layer;
- weight decay: `1e-4` or `1e-3`;
- break-cost multiplier: `2`, `4`, or `8` (neutral and fix costs are `1`).

Optimization uses AdamW, learning rate `3e-4`, batch size 256, exactly 10
epochs, deterministic seed 0, and gradient clipping at 1.0. There is no
fold-held-out early stopping.

At inference, the residual scores rank alternatives but do not directly
replace the baseline. A switch is permitted only when all of the following
hold:

1. the alternative is valid and differs from the baseline;
2. the predicted weighted gain `2 * delta025 + delta050` is positive;
3. the predicted gain exceeds the fixed minimum margin selected from pooled
   out-of-fold data.

For each grid candidate, intervention margins are the no-switch sentinel plus
the 50th, 60th, 70th, 80th, 90th, 95th, 97.5th, and 99th percentiles of its
positive pooled out-of-fold gain. A candidate/margin pair is eligible only when
every fold has non-negative exact hit change at both thresholds and a
10,000-replicate seed-0 scene bootstrap gives a one-sided 95% lower bound of at
least zero for both pooled hit changes. Eligible pairs are ranked by
`2 * delta_hits025 + delta_hits050`; ties prefer larger margins, fewer switches,
the linear model, larger weight decay, and larger break cost, in that order.

If any condition fails, the baseline candidate is returned. Ties retain the
existing stable flat-index order. No target IoU, GT box, or target mask is
present in the runtime input graph.

## Selection Gates And Best-Weight Preservation

The residual direction is rejected before any backbone/full-model GPU work
unless every fold has non-negative net change at both thresholds, the pooled out-of-fold
`0.25` gain is positive with a scene-clustered lower bound above zero, and the
fixed calibration pass meets all of these requirements:

- `hits025 >= 3,524` (at least the proportional `+63` needed by the official
  target);
- offline cache `hits050 >= 3,315` (no regression from its cache baseline);
- online deterministic `hits050 >= 3,316` (no regression from the authoritative
  step-0 calibration);
- candidate-oracle counts and ordered raw-query IoU digests are unchanged;
- predicted boxes and all frozen artifact digests are unchanged.

The `+63` criterion is intentionally conservative: only 145 calibration rows
are recoverable at `0.25`, so it requires recovering at least 43.4% of that
known opportunity before spending a GPU probe or an official validation run.

Every candidate model, normalization record, cross-fit prediction digest,
calibration record, and checkpoint is written under a new experiment directory
using atomic publication. The existing three artifacts are never overwritten.
The selected candidate is copied to a new read-only path only after all gates
pass. Any failed gate deletes no baseline data and records `selected=baseline`.

## Runtime Integration

The residual scorer is an optional, provenance-checked layer after the existing
geometry scorer. Its artifact binds:

- all three immutable input SHAs;
- the exact feature and candidate schemas;
- the five scene-fold mapping and canonical row digest;
- loss weights, model configuration, and the single gate margin;
- `validation_data_accessed=false`.

The default runtime path remains byte-for-byte identical when the layer is
disabled. When enabled, it receives the frozen geometry output and returns a
single selected candidate on the same axis. It never attaches training-only
IoUs or diagnostics to `end_points`.

## Diagnostics And Failure Handling

Each fit-fold and calibration record reports:

- baseline, proposed, and oracle hits at both thresholds;
- fixes, breaks, switches, abstentions, and switch rate;
- per-scene net changes and clustered confidence intervals;
- candidate rank/margin distributions and gate coverage;
- recoverable-miss coverage and wrong-query versus wrong-variant counts;
- exact ordered selected-IoU and raw-query-IoU SHA-256 digests.

Non-finite features, missing candidates, duplicate query indices, a malformed
artifact binding, any validation path, or any digest mismatch is a hard error.
The runner restores the immutable baseline state before writing a failure
receipt. A partial or failed output is explicitly non-deployable.

## Verification Plan

Before training, CPU tests must cover pair-label construction, row-balanced
losses, zero-initialized baseline parity, stable ties, abstention behavior,
strict threshold semantics, intervention-gate calibration, scene-fold isolation,
GT isolation, artifact tamper rejection, and rollback.

A train-only cross-fit run must reproduce its pooled prediction digest and
metrics on a second invocation. A one-batch GPU smoke then verifies eval-mode
runtime parity, frozen-box/raw-oracle invariance, and checkpoint restore. Only
after the fixed calibration gate passes may a single official 9,508-row run be
launched. The official launcher must report `inference_uses_ground_truth=false`
and preserve the pre-existing best artifact regardless of the result.

## Alternatives Considered

1. **Hierarchical query-then-variant scoring:** directly targets wrong-query
   errors but changes every candidate decision and has a larger regression
   surface.
2. **Constrained geometry fine-tuning with rank distillation:** offers more
   capacity but repeats the high-risk behavior of the rejected joint probe.
3. **Selective residual scoring (chosen):** limits changes to evidence-backed
   switches, has an explicit abstention policy, and can be described as a
   conservative selective-ranking method with scene-level cross-fitting.
