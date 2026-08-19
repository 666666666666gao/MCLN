# V99 ScanRefer REC post-processing architecture

V99 is the frozen two-stage ScanRefer post-processing system that produced the
best single-system `REC@0.25` result in this repository. It does **not** retrain
the MCLN backbone and it does **not** include the later Nr3D/Sr3D migration
pipeline as part of the V99 innovation.

## Result contract

The official ScanRefer validation receipt contains 9,508 expressions.

| Metric | Overall | Unique | Multiple |
|---|---:|---:|---:|
| REC@0.25 | **58.6033% (5572/9508)** | 88.8654% | 53.2946% |
| REC@0.50 | 50.4523% (4797/9508) | 80.5497% | 45.1725% |
| Mask@0.25 | **59.8443%** | 90.2044% | 54.5185% |
| Mask@0.50 | **52.3349%** | 80.1268% | 47.4595% |
| Mask mIoU | **45.9303%** | - | - |

The result is one complete V99 system. It does not combine V99's best
`REC@0.25` with V113's best `REC@0.50`.

## Frozen post-processing pipeline

```text
frozen MCLN checkpoint
        |
        v
Top-16 REC queries + frozen parent/geometry features
        |
        v
16 queries x 7 valid mask/geometry variants = at most 112 candidates
        |
        v
query/variant encoders (hidden dimension 128, dropout 0.1)
        |
        v
one 4-head Transformer encoder layer over the 16-query set
(FFN 256, GELU, padding mask, permutation equivariant)
        |
        +--> query head: two threshold-aligned logits per query
        |
        +--> variant head: two threshold-aligned logits per variant
        |
        v
hierarchical query-then-variant proposal
        |
        v
fixed Pareto safety gate against the frozen parent
        |
        v
selected REC query also selects the deployed Mask
```

### Inputs

- Query tensor: 16 candidates, 152 frozen features per query.
- Variant tensor: 7 variants per query, 25 frozen features per variant.
- Query auxiliary values: default/parent scores and ranks plus Top-1 flags.
- Variant auxiliary values: geometry score/rank plus geometry/baseline Top-1
  flags.
- Invalid queries and variants are masked in aggregation, attention, scoring,
  and final selection.

### Contextual hierarchy

`ParetoContextualHierarchicalReranker` first embeds every variant, aggregates
valid variants with masked mean and max pooling, and concatenates those values
with the frozen query features. A single Transformer layer then lets every
valid query compare itself with the other queries from the same expression.
The contextual query embedding conditions the per-variant head, so selection
remains explicitly hierarchical: query first, variant second.

### Threshold-aligned objective

The artifact contract fixes a soft-listwise bounded target derived from:

```text
IoU + 2 * hit@0.25 + hit@0.50
```

The model emits separate `@0.25` and `@0.50` logits. Monotone hit
probabilities are used to compare the proposed candidate with the frozen
parent. Training is fixed to seed 0, 12 epochs, batch size 256, learning rate
`3e-4`, weight decay `1e-3`, gradient clipping at 1.0, and target temperature
0.25.

### Fixed Pareto deployment gate

A proposal replaces the parent only when both predicted threshold gains are
strictly positive and

```text
2 * delta@0.25 + delta@0.50 >= 0.13312220573425293
```

Otherwise V99 returns the frozen parent selection. The margin is part of the
sealed V99 artifact; it is not searched again on ScanRefer validation.

### Leakage and provenance controls

- Five scene-disjoint folds create out-of-fold decisions.
- All five folds have positive gains.
- OOF net gains are `+175/+474` hits at `@0.25/@0.50`.
- Scene-bootstrap 95% lower bounds are `+132/+385` hits.
- The final artifact binds the backbone, parent, geometry model,
  normalization, candidate materialization and OOF receipt by SHA-256.
- The backbone and upstream rerankers remain frozen; only the compact V99
  post-processing head is fitted.

## Source map

| Responsibility | Primary source |
|---|---|
| V99 contextual network and Pareto policy | `models/rec_pareto_contextual_hierarchy.py` |
| Base 16x7 hierarchy and selection helpers | `models/rec_hierarchical_reranker.py` |
| V99 scene-disjoint OOF runner | `scripts/run_v99_pareto_contextual_hierarchical.py` |
| Full-fit artifact builder and validator | `scripts/build_v99_pareto_contextual_artifact.py` |
| Frozen official evaluation runner | `scripts/run_frozen_v99_pareto_contextual_official.py` |
| Full-fit replay audit | `scripts/audit_v99_fullfit_replay.py` |
| Runtime parity audit | `scripts/audit_v99_runtime_parity_train.py` |
| Immutable best-result manifest generator | `scripts/archive_v99_rec025_best.py` |
| Core architecture tests | `tests/test_rec_pareto_contextual_hierarchy.py` |
| OOF policy tests | `tests/test_v99_pareto_contextual_hierarchical.py` |
| Official runner tests | `tests/test_run_frozen_v99_pareto_contextual_official.py` |

The detailed result and artifact hashes are preserved in
`docs/archive/V99_REC025_BEST_ARCHIVE.md`. Checkpoints, model artifacts,
datasets, caches and experiment outputs are intentionally excluded from Git.

## Scope boundary

The later `run_nr3d_v99.sh`, `run_sr3d_v99.sh`, source-choice selector and
dataset pipeline are portability experiments built around the V99 idea. They
are not required to define the original V99 ScanRefer innovation and should
not be cited as part of the model that produced the result above.
