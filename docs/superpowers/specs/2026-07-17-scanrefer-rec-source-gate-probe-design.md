# ScanRefer REC Source-Gate Probe Design

## Status And Objective

This design follows the failed 306-step joint REC fine-tune and the subsequent
train-only cache diagnostics. The project objective remains a new, frozen,
9,508-sample ScanRefer validation result with:

- `last_ position alignment Acc0.25 Top-1 >= 0.60000`;
- `last_ position alignment Acc0.50 Top-1 >= 0.47000`;
- no inference-time ground truth.

The current frozen result is `0.58288/0.48601`. The new probe is not a
completion claim and must not access ScanRefer validation.

## Evidence

The authoritative train calibration partition contains 3,625 expressions.
Its frozen step-0 diagnostics are:

| Stage | Hits@0.25 | Hits@0.50 |
| --- | ---: | ---: |
| Default Top-1 | 3,406 | 3,175 |
| Parent Top-1 | 3,421 | 3,202 |
| Geometry Top-1 | 3,461 | 3,316 |
| Parent-candidate oracle | 3,599 | 3,521 |
| Geometry oracle | 3,606 | 3,588 |
| Raw 256-query oracle | 3,615 | 3,559 |

After 306 joint updates, raw-query oracle improved by `+4/+5`, but the parent
candidate oracle fell by `-28/-30` and final geometry Top-1 fell by `-30/-29`.
The updated source scores therefore removed useful queries from the fixed
Top-16 pool even though the boxes themselves did not degrade.

Additional frozen-cache experiments rejected the following as primary fixes:

- auxiliary threshold/IoU-head score fusion;
- dual-threshold positive-mass fine-tuning;
- listwise/pairwise alpha endpoints;
- zero-margin Top-1 structured hinge;
- five-model rank ensembles;
- query-level mean/logsumexp aggregation;
- MLP and LightGBM query-level second-stage scorers.

At `IoU>0.25`, 145 of the 164 frozen geometry misses are recoverable; 133 are
wrong-query errors and only 12 are wrong-variant errors. The next probe must
therefore improve source-gate membership while leaving boxes and the two
rerankers frozen.

## Immutable Inputs And Data Boundary

The probe uses the same immutable epoch-71 backbone, parent artifact, geometry
artifact, seed-0 train scene split, and deterministic calibration contract as
the one-epoch fine-tune design. It may construct only
`Joint3DDataset(split="train", dataset_dict={"scanrefer": 1})`.

The runner must fail closed if it opens a ScanRefer validation annotation,
`/val` cache, `geometry_val`, official result, official log, claim, or receipt.
It writes only to a new disposable probe directory. It never overwrites or
relabels the frozen three-artifact baseline.

## Audited Loader Execution

The original `num_workers=2, pin_memory=True` loader stalled at the first
calibration batch only when the process tree was traced. Both workers had
already filled their two prefetched batches and created shared-memory objects,
placing the stall after worker-side dataset loading. A `num_workers=0` control
completed the same first batch in 2.129 seconds, but would change worker RNG
and therefore the augmented fit-batch contract.

A single-variable `num_workers=2, pin_memory=False` run under the same strace
completed the first batch in 4.221 seconds, retained indices `1275..1292`, and
exited zero. The source-gate probe therefore keeps two workers and the existing
worker seeding, but disables pin memory for both fit and calibration loaders.
The legacy fine-tune loader remains unchanged. The source-gate train-data
contract and receipt bind the live `num_workers` and `pin_memory` values for
both loaders and reject drift.

## Full-Query State

Factor the existing REC score calculation in `models/rec_candidate_adapter.py`
into a pure full-query builder. For every one of the 256 queries it returns:

- deployable default semantic score;
- deployable contrastive score;
- predicted box `[cx,cy,cz,w,h,d]`;
- the existing 152D per-query deployable feature vector.

`build_rec_candidate_batch` must call this builder and then apply the unchanged
stable `default Top-8 union contrastive Top-8`, deduplication, default-fill,
and Top-16 truncation contract. Existing runtime outputs must remain exactly
equal. Ground-truth fields are attached only after the deployable state exists.

## Trainability

Only the final decoder prediction head's semantic classifier is trainable:

```text
prediction_heads.5.sem_cls_scores_head.*
```

Every decoder, box/size head, mask head, projection, source selector, parent
reranker, and geometry reranker parameter remains frozen and in eval mode.
Consequently raw predicted boxes and raw-query oracle IoUs must be bitwise
stable across an optimizer step. The probe uses a fresh AdamW, constant
learning rate, no scheduler, and gradient clipping only on this exact group.

## Top-K Membership Objective

For each strict threshold `t in {0.25, 0.50}`, let `P_t` be full queries whose
detached root-only IoU is greater than `t`. Let `s+` be the maximum default
source score in `P_t`, and let `s_k-` be the eighth-largest default score among
queries outside `P_t`.

When `P_t` is empty, or fewer than eight negative queries exist, the row is not
informative for that threshold. Otherwise use:

```text
softplus((s_k- + margin - s+) / temperature)
```

The probe fixes `margin=0`, `temperature=1`, and threshold weights
`w0.25=2`, `w0.50=1`. Losses are averaged within each threshold over
informative rows, then combined by the fixed weights. The contrastive source is
frozen and acts as the second, unchanged half of the union gate.

The loss publishes per-threshold informative-row counts, active membership
violations, positive counts, and the mean score gap `s+ - s_k-`. A zero-active
batch is legal and returns a differentiable zero. A complete probe with zero
informative rows is invalid.

## Probe And Selection Gate

Run the ordered, unaugmented train calibration at step 0 and after 306 fit
updates. Training uses the existing augmented fit view, batch size 18, natural
remainder semantics, and the same deterministic seed-0 scene split.

The step-306 state is eligible only if all conditions hold:

1. geometry Top-1 hits at both thresholds are not below step 0;
2. parent-candidate and geometry-candidate oracle hits do not fall below step 0;
3. raw-query oracle hits and ordered raw-query IoU digest equal step 0 exactly;
4. at least one of default Top-8 positive membership, parent-candidate oracle,
   or final geometry Top-1 improves at `0.25`;
5. all frozen parameter and selected-output reproduction digests match their
   declared contracts.

Failure restores step 0 and publishes only a nondeployable failure receipt.
Success still publishes only a nondeployable probe receipt. A separate design
revision is required before producing deployable artifacts or launching the
one-shot official validation.

## Testing And Audit

Focused tests must prove strict thresholds, eighth-negative cutoff behavior,
invalid-row handling, differentiable zero, exact trainability, frozen box
outputs, full-query/candidate-builder parity, fixed Top-16 tie policy, target
isolation, ordered calibration digests, regression rollback, and rejection of
all validation paths.

Before the 306-step probe, run the complete CPU suite and an audited one-step
GPU smoke. The file-access audit must report zero violations and the smoke must
exactly reproduce step-0 calibration before and after staged reload.
