# Density-Aware Target Box Auxiliary — Frozen Experiment Contract

## Objective

Improve sparse and small referred-object proposals without changing the V99
inference architecture, score path, masks, selector, or candidate count.

The motivating Nr3D diagnosis is independent of ranking: the sparsest target
quartile has much lower Top-1 accuracy and materially lower Top-16 oracle
coverage than dense targets. A ranking-only verifier cannot repair rows where
no sufficiently accurate proposal exists.

## Exact intervention

The intervention is a training-only auxiliary box-regression loss. It has no
parameters and creates no inference branch.

For each referring sample, the dataset contract assigns the referred target to
GT row 0 and labels its sampled points with `point_instance_label == 0`.
ScanNet detection-only rows are excluded.

Let `n` be the number of sampled target points and fix `T = 256`. The row is
active exactly when `0 < n < T`, with detached density weight

```text
w = 1 - n / T.
```

At the final decoder only, use the existing Hungarian assignment to find the
query matched to GT target 0. Its auxiliary error is

```text
sum(abs(pred_center - gt_center))
  + 0.2 * sum(abs(pred_size - gt_size)).
```

The batch auxiliary is the density-weighted mean over active rows. A referring
row without exactly one Hungarian match for target 0 is a contract error.

## Frozen boundaries

- `T = 256` and size coefficient `0.2` are fixed, not validation-tuned.
- Only the scalar auxiliary loss weight is exposed; its default is exactly 0.
- Weight 0 must preserve the old loss/output path exactly.
- No dataset ID, Unique/Multiple label, validation margin, GT anchor sidecar,
  new proposal, score residual, gate, or inference-time box refinement.
- The same code path must support ScanRefer, Nr3D, and Sr3D.
- Gradients may flow only through the final matched predicted box; density,
  matching, and GT tensors are supervision only.

## Required audit before training

The first deployment is audit-only from a protected V99 checkpoint:

- exactly 100 training micro-batches;
- no validation, no checkpoint save, no official-best promotion;
- finite total loss and gradients;
- nonzero active-row ratio and auxiliary loss;
- all reported density statistics finite;
- protected source checkpoint unchanged;
- default-off regression tests pass.

Passing the 100-batch audit does not authorize long training. The next gate is
a scene-disjoint short experiment with the same fixed intervention. Formal
7,899/17,726-row evaluation is allowed only after held-out evidence is positive.

The cancelled baseline reproduction, the prior Section 7, and the prior E0–E7
matrix are outside this contract.
