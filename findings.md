# Research Findings

## 2026-09-01 — Nr3D A-V4 Fold-4 scene-disjoint audit

### Intended claim

A-V4 Counterfactual-Parent supervision could improve unseen-scene safe Top-K switching enough to justify formal Nr3D validation or long training while preserving the V99 parent.

### Verdict

`claim_supported = no`，`confidence = high`。

### What the experiment supports

- Counterfactual and actual score axes both executed with finite, nonzero gradients.
- The method changed deployment decisions: `805` held-out samples switched.
- The pre-registered audit completed all `27004` fit rows and `1688` optimizer steps.
- The fail-closed gate correctly rejected the harmful policy and no weight was produced.

### What it does not support

- It does not improve unseen-scene safe Top-K switching.
- It does not justify formal `7899`-row validation, Sr3D transfer, or long training.
- It does not preserve the parent through learned switching; only abstention at the experiment gate preserved the protected parent.

Paired held-out results were negative at both thresholds:

| Metric | Parent | A-V4 | Fix / Break | Net |
|---|---:|---:|---:|---:|
| REC@0.25 | `5661/5915` | `5641/5915` | `38 / 58` | `-20` |
| REC@0.50 | `5011/5915` | `4860/5915` | `108 / 259` | `-151` |

### Postmortem

Counterfactual supervision increased positive-row density and remained differentiable, but the added signal did not separate safe repair from break risk on unseen scenes. The method learned to switch, not to switch reliably.

This is not a convergence or infrastructure failure: sample identity, optimizer-step count, nonfinite checks, receipt/metrics hashes, runtime/data postflight, no-weight output, GPU cleanup, and lock release all passed.

### Constraints for future work

- Seal A-V4. Do not tune threshold, margin, Top-K, loss, LR, or epoch on consumed Fold-4.
- Do not use formal validation or long training to compensate for a failed safety gate.
- Do not revive baseline fair reproduction, rejected Section 7/8, or E0-E7.
- Any next hypothesis must be orthogonal to A-V4 and use a new pre-registered, unconsumed scene-disjoint split.
- A future gate must require paired REC@0.25 gain, REC@0.50 non-degradation, `fix > break`, and exact parent fallback.

### Immutable evidence

```text
receipt  53062ce3110bc5d0f7a2ab9273797a764f06d3234a340b7819d997308abe2605
metrics  97baf04157af257210b8973bdffffc57c1df09331db5fd9505cb005ff07b2781
decision a1c93a71ce62e0c96d02e65241579d520294bf4b868ceccd7178fe9248fd5109
```
