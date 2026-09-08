# P2 v1 completed: pair readout fails the fixed screen

This is the completed single-seed, new-head holdout experiment on **6,172
expressions / 98 train scenes already seen by the frozen backbone**. It is not
the official 7,899-row Nr3D validation and does not establish unseen-system
generalization. Both the mechanism and practical screens failed; this variant
does not advance to Decoder joint training or formal promotion.

## Matched execution

Both heads completed 6,687 AdamW updates on the same 26,747 fit expressions and
the same single frozen backbone forward per batch. 26,483 rows had qualifying
legal Top-32 boxes; no batch was entirely uncovered. Training took 2,402.461 s;
training plus evaluation took 2,995.545 s, excluding initialization. The final
global/pair weights are separately archived on the server. No protected
checkpoint, production model or ScanRefer/Sr3D result was changed.

An independent terminal check verified source and checkpoint hashes, final
head hashes, row identities, equal update counts, metric and fix/break
arithmetic, and the original decision. The trainer also checked protected REC
row hits against the existing evaluator. GPU is idle after the completed run.

## Complete holdout result

All denominators below are 6,172. Protected Mask uses its actual Mask choice;
other rows use the single Query selected by their corresponding score.

| 方法 | REC hits@.25 | REC hits@.50 | Mask hits@.25 | Mask hits@.50 | Mask mIoU |
|---|---:|---:|---:|---:|---:|
| protected | 6005 | 5312 | 5767 | 5056 | 68.8766% |
| default | 6005 | 5312 | 5754 | 5046 | 68.7141% |
| global | 6003 | 5514 | 5752 | 5044 | 68.6682% |
| pair | 6002 | 5430 | 5751 | 5038 | 68.7242% |

Pair minus global: **-1 / -84 REC hits**. At .25 there were 9 fixes and 10 breaks;
at .50 there were 79 fixes and 163 breaks. Long expressions (1,844 rows) and
2+ distractors (2,891 rows) both had zero net .25 improvement over global.
Their .50 changes were -13 and -46, respectively.

Pair minus protected: **-3 / +118 REC hits**, with .25 fixes/breaks 18/21 and
.50 fixes/breaks 250/132. Mask mIoU was 0.1524 percentage points lower.
Global's +202 strict REC hits versus protected do not establish the benefit
of candidate-pair text reading: the simpler global control achieved more of
that gain. Neither head improved the registered primary .25 criterion.

The paired whole-scene 95% percentile interval for pair-minus-global REC@.50
is **[-2.1283, -0.6404] percentage points**, estimate -1.3610. The .25 interval
is [-0.1688, 0.1312]. Pair-minus-protected .50 is +1.9119 points, interval
[0.8706, 3.0027], while its .25 interval includes zero. These are descriptive
single-seed intervals and did not change the screening gates.

This fixed, one-epoch variant failed. It does not prove that every possible
pair-conditioned relation model or training schedule is ineffective. No
LR/epoch/margin sweep or failed-head production deployment follows this result.

## What the candidate and Mask diagnostics show

The protected score already obtains 6,005/6,172 = 97.2942% REC@.25 on these
seen train scenes. Full-256 oracle is 6,147/6,172; legal Top-32 oracle is 6,146.
Its 167 failures partition into 141 correctable within legal Top-32, one with
a qualifying box only beyond Top-32, zero removed by the filter, and 25 with
no qualifying Full-256 box. There are zero rows with no legal Query; 1,261
rows have fewer than 32 legal Queries. These frequencies do not describe
the much harder official validation set.

Across expression-repeated detector-object slots, full legal Query memory
covers 64,134/214,964 = 29.83%; target Top-32 covers 34,995/214,964 = 16.28%.
This confirms incomplete object availability, not annotated text-anchor recall
and not a causal explanation of the failed pair head.

Full Query Mask oracle mean is 73.0316%. Among 5,933 rows with a legal box above
.50, 712 (12.00%) still have no Query Mask above .50. Those rows cannot be
fully repaired by selecting another existing Mask. Again, this is a train-scene
audit frequency, not an official validation prevalence estimate.

## Next work

Keep this variant archived. Before another architecture change, finish the
user-requested P1 diagnostic on the protected official output path: verify the
checkpoint, source and evaluation protocol, then record pre/post-filter
Full-256 coverage, actual selection and conditional Mask quality in a fixed
read-only run. This is not a promotion of G0 or of the failed P2 head, and not
a restart of the canceled baseline retraining or gate/reranker sweeps.

All raw evidence is in `refine-logs/p2_readout_v1_20260905/`, including the
7.75 MB row JSONL, receipt, original mechanical decision, independent checker
and completed diagnostic tables/scene intervals. The protected formal Nr3D
best remains 4,475 / 3,759 REC hits; the three-benchmark objective is unfinished.
