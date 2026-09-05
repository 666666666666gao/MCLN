# Protected Nr3D native-path diagnosis: complete, historical reproduction differs slightly

The fixed 7,899-expression / 130-scene native evaluation completed in
648.115 seconds. Independent CSV identity/order, source/checkpoint hashes,
and native metric arithmetic checks passed. All loaded parameters/buffers
equal the protected checkpoint before and after evaluation. No model was
trained or promoted. The controller and workers ended; GPU was idle at
16:46:31 CST on 2026-09-05.

## Protocol closure and its limit

| Metric | Protected historical receipt | Current-source fixed check |
|---|---:|---:|
| REC hits@.25 | 4475 | 4478 |
| REC hits@.50 | 3759 | 3763 |
| Mask hits@.25 | 4192 | 4194 |
| Mask hits@.50 | 3479 | 3482 |
| Mask mIoU | 37.433749% | 37.459280% |

The small differences are **not a method improvement**: weights are identical,
and exact historical core source bytes were not recovered. All five historical
metric-equality checks are false. The current diagnostic is aligned with its
own complete native output; the protected historical best remains 4475/3759.
There was no score-based choice among multiple reproduction runs.

The same historical receipt reports 4271/3705 in its unfiltered source-choice
diagnostic, versus 4475/3759 in the actual filtered REC output. This run similarly
reports 4274/3708 unfiltered versus 4478/3763 filtered. The primary evaluator
uses detector-overlap filtering before ranking; source-choice diagnostics use
unfiltered argmax. Both use the root target. This directly explains why these
two counter families are not interchangeable, but does not isolate all causes
of the separate older cache's 4275 result.

## Full candidate coverage changes the failure diagnosis

All numbers below describe this single current-source native-path check.
IoU comparisons are strictly greater than the threshold.

| Candidate set | Before filter hits@.25 | After filter hits@.25 | After filter hits@.50 |
|---|---:|---:|---:|
| Top16 | 6325 | 6564 | 6017 |
| Top32 | 6882 | 7086 | 6491 |
| Top64 | 7208 | 7349 | 6719 |
| Full256 | 7473 | 7473 | 6823 |

Default and protected score profiles have the same aggregate oracle counts.
Full256 oracle@.25 is 94.6069%; this is GT-assisted availability, not an
achievable deployment score.

The 3421 REC@.25 failures partition as follows:

| Failure category | Top16 boundary | Top32 boundary | Top64 boundary |
|---|---:|---:|---:|
| Reselectable within the legal TopK | 2086 | 2608 | 2871 |
| Qualifying box only beyond legal TopK | 909 | 387 | 124 |
| All qualifying Full256 boxes removed by filtering | 0 | 0 | 0 |
| Full256 contains no qualifying box | 426 | 426 | 426 |

Thus 2995/3421 = 87.5475% of these errors have a qualifying legal Query somewhere
in Full256; only 426/3421 = 12.4525% are Full256 coverage failures. Do not carry
over the old Top16-only proposal-failure fraction as a Full256 failure estimate.
This partition still does not label physical object identity versus box-boundary
error. One expression has no legal Query and is counted as a REC miss.

## Small targets: real coverage failure, concentrated rather than universal

| Input target points | Rows | REC hits@.25 / @.50 | Legal Full256 hits@.25 |
|---|---:|---:|---:|
| <=32 | 131 | 28 / 3 | 42 |
| 33–227 | 1850 | 945 / 564 | 1617 |
| 228–1000 | 3793 | 2198 / 1945 | 3710 |
| >1000 | 2125 | 1307 / 1251 | 2104 |

Of the 426 Full256 misses, 322 have <=227 input target points, including 89
with <=32. Ninety have no target center among fp2 seeds; 149 have no target
center among KPS Queries. The remaining misses cannot all be attributed to
zero seed coverage. Conversely, a target can be covered by a regressed box even
when no selected seed lies inside it: sampled centers are not receptive fields.

Across all rows, the zero-target-center counts at SA1/SA2/SA3/SA4 are
75/156/539/1165; fp2 seeds and KPS Queries have zero counts on 156/221 rows.
These are observations of actual returned FPS indices, not assumptions about
successive indices being prefixes.

| Class | Rows | REC hits@.25 | Legal Top16 oracle | Legal Full256 oracle |
|---|---:|---:|---:|---:|
| mouse | 34 | 5 | 5 | 5 |
| soap dish | 56 | 10 | 21 | 25 |
| bottle | 47 | 15 | 28 | 29 |
| book | 84 | 23 | 49 | 71 |
| cup | 70 | 23 | 53 | 53 |
| bag | 87 | 34 | 73 | 81 |
| door | 428 | 225 | 345 | 416 |

Mouse still primarily requires candidate generation work, whereas much of
book/door failure previously counted as Top16 absence is a ranking/truncation
problem. Do not build scene-specific or validation-class-specific training rules.

## Mask selection, prediction quality, and representation ceiling

Native REC and Mask choose different Query indices on 534 expressions,
including the one without a legal REC Query. Among the 7898 expressions with
both choices, using the REC-selected Query's existing Mask would repair/break
64/17 at .25 and 36/9 at .50: net +47/+27, with +0.288417 Mask mIoU percentage
points. This is a descriptive counterfactual; the deployed Mask path was not
changed. Its small size cannot explain the entire gap to the original paper.

The best existing Query Mask per expression has 7126/5973 hits and 64.6422%
mIoU. A best legal box above .50 exists on 6823 rows; its corresponding Mask
averages 67.1467% IoU. Within those 6823 rows, 1105 (16.1952%) still have no
existing Query Mask above .50. Selection alone cannot fix those strict Mask
misses without changing the predicted masks.

Both native Mask branches produce superpoint logits and map them to points.
A separate CPU-only GT oracle computed the exact best binary union of the
existing superpoints, covering 1213 unique scene/target pairs and all 7899
expressions. It achieves 7843/7476 hits and 83.37498% mIoU. The prefix algorithm
was checked against all unions for 255 nonempty target masks in a small example;
every native Query Mask oracle is at or below this representational bound,
and all target point counts match the native evaluation.

Of the 1105 strict Mask failures with a good box, only 154 also have a
superpoint bound <=.50; the other 951 could be represented by the current
superpoints. This separates a real geometric partition limit from substantial
remaining mask-prediction error. It does not prove a unique faulty network layer.

## Language groups and reference-memory availability

For comparison with the older cache table, lengths below use the **raw Nr3D
CSV `tokens` list**, not whitespace tokens after scene-graph normalization.

| CSV token count | Rows | REC hits@.25 | Legal Top16 oracle | Legal Full256 oracle |
|---|---:|---:|---:|---:|
| 2–6 | 1639 | 1001 | 1365 | 1556 |
| 7–8 | 1576 | 949 | 1343 | 1501 |
| 9–12 | 2389 | 1341 | 1998 | 2267 |
| 13+ | 2295 | 1187 | 1858 | 2149 |

**Schema correction:** the sealed observer named its normalized annotation
whitespace count `raw_token_count`. The original `analysis.json` length groups
use that field. `enrichment.json` explicitly corrects this label and supplies
CSV token counts and the groups above without changing any predictions, row
identities or original evidence files. Use the enriched groups for comparison.

Full legal Query memory covers 102067/311948 = 32.7192% of detector-object slots;
target Default Top32 covers 50520/311948 = 16.1950%. These slots are repeated
per expression, and this is not ground-truth text-anchor recall. It does show
why full Query memory cannot be assumed to contain every potential reference
object. The existing `butd_cls` object-input branch is a concrete interface to
inspect before another relation experiment, rather than assuming a larger
target-ranked Query list supplies complete reference evidence.

## Next decision and artifacts

P2 v1 remains failed and does not advance to P3. The new P1 evidence supports
keeping candidate disambiguation first, but it does not retroactively validate
the failed pair head. Before another structural trial, inspect reference-memory
availability and the actual object-feature interface on training inputs. Any
new experiment needs one explicit change and a matched old-readout control.
Local geometry work should target the independently identified Full256 misses;
Mask work should distinguish prediction error from the superpoint ceiling.
No old gate, replay, LR, or canceled baseline sweep was resumed.

Evidence directory: `refine-logs/official_candidate_audit_20260905_v1/`.
The 23,614,096-byte raw JSONL is archived losslessly as `rows.jsonl.gz`;
decompressed SHA-256 is
`d99a504828876eb18d8fe77de3b901dc3ce374e58363ba662e317871fe6d5767`.
The receipt, independent verification, complete class/scene tables, superpoint
oracle, CSV enrichment, raw native log, input manifest, recovered historical
receipts, and 627-file data inventory accompany it. The protected checkpoint
and ScanRefer/Sr3D models were not changed. The overall three-benchmark goal
remains unmet.
