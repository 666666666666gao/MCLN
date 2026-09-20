# G: completed semantic-label replacement and ScanRefer formal evaluation

Published on 2026-09-20; execution completed on 2026-09-18. This replaces the
pending status recorded in handoff section 20.239, without rewriting that history.

G keeps D's full pretrained PV-Ground and observation/task-conditioned reader.
The only training change relative to D replaces the final-layer no-object CE
target for unmatched queries with detached root-box IoU > 0.5. Geometry matching,
other losses, and native inference remain unchanged. No V99 sidecar is attached.

## Fixed training and module comparison

Seed 2027, batch 8, 29,778 fit rows exactly once, 3,723 optimizer steps. Training
took 8,704.871 seconds and ended 2026-09-18 03:07 CST. The terminal 6,887-row
evaluation ended 03:18; independent endpoint audit passed at 03:18:35.

| Native bbs, module holdout | Hits@0.25 | Hits@0.50 |
| --- | ---: | ---: |
| G actual initial | 6147 | 5549 |
| D fixed terminal | 6136 | 5547 |
| G fixed terminal | 6176 | 5602 |
| G minus own initial | +29 | +53 |
| G minus D | +40 | +55 |

D/G initial row, box and score exports are byte identical; all 3,723 fit-step row
orders match. Relative to D, fixes/breaks are 188/148 at .25 and 354/299 at .50.
The holdout scenes were seen during pretraining and are not unseen-scene results.
G passed its fixed module screen and therefore proceeded to formal evaluation.

## Completed formal evaluation

Both arms cover the same 9,508 ScanRefer validation expressions. Evaluation ended
2026-09-18 03:51:17 CST; independent recount ended 03:51:26. All controllers and
audits exited zero. Model states remained unchanged; no optimizer or new weight
was used during evaluation. Main output is fixed bbs.

| Model / output | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| --- | ---: | ---: | ---: | ---: |
| Published PV-Ground parent, native bbs | 5579 | 4381 | 58.6769% | 46.0770% |
| G fixed terminal, native bbs | 5615 | 4495 | 59.0555% | 47.2760% |
| Historical MCLN + V99 protected complete system | 5572 | 4797 | 58.6033% | 50.4523% |

G versus contemporaneous parent: +36/+114 hits (+0.3786/+1.1990 percentage
points); fixes/breaks 346/310 and 494/380. This comparison includes reader
architecture and adaptation, and does not isolate the label strategy alone.

G versus protected V99: +43/-302 hits. Acc@0.25 exceeds 59%, but strict REC
remains below the required floor. **Positive native result, not full promotion.**
No Nr3D/Sr3D training was started for G. Mask did not block promotion and has no
acceptance gate. Diagnostic bbf is 5654/4520, not a replacement primary output.

## Existing-output analysis

Formal raw-256 strict-IoU oracle rises from 7789 to 7884; actual hits rise from
4381 to 4495. The oracle-selection gap falls from 3408 to 3389. The arithmetic
decomposition +114 = +95 + 19 is not causal attribution to separate modules.
Conservative Top-16 good-box presence falls from 5867 to 5562, so the result does
not establish improvement throughout the whole ranking. These are GT diagnostics
over raw candidates, not deployable results or legal-filter recall.

For D versus G module terminals, raw-256 oracle at .50 is 6701 versus 6694;
selected hits improve 55 while the oracle-selection gap narrows by 62. Of 299
strict breaks, only 4 lack a qualifying raw candidate. No inference threshold,
Top-K, checkpoint or seed was selected from these analyses.

The 2026-09-20 CPU overlap analysis reuses verified scene object boxes. In 6,114
D/G rows where both selected boxes have the root as unique maximum-overlap
object, strict fixes/breaks are 274/208 (+66); the remaining rows contribute -11,
giving +55 overall. This is a geometric correspondence proxy, not proof of
semantic identity or a causal separation of box regression and query selection.

## Provenance and retention

- Terminal checkpoint SHA256: `0575dfae333dabdf288a470fc09d8967f9867d1853a5f61d9cfc3cbcf7964522`.
- Published parent SHA256: `6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3`.
- Formal receipt SHA256: `60af173412696ee53ac13e60116460afa6bae5050cd96af84249088dba296374`.
- Formal protocol SHA256: `953f7074fbad0cd5b8fae139c7a4d80db77d5d6f64927f18bd9937bc9a8271de`.
- Endpoint audit SHA256: `0b5e007fb371e35907411b35fc7ef3ee6b4ec4e071ad4f90406eb92a6f52fec6`.

Receipts, per-stage results, training log and CPU comparison are archived under
`refine-logs/pvground_scanrefer_*_20260918_semantic_assignment_v1` and
`refine-logs/pvground_semantic_assignment_comparison_20260918_v1`.
The redundant latest checkpoint (342,296,367 bytes) was deleted after the endpoint
audit; the useful G terminal and protected checkpoints are retained.

On 2026-09-20 the user authorized a complete alternative pretrained baseline.
The next priority is native EG-3DVG acceptance, independently of further G work.
No new G sweep, Ignore training, or cross-dataset claim is implied.
