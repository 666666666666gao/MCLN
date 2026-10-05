# Query-supported geometry responsibility: closed result

Finished CST: 2026-10-05T13:53:50.086084+08:00. Primary native last/bbs Acc@0.50.

| System | Hits@.25 | Hits@.50 | Acc@.25% | Acc@.50% |
|---|---:|---:|---:|---:|
| protected_geometry_parent | 5616 | 4506 | 59.0660 | 47.3917 |
| control | 5616 | 4499 | 59.0660 | 47.3180 |
| query_supported | 5614 | 4509 | 59.0450 | 47.4232 |

Strategy vs same-budget control: repairs 22, damages 12, net +10.

Strategy vs protected parent: repairs 16, damages 13, net +3.

Strategy same-Query coarse/final refinement: repairs 33, damages 19, net +14.

Same29778 rows/order once,each3723 updates/B8/tailB2. Only geometry head updated.
Fresh terminal integrity review and weight retention pending; this analyzer never loads/deletes/archives weights.
Single seed2027; two actual same-start full fits, parents and zeroR frozen; existing boundary head only. Native9508 is development validation,6887 is pretrained-seen module holdout. CPU independently checks selected Box/GT thresholds; Masks and full candidate coverage are scalar recounts, not raw-array replay. GT qualification and volume grouping are offline only. Candidate support/overlap is not physical identity proof. Cached-upstream preflight exactness does not prove cross-CUDA full-forward bitwise equality. No Nr3D/Sr3D new result or complete three-module novelty proof.
