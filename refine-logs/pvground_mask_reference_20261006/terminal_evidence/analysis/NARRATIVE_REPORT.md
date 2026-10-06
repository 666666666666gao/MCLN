# Mask-reference terminal evidence

Status: actual fit and collection closed; CPU recount and selected-state reconstruction complete; fresh integrity audit pending.

| Arm | Stage | REC hits@0.25 | REC hits@0.50 | Actual new optimizer updates |
|---|---|---:|---:|---:|
| protected_geometry_parent | protected | 5616 | 4511 | 3723 |
| native_reference | initial_formal | 5615 | 4495 | 0 |
| native_reference | formal | 5617 | 4510 | 3723 |
| fused_mask_reference | initial_formal | 5598 | 4848 | 0 |
| fused_mask_reference | formal | 5593 | 4832 | 3723 |

All rows use complete9508 ScanRefer development expressions and native last/bbs. Zero-update architecture and trained terminals are separate models; historical parent keeps its earlier training budget.

The highest strict result is the zero-update fused-reference model. Its initial final boxes equal its spatial references. It is not an added-training improvement. The trained fused model did not improve on its own zero-update result.

Same input/GT and frozen scoring do not imply bitwise identity between fresh CUDA processes: inspect recorded selected Query changes and repairs/damages. CPU evidence covers all stored256 prior/reference/final boxes; it does not recompute raw Mask membership for every formal row.

The selected checkpoint retains real step0 and empty optimizer state. The separate actual CPU restoration receipt proves construction/state restoration; it is not a new GPU evaluation.

ScanRefer remains one seed and a repeatedly used development split. No new Nr3D/Sr3D performance, zero-shot generalization, multi-seed robustness, learned quality ranking or teacher transfer is established. No nonbest weight has been deleted yet.
