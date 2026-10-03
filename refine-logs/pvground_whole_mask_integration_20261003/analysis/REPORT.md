# Actual whole-Mask range engineering preflights

Controller completed at 2026-10-03T20:24:49.858877+08:00. Both arms passed, native exit 0.

| Arm | Batch / steps | Head parameters | Allocated GB | Reserved GB | Runner wall s |
|---|---:|---:|---:|---:|---:|
| local_range | 8 / 2 | 400614 | 20.216 | 22.425 | 467.24 |
| whole_range | 8 / 2 | 400614 | 20.174 | 22.421 | 456.56 |

Whole-range step-2 BBox+GIoU evidence gradient: 5.3246665629558265e-05.
Isolated range-path gradients to actual Text / Query Mask / alpha: {"alpha": 4.993092375116248e-07, "query": 1.4795682767854146e-07, "text": 6.637999970873776e-09}.
Control isolated range-path gradients are zero. All eight native Mask losses and final semantic/G loss have no direct gradient to the refiner.
First-step internal head gradients are zero with the zero output layer; all ten head parameter tensors receive nonzero gradients on step 2.
Each process has exact zero-output/cache replay and native call order; strict in-memory model restore and 806 AdamW states / 3 groups verified.

Interpretation limits:

- Two repeated updates on one real augmented batch per arm, not full training or validation.
- Zero-head information on/off replay shares one captured upstream input within each process; cross-process bitwise equality is not proved.
- BBox+GIoU VJP is an isolated diagnostic; actual optimizer updates use the full native weighted loss plus G.
- Allocator peak is for this batch/setup through in-memory restore, not a worst-case-all-scenes capacity or nvidia-smi total.
- All-state and AdamW moment/group/step restoration is in BytesIO; no disk checkpoint or resumed-update/RNG equivalence.
- No localization/segmentation accuracy, teacher knowledge, quality supervision, or Nr3D/Sr3D result.
- The 109-dimensional range summary is not an exact boundary; sizes still use the existing additive output.
