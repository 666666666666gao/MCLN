# CS-MCLN ScanRefer first experiment

Status: three modules implemented; GPU preflights passed at batch 4/8/12. Formal
ScanRefer CS training started around 2026-09-23 08:20 CST. No completed formal
training epoch or new REC result had been recorded at launch.

## Research contract

The protected ScanRefer system remains E71 plus Parent, Geometry and V99 at
5572/4797 REC hits on 9508 validation expressions. The new native model starts
from the E71 MCLN core but does not deploy the three sidecars or any source
selector/MoE. It produces one native Query score, one native box and the
corresponding Query Mask. Mask stays in native training and diagnosis; only
ScanRefer REC Acc@0.25 and Acc@0.50 gate progression.

The three proposed forward modules are M1 observation-aware cross-scale
structure enhancement, M2 context and target-support reading, and M3
Mask-support-driven native box refinement. G-style semantic assignment is a
separate training strategy, not a fourth network module; it is deferred for
this structure-only paired experiment. None of these additions is yet an
accuracy contribution.

## Implementation and comparison

The same E71 model tensors initialize a native control and the CS-MCLN arm.
Core state names, shapes and dtypes must match exactly. Only documented
SourceChoice-only tensors may be omitted, and only the new CS tensors may be
new; do not use an unbounded non-strict load. Initialize the new residual
outputs to zero and verify zero-update native outputs on the same real input.
The control and method use the same fit rows, order, seed 2027, effective batch,
optimizer update count, native losses, inference score and 9508-row evaluator.
No V99 sidecar is added to either arm. Report both arms separately from V99.

Before full training, run full forward/backward/AdamW capacity tests on the
actual A100 40GB at physical batch 4, then 8, 12 and 16 while capacity and
throughput justify progression. Choose one effective batch and parameter-group
learning rates before the run. Proposed B_eff=16 anchors are 1e-4 for M1-M3,
2e-5 for original multimodal/decoder/heads and 2e-6 for PointNet++;
scale each by sqrt(B_eff/16). RoBERTa body stays frozen. Match training by
complete data exposure, not a fixed step count carried across batch sizes.
The first bounded budget is one adapter epoch plus 20 joint epochs (21 total),
fixed before measuring new formal results. The first epoch uses full new-module
LR and 0.1 times the reference core/backbone LR; the remaining 20 epochs use
a fixed cosine schedule that stays positive through the last training epoch.
The first development gate for moving
to other datasets is one model/output at least 58.3%/50.0%; the desired V99
parity is 5572/4797 hits or better. These are proposed thresholds, not results.

The data disk had about 2.50 GB free and the system disk about 2.87 GB free
after old checkpoint cleanup. Before full training, measure actual optimizer
checkpoint size and reserve capacity for one atomic replacement plus logs.
Do not delete the verified V99 four-file chain. The deployed isolated source
snapshot is `/root/autodl-tmp/cs_mcln_source_20260923_v1` (about 40 MiB with
the required existing PointNet++ CUDA extension and original
`data/class_embeddings3d.npy` asset); preflight and current-run
optimizer state go on the system disk. Save one `latest.pth` atomically and
small per-epoch JSON metrics. Formal training also evaluates all 9508 rows
after each complete epoch and keeps one model-only `best.pth` on the data disk;
the predeclared selection key first maximizes the weaker of
`hits025/5572` and `hits050/4797`, then their hit sum. This best weight is
separate from the resumable latest state and from protected V99. There is no
per-epoch checkpoint pile. The first
preflight is CS at physical batch 4. It must prove native/CS zero-update
agreement on one real ScanRefer row, two real AdamW updates, all three module
paths' gradients, and actual checkpoint size before batch selection.

Capacity checks completed on the A100 40GB with the same 48,655-row training
and 9,508-row validation loader: physical batch 4, 8 and 12 all passed exact
zero-update native/CS comparison on `last_center`, `last_pred_size`,
`last_sem_cls_scores` and both predicted-Mask outputs (max absolute difference
zero). Each completed two AdamW steps and the stated gradient checks. Peak
reserved CUDA memory was respectively 9,483,321,344 / 18,635,292,672 /
28,789,702,656 bytes; diagnostic-inclusive two-step times were 3.45 / 6.13 /
8.73 seconds. The full resumable checkpoint measured 813,104,428 bytes and
model-only weight 605,462,146 bytes. Physical/effective batch 12 is fixed for
the first formal run; batch 16 was not attempted because the observed memory
growth predicts too little engineering margin and limited throughput gain.
The batch-12 LR anchors are 8.660254e-5 (new modules), 1.732051e-5 (core),
and 1.732051e-6 (PointNet++), before the epoch schedule.

After a ScanRefer result justifies continuing, freeze the structure and train
Nr3D and Sr3D independently from the same E71 core initialization; do not
inherit the ScanRefer or previous dataset's adapted weights. Fixed single seed;
no seed search or cross-seed ensemble. Main analysis includes both REC
thresholds, repairs/breaks, all-candidate oracle versus selected Query, native
Mask diagnosis, memory, parameters and throughput. Ablate M1/M2/M3 only after
the complete model and its native control are validated.
