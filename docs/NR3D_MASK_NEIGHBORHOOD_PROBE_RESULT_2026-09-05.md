# M4: real empty-radius Mask neighborhoods and nearest-two intervention

Completed2026-09-05 at19:46 CST, controller exit0. Eight forwards use the same16
fit expressions, B4 batches, seed0 and point-cloud hashes as M3. No training,
validation or checkpoint writes. Native raw Query Mask IoUs exactly reproduce
M3. Original and nearest-two arms have identical Box tensors, selected-source
scores, final Decoder Queries, projected seed features and Query selection.
All16 seed_xyz tensors equal input points indexed by seed_inds exactly; the
suspected seed-index misalignment is not observed.

The installed CUDA extension reproduces the synthetic case: it takes the
first two in-radius indices rather than the nearest, and returns [0,0] when
the radius contains no seed. Binary SHA256:
19bf15db9a58b605d49e9241906038a459c737a48065fc6c17e00a38bae5f839.

| Present superpoint cohort | Count | Empty radius | Native without target seed center | Nearest-two without target seed center |
|---|---:|---:|---:|---:|
| All occupied superpoints | 32370 | 1641 | 32009 | 31912 |
| Contains at least one target point | 459 | 43 | 156 | 96 |
| Majority of points belong to target | 302 | 31 | 66 | 19 |

Most all-scene superpoints are background; absence of target seed centers there
is expected. Seed-center membership is not receptive-field coverage.
The31 empty majority-positive neighborhoods read seed0 at distances3.092–6.160m
(median4.209m); actual nearest seeds are.200–.285m away (median.222m).
Nearest-two restores a target seed center in48 majority-positive neighborhoods:
25 empty-radius and23 nonempty-radius cases. It loses one such neighborhood,
giving the net47 reduction from66 to19. Neither selection uses GT.

The empty foreground cases are concentrated:22/31 belong to fit row15. Do not
present their superpoint-level frequency as an independent scene-level rate.
Upstream source9744a4ed219062d448ed0dba587eeb864491f158 also uses radius.2,
nsample2 and the same relative-encoding/max-pooling path. This is an inherited
architectural limitation, not a proven regression from the added modules.

| Fixed16 fit expressions | Original | Nearest-two | Difference |
|---|---:|---:|---:|
| Native selected fused Mask mIoU | 60.38935749% | 62.67710096% | +2.28774347pp |
| Native selected fused Mask@.25 | 16/16 | 16/16 | 0 fixes,0 breaks |
| Native selected fused Mask@.50 | 10/16 | 11/16 | 1 fix,0 breaks |
| Native selected raw Query Mask mIoU | 60.49106961% | 62.79548704% | +2.30441743pp |
| Original matched Query raw Mask mIoU | 60.11613701% | 62.67462739% | +2.55849038pp |
| Full256 raw Mask oracle mIoU (uses GT) | 62.70907992% | 66.44301835% | +3.73393843pp |

Native fused Mask IoU improves on6 rows, worsens on6 and is unchanged on4.
Fit row19 improves.63810→.94855 and row15 .42199→.55971; these two rows account
for most positive change. Row21 worsens.83436→.76529, for example. The text
branch's mean increases but it loses one hit at each threshold. All16 Hungarian
matches stay unchanged. This is a small, backbone-seen training slice, not
evidence of formal or cross-dataset gains.

Runtime28.661461s excludes dataset initialization and includes final integrity
checks. Maximum allocated GPU memory1,676,728,832 bytes. The612 source files,
36 data files, protected checkpoint and all parameters/buffers remain equal;
the original grouper is restored. At19:46:21 GPU is1MiB/0%, controller complete.

Full actual centroids, seed coordinates, neighbor IDs/distances, target seed
membership, per-Query Mask IoUs, synthetic evidence and paired native outputs
are in refine-logs/mask_neighborhood_probe_20260905_v1/receipt.json.
Its7,515,376 bytes have SHA256
666873594228925127da50e96180f9a5292266c757c0f66237192c4957d855b9.
The CPU summary is derived only from that receipt; original run.log and exit
are retained. CPU tests2 PASS and remote Python3.7 compilation PASS.

Decision: the locality hypothesis merits the separately preregistered M5
matched short training of existing Mask projections. Do not adopt the inference
substitution directly. M5 cannot improve REC because that path stays frozen;
the wider three-benchmark objective remains unmet. Protected Nr3D is still
4475/3759 on7899 expressions; no ScanRefer/Sr3D production artifact changed.
