# Cached reference error review — PASS

Scope: SOURCE_AND_CACHED_EVIDENCE. Reviewed at 2026-10-07T00:46:43.7456670+08:00. No blocking or nonblocking finding. Same-family/provisional; backend identity is unattested.

An independent Python standard-library CPU check actually read all9508 old retained initial_formal rows, recomputed coordinate IoUs and geometric features, and compared all1460 cohort-file rows. All four row lists, every feature dictionary and every summary cohort number match exactly. Recomputed IoUs produce zero.25/.5 threshold flips; max absolute deviations from stored float32 values are2.83959e-6 for prior and6.62009e-6 for reference.

| Threshold | Same-query prior hits | Reference hits | Repairs | Damages | Net |
|---|---:|---:|---:|---:|---:|
|0.25|5615|5598|174|191|−17|
|0.5|4495|4848|724|371|+353|

This is the same selected Query's coarse/native prior versus its neutral Mask reference. It is distinct from the earlier complete protected5616/4511→5598/4848 comparison, net−18/+337.

Among191 loose-threshold damages,128 cover at least95% of GT box volume while the reference volume exceeds4×GT. Separately,52 have stored fused Mask IoU>.5. Median reference/GT volume is4.880391554; median maximum absolute face error is0.503725171m. Damage overlap is118 at both thresholds,73 only at0.25 and253 only at0.5. These are geometric observations, not proof of physical identity, outlier contamination or a causal failure mechanism.

The final diagnostic source SHA db56962e… and SUMMARY SHA93dc5c70… match EXECUTION. Its recorded postexecution simplification remains provenance: the prior source snapshot was not supplied. The reviewer read the final source but never imported or reran it; the separate independent check verifies its numerical definitions against the existing artifacts. Logged Mask IoUs were checked for correct use, not rebuilt from raw Mask members.

No NN evaluation, optimizer update, SSH/GPU/progress query, package installation, deletion or active-fit/observer change occurred. One initial interpreter startup failed before checker execution; the single successful CPU recount used the existing uv archive Python3.13.12. The original SOURCE_REVIEW and FIT_SOURCE_REVIEW files remain preserved. Only this new review pair was written.

The JSON review records9 stable source/input/output identities, the actual independent checker and detailed results. The diagnosis supports examining the loose-threshold errors; it supplies no new performance or module-effectiveness claim.

