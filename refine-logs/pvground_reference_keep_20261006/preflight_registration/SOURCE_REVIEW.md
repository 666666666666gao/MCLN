# SOURCE_ONLY review — PASS

Reviewed at 2026-10-07T00:02:14.8311785+08:00. No unresolved blocking or nonblocking defect was found in the final source. Review is same-family/provisional; backend identity and current runtime are not attested. This verdict supplies no launch authorization.

One real startup defect was found and resolved before sealing: the actual native Hungarian matcher returns CPU int64 indices, while extra indices come from CUDA Mask eligibility. The first version concatenated them directly in reference_keep.py. Current lines30–31 move both local matched index tensors to boxes.device. The original matcher output and existing losses are unchanged. Current loss SHA: d9c462a7fb66a9e8f20215e5dee8dbee88bc202c5775a34343fb2d28d6c8b244; both specs match it.

The inspected matcher source has the actual prior imported SHA920051cc…; its prefix order proves matches[1] is last_. Matched targets address the same filtered GT list. Extras require own and fused root Mask IoU>.5, exclude every native match, and have no final-Box cutoff. References are detached and use the decoder's1e-6 size clamp. The loss averages squared positive degradation within each expression, then over actual batch size. The original native/G/DFL/extra geometry losses remain unchanged; the specs differ only output root and keep weight0/1.

Both arms reconstruct the retained5598/4848 step0 from official PV, original G and all10 geometry tensors. The selected factory is byte-identical to the prior accepted factory; its actual prior CPU receipt covers1304 complete states and empty Adam. The new execution path does not open retired4511. Exactly456102 geometry parameters are trainable; parent, Mask/text, native heads and zero R remain frozen. SameQuery Box/Mask, all256 candidates and unique last/bbs inference remain.

M0 source runs two real updates per arm, checks direct output gradient scope and exact initial zero keep loss, records the keep output-gradient norm, and checks a CPU-mapped serialization/reload of geometry and Adam. These checks have not been run by this reviewer. Formal runner source preserves fresh optimizer,29778 rows once/3723 updates, B8 and finalB2, and complete9508 plus6887 evaluation outputs.

The exact M0 launcher and timed observer, including embedded remote code, are reviewed. They preserve prior best, use the existing nonblocking GPU lock, launch one sequential preflight controller, wait720s before first observation and then240s or the measured arm estimate, and perform no deletion or package change. The M0 launch does not start formal training. Formal capacity, concrete final selection/cleanup tooling and actual runtime acceptance are later stages.

The current goal remains unmet: the same full ScanRefer checkpoint needs5620/4764 plus three demonstrated effective modules before corresponding-author Sr3D/Nr3D training. No new M0 success, trained gain, three-module result or cross-benchmark result is claimed. Single2027 only.

SOURCE_REVIEW.json records the original/resolved finding, checks, qualifications, and 49 actual paths with byte sizes, SHA256 and review extent. All17 declared helper identities matched locally. No SSH/network/GPU call, model/checkpoint load, experiment-source execution, package change, retired-weight read or deletion occurred in this review. Only SOURCE_REVIEW.json/md were written.

