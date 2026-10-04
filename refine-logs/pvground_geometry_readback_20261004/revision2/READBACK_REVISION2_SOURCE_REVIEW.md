# Readback V2 source review

**Verdict: PASS — SOURCE_ONLY.** No blocking or nonblocking correctness defects were found in the reviewed V2 implementation. This satisfies the fresh source gate for CPU construction and a two-update real-batch preflight per arm only. It does not approve runtime success, accuracy claims, or formal fitting.

This remains a fresh same-family review with provisional acceptance. The requested Astra/max backend and effort are not independently attested. No remote calls, credentials, Torch import, model forward, launch, or primary-source modification occurred in this review.

## What was checked

The accompanying JSON lists 34 current V2 files with exact paths, byte lengths, and SHA256 values for the launcher's existing guard. It covers all 16 runtime modules, both model overrides, launcher/controller/source creation/observer/collector, both templates, plan, source-check manifest, local sealing script and six root helper copies. The reviewed files exclude these review outputs and private traces.

The model previews are byte-identical to the complete sources read during the failure review. Six root/helper pairs are byte-identical to the deployed bundle. Local stdlib verification parsed all 30 Python files using Python 3.7 grammar, confirmed the 16 runner and two override manifest entries, and confirmed that the two arm templates differ only in `root` and `use_geometry_evidence`. These are actual local source checks, not a remote package/API test.

The corrected face-artifact path resolves to the intended sibling project. Its existing local intake reports complete/exit0/controller closed, and its audit has no blocking issues. Protected remote weights and current GPU availability were not queried by this reviewer.

## Why the correction is valid

V2 records the two disabled full-forward outputs and disabled-versus-zero-R outputs as endpoint difference diagnostics. It does not turn an `allclose` tolerance into acceptance. The R-isolation witness now clones every protected geometry/evidence/contrast tensor and every Text/Query Mask/alpha tensor immediately before R, then compares each to the final output of that same forward with exact equality. The full 256 candidate ordering and values remain covered. The floor flags are also protected.

The actual native final semantic head has pre/post input/output witnesses and exactly one call in a real full forward, after refiner and R. Zero initialization is checked on the same cached semantic Query; the existing frozen head is replayed only in a separate labeled diagnostic outside the model forward and counting hooks. R's reviewed implementation returns an out-of-place residual and does not modify its input Query or geometry. The source therefore checks zero-start scores without rerunning the stochastic/inherited geometric pipeline.

Fresh dictionaries are **shallow** copies. Original tensor leaves have independent cloned snapshots, and after each forward the original tensor values, text and dictionary keys must remain exact. This catches raw-value mutation and prevents normal backbone key additions from accumulating in the original dictionary. It is not a claim that every tensor is independently cloned for every full forward.

Strict official/G/protected-distribution loading, upstream freezing/eval mode, the 23 R tensors / 96672 parameters, and final unchanged frozen-state checks remain intact. The sameframe witnesses are used for zero init and both update forwards. Two fresh AdamW steps, finite/present R gradients, absent upstream parameter gradients, isolated native final CE+G gradient witnesses, all256 native bbs score/rank/logit-gradient checks, and strict in-memory model/optimizer restoration are still required.

I also checked the relevant inherited criterion/evaluator source excerpts. The criterion's prefix order places the final layer at matcher result1 and weights ScanRefer CE by0.5/7. The bbs helper follows the actual root token reductions, including binarizing only the main positive map. Training losses use actual dataset labels, with detached final-box IoU used only by the existing G correction; no new deployed scorer, quality loss, or inference GT gate is introduced.

## Execution source scope

The launcher validates this report's existing byte guard and targets only the new `pvground_readback_preflight_20261004_v2` directory. Its preconditions check the old failed/exit1 artifact, protected geometry identity and absence of compute processes. Source creation copies the sealed parent then applies only the two reviewed model overrides in V2. It does not edit the failed attempt or the parent source.

The controller performs hidden CPU/preflight then visible CPU/preflight sequentially. It stops on a nonzero child exit and checks receipts before counting successful phases. The observer is read-only, starts at the declared milestone, polls240s and never restarts work. The collector requires a closed observation/controller and collects JSON/log/exit files only. Formal training and disk checkpoint creation are outside this entry point.

## What remains unproven

The original failed attempt remains a genuine CPU PASS followed by GPU failure at `last_center`, before any optimizer update. V2 has not been executed by this reviewer, and the original discrepancy's cause and scale are not declared resolved.

Real V2 receipts must establish both arms' CPU/GPU outcomes, same-forward equality, cached semantic equality, gradients, updates, restoration, memory and timing. Its difference JSON will quantify endpoint repeatability; it is not a full first-divergent-kernel trace. Even an eventual V2 preflight PASS must not be described as proof that every repeated full-model forward is bitwise identical.

There is no formal-fit approval, native accuracy result, accuracy improvement, target-attainment or cross-dataset claim in this review.

