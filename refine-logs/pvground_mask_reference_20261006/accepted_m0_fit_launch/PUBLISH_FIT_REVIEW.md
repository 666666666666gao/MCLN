# Doc83 fit-launch publication source review

PASS — zero unresolved blocking findings. Reviewed 2026-10-06T11:27:03.823545+08:00.

Scope: SOURCE_ONLY; same-context followup, same-family, provisional; backend identity not_attested. No network, GPU, model forward or publisher execution occurred. Main sources and SOURCE_REVIEW were unchanged.

- All 24 M0 intake files match their recorded sizes/SHA (154131 B). Controller status, raw log and wait receipt agree; controller/child exit files are 0. Both arm receipts match their raw logs and current runner/spec hashes.
- Each arm records 2 real updates on the same 8 inputs, 456102 parameters/10 states, frozen parent/R, all256 raw bounds error0, actual39 empty-support prior retention, nonzero step2 gradients for all10 parameters, and strict CPU model/Adam restore. The serialized5481547 B stayed in memory; no saved weights or accuracy. Upstream exactness is limited to cached replay.
- The source accurately retains the small-batch range limitations:

| Observed count | Native step1 / step2 | Fused Mask step1 / step2 |
|---|---:|---:|
| Matched DFL outside, of48 faces | 0 / 0 | 2 / 2 |
| Extra DFL outside | 120/858 / 126/852 | 196/768 / 200/768 |
| Extra candidates | 143 / 142 | 128 / 128 |
| Invalid reference candidates, of2048 | 1033 / 1045 | 1033 / 1045 |

M1 acceptance embeds the exact launch record: controller711378, sole observer27853, launched11:14:14, first observation17:14:14, estimated completion17:20:54, then240 s polling. Each arm retains3723 updates/29778 rows/B8/freshAdam and the declared initial/terminal evaluations. The document reports a launch, no completed M1 or new accuracy. Prelaunch GPU/disk values remain explicitly a launch snapshot.

Doc82 is the unchanged2329227 B local baseline, SHA `fe46002afa3ad606a5491dbf0d3f2f86946d6711999bb63e21518ddc571aa62e`. Four copies match; all three local HEADs match the previous receipt and are clean. The remote copy is supported by the prior publication receipt only. Doc83 preserves the byte prefix and has a unique new section. The exact remote evidence destination is `/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006/accepted_m0_fit_launch`.

The35 payload names (36 with .gitattributes) exclude NPZ, weight, .aris and credential files. Existing path containment, exclusive creation, byte checks, restricted staging, three commits, primary-main push verification and one-occurrence guard update are retained. Actual publication success remains an execution outcome.

The earlier replacement-order bug is resolved: the generated state uses `reference_fit_launch`, the receipt uses `fit_controller`, and docstring/memory text now describe the active full fit. In-memory generator replay reproduced publisher bytes exactly; generator, publisher, record script and remote embedded Python pass AST parsing.

Current SHA bindings:

- prepare_fit_publication.py: `2a766cd5ea50811e3465ed3cb2de2d4ec393c1cc8a5b6e88575d3fee3543957d`
- publish_fit_launch.py: `628b6317e0bca3c885f14bcb041c21d7cb520870b2c94582a22620145114e378`
- SOURCE_REVIEW.json remains `e5849720caf8894ac0ae33b9208a17edc72070e5c72f564fbe3efd58fbac6ed4`; all101 source-bound inputs still match.

The JSON report binds 47 reviewed files. It does not attest current remote liveness, a reviewer backend identity, a new accuracy result, or the planned selected-checkpoint M2 restore.
