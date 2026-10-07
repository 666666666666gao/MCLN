# SOURCE_ONLY review: FAIL

Generated: 2026-10-08T02:09:56.108878+08:00

Fresh-context experiment-bridge Phase2.5 review; same-family and provisional. Requested Astra/max is not backend attestation. Actual model/backend/effort: UNATTESTED.

Two blocking source findings; no new model/GPU/SSH/network/checkpoint execution. No source edits.

## B1: Independent native forwards are used as the implementation-equivalence witness

`paired_support_loop.py` lines 191, 192, 194, 195, 199, 204, 207.

The paired output at line194 and integrated output at line199 are produced by separate complete native forwards, with reset_rng between them. Mask equality is then gated at 2e-6.

The existing revision2 evidence_hidden/evidence_visible preflight records explicitly report same-RNG, fresh-input complete-forward drift. Query-mask maxima across their recorded arrays range from 4.1961669921875e-05 to 0.0001201629638671875. Text masks and alpha also drift. These are actual old records, not a new execution or a speculative numerical case.

This M0 gate cannot distinguish a correct correction-head integration from the native repeatability already violated in this runtime; it can reject the unchanged/correct route. Hard Mask thresholds also make Box differences discontinuous, so increasing one scalar tolerance does not repair the witness.

Required correction: Validate helper versus installed native correction/refiner using the same captured original native inputs from one actual integrated forward, retaining the complete CPU rebuild and actual GPU head/optimizer restoration. Treat independent complete-forward differences as recorded diagnostics, with honest limits, instead of implementation equality. Do not widen tolerances until PASS or add a fallback.

## B2: Formal-fit completion requires cross-pass parent Box equality that existing full data contradicts

`paired_support_loop.py` lines 359, 360, 361, 362.

The final train path asserts old["parent_box"] == new["parent_box"] after initial and terminal independent 6887-row passes.

The previous face-support loop records geometry_reference_center/size as reference_box (paired_geometry_loop.py309,341). A stdlib comparison of all6887 original initial/terminal rows found row_id, point_sha256, Query and coarse_box all equal, but reference_box differs at row16804. Both references are valid.

row16804 initial reference_box=[-1.0175853967666626,-2.9135074615478516,0.9893376231193542,0.2782527804374695,0.6530201435089111,0.5007940530776978]; terminal=[-1.0312237739562988,-2.9135074615478516,1.0065933465957642,0.3055294156074524,0.6530201435089111,0.5353055000305176]. Max absolute difference=0.034511446952819824.

A real known native support-threshold drift can terminate the controller after3723 updates and terminal holdout evaluation, before its receipt and complete9508 formal mode, although frozen model state and inputs remain correct.

Required correction: Keep exact input/row/GT and frozen-state checks. Preserve both per-pass parent outputs and report the observed parent/reference differences; verify arm scoring equality within each shared native forward. Do not require independent threshold-derived Box equality as the frozen-state proof.

## Source checks that passed

- Equal27841-parameter heads; zero output initialization; only9explicit coarse-Box channels differ. Common native features retain spatial information.
- Frozen-parent native Hungarian matches retain actual validGT labels. Native majority-SP supervision and coefficients5/1/10/2 match original criterion; no /7 or extra positives/losses.
- Correction affects each Query Mask before native scalar-alpha fusion and detached member-extrema reference. Parent/Box/R/semantic state stays frozen.
- Native evaluator and row metrics use datasetGT and the same bbs-selected Query for Box/Mask; all256 and complete9508 modes are preserved.
- Seed2027,29778rows once/3723updates per arm, independent optimizers, fresh fit construction and CPU/GPU restore checks are source-prepared.
- All declared local/script/helper/environment bindings match. Old/new port lists contain98files with identical path sets; onlyPVforward changes. The briefing's95was corrected to98 with the parent.
- Observer and publisher source plus embedded programs parsed; observer remains first-endpoint/240s and closed-text-only. Publication guards actual launch and states M0hasnotpassed. No management-source blocker beyond updatingB1-related wording.

17 prepared Python sources and3embedded remote programs pass Python3.7 AST grammar via the existing offline stdlib interpreter. No imported-torch/runtime success is inferred.

Full state/capacity/gradient/restore/native GPU tests and formal accuracy remain NOTRUN here. B1/B2 require minimal fixes and a new source review before deployment.

Exact reviewed paths, byte counts andSHA256bindings are in SOURCE_REVIEW.json. Supplemental closed-artifact checks compared all6887 paired rows and read the original repeatability records; they add no new performance or historical-recovery claim.
