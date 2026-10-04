# Boundary distribution experiment source review

**Verdict: PASS. Open blocking findings: 0. Open nonblocking findings: 0.** The source supports the planned real-model engineering probes. This review does not satisfy either probe or independently authorize full fitting.

Scope: `SOURCE_ONLY`; fresh same-family review, acceptance provisional. Requested route: `gpt-6-astra`, max reasoning, `fork_turns=none`; backend SKU was not independently verified. The JSON report lists the actual bytes and SHA256 for all40 original primary files, six supplied lifecycle files, and two correction records (48 files).

## Resolved review finding

**N1 — explicit raw-target-finiteness witness.** The original source checked finite DFL loss only after target clamping, whereas `EXPERIMENT_PLAN.md:50` requires finite-target evidence. There was no evidence of an actual nonfinite target or training failure.

The parent applied the minimal correction. `pvground_boundary_box_refiner.py:53` now asserts `torch.isfinite(target).all()` before clamping; line64 returns `boundary_targets_finite=True`. The existing runner already merges those counts into each step receipt. Removing exactly these two lines reproduces the previously reviewed module digest. All four specs and the source-preparation record point to corrected module SHA256 `88debfef64f4295f87d697b7b9d4cd1eab17b39f4041f6d9ae773fc65dda9461`. I re-read the six changed primary files and both correction records. Loss equations, matching, support and budget are unchanged.

## Correct source behavior

- **Native matching:** `models.losses.py:852-853` orders the seven calls as proposal, last, then heads0–4. `SetCriterion` has one active matcher call per prefix, so the runner's `matching[1]` is the actual final assignment. DFL applies the same GT mask before target indexing and includes every native matched GT. No separate root matcher or matching union is added.
- **Representation and loss:** face order is x−,y−,z−,x+,y+,z+. Scale-four target encoding and decoding agree before the explicit floor. The33 symmetric nonuniform knots, coordinate-distance interpolation, endpoint saturation, matched-box × six-face mean and separate1/7 coefficient are correct. Native loss normalization is retained. Targets are detached, and native-box/DFL-only gradient witnesses are separate.
- **Zero head and dimensions:** both arms apply the common1e-6 reference/final floor in the forward. Centered uniform integration gives zero offsets. Cached-input and native-reference comparisons use this floor and separately count raw native dimensions below it. The implementation does not falsely claim to preserve unmodified negative native sizes.
- **Support and parameters:** 288 query +896 pooled local +3 size +6 normalized coarse +109 range gives1302 aggregate inputs. Seven × sixteen local observations, fused predicted Mask support and whole observed-superpoint evidence remain. Independent counts are400614 residual versus456102 distribution parameters, difference55488. All256 candidates remain.
- **Frozen original G and restore:** only ten refiner tensors train. G stays in eval mode, with frozen-gradient and original-state equality checks. The real probe checks second-step member/condition/aggregate gradients, serializes/restores delta and AdamW in memory, and compares moments, steps, keys and groups. Formal mode strictly restores the terminal head delta on the protected parent chain. These are reviewed source paths, not executed runtime evidence.
- **Budget and pairing:** seed2027, LR1e-5, weight decay5e-4 and clip0.1 match the plan.29778 rows once at batch8 yields3723 updates, with tailbatch2. Seen-row multiset, physical split separation and distribution-arm loader order versus residual are checked. Actual initial cross-process output differences are recorded rather than assumed bitwise equal.
- **Evaluation and GT use:** forward inference occurs before training GT insertion. The adapter receives points and predicted support. Native unique last/bbs scoring, root GT, strict IoU thresholds and fused Mask rules are preserved and checked against the native evaluator. No GT candidate filter is enabled. Full256/rank-bucket coverage stays offline oracle evidence.6887 holdout rows are pretrained-seen;9508 rows are development validation.
- **Claims:** plan and contract correctly describe a representation-plus-supervision package with altered parameter count. They do not claim an isolated architecture/DFL effect, completed face-conditioned decoder, established novelty, accuracy or Nr3D/Sr3D generalization.

## Lifecycle and mutation boundaries

Both launchers require a source review without blockers and verify the existing reviewed-file identities. Preflight requires the previous pair to be closed/audited, checks GPU idleness, uploads exclusive new files with byte readback, and holds the existing GPU `flock`. Its two serial probes perform four disposable updates and save no disk checkpoint.

The formal launcher requires a closed complete engineering intake and separate successful residual and distribution two-update receipts before fitting. It uses the same GPU lock. The controller repeats per-arm gates and runs residual/train → residual/formal → distribution/train → distribution/formal.

Upload destinations are only the new `/root/autodl-tmp/pvground_boundary_preflight_20261004` and `/root/autodl-tmp/pvground_boundary_fit_20261004` roots. Recovery replacement stays inside the current arm. Retention can unlink only a verified owned `terminal.pth` after full formal restore/evaluation and CPU threshold recount; original G is hash-checked and protected. The collector downloads only permitted source/text suffixes and rejects weight files. The observer reads remotely and writes local JSON; the waiter uses its estimated milestone and then240-second observations. No lifecycle tool was executed by this review.

## Actual validation and remaining gate

All31 Python files passed full-file AST parsing; all10 JSON inputs parsed. The current source-preparation, lifecycle and per-spec identities match. Two spec hashes in `control_tools_preparation.json` are historical: reversing only the recorded module-digest correction exactly reproduces their prepared bytes. They are not represented as current spec hashes; the review list contains the actual current identities.

Standard-library scalar checks covered33 knots,32 interval midpoints and two saturated targets (67 cases), with zero reconstruction error, and confirmed parameter/update/tail counts. The local PATH Python alias initially failed before execution with `No pyvenv.cfg file`; an existing interpreter found offline performed validation. No environment change was made. The first report-write attempt stopped before output when its snapshot assertion detected the parent's correction; these final reports describe the re-read corrected bytes.

No model library was imported; no CUDA or SSH call was made; no weight, credential, auth-wrapper or private-goal file was read. I changed no experiment source and wrote only these two review reports.

The remaining gate is the actual batch8/two-update probe for **each** arm, including collected gradient, fixed-state, optimizer-restore, memory, matching, finite-target/saturation and floor evidence. Static checks establish none of the runtime outcomes, throughput or accuracy.
