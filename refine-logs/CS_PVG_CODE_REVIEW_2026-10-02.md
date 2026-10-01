# PV-Ground + CS source review — 2026-10-02

**Verdict: no remaining blocking source issue after the corrections reviewed below.**
This is a fresh same-family, provisional review. It is not an external scientific
review, a successful runtime preflight, or an accuracy result. No experiment,
remote training change, or implementation edit was performed by this reviewer.

## Scope and evidence

Reviewed the current working copies of:

- `docs/CS_PVG_SCANREFER_PLAN_2026-10-02.md` and the updated research contract;
- `models/cs_pvground_modules.py`;
- `scripts/prepare_cs_pvground_source.py`;
- `scripts/train_cs_pvground_scanrefer.py`;
- the existing `src/joint_det_dataset.py` and `utils/scatter_util.py`.

The comparison source was the supplied verified parent snapshot at
`C:/Users/gb/.codex/tmp/pvground_cs_restart_20261002/source`, including
`models/pv_ground.py`, `pv_backbone.py`, `pv_utils.py`, `modules.py`, `losses.py`,
`main_utils.py`, `train_dist_mod.py`, and `wandb_config.yaml`. The exact runtime
`src/grounding_evaluator.py` was subsequently copied into that snapshot and read.
The review concerns the new working files, not an already committed Git diff.

## Standards

No remaining documented-standard violation was found. The changes stay within
the requested PV-Ground port; the constructor corrections address concrete
upstream behavior and do not introduce fallback paths or unrelated refactors.

## Spec and correctness

### Issues found and resolved during review

1. **Shared configuration mutation would break the second preflight model.**
   Upstream `VoxelSetAbstraction.__init__` changes the configured MLP lists in
   place (`pv_utils.py:341–343, 361–363`), and `pv_backbone.py:29` passes that
   configuration directly. Reusing `cfg` for the improved and native models
   therefore adds layers on the second construction and prevents strict parent
   loading. Current `build_model`, at trainer line 179, passes
   `copy.deepcopy(cfg)` on every construction. This fixes the cause.

2. **Native preflight construction used a different working directory.**
   PV-Ground reads `data/class_embeddings3d.npy` relative to the current directory
   (`pv_ground.py:187–189`). The later native construction happened after the
   trainer changed to `dataset_source`; the initial construction used
   `model_source`. Current trainer lines 177–185 explicitly scope both
   constructions to `model_source` and restore the caller's directory. This
   fixes the inconsistent resource resolution without changing the parent.

3. **Optimizer reload evidence was initially too weak.**
   The original probe changed only a model tensor and left the optimizer intact.
   Current trainer lines 347–354 also corrupt a populated Adam `exp_avg`, restore,
   and compare it with its saved value. All model delta tensors are compared as
   before. This is a concrete optimizer restoration check; it does not constitute
   an independent comparison of every optimizer tensor or a resumed trajectory.

4. **The preset best rule is now explicit.**
   The plan records `min(hits025/5572, hits050/4797)`, then total hits, and
   distinguishes those reference scales from the development gate 5544/4754.
   The implementation agrees with this declared rule.

### Reviewed invariants

- **M1 sources and membership:** the six slices match the actual upstream
  concatenation order: BEV, raw points, then x_conv1 through x_conv4. Their widths
  match the supplied configuration. FPS indices select the actual raw-point
  superpoint labels; encoded seeds are added as a separate source. Counts are
  membership availability, not measured VSA neighborhood quality.
- **Insertion order:** seed enhancement precedes the native cross-encoder;
  superpoint enhancement follows native grouping. M2 reads the encoded scene and
  nearest 16 available superpoints before the last two prediction heads. M3
  refines those heads' boxes through differentiable Query-Mask support. R consumes
  final geometry evidence and changes only the final semantic input. The final
  semantic head executes once; contrastive projections and final query masks
  retain the pre-R query. The independent `cs` arm remains available.
- **Zero initialization and RNG:** all added terminal residual projections are
  zero initialized. Their forward paths introduce no nonzero dropout. Upstream
  Gumbel sampling remains stochastic in evaluation; preflight resets the same
  seed immediately before each native/improved forward and compares final boxes,
  semantic logits, and both mask outputs. Both models use the shared center
  computation. GPU equality remains unmeasured by this review.
- **Native losses:** the generated port does not edit `losses.py`; the trainer
  uses the parent criterion and all native intermediate/final supervision. The
  dataset preserves ScanRefer language identity for joint ScanNet samples, and
  its integer 0/1 masks are converted to Boolean for the native loss. There is no
  added quality, teacher, or G objective.
- **Validation:** validation loads only ScanRefer, disables training augmentation,
  iterates without dropping the last batch, and requires 9508 examples and 9508
  evaluator GT counts. Official `bbs` uses final semantic scores and IoU against
  `center_label` plus `size_gts`; `only_root=True` selects the referred object and
  `filter_non_gt_boxes=False` disables detection-box filtering. The official
  evaluator also calculates its native diagnostic metrics, but the runner reports
  and selects only the declared `last_`/`bbs` top-1 result. No alternative ranking
  is fused into that result.
- **Parent and delta state:** strict loading requires exactly 1234 parent state
  entries, complete key/shape/dtype agreement, and only the declared CS additions.
  The known RoBERTa `position_ids` buffer must equal its canonical range and be
  absent from the checkpoint before it is made nonpersistent. No learned tensor
  is skipped. Delta state includes every trainable parameter and every nontext
  buffer, including mutable batch-normalization state. The frozen text encoder
  remains supplied by the immutable parent. Resume constructs and loads that
  parent before applying the complete checked delta and optimizer.
- **Training and claims:** fresh optimizer, epoch 0, seed 2027, the 21-epoch LR
  recipe, clipping, and separate new/core/backbone rates agree with the plan.
  E0, each epoch, preset best, and the fixed endpoint are recorded. Historical
  results are not relabeled as this model's results; no accuracy gain is claimed.

### Documentation note resolved on final readback

The plan now explicitly says that **superpoint centers** use the same
deterministic mean in native and improved arms, while the extra raw/FPS member
means exist only when M1 is enabled. The earlier imprecise wording is resolved;
the implementation and valid native/improved center comparison are unchanged.

## Verification and remaining runtime gates

Python 3.10 AST parsing passed for all three implementation files. Replaying the
source generator's literal replacements entirely in memory found all 11 anchors
exactly once, and both generated model files parsed successfully. No generated
source was written by that check. The initial generic Python command failed
because its local launcher had no `pyvenv.cfg`; the check was rerun successfully
with an existing explicit Python interpreter, without changing an environment.

This reviewer did not load the actual checkpoint or run CPU/GPU model execution.
Subsequent CPU-load evidence is recorded below; the real GPU preflight remains
required. In particular, 40 GB capacity cannot be inferred from source: native masks/losses,
the six-source features, M2 attention, and M3 support retain real tensors. The
existing two-update probe measures actual requested-batch allocation and checks
terminal/internal module gradients and save/reload behavior. Its result must be
obtained before formal training; it is not a formal accuracy measurement.

Outstanding blocking findings: **0**. Outstanding nonblocking findings: **0**.
No broader redesign is recommended.

## Corrected runtime diagnosis and same-family rescue

**Correction:** the earlier explanation that the R overlay's `src` directory was
a symlink was unsupported and is retracted. The executing parent subsequently
checked the remote directory directly: the selected R overlay has **no `src`
directory**. The incorrect `--dataset-source` allowed Python's import search to
reach `runtime/PV-Ground/src` instead. The strict namespace assertion correctly
rejected that different dataset implementation before checkpoint loading.

The first two CPU attempts therefore had **0 model forwards and 0 optimizer
updates** and did not complete parent loading. Their failures are launch-source
selection failures, not neural comparison results. Per the parent, R remained
running while this CPU gate was unresolved.

Reviewed current trainer lines 146–147: both model and dataset checks compare
`Path(actual_file).resolve()` with `(expected_source / relative_file).resolve()`.
Symmetric resolution retains exact canonical-file equality; it does not relax
the namespace check or cure selection of a genuinely different source. Python
3.10 AST parsing passed after that narrow trainer edit. No further trainer patch
was needed for the actual cause.

Reviewed the corrected launch configuration directly in the local helpers:

- `C:/Users/gb/.codex/tmp/deploy_cs_pvground_20261002.py`, constant at line 17 and
  CPU argv at lines 50–55;
- `C:/Users/gb/.codex/tmp/switch_to_cs_pvground_20261002.py`, preflight argv at
  lines 71–76.

Both now select `/home/gb/new butd/butd_detr-main/MCLN-main` as
`--dataset-source`, the existing corrected dataset root identified by the parent.
Both construct the command with `shlex.join`, so the space in this path remains
inside one argument. The PV-Ground model source stays separate and unchanged.
This fixes the observed configuration error without an import fallback or an
algorithm change.

Also read the resulting local
`C:/Users/gb/.codex/tmp/pvground_cs_restart_20261002/cpu_load.json`. The parent
identifies it as the successful **third CPU execution** under the corrected argv.
It records `status=pass`, `arm=cs_readback`, **1234 core state entries and 70 new
state entries**, **0 GPU forwards**, and **0 optimizer updates**. This supports a
successful strict CPU parent load; GPU equality, capacity, gradients, and
optimizer reload still require the separate preflight.

This is a same-family provisional rescue review of the path configuration and
reported runtime evidence. No helper script was executed by this reviewer, and
no experiment or remote job was changed. No outstanding blocking finding remains
in the corrected source-selection path.
