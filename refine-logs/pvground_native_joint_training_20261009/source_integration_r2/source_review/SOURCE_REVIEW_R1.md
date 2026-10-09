# Native PV normal joint training — source review R1

**Verdict: FAIL.** One confirmed source defect affects best-checkpoint retention after training resume. The forward/loss/optimizer integration is present in source; this review is not deployment approval or an actual M0 pass.

Scope: **SOURCE_ONLY**. Fresh reviewer context, native Codex agent `/root/pvg_native_joint_source_20261009`. Requested route: `gpt-6-astra`, reasoning `max`; actual model and effort: **UNATTESTED** because this reviewer has no route receipt. Review independence: **same-family**. Acceptance status: **provisional**.

No SSH, GPU, Torch/model imports, pickle/checkpoint loads, package installation, training, parameter update, network access, or computational-source edit was performed. The requested warm-runtime ledger was absent at the exact supplied path; no credential or global-memory search followed.

## Confirmed blocker B1 — resume loses the retained best checkpoint

**Severity: P1; fix before a resumable run is deployed.**

`source/main_utils.py:188–194` unconditionally constructs a new timestamp experiment directory. `load_checkpoint` restores the complete current state and the scalar `retained_metrics` at lines 134–145, but carries neither the retained best checkpoint nor its location into that new directory. The initial best save is skipped whenever `args.checkpoint_path` is set (lines 361–366). Subsequent `best.pth` writes happen only on a strict improvement (lines 378–380), and the final return is always the new directory's `best.pth` (line 382).

Consequently, an ordinary resume from `latest.pth` followed by epochs that do not improve the previous best writes `latest.pth` but returns a nonexistent `best.pth`. This follows from source control flow; no hypothetical corrupt file or unusual input is required. The historical best weights can also differ from the resumed latest weights, so simply saving latest as best would mislabel the retained metrics.

The minimal correction is to give training resume an explicit directory/retained-best policy that keeps the actual prior best reachable. Reusing the checkpoint's original experiment directory is consistent with the existing best/latest convention. Verify the no-improvement resume branch and preserve the correspondence between best weights and retained metrics.

## Source findings

### Native model and graph: PASS within source scope

- `source/train_dist_mod.py:99–128` constructs the selected `PVGround` through the normal `get_model` route and calls `configure_native_model`. The executable entry restricts the model, ScanRefer dataset, color-only input, BUTD flags, batch size 8 and seed 2027 at lines 405–413.
- Mask support correction is executed inside `PVGround.forward` before prediction-only Mask geometry and Span (`source/models/pv_ground.py:564–617`). The final center/size overwrite occurs before the returned dictionary. The native criterion reads those final `last_center/last_pred_size` tensors (`source/models/losses.py:898–919`); the ordinary epoch loop calls that model, merges dataset supervision afterward, then backpropagates the resulting loss (`source/main_utils.py:439–466`).
- Support Query and superpoint feature projections and soft mask state remain attached (`source/mask_support_corrector.py:39–52`). Span's native center/size, Query/support features and soft text/own/fused mask state are attached (`source/extremal_span_mixer.py:38–59, 89–102`). The old frozen-parent detaches were removed as shown in `R1_span_mixer.diff`.
- Hard thresholding, foreground membership, native member geometry and extremal-source selection remain discrete (`source/native_mask_geometry.py:10–20`; `source/extremal_span_mixer.py:48–76`). This is not a claim that all geometry decisions are differentiable. A zero or saturated Span gate may produce zero upstream gradients on a given step; actual gradient support must be measured at M0.
- The all-256 candidate layout and native semantic score remain. Neither new module defines a ranking head, prunes candidates, or takes GT boxes/masks. `TrainTester._get_inputs` passes points, text, superpoints and native detected-box inputs, not GT targets (`source/train_dist_mod.py:162–174`). GT is merged after forward. The allowed entry disables `butd_gt` and `butd_cls`.
- Initialization leaves the old geometry/R objects uninstalled (`source/native_model_initialization.py:21–22`); their dormant original forward branches are not executed in this architecture. The evaluator receives the same final model output; there is no evaluation-only box replacement.

### Native criterion and training-only additions: PASS within the pinned one-rank, six-layer scope

AST comparison confirms that `HungarianMatcher`, `SetCriterion`, focal/dice/box helpers and the native criterion factory are unchanged from the listed originals. Original matching costs and every native loss coefficient remain, including seven-output averaging and ScanRefer's 0.5 language weighting (`source/main_utils.py:275–283`; `source/models/losses.py:945–958`).

G is the difference between replacement and original eos-weighted last-layer CE, with its original entropy constant and 0.5/7 scaling (`source/pvground_semantic_assignment.py:34–50`). Thus it removes the old label contribution before adding the new one. It excludes every currently matched Query and uses detached current final geometry only to qualify unmatched slots. Its denominator matches the native criterion under **WORLD_SIZE=1**. This is not a multi-rank equivalence claim.

The selected-mask term uses the current native root-bbs winner with stopped selection gradient. It skips that winner if it occurs anywhere in the native match set, including a match to another GT object; only an unmatched winner receives root-mask supervision. It uses the original 5/1/10/2 own/fused focal/dice coefficients and divides by the number of expressions, retaining zero contribution for skipped expressions (`source/selected_query_mask_objective.py:7–41`). Root validity and ScanRefer assertions are present. The extra losses are applied only when the model's training flag is true (`source/models/losses.py:959–970`).

The current matcher runs on the current model's final output. This is deliberately different from the earlier frozen-parent assignments in `runner_v1/paired_span_loop.py:126–136`. The new training must not be described as the same frozen-parent control. `verify_native_replacement` is a witness helper, not an automatically executed M0 test; the old selected-output gradient witness was not retained in the ordinary trainer.

### Initialization, trainability and state: WARN; no concrete payload mismatch demonstrated

The declared construction sequence is official native state, task observation reader, G delta, support state and optional Span state. Official/full-model restoration and module state restoration use strict loading; the G loop checks supplied names exist and verifies each supplied shape/dtype. The source preserves the original trainable core and frozen RoBERTa parameter policy (`source/models/pv_ground.py:183–188`; `source/native_model_initialization.py:23–60`).

Independent AST arithmetic gives **27,841 Support parameters / 10 tensors** and **29,793 Span parameters / 14 tensors**. The stated state arithmetic is **1234 + 37 + 10 + 14 = 1295**. These are declarations and consistency checks, not actual constructor counts, successful state loading, or numerical initial equivalence.

Two limits remain explicit:

1. The original `selected_mask_reference_factory.py:38–42` reconciles the G key set against exactly the native trainable parameters plus retained buffers. The new initializer checks 1072 supplied keys, membership and shape/dtype, but does not independently compare the full expected key set (`source/native_model_initialization.py:36–42`). No incorrect actual payload has been observed. Exact key-set reconciliation is still required in the eventual payload/constructor evidence; the original assertion can be preserved if this invariant is to be enforced in source.
2. Optional Span loading checks source mode and the support-parent SHA only (`source/native_model_initialization.py:54–57`). The earlier Span identity also includes official/G, selected-reference, source-port and environment hashes (`runner_v1/paired_span_loop.py:38–40`). No concrete native initialization spec was supplied in this tree, so complete lineage binding is still pending. This is not evidence that a mismatched parent was loaded.

No requirement is imposed to restore deleted neutral old geometry/R deltas. Their numerical dispensability in the actual reconstructed model remains an initial-equivalence M0 check.

### Optimizer, epochs, seed and restoration: PASS source wiring; B1 retention failure remains

The optimizer is created before DDP prefixing and covers every `requires_grad` parameter exactly once, splitting original core, backbone, the empty frozen text group and both additions (`source/main_utils.py:287–302`). The ordinary training method performs `model.train()`, model forward, original criterion plus additions, zero-grad/backward, clipping, AdamW step and scheduler step (`source/main_utils.py:424–466`). There is no head-only parent freeze or `no_grad` enclosing the native forward.

Both full model and optimizer/scheduler states are serialized; Python, NumPy, CPU Torch and all CUDA RNG states are restored for training resume, and resume starts at saved epoch + 1 (`source/main_utils.py:133–164`). Sampler epoch and loader generator are reset deterministically from the fixed seed (`source/main_utils.py:224–264, 367–369`). This supports an epoch-boundary design only; neither exact mid-epoch replay nor actual round-trip equality was tested.

The one-rank DDP path is coherent in source. The program does not itself assert `WORLD_SIZE == 1`; launch admission must bind one rank. Multi-rank use is outside this verdict: G normalization, evaluator aggregation on only the main rank and checkpoint writes would require separate handling. No multi-seed sweep is present.

### Data, ordering, metrics and artifacts: PASS source design, runtime binding pending

Train/eval DataProcessor instances are separate and mode-specific (`source/train_dist_mod.py:49–50, 131–152`). The new equality assertion ensures any point filtering/reordering fails instead of silently desynchronizing the original superpoint/GT point order. The listed `prepare_data.py` does contain configurable filtering/shuffling/sampling methods; which queue is active depends on the yet-unbound runtime YAML. The assertion is a guard, not proof that the deployed queue passes.

The listed native dataset derives root masks from dataset object point memberships and boxes from the annotated objects, with ordinary train augmentations (`audit_sources/src_joint_det_dataset.py:1086–1119, 1324–1329`). The native evaluator uses dataset GT, a single native bbs top-1 ranking, strict IoU thresholds and expression counts (`audit_sources/native_evaluator.py:209–303, 535–550`). There is no metric normalization by prediction maxima. `native_root_bbs` matches that root expression score.

The ordinary trainer requests root-only evaluation, consumes only `last_`, and asserts both primary bbs counts equal **9508** before returning raw hits (`source/train_dist_mod.py:207–256`). Checkpoint selection compares both hits from the same completed evaluation: complete 5658/4850 gate, strict gate, then wide and strict counts (`source/main_utils.py:167–176, 376–381`). No combining thresholds across models occurs. Fresh-run retention writes only best/latest filenames and temporary atomic-save files; B1 breaks resumed-run best reachability. No accuracy result exists for this native joint-training source.

The historical `SOURCE_PREPARATION` and `ENTRY_PREPARATION` hashes are stage receipts, not current-source seals. `METRIC_PREPARATION.json` exactly matches all 14 current computational source hashes at review time. The captured entry snapshots and preparation scripts explain the staged differences.

## Explicitly pending admission and evidence

These are **not additional observed source bugs**:

- Concrete native initialization manifest; exact official/G/support/optional-Span hashes, flags/source mode, expected G key set and full lineage.
- Deployment overlay and actual Python import origins for `models`, native data/evaluator, `prepare_data`, `whole_mask_range`, modified backbone/decoder dependencies and YAML. The provided historical import receipts are not receipts for this new entry.
- Warm Torch 1.10.2 environment and compiled dependency admission. The new loader omits the incompatible newer `weights_only` argument from the historical loader; no actual Torch import/API execution was performed.
- One-rank real-batch constructor/state checks, initial prediction equivalence, actual trainable counts and optimizer membership, finite losses/gradients and full checkpoint round trip.
- Full training plus validation memory, disk capacity, best/latest/temp checkpoint storage and data-scope admission.
- Actual M0, normal joint-training results, accuracy and direct evidence for three effective contributions.

The frozen pair specification and all supplied original files remained byte-identical during this audit. Its historical results cannot be relabeled as normal joint-training results.

## Evidence and trace

42 exact input snapshots are bound in `INPUT_MANIFEST_R1.json`; **35 Python files parse** with the existing Python 3.10 interpreter using only stdlib AST/file operations. `R1_STATIC_VERIFICATION.json` records declaration arithmetic, unchanged original criterion definitions, current manifest reconciliation and the static B1 control-flow proof. No source changed between intake and verification.

Native failures are preserved: the shell's default Python failed with `No pyvenv.cfg file`; executable lookup returned exit 1 while listing available commands; the first audit-verifier attempt hit `StopIteration` because the workspace junction resolved to a different path spelling. Its exact failed helper is `R1_VERIFY_ATTEMPT_1_ORIGINAL.py`; using the explicit requested root repaired only the audit helper. These are audit-tool failures, not model failures.

`RAW_RESPONSE_R1.md` and the trace response contain this full review verbatim. `R1_NATIVE_TOOL_RECEIPTS.json` preserves the collected native command outputs/failures; initial bootstrap source/skill/policy reads are preserved as exact files and noted separately in trace metadata. The immutable input/report seal is `SOURCE_REVIEW_R1.seal.json`. All runtime/accuracy/effectiveness flags are false.

