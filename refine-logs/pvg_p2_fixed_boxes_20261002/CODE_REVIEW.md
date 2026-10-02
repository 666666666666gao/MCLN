# PV-Ground fixed-box diagnostic review

**PASS — no remaining concrete correctness blocker.** The replay success-label issue found in the initial runner was corrected by the parent and re-reviewed.

Static pre-execution review, 2026-10-02. Reviewer request: `gpt-6-astra / max`. Attribution: **same-family/provisional**; independently verified cross-provider or cross-family separation is not claimed.

- Reviewed `run.py` SHA256: `8e3e598f8cb8444e4c553e2f0de9bea0b129c59d25eef6df2be52521a3531bb7`
- Reviewed `controller.py` SHA256: `8b3542ea5229c0750131abb66306ee4fd4c0f3eeffc8b27438d080fe6bce8794`
- Reviewed `PLAN.md` SHA256: `25e43a1413c038c591d492631f5efdc6646db32e8298b75fcec6876c5d53a29b`

Paths: `run.py`, `controller.py` and `PLAN.md` mean files in `C:/Users/gb/.codex/tmp/pvg_p2_fixed_boxes_20261002/`. `sealed run.py` means `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/run.py`. Model filenames refer to that sealed source's `ported/models/`; the task/evidence modules and evaluator are directly under its `source/`. `REPORT.md` is the sealed `complete/REPORT.md`. Full input paths and hashes are in `CODE_REVIEW.json`.

## Resolved finding

The initial runner (`b3d60317503ff56a812f86488173b65778281be23ae9abb880dd1ee26792133f`, lines 394–404) wrote `status=complete` and printed `FIXED_BOX_COMPLETE` before asserting historical replay. A failed replay therefore retained success-labelled artifacts. Current `run.py:394–406` saves `replay_mismatch` on failure, preserves the diagnostic, and emits completion only after the assertion passes. No further change is required.

## Checked behavior

- **One trained P2 state and one full forward.** Author state, G delta and final P2 delta are restored strictly; terminal identity is pinned to 60371c5ecf6127f4738aa39d77e754a414a03a302ccdbc605173e1612692d9e9. Only model(inputs) at line 312 performs a full forward per batch. Evidence: `run.py:127-164`; `run.py:204-213`; `run.py:277-280`; `run.py:308-317`; `g_p2_spec.json:2-14,25-29,49-51`.
- **Bypass removes only the final semantic P2 addition.** The source-reader hook stores the original residual tuple. P2 addition creates a new tuple out of place. The wrapper computes actual and bypass decoder tails from their respective residuals without subtraction, stores only bypass semantic values, and returns the original actual pair. Evidence: `run.py:290-303`; `encoder_decoder_layers.py:526-535`; `pvground_task_observation_query.py:25-30,33-53`; `pvground_expression_evidence.py:37-62`.
- **Hook lifetime, tensor dimensions and evaluation state.** Only the last layer has task_read; cache clears before each forward and exact keys are asserted. Decoder tails map Q,B,288 to B,Q,288; the semantic head maps B,288,256 to B,256,256 logits. model.eval disables tail/head dropout and freezes BatchNorm statistics; no_grad prevents graph retention. Exact actual-head replay and final buffer equality are runtime gates. Evidence: `run.py:280,285-318,375-377`; `pvground_task_observation_query.py:38-53`; `modules.py:89-108,169-178`.
- **Actual all-256 boxes, masks and contrastive outputs stay fixed.** The counterfactual is a shallow copy replacing only last_sem_cls_scores. Center, clamped size and both mask structures are identical objects; other fields, including contrastive projections, adaptive weights and superpoints, remain shared. No candidate subset or GT box filter is used. The discarded bypass geometry never reaches predictions. Evidence: `run.py:134-136,314-335`; `pv_ground.py:496-517,519-552`; `modules.py:145-175`; `grounding_evaluator.py:56-66,232-249,594-605`.
- **Official bbs score and manual hit accounting.** Both paths softmax token logits, binarize only the main positive map, add modifier/pronoun/relation terms, subtract other-entity terms and rank all candidates descending. Thresholds are strict >0.25 and >0.5. Native/manual hit equality is required for each arm. Manual size-product and native corner-volume IoU are mathematically equivalent but bitwise equality is not claimed. Evidence: `run.py:305-306,345-363,379-390`; `grounding_evaluator.py:217-303,535-553`; `losses.py:35-75`.
- **Input binding and offline-only GT.** The sealed formal dataset/loader/prepare path and seed are retained; both runs use batch 8 without augmentation. Only point/voxel/text/detected-box/superpoint inputs reach the model. Real GT is attached after inference for evaluation. Historical records are used only for row/point/GT and selected-query/hit replay, not as prediction labels. Evidence: `run.py:50-85,93-125,216-235,257-271,279-284,309-312,320-344`; `sealed run.py:196-263,362-399`.
- **Row scope, new output files and historical gate.** Only 8 or 9508 rows are accepted, fresh output mkdir is required, ordered row IDs and final row count are checked, and gzip JSONL writes a real newline per row. All boxes, IoUs and both score arrays are retained. A replay mismatch now saves status=replay_mismatch and exits before FIXED_BOX_COMPLETE. Evidence: `run.py:39-49,281-284,337-370,375-406`; `controller.py:26-51`.
- **Immutability and pilot/full controller.** There is no optimizer construction/update or checkpoint save in the diagnostic. Named buffers and terminal file hash are compared after evaluation. The controller uses the existing environment and GPU lock, invokes the pilot and full serially in fresh directories, requires child exit 0 and complete receipts, checks pilot replay and available disk before full, and writes only logs/status/diagnostic outputs. Evidence: `run.py:304,308,375-378`; `controller.py:12-17,26-54`; `PLAN.md:14-26,33-37`.
- **Removing native_loss does not add a second stochastic path.** The inspected matcher/loss path has no random sampling call; it adds loss fields rather than replacing the bbs predictions. The native Gumbel sampler still runs once per model forward. Historical replay remains a runtime requirement rather than a claimed result. Evidence: `sealed run.py:265-270,371-378`; `losses.py:298-390,562-641,808-846,849-973`; `pv_ground.py:576-590`.
- **Scientific interpretation.** This is a direct final-semantic forward intervention at the trained P2 state with actual P2 geometry fixed. It is not a trained no-P2 model, an estimate of total P2 training contribution, or a causal explanation for original-G retention loss. Evidence: `run.py:1-3,402`; `PLAN.md:3-12,28-31`; `REPORT.md:23-30`.

## Evidence limits

- Local source text and file hashes were inspected. The reviewer did not execute the runner/controller, import model code, run AST or neural tests, use SSH/GPU, or edit source.
- Remote checkpoint bytes, deployed module origins, dataset bytes, environment availability, disk capacity and runtime assertion outcomes were not independently observed in this review.
- The supplied launch context says the existing environment and sources remain unchanged and output directories are new; the verdict applies to that context and the recorded hashes.
- Input replay checks bind row ID, sampled point bytes and root GT box. Historical output replay checks bind selected query and threshold-hit decisions, not exact historical logits or every box coordinate.
- Pilot 8 and full 9508 numerical equality, native/manual counts and historical replay have not yet run. PASS is a source review result, not an experiment result.
- Only the final semantic head is counterfactually recomputed. Contrastive outputs and all geometry/mask predictions intentionally remain from actual P2; no bypass bbf effect is claimed.
- The development-validation and single-seed limitations in the sealed REPORT.md remain applicable.

All ten originally requested files were read directly. The subsequently requested controller and plan were also read directly; `losses.py` received a targeted additional inspection. Only the two review reports were written by this reviewer. No runner/model execution, SSH/GPU use or source changes were performed.

Required changes: **none**. Runtime acceptance still requires the existing pilot and full-run assertions to pass.

