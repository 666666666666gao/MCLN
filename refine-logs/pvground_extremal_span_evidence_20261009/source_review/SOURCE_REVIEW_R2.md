# R2 minimal source follow-up

Date: 2026-10-09. Reviewer: the same `/root/pvg_extremal_span_source_20261009` agent continuing its completed fresh R1 review. Requested route remains Codex / `gpt-6-astra` / `max`; actual backend, model and effort remain **UNATTESTED**. `review_independence=same-family`, `acceptance_status=provisional`. R2 is a continuation, not a newly fresh reviewer call.

**Neural source verdict: PASS, STATIC_ONLY. Both concrete R1 blockers are resolved. No remaining concrete source blocker was found in this minimal change.** This is not an executed checkpoint, actual serialization/restoration, gradient witness, M0 pass, performance result or launch approval. The runner and real two-step GPU M0 remain absent.

**Closed diagnostic verdict remains R1 WARN overall / deterministic cached arithmetic PASS.** Its inputs are unchanged. The full 9,508-row diagnostic was not rerun in R2, and its GT-conditioned counts remain non-deployable and ineligible for training labels or model promotion.

## Exact reviewed revisions

The current bytes were read directly and diffed against the immutable R1 snapshots. R2 copies and hashes are saved in `r2_source_snapshots/`, `R2_SOURCE_SNAPSHOTS.json` and `R2_INPUT_HASHES.json`.

| Artifact | R2 SHA256 | Change |
|---|---|---|
| `span_refinement_model.py` | `1139094df8f282715a795469cedc0dd6c4c95fea3f6b68ce13ab56b4a6ffb68b` | Native dictionary forward; two small checkpoint helpers. |
| `extremal_span_mixer.py` | `523203b7a5304ba41ca3bfb66c6911777adb0b55287806ddfed57d8ae9566adf` | Comments only, following the separately recorded parent message. |
| `matched_span_objective.py` | `fa2df7209430d9587891b6519b447b6dc0c23aef37e6c89b609f6bd337a145c8` | Unchanged bytes. |
| `METHOD_PROPOSAL.md` | `f210fa47bf8628d93d74f38df13400e352a7aa49c743c9ad54b66222c5a289d8` | Explains the repaired contract while retaining the actual-M0 limitation. |

All 38 files covered by `R1_ARTIFACT_HASHES.json` were verified unchanged before and after the R2 source check. No R1 raw response, machine result, input hash manifest, failed invocation, source snapshot or trace was overwritten. The original native-dictionary mismatch witness and official Torch v1.10.2 state/derivative source observations remain intact.

## Resolution of R1 findings

**B1 resolved.** R2 `span_refinement_model.py:41-44` now declares `forward(self, inputs)` and executes exactly one `self.parent(inputs)` inside the existing no-grad scope. The parent accepts that same single dictionary. Python signature binding accepts the established positional input; AST inspection counts one parent call with exactly one positional argument and no keyword dispatch. `apply_span_mixer` and the constructor are AST-identical to R1. No actual model forward was used for this check.

**B2 resolved at source-contract scope.** `span_checkpoint_payload` at `:47-52` now carries `source_mode`, `mixer_state`, `optimizer`, `step` and `parent_identity`. `restore_span_checkpoint` at `:55-61` first asserts the saved mode equals the model's mode, then asserts the supplied parent identity equals the saved identity. Only after both checks does it strictly load the head state, load the optimizer state and return the saved step. Thus a different declared source mode or parent identity is rejected before either load. The mode remains external metadata rather than another trainable tensor, preserving the 14-tensor architecture.

The helper does not itself reconstruct or hash the parent, and the identity argument is caller-supplied. That is an explicit future-runner obligation, not evidence of an implemented parent reconstruction. The payload is a state dictionary for subsequent serialization; no real checkpoint write/read, tensor equality or Adam restoration occurred here. No claim that metadata equality proves the actual live parent's state is accepted without the future runner's identity derivation and restoration witness. This resolves the missing source contract identified in R1 without pretending to complete M0. The original state-dictionary observation remains supported by the preserved [official Torch v1.10.2 Module source](https://raw.githubusercontent.com/pytorch/pytorch/v1.10.2/torch/nn/modules/module.py).

**W1 wording resolved.** The later message changed only `extremal_span_mixer.py:75-78`. It now distinguishes archived absent-or-degenerate invalid references and describes zero pooled evidence separately from a possible bias-derived encoder token. Direct AST comparison with the R1 mixer is identical; no geometry, empty-support logic or fallback was added. The exact follow-up message is retained in `002-source-audit-r2.request.json`.

## Unchanged mechanisms and limits

The executable mixer and objective are unchanged, so the R1 source findings about native member mapping, all 256 candidates, all tied extreme-source SPs, positive convex-axis dimensions, zero-output algebra, native loss coefficients/averaging and lack of GT/diagnostic/score/ranking inputs are carried forward. Architecture remains **29,793 parameters / 14 trainable tensors per arm**. The new helpers have no compatibility dispatch, exception path, model call, hidden GT input, score path or new trainable module. There is no extra parent forward.

The proposal at `METHOD_PROPOSAL.md:33` explicitly calls the repair a source contract and defers actual restoration to M0. Its `:39-48` continues to treat identical paired initialization, one shared parent cache, a common pre-span frozen-parent assignment, exact optimizer/state restoration, nonzero first-step output gradient, nonzero second-step source-encoder gradient and runtime measurements as unexecuted requirements. These remain necessary when an actual runner is prepared. R1's parent-eval/RNG and active-runtime-attestation limits also remain; no speculative source refactor is requested.

The R1 v1.10.2 derivative observation remains a static source fact, not a real gradient observation: clamp includes raw zero in its propagating interval. The official source was preserved rather than downloaded or tested again. No B8 batch, actual head, optimizer, GPU or training process was used in R2.

## A-F continuation status

| Check | R2 finding |
|---|---|
| A. GT provenance | PASS at static inference-source scope: changed functions accept state/identity metadata and the existing input dictionary, with no diagnostic/GT injection. Actual future matching/split wiring remains unverified. Closed cached-GT provenance warning is unchanged. |
| B. Score normalization | PASS, STATIC_ONLY: no changed scoring or metric computation. |
| C. Result existence | PASS for truthful source status: the two fixes exist at the bound hashes; no runner/M0/model result is claimed. |
| D. Executed code | WARN for neural execution: AST/hash/signature checks ran, but checkpoint helpers and neural model were not executed. |
| E. Scope | WARN for runtime evidence: the review covers only this minimal source repair; no actual restoration, training, accuracy or paired-initialization evidence. |
| F. Evaluation type | `STATIC_ONLY_SOURCE_PREPARATION`; it is not an executed model evaluation. The unchanged diagnostic remains `real_gt_conditioned_offline_representational_diagnostic`. |

## Actual verification receipts

`verify_r2_002.invocation.json` records a real exit **0**, chunk `601b7c`, using the already cached offline uv Python. `R2_STATIC_VERIFICATION.json` records native positional binding, one parent-call AST, both identity checks before loading, strict head loading, unchanged executable mixer/constructor/application, unchanged diagnostic/objective inputs and unchanged R1 artifacts. Unified diffs are `R2_WRAPPER.diff`, `R2_MIXER_COMMENTS.diff` and `R2_PROPOSAL.diff`.

The first R2 attempt actually failed at the expected-unchanged mixer hash, chunk `332dcc`, exit **1**. The parent then supplied the authorized comment-only revision and requested that exact revision be audited. The failed source, log and invocation are preserved as `verify_r2_source.failed001.py`, `verify_r2_001.log` and `verify_r2_001.invocation.json`. The second check explicitly admits the stated comment change, binds its hash, and requires executable AST identity; it does not silently ignore the changed file. No numerical diagnostic was repeated.

The two R1 blockers are closed at the requested source scope. No current launch is approved. Real two-step M0 and any runner execution still wait for the active job's actual closure and required handling, and require their own actual receipts.
