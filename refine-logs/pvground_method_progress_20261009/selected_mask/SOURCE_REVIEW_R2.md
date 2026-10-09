# SOURCE_ONLY computational review R2 — PASS

No remaining computational source blocker was found. **B1, B2 and W1 are closed at source level.** Actual B8 sanity and the separate launcher review remain outstanding; no training or neural forward was executed here.

Requested reviewer: gpt-6-astra / max. Actual backend: **UNATTESTED**. Independence: **same-family/provisional**.

## R1 findings and their repairs

| Finding | R2 evidence |
|---|---|
| B1: startup threshold assertion | `pair_spec.json:54` is numeric `0.5`; the promotion rule is separate at `:55`. The exact assertion at `run_mask_support_pair.py:143` now evaluates true against the current JSON. |
| B2: missing extra-loss sanity | Both actual preflight steps call `selected_output_gradient_witness` before combined backward (`paired_support_loop.py:132-134`). Its scalar and gradient checks are described below. |
| W1: misleading treatment record | `paired_support_loop.py:135-139` stores `native_matched_loss`, reports the combined total as `loss`, and replaces treatment `expanded_positive_queries` with actual `extra_rows`. |

The completed witness at `selected_query_mask_objective.py:45-92` reads root GT directly from `batch['gt_masks'][:,0]`, verifies equality to target slot 0, recomputes majority-superpoint labels, independently accumulates the four native focal/Dice terms, and compares the weighted scalar using `[5,1,10,2] / len(indices)` at `rtol=atol=1e-6`.

It then uses `torch.autograd.grad(extra, outputs, retain_graph=True, allow_unused=True)` on corrected Query-mask outputs. Eligible selected outputs must have nonzero direct gradient; every other output and every native-matched Query must have exactly zero direct gradient. A preflight with no eligible row is rejected, so the positive-gradient witness cannot pass vacuously. The detached scalar recomputation and output-gradient query leave the normal combined backward intact.

The objective and receipt distinguish selected Queries matched to root, matched to another native GT, and unmatched. Role counts describe the actual B8. A zero count is not a claim that the absent branch was exercised. The witness concerns direct output gradients; shared corrector parameters may still affect other Query outputs after an update, as the plan already acknowledges.

## Fresh computational review

All nine current computational Python files, the spec and the plan were read. Seven Python files and the plan are unchanged from R1. The revised loop and selected objective were read again after the scalar witness was added.

| Requirement | Current source conclusion |
|---|---|
| Same retained warm state | Both independent heads restore the archived nonzero content checkpoint with 7446 support updates and identical parameters; each creates a fresh AdamW optimizer. The pinned checkpoint SHA is `6f9109cb690399880b8dc3bcf6e4b98d1053145732ee2a6b504bfaecbd1eaac2`. |
| Frozen parent, all256, one score | One frozen parent forward supplies both heads. PV/G/reference/Box/semantic/contrast parameters are frozen; all256 Queries remain; both arms use the same single native semantic score. No ranking, quality or score loss is added. |
| Native selection and role protection | The native-bbs top Query is selected by descending argsort. The extra term is skipped if that Query is matched to any native GT, including root and auxiliary instances. |
| Real target and coefficients | The verified dataset places the ScanRefer root in GT slot0 and builds its mask from object point membership. The objective uses native majority-superpoint labels and Query/fused focal/Dice coefficients `[5,1,10,2]`. |
| Budget and empty cases | At most one extra Query per expression; divide by actual batch size, including skipped expressions as zero. All-ineligible fit batches have zero extra loss. The final B2 fit batch divides by2. |
| Same architecture | Both arms have 27841 trainable parameters/10 tensors and geometry disabled. Factory supports `selected_query` with geometry encoding zero. The inherited signed-log branch is inactive. |
| Restoration code | The existing preflight reconstruction compares all1314 CPU state tensors, restores actual AdamW state on CPU/GPU, and tests native integration against paired outputs on the same captured inputs. It explicitly does not claim a cold GPU-parent reconstruction. |
| Warm equivalence scope | Step0 compares both arms' Mask/Box/score outputs and checks masks against the historical warm head. Historical Box equivalence follows from identical masks and the same frozen reference path; no separate historical full GPU forward is claimed. |
| No preflight carry | Preflight returns without weight files. A separate train invocation constructs the protected warm heads and fresh optimizers. |
| Fit schedule | Seed2027, B8, LR1e-5, WD.0005, clip.1;29778 rows exactly once,3723 updates, terminal11169 support updates. Train starts fit directly without initial full/6887 validation. |
| Formal evaluation | Formal mode verifies terminal lineage/fit-row coverage and evaluates all9508 with the sealed native evaluator, preserving row hashes, GT and all256 outputs and reconciling manual hits to evaluator detections. |
| Reporting and research claims | Historical5599/4859 remains historical. No new accuracy, efficacy, novelty or three-contribution claim is made. C3 remains a hypothesis under separate review. |

The remaining `3723`, `7446` and `11169` constants retain their proper round/prior/terminal meanings. `retained_hidden_prior_updates=11169` describes the inherited selected geometry checkpoint history, not an additional support run. No active stale `box_conditioned` arm or signed-log payload remains. The matched helper's zero-expansion field now remains appropriate for control and is overwritten with actual extra rows for treatment.

## Validation and seals

All **9/9** computational files pass standard-library AST parsing with **Python3.7 grammar**, using the installed uv-managed CPython3.13. All nine hashes match `new_runner_files`; all **17/17** existing helper hashes match the local runtime mirror. The existing source/runtime/dependency seals were used; no new framework or package was introduced.

- `pair_spec.json`: `fcdccf7dc1694a266927552057268f9b5d7c90593e23fce37a545a92d45d4a56`
- `EXPERIMENT_PLAN.md`: `4deff45cbe072c5ad6d6b02f447e580c43570b11408b45af5f2e20d734bd5fbb`

| Computational source | SHA-256 |
|---|---|
| `mask_reference.py` | `fe3075a600359ace30b0660b5afb5b1eb5237b2da0e9eb2ad1f9dd6b727463ce` |
| `mask_support_corrector.py` | `14c37bec1babb36d4f7b8fc2cc5883f6487fd403f9543e258e04c0a6c7cc4b7a` |
| `mask_support_model_factory.py` | `e9fd9ad7dab0d18dfc4c9089d85af57eb958d86768ec750c5f609551a8c87ab6` |
| `matched_mask_objective.py` | `67d05d069e1b748477af1a3c84b586375effbc21f1fd99b328ec7029f7e17788` |
| `paired_support_loop.py` | `3d5ce836c19eb90de6a69aa7c2e0fbe27eeb3e731c2d211f0e69a64db8acd484` |
| `run_mask_support_pair.py` | `51413967adca1a644731040565f312856f5854ca85f881c23791bca20b1ab69a` |
| `selected_mask_reference_factory.py` | `09995a92ff93db5d475e2e64430219e83c77ed181213b8210695b88c5a991f84` |
| `selected_query_mask_objective.py` | `3d917a0ca8a14e11ae327875b65179cd0ea4df5de89fa12fc8debe831c3eb702` |
| `support_pair_forward.py` | `74d0c3d2347aabf7087be42e1de4f9a32ffcba8d65b51f831106813cffa8ac9a` |

R1 remains unchanged: JSON `b0c90bda0e09041a2701e9399df41bd6039f9441047dc1ae7ffa8c3bc7b4ef85`; Markdown `00dbd4f0f67a20bbc67bfb0a53018afcb8e6db1ef7b334824aa655f31ac706cc`.

## Scope and next gate

This is a **computational SOURCE_ONLY PASS**, not an actual-result audit. No SSH/GPU/NN, optimizer update, pickle load, installation, credential/memory read, computational-source edit or old-evidence overwrite occurred.

Controller, deployment, polling and fit-launch sources are being reviewed separately. Actual B8 still must produce its two-step scalar/gradient/restoration evidence before formal fit. The exact R2 request is in `SOURCE_REVIEW_REQUEST_R2.txt`; this full response is `SOURCE_REVIEW_R2.md`, with structured companion `SOURCE_REVIEW_R2.json`.

