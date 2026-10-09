# SOURCE_ONLY review R1 — FAIL

The sealed pilot is blocked by **B1**, a startup assertion that is false for the submitted spec, and **B2**, missing executable witnesses for the new selected-query objective. No training or neural forward was executed.

Requested reviewer: gpt-6-astra / max. Actual backend: **UNATTESTED**. Review independence is **same-family/provisional**, not externally attested.

## Blocking findings

### B1 — Every run mode fails the primary-threshold assertion

- Locations: `pair_spec.json:54`, `run_mask_support_pair.py:143`.
- The spec stores `"joint: strict >=4850, then maximize wide toward5658"` in `primary_threshold`; the entrypoint requires numeric `0.5`.
- The exact assertion's AST was evaluated against this JSON without importing training code and returned **false**. All four modes fail at this point. This check occurs after model loading and CUDA initialization.
- Minimal fix: restore the numeric threshold to `0.5`; keep the model-selection policy in `selection_rule` or a separately named policy field. Refresh the spec seal.

### B2 — The new-objective preflight witnesses required by the plan are absent

- Locations: `EXPERIMENT_PLAN.md:9`; `paired_support_loop.py:127`, `:130`, `:134`, `:408`, `:410`.
- The code reconciles only the native matched Mask loss, adds the selected-query extra loss, and then checks gradients of the sum on head parameters. Native matched loss alone can satisfy all of those assertions.
- There is no isolated direct-gradient check on corrected Query-mask logits. The actual skip branch exists at `selected_query_mask_objective.py:18`, but preflight has no executable witness that extra-loss gradients exclude matched-root, matched-other, and all nonselected Queries, or that an ineligible expression contributes exactly zero.
- This is a missing required gate, not evidence that the current selected-loss formula is itself wrong.
- Minimal fix: on the actual B8 preflight cache, isolate the extra objective; assert that only the native-bbs-selected unmatched Query can receive direct nonzero gradient, and verify zero contribution elsewhere. Reconcile its scalar from real root-GT majority-superpoint labels with coefficients `[5,1,10,2]` and actual batch size. Record matched-to-root, matched-to-other, and unmatched counts; do not claim an unobserved branch was exercised. These checks must not carry optimizer state into fit.

## Nonblocking reporting issue

**W1:** `matched_mask_objective.py:57` returns `expanded_positive_queries=0`, and its `loss` is matched-only. `paired_support_loop.py:127-156` reuses this record for the treatment after adding extra loss. The top-level expansion map at `:164` and nested `selected_query_objective` are correct, so optimization is unaffected. Rename the matched component fields, or report treatment `total_loss` and actual expansion explicitly.

## Verified source behavior

| Check | Source conclusion |
|---|---|
| Retained checkpoint | New warm path/hash exactly matches the archived content best: `6f9109cb690399880b8dc3bcf6e4b98d1053145732ee2a6b504bfaecbd1eaac2`, local round step 3723, cumulative support updates 7446, geometry zero. Historical hits are 5599/4859. No checkpoint was deserialized in this review. |
| Equal starting heads | `paired_support_loop.py:36-63` loads the warm state, asserts nonzero output weights, copies it to a separate parameter allocation, and creates fresh AdamW optimizers. The inherited zeroing of the last nine input columns is a no-op for the sealed content lineage, which initialized those columns to zero and only receives zero geometry. |
| Frozen state and scoring | One frozen PV/G/reference parent forward feeds both heads. Box parameters, semantic/contrast paths and zero-output readback are frozen. Scores are the original native semantic scores; no second inference score or quality/ranking loss is introduced. All 256 Queries remain. |
| Native selection | The sealed `native_root_bbs` formula matches the official evaluator: root-positive + modifier + pronoun + relation − other-entity token probabilities, descending argsort. |
| Matched role protection | Both arms use the same assignments from frozen uncorrected-parent predictions. `selected_query_mask_objective.py:18` skips a selected Query matched to **any** native GT. The new term therefore does not repeat root loss on an already matched root Query or override a matched auxiliary instance. Runtime witnesses are still required by B2. |
| Actual root GT | The manifest-verified dataset mirror initializes `tids=[target_id]`, appends an auxiliary anchor afterward, and fills masks from actual object point membership (`joint_det_dataset.py:1086-1117`). GT slot 0 is therefore the ScanRefer root. The new objective does not use predicted masks as labels. |
| Loss and budget | Root labels are `scatter_mean(..., dim=-1) > .5`, matching native majority-superpoint labels. The four terms are Query focal/Dice and fused focal/Dice with coefficients `[5,1,10,2]`. One selected unmatched Query maximum per expression; division is by `len(indices)`, including skipped expressions as zero. The last B2 minibatch divides by 2. |
| Same architecture | Both content and selected_query use the same 27841 parameters/10 tensors with geometry disabled. `mask_support_model_factory.py:6-28` reconstructs either current arm with `geometry_encoding='zero'`. The inherited signed-log branch is unreachable for both arms. |
| Full restoration code | `paired_support_loop.py:211-287` reconstructs all 1314 CPU state tensors, compares each tensor, restores actual AdamW states on CPU and GPU, and tests the deployed native-forward corrector against the paired path on the same captured inputs. Mask/Box/score exactness and one semantic-head call are asserted. This code has not been executed here. |
| Restoration scope | This integration check reuses the existing GPU parent; it is not a full cold GPU-parent reconstruction. Its receipt explicitly says that. Step-0 checks compare both arms' masks/boxes/scores and compare warm masks to the historical head. Historical Box equivalence follows from equal masks and the same frozen reference path, rather than a separate historical full-model GPU forward. |
| No preflight carry | Preflight returns without saving weights. A fresh train invocation restores the retained warm state and creates fresh optimizers. The external launcher must enforce the source/actual-sanity gate. |
| Training budget | Seed 2027, batch 8, LR 1e-5, WD .0005, clip .1; 29778 fit rows exactly once, 3723 updates, final batch size 2; cumulative support updates 11169. No automatic initial full or 6887-row evaluation remains in the train path. |
| Formal evaluation | Formal mode verifies terminal lineage and fit-row coverage, evaluates all 9508 rows using the sealed native evaluator, preserves row IDs/root GT/point hashes and all256 box/score arrays, and reconciles manual strict/wide hits to evaluator counts. No new formal evaluation occurred in this review. |
| Historical reporting | 5599/4859 remains historical. Train receipt explicitly avoids a new initial/holdout accuracy claim. C3 remains a hypothesis; this review makes no novelty, performance or three-contribution claim. |

The remaining `3723` values are the current round's step count or the inherited geometry checkpoint's valid step. Prior support is 7446 and final support is 11169. `retained_hidden_prior_updates=11169` is the inherited selected geometry lineage and is not an extra support run. No active `box_conditioned` arm or `signed_log` payload remains. The remaining zero-expansion reporting ambiguity is W1.

## Static verification and provenance

All **9/9 Python files** parsed with the installed uv-managed CPython 3.13 interpreter, using only standard-library AST/hash/JSON/diff operations. The exact file set and all nine hashes match `new_runner_files`. **17/17** existing helper files match the sealed hashes in the local runtime mirror. No new third-party dependency is introduced.

Compared against the prior compressed experiment, four Python files are byte-identical: `mask_reference.py`, `mask_support_corrector.py`, `matched_mask_objective.py`, and `selected_mask_reference_factory.py`. Four are adapted; `selected_query_mask_objective.py` is new.

The native PV/loss files match source port `765a47c0c07758f3e42a426036b27a840df8ea010b63df3682439ff5455e9eff`. The dataset file matches `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d` under source manifest `75cd5f87a8e715c15b458ed1964c2e6f1d35046bb6beffca1b3388d5903bbe36`. The native evaluator matches `39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677`.

Reviewed seals:

- `pair_spec.json`: `17eb93ae6b8821bc849719185f2b00d05b700e8c44215143d651026dd390982d`.
- `EXPERIMENT_PLAN.md`: `4deff45cbe072c5ad6d6b02f447e580c43570b11408b45af5f2e20d734bd5fbb`.

| Python source | SHA-256 |
|---|---|
| `mask_reference.py` | `fe3075a600359ace30b0660b5afb5b1eb5237b2da0e9eb2ad1f9dd6b727463ce` |
| `mask_support_corrector.py` | `14c37bec1babb36d4f7b8fc2cc5883f6487fd403f9543e258e04c0a6c7cc4b7a` |
| `mask_support_model_factory.py` | `e9fd9ad7dab0d18dfc4c9089d85af57eb958d86768ec750c5f609551a8c87ab6` |
| `matched_mask_objective.py` | `67d05d069e1b748477af1a3c84b586375effbc21f1fd99b328ec7029f7e17788` |
| `paired_support_loop.py` | `e53337ce73354fb8e1bb0d02aa5767559fee14f8f2b7efe27503e8ea0f16508f` |
| `run_mask_support_pair.py` | `51413967adca1a644731040565f312856f5854ca85f881c23791bca20b1ab69a` |
| `selected_mask_reference_factory.py` | `09995a92ff93db5d475e2e64430219e83c77ed181213b8210695b88c5a991f84` |
| `selected_query_mask_objective.py` | `d31b488d8956862d752adb6013fb6673ba0b53b60668885fe11deea596f445c3` |
| `support_pair_forward.py` | `74d0c3d2347aabf7087be42e1de4f9a32ffcba8d65b51f831106813cffa8ac9a` |

## Scope and disposition

No SSH, GPU, neural forward, optimizer update, pickle load, dependency installation, credential/memory read, computational-source edit, or overwrite of old experiment evidence occurred.

The external launcher is not among the reviewed nine files. It must bind the reviewed source/spec and successful actual preflight before invoking train, and create `output/content` and `output/selected_query` before `save()` at `paired_support_loop.py:185`. Those are launch preconditions, not evidence of a new source defect. Remote runtime/data existence was not probed.

Fix B1, complete B2, refresh seals, and obtain a new source review before actual B8 sanity. R1 is **FAIL** and is not an actual-result audit. The exact request is preserved in `SOURCE_REVIEW_REQUEST_R1.txt`; this entire response is preserved in `SOURCE_REVIEW_R1.md`, with its structured companion `SOURCE_REVIEW_R1.json`.

