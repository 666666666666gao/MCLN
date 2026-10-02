# PV-Ground runtime-observation review

Verdict: **WARN — the negative final semantic loss is mathematically permitted by the native objective; no new implementation or scientific blocker for the planned bounded comparison was found.**

Review time: 2026-10-03 05:55 CST. Attribution: delegated Codex reviewer, **same-family / provisional**. This is a review of completed sanity artifacts and the actual source formula. It does not replace the prior code reviews, independently rerun the model, or establish efficacy. Prior `EXPERIMENT_CODE_REVIEW.md/json` and `review_round1` artifacts are preserved unchanged.

There is a substantive **effective loss-weighting concern**: many new semantic correspondences receive the native matched-query weight while the denominator remains the original eight matches. The experiment therefore tests the complete target-and-weighting strategy induced by correspondence expansion. It cannot isolate removal of supervision inconsistency as the sole cause of any future result. No mathematically required change follows from the loss sign. The existing storage/authorization hold still prevents full-pair launch.

Task-relative paths use `C:/Users/gb/.codex/tmp/pvground_candidate_consistency_20261003`. `P` denotes `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source`.

## What actually ran

`preflight.exit` is 0. The `G_CANDIDATE_CONSISTENCY_PREFLIGHT_PASS` JSON in `preflight.log` exactly matches `preflight.json`, timestamped 2026-10-03 05:48:01.978417 CST. The recorded imports' existing digests for native losses, the model, and the candidate module match the inspected local files.

The actual receipt reports batch 8, original G strict loading, zero new model states, two optimizer steps, finite gradients, and an in-memory model/Adam state serialization/reload witness. Peak allocated CUDA memory is 20,146,080,256 bytes (18.7625 GiB); measured serialization is 342,194,609 bytes. `load.json` records the original G delta and fresh optimizer. This is mechanical sanity evidence, not a full training result or a disk-checkpoint restore test.

| Forward used by sanity | Original matches | Added correspondences | Expressions with additions | Native final semantic | Expanded final semantic | Weighted correction | Total objective | Total pre-clip gradient norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Capacity, zero optimizer update | 8 | 282 | 7 | 14.554088 | -12.212540 | -1.911902 | 12.027379 | 55.875031 |
| Update 1 forward | 8 | 263 | 7 | 16.027390 | -7.333792 | -1.668656 | 12.788082 | 128.800095 |
| Update 2 forward | 8 | 248 | 7 | 15.656777 | -6.424051 | -1.577202 | 12.822609 | 138.466263 |

All three records have original-native reconstruction error **0.0**. All eight row IDs are in the archived fit partition, and the same eight expressions are reused by all three forwards. These are not three independent batches or 24 distinct examples. The update rows report the objective computed before their corresponding optimizer step. Training-mode stochastic operations and buffer changes also occur, so the changing qualification counts cannot alone establish learning progress or damage.

`cpu_test.json` now actually reports both fixtures passed: no-qualified zero correction/exact query-projection gradient equality, protected other-GT matching, unchanged pair tensors, and qualified expanded-native loss/gradient agreement within the stated tolerances. The fixture's projection tensors are synthetic random inputs; its purpose is formula/gradient equivalence, not representative runtime scale or effectiveness.

## Why a negative scalar is allowed by this exact native formula

### Query-to-token direction

Let `z[q,t]` be the projected query/token dot product divided by temperature. Ignoring only the displayed `1e-6` stabilizers for readability, the native query-to-token term is:

```text
logsumexp_t(z[q,t] + 1[other_map[q,t] > 0] * z[q,t])
  - mean_main(z[q,:])
  - 0.2 * mean_modifier(z[q,:])
  - 0.2 * mean_pronoun(z[q,:])
  - 0.1 * mean_relation(z[q,:])
```

Empty auxiliary groups contribute zero. The code multiplies each query's term by weight 1 for semantic matches and by `eos_coef=0.1` otherwise (`P/ported/models/losses.py:695-702`, `P/ported/models/losses.py:713-747`). The positive coefficients can sum to **1.5**, but there is only **one** log-sum-exp term. They are not normalized into one target probability distribution. The quantity named `entropy` at native line 739 is not used in this direction.

Consequently this is not a nonnegative negative-log-likelihood or KL divergence. For a formula illustration, with all four positive groups present, no other-entity term, and all L logits equal to an admissible constant c, the per-query term is approximately `log(L) - 0.5*c`. With L=16 and c=10 it is about **-2.2274**. This is an algebraic illustration, not an extra experiment or a reconstruction of the recorded rows. Because the query direction sums such terms over queries, its negative contribution need not be offset by the separately averaged token direction. The actual returned combined loss has no general nonnegativity guarantee.

### Token-to-query direction

The native token direction retains a global `logsumexp` across all queries, divides its positive sum by `nb_pos_obj`, adds `-log(nb_pos_obj + 1e-6) / nb_pos_obj`, then applies the native token weight (`P/ported/models/losses.py:752-774`). Adding root correspondences changes the positive sum and its per-token denominator; it does not simply append independent text losses.

There is an additional native detail that matters when interpreting this as a probability loss: positive-logit numerators use boolean support for modifier/pronoun/relation maps, while `nb_pos_obj` sums those maps' actual floating values (`losses.py:718-720`, `losses.py:767-768`). The dataset builds normalized span maps and sums them across spans (`P/joint_det_dataset.py:1043-1060`, `P/joint_det_dataset.py:1648-1651`), so these values need not be binary. This is the inherited formula, not a new implementation mismatch introduced here. Token weights are also assigned by the native overwrite rules, rather than summed as the query-side coefficients are.

The negative entropy-like term by itself is **not** a diagnosis of the observed sign: in the simple binary, single-positive-group case with n positive queries, `logsumexp - mean_positive >= log(n)`, so that direction remains nonnegative after subtracting `log(n)/n` (apart from tiny stabilizer effects). The general weighted native expression is not the same normalized case. The saved aggregate receipt does not contain the separate directional sums or token-map/logit distributions, so it cannot identify how much of -12.212540 came from each direction or term.

### Logit normalization and boundedness

The actual model applies `F.normalize(..., p=2, dim=-1)` to text projections and every query projection (`P/ported/models/pv_ground.py:413-418`, `P/ported/models/pv_ground.py:500-504`). Native temperature is fixed at 0.07. Thus the ordinary contrastive logits satisfy approximately `|z| <= 1/0.07 = 14.285714`, up to floating-point effects. The other-entity modification can double individual logits inside the query-side log-sum-exp; it does not remove the finite bound.

For fixed maps/counts and these normalized projections, the native semantic scalar cannot be driven to minus infinity merely by increasing embedding norms. A negative value therefore does not indicate an unbounded raw-dot-product implementation. Bounded logits do not, however, certify benign full-model gradient norms or long-run optimization behavior. The observed large gradients must be described separately from the sign.

## Replacement is correct, while effective weighting changes substantially

`pvground_candidate_consistency.py:27-33` computes old and expanded targets with the same native method and the same ORIGINAL match count, then returns `(new - old) * (0.5 / 7)`. At the capacity forward, the weighted final semantic contribution changes from **+1.039578** to **-0.872324**, a correction of **-1.911902**. The total arithmetic is:

```text
13.990013 native objective
-0.050731 existing G CE correction
-1.911902 final semantic replacement correction
= 12.027379 total objective (within float rounding)
```

I also checked this arithmetic for both update records. The recorded `loss_sem_align` field is the native seven-head sum before correction, not the replaced final-layer term. After substitution, the weighted seven-head semantic contribution remains **+5.265961**, **+5.688118**, and **+5.850843** on the three forwards respectively. Negative final-layer values and positive total values are thus mutually consistent; neither sign alone establishes success or failure.

The scale concern is real:

- Capacity has 290 semantic correspondences after expansion, compared with eight original matches; `290/8 = 36.25`. There are on average 40.29 additions per expression that has any additions. The fixed denominator does not average these extra query terms back down by the number of added correspondences.
- Each of the 282 newly matched semantic queries changes from native unmatched weight **0.1** to native positive weight **1** in the query-to-token direction. This is required by the declared use of native positive weights. It is additional effective supervision mass even though the explicit `.5/7` coefficient is unchanged. Existing G CE replacement instead retains its eos weighting; the two losses do not gain identical effective strength merely because their selected sets agree.
- Across 8 x 256 queries, the native query-weight sum is `8 + 0.1*2040 = 212`. After the capacity expansion it is `290 + 0.1*1758 = 465.8`, about **2.1972 times** the original. The update records give 448.7 and 435.2, about 2.1165 and 2.0528 times the original. These are exact mask-weight accounting figures, **not** measured loss or gradient multipliers. In particular, 36.25 is not a 36.25-fold total-loss amplification.
- Query-to-token contributions are summed over matched queries, whereas token-to-query positives are normalized per token. Adding many queries can change their relative influence as well as the final head's influence relative to intermediate semantic, CE, box, and Mask terms. Expressions with many qualified queries also receive more query-side weight. The count is prediction-dependent.

Therefore a future controlled result could reflect both changed semantic correspondence and the induced weighting/normalization balance. It would support or reject this declared strategy as a package; it would not prove that contradictory targets were the unique cause of earlier errors. High IoU still establishes geometric qualification only, not correct physical identity.

## Gradient and geometry interpretation

The logged norms are the **total model norm before clipping**, because `run.py:311` records `clip_grad_norm_`'s returned norm. With clip threshold 0.1, the two update gradients receive approximate global scale factors **0.0007764** and **0.0007222**. Finite values and successful Adam state updates prove mechanical progress; they do not show that clipping is harmless over a full pass. Conversely, there is no same-forward control norm or component-gradient decomposition in these artifacts, so the norms do not prove that the new semantic term caused the large values or dominates the update.

The capacity witness reports a finite corrected-semantic projection-gradient norm of 0.956549 and no direct correction gradient into final center/size tensors. Qualification is detached, and expanded semantic pairs are never fed back into box or Mask matching. Those boundaries remain correct. They do **not** imply unchanged geometry learning: semantic and geometry paths share parameters, and changed total gradients/global clipping can alter updates to the shared model and to the balance of regression learning. Actual regression/Mask matching remains unchanged; downstream geometric behavior is an empirical question.

## Decision for the planned comparison

**No new blocking finding. No code or formula change is mathematically required by this observation.** The sign is consistent with the preserved native objective, original-denominator design, exact old-loss reconstruction, and completed CPU/GPU mechanical checks. The observed gradients remain finite and the documented clipping executes. These observations do not supply evidence of a failed optimizer step, altered matching, nonfinite state, or a violated declared formula that would require stopping the scientific comparison.

The bounded same-original-G, fresh-AdamW, same-seed/budget comparison remains a legitimate test of the explicitly defined strategy, including its effective reweighting. Its single pass and separate formal evaluation constrain the conclusion; two updates on one reused batch establish neither accuracy improvement nor harmful semantic/geometry conflict. Keep the native formula, weights and original denominator as declared for this comparison. Replacing the denominator with the expanded count, renormalizing positive coefficients, or clamping negative values would change the objective; none is a correctness repair justified solely by this sign. This review proposes no new experiment or hyperparameter search.

The full pair still must **not launch now** under the existing storage hold. The measured serialization size implies the existing initial pair gate is **1,295,019,283 bytes**, and the per-training-arm gate is **818,606,946 bytes**. The last supplied output-directory inventory is only 437,751,808 free bytes; this review did not refresh it. Adequate actual output-path capacity and any required storage-action approval must be established before launch. No deletion permission is inferred from this review.

## Actual evidence and NOT_RUN limits

Read: `preflight.json`, `preflight.log`, `preflight.exit`, `cpu_test.json`, current candidate module, relevant current runner/witness code, actual native `loss_sem_align`, normalized projection code, and supporting `imports.json`, `load.json`, `preflight_intake.json`, and map/split source. Performed only local artifact parsing, existing recorded-source identity comparison, fit-row membership checks, and scalar arithmetic/formula analysis. No model, GPU, or network job was run by this reviewer; no implementation file was edited.

**Actually completed in the supplied experiment artifacts:** the two CPU fixtures and real batch-8 GPU sanity, with two optimizer updates and an in-memory model/optimizer serialization/reload witness.

**NOT_RUN / not established:** full pair, full-pass stability, new holdout/formal-9,508 evaluation, accuracy gain, novelty, physical-identity correctness of the added candidates, semantic/geometry gradient conflict or dominance, and per-direction attribution of the negative scalar. Disk checkpoint restore and a fresh storage measurement are also not certified here. The two CODE_REVIEW reports retain their historical NOT_RUN statements as-of their own review times and are not rewritten to imply foresight.
