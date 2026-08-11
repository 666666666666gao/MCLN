# ScanRefer REC Query Reranker Design

## Objective

Reach both of the following on the ScanRefer validation split using the
`last_ position alignment` Top-1 REC metric:

- `Acc@0.25 >= 0.60000`
- `Acc@0.50 >= 0.47000`

Inference must not use ground-truth boxes or masks. The frozen MCLN checkpoint
plus a lightweight learned reranker counts as one inference system.

## Evidence And Constraint

The best preserved checkpoint is:

`/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth`

Its Top-1 result is `0.57993/0.46340`, while its Top-5 result is
`0.65524/0.56679`. The candidates therefore contain enough recall to reach
the target if ranking improves. Existing source-choice methods cannot do so:
their strongest four-source oracle is `0.58498/0.47013`, because they choose
between each source's Top-1 result rather than reranking individual queries.

## Recommended Architecture

### Candidate extraction

Run the frozen checkpoint once over ScanRefer train and validation data. For
each expression, form a compact candidate set from the union of the highest
ranked queries under the default position score and contrastive text score.
Keep at most 16 unique query indices.

For each candidate, cache only deployable features:

- final projected query feature;
- predicted center and size, normalized by the input scene extent;
- main, modifier, pronoun, relation, and other-entity position scores;
- default score and contrastive score;
- within-sample score ranks, top score, and top-two margin;
- query objectness where the sampled-query mapping is available;
- query-mask confidence, foreground ratio, and text/query soft Dice;
- pooled text projection and query/text cosine similarity.

Training caches also contain candidate IoUs and threshold labels. Validation
caches may contain IoUs for metric computation, but the reranker input schema
must exclude all ground-truth-derived values.

Cache files are sharded so interrupted extraction can resume and temporary
storage stays bounded.

### Oracle gate

Before fitting a reranker, compute candidate-set oracle accuracy. Continue
with ranking only if the candidate set reaches at least `0.62000/0.50000`.
This margin is required because a deployable ranker will not recover the full
oracle. If the gate fails, augment the pool with mask-derived boxes before
training.

### Reranker

Use a small pointwise MLP shared across candidates. Concatenate each candidate
feature with per-sample mean/max context, then emit:

- one ranking logit;
- one `IoU >= 0.25` logit;
- one `IoU >= 0.50` logit;
- one bounded IoU estimate.

Train with a weighted sum of listwise cross entropy, the two threshold BCE
losses, and smooth-L1 IoU regression. The listwise target uses the candidate
with the highest lexicographic quality: first `IoU >= 0.50`, then
`IoU >= 0.25`, then continuous IoU. This directly aligns training with the two
acceptance thresholds.

Split ScanRefer training scenes deterministically into fit and calibration
sets. Select hyperparameters and early stopping on calibration scenes only.
Do not select epochs or thresholds using ScanRefer validation performance.

### Evaluation integration

The reranker returns a score for every original query. Queries outside the
cached candidate-selection rule receive a very low score. The existing
`GroundingEvaluator` consumes this score for `last_ position alignment`, while
the predicted box geometry remains the original MCLN output.

Report:

- Top-1 Acc@0.25 and Acc@0.50;
- Top-5 oracle for the exact candidate rule;
- fixes and breaks relative to the default score;
- easy/hard, unique/multi, and view-dependent slices;
- the exact checkpoint and reranker artifact paths.

## Fallback 1: Mask-Derived Geometry

If query reranking does not pass both targets, derive additional AABB
candidates from fused superpoint masks. Expand selected superpoints back to
the original 50,000 points, remove low-confidence disconnected components,
and take coordinate min/max. Include the regressed box, mask box, and a small
set of coordinate blends in the candidate pool.

Require the augmented pool oracle to reach at least `0.62000/0.50000` before
training a geometry gate. Mask IoU `overall25/50` is diagnostic evidence only;
it is not interchangeable with REC box accuracy.

## Fallback 2: REC-Specific Fine-Tuning

If deployable ranking still misses the target despite sufficient oracle,
fine-tune from the best checkpoint for at most one epoch:

- freeze PointNet++, RoBERTa, and the mask/SWA branch;
- train the decoder, box heads, and reranker;
- restore a nonzero Hungarian L1 box cost;
- disable the source-choice loss;
- reduce mask and consistency loss influence;
- validate frequently and stop at the first calibration regression.

Long continuation training is excluded because the existing runs consistently
degrade after the first short improvement.

## Testing And Acceptance

Unit tests must cover candidate de-duplication, feature normalization, absence
of ground-truth fields in model inputs, oracle computation, deterministic
scene splitting, loss behavior, checkpoint round trips, and evaluator score
override behavior.

The project goal is complete only after a full frozen ScanRefer validation run
prints both `last_ position alignment Acc0.25 Top-1 >= 0.60000` and
`Acc0.50 Top-1 >= 0.47000`. Unit tests, train accuracy, mask metrics, oracle
metrics, or partial validation runs do not satisfy the goal.
