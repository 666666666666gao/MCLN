# ScanRefer REC Hierarchical Risk-Controlled Reranking Design

## Status And Decision

Approach A is approved. Work proceeds in three ordered and isolated phases:

1. re-evaluate the immutable best system and report true position-alignment
   ScanRefer Unique and Multiple metrics at strict IoU thresholds 0.25 and
   0.50;
2. replay the rejected selective-residual five-fold OOF experiment with the
   original numerical protocol and add complete failure diagnostics;
3. use train-only evidence to build a hierarchical query-then-variant
   reranker with an OOF-calibrated abstention policy.

The first phase is descriptive reporting only. Its validation results cannot
be read by either later training phase, cannot select a configuration, and
cannot change a threshold. The current best deployment remains authoritative
until a new method passes every train-only, calibration, provenance, and
official-evaluation gate.

The paper objective remains a fresh 9,508-expression official evaluation with
strict position-alignment `Acc@0.25 >= 0.60000`, `Acc@0.50 >= 0.47000`, and no
inference-time ground truth.

This roadmap is implemented through three separate plans, one per phase. A
phase must finish with its own tests and sealed evidence before the next plan
starts. The first plan covers only the position subgroup report; it does not
modify residual training or create the hierarchical model.

## Immutable Baseline

The best system is the frozen epoch-71 backbone followed by the parent query
reranker and geometry reranker. All three files are mode `0444` and remain
read-only:

```text
backbone  3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208
parent    f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b
geometry  835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f
```

The authoritative total position metrics are currently `5542/9508 =
0.58288` at 0.25 and `4621/9508 = 0.48601` at 0.50. The old log's unqualified
`unique`, `multi`, `unique50`, and `multi50` values are semantic-alignment
statistics. They are not position-alignment subgroup results and must not be
reported as such.

Every formal phase snapshots each protected file's path, device, inode, mode,
size, mtime, ctime, and SHA-256 before and after execution. Any difference is
a hard failure. No phase may modify, rename, copy over, or chmod a protected
file.

## Phase 1: Position Subgroup Re-evaluation

### Metric Contract

`multiply` in the request maps to the ScanRefer **Multiple** split represented
by `is_unique == false`; **Unique** is `is_unique == true`. The evaluator adds
position-only namespaced accumulators for:

```text
position / unique   / 0.25
position / unique   / 0.50
position / multiple / 0.25
position / multiple / 0.50
```

The same run also reports position easy/hard and
view-dependent/view-independent splits at both thresholds. These additional
cuts cannot replace the four requested Unique/Multiple values. Subgroup
updates occur only for `prefix == "last_"`, Top-1, and `only_root == true`. A
prediction is a hit only when IoU is strictly greater than the threshold;
equality at 0.25 or 0.50 is a miss.

The new keys are separate from the existing semantic keys and mask keys.
Changing the active geometry candidate or its score must affect the position
subgroups exactly as it affects the total `bbs` metric, while leaving semantic
subgroups unchanged. At each threshold,
`unique_total + multiple_total == 9508` and
`unique_hits + multiple_hits == position_top1_hits`.

### Frozen Report Run

The run uses the same model arguments, dataset order, batch size, worker count,
GPU, and three artifact paths as the authoritative best run. It writes to a
new report-only directory under:

```text
/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/
    geometry_position_subgroup_reports/
```

It does not reuse the old one-shot claim or write into
`geometry_official_val`. The report records the exact subgroup hit count,
denominator, full-precision ratio, and five-decimal ratio; the exact overall
counts; command and environment; code digests; protected-file snapshots; and
these flags:

```text
report_only=true
eligible_for_model_selection=false
selection_uses_validation=false
inference_uses_ground_truth=false
```

Ground truth is used by the evaluator only after inference to measure IoU and
assign the pre-existing ScanRefer subgroup. It is never an input to candidate
construction, scoring, or selection.

The report is authoritative only if the fresh total position counts reproduce
`5542/9508` and `4621/9508`, the Unique and Multiple denominators sum to 9,508,
their hits reconcile with both totals, and every protected snapshot is
unchanged. A mismatch is preserved as a non-authoritative diagnostic rather
than silently normalized or retried until a preferred value appears.

The workspace has no Git metadata. The report therefore records a canonical
SHA-256 manifest for the tested source files and this design instead of
claiming a Git commit identity.

## Phase 2: Selective-Residual OOF Diagnostic Replay

### Numerical Parity

Replay uses the existing 506 fit scenes only. The following are unchanged:

- scene split and seed-0 five-fold mapping;
- the 12 configurations formed by hidden dimension, weight decay, and break
  cost;
- optimizer, learning rate, batch size, ten epochs, clipping, and seed;
- no-switch sentinel and the existing eight positive-gain percentiles;
- strict 0.25 and 0.50 labels, selection ordering, bootstrap seed and 10,000
  scene-level replicates;
- the original eligibility and tie-break predicates.

Instrumentation is downstream of the existing predictions and selection
inputs. It must not alter tensor values, iteration order, random-number
consumption, training batches, margins, or the selected choice. The replay
opens only the bound train and geometry-train caches plus the frozen parent
and geometry artifacts. The 56 calibration scenes and every validation path
remain inaccessible.

### Required Diagnostic Record

For every configuration and every available margin, persist:

- configuration identity, margin percentile/value, fold sizes, fold mapping
  digest, OOF gain digest, and prediction count;
- baseline/proposed hits and deltas at both thresholds;
- switches, abstentions, fixes, breaks, neutral switches, and switch rate;
- all five fold deltas at both thresholds;
- scene-bootstrap estimate, one-sided 95% lower bound, and seed/replicate
  count at both thresholds;
- positive and all-valid gain counts, minimum, maximum, mean, standard
  deviation, and fixed quantiles;
- each eligibility predicate as a named boolean and the exact list of failed
  predicates;
- train-label distributions for break/neutral/fix at both thresholds,
  separated by held-out fold and by same-query versus different-query pairs.

The no-switch candidate is recorded explicitly. If no residual candidate is
eligible, the failure receipt reports `selected=baseline`, preserves the full
diagnostic table, and represents calibration as `not_run`; it must not write
hard-coded zero calibration hits that look like measurements.

The diagnostic schema includes enough canonical counts and SHA-256 digests to
compare two identical replays. Large per-candidate tensors need not be placed
in JSON, but their canonical ordered digests must be retained.

### Publication Safety

Before replay, both the audit report writer and experiment publisher are
changed to fresh-only publication. A final directory is reserved with an
exclusive create, every file is created with no-replace semantics, and the
read-only completion receipt is written last. `os.replace`, overwrite-capable
renames, preflight-then-rename publication, and reuse of a failed output path
are prohibited. A partial directory is non-deployable and cannot contain a
completion receipt.

The formal replay runs inside the file-access audit. It is valid only when the
runner exits zero, the audit reports `pass=true`, `violations=0`, and
`validation_data_accessed=false`, and protected snapshots are identical.

## Phase 3: Hierarchical Risk-Controlled Query-Variant Reranker

### Input And Hierarchy

Each expression retains its deployable `16 queries x 7 variants` structure.
The frozen caches provide:

- 152D query features and query validity;
- 25D geometry-variant features and variant validity;
- frozen default/parent scores, parent Top-1 identity, geometry scores and
  ranks;
- stable query positions and variant indices.

Training-only candidate IoUs create labels and diagnostics but are not model
inputs. Feature normalization is fitted separately inside each OOF training
fold and then refitted on all 506 fit scenes. No normalization statistic comes
from held-out scenes.

A variant encoder maps each valid 25D variant feature to an embedding. Masked
elementwise mean and maximum pooling are concatenated to summarize the seven
variant embeddings for each query. The query encoder combines that summary
with the 152D query feature, normalized default and parent scores/ranks, and
default/parent Top-1 flags. It predicts query-level hit probabilities at 0.25
and 0.50. The query label at a threshold is whether any valid variant belonging
to that query strictly exceeds the threshold.

Both query and variant heads use a monotone two-logit parameterization:
`P50 = P25 * sigmoid(conditional_50_logit)`. This guarantees
`0 <= P50 <= P25 <= 1` without post-hoc clipping.

The first stage selects one query by the weighted expected utility
`2 * P(hit@0.25) + P(hit@0.50)`, with stable frozen-axis tie breaking. The
variant head then combines the selected query embedding with each of its
seven variant embeddings and frozen geometry indicators, predicts
variant-level hit probabilities, and selects one valid variant by the same
weighted utility. The head may score all valid variants in a batched tensor,
but only variants belonging to the selected query participate in the second
argmax. Thus inference explicitly selects a query first and a variant only
inside that query; it does not flatten all 112 candidates into a single
learned decision.

### Objective And Predetermined Search

The loss is the sum of row-balanced query and variant binary cross-entropies.
Threshold losses have fixed weights 2:1. Invalid candidates contribute no
loss, variants are averaged within a query before queries are averaged within
a row, and rows are averaged equally. False-positive cost multiplies the
negative-label BCE term and protects against replacing a correct frozen
decision with a miss.

The first experiment uses a bounded OOF grid only:

- shared hidden dimension: 64 or 128;
- weight decay: `1e-4` or `1e-3`;
- false-positive cost: 2 or 4;
- AdamW, learning rate `3e-4`, batch size 256, 12 epochs, gradient clipping
  1.0, dropout 0.1, deterministic seed 0.

There is no fold-held-out early stopping. Any later search requires a new
written protocol and still cannot read the 56 calibration scenes.

### Risk-Controlled Policy

The frozen geometry Top-1 candidate remains the default action. For each OOF
row, the hierarchical model proposes one query-variant pair and computes its
predicted utility gain over the frozen candidate. A switch is possible only
when the proposal differs, both candidates are valid, predicted gain is
positive, and it exceeds one OOF-selected intervention margin.

Margins use the same no-switch sentinel and fixed positive-gain percentiles as
the residual experiment. A configuration/margin pair is eligible only if all
five folds have non-negative exact hit deltas at both thresholds, pooled
0.25 delta is positive, and scene-clustered one-sided 95% lower bounds are at
least zero at both thresholds. Eligible pairs are ordered by
`2 * delta_hits025 + delta_hits050`, then larger margin, fewer switches,
smaller hidden dimension, larger weight decay, and larger false-positive cost.
If no pair is eligible, the baseline is retained and the 56-scene calibration
partition is not opened.

After OOF selection, one model is refit on all 506 fit scenes and evaluated
exactly once on the untouched 56-scene calibration partition. It must reach
at least `3524/3625` at 0.25, preserve at least the cache baseline
`3315/3625` at 0.50, and preserve candidate-oracle and raw-query-IoU digests.
The separate online deterministic calibration must preserve its authoritative
`3316/3625` 0.50 baseline.
Only then may a fresh official 9,508-expression run be launched. Failure at
any stage publishes no deployable model and leaves the three current best
artifacts unchanged.

## Data Flow And Isolation

```text
frozen train caches
  -> 506-scene, five-fold OOF diagnostics/model selection
  -> fixed configuration and margin
  -> refit on all 506 fit scenes
  -> one 56-scene calibration gate
  -> optional online calibration reproduction
  -> optional one-shot official validation

frozen official validation
  -> Phase 1 subgroup report only
  -> no edge back into any training or selection stage
```

Separate processes and output roots enforce the missing edge. Training and
audit code deny validation paths. The Phase 1 report is never passed as a
runner argument, imported as a module, or placed below an allowed train-cache
root.

All three phases share the same fresh-only publication rule: reserve a new
directory exclusively, create files without replacement, write the completion
receipt last, and never turn a partial directory into evidence of completion.

## Testing And Verification

Tests are written before implementation and must cover:

- position and semantic subgroup accumulators are distinct;
- geometry score overrides affect position subgroup results only;
- exact-threshold equality is a miss;
- Unique/Multiple reconciliation and distributed accumulator merging;
- existing total position and semantic metrics are unchanged;
- report parsing rejects missing, duplicated, rounded-only, or inconsistent
  subgroup counts;
- report-only validation cannot satisfy or influence a model-selection gate;
- protected artifact tampering and output-path reuse fail closed;
- diagnostic collection preserves selector inputs and random-number state;
- every failed OOF predicate is named and measured, including no-switch;
- calibration fields are `not_run` rather than fabricated when OOF fails;
- hierarchical shape/mask contracts, row-balanced loss, stable ties,
  baseline parity, scene-fold isolation, normalization isolation, strict
  thresholds, artifact binding, and disabled-path parity.

Before any result is reported, run the focused CPU suites, the existing REC
regression suites, a subgroup parser dry run, and the full frozen evaluation.
After the evaluation, verify exact count reconciliation, fresh output
publication, log/result digests, process exit status, and unchanged protected
snapshots.

## Alternatives Considered

1. Parse the old `unique/multi` lines. Rejected because those lines measure
   semantic alignment, not the active geometry position decision.
2. Compute subgroups from the frozen validation sidecar cache. Rejected as the
   authoritative answer because its 0.50 total is `4623`, while the online
   official evaluator recorded `4621`; it is useful only as a diagnostic.
3. Add only a flat 112-candidate model. Rejected because the prior residual
   experiment already used that decision surface and discarded the query and
   variant hierarchy that the cache exposes.
4. Chosen approach: instrument the authoritative online evaluator for the
   report, diagnose the unchanged residual experiment, then use an explicit
   query-then-variant model with train-only OOF risk control.

## Non-Goals

This protocol does not tune on subgroup performance, change the backbone,
regenerate boxes, add validation examples to training, alter the official
metric definition, overwrite an old run, or claim that a report-only
validation metric is evidence for selecting the next model.
