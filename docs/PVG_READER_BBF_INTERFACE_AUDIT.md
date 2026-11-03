# Reader And BBF Interface Audit

## Scope

2026-11-03:read-only audit while the registered Span comparison runs.
No new Reader implementation,model import,forward,evaluation or test.
The immutable B2 diagnostics motivate the investigation:3388of3604
strict050 oracle-to-selected misses are multiple-instance examples.
An oracle gap is not attainable accuracy or proof of a particular cause.
Do not modify the running pilot or restart the completed B trainer.

## Running Source Identity

The source-only helper verified that live GPU worker236606 has cwd
`scanrefer_span_gate_comparison_20261103/source`. It parses source with
Python AST without importing any model code. Receipt:
`results/mcln_pvground_20261101/scanrefer-reader-bbf-source-audit-normalized-20261103.json`.

Five complete files were hashed:encoder_decoder_layers.py,
candidate_local_visual.py,mcln.py,losses.py,and grounding_evaluator.py.
All five LF-normalized Git blob identities equal the corresponding
objects at repository commit e73dc4a. Raw remote SHA256 values are retained.
The raw Git blob identities differ because deployed files have CRLF;
normalization is ONLY for the identity comparison,no source was changed.
The four checked-out working files also match remote raw SHA256 exactly.
The sparse encoder file was compared to its committed object.

| File | LF-Normalized Git Blob SHA1 |
| --- | --- |
| models/encoder_decoder_layers.py | f4d2ee64e71764884c68b9a48dbbc35e8fd6faaf |
| models/candidate_local_visual.py | ff1a7d34f0ba8d5c19786bc2906fd7a8fc15883b |
| models/mcln.py | ee41a859f8a124f023b3f29977e30e57a2b28d3b |
| models/losses.py | 9b3164ae5a09258d0b436e2b04b840f1469bcbc7 |
| src/grounding_evaluator.py | 32e7bba873d8d1b50aafa5e4990df3bd8e99604b |

## Context Already In The Baseline

The running `BiDecoderLayer` already calls `MultiHeadAttentionSpatial`
with candidate-pair center geometry and a max-pooled language feature.
Its default conditional branch computes language/query-conditioned
spatial weights,applies sigmoid to the geometry interaction,and adds
log spatial weights to ordinary attention logits before softmax.
Both candidate self-attention and language cross-attention precede Reader.

Reader then reads64raw-point neighbors separately for each candidate,
using interpolated SA1 features,RGB and relative coordinates. Its learned
local residual is added to visual cross-attention before LayerNorm/FFN.
The resulting shared query feeds both the normalized64-dimensional BBF
projection and native box heads.

Thus "add language-conditioned pairwise attention" is not an absent
baseline mechanism. The Reader itself has no explicit competitor-local
witness input,but its incoming query already contains scene/language
context. Neither fact establishes insufficient relational reasoning.
The fixed-B2 Reader bypass changing geometry with the same selected query
is compatible with this shared prediction path;it is not a trained ablation.

## Actual BBF And A No-Effect Placement

The running `evaluate_bbox_by_sem_align` computes

```text
logits[q,t] = dot(normalized_query[q],normalized_token[t]) / 0.07
prob[q,:] = softmax_over_tokens(logits[q,:])
score[q] = main[q] + modifier[q] + pronoun[q] + relation[q] - other[q]
```

These role scores are token-map weighted sums of the same probabilities.
BBS instead uses classification logits. Changing only a BBS helper is
therefore not a BBF verifier.

For a proposed candidate scalar v[q],adding v[q] to every token logit
before this softmax has exactly no effect:

```text
exp(logits[q,t] + v[q]) / sum_u exp(logits[q,u] + v[q])
= exp(logits[q,t]) / sum_u exp(logits[q,u])
```

This is a constraint on a FUTURE implementation,not a bug found in the
current Reader. The current query residual can change relative token
similarities and is not the canceled scalar construction.
A future explicit score correction would have to affect token-specific
comparisons or the actual post-aggregation candidate score. Any changed
scoring rule must be declared and compared against unmodified BBF;
renaming a BBS score or silently changing an evaluator is not acceptable.
Prefer investigating evidence inside Reader's existing residual path
before introducing a separate scoring branch.

## Supervision Assignment

`loss_sem_align` initializes unmatched queries with the final two
"not mentioned" tokens. Hungarian-matched queries overwrite that map with
their assigned object mention;modifier/pronoun/relation maps are also
assigned through those matches. Existing object-to-token and token-to-object
losses already train semantic separation.

Consequently,a hypothetical all-query/rootGT positive objective could
compete with existing unmatched-query supervision. This is an observed
assignment difference to resolve before implementation,not a measured
training failure or proof that a detached ranking head would fix it.
Any future verification experiment must identify its eligible candidates,
positive/negative targets and gradient destinations explicitly.
No GT box,GT object label or selected-oracle identity may be inference input.
The current Span pilot already uses actual Hungarian matches and does
not change the native semantic assignments.

## Pinned Primary Implementations

Only source text was downloaded/read locally;no reference code executed.

- Official MCLN commit9744a4ed219062d448ed0dba587eeb864491f158,
  `models/encoder_decoder_layers.py`:
  https://github.com/qzp2018/MCLN/blob/9744a4ed219062d448ed0dba587eeb864491f158/models/encoder_decoder_layers.py
  Its conditional spatial attention and decoder language pooling already
  implement the baseline mechanism above. Downloaded file SHA256:
  6be03732f8067a6d6138d7a4f33378e7448df61ed9bda641db9c590160edd1d0.
- ViL3DRef commit82a95d9e41ecd9920d678823eda1b366947da204,
  `og3d_src/model/cmt_module.py`:
  https://github.com/cshizhe/vil3dref/blob/82a95d9e41ecd9920d678823eda1b366947da204/og3d_src/model/cmt_module.py
  The conditional branch also uses language-conditioned sigmoid geometry
  weights with log fusion. It is a useful source comparison,not evidence
  that copying that attention adds a new capability to our baseline.
  Downloaded file SHA256:
  3f1b182ddaa5e03a2ae958cd23855cfa03668c682972501cfe68feeba5fd944e.

## Next Decision

Finish the current matched full-epoch Span control/treatment first.
Then choose the smallest Reader experiment supported by the measured
remaining failures. It must specify what competitor-local or role-specific
evidence is new beyond existing spatial attention,retain Reader/Support/Span
responsibilities,and compare against both its own matched control and B2.
Zero residual must preserve the chosen parent;real GPU updates,full9508
evaluation and own CPU checkpoint/SHA receipts remain required.
The supervision/score-placement checks above are not yet a launched or
validated Reader method. No target is claimed reached.
