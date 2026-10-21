# Minimal candidate-pair relation readout prototype

This is an isolated readout prototype. Its first supervised paired experiment
completed on 2026-09-05 and failed both registered P2 screens. See
[the result report](NR3D_PAIR_READOUT_P2_RESULT_2026-09-05.md). This variant
is archived and does not advance to Decoder training or formal promotion. It is not connected to `MCLN.forward`,
the production evaluator. The G0/P1 jobs have now completed. It provides a concrete
implementation of the user's narrower P2 mechanism comparison, replacing the
earlier CEGD draft with multiple simultaneous evidence branches.

## Controlled comparison

Both modes receive the same final 288-dimensional Query features, final boxes,
cross-encoder text sequence and text padding mask. The caller supplies one
shared target Top-32 selection and the complete legal Query mask. The memory
contains all 256 Queries, masked by that legality rule, plus one learned null
anchor. Target selection does not truncate anchor memory. The null state's
five-dimensional geometry is zero; real edges use the existing
`calc_pairwise_locs`, with direction `center_i - center_j` and no distance
normalization. Self-Query memory remains available as in the existing spatial
attention; Query identity is not an annotation of anchor-instance identity.

The global control calls the existing `MultiHeadAttentionSpatial` directly:

```text
t = masked_max(text_memory)
theta_i = lang_cond_fc(q_i + t)
bias_ij = log(clamp(sigmoid(theta_i[1:] · phi_ij + theta_i[0]), min=1e-6))
```

The pair mode computes a separate text context for each ordered edge:

```text
e_ij = MLP(concat(q_i, q_j, phi_ij))
r_ij = MultiheadAttention(e_ij, text_memory, text_memory; text_padding_mask)
theta_ij = lang_cond_fc(q_i + r_ij)
bias_ij = log(clamp(sigmoid(theta_ij[1:] · phi_ij + theta_ij[0]), min=1e-6))
```

Content Q/K/V projections, geometric modulation, residual/layer normalization,
null state and scalar output-head architecture are common. The new mode keeps
the old log-sigmoid formula rather than simultaneously switching to an
unbounded additive relation bias. Its additional pair MLP and token attention
have extra trainable parameters; this is not a parameter-count-matched study.
Common modules initialize identically under the same seed because pair-only
modules are constructed after them. The later runner must explicitly bind the
same shared initialization and frozen backbone checkpoint for both roles.

This is a **post-box readout comparison using the old conditional formula**,
not a reconstruction of the complete old Decoder. In the live Decoder,
position embeddings also enter Q/K and its input boxes are from the preceding
prediction stage. Both prototype modes instead use the same final Query and
final-box inputs. A positive result would justify subsequent Decoder
integration; it would not itself establish a new backbone contribution.

The final scalar head produces uncalibrated candidate logits. They are
scattered onto the original Query axis, with non-selected and invalid Queries
excluded using `-inf`. There is no base-score residual, Source Selector, Parent
fallback, probability interpretation, box update, mask update or root-target /
annotated-anchor label input. The existing `butd_cls` proposal protocol remains
the source of the detector-overlap legality mask.
The primitive does not choose a winner when no target slot is valid; the
experiment must settle that observed candidate contract before integration.

`models/candidate_edge_adapter.py` now makes the shared input construction
executable. It receives `decoder_query_last` explicitly, reads
`end_points['text_memory']`, invokes the existing detector-overlap filter on
the existing detector inputs, and selects Default Top-K after masking illegal
Queries. The exact same returned dictionary can be passed to either readout.
It does not reuse the SourceChoice adapter's all-true validity or its
64-dimensional contrastive Query projection. If the compact selection includes
invalid padding slots, the readout excludes their logits using the same mask.

```python
shared_inputs = build_candidate_edge_inputs(end_points, inputs, decoder_query_last)
global_output = global_readout(**shared_inputs)
pair_output = pair_readout(**shared_inputs)
```

## Registered training comparison

The independent P2 runner and fixed contract now implement one fit epoch of
both addon heads on the same frozen backbone forward. See
[the registered P2 plan](NR3D_PAIR_READOUT_P2_PLAN_2026-09-05.md) for the fixed
split, positive-only listwise objective, initialization, sole decision score,
optimizer, common update count and promotion screens. Uncovered rows are
excluded from ranking supervision and remain part of evaluation. No production
evaluator override or benchmark promotion is part of this experiment.

Both new readouts mask padding, but this only guarantees invariance for fixed
input features. It does not fix or establish invariance of the upstream
RoBERTa, seed selection, Encoder, Decoder or SWA. The completed P1 identity diagnostic found Query reordering and SWA token
changes under added padding; its four selected physical Queries and binary
Masks stayed unchanged. This prototype does not establish whole-model
padding invariance.

A soft anchor distribution and a learned null state do not implement logical
AND/NOT, prove multiple-anchor reasoning, or provide ground-truth anchor
coverage. Those remain hypotheses to evaluate with held-out evidence. The
protected backbone has already seen the train scenes used by the proposed
module holdout; that audit cannot be labeled unseen-scene system generalization.

## CPU validation

`tests/test_candidate_edge_direct_scorer.py`: **10 passed in 1.43 s** in the
existing remote Python 3.7 / PyTorch environment, with CUDA hidden and two CPU
threads. The test package uses the exact immutable G0 spatial-layer file
(SHA `0444ef048acf9e2f760eb9661696a388b9969085eda83bcad0db70f4118c4831`)
and the new prototype source; the running source trees are untouched.

The checks cover:

- equality to the original conditional path when pair text is replaced by the
  same global text context, with identical common weights;
- existing/appended padding exclusion for fixed features in both modes;
- memory outside target Top-K, illegal Query exclusion and explicit null state;
- edge-specific token attention depending on its own memory Query;
- Query permutation, original-axis score mapping and input preservation;
- gradients from the sole score head through pair text, geometry and null state;
- the actual 288-dimensional, Top-32 / full-256 interface.

These are synthetic tensor forward/backward checks, with zero optimizer steps,
zero benchmark rows and no new checkpoint. They establish implementation
properties, not Nr3D accuracy or reliable relation interpretation.

The additional `scripts/audit_pair_readout_prototype.py` probe strictly loads
layer 5 spatial attention from protected checkpoint SHA
`76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`
into both modes, verifies every common initial state tensor is identical, and
checks finite/nonzero score gradients through the new text reader under that
initialization. This passed on PyTorch `1.10.2+cu111` with CPU tensors only.
The global/pair prototypes contain **347,953 / 931,729 parameters**, respectively;
the pair reader adds 583,776 parameters. The combined source/validation receipt
is `refine-logs/PAIR_READOUT_PROTOTYPE_CPU_20260905.json`, SHA
`1f99b5a9484993da3832ce7edb8a4991d728596d70a56c7c6d0e866f89068c8f`.

The input adapter adds **3 passing CPU tests (0.04 s)** against the actual
frozen MCLN Default scorer and overlap filter: filter-before-selection with
full memory preservation, root-supervision independence and invalid compact
slot exclusion. The receipt is
`refine-logs/PAIR_READOUT_ADAPTER_CPU_20260905.json`, SHA
`cbfd3e8f3c62b9ce415973df2b55f0e9188fa4f815a8f2c57f78becfade848e0`.

## Real four-fit-row GPU integration

`scripts/audit_pair_readout_scene.py` **passed** on the A100 after G0 and P1
completed. The launcher acquired the shared GPU lock. It used fixed fit rows
0/1/3/4 and one frozen protected-backbone forward, captured the actual
288-dimensional Query before `x_query`, and passed the same adapted inputs to
both readouts. Shared initial tensors were identical and inputs were unchanged.
Legal full-memory counts were 43/40/29/39; legal target counts were 32/32/29/32.
Finite nonzero derivatives reached pair text, geometric modulation, null state
and the final score head; the backbone had no gradients.

This was a score-derivative probe: **zero optimizer steps, zero weights written,
no new-head accuracy and no formal validation**. It is not positive evidence of
Nr3D improvement. Receipt: `refine-logs/PAIR_READOUT_SCENE_RECEIPT_20260905.json`,
SHA `fead3257a26b024b8187c8113ffce7eecd0f82833beee1b0507b9b42353f3d91`.

The separate P1 candidate audit found full legal memory covered only 5–14 of
31–66 existing detector-object slots on those four rows. This is an object
availability proxy, not annotated text-anchor recall. Full Query memory must
not be described as guaranteed complete anchor evidence. One row had a selected
box IoU of .783 but a full-Query best Mask IoU of .358; reranking alone cannot
repair that example's Mask quality. These four rows do not estimate prevalence.

G0's fixed augmentation completed but failed its preregistered view-dependent
REC@.25 gate (zero net gain). Its augmentation performance route remains sealed.
The user's later P2 relationship comparison is an independent experiment; it
must not be reported as a promoted or successful G0 result.
