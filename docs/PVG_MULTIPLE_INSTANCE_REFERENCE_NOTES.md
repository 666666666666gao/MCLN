# Multiple-Instance Diagnostic References

Inspected 2026-11-02. This records source-backed options for the observed
ScanRefer subgroup weakness, not a diagnosis or a new method implementation.
Only source was read. No reference repository code, dependency, model or
test was executed on the local machine.

## Observed Problem And Missing Evidence

Current selected ScanRefer checkpoint is epoch8, full9508 examples,
5422/4505 semantic Top1 hits025/050. Unique025/050 is
86.962650%/74.559549%; multiple025/050 is51.774014%/42.613426%.
The multiple subgroup contains8089 examples versus1419 unique examples.
This establishes a subgroup deficit, not its cause: missing proposals,
semantic misranking, noisy support masks and harmful box correction can
all contribute. Formal overall metrics remain those of the same epoch8.

The new offline audit observes chosen/oracle IoU,first good rank,top10 IoUs,
Reader residual/query norms,Support correction/foreground flips,and
native/reference/final selected boxes with Span gates and axis errors.
No real-sample export or complete matched bypass run is available yet.
Do not turn synthetic fixture results into a module-failure claim.

## ViL3DRef

Repository: https://github.com/cshizhe/vil3dref

Pinned inspected commit:`82a95d9e41ecd9920d678823eda1b366947da204`.
Primary source:`og3d_src/model/cmt_module.py`,
`MultiHeadAttentionSpatial.forward` and `CMT.calc_pairwise_locs`.

The spatial descriptors encode object-relative distance and directions.
In the conditional fusion branch,query features plus a text embedding
generate spatial weights and bias. Their interaction with pairwise geometry
passes through sigmoid; log spatial weights are added to ordinary attention
logits before softmax. The source supports several fusion modes; this
description specifically concerns`spatial_attn_fusion="cond"`,not every
configuration or a claim that this setting is optimal for our model.

Possible relevance: only if our complete diagnostic finds adequate box
oracle coverage but wrong top1 choices associated with competing instances
and relational descriptions. Relative competing-object evidence could then
be investigated inside Reader's candidate-conditioned evidence processing.
It does not justify replacing the decoder or adding a fourth main module.
Our decoder queries already contain context; the absence of explicit
pairwise relations in the local Reader is not proof the full model lacks
spatial reasoning. Their object-candidate/teacher-student setting differs
from our end-to-end predicted-box setting; do not compare headline scores
or import ground-truth object-label conditioning into evaluation.

## ConcreteNet

Repository: https://github.com/ouenal/concretenet

Pinned inspected commit:`7ab4628f19f0442fefaff45550a32e2f30b82189`.
Primary sources:`network/modules/baf.py::BAFModule` and
`utils/losses.py::DenseGroundingLoss.forward`.

BAF processes candidate features and language through successive decoder
stages whose candidate self-attention is masked by center-distance radii
(default1.0,2.5,100.0). The stage features are concatenated and projected.
It also includes a camera token/head. This is candidate-level relational
fusion, not the raw-point neighborhoods used by our Reader.

DenseGroundingLoss normalizes instance/sentence features, computes
exponentiated cosine similarity with temperature0.3, and applies a log-ratio
whose denominator sums candidates in the same scene, including the positive.
The source additionally uses selection and camera losses. This description
does not imply copying that loss into MCLN is a fix: our model already has
semantic alignment/contrastive supervision, and supervision assignment and
denominator differences must be compared before adding another objective.
Dense grounding and box grounding metrics are not interchangeable.

Possible relevance: if matched evidence shows semantically wrong candidates
retain plausible masks/boxes, inspect within-scene positive-versus-distractor
separation. A controlled internal Reader/Support experiment could investigate
candidate-specific evidence,without transplanting the entire fusion stack,
camera token or multi-view inference. None is currently implemented.

## Decision Rules For The Three Existing Modules

| Diagnostic evidence required | Next internal investigation |
| --- | --- |
| Good oracle coverage but poor top1,with relational distractors and negligible or misleading Reader residuals | Inspect Reader candidate evidence and competing-instance context using ViL3DRef/BAF as conceptual references; check existing decoder context first. |
| Semantic choice is plausible,but Support changes membership around distractors or foreground extrema and worsens reference/box IoU under its matched bypass | Inspect member-specific support calibration and within-scene separation; compare against existing mask and contrastive losses before modifying supervision. |
| Selected native box hits a threshold,reference does not,and nonzero Span mixing converts the hit to a miss | Inspect extremal-face evidence and gate calibration using the existing MaskDINO/Mask3D references; do not infer this from gate boundary occupancy alone. |
| Oracle coverage is low | Ranking-only changes cannot recover missing candidate boxes; inspect Reader/localization evidence and native/reference geometry first. |

These are conditions for choosing an experiment,not observed findings.
Full matched9508-sample exports,paired identities/point hashes and the
evaluator's own hits are needed. Interventions on one trained checkpoint
do not replace retrained ablations or a matched official initialization.
Keep the three main module roles and dataset tuning policy unchanged.
Run all future tests/diagnostics remotely,when GPU resources allow.

GitHub API lookup hit an unauthenticated rate limit during inspection.
Exact source was instead read through Git object access at the pinned
commits above; no API result or README alone was treated as implementation
evidence. The reference-only checkouts contain no executed experiment.
