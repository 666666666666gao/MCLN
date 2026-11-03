# Reader Anchor Supervision Source Check

## Scope And Evidence

2026-11-03:source-only investigation while the registered Span pilot
continues. No new Reader implementation,reference-code execution,model
import or local ML test. Complete both pilot conditions before choosing
another trained method.

Measured motivation is the immutable B2 audit,not a presumed lack of
all contextual reasoning:3388of3604 strict050 oracle-to-selected misses
are multiple-instance examples. That oracle gap is not recoverable
accuracy. MCLN already contains language-conditioned spatial candidate
attention;see PVG_READER_BBF_INTERFACE_AUDIT.md.

The public CoT3DRef repository was resolved with git ls-remote and read
with git show at commit ad05bddca844c81d6c7933915f5f83bd547327df.
The shallow filtered cache has no working checkout and no code was run.
Public GitHub API requests hit a rate limit;no API response or empty
web-search output was treated as source evidence. No credentials recorded
or embedded.
The pinned README calls this the implementation,but its paper/website/
citation fields are empty. This note verifies code behavior,not paper
identity,reported benchmark accuracy or reproducibility of its training.

| Pinned File | Git Blob SHA1 |
| --- | --- |
| README.md | 08f13c766ebb5e0422744a14e1c62c4493a38fd8 |
| refering_codes/MVT-3DVG/referit3d/models/referit3d_net.py | 3ec94011ec0ff1b62ee60e24cf2fa5edbd9af4b9 |
| refering_codes/MVT-3DVG/referit3d/in_out/pt_datasets/listening_dataset.py | 74731589396a0722fc71be2d938c6875440e88d5 |
| refering_codes/MVT-3DVG/referit3d/models/referit3d_net_utils.py | 90c134584cf549f54256d04aa7e91b45064705be |

## What The Implementation Actually Supervises

The model constructs a sequence of labelled anchor positions followed by
the target position,and applies object-index cross entropy. This is
explicit role/anchor supervision,not merely an added pairwise spatial
attention weight. Language classification can also supervise anchor
classes. Primary source,compute_loss:
https://github.com/CoT3DRef/CoT3DRef/blob/ad05bddca844c81d6c7933915f5f83bd547327df/refering_codes/MVT-3DVG/referit3d/models/referit3d_net.py#L222

Its cross branch first predicts anchor/target slots in parallel,takes
argmax object indices,gathers the predicted object features,and conditions
a Transformer decoder on those features. The causal branch instead
predicts an anchor before its target step. Do not describe the cross
branch as an autoregressive natural-language reasoning chain. The
inspected prediction path selects anchors from logits,not GT anchor IDs:
https://github.com/CoT3DRef/CoT3DRef/blob/ad05bddca844c81d6c7933915f5f83bd547327df/refering_codes/MVT-3DVG/referit3d/models/referit3d_net.py#L387

The optional distractor head consumes shared aggregated visual features.
Its BCE target is ONE for an object with the target's class that is NOT
the target. A high score therefore means a distractor label under that
objective;it is not an IoU/target-quality score. A separate head also
does not imply detached backbone gradients:
https://github.com/CoT3DRef/CoT3DRef/blob/ad05bddca844c81d6c7933915f5f83bd547327df/refering_codes/MVT-3DVG/referit3d/models/referit3d_net.py#L265
https://github.com/CoT3DRef/CoT3DRef/blob/ad05bddca844c81d6c7933915f5f83bd547327df/refering_codes/MVT-3DVG/referit3d/in_out/pt_datasets/listening_dataset.py#L299

## Why This Is Not A Direct MCLN Patch

The dataset reads anchor IDs/paths,builds a context of segmented scene
objects with same-class distractors,and explicitly inserts target and
anchor objects. This is a different candidate protocol from MCLN's
full-scene predicted queries. It is a protocol distinction,not an
allegation of leakage within the reference benchmark:
https://github.com/CoT3DRef/CoT3DRef/blob/ad05bddca844c81d6c7933915f5f83bd547327df/refering_codes/MVT-3DVG/referit3d/in_out/pt_datasets/listening_dataset.py#L90
https://github.com/CoT3DRef/CoT3DRef/blob/ad05bddca844c81d6c7933915f5f83bd547327df/refering_codes/MVT-3DVG/referit3d/in_out/pt_datasets/listening_dataset.py#L141
https://github.com/CoT3DRef/CoT3DRef/blob/ad05bddca844c81d6c7933915f5f83bd547327df/refering_codes/MVT-3DVG/referit3d/in_out/pt_datasets/listening_dataset.py#L183

The README's NR command also includes SR augmentation. Its initialization,
input candidate protocol,supervision and evaluation are not an equivalent
control for our current NR scratch run. Do not copy its launch recipe
into the running NR/SR240/140budget experiments:
https://github.com/CoT3DRef/CoT3DRef/blob/ad05bddca844c81d6c7933915f5f83bd547327df/README.md

## Constraints For A Later Reader Experiment

### Actual Current ScanRefer Training Inputs

Remote source/data-only audit19:31:22UTC verified live worker240217's
source cwd and current config:use_sacr_source=false,augment_det=true.
The raw TRAIN JSON contains36665 records,each with only ann_id,
description,object_id,object_name,scene_id,token. Fields anchors,
anchor_ids,anchors_types,anchor_boxes,path have0present/0nonempty records.
Raw train SHA256:
93b0bd2a884de8077ba659aa1ee341dd7f571b32c46986e0ce388d6aad349521.
This is the language annotation count;it does not replace the augmented
48655-entry training budget. No validation annotations were read.

The actual source load_scanrefer_annos lines585-732 initializes both
anchors and anchor_ids to empty lists. Its structured-source branch is
disabled by this run's config. Source SHA256:
f6031ee7ffb3f92a9da4611741532a4d9e0bbd1d4cfe3b28b9d9a047e41a9346.

This does NOT mean the baseline lacks contextual/auxiliary supervision.
The source's Scene_graph_parse can select the first non-root Object
entity. Its _get_auxi_boxes lines1161-1186 then chooses the valid
non-target scene box of that parsed class with minimum squared distance
to the GT target center,subject to its existing dis_min=100 cutoff.
This is a source-defined target-nearest heuristic,not an explicit
human-labelled relation/anchor object identity. The current audit did
not measure runtime coverage,parser/cache correctness,which relation
the selected object satisfies,or downstream model input usage.
It neither accuses the baseline of inference leakage nor establishes
that it provides CoT3DRef-equivalent anchor supervision.

Consequently do not silently reuse these heuristic boxes as ground-truth
semantic anchors or directly port object-index anchor cross entropy.
A later Reader experiment must explicitly choose and disclose its
training supervision,quantify any heuristic label coverage/quality,and
map object identities to predicted queries while preserving inference
without GT anchors. No Reader code or running recipe was changed.
Helper:tools/audit_scanrefer_train_anchor_annotations.sh;raw receipt:
results/mcln_pvground_20261101/scanrefer-train-anchor-annotation-audit-20261103T1931.json.

### Which Auxiliary Path Is Actually Consumed

Follow-up source-only audit19:59:10UTC verified the active worker's
actual losses.py and main_utils.py,not merely a historical local helper.
The criterion forward at losses.py1702-1740 constructs auxi_target,
but its matcher call is commented and auxi_indices is explicitlyNone.
AST inspection finds0body uses of the auxi_indices argument and0direct
auxi_box/auxi_entity_positive_map field references in each native
loss_pos_align,loss_boxes,loss_masks,loss_sem_align function.
Thus this native path does NOT implement auxiliary-object matching or
an explicit object-index anchor loss merely because those fields exist.
Do not turn the existence of heuristic auxiliary boxes into a diagnosis
that an active anchor-matching loss is causing the current failures.

This does not mean that other-entity/relation language has no training
effect:the existing semantic loss still consumes other_entity_map and
rel_positive_map. Nor is it a statement about every optional/historical
loss in the repository. Current config explicitly disables
use_parent_relative_text_verifier,use_joint_query_quality_reranker,
use_sacr_source,use_sacr_score_refiner,use_source_moe and
use_source_choice_selector;relation_counterfactual_aux_loss_weight=0.
Some disabled optional heads retain nonzero default loss weights;
the weights alone do not establish an active branch.

The actual _get_inputs at main_utils.py6690-6694 returns point_clouds
and text only. Both inspected train/evaluation paths call model(inputs)
before appending batch labels into end_points for losses/evaluation.
No auxi_box is passed through that input function. This specifically
rules out interpreting its training-label dictionary as a direct
GT-anchor model input through this entry point;it is not a blanket
audit of every possible source of data leakage.

Actual source SHA256:losses.py
aca2a501fed32a551569eb35079e147466bb21f66868143b59d8db5ad1a05fac;
main_utils.py
ec60c8b32e5d2d29b12f0387e39fd3813f1f6a457f10c72500fab0d075981b72.
Helper:tools/audit_scanrefer_auxiliary_supervision_source.sh;receipt:
results/mcln_pvground_20261101/scanrefer-auxiliary-supervision-source-audit-20261103T1959.json.
No model/dataset imports,forward,evaluation,tests or source changes.

An anchor-conditioned Reader hypothesis would need verified train-split
anchor supervision availability and coverage,a mapping to real predicted
query/Hungarian identities,and an inference path using only predictions.
These inputs/mappings have NOT been established by this source check.
Do not import segmented GT candidate context or GT anchor identities into
our inference. Any new annotations,language-model labels or cross-dataset
training data would require explicit protocol disclosure.

Specify positive/negative semantics and gradient destinations before
adding a distractor or verification objective. Native unmatched-query
"not mentioned" supervision must remain coherent. Any score intervention
must reach actual BBF;uniform candidate scalars before token softmax
cancel,as the existing interface audit shows.

Explicit anchor supervision and predicted-anchor conditioning already
exist in this reference;copying either alone is not our innovation.
A later method must resolve a measured remaining MCLN failure beyond its
existing spatial attention,retain Reader/Support/Span roles,and be tested
against its matched trained control and B2. No accuracy gain or target
attainment follows from this source comparison.
