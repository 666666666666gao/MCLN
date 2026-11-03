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
