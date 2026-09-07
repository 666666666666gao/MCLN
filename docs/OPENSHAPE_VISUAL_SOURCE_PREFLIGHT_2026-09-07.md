# ScanRefer pretrained object visual source: fixed input feasibility

Status: the fixed input and frozen-feature preflight completed on 2026-09-08;
see SCANREFER_PRETRAINED_OBJECT_INPUT_RESULT_2026-09-07.md for the outcome.
The protocol below records the original scope; no REC gain is established.

The recent native-score audit found zero deployed Top-1 disagreements on its
fixed 512 fit expressions. Source inspection also confirms that SourceMoE,
SACR/JQQ, Tier hard-query and Relation-CF already contain root-quality or
deployed-score supervision. Another ordinary score-to-IoU loss would repeat an
existing direction. The next question is whether stronger pretrained visual
evidence can enrich the current box/class object memory.

## Evidence source and scope

Use the official [OpenShape PointBERT G14 RGB checkpoint](https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb),
revision `d771a992218de10b967eea690e8f474aa96dd758`, 388091433 bytes,
SHA256 `34949c162aca01b6fd3147ed7ccf34b448a34bdebc4a857605ea62412ad54fb9`.
The [author repository](https://github.com/Colin97/OpenShape_code) links the
[inference support library](https://huggingface.co/OpenShape/openshape-demo-support),
revision `70dbc29fa30520cb78b4982de671f90600c08685`.
Keep its actual PointBERT/PointNet grouping implementation unchanged. This is
adoption of published pretraining, not an original architecture contribution.

The published encoder was trained on object shapes, whereas our observed
ScanNet inputs are partial, noisy and may contain neighboring instances.
Compatibility and semantic usefulness must be measured; published object
classification results do not establish grounding performance.

## Fixed preflight

1. Select the first 16 rows, in stored order, of the previous 512-row ScanRefer
   fit export (`rows_000_127.json.gz`). No selection by correctness, category or
   feature quality. These are backbone-seen training inputs, never formal data.
2. Reconstruct each native 50000-point XYZ/RGB input from its serialized scene
   and require its byte SHA to match the previous actual forward. Crop the
   existing GroupFree detector boxes, with the same float32 center/size slots.
   No GT instance masks, GT boxes, labels or text affect cropping.
3. The G14 factory asks for 384 FPS centers. Record crops below 384 points as
   unavailable in this initial feasibility test. Do not repeat points to
   imply additional observations. For other crops, retain all points up to
   10000, then sample without replacement with a fixed row/slot seed.
4. Center XYZ and normalize to the unit sphere. Keep gravity on Z and RGB in
   [0,1]. Retain the original point count and normalization contract.
5. Strictly load the pinned pretrained state into the official factory. Run a
   seeded real CUDA witness and then frozen object forwards. Record feature
   norms, finite values, costs, input availability and diagnostic LVIS top-5
   cosine matches using the authors' fixed text features. These labels are not
   ScanRefer category accuracy or REC evaluation.
6. No optimizer, MCLN weight change, validation run or new performance claim.
   An import or kernel pass alone does not authorize a claim of useful visual
   evidence. Inspect actual object outputs and coverage before designing the
   ScanRefer trainable integration.

## Runtime and storage

Reuse the existing conda `bdetr` interpreter (Python 3.7.11,
Torch 1.10.2+cu111). Place only the missing DGL CUDA111 and torch-redstone
packages in an isolated dependency prefix; do not modify the base environment.
The base environment is about 10 GB and both current filesystems have less
than 10 GiB free, so a full environment clone is inappropriate.

Before installation, save a declarative JSON spec with exact source/weight
revisions and dependency hashes. Keep its provider ledger in this report and
the run directory. Execute the real CUDA witness and have a fresh agent follow
the documented invocation before declaring this composed runtime ready.
The checkpoint and binary wheels stay outside Git; only code, hashes and
bounded result files are published. No OpenShape training dataset is needed.

## Integration decision, still pending

If this source supplies meaningful object-level evidence, compare an upstream
object-memory integration against an equal-budget ScanRefer control from E71,
with preserved V99 output and Scan Mask floors. Distinguish frozen pretrained
input features from newly trained features, object position/class from
appearance, and object semantics from relations requiring global context.
Do not add another post-hoc source selector solely because a new feature
vector exists. Formal ScanRefer passage is still required before Nr/Sr
training; no failed prior endpoint is promoted.
