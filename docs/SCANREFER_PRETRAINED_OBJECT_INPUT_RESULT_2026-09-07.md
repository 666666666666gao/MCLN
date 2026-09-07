# ScanRefer pretrained object source: actual input result

The fixed input and frozen feature preflight are complete. This establishes
source execution and slot-level feature separation, not REC improvement or
MCLN training. The full three-dataset objective is still incomplete.

| Observed quantity | Result |
|---|---:|
| Fixed historical fit expressions | 16 |
| Distinct scenes | 10 |
| Detector object-slot occurrences | 591 |
| Unique scene/object slots | 314 |
| Repeated slot occurrences | 277 |
| Empty crops | 0 |
| Occurrences with at least 384 points | 409 |
| Occurrences below 384 points | 182 |
| Unique objects below 384 points | 86 |
| Minimum / maximum observed crop points | 22 / 30980 |

All sixteen native XYZ/RGB byte hashes equal the saved actual E71 ScanRefer
fit forwards. The crop inputs are the existing GroupFree predicted boxes,
with the dataset's min/max-to-center/size conversion and float32 slot type.
No instance segmentation label or target box is used to clean the crops.
Original scene objects are loaded only to obtain their scene XYZ and color.

The pinned official OpenShape G14 factory requests 384 FPS centers. In the
actual DGL 0.9.1.post1 wheel, `dgl/geometry/capi.py` asserts that the input point
count is at least the batch size times requested centers. The recorded missing
coverage is a real input constraint for the unchanged factory, not a guessed
failure. The probe excludes those crops and records them as unavailable;
that rule is not a final deployed integration design. Large crops are capped
at 10000 without replacement; no points are fabricated.

Receipt SHA256:
`5770b71d407e2bbfbffe0b99fbf0de0e2ddf3ad03259d456d34670691fff089f`.
Inputs and their per-file hashes remain at
`/root/autodl-tmp/mcln_openshape_object_source_20260907_v1/inputs`.
The export ran in the unchanged `bdetr` environment on CPU, with zero MCLN
forwards, optimizer updates, generated checkpoints or formal evaluation rows.

## Why this changes the next action

The previous 512-row audit excludes a deployed-score map discrepancy as a
primary failure in that sample. Source inspection confirms existing
SourceMoE listwise quality, SACR/JQQ quality losses, and Tier hard-query /
Relation-CF supervision of deployed scores. These are distinct historical
implementations and experiments, not a claim that all were enabled in E71.
Another generic root-IoU scoring loss is not a new information source.

The actual native object memory concatenates box and class embeddings in
`models/mcln.py`. With `butd` enabled, its use in the three-layer encoder is
enabled as well, and it also feeds all six Decoder object-attention paths.
This gives an existing upstream access path for stronger pretrained visual
evidence. It differs from the failed O1 lightweight point MLP injected at the
last object attention, but no gain is implied by that difference alone.

The completed fixed OpenShape test measured real feature finiteness, source
coverage, computation, and repeated-object sampling stability. Author-provided
LVIS text-feature matches are descriptions for inspection, not REC or category
accuracy. In particular, 16 expressions in 10 backbone-seen training scenes
cannot prove generalization.

## Actual frozen features and local recount

All 409 eligible object occurrences were encoded, representing 228 distinct
scene/slot objects. The 181 repeated occurrences have cosine minimum0.980786,
median0.995989 and mean0.995174. The 2915 distinct same-scene slot pairs have
median0.658266; the 304 pairs sharing a detector class have median0.768863.
All 181 repeated occurrences retrieve their own slot from the first occurrence
of each same-scene slot. This is a sampling/slot-separation check, not target
identity accuracy: detector slots can duplicate physical objects and no
language query was used.

Frozen real-object encoding took8.817929 seconds with peak allocated CUDA
memory336674304 bytes (321.08MiB), excluding model initialization and the
separate witness. The 1280-dimensional features are finite and nonzero.
The feature NPZ SHA is
`3e5c218a1f67207324d918447323f6a29b34854a97ea4998d19f324e90100fa3`.
A local CPU recount from the hash-verified tensor agrees with the remote
analysis within1e-6; the original remote result is retained separately.

Semantic outputs are mixed. All four unique detector-bed crops have bed in
LVIS top5, while several cabinet crops match curtain, radiator or bulletin
board, and some chairs match clothing. These are detector-label comparisons,
not GT classification accuracy. The evidence supports a trainable native
integration control, not immediate substitution of detector categories or
an assertion of a reliable language-grounding source.

## Runtime outcome and next action

The pinned official checkpoint (388091433 bytes) passed strict loading in the
unchanged conda bdetr/Torch1.10.2+cu111 base. The first build failed because
networkx was absent; its original exit1 and log are retained. The v2 declared
spec adds networkx2.6.3 to the isolated prefix alongside DGL CUDA1110.9.1.post1
and torch-redstone0.0.6. Active canonical spec SHA:
`fe8ac66b8f51ed0e179a279717d0d1c2213e9a6b463785ccaa6a6e5eaf2385ac`.
The base package inventory is unchanged. Both build_v2 and object probe exit0.

A fresh Codex same-family reviewer executed the documented loaded CUDA witness
verbatim and returned provisional PASS. The actual output was shape[1,1280],
norm1134.481567, A10040GB, and32326080 model parameters. This review covers the
runtime witness only, not scientific quality. No model/dataset is downloaded
into Git, and the205GB upstream training dataset was not downloaded.

The next work is scene-level frozen preprocessing and a default-off native
object-memory appearance path, documented in
`SCANREFER_OBJECT_APPEARANCE_INTEGRATION_2026-09-08.md`. The actual G14 README
explicitly requires gravityZ; its demonstration coordinate swap is not an
instruction to swap this already-Z-up dataset blindly. Current partial crops
and384-point minimum remain material limitations.

Protected ScanRefer/Nr3D/Sr3D results remain unchanged. ScanRefer still has to
pass the same V99 REC and Scan Mask floors before Nr/Sr training; their Mask
metrics remain waived for this phase. No previous negative method is reopened.
