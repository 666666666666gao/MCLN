# Nr3D / Sr3D preparation for the ScanRefer appearance structure

ScanRefer training and qualification remain first. This work uses CPU input
metadata and protected checkpoint states; it does not start Nr/Sr optimization
or a new validation evaluation. No sealed O1 or candidate-range model is loaded.

## Actual object protocol

ScanRefer `butd` memory uses predicted GroupFree boxes. Nr3D/Sr3D `butd_cls`
replaces those with the protocol's instance boxes and predicted object classes.
The new appearance cache must follow those actual slots, not reuse ScanRefer
GroupFree regions. This is the existing instance-proposal protocol, not pure
predicted-box detection. Box crops retain background and neighboring points;
target IDs and instance segmentation masks must not clean their contents.

The current native `_get_scene_objects` keeps original instance indices. Four
Nr3D training scans have holes among valid slots:

| Scan | Gap in the valid-slot sequence |
|---|---|
| scene0074_00 | 18 |
| scene0216_00 | 0 |
| scene0134_00 | 12 |
| scene0228_00 | 1, 7 |

Thus a compact feature array must scatter by saved `slot_ids`. Prefix assignment
`[:count]` would attach some features to the wrong objects. The observed Sr3D
train audit has no intermediate gaps; its maximum valid slot is131, versus99
for Nr3D. The model's native padding and valid mask remain authoritative.

## Source availability at the existing384-point requirement

The completed historical crop audits record all valid train-scene object slots.
Using the same OpenShape minimum, without altering the encoder:

| Dataset | Scans | Valid slots | At least384 points | Below384 |
|---|---:|---:|---:|---:|
| Nr3D | 511 | 16181 | 11560 (71.4418%) | 4621 |
| Sr3D | 1018 | 34865 | 24219 (69.4651%) | 10646 |

These are unique per-scene object slots, not expression-weighted target recall,
Anchor coverage or repairable REC errors. Sr's two previously observed single
point objects are naturally unavailable under the same384-point rule. Missing
appearance retains native box/class memory; no duplicated points or invented
geometry are introduced to make a crop pass.

The CPU exporter reuses the current native `_get_scene_objects` method on the
unchanged50,000-point serialized scenes with augmentation disabled. It compares
full padded box hashes, native point hashes, original slot IDs and every explicit
AABB crop count against prior complete records. It writes compact boxes,
predicted classes, slot IDs and hashes, not point-cloud copies or embeddings.
Overlapping468 scans must agree between the Nr and Sr evidence; their union is
1061 scans. No text parsing or model forward is needed for this metadata export.

The actual CPU export completed normally and independently recounted all1061
scans. Every point/box/slot/count comparison passed. Processing plus the final
input rehash took121.515seconds after scene loading. Metadata SHA256 is
`31d5bb22f90556ae2f6087017c603203647e849568fd66e24212b1000e1fd11f`.
The5.53MB region metadata remains remote (plus a verified local temporary copy);
Git contains source, receipts and summary counts, not the complete region boxes.

## Pretrained initialization evidence

The actual protected E71 and averaged Nr3D E57 files were SHA-verified and loaded
on CPU. Both have1144 state tensors with identical keys and shapes;206 tensors
are equal in value. E57 is evaluation-only and contains no optimizer state, so
using it for a new run requires a new optimizer, not a claimed historical resume.
This removes a known shape-level obstacle to pretrained initialization but does
not replace strict model loading and native forward/gradient checks.

Keep dataset input flags distinct: E71 uses `butd`; Nr E57 uses `butd_cls`; both
have6 Decoder layers,256 queries, RGB input,485 object-class embeddings and the
same two-source selector configuration. Do not relabel Nr weights as the missing
historical Sr checkpoint. Sr initialization and its provenance will be specified
as part of the next qualified training run.

## Remaining augmentation and training work

The native train path actually rotates/flips/noises/scales points, perturbs RGB
and independently jitters object boxes. The current Scan appearance comparison
explicitly has augmentation disabled. Therefore its exact unaugmented point/box
cache assertion cannot simply be copied into an augmented Nr/Sr loader.

Before those runs, specify whether cached canonical appearance is an explicitly
separate observation from augmented geometry, or whether appearance is encoded
from transformed crops. Validate that choice on actual training inputs, retaining
the corrected view-word augmentation handling. Do not silently call the current
cache rotation-invariant, disable augmentation in an otherwise unchanged recipe,
or adjust jitter based on formal validation results.

After Scan qualification: encode/bind source features with original slot IDs,
verify actual model loading and the chosen augmentation contract, then launch
Nr/Sr REC training using available pretrained weights. The same architecture
and output policy must be reported across datasets; Nr/Sr Mask remains waived.

## Joint detection coverage completion (2026-09-08)

The 1061-scene export above covers the two language training sets, not every
input in the native `joint_det=True` recipe. Calling the unchanged native
`load_scannet_annos` on actual serialized training scenes returns1199 detection
rows/scenes. Their intersection with language scenes is1060, leaving139
additional detection scenes. Language-only `scene0154_00` is retained; the full
union is1200 scenes. No dataset row, native detection exclusion or repetition
factor was changed to fit the cache.

`extend_referit3d_joint_detection_slots.py` has completed the additional139
scenes on CPU. It uses the same native `_get_scene_objects`, preserves original
instance slots and predicted classes, and counts every sampled point inside
each protocol AABB without target or instance-mask cleaning. The original1061
records are identical except for an added `scannet` membership tag where
applicable. It creates no embeddings, model predictions or training updates.

The detection subset contains38304 valid slots,27017 meeting the same384-point
encoding minimum;10 scenes have noncontiguous slots. These are object-slot
counts, not language target coverage or REC gains. Nr/Sr language counts in the
earlier table are unchanged. All1200 scenes have predicted-class records.

Complete metadata is in
`/root/autodl-tmp/mcln_referit3d_joint_detection_slots_20260908_v1/scene_slots.json`,
SHA256 `11cc74f6a2f3aef4b3b94e4d892541e256a1943466ec56014f03f24e92ddcf73`.
The original1061-scene artifact remains available. Source, native coverage
receipt and independent local recount are archived in
`refine-logs/referit3d_joint_detection_slots_20260908_v1`; full region metadata
stays outside Git. This completes scene/slot metadata coverage for the inspected
joint recipe, not augmented feature binding or model/gradient validation.
