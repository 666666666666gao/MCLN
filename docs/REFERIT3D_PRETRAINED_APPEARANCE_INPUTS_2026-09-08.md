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

## Native augmentation and slot-binding check (2026-09-08)

An isolated CPU check called native `_get_pc` and `_get_scene_objects` on the
previously fixed32 Nr/Sr preflight annotations plus10 detection scenes with
noncontiguous slots. Four fixed seeds per case produced168 geometry checks.
All retained the canonical slot indices while changing point coordinates and
boxes; original XYZ/RGB remained unchanged. The24 view-restricted checks kept
Z rotation within5 degrees and did not enable the large-rotation flips.
Other native small rotations, noise, scaling and box jitter were retained.

`src/object_appearance_inputs.py` now supplies a compact-to-native-slot scatter
function. It requires the compact slot IDs to equal the actual valid-mask
indices, places available features at those indices, and leaves unavailable
and padded slots zero. It does not compare augmented box coordinates with
unaugmented cache boxes. The168 checks used synthetic slot-number markers to
verify this indexing function, including all10 gap scenes.

This is a targeted native geometry/identity check, not the complete dataset
`__getitem__`, model forward or pretrained-feature invariance test. The ongoing
Scan training source is unchanged and does not import this new helper.
Receipts and source are archived in
`refine-logs/referit3d_appearance_augmentation_20260908_v1`.

For subsequent qualified Nr/Sr integration, use canonical frozen appearance as
an explicitly separate visual observation associated with the same protocol
instance slot, while the native point/box branch keeps its existing training
augmentation. Canonical point/box hashes must be verified against the original
scene before augmentation; transformed sample boxes are validated by slot
identity, not claimed equal to cache coordinates. Validation uses the same
canonical observation without training augmentation. This choice reuses the
available pretrained encoder and preserves the original augmentation recipe;
it is not a claim that the encoder itself is rotation-invariant.

Actual frozen-feature encoding, full data-loader attachment and model/gradient
checks remain required before Nr/Sr training. They must establish native
zero-initialization parity and available-feature gradients; only REC evaluation
can determine whether this two-observation design is useful. No inference
target label or instance-mask cleanup may be introduced in those steps.

## Conditional frozen encoding queue (2026-09-08)

The train-cache entrypoint is now prepared and its conditional queue is live at
`/root/autodl-tmp/mcln_referit3d_openshape_cache_preparation_20260908_v1`.
PID72673 was confirmed at02:22:51CST. It waits240seconds between checks of the
existing Scan formal `queue.exit`. Prior failure, module rejection or formal
nonqualification does not create the cache or launch an encoder subprocess.
Those three paths passed CPU tests locally and in the actual bdetr environment.

After Scan qualification, it verifies agreement between the formal decision,
receipt and independent audit and records their hashes. The actual encoder
checks that qualification again, acquires the same GPU execution lock through
its launcher and uses the unchanged official G14 runtime. It encodes the1200
canonical training scenes, retaining original slot IDs, predicted classes,
canonical boxes, per-crop counts and point/file hashes in the result.

The crop normalization/sampling block has the same parsed Python syntax as the
already executed Scan train-cache implementation. Both require at least384
actual points, subsample above10000 using the scene/original-slot seed, retain
RGB and Z-up orientation, and normalize around the crop centroid/unit radius.
The meaningful input difference remains the protocol instance boxes in Nr/Sr.
No target instance mask cleans either crop. This AST comparison proves recipe
equivalence for that block, not encoder quality or complete execution.

At preparation time both entrypoints passed actual bdetr `--help`; the new full
encoder path has not run. The cache directory does not yet exist, no Nr/Sr
embeddings or optimizer steps have occurred, and successful encoding will still
leave full-loader/model loading/gradient checks and training to complete.
The queue does not automatically treat cache creation as dataset qualification.
All scripts, fixed plan and preparation receipts are archived in
`refine-logs/referit3d_openshape_cache_preparation_20260908_v1`.

## Full native sample attachment check (2026-09-08)

The CPU check completed with exit0 and was collected at02:47:37CST. It uses
the same fixed twelve language and four detection rows for each benchmark,
with seeds0/1 and native augmentation both enabled and disabled. Across128
comparisons (256 actual `Joint3DDataset.__getitem__` calls), all37 original
returned fields and Python/NumPy/Torch RNG states matched exactly. The counts
are48 Nr3D,48 Sr3D and32 joint detection comparisons. Eight actual default
collations produced appearance tensors of shape16x132x1280 and boolean masks.

`attach_object_appearance` adds only `det_visual_features` and
`det_visual_available` to a shallow copy of the native sample. It uses the
existing original-slot scatter and does not alter augmented boxes, text,
superpoints, masks or predicted class labels. Canonical scene-point hashes and
the selected correct-mesh superpoint hashes were checked against the fixed
input records. The source snapshot and annotation hashes were also verified.

This closes the complete sample construction/collation portion of the earlier
input audit. It used synthetic slot identity markers, not actual G14 features;
the2784 available slot occurrences are indexing coverage, not semantic recall.
Disk cache loading, standard training-input plumbing, real model zero-init
parity and gradient checks remain before qualified Nr/Sr training. No model
forward, optimizer step or formal evaluation occurred. The running Scan source
and all its fixed queues remain unchanged. Evidence is archived in
`refine-logs/referit3d_appearance_getitem_20260908_v1`.

## Disk cache wrapper and training-input mapping (2026-09-08)

`ReferItObjectAppearanceDataset` now wraps the native dataset. At construction
it verifies the expected cache receipt, scene index and individual NPZ hashes,
original scene points, canonical boxes, predicted classes and original slot
IDs. A shallow dataset/scan copy with augmentation disabled is used only for
canonical verification; the actual native dataset retains its augmentation.
Required scene features are loaded once. Each sample then uses the original
native `__getitem__` and adds appearance through the verified slot binding.

The standard `TrainTester._get_inputs` now includes the two appearance keys in
its existing optional field list. This is a two-line mapping addition; the
running Scan snapshot and its custom input path are unchanged.

A new CPU run completed and was collected at03:04:18CST. It created on-disk
fixtures containing synthetic slot markers with real canonical scene boxes and
predicted classes. The new wrapper loaded and validated those files, then ran
128 plain/wrapped comparisons (256 native calls), with augmentation on/off and
seeds0/1. All37 original fields and RNG states matched. Eight collated batches
passed through the exact `_get_inputs` method body extracted from the modified
training source; the new fields arrived unchanged and the old input fields
matched. This checks the actual pure mapping function, not a full training
process or MCLN forward. The fixtures are explicitly marked as synthetic and
are outside the qualified G14 cache directory.

This completes preparation of disk loading and input plumbing. Real G14 cache,
model zero-initialization parity, actual gradients and subsequent REC training
remain conditional on Scan qualification. Source, manifests and results are
in `refine-logs/referit3d_appearance_cache_loader_20260908_v1`. No main training
settings, formal thresholds or protected weights changed.
