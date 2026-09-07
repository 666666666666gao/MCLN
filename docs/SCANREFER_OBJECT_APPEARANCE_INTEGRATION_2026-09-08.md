# ScanRefer: pretrained appearance in the existing object memory

The native input-path engineering probe passed. The fixed paired training
job has launched; it has no completed ScanRefer endpoint or REC gain. No previous failed endpoint is used.

## Mechanism and its limits

The current 288-dimensional object memory concatenates 128 box-position
channels and 160 predicted-class channels. The proposed change adds a learned
projection of the frozen 1280-dimensional OpenShape feature to the semantic
160 channels. The position channels remain intact. The existing three-layer
bidirectional encoder and six Decoder layers then read this enriched memory
through their original object-attention paths.

The implementation is `models/pretrained_object_appearance.py`, activated by
the optional `use_pretrained_object_appearance` MCLN constructor argument.
Default is off. No additional score source, routing gate, matching loss or
box/Mask head is introduced. The 1280-to-160 projection has 204800 trainable
parameters, zero initialization and no bias. Layer normalization is applied
to the frozen source feature before projection. A required availability mask
zeros the contribution of the observed undersized crops and padded slots;
the original box/class memory is retained for those objects.

This uses published pretraining. A simple projection is not claimed as a
novel architecture or as a solution to compositional language. The practical
question is whether richer visual evidence helps the existing upstream
language/object interactions. O1 used a small point MLP at the last object
attention; this path changes both the source and where it is consumed, so a
negative or positive result will not isolate those two differences by itself.

## Frozen scene features and training input

The first 16 fixed fit expressions contain 591 object-slot occurrences in
10 scenes, but only 314 unique scene/slot objects. Reusing one frozen feature
per scene/slot avoids repeated encoding for every expression.

`cache_scanrefer_openshape_objects.py` reads the existing 562-scene ScanRefer
training list and the same 50000-point native XYZ/RGB arrays. It crops only
the original GroupFree predicted boxes and records both the detector SHA and
native input SHA. It does not inspect target IDs, text or instance masks.
Each crop uses a scene/slot hash seed; XYZ is centered and normalized to a
unit sphere, Z remains gravity, and RGB remains in [0,1]. Above 10000 points
the sample is without replacement. Below 384 points the cache stores zeros
and `available=false`, reflecting the unchanged pretrained FPS requirement.
The pretrained encoder is frozen. No class name is an encoder input.

Features remain outside Git and are preprocessing artifacts, not cached
predictions or GT-assisted scores. This first cache uses unaugmented inputs;
the first native integration probe disables augmentation and compares every
input box and native point hash with its cache. A future augmented training
run must explicitly define how semantic appearance and transformed box
positions stay aligned, rather than silently using a cache for changed slots.
Nr3D/Sr3D use a different existing object-input protocol and will need their
own matching feature extraction after ScanRefer qualifies.

## Executed and pending checks

The CPU module contract passed: exact zero-initialization identity, unchanged
position channels, unchanged unavailable slots, finite nonzero projection
gradient, and equivariance to a joint permutation of object slots. One
synthetic parameter update was performed and discarded; this is not MCLN
training.

The native probe is fixed to the same 16 ScanRefer fit rows, batch size4.
It loads all 1144 protected E71 state tensors strictly, attaches only the new
projection, and checks exact initial REC/Mask tensors and V99 runtime parity.
It then measures native GT gradients and executes two disposable projection
updates on the first batch. Every original parameter and buffer must remain
unchanged. No checkpoint is written; no formal evaluator is run.

The runtime source is an isolated copy of the previous verified 622-file
native source plus the narrow MCLN patch, the new module and needed helpers,
625 files in total. Its manifest SHA is
`190d0011bc5bfefab4a3965d972606f4dc21a946f68b2382f5a90837b3a4c4ef`.
This does not silently overwrite unrelated canonical-source differences.

Native probe root:
`/root/autodl-tmp/mcln_scanrefer_object_appearance_native_20260908_v1`.
Cache root: `/root/autodl-tmp/mcln_scanrefer_openshape_cache_20260908_v1`.
Both cache and native probe exited0. The cache contains562 scenes,16759
slots and12392 encoded slots (4367 unavailable), encoding elapsed310.18s.
The16-row probe verified identical initial native REC/Mask and V99 runtime
in all four batches. Projection gradient norms ranged11.96 to33.43; two
disposable first-batch updates produced losses10.52475 and9.90343. All1144
original state tensors remained exact. No checkpoint or formal result was
created by the probe. Native receipt SHA:
`f2460cd13446984b509b80425825636a14b3045f6cbfe1923575c106a81c5967`.
The updated probe weights were discarded before the paired run.

## Next performance experiment

After real native input/parity/gradient verification, use a paired ScanRefer
training control from the same E71 and the same correct mesh inputs. Compare
native memory against the pretrained appearance path with the same native GT
loss, sample order, core trainable scope and update count; distinguish the
additional projection parameters and preprocessing cost. Freeze the existing
Parent/Geometry/V99 readouts for the main comparison and report native and
full-system results separately, including REC repairs and breaks at both
thresholds. Do not introduce a new auxiliary ranking loss in that comparison.

The next performance plan is now fixed in
`SCANREFER_OBJECT_APPEARANCE_PAIR_PLAN_2026-09-08.md`: same E71, same29778 fit
rows, one2482-step traversal per arm, all existing cross_encoder/Decoder/
prediction-head parameters at1e-6, new appearance projection at1e-4, native GT
loss and frozen V99 readouts. No failed local module is loaded. Full chosen
scope batch12 capacity is checked before baseline evaluation and fitting.
The paired job launched in screen `70235.mcln_os_appearance_pair_v1` at
2026-09-08 00:40 CST; launch is not an optimizer-step or quality result.
