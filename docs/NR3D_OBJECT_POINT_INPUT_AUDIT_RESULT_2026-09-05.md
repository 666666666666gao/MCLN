# Existing object-input appearance audit: completed

The fixed CPU audit completed with exit0 on16 fit expressions from16 scenes.
It made no model/GPU forwards or optimizer updates. Input point hashes and
backbone seed indices/coordinates match the earlier M3/M4 probes. The frozen
source,36 data files and both input receipts passed their before/after checks.

| Existing input measurement | Result |
|---|---:|
| Valid object slots |683|
| Empty raw point crops inside the supplied box |0|
| Empty1024-seed regions inside the supplied box |8|
| Empty seed regions with at least32 raw points |4|
| Nonpositive box axes |0|
| Median true-instance fraction in a box crop |0.7429577465|
| Crops with less than half their points from the instance |123|

The first four expressions reproduce R1's190 object slots and155 correct
predicted classes. In the actual butd_cls path, the supplied boxes equal the
existing instance boxes and the classes are detector predictions. This audit
does not turn that protocol into a detector-box-only evaluation.

Raw colored points are available for object appearance encoding. However,
spatial box cropping also includes background and neighboring objects. Instance
point memberships were used only to measure this contamination; they must not
be used to clean crops supplied to a new model. Eight empty seed regions also
show that pooling only the final backbone seeds can miss available raw inputs.
These observations establish input availability, not appearance-feature quality,
text-anchor coverage, or improved REC.

The official ReferIt3D and MVT implementations encode individual colored object
point clouds before fusion. Their segmented-object input convention is a
material difference from taking all points inside our existing boxes. No
upstream object encoder, segmentation input or new score was added here.

Primary source references:

- https://github.com/referit3d/referit3d/blob/eccv/referit3d/models/referit3d_net.py
- https://github.com/sega-hsj/MVT-3DVG/blob/main/referit3d/models/referit3d_net.py

Evidence: refine-logs/object_point_input_audit_20260905_v1/receipt.json
(648400 bytes), SHA256
723a49de580d7770de2bd7523473a84bf428c71e84e218d12239e96c28fba11b.
Manifest SHA256
f17f1847b0f900233ea93618beaf00affa3f4c7c82a2f0f8b787a8064b4b7524.
The25.1856 seconds exclude dataset initialization. R1/P2 remain failed and
sealed. M5 continues independently with its original endpoint contract.
