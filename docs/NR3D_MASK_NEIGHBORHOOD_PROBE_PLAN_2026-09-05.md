# M4: actual superpoint neighborhood and fixed-input substitution probe

Use the exact16 fit rows, B4 order, seed0, augmentation disabled, protected
checkpoint and frozen source/data manifest from M3. Input point hashes must
equal M3. No validation rows, optimizer, parameter/buffer changes or new labels.
Do not choose rows from M3 errors or change the existing scene partition.

The active CUDA ball query scans seed indices, returns the first two inside
radius.2, repeats the first when only one exists, and leaves both indices zero
when none exist. First verify these behaviors with a small synthetic CUDA
example using the installed extension. Record its binary path and hash.

Run two forwards per B4 batch: original ball query and direct nearest-two
grouping. The alternative replaces only seed selection, keeps nsample=2,
relative-coordinate encoding, max pooling, both Mask projections, fusion,
Query selection and weights unchanged. It has no radius fallback or additional
trainable parameters. Restore the original grouper after each pair. REC boxes,
raw scores, Query identities and shared seed/query features must remain equal.

Record all actual superpoint IDs, point/target counts, centroids, seed centers,
target seed memberships, original/nearest indices and distances. Distinguish
present superpoints from absent integer slots. Report on all present, target
bearing and majority-positive superpoints separately:

- Empty-radius neighborhoods and their actual seed0 distances;
- Original index-ordered neighbors versus nearest neighbors;
- Selected neighborhoods without target seed centers (not receptive fields);
- Coordinate equality between seed_xyz and input points indexed by seed_inds.

Use native thresholding/evaluator and original Hungarian matching from each
arm, while also keeping the original matched Query fixed for the paired Mask
comparison. Report native text/raw Query/fused Mask, same matched Query Mask,
GT Full256 raw-Mask oracle, fixes and breaks. Native grounding outputs must be
unchanged; changed training matching, if any, is an observed consequence of
its Mask cost rather than a REC change. No threshold/radius/K sweep.

This is an inference intervention on a pretrained checkpoint, not a training
or generalization test. A worse substituted Mask can reflect changed feature
distribution. A structural training experiment needs documented nonlocal
foreground neighborhoods and a same-start, same-data, same-update original
grouper control. Neither a positive16-row result nor a code defect is grounds
for replacing a protected model. If no foreground locality problem is observed,
do not advance this locality hypothesis from these inputs.
