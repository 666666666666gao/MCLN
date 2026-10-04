# Frozen original G: residual / six-face distribution control

Status: implementation in progress; no model forward, deployment or accuracy result.

The completed frozen-G whole_range arm has formal bbs5589/4456, versus
local5588/4447 and original G5615/4495. In the whole arm, the same selected
Query is4495→4456,84 repairs/123 damages, median maximum face displacement
16.843mm. Strict Full256 coverage7884→7974 while actual hits fall39. This is
not proof that more training, larger residuals, or quality attention will help.
Original G parameters/buffers stayed fixed. Historical joint results and
frozen results are different adaptation protocols, not an isolated freeze effect.

Now isolate a boundary representation/supervision package under the SAME whole
109-dimensional range summary and112-member fused local support. Keep the
1302→288 feature extractor. Original G remains frozen/eval, native unique bbs,
all256 candidates, original Mask, all native matching duties, seed2027.
No direction-token decoder yet: the plain distribution adapter is a direct
control with prior work, not a novel three-module method.

Two arms each start from protected official PV plus original G, fresh AdamW:

| Arm | Prediction | Loss | Parameters |
|---|---|---|---:|
| residual | original six center/size residuals | native deployed-box/G loss |400614|
| distribution | six outward offsets,33 nonuniform bins per face | same native deployed-box/G loss plus DFL/7 |456102|

The comparison changes representation and explicit supervision together. It
does not isolate either alone; no causal claim about the DFL coefficient or
parameter count. DFL targets use exact native final-layer Hungarian matches,
all matched GTs, no extra root-only matching or union. Mean over matched
boxes×6 faces; coefficient1/7, separate from G or contrast denominator.
Target offsets are normalized by clamped coarse dimensions, regscale4;
knots are symmetric±4, with denser near-zero spacing. Values outside the
representable interval saturate at endpoints; count them explicitly.

Both arms use the SAME1e-6 reference/final size floor. The existing source
already floors size for sampling/evaluation, while new signed face offsets can
produce crossed faces (constructive CPU equation check, not observed model
incidence). Apply the floor in the training forward too; no postprocessing
answer or fallback. At zero heads, compare the same cached inputs to original
G with this common floor. Record any raw native-size differences; do not claim
bitwise equality with unmodified negative sizes. Uniform logits subtract the
floating uniform reference before integration, preserving zero offsets.

First a serial engineering probe, one actual batch8/two updates EACH arm:
verify original G state unchanged, final box losses reach output weights,
DFL alone reaches distribution output, second-step member/aggregate gradients,
same-input zero offsets, all final dimensions positive, actual native last
matching, checkpoint/AdamW exact in-memory restore. No disk checkpoint.
Record finite targets, target saturation, size-floor count, native/DFL gradient
norms independently, and actual peak allocator memory. CPU equations are not
native-model or GPU evidence.

After fresh code review and successful real probes: serial residual/train,
residual/formal, distribution/train, distribution/formal on the single A100.
Per arm effective/physicalbatch8, LR1e-5,WD5e-4,clip0.1,29778 fit rows once,
3723 updates (tailbatch2),6887 pretrained-seen initial/terminal module holdout,
9508 development validation. Same loader-row order; model initialization has
different head shape, so preserve actual starting-query/frame comparison and
nonbitwise numerical limits. Do not change batch or sample/step budget.
Estimate about4.5–5GPU hours total from the completed head-only pair, including
formal evaluations; no blind repeating sweep. Poll near estimated phase end,
then240s minimum. Fit stays blocked on this experiment's real preflight.

Primary selection is same-checkpoint native last/bbs Acc@.50, record .25,
Mask, same-query repairs/damages, full256 GT-offline coverage and rank buckets.
Compare with original G4495 strict hits, not only a regressed control; target
same checkpoint5615/4754. Do not interpret oracle as deployable selection.
No new semantic output, quality readback, teacher, multi-positive loss or
backbone change in this comparison. Direction-preserving six-face support and
final geometry readback are later changes only if evidence warrants them.

Retain protected G/official PV/V99 chain. One necessary active recovery per
arm until actual restore/formal verification; retire our verified nonbest
endpoint using existing user authorization, no failed full-weight archives.
Preserve source, logs and rows. No promotion or Nr/Sr generalization claim
without actual results. After ScanRefer selection, fixed structure will require
Nr3D/Sr3D separate training with their native label/input protocols.
