# Final boundary evidence readback: unintegrated draft

This is source preparation while the separately reviewed face-conditioned fit
runs. It does not modify or deploy anything into that active source. No new
native factory, GPU forward, optimizer update, parameter measurement or accuracy
result exists. Current metric best5616/4506 and its parent chain remain protected.

The contract uses fields actually emitted by both completed flat distribution
and active face-conditioned distribution heads: six33-bin final logits, whole
Mask109 statistics, coarse/final boxes and actual size-floor flags. Which
geometry provider is used must be selected from real terminal evidence and
disclosed; this draft does not silently load either trained endpoint.

Each face has44 inputs: probability33, expected offset1, entropy1, scene-normalized
coarse/final face2, corresponding-axis moments4, support mass1, face sign1 and
actual floor flag1. Entropy and probability concentration are model evidence,
not calibrated IoU or an instance identity label. All256 remain accessible.

The44-D evidence is encoded to64-D. Six distinct face roles read complete text
tokens using the native padding mask, then a candidate semantic Query reads its
own six roles. Zero-initialized64-to288 output returns a residual to the same
native semantic Query. There is no separate quality rank, scalar broadcast over
token logits, hard Mask crop, candidate truncation, teacher or inference GT.
No candidate-set self-attention or new supervision is included in this draft.

Native integration is pending. The actual pinned dependency inspection shows
final semantic scores currently precede Mask/tail; the original semantic
subhead contains BN/Dropout. Future wiring must delay only the final semantic
subhead, call it once after readback, preserve coarse geometry/contrast/Mask
responsibilities and prove zero-residual equality with real inputs. Recalling
the entire box head would replace geometry again and is not the intended path.

Expected source count96672 parameters/23 trainable tensors is a calculation,
not a measured native construction result. First output update before internal
gradient checks, actual final-score response, frozen provider/state and exact
optimizer restore all require a future source gate plus real two-step probe.

An eventual fixed-geometry study must use the same selected geometry provider,
initialization and budget for an explicit geometry-evidence visibility control.
Extra adaptation history is counted. Structural evidence visibility and any
within-instance final-IoU supervision are separate experiments; neither has
been shown to work here. Original G's unnormalized ScanRefer target and other
dataset coefficients cannot be replaced with generic binary targets by name.

This is an implementation candidate for the requested geometry-to-decision
stage, not an established third contribution. Current face terminal and the
ScanRefer target, followed by independent protocol-correct Nr3D/Sr3D, remain
required. Source AST does not prove runtime correctness or precision gains.
