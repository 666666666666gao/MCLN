# M3: native Mask supervision and gradient result

Completed on 2026-09-05 at19:25 CST, controller exit0. The16 preselected fit
expressions come from16 distinct training scenes, in CSV order, using the
unchanged P2/R1 scene partition. They are not validation or whole-system unseen
scenes. Four B4 native forwards and eight gradient probes used eval mode with
autograd enabled, no augmentation, seed0, no optimizer or checkpoint writes.

The actual Hungarian matching and native loss graph are connected correctly.
For every expression, direct raw Query Mask gradient support equals its one
Hungarian-matched Query. There are104 legal Box IoU>.5 Queries in total;88 have
no direct raw Mask gradient. This is the current one-to-one training objective,
not an indexing bug or evidence that extra positive Mask supervision will help.
V44 candidate supervision already exists and must not be reinvented without
checking its previous results and training-target compatibility.

All16 matched Boxes have IoU>.5;10 matched raw Masks pass .5 and6 do not. The
matched Query equals native REC/Mask selection on9 rows and differs on7. The
16-row slice is too small and training-biased to estimate validation rates.

| Observed path | Mask gradient connection in all four batches |
|---|---|
| Final Decoder Query and x_query output | Connected; only the matched Query has direct nonzero support |
| Shared seed features and x_mask output | Connected and nonzero |
| x_query parameters | All6 connected and nonzero |
| x_mask parameters | All6 connected and nonzero |
| rel_encoder parameters | All4 connected and nonzero |
| Last Box prediction head parameters | All24 disconnected from this Mask objective |

The Box head still receives the native grounding objective; this result does
not say it lacks training gradients. Mask/grounding gradient cosine at the
matched final Query is negative on6/16 rows, ranging from-.166484 to+.141356.
This small eval-mode probe does not establish harmful training-wide gradient
conflict or justify loss-weight tuning/gradient surgery.

Among302 majority-positive superpoints,66 have no selected seed center inside
the target instance. Of those66,35 belong to one expression (fit row15,
scene0592_00). Seed-center membership is not receptive-field coverage, and the
current receipt stores counts rather than the actual neighbor IDs/distances.
Therefore it cannot distinguish empty-radius neighborhoods from neighborhoods
containing only other-instance centers. That is the next narrowly scoped check.

Source inspection after M3 identifies the actual grouper as ball query,
radius=.2 and nsample=2, followed by relative-coordinate encoding and max
pooling. The CUDA kernel takes the first two seeds in index order satisfying
distance<.2, not the nearest two. The C++ output indices start at zero; if no
seed satisfies the radius, both returned indices remain zero. These are code
facts; M3 alone has not measured how often they affect real foreground.

The native Query-feature/superpoint-feature einsum exactly reconstructs raw
Query Mask logits on all16 rows. The612-file frozen source,36 data files,
protected checkpoint, and every model parameter/buffer pass before/after
checks. No model .grad accumulation, tensor-leaf injection, loss replacement,
or requires_grad flag changes occurred. Runtime elapsed27.473553s excludes
dataset initialization; maximum allocated GPU memory7,398,736,384 bytes.

Evidence: `refine-logs/mask_supervision_probe_20260905_v1/` contains the input
manifest, CPU preflight, launch, native log, controller exit, full receipt and
derived summary. CPU tests2 PASS; the runner compiled under remote Python3.7.
Receipt SHA256:
`b6b7b1d7903f766c0f54f576c33641d89d60e455d0e5b5159db0680d1bba163f`.

No performance promotion. Protected Nr3D remains4475/3759 on7899 expressions;
ScanRefer/Sr3D protected artifacts remain in place. R1/P2 remain failed and
sealed. The next check records actual neighborhood distances and compares the
unchanged grouper with a direct two-nearest-seed alternative on these same
fixed training inputs. No new fusion gate or Mask residual head is implied.
