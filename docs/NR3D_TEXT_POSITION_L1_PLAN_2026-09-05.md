# L1: position evidence inside the existing Query-to-text attention

P2/R1's candidate readouts failed their fixed screens. They remain sealed.
This is a separate hypothesis: supplying tokens with point-position evidence
inside the last Decoder's existing text attention may change the learned Query,
its Box and its Mask together. It is not promotion of either failed readout.

The primary implementation reference is EG3DVG's PECA, inspected at commit
174e34894aea6513442da6b5dfa9b3e2bf8a1efa:
https://github.com/Gwan9Wook/EG3DVG/blob/174e34894aea6513442da6b5dfa9b3e2bf8a1efa/models/encoder_decoder_layers.py

Its PECA computes per-head text-to-point attention, pools projected point
position embeddings, and adds that result to the text key. This idea is
already published; L1 is an independent minimal mechanism test in our frozen
MCLN, not a novelty claim or reproduction of the full EG3DVG system.
The inspected release's GMA also references spatial_n_head without defining
it in that class; no repair was found in the inspected model/launcher code.
Its hard_sigmoid is not called by GMA.forward and PECA.kp_linear is unused.
These are static code observations, not an execution or paper-result audit.
The upstream LICENSE is CC BY-NC-SA4.0. No upstream source files are copied.

Keep the protected averaged checkpoint and all existing model parameters and
buffers fixed. Add only one zero-initialized, bias-free288x288 weight per arm,
82944 parameters. Run the original model in eval mode with autograd enabled.
Use the existing post-CrossEncoder text and1024 point features, and the exact
point-position embedding already supplied to that CrossEncoder.

- Text-key control: add W(text_token) to the projected text key.
- Position-key arm: alpha(token,point)=softmax over points of the per-head
  text/point dot product; add sum(alpha*W(point_position)) to the text key.

Implement each addition as an additive logit bias in the last Decoder's
existing cross_l MultiheadAttention. Its frozen query/key/value projections,
padding mask, values and output projection stay native. There is no new
attention layer, source selector, score fusion, auxiliary loss, GT input,
coordinate-frame estimate or candidate rule. A shared bias vector would
cancel in softmax, so the extra weight has no bias.

Zero initialization must exactly reproduce native full-model outputs. Before
any long run, test attention equivalence, padded-token exclusion, point-order
invariance, position sensitivity, and gradients through the frozen attention.
Then run a separate real-fit preflight on the same16 M3/M4 expressions, with
zero optimizer updates. Verify all frozen states and inputs, nonzero finite
gradients, actual native-loss connectivity, and finite last Query/Box/Mask
responses to one fixed small parameter intervention. Restore zero weights
after that diagnostic. A failed preflight is an implementation failure, not
a trained scientific result.

Only after the preflight passes, train both arms from identical zero starts
on all26747 fit expressions/413 scenes from the sealed P2/R1 split, batch4,
one epoch,6687 updates, same seed0 shuffled batches and no augmentation.
Use fresh AdamW LR1e-5, saved weight_decay=.0005 and clip_norm=.1, and the
unchanged native TrainTester loss/Hungarian matching. M5 states are not used.
Use the full6172-expression/98-scene module holdout once at the zero start and
once at the fixed terminal step, batch16, seed1000, native filtering/selection.
The frozen backbone has seen these holdout scenes; this is not new-scene
generalization of the whole system or a formal7899-row evaluation.

Pre-register the terminal position arm's screen against BOTH terminal text-key
control and protected zero start: at least10 net REC@.25 hits, no decrease in
REC@.50, either Mask hit threshold, or selected Mask mIoU. Report paired fixes,
breaks, per-scene metrics and2000 scene-bootstrap draws(seed0), including failed
screens. No intermediate evaluation, early checkpoint choice, rate/step sweep,
or control-arm promotion. A pass supports replication and a formal protocol
check; it does not automatically promote a benchmark result.

Save only isolated adapter/optimizer states. Require native model state and
all pinned source/data/parent weights unchanged, inputs identical between
arms, and candidate filtering with its actual updated Box outputs throughout.
Observe full256 legal coverage because this experiment can move Boxes;
do not mislabel its changes as fixed-candidate reranking.

The complete task still requires stronger Nr3D and Sr3D formal REC with the
protected ScanRefer result preserved. A Mask-only improvement in M5 cannot
satisfy those requirements.
