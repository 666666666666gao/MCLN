# Final geometry to native semantic score: actual interface finding

This is a read-only dependency finding, not an implemented or trained S3. The
current face fit is unchanged. Its eventual metrics must decide the geometry
provider before any next source/factory is deployed.

Pinned source follows the actual preflight model/import receipts. The tail port
changes only models/pv_ground.py; unchanged models/modules.py is SHA-bound by
the same port. READBACK_INTERFACE_WITNESS lists exact local bytes and paths.

## Actual native order

1. Final decoder returns distinct semantic query and geometry query.
2. Contrast projection uses the semantic query before the prediction head.
3. ClsAgnosticPredictHead uses semantic features for sem_cls_scores_head;
   center/size use geometry_features. The semantic branch is ThreeLayerMLP,
   containing two BatchNorm1d and two Dropout(0.3) operations.
4. Query Mask uses the final geometry query; Text Mask uses original text
   features and its separate text prediction head. Native adaptive fusion and
   both final Mask lists are produced before candidate_box_refiner.
5. CandidateBoxRefiner replaces only last_center/last_pred_size. The current
   face module also stores boundary_logits and Bx256x6x64 face states. Those
   states are not read by the semantic branch or native evaluator.
6. The evaluator computes token-softmax and expression-weighted token evidence.
   One scalar broadcast over all token logits cancels in that softmax. No
   additional quality score is currently combined with bbs.

## Minimal future insertion, conditional on terminal evidence

A future readback needs the original semantic query, text features/padding and
the actual final geometry evidence. None is currently a new GT inference input.
Final semantic scoring is not required to form the two Mask paths or the
current support/face refinement; the final score can therefore be delayed in
an isolated source port. Preserve all other prediction responsibilities and
the existing parameter names. Skip the final semantic subhead inside the last
box head and invoke that same semantic subhead once after readback. Do not call
the entire box prediction head again, or replay a scoring head with Dropout/BN
twice. Actual zero-readback/native call count/state/score equality still needs
a new real-model preflight; this source graph alone does not prove it.

For a fixed-geometry source experiment, preserve the selected geometry provider
and compare visibility of its final evidence under a common same-start/same
budget semantic adaptation. This is a control choice, not a claim that semantic
gradients have been proved to damage geometry. Count the additional adaptation
history explicitly; a new stage is not a free continuation from official PV.

## Supervision and scope limits

Current G preserves every Hungarian-matched query, including other instances,
and replaces only unmatched final-IoU>.5 root CE targets. Its native ScanRefer
root target is .6positive+.2modify+.2pron+.1rel with unnormalized target mass;
its .5/7 coefficient is not a generic normalized binary probability target.
Do not silently substitute BCE or renormalize this target when adding geometry
quality. Any proposed within-root quality ranking would have to use actual
final boxes, stop-gradient training qualifications and the actual bbs score,
protect matched others and uncertain candidates, and be tested separately from
structural readback. It has not been implemented or approved as a result here.

All256 candidates remain. Existing score maps and two-stage object inputs are
native protocol dependencies and must remain disclosed. No teacher, GT gate,
second deployed ranking, candidate truncation or new loss was added by this
inspection. Same checkpoint ScanRefer5615/4754 and later independent Nr/Sr
remain unmet. The current metric best remains5616/4506 pending face terminal.
