# L1 native preflight completed; paired training launched

The isolated v2 preflight completed with exit0:16 fit expressions from16
scenes,20 forwards, zero optimizer updates, no holdout/formal rows.
All input hashes match M3. Both zero-initialized adapters exactly reproduce
the native final Query, centers, sizes, selector scores and raw Query Mask
logits on these inputs.

The original total, grounding and Mask losses each produce finite nonzero
gradients for both82944-parameter adapters. A fixed.001 diagonal intervention
changes Query/Box/Mask tensors finitely; the weights are then restored to zero.
This proves connectivity and implementation behavior, not improved accuracy.

| Arm | Total gradient norm range | Grounding range | Mask range |
|---|---:|---:|---:|
|text|0.023021668–0.041796118|0.015484120–0.022019995|0.011947145–0.036754444|
|position|0.001512128–0.002899771|0.000676328–0.001102064|0.000799710–0.002451673|

All original state keys and every tensor value match the protected checkpoint.
Source/data/parent hashes are unchanged. The earlier v1 attempt completed its
four probe batches but failed the final verifier because it compared dictionary
insertion order. The checkpoint has the same keys/values in a different order.
v2 changes only this verifier to compare the key set and then every named tensor,
and records the order difference explicitly. v1 logs remain archived; it made
zero updates and is not a failed trained scientific screen.

The earlier CPU-only revision corrected reading M3's actual batches/rows
receipt nesting before any GPU forward. Its original CPU artifacts are retained.
No attention mechanism, input selection, loss, learning rate or quality gate
was changed between these preflight revisions.

Preflight receipt SHA256: ba8d31f3ac71d1f4bfd88314cb7627b660a8948cf96ed89d5162508a711814c3
Active manifest SHA256: 354c680b44ac799a1c6d573aebfd8653b474e9a0cda8ae71818a76ac237a3746
Runtime excluding dataset initialization: 33.944 seconds.

Training launch: 2026-09-05T21:13:24.498669+08:00, screen mcln_l1_train.
Remote directory: /root/autodl-tmp/mcln_text_position_l1_20260905_v2/train/.


Run the registered native zero-start holdout first, then both fixed6687-step
trajectories and their terminal holdout. The geometry-key arm must clear BOTH
the equal-parameter text-key control and protected start: REC25 at least10 net
hits, with no REC50 or Mask threshold/mIoU decrease. No intermediate endpoint
selection. Actual training throughput is still pending at launch.

M5 remains a separate failed two-threshold screen. Its nearest-grouping mean
IoU advantage is supported on the module holdout, but training did not further
increase either arm's mean IoU and Mask25 suffered. Neither M5 nor these L1
preflight checks constitute a new formal result or meet the three-benchmark goal.
