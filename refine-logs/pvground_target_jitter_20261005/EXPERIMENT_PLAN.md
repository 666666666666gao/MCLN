# Native target-box jitter check

Continue the closed PV-Ground geometry-responsibility comparison. This is a CPU data diagnostic, not training or a new accuracy result.

Recreate the exact 64 augmented fit rows used by `pvground_query_geometry_cohort_20261005`, with its native dataset, seed2027, two data workers, batch8 and shuffled fit partition. Keep all native random calls and returned fields. Add a read-only capture of the annotated root box from the already augmented scan immediately before `_get_target_boxes` applies its independent six-coordinate jitter. Do not derive truth from a predicted mask or sampled point approximation.

Assert that row order, scan IDs, sampled-point bytes, noisy root labels, valid GT slots and no-augmentation reference fixtures reproduce the closed diagnostic. Failure stops this check; there is no relaxed matching.

Save 64 clean/noisy root boxes. On the existing saved 16384 candidates, recompute IoU and boundary targets against each label definition. Report how many supported, unmatched candidates switch the Box<=0.5 qualification, how many boundary targets remain outside the original nodes, and geometry-head changes on the same fixed noisy-label cohort. Existing matching and mask support are held fixed, not recomputed under alternative boxes.

Use the warm remote CPU environment; no model imports, checkpoint loads, optimizer, GPU forward, package installation or weight creation. Estimated 2-5 minutes; first remote completion check after180 seconds, later240 seconds. Then collect small JSON output and run NumPy recount locally. Review source before execution and closed evidence before claiming results.

Success means an exact replay and a bounded answer about label jitter. It does not establish the cause of formal accuracy deficits or authorize changing the dataset. Preserve current best5614/4509, originalG/PV and the protected V99 chain. No new negative checkpoints.
