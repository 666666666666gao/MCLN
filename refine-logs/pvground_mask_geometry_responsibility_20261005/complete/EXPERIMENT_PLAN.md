# Actual fit-batch Mask/Box geometry responsibility probe

This is the next read-only diagnostic recorded in handoff §20.376.61, not a new training experiment.

- Reconstruct protected official-PV → original-G → 4506 distribution geometry checkpoint. Reuse the executed warm runtime and actual DATA loader. Install the existing fresh zero-output R only because the reviewed source port expects it; freeze/eval all model state.
- Seed2027, B8, first eight shuffled fit batches (64/29778 rows), train point and detected-box augmentation and train voxel preprocessing. Assert row IDs match the completed quality fit's first64. No outcome-based row selection.
- Retain all256 candidates. Compute current Box/root IoU, exact native fused Mask/root point IoU using member counts, actual native last Hungarian assignment to valid training GT slots, native signed bbs and selected query.
- Replay final native L1/GIoU on detached Box-output leaves and the executed boundary distribution loss on detached boundary-logit leaves. Record direct gradients for every candidate, assert unmatched direct gradients zero. This is not an assertion about total model gradients or shared-parameter effects.
- No optimizer constructed, zero steps, no backward through model parameters, no weights saved. Model state must be exact before/after. Existing parent weights stay protected.
- Outputs: 64-row JSONL, 8 compact per-candidate NPZ files, actual imports/load/receipt, CPU recount. No formal accuracy result and no re-run of 9508 validation.
- Interpret role counts only for these augmented training inputs at the protected checkpoint. The prior validation root-only match audit is not evidence of actual training responsibility; neither are these current matches historical per-step assignments. Report if native targets contain only root, rather than claim nonempty multiGT protection was exercised.
- Primary next decision: is there an observed population of root-Mask-qualified, Box-poor, unmatched candidates lacking a direct final Box/boundary target? Do not add a new loss or claim causality before observing the probe.

Estimated warm reconstruction plus8 forwards/criterion and tiny leaf-gradient probes: 5–8 minutes. First remote check after300 seconds, later checks240 seconds. One A100 job, no dependency installation.
