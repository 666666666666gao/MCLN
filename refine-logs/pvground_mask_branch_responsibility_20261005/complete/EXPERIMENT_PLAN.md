# Candidate-specific Query support versus shared Text support

This is the source-identified prerequisite in the closed Mask/Box fit-role diagnostic, not a new accuracy experiment. In the protected 4506 model, the Text Mask is copied to every candidate. Fused Mask qualification alone cannot identify the candidate's own support.

Reuse the exact frozen official PV -> original G -> protected boundary head factory and existing warm environment. Use seed 2027, B8, the same first eight shuffled augmented fit batches (64 rows), all 256 candidates, native matching and native signed bbs. Do not create an optimizer, update the model, replay detached loss gradients, save weights, run full validation, alter inference or discard candidates.

Save separate Text, Query and native fused Mask intersection/union counts and root IoU, fusion alpha, current Box IoU, native matched GT slots, bbs and selected Query. Verify all candidate Text logits are identical, compare row/point/GT identity with the closed probe, and independently expand each selected Query's three masks to input points. Independent CUDA forwards need not reproduce boxes/logits bitwise; record threshold differences without rewriting previous results.

CPU recount must quantify how many fused-Mask-qualified / Box-poor unmatched candidates also have candidate-specific Query IoU >0.5, how many qualify only with shared Text support, and row coverage. Qualification uses training GT only and is a proxy, not proof of physical identity. No geometry-positive pool or new loss is approved by this probe alone.

Estimated GPU duration: 7-8 minutes from warm parent reconstruction and eight forwards. First remote check 300 seconds, subsequent checks 240 seconds. Data-root-only outputs, no new weight archives, retain best and reconstruction parents. Fresh source review precedes deployment; fresh result audit precedes publication. Best remains 5616/4506; target 5615/4754, gap248, ACTIVE_UNMET.
