# Auxiliary-target formal launch readiness

**PASS.** No blocking or unresolved nonblocking defect found. Scope: `CLOSED_PREFLIGHT_LAUNCH_READINESS`. Requested Astra/max; backend unattested, same-family/provisional. This is a delegated follow-up in the same reviewer context.

Both actual GPU sanity arms closed with exit 0, exactly 2 optimizer updates and 0 weight files. Controller completion is 2026-10-05 16:51:05.651446+08:00, elapsed 930.3403 seconds. Controller/status/wait agree; each child log completion JSON exactly matches its receipt. All 21 downloaded artifacts match their existing intake sizes/SHA values, and uploaded source/specs match current local bytes. All 45 original source-review bindings remain exact.

Actual control extra-candidate counts are 146/143, with 120/126 outside-range faces. Member-target counts are 122/140, with 45/40 outside faces. All four updates contain empty rows and have finite positive isolated extra-head gradients. Native sampled inputs, native noisy GT and captured member target pass their actual exact checks. These counts come from the new GPU receipts, not the earlier cached cohort.

Both actual runs confirm frozen parent/zeroR states, qualified-only extra-output gradients, exact cached-upstream bbs/Mask after head updates, one native semantic-head call per forward, and strict in-memory model/optimizer restoration of 10 states / 1 group at step 2. The cached witness does not assert bitwise equality between separate complete GPU forwards. Imports match the actual predecessor, and both loads use the retained 4509 head with original G and official PV.

The formal controller launches fresh child processes. Training reconstructs 4509 and creates a fresh optimizer; no preflight weights or optimizer state are carried. Each arm keeps 29778 rows once, B8/tail2, 3723 updates, initial/terminal 6887 holdout and formal 9508. Geometry history is 7446 + 3723 = 11169. Both extra weights are 1; only native_gt versus member_gt changes auxiliary qualification and targets. Native matching/loss/G and official evaluation stay unchanged.

The launcher retains reviewed-source and actual-preflight gates, live remote byte comparison, completed-preflight/no-fit checks, GPU-idle check, 284880097-byte disk reserve, and exclusive GPU lock. Its comparison-specific derivation exactly reproduces the prepared launcher. The previous launch-to-finish 16295.315688 seconds substantiates the 16400-second estimate; first check 6200 seconds and later 240 seconds are estimates rather than completion evidence.

Ready to invoke the authorized launcher subject to those live gates. This reviewer did not query remote resources, deploy, load weights, train, or modify source/raw receipts. Exact reviewed paths/SHA values and clause-by-clause evidence are in `LAUNCH_REVIEW.json`.
