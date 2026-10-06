# Actual execution tracker — 2026-10-06

The fixed EXPERIMENT_TRACKER.md preserves its source-review preparation snapshot. This file records actual execution and does not alter the reviewed/deployed model, diagnostic or plan.

| Stage | Actual state | Evidence |
|---|---|---|
| Source review | WARN, zero blocking findings; fresh same-family/provisional, SOURCE_ONLY | SOURCE_REVIEW.json and SOURCE_REVIEW_CALL.json |
| M0 actual GPU first8 | Closed exit0 at 09:25:21 CST; no optimizer, updates or new weights; model states unchanged | preflight_wait.json, complete/preflight/receipt.json |
| M0 independent local raw-point CPU check | Actual8 rows replayed using all50000 input points; matches remote CPU summary, zero threshold flips | local_cpu_m0/CPU_SUMMARY.json |
| Formal launcher/observer review | PASS, zero blocking findings; same-context followup, SOURCE_ONLY | FORMAL_LAUNCH_REVIEW.json and FORMAL_LAUNCH_REVIEW_CALL.json |
| M1 actual9508 | Controller699895 / child699896 started 09:36:59 CST; not yet complete | formal_launch.json, formal_observer_live.json |
| Sole M1 observer | Native session59532; first remote completion check10:03:59 CST, then240 seconds | FORMAL_LIVE_SESSIONS.json |

M0 counts are implementation checks, not formal accuracy. No new training has started. Protected metric-best remains5616/4511, and previous support-reference closed nonbest weights were already removed. The inherited training counters in spec.json describe history, not updates performed by this diagnostic.

M1 preserves all256 candidates, one native last/bbs selection and the sameQueryBox/Mask identity. Exact and q0.005 extent boxes are offline counterfactuals only. GT grouping is diagnostic, not a deployment input. Full raw-point independent sorting replay is limited to M0; M1 records member extrema and neighboring order values/ranks for CPU decoding verification.
