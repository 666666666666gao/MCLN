# Actual closed execution tracker — 2026-10-06

This supersedes the frozen Doc80 launch snapshot ACTUAL_EXECUTION_TRACKER.md. The launch snapshot is kept unchanged because it is bound by the prior publication/source review SHA; its M1-running state describes that earlier observation, not this terminal state.

| Stage | Actual terminal state | Evidence |
|---|---|---|
| M0 | Actual8row GPU closed exit0; independent local all50000point CPU replay agrees;0optimizers/updates/newweights | preflight_wait.json and local_cpu_m0/CPU_SUMMARY.json |
| M1 inference | All9508 rows closed10:08:24 CST, controller699895/child699896 exited0 | formal_wait.json, complete/formal_status.json and formal.exit |
| Sole M1 observer | Native session59532 closed0;1197files/462713589bytes collected; no weights | formal_wait.json and FORMAL_CLOSED_SESSIONS.json |
| M1 CPU | ActualSPmember extrema/intersections and quantile neighboring value/rank decoding checked on9508; zero threshold flips | complete/formal/CPU_SUMMARY.json |
| OwnMask CPU breakdown | Actual ownMask member intersections checked on9508; native session34764 closed0 | complete/formal/FAILURE_BREAKDOWN.json |
| New network training | NOT_STARTED; prepared source/plan only | ../pvground_mask_reference_20261006/EXPERIMENT_PLAN.md |

Current trained metric-best remains5616/4511. Offline exact extent5598/4848 and q0055590/4781 are counterfactuals, not trained-model accuracy. They are never used to overwrite the model or native selection. Goal remains unmet.

Fresh terminal integrity review is in progress. Formal fullraw independent sorting replay was not performed; onlyM0 actual8 raw rows had that replay. Full9508 stores actual member extrema and quantile neighbors/ranks. Model states unchanged, no optimizer/updates/weights, all256 candidates retained and one native score.
