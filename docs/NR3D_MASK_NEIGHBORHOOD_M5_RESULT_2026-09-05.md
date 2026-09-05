# M5 terminal result: FAIL

Both arms completed the registered1024 updates on the same2048 fit rows.
The full6172-row/98-scene module holdout was evaluated at start and terminal.
These training scenes were seen by the frozen backbone. They are not
formal validation or whole-system unseen-scene generalization.

| Stage/arm | REC25 hits | REC50 hits | Mask25 hits | Mask50 hits | Mask mIoU |
|---|---:|---:|---:|---:|---:|
|baseline/native|6005|5306|5767|5057|68.88151972%|
|baseline/nearest|6005|5306|5745|5143|70.12066965%|
|terminal/native|6005|5306|5753|5050|68.76133394%|
|terminal/nearest|6005|5306|5750|5139|70.05065411%|

| Terminal nearest comparison | Mask mIoU delta(pp) | Mask25 fixes/breaks/net | Mask50 fixes/breaks/net | Fixed screen |
|---|---:|---:|---:|---|
|terminal_nearest_minus_protected_start|+1.16913440|31/48/-17|190/108/82|FAIL|
|terminal_nearest_minus_terminal_native|+1.28932018|38/41/-3|200/111/89|FAIL|

The nearest-two substitution already raises mean IoU before training, while
reducing Mask25 hits. Training changes nearest mIoU by-0.07001553pp,
Mask25 by+5 and Mask50 by-4 relative to its own start. Native training changes
mIoU by-0.12018578pp and Mask hits by-14/-7. The observed mean-IoU advantage
therefore comes from the grouping intervention; this short training adds no
mean-IoU gain to either arm.

This run's native protected-start REC is6005/5306. The earlier P2/R1 cached
start was6005/5312; cross-run input/output identity has not been established.
Use the matched M5 start for all M5 changes, without combining those baselines.

The fixed criterion requires at least.2pp mIoU gain against BOTH terminal
native and protected native start, with neither Mask hit threshold declining.
The endpoint, thresholds, learning rate and data were not selected after
viewing intermediate quality. Paired scene-bootstrap95% intervals(seed0,
2000 draws) and per-scene changes are in terminal_summary.json.

terminal_nearest_minus_protected_start: mIoU delta95% scene interval [+0.25020238, +2.50210552]pp.
terminal_nearest_minus_terminal_native: mIoU delta95% scene interval [+0.36901974, +2.59253195]pp.

All frozen model parameters/buffers and source/data/parent checkpoint checks
passed. REC tensors, input point hashes, REC Query and Mask Query identities
exactly match the start for every holdout row. All16 permitted Mask parameter
tensors changed in each arm; both fresh optimizers ended at1024 steps.
Only addon projection/optimizer states were saved in the isolated remote
directory. No protected/full checkpoint was replaced or uploaded.

Receipt SHA256: d6998a11c1ec9611bbbdd84beb8a753049a634dfe66e6da5f1d278d463b5d2d5
Manifest SHA256: abc46c2e543ec05ade30523f83cbded0689d7d88def07ca8dd06b4f79ba439c1
Runtime excluding dataset initialization: 3552.081 seconds.

The registered nearest-two training screen failed. Archive both endpoints;
do not promote nearest grouping, the trained native control, an earlier
checkpoint, or a retuned M5 variant on this result. The M4 direction does
transfer to held-out mean IoU, with a positive scene-bootstrap interval, but
does not meet the two-threshold requirement. L1 remains a separate REC
experiment and the formal three-benchmark target remains open.
