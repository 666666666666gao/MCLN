# PV-Ground P2 fixed-box semantic diagnostic

Full ScanRefer development validation: 9508 expressions, native last/bbs,
GroupFree predicted-object-assisted two-stage protocol, seed2027. The same
trained P2 step3723 checkpoint supplies every candidate box and Mask for both
branches. Only the final semantic residual from P2 is bypassed.

| Semantic branch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
|---|---:|---:|---:|---:|
| P2 actual | 5613 | 4419 | 59.0345% | 46.4767% |
| P2 semantic bypass | 5612 | 4416 | 59.0240% | 46.4451% |

Actual minus bypass: +1/+3 hits. At0.25:2 repairs/1 damage; at0.50:4 repairs/1
damage. Selected Query changes in99/9508 expressions. This is a small positive
direct forward effect at this checkpoint, with no independent repeat.

All256 boxes, both score arrays, GT and input identities are saved privately.
CPU recount agrees with the native/manual evaluator receipts. All2434048
candidate IoUs were reconstructed from saved boxes and real root GT;
maximum absolute floating-point difference 5.484273699e-06,
threshold classification differences {'0.25': 0, '0.5': 0}.
Historical actual selection/threshold replay mismatches:0.
Model buffers and checkpoint unchanged; optimizer updates:0.

This does **not** compare a trained model without P2. Upstream features, D/G
reader, semantic head and geometry were trained with P2. It cannot prove why
the new model lost original-G capability, nor establish gradient conflict.

The completed same-budget training comparison remains G5600/4452 versus
G+P25613/4419 (+13/-33). Original verified G5615/4495 remains the strong
starting point. P2 is not promoted and the strict4754 target remains unmet.
The present result supplies little evidence for expanding P2 or immediately
adding P3. Keep the existing G weights and distinguish expression-conditioned
selection learning from changes to the geometry path in the next experiment.

Evidence: full/receipt.json, full/rows.jsonl.gz, INTAKE.json, SUMMARY.json,
run.py and the original paired formal rows. Compressed rows are private;
public receipts and aggregate report retain provenance without checkpoint copies.
