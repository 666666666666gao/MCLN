# Existing object-reference input audit

The actual final Decoder object memory was verified on four original P2 fit
rows (IDs 0, 1, 3, 4). This was one frozen-backbone forward with zero optimizer
updates and zero heldout or official rows. No new head accuracy was evaluated.

The observed `detected_feats` and `detected_mask` exactly matched reconstruction
through the existing protected object-input branch. All model parameters and
buffers remained equal to the protected checkpoint. Input boxes and validity
were the existing `all_bboxes` and `all_bbox_label_mask` under `butd_cls`.

The 288D features concatenate 128D box-position encoding and 160D predicted-class
encoding. They are not independently pooled object appearance features. The
forward uses predicted classes; GT classes are used only afterward to describe
input accuracy. Existing instance-box inputs remain part of the benchmark
protocol, so this audit is not evidence of GT-object-free inference.

| Fit row | Scene | Valid object slots | Full-256 Query coverage | Target Top32 coverage | Correct predicted classes |
|---|---|---:|---:|---:|---:|
| 0 | scene0525_00 | 66 | 12 | 8 | 51 |
| 1 | scene0265_00 | 31 | 5 | 4 | 28 |
| 3 | scene0668_00 | 36 | 7 | 7 | 33 |
| 4 | scene0505_00 | 57 | 14 | 13 | 43 |
| Total | Four training expressions | 190 | 38 | 32 | 155 |

Coverage uses object-box/Query-box IoU > .25 and is an object-availability proxy,
not labelled text-anchor recall. The 35 incorrectly predicted classes also
show that broader object memory does not guarantee correct semantic evidence.
These four rows establish the runtime interface and a concrete missing-memory
case; they do not estimate dataset-wide accuracy or prove scorer improvement.

The separate R1 experiment compares the same global/pair readout under Query
versus existing object memory. Its fixed primary is `object_pair`, with all four
arms reported. P2 v1 remains failed and sealed. R1 is tracked in draft PR #9:
https://github.com/666666666666gao/MCLN/pull/9

Evidence is in `refine-logs/object_reference_inputs_20260905_v1/`. The remote
audit directory is `/root/autodl-tmp/mcln_object_reference_input_audit_20260905_v1`.
The controller completed with exit 0. Receipt SHA-256:
`59a8c6faa1dc8e8cfe5a217b9025f30e2758af5da3bde8eb07cdba40ddfd4fc4`.
Protected checkpoint SHA-256:
`76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`.
