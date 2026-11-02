# ScanRefer Training Observations

2026-11-02; current run1793579788 only. Raw remote log extraction:
`docs/results/mcln_pvground_20261101/scanrefer-periodic-training-loss-e1-e10.json`.
Collector: `tools/collect_pvground_training_loss_log.sh`.

These are cumulative running means at batch6000/6081 in each epoch,
not exact full-epoch means. The raw timestamped loss lines are preserved.
Only existing logs were read; no model forward or local tests were run.

| Epoch | Loss | Box L1 | GIoU | Semantic alignment | Superpoint Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.3920 | 1.2628 | 2.8827 | 35.6224 | 0.1733 |
| 8 | 7.2987 | 1.2284 | 2.8418 | 35.5939 | 0.1682 |
| 9 | 7.2600 | 1.2358 | 2.8480 | 34.9410 | 0.1680 |
| 10 | 7.2505 | 1.2302 | 2.8484 | 35.1608 | 0.1669 |

Total running loss falls about1.91% from epoch1 to10. In the first ten
complete validations,Acc050 varies46.287337%--47.381153%; epoch8 is the
current050-first best. Epoch10 completed17:26:28UTC on9508 samples,
5366/4461 hits=56.436685%/46.918385%,CPU own epoch10 verified and
hardlink preserved. Lower training loss has not established a material
validation gain. These limited observations neither diagnose overfitting
nor demonstrate a missing loss,dead module,or bad learning rate.
Component magnitudes are unweighted reported terms with different
aggregation/scales; comparing35.6 to1.26 does not prove gradient dominance.

Static supervision trace in the current code:

- Reader adds its residual before the final decoder's visual normalization
  and feedforward computation (`models/encoder_decoder_layers.py`).
- Support replaces `sp_last_pred_masks` before the model returns.
- Span replaces `last_center` and `last_pred_size` before the model returns.
- `models/losses.py` feeds those final fields to the last-layer set criterion.

This trace and the run's first-update witness exclude a simple claim that
the modules were never connected to training. They do not establish useful
learned behavior. Real chosen/oracle IoUs,Support effects and Span gate
distributions remain required before a module change.

The observed unique/multiple gap remains the reason to inspect candidate
context and support quality. Source-pinned ViL3DRef,ConcreteNet,MaskDINO and
Mask3D comparisons and conditional decisions are documented in
`docs/PVG_MULTIPLE_INSTANCE_REFERENCE_NOTES.md` and the current technical
handoff. Existing contextual queries and contrastive losses must be
accounted for; do not infer a missing capability from another repository.

## Target Hits And Subgroup Constraints

The selected epoch8 has1234/1058 unique hits on1419 examples and
4188/3447 multiple hits on8089. These sum to5422/4505 on9508.
The user targets require at least5705/4945 hits on the same checkpoint:
ceil(9508*0.60) and ceil(9508*0.52). The deficits are283/440 hits.

| Threshold | Required total hits | Current deficit | Remaining unique errors | Extra multiple hits needed even with perfect unique | Required multiple hits if unique stays at epoch8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 025 | 5705 | 283 | 185 | 98 | 4471/8089 =55.272592% |
| 050 | 4945 | 440 | 361 | 79 | 3887/8089 =48.052911% |

Thus improving unique alone cannot meet either target from this checkpoint.
If unique performance stays unchanged,multiple needs+3.498578/+5.439486pp.
The perfect-unique calculation is an arithmetic upper bound,not an
achievable intervention or a forecast. Gains at one threshold need not
imply gains at the other; final acceptance still uses both overall metrics
from one complete checkpoint.

Existing full-epoch subgroup evidence,without model execution:

| Epoch | Unique hits025/050 of1419 | Multiple hits025/050 of8089 |
| --- | ---: | ---: |
| 1 | 1224/1041 | 4200/3397 |
| 2 | 1222/1047 | 4193/3449 |
| 3 | 1230/1058 | 4171/3395 |
| 4 | 1230/1056 | 4136/3353 |
| 5 | 1213/1031 | 4128/3370 |
| 6 | 1235/1059 | 4157/3386 |
| 7 | 1219/1040 | 4139/3410 |
| 8 | 1234/1058 | 4188/3447 |
| 9 | 1233/1050 | 4157/3423 |
| 10 | 1231/1052 | 4135/3409 |

Source:`docs/results/mcln_pvground_20261101/mcln-current-scanrefer-evidence.jsonl`.
Integer hits are reconstructed from logged subgroup rates and their fixed
denominators; each pair sums to the complete overall hits already audited.
Epoch2 has two more multiple050 hits than8,but fewer unique050 hits:
this does not change the overall050-first selection of8. Do not construct
a result from the best subgroup of different epochs.

The necessary multiple improvement prioritizes the existing chosen/oracle
and paired bypass diagnostics. It does not distinguish proposal coverage,
semantic ranking,Support masks or Span corrections. No module change or
benefit is established by this decomposition.

## Log Precision Fix

The original epoch logger prints rates with five decimal places. A1e-6
rate becomes0.00000,so future decay logs can look like training stopped.
Changed the repository logger to eight significant digits and made the
collector accept scientific notation. This changes display only,not LR,
schedule,optimizer or model. Existing live training source was not changed.
Historical logs remain precision-limited; use saved optimizer/scheduler
states for exact rates,as in `docs/PVG_OPTIMIZER_RECIPE_AUDIT.md`.
