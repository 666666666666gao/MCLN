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

## Log Precision Fix

The original epoch logger prints rates with five decimal places. A1e-6
rate becomes0.00000,so future decay logs can look like training stopped.
Changed the repository logger to eight significant digits and made the
collector accept scientific notation. This changes display only,not LR,
schedule,optimizer or model. Existing live training source was not changed.
Historical logs remain precision-limited; use saved optimizer/scheduler
states for exact rates,as in `docs/PVG_OPTIMIZER_RECIPE_AUDIT.md`.
