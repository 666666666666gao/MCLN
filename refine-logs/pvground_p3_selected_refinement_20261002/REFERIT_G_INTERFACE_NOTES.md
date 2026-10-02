# PV-Ground G: Nr3D / Sr3D interface notes

Read-only local source inspection on 2026-10-02. This is a future adaptation checklist, not an implemented module, runtime preflight, or new benchmark result. ScanRefer development and the active same-tail raw/fused pair remain unchanged.

| Native recipe | Root token target weights: positive / modify / pronoun / relation | CE and contrastive multiplier before decoder-layer averaging |
|---|---|---|
| ScanRefer | 0.6 / 0.2 / 0.2 / 0.1 | 0.5 |
| Nr3D | 0.6 / 0.2 / 0.2 / 0.1 | 1.0 |
| Sr3D | 0.625 / 0.125 / 0.125 / 0.125 | 1.0 |

Evidence: the actual imported PV source retained under `../pvground_p3_next_20261002/complete/source/imported/models.losses.py`, lines 484–487 and 944–955. Native layer averaging uses `num_decoder_layers + 1`; the present six-layer ScanRefer G helper hardcodes `0.5 / 7` and ScanRefer token weights. Merely widening its dataset-name assertion would produce the wrong correction for Nr3D/Sr3D.

The actual dataset source `src.joint_det_dataset.py`, lines 1264–1272 and 1402–1410, distinguishes two fields: `language_dataset` selects the benchmark loss recipe, while `sample_dataset` identifies the annotation source. Joint ScanNet detection rows can carry the referring benchmark's `language_dataset`. Future G eligibility must inspect the actual referring annotation source and preserve detection-row labels.

The existing `models/nr3d_semantic_assignment.py` is an old MCLN Nr3D-only adapter. It skips ScanNet rows, returns an unscaled CE delta, and lets the native criterion apply its overall coefficient. This is useful prior code, but it is not a validated PV-Ground/Sr3D implementation. Its interrupted historical experiment is neither a completed negative result nor a demonstrated cross-benchmark gain.

Reuse the prepared September 17 formal input contracts: Nr3D 7899 rows, Sr3D 17726 rows, native `butd_cls` instance boxes and predicted categories, `butd=false`, `butd_gt=false`, primary `bbs`, no Mask gate. These are saved preparation receipts with zero forwards/updates, not fresh remote validation. Keep this object-input protocol visible in reporting.

When ScanRefer passes the agreed development criterion and the structure is fixed, adapt only the necessary G label/scale and sample-eligibility interfaces. Verify replacement against the actual native final-layer CE and its gradient, preserving Hungarian assignments and other matched instances. Do not revive the old competition runner, add inference dataset gates, or launch Nr3D/Sr3D now.

Source hashes and saved contract summaries are recorded in `referit_g_interface_receipt.json`. No source mutation, remote operation, model forward, or optimizer update was performed for this inspection.
