# Current MCLN + PV-Ground Handoff

Update15:31UTC: SR e8 full17726 validation completed15:24:27UTC,
11124/8586 semantic Top1 hits=62.755275%/48.437324%. CPU own epoch8,
1134 tensors,optimizer/scheduler confirmed and hardlinks preserved.
All three current trainers and metric keepers verified live at15:29UTC;
NR/SR task checkpoints null, configurations unchanged. Scan e9 training,
NR e22 training, SR e9 training. Keeper successfully advanced SR best.
Publication commitbc70601 pushed; full table now37 rows. Targets unmet.

Update15:23UTC: independent metric keepers are deployed and verified live
on all3 hosts. Screen names are pvg_DATASET_metric_keeper; interval240s.
They CPU-verify a newly selected best and atomically replace
`RUN_DIR/semantic_best.pth` (hard link) and `semantic_best.json`.
Existing retained bests: Scan e8, NR e21, SR e7. Actual Python processes,
first successful loop logs and link counts were checked, not just screens.
Training code/processes and two-hour user notification cadence are unchanged.
The trainer still returns None for ordinary evaluation; this is an external
mitigation, not an in-process evaluator repair. Keepers must remain live.
They exit on a missing best weight or inconsistent metrics; inspect their
logs at each monitor run. They cannot recover historical deleted weights
or guarantee survival through host/process failure.
Tests now7 passing:3 module tests plus4 keeper tests covering trainer
unlink, best replacement, unchanged best, Scan semantic/mask separation,
partial validation and missing-best rejection.

Updated: 2026-11-02, approximately 15:07 UTC / 10:07 America/New_York.
This section supersedes historical September and clean-baseline instructions.
Do not restart stopped clean-baseline or previous failed fine-tunes.

## Acceptance Targets

Require one complete checkpoint to satisfy both ScanRefer thresholds:
Acc025 >= 60.0% and Acc050 >= 52.0%, on all 9508 validation examples.
NR3D requires Acc025 >= 59.8% on 7899 examples.
SR3D requires Acc025 >= 69.4% on 17726 examples.
Use last semantic alignment Top-1; never substitute position or mask metrics.
Select ScanRefer by Acc050 then Acc025, NR/SR by Acc025 then Acc050.
The goal remains active and none of these targets has been demonstrated.

## Verified Current Results

| Dataset | Epoch | Hits025 / Hits050 | Acc025 / Acc050 (%) | Gap To Target (pp) |
| --- | --- | --- | --- | --- |
| ScanRefer | 8 | 5422 / 4505 | 57.025663 / 47.381153 | 2.974337 / 4.618847 |
| NR3D | 21 | 3557 / 2642 | 45.031017 / 33.447272 | 14.768983 at 025 |
| SR3D | 8 | 11124 / 8586 | 62.755275 / 48.437324 | 6.644725 at 025 |

Each listed checkpoint was loaded on CPU and its own epoch checked.
ScanRefer has 1168 model tensors; NR/SR have 1134. All contain optimizer
and scheduler state. Hard links were preserved under
`runs/DATASET/preserved_checkpoints/ckpt_epoch_N.pth`.
NR e21 improves Acc025 over e20 but its Acc050 is lower: report the e21
pair together. Do not replace its 33.447272% with e20's 33.523231%.
All 37 completed epoch rows recovered so far are in the accompanying CSV:
ScanRefer 1-8, NR3D 1-21, SR3D 1-8.
Rows without weights are explicitly marked `missing_retention_deleted`.

Log timestamps are server UTC+08:00. Subtract eight hours for UTC.
Earlier chronological notes contain incorrect conversions for some epochs.
For example, NR e15 completed at server16:33:10, which is08:33:10UTC,
and SR e5 at server14:58:52, which is06:58:52UTC.
The raw evidence JSONL retains server timestamps to prevent ambiguity.

## Runs And Method Attribution

Remote root: `/root/autodl-tmp/mcln_pvground_ft_20261101`.
Only use `work/remote_pvground.ps1`; credentials must never enter documents.

| Dataset | Endpoint | Screen | Current Training Recipe |
| --- | --- | --- | --- |
| ScanRefer | connect.bjb2.seetacloud.com:48073 | pvg_scanrefer_mcln_finetune | run1793579788, batch8, LR1e-5, backbone2e-5, module1e-5, 100epochs, decay10/30 |
| NR3D | connect.nmb1.seetacloud.com:32749 | pvg_nr3d_mcln_scratch | run1793553113, batch12, accumulation4, LR1e-4, backbone1e-3, 240epochs, decay150 |
| SR3D | connect.nmb2.seetacloud.com:45914 | pvg_sr3d_mcln_scratch | run1793552465, batch12, accumulation2, LR1e-4, backbone1e-3, 140epochs, decay30/40 |

All three screens and GPU processes were verified live this turn.
NR e21 completed validation while observing; Scan e9 and SR e8 continue.
NR/SR checkpoint_path and pp_checkpoint remain null. Frozen local RoBERTa
is pretrained; do not describe the text encoder as random initialization.
NR/SR are currently original MCLN scratch runs, without the additional
Reader/Support/Span tensors. Their metrics are not evidence that those
three modules improve NR/SR. Preserve these runs as references; do not
silently relabel them as the proposed three-module method.

ScanRefer loads only the official epoch54 model initialization, with a
fresh optimizer/scheduler and Reader/Support/Span enabled. Checkpoint SHA256:
`a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d`.
Its epoch9 batch1 witness shows finite loss5.888957 and finite nonzero
gradient/parameter changes in all three modules. Latest logged average
loss was7.2812. This proves optimization runs, not improvement over the
official initialization, which has no matched epoch0 evaluation here.

## Problems, Evidence, And Next Experiments

1. Checkpoint retention is not metric-aware in these runs. The ordinary
   evaluator logs metrics but does not emit the expected JSON receipt;
   retention records/best are empty. Two-hour polling exceeds the roughly
   one-hour NR epoch duration, so several checkpoints disappeared before
   inspection. New collector CPU-audits every surviving numbered checkpoint
   and preserves hard links. It cannot recover deleted weights.
   Next implementation: preserve a validated metric-best checkpoint at the
   epoch save/evaluation boundary. Keep the two-hour notification schedule;
   a two-hour observer alone cannot guarantee preservation.
2. Old status checks truncate logs to a short tail. This hid complete
   ScanRefer epochs3-8 and some NR rows. The new collector scans the entire
   current log and reconstructs integer hits using semantic subgroup counts.
   Scan uses unique1419 + multiple8089. It accepts only the FIRST occurrence
   of subgroup values per epoch, because later unique50/multi50 are masks.
   Reconstructed rates must agree with the rounded overall semantic rate.
3. ScanRefer Acc050 oscillates46.29-47.38% over epochs1-8; e8 exceeds e2 by
   only9 hits (0.094657pp). This is not evidence of a robust method gain.
   Epoch8 unique/multiple semantic025 are86.962650%/51.774014%;
   semantic050 are74.559549%/42.613426%. Multiple-object disambiguation is
   a measured weakness, but subgroup difficulty alone does not prove cause.
   Next: evaluate the SAME retained e8 checkpoint with module bypasses,
   matched preprocessing and per-example predictions; quantify ranking,
   box localization and mask support separately before modifying modules.
4. Span currently uses a hard0.5 mask reference with detached masks and
   clamps raw mixing weights to[0,1]. These are code facts, not a diagnosed
   failure. Collect reference-valid fraction, gate boundary occupancy and
   per-axis errors; test whether a learned correction actually changes
   relevant boxes. Only change its confidence/extent mechanism when those
   measurements identify a failure.

The three intended module responsibilities remain Reader (local visual
evidence), Support (candidate mask membership correction), and Span (axis
extent correction). No novelty claim is established by their names.
Require a matched baseline and module ablations before claiming benefits.

## Open-Source Reference Checked

Official MCLN repository: https://github.com/qzp2018/MCLN
MaskDINO: https://github.com/IDEA-Research/MaskDINO
Inspected MaskDINO commit3831d8514a3728535ace8d4ecc7d28044c42dd14,
`maskdino/modeling/transformer_decoder/maskdino_decoder.py`, via GitHub API.
It exposes mask-to-box initialization and separates proposal initialization
from subsequent decoder refinement. This motivates a diagnostic comparison
of our hard mask extent reference with native box geometry. It does not
justify copying a 2D module unchanged into 3D or claiming a measured gain.
The next experiment must retain Reader/Support/Span and test one diagnosed
mechanism at a time, with source attribution and full validation evidence.

## Tuning Policy And Validation

When a dataset reaches target, freeze its accepted configuration. If another
is approximately1pp away, first investigate dataset-specific module widths,
depths or internal parameters without changing the architecture/module set.
Within approximately0.3pp, prefer LR/batch/seed trials before a method change.
No current deficit is within0.3pp. No hyperparameters changed this turn.

Local targeted verification: `python -m pytest tests/test_pvground_innovations.py -q`
passed3 tests. Initial collection failed because the sparse checkout omitted
the tracked reader source; materializing that existing file fixed collection.
This is focused CPU coverage, not a full GPU training regression test.
Active ScanRefer's first-update witnesses supply independent runtime evidence.

User's canonical desktop document:
`C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`.
Keep its historical sections; the current dated section takes precedence.
Publish code and this document to https://github.com/666666666666gao/MCLN .
