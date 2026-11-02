# Current MCLN + PV-Ground Handoff

Update16:31UTC: all three current trainers and independent metric keepers
verified live through the secure wrapper; configurations unchanged.
Scan e10 training3075/6081, GPU23671MiB; NR e23 training1551/3742,
GPU22833MiB; SR e9 training2870/6486, GPU22924MiB.
No new complete validation: latest complete Scan9/NR22/SR8,
selected best Scan8/NR21/SR8. Estimated next full validations:
NR approximately17:10UTC, Scan17:20-17:30UTC, SR approximately18:00UTC.
These are estimates from observed training rates plus previous validation
durations, not completion promises. No concurrent GPU diagnostic launched.

User execution boundary: no tests on the local machine, including Windows
or local WSL CPU/GPU diagnostics, model forward, evaluation or training.
Do not continue local-probe environment preparation or download model/data
for local testing. The prior isolated conda clone completed; this does not
mean a model test ran. Its proposed model probe is cancelled. Future needed
tests and diagnostics run remotely only. Historical CPU test counts below
describe past work and do not authorize rerunning them locally.

The desktop brief MCLN-PV-Ground-training-handoff.md has been merged into
the user's sole desktop handoff:
`C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`.
Keep the source as history, do not update it again. The canonical file's
Consolidated Current Handoff section distinguishes historical attempts and
snapshots from the current run. This repository file remains technical
documentation published to GitHub, not a second desktop handoff.
The ACTIVE two-hour monitor now preserves both constraints and the current
39-row/latest-best state, with its existing notification policy unchanged.

Update16:12UTC: NR e22 full7899 semantic validation completed16:08:11UTC
(server11/03 00:08:11). Hits3543/2754=44.853779%/34.865173%.
CPU own epoch22,1134 tensors,optimizer/scheduler verified and hardlinked.
Acc025 falls14 hits versus e21 while050 rises112 hits; best stayse21.
Do not combine e21Acc025 with e22Acc050. Full table39 rows:
Scan1-9,NR1-22,SR1-8. BestScan8/NR21/SR8,targets still unmet.

Update16:10UTC: semantic diagnostics pushedc8437a3;15 CPU tests passed.
Helpers staged on Scan server under`offline_audits/c8437a3/tools`,archiveSHA
34a234ad96bbaea6b74a48550d0aeb868154d019118943961c921472f7cbca67.
Remote imports of current trainer,evaluator and IoU functions passed,
CUDA remained uninitialized. Active training source untouched. This is
import preflight only,not model/checkpoint/GPU-forward validation.

Update15:47UTC: ScanRefer e9 completed15:45:32UTC, full9508
last semantic Top1 hits5390/4473=56.689104%/47.044594%.
CPU own epoch9/1168 tensors/optimizer+scheduler verified, fixed hardlink
preserved. Both thresholds fell32 hits versus e8; selected best stays e8.
Full table now38 rows: Scan1-9,NR1-21,SR1-8. Targets remain unmet.
Offline audit hooks/eval entry,CPU evidence and11 passing tests pushed
commit31373aa. No real-sample bypass result or method change yet.

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
All 39 completed epoch rows recovered so far are in the accompanying CSV:
ScanRefer 1-9, NR3D 1-22, SR3D 1-8.
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
   Mitigation deployed: CPU metric keepers poll every240sec and preserve
   the audited selected best through atomic hardlink replacement. SR e8
   advancement was observed. Native evaluator remains unfixed; inspect
   keeper process/log health during two-hour monitoring.
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

   CPU weight audit at15:38UTC of the retained Scan e8 checkpoint,
   SHA2563572193fef0f32d1d6ca22b41cbbb02932a30de0a51b18deb3b0c40ee37bcf38:
   all34 innovation tensors finite; all six zero-initialized output
   weight/bias tensors have every element nonzero. Reader output weight
   norm0.531269,bias0.026472; Support0.208826/0.005804;
   Span0.183130/0.010051. Optimizer state steps all48648.
   This excludes unchanged zero outputs, but does not establish forward
   activation magnitude, clamp occupancy, gradient health at e8, causal
   contribution or a method gain. Raw evidence:
   `docs/results/mcln_pvground_20261101/scanrefer-e8-module-weight-audit.json`.
   Use sample-based hooks in an isolated evaluation process for raw/clamped
   gates and reference validity; do not launch an additional GPU evaluation
   on the occupied4090 or modify the active trainer to obtain those values.

   Offline instrumentation implemented:
   `tools/eval_pvground_audit.py` accepts normal single-GPU MCLN eval flags
   plus `--audit-jsonl PATH --audit-bypass none|reader|support|span`.
   Run through torch.distributed.launch with a separate port, `--eval`,
   `--use_pvground_innovations`, complete trained checkpoint_path and matched
   preprocessing; omit model_only_initialization and initialization-only
   pp_checkpoint. Pin a preserved epoch path, never a moving best alias.
   `tools/pvground_forward_audit.py` hooks only the offline model and writes
   per-sample candidate gate boundary counts, means, valid reference counts,
   and absolute axis shifts. Support bypass keeps measured geometry but
   returns original member logits; Span bypass restores exact native boxes;
   Reader bypass zeros its residual. No checkpoint keys or weights change.
   Metadata includes checkpoint SHA and invocation. Extended helper
   `tools/pvground_semantic_audit.py` exports selected-query gates, identity/
   pointSHA, semantic chosen/oracle IoU, first-good rank,top10, and selected
   native/reference/final signed center/size errors against root GT.
   Completion requires exactly9508 records and audit hits/counts matching
   evaluator last/bbf025/050. CPU tests run the tracked semantic evaluator
   method and IoU functions in isolation; three seeds match selected boxes,
   IoUs and counters. Altered counter and strict-threshold checks pass.
   Total15 focused tests pass. Full command/limitations:
   `docs/PVG_FORWARD_AUDIT.md`. Remote import preflight passed from isolated
   staging without CUDA initialization; GPU entry point has not run. No real-sample
   bypass result,gate distribution,GT error result or method gain is claimed.

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

Additional primary-source inspection,2026-11-02 approximately16:05UTC:
Mask3D,https://github.com/JonasSchult/Mask3D,
commit11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5,
`trainer/trainer.py`,function`get_mask_and_scores`,retrieved via GitHub API.
It forms a binary foreground from positive mask logits,averages sigmoid
mask probabilities over that foreground,and combines that quality estimate
with per-query class confidence for ranking. This is instance segmentation,
not language grounding; class confidence is not our decomposed semantic score.
No implementation copied or deployed. It motivates measuring agreement
between semantic candidate ranks,member support and actual box IoUs AFTER
the fixed-checkpoint export. It does not justify adding a scoring module,
claiming a causal weakness,or a gain. If oracle coverage is high but semantic
selection fails,inspect chosen-vs-good candidate support confidence before
considering an internal Support calibration; if oracle coverage is low,
reranking alone cannot recover the missing boxes. These are experiment
decision rules,not observed outcomes.

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
