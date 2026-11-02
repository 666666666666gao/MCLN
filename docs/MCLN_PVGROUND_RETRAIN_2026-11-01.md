# MCLN + PV-Ground Training: NR3D and SR3D

NR3D e17 full7899 last semantic Top1 completed2026-11-02
10:37:26UTC:3246/2397 hits=41.093809%/30.345613%, new Acc025-first
best. CPU own e17/1134 tensors/optimizer+scheduler confirmed;793092691-
byte hardlink preserved. Complete-log semantic rows for e12-e14/e16
were recovered; their checkpoints are retention-deleted and remain
metric-only in the CSV. SR3D e6 full17726 completed09:52:42UTC:
10923/8133 hits=61.621347%/45.881756%, new Acc025-first best. CPU own
e6/1134 tensors/optimizer+scheduler confirmed and hardlink preserved.
All-epoch table now25 rows. At11:29UTC all jobs active, NR/SR task
checkpoint fields null. Best Scan e2/NR e17/SR e6. No training changes.

NR3D e15 full7899 last semantic Top1 completed2026-11-02
06:33:10UTC:2977/2193 hits=37.688315%/27.763008%, new Acc025-first
best. CPU own e15/1134 tensors/optimizer+scheduler confirmed;
793092691-byte hardlink preserved. SR3D e5 full17726 completed
05:58:52UTC:10671/7894 hits=60.199707%/44.533454%, new Acc025-first
best. CPU own e5/1134 tensors/optimizer+scheduler confirmed and
793092691-byte hardlink preserved. All-epoch table now19 rows.
At07:26UTC all3 screens/GPU active; NR/SR checkpoint fields null.
Best Scan e2/NR e15/SR e5. No training changes or performance claim.

NR3D e11 full7899 last semantic Top1 completed2026-11-02
04:23:27UTC:2708/1912 hits=34.282821%/24.205596%, new Acc025-first
best. CPU own e11/1134 tensors/optimizer+scheduler confirmed;
793092691-byte hardlink at runs/nr3d/preserved_checkpoints/ckpt_epoch_11.pth.
All-epoch table now17 rows. At04:33UTC all3 screens/GPU processes
active, NR/SR task checkpoint fields null. Best Scan e2/NR e11/SR e4.
No training changes or paper-level performance claim.

SR3D e4 full17726-row last semantic Top1 completed2026-11-02
04:07:25UTC:10408/7179 hits=58.716010%/40.499831%, new Acc025-first
best. CPU own e4/1134 model tensors/optimizer+scheduler confirmed.
793092691-byte hardlink at runs/sr3d/preserved_checkpoints/ckpt_epoch_4.pth.
Full table now16 rows, with Scan best e2 and NR best e10. Training unchanged.

ScanRefer run1793579788 epoch2 full9508 evaluation completed at
2026-11-02 04:00:59UTC: last semantic Top1 hits5415/4496,
Acc025/050=56.952040%/47.286496%. CPU own epoch2/1168 tensors/
optimizer+scheduler verified;795607563-byte hardlink at
runs/scanrefer/preserved_checkpoints/ckpt_epoch_2.pth.
Acc050-first best advances to e2; keep this checkpoint's pair together.

NR3D e10 complete at2026-11-02 03:23:08UTC:7899 last semantic Top1
rows, hits2632/1697=33.320674%/21.483732%. New Acc025-first best;
do not replace its Acc050 with epoch9's higher value. CPU checkpoint
epoch10/1134 model tensors/optimizer+scheduler verified;793092691-byte
hardlink preserved at runs/nr3d/preserved_checkpoints/ckpt_epoch_10.pth.
Scan e2/SR e4 still training at this heartbeat's batch check. No source,
configuration, or process changes. Table now14 completed rows.

2026-11-02 02:21UTC: Scan run1793579788 e1 full9508 last semantic
Top1 hits5424/4438=57.046698%/46.676483%. Semantic counts reconstructed
from1419 unique+8089 multiple subgroup ratios and agree with overall
logged0.57047/0.46676. Position and mask rows excluded. CPU own epoch1,
1168 tensors,optimizer/scheduler confirmed;795607563-byte hardlink at
runs/scanrefer/preserved_checkpoints/ckpt_epoch_1.pth. First measured
fine-tuned checkpoint; official initialization comparison not measured.
NR e9 full7899:2559/1730=32.396506%/21.901507%, CPU own epoch9
1134 tensors and optimizer/scheduler verified;793092691-byte hardlink
preserved. Acc025-first best stays NR e8, SR e3. All3 jobs continue.

2026-11-02 01:22UTC update: NR3D epoch8 full7899 last semantic
Top1 hits2563/1642=32.447145%/20.787441%; SR3D epoch3 full17726
hits9521/5959=53.712061%/33.617285%. Both become observed Acc025-first
best. CPU checkpoints each confirm own epoch8/3,1134 model tensors and
optimizer/scheduler. Preserved hard links under
runs/DATASET/preserved_checkpoints/ckpt_epoch_8.pth (NR) and
ckpt_epoch_3.pth (SR), each793092691 bytes. JSON best remains empty;
metrics paired from complete own-epoch logs. ScanRefer continues,
finite logged loss7.4154; first full epoch pending. No source/config changes.

Latest audit at2026-11-02 01:03UTC: NR3D epoch7 full7899-row last
semantic Top1 hits2415/1509=30.573490%/19.103684%; new Acc025-first
best. CPU audit confirms epoch7,1134 tensors,optimizer/scheduler.
Hard link preserved at runs/nr3d/preserved_checkpoints/ckpt_epoch_7.pth
(793092691 bytes). Empty JSON retention best remains a known limitation;
use the same-epoch complete log pair. NR epoch8 and ScanRefer run1793579788
training active; SR epoch3 validating. NR/SR task checkpoints remain null.

## ScanRefer Fine-Tuning Launch

Current run `1793579788` passed its own first-update witness after
loading 48,655 training and 9,508 validation samples: batch1 size8,
finite loss `7.682458877563477`, and finite nonzero gradients plus
parameter deltas for the local visual, Support, and Span modules. The
screen and GPU remain active; full epoch validation is pending.

Current attempt restarted at 2026-11-02 00:26:09 UTC after targeted CPU
tests passed. Real batch update remains pending; earlier attempts below
failed before their first update. No NR3D/SR3D source or jobs changed.

This attempt completed its first real optimizer update before a later
CUDA out-of-memory failure: batch1 had finite loss 7.026811, with finite
nonzero gradients and parameter deltas for the local visual, Support, and
Span modules. The failed run is preserved under
`runs/scanrefer_failed_oom_1793579176`. The current restart keeps the
same checkpoint, learning rates, and model, with ScanRefer batch size
reduced from10 to8 to fit the 24 GiB GPU.

The second attempt (1793578408) successfully loaded 1,134 matching model
tensors and 96 backbone tensors from official epoch54, with fresh
optimizer/scheduler. Its first forward failed because per-scene masks
have different superpoint widths (1,236 vs 957) and were concatenated.
Keep the masks as per-scene lists for mask_extent_reference instead.
Also corrected Span's reference to the computed mask extent (it had
mixed identical native boxes), and its axis decoder input width from
115 to the actual 105 features. All changed tensors are newly introduced
innovation tensors; official MCLN weights retain their shapes.

CPU test passed for batch2, superpoint widths4/6, independent per-scene
extent equivalence, exact identity initialization, and finite nonzero
Support/Span output-weight gradients. Both model files compile.
Deployed SHA256s:
- models/mcln.py: ae991df46dbd15e575a948683dd39526d556a3623937246490548060f5233c7c
- models/pvground_innovations.py: b4f6f8417f37c9450e3f4e1830601009eedd5da72b3da1be6625575ba40ed112

Failed second-attempt logs are retained under
`runs/scanrefer_failed_ragged_masks_1793578408`. Passing CPU tests and
creating the screen do not establish a successful training update.

User authorized starting ScanRefer fine-tuning. Deployment completed
2026-11-02 00:03 UTC (2026-11-01 19:03 America/New_York).
Endpoint: connect.bjb2.seetacloud.com:48073.
Root: `/root/autodl-tmp/mcln_pvground_ft_20261101`.
Screen: `pvg_scanrefer_mcln_finetune`; run `1793577788`.
Log: `runs/scanrefer/train.log`.

The official epoch54 checkpoint SHA256 matches
`a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d`.
Source archive SHA matched the recorded deployment artifact. Restored
official mapping/class-embedding assets; reused the existing PointNet++
extension and Torch2.1.2+cu118 environment. A seeded CUDA matrix-kernel
witness passed on RTX4090 before launch.

Config: `use_pvground_innovations=true` (Reader/Support/Span),
model-only initialization from official epoch54, start_epoch1, fresh
optimizer/scheduler, batch10, accumulation1, LR1e-5/backbone2e-5/module1e-5,
seed0, max100, decay10/30; `frozen=false`, local RoBERTa frozen by model
code. At the first check, datasets were loading and the real first
optimizer update remained unverified. Check `pvground_first_update.json`
for finite loss and nonzero gradients/updates of all three modules.
The monitor now includes ScanRefer; do not equate a screen with an update.

The first attempt exited at checkpoint validation before training:
the official checkpoint contains the obsolete non-trainable RoBERTa
`text_encoder.embeddings.position_ids` buffer, absent from the installed
Transformers model. All 34 missing keys were exactly the newly introduced
Reader/Support/Span tensors; this one legacy buffer was the sole extra key.
Changed only the validator to allow that extra buffer in model-only
initialization. Resume remains strict; unknown keys and shape mismatches
are still rejected. Four CPU load-contract checks passed. Patched validator
SHA256: `45539aa7039ad3920eee441d730693ba9c4dad547bf6dc0ba13cc01dfd769ab5`.

Preserved failed logs/config under
`runs/scanrefer_failed_position_ids_1793577788`. Restarted the requested
fine-tune at 2026-11-02 00:13 UTC in the same screen name. Real checkpoint
loading and first update on the restarted attempt are still to be checked.
No NR3D/SR3D files or processes were changed.

## User Direction

NR3D and SR3D have no dataset-specific MCLN checkpoint. Train those two
datasets with the official MCLN recipes, without loading the ScanRefer
MCLN checkpoint or the existing PV-Ground checkpoints. ScanRefer remains
a separate run initialized from the verified official MCLN ScanRefer
checkpoint with the three PV-Ground innovation modules enabled.

## Initialization

NR3D and SR3D initialize the MCLN model and PointNet++ backbone from the
code's default random initialization. `checkpoint_path` and
`pp_checkpoint` are omitted, so both optimizer and scheduler are fresh.
The local `roberta-base` model is still loaded and frozen as required by
the MCLN implementation.

The official MCLN README marks the generic Group-Free PointNet++ detector
checkpoint optional. It is not present in the prepared data roots, and
the official OneDrive share returned HTTP 403 during retrieval. No
unverified or cross-dataset checkpoint is substituted.

## Training Recipe

Official source:
[MCLN NR3D script](https://github.com/qzp2018/MCLN/blob/main/scripts/train_nr3d_mcln_sp.sh),
[MCLN SR3D script](https://github.com/qzp2018/MCLN/blob/main/scripts/train_sr3d_mcln_sp.sh).

| Setting | NR3D | SR3D |
| --- | ---: | ---: |
| GPUs available | 1 | 1 |
| Official GPUs | 4 | 2 |
| Per-rank batch | 12 | 12 |
| Gradient accumulation | 4 | 2 |
| Effective batch | 48 | 24 |
| Learning rate | 1e-4 | 1e-4 |
| Backbone learning rate | 1e-3 | 1e-3 |
| Weight decay | 0.0005 | 0.0005 |
| Max epochs | 240 | 140 |
| LR decay epochs | 150 | 30, 40 |
| Decoder layers | 6 | 6 |
| Dataset flags | `butd_cls`, `self_attend` | `butd_cls`, `self_attend` |
| Detection/loss flags | `joint_det`, `detect_intermediate`, `use_soft_token_loss`, `use_contrastive_align` | same |

Single-GPU gradient accumulation preserves the official effective batch
size. It is not a claim of bitwise or distributed-BN equivalence to the
multi-GPU official run. RNG seed is the official parser default, 0.
Validation runs every epoch on all 7,899 NR3D or 17,726 SR3D rows.

## Deployment

Remote root on each host:
`/root/autodl-tmp/mcln_pvground_ft_20261101`.

| Dataset | Endpoint | Screen | Server launch time |
| --- | --- | --- | --- |
| NR3D | `connect.nmb1.seetacloud.com:32749` | `pvg_nr3d_mcln_scratch` | 2026-11-02 00:37:24 CST |
| SR3D | `connect.nmb2.seetacloud.com:45914` | `pvg_sr3d_mcln_scratch` | 2026-11-02 00:37:25 CST |

Run logs are written to `runs/nr3d/train.log` and `runs/sr3d/train.log`;
the trainer creates a timestamped subdirectory containing `config.json`,
native logs, checkpoints, and metrics.

At launch verification, both detached screens and their distributed
launcher/trainer processes were present. The logged process arguments
contained no `--checkpoint_path` or `--pp_checkpoint`; GPU memory had
not yet risen, so actual model updates and first-epoch progress remain
unverified pending the next log check.

The uploaded source archive SHA256 is
`2f780ab28bbeebc1463a1612d8095f670d3eff787428bcd1c7e67c7366dac1c9`.
The run script SHA256 is
`d6ef5e3e6b7fe3425762eefec8424c61340d3b1f9fac88a00ed21b3ae1cf8cc7`.

## Next Checks

After initial data preprocessing, verify each generated `config.json`
records `checkpoint_path=null`, `pp_checkpoint=null`, the intended
dataset flags and official hyperparameters. Then confirm finite training
loss, GPU use, and a completed optimizer update before describing either
run as training-verified. Near the first full validation, check exact
row counts and same-checkpoint Acc@0.25/Acc@0.50. Retain/report NR3D and
SR3D best by Acc@0.25, with Acc@0.50 as tie-breaker. Do not claim the
MCLN paper target was reached until a full native validation supports it.

ScanRefer checkpoint upload remains a separate task; its checkpoint must
not be used by either run above.

## Launch Repair and Current State

The initial launch on both one-GPU hosts exited because the parser's
`local_rank` default is 1. The runner now explicitly passes
`--local_rank 0`; the single-process distributed launch remains at
`--nproc_per_node 1`.

Dataset initialization then exposed two repository assets omitted from
the uploaded source archive. The exact `mapping_full2rio27.json` and
`data/class_embeddings3d.npy` files were retrieved from the official
`qzp2018/MCLN` main branch and installed at the paths required by the
official source. Their SHA-256 values are
`bf286abc15663e25c3197a01c7bb90cb20f3478b600a6256ad974bb40c0e239f`
and `97ee6d4a7633229f6fba2580428887f5acf0603081bc06f65b5d8a0baf6fe5e5`.
Failed-attempt logs were retained in dataset-specific
`runs/*_failed_*_20261102` directories.

The current NR3D run is in screen `pvg_nr3d_mcln_scratch`, with config
`runs/nr3d/nr3d/mcln_pvground_nr3d_20261101/1793553113/config.json`.
The current SR3D run is in screen `pvg_sr3d_mcln_scratch`, with config
`runs/sr3d/sr3d/mcln_pvground_sr3d_20261101/1793552465/config.json`.
Both configs have null `checkpoint_path` and `pp_checkpoint` and match the
official dataset recipes. The full train/validation datasets loaded:
44,909/7,899 NR3D rows and 77,836/17,726 SR3D rows. At the latest check,
both trainer processes were active and GPU memory was about 21.8 GiB on
NR3D and 22.6 GiB on SR3D. The runner logs every 500 batches, so finite
loss and optimizer-update evidence is still pending; GPU use alone is
not treated as proof of a completed update.

## First Full Validation

At the 2026-11-02 02:24 (+08) remote check, NR3D had completed epoch1
training and full validation, then continued into epoch2. The saved
`ckpt_epoch_last.pth` is 793,092,691 bytes; a CPU load confirms epoch1,
1,134 model tensors, and optimizer/scheduler state. The complete 7,899-row
log evaluation reports semantic-alignment Top-1 Acc@0.25=0.08229 and
Acc@0.50=0.00114. The distinct position-alignment branch reports
0.06494/0.00114 and must not be substituted for the semantic metric.
Finite loss-component values were logged during training, with GPU
activity and a completed trained checkpoint.

No `eval_metrics_epoch_1.json` was emitted and the retention manifest has
`latest_epoch=1`, empty `records`, and empty `best`. The deployed
evaluator returns a metrics dictionary only when source-choice selector
or MoE is enabled; both are disabled in these official-recipe runs.
Consequently, the full metrics are present in the training log but
machine-readable receipt and metric-best retention are unavailable.
Do not restart or alter the active process for this; at later validations
pair the full logged row with a CPU-audited checkpoint of the same epoch
and maintain best selection from semantic-alignment Acc@0.25, with
Acc@0.50 as tie-breaker.

SR3D remained active with its official configuration. The sampled
progress was about 3,500/6,486 effective training batches for epoch1;
no full validation result was available at that check. No active run
settings or processes were changed.

## NR3D Epoch 2

At the 2026-11-02 03:18 (+08) evaluation, NR3D epoch2 completed all
7,899 validation rows. Semantic-alignment Top-1 Acc@0.25/0.50 is
0.13837/0.02114; the separate position-alignment branch is
0.11204/0.02000. The CPU-audited `ckpt_epoch_2.pth` reports epoch2,
1,134 model tensors, and optimizer/scheduler state. This improves over
epoch1 on both semantic thresholds.

The retention code deleted the epoch1 checkpoint when epoch2 was saved,
before validation, because metric-return gating left its manifest empty.
The epoch1 metric remains in the log but its weights are no longer
available. Epoch2 is now hard-linked to
`runs/nr3d/preserved_checkpoints/ckpt_epoch_2.pth`; future completed
checkpoints will be preserved after their full validation and ranked
from the semantic-alignment log values. This does not alter the live
training process.

At the 03:30 (+08) sample, SR3D had finished epoch1 training and was
still validating all 17,726 rows; no final metric lines had appeared.
Both active runs retain the intended null task-checkpoint configuration.

## SR3D Epoch 1

SR3D epoch1 completed validation at 2026-11-02 03:38 (+08) on all
17,726 rows. Semantic-alignment Top-1 Acc@0.25/0.50=0.2440483/0.0467675
(4,326/17,726 and 829/17,726). The separate position-alignment branch
is 0.17094/0.03154. CPU audit confirms `ckpt_epoch_last.pth` reports
epoch1 and contains 1,134 model tensors plus optimizer/scheduler state.
The checkpoint is hard-linked at
`runs/sr3d/preserved_checkpoints/ckpt_epoch_1.pth`.

The first-evaluation metrics are logged, but the evaluator-return issue
also leaves SR3D's JSON receipt/retention best empty. Continue selecting
by semantic Acc@0.25, with Acc@0.50 as tie-breaker, after pairing each
full logged evaluation with its own CPU-audited, preserved checkpoint.
At 03:39 (+08), NR3D was training epoch3 and SR3D had entered epoch2;
both remain active on the requested official configurations.

## NR3D Epoch 3 Update

At 04:18 (+08), NR3D completed the full 7,899-row validation.
`last_ semantic alignment` Top-1 Acc@0.25/0.50 was 0.23864/0.08571,
or 1,885/7,899 and 677/7,899 (23.863780%/8.570705%). Position
alignment is a separate branch and is not substituted for semantic
alignment.

CPU audit confirms the saved checkpoint is epoch3 with 1,134 model
tensors and optimizer/scheduler state. It is preserved at
`runs/nr3d/preserved_checkpoints/ckpt_epoch_3.pth`. NR3D has continued
into epoch4. The current best observed semantic Acc@0.25-first row is
epoch3.

At 04:20 (+08), SR3D was still training epoch2 at 2,017/6,486 effective
batches. No new complete validation was available; epoch1 remains its
latest audited checkpoint. The full available metric table is maintained
in `outputs/MCLN-PV-Ground-2026-11-01-all-epochs.csv`. NR3D epoch1's
metric row is log-only because automatic retention deleted its checkpoint
before it could be preserved. No training settings or processes were
changed, and these early scratch-run results are not claimed as matching
the MCLN paper.

## NR3D Epoch 4 Update

At 05:19 (+08), NR3D completed the full 7,899-row validation.
`last_ semantic alignment` Top-1 Acc@0.25/0.50 was 0.26915/0.11837,
or 2,126/7,899 and 935/7,899 (26.914799%/11.836941%). Position
alignment remains a separate metric branch.

CPU audit confirms the saved checkpoint is epoch4 with 1,134 model
tensors and optimizer/scheduler state. It is preserved at
`runs/nr3d/preserved_checkpoints/ckpt_epoch_4.pth`. This is the current
best observed NR3D checkpoint under semantic Acc@0.25-first ranking;
NR3D has proceeded to epoch5. SR3D remains in epoch2 without a new full
validation as of 05:19 (+08). Both configs still have null
`checkpoint_path` and `pp_checkpoint`.

The all-available-epochs table is at
`outputs/MCLN-PV-Ground-2026-11-01-all-epochs.csv`. NR3D epoch1 remains
log-only because retention removed its checkpoint before it was
preserved. No training settings or processes were changed, and these
early scratch-run metrics are not claimed as matching the MCLN paper.

## NR3D Epoch 5 and SR3D Epoch 2 Update

Full validation finished at 2026-11-02 06:19:53 (+08) for NR3D epoch5
and 06:17:46 (+08) for SR3D epoch2. Matching `last_ semantic alignment`
Top-1 Acc@0.25/0.50:

| Dataset | Epoch | Validation rows | Hits025 / Hits050 | Acc025 / Acc050 |
| --- | ---: | ---: | ---: | ---: |
| NR3D | 5 | 7,899 | 2,268 / 1,230 | 28.712495% / 15.571591% |
| SR3D | 2 | 17,726 | 6,996 / 3,330 | 39.467449% / 18.785964% |

CPU audits confirm epoch5/epoch2 respectively, 1,134 model tensors,
optimizer/scheduler state, and 793,092,691 bytes per checkpoint. Hard
links preserve these at `runs/nr3d/preserved_checkpoints/ckpt_epoch_5.pth`
and `runs/sr3d/preserved_checkpoints/ckpt_epoch_2.pth`. Both are the
current observed best by semantic Acc025, with Acc050 breaking ties.
Metrics are paired from their own completed log evaluations; metric JSON
receipts/automatic best records remain absent. Do not mix position/mask
branches into these pairs.

At 06:21 (+08), both screens/GPUs remained active and both task-checkpoint
fields remained null. No training configuration or process was changed.
Updated the seven-row all-available-epochs CSV. NR3D epoch1 remains
log-only; these early scratch results do not establish paper reproduction.

## NR3D Epoch 6 Update

At 2026-11-02 07:20:10 (+08), all 7,899 validation rows completed.
Last semantic Top1 Acc025/Acc050=28.940372%/14.685403%, or 2,286/1,160
hits. Acc025 increased 0.227877 percentage points from epoch5 while
Acc050 decreased 0.886188 points. Epoch6 is the observed Acc025-first
best; report its own pair, never epoch5's larger Acc050 with epoch6.

CPU audit confirms checkpoint epoch6, 1,134 model tensors and
optimizer/scheduler state. Preserved by hard link at
`runs/nr3d/preserved_checkpoints/ckpt_epoch_6.pth` (793,092,691 bytes).
Task-checkpoint fields remain null and the run continues unchanged.
SR3D latest complete remains epoch2 at the 07:16 (+08) check. Updated
the eight-row full-available-epochs CSV and desktop handoff. No paper
reproduction claim; missing metric JSON receipts remain unresolved.
