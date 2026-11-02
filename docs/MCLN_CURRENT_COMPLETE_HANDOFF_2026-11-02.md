# Current MCLN + PV-Ground Handoff

21:26UTC A1 semantic log-only subgroup audit:unique1232/1070over1419,
multiple4214/3473over8089,sum5446/4543. Multiplemisses3875/4616,
uniquemisses187/349. OriginalE0 subgroup differences are unmatched
aggregate differences,not paired fixes or established module attribution.
Details/rawloglines:docs/PVG_SCANREFER_STAGE_A_RESTART.md and
docs/results/mcln_pvground_20261101/scanrefer-stage-a-epoch1-group-evidence.json.
ActualScan189152/188533/188537,NR96960/135075,SR92354/129032verifiedlive.
NRfree6907711488bytes(88%used),SR14190972928(74%used);at most8/17
new same-sizeweights fit ignoringlogs/transientgrowth. Historicalcleanup
proposal requires freshselection and explicitapproval;no cleanup done.
No localtests/modelruns/configchanges.

Scan restart A1 completed21:19:22UTC2026-11-02, full9508 last-semantic
Top1 hits5446/4543=57.278082%/47.780816%. Owncheckpoint1/1168tensors/
optimizer/scheduler CPUverified and hardlinked; semantic_best selects1.
Checkpoint601225876bytes/4links,SHA256
9fc7999242e766ec44ea569042eb750ca35662947713c8177bef51badf28e519.
Original E0 delta+11/+216 hits crosses the documented protocol boundary;
not established module attribution. Matchedvalidation remains required.
A1 meets declared5425/4328metricgate, but decision remains pending until
all3Aepochs and successfulpipelineexit. Finaltargetdeficits259/402hits.
At21:20:34UTC A2=42/6081,actual two A2updates finite6.6050129/8.2219801,
all3modules nonzero gradients/deltas, frozen parameters/buffers unchanged.
Immutablestartupreceipt retains A1proof;per-epochupdatefiles nowA2.
NewScanCSV E0official/E0identity/A1 separate from historicaloldScan.
A2ETA23:50-00:05UTC,A3~02:25-02:45UTCNov3,subject to throughput.
NoBlaunch or configurationchange. NRbestalias27 verified21:16:37UTC,
793092691bytes/4links,NR28=271/3742;nextfull~22:10-22:20UTC.
NR27resultpush09ed9bf. No localtests or storagecleanup.
Raw docs/results/mcln_pvground_20261101/scanrefer-stage-a-epoch1-identity.json
and scanrefer-stage-a-complete-epochs-through1.json.

NR27 completed 2026-11-02 21:12:16 UTC, full7899 last-semantic Top1:
3775/2908 hits = 47.790860%/36.814787%. Own CPU checkpoint epoch27,
1134 tensors, optimizer/scheduler verified, CUDA uninitialized; hardlink
runs/nr3d/preserved_checkpoints/ckpt_epoch_27.pth. New observed025-first
best27; versus26 +90/+95 hits, versus25 +48/-37. Target025 deficit949.
Never combine27's025 with25's050. Historical table47rows: oldScan10,
NR27,SR10. Original240epoch recipe unchanged; trainer96960/keeper135075
live at21:12:17 and next epoch starting. Best alias refresh not yet read.
ScanStageA collector21:12 returns no complete rows, pending decision;
no StageB launch. No localtests or cleanup. Raw
docs/results/mcln_pvground_20261101/nr3d-epoch27.json.

21:05 UTC verification: SR semantic_best.json now selects epoch 10 with
11624/9119 hits over 17726 examples; the 793092691-byte best weight has
4 hardlinks. This supersedes the pending alias note below. Original
SR11 is at 301/6486, NR27 at 3680/3742, and Scan restart A1 at 5954/6081.
Scan has no complete new Stage A validation yet. Its first two actual
updates have finite loss, nonzero three-module updates and unchanged
frozen parameters/buffers. SR10 results and raw evidence were pushed
in commit 8427315. No local tests or training changes; cleanup unapproved.

SR10 completed2026-11-02 20:58:39UTC:full17726 last-semantic Top1,
hits11624/9119=65.575990%/51.444206%,new selected025-firstbest10.
Owncheckpoint10/1134tensors/optimizer/scheduler remotelyCPUverified and
hardlinked. Versus9:+522/+487hits;versuspreviousbest8:+500/+533.
Target025deficit678hits;not achieved. Keep original140epochs unchanged.
Historicaltable46rows:oldScan10/NR26/SR10;newScan separate.
Raw docs/results/mcln_pvground_20261101/sr3d-epoch10.json.
At21:01:31UTC actualSR92354/keeper129032 live,SR11=117/6486,
GPU22924MiB. Bestalias still8 on that read,pending keeper refresh;
numeric10 independently preserved. ScanA1=5769/6081,NR27=3394/3742.
Nextfull estimates NR27~21:10-21:15UTC,ScanA1~21:15-21:25UTC,
SR11~23:45-23:55UTC. No localtests,trainingchanges or approvedcleanup.

20:34UTC real storage risk:NRfree7700983808bytes(86%used),SR14984093696.
Each preserved full checkpoint793092691bytes;only9/18additional same-size
weights fit ignoring transient/log growth. Keeping every epoch cannot
support NR240/SR140 on50GiB disks. Read-only exact inventory and proposal:
docs/MCLN_STORAGE_RETENTION_PROPOSAL.md. Keep best+latesttwo and allmetrics
would currently exclude NR17oldweights/SR7oldweights,free13.48/5.55GB
on separate disks. No weights deleted or bulk archived;no recurringcleanup
deployed. Destructive cleanup awaits explicit human approval per supplied
AGENTS.md. Continue live training unchanged and monitor free space.

20:26UTC prepared Stage B now enforces successful full matched validation
of its selected epoch/SHA before importing the GPU trainer. Prior entry
could bypass that documented prerequisite. Eleven remote CPU tests passed;
entry SHA dcb44691123301a9806ae700e51d038896647d6ca33cbda95b65c1a9fc3ce2a6.
See later stage_b_matched_gate_prepared.json,which supersedes the entry
hash in historical stage_b_prepared.json. Live Stage A unchanged.
No matched GPU evaluation or Stage B training has occurred. No local tests.

NR26 completed2026-11-02 20:11:19UTC:7899 last-semantic Top1 examples,
hits3685/2813=46.651475%/35.612103%. Own epoch26/1134tensors/optimizer/
scheduler remotely CPU verified and checkpoint hardlinked. Down42/132hits
versus25;selected025-firstbest remains25,3727/2945=47.183188/37.283200%.
Target025deficit997hits for that best checkpoint. Continue original240
epochs unchanged;no early stop or method failure inferred from one epoch.
Historical current-run table45rows:oldScan10/NR26/SR9;newScan separate.
Raw docs/results/mcln_pvground_20261101/nr3d-epoch26.json.
At20:13:24UTC original trainer96960/keeper135075 live,NR27=129/3742,
GPU22833MiB,next full ETA~21:10-21:15UTC. Current-root-only progress helper:
work/check_nr3d_current_mcln_progress.sh. Do not use stale
work/inspect_nr3d_current_epoch.sh,which points to a stopped20261030 run.

New ScanStageA still awaits complete metrics. StageB source/entry is
prepared separately (5remoteCPUtests) with correct all3module2e-5 rates
and semantic two-decrease stop. Matched official/selected initialization
entry is prepared in protocol_audit (6remoteCPUloadingtests);when StageA
is complete and GPU free,run the matched check before attributing tiny
gains and before launching StageB. No matchedGPUevaluation or StageB
training occurred. OriginalE0/currentgate/live source unchanged.
Details,raw test/preparation evidence and secure helper commands:
docs/PVG_SCANREFER_STAGE_A_RESTART.md. No local tests.

19:16UTC keeper refresh confirmed:NRsemantic_best.json/pth nowepoch25,
3727/2945hits,1134tensors,optimizer/scheduler,793092691bytes/4links.
Previous pendingalias24 note is superseded. Raw manifest:
docs/results/mcln_pvground_20261101/nr3d-semantic-best-epoch25.json.
Original NRmain96960/keeper135075 live,NR26=415/3742,config unchanged.
ScanStageA1=1347/6081,GPU11369MiB;SRmain92354/keeper129032 live,
SR10=2758/6486,GPU22924MiB,nullcheckpoints. No new completeepoch,
44rows unchanged. Next full validation estimatesNR26~20:10-20:15UTC,
SR10~21:00-21:10,Scan1~21:15-21:25. PollnearETA,no localtests.

NR25 complete19:10:21UTC,full7899 semantic Top1 hits3727/2945=
47.183188%/37.283200%,new observed025-firstbest25. Owncheckpoint25,
1134tensors/optimizer/scheduler remoteCPUverified and hardlinked.
44 historicalrows:oldScan10/NR25/SR9. NR target4724hits025,deficit997.
At19:13:36UTC original main96960/keeper135075 alive,NR26=216/3742,
config unchanged;semantic_best alias still24 on that read,pending240sec
keeper cycle. Do not claim25alias before confirming it.
ScanStageA still pending first full epoch;SRlatest9/best8 unchanged.
New tools/collect_scanrefer_stage_a_epochs.py deployed separately,
real initial emptyrows and2remoteCPUfixtures passed,CUDAuninitialized;
receipt docs/results/mcln_pvground_20261101/scanrefer-stage-a-collector-remote-tests.json.
Collector90251d6 and final acceptance policy6181634 pushed;no localtests.

19:00UTC live-process verification: Scan main189152/keeper188533/
preservation188537 alive;keeper waiting_for_complete_validation.
Scan18:58=604/6081,logged cumulative batch600loss6.9913 finite.
NR original main96960/keeper135075 live,epoch25=3308/3742;
SR original main92354/keeper129032 live,epoch10=1991/6486.
NR/SR configs unchanged;no new complete metrics or best selection.
Next full validation estimates:NR25~19:10-19:15UTC,Scan1~21:10UTC,
SR10~21:00UTC. Monitor near ETA with180-300sec intervals.
All targets remain unachieved;no method change justified by partial logs.

VERIFIED18:48UTC: Scan StageA run1793644724 has two actual optimizer
updates at epoch1 batch6/12,finite losses6.477235794067383/7.649016857147217.
All Reader/Support/Span gradients and deltas finite/nonzero;frozen native
parameters and buffers exactly unchanged twice. Full9508 E0 passed,
official+identity5435/4327hits=57.162390%/45.509045%,same input/orderSHA.
Startup immutable evidence:
docs/results/mcln_pvground_20261101/scanrefer-stage-a-startup-receipt.json.
Progress129/6081,GPU10125MiB;first full trained validation ETA21:10UTC.
No complete StageA epoch yet. Source/config recipe and conditionalStageB
in docs/PVG_SCANREFER_STAGE_A_RESTART.md. Native/BNfrozen,batch8/accum6,
moduleLR1e-4,max3epochs. Per-epoch update witnesses may later be overwritten.
NR25/SR10 continue original trainers/source/config/RNG unchanged,verified
18:41:47UTC. NRbest24/SRbest8;historical43rows;new E0 table separate.
No local Windows/WSL tests/model execution. This supersedes pending notes.

Latest user request overrides historical status below: restart ScanRefer
using the supplied plan, while NR/SR continue original full scratch runs.
Details and remote launch/evidence:docs/PVG_SCANREFER_STAGE_A_RESTART.md.
OldScan1793579788 stopped18:06:56UTC November2;best8/latest10/log preserved.
New isolated scanrefer_restart_20261102 pipeline second attempt18:13:26UTC:
matched full9508 official-versus-identity E0 then conditional module-only
StageA,batch8/accum6/moduleLR1e-4,3epochs,native+BNfrozen.
Do not claim real StageA training until own first-update receipt passes.
First E0 attempt failed before eval on training-only loader restriction;
independent E0 initializer now checks1134 native tensors/checkpointepoch54.
Production training loader restriction unchanged. No localtests.

NR24 complete18:10:00UTC7899hits3666/2805=46.410938%/35.510824%;
own checkpoint24/1134 tensors/optimizer/scheduler audited and preserved.
New NR025-firstbest24. SR9 complete18:12:48UTC17726hits11102/8632=
62.631163%/48.696830%;own9/1134 audited,preserved;bestSRremains8.
Combined historical current-run table43rows:oldScan10/NR24/SR9.
Keep newScanE0/StageA rows separate. All targets remain unverified/unmet.
NR/SR trainer/config/source/RNG unchanged;never loadScanweightsinto them.
Two-hour automation now points at restart pipeline;notification preference preserved.

Update17:40--17:44UTC: all three current trainers and metric keepers
verified live17:40UTC,GPU23671/22833/22924MiB;configs unchanged.
Scan e11=957/6081,NR e24=2115/3742,SR e9=5889/6486.
No new complete validation:41 rows,best8/23/8 unchanged.
SR about14min train plus validation;NR about23min plus validation;
next useful complete checks remain18:00--18:10UTC and about18:10.
No extra GPU evaluation,no local tests,no active source/config changes.

New target-hit decomposition from existing Scan1--10 subgroup receipts:
selectede8 unique1234/1058 of1419,multiple4188/3447 of8089.
Targets require5705/4945 total hits,current deficits283/440.
Even perfect unique leaves at least98/79 extra multiple hits necessary.
With unique unchanged,multiple needs4471/3887 hits=55.272592/48.052911%.
Full subgroup table and caveats:docs/PVG_SCANREFER_TRAINING_OBSERVATIONS.md.
Do not mix epoch2's higher multiple050 with8's unique050. This establishes
a necessary subgroup improvement,not a module failure or gain.
Prioritize matched oracle/ranking/mask/Span evidence when GPU available.

Update17:31UTC: Scan e10 full9508 semantic Top1 completed17:26:28UTC,
5366/4461 hits=56.436685%/46.918385%. Remote CPU own epoch10/1168
tensors/optimizer/scheduler verified; hardlink preserved. Best remains8.
Full41 rows Scan10/NR23/SR8; selected8/23/8,all targets unmet.
Log precision fix and raw training observations pushed57e552b.
No active trainer/source/config change,no local tests.
SR trainer92354/keeper129032 verified live17:30:13UTC,e9=5421/6486,
GPU22924MiB/41%,task checkpoints null. About24min train plus validation
remain; next useful full check18:00--18:10UTC. NR24 full check about18:10.
Two-hour ACTIVE monitor updated with41-row evidence,new diagnostic staging,
explicit offline local_rank0,remote-only testing and current limitations.

Historical17:25UTC: Scan trainer154585/keeper178546 verified live17:24:45UTC,
e10 validation992/1189,about2min remaining; no completed e10 claimed yet.
Remote log-only extraction preserves cumulative training means at the
same6000/6081 batch point for epochs1--10. Loss7.3920->7.2505,
box1.2628->1.2302,semantic35.6224->35.1608,SP Dice0.1733->0.1669.
Not exact full-epoch means,not proof of generalization/module benefit.
Raw JSON and interpretation: docs/PVG_SCANREFER_TRAINING_OBSERVATIONS.md.
Static trace confirms corrected masks and mixed boxes reach final criterion;
no disconnected-supervision bug established. Real forward diagnostics remain
missing. Native evaluator None also cannot simply be fixed by calling
export_retrain_metrics: that existing exporter requires source-choice
counters and uses position metrics,not the required last/bbf semantic pair.
The independent keeper remains the current mitigation.

Found actual log precision problem: epoch LR formatting .5f makes1e-6
look like0.00000. Repository format changed to .8g; log-only collector
accepts scientific notation. Display only,no live source/schedule change.
No local tests or additional GPU evaluation.

Update2026-11-02 approximately17:12UTC: NR e23 full7899 semantic Top1
validation completed17:09:23UTC,3623/2643 hits=45.866565%/33.459932%.
CPU checkpoint own epoch23,1134 tensors,optimizer/scheduler verified and
hardlinked. New Acc025-first best e23; do not reuse e22's higher050.
Selected Scan8/NR23/SR8; full table40 rows:9/23/8. Targets still unmet.
At17:07UTC all three current trainers and240s keepers verified live,
GPU23671/22833/22924MiB; configs unchanged,NR/SR still no innovations.
Scan e10 about9min training remaining plus full validation; SR e9 about47min
training remaining plus validation. No spare GPU diagnostic launched.

New saved-state audit: Scan e9 optimizer54729 steps=6081*9;
NR e22=20592=936*22; SR e8=25944=3243*8. Scheduler steps agree.
Measured decay milestones correspond to11/31,151,31/41 completed epochs,
respectively. Pinned official scheduler also subtracts warmup_epoch;
default-1 produces this offset. It is inherited,not a demonstrated new
training bug. NR live config explicitly confirms warmup_epoch=-1/start1.
No active scheduler changed. Raw remote receipts and exact interpretation
are in docs/PVG_OPTIMIZER_RECIPE_AUDIT.md. Empty third NR/SR optimizer
group has zero parameters; its1e-5 LR does not affect any task parameter.
Real trained-model forward distributions and matched9508 bypass results
remain unmeasured; no mechanism change or gain justified by this audit.

Update approximately16:56UTC: fixed a concrete offline invocation error.
Current main_utils parser accepts only --local_rank(default1),and ignores
the launcher's --local-rank=0 via parse_known_args. The documented eval
command omitted the explicit underscore option,so it would select device1
on these single-GPU hosts. Added `--local_rank 0` (see exact
command in docs/PVG_FORWARD_AUDIT.md),matching the existing training fix.
Remote parser-only audit confirmed old=1/corrected=0,CUDA uninitialized;
no model/device selection or training source change. Reproducer:
tools/audit_pvground_eval_rank.sh. Raw receipt is stored with current results.

Update16:51UTC: NR current trainer96960 and keeper135075 verified live;
e23 training2873/3742,GPU22833MiB,task checkpoints still null.
Latest complete e22/selectede21 unchanged. Approximately13min training
remaining plus prior validation duration; next useful full check near17:10UTC.

Inspected two additional primary GitHub implementations for the observed
ScanRefer unique/multiple gap,without executing reference code:
ViL3DRef82a95d9e41ecd9920d678823eda1b366947da204,conditional spatial
attention; ConcreteNet7ab4628f19f0442fefaff45550a32e2f30b82189,
radius-masked candidate fusion and scene-restricted contrastive denominator.
Detailed source paths,mechanisms,differences and conditional experiment
rules are in docs/PVG_MULTIPLE_INSTANCE_REFERENCE_NOTES.md.
Our decoder queries already contain context and MCLN has contrastive
supervision; neither reference proves an absent capability or a model cause.
No reference module/loss has been copied or deployed. First obtain real
chosen/oracle/Support/Span/bypass evidence before mechanism changes.

Update approximately16:43UTC: identified an instrumentation gap, not a
proven model failure: prior offline hooks observed Span but omitted Reader
activation size and Support's actual member-mask corrections.
Added Reader query/residual L2, Support per-query mean/max absolute delta
and fused-foreground flips on observed member slots only. Ragged scene
statistics remain separate. Selected-query evidence joins these to existing
semantic rank, IoU and Span gates. Values are explicitly before_bypass;
separate flags identify whether the corresponding result is applied.
No model architecture, weights, training config or running source changed.

Nine focused fixture tests passed on the remote Scan server,CUDA
uninitialized,Torch2.1.2+cu118,pytest9.2.0. Covered all four bypass settings,
unobserved-slot exclusion/ragged scenes,selected-query association and
tracked evaluator/IoU agreement. This is not a real model/dataset forward.
Training env lacked pytest; test-only dependencies installed into the
isolated observations_v2/test_dependencies directory. No local tests run.
New helpers staged in offline_audits/observations_v2; active trainer/evaluator
imports and new helper paths verified remotely. ArchiveSHA
bf7aba8910d14a57c76634fbf805aa9dbd4e00a29ae752e55e8a3d9cd0c4c568.
Raw remote test/import receipts saved under docs/results/mcln_pvground_20261101.
See docs/PVG_FORWARD_AUDIT.md for the updated remote command and limits.
Real gate/activation/correction distributions and full matched bypass
metrics remain missing; do not choose a mechanism change from fixtures.

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
