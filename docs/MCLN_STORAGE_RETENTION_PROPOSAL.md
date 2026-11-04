# Current Checkpoint Storage Proposal

## Current Override: 2026-11-04 04:03 UTC

NR3D watcher/collector verified best49 and latest57 after epoch57 full
validation; both own-epoch CPU checkpoints have1134 tensors and nonempty
optimizer/scheduler. Epoch57 SHA
da6ab7a97bb4cd63d23479e50701251852526cf27695c2bb9d48a378ba3555ff.
The initial collector coincided with checkpoint promotion and asserted;
retry after transition passed. No manual deletion or watcher restart.
See `nr3d-current-full-history-20261104T0401.json`.

## Current Override: 2026-11-04 03:10 UTC

SR3D watcher applied at03:08:49UTC, retaining best15/latest21 and removing
only preserved20. The CPU collector verified both own epochs,1134 tensors,
nonempty optimizer/scheduler and SHA; epoch21 SHA is
1114d6f8d0e9a087fd2231692c54b194c0470736da170558cae9cb9407cc22c1.
Historical metrics and receipts remain; no manual deletion or extra watcher.
See `sr3d-current-full-history-20261104T0309.json`.

## Current Override: 2026-11-04 03:00 UTC

NR3D watcher applied at02:57:18UTC: best49 and latest56 were CPU audited
and retained; only preserved55 was removed. Epoch56 own epoch,1134 tensors,
nonempty optimizer/scheduler and SHA
b5b62f509cf9776ba4d3b87ea52fc448f7951edce194243dac6dcd740e19975f
were independently verified. Historical metrics and receipts remain.
See `nr3d-current-full-history-20261104T0259.json`.
SR21 validation is still in progress; its last audited kept set is15/20.
No manual deletion or additional watcher was used.

## Current Override: 2026-11-04 02:00 UTC

NR3D retention audited current best49 and latest55 at01:57:07UTC; both
own-epoch CPU checkpoints have1134 model tensors and nonempty optimizer/
scheduler state. SHA: best49
0987ce9208dca84686a45a7d80bad4f93f126f34be9c8303899a1b6bf68ed66b;
latest55 721e834de0e1cd7605e66e06fd5a68fe44cc54926b0b91299fc595df2159873d.
The watcher kept49/55 and removed only preserved54. Its receipt is in
`nr3d-current-full-history-20261104T0156.json`.

SR3D remains best15/latest20, both audited in the01:58UTC full-history
receipt. Existing best/latest watchers remain the only retention authority;
no manual deletion, relinking or watcher restart was performed.

## Current Override: 2026-11-04 01:12 UTC

Current-run health at01:11UTC:NR3D run1793553113 trainer96960/keeper
135075/retention156800 live,epoch55=1139/3742;69 recent losses finite,
last three24.3912/9.946/9.9954 (isolated higher finite value,insufficient
to diagnose instability),no logged errors. SR3D run1793552465 trainer92354/keeper
129032/retention148725 live,epoch21=1914/6486;64 recent losses finite,
latest9.6869,no logged errors. Both retain their original null-init
scratch configuration and existing best/latest watcher. No new complete
validation or retention event was observed;do not infer a new checkpoint
or alter retained weights. See the canonical handoff for ETA and latest
complete metrics.

## Current Override: 2026-11-04 01:00 UTC

NR3D epoch54 completed full7899 last semantic Top1 validation at
4004/3233 hits (50.689961%/40.929232%). Best remains epoch49=4123/3325;
epoch54 is -119/-92 hits versus best and720 Acc@0.25 hits below the
4724-hit target. The CPU audit verified own epoch54,1134 model tensors,
nonempty optimizer/scheduler and SHA256
497ac38f13ddc83612953d019891a9c4d7100d6f1ec91acea2d5cfa166551b8e.
The existing watcher applied at00:56:55UTC,kept49/54 and removed only
preserved epoch53;its audit remains in the full-history receipt and the
epoch53 metric remains marked with a removed-weight status.

At00:59UTC NR trainer96960/keeper135075/retention156800 were live at
358/3742 for epoch55,with unchanged original null-init scratch config,
finite recent losses and no logged errors. Next full result is estimated
around01:45-02:00UTC. SR remains best15=11743/9434,latest20=11445/9060;
its00:53UTC progress was1123/6486 on the unchanged config,with finite
recent losses and no errors; next full result is estimated around
02:55-03:20UTC. Existing watchers remain the sole retention authority.
Receipt:
`docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261104T0058.json`
and`docs/results/mcln_pvground_20261101/sr3d-current-full-history-20261104T0036.json`.

## Prior Override: 2026-11-04 00:00 UTC

At00:35UTC the collector confirmed SR3D epoch20 completed full17726
semantic Top1 validation:
11445/9060 hits (Acc@0.25/0.5=64.566174%/51.111362%). Best remains
epoch15=11743/9434; latest20 is below best by298/374hits and is857
Acc@0.25 hits short of target12302. Independent collector verified the
own-epoch20 CPU checkpoint,1134 model tensors,nonempty optimizer/scheduler,
SHA256 b1be7c65b6dd726b0ae94b052a9f19f8b5378d3e8ad9b6fb4bcd822bfa329b82,
preserved path and retention latest20. Best15 SHA remains
386647b8ecac817614090cb23f11b5d7a4d36b7449f67c07f43f84baf23e3969.
The existing watcher applied at00:32UTC,kept15/20 and removed only the
older preserved epoch19 (793096192 allocated bytes freed);the full audit
is retained in the receipt. Epoch19's metric remains,but its weight is
marked removed in the all-epochs table.
At00:36UTC the original SR trainer/keeper/retention processes were live
at379/6486 on the unchanged scratch recipe; current progress had finite
recent losses and no logged errors. No manual checkpoint deletion.

NR3D latest53=4107/3258 and best49=4123/3325 over7899. Full CPU audit
and preservation sync are confirmed in the23:58 receipt; epoch53 SHA
e8d6d69f9af514f3f5d51973c64642d3f7a992eb362723527d4eda60a5dba991.
Existing watcher retained best49/latest53; no manual retention action.
At00:36UTC NR trainer96960/keeper135075/retention156800 were live at
2906/3742 on the unchanged scratch recipe; recent losses were finite and
no errors were logged. Its next full validation is expected around
01:00-01:15UTC, subject to validation time and actual speed. Its watcher
remains the only cleanup authority.
Receipt:
`docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261103T2358.json`
and`docs/results/mcln_pvground_20261101/sr3d-current-full-history-20261104T0036.json`.

## Current Override: 2026-11-03 23:09 UTC

NR3D best49/latest52 remain verified by the22:56CPU receipt; no newer
full validation or retention event. SR3D's23:08 live check confirms the
trainer and log are advancing, but reports no new complete validation or
retention event. Existing best/latest watchers remain authoritative; do
not manually remove or relink checkpoints.
Progress receipt:
`docs/results/mcln_pvground_20261101/sr3d-current-scratch-progress-20261103T2308.json`.

## Current Override: 2026-11-03 22:56 UTC

NR3D run1793553113 remains live after full epoch52 validation. Current
best49=4123/3325 and latest52=4048/3238 over7899. CPU audit verified
semantic_best.pth and ckpt_epoch_last.pth own epochs,1134 model tensors,
nonempty optimizer/scheduler and SHA; epoch52 SHA
a94838e87ddace0e0be872e6c8ac7216a9ad92dc3692b5a76ed5db59e6954ff9.
The 22:56 full-history receipt records current retention state. Do not
delete either retained checkpoint; existing watcher owns retention.

SR3D remains at audited best15=11743/9434 and latest19=11426/9256 over
17726; the22:56 full-history audit confirms both files and their own-epoch
state. No new validation or retention event was observed. Continue using
the existing watcher only. Receipts:
`docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261103T2255.json`
and `sr3d-current-full-history-20261103T2255.json`.

## Current Override: 2026-11-03 22:02 UTC

Fresh current-run audits show NR best49/latest51 and SR best15/latest19.
Both were CPU-verified from the trainer's semantic-best/latest aliases and
their dataset-level preserved links: own epoch,1134 model tensors,
nonempty optimizer/scheduler, SHA256. NR49:
0987ce9208dca84686a45a7d80bad4f93f126f34be9c8303899a1b6bf68ed66b;
NR51: b1ff22ea55d9cc26dd21b81e7f5221350ccae1ee31de346834d3ce0892aa8841.
SR15: 386647b8ecac817614090cb23f11b5d7a4d36b7449f67c07f43f84baf23e3969;
SR19: 7a179f1a0c49c30b72a28ef2ef1ae1e951ac0c2e7cf94bad6581621c1940338c.
The watchers removed NR48/50 and SR18 under the authorized policy;
the latest full-history receipt records those removals and retains all
metrics/audits. No manual cleanup or duplicate watcher was started.

Current complete results: NR49=4123/3325 and NR51=4100/3295 over7899;
SR15=11743/9434 and SR19=11426/9256 over17726. These are same-checkpoint
last semantic alignment Top1 hits. NR/SR goals remain unmet. See
nr3d-current-full-history-20261103T2202.json,
sr3d-current-full-history-20261103T2202.json and the combined all-epochs CSV.
The scoped collector uses the actual retention directory at
runs/DATASET/preserved_checkpoints and audits semantic_best.pth plus
ckpt_epoch_last.pth. Historical run directories remain out of scope.

## Authorized Policy Applied: 2026-11-03

Update19:24UTC:SR full18=11563/9150 over17726;best remains15=
11743/9434. Remote CPU own15/18/1134tensors/nonempty optimizer/scheduler/
SHA verified,CUDAoff. Latest18 SHA
9505e32be485ef25c9fabca4f37cd5bbcc442bca78a49d6c5ae6367b4e8e23ca.
Retainer19:15:37UTC kept15/18 and removed only preserved17;
free19739729920->20532826112bytes. Receipt:
sr3d-current-full-history-20261103T1920.json. The first collector met a
save/keeper synchronization window;an unchanged retry after180s passed.
No fallback,manual link,new watcher or trainer patch was added.
NR last audited kept47;endpoint failures do not establish current file
presence or trainer status. CompletedScanB2/B4 and official/A3 remain.
The separate pilot control own1 is audited,795607435bytes/4links,SHA
4f18e127321107f9f709c876226437bbc871ec3e5cd101cbf992ceede172f8d8.
Treatment is training;no complete treatment checkpoint is claimed.
HistoricalCSV75rows,restart11,pilot1. Removed17 is marked
removed_authorized_best_latest_20261103;historical audits remain.

Update18:01UTC:NR47 full7899=4114/3248,best/latest47. OwnCPU47/
1134/nonempty optimizer/scheduler/SHA audited,CUDAoff:
a89e77471e8a5461838f60648651be891baaa658cb79a0f5d446fd54b8f673c3.
Retainer17:55:38UTC kept47 and removed only preserved46,after its own
CPU audit;free21177679872->21754355712bytes. Receipt:
nr3d-current-full-history-20261103T1759.json. Initial collection found
preserved47 missing while keeper/source47 existed;180s wait resolved
the window without code/trainer changes or a new watcher/cleanup.
Kept sets at this snapshot:NR47,SR15/17,completedScanB2/B4;
pilot has no complete checkpoint yet. Tables74/11rows,46path marked
removed_authorized_best_latest_20261103;historical audit remains.

Update17:14UTC:NR best/latest46 CPU-audited,1134/nonempty optimizer/
scheduler/SHA;SR best15/latest17 similarly audited. Retained sets now
NR46,SR15/17,ScanB2/B4. NR46 SHA
05370b318703e972d1ba18afdf81495f51b99daad8f8db417f77a2764a14dbf8.
SR17 SHA71f9c5fa2d828a54a91f2055d3405762e87e32b7ac8955a1093a4060aa39da00.
SR retention16:39:15UTC removed16,free19738275840->20531372032bytes.
Current receipts:nr3d-current-full-history-20261103T1704.json and
sr3d-current-full-history-20261103T1642.json. Tables73/11rows.
No new watcher/cleanup;all original NR/SR recipes/budgets unchanged.

Update15:49UTC:Scan StageB has completed by normal early stop,exit0.
Current kept sets:NR44(best)/45(latest),SR15(best)/16(latest),
ScanB2(best)/B4(latest). All kept own epochs/model/optimizer/scheduler/
SHA are remotelyCPU-audited;CUDAoff. ScanGPUfree,trainerexited.
NR retention14:47:11UTC removed43;15:47:21UTC kept44/45,removednone.
Scan retention15:07:20UTC audited2/4 and removed onlyolder3;
free11485032448->12280643584bytes. B1/B3 paths are marked deleted.
MainCSV71rows,Scanrestart11rows. All logs/configs/metrics/audits and
official/A3 inputs remain. No duplicate watcher or cleanup was launched.
The Scan retention watcher is idle,not a trainer;leave the completed run
archived rather than restarting or polling it in future monitoring.
FinalScan receipt:scanrefer-stage-b-final-full-history-20261103T1544.json.
NR receipt:nr3d-current-full-history-20261103T1547.json.
SR receipt:sr3d-current-full-history-20261103T1540.json.

Update14:05UTC:SR16 is complete11534/9328 over17726 and belowbest15.
Current SR keep set15(best)/16(latest),both independentlyCPU-audited/
1134tensors/nonemptyoptimizer/scheduler/SHA;retention latest_epoch16.
SR16 SHA d0278332a96079086f65024f22b74350da31b1ff871bf2d18fa6ddaa6746824a.
Rawreceipt:sr3d-current-full-history-20261103T1404.json.
MainCSV69rows;all prior full results/audit histories retained.
No duplicate watcher or additional cleanup was launched.

Update13:50UTC: current audited kept sets are NR43(best/latest),
SR15(best/latest),Scan B2(best)/B3(latest).
NR retention13:47:02UTC verified43 and removed only older41/42;
free20385718272->21971910656bytes.
SR retention11:14:32UTC verified15 and removed only superseded10/14;
free19739611136->21325803520bytes.
Scan retention12:07:13UTC verified2/3 and removednone;
free12283387904bytes. All kept weights have own-epoch/model/optimizer/
scheduler/SHA evidence. Full histories include every retention receipt.
No duplicate watcher or one-time cleanup was started.

Current raw receipts:
nr3d-current-full-history-20261103T1349.json,
sr3d-current-full-history-20261103T1345.json,
scanrefer-stage-b-full-history-20261103T1345.json.
Historical CSV68rows and restart CSV10rows include all completed epochs.
Superseded37,SR10/14 and B1 paths are now marked deleted;new NR38-42
CPU audit fields reflect retained historical audits,not current file presence.
Official/A3 initialization inputs and all logs/configs/metrics/audits remain.

Update09:40UTC: SR14full17726=11561/9002hits,65.220580/50.784159%.
025-firstbest10/latest14;14is63hitsbelowbest10. OwnCPU14/1134/
nonemptyoptimizer/scheduler verified,CUDAoff;793092691bytes,
SHA1c8759ef21a5ee0de5cb04fc817b81df98aad3d9d814b379300be997ba8424be.
Retention08:18:13UTC auditedbest10/latest14/SHA/resumestates,removedonly13.
Free19739271168->20532367360bytes. Both weights were retained then.
Receipt:sr3d-best-latest-retention-epoch-14.json. Fullhistory61rows,
CSV13 path marked removed;logs/configs/audits remain.

Update07:42UTC: NR37full7899=3994/3152hits,50.563362/39.903785%.
025-firstbest/latest37. OwnCPU37/1134/optimizer/scheduler verified;
bestmanifest37/SHA independentlyconfirmed07:34:15UTC.
Retention07:34:08UTC auditednonemptyoptimizer/scheduler/SHA,removedonly36,
kept37. Free21180428288->21973524480bytes;4hardlinks07:40:07UTC.
SHA8cd1a07bb9ea191a3a243864bdc71a0a6c28f2665f970d777f25e61d0494bb3b.
Raw receipt:nr3d-best-latest-retention-epoch37.json.
Logs/configs/60historicalrows/allrawauditsretained;CSV36pathmarkeddeleted.

Update06:35UTC: NR36full7899=3953/3101hits,50.044309/39.258134%.
025-firstbest/latest36. Keepermanifest36/SHAverified06:34:17UTC.
Retention06:34:00UTC independentlyaudited36/1134/nonemptyoptimizer/
scheduler/SHA,thenremovedonly34/35. Free20387758080->21973950464bytes.
SHA13d3373c28f10271872982a105cbb7099be6f7b3a9c0acb1c143ab67a31220ba.
At06:34:55UTC only36remainsinpreservedwith4hardlinks.
Alllogs/configs/59historicalrows/auditreceiptsremain.

Update06:06UTC: ScanB1 full9508 audited5474/4653hits. Best/latestboth1.
Retention06:02:53UTC audited own1/1168/nonemptyoptimizer/scheduler/SHA,
removednone,free13081960448bytes. Weight795607307bytes,
SHA cda46832c35f95fb70710da23768ea28eb27f8130bb6ae787de2665f0cf92f48.
Receipt lives in the B run directory,not the restart root; the watcher
log lives at the restart root. A3 and official initialization inputs remain.

Update05:24UTC: current complete results are NR35 and SR13.
NR keepbest34/latest35; SRkeepbest10/latest13. SRretention05:21:51UTC
independently audited10/13includingnonemptyoptimizer/scheduler/SHA and
removedonlythe old12preservedweight. Free19739373568->20532469760bytes.
NR35 isCPU-audited/hashed and its automatic retention receipt is collected
separately. All metric/log/config/audit records are preserved.
NRreceipt05:21:52UTC auditsbest34/latest35,nonemptyoptimizer/scheduler/SHA,
removednone,free21181136896bytes.

Update04:23UTC: NR34 full validation completed at3891/3146 hits
(49.259400%/39.827826%). At04:26:45UTC best manifest34 was verified;
best and latest are34. Epoch34 is CPU-audited and hashed
b6964cd073ae582b88eabc57de28173dbe99b64a42b688583684d7ff5099ce20.
The watcher waited for checkpoint-save/metric-best manifest synchronization;
it must not delete until that gate passes.

At04:25:43UTC the watcher audited34/1134/nonemptyoptimizer/scheduler/SHA,
then removed the exact older32/33 preserved files. Only34 remains,
with4hardlinks at04:27:27UTC. Free bytes20388360192->21974556672.
Raw receipt:nr3d-best-latest-retention-epoch34.json.

Update03:20UTC: NR33 completed and its own CPU checkpoint was audited.
Current NR retention set is best32 plus latest33. File33 SHA256
964c70a97f3b8895e05c200c25bbd4b7a8f041eae9ad19bcaced2bd4af63f967,
793092691bytes/3links; free21181689856bytes at03:19:49UTC.
The initial deletion table below records the earlier32-only state.

The user explicitly authorized retaining only the highest-metric checkpoint
and the latest checkpoint for resume. This supersedes every pending-approval
statement and latest-two proposal below.

At 02:49 UTC, remote CPU audits verified the kept checkpoints' own epochs,
model tensor counts, nonempty optimizer state, scheduler state and SHA256.
Then 24 older NR weights and 10 older SR weights were deleted only from the
current runs' `preserved_checkpoints` directories. No stopped run was inspected.

| Dataset | Best | Latest | Free Before | Free After |
| --- | ---: | ---: | ---: | ---: |
| NR3D | 32 | 32 | 2,940,702,720 | 21,975,019,520 |
| SR3D | 10 | 12 | 12,601,610,240 | 20,532,576,256 |

All logs, configurations, validation rows and identity receipts remain.
The 54-row historical CSV marks deleted weights
`removed_authorized_best_latest_20261103`; historical CPU audits remain valid
but do not imply the corresponding weights still exist.

Independent `pvg_nr3d_best_latest_retention` and
`pvg_sr3d_best_latest_retention` screens check every 240 seconds.
`tools/retain_mcln_best_latest.py` waits for complete validation, checkpoint
save and the correct semantic-best manifest before deleting older preserved
weights. It does not modify the trainers, model configuration or metric keeper.
Raw initial deletion receipts include every removed path and kept SHA.
Remote CPU fixture checks passed, including ScanRefer Acc050-first ordering;
no local tests ran.

ScanRefer Stage B's own first update passed: finite loss7.021632194519043,
all three modules have finite nonzero gradients and parameter updates.
At03:02:11UTC, A1/A2 were deleted after startup completed; A3 remains the
warmup best/latest and B initialization input. Free space increased from
12,677,091,328 to13,879,545,856bytes. At03:03:25UTC the actual Scan
retention Python PID209444 was live, waiting for the first complete B
validation. Its policy is Acc050-first best plus latest, with Acc025 as
tie-breaker. Official pretrained input weights remain intact.

The live NR/SR watchers loaded the initial implementation, which deletes
older preserved weights; their native trainer already retains latest only.
The Scan watcher loaded the updated implementation covering both numbered
run-directory weights and preserved links. Uploading a new source file
does not change already-running Python processes.

Update2026-11-03 02:20:04UTC:NRfree2940784640bytes,95%used.
At most3additional793092691-byteweights fit ignoringothergrowth.
Actualtrainer96960/keeper135075live;best/latest32. No cleanupconsent
orweightdeletion. Recompute then-currentbest/latest/pathinventory after
approval;all historical keep-lists below are stale.

Update2026-11-03 01:19:19UTC:NRfree3734265856bytes,94%used.
At most4additional793092691-byteweights fit ignoringlogs/transientwrites.
Currentbest/latest31 preservedweight hasfourlinks,SHA256
e9d220e4b7a0917e59702d3c5021ab66da288e494a9ca0e1d5d873edf3be41cb.
Trainer96960/keeper135075live. No cleanupapproval/deletion.
Historicalkeep29/30listsbelowarestale;recompute best/latest/paths after
approval. Currentbest-plus-latest-two wouldkeep30/31,subjecttofreshinventory.

Update2026-11-03 00:25:30UTC:NR free4527525888bytes,92%used,
at most5additional793092691-byte weights ignoringlogs/temporarywrites.
Current trainer96960/keeper135075 are live. Best/latest30 sharesinode
15040694278 with last/semanticbest/preserved aliases,4links;23distinct
preservedepochs remain. This supersedes previous free-space snapshots.
No cleanup consent or deletion. The best-plus-latest-two proposal would
currently keep29/30;recompute again after approval,not from this snapshot.
All-epoch metric rows/logs/configs/audits remain required regardless of
weight retention. Raw snapshot:
`docs/results/mcln_pvground_20261101/nr3d-storage-20261103T002530Z.json`.

Update2026-11-02 21:25-21:26UTC:current-run metadata and actual trainer/
keeper processes checked again. NRfree6907711488bytes(88%used),SRfree
14190972928bytes(74%used). At793092691bytes per new distinct weight,
only8NR/17SR additional weights fit before other growth and temporary
writes. This is not an exact failure ETA. NRbest/latest27,SRbest/latest10;
the older keep-lists below are historical,not actionable deletion lists.
No cleanup consent has arrived and no files were removed. Recompute
current best/latest and resolved target paths after explicit approval.

Read-only remote inventory at2026-11-02 20:34UTC. No files were deleted,
moved,or downloaded as model weights. No local model tests were run.
Actual current trainers and metric keepers were verified live.

| Dataset | Free bytes | One checkpoint | Additional same-size weights fitting | Current preserved weights |
| --- | ---: | ---: | ---: | ---: |
| NR3D | 7700983808 | 793092691 | 9 | 19 |
| SR3D | 14984093696 | 793092691 | 18 | 9 |

The fit counts ignore future logs,temporary checkpoint writes,and other
disk growth;they are capacity upper bounds,not a guaranteed remaining
epoch count or an exact failure ETA. Keeping every subsequent full
checkpoint is incompatible with NR240/SR140 budgets on these50GiB disks.
Hard links protect weights from native retention,but each newly preserved
epoch still consumes another distinct checkpoint's storage.

The proposed bounded policy retains the semantic Acc025-first best plus
the latest two epochs,and all epoch metrics,configs,logs and audit receipts.
At this snapshot best overlaps the latest two:

- NR:keep25/26;candidate older epochs2-11,15,17,20-24,17files.
- SR:keep8/9;candidate older epochs1-7,7files.

All candidate paths are inside the current dataset's
`runs/DATASET/preserved_checkpoints` directory. Every candidate had one
hard link at inventory time. Removing the proposed candidates would free
13482635264bytes on NR and5551673344bytes on SR,using allocated block
counts. These are separate disks and must not be added as one free pool.
Best/latest aliases are excluded. Do not inspect or delete stopped
clean-baseline or older experiment directories for this proposal.

This is a proposal,not authorization to delete. The user's supplied
AGENTS.md instructions require asking before destructive commands.
No cleanup process or recurring deletion has been deployed.
Before an approved cleanup,recompute the selection against the then-current
best/latest and verify actual target paths;the snapshot can become stale.
Record removed epochs and mark their weights unavailable in current
result records,without removing their validation rows.

The read-only tool `tools/audit_mcln_storage.py` uses filesystem metadata
and the current semantic-best manifest. It does not load model tensors,
execute a model,change configuration,or stop training.
Raw inventories:

- `docs/results/mcln_pvground_20261101/mcln-nr3d-storage-retention-proposal.json`
- `docs/results/mcln_pvground_20261101/mcln-sr3d-storage-retention-proposal.json`

Alternative:archive candidate weights to other storage before removing
remote copies,with file size and SHA verification. Local C:has only about
10.3GB free,D:34.3GB,E:179.0GB at inventory time. These are available
spaces only;no archive destination or bulk transfer has been selected.
All-model archival for both remaining budgets would itself exceed E:'s
currently available179GB. A bounded policy or additional storage is needed.
