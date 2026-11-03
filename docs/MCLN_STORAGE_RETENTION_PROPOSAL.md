# Current Checkpoint Storage Proposal

## Authorized Policy Applied: 2026-11-03

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
