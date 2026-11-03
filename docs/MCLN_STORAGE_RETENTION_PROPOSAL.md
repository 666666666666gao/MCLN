# Current Checkpoint Storage Proposal

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
