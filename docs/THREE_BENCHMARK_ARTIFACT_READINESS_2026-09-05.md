# Three-benchmark artifacts: current read-only checks

Nr3D L1 uses the protected averaged checkpoint, SHA256
76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1.
The complete L1 preflight rechecked source/data/checkpoint integrity. The
isolated active run is text_position_l1_20260905_v2. No new formal result has
been adopted.

ScanRefer's four protected pipeline artifacts were checked on the server at
21:22 CST. All exist, retain mode0444, and match their documented SHA256:

| Artifact | SHA256 |
|---|---|
| Epoch71 backbone |3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208|
| Parent reranker |f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b|
| Geometry reranker |835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f|
| V99 |9752990c393fa6e45173a9dd129c4de4bb740924094dcbbec2f3121cbf39d1f2|

This is an artifact integrity check, not a new ScanRefer forward or evaluation.
Full paths and sizes are in refine-logs/scanrefer_protected_chain_20260905.json.

Sr3D's83572-row CSV and the common train/val scene caches exist. The documented
protected checkpoint (expected SHA256
da985736e5bc116c03cca51a523a211cade515d9b7580deb8e9d48bf8a4499d3),
its averaged-candidate original, and the documented E26/E29 parents are absent
from their expected paths on the current33476 instance. Their control,
evaluation and backbone subdirectories are also absent. A bounded inventory
of DATA_ROOT/output found no Sr3D checkpoint; the mounted storage root offered
no additional archive candidate. This does not establish that external backups
or another instance lack these files.

An initial generic filename check included test_v3scans.pkl. The actual dataset
code uses split_v3scans.pkl and formal evaluation uses split=val;
val_v3scans.pkl exists(2107695498 bytes). Absence of a file named test is not
a missing formal-evaluation input. This correction is explicit in
refine-logs/sr3d_readiness_exact_paths_20260905.json.

The backup location was requested while the independent Nr3D run continued.
Do not substitute an arbitrary checkpoint or restart the canceled long baseline
to bypass this missing artifact. No Sr3D trainer, validation, weight deletion
or data change was performed in these checks.

The complete goal remains Nr3D REC>60%, Sr3D REC>68.9%, with the protected
ScanRefer result preserved. A passed input/gradient check, a module-holdout
gain, or a Mask-only result does not satisfy those benchmark requirements.
