# Current ScanRefer Data Audit

Observed 2026-11-03 01:25-01:27 UTC on the current restart host.
The Stage A trainer189152, keeper188533 and preserver188537 were live
at01:25UTC. No training data, model source or running configuration changed.

Current restart source `src/joint_det_dataset.py:205` loads
`DATA_ROOT/{train,val}_v3scans.pkl`, and lines211-226 load
`DATA_ROOT/superpoints/{train,val}/SCENE_superpoint.pth`.
The current root is `/root/autodl-tmp/PV-Ground/data`.
Its `superpoints` directory is not a symlink. The loader does not
automatically select a separate mesh directory.

All312 current validation superpoint files were fingerprinted and then
read on the remote CPU. Each file matched its recorded SHA256.
All arrays have50,000 entries. Unique superpoint counts:

| Statistic | Count |
| --- | ---: |
| Minimum | 157 |
| 25th percentile | 569.75 |
| Median | 1001 |
| 75th percentile | 1373.5 |
| Maximum | 3894 |
| Single-superpoint scenes | 0 |
| Scenes with negative labels | 0 |

These checks exclude single-cluster and negative-label degeneration in
the inspected validation files. They do not establish geometric quality,
historical mesh equivalence, train-split quality or a cause of metric gaps.
The historical mesh reference manifest has not been compared; planC9/D2
therefore remains only partially resolved.

Manifest digest, formed from sorted relative-path/TAB/file-SHA/newline:
`f989fc5c7aeb25707ad75ae26e6db0dae400163e0ea9b34fc0f50113fdb5acc2`.
Validation annotation SHA:
`c9a44fa2cfb83ea1893a76f4041f206bb0432614600b43d87b47ca2d791e76f3`,
matching the annotation identity recorded in the canonical desktop handoff.
Dataset source SHA:
`f6031ee7ffb3f92a9da4611741532a4d9e0bbd1d4cfe3b28b9d9a047e41a9346`.

Raw per-file identities and per-scene statistics are in
`docs/results/mcln_pvground_20261101/scanrefer-current-data-fingerprint.json`
and `scanrefer-current-superpoint-statistics.json`.
Reproduction helpers are `tools/fingerprint_scanrefer_current_data.sh`
and `tools/audit_scanrefer_current_superpoints.sh`, executed remotely via
the credential wrapper. The latter loaded data arrays only, kept CUDA
uninitialized, and did not import or execute a model. No local tests ran.
