# Current ScanRefer Data Audit

## Training Directory Audit

At2026-11-03 03:28:29UTC all1,201 training-directory superpoint files
were read on the remote CPU. Their scene IDs exactly match the current
source's `data/meta_data/scannetv2_train.txt`. Every array has50,000 integer
labels; none has negative labels or only one superpoint. Counts range
from90 to4,363, with p25=516, median843, p75=1,280.
The total file size is508,340,607bytes; the sorted path/TAB/SHA/newline
manifest digest is:
`459732f594eab7990506122f244403687133babf4c1fc1972e0506b2e59603c0`.

The current ScanRefer training JSON contains36,665 examples over562 scenes,
after applying its actual train TXT scene filter. Its SHA256 still matches
the earlier fingerprint:
`93b0bd2a884de8077ba659aa1ee341dd7f571b32c46986e0ce388d6aad349521`.
None of the1,201 audited training scene IDs overlaps the141 scene IDs
referenced by the validation JSON. The earlier312-file validation audit
covered the full validation directory, a larger set than the141 scenes
used by ScanRefer's9,508 validation examples.

The observed training dataset length48,655 does not equal the ScanRefer
JSON length because joint detection is enabled. The active
`train_dist_mod.py::get_datasets` adds `scannet:10`; the active
`Joint3DDataset.__init__` appends each dataset's annotations times its
configured count. Thus48,655=36,665+1,199*10. The1,199 count is inferred
from this source recipe and logged length; the audit did not reconstruct
the detection loader's object-vocabulary filtering. The loader also
excludes detection annotation indices965/977. This is an explained
sampling composition, not evidence of a data defect or a metric cause.
The initial562-scene audit covered only language annotations; the final
1,201-file audit extends coverage to the training directory used for
joint detection as well.

Trainer207977 and retention209444 were verified live before and after.
CUDA remained uninitialized in the audit process. No model import,
forward, evaluation, local test, data edit or training change occurred.
This rules out the checked missing-file/shape/type/single-cluster/negative
label problems in these files. Geometry quality, historical mesh
equivalence and the actual cause of metric gaps remain unproven.

Raw receipt:
`docs/results/mcln_pvground_20261101/scanrefer-current-all-train-superpoint-statistics.json`.
Reproduction:
`tools/audit_scanrefer_current_train_superpoints.sh` and
`tools/inspect_scanrefer_current_annotation_source.sh`.

## Earlier Validation Audit

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
historical mesh equivalence or a cause of metric gaps. The later training
audit above extends file-integrity coverage without establishing geometry quality.
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
