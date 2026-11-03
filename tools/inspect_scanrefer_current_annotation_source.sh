#!/usr/bin/env bash
set -euo pipefail
stage=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102
date -u -Is
sed -n '585,655p' "$stage/source/src/joint_det_dataset.py"
sed -n '736,790p' "$stage/source/src/joint_det_dataset.py"
grep -nE 'dataset_dict|Joint3D|augment_det' "$stage/source/train_dist_mod.py" | tail -n 25
find /root/autodl-tmp/PV-Ground/data/ScanRefer -maxdepth 1 \
    -type f -name 'ScanRefer_filtered*' -printf '%f %s bytes\n'
