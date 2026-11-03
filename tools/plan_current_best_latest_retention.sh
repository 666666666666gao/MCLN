#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
for dataset in nr3d sr3d; do
    if [ -f "$root/runs/$dataset/train.log" ]; then
        CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python \
            "$root/retain_mcln_best_latest.py" --dataset "$dataset"
    fi
done
