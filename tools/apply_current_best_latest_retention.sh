#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
python=/root/autodl-tmp/PV-Ground/env/bin/python
for dataset in nr3d sr3d; do
    if [ -f "$root/runs/$dataset/train.log" ]; then
        CUDA_VISIBLE_DEVICES="" "$python" "$root/retain_mcln_best_latest.py" \
            --dataset "$dataset" --apply
        session="pvg_${dataset}_best_latest_retention"
        test -z "$(screen -ls | grep "[.]${session}[[:space:]]" || true)"
        screen -dmS "$session" bash -c \
            'export CUDA_VISIBLE_DEVICES=""; exec "$1" -u "$2/retain_mcln_best_latest.py" --dataset "$3" --apply --watch >> "$2/runs/$3/best_latest_retention.log" 2>&1' \
            retention "$python" "$root" "$dataset"
        screen -ls | grep "$session"
    fi
done
