#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
date -u -Is
ps -eo pid,ppid,etime,args | grep '[r]etain_mcln_best_latest.py' | cut -c1-300
for dataset in nr3d sr3d; do
    if [ -f "$root/runs/$dataset/best_latest_retention.json" ]; then
        tail -n 3 "$root/runs/$dataset/best_latest_retention.log"
        find "$root/runs/$dataset/preserved_checkpoints" -maxdepth 1 \
            -type f -name '*.pth' -printf '%f %s %n links\n'
    fi
done
if [ -f "$root/scanrefer_restart_20261102/best_latest_retention.log" ]; then
    tail -n 3 "$root/scanrefer_restart_20261102/best_latest_retention.log"
fi
