#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
python=/root/autodl-tmp/PV-Ground/env/bin/python
test -f "$root/scanrefer_restart_20261102/stage_b_start.json"
test -z "$(screen -ls | grep '[.]pvg_scanrefer_best_latest_retention[[:space:]]' || true)"
screen -dmS pvg_scanrefer_best_latest_retention bash -c \
    'export CUDA_VISIBLE_DEVICES=""; exec "$1" -u "$2/retain_mcln_best_latest.py" --dataset scanrefer --apply --watch >> "$2/scanrefer_restart_20261102/best_latest_retention.log" 2>&1' \
    retention "$python" "$root"
screen -ls | grep pvg_scanrefer_best_latest_retention
