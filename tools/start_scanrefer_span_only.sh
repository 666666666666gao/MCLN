#!/usr/bin/env bash
set -euo pipefail
trial=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_only_20261104
test -f "$trial/preparation.json"
test ! -e "$trial/started.json"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
screen -dmS pvg_scanrefer_span_only bash /tmp/mcln_run_span_only_20261104.sh
screen -ls | grep '[.]pvg_scanrefer_span_only[[:space:]]'
