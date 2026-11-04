#!/usr/bin/env bash
set -euo pipefail
trial=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_reader_competitor_20261104
test -f "$trial/preparation.json"
test ! -e "$trial/started.json"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
test -z "$(screen -ls | grep '[.]pvg_scanrefer_reader_competitor[[:space:]]' || true)"
screen -dmS pvg_scanrefer_reader_competitor bash "$trial/source/tools/run_scanrefer_reader_competitor.sh"
screen -ls | grep pvg_scanrefer_reader_competitor
