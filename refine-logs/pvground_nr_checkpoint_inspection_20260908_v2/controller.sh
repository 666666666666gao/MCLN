#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v2
trap 'printf "%s\n" "$?" > controller.exit' EXIT
/root/miniconda3/envs/bdetr/bin/python -u controller.py
