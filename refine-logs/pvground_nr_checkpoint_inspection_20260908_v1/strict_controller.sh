#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1
trap 'printf "%s\n" "$?" > strict_controller.exit' EXIT
exec_python=/root/miniconda3/envs/bdetr/bin/python
"$exec_python" -u nr_strict_queue.py
