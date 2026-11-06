#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_v99_x0_controls_20261106
mkdir -p "$root"
tar -xf /root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_x0_controls_20261106.tar -C "$root"
export CUDA_VISIBLE_DEVICES=''
/root/autodl-tmp/PV-Ground/env/bin/python "$root/tools/check_scanrefer_x0_controls.py"
