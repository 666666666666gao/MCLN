#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
trap 'printf "%s\n" "$?" > cpu_inventory.exit' EXIT
/root/miniconda3/envs/bdetr/bin/python -u inspect_transferred_checkpoint.py
