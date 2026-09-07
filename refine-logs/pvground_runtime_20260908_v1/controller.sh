#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/mcln_pvground_runtime_20260908_v1
trap 'printf "%s\n" "$?" > controller.exit' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u build_runtime.py
