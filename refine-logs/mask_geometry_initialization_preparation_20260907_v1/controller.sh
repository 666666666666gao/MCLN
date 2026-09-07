#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/mcln_mask_geometry_initialization_preparation_20260907_v1
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests > cpu_tests.txt 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  /root/miniconda3/envs/bdetr/bin/python -u check_mask_geometry_initialization_cpu.py > cpu_load_stdout.txt 2> cpu_load_stderr.txt
  status=$?
fi
printf '%s\n' "$status" > controller.exit
exit "$status"
