#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/mcln_scanrefer_instance_overlap_20260907_v1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'printf "%s\n" "$?" > controller.exit' EXIT
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_instance_overlap_diagnostic.py > cpu_tests.txt 2>&1
/root/miniconda3/envs/bdetr/bin/python -u -m scripts.analyze_scanrefer_instance_overlap --directory .
