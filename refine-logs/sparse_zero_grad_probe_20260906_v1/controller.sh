#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/mcln_sparse_zero_grad_probe_20260906_v1
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/autodl-tmp/mcln_sparse_runtime_20260906_v3/venv/bin/python -u scripts/probe_nr3d_sparse_zero_gradients.py --manifest /root/autodl-tmp/mcln_sparse_zero_grad_probe_20260906_v1/input_manifest.json
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
