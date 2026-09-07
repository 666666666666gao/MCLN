#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/mcln_native_score_fit_20260907_v1
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u run_scanrefer_native_score_fit_audit.py --manifest input_manifest.json
status=$?
printf "%s\n" "$status" > controller.exit
exit "$status"
