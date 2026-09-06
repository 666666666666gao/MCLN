#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd /root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v3
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/evaluate_scanrefer_local_visual_official.py --manifest /root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v3/input_manifest.json
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
