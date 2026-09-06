#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /root/autodl-tmp/mcln_scanrefer_local_visual_mesh_posttraining_20260906_v1
/root/miniconda3/envs/bdetr/bin/python -u wait_for_training.py
status=$?
printf '%s\n' "$status" > queue_controller.exit
exit "$status"
