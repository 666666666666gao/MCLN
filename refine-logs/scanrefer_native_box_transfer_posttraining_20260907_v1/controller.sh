#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/mcln_scanrefer_native_box_transfer_posttraining_20260907_v1
/root/miniconda3/envs/bdetr/bin/python -u posttraining_queue.py --manifest /root/autodl-tmp/mcln_scanrefer_native_box_transfer_posttraining_20260907_v1/input_manifest.json
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
