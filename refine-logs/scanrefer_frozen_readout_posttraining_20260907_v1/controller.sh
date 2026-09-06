#!/usr/bin/env bash
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=''
cd /root/autodl-tmp/mcln_scanrefer_frozen_readout_posttraining_20260907_v1
/root/miniconda3/envs/bdetr/bin/python -u queue.py --manifest /root/autodl-tmp/mcln_scanrefer_frozen_readout_posttraining_20260907_v1/input_manifest.json
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
