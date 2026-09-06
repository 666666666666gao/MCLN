#!/usr/bin/env bash
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
cd /root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_frozen_readout_pair.py --manifest /root/autodl-tmp/mcln_scanrefer_frozen_readout_pair_20260907_v1/input_manifest.json
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
