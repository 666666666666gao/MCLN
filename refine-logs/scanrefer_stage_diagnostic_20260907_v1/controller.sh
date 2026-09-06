#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd /root/autodl-tmp/mcln_scanrefer_stage_diagnostic_20260907_v1
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/diagnose_scanrefer_readout_stages.py --manifest /root/autodl-tmp/mcln_scanrefer_stage_diagnostic_20260907_v1/input_manifest.json
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
