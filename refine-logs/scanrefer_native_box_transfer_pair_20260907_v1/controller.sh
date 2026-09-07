#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_native_box_transfer_pair.py --manifest /root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1/input_manifest.json
status=$?
printf '%s\n' "$status" > training.exit
if [ "$status" -eq 0 ]; then
  CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/bdetr/bin/python -u scripts/audit_scanrefer_native_box_transfer_pair.py --manifest /root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1/input_manifest.json
  status=$?
fi
printf '%s\n' "$status" > controller.exit
exit "$status"
