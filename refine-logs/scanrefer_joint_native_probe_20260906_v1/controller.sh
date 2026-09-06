#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/mcln_scanrefer_joint_native_probe_20260906_v1
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/probe_scanrefer_joint_readout_native.py --manifest /root/autodl-tmp/mcln_scanrefer_joint_native_probe_20260906_v1/input_manifest.json
status=$?
printf '%s\n' "$status" > controller.exit
exit "$status"
