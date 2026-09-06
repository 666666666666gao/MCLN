#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/mcln_scanrefer_teacher_transfer_20260906_v3
while [ ! -f /root/autodl-tmp/mcln_scanrefer_joint_native_probe_20260906_v1/controller.exit ]; do sleep 240; done
/root/miniconda3/envs/bdetr/bin/python dependency_check.py
status=$?
if [ "$status" -eq 0 ]; then
    flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/audit_scanrefer_v99_teacher_transfer.py --manifest /root/autodl-tmp/mcln_scanrefer_teacher_transfer_20260906_v3/input_manifest.json
    status=$?
fi
printf '%s\n' "$status" > controller.exit
exit "$status"
