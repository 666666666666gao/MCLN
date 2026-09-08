#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1
trap 'printf "%s\n" "$?" > inventory.exit' EXIT
env CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 MAX_JOBS=4 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground/pointnet2:/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground:/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/OpenPCDet TOKENIZERS_PARALLELISM=false TORCH_CUDA_ARCH_LIST=8.0 TRANSFORMERS_OFFLINE=1 /root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python -u /root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1/inventory.py
