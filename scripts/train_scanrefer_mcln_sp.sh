#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/DATA_ROOT/output/logs/}"
PP_CHECKPOINT="${PP_CHECKPOINT:-/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-4444}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export PYTHONPATH="${PWD}:${PWD}/pointnet2:${PYTHONPATH:-}"

TORCH_DISTRIBUTED_DEBUG=INFO "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root "${DATA_ROOT}" \
    --val_freq 1 --batch_size 12 --save_freq 1 --print_freq 500 \
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir "${LOG_DIR}" \
    --lr_decay_epochs 50 75 \
    --pp_checkpoint "${PP_CHECKPOINT}" \
    --butd --self_attend --augment_det \
    --max_epoch 100 \
    --model MCLN \
    --exp MCLN_source_choice \
    "$@"
