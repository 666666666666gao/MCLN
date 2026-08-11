#!/usr/bin/env bash
# Frozen-backbone mask-head fine-tune probe.
# Trains ONLY x_mask/x_query/seed_decoder (--frozen) from the protected epoch-71
# backbone. Backbone/box/reranker paths are frozen. New log dir only; the
# protected checkpoint is read-only and loaded, never written.
set -euo pipefail

DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
CKPT="/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/DATA_ROOT/output/mask_head_finetune/probe_$(date -u +%Y%m%dT%H%M%SZ)}"
LR="${LR:-2e-4}"
MAX_EPOCH="${MAX_EPOCH:-80}"
BATCH="${BATCH:-12}"
MASTER_PORT="${MASTER_PORT:-29681}"

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="${PWD}:${PWD}/pointnet2:${PYTHONPATH:-}"

echo "LOG_DIR=${LOG_DIR}"
echo "LR=${LR} MAX_EPOCH=${MAX_EPOCH} BATCH=${BATCH}"

TORCH_DISTRIBUTED_DEBUG=INFO /root/miniconda3/envs/bdetr/bin/python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port "${MASTER_PORT}" \
    train_dist_mod.py --num_decoder_layers 6 \
    --num_target 256 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root "${DATA_ROOT}" \
    --val_freq 1 --batch_size "${BATCH}" --save_freq 1 --print_freq 500 \
    --lr "${LR}" --lr_backbone 0 --text_encoder_lr 0 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --skip_missing_superpoints \
    --log_dir "${LOG_DIR}" \
    --checkpoint_path "${CKPT}" \
    --frozen --reduce_lr \
    --augment_det \
    --max_epoch "${MAX_EPOCH}" \
    --model MCLN \
    --exp mask_head_frozen_probe \
    "$@"
