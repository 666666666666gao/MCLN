#!/usr/bin/env bash
# eval-mode mask-head fine-tune: permanently eval(), trains ONLY x_mask + x_query
# BN running stats never updated → backbone/box path bit-identical to protected baseline
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
CKPT="/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/DATA_ROOT/output/mask_head_finetune/evalmode_$(date -u +%Y%m%dT%H%M%SZ)}"
LR="${LR:-2e-4}"
MAX_EPOCH="${MAX_EPOCH:-80}"
BATCH="${BATCH:-12}"
MASTER_PORT="${MASTER_PORT:-29682}"

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PWD}/pointnet2:${PYTHONPATH:-}"

echo "LOG_DIR=${LOG_DIR}"
echo "LR=${LR}  MAX_EPOCH=${MAX_EPOCH}  BATCH=${BATCH}"

# pass ALL MCLN-required flags so parse_option() builds the right config
TORCH_DISTRIBUTED_DEBUG=INFO /root/miniconda3/envs/bdetr/bin/python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port "${MASTER_PORT}" \
    scripts/run_mask_head_evalmode_ft.py \
    --num_decoder_layers 6 \
    --num_target 256 \
    --model MCLN \
    --use_color \
    --butd \
    --self_attend \
    --detect_intermediate \
    --joint_det \
    --use_soft_token_loss \
    --use_contrastive_align \
    --use_source_choice_selector \
    --source_choice_selector_sources "default,default_rank_blend_contrastive010" \
    --source_choice_selector_hidden_dim 288 \
    --source_choice_selector_loss_weight 0.0 \
    --skip_missing_superpoints \
    --augment_det \
    --data_root "${DATA_ROOT}" \
    --checkpoint_path "${CKPT}" \
    --log_dir "${LOG_DIR}" \
    --exp "mask_head_evalmode" \
    --dataset scanrefer \
    --test_dataset scanrefer \
    --batch_size "${BATCH}" \
    --num_workers 2 \
    --lr "${LR}" \
    --lr_backbone 0.0 \
    --text_encoder_lr 0.0 \
    --weight_decay 0.0005 \
    --max_epoch "${MAX_EPOCH}" \
    --val_freq 1 \
    --save_freq 1 \
    --print_freq 200 \
    --clip_norm 0.1 \
    --warmup-epoch 2 \
    --lr_decay_epochs 50 70 \
    --lr_decay_rate 0.1 \
    "$@"
