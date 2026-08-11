#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_network_best_v19_rec025_0.581195_mask025_0.598233.pth}"
PP_CHECKPOINT="${PP_CHECKPOINT:-${DATA_ROOT%/}/gf_detector_l6o256.pth}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT%/}/output/query_mask_fusion}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_PORT="${MASTER_PORT:-4472}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-1}"
PERSISTENT_TRAIN_WORKERS="${PERSISTENT_TRAIN_WORKERS:-0}"
START_EPOCH="${START_EPOCH:-1}"
MAX_EPOCH="${MAX_EPOCH:-5}"
VAL_FREQ="${VAL_FREQ:-1}"
PRINT_FREQ="${PRINT_FREQ:-50}"
EXPECTED_EVAL_SAMPLE_COUNT="${EXPECTED_EVAL_SAMPLE_COUNT:-9508}"
EXP="${EXP:-query_mask_fusion_h128_d0_lr1e3_delta025}"
QUERY_MASK_FUSION_LR="${QUERY_MASK_FUSION_LR:-1e-3}"
QUERY_MASK_FUSION_HIDDEN_DIM="${QUERY_MASK_FUSION_HIDDEN_DIM:-128}"
QUERY_MASK_FUSION_DROPOUT="${QUERY_MASK_FUSION_DROPOUT:-0.0}"
QUERY_MASK_FUSION_MAX_DELTA="${QUERY_MASK_FUSION_MAX_DELTA:-0.25}"
DEBUG="${DEBUG:-0}"
QUERY_MASK_FUSION_RESUME_OPTIMIZER="${QUERY_MASK_FUSION_RESUME_OPTIMIZER:-0}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PWD}/pointnet2:${PYTHONPATH:-}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ -z "${NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="${#VISIBLE_GPU_IDS[@]}"
fi
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer" >&2
  exit 2
fi
if (( NPROC_PER_NODE > ${#VISIBLE_GPU_IDS[@]} )); then
  echo "NPROC_PER_NODE cannot exceed visible GPU count" >&2
  exit 2
fi

DEBUG_ARGS=()
if [[ "${DEBUG}" == "1" ]]; then
  DEBUG_ARGS=(--debug)
elif [[ "${DEBUG}" != "0" ]]; then
  echo "DEBUG must be 0 or 1" >&2
  exit 2
fi

RESUME_ARGS=()
if [[ "${QUERY_MASK_FUSION_RESUME_OPTIMIZER}" == "1" ]]; then
  RESUME_ARGS=(--query_mask_fusion_resume_optimizer)
elif [[ "${QUERY_MASK_FUSION_RESUME_OPTIMIZER}" != "0" ]]; then
  echo "QUERY_MASK_FUSION_RESUME_OPTIMIZER must be 0 or 1" >&2
  exit 2
fi

PERSISTENT_WORKER_ARGS=()
if [[ "${PERSISTENT_TRAIN_WORKERS}" == "1" ]]; then
  PERSISTENT_WORKER_ARGS=(--persistent_train_workers)
elif [[ "${PERSISTENT_TRAIN_WORKERS}" != "0" ]]; then
  echo "PERSISTENT_TRAIN_WORKERS must be 0 or 1" >&2
  exit 2
fi

"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  train_dist_mod.py \
  --num_decoder_layers 6 \
  --use_color \
  --weight_decay 0.0005 \
  --data_root "${DATA_ROOT}" \
  --val_freq "${VAL_FREQ}" --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  "${PERSISTENT_WORKER_ARGS[@]}" \
  --save_freq 1 --print_freq "${PRINT_FREQ}" \
  --lr_backbone 2e-4 --lr 2e-5 --text_encoder_lr 3e-6 \
  --dataset scanrefer --test_dataset scanrefer \
  --detect_intermediate \
  --use_soft_token_loss --use_contrastive_align \
  --log_dir "${LOG_DIR}" \
  --lr_decay_epochs 50 75 \
  --pp_checkpoint "${PP_CHECKPOINT}" \
  --butd --self_attend --augment_det \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --start_epoch "${START_EPOCH}" --max_epoch "${MAX_EPOCH}" \
  --model MCLN --exp "${EXP}" \
  --use_source_moe --source_moe_use_fallback_gate \
  --eval_use_selector_choice_scores \
  --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}" \
  --use_query_mask_fusion_calibrator \
  --query_mask_fusion_train_only \
  --query_mask_fusion_lr "${QUERY_MASK_FUSION_LR}" \
  --query_mask_fusion_hidden_dim "${QUERY_MASK_FUSION_HIDDEN_DIM}" \
  --query_mask_fusion_dropout "${QUERY_MASK_FUSION_DROPOUT}" \
  --query_mask_fusion_max_delta "${QUERY_MASK_FUSION_MAX_DELTA}" \
  --checkpoint_metric_retention \
  "${RESUME_ARGS[@]}" \
  "${DEBUG_ARGS[@]}" \
  "$@"
