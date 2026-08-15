#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_network_best_v19_rec025_0.581195_mask025_0.598233.pth}"
PP_CHECKPOINT="${PP_CHECKPOINT:-${DATA_ROOT%/}/gf_detector_l6o256.pth}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT%/}/output/v105_egqs_mask}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-4765}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-1}"
PERSISTENT_TRAIN_WORKERS="${PERSISTENT_TRAIN_WORKERS:-0}"
START_EPOCH="${START_EPOCH:-1}"
MAX_EPOCH="${MAX_EPOCH:-3}"
VAL_FREQ="${VAL_FREQ:-1}"
PRINT_FREQ="${PRINT_FREQ:-50}"
EXPECTED_EVAL_SAMPLE_COUNT="${EXPECTED_EVAL_SAMPLE_COUNT:-9508}"
EXP="${EXP:-v105_egqs_all_h32_lr3e4_delta2}"
EGQS_MASK_REFINER_LR="${EGQS_MASK_REFINER_LR:-3e-4}"
EGQS_MASK_REFINER_ARCH="${EGQS_MASK_REFINER_ARCH:-egqs}"
EGQS_MASK_REFINER_HIDDEN_DIM="${EGQS_MASK_REFINER_HIDDEN_DIM:-32}"
EGQS_MASK_REFINER_MAX_DELTA="${EGQS_MASK_REFINER_MAX_DELTA:-2.0}"
EGQS_MASK_REFINER_COMPONENTS="${EGQS_MASK_REFINER_COMPONENTS:-all}"
EGQS_MASK_REFINER_GRAPH_MODE="${EGQS_MASK_REFINER_GRAPH_MODE:-bilateral}"
EGQS_MASK_REFINER_NEIGHBOR_COUNT="${EGQS_MASK_REFINER_NEIGHBOR_COUNT:-8}"
DEBUG="${DEBUG:-0}"

case "${EGQS_MASK_REFINER_ARCH}" in
  egqs|graph) ;;
  *) echo "invalid EGQS_MASK_REFINER_ARCH" >&2; exit 2 ;;
esac
case "${EGQS_MASK_REFINER_COMPONENTS}" in
  content|evidence|geometry|all) ;;
  *) echo "invalid EGQS_MASK_REFINER_COMPONENTS" >&2; exit 2 ;;
esac
case "${EGQS_MASK_REFINER_GRAPH_MODE}" in
  spatial|bilateral) ;;
  *) echo "invalid EGQS_MASK_REFINER_GRAPH_MODE" >&2; exit 2 ;;
esac
for value in "${BATCH_SIZE}" "${NUM_WORKERS}" "${START_EPOCH}" \
             "${MAX_EPOCH}" "${VAL_FREQ}" "${PRINT_FREQ}" \
             "${EGQS_MASK_REFINER_NEIGHBOR_COUNT}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "integer training settings must be positive" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PWD}/pointnet2:${PYTHONPATH:-}"

DEBUG_ARGS=()
if [[ "${DEBUG}" == "1" ]]; then
  DEBUG_ARGS=(--debug)
elif [[ "${DEBUG}" != "0" ]]; then
  echo "DEBUG must be 0 or 1" >&2
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
  --nproc_per_node 1 --master_port "${MASTER_PORT}" \
  train_dist_mod.py \
  --num_decoder_layers 6 --use_color --weight_decay 0.0005 \
  --data_root "${DATA_ROOT}" --val_freq "${VAL_FREQ}" \
  --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  "${PERSISTENT_WORKER_ARGS[@]}" \
  --save_freq 1 --print_freq "${PRINT_FREQ}" \
  --lr_backbone 2e-4 --lr 2e-5 --text_encoder_lr 3e-6 \
  --dataset scanrefer --test_dataset scanrefer \
  --detect_intermediate --use_soft_token_loss --use_contrastive_align \
  --log_dir "${LOG_DIR}" --lr_decay_epochs 50 75 \
  --pp_checkpoint "${PP_CHECKPOINT}" --butd --self_attend --augment_det \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --start_epoch "${START_EPOCH}" --max_epoch "${MAX_EPOCH}" \
  --model MCLN --exp "${EXP}" \
  --use_source_moe --source_moe_use_fallback_gate \
  --eval_use_selector_choice_scores \
  --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}" \
  --use_egqs_mask_refiner --egqs_mask_refiner_train_only \
  --egqs_mask_refiner_lr "${EGQS_MASK_REFINER_LR}" \
  --egqs_mask_refiner_arch "${EGQS_MASK_REFINER_ARCH}" \
  --egqs_mask_refiner_hidden_dim "${EGQS_MASK_REFINER_HIDDEN_DIM}" \
  --egqs_mask_refiner_max_delta "${EGQS_MASK_REFINER_MAX_DELTA}" \
  --egqs_mask_refiner_components "${EGQS_MASK_REFINER_COMPONENTS}" \
  --egqs_mask_refiner_graph_mode "${EGQS_MASK_REFINER_GRAPH_MODE}" \
  --egqs_mask_refiner_neighbor_count "${EGQS_MASK_REFINER_NEIGHBOR_COUNT}" \
  --checkpoint_metric_retention \
  "${DEBUG_ARGS[@]}" "$@"
