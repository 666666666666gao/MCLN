#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${MODE:-smoke}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-${DATA_ROOT%/}/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth}"
PARENT_SHA256="${PARENT_SHA256:-3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT%/}/output/network_v132_decoder_query_adapter}"
GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-5132}"
HIDDEN_DIM="${HIDDEN_DIM:-288}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
MAX_DELTA="${MAX_DELTA:-0.25}"
ADAPTER_LR="${ADAPTER_LR:-3e-4}"

case "${MODE}" in
  baseline)
    BATCH_SIZE="${BATCH_SIZE:-8}"
    MAX_EPOCH=1
    VAL_FREQ=1
    PRINT_FREQ="${PRINT_FREQ:-1}"
    EXPECTED_EVAL_SAMPLE_COUNT=128
    SPLIT_ARGS=(--debug --debug_train_holdout --eval)
    ;;
  smoke)
    BATCH_SIZE="${BATCH_SIZE:-8}"
    MAX_EPOCH="${MAX_EPOCH:-2}"
    VAL_FREQ="${VAL_FREQ:-1}"
    PRINT_FREQ="${PRINT_FREQ:-1}"
    EXPECTED_EVAL_SAMPLE_COUNT=128
    SPLIT_ARGS=(--debug --debug_train_holdout)
    ;;
  formal)
    BATCH_SIZE="${BATCH_SIZE:-8}"
    MAX_EPOCH="${MAX_EPOCH:-12}"
    VAL_FREQ="${VAL_FREQ:-2}"
    PRINT_FREQ="${PRINT_FREQ:-100}"
    EXPECTED_EVAL_SAMPLE_COUNT=9508
    SPLIT_ARGS=()
    ;;
  *)
    echo "MODE must be baseline, smoke, or formal" >&2
    exit 2
    ;;
esac

EXP="v132_decoder_query_adapter_${MODE}_e1_e${MAX_EPOCH}_b${BATCH_SIZE}x${NPROC_PER_NODE}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/${EXP}}"
LAUNCH_DIR="${OUTPUT_ROOT}/launch"
LOCK_FILE="${OUTPUT_ROOT}/v132_${MODE}.lock"
mkdir -p "${LOG_DIR}" "${LAUNCH_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another V132 ${MODE} run owns ${LOCK_FILE}" >&2
  exit 3
fi

actual_parent_sha="$(sha256sum "${PARENT_CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_parent_sha}" != "${PARENT_SHA256}" ]]; then
  echo "protected epoch71 parent SHA-256 changed" >&2
  exit 4
fi

timestamp="$(date '+%Y%m%d_%H%M%S')"
LAUNCH_LOG="${LAUNCH_DIR}/${EXP}_${timestamp}.log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pointnet2:${PYTHONPATH:-}"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] starting ${EXP} on GPU ${GPU_IDS}"
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  train_dist_mod.py \
  --num_decoder_layers 6 \
  --use_color \
  --weight_decay 0.0005 \
  --data_root "${DATA_ROOT}" \
  --val_freq "${VAL_FREQ}" --batch_size "${BATCH_SIZE}" \
  --num_workers 2 --dataloader_prefetch_factor 1 \
  --save_freq 1 --print_freq "${PRINT_FREQ}" \
  --lr_backbone 2e-4 --lr 2e-5 --text_encoder_lr 3e-6 \
  --dataset scanrefer --test_dataset scanrefer \
  --detect_intermediate --augment_det \
  --use_soft_token_loss --use_contrastive_align \
  --log_dir "${LOG_DIR}" \
  --lr_decay_epochs 8 10 \
  --pp_checkpoint "${DATA_ROOT%/}/gf_detector_l6o256.pth" \
  --butd --self_attend \
  --checkpoint_path "${PARENT_CHECKPOINT}" \
  --start_epoch 1 --max_epoch "${MAX_EPOCH}" \
  --model MCLN --exp "${EXP}" \
  --use_source_choice_selector \
  --source_choice_selector_sources default,default_rank_blend_contrastive010 \
  --source_choice_selector_hidden_dim 288 \
  --source_choice_selector_default_source default \
  --source_choice_selector_choice_target precision_gain_default_sourcewise_focal_bce \
  --source_choice_selector_min_iou_gap 0.03 \
  --eval_use_selector_choice_scores \
  --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}" \
  --use_decoder_query_adapter \
  --decoder_query_adapter_train_only \
  --decoder_query_adapter_lr "${ADAPTER_LR}" \
  --decoder_query_adapter_hidden_dim "${HIDDEN_DIM}" \
  --decoder_query_adapter_heads "${HEADS}" \
  --decoder_query_adapter_dropout "${DROPOUT}" \
  --decoder_query_adapter_max_delta "${MAX_DELTA}" \
  --checkpoint_metric_retention \
  "${SPLIT_ARGS[@]}" \
  "$@"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] finished ${EXP}"
