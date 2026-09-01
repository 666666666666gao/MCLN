#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${MODE:-smoke}"
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v134_parent_relative_sacr"
readonly GPU_IDS=0
readonly NPROC_PER_NODE=1
readonly BATCH_SIZE=8
readonly REFINER_LR="0.0003"
readonly MAX_DELTA="0.25"
readonly PROMOTION_MARGIN="0.01"
readonly TEMPERATURE="0.1"
readonly MASK_WEIGHT="0.25"
MASTER_PORT="${MASTER_PORT:-5134}"

PARENT_CHECKPOINT="${V134_PARENT_CHECKPOINT:-}"
PARENT_SHA256="${V134_PARENT_SHA256:-}"
CONTRACT_RECEIPT="${V134_CONTRACT_RECEIPT:-}"
SMOKE_GATE_RECEIPT="${V134_SMOKE_GATE_RECEIPT:-${OUTPUT_ROOT}/gates/v134_smoke_gate.json}"

if (($# != 0)); then
  echo "V134 uses one frozen configuration; positional overrides are forbidden" >&2
  exit 2
fi
if [[ -z "${PARENT_CHECKPOINT}" || -z "${PARENT_SHA256}" ]]; then
  echo "V134_PARENT_CHECKPOINT and V134_PARENT_SHA256 are required" >&2
  exit 2
fi
if [[ -z "${CONTRACT_RECEIPT}" ]]; then
  echo "V134_CONTRACT_RECEIPT is required" >&2
  exit 2
fi
if [[ "${GPU_IDS}" == *,* ]]; then
  echo "V134 is restricted to one GPU" >&2
  exit 2
fi
if [[ ! -f "${PARENT_CHECKPOINT}" ]]; then
  echo "V134 parent checkpoint is missing" >&2
  exit 4
fi
actual_parent_sha="$(sha256sum "${PARENT_CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_parent_sha}" != "${PARENT_SHA256}" ]]; then
  echo "V134 parent checkpoint SHA-256 changed" >&2
  exit 4
fi

case "${MODE}" in
  smoke)
    MAX_EPOCH=2
    VAL_FREQ=1
    PRINT_FREQ=1
    EXPECTED_EVAL_SAMPLE_COUNT=128
    SPLIT_ARGS=(--debug --debug_train_holdout)
    ;;
  formal)
    MAX_EPOCH=4
    VAL_FREQ=1
    PRINT_FREQ=100
    EXPECTED_EVAL_SAMPLE_COUNT=9508
    SPLIT_ARGS=()
    if [[ ! -f "${SMOKE_GATE_RECEIPT}" ]]; then
      echo "V134 formal run requires a smoke gate receipt" >&2
      exit 5
    fi
    "${PYTHON_BIN}" scripts/audit_v134_smoke_gate.py verify \
      --receipt "${SMOKE_GATE_RECEIPT}" \
      --repo-root "${ROOT_DIR}" \
      --parent-checkpoint "${PARENT_CHECKPOINT}" \
      --contract-receipt "${CONTRACT_RECEIPT}"
    ;;
  *)
    echo "MODE must be smoke or formal" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" scripts/audit_v134_sacr_parent_relative_contract.py \
  --verify "${CONTRACT_RECEIPT}" --repo-root "${ROOT_DIR}"

EXP="v134_parent_relative_${MODE}_e1_e${MAX_EPOCH}_b${BATCH_SIZE}x${NPROC_PER_NODE}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/${EXP}}"
LAUNCH_DIR="${OUTPUT_ROOT}/launch"
LOCK_FILE="${OUTPUT_ROOT}/v134_gpu.lock"
mkdir -p "${LOG_DIR}" "${LAUNCH_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another V134 run owns ${LOCK_FILE}" >&2
  exit 3
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
echo "parent_checkpoint=${PARENT_CHECKPOINT}"
echo "parent_sha256=${PARENT_SHA256}"
echo "contract_receipt=${CONTRACT_RECEIPT}"
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
  --use_sacr_score_refiner \
  --sacr_score_refiner_train_only \
  --sacr_score_use_parent_relative_abstention \
  --sacr_score_parent_gate_hidden_dim 32 \
  --sacr_score_refiner_lr "${REFINER_LR}" \
  --sacr_score_refiner_loss_weight 1.0 \
  --sacr_score_temperature "${TEMPERATURE}" \
  --sacr_score_mask_weight "${MASK_WEIGHT}" \
  --sacr_score_max_delta "${MAX_DELTA}" \
  --sacr_score_min_box_advantage 0.03 \
  --sacr_score_promotion_margin "${PROMOTION_MARGIN}" \
  --sacr_score_mask_tolerance 0.02 \
  --sacr_score_raw_margin 0.1 \
  --sacr_score_dense_weight 0.25 \
  --sacr_score_preserve_weight 1.0 \
  --sacr_score_gate_weight 0.05 \
  --sacr_score_saturation_weight 0.05 \
  --sacr_hidden_dim 288 \
  --sacr_max_pairs 3 \
  --sacr_top_m_targets 32 \
  --sacr_top_k_anchors 16 \
  --sacr_geo_dim 16 \
  --sacr_min_parse_confidence 0.0 \
  --checkpoint_metric_retention \
  "${SPLIT_ARGS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] finished ${EXP}"
echo "launch_log=${LAUNCH_LOG}"
