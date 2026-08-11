#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
LANGUAGE_DATASET="${LANGUAGE_DATASET:-scanrefer}"
TEST_DATASET="${TEST_DATASET:-${LANGUAGE_DATASET}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
PP_CHECKPOINT="${PP_CHECKPOINT:-${DATA_ROOT%/}/gf_detector_l6o256.pth}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT%/}/output/joint_query_quality}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_PORT="${MASTER_PORT:-4483}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-1}"
PERSISTENT_TRAIN_WORKERS="${PERSISTENT_TRAIN_WORKERS:-0}"
START_EPOCH="${START_EPOCH:-1}"
MAX_EPOCH="${MAX_EPOCH:-5}"
VAL_FREQ="${VAL_FREQ:-1}"
PRINT_FREQ="${PRINT_FREQ:-50}"
EXPECTED_EVAL_SAMPLE_COUNT="${EXPECTED_EVAL_SAMPLE_COUNT:-}"
EXP="${EXP:-v41_joint_query_quality_h128_l1_d01_lr3e4_delta125}"
JOINT_QUERY_QUALITY_LR="${JOINT_QUERY_QUALITY_LR:-3e-4}"
JOINT_QUERY_QUALITY_HIDDEN_DIM="${JOINT_QUERY_QUALITY_HIDDEN_DIM:-128}"
JOINT_QUERY_QUALITY_HEADS="${JOINT_QUERY_QUALITY_HEADS:-4}"
JOINT_QUERY_QUALITY_LAYERS="${JOINT_QUERY_QUALITY_LAYERS:-1}"
JOINT_QUERY_QUALITY_DROPOUT="${JOINT_QUERY_QUALITY_DROPOUT:-0.1}"
JOINT_QUERY_QUALITY_MAX_DELTA="${JOINT_QUERY_QUALITY_MAX_DELTA:-1.25}"
JOINT_QUERY_QUALITY_MASK_WEIGHT="${JOINT_QUERY_QUALITY_MASK_WEIGHT:-0.25}"
JOINT_QUERY_QUALITY_SCORE_WEIGHT="${JOINT_QUERY_QUALITY_SCORE_WEIGHT:-1.0}"
JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION="${JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION:-0}"
JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE="${JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE:-0}"
JOINT_QUERY_QUALITY_USE_GATE_EVIDENCE="${JOINT_QUERY_QUALITY_USE_GATE_EVIDENCE:-0}"
JOINT_QUERY_QUALITY_USE_SPATIAL_MASK_REFINER="${JOINT_QUERY_QUALITY_USE_SPATIAL_MASK_REFINER:-0}"
JOINT_QUERY_QUALITY_MAX_MASK_ALPHA_DELTA="${JOINT_QUERY_QUALITY_MAX_MASK_ALPHA_DELTA:-1.0}"
JOINT_QUERY_QUALITY_MAX_MASK_LOGIT_BIAS="${JOINT_QUERY_QUALITY_MAX_MASK_LOGIT_BIAS:-2.0}"
JOINT_QUERY_QUALITY_SPATIAL_MASK_HIDDEN_DIM="${JOINT_QUERY_QUALITY_SPATIAL_MASK_HIDDEN_DIM:-32}"
JOINT_QUERY_QUALITY_MAX_SPATIAL_MASK_DELTA="${JOINT_QUERY_QUALITY_MAX_SPATIAL_MASK_DELTA:-2.0}"
JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING="${JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING:-0}"
JOINT_QUERY_QUALITY_USE_SOURCE_DISTRIBUTION_RELIABILITY="${JOINT_QUERY_QUALITY_USE_SOURCE_DISTRIBUTION_RELIABILITY:-0}"
JOINT_QUERY_QUALITY_SOURCE_NAMES="${JOINT_QUERY_QUALITY_SOURCE_NAMES:-}"
JOINT_QUERY_QUALITY_MAX_SOURCE_MIX_DELTA="${JOINT_QUERY_QUALITY_MAX_SOURCE_MIX_DELTA:-1.0}"
JOINT_QUERY_QUALITY_SOURCE_MIX_TEMPERATURE="${JOINT_QUERY_QUALITY_SOURCE_MIX_TEMPERATURE:-0.5}"
JOINT_QUERY_QUALITY_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_LOSS_WEIGHT:-1.0}"
JOINT_QUERY_QUALITY_TEMPERATURE="${JOINT_QUERY_QUALITY_TEMPERATURE:-0.25}"
JOINT_QUERY_QUALITY_AUX_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_AUX_LOSS_WEIGHT:-1.0}"
JOINT_QUERY_QUALITY_ANCHOR_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_ANCHOR_LOSS_WEIGHT:-0.5}"
JOINT_QUERY_QUALITY_ANCHOR_MARGIN="${JOINT_QUERY_QUALITY_ANCHOR_MARGIN:-0.05}"
JOINT_QUERY_QUALITY_SOURCE_MIX_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_SOURCE_MIX_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_SOURCE_MIX_ALIGNMENT_TEMPERATURE="${JOINT_QUERY_QUALITY_SOURCE_MIX_ALIGNMENT_TEMPERATURE:-0.25}"
JOINT_QUERY_QUALITY_SOURCE_MIX_QUERY_FOCUS_WEIGHT="${JOINT_QUERY_QUALITY_SOURCE_MIX_QUERY_FOCUS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K="${JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K:-16}"
USE_SACR_SOURCE="${USE_SACR_SOURCE:-0}"
SACR_HIDDEN_DIM="${SACR_HIDDEN_DIM:-288}"
SACR_MAX_PAIRS="${SACR_MAX_PAIRS:-3}"
SACR_TOP_M_TARGETS="${SACR_TOP_M_TARGETS:-32}"
SACR_TOP_K_ANCHORS="${SACR_TOP_K_ANCHORS:-16}"
SACR_GEO_DIM="${SACR_GEO_DIM:-16}"
SACR_MIN_PARSE_CONFIDENCE="${SACR_MIN_PARSE_CONFIDENCE:-0.0}"
SACR_RESIDUAL_SCALE_INIT="${SACR_RESIDUAL_SCALE_INIT:-0.1}"
MODEL_STAGE="${MODEL_STAGE:-two}"
SOURCE_ARBITER="${SOURCE_ARBITER:-moe}"
SOURCE_CHOICE_SELECTOR_SOURCES="${SOURCE_CHOICE_SELECTOR_SOURCES:-default,default_rank_blend_contrastive010}"
SOURCE_CHOICE_SELECTOR_HIDDEN_DIM="${SOURCE_CHOICE_SELECTOR_HIDDEN_DIM:-288}"
DEBUG="${DEBUG:-1}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PWD}/pointnet2:${PYTHONPATH:-}"

case "${LANGUAGE_DATASET}" in
  scanrefer|nr3d|sr3d)
    ;;
  *)
    echo "LANGUAGE_DATASET must be scanrefer, nr3d, or sr3d" >&2
    exit 2
    ;;
esac
case "${TEST_DATASET}" in
  scanrefer|nr3d|sr3d)
    ;;
  *)
    echo "TEST_DATASET must be scanrefer, nr3d, or sr3d" >&2
    exit 2
    ;;
esac
if [[ "${LANGUAGE_DATASET}" != "${TEST_DATASET}" ]]; then
  echo "LANGUAGE_DATASET and TEST_DATASET must match for joint-query training" >&2
  exit 2
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  if [[ "${LANGUAGE_DATASET}" == "scanrefer" ]]; then
    CHECKPOINT_PATH="${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_network_best_v19_rec025_0.581195_mask025_0.598233.pth"
  else
    echo "CHECKPOINT_PATH=<${LANGUAGE_DATASET} checkpoint> is required" >&2
    exit 2
  fi
fi

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
  EXPECTED_EVAL_SAMPLE_COUNT="${EXPECTED_EVAL_SAMPLE_COUNT:-128}"
elif [[ "${DEBUG}" != "0" ]]; then
  echo "DEBUG must be 0 or 1" >&2
  exit 2
else
  if [[ -z "${EXPECTED_EVAL_SAMPLE_COUNT}" ]]; then
    if [[ "${TEST_DATASET}" == "scanrefer" ]]; then
      EXPECTED_EVAL_SAMPLE_COUNT=9508
    else
      echo "EXPECTED_EVAL_SAMPLE_COUNT is required for ${TEST_DATASET}" >&2
      exit 2
    fi
  fi
fi
if [[ ! "${EXPECTED_EVAL_SAMPLE_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_EVAL_SAMPLE_COUNT must be a positive integer" >&2
  exit 2
fi

PERSISTENT_WORKER_ARGS=()
if [[ "${PERSISTENT_TRAIN_WORKERS}" == "1" ]]; then
  PERSISTENT_WORKER_ARGS=(--persistent_train_workers)
elif [[ "${PERSISTENT_TRAIN_WORKERS}" != "0" ]]; then
  echo "PERSISTENT_TRAIN_WORKERS must be 0 or 1" >&2
  exit 2
fi

MASK_CALIBRATION_ARGS=()
if [[ "${JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION}" == "1" ]]; then
  MASK_CALIBRATION_ARGS=(--joint_query_quality_use_mask_calibration)
elif [[ "${JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION must be 0 or 1" >&2
  exit 2
fi

SOURCE_MASK_EVIDENCE_ARGS=()
if [[ "${JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION}" != "1" ]]; then
    echo "source mask evidence requires mask calibration" >&2
    exit 2
  fi
  SOURCE_MASK_EVIDENCE_ARGS=(
    --joint_query_quality_use_source_mask_evidence
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE must be 0 or 1" >&2
  exit 2
fi

GATE_EVIDENCE_ARGS=()
if [[ "${JOINT_QUERY_QUALITY_USE_GATE_EVIDENCE}" == "1" ]]; then
  GATE_EVIDENCE_ARGS=(--joint_query_quality_use_gate_evidence)
elif [[ "${JOINT_QUERY_QUALITY_USE_GATE_EVIDENCE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_GATE_EVIDENCE must be 0 or 1" >&2
  exit 2
fi

SPATIAL_MASK_REFINER_ARGS=()
if [[ "${JOINT_QUERY_QUALITY_USE_SPATIAL_MASK_REFINER}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION}" != "1" ]]; then
    echo "spatial mask refiner requires mask calibration" >&2
    exit 2
  fi
  SPATIAL_MASK_REFINER_ARGS=(
    --joint_query_quality_use_spatial_mask_refiner
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_SPATIAL_MASK_REFINER}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_SPATIAL_MASK_REFINER must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${JOINT_QUERY_QUALITY_SPATIAL_MASK_HIDDEN_DIM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOINT_QUERY_QUALITY_SPATIAL_MASK_HIDDEN_DIM must be positive" >&2
  exit 2
fi

ADAPTIVE_SOURCE_MIXING_ARGS=()
if [[ "${JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING}" == "1" ]]; then
  source_pool="${JOINT_QUERY_QUALITY_SOURCE_NAMES:-${SOURCE_CHOICE_SELECTOR_SOURCES}}"
  IFS=',' read -r -a SOURCE_NAMES <<< "${source_pool}"
  if (( ${#SOURCE_NAMES[@]} < 2 )); then
    echo "adaptive source mixing requires at least two sources" >&2
    exit 2
  fi
  ADAPTIVE_SOURCE_MIXING_ARGS=(
    --joint_query_quality_use_adaptive_source_mixing
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING must be 0 or 1" >&2
  exit 2
fi

SOURCE_DISTRIBUTION_RELIABILITY_ARGS=()
if [[ "${JOINT_QUERY_QUALITY_USE_SOURCE_DISTRIBUTION_RELIABILITY}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING}" != "1" ]]; then
    echo "source distribution reliability requires adaptive source mixing" >&2
    exit 2
  fi
  SOURCE_DISTRIBUTION_RELIABILITY_ARGS=(
    --joint_query_quality_use_source_distribution_reliability
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_SOURCE_DISTRIBUTION_RELIABILITY}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_SOURCE_DISTRIBUTION_RELIABILITY must be 0 or 1" >&2
  exit 2
fi

JOINT_SOURCE_NAME_ARGS=()
if [[ -n "${JOINT_QUERY_QUALITY_SOURCE_NAMES}" ]]; then
  JOINT_SOURCE_NAME_ARGS=(
    --joint_query_quality_source_names "${JOINT_QUERY_QUALITY_SOURCE_NAMES}"
  )
fi

SACR_ARGS=()
if [[ "${USE_SACR_SOURCE}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING}" != "1" ]]; then
    echo "SACR requires adaptive source mixing" >&2
    exit 2
  fi
  case ",${JOINT_QUERY_QUALITY_SOURCE_NAMES}," in
    *,sacr_structured,*) ;;
    *)
      echo "SACR requires sacr_structured in JOINT_QUERY_QUALITY_SOURCE_NAMES" >&2
      exit 2
      ;;
  esac
  SACR_ARGS=(
    --use_sacr_source
    --sacr_hidden_dim "${SACR_HIDDEN_DIM}"
    --sacr_max_pairs "${SACR_MAX_PAIRS}"
    --sacr_top_m_targets "${SACR_TOP_M_TARGETS}"
    --sacr_top_k_anchors "${SACR_TOP_K_ANCHORS}"
    --sacr_geo_dim "${SACR_GEO_DIM}"
    --sacr_min_parse_confidence "${SACR_MIN_PARSE_CONFIDENCE}"
    --sacr_residual_scale_init "${SACR_RESIDUAL_SCALE_INIT}"
  )
elif [[ "${USE_SACR_SOURCE}" != "0" ]]; then
  echo "USE_SACR_SOURCE must be 0 or 1" >&2
  exit 2
fi

STAGE_ARGS=()
case "${MODEL_STAGE}" in
  two)
    STAGE_ARGS=(--butd)
    ;;
  single)
    STAGE_ARGS=(--joint_det --skip_missing_superpoints)
    ;;
  *)
    echo "MODEL_STAGE must be two or single" >&2
    exit 2
    ;;
esac

SOURCE_ARBITER_ARGS=()
case "${SOURCE_ARBITER}" in
  moe)
    SOURCE_ARBITER_ARGS=(--use_source_moe --source_moe_use_fallback_gate)
    ;;
  selector)
    if [[ "${JOINT_QUERY_QUALITY_USE_GATE_EVIDENCE}" == "1" ]]; then
      echo "selector source arbiter cannot provide gate evidence" >&2
      exit 2
    fi
    SOURCE_ARBITER_ARGS=(
      --use_source_choice_selector
      --source_choice_selector_sources "${SOURCE_CHOICE_SELECTOR_SOURCES}"
      --source_choice_selector_hidden_dim "${SOURCE_CHOICE_SELECTOR_HIDDEN_DIM}"
    )
    ;;
  *)
    echo "SOURCE_ARBITER must be moe or selector" >&2
    exit 2
    ;;
esac

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
  --dataset "${LANGUAGE_DATASET}" --test_dataset "${TEST_DATASET}" \
  --detect_intermediate \
  --use_soft_token_loss --use_contrastive_align \
  --log_dir "${LOG_DIR}" \
  --lr_decay_epochs 50 75 \
  --pp_checkpoint "${PP_CHECKPOINT}" \
  "${STAGE_ARGS[@]}" --self_attend --augment_det \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --start_epoch "${START_EPOCH}" --max_epoch "${MAX_EPOCH}" \
  --model MCLN --exp "${EXP}" \
  "${SOURCE_ARBITER_ARGS[@]}" \
  --source_moe_balance_loss_weight 0.0 \
  --source_moe_rank_loss_weight 0.0 \
  --source_moe_gate_loss_weight 0.0 \
  --eval_use_selector_choice_scores \
  --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}" \
  --use_joint_query_quality_reranker \
  --joint_query_quality_train_only \
  --joint_query_quality_lr "${JOINT_QUERY_QUALITY_LR}" \
  --joint_query_quality_hidden_dim "${JOINT_QUERY_QUALITY_HIDDEN_DIM}" \
  --joint_query_quality_heads "${JOINT_QUERY_QUALITY_HEADS}" \
  --joint_query_quality_layers "${JOINT_QUERY_QUALITY_LAYERS}" \
  --joint_query_quality_dropout "${JOINT_QUERY_QUALITY_DROPOUT}" \
  --joint_query_quality_max_delta "${JOINT_QUERY_QUALITY_MAX_DELTA}" \
  --joint_query_quality_mask_weight "${JOINT_QUERY_QUALITY_MASK_WEIGHT}" \
  --joint_query_quality_score_weight "${JOINT_QUERY_QUALITY_SCORE_WEIGHT}" \
  --joint_query_quality_max_mask_alpha_delta "${JOINT_QUERY_QUALITY_MAX_MASK_ALPHA_DELTA}" \
  --joint_query_quality_max_mask_logit_bias "${JOINT_QUERY_QUALITY_MAX_MASK_LOGIT_BIAS}" \
  "${MASK_CALIBRATION_ARGS[@]}" \
  "${SOURCE_MASK_EVIDENCE_ARGS[@]}" \
  "${GATE_EVIDENCE_ARGS[@]}" \
  "${SPATIAL_MASK_REFINER_ARGS[@]}" \
  --joint_query_quality_spatial_mask_hidden_dim "${JOINT_QUERY_QUALITY_SPATIAL_MASK_HIDDEN_DIM}" \
  --joint_query_quality_max_spatial_mask_delta "${JOINT_QUERY_QUALITY_MAX_SPATIAL_MASK_DELTA}" \
  "${ADAPTIVE_SOURCE_MIXING_ARGS[@]}" \
  "${SOURCE_DISTRIBUTION_RELIABILITY_ARGS[@]}" \
  "${JOINT_SOURCE_NAME_ARGS[@]}" \
  --joint_query_quality_max_source_mix_delta "${JOINT_QUERY_QUALITY_MAX_SOURCE_MIX_DELTA}" \
  --joint_query_quality_source_mix_temperature "${JOINT_QUERY_QUALITY_SOURCE_MIX_TEMPERATURE}" \
  --joint_query_quality_loss_weight "${JOINT_QUERY_QUALITY_LOSS_WEIGHT}" \
  --joint_query_quality_temperature "${JOINT_QUERY_QUALITY_TEMPERATURE}" \
  --joint_query_quality_aux_loss_weight "${JOINT_QUERY_QUALITY_AUX_LOSS_WEIGHT}" \
  --joint_query_quality_anchor_loss_weight "${JOINT_QUERY_QUALITY_ANCHOR_LOSS_WEIGHT}" \
  --joint_query_quality_anchor_margin "${JOINT_QUERY_QUALITY_ANCHOR_MARGIN}" \
  --joint_query_quality_source_mix_loss_weight "${JOINT_QUERY_QUALITY_SOURCE_MIX_LOSS_WEIGHT}" \
  --joint_query_quality_source_mix_alignment_temperature "${JOINT_QUERY_QUALITY_SOURCE_MIX_ALIGNMENT_TEMPERATURE}" \
  --joint_query_quality_source_mix_query_focus_weight "${JOINT_QUERY_QUALITY_SOURCE_MIX_QUERY_FOCUS_WEIGHT}" \
  --joint_query_quality_candidate_mask_loss_weight "${JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT}" \
  --joint_query_quality_candidate_lovasz_loss_weight "${JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT}" \
  --joint_query_quality_candidate_mask_top_k "${JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K}" \
  "${SACR_ARGS[@]}" \
  --checkpoint_metric_retention \
  "${DEBUG_ARGS[@]}" \
  "$@"
