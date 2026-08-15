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
AUGMENT_DET="${AUGMENT_DET:-1}"
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
JOINT_QUERY_QUALITY_DIRECT_RESIDUAL_SCALE="${JOINT_QUERY_QUALITY_DIRECT_RESIDUAL_SCALE:-1.0}"
JOINT_QUERY_QUALITY_USE_METRIC_ALIGNED_UTILITY="${JOINT_QUERY_QUALITY_USE_METRIC_ALIGNED_UTILITY:-0}"
JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE="${JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE:-0}"
JOINT_QUERY_QUALITY_CANDIDATE_PROMOTION_MARGIN="${JOINT_QUERY_QUALITY_CANDIDATE_PROMOTION_MARGIN:-0.0}"
JOINT_QUERY_QUALITY_USE_PARENT_TRANSITION_ADVANTAGE="${JOINT_QUERY_QUALITY_USE_PARENT_TRANSITION_ADVANTAGE:-0}"
JOINT_QUERY_QUALITY_USE_DECOMPOSED_TRANSITION_ADVANTAGE="${JOINT_QUERY_QUALITY_USE_DECOMPOSED_TRANSITION_ADVANTAGE:-0}"
JOINT_QUERY_QUALITY_USE_SETWISE_TIER_ADVANTAGE="${JOINT_QUERY_QUALITY_USE_SETWISE_TIER_ADVANTAGE:-0}"
JOINT_QUERY_QUALITY_USE_DECOUPLED_SETWISE_HEADS="${JOINT_QUERY_QUALITY_USE_DECOUPLED_SETWISE_HEADS:-0}"
JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_SAFETY="${JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_SAFETY:-0}"
JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_RISK_BOUND="${JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_RISK_BOUND:-0}"
JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_VETO_GATE="${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_VETO_GATE:-0}"
JOINT_QUERY_QUALITY_USE_COST_CALIBRATED_SETWISE_RISK_BOUND="${JOINT_QUERY_QUALITY_USE_COST_CALIBRATED_SETWISE_RISK_BOUND:-0}"
JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_QUANTILE_BOUND="${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_QUANTILE_BOUND:-0}"
JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_PAIRWISE_ORDER="${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_PAIRWISE_ORDER:-0}"
JOINT_QUERY_QUALITY_USE_PROPOSAL_CONDITIONED_SAFETY="${JOINT_QUERY_QUALITY_USE_PROPOSAL_CONDITIONED_SAFETY:-0}"
JOINT_QUERY_QUALITY_USE_PARENT_REFERENCED_SAFETY="${JOINT_QUERY_QUALITY_USE_PARENT_REFERENCED_SAFETY:-0}"
JOINT_QUERY_QUALITY_USE_COUPLED_SAFE_REPAIR_WITNESS="${JOINT_QUERY_QUALITY_USE_COUPLED_SAFE_REPAIR_WITNESS:-0}"
JOINT_QUERY_QUALITY_USE_BIDIRECTIONAL_COUPLED_BOUNDARY="${JOINT_QUERY_QUALITY_USE_BIDIRECTIONAL_COUPLED_BOUNDARY:-0}"
JOINT_QUERY_QUALITY_USE_CENTERED_COUPLED_SEPARATION="${JOINT_QUERY_QUALITY_USE_CENTERED_COUPLED_SEPARATION:-0}"
JOINT_QUERY_QUALITY_USE_HAZARD_CONDITIONED_COUPLED_SEPARATION="${JOINT_QUERY_QUALITY_USE_HAZARD_CONDITIONED_COUPLED_SEPARATION:-0}"
JOINT_QUERY_QUALITY_USE_MONOTONIC_BOX_SAFETY_FOLDING="${JOINT_QUERY_QUALITY_USE_MONOTONIC_BOX_SAFETY_FOLDING:-0}"
JOINT_QUERY_QUALITY_USE_SAME_CANDIDATE_BRANCHWISE_WITNESS="${JOINT_QUERY_QUALITY_USE_SAME_CANDIDATE_BRANCHWISE_WITNESS:-0}"
JOINT_QUERY_QUALITY_USE_PARENT_NON_DEGRADATION_CERTIFICATE="${JOINT_QUERY_QUALITY_USE_PARENT_NON_DEGRADATION_CERTIFICATE:-0}"
JOINT_QUERY_QUALITY_USE_CRITERION_RESPONSIBLE_HAZARD_ATTRIBUTION="${JOINT_QUERY_QUALITY_USE_CRITERION_RESPONSIBLE_HAZARD_ATTRIBUTION:-0}"
JOINT_QUERY_QUALITY_USE_INDEPENDENT_JOINT_HAZARD_CERTIFICATE="${JOINT_QUERY_QUALITY_USE_INDEPENDENT_JOINT_HAZARD_CERTIFICATE:-0}"
JOINT_QUERY_QUALITY_USE_FROZEN_RAW_JOINT_HAZARD_FEATURES="${JOINT_QUERY_QUALITY_USE_FROZEN_RAW_JOINT_HAZARD_FEATURES:-0}"
JOINT_QUERY_QUALITY_USE_FACTORIZED_HIT_ADVANTAGE="${JOINT_QUERY_QUALITY_USE_FACTORIZED_HIT_ADVANTAGE:-0}"
JOINT_QUERY_QUALITY_USE_FACTORIZED_NESTED_DOMINANCE="${JOINT_QUERY_QUALITY_USE_FACTORIZED_NESTED_DOMINANCE:-0}"
JOINT_QUERY_QUALITY_FACTORIZED_HIT_BREAK_COST="${JOINT_QUERY_QUALITY_FACTORIZED_HIT_BREAK_COST:-4.0}"
JOINT_QUERY_QUALITY_PARENT_TRANSITION_BREAK_COST="${JOINT_QUERY_QUALITY_PARENT_TRANSITION_BREAK_COST:-4.0}"
JOINT_QUERY_QUALITY_PARENT_TRANSITION_CANDIDATE_TOP_K="${JOINT_QUERY_QUALITY_PARENT_TRANSITION_CANDIDATE_TOP_K:-0}"
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
JOINT_QUERY_QUALITY_BIDIRECTIONAL_ANCHOR="${JOINT_QUERY_QUALITY_BIDIRECTIONAL_ANCHOR:-0}"
JOINT_QUERY_QUALITY_ANCHOR_MARGIN_050="${JOINT_QUERY_QUALITY_ANCHOR_MARGIN_050:-0.10}"
JOINT_QUERY_QUALITY_METRIC_UTILITY_TEMPERATURE="${JOINT_QUERY_QUALITY_METRIC_UTILITY_TEMPERATURE:-0.05}"
JOINT_QUERY_QUALITY_PAIRWISE_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_PAIRWISE_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_LISTWISE_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_LISTWISE_LOSS_WEIGHT:-1.0}"
JOINT_QUERY_QUALITY_TRANSITION_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_TRANSITION_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_SETWISE_REPAIR_BOUNDARY_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_SETWISE_REPAIR_BOUNDARY_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_SETWISE_NEGATIVE_TAIL_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_SETWISE_NEGATIVE_TAIL_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_SETWISE_RANK_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_SETWISE_RANK_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_SETWISE_DENSE_SAFETY_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_SETWISE_DENSE_SAFETY_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_SETWISE_BALANCED_SAFETY_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_SETWISE_BALANCED_SAFETY_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_SETWISE_FACTORIZED_SAFETY_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_SETWISE_FACTORIZED_SAFETY_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_SETWISE_FACTORIZED_RISK_BOUND_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_SETWISE_FACTORIZED_RISK_BOUND_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_FACTORIZED_HIT_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_FACTORIZED_HIT_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_FACTORIZED_PAIR_LOSS_WEIGHT="${JOINT_QUERY_QUALITY_FACTORIZED_PAIR_LOSS_WEIGHT:-0.0}"
JOINT_QUERY_QUALITY_TRANSITION_BREAK_COST="${JOINT_QUERY_QUALITY_TRANSITION_BREAK_COST:-4.0}"
JOINT_QUERY_QUALITY_TRANSITION_NEUTRAL_WEIGHT="${JOINT_QUERY_QUALITY_TRANSITION_NEUTRAL_WEIGHT:-0.25}"
JOINT_QUERY_QUALITY_DEPLOY_CANDIDATE_TOP_K="${JOINT_QUERY_QUALITY_DEPLOY_CANDIDATE_TOP_K:-0}"
JOINT_QUERY_QUALITY_SOURCE_CANDIDATE_TOP_K="${JOINT_QUERY_QUALITY_SOURCE_CANDIDATE_TOP_K:-0}"
JOINT_QUERY_QUALITY_ORACLE_CANDIDATE_TOP_K="${JOINT_QUERY_QUALITY_ORACLE_CANDIDATE_TOP_K:-0}"
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

AUGMENT_DET_ARGS=()
if [[ "${AUGMENT_DET}" == "1" ]]; then
  AUGMENT_DET_ARGS=(--augment_det)
elif [[ "${AUGMENT_DET}" != "0" ]]; then
  echo "AUGMENT_DET must be 0 or 1" >&2
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

METRIC_UTILITY_ARGS=()
if [[ "${JOINT_QUERY_QUALITY_USE_METRIC_ALIGNED_UTILITY}" == "1" ]]; then
  METRIC_UTILITY_ARGS=(
    --joint_query_quality_use_metric_aligned_utility
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_METRIC_ALIGNED_UTILITY}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_METRIC_ALIGNED_UTILITY must be 0 or 1" >&2
  exit 2
fi

PARENT_SCORE_ARGS=()
if [[ "${JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE}" == "1" ]]; then
  PARENT_SCORE_ARGS=(--joint_query_quality_preserve_parent_score)
elif [[ "${JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE}" != "1"       && "${JOINT_QUERY_QUALITY_CANDIDATE_PROMOTION_MARGIN}" != "0"       && "${JOINT_QUERY_QUALITY_CANDIDATE_PROMOTION_MARGIN}" != "0.0" ]]; then
  echo "candidate promotion margin requires preserved parent score" >&2
  exit 2
fi

PARENT_TRANSITION_ARGS=()
if [[ "${JOINT_QUERY_QUALITY_USE_PARENT_TRANSITION_ADVANTAGE}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE}" != "1" ]]; then
    echo "parent transition advantage requires preserved parent score" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS=(
    --joint_query_quality_use_parent_transition_advantage
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_PARENT_TRANSITION_ADVANTAGE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_PARENT_TRANSITION_ADVANTAGE must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_DECOMPOSED_TRANSITION_ADVANTAGE}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE}" != "1" ]]; then
    echo "decomposed transition advantage requires preserved parent score" >&2
    exit 2
  fi
  if [[ "${JOINT_QUERY_QUALITY_USE_PARENT_TRANSITION_ADVANTAGE}" == "1" ]]; then
    echo "transition advantage modes are exclusive" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_decomposed_transition_advantage
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_DECOMPOSED_TRANSITION_ADVANTAGE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_DECOMPOSED_TRANSITION_ADVANTAGE must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_TIER_ADVANTAGE}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE}" != "1" ]]; then
    echo "setwise tier advantage requires preserved parent score" >&2
    exit 2
  fi
  if [[ "${JOINT_QUERY_QUALITY_USE_PARENT_TRANSITION_ADVANTAGE}" == "1"       || "${JOINT_QUERY_QUALITY_USE_DECOMPOSED_TRANSITION_ADVANTAGE}" == "1" ]]; then
    echo "transition advantage modes are exclusive" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_setwise_tier_advantage
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_TIER_ADVANTAGE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_SETWISE_TIER_ADVANTAGE must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_DECOUPLED_SETWISE_HEADS}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_TIER_ADVANTAGE}" != "1" ]]; then
    echo "decoupled setwise heads require setwise tier advantage" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_decoupled_setwise_heads
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_DECOUPLED_SETWISE_HEADS}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_DECOUPLED_SETWISE_HEADS must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_SAFETY}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_DECOUPLED_SETWISE_HEADS}" != "1" ]]; then
    echo "factorized setwise safety requires decoupled setwise heads" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_factorized_setwise_safety
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_SAFETY}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_SAFETY must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_RISK_BOUND}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_SAFETY}" != "1" ]]; then
    echo "factorized setwise risk bound requires factorized safety" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_factorized_setwise_risk_bound
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_RISK_BOUND}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_RISK_BOUND must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_VETO_GATE}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_DECOUPLED_SETWISE_HEADS}" != "1" ]]; then
    echo "setwise safety veto gate requires decoupled setwise heads" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_setwise_safety_veto_gate
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_VETO_GATE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_VETO_GATE must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_COST_CALIBRATED_SETWISE_RISK_BOUND}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_RISK_BOUND}" != "1" ]]; then
    echo "cost-calibrated risk bound requires factorized risk bound" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_cost_calibrated_setwise_risk_bound
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_COST_CALIBRATED_SETWISE_RISK_BOUND}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_COST_CALIBRATED_SETWISE_RISK_BOUND must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_QUANTILE_BOUND}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_SETWISE_RISK_BOUND}" != "1" ]]; then
    echo "safety-slack quantile bound requires factorized risk bound" >&2
    exit 2
  fi
  if [[ "${JOINT_QUERY_QUALITY_USE_COST_CALIBRATED_SETWISE_RISK_BOUND}" == "1" ]]; then
    echo "safety-slack quantile bound and cost calibration are exclusive" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_setwise_safety_slack_quantile_bound
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_QUANTILE_BOUND}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_QUANTILE_BOUND must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_PAIRWISE_ORDER}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_QUANTILE_BOUND}" != "1" ]]; then
    echo "safety-slack pairwise order requires slack quantile bound" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_setwise_safety_slack_pairwise_order
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_PAIRWISE_ORDER}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_PAIRWISE_ORDER must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_PROPOSAL_CONDITIONED_SAFETY}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_PAIRWISE_ORDER}" != "1" ]]; then
    echo "proposal-conditioned safety requires safety-slack pairwise order" >&2
    exit 2
  fi
  if [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_VETO_GATE}" != "1" ]]; then
    echo "proposal-conditioned safety requires the safety veto gate" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_proposal_conditioned_safety
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_PROPOSAL_CONDITIONED_SAFETY}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_PROPOSAL_CONDITIONED_SAFETY must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_PARENT_REFERENCED_SAFETY}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_SETWISE_SAFETY_SLACK_PAIRWISE_ORDER}" != "1" ]]; then
    echo "parent-referenced safety requires safety-slack pairwise order" >&2
    exit 2
  fi
  if [[ "${JOINT_QUERY_QUALITY_USE_PROPOSAL_CONDITIONED_SAFETY}" == "1" ]]; then
    echo "parent-referenced and proposal-conditioned safety are exclusive" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_parent_referenced_safety
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_PARENT_REFERENCED_SAFETY}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_PARENT_REFERENCED_SAFETY must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_COUPLED_SAFE_REPAIR_WITNESS}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_PARENT_REFERENCED_SAFETY}" != "1" ]]; then
    echo "coupled safe-repair witness requires parent-referenced safety" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_coupled_safe_repair_witness
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_COUPLED_SAFE_REPAIR_WITNESS}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_COUPLED_SAFE_REPAIR_WITNESS must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_BIDIRECTIONAL_COUPLED_BOUNDARY}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_COUPLED_SAFE_REPAIR_WITNESS}" != "1" ]]; then
    echo "bidirectional coupled boundary requires coupled safe-repair witness" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_bidirectional_coupled_boundary
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_BIDIRECTIONAL_COUPLED_BOUNDARY}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_BIDIRECTIONAL_COUPLED_BOUNDARY must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_CENTERED_COUPLED_SEPARATION}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_BIDIRECTIONAL_COUPLED_BOUNDARY}" != "1" ]]; then
    echo "centered coupled separation requires bidirectional boundary" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_centered_coupled_separation
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_CENTERED_COUPLED_SEPARATION}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_CENTERED_COUPLED_SEPARATION must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_HAZARD_CONDITIONED_COUPLED_SEPARATION}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_CENTERED_COUPLED_SEPARATION}" != "1" ]]; then
    echo "hazard-conditioned separation requires centered separation" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_hazard_conditioned_coupled_separation
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_HAZARD_CONDITIONED_COUPLED_SEPARATION}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_HAZARD_CONDITIONED_COUPLED_SEPARATION must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_MONOTONIC_BOX_SAFETY_FOLDING}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_HAZARD_CONDITIONED_COUPLED_SEPARATION}" != "1" ]]; then
    echo "monotonic box-safety folding requires hazard-conditioned separation" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_monotonic_box_safety_folding
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_MONOTONIC_BOX_SAFETY_FOLDING}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_MONOTONIC_BOX_SAFETY_FOLDING must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_SAME_CANDIDATE_BRANCHWISE_WITNESS}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_MONOTONIC_BOX_SAFETY_FOLDING}" != "1" ]]; then
    echo "same-candidate branchwise witness requires monotonic box-safety folding" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_same_candidate_branchwise_witness
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_SAME_CANDIDATE_BRANCHWISE_WITNESS}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_SAME_CANDIDATE_BRANCHWISE_WITNESS must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_PARENT_NON_DEGRADATION_CERTIFICATE}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_SAME_CANDIDATE_BRANCHWISE_WITNESS}" != "1" ]]; then
    echo "parent non-degradation certificate requires same-candidate branchwise witness" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_parent_non_degradation_certificate
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_PARENT_NON_DEGRADATION_CERTIFICATE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_PARENT_NON_DEGRADATION_CERTIFICATE must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_CRITERION_RESPONSIBLE_HAZARD_ATTRIBUTION}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_PARENT_NON_DEGRADATION_CERTIFICATE}" != "1" ]]; then
    echo "criterion-responsible hazard attribution requires parent non-degradation certificate" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_criterion_responsible_hazard_attribution
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_CRITERION_RESPONSIBLE_HAZARD_ATTRIBUTION}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_CRITERION_RESPONSIBLE_HAZARD_ATTRIBUTION must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_INDEPENDENT_JOINT_HAZARD_CERTIFICATE}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_PARENT_NON_DEGRADATION_CERTIFICATE}" != "1" ]]; then
    echo "independent joint-hazard certificate requires parent non-degradation certificate" >&2
    exit 2
  fi
  if [[ "${JOINT_QUERY_QUALITY_USE_CRITERION_RESPONSIBLE_HAZARD_ATTRIBUTION}" == "1" ]]; then
    echo "independent joint-hazard certificate and criterion-responsible attribution are exclusive" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_independent_joint_hazard_certificate
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_INDEPENDENT_JOINT_HAZARD_CERTIFICATE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_INDEPENDENT_JOINT_HAZARD_CERTIFICATE must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_FROZEN_RAW_JOINT_HAZARD_FEATURES}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_INDEPENDENT_JOINT_HAZARD_CERTIFICATE}" != "1" ]]; then
    echo "frozen raw joint-hazard features require independent joint-hazard certificate" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_frozen_raw_joint_hazard_features
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_FROZEN_RAW_JOINT_HAZARD_FEATURES}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_FROZEN_RAW_JOINT_HAZARD_FEATURES must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_HIT_ADVANTAGE}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_PRESERVE_PARENT_SCORE}" != "1" ]]; then
    echo "factorized hit advantage requires preserved parent score" >&2
    exit 2
  fi
  if [[ "${JOINT_QUERY_QUALITY_USE_PARENT_TRANSITION_ADVANTAGE}" == "1"       || "${JOINT_QUERY_QUALITY_USE_DECOMPOSED_TRANSITION_ADVANTAGE}" == "1"       || "${JOINT_QUERY_QUALITY_USE_SETWISE_TIER_ADVANTAGE}" == "1" ]]; then
    echo "transition advantage modes are exclusive" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_factorized_hit_advantage
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_HIT_ADVANTAGE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_FACTORIZED_HIT_ADVANTAGE must be 0 or 1" >&2
  exit 2
fi
if [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_NESTED_DOMINANCE}" == "1" ]]; then
  if [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_HIT_ADVANTAGE}" != "1" ]]; then
    echo "factorized nested dominance requires factorized hit advantage" >&2
    exit 2
  fi
  PARENT_TRANSITION_ARGS+=(
    --joint_query_quality_use_factorized_nested_dominance
  )
elif [[ "${JOINT_QUERY_QUALITY_USE_FACTORIZED_NESTED_DOMINANCE}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_USE_FACTORIZED_NESTED_DOMINANCE must be 0 or 1" >&2
  exit 2
fi

BIDIRECTIONAL_ANCHOR_ARGS=()
if [[ "${JOINT_QUERY_QUALITY_BIDIRECTIONAL_ANCHOR}" == "1" ]]; then
  BIDIRECTIONAL_ANCHOR_ARGS=(
    --joint_query_quality_bidirectional_anchor
  )
elif [[ "${JOINT_QUERY_QUALITY_BIDIRECTIONAL_ANCHOR}" != "0" ]]; then
  echo "JOINT_QUERY_QUALITY_BIDIRECTIONAL_ANCHOR must be 0 or 1" >&2
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
  "${STAGE_ARGS[@]}" --self_attend "${AUGMENT_DET_ARGS[@]}" \
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
  --joint_query_quality_direct_residual_scale "${JOINT_QUERY_QUALITY_DIRECT_RESIDUAL_SCALE}" \
  "${METRIC_UTILITY_ARGS[@]}" \
  "${PARENT_SCORE_ARGS[@]}" \
  "${PARENT_TRANSITION_ARGS[@]}" \
  --joint_query_quality_candidate_promotion_margin "${JOINT_QUERY_QUALITY_CANDIDATE_PROMOTION_MARGIN}" \
  --joint_query_quality_factorized_hit_break_cost "${JOINT_QUERY_QUALITY_FACTORIZED_HIT_BREAK_COST}" \
  --joint_query_quality_parent_transition_break_cost "${JOINT_QUERY_QUALITY_PARENT_TRANSITION_BREAK_COST}" \
  --joint_query_quality_parent_transition_candidate_top_k "${JOINT_QUERY_QUALITY_PARENT_TRANSITION_CANDIDATE_TOP_K}" \
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
  "${BIDIRECTIONAL_ANCHOR_ARGS[@]}" \
  --joint_query_quality_anchor_margin_050 "${JOINT_QUERY_QUALITY_ANCHOR_MARGIN_050}" \
  --joint_query_quality_metric_utility_temperature "${JOINT_QUERY_QUALITY_METRIC_UTILITY_TEMPERATURE}" \
  --joint_query_quality_pairwise_loss_weight "${JOINT_QUERY_QUALITY_PAIRWISE_LOSS_WEIGHT}" \
  --joint_query_quality_listwise_loss_weight "${JOINT_QUERY_QUALITY_LISTWISE_LOSS_WEIGHT}" \
  --joint_query_quality_transition_loss_weight "${JOINT_QUERY_QUALITY_TRANSITION_LOSS_WEIGHT}" \
  --joint_query_quality_setwise_repair_boundary_loss_weight "${JOINT_QUERY_QUALITY_SETWISE_REPAIR_BOUNDARY_LOSS_WEIGHT}" \
  --joint_query_quality_setwise_negative_tail_loss_weight "${JOINT_QUERY_QUALITY_SETWISE_NEGATIVE_TAIL_LOSS_WEIGHT}" \
  --joint_query_quality_setwise_rank_loss_weight "${JOINT_QUERY_QUALITY_SETWISE_RANK_LOSS_WEIGHT}" \
  --joint_query_quality_setwise_dense_safety_loss_weight "${JOINT_QUERY_QUALITY_SETWISE_DENSE_SAFETY_LOSS_WEIGHT}" \
  --joint_query_quality_setwise_balanced_safety_loss_weight "${JOINT_QUERY_QUALITY_SETWISE_BALANCED_SAFETY_LOSS_WEIGHT}" \
  --joint_query_quality_setwise_factorized_safety_loss_weight "${JOINT_QUERY_QUALITY_SETWISE_FACTORIZED_SAFETY_LOSS_WEIGHT}" \
  --joint_query_quality_setwise_factorized_risk_bound_loss_weight "${JOINT_QUERY_QUALITY_SETWISE_FACTORIZED_RISK_BOUND_LOSS_WEIGHT}" \
  --joint_query_quality_factorized_hit_loss_weight "${JOINT_QUERY_QUALITY_FACTORIZED_HIT_LOSS_WEIGHT}" \
  --joint_query_quality_factorized_pair_loss_weight "${JOINT_QUERY_QUALITY_FACTORIZED_PAIR_LOSS_WEIGHT}" \
  --joint_query_quality_transition_break_cost "${JOINT_QUERY_QUALITY_TRANSITION_BREAK_COST}" \
  --joint_query_quality_transition_neutral_weight "${JOINT_QUERY_QUALITY_TRANSITION_NEUTRAL_WEIGHT}" \
  --joint_query_quality_deploy_candidate_top_k "${JOINT_QUERY_QUALITY_DEPLOY_CANDIDATE_TOP_K}" \
  --joint_query_quality_source_candidate_top_k "${JOINT_QUERY_QUALITY_SOURCE_CANDIDATE_TOP_K}" \
  --joint_query_quality_oracle_candidate_top_k "${JOINT_QUERY_QUALITY_ORACLE_CANDIDATE_TOP_K}" \
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
