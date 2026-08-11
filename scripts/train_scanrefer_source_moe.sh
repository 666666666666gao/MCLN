#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT%/}/output/source_moe}"
PP_CHECKPOINT="${PP_CHECKPOINT:-${DATA_ROOT%/}/gf_detector_l6o256.pth}"
BASELINE_CHECKPOINT="${DATA_ROOT%/}/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_PORT="${MASTER_PORT:-4462}"
PHASE="${PHASE:-router}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_FREQ="${VAL_FREQ:-1}"
EXPECTED_EVAL_SAMPLE_COUNT="${EXPECTED_EVAL_SAMPLE_COUNT:-9508}"
DEBUG="${DEBUG:-0}"
PRINT_FREQ="${PRINT_FREQ:-500}"
DECODER_LR="${DECODER_LR:-2e-5}"
BACKBONE_LR="${BACKBONE_LR:-2e-4}"
TEXT_ENCODER_LR="${TEXT_ENCODER_LR:-3e-6}"
if [[ -n "${LR_DECAY_EPOCHS:-}" ]]; then
  LR_DECAY_EPOCHS_TEXT="${LR_DECAY_EPOCHS}"
elif [[ "${PHASE}" == "continue" ]]; then
  # A fresh optimizer starts its scheduler at step zero.  Relative milestone
  # 3, together with warmup_epoch=-1, decays after four completed epochs so
  # an epoch-72 continuation enters epoch 76 at 0.1x LR.
  LR_DECAY_EPOCHS_TEXT="3"
else
  LR_DECAY_EPOCHS_TEXT="50 75"
fi
SOURCE_MOE_LR="${SOURCE_MOE_LR:-3e-4}"
SOURCE_MOE_TOP_K="${SOURCE_MOE_TOP_K:-1}"
SOURCE_MOE_BALANCE_WEIGHT="${SOURCE_MOE_BALANCE_WEIGHT:-0.01}"
SOURCE_MOE_RANK_WEIGHT="${SOURCE_MOE_RANK_WEIGHT:-1.0}"
SOURCE_MOE_MASK_RANK_WEIGHT="${SOURCE_MOE_MASK_RANK_WEIGHT:-0.25}"
SOURCE_MOE_TEMPERATURE="${SOURCE_MOE_TEMPERATURE:-0.1}"
SOURCE_MOE_ANCHOR_WEIGHT="${SOURCE_MOE_ANCHOR_WEIGHT:-1.0}"
SOURCE_MOE_ANCHOR_MARGIN="${SOURCE_MOE_ANCHOR_MARGIN:-0.05}"
SOURCE_MOE_QUERY_MAX_DELTA="${SOURCE_MOE_QUERY_MAX_DELTA:-0.25}"
SOURCE_MOE_USE_FALLBACK_GATE="${SOURCE_MOE_USE_FALLBACK_GATE:-0}"
SOURCE_MOE_GATE_LR="${SOURCE_MOE_GATE_LR:-3e-4}"
SOURCE_MOE_GATE_RESUME_OPTIMIZER="${SOURCE_MOE_GATE_RESUME_OPTIMIZER:-0}"
SOURCE_MOE_GATE_NEW_HEADS_ONLY="${SOURCE_MOE_GATE_NEW_HEADS_ONLY:-0}"
SOURCE_MOE_GATE_HIDDEN_DIM="${SOURCE_MOE_GATE_HIDDEN_DIM:-128}"
SOURCE_MOE_GATE_TOP_K="${SOURCE_MOE_GATE_TOP_K:-8}"
SOURCE_MOE_GATE_BREAK_COST="${SOURCE_MOE_GATE_BREAK_COST:-2.0}"
SOURCE_MOE_GATE_DECISION_MARGIN="${SOURCE_MOE_GATE_DECISION_MARGIN:-0.0}"
SOURCE_MOE_GATE_MASK_UTILITY_WEIGHT="${SOURCE_MOE_GATE_MASK_UTILITY_WEIGHT:-0.25}"
SOURCE_MOE_GATE_UNCERTAINTY_WEIGHT="${SOURCE_MOE_GATE_UNCERTAINTY_WEIGHT:-0.0}"
SOURCE_MOE_GATE_USE_EVIDENCE_FEATURES="${SOURCE_MOE_GATE_USE_EVIDENCE_FEATURES:-0}"
SOURCE_MOE_GATE_CONTEXT_LAYERS="${SOURCE_MOE_GATE_CONTEXT_LAYERS:-0}"
SOURCE_MOE_GATE_CONTEXT_HEADS="${SOURCE_MOE_GATE_CONTEXT_HEADS:-4}"
SOURCE_MOE_GATE_CONTEXT_DROPOUT="${SOURCE_MOE_GATE_CONTEXT_DROPOUT:-0.1}"
SOURCE_MOE_GATE_ACTION_MODE="${SOURCE_MOE_GATE_ACTION_MODE:-}"
SOURCE_MOE_GATE_LOSS_WEIGHT="${SOURCE_MOE_GATE_LOSS_WEIGHT:-0.0}"
SOURCE_MOE_GATE_MASK_LOSS_WEIGHT="${SOURCE_MOE_GATE_MASK_LOSS_WEIGHT:-0.25}"
SOURCE_MOE_GATE_FOCAL_GAMMA="${SOURCE_MOE_GATE_FOCAL_GAMMA:-2.0}"
SOURCE_MOE_GATE_FALSE_OVERRIDE_WEIGHT="${SOURCE_MOE_GATE_FALSE_OVERRIDE_WEIGHT:-2.0}"
SOURCE_MOE_GATE_OBJECTIVE_EXPLICIT=0
if [[ "${SOURCE_MOE_GATE_OBJECTIVE+x}" == "x" ]]; then
  SOURCE_MOE_GATE_OBJECTIVE_EXPLICIT=1
fi
SOURCE_MOE_GATE_OBJECTIVE="${SOURCE_MOE_GATE_OBJECTIVE:-balanced_focal}"
SOURCE_MOE_GATE_SETWISE_TEMPERATURE="${SOURCE_MOE_GATE_SETWISE_TEMPERATURE:-0.0}"
SOURCE_MOE_GATE_BOUNDARY_LOSS_WEIGHT="${SOURCE_MOE_GATE_BOUNDARY_LOSS_WEIGHT:-0.0}"
JOINT_RESUME="${JOINT_RESUME:-0}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
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
GATE_ENABLE_ARGS=()
GATE_EVIDENCE_ARGS=()
GATE_ACTION_ARGS=()
GATE_OBJECTIVE_ARGS=()
GATE_RESUME_ARGS=()
GATE_NEW_HEADS_ARGS=()
CHECKPOINT_START_ARGS=()
RETENTION_ARGS=()
DEBUG_ARGS=()
DATASET_ARGS=(--dataset scanrefer)
read -r -a LR_DECAY_EPOCHS_ARGS <<< "${LR_DECAY_EPOCHS_TEXT}"
if [[ "${#LR_DECAY_EPOCHS_ARGS[@]}" -eq 0 ]]; then
  echo "LR_DECAY_EPOCHS must contain at least one non-negative integer" >&2
  exit 2
fi
for decay_epoch in "${LR_DECAY_EPOCHS_ARGS[@]}"; do
  if [[ ! "${decay_epoch}" =~ ^[0-9]+$ ]]; then
    echo "LR_DECAY_EPOCHS must contain non-negative integers" >&2
    exit 2
  fi
done

case "${PHASE}" in
  router)
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BASELINE_CHECKPOINT}}"
    START_EPOCH="${START_EPOCH:-1}"
    MAX_EPOCH="${MAX_EPOCH:-2}"
    EXP="${EXP:-ssq_moe_router_pretrain}"
    PHASE_ARGS=(--source_moe_train_only)
    ;;
  gate)
    if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
      echo "PHASE=gate requires CHECKPOINT_PATH=<trained MoE checkpoint>" >&2
      exit 2
    fi
    START_EPOCH="${START_EPOCH:-1}"
    MAX_EPOCH="${MAX_EPOCH:-1}"
    EXP="${EXP:-ssq_moe_safe_fallback_gate}"
    SOURCE_MOE_BALANCE_WEIGHT=0.0
    SOURCE_MOE_RANK_WEIGHT=0.0
    SOURCE_MOE_ANCHOR_WEIGHT=0.0
    SOURCE_MOE_GATE_LOSS_WEIGHT="${SOURCE_MOE_GATE_LOSS_WEIGHT_GATE:-1.0}"
    SOURCE_MOE_USE_FALLBACK_GATE=1
    # Gate-only runs are also evaluated every epoch; retain only the latest
    # checkpoint and the five independently best REC/mask metrics.
    RETENTION_ARGS=(--checkpoint_metric_retention)
    PHASE_ARGS=(--source_moe_gate_train_only)
    ;;
  continue)
    if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
      echo "PHASE=continue requires CHECKPOINT_PATH=<trained gated MoE checkpoint>" >&2
      exit 2
    fi
    START_EPOCH="${START_EPOCH:-72}"
    MAX_EPOCH="${MAX_EPOCH:-80}"
    EXP="${EXP:-ssq_moe_safe_continue_e72_e80}"
    SOURCE_MOE_USE_FALLBACK_GATE=1
    SOURCE_MOE_GATE_LOSS_WEIGHT="${SOURCE_MOE_GATE_LOSS_WEIGHT_CONTINUE:-1.0}"
    RETENTION_ARGS=(--checkpoint_metric_retention)
    PHASE_ARGS=(--source_moe_train_only)
    ;;
  joint)
    if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
      echo "PHASE=joint requires CHECKPOINT_PATH=<router checkpoint>" >&2
      exit 2
    fi
    START_EPOCH="${START_EPOCH:-1}"
    MAX_EPOCH="${MAX_EPOCH:-4}"
    EXP="${EXP:-ssq_moe_joint_finetune}"
    DATASET_ARGS=(--dataset scannet scanrefer)
    RETENTION_ARGS=(--checkpoint_metric_retention)
    if [[ "${JOINT_RESUME}" == "0" ]]; then
      # Router/gate initialization has a different optimizer layout. Load all
      # model weights, start a fresh joint optimizer, and preserve epoch labels.
      PHASE_ARGS=(--joint_det --reduce_lr)
      CHECKPOINT_START_ARGS=(--checkpoint_start_epoch "${START_EPOCH}")
    elif [[ "${JOINT_RESUME}" == "1" ]]; then
      # A retained joint checkpoint has a numeric epoch plus the exact joint
      # optimizer and scheduler state. load_checkpoint advances it by one.
      PHASE_ARGS=(--joint_det)
    else
      echo "JOINT_RESUME must be 0 or 1" >&2
      exit 2
    fi
    ;;
  *)
    echo "PHASE must be router, gate, continue, or joint" >&2
    exit 2
    ;;
esac

if [[ "${SOURCE_MOE_GATE_RESUME_OPTIMIZER}" == "1" ]]; then
  if [[ "${PHASE}" != "gate" ]]; then
    echo "SOURCE_MOE_GATE_RESUME_OPTIMIZER=1 requires PHASE=gate" >&2
    exit 2
  fi
  GATE_RESUME_ARGS=(--source_moe_gate_resume_optimizer)
elif [[ "${SOURCE_MOE_GATE_RESUME_OPTIMIZER}" != "0" ]]; then
  echo "SOURCE_MOE_GATE_RESUME_OPTIMIZER must be 0 or 1" >&2
  exit 2
fi

if [[ "${DEBUG}" == "1" ]]; then
  DEBUG_ARGS=(--debug)
elif [[ "${DEBUG}" != "0" ]]; then
  echo "DEBUG must be 0 or 1" >&2
  exit 2
fi

if [[ "${SOURCE_MOE_GATE_NEW_HEADS_ONLY}" == "1" ]]; then
  if [[ "${PHASE}" != "gate" ]]; then
    echo "SOURCE_MOE_GATE_NEW_HEADS_ONLY=1 requires PHASE=gate" >&2
    exit 2
  fi
  VALID_NEW_HEADS_CONTRACT=0
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_absolute_quality_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_absolute_quality_calibrated" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_opportunity_quality_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_opportunity_balanced_calibrated" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_opportunity_verified_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_opportunity_verified_calibrated" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_joint_risk_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_joint_risk_calibrated" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v19_fallback_set_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v19_fallback_set_risk_calibrated" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v19_rich_set_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v19_rich_set_empirical_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v23_dense_quality_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v23_dense_quality_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v23_dense_quality_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v27_uncertainty_quality_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v24_relative_risk_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v24_relative_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v25_pairwise_calibrated_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v25_pairwise_calibrated_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v26_prior_restored_pairwise_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v26_prior_restored_pairwise_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v28_selected_abstention_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v28_selected_abstention_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v29_counterfactual_selected_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v29_counterfactual_selected_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v37_counterfactual_benefit_hazard_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v37_counterfactual_benefit_hazard_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v38_complementary_logodds_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v38_complementary_logodds_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" == "cascade_v39_hazard_residual_correction" \
        && "${SOURCE_MOE_GATE_OBJECTIVE}" == "cascade_v39_hazard_residual_risk" ]]; then
    VALID_NEW_HEADS_CONTRACT=1
  fi
  if [[ "${VALID_NEW_HEADS_CONTRACT}" != "1" ]]; then
    echo "SOURCE_MOE_GATE_NEW_HEADS_ONLY=1 requires a matching cascade action and objective" >&2
    exit 2
  fi
  GATE_NEW_HEADS_ARGS=(--source_moe_gate_new_heads_only)
elif [[ "${SOURCE_MOE_GATE_NEW_HEADS_ONLY}" != "0" ]]; then
  echo "SOURCE_MOE_GATE_NEW_HEADS_ONLY must be 0 or 1" >&2
  exit 2
fi

if [[ "${SOURCE_MOE_USE_FALLBACK_GATE}" == "1" ]]; then
  GATE_ENABLE_ARGS=(--source_moe_use_fallback_gate)
fi
if [[ "${SOURCE_MOE_GATE_USE_EVIDENCE_FEATURES}" == "1" ]]; then
  GATE_EVIDENCE_ARGS=(--source_moe_gate_use_evidence_features)
fi
if [[ -n "${SOURCE_MOE_GATE_ACTION_MODE}" ]]; then
  if [[ "${SOURCE_MOE_GATE_ACTION_MODE}" != "decision" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "expected_utility" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "direct_utility" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "hierarchical_utility" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "pairwise_verifier" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "topn_pairwise_verifier" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "topn_dual_evidence_verifier" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "topn_absolute_quality_delta" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_absolute_quality_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_opportunity_quality_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_opportunity_verified_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_joint_risk_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v19_fallback_set_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v19_rich_set_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v23_dense_quality_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v24_relative_risk_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v25_pairwise_calibrated_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v26_prior_restored_pairwise_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v28_selected_abstention_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v29_counterfactual_selected_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v37_counterfactual_benefit_hazard_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v38_complementary_logodds_correction" \
        && "${SOURCE_MOE_GATE_ACTION_MODE}" != "cascade_v39_hazard_residual_correction" ]]; then
    echo "SOURCE_MOE_GATE_ACTION_MODE is invalid" >&2
    exit 2
  fi
  GATE_ACTION_ARGS=(
    --source_moe_gate_action_mode "${SOURCE_MOE_GATE_ACTION_MODE}"
  )
fi
if [[ "${SOURCE_MOE_GATE_OBJECTIVE_EXPLICIT}" == "1" ]]; then
  GATE_OBJECTIVE_ARGS=(
    --source_moe_gate_objective "${SOURCE_MOE_GATE_OBJECTIVE}"
  )
fi

"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  train_dist_mod.py \
  --num_decoder_layers 6 \
  --use_color \
  --weight_decay 0.0005 \
  --data_root "${DATA_ROOT}" \
  --val_freq "${VAL_FREQ}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --save_freq 1 --print_freq "${PRINT_FREQ}" \
  --lr_backbone "${BACKBONE_LR}" --lr "${DECODER_LR}" \
  --text_encoder_lr "${TEXT_ENCODER_LR}" \
  --source_moe_lr "${SOURCE_MOE_LR}" \
  --source_moe_gate_lr "${SOURCE_MOE_GATE_LR}" \
  "${DATASET_ARGS[@]}" --test_dataset scanrefer \
  --detect_intermediate \
  --use_soft_token_loss --use_contrastive_align \
  --log_dir "${LOG_DIR}" \
  --lr_decay_epochs "${LR_DECAY_EPOCHS_ARGS[@]}" \
  --pp_checkpoint "${PP_CHECKPOINT}" \
  --butd --self_attend --augment_det \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --start_epoch "${START_EPOCH}" --max_epoch "${MAX_EPOCH}" \
  --model MCLN --exp "${EXP}" \
  --use_source_moe --eval_use_selector_choice_scores \
  --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}" \
  --source_choice_selector_sources default,contrastive_text,mask_text \
  --source_moe_shared_source default --source_moe_top_k "${SOURCE_MOE_TOP_K}" \
  --source_moe_balance_loss_weight "${SOURCE_MOE_BALANCE_WEIGHT}" \
  --source_moe_rank_loss_weight "${SOURCE_MOE_RANK_WEIGHT}" \
  --source_moe_mask_rank_loss_weight "${SOURCE_MOE_MASK_RANK_WEIGHT}" \
  --source_moe_rank_temperature "${SOURCE_MOE_TEMPERATURE}" \
  --source_moe_anchor_loss_weight "${SOURCE_MOE_ANCHOR_WEIGHT}" \
  --source_moe_anchor_margin "${SOURCE_MOE_ANCHOR_MARGIN}" \
  --source_moe_query_layers 1 --source_moe_query_heads 4 \
  --source_moe_query_dropout 0.1 \
  --source_moe_query_max_delta "${SOURCE_MOE_QUERY_MAX_DELTA}" \
  --source_moe_gate_hidden_dim "${SOURCE_MOE_GATE_HIDDEN_DIM}" \
  --source_moe_gate_candidate_top_k "${SOURCE_MOE_GATE_TOP_K}" \
  --source_moe_gate_break_cost "${SOURCE_MOE_GATE_BREAK_COST}" \
  --source_moe_gate_decision_margin "${SOURCE_MOE_GATE_DECISION_MARGIN}" \
  --source_moe_gate_mask_utility_weight "${SOURCE_MOE_GATE_MASK_UTILITY_WEIGHT}" \
  --source_moe_gate_uncertainty_weight "${SOURCE_MOE_GATE_UNCERTAINTY_WEIGHT}" \
  --source_moe_gate_context_layers "${SOURCE_MOE_GATE_CONTEXT_LAYERS}" \
  --source_moe_gate_context_heads "${SOURCE_MOE_GATE_CONTEXT_HEADS}" \
  --source_moe_gate_context_dropout "${SOURCE_MOE_GATE_CONTEXT_DROPOUT}" \
  --source_moe_gate_loss_weight "${SOURCE_MOE_GATE_LOSS_WEIGHT}" \
  --source_moe_gate_mask_loss_weight "${SOURCE_MOE_GATE_MASK_LOSS_WEIGHT}" \
  --source_moe_gate_focal_gamma "${SOURCE_MOE_GATE_FOCAL_GAMMA}" \
  --source_moe_gate_false_override_weight "${SOURCE_MOE_GATE_FALSE_OVERRIDE_WEIGHT}" \
  "${GATE_OBJECTIVE_ARGS[@]}" \
  --source_moe_gate_setwise_temperature "${SOURCE_MOE_GATE_SETWISE_TEMPERATURE}" \
  --source_moe_gate_boundary_loss_weight "${SOURCE_MOE_GATE_BOUNDARY_LOSS_WEIGHT}" \
  "${GATE_ENABLE_ARGS[@]}" \
  "${GATE_EVIDENCE_ARGS[@]}" \
  "${GATE_ACTION_ARGS[@]}" \
  "${GATE_RESUME_ARGS[@]}" \
  "${GATE_NEW_HEADS_ARGS[@]}" \
  "${CHECKPOINT_START_ARGS[@]}" \
  "${RETENTION_ARGS[@]}" \
  "${PHASE_ARGS[@]}" \
  "${DEBUG_ARGS[@]}" \
  "$@"
