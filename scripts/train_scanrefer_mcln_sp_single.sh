#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT%/}/output/single_stage_best_postprocess}"
PP_CHECKPOINT="${PP_CHECKPOINT:-${DATA_ROOT%/}/gf_detector_l6o256.pth}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_best_backbone_acc025_0.582878_component.pth}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_PORT="${MASTER_PORT:-4511}"
BATCH_SIZE="${BATCH_SIZE:-18}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-1}"
START_EPOCH="${START_EPOCH:-1}"
MAX_EPOCH="${MAX_EPOCH:-100}"
VAL_FREQ="${VAL_FREQ:-1}"
PRINT_FREQ="${PRINT_FREQ:-50}"
EXP="${EXP:-mcln_epoch71_parent_geometry_single_stage_e1_e100_b18x4}"
EXPECTED_EVAL_SAMPLE_COUNT="${EXPECTED_EVAL_SAMPLE_COUNT:-9508}"

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
for name in BATCH_SIZE NUM_WORKERS DATALOADER_PREFETCH_FACTOR START_EPOCH MAX_EPOCH VAL_FREQ; do
  value="${!name}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer" >&2
    exit 2
  fi
done
if (( BATCH_SIZE == 0 || DATALOADER_PREFETCH_FACTOR == 0 || MAX_EPOCH < START_EPOCH )); then
  echo "invalid batch, prefetch, or epoch range" >&2
  exit 2
fi
for path in "${PYTHON_BIN}" "${PP_CHECKPOINT}" "${CHECKPOINT_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required file is missing: ${path}" >&2
    exit 2
  fi
done

# The protected epoch-71 checkpoint was trained with the detected-box stream.
# Require that the only dropped tensors are that stream before transferring it
# into a network whose runtime and saved config are genuinely single-stage.
"${PYTHON_BIN}" scripts/audit_scanrefer_single_stage_transfer.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --data-root "${DATA_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  train_dist_mod.py \
  --num_decoder_layers 6 --num_target 256 \
  --use_color --weight_decay 0.0005 \
  --data_root "${DATA_ROOT}" \
  --val_freq "${VAL_FREQ}" --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --save_freq 1 --print_freq "${PRINT_FREQ}" \
  --lr_backbone 2e-4 --lr 2e-5 --text_encoder_lr 3e-6 \
  --dataset scanrefer --test_dataset scanrefer \
  --detect_intermediate --joint_det \
  --use_soft_token_loss --use_contrastive_align \
  --log_dir "${LOG_DIR}" \
  --lr_decay_epochs 50 75 \
  --pp_checkpoint "${PP_CHECKPOINT}" \
  --self_attend --augment_det --skip_missing_superpoints \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --checkpoint_start_epoch "${START_EPOCH}" --reduce_lr \
  --start_epoch "${START_EPOCH}" --max_epoch "${MAX_EPOCH}" \
  --model MCLN --exp "${EXP}" \
  --use_source_choice_selector --eval_use_selector_choice_scores \
  --source_choice_selector_sources default,default_rank_blend_contrastive010 \
  --source_choice_selector_default_source default \
  --source_choice_selector_hidden_dim 288 \
  --source_choice_selector_lr 5e-4 \
  --source_choice_selector_loss_weight 0.5 \
  --source_choice_selector_choice_target precision_gain_default_sourcewise_focal_bce \
  --source_choice_selector_min_iou_gap 0.03 \
  --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}" \
  --checkpoint_metric_retention \
  "$@"
