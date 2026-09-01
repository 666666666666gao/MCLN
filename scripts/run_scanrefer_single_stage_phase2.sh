#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_scanrefer_single_stage_phase2"
readonly SOURCE_CHECKPOINT="${DATA_ROOT%/}/output/single_stage_best_postprocess/scanrefer/mcln_epoch71_parent_geometry_single_stage_e1_e100_b18x4/1785907694/ckpt_best_rec_acc025.pth"
readonly SOURCE_SHA256="8804109f0db25113cc6683314dcc7ab1ca2f7a93c1307a1fcf76420c6dc43eec"
readonly SOURCE_EPOCH=7
readonly SOURCE_AUDIT="${OUTPUT_ROOT}/gates/source_epoch7_audit_r3_pure_scanrefer.json"
readonly SMOKE_GATE="${OUTPUT_ROOT}/gates/smoke_gate_r2_pure_scanrefer.json"
readonly GPU_IDS=0
readonly NPROC_PER_NODE=1
readonly BATCH_SIZE=18
MODE="${MODE:-smoke}"
MASTER_PORT="${MASTER_PORT:-5142}"

if (($# != 0)); then
  echo "single-stage phase2 uses one frozen configuration" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "single-stage source checkpoint is missing" >&2
  exit 3
fi
actual_sha="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha}" != "${SOURCE_SHA256}" ]]; then
  echo "single-stage source checkpoint SHA-256 changed" >&2
  exit 3
fi

mkdir -p "${OUTPUT_ROOT}/gates" "${OUTPUT_ROOT}/launch"
COMMON_AUDIT_ARGS=(
  --repo-root "${ROOT_DIR}"
  --checkpoint "${SOURCE_CHECKPOINT}"
  --expected-sha256 "${SOURCE_SHA256}"
  --expected-epoch "${SOURCE_EPOCH}"
  --data-root "${DATA_ROOT}"
)
if [[ -f "${SOURCE_AUDIT}" ]]; then
  "${PYTHON_BIN}" scripts/audit_scanrefer_single_stage_phase2.py \
    --mode source-verify --source-audit "${SOURCE_AUDIT}" \
    "${COMMON_AUDIT_ARGS[@]}"
else
  "${PYTHON_BIN}" scripts/audit_scanrefer_single_stage_phase2.py \
    --mode source-build --output "${SOURCE_AUDIT}" \
    "${COMMON_AUDIT_ARGS[@]}"
fi

case "${MODE}" in
  smoke)
    MAX_EPOCH=1
    EXPECTED_EVAL_SAMPLE_COUNT=128
    PRINT_FREQ=1
    SPLIT_ARGS=(--debug --debug_train_holdout)
    EXP="scanrefer_single_stage_phase2_pure_scanrefer_smoke_e1_b18x1"
    ;;
  formal)
    MAX_EPOCH=12
    EXPECTED_EVAL_SAMPLE_COUNT=9508
    PRINT_FREQ=100
    SPLIT_ARGS=()
    if [[ ! -f "${SMOKE_GATE}" ]]; then
      echo "single-stage formal requires ${SMOKE_GATE}" >&2
      exit 4
    fi
    "${PYTHON_BIN}" scripts/audit_scanrefer_single_stage_phase2.py \
      --mode smoke-verify --source-audit "${SOURCE_AUDIT}" \
      --gate "${SMOKE_GATE}" "${COMMON_AUDIT_ARGS[@]}"
    EXP="scanrefer_single_stage_phase2_pure_scanrefer_formal_e1_e12_b18x1_lrscaled"
    ;;
  *)
    echo "MODE must be smoke or formal" >&2
    exit 2
    ;;
esac

LOG_DIR="${OUTPUT_ROOT}/${EXP}"
LOCK_FILE="${OUTPUT_ROOT}/single_gpu.lock"
mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another single-stage phase2 run owns ${LOCK_FILE}" >&2
  exit 5
fi

timestamp="$(date '+%Y%m%d_%H%M%S')"
LAUNCH_LOG="${OUTPUT_ROOT}/launch/${EXP}_${timestamp}.log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pointnet2:${PYTHONPATH:-}"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] starting ${EXP}"
echo "source_checkpoint=${SOURCE_CHECKPOINT}"
echo "source_sha256=${SOURCE_SHA256}"
echo "training_datasets=scanrefer_only"
echo "optimizer_policy=fresh optimizer; linear LR scale 0.25 for global batch 72->18"
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  train_dist_mod.py \
  --num_decoder_layers 6 --num_target 256 \
  --use_color --weight_decay 0.0005 \
  --data_root "${DATA_ROOT}" \
  --val_freq 1 --batch_size "${BATCH_SIZE}" \
  --num_workers 4 --dataloader_prefetch_factor 1 \
  --save_freq 1 --print_freq "${PRINT_FREQ}" \
  --lr_backbone 5e-5 --lr 5e-6 --text_encoder_lr 7.5e-7 \
  --dataset scanrefer --test_dataset scanrefer \
  --detect_intermediate \
  --use_soft_token_loss --use_contrastive_align \
  --log_dir "${LOG_DIR}" \
  --lr_decay_epochs 8 11 \
  --pp_checkpoint "${DATA_ROOT%/}/gf_detector_l6o256.pth" \
  --self_attend --augment_det --skip_missing_superpoints \
  --checkpoint_path "${SOURCE_CHECKPOINT}" \
  --checkpoint_start_epoch 1 --reduce_lr \
  --start_epoch 1 --max_epoch "${MAX_EPOCH}" \
  --model MCLN --exp "${EXP}" \
  --use_source_choice_selector --eval_use_selector_choice_scores \
  --source_choice_selector_sources default,default_rank_blend_contrastive010 \
  --source_choice_selector_default_source default \
  --source_choice_selector_hidden_dim 288 \
  --source_choice_selector_lr 1.25e-4 \
  --source_choice_selector_loss_weight 0.5 \
  --source_choice_selector_choice_target precision_gain_default_sourcewise_focal_bce \
  --source_choice_selector_min_iou_gap 0.03 \
  --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}" \
  --checkpoint_metric_retention \
  "${SPLIT_ARGS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] finished ${EXP}"
echo "launch_log=${LAUNCH_LOG}"
