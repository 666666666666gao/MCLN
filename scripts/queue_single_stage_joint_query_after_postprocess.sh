#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
TRAIN_EXP="${TRAIN_EXP:-mcln_epoch71_parent_geometry_single_stage_e1_e100_b18x4}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${DATA_ROOT%/}/output/single_stage_best_postprocess/scanrefer/${TRAIN_EXP}/1785907694}"
POST_ROOT="${POST_ROOT:-${ROOT_DIR}/experiment_output/single_stage_best_postprocess/1785907694}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiment_output/single_stage_joint_query}"
PANEL_MAX_EPOCH="${PANEL_MAX_EPOCH:-3}"
PANEL_BATCH_SIZE="${PANEL_BATCH_SIZE:-64}"
PANEL_EXPECTED_SAMPLE_COUNT="${PANEL_EXPECTED_SAMPLE_COUNT:-128}"
FORMAL_MAX_EPOCH="${FORMAL_MAX_EPOCH:-80}"
FORMAL_BATCH_SIZE="${FORMAL_BATCH_SIZE:-64}"
FORMAL_EXPECTED_SAMPLE_COUNT="${FORMAL_EXPECTED_SAMPLE_COUNT:-9508}"
FORMAL_EXPECTED_STEPS_PER_EPOCH="${FORMAL_EXPECTED_STEPS_PER_EPOCH:-191}"
FORMAL_EXP="${FORMAL_EXP:-single_v47_selector_joint_query_mask_full80_b64x4_lr3e4_cmw025_clw010_k16}"
FORMAL_MASTER_PORT="${FORMAL_MASTER_PORT:-4614}"
RUN_FORMAL_AFTER_SMOKE="${RUN_FORMAL_AFTER_SMOKE:-0}"
MIN_FREE_KIB="${MIN_FREE_KIB:-4194304}"
GPU_IDLE_WAIT_SECONDS="${GPU_IDLE_WAIT_SECONDS:-60}"
QUEUE_LOG="${QUEUE_LOG:-${OUTPUT_ROOT}/single_v47_after_postprocess_queue.log}"
LOCK_FILE="${LOCK_FILE:-${OUTPUT_ROOT}/single_v47_after_postprocess_queue.lock}"

for value in "${PANEL_MAX_EPOCH}" "${PANEL_BATCH_SIZE}" \
             "${PANEL_EXPECTED_SAMPLE_COUNT}" "${FORMAL_MAX_EPOCH}" \
             "${FORMAL_BATCH_SIZE}" "${FORMAL_EXPECTED_SAMPLE_COUNT}" \
             "${FORMAL_EXPECTED_STEPS_PER_EPOCH}" "${FORMAL_MASTER_PORT}" \
             "${MIN_FREE_KIB}" "${GPU_IDLE_WAIT_SECONDS}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "queue integer settings must be positive" >&2
    exit 2
  fi
done
if [[ "${RUN_FORMAL_AFTER_SMOKE}" != "0" \
      && "${RUN_FORMAL_AFTER_SMOKE}" != "1" ]]; then
  echo "RUN_FORMAL_AFTER_SMOKE must be 0 or 1" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another single-stage joint-query queue owns ${LOCK_FILE}" >&2
  exit 3
fi
exec > >(tee -a "${QUEUE_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[$(timestamp)] waiting for the single-stage postprocess pipeline"
postprocess_pid="${POSTPROCESS_PID:-}"
if [[ -z "${postprocess_pid}" ]]; then
  postprocess_pid="$(pgrep -o -f 'bash scripts/queue_single_stage_best_postprocess.sh' || true)"
fi
if [[ -n "${postprocess_pid}" ]]; then
  echo "[$(timestamp)] bound to process ${postprocess_pid}; no log polling"
  tail --pid="${postprocess_pid}" -f /dev/null
fi

best_checkpoint="${TRAIN_RUN_DIR}/ckpt_best_rec_acc025.pth"
for path in "${POST_ROOT}/training_completion.json" \
            "${POST_ROOT}/best_checkpoint.sha256" \
            "${POST_ROOT}/parent_reranker.pth" \
            "${POST_ROOT}/geometry_reranker.pth" \
            "${POST_ROOT}/official_eval.log" \
            "${best_checkpoint}"; do
  if [[ ! -f "${path}" ]]; then
    echo "required postprocess artifact is missing: ${path}" >&2
    exit 4
  fi
done
expected_sha="$(awk 'NR == 1 {print $1}' "${POST_ROOT}/best_checkpoint.sha256")"
actual_sha="$(sha256sum "${best_checkpoint}" | awk '{print $1}')"
if [[ -z "${expected_sha}" || "${actual_sha}" != "${expected_sha}" ]]; then
  echo "single-stage best checkpoint fingerprint changed" >&2
  exit 4
fi

free_kib="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
if [[ ! "${free_kib}" =~ ^[1-9][0-9]*$ \
      || "${free_kib}" -lt "${MIN_FREE_KIB}" ]]; then
  echo "joint-query output filesystem has insufficient free space: ${free_kib:-unknown} KiB" >&2
  exit 4
fi

initialization_audit="${OUTPUT_ROOT}/single_v43_selector_initialization_audit.json"
"${PYTHON_BIN}" scripts/audit_joint_query_initialization.py \
  --checkpoint "${best_checkpoint}" --profile v43 \
  --output "${initialization_audit}"

while true; do
  compute_pids="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      | sed '/^[[:space:]]*$/d' | sort -u
  )"
  if [[ -z "${compute_pids}" ]]; then
    break
  fi
  echo "[$(timestamp)] GPUs still have compute processes; waiting ${GPU_IDLE_WAIT_SECONDS}s"
  sleep "${GPU_IDLE_WAIT_SECONDS}"
done

variants=(
  "single_v47_smoke_base"
  "single_v47_smoke_cmw010_k8"
  "single_v47_smoke_cmw025_k16"
  "single_v47_smoke_cmw025_clw010_k16"
)
candidate_mask_weights=("0.00" "0.10" "0.25" "0.25")
candidate_lovasz_weights=("0.00" "0.00" "0.00" "0.10")
candidate_top_ks=("16" "8" "16" "16")
ports=("4610" "4611" "4612" "4613")

job_pids=()
job_logs=()
for gpu in 0 1 2 3; do
  variant="${variants[${gpu}]}"
  job_log="${OUTPUT_ROOT}/${variant}_launcher.log"
  job_logs+=("${job_log}")
  echo "[$(timestamp)] starting ${variant} on GPU ${gpu}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    NPROC_PER_NODE=1 \
    MASTER_PORT="${ports[${gpu}]}" \
    CHECKPOINT_PATH="${best_checkpoint}" \
    LOG_DIR="${OUTPUT_ROOT}" \
    EXP="${variant}" \
    BATCH_SIZE="${PANEL_BATCH_SIZE}" \
    NUM_WORKERS=4 \
    DATALOADER_PREFETCH_FACTOR=1 \
    PERSISTENT_TRAIN_WORKERS=0 \
    START_EPOCH=1 \
    MAX_EPOCH="${PANEL_MAX_EPOCH}" \
    VAL_FREQ=1 \
    PRINT_FREQ=1 \
    EXPECTED_EVAL_SAMPLE_COUNT="${PANEL_EXPECTED_SAMPLE_COUNT}" \
    DEBUG=1 \
    JOINT_QUERY_QUALITY_LR=3e-4 \
    JOINT_QUERY_QUALITY_DROPOUT=0.1 \
    JOINT_QUERY_QUALITY_MASK_WEIGHT=0.25 \
    JOINT_QUERY_QUALITY_TEMPERATURE=0.25 \
    JOINT_QUERY_QUALITY_ANCHOR_LOSS_WEIGHT=0.5 \
    JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT="${candidate_mask_weights[${gpu}]}" \
    JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT="${candidate_lovasz_weights[${gpu}]}" \
    JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K="${candidate_top_ks[${gpu}]}" \
    bash scripts/train_scanrefer_single_stage_joint_query_quality.sh \
      >"${job_log}" 2>&1
  ) &
  job_pids+=("$!")
done

failed=0
for index in 0 1 2 3; do
  if ! wait "${job_pids[${index}]}"; then
    echo "[$(timestamp)] ${variants[${index}]} failed; see ${job_logs[${index}]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 5
fi

expected_step=$((2 * PANEL_MAX_EPOCH))
run_dirs=()
for index in 0 1 2 3; do
  variant="${variants[${index}]}"
  variant_root="${OUTPUT_ROOT}/scanrefer/${variant}"
  run_dir="$(
    find "${variant_root}" -mindepth 1 -maxdepth 1 -type d \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "${run_dir}" ]]; then
    echo "missing run directory for ${variant}" >&2
    exit 6
  fi
  run_dirs+=("${run_dir}")
  "${PYTHON_BIN}" scripts/audit_training_completion.py \
    --metrics "${run_dir}/eval_metrics_epoch_${PANEL_MAX_EPOCH}.json" \
    --checkpoint "${run_dir}/ckpt_epoch_last.pth" \
    --expected-epoch "${PANEL_MAX_EPOCH}" \
    --expected-sample-count "${PANEL_EXPECTED_SAMPLE_COUNT}" \
    --require-position-subgroups \
    --output "${run_dir}/audit_completion_epoch_${PANEL_MAX_EPOCH}.json"
  "${PYTHON_BIN}" scripts/audit_source_moe_checkpoint.py \
    --profile v43_selector --baseline "${best_checkpoint}" \
    --checkpoint "${run_dir}/ckpt_epoch_last.pth" \
    --expected-epoch "${PANEL_MAX_EPOCH}" \
    --expected-step "${expected_step}" \
    --output "${run_dir}/audit_v43_selector.json"
done

"${PYTHON_BIN}" scripts/summarize_v41_smoke_panel.py \
  --output "${OUTPUT_ROOT}/single_v47_base_panel_summary.json" \
  --profile v43_selector --epoch "${PANEL_MAX_EPOCH}" \
  --expected-sample-count "${PANEL_EXPECTED_SAMPLE_COUNT}" \
  --record "${variants[0]}" "${run_dirs[0]}" "${job_logs[0]}" \
  --record "${variants[1]}" "${run_dirs[1]}" "${job_logs[1]}"
"${PYTHON_BIN}" scripts/summarize_v41_smoke_panel.py \
  --output "${OUTPUT_ROOT}/single_v47_candidate_panel_summary.json" \
  --profile v43_selector --require-candidate-mask \
  --require-lovasz-variant "${variants[3]}" \
  --epoch "${PANEL_MAX_EPOCH}" \
  --expected-sample-count "${PANEL_EXPECTED_SAMPLE_COUNT}" \
  --record "${variants[2]}" "${run_dirs[2]}" "${job_logs[2]}" \
  --record "${variants[3]}" "${run_dirs[3]}" "${job_logs[3]}"

for run_dir in "${run_dirs[@]}"; do
  find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.pth' \
    -print -delete
done

if [[ "${RUN_FORMAL_AFTER_SMOKE}" == "0" ]]; then
  echo "[$(timestamp)] formal single-stage joint-query launch disabled"
  exit 0
fi

if [[ "$(sha256sum "${best_checkpoint}" | awk '{print $1}')" != "${expected_sha}" ]]; then
  echo "single-stage best checkpoint changed before formal launch" >&2
  exit 7
fi
formal_log="${OUTPUT_ROOT}/${FORMAL_EXP}_launcher.log"
echo "[$(timestamp)] starting four-GPU formal single-stage joint-query run"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT="${FORMAL_MASTER_PORT}" \
CHECKPOINT_PATH="${best_checkpoint}" \
LOG_DIR="${OUTPUT_ROOT}" \
EXP="${FORMAL_EXP}" \
BATCH_SIZE="${FORMAL_BATCH_SIZE}" \
NUM_WORKERS=4 \
DATALOADER_PREFETCH_FACTOR=1 \
PERSISTENT_TRAIN_WORKERS=0 \
START_EPOCH=1 \
MAX_EPOCH="${FORMAL_MAX_EPOCH}" \
VAL_FREQ=1 \
PRINT_FREQ=50 \
EXPECTED_EVAL_SAMPLE_COUNT="${FORMAL_EXPECTED_SAMPLE_COUNT}" \
DEBUG=0 \
JOINT_QUERY_QUALITY_LR=3e-4 \
JOINT_QUERY_QUALITY_DROPOUT=0.1 \
JOINT_QUERY_QUALITY_MASK_WEIGHT=0.25 \
JOINT_QUERY_QUALITY_TEMPERATURE=0.25 \
JOINT_QUERY_QUALITY_ANCHOR_LOSS_WEIGHT=0.5 \
JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT=0.25 \
JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT=0.10 \
JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K=16 \
bash scripts/train_scanrefer_single_stage_joint_query_quality.sh \
  >"${formal_log}" 2>&1

formal_root="${OUTPUT_ROOT}/scanrefer/${FORMAL_EXP}"
formal_run_dir="$(
  find "${formal_root}" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
)"
if [[ -z "${formal_run_dir}" ]]; then
  echo "missing formal run directory" >&2
  exit 8
fi
formal_step=$((FORMAL_EXPECTED_STEPS_PER_EPOCH * FORMAL_MAX_EPOCH))
"${PYTHON_BIN}" scripts/audit_training_completion.py \
  --metrics "${formal_run_dir}/eval_metrics_epoch_${FORMAL_MAX_EPOCH}.json" \
  --checkpoint "${formal_run_dir}/ckpt_epoch_last.pth" \
  --expected-epoch "${FORMAL_MAX_EPOCH}" \
  --expected-sample-count "${FORMAL_EXPECTED_SAMPLE_COUNT}" \
  --require-position-subgroups \
  --output "${formal_run_dir}/audit_completion_epoch_${FORMAL_MAX_EPOCH}.json"
"${PYTHON_BIN}" scripts/audit_source_moe_checkpoint.py \
  --profile v43_selector --baseline "${best_checkpoint}" \
  --checkpoint "${formal_run_dir}/ckpt_epoch_last.pth" \
  --expected-epoch "${FORMAL_MAX_EPOCH}" \
  --expected-step "${formal_step}" \
  --output "${formal_run_dir}/audit_v43_selector_final.json"
echo "[$(timestamp)] formal single-stage joint-query run completed and audited"
