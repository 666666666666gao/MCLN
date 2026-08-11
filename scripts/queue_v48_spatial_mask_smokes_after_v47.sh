#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
PROTECTED_V19="${PROTECTED_V19:-${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_network_best_v19_rec025_0.581195_mask025_0.598233.pth}"
PROTECTED_V19_SHA256="${PROTECTED_V19_SHA256:-2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiment_output/v48_spatial_mask}"
PANEL_MAX_EPOCH="${PANEL_MAX_EPOCH:-3}"
PANEL_BATCH_SIZE="${PANEL_BATCH_SIZE:-64}"
PANEL_EXPECTED_SAMPLE_COUNT="${PANEL_EXPECTED_SAMPLE_COUNT:-128}"
MIN_FREE_KIB="${MIN_FREE_KIB:-4194304}"
GPU_IDLE_WAIT_SECONDS="${GPU_IDLE_WAIT_SECONDS:-60}"
QUEUE_LOG="${QUEUE_LOG:-${OUTPUT_ROOT}/v48_after_v47_queue.log}"
LOCK_FILE="${LOCK_FILE:-${OUTPUT_ROOT}/v48_after_v47_queue.lock}"

for value in "${PANEL_MAX_EPOCH}" "${PANEL_BATCH_SIZE}" \
             "${PANEL_EXPECTED_SAMPLE_COUNT}" "${MIN_FREE_KIB}" \
             "${GPU_IDLE_WAIT_SECONDS}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "V48 queue integer settings must be positive" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another V48 spatial-mask queue owns ${LOCK_FILE}" >&2
  exit 3
fi
exec > >(tee -a "${QUEUE_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[$(timestamp)] waiting for the V47 single-stage queue"
predecessor_pid="${PREDECESSOR_PID:-}"
if [[ -z "${predecessor_pid}" ]]; then
  predecessor_pid="$(
    pgrep -o -f \
      'bash scripts/queue_single_stage_joint_query_after_postprocess.sh' \
      || true
  )"
fi
if [[ -n "${predecessor_pid}" ]]; then
  if [[ ! "${predecessor_pid}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREDECESSOR_PID must be a positive process id" >&2
    exit 2
  fi
  echo "[$(timestamp)] bound to V47 process ${predecessor_pid}; no log polling"
  tail --pid="${predecessor_pid}" -f /dev/null
  V47_SUMMARY_ROOT="${V47_SUMMARY_ROOT:-${ROOT_DIR}/experiment_output/single_stage_joint_query}"
  for summary_name in single_v47_base_panel_summary.json \
                      single_v47_candidate_panel_summary.json; do
    summary_path="${V47_SUMMARY_ROOT}/${summary_name}"
    if [[ ! -f "${summary_path}" ]]; then
      echo "V47 queue ended without ${summary_path}" >&2
      exit 5
    fi
    "${PYTHON_BIN}" - "${summary_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("pass") is not True:
    raise SystemExit("V47 smoke summary did not pass")
PY
  done
fi

if [[ ! -f "${PROTECTED_V19}" ]]; then
  echo "protected V19 checkpoint is missing: ${PROTECTED_V19}" >&2
  exit 4
fi
actual_sha256="$(sha256sum "${PROTECTED_V19}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${PROTECTED_V19_SHA256}" ]]; then
  echo "protected V19 SHA-256 changed" >&2
  exit 4
fi
free_kib="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
if [[ ! "${free_kib}" =~ ^[1-9][0-9]*$ \
      || "${free_kib}" -lt "${MIN_FREE_KIB}" ]]; then
  echo "V48 output filesystem has insufficient free space: ${free_kib:-unknown} KiB" >&2
  exit 4
fi

initialization_audit="${OUTPUT_ROOT}/v48_protected_v19_initialization.json"
"${PYTHON_BIN}" scripts/audit_joint_query_initialization.py \
  --checkpoint "${PROTECTED_V19}" --profile v48 \
  --output "${initialization_audit}"
chmod 0444 "${initialization_audit}"

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
  "v48_smoke_cmw010_clw000_k8"
  "v48_smoke_cmw025_clw000_k16"
  "v48_smoke_cmw025_clw005_k16"
  "v48_smoke_cmw025_clw010_k16"
)
candidate_mask_weights=("0.10" "0.25" "0.25" "0.25")
candidate_lovasz_weights=("0.00" "0.00" "0.05" "0.10")
candidate_top_ks=("8" "16" "16" "16")
ports=("4620" "4621" "4622" "4623")

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
    CHECKPOINT_PATH="${PROTECTED_V19}" \
    LOG_DIR="${OUTPUT_ROOT}" \
    EXP="${variant}" \
    MODEL_STAGE=two \
    SOURCE_ARBITER=moe \
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
    JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION=1 \
    JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE=1 \
    JOINT_QUERY_QUALITY_USE_GATE_EVIDENCE=0 \
    JOINT_QUERY_QUALITY_USE_SPATIAL_MASK_REFINER=1 \
    JOINT_QUERY_QUALITY_SPATIAL_MASK_HIDDEN_DIM=32 \
    JOINT_QUERY_QUALITY_MAX_SPATIAL_MASK_DELTA=2.0 \
    JOINT_QUERY_QUALITY_MAX_MASK_ALPHA_DELTA=1.0 \
    JOINT_QUERY_QUALITY_MAX_MASK_LOGIT_BIAS=2.0 \
    JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT="${candidate_mask_weights[${gpu}]}" \
    JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT="${candidate_lovasz_weights[${gpu}]}" \
    JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K="${candidate_top_ks[${gpu}]}" \
    bash scripts/train_scanrefer_joint_query_quality.sh \
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
    --profile v48 --baseline "${PROTECTED_V19}" \
    --checkpoint "${run_dir}/ckpt_epoch_last.pth" \
    --expected-epoch "${PANEL_MAX_EPOCH}" \
    --expected-step "${expected_step}" \
    --output "${run_dir}/audit_v48.json"
done

summary_path="${OUTPUT_ROOT}/v48_spatial_mask_smoke_summary.json"
"${PYTHON_BIN}" scripts/summarize_v41_smoke_panel.py \
  --output "${summary_path}" --profile v48 \
  --require-candidate-mask \
  --require-lovasz-variant "${variants[2]}" \
  --require-lovasz-variant "${variants[3]}" \
  --epoch "${PANEL_MAX_EPOCH}" \
  --expected-sample-count "${PANEL_EXPECTED_SAMPLE_COUNT}" \
  --record "${variants[0]}" "${run_dirs[0]}" "${job_logs[0]}" \
  --record "${variants[1]}" "${run_dirs[1]}" "${job_logs[1]}" \
  --record "${variants[2]}" "${run_dirs[2]}" "${job_logs[2]}" \
  --record "${variants[3]}" "${run_dirs[3]}" "${job_logs[3]}"

for run_dir in "${run_dirs[@]}"; do
  echo "[$(timestamp)] removing audited V48 debug checkpoints from ${run_dir}"
  find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.pth' \
    -print -delete
done
echo "[$(timestamp)] V48 spatial-mask smoke panel completed: ${summary_path}"
