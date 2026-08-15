#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
PROTECTED_V19="${PROTECTED_V19:-${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_network_best_v19_rec025_0.581195_mask025_0.598233.pth}"
PROTECTED_V19_SHA256="${PROTECTED_V19_SHA256:-2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT%/}/output/v106_graph_mask_20260814}"
GPU_ID="${GPU_ID:-0}"
PANEL_MAX_EPOCH="${PANEL_MAX_EPOCH:-3}"
PANEL_BATCH_SIZE="${PANEL_BATCH_SIZE:-64}"
PANEL_EXPECTED_SAMPLE_COUNT="${PANEL_EXPECTED_SAMPLE_COUNT:-128}"
MIN_FREE_KIB="${MIN_FREE_KIB:-4194304}"
QUEUE_LOG="${QUEUE_LOG:-${OUTPUT_ROOT}/v106_graph_queue.log}"
LOCK_FILE="${LOCK_FILE:-${OUTPUT_ROOT}/v106_graph_queue.lock}"

for value in "${GPU_ID}" "${PANEL_MAX_EPOCH}" "${PANEL_BATCH_SIZE}" \
             "${PANEL_EXPECTED_SAMPLE_COUNT}" "${MIN_FREE_KIB}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "V106 queue integer settings must be non-negative integers" >&2
    exit 2
  fi
done
if (( PANEL_MAX_EPOCH < 1 || PANEL_BATCH_SIZE < 1 \
      || PANEL_EXPECTED_SAMPLE_COUNT < 1 || MIN_FREE_KIB < 1 )); then
  echo "V106 epoch/batch/sample/free-space settings must be positive" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another V106 queue owns ${LOCK_FILE}" >&2
  exit 3
fi
exec > >(tee -a "${QUEUE_LOG}") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

if [[ ! -f "${PROTECTED_V19}" ]]; then
  echo "protected V19 checkpoint is missing" >&2
  exit 4
fi
actual_sha256="$(sha256sum "${PROTECTED_V19}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${PROTECTED_V19_SHA256}" ]]; then
  echo "protected V19 SHA-256 changed" >&2
  exit 4
fi
if ! nvidia-smi -i "${GPU_ID}" --query-gpu=index --format=csv,noheader \
     >/dev/null 2>&1; then
  echo "requested GPU ${GPU_ID} is unavailable" >&2
  exit 4
fi
compute_pids="$(
  nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid \
    --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u
)"
if [[ -n "${compute_pids}" ]]; then
  echo "GPU ${GPU_ID} is busy with PIDs: ${compute_pids}" >&2
  exit 4
fi
free_kib="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
if [[ ! "${free_kib}" =~ ^[1-9][0-9]*$ \
      || "${free_kib}" -lt "${MIN_FREE_KIB}" ]]; then
  echo "V106 output filesystem has insufficient free space" >&2
  exit 4
fi

contract_receipt="${OUTPUT_ROOT}/v106_graph_contract_smoke.json"
"${PYTHON_BIN}" scripts/smoke_v106_graph_mask_contract.py \
  --output "${contract_receipt}"
chmod 0444 "${contract_receipt}"

variants=("v106_smoke_spatial" "v106_smoke_bilateral")
graph_modes=("spatial" "bilateral")
ports=("4770" "4771")
run_dirs=()
job_logs=()
expected_step=$((2 * PANEL_MAX_EPOCH))

for index in 0 1; do
  variant="${variants[${index}]}"
  graph_mode="${graph_modes[${index}]}"
  job_log="${OUTPUT_ROOT}/${variant}_launcher.log"
  job_logs+=("${job_log}")
  echo "[$(timestamp)] starting ${variant} (${graph_mode}) serially on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  MASTER_PORT="${ports[${index}]}" \
  CHECKPOINT_PATH="${PROTECTED_V19}" \
  LOG_DIR="${OUTPUT_ROOT}" EXP="${variant}" \
  BATCH_SIZE="${PANEL_BATCH_SIZE}" NUM_WORKERS=4 \
  DATALOADER_PREFETCH_FACTOR=1 PERSISTENT_TRAIN_WORKERS=0 \
  START_EPOCH=1 MAX_EPOCH="${PANEL_MAX_EPOCH}" \
  VAL_FREQ=1 PRINT_FREQ=1 \
  EXPECTED_EVAL_SAMPLE_COUNT="${PANEL_EXPECTED_SAMPLE_COUNT}" \
  DEBUG=1 EGQS_MASK_REFINER_LR=3e-4 \
  EGQS_MASK_REFINER_ARCH=graph \
  EGQS_MASK_REFINER_MAX_DELTA=2.0 \
  EGQS_MASK_REFINER_GRAPH_MODE="${graph_mode}" \
  EGQS_MASK_REFINER_NEIGHBOR_COUNT=8 \
  bash scripts/train_scanrefer_egqs_mask_refiner.sh \
    >"${job_log}" 2>&1

  variant_root="${OUTPUT_ROOT}/scanrefer/${variant}"
  run_dir="$(
    find "${variant_root}" -mindepth 1 -maxdepth 1 -type d \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "${run_dir}" ]]; then
    echo "missing run directory for ${variant}" >&2
    exit 5
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
    --profile v106 --baseline "${PROTECTED_V19}" \
    --checkpoint "${run_dir}/ckpt_epoch_last.pth" \
    --expected-epoch "${PANEL_MAX_EPOCH}" \
    --expected-step "${expected_step}" \
    --output "${run_dir}/audit_v106.json"
  chmod 0444 "${run_dir}/audit_completion_epoch_${PANEL_MAX_EPOCH}.json" \
    "${run_dir}/audit_v106.json"

  if command -v lsof >/dev/null 2>&1 \
      && lsof +D "${run_dir}" 2>/dev/null | grep -q '\.pth'; then
    echo "refusing to delete open V106 checkpoint files" >&2
    exit 6
  fi
  while IFS= read -r checkpoint; do
    resolved="$(readlink -f "${checkpoint}")"
    case "${resolved}" in
      "$(readlink -f "${run_dir}")"/*) ;;
      *) echo "checkpoint escapes run directory: ${checkpoint}" >&2; exit 6 ;;
    esac
  done < <(find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.pth')
  echo "[$(timestamp)] deleting audited, reproducible smoke weights in ${run_dir}"
  find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.pth' \
    -printf '%i %n %s %p\n' -delete
done

summary_path="${OUTPUT_ROOT}/v106_graph_mask_smoke_summary.json"
set +e
"${PYTHON_BIN}" scripts/summarize_v106_graph_mask_smoke.py \
  --output "${summary_path}" --epoch "${PANEL_MAX_EPOCH}" \
  --expected-sample-count "${PANEL_EXPECTED_SAMPLE_COUNT}" \
  --expected-rec025 64 --expected-rec050 57 --miou-margin 0.0003 \
  --v105-all-miou 0.3508351807200742 \
  --record "${variants[0]}" "${run_dirs[0]}" "${job_logs[0]}" "${graph_modes[0]}" \
  --record "${variants[1]}" "${run_dirs[1]}" "${job_logs[1]}" "${graph_modes[1]}"
summary_status=$?
set -e
chmod 0444 "${summary_path}"
if (( summary_status != 0 )); then
  echo "[$(timestamp)] V106 graph smoke gate failed: ${summary_path}" >&2
  exit "${summary_status}"
fi
echo "[$(timestamp)] V106 graph smoke gate passed: ${summary_path}"
