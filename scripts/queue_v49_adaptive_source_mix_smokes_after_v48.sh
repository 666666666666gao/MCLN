#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
PROTECTED_V19="${PROTECTED_V19:-${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_network_best_v19_rec025_0.581195_mask025_0.598233.pth}"
PROTECTED_V19_SHA256="${PROTECTED_V19_SHA256:-2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiment_output/v49_adaptive_source_mix}"
V48_SUMMARY="${V48_SUMMARY:-${ROOT_DIR}/experiment_output/v48_spatial_mask/v48_spatial_mask_smoke_summary.json}"
PANEL_MAX_EPOCH="${PANEL_MAX_EPOCH:-3}"
PANEL_BATCH_SIZE="${PANEL_BATCH_SIZE:-64}"
PANEL_EXPECTED_SAMPLE_COUNT="${PANEL_EXPECTED_SAMPLE_COUNT:-128}"
MIN_FREE_KIB="${MIN_FREE_KIB:-4194304}"
GPU_IDLE_WAIT_SECONDS="${GPU_IDLE_WAIT_SECONDS:-60}"
QUEUE_LOG="${QUEUE_LOG:-${OUTPUT_ROOT}/v49_after_v48_queue.log}"
LOCK_FILE="${LOCK_FILE:-${OUTPUT_ROOT}/v49_after_v48_queue.lock}"

for value in "${PANEL_MAX_EPOCH}" "${PANEL_BATCH_SIZE}" \
             "${PANEL_EXPECTED_SAMPLE_COUNT}" "${MIN_FREE_KIB}" \
             "${GPU_IDLE_WAIT_SECONDS}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "V49 queue integer settings must be positive" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another V49 queue owns ${LOCK_FILE}" >&2
  exit 3
fi
exec > >(tee -a "${QUEUE_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[$(timestamp)] waiting for the V48 spatial-mask smoke queue"
predecessor_pid="${PREDECESSOR_PID:-}"
if [[ -z "${predecessor_pid}" ]]; then
  predecessor_pid="$(pgrep -o -f \
    'bash scripts/queue_v48_spatial_mask_smokes_after_v47.sh' || true)"
fi
if [[ -n "${predecessor_pid}" ]]; then
  if [[ ! "${predecessor_pid}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREDECESSOR_PID must be a positive process id" >&2
    exit 2
  fi
  echo "[$(timestamp)] bound to V48 process ${predecessor_pid}; no log polling"
  tail --pid="${predecessor_pid}" -f /dev/null
fi
if [[ ! -f "${V48_SUMMARY}" ]]; then
  echo "V48 queue ended without a smoke summary: ${V48_SUMMARY}" >&2
  exit 5
fi

selection_path="${OUTPUT_ROOT}/selected_v48_mask_config.json"
"${PYTHON_BIN}" - "${V48_SUMMARY}" "${selection_path}" <<'PY'
import json
import os
import re
import sys

summary_path, output_path = sys.argv[1:]
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("schema") != "mcln-v48-smoke-panel-v1":
    raise SystemExit("V48 smoke summary schema is invalid")
if summary.get("pass") is not True:
    raise SystemExit("V48 smoke summary did not pass")
records = [record for record in summary.get("records", [])
           if record.get("pass") is True]
if not records:
    raise SystemExit("V48 smoke summary has no passing records")

def rates(record):
    receipt = record["receipt"]
    count = float(receipt["sample_count"])
    position = receipt["position"]["learned_selector"]
    mask = receipt["mask"]
    return (
        position["hits025"] / count,
        position["hits050"] / count,
        mask["hits025"] / count,
        mask["hits050"] / count,
        float(mask["miou"]),
    )

selected = max(
    records,
    key=lambda record: (
        min(rates(record)[0] / 0.59, rates(record)[1] / 0.49),
        rates(record)[0], rates(record)[1], rates(record)[2],
        rates(record)[3], rates(record)[4],
    ),
)
variant = selected.get("variant", "")
match = re.fullmatch(
    r"v48_smoke_cmw(010|025)_clw(000|005|010)_k(8|16)", variant
)
if match is None:
    raise SystemExit("V48 variant name is not a supported V49 initializer")
mask_code, lovasz_code, top_k = match.groups()
result = {
    "schema": "mcln-v49-mask-selection-v1",
    "summary": summary_path,
    "selected_variant": variant,
    "candidate_mask_code": mask_code,
    "candidate_lovasz_code": lovasz_code,
    "candidate_mask_loss_weight": int(mask_code) / 100.0,
    "candidate_lovasz_loss_weight": int(lovasz_code) / 100.0,
    "candidate_mask_top_k": int(top_k),
}
temporary = output_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output_path)
print(json.dumps(result, indent=2, sort_keys=True))
PY
read -r selected_mask_code selected_lovasz_code selected_top_k \
  selected_mask_weight selected_lovasz_weight < <(
  "${PYTHON_BIN}" - "${selection_path}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    value = json.load(handle)
print(
    value["candidate_mask_code"],
    value["candidate_lovasz_code"],
    value["candidate_mask_top_k"],
    value["candidate_mask_loss_weight"],
    value["candidate_lovasz_loss_weight"],
)
PY
)

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
  echo "V49 output filesystem has insufficient free space" >&2
  exit 4
fi

initialization_audit="${OUTPUT_ROOT}/v49_protected_v19_initialization.json"
"${PYTHON_BIN}" scripts/audit_joint_query_initialization.py \
  --checkpoint "${PROTECTED_V19}" --profile v49 \
  --output "${initialization_audit}"
chmod 0444 "${initialization_audit}"

while true; do
  compute_pids="$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)"
  if [[ -z "${compute_pids}" ]]; then
    break
  fi
  echo "[$(timestamp)] GPUs still have compute processes; waiting ${GPU_IDLE_WAIT_SECONDS}s"
  sleep "${GPU_IDLE_WAIT_SECONDS}"
done

source_mix_loss_codes=("000" "010" "025" "050")
source_mix_loss_weights=("0.00" "0.10" "0.25" "0.50")
variants=()
for code in "${source_mix_loss_codes[@]}"; do
  variants+=(
    "v49_mix_cmw${selected_mask_code}_clw${selected_lovasz_code}_k${selected_top_k}_smw${code}"
  )
done
ports=("4720" "4721" "4722" "4723")

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
    JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING=1 \
    JOINT_QUERY_QUALITY_SPATIAL_MASK_HIDDEN_DIM=32 \
    JOINT_QUERY_QUALITY_MAX_SPATIAL_MASK_DELTA=2.0 \
    JOINT_QUERY_QUALITY_MAX_SOURCE_MIX_DELTA=1.0 \
    JOINT_QUERY_QUALITY_SOURCE_MIX_TEMPERATURE=0.5 \
    JOINT_QUERY_QUALITY_SOURCE_MIX_LOSS_WEIGHT="${source_mix_loss_weights[${gpu}]}" \
    JOINT_QUERY_QUALITY_SOURCE_MIX_ALIGNMENT_TEMPERATURE=0.25 \
    JOINT_QUERY_QUALITY_MAX_MASK_ALPHA_DELTA=1.0 \
    JOINT_QUERY_QUALITY_MAX_MASK_LOGIT_BIAS=2.0 \
    JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT="${selected_mask_weight}" \
    JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT="${selected_lovasz_weight}" \
    JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K="${selected_top_k}" \
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
  run_dir="$(find "${variant_root}" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
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
    --profile v49 --baseline "${PROTECTED_V19}" \
    --checkpoint "${run_dir}/ckpt_epoch_last.pth" \
    --expected-epoch "${PANEL_MAX_EPOCH}" \
    --expected-step "${expected_step}" \
    --expected-source-mix-loss-weight "${source_mix_loss_weights[${index}]}" \
    --expected-source-mix-alignment-temperature 0.25 \
    --output "${run_dir}/audit_v49.json"
done

summary_path="${OUTPUT_ROOT}/v49_adaptive_source_mix_smoke_summary.json"
lovasz_args=()
if [[ "${selected_lovasz_weight}" != "0" \
      && "${selected_lovasz_weight}" != "0.0" ]]; then
  for variant in "${variants[@]}"; do
    lovasz_args+=(--require-lovasz-variant "${variant}")
  done
fi
"${PYTHON_BIN}" scripts/summarize_v41_smoke_panel.py \
  --output "${summary_path}" --profile v49 \
  --require-candidate-mask \
  "${lovasz_args[@]}" \
  --epoch "${PANEL_MAX_EPOCH}" \
  --expected-sample-count "${PANEL_EXPECTED_SAMPLE_COUNT}" \
  --record "${variants[0]}" "${run_dirs[0]}" "${job_logs[0]}" \
  --record "${variants[1]}" "${run_dirs[1]}" "${job_logs[1]}" \
  --record "${variants[2]}" "${run_dirs[2]}" "${job_logs[2]}" \
  --record "${variants[3]}" "${run_dirs[3]}" "${job_logs[3]}"

for run_dir in "${run_dirs[@]}"; do
  echo "[$(timestamp)] removing audited V49 debug checkpoints from ${run_dir}"
  find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.pth' \
    -print -delete
done
echo "[$(timestamp)] V49 adaptive-source smoke panel completed: ${summary_path}"
