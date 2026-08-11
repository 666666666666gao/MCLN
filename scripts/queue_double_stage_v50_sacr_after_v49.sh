#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
PREDECESSOR_PID="${PREDECESSOR_PID:-}"
PROTECTED_V19="${PROTECTED_V19:-${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_network_best_v19_rec025_0.581195_mask025_0.598233.pth}"
PROTECTED_V19_SHA256="${PROTECTED_V19_SHA256:-2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe}"
V49_ROOT="${V49_ROOT:-${ROOT_DIR}/experiment_output/double_stage_v49_formal}"
V49_SELECTION="${V49_SELECTION:-${V49_ROOT}/selected_v49_formal_config.json}"
V49_COMPLETION_AUDIT="${V49_COMPLETION_AUDIT:-${V49_ROOT}/formal_completion_audit.json}"
V49_CHECKPOINT_AUDIT="${V49_CHECKPOINT_AUDIT:-${V49_ROOT}/formal_v49_checkpoint_audit.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiment_output/v50_sacr}"
SMOKE_LOG_DIR="${SMOKE_LOG_DIR:-${DATA_ROOT%/}/output/double_stage_v50_sacr_smoke}"
FORMAL_LOG_DIR="${FORMAL_LOG_DIR:-${DATA_ROOT%/}/output/double_stage_v50_sacr_formal}"
SMOKE_MAX_EPOCH="${SMOKE_MAX_EPOCH:-2}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-12}"
SMOKE_EXPECTED_SAMPLE_COUNT="${SMOKE_EXPECTED_SAMPLE_COUNT:-128}"
SMOKE_MASTER_PORT="${SMOKE_MASTER_PORT:-4820}"
FORMAL_MAX_EPOCH="${FORMAL_MAX_EPOCH:-80}"
FORMAL_BATCH_SIZE="${FORMAL_BATCH_SIZE:-12}"
FORMAL_EXPECTED_SAMPLE_COUNT="${FORMAL_EXPECTED_SAMPLE_COUNT:-9508}"
FORMAL_MASTER_PORT="${FORMAL_MASTER_PORT:-4821}"
MIN_FREE_KIB="${MIN_FREE_KIB:-8388608}"
GPU_IDLE_WAIT_SECONDS="${GPU_IDLE_WAIT_SECONDS:-60}"
QUEUE_LOG="${QUEUE_LOG:-${OUTPUT_ROOT}/v50_sacr_after_v49_queue.log}"
LOCK_FILE="${LOCK_FILE:-${OUTPUT_ROOT}/v50_sacr_after_v49_queue.lock}"

for value in "${SMOKE_MAX_EPOCH}" "${SMOKE_BATCH_SIZE}" \
             "${SMOKE_EXPECTED_SAMPLE_COUNT}" "${SMOKE_MASTER_PORT}" \
             "${FORMAL_MAX_EPOCH}" "${FORMAL_BATCH_SIZE}" \
             "${FORMAL_EXPECTED_SAMPLE_COUNT}" "${FORMAL_MASTER_PORT}" \
             "${MIN_FREE_KIB}" "${GPU_IDLE_WAIT_SECONDS}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "V50 queue integer settings must be positive" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${SMOKE_LOG_DIR}" "${FORMAL_LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another V50 SACR queue owns ${LOCK_FILE}" >&2
  exit 3
fi
exec > >(tee -a "${QUEUE_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

require_pass_receipt() {
  local path="$1"
  local label="$2"
  "${PYTHON_BIN}" - "${path}" "${label}" <<'PY'
import json
import sys
path, label = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    value = json.load(handle)
if not isinstance(value, dict) or value.get("pass") is not True:
    raise SystemExit("{} did not pass: {}".format(label, path))
PY
}

infer_steps_per_epoch() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import re
import sys
log_path, max_epoch_text = sys.argv[1:]
max_epoch = int(max_epoch_text)
pattern = re.compile(r"Train: \[(\d+)\]\[\d+/(\d+)\]")
totals = {}
with open(log_path, "r", encoding="utf-8") as handle:
    for line in handle:
        match = pattern.search(line)
        if match is not None:
            totals.setdefault(int(match.group(1)), set()).add(
                int(match.group(2))
            )
for epoch in (1, max_epoch):
    if epoch not in totals or len(totals[epoch]) != 1:
        raise SystemExit(
            "log does not prove one steps-per-epoch value for epoch {}"
            .format(epoch)
        )
first = next(iter(totals[1]))
last = next(iter(totals[max_epoch]))
if first <= 0 or first != last:
    raise SystemExit(
        "steps-per-epoch changed from {} to {}".format(first, last)
    )
print(first)
PY
}

latest_run_dir() {
  local root="$1"
  find "${root}" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

echo "[$(timestamp)] waiting for formal V49 queue"
if [[ -z "${PREDECESSOR_PID}" ]]; then
  PREDECESSOR_PID="$(pgrep -o -f \
    'bash scripts/queue_double_stage_v49_formal_after_smoke.sh' || true)"
fi
if [[ -n "${PREDECESSOR_PID}" ]]; then
  if [[ ! "${PREDECESSOR_PID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREDECESSOR_PID must be a positive process id" >&2
    exit 2
  fi
  echo "[$(timestamp)] bound to formal V49 process ${PREDECESSOR_PID}; no log polling"
  tail --pid="${PREDECESSOR_PID}" -f /dev/null
fi

for required in "${V49_SELECTION}" "${V49_COMPLETION_AUDIT}" \
                "${V49_CHECKPOINT_AUDIT}" "${PROTECTED_V19}"; do
  if [[ ! -f "${required}" ]]; then
    echo "required V50 input is missing: ${required}" >&2
    exit 4
  fi
done
require_pass_receipt "${V49_COMPLETION_AUDIT}" "V49 completion audit"
require_pass_receipt "${V49_CHECKPOINT_AUDIT}" "V49 checkpoint audit"
actual_sha256="$(sha256sum "${PROTECTED_V19}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${PROTECTED_V19_SHA256}" ]]; then
  echo "protected V19 SHA-256 changed" >&2
  exit 4
fi
free_kib="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
if [[ ! "${free_kib}" =~ ^[1-9][0-9]*$ \
      || "${free_kib}" -lt "${MIN_FREE_KIB}" ]]; then
  echo "V50 output filesystem has insufficient free space" >&2
  exit 4
fi

selection_path="${OUTPUT_ROOT}/selected_v50_sacr_config.json"
"${PYTHON_BIN}" - "${V49_SELECTION}" "${selection_path}" <<'PY'
import json
import os
import sys
source_path, output_path = sys.argv[1:]
with open(source_path, "r", encoding="utf-8") as handle:
    source = json.load(handle)
if source.get("schema") != "mcln-v49-formal-selection-v1":
    raise SystemExit("V49 formal selection schema is invalid")
required = (
    "joint_query_quality_candidate_mask_loss_weight",
    "joint_query_quality_candidate_lovasz_loss_weight",
    "joint_query_quality_candidate_mask_top_k",
    "joint_query_quality_source_mix_loss_weight",
)
if any(name not in source for name in required):
    raise SystemExit("V49 formal selection is incomplete")
selected_mix = float(source[required[-1]])
result = {
    "schema": "mcln-v50-sacr-selection-v1",
    "v49_selection": source_path,
    "candidate_mask_loss_weight": float(source[required[0]]),
    "candidate_lovasz_loss_weight": float(source[required[1]]),
    "candidate_mask_top_k": int(source[required[2]]),
    "v49_selected_source_mix_loss_weight": selected_mix,
    # SACR needs an explicit differentiable source-alignment signal.
    "source_mix_loss_weight": max(selected_mix, 0.25),
    "source_mix_alignment_temperature": 0.25,
    "source_mix_query_focus_weight": 0.75,
    "parent_source_names": [
        "default", "contrastive_text", "mask_text"
    ],
    "joint_source_names": [
        "default", "contrastive_text", "mask_text", "sacr_structured"
    ],
    "sacr_residual_scale_init": 0.1,
    "sacr_min_parse_confidence": 0.0,
}
temporary = output_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output_path)
print(json.dumps(result, indent=2, sort_keys=True))
PY

read -r candidate_mask_weight candidate_lovasz_weight candidate_top_k \
  source_mix_loss_weight source_mix_query_focus_weight < <(
  "${PYTHON_BIN}" - "${selection_path}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    value = json.load(handle)
print(
    value["candidate_mask_loss_weight"],
    value["candidate_lovasz_loss_weight"],
    value["candidate_mask_top_k"],
    value["source_mix_loss_weight"],
    value["source_mix_query_focus_weight"],
)
PY
)

data_audits=()
for structured_split in train val; do
  data_audit="${OUTPUT_ROOT}/sacr_structured_data_${structured_split}_audit.json"
  "${PYTHON_BIN}" scripts/audit_sacr_structured_data.py \
    --data-root "${DATA_ROOT}" --datasets scanrefer,nr3d,sr3d \
    --split "${structured_split}" --output "${data_audit}"
  require_pass_receipt \
    "${data_audit}" "SACR ${structured_split} structured-data audit"
  data_audits+=("${data_audit}")
done

initialization_audit="${OUTPUT_ROOT}/v50_sacr_protected_v19_initialization.json"
"${PYTHON_BIN}" scripts/audit_joint_query_initialization.py \
  --checkpoint "${PROTECTED_V19}" --profile v50_sacr \
  --output "${initialization_audit}"
require_pass_receipt "${initialization_audit}" "V50 initialization audit"
chmod 0444 "${data_audits[@]}" "${initialization_audit}"

while true; do
  compute_pids="$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)"
  if [[ -z "${compute_pids}" ]]; then
    break
  fi
  echo "[$(timestamp)] GPUs still have compute processes; waiting ${GPU_IDLE_WAIT_SECONDS}s"
  sleep "${GPU_IDLE_WAIT_SECONDS}"
done

smoke_exp="v50_sacr_four_source_smoke_e1_e${SMOKE_MAX_EPOCH}_b${SMOKE_BATCH_SIZE}x4"
smoke_log="${OUTPUT_ROOT}/${smoke_exp}_launcher.log"
echo "[$(timestamp)] starting four-GPU V50 SACR smoke ${smoke_exp}"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_ROOT="${DATA_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
NPROC_PER_NODE=4 MASTER_PORT="${SMOKE_MASTER_PORT}" \
CHECKPOINT_PATH="${PROTECTED_V19}" LOG_DIR="${SMOKE_LOG_DIR}" \
EXP="${smoke_exp}" MODEL_STAGE=two SOURCE_ARBITER=moe \
BATCH_SIZE="${SMOKE_BATCH_SIZE}" NUM_WORKERS=4 \
DATALOADER_PREFETCH_FACTOR=1 PERSISTENT_TRAIN_WORKERS=0 \
START_EPOCH=1 MAX_EPOCH="${SMOKE_MAX_EPOCH}" VAL_FREQ=1 PRINT_FREQ=1 \
EXPECTED_EVAL_SAMPLE_COUNT="${SMOKE_EXPECTED_SAMPLE_COUNT}" DEBUG=1 \
JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION=1 \
JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE=1 \
JOINT_QUERY_QUALITY_USE_SPATIAL_MASK_REFINER=1 \
JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING=1 \
JOINT_QUERY_QUALITY_USE_SOURCE_DISTRIBUTION_RELIABILITY=0 \
JOINT_QUERY_QUALITY_SOURCE_NAMES=default,contrastive_text,mask_text,sacr_structured \
JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT="${candidate_mask_weight}" \
JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT="${candidate_lovasz_weight}" \
JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K="${candidate_top_k}" \
JOINT_QUERY_QUALITY_ANCHOR_LOSS_WEIGHT=0.5 \
JOINT_QUERY_QUALITY_MASK_WEIGHT=0.25 \
JOINT_QUERY_QUALITY_TEMPERATURE=0.25 \
JOINT_QUERY_QUALITY_MAX_SPATIAL_MASK_DELTA=2.0 \
JOINT_QUERY_QUALITY_MAX_SOURCE_MIX_DELTA=1.0 \
JOINT_QUERY_QUALITY_SOURCE_MIX_TEMPERATURE=0.5 \
JOINT_QUERY_QUALITY_SOURCE_MIX_LOSS_WEIGHT="${source_mix_loss_weight}" \
JOINT_QUERY_QUALITY_SOURCE_MIX_ALIGNMENT_TEMPERATURE=0.25 \
JOINT_QUERY_QUALITY_SOURCE_MIX_QUERY_FOCUS_WEIGHT="${source_mix_query_focus_weight}" \
USE_SACR_SOURCE=1 SACR_RESIDUAL_SCALE_INIT=0.1 \
SACR_MIN_PARSE_CONFIDENCE=0.0 \
bash scripts/train_scanrefer_joint_query_quality.sh \
  >"${smoke_log}" 2>&1

smoke_run_dir="$(latest_run_dir \
  "${SMOKE_LOG_DIR}/scanrefer/${smoke_exp}")"
if [[ -z "${smoke_run_dir}" ]]; then
  echo "V50 smoke run directory is missing" >&2
  exit 6
fi
"${PYTHON_BIN}" scripts/audit_training_completion.py \
  --metrics "${smoke_run_dir}/eval_metrics_epoch_${SMOKE_MAX_EPOCH}.json" \
  --checkpoint "${smoke_run_dir}/ckpt_epoch_last.pth" \
  --expected-epoch "${SMOKE_MAX_EPOCH}" \
  --expected-sample-count "${SMOKE_EXPECTED_SAMPLE_COUNT}" \
  --require-position-subgroups \
  --output "${smoke_run_dir}/audit_completion_epoch_${SMOKE_MAX_EPOCH}.json"
smoke_steps_per_epoch="$(infer_steps_per_epoch \
  "${smoke_run_dir}/log.txt" "${SMOKE_MAX_EPOCH}")"
smoke_optimizer_step=$((smoke_steps_per_epoch * SMOKE_MAX_EPOCH))
"${PYTHON_BIN}" scripts/audit_source_moe_checkpoint.py \
  --profile v50_sacr --baseline "${PROTECTED_V19}" \
  --checkpoint "${smoke_run_dir}/ckpt_epoch_last.pth" \
  --expected-epoch "${SMOKE_MAX_EPOCH}" \
  --expected-step "${smoke_optimizer_step}" \
  --expected-source-mix-loss-weight "${source_mix_loss_weight}" \
  --expected-source-mix-alignment-temperature 0.25 \
  --expected-source-mix-query-focus-weight "${source_mix_query_focus_weight}" \
  --output "${smoke_run_dir}/audit_v50_sacr.json"

summary_args=()
if [[ "${candidate_lovasz_weight}" != "0" \
      && "${candidate_lovasz_weight}" != "0.0" ]]; then
  summary_args+=(--require-lovasz-variant "${smoke_exp}")
fi
smoke_summary="${OUTPUT_ROOT}/v50_sacr_smoke_summary.json"
"${PYTHON_BIN}" scripts/summarize_v41_smoke_panel.py \
  --output "${smoke_summary}" --profile v50_sacr \
  --require-candidate-mask "${summary_args[@]}" \
  --epoch "${SMOKE_MAX_EPOCH}" \
  --expected-sample-count "${SMOKE_EXPECTED_SAMPLE_COUNT}" \
  --record "${smoke_exp}" "${smoke_run_dir}" "${smoke_log}"
require_pass_receipt "${smoke_summary}" "V50 SACR smoke summary"

echo "[$(timestamp)] removing audited V50 smoke checkpoints"
find "${smoke_run_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.pth' \
  -print -delete

formal_exp="double_v50_sacr_e1_e${FORMAL_MAX_EPOCH}_b${FORMAL_BATCH_SIZE}x4"
formal_log="${OUTPUT_ROOT}/${formal_exp}_launcher.log"
echo "[$(timestamp)] starting formal four-GPU V50 SACR ${formal_exp}"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_ROOT="${DATA_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
NPROC_PER_NODE=4 MASTER_PORT="${FORMAL_MASTER_PORT}" \
CHECKPOINT_PATH="${PROTECTED_V19}" LOG_DIR="${FORMAL_LOG_DIR}" \
EXP="${formal_exp}" MODEL_STAGE=two SOURCE_ARBITER=moe \
BATCH_SIZE="${FORMAL_BATCH_SIZE}" NUM_WORKERS=4 \
DATALOADER_PREFETCH_FACTOR=1 PERSISTENT_TRAIN_WORKERS=0 \
START_EPOCH=1 MAX_EPOCH="${FORMAL_MAX_EPOCH}" VAL_FREQ=1 PRINT_FREQ=100 \
EXPECTED_EVAL_SAMPLE_COUNT="${FORMAL_EXPECTED_SAMPLE_COUNT}" DEBUG=0 \
JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION=1 \
JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE=1 \
JOINT_QUERY_QUALITY_USE_SPATIAL_MASK_REFINER=1 \
JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING=1 \
JOINT_QUERY_QUALITY_USE_SOURCE_DISTRIBUTION_RELIABILITY=0 \
JOINT_QUERY_QUALITY_SOURCE_NAMES=default,contrastive_text,mask_text,sacr_structured \
JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT="${candidate_mask_weight}" \
JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT="${candidate_lovasz_weight}" \
JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K="${candidate_top_k}" \
JOINT_QUERY_QUALITY_ANCHOR_LOSS_WEIGHT=0.5 \
JOINT_QUERY_QUALITY_MASK_WEIGHT=0.25 \
JOINT_QUERY_QUALITY_TEMPERATURE=0.25 \
JOINT_QUERY_QUALITY_MAX_SPATIAL_MASK_DELTA=2.0 \
JOINT_QUERY_QUALITY_MAX_SOURCE_MIX_DELTA=1.0 \
JOINT_QUERY_QUALITY_SOURCE_MIX_TEMPERATURE=0.5 \
JOINT_QUERY_QUALITY_SOURCE_MIX_LOSS_WEIGHT="${source_mix_loss_weight}" \
JOINT_QUERY_QUALITY_SOURCE_MIX_ALIGNMENT_TEMPERATURE=0.25 \
JOINT_QUERY_QUALITY_SOURCE_MIX_QUERY_FOCUS_WEIGHT="${source_mix_query_focus_weight}" \
USE_SACR_SOURCE=1 SACR_RESIDUAL_SCALE_INIT=0.1 \
SACR_MIN_PARSE_CONFIDENCE=0.0 \
bash scripts/train_scanrefer_joint_query_quality.sh \
  >"${formal_log}" 2>&1

formal_run_dir="$(latest_run_dir \
  "${FORMAL_LOG_DIR}/scanrefer/${formal_exp}")"
if [[ -z "${formal_run_dir}" ]]; then
  echo "V50 formal run directory is missing" >&2
  exit 6
fi
"${PYTHON_BIN}" scripts/audit_training_completion.py \
  --metrics "${formal_run_dir}/eval_metrics_epoch_${FORMAL_MAX_EPOCH}.json" \
  --checkpoint "${formal_run_dir}/ckpt_epoch_last.pth" \
  --expected-epoch "${FORMAL_MAX_EPOCH}" \
  --expected-sample-count "${FORMAL_EXPECTED_SAMPLE_COUNT}" \
  --require-position-subgroups \
  --output "${OUTPUT_ROOT}/formal_completion_audit.json"
formal_steps_per_epoch="$(infer_steps_per_epoch \
  "${formal_run_dir}/log.txt" "${FORMAL_MAX_EPOCH}")"
formal_optimizer_step=$((formal_steps_per_epoch * FORMAL_MAX_EPOCH))
"${PYTHON_BIN}" scripts/audit_source_moe_checkpoint.py \
  --profile v50_sacr --baseline "${PROTECTED_V19}" \
  --checkpoint "${formal_run_dir}/ckpt_epoch_last.pth" \
  --expected-epoch "${FORMAL_MAX_EPOCH}" \
  --expected-step "${formal_optimizer_step}" \
  --expected-source-mix-loss-weight "${source_mix_loss_weight}" \
  --expected-source-mix-alignment-temperature 0.25 \
  --expected-source-mix-query-focus-weight "${source_mix_query_focus_weight}" \
  --output "${OUTPUT_ROOT}/formal_v50_sacr_checkpoint_audit.json"
echo "[$(timestamp)] formal double-stage V50 SACR completed and audited"
