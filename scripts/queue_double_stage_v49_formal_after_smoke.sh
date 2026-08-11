#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
PREDECESSOR_PID="${PREDECESSOR_PID:-}"
V49_SUMMARY="${V49_SUMMARY:-${ROOT_DIR}/experiment_output/v49_adaptive_source_mix/v49_adaptive_source_mix_smoke_summary.json}"
PROTECTED_V19="${PROTECTED_V19:-${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_network_best_v19_rec025_0.581195_mask025_0.598233.pth}"
PROTECTED_V19_SHA256="${PROTECTED_V19_SHA256:-2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiment_output/double_stage_v49_formal}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT%/}/output/double_stage_v49_formal}"
FORMAL_MAX_EPOCH="${FORMAL_MAX_EPOCH:-80}"
FORMAL_BATCH_SIZE="${FORMAL_BATCH_SIZE:-12}"
FORMAL_EXPECTED_SAMPLE_COUNT="${FORMAL_EXPECTED_SAMPLE_COUNT:-9508}"
FORMAL_MASTER_PORT="${FORMAL_MASTER_PORT:-4814}"
MIN_FREE_KIB="${MIN_FREE_KIB:-7340032}"
GPU_IDLE_WAIT_SECONDS="${GPU_IDLE_WAIT_SECONDS:-60}"
QUEUE_LOG="${QUEUE_LOG:-${OUTPUT_ROOT}/formal_queue.log}"
LOCK_FILE="${LOCK_FILE:-${OUTPUT_ROOT}/formal_queue.lock}"

for value in "${FORMAL_MAX_EPOCH}" "${FORMAL_BATCH_SIZE}" \
             "${FORMAL_EXPECTED_SAMPLE_COUNT}" "${FORMAL_MASTER_PORT}" \
             "${MIN_FREE_KIB}" "${GPU_IDLE_WAIT_SECONDS}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "formal queue integer settings must be positive" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another formal double-stage queue owns ${LOCK_FILE}" >&2
  exit 3
fi
exec > >(tee -a "${QUEUE_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[$(timestamp)] waiting for V49 adaptive-source smoke summary"
if [[ -z "${PREDECESSOR_PID}" ]]; then
  PREDECESSOR_PID="$(pgrep -o -f \
    'bash scripts/queue_v49_adaptive_source_mix_smokes_after_v48.sh' \
    || true)"
fi
if [[ -n "${PREDECESSOR_PID}" ]]; then
  if [[ ! "${PREDECESSOR_PID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREDECESSOR_PID must be a positive process id" >&2
    exit 2
  fi
  echo "[$(timestamp)] bound to V49 process ${PREDECESSOR_PID}; no log polling"
  tail --pid="${PREDECESSOR_PID}" -f /dev/null
fi

if [[ ! -f "${V49_SUMMARY}" ]]; then
  echo "V49 smoke summary is missing: ${V49_SUMMARY}" >&2
  exit 5
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
  echo "formal output filesystem has insufficient free space: ${free_kib:-unknown} KiB" >&2
  exit 4
fi

selection_path="${OUTPUT_ROOT}/selected_v49_formal_config.json"
"${PYTHON_BIN}" - "${V49_SUMMARY}" "${selection_path}" <<'PY'
import json
import os
import re
import sys

summary_path, output_path = sys.argv[1:]
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("schema") != "mcln-v49-smoke-panel-v1":
    raise SystemExit("V49 smoke summary schema is invalid")
if summary.get("pass") is not True:
    raise SystemExit("V49 smoke panel did not pass")
records = [record for record in summary.get("records", [])
           if record.get("pass") is True]
if not records:
    raise SystemExit("V49 smoke summary has no passing records")

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
    r"v49_mix_cmw(010|025)_clw(000|005|010)_k(8|16)_smw(000|010|025|050)",
    variant,
)
if match is None:
    raise SystemExit("V49 variant name is not a supported formal profile")
mask_weight, lovasz_weight, top_k, source_mix_weight = match.groups()
result = {
    "schema": "mcln-v49-formal-selection-v1",
    "summary": summary_path,
    "selected_variant": variant,
    "selected_rates": {
        "rec_acc025": rates(selected)[0],
        "rec_acc050": rates(selected)[1],
        "mask_acc025": rates(selected)[2],
        "mask_acc050": rates(selected)[3],
        "mask_miou": rates(selected)[4],
    },
    "joint_query_quality_candidate_mask_loss_weight": int(mask_weight) / 100.0,
    "joint_query_quality_candidate_lovasz_loss_weight": int(lovasz_weight) / 100.0,
    "joint_query_quality_candidate_mask_top_k": int(top_k),
    "joint_query_quality_source_mix_loss_weight": (
        int(source_mix_weight) / 100.0
    ),
}
temporary = output_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output_path)
print(json.dumps(result, indent=2, sort_keys=True))
PY

read -r selected_variant candidate_mask_weight candidate_lovasz_weight \
  candidate_top_k source_mix_loss_weight < <(
  "${PYTHON_BIN}" - "${selection_path}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    value = json.load(handle)
print(
    value["selected_variant"],
    value["joint_query_quality_candidate_mask_loss_weight"],
    value["joint_query_quality_candidate_lovasz_loss_weight"],
    value["joint_query_quality_candidate_mask_top_k"],
    value["joint_query_quality_source_mix_loss_weight"],
)
PY
)
formal_exp="double_v49_${selected_variant}_e1_e${FORMAL_MAX_EPOCH}_b${FORMAL_BATCH_SIZE}x4"
formal_log="${OUTPUT_ROOT}/${formal_exp}_launcher.log"
echo "[$(timestamp)] selected ${selected_variant}; starting formal ${formal_exp}"

while true; do
  compute_pids="$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)"
  if [[ -z "${compute_pids}" ]]; then
    break
  fi
  echo "[$(timestamp)] GPUs still have compute processes; waiting ${GPU_IDLE_WAIT_SECONDS}s"
  sleep "${GPU_IDLE_WAIT_SECONDS}"
done

CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_ROOT="${DATA_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
NPROC_PER_NODE=4 \
MASTER_PORT="${FORMAL_MASTER_PORT}" \
CHECKPOINT_PATH="${PROTECTED_V19}" \
LOG_DIR="${LOG_DIR}" \
EXP="${formal_exp}" \
MODEL_STAGE=two \
SOURCE_ARBITER=moe \
BATCH_SIZE="${FORMAL_BATCH_SIZE}" \
NUM_WORKERS=4 \
DATALOADER_PREFETCH_FACTOR=1 \
START_EPOCH=1 \
MAX_EPOCH="${FORMAL_MAX_EPOCH}" \
VAL_FREQ=1 \
PRINT_FREQ=100 \
EXPECTED_EVAL_SAMPLE_COUNT="${FORMAL_EXPECTED_SAMPLE_COUNT}" \
DEBUG=0 \
JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION=1 \
JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE=1 \
JOINT_QUERY_QUALITY_USE_SPATIAL_MASK_REFINER=1 \
JOINT_QUERY_QUALITY_USE_ADAPTIVE_SOURCE_MIXING=1 \
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
bash scripts/train_scanrefer_joint_query_quality.sh \
  >"${formal_log}" 2>&1

formal_root="${LOG_DIR}/scanrefer/${formal_exp}"
formal_run_dir="$(find "${formal_root}" -mindepth 1 -maxdepth 1 -type d \
  -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "${formal_run_dir}" ]]; then
  echo "formal run directory is missing" >&2
  exit 6
fi
"${PYTHON_BIN}" scripts/audit_training_completion.py \
  --metrics "${formal_run_dir}/eval_metrics_epoch_${FORMAL_MAX_EPOCH}.json" \
  --checkpoint "${formal_run_dir}/ckpt_epoch_last.pth" \
  --expected-epoch "${FORMAL_MAX_EPOCH}" \
  --expected-sample-count "${FORMAL_EXPECTED_SAMPLE_COUNT}" \
  --require-position-subgroups \
  --output "${OUTPUT_ROOT}/formal_completion_audit.json"
formal_steps_per_epoch="$("${PYTHON_BIN}" - \
  "${formal_run_dir}/log.txt" "${FORMAL_MAX_EPOCH}" <<'PY'
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
            "formal log does not prove one steps-per-epoch value for epoch {}"
            .format(epoch)
        )
first = next(iter(totals[1]))
last = next(iter(totals[max_epoch]))
if first <= 0 or first != last:
    raise SystemExit(
        "formal steps-per-epoch changed from {} to {}".format(first, last)
    )
print(first)
PY
)"
formal_optimizer_step=$((formal_steps_per_epoch * FORMAL_MAX_EPOCH))
"${PYTHON_BIN}" scripts/audit_source_moe_checkpoint.py \
  --profile v49 --baseline "${PROTECTED_V19}" \
  --checkpoint "${formal_run_dir}/ckpt_epoch_last.pth" \
  --expected-epoch "${FORMAL_MAX_EPOCH}" \
  --expected-step "${formal_optimizer_step}" \
  --expected-source-mix-loss-weight "${source_mix_loss_weight}" \
  --expected-source-mix-alignment-temperature 0.25 \
  --output "${OUTPUT_ROOT}/formal_v49_checkpoint_audit.json"
echo "[$(timestamp)] formal double-stage V49 run completed and audited"
