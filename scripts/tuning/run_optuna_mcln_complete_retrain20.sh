#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${REPO_ROOT}/pretained model/ckpt_epoch_54.pth}"
BASE_SHA256="${BASE_SHA256:-a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d}"
BASE_SIZE="${BASE_SIZE:-793041121}"
PP_CHECKPOINT="${PP_CHECKPOINT:-${DATA_ROOT%/}/gf_detector_l6o256.pth}"
N_TRIALS="${N_TRIALS:-20}"
GPU="${GPU:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="${RUN_NAME:-mcln_complete_retrain20_${RUN_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT%/}/output/tuning/${RUN_NAME}}"
STUDY_NAME="${STUDY_NAME:-${RUN_NAME}}"
STORAGE="${STORAGE:-sqlite:///${OUTPUT_ROOT}/optuna.db}"

if [[ "${N_TRIALS}" != "20" ]]; then
  echo "formal MCLN search requires N_TRIALS=20" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/control" "${OUTPUT_ROOT}/provenance"
chmod 0444 "${BASE_CHECKPOINT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/pointnet2:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" scripts/tuning/mcln_retrain_provenance.py \
  --repo-root "${REPO_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --base-checkpoint "${BASE_CHECKPOINT}" \
  --base-sha256 "${BASE_SHA256}" \
  --base-size "${BASE_SIZE}" \
  --base-mode 0444 \
  --pp-checkpoint "${PP_CHECKPOINT}" \
  --python-bin "${PYTHON_BIN}" \
  > "${OUTPUT_ROOT}/control/provenance_stdout.log" 2>&1

PROVENANCE_MANIFEST="${OUTPUT_ROOT}/provenance/run_manifest.json"
STDOUT_PATH="${OUTPUT_ROOT}/control/orchestrator_stdout.log"
PID_PATH="${OUTPUT_ROOT}/control/orchestrator.pid"
COMMAND_PATH="${OUTPUT_ROOT}/control/orchestrator_command.txt"

ORCHESTRATOR_COMMAND=(
  "${PYTHON_BIN}"
  "scripts/tuning/optuna_mcln_complete_retrain.py"
  --study-name "${STUDY_NAME}"
  --storage "${STORAGE}"
  --output-root "${OUTPUT_ROOT}"
  --data-root "${DATA_ROOT}"
  --pp-checkpoint "${PP_CHECKPOINT}"
  --base-checkpoint "${BASE_CHECKPOINT}"
  --base-sha256 "${BASE_SHA256}"
  --gpu "${GPU}"
  --master-port-base "${MASTER_PORT_BASE}"
  --target-successful-trials "${N_TRIALS}"
  --max-process-attempts 60
  --python-bin "${PYTHON_BIN}"
  --repo-root "${REPO_ROOT}"
  --provenance-manifest "${PROVENANCE_MANIFEST}"
)

printf '%q ' "${ORCHESTRATOR_COMMAND[@]}" > "${COMMAND_PATH}"
printf '\n' >> "${COMMAND_PATH}"
printf '%s\n' "${BASHPID}" > "${PID_PATH}"
printf '%s\n' "${STDOUT_PATH}" > "${OUTPUT_ROOT}/control/stdout_path.txt"

cd "${REPO_ROOT}"
exec "${ORCHESTRATOR_COMMAND[@]}" >> "${STDOUT_PATH}" 2>&1
