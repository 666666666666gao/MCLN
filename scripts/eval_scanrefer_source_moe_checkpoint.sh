#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  echo "CHECKPOINT_PATH=<checkpoint> is required" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

EVAL_EPOCH="${EVAL_EPOCH:-}"
if [[ ! "${EVAL_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "EVAL_EPOCH must be a non-negative integer" >&2
  exit 2
fi

ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
if [[ "${ALLOW_BUSY_GPU}" != "0" && "${ALLOW_BUSY_GPU}" != "1" ]]; then
  echo "ALLOW_BUSY_GPU must be 0 or 1" >&2
  exit 2
fi
if [[ "${ALLOW_BUSY_GPU}" == "0" ]] && command -v nvidia-smi >/dev/null; then
  BUSY_GPU_PIDS="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null || true
  )"
  if [[ -n "${BUSY_GPU_PIDS//[[:space:]]/}" ]]; then
    echo "GPU has an active compute process; refusing concurrent eval" >&2
    exit 3
  fi
fi

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT%/}/output/source_moe_independent_eval}"
EXP="${EXP:-ssq_moe_eval_epoch_${EVAL_EPOCH}}"
MASTER_PORT="${MASTER_PORT:-4465}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
RUN_AUDIT="${RUN_AUDIT:-1}"
BASELINE_METRICS="${BASELINE_METRICS:-${DATA_ROOT%/}/output/source_moe_v4_contract_eval/scannet,scanrefer/ssq_moe_v4_contract_eval/1785534811/eval_metrics_epoch_1.json}"
if [[ "${RUN_AUDIT}" != "0" && "${RUN_AUDIT}" != "1" ]]; then
  echo "RUN_AUDIT must be 0 or 1" >&2
  exit 2
fi
if [[ "${RUN_AUDIT}" == "1" && ! -f "${BASELINE_METRICS}" ]]; then
  echo "baseline metrics do not exist: ${BASELINE_METRICS}" >&2
  exit 2
fi

EVAL_MARKER="$(mktemp)"
trap 'rm -f "${EVAL_MARKER}"' EXIT

PHASE=joint \
JOINT_RESUME=0 \
START_EPOCH="${EVAL_EPOCH}" \
MAX_EPOCH="${EVAL_EPOCH}" \
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
DATA_ROOT="${DATA_ROOT}" \
LOG_DIR="${LOG_DIR}" \
EXP="${EXP}" \
MASTER_PORT="${MASTER_PORT}" \
PYTHON_BIN="${PYTHON_BIN}" \
EXPECTED_EVAL_SAMPLE_COUNT=9508 \
SOURCE_MOE_GATE_LOSS_WEIGHT=1.0 \
bash "${SCRIPT_DIR}/train_scanrefer_source_moe.sh" --eval "$@"

if [[ "${RUN_AUDIT}" == "1" ]]; then
  RUN_ROOT="${LOG_DIR}/scannet,scanrefer/${EXP}"
  LATEST_RUN=""
  for candidate in "${RUN_ROOT}"/*; do
    if [[ -d "${candidate}" && "${candidate}" -nt "${EVAL_MARKER}" ]]; then
      if [[ -z "${LATEST_RUN}" || "${candidate}" -nt "${LATEST_RUN}" ]]; then
        LATEST_RUN="${candidate}"
      fi
    fi
  done
  if [[ -z "${LATEST_RUN}" ]]; then
    echo "cannot locate the completed eval run under ${RUN_ROOT}" >&2
    exit 4
  fi
  METRICS_RECEIPT="${LATEST_RUN}/eval_metrics_epoch_${EVAL_EPOCH}.json"
  DIAGNOSTICS_RECEIPT="${LATEST_RUN}/source_choice_diagnostics_epoch_${EVAL_EPOCH}.json"
  if [[ ! -f "${METRICS_RECEIPT}" || ! -f "${DIAGNOSTICS_RECEIPT}" ]]; then
    echo "eval did not produce both required receipts in ${LATEST_RUN}" >&2
    exit 4
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_source_moe_candidate_oracle.py" \
    --metrics "${METRICS_RECEIPT}" \
    --diagnostics "${DIAGNOSTICS_RECEIPT}" \
    --baseline "${BASELINE_METRICS}" \
    --output "${LATEST_RUN}/source_moe_oracle_audit.json"
fi
