#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}" || exit 1

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-mcln_source_choice_continue_optuna20_${STAMP}}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/DATA_ROOT/output/tuning/${RUN_NAME}}
REPORT_DIR=${REPORT_DIR:-${OUTPUT_ROOT}/reports}
STORAGE=${STORAGE:-sqlite:///${OUTPUT_ROOT}/optuna.db}
GPU=${GPU:-0}
N_TRIALS=${N_TRIALS:-20}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-/root/autodl-tmp/DATA_ROOT/output/logs/scanrefer/MCLN_source_choice_full_joint_restart_save1_keep3_seed0_20260620_110332/1781953653}
ACC25_CKPT=${ACC25_CKPT:-${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_best_backbone_acc025_0.582878_component.pth}
ACC50_CKPT=${ACC50_CKPT:-${SOURCE_RUN_DIR}/best_available_rec_acc050_epoch68.pth}

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}"

exec "${PYTHON_BIN}" scripts/tuning/optuna_mcln_source_choice_continue.py \
  --study-name "${RUN_NAME}" \
  --storage "${STORAGE}" \
  --n-trials "${N_TRIALS}" \
  --output-root "${OUTPUT_ROOT}/logs" \
  --report-dir "${REPORT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --acc25-checkpoint "${ACC25_CKPT}" \
  --acc50-checkpoint "${ACC50_CKPT}" \
  --gpu "${GPU}" \
  "$@"
