#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
QMASK_EXP="${QMASK_EXP:-qmask_full80_b64x4_lr1e4_delta010_d0_fastresume_w4p1}"
QMASK_RUN_DIR="${QMASK_RUN_DIR:-${DATA_ROOT%/}/output/query_mask_fusion/scanrefer/${QMASK_EXP}/1785879774}"
QMASK_FINAL_EPOCH="${QMASK_FINAL_EPOCH:-80}"
QMASK_EXPECTED_SAMPLE_COUNT="${QMASK_EXPECTED_SAMPLE_COUNT:-9508}"
PROTECTED_V19="${PROTECTED_V19:-${DATA_ROOT%/}/protected_mcln_artifacts/scanrefer_network_best_v19_rec025_0.581195_mask025_0.598233.pth}"
PROTECTED_V19_SHA256="${PROTECTED_V19_SHA256:-2d6a3cf2914e5a7394ff2072378613314aae6d44c0dfa03762dcbb6e55ececbe}"
PANEL_LOG_DIR="${PANEL_LOG_DIR:-${ROOT_DIR}/experiment_output/joint_query_quality}"
PANEL_MIN_FREE_KIB="${PANEL_MIN_FREE_KIB:-8388608}"
PANEL_MAX_EPOCH="${PANEL_MAX_EPOCH:-3}"
PANEL_BATCH_SIZE="${PANEL_BATCH_SIZE:-64}"
PANEL_EXPECTED_SAMPLE_COUNT="${PANEL_EXPECTED_SAMPLE_COUNT:-128}"
RUN_FORMAL_AFTER_SMOKE="${RUN_FORMAL_AFTER_SMOKE:-1}"
FORMAL_EXP="${FORMAL_EXP:-v46_gate_evidence_lovasz_candidate_mask_full80_b64x4_lr3e4_mw025_cmw025_clw010_k16}"
FORMAL_MAX_EPOCH="${FORMAL_MAX_EPOCH:-80}"
FORMAL_BATCH_SIZE="${FORMAL_BATCH_SIZE:-64}"
FORMAL_EXPECTED_SAMPLE_COUNT="${FORMAL_EXPECTED_SAMPLE_COUNT:-9508}"
FORMAL_EXPECTED_STEPS_PER_EPOCH="${FORMAL_EXPECTED_STEPS_PER_EPOCH:-143}"
FORMAL_MASTER_PORT="${FORMAL_MASTER_PORT:-4584}"
FORMAL_LOG="${FORMAL_LOG:-${PANEL_LOG_DIR}/${FORMAL_EXP}_launcher.log}"
GPU_IDLE_WAIT_SECONDS="${GPU_IDLE_WAIT_SECONDS:-60}"
QUEUE_LOG="${QUEUE_LOG:-${DATA_ROOT%/}/output/joint_query_quality/v46_after_qmask_queue.log}"
LOCK_FILE="${LOCK_FILE:-${DATA_ROOT%/}/output/joint_query_quality/v43_after_qmask_queue.lock}"

mkdir -p "$(dirname "${QUEUE_LOG}")"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another V46 queue supervisor already owns ${LOCK_FILE}" >&2
  exit 3
fi
exec > >(tee -a "${QUEUE_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

for value in "${QMASK_FINAL_EPOCH}" "${QMASK_EXPECTED_SAMPLE_COUNT}" \
             "${PANEL_MAX_EPOCH}" "${PANEL_BATCH_SIZE}" \
             "${PANEL_EXPECTED_SAMPLE_COUNT}" "${FORMAL_MAX_EPOCH}" \
             "${FORMAL_BATCH_SIZE}" "${FORMAL_EXPECTED_SAMPLE_COUNT}" \
             "${FORMAL_EXPECTED_STEPS_PER_EPOCH}" "${FORMAL_MASTER_PORT}" \
             "${GPU_IDLE_WAIT_SECONDS}" "${PANEL_MIN_FREE_KIB}"; do
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

echo "[$(timestamp)] waiting for qmask experiment ${QMASK_EXP}"
qmask_pid="$(pgrep -o -f "train_dist_mod.py.*--exp ${QMASK_EXP}" || true)"
if [[ -n "${qmask_pid}" ]]; then
  echo "[$(timestamp)] bound to qmask process ${qmask_pid}; no log polling"
  tail --pid="${qmask_pid}" -f /dev/null
fi
echo "[$(timestamp)] qmask process exited; validating final artifacts"

qmask_receipt="${QMASK_RUN_DIR}/eval_metrics_epoch_${QMASK_FINAL_EPOCH}.json"
qmask_checkpoint="${QMASK_RUN_DIR}/ckpt_epoch_last.pth"
"${PYTHON_BIN}" scripts/audit_training_completion.py \
  --metrics "${qmask_receipt}" \
  --checkpoint "${qmask_checkpoint}" \
  --expected-epoch "${QMASK_FINAL_EPOCH}" \
  --expected-sample-count "${QMASK_EXPECTED_SAMPLE_COUNT}" \
  --output "${QMASK_RUN_DIR}/audit_completion_epoch_${QMASK_FINAL_EPOCH}.json"

actual_sha256="$(sha256sum "${PROTECTED_V19}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${PROTECTED_V19_SHA256}" ]]; then
  echo "protected V19 SHA-256 changed" >&2
  exit 4
fi

mkdir -p "${PANEL_LOG_DIR}"
panel_free_kib="$(df -Pk "${PANEL_LOG_DIR}" | awk 'NR == 2 {print $4}')"
if [[ ! "${panel_free_kib}" =~ ^[1-9][0-9]*$ \
      || "${panel_free_kib}" -lt "${PANEL_MIN_FREE_KIB}" ]]; then
  echo "V46 output filesystem has insufficient free space: ${panel_free_kib:-unknown} KiB" >&2
  exit 4
fi

for profile in v43 v46; do
  initialization_audit="${PANEL_LOG_DIR}/${profile}_protected_v19_initialization_audit.json"
  "${PYTHON_BIN}" scripts/audit_joint_query_initialization.py \
    --checkpoint "${PROTECTED_V19}" \
    --profile "${profile}" \
    --output "${initialization_audit}"
  chmod 0444 "${initialization_audit}"
done

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
  "v46panel_smoke_v45_gate0_clw000"
  "v46panel_smoke_v45_gate0_clw010"
  "v46panel_smoke_v46_gate1_clw000"
  "v46panel_smoke_v46_gate1_clw010"
)
lrs=("3e-4" "3e-4" "3e-4" "3e-4")
dropouts=("0.1" "0.1" "0.1" "0.1")
mask_weights=("0.25" "0.25" "0.25" "0.25")
temperatures=("0.25" "0.25" "0.25" "0.25")
anchor_weights=("0.5" "0.5" "0.5" "0.5")
candidate_mask_weights=("0.25" "0.25" "0.25" "0.25")
candidate_lovasz_weights=("0.00" "0.10" "0.00" "0.10")
gate_evidence=("0" "0" "1" "1")
candidate_mask_top_ks=("16" "16" "16" "16")
ports=("4580" "4581" "4582" "4583")

job_pids=()
job_logs=()
for gpu in 0 1 2 3; do
  variant="${variants[${gpu}]}"
  job_log="${PANEL_LOG_DIR}/${variant}_launcher.log"
  job_logs+=("${job_log}")
  echo "[$(timestamp)] starting ${variant} on GPU ${gpu}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    NPROC_PER_NODE=1 \
    MASTER_PORT="${ports[${gpu}]}" \
    CHECKPOINT_PATH="${PROTECTED_V19}" \
    LOG_DIR="${PANEL_LOG_DIR}" \
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
    JOINT_QUERY_QUALITY_LR="${lrs[${gpu}]}" \
    JOINT_QUERY_QUALITY_DROPOUT="${dropouts[${gpu}]}" \
    JOINT_QUERY_QUALITY_MASK_WEIGHT="${mask_weights[${gpu}]}" \
    JOINT_QUERY_QUALITY_TEMPERATURE="${temperatures[${gpu}]}" \
    JOINT_QUERY_QUALITY_ANCHOR_LOSS_WEIGHT="${anchor_weights[${gpu}]}" \
    JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION=1 \
    JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE=1 \
    JOINT_QUERY_QUALITY_USE_GATE_EVIDENCE="${gate_evidence[${gpu}]}" \
    JOINT_QUERY_QUALITY_MAX_MASK_ALPHA_DELTA=1.0 \
    JOINT_QUERY_QUALITY_MAX_MASK_LOGIT_BIAS=2.0 \
    JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT="${candidate_mask_weights[${gpu}]}" \
    JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT="${candidate_lovasz_weights[${gpu}]}" \
    JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K="${candidate_mask_top_ks[${gpu}]}" \
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
audit_profiles=("v43" "v43" "v46" "v46")
for index in 0 1 2 3; do
  variant="${variants[${index}]}"
  variant_root="${PANEL_LOG_DIR}/scanrefer/${variant}"
  run_dir="$(
    find "${variant_root}" -mindepth 1 -maxdepth 1 -type d \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "${run_dir}" ]]; then
    echo "missing run directory for ${variant}" >&2
    exit 6
  fi
  run_dirs+=("${run_dir}")
  "${PYTHON_BIN}" scripts/audit_source_moe_checkpoint.py \
    --profile "${audit_profiles[${index}]}" \
    --baseline "${PROTECTED_V19}" \
    --checkpoint "${run_dir}/ckpt_epoch_last.pth" \
    --expected-epoch "${PANEL_MAX_EPOCH}" \
    --expected-step "${expected_step}" \
    --output "${run_dir}/audit_${audit_profiles[${index}]}.json"
done

control_summary_path="${PANEL_LOG_DIR}/v46_panel_v45_controls_summary.json"
"${PYTHON_BIN}" scripts/summarize_v41_smoke_panel.py \
  --output "${control_summary_path}" \
  --profile v43 \
  --require-candidate-mask \
  --require-lovasz-variant "${variants[1]}" \
  --epoch "${PANEL_MAX_EPOCH}" \
  --expected-sample-count "${PANEL_EXPECTED_SAMPLE_COUNT}" \
  --record "${variants[0]}" "${run_dirs[0]}" "${job_logs[0]}" \
  --record "${variants[1]}" "${run_dirs[1]}" "${job_logs[1]}"

gate_summary_path="${PANEL_LOG_DIR}/v46_panel_v46_gate_summary.json"
"${PYTHON_BIN}" scripts/summarize_v41_smoke_panel.py \
  --output "${gate_summary_path}" \
  --profile v46 \
  --require-candidate-mask \
  --require-lovasz-variant "${variants[3]}" \
  --epoch "${PANEL_MAX_EPOCH}" \
  --expected-sample-count "${PANEL_EXPECTED_SAMPLE_COUNT}" \
  --record "${variants[2]}" "${run_dirs[2]}" "${job_logs[2]}" \
  --record "${variants[3]}" "${run_dirs[3]}" "${job_logs[3]}"

echo "[$(timestamp)] V46 2x2 smoke panel passed: controls=${control_summary_path} gate=${gate_summary_path}"

for run_dir in "${run_dirs[@]}"; do
  echo "[$(timestamp)] removing audited debug-smoke checkpoints from ${run_dir}"
  find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.pth' \
    -print -delete
done

if [[ "${RUN_FORMAL_AFTER_SMOKE}" == "0" ]]; then
  echo "[$(timestamp)] formal V46 launch disabled by configuration"
  exit 0
fi

actual_sha256="$(sha256sum "${PROTECTED_V19}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${PROTECTED_V19_SHA256}" ]]; then
  echo "protected V19 SHA-256 changed before formal V46 launch" >&2
  exit 7
fi

echo "[$(timestamp)] starting pre-registered four-GPU formal V46: ${FORMAL_EXP}"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT="${FORMAL_MASTER_PORT}" \
CHECKPOINT_PATH="${PROTECTED_V19}" \
LOG_DIR="${PANEL_LOG_DIR}" \
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
JOINT_QUERY_QUALITY_SCORE_WEIGHT=1.0 \
JOINT_QUERY_QUALITY_TEMPERATURE=0.25 \
JOINT_QUERY_QUALITY_ANCHOR_LOSS_WEIGHT=0.5 \
JOINT_QUERY_QUALITY_USE_MASK_CALIBRATION=1 \
JOINT_QUERY_QUALITY_USE_SOURCE_MASK_EVIDENCE=1 \
JOINT_QUERY_QUALITY_USE_GATE_EVIDENCE=1 \
JOINT_QUERY_QUALITY_MAX_MASK_ALPHA_DELTA=1.0 \
JOINT_QUERY_QUALITY_MAX_MASK_LOGIT_BIAS=2.0 \
JOINT_QUERY_QUALITY_CANDIDATE_MASK_LOSS_WEIGHT=0.25 \
JOINT_QUERY_QUALITY_CANDIDATE_LOVASZ_LOSS_WEIGHT=0.10 \
JOINT_QUERY_QUALITY_CANDIDATE_MASK_TOP_K=16 \
bash scripts/train_scanrefer_joint_query_quality.sh \
  >"${FORMAL_LOG}" 2>&1

formal_root="${PANEL_LOG_DIR}/scanrefer/${FORMAL_EXP}"
formal_run_dir="$(
  find "${formal_root}" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
)"
if [[ -z "${formal_run_dir}" ]]; then
  echo "missing formal V46 run directory" >&2
  exit 8
fi
formal_step=$((FORMAL_EXPECTED_STEPS_PER_EPOCH * FORMAL_MAX_EPOCH))
"${PYTHON_BIN}" scripts/audit_training_completion.py \
  --metrics "${formal_run_dir}/eval_metrics_epoch_${FORMAL_MAX_EPOCH}.json" \
  --checkpoint "${formal_run_dir}/ckpt_epoch_last.pth" \
  --expected-epoch "${FORMAL_MAX_EPOCH}" \
  --expected-sample-count "${FORMAL_EXPECTED_SAMPLE_COUNT}" \
  --output "${formal_run_dir}/audit_completion_epoch_${FORMAL_MAX_EPOCH}.json"
"${PYTHON_BIN}" scripts/audit_source_moe_checkpoint.py \
  --profile v46 \
  --baseline "${PROTECTED_V19}" \
  --checkpoint "${formal_run_dir}/ckpt_epoch_last.pth" \
  --expected-epoch "${FORMAL_MAX_EPOCH}" \
  --expected-step "${formal_step}" \
  --output "${formal_run_dir}/audit_v46_final.json"
echo "[$(timestamp)] formal V46 completed and audited: ${formal_run_dir}"
