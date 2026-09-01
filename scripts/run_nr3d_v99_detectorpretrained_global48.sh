#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly SOURCE_CHECKPOINT="${DATA_ROOT%/}/gf_detector_l6o256.pth"
readonly SOURCE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly DATASET="nr3d"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/nr3d"
readonly EXP="nr3d_mcln_joint_butdcls_v99_detectorpretrained_no_taskresume_global48_e1_e240_b16a3_w4p2"
readonly REQUIRED_E62_RECEIPT="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_v99_plateau_lr10_recovery_e61_e62_b16x1_w4p2_20260823_075109/nr3d/nr3d_mcln_joint_butdcls_v99_plateau_lr10_recovery_e61_e62_b16x1_w4p2/1787442680/eval_metrics_epoch_62.json"
readonly REQUIRED_E62_RECEIPT_SHA256="7202838e826f2a21ae8226e58913ba2d4df5482af671f87c1ea4ad5cd8ddec8a"
readonly REQUIRED_E62_SCHEMA="mcln-retrain-metrics-v1"
readonly REQUIRED_E62_SAMPLE_COUNT=7899
readonly PROTECTED_E57_HITS025=4463
readonly BATCH_SIZE=16
readonly MAX_EPOCH=240
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5299
readonly MIN_FREE_GB=7
readonly BACKBONE_JOINT_TRAINING=1
readonly INFERENCE_USES_GROUND_TRUTH=1
readonly USE_BACKBONE_INITIALIZATION=1
readonly TASK_CHECKPOINT_TRANSFER=0
readonly BACKBONE_AUGMENT_DET=0
CHECKPOINT_RETENTION_METRICS=(rec_acc025)
DATASET_LR_ARGS=(
  --lr_backbone 1e-3 --lr 1e-4
  --lr_decay_epochs 150
  --warmup-epoch 0
)
BACKBONE_EXTRA_ARGS=(
  --print_freq 20
  --joint_det
  --butd_cls
  --gradient_accumulation_steps 3
  --drop_incomplete_accumulation_group
)

for variable_name in \
    BACKBONE_RESUME_CHECKPOINT BACKBONE_RESUME_SHA256 BACKBONE_RESUME_EPOCH; do
  if [[ -n "${!variable_name:-}" ]]; then
    echo "${variable_name} is forbidden for the no-task-resume global48 run" >&2
    exit 2
  fi
done
unset BACKBONE_RESUME_CHECKPOINT BACKBONE_RESUME_SHA256 BACKBONE_RESUME_EPOCH
export MODE="${MODE:-backbone}"
case "${MODE}" in
  preflight|backbone) ;;
  *)
    echo "this wrapper supports MODE=preflight or MODE=backbone only" >&2
    exit 2
    ;;
esac
echo "prerequisite_e62_receipt=${REQUIRED_E62_RECEIPT}"
echo "prerequisite_e62_receipt_sha256=${REQUIRED_E62_RECEIPT_SHA256}"
if [[ "${MODE}" == "backbone" || "${MODE}" == "preflight" ]]; then
  if [[ "${REQUIRED_E62_RECEIPT_SHA256}" == "PENDING_E62_RECEIPT_SHA256" ]]; then
    echo "E62 receipt SHA-256 has not been pinned; refusing launch" >&2
    exit 2
  fi
  [[ -f "${REQUIRED_E62_RECEIPT}" ]] || {
    echo "required E62 formal receipt is missing: ${REQUIRED_E62_RECEIPT}" >&2
    exit 2
  }
  actual_e62_sha256="$(sha256sum "${REQUIRED_E62_RECEIPT}" | awk '{print $1}')"
  if [[ "${actual_e62_sha256}" != "${REQUIRED_E62_RECEIPT_SHA256}" ]]; then
    echo "E62 formal receipt SHA-256 changed: ${actual_e62_sha256}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" - \
      "${REQUIRED_E62_RECEIPT}" \
      "${REQUIRED_E62_SCHEMA}" \
      "${REQUIRED_E62_SAMPLE_COUNT}" \
      "${PROTECTED_E57_HITS025}" <<'PY'
import json
import math
import sys

path, schema, sample_count_text, protected_hits_text = sys.argv[1:]
sample_count = int(sample_count_text)
protected_hits = int(protected_hits_text)
with open(path) as handle:
    receipt = json.load(handle)
multiple = receipt.get("position_subgroups", {}).get("multiple", {})
if receipt.get("schema") != schema:
    raise SystemExit("E62 formal receipt schema mismatch")
if receipt.get("sample_count") != sample_count:
    raise SystemExit("E62 top-level sample count mismatch")
if multiple.get("sample_count") != sample_count:
    raise SystemExit("E62 formal multiple sample count mismatch")
hits025 = multiple.get("hits025")
acc025 = multiple.get("acc025")
if not isinstance(hits025, int) or isinstance(hits025, bool):
    raise SystemExit("E62 formal hits025 is invalid")
if not isinstance(acc025, (int, float)) or isinstance(acc025, bool):
    raise SystemExit("E62 formal acc025 is invalid")
if not math.isclose(
        float(acc025), float(hits025) / float(sample_count),
        rel_tol=0.0, abs_tol=1e-12,
):
    raise SystemExit("E62 formal hits025/acc025 mismatch")
if hits025 > protected_hits:
    raise SystemExit(
        "E62 refreshed the protected best ({} > {}); refusing restart".format(
            hits025, protected_hits
        )
    )
print("E62_NO_REFRESH_GATE=PASS hits025={} protected={}".format(
    hits025, protected_hits
))
PY
fi
if [[ "${MODE}" == "preflight" ]]; then
  free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
  free_gb="$((free_kb / 1024 / 1024))"
  gpu_used="$(nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader,nounits -i 0 | tr -d ' ')"
  echo "preflight_gpu0_memory_used_mib=${gpu_used}"
  echo "preflight_free_disk_gib=${free_gb}"
  if ((gpu_used >= 500)); then
    echo "GPU0 is busy (${gpu_used} MiB)" >&2
    exit 4
  fi
  if ((free_gb < MIN_FREE_GB)); then
    echo "need at least ${MIN_FREE_GB} GiB free under DATA_ROOT" >&2
    exit 5
  fi
  lock_file="${DATA_ROOT%/}/output/network_v99/single_gpu.lock"
  mkdir -p "$(dirname "${lock_file}")"
  exec 8>"${lock_file}"
  if ! flock -n 8; then
    echo "another V99 job owns ${lock_file}" >&2
    exit 6
  fi
  flock -u 8
fi

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
