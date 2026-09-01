#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly SOURCE_CHECKPOINT="${DATA_ROOT%/}/gf_detector_l6o256.pth"
readonly SOURCE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly DATASET="sr3d"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/sr3d"
readonly EXP="sr3d_mcln_joint_butdcls_v99_detectorpretrained_no_taskresume_global24_e1_e140_b12a2_w4p2"
readonly REQUIRED_E40_RECEIPT="${OUTPUT_ROOT}/backbone/sr3d_mcln_joint_butdcls_v99_relation_cf_aux_e35_e46_b14x1_w4p2_20260822_194805/sr3d/sr3d_mcln_joint_butdcls_v99_relation_cf_aux_e35_e46_b14x1_w4p2/1787399294/eval_metrics_epoch_40.json"
readonly REQUIRED_E40_RECEIPT_SHA256="7d108e64a1d56662f282409f07bfc233e36693da1a21fdd119750ef44e0a1a81"
readonly REQUIRED_E40_SCHEMA="mcln-retrain-metrics-v1"
readonly REQUIRED_E40_SAMPLE_COUNT=17726
readonly PROTECTED_E38_HITS025=12025
readonly BATCH_SIZE=12
readonly MAX_EPOCH=140
readonly EXPECTED_EVAL_SAMPLE_COUNT=17726
readonly MASTER_PORT=5399
readonly MIN_FREE_GB=7
readonly EXPECTED_MAIN_UTILS_SHA256="31acf4810857480121dbf3f4527211f452c3ea3ff835e483c9763de329cf4e40"
readonly EXPECTED_TRAIN_DATASET_SIZE=77836
readonly EXPECTED_TRAIN_LOADER_BATCH_COUNT=6486
readonly EXPECTED_EFFECTIVE_TRAIN_BATCH_COUNT=6486
readonly EXPECTED_DROPPED_TRAIN_BATCH_COUNT=0
readonly EXPECTED_OPTIMIZER_STEPS_PER_EPOCH=3243
readonly BACKBONE_JOINT_TRAINING=1
readonly INFERENCE_USES_GROUND_TRUTH=1
readonly USE_BACKBONE_INITIALIZATION=1
readonly TASK_CHECKPOINT_TRANSFER=0
readonly BACKBONE_AUGMENT_DET=0
CHECKPOINT_RETENTION_METRICS=(rec_acc025)
DATASET_LR_ARGS=(
  --lr_backbone 1e-3 --lr 1e-4
  --lr_decay_epochs 30 40
  --warmup-epoch 0
)
BACKBONE_EXTRA_ARGS=(
  --print_freq 20
  --joint_det
  --butd_cls
  --gradient_accumulation_steps 2
  --drop_incomplete_accumulation_group
  --expected_train_dataset_size "${EXPECTED_TRAIN_DATASET_SIZE}"
  --expected_train_loader_batch_count "${EXPECTED_TRAIN_LOADER_BATCH_COUNT}"
  --expected_effective_train_batch_count "${EXPECTED_EFFECTIVE_TRAIN_BATCH_COUNT}"
  --expected_dropped_train_batch_count "${EXPECTED_DROPPED_TRAIN_BATCH_COUNT}"
  --expected_optimizer_steps_per_epoch "${EXPECTED_OPTIMIZER_STEPS_PER_EPOCH}"
)

for variable_name in \
    BACKBONE_RESUME_CHECKPOINT BACKBONE_RESUME_SHA256 BACKBONE_RESUME_EPOCH; do
  if [[ -n "${!variable_name:-}" ]]; then
    echo "${variable_name} is forbidden for the no-task-resume global24 run" >&2
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
actual_main_utils_sha256="$(sha256sum "${ROOT_DIR}/main_utils.py" | awk '{print $1}')"
if [[ "${actual_main_utils_sha256}" != "${EXPECTED_MAIN_UTILS_SHA256}" ]]; then
  echo "main_utils.py SHA-256 changed: ${actual_main_utils_sha256}" >&2
  exit 2
fi
"${PYTHON_BIN}" - "${ROOT_DIR}" <<'PY'
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, sys.argv[1])
import main_utils

plan = main_utils._gradient_accumulation_plan(
    loader_batch_count=6486,
    max_train_batches=0,
    accumulation_steps=2,
    drop_incomplete_accumulation_group=True,
)
expected = {
    "requested_batch_count": 6486,
    "effective_batch_count": 6486,
    "dropped_batch_count": 0,
    "optimizer_step_count": 3243,
}
if plan != expected:
    raise SystemExit("Sr3D global24 accumulation plan mismatch: {}".format(plan))
realized = main_utils._validate_expected_training_plan(
    dataset_size=77836,
    loader_batch_count=6486,
    accumulation_plan=plan,
    expected_dataset_size=77836,
    expected_loader_batch_count=6486,
    expected_effective_batch_count=6486,
    expected_dropped_batch_count=0,
    expected_optimizer_step_count=3243,
)
parameter = torch.nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([parameter], lr=1.0)
args = SimpleNamespace(
    lr_scheduler="step", lr_decay_epochs=[30, 40], lr_decay_rate=0.1,
    warmup_epoch=0, warmup_multiplier=100, max_epoch=140,
)
scheduler = main_utils.get_scheduler(optimizer, 3243, args)
milestones = sorted(scheduler.milestones.elements())
if milestones != [97290, 129720]:
    raise SystemExit("Sr3D global24 scheduler milestones mismatch: {}".format(
        milestones
    ))
print("SR3D_GLOBAL24_CORE_GATE=PASS plan={} realized={} milestones={}".format(
    plan, realized, milestones
))
PY
echo "prerequisite_e40_receipt=${REQUIRED_E40_RECEIPT}"
echo "prerequisite_e40_receipt_sha256=${REQUIRED_E40_RECEIPT_SHA256}"
if [[ "${MODE}" == "backbone" || "${MODE}" == "preflight" ]]; then
  if [[ "${REQUIRED_E40_RECEIPT_SHA256}" == "PENDING_E40_RECEIPT_SHA256" ]]; then
    echo "E40 receipt SHA-256 has not been pinned; refusing launch" >&2
    exit 2
  fi
  [[ -f "${REQUIRED_E40_RECEIPT}" ]] || {
    echo "required E40 formal receipt is missing: ${REQUIRED_E40_RECEIPT}" >&2
    exit 2
  }
  actual_e40_sha256="$(sha256sum "${REQUIRED_E40_RECEIPT}" | awk '{print $1}')"
  if [[ "${actual_e40_sha256}" != "${REQUIRED_E40_RECEIPT_SHA256}" ]]; then
    echo "E40 formal receipt SHA-256 changed: ${actual_e40_sha256}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" - \
      "${REQUIRED_E40_RECEIPT}" \
      "${REQUIRED_E40_SCHEMA}" \
      "${REQUIRED_E40_SAMPLE_COUNT}" \
      "${PROTECTED_E38_HITS025}" <<'PY'
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
    raise SystemExit("E40 formal receipt schema mismatch")
if receipt.get("sample_count") != sample_count:
    raise SystemExit("E40 top-level sample count mismatch")
if multiple.get("sample_count") != sample_count:
    raise SystemExit("E40 formal multiple sample count mismatch")
hits025 = multiple.get("hits025")
acc025 = multiple.get("acc025")
if not isinstance(hits025, int) or isinstance(hits025, bool):
    raise SystemExit("E40 formal hits025 is invalid")
if not isinstance(acc025, (int, float)) or isinstance(acc025, bool):
    raise SystemExit("E40 formal acc025 is invalid")
if not math.isclose(
        float(acc025), float(hits025) / float(sample_count),
        rel_tol=0.0, abs_tol=1e-12,
):
    raise SystemExit("E40 formal hits025/acc025 mismatch")
if hits025 > protected_hits:
    raise SystemExit(
        "E40 refreshed the protected best ({} > {}); refusing restart".format(
            hits025, protected_hits
        )
    )
print("E40_NO_REFRESH_GATE=PASS hits025={} protected={}".format(
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
