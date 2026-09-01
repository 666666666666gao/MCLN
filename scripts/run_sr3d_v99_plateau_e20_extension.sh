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
readonly PARENT_RUN="${OUTPUT_ROOT}/backbone/sr3d_mcln_joint_butdcls_v99_plateau_lr10_e16_e20_b12a2_w4p2_20260825_202643/sr3d/sr3d_mcln_joint_butdcls_v99_plateau_lr10_e16_e20_b12a2_w4p2/1787660811"
readonly REQUIRED_RESUME_CHECKPOINT="${PARENT_RUN}/ckpt_epoch_20.pth"
readonly REQUIRED_RESUME_SHA256="be9d1f43d40c3b84576eeb907279b37d33c76695ba452e019f19a918cdb4c870"
readonly REQUIRED_RESUME_EPOCH=20
readonly REQUIRED_E20_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_20.json"
readonly REQUIRED_E20_RECEIPT_SHA256="a863b2ab9bff4d6c16fe211979b1689daae8792c3edef94598880c364b91b263"
readonly REQUIRED_E19_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_19.json"
readonly REQUIRED_E19_RECEIPT_SHA256="ca4b2b2c0edaf04ca701a94c968afe41dfaada19e1c2082df3a4625fcb0cc619"
readonly REQUIRED_MAIN_UTILS_SHA256="6592fa938680240cd75dd181cc1e63cc0624714d78c7d92bbba8ae6f8622a850"
readonly REQUIRED_TEST_SHA256="4645376bcb66c924d57df0681a1c1d98849d58492000b381af4f79ebbdf27ad5"
readonly REQUIRED_PIPELINE_SHA256="d6a11499ff250b1c3c8b7a338e0e09556423742bb4592f10063dcb4f81db6038"
readonly EXP="sr3d_mcln_joint_butdcls_v99_plateau_e20_extension_e21_e28_b12a2_w4p2"
readonly BATCH_SIZE=12
readonly MAX_EPOCH=28
readonly EXPECTED_EVAL_SAMPLE_COUNT=17726
readonly MASTER_PORT=5499
readonly MIN_FREE_GB=7
readonly EXPECTED_TRAIN_DATASET_SIZE=77836
readonly EXPECTED_TRAIN_LOADER_BATCH_COUNT=6486
readonly EXPECTED_EFFECTIVE_TRAIN_BATCH_COUNT=6486
readonly EXPECTED_DROPPED_TRAIN_BATCH_COUNT=0
readonly EXPECTED_OPTIMIZER_STEPS_PER_EPOCH=3243
readonly BACKBONE_JOINT_TRAINING=1
readonly INFERENCE_USES_GROUND_TRUTH=1
readonly USE_BACKBONE_INITIALIZATION=0
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
  --resume_lr_scale 1.0
)

export BACKBONE_RESUME_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT}"
export BACKBONE_RESUME_SHA256="${REQUIRED_RESUME_SHA256}"
export BACKBONE_RESUME_EPOCH="${REQUIRED_RESUME_EPOCH}"
export VALIDATE_BACKBONE_RESUME=0
export MODE="${MODE:-backbone}"
case "${MODE}" in
  preflight|backbone) ;;
  *) echo "MODE must be preflight or backbone" >&2; exit 2 ;;
esac

if [[ "${REQUIRED_RESUME_SHA256}" == PENDING_* ||
      "${REQUIRED_E20_RECEIPT_SHA256}" == PENDING_* ]]; then
  echo "E20 checkpoint/receipt SHA-256 is not pinned; refusing launch" >&2
  exit 2
fi

require_sha() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; exit 3; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA-256 changed: ${actual}" >&2
    exit 3
  }
}

require_sha "${ROOT_DIR}/main_utils.py" "${REQUIRED_MAIN_UTILS_SHA256}" "main_utils.py"
require_sha "${ROOT_DIR}/tests/test_main_utils_source_choice_checkpoint.py" "${REQUIRED_TEST_SHA256}" "LR-scale tests"
require_sha "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh" "${REQUIRED_PIPELINE_SHA256}" "shared pipeline"
require_sha "${REQUIRED_E19_RECEIPT}" "${REQUIRED_E19_RECEIPT_SHA256}" "E19 receipt"
require_sha "${REQUIRED_E20_RECEIPT}" "${REQUIRED_E20_RECEIPT_SHA256}" "E20 receipt"
require_sha "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" "E20 checkpoint"

"${PYTHON_BIN}" - "${REQUIRED_E19_RECEIPT}" "${REQUIRED_E20_RECEIPT}" "${REQUIRED_RESUME_CHECKPOINT}" <<'PY'
import json
import math
import sys
from collections import Counter
import torch

e19_path, e20_path, checkpoint_path = sys.argv[1:]
for epoch, path in ((19, e19_path), (20, e20_path)):
    with open(path) as handle:
        payload = json.load(handle)
    metric = payload.get("position_subgroups", {}).get("multiple", {})
    if payload.get("schema") != "mcln-retrain-metrics-v1":
        raise SystemExit("E{} receipt schema mismatch".format(epoch))
    if payload.get("sample_count") != 17726 or metric.get("sample_count") != 17726:
        raise SystemExit("E{} sample-count mismatch".format(epoch))
    hits = int(metric.get("hits025"))
    if not math.isclose(float(metric.get("acc025")), hits / 17726.0,
                        rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("E{} hits/accuracy mismatch".format(epoch))
    if epoch == 19 and hits != 11976:
        raise SystemExit("E19 hits mismatch")
    if epoch == 20 and hits >= 12125:
        raise SystemExit("Sr3D strict target is already achieved; refusing extension")

checkpoint = torch.load(checkpoint_path, map_location="cpu")
config = vars(checkpoint["config"])
optimizer = checkpoint["optimizer"]
scheduler = checkpoint["scheduler"]
current_lrs = [float(group["lr"]) for group in optimizer["param_groups"]]
expected_current_lrs = [1e-5, 1e-4, 1e-5, 1.25e-5]
if int(checkpoint["epoch"]) != 20:
    raise SystemExit("checkpoint epoch mismatch")
if len(optimizer["state"]) != 716 or len(current_lrs) != 4:
    raise SystemExit("optimizer state is incomplete")
if current_lrs != expected_current_lrs:
    raise SystemExit("E20 current LR mismatch")
required = {
    "dataset": ["sr3d"],
    "test_dataset": "sr3d",
    "batch_size": 12,
    "gradient_accumulation_steps": 2,
    "drop_incomplete_accumulation_group": True,
    "lr_scheduler": "step",
    "warmup_epoch": 0,
    "joint_det": True,
    "butd_cls": True,
    "use_source_choice_selector": True,
    "resume_lr_scale_lineage": 0.1,
}
for name, value in required.items():
    if config.get(name) != value:
        raise SystemExit("checkpoint config mismatch: {}={!r}".format(name, config.get(name)))
if int(scheduler["last_epoch"]) != 20 * 3243:
    raise SystemExit("scheduler progress mismatch")
if [float(value) for value in scheduler["_last_lr"]] != expected_current_lrs:
    raise SystemExit("scheduler current LR mismatch")
if scheduler["milestones"] != Counter({97290: 1, 129720: 1}):
    raise SystemExit("scheduler milestones mismatch")
print("SR3D_E20_EXTENSION_PREFLIGHT=PASS epoch=20 current_lrs={} scheduler_steps={}".format(
    current_lrs, scheduler["last_epoch"]))
PY

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
