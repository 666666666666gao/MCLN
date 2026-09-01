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
readonly BASE_RUN="${OUTPUT_ROOT}/backbone/sr3d_mcln_joint_butdcls_v99_official_schedule_resume_e15_e16_e140_b12a2_w4p2_20260826_230842/sr3d/sr3d_mcln_joint_butdcls_v99_official_schedule_resume_e15_e16_e140_b12a2_w4p2/1787756930"
readonly REQUIRED_RESUME_CHECKPOINT="${BASE_RUN}/ckpt_epoch_21.pth"
readonly REQUIRED_RESUME_SHA256="d96f19e47b41920ec0a60b5107c4cc7e7f26fa9bbde8fd59dade7dee789cdcef"
readonly REQUIRED_RESUME_EPOCH=21
readonly REQUIRED_E19_RECEIPT="${BASE_RUN}/eval_metrics_epoch_19.json"
readonly REQUIRED_E19_RECEIPT_SHA256="f6636ec8d24182aec4fb3ae9656a0f8dcd35d6b4106f2e4aa824446ac16b5d94"
readonly REQUIRED_E20_RECEIPT="${BASE_RUN}/eval_metrics_epoch_20.json"
readonly REQUIRED_E20_RECEIPT_SHA256="4e95b5f582d97afee90506aacefde5e57b8aa56c2e51038099fb5f9a42c34132"
readonly REQUIRED_E21_RECEIPT="${BASE_RUN}/eval_metrics_epoch_21.json"
readonly REQUIRED_E21_RECEIPT_SHA256="badc996080cd79b17273e502f228f3fdf4dac5fe7601b5b76e337ac8da203564"
readonly REQUIRED_MAIN_UTILS_SHA256="68760df2095b44711395cc87b9f23b258637d50b5738c389acfb2f1db09367ce"
readonly REQUIRED_TEST_SHA256="73f41dccf7151198dad68c76d7aa82a36e74d6c01d9882ebc5f615af45d1f853"
readonly REQUIRED_PIPELINE_SHA256="d6a11499ff250b1c3c8b7a338e0e09556423742bb4592f10063dcb4f81db6038"
readonly EXP="sr3d_mcln_joint_butdcls_v99_official_e21_plateau_lr10_e22_e25_b12a2_w4p2"
readonly BATCH_SIZE=12
readonly MAX_EPOCH=25
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
  --resume_lr_scale 0.1
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
require_sha "${REQUIRED_E21_RECEIPT}" "${REQUIRED_E21_RECEIPT_SHA256}" "E21 receipt"
require_sha "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" "E21 checkpoint"

"${PYTHON_BIN}" - \
  "${ROOT_DIR}" \
  "${REQUIRED_E19_RECEIPT}" \
  "${REQUIRED_E20_RECEIPT}" \
  "${REQUIRED_E21_RECEIPT}" \
  "${REQUIRED_RESUME_CHECKPOINT}" <<'PY'
import json
import math
import sys
from collections import Counter

import torch

root, e19_path, e20_path, e21_path, checkpoint_path = sys.argv[1:]
sys.path.insert(0, root)
import main_utils

expected_hits = {19: 11903, 20: 11863, 21: 11730}
observed_hits = {}
for epoch, path in ((19, e19_path), (20, e20_path), (21, e21_path)):
    with open(path) as handle:
        payload = json.load(handle)
    metric = payload.get("position_subgroups", {}).get("multiple", {})
    if payload.get("schema") != "mcln-retrain-metrics-v1":
        raise SystemExit("E{} receipt schema mismatch".format(epoch))
    if payload.get("sample_count") != 17726 or metric.get("sample_count") != 17726:
        raise SystemExit("E{} sample-count mismatch".format(epoch))
    hits = int(metric.get("hits025"))
    if hits != expected_hits[epoch]:
        raise SystemExit("E{} hits mismatch".format(epoch))
    if not math.isclose(float(metric.get("acc025")), hits / 17726.0,
                        rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("E{} hits/accuracy mismatch".format(epoch))
    observed_hits[epoch] = hits

if not (observed_hits[20] <= observed_hits[19] and
        observed_hits[21] <= observed_hits[19]):
    raise SystemExit("E20/E21 do not prove two consecutive non-refreshes")
if max(observed_hits.values()) >= 12125:
    raise SystemExit("Sr3D strict target is already achieved; refusing decay")

checkpoint = torch.load(checkpoint_path, map_location="cpu")
config = vars(checkpoint["config"])
optimizer = checkpoint["optimizer"]
scheduler = checkpoint["scheduler"]
current_lrs = [float(group["lr"]) for group in optimizer["param_groups"]]
expected_current_lrs = [1e-4, 1e-3, 1e-4, 1.25e-4]
if int(checkpoint["epoch"]) != 21:
    raise SystemExit("checkpoint epoch mismatch")
if len(optimizer["state"]) != 716 or len(current_lrs) != 4:
    raise SystemExit("optimizer state is incomplete")
if current_lrs != expected_current_lrs:
    raise SystemExit("E21 current LR mismatch")
if [float(group["initial_lr"]) for group in optimizer["param_groups"]] != expected_current_lrs:
    raise SystemExit("E21 initial LR mismatch")
required = {
    "dataset": ["sr3d"],
    "test_dataset": "sr3d",
    "batch_size": 12,
    "gradient_accumulation_steps": 2,
    "drop_incomplete_accumulation_group": True,
    "lr_scheduler": "step",
    "warmup_epoch": 0,
    "lr_decay_epochs": [30, 40],
    "joint_det": True,
    "butd_cls": True,
    "use_source_choice_selector": True,
    "eval_use_selector_choice_scores": True,
    "resume_lr_scale": 1.0,
    "resume_lr_scale_lineage": 1.0,
}
for name, value in required.items():
    if config.get(name) != value:
        raise SystemExit(
            "checkpoint config mismatch: {}={!r}".format(name, config.get(name))
        )
if int(scheduler["last_epoch"]) != 21 * 3243:
    raise SystemExit("scheduler progress mismatch")
if int(scheduler["_step_count"]) != 21 * 3243 + 1:
    raise SystemExit("scheduler step-count mismatch")
if [float(value) for value in scheduler["_last_lr"]] != expected_current_lrs:
    raise SystemExit("scheduler current LR mismatch")
if [float(value) for value in scheduler["base_lrs"]] != expected_current_lrs:
    raise SystemExit("scheduler base LR mismatch")
if scheduler["milestones"] != Counter({97290: 1, 129720: 1}):
    raise SystemExit("scheduler milestones mismatch")

plan = main_utils._gradient_accumulation_plan(
    loader_batch_count=6486,
    max_train_batches=0,
    accumulation_steps=2,
    drop_incomplete_accumulation_group=True,
)
expected_plan = {
    "requested_batch_count": 6486,
    "effective_batch_count": 6486,
    "dropped_batch_count": 0,
    "optimizer_step_count": 3243,
}
if plan != expected_plan:
    raise SystemExit("global24 accumulation plan mismatch")
target_lrs = [1e-5, 1e-4, 1e-5, 1.25e-5]
if [value * 0.1 for value in expected_current_lrs] != target_lrs:
    raise SystemExit("target LR proof mismatch")
print(
    "SR3D_OFFICIAL_E21_PLATEAU_LR10_PREFLIGHT=PASS epoch=21 "
    "plateau_hits={} current_lrs={} target_lrs={} scheduler_steps={}".format(
        observed_hits, current_lrs, target_lrs, scheduler["last_epoch"]
    )
)
PY

free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
gpu_used="$(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 |
    tr -d ' '
)"
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
exec 8>&-
echo "SR3D_OFFICIAL_E21_PLATEAU_LR10_RESOURCE_PREFLIGHT=PASS gpu_mib=${gpu_used} free_gib=${free_gb}"

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
