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
readonly PARENT_RUN="${OUTPUT_ROOT}/backbone/sr3d_mcln_joint_butdcls_v99_detectorpretrained_no_taskresume_global24_e1_e140_b12a2_w4p2_20260823_173119/sr3d/sr3d_mcln_joint_butdcls_v99_detectorpretrained_no_taskresume_global24_e1_e140_b12a2_w4p2/1787477483"
readonly REQUIRED_RESUME_CHECKPOINT="${PARENT_RUN}/ckpt_epoch_15.pth"
readonly REQUIRED_RESUME_SHA256="5c5921939b2476925d4d3afa2ac9eaa2cc706e28bb967c5fa1db9075b15999f4"
readonly REQUIRED_RESUME_EPOCH=15
readonly REQUIRED_E15_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_15.json"
readonly REQUIRED_E15_RECEIPT_SHA256="2b436a0c90a0443ecb4b95596035725ade29c90c9172622373e9f48ffe4a0d93"
readonly REQUIRED_MAIN_UTILS_SHA256="68760df2095b44711395cc87b9f23b258637d50b5738c389acfb2f1db09367ce"
readonly REQUIRED_TRAIN_ENTRY_SHA256="0669b5535eeb75954c76422c333556679faa149d8e2c289e60668a5377e2c7cc"
readonly REQUIRED_PIPELINE_SHA256="d6a11499ff250b1c3c8b7a338e0e09556423742bb4592f10063dcb4f81db6038"
readonly REQUIRED_MODEL_SHA256="3b9a2b88e9fa36c6f94b1595ae3812818ca5fdeb4b1db64c68f467912c04c9b1"
readonly EXP="sr3d_mcln_joint_butdcls_v99_official_schedule_resume_e15_e16_e140_b12a2_w4p2"
readonly BATCH_SIZE=12
readonly MAX_EPOCH=140
readonly EXPECTED_EVAL_SAMPLE_COUNT=17726
readonly MASTER_PORT=5599
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
require_sha "${ROOT_DIR}/train_dist_mod.py" "${REQUIRED_TRAIN_ENTRY_SHA256}" "train entry"
require_sha "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh" "${REQUIRED_PIPELINE_SHA256}" "shared pipeline"
require_sha "${ROOT_DIR}/models/mcln.py" "${REQUIRED_MODEL_SHA256}" "V99 MCLN model"
require_sha "${REQUIRED_E15_RECEIPT}" "${REQUIRED_E15_RECEIPT_SHA256}" "E15 receipt"
require_sha "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" "E15 checkpoint"

"${PYTHON_BIN}" - "${REQUIRED_E15_RECEIPT}" "${REQUIRED_RESUME_CHECKPOINT}" <<'PY'
import json
import math
import sys
from collections import Counter

import torch

receipt_path, checkpoint_path = sys.argv[1:]
with open(receipt_path) as handle:
    receipt = json.load(handle)
metric = receipt.get("position_subgroups", {}).get("multiple", {})
if receipt.get("schema") != "mcln-retrain-metrics-v1":
    raise SystemExit("E15 receipt schema mismatch")
if receipt.get("sample_count") != 17726 or metric.get("sample_count") != 17726:
    raise SystemExit("E15 receipt sample-count mismatch")
if metric.get("hits025") != 11522 or metric.get("hits050") != 9113:
    raise SystemExit("E15 receipt hit-count mismatch")
if not math.isclose(float(metric.get("acc025")), 11522.0 / 17726.0,
                    rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("E15 receipt accuracy mismatch")

checkpoint = torch.load(checkpoint_path, map_location="cpu")
config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else config
optimizer = checkpoint.get("optimizer", {})
scheduler = checkpoint.get("scheduler", {})
expected_lrs = [1e-4, 1e-3, 1e-4, 1.25e-4]
observed_lrs = [float(group["lr"]) for group in optimizer.get("param_groups", [])]
if int(checkpoint.get("epoch", -1)) != 15:
    raise SystemExit("resume checkpoint is not E15")
if len(optimizer.get("state", {})) != 716 or len(observed_lrs) != 4:
    raise SystemExit("E15 optimizer state is incomplete")
if observed_lrs != expected_lrs:
    raise SystemExit("E15 current LR is not the un-decayed official LR")
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
    "source_choice_selector_sources": "default,default_rank_blend_contrastive010",
}
for name, value in required.items():
    if config.get(name) != value:
        raise SystemExit("E15 config mismatch: {}={!r}".format(name, config.get(name)))
if float(config.get("relation_counterfactual_aux_loss_weight", 0.0)) != 0.0:
    raise SystemExit("E15 is not a pure V99 checkpoint")
if int(scheduler.get("last_epoch", -1)) != 15 * 3243:
    raise SystemExit("E15 scheduler progress mismatch")
if [float(value) for value in scheduler.get("_last_lr", [])] != expected_lrs:
    raise SystemExit("E15 scheduler LR mismatch")
if scheduler.get("milestones") != Counter({97290: 1, 129720: 1}):
    raise SystemExit("E15 scheduler milestones mismatch")
print("SR3D_OFFICIAL_SCHEDULE_RESUME_GATE=PASS epoch=15 lrs={} scheduler_steps={}".format(
    observed_lrs, scheduler["last_epoch"]))
PY

if [[ "${MODE}" == "preflight" ]]; then
  free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
  free_gb="$((free_kb / 1024 / 1024))"
  gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
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
  exec 8>&-
fi

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
