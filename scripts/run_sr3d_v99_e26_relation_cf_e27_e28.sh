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
readonly REQUIRED_RESUME_CHECKPOINT="${OUTPUT_ROOT}/backbone/sr3d_mcln_joint_butdcls_v99_official_e25_plateau_lr100_e26_e29_b12a2_w4p2_20260828_174605/sr3d/sr3d_mcln_joint_butdcls_v99_official_e25_plateau_lr100_e26_e29_b12a2_w4p2/1787910373/ckpt_epoch_26.pth"
readonly REQUIRED_RESUME_SHA256="4ac72dd3d33bb6aa13278e4e67208d98f006a9863396b3f2ab3713a9c904fd1d"
readonly REQUIRED_RESUME_EPOCH=26
readonly AUDIT_RUN_ROOT="${OUTPUT_ROOT}/audit/sr3d_mcln_joint_butdcls_v99_e26_relation_cf_audit_e27_b100_b12a2_20260831_014726"
readonly AUDIT_RECEIPT="${AUDIT_RUN_ROOT}/sr3d/sr3d_mcln_joint_butdcls_v99_e26_relation_cf_audit_e27_b100_b12a2/1788112049/train_audit_receipt_epoch_27.json"
readonly AUDIT_RECEIPT_SHA256="0d2e84ca112f8f3aa78e12b2cbfc9faba2793eb9da14299df97f440efc99f0fd"
readonly EXP="sr3d_mcln_joint_butdcls_v99_e26_relation_cf_e27_e28_b12a2_w0p5"
readonly BATCH_SIZE=12
readonly MAX_EPOCH=28
readonly EXPECTED_EVAL_SAMPLE_COUNT=17726
readonly MASTER_PORT=5627
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

readonly TRAIN_ENTRY_SHA256="0669b5535eeb75954c76422c333556679faa149d8e2c289e60668a5377e2c7cc"
readonly MAIN_UTILS_SHA256="68760df2095b44711395cc87b9f23b258637d50b5738c389acfb2f1db09367ce"
readonly LOSSES_SHA256="1a08a7febc9bd94e1d389d2ef7f987908dd796a5f25cebccd433e155bbaa5aaf"
readonly AUXILIARY_SHA256="f1b90a6f40c8da20a362f91cccccf9a6b8a160b017cdb5be579115ea3bf30be5"
readonly MODEL_SHA256="3b9a2b88e9fa36c6f94b1595ae3812818ca5fdeb4b1db64c68f467912c04c9b1"
readonly SELECTOR_SHA256="61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"
readonly DATASET_SHA256="49af3c83091970ab307d0ca2e95a68ae42a6fcf52ddf72566b9da966ce6ad7a2"
readonly PIPELINE_SHA256="d6a11499ff250b1c3c8b7a338e0e09556423742bb4592f10063dcb4f81db6038"

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
  --relation_counterfactual_aux_loss_weight 0.5
  --relation_counterfactual_aux_parent_top_k 32
  --relation_counterfactual_aux_target_tolerance 0.10
  --relation_counterfactual_aux_attribute_tolerance 0.10
  --relation_counterfactual_aux_geometry_threshold 0.08
  --relation_counterfactual_aux_correct_iou_threshold 0.25
  --relation_counterfactual_aux_pair_margin 0.05
  --relation_counterfactual_aux_max_negatives 8
  --relation_counterfactual_aux_target_confidence_floor 0.05
  --relation_counterfactual_aux_attribute_confidence_floor 0.02
  --relation_counterfactual_aux_acc025_pair_weight 2.0
)

export BACKBONE_RESUME_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT}"
export BACKBONE_RESUME_SHA256="${REQUIRED_RESUME_SHA256}"
export BACKBONE_RESUME_EPOCH="${REQUIRED_RESUME_EPOCH}"
export VALIDATE_BACKBONE_RESUME=0
export MODE="${MODE:-preflight}"
case "${MODE}" in
  preflight|backbone) ;;
  *) echo "MODE must be preflight or backbone" >&2; exit 2 ;;
esac
if (($# != 0)); then
  echo "usage: MODE=preflight|backbone $0" >&2
  exit 2
fi
cd "${ROOT_DIR}"

require_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; exit 3; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA changed: ${actual}" >&2
    exit 3
  }
}

require_sha256 "${ROOT_DIR}/train_dist_mod.py" "${TRAIN_ENTRY_SHA256}" "train entry"
require_sha256 "${ROOT_DIR}/main_utils.py" "${MAIN_UTILS_SHA256}" "main_utils"
require_sha256 "${ROOT_DIR}/models/losses.py" "${LOSSES_SHA256}" "losses"
require_sha256 "${ROOT_DIR}/models/relation_counterfactual_auxiliary.py" "${AUXILIARY_SHA256}" "relation auxiliary"
require_sha256 "${ROOT_DIR}/models/mcln.py" "${MODEL_SHA256}" "MCLN model"
require_sha256 "${ROOT_DIR}/models/source_choice_selector.py" "${SELECTOR_SHA256}" "V99 selector"
require_sha256 "${ROOT_DIR}/src/joint_det_dataset.py" "${DATASET_SHA256}" "dataset"
require_sha256 "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh" "${PIPELINE_SHA256}" "shared pipeline"
require_sha256 "${SOURCE_CHECKPOINT}" "${SOURCE_SHA256}" "GroupFree checkpoint"
require_sha256 "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" "E26 resume checkpoint"
require_sha256 "${AUDIT_RECEIPT}" "${AUDIT_RECEIPT_SHA256}" "100-batch audit receipt"

"${PYTHON_BIN}" - \
    "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" \
    "${AUDIT_RECEIPT}" "${AUDIT_RECEIPT_SHA256}" <<'PY'
import hashlib
import json
import math
import sys
from collections import Counter

import torch

checkpoint_path, checkpoint_sha, receipt_path, receipt_sha = sys.argv[1:]

with open(checkpoint_path, "rb") as handle:
    before = hashlib.sha256(handle.read()).hexdigest()
    if before != checkpoint_sha:
        raise SystemExit("E26 checkpoint SHA mismatch")
    handle.seek(0)
    checkpoint = torch.load(handle, map_location="cpu")
    handle.seek(0)
    after = hashlib.sha256(handle.read()).hexdigest()
if after != before:
    raise SystemExit("E26 checkpoint changed while loading")

with open(receipt_path, "rb") as handle:
    receipt_raw = handle.read()
if hashlib.sha256(receipt_raw).hexdigest() != receipt_sha:
    raise SystemExit("audit receipt SHA mismatch")
receipt = json.loads(receipt_raw.decode("utf-8"))
if receipt.get("schema") != "mcln-train-loss-epoch-v1":
    raise SystemExit("unexpected audit receipt schema")
if (receipt.get("epoch"), receipt.get("batch_count"),
        receipt.get("max_train_batches")) != (27, 100, 100):
    raise SystemExit("audit receipt epoch/batch mismatch")
if receipt.get("checkpoint_path") != checkpoint_path:
    raise SystemExit("audit used another checkpoint")
stats = receipt.get("stat_means", {})
gates = {
    "exact_gt_anchor_ratio": stats.get(
        "relation_counterfactual_aux_exact_gt_anchor_ratio", -1.0) >= 0.50,
    "reference_valid_ratio": stats.get(
        "relation_counterfactual_aux_relation_reference_valid_ratio", -1.0) >= 0.50,
    "hard_negative_row_ratio": stats.get(
        "relation_counterfactual_aux_hard_negative_row_ratio", -1.0) >= 0.01,
    "selected_negative_count_mean": stats.get(
        "relation_counterfactual_aux_selected_negative_count_mean", -1.0) >= 0.02,
    "pair_violation_ratio": stats.get(
        "relation_counterfactual_aux_pair_violation_ratio", -1.0) >= 0.05,
    "positive_grad_norm": stats.get("grad_norm", 0.0) > 0.0,
}
if not all(gates.values()):
    raise SystemExit("100-batch density audit no longer approves training")

config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else dict(config or {})
optimizer = checkpoint.get("optimizer", {})
scheduler = checkpoint.get("scheduler", {})
groups = optimizer.get("param_groups", [])
expected_current = [1e-6, 1e-5, 1e-6, 1.25e-6]
expected_initial = [1e-4, 1e-3, 1e-4, 1.25e-4]
if checkpoint.get("epoch") != 26:
    raise SystemExit("resume checkpoint is not completed E26")
required = {
    "dataset": ["sr3d"],
    "test_dataset": "sr3d",
    "batch_size": 12,
    "gradient_accumulation_steps": 2,
    "drop_incomplete_accumulation_group": True,
    "joint_det": True,
    "butd_cls": True,
    "use_source_choice_selector": True,
    "eval_use_selector_choice_scores": True,
    "warmup_epoch": 0,
    "lr_decay_epochs": [30, 40],
}
for key, expected in required.items():
    if config.get(key) != expected:
        raise SystemExit("E26 config mismatch: {}={!r}".format(key, config.get(key)))
if float(config.get("relation_counterfactual_aux_loss_weight", 0.0)) != 0.0:
    raise SystemExit("E26 already used relation auxiliary")
if not math.isclose(float(config.get("resume_lr_scale_lineage")), 0.01,
                    rel_tol=0.0, abs_tol=1e-15):
    raise SystemExit("E26 LR lineage mismatch")
if len(groups) != 4 or len(optimizer.get("state", {})) != 716:
    raise SystemExit("optimizer topology mismatch")
for actual, expected in zip([float(group["lr"]) for group in groups], expected_current):
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
        raise SystemExit("optimizer current LR mismatch")
if [float(group["initial_lr"]) for group in groups] != expected_initial:
    raise SystemExit("optimizer initial LR mismatch")
if scheduler.get("last_epoch") != 26 * 3243:
    raise SystemExit("scheduler progress mismatch")
if scheduler.get("milestones") != Counter({97290: 1, 129720: 1}):
    raise SystemExit("scheduler milestones mismatch")
actual_scheduler_lr = [float(value) for value in scheduler.get("_last_lr", [])]
if len(actual_scheduler_lr) != 4 or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
        for actual, expected in zip(actual_scheduler_lr, expected_current)):
    raise SystemExit("scheduler current LR mismatch")
print("controlled_contract=E26_to_E27_E28_relation_cf_verified")
print("density_gates={}".format(json.dumps(gates, sort_keys=True)))
PY

if find "${AUDIT_RUN_ROOT}" -type f -name '*.pth' -print -quit | grep -q .; then
  echo "bounded audit unexpectedly contains a checkpoint" >&2
  exit 3
fi

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

echo "formal_contract=controlled_E27_E28_only target_hits025=12214"
echo "excluded_routes=baseline_reproduction,proposal_section7,experiment_matrix_section8"

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
