#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly DATASET="sr3d"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/sr3d"
readonly GROUPFREE_CHECKPOINT="${DATA_ROOT%/}/gf_detector_l6o256.pth"
readonly GROUPFREE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly V97_SOURCE="${ROOT_DIR}/experiment_output/historical_e71_geometry/v97_contextual_listwise_hierarchical_trainonly_v1.json"
readonly V97_SOURCE_SHA256="ca04b4cbd1804b92d676d815b79bfcacdaab3e8745742177bd94283cedda7f8d"
readonly RESUME_CHECKPOINT="${OUTPUT_ROOT}/backbone/sr3d_mcln_joint_butdcls_v99_official_e25_plateau_lr100_e26_e29_b12a2_w4p2_20260828_174605/sr3d/sr3d_mcln_joint_butdcls_v99_official_e25_plateau_lr100_e26_e29_b12a2_w4p2/1787910373/ckpt_epoch_26.pth"
readonly RESUME_SHA256="4ac72dd3d33bb6aa13278e4e67208d98f006a9863396b3f2ab3713a9c904fd1d"
readonly EXP="sr3d_mcln_joint_butdcls_v99_e26_relation_cf_audit_e27_b100_b12a2"
readonly AUDIT_EPOCH=27
readonly AUDIT_BATCHES=100
readonly BATCH_SIZE=12
readonly ACCUMULATION=2
readonly MASTER_PORT=5599
readonly EXPECTED_EVAL_SAMPLE_COUNT=17726
readonly EXPECTED_TRAIN_DATASET_SIZE=77836
readonly EXPECTED_TRAIN_LOADER_BATCH_COUNT=6486
readonly EXPECTED_EFFECTIVE_TRAIN_BATCH_COUNT=100
readonly EXPECTED_DROPPED_TRAIN_BATCH_COUNT=0
readonly EXPECTED_OPTIMIZER_STEPS_PER_EPOCH=50
readonly MIN_FREE_GB=7

readonly TRAIN_ENTRY_SHA256="0669b5535eeb75954c76422c333556679faa149d8e2c289e60668a5377e2c7cc"
readonly MAIN_UTILS_SHA256="68760df2095b44711395cc87b9f23b258637d50b5738c389acfb2f1db09367ce"
readonly LOSSES_SHA256="1a08a7febc9bd94e1d389d2ef7f987908dd796a5f25cebccd433e155bbaa5aaf"
readonly AUXILIARY_SHA256="f1b90a6f40c8da20a362f91cccccf9a6b8a160b017cdb5be579115ea3bf30be5"
readonly MODEL_SHA256="3b9a2b88e9fa36c6f94b1595ae3812818ca5fdeb4b1db64c68f467912c04c9b1"
readonly SELECTOR_SHA256="61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"
readonly DATASET_SHA256="49af3c83091970ab307d0ca2e95a68ae42a6fcf52ddf72566b9da966ce6ad7a2"

MODE="${MODE:-preflight}"
readonly MODE
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
require_sha256 "${V97_SOURCE}" "${V97_SOURCE_SHA256}" "V99 lineage source"
require_sha256 "${GROUPFREE_CHECKPOINT}" "${GROUPFREE_SHA256}" "GroupFree checkpoint"
require_sha256 "${RESUME_CHECKPOINT}" "${RESUME_SHA256}" "E26 resume checkpoint"

"${PYTHON_BIN}" - "${RESUME_CHECKPOINT}" <<'PY'
import math
import sys
from collections import Counter

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
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
        raise SystemExit("checkpoint config mismatch: {}={!r}".format(key, config.get(key)))
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
if len(actual_scheduler_lr) != len(expected_current) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
        for actual, expected in zip(actual_scheduler_lr, expected_current)):
    raise SystemExit("scheduler current LR mismatch")
print("resume_provenance=E26_B12xA2_V99_full_state_verified")
PY

free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
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
if [[ "${MODE}" == "preflight" ]]; then
  exec 8>"${lock_file}"
  if ! flock -n 8; then
    echo "another V99 job owns ${lock_file}" >&2
    exit 6
  fi
  flock -u 8
  exec 8>&-
  echo "preflight=pass audit_only=true audit_batches=${AUDIT_BATCHES} target_hits025=12214"
  exit 0
fi
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "another V99 job owns ${lock_file}" >&2
  exit 6
fi

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pointnet2"

timestamp="$(date '+%Y%m%d_%H%M%S')"
readonly timestamp
AUDIT_ROOT="${OUTPUT_ROOT}/audit/${EXP}_${timestamp}"
readonly AUDIT_ROOT
mkdir -p "${AUDIT_ROOT}"
launch_log="${AUDIT_ROOT}/launch.log"
readonly launch_log
exec > >(tee -a "${launch_log}") 2>&1
echo "audit_only=true long_training_authorized=false"
echo "resume_checkpoint=${RESUME_CHECKPOINT}"
echo "resume_sha256=${RESUME_SHA256}"
echo "audit_root=${AUDIT_ROOT}"

train_args=(
  train_dist_mod.py
  --num_decoder_layers 6 --num_target 256
  --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}"
  --val_freq 1 --batch_size "${BATCH_SIZE}"
  --num_workers 4 --dataloader_prefetch_factor 2 --persistent_train_workers
  --save_freq 1 --print_freq 20
  --lr_backbone 1e-3 --lr 1e-4 --lr_decay_epochs 30 40 --warmup-epoch 0
  --dataset "${DATASET}" --test_dataset "${DATASET}"
  --joint_det --butd_cls
  --gradient_accumulation_steps "${ACCUMULATION}"
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
  --max_train_batches "${AUDIT_BATCHES}"
  --detect_intermediate --use_soft_token_loss --use_contrastive_align
  --log_dir "${AUDIT_ROOT}"
  --pp_checkpoint "${GROUPFREE_CHECKPOINT}"
  --self_attend --skip_missing_superpoints
  --checkpoint_path "${RESUME_CHECKPOINT}"
  --start_epoch 1 --max_epoch "${AUDIT_EPOCH}"
  --model MCLN --exp "${EXP}"
  --use_source_choice_selector --eval_use_selector_choice_scores
  --source_choice_selector_sources default,default_rank_blend_contrastive010
  --source_choice_selector_default_source default
  --source_choice_selector_hidden_dim 288
  --source_choice_selector_lr 1.25e-4
  --source_choice_selector_loss_weight 0.5
  --source_choice_selector_choice_target precision_gain_default_sourcewise_focal_bce
  --source_choice_selector_min_iou_gap 0.03
  --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}"
)
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node 1 --master_port "${MASTER_PORT}" \
  "${train_args[@]}"

mapfile -t receipts < <(
  find "${AUDIT_ROOT}" -type f \
    -name "train_audit_receipt_epoch_${AUDIT_EPOCH}.json" -print
)
if ((${#receipts[@]} != 1)); then
  echo "expected one bounded-audit receipt, found ${#receipts[@]}" >&2
  exit 8
fi
receipt="${receipts[0]}"
"${PYTHON_BIN}" - "${receipt}" "${RESUME_CHECKPOINT}" <<'PY'
import json
import math
import sys

receipt_path, checkpoint_path = sys.argv[1:]
with open(receipt_path, "r", encoding="utf-8") as handle:
    receipt = json.load(handle)
if receipt.get("schema") != "mcln-train-loss-epoch-v1":
    raise SystemExit("unexpected audit receipt schema")
if receipt.get("epoch") != 27 or receipt.get("max_train_batches") != 100:
    raise SystemExit("audit epoch/batch contract changed")
if receipt.get("batch_count") != 100:
    raise SystemExit("audit did not process exactly 100 batches")
if receipt.get("checkpoint_path") != checkpoint_path:
    raise SystemExit("audit resumed a different checkpoint")
for section in ("loss_means", "stat_means"):
    values = receipt.get(section)
    if not isinstance(values, dict) or not values:
        raise SystemExit("missing {}".format(section))
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(float(value)) for value in values.values()):
        raise SystemExit("non-finite value in {}".format(section))
stats = receipt["stat_means"]
required = (
    "grad_norm",
    "relation_counterfactual_aux_exact_gt_anchor_ratio",
    "relation_counterfactual_aux_relation_reference_valid_ratio",
    "relation_counterfactual_aux_hard_negative_row_ratio",
    "relation_counterfactual_aux_selected_negative_count_mean",
    "relation_counterfactual_aux_pair_violation_ratio",
)
missing = [name for name in required if name not in stats]
if missing:
    raise SystemExit("missing audit stats: {}".format(",".join(missing)))
gates = {
    "exact_gt_anchor_ratio": stats["relation_counterfactual_aux_exact_gt_anchor_ratio"] >= 0.50,
    "reference_valid_ratio": stats["relation_counterfactual_aux_relation_reference_valid_ratio"] >= 0.50,
    "hard_negative_row_ratio": stats["relation_counterfactual_aux_hard_negative_row_ratio"] >= 0.01,
    "selected_negative_count_mean": stats["relation_counterfactual_aux_selected_negative_count_mean"] >= 0.02,
    "pair_violation_ratio": stats["relation_counterfactual_aux_pair_violation_ratio"] >= 0.05,
    "positive_grad_norm": stats["grad_norm"] > 0.0,
}
print("density_gates={}".format(json.dumps(gates, sort_keys=True)))
print("density_values={}".format(json.dumps({
    "exact_gt_anchor_ratio": stats["relation_counterfactual_aux_exact_gt_anchor_ratio"],
    "reference_valid_ratio": stats["relation_counterfactual_aux_relation_reference_valid_ratio"],
    "hard_negative_row_ratio": stats["relation_counterfactual_aux_hard_negative_row_ratio"],
    "selected_negative_count_mean": stats["relation_counterfactual_aux_selected_negative_count_mean"],
    "pair_violation_ratio": stats["relation_counterfactual_aux_pair_violation_ratio"],
    "grad_norm": stats["grad_norm"],
}, sort_keys=True)))
if not all(gates.values()):
    raise SystemExit("density audit gates failed")
print("bounded_audit_receipt=approved_for_controlled_E27_E28")
PY
chmod 0444 "${receipt}"
echo "audit_receipt=${receipt}"
echo "audit_receipt_sha256=$(sha256sum "${receipt}" | awk '{print $1}')"
