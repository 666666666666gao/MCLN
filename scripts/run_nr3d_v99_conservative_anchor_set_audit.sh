#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly GROUPFREE_CHECKPOINT="${DATA_ROOT%/}/gf_detector_l6o256.pth"
readonly GROUPFREE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly V97_SOURCE="${ROOT_DIR}/experiment_output/historical_e71_geometry/v97_contextual_listwise_hierarchical_trainonly_v1.json"
readonly V97_SOURCE_SHA256="ca04b4cbd1804b92d676d815b79bfcacdaab3e8745742177bd94283cedda7f8d"
readonly DATASET="nr3d"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/nr3d"
readonly REQUIRED_RESUME_CHECKPOINT="${OUTPUT_ROOT}/control/official_rec_monitor/official_best_rec025_epoch_57_0p56500823.pth"
readonly REQUIRED_RESUME_SHA256="fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
readonly REQUIRED_RESUME_EPOCH=57
readonly AUDIT_ACCUMULATION=1
readonly AUDIT_BATCHES=100
readonly AUDIT_EPOCH=58
readonly EXP="nr3d_mcln_joint_butdcls_v99_relation_cf_conservative_anchor_audit_e58_b100_b16x1_w4p2"
readonly BATCH_SIZE=16
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5299
readonly MIN_FREE_GB=7
readonly REQUIRED_TRAIN_ENTRY_SHA256="3429c73b809b49e62b65720694cabb33fde4f465c8a2722fe93bdf12b220782d"
readonly REQUIRED_MAIN_UTILS_SHA256="555b122fa44ab91d73113094a5220f768e2b39130d389f69a5d26262c1cd7f21"
readonly REQUIRED_LOSSES_SHA256="cb0ba618ea5a126eb41503691a0c2853aceb3a803bd3fae557178b0e81a29816"
readonly REQUIRED_AUXILIARY_SHA256="9f1eb4e07d058df63d3001a54b49a234db8f1226886d99c60d9f494fc21fafb7"
readonly REQUIRED_DATASET_SHA256="1f8a4e484da95797ce27824f2dbfb4dd680e838a86e0525df20be3b1dae97a03"

MODE="${MODE:-preflight}"
readonly MODE
case "${MODE}" in
  preflight|backbone) ;;
  *) echo "this audit-only wrapper supports MODE=preflight or MODE=backbone" >&2; exit 2 ;;
esac
if (($# != 0)); then
  echo "usage: MODE=preflight|backbone $0" >&2
  exit 2
fi
cd "${ROOT_DIR}"

require_sha256() {
  local path="$1" expected_sha="$2" label="$3" actual_sha
  [[ -f "${path}" ]] || {
    echo "missing ${label}: ${path}" >&2
    exit 3
  }
  actual_sha="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual_sha}" == "${expected_sha}" ]] || {
    echo "${label} SHA changed: ${actual_sha}" >&2
    exit 3
  }
}
require_sha256 "${ROOT_DIR}/train_dist_mod.py" \
  "${REQUIRED_TRAIN_ENTRY_SHA256}" "training entrypoint"
require_sha256 "${ROOT_DIR}/main_utils.py" \
  "${REQUIRED_MAIN_UTILS_SHA256}" "main_utils"
require_sha256 "${ROOT_DIR}/models/losses.py" \
  "${REQUIRED_LOSSES_SHA256}" "loss implementation"
require_sha256 "${ROOT_DIR}/models/relation_counterfactual_auxiliary.py" \
  "${REQUIRED_AUXILIARY_SHA256}" "relation auxiliary"
require_sha256 "${ROOT_DIR}/src/joint_det_dataset.py" \
  "${REQUIRED_DATASET_SHA256}" "dataset implementation"
require_sha256 "${GROUPFREE_CHECKPOINT}" \
  "${GROUPFREE_SHA256}" "GroupFree checkpoint"
require_sha256 "${V97_SOURCE}" "${V97_SOURCE_SHA256}" "V99 lineage source"
require_sha256 "${REQUIRED_RESUME_CHECKPOINT}" \
  "${REQUIRED_RESUME_SHA256}" "pinned official-best E57 checkpoint"

"${PYTHON_BIN}" - "${REQUIRED_RESUME_CHECKPOINT}" <<'PY'
import math
import sys

import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu")
config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else dict(config or {})
optimizer = checkpoint.get("optimizer", {})
param_groups = optimizer.get("param_groups", [])
expected_lrs = [1e-5, 1e-4, 1e-5, 1.25e-5]
actual_lrs = [group.get("lr") for group in param_groups]
if checkpoint.get("epoch") != 57:
    raise SystemExit("resume checkpoint is not completed E57")
if config.get("batch_size") != 16:
    raise SystemExit("resume checkpoint batch size is not 16")
if config.get("gradient_accumulation_steps", 1) != 1:
    raise SystemExit("resume checkpoint accumulation is not 1")
if config.get("joint_det") is not True or config.get("butd_cls") is not True:
    raise SystemExit("resume checkpoint is not joint_det + butd_cls")
lineage = config.get(
    "resume_lr_scale_lineage", config.get("resume_lr_scale", 1.0)
)
if lineage != 1.0:
    raise SystemExit("resume checkpoint LR lineage is not legacy 1.0")
if len(param_groups) != 4 or len(optimizer.get("state", {})) != 716:
    raise SystemExit("resume optimizer topology changed")
if len(actual_lrs) != len(expected_lrs) or any(
        value is None or not math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=1e-12
        )
        for value, expected in zip(actual_lrs, expected_lrs)):
    raise SystemExit("resume optimizer current LRs changed")
if config.get("use_source_choice_selector") is not True:
    raise SystemExit("resume checkpoint is not the V99 selector model")
print("resume_provenance=official_best_E57_B16x1_V99_verified")
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
  echo "preflight=pass audit_only=true audit_batches=${AUDIT_BATCHES}"
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
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pointnet2:${PYTHONPATH:-}"

timestamp="$(date '+%Y%m%d_%H%M%S')"
readonly timestamp
AUDIT_ROOT="${OUTPUT_ROOT}/audit/${EXP}_${timestamp}"
readonly AUDIT_ROOT
mkdir -p "${AUDIT_ROOT}"
launch_log="${AUDIT_ROOT}/launch.log"
readonly launch_log
exec > >(tee -a "${launch_log}") 2>&1
echo "audit_only=true long_training_authorized=false"
echo "audit_batches=${AUDIT_BATCHES} audit_accumulation=${AUDIT_ACCUMULATION}"
echo "resume_checkpoint=${REQUIRED_RESUME_CHECKPOINT}"
echo "resume_sha256=${REQUIRED_RESUME_SHA256}"
echo "audit_root=${AUDIT_ROOT}"

train_args=(
  train_dist_mod.py
  --num_decoder_layers 6 --num_target 256
  --use_color --weight_decay 0.0005
  --data_root "${DATA_ROOT}"
  --val_freq 1 --batch_size "${BATCH_SIZE}"
  --num_workers 4 --dataloader_prefetch_factor 2 --persistent_train_workers
  --save_freq 1 --print_freq 20
  --lr_backbone 1e-3 --lr 1e-4 --lr_decay_epochs 150 --warmup-epoch -1
  --dataset "${DATASET}" --test_dataset "${DATASET}"
  --joint_det --butd_cls
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
  --relation_counterfactual_aux_conservative_anchor_set
  --max_train_batches "${AUDIT_BATCHES}"
  --gradient_accumulation_steps "${AUDIT_ACCUMULATION}"
  --detect_intermediate --use_soft_token_loss --use_contrastive_align
  --log_dir "${AUDIT_ROOT}"
  --pp_checkpoint "${GROUPFREE_CHECKPOINT}"
  --self_attend --skip_missing_superpoints
  --checkpoint_path "${REQUIRED_RESUME_CHECKPOINT}"
  --resume_lr_scale 1.0
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
"${PYTHON_BIN}" - "${receipt}" "${REQUIRED_RESUME_CHECKPOINT}" <<'PY'
import json
import math
import sys

receipt_path, checkpoint_path = sys.argv[1:]
with open(receipt_path, "r", encoding="utf-8") as handle:
    receipt = json.load(handle)
if receipt.get("schema") != "mcln-train-loss-epoch-v1":
    raise SystemExit("unexpected bounded-audit receipt schema")
if receipt.get("epoch") != 58 or receipt.get("max_train_batches") != 100:
    raise SystemExit("bounded-audit epoch/batch contract changed")
if receipt.get("batch_count") != 100:
    raise SystemExit("bounded audit did not process exactly 100 batches")
if receipt.get("checkpoint_path") != checkpoint_path:
    raise SystemExit("bounded audit resumed a different checkpoint")
for section in ("loss_means", "stat_means"):
    values = receipt.get(section)
    if not isinstance(values, dict) or not values:
        raise SystemExit("missing {}".format(section))
    if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values.values()):
        raise SystemExit("non-finite value in {}".format(section))
required_stats = (
    "grad_norm",
    "relation_counterfactual_aux_anchor_reliable_ratio",
    "relation_counterfactual_aux_conservative_row_ratio",
    "relation_counterfactual_aux_relation_reference_valid_ratio",
    "relation_counterfactual_aux_hard_negative_row_ratio",
    "relation_counterfactual_aux_selected_negative_count_mean",
)
missing = [
    name for name in required_stats
    if name not in receipt["stat_means"]
]
if missing:
    raise SystemExit("missing audit stats: {}".format(",".join(missing)))
if receipt["stat_means"][
        "relation_counterfactual_aux_conservative_row_ratio"] <= 0.0:
    raise SystemExit("conservative anchor-set path was not exercised")
print("bounded_audit_receipt=validated_finite_unapproved")
PY
chmod 0444 "${receipt}"
receipt_sha="$(sha256sum "${receipt}" | awk '{print $1}')"
echo "audit_receipt=${receipt}"
echo "audit_receipt_sha256=${receipt_sha}"
echo "approval_status=pending_independent_density_review"
