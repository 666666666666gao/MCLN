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
readonly EXP="nr3d_mcln_joint_butdcls_v99_tier_hard_query_audit_e58_b100_b16x1"
readonly BATCH_SIZE=16
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5329
readonly MIN_FREE_GB=7
readonly REQUIRED_TRAIN_ENTRY_SHA256="8f78cca50174423d0c4ab0b3c76a1fa6f22bbd1b179bd547013243ad199996f1"
readonly REQUIRED_MAIN_UTILS_SHA256="f677ada134f36bfd194a0694002fa3df37c1d2106d56cbebe59b608a1abbf065"
readonly REQUIRED_LOSSES_SHA256="48de298038ca9996e0c135dfc42ad5d271a0827d1a8c03309cc71d51a4e3082f"
readonly REQUIRED_AUXILIARY_SHA256="67f602ce84c5c5adce98553d65fe58b15f6dbd1dae0a52e430ec2beb29257c2b"
readonly REQUIRED_MODEL_SHA256="3b9a2b88e9fa36c6f94b1595ae3812818ca5fdeb4b1db64c68f467912c04c9b1"
readonly REQUIRED_SELECTOR_SHA256="61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"
readonly REQUIRED_DETECTOR_FILTER_SHA256="49a43b89a1ff129d09dcbdf0f6b61ff817aca50fb2c0edcb49072c60ded1a7e7"
readonly REQUIRED_SOURCE_MOE_SHA256="f09b2c5a5fb609a1b474baede83e21af0f034ec0fa9b050ced6613f66162fbd3"
readonly REQUIRED_AFFINITY_SHA256="39ecf930684e8936bce3472ea19cad2d59aab37dbb4b0b5b84d2fe842d12039c"
readonly REQUIRED_DATASET_SHA256="800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0"
readonly REQUIRED_TEST_SHA256="8e4a92124a8007b11b05a76500ad97d8f5493d56f580418fc8bba00ee72bd0ed"
readonly AUX_LOSS_WEIGHT="0.10"
readonly AUX_CANDIDATE_TOP_K=128
readonly AUX_MAX_NEGATIVES=8
readonly AUX_TARGET_TOLERANCE="0.15"
readonly AUX_TARGET_CONFIDENCE_FLOOR="0.01"
readonly AUX_PAIR_MARGIN="0.05"
readonly AUX_PRESERVE_WEIGHT="0.25"
readonly AUX_ACC025_PAIR_WEIGHT="2.0"
readonly MIN_REPAIR_ROW_RATIO="0.05"
readonly MIN_SUPERVISED_ROW_RATIO="0.70"
readonly MIN_SELECTED_NEGATIVES="1.0"
readonly MIN_PAIR_VIOLATION_RATIO="0.01"
readonly MIN_SELECTED_SCORE_GRADIENT_L1="0.01"

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
readonly AUDIT_PARENT="${OUTPUT_ROOT}/audit"
readonly AUDIT_ROOT="${AUDIT_PARENT}/${EXP}_once"
readonly LAUNCHER_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly LAUNCHER_SHA256_BEFORE="$(
  sha256sum "${LAUNCHER_PATH}" | awk '{print $1}'
)"

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
require_sha256 "${ROOT_DIR}/models/tier_hard_query_auxiliary.py" \
  "${REQUIRED_AUXILIARY_SHA256}" "tier hard-query auxiliary"
require_sha256 "${ROOT_DIR}/models/mcln.py" \
  "${REQUIRED_MODEL_SHA256}" "MCLN model"
require_sha256 "${ROOT_DIR}/models/source_choice_selector.py" \
  "${REQUIRED_SELECTOR_SHA256}" "V99 source-choice selector"
require_sha256 "${ROOT_DIR}/models/rec_evaluator_filter.py" \
  "${REQUIRED_DETECTOR_FILTER_SHA256}" "detector-overlap filter"
require_sha256 "${ROOT_DIR}/models/source_moe.py" \
  "${REQUIRED_SOURCE_MOE_SHA256}" "query box-IoU helper"
require_sha256 "${ROOT_DIR}/models/sacr_relation_counterfactual.py" \
  "${REQUIRED_AFFINITY_SHA256}" "target-affinity helper"
require_sha256 "${ROOT_DIR}/src/joint_det_dataset.py" \
  "${REQUIRED_DATASET_SHA256}" "dataset implementation"
require_sha256 "${ROOT_DIR}/tests/test_tier_hard_query_auxiliary.py" \
  "${REQUIRED_TEST_SHA256}" "tier hard-query regression test"
require_sha256 "${GROUPFREE_CHECKPOINT}" \
  "${GROUPFREE_SHA256}" "GroupFree checkpoint"
require_sha256 "${V97_SOURCE}" "${V97_SOURCE_SHA256}" "V99 lineage source"

"${PYTHON_BIN}" - \
  "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" <<'PY'
import hashlib
import math
import sys

import torch

path, expected_sha256 = sys.argv[1:]


def sha256_open_file(handle):
    digest = hashlib.sha256()
    handle.seek(0)
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


with open(path, "rb") as handle:
    before_sha256 = sha256_open_file(handle)
    if before_sha256 != expected_sha256:
        raise SystemExit("resume checkpoint SHA changed")
    handle.seek(0)
    checkpoint = torch.load(handle, map_location="cpu")
    after_sha256 = sha256_open_file(handle)
if after_sha256 != before_sha256:
    raise SystemExit("resume checkpoint changed while it was loaded")
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

if [[ -e "${AUDIT_ROOT}" ]]; then
  echo "one-shot audit root already exists: ${AUDIT_ROOT}" >&2
  exit 7
fi

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
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pointnet2"

mkdir -p "${AUDIT_PARENT}"
if ! mkdir "${AUDIT_ROOT}"; then
  echo "one-shot audit root already exists: ${AUDIT_ROOT}" >&2
  exit 7
fi
launch_log="${AUDIT_ROOT}/launch.log"
readonly launch_log
exec > >(tee -a "${launch_log}") 2>&1
echo "audit_only=true long_training_authorized=false"
echo "audit_batches=${AUDIT_BATCHES} audit_accumulation=${AUDIT_ACCUMULATION}"
echo "aux_loss_weight=${AUX_LOSS_WEIGHT}"
echo "launcher_sha256=${LAUNCHER_SHA256_BEFORE}"
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
  --tier_hard_query_aux_loss_weight "${AUX_LOSS_WEIGHT}"
  --tier_hard_query_aux_candidate_top_k "${AUX_CANDIDATE_TOP_K}"
  --tier_hard_query_aux_max_negatives "${AUX_MAX_NEGATIVES}"
  --tier_hard_query_aux_target_tolerance "${AUX_TARGET_TOLERANCE}"
  --tier_hard_query_aux_target_confidence_floor "${AUX_TARGET_CONFIDENCE_FLOOR}"
  --tier_hard_query_aux_pair_margin "${AUX_PAIR_MARGIN}"
  --tier_hard_query_aux_preserve_weight "${AUX_PRESERVE_WEIGHT}"
  --tier_hard_query_aux_acc025_pair_weight "${AUX_ACC025_PAIR_WEIGHT}"
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
provenance_path="${AUDIT_ROOT}/audit_provenance.json"
readonly provenance_path
provenance_files=(
  launcher "${LAUNCHER_PATH}" "${LAUNCHER_SHA256_BEFORE}"
  training_entry "${ROOT_DIR}/train_dist_mod.py" "${REQUIRED_TRAIN_ENTRY_SHA256}"
  main_utils "${ROOT_DIR}/main_utils.py" "${REQUIRED_MAIN_UTILS_SHA256}"
  losses "${ROOT_DIR}/models/losses.py" "${REQUIRED_LOSSES_SHA256}"
  tier_auxiliary "${ROOT_DIR}/models/tier_hard_query_auxiliary.py" "${REQUIRED_AUXILIARY_SHA256}"
  mcln "${ROOT_DIR}/models/mcln.py" "${REQUIRED_MODEL_SHA256}"
  selector "${ROOT_DIR}/models/source_choice_selector.py" "${REQUIRED_SELECTOR_SHA256}"
  detector_filter "${ROOT_DIR}/models/rec_evaluator_filter.py" "${REQUIRED_DETECTOR_FILTER_SHA256}"
  source_moe "${ROOT_DIR}/models/source_moe.py" "${REQUIRED_SOURCE_MOE_SHA256}"
  target_affinity "${ROOT_DIR}/models/sacr_relation_counterfactual.py" "${REQUIRED_AFFINITY_SHA256}"
  dataset "${ROOT_DIR}/src/joint_det_dataset.py" "${REQUIRED_DATASET_SHA256}"
  regression_test "${ROOT_DIR}/tests/test_tier_hard_query_auxiliary.py" "${REQUIRED_TEST_SHA256}"
  groupfree "${GROUPFREE_CHECKPOINT}" "${GROUPFREE_SHA256}"
  v99_lineage "${V97_SOURCE}" "${V97_SOURCE_SHA256}"
  resume_e57 "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}"
)
"${PYTHON_BIN}" - \
  "${provenance_path}" "${EXP}" "${AUDIT_ROOT}" \
  --files "${provenance_files[@]}" \
  --command "${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node 1 --master_port "${MASTER_PORT}" \
  "${train_args[@]}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


arguments = sys.argv[1:]
output_path, experiment, audit_root = arguments[:3]
files_index = arguments.index("--files")
command_index = arguments.index("--command")
file_arguments = arguments[files_index + 1:command_index]
command = arguments[command_index + 1:]
if len(file_arguments) % 3 != 0:
    raise SystemExit("provenance file arguments are not triplets")
files = {}
for offset in range(0, len(file_arguments), 3):
    label, path, expected_sha256 = file_arguments[offset:offset + 3]
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise SystemExit("{} SHA changed before audit".format(label))
    files[label] = {
        "path": os.path.realpath(path),
        "sha256": actual_sha256,
    }
payload = {
    "schema": "mcln-tier-hard-query-audit-provenance-v1",
    "audit_only": True,
    "long_training_authorized": False,
    "experiment": experiment,
    "audit_root": os.path.realpath(audit_root),
    "command": command,
    "environment": {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE",
            "PYTHONPATH",
        )
    },
    "files": files,
}
parent = os.path.dirname(output_path)
descriptor, temporary_path = tempfile.mkstemp(
    prefix=".audit_provenance.", suffix=".tmp", dir=parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, output_path)
    os.chmod(output_path, 0o444)
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if os.path.exists(temporary_path):
        os.unlink(temporary_path)
print("audit_provenance={}".format(output_path))
PY
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
require_sha256 "${ROOT_DIR}/train_dist_mod.py" \
  "${REQUIRED_TRAIN_ENTRY_SHA256}" "post-audit training entrypoint"
require_sha256 "${ROOT_DIR}/main_utils.py" \
  "${REQUIRED_MAIN_UTILS_SHA256}" "post-audit main_utils"
require_sha256 "${ROOT_DIR}/models/losses.py" \
  "${REQUIRED_LOSSES_SHA256}" "post-audit loss implementation"
require_sha256 "${ROOT_DIR}/models/tier_hard_query_auxiliary.py" \
  "${REQUIRED_AUXILIARY_SHA256}" "post-audit tier auxiliary"
require_sha256 "${ROOT_DIR}/models/mcln.py" \
  "${REQUIRED_MODEL_SHA256}" "post-audit MCLN model"
require_sha256 "${ROOT_DIR}/models/source_choice_selector.py" \
  "${REQUIRED_SELECTOR_SHA256}" "post-audit V99 selector"
require_sha256 "${ROOT_DIR}/models/rec_evaluator_filter.py" \
  "${REQUIRED_DETECTOR_FILTER_SHA256}" "post-audit detector filter"
require_sha256 "${ROOT_DIR}/models/source_moe.py" \
  "${REQUIRED_SOURCE_MOE_SHA256}" "post-audit query box-IoU helper"
require_sha256 "${ROOT_DIR}/models/sacr_relation_counterfactual.py" \
  "${REQUIRED_AFFINITY_SHA256}" "post-audit target-affinity helper"
require_sha256 "${ROOT_DIR}/src/joint_det_dataset.py" \
  "${REQUIRED_DATASET_SHA256}" "post-audit dataset implementation"
require_sha256 "${ROOT_DIR}/tests/test_tier_hard_query_auxiliary.py" \
  "${REQUIRED_TEST_SHA256}" "post-audit regression test"
require_sha256 "${GROUPFREE_CHECKPOINT}" \
  "${GROUPFREE_SHA256}" "post-audit GroupFree checkpoint"
require_sha256 "${V97_SOURCE}" \
  "${V97_SOURCE_SHA256}" "post-audit V99 lineage source"
require_sha256 "${REQUIRED_RESUME_CHECKPOINT}" \
  "${REQUIRED_RESUME_SHA256}" "post-audit official-best E57 checkpoint"
launcher_sha256_after="$(sha256sum "${LAUNCHER_PATH}" | awk '{print $1}')"
if [[ "${launcher_sha256_after}" != "${LAUNCHER_SHA256_BEFORE}" ]]; then
  echo "launcher changed during audit" >&2
  exit 3
fi
if find "${AUDIT_ROOT}" -type f -name '*.pth' -print -quit | grep -q .; then
  echo "bounded audit unexpectedly wrote a checkpoint" >&2
  exit 9
fi
decision_path="${AUDIT_ROOT}/audit_decision.json"
readonly decision_path
"${PYTHON_BIN}" - \
  "${receipt}" "${REQUIRED_RESUME_CHECKPOINT}" \
  "${provenance_path}" "${decision_path}" \
  "${MIN_REPAIR_ROW_RATIO}" "${MIN_SUPERVISED_ROW_RATIO}" \
  "${MIN_SELECTED_NEGATIVES}" "${MIN_PAIR_VIOLATION_RATIO}" \
  "${MIN_SELECTED_SCORE_GRADIENT_L1}" <<'PY'
import hashlib
import json
import math
import os
import sys
import tempfile

(
    receipt_path,
    checkpoint_path,
    provenance_path,
    decision_path,
    min_repair_ratio,
    min_supervised_ratio,
    min_selected_negatives,
    min_pair_violation_ratio,
    min_gradient_l1,
) = sys.argv[1:]
thresholds = {
    "repair_row_ratio": float(min_repair_ratio),
    "supervised_row_ratio": float(min_supervised_ratio),
    "selected_negative_count_mean": float(min_selected_negatives),
    "pair_violation_ratio": float(min_pair_violation_ratio),
    "selected_score_gradient_l1": float(min_gradient_l1),
}
with open(receipt_path, "rb") as handle:
    receipt_bytes = handle.read()
receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
receipt = json.loads(receipt_bytes.decode("utf-8"))
with open(provenance_path, "rb") as handle:
    provenance_bytes = handle.read()
provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
provenance = json.loads(provenance_bytes.decode("utf-8"))
if provenance.get("schema") != "mcln-tier-hard-query-audit-provenance-v1":
    raise SystemExit("unexpected audit provenance schema")
if provenance.get("long_training_authorized") is not False:
    raise SystemExit("audit provenance incorrectly authorizes training")
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
required_losses = ("loss", "tier_hard_query_aux_loss")
missing_losses = [
    name for name in required_losses
    if name not in receipt["loss_means"]
]
if missing_losses:
    raise SystemExit(
        "missing audit losses: {}".format(",".join(missing_losses))
    )
required_stats = (
    "grad_norm",
    "tier_hard_query_aux_sample_row_ratio",
    "tier_hard_query_aux_active_row_ratio",
    "tier_hard_query_aux_detector_valid_row_ratio",
    "tier_hard_query_aux_repair_row_ratio",
    "tier_hard_query_aux_preserve_row_ratio",
    "tier_hard_query_aux_same_tier_row_ratio",
    "tier_hard_query_aux_empty_row_ratio",
    "tier_hard_query_aux_selected_negative_count_mean",
    "tier_hard_query_aux_pair_violation_ratio",
    "tier_hard_query_aux_coarse_break_selected_ratio",
    "tier_hard_query_aux_selected_score_gradient_l1",
    "tier_hard_query_aux_parent_acc025",
    "tier_hard_query_aux_parent_acc050",
    "tier_hard_query_aux_teacher_oracle_acc025",
    "tier_hard_query_aux_teacher_oracle_acc050",
    "tier_hard_query_aux_teacher_oracle_conditional_acc025",
    "tier_hard_query_aux_teacher_oracle_conditional_acc050",
)
missing_stats = [
    name for name in required_stats
    if name not in receipt["stat_means"]
]
if missing_stats:
    raise SystemExit(
        "missing audit stats: {}".format(",".join(missing_stats))
    )
stats = receipt["stat_means"]
prefix = "tier_hard_query_aux_"
ratio_names = (
    "sample_row_ratio",
    "active_row_ratio",
    "detector_valid_row_ratio",
    "repair_row_ratio",
    "preserve_row_ratio",
    "same_tier_row_ratio",
    "empty_row_ratio",
    "pair_violation_ratio",
    "coarse_break_selected_ratio",
    "parent_acc025",
    "parent_acc050",
    "teacher_oracle_acc025",
    "teacher_oracle_acc050",
    "teacher_oracle_conditional_acc025",
    "teacher_oracle_conditional_acc050",
)
if any(
        stats[prefix + name] < -1e-8
        or stats[prefix + name] > 1.0 + 1e-8
        for name in ratio_names):
    raise SystemExit("tier ratio statistic is outside [0, 1]")
categories = sum(
    stats[prefix + name]
    for name in (
        "repair_row_ratio",
        "preserve_row_ratio",
        "same_tier_row_ratio",
        "empty_row_ratio",
    )
)
if not math.isclose(categories, 1.0, rel_tol=0.0, abs_tol=1e-5):
    raise SystemExit("tier row categories are not exhaustive")
if stats[prefix + "sample_row_ratio"] <= 0.0:
    raise SystemExit("bounded audit contains no grounding sample rows")
if stats[prefix + "detector_valid_row_ratio"] > (
        stats[prefix + "active_row_ratio"] + 1e-8):
    raise SystemExit("detector-valid density exceeds active density")
if stats[prefix + "parent_acc025"] > (
        stats[prefix + "active_row_ratio"] + 1e-8):
    raise SystemExit("parent accuracy exceeds active density")
for coarse, strict in (
        ("parent_acc025", "parent_acc050"),
        ("teacher_oracle_acc025", "teacher_oracle_acc050"),
        ("teacher_oracle_conditional_acc025",
         "teacher_oracle_conditional_acc050")):
    if stats[prefix + strict] > stats[prefix + coarse] + 1e-8:
        raise SystemExit("strict-IoU audit metric exceeds coarse metric")
if stats[prefix + "teacher_oracle_acc025"] > (
        stats[prefix + "detector_valid_row_ratio"] + 1e-8):
    raise SystemExit("teacher oracle exceeds detector-valid density")
selected_negative_count = stats[
    prefix + "selected_negative_count_mean"
]
if selected_negative_count < 0.0 or selected_negative_count > 8.0 + 1e-8:
    raise SystemExit("selected negative count is outside [0, 8]")
supervised_ratio = (
    stats[prefix + "repair_row_ratio"]
    + stats[prefix + "preserve_row_ratio"]
)
density_values = {
    "repair_row_ratio": stats[prefix + "repair_row_ratio"],
    "supervised_row_ratio": supervised_ratio,
    "selected_negative_count_mean": selected_negative_count,
    "pair_violation_ratio": stats[prefix + "pair_violation_ratio"],
    "selected_score_gradient_l1": stats[
        prefix + "selected_score_gradient_l1"
    ],
}
density_checks = {
    name: density_values[name] >= threshold
    for name, threshold in thresholds.items()
}
decision = {
    "schema": "mcln-tier-hard-query-audit-decision-v1",
    "audit_only": True,
    "integrity_passed": True,
    "bounded_receipt_validated": True,
    "post_provenance_verified": True,
    "no_checkpoint_written": True,
    "density_gate_passed": all(density_checks.values()),
    "density_checks": density_checks,
    "density_values": density_values,
    "density_thresholds": thresholds,
    "long_training_authorized": False,
    "approval_status": "pending_independent_density_review",
    "receipt": {
        "path": os.path.realpath(receipt_path),
        "sha256": receipt_sha256,
    },
    "provenance": {
        "path": os.path.realpath(provenance_path),
        "sha256": provenance_sha256,
    },
    "loss_means": {
        name: receipt["loss_means"][name]
        for name in required_losses
    },
    "stat_means": {
        name: stats[name]
        for name in required_stats
    },
    "terminal_exit_code": 20,
}
parent = os.path.dirname(decision_path)
descriptor, temporary_path = tempfile.mkstemp(
    prefix=".audit_decision.", suffix=".tmp", dir=parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, decision_path)
    os.chmod(decision_path, 0o444)
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if os.path.exists(temporary_path):
        os.unlink(temporary_path)
print("bounded_audit_receipt=validated_finite_unapproved")
print("density_gate_passed={}".format(all(density_checks.values())))
for name in sorted(density_checks):
    print("density_check_{}={}".format(name, density_checks[name]))
for name in required_stats:
    print("{}={:.10f}".format(name, float(stats[name])))
print("audit_receipt_sha256={}".format(receipt_sha256))
print("audit_provenance_sha256={}".format(provenance_sha256))
print("audit_decision={}".format(decision_path))
PY
chmod 0444 "${receipt}"
echo "audit_receipt=${receipt}"
echo "approval_status=pending_independent_density_review"
echo "long_training_authorized=false"
echo "terminal_exit_code=20"
chmod 0444 "${launch_log}"
exit 20
