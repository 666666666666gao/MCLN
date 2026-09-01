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
readonly EXP="nr3d_mcln_joint_butdcls_v99_relation_cf_conservative_anchor_density_v2_audit_e58_b100_b16x1_w4p2"
readonly BATCH_SIZE=16
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5303
readonly MIN_FREE_GB=7
readonly MIN_HARD_NEGATIVE_ROW_RATIO="0.01"
readonly MIN_SELECTED_NEGATIVE_COUNT_MEAN="0.0625"
readonly MIN_NONZERO_LOSS_BATCH_RATIO="0.10"
readonly MIN_VIOLATING_SELECTED_COUNT_MEAN="0.01"
readonly MIN_SELECTED_SCORE_GRADIENT_L1="0.05"
readonly REQUIRED_TRAIN_ENTRY_SHA256="3429c73b809b49e62b65720694cabb33fde4f465c8a2722fe93bdf12b220782d"
readonly REQUIRED_MAIN_UTILS_SHA256="555b122fa44ab91d73113094a5220f768e2b39130d389f69a5d26262c1cd7f21"
readonly REQUIRED_LOSSES_SHA256="cb0ba618ea5a126eb41503691a0c2853aceb3a803bd3fae557178b0e81a29816"
readonly REQUIRED_AUXILIARY_SHA256="5fbbc330e3acda85783b3d0ad7185cddcd197cdc4df07bf7ea7d7712d4a77496"
readonly REQUIRED_DATASET_SHA256="1f8a4e484da95797ce27824f2dbfb4dd680e838a86e0525df20be3b1dae97a03"
readonly REQUIRED_AUXILIARY_TEST_SHA256="f5e3a13c40780a789ea95c511da04acff62bc4d4c4eab95430c56b27265e9077"
readonly REQUIRED_MODEL_SHA256="3b9a2b88e9fa36c6f94b1595ae3812818ca5fdeb4b1db64c68f467912c04c9b1"
readonly REQUIRED_SELECTOR_SHA256="61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"

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
require_sha256 "${ROOT_DIR}/tests/test_relation_counterfactual_anchor_set.py" \
  "${REQUIRED_AUXILIARY_TEST_SHA256}" "relation auxiliary tests"
require_sha256 "${ROOT_DIR}/models/mcln.py" \
  "${REQUIRED_MODEL_SHA256}" "MCLN model"
require_sha256 "${ROOT_DIR}/models/source_choice_selector.py" \
  "${REQUIRED_SELECTOR_SHA256}" "source-choice selector"
require_sha256 "${GROUPFREE_CHECKPOINT}" \
  "${GROUPFREE_SHA256}" "GroupFree checkpoint"
require_sha256 "${V97_SOURCE}" "${V97_SOURCE_SHA256}" "V99 lineage source"
require_sha256 "${REQUIRED_RESUME_CHECKPOINT}" \
  "${REQUIRED_RESUME_SHA256}" "pinned official-best E57 checkpoint"

"${PYTHON_BIN}" - \
  "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" <<'PY'
import hashlib
import math
import sys

import torch

path, expected_sha256 = sys.argv[1:]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise SystemExit("resume checkpoint SHA changed before load")
    handle.seek(0)
    checkpoint = torch.load(handle, map_location="cpu")
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

AUDIT_ROOT="${OUTPUT_ROOT}/audit/${EXP}_one_shot"
readonly AUDIT_ROOT
if [[ -e "${AUDIT_ROOT}" ]]; then
  echo "density-v2 audit was already claimed: ${AUDIT_ROOT}" >&2
  exit 7
fi
mkdir -p "${AUDIT_ROOT}"
readonly RESUME_SNAPSHOT="${AUDIT_ROOT}/resume_e57.pth"
readonly GROUPFREE_SNAPSHOT="${AUDIT_ROOT}/gf_detector_l6o256.pth"
ln "${REQUIRED_RESUME_CHECKPOINT}" "${RESUME_SNAPSHOT}"
ln "${GROUPFREE_CHECKPOINT}" "${GROUPFREE_SNAPSHOT}"
launch_log="${AUDIT_ROOT}/launch.log"
readonly launch_log
exec > >(tee -a "${launch_log}") 2>&1
echo "audit_only=true long_training_authorized=false"
echo "audit_batches=${AUDIT_BATCHES} audit_accumulation=${AUDIT_ACCUMULATION}"
echo "resume_checkpoint=${REQUIRED_RESUME_CHECKPOINT}"
echo "resume_snapshot=${RESUME_SNAPSHOT}"
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
  --relation_counterfactual_aux_parent_top_k 128
  --relation_counterfactual_aux_target_tolerance 0.15
  --relation_counterfactual_aux_attribute_tolerance 0.15
  --relation_counterfactual_aux_geometry_threshold 0.04
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
  --pp_checkpoint "${GROUPFREE_SNAPSHOT}"
  --self_attend --skip_missing_superpoints
  --checkpoint_path "${RESUME_SNAPSHOT}"
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
full_command=(
  "${PYTHON_BIN}" -m torch.distributed.launch
  --nproc_per_node 1 --master_port "${MASTER_PORT}" \
  "${train_args[@]}"
)
printf '%q ' "${full_command[@]}" >"${AUDIT_ROOT}/train_command.txt"
printf '\n' >>"${AUDIT_ROOT}/train_command.txt"
chmod 0444 "${AUDIT_ROOT}/train_command.txt"
"${PYTHON_BIN}" - \
  "${AUDIT_ROOT}/pre_audit_provenance.json" \
  "${ROOT_DIR}" "${BASH_SOURCE[0]}" \
  "${AUDIT_ROOT}/train_command.txt" \
  "${RESUME_SNAPSHOT}" "${GROUPFREE_SNAPSHOT}" "${V97_SOURCE}" \
  "${REQUIRED_TRAIN_ENTRY_SHA256}" "${REQUIRED_MAIN_UTILS_SHA256}" \
  "${REQUIRED_LOSSES_SHA256}" "${REQUIRED_AUXILIARY_SHA256}" \
  "${REQUIRED_DATASET_SHA256}" "${REQUIRED_AUXILIARY_TEST_SHA256}" \
  "${REQUIRED_MODEL_SHA256}" "${REQUIRED_SELECTOR_SHA256}" \
  "${REQUIRED_RESUME_SHA256}" "${GROUPFREE_SHA256}" \
  "${V97_SOURCE_SHA256}" <<'PY'
import hashlib
import json
import os
import sys

(
    output_path,
    root_dir,
    launcher_path,
    command_path,
    resume_path,
    groupfree_path,
    v97_path,
    train_entry_sha,
    main_utils_sha,
    losses_sha,
    auxiliary_sha,
    dataset_sha,
    test_sha,
    model_sha,
    selector_sha,
    resume_sha,
    groupfree_sha,
    v97_sha,
) = sys.argv[1:]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_json_with_sha(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()

paths = {
    "launcher": os.path.realpath(launcher_path),
    "train_entry": os.path.join(root_dir, "train_dist_mod.py"),
    "main_utils": os.path.join(root_dir, "main_utils.py"),
    "losses": os.path.join(root_dir, "models", "losses.py"),
    "relation_auxiliary": os.path.join(
        root_dir, "models", "relation_counterfactual_auxiliary.py"
    ),
    "dataset": os.path.join(root_dir, "src", "joint_det_dataset.py"),
    "relation_auxiliary_tests": os.path.join(
        root_dir, "tests", "test_relation_counterfactual_anchor_set.py"
    ),
    "model": os.path.join(root_dir, "models", "mcln.py"),
    "source_choice_selector": os.path.join(
        root_dir, "models", "source_choice_selector.py"
    ),
    "resume_checkpoint": os.path.realpath(resume_path),
    "groupfree_checkpoint": os.path.realpath(groupfree_path),
    "v97_lineage": os.path.realpath(v97_path),
    "train_command": os.path.realpath(command_path),
}
expected = {
    "train_entry": train_entry_sha,
    "main_utils": main_utils_sha,
    "losses": losses_sha,
    "relation_auxiliary": auxiliary_sha,
    "dataset": dataset_sha,
    "relation_auxiliary_tests": test_sha,
    "model": model_sha,
    "source_choice_selector": selector_sha,
    "resume_checkpoint": resume_sha,
    "groupfree_checkpoint": groupfree_sha,
    "v97_lineage": v97_sha,
}
observed = {name: sha256_file(path) for name, path in paths.items()}
for name, expected_sha in expected.items():
    if observed[name] != expected_sha:
        raise SystemExit("{} drifted before audit".format(name))
payload = {
    "schema": "mcln-relation-counterfactual-pre-audit-v1",
    "paths": paths,
    "expected_sha256": expected,
    "observed_sha256": observed,
}
with open(output_path, "x") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
chmod 0444 "${AUDIT_ROOT}/pre_audit_provenance.json"
"${full_command[@]}"

mapfile -t receipts < <(
  find "${AUDIT_ROOT}" -type f \
    -name "train_audit_receipt_epoch_${AUDIT_EPOCH}.json" -print
)
if ((${#receipts[@]} != 1)); then
  echo "expected one bounded-audit receipt, found ${#receipts[@]}" >&2
  exit 8
fi
receipt="${receipts[0]}"
decision="${AUDIT_ROOT}/density_decision.json"
"${PYTHON_BIN}" - \
  "${receipt}" "${RESUME_SNAPSHOT}" "${decision}" \
  "${MIN_HARD_NEGATIVE_ROW_RATIO}" \
  "${MIN_SELECTED_NEGATIVE_COUNT_MEAN}" \
  "${MIN_NONZERO_LOSS_BATCH_RATIO}" \
  "${MIN_VIOLATING_SELECTED_COUNT_MEAN}" \
  "${MIN_SELECTED_SCORE_GRADIENT_L1}" \
  "${AUDIT_ROOT}/pre_audit_provenance.json" \
  "${BATCH_SIZE}" "${AUDIT_BATCHES}" <<'PY'
import hashlib
import json
import math
import os
import sys

(
    receipt_path,
    checkpoint_path,
    decision_path,
    min_hard_rows,
    min_selected,
    min_nonzero_batches,
    min_violating,
    min_score_gradient,
    pre_audit_path,
    batch_size,
    audit_batches,
) = sys.argv[1:]
batch_size = int(batch_size)
audit_batches = int(audit_batches)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_json_with_sha(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()

thresholds = {
    "relation_counterfactual_aux_hard_negative_row_ratio": float(
        min_hard_rows
    ),
    "relation_counterfactual_aux_selected_negative_count_mean": float(
        min_selected
    ),
    "relation_counterfactual_aux_nonzero_loss_batch_ratio": float(
        min_nonzero_batches
    ),
    "relation_counterfactual_aux_violating_selected_count_mean": float(
        min_violating
    ),
    "relation_counterfactual_aux_selected_score_gradient_l1": float(
        min_score_gradient
    ),
}
receipt, receipt_sha = load_json_with_sha(receipt_path)
pre_audit, pre_audit_sha = load_json_with_sha(pre_audit_path)
current_sha256 = {
    name: sha256_file(path)
    for name, path in pre_audit["paths"].items()
}
if current_sha256 != pre_audit.get("observed_sha256"):
    raise SystemExit("code or input artifact drifted during audit")
for name, expected_sha in pre_audit.get("expected_sha256", {}).items():
    if current_sha256.get(name) != expected_sha:
        raise SystemExit("{} no longer matches its fixed SHA".format(name))
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
    "relation_counterfactual_aux_nonzero_loss_batch_ratio",
    "relation_counterfactual_aux_violating_selected_count_mean",
    "relation_counterfactual_aux_selected_score_gradient_l1",
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
observed = {
    key: float(receipt["stat_means"][key])
    for key in thresholds
}
passed = all(observed[key] >= value for key, value in thresholds.items())
decision = {
    "schema": "mcln-relation-counterfactual-density-audit-v2",
    "audit_only": True,
    "long_training_authorized": False,
    "batch_count": audit_batches,
    "sample_count": batch_size * audit_batches,
    "density_gate_passed": passed,
    "thresholds": thresholds,
    "observed": observed,
    "receipt": os.path.realpath(receipt_path),
    "receipt_sha256": receipt_sha,
    "checkpoint": os.path.realpath(checkpoint_path),
    "checkpoint_sha256": current_sha256["resume_checkpoint"],
    "groupfree_checkpoint_sha256": current_sha256[
        "groupfree_checkpoint"
    ],
    "v97_lineage_sha256": current_sha256["v97_lineage"],
    "train_command": pre_audit["paths"]["train_command"],
    "train_command_sha256": current_sha256["train_command"],
    "pre_audit_provenance": os.path.realpath(pre_audit_path),
    "pre_audit_provenance_sha256": pre_audit_sha,
    "code_and_input_sha256": current_sha256,
    "configuration": {
        "parent_top_k": 128,
        "target_tolerance": 0.15,
        "attribute_tolerance": 0.15,
        "geometry_threshold": 0.04,
    },
}
temporary = decision_path + ".tmp.{}".format(os.getpid())
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(decision, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, decision_path)
print("bounded_audit_receipt=validated_finite_unapproved")
print("density_gate_passed={}".format(str(passed).lower()))
PY
chmod 0444 "${receipt}" "${decision}"
receipt_sha="$(sha256sum "${receipt}" | awk '{print $1}')"
decision_sha="$(sha256sum "${decision}" | awk '{print $1}')"
echo "audit_receipt=${receipt}"
echo "audit_receipt_sha256=${receipt_sha}"
echo "density_decision=${decision}"
echo "density_decision_sha256=${decision_sha}"
echo "approval_status=pending_independent_density_review"
