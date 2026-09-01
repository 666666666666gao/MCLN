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
readonly REQUIRED_RESUME_CHECKPOINT="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_fromscratch_v99_e1_e240_b16x1_w4p2_20260822_131833/nr3d/nr3d_mcln_joint_butdcls_fromscratch_v99_e1_e240_b16x1_w4p2/1787376578/ckpt_epoch_62.pth"
readonly REQUIRED_RESUME_SHA256="93d0539c7fd8ffa19d167e788ba740a990293ff492a998b15be17700ac44a04d"
readonly REQUIRED_RESUME_EPOCH=62
readonly AUDIT_BATCHES=100
readonly AUDIT_RECEIPT="${OUTPUT_ROOT}/control/relation_counterfactual_aux/nr3d_e62_density_audit_approved.json"
RELATION_AUX_PHASE="${RELATION_AUX_PHASE:-audit}"
case "${RELATION_AUX_PHASE}" in
  audit)
    EXP="nr3d_mcln_joint_butdcls_v99_relation_cf_aux_audit_e63_b100_b16x1_w4p2"
    MAX_EPOCH=63
    ;;
  train)
    EXP="nr3d_mcln_joint_butdcls_v99_relation_cf_aux_e63_e74_b16x1_w4p2"
    MAX_EPOCH=74
    ;;
  *)
    echo "RELATION_AUX_PHASE must be audit or train" >&2
    exit 2
    ;;
esac
readonly RELATION_AUX_PHASE EXP MAX_EPOCH
readonly BATCH_SIZE=16
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
)
BACKBONE_EXTRA_ARGS=(
  --print_freq 20
  --joint_det
  --butd_cls
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

if [[ "${RELATION_AUX_PHASE}" == "audit" ]]; then
  BACKBONE_EXTRA_ARGS+=(--max_train_batches "${AUDIT_BATCHES}")
fi

for variable_name in \
    BACKBONE_RESUME_CHECKPOINT BACKBONE_RESUME_SHA256 BACKBONE_RESUME_EPOCH; do
  if [[ -n "${!variable_name:-}" ]]; then
    case "${variable_name}" in
      BACKBONE_RESUME_CHECKPOINT) required_value="${REQUIRED_RESUME_CHECKPOINT}" ;;
      BACKBONE_RESUME_SHA256) required_value="${REQUIRED_RESUME_SHA256}" ;;
      BACKBONE_RESUME_EPOCH) required_value="${REQUIRED_RESUME_EPOCH}" ;;
    esac
    if [[ "${!variable_name}" != "${required_value}" ]]; then
      echo "${variable_name} conflicts with the pinned E62 resume" >&2
      exit 2
    fi
  fi
done
export BACKBONE_RESUME_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT}"
export BACKBONE_RESUME_SHA256="${REQUIRED_RESUME_SHA256}"
export BACKBONE_RESUME_EPOCH="${REQUIRED_RESUME_EPOCH}"
export VALIDATE_BACKBONE_RESUME=0
export MODE="${MODE:-backbone}"
case "${MODE}" in
  preflight|backbone) ;;
  *) echo "this wrapper supports MODE=preflight or MODE=backbone only" >&2; exit 2 ;;
esac

if [[ "${RELATION_AUX_PHASE}" == "train" ]]; then
  [[ -f "${AUDIT_RECEIPT}" ]] || {
    echo "missing approved density-audit receipt: ${AUDIT_RECEIPT}" >&2
    exit 3
  }
  "${PYTHON_BIN}" - "${AUDIT_RECEIPT}" \
      "${REQUIRED_RESUME_SHA256}" "${ROOT_DIR}" <<'PY'
import hashlib
import json
import os
import sys

receipt_path, resume_sha, root = sys.argv[1:]
with open(receipt_path, "r", encoding="utf-8") as handle:
    receipt = json.load(handle)
expected_files = (
    "models/relation_counterfactual_auxiliary.py",
    "models/losses.py",
    "main_utils.py",
    "src/joint_det_dataset.py",
)
actual_hashes = {}
for relative in expected_files:
    with open(os.path.join(root, relative), "rb") as handle:
        actual_hashes[relative] = hashlib.sha256(handle.read()).hexdigest()
if (
        receipt.get("status") != "approved"
        or receipt.get("resume_sha256") != resume_sha
        or receipt.get("audit_batches") != 100
        or receipt.get("code_sha256") != actual_hashes):
    raise SystemExit("density-audit receipt does not match current code/E62")
PY
fi

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
