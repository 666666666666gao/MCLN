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
readonly REQUIRED_RESUME_CHECKPOINT="${OUTPUT_ROOT}/backbone/sr3d_mcln_joint_butdcls_fromscratch_v99_e1_e140_b14x1_w4p2_20260822_131833/sr3d/sr3d_mcln_joint_butdcls_fromscratch_v99_e1_e140_b14x1_w4p2/1787377370/ckpt_epoch_last.pth"
readonly REQUIRED_RESUME_SHA256="fe5d987aebb57e4dfbed06891a9c9b5f2767107135bae372ef8f30ccf0c698db"
readonly REQUIRED_RESUME_EPOCH=34
readonly AUDIT_BATCHES=100
readonly AUDIT_RECEIPT="${OUTPUT_ROOT}/control/relation_counterfactual_aux/sr3d_e34_density_audit_approved.json"
RELATION_AUX_PHASE="${RELATION_AUX_PHASE:-audit}"
case "${RELATION_AUX_PHASE}" in
  audit)
    EXP="sr3d_mcln_joint_butdcls_v99_relation_cf_aux_audit_e35_b100_b14x1_w4p2"
    MAX_EPOCH=35
    ;;
  train)
    EXP="sr3d_mcln_joint_butdcls_v99_relation_cf_aux_e35_e46_b14x1_w4p2"
    MAX_EPOCH=46
    ;;
  *)
    echo "RELATION_AUX_PHASE must be audit or train" >&2
    exit 2
    ;;
esac
readonly RELATION_AUX_PHASE EXP MAX_EPOCH
readonly BATCH_SIZE=14
readonly EXPECTED_EVAL_SAMPLE_COUNT=17726
readonly MASTER_PORT=5399
readonly MIN_FREE_GB=7
readonly BACKBONE_JOINT_TRAINING=1
readonly INFERENCE_USES_GROUND_TRUTH=1
readonly USE_BACKBONE_INITIALIZATION=1
readonly TASK_CHECKPOINT_TRANSFER=0
readonly BACKBONE_AUGMENT_DET=0
CHECKPOINT_RETENTION_METRICS=(rec_acc025)
DATASET_LR_ARGS=(
  --lr_backbone 1e-3 --lr 1e-4
  --lr_decay_epochs 30 40
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
      echo "${variable_name} conflicts with the pinned E34 resume" >&2
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
    raise SystemExit("density-audit receipt does not match current code/E34")
PY
fi

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
